import dataclasses
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from larry.common.config.model.model_configs import LarryRecurrentTrunkConfig, LLMBlockModuleConfig
from larry.common.metrics.metrics import MetricDict
from larry.common.model import recurrent_criteria
from larry.common.model.attention import LarryAttentionOutput
from larry.common.model.kv_cache import RecurrentKVCache
from larry.common.model.llm import LarrySimpleEncoderBlock
from larry.common.model.norms import AdaLayerNorm, get_norm_type
from larry.common.training.training_model import LarryModel


@dataclasses.dataclass(kw_only=True)
class LarryRecurrentTrunkOutput(LarryAttentionOutput[RecurrentKVCache | None]):
    n_iterations: int
    kv_cache: RecurrentKVCache | None = None


class LarryRecurrentTrunk(LarryModel[LarryRecurrentTrunkConfig]):
    def __init__(self, config: LarryRecurrentTrunkConfig) -> None:
        super().__init__(config)

        block_config = LLMBlockModuleConfig.from_child(config)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(block_config)
            for _ in range(config.n_layers)
        ])

        self.iteration_projection = None
        self.pre_iteration_projection_norm = None
        self.post_iteration_projection_norm = None
        if config.thought_state_injection_type == "concat":
            self.iteration_projection = nn.Linear(config.d_model * 2, config.d_model)
            with torch.no_grad():
                identity = torch.eye(config.d_model)
                self.iteration_projection.weight.copy_(torch.cat([identity, identity], dim=1))
                self.iteration_projection.bias.zero_()

            if config.iteration_projection_norm_placement:
                iteration_projection_norm_type = get_norm_type(config.iteration_projection_norm_type)
                if iteration_projection_norm_type == AdaLayerNorm:
                    # TODO: this might be interesting to support
                    raise ValueError(f"{type(self).__name__} doesn't support {iteration_projection_norm_type.__name__}")

                if config.iteration_projection_norm_placement == 'pre':
                    self.pre_iteration_projection_norm = iteration_projection_norm_type(config.d_model * 2)
                elif config.iteration_projection_norm_placement == 'post':
                    self.post_iteration_projection_norm = iteration_projection_norm_type(config.d_model)

        if config.exit_criteria == 'logit-kl':
            self.exit_criteria = recurrent_criteria.LogitKLCriteria(config.logits_kl_exit_criteria_threshold)
        elif config.exit_criteria == 'latent-diff':
            self.exit_criteria = recurrent_criteria.LatentDiffCriteria(config.latent_diff_exit_criteria_threshold)
        elif config.exit_criteria in ('none', None):
            self.exit_criteria = recurrent_criteria.NoOpCriteria()

        self.step = 0

    def initialize_thinking_state(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Taken directly from the original Huginn implementation.

        See https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py
        """
        if self.config.thought_state_init == "none":
            x = input_embeds
        elif self.config.thought_state_init == "normal":
            x = torch.randn_like(input_embeds)
        elif self.config.thought_state_init == "embed":
            x = torch.randn_like(input_embeds).mul(1 / math.sqrt(input_embeds.shape[-1]))
        elif self.config.thought_state_init == "like-init":
            x = torch.randn_like(input_embeds)
            std = self.config.thought_init_std
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif self.config.thought_state_init == "zero":
            x = torch.zeros_like(input_embeds)
        elif self.config.thought_state_init == "unit":
            x = torch.randn_like(input_embeds)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        else:
            raise ValueError(f"Unknown config.thought_state_init={self.config.thought_state_init}")

        return x

    def _run_iteration(
            self,
            x_0: torch.Tensor,
            thought: torch.Tensor,
            iteration: int,
            attention_mask: torch.Tensor | None,
            kv_cache: RecurrentKVCache | None,
            position_offset: int,
            use_cache: bool,
            _extend_mask_fn=None,
            position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MetricDict | None]:
        h = self._inject(x_0, thought)

        metric_dict: MetricDict | None = None
        for block_idx, block in enumerate(self.blocks):
            iter_cache = None
            if use_cache and kv_cache is not None:
                iter_cache = kv_cache.read(iteration, block_idx, position_offset)

            if _extend_mask_fn:
                mask = _extend_mask_fn(attention_mask, iter_cache)
            else:
                mask = attention_mask

            if self.config.gradient_checkpointing and self.training and not use_cache:
                block_output = checkpoint(
                    block,
                    h,
                    mask,
                    iter_cache,
                    position_ids,
                    position_offset,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                block_output = block(
                    h,
                    attention_mask=mask,
                    kv_cache=iter_cache,
                    position_ids=position_ids,
                    position_offset=position_offset,
                    use_cache=use_cache,
                )

            h = block_output.logits
            if metric_dict:
                metric_dict = metric_dict + block_output.metrics
            else:
                metric_dict = block_output.metrics

            if use_cache and block_output.kv_cache is not None and kv_cache is not None:
                kv_cache.write(iteration, block_idx, block_output.kv_cache)

        return h, metric_dict

    def _inject(self, x_0: torch.Tensor, thought: torch.Tensor) -> torch.Tensor:
        if self.config.thought_state_injection_type == "sum":
            return x_0 + thought

        h = torch.cat([x_0, thought], dim=-1)
        if self.pre_iteration_projection_norm is not None:
            h = self.pre_iteration_projection_norm(h)

        if self.iteration_projection is not None:
            h = self.iteration_projection(h)

        if self.post_iteration_projection_norm is not None:
            h = self.post_iteration_projection_norm(h)

        return h

    def _n_k_steps(self, mean_steps: int, backprop_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed_n = 514229 + self.step  # easiest way to make the sampler re-runnable in checkpointing
        seed_k = 317811 + self.step
        if not self.config.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.config.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2 ** 31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2 ** 31 - 1))

        t = max(mean_steps - backprop_depth, 0)
        s = backprop_depth

        if self.training:
            # poisson log normal filling
            sigma = 0.5
            mu = math.log(t + s) - (sigma ** 2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            self.step += 1
        else:
            # during evaluation, we use the mean number of steps as a maximum and let exit criteria determine when to
            # stop if it can do so before then. no backprop necessary
            n, k = torch.tensor(mean_steps), torch.tensor(0)

        return n.to(torch.long), k.to(torch.long)

    def forward(
            self,
            x_0: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            kv_cache: RecurrentKVCache | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
            max_iterations_override: int | None = None,
            position_ids: torch.Tensor | None = None,
            readout: Callable[[torch.Tensor], torch.Tensor] | dict[str, Callable] | None = None,
            converge_eligible: torch.Tensor | dict[str, torch.Tensor] | None = None,
            m_rope_dim_mask: torch.Tensor | None = None,
    ) -> LarryRecurrentTrunkOutput:
        """The core of any generative model working with text as either input, output, or both in the larry project.

        Args:
            :param x_0: input; injected at every iteration along with the previous iteration as the "thought state".
            :param attention_mask: to pass to the blocks.
            :param kv_cache: to pass to the blocks.
            :param position_offset: to pass to the blocks.
            :param use_cache: whether to use a provided (or generate a new kv cache); ignored during training.
            :param max_iterations_override: to force a maximum number of iterations for evaluation; ignored during
                training.
            :param position_ids: to pass to the blocks.
            :param readout: Optional callable or dict of callables to produce logits from the thought state for exit
                criteria evaluation. This is for the exit criteria Huginn uses, logit-kl divergence which requires full
                evaluation of the coda(s) and for their logits to be softmax-coherent. If a dict is provided, the keys
                are used to match against the converge_eligible dict to determine which tokens are eligible for
                convergence evaluation for each readout function. If it is a single callable, it is used for all tokens.
            :param converge_eligible: a mask of which tokens are eligible for convergence; masked tokens have converged.
            :param m_rope_dim_mask: a mask of integer values signifying what number of slices of dims m-rope should act
                on; 1 = global rope only, 2 = global rope (half) and local rope (other half), 3 = global rope (half) and
                row rope (one quarter of the remaining half) and column rope (the remaining quarter) - both local only
        """
        if self.training:
            use_cache = False
        elif readout is None and isinstance(self.exit_criteria, recurrent_criteria.LogitKLCriteria):
            raise ValueError("exit_criteria='logit-kl' needs a readout to produce logits outside of training")

        effective_mean_steps = self.config.mean_thinking_steps
        if max_iterations_override is not None and not self.training:
            effective_mean_steps = max_iterations_override

        n_steps_no_grad, k_steps_grad = self._n_k_steps(effective_mean_steps, self.config.backprop_depth)

        last_thought_state = self.initialize_thinking_state(x_0)  # (batch_size, seq_len, d_model)

        converged: torch.Tensor | None = None
        if not self.training and self.exit_criteria is not None:
            converged = torch.zeros(x_0.shape[0], x_0.shape[1], dtype=torch.bool, device=x_0.device)

        readout_map: dict[str, Callable] = {}
        eligible_map: dict[str, torch.Tensor | None] = {}
        if readout is not None:
            readout_map = readout if isinstance(readout, dict) else {"_default": readout}
            if isinstance(converge_eligible, dict):
                eligible_map = {
                    k: converge_eligible.get(k)
                    for k in readout_map
                }
            else:
                eligible_map = {
                    k: converge_eligible
                    for k in readout_map
                }

        eligible_any: torch.Tensor | None = None
        if isinstance(converge_eligible, dict):
            for eligible_mask in converge_eligible.values():
                if eligible_mask is None:
                    continue

                if eligible_any is None:
                    eligible_any = eligible_mask
                else:
                    eligible_any = (eligible_any | eligible_mask)
        else:
            eligible_any = converge_eligible

        use_logit_kl = converged is not None and self.exit_criteria.needs_readout and readout_map
        prev_log_probs: dict[str, torch.Tensor] = {}

        if use_cache and kv_cache is not None:
            new_kv_cache = kv_cache
        else:
            new_kv_cache = None

        if use_cache and new_kv_cache is None:
            new_kv_cache = RecurrentKVCache(
                strategy=self.config.kv_cache_strategy,
                cache_budget=self.config.kv_cache_budget,
                max_iterations=self.config.kv_cache_max_iterations,
            )

        iteration = 0

        def _extend_mask_for_cache(mask, iter_cache):
            if mask is None or iter_cache is None or iter_cache.key_cache is None or iter_cache.seq_len == 0:
                return mask

            ones = torch.ones(mask.shape[0], iter_cache.seq_len, device=mask.device, dtype=mask.dtype)
            return torch.cat([ones, mask], dim=-1)

        metric_dict: MetricDict | None = None
        if n_steps_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_steps_no_grad):
                    thought_state, iteration_metrics = self._run_iteration(
                        x_0,
                        last_thought_state,
                        iteration,
                        attention_mask,
                        new_kv_cache,
                        position_offset,
                        use_cache,
                        _extend_mask_for_cache,
                        position_ids=position_ids,
                    )

                    if metric_dict and iteration_metrics:
                        metric_dict = metric_dict + iteration_metrics
                    else:
                        metric_dict = iteration_metrics

                    if not self.training:
                        newly_converged: torch.Tensor | None = None
                        kl_per_token: torch.Tensor | None = None
                        if use_logit_kl and isinstance(self.exit_criteria, recurrent_criteria.LogitKLCriteria):
                            exit_criteria: recurrent_criteria.LogitKLCriteria = self.exit_criteria
                            for key, readout_fn in readout_map.items():
                                eligible_mask = eligible_map.get(key)
                                if eligible_mask is not None and not bool(eligible_mask.any()):
                                    continue

                                logits = readout_fn(thought_state)
                                if logits is None:
                                    continue

                                if key not in prev_log_probs:
                                    prev_log_probs[key] = exit_criteria.init_state(logits)

                                log_probs = F.log_softmax(logits.float(), dim=-1)

                                exit_result = exit_criteria.step(
                                    prev_log_probs[key], log_probs,
                                )
                                converged_mask = exit_result.converged_mask
                                exit_values = exit_result.exit_values

                                prev_log_probs[key] = log_probs

                                if eligible_mask is not None:
                                    converged_mask = converged_mask & eligible_mask
                                    exit_values = torch.where(
                                        eligible_mask,
                                        exit_values,
                                        torch.zeros_like(exit_values)
                                    )

                                if newly_converged is None:
                                    newly_converged = converged_mask
                                else:
                                    newly_converged = (newly_converged | converged_mask)

                                if kl_per_token is None:
                                    kl_per_token = exit_values
                                else:
                                    kl_per_token = torch.maximum(kl_per_token, exit_values)
                        else:
                            exit_result = self.exit_criteria.step(last_thought_state, thought_state)
                            newly_converged = exit_result.converged_mask

                        if converged is not None and newly_converged is not None:
                            if eligible_any is not None:
                                newly_converged = newly_converged & eligible_any

                            already_converged = converged
                            converged = converged | newly_converged
                            thought_state = torch.where(
                                already_converged.unsqueeze(-1), last_thought_state, thought_state
                            )
                            if converged.all():
                                return LarryRecurrentTrunkOutput(
                                    logits=thought_state,
                                    kv_cache=new_kv_cache,
                                    n_iterations=iteration + 1,
                                    metrics=metric_dict,
                                )

                    last_thought_state = thought_state
                    iteration += 1

        if k_steps_grad > 0 and n_steps_no_grad > 0 and torch.is_autocast_enabled():
            torch.clear_autocast_cache()

        # only runs in training; no exit criteria handling required
        if k_steps_grad > 0:
            for _ in range(k_steps_grad):
                thought_state, iteration_metrics = self._run_iteration(
                    x_0,
                    last_thought_state,
                    iteration,
                    attention_mask,
                    new_kv_cache,
                    position_offset,
                    use_cache,
                    _extend_mask_for_cache,
                    position_ids=position_ids,
                )

                if metric_dict and iteration_metrics:
                    metric_dict = metric_dict + iteration_metrics
                else:
                    metric_dict = iteration_metrics

                last_thought_state = thought_state
                iteration += 1

        return LarryRecurrentTrunkOutput(
            logits=last_thought_state,
            kv_cache=new_kv_cache,
            n_iterations=iteration,
            metrics=metric_dict,
        )
