import dataclasses
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from larry.common.config.model.model_configs import LarryConditionalEncoderBlockModuleConfig, LLMBlockModuleConfig
from larry.common.model.attention import LarryAttentionOutput
from larry.common.model.flow import ConditionalFlowMatching, LarryFlowMatchingOutput, TimestepEmbedding
from larry.common.model.init import WeightInitializer
from larry.common.model.kv_cache import KVCache, LarryMultimodalModelKVCache
from larry.common.model.llm import LarryConditionalEncoderBlock, LarrySimpleEncoderBlock
from larry.common.model.norms import AdaLayerNorm
from larry.common.model.recurrent import LarryRecurrentTrunk
from larry.common.training.training_model import LarryModel
from larry.image.model.image_modules import Patchify, Unpatchify
from larry.multimodal.config.model.model_configs import (
    ImageCodaConfig,
    ImagePreludeConfig,
    LarryMultimodalModelConfig,
    TextModuleConfig,
    VoiceCodaConfig,
    VoicePreludeConfig,
)
from larry.multimodal.processor.processor import (
    BOI,
    BOV,
    EOI,
    EOS,
    EOV,
    IPH,
    VPH,
    ImagePlaceholder,
    LarryProcessorBase,
    TextSpan,
    VoicePlaceholder,
)

INIT_STD = 0.02
PROCESSOR_DIR = "processor"


@dataclasses.dataclass(kw_only=True)
class LarryModalityModuleOutput(LarryAttentionOutput[list[KVCache | None]]):
    ...


@dataclasses.dataclass(kw_only=True)
class VoicePreludeOutput(LarryModalityModuleOutput):
    conv_state: torch.Tensor


@dataclasses.dataclass(kw_only=True)
class VoiceGroups:
    unit_ids: torch.Tensor  # (B, G, k)
    unit_valid: torch.Tensor  # (B, G, k)
    group_valid: torch.Tensor  # (B, G)


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalEncoding:
    thought_state: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    key_valid: torch.Tensor
    image_slots: torch.Tensor
    query_slots: torch.Tensor
    n_iterations: int
    text_states: torch.Tensor
    voice_states: torch.Tensor | None = None
    voice_groups: VoiceGroups | None = None
    grid_valid: torch.Tensor | None = None
    kv_cache: LarryMultimodalModelKVCache | None = None


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalModelOutput(LarryAttentionOutput[LarryMultimodalModelKVCache]):
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    n_iterations: int
    text_route: torch.Tensor
    text_targets: torch.Tensor
    voice_route: torch.Tensor
    voice_logits: torch.Tensor | None = None
    voice_targets: torch.Tensor | None = None
    image_route: torch.Tensor
    image_query_states: torch.Tensor | None = None
    image_query_mask: torch.Tensor | None = None
    image_flow: LarryFlowMatchingOutput | None = None


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalGeneration:
    input_ids: torch.Tensor  # (1, T) prompt and generated positions, stretched like the collator's
    modality_ids: torch.Tensor
    voice_unit_ids: list[torch.Tensor]  # generated units, one tensor per voice segment
    images: list[torch.Tensor]  # generated latents, (C, H, W) each
    finished: bool
    text_logits: list[torch.Tensor] = dataclasses.field(default_factory=list)
    voice_logits: list[torch.Tensor] = dataclasses.field(default_factory=list)
    image_query_states: list[torch.Tensor] = dataclasses.field(default_factory=list)


class LarryModalityModule(nn.Module):
    def __init__(self, config: LLMBlockModuleConfig, n_layers: int) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(n_layers)
        ])

    def run_blocks(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: list[KVCache | None] | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryModalityModuleOutput:
        new_kv_cache: list[KVCache | None] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = kv_cache[i] if kv_cache is not None else None
            output = block(
                hidden_states=x,
                attention_mask=block_cache.extend_mask(attention_mask) if block_cache is not None else attention_mask,
                kv_cache=block_cache,
                position_ids=position_ids,
                position_offset=position_offset,
                use_cache=use_cache,
            )
            x = output.logits
            if new_kv_cache is not None:
                new_kv_cache.append(output.kv_cache)

        return LarryModalityModuleOutput(logits=x, kv_cache=new_kv_cache)


class TextPrelude(LarryModalityModule):
    def __init__(self, config: TextModuleConfig) -> None:
        super().__init__(config, config.n_layers)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: list[KVCache | None] | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryModalityModuleOutput:
        return self.run_blocks(self.wte(input_ids), attention_mask, position_ids, kv_cache, position_offset, use_cache)


class ImagePrelude(LarryModalityModule):
    def __init__(self, config: ImagePreludeConfig) -> None:
        super().__init__(config, config.n_layers)

        self.patch_embed = nn.Conv2d(
            config.latent_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(
            self,
            latents: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> LarryModalityModuleOutput:
        x = self.patch_embed(latents).flatten(2).transpose(1, 2)  # (B, rows * cols, d_model), row-major
        return self.run_blocks(x, attention_mask, position_ids)


class VoicePrelude(LarryModalityModule):
    def __init__(self, config: VoicePreludeConfig) -> None:
        super().__init__(config, config.n_layers)

        self.ute = nn.Embedding(config.vocab_size, config.d_model)
        self.code_proj = nn.Linear(config.fsq_dims, config.d_model)
        unit_ids = torch.arange(config.vocab_size)
        radix = config.fsq_levels ** torch.arange(config.fsq_dims)
        # inverse of FSQ.indices: each unit id's fixed code in {-halve..halve} per FSQ dim
        codes = (unit_ids[:, None] // radix) % config.fsq_levels - config.fsq_levels // 2
        self.register_buffer("codes", codes.float(), persistent=False)

        self.units_per_position = config.units_per_position
        self.group_combine = config.group_combine
        self.group_proj = None
        self.slot_embeddings = None
        if self.units_per_position > 1 and self.group_combine == "concat":
            self.group_proj = nn.Linear(self.units_per_position * config.d_model, config.d_model)
        elif self.units_per_position > 1:
            self.slot_embeddings = nn.Parameter(torch.zeros(self.units_per_position, config.d_model))

        self.context_size = config.conv_kernel_size - 1
        self.local_conv = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=config.conv_kernel_size, groups=config.d_model
        )

    def unit_embeddings(self) -> torch.Tensor:
        return self.ute.weight + self.code_proj(self.codes)

    def combine(self, x: torch.Tensor, unit_valid: torch.Tensor) -> torch.Tensor:
        # (B, G, k, d) unit embeddings -> (B, G, d), one vector per trunk position
        if self.group_proj is not None:
            return self.group_proj(x.flatten(2))

        if self.slot_embeddings is not None:
            return (x + self.slot_embeddings * unit_valid[..., None]).sum(dim=2)

        return x[:, :, 0]

    def forward(
            self,
            unit_ids: torch.Tensor,
            unit_valid: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            kv_cache: list[KVCache | None] | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
            conv_state: torch.Tensor | None = None,
    ) -> VoicePreludeOutput:
        # padded units in a partial last group embed to zero
        x = F.embedding(unit_ids, self.unit_embeddings()) * unit_valid[..., None]
        x = self.combine(x, unit_valid)
        if conv_state is None:
            conv_state = x.new_zeros(x.shape[0], self.context_size, x.shape[-1])

        # previous groups first so the depthwise conv only ever looks back
        context = torch.cat([conv_state, x], dim=1)
        x = x + self.local_conv(context.transpose(1, 2)).transpose(1, 2)
        output = self.run_blocks(x, attention_mask, None, kv_cache, position_offset, use_cache)
        return VoicePreludeOutput(
            logits=output.logits,
            kv_cache=output.kv_cache,
            conv_state=context[:, context.shape[1] - self.context_size:]
        )


class TextCoda(LarryModalityModule):
    def __init__(self, config: TextModuleConfig) -> None:
        super().__init__(config, config.n_layers)
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: list[KVCache | None] | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryModalityModuleOutput:
        return self.run_blocks(x, attention_mask, position_ids, kv_cache, position_offset, use_cache)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(x))


class ImageCoda(nn.Module):
    def __init__(self, config: ImageCodaConfig) -> None:
        super().__init__()

        patch_dim = config.latent_channels * config.patch_size ** 2
        self.patchify = Patchify(config.patch_size)
        self.unpatchify = Unpatchify(config.patch_size)
        self.flow = ConditionalFlowMatching(config)
        self.cfg_dropout_rate = config.cfg_dropout_rate

        self.query = nn.Parameter(torch.zeros(config.d_model))
        self.row_queries = nn.Embedding(config.max_query_grid, config.d_model)
        self.column_queries = nn.Embedding(config.max_query_grid, config.d_model)
        # stands in for every query state of an image when conditioning is dropped
        self.null_query_state = nn.Parameter(torch.zeros(config.d_model))

        self.embed = nn.Linear(patch_dim + config.d_model, config.d_model)
        self.time = TimestepEmbedding(config.d_model)
        self.blocks = nn.ModuleList([
            LarryConditionalEncoderBlock(LarryConditionalEncoderBlockModuleConfig.from_child(config))
            for _ in range(config.n_layers)
        ])
        self.final_norm = AdaLayerNorm(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, patch_dim)

    def queries(self, rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        return self.query + self.row_queries(rows) + self.column_queries(columns)

    def velocity(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            query_states: torch.Tensor,
            mask: torch.Tensor,
            position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # each patch sees its own noisy latent and its query's trunk state; patches attend to each other
        x = self.embed(torch.cat([xt, query_states], dim=-1))
        time = self.time(t)
        for block in self.blocks:
            x = block(x, time, mask, position_ids=position_ids).logits

        return self.out(self.final_norm(x, time)) * mask[..., None]

    def forward(
            self,
            query_states: torch.Tensor,
            latents: torch.Tensor,
            mask: torch.Tensor,
            position_ids: torch.Tensor | None = None,
    ) -> LarryFlowMatchingOutput:
        if self.training and self.cfg_dropout_rate > 0:
            # drop whole images' conditioning so sampling can extrapolate away from the unconditional velocity
            dropped = torch.rand(query_states.shape[0], device=query_states.device) < self.cfg_dropout_rate
            query_states = torch.where(dropped[:, None, None], self.null_query_state, query_states)

        xt, t, target = self.flow.path(self.patchify(latents))
        prediction = self.velocity(xt, t, query_states, mask, position_ids)
        # (N, patch_dim, patches) to match the flow criterion's (B, channels, frames) layout
        return LarryFlowMatchingOutput(
            logits=prediction.transpose(1, 2),
            targets=target.transpose(1, 2),
            mask=mask.to(prediction.dtype),
        )

    def sample(
            self,
            query_states: torch.Tensor,
            mask: torch.Tensor,
            rows: int,
            columns: int,
            steps: int,
            position_ids: torch.Tensor | None = None,
            generator: torch.Generator | None = None,
            sway_coefficient: float | None = None,
            cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        shape = (*query_states.shape[:2], self.out.out_features)
        null_states = self.null_query_state.expand_as(query_states)

        def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            conditional = self.velocity(x, t, query_states, mask, position_ids)
            if cfg_scale == 1.0:
                return conditional

            unconditional = self.velocity(x, t, null_states, mask, position_ids)
            return unconditional + cfg_scale * (conditional - unconditional)

        patches = self.flow.sample(velocity_fn, shape, steps, query_states.device, generator, sway_coefficient)
        return self.unpatchify(patches, rows, columns)


class VoiceCoda(LarryModalityModule):
    def __init__(self, config: VoiceCodaConfig) -> None:
        super().__init__(config, config.n_layers)

        self.norm = nn.RMSNorm(config.d_model)
        self.eov = nn.Parameter(torch.zeros(config.d_model))
        self.units_per_position = config.units_per_position

        self.depth = None
        self.depth_input_norm = None
        if self.units_per_position > 1:
            # small causal decoder over the k units of one group, started from the trunk position's state
            self.depth = LarryModalityModule(config, config.depth_n_layers)
            self.depth_input_norm = nn.RMSNorm(config.d_model)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: list[KVCache | None] | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryModalityModuleOutput:
        return self.run_blocks(x, attention_mask, position_ids, kv_cache, position_offset, use_cache)

    def output_embeddings(self, unit_embeddings: torch.Tensor) -> torch.Tensor:
        # unit rows tied to the voice prelude's embeddings, plus a learned <eov> row
        return torch.cat([unit_embeddings, self.eov[None]])

    def head(self, x: torch.Tensor, unit_embeddings: torch.Tensor) -> torch.Tensor:
        return self.norm(x) @ self.output_embeddings(unit_embeddings).T

    def group_logits(
            self,
            states: torch.Tensor,
            previous: torch.Tensor,
            unit_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # (n, d) states and (n, j) earlier classes of each group -> (n, j + 1, units + 1) logits
        if self.depth is None or self.depth_input_norm is None:
            return self.head(states, unit_embeddings)[:, None]

        inputs = torch.cat([states[:, None], self.output_embeddings(unit_embeddings)[previous]], dim=1)
        steps = torch.arange(inputs.shape[1], device=inputs.device)
        # global axis only; the local axes stay 0 inside a group
        position_ids = F.pad(steps[:, None], (0, 2)).expand(inputs.shape[0], -1, -1)
        x = self.depth.run_blocks(self.depth_input_norm(inputs), position_ids=position_ids).logits
        return self.head(x, unit_embeddings)


class LarryMultimodalModel(LarryModel[LarryMultimodalModelConfig], PyTorchModelHubMixin):
    def __init__(self, config: LarryMultimodalModelConfig) -> None:
        super().__init__(config)

        self._register_preludes()
        self.recurrent_block = LarryRecurrentTrunk(config.recurrent_trunk_config)
        self._register_codas()

        self.text_coda.lm_head.weight = self.text_prelude.wte.weight
        self.special_base = config.text_prelude_config.vocab_size - (EOV + 1)
        self.voice_units_per_position = config.voice_units_per_position
        self.init_weights()

    @classmethod
    def _decode_arg(cls, expected_type: Any, value: Any) -> Any:
        # the hub mixin rebuilds dataclass configs one level deep; module configs nest, so rebuild them fully
        if expected_type is LarryMultimodalModelConfig and isinstance(value, dict):
            return LarryMultimodalModelConfig.from_dict(value)

        return super()._decode_arg(expected_type, value)

    def _register_preludes(self) -> None:
        self.text_prelude = TextPrelude(self.config.text_prelude_config)

        self.image_prelude = None
        if self.config.image_prelude_config is not None:
            self.image_prelude = ImagePrelude(self.config.image_prelude_config)

        self.voice_prelude = None
        if self.config.voice_prelude_config is not None:
            self.voice_prelude = VoicePrelude(self.config.voice_prelude_config)

    def _register_codas(self) -> None:
        self.text_coda = TextCoda(self.config.text_coda_config)

        self.image_coda = None
        if self.config.image_coda_config is not None:
            self.image_coda = ImageCoda(self.config.image_coda_config)

        self.voice_coda = None
        if self.config.voice_coda_config is not None:
            self.voice_coda = VoiceCoda(self.config.voice_coda_config)

    def _block_stacks(self) -> list[tuple[nn.ModuleList, int]]:
        trunk = self.config.recurrent_trunk_config
        # residual branches are summed once per layer per thinking step in the trunk
        stacks = [(self.recurrent_block.blocks, trunk.n_layers * max(trunk.mean_thinking_steps, 1))]
        for module in (self.text_prelude, self.image_prelude, self.voice_prelude, self.text_coda, self.voice_coda):
            if module is not None:
                stacks.append((module.blocks, len(module.blocks)))

        if self.voice_coda is not None and self.voice_coda.depth is not None:
            stacks.append((self.voice_coda.depth.blocks, len(self.voice_coda.depth.blocks)))

        if self.image_coda is not None:
            stacks.append((self.image_coda.blocks, len(self.image_coda.blocks)))

        return stacks

    @torch.no_grad()
    def init_weights(self) -> None:
        initializer = WeightInitializer("normal", INIT_STD)
        # the trunk's concat projection starts as identity, like a sum injection
        keep = set()
        if self.recurrent_block.iteration_projection is not None:
            keep = {id(p) for p in self.recurrent_block.iteration_projection.parameters()}

        initializer.apply(self, keep)
        for blocks, depth in self._block_stacks():
            initializer.scale_residuals(blocks, depth)

        if self.voice_coda is not None:
            nn.init.normal_(self.voice_coda.eov, std=INIT_STD)
        if self.image_coda is not None:
            nn.init.normal_(self.image_coda.query, std=INIT_STD)
            nn.init.normal_(self.image_coda.null_query_state, std=INIT_STD)
            # velocity starts at zero, as in the speech decoder
            nn.init.zeros_(self.image_coda.out.weight)
            nn.init.zeros_(self.image_coda.out.bias)

    @staticmethod
    def key_valid(input_ids: torch.Tensor, input_lengths: torch.Tensor | None) -> torch.Tensor:
        if input_lengths is None:
            return torch.ones_like(input_ids, dtype=torch.bool)

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        # (B, T) True for real tokens, False for padding
        return positions[None, :] < input_lengths[:, None].to(input_ids.device)

    @staticmethod
    def run_index(mask: torch.Tensor) -> torch.Tensor:
        # running count of masked tokens so far
        counts = mask.long().cumsum(dim=-1)
        previous = torch.zeros_like(mask)
        previous[:, 1:] = mask[:, :-1]
        # count before each run's first token
        run_starts = torch.where(mask & ~previous, counts - 1, torch.zeros_like(counts))
        # carry latest run start forward; offset within run
        return (counts - 1 - run_starts.cummax(dim=-1).values) * mask

    def modality_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # the processor's labels: delimiters text, except <eov> which the voice head emits
        voice = (input_ids == self.special_base + VPH) | (input_ids == self.special_base + EOV)
        image = input_ids == self.special_base + IPH
        modality_ids = torch.full_like(input_ids, TextSpan.modality_id)
        modality_ids[voice] = VoicePlaceholder.modality_id
        modality_ids[image] = ImagePlaceholder.modality_id
        return modality_ids

    def segment_pairs(self, slots: torch.Tensor, input_ids: torch.Tensor, opening_offset: int) -> torch.Tensor:
        # segment number each token falls after
        segment = (input_ids == self.special_base + opening_offset).cumsum(dim=-1)
        # (B, T, T) slot pairs within one segment
        return slots[:, :, None] & slots[:, None, :] & (segment[:, :, None] == segment[:, None, :])

    def attention_mask(
            self,
            input_ids: torch.Tensor,
            key_valid: torch.Tensor,
            is_synthesis: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        image = input_ids == self.special_base + IPH  # (B, T) image patch or query slots
        # input voice only; generated voice stays causal
        input_voice = (input_ids == self.special_base + VPH) & ~is_synthesis[:, None]
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device).tril()
        bidirectional = self.segment_pairs(image, input_ids, BOI) | self.segment_pairs(input_voice, input_ids, BOV)
        # hide padded keys; (B, 1, T, T) for heads
        return ((causal | bidirectional) & key_valid[:, None, :]).unsqueeze(1)

    @staticmethod
    def voice_prelude_mask(group_valid: torch.Tensor, is_synthesis: torch.Tensor) -> torch.Tensor:
        n_groups = group_valid.shape[1]
        causal = torch.ones(n_groups, n_groups, dtype=torch.bool, device=group_valid.device).tril()
        # (B, 1, G, G): causal for generated voice, bidirectional for input voice
        return ((causal | ~is_synthesis[:, None, None]) & group_valid[:, None, :]).unsqueeze(1)

    def group_units(self, voice_unit_ids: torch.Tensor, voice_unit_id_lengths: torch.Tensor) -> VoiceGroups:
        k = self.voice_units_per_position
        batch_size, n_units = voice_unit_ids.shape
        n_groups = math.ceil(n_units / k)
        unit_ids = F.pad(voice_unit_ids, (0, n_groups * k - n_units))
        positions = torch.arange(n_groups * k, device=voice_unit_ids.device)
        unit_valid = positions[None, :] < voice_unit_id_lengths[:, None].to(voice_unit_ids.device)
        return VoiceGroups(
            unit_ids=unit_ids.view(batch_size, n_groups, k),
            unit_valid=unit_valid.view(batch_size, n_groups, k),
            group_valid=unit_valid.view(batch_size, n_groups, k)[..., 0],
        )

    @staticmethod
    def grid_position_ids(rows: int, columns: int, device: torch.device) -> torch.Tensor:
        index = torch.arange(rows * columns, device=device)
        # (rows * cols, 3) global = flat patch index, then row and column
        return torch.stack([index, index // columns, index % columns], dim=-1)

    def position_ids(self, input_ids: torch.Tensor, image_sizes: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long, device=input_ids.device)
        # global axis: every position
        position_ids[..., 0] = torch.arange(seq_len, device=input_ids.device)
        # axis 1: group index within each voice
        position_ids[..., 1] += self.run_index(input_ids == self.special_base + VPH)

        if image_sizes is not None and self.config.image_patch_size is not None:
            # row-major patch index within each image
            patch_index = self.run_index(input_ids == self.special_base + IPH)
            # patches per row; clamp avoids /0 on imageless rows
            columns = (image_sizes[:, 0] // self.config.image_patch_size).clamp(min=1)
            # axis 1: patch row
            position_ids[..., 1] += patch_index // columns[:, None].to(input_ids.device)
            # axis 2: patch column
            position_ids[..., 2] += patch_index % columns[:, None].to(input_ids.device)

        return position_ids

    def valid_patches(self, images: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        patch_size = self.config.image_patch_size
        if patch_size is None:
            raise ValueError(
                "Cannot calculate valid patches for multimodal model with no image prelude or coda config.")

        # patch rows of the padded batch grid
        rows = torch.arange(images.shape[-2] // patch_size, device=images.device)
        # patch columns of the padded batch grid
        columns = torch.arange(images.shape[-1] // patch_size, device=images.device)
        # real patch columns per image
        widths = (image_sizes[:, 0] // patch_size).to(images.device)
        # real patch rows per image
        heights = (image_sizes[:, 1] // patch_size).to(images.device)
        # (B, rows, cols) inside the real image
        valid = (rows[None, :, None] < heights[:, None, None]) & (columns[None, None, :] < widths[:, None, None])
        return valid.flatten(1)  # row-major, matching patchify order

    @staticmethod
    def target_modality(modality_ids: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
        # -1: no target (last or padded position)
        target = torch.full_like(modality_ids, -1)
        # position t takes token t+1's modality
        target[:, :-1] = torch.where(key_valid[:, 1:], modality_ids[:, 1:], -1)
        return target

    def encode(
            self,
            input_ids: torch.Tensor,
            is_synthesis: torch.Tensor | None = None,
            input_lengths: torch.Tensor | None = None,
            images: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            voice_unit_ids: torch.Tensor | None = None,
            voice_unit_id_lengths: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: LarryMultimodalModelKVCache | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
            run_voice_coda: bool = False,
    ) -> LarryMultimodalEncoding:
        if self.training:
            use_cache = False

        if is_synthesis is None:
            is_synthesis = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        is_synthesis = is_synthesis.to(device=input_ids.device, dtype=torch.bool)

        key_valid = self.key_valid(input_ids, input_lengths)
        if attention_mask is None:
            attention_mask = self.attention_mask(input_ids, key_valid, is_synthesis)

        if position_ids is None:
            position_ids = self.position_ids(input_ids, image_sizes)

        cache = kv_cache if kv_cache is not None else LarryMultimodalModelKVCache()
        # modules that don't run on this chunk keep their earlier cache
        new_cache = cache.copy() if use_cache else None

        prelude_output = self.text_prelude(
            input_ids,
            attention_mask,
            position_ids,
            cache.text_prelude_cache,
            position_offset,
            use_cache
        )

        input_embeddings = prelude_output.logits
        if new_cache is not None:
            new_cache.text_prelude_cache = prelude_output.kv_cache

        image_slots = input_ids == self.special_base + IPH
        input_image_slots = image_slots & ~is_synthesis[:, None]
        query_slots = image_slots & is_synthesis[:, None]

        grid_valid: torch.Tensor | None = None
        # collators always send an images tensor, empty for text and voice rows
        if images is not None and image_sizes is not None and image_slots.any():
            grid_valid = self.valid_patches(images, image_sizes)  # (B, rows * cols) real patches of each latent

        patch_size = self.config.image_patch_size
        if input_image_slots.any() and patch_size is not None:
            if self.image_prelude is None or images is None or grid_valid is None:
                raise ValueError("input images need an image prelude, images and image_sizes")

            input_rows = input_image_slots.any(dim=1)  # only rows that have an input image run the prelude
            row_valid = grid_valid[input_rows]
            grid_positions = self.grid_position_ids(
                images.shape[-2] // patch_size,
                images.shape[-1] // patch_size,
                images.device
            ).expand(*row_valid.shape, 3)
            patches = self.image_prelude(images[input_rows], row_valid, grid_positions).logits
            patches = patches[row_valid]  # (n_patches, d) drops padding patches
            # fills slots in row-major order
            input_embeddings = input_embeddings.masked_scatter(
                input_image_slots[..., None], patches.to(input_embeddings.dtype)
            )

        if query_slots.any():
            if self.image_coda is None or image_sizes is None:
                raise ValueError("image synthesis needs an image coda and image_sizes for its queries")

            # learned queries placed by their patch row and column
            queries = self.image_coda.queries(position_ids[..., 1][query_slots], position_ids[..., 2][query_slots])
            input_embeddings = input_embeddings.masked_scatter(
                query_slots[..., None],
                queries.to(input_embeddings.dtype)
            )

        voice_slots = (input_ids == self.special_base + VPH)[..., None]  # (B, T, 1) broadcasts over d_model
        voice_groups: VoiceGroups | None = None
        if voice_unit_ids is not None and voice_unit_id_lengths is not None:
            voice_groups = self.group_units(voice_unit_ids, voice_unit_id_lengths)

        if voice_groups is not None and self.voice_prelude is not None and voice_slots.any():
            voice_prelude_offset = 0
            if cache.voice_prelude_cache and cache.voice_prelude_cache[0] is not None:
                # groups already cached in the voice prelude
                voice_prelude_offset = cache.voice_prelude_cache[0].seq_len

            voice_output = self.voice_prelude(
                voice_groups.unit_ids,
                voice_groups.unit_valid,
                self.voice_prelude_mask(voice_groups.group_valid, is_synthesis),
                cache.voice_prelude_cache,
                voice_prelude_offset,
                use_cache,
                cache.voice_prelude_conv_state,
            )
            if new_cache is not None:
                new_cache.voice_prelude_cache = voice_output.kv_cache
                new_cache.voice_prelude_conv_state = voice_output.conv_state

            groups = voice_output.logits[voice_groups.group_valid]  # (n_groups, d) in row-major order
            # group i fills the i-th <vph> slot
            input_embeddings = input_embeddings.masked_scatter(voice_slots, groups.to(input_embeddings.dtype))

        trunk_output = self.recurrent_block(
            input_embeddings,
            attention_mask=attention_mask,
            kv_cache=cache.recurrent_cache,
            position_offset=position_offset,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        thought_state = trunk_output.logits
        if new_cache is not None:
            new_cache.recurrent_cache = trunk_output.kv_cache

        text_output = self.text_coda(
            thought_state,
            attention_mask,
            position_ids,
            cache.text_coda_cache,
            position_offset,
            use_cache
        )

        if new_cache is not None:
            new_cache.text_coda_cache = text_output.kv_cache

        voice_states = None
        if run_voice_coda and self.voice_coda is not None:
            voice_output = self.voice_coda(
                thought_state,
                attention_mask,
                position_ids,
                cache.voice_coda_cache,
                position_offset,
                use_cache
            )

            voice_states = voice_output.logits
            if new_cache is not None:
                new_cache.voice_coda_cache = voice_output.kv_cache

        return LarryMultimodalEncoding(
            thought_state=thought_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            key_valid=key_valid,
            image_slots=image_slots,
            query_slots=query_slots,
            n_iterations=trunk_output.n_iterations,
            text_states=text_output.logits,
            voice_states=voice_states,
            voice_groups=voice_groups,
            grid_valid=grid_valid,
            kv_cache=new_cache,
        )

    def voice_group_targets(
            self,
            voice_route: torch.Tensor,
            input_ids: torch.Tensor,
            voice_unit_ids: torch.Tensor,
            voice_unit_id_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.voice_coda_config is None:
            raise ValueError("Don't call voice_group_targets if no voice modules are configured")

        k = self.voice_units_per_position
        eov_class = self.config.voice_coda_config.vocab_size - 1
        route = torch.zeros_like(voice_route)
        targets = []
        for row in voice_route.any(dim=1).nonzero().flatten().tolist():
            if int((input_ids[row] == self.special_base + BOV).sum()) > 1:
                raise NotImplementedError("voice targets support one generated voice segment per row")

            n_units = int(voice_unit_id_lengths[row])
            # the stream the voice head emits: every unit, then <eov>, chunked into groups of k
            stream = torch.cat([voice_unit_ids[row, :n_units], voice_unit_ids.new_tensor([eov_class])])
            n_target_groups = math.ceil(stream.shape[0] / k)
            groups = F.pad(stream, (0, n_target_groups * k - stream.shape[0]), value=-100).view(n_target_groups, k)
            # a partial last group already holds <eov>, so the inserted <eov> after it has no group to predict
            positions = voice_route[row].nonzero().flatten()[:n_target_groups]
            route[row, positions] = True
            targets.append(groups[:positions.shape[0]])

        return route, torch.cat(targets) if targets else voice_unit_ids.new_zeros(0, k)

    def forward(
            self,
            input_ids: torch.Tensor,
            modality_ids: torch.Tensor,
            input_lengths: torch.Tensor | None = None,
            is_synthesis: torch.Tensor | None = None,
            images: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            voice_unit_ids: torch.Tensor | None = None,
            voice_unit_id_lengths: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            kv_cache: LarryMultimodalModelKVCache | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryMultimodalModelOutput:
        if is_synthesis is None:
            is_synthesis = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        is_synthesis = is_synthesis.to(device=input_ids.device, dtype=torch.bool)

        key_valid = self.key_valid(input_ids, input_lengths)
        target_modality = self.target_modality(modality_ids, key_valid)
        # positions predicting a generated group of units or <eov>
        voice_route = (target_modality == VoicePlaceholder.modality_id) & is_synthesis[:, None]
        wants_voice = (
                self.voice_coda is not None
                and self.voice_prelude is not None
                and voice_unit_ids is not None
                and voice_unit_id_lengths is not None
                and bool(voice_route.any())
        )

        encoding = self.encode(
            input_ids,
            is_synthesis,
            input_lengths,
            images,
            image_sizes,
            voice_unit_ids,
            voice_unit_id_lengths,
            attention_mask,
            position_ids,
            kv_cache,
            position_offset,
            use_cache,
            run_voice_coda=wants_voice,
        )

        image_slots, query_slots = encoding.image_slots, encoding.query_slots

        next_ids = torch.full_like(input_ids, -100)
        # target id for each position: the next token
        next_ids[:, :-1] = input_ids[:, 1:]

        # positions predicting a text token; image slots never do, <eoi> is inserted
        text_route = (target_modality == TextSpan.modality_id) & ~image_slots
        # head only on routed positions
        text_logits = self.text_coda.head(encoding.text_states[text_route])

        voice_logits = voice_targets = None
        if (self.voice_prelude is not None
                and self.voice_coda is not None
                and voice_unit_ids is not None
                and voice_unit_id_lengths is not None
                and encoding.voice_states is not None):
            voice_route, voice_targets = self.voice_group_targets(
                voice_route,
                input_ids,
                voice_unit_ids,
                voice_unit_id_lengths
            )
            # teacher forcing: each group's earlier classes come from its targets
            previous = voice_targets[:, :-1].masked_fill(voice_targets[:, :-1] < 0, 0)
            voice_logits = self.voice_coda.group_logits(
                encoding.voice_states[voice_route],
                previous,
                self.voice_prelude.unit_embeddings()
            )
        else:
            voice_route = torch.zeros_like(voice_route)

        # image queries predict their own patch; not shifted
        image_route = query_slots
        image_query_states = image_query_mask = image_flow = None
        patch_size = self.config.image_patch_size
        if (self.image_coda is not None
                and query_slots.any()
                and encoding.grid_valid is not None
                and images is not None
                and patch_size is not None):
            synthesis_rows = query_slots.any(dim=1)
            image_query_mask = encoding.grid_valid[synthesis_rows]
            thought_state = encoding.thought_state

            # (n_images, rows * cols, d) query states laid out on each latent's padded patch grid
            image_query_states = thought_state.new_zeros(*image_query_mask.shape, thought_state.shape[-1])
            image_query_states = image_query_states.masked_scatter(
                image_query_mask[..., None],
                thought_state[query_slots]
            )

            grid_positions = self.grid_position_ids(
                images.shape[-2] // patch_size,
                images.shape[-1] // patch_size,
                images.device
            ).expand(*image_query_mask.shape, 3)

            image_flow = self.image_coda(image_query_states, images[synthesis_rows], image_query_mask, grid_positions)

        return LarryMultimodalModelOutput(
            logits=text_logits,
            attention_mask=encoding.attention_mask,
            position_ids=encoding.position_ids,
            n_iterations=encoding.n_iterations,
            text_route=text_route,
            text_targets=next_ids[text_route],
            voice_route=voice_route,
            voice_logits=voice_logits,
            voice_targets=voice_targets,
            image_route=image_route,
            image_query_states=image_query_states,
            image_query_mask=image_query_mask,
            image_flow=image_flow,
            kv_cache=encoding.kv_cache,
        )

    def check_processor(self, processor: LarryProcessorBase) -> None:
        mismatches = {
            "text vocab": (processor.text_tokenizer.vocab_size() + EOV + 1, self.config.text_prelude_config.vocab_size),
            "image_patch_size": (processor.config.image_patch_size, self.config.image_patch_size),
            "voice_units_per_position": (processor.config.voice_units_per_position, self.voice_units_per_position),
        }
        for name, (from_processor, from_model) in mismatches.items():
            if from_model is not None and from_processor != from_model:
                raise ValueError(f"processor and model disagree on {name}: {from_processor} != {from_model}")

    def save_with_processor(self, save_directory: str | Path, processor: LarryProcessorBase, **kwargs: Any) -> None:
        # the processor decides how prompts stretch, so it ships next to the weights it was trained with
        self.check_processor(processor)
        self.save_pretrained(save_directory, **kwargs)
        processor.save_pretrained(Path(save_directory) / PROCESSOR_DIR)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> LarryMultimodalGeneration:
        generator_keys = {field.name for field in dataclasses.fields(LarryMultimodalGenerationConfig)}
        config = LarryMultimodalGenerationConfig(**{k: v for k, v in kwargs.items() if k in generator_keys})
        prompt = {
            k: v
            for k, v in kwargs.items()
            if k not in generator_keys
        }
        return LarryMultimodalGenerator(self, config).run(input_ids, **prompt)


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalGenerationConfig:
    max_new_positions: int = 512
    temperature: float = 1.0
    top_k: int | None = None
    voice_temperature: float = 1.0
    image_size: tuple[int, int] | None = None  # (width, height) of generated latents
    image_steps: int = 32
    image_cfg_scale: float = 1.0
    sway_coefficient: float | None = None
    generator: torch.Generator | None = None
    record: bool = False


class LarryMultimodalGenerator:
    def __init__(self, model: LarryMultimodalModel, config: LarryMultimodalGenerationConfig) -> None:
        self.model = model
        self.config = config
        self.base = model.special_base
        self.ids: list[int] = []
        self.cache: LarryMultimodalModelKVCache | None = None
        self.offset = 0
        self.generated = 0
        self.voices: list[torch.Tensor] = []
        self.images: list[torch.Tensor] = []
        self.text_logits: list[torch.Tensor] = []
        self.voice_logits: list[torch.Tensor] = []
        self.image_query_states: list[torch.Tensor] = []

    @property
    def device(self) -> torch.device:
        return self.model.text_prelude.wte.weight.device

    def sample(self, logits: torch.Tensor, head: Literal["text", "voice"]) -> int:
        temperature = self.config.temperature if head == "text" else self.config.voice_temperature
        if temperature <= 0:
            return int(logits.argmax())

        logits = logits / temperature
        if self.config.top_k is not None:
            threshold = logits.topk(min(self.config.top_k, logits.shape[-1])).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        return int(torch.multinomial(logits.softmax(dim=-1), 1, generator=self.config.generator))

    def step(self, chunk: list[int], is_synthesis: bool, **media: torch.Tensor | None) -> LarryMultimodalEncoding:
        encoding = self.model.encode(
            torch.tensor([chunk], device=self.device),
            torch.tensor([is_synthesis], device=self.device),
            kv_cache=self.cache,
            position_offset=self.offset,
            use_cache=True,
            run_voice_coda=self.model.voice_coda is not None,
            **media,
        )
        self.cache = encoding.kv_cache
        self.offset += len(chunk)
        self.ids.extend(chunk)
        return encoding

    def next_text_logits(self, encoding: LarryMultimodalEncoding) -> torch.Tensor:
        logits = self.model.text_coda.head(encoding.text_states[0, -1])
        if self.config.record:
            self.text_logits.append(logits)

        # tokens the text head never emits: slots, closing delimiters, and openers without a coda
        blocked = [VPH, IPH, EOV, EOI]
        if self.model.voice_coda is None:
            blocked.append(BOV)

        if self.model.image_coda is None:
            blocked.append(BOI)

        return logits.index_fill(0, torch.tensor(blocked, device=logits.device) + self.base, float("-inf"))

    def voice_segment(self, encoding: LarryMultimodalEncoding) -> LarryMultimodalEncoding:
        if (self.model.config.voice_coda_config is None
                or self.model.voice_prelude is None
                or self.model.voice_coda is None):
            raise ValueError("Don't call voice_segment if no voice modules are configured")

        if self.cache is None:
            raise ValueError("KV Cache needs to be used for voice generation")

        k = self.model.voice_units_per_position
        eov_class = self.model.config.voice_coda_config.vocab_size - 1
        unit_embeddings = self.model.voice_prelude.unit_embeddings()

        # every voice segment starts its own prelude context, as each training row has one
        self.cache.voice_prelude_cache = None
        self.cache.voice_prelude_conv_state = None
        units: list[int] = []
        n_groups = 0

        while self.generated < self.config.max_new_positions:
            # each group starts from the newest position's state, so read it from the latest encoding
            voice_states = encoding.voice_states
            if voice_states is None:
                return encoding

            state = voice_states[0, -1:]
            group: list[int] = []
            group_logits = []
            while len(group) < k and (not group or group[-1] != eov_class):
                previous = torch.tensor([group], dtype=torch.long, device=self.device)
                logits = self.model.voice_coda.group_logits(state, previous, unit_embeddings)[0, -1]
                group_logits.append(logits)
                group.append(self.sample(logits, "voice"))

            if self.config.record:
                self.voice_logits.append(torch.stack(group_logits))

            group_units = [
                unit
                for unit in group
                if unit != eov_class
            ]
            ended = group[-1] == eov_class
            chunk = ([self.base + VPH] if group_units else []) + ([self.base + EOV] if ended else [])
            chunk_ids = torch.tensor([chunk], device=self.device)
            position_ids = self.model.position_ids(chunk_ids, None)
            # the group index continues across chunks
            position_ids[..., 1] += (chunk_ids == self.base + VPH) * n_groups
            media = {}
            if group_units:
                media = {
                    "voice_unit_ids": torch.tensor([group_units], device=self.device),
                    "voice_unit_id_lengths": torch.tensor([len(group_units)], device=self.device),
                }
            encoding = self.step(chunk, True, position_ids=position_ids, **media)
            self.generated += len(chunk)
            units.extend(group_units)
            n_groups += bool(group_units)
            if ended:
                break

        self.voices.append(torch.tensor(units, dtype=torch.long))
        return encoding

    def image(self) -> LarryMultimodalEncoding:
        if self.model.image_coda is None:
            raise ValueError("Image coda is required to be configured for image generation")

        if self.config.image_size is None:
            raise ValueError("generating an image needs image_size=(width, height) in latent cells")

        patch_size = self.model.config.image_patch_size
        if patch_size is None:
            raise ValueError("generating an image needs a configured patch_size")

        if self.cache is None:
            raise ValueError("KV Cache is required for image generation")

        width, height = self.config.image_size
        rows = height // patch_size
        columns = width // patch_size

        n_patches = rows * columns
        image_sizes = torch.tensor([[width, height]], device=self.device)
        before_queries = self.cache.copy()

        encoding = self.step([self.base + IPH] * n_patches, True, image_sizes=image_sizes)
        states = encoding.thought_state[:, -n_patches:]
        if self.config.record:
            self.image_query_states.append(states[0])

        latents = self.model.image_coda.sample(
            states,
            torch.ones(1, n_patches, dtype=torch.bool, device=self.device),
            rows,
            columns,
            self.config.image_steps,
            self.model.grid_position_ids(rows, columns, self.device)[None],
            self.config.generator,
            self.config.sway_coefficient,
            self.config.image_cfg_scale,
        )
        self.images.append(latents[0])
        self.generated += n_patches + 1

        if self.model.image_prelude is None:
            # no prelude to re-encode with: text continues from the query states, exactly as in training
            return self.step([self.base + EOI], False)

        # the finished image replaces its queries and is re-encoded as an input image
        self.cache = before_queries
        self.offset -= n_patches
        del self.ids[-n_patches:]
        chunk = [self.base + IPH] * n_patches + [self.base + EOI]
        return self.step(chunk, False, images=latents, image_sizes=image_sizes)

    @torch.no_grad()
    def run(
            self,
            input_ids: torch.Tensor,
            images: torch.Tensor | None = None,
            image_sizes: torch.Tensor | None = None,
            voice_unit_ids: torch.Tensor | None = None,
            voice_unit_id_lengths: torch.Tensor | None = None,
    ) -> LarryMultimodalGeneration:
        if input_ids.shape[0] != 1:
            raise ValueError("generate decodes one sequence at a time")

        if int(input_ids[0, -1]) == self.base + EOS:
            raise ValueError("the prompt ends with <eos>; drop it so the model continues the sequence")

        # prompt media are inputs
        encoding = self.step(
            input_ids[0].tolist(),
            False,
            images=images,
            image_sizes=image_sizes,
            voice_unit_ids=voice_unit_ids,
            voice_unit_id_lengths=voice_unit_id_lengths,
        )

        last = int(input_ids[0, -1])
        finished = False
        while self.generated < self.config.max_new_positions:
            if last == self.base + BOV:
                encoding = self.voice_segment(encoding)
            elif last == self.base + BOI:
                encoding = self.image()

            last = self.sample(self.next_text_logits(encoding), "text")
            self.generated += 1
            if last == self.base + EOS:
                self.ids.append(last)
                finished = True
                break

            encoding = self.step([last], False)

        ids = torch.tensor([self.ids], device=self.device)
        return LarryMultimodalGeneration(
            input_ids=ids,
            modality_ids=self.model.modality_ids(ids),
            voice_unit_ids=self.voices,
            images=self.images,
            finished=finished,
            text_logits=self.text_logits,
            voice_logits=self.voice_logits,
            image_query_states=self.image_query_states,
        )
