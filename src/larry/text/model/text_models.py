import dataclasses
from abc import ABC

import torch
from torch import nn

from larry.common.config.model.model_configs import LLMBlockModuleConfig
from larry.common.model.llm import LarrySimpleLLMBlock
from larry.common.model.model import LarryModelOutput
from larry.common.training.training_model import LarryModel, GenerationOutput, GenerativeModel
from larry.text.config.model.text_model_configs import TextModelConfig, SmokeTestLLMModelConfig


@dataclasses.dataclass(kw_only=True)
class TextGenerationOutput(GenerationOutput):
    sequences: torch.Tensor  # [B*num_samples, T+new]
    prompt_length: int

    @property
    def new_tokens(self) -> torch.Tensor:
        return self.sequences[:, self.prompt_length:]


class TextModel[C: TextModelConfig = TextModelConfig](LarryModel[C], ABC):
    ...


class SmokeTestLLM(GenerativeModel[torch.Tensor, TextGenerationOutput], TextModel[SmokeTestLLMModelConfig]):
    def __init__(self, config: SmokeTestLLMModelConfig) -> None:
        super().__init__(config)

        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        block_fields = {f.name for f in dataclasses.fields(LLMBlockModuleConfig)}
        self.blocks = nn.Sequential(*[
            LarrySimpleLLMBlock(LLMBlockModuleConfig(**{
                k: v
                for k, v in dataclasses.asdict(config).items()
                if k in block_fields
            }))
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        self.lm_head.weight = self.embed.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **_
    ) -> LarryModelOutput:
        hidden_states = self.embed(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.final_norm(hidden_states)
        return LarryModelOutput(
            logits=self.lm_head(hidden_states)
        )

    @torch.no_grad()
    def generate(
            self,
            conditioning: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None = None,
            num_samples: int = 1,
            max_new_tokens: int = 100,
            eos_token_id: int | None = None,
            pad_token_id: int | None = None,
            generator: torch.Generator | None = None,
            **sampling_kwargs
    ) -> TextGenerationOutput:
        if conditioning.ndim == 1:
            conditioning = conditioning.unsqueeze(0)
        elif conditioning.ndim != 2:
            raise ValueError(f"Accepted conditioning input shapes: [B, N], [N]; got: {conditioning.shape}")
        if eos_token_id is None:
            raise ValueError("Please specify eos_token_id.")
        if "temperature" in sampling_kwargs and sampling_kwargs["temperature"] == 0.0:
            self.log.info("Setting num_samples to 1 for greedy decoding; outputs would be redundant otherwise.")
            num_samples = 1

        pad_token_id = eos_token_id if pad_token_id is None else pad_token_id

        input_ids = conditioning
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if num_samples > 1:
            input_ids = input_ids.repeat_interleave(num_samples, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)

        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        prompt_length = input_ids.shape[1]

        for _ in range(max_new_tokens):
            model_output = self.forward(input_ids, attention_mask=attention_mask)
            logits = model_output.logits[:, -1, :]  # (B, V)

            next_tokens = self._sample(logits, generator=generator, **sampling_kwargs)

            next_tokens = torch.where(unfinished, next_tokens, pad_token_id)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, unfinished[:, None].to(attention_mask.dtype)], dim=1)

            unfinished &= next_tokens != eos_token_id
            if not unfinished.any():
                break
        return TextGenerationOutput(sequences=input_ids, prompt_length=prompt_length)

    @staticmethod
    @torch.no_grad()
    def _sample(
            logits: torch.Tensor,
            temperature: float = 1.0,
            generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)
