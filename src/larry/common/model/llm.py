import dataclasses

import torch
from torch import nn

from larry.common.config.model.model_configs import (
    AttentionModuleConfig,
    LarryConditionalEncoderBlockModuleConfig,
    LLMBlockModuleConfig, )
from larry.common.model.attention import LarryAttention, LarryAttentionOutput
from larry.common.model.kv_cache import KVCache
from larry.common.model.norms import AdaLayerNorm, AdaLayerNormZero, get_norm_type


@dataclasses.dataclass(kw_only=True)
class LarryEncoderBlockOutput(LarryAttentionOutput[KVCache]):
    kv_cache: KVCache | None = None


@dataclasses.dataclass(kw_only=True)
class LarryDecoderBlockOutput(LarryAttentionOutput[tuple[KVCache | None, KVCache | None]]):
    kv_cache: tuple[KVCache | None, KVCache | None] | None = None


class LarrySimpleEncoderBlock(nn.Module):
    """Simple transformer block consisting self attention and an MLP with post-norms."""

    def __init__(self, config: LLMBlockModuleConfig) -> None:
        """Sets up the modules for this encoder block: self attention and an MLP with post-norms."""
        super().__init__()

        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        norm1_type = get_norm_type(config.pre_attn_norm_type)
        if norm1_type == AdaLayerNorm:
            raise ValueError(f"{type(self).__name__} doesn't support {norm1_type.__name__}")
        else:
            self.norm1 = norm1_type(config.d_model)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )

        norm2_type = get_norm_type(config.pre_mlp_norm_type)
        if norm2_type == AdaLayerNorm:
            raise ValueError(f"{type(self).__name__} doesn't support {norm2_type.__name__}")
        else:
            self.norm2 = norm2_type(config.d_model)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            kv_cache: KVCache | None = None,
            position_ids: torch.Tensor | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryEncoderBlockOutput:
        """Simple forward pass through all modules with residuals."""
        attention_output = self.self_attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output.logits
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        attention_output.logits = hidden_states
        return attention_output


class LarrySimpleDecoderBlock(nn.Module):
    """Simple transformer decoder block consisting of self attention, cross attention, and an MLP with post-norms."""

    def __init__(self, config: LLMBlockModuleConfig) -> None:
        """Initializes a decoder block module with self attention, cross attention, and an MLP with post-norm."""
        super().__init__()

        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        self.self_attn_norm = nn.RMSNorm(config.d_model)

        decoder_config = AttentionModuleConfig.from_child(config)
        decoder_config.is_causal = False
        self.cross_attn = LarryAttention(decoder_config)
        self.cross_attn_norm = nn.RMSNorm(config.d_model)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.mlp_norm = nn.RMSNorm(config.d_model)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            position_offset: int = 0,
            kv_caches: tuple[KVCache | None, KVCache | None] | None = None,
            use_cache: bool = False,
    ) -> LarryDecoderBlockOutput:
        """Simple decoder forward through all modules with residuals."""
        self_attn_output = self.self_attn(
            self.self_attn_norm(hidden_states),
            attention_mask=self_attention_mask,
            kv_cache=kv_caches[0] if kv_caches else None,
            position_ids=position_ids,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self_attn_output.logits

        cross_attn_output = self.cross_attn(
            self.cross_attn_norm(hidden_states),
            kv_cache=kv_caches[1] if kv_caches else None,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + cross_attn_output.logits
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        kv_caches = (self_attn_output.kv_cache, cross_attn_output.kv_cache)

        if self_attn_output.metrics and cross_attn_output.metrics:
            combined_metrics = self_attn_output.metrics + cross_attn_output.metrics
        else:
            combined_metrics = cross_attn_output.metrics or self_attn_output.metrics

        return LarryDecoderBlockOutput(
            logits=hidden_states,
            metrics=combined_metrics,
            kv_cache=kv_caches,
        )


class LarryConditionalEncoderBlock(nn.Module):
    """Transformer block that uses AdaLN-Zero to condition simple attention/MLP modules."""

    def __init__(self, config: LarryConditionalEncoderBlockModuleConfig) -> None:
        """Sets up the modules for this encoder block: self attention and an MLP, each behind a gated AdaLN."""
        super().__init__()

        d_cond = config.d_cond or config.d_model
        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        self.attn_norm = AdaLayerNormZero(config.d_model, d_cond)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.mlp_norm = AdaLayerNormZero(config.d_model, d_cond)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cond: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            kv_cache: KVCache | None = None,
            position_ids: torch.Tensor | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
    ) -> LarryEncoderBlockOutput:
        """Pre-norm residual forward with conditioned norms and gates on each residual branch."""
        normed, gate = self.attn_norm(hidden_states, cond)
        attention_output = self.self_attn(
            normed,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
            position_offset=position_offset,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + gate * attention_output.logits
        normed, gate = self.mlp_norm(hidden_states, cond)
        hidden_states = hidden_states + gate * self.mlp(normed)
        return LarryEncoderBlockOutput(
            logits=hidden_states,
            metrics=attention_output.metrics,
            kv_cache=attention_output.kv_cache
        )
