import torch
from torch import nn

from larry.common.config.model.model_configs import (
    AttentionModuleConfig,
    LarryConditionalEncoderBlockModuleConfig,
    LLMBlockModuleConfig,
)
from larry.common.model.attention import LarryAttention
from larry.common.model.norms import AdaLayerNorm, AdaLayerNormZero, get_norm_type


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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Simple forward pass through all modules with residuals."""
        hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states), attention_mask=attention_mask)[0]
        return hidden_states + self.mlp(self.norm2(hidden_states))


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
    ) -> torch.Tensor:
        """Simple decoder forward through all modules with residuals."""
        attended, _ = self.self_attn(
            self.self_attn_norm(hidden_states),
            attention_mask=self_attention_mask,
        )
        hidden_states = hidden_states + attended

        crossed, _ = self.cross_attn(
            self.cross_attn_norm(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + crossed

        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


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
            attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pre-norm residual forward with conditioned norms and gates on each residual branch."""
        normed, gate = self.attn_norm(hidden_states, cond)
        hidden_states = hidden_states + gate * self.self_attn(normed, attention_mask=attention_mask)[0]
        normed, gate = self.mlp_norm(hidden_states, cond)
        return hidden_states + gate * self.mlp(normed)
