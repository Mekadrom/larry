import torch
from torch import nn

from larry.common.config.model.model_configs import LLMBlockModuleConfig, AttentionModuleConfig
from larry.common.model.attention import LarryAttention


class LarrySimpleEncoderBlock(nn.Module):
    def __init__(self, config: LLMBlockModuleConfig) -> None:
        super().__init__()

        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        self.norm1 = nn.LayerNorm(config.d_model)
        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states), attention_mask=attention_mask)[0]
        return hidden_states + self.mlp(self.norm2(hidden_states))


class LarrySimpleDecoderBlock(nn.Module):
    def __init__(self, config: LLMBlockModuleConfig) -> None:
        super().__init__()

        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        self.self_attn_norm = nn.LayerNorm(config.d_model)

        decoder_config = AttentionModuleConfig.from_child(config)
        decoder_config.is_causal = False
        self.cross_attn = LarryAttention(decoder_config)
        self.cross_attn_norm = nn.LayerNorm(config.d_model)

        d_inner = config.d_model * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(d_inner, config.d_model),
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
