import torch
from torch import nn

from larry.common.config.model.model_configs import LLMBlockModuleConfig, AttentionModuleConfig, ModelConfig
from larry.common.model.attention import LarryAttention


class LarrySimpleEncoderBlock(nn.Module):
    def __init__(self, config: LLMBlockModuleConfig) -> None:
        super().__init__()

        self.self_attn = LarryAttention(AttentionModuleConfig.from_child(config))
        self.norm1 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_inner),
            nn.GELU(),
            nn.Dropout(p=config.mlp_dropout_p),
            nn.Linear(config.d_inner, config.d_model),
        )
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.norm1(hidden_states), attention_mask=attention_mask)[0]
        return hidden_states + self.mlp(self.norm2(hidden_states))
