import math
from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn

from larry.common.model.norms import AdaLayerNorm, AdaLayerNormZero

type InitScheme = Literal["normal", "trunc_normal", "xavier_uniform"]


class WeightInitializer:
    def __init__(self, scheme: InitScheme = "normal", std: float = 0.02) -> None:
        self.scheme = scheme
        self.std = std

    @staticmethod
    def zero_initialized(root: nn.Module) -> set[int]:
        # AdaLN modulations start at zero so every conditioned block starts as the identity
        return {
            id(parameter)
            for module in root.modules() if isinstance(module, (AdaLayerNorm, AdaLayerNormZero))
            for parameter in module.parameters()
        }

    def init_weight(self, weight: torch.Tensor) -> None:
        match self.scheme:
            case "normal":
                nn.init.normal_(weight, std=self.std)
            case "trunc_normal":
                nn.init.trunc_normal_(weight, std=self.std, a=-2 * self.std, b=2 * self.std)
            case "xavier_uniform":
                nn.init.xavier_uniform_(weight)

    @torch.no_grad()
    def apply(self, root: nn.Module, keep: set[int] | None = None) -> None:
        keep = self.zero_initialized(root) | (keep or set())
        for module in root.modules():
            if isinstance(module, nn.Embedding) and id(module.weight) not in keep:
                nn.init.normal_(module.weight, std=self.std)
            elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and id(module.weight) not in keep:
                self.init_weight(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def residual_outputs(block: nn.Module) -> list[nn.Module]:
        outputs = []
        attention = getattr(block, "self_attn", None)
        if attention is None:
            attention = getattr(block, "attn", None)

        if attention is not None:
            outputs.append(attention.o_proj)
            # the FSMN memory branch adds into the same residual as the attention output
            if getattr(attention, "fsmn", None) is not None:
                outputs.append(attention.fsmn)

        mlp = getattr(block, "mlp", None)
        if isinstance(mlp, nn.Sequential):
            outputs.append(mlp[-1])

        return outputs

    @torch.no_grad()
    def scale_residuals(self, blocks: Iterable[nn.Module], depth: int) -> None:
        # GPT-2 style: residual branch outputs shrink with the number of branches summed into the stream
        scale = 1 / math.sqrt(2 * max(depth, 1))
        for block in blocks:
            for output in self.residual_outputs(block):
                output.weight.mul_(scale)  # pyright: ignore[reportCallIssue]
