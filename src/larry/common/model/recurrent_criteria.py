import dataclasses
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


@dataclasses.dataclass(kw_only=True)
class RecurrentExitCriteriaOutput:
    converged_mask: torch.Tensor
    exit_values: torch.Tensor


class RecurrentExitCriteria(ABC):
    needs_readout = False

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @abstractmethod
    def step(self, last_output: torch.Tensor, output: torch.Tensor) -> RecurrentExitCriteriaOutput:
        ...


class NoOpCriteria(RecurrentExitCriteria):
    def __init__(self) -> None:
        super().__init__(float("-inf"))

    def step(self, last_output: torch.Tensor, output: torch.Tensor) -> RecurrentExitCriteriaOutput:
        exit_values = torch.zeros(output.shape[:-1], device=output.device)
        return RecurrentExitCriteriaOutput(converged_mask=exit_values < self.threshold, exit_values=exit_values)


class LatentDiffCriteria(RecurrentExitCriteria):
    @staticmethod
    def exit_values(last_output: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        delta = (output - last_output).norm(dim=-1)
        return delta / output.norm(dim=-1).clamp_min(1e-6)

    def step(self, last_output: torch.Tensor, output: torch.Tensor) -> RecurrentExitCriteriaOutput:
        exit_values = self.exit_values(last_output, output)
        return RecurrentExitCriteriaOutput(converged_mask=exit_values < self.threshold, exit_values=exit_values)


class LogitKLCriteria(RecurrentExitCriteria):
    needs_readout: bool = True

    @staticmethod
    def init_state(logits: torch.Tensor) -> torch.Tensor:
        vocab = logits.shape[-1]
        return torch.full(logits.shape, 1.0 / vocab, device=logits.device, dtype=torch.float32).log()

    def step(
            self,
            last_output: torch.Tensor,
            output: torch.Tensor
    ) -> RecurrentExitCriteriaOutput:
        exit_values = F.kl_div(output, last_output, reduction="none", log_target=True).sum(dim=-1)
        return RecurrentExitCriteriaOutput(converged_mask=exit_values < self.threshold, exit_values=exit_values)
