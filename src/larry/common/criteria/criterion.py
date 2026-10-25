import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import nn

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.model.model import LarryModelOutput
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class LossTerm:
    """One weighted component of a loss; per row so it can be summed over any grouping of rows.

    The component's value over a set of rows is ``sums.sum() / counts.sum()``. A component averaged per row has counts
    of 1; one averaged per token has counts of each row's tokens.
    """

    weight: float
    sums: torch.Tensor
    counts: torch.Tensor

    @property
    def value(self) -> torch.Tensor:
        return self.sums.sum() / self.counts.sum().clamp(min=1e-8)


def weighted_total(terms: dict[str, LossTerm]) -> torch.Tensor:
    values = [term.weight * term.value for term in terms.values()]
    return torch.stack(values).sum() if values else torch.zeros(())


class Criterion[O: LarryModelOutput, C: CriterionConfig](nn.Module, Registrable, ABC, root=True):
    """Superclass for all ``Criterion``."""

    REGISTRY: ClassVar[TypeRegistry[Criterion]]

    _config: C

    def __init__(self, config: C) -> None:
        """Simple init."""
        super().__init__()
        self._config = config

    @property
    def config(self) -> C:
        """Returns immutable type-hinted instance of the stored config instance."""
        return self._config

    @abstractmethod
    def forward(self, model_output: O, **batch_kwargs: Any) -> torch.Tensor:
        """Forces subclasses to implement a forward function.

        Takes an abstract model output dataclass and arbitrary kwargs.
        """
        ...

    def loss_terms(self, model_output: O, **batch_kwargs: Any) -> dict[str, LossTerm]:
        """The weighted components of the loss, per row. Defaults to the whole loss as one term without rows."""
        loss = self(model_output, **batch_kwargs)
        return {"loss": LossTerm(weight=1.0, sums=loss.reshape(1), counts=torch.ones(1, device=loss.device))}


class TermCriterion[O: LarryModelOutput, C: CriterionConfig](Criterion[O, C], ABC):
    @abstractmethod
    def loss_terms(self, model_output: O, **batch_kwargs: Any) -> dict[str, LossTerm]:
        ...

    def forward(self, model_output: O, **batch_kwargs: Any) -> torch.Tensor:
        return weighted_total(self.loss_terms(model_output, **batch_kwargs))


class SamplableCriterion[O: LarryModelOutput = LarryModelOutput, C: CriterionConfig = CriterionConfig](
    TermCriterion[O, C], ABC
):
    @abstractmethod
    def sample_losses(
            self,
            model_output: O,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ...
