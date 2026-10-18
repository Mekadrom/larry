from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import nn

from larry.common.config.criteria.criterion_configs import CriterionConfig
from larry.common.model.model import LarryModelOutput
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


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


class SamplableCriterion[O: LarryModelOutput = LarryModelOutput](ABC):
    @abstractmethod
    def sample_losses(
            self,
            model_output: O,
            targets: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ...
