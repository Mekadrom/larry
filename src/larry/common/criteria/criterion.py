from abc import abstractmethod, ABC
from typing import ClassVar

import torch
from torch import nn
from torch.nn import Module

from larry.common.config.criteria.criterion_configs import WeightedCompositeCriterionConfig, CriterionConfig
from larry.common.model.model import LarryModelOutput
from larry.common.training.training_model import LarryModel
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Criterion[C: CriterionConfig, M: Module](nn.Module, Registrable, ABC, root=True):
    """Superclass for all ``Criterion``."""
    REGISTRY: ClassVar[TypeRegistry[Criterion]]

    _config: C
    _model: M

    def __init__(self, config: C, model: M):
        super().__init__()
        self._config = config
        self._model = model

    @property
    def config(self) -> C:
        """Returns immutable type-hinted instance of the stored config instance."""
        return self._config

    @property
    def model(self) -> M:
        """Returns immutable type-hinted instance of the stored model instance."""
        return self._model

    @abstractmethod
    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        """Forces subclasses to implement a forward function.

        Takes an abstract model output dataclass and arbitrary kwargs.
        """
        ...


class WeightedCompositeCriterion[M: LarryModel = LarryModel](Criterion[WeightedCompositeCriterionConfig, M]):
    """An implementation of ``Criterion`` which delegates.

     Delegates to one or more other ``Criterion`` with weights prior to summation.
     """

    def __init__(self, config: WeightedCompositeCriterionConfig, model: M) -> None:
        """Simply initializes a dictionary of delegate criteria and their weight configuration."""
        super().__init__(config, model)
        self.criteria = {}

    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        """Calls each delegate criterion and multiplies their output by their corresponding weight prior to summing."""
        # TODO: stub
        return torch.tensor(0.0, device=self.model.device)
