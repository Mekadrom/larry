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
    REGISTRY: ClassVar[TypeRegistry[Criterion]]

    _config: C
    _model: M

    def __init__(self, config: C, model: M):
        super().__init__()
        self._config = config
        self._model = model

    @property
    def config(self) -> C:
        return self._config

    @property
    def model(self) -> M:
        return self._model

    @abstractmethod
    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        ...


class WeightedCompositeCriterion[M: LarryModel = LarryModel](Criterion[WeightedCompositeCriterionConfig, M]):
    def __init__(self, config: WeightedCompositeCriterionConfig, model: LarryModel) -> None:
        super().__init__(config, model)
        self.criteria = {}

    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        return torch.tensor(0.0, device=self.model.device)
