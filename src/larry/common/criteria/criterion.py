from abc import abstractmethod, ABC
from typing import ClassVar

import torch
from torch import nn

from larry.common.config.criteria.criterion_configs import WeightedCompositeCriterionConfig, CriterionConfig
from larry.common.model.model import LarryModelOutput
from larry.common.training import training_model
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


class Criterion[C: CriterionConfig](nn.Module, Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[Criterion]]

    config: C

    def __init__(self, config: C):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        ...


class WeightedCompositeCriterion(Criterion[WeightedCompositeCriterionConfig]):
    def __init__(self, config: WeightedCompositeCriterionConfig, model: training_model.LarryModel) -> None:
        super().__init__(config)
        self.model = model
        self.criteria = {}

    def forward(self, model_output: LarryModelOutput, **batch_kwargs) -> torch.Tensor:
        return torch.tensor(0.0, device=self.model.device)
