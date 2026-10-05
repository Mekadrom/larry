import dataclasses
import logging
from abc import abstractmethod, ABC
from typing import ClassVar

import torch
from torch import nn

from larry.common.config.model.model_configs import ModelConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class GenerationOutput:
    ...


class GenerativeModel[I, O: GenerationOutput](ABC):
    @abstractmethod
    def generate(
            self,
            conditioning: I,
            *,
            num_samples: int = 1,
            generator: torch.Generator | None = None,
            **sampling_kwargs
    ) -> O:
        ...


class LarryModel[C: ModelConfig = ModelConfig](nn.Module, Registrable, ABC, root=True):
    REGISTRY: ClassVar[TypeRegistry[LarryModel]]

    def __init__(self, config: C):
        super().__init__()
        self._config = config
        self.log = logging.getLogger(type(self).__name__)

    @property
    def config(self) -> C:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
