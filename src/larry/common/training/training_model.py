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
    """Superclass for any sort of model output from models/modules in this project."""
    ...


class GenerativeModel[I, O: GenerationOutput](ABC):
    """Superclass for all models that can generate output at inference."""

    @abstractmethod
    def generate(
            self,
            conditioning: I,
            *,
            num_samples: int = 1,
            generator: torch.Generator | None = None,
            **sampling_kwargs
    ) -> O:
        """Accepts various configurations and evaluates a model with successive iterative state to generate outputs."""
        ...


class LarryModel[C: ModelConfig = ModelConfig](nn.Module, Registrable, ABC, root=True):
    """Superclass for most models in this project that need a config and device."""
    REGISTRY: ClassVar[TypeRegistry[LarryModel]]

    def __init__(self, config: C):
        """Simple init to provide a config and logger."""
        self.log = logging.getLogger(type(self).__name__)

        super().__init__()
        self._config = config

    @property
    def config(self) -> C:
        """Immutable type-hint getter for the config instance."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Immutable type-hint getter for the device this model is on."""
        return next(self.parameters()).device
