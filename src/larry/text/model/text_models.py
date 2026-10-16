import dataclasses
from abc import ABC

import torch

from larry.common.training.training_model import GenerationOutput, LarryModel
from larry.text.config.model.text_model_configs import TextModelConfig


@dataclasses.dataclass(kw_only=True)
class TextGenerationOutput(GenerationOutput):
    """Helper dataclass for returning the results of the model's generation."""

    sequences: torch.Tensor  # [B*num_samples, T+new]
    prompt_length: int

    @property
    def new_tokens(self) -> torch.Tensor:
        """Returns only the new tokens."""
        return self.sequences[:, self.prompt_length:]


class TextModel[C: TextModelConfig = TextModelConfig](LarryModel[C], ABC):
    """Superclass for text models."""

    ...
