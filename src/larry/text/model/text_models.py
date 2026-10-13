import dataclasses
from abc import ABC

import torch

from larry.common.training.training_model import LarryModel, GenerationOutput
from larry.text.config.model.text_model_configs import TextModelConfig


@dataclasses.dataclass(kw_only=True)
class TextGenerationOutput(GenerationOutput):
    sequences: torch.Tensor  # [B*num_samples, T+new]
    prompt_length: int

    @property
    def new_tokens(self) -> torch.Tensor:
        return self.sequences[:, self.prompt_length:]


class TextModel[C: TextModelConfig = TextModelConfig](LarryModel[C], ABC):
    ...
