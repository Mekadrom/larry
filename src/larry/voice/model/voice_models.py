import dataclasses
from abc import ABC
from collections.abc import Mapping

import torch

from larry.common.model.flow import LarryFlowMatchingOutput
from larry.common.model.model import LarryModelOutput
from larry.common.training.training_model import LarryModel
from larry.voice.config.model.voice_model_configs import VoiceModelConfig


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderModelOutput(LarryModelOutput):
    """``SpeechTokenizerModel`` returns CTC log probs, frame lengths, and unit_ids in addition to logits or metrics."""

    ctc_log_probs: Mapping[str, torch.Tensor]
    frame_lengths: torch.Tensor
    ctc_lengths: torch.Tensor
    unit_ids: torch.Tensor
    mels: torch.Tensor
    mel_lengths: torch.Tensor


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderModelOutput(LarryFlowMatchingOutput, LarryModelOutput):
    speaker_mus: torch.Tensor
    speaker_logvars: torch.Tensor
    kl: torch.Tensor
    speaker_embeddings: torch.Tensor

    prior: torch.Tensor
    prior_targets: torch.Tensor
    frame_mask: torch.Tensor


@dataclasses.dataclass(kw_only=True)
class LarryVocoderOutput(LarryFlowMatchingOutput):
    t: torch.Tensor
    waveform_estimates: torch.Tensor


class VoiceModel[C: VoiceModelConfig = VoiceModelConfig](LarryModel[C], ABC):
    """Superclass for voice models."""

    ...
