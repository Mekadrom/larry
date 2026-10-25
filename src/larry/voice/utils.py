import numpy as np
import torch
from torchcodec.decoders import AudioDecoder

from larry.common.utils.convert_np import make_np


def mel_to_db(mels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts log10 power mels to DB scale."""
    mels = make_np(mels)
    if mels.ndim == 3:
        mels = mels.squeeze(0)
    return mels * 10.0


def bytes_to_waveforms(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    """Converts audio bytes to a decoded waveform tensor."""
    return AudioDecoder(audio_bytes, sample_rate=sample_rate, num_channels=1).get_all_samples().data


def collapse_ctc_emissions(frames: torch.Tensor, blank: int) -> list[int]:
    """Collapses consecutive values into single frames and removes blank indices."""
    ids = torch.unique_consecutive(frames)
    return ids[ids != blank].tolist()
