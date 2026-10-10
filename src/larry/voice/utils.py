import numpy as np
import torch
from torchcodec.decoders import AudioDecoder

from larry.common.utils.convert_np import make_np


def mel_to_db(mels: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    mels = make_np(mels)
    if mels.ndim == 3:
        mels = mels.squeeze(0)
    return mels * 10.0 / np.log(10.0)


def bytes_to_waveforms(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    return AudioDecoder(audio_bytes, sample_rate=sample_rate, num_channels=1).get_all_samples().data


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    if max_len <= 0:
        max_len = int(lengths.max().item())

    seq_range = torch.arange(
        0,
        max_len,
        dtype=torch.int64,
        device=lengths.device
    )

    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask
