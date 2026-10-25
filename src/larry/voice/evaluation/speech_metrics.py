import numpy as np
import torch
import torchaudio.functional as AF
from pesq import PesqError, pesq

PESQ_WB_SAMPLE_RATE = 16_000


def pesq_wb(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float | None:
    if sample_rate != PESQ_WB_SAMPLE_RATE:
        reference = AF.resample(reference, sample_rate, PESQ_WB_SAMPLE_RATE)
        degraded = AF.resample(degraded, sample_rate, PESQ_WB_SAMPLE_RATE)

    n = min(reference.shape[-1], degraded.shape[-1])
    try:
        return float(pesq(
            PESQ_WB_SAMPLE_RATE,
            reference[:n].double().cpu().numpy().astype(np.float64),
            degraded[:n].double().cpu().numpy().astype(np.float64),
            "wb",
        ))
    except PesqError:
        return None
