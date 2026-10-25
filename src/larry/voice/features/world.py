import dataclasses

import numpy as np
import pyworld
import torch

from larry.voice.config.preprocessing.voice_configs import WorldConfig


@dataclasses.dataclass(kw_only=True)
class WorldFeatures:
    f0: np.ndarray
    aperiodicity: np.ndarray

    @property
    def voiced(self) -> np.ndarray:
        return self.f0 > 0

    def normalized_log_f0(self) -> np.ndarray:
        voiced = self.voiced
        normalized = np.zeros_like(self.f0)
        if voiced.sum() < 2:
            return normalized

        log_f0 = np.log(self.f0[voiced])
        normalized[voiced] = (log_f0 - log_f0.mean()) / max(float(log_f0.std()), 1e-3)
        return normalized


class WorldAnalyzer:
    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self.frame_period = 1000 * config.hop_length / config.sample_rate
        self.band_edges = self.make_band_edges()

    def make_band_edges(self) -> np.ndarray:
        n_bins = self.fft_size() // 2 + 1
        nyquist = self.config.sample_rate / 2
        mels = np.linspace(0.0, 2595 * np.log10(1 + nyquist / 700), self.config.n_aperiodicity_bands + 1)
        hz = 700 * (10 ** (mels / 2595) - 1)
        return np.round(hz / nyquist * (n_bins - 1)).astype(int)

    def fft_size(self) -> int:
        return pyworld.get_cheaptrick_fft_size(self.config.sample_rate, self.config.f0_floor)

    def track_f0(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_rate, floor, ceil = self.config.sample_rate, self.config.f0_floor, self.config.f0_ceil
        if self.config.f0_method == "harvest":
            return pyworld.harvest(x, sample_rate, f0_floor=floor, f0_ceil=ceil, frame_period=self.frame_period)

        f0, times = pyworld.dio(x, sample_rate, f0_floor=floor, f0_ceil=ceil, frame_period=self.frame_period)
        return pyworld.stonemask(x, f0, times, sample_rate), times

    def band_aperiodicity(self, aperiodicity: np.ndarray) -> np.ndarray:
        return np.stack([
            aperiodicity[:, lo:max(hi, lo + 1)].mean(axis=1)
            for lo, hi in zip(self.band_edges[:-1], self.band_edges[1:], strict=True)
        ], axis=1)

    def analyze(self, waveform: np.ndarray | torch.Tensor) -> WorldFeatures:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().float().cpu().numpy()

        x = np.ascontiguousarray(waveform, dtype=np.float64)
        f0, times = self.track_f0(x)
        aperiodicity = pyworld.d4c(x, f0, times, self.config.sample_rate, fft_size=self.fft_size())
        return WorldFeatures(
            f0=f0.astype(np.float32),
            aperiodicity=self.band_aperiodicity(aperiodicity).astype(np.float32),
        )

    @staticmethod
    def compare(real: WorldFeatures, fake: WorldFeatures) -> dict[str, float]:
        frames = min(len(real.f0), len(fake.f0))
        real_f0, fake_f0 = real.f0[:frames], fake.f0[:frames]
        real_voiced, fake_voiced = real_f0 > 0, fake_f0 > 0
        metrics = {
            "vuv_error": float((real_voiced != fake_voiced).mean()) if frames else float("nan"),
            "aperiodicity_l1": float(np.abs(real.aperiodicity[:frames] - fake.aperiodicity[:frames]).mean()),
        }

        both = real_voiced & fake_voiced
        if both.sum() >= 2:
            real_log, fake_log = np.log2(real_f0[both]), np.log2(fake_f0[both])
            metrics["f0_rmse_cents"] = float(np.sqrt(np.mean((1200 * (fake_log - real_log)) ** 2)))
            if real_log.std() > 0 and fake_log.std() > 0:
                metrics["f0_corr"] = float(np.corrcoef(real_log, fake_log)[0, 1])

        return metrics
