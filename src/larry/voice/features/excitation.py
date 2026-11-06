import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.stats import kurtosis

from larry.voice.config.preprocessing.voice_configs import ExcitationConfig


class ExcitationAnalyzer:
    METRICS = (
        "pulse_sharpness_real",
        "pulse_sharpness_generated",
        "pulse_sharpness_gap",
        "cpp_real",
        "cpp_generated",
        "cpp_gap",
    )

    def __init__(self, config: ExcitationConfig) -> None:
        self.config = config

    def voiced_frames(self, x: np.ndarray, size: int, voiced: np.ndarray, voiced_hop: int) -> np.ndarray:
        if len(x) < size or len(voiced) == 0:
            return np.zeros((0, size))

        frames = sliding_window_view(x.astype(np.float64), size)[::self.config.hop_length]
        centers = np.arange(len(frames)) * self.config.hop_length + size // 2
        keep = voiced[np.minimum(centers // voiced_hop, len(voiced) - 1)]
        keep &= np.abs(frames).max(axis=1) >= self.config.min_peak
        return frames[keep] * np.hanning(size)

    def lpc_residual(self, frame: np.ndarray) -> np.ndarray:
        order = self.config.lpc_order
        autocorrelation = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
        autocorrelation[0] *= 1 + self.config.lpc_regularization
        coefficients = solve_toeplitz(autocorrelation[:order], autocorrelation[1:order + 1])
        residual = lfilter(np.r_[1.0, -coefficients], [1.0], frame)
        return residual[self.config.residual_trim:-self.config.residual_trim]

    def pulse_sharpness(self, x: np.ndarray, voiced: np.ndarray, voiced_hop: int) -> float | None:
        frames = self.voiced_frames(x, self.config.lpc_window, voiced, voiced_hop)
        if len(frames) == 0:
            return None

        return float(np.median([kurtosis(self.lpc_residual(frame)) for frame in frames]))

    def cepstral_peak_prominence(self, x: np.ndarray, voiced: np.ndarray, voiced_hop: int) -> float | None:
        frames = self.voiced_frames(x, self.config.cepstrum_window, voiced, voiced_hop)
        if len(frames) == 0:
            return None

        log_power = np.log(np.abs(np.fft.fft(frames, axis=1)) ** 2 + 1e-10)
        cepstrum = 10 * np.log10(np.abs(np.fft.ifft(log_power, axis=1)) ** 2 + 1e-12)
        low = int(self.config.sample_rate / self.config.f0_ceil)
        high = int(self.config.sample_rate / self.config.f0_floor)
        quefrencies = np.arange(low, high)
        searched = cepstrum[:, low:high]
        slope, intercept = np.polyfit(quefrencies, searched.T, 1)
        peaks = searched.argmax(axis=1)
        trend = slope * quefrencies[peaks] + intercept
        return float(np.median(searched[np.arange(len(frames)), peaks] - trend))

    def compare(self, real: np.ndarray, fake: np.ndarray, voiced: np.ndarray, voiced_hop: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        measures = (("pulse_sharpness", self.pulse_sharpness), ("cpp", self.cepstral_peak_prominence))
        for name, measure in measures:
            before, after = measure(real, voiced, voiced_hop), measure(fake, voiced, voiced_hop)
            if before is None or after is None:
                continue

            metrics[f"{name}_real"] = before
            metrics[f"{name}_generated"] = after
            metrics[f"{name}_gap"] = after - before

        return metrics
