import numpy as np
import pytest
import torch

from larry.voice.config.preprocessing.voice_configs import WorldConfig
from larry.voice.evaluation.speech_metrics import pesq_wb
from larry.voice.features.world import WorldAnalyzer, WorldFeatures


def _voice(f0: float, seconds: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    phase = 2 * np.pi * f0 * (t + 0.002 * np.sin(2 * np.pi * 4 * t))
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 20))
    return (0.2 * harmonics + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.mark.parametrize("method", ["dio", "harvest"])
def test_analysis_tracks_f0_on_the_mel_frame_grid(method: str) -> None:
    analyzer = WorldAnalyzer(WorldConfig(f0_method=method))
    x = _voice(140.0)

    features = analyzer.analyze(torch.from_numpy(x))

    assert features.f0.shape == (len(x) // 160 + 1,)
    assert features.aperiodicity.shape == (len(features.f0), 8)
    assert features.voiced.mean() > 0.8
    assert np.median(features.f0[features.voiced]) == pytest.approx(140.0, rel=0.03)


def test_comparison_of_identical_features_is_perfect() -> None:
    analyzer = WorldAnalyzer(WorldConfig(f0_method="dio"))
    features = analyzer.analyze(_voice(140.0))

    metrics = WorldAnalyzer.compare(features, features)

    assert metrics["f0_rmse_cents"] == 0 and metrics["vuv_error"] == 0 and metrics["aperiodicity_l1"] == 0
    assert metrics["f0_corr"] == pytest.approx(1.0)


def test_an_octave_error_is_1200_cents() -> None:
    f0 = np.array([0, 100, 120, 140, 0], dtype=np.float32)
    aperiodicity = np.zeros((5, 8), dtype=np.float32)

    metrics = WorldAnalyzer.compare(
        WorldFeatures(f0=f0, aperiodicity=aperiodicity), WorldFeatures(f0=2 * f0, aperiodicity=aperiodicity)
    )

    assert metrics["f0_rmse_cents"] == pytest.approx(1200.0)
    assert metrics["vuv_error"] == 0


def test_normalized_log_f0_is_speaker_invariant() -> None:
    f0 = np.array([0, 100, 120, 140, 0, 110], dtype=np.float32)
    aperiodicity = np.zeros((6, 8), dtype=np.float32)

    low = WorldFeatures(f0=f0, aperiodicity=aperiodicity).normalized_log_f0()
    high = WorldFeatures(f0=1.8 * f0, aperiodicity=aperiodicity).normalized_log_f0()

    assert np.allclose(low, high, atol=1e-5)
    assert low[0] == low[4] == 0
    assert low[f0 > 0].mean() == pytest.approx(0.0, abs=1e-5)


def test_pesq_wb_ranks_degradation_and_skips_silence() -> None:
    clean = torch.from_numpy(_voice(140.0, seconds=2.0))
    noisy = clean + 0.05 * torch.randn_like(clean)

    assert pesq_wb(clean, clean, 16_000) > 4.0
    assert pesq_wb(clean, noisy, 16_000) < pesq_wb(clean, clean, 16_000)
    assert pesq_wb(torch.zeros(32_000), torch.zeros(32_000), 16_000) is None
