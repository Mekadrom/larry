import numpy as np
import pytest
import yaml

from larry.voice.config.preprocessing.voice_configs import ExcitationConfig
from larry.voice.features.excitation import ExcitationAnalyzer
from tests.helpers.config_recipes import CONFIG_ROOT

SAMPLE_RATE = 16_000
HOP = 160


def _harmonics(f0: float = 120.0, seconds: float = 1.0, dispersed: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    count = int(SAMPLE_RATE / 2 / f0) - 1
    phases = rng.uniform(0, 2 * np.pi, count) if dispersed else np.zeros(count)
    wave = sum(np.cos(2 * np.pi * f0 * (k + 1) * t + phases[k]) for k in range(count))
    return (0.5 * wave / np.abs(wave).max()).astype(np.float32)


def _voiced(x: np.ndarray) -> np.ndarray:
    return np.ones(len(x) // HOP + 1, dtype=bool)


def test_aligned_harmonics_are_sharper_than_the_same_spectrum_with_dispersed_phase() -> None:
    analyzer = ExcitationAnalyzer(ExcitationConfig())
    aligned, dispersed = _harmonics(), _harmonics(dispersed=True)

    assert analyzer.pulse_sharpness(aligned, _voiced(aligned), HOP) > 3 * analyzer.pulse_sharpness(
        dispersed, _voiced(dispersed), HOP
    )


def test_periodic_audio_has_more_cepstral_prominence_than_noise() -> None:
    analyzer = ExcitationAnalyzer(ExcitationConfig())
    voice = _harmonics()
    noise = (0.2 * np.random.default_rng(1).standard_normal(len(voice))).astype(np.float32)

    assert analyzer.cepstral_peak_prominence(voice, _voiced(voice), HOP) > (
        analyzer.cepstral_peak_prominence(noise, _voiced(noise), HOP) + 5
    )


def test_gaps_are_generated_minus_real() -> None:
    analyzer = ExcitationAnalyzer(ExcitationConfig())
    real, fake = _harmonics(), _harmonics(dispersed=True)

    metrics = analyzer.compare(real, fake, _voiced(real), HOP)

    assert set(metrics) == set(ExcitationAnalyzer.METRICS)
    assert metrics["pulse_sharpness_gap"] == pytest.approx(
        metrics["pulse_sharpness_generated"] - metrics["pulse_sharpness_real"]
    )
    assert metrics["pulse_sharpness_gap"] < 0


def test_only_voiced_frames_are_measured() -> None:
    analyzer = ExcitationAnalyzer(ExcitationConfig())
    voice = _harmonics()
    noise = (0.2 * np.random.default_rng(1).standard_normal(len(voice))).astype(np.float32)
    joined = np.concatenate([voice, noise])
    voiced = np.zeros(len(joined) // HOP + 1, dtype=bool)
    voiced[:len(voice) // HOP - 8] = True

    assert analyzer.pulse_sharpness(joined, voiced, HOP) == pytest.approx(
        analyzer.pulse_sharpness(voice, _voiced(voice), HOP), rel=0.1
    )


def test_clips_without_voiced_frames_report_nothing() -> None:
    analyzer = ExcitationAnalyzer(ExcitationConfig())
    voice = _harmonics()
    silent = np.zeros_like(voice)

    assert analyzer.compare(voice, voice, np.zeros(len(voice) // HOP + 1, dtype=bool), HOP) == {}
    assert analyzer.compare(silent, silent, _voiced(silent), HOP) == {}
    assert analyzer.compare(voice[:100], voice[:100], _voiced(voice[:100]), HOP) == {}


def test_auxiliary_loss_overlays_each_change_one_term() -> None:
    overlays = CONFIG_ROOT / "training/voice/vocoder/overlays"

    def criterion(name: str) -> dict[str, object]:
        return yaml.safe_load((overlays / f"{name}.yaml").read_text())["trainer"]["criterion"]

    assert criterion("aux_spectral") == {"stft_weight": 0.1}
    assert criterion("aux_mel") == {"mel_weight": 0.1}
    assert criterion("aux_spectral_short") == {"stft_weight": 0.1, "stft_fft_sizes": [64, 128, 256, 512]}
