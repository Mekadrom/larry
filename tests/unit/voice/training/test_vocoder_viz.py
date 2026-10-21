import contextlib
from types import SimpleNamespace
from typing import Any

import torch
from datasets import Dataset

from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.config.model.voice_model_configs import LarryVocoderConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.model.vocoder import LarryVocoder
from larry.voice.training.voice_trainers import VocoderTrainer
from larry.voice.training.voice_viz_callbacks import VocoderVizCallback
from tests.helpers.audio import wav
from tests.helpers.common import uninitialized


class RecordingMetrics:
    def __init__(self) -> None:
        self.audio: dict[str, torch.Tensor] = {}
        self.figures: list[str] = []
        self.scalars: dict[str, float] = {}

    def add_audio(self, tag: str, waveform: torch.Tensor, step: int, **kwargs: Any) -> None:
        self.audio[tag] = waveform

    def add_figure(self, tag: str, figure: Any, step: int, **kwargs: Any) -> None:
        self.figures.append(tag)

    def add_scalar(self, tag: str, value: float, step: int, **kwargs: Any) -> None:
        self.scalars[tag] = value


def _dataset(name: str, seconds: list[float]) -> Dataset:
    return Dataset.from_dict({
        "id": [f"{name}{i}" for i in range(len(seconds))],
        "audio": [{"bytes": wav(s, 16_000, frequency=200 + 90 * i), "path": None} for i, s in enumerate(seconds)],
        "duration_s": seconds,
    })


def _callback() -> tuple[VocoderVizCallback, RecordingMetrics]:
    torch.manual_seed(0)
    configs = [SpeechSourceConfig(name="a", index_path="a"), SpeechSourceConfig(name="b", index_path="b")]
    dataset = IndexedSpeechDataset(configs, [_dataset("a", [1.3, 0.6]), _dataset("b", [0.9])], [None, None], 16_000)
    metrics = RecordingMetrics()
    trainer = uninitialized(
        VocoderTrainer,
        _model=LarryVocoder(LarryVocoderConfig(d_model=32, d_inner=64, n_layers=2)),
        _config=SimpleNamespace(mixture_config=SimpleNamespace(eval_sources=configs)),
        metrics=metrics,
        eval_loader=SimpleNamespace(dataset=dataset),
        accelerator=SimpleNamespace(
            is_main_process=True,
            device=torch.device("cpu"),
            unwrap_model=lambda m: m,
            autocast=contextlib.nullcontext,
        ),
    )
    return VocoderVizCallback(trainer, viz_steps=10, generation_steps=2, step_sweep=(1, 2)), metrics


def test_viz_logs_audio_figures_and_scalars_for_every_example() -> None:
    callback, metrics = _callback()

    callback.viz(10)

    assert sorted(metrics.figures) == ["viz/0/a/comparison", "viz/1/a/comparison", "viz/2/b/comparison"]
    assert {t for t in metrics.audio if t.endswith("/real")} == {"viz/0/a/real", "viz/1/a/real", "viz/2/b/real"}
    assert {"viz/0/a/mel_l1", "viz/0/a/mel_l1_high_band", "viz/0/a/loudness_db_delta"} <= set(metrics.scalars)
    assert {"viz/mel_l1_by_steps/1", "viz/mel_l1_by_steps/2"} <= set(metrics.scalars)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.scalars.values())


def test_real_and_generated_audio_have_matching_trimmed_lengths() -> None:
    callback, metrics = _callback()

    callback.viz(10)

    for i, source in ((0, "a"), (1, "a"), (2, "b")):
        assert len(metrics.audio[f"viz/{i}/{source}/real"]) == len(metrics.audio[f"viz/{i}/{source}/generated"])


def test_reference_audio_is_logged_once_and_generations_are_seeded() -> None:
    callback, metrics = _callback()

    callback.viz(10)
    first = metrics.audio["viz/0/a/generated"].clone()
    metrics.audio.clear()
    callback.viz(20)

    assert not any(t.endswith("/real") for t in metrics.audio)
    assert torch.equal(metrics.audio["viz/0/a/generated"], first)


def test_viz_restores_training_mode() -> None:
    callback, _ = _callback()
    callback.trainer.model.train()

    callback.viz(10)

    assert callback.trainer.model.training
