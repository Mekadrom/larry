from typing import Any

import pytest
import torch
from datasets import Dataset

from larry.voice.config.data.voice_data_configs import SpeechSourceConfig
from larry.voice.config.evaluation.evaluation_configs import IntelligibilityConfig, PretrainedVocoderConfig
from larry.voice.config.evaluation.recognizer_configs import (
    LarrySpeechEncoderRecognizerConfig,
    Wav2Vec2RecognizerConfig,
)
from larry.voice.config.training.voice_trainer_configs import SpeechDecoderVizConfig, VocoderVizConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.data.dataloading.voice_dataloading import WaveformCollator
from larry.voice.evaluation.intelligibility import IntelligibilityScorer
from larry.voice.training.intelligibility_probe import IntelligibilityProbe
from larry.voice.utils import unpad
from tests.helpers.audio import wav
from tests.unit.voice.evaluation.test_speech_recognition import EchoRecognizer, EchoRecognizerConfig


def echo_probe(
        dataset: IndexedSpeechDataset, names: list[str], collate_fn: Any, per_source: int = 2,
) -> IntelligibilityProbe:
    return IntelligibilityProbe(
        [EchoRecognizer(EchoRecognizerConfig(), torch.device("cpu"))],
        IntelligibilityScorer(str.lower),
        dataset,
        names,
        collate_fn,
        IntelligibilityConfig(clips_per_source=per_source, batch_size=2),
    )


def _dataset() -> IndexedSpeechDataset:
    configs = [SpeechSourceConfig(name="a", index_path="a"), SpeechSourceConfig(name="b", index_path="b")]
    sources = []
    for name, seconds in (("a", [0.5, 0.25]), ("b", [0.375])):
        lengths = [int(s * 16_000) for s in seconds]
        sources.append(Dataset.from_dict({
            "id": [f"{name}{i}" for i in range(len(seconds))],
            "audio": [{"bytes": wav(s, 16_000), "path": None} for s in seconds],
            "duration_s": seconds,
            "text_normalized": [f"LEN {n}" for n in lengths],
        }))
    return IndexedSpeechDataset(configs, sources, [None, None], 16_000)


def test_the_scorer_reports_every_clip_set_and_its_gap_to_the_baseline_per_group() -> None:
    scorer = IntelligibilityScorer(str.lower)
    references = ["the cat sat", "a dog ran", "hello there"]
    transcripts = {
        "original": ["the cat sat", "a dog ran", "hello where"],
        "generated": ["the bat sat", "a dog", "hello where"],
    }

    scores = scorer.compare(references, transcripts, "original", ["clean", "clean", "noisy"])

    assert scores["wer_original"] == pytest.approx(1 / 8)
    assert scores["wer_generated"] == pytest.approx(3 / 8)
    assert scores["wer_gap_generated"] == pytest.approx(2 / 8)
    assert scores["wer_gap_generated/clean"] == pytest.approx(2 / 6)
    assert scores["wer_gap_generated/noisy"] == 0
    assert scores["cer_original/clean"] == 0
    assert not any(name.startswith(("wer_gap_original", "cer_gap_original")) for name in scores)


def test_the_probe_renders_and_transcribes_fixed_clip_sets_once() -> None:
    probe = echo_probe(_dataset(), ["a", "b"], WaveformCollator())
    calls = {"fixed": 0, "render": 0}

    def fixed(batch: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
        calls["fixed"] += 1
        return {"original": unpad(batch["waveforms"], batch["waveform_lengths"])}

    def render(batch: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
        calls["render"] += 1
        return {"generated": [clip[: len(clip) // 2] for clip in unpad(batch["waveforms"], batch["waveform_lengths"])]}

    first = probe.score(render, 16_000, baseline="original", fixed=fixed)
    second = probe.score(render, 16_000, baseline="original", fixed=fixed)

    assert first == second
    assert calls == {"fixed": 2, "render": 4}
    assert len(probe.recognizers[0].batches) == 2 + 2 * 2
    assert first["echo/wer_original"] == 0
    assert first["echo/wer_gap_generated/a"] == pytest.approx(0.5)
    assert first["echo/wer_generated/b"] == pytest.approx(0.5)


def test_the_probe_reads_references_from_the_source_rows_by_eval_source() -> None:
    probe = echo_probe(_dataset(), ["a", "b"], WaveformCollator(), per_source=1)

    assert probe.references == (["LEN 8000", "LEN 6000"], ["a", "b"])


def test_viz_configs_choose_recognizers_by_type() -> None:
    config = VocoderVizConfig.from_dict({"intelligibility": {"recognizers": [
        {"type": "LarrySpeechEncoderRecognizerConfig", "model_path": "local/encoder"},
        {"type": "Wav2Vec2RecognizerConfig"},
    ]}})

    recognizers = config.intelligibility.recognizers
    assert [type(r) for r in recognizers] == [LarrySpeechEncoderRecognizerConfig, Wav2Vec2RecognizerConfig]
    assert recognizers[0].model_path == "local/encoder"
    assert VocoderVizConfig().intelligibility.recognizers == []


def test_no_probe_is_built_without_recognizers() -> None:
    assert IntelligibilityProbe.from_config(
        IntelligibilityConfig(), _dataset(), ["a", "b"], WaveformCollator(), torch.device("cpu")
    ) is None


def test_decoder_recognizers_need_a_vocoder() -> None:
    recognizers = IntelligibilityConfig(recognizers=[Wav2Vec2RecognizerConfig()])

    with pytest.raises(ValueError, match="vocoder"):
        SpeechDecoderVizConfig(intelligibility=recognizers)

    SpeechDecoderVizConfig(intelligibility=recognizers, vocoder=PretrainedVocoderConfig(model_path="vocoder"))
