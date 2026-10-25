from pathlib import Path

import numpy as np
import pytest
import torch

from larry.multimodal.config.training.trainer_configs import MultimodalVizConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import LarryMultimodalGenerator
from larry.multimodal.training.multimodal_viz_callbacks import (
    ImageSynthesisVizCallback,
    MultimodalVizCallbacks,
    TextContinuationVizCallback,
    VoiceSynthesisVizCallback,
    distinct_n,
    latent_preview,
    longest_run_fraction,
    repeated_fraction,
    token_f1,
)
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.vocoder import LarryVocoder
from tests.unit.multimodal.test_multimodal_trainer import ALL_TASKS, _trainer
from tests.unit.voice.model.test_speech_decoder import tiny_config
from tests.unit.voice.model.test_vocoder import TINY

TASK_METRICS = {
    "text_only": {"eos_rate", "new_tokens", "repeated_4gram", "reference_prefix_match", "distinct_2"},
    "voice_text": {"eos_rate", "length_ratio", "wer", "cer"},
    "image_text": {"token_f1", "eos_rate", "length_ratio", "distinct_2"},
    "text_voice": {"eov_rate", "length_ratio", "unit_error_rate", "unique_units", "repeated_4gram", "longest_run"},
    "text_image": {
        "channel_mean_error", "channel_std_ratio_error", "relative_mse", "seed_diversity", "round_trip_f1",
    },
}


def test_token_f1_counts_shared_words_case_and_punctuation_free() -> None:
    assert token_f1("A cat, on a mat!", "a cat on a mat") == 1.0
    assert token_f1("a dog", "a cat") == pytest.approx(0.5)
    assert token_f1("", "a cat") == 0.0


def test_repetition_metrics_separate_varied_from_looping_sequences() -> None:
    assert repeated_fraction([1, 2, 3, 4, 5, 6], 2) == 0.0
    assert repeated_fraction([1, 2, 1, 2, 1, 2], 2) == pytest.approx(1 - 2 / 5)
    assert longest_run_fraction([3, 3, 3, 1]) == 0.75
    assert distinct_n([[1, 2, 3], [1, 2, 3]], 2) == 0.5


def test_latent_previews_are_rgb_images_in_unit_range() -> None:
    preview = latent_preview(torch.randn(16, 4, 6) * 5)

    assert preview.shape == (4, 6, 3)
    assert preview.min() == 0.0 and preview.max() == 1.0
    assert latent_preview(torch.randn(1, 2, 2)).shape == (2, 2, 3)


def test_each_task_in_the_eval_mixture_gets_its_callback_and_logs_its_metrics(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    trainer = _trainer(collator, tmp_path, ALL_TASKS)
    callbacks = MultimodalVizCallbacks(trainer, 1, MultimodalVizConfig(examples_per_source=1, max_new_tokens=6))

    callbacks.viz(1)

    assert {callback.task for callback in callbacks.callbacks} == set(ALL_TASKS)
    for task, names in TASK_METRICS.items():
        logged = {tag.split("/")[-1] for tag in trainer.metrics.scalars if tag.startswith(f"viz/{task}/")}
        assert names <= logged, (task, names - logged)
    assert all(np.isfinite(value) for value in trainer.metrics.scalars.values())
    assert trainer.accelerator.unwrap_model(trainer.model).training


def test_only_tasks_with_eval_sources_are_visualized(collator: PretrainingMultimodalCollator, tmp_path: Path) -> None:
    trainer = _trainer(collator, tmp_path, ["text_voice", "text_only"])

    callbacks = MultimodalVizCallbacks(trainer, 1, MultimodalVizConfig())

    assert [type(c) for c in callbacks.callbacks] == [TextContinuationVizCallback, VoiceSynthesisVizCallback]


def test_text_continuation_shows_prompt_generation_and_reference(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    trainer = _trainer(collator, tmp_path, ["text_only"])

    TextContinuationVizCallback(trainer, 1, MultimodalVizConfig(examples_per_source=2, max_new_tokens=4)).viz(1)

    texts = [text for tag, text, _ in trainer.metrics.text if tag.endswith("/continuation")]
    assert len(texts) == 2
    assert all("**prompt:**" in t and "**generated:**" in t and "**reference:**" in t for t in texts)
    assert trainer.metrics.scalars["viz/text_only/new_tokens"] <= 4


def test_voice_synthesis_is_heard_back_and_vocalized_when_decoders_are_configured(
        collator: PretrainingMultimodalCollator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    SpeechDecoderModel(tiny_config()).save_pretrained(tmp_path / "decoder")
    LarryVocoder(TINY).save_pretrained(tmp_path / "vocoder")
    trainer = _trainer(collator, tmp_path, ["text_voice", "voice_text"])
    config = MultimodalVizConfig(
        examples_per_source=1, vocoder_steps=2,
        speech_decoder_model_path=str(tmp_path / "decoder"), vocoder_model_path=str(tmp_path / "vocoder"),
    )
    callback = VoiceSynthesisVizCallback(trainer, 1, config)
    # always emit a couple of units and then <eov>, so there is something to vocalize and hear back
    eov = callback.trainer.accelerator.unwrap_model(trainer.model).config.voice_coda_config.vocab_size - 1
    script = iter([3, 5, eov] * 4)
    sample = LarryMultimodalGenerator.sample
    monkeypatch.setattr(
        LarryMultimodalGenerator, "sample",
        lambda self, logits, head: next(script) if head == "voice" else sample(self, logits, head),
    )

    callback.viz(1)
    callback.viz(2)

    audio_tags = [tag for tag, _ in trainer.metrics.audio]
    assert sum(tag.endswith("/generated") for tag in audio_tags) == 2
    assert sum(tag.endswith("/reference") for tag in audio_tags) == 1
    assert any(tag.endswith("/mels") for tag, _ in trainer.metrics.figures)
    assert "viz/text_voice/round_trip_wer" in trainer.metrics.scalars
    assert trainer.metrics.scalars["viz/text_voice/eov_rate"] == 1.0
    assert any("**heard back:**" in text for _, text, _ in trainer.metrics.text)


def test_image_synthesis_samples_differ_by_seed(collator: PretrainingMultimodalCollator, tmp_path: Path) -> None:
    trainer = _trainer(collator, tmp_path, ["text_image"])

    ImageSynthesisVizCallback(trainer, 1, MultimodalVizConfig(examples_per_source=1, image_steps=2)).viz(1)

    assert trainer.metrics.scalars["viz/text_image/seed_diversity"] > 0
    assert "viz/text_image/round_trip_f1" not in trainer.metrics.scalars
