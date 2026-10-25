import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from larry.multimodal.config.criterion.criterion_configs import LarryMultimodalCriterionConfig
from larry.multimodal.config.model.config_builder import LarryMultimodalModelConfigBuilder
from larry.multimodal.criteria.multimodal_criterion import LarryMultimodalCriterion
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import PROCESSOR_DIR, LarryMultimodalModel
from larry.multimodal.processor import processor as P
from larry.multimodal.processor.processor import LarryMultimodalProcessor
from tests.unit.multimodal.test_multimodal_generation import _image, _model, _vocab, _voice
from tests.unit.multimodal.test_multimodal_model import _config

MODEL_KEYS = (
    "input_ids", "modality_ids", "input_lengths", "voice_unit_ids", "voice_unit_id_lengths", "images", "image_sizes",
)


def _forward(model: LarryMultimodalModel, batch: dict[str, Any]) -> Any:
    return model(**{key: batch[key] for key in MODEL_KEYS}, is_synthesis=batch["is_synthesis_mask"])


def _criterion(**overrides: Any) -> LarryMultimodalCriterion:
    return LarryMultimodalCriterion(LarryMultimodalCriterionConfig(**overrides))


def test_the_loss_mask_is_shifted_onto_the_predicting_positions() -> None:
    loss_mask = torch.tensor([[False, True, True, False, True]])

    assert LarryMultimodalCriterion.next_in_loss(loss_mask).tolist() == [[True, True, False, True, False]]


def test_synthesis_trains_on_voice_and_the_delimiter_text_targets_only(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator)
    batch = collator([{"text": "say this", "voice": np.array([4, 5, 6]), "source": 0}])
    criterion = _criterion()

    with torch.no_grad():
        out = _forward(model, batch)
        text = criterion.text_loss(out, criterion.next_in_loss(batch["loss_mask"]))
        voice = criterion.voice_loss(out, criterion.next_in_loss(batch["loss_mask"]))

    targets = out.text_targets
    trained = (targets == P.BOV + collator.processor.text_tokenizer.vocab_size()) | (
        targets == P.EOS + collator.processor.text_tokenizer.vocab_size()
    )
    assert torch.allclose(text, F.cross_entropy(out.logits[trained], targets[trained]))
    assert torch.allclose(voice, F.cross_entropy(out.voice_logits.flatten(0, 1), out.voice_targets.flatten()))


def test_captioning_trains_on_text_and_never_on_the_input_voice(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator([{"text": "say this", "voice": np.array([4, 5, 6]), "is_modal_synthesis": False, "source": 0}])
    criterion = _criterion()

    with torch.no_grad():
        out = _forward(model, batch)

    in_loss = criterion.next_in_loss(batch["loss_mask"])
    assert criterion.voice_loss(out, in_loss) is None
    assert in_loss[out.text_route].sum() == len(collator.processor.text_tokenizer.encode("say this")) + 1


def test_heads_are_weighted_and_logged_per_split(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_image())
    batch = collator([
        {"text": "say this", "voice": np.array([4, 5, 6]), "source": 0},
        {"text": "a cat", "image": torch.randn(3, 2, 4), "source": 1},
    ])
    criterion = _criterion(text_weight=0.5, voice_weight=2.0, image_weight=3.0)

    with torch.no_grad():
        out = _forward(model, batch)
        total = criterion(out, **batch)
    in_loss = criterion.next_in_loss(batch["loss_mask"])
    expected = (
        0.5 * criterion.text_loss(out, in_loss)
        + 2.0 * criterion.voice_loss(out, in_loss)
        + 3.0 * criterion.image_loss(out)
    )

    assert torch.allclose(total, expected)
    assert set(out.metrics) == {"eval/loss/text", "eval/loss/voice", "eval/loss/image"}


def test_a_training_step_reaches_every_head_it_uses(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_image(with_prelude=False)).train()
    batch = collator([
        {"text": "say this", "voice": np.array([4, 5, 6]), "source": 0},
        {"text": "a cat", "image": torch.randn(3, 2, 4), "source": 1},
    ])

    out = _forward(model, batch)
    _criterion()(out, **batch).backward()

    assert set(out.metrics) == {"train/loss/text", "train/loss/voice", "train/loss/image"}
    assert model.voice_coda.eov.grad.abs().sum() > 0
    assert model.image_coda.out.weight.grad.abs().sum() > 0
    assert model.text_prelude.wte.weight.grad.abs().sum() > 0


def test_the_criterion_needs_the_loss_mask(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator([{"text": "say this", "source": 0}])

    with torch.no_grad(), pytest.raises(ValueError, match="loss_mask"):
        _criterion()(_forward(model, batch))


def _template(collator: PretrainingMultimodalCollator, tasks: list[str], modes: list[str]) -> Any:
    return _config(_vocab(collator), include_tasks=tasks, include_modes=modes, **_voice(1), **_image())


@pytest.mark.parametrize(("tasks", "modes", "kept"), [
    (["voice_text"], ["text", "voice"], {"voice_prelude_config"}),
    (["text_voice"], ["text", "voice"], {"voice_prelude_config", "voice_coda_config"}),
    (["text_image"], ["text", "image"], {"image_coda_config"}),
    (["image_text"], ["text", "image"], {"image_prelude_config"}),
    (["text_only"], ["text"], set()),
    (["voice_text", "text_image"], ["text", "voice", "image"], {"voice_prelude_config", "image_coda_config"}),
])
def test_the_builder_keeps_only_the_modules_the_tasks_need(
        collator: PretrainingMultimodalCollator, tasks: list[str], modes: list[str], kept: set[str],
) -> None:
    built = LarryMultimodalModelConfigBuilder(_template(collator, tasks, modes)).build(collator.processor)

    media = {"image_prelude_config", "image_coda_config", "voice_prelude_config", "voice_coda_config"}
    assert {name for name in media if getattr(built, name) is not None} == kept
    model = LarryMultimodalModel(built)
    assert (model.voice_coda is not None) == ("voice_coda_config" in kept)
    assert (model.image_prelude is not None) == ("image_prelude_config" in kept)


def test_the_builder_takes_sizes_from_the_processor_and_encoders(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    processor = LarryMultimodalProcessor(
        dataclasses.replace(collator.processor.config, image_patch_size=4, voice_units_per_position=2)
    )
    template = _template(collator, ["text_voice", "text_image"], ["text", "voice", "image"])

    built = LarryMultimodalModelConfigBuilder(template).build(processor, fsq_dims=3, fsq_levels=3, latent_channels=8)

    assert built.text_prelude_config.vocab_size == built.text_coda_config.vocab_size == _vocab(collator)
    assert (built.voice_prelude_config.vocab_size, built.voice_coda_config.vocab_size) == (27, 28)
    assert built.voice_prelude_config.fsq_dims == 3
    assert built.voice_units_per_position == 2
    assert (built.image_coda_config.patch_size, built.image_coda_config.latent_channels) == (4, 8)
    LarryMultimodalModel(built).check_processor(processor)


@pytest.mark.parametrize(("tasks", "modes", "match"), [
    (["voice_text"], ["text"], "missing from include_modes"),
    (["voice_text"], ["text", "voice", "image"], "no task uses"),
    (["speech_to_speech"], ["text", "voice"], "unknown include_tasks"),
])
def test_the_builder_rejects_inconsistent_tasks_and_modes(
        collator: PretrainingMultimodalCollator, tasks: list[str], modes: list[str], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        LarryMultimodalModelConfigBuilder(_template(collator, tasks, modes)).build(collator.processor)


def test_the_builder_rejects_a_template_without_a_needed_module(collator: PretrainingMultimodalCollator) -> None:
    template = _config(
        _vocab(collator), include_tasks=["text_image"], include_modes=["text", "image"]
    )

    with pytest.raises(ValueError, match="image_coda_config"):
        LarryMultimodalModelConfigBuilder(template).build(collator.processor)


def test_the_model_round_trips_through_the_hub_format(collator: PretrainingMultimodalCollator, tmp_path: Path) -> None:
    model = _model(collator, **_voice(1), **_image())

    model.save_with_processor(tmp_path, collator.processor)
    loaded = LarryMultimodalModel.from_pretrained(tmp_path)
    processor = LarryMultimodalProcessor.from_pretrained(tmp_path / PROCESSOR_DIR)

    assert loaded.config == model.config
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[key], restored[key]) for key in original)
    assert processor.text_tokenizer.vocab_size() == collator.processor.text_tokenizer.vocab_size()
    loaded.check_processor(processor)


def test_saving_with_a_mismatched_processor_is_refused(collator: PretrainingMultimodalCollator, tmp_path: Path) -> None:
    model = _model(collator)
    processor = LarryMultimodalProcessor(dataclasses.replace(collator.processor.config, voice_units_per_position=3))

    with pytest.raises(ValueError, match="voice_units_per_position"):
        model.save_with_processor(tmp_path, processor)
    assert not any(tmp_path.iterdir())
