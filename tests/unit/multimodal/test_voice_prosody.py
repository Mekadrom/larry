import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from larry.multimodal.config.model.config_builder import LarryMultimodalModelConfigBuilder
from larry.multimodal.config.model.model_configs import VoiceCodaConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import LarryMultimodalModel, VoiceProsody
from larry.multimodal.processor import processor as P
from larry.multimodal.training.trainers import MODEL_KEYS, PROSODY_KEYS, LarryMultimodalTrainer
from larry.voice.model.prosody import F0Statistics
from tests.unit.multimodal.test_multimodal_generation import (
    ScriptedGenerator,
    _grouped_collator,
    _model,
    _prompt,
    _teacher_forced,
    _voice,
)
from tests.unit.multimodal.test_multimodal_training import _criterion

FRAMES = 4
BANDS = 8


def _f0_item(units: list[int], source: int = 0, frames: int | None = None) -> dict[str, Any]:
    n = len(units) * FRAMES if frames is None else frames
    rng = np.random.default_rng(source)
    f0 = 120.0 * (1 + 0.1 * np.sin(np.arange(n)))
    f0[::3] = 0.0
    return {
        "text": "say this",
        "voice": np.array(units),
        "f0": f0.astype(np.float32),
        "aperiodicity": rng.random((n, BANDS)).astype(np.float32),
        "source": source,
    }


def _forward(model: LarryMultimodalModel, batch: dict[str, Any]) -> Any:
    return model(
        **{key: batch[key] for key in MODEL_KEYS},
        **{key: batch[key] for key in PROSODY_KEYS if key in batch},
        is_synthesis=batch["is_synthesis_mask"],
    )


def _collator(collator: PretrainingMultimodalCollator, tmp_path: Path, k: int) -> PretrainingMultimodalCollator:
    return collator if k == 1 else _grouped_collator(collator, tmp_path, k)


def test_the_collator_pads_f0_and_aperiodicity_only_when_rows_have_them(
        collator: PretrainingMultimodalCollator,
) -> None:
    batch = collator([_f0_item([1, 2, 3]), {"text": "hello there", "source": 1}])

    assert batch["voice_f0"].shape == (2, 3 * FRAMES)
    assert batch["voice_f0_lengths"].tolist() == [3 * FRAMES, 0]
    assert batch["voice_aperiodicity"].shape == (2, 3 * FRAMES, BANDS)
    assert not set(PROSODY_KEYS) & set(collator([{"text": "hello there", "source": 0}]))


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("frames", [None, 9])
def test_each_unit_targets_its_own_frames(
        collator: PretrainingMultimodalCollator, tmp_path: Path, k: int, frames: int | None,
) -> None:
    model = _model(collator, **_voice(k))
    units = [1, 2, 3]
    item = _f0_item(units, frames=frames)
    batch = _collator(collator, tmp_path, k)([item])

    with torch.no_grad():
        out = _forward(model, batch)

    n_frames = len(item["f0"])
    stats = F0Statistics.from_f0(torch.from_numpy(item["f0"])[None], torch.ones(1, n_frames, dtype=torch.bool))
    targets = out.voice_prosody_targets
    assert out.voice_prosody.shape == (*targets.frame_mask.shape, 2 + BANDS)
    per_unit = targets.targets.flatten(0, 1)
    mask = targets.frame_mask.flatten(0, 1)
    for u in range(per_unit.shape[0]):
        for f in range(FRAMES):
            frame = u * FRAMES + f
            expected_valid = u < len(units) and frame < n_frames
            assert bool(mask[u, f]) == expected_valid
            if expected_valid:
                assert float(per_unit[u, f, 0]) == float(stats.voiced[0, frame])
                assert torch.isclose(per_unit[u, f, 1], stats.contour[0, frame])
                assert torch.allclose(per_unit[u, f, 2:], torch.from_numpy(item["aperiodicity"][frame]))

    assert torch.equal(targets.contour_mask, targets.frame_mask & (targets.targets[..., 0] > 0))


def test_prosody_trains_only_on_synthesized_voice_and_reaches_the_head(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator, **_voice(1)).train()
    asr = {**_f0_item([4, 5], source=1), "is_modal_synthesis": False}
    batch = collator([_f0_item([1, 2, 3]), asr, {"text": "hello there", "source": 2}])
    criterion = _criterion()

    out = _forward(model, batch)
    terms = criterion.loss_terms(out, loss_mask=batch["loss_mask"])
    sum(term.weight * term.value for term in terms.values()).backward()

    assert terms["voiced"].counts.tolist() == [3.0 * FRAMES, 0.0, 0.0]
    assert terms["aperiodicity"].counts.tolist() == [3.0 * FRAMES, 0.0, 0.0]
    assert 0 < terms["contour"].counts[0] < 3 * FRAMES
    assert model.voice_coda is not None and model.voice_coda.prosody is not None
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.voice_coda.prosody.parameters())


def test_rows_without_f0_and_models_without_the_head_have_no_prosody_terms(
        collator: PretrainingMultimodalCollator,
) -> None:
    item = {"text": "say this", "voice": np.array([1, 2, 3]), "source": 0}
    with_head = _model(collator, **_voice(1))
    voice = _voice(1)
    voice["voice_coda_config"] = dataclasses.replace(voice["voice_coda_config"], prosody_head=False)
    without_head = _model(collator, **voice)

    with torch.no_grad():
        no_f0 = _forward(with_head, collator([item]))
        no_head = _forward(without_head, collator([_f0_item([1, 2, 3])]))

    for out in (no_f0, no_head):
        assert out.voice_prosody is None
        terms = _criterion().loss_terms(out, loss_mask=collator([item])["loss_mask"])
        assert all(terms[name].counts.sum() == 0 for name in ("voiced", "contour", "aperiodicity"))


@pytest.mark.parametrize(("k", "voice", "units"), [
    (1, [1, 2, 3, 9], [1, 2, 3]),
    (2, [1, 2, 3, 9], [1, 2, 3]),
    (2, [1, 2, 3, 4, 9], [1, 2, 3, 4]),
])
def test_generated_prosody_matches_teacher_forcing(
        collator: PretrainingMultimodalCollator, k: int, voice: list[int], units: list[int],
) -> None:
    model = _model(collator, **_voice(k))
    eos = collator.processor.text_tokenizer.vocab_size() + P.EOS
    generation = ScriptedGenerator(model, [3, eos], voice).run(_prompt(collator, "say this", P.BOV))
    frames = len(units) * FRAMES
    out = _teacher_forced(
        model,
        generation,
        voice_unit_ids=generation.voice_unit_ids[0][None],
        voice_unit_id_lengths=torch.tensor([len(units)]),
        voice_f0=torch.zeros(1, frames),
        voice_f0_lengths=torch.tensor([frames]),
        voice_aperiodicity=torch.zeros(1, frames, BANDS),
    )

    generated = generation.voice_prosody[0]
    forced = VoiceProsody.from_head(out.voice_prosody.flatten(0, 1)[:len(units)].flatten(0, 1))
    assert generated.voiced.shape == (frames,) and generated.aperiodicity.shape == (frames, BANDS)
    assert torch.allclose(generated.voiced, forced.voiced, atol=1e-5)
    assert torch.allclose(generated.contour, forced.contour, atol=1e-5)
    assert torch.allclose(generated.aperiodicity, forced.aperiodicity, atol=1e-5)


@pytest.mark.parametrize(("encoder", "frames"), [({}, 4), ({"frame_merge": 1}, 2), ({"frame_merge": 4}, 8)])
def test_frames_per_unit_follow_the_speech_encoder(
        collator: PretrainingMultimodalCollator, encoder: dict[str, int], frames: int,
) -> None:
    unit_samples = LarryMultimodalTrainer.unit_samples(encoder)
    coda = LarryMultimodalModelConfigBuilder.fit(
        "voice_coda_config", VoiceCodaConfig(vocab_size=10), collator.processor, None, None, unit_samples,
    )

    assert isinstance(coda, VoiceCodaConfig) and coda.prosody_frames_per_unit == frames


def test_units_must_span_whole_prosody_frames(collator: PretrainingMultimodalCollator) -> None:
    with pytest.raises(ValueError):
        LarryMultimodalModelConfigBuilder.fit(
            "voice_coda_config", VoiceCodaConfig(vocab_size=10), collator.processor, None, None, 1000,
        )
