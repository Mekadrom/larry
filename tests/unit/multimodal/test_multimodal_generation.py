import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from larry.multimodal.config.model.model_configs import (
    ImageCodaConfig,
    ImagePreludeConfig,
    LarryMultimodalModelConfig,
    VoiceCodaConfig,
    VoicePreludeConfig,
)
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import (
    LarryMultimodalGenerationConfig,
    LarryMultimodalGenerator,
    LarryMultimodalModel,
)
from larry.multimodal.processor import processor as P
from tests.unit.multimodal.test_multimodal_model import (
    BLOCK,
    D_MODEL,
    N_UNITS,
    SPLIT,
    _config,
    _special,
)


def _vocab(collator: PretrainingMultimodalCollator) -> int:
    return collator.processor.text_tokenizer.vocab_size() + P.EOV + 1


def _voice(k: int, combine: str = "concat") -> dict[str, Any]:
    return {
        "voice_prelude_config": VoicePreludeConfig(
            vocab_size=N_UNITS, fsq_dims=2, n_layers=1, units_per_position=k, group_combine=combine, **BLOCK
        ),
        "voice_coda_config": VoiceCodaConfig(
            vocab_size=N_UNITS + 1, n_layers=1, units_per_position=k, depth_n_layers=1, **BLOCK, **SPLIT
        ),
    }


def _image(with_prelude: bool = True, **coda: Any) -> dict[str, Any]:
    image = {"n_layers": 1, "latent_channels": 3, **BLOCK, **SPLIT}
    return {
        "image_prelude_config": ImagePreludeConfig(**image) if with_prelude else None,
        "image_coda_config": ImageCodaConfig(max_query_grid=8, **{**image, **coda}),
    }


def _model(collator: PretrainingMultimodalCollator, **overrides: Any) -> LarryMultimodalModel:
    torch.manual_seed(0)
    return LarryMultimodalModel(_config(_vocab(collator), **overrides)).eval()


class ScriptedGenerator(LarryMultimodalGenerator):
    def __init__(self, model: LarryMultimodalModel, text: list[int], voice: list[int], **config: Any) -> None:
        super().__init__(model, LarryMultimodalGenerationConfig(record=True, **config))
        self.script = {"text": list(text), "voice": list(voice)}

    def sample(self, logits: torch.Tensor, head: str) -> int:
        return self.script[head].pop(0)


def _prompt(collator: PretrainingMultimodalCollator, text: str, *specials: int) -> torch.Tensor:
    ids = collator.processor.text_tokenizer.encode(text) + [_special(collator, offset) for offset in specials]
    return torch.tensor([ids])


def _teacher_forced(model: LarryMultimodalModel, generation: Any, **media: Any) -> Any:
    with torch.no_grad():
        return model(
            generation.input_ids,
            generation.modality_ids,
            is_synthesis=torch.tensor([True]),
            **media,
        )


def _generated_text_logits(out: Any, prompt_length: int, upto: int | None = None) -> torch.Tensor:
    positions = out.text_route[0].nonzero().flatten()
    keep = positions >= prompt_length - 1
    if upto is not None:
        keep &= positions < upto
    return out.logits[keep]


def test_model_labels_match_the_collators(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator([
        {"text": "say this", "voice": np.array([4, 5, 6]), "source": 0},
        {"text": "a cat", "image": torch.randn(3, 2, 4), "is_modal_synthesis": False, "source": 1},
    ])

    valid = model.key_valid(batch["input_ids"], batch["input_lengths"])
    assert torch.equal(model.modality_ids(batch["input_ids"])[valid], batch["modality_ids"][valid])


def test_greedy_text_generation_sees_what_training_sees(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    prompt = _prompt(collator, "say this")

    generation = model.generate(prompt, max_new_positions=6, temperature=0.0, record=True)
    out = _teacher_forced(model, generation)

    assert generation.input_ids.shape[1] == prompt.shape[1] + 6
    assert torch.allclose(torch.stack(generation.text_logits), _generated_text_logits(out, prompt.shape[1]), atol=1e-4)


@pytest.mark.parametrize(("k", "voice", "units"), [
    (1, [4, 5, 6, N_UNITS], [4, 5, 6]),
    (2, [4, 5, 6, N_UNITS], [4, 5, 6]),
    (2, [4, 5, 6, 7, N_UNITS], [4, 5, 6, 7]),
    (3, [1, 2, 3, 4, N_UNITS], [1, 2, 3, 4]),
])
def test_voice_generation_sees_what_training_sees(
        collator: PretrainingMultimodalCollator, k: int, voice: list[int], units: list[int],
) -> None:
    model = _model(collator, **_voice(k))
    prompt = _prompt(collator, "say this", P.BOV)
    eos = _special(collator, P.EOS)

    generation = ScriptedGenerator(model, [3, eos], voice).run(prompt)
    out = _teacher_forced(
        model,
        generation,
        voice_unit_ids=generation.voice_unit_ids[0][None],
        voice_unit_id_lengths=torch.tensor([len(units)]),
    )

    assert generation.finished
    assert generation.voice_unit_ids[0].tolist() == units
    ids = generation.input_ids[0]
    assert int((ids == _special(collator, P.VPH)).sum()) == math.ceil(len(units) / k)
    assert len(generation.voice_logits) == out.voice_logits.shape[0]
    for group, recorded in enumerate(generation.voice_logits):
        assert torch.allclose(recorded, out.voice_logits[group, :recorded.shape[0]], atol=1e-4)
    assert torch.allclose(torch.stack(generation.text_logits), _generated_text_logits(out, prompt.shape[1]), atol=1e-4)


def test_text_can_open_a_voice_segment_and_continue_after_it(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_voice(2))
    prompt = _prompt(collator, "say this")
    bov, eos = _special(collator, P.BOV), _special(collator, P.EOS)

    generation = ScriptedGenerator(model, [3, bov, 4, eos], [5, 6, 7, N_UNITS]).run(prompt)
    out = _teacher_forced(
        model,
        generation,
        voice_unit_ids=generation.voice_unit_ids[0][None],
        voice_unit_id_lengths=torch.tensor([3]),
    )

    for group, recorded in enumerate(generation.voice_logits):
        assert torch.allclose(recorded, out.voice_logits[group, :recorded.shape[0]], atol=1e-4)
    assert torch.allclose(torch.stack(generation.text_logits), _generated_text_logits(out, prompt.shape[1]), atol=1e-4)


def test_text_to_image_without_a_prelude_matches_training_end_to_end(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_image(with_prelude=False))
    prompt = _prompt(collator, "a cat")
    boi, eos = _special(collator, P.BOI), _special(collator, P.EOS)

    generation = ScriptedGenerator(model, [boi, 4, eos], [], image_size=(4, 2), image_steps=2).run(prompt)
    latents = generation.images[0]
    out = _teacher_forced(model, generation, images=latents[None], image_sizes=torch.tensor([[4, 2]]))

    assert latents.shape == (3, 2, 4)
    assert (generation.input_ids == _special(collator, P.IPH)).sum() == 2
    assert torch.allclose(generation.image_query_states[0], out.image_query_states[0], atol=1e-4)
    assert torch.allclose(torch.stack(generation.text_logits), _generated_text_logits(out, prompt.shape[1]), atol=1e-4)


def test_a_generated_image_is_re_encoded_as_input_when_there_is_a_prelude(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator, **_image())
    prompt = _prompt(collator, "a cat")
    boi, eos = _special(collator, P.BOI), _special(collator, P.EOS)

    generation = ScriptedGenerator(model, [3, boi, 4, eos], [], image_size=(4, 2), image_steps=2).run(prompt)
    latents = generation.images[0]
    out = _teacher_forced(model, generation, images=latents[None], image_sizes=torch.tensor([[4, 2]]))

    first_slot = int((generation.input_ids[0] == _special(collator, P.IPH)).nonzero()[0])
    before_image = _generated_text_logits(out, prompt.shape[1], upto=first_slot)
    assert torch.allclose(torch.stack(generation.text_logits[:2]), before_image, atol=1e-4)
    assert torch.allclose(generation.image_query_states[0], out.image_query_states[0], atol=1e-4)

    with torch.no_grad():
        captioned = model(
            generation.input_ids, generation.modality_ids, images=latents[None], image_sizes=torch.tensor([[4, 2]])
        )
    after_image = captioned.logits[captioned.text_route[0].nonzero().flatten() > first_slot]
    assert torch.allclose(torch.stack(generation.text_logits[2:]), after_image, atol=1e-4)


def test_heads_the_model_lacks_cannot_be_opened(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, voice_prelude_config=None, voice_coda_config=None)
    generator = LarryMultimodalGenerator(model, LarryMultimodalGenerationConfig())

    with torch.no_grad():
        logits = generator.next_text_logits(generator.step(_prompt(collator, "say")[0].tolist(), False))

    assert torch.isinf(logits[[_special(collator, o) for o in (P.BOV, P.BOI, P.VPH, P.IPH, P.EOV, P.EOI)]]).all()
    assert torch.isfinite(logits[_special(collator, P.EOS)])


def test_generation_rejects_a_prompt_that_already_ended(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)

    with pytest.raises(ValueError, match="<eos>"):
        model.generate(_prompt(collator, "say", P.EOS))


def test_sampling_is_seeded(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    prompt = _prompt(collator, "say this")

    def run() -> list[int]:
        return model.generate(
            prompt, max_new_positions=5, top_k=5, generator=torch.Generator().manual_seed(3)
        ).input_ids[0].tolist()

    assert run() == run()


def _grouped_collator(collator: PretrainingMultimodalCollator, root: Path, k: int) -> PretrainingMultimodalCollator:
    from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
    from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor

    config = dataclasses.replace(collator.processor.config, voice_units_per_position=k)
    LarryMultimodalPretrainProcessor(config).save_pretrained(root)
    return PretrainingMultimodalCollator(PretrainingMultimodalCollatorConfig(processor_model_path=str(root)))


def test_collated_groups_stretch_to_one_slot_per_k_units(
        collator: PretrainingMultimodalCollator, tmp_path: Path,
) -> None:
    grouped = _grouped_collator(collator, tmp_path, 2)
    batch = grouped([
        {"text": "say", "voice": np.array([1, 2, 3]), "source": 0},
        {"text": "say", "voice": np.array([1, 2, 3, 4]), "source": 1},
    ])

    assert (batch["input_ids"] == _special(collator, P.VPH)).sum(dim=1).tolist() == [2, 2]


@pytest.mark.parametrize(("n_units", "routed", "targets"), [
    (3, 2, [[1, 2], [3, N_UNITS]]),
    (4, 3, [[1, 2], [3, 4], [N_UNITS, -100]]),
])
def test_grouped_targets_end_with_eov_and_skip_the_inserted_one(
        collator: PretrainingMultimodalCollator, tmp_path: Path, n_units: int, routed: int, targets: list[list[int]],
) -> None:
    model = _model(collator, **_voice(2))
    grouped = _grouped_collator(collator, tmp_path, 2)
    batch = grouped([{"text": "say", "voice": np.arange(1, n_units + 1), "source": 0}])

    with torch.no_grad():
        out = model(
            batch["input_ids"], batch["modality_ids"], batch["input_lengths"], batch["is_synthesis_mask"],
            voice_unit_ids=batch["voice_unit_ids"], voice_unit_id_lengths=batch["voice_unit_id_lengths"],
        )

    assert int(out.voice_route.sum()) == routed
    assert out.voice_targets.tolist() == targets
    assert out.voice_logits.shape == (routed, 2, N_UNITS + 1)


@pytest.mark.parametrize("combine", ["concat", "sum"])
def test_grouped_prelude_gives_one_vector_per_group(collator: PretrainingMultimodalCollator, combine: str) -> None:
    model = _model(collator, **_voice(3, combine))
    groups = model.group_units(torch.tensor([[1, 2, 3, 4, 5, 0, 0]]), torch.tensor([5]))

    with torch.no_grad():
        out = model.voice_prelude(groups.unit_ids, groups.unit_valid)

    assert groups.unit_ids.shape == (1, 3, 3)
    assert groups.unit_valid.tolist() == [[[True] * 3, [True, True, False], [False] * 3]]
    assert out.logits.shape == (1, 3, D_MODEL)


def test_a_group_only_depends_on_its_own_units(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_voice(2))
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    synthesis = torch.tensor([True])

    with torch.no_grad():
        mask = model.voice_prelude_mask(valid[..., 0], synthesis)
        base = model.voice_prelude(torch.tensor([[[1, 2], [3, 4]]]), valid, mask).logits
        moved = model.voice_prelude(torch.tensor([[[1, 2], [3, 8]]]), valid, mask).logits

    assert torch.allclose(base[:, 0], moved[:, 0], atol=1e-6)
    assert not torch.allclose(base[:, 1], moved[:, 1])


def test_weights_start_small_and_zero_initialized_pieces_stay_zero(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_voice(2), **_image())

    assert model.text_prelude.wte.weight.std().item() == pytest.approx(0.02, rel=0.15)
    assert model.voice_prelude.ute.weight.std().item() == pytest.approx(0.02, rel=0.3)
    assert not model.image_coda.out.weight.any() and not model.image_coda.out.bias.any()
    assert not model.image_coda.blocks[0].attn_norm.modulation[1].weight.any()
    assert not model.image_coda.final_norm.scale_shift[1].weight.any()
    trunk_o_proj = model.recurrent_block.blocks[0].self_attn.o_proj.weight.std().item()
    coda_o_proj = model.text_coda.blocks[0].self_attn.o_proj.weight.std().item()
    assert trunk_o_proj < coda_o_proj


def test_init_keeps_the_trunks_identity_projection(collator: PretrainingMultimodalCollator) -> None:
    from larry.common.config.model.model_configs import LarryRecurrentTrunkConfig

    trunk = LarryRecurrentTrunkConfig(
        n_layers=1, thought_state_init="zero", mean_thinking_steps=2, backprop_depth=1, exit_criteria="none",
        thought_state_injection_type="concat", **BLOCK, **SPLIT,
    )
    model = _model(collator, recurrent_trunk_config=trunk)

    identity = torch.eye(D_MODEL)
    assert torch.equal(model.recurrent_block.iteration_projection.weight, torch.cat([identity, identity], dim=1))


def _capture_query_states(model: LarryMultimodalModel) -> dict[str, torch.Tensor]:
    seen: dict[str, torch.Tensor] = {}
    velocity = model.image_coda.velocity

    def capture(xt: torch.Tensor, t: torch.Tensor, query_states: torch.Tensor, *args: Any) -> torch.Tensor:
        seen["query_states"] = query_states
        return velocity(xt, t, query_states, *args)

    model.image_coda.velocity = capture
    return seen


@pytest.mark.parametrize(("rate", "training", "dropped"), [(1.0, True, True), (1.0, False, False), (0.0, True, False)])
def test_cfg_dropout_replaces_an_images_query_states_only_while_training(
        collator: PretrainingMultimodalCollator, rate: float, training: bool, dropped: bool,
) -> None:
    model = _model(collator, **_image(with_prelude=False, cfg_dropout_rate=rate)).train(training)
    seen = _capture_query_states(model)
    boi, iph, eoi = (_special(collator, o) for o in (P.BOI, P.IPH, P.EOI))

    model(
        torch.tensor([[3, boi, iph, iph, eoi]]), torch.tensor([[0, 0, 1, 1, 0]]), is_synthesis=torch.tensor([True]),
        images=torch.randn(1, 3, 2, 4), image_sizes=torch.tensor([[4, 2]]),
    )

    null = model.image_coda.null_query_state.expand_as(seen["query_states"])
    assert torch.equal(seen["query_states"], null) is dropped


def test_cfg_sampling_extrapolates_from_the_unconditional_velocity(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_image(with_prelude=False))
    coda = model.image_coda
    patch_dim = coda.out.out_features
    coda.velocity = lambda xt, t, query_states, *_: query_states[..., :patch_dim]
    states = torch.randn(1, 2, D_MODEL)
    mask = torch.ones(1, 2, dtype=torch.bool)

    def sample(scale: float) -> torch.Tensor:
        return coda.sample(states, mask, 1, 2, steps=4, generator=torch.Generator().manual_seed(0), cfg_scale=scale)

    difference = coda.patchify(sample(3.0) - sample(1.0))
    null = coda.null_query_state[:patch_dim]
    assert torch.allclose(difference, 2 * (states[..., :patch_dim] - null), atol=1e-5)


def test_standalone_task_models_build_and_run(collator: PretrainingMultimodalCollator) -> None:
    vocab = _vocab(collator)
    asr = LarryMultimodalModel(_config(vocab, voice_coda_config=None, include_tasks=["voice_text"])).eval()
    tts = LarryMultimodalModel(_config(vocab, include_tasks=["text_voice"])).eval()
    t2i = LarryMultimodalModel(_config(
        vocab, voice_prelude_config=None, voice_coda_config=None, include_modes=["text", "image"],
        include_tasks=["text_image"], **_image(with_prelude=False),
    )).eval()
    batch = collator([{"text": "say this", "voice": np.array([4, 5, 6]), "is_modal_synthesis": False, "source": 0}])

    with torch.no_grad():
        asr_out = asr(
            batch["input_ids"], batch["modality_ids"], batch["input_lengths"], batch["is_synthesis_mask"],
            voice_unit_ids=batch["voice_unit_ids"], voice_unit_id_lengths=batch["voice_unit_id_lengths"],
        )

    assert asr_out.voice_logits is None and asr_out.logits.shape[0] > 0
    assert tts.voice_coda is not None and t2i.image_prelude is None and t2i.voice_prelude is None
    assert isinstance(t2i.config, LarryMultimodalModelConfig)


def _inference_collator(collator: PretrainingMultimodalCollator) -> Any:
    from larry.multimodal.config.data.dataloading.dataloading_configs import InferenceMultimodalCollatorConfig
    from larry.multimodal.data.dataloading.dataloading import InferenceMultimodalCollator

    return InferenceMultimodalCollator(
        InferenceMultimodalCollatorConfig(processor_model_path=collator.config.processor_model_path)
    )


VOICE = "<|voice_placeholder|>"
IMAGE = "<|image_placeholder|>"


@pytest.mark.parametrize(("pretraining", "inference"), [
    ({"text": "say this", "voice": np.array([4, 5, 6]), "is_modal_synthesis": False},
     {"text": f"{VOICE}say this", "voices": [np.array([4, 5, 6])]}),
    ({"text": "a cat", "image": np.ones((3, 2, 4), dtype=np.float32), "is_modal_synthesis": False},
     {"text": f"{IMAGE}a cat", "images": [np.ones((3, 2, 4), dtype=np.float32)]}),
    ({"text": "hello there"}, {"text": "hello there"}),
])
def test_an_inference_prompt_collates_like_its_training_row_without_eos(
        collator: PretrainingMultimodalCollator, pretraining: dict[str, Any], inference: dict[str, Any],
) -> None:
    trained = collator([{**pretraining, "source": 0}])
    prompted = _inference_collator(collator)([inference])

    assert torch.equal(prompted["input_ids"], trained["input_ids"][:, :-1])
    assert torch.equal(prompted["modality_ids"], trained["modality_ids"][:, :-1])
    for key in ("voice_unit_ids", "voice_unit_id_lengths", "images", "image_sizes"):
        assert torch.equal(prompted[key], trained[key]), key
    assert "loss_mask" not in prompted and not prompted["is_synthesis_mask"].any()


def test_an_inference_prompt_can_ask_for_voice_and_generate_from_it(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    inference = _inference_collator(collator)
    batch = inference([{"text": "say this", "open_modality": "voice"}])
    eos = _special(collator, P.EOS)

    generation = ScriptedGenerator(model, [eos], [4, N_UNITS]).run(**inference.generate_kwargs(batch, 0))

    assert batch["input_ids"][0, -1] == _special(collator, P.BOV)
    assert generation.voice_unit_ids[0].tolist() == [4]
    assert generation.finished


def test_generate_kwargs_strip_a_rows_padding(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, **_image())
    inference = _inference_collator(collator)
    batch = inference([
        {"text": f"{VOICE}hello there friend", "voices": [np.array([1, 2, 3, 4])]},
        {"text": f"{IMAGE}a cat", "images": [np.ones((3, 2, 4), dtype=np.float32)]},
    ])

    voice_row = inference.generate_kwargs(batch, 0)
    image_row = inference.generate_kwargs(batch, 1)
    eos = _special(collator, P.EOS)

    assert voice_row["input_ids"].shape[1] == int(batch["input_lengths"][0])
    assert voice_row["images"] is None and voice_row["voice_unit_ids"].tolist() == [[1, 2, 3, 4]]
    assert image_row["voice_unit_ids"] is None and image_row["images"].shape == (1, 3, 2, 4)
    assert ScriptedGenerator(model, [eos], []).run(**image_row).finished
    assert ScriptedGenerator(model, [eos], []).run(**voice_row).finished
