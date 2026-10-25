from typing import Any

import numpy as np
import pytest
import torch

from larry.common.config.model.model_configs import AttentionModuleConfig, LarryRecurrentTrunkConfig
from larry.common.model.attention import LarryAttention
from larry.common.model.kv_cache import KVCache
from larry.multimodal.config.model.model_configs import (
    ImagePreludeConfig,
    LarryMultimodalModelConfig,
    TextModuleConfig,
    VoiceCodaConfig,
    VoiceModuleConfig,
    VoicePreludeConfig,
)
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import LarryMultimodalModel
from larry.multimodal.processor import processor as P

D_MODEL = 32
N_UNITS = 9
BLOCK = {
    "d_model": D_MODEL,
    "n_heads": 2,
    "n_query_groups": 2,
    "rotary_embedding_dim": 16,
    "max_seq_len": 128,
    "attention_dropout_p": 0.0,
    "proj_dropout_p": 0.0,
    "mlp_dropout_p": 0.0,
}
SPLIT = {"use_split_rope": True, "split_rope_local_axes": 2}


def _config(text_vocab_size: int, **overrides: Any) -> LarryMultimodalModelConfig:
    text = TextModuleConfig(vocab_size=text_vocab_size, n_layers=1, **BLOCK, **SPLIT)
    return LarryMultimodalModelConfig(**{
        "d_model": D_MODEL,
        "text_prelude_config": text,
        "text_coda_config": TextModuleConfig(vocab_size=text_vocab_size, n_layers=1, **BLOCK, **SPLIT),
        "voice_prelude_config": VoicePreludeConfig(vocab_size=N_UNITS, fsq_dims=2, n_layers=1, **BLOCK),
        "voice_coda_config": VoiceCodaConfig(vocab_size=N_UNITS + 1, n_layers=1, **BLOCK, **SPLIT),
        "recurrent_trunk_config": LarryRecurrentTrunkConfig(
            n_layers=1, thought_state_init="zero", mean_thinking_steps=2, backprop_depth=1, exit_criteria="none",
            **BLOCK, **SPLIT,
        ),
        "include_modes": ["text", "voice"],
        "include_tasks": ["text_only", "text_voice", "voice_text"],
        **overrides,
    })


def _model(collator: PretrainingMultimodalCollator) -> LarryMultimodalModel:
    torch.manual_seed(0)
    return LarryMultimodalModel(_config(collator.processor.text_tokenizer.vocab_size() + P.EOV + 1)).eval()


def _items() -> list[dict[str, Any]]:
    return [
        {"text": "say this", "voice": np.array([4, 5, 6]), "source": 0},
        {"text": "hello there friend", "voice": np.array([1, 2]), "is_modal_synthesis": False, "source": 1},
        {"text": "a cat", "source": 2},
    ]


def _forward(model: LarryMultimodalModel, batch: dict[str, Any]) -> Any:
    keys = ("input_ids", "modality_ids", "input_lengths", "voice_unit_ids", "voice_unit_id_lengths")
    return model(**{key: batch[key] for key in keys}, is_synthesis=batch["is_synthesis_mask"])


def _special(collator: PretrainingMultimodalCollator, offset: int) -> int:
    return collator.processor.text_tokenizer.vocab_size() + offset


def test_the_lm_head_is_tied_to_the_text_embedding(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)

    assert model.text_coda.lm_head.weight is model.text_prelude.wte.weight
    assert model.special_base == collator.processor.text_tokenizer.vocab_size()


def test_logits_and_targets_line_up_for_each_routed_head(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator(_items())

    with torch.no_grad():
        out = _forward(model, batch)

    assert out.logits.shape == (int(out.text_route.sum()), model.config.text_prelude_config.vocab_size)
    assert out.text_targets.shape == (int(out.text_route.sum()),)
    assert out.voice_logits.shape == (int(out.voice_route.sum()), 1, N_UNITS + 1)
    assert out.voice_targets.shape == (int(out.voice_route.sum()), 1)
    assert not (out.text_route & out.voice_route).any()


def test_routes_follow_the_target_token(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator(_items())

    with torch.no_grad():
        out = _forward(model, batch)

    ids = batch["input_ids"][0]
    bov = int((ids == _special(collator, P.BOV)).nonzero())
    eov = int((ids == _special(collator, P.EOV)).nonzero())
    length = int(batch["input_lengths"][0])
    assert out.text_route[0, bov - 1]
    assert out.voice_route[0, bov:eov].all()
    assert out.text_route[0, eov:length - 1].all()
    assert not (out.text_route[0, length - 1:] | out.voice_route[0, length - 1:]).any()


def test_voice_targets_are_the_generated_units_then_the_eov_class(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator(_items())

    with torch.no_grad():
        out = _forward(model, batch)

    assert out.voice_targets.flatten().tolist() == [4, 5, 6, N_UNITS]
    assert not out.voice_route[1].any()


def test_text_targets_are_the_next_input_ids(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator(_items())

    with torch.no_grad():
        out = _forward(model, batch)

    row = 2
    length = int(batch["input_lengths"][row])
    expected = batch["input_ids"][row, 1:length].tolist()
    assert out.text_targets[-len(expected):].tolist() == expected


def test_padding_and_other_rows_do_not_change_a_rows_outputs(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    alone = collator(_items()[:1])
    batched = collator(_items())

    with torch.no_grad():
        single = _forward(model, alone)
        together = _forward(model, batched)

    n_text = int(single.text_route.sum())
    n_voice = int(single.voice_route.sum())
    assert torch.allclose(together.logits[:n_text], single.logits, atol=1e-5)
    assert torch.allclose(together.voice_logits[:n_voice], single.voice_logits, atol=1e-5)


def test_unit_embeddings_reach_the_trunk(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator).train()
    out = _forward(model, collator(_items()))

    out.logits.logsumexp(-1).mean().backward()

    assert model.voice_prelude.ute.weight.grad is not None
    assert model.voice_prelude.ute.weight.grad[[1, 2, 4, 5, 6]].abs().sum(-1).gt(0).all()
    assert not model.voice_prelude.ute.weight.grad[[0, 3, 7, 8]].any()
    assert model.voice_prelude.code_proj.weight.grad.abs().sum() > 0


def test_text_only_batches_skip_the_voice_modules(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    batch = collator([{"text": "hello there", "source": 0}])

    with torch.no_grad():
        out = _forward(model, batch)

    assert out.voice_logits is None and out.voice_targets is None
    assert int(out.text_route.sum()) == int(batch["input_lengths"][0]) - 1


def test_the_attention_mask_is_causal_bidirectional_within_an_image_and_hides_padding(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator)
    boi, iph, eoi = (_special(collator, offset) for offset in (P.BOI, P.IPH, P.EOI))
    input_ids = torch.tensor([[3, boi, iph, iph, iph, eoi, boi, iph, iph, eoi, 4, 0]])

    valid = model.key_valid(input_ids, torch.tensor([11]))
    mask = model.attention_mask(input_ids, valid, torch.tensor([False]))[0, 0]

    assert mask.shape == (12, 12)
    assert mask[2:5, 2:5].all()
    assert mask[7:9, 7:9].all()
    assert not mask[2:5, 7:9].any()
    assert not mask[0, 1:].any()
    assert mask[10, :11].all() and not mask[:, 11].any()


def test_position_ids_index_voice_units_and_image_rows_and_columns(collator: PretrainingMultimodalCollator) -> None:
    torch.manual_seed(0)
    vocab = collator.processor.text_tokenizer.vocab_size() + P.EOV + 1
    image_prelude = ImagePreludeConfig(n_layers=1, latent_channels=3, **BLOCK)
    model = LarryMultimodalModel(_config(vocab, image_prelude_config=image_prelude))
    bov, vph, eov, boi, iph, eoi = (_special(collator, o) for o in (P.BOV, P.VPH, P.EOV, P.BOI, P.IPH, P.EOI))
    input_ids = torch.tensor([
        [3, bov, vph, vph, vph, eov, 4, 0, 0, 0],
        [boi, iph, iph, iph, iph, iph, iph, eoi, 3, 4],
    ])

    positions = model.position_ids(input_ids, torch.tensor([[0, 0], [6, 4]]))

    assert positions[0, :, 0].tolist() == list(range(10))
    assert positions[0, :, 1].tolist() == [0, 0, 0, 1, 2, 0, 0, 0, 0, 0]
    assert positions[1, :, 1].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert positions[1, :, 2].tolist() == [0, 0, 1, 2, 0, 1, 2, 0, 0, 0]
    assert not positions[0, :, 2].any()


def test_vocab_sizes_are_validated(collator: PretrainingMultimodalCollator) -> None:
    with pytest.raises(ValueError, match="tied"):
        _config(40, text_coda_config=TextModuleConfig(vocab_size=41, **BLOCK))
    with pytest.raises(ValueError, match="eov"):
        _config(40, voice_coda_config=VoiceModuleConfig(vocab_size=N_UNITS, **BLOCK))


def _attention(**overrides: Any) -> LarryAttention:
    torch.manual_seed(0)
    block = {key: value for key, value in BLOCK.items() if key != "mlp_dropout_p"}
    return LarryAttention(AttentionModuleConfig(**{**block, **overrides})).eval()


def test_a_four_dimensional_causal_mask_matches_the_built_in_causal_attention() -> None:
    attention = _attention()
    x = torch.randn(2, 6, D_MODEL)
    mask = torch.ones(6, 6, dtype=torch.bool).tril().expand(2, 1, 6, 6)

    with torch.no_grad():
        assert torch.allclose(attention(x, attention_mask=mask).logits, attention(x).logits, atol=1e-6)


def test_a_four_dimensional_mask_can_open_attention_to_later_positions() -> None:
    attention = _attention()
    x = torch.randn(1, 4, D_MODEL)
    full = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    with torch.no_grad():
        opened = attention(x, attention_mask=full).logits
        causal = attention(x).logits

    assert not torch.allclose(opened[:, 0], causal[:, 0])
    assert torch.allclose(opened[:, -1], causal[:, -1], atol=1e-6)


@pytest.mark.parametrize(("local_axes", "second_axis_matters"), [(1, False), (2, True)])
def test_the_second_local_rope_axis_is_only_used_when_configured(local_axes: int, second_axis_matters: bool) -> None:
    attention = _attention(use_split_rope=True, split_rope_local_axes=local_axes)
    x = torch.randn(1, 5, D_MODEL)
    positions = torch.zeros(1, 5, 3, dtype=torch.long)
    positions[..., 0] = torch.arange(5)
    positions[..., 1] = torch.tensor([0, 0, 1, 1, 0])
    shifted = positions.clone()
    shifted[..., 2] = torch.tensor([0, 1, 0, 1, 0])

    with torch.no_grad():
        base = attention(x, position_ids=positions).logits
        moved = attention(x, position_ids=shifted).logits

    assert torch.allclose(base, moved, atol=1e-6) is not second_axis_matters


def test_two_local_axes_need_three_position_columns() -> None:
    attention = _attention(use_split_rope=True, split_rope_local_axes=2)

    with pytest.raises(ValueError, match="3"):
        attention(torch.randn(1, 3, D_MODEL), position_ids=torch.zeros(1, 3, 2, dtype=torch.long))


def test_extending_a_mask_prepends_visible_cached_keys() -> None:
    cache = KVCache(key_cache=torch.zeros(1, 2, 3, 4), value_cache=torch.zeros(1, 2, 3, 4))
    padding = torch.tensor([[1, 0]])
    block = torch.tensor([[[[True, False], [True, True]]]])

    assert cache.extend_mask(padding).tolist() == [[1, 1, 1, 1, 0]]
    assert cache.extend_mask(block).shape == (1, 1, 2, 5)
    assert cache.extend_mask(block)[..., :3].all()
    assert KVCache().extend_mask(padding) is padding
    assert cache.extend_mask(None) is None


def test_run_index_restarts_for_each_run() -> None:
    mask = torch.tensor([[False, True, True, False, True, True, True, False]])

    assert LarryMultimodalModel.run_index(mask).tolist() == [[0, 0, 1, 0, 0, 1, 2, 0]]


def test_generated_voice_slots_stay_causal(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    bov, vph, eov = (_special(collator, offset) for offset in (P.BOV, P.VPH, P.EOV))
    input_ids = torch.tensor([[3, bov, vph, vph, vph, eov]])

    mask = model.attention_mask(input_ids, model.key_valid(input_ids, None), torch.tensor([True]))[0, 0]

    assert torch.equal(mask, torch.ones(6, 6, dtype=torch.bool).tril())


def test_image_patches_fill_their_slots_in_order_and_padding_patches_are_dropped(
        collator: PretrainingMultimodalCollator,
) -> None:
    from larry.multimodal.model.multimodal_model import LarryModalityModuleOutput

    torch.manual_seed(0)
    vocab = collator.processor.text_tokenizer.vocab_size() + P.EOV + 1
    image_prelude = ImagePreludeConfig(n_layers=1, latent_channels=3, **BLOCK)
    model = LarryMultimodalModel(_config(vocab, image_prelude_config=image_prelude)).eval()
    boi, iph, eoi = (_special(collator, offset) for offset in (P.BOI, P.IPH, P.EOI))
    input_ids = torch.tensor([[3, boi, iph, iph, eoi, 4], [boi, iph, iph, eoi, 3, 4]])
    images = torch.zeros(2, 3, 4, 4)
    image_sizes = torch.tensor([[4, 2], [2, 4]])

    patch_ids = torch.tensor([[0.0, 1, 2, 3], [10, 11, 12, 13]])
    model.image_prelude.forward = lambda *_: LarryModalityModuleOutput(
        logits=patch_ids[..., None].expand(2, 4, D_MODEL)
    )
    seen: dict[str, torch.Tensor] = {}
    model.recurrent_block.register_forward_pre_hook(lambda _, args: seen.update(x=args[0]))

    with torch.no_grad():
        model(input_ids, torch.zeros_like(input_ids), images=images, image_sizes=image_sizes)

    slots = input_ids == _special(collator, P.IPH)
    assert seen["x"][slots][:, 0].tolist() == [0.0, 1.0, 10.0, 12.0]
    assert (seen["x"][slots] == seen["x"][slots][:, :1]).all()


def test_cached_chunked_decoding_matches_a_full_forward(collator: PretrainingMultimodalCollator) -> None:
    torch.manual_seed(0)
    vocab = collator.processor.text_tokenizer.vocab_size() + P.EOV + 1
    config = _config(vocab)
    config.recurrent_trunk_config.kv_cache_strategy = "per_iteration"
    model = LarryMultimodalModel(config).eval()
    input_ids = torch.tensor([collator.processor.text_tokenizer.encode("say this out loud hello there")[:8]])
    modality_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        full = model(input_ids, modality_ids).logits
        cache = None
        chunked = []
        for start in range(0, input_ids.shape[1], 2):
            step = model(
                input_ids[:, start:start + 2], modality_ids[:, start:start + 2],
                kv_cache=cache, position_offset=start, use_cache=True,
            )
            cache = step.kv_cache
            chunked.append(step.logits)

    assert torch.allclose(torch.cat(chunked), full[0::2], atol=1e-5)


def _voice_prelude(**overrides: Any) -> Any:
    from larry.multimodal.model.multimodal_model import VoicePrelude

    torch.manual_seed(0)
    return VoicePrelude(VoicePreludeConfig(**{"vocab_size": N_UNITS, "fsq_dims": 2, "n_layers": 1, **BLOCK, **overrides}))


def test_voice_prelude_codes_round_trip_through_fsq_indices() -> None:
    from larry.voice.model.fsq import FSQ

    prelude = _voice_prelude(vocab_size=3 ** 8, fsq_dims=8, fsq_levels=3)

    assert torch.equal(FSQ(D_MODEL, dims=8, levels=3).indices(prelude.codes), torch.arange(3 ** 8))
    assert set(prelude.codes.unique().tolist()) == {-1.0, 0.0, 1.0}


def test_voice_prelude_is_causal_for_generated_voice_and_bidirectional_for_input_voice() -> None:
    prelude = _voice_prelude().eval()
    units = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    changed = units.clone()
    changed[:, -1] = 8
    valid = torch.ones(2, 4, dtype=torch.bool)
    mask = LarryMultimodalModel.voice_prelude_mask(valid, torch.tensor([True, False]))

    with torch.no_grad():
        before = prelude(units[..., None], valid[..., None], mask).logits
        after = prelude(changed[..., None], valid[..., None], mask).logits

    assert torch.allclose(before[0, :3], after[0, :3], atol=1e-6)
    assert not torch.allclose(before[1, :3], after[1, :3])


def test_voice_prelude_decodes_one_unit_at_a_time_with_its_cache_and_conv_state() -> None:
    prelude = _voice_prelude(conv_kernel_size=3).eval()
    units = torch.tensor([[1, 5, 2, 8, 0, 3]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    synthesis = torch.tensor([True])

    with torch.no_grad():
        full_mask = LarryMultimodalModel.voice_prelude_mask(valid, synthesis)
        full = prelude(units[..., None], valid[..., None], full_mask).logits
        cache, conv_state, steps = None, None, []
        for i in range(6):
            mask = LarryMultimodalModel.voice_prelude_mask(valid[:, :1], synthesis)
            step = prelude(units[:, i:i + 1, None], valid[:, :1, None], mask, cache, i, True, conv_state)
            cache, conv_state = step.kv_cache, step.conv_state
            steps.append(step.logits)

    assert conv_state.shape == (1, 2, D_MODEL)
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_the_voice_head_is_tied_to_the_prelude_unit_embeddings_with_its_own_eov_row(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator)
    x = torch.randn(5, D_MODEL)

    logits = model.voice_coda.head(x, model.voice_prelude.unit_embeddings())

    normed = model.voice_coda.norm(x)
    assert logits.shape == (5, N_UNITS + 1)
    assert torch.allclose(logits[:, :N_UNITS], normed @ model.voice_prelude.unit_embeddings().T, atol=1e-5)
    assert torch.allclose(logits[:, N_UNITS], normed @ model.voice_coda.eov, atol=1e-5)


def test_voice_loss_trains_every_tied_unit_row(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator).train()

    _forward(model, collator(_items())).voice_logits.logsumexp(-1).mean().backward()

    assert model.voice_prelude.ute.weight.grad.abs().sum(-1).gt(0).all()
    assert model.voice_coda.eov.grad.abs().sum() > 0


def test_input_voice_is_bidirectional_in_the_trunk_mask(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator)
    bov, vph, eov = (_special(collator, offset) for offset in (P.BOV, P.VPH, P.EOV))
    input_ids = torch.tensor([[bov, vph, vph, vph, eov, 3]] * 2)

    mask = model.attention_mask(input_ids, model.key_valid(input_ids, None), torch.tensor([False, True]))[:, 0]

    assert mask[0, 1:4, 1:4].all()
    assert not mask[0, 0, 1:].any() and not mask[0, 1:4, 4:].any()
    assert torch.equal(mask[1], torch.ones(6, 6, dtype=torch.bool).tril())


def test_patchify_is_row_major_and_unpatchify_inverts_it() -> None:
    from larry.image.model.image_modules import Patchify, Unpatchify

    latents = torch.arange(2 * 3 * 4 * 6, dtype=torch.float).view(2, 3, 4, 6)

    patches = Patchify(2)(latents)

    assert patches.shape == (2, 2 * 3, 3 * 2 * 2)
    assert patches[0, 1, :4].tolist() == latents[0, 0, 0:2, 2:4].flatten().tolist()
    assert patches[0, 3, :4].tolist() == latents[0, 0, 2:4, 0:2].flatten().tolist()
    assert torch.equal(Unpatchify(2)(patches, 2, 3), latents)


def test_padding_patches_do_not_change_the_image_preludes_real_patches() -> None:
    from larry.multimodal.model.multimodal_model import ImagePrelude

    torch.manual_seed(0)
    prelude = ImagePrelude(ImagePreludeConfig(n_layers=1, latent_channels=3, **BLOCK, **SPLIT)).eval()
    latent = torch.randn(1, 3, 2, 4)
    padded = torch.randn(1, 3, 4, 4)
    padded[..., :2, :4] = latent
    positions = torch.stack([torch.arange(4), torch.arange(4) // 2, torch.arange(4) % 2], dim=-1)[None]

    with torch.no_grad():
        alone = prelude(latent, torch.ones(1, 2, dtype=torch.bool), positions[:, :2]).logits
        among_padding = prelude(padded, torch.tensor([[True, True, False, False]]), positions).logits

    assert among_padding.shape == (1, 4, D_MODEL)
    assert torch.allclose(among_padding[:, :2], alone, atol=1e-5)


def _image_model(collator: PretrainingMultimodalCollator, with_prelude: bool = True) -> LarryMultimodalModel:
    from larry.multimodal.config.model.model_configs import ImageCodaConfig

    torch.manual_seed(0)
    vocab = collator.processor.text_tokenizer.vocab_size() + P.EOV + 1
    image = {"n_layers": 1, "latent_channels": 3, **BLOCK, **SPLIT}
    return LarryMultimodalModel(_config(
        vocab,
        image_prelude_config=ImagePreludeConfig(**image) if with_prelude else None,
        image_coda_config=ImageCodaConfig(max_query_grid=8, **image),
    )).eval()


def _image_batch(collator: PretrainingMultimodalCollator) -> dict[str, torch.Tensor]:
    boi, iph, eoi, eos = (_special(collator, offset) for offset in (P.BOI, P.IPH, P.EOI, P.EOS))
    generator = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.tensor([[3, boi, iph, iph, eoi, eos], [boi, iph, iph, eoi, 3, eos]]),
        "modality_ids": torch.tensor([[0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0]]),
        "is_synthesis": torch.tensor([True, False]),
        "images": torch.randn(2, 3, 4, 4, generator=generator),
        "image_sizes": torch.tensor([[4, 2], [2, 4]]),
    }


def test_synthesis_slots_hold_queries_and_route_to_the_image_coda_unshifted(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _image_model(collator)
    batch = _image_batch(collator)
    called_with: list[torch.Tensor] = []
    forward = model.image_prelude.forward
    model.image_prelude.forward = lambda latents, *args: called_with.append(latents) or forward(latents, *args)

    with torch.no_grad():
        out = model(**batch)

    slots = batch["input_ids"] == _special(collator, P.IPH)
    assert torch.equal(out.image_route, slots & torch.tensor([[True], [False]]))
    assert not (out.text_route & slots).any()
    assert out.text_route[0, 0] and not out.text_route[0, 1]
    assert len(called_with) == 1 and torch.equal(called_with[0], batch["images"][1:])


def test_image_flow_targets_cover_only_the_real_patches(collator: PretrainingMultimodalCollator) -> None:
    model = _image_model(collator)
    batch = _image_batch(collator)

    with torch.no_grad():
        out = model(**batch)

    flow = out.image_flow
    assert flow.logits.shape == flow.targets.shape == (1, 3 * 2 * 2, 4)
    assert flow.mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert not flow.logits.any()
    assert out.image_query_states.shape == (1, 4, D_MODEL)
    assert not out.image_query_states[0, 2:].any()


def test_text_to_image_needs_no_image_prelude(collator: PretrainingMultimodalCollator) -> None:
    model = _image_model(collator, with_prelude=False)
    batch = _image_batch(collator)
    batch = {key: value[:1] for key, value in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    assert model.image_prelude is None
    assert out.image_flow is not None


def test_image_queries_follow_their_patch_row_and_column(collator: PretrainingMultimodalCollator) -> None:
    model = _image_model(collator)
    batch = _image_batch(collator)
    seen: dict[str, torch.Tensor] = {}
    model.recurrent_block.register_forward_pre_hook(lambda _, args: seen.update(x=args[0]))

    with torch.no_grad():
        model(**batch)

    coda = model.image_coda
    expected = coda.queries(torch.tensor([0, 0]), torch.tensor([0, 1]))
    assert torch.allclose(seen["x"][0, 2:4], expected, atol=1e-6)


def test_image_synthesis_needs_image_sizes_for_its_queries(collator: PretrainingMultimodalCollator) -> None:
    model = _image_model(collator)
    batch = _image_batch(collator)
    del batch["image_sizes"]

    with pytest.raises(ValueError, match="image_sizes"):
        model(**batch)


def test_image_sampling_returns_latents_of_the_requested_grid(collator: PretrainingMultimodalCollator) -> None:
    model = _image_model(collator)
    states = torch.randn(2, 6, D_MODEL)
    mask = torch.ones(2, 6, dtype=torch.bool)
    positions = torch.stack([torch.arange(6), torch.arange(6) // 3, torch.arange(6) % 3], dim=-1).expand(2, 6, 3)

    latents = model.image_coda.sample(states, mask, 2, 3, steps=2, position_ids=positions)

    assert latents.shape == (2, 3, 4, 6)
    assert torch.isfinite(latents).all()


@pytest.mark.parametrize(("overrides", "match"), [
    ({"voice_prelude_config": VoicePreludeConfig(vocab_size=10, fsq_dims=2, **BLOCK)}, "fsq_levels"),
    ({"voice_prelude_config": None}, "tied to the voice prelude"),
    ({"voice_coda_config": VoiceModuleConfig(vocab_size=N_UNITS + 1, **{**BLOCK, "d_model": 16})}, "d_model"),
])
def test_module_configs_are_validated(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(40, **overrides)


def test_image_prelude_and_coda_must_share_a_latent_layout() -> None:
    from larry.multimodal.config.model.model_configs import ImageCodaConfig

    with pytest.raises(ValueError, match="latent_channels"):
        _config(
            40,
            image_prelude_config=ImagePreludeConfig(latent_channels=4, **BLOCK),
            image_coda_config=ImageCodaConfig(latent_channels=8, **BLOCK),
        )
