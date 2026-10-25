import json
from pathlib import Path

import numpy as np
import pytest
import sentencepiece
from PIL import Image

from larry.multimodal.config.processor_configs import LarryProcessorConfig
from larry.multimodal.processor import processor as P
from larry.multimodal.processor.processor import (
    ImagePlaceholder,
    LarryMultimodalPretrainProcessor,
    LarryMultimodalProcessor,
    TextSpan,
    VoicePlaceholder,
)

VOICE = "<|voice_placeholder|>"
IMAGE = "<|image_placeholder|>"


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> LarryProcessorConfig:
    root: Path = tmp_path_factory.mktemp("sp")
    (root / "text.txt").write_text("\n".join(["hello there friend", "say this out loud", "what did they say"] * 40))
    sentencepiece.SentencePieceTrainer.train(
        input=str(root / "text.txt"),
        model_prefix=str(root / "sp"),
        vocab_size=20,
        model_type="unigram",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    return LarryProcessorConfig(text_tokenizer_model_file_path=str(root / "sp.model"))


@pytest.fixture(scope="module")
def processor(config: LarryProcessorConfig) -> LarryMultimodalProcessor:
    return LarryMultimodalProcessor(config)


def _special(processor: LarryMultimodalProcessor, offset: int) -> int:
    return processor.text_tokenizer.vocab_size() + offset


def _units(*ids: int) -> np.ndarray:
    return np.array(ids, dtype=np.int64)


def test_text_only_is_its_tokens_followed_by_eos(processor: LarryMultimodalProcessor) -> None:
    out = processor.encode("hello there")

    tokens = processor.text_tokenizer.encode("hello there")
    assert out.text_token_ids == [*tokens, _special(processor, P.EOS)]
    assert out.modality_ids == [0] * len(out.text_token_ids)
    assert [type(s) for s in out.input_placeholders] == [TextSpan]


def test_a_voice_placeholder_becomes_delimited_special_tokens_in_place(processor: LarryMultimodalProcessor) -> None:
    out = processor.encode(f"say {VOICE} this", voice_examples=[_units(5, 9, 9, 2)])

    before = processor.text_tokenizer.encode("say ")
    after = processor.text_tokenizer.encode(" this")
    delimited = [_special(processor, P.BOV), _special(processor, P.VPH), _special(processor, P.EOV)]
    assert out.text_token_ids == [*before, *delimited, *after, _special(processor, P.EOS)]
    assert [type(s) for s in out.input_placeholders] == [TextSpan, VoicePlaceholder, TextSpan]


def test_the_voice_segment_carries_its_units_and_points_at_its_placeholder_token(
        processor: LarryMultimodalProcessor
) -> None:
    out = processor.encode(f"say {VOICE} this", voice_examples=[_units(5, 9, 9, 2)])

    voice = out.input_placeholders[1]
    assert isinstance(voice, VoicePlaceholder)
    assert voice.payload.dtype == np.int32
    assert voice.payload.tolist() == [5, 9, 9, 2]
    assert voice.size == 4
    assert out.text_token_ids[voice.text_idx] == _special(processor, P.VPH)


@pytest.mark.parametrize("text", [
    f"{VOICE} say {VOICE} this",
    f"say this {VOICE} out {VOICE}",
    f"{VOICE}{VOICE}",
    f"say {VOICE} this {VOICE} out",
])
def test_every_placeholder_takes_the_next_example_in_order(processor: LarryMultimodalProcessor, text: str) -> None:
    examples = [_units(1, 2), _units(3, 4, 5)]

    out = processor.encode(text, voice_examples=examples)

    voices = [s for s in out.input_placeholders if isinstance(s, VoicePlaceholder)]
    assert [v.payload.tolist() for v in voices] == [[1, 2], [3, 4, 5]]
    assert [out.text_token_ids[v.text_idx] for v in voices] == [_special(processor, P.VPH)] * 2
    assert all(s.text_token_ids for s in out.input_placeholders if isinstance(s, TextSpan))


@pytest.mark.parametrize(("text", "examples"), [
    (f"say {VOICE} this", []),
    (f"say {VOICE} this", [_units(1), _units(2)]),
    ("say this", [_units(1)]),
])
def test_placeholder_and_example_counts_must_match(
        processor: LarryMultimodalProcessor, text: str, examples: list[np.ndarray]
) -> None:
    with pytest.raises(ValueError):
        processor.encode(text, voice_examples=examples)


def test_modality_ids_label_every_token(processor: LarryMultimodalProcessor) -> None:
    out = processor.encode(f"say {VOICE} this out loud", voice_examples=[_units(1, 2, 3)])

    assert len(out.modality_ids) == len(out.text_token_ids)
    voice = out.input_placeholders[1]
    expected = [0, VoicePlaceholder.modality_id, VoicePlaceholder.modality_id]
    assert out.modality_ids[voice.text_idx - 1:voice.text_idx + 2] == expected
    assert out.modality_ids.count(VoicePlaceholder.modality_id) == 2


def test_image_delimiters_are_text_around_an_image_placeholder(processor: LarryMultimodalProcessor) -> None:
    out = processor.encode(f"look {IMAGE} here", image_examples=[Image.new("RGB", (6, 4))])

    image = out.input_placeholders[1]
    assert isinstance(image, ImagePlaceholder)
    assert image.size == (6, 4)
    assert image.payload.shape == (3, 4, 6)
    assert out.text_token_ids[image.text_idx] == _special(processor, P.IPH)
    assert out.modality_ids[image.text_idx - 1:image.text_idx + 2] == [0, ImagePlaceholder.modality_id, 0]


def test_literal_special_token_text_stays_text(processor: LarryMultimodalProcessor) -> None:
    out = processor.encode("say <|voice|> this")

    specials = {_special(processor, offset) for offset in (P.BOV, P.VPH, P.EOV)}
    assert not specials & set(out.text_token_ids)


@pytest.mark.parametrize("units", [[1, 2, 3], np.array([1, 2, 3]), np.array([[1, 2, 3]])])
def test_voice_examples_accept_lists_and_arrays(processor: LarryMultimodalProcessor, units: object) -> None:
    out = processor.encode(VOICE, voice_examples=[units])

    assert out.input_placeholders[0].payload.tolist() == [1, 2, 3]


def test_batches_encode_each_row_independently(processor: LarryMultimodalProcessor) -> None:
    texts = ["hello there", f"say {VOICE}"]

    batch = processor.encode_batch(texts, [None, None], [None, [_units(7)]])

    assert [b.text_token_ids for b in batch] == [
        processor.encode(texts[0]).text_token_ids,
        processor.encode(texts[1], voice_examples=[_units(7)]).text_token_ids,
    ]


@pytest.fixture(scope="module")
def pretrain(config: LarryProcessorConfig) -> LarryMultimodalPretrainProcessor:
    return LarryMultimodalPretrainProcessor(config)


def _is_text(processor: LarryMultimodalPretrainProcessor, token: int) -> bool:
    return token < processor.text_tokenizer.vocab_size()


def test_synthesis_appends_the_voice_after_the_prompt_and_ends_with_eos(
        pretrain: LarryMultimodalPretrainProcessor,
) -> None:
    out = pretrain.encode("say this", voice_example=_units(1, 2, 3), is_modal_synthesis=True)

    specials = [_special(pretrain, offset) for offset in (P.BOV, P.VPH, P.EOV, P.EOS)]
    assert out.text_token_ids[-4:] == specials
    assert all(_is_text(pretrain, t) for t in out.text_token_ids[:-4])
    assert out.input_placeholders[-1].text_idx == len(out.text_token_ids) - 3


def test_captioning_prepends_the_voice_before_the_text(pretrain: LarryMultimodalPretrainProcessor) -> None:
    out = pretrain.encode("say this", voice_example=_units(1, 2, 3), is_modal_synthesis=False)

    specials = [_special(pretrain, offset) for offset in (P.BOV, P.VPH, P.EOV)]
    assert out.text_token_ids[:3] == specials
    assert out.text_token_ids[-1] == _special(pretrain, P.EOS)
    assert out.input_placeholders[0].text_idx == 1


def test_synthesis_trains_on_the_voice_and_not_the_prompt(pretrain: LarryMultimodalPretrainProcessor) -> None:
    out = pretrain.encode("say this", voice_example=_units(1, 2, 3), is_modal_synthesis=True)

    assert len(out.loss_mask) == len(out.text_token_ids)
    text_flags = [m for t, m in zip(out.text_token_ids, out.loss_mask, strict=True) if _is_text(pretrain, t)]
    assert not any(text_flags)
    voice = out.input_placeholders[-1]
    assert out.loss_mask[voice.text_idx - 1:voice.text_idx + 2] == [True] * 3
    assert out.loss_mask[-1]


def test_captioning_trains_on_the_text_and_not_the_voice(pretrain: LarryMultimodalPretrainProcessor) -> None:
    out = pretrain.encode("say this", voice_example=_units(1, 2, 3), is_modal_synthesis=False)

    text_flags = [m for t, m in zip(out.text_token_ids, out.loss_mask, strict=True) if _is_text(pretrain, t)]
    assert all(text_flags)
    voice = out.input_placeholders[0]
    assert out.loss_mask[voice.text_idx - 1:voice.text_idx + 2] == [False] * 3
    assert out.loss_mask[-1]


@pytest.mark.parametrize("is_modal_synthesis", [True, False])
def test_text_without_media_is_always_trained_on(
        pretrain: LarryMultimodalPretrainProcessor, is_modal_synthesis: bool,
) -> None:
    out = pretrain.encode("say this", is_modal_synthesis=is_modal_synthesis)

    assert all(out.loss_mask)
    assert len(out.loss_mask) == len(out.text_token_ids)
    assert out.input_placeholders == [TextSpan(text_token_ids=out.text_token_ids[:-1])]


@pytest.mark.parametrize("text", [f"say {VOICE}", "say <|image_placeholder|>"])
def test_pretraining_rejects_placeholders_in_the_text(pretrain: LarryMultimodalPretrainProcessor, text: str) -> None:
    with pytest.raises(ValueError, match="special tokens"):
        pretrain.encode(text, voice_example=_units(1))


def test_pretraining_rejects_an_image_and_a_voice_together(pretrain: LarryMultimodalPretrainProcessor) -> None:
    with pytest.raises(ValueError, match="multiple"):
        pretrain.encode("x", image_example=Image.new("RGB", (4, 4)), voice_example=_units(1))


def test_image_synthesis_trains_on_the_opening_delimiter_and_image_but_not_the_closing_one(
        pretrain: LarryMultimodalPretrainProcessor,
) -> None:
    out = pretrain.encode("draw this", image_example=Image.new("RGB", (4, 4)), is_modal_synthesis=True)

    image = out.input_placeholders[-1]
    assert out.loss_mask[image.text_idx - 1:image.text_idx + 2] == [True, True, False]
    assert not any(out.loss_mask[:image.text_idx - 1])
    assert out.loss_mask[-1]


def test_image_captioning_keeps_every_image_token_out_of_the_loss(pretrain: LarryMultimodalPretrainProcessor) -> None:
    out = pretrain.encode("a cat", image_example=Image.new("RGB", (4, 4)), is_modal_synthesis=False)

    image = out.input_placeholders[0]
    assert out.loss_mask[:3] == [False] * 3
    assert all(out.loss_mask[image.text_idx + 2:])


def test_saving_bundles_the_tokenizer_and_a_relative_config(processor: LarryMultimodalProcessor, tmp_path: Path) -> None:
    processor.save_pretrained(tmp_path)

    saved = (tmp_path / P.TOKENIZER_FILE).read_bytes()
    assert saved == Path(processor.config.text_tokenizer_model_file_path).read_bytes()
    assert json.loads((tmp_path / "config.json").read_text()) == {
        "text_tokenizer_model_file_path": P.TOKENIZER_FILE,
        "image_patch_size": 2,
        "voice_units_per_position": 1,
    }


@pytest.mark.parametrize("cls", [LarryMultimodalProcessor, LarryMultimodalPretrainProcessor])
def test_a_saved_processor_loads_back_with_the_same_tokenizer(
        processor: LarryMultimodalProcessor, tmp_path: Path, cls: type,
) -> None:
    processor.save_pretrained(tmp_path)

    loaded = cls.from_pretrained(tmp_path)

    assert isinstance(loaded, cls)
    assert loaded.config.text_tokenizer_model_file_path == str(tmp_path / P.TOKENIZER_FILE)
    assert loaded.encode("say this out loud").text_token_ids == processor.encode("say this out loud").text_token_ids


def test_a_loaded_processor_can_be_saved_over_its_own_directory(
        processor: LarryMultimodalProcessor, tmp_path: Path,
) -> None:
    processor.save_pretrained(tmp_path)
    loaded = LarryMultimodalProcessor.from_pretrained(tmp_path)

    loaded.save_pretrained(tmp_path)

    assert LarryMultimodalProcessor.from_pretrained(tmp_path).text_tokenizer.vocab_size() == 20


def test_loading_from_the_hub_downloads_the_tokenizer_named_in_the_config(
        processor: LarryMultimodalProcessor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor.save_pretrained(tmp_path)
    requested: list[tuple[str, str]] = []

    def download(*, repo_id: str, filename: str, **_: object) -> str:
        requested.append((repo_id, filename))
        return str(tmp_path / filename)

    monkeypatch.setattr(P, "hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.hub_mixin.hf_hub_download", download)

    loaded = LarryMultimodalProcessor.from_pretrained("someone/larry-processor")

    assert requested == [("someone/larry-processor", "config.json"), ("someone/larry-processor", P.TOKENIZER_FILE)]
    assert loaded.config.text_tokenizer_model_file_path == str(tmp_path / P.TOKENIZER_FILE)


def test_image_latents_keep_their_layout_and_size_in_latent_cells(processor: LarryMultimodalProcessor) -> None:
    latents = np.random.default_rng(0).standard_normal((16, 4, 6)).astype(np.float64)

    out = processor.encode(f"look {IMAGE} here", image_examples=[latents])

    image = out.input_placeholders[1]
    assert image.size == (6, 4)
    assert image.payload.dtype == np.float32
    assert np.allclose(image.payload, latents)


def test_image_latents_must_be_channel_first_three_dimensional(processor: LarryMultimodalProcessor) -> None:
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        processor.encode(f"look {IMAGE} here", image_examples=[np.zeros((4, 6))])


def test_an_inference_prompt_can_continue_or_open_a_modality(processor: LarryMultimodalProcessor) -> None:
    tokens = processor.text_tokenizer.encode("say this")

    assert processor.encode("say this", add_eos=False).text_token_ids == tokens
    opened = processor.encode("say this", open_modality="voice")
    assert opened.text_token_ids == [*tokens, _special(processor, P.BOV)]
    assert opened.modality_ids[-1] == TextSpan.modality_id
    assert processor.encode("say this", open_modality="image").text_token_ids[-1] == _special(processor, P.BOI)


def test_stretching_repeats_each_slot_and_keeps_payloads_in_order(processor: LarryMultimodalProcessor) -> None:
    latents = np.zeros((3, 4, 6), dtype=np.float32)
    encoded = processor.encode(
        f"{VOICE} look {IMAGE} say {VOICE}", image_examples=[latents], voice_examples=[_units(1, 2, 3), _units(7)]
    )

    stretched = processor.stretch(encoded)

    ids = stretched.input_ids
    assert ids.count(_special(processor, P.VPH)) == 4
    assert ids.count(_special(processor, P.IPH)) == (6 // 2) * (4 // 2)
    assert len(stretched.modality_ids) == len(ids)
    assert [v.tolist() for v in stretched.voices] == [[1, 2, 3], [7]]
    assert stretched.image_sizes == [(6, 4)]
    assert stretched.loss_mask is None


def test_grouped_voice_stretches_to_one_slot_per_group(processor: LarryMultimodalProcessor) -> None:
    import dataclasses

    grouped = LarryMultimodalProcessor(dataclasses.replace(processor.config, voice_units_per_position=2))

    stretched = grouped.stretch(grouped.encode(VOICE, voice_examples=[_units(1, 2, 3)]))

    assert stretched.input_ids.count(_special(processor, P.VPH)) == 2
