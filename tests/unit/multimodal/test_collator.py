import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sentencepiece
import torch
from PIL import Image

from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
from larry.multimodal.config.processor_configs import LarryProcessorConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.processor import processor as P
from larry.multimodal.processor.processor import ImagePlaceholder, LarryMultimodalPretrainProcessor, VoicePlaceholder


@pytest.fixture(scope="module")
def collator(tmp_path_factory: pytest.TempPathFactory) -> PretrainingMultimodalCollator:
    root: Path = tmp_path_factory.mktemp("collator")
    (root / "text.txt").write_text("\n".join(["hello there friend", "say this out loud", "a cat on a mat"] * 40))
    sentencepiece.SentencePieceTrainer.train(
        input=str(root / "text.txt"),
        model_prefix=str(root / "sp"),
        vocab_size=24,
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        minloglevel=2,
    )
    config = LarryProcessorConfig(text_tokenizer_model_file_path=str(root / "sp.model"))
    LarryMultimodalPretrainProcessor(config).save_pretrained(root / "processor")
    collator_config = PretrainingMultimodalCollatorConfig(processor_model_path=str(root / "processor"))
    return PretrainingMultimodalCollator(collator_config)


def _special(collator: PretrainingMultimodalCollator, offset: int) -> int:
    return collator.processor.text_tokenizer.vocab_size() + offset


def _items() -> list[dict[str, Any]]:
    return [
        {"text": "say this", "voice": np.array([4, 5, 6]), "source": 0},
        {"text": "a cat", "image": Image.new("RGB", (6, 4), (255, 0, 0)), "is_modal_synthesis": False, "source": 1},
        {"text": "hello there", "source": 2},
    ]


def test_every_per_position_tensor_has_the_padded_sequence_shape(collator: PretrainingMultimodalCollator) -> None:
    out = collator(_items())

    shape = out["input_ids"].shape
    assert out["modality_ids"].shape == out["loss_mask"].shape == shape
    assert shape[1] == int(out["input_lengths"].max())
    assert out["loss_mask"].dtype == torch.bool


def test_rows_match_the_processor_once_placeholders_are_stretched(collator: PretrainingMultimodalCollator) -> None:
    out = collator(_items())

    text_only = collator.processor.encode("hello there")
    length = int(out["input_lengths"][2])
    assert out["input_ids"][2, :length].tolist() == text_only.text_token_ids
    assert out["loss_mask"][2, :length].tolist() == text_only.loss_mask
    assert all(out["input_ids"][row, int(out["input_lengths"][row]) - 1] == _special(collator, P.EOS) for row in range(3))


def test_a_voice_placeholder_is_stretched_to_one_position_per_unit(collator: PretrainingMultimodalCollator) -> None:
    out = collator(_items())

    vph = out["input_ids"] == _special(collator, P.VPH)
    assert vph.sum(dim=1).tolist() == [3, 0, 0]
    assert (out["modality_ids"][vph] == VoicePlaceholder.modality_id).all()
    assert out["loss_mask"][0][vph[0]].all()
    positions = vph[0].nonzero().flatten()
    assert out["input_ids"][0, positions[0] - 1] == _special(collator, P.BOV)
    assert out["input_ids"][0, positions[-1] + 1] == _special(collator, P.EOV)


def test_an_image_placeholder_is_stretched_to_one_position_per_patch(collator: PretrainingMultimodalCollator) -> None:
    out = collator(_items())

    iph = out["input_ids"] == _special(collator, P.IPH)
    assert iph.sum(dim=1).tolist() == [0, (6 // 2) * (4 // 2), 0]
    assert (out["modality_ids"][iph] == ImagePlaceholder.modality_id).all()
    assert not out["loss_mask"][1][iph[1]].any()


def test_unit_ids_line_up_with_the_stretched_voice_positions_in_row_major_order(
        collator: PretrainingMultimodalCollator,
) -> None:
    items = [*_items(), {"text": "say", "voice": np.array([7, 8]), "source": 3}]

    out = collator(items)

    units = out["voice_unit_ids"]
    valid = torch.arange(units.shape[1]) < out["voice_unit_id_lengths"][:, None]
    vph = out["input_ids"] == _special(collator, P.VPH)
    assert out["voice_unit_id_lengths"].tolist() == [3, 0, 0, 2]
    assert units[valid].tolist() == [4, 5, 6, 7, 8]
    assert int(vph.sum()) == int(valid.sum())


def test_images_are_channel_first_and_padded_to_the_largest(collator: PretrainingMultimodalCollator) -> None:
    items = [*_items(), {"text": "a mat", "image": Image.new("RGB", (2, 8)), "source": 3}]

    out = collator(items)

    assert out["images"].shape == (4, 3, 8, 6)
    assert out["image_sizes"].tolist() == [[0, 0], [6, 4], [0, 0], [2, 8]]
    assert torch.equal(out["images"][1, :, :4, :6], torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1).expand(3, 4, 6))
    assert not out["images"][1, :, 4:].any()


def test_text_only_batches_still_collate(collator: PretrainingMultimodalCollator) -> None:
    out = collator([{"text": "hello there", "source": 0}, {"text": "say this out loud", "source": 1}])

    assert out["voice_unit_ids"].shape == (2, 0)
    assert out["images"].shape == (2, 0, 0, 0)
    assert out["loss_mask"][1, :int(out["input_lengths"][1])].all()


def test_row_flags_are_carried_through(collator: PretrainingMultimodalCollator) -> None:
    out = collator(_items())

    assert out["is_synthesis_mask"].tolist() == [True, False, True]
    assert out["source_ids"].tolist() == [0, 1, 2]


def test_the_collator_survives_pickling_for_dataloader_workers(collator: PretrainingMultimodalCollator) -> None:
    restored = pickle.loads(pickle.dumps(collator))

    assert torch.equal(restored(_items())["input_ids"], collator(_items())["input_ids"])


def test_latent_images_are_padded_with_their_own_channel_count(collator: PretrainingMultimodalCollator) -> None:
    latent = torch.randn(16, 4, 6)
    items = [
        {"text": "a cat", "image": latent, "is_modal_synthesis": False, "source": 0},
        {"text": "a mat", "image": torch.randn(16, 2, 2), "source": 1},
    ]

    out = collator(items)

    assert out["images"].shape == (2, 16, 4, 6)
    assert torch.equal(out["images"][0], latent)
    assert out["image_sizes"].tolist() == [[6, 4], [2, 2]]
    assert (out["input_ids"] == _special(collator, P.IPH)).sum(dim=1).tolist() == [6, 1]
