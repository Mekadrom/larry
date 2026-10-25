import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from larry.common.data.dataloading.indexed_dataset import load_index, load_source
from larry.multimodal.config.data.data_configs import MultimodalSourceConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.data.dataloading.indexed_multimodal_dataset import IndexedMultimodalDataset
from tests.helpers.common import write_parquet_split

TEXTS = {
    "voice_text": ["say this", "hello there friend"],
    "text_voice": ["a cat on a mat", "say"],
    "text_image": ["a cat", "hello"],
    "image_text": ["a mat", "say this out loud"],
    "text_only": ["hello there friend", "say this out loud"],
}
UNITS = [[4, 5, 6], [1, 2, 3, 4, 5]]
SHAPES = [[3, 2, 4], [3, 4, 6]]


def task_columns(task: str, tokenizer: Any | None = None) -> dict[str, list[Any]]:
    texts = TEXTS[task]
    columns: dict[str, list[Any]] = {"id": [f"{task}-{i}" for i in range(len(texts))], "text": texts}
    if "voice" in task:
        columns["unit_ids"] = UNITS
    if "image" in task:
        columns["latents"] = [np.arange(np.prod(shape), dtype=np.float32).tolist() for shape in SHAPES]
        columns["latent_shape"] = SHAPES
    if tokenizer is not None:
        columns["text_tokens"] = [len(tokenizer.encode(text)) for text in texts]
    return columns


def source_configs(root: Path, tasks: list[str], with_tokens: bool = False) -> list[MultimodalSourceConfig]:
    return [
        MultimodalSourceConfig(
            name=task,
            index_path=str(root / task),
            task=task,
            text_tokens_column="text_tokens" if with_tokens else None,
        )
        for task in tasks
    ]


def write_sources(root: Path, tasks: list[str], tokenizer: Any | None = None) -> list[MultimodalSourceConfig]:
    for task in tasks:
        write_parquet_split(root / task, "train", task_columns(task, tokenizer))
    return source_configs(root, tasks, with_tokens=tokenizer is not None)


def load(configs: list[MultimodalSourceConfig], **kwargs: Any) -> IndexedMultimodalDataset:
    indexes = [load_index(config) for config in configs]
    return IndexedMultimodalDataset(configs, indexes, [load_source(config) for config in configs], **kwargs)


def test_rows_become_collator_items_with_each_tasks_media(tmp_path: Path) -> None:
    dataset = load(write_sources(tmp_path, list(TEXTS)))

    items = [dataset[i] for i in range(len(dataset))]

    by_task = {task: [item for item in items if dataset.configs[item["source"]].task == task] for task in TEXTS}
    assert [item["voice"].tolist() for item in by_task["voice_text"]] == UNITS
    assert [item["image"].shape for item in by_task["text_image"]] == [(3, 2, 4), (3, 4, 6)]
    assert by_task["image_text"][0]["image"][0, 1].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert not any("voice" in item or "image" in item for item in by_task["text_only"])
    assert [item["is_modal_synthesis"] for item in (by_task["voice_text"][0], by_task["text_voice"][0])] == [False, True]


@pytest.mark.parametrize("k", [1, 2])
def test_lengths_match_the_collated_sequences_when_token_counts_are_stored(
        collator: PretrainingMultimodalCollator, tmp_path: Path, k: int,
) -> None:
    tokenizer = collator.processor.text_tokenizer
    dataset = load(write_sources(tmp_path, list(TEXTS), tokenizer), voice_units_per_position=k)
    processor = type(collator.processor)(dataclasses.replace(collator.processor.config, voice_units_per_position=k))

    _, lengths = dataset.source_ids_and_lengths()

    for i in range(len(dataset)):
        item = dataset[i]
        stretched = processor.stretch(processor.encode(
            item["text"], item.get("image"), item.get("voice"), is_modal_synthesis=item["is_modal_synthesis"]
        ))
        assert int(lengths[i]) == len(stretched.input_ids), dataset.configs[item["source"]].task


def test_lengths_estimate_text_from_characters_without_token_counts(tmp_path: Path) -> None:
    dataset = load(write_sources(tmp_path, ["text_only", "voice_text"]), chars_per_token=2.0)

    source_ids, lengths = dataset.source_ids_and_lengths()

    assert source_ids.tolist() == [0, 0, 1, 1]
    assert lengths[:2].tolist() == [9 + 1, 9 + 1]
    assert lengths[2:].tolist() == [4 + 3 + 2 + 1, 9 + 5 + 2 + 1]


def test_media_can_live_in_a_separate_source_dataset(tmp_path: Path) -> None:
    source_root = write_parquet_split(tmp_path / "source", "train", task_columns("text_voice"))
    index_root = write_parquet_split(tmp_path / "index", "train", {
        "id": ["text_voice-1", "text_voice-0"], "row": [1, 0], "text": ["say", "a cat on a mat"],
    })
    config = MultimodalSourceConfig(
        name="tts", index_path=str(index_root), source_path=str(source_root), task="text_voice"
    )

    dataset = load([config])

    assert [dataset[i]["voice"].tolist() for i in range(2)] == [UNITS[1], UNITS[0]]
    assert dataset.source_ids_and_lengths()[1].tolist() == [1 + 5 + 2 + 1, 4 + 3 + 2 + 1]
    assert torch.equal(dataset.source_ids_and_lengths()[0], torch.zeros(2, dtype=torch.long))
