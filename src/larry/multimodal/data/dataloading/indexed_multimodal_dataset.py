from typing import Any

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset as HFDataset

from larry.common.data.dataloading.indexed_dataset import IndexedDataset
from larry.multimodal.config.data.data_configs import MultimodalSourceConfig


class IndexedMultimodalDataset(IndexedDataset[MultimodalSourceConfig]):
    def __init__(
            self,
            configs: list[MultimodalSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            chars_per_token: float = 4.0,
            image_patch_size: int = 2,
            voice_units_per_position: int = 1,
    ) -> None:
        super().__init__(configs, indexes, sources)
        self.chars_per_token = chars_per_token
        self.image_patch_size = image_patch_size
        self.voice_units_per_position = voice_units_per_position

    @staticmethod
    def has_voice(config: MultimodalSourceConfig) -> bool:
        return "voice" in config.task

    @staticmethod
    def has_image(config: MultimodalSourceConfig) -> bool:
        return "image" in config.task

    def get_data(self, source_idx: int, entry: dict[Any, Any]) -> Any:
        return None

    def format_example_dict(self, source_idx: int, entry: dict[Any, Any], data: Any) -> dict[str, Any]:
        config = self.configs[source_idx]

        item: dict[str, Any] = {
            "text": entry[config.text_column],
            "is_modal_synthesis": config.is_modal_synthesis,
            "source": source_idx,
        }

        if self.has_voice(config):
            item["voice"] = np.asarray(entry[config.voice_column])

        if self.has_image(config):
            item["image"] = np.asarray(entry[config.image_column], dtype=np.float32).reshape(
                entry[config.image_shape_column]
            )

        return item

    def text_lengths(self, source_idx: int) -> np.ndarray:
        config = self.configs[source_idx]
        if config.text_tokens_column is not None:
            return self.column(source_idx, config.text_tokens_column).to_numpy().astype(np.int64)

        characters = pc.utf8_length(self.column(source_idx, config.text_column)).to_numpy(zero_copy_only=False)
        return np.ceil(characters / self.chars_per_token).astype(np.int64)

    def media_lengths(self, source_idx: int) -> np.ndarray:
        config = self.configs[source_idx]
        n_rows = len(self.indexes[source_idx])
        # two delimiters around the stretched slots
        if self.has_voice(config):
            units = pc.list_value_length(self.column(source_idx, config.voice_column)).to_numpy(zero_copy_only=False)
            return np.ceil(units / self.voice_units_per_position).astype(np.int64) + 2

        if self.has_image(config):
            shapes = self.column(source_idx, config.image_shape_column)
            heights = pc.list_element(shapes, 1).to_numpy(zero_copy_only=False)
            widths = pc.list_element(shapes, 2).to_numpy(zero_copy_only=False)
            p = self.image_patch_size
            return ((heights // p) * (widths // p)).astype(np.int64) + 2

        return np.zeros(n_rows, dtype=np.int64)

    def source_ids_and_lengths(self) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = []
        lengths = []
        for s in range(len(self.configs)):
            # text, the media slots, and <eos>
            row_lengths = self.text_lengths(s) + self.media_lengths(s) + 1
            lengths.append(torch.from_numpy(row_lengths))
            source_ids.append(torch.full((len(row_lengths),), s, dtype=torch.long))

        return torch.cat(source_ids), torch.cat(lengths)
