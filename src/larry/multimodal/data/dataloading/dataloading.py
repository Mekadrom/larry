from typing import Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence

from larry.multimodal.config.data.dataloading.dataloading_configs import (
    InferenceMultimodalCollatorConfig,
    PretrainingMultimodalCollatorConfig,
)
from larry.multimodal.processor.processor import (
    LarryMultimodalPretrainProcessor,
    LarryMultimodalProcessor,
    LarryStretchedInput,
)


class MultimodalCollator:
    @staticmethod
    def tensors(rows: list[LarryStretchedInput]) -> dict[str, torch.Tensor]:
        voice_unit_ids = []
        images = []
        image_sizes = []
        for row in rows:
            if len(row.voices) > 1 or len(row.images) > 1:
                raise NotImplementedError("the model takes at most one voice and one image per row")

            if row.voices:
                units = torch.from_numpy(row.voices[0]).long()
            else:
                units = torch.zeros(0, dtype=torch.long)

            voice_unit_ids.append(units)

            if row.images:
                images.append(torch.from_numpy(row.images[0]))
            else:
                images.append(torch.zeros(0, 0, 0))

            if row.image_sizes:
                image_sizes.append(row.image_sizes[0])
            else:
                image_sizes.append((0, 0))

        # images are (C, H, W) pixels or VAE latents
        max_w = max(w for w, _ in image_sizes)
        max_h = max(h for _, h in image_sizes)
        channels = max(image.shape[0] for image in images)
        padded_images = torch.zeros(len(rows), channels, max_h, max_w)
        for i, image in enumerate(images):
            padded_images[i, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        input_ids = [torch.tensor(row.input_ids, dtype=torch.long) for row in rows]
        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "input_lengths": torch.tensor([len(ids) for ids in input_ids]),
            "modality_ids": pad_sequence([
                torch.tensor(row.modality_ids, dtype=torch.long)
                for row in rows
            ], batch_first=True),
            "voice_unit_ids": pad_sequence(voice_unit_ids, batch_first=True),
            "voice_unit_id_lengths": torch.tensor([len(units) for units in voice_unit_ids]),
            "images": padded_images,
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
        }
        if all(row.loss_mask is not None for row in rows):
            batch["loss_mask"] = pad_sequence(
                [torch.tensor(row.loss_mask, dtype=torch.bool) for row in rows], batch_first=True
            )

        return batch


class PretrainingMultimodalCollator(MultimodalCollator):
    def __init__(self, config: PretrainingMultimodalCollatorConfig) -> None:
        self.config = config
        self.processor: LarryMultimodalPretrainProcessor = LarryMultimodalPretrainProcessor.from_pretrained(
            config.processor_model_path
        )

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for item in items:
            encoded = self.processor.encode(
                item["text"], item.get("image"), item.get("voice"),
                is_modal_synthesis=item.get("is_modal_synthesis", True),
            )
            rows.append(self.processor.stretch(encoded))

        return {
            **self.tensors(rows),
            "is_synthesis_mask": torch.tensor([item.get("is_modal_synthesis", True) for item in items]),
            "source_ids": torch.tensor([item["source"] for item in items], dtype=torch.long),
        }


class InferenceMultimodalCollator(MultimodalCollator):
    def __init__(self, config: InferenceMultimodalCollatorConfig) -> None:
        self.config = config
        self.processor: LarryMultimodalProcessor = LarryMultimodalProcessor.from_pretrained(config.processor_model_path)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for item in items:
            open_modality: Literal["voice", "image"] | None = item.get("open_modality")
            encoded = self.processor.encode(
                item["text"], item.get("images"), item.get("voices"), add_eos=False, open_modality=open_modality
            )
            rows.append(self.processor.stretch(encoded))

        # prompt media are always inputs
        return {**self.tensors(rows), "is_synthesis_mask": torch.zeros(len(items), dtype=torch.bool)}

    @staticmethod
    def generate_kwargs(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor | None]:
        length = int(batch["input_lengths"][row])
        n_units = int(batch["voice_unit_id_lengths"][row])
        width, height = batch["image_sizes"][row].tolist()
        has_image = width > 0 and height > 0
        return {
            "input_ids": batch["input_ids"][row:row + 1, :length],
            "images": batch["images"][row:row + 1, :, :height, :width] if has_image else None,
            "image_sizes": batch["image_sizes"][row:row + 1] if has_image else None,
            "voice_unit_ids": batch["voice_unit_ids"][row:row + 1, :n_units] if n_units else None,
            "voice_unit_id_lengths": batch["voice_unit_id_lengths"][row:row + 1] if n_units else None,
        }
