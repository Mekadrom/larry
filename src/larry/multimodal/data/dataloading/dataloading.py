from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
from larry.multimodal.processor.processor import (
    ImagePlaceholder,
    LarryMultimodalPretrainProcessor,
    ModalitySegment,
    VoicePlaceholder,
)


class PretrainingMultimodalCollator:
    def __init__(self, config: PretrainingMultimodalCollatorConfig) -> None:
        self.config = config
        self.processor: LarryMultimodalPretrainProcessor = LarryMultimodalPretrainProcessor.from_pretrained(
            config.processor_model_path
        )

    def n_positions(self, segment: ModalitySegment) -> int:
        if isinstance(segment, ImagePlaceholder):
            width, height = segment.size
            patch_w, patch_h = self.config.image_patch_size
            return (width // patch_w) * (height // patch_h)
        return segment.size

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = []
        modality_ids = []
        loss_mask = []
        voice_unit_ids = []
        images = []
        image_sizes = []
        source_ids = []
        is_synthesis_mask = []

        for item in items:
            is_modal_synthesis = item.get("is_modal_synthesis", True)
            encoded = self.processor.encode(
                item["text"], item.get("image"), item.get("voice"), is_modal_synthesis=is_modal_synthesis
            )
            ids = encoded.text_token_ids[:]
            modalities = encoded.modality_ids[:]
            mask = encoded.loss_mask[:]

            row_units = torch.zeros(0, dtype=torch.long)
            row_image = torch.zeros(3, 0, 0)
            row_image_size = (0, 0)
            segments = [s for s in encoded.input_placeholders if isinstance(s, ModalitySegment)]
            for segment in reversed(segments):
                i, n = segment.text_idx, self.n_positions(segment)
                ids[i:i + 1] = [ids[i]] * n
                modalities[i:i + 1] = [modalities[i]] * n
                mask[i:i + 1] = [mask[i]] * n
                if isinstance(segment, VoicePlaceholder):
                    row_units = torch.from_numpy(segment.payload).long()
                else:
                    row_image = torch.from_numpy(segment.payload).permute(2, 0, 1)
                    row_image_size = segment.size

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            modality_ids.append(torch.tensor(modalities, dtype=torch.long))
            loss_mask.append(torch.tensor(mask, dtype=torch.bool))
            voice_unit_ids.append(row_units)
            images.append(row_image)
            image_sizes.append(row_image_size)
            source_ids.append(item["source"])
            is_synthesis_mask.append(is_modal_synthesis)

        # images are pixel-space; the stretched input ids hold one position per patch of the patchified image
        max_w = max(w for w, _ in image_sizes)
        max_h = max(h for _, h in image_sizes)
        padded_images = torch.zeros(len(items), 3, max_h, max_w)
        for row, image in enumerate(images):
            padded_images[row, :, :image.shape[1], :image.shape[2]] = image

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "input_lengths": torch.tensor([len(ids) for ids in input_ids]),
            "modality_ids": pad_sequence(modality_ids, batch_first=True),
            "loss_mask": pad_sequence(loss_mask, batch_first=True),
            "voice_unit_ids": pad_sequence(voice_unit_ids, batch_first=True),
            "voice_unit_id_lengths": torch.tensor([len(units) for units in voice_unit_ids]),
            "images": padded_images,
            "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
            "is_synthesis_mask": torch.tensor(is_synthesis_mask, dtype=torch.bool),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
        }
