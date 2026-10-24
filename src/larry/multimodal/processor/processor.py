import dataclasses
import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from huggingface_hub import ModelHubMixin, constants, hf_hub_download

from larry.multimodal.config.processor_configs import LarryProcessorConfig
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer

type NPImage = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type NPAudio = np.ndarray[tuple[int], np.dtype[np.int32]]

TOKENIZER_FILE = "tokenizer.model"

EOS = 1
BOI = 2
IPH = 3
EOI = 4
BOV = 5
VPH = 6
EOV = 7


@dataclasses.dataclass(kw_only=True)
class Segment(ABC):
    modality_id: ClassVar[int]
    segment_type: ClassVar[str | None] = None


@dataclasses.dataclass(kw_only=True)
class TextSpan(Segment):
    modality_id: ClassVar[int] = 0
    segment_type: ClassVar[str | None] = "text"

    text_token_ids: list[int]


@dataclasses.dataclass(kw_only=True)
class ModalitySegment(Segment, ABC):
    eos_modality_id: ClassVar[int]

    text_idx: int = -1
    # length or (width, height)
    size: int | tuple[int, int]

    payload: NPImage | NPAudio

    @property
    def bos(self) -> str:
        return f"<|{self.segment_type}|>"

    @property
    @abstractmethod
    def bos_offset(self) -> int:
        ...

    @property
    def eos(self) -> str:
        return f"<|/{self.segment_type}|>"

    @property
    @abstractmethod
    def eos_offset(self) -> int:
        ...

    @classmethod
    def placeholder(cls) -> str:
        return f"<|{cls.segment_type}_placeholder|>"

    @property
    @abstractmethod
    def placeholder_offset(self) -> int:
        ...


@dataclasses.dataclass(kw_only=True)
class ImagePlaceholder(ModalitySegment):
    modality_id: ClassVar[int] = 1
    eos_modality_id: ClassVar[int] = TextSpan.modality_id
    segment_type: ClassVar[str | None] = "image"

    @property
    def bos_offset(self) -> int:
        return BOI

    @property
    def eos_offset(self) -> int:
        return EOI

    @property
    def placeholder_offset(self) -> int:
        return IPH


@dataclasses.dataclass(kw_only=True)
class VoicePlaceholder(ModalitySegment):
    modality_id: ClassVar[int] = 2
    eos_modality_id: ClassVar[int] = 2
    segment_type: ClassVar[str | None] = "voice"

    @property
    def bos_offset(self) -> int:
        return BOV

    @property
    def eos_offset(self) -> int:
        return EOV

    @property
    def placeholder_offset(self) -> int:
        return VPH


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalInput:
    input_placeholders: list[Segment]
    text_token_ids: list[int]
    modality_ids: list[int]


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalPretrainingInput(LarryMultimodalInput):
    loss_mask: list[bool]


def _process_audio(audio_example: list[int] | NPAudio | torch.Tensor) -> VoicePlaceholder:
    unit_ids: NPAudio
    if isinstance(audio_example, np.ndarray):
        if audio_example.ndim == 1:
            unit_ids = audio_example
        elif audio_example.ndim == 2:
            if audio_example.shape[0] != 1:
                raise ValueError(f"Batch of audio passed to _process_audio; don't do this.")
            unit_ids = audio_example[0]
        else:
            raise ValueError(f"Unsupported audio shape for np.ndarray audio input: {audio_example.shape}")
    elif isinstance(audio_example, torch.Tensor):
        if audio_example.ndim == 1:
            unit_ids = audio_example.long().cpu().numpy()
        elif audio_example.ndim == 2:
            if audio_example.shape[0] != 1:
                raise ValueError(f"Batch of audio passed to _process_audio; don't do this.")
            unit_ids = audio_example[0].long().cpu().numpy()
        else:
            raise ValueError(f"Unsupported audio shape for torch.Tensor audio input: {audio_example.shape}")
    elif isinstance(audio_example, list):
        return _process_audio(np.array(audio_example))
    else:
        raise ValueError(f"Unsupported input audio type={type(audio_example).__name__}")

    return VoicePlaceholder(payload=unit_ids.astype(np.long), size=len(unit_ids))


def _process_image(image: Image.Image) -> ImagePlaceholder:
    image = image.convert("RGB")
    # width, height
    image_size = image.size
    np_image = TVF.pil_to_tensor(image).permute(1, 2, 0).numpy()
    assert np_image.ndim == 3, f"Expected image to have shape (H, W, 3), got {np_image.shape}"
    assert np_image.shape[2] == 3, f"Expected image to have shape (H, W, 3), got {np_image.shape}"
    return ImagePlaceholder(payload=np_image.astype(np.float32) / 255.0, size=image_size)


def _split_segments(
        cls: type[ModalitySegment],
        inputs: list[str | ModalitySegment],
        examples: Sequence[ModalitySegment],
) -> list[str | ModalitySegment]:
    segments = []
    pending = list(examples)
    for item in inputs:
        if not isinstance(item, str):
            segments.append(item)
            continue

        splits = item.split(cls.placeholder())

        for i, split in enumerate(splits):
            if split:
                segments.append(split)

            if i < len(splits) - 1:
                if not pending:
                    raise ValueError(f"more {cls.segment_type} placeholders than examples")

                segments.append(pending.pop(0))

    if pending:
        raise ValueError(f"{len(pending)} {cls.segment_type} examples have no placeholder")

    return segments


def _process_sequence(
        text_tokenizer: SentencePieceTokenizer,
        text: str,
        image_examples: list[ImagePlaceholder],
        voice_examples: list[VoicePlaceholder],
) -> tuple[list[Segment], list[int], list[int]]:
    raw_segments = _split_segments(ImagePlaceholder, [text], image_examples)
    raw_segments = _split_segments(VoicePlaceholder, raw_segments, voice_examples)

    base = text_tokenizer.vocab_size()
    segments: list[Segment] = []
    text_token_ids: list[int] = []
    modality_ids: list[int] = []
    for raw_segment in raw_segments:
        if isinstance(raw_segment, str):
            token_ids = text_tokenizer.encode(raw_segment)
            segments.append(TextSpan(text_token_ids=token_ids))
            text_token_ids += token_ids
            modality_ids += [TextSpan.modality_id] * len(token_ids)
        else:
            raw_segment.text_idx = len(text_token_ids) + 1
            segments.append(raw_segment)
            text_token_ids += [
                base + raw_segment.bos_offset,
                base + raw_segment.placeholder_offset,
                base + raw_segment.eos_offset,
            ]
            modality_ids += [TextSpan.modality_id, raw_segment.modality_id, raw_segment.eos_modality_id]

    text_token_ids.append(base + EOS)
    modality_ids.append(TextSpan.modality_id)
    return segments, text_token_ids, modality_ids


class LarryProcessorBase(ModelHubMixin):
    def __init__(self, config: LarryProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.text_tokenizer = SentencePieceTokenizer(config.text_tokenizer_model_file_path)

    def _save_pretrained(self, save_directory: Path) -> None:
        source = Path(self.config.text_tokenizer_model_file_path)
        target = save_directory / TOKENIZER_FILE
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)

        config = dataclasses.replace(self.config, text_tokenizer_model_file_path=TOKENIZER_FILE)
        (save_directory / constants.CONFIG_NAME).write_text(json.dumps(dataclasses.asdict(config), indent=2))

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: str | None,
            cache_dir: str | Path | None,
            force_download: bool,
            local_files_only: bool,
            token: str | bool | None,
            config: LarryProcessorConfig,
            **kwargs: Any,
    ) -> Self:
        if Path(model_id).is_dir():
            tokenizer_file = str(Path(model_id) / config.text_tokenizer_model_file_path)
        else:
            tokenizer_file = hf_hub_download(
                repo_id=model_id,
                filename=config.text_tokenizer_model_file_path,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        return cls(dataclasses.replace(config, text_tokenizer_model_file_path=tokenizer_file), **kwargs)


class LarryMultimodalProcessor(LarryProcessorBase):
    def encode_batch(
            self,
            text_batch: list[str],
            image_batch_examples: list[list[Image.Image] | None],
            voice_batch_examples: list[list[list[int] | NPAudio | torch.Tensor] | None],
    ) -> list[LarryMultimodalInput]:
        assert len(text_batch) == len(voice_batch_examples) == len(image_batch_examples)

        batch = []
        for text, image_examples, voice_examples in zip(text_batch, image_batch_examples, voice_batch_examples):
            batch.append(self.encode(text, image_examples, voice_examples))

        return batch

    def encode(
            self,
            text: str,
            image_examples: list[Image.Image] | None = None,
            voice_examples: list[list[int] | NPAudio | torch.Tensor] | None = None,
    ) -> LarryMultimodalInput:
        processed_images: list[ImagePlaceholder] = []
        if image_examples:
            for processed_image in image_examples:
                processed_images.append(_process_image(processed_image))

        processed_voices: list[VoicePlaceholder] = []
        if voice_examples:
            for processed_voice in voice_examples:
                processed_voices.append(_process_audio(processed_voice))

        placeholders_sequence, text_token_ids, modality_ids = _process_sequence(
            self.text_tokenizer,
            text,
            processed_images,
            processed_voices,
        )

        return LarryMultimodalInput(
            input_placeholders=placeholders_sequence,
            text_token_ids=text_token_ids,
            modality_ids=modality_ids,
        )


class LarryMultimodalPretrainProcessor(LarryProcessorBase):
    def encode_batch(
            self,
            text_batch: list[str],
            is_synthesis_batch: list[bool],
            image_batch_examples: list[Image.Image | None],
            voice_batch_examples: list[list[int] | NPAudio | torch.Tensor | None],
    ) -> list[LarryMultimodalPretrainingInput]:
        assert len(text_batch) == len(is_synthesis_batch) == len(voice_batch_examples) == len(image_batch_examples)

        batch = []
        for t, i, v, is_synthesis in zip(text_batch, image_batch_examples, voice_batch_examples, is_synthesis_batch):
            batch.append(self.encode(t, i, v, is_modal_synthesis=is_synthesis))

        return batch

    def encode(
            self,
            text: str,
            image_example: Image.Image | None = None,
            voice_example: list[int] | NPAudio | torch.Tensor | None = None,
            is_modal_synthesis: bool = True,
    ) -> LarryMultimodalPretrainingInput:
        if any(cls.placeholder() in text for cls in [ImagePlaceholder, VoicePlaceholder]):
            raise ValueError(f"Don't pass text with modality special tokens to encode on {type(self).__name__}.")

        processed_images: list[ImagePlaceholder] = []
        if image_example is not None:
            processed_images.append(_process_image(image_example))

        processed_voices: list[VoicePlaceholder] = []
        if voice_example is not None:
            processed_voices.append(_process_audio(voice_example))

        if len(processed_images + processed_voices) > 1:
            raise ValueError(f"Pretraining doesn't support multiple media examples.")

        media = processed_images or processed_voices
        # no image or voice examples means text must be included in loss
        if media:
            ph = media[0].placeholder()
            if is_modal_synthesis:
                text += ph
            else:
                text = ph + text

        placeholders_sequence, text_token_ids, modality_ids = _process_sequence(
            self.text_tokenizer,
            text,
            processed_images,
            processed_voices,
        )

        text_in_loss = not (media and is_modal_synthesis)
        loss_mask = [
            text_in_loss if m == TextSpan.modality_id else is_modal_synthesis
            for m in modality_ids
        ]
        if media:
            segment = media[0]
            loss_mask[segment.text_idx - 1] = is_modal_synthesis
            loss_mask[segment.text_idx + 1] = is_modal_synthesis and segment.eos_modality_id == segment.modality_id
        loss_mask[-1] = True

        return LarryMultimodalPretrainingInput(
            input_placeholders=placeholders_sequence,
            text_token_ids=text_token_ids,
            loss_mask=loss_mask,
            modality_ids=modality_ids,
        )
