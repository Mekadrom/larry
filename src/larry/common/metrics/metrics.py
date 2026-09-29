import logging
from abc import ABC, abstractmethod
from typing import Callable, Any

import librosa
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from larry.utils.convert_np import make_np
from larry.voice.utils import mel_to_db

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    log.debug("Could not import wandb, hopefully you didn't intend to use it.")
    wandb = None


def _patch_image(
        img_array: np.ndarray,
        img_transform: Callable[[Image.Image], Image.Image],
) -> np.ndarray:
    if img_array.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {img_array.dtype}")
    if not (img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] in (3, 4))):
        raise ValueError(f"expected (H, W), (H, W, 3), or (H, W, 4); got {img_array.shape}")
    img = Image.fromarray(img_array)
    img = img_transform(img)
    return np.array(img)


def _patch_image_with_padded_label(
        img_array: np.ndarray,
        label: str,
        pad: int = 20,
        font_size: int = 8,
        pad_color: float | tuple[float, ...] | str = "white",
        font_color: float | tuple[float, ...] | str = "black"
) -> np.ndarray:
    def _patch(img: Image.Image) -> Image.Image:
        out = ImageOps.expand(img, border=(0, 0, 0, pad), fill=pad_color)
        font = ImageFont.load_default(size=font_size)
        x, y = img.width / 2, img.height + pad / 2
        ImageDraw.Draw(out).text((x, y), label, fill=font_color, font=font, anchor="mm")
        return out

    return _patch_image(img_array, _patch)


def _patch_image_with_label(
        img_array: np.ndarray,
        label: str,
        position: tuple[int, int] = (4, 4),
        font_size: int = 8,
        color: float | tuple[float, ...] | str = "black",
        anchor: str = "la",
) -> np.ndarray:
    def _patch(img: Image.Image) -> Image.Image:
        font = ImageFont.load_default(size=font_size)
        ImageDraw.Draw(img).text(position, label, fill=color, font=font, anchor=anchor)
        return img

    return _patch_image(img_array, _patch)


def patch_image_with_label(
        img_array: np.ndarray,
        label: str | None = None,
        label_kwargs: dict[str, Any] | None = None,
        label_padding_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    if label_kwargs is not None and label_padding_kwargs is not None:
        raise ValueError("Please only specify one of either label_kwargs or label_padding_kwargs.")
    img_array = make_np(img_array)
    if label is not None:
        if label_kwargs is not None:
            img_array = _patch_image_with_label(img_array, label, **label_kwargs)
        else:
            img_array = _patch_image_with_padded_label(img_array, label, **(label_padding_kwargs or {}))
    return img_array


def render_mels_fig(
        mels: torch.Tensor | np.ndarray,
        hop_length: int = 256,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        f_max: float = 8000,
        db_range: tuple = (-100.0, 20.0),
) -> plt.Figure:
    mels = mel_to_db(make_np(mels))
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mels,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        n_fft=n_fft,
        fmin=0,
        fmax=f_max,
        vmin=db_range[0],
        vmax=db_range[1],
        ax=ax,
    )
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.close(fig)
    return fig


def prepare_image_add_label(
        img_array: torch.Tensor | np.ndarray,
        label: str | None = None,
        label_kwargs: dict[str, Any] | None = None,
        label_padding_kwargs: dict[str, Any] | None = None
) -> np.ndarray:
    return patch_image_with_label(img_array, label, label_kwargs, label_padding_kwargs)


def prepare_audio_and_mels(
        audio_array: torch.Tensor | np.ndarray,
        mels: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, plt.Figure | None]:
    audio = make_np(audio_array)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[0] < audio.shape[1]:
            audio = audio.T
    return audio, (render_mels_fig(mels) if mels is not None else None)


class Metrics(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def add_scalar(self, tag: str, value: float, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_image(
            self,
            tag: str,
            img_array: torch.Tensor | np.ndarray,
            step: int,
            caption: str | None = None,
            caption_tag: str | None = None,
            label: str | None = None,
            label_kwargs: dict[str, Any] | None = None,
            label_padding_kwargs: dict[str, Any] | None = None,
            **kwargs
    ) -> None:
        ...

    @abstractmethod
    def add_audio(
            self,
            tag: str,
            audio_array: torch.Tensor | np.ndarray,
            step: int,
            sample_rate: int,
            mels: torch.Tensor | np.ndarray | None = None,
            mels_tag: str | None = None,
            caption: str | None = None,
            caption_tag: str | None = None,
            **kwargs
    ) -> None:
        ...

    @abstractmethod
    def add_figure(self, tag: str, figure: plt.Figure, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_text(self, tag: str, text: str, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def flush(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class TensorBoardMetrics(Metrics):
    def __init__(self, writer=None, log_dir: str | None = None) -> None:
        super().__init__()
        if writer is not None:
            self._writer = writer
        else:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int, **kwargs) -> None:
        self._writer.add_scalar(tag, value, step)

    def add_image(
            self,
            tag: str,
            img_array: torch.Tensor | np.ndarray,
            step: int,
            caption: str | None = None,
            caption_tag: str | None = None,
            label: str | None = None,
            label_kwargs: dict[str, Any] | None = None,
            label_padding_kwargs: dict[str, Any] | None = None,
            **kwargs
    ) -> None:
        img_array = prepare_image_add_label(img_array, label, label_kwargs, label_padding_kwargs)
        if caption is not None:
            self.add_text(caption_tag or f"{tag}/caption", caption, step)
        self._writer.add_image(tag, img_array, step, dataformats="HWC" if img_array.ndim == 3 else "HW")

    def add_audio(
            self,
            tag: str,
            audio_array: torch.Tensor | np.ndarray,
            step: int,
            sample_rate: int,
            mels: torch.Tensor | np.ndarray | None = None,
            mels_tag: str | None = None,
            caption: str | None = None,
            caption_tag: str | None = None,
            **kwargs
    ) -> None:
        audio_array, mels_fig = prepare_audio_and_mels(audio_array, mels)
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)
        if mels_fig is not None:
            self.add_figure(mels_tag or f"{tag}/mels", mels_fig, step)
        if caption is not None:
            self.add_text(caption_tag or f"{tag}/caption", caption, step)
        # technically this accepts np.ndarray and even uses ``make_np()``
        # noinspection PyTypeChecker
        self._writer.add_audio(tag, audio_array, step, sample_rate=sample_rate)

    def add_figure(self, tag: str, figure: plt.Figure, step: int, **kwargs) -> None:
        self._writer.add_figure(tag, figure, step)

    def add_text(self, tag: str, text: str, step: int, **kwargs) -> None:
        self._writer.add_text(tag, text, step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        self._writer.add_histogram(tag, values, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandBMetrics(TensorBoardMetrics):
    def __init__(self, log_dir: str | None = None, **wandb_kwargs) -> None:
        if wandb is None:
            raise ImportError("WandBMetrics requires wandb; pip install wandb")

        wandb.init(sync_tensorboard=True, **wandb_kwargs)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        super().__init__(log_dir=log_dir)

    @staticmethod
    def _log(data: dict[str, Any], step: int) -> None:
        # noinspection PyArgumentList
        wandb.log({**data, "global_step": step})

    def add_image(
            self,
            tag: str,
            img_array: torch.Tensor | np.ndarray,
            step: int,
            caption: str | None = None,
            caption_tag: str | None = None,
            label: str | None = None,
            label_kwargs: dict[str, Any] | None = None,
            label_padding_kwargs: dict[str, Any] | None = None,
            **kwargs
    ) -> None:
        img_array = prepare_image_add_label(img_array, label, label_kwargs, label_padding_kwargs)
        self._log({tag: wandb.Image(img_array, caption=caption)}, step)

    def add_audio(
            self,
            tag: str,
            audio_array: torch.Tensor | np.ndarray,
            step: int,
            sample_rate: int,
            mels: torch.Tensor | np.ndarray | None = None,
            mels_tag: str | None = None,
            caption: str | None = None,
            caption_tag: str | None = None,
            **kwargs
    ) -> None:
        audio_array, mels_fig = prepare_audio_and_mels(audio_array, mels)
        audio = wandb.Audio(audio_array, sample_rate=sample_rate, caption=caption)

        data: dict[str, Any] = {tag: audio}

        if mels_fig is not None:
            data[mels_tag or f"{tag}/mels"] = wandb.Image(mels_fig, caption=caption)

        self._log(data, step)

    def close(self) -> None:
        super().close()
        wandb.finish()


class NoneMetrics(Metrics):
    """For processes on devices when world_size > 1. Logging should only occur after the gather."""

    def add_scalar(self, tag: str, value: float, step: int, **kwargs) -> None:
        pass

    def add_image(
            self,
            tag: str,
            img_array: torch.Tensor | np.ndarray,
            step: int,
            caption: str | None = None,
            caption_tag: str | None = None,
            label: str | None = None,
            label_kwargs: dict[str, Any] | None = None,
            label_padding_kwargs: dict[str, Any] | None = None,
            **kwargs
    ) -> None:
        pass

    def add_audio(
            self,
            tag: str,
            audio_array: torch.Tensor | np.ndarray,
            step: int,
            sample_rate: int,
            mels: torch.Tensor | np.ndarray | None = None,
            mels_tag: str | None = None,
            caption: str | None = None,
            caption_tag: str | None = None,
            **kwargs
    ) -> None:
        pass

    def add_figure(self, tag: str, figure: plt.Figure, step: int, **kwargs) -> None:
        pass

    def add_text(self, tag: str, text: str, step: int, **kwargs) -> None:
        pass

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
