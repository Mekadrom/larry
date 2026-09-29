from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from larry.voice.utils import mel_to_db


class Metrics(ABC):
    @abstractmethod
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        ...

    @abstractmethod
    def add_image(self, tag: str, img_array: np.ndarray, step: int) -> None:
        ...

    @abstractmethod
    def add_audio(
            self,
            tag: str,
            audio_array: np.ndarray,
            step: int,
            sample_rate: int,
            mels: np.ndarray | None = None,
            mels_tag: str | None = None,
            **kwargs
    ) -> None:
        ...

    @abstractmethod
    def add_figure(self, tag: str, figure: Figure, step: int) -> None:
        ...

    @abstractmethod
    def add_text(self, tag: str, text: str, step: int) -> None:
        ...

    @abstractmethod
    def add_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        ...

    @abstractmethod
    def flush(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class TensorBoardMetrics(Metrics):
    def __init__(self, writer=None, log_dir: str | None = None) -> None:
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
            img_array: np.ndarray,
            step: int,
            label: str | None = None,
            padded_label: str | None = None,
            **kwargs
    ) -> None:
        if label is not None:
            img_array = self._patch_image_with_label(img_array, label)
        if padded_label is not None:
            img_array = self._patch_image_with_padded_label(img_array, padded_label)
        self._writer.add_image(tag, img_array, step)

    def add_audio(
            self,
            tag: str,
            audio_array: np.ndarray,
            step: int,
            sample_rate: int,
            mels: np.ndarray | None = None,
            mels_tag: str | None = None,
            **kwargs
    ) -> None:
        if mels is not None:
            self.add_figure(mels_tag, self._render_mels_chart(mels), step, **kwargs)
        # technically this accepts np.ndarray
        # noinspection PyTypeChecker
        self._writer.add_audio(tag, audio_array, step, sample_rate=sample_rate)

    def add_figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        self._writer.add_figure(tag, figure, step)

    def add_text(self, tag: str, text: str, step: int, **kwargs) -> None:
        self._writer.add_text(tag, text, step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        self._writer.add_histogram(tag, values, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    @staticmethod
    def _patch_image_with_padded_label(
            img_array: np.ndarray,
            label: str,
            pad: int = 20,
            font_size: int = 8,
            pad_color: float | tuple[float, ...] | str = "white",
            font_color: float | tuple[float, ...] | str = "black"
    ) -> np.ndarray:
        if img_array.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {img_array.dtype}")
        if not (img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] in (3, 4))):
            raise ValueError(f"expected (H, W), (H, W, 3), or (H, W, 4); got {img_array.shape}")
        img = Image.fromarray(img_array)
        out = ImageOps.expand(img, border=(0, 0, 0, pad), fill=pad_color)
        draw = ImageDraw.Draw(out)
        font = ImageFont.load_default(size=font_size)
        cx = img.width / 2
        cy = img.height + pad / 2
        draw.text((cx, cy), label, fill=font_color, font=font, anchor="mm")
        return np.array(img)

    @staticmethod
    def _patch_image_with_label(
            img_array: np.ndarray,
            label: str,
            position: tuple[int, int] = (4, 4),
            font_size: int = 8,
            color: float | tuple[float, ...] | str = "black",
            anchor: str = "la",
    ) -> np.ndarray:
        if img_array.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {img_array.dtype}")
        if not (img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] in (3, 4))):
            raise ValueError(f"expected (H, W), (H, W, 3), or (H, W, 4); got {img_array.shape}")
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=font_size)
        draw.text(position, label, fill=color, font=font, anchor=anchor)
        return np.array(img)

    @staticmethod
    def _render_mels_chart(
            mels: torch.Tensor | np.ndarray,
            hop_length: int = 256,
            sample_rate: int = 16000,
            n_fft: int = 1024,
            f_max: float = 8000,
            db_range: tuple = (-100.0, 20.0),
    ) -> plt.Figure:
        if isinstance(mels, torch.Tensor):
            mels = mels.detach().cpu().numpy()
        mel_db = mel_to_db(mels)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_db,
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


class NoneMetrics:
    def add_scalar(self, tag: str, value: float, step: int, **kwargs) -> None:
        pass

    def add_image(self, tag: str, img_array: np.ndarray, step: int, **kwargs) -> None:
        pass

    def add_audio(self, tag: str, audio_array: np.ndarray, step: int, sample_rate: int, **kwargs) -> None:
        pass

    def add_figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        pass

    def add_text(self, tag: str, text: str, step: int, **kwargs) -> None:
        pass

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
