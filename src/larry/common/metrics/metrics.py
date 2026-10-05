import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Iterable
from typing import Callable, Any, Literal, assert_never

import librosa
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from larry.common.utils.convert_np import make_np
from larry.voice.utils import mel_to_db

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    log.debug("Could not import wandb, hopefully you didn't intend to use it.")
    wandb = None

type Color = int | tuple[int, ...] | str


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
        pad_color: Color = "white",
        font_color: Color = "black"
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
        color: Color = "black",
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
) -> Figure:
    mels = mel_to_db(make_np(mels))
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        make_np(mels),
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
    return patch_image_with_label(make_np(img_array), label, label_kwargs, label_padding_kwargs)


def prepare_audio_and_mels(
        audio_array: torch.Tensor | np.ndarray,
        mels: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, Figure | None]:
    audio = make_np(audio_array)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[0] < audio.shape[1]:
            audio = audio.T
    return audio, (render_mels_fig(mels) if mels is not None else None)


@dataclasses.dataclass(kw_only=True)
class MetricEntry[V](ABC):
    tag: str

    @property
    @abstractmethod
    def value(self) -> V:
        ...


@dataclasses.dataclass(kw_only=True)
class TextMetricEntry(MetricEntry[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


@dataclasses.dataclass(kw_only=True)
class ScalarMetricEntry(MetricEntry[float | int]):
    scalar: float | int

    @property
    def value(self) -> float | int:
        return self.scalar


@dataclasses.dataclass(kw_only=True)
class CountableMetricEntry[V](MetricEntry[V], ABC):
    @property
    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def add_(self, other: Any) -> None:
        """In-place."""
        ...


@dataclasses.dataclass(kw_only=True)
class AggregateMetricEntry(CountableMetricEntry[float | int]):
    average_reduce: Literal["all", "none"] = "all"
    alert_naninf: bool = True

    total: float | int = 0.0
    n_items: int = 0

    @property
    def value(self) -> float | int:
        if self.average_reduce == "all":
            return float(self.total) / float(self.count)
        return self.total

    @property
    def count(self) -> int:
        return self.n_items

    def add_(self, other: Any) -> None:
        if isinstance(other, (float, int)):
            self.total += other
            self.n_items += 1
            return
        if isinstance(other, AggregateMetricEntry):
            self.total += other.total
            self.n_items += 1
            return
        raise NotImplementedError(f"Can't add other={other} to {type(self).__name__}={self}")


@dataclasses.dataclass(kw_only=True)
class BatchMetricEntry(MetricEntry[list], ABC):
    metrics_list: list = dataclasses.field(default_factory=list)

    @property
    def value(self) -> list:
        return self.metrics_list

    def add_(self, other: BatchMetricEntry) -> None:
        self.metrics_list.extend(other.metrics_list)


class ImageMetric(MetricEntry[torch.Tensor]):
    image: torch.Tensor

    @property
    def value(self) -> torch.Tensor:
        return self.image


@dataclasses.dataclass(kw_only=True)
class CaptionedImageMetric(MetricEntry[tuple[ImageMetric, str]]):
    image: ImageMetric
    caption: str

    @property
    def value(self) -> tuple[ImageMetric, str]:
        return self.image, self.caption


@dataclasses.dataclass(kw_only=True)
class AudioMetric(MetricEntry[torch.Tensor]):
    audio: torch.Tensor
    sample_rate: int

    @property
    def value(self) -> torch.Tensor:
        return self.audio


@dataclasses.dataclass(kw_only=True)
class CaptionedAudioMetric(MetricEntry[tuple[AudioMetric, str]]):
    audio: AudioMetric
    caption: str

    @property
    def value(self) -> tuple[AudioMetric, str]:
        return self.audio, self.caption


class MetricDict(UserDict[str, MetricEntry]):
    def __setitem__(self, key: str, item: MetricEntry) -> None:
        if not isinstance(item, MetricEntry):
            raise ValueError(
                f"MetricDict entries must be type MetricEntry, received type={type(item).__name__}"
            )

        if key in self.data:
            existing = self.data[key]
            if isinstance(item, BatchMetricEntry) and isinstance(existing, BatchMetricEntry):
                existing.add_(item)
            elif isinstance(item, CountableMetricEntry) and isinstance(existing, CountableMetricEntry):
                existing.add_(item)
            else:
                raise ValueError(
                    f"Unsupported metric type: {type(item.value).__name__} "
                    f"tried to merge with {type(existing).__name__} at key={key}"
                )
            return
        self.data[key] = copy.deepcopy(item)

    def add_metric(self, m: MetricEntry) -> None:
        self[m.tag] = m

    def add_metrics(self, l: Iterable[MetricEntry]) -> None:
        for m in l:
            self.add_metric(m)


class Metrics(ABC):
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

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
    def add_figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_text(self, tag: str, text: str | Any, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        ...

    @abstractmethod
    def add_hparams(
            self,
            hparams: dict[str, bool | str | float | int | None],
            metrics: dict[str, bool | str | float | int | None],
            step: int,
            hparam_domain_discrete: dict[str, list[Any]] | None = None,
            **kwargs
    ) -> None:
        ...

    @abstractmethod
    @torch.no_grad()
    def add_grad_norms(
            self,
            model: nn.Module,
            step: int,
            prefix: str = "grad_norm",
            per_param_scalars: bool = False
    ) -> None:
        ...

    def batch_metrics(self, step_metrics: MetricDict, step: int) -> None:
        for tag, metric in step_metrics.items():
            self.add_metric(tag, metric, step)

    def add_metric(self, tag: str, m: MetricEntry, step: int) -> None:
        if isinstance(m, BatchMetricEntry):
            for i, v in enumerate(m.value):
                self.add_metric(f"{tag}_{i}", v, step)
        elif isinstance(m, AggregateMetricEntry):
            self.add_scalar(tag, m.value, step)
        elif isinstance(m, CaptionedImageMetric):
            self.add_image(tag, m.image.image, step, caption=m.caption)
        elif isinstance(m, ImageMetric):
            self.add_image(tag, m.image, step)
        elif isinstance(m, CaptionedAudioMetric):
            self.add_audio(tag, m.audio.audio, step, sample_rate=m.audio.sample_rate, caption=m.caption)
        elif isinstance(m, MetricEntry):
            if isinstance(m.value, str):
                self.add_text(tag, m.value, step)
            elif isinstance(m.value, (float, int)):
                self.add_scalar(tag, m.value, step)
            elif isinstance(m.value, torch.Tensor):
                self.add_histogram(tag, m.value.cpu().numpy(), step)
            else:
                raise ValueError(f"Unsupported raw metric type: {type(m.value).__name__} for tag={tag}")
        else:
            assert_never(f"Unsupported metric type: {type(m).__name__} for tag={tag}")

    @abstractmethod
    def flush(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class TensorBoardMetrics(Metrics):
    def __init__(self, log_dir: str, writer=None) -> None:
        super().__init__(log_dir=log_dir)
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

    def add_figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        self._writer.add_figure(tag, figure, step)

    def add_text(self, tag: str, text: str | Any, step: int, **kwargs) -> None:
        self._writer.add_text(tag, str(text), step)

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        self._writer.add_histogram(tag, values, step)

    def add_hparams(
            self,
            hparams: dict[str, bool | str | float | int | None],
            metrics: dict[str, bool | str | float | int | None],
            step: int,
            hparam_domain_discrete: dict[str, list[Any]] | None = None,
            **kwargs
    ) -> None:
        self._writer.add_hparams(hparams, metrics, step, hparam_domain_discrete, **kwargs)

    @torch.no_grad()
    def add_grad_norms(
            self,
            model: nn.Module,
            step: int,
            prefix: str = "grad_norm",
            per_param_scalars: bool = False
    ) -> None:
        prefix = prefix.rstrip("/")
        names, norms = [], []
        for name, p in model.named_parameters():
            if p.grad is not None:
                names.append(name)
                norms.append(p.grad.detach().norm())
        if not norms:
            return
        norms = torch.stack(norms).float().cpu()  # single device sync
        self.add_scalar(f"{prefix}/total", norms.pow(2).sum().sqrt().item(), step)
        self.add_histogram(f"{prefix}/per_param", norms.numpy(), step)
        if per_param_scalars:
            for name, n in zip(names, norms.tolist()):
                self.add_scalar(f"{prefix}/{name}", n, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandBMetrics(TensorBoardMetrics):
    def __init__(self, log_dir: str, **wandb_kwargs) -> None:
        if wandb is None:
            raise ImportError("WandBMetrics requires wandb; pip install wandb")

        wandb.init(sync_tensorboard=True, **wandb_kwargs)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        super().__init__(log_dir=log_dir)

    @staticmethod
    def _log(data: dict[str, Any], step: int) -> None:
        if wandb is None:
            return
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
        if wandb is None:
            return
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
        if wandb is None:
            return

        audio_array, mels_fig = prepare_audio_and_mels(audio_array, mels)
        audio = wandb.Audio(audio_array, sample_rate=sample_rate, caption=caption)

        data: dict[str, Any] = {tag: audio}

        if mels_fig is not None:
            data[mels_tag or f"{tag}/mels"] = wandb.Image(mels_fig, caption=caption)

        self._log(data, step)

    def close(self) -> None:
        super().close()
        if wandb is None:
            return
        wandb.finish()


class NoneMetrics(Metrics):
    """For processes on devices when world_size > 1. Logging should only occur after the gather."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def add_figure(self, tag: str, figure: Figure, step: int, **kwargs) -> None:
        pass

    def add_text(self, tag: str, text: str | Any, step: int, **kwargs) -> None:
        pass

    def add_histogram(self, tag: str, values: np.ndarray, step: int, **kwargs) -> None:
        pass

    def add_hparams(
            self,
            hparams: dict[str, bool | str | float | int | None],
            metrics: dict[str, bool | str | float | int | None],
            step: int,
            hparam_domain_discrete: dict[str, list[Any]] | None = None,
            **kwargs
    ) -> None:
        pass

    def add_grad_norms(
            self,
            model: nn.Module,
            step: int,
            prefix: str = "grad_norm",
            per_param_scalars: bool = False
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
