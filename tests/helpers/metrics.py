from typing import Any

import torch

from larry.common.metrics.metrics import Metrics


class NullMetrics(Metrics):
    def add_scalar(self, *args: Any, **kwargs: Any) -> None: ...

    def add_image(self, *args: Any, **kwargs: Any) -> None: ...

    def add_audio(self, *args: Any, **kwargs: Any) -> None: ...

    def add_figure(self, *args: Any, **kwargs: Any) -> None: ...

    def add_grad_norms(self, *args: Any, **kwargs: Any) -> None: ...

    def add_histogram(self, *args: Any, **kwargs: Any) -> None: ...

    def add_hparams(self, *args: Any, **kwargs: Any) -> None: ...

    def add_text(self, *args: Any, **kwargs: Any) -> None: ...

    def close(self) -> None: ...

    def flush(self) -> None: ...


class RecordingMetrics(NullMetrics):
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        self.scalars: dict[str, float] = {}
        self.audio: list[tuple[str, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.figures: list[tuple[str, int]] = []

    def add_scalar(self, tag: str, value: float, step: int, **kwargs: Any) -> None:
        self.scalars[tag] = value

    def add_audio(self, tag: str, audio_array: torch.Tensor, step: int, sample_rate: int, **kwargs: Any) -> None:
        self.audio.append((tag, step))

    def add_text(self, tag: str, text: str, step: int, **kwargs: Any) -> None:
        self.text.append((tag, text, step))

    def add_figure(self, tag: str, figure: Any, step: int, **kwargs: Any) -> None:
        self.figures.append((tag, step))
