import dataclasses

from larry.common.config.data.preprocessing.pruner_configs import PrunerConfig


@dataclasses.dataclass(kw_only=True)
class ImagePrunerConfig(PrunerConfig):
    """Superclass for image pruner configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class ImageAspectRatioPrunerConfig(ImagePrunerConfig):
    input_columns: list[str] = dataclasses.field(default_factory=lambda: ["image"])
    max_aspect_ratio: float = 4.0
