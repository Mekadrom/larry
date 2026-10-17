import dataclasses
import functools
from typing import Any

from larry.common.config.data.data_configs import DatasetMixtureConfig
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.config.data.voice_data_configs import SpeechMixtureConfig


@dataclasses.dataclass(kw_only=True)
class SpeakerLeakageProbesConfig:
    """Config for ``SpeakerLeakageProbesCallback``: which activations to probe, how often, and probe training."""

    probe_every: int
    layers: list[str] = dataclasses.field(default_factory=lambda: ["encoder1", "fsq", "encoder2_norm"])
    train_epochs: int = 1000
    validate_every: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-2
    mlp_inner_dim: int = 512
    min_clips: int = 10
    max_clips: int = 20
    validation_fraction: float = 0.15
    test_fraction: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        """Every speaker needs at least one fit, one validation, and one test clip."""
        if self.min_clips < 3:
            raise ValueError(f"min_clips must be at least 3 for a fit/validation/test split, got {self.min_clips}")


@dataclasses.dataclass(kw_only=True)
class VoiceTrainerConfig[M: DatasetMixtureConfig[Any] = DatasetMixtureConfig](LarryTrainerConfig[M]):
    """Superclass for voice trainer configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderDecoderTrainerConfig(VoiceTrainerConfig[SpeechMixtureConfig]):
    tokenizer_model_path: str
    speaker_mapping_path: str

    @classmethod
    def mixture_config_type(cls) -> type[SpeechMixtureConfig]:
        """Speech trainers parse ``dataset_config`` into a ``SpeechMixtureConfig``."""
        return SpeechMixtureConfig


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderTrainerConfig(SpeechEncoderDecoderTrainerConfig):
    """Config for ``SpeechTokenizerTrainer`` to configure the training and eval dataset sources."""

    speed_augment_factors: list[float] = dataclasses.field(default_factory=lambda: [1.0])

    speaker_leakage_steps: int | None = None
    speaker_leakage_probes_callback_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

    @functools.cached_property
    def speaker_leakage_probes_config(self) -> SpeakerLeakageProbesConfig | None:
        """Builds the probes callback config, or None when ``speaker_leakage_steps`` is unset."""
        if not self.speaker_leakage_steps:
            return None
        return SpeakerLeakageProbesConfig(
            probe_every=self.speaker_leakage_steps,
            **self.speaker_leakage_probes_callback_overrides
        )


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderTrainerConfig(SpeechEncoderDecoderTrainerConfig):
    """Config for ``SpeechDecoderTrainer``."""
