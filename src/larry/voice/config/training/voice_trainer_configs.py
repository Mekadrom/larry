import dataclasses
from typing import Any

from larry.common.config.config import Config
from larry.common.config.training.trainer_configs import LarryTrainerConfig, TextTokenizerConfig
from larry.voice.config.criterion.voice_criterion_configs import SpeechDecoderCriterionConfig, VocoderCriterionConfig
from larry.voice.config.data.voice_data_configs import (
    ASRMixtureConfig,
    SpeechMixtureConfig,
    TTSMixtureConfig,
    VocoderMixtureConfig,
)
from larry.voice.config.preprocessing.voice_configs import WorldConfig


@dataclasses.dataclass(kw_only=True)
class SpeakerLeakageProbesConfig(Config):
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
class VoiceTrainerConfig[M: SpeechMixtureConfig[Any] = SpeechMixtureConfig](LarryTrainerConfig[M]):
    """Superclass for voice trainer configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class MelAugmentConfig(Config):
    noise_p: float = 0.0
    noise_std: float = 0.1
    time_blur_p: float = 0.0
    time_blur_sigma: float = 1.0
    freq_blur_p: float = 0.0
    freq_blur_sigma: float = 1.0


@dataclasses.dataclass(kw_only=True)
class VocoderVizConfig(Config):
    examples_per_source: int = 2
    generation_steps: int = 32
    step_sweep: list[int] = dataclasses.field(default_factory=lambda: [4, 8, 16, 32])
    sway_coefficient: float | None = -1.0
    seed: int = 0
    world: WorldConfig = dataclasses.field(default_factory=WorldConfig)


@dataclasses.dataclass(kw_only=True)
class VocoderTrainerConfig(VoiceTrainerConfig[VocoderMixtureConfig]):
    """Config for ``VocoderTrainer``."""

    crop_seconds: float = 2.0
    crops_per_clip: int = 4
    clips_per_batch: int = 32
    criterion: VocoderCriterionConfig = dataclasses.field(default_factory=VocoderCriterionConfig)
    mel_augment: MelAugmentConfig = dataclasses.field(default_factory=MelAugmentConfig)
    viz: VocoderVizConfig = dataclasses.field(default_factory=VocoderVizConfig)


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderDecoderTrainerConfig[M: SpeechMixtureConfig[Any] = SpeechMixtureConfig](VoiceTrainerConfig[M]):
    speaker_mapping_path: str


@dataclasses.dataclass(kw_only=True)
class SpeechEncoderTrainerConfig(SpeechEncoderDecoderTrainerConfig[ASRMixtureConfig], TextTokenizerConfig):
    """Config for ``SpeechTokenizerTrainer`` to configure the training and eval dataset sources."""

    speed_augment_factors: list[float] = dataclasses.field(default_factory=lambda: [1.0])

    speaker_leakage_probes: SpeakerLeakageProbesConfig | None = None

    ctc_layer_loss_weights: dict[str, float] = dataclasses.field(default_factory=lambda: {
        "encoder2_norm": 1.0,
    })


@dataclasses.dataclass(kw_only=True)
class SpeechDecoderTrainerConfig(SpeechEncoderDecoderTrainerConfig[TTSMixtureConfig]):
    """Config for ``SpeechDecoderTrainer``."""

    criterion: SpeechDecoderCriterionConfig = dataclasses.field(default_factory=SpeechDecoderCriterionConfig)
