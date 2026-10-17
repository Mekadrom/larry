import dataclasses
import functools
from typing import Any

from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.config.preprocessing.voice_configs import SpeechMixtureConfig, SpeechSourceConfig


@dataclasses.dataclass(kw_only=True)
class SpeakerLeakageProbesConfig:
    """Config for ``SpeakerLeakageProbesCallback``: which activations to probe, how often, and probe training."""

    probe_every: int
    layers: list[str] = dataclasses.field(default_factory=lambda: ["encoder1", "fsq", "encoder2_norm"])
    train_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2
    mlp_inner_dim: int = 512
    min_clips: int = 10
    max_clips: int = 20
    test_fraction: float = 0.25
    seed: int = 42


@dataclasses.dataclass(kw_only=True)
class VoiceTrainerConfig(LarryTrainerConfig):
    """Superclass for voice trainer configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerTrainerConfig(VoiceTrainerConfig):
    """Config for ``SpeechTokenizerTrainer`` to configure the training and eval dataset sources."""

    tokenizer_model_path: str
    speaker_mapping_path: str

    speed_augment_factors: list[float] = dataclasses.field(default_factory=lambda: [
        1.0
    ])

    @functools.cached_property
    def mixture_config(self) -> SpeechMixtureConfig:
        """Converts the raw dictionary config from yaml to a nested structure of config dataclasses."""
        raw = dict(self.dataset_config)
        train_sources = [
            SpeechSourceConfig(**source)
            for source in raw.pop("train_sources")
        ]
        eval_sources = [
            SpeechSourceConfig(**source)
            for source in raw.pop("eval_sources")
        ]
        return SpeechMixtureConfig(
            train_sources=train_sources,
            eval_sources=eval_sources,
            **raw
        )

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
