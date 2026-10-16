import dataclasses
import functools

from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.config.preprocessing.voice_configs import SpeechMixtureConfig, SpeechSourceConfig


@dataclasses.dataclass(kw_only=True)
class VoiceTrainerConfig(LarryTrainerConfig):
    """Superclass for voice trainer configs."""

    ...


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerTrainerConfig(VoiceTrainerConfig):
    """Config for ``SpeechTokenizerTrainer`` to configure the training and eval dataset sources."""

    tokenizer_model_path: str

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
