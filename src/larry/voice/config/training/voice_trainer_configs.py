import dataclasses
import functools

from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.voice.data.dataloading.voice_dataloading import SpeechMixtureConfig, SpeechSourceConfig


@dataclasses.dataclass(kw_only=True)
class VoiceTrainerConfig(LarryTrainerConfig):
    ...


@dataclasses.dataclass(kw_only=True)
class SpeechTokenizerTrainerConfig(VoiceTrainerConfig):
    tokenizer_model_path: str

    @functools.cached_property
    def mixture_config(self) -> SpeechMixtureConfig:
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
