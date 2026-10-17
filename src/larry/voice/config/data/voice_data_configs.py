import dataclasses

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig
from larry.voice.config.preprocessing.voice_configs import SampleRateConfig


@dataclasses.dataclass(kw_only=True)
class SpeechSourceConfig(DatasetSourceConfig):
    audio_column: str = "audio"
    duration_column: str = "duration_s"
    speaker_id_column: str | None = "speaker_id"  # where None means no speaker id column (single speaker source)
    speaker_namespace: str | None = None  # scope that speaker ids are unique within; None means the source name


@dataclasses.dataclass(kw_only=True)
class ASRSourceConfig(SpeechSourceConfig):
    """A config dataclass for configuring a single dataset source."""

    text_column: str = "text_normalized"


@dataclasses.dataclass(kw_only=True)
class TTSSourceConfig(SpeechSourceConfig):
    token_ids_column: str = "token_ids"


@dataclasses.dataclass(kw_only=True)
class SpeechMixtureConfig(DatasetMixtureConfig[ASRSourceConfig], SampleRateConfig):
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""

    @classmethod
    def source_config_type(cls) -> type[ASRSourceConfig]:
        """Speech mixtures build their sources as ``SpeechSourceConfig`` instances."""
        return ASRSourceConfig
