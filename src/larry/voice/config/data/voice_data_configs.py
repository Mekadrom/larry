import dataclasses

from larry.common.config.data.data_configs import DatasetMixtureConfig, DatasetSourceConfig, TextSourceConfig
from larry.voice.config.preprocessing.voice_configs import SampleRateConfig


@dataclasses.dataclass(kw_only=True)
class SpeechSourceConfig(DatasetSourceConfig):
    audio_column: str = "audio"
    duration_column: str = "duration_s"
    speaker_id_column: str | None = "speaker_id"  # where None means no speaker id column (single speaker source)
    speaker_namespace: str | None = None  # scope that speaker ids are unique within; None means the source name


@dataclasses.dataclass(kw_only=True)
class ASRSourceConfig(SpeechSourceConfig, TextSourceConfig):
    """A config dataclass for configuring a single dataset source."""

    text_column: str = "text_normalized"


@dataclasses.dataclass(kw_only=True)
class TTSSourceConfig(SpeechSourceConfig):
    unit_ids_column: str = "unit_ids"
    f0_column: str | None = "f0"
    aperiodicity_column: str | None = "aperiodicity"


@dataclasses.dataclass(kw_only=True)
class SpeechMixtureConfig[T: SpeechSourceConfig = SpeechSourceConfig](DatasetMixtureConfig[T], SampleRateConfig):
    """A mixture of speech sources, all resampled to one sample rate."""


@dataclasses.dataclass(kw_only=True)
class VocoderMixtureConfig(SpeechMixtureConfig[SpeechSourceConfig]):
    ...


@dataclasses.dataclass(kw_only=True)
class ASRMixtureConfig(SpeechMixtureConfig[ASRSourceConfig]):
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""


@dataclasses.dataclass(kw_only=True)
class TTSMixtureConfig(SpeechMixtureConfig[TTSSourceConfig]):
    """A config dataclass for defining the entire training and eval mixture of dataset sources and batch hyperparams."""
