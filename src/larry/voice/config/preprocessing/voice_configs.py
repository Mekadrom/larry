import dataclasses
from typing import ClassVar

from larry.common.config.data.preprocessing.preprocessor_configs import PreprocessorConfig
from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class VoiceConfig(PreprocessorConfig):
    sample_rate: int = 16000


@dataclasses.dataclass(kw_only=True)
class AlignerConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[AlignerConfig]]

    stride: int = 320
    model_id: str = "facebook/wav2vec2-large-960h-lv60-self"
    sample_rate: int = 16000
    telomere_size: int = 400
    chunk_seconds: float = 30.0
    overlap_seconds: float = 3.0
    emission_batch_size: int = 8


@dataclasses.dataclass(kw_only=True)
class TranscriptConfig:
    REGISTRY: ClassVar[TypeRegistry[TranscriptConfig]]
    ...


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptConfig(TranscriptConfig):
    max_note_lines: int = 4
