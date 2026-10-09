import dataclasses
from typing import ClassVar

from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class TranscriptConfig:
    REGISTRY: ClassVar[TypeRegistry[TranscriptConfig]]
    ...


@dataclasses.dataclass(kw_only=True)
class SCOTUSTranscriptConfig(TranscriptConfig):
    max_note_lines: int = 4
