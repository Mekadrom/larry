import dataclasses


@dataclasses.dataclass(kw_only=True)
class VoiceConfig:
    sample_rate: int = 16000
