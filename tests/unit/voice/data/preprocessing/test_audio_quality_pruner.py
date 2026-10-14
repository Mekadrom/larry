import io
import math
import struct
import wave

import pytest

from larry.common.config.data.preprocessing.pipeline_config import Provenance, ProvenanceAncestry
from larry.voice.config.preprocessing.voice_pruner_configs import AudioQualityPrunerConfigForASR
from larry.voice.data.preprocessing.voice_pruners import AudioQualityPruner


def _tone_wav(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / sample_rate)) if (i // 1600) % 2 == 0 else 0)
        for i in range(int(seconds * sample_rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buffer.getvalue()


def _pruner() -> AudioQualityPruner:
    provenance = Provenance(id_columns=[], ancestry=ProvenanceAncestry(datasets=set(), models=set()))
    return AudioQualityPruner(provenance, AudioQualityPrunerConfigForASR())


@pytest.mark.parametrize("corrupt", [b"", b"\x00" * 64, _tone_wav()[:20]])
def test_undecodable_audio_is_pruned_instead_of_failing_the_step(corrupt: bytes) -> None:
    audios = [{"bytes": _tone_wav(), "path": "good.wav"}, {"bytes": corrupt, "path": "corrupt.wav"}]

    assert _pruner().keep_batched(audios) == [True, False]
