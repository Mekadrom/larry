import pytest

from larry.voice.config.preprocessing.voice_pruner_configs import AudioQualityPrunerConfigForASR
from larry.voice.data.preprocessing.voice_pruners import AudioQualityPruner
from tests.helpers.audio import wav
from tests.helpers.common import provenance

TONE = wav(1.0, 16000, amplitude=8000, frequency=220, gate=1600)


@pytest.mark.parametrize("corrupt", [b"", b"\x00" * 64, TONE[:20]])
def test_undecodable_audio_is_pruned_instead_of_failing_the_step(corrupt: bytes) -> None:
    audios = [{"bytes": TONE, "path": "good.wav"}, {"bytes": corrupt, "path": "corrupt.wav"}]

    assert AudioQualityPruner(provenance(), AudioQualityPrunerConfigForASR()).keep_batched(audios) == [True, False]
