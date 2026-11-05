import dataclasses

from larry.common.config.config import TaggedConfig


@dataclasses.dataclass(kw_only=True)
class SpeechRecognizerConfig(TaggedConfig):
    """Base config for ``SpeechRecognizer`` implementations; ``name`` labels its metrics."""

    name: str
    sample_rate: int = 16000
    batch_size: int = 16


@dataclasses.dataclass(kw_only=True)
class LarrySpeechEncoderRecognizerConfig(SpeechRecognizerConfig):
    """Greedy CTC transcription with a larry speech encoder and the SentencePiece model packaged beside it."""

    name: str = "larry-speech-encoder"
    model_path: str = "thelarryproject/larry-speech-encoder"
    tokenizer_file: str = "larry-ctc-sp512.model"
    revision: str | None = None


@dataclasses.dataclass(kw_only=True)
class Wav2Vec2RecognizerConfig(SpeechRecognizerConfig):
    """Greedy CTC transcription with a huggingface ``Wav2Vec2ForCTC`` model."""

    name: str = "wav2vec2-large-960h-lv60-self"
    model_id: str = "facebook/wav2vec2-large-960h-lv60-self"
    revision: str | None = None
