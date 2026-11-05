import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

from larry.voice.config.evaluation.recognizer_configs import (
    LarrySpeechEncoderRecognizerConfig,
    SpeechRecognizerConfig,
    Wav2Vec2RecognizerConfig,
)
from larry.voice.config.training.voice_trainer_configs import VocoderVizConfig
from larry.voice.evaluation.intelligibility import IntelligibilityScorer
from larry.voice.evaluation.speech_recognition import (
    LarrySpeechEncoderRecognizer,
    SpeechRecognizer,
    Wav2Vec2Recognizer,
)
from larry.voice.training.voice_viz_callbacks import VocoderVizCallback
from tests.helpers.voice import CharTokenizer


@dataclasses.dataclass(kw_only=True)
class EchoRecognizerConfig(SpeechRecognizerConfig):
    name: str = "echo"
    batch_size: int = 2


class EchoRecognizer(SpeechRecognizer[EchoRecognizerConfig]):
    def __init__(self, config: EchoRecognizerConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.batches: list[tuple[tuple[int, ...], list[int]]] = []

    def transcribe_batch(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> list[str]:
        self.batches.append((tuple(waveforms.shape), waveform_lengths.tolist()))
        return [f"LEN {n}" for n in waveform_lengths.tolist()]


def test_implementations_are_found_from_their_configs() -> None:
    assert SpeechRecognizer.REGISTRY.for_config(LarrySpeechEncoderRecognizerConfig()) is LarrySpeechEncoderRecognizer
    assert SpeechRecognizer.REGISTRY.for_config(Wav2Vec2RecognizerConfig()) is Wav2Vec2Recognizer


def test_a_new_recognizer_needs_only_its_class_and_config() -> None:
    recognizer = SpeechRecognizer.from_config(EchoRecognizerConfig(), torch.device("cpu"))

    assert isinstance(recognizer, EchoRecognizer) and recognizer.name == "echo"


def test_transcription_batches_pads_and_resamples_to_the_recognizers_rate() -> None:
    recognizer = EchoRecognizer(EchoRecognizerConfig(), torch.device("cpu"))
    clips = [torch.randn(32000), torch.randn(16000), torch.randn(8000)]

    transcripts = recognizer.transcribe(clips, sample_rate=32000)

    assert transcripts == ["LEN 16000", "LEN 8000", "LEN 4000"]
    assert recognizer.batches == [((2, 16000), [16000, 8000]), ((1, 4000), [4000])]


def test_the_larry_recognizer_greedily_decodes_ctc_frames() -> None:
    recognizer = object.__new__(LarrySpeechEncoderRecognizer)
    recognizer.tokenizer = CharTokenizer()
    blank = recognizer.tokenizer.vocab_size()
    h, i = recognizer.tokenizer.encode("HI")
    (a,) = recognizer.tokenizer.encode("A")
    frames = torch.full((2, 6), blank)
    frames[0, :5] = torch.tensor([h, h, blank, i, i])
    frames[1, :2] = torch.tensor([a, a])
    logits = torch.nn.functional.one_hot(frames, blank + 1).float()
    recognizer.model = type("Model", (), {
        "config": SimpleNamespace(vocab_size=blank),
        "__call__": lambda self, w, n: SimpleNamespace(logits=logits, ctc_lengths=torch.tensor([5, 1])),
    })()

    assert recognizer.transcribe_batch(torch.zeros(2, 100), torch.tensor([100, 20])) == ["HI", "A"]


@pytest.fixture
def wav2vec2(tmp_path: Path) -> Wav2Vec2Recognizer:
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    tokenizer = Wav2Vec2CTCTokenizer(str(tmp_path / "vocab.json"))
    extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True, do_normalize=True)
    torch.manual_seed(0)
    model = Wav2Vec2ForCTC(Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2, feat_extract_norm="layer", do_stable_layer_norm=True,
    )).eval()
    recognizer = object.__new__(Wav2Vec2Recognizer)
    recognizer.config = Wav2Vec2RecognizerConfig(batch_size=4)
    recognizer.device = torch.device("cpu")
    recognizer.processor = Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)
    recognizer.model = model
    return recognizer


def test_the_wav2vec2_recognizer_transcribes_each_clip_of_a_padded_batch(wav2vec2: Wav2Vec2Recognizer) -> None:
    torch.manual_seed(1)
    clips = [torch.randn(4000), torch.randn(2500), torch.randn(3200)]

    batched = wav2vec2.transcribe(clips, sample_rate=16000)
    alone = [wav2vec2.transcribe([clip], sample_rate=16000)[0] for clip in clips]

    assert len(batched) == 3 and all(isinstance(text, str) for text in batched)
    assert batched[0] == alone[0]


def test_the_scorer_reports_both_error_rates_and_their_gap_per_group() -> None:
    scorer = IntelligibilityScorer(str.lower)
    references = ["the cat sat", "a dog ran", "hello there"]
    originals = ["the cat sat", "a dog ran", "hello where"]
    regenerated = ["the bat sat", "a dog", "hello where"]

    scores = scorer.compare(references, originals, regenerated, ["clean", "clean", "noisy"])

    assert scores["wer_original"] == pytest.approx(1 / 8)
    assert scores["wer_generated"] == pytest.approx(3 / 8)
    assert scores["wer_gap"] == pytest.approx(2 / 8)
    assert scores["wer_gap/clean"] == pytest.approx(2 / 6)
    assert scores["wer_gap/noisy"] == 0
    assert scores["cer_original/clean"] == 0


def test_viz_configs_choose_recognizers_by_type() -> None:
    config = VocoderVizConfig.from_dict({"recognizers": [
        {"type": "LarrySpeechEncoderRecognizerConfig", "model_path": "local/encoder"},
        {"type": "Wav2Vec2RecognizerConfig"},
    ]})

    assert [type(r) for r in config.recognizers] == [LarrySpeechEncoderRecognizerConfig, Wav2Vec2RecognizerConfig]
    assert config.recognizers[0].model_path == "local/encoder"
    assert VocoderVizConfig().recognizers == []


class Metrics:
    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars[tag] = value


def test_viz_scores_every_injected_recognizer_and_transcribes_originals_once(monkeypatch: pytest.MonkeyPatch) -> None:
    callback = object.__new__(VocoderVizCallback)
    callback.recognizers = [EchoRecognizer(EchoRecognizerConfig(), torch.device("cpu"))]
    callback.original_transcripts = {}
    callback.trainer = SimpleNamespace(metrics=Metrics())
    callback.__dict__["scorer"] = IntelligibilityScorer(str.lower)
    callback.__dict__["intelligibility_references"] = (["len 4", "len 3"], ["vctk", "scotus"])
    clips = ([torch.zeros(4), torch.zeros(3)], [torch.zeros(4), torch.zeros(2)])
    monkeypatch.setattr(callback, "regenerate", lambda model, device: clips)
    model: Any = SimpleNamespace(config=SimpleNamespace(sample_rate=16000))

    callback.log_intelligibility(10, model, torch.device("cpu"))
    callback.log_intelligibility(20, model, torch.device("cpu"))

    scalars = callback.trainer.metrics.scalars
    assert scalars["viz/intelligibility/echo/wer_original"] == 0
    assert scalars["viz/intelligibility/echo/wer_gap/scotus"] == pytest.approx(0.5)
    assert len(callback.recognizers[0].batches) == 3
