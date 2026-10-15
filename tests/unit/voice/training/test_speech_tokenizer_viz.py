import io
import math
import struct
import wave
from types import SimpleNamespace
from typing import Any

import torch
from datasets import Dataset

from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.config.model.voice_model_configs import SpeechTokenizerModelConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.config.preprocessing.voice_configs import SpeechSourceConfig
from larry.voice.data.dataloading.voice_dataloading import SpeechCTCCollator
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.training.voice_viz_callbacks import SpeechTokenizerVizCallback, collapse, pick_rows, timeline

ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CharTokenizer(TextTokenizer):
    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in token_ids)

    def vocab_size(self) -> int:
        return len(ALPHABET)

    def id_to_piece(self, token_id: int) -> str:
        return ALPHABET[token_id]


class RecordingMetrics:
    def __init__(self) -> None:
        self.audio: list[tuple[str, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.figures: list[tuple[str, int]] = []

    def add_audio(self, tag: str, audio_array: torch.Tensor, step: int, sample_rate: int, **kwargs: Any) -> None:
        self.audio.append((tag, step))

    def add_text(self, tag: str, text: str, step: int, **kwargs: Any) -> None:
        self.text.append((tag, text, step))

    def add_figure(self, tag: str, figure: Any, step: int, **kwargs: Any) -> None:
        self.figures.append((tag, step))


def _wav(seconds: float, sample_rate: int) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 300 * i / sample_rate)))
        for i in range(int(seconds * sample_rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buffer.getvalue()


def _model() -> SpeechTokenizerModel:
    torch.manual_seed(0)
    return SpeechTokenizerModel(SpeechTokenizerModelConfig(
        d_model=32, n_heads=4, n_query_groups=4, rotary_embedding_dim=8,
        n_encoder1_layers=1, n_encoder2_layers=1, text_vocab_size=len(ALPHABET),
    ))


def test_frame_seconds_matches_the_encoder_frame_rate() -> None:
    model = _model().eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 32000), torch.tensor([32000]))

    assert model.encoder1.frame_seconds == 0.04
    assert abs(int(output.frame_lengths[0]) * model.encoder1.frame_seconds - 2.0) <= model.encoder1.frame_seconds + 1e-9


def test_rows_are_spread_over_each_source_and_never_repeat() -> None:
    assert pick_rows([0, 10, 11, 31], per_source=2) == [0, 5, 10, 11, 21]


def test_greedy_frames_collapse_and_time_each_new_token() -> None:
    blank = 0
    frames = torch.tensor([0, 3, 3, 0, 3, 5, 5, 0])
    tokenizer = CharTokenizer()

    assert collapse(frames, blank) == [3, 3, 5]
    assert timeline(frames, blank, tokenizer, 0.04) == "`0.04` C `0.16` C `0.20` E"
    assert timeline(None, blank, tokenizer, 0.04) == "(no alignment)"


def test_viz_logs_audio_once_and_alignments_for_every_example_each_time() -> None:
    index = Dataset.from_dict({
        "audio": [{"bytes": _wav(1.0, 16000), "path": "a.wav"}, {"bytes": _wav(1.5, 16000), "path": "b.wav"}],
        "text_normalized": ["HI", "OK GO"],
        "duration_s": [1.0, 1.5],
    })
    source = SpeechSourceConfig(name="toy", index_path="toy")
    dataset = IndexedSpeechDataset([source], [index], [None], sample_rate=16000)
    tokenizer = CharTokenizer()
    model = _model()
    metrics = RecordingMetrics()
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            is_main_process=True,
            device=torch.device("cpu"),
            unwrap_model=lambda m: m,
            autocast=torch.no_grad,
        ),
        eval_loader=SimpleNamespace(dataset=dataset),
        make_collate_fn=lambda: SpeechCTCCollator(tokenizer),
        model=model,
        tokenizer=tokenizer,
        model_config=model.config,
        config=SimpleNamespace(mixture_config=SimpleNamespace(eval_sources=[source], sample_rate=16000)),
        metrics=metrics,
    )
    callback = SpeechTokenizerVizCallback(viz_steps=10, examples_per_source=2)

    callback.viz(trainer, 0)  # pyright: ignore[reportArgumentType]
    callback.viz(trainer, 10)  # pyright: ignore[reportArgumentType]

    assert metrics.audio == [("viz/0/toy/audio", 0), ("viz/1/toy/audio", 0)]
    assert [(tag, step) for tag, _, step in metrics.text] == [
        ("viz/0/toy/ctc", 0), ("viz/1/toy/ctc", 0), ("viz/0/toy/ctc", 10), ("viz/1/toy/ctc", 10),
    ]
    assert len(metrics.figures) == 4
    assert "**target:** OK GO" in metrics.text[1][1]
    assert model.training
