from typing import Any

import torch
from datasets import Dataset

from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.config.data.voice_data_configs import ASRSourceConfig
from larry.voice.config.model.voice_model_configs import SpeechEncoderModelConfig
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedASRDataset
from larry.voice.model.speech_encoder import SpeechEncoderModel
from larry.voice.model.voice_models import SpeechEncoderModelOutput
from tests.helpers.audio import wav

LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CharTokenizer(TextTokenizer):
    def __init__(self, alphabet: str = LETTERS) -> None:
        self.alphabet = alphabet

    def encode(self, text: str) -> list[int]:
        return [self.alphabet.index(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.alphabet[i] for i in token_ids)

    def vocab_size(self) -> int:
        return len(self.alphabet)

    def id_to_piece(self, token_id: int) -> str:
        return self.alphabet[token_id]


def tiny_speech_encoder_config(vocab_size: int, **overrides: Any) -> SpeechEncoderModelConfig:
    return SpeechEncoderModelConfig(**{
        "d_model": 32,
        "n_heads": 4,
        "n_query_groups": 4,
        "rotary_embedding_dim": 8,
        "n_encoder1_layers": 2,
        "n_premerge_layers": 1,
        "n_encoder2_layers": 1,
        "vocab_size": vocab_size,
        **overrides,
    })


def tiny_speech_encoder(vocab_size: int, **overrides: Any) -> SpeechEncoderModel:
    torch.manual_seed(0)
    return SpeechEncoderModel(tiny_speech_encoder_config(vocab_size, **overrides))


def ctc_output(log_probs: torch.Tensor, frame_lengths: torch.Tensor) -> SpeechEncoderModelOutput:
    return SpeechEncoderModelOutput(
        logits=log_probs,
        ctc_log_probs={"encoder2_norm": log_probs},
        frame_lengths=frame_lengths,
        ctc_lengths=frame_lengths,
        unit_ids=torch.zeros(frame_lengths.shape[0], log_probs.shape[1], dtype=torch.long),
        mels=torch.zeros(frame_lengths.shape[0], 100, 4 * log_probs.shape[1]),
        mel_lengths=4 * frame_lengths,
    )


def toy_source() -> ASRSourceConfig:
    return ASRSourceConfig(name="toy", index_path="toy")


def clip_dataset(clips: list[tuple[str, float]], **kwargs: Any) -> IndexedASRDataset:
    index = Dataset.from_dict({
        "audio": [{"bytes": wav(seconds, 16000), "path": f"{chr(ord('a') + i)}.wav"}
                  for i, (_, seconds) in enumerate(clips)],
        "text_normalized": [text for text, _ in clips],
        "duration_s": [seconds for _, seconds in clips],
    })
    return IndexedASRDataset([toy_source()], [index], [None], sample_rate=16000, **kwargs)
