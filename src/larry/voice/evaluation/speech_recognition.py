from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from larry.common.utils.registrable import Registrable
from larry.text.tokenizer.text_tokenizer import SentencePieceTokenizer
from larry.voice.config.evaluation.recognizer_configs import (
    LarrySpeechEncoderRecognizerConfig,
    SpeechRecognizerConfig,
    Wav2Vec2RecognizerConfig,
)
from larry.voice.model.speech_encoder import SpeechEncoderModel
from larry.voice.utils import collapse_ctc_emissions


class SpeechRecognizer[C: SpeechRecognizerConfig = SpeechRecognizerConfig](
    Registrable, ABC, root=True, config_root=SpeechRecognizerConfig
):
    """Transcribes mono waveforms to text; implementations are built from their config with ``from_config``."""

    def __init__(self, config: C, device: torch.device) -> None:
        self.config = config
        self.device = device

    @property
    def name(self) -> str:
        return self.config.name

    @classmethod
    def from_config(cls, config: SpeechRecognizerConfig, device: torch.device) -> SpeechRecognizer:
        return cls.REGISTRY.for_config(config)(config, device)

    @abstractmethod
    def transcribe_batch(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> list[str]:
        ...

    @torch.inference_mode()
    def transcribe(self, waveforms: list[torch.Tensor], sample_rate: int) -> list[str]:
        transcripts: list[str] = []
        for start in range(0, len(waveforms), self.config.batch_size):
            chunk = [
                self.resample(w.float().flatten(), sample_rate)
                for w in waveforms[start:start + self.config.batch_size]
            ]
            lengths = torch.tensor([len(w) for w in chunk], device=self.device)
            transcripts += self.transcribe_batch(pad_sequence(chunk, batch_first=True).to(self.device), lengths)

        return transcripts

    def resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.config.sample_rate:
            return waveform

        return AF.resample(waveform, sample_rate, self.config.sample_rate)


class LarrySpeechEncoderRecognizer(SpeechRecognizer[LarrySpeechEncoderRecognizerConfig]):
    def __init__(self, config: LarrySpeechEncoderRecognizerConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.model = SpeechEncoderModel.from_pretrained(config.model_path, revision=config.revision).to(device).eval()
        self.tokenizer = SentencePieceTokenizer(self.tokenizer_path())

    def tokenizer_path(self) -> str:
        local = Path(self.config.model_path).expanduser() / self.config.tokenizer_file
        if local.is_file():
            return str(local)

        return hf_hub_download(self.config.model_path, self.config.tokenizer_file, revision=self.config.revision)

    def transcribe_batch(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> list[str]:
        output = self.model(waveforms, waveform_lengths)
        greedy = output.logits.argmax(dim=-1).cpu()
        blank = self.model.config.vocab_size
        return [
            self.tokenizer.decode(collapse_ctc_emissions(greedy[i, :n], blank))
            for i, n in enumerate(output.ctc_lengths.cpu().tolist())
        ]


class Wav2Vec2Recognizer(SpeechRecognizer[Wav2Vec2RecognizerConfig]):
    def __init__(self, config: Wav2Vec2RecognizerConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_id, revision=config.revision)
        model = Wav2Vec2ForCTC.from_pretrained(config.model_id, revision=config.revision)
        self.model = model.to(device).eval()  # pyright: ignore[reportCallIssue]

    def transcribe_batch(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> list[str]:
        clips = [waveforms[i, :n].cpu().numpy() for i, n in enumerate(waveform_lengths.tolist())]
        inputs = self.processor(
            clips, sampling_rate=self.config.sample_rate, return_tensors="pt", padding=True, return_attention_mask=True,
        )
        logits = self.model(
            inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device),
        ).logits
        return self.processor.batch_decode(logits.argmax(dim=-1).cpu())
