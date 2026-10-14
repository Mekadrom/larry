from abc import ABC
from typing import TYPE_CHECKING, cast

import torch
import torchaudio.functional as AF
from matplotlib.figure import Figure

import larry.common.training.trainers as trainers
from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel

if TYPE_CHECKING:
    from larry.voice.training.voice_trainers import SpeechTokenizerTrainer


class VoiceVizCallback(trainers.VizCallback, ABC):
    ...


def pick_rows(offsets: list[int], per_source: int) -> list[int]:
    rows: list[int] = []
    for start, end in zip(offsets, offsets[1:]):
        span = end - start
        rows += sorted({start + (j * span) // per_source for j in range(min(per_source, span))})
    return rows


def collapse(frames: torch.Tensor, blank: int) -> list[int]:
    ids = torch.unique_consecutive(frames)
    return ids[ids != blank].tolist()


def forced_path(log_probs: torch.Tensor, target: torch.Tensor, blank: int) -> torch.Tensor | None:
    if len(target) == 0:
        return None
    try:
        path, _ = AF.forced_align(log_probs.unsqueeze(0), target.unsqueeze(0).int(), blank=blank)
    except RuntimeError:
        return None
    return path[0]


def token_starts(frames: torch.Tensor, blank: int) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    previous = blank
    for t, token in enumerate(frames.tolist()):
        if token != blank and token != previous:
            starts.append((t, token))
        previous = token
    return starts


def timeline(frames: torch.Tensor | None, blank: int, tokenizer: TextTokenizer, frame_seconds: float) -> str:
    if frames is None:
        return "(no alignment)"
    return " ".join(
        f"`{t * frame_seconds:.2f}` {tokenizer.id_to_piece(token)}"
        for t, token in token_starts(frames, blank)
    )


def alignment_figure(
        mels: torch.Tensor,
        mel_seconds: float,
        forced: torch.Tensor | None,
        blank: int,
        tokenizer: TextTokenizer,
        frame_seconds: float,
) -> Figure:
    fig = Figure(figsize=(14, 4))
    ax = fig.subplots()
    duration = mels.shape[-1] * mel_seconds
    ax.imshow(mels.numpy(), origin="lower", aspect="auto", extent=(0.0, duration, 0.0, mels.shape[0]))
    if forced is not None:
        for t, token in token_starts(forced, blank):
            x = t * frame_seconds
            ax.axvline(x, color="white", linewidth=0.6, alpha=0.7)
            ax.text(x, mels.shape[0] * 0.98, tokenizer.id_to_piece(token), color="white", fontsize=7,
                    rotation=90, va="top", ha="right")
    ax.set_xlabel("seconds")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    return fig


class SpeechTokenizerVizCallback(VoiceVizCallback):
    def __init__(self, viz_steps: int, examples_per_source: int = 2) -> None:
        super().__init__(viz_steps)
        self.examples_per_source = examples_per_source
        self.rows: list[int] | None = None
        self.audio_logged = False

    def viz(self, trainer: trainers.TrainerBase, step: int) -> None:
        if not trainer.accelerator.is_main_process:
            return
        speech_trainer = cast("SpeechTokenizerTrainer", trainer)
        dataset = cast(IndexedSpeechDataset, speech_trainer.eval_loader.dataset)
        if self.rows is None:
            self.rows = pick_rows(dataset.offsets, self.examples_per_source)

        items = [dataset[k] for k in self.rows]
        collate_fn = speech_trainer.make_collate_fn()
        assert collate_fn is not None
        batch = collate_fn(items)

        model = cast(SpeechTokenizerModel, speech_trainer.accelerator.unwrap_model(speech_trainer.model))
        device = speech_trainer.accelerator.device
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), speech_trainer.accelerator.autocast():
                output = model(waveforms, waveform_lengths)
                mels, mel_lengths = model.frontend(waveforms, waveform_lengths)
        finally:
            model.train(was_training)

        tokenizer = speech_trainer.tokenizer
        blank = speech_trainer.model_config.text_vocab_size
        frame_seconds = model.encoder1.frame_seconds
        mel_seconds = model.frontend.hop_length / speech_trainer.model_config.sample_rate
        sources = speech_trainer.config.mixture_config.eval_sources
        for i, item in enumerate(items):
            log_probs = output.ctc_log_probs[i, :int(output.frame_lengths[i])].float().cpu()
            target = batch["targets"][i, :int(batch["target_lengths"][i])].cpu()

            greedy = log_probs.argmax(-1)
            forced = forced_path(log_probs, target, blank)

            tag = f"viz/{sources[item['source']].name}/{i}"
            if not self.audio_logged:
                trainer.metrics.add_audio(
                    f"{tag}/audio",
                    batch["waveforms"][i, :int(batch["waveform_lengths"][i])],
                    step,
                    sample_rate=speech_trainer.config.mixture_config.sample_rate,
                )
            trainer.metrics.add_text(f"{tag}/ctc", "  \n".join([
                f"**target:** {tokenizer.decode(target.tolist())}",
                f"**greedy:** {tokenizer.decode(collapse(greedy, blank))}",
                f"**greedy alignment:** {timeline(greedy, blank, tokenizer, frame_seconds)}",
                f"**target alignment:** {timeline(forced, blank, tokenizer, frame_seconds)}",
            ]), step)
            trainer.metrics.add_figure(f"{tag}/alignment", alignment_figure(
                mels[i, :, :int(mel_lengths[i])].float().cpu(), mel_seconds, forced, blank, tokenizer, frame_seconds,
            ), step)
        self.audio_logged = True
