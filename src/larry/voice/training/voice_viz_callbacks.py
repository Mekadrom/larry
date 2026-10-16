import textwrap
from abc import ABC
from typing import cast

import torch
import torchaudio.functional as AF
from matplotlib.figure import Figure

import larry.common.training.trainers as trainers
from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.data.dataloading.indexed_speech_dataset import IndexedSpeechDataset
from larry.voice.model.speech_tokenizer import SpeechTokenizerModel
from larry.voice.training import voice_trainers
from larry.voice.utils import collapse_ctc_emissions


class VoiceVizCallback(trainers.VizCallback, ABC):
    """Superclass for voice viz callbacks."""

    ...


def pick_rows(offsets: list[int], per_source: int) -> list[int]:
    """Picks the same rows from each dataset source for visualization."""
    rows: list[int] = []
    for start, end in zip(offsets, offsets[1:], strict=False):
        span = end - start
        rows += sorted({start + (j * span) // per_source for j in range(min(per_source, span))})

    return rows


def forced_path(log_probs: torch.Tensor, target: torch.Tensor, blank: int) -> torch.Tensor | None:
    """Judges CTC emissions for the ground truth labels per position to find where it aligns them."""
    if len(target) == 0:
        return None

    try:
        path, _ = AF.forced_align(log_probs.unsqueeze(0), target.unsqueeze(0).int(), blank=blank)
    except RuntimeError:
        return None

    return path[0]


def token_starts(frames: torch.Tensor, blank: int) -> list[tuple[int, int]]:
    """For non-blank provided tokens, returns the frame where each token appears first."""
    starts: list[tuple[int, int]] = []
    previous = blank
    for t, token in enumerate(frames.tolist()):
        if token != blank and token != previous:
            starts.append((t, token))

        previous = token

    return starts


def timeline(frames: torch.Tensor | None, blank: int, tokenizer: TextTokenizer, frame_seconds: float) -> str:
    """Produces the alignment provided with timings prefixing each token."""
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
        target_text: str,
        greedy_text: str,
        label_rows: int = 3,
) -> Figure:
    """Makes a figure of mels with the top containing the target transcript and alignments and the greedy output."""
    n_mels = mels.shape[0]
    band = n_mels * 0.12 * label_rows
    top = n_mels + band
    duration = mels.shape[-1] * mel_seconds

    fig = Figure(figsize=(14, 5.5))
    ax = fig.subplots()
    ax.set_facecolor("black")
    ax.imshow(mels.numpy(), origin="lower", aspect="auto", extent=(0.0, duration, 0.0, n_mels))
    ax.set_xlim(0.0, duration)
    ax.set_ylim(0.0, top)
    ax.set_yticks([0, n_mels // 2, n_mels])

    if forced is not None:
        for i, (t, token) in enumerate(token_starts(forced, blank)):
            x = t * frame_seconds
            y = n_mels + band * (label_rows - i % label_rows - 0.5) / label_rows
            ax.vlines(x, 0.0, y, colors="white", linewidth=0.6, alpha=0.6)
            piece = tokenizer.id_to_piece(token).replace("\u2581", " ").strip() or "\u2581"
            ax.text(x, y, piece, color="white", fontsize=8, ha="left", va="center", clip_on=True,
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": "black", "edgecolor": "none", "alpha": 0.7})

    wrap = 170
    ax.set_title(
        "\n".join(textwrap.wrap(f"target: {target_text}", wrap) + textwrap.wrap(f"greedy: {greedy_text}", wrap)),
        loc="left", fontsize=8, family="monospace",
    )
    ax.set_xlabel("seconds")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    return fig


class SpeechTokenizerVizCallback(VoiceVizCallback):
    """Visualization callback for a speech tokenizer tasked with CTC alignment.

    Saves audio with captions and mel figures that are captioned/labeled with aligned token placements according to the
    model's confidence when force to output the ground truth tokens.
    """

    def __init__(self, viz_steps: int, examples_per_source: int = 2) -> None:
        """Simple init."""
        super().__init__(viz_steps)
        self.examples_per_source = examples_per_source
        self.rows: list[int] | None = None
        self.audio_logged = False

    def viz(self, trainer: trainers.TrainerBase, step: int) -> None:
        """Logs visualizations of audio as mels with aligned CTC tokens, as well as saving the audio and labels."""
        if not trainer.accelerator.is_main_process:
            return

        self.log.info(f"Performing visualization at step {step}")

        speech_trainer = cast(voice_trainers.SpeechTokenizerTrainer, trainer)
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

            tag = f"viz/{i}/{sources[item['source']].name}"
            if not self.audio_logged:
                trainer.metrics.add_audio(
                    f"{tag}/audio",
                    batch["waveforms"][i, :int(batch["waveform_lengths"][i])],
                    step,
                    sample_rate=speech_trainer.config.mixture_config.sample_rate,
                )

            target_text = tokenizer.decode(target.tolist())
            greedy_text = tokenizer.decode(collapse_ctc_emissions(greedy, blank))
            trainer.metrics.add_text(f"{tag}/ctc", "  \n".join([
                f"**target:** {target_text}",
                f"**greedy:** {greedy_text}",
                f"**greedy alignment:** {timeline(greedy, blank, tokenizer, frame_seconds)}",
                f"**target alignment:** {timeline(forced, blank, tokenizer, frame_seconds)}",
            ]), step)
            trainer.metrics.add_figure(f"{tag}/alignment", alignment_figure(
                mels[i, :, :int(mel_lengths[i])].float().cpu(), mel_seconds, forced, blank, tokenizer, frame_seconds,
                target_text, greedy_text,
            ), step)

        self.audio_logged = True
