import itertools
import textwrap
from abc import ABC, abstractmethod
from typing import Any, cast

import torch
import torchaudio.functional as AF
from matplotlib.figure import Figure

import larry.common.training.trainers as trainers
from larry.text.tokenizer.text_tokenizer import TextTokenizer
from larry.voice.config.model.voice_model_configs import SpeechEncoderDecoderModelConfig
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.speech_encoder import SpeechEncoderModel, SpeechEncoderPreBottleneck
from larry.voice.training import voice_trainers
from larry.voice.utils import collapse_ctc_emissions


def pick_rows(offsets: list[int], per_source: int) -> list[int]:
    """Picks the same rows from each dataset source for visualization."""
    rows: list[int] = []
    for start, end in itertools.pairwise(offsets):
        span = end - start
        rows += sorted({start + (j * span) // per_source for j in range(min(per_source, span))})

    return rows


class VoiceVizCallback(trainers.VizCallback, ABC):
    """Superclass for voice viz callbacks."""

    ...


class SpeechEncoderDecoderVizCallback(VoiceVizCallback, ABC):
    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            examples_per_source: int = 2,
            figure_label_rows: int = 3,
    ) -> None:
        """Simple init."""
        super().__init__(trainer, viz_steps)

        self.examples_per_source = examples_per_source
        self.figure_label_rows = figure_label_rows

        self.audio_logged = False

        self.dataset = trainer.eval_loader.dataset
        self.rows = pick_rows(self.dataset.offsets, self.examples_per_source)

    def viz(self, step: int) -> None:
        if not self.trainer.accelerator.is_main_process:
            return

        self.log.info(f"Performing visualization at step {step}")

        if not isinstance(self.trainer, voice_trainers.SpeechEncoderDecoderTrainerBase):
            raise ValueError(f"{type(self).__name__} requires a SpeechEncoderDecoderTrainerBase trainer type")

        items = [self.dataset[k] for k in self.rows]
        collate_fn = self.trainer.make_collate_fn()

        assert collate_fn is not None
        batch = collate_fn(items)

        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        device = self.trainer.accelerator.device

        self._viz(items, batch, model, model.config, device, step)

    @abstractmethod
    def _viz(
            self,
            items: list[Any],
            batch: dict[str, Any],
            model: SpeechEncoderModel | SpeechDecoderModel,
            model_config: SpeechEncoderDecoderModelConfig,
            device: torch.device,
            step: int,
    ) -> None:
        ...


class SpeechEncoderVizCallback(SpeechEncoderDecoderVizCallback):
    """Visualization callback for a speech tokenizer tasked with CTC alignment.

    Saves audio with captions and mel figures that are captioned/labeled with aligned token placements according to the
    model's confidence when force to output the ground truth tokens.
    """

    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            blank: int,
            tokenizer: TextTokenizer,
    ) -> None:
        super().__init__(trainer, viz_steps)
        self.tokenizer = tokenizer
        self.blank = blank

    def _viz(
            self,
            items: list[Any],
            batch: dict[str, Any],
            model: SpeechEncoderModel | SpeechDecoderModel,
            model_config: SpeechEncoderDecoderModelConfig,
            device: torch.device,
            step: int,
    ) -> None:
        """Logs visualizations of audio as mels with aligned CTC tokens, as well as saving the audio and labels."""
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), self.trainer.accelerator.autocast():
                output = model(waveforms, waveform_lengths)
                mels, mel_lengths = model.frontend(waveforms, waveform_lengths)
        finally:
            model.train(was_training)

        frame_seconds = cast(SpeechEncoderPreBottleneck, model.encoder1).frame_seconds
        mel_seconds = model.frontend.hop_length / model_config.sample_rate

        for i, item in enumerate(items):
            log_probs = output.logits[i, :int(output.frame_lengths[i])].float().cpu()
            target = batch["targets"][i, :int(batch["target_lengths"][i])].cpu()

            greedy = log_probs.argmax(-1)
            forced = self._forced_path(log_probs, target)

            tag = f"viz/{i}/{self.trainer.config.mixture_config.eval_sources[item['source']].name}"
            if not self.audio_logged:
                self.trainer.metrics.add_audio(
                    f"{tag}/audio",
                    batch["waveforms"][i, :int(batch["waveform_lengths"][i])],
                    step,
                    sample_rate=self.trainer.config.mixture_config.sample_rate,
                )

            target_text = self.tokenizer.decode(target.tolist())
            greedy_text = self.tokenizer.decode(collapse_ctc_emissions(greedy, self.blank))
            self.trainer.metrics.add_text(f"{tag}/ctc", "  \n".join([
                f"**target:** {target_text}",
                f"**greedy:** {greedy_text}",
                f"**greedy alignment:** {self._timeline(greedy, frame_seconds)}",
                f"**target alignment:** {self._timeline(forced, frame_seconds)}",
            ]), step)
            self.trainer.metrics.add_figure(f"{tag}/alignment", self._alignment_figure(
                mels[i, :, :int(mel_lengths[i])].float().cpu(),
                mel_seconds,
                forced,
                frame_seconds,
                target_text,
                greedy_text,
            ), step)

        self.audio_logged = True

    def _forced_path(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        """Judges CTC emissions for the ground truth labels per position to find where it aligns them."""
        if len(target) == 0:
            return None

        try:
            path, _ = AF.forced_align(log_probs.unsqueeze(0), target.unsqueeze(0).int(), blank=self.blank)
        except RuntimeError:
            return None

        return path[0]

    def _token_starts(self, frames: torch.Tensor) -> list[tuple[int, int]]:
        """For non-blank provided tokens, returns the frame where each token appears first."""
        starts: list[tuple[int, int]] = []
        previous = self.blank
        for t, token in enumerate(frames.tolist()):
            if token != self.blank and token != previous:
                starts.append((t, token))

            previous = token

        return starts

    def _timeline(self, frames: torch.Tensor | None, frame_seconds: float) -> str:
        """Produces the alignment provided with timings prefixing each token."""
        if frames is None:
            return "(no alignment)"

        return " ".join(
            f"`{t * frame_seconds:.2f}` {self.tokenizer.id_to_piece(token)}"
            for t, token in self._token_starts(frames)
        )

    def _alignment_figure(
            self,
            mels: torch.Tensor,
            mel_seconds: float,
            forced: torch.Tensor | None,
            frame_seconds: float,
            target_text: str,
            greedy_text: str,
    ) -> Figure:
        """Makes a figure of mels with the top containing the target transcript and alignments and the greedy output."""
        n_mels = mels.shape[0]
        band = n_mels * 0.12 * self.figure_label_rows
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
            for i, (t, token) in enumerate(self._token_starts(forced)):
                x = t * frame_seconds
                y = n_mels + band * (self.figure_label_rows - i % self.figure_label_rows - 0.5) / self.figure_label_rows
                ax.vlines(x, 0.0, y, colors="white", linewidth=0.6, alpha=0.6)
                piece = self.tokenizer.id_to_piece(token).replace("\u2581", " ").strip() or "\u2581"
                ax.text(
                    x,
                    y,
                    piece,
                    color="white",
                    fontsize=8,
                    ha="left",
                    va="center",
                    clip_on=True,
                    bbox={"boxstyle": "square,pad=0.1", "facecolor": "black", "edgecolor": "none", "alpha": 0.7}
                )

        wrap = 170
        ax.set_title(
            "\n".join(
                textwrap.wrap(f"target: {target_text}", wrap) + textwrap.wrap(f"greedy: {greedy_text}", wrap)
            ),
            loc="left",
            fontsize=8,
            family="monospace",
        )
        ax.set_xlabel("seconds")
        ax.set_ylabel("mel bin")
        fig.tight_layout()
        return fig


class SpeechDecoderVizCallback(SpeechEncoderDecoderVizCallback):
    def _viz(
            self,
            items: list[Any],
            batch: dict[str, Any],
            model: SpeechEncoderModel | SpeechDecoderModel,
            model_config: SpeechEncoderDecoderModelConfig,
            device: torch.device,
            step: int,
    ) -> None:
        ...
