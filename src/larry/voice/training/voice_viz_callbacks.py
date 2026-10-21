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
from larry.voice.data.dataloading.voice_dataloading import WaveformCollator
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.speech_encoder import SpeechEncoderModel, SpeechEncoderPreBottleneck
from larry.voice.model.vocoder import LarryVocoder, LarryVocoderGeneration
from larry.voice.training import voice_trainers
from larry.voice.utils import collapse_ctc_emissions, make_non_pad_mask


def pick_rows(offsets: list[int], per_source: int) -> list[int]:
    """Picks the same rows from each dataset source for visualization."""
    rows: list[int] = []
    for start, end in itertools.pairwise(offsets):
        span = end - start
        rows += sorted({start + (j * span) // per_source for j in range(min(per_source, span))})

    return rows


class VoiceVizCallback(trainers.VizCallback, ABC):
    """Superclass for voice viz callbacks."""

    @staticmethod
    def comparison_figure(real_mel: torch.Tensor, fake_mel: torch.Tensor, frame_seconds: float) -> Figure:
        seconds = real_mel.shape[-1] * frame_seconds
        extent = (0.0, seconds, 0.0, float(real_mel.shape[0]))
        lo, hi = float(real_mel.min()), float(real_mel.max())
        diff = fake_mel - real_mel
        limit = max(float(diff.abs().quantile(0.99)), 1e-6)

        fig = Figure(figsize=(14, 10))
        axes = fig.subplots(4, 1, gridspec_kw={"height_ratios": [3, 3, 3, 2]})
        for ax, data, title in ((axes[0], real_mel, "real"), (axes[1], fake_mel, "generated")):
            ax.imshow(data.numpy(), origin="lower", aspect="auto", extent=extent, vmin=lo, vmax=hi)
            ax.set_title(title, loc="left")
            ax.set_ylabel("mel bin")

        image = axes[2].imshow(
            diff.numpy(), origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axes[2].set_title("generated - real", loc="left")
        axes[2].set_ylabel("mel bin")
        axes[2].set_xlabel("seconds")
        fig.colorbar(image, ax=axes[2], fraction=0.02)

        axes[3].plot(real_mel.mean(dim=-1).numpy(), label="real")
        axes[3].plot(fake_mel.mean(dim=-1).numpy(), label="generated")
        axes[3].set_title("average log-mel spectrum", loc="left")
        axes[3].set_xlabel("mel bin")
        axes[3].legend()
        fig.tight_layout()
        return fig


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
    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            generation_steps: int = 32,
            cfg_scale: float = 2.0,
            seed: int = 0,
    ) -> None:
        super().__init__(trainer, viz_steps)
        self.generation_steps = generation_steps
        self.cfg_scale = cfg_scale
        self.seed = seed

    def _viz(
            self,
            items: list[Any],
            batch: dict[str, Any],
            model: SpeechEncoderModel | SpeechDecoderModel,
            model_config: SpeechEncoderDecoderModelConfig,
            device: torch.device,
            step: int,
    ) -> None:
        decoder = cast(SpeechDecoderModel, model)
        was_training = decoder.training
        decoder.eval()
        try:
            with torch.no_grad(), self.trainer.accelerator.autocast():
                real, real_lengths = decoder.frontend(
                    batch["waveforms"].to(device), batch["waveform_lengths"].to(device)
                )
                generated = decoder.generate(
                    batch["unit_ids"].to(device),
                    batch["unit_id_lengths"].to(device),
                    batch["speaker_ids"].to(device),
                    steps=self.generation_steps,
                    cfg_scale=self.cfg_scale,
                    generator=torch.Generator(device=device).manual_seed(self.seed),
                )
        finally:
            decoder.train(was_training)

        frame_seconds = model_config.hop_length / model_config.sample_rate
        l1s = []
        for i, item in enumerate(items):
            tag = f"viz/{i}/{self.trainer.config.mixture_config.eval_sources[item['source']].name}"
            frames = int(torch.minimum(real_lengths[i], generated.mel_lengths[i]))
            real_mel = real[i, :, :frames].float().cpu()
            fake_mel = generated.mels[i, :, :frames].float().cpu()
            l1 = float((fake_mel - real_mel).abs().mean())
            l1s.append(l1)

            self.trainer.metrics.add_figure(
                f"{tag}/comparison", self.comparison_figure(real_mel, fake_mel, frame_seconds), step
            )
            self.trainer.metrics.add_scalar(f"{tag}/mel_l1", l1, step)

        self.trainer.metrics.add_scalar("viz/mel_l1", sum(l1s) / max(len(l1s), 1), step)


class VocoderVizCallback(VoiceVizCallback):
    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            examples_per_source: int = 2,
            generation_steps: int = 32,
            step_sweep: tuple[int, ...] = (4, 8, 16, 32),
            seed: int = 0,
    ) -> None:
        super().__init__(trainer, viz_steps)
        self.generation_steps = generation_steps
        self.step_sweep = step_sweep
        self.seed = seed
        self.dataset = trainer.eval_loader.dataset
        self.rows = pick_rows(self.dataset.offsets, examples_per_source)
        self.reference_logged = False

    def viz(self, step: int) -> None:
        if not self.trainer.accelerator.is_main_process:
            return

        if not isinstance(self.trainer, voice_trainers.VocoderTrainer):
            raise ValueError(f"{type(self).__name__} requires a VocoderTrainer trainer type")

        self.log.info(f"Performing visualization at step {step}")

        items = [self.dataset[k] for k in self.rows]
        batch = WaveformCollator()(items)
        model = cast(LarryVocoder, self.trainer.accelerator.unwrap_model(self.trainer.model))
        device = self.trainer.accelerator.device
        waveforms = batch["waveforms"].to(device)
        waveform_lengths = batch["waveform_lengths"].to(device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                mels, mel_lengths = model.frontend(waveforms, waveform_lengths)
                generated = self._generate(model, mels, mel_lengths, self.generation_steps, device)
                fake_mels, fake_lengths = model.frontend(generated.waveforms, generated.waveform_lengths)
                sweep = {
                    n: self._mel_l1(model, mels, mel_lengths, self._generate(model, mels, mel_lengths, n, device))
                    for n in self.step_sweep
                }
        finally:
            model.train(was_training)

        sample_rate = model.config.sample_rate
        for i, item in enumerate(items):
            tag = f"viz/{i}/{self.trainer.config.mixture_config.eval_sources[item['source']].name}"
            n = int(generated.waveform_lengths[i])
            frames = int(torch.minimum(fake_lengths[i], mel_lengths[i]))
            real = waveforms[i, :n].float().cpu()
            fake = generated.waveforms[i, :n].float().cpu()
            real_mel = mels[i, :, :frames].float().cpu()
            fake_mel = fake_mels[i, :, :frames].float().cpu()

            if not self.reference_logged:
                self.trainer.metrics.add_audio(f"{tag}/real", real, step, sample_rate=sample_rate)

            self.trainer.metrics.add_audio(f"{tag}/generated", fake, step, sample_rate=sample_rate)
            frame_seconds = model.config.hop_length / sample_rate
            self.trainer.metrics.add_figure(
                f"{tag}/comparison", self.comparison_figure(real_mel, fake_mel, frame_seconds), step
            )
            for name, value in self._scalars(real, fake, real_mel, fake_mel).items():
                self.trainer.metrics.add_scalar(f"{tag}/{name}", value, step)

        for n, value in sweep.items():
            self.trainer.metrics.add_scalar(f"viz/mel_l1_by_steps/{n}", value, step)

        self.reference_logged = True

    def _generate(
            self,
            model: LarryVocoder,
            mels: torch.Tensor,
            mel_lengths: torch.Tensor,
            steps: int,
            device: torch.device,
    ) -> LarryVocoderGeneration:
        generator = torch.Generator(device=device).manual_seed(self.seed)
        with self.trainer.accelerator.autocast():
            return model.generate(mels, mel_lengths, steps=steps, generator=generator)

    @staticmethod
    def _mel_l1(
            model: LarryVocoder,
            mels: torch.Tensor,
            mel_lengths: torch.Tensor,
            generated: LarryVocoderGeneration,
    ) -> float:
        fake, fake_lengths = model.frontend(generated.waveforms, generated.waveform_lengths)
        frames = min(fake.shape[-1], mels.shape[-1])
        mask = make_non_pad_mask(torch.minimum(fake_lengths, mel_lengths), frames).unsqueeze(1)
        diff = (fake[..., :frames] - mels[..., :frames]).abs() * mask
        return float(diff.sum() / (mask.sum() * mels.shape[1]).clamp(min=1))

    @staticmethod
    def _scalars(
            real: torch.Tensor,
            fake: torch.Tensor,
            real_mel: torch.Tensor,
            fake_mel: torch.Tensor,
    ) -> dict[str, float]:
        third = real_mel.shape[0] // 3
        l1 = (fake_mel - real_mel).abs()
        real_rms = real.square().mean().sqrt().clamp(min=1e-8)
        fake_rms = fake.square().mean().sqrt().clamp(min=1e-8)
        return {
            "mel_l1": float(l1.mean()),
            "mel_l1_high_band": float(l1[2 * third:].mean()),
            "loudness_db_delta": float(20 * torch.log10(fake_rms / real_rms)),
        }
