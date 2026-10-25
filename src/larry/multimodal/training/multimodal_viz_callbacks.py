import dataclasses
import functools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from larry.common.training import trainers
from larry.common.training.viz_callbacks import VizCallback, pick_rows
from larry.multimodal.config.data.dataloading.dataloading_configs import InferenceMultimodalCollatorConfig
from larry.multimodal.config.model.config_builder import Task
from larry.multimodal.config.training.trainer_configs import MultimodalVizConfig
from larry.multimodal.data.dataloading.dataloading import InferenceMultimodalCollator
from larry.multimodal.model.multimodal_model import (
    LarryMultimodalGeneration,
    LarryMultimodalGenerationConfig,
    LarryMultimodalGenerator,
    LarryMultimodalModel,
)
from larry.multimodal.processor.processor import EOS, EOV, ImagePlaceholder, VoicePlaceholder
from larry.voice.model.speech_decoder import SpeechDecoderModel
from larry.voice.model.vocoder import LarryVocoder
from larry.voice.training.voice_trainers import error_counts
from larry.voice.training.voice_viz_callbacks import VoiceVizCallback

VOICE = VoicePlaceholder.placeholder()
IMAGE = ImagePlaceholder.placeholder()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize(text: str) -> str:
    return " ".join(words(text))


def token_f1(hypothesis: str, reference: str) -> float:
    hyp, ref = words(hypothesis), words(reference)
    if not hyp or not ref:
        return float(hyp == ref)

    overlap = sum(min(hyp.count(w), ref.count(w)) for w in set(hyp))
    precision, recall = overlap / len(hyp), overlap / len(ref)
    return 0.0 if overlap == 0 else 2 * precision * recall / (precision + recall)


def ngrams(tokens: Sequence[Any], n: int) -> list[tuple[Any, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(sequences: Sequence[Sequence[Any]], n: int) -> float:
    # share of n-grams across all generations that are unique; low means the model says the same thing to everything
    grams = [gram for sequence in sequences for gram in ngrams(sequence, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_fraction(tokens: Sequence[Any], n: int) -> float:
    # share of a sequence's n-grams that already occurred in it; high means looping
    grams = ngrams(tokens, n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def longest_run_fraction(tokens: Sequence[Any]) -> float:
    if not tokens:
        return 0.0

    longest = run = 1
    for previous, current in zip(tokens, tokens[1:], strict=False):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)

    return longest / len(tokens)


def latent_preview(latents: torch.Tensor) -> np.ndarray:
    # the first three latent channels, each stretched to [0, 1], as an RGB-ish HWC image until a VAE can decode them
    channels = latents[:3].float()
    if channels.shape[0] < 3:
        channels = channels.repeat(math.ceil(3 / channels.shape[0]), 1, 1)[:3]

    flat = channels.flatten(1)
    low, high = flat.min(dim=1).values[:, None, None], flat.max(dim=1).values[:, None, None]
    return ((channels - low) / (high - low).clamp(min=1e-6)).permute(1, 2, 0).cpu().numpy()


@dataclasses.dataclass(kw_only=True)
class Vocalization:
    waveform: torch.Tensor
    mels: torch.Tensor
    sample_rate: int
    frame_seconds: float


class UnitVocalizer:
    def __init__(self, config: MultimodalVizConfig, device: torch.device) -> None:
        if config.speech_decoder_model_path is None:
            raise ValueError("Config missing for speech decoder")

        if config.vocoder_model_path is None:
            raise ValueError("Config missing for vocoder")

        self.config = config
        self.device = device
        self.decoder_model_path = config.speech_decoder_model_path
        self.vocoder_model_path = config.vocoder_model_path

    @staticmethod
    def configured(config: MultimodalVizConfig) -> bool:
        return config.speech_decoder_model_path is not None and config.vocoder_model_path is not None

    @functools.cached_property
    def decoder(self) -> SpeechDecoderModel:
        return SpeechDecoderModel.from_pretrained(self.decoder_model_path).to(self.device).eval()

    @functools.cached_property
    def vocoder(self) -> LarryVocoder:
        return LarryVocoder.from_pretrained(self.vocoder_model_path).to(self.device).eval()

    @torch.no_grad()
    def audio(self, units: torch.Tensor) -> Vocalization:
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        unit_ids = units.to(self.device)[None]
        mels = self.decoder.generate(
            unit_ids,
            torch.tensor([units.shape[0]], device=self.device),
            None,
            steps=self.config.vocoder_steps,
            generator=generator,
        )
        waveforms = self.vocoder.generate(
            mels.mels,
            mels.mel_lengths,
            steps=self.config.vocoder_steps,
            generator=generator
        )

        n_samples = int(waveforms.waveform_lengths[0])
        frames = int(mels.mel_lengths[0])
        vocoder_config = self.vocoder.config
        return Vocalization(
            waveform=waveforms.waveforms[0, :n_samples].float().cpu(),
            mels=mels.mels[0, :, :frames].float().cpu(),
            sample_rate=vocoder_config.sample_rate,
            frame_seconds=vocoder_config.hop_length / vocoder_config.sample_rate,
        )


class MultimodalTaskVizCallback(VizCallback, ABC):
    task: ClassVar[Task]

    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            config: MultimodalVizConfig,
            vocalizer: UnitVocalizer | None = None,
    ) -> None:
        super().__init__(trainer, viz_steps)
        self.config = config
        self.vocalizer = vocalizer
        self.dataset = trainer.eval_loader.dataset
        sources = {
            s
            for s, source in enumerate(self.dataset.configs)
            if source.task == self.task
        }
        self.rows = pick_rows(self.dataset.offsets, config.examples_per_source, sources)
        self.inference = InferenceMultimodalCollator(
            InferenceMultimodalCollatorConfig(processor_model_path=trainer.config.processor_model_path)
        )
        self.tokenizer = self.inference.processor.text_tokenizer
        self.base = self.tokenizer.vocab_size()
        self.references_logged = False

    @property
    def device(self) -> torch.device:
        return self.trainer.accelerator.device

    def viz(self, step: int) -> None:
        if not self.trainer.accelerator.is_main_process or not self.rows:
            return

        self.log.info(f"Visualizing {self.task} at step {step}")
        model: LarryMultimodalModel = self.trainer.accelerator.unwrap_model(self.trainer.model)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), self.trainer.accelerator.autocast():
                self.viz_items([self.dataset[row] for row in self.rows], model, step)
        finally:
            model.train(was_training)

        self.references_logged = True

    @abstractmethod
    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        ...

    def generate(
            self,
            model: LarryMultimodalModel,
            prompt: dict[str, Any],
            **config: Any
    ) -> tuple[LarryMultimodalGeneration, int]:
        batch = self.inference([prompt])
        kwargs: dict[str, Any] = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in InferenceMultimodalCollator.generate_kwargs(batch, 0).items()
        }
        generation_config = LarryMultimodalGenerationConfig(**{
            "max_new_positions": self.config.max_new_tokens,
            "temperature": 0.0,
            "voice_temperature": self.config.voice_temperature,
            "generator": torch.Generator(device=self.device).manual_seed(self.config.seed),
            **config,
        })
        generation = LarryMultimodalGenerator(model, generation_config).run(**kwargs)
        return generation, int(kwargs["input_ids"].shape[1])

    def new_text(self, generation: LarryMultimodalGeneration, start: int) -> tuple[str, list[int], bool]:
        # text tokens the model wrote after the prompt; specials are structure, not text
        new_ids = generation.input_ids[0, start:].tolist()
        tokens = [token for token in new_ids if token < self.base]
        return self.tokenizer.decode(tokens), tokens, self.base + EOS in new_ids

    def caption(self, model: LarryMultimodalModel, prompt: dict[str, Any]) -> tuple[str, list[int], bool]:
        generation, start = self.generate(model, prompt)
        return self.new_text(generation, start)

    def add_text(self, tag: str, values: dict[str, Any], ended: bool, text: str, item: dict[str, Any],
                 step: int) -> None:
        values["eos_rate"].append(float(ended))
        values["length_ratio"].append(len(words(text)) / max(len(words(item["text"])), 1))

        self.trainer.metrics.add_text(f"{tag}/caption", self.comparison(item["text"], text), step)

    @staticmethod
    def comparison(reference: str, generated: str) -> str:
        return f"**reference:** {reference}  \n**generated:** {generated}"

    def tag(self, i: int, item: dict[str, Any]) -> str:
        return f"viz/{self.task}/{i}/{self.dataset.configs[item['source']].name}"

    def log_means(self, values: dict[str, list[float]], step: int) -> None:
        for name, samples in values.items():
            if samples:
                self.trainer.metrics.add_scalar(f"viz/{self.task}/{name}", sum(samples) / len(samples), step)


class TextContinuationVizCallback(MultimodalTaskVizCallback):
    task = "text_only"

    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        values: dict[str, list[float]] = {
            "eos_rate": [], "new_tokens": [], "repeated_4gram": [], "reference_prefix_match": [],
        }
        generations = []
        for i, item in enumerate(items):
            tokens = self.tokenizer.encode(item["text"])
            split = max(len(tokens) // 2, 1)
            prompt = self.tokenizer.decode(tokens[:split])
            reference = tokens[split:]
            text, generated, ended = self.caption(model, {"text": prompt})
            generations.append(generated)

            matched = next((j for j, (a, b) in enumerate(zip(generated, reference, strict=False)) if a != b), None)
            values["eos_rate"].append(float(ended))
            values["new_tokens"].append(float(len(generated)))
            values["repeated_4gram"].append(repeated_fraction(generated, 4))
            # how many tokens the greedy continuation follows the real one before diverging
            followed = min(len(generated), len(reference)) if matched is None else matched
            values["reference_prefix_match"].append(float(followed))
            self.trainer.metrics.add_text(f"{self.tag(i, item)}/continuation", "  \n".join([
                f"**prompt:** {prompt}",
                f"**generated:** {text}",
                f"**reference:** {self.tokenizer.decode(reference)}",
            ]), step)

        self.log_means(values, step)
        self.trainer.metrics.add_scalar(f"viz/{self.task}/distinct_2", distinct_n(generations, 2), step)


class VoiceCaptioningVizCallback(MultimodalTaskVizCallback):
    task = "voice_text"

    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        totals = np.zeros(4)
        values: dict[str, list[float]] = {"eos_rate": [], "length_ratio": []}
        for i, item in enumerate(items):
            text, _, ended = self.caption(model, {"text": VOICE, "voices": [item["voice"]]})
            counts = error_counts(normalize(text), normalize(item["text"]))
            totals += [counts.word_errors, counts.words, counts.char_errors, counts.chars]
            tag = self.tag(i, item)
            self.add_text(tag, values, ended, text, item, step)
            if self.vocalizer is not None and not self.references_logged:
                vocalization = self.vocalizer.audio(torch.as_tensor(item["voice"]).long())
                self.trainer.metrics.add_audio(
                    f"{tag}/input",
                    vocalization.waveform,
                    step,
                    sample_rate=vocalization.sample_rate,
                )

        self.log_means(values, step)
        self.trainer.metrics.add_scalar(f"viz/{self.task}/wer", totals[0] / max(totals[1], 1), step)
        self.trainer.metrics.add_scalar(f"viz/{self.task}/cer", totals[2] / max(totals[3], 1), step)


class ImageCaptioningVizCallback(MultimodalTaskVizCallback):
    task = "image_text"

    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        values: dict[str, list[float]] = {"token_f1": [], "eos_rate": [], "length_ratio": []}
        captions = []
        for i, item in enumerate(items):
            text, generated, ended = self.caption(model, {"text": IMAGE, "images": [item["image"]]})
            captions.append(generated)
            values["token_f1"].append(token_f1(text, item["text"]))
            tag = self.tag(i, item)
            self.add_text(tag, values, ended, text, item, step)
            if not self.references_logged:
                self.trainer.metrics.add_image(f"{tag}/latents", latent_preview(torch.as_tensor(item["image"])), step)

        self.log_means(values, step)

        # captions that all read alike mean the model ignores the image
        self.trainer.metrics.add_scalar(f"viz/{self.task}/distinct_2", distinct_n(captions, 2), step)


class VoiceSynthesisVizCallback(MultimodalTaskVizCallback):
    task = "text_voice"

    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        values: dict[str, list[float]] = {
            "eov_rate": [],
            "length_ratio": [],
            "unit_error_rate": [],
            "unique_units": [],
            "repeated_4gram": [],
            "longest_run": [],
        }

        round_trip = np.zeros(2)
        can_transcribe = "voice_text" in model.config.include_tasks
        for i, item in enumerate(items):
            reference = [int(u) for u in item["voice"]]
            groups = math.ceil(len(reference) / model.voice_units_per_position)
            generation, start = self.generate(
                model,
                {"text": item["text"], "open_modality": "voice"},
                max_new_positions=math.ceil(groups * self.config.voice_length_budget) + 8,
            )

            units = generation.voice_unit_ids[0].tolist() if generation.voice_unit_ids else []

            values["eov_rate"].append(float(model.special_base + EOV in generation.input_ids[0, start:].tolist()))
            # too short means it stopped early; too long means it never found <eov>
            values["length_ratio"].append(len(units) / max(len(reference), 1))
            # speakers and prosody differ, so this never reaches 0, but it should fall as units start to match the text
            values["unit_error_rate"].append(Levenshtein.distance(units, reference) / max(len(reference), 1))
            values["unique_units"].append(len(set(units)) / max(len(units), 1))
            values["repeated_4gram"].append(repeated_fraction(units, 4))
            values["longest_run"].append(longest_run_fraction(units))

            tag = self.tag(i, item)
            summary = [f"**text:** {item['text']}", f"**units:** {len(units)} generated / {len(reference)} reference"]
            if can_transcribe and units:
                # the model's own ASR on what it said: intelligibility without an external recognizer
                heard, _, _ = self.caption(model, {"text": VOICE, "voices": [np.array(units)]})
                counts = error_counts(normalize(heard), normalize(item["text"]))
                round_trip += [counts.word_errors, counts.words]
                summary.append(f"**heard back:** {heard}")

            self.trainer.metrics.add_text(f"{tag}/synthesis", "  \n".join(summary), step)

            if self.vocalizer is not None and units:
                generated = self.vocalizer.audio(torch.tensor(units))
                self.trainer.metrics.add_audio(
                    f"{tag}/generated",
                    generated.waveform,
                    step,
                    sample_rate=generated.sample_rate,
                )
                reference_audio = self.vocalizer.audio(torch.tensor(reference))
                if not self.references_logged:
                    self.trainer.metrics.add_audio(
                        f"{tag}/reference",
                        reference_audio.waveform,
                        step,
                        sample_rate=reference_audio.sample_rate
                    )

                frames = min(generated.mels.shape[-1], reference_audio.mels.shape[-1])
                self.trainer.metrics.add_figure(f"{tag}/mels", VoiceVizCallback.comparison_figure(
                    reference_audio.mels[:, :frames],
                    generated.mels[:, :frames],
                    generated.frame_seconds
                ), step)

        self.log_means(values, step)
        if round_trip[1] > 0:
            self.trainer.metrics.add_scalar(f"viz/{self.task}/round_trip_wer", round_trip[0] / round_trip[1], step)


class ImageSynthesisVizCallback(MultimodalTaskVizCallback):
    task = "text_image"

    def viz_items(self, items: list[dict[str, Any]], model: LarryMultimodalModel, step: int) -> None:
        values: dict[str, list[float]] = {
            "channel_mean_error": [],
            "channel_std_ratio_error": [],
            "relative_mse": [],
            "seed_diversity": [],
            "round_trip_f1": [],
        }

        if model.config.image_patch_size is None:
            raise ValueError("Configuration for patch size missing on model.config")

        can_caption = "image_text" in model.config.include_tasks and model.image_prelude is not None
        for i, item in enumerate(items):
            reference = torch.as_tensor(item["image"]).float()
            _, height, width = reference.shape
            patches = (height // model.config.image_patch_size) * (width // model.config.image_patch_size)
            samples = []
            for seed_offset in (0, 1):
                generation, _ = self.generate(
                    model, {"text": item["text"], "open_modality": "image"},
                    image_size=(width, height),
                    image_steps=self.config.image_steps,
                    image_cfg_scale=self.config.image_cfg_scale,
                    max_new_positions=patches + 4,
                    generator=torch.Generator(device=self.device).manual_seed(self.config.seed + seed_offset),
                )
                samples.append(generation.images[0].float().cpu())

            generated = samples[0]
            power = reference.square().mean().clamp(min=1e-8)

            # latent statistics drifting from the data's is the earliest sign of a broken or collapsing sampler
            mean_g, mean_r = generated.mean(dim=(1, 2)), reference.mean(dim=(1, 2))
            std_g, std_r = generated.std(dim=(1, 2)), reference.std(dim=(1, 2)).clamp(min=1e-6)
            values["channel_mean_error"].append(float((mean_g - mean_r).abs().mean()))
            values["channel_std_ratio_error"].append(float((std_g / std_r - 1).abs().mean()))
            values["relative_mse"].append(float((generated - reference).square().mean() / power))

            # two seeds, one prompt: near zero means the guidance has collapsed every sample to one image
            values["seed_diversity"].append(float((samples[0] - samples[1]).square().mean() / power))

            tag = self.tag(i, item)
            self.trainer.metrics.add_image(f"{tag}/generated", latent_preview(generated), step, caption=item["text"])
            if not self.references_logged:
                self.trainer.metrics.add_image(f"{tag}/reference", latent_preview(reference), step)

            if can_caption:
                heard, _, _ = self.caption(model, {"text": IMAGE, "images": [generated.numpy()]})
                values["round_trip_f1"].append(token_f1(heard, item["text"]))
                self.trainer.metrics.add_text(f"{tag}/caption_of_generated", heard, step)

        self.log_means(values, step)


TASK_VIZ_CALLBACKS: dict[Task, type[MultimodalTaskVizCallback]] = {
    callback.task: callback
    for callback in (
        TextContinuationVizCallback,
        VoiceCaptioningVizCallback,
        ImageCaptioningVizCallback,
        VoiceSynthesisVizCallback,
        ImageSynthesisVizCallback,
    )
}


class MultimodalVizCallbacks(VizCallback):
    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            config: MultimodalVizConfig
    ) -> None:
        super().__init__(trainer, viz_steps)
        tasks = {
            source.task
            for source in trainer.config.mixture_config.eval_sources
        }

        vocalizer = UnitVocalizer(config, trainer.accelerator.device) if UnitVocalizer.configured(config) else None
        self.callbacks = [
            callback(trainer, viz_steps, config, vocalizer)
            for task, callback in TASK_VIZ_CALLBACKS.items()
            if task in tasks
        ]

    def viz(self, step: int) -> None:
        for callback in self.callbacks:
            callback.viz(step)
