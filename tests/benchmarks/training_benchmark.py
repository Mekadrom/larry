import argparse
import contextlib
import csv
import dataclasses
import gc
import json
import os
import platform
import random
import statistics
import string
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import accelerate
import numpy as np
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import KwargsHandler
from torch import nn
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.flop_counter import FlopCounterMode

from larry.common.config.config import ConfigValues
from larry.common.config.model.model_configs import ModelConfig
from larry.common.config.training.definition import TrainingDefinition
from larry.common.config.training.trainer_configs import LarryTrainerConfig
from larry.common.criteria.criterion import weighted_total
from larry.common.training import trainers
from larry.common.training.training_model import LarryModel
from larry.common.utils.init_helpers import import_submodules
from larry.image.config.model.image_model_configs import ImageVAEConfig
from larry.image.config.training.image_trainer_configs import ImageVAETrainerConfig
from larry.multimodal.config.model.model_configs import LarryMultimodalModelConfig
from larry.multimodal.config.training.trainer_configs import LarryMultimodalTrainerConfig
from larry.voice.config.model.voice_model_configs import (
    LarryVocoderConfig,
    SpeechDecoderModelConfig,
    SpeechEncoderModelConfig,
)
from larry.voice.config.training.voice_trainer_configs import (
    SpeechDecoderTrainerConfig,
    SpeechEncoderTrainerConfig,
    VocoderTrainerConfig,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "training"
SMI_FIELDS = [
    "index", "utilization.gpu", "utilization.memory", "power.draw", "power.limit", "clocks.sm", "clocks.mem",
    "temperature.gpu", "memory.used", "pstate", "clocks_event_reasons.active",
]
BF16_DENSE_TFLOPS = {"NVIDIA GeForce RTX 4090": 165.2}


@dataclasses.dataclass(kw_only=True)
class BenchmarkOptions:
    output_dir: Path
    warmup_steps: int = 5
    steps: int = 20
    profile_steps: int = 4
    seed: int = 0
    smi_interval_ms: int = 200
    ddp_comm_hook: str = "no"
    cache_dir: Path = dataclasses.field(default_factory=lambda: Path("~/.cache/larry/benchmark").expanduser())


class BenchmarkCase[M: ModelConfig, T: LarryTrainerConfig[Any]](ABC):
    name: str
    config_files: list[str]
    work_unit: str

    def __init__(self) -> None:
        self.definition = ConfigValues([CONFIG_ROOT / f for f in self.config_files]).load(TrainingDefinition)
        self.template = cast(M, self.definition.model)
        self.trainer_config = cast(T, self.definition.trainer)
        self.trainer_type = trainers.TrainerBase.REGISTRY.for_config(self.trainer_config)

    def model_config(self) -> Any:
        return self.trainer_type.prepare_model_config(self.template, self.trainer_config)

    def build_model(self) -> nn.Module:
        model_config = self.model_config()
        return LarryModel.REGISTRY.for_config(model_config)(model_config)

    def trainer_stub(self) -> Any:
        stub = self.trainer_type.__new__(self.trainer_type)
        stub._config = self.trainer_config
        return stub

    def optimizers(self, model: nn.Module) -> list[Optimizer]:
        return self.trainer_stub().make_optimizers(model)

    def kwargs_handlers(self) -> list[KwargsHandler]:
        return self.trainer_stub().kwargs_handlers()

    @abstractmethod
    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def work(self, batch: Mapping[str, Any]) -> float:
        ...

    @abstractmethod
    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        ...


class SpeechEncoderCase(BenchmarkCase[SpeechEncoderModelConfig, SpeechEncoderTrainerConfig]):
    name = "speech_encoder"
    config_files = ["voice/mixtures/asr.yaml", "voice/ctc/larry-speech-encoder.yaml"]
    work_unit = "audio_seconds"
    clip_seconds = 14.0

    def __init__(self) -> None:
        super().__init__()
        from larry.voice.config.criterion.voice_criterion_configs import CTCCriterionConfig
        from larry.voice.criteria.ctc import CTCCriterion

        model_config = self.template
        self.criterion = CTCCriterion(CTCCriterionConfig(
            blank=model_config.vocab_size, ctc_layer_loss_weights=self.trainer_config.ctc_layer_loss_weights,
        ))

    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        sample_rate = self.trainer_config.mixture_config.sample_rate
        clips = max(1, int(self.trainer_config.mixture_config.max_batch_seconds * scale // self.clip_seconds))
        samples = int(self.clip_seconds * sample_rate)
        targets = int(self.clip_seconds * 4)
        return {
            "waveforms": torch.randn(clips, samples, generator=generator, device=device) * 0.1,
            "waveform_lengths": torch.full((clips,), samples, device=device),
            "targets": torch.randint(0, self.template.vocab_size, (clips, targets), generator=generator,
                                     device=device),
            "target_lengths": torch.full((clips,), targets, device=device),
        }

    def work(self, batch: Mapping[str, Any]) -> float:
        return float(batch["waveform_lengths"].sum()) / self.trainer_config.mixture_config.sample_rate

    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        output = model(batch["waveforms"], batch["waveform_lengths"])
        terms = self.criterion.loss_terms(output, targets=batch["targets"], target_lengths=batch["target_lengths"])
        return weighted_total(terms)


class SpeechDecoderCase(BenchmarkCase[SpeechDecoderModelConfig, SpeechDecoderTrainerConfig]):
    name = "speech_decoder"
    config_files = ["voice/mixtures/tts.yaml", "voice/tts/decoder/larry-speech-decoder.yaml"]
    work_unit = "audio_seconds"
    clip_seconds = 14.0

    def __init__(self) -> None:
        super().__init__()
        from larry.voice.criteria.flow_matching import SpeechDecoderCriterion

        self.criterion = SpeechDecoderCriterion(self.trainer_config.criterion)

    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        model_config = self.template
        sample_rate = self.trainer_config.mixture_config.sample_rate
        clips = max(1, int(self.trainer_config.mixture_config.max_batch_seconds * scale // self.clip_seconds))
        samples = int(self.clip_seconds * sample_rate)
        units = int(self.clip_seconds * sample_rate / model_config.hop_length / model_config.unit_upsample)
        return {
            "unit_ids": torch.randint(0, model_config.n_units, (clips, units), generator=generator, device=device),
            "unit_id_lengths": torch.full((clips,), units, device=device),
            "speaker_ids": torch.randint(0, model_config.n_speakers, (clips,), generator=generator, device=device),
            "waveforms": torch.randn(clips, samples, generator=generator, device=device) * 0.1,
            "waveform_lengths": torch.full((clips,), samples, device=device),
        }

    def work(self, batch: Mapping[str, Any]) -> float:
        return float(batch["waveform_lengths"].sum()) / self.trainer_config.mixture_config.sample_rate

    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        output = model(
            batch["unit_ids"], batch["unit_id_lengths"], batch["speaker_ids"], batch["waveforms"],
            batch["waveform_lengths"],
        )
        return weighted_total(self.criterion.loss_terms(output, step=0))


class VocoderCase(BenchmarkCase[LarryVocoderConfig, VocoderTrainerConfig]):
    name = "vocoder"
    config_files = ["voice/mixtures/vocoder.yaml", "voice/vocoder/larry-vocoder.yaml"]
    work_unit = "audio_seconds"

    def __init__(self) -> None:
        super().__init__()
        from larry.voice.criteria.flow_matching import VocoderCriterion
        from larry.voice.model.voice_modules import MelAugmenter

        self.augmenter = MelAugmenter(self.trainer_config.mel_augment)
        self.criterion = VocoderCriterion(self.trainer_config.criterion, self.template)

    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        self.criterion.to(device)
        crops = max(1, int(self.trainer_config.clips_per_batch * self.trainer_config.crops_per_clip * scale))
        samples = int(self.trainer_config.crop_seconds * self.trainer_config.mixture_config.sample_rate)
        return {
            "waveforms": torch.randn(crops, samples, generator=generator, device=device) * 0.1,
            "waveform_lengths": torch.full((crops,), samples, device=device),
        }

    def work(self, batch: Mapping[str, Any]) -> float:
        return float(batch["waveform_lengths"].sum()) / self.trainer_config.mixture_config.sample_rate

    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        waveforms, waveform_lengths = batch["waveforms"], batch["waveform_lengths"]
        mels, mel_lengths = unwrapped.frontend(waveforms, waveform_lengths)
        output = model(unwrapped.codec.encode(waveforms), self.augmenter(mels), mel_lengths)
        terms = self.criterion.loss_terms(
            output, waveforms=waveforms, waveform_lengths=waveform_lengths, mels=mels, mel_lengths=mel_lengths
        )
        return weighted_total(terms)


class ImageVAECase(BenchmarkCase[ImageVAEConfig, ImageVAETrainerConfig]):
    name = "image_vae"
    config_files = ["image/mixtures/vae.yaml", "image/vae/larry-image-vae.yaml"]
    work_unit = "images"

    def __init__(self) -> None:
        super().__init__()
        from larry.image.criteria.vae import ImageVAECriterion

        self.criterion = ImageVAECriterion(self.trainer_config.criterion)

    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        size = self.trainer_config.mixture_config.crop_size
        shape = (max(1, int(self.trainer_config.train_batch_size * scale)), 3, size, size)
        return {"images": torch.rand(shape, generator=generator, device=device) * 2 - 1}

    def work(self, batch: Mapping[str, Any]) -> float:
        return float(batch["images"].shape[0])

    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        from larry.image.model.vae import SpatialTransform

        criterion = self.trainer_config.criterion
        output = model(
            batch["images"],
            SpatialTransform(rotations=1, scale=min(criterion.equivariance_scales, default=1.0)),
            equivariance_gradients=criterion.equivariance_weight != 0,
            adversarial_gradients=criterion.adversarial_weight != 0,
        )
        return weighted_total(self.criterion.loss_terms(output, images=batch["images"], step=0))


class MultimodalCase(BenchmarkCase[LarryMultimodalModelConfig, LarryMultimodalTrainerConfig]):
    name = "multimodal"
    config_files = [
        "multimodal/mixtures/text-only.yaml",
        "multimodal/mixtures/text-voice.yaml",
        "multimodal/mixtures/voice-text.yaml",
        "multimodal/larry-multimodal.yaml",
    ]
    work_unit = "tokens"
    row_tokens = 512
    words = 4096

    def __init__(self, cache_dir: Path) -> None:
        super().__init__()
        from larry.multimodal.criteria.multimodal_criterion import LarryMultimodalCriterion

        self.criterion = LarryMultimodalCriterion(self.trainer_config.criterion)
        self.processor_path = cache_dir / "multimodal_processor"
        self.vocabulary = self.make_vocabulary()

    def make_vocabulary(self) -> list[str]:
        rng = random.Random(0)
        return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(self.words)]

    def ensure_processor(self) -> None:
        import sentencepiece

        from larry.multimodal.config.processor_configs import LarryProcessorConfig
        from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor

        state = PartialState()
        if state.is_main_process and not (self.processor_path / "done").exists():
            self.processor_path.mkdir(parents=True, exist_ok=True)
            rng = random.Random(1)
            corpus = self.processor_path / "corpus.txt"
            corpus.write_text("\n".join(" ".join(rng.choices(self.vocabulary, k=24)) for _ in range(60000)))
            sentencepiece.SentencePieceTrainer.train(
                input=str(corpus),
                model_prefix=str(self.processor_path / "sp"),
                vocab_size=self.trainer_config.text_vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
                minloglevel=2,
            )
            LarryMultimodalPretrainProcessor(LarryProcessorConfig(
                text_tokenizer_model_file_path=str(self.processor_path / "sp.model"),
                image_patch_size=self.trainer_config.image_patch_size,
                voice_units_per_position=self.trainer_config.voice_units_per_position,
            )).save_pretrained(self.processor_path / "processor")
            (self.processor_path / "done").touch()
        state.wait_for_everyone()

    def model_config(self) -> Any:
        from larry.multimodal.config.model.config_builder import LarryMultimodalModelConfigBuilder
        from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor

        self.ensure_processor()
        processor = LarryMultimodalPretrainProcessor.from_pretrained(self.processor_path / "processor")
        voice = self.template.voice_prelude_config
        if voice is None:
            raise ValueError("the multimodal benchmark trains voice tasks, so the template needs a voice_prelude_config")
        return LarryMultimodalModelConfigBuilder(self.template).build(processor, voice.fsq_dims, voice.fsq_levels)

    def text(self, rng: random.Random, n_words: int) -> str:
        return " ".join(rng.choices(self.vocabulary, k=n_words))

    def batch(
            self, model: Any, device: torch.device, generator: torch.Generator, scale: float
    ) -> dict[str, Any]:
        from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
        from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator

        collator = PretrainingMultimodalCollator(PretrainingMultimodalCollatorConfig(
            processor_model_path=str(self.processor_path / "processor")
        ))
        rng = random.Random(int(torch.randint(0, 2 ** 31, (1,), generator=generator, device=device)))
        n_units = model.config.voice_prelude_config.vocab_size
        rows = max(1, int(self.trainer_config.mixture_config.max_batch_tokens * scale // self.row_tokens))
        items = []
        for row in range(rows):
            kind = row % 3
            if kind == 0:
                items.append({"text": self.text(rng, self.row_tokens // 3), "source": 0, "is_modal_synthesis": True})
                continue

            units = np.asarray([rng.randrange(n_units) for _ in range(self.row_tokens * 3 // 4)])
            items.append({
                "text": self.text(rng, self.row_tokens // 12),
                "voice": units,
                "source": kind,
                "is_modal_synthesis": kind == 1,
            })

        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collator(items).items()}

    def work(self, batch: Mapping[str, Any]) -> float:
        return float(batch["input_lengths"].sum())

    def loss(self, model: nn.Module, unwrapped: Any, batch: Mapping[str, Any]) -> torch.Tensor:
        from larry.multimodal.training.trainers import MODEL_KEYS

        output = model(**{key: batch[key] for key in MODEL_KEYS}, is_synthesis=batch["is_synthesis_mask"])
        return weighted_total(self.criterion.loss_terms(output, loss_mask=batch["loss_mask"]))


class NvidiaSmiSampler:
    def __init__(self, path: Path, interval_ms: int) -> None:
        self.path = path
        self.interval_ms = interval_ms
        self.process: subprocess.Popen[bytes] | None = None
        self.file: Any = None

    def __enter__(self) -> NvidiaSmiSampler:
        self.file = open(self.path, "w")
        self.file.write(",".join(SMI_FIELDS) + "\n")
        self.file.flush()
        self.process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}"],
            stdout=self.file,
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
        if self.file is not None:
            self.file.close()

    @staticmethod
    def summarize(path: Path, indices: list[int]) -> dict[str, dict[str, float]]:
        per_gpu: dict[str, dict[str, list[float]]] = {}
        with open(path) as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                index = row["index"].strip()
                if int(index) not in indices:
                    continue
                values = per_gpu.setdefault(index, {})
                for field in SMI_FIELDS[1:9]:
                    try:
                        values.setdefault(field, []).append(float(row[field]))
                    except (TypeError, ValueError):
                        continue
                throttled = int(row["clocks_event_reasons.active"].strip(), 16) & ~0x1 != 0
                values.setdefault("throttled_fraction", []).append(float(throttled))

        summary: dict[str, dict[str, float]] = {}
        for index, values in per_gpu.items():
            summary[index] = {f"{field}_mean": statistics.fmean(v) for field, v in values.items() if v}
            summary[index].update({f"{field}_max": max(v) for field, v in values.items() if v})
            summary[index]["samples"] = float(len(values.get("utilization.gpu", [])))
        return summary


class TrainingBenchmark:
    def __init__(self, options: BenchmarkOptions, case_names: list[str]) -> None:
        self.options = options
        self.case_names = case_names
        self.state = PartialState()

    def make_case(self, name: str) -> BenchmarkCase[Any, Any]:
        cases: dict[str, Callable[[], BenchmarkCase[Any, Any]]] = {
            "speech_encoder": SpeechEncoderCase,
            "speech_decoder": SpeechDecoderCase,
            "vocoder": VocoderCase,
            "image_vae": ImageVAECase,
            "multimodal": lambda: MultimodalCase(self.options.cache_dir),
        }
        case = cases[name]()
        case.trainer_config.ddp_comm_hook = self.options.ddp_comm_hook
        return case

    def visible_indices(self) -> list[int]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            return [int(i) for i in visible.split(",")]
        return list(range(torch.cuda.device_count()))

    def write_environment(self) -> None:
        if not self.state.is_main_process:
            return

        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        smi = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout
        (self.options.output_dir / "nvidia-smi-q.txt").write_text(smi)
        environment = {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "accelerate": accelerate.__version__,
            "num_processes": self.state.num_processes,
            "visible_devices": self.visible_indices(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "options": {k: str(v) for k, v in dataclasses.asdict(self.options).items()},
        }
        (self.options.output_dir / "environment.json").write_text(json.dumps(environment, indent=2))

    def step(self, accelerator: Accelerator, model: nn.Module, optimizers: list[Optimizer], case: BenchmarkCase[Any, Any],
             batch: Mapping[str, Any], max_grad_norm: float | None) -> torch.Tensor:
        with accelerator.autocast():
            loss = case.loss(model, accelerator.unwrap_model(model), batch)
        accelerator.backward(loss)
        if max_grad_norm is not None:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return loss.detach()

    def fitting_batch(
            self,
            accelerator: Accelerator,
            model: nn.Module,
            optimizers: list[Optimizer],
            case: BenchmarkCase[Any, Any],
            generator: torch.Generator,
    ) -> tuple[dict[str, Any], float]:
        scale = 1.0
        while True:
            batch = case.batch(accelerator.unwrap_model(model), accelerator.device, generator, scale)
            failed = torch.zeros(1, device=accelerator.device)
            try:
                self.step(accelerator, model, optimizers, case, batch, case.trainer_config.max_grad_norm)
            except torch.OutOfMemoryError:
                failed += 1
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                batch = {}
                gc.collect()
                torch.cuda.empty_cache()

            if float(cast(torch.Tensor, accelerator.reduce(failed, reduction="sum"))) == 0:
                return batch, scale

            if accelerator.is_main_process:
                print(f"{case.name}: out of memory at batch scale {scale}; halving", flush=True)
            scale /= 2

    def count_flops(self, accelerator: Accelerator, model: nn.Module, optimizers: list[Optimizer],
                    case: BenchmarkCase[Any, Any], batch: Mapping[str, Any]) -> float:
        counter = FlopCounterMode(display=False)
        with counter:
            with accelerator.autocast():
                loss = case.loss(model, accelerator.unwrap_model(model), batch)
            accelerator.backward(loss)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return float(counter.get_total_flops())

    def run_case(self, name: str) -> dict[str, Any]:
        options = self.options
        case = self.make_case(name)
        out = options.output_dir / name
        if self.state.is_main_process:
            out.mkdir(parents=True, exist_ok=True)

        accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=case.kwargs_handlers())
        device = accelerator.device
        torch.manual_seed(options.seed)
        model = case.build_model()
        n_parameters = sum(p.numel() for p in model.parameters())
        optimizers = case.optimizers(model)
        prepared = accelerator.prepare(model, *optimizers)
        model, optimizers = prepared[0], list(prepared[1:])
        model.train()

        generator = torch.Generator(device=device).manual_seed(options.seed + accelerator.process_index)
        batch, scale = self.fitting_batch(accelerator, model, optimizers, case, generator)
        work = case.work(batch)
        max_grad_norm = case.trainer_config.max_grad_norm

        for _ in range(options.warmup_steps):
            self.step(accelerator, model, optimizers, case, batch, max_grad_norm)
        torch.cuda.synchronize(device)

        flops = self.count_flops(accelerator, model, optimizers, case, batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        smi_path = out / "nvidia_smi.csv"
        times: list[float] = []
        losses: list[float] = []
        sampler = NvidiaSmiSampler(smi_path, options.smi_interval_ms)
        with sampler if self.state.is_main_process else contextlib.nullcontext():
            accelerator.wait_for_everyone()
            for _ in range(options.steps):
                start = time.perf_counter()
                loss = self.step(accelerator, model, optimizers, case, batch, max_grad_norm)
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
                losses.append(float(loss))
            accelerator.wait_for_everyone()

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        def write_trace(prof: Any) -> None:
            prof.export_chrome_trace(str(out / "trace_rank0.json"))
            (out / "profile_table.txt").write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))

        trace_handler = write_trace if accelerator.is_main_process else None
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=0, warmup=1, active=options.profile_steps, repeat=1),
                on_trace_ready=trace_handler,
                profile_memory=True,
        ) as prof:
            for _ in range(options.profile_steps + 1):
                self.step(accelerator, model, optimizers, case, batch, max_grad_norm)
                torch.cuda.synchronize(device)
                prof.step()

        per_rank = torch.tensor(
            [statistics.median(times), statistics.fmean(times), max(times), work, flops,
             float(peak_allocated), float(peak_reserved)],
            device=device, dtype=torch.float64,
        )
        gathered = cast(torch.Tensor, accelerator.gather(per_rank[None])).cpu()
        result: dict[str, Any] = {}
        if accelerator.is_main_process:
            result = self.summarize(case, gathered, times, losses, n_parameters, smi_path, scale)
            (out / "summary.json").write_text(json.dumps(result, indent=2))

        accelerator.free_memory()
        del model, optimizers, prepared, batch
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        return result

    def summarize(self, case: BenchmarkCase[Any, Any], gathered: torch.Tensor, times: list[float], losses: list[float],
                  n_parameters: int, smi_path: Path, scale: float) -> dict[str, Any]:
        median_step = float(gathered[:, 0].max())
        mean_step = float(gathered[:, 1].max())
        global_work = float(gathered[:, 3].sum())
        flops_per_gpu = float(gathered[:, 4].mean())
        device_name = torch.cuda.get_device_name(0)
        peak_tflops = BF16_DENSE_TFLOPS.get(device_name)
        achieved = flops_per_gpu / median_step / 1e12
        return {
            "model": case.name,
            "parameters": n_parameters,
            "num_processes": int(gathered.shape[0]),
            "batch_scale": scale,
            "ddp_comm_hook": case.trainer_config.ddp_comm_hook,
            "steps_timed": len(times),
            "step_seconds_median": median_step,
            "step_seconds_mean": mean_step,
            "step_seconds_max": float(gathered[:, 2].max()),
            "step_seconds_rank0": times,
            "work_unit": case.work_unit,
            "work_per_step_global": global_work,
            "throughput_per_second": global_work / median_step,
            "steps_per_second": 1.0 / median_step,
            "train_flops_per_step_per_gpu": flops_per_gpu,
            "achieved_tflops_per_gpu": achieved,
            "mfu_vs_bf16_dense_peak": achieved / peak_tflops if peak_tflops else None,
            "peak_memory_allocated_gib_per_rank": [float(v) / 2 ** 30 for v in gathered[:, 5]],
            "peak_memory_reserved_gib_per_rank": [float(v) / 2 ** 30 for v in gathered[:, 6]],
            "loss_first_last": [losses[0], losses[-1]],
            "nvidia_smi": NvidiaSmiSampler.summarize(smi_path, self.visible_indices()),
        }

    def write_report(self, results: list[dict[str, Any]]) -> None:
        if not self.state.is_main_process:
            return

        (self.options.output_dir / "summary.json").write_text(json.dumps(results, indent=2))
        header = ("| model | params (M) | batch scale | step s (median) | throughput | TFLOP/s per GPU | MFU "
                  "| peak reserved GiB (max rank) | GPU util % | power W | SM MHz |")
        lines = [header, "|" + "---|" * 11]
        for r in results:
            smi = r["nvidia_smi"].values()
            util = statistics.fmean(g.get("utilization.gpu_mean", 0.0) for g in smi) if smi else 0.0
            power = statistics.fmean(g.get("power.draw_mean", 0.0) for g in smi) if smi else 0.0
            clock = statistics.fmean(g.get("clocks.sm_mean", 0.0) for g in smi) if smi else 0.0
            mfu = f"{r['mfu_vs_bf16_dense_peak']:.1%}" if r["mfu_vs_bf16_dense_peak"] is not None else "n/a"
            lines.append(
                f"| {r['model']} | {r['parameters'] / 1e6:.1f} | {r['batch_scale']:g} | {r['step_seconds_median']:.3f} | "
                f"{r['throughput_per_second']:.1f} {r['work_unit']}/s | {r['achieved_tflops_per_gpu']:.1f} | {mfu} | "
                f"{max(r['peak_memory_reserved_gib_per_rank']):.1f} | {util:.0f} | {power:.0f} | {clock:.0f} |"
            )
        (self.options.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    def run(self) -> None:
        import_submodules("larry")
        self.write_environment()
        results = []
        for name in self.case_names:
            if self.state.is_main_process:
                print(f"benchmarking {name}", flush=True)
            results.append(self.run_case(name))
        self.write_report(results)


def main() -> None:
    """Benchmarks a few training steps of every model on dummy batches; launch with ``accelerate launch -m``."""
    parser = argparse.ArgumentParser(prog="training-benchmark")
    parser.add_argument("--output_dir", type=Path, default=Path("~/logs/nvidia/before").expanduser())
    parser.add_argument(
        "--models", nargs="+",
        default=["speech_encoder", "speech_decoder", "vocoder", "image_vae", "multimodal"],
    )
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profile_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ddp_comm_hook", default="no", choices=["no", "fp16", "bf16", "power_sgd", "batched_power_sgd"]
    )
    parser.add_argument("--cache_dir", type=Path, default=Path("~/.cache/larry/benchmark").expanduser())
    args = parser.parse_args()

    options = BenchmarkOptions(
        output_dir=args.output_dir.expanduser(),
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        profile_steps=args.profile_steps,
        seed=args.seed,
        cache_dir=args.cache_dir.expanduser(),
        ddp_comm_hook=args.ddp_comm_hook,
    )
    TrainingBenchmark(options, args.models).run()


if __name__ == "__main__":
    main()
