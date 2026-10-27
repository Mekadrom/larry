import functools
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import DistributedDataParallelKwargs, KwargsHandler
from datasets import Dataset as HFDataset
from huggingface_hub import hf_hub_download
from torch import nn

from larry.common.config.model.model_configs import ModelConfig
from larry.common.criteria.criterion import Criterion
from larry.common.data.dataloading.weighted_batch_sampler import (
    LengthBucketingWeightedBatchSampler,
    WeightedBatchSampler,
    length_batches,
)
from larry.common.training.trainers import MODEL_CONFIG_FILE, LarryTrainer, StepOutput
from larry.common.training.viz_callbacks import VizCallback
from larry.multimodal.config.data.data_configs import MultimodalSourceConfig
from larry.multimodal.config.data.dataloading.dataloading_configs import PretrainingMultimodalCollatorConfig
from larry.multimodal.config.model.config_builder import LarryMultimodalModelConfigBuilder
from larry.multimodal.config.model.model_configs import LarryMultimodalModelConfig
from larry.multimodal.config.training.trainer_configs import LarryMultimodalTrainerConfig
from larry.multimodal.criteria.multimodal_criterion import LarryMultimodalCriterion
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.data.dataloading.indexed_multimodal_dataset import IndexedMultimodalDataset
from larry.multimodal.model.multimodal_model import LarryMultimodalModel, LarryMultimodalModelOutput
from larry.multimodal.processor.processor import LarryMultimodalPretrainProcessor
from larry.multimodal.training.multimodal_viz_callbacks import MultimodalVizCallbacks

MODEL_KEYS = (
    "input_ids", "modality_ids", "input_lengths", "voice_unit_ids", "voice_unit_id_lengths", "images", "image_sizes",
)
EXPORT_DIR = "export"


class LarryMultimodalTrainer(
    LarryTrainer[
        LarryMultimodalModel,
        LarryMultimodalTrainerConfig,
        MultimodalSourceConfig,
        IndexedMultimodalDataset,
    ]
):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accelerator.unwrap_model(self.model).check_processor(self.processor)

    @classmethod
    def prepare_model_config(cls, model_config: ModelConfig, config: LarryMultimodalTrainerConfig) -> ModelConfig:
        if not isinstance(model_config, LarryMultimodalModelConfig):
            raise TypeError(f"{cls.__name__} trains a LarryMultimodalModelConfig, got {type(model_config).__name__}")

        cls.check_sources(model_config, config)
        fsq_dims = fsq_levels = None
        if config.speech_encoder_model_path is not None:
            encoder_config = cls.speech_encoder_config(config.speech_encoder_model_path)
            fsq_dims, fsq_levels = encoder_config["fsq_dims"], encoder_config["fsq_levels"]

        processor = LarryMultimodalPretrainProcessor.from_pretrained(config.processor_model_path)
        cls.check_processor_settings(processor, config)
        # only the modules the configured tasks need are built
        return LarryMultimodalModelConfigBuilder(model_config).build(
            processor,
            fsq_dims,
            fsq_levels,
            config.image_latent_channels,
        )

    @staticmethod
    def check_processor_settings(
            processor: LarryMultimodalPretrainProcessor,
            config: LarryMultimodalTrainerConfig,
    ) -> None:
        settings = {
            "text_vocab_size": (processor.text_tokenizer.vocab_size(), config.text_vocab_size),
            "image_patch_size": (processor.config.image_patch_size, config.image_patch_size),
            "voice_units_per_position": (processor.config.voice_units_per_position, config.voice_units_per_position),
        }

        stale = {
            name: values
            for name, values in settings.items()
            if values[0] != values[1]
        }

        if stale:
            raise ValueError(
                f"the processor at {config.processor_model_path} was built with other settings "
                f"(processor, config): {stale}; rerun tokenizer-training"
            )

    @staticmethod
    def speech_encoder_config(path: str) -> dict[str, Any]:
        config_file = Path(path) / MODEL_CONFIG_FILE
        if not config_file.is_file():
            config_file = Path(hf_hub_download(path, MODEL_CONFIG_FILE))

        return json.loads(config_file.read_text())

    @staticmethod
    def check_sources(model_config: LarryMultimodalModelConfig, config: LarryMultimodalTrainerConfig) -> None:
        tasks = set(model_config.include_tasks)
        mixture = config.mixture_config
        for split, sources in (("train", mixture.train_sources), ("eval", mixture.eval_sources)):
            stray = sorted({source.task for source in sources} - tasks)
            if stray:
                raise ValueError(f"{split} sources have tasks outside include_tasks: {stray}")

        # every configured task needs training data, or its modules train on nothing
        untrained = sorted(tasks - {source.task for source in mixture.train_sources})
        if untrained:
            raise ValueError(f"include_tasks without a train source: {untrained}")

    @functools.cached_property
    def processor(self) -> LarryMultimodalPretrainProcessor:
        return LarryMultimodalPretrainProcessor.from_pretrained(self.config.processor_model_path)

    def compile_model(self, model: nn.Module) -> int:
        if self.config.compile.mode == "model":
            raise ValueError(
                "compile.mode=model would trace the recurrent trunk's data-dependent iteration loop and routing; "
                "use compile.mode=blocks, which compiles each repeated block and keeps the loop eager"
            )
        return super().compile_model(model)

    def accelerator_kwargs_handlers(self) -> list[KwargsHandler]:
        # a batch without voice or image rows never runs those modules; plain DDP errors on parameters with no grad
        return [DistributedDataParallelKwargs(find_unused_parameters=True)]

    def muon_excluded_names(self, model: nn.Module) -> list[str]:
        # embeddings, including the tied text head, and the zero-initialized velocity output stay on AdamW
        embeddings = [f"{name}.weight" for name, module in model.named_modules() if isinstance(module, nn.Embedding)]
        image_coda = getattr(model, "image_coda", None)
        image_output = ["image_coda.out.weight"] if image_coda is not None else []
        return super().muon_excluded_names(model) + embeddings + image_output

    @staticmethod
    def used_columns(config: MultimodalSourceConfig) -> list[str]:
        columns = ["id", "row", config.text_column]
        if config.text_tokens_column is not None:
            columns.append(config.text_tokens_column)

        if IndexedMultimodalDataset.has_voice(config):
            columns.append(config.voice_column)

        if IndexedMultimodalDataset.has_image(config):
            columns += [config.image_column, config.image_shape_column]

        return columns

    def transform_as_source(self, config: MultimodalSourceConfig, dataset: HFDataset) -> HFDataset:
        return dataset.select_columns([c for c in self.used_columns(config) if c in dataset.column_names])

    def load_indexed(
            self,
            configs: list[MultimodalSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False,
    ) -> IndexedMultimodalDataset:
        return IndexedMultimodalDataset(
            configs,
            indexes,
            sources,
            chars_per_token=self.config.mixture_config.chars_per_token,
            image_patch_size=self.processor.config.image_patch_size,
            voice_units_per_position=self.processor.config.voice_units_per_position,
        )

    def make_train_sampler(self, dataset: IndexedMultimodalDataset, weights: list[float]) -> WeightedBatchSampler:
        source_ids, lengths = dataset.source_ids_and_lengths()
        return LengthBucketingWeightedBatchSampler(
            source_ids,
            lengths,
            weights,
            max_batch_length=self.config.mixture_config.max_batch_tokens,
            pool_size=self.config.mixture_config.pool_size,
            seed=self.config.mixture_config.seed,
        )

    def make_eval_sampler(self, dataset: IndexedMultimodalDataset) -> Iterable[list[int]]:
        # shortest first, under the same padded token budget as training
        _, lengths = dataset.source_ids_and_lengths()
        rows = torch.argsort(lengths).tolist()
        return length_batches(rows, lengths, self.config.mixture_config.max_batch_tokens)

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return PretrainingMultimodalCollator(
            PretrainingMultimodalCollatorConfig(processor_model_path=self.config.processor_model_path)
        )

    def make_criterion(self) -> Criterion:
        return LarryMultimodalCriterion(self.config.criterion)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        # one callback per task the eval mixture holds
        return MultimodalVizCallbacks(self, viz_steps, self.config.viz)

    def model_forward(self, batch: Mapping[str, Any]) -> LarryMultimodalModelOutput:
        return self.model(**{key: batch[key] for key in MODEL_KEYS}, is_synthesis=batch["is_synthesis_mask"])

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        output = self.model_forward(batch)
        return self.step_output(output, self.criterion.loss_terms(output, loss_mask=batch["loss_mask"]))

    @functools.cached_property
    def eval_tasks(self) -> list[str]:
        return sorted({
            source.task
            for source in self.config.mixture_config.eval_sources
        })

    @functools.cached_property
    def eval_source_tasks(self) -> torch.Tensor:
        # each eval source's index into eval_tasks
        tasks = [self.eval_tasks.index(source.task) for source in self.config.mixture_config.eval_sources]
        return torch.tensor(tasks, dtype=torch.long, device=self.accelerator.device)

    def eval_groupings(self) -> list[tuple[str, list[str]]]:
        # per source, then per task across every source that holds it
        return [*super().eval_groupings(), ("task/", self.eval_tasks)]

    def eval_group_ids(self, batch: Mapping[str, Any]) -> list[torch.Tensor] | None:
        source_ids = batch["source_ids"].to(self.accelerator.device)
        return [source_ids, self.eval_source_tasks[source_ids]]

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        self._pending_tokens += int(batch["input_lengths"].sum())

    def save(self, step: int) -> None:
        super().save(step)
        if self.config.export_on_save and self.accelerator.is_main_process:
            checkpoint = Path(self.metrics.log_dir).expanduser().resolve() / f"checkpoint-{step}"
            self.accelerator.unwrap_model(self.model).save_with_processor(checkpoint / EXPORT_DIR, self.processor)
