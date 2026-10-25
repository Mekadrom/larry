import functools
import random
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from accelerate import PartialState
from datasets import Dataset as HFDataset

from larry.common.criteria.criterion import Criterion
from larry.common.data.dataloading.weighted_batch_sampler import FixedSizeWeightedBatchSampler, WeightedBatchSampler
from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.training.trainers import StepOutput, VizCallback
from larry.image.config.data.image_data_configs import ImageMixtureConfig, ImageSourceConfig
from larry.image.config.training.image_trainer_configs import ImageVAETrainerConfig
from larry.image.criteria.vae import ImageVAECriterion
from larry.image.data.dataloading.indexed_image_dataset import ImageCollator, IndexedImageDataset
from larry.image.model.vae import ImageVAE, ImageVAEOutput, SpatialTransform
from larry.image.training.image_trainers import ImageTrainer
from larry.image.training.vae_viz_callbacks import ImageVAEVizCallback, psnr


class ImageVAETrainer(ImageTrainer[ImageVAE, ImageVAETrainerConfig, ImageSourceConfig, IndexedImageDataset]):
    @property
    def mixture(self) -> ImageMixtureConfig:
        return self.config.mixture_config

    @functools.cached_property
    def transform_random(self) -> random.Random:
        return random.Random(self.config.seed * 1000 + PartialState().process_index)

    @functools.cached_property
    def vae_criterion(self) -> ImageVAECriterion:
        return cast(ImageVAECriterion, self.criterion)

    def transform_as_source(self, config: ImageSourceConfig, dataset: HFDataset) -> HFDataset:
        return dataset.select_columns([c for c in ("id", "row", config.image_column) if c in dataset.column_names])

    def load_indexed(
            self,
            configs: list[ImageSourceConfig],
            indexes: list[HFDataset],
            sources: list[HFDataset | None],
            is_eval: bool = False,
    ) -> IndexedImageDataset:
        return IndexedImageDataset(
            configs,
            indexes,
            sources,
            crop_size=self.mixture.crop_size,
            train=not is_eval,
            random_flip=self.mixture.random_flip,
            seed=self.config.seed,
            process_index=PartialState().process_index,
        )

    def make_train_sampler(self, dataset: IndexedImageDataset, weights: list[float]) -> WeightedBatchSampler:
        return FixedSizeWeightedBatchSampler(
            dataset.source_ids(),
            weights,
            batch_size=self.config.train_batch_size,
            pool_size=self.mixture.pool_size,
            seed=self.mixture.seed,
        )

    def make_eval_sampler(self, dataset: IndexedImageDataset) -> Iterable[list[int]]:
        size = self.config.eval_batch_size
        return [list(range(start, min(start + size, len(dataset)))) for start in range(0, len(dataset), size)]

    def make_collate_fn(self) -> Callable[..., dict[str, Any]] | None:
        return ImageCollator()

    def make_criterion(self) -> Criterion:
        return ImageVAECriterion(self.config.criterion)

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return ImageVAEVizCallback(self, viz_steps, self.config.viz)

    def sample_transform(self) -> SpatialTransform:
        scales = self.config.criterion.equivariance_scales
        if not self.model.training:
            return SpatialTransform(rotations=1, scale=min(scales, default=1.0))

        rng = self.transform_random
        return SpatialTransform(rotations=rng.randrange(4), scale=rng.choice([1.0, *scales]))

    def _forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        images = batch["images"].to(self.accelerator.device)
        step = self.train_state.global_step
        output = self.model(
            images,
            self.sample_transform(),
            equivariance_gradients=self.config.criterion.equivariance_weight != 0,
            adversarial_gradients=self.vae_criterion.adversarial_weight(step) != 0,
        )
        return self.step_output(output, self.criterion.loss_terms(output, images=images, step=step))

    def count_tokens(self, batch: Mapping[str, Any]) -> None:
        images = batch["images"]
        self._pending_tokens += images.shape[0] * images.shape[-2] * images.shape[-1]

    def eval_statistics(self) -> dict[str, torch.Tensor]:
        n_sources = len(self.mixture.eval_sources)
        return {"psnr": torch.zeros(n_sources, 2, dtype=torch.float64, device=self.accelerator.device)}

    def update_eval_statistics(
            self,
            statistics: dict[str, torch.Tensor],
            batch: Mapping[str, Any],
            output: StepOutput,
    ) -> None:
        model_output = cast(ImageVAEOutput, output.model_output)
        images = batch["images"].to(model_output.reconstructions.device)
        values = psnr(model_output.reconstructions, images).double()
        rows = torch.stack([values, torch.ones_like(values)], dim=1)
        statistics["psnr"].index_add_(0, batch["source_ids"].to(rows.device), rows)

    def eval_statistics_metrics(self, statistics: dict[str, torch.Tensor]) -> MetricDict:
        metrics = MetricDict()
        totals = statistics["psnr"]
        overall = totals.sum(dim=0)
        if overall[1] > 0:
            metrics.add_metric(ScalarMetricEntry(tag="eval/psnr", scalar=float(overall[0] / overall[1])))

        for source, (total, count) in zip(self.mixture.eval_sources, totals.tolist(), strict=True):
            if count > 0:
                metrics.add_metric(ScalarMetricEntry(tag=f"eval/psnr/{source.name}", scalar=total / count))

        return metrics
