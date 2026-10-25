from typing import Any, cast

import torch

from larry.common.training import trainers
from larry.common.training.viz_callbacks import VizCallback, pick_rows
from larry.image.config.training.image_trainer_configs import ImageVAEVizConfig
from larry.image.data.dataloading.indexed_image_dataset import ImageCollator
from larry.image.model.vae import ImageVAE, SpatialTransform


def psnr(reconstructions: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    mse = (reconstructions.float().clamp(-1, 1) - images.float()).square().flatten(1).mean(dim=1)
    return 10 * torch.log10(4.0 / mse.clamp(min=1e-10))


def to_unit(images: torch.Tensor) -> torch.Tensor:
    return ((images.float().clamp(-1, 1) + 1) / 2).cpu()


class ImageVAEVizCallback(VizCallback):
    def __init__(
            self,
            trainer: trainers.TrainerBase[Any, Any, Any],
            viz_steps: int,
            config: ImageVAEVizConfig,
    ) -> None:
        super().__init__(trainer, viz_steps)
        self.config = config
        self.dataset = trainer.eval_loader.dataset
        self.rows = pick_rows(self.dataset.offsets, config.examples_per_source)
        self.references_logged = False

    def viz(self, step: int) -> None:
        if not self.trainer.accelerator.is_main_process or not self.rows:
            return

        self.log.info(f"Performing visualization at step {step}")
        items = [self.dataset[row] for row in self.rows]
        images = ImageCollator()(items)["images"].to(self.trainer.accelerator.device)
        model = cast(ImageVAE, self.trainer.accelerator.unwrap_model(self.trainer.model))
        transform = SpatialTransform(rotations=1, scale=0.5)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), self.trainer.accelerator.autocast():
                mean, _ = model.encode(images)
                reconstructions = model.decode(mean).float()
                equivariant = model.decode(transform.apply(mean, transform.latent_size(mean))).float()
        finally:
            model.train(was_training)

        targets = transform.apply(images, (equivariant.shape[-2], equivariant.shape[-1]))
        scores = psnr(reconstructions, images)
        equivariant_scores = psnr(equivariant, targets)
        sources = self.trainer.config.mixture_config.eval_sources
        for i, item in enumerate(items):
            tag = f"viz/{i}/{sources[item['source']].name}"
            if not self.references_logged:
                self.trainer.metrics.add_image(f"{tag}/real", to_unit(images[i]).permute(1, 2, 0).numpy(), step)

            pair = torch.cat([to_unit(reconstructions[i]), (to_unit(reconstructions[i]) - to_unit(images[i])).abs()], 2)
            self.trainer.metrics.add_image(f"{tag}/reconstruction", pair.permute(1, 2, 0).numpy(), step)
            self.trainer.metrics.add_image(
                f"{tag}/equivariant", to_unit(equivariant[i]).permute(1, 2, 0).numpy(), step
            )
            self.trainer.metrics.add_scalar(f"{tag}/psnr", float(scores[i]), step)

        latents = mean.float()
        self.trainer.metrics.add_scalar("viz/psnr", float(scores.mean()), step)
        self.trainer.metrics.add_scalar("viz/equivariant_psnr", float(equivariant_scores.mean()), step)
        self.trainer.metrics.add_scalar("viz/latent_std", float(latents.std()), step)
        self.trainer.metrics.add_scalar("viz/latent_mean_abs", float(latents.mean(dim=(0, 2, 3)).abs().mean()), step)
        self.references_logged = True
