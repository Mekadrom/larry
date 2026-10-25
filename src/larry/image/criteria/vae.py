import contextlib
from typing import Any

import torch
import torch.nn.functional as F

from larry.common.criteria.criterion import LossTerm, TermCriterion
from larry.image.config.criterion.image_criterion_configs import ImageVAECriterionConfig
from larry.image.model.vae import ImageVAEOutput


class ImageVAECriterion(TermCriterion[ImageVAEOutput, ImageVAECriterionConfig]):
    @staticmethod
    def per_row(values: torch.Tensor) -> torch.Tensor:
        return values.float().flatten(1).mean(dim=1)

    @staticmethod
    def term(weight: float, values: torch.Tensor) -> LossTerm:
        return LossTerm(weight=weight, sums=values, counts=torch.ones_like(values))

    @staticmethod
    def gradients_for(weight: float) -> contextlib.AbstractContextManager[Any]:
        return contextlib.nullcontext() if weight != 0 else torch.no_grad()

    def adversarial_weight(self, step: int | None) -> float:
        if step is not None and step < self.config.adversarial_start_step:
            return 0.0

        return self.config.adversarial_weight

    def frequency_values(self, reconstructions: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=images.device.type, enabled=False):
            fake = torch.fft.rfft2(reconstructions.float(), norm="ortho").abs()
            real = torch.fft.rfft2(images.float(), norm="ortho").abs()
            return self.per_row((fake.log1p() - real.log1p()).abs())

    def kl_values(self, output: ImageVAEOutput, images: torch.Tensor) -> torch.Tensor:
        mean, logvar = output.mean.float(), output.logvar.float()
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar)
        return kl.flatten(1).sum(dim=1) / images[0].numel()

    def discriminator_values(self, output: ImageVAEOutput) -> torch.Tensor:
        real = self.per_row(F.relu(1 - output.real_logits.float()))
        fake = self.per_row(F.relu(1 + output.fake_logits.float()))
        return 0.5 * (real + fake)

    def loss_terms(
            self,
            model_output: ImageVAEOutput,
            images: torch.Tensor | None = None,
            step: int | None = None,
            **batch_kwargs: Any,
    ) -> dict[str, LossTerm]:
        if images is None:
            raise ValueError("the image VAE criterion needs the input images")

        config = self.config
        reconstructions = model_output.reconstructions
        with self.gradients_for(config.frequency_weight):
            frequency = self.frequency_values(reconstructions, images)

        equivariance = model_output.equivariant_reconstructions.float() - model_output.equivariant_targets.float()
        return {
            "reconstruction": self.term(
                config.reconstruction_weight, self.per_row((reconstructions.float() - images.float()).abs())
            ),
            "frequency": self.term(config.frequency_weight, frequency),
            "kl": self.term(config.kl_weight, self.kl_values(model_output, images)),
            "equivariance": self.term(config.equivariance_weight, self.per_row(equivariance.abs())),
            "adversarial": self.term(
                self.adversarial_weight(step), self.per_row(-model_output.generator_logits.float())
            ),
            "discriminator": self.term(config.discriminator_weight, self.discriminator_values(model_output)),
        }
