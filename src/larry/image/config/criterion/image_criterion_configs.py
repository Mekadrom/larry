import dataclasses

from larry.common.config.criteria.criterion_configs import CriterionConfig


@dataclasses.dataclass(kw_only=True)
class ImageVAECriterionConfig(CriterionConfig):
    reconstruction_weight: float = 1.0
    frequency_weight: float = 0.0
    kl_weight: float = 1e-6
    equivariance_weight: float = 0.0
    equivariance_scales: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.75])
    adversarial_weight: float = 0.0
    adversarial_start_step: int = 0
    discriminator_weight: float = 1.0
