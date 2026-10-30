import dataclasses

from larry.common.config.criteria.criterion_configs import CriterionConfig


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalCriterionConfig(CriterionConfig):
    text_weight: float = 1.0
    voice_weight: float = 1.0
    image_weight: float = 1.0
    text_label_smoothing: float = 0.1
    voice_label_smoothing: float = 0.1
    voiced_weight: float = 1.0
    contour_weight: float = 1.0
    aperiodicity_weight: float = 1.0
