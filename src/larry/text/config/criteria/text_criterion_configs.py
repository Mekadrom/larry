import dataclasses

from larry.common.config.criteria.criterion_configs import CriterionConfig


@dataclasses.dataclass(kw_only=True)
class TextCriterionConfig(CriterionConfig):
    ...
