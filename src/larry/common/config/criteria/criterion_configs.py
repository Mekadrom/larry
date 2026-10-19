import dataclasses

from larry.common.config.config import Config


@dataclasses.dataclass(kw_only=True)
class CriterionConfig(Config):
    """Superclass for all criterion configs."""



@dataclasses.dataclass(kw_only=True)
class TeacherForcedVocabularyCrossEntropyCriterionConfig(CriterionConfig):
    """Config for simple causal teacher forced criterion (Cross Entropy on next position targets)."""

    label_smoothing: float = 0.0
