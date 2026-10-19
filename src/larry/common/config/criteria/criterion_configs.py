import dataclasses
from typing import ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class CriterionConfig(Registrable, root=True):
    """Superclass for all criterion configs."""

    REGISTRY: ClassVar[TypeRegistry[CriterionConfig]]


@dataclasses.dataclass(kw_only=True)
class TeacherForcedVocabularyCrossEntropyCriterionConfig(CriterionConfig):
    """Config for simple causal teacher forced criterion (Cross Entropy on next position targets)."""

    label_smoothing: float = 0.0
