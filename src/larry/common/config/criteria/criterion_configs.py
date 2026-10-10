import dataclasses
from typing import Any, ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class CriterionConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[CriterionConfig]]


@dataclasses.dataclass(kw_only=True)
class TeacherForcedVocabularyCrossEntropyCriterionConfig(CriterionConfig):
    label_smoothing: float = 0.0


@dataclasses.dataclass(kw_only=True)
class WeightedCompositeCriterionConfig(CriterionConfig):
    criterion_configs: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class CTCCriterionConfig(CriterionConfig):
    blank: int
    zero_infinity: bool = True
