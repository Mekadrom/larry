import inspect
from typing import Any, ClassVar

from larry.common.utils.types import TypeRegistry


class Registrable:
    REGISTRY: ClassVar[TypeRegistry[Any]]

    def __init_subclass__(cls, *, root: bool = False, key: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if root:
            cls.REGISTRY = TypeRegistry(cls)
            cls.REGISTRY.default = cls
        elif not inspect.isabstract(cls):
            cls.REGISTRY[key or cls.__name__] = cls
