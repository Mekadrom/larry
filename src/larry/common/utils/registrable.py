import inspect
from typing import Any, ClassVar

from larry.common.utils.types import TypeRegistry


class Registrable:
    """Helper class to allow a superclass to store a registry of all its subclass implementations."""

    REGISTRY: ClassVar[TypeRegistry[Any]]

    def __init_subclass__(cls, *, root: bool = False, key: str | None = None, **kwargs: Any) -> None:
        """The core of creating registries that are accessible from any subclass while still being a single instance."""
        super().__init_subclass__(**kwargs)
        if root:
            cls.REGISTRY = TypeRegistry(cls)
            cls.REGISTRY.default = cls
        elif not inspect.isabstract(cls):
            cls.REGISTRY[key or cls.__name__] = cls
