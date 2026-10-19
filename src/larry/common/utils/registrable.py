import inspect
import types
import typing
from typing import Any, ClassVar

from larry.common.utils.types import TypeRegistry


class Registrable:
    """Helper class to allow a superclass to store a registry of all its subclass implementations."""

    REGISTRY: ClassVar[TypeRegistry[Any]]

    def __init_subclass__(
            cls,
            *,
            root: bool = False,
            key: str | None = None,
            config_root: type | None = None,
            register: bool = True,
            **kwargs: Any,
    ) -> None:
        """The core of creating registries that are accessible from any subclass while still being a single instance."""
        super().__init_subclass__(**kwargs)
        if root:
            cls.REGISTRY = TypeRegistry(cls, config_root)
            return

        if not register:
            return

        if not inspect.isabstract(cls):
            cls.REGISTRY[key or cls.__name__] = cls

        config_root = cls.REGISTRY.config_root
        if config_root is not None:
            for config_type in Registrable._declared_config_types(cls, config_root):
                cls.REGISTRY.claim(config_type, cls)

    @staticmethod
    def _declared_config_types(cls: type, config_root: type) -> set[type]:
        declared = set()
        for base in types.get_original_bases(cls):
            for arg in typing.get_args(base):
                if isinstance(arg, typing.TypeVar):
                    if not arg.has_default():
                        continue
                    arg = arg.__default__

                arg = typing.get_origin(arg) or arg
                if isinstance(arg, type) and issubclass(arg, config_root):
                    declared.add(arg)

        return declared
