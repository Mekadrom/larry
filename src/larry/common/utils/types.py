import inspect
from collections.abc import Callable
from typing import ClassVar, Literal, TypedDict

from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from torch import nn
from torch.utils.data import DataLoader

from larry.common.utils.init_helpers import import_submodules

type Modality = Literal["image", "text", "voice"]

type Factory[T] = Callable[..., T]
type Registry[K, V] = dict[K, V]
type StringRegistry[V] = Registry[str, V]
type FactoryRegistry[T] = StringRegistry[Factory[T]]


class TypeRegistry[T](dict[str, type[T]]):
    """Helper type for registries."""

    _all_imported: ClassVar[bool] = False

    def __init__(self, base: type[T], config_root: type | None = None) -> None:
        super().__init__()
        self.base = base
        self.config_root = config_root
        self.claims: dict[type, list[type[T]]] = {}

    def __missing__(self, key: str) -> type[T]:
        """Tries to recover when a key is missing, assuming that the modules weren't walked previously."""
        if self._import_all():
            return self[key]

        raise KeyError(f"{key!r} not registered for {self.base.__qualname__}; known: {sorted(self)}")

    @staticmethod
    def _import_all() -> bool:
        if TypeRegistry._all_imported:
            return False

        TypeRegistry._all_imported = True
        import_submodules("larry")
        return True

    def resolve(self, key: str | None = None) -> type[T]:
        """Resolves a requested key by providing non-abstract types."""
        if key is not None:
            return self[key]

        if inspect.isabstract(self.base):
            raise KeyError(f"{self.base.__qualname__} is abstract; a registry key is required")

        return self.base

    def claim(self, config_type: type, implementation: type[T]) -> None:
        claimants = self.claims.setdefault(config_type, [])
        if implementation not in claimants:
            claimants.append(implementation)

    def for_config(self, config: object) -> type[T]:
        implementation = self._nearest_claimant(type(config))
        if implementation is None and self._import_all():
            implementation = self._nearest_claimant(type(config))

        if implementation is None:
            raise KeyError(f"no {self.base.__qualname__} implementation declares {type(config).__qualname__}")

        return implementation

    def _nearest_claimant(self, config_type: type) -> type[T] | None:
        for ancestor in config_type.__mro__:
            concrete = [c for c in self.claims.get(ancestor, []) if not inspect.isabstract(c)]
            if len(concrete) > 1:
                names = sorted(c.__qualname__ for c in concrete)
                raise TypeError(f"{ancestor.__qualname__} is declared by several implementations: {names}")
            if concrete:
                return concrete[0]

        return None


type PreparedState = tuple[nn.Module, AcceleratedOptimizer, DataLoader, DataLoader, AcceleratedScheduler]


class EncodedImage(TypedDict):
    """Helper dataclass type for a pair of (bytes, path) representing an image."""

    bytes: bytes
    path: str | None


class EncodedAudio(TypedDict):
    """Helper dataclass type for a pair of (bytes, path) representing audio."""

    bytes: bytes
    path: str | None


type Color = int | tuple[int, ...] | str
