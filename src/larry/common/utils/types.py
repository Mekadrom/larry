import inspect
from collections.abc import Callable
from typing import Literal, TypedDict, ClassVar

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

    def __init__(self, base: type[T]) -> None:
        """Initializes as a subclass of dict, providing a default default."""
        super().__init__()
        self.base = base
        self.default: type[T] | None = None

    def __missing__(self, key: str) -> type[T]:
        """Tries to recover when a key is missing, assuming that the modules weren't walked previously."""
        if not TypeRegistry._all_imported:
            TypeRegistry._all_imported = True
            import_submodules("larry")
            if key in self:
                return self[key]

        raise KeyError(f"{key!r} not registered for {self.base.__qualname__}; known: {sorted(self)}")

    def resolve(self, key: str | None = None) -> type[T]:
        """Resolves a requested key by providing non-abstract types."""
        if key is not None:
            return self[key]

        if inspect.isabstract(self.base):
            raise KeyError(f"{self.base.__qualname__} is abstract; a registry key is required")

        return self.base


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
