import inspect
from collections.abc import Iterable, Callable
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
    _all_imported: ClassVar[bool] = False

    def __init__(self, base: type[T]) -> None:
        super().__init__()
        self.base = base
        self.default: type[T] | None = None

    def __missing__(self, key: str) -> type[T]:
        if not TypeRegistry._all_imported:
            TypeRegistry._all_imported = True
            import_submodules("larry")
            if key in self:
                return self[key]
        raise KeyError(f"{key!r} not registered for {self.base.__qualname__}; known: {sorted(self)}")

    def resolve(self, key: str | None = None) -> type[T]:
        if key is not None:
            return self[key]
        if inspect.isabstract(self.base):
            raise KeyError(f"{self.base.__qualname__} is abstract; a registry key is required")
        return self.base


type PreparedState = tuple[nn.Module, AcceleratedOptimizer, DataLoader, DataLoader, AcceleratedScheduler]


class EncodedImage(TypedDict):
    bytes: bytes
    path: str | None


class EncodedAudio(TypedDict):
    bytes: bytes
    path: str | None


def filter_by_type[T](i: Iterable[object], t: type[T]) -> list[T]:
    return [o for o in i if isinstance(o, t)]


def filter_by_ntype[T](i: Iterable[T], t: type[T]) -> list[T]:
    return [o for o in i if not isinstance(o, t)]


def filter_by_type_exact[T](i: Iterable[object], t: type[T]) -> list[T]:
    return [o for o in i if type(o) is t]


def filter_by_type_nexact[T](i: Iterable[T], t: type[T]) -> list[T]:
    return [o for o in i if type(o) is not t]
