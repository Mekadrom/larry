from typing import Literal, Callable

type Modality = Literal["voice", "text", "image"]

type Factory[T] = Callable[..., T]
type Registry[K, V] = dict[K, V]
type StringRegistry[V] = Registry[str, V]
type FactoryRegistry[V] = StringRegistry[Factory[V]]
