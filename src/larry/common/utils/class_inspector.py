import importlib
import inspect
import pkgutil
from types import ModuleType


def get_classes[T](base: type[T], module: ModuleType) -> dict[str, type[T]]:
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and issubclass(obj, base)
    }


def get_classes_recursive[T](base: type[T], package: str | ModuleType) -> dict[str, type[T]]:
    if isinstance(package, str):
        package = importlib.import_module(package)
    found = get_classes(base, package)
    if hasattr(package, "__path__"):  # only packages have submodules
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            module = importlib.import_module(info.name)
            found.update(get_classes(base, module))
    return found
