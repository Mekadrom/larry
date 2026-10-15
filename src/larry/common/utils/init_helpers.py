import importlib
import pkgutil
from types import ModuleType


def import_submodules(module: ModuleType | str, skip: tuple[str, ...] = ()) -> None:
    """Walks all modules within the provided module, importing each visited. Optionally skips some package paths."""
    if isinstance(module, str):
        module = importlib.import_module(module)

    for info in pkgutil.walk_packages(module.__path__, prefix=f"{module.__name__}."):
        if info.name.rsplit(".", 1)[-1] == "__main__" or info.name.startswith(skip):
            continue

        importlib.import_module(info.name)
