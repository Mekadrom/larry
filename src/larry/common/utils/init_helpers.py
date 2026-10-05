import importlib
import pkgutil
from types import ModuleType


def import_submodules(package: ModuleType | str, skip: tuple[str, ...] = ()) -> None:
    if isinstance(package, str):
        package = importlib.import_module(package)

    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        if info.name.rsplit(".", 1)[-1] == "__main__" or info.name.startswith(skip):
            continue
        importlib.import_module(info.name)
