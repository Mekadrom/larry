import inspect
from importlib.resources import Package


def get_classes(module: Package) -> dict[str, type]:
    """Takes a package, iterates its members looking for classes owned by that module. Returns a {name: type} dict."""
    classes_defined_here = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes_defined_here[name] = obj
    return classes_defined_here
