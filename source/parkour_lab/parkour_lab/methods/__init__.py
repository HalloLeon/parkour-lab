"""Named learning backends, loaded only when selected.

External adapters register an installed ``parkour_lab.methods`` entry point
whose value is a module implementing MethodBackend. Installing a plugin grants
it code execution; an artifact cannot supply an arbitrary Python import path.
"""

from importlib import import_module
from importlib.metadata import entry_points


def get_backend(name):
    if not isinstance(name, str) or not name or not name.isidentifier():
        raise ValueError("Invalid method name")
    external = list(entry_points(group="parkour_lab.methods", name=name))
    if len(external) > 1 or (external and name == "roa"):
        raise ValueError(f"Ambiguous learning method: {name}")
    if name == "roa":
        return import_module(f"parkour_lab.methods.{name}")
    if external:
        return external[0].load()
    raise ValueError(f"Unknown learning method: {name}; install its adapter first")
