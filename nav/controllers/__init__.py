"""Controller registry. Drop a module in this package with @register("name") and it is
selectable via `--controller name`; nothing else needs to change."""
import importlib
import pkgutil

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def _discover():
    for m in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{m.name}")


def names():
    _discover()
    return sorted(REGISTRY)


def make(name, cfg, **params):
    _discover()
    if name not in REGISTRY:
        raise SystemExit(f"unknown controller '{name}', have: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name](cfg, **params)
