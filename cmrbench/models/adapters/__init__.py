"""Model-specific reconstruction adapters, imported only when selected."""

from importlib import import_module
from typing import Any

__all__ = ["PlaceholderModel", "RAMAdapter", "SDUMAdapter", "ZeroFilledAdapter"]


_LAZY_ADAPTERS = {
    "PlaceholderModel": (
        "cmrbench.models.adapters.placeholder",
        "PlaceholderModel",
    ),
    "RAMAdapter": ("cmrbench.models.adapters.ram", "RAMAdapter"),
    "SDUMAdapter": ("cmrbench.models.adapters.sdum", "SDUMAdapter"),
    "ZeroFilledAdapter": (
        "cmrbench.models.adapters.zero_filled",
        "ZeroFilledAdapter",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, class_name = _LAZY_ADAPTERS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value
