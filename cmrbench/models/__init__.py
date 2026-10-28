"""Model adapter boundary and canonical reconstruction objects.

Adapter classes are resolved lazily so data preparation and evaluation can use
the canonical input/result objects without importing any foundation-model
runtime.
"""

from importlib import import_module
from typing import Any

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord

__all__ = [
    "ModelInfo",
    "RAMAdapter",
    "SDUMAdapter",
    "ReconstructionInput",
    "ReconstructionModel",
    "ReconstructionResult",
    "TransformRecord",
    "ZeroFilledAdapter",
]


_LAZY_ADAPTERS = {
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
