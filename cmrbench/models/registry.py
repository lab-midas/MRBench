"""Minimal model adapter registry."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from cmrbench.models.base import ReconstructionModel

ModelFactory = Callable[..., ReconstructionModel]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str, factory: ModelFactory) -> None:
        if name in self._factories:
            raise ValueError(f"model {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, **parameters: object) -> ReconstructionModel:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown model {name!r}; register its adapter before running"
            ) from exc
        return factory(**parameters)

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


def _lazy_adapter_factory(module_name: str, class_name: str) -> ModelFactory:
    """Import only the adapter selected for the current isolated runtime."""

    def create(**parameters: object) -> ReconstructionModel:
        module = import_module(module_name)
        factory = getattr(module, class_name)
        return factory(**parameters)

    return create


default_model_registry = ModelRegistry()
default_model_registry.register(
    "placeholder",
    _lazy_adapter_factory(
        "cmrbench.models.adapters.placeholder", "PlaceholderModel"
    ),
)
default_model_registry.register(
    "zero_filled",
    _lazy_adapter_factory(
        "cmrbench.models.adapters.zero_filled", "ZeroFilledAdapter"
    ),
)
default_model_registry.register(
    "ram", _lazy_adapter_factory("cmrbench.models.adapters.ram", "RAMAdapter")
)
default_model_registry.register(
    "sdum", _lazy_adapter_factory("cmrbench.models.adapters.sdum", "SDUMAdapter")
)
default_model_registry.register(
    "cardiomm",
    _lazy_adapter_factory("cmrbench.models.adapters.cardiomm", "CardioMMAdapter"),
)
