"""Minimal registry for configuration-selected sampling protocols."""

from __future__ import annotations

from typing import Callable

from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.cardiomm import (
    CardioMMKtGaussianSamplingProtocol,
    CardioMMKtRadialSamplingProtocol,
    CardioMMKtUniformSamplingProtocol,
    CardioMMUniformSamplingProtocol,
    CardioMMRandomSamplingProtocol,
    CardioMMRadialSamplingProtocol,
)
from cmrbench.sampling.external import ExternalMaskSamplingProtocol
from cmrbench.sampling.nvidia import (
    CanonicalUniformSamplingProtocol,
    KtGaussianSamplingProtocol,
    KtRadialSamplingProtocol,
    KtUniformSamplingProtocol,
    UniformSamplingProtocol,
    RandomSamplingProtocol,
    EquispacedSamplingProtocol,
)
from cmrbench.sampling.vista import VISTASamplingProtocol

SamplingFactory = Callable[..., SamplingProtocol]


class SamplingRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, SamplingFactory] = {}
        self._implementation_factories: dict[tuple[str, str], SamplingFactory] = {}

    def register(
        self,
        name: str,
        factory: SamplingFactory,
        *,
        implementation: str | None = None,
    ) -> None:
        if implementation is not None:
            key = (name, implementation)
            if key in self._implementation_factories:
                raise ValueError(
                    f"sampling method/implementation {key!r} is already registered"
                )
            self._implementation_factories[key] = factory
            return
        if name in self._factories:
            raise ValueError(f"sampling method {name!r} is already registered")
        self._factories[name] = factory

    def create(
        self,
        name: str,
        *,
        implementation_selector: str | None = None,
        **parameters: object,
    ) -> SamplingProtocol:
        if implementation_selector is not None:
            factory = self._implementation_factories.get(
                (name, implementation_selector)
            )
            if factory is not None:
                return factory(**parameters)
            available = self.implementations(name)
            if available:
                raise KeyError(
                    f"unknown implementation {implementation_selector!r} for sampling "
                    f"method {name!r}; expected one of {available}"
                )
        available = self.implementations(name)
        if available and implementation_selector is None and name not in self._factories:
            raise KeyError(
                f"sampling method {name!r} requires an implementation selector; "
                f"expected one of {available}"
            )
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown sampling method {name!r}") from exc
        return factory(**parameters)

    def names(self) -> tuple[str, ...]:
        """Return stable canonical generator/protocol identifiers."""

        ordered = list(self._factories)
        for method, _implementation in self._implementation_factories:
            if method not in ordered:
                ordered.append(method)
        return tuple(ordered)

    def implementations(self, name: str) -> tuple[str, ...]:
        return tuple(
            implementation
            for method, implementation in self._implementation_factories
            if method == name
        )


default_sampling_registry = SamplingRegistry()
default_sampling_registry.register("vista", VISTASamplingProtocol)
default_sampling_registry.register(
    "vista",
    lambda **parameters: VISTASamplingProtocol(
        implementation="precomputed", **parameters
    ),
    implementation="precomputed",
)
default_sampling_registry.register(
    "vista",
    lambda **parameters: VISTASamplingProtocol(
        implementation="cmrbench", **parameters
    ),
    implementation="cmrbench",
)
default_sampling_registry.register("external_mask", ExternalMaskSamplingProtocol)
default_sampling_registry.register("uniform", CanonicalUniformSamplingProtocol)
default_sampling_registry.register(
    "uniform", UniformSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "uniform", CardioMMUniformSamplingProtocol, implementation="cardiomm"
)
default_sampling_registry.register(
    "kt_uniform", KtUniformSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "kt_gaussian", KtGaussianSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "kt_radial", KtRadialSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "kt_uniform", CardioMMKtUniformSamplingProtocol, implementation="cardiomm"
)
default_sampling_registry.register(
    "kt_gaussian", CardioMMKtGaussianSamplingProtocol, implementation="cardiomm"
)
default_sampling_registry.register(
    "kt_radial", CardioMMKtRadialSamplingProtocol, implementation="cardiomm"
)
default_sampling_registry.register(
    "random", RandomSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "random", CardioMMRandomSamplingProtocol, implementation="cardiomm"
)
default_sampling_registry.register(
    "equispaced", EquispacedSamplingProtocol, implementation="sdum"
)
default_sampling_registry.register(
    "radial", CardioMMRadialSamplingProtocol, implementation="cardiomm"
)
