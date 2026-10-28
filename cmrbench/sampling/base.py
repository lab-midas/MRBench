"""Sampling protocol interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from cmrbench.sampling.result import SamplingResult

KSpaceShape = Tuple[int, int, int, int, int]


class SamplingProtocol(ABC):
    """Generate a coil-independent mask for retrospective experiments."""

    @abstractmethod
    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        """Return a canonical ``[S,T,Kx,Ky]`` mask and its provenance."""
