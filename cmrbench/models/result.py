"""Canonical reconstruction output and transparent transform records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cmrbench.data.sample import Tensor, _shape


@dataclass(frozen=True)
class TransformRecord:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Model output restored to canonical ``[S,T,X,Y]`` geometry."""

    subject_id: str
    reconstruction: Tensor
    model_name: str
    runtime_seconds: float | None = None
    transforms: list[TransformRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reconstruction_shape = _shape(self.reconstruction, "reconstruction")
        if len(reconstruction_shape) != 4:
            raise ValueError(
                "reconstruction must have canonical [S,T,X,Y] shape; "
                f"received {reconstruction_shape}"
            )
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
