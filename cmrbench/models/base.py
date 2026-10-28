"""Minimal reconstruction model adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult


@dataclass
class ModelInfo:
    model_name: str
    model_version: str | None = None
    checkpoint: str | None = None
    checkpoint_identifier: str | None = None
    git_commit: str | None = None
    parameter_count: int | None = None
    device: str | None = None
    precision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReconstructionModel(ABC):
    """Adapter implemented independently for each reconstruction model."""

    @abstractmethod
    def load(self) -> None:
        """Load model code and weights without changing global sample data."""

    @abstractmethod
    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        """Reconstruct and restore output to canonical ``[S,T,X,Y]``."""

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Return available model and execution provenance."""
