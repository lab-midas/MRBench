"""Test-only adapter used to exercise orchestration before real integrations."""

from __future__ import annotations

import numpy as np

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord


class PlaceholderModel(ReconstructionModel):
    """Return zeros in canonical geometry; never use for scientific results."""

    def load(self) -> None:
        return None

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        shape = tuple(int(size) for size in input.kspace.shape[:4])
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=np.zeros(shape, dtype=np.float32),
            model_name="placeholder",
            transforms=[
                TransformRecord(
                    name="placeholder_zero_output",
                    parameters={"test_only": True, "scientific_result": False},
                )
            ],
            metadata={"test_only": True, "scientific_result": False},
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="placeholder",
            model_version="0.1-test-only",
            device="cpu",
            precision="float32",
            metadata={"scientific_model": False},
        )

