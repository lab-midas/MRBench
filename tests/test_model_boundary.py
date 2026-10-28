from __future__ import annotations

import unittest

import numpy as np

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord


class DummyAdapter(ReconstructionModel):
    def load(self) -> None:
        self.loaded = True

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        private_copy = np.array(input.kspace, copy=True)
        canonical = np.abs(private_copy).sum(axis=-1)
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=canonical,
            model_name="dummy",
            transforms=[TransformRecord("coil_magnitude_sum", {"test_only": True})],
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(model_name="dummy", device="cpu", precision="complex64")


class ModelBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kspace = np.ones((2, 3, 8, 6, 4), dtype=np.complex64)
        self.mask = np.ones((2, 3, 8, 6), dtype=bool)
        self.input = ReconstructionInput("subject", self.kspace, self.mask)

    def test_dummy_adapter_accepts_reconstruction_input(self) -> None:
        result = DummyAdapter().reconstruct(self.input)
        self.assertEqual(result.subject_id, "subject")

    def test_dummy_adapter_returns_canonical_geometry(self) -> None:
        result = DummyAdapter().reconstruct(self.input)
        self.assertEqual(result.reconstruction.shape, (2, 3, 8, 6))

    def test_adapter_preprocessing_does_not_mutate_input(self) -> None:
        before = self.kspace.copy()
        DummyAdapter().reconstruct(self.input)
        np.testing.assert_array_equal(self.input.kspace, before)


if __name__ == "__main__":
    unittest.main()

