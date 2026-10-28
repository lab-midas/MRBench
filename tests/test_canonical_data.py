from __future__ import annotations

import unittest

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.models.input import ReconstructionInput


class CanonicalDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kspace = np.ones((2, 3, 8, 6, 4), dtype=np.complex64)

    def test_valid_five_dimensional_kspace_is_accepted(self) -> None:
        sample = CineMRISample("subject", self.kspace)
        self.assertEqual(sample.kspace.shape, (2, 3, 8, 6, 4))

    def test_incorrect_kspace_dimensionality_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"exactly 5 dimensions.*\[S,T,Kx,Ky,C\]"):
            CineMRISample("subject", np.ones((3, 8, 6, 4), dtype=np.complex64))

    def test_compatible_mask_is_accepted(self) -> None:
        mask = np.ones(self.kspace.shape[:4], dtype=bool)
        sample = CineMRISample("subject", self.kspace, mask=mask)
        self.assertIs(sample.mask, mask)

    def test_mask_with_coil_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no coil dimension"):
            CineMRISample("subject", self.kspace, mask=np.ones(self.kspace.shape))

    def test_time_varying_mask_is_supported(self) -> None:
        mask = np.zeros(self.kspace.shape[:4], dtype=bool)
        mask[:, 0, ::2, :] = True
        mask[:, 1, 1::2, :] = True
        mask[:, 2, ::3, :] = True
        sample = CineMRISample("subject", self.kspace, mask=mask)
        self.assertFalse(np.array_equal(sample.mask[:, 0], sample.mask[:, 1]))

    def test_optional_reference_is_supported(self) -> None:
        reference = np.ones((2, 3, 10, 7), dtype=np.float32)
        sample = CineMRISample("subject", self.kspace, reference=reference)
        self.assertIs(sample.reference, reference)

    def test_optional_sensitivity_maps_are_supported(self) -> None:
        static = np.ones((2, 1, 8, 6, 4), dtype=np.complex64)
        time_dependent = np.ones((2, 3, 8, 6, 4), dtype=np.complex64)
        self.assertIs(
            CineMRISample("static", self.kspace, sensitivity_maps=static).sensitivity_maps,
            static,
        )
        self.assertIs(
            CineMRISample(
                "time_dependent", self.kspace, sensitivity_maps=time_dependent
            ).sensitivity_maps,
            time_dependent,
        )

    def test_reconstruction_input_rejects_mask_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            ReconstructionInput(
                "subject", self.kspace, np.ones((2, 3, 8, 5), dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
