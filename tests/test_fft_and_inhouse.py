from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cmrbench.data.inhouse_cine import (
    build_inhouse_cine_sample,
    inhouse_array_paths,
    load_inhouse_arrays,
)
from cmrbench.utils.fft import compare_fft_conventions, fft2c, ifft2c


class FFTAndInHouseTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(12)
        self.images = (
            generator.normal(size=(2, 3, 8, 6, 4))
            + 1j * generator.normal(size=(2, 3, 8, 6, 4))
        ).astype(np.complex64)

    def test_numpy_fft_round_trip(self) -> None:
        reconstructed = ifft2c(fft2c(self.images))
        np.testing.assert_allclose(reconstructed, self.images, rtol=1e-6, atol=1e-7)

    def test_single_slice_fft_round_trip(self) -> None:
        reconstructed = ifft2c(fft2c(self.images[0]))
        np.testing.assert_allclose(reconstructed, self.images[0], rtol=1e-6, atol=1e-7)

    def test_legacy_layout_and_direct_axes_are_equivalent(self) -> None:
        regression = compare_fft_conventions(self.images)
        self.assertTrue(regression.passed)
        self.assertLessEqual(regression.fft.max_absolute_difference, 1e-7)
        self.assertLessEqual(regression.ifft.max_absolute_difference, 1e-7)

    def test_subject_stem_and_established_filenames(self) -> None:
        image_path, sensitivity_path = inhouse_array_paths(
            "folder/Sub0116.h5", "/data/preprocess"
        )
        self.assertEqual(image_path.name, "norm_img_Sub0116.txt.npy")
        self.assertEqual(sensitivity_path.name, "cc_smap_15_Sub0116.txt.npy")

    def test_temporary_real_layout_arrays_load_and_build_sample(self) -> None:
        norm_imgc = self.images[..., :1]
        sensitivity_maps = np.ones((2, 1, 8, 6, 4), dtype=np.complex64) / 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "norm_img_SubX.txt.npy", norm_imgc)
            np.save(root / "cc_smap_15_SubX.txt.npy", sensitivity_maps)
            loaded_image, loaded_maps = load_inhouse_arrays("SubX.h5", root)
            sample = build_inhouse_cine_sample(
                "SubX.h5", loaded_image, loaded_maps, manifest_metadata={"cohort_type": "test"}
            )
        self.assertEqual(sample.subject_id, "SubX")
        self.assertEqual(sample.kspace.shape, (2, 3, 8, 6, 4))
        self.assertEqual(sample.reference.shape, (2, 3, 8, 6))
        self.assertEqual(sample.sensitivity_maps.shape, (2, 1, 8, 6, 4))
        self.assertIsNone(sample.mask)
        self.assertEqual(sample.metadata["matrix_x"], 8)
        self.assertEqual(sample.metadata["matrix_y"], 6)
        self.assertEqual(sample.metadata["number_of_coils"], 4)

    def test_inhouse_shape_validation_rejects_non_singleton_image_coil(self) -> None:
        maps = np.ones((2, 1, 8, 6, 4), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, r"\[S,T,X,Y,1\]"):
            build_inhouse_cine_sample("SubX", self.images, maps)


if __name__ == "__main__":
    unittest.main()

