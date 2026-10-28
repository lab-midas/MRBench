from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.data.sensitivity import (
    detect_realized_centered_acs,
    prepare_sensitivity_maps,
)
from cmrbench.experiment.prepare import (
    attach_prepared_sensitivity_maps,
    prepare_reconstruction_input,
)
from cmrbench.sampling.external import ExternalMaskSamplingProtocol
from cmrbench.sampling.nvidia import (
    NVIDIAStaticLineSamplingProtocol,
    NVIDIASamplingProtocol,
)


def line_mask(ky: int, start: int, stop: int, *, kx: int = 8) -> np.ndarray:
    mask = np.zeros((1, 1, kx, ky), dtype=bool)
    mask[..., start:stop] = True
    return mask


class RealizedACSTests(unittest.TestCase):
    def test_odd_geometry_detects_largest_contiguous_center_strip(self) -> None:
        acs = detect_realized_centered_acs(line_mask(9, 2, 7))
        self.assertEqual((acs.ky_start, acs.ky_stop, acs.extent), (2, 7, 5))
        self.assertEqual(acs.center_indices, (4,))

    def test_even_geometry_requires_both_center_adjacent_lines(self) -> None:
        acs = detect_realized_centered_acs(line_mask(10, 3, 7))
        self.assertEqual((acs.ky_start, acs.ky_stop, acs.extent), (3, 7, 4))
        self.assertEqual(acs.center_indices, (4, 5))
        incomplete = line_mask(10, 4, 5)
        with self.assertRaisesRegex(ValueError, "central Ky line.*missing"):
            detect_realized_centered_acs(incomplete)

    def test_fully_sampled_lines_outside_center_component_are_ignored(self) -> None:
        mask = line_mask(12, 4, 8)
        mask[..., 0] = True
        mask[..., 10] = True
        acs = detect_realized_centered_acs(mask)
        self.assertEqual((acs.ky_start, acs.ky_stop, acs.extent), (4, 8, 4))

    def test_random_uniform_and_external_realized_masks_are_detected(self) -> None:
        shape = (1, 1, 8, 32, 3)
        random = NVIDIAStaticLineSamplingProtocol(
            family="random", center_fraction=0.125
        ).generate(shape, 4, seed=7)
        uniform = NVIDIASamplingProtocol(
            family="uniform", ncalib=4, implementation="canonical"
        ).generate(shape, 4, seed=7)
        external_mask = line_mask(32, 14, 18)
        external = ExternalMaskSamplingProtocol(
            mask=external_mask, layout="S,T,Kx,Ky"
        ).generate(shape, 4, seed=7)

        for sampling in (random, uniform, external):
            acs = detect_realized_centered_acs(sampling.mask)
            self.assertGreaterEqual(acs.extent, 4)

    def test_partial_kx_center_line_is_not_acs(self) -> None:
        mask = line_mask(9, 3, 6)
        mask[..., 4] = False
        mask[..., 0, 4] = True
        with self.assertRaisesRegex(ValueError, "no complete central ACS"):
            detect_realized_centered_acs(mask)


class SensitivityStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kspace = np.ones((1, 1, 8, 10, 3), dtype=np.complex64)
        self.mask = line_mask(10, 3, 7)

    def test_provided_maps_are_validated_and_values_remain_identical(self) -> None:
        maps = (
            np.arange(1 * 1 * 8 * 10 * 3).reshape(1, 1, 8, 10, 3)
            + 1j
        ).astype(np.complex64)
        result = prepare_sensitivity_maps(
            kspace=self.kspace,
            mask=self.mask,
            provided_maps=maps,
            config="provided",
            prewhitened=True,
        )
        self.assertIs(result.maps, maps)
        np.testing.assert_array_equal(result.maps, maps)
        self.assertEqual(result.provenance["normalization"], "none; values preserved exactly")
        self.assertTrue(result.provenance["prewhitened"])

    def test_provided_maps_reject_spatial_and_coil_mismatch(self) -> None:
        bad_spatial = np.ones((1, 1, 7, 10, 3), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "spatial dimensions"):
            prepare_sensitivity_maps(
                kspace=self.kspace,
                mask=self.mask,
                provided_maps=bad_spatial,
                config="provided",
            )

    def test_provided_maps_reject_rank_slice_and_time_mismatch(self) -> None:
        invalid_values = (
            (np.ones((1, 8, 10, 3), dtype=np.complex64), "must have"),
            (np.ones((2, 1, 8, 10, 3), dtype=np.complex64), "slice dimension"),
            (np.ones((1, 2, 8, 10, 3), dtype=np.complex64), "time dimension"),
        )
        for maps, message in invalid_values:
            with self.subTest(shape=maps.shape):
                with self.assertRaisesRegex(ValueError, message):
                    prepare_sensitivity_maps(
                        kspace=self.kspace,
                        mask=self.mask,
                        provided_maps=maps,
                        config="provided",
                    )
        bad_coils = np.ones((1, 1, 8, 10, 2), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "coil dimension"):
            prepare_sensitivity_maps(
                kspace=self.kspace,
                mask=self.mask,
                provided_maps=bad_coils,
                config="provided",
            )

    def test_provided_maps_reject_noncomplex_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be complex"):
            prepare_sensitivity_maps(
                kspace=self.kspace,
                mask=self.mask,
                provided_maps=np.ones((1, 1, 8, 10, 3), dtype=np.float32),
                config="provided",
            )
        nonfinite = np.ones((1, 1, 8, 10, 3), dtype=np.complex64)
        nonfinite[0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            prepare_sensitivity_maps(
                kspace=self.kspace,
                mask=self.mask,
                provided_maps=nonfinite,
                config="provided",
            )

    def test_espirit_uses_only_realized_mask_and_selected_acs(self) -> None:
        observed: list[np.ndarray] = []

        def estimator(value: np.ndarray, **_kwargs: object) -> np.ndarray:
            observed.append(value.copy())
            scale = np.sum(value)
            return np.ones_like(value) * (scale if scale != 0 else 1 + 0j)

        first_full = self.kspace.copy()
        second_full = self.kspace.copy()
        outside = ~np.broadcast_to(self.mask[..., None], second_full.shape)
        second_full[outside] = 1000 + 2000j
        first = prepare_sensitivity_maps(
            kspace=first_full,
            mask=self.mask,
            provided_maps=None,
            config={"strategy": "espirit", "calib_width": 4},
            estimator=estimator,
            backend_version="fake-1",
        )
        second = prepare_sensitivity_maps(
            kspace=second_full,
            mask=self.mask,
            provided_maps=None,
            config={"strategy": "espirit", "calib_width": 4},
            estimator=estimator,
            backend_version="fake-1",
        )

        np.testing.assert_array_equal(observed[0], observed[1])
        np.testing.assert_array_equal(first.maps, second.maps)
        self.assertTrue(np.all(observed[0][:, :, :3] == 0))
        self.assertTrue(np.all(observed[0][:, :, 7:] == 0))
        self.assertEqual(
            first.provenance["calibration"]["source"],
            "realized mask and masked/acquired k-space only",
        )
        self.assertEqual(first.provenance["source"], "espirit")
        self.assertEqual(first.provenance["actual_acs_width"], 4)
        self.assertTrue(
            first.provenance["calibration"]["outside_selected_acs_zeroed"]
        )

    def test_attach_step_runs_after_existing_sampling_and_preserves_contract(self) -> None:
        full = np.ones((1, 1, 8, 10, 3), dtype=np.complex64)
        sample = CineMRISample("static", full)
        external = ExternalMaskSamplingProtocol(
            mask=self.mask, layout="S,T,Kx,Ky"
        )
        measured = prepare_reconstruction_input(
            sample,
            mode="retrospective",
            protocol=external,
            acceleration=2.5,
            seed=2,
        )
        seen: list[np.ndarray] = []

        def estimator(value: np.ndarray, **_kwargs: object) -> np.ndarray:
            seen.append(value.copy())
            return np.ones_like(value)

        final = attach_prepared_sensitivity_maps(
            measured,
            config={"strategy": "espirit", "calib_width": 4},
            estimator=estimator,
            backend_version="fake-2",
        )

        self.assertIs(final.kspace, measured.kspace)
        self.assertIs(final.mask, measured.mask)
        self.assertIs(final.sampling, measured.sampling)
        self.assertEqual(final.sensitivity_maps.shape, (1, 1, 8, 10, 3))
        self.assertEqual(
            final.metadata["cmrbench_sensitivity_maps"]["backend_version"],
            "fake-2",
        )
        self.assertEqual(len(seen), 1)

    @unittest.skipUnless(
        importlib.util.find_spec("sigpy") is not None,
        "sigpy optional dependency is not installed",
    )
    def test_sigpy_espirit_smoke(self) -> None:
        height = width = 16
        y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        image = np.exp(-4 * (x**2 + y**2)).astype(np.complex64)
        sensitivities = np.stack(
            [np.ones_like(image), np.exp(1j * np.pi * x)], axis=0
        ).astype(np.complex64)
        coil_images = sensitivities * image[None]
        coil_kspace = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(coil_images, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        )
        canonical = np.transpose(coil_kspace, (1, 2, 0))[None, None]
        mask = np.ones((1, 1, height, width), dtype=bool)
        result = prepare_sensitivity_maps(
            kspace=canonical,
            mask=mask,
            provided_maps=None,
            config={
                "source": "espirit",
                "implementation": "sigpy",
                "calibration": "sampling_acs",
                "calib_width": 12,
                "kernel_width": 4,
                "max_iterations": 10,
            },
        )
        self.assertEqual(result.maps.shape, canonical.shape)
        self.assertTrue(np.all(np.isfinite(result.maps)))
        self.assertEqual(result.provenance["backend"], "sigpy")


if __name__ == "__main__":
    unittest.main()
