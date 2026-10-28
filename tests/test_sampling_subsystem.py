from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

import cmrbench.sampling as sampling_api
from cmrbench.models.adapters.sdum import SDUMConditioning, preflight_conditioning
from cmrbench.sampling.external import (
    ExternalMaskSamplingProtocol,
    canonicalize_external_mask,
)
from cmrbench.sampling.cartesian import RandomCartesianSampling
from cmrbench.sampling.cardiomm import (
    CardioMMSamplingProtocol,
    cardiomm_kt_gaussian_sampling,
    cardiomm_kt_uniform_sampling,
    cardiomm_uniform_sampling,
)
from cmrbench.sampling.nvidia import (
    CanonicalUniformSamplingProtocol,
    NVIDIASamplingProtocol,
    kt_gaussian_sampling,
    kt_uniform_sampling,
    uniform_sampling,
)
from cmrbench.sampling.registry import default_sampling_registry
from cmrbench.sampling.vista import VISTASamplingProtocol


SHAPE = (2, 5, 8, 6, 3)


class ExternalMaskTests(unittest.TestCase):
    def test_canonical_external_mask_is_accepted_without_axis_inference(self) -> None:
        mask = np.ones(SHAPE[:4], dtype=np.uint8)
        result = ExternalMaskSamplingProtocol(mask=mask).generate(
            SHAPE, acceleration=1
        )
        self.assertEqual(result.mask.shape, SHAPE[:4])
        self.assertEqual(result.mask.dtype, np.bool_)
        self.assertEqual(result.actual_acceleration, 1.0)
        self.assertEqual(result.parameters["input_layout"], "S,T,Kx,Ky")

    def test_bad_canonical_shape_is_rejected_not_guessed(self) -> None:
        compact = np.ones((SHAPE[3], SHAPE[1]), dtype=bool)
        with self.assertRaisesRegex(ValueError, "canonical mask must have shape"):
            canonicalize_external_mask(compact, shape=SHAPE)

    def test_compact_layout_conversion_requires_explicit_declaration(self) -> None:
        compact = np.zeros((SHAPE[3], SHAPE[1]), dtype=np.uint8)
        compact[::2] = 1
        canonical = canonicalize_external_mask(
            compact, shape=SHAPE, layout="Ky,T"
        )
        self.assertEqual(canonical.shape, SHAPE[:4])
        self.assertTrue(np.all(canonical[:, :, :, ::2]))
        self.assertFalse(np.any(canonical[:, :, :, 1::2]))

    def test_nonbinary_nonfinite_and_empty_masks_are_rejected(self) -> None:
        for value, message in (
            (np.full(SHAPE[:4], 0.5), "exact 0/1"),
            (np.full(SHAPE[:4], np.nan), "NaN or Inf"),
            (np.zeros(SHAPE[:4]), "no sampled support"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    canonicalize_external_mask(value, shape=SHAPE)

    def test_requested_and_measured_acceleration_stay_distinct(self) -> None:
        mask = np.zeros(SHAPE[:4], dtype=bool)
        mask[..., ::2] = True
        result = ExternalMaskSamplingProtocol(mask=mask).generate(
            SHAPE, acceleration=4
        )
        self.assertEqual(result.requested_acceleration, 4.0)
        self.assertEqual(result.actual_acceleration, 2.0)
        self.assertEqual(result.parameters["implementation"], "external")

    def test_external_actual_r_is_recomputed_not_trusted_from_metadata(self) -> None:
        mask = np.zeros(SHAPE[:4], dtype=bool)
        mask[..., ::2] = True
        result = ExternalMaskSamplingProtocol(
            mask=mask,
            implementation="cardiomm",
            parameters={"actual_acceleration": 999.0},
        ).generate(SHAPE, acceleration=4)
        self.assertEqual(result.actual_acceleration, 2.0)
        self.assertEqual(result.parameters["implementation"], "cardiomm")
        self.assertNotIn("actual_acceleration", result.parameters)
        self.assertEqual(
            result.parameters["declared_actual_acceleration_ignored"], 999.0
        )

    def test_npy_mask_path_loads_without_pickle_and_records_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npy"
            mask = np.zeros((SHAPE[1], SHAPE[2], SHAPE[3]), dtype=np.uint8)
            mask[..., ::2] = 1
            np.save(path, mask, allow_pickle=False)
            result = ExternalMaskSamplingProtocol(
                mask_path=path, layout="T,Kx,Ky", source="precomputed"
            ).generate(SHAPE, acceleration=4)
        self.assertEqual(result.mask.shape, SHAPE[:4])
        self.assertEqual(result.parameters["mask_path"], str(path.resolve()))
        self.assertEqual(len(result.parameters["mask_file_sha256"]), 64)

    def test_external_mask_requires_exactly_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ExternalMaskSamplingProtocol()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ExternalMaskSamplingProtocol(mask=np.ones(SHAPE[:4]), mask_path="mask.npy")


class GeneratedSamplingTests(unittest.TestCase):
    def test_registry_contains_external_vista_and_four_nvidia_families(self) -> None:
        names = default_sampling_registry.names()
        for name in (
            "external_mask",
            "vista",
            "uniform",
            "kt_uniform",
            "kt_gaussian",
            "kt_radial",
        ):
            self.assertIn(name, names)
        self.assertNotIn("random_cartesian", names)
        self.assertFalse(hasattr(sampling_api, "RandomCartesianSampling"))
        with self.assertRaisesRegex(KeyError, "unknown sampling method"):
            default_sampling_registry.create("random_cartesian")
        self.assertEqual(
            set(default_sampling_registry.implementations("uniform")),
            {"sdum", "cardiomm"},
        )


    def _assert_common(self, family: str, *, seed: int | None = 7) -> object:
        result = NVIDIASamplingProtocol(family=family, ncalib=2).generate(
            SHAPE, acceleration=4, seed=seed
        )
        self.assertEqual(result.mask.shape, SHAPE[:4])
        self.assertEqual(result.parameters["native_shape"], [8, 6, 5])
        self.assertEqual(result.parameters["native_layout"], "[Kx,Ky,T]")
        self.assertEqual(result.parameters["canonical_layout"], "[S,T,Kx,Ky]")
        self.assertEqual(result.parameters["upstream_license"], "Apache-2.0")
        self.assertEqual(result.parameters["implementation"], "sdum")
        self.assertEqual(
            result.parameters["upstream_commit"],
            "99be712b2e9fd34bf6a2f709373d311354fbda96",
        )
        np.testing.assert_array_equal(result.mask[0], result.mask[1])
        self.assertIsNotNone(result.actual_acceleration)
        return result

    def test_uniform_output_is_static_over_time_and_deterministic(self) -> None:
        first = self._assert_common("uniform")
        second = self._assert_common("uniform", seed=99)
        np.testing.assert_array_equal(first.mask, second.mask)
        for phase in range(1, SHAPE[1]):
            np.testing.assert_array_equal(first.mask[:, 0], first.mask[:, phase])

    def test_kt_uniform_output_is_dynamic_and_deterministic(self) -> None:
        first = self._assert_common("kt_uniform")
        second = self._assert_common("kt_uniform", seed=99)
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertFalse(np.array_equal(first.mask[:, 0], first.mask[:, 1]))

    def test_kt_gaussian_seed_is_reproducible(self) -> None:
        first = self._assert_common("kt_gaussian", seed=7)
        second = self._assert_common("kt_gaussian", seed=7)
        np.testing.assert_array_equal(first.mask, second.mask)
        with self.assertRaisesRegex(ValueError, "explicit recorded seed"):
            NVIDIASamplingProtocol(family="kt_gaussian", ncalib=2).generate(
                SHAPE, acceleration=4
            )

    def test_pinned_native_geometry_counts_actual_r_and_hashes(self) -> None:
        cases = (
            ("uniform", 4, uniform_sampling(192, 156, 25, 20, 4), 259200, 2.888888888889, "1d53850aa3081f7054dfe93c2b6bbc127d5d90e70470f675a5cb695d3866d54d"),
            ("uniform", 8, uniform_sampling(192, 156, 25, 20, 8), 182400, 4.105263157895, "046c888a75bc05d4c3f20b23af4eb26961cb5b9830e546bb1ea6aa14fc376cba"),
            ("kt_uniform", 4, kt_uniform_sampling(192, 156, 25, 20, 4), 259200, 2.888888888889, "7c9238479d86f354e50698aeeb7a73f990a088ef5c8b7df375898ece9af81f58"),
            ("kt_uniform", 8, kt_uniform_sampling(192, 156, 25, 20, 8), 177792, 4.211663066955, "388184fc5c29e97d1abb3e160ee29f3681239fa93063e3db5d500de2edf27850"),
            ("kt_gaussian", 4, kt_gaussian_sampling(192, 156, 25, 20, 4, 0.2, 1), 244032, 3.068450039339, "9b1e4ff430308ef764f1a3d351678d6834820fa77f7347c68a8ee0dffbb78793"),
            ("kt_gaussian", 8, kt_gaussian_sampling(192, 156, 25, 20, 8, 0.2, 1), 170688, 4.386951631046, "9c967ac97a9713149ed012e063d889bf5bd4efaf90f91429b59145e49ab4ceff"),
        )
        for family, nominal, native, count, actual, digest in cases:
            with self.subTest(family=family, nominal=nominal):
                binary = np.ascontiguousarray(np.asarray(native) > 0, dtype=np.uint8)
                self.assertEqual(int(np.count_nonzero(binary)), count)
                self.assertAlmostEqual(float(binary.size / count), actual, places=12)
                self.assertEqual(hashlib.sha256(binary.tobytes()).hexdigest(), digest)

    def test_uniform_equivalence_domain_across_ncalib_r_and_geometries(self) -> None:
        records = []
        for nx, ny, nt in ((192, 156, 25), (31, 29, 7)):
            for ncalib in (8, 12, 15, 16, 20, 24):
                for nominal in (4, 8, 16, 24):
                    with self.subTest(
                        geometry=(nx, ny, nt), ncalib=ncalib, nominal=nominal
                    ):
                        sdum = uniform_sampling(nx, ny, nt, ncalib, nominal)
                        cardiomm = cardiomm_uniform_sampling(
                            nx, ny, nt, ncalib, nominal
                        )
                        expected_equal = not (ny % 2 == 0 and ncalib % 2 == 1)
                        self.assertEqual(np.array_equal(sdum, cardiomm), expected_equal)
                        sdum_start = (ny - ncalib) // 2
                        cardiomm_start = ny // 2 - ncalib // 2
                        self.assertTrue(
                            np.all(sdum[:, sdum_start : sdum_start + ncalib, :])
                        )
                        self.assertTrue(
                            np.all(
                                cardiomm[
                                    :, cardiomm_start : cardiomm_start + ncalib, :
                                ]
                            )
                        )
                        sdum_binary = np.ascontiguousarray(
                            sdum > 0, dtype=np.uint8
                        )
                        cardiomm_binary = np.ascontiguousarray(
                            cardiomm > 0, dtype=np.uint8
                        )
                        sdum_sha = hashlib.sha256(sdum_binary.tobytes()).hexdigest()
                        cardiomm_sha = hashlib.sha256(
                            cardiomm_binary.tobytes()
                        ).hexdigest()
                        sdum_count = int(np.count_nonzero(sdum_binary))
                        cardiomm_count = int(np.count_nonzero(cardiomm_binary))
                        records.append(
                            {
                                "array_equal": expected_equal,
                                "sdum_sha256": sdum_sha,
                                "cardiomm_sha256": cardiomm_sha,
                                "sdum_count": sdum_count,
                                "cardiomm_count": cardiomm_count,
                                "sdum_actual": sdum_binary.size / sdum_count,
                                "cardiomm_actual": (
                                    cardiomm_binary.size / cardiomm_count
                                ),
                                "acs_geometry": "centered_ky_strip",
                            }
                        )
                        self.assertEqual(sdum_sha == cardiomm_sha, expected_equal)
        self.assertEqual(len(records), 48)
        self.assertEqual(sum(record["array_equal"] for record in records), 44)

    def test_uniform_canonical_and_explicit_boundary_implementations(self) -> None:
        canonical = CanonicalUniformSamplingProtocol(ncalib=16).generate(
            (2, 7, 31, 30, 3), acceleration=24, seed=9
        )
        self.assertEqual(canonical.parameters["implementation"], "canonical")
        self.assertEqual(
            canonical.parameters["canonical_equivalence"]["status"], "verified"
        )
        self.assertEqual(
            canonical.parameters["canonical_equivalence"]["verified_ncalib"],
            [8, 12, 15, 16, 20, 24],
        )
        odd_on_odd = CanonicalUniformSamplingProtocol(ncalib=15).generate(
            (1, 7, 31, 29, 1), acceleration=8
        )
        self.assertEqual(odd_on_odd.requested_acceleration, 8.0)
        with self.assertRaisesRegex(ValueError, "even Ky and odd ncalib"):
            CanonicalUniformSamplingProtocol(ncalib=15).generate(
                (1, 7, 31, 30, 1), acceleration=8
            )

        sdum = default_sampling_registry.create(
            "uniform", implementation_selector="sdum", ncalib=15
        ).generate((1, 7, 31, 30, 1), acceleration=8)
        cardiomm = default_sampling_registry.create(
            "uniform", implementation_selector="cardiomm", ncalib=15
        ).generate((1, 7, 31, 30, 1), acceleration=8)
        self.assertEqual(sdum.parameters["implementation"], "sdum")
        self.assertEqual(cardiomm.parameters["implementation"], "cardiomm")
        self.assertFalse(np.array_equal(sdum.mask, cardiomm.mask))

    def test_uniform_ncalib_changes_actual_not_requested_acceleration(self) -> None:
        narrow = CanonicalUniformSamplingProtocol(ncalib=8).generate(
            (1, 5, 32, 30, 1), acceleration=8
        )
        wide = CanonicalUniformSamplingProtocol(ncalib=24).generate(
            (1, 5, 32, 30, 1), acceleration=8
        )
        self.assertEqual(narrow.requested_acceleration, 8.0)
        self.assertEqual(wide.requested_acceleration, 8.0)
        self.assertNotEqual(narrow.actual_acceleration, wide.actual_acceleration)
        self.assertEqual(
            narrow.actual_acceleration,
            narrow.mask.size / np.count_nonzero(narrow.mask),
        )
        self.assertEqual(
            wide.actual_acceleration,
            wide.mask.size / np.count_nonzero(wide.mask),
        )

    def test_ncalib_must_fit_family_specific_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit uniform Ky"):
            CanonicalUniformSamplingProtocol(ncalib=7).generate(SHAPE, 4)
        with self.assertRaisesRegex(ValueError, "does not fit kt_radial Kx and Ky"):
            NVIDIASamplingProtocol(family="kt_radial", ncalib=9).generate(
                SHAPE, 4
            )

    def test_cardiomm_fixed_geometry_hashes_match_pinned_port(self) -> None:
        cases = (
            (
                "kt_uniform",
                cardiomm_kt_uniform_sampling(192, 156, 25, 20, 4),
                256128,
                "970996efe6d0b79c97f5555ae87e87818d0b23a4d7c213f79387a79f216225af",
            ),
            (
                "kt_gaussian",
                cardiomm_kt_gaussian_sampling(192, 156, 25, 20, 8, 0.2, 1),
                170304,
                "fec063ad8dde37cfcedda61e52411db8f6e1c7db8a08b3da15b07797bc408b3d",
            ),
        )
        for family, native, count, digest in cases:
            with self.subTest(family=family):
                binary = np.ascontiguousarray(native > 0, dtype=np.uint8)
                self.assertEqual(int(np.count_nonzero(binary)), count)
                self.assertEqual(hashlib.sha256(binary.tobytes()).hexdigest(), digest)

    def test_implementation_selector_and_provenance_distinguish_sources(self) -> None:
        for family in ("kt_uniform", "kt_gaussian"):
            with self.subTest(family=family):
                sdum = default_sampling_registry.create(
                    family, implementation_selector="sdum", ncalib=2
                ).generate(SHAPE, acceleration=4, seed=1)
                cardiomm = default_sampling_registry.create(
                    family, implementation_selector="cardiomm", ncalib=2
                ).generate(SHAPE, acceleration=4, seed=1)
                self.assertEqual(sdum.parameters["implementation"], "sdum")
                self.assertEqual(cardiomm.parameters["implementation"], "cardiomm")
                self.assertIn("NV-Raw2Insights", sdum.parameters["generator"])
                self.assertIn("CardioMM", cardiomm.parameters["generator"])
                self.assertNotEqual(
                    sdum.parameters["binary_mask_sha256"],
                    cardiomm.parameters["binary_mask_sha256"],
                )
        with self.assertRaisesRegex(KeyError, "expected one of"):
            default_sampling_registry.create(
                "kt_gaussian", implementation_selector="invented"
            )

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") and importlib.util.find_spec("skimage"),
        "both source rasterizer dependencies are required",
    )
    def test_radial_implementations_preserve_distinct_provenance(self) -> None:
        shape = (2, 5, 32, 30, 1)
        sdum = NVIDIASamplingProtocol(family="kt_radial", ncalib=4).generate(
            shape, acceleration=8, seed=1
        )
        cardiomm = CardioMMSamplingProtocol(family="kt_radial", ncalib=4).generate(
            shape, acceleration=8, seed=1
        )
        self.assertEqual(sdum.parameters["implementation"], "sdum")
        self.assertEqual(cardiomm.parameters["implementation"], "cardiomm")
        # Different upstream implementations may realize identical binary
        # support. Identity is provenance, not a promise of unequal masks.
        self.assertEqual(sdum.mask.shape, cardiomm.mask.shape)
        self.assertEqual(sdum.mask.shape, shape[:4])
        self.assertNotEqual(
            sdum.parameters["upstream_repository"],
            cardiomm.parameters["upstream_repository"],
        )

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy not installed")
    def test_kt_radial_preserves_released_wrapper_parameters(self) -> None:
        result = self._assert_common("kt_radial", seed=7)
        parameters = result.parameters["generator_parameters"]
        self.assertEqual(parameters["nominal_acceleration"], 4.0)
        self.assertEqual(parameters["wrapper_radial_acceleration_scale"], 0.6)
        self.assertEqual(parameters["R"], 2.4)
        self.assertEqual(parameters["angle4next"], 137.5)
        self.assertTrue(parameters["cropcorner"])

    def test_generator_availability_is_separate_from_documented_sdum_exposure(self) -> None:
        self.assertIn("kt_uniform", default_sampling_registry.names())
        self.assertEqual(
            preflight_conditioning(SDUMConditioning("ktUniform16", 16, "Cine"))[
                "status"
            ],
            "SUPPORTED",
        )
        self.assertEqual(
            preflight_conditioning(SDUMConditioning("VISTA", 16, "Cine"))[
                "status"
            ],
            "UNDOCUMENTED",
        )
        self.assertEqual(
            preflight_conditioning(SDUMConditioning("ktRadial16", 16, "Cine"))[
                "status"
            ],
            "SUPPORTED",
        )
        self.assertEqual(
            preflight_conditioning(SDUMConditioning("ktGaussian24", 24, "Cine"))[
                "status"
            ],
            "SUPPORTED",
        )


class RandomCartesianTests(unittest.TestCase):
    def test_origin_native_selection_and_canonical_axis_semantics(self) -> None:
        result = RandomCartesianSampling(
            phase_encoding_axis=3, time_varying=True
        ).generate((2, 4, 7, 12, 1), acceleration=3, seed=11)
        self.assertEqual(result.mask.shape, (2, 4, 7, 12))
        self.assertEqual(result.parameters["implementation"], "cmrbench_native")
        self.assertFalse(result.parameters["origin_audit"]["fastmri_derived"])
        self.assertEqual(result.parameters["sampled_axis"], "Ky")
        self.assertEqual(result.parameters["fully_sampled_orthogonal_axis"], "Kx")
        self.assertFalse(result.parameters["native_mask_array_materialized"])
        np.testing.assert_array_equal(result.mask[0], result.mask[1])
        for frame in range(4):
            selected = result.mask[0, frame, 0]
            np.testing.assert_array_equal(
                result.mask[0, frame], np.broadcast_to(selected, (7, 12))
            )
        self.assertTrue(
            any(
                not np.array_equal(result.mask[0, 0], result.mask[0, frame])
                for frame in range(1, 4)
            )
        )

    def test_static_kx_selection_is_shared_over_time_and_slices(self) -> None:
        result = RandomCartesianSampling(
            phase_encoding_axis=2, time_varying=False
        ).generate((3, 5, 10, 6, 1), acceleration=2, seed=3)
        self.assertEqual(result.parameters["sampled_axis"], "Kx")
        for frame in range(1, 5):
            np.testing.assert_array_equal(result.mask[:, 0], result.mask[:, frame])
        np.testing.assert_array_equal(result.mask[0], result.mask[2])


class VistaPreservationTests(unittest.TestCase):
    def test_registry_dispatches_both_vista_implementations(self) -> None:
        self.assertEqual(
            set(default_sampling_registry.implementations("vista")),
            {"precomputed", "cmrbench"},
        )
        native = default_sampling_registry.create(
            "vista", implementation_selector="cmrbench"
        )
        precomputed = default_sampling_registry.create(
            "vista", implementation_selector="precomputed"
        )
        self.assertEqual(native.implementation, "cmrbench")
        self.assertEqual(precomputed.implementation, "precomputed")

    def test_native_vista_seed_and_pre_acs_frame_count_are_reproducible(self) -> None:
        protocol = default_sampling_registry.create(
            "vista", implementation_selector="cmrbench"
        )
        shape = (2, 4, 5, 12, 3)
        first = protocol.generate(shape, 3, seed=4)
        repeated = protocol.generate(shape, 3, seed=4)
        different = protocol.generate(shape, 3, seed=5)

        np.testing.assert_array_equal(first.mask, repeated.mask)
        self.assertEqual(
            first.parameters["binary_mask_sha256"],
            repeated.parameters["binary_mask_sha256"],
        )
        self.assertNotEqual(
            first.parameters["binary_mask_sha256"],
            different.parameters["binary_mask_sha256"],
        )
        self.assertEqual(first.mask[0, :, 0, :].sum(axis=1).tolist(), [4] * 4)
        np.testing.assert_array_equal(first.mask[0], first.mask[1])
        for kx in range(1, shape[2]):
            np.testing.assert_array_equal(first.mask[:, :, 0], first.mask[:, :, kx])

    def test_historical_precomputed_vista_is_still_preferred_bitwise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            compact = np.zeros((6, 5), dtype=int)
            compact[::2] = 1
            path = Path(directory) / "mask_VISTA_6x5_acc2_1.txt"
            np.savetxt(path, compact, fmt="%d", delimiter=",")

            def forbidden(*_args: object) -> np.ndarray:
                raise AssertionError("historical precomputed path must win")

            result = VISTASamplingProtocol(
                forbidden, mask_root=directory
            ).generate((2, 5, 8, 6, 3), 2, seed=1)
            self.assertEqual(result.parameters["mask_source"], "precomputed")
            expected = np.broadcast_to(
                compact.T.astype(bool)[None, :, None, :], (2, 5, 8, 6)
            )
            np.testing.assert_array_equal(result.mask, expected)


if __name__ == "__main__":
    unittest.main()
