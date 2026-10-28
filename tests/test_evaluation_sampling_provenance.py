from __future__ import annotations

import copy
from pathlib import Path
import unittest
from xml.etree import ElementTree

import numpy as np

from cmrbench.evaluate import _verify_sampling_parity
from cmrbench.sampling.result import SamplingResult, binary_mask_provenance


ROOT = Path(__file__).resolve().parents[1]


def make_sampling(
    *,
    scikit_image_version: str = "0.25.2",
    other_version: str = "1.0",
) -> SamplingResult:
    mask = np.zeros((1, 2, 4, 6), dtype=bool)
    mask[..., ::2] = True
    return SamplingResult(
        mask=mask,
        requested_acceleration=16.0,
        actual_acceleration=2.0,
        method="kt_radial",
        seed=1,
        parameters={
            "family": "kt_radial",
            "implementation": "cardiomm",
            "generator": "CardioMM::ktRadial",
            "ncalib": 2,
            "acs_geometry": "centered_square",
            "generator_parameters": {
                "R": 9.6,
                "angle4next": 137.5,
                "cropcorner": True,
                "scikit_image_version": scikit_image_version,
                "other_version": other_version,
            },
            **binary_mask_provenance(mask),
        },
    )


class EvaluationSamplingProvenanceTests(unittest.TestCase):
    def test_identical_sampling_parameters_pass(self) -> None:
        sampling = make_sampling()
        recorded = sampling.as_metadata()
        verified = _verify_sampling_parity("Example0132", sampling, recorded)
        self.assertEqual(
            verified["binary_mask_sha256"],
            recorded["sampling_parameters"]["binary_mask_sha256"],
        )

    def test_only_scikit_image_version_difference_with_same_sha_passes(self) -> None:
        sampling = make_sampling(scikit_image_version="0.25.2")
        recreated_before = copy.deepcopy(sampling.as_metadata())
        recorded = copy.deepcopy(recreated_before)
        recorded["sampling_parameters"]["generator_parameters"][
            "scikit_image_version"
        ] = "0.23.2"
        recorded_before = copy.deepcopy(recorded)

        with self.assertWarnsRegex(
            RuntimeWarning,
            r"stored='0\.23\.2'.*recreated='0\.25\.2'.*SHA256 is identical",
        ):
            _verify_sampling_parity("Example0132", sampling, recorded)

        self.assertEqual(recorded, recorded_before)
        self.assertEqual(sampling.as_metadata(), recreated_before)
        self.assertEqual(
            recorded["sampling_parameters"]["generator_parameters"][
                "scikit_image_version"
            ],
            "0.23.2",
        )
        self.assertEqual(
            sampling.parameters["generator_parameters"]["scikit_image_version"],
            "0.25.2",
        )

    def test_scikit_image_version_difference_with_different_sha_fails(self) -> None:
        sampling = make_sampling(scikit_image_version="0.25.2")
        recorded = copy.deepcopy(sampling.as_metadata())
        recorded["sampling_parameters"]["generator_parameters"][
            "scikit_image_version"
        ] = "0.23.2"
        recorded["sampling_parameters"]["binary_mask_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "binary mask SHA256 mismatch"):
            _verify_sampling_parity("Example0132", sampling, recorded)

    def test_scientific_and_physical_sampling_mismatches_still_fail(self) -> None:
        mutations = {
            "family": lambda parameters: parameters.__setitem__(
                "family", "kt_gaussian"
            ),
            "implementation": lambda parameters: parameters.__setitem__(
                "implementation", "sdum"
            ),
            "ncalib": lambda parameters: parameters.__setitem__("ncalib", 4),
            "generator parameter": lambda parameters: parameters[
                "generator_parameters"
            ].__setitem__("angle4next", 111.0),
            "canonical shape": lambda parameters: parameters.__setitem__(
                "canonical_shape", [1, 2, 4, 7]
            ),
            "sampled count": lambda parameters: parameters.__setitem__(
                "sampled_points", parameters["sampled_points"] + 1
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                sampling = make_sampling()
                recorded = copy.deepcopy(sampling.as_metadata())
                mutate(recorded["sampling_parameters"])
                with self.assertRaisesRegex(
                    ValueError, "recreated sampling sampling_parameters mismatch"
                ):
                    _verify_sampling_parity("Example0132", sampling, recorded)

    def test_other_version_fields_are_not_ignored(self) -> None:
        sampling = make_sampling(other_version="2.0")
        recorded = copy.deepcopy(sampling.as_metadata())
        recorded["sampling_parameters"]["generator_parameters"][
            "other_version"
        ] = "1.0"
        with self.assertRaisesRegex(
            ValueError, "recreated sampling sampling_parameters mismatch"
        ):
            _verify_sampling_parity("Example0132", sampling, recorded)

        source = (ROOT / "cmrbench" / "evaluate.py").read_text(encoding="utf-8")
        self.assertNotIn('endswith("_version")', source)
        self.assertNotIn("endswith('_version')", source)



if __name__ == "__main__":
    unittest.main()
