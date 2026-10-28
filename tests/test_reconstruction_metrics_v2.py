from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from cmrbench.evaluation.reconstruction_metrics import (
    METRIC_PROTOCOL_V1,
    METRIC_PROTOCOL_V2,
    compute_cine_frame_metrics,
    compute_frame_metrics,
    compute_normalized_frame_metrics_v2,
    metric_protocol_provenance,
    normalize_subject_magnitude_p995,
    summarize_subject_metrics,
)


HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
ROOT = Path(__file__).resolve().parents[1]


def cine(values: np.ndarray, *, slices: int = 1, phases: int = 2) -> np.ndarray:
    frame = np.asarray(values, dtype=np.float64).reshape(8, 8)
    return np.broadcast_to(frame, (slices, phases, 8, 8)).copy()


def metric_rows(reference: np.ndarray, candidate: np.ndarray, *, model: str = "ram"):
    return compute_cine_frame_metrics(
        reference,
        candidate,
        subject="Synthetic",
        model=model,
        acceleration_requested=4,
        acceleration_actual=4.0,
        sampling_method="synthetic",
        seed=1,
        protocol="v2_subject_p995",
    )


class SubjectP995NormalizationTests(unittest.TestCase):
    def test_one_scale_is_computed_over_complete_subject_without_clipping(self) -> None:
        subject = np.arange(1, 129, dtype=np.float64).reshape(1, 2, 8, 8)
        normalized, metadata = normalize_subject_magnitude_p995(
            subject, source="synthetic"
        )
        self.assertEqual(metadata["p99_5"], float(np.percentile(subject, 99.5)))
        self.assertNotEqual(metadata["p99_5"], float(np.percentile(subject[0, 0], 99.5)))
        self.assertNotEqual(metadata["p99_5"], float(np.percentile(subject[0, 1], 99.5)))
        self.assertGreater(float(np.max(normalized)), 1.0)
        self.assertFalse(metadata["scientific_clipping"])

    def test_extreme_maximum_does_not_become_the_p995_scale(self) -> None:
        subject = np.linspace(1.0, 2.0, 200).reshape(1, 2, 10, 10)
        subject[0, 0, 0, 0] = 1000.0
        _, metadata = normalize_subject_magnitude_p995(subject, source="outlier")
        self.assertLess(metadata["p99_5"], float(np.max(subject)))
        self.assertGreater(metadata["normalized_max"], 1.0)

    def test_invalid_zero_and_nonfinite_subjects_fail_without_epsilon(self) -> None:
        with self.assertRaisesRegex(ValueError, "p99.5 scale must be positive and finite"):
            normalize_subject_magnitude_p995(
                np.zeros((1, 1, 8, 8)), source="zero"
            )
        nonfinite = np.ones((1, 1, 8, 8))
        nonfinite[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "contains NaN or Inf"):
            normalize_subject_magnitude_p995(nonfinite, source="nonfinite")

    def test_protocol_metadata_freezes_v1_and_v2_roles(self) -> None:
        v1 = metric_protocol_provenance("v1_raw")
        v2 = metric_protocol_provenance("v2_subject_p995")
        self.assertEqual(v1["protocol_version"], METRIC_PROTOCOL_V1)
        self.assertEqual(v2["protocol_version"], METRIC_PROTOCOL_V2)
        self.assertEqual(v2["psnr_data_range"], 1.0)
        self.assertEqual(v2["ssim_data_range"], 1.0)
        self.assertTrue(v2["normalization_independent_per_source"])
        self.assertFalse(v2["per_frame_normalization"])
        self.assertFalse(v2["clipping"])
        self.assertFalse(v2["gain_fitting"])




class ReconstructionMetricsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._skimage_patcher = None
        if not HAS_SKIMAGE:
            skimage_module = types.ModuleType("skimage")
            skimage_module.__path__ = []  # type: ignore[attr-defined]
            metrics_module = types.ModuleType("skimage.metrics")

            def structural_similarity(reference, candidate, **kwargs):
                del kwargs
                return 1.0 if np.allclose(reference, candidate) else 0.5

            metrics_module.structural_similarity = structural_similarity
            skimage_module.metrics = metrics_module
            self._skimage_patcher = patch.dict(
                sys.modules,
                {"skimage": skimage_module, "skimage.metrics": metrics_module},
            )
            self._skimage_patcher.start()

    def tearDown(self) -> None:
        if self._skimage_patcher is not None:
            self._skimage_patcher.stop()

    def test_metrics_v1_numerical_semantics_are_unchanged(self) -> None:
        reference = np.linspace(0.1, 1.0, 64).reshape(8, 8)
        result = compute_frame_metrics(reference, 2.0 * reference)
        self.assertAlmostEqual(result["nrmse"], 1.0, places=14)
        self.assertAlmostEqual(result["reference_data_range"], 0.9, places=14)
        self.assertEqual(
            metric_protocol_provenance()["protocol_version"], METRIC_PROTOCOL_V1
        )


    def test_subject_level_not_frame_level_normalization(self) -> None:
        base = np.linspace(0.1, 1.0, 64).reshape(8, 8)
        reference = cine(base)
        reference[0, 1] *= 3.0
        candidate = reference.copy()
        candidate[0, 0] *= 2.0
        candidate[0, 1] *= 4.0
        rows = metric_rows(reference, candidate)
        self.assertEqual(len({row["reference_p99_5"] for row in rows}), 1)
        self.assertEqual(len({row["candidate_p99_5"] for row in rows}), 1)
        self.assertGreater(rows[0]["nrmse"], 0.0)
        self.assertNotAlmostEqual(rows[0]["nrmse"], rows[1]["nrmse"])

    def test_independent_global_gain_normalization(self) -> None:
        reference = cine(np.linspace(0.1, 1.0, 64))
        rows = metric_rows(reference, 5.0 * reference)
        self.assertTrue(all(abs(row["nrmse"]) < 1e-14 for row in rows))
        self.assertTrue(all(abs(row["ssim"] - 1.0) < 1e-14 for row in rows))
        self.assertAlmostEqual(rows[0]["candidate_to_reference_p99_5_ratio"], 5.0)

    def test_real_and_complex_candidates_both_use_magnitude(self) -> None:
        base = cine(np.linspace(0.1, 1.0, 64))
        real_rows = metric_rows(base, 2.0 * base, model="sdum_small")
        complex_reference = base * np.exp(1j * 0.3)
        complex_candidate = 2.0 * base * np.exp(-1j * 0.7)
        complex_rows = metric_rows(complex_reference, complex_candidate, model="ram")
        self.assertLess(real_rows[0]["nrmse"], 1e-14)
        self.assertLess(complex_rows[0]["nrmse"], 1e-14)

    def test_fixed_data_range_one_even_when_values_exceed_one(self) -> None:
        reference = np.linspace(0.0, 1.5, 64).reshape(8, 8)
        candidate = reference.copy()
        candidate[0, 0] = 2.0
        with patch("skimage.metrics.structural_similarity", return_value=0.75) as mocked:
            result = compute_normalized_frame_metrics_v2(reference, candidate)
        self.assertEqual(result["reference_data_range"], 1.0)
        self.assertEqual(mocked.call_args.kwargs["data_range"], 1.0)
        self.assertEqual(mocked.call_args.kwargs["win_size"], 7)
        self.assertFalse(mocked.call_args.kwargs["gaussian_weights"])
        self.assertGreater(float(np.max(mocked.call_args.args[1])), 1.0)

    def test_zero_filled_receives_its_own_subject_scale(self) -> None:
        reference = cine(np.linspace(0.1, 1.0, 64))
        rows = metric_rows(reference, 3.0 * reference, model="zero_filled")
        self.assertAlmostEqual(rows[0]["candidate_to_reference_p99_5_ratio"], 3.0)
        self.assertLess(rows[0]["nrmse"], 1e-14)

    def test_equal_mean_aggregation_and_scale_metadata(self) -> None:
        reference = cine(np.linspace(0.1, 1.0, 64), slices=2, phases=3)
        candidate = reference.copy()
        gains = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8, 2.0]).reshape(2, 3, 1, 1)
        candidate *= gains
        rows = metric_rows(reference, candidate)
        summary = summarize_subject_metrics(rows)
        self.assertEqual(summary["frame_count"], 6)
        self.assertAlmostEqual(
            summary["mean_nrmse"], np.mean([row["nrmse"] for row in rows])
        )
        self.assertIn("reference_p99_5", summary)
        self.assertIn("candidate_p99_5", summary)
        self.assertIn("candidate_to_reference_p99_5_ratio", summary)
        self.assertEqual(summary["metric_protocol_version"], METRIC_PROTOCOL_V2)
        self.assertFalse(summary["scientific_clipping"])


if __name__ == "__main__":
    unittest.main()
