from __future__ import annotations

import importlib.util
import unittest
import tempfile
from pathlib import Path

import numpy as np

from cmrbench.evaluation.reconstruction_metrics import (
    METRIC_PROTOCOL_VERSION,
    compute_cine_frame_metrics,
    compute_frame_metrics,
    compute_paired_frame_win_rates,
    metric_protocol_provenance,
    summarize_frame_metric_distribution,
    summarize_phase_metrics,
    summarize_subject_metrics,
)
from cmrbench.evaluation.reconstruction_qc import (
    amplitude_sensitive_fidelity_error,
    save_anatomy_qc,
    save_fidelity_qc,
    scale_invariant_magnitude_error,
)


def varying_frame() -> np.ndarray:
    return np.linspace(0.1, 1.0, 64, dtype=np.float64).reshape(8, 8)


class ReconstructionMetricsV1Tests(unittest.TestCase):
    def test_nrmse_identical_and_doubled_reference(self) -> None:
        reference = varying_frame()
        identical = compute_frame_metrics(reference, reference)
        doubled = compute_frame_metrics(reference, 2.0 * reference)
        self.assertEqual(identical["nrmse"], 0.0)
        self.assertAlmostEqual(doubled["nrmse"], 1.0, places=14)

    def test_psnr_identical_known_case_and_reference_only_range(self) -> None:
        reference = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))
        identical = compute_frame_metrics(reference, reference)
        shifted = compute_frame_metrics(reference, reference + 1.0)
        doubled = compute_frame_metrics(reference, 2.0 * reference)
        self.assertEqual(identical["psnr"], float("inf"))
        self.assertAlmostEqual(shifted["mse"], 1.0)
        self.assertAlmostEqual(shifted["psnr"], 0.0)
        self.assertAlmostEqual(shifted["reference_data_range"], 1.0)
        self.assertAlmostEqual(doubled["reference_data_range"], 1.0)

    def test_ssim_identical_uses_explicit_supported_geometry(self) -> None:
        reference = varying_frame()
        metrics = compute_frame_metrics(reference, reference)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=14)
        with self.assertRaisesRegex(ValueError, "win_size=7"):
            compute_frame_metrics(reference[:6], reference[:6])

    def test_zero_or_constant_reference_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-reference"):
            compute_frame_metrics(np.zeros((8, 8)), np.zeros((8, 8)))
        with self.assertRaisesRegex(ValueError, "positive reference magnitude data range"):
            compute_frame_metrics(np.ones((8, 8)), 2.0 * np.ones((8, 8)))

    def test_structural_and_fidelity_errors_have_complementary_gain_behavior(self) -> None:
        reference = varying_frame()
        doubled = 2.0 * reference
        structural = scale_invariant_magnitude_error(reference, doubled)
        fidelity = amplitude_sensitive_fidelity_error(reference, doubled)
        np.testing.assert_allclose(structural, 0.0, atol=1e-15)
        self.assertGreater(float(np.max(fidelity)), 0.0)

    def test_qc_error_maps_handle_zero_candidates_and_references_safely(self) -> None:
        reference = varying_frame()
        zero = np.zeros_like(reference)
        structural = scale_invariant_magnitude_error(reference, zero)
        self.assertTrue(np.all(np.isfinite(structural)))
        self.assertGreater(float(np.max(structural)), 0.0)
        np.testing.assert_array_equal(
            scale_invariant_magnitude_error(zero, zero), zero
        )
        with self.assertRaisesRegex(ValueError, "positive reference magnitude data range"):
            amplitude_sensitive_fidelity_error(zero, zero)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is not installed in the local test runtime",
    )
    def test_both_complementary_qc_figures_render(self) -> None:
        reference = np.broadcast_to(varying_frame(), (1, 1, 8, 8)).copy()
        zero_filled = 0.8 * reference
        reconstruction = 1.2 * reference
        canonical_mask = np.ones((1, 1, 8, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anatomy_path = root / "anatomy.png"
            fidelity_path = root / "fidelity.png"
            anatomy = save_anatomy_qc(
                anatomy_path,
                subject="Synthetic",
                acceleration=4,
                reference=reference,
                zero_filled=zero_filled,
                reconstruction=reconstruction,
                frames=[(0, 0)],
                canonical_mask=canonical_mask,
                actual_acceleration=4.25,
                reconstruction_label="SDUM",
            )
            fidelity = save_fidelity_qc(
                fidelity_path,
                subject="Synthetic",
                acceleration=4,
                reference=reference,
                zero_filled=zero_filled,
                reconstruction=reconstruction,
                frames=[(0, 0)],
                canonical_mask=canonical_mask,
                actual_acceleration=4.25,
                reconstruction_label="SDUM",
            )
            self.assertTrue(anatomy_path.is_file())
            self.assertTrue(fidelity_path.is_file())
            self.assertEqual(anatomy["image_scaling"], "independent panel autoscaling")
            self.assertEqual(fidelity["image_scaling"], "per-frame reference-only window")
            self.assertEqual(anatomy["layout"]["rows"], 2)
            self.assertEqual(anatomy["layout"]["columns"], 3)
            self.assertEqual(fidelity["layout"]["rows"], 2)
            self.assertEqual(fidelity["layout"]["columns"], 3)
            self.assertEqual(anatomy["reconstruction_label"], "SDUM")
            self.assertEqual(fidelity["reconstruction_label"], "SDUM")
            self.assertIn("Mask/ZF Error map/Reconstruction Error map", anatomy["layout"]["frame_block"])
            self.assertIn("Mask/ZF Error map/Reconstruction Error map", fidelity["layout"]["frame_block"])
            for metadata in (anatomy, fidelity):
                layout = metadata["layout"]
                self.assertTrue(layout["shared_scientific_grid"])
                self.assertTrue(layout["dedicated_external_colorbar_axis"])
                self.assertEqual(layout["outer_rows"], 1)
                self.assertEqual(layout["outer_columns"], 1)
                self.assertEqual(layout["phase_block_rows"], 2)
                self.assertEqual(layout["phase_block_columns"], 3)
                self.assertEqual(layout["row_hspace"], 0.28)
                self.assertEqual(layout["phase_blocks_arranged"], "side_by_side")
                alignment = layout["alignment"]
                for candidate in ("zero_filled", "reconstruction"):
                    self.assertTrue(alignment[candidate]["centers_aligned"])
                    self.assertTrue(alignment[candidate]["widths_aligned"])
                    self.assertLessEqual(
                        alignment[candidate]["maximum_center_x_difference"],
                        alignment["tolerance"],
                    )
                    self.assertLessEqual(
                        alignment[candidate]["maximum_axes_width_difference"],
                        alignment["tolerance"],
                    )
                spacing = layout["spacing"]
                self.assertEqual(spacing["grid_hspace"], 0.28)
                self.assertTrue(
                    spacing["titles_clear_of_preceding_panels"],
                    msg=str(spacing),
                )
                self.assertGreater(spacing["minimum_title_clearance"], 0.0)
                colorbar = layout["colorbar"]
                self.assertEqual(colorbar["orientation"], "vertical")
                self.assertEqual(colorbar["anchor_panel"], "final SDUM error map")
                self.assertTrue(colorbar["to_right_of_final_reconstruction_error"])
                self.assertTrue(
                    colorbar["to_right_of_final_ram_error"], msg=str(colorbar)
                )
                self.assertTrue(
                    colorbar["height_matches_error_panel"], msg=str(colorbar)
                )
                self.assertLessEqual(
                    abs(colorbar["height_to_error_panel_ratio"] - 1.0),
                    colorbar["height_ratio_tolerance"],
                )
                self.assertLessEqual(colorbar["vertical_center_difference"], 1e-9)
                self.assertGreater(colorbar["horizontal_padding"], 0.0)
                self.assertLess(colorbar["width_to_error_panel_width_ratio"], 0.1)
                self.assertIsNone(colorbar["label"])
                self.assertEqual(metadata["error_colormap"], "turbo")
                self.assertEqual(metadata["error_panel_title"], "Error map")
                self.assertIsNone(metadata["colorbar_label"])
                self.assertNotIn("Numerical errors are not clipped", metadata["title"])
                self.assertTrue(metadata["mask"]["shown"])
                self.assertEqual(metadata["mask"]["actual_acceleration"], 4.25)
                self.assertEqual(
                    metadata["mask"]["actual_acceleration_scope"],
                    "global complete canonical sampling mask",
                )
            self.assertFalse(anatomy["numerical_error_clipping"])
            self.assertFalse(fidelity["numerical_error_clipping"])
            self.assertTrue(anatomy["display_saturation_above_vmax"])
            self.assertTrue(fidelity["display_saturation_above_vmax"])

    def test_frame_rows_and_subject_equal_weight_mean(self) -> None:
        base = varying_frame()
        reference = np.broadcast_to(base, (2, 3, 8, 8)).copy()
        gains = np.asarray([1.05, 1.1, 1.2, 1.3, 1.4, 1.5]).reshape(2, 3, 1, 1)
        reconstruction = reference * gains
        rows = compute_cine_frame_metrics(
            reference,
            reconstruction,
            subject="Synthetic",
            model="example",
            acceleration_requested=4,
            acceleration_actual=4.0,
            sampling_method="synthetic",
            seed=1,
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [(row["slice"], row["phase"], row["flat_frame_index"]) for row in rows],
            [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 3), (1, 1, 4), (1, 2, 5)],
        )
        summary = summarize_subject_metrics(rows)
        self.assertEqual(summary["frame_count"], 6)
        self.assertAlmostEqual(
            summary["mean_nrmse"], np.mean([row["nrmse"] for row in rows])
        )
        self.assertAlmostEqual(
            summary["mean_psnr"], np.mean([row["psnr"] for row in rows])
        )
        self.assertAlmostEqual(
            summary["mean_ssim"], np.mean([row["ssim"] for row in rows])
        )
        self.assertEqual(summary["min_nrmse"], min(row["nrmse"] for row in rows))
        self.assertEqual(summary["max_nrmse"], max(row["nrmse"] for row in rows))

    def test_secondary_frame_distribution_and_phase_summary(self) -> None:
        base = varying_frame()
        reference = np.broadcast_to(base, (2, 2, 8, 8)).copy()
        reconstruction = reference * np.asarray([1.1, 1.2, 1.3, 1.4]).reshape(
            2, 2, 1, 1
        )
        rows = compute_cine_frame_metrics(
            reference,
            reconstruction,
            subject="Synthetic",
            model="ram",
            acceleration_requested=8,
            acceleration_actual=8.0,
            sampling_method="synthetic",
            seed=1,
        )
        distribution = summarize_frame_metric_distribution(rows)
        self.assertEqual(distribution["frame_count"], 4)
        self.assertEqual(
            distribution["metrics"]["nrmse"]["min"],
            min(row["nrmse"] for row in rows),
        )
        self.assertEqual(
            distribution["metrics"]["nrmse"]["max"],
            max(row["nrmse"] for row in rows),
        )
        phase = summarize_phase_metrics(rows)
        self.assertEqual(len(phase), 2)
        self.assertEqual([row["phase"] for row in phase], [0, 1])
        self.assertEqual([row["slice_count"] for row in phase], [2, 2])

    def test_paired_frame_win_rates_match_slice_phase(self) -> None:
        common = {
            "subject": "Synthetic",
            "acceleration_requested": 8.0,
            "acceleration_actual": 8.0,
            "sampling_method": "synthetic",
            "seed": 1,
        }
        values = {
            "zero_filled": [
                (0.5, 20.0, 0.5),
                (0.4, 22.0, 0.6),
                (0.3, 24.0, 0.7),
            ],
            "ram": [
                (0.4, 21.0, 0.6),
                (0.5, 21.0, 0.7),
                (0.2, 23.0, 0.6),
            ],
        }
        rows = []
        for model, metric_values in values.items():
            for phase, (nrmse, psnr, ssim) in enumerate(metric_values):
                rows.append(
                    {
                        **common,
                        "model": model,
                        "slice": 0,
                        "phase": phase,
                        "nrmse": nrmse,
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )
        paired = compute_paired_frame_win_rates(rows)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["metrics"]["nrmse"]["ram_wins"], 2)
        self.assertEqual(paired[0]["metrics"]["psnr"]["ram_wins"], 1)
        self.assertEqual(paired[0]["metrics"]["ssim"]["ram_wins"], 2)
        self.assertAlmostEqual(
            paired[0]["metrics"]["nrmse"]["percentage"], 200.0 / 3.0
        )

    def test_protocol_provenance_freezes_primary_semantics_and_versions(self) -> None:
        protocol = metric_protocol_provenance()
        self.assertEqual(protocol["protocol_version"], METRIC_PROTOCOL_VERSION)
        self.assertEqual(protocol["domain"], "raw_magnitude")
        self.assertFalse(protocol["candidate_specific_normalization"])
        self.assertFalse(protocol["clipping"])
        self.assertFalse(protocol["p99_5_primary_metric_usage"])
        self.assertEqual(protocol["ssim_win_size"], 7)
        self.assertFalse(protocol["ssim_gaussian_weights"])
        self.assertNotEqual(protocol["package_versions"]["scikit-image"], "NOT INSTALLED")


if __name__ == "__main__":
    unittest.main()
