from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import warnings

import numpy as np

from cmrbench.data.base import CineMRIDataset
from cmrbench.data.inhouse_cine import build_inhouse_cine_sample
from cmrbench.data.sample import CineMRISample
from cmrbench.evaluate import (
    METRICS_COLUMNS,
    QC_PROTOCOL,
    _verify_sampling_parity,
    evaluate_reconstruction,
    mid_slice_fixed_phase_qc_frames,
    prepare_zero_filled_input,
)
from cmrbench.evaluation.reconstruction_gif import (
    GIF_PROTOCOL,
    prepare_mid_slice_gif_frames,
    save_mid_slice_reconstruction_gif,
)
from cmrbench.evaluation.reconstruction_metrics import (
    METRIC_PROTOCOL_V2,
    compute_cine_frame_metrics,
    summarize_subject_metrics,
)
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import (
    attach_dataset_sensitivity_maps,
    prepare_reconstruction_input,
)
from cmrbench.models.adapters.sdum import SDUMAdapter
from cmrbench.models.adapters.zero_filled import ZeroFilledAdapter
from cmrbench.reconstruct import create_sampling_protocol
from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.result import SamplingResult, binary_mask_provenance


ROOT = Path(__file__).resolve().parents[1]


class SyntheticDataset(CineMRIDataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SyntheticSampling(SamplingProtocol):
    def generate(self, shape, acceleration, seed=None, **kwargs):
        del kwargs
        mask = np.zeros(shape[:4], dtype=bool)
        mask[..., ::2] = True
        return SamplingResult(
            mask=mask,
            requested_acceleration=acceleration,
            actual_acceleration=2.0,
            method="uniform",
            seed=seed,
            parameters={
                "implementation": "canonical",
                "ncalib": 2,
                "generator_parameters": {"R": 2, "ncalib": 2},
                **binary_mask_provenance(mask),
            },
        )


def make_sample(subject: str, offset: float = 0.0):
    values = np.linspace(0.1 + offset, 1.0 + offset, 128).reshape(1, 2, 8, 8, 1)
    maps = np.ones((1, 1, 8, 8, 1), dtype=np.complex64)
    return build_inhouse_cine_sample(
        subject,
        values.astype(np.complex64),
        maps,
    )


def make_config(output_root: Path, *, subjects: int = 1) -> ExperimentConfig:
    parameters = {"subject": "Example0001"} if subjects == 1 else {}
    return ExperimentConfig.from_mapping(
        {
            "experiment": {"name": "synthetic-generic-evaluation", "seed": 9},
            "dataset": {
                "name": "inhouse_cine",
                "manifest": "synthetic.csv",
                "split": "development",
                "parameters": parameters,
            },
            "sampling": {
                "mode": "retrospective",
                "method": "uniform",
                "implementation": "canonical",
                "acceleration": 2,
                "seed": 7,
                "parameters": {"ncalib": 2},
            },
            "model": {"name": "synthetic", "implementation": "test_impl"},
            "output": {
                "root": str(output_root),
                "save_reconstruction": True,
                "save_metadata": True,
                "save_qc": False,
            },
            "evaluation": {
                "qc": {"phase_indices": [0, 1]},
            },
        }
    )


def qc_stub(path: Path, **kwargs):
    path.write_bytes(b"frozen-qc-stub")
    return {
        "path": str(path.resolve()),
        "mode": "anatomy_qc" if "anatomy" in path.name else "fidelity_qc",
        "frames": [
            {"slice": slice_index, "phase": phase_index}
            for slice_index, phase_index in kwargs["frames"]
        ],
        "error_display_range": [0.0, kwargs["error_vmax"]],
        "reconstruction_label": kwargs["reconstruction_label"],
    }


def gif_stub(path: Path, **kwargs):
    reconstruction = kwargs["reconstruction"]
    frames, metadata = prepare_mid_slice_gif_frames(reconstruction)
    path.write_bytes(b"candidate-gif-stub")
    return {
        **metadata,
        "path": str(path.resolve()),
        "encoded_frame_count": int(frames.shape[0]),
    }


def write_generic_output(root: Path, config: ExperimentConfig, samples) -> None:
    root.mkdir(parents=True)
    subjects_root = root / "subjects"
    subjects_root.mkdir()
    protocol = SyntheticSampling()
    rows = []
    actuals = []
    for sample in samples:
        prepared = prepare_reconstruction_input(
            sample,
            mode="retrospective",
            protocol=protocol,
            acceleration=2,
            seed=7,
        )
        sampling = prepared.sampling.as_metadata()
        candidate = np.abs(sample.reference) * np.linspace(0.9, 1.1, 64).reshape(
            1, 1, 8, 8
        )
        np.save(
            subjects_root / f"{sample.subject_id}_reconstruction.npy",
            candidate,
            allow_pickle=False,
        )
        subject_metadata = {
            "subject_id": sample.subject_id,
            "model_name": "synthetic",
            "model": {"model_name": "synthetic", "model_version": "1"},
            "metadata": {
                "implementation": "test_impl",
                "cmrbench_sampling": sampling,
                "model_sampling": {
                    "model_conditioning_mode": "explicit",
                    "model_conditioning_family": "random",
                    "physical_sampling_validity": "valid",
                },
            },
            "sampling": sampling,
        }
        (subjects_root / f"{sample.subject_id}_metadata.json").write_text(
            json.dumps(subject_metadata), encoding="utf-8"
        )
        rows.append(
            {
                "subject_id": sample.subject_id,
                "model_name": "synthetic",
                "implementation": "test_impl",
                "variant": "",
                "runtime_seconds": "0.1",
                **sampling,
            }
        )
        actuals.append(2.0)
    (root / "config.yaml").write_text("effective config fixture\n", encoding="utf-8")
    experiment = {
        "experiment_name": config.experiment.name,
        "config": config.as_dict(),
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "sampling_mode": config.sampling.mode,
        "sampling_method": config.sampling.method,
        "sampling_implementation": config.sampling.implementation,
        "requested_acceleration": config.sampling.acceleration,
        "sampling_seed": config.sampling.seed,
        "actual_accelerations": actuals,
        "resolved_sampling_implementations": ["canonical"],
        "model": {"model_name": "synthetic", "model_version": "1"},
    }
    (root / "experiment_metadata.json").write_text(
        json.dumps(experiment), encoding="utf-8"
    )
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "subject_id",
            "model_name",
            "implementation",
            "variant",
            "runtime_seconds",
            "sampling_method",
            "sampling_implementation",
            "requested_acceleration",
            "actual_acceleration",
            "sampling_seed",
            "sampling_parameters",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["sampling_parameters"] = json.dumps(
                serialized["sampling_parameters"], sort_keys=True
            )
            writer.writerow(serialized)


class GenericEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skimage_patcher = None
        if importlib.util.find_spec("skimage") is None:
            skimage = types.ModuleType("skimage")
            skimage.__path__ = []
            metrics = types.ModuleType("skimage.metrics")
            metrics.structural_similarity = lambda reference, candidate, **kwargs: float(
                1.0 - np.mean(np.abs(reference - candidate))
            )
            skimage.metrics = metrics
            cls.skimage_patcher = mock.patch.dict(
                sys.modules, {"skimage": skimage, "skimage.metrics": metrics}
            )
            cls.skimage_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if cls.skimage_patcher is not None:
            cls.skimage_patcher.stop()

    def run_fixture(self, root: Path, config: ExperimentConfig, samples, **kwargs):
        with mock.patch(
            "cmrbench.evaluate.ExperimentConfig.load", return_value=config
        ):
            return evaluate_reconstruction(
                root,
                dataset_factory=lambda _config: SyntheticDataset(samples),
                sampling_factory=lambda _config: SyntheticSampling(),
                anatomy_qc_writer=qc_stub,
                fidelity_qc_writer=qc_stub,
                **kwargs,
            )

    def test_static_whole_subject_evaluates_all_slices_and_qc_middle_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            raw = make_config(root).as_dict()
            raw["evaluation"]["qc"]["phase_indices"] = [0]
            config = ExperimentConfig.from_mapping(raw)
            values = np.linspace(0.1, 1.0, 3 * 8 * 8).reshape(3, 1, 8, 8, 1)
            sample = build_inhouse_cine_sample(
                "Example0001", values.astype(np.complex64),
                np.ones((3, 1, 8, 8, 1), dtype=np.complex64),
            )
            write_generic_output(root, config, [sample])
            rows = self.run_fixture(root, config, [sample])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["frame_count"], 3)
            metadata = json.loads(
                (root / "evaluation" / "qc" / "Example0001_qc_metadata.json").read_text()
            )
            self.assertEqual(metadata["frames"], [{"slice": 1, "phase": 0}])
            self.assertTrue((root / "evaluation" / "qc" / "Example0001_anatomy_qc.png").is_file())
            self.assertTrue((root / "evaluation" / "qc" / "Example0001_fidelity_qc.png").is_file())

    def test_valid_output_reuses_frozen_metrics_zero_fill_and_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            reference_before = sample.reference.copy()
            candidate_path = root / "subjects" / "Example0001_reconstruction.npy"
            candidate_before = np.load(candidate_path, allow_pickle=False).copy()
            with mock.patch(
                "cmrbench.evaluate.compute_cine_frame_metrics",
                wraps=compute_cine_frame_metrics,
            ) as metric_function:
                rows = self.run_fixture(root, config, [sample])

            np.testing.assert_array_equal(sample.reference, reference_before)
            np.testing.assert_array_equal(
                np.load(candidate_path, allow_pickle=False), candidate_before
            )

            self.assertEqual(metric_function.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["protocol"] == METRIC_PROTOCOL_V2
                    for call in metric_function.call_args_list
                )
            )
            prepared = prepare_reconstruction_input(
                sample,
                mode="retrospective",
                protocol=SyntheticSampling(),
                acceleration=2,
                seed=7,
            )
            zero_filled = ZeroFilledAdapter().reconstruct(prepared).reconstruction
            direct_zf = summarize_subject_metrics(
                compute_cine_frame_metrics(
                    sample.reference,
                    zero_filled,
                    subject="Example0001",
                    model="zero_filled",
                    acceleration_requested=2,
                    acceleration_actual=2.0,
                    sampling_method="uniform",
                    seed=7,
                    protocol=METRIC_PROTOCOL_V2,
                )
            )
            candidate = np.load(
                root / "subjects" / "Example0001_reconstruction.npy",
                allow_pickle=False,
            )
            direct_candidate = summarize_subject_metrics(
                compute_cine_frame_metrics(
                    sample.reference,
                    candidate,
                    subject="Example0001",
                    model="synthetic",
                    acceleration_requested=2,
                    acceleration_actual=2.0,
                    sampling_method="uniform",
                    seed=7,
                    protocol=METRIC_PROTOCOL_V2,
                )
            )
            self.assertEqual(rows[0]["zf_nrmse"], direct_zf["mean_nrmse"])
            self.assertEqual(
                rows[0]["reconstruction_nrmse"], direct_candidate["mean_nrmse"]
            )
            self.assertEqual(
                rows[0]["reconstruction_psnr"], direct_candidate["mean_psnr"]
            )
            self.assertEqual(
                rows[0]["reconstruction_ssim"], direct_candidate["mean_ssim"]
            )
            self.assertEqual(rows[0]["metric_protocol"], METRIC_PROTOCOL_V2)
            self.assertEqual(rows[0]["ncalib"], 2)
            self.assertEqual(rows[0]["sampling_implementation"], "canonical")
            self.assertEqual(rows[0]["implementation"], "test_impl")
            self.assertIsNone(rows[0]["variant"])
            self.assertEqual(rows[0]["frame_count"], 2)

            with (root / "evaluation" / "metrics_v2.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
                self.assertEqual(tuple(csv_rows[0]), METRICS_COLUMNS)
            metrics_json = json.loads(
                (root / "evaluation" / "metrics_v2.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("cohort", metrics_json)
            self.assertNotIn("v1", json.dumps(metrics_json).lower())
            metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["evaluation_protocol_identifier"], METRIC_PROTOCOL_V2)
            self.assertEqual(metadata["qc_protocol_identifier"], QC_PROTOCOL)
            self.assertEqual(metadata["number_of_subjects_evaluated"], 1)
            self.assertTrue(
                metadata["subjects"][0]["zero_filled_sensitivity_maps"]["reused_without_reestimation"]
            )
            self.assertEqual(
                metadata["model_conditioning_provenance"][0][
                    "model_conditioning_family"
                ],
                "random",
            )
            self.assertEqual(
                metadata["subjects"][0]["qc_frames"],
                [{"slice": 0, "phase": 0}, {"slice": 0, "phase": 1}],
            )
            self.assertTrue(
                (root / "evaluation" / "qc" / "Example0001_anatomy_qc.png").is_file()
            )
            self.assertTrue(
                (root / "evaluation" / "qc" / "Example0001_fidelity_qc.png").is_file()
            )

    def test_results_csv_is_authoritative_and_multiple_subjects_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root, subjects=2)
            samples = [make_sample("Example0001"), make_sample("Example0002", 0.1)]
            write_generic_output(root, config, samples)
            np.save(
                root / "subjects" / "Stale_reconstruction.npy",
                np.zeros((1, 2, 8, 8)),
            )
            rows = self.run_fixture(root, config, samples)
            self.assertEqual([row["subject"] for row in rows], ["Example0001", "Example0002"])
            self.assertFalse(
                (root / "evaluation" / "qc" / "Stale_anatomy_qc.png").exists()
            )

    def test_candidate_shape_and_finite_values_fail_with_failed_status(self):
        for invalid in (
            np.ones((1, 1, 8, 8)),
            np.full((1, 2, 8, 8), np.nan),
        ):
            with self.subTest(shape=invalid.shape), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "reconstruction"
                config = make_config(root)
                sample = make_sample("Example0001")
                write_generic_output(root, config, [sample])
                np.save(root / "subjects" / "Example0001_reconstruction.npy", invalid)
                with self.assertRaisesRegex(ValueError, "candidate"):
                    self.run_fixture(root, config, [sample])
                metadata = json.loads(
                    (root / "evaluation" / "evaluation_metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metadata["status"], "failed")
                self.assertFalse((root / "evaluation" / "metrics_v2.csv").exists())

    def test_sampling_seed_parameters_and_mask_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            results = (root / "results.csv").read_text(encoding="utf-8")
            (root / "results.csv").write_text(
                results.replace(",7,", ",8,"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sampling_seed mismatch"):
                self.run_fixture(root, config, [sample])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            results_path = root / "results.csv"
            with results_path.open(encoding="utf-8", newline="") as handle:
                result_rows = list(csv.DictReader(handle))
                fieldnames = list(result_rows[0])
            parameters = json.loads(result_rows[0]["sampling_parameters"])
            parameters["binary_mask_sha256"] = "bad"
            result_rows[0]["sampling_parameters"] = json.dumps(parameters, sort_keys=True)
            with results_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result_rows)
            metadata_path = root / "subjects" / "Example0001_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sampling"]["sampling_parameters"]["binary_mask_sha256"] = "bad"
            metadata["metadata"]["cmrbench_sampling"]["sampling_parameters"][
                "binary_mask_sha256"
            ] = "bad"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binary mask SHA256 mismatch"):
                self.run_fixture(root, config, [sample])

    def test_incomplete_or_subject_inconsistent_output_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            root.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "effective config"):
                evaluate_reconstruction(root)
            metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                self.run_fixture(root, config, [sample, make_sample("Example0002")])

    def test_explicit_qc_frames_and_rerun_overwrite_standard_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            first = self.run_fixture(root, config, [sample], qc_frames=[(0, 1)])
            candidate_path = root / "subjects" / "Example0001_reconstruction.npy"
            candidate = np.load(candidate_path)
            candidate[..., 0, 0] *= 0.25
            np.save(candidate_path, candidate)
            second = self.run_fixture(root, config, [sample], qc_frames=[(0, 1)])
            self.assertNotEqual(
                first[0]["reconstruction_nrmse"],
                second[0]["reconstruction_nrmse"],
            )
            qc_metadata = json.loads(
                (
                    root
                    / "evaluation"
                    / "qc"
                    / "Example0001_qc_metadata.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(qc_metadata["frames"], [{"slice": 0, "phase": 1}])
            self.assertEqual(
                qc_metadata["selection_protocol_identifier"],
                "explicit_cli_slice_phase_v1",
            )
            evaluation_metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                evaluation_metadata["qc_selection_protocol_identifier"],
                "explicit_cli_slice_phase_v1",
            )

    def test_status_is_running_during_work_and_failed_after_qc_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            observed = []

            def failing_qc(path, **kwargs):
                del path, kwargs
                observed.append(
                    json.loads(
                        (
                            root / "evaluation" / "evaluation_metadata.json"
                        ).read_text(encoding="utf-8")
                    )["status"]
                )
                raise RuntimeError("synthetic QC failure")

            with mock.patch(
                "cmrbench.evaluate.ExperimentConfig.load", return_value=config
            ), self.assertRaisesRegex(RuntimeError, "synthetic QC failure"):
                evaluate_reconstruction(
                    root,
                    dataset_factory=lambda _config: SyntheticDataset([sample]),
                    sampling_factory=lambda _config: SyntheticSampling(),
                    anatomy_qc_writer=failing_qc,
                    fidelity_qc_writer=qc_stub,
                )
            self.assertEqual(observed, ["running"])
            metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "failed")

    def test_import_is_foundation_model_runtime_independent(self):
        executable = sys.executable
        code = (
            "import sys, cmrbench.evaluate; "
            "names=('cmrbench.models.adapters.ram',"
            "'cmrbench.models.adapters.sdum',"
            "'cmrbench.models.adapters.cardiomm',"
            "'cmrbench.models.adapters.zero_filled'); "
            "print(','.join(name for name in names if name in sys.modules))"
        )
        completed = subprocess.run(
            [executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")

    def test_evaluation_never_uses_model_registry_or_model_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            with mock.patch(
                "cmrbench.models.registry.ModelRegistry.create",
                side_effect=AssertionError("foundation model creation is forbidden"),
            ) as create_model:
                rows = self.run_fixture(root, config, [sample])
            create_model.assert_not_called()
            self.assertEqual(len(rows), 1)


    def test_default_qc_selector_is_actual_mid_slice_and_fixed_zero_based_phases(self):
        self.assertEqual(
            mid_slice_fixed_phase_qc_frames((12, 25, 192, 156)),
            ((6, 6), (6, 22)),
        )
        self.assertNotEqual(
            mid_slice_fixed_phase_qc_frames((12, 25, 192, 156)),
            ((0, 0), (6, 12), (11, 24)),
        )

    def test_default_qc_fails_for_short_cine_and_explicit_override_is_supported(self):
        with self.assertRaisesRegex(ValueError, "zero-based QC phase 22.*T=22"):
            mid_slice_fixed_phase_qc_frames((5, 22, 32, 32))
        self.assertEqual(
            mid_slice_fixed_phase_qc_frames((5, 2, 32, 32), (0, 1)),
            ((2, 0), (2, 1)),
        )

    def test_candidate_gif_uses_mid_slice_all_phases_and_one_fixed_window(self):
        candidate = np.zeros((3, 4, 2, 2), dtype=np.complex64)
        candidate[1, 0] = 1.0
        candidate[1, 1] = 2.0j
        candidate[1, 2] = 3.0
        candidate[1, 3] = 4.0
        before = candidate.copy()
        frames, metadata = prepare_mid_slice_gif_frames(candidate)
        np.testing.assert_array_equal(candidate, before)
        self.assertEqual(frames.shape, (4, 2, 2))
        self.assertEqual(metadata["protocol_identifier"], GIF_PROTOCOL)
        self.assertEqual(metadata["slice_index"], 1)
        self.assertEqual(metadata["phase_indices"], [0, 1, 2, 3])
        self.assertEqual(metadata["display_window"], [0.0, 4.0])
        self.assertFalse(metadata["per_frame_autoscaling"])
        self.assertEqual(int(frames[0, 0, 0]), 64)
        self.assertEqual(int(frames[-1, 0, 0]), 255)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.gif"
            written = save_mid_slice_reconstruction_gif(
                path, reconstruction=candidate
            )
            self.assertEqual(written["frame_count"], 4)
            from PIL import Image

            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 4)

    def test_optional_gif_is_candidate_only_and_recorded_in_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            self.run_fixture(
                root,
                config,
                [sample],
                gif_enabled=True,
                gif_writer=gif_stub,
            )
            qc_root = root / "evaluation" / "qc"
            self.assertTrue(
                (qc_root / "Example0001_mid_slice_reconstruction.gif").is_file()
            )
            self.assertFalse((qc_root / "Example0001_reference.gif").exists())
            self.assertFalse((qc_root / "Example0001_zero_filled.gif").exists())
            metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["gif_protocol"]["identifier"], GIF_PROTOCOL)
            self.assertTrue(metadata["gif_protocol"]["enabled"])
            self.assertEqual(metadata["subjects"][0]["gif"]["frame_count"], 2)

    def test_qc_receives_verified_canonical_mask_and_global_actual_r(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            config = make_config(root)
            sample = make_sample("Example0001")
            write_generic_output(root, config, [sample])
            captured = []

            def capture_qc(path, **kwargs):
                captured.append(kwargs)
                return qc_stub(path, **kwargs)

            with mock.patch(
                "cmrbench.evaluate.ExperimentConfig.load", return_value=config
            ):
                evaluate_reconstruction(
                    root,
                    dataset_factory=lambda _config: SyntheticDataset([sample]),
                    sampling_factory=lambda _config: SyntheticSampling(),
                    anatomy_qc_writer=capture_qc,
                    fidelity_qc_writer=capture_qc,
                )
            self.assertEqual(len(captured), 2)
            expected_mask = SyntheticSampling().generate(
                sample.kspace.shape, acceleration=2, seed=7
            ).mask
            for call in captured:
                np.testing.assert_array_equal(call["canonical_mask"], expected_mask)
                self.assertEqual(call["actual_acceleration"], 2.0)
                self.assertEqual(call["error_cmap"], "turbo")
                self.assertEqual(tuple(call["frames"]), ((0, 0), (0, 1)))
            metadata = json.loads(
                (
                    root
                    / "evaluation"
                    / "qc"
                    / "Example0001_qc_metadata.json"
                ).read_text(encoding="utf-8")
            )
            display = metadata["display_provenance"]
            self.assertEqual(display["error_map_colormap"], "turbo")
            self.assertEqual(display["anatomy_error_display_range"], [0.0, 0.2])
            self.assertEqual(display["fidelity_error_display_range"], [0.0, 1.0])
            self.assertEqual(display["actual_acceleration_displayed"], 2.0)
            self.assertEqual(
                display["actual_acceleration_scope"],
                "global complete canonical sampling mask",
            )


class FastMRIEvaluationOnlySensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skimage_patcher = None
        if importlib.util.find_spec("skimage") is None:
            skimage = types.ModuleType("skimage")
            skimage.__path__ = []
            metrics = types.ModuleType("skimage.metrics")
            metrics.structural_similarity = lambda reference, candidate, **kwargs: float(
                1.0 - np.mean(np.abs(reference - candidate))
            )
            skimage.metrics = metrics
            cls.skimage_patcher = mock.patch.dict(
                sys.modules, {"skimage": skimage, "skimage.metrics": metrics}
            )
            cls.skimage_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if cls.skimage_patcher is not None:
            cls.skimage_patcher.stop()

    def make_config(self, *, reconstruction_maps: bool = False) -> ExperimentConfig:
        parameters = {
            "anatomy": "brain",
            "slice_index": 8,
            "scaling": "none",
            "prewhitened": False,
        }
        if reconstruction_maps:
            parameters["sensitivity_maps"] = {
                "source": "espirit",
                "implementation": "sigpy",
                "calibration": "sampling_acs",
                "calib_width": 24,
            }
        return ExperimentConfig.from_mapping(
            {
                "experiment": {"name": "fastmri-evaluation-only-test", "seed": 1},
                "dataset": {
                    "name": "fastmri_multicoil",
                    "manifest": "file_brain_test.h5",
                    "split": "validation",
                    "preprocess_root": "unused",
                    "parameters": parameters,
                },
                "sampling": {
                    "mode": "retrospective",
                    "method": "uniform",
                    "implementation": "canonical",
                    "acceleration": 4,
                    "seed": 1,
                    "parameters": {"ncalib": 24},
                },
                "model": {"name": "ram" if reconstruction_maps else "sdum"},
                "output": {"root": "unused"},
                "evaluation": {"qc": {"phase_indices": [0]}},
            }
        )

    def make_prepared(self, config: ExperimentConfig):
        random = np.random.default_rng(7)
        full = (
            random.normal(size=(1, 1, 32, 32, 3))
            + 1j * random.normal(size=(1, 1, 32, 32, 3))
        ).astype(np.complex64)
        sample = CineMRISample("fastmri-synthetic", full)
        prepared = prepare_reconstruction_input(
            sample,
            mode="retrospective",
            protocol=create_sampling_protocol(config),
            acceleration=4,
            seed=1,
        )
        return sample, prepared

    def test_missing_maps_are_evaluation_only_and_zero_filled_is_finite(self):
        config = self.make_config()
        _sample, model_input = self.make_prepared(config)
        self.assertIs(
            attach_dataset_sensitivity_maps(model_input, config.dataset.parameters),
            model_input,
        )
        seen = []

        def estimator(value, **kwargs):
            seen.append(value.copy())
            self.assertEqual(kwargs["calib_width"], 24)
            return np.ones_like(value)

        zero_fill_input, provenance = prepare_zero_filled_input(
            model_input, config, estimator=estimator, backend_version="test-sigpy"
        )
        self.assertIsNone(model_input.sensitivity_maps)
        self.assertIsNot(zero_fill_input, model_input)
        self.assertEqual(len(seen), 1)
        self.assertEqual(zero_fill_input.sensitivity_maps.shape, model_input.kspace.shape)
        self.assertEqual(provenance["purpose"], "evaluation_zero_filled")
        self.assertEqual(provenance["backend"], "sigpy")
        self.assertEqual(provenance["backend_version"], "test-sigpy")
        self.assertEqual(provenance["calib_width"], 24)
        self.assertGreaterEqual(provenance["realized_acs_extent"], 24)
        self.assertFalse(provenance["model_input_used"])
        self.assertTrue(provenance["finite"])
        zero_filled = ZeroFilledAdapter().reconstruct(zero_fill_input).reconstruction
        self.assertEqual(zero_filled.shape, (1, 1, 32, 32))
        self.assertTrue(np.isfinite(zero_filled).all())

    def test_unacquired_source_values_cannot_reach_estimator_or_zero_fill(self):
        config = self.make_config()
        sample, model_input = self.make_prepared(config)
        changed_source = sample.kspace.copy()
        changed_source[~np.broadcast_to(model_input.mask[..., None], changed_source.shape)] = 1e6 + 2e6j
        changed_sample = replace(sample, kspace=changed_source)
        recreated = prepare_reconstruction_input(
            changed_sample,
            mode="retrospective",
            protocol=create_sampling_protocol(config),
            acceleration=4,
            seed=1,
        )
        np.testing.assert_array_equal(recreated.kspace, model_input.kspace)
        observed = []

        def estimator(value, **_kwargs):
            observed.append(value.copy())
            return np.ones_like(value) * (1 + np.mean(np.abs(value)))

        first, _ = prepare_zero_filled_input(model_input, config, estimator=estimator)
        # Defense in depth: even a malformed acquired input is re-masked for evaluation.
        poisoned = replace(model_input, kspace=changed_source)
        second, _ = prepare_zero_filled_input(poisoned, config, estimator=estimator)
        np.testing.assert_array_equal(observed[0], observed[1])
        np.testing.assert_array_equal(first.sensitivity_maps, second.sensitivity_maps)
        np.testing.assert_array_equal(first.kspace, second.kspace)
        np.testing.assert_array_equal(
            ZeroFilledAdapter().reconstruct(first).reconstruction,
            ZeroFilledAdapter().reconstruct(second).reconstruction,
        )

    def test_ram_maps_are_reused_without_second_estimation(self):
        config = self.make_config(reconstruction_maps=True)
        _sample, prepared = self.make_prepared(config)
        with mock.patch(
            "cmrbench.data.sensitivity._sigpy_espirit",
            side_effect=lambda value, **_kwargs: np.ones_like(value),
        ) as estimator:
            with_maps = attach_dataset_sensitivity_maps(
                prepared, config.dataset.parameters
            )
            zero_fill_input, provenance = prepare_zero_filled_input(
                with_maps, config
            )
        estimator.assert_called_once()
        self.assertIs(zero_fill_input, with_maps)
        self.assertTrue(provenance["reused_without_reestimation"])
        self.assertFalse(provenance["estimated_in_evaluation"])
        self.assertEqual(provenance["preparation"]["source"], "espirit")

    def test_sdum_model_input_stays_map_free_and_mask_hash_is_unchanged(self):
        config = self.make_config()
        sample, model_input = self.make_prepared(config)
        model_input = attach_dataset_sensitivity_maps(
            model_input, config.dataset.parameters
        )
        expected_hash = model_input.sampling.parameters["binary_mask_sha256"]

        class FakeRuntime:
            def load(self):
                return {}

            def infer_volume(self, kspace, mask, conditioning, debug_hook=None):
                self_kspace_shape = kspace.shape
                self_mask_shape = mask.shape
                self_conditioning = conditioning
                assert self_kspace_shape == (1, 1, 3, 32, 32)
                assert self_mask_shape == (1, 1, 32, 32)
                assert self_conditioning.mask_type == "Uniform4"
                return np.ones((1, 1, 32, 32), dtype=np.float32), {
                    "runtime_seconds": 0.0,
                    "model_eval_mode": True,
                    "inference_context": "torch.no_grad",
                    "autocast": "CUDA bfloat16",
                }

        adapter = SDUMAdapter(
            sdum_source_root="unused",
            checkpoint="unused",
            variant="base",
            sampling_condition={"mode": "auto"},
            runtime_factory=lambda **_kwargs: FakeRuntime(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reconstruction = adapter.reconstruct(model_input)
        self.assertFalse(reconstruction.metadata["external_smaps_present"])
        self.assertFalse(reconstruction.metadata["external_smaps_used"])

        recreated = prepare_reconstruction_input(
            sample,
            mode="retrospective",
            protocol=create_sampling_protocol(config),
            acceleration=4,
            seed=1,
        )
        verified = _verify_sampling_parity(
            sample.subject_id, recreated.sampling, model_input.sampling.as_metadata()
        )
        self.assertEqual(verified["binary_mask_sha256"], expected_hash)
        with mock.patch(
            "cmrbench.data.sensitivity._sigpy_espirit",
            side_effect=lambda value, **_kwargs: np.ones_like(value),
        ):
            zero_fill_input, _ = prepare_zero_filled_input(recreated, config)
        self.assertEqual(
            zero_fill_input.sampling.parameters["binary_mask_sha256"], expected_hash
        )
        np.testing.assert_array_equal(zero_fill_input.mask, model_input.mask)

    def test_generic_fastmri_evaluation_completes_with_evaluation_only_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reconstruction"
            source_config = self.make_config()
            config = replace(source_config, output=replace(source_config.output, root=str(root)))
            sample, prepared = self.make_prepared(config)
            reference = np.linspace(0.1, 1.0, 32 * 32, dtype=np.float32).reshape(
                1, 1, 32, 32
            )
            sample = replace(sample, reference=reference)
            subject = sample.subject_id
            root.mkdir()
            subjects_root = root / "subjects"
            subjects_root.mkdir()
            candidate_path = subjects_root / f"{subject}_reconstruction.npy"
            candidate = reference * 0.9
            np.save(candidate_path, candidate, allow_pickle=False)
            sampling = prepared.sampling.as_metadata()
            model = {"model_name": "sdum", "model_version": "test"}
            subject_metadata = {
                "subject_id": subject,
                "model_name": "sdum",
                "model": model,
                "metadata": {"cmrbench_sampling": sampling},
                "sampling": sampling,
            }
            (subjects_root / f"{subject}_metadata.json").write_text(
                json.dumps(subject_metadata), encoding="utf-8"
            )
            (root / "config.yaml").write_text("effective config fixture\n", encoding="utf-8")
            (root / "experiment_metadata.json").write_text(
                json.dumps(
                    {
                        "experiment_name": config.experiment.name,
                        "config": config.as_dict(),
                        "sampling_mode": config.sampling.mode,
                        "sampling_method": config.sampling.method,
                        "sampling_implementation": config.sampling.implementation,
                        "requested_acceleration": config.sampling.acceleration,
                        "sampling_seed": config.sampling.seed,
                        "actual_accelerations": [prepared.sampling.actual_acceleration],
                        "resolved_sampling_implementations": ["canonical"],
                        "model": model,
                    }
                ),
                encoding="utf-8",
            )
            with (root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "subject_id", "model_name", "implementation", "variant",
                        "runtime_seconds", "sampling_method", "sampling_implementation",
                        "requested_acceleration", "actual_acceleration",
                        "sampling_seed", "sampling_parameters",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "subject_id": subject,
                        "model_name": "sdum",
                        "implementation": "",
                        "variant": "",
                        "runtime_seconds": 0.0,
                        **{
                            key: json.dumps(value) if key == "sampling_parameters" else value
                            for key, value in sampling.items()
                        },
                    }
                )

            with mock.patch("cmrbench.evaluate.ExperimentConfig.load", return_value=config), mock.patch(
                "cmrbench.data.sensitivity._sigpy_espirit",
                side_effect=lambda value, **_kwargs: np.ones_like(value),
            ) as estimator:
                rows = evaluate_reconstruction(
                    root,
                    dataset_factory=lambda _config: SyntheticDataset([sample]),
                    sampling_factory=create_sampling_protocol,
                    anatomy_qc_writer=qc_stub,
                    fidelity_qc_writer=qc_stub,
                )

            estimator.assert_called_once()
            self.assertEqual(len(rows), 1)
            self.assertTrue(np.isfinite(rows[0]["zf_nrmse"]))
            np.testing.assert_array_equal(np.load(candidate_path, allow_pickle=False), candidate)
            metadata = json.loads(
                (root / "evaluation" / "evaluation_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(
                metadata["subjects"][0]["sampling"]["verified_binary_mask_sha256"],
                prepared.sampling.parameters["binary_mask_sha256"],
            )
            maps = metadata["subjects"][0]["zero_filled_sensitivity_maps"]
            self.assertFalse(maps["model_input_used"])
            self.assertEqual(maps["calib_width"], 24)
            self.assertEqual(
                maps["binary_mask_sha256"],
                prepared.sampling.parameters["binary_mask_sha256"],
            )
            self.assertTrue(maps["finite"])
            self.assertTrue((root / "evaluation" / "metrics_v2.csv").is_file())
            self.assertTrue((root / "evaluation" / "metrics_v2.json").is_file())
            self.assertTrue((root / "evaluation" / "qc" / f"{subject}_anatomy_qc.png").is_file())
            self.assertTrue((root / "evaluation" / "qc" / f"{subject}_fidelity_qc.png").is_file())


if __name__ == "__main__":
    unittest.main()
