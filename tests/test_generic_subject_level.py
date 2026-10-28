"""Subject-level dataset migration and static multi-slice regression tests."""

from __future__ import annotations

import sys
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.data.sensitivity import prepare_sensitivity_maps
from cmrbench.evaluate import prepare_zero_filled_input
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import prepare_reconstruction_input
from cmrbench.experiment.runner import ExperimentRunner
from cmrbench.models.adapters.sdum import (
    resolve_sdum_conditioning, sdum_sampling_compatibility,
)
from cmrbench.reconstruct import DatasetRegistry, create_cmr_dataset, create_sampling_protocol
from cmrbench.sampling.result import binary_mask_provenance
from tests.test_generic_reconstruction import FakeModel


def config_for(dataset: dict, *, model: str = "sdum") -> ExperimentConfig:
    return ExperimentConfig.from_mapping({
        "experiment": {"name": "subject-test"},
        "dataset": dataset,
        "sampling": {
            "mode": "retrospective", "method": "uniform",
            "implementation": "canonical", "acceleration": 8,
            "seed": 1, "parameters": {"ncalib": 4},
        },
        "model": {"name": model},
        "output": {"root": "outputs/subject-test"},
        "evaluation": {"qc": {"phase_indices": [0]}},
    })


class _FakeH5:
    def __init__(self, kspace: np.ndarray, reference: np.ndarray) -> None:
        self.data = {"kspace": kspace, "reconstruction_rss": reference}
        self.attrs = {"acquisition": "AXFLAIR"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __contains__(self, name):
        return name in self.data

    def __getitem__(self, name):
        return self.data[name]


class GenericSubjectLevelTests(unittest.TestCase):

    def test_old_cine_selector_resolves_complete_subject(self) -> None:
        config = config_for({
            "name": "inhouse_cine", "manifest": "cohort.csv",
            "split": "ineligible", "parameters": {"subject": "Example0132"},
        })
        self.assertEqual(config.dataset.domain, "cardiac_cine")
        self.assertEqual(config.dataset.selection, {"subject": "Example0132", "slices": "all"})
        self.assertEqual(config.dataset.resolution["subject"], "dataset.parameters.subject")
        self.assertEqual(
            ExperimentConfig.from_mapping(config.as_dict()).as_dict(), config.as_dict()
        )

    def test_old_fastmri_slice_selector_preserves_smoke_semantics(self) -> None:
        config = config_for({
            "name": "fastmri_multicoil", "manifest": "brain.h5",
            "split": "validation", "preprocess_root": "/data",
            "parameters": {"anatomy": "brain", "slice_index": 8, "scaling": "none"},
        })
        self.assertEqual(config.dataset.selection, {"subject": "brain.h5", "slices": 8})
        self.assertEqual(config.dataset.resolution["slices"], "dataset.parameters.slice_index")
        self.assertEqual(ExperimentConfig.from_mapping(config.as_dict()).as_dict(), config.as_dict())
        dataset = create_cmr_dataset(config)
        self.assertEqual(dataset.slice_index, 8)

    def test_new_volume_selector_and_reader_registry(self) -> None:
        config = config_for({
            "name": "fastmri_multicoil", "domain": "brain",
            "manifest": "brain.h5", "split": "validation",
            "preprocess_root": "/data",
            "selection": {"subject": "brain.h5", "slices": "all"},
            "parameters": {"scaling": "none"},
        })
        self.assertEqual(create_cmr_dataset(config).slice_index, "all")
        self.assertEqual(config.dataset.resolution["slices"], "dataset.selection.slices")
        registry = DatasetRegistry()
        registry.register("fastmri_multicoil", lambda _config: "registered-reader")
        self.assertEqual(create_cmr_dataset(config, registry), "registered-reader")
        custom = config_for({
            "name": "future_custom", "manifest": "subject.raw",
            "split": "validation", "preprocess_root": "/data",
            "domain": "custom_domain",
            "selection": {"subject": "subject.raw", "slices": "all"},
        })
        registry.register("future_custom", lambda passed: passed.dataset.selection)
        self.assertEqual(
            create_cmr_dataset(custom, registry),
            {"subject": "subject.raw", "slices": "all"},
        )
        with self.assertRaisesRegex(ValueError, "requires selection.slices"):
            config_for({
                "name": "fastmri_multicoil", "domain": "brain",
                "manifest": "brain.h5", "split": "validation",
            })

    def test_whole_hdf5_volume_and_one_mask_across_slices_and_coils(self) -> None:
        config = config_for({
            "name": "fastmri_multicoil", "domain": "brain",
            "manifest": "brain.h5", "split": "validation",
            "preprocess_root": "/data",
            "selection": {"subject": "brain.h5", "slices": "all"},
            "parameters": {"scaling": "none"},
        })
        rng = np.random.default_rng(12)
        source = (rng.standard_normal((3, 2, 10, 8)) +
                  1j * rng.standard_normal((3, 2, 10, 8))).astype(np.complex64)
        reference = np.ones((3, 8, 8), dtype=np.float32)
        h5py = types.ModuleType("h5py")
        h5py.File = lambda *_args, **_kwargs: _FakeH5(source, reference)
        with mock.patch.dict(sys.modules, {"h5py": h5py}):
            sample = create_cmr_dataset(config)[0]
        self.assertEqual(sample.subject_id, "brain")
        self.assertEqual(sample.kspace.shape, (3, 1, 8, 8, 2))
        self.assertEqual(sample.reference.shape, (3, 1, 8, 8))
        self.assertEqual(sample.metadata["number_of_slices"], 3)
        self.assertEqual(
            sample.metadata["input_preprocessing"],
            {
                "prewhitened": False,
                "source": "fastmri_multicoil raw-input default",
                "prewhitening_applied_by_cmrbench": False,
            },
        )
        self.assertIsNone(sample.sensitivity_maps)
        prepared = prepare_reconstruction_input(
            sample, mode="retrospective", protocol=create_sampling_protocol(config),
            acceleration=8, seed=1,
        )
        self.assertEqual(prepared.kspace.shape, (3, 1, 8, 8, 2))
        self.assertEqual(prepared.mask.shape, (3, 1, 8, 8))
        for index in (1, 2):
            np.testing.assert_array_equal(prepared.mask[0], prepared.mask[index])
        self.assertTrue(np.all(prepared.kspace[~np.broadcast_to(
            prepared.mask[..., None], prepared.kspace.shape
        )] == 0))
        self.assertEqual(
            prepared.sampling.parameters["binary_mask_sha256"],
            binary_mask_provenance(prepared.mask)["binary_mask_sha256"],
        )

    def test_runner_writes_one_complete_static_subject_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "formal"
            raw = config_for({
                "name": "fastmri_multicoil", "domain": "brain",
                "manifest": "brain.h5", "split": "validation",
                "preprocess_root": "/data",
                "selection": {"subject": "brain.h5", "slices": "all"},
                "parameters": {"scaling": "none"},
            }, model="fake").as_dict()
            raw["output"]["root"] = str(output)
            config = ExperimentConfig.from_mapping(raw)
            source = np.ones((3, 2, 10, 8), dtype=np.complex64)
            reference = np.ones((3, 8, 8), dtype=np.float32)
            h5py = types.ModuleType("h5py")
            h5py.File = lambda *_args, **_kwargs: _FakeH5(source, reference)
            yaml = types.ModuleType("yaml")
            yaml.safe_dump = lambda value, handle, **_kwargs: json.dump(value, handle)
            with mock.patch.dict(sys.modules, {"h5py": h5py, "yaml": yaml}):
                results = ExperimentRunner(
                    config,
                    dataset_factory=create_cmr_dataset,
                    model=FakeModel(),
                    sampling_protocol=create_sampling_protocol(config),
                ).run()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].subject_id, "brain")
            self.assertEqual(
                np.load(output / "subjects" / "brain_reconstruction.npy").shape,
                (3, 1, 8, 8),
            )
            metadata = json.loads(
                (output / "subjects" / "brain_metadata.json").read_text()
            )
            self.assertEqual(metadata["dataset"]["canonical_kspace_shape"], [3, 1, 8, 8, 2])
            self.assertEqual(metadata["dataset"]["selection"]["slices"], "all")
            self.assertFalse(
                metadata["dataset"]["source_metadata"]["input_preprocessing"]["prewhitened"]
            )
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "experiment_metadata.json").is_file())

    def test_espirit_estimates_each_slice_from_its_own_acquired_acs(self) -> None:
        kspace = np.zeros((3, 1, 8, 8, 2), dtype=np.complex64)
        for index in range(3):
            kspace[index] = index + 1
            kspace[index, :, :, :2, :] = 1000 + index  # outside mask
        mask = np.zeros((3, 1, 8, 8), dtype=bool)
        mask[..., 2:6] = True
        calls: list[np.ndarray] = []

        def estimator(value, **_kwargs):
            calls.append(value.copy())
            return np.ones_like(value)

        prepared = prepare_sensitivity_maps(
            kspace=kspace, mask=mask, provided_maps=None,
            config={"strategy": "espirit", "calib_width": 4},
            estimator=estimator, backend_version="test",
        )
        self.assertEqual(prepared.maps.shape, (3, 1, 8, 8, 2))
        self.assertEqual(len(calls), 3)
        for index, value in enumerate(calls):
            self.assertTrue(np.all(value[..., 2:6] == index + 1))
            self.assertTrue(np.all(value[..., :2] == 0))
            self.assertTrue(np.all(value[..., 6:] == 0))

    def test_evaluation_only_espirit_is_per_slice_and_does_not_modify_model_input(self):
        raw = config_for({
            "name": "fastmri_multicoil", "domain": "brain",
            "manifest": "brain.h5", "split": "validation",
            "preprocess_root": "/data",
            "selection": {"subject": "brain.h5", "slices": "all"},
            "parameters": {"scaling": "none"},
        }).as_dict()
        raw["sampling"]["parameters"]["ncalib"] = 24
        config = ExperimentConfig.from_mapping(raw)
        complete = np.zeros((3, 1, 8, 32, 2), dtype=np.complex64)
        for index in range(3):
            complete[index] = index + 1
        sample = CineMRISample("brain", complete)
        model_input = prepare_reconstruction_input(
            sample, mode="retrospective", protocol=create_sampling_protocol(config),
            acceleration=8, seed=1,
        )
        calls: list[np.ndarray] = []

        def estimator(value, **_kwargs):
            calls.append(value.copy())
            return np.ones_like(value)

        zero_fill, provenance = prepare_zero_filled_input(
            model_input, config, estimator=estimator, backend_version="test"
        )
        self.assertIsNone(model_input.sensitivity_maps)
        self.assertEqual(zero_fill.sensitivity_maps.shape, (3, 1, 8, 32, 2))
        self.assertEqual(len(calls), 3)
        for index, value in enumerate(calls):
            self.assertTrue(np.all(value[:, :, 4:28] == index + 1))
        self.assertTrue(provenance["estimated_in_evaluation"])

    def test_exp101_uniform_r8_has_matched_small_conditioning(self):
        raw = config_for({
            "name": "fastmri_multicoil", "domain": "brain",
            "manifest": "brain.h5", "split": "validation",
            "preprocess_root": "/data",
            "selection": {"subject": "brain.h5", "slices": "all"},
            "parameters": {"scaling": "none"},
        }).as_dict()
        raw["sampling"]["parameters"]["ncalib"] = 24
        config = ExperimentConfig.from_mapping(raw)
        sample = CineMRISample(
            "brain", np.ones((3, 1, 32, 32, 2), dtype=np.complex64)
        )
        input = prepare_reconstruction_input(
            sample, mode="retrospective", protocol=create_sampling_protocol(config),
            acceleration=8, seed=1,
        )
        condition, details = resolve_sdum_conditioning(
            input, sampling_condition={"mode": "auto"},
            legacy_conditioning=None, default_acquisition="Cine",
        )
        record = sdum_sampling_compatibility(
            input, condition, variant="small", conditioning_metadata=details
        )
        self.assertEqual((condition.mask_type, condition.acceleration), ("Uniform8", 8))
        self.assertEqual(record["resolved_acc_idx"], 0)
        self.assertEqual(record["compatibility_status"], "supported")
        self.assertFalse(record["matches_documented_reference_ncalib"])


if __name__ == "__main__":
    unittest.main()
