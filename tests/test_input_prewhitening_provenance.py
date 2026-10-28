"""Input prewhitening is dataset provenance, never an action."""

from __future__ import annotations

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.experiment.config import ExperimentConfig, resolve_dataset_section
from cmrbench.reconstruct import create_cmr_dataset
from tests.test_generic_subject_level import config_for


def fastmri_dataset(parameters: dict | None = None) -> dict:
    return {
        "name": "fastmri_multicoil",
        "domain": "brain",
        "manifest": "brain.h5",
        "split": "validation",
        "preprocess_root": "/data",
        "selection": {"subject": "brain.h5", "slices": "all"},
        "parameters": {"scaling": "none", **(parameters or {})},
    }


class InputPrewhiteningProvenanceTests(unittest.TestCase):
    def test_fastmri_default_and_legacy_alias_values(self):
        default = config_for(fastmri_dataset())
        self.assertFalse(default.dataset.input_prewhitened)
        self.assertFalse(default.dataset.parameters["prewhitened"])
        self.assertEqual(
            default.dataset.resolution["input_prewhitened"],
            "fastmri_multicoil raw-input default",
        )
        self.assertEqual(
            config_for(fastmri_dataset({"prewhitened": False})).dataset.input_prewhitened,
            False,
        )
        self.assertEqual(
            config_for(fastmri_dataset({"prewhitened": True})).dataset.input_prewhitened,
            True,
        )

    def test_new_and_legacy_declarations_must_agree(self):
        matching = config_for(fastmri_dataset({
            "input_prewhitened": True, "prewhitened": True,
        }))
        self.assertTrue(matching.dataset.input_prewhitened)
        top_level = fastmri_dataset({"prewhitened": True})
        top_level["input_prewhitened"] = True
        self.assertTrue(config_for(top_level).dataset.input_prewhitened)
        self.assertEqual(
            ExperimentConfig.from_mapping(matching.as_dict()).as_dict(), matching.as_dict()
        )
        with self.assertRaisesRegex(ValueError, "conflicts with"):
            config_for(fastmri_dataset({
                "input_prewhitened": True, "prewhitened": False,
            }))

    def test_true_is_metadata_only_and_kspace_is_unchanged(self):
        source = np.ones((1, 1, 8, 8, 2), dtype=np.complex64) * (2 + 3j)
        reference = np.ones((1, 1, 8, 8), dtype=np.float32)

        def read_volume(_path):
            return CineMRISample("brain", source.copy(), reference=reference.copy())

        with mock.patch("cmrbench.data.fastmri.load_fastmri_multicoil_volume", read_volume):
            plain = create_cmr_dataset(config_for(fastmri_dataset()))[0]
            declared = create_cmr_dataset(
                config_for(fastmri_dataset({"prewhitened": True}))
            )[0]
        np.testing.assert_array_equal(declared.kspace, plain.kspace)
        self.assertTrue(declared.metadata["input_preprocessing"]["prewhitened"])
        self.assertFalse(
            declared.metadata["input_preprocessing"]["prewhitening_applied_by_cmrbench"]
        )

    def test_inhouse_unknown_unless_explicit_and_custom_can_declare(self):
        cine = {
            "name": "inhouse_cine", "manifest": "cohort.csv", "split": "validation",
        }
        self.assertIsNone(resolve_dataset_section(cine)["input_prewhitened"])
        self.assertTrue(resolve_dataset_section({
            **cine, "parameters": {"prewhitened": True},
        })["input_prewhitened"])
        custom = {
            "name": "future_custom", "manifest": "subject.raw", "split": "validation",
            "parameters": {"input_prewhitened": True},
        }
        self.assertTrue(resolve_dataset_section(custom)["input_prewhitened"])

    def test_legacy_inhouse_value_reaches_subject_metadata_without_transform(self):
        original = np.ones((1, 1, 8, 8, 2), dtype=np.complex64) * (4 + 1j)

        class Source:
            records = [SimpleNamespace(subject_id="Example0132")]
            split = "validation"

            def __getitem__(self, _index):
                return CineMRISample("Example0132", original.copy())

        config = config_for({
            "name": "inhouse_cine", "manifest": "cohort.csv",
            "split": "validation", "preprocess_root": "/data",
            "parameters": {"subject": "Example0132", "prewhitened": True},
        })
        with mock.patch("cmrbench.reconstruct.CohortManifest.load", return_value=object()), \
                mock.patch("cmrbench.reconstruct.InHouseCineDataset", return_value=Source()):
            sample = create_cmr_dataset(config)[0]
        np.testing.assert_array_equal(sample.kspace, original)
        self.assertTrue(sample.metadata["input_preprocessing"]["prewhitened"])
        self.assertFalse(
            sample.metadata["input_preprocessing"]["prewhitening_applied_by_cmrbench"]
        )


if __name__ == "__main__":
    unittest.main()
