"""Physical-sampling calibration policy shared by reconstruction/evaluation."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.evaluate import prepare_zero_filled_input
from cmrbench.experiment.calibration import (
    resolve_requested_calibration_width,
)
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import (
    attach_dataset_sensitivity_maps,
    prepare_reconstruction_input,
)
from cmrbench.models.adapters.cardiomm import cardiomm_sampling_compatibility
from cmrbench.models.input import ReconstructionInput
from cmrbench.sampling.nvidia import (
    CanonicalUniformSamplingProtocol,
    NVIDIAStaticLineSamplingProtocol,
)
from cmrbench.sampling.result import SamplingResult, calculate_actual_acceleration


def fastmri_config(*, ncalib: int) -> ExperimentConfig:
    return ExperimentConfig.from_mapping({
        "experiment": {"name": "calibration-test"},
        "dataset": {
            "name": "fastmri_multicoil", "domain": "brain",
            "manifest": "brain.h5", "split": "validation",
            "preprocess_root": "/data",
            "selection": {"subject": "brain.h5", "slices": "all"},
            "parameters": {"scaling": "none"},
        },
        "sampling": {
            "mode": "retrospective", "method": "uniform",
            "implementation": "canonical", "acceleration": 4,
            "seed": 1, "parameters": {"ncalib": ncalib},
        },
        "model": {"name": "ram"},
        "output": {"root": "unused"},
        "evaluation": {"qc": {"phase_indices": [0]}},
    })


def sampled_uniform(*, ncalib: int, ky: int = 320) -> ReconstructionInput:
    sample = CineMRISample(
        "brain", np.ones((1, 1, 8, ky, 2), dtype=np.complex64)
    )
    return prepare_reconstruction_input(
        sample, mode="retrospective",
        protocol=CanonicalUniformSamplingProtocol(ncalib=ncalib),
        acceleration=4, seed=1,
    )


def espirit_config(width=None):
    result = {"source": "espirit", "implementation": "sigpy",
              "calibration": "sampling_acs"}
    if width is not None:
        result["calib_width"] = width
    return result


class GenericCalibrationTests(unittest.TestCase):
    def test_uniform_24_and_20_drive_reconstruction_and_evaluation(self):
        for width in (24, 20):
            with self.subTest(width=width):
                input = sampled_uniform(ncalib=width)
                config = fastmri_config(ncalib=width)
                with mock.patch(
                    "cmrbench.data.sensitivity._sigpy_espirit",
                    side_effect=lambda value, **_kwargs: np.ones_like(value),
                ) as estimate:
                    prepared = attach_dataset_sensitivity_maps(
                        input, {"sensitivity_maps": espirit_config()},
                        configured_sampling=config.sampling,
                    )
                self.assertEqual(estimate.call_args.kwargs["calib_width"], width)
                self.assertEqual(
                    prepared.metadata["cmrbench_calibration"]["reconstruction_espirit_calib_width"],
                    width,
                )
                plain = sampled_uniform(ncalib=width)
                observed = []

                def evaluator(value, **kwargs):
                    observed.append(kwargs["calib_width"])
                    return np.ones_like(value)

                evaluated, provenance = prepare_zero_filled_input(
                    plain, config, estimator=evaluator, backend_version="fake"
                )
                self.assertEqual(observed, [width])
                self.assertEqual(provenance["requested_calibration_width"], width)
                self.assertEqual(
                    evaluated.metadata["cmrbench_calibration"]["evaluation_espirit_calib_width"],
                    width,
                )

    def test_realized_25_does_not_replace_requested_24(self):
        input = sampled_uniform(ncalib=24)
        record = input.metadata["cmrbench_calibration"]
        self.assertEqual(record["requested_calibration_width"], 24)
        self.assertEqual(record["realized_acs_width"], 25)
        _, provenance = prepare_zero_filled_input(
            input, fastmri_config(ncalib=24),
            estimator=lambda value, **_kwargs: np.ones_like(value),
            backend_version="fake",
        )
        self.assertEqual(provenance["calib_width"], 24)
        self.assertEqual(provenance["realized_acs_width"], 25)

    def test_realized_acs_less_than_requested_fails(self):
        mask = np.zeros((1, 1, 8, 32), dtype=bool)
        mask[..., 6:26] = True
        sampling = SamplingResult(
            mask=mask, requested_acceleration=4,
            actual_acceleration=calculate_actual_acceleration(mask),
            method="uniform", parameters={"implementation": "canonical", "ncalib": 24},
        )
        input = ReconstructionInput(
            "brain", np.ones((1, 1, 8, 32, 2), dtype=np.complex64) * mask[..., None],
            mask, sampling=sampling,
        )
        with self.assertRaisesRegex(ValueError, "exceeds realized central ACS extent"):
            attach_dataset_sensitivity_maps(
                input, {"sensitivity_maps": espirit_config()}
            )

    def test_center_fraction_uses_runtime_ky_for_random_and_equispaced(self):
        for family in ("random", "equispaced"):
            for ky, expected in ((320, 26), (400, 32)):
                with self.subTest(family=family, ky=ky):
                    sample = CineMRISample(
                        "brain", np.ones((1, 1, 8, ky, 2), dtype=np.complex64)
                    )
                    input = prepare_reconstruction_input(
                        sample, mode="retrospective",
                        protocol=NVIDIAStaticLineSamplingProtocol(
                            family=family, center_fraction=0.08
                        ), acceleration=4, seed=1,
                    )
                    resolution = resolve_requested_calibration_width(
                        input.sampling, tuple(input.kspace.shape)
                    )
                    self.assertEqual(resolution.calibration_source, "center_fraction")
                    self.assertEqual(resolution.requested_calibration_width, expected)
                    config_raw = fastmri_config(ncalib=24).as_dict()
                    config_raw["sampling"]["method"] = family
                    config_raw["sampling"]["implementation"] = "sdum"
                    config_raw["sampling"]["parameters"] = {"center_fraction": 0.08}
                    config = ExperimentConfig.from_mapping(config_raw)
                    observed = []

                    def estimator(value, **kwargs):
                        observed.append(kwargs["calib_width"])
                        return np.ones_like(value)

                    prepare_zero_filled_input(
                        input, config, estimator=estimator, backend_version="fake"
                    )
                    self.assertEqual(observed, [expected])

    def test_odd_rounded_center_fraction_width_is_resolved_without_geometry_hardcode(self):
        resolution = resolve_requested_calibration_width(
            type("Policy", (), {"method": "random", "parameters": {"center_fraction": 0.08}})(),
            (1, 1, 8, 384, 2),
        )
        self.assertEqual(resolution.requested_calibration_width, 31)

    def test_matching_legacy_calib_width_passes_and_conflict_fails(self):
        input = sampled_uniform(ncalib=24)
        with mock.patch(
            "cmrbench.data.sensitivity._sigpy_espirit",
            side_effect=lambda value, **_kwargs: np.ones_like(value),
        ):
            prepared = attach_dataset_sensitivity_maps(
                input, {"sensitivity_maps": espirit_config(24)}
            )
        self.assertEqual(
            prepared.metadata["cmrbench_sensitivity_maps"]["calibration"]["width"], 24
        )
        with self.assertRaisesRegex(ValueError, "explicit ESPIRiT calib_width=20 conflicts"):
            attach_dataset_sensitivity_maps(
                input, {"sensitivity_maps": espirit_config(20)}
            )

    def test_legacy_width_conflicts_with_runtime_center_fraction(self):
        sample = CineMRISample(
            "brain", np.ones((1, 1, 8, 320, 2), dtype=np.complex64)
        )
        input = prepare_reconstruction_input(
            sample, mode="retrospective",
            protocol=NVIDIAStaticLineSamplingProtocol(
                family="random", center_fraction=0.08
            ), acceleration=4, seed=1,
        )
        with self.assertRaisesRegex(
            ValueError, "explicit ESPIRiT calib_width=24 conflicts.*width=26"
        ):
            attach_dataset_sensitivity_maps(
                input, {"sensitivity_maps": espirit_config(24)}
            )

    def test_cardiomm_requested_width_and_explicit_compatibility(self):
        input = sampled_uniform(ncalib=24)
        self.assertEqual(
            cardiomm_sampling_compatibility(input)["model_internal_num_low_frequencies"],
            24,
        )
        self.assertEqual(
            cardiomm_sampling_compatibility(
                input, default_num_low_frequencies=24
            )["model_internal_num_low_frequencies"], 24,
        )
        with self.assertRaisesRegex(ValueError, "num_low_frequencies=20 conflicts"):
            cardiomm_sampling_compatibility(input, default_num_low_frequencies=20)
        with self.assertRaisesRegex(ValueError, "sampling_condition.num_low_frequencies=20 conflicts"):
            cardiomm_sampling_compatibility(
                input, {"mode": "explicit", "mask_type": "uniform",
                        "num_low_frequencies": 20},
            )


if __name__ == "__main__":
    unittest.main()
