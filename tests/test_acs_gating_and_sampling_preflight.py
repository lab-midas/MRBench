from __future__ import annotations

import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from cmrbench.sampling.acs import add_centered_strip_acs
from cmrbench.data.base import CineMRIDataset
from cmrbench.data.sample import CineMRISample
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import prepare_reconstruction_input
from cmrbench.models.adapters.cardiomm import cardiomm_sampling_compatibility
from cmrbench.models.adapters.ram import RAMAdapter
from cmrbench.models.adapters.sdum import SDUMConditioning, sdum_sampling_compatibility
from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.registry import ModelRegistry
from cmrbench.models.result import ReconstructionResult
from cmrbench.reconstruct import run_reconstruction
from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.cardiomm import cardiomm_kt_radial_sampling
from cmrbench.sampling.external import ExternalMaskSamplingProtocol
from cmrbench.sampling.nvidia import (
    NVIDIASamplingProtocol,
    kt_radial_sampling,
    nvidia_equispaced_line_mask,
    nvidia_random_line_mask,
)
from cmrbench.sampling.preflight import (
    _sampling_config,
    registered_sampling_implementations,
    run_current_runtime_preflight,
)
from cmrbench.sampling.registry import SamplingRegistry
from cmrbench.sampling.result import (
    SamplingResult,
    calculate_actual_acceleration,
    validate_declared_central_acs,
)
from cmrbench.sampling.vista import VISTASamplingProtocol


def no_acs_mask(shape: tuple[int, int, int, int] = (1, 3, 8, 8)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[..., 0] = True
    mask[..., -1] = True
    return mask


def no_acs_input(*, method: str = "external_mask") -> ReconstructionInput:
    mask = no_acs_mask()
    kspace = np.ones(mask.shape + (2,), dtype=np.complex64) * mask[..., None]
    sampling = SamplingResult(
        mask=mask,
        requested_acceleration=4.0,
        actual_acceleration=calculate_actual_acceleration(mask),
        method=method,
        seed=3,
        parameters={
            "implementation": "external",
            "source_identity": "synthetic no-ACS development mask",
            "nominal_acceleration": 4,
        },
    )
    return ReconstructionInput(
        "synthetic-no-acs",
        kspace,
        mask,
        sensitivity_maps=np.ones((1, 1, 8, 8, 2), dtype=np.complex64),
        sampling=sampling,
    )


class ACSLayerTests(unittest.TestCase):
    def test_centered_strip_helper_is_pure_and_axis_explicit(self) -> None:
        source = np.zeros((6, 3), dtype=np.uint8)
        source[[0, 5], :] = 1
        result = add_centered_strip_acs(source, axis=0, width=2)
        np.testing.assert_array_equal(source[1:5], 0)
        self.assertEqual(result.dtype, np.bool_)
        self.assertTrue(np.all(result[2:4]))
        self.assertTrue(np.all(result[[0, 5]]))

    def test_generic_external_and_canonical_preparation_accept_no_acs(self) -> None:
        mask = no_acs_mask()
        protocol = ExternalMaskSamplingProtocol(mask=mask, layout="S,T,Kx,Ky")
        sample = CineMRISample(
            "synthetic",
            np.ones(mask.shape + (2,), dtype=np.complex64),
            sensitivity_maps=np.ones((1, 1, 8, 8, 2), dtype=np.complex64),
        )
        prepared = prepare_reconstruction_input(
            sample,
            mode="retrospective",
            protocol=protocol,
            acceleration=4,
            seed=3,
        )
        metadata = prepared.sampling.physical_mask_validation
        self.assertFalse(metadata["has_complete_central_acs"])
        self.assertEqual(metadata["detected_acs_geometry"], "none")
        np.testing.assert_array_equal(prepared.mask, mask)
        np.testing.assert_array_equal(
            prepared.kspace, sample.kspace * mask[..., None]
        )

    def test_ram_pre_inference_interface_accepts_external_smaps_and_no_acs(self) -> None:
        value = no_acs_input()
        kspace, mask, smaps = RAMAdapter._validate_input(value)
        self.assertEqual(kspace.shape, (1, 3, 8, 8, 2))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(smaps.shape, (1, 1, 8, 8, 2))

    def test_vista_precomputed_no_acs_is_not_globally_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact = np.zeros((8, 3), dtype=np.uint8)
            compact[[0, 7], :] = 1
            np.savetxt(root / "mask_VISTA_8x3_acc4_3.txt", compact, fmt="%d", delimiter=",")
            result = VISTASamplingProtocol(mask_root=root).generate(
                (1, 3, 8, 8, 2), 4, seed=3
            )
        self.assertFalse(
            result.physical_mask_validation["has_complete_central_acs"]
        )

    def test_sdum_and_cardiomm_defer_undeclared_no_acs_to_model_runtime(self) -> None:
        value = no_acs_input()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdum = sdum_sampling_compatibility(
                value,
                SDUMConditioning("ktGaussian8", 8, "Cine"),
                variant="base",
                conditioning_metadata={"conditioning_relation_to_physical_sampling": "user_forced"},
            )
            cardiomm = cardiomm_sampling_compatibility(
                value,
                {
                    "mode": "explicit",
                    "mask_type": "random",
                    "acceleration": 8,
                    "num_low_frequencies": 4,
                },
            )
        for record in (sdum, cardiomm):
            self.assertFalse(record["has_complete_central_acs"])
            self.assertEqual(record["physical_sampling_validity"], "valid")
        self.assertEqual(sdum["acs_requirement_layer"], "upstream_sdum_runtime")
        self.assertEqual(sdum["runtime_compatibility"], "upstream_expected_rejection")
        self.assertEqual(
            cardiomm["acs_requirement_layer"], "upstream_cardiomm_runtime"
        )
        self.assertEqual(cardiomm["runtime_compatibility"], "upstream_deferred")

    def test_declared_generator_acs_contract_remains_hard(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "declared centered_ky_strip"):
            validate_declared_central_acs(
                no_acs_mask(),
                ncalib=2,
                geometry="centered_ky_strip",
                source="synthetic generator",
            )
        result = NVIDIASamplingProtocol(family="uniform", ncalib=4).generate(
            (1, 3, 8, 16, 1), 4
        )
        self.assertTrue(result.parameters["generator_acs_contract_enforced"])
        self.assertGreaterEqual(
            result.parameters["detected_extent_for_declared_geometry"], 4
        )

    def test_declared_external_acs_mismatch_is_truthfully_rejected_by_models(self) -> None:
        value = no_acs_input()
        value.sampling.parameters["ncalib"] = 4
        with self.assertRaisesRegex(ValueError, "declared ncalib provenance conflicts"):
            sdum_sampling_compatibility(
                value, SDUMConditioning("ktGaussian8", 8, "Cine"), variant="base"
            )
        with self.assertRaisesRegex(ValueError, "declared ncalib provenance conflicts"):
            cardiomm_sampling_compatibility(
                value,
                {
                    "mode": "explicit",
                    "mask_type": "random",
                    "acceleration": 8,
                    "num_low_frequencies": 4,
                },
            )

    def test_actual_acceleration_is_part_of_generic_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "actual_acceleration is inconsistent"):
            SamplingResult(no_acs_mask(), 4, 99, "external_mask")


class RasterizerAndPreflightTests(unittest.TestCase):
    def test_acs20_is_mathematically_invalid_at_ky156_r8_for_fastmri_paths(self) -> None:
        center_fraction = 20 / 156
        with self.assertRaisesRegex(ValueError, "invalid outer sampling probability"):
            nvidia_random_line_mask(156, center_fraction, 8, 7)
        with self.assertRaisesRegex(ValueError, "invalid adjusted acceleration"):
            nvidia_equispaced_line_mask(156, center_fraction, 8, 7)
        random_mask = nvidia_random_line_mask(156, 0.04, 8, 7)
        equispaced_mask, adjusted, _offset = nvidia_equispaced_line_mask(
            156, 0.04, 8, 7
        )
        self.assertEqual(int(round(156 * 0.04)), 6)
        self.assertTrue(random_mask.any())
        self.assertTrue(equispaced_mask.any())
        self.assertGreater(adjusted, 0)

    def test_missing_radial_dependencies_are_not_substituted(self) -> None:
        real_import = builtins.__import__

        def missing_scipy(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise ImportError("synthetic missing scipy")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing_scipy):
            with self.assertRaisesRegex(RuntimeError, "requires scipy"):
                kt_radial_sampling(8, 8, 1, 2, 4.8, 137.5, True)

        def missing_skimage(name, *args, **kwargs):
            if name.startswith("skimage"):
                raise ImportError("synthetic missing skimage")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing_skimage):
            with self.assertRaisesRegex(RuntimeError, "requires scikit-image"):
                cardiomm_kt_radial_sampling(8, 8, 1, 2, 4.8, 137.5, True)

    def test_runtime_preflight_enumerates_actual_registry_and_never_loads_model(self) -> None:
        rows = registered_sampling_implementations()
        self.assertEqual(len(rows), 16)
        self.assertIn(
            {
                "method": "vista",
                "implementation": "cmrbench",
                "dependencies": ["numpy"],
            },
            rows,
        )
        self.assertIn(
            {"method": "kt_radial", "implementation": "sdum", "dependencies": ["numpy", "scipy"]},
            rows,
        )
        config = _sampling_config("uniform", "canonical", {"ncalib": 2})
        with mock.patch(
            "cmrbench.sampling.preflight._preflight_configs",
            return_value=[config],
        ):
            report = run_current_runtime_preflight()
        self.assertEqual(report["pass_count"], 1)
        self.assertFalse(report["model_instantiated"])
        self.assertFalse(report["model_load_called"])
        self.assertFalse(report["checkpoint_loaded"])
        self.assertFalse(report["model_forward_called"])


class _OneSample(CineMRIDataset):
    def __init__(self, sample: CineMRISample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> CineMRISample:
        if index:
            raise IndexError(index)
        return self.sample


class _DependencyFailingSampling(SamplingProtocol):
    def generate(self, shape, acceleration, seed=None, **kwargs):
        raise RuntimeError("kt_radial generation requires scipy")


class _CallTrackingModel(ReconstructionModel):
    reconstruct_calls = 0

    def load(self) -> None:
        return None

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        type(self).reconstruct_calls += 1
        raise AssertionError("model inference must not be called")

    def model_info(self) -> ModelInfo:
        return ModelInfo("tracking", "1", {})


class SamplingFailureClassificationTests(unittest.TestCase):
    def test_sampling_dependency_failure_is_preparation_not_model_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.from_mapping(
                {
                    "experiment": {"name": "dependency-failure"},
                    "dataset": {"name": "inhouse_cine", "manifest": "unused", "split": "development"},
                    "sampling": {"mode": "retrospective", "method": "failing", "acceleration": 8},
                    "model": {"name": "tracking"},
                    "output": {"root": str(Path(directory) / "run")},
                }
            )
            sample = CineMRISample(
                "synthetic",
                np.ones((1, 2, 4, 4, 1), dtype=np.complex64),
                sensitivity_maps=np.ones((1, 1, 4, 4, 1), dtype=np.complex64),
            )
            models = ModelRegistry()
            models.register("tracking", _CallTrackingModel)
            samplings = SamplingRegistry()
            samplings.register("failing", _DependencyFailingSampling)
            _CallTrackingModel.reconstruct_calls = 0
            with self.assertRaisesRegex(RuntimeError, "requires scipy"):
                run_reconstruction(
                    config,
                    dataset_factory=lambda _config: _OneSample(sample),
                    model_registry=models,
                    sampling_registry=samplings,
                )
            failure = json.loads(
                (Path(config.output.root) / "reconstruction_failure.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(failure["stage"], "physical_sampling_preparation")
        self.assertEqual(failure["source"], "cmrbench")
        self.assertEqual(_CallTrackingModel.reconstruct_calls, 0)


if __name__ == "__main__":
    unittest.main()
