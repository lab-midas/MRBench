from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from cmrbench.models.adapters import sdum as sdum_module
from cmrbench.models.adapters.sdum import (
    SDUMAdapter,
    SDUMConditioning,
    canonical_to_sdum_private,
    complex_zscore_reference,
    cyclic_temporal_indices,
    expand_private_mask,
    preflight_conditioning,
    sdum_sampling_compatibility,
    sdum_variant_spec,
    sdum_rss_to_canonical,
    strict_mask_support,
    temporary_torch_threads,
)
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.registry import default_model_registry
from cmrbench.sampling.result import SamplingResult, calculate_actual_acceleration


class FakeSDUMRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def load(self) -> dict[str, object]:
        return {
            "source_commit_actual": sdum_module.SDUM_COMMIT,
            "checkpoint_sha256": sdum_module.SMALL_CHECKPOINT_SHA256,
            "parameter_count": 1,
        }

    def infer_volume(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        conditioning: SDUMConditioning,
        debug_hook=None,
    ):
        self.calls.append(
            {
                "kspace": kspace.copy(),
                "mask": mask.copy(),
                "conditioning": conditioning,
                "debug_hook": debug_hook,
            }
        )
        # [T,S,C,H,W] -> a deterministic real [T,S,H,W] test result.
        stage_m = np.abs(kspace[:, :, 0]).astype(np.float32)
        return stage_m, {
            "runtime_seconds": 0.01,
            "model_eval_mode": True,
            "inference_context": "torch.no_grad",
            "autocast": "CUDA bfloat16",
            "preprocessing": {
                "thread_count_restored": True,
                "torch_num_threads": {"before": 8, "during": 1, "after": 8},
            },
        }


class FakeTorchThreads:
    def __init__(self, count: int = 8) -> None:
        self.count = count

    def get_num_threads(self) -> int:
        return self.count

    def set_num_threads(self, value: int) -> None:
        self.count = int(value)


def make_input(
    *,
    slices=2,
    phases=6,
    height=3,
    width=4,
    coils=2,
    smaps=False,
    sampling_method="kt_radial",
    acceleration=16,
    ncalib=None,
    implementation="sdum",
):
    resolved_ncalib = min(height, width) if ncalib is None else int(ncalib)
    values = np.arange(
        slices * phases * height * width * coils, dtype=np.float32
    ).reshape(slices, phases, height, width, coils)
    kspace = (values + 1j * (values + 0.5)).astype(np.complex64)
    mask = np.ones((slices, phases, height, width), dtype=np.float32)
    sensitivity_maps = (
        np.ones((slices, 1, height, width, coils), dtype=np.complex64)
        if smaps
        else None
    )
    sampling = SamplingResult(
        mask=mask,
        requested_acceleration=float(acceleration),
        actual_acceleration=calculate_actual_acceleration(mask),
        method=sampling_method,
        parameters={
            "family": sampling_method,
            "implementation": implementation,
            "ncalib": resolved_ncalib,
        },
    )
    return ReconstructionInput(
        subject_id="synthetic",
        kspace=kspace,
        mask=mask,
        sensitivity_maps=sensitivity_maps,
        sampling=sampling,
    )


def make_adapter(runtime: FakeSDUMRuntime, **conditioning) -> SDUMAdapter:
    values = {
        "mask_type": "ktRadial16",
        "acceleration": 16,
        "acquisition": "Cine",
    }
    values.update(conditioning)
    return SDUMAdapter(
        sdum_source_root="unused",
        checkpoint="unused",
        device="cuda",
        runtime_factory=lambda **_: runtime,
        **values,
    )


class SDUMAdapterTests(unittest.TestCase):
    def test_variant_metadata_roles_cascades_and_checkpoint_ids_are_exact(self):
        expected = {
            "small": (
                "engineering / adapter parity smoke",
                6,
                sdum_module.SMALL_CHECKPOINT_SHA256,
                False,
            ),
            "base": (
                "primary released scientific model",
                18,
                sdum_module.BASE_CHECKPOINT_SHA256,
                True,
            ),
            "large": (
                "released model under validation",
                34,
                sdum_module.LARGE_CHECKPOINT_SHA256,
                True,
            ),
        }
        for variant, (role, cascades, checkpoint_sha, label_conditioning) in expected.items():
            with self.subTest(variant=variant):
                spec = sdum_variant_spec(variant)
                self.assertEqual(spec.checkpoint_role, role)
                self.assertEqual(spec.cascade_count, cascades)
                self.assertEqual(spec.checkpoint_sha256, checkpoint_sha)
                self.assertEqual(spec.label_conditioning, label_conditioning)

                adapter = SDUMAdapter(
                    variant=variant,
                    sdum_source_root="unused",
                    checkpoint="unused",
                    mask_type="ktRadial16",
                    acceleration=16,
                    acquisition="Cine",
                    runtime_factory=lambda **_: FakeSDUMRuntime(),
                )
                adapter.load()
                info = adapter.model_info()
                self.assertEqual(info.metadata["variant"], variant)
                self.assertEqual(info.metadata["checkpoint_role"], role)
                self.assertEqual(info.metadata["cascade_count_expected"], cascades)
                self.assertEqual(info.metadata["label_conditioning"], label_conditioning)
                self.assertEqual(info.checkpoint_identifier, checkpoint_sha)

    def test_canonical_axis_conversion_roundtrip(self):
        value = make_input()
        private_kspace, private_mask = canonical_to_sdum_private(
            value.kspace, value.mask
        )
        self.assertEqual(private_kspace.shape, (6, 2, 2, 3, 4))
        self.assertEqual(private_mask.shape, (6, 2, 3, 4))
        restored_kspace = np.transpose(private_kspace, (1, 0, 3, 4, 2))
        restored_mask = np.transpose(private_mask, (1, 0, 2, 3))
        np.testing.assert_array_equal(restored_kspace, value.kspace)
        np.testing.assert_array_equal(restored_mask, value.mask)

    def test_mask_expansion_and_strict_positive_binarization(self):
        mask = np.array([[[[-2.0, 0.0, 0.1]]]], dtype=np.float32)
        support = strict_mask_support(mask)
        np.testing.assert_array_equal(support, [[[[False, False, True]]]])
        expanded = expand_private_mask(mask, coils=3)
        self.assertEqual(expanded.shape, (1, 1, 3, 1, 3, 2))
        for coil in range(3):
            np.testing.assert_array_equal(expanded[0, 0, coil, :, :, 0], support[0, 0])
            np.testing.assert_array_equal(expanded[0, 0, coil, :, :, 1], support[0, 0])

    def test_five_frame_cyclic_indices(self):
        self.assertEqual(cyclic_temporal_indices(0, 12), [10, 11, 0, 1, 2])
        self.assertEqual(cyclic_temporal_indices(11, 12), [9, 10, 11, 0, 1])
        self.assertEqual(cyclic_temporal_indices(0, 2), [0, 1, 0, 1, 0])

    def test_normalization_formula_has_complex_mean_population_sigma_and_no_epsilon(self):
        value = np.array([[[1 + 2j, 3 + 4j]], [[5 + 6j, 7 + 8j]]], dtype=np.complex64)
        normalized, mean, sigma = complex_zscore_reference(value)
        expected_mean = np.mean(value)
        expected_sigma = np.sqrt(np.mean(np.abs(value - expected_mean) ** 2))
        self.assertEqual(mean, complex(expected_mean))
        self.assertEqual(sigma, float(expected_sigma))
        np.testing.assert_allclose(normalized, (value - expected_mean) / expected_sigma)

    def test_thread_context_restores_original_on_success_and_exception(self):
        fake = FakeTorchThreads(8)
        with temporary_torch_threads(fake, 1) as evidence:
            self.assertEqual(fake.get_num_threads(), 1)
        self.assertEqual(fake.get_num_threads(), 8)
        self.assertEqual(evidence, {"before": 8, "during": 1, "after": 8})
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with temporary_torch_threads(fake, 1):
                raise RuntimeError("boom")
        self.assertEqual(fake.get_num_threads(), 8)

    def test_conditioning_preflight_separates_documented_from_executable(self):
        supported = preflight_conditioning(SDUMConditioning("ktRadial16", 16, "Cine"))
        self.assertEqual(supported["status"], "SUPPORTED")
        self.assertFalse(supported["silent_fallback"])
        supported_r8 = preflight_conditioning(SDUMConditioning("ktRadial8", 8, "Cine"))
        self.assertEqual(supported_r8["status"], "SUPPORTED")
        self.assertFalse(supported_r8["silent_fallback"])
        self.assertEqual(
            supported_r8["released_conditions"]["mask_families"],
            ["uniform", "kt_gaussian", "kt_radial"],
        )
        for condition in (
            SDUMConditioning("Uniform24", 24, "Cine"),
            SDUMConditioning("ktGaussian24", 24, "Cine"),
            SDUMConditioning("ktRadial24", 24, "Cine"),
        ):
            self.assertEqual(preflight_conditioning(condition)["status"], "SUPPORTED")
        self.assertEqual(
            preflight_conditioning(SDUMConditioning("ktUniform8", 8, "Cine"))["status"],
            "SUPPORTED",
        )
        mismatched = preflight_conditioning(SDUMConditioning("ktRadial16", 8, "Cine"))
        self.assertEqual(mismatched["status"], "UNDOCUMENTED")
        rejected = preflight_conditioning(SDUMConditioning("VISTA", 4, "Cine"))
        self.assertEqual(rejected["status"], "UNDOCUMENTED")
        self.assertFalse(rejected["silent_fallback"])
        legacy_raw = make_adapter(FakeSDUMRuntime(), mask_type="VISTA", acceleration=4)
        legacy_raw.load()
        with self.assertRaisesRegex(ValueError, "allow_unmatched=true"):
            legacy_raw.reconstruct(make_input())
        runtime_created = False

        def forbidden_runtime(**_parameters):
            nonlocal runtime_created
            runtime_created = True
            return FakeSDUMRuntime()

        invalid = SDUMAdapter(
            variant="small",
            sdum_source_root="unused",
            checkpoint="unused",
            mask_type="ktUniform8",
            acceleration=8,
            acquisition="Cine",
            runtime_factory=forbidden_runtime,
        )
        invalid.load()
        self.assertTrue(runtime_created)

    def test_output_shape_real_rss_metadata_and_external_smaps_ignored(self):
        runtime = FakeSDUMRuntime()
        value = make_input(smaps=True)
        result = make_adapter(runtime).reconstruct(value)
        self.assertEqual(result.reconstruction.shape, (2, 6, 3, 4))
        self.assertFalse(np.iscomplexobj(result.reconstruction))
        self.assertEqual(result.metadata["output_representation"], "real_rss_magnitude")
        self.assertTrue(result.metadata["external_smaps_present"])
        self.assertFalse(result.metadata["external_smaps_used"])
        self.assertEqual(result.metadata["model_sampling"]["cmrbench_family"], "kt_radial")
        self.assertFalse(
            result.metadata["model_sampling"]["measured_actual_used_as_conditioning"]
        )
        self.assertNotIn("sensitivity_maps", runtime.calls[0])
        expected = sdum_rss_to_canonical(
            np.abs(runtime.calls[0]["kspace"][:, :, 0]).astype(np.float32)
        )
        np.testing.assert_array_equal(result.reconstruction, expected)

    def test_short_temporal_input_emits_warning_and_records_repeated_index_risk(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = make_adapter(FakeSDUMRuntime()).reconstruct(
                make_input(phases=2)
            )
        self.assertEqual(len(caught), 1)
        self.assertIn("repeated cyclic indices", str(caught[0].message))
        self.assertIn("T=2", result.metadata["temporal_warning"])

    def test_static_multislice_volume_preserves_s_and_repeats_only_t(self):
        runtime = FakeSDUMRuntime()
        value = make_input(
            slices=3, phases=1, height=24, width=24,
            sampling_method="uniform", acceleration=8,
            ncalib=20, implementation="canonical",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = make_adapter(
                runtime, mask_type="Uniform8", acceleration=8
            ).reconstruct(value)
        self.assertEqual(runtime.calls[0]["kspace"].shape, (1, 3, 2, 24, 24))
        self.assertEqual(result.reconstruction.shape, (3, 1, 24, 24))
        self.assertEqual(result.metadata["model_sampling"]["resolved_acc_idx"], 0)
        self.assertEqual(result.metadata["model_sampling"]["compatibility_status"], "supported")
        self.assertTrue(any("T=1" in str(item.message) for item in caught))

    def test_registry_creates_adapter_without_importing_torch_or_monai(self):
        runtime = FakeSDUMRuntime()
        adapter = default_model_registry.create(
            "sdum",
            sdum_source_root="unused",
            checkpoint="unused",
            device="cuda",
            mask_type="ktRadial16",
            acceleration=16,
            acquisition="Cine",
            runtime_factory=lambda **_: runtime,
        )
        self.assertIsInstance(adapter, SDUMAdapter)
        self.assertEqual(adapter.model_info().model_name, "sdum")

    def test_actual_sampling_must_match_conditioning_before_runtime_load(self):
        positive = (
            ("kt_radial", "ktRadial8", 8),
            ("kt_gaussian", "ktGaussian16", 16),
            ("uniform", "Uniform24", 24),
        )
        for method, mask_type, acceleration in positive:
            with self.subTest(method=method):
                value = make_input(
                    sampling_method=method,
                    acceleration=acceleration,
                    height=24,
                    width=24,
                    ncalib=20,
                )
                record = sdum_sampling_compatibility(
                    value,
                    SDUMConditioning(mask_type, acceleration, "Cine"),
                    variant="base",
                )
                self.assertEqual(record["cmrbench_family"], method)
                self.assertEqual(record["conditioning_acceleration"], acceleration)
                self.assertEqual(record["sampling_implementation"], "sdum")
                self.assertEqual(record["compatibility_status"], "supported")
                self.assertEqual(record["runtime_compatibility"], "valid")
                self.assertEqual(record["checkpoint_training_acs_exposure"], "unknown")
                self.assertTrue(record["matches_documented_reference_ncalib"])

        cross_source = make_input(sampling_method="external_mask", acceleration=16)
        cross_source.sampling.parameters = {
            "family": "kt_radial",
            "implementation": "cardiomm",
            "source_identity": "pinned CardioMM ktRadial",
            "ncalib": 3,
        }
        cross_record = sdum_sampling_compatibility(
            cross_source,
            SDUMConditioning("ktRadial16", 16, "Cine"),
            variant="base",
        )
        self.assertEqual(cross_record["sampling_implementation"], "cardiomm")
        self.assertEqual(cross_record["compatibility_status"], "ood_valid")

        generated_cross_source = make_input(
            sampling_method="kt_gaussian", acceleration=24
        )
        generated_cross_source.sampling.parameters.update(
            {
                "implementation": "cardiomm",
                "generator": "CardioMM::ktGaussian",
            }
        )
        generated_record = sdum_sampling_compatibility(
            generated_cross_source,
            SDUMConditioning("ktGaussian24", 24, "Cine"),
            variant="small",
        )
        self.assertEqual(generated_record["compatibility_status"], "ood_valid")

    def test_nondefault_ncalib_is_runtime_valid_with_unknown_training_exposure(self):
        value = make_input(
            sampling_method="uniform",
            acceleration=8,
            height=24,
            width=26,
            ncalib=16,
        )
        record = sdum_sampling_compatibility(
            value,
            SDUMConditioning("Uniform8", 8, "Cine"),
            variant="small",
        )
        self.assertEqual(record["compatibility_status"], "supported")
        self.assertEqual(record["runtime_compatibility"], "valid")
        self.assertEqual(record["checkpoint_training_acs_exposure"], "unknown")
        self.assertEqual(record["ncalib"], 16)
        self.assertEqual(record["documented_reference_ncalib"], 20)
        self.assertFalse(record["matches_documented_reference_ncalib"])
        self.assertEqual(record["acs_geometry"], "centered_ky_strip")
        self.assertEqual(record["conditioning_acceleration"], 8)
        self.assertEqual(
            record["actual_acceleration"], value.sampling.actual_acceleration
        )
        self.assertFalse(record["measured_actual_used_as_conditioning"])

    def test_radial_square_and_nonradial_strip_acs_are_distinct(self):
        ncalib = 6
        value = make_input(
            sampling_method="kt_radial",
            acceleration=16,
            height=24,
            width=26,
            ncalib=ncalib,
        )
        value.mask[:] = False
        kx_start = (24 - ncalib) // 2
        ky_start = (26 - ncalib) // 2
        value.mask[:, :, kx_start : kx_start + ncalib, ky_start : ky_start + ncalib] = True
        value.kspace *= value.mask[..., None]
        value.sampling.mask = value.mask
        value.sampling.actual_acceleration = calculate_actual_acceleration(value.mask)
        radial = sdum_sampling_compatibility(
            value,
            SDUMConditioning("ktRadial16", 16, "Cine"),
            variant="base",
        )
        self.assertEqual(radial["acs_geometry"], "centered_square")

        value.sampling.method = "kt_gaussian"
        value.sampling.parameters["family"] = "kt_gaussian"
        shifted = sdum_sampling_compatibility(
            value,
            SDUMConditioning("ktGaussian16", 16, "Cine"),
            variant="base",
        )
        self.assertEqual(shifted["acs_geometry"], "centered_square")

    def test_sampling_conditioning_mismatches_are_explicitly_recorded(self):
        runtime = FakeSDUMRuntime()
        mismatch = make_adapter(runtime, mask_type="ktRadial16", acceleration=16)
        family_result = mismatch.reconstruct(
            make_input(sampling_method="kt_gaussian", acceleration=16)
        )
        self.assertEqual(
            family_result.metadata["model_sampling"]["conditioning_relation_to_physical_sampling"],
            "user_forced",
        )
        self.assertTrue(runtime.calls)

        runtime = FakeSDUMRuntime()
        mismatch = make_adapter(runtime, mask_type="ktRadial16", acceleration=16)
        acceleration_result = mismatch.reconstruct(
            make_input(sampling_method="kt_radial", acceleration=8)
        )
        self.assertEqual(
            acceleration_result.metadata["model_sampling"]["model_conditioning_acceleration"],
            16,
        )
        self.assertTrue(runtime.calls)

    def test_source_commit_and_checkpoint_validation_fail_closed(self):
        with mock.patch.object(
            sdum_module.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(stdout="wrong\n"),
                SimpleNamespace(stdout=""),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "pinned commit"):
                sdum_module.verify_sdum_source(Path("."))
        with self.assertRaisesRegex(RuntimeError, "checkpoint does not exist"):
            sdum_module.verify_small_checkpoint(Path("missing-checkpoint.pt"))

    def test_production_source_never_imports_common_cmrbench_fft(self):
        source = Path(sdum_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("cmrbench.transforms.fft", source)
        self.assertNotIn("from cmrbench.transforms", source)
        self.assertIn("common_cmrbench_fft_used", source)
        self.assertIn("released MONAI centered orthonormal IFFT", source)

    def test_canonical_rank_and_nonfinite_errors_are_actionable(self):
        value = make_input()
        value.kspace = value.kspace.real
        with self.assertRaisesRegex(ValueError, "complex canonical k-space"):
            make_adapter(FakeSDUMRuntime()).reconstruct(value)
        nonfinite = make_input()
        nonfinite.mask[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "mask contains NaN or Inf"):
            make_adapter(FakeSDUMRuntime()).reconstruct(nonfinite)


if __name__ == "__main__":
    unittest.main()
