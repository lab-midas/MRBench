from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from cmrbench.models.adapters.cardiomm import (
    INFERENCE_BATCH_SIZE,
    INFERENCE_CONTEXT_DEFAULT,
    NUM_CASCADES,
    SOURCE_IDENTITY,
    TEXT_CONDITIONING_CACHE_DEFAULT,
    UNDERSAMPLING_PROMPT,
    CardioMMAdapter,
    build_metadata_prompt,
    build_undersampling_prompt,
    cardiomm_sampling_compatibility,
    canonical_kspace_batch_to_cardiomm,
    canonical_kspace_frame_to_cardiomm,
    canonical_mask_batch_to_cardiomm,
    canonical_mask_frame_to_cardiomm,
    conditioning_cache_key,
)
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.registry import default_model_registry
from cmrbench.sampling.nvidia import UniformSamplingProtocol
from cmrbench.sampling.nvidia import CanonicalUniformSamplingProtocol
from cmrbench.sampling.cardiomm import CardioMMSamplingProtocol
from cmrbench.sampling.result import SamplingResult, calculate_actual_acceleration


VALIDATED_METADATA = {
    "lifespan": "Adult",
    "field_strength": "1.5T",
    "vendor": "Siemens",
    "scanner": "Aera",
    "modality": "cine",
    "view": "short-axis",
}
METADATA_PROMPT = (
    "Adult cardiac MRI reconstruction.Vendor: 1.5T, Siemens, Aera. "
    "Modality: cine, short-axis."
)


class FakeRuntime:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.load_calls = 0
        self.calls = []
        self.conditioning_keys = set()
        FakeRuntime.instances.append(self)

    def load(self):
        self.load_calls += 1
        return {
            "source_commit": "abb6f8a4e293426736467c9859a5f5755c38c8aa",
            "checkpoint_loaded_once": True,
            "model_constructed_once": True,
            "clip_parameters_frozen": True,
            "clip_offline": True,
            "parameter_count": 123,
        }

    def infer_batch(
        self,
        kspace,
        mask,
        metadata_prompt,
        undersampling_prompt,
        expected_shape,
        *,
        mask_semantic="uniform",
        text_conditioning_cache=False,
        inference_context="no_grad",
    ):
        key = (metadata_prompt, undersampling_prompt)
        cache_hit = bool(text_conditioning_cache and key in self.conditioning_keys)
        if text_conditioning_cache:
            self.conditioning_keys.add(key)
        self.calls.append(
            (
                kspace.copy(),
                mask.copy(),
                metadata_prompt,
                undersampling_prompt,
                mask_semantic,
                text_conditioning_cache,
                inference_context,
            )
        )
        values = [float(kspace[index, 0, 0, 0, 0]) for index in range(kspace.shape[0])]
        output = np.stack(
            [np.full(expected_shape, value, dtype=np.float32) for value in values]
        )
        return output, {
            "internal_sensitivity_maps_finite": True,
            "internal_sensitivity_maps_shape": list(kspace.shape),
            "text_conditioning_recomputed": not cache_hit,
            "conditioning_cache_hit": cache_hit,
            "batch_size": int(kspace.shape[0]),
        }


def make_input(
    *,
    method="external_mask",
    source_identity=SOURCE_IDENTITY,
    acceleration=8,
    kx=4,
    ky=24,
    ncalib=20,
    implementation=None,
    pattern="Uniform",
):
    slices, phases, coils = 2, 3, 15
    mask = np.zeros((slices, phases, kx, ky), dtype=bool)
    mask[..., ::8] = True
    resolved_implementation = implementation or (
        "cardiomm" if method == "external_mask" else "sdum"
    )
    start = (
        ky // 2 - ncalib // 2
        if resolved_implementation == "cardiomm"
        else (ky - ncalib) // 2
    )
    mask[..., start : start + ncalib] = True
    kspace = np.zeros((slices, phases, kx, ky, coils), dtype=np.complex64)
    for phase in range(phases):
        for slice_index in range(slices):
            marker = phase * slices + slice_index + 1
            kspace[slice_index, phase] = (
                marker + 1j * (marker + 10)
            ) * mask[slice_index, phase, :, :, None]
    sampling = SamplingResult(
        mask=mask,
        requested_acceleration=float(acceleration),
        actual_acceleration=calculate_actual_acceleration(mask),
        method=method,
        parameters={
            "implementation": resolved_implementation,
            "source_identity": source_identity,
            "pattern": pattern,
            "ncalib": ncalib,
            "nominal_acceleration": acceleration,
        },
    )
    smaps = np.full((slices, 1, kx, ky, coils), 99 + 3j, dtype=np.complex64)
    return ReconstructionInput(
        subject_id="synthetic",
        kspace=kspace,
        mask=mask,
        sensitivity_maps=smaps,
        sampling=sampling,
    )


def make_adapter(**overrides):
    FakeRuntime.instances.clear()
    parameters = dict(
        source_root="/external/CardioMM",
        checkpoint="/checkpoints/epochbest.ckpt",
        clip_dir="/external/CardioMM/clip-vit-base-patch16",
        metadata=VALIDATED_METADATA,
        runtime_factory=FakeRuntime,
    )
    parameters.update(overrides)
    return CardioMMAdapter(**parameters)


class CardioMMConversionTests(unittest.TestCase):
    def test_exact_prompts(self):
        self.assertEqual(build_metadata_prompt(VALIDATED_METADATA), METADATA_PROMPT)
        self.assertEqual(
            METADATA_PROMPT,
            "Adult cardiac MRI reconstruction.Vendor: 1.5T, Siemens, Aera. Modality: cine, short-axis.",
        )
        self.assertEqual(build_undersampling_prompt(8, "uniform"), UNDERSAMPLING_PROMPT)
        self.assertEqual(UNDERSAMPLING_PROMPT, "Undersampling: 8x uniform.")
        self.assertEqual(
            build_undersampling_prompt(16, "random"),
            "Undersampling: 16x random.",
        )
        self.assertEqual(
            build_undersampling_prompt(24, "radial"),
            "Undersampling: 24x radial.",
        )
        with self.assertRaisesRegex(ValueError, "missing required acquisition metadata"):
            build_metadata_prompt({})

    def test_structured_metadata_is_required_but_not_dataset_hard_coded(self):
        for field in VALIDATED_METADATA:
            with self.subTest(field=field):
                incomplete = dict(VALIDATED_METADATA)
                incomplete.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    build_metadata_prompt(incomplete)
        other = {
            "lifespan": "Pediatric",
            "field_strength": "3.0T",
            "vendor": "GE",
            "scanner": "Premier",
            "modality": "cine",
            "view": "four-chamber",
        }
        self.assertEqual(
            build_metadata_prompt(other),
            "Pediatric cardiac MRI reconstruction.Vendor: 3.0T, GE, Premier. "
            "Modality: cine, four-chamber.",
        )

    def test_kspace_layout_preserves_spatial_axes_real_imag_and_coils(self):
        frame = np.empty((3, 5, 15), dtype=np.complex64)
        for x in range(3):
            for y in range(5):
                for coil in range(15):
                    frame[x, y, coil] = (100 * x + 10 * y + coil) + 1j * (
                        1000 + 100 * x + 10 * y + coil
                    )
        converted = canonical_kspace_frame_to_cardiomm(frame)
        self.assertEqual(converted.shape, (1, 15, 3, 5, 2))
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted[0, :, :, :, 0], frame.real.transpose(2, 0, 1))
        np.testing.assert_array_equal(converted[0, :, :, :, 1], frame.imag.transpose(2, 0, 1))

    def test_mask_layout_and_acquired_semantics(self):
        mask = np.zeros((3, 5), dtype=np.uint8)
        mask[:, 2] = 1
        converted = canonical_mask_frame_to_cardiomm(mask)
        self.assertEqual(converted.shape, (1, 1, 3, 5, 1))
        self.assertEqual(converted.dtype, np.bool_)
        np.testing.assert_array_equal(converted[0, 0, :, :, 0], mask.astype(bool))

    def test_batch_layout_preserves_items_spatial_axes_and_15_coils(self):
        frames = np.empty((4, 3, 5, 15), dtype=np.complex64)
        for item in range(4):
            frames[item] = item + 1 + 1j * (item + 11)
        converted = canonical_kspace_batch_to_cardiomm(frames)
        self.assertEqual(converted.shape, (4, 15, 3, 5, 2))
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted[:, :, :, :, 0], frames.real.transpose(0, 3, 1, 2))
        np.testing.assert_array_equal(converted[:, :, :, :, 1], frames.imag.transpose(0, 3, 1, 2))
        masks = np.ones((4, 3, 5), dtype=np.uint8)
        converted_masks = canonical_mask_batch_to_cardiomm(masks)
        self.assertEqual(converted_masks.shape, (4, 1, 3, 5, 1))
        self.assertEqual(converted_masks.dtype, np.bool_)

    def test_conditioning_cache_key_covers_prompts_device_dtype_and_models(self):
        base = conditioning_cache_key("meta", "us", "cuda:0", "float32", (1, 2, 3))
        variants = (
            conditioning_cache_key("other", "us", "cuda:0", "float32", (1, 2, 3)),
            conditioning_cache_key("meta", "other", "cuda:0", "float32", (1, 2, 3)),
            conditioning_cache_key("meta", "us", "cuda:1", "float32", (1, 2, 3)),
            conditioning_cache_key("meta", "us", "cuda:0", "float64", (1, 2, 3)),
            conditioning_cache_key("meta", "us", "cuda:0", "float32", (4, 2, 3)),
        )
        self.assertTrue(all(value != base for value in variants))


class CardioMMAdapterTests(unittest.TestCase):
    def test_production_defaults_and_composed_model_config_are_frozen(self):
        self.assertEqual(INFERENCE_BATCH_SIZE, 1)
        self.assertEqual(NUM_CASCADES, 10)
        self.assertTrue(TEXT_CONDITIONING_CACHE_DEFAULT)
        self.assertEqual(INFERENCE_CONTEXT_DEFAULT, "no_grad")
        adapter = make_adapter()
        self.assertEqual(adapter.inference_batch_size, 1)
        self.assertTrue(adapter.text_conditioning_cache)
        self.assertEqual(adapter.inference_context, "no_grad")

        root = Path(__file__).resolve().parents[1]
        model_config = (root / "configs/models/cardiomm.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: cardiomm", model_config)
        self.assertIn("inference_batch_size: 1", model_config)
        self.assertIn("text_conditioning_cache: true", model_config)
        self.assertIn("inference_context: no_grad", model_config)

    def test_complete_cine_frame_order_output_and_no_external_smaps(self):
        adapter = make_adapter()
        result = adapter.reconstruct(make_input())
        runtime = FakeRuntime.instances[0]
        self.assertEqual(runtime.load_calls, 1)
        self.assertEqual(len(runtime.calls), 6)
        markers = [int(call[0][0, 0, 0, 0, 0]) for call in runtime.calls]
        self.assertEqual(markers, [1, 2, 3, 4, 5, 6])
        self.assertEqual(result.reconstruction.shape, (2, 3, 4, 24))
        self.assertEqual(result.reconstruction.dtype, np.float32)
        self.assertTrue(np.all(result.reconstruction >= 0))
        self.assertFalse(np.iscomplexobj(result.reconstruction))
        self.assertEqual(result.metadata["input_coils"], 15)
        self.assertFalse(result.metadata["coil_compression"])
        self.assertTrue(result.metadata["external_smaps_present"])
        self.assertFalse(result.metadata["external_smaps_used"])
        self.assertEqual(result.metadata["nominal_acceleration"], 8)
        self.assertNotEqual(result.metadata["measured_actual_acceleration"], 8)
        self.assertEqual(result.metadata["undersampling_prompt"], UNDERSAMPLING_PROMPT)
        self.assertEqual(result.metadata["model_sampling"]["semantic_family"], "uniform")
        self.assertEqual(result.metadata["model_sampling"]["compatibility_status"], "supported")
        self.assertEqual(result.metadata["unique_text_conditioning_inputs"], 1)
        self.assertEqual(result.metadata["unique_text_conditioning_evaluations"], 1)
        self.assertEqual(result.metadata["text_conditioning_cache_hits_this_run"], 5)
        self.assertEqual(result.metadata["batching_validation_status"], "validated_production_b1")
        self.assertEqual(
            result.metadata["internal_sensitivity_estimation"],
            "internally_estimated_per_frame",
        )
        self.assertTrue(result.metadata["checkpoint_loaded_once"])
        self.assertTrue(result.metadata["model_constructed_once"])
        self.assertTrue(result.metadata["clip_parameters_frozen"])
        self.assertTrue(result.metadata["clip_offline"])
        self.assertEqual(
            [record["time_major_frame_index"] for record in result.metadata["frame_inference"]],
            list(range(6)),
        )
        self.assertNotIn("checkpoint_sha256", str(result.metadata).lower())
        self.assertEqual(adapter.model_info().checkpoint, "epochbest.ckpt")
        self.assertFalse(adapter.model_info().metadata["checkpoint_full_hash_serialized"])

    def test_static_multislice_subject_reassembles_every_slice(self):
        value = make_input()
        value.kspace = value.kspace[:, :1]
        value.mask = value.mask[:, :1]
        value.sampling = SamplingResult(
            mask=value.mask,
            requested_acceleration=8.0,
            actual_acceleration=calculate_actual_acceleration(value.mask),
            method="external_mask",
            parameters={
                "implementation": "cardiomm", "source_identity": SOURCE_IDENTITY,
                "pattern": "Uniform", "ncalib": 20, "nominal_acceleration": 8,
            },
        )
        result = make_adapter().reconstruct(value)
        self.assertEqual(result.reconstruction.shape, (2, 1, 4, 24))
        self.assertEqual(len(FakeRuntime.instances[0].calls), 2)
        self.assertEqual(
            [(record["slice"], record["phase"]) for record in result.metadata["frame_inference"]],
            [(0, 0), (1, 0)],
        )

    def test_b1_b2_b4_preserve_time_major_order_and_canonical_reassembly(self):
        expected = np.empty((2, 3, 4, 24), dtype=np.float32)
        for phase in range(3):
            for slice_index in range(2):
                expected[slice_index, phase] = phase * 2 + slice_index + 1
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                adapter = make_adapter(
                    inference_batch_size=batch_size,
                    text_conditioning_cache=True,
                )
                result = adapter.reconstruct(make_input())
                runtime = FakeRuntime.instances[0]
                markers = [
                    int(call[0][item, 0, 0, 0, 0])
                    for call in runtime.calls
                    for item in range(call[0].shape[0])
                ]
                self.assertEqual(markers, [1, 2, 3, 4, 5, 6])
                np.testing.assert_array_equal(result.reconstruction, expected)
                self.assertEqual(result.metadata["inference_batch_size"], batch_size)
                self.assertTrue(result.metadata["text_conditioning_cached"])
                self.assertTrue(result.metadata["framewise_independent_inference"])
                self.assertEqual(
                    [record["time_major_frame_index"] for record in result.metadata["frame_inference"]],
                    list(range(6)),
                )
                self.assertTrue(all(call[0].shape[1] == 15 for call in runtime.calls))
                self.assertTrue(all(call[1].shape[1:] == (1, 4, 24, 1) for call in runtime.calls))
                self.assertTrue(
                    all(
                        record["internal_sensitivity_maps_shape"][0] == record["frames"]
                        for record in result.metadata["batch_inference"]
                    )
                )
                expected_status = (
                    "validated_production_b1"
                    if batch_size == 1
                    else "experimental_non_frozen_b_greater_than_1"
                )
                self.assertEqual(result.metadata["batching_validation_status"], expected_status)

    def test_cached_and_uncached_fake_runtime_are_numerically_identical(self):
        uncached = make_adapter(inference_batch_size=1, text_conditioning_cache=False)
        uncached_result = uncached.reconstruct(make_input()).reconstruction.copy()
        cached = make_adapter(inference_batch_size=1, text_conditioning_cache=True)
        cached_result = cached.reconstruct(make_input()).reconstruction
        np.testing.assert_array_equal(cached_result, uncached_result)

    def test_cache_reuses_same_prompts_and_invalidates_changed_prompts(self):
        adapter = make_adapter()
        first = adapter.reconstruct(make_input())
        second = adapter.reconstruct(make_input())
        self.assertEqual(first.metadata["unique_text_conditioning_evaluations"], 1)
        self.assertEqual(second.metadata["unique_text_conditioning_evaluations"], 0)
        self.assertEqual(second.metadata["text_conditioning_cache_hits_this_run"], 6)

        adapter.metadata_prompt = build_metadata_prompt(
            {**VALIDATED_METADATA, "scanner": "Vida"}
        )
        changed_metadata = adapter.reconstruct(make_input())
        self.assertEqual(changed_metadata.metadata["unique_text_conditioning_evaluations"], 1)

        with mock.patch(
            "cmrbench.models.adapters.cardiomm.build_undersampling_prompt",
            return_value="Undersampling: 8x uniform changed-for-cache-test.",
        ):
            changed_sampling_text = adapter.reconstruct(make_input())
        self.assertEqual(
            changed_sampling_text.metadata["unique_text_conditioning_evaluations"], 1
        )

        other_adapter = make_adapter()
        isolated = other_adapter.reconstruct(make_input())
        self.assertEqual(isolated.metadata["unique_text_conditioning_evaluations"], 1)

    def test_cache_tensors_are_not_serialized_in_result_metadata(self):
        result = make_adapter().reconstruct(make_input())

        def assert_plain(value):
            self.assertNotIsInstance(value, np.ndarray)
            if isinstance(value, dict):
                for nested in value.values():
                    assert_plain(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    assert_plain(nested)

        assert_plain(result.metadata)
        self.assertNotIn("conditioning_tensor", str(result.metadata).lower())

    def test_b1_legacy_runtime_fallback_remains_available(self):
        class LegacyRuntime(FakeRuntime):
            infer_batch = None

            def infer_frame(
                self, kspace, mask, metadata_prompt, undersampling_prompt, expected_shape
            ):
                self.calls.append((kspace.copy(), mask.copy(), metadata_prompt, undersampling_prompt))
                value = float(kspace[0, 0, 0, 0, 0])
                return np.full(expected_shape, value, dtype=np.float32), {}

        class LegacyAdapter(CardioMMAdapter):
            pass

        FakeRuntime.instances.clear()
        adapter = LegacyAdapter(
            source_root="/external/CardioMM",
            checkpoint="/checkpoints/epochbest.ckpt",
            clip_dir="/external/CardioMM/clip-vit-base-patch16",
            metadata=VALIDATED_METADATA,
            text_conditioning_cache=False,
            runtime_factory=LegacyRuntime,
        )
        result = adapter.reconstruct(make_input())
        self.assertEqual(result.reconstruction.shape, (2, 3, 4, 24))

    def test_load_is_idempotent_and_constructs_one_runtime(self):
        adapter = make_adapter()
        adapter.load()
        adapter.load()
        adapter.reconstruct(make_input())
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(FakeRuntime.instances[0].load_calls, 1)

    def test_generated_sampling_is_accepted_and_truthfully_conditioned(self):
        for method, semantic, source_pattern, status in (
            ("uniform", "uniform", "Uniform", "ood_valid"),
            ("kt_uniform", "uniform", "ktUniform", "ood_valid"),
            ("kt_gaussian", "random", "ktGaussian", "ood_valid"),
        ):
            with self.subTest(method=method):
                adapter = make_adapter()
                value = make_input(method=method)
                result = adapter.reconstruct(value)
                model_sampling = result.metadata["model_sampling"]
                self.assertEqual(model_sampling["semantic_family"], semantic)
                self.assertEqual(model_sampling["source_pattern"], source_pattern)
                self.assertEqual(model_sampling["compatibility_status"], status)
                self.assertEqual(model_sampling["sampling_implementation"], "sdum")
                self.assertEqual(result.reconstruction.shape, (2, 3, 4, 24))
                runtime = FakeRuntime.instances[0]
                self.assertTrue(all(call[4] == semantic for call in runtime.calls))
                np.testing.assert_array_equal(
                    runtime.calls[0][1][0, 0, :, :, 0], value.mask[0, 0]
                )

    def test_cmrbench_uniform_generator_mask_reaches_runtime_unchanged_with_full_t(self):
        value = make_input(method="uniform")
        generated = UniformSamplingProtocol(ncalib=20).generate(
            value.kspace.shape, acceleration=8, seed=1
        )
        value.mask = generated.mask
        value.sampling = generated
        value.kspace = value.kspace * generated.mask[..., None]
        result = make_adapter().reconstruct(value)
        runtime = FakeRuntime.instances[0]
        received = np.stack([call[1][0, 0, :, :, 0] for call in runtime.calls])
        expected = np.stack(
            [generated.mask[slice_index, phase_index] for phase_index in range(3) for slice_index in range(2)]
        )
        np.testing.assert_array_equal(received, expected)
        self.assertEqual(result.reconstruction.shape, (2, 3, 4, 24))
        self.assertEqual(len(result.metadata["frame_inference"]), 6)

    def test_canonical_uniform_and_native_cardiomm_generator_reach_adapter(self):
        value = make_input(method="uniform")
        canonical = CanonicalUniformSamplingProtocol().generate(
            value.kspace.shape, acceleration=8, seed=1
        )
        value.mask = canonical.mask
        value.sampling = canonical
        value.kspace = value.kspace * canonical.mask[..., None]
        canonical_record = cardiomm_sampling_compatibility(value)
        self.assertEqual(canonical_record["compatibility_status"], "supported")
        self.assertEqual(canonical_record["sampling_implementation"], "canonical")

        value = make_input(method="kt_gaussian")
        native = CardioMMSamplingProtocol(
            family="kt_gaussian", ncalib=20
        ).generate(value.kspace.shape, acceleration=8, seed=1)
        value.mask = native.mask
        value.sampling = native
        value.kspace = value.kspace * native.mask[..., None]
        result = make_adapter().reconstruct(value)
        model_sampling = result.metadata["model_sampling"]
        self.assertEqual(model_sampling["sampling_implementation"], "cardiomm")
        self.assertEqual(model_sampling["compatibility_status"], "supported")
        self.assertEqual(model_sampling["conditioning_label"], "Undersampling: 8x random.")
        received = np.stack(
            [call[1][0, 0, :, :, 0] for call in FakeRuntime.instances[0].calls]
        )
        expected = np.stack(
            [
                native.mask[slice_index, phase_index]
                for phase_index in range(3)
                for slice_index in range(2)
            ]
        )
        np.testing.assert_array_equal(received, expected)

    def test_nondefault_ncalib_is_truthful_ood_not_hard_rejected(self):
        value = make_input(
            method="uniform",
            kx=24,
            ky=32,
            ncalib=16,
            implementation="canonical",
        )
        record = cardiomm_sampling_compatibility(value)
        self.assertEqual(record["compatibility_status"], "ood_valid")
        self.assertEqual(record["ncalib"], 16)
        self.assertEqual(record["released_training_ncalib"], 20)
        self.assertEqual(record["model_internal_num_low_frequencies"], 16)
        self.assertEqual(record["acs_geometry"], "centered_ky_strip")
        self.assertEqual(record["nominal_acceleration"], 8)
        self.assertEqual(record["conditioning_label"], "Undersampling: 8x uniform.")

    def test_radial_uses_square_acs_while_nonradial_requires_full_kx_strip(self):
        ncalib = 6
        value = make_input(
            method="kt_radial",
            kx=24,
            ky=26,
            ncalib=ncalib,
            implementation="cardiomm",
            pattern="ktRadial",
        )
        value.mask[:] = False
        kx_start = 24 // 2 - ncalib // 2
        ky_start = 26 // 2 - ncalib // 2
        value.mask[:, :, kx_start : kx_start + ncalib, ky_start : ky_start + ncalib] = True
        value.kspace *= value.mask[..., None]
        value.sampling.mask = value.mask
        value.sampling.actual_acceleration = calculate_actual_acceleration(value.mask)
        radial = cardiomm_sampling_compatibility(value)
        self.assertEqual(radial["acs_geometry"], "centered_square")
        self.assertEqual(radial["compatibility_status"], "ood_valid")

        value.sampling.method = "kt_gaussian"
        value.sampling.parameters["pattern"] = "ktGaussian"
        with self.assertRaisesRegex(ValueError, "centered_ky_strip"):
            cardiomm_sampling_compatibility(value)

    def test_ambiguous_sampling_requires_explicit_semantic_but_arbitrary_r_runs(self):
        for value, message in (
            (make_input(method="vista"), "no unambiguous automatic mask_type"),
            (make_input(method="random_cartesian"), "no unambiguous automatic mask_type"),
        ):
            with self.subTest(method=value.sampling.method):
                adapter = make_adapter()
                with self.assertRaisesRegex(ValueError, message):
                    adapter.reconstruct(value)
                self.assertEqual(FakeRuntime.instances, [])
        value = make_input(method="uniform", acceleration=12)
        result = make_adapter().reconstruct(value)
        self.assertEqual(
            result.metadata["model_sampling"]["conditioning_label"],
            "Undersampling: 12x uniform.",
        )
        self.assertFalse(
            result.metadata["model_sampling"]["documented_training_acceleration_match"]
        )

    def test_external_mask_requires_truthful_metadata_and_checks_measurable_claims(self):
        missing_identity = make_input(source_identity="")
        with self.assertRaisesRegex(ValueError, "source_identity"):
            cardiomm_sampling_compatibility(missing_identity)

        dynamic_uniform = make_input()
        dynamic_uniform.mask[:, 1, 0, 0] = ~dynamic_uniform.mask[:, 1, 0, 0]
        dynamic_uniform.kspace[:, 1, 0, 0] = 0
        dynamic_uniform.sampling.mask = dynamic_uniform.mask
        dynamic_uniform.sampling.actual_acceleration = calculate_actual_acceleration(
            dynamic_uniform.mask
        )
        dynamic_record = cardiomm_sampling_compatibility(dynamic_uniform)
        self.assertEqual(dynamic_record["physical_sampling_validity"], "valid")

        radial_claim = make_input(kx=24)
        radial_claim.sampling.parameters["pattern"] = "ktRadial"
        radial_claim.mask[:, :, 12, 12] = False
        radial_claim.kspace[:, :, 12, 12] = 0
        radial_claim.sampling.mask = radial_claim.mask
        radial_claim.sampling.actual_acceleration = calculate_actual_acceleration(
            radial_claim.mask
        )
        with self.assertRaisesRegex(ValueError, "centered_square"):
            cardiomm_sampling_compatibility(radial_claim)

    def test_nonfinite_output_is_rejected(self):
        class NonFiniteRuntime(FakeRuntime):
            def infer_batch(
                self,
                kspace,
                mask,
                metadata_prompt,
                undersampling_prompt,
                expected_shape,
                **_kwargs
            ):
                return np.full((kspace.shape[0],) + expected_shape, np.nan, dtype=np.float32), {}

        adapter = CardioMMAdapter(
            source_root="/external/CardioMM",
            checkpoint="/checkpoints/epochbest.ckpt",
            clip_dir="/external/CardioMM/clip-vit-base-patch16",
            metadata=VALIDATED_METADATA,
            runtime_factory=NonFiniteRuntime,
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite or negative"):
            adapter.reconstruct(make_input())

    def test_default_registry_dispatches_lazily(self):
        adapter = default_model_registry.create(
            "cardiomm",
            source_root="/external/CardioMM",
            checkpoint="/checkpoints/epochbest.ckpt",
            clip_dir="/external/CardioMM/clip-vit-base-patch16",
            metadata=VALIDATED_METADATA,
            runtime_factory=FakeRuntime,
        )
        self.assertIsInstance(adapter, CardioMMAdapter)

    def test_source_parses_with_python38_grammar(self):
        path = Path(__file__).resolve().parents[1] / "cmrbench/models/adapters/cardiomm.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 8))
        self.assertNotIn("operator_parity", source)
        self.assertNotIn("outputs/side_tasks", source)
        self.assertNotIn("checkpoint_sha256", source)


if __name__ == "__main__":
    unittest.main()
