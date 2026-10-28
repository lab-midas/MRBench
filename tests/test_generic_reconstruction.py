from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from cmrbench.data.base import CineMRIDataset
from cmrbench.data.inhouse_cine import build_inhouse_cine_sample
from cmrbench.data.sample import CineMRISample
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import prepare_reconstruction_input
from cmrbench.models.adapters.zero_filled import ZeroFilledAdapter
from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.errors import UpstreamModelInferenceError
from cmrbench.models.registry import ModelRegistry
from cmrbench.models.result import ReconstructionResult, TransformRecord
from cmrbench.reconstruct import (
    create_cmr_dataset,
    create_reconstruction_model,
    create_sampling_protocol,
    main as reconstruct_main,
    run_reconstruction,
    with_resolved_sampling_implementation,
)
from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.registry import SamplingRegistry
from cmrbench.sampling.result import SamplingResult
from cmrbench.sampling.vista import VISTASamplingProtocol


def make_config(
    *,
    model_name: str = "fake",
    model_parameters: dict[str, object] | None = None,
    output_root: str = "outputs/test",
) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "experiment": {"name": "generic-test", "seed": 1},
            "dataset": {
                "name": "inhouse_cine",
                "manifest": "cohort.csv",
                "split": "development",
            },
            "sampling": {
                "mode": "retrospective",
                "method": "fake_sampling",
                "acceleration": 2,
                "seed": 7,
            },
            "model": {
                "name": model_name,
                "parameters": model_parameters or {},
            },
            "output": {"root": output_root, "save_qc": False},
        }
    )


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


class OneSampleDataset(CineMRIDataset):
    def __init__(self, sample: CineMRISample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> CineMRISample:
        if index != 0:
            raise IndexError(index)
        return self.sample


class FakeSampling(SamplingProtocol):
    def generate(self, shape, acceleration, seed=None, **kwargs):
        mask = np.ones(shape[:4], dtype=bool)
        return SamplingResult(
            mask=mask,
            requested_acceleration=acceleration,
            actual_acceleration=1.0,
            method="fake_sampling",
            seed=seed,
            parameters={"implementation": "fake", "preserved": True},
        )


class FakeModel(ReconstructionModel):
    def __init__(self, **parameters) -> None:
        self.parameters = parameters
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def reconstruct(self, reconstruction_input) -> ReconstructionResult:
        if not self.loaded:
            raise RuntimeError("not loaded")
        return ReconstructionResult(
            subject_id=reconstruction_input.subject_id,
            reconstruction=np.asarray(reconstruction_input.kspace)[..., 0],
            model_name="fake",
            runtime_seconds=0.25,
            transforms=[TransformRecord("unchanged", {"canonical": True})],
            metadata={
                "implementation": self.parameters.get("implementation"),
                "variant": self.parameters.get("variant"),
                "adapter_provenance": {"exact": "retained"},
                "model_sampling": {
                    "physical_sampling_validity": "valid",
                    "model_conditioning_mode": "auto",
                    "model_conditioning_family": "fake_semantic",
                },
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="fake",
            model_version="1",
            metadata={"parameters": self.parameters},
        )


class GenericReconstructionTests(unittest.TestCase):
    def test_upstream_inference_error_writes_classified_failure_sidecar(self) -> None:
        class FailingModel(FakeModel):
            def reconstruct(self, reconstruction_input):
                raise UpstreamModelInferenceError(
                    "synthetic_upstream", AssertionError("OOD condition rejected")
                )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "run"
            config = make_config(output_root=str(output_root))
            sample = CineMRISample(
                subject_id="Example0132",
                kspace=np.ones((1, 2, 3, 4, 1), dtype=np.complex64),
                reference=np.ones((1, 2, 3, 4), dtype=np.complex64),
                sensitivity_maps=np.ones((1, 1, 3, 4, 1), dtype=np.complex64),
            )
            models = ModelRegistry()
            models.register("fake", FailingModel)
            samplings = SamplingRegistry()
            samplings.register("fake_sampling", FakeSampling)
            with self.assertRaises(UpstreamModelInferenceError):
                run_reconstruction(
                    config,
                    dataset_factory=lambda _config: OneSampleDataset(sample),
                    model_registry=models,
                    sampling_registry=samplings,
                )
            failure = json.loads(
                (output_root / "reconstruction_failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["stage"], "model_inference")
            self.assertEqual(failure["source"], "upstream_model_runtime")
            self.assertEqual(failure["exception"]["type"], "AssertionError")
            self.assertEqual(
                failure["exception"]["message"], "OOD condition rejected"
            )

    def test_model_factory_forwards_explicit_ram_and_sdum_choices(self) -> None:
        registry = ModelRegistry()
        calls: list[dict[str, object]] = []

        def factory(**parameters):
            calls.append(parameters)
            return FakeModel(**parameters)

        registry.register("ram", factory)
        registry.register("sdum", factory)
        choices = (
            ("ram", {"implementation": "official_github"}),
            ("ram", {"implementation": "deepinv"}),
            ("sdum", {"variant": "small"}),
            ("sdum", {"variant": "base"}),
            ("sdum", {"variant": "large"}),
        )
        for name, parameters in choices:
            create_reconstruction_model(
                make_config(model_name=name, model_parameters=parameters), registry
            )
        self.assertEqual(calls, [parameters for _name, parameters in choices])

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown model"):
            create_reconstruction_model(make_config(model_name="not_registered"), ModelRegistry())

    def test_invalid_ram_implementation_is_rejected_by_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "implementation"):
            create_reconstruction_model(
                make_config(
                    model_name="ram",
                    model_parameters={"implementation": "invented"},
                )
            )

    def test_invalid_sdum_variant_is_rejected_by_adapter(self) -> None:
        with self.assertRaisesRegex(ValueError, "variant"):
            create_reconstruction_model(
                make_config(
                    model_name="sdum",
                    model_parameters={
                        "variant": "invented",
                        "sdum_source_root": "/source",
                        "checkpoint": "/checkpoint.pt",
                        "config": "/config.json",
                        "mask_type": "ktRadial8",
                        "acceleration": 8,
                        "acquisition": "Cine",
                    },
                )
            )

    def test_registry_lazy_factory_imports_only_selected_adapter(self) -> None:
        import cmrbench.models.registry as registry_module

        registry = ModelRegistry()
        registry.register(
            "selected",
            registry_module._lazy_adapter_factory("selected.module", "Selected"),
        )
        registry.register(
            "unrelated",
            registry_module._lazy_adapter_factory("unrelated.module", "Unrelated"),
        )
        with mock.patch.object(
            registry_module,
            "import_module",
            return_value=SimpleNamespace(Selected=FakeModel),
        ) as importer:
            created = registry.create("selected", marker=1)
        self.assertIsInstance(created, FakeModel)
        importer.assert_called_once_with("selected.module")

    def test_slice_or_phase_config_is_rejected_before_dataset_loading(self) -> None:
        for selector in ("slice", "phase", "slice_index", "phase_index"):
            raw = make_config().as_dict()
            raw["dataset"]["parameters"] = {selector: 1}
            config = ExperimentConfig.from_mapping(raw)
            with self.subTest(selector=selector):
                with self.assertRaisesRegex(ValueError, "complete canonical S/T"):
                    create_cmr_dataset(config)

    def test_generic_run_preserves_outputs_and_provenance_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "run"
            config = make_config(
                model_parameters={"implementation": "deepinv", "variant": "base"},
                output_root=str(output_root),
            )
            sample = CineMRISample(
                subject_id="Example0132",
                kspace=np.ones((2, 25, 3, 4, 2), dtype=np.complex64),
                reference=np.ones((2, 25, 3, 4), dtype=np.complex64),
                sensitivity_maps=np.ones((2, 1, 3, 4, 2), dtype=np.complex64),
            )
            models = ModelRegistry()
            models.register("fake", FakeModel)
            samplings = SamplingRegistry()
            samplings.register("fake_sampling", FakeSampling)
            with mock.patch(
                "cmrbench.experiment.runner._write_config",
                side_effect=lambda path, value: path.write_text(
                    json.dumps(value), encoding="utf-8"
                ),
            ):
                results = run_reconstruction(
                    config,
                    dataset_factory=lambda _config: OneSampleDataset(sample),
                    model_registry=models,
                    sampling_registry=samplings,
                )

            self.assertEqual(results[0].reconstruction.shape, (2, 25, 3, 4))
            saved = np.load(
                output_root / "subjects" / "Example0132_reconstruction.npy",
                allow_pickle=False,
            )
            self.assertTrue(np.array_equal(saved, results[0].reconstruction))
            metadata = json.loads(
                (output_root / "subjects" / "Example0132_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["metadata"]["adapter_provenance"], {"exact": "retained"})
            self.assertEqual(metadata["model"]["model_version"], "1")
            with (output_root / "results.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["implementation"], "deepinv")
            self.assertEqual(row["variant"], "base")
            self.assertEqual(row["sampling_implementation"], "fake")
            serialized_condition = json.loads(row["model_sampling"])
            self.assertEqual(serialized_condition["model_conditioning_mode"], "auto")
            self.assertEqual(
                metadata["sampling"]["sampling_implementation"], "fake"
            )
            experiment_metadata = json.loads(
                (output_root / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                experiment_metadata["resolved_sampling_implementations"], ["fake"]
            )
            self.assertEqual(
                experiment_metadata["model_sampling_conditions"][0][
                    "model_conditioning_family"
                ],
                "fake_semantic",
            )
            self.assertNotIn("nrmse", {key.lower() for key in row})

    def test_selecting_ram_or_sdum_does_not_reduce_complete_time(self) -> None:
        sample = CineMRISample(
            subject_id="Example0132",
            kspace=np.ones((2, 25, 3, 4, 2), dtype=np.complex64),
            reference=np.ones((2, 25, 3, 4), dtype=np.complex64),
            sensitivity_maps=np.ones((2, 1, 3, 4, 2), dtype=np.complex64),
        )
        seen_shapes: list[tuple[int, ...]] = []

        class CapturingModel(FakeModel):
            def reconstruct(self, reconstruction_input):
                seen_shapes.append(tuple(reconstruction_input.kspace.shape))
                return super().reconstruct(reconstruction_input)

        for model_name in ("ram", "sdum"):
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as directory:
                config = make_config(
                    model_name=model_name,
                    output_root=str(Path(directory) / "run"),
                )
                models = ModelRegistry()
                models.register(model_name, CapturingModel)
                samplings = SamplingRegistry()
                samplings.register("fake_sampling", FakeSampling)
                with mock.patch(
                    "cmrbench.experiment.runner._write_config",
                    side_effect=lambda path, value: path.write_text(
                        json.dumps(value), encoding="utf-8"
                    ),
                ):
                    run_reconstruction(
                        config,
                        dataset_factory=lambda _config: OneSampleDataset(sample),
                        model_registry=models,
                        sampling_registry=samplings,
                    )
        self.assertEqual(seen_shapes, [(2, 25, 3, 4, 2), (2, 25, 3, 4, 2)])

    def test_sub0132_vista_r4_uses_ky156_and_complete_t25(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_root = Path(directory)
            mask_file = mask_root / "mask_VISTA_156x25_acc4_1.txt"
            compact = np.zeros((156, 25), dtype=np.uint8)
            compact[::4, :] = 1
            np.savetxt(mask_file, compact, fmt="%d", delimiter=",")
            config = ExperimentConfig.from_mapping(
                {
                    "experiment": {"name": "full-cine-vista", "seed": 1},
                    "dataset": {
                        "name": "inhouse_cine",
                        "manifest": "cohort.csv",
                        "split": "development",
                        "parameters": {"subject": "Example0132"},
                    },
                    "sampling": {
                        "mode": "retrospective",
                        "method": "vista",
                        "acceleration": 4,
                        "seed": 1,
                        "mask_root": str(mask_root),
                    },
                    "model": {"name": "fake"},
                    "output": {"root": "outputs/test"},
                }
            )
            sample = CineMRISample(
                subject_id="Example0132",
                kspace=np.ones((2, 25, 4, 156, 2), dtype=np.complex64),
                reference=np.ones((2, 25, 4, 156), dtype=np.complex64),
                sensitivity_maps=np.ones((2, 1, 4, 156, 2), dtype=np.complex64),
            )
            prepared = prepare_reconstruction_input(
                sample,
                mode="retrospective",
                protocol=create_sampling_protocol(config),
                acceleration=4,
                seed=1,
            )
        self.assertEqual(prepared.kspace.shape, (2, 25, 4, 156, 2))
        self.assertEqual(prepared.mask.shape, (2, 25, 4, 156))
        self.assertEqual(sample.reference.shape, (2, 25, 4, 156))
        self.assertEqual(prepared.sensitivity_maps.shape, (2, 1, 4, 156, 2))
        self.assertEqual(prepared.sampling.parameters["Ky"], 156)
        self.assertEqual(prepared.sampling.parameters["T"], 25)
        self.assertTrue(
            prepared.sampling.parameters["mask_file"].endswith(
                "mask_VISTA_156x25_acc4_1.txt"
            )
        )

    def test_generic_vista_uses_default_root_after_subject_geometry_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_root = Path(directory)
            mask_file = mask_root / "mask_VISTA_156x25_acc20_1.txt"
            compact = np.zeros((156, 25), dtype=np.uint8)
            compact[::5, :] = 1
            np.savetxt(mask_file, compact, fmt="%d", delimiter=",")
            config = ExperimentConfig.from_mapping(
                {
                    "experiment": {"name": "default-vista-root", "seed": 1},
                    "dataset": {
                        "name": "inhouse_cine",
                        "manifest": "cohort.csv",
                        "split": "development",
                    },
                    "sampling": {
                        "mode": "retrospective",
                        "method": "vista",
                        "implementation": "precomputed",
                        "acceleration": 20,
                        "seed": 1,
                        "parameters": {"ncalib": 20},
                    },
                    "model": {"name": "fake"},
                    "output": {"root": "outputs/test"},
                }
            )
            sample = CineMRISample(
                subject_id="Example0132",
                kspace=np.ones((1, 25, 192, 156, 1), dtype=np.complex64),
            )
            with mock.patch(
                "cmrbench.sampling.vista.DEFAULT_VISTA_MASK_ROOT", mask_root
            ):
                prepared = prepare_reconstruction_input(
                    sample,
                    mode="retrospective",
                    protocol=create_sampling_protocol(config),
                    acceleration=20,
                    seed=1,
                )
        self.assertEqual(prepared.mask.shape, (1, 25, 192, 156))
        self.assertEqual(
            prepared.sampling.parameters["resolved_mask_filename"],
            "mask_VISTA_156x25_acc20_1.txt",
        )
        self.assertEqual(
            prepared.sampling.parameters["resolution_mode"], "default_root"
        )
        self.assertEqual(prepared.sampling.requested_acceleration, 20.0)

    def test_generic_vista_force_acs_uses_final_mask_without_changing_model_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_root = Path(directory)
            mask_file = mask_root / "mask_VISTA_24x3_acc8_1.txt"
            compact = np.zeros((24, 3), dtype=np.uint8)
            compact[[0, 23], :] = 1
            np.savetxt(mask_file, compact, fmt="%d", delimiter=",")
            sampling_condition = {
                "mode": "explicit",
                "family": "gaussian",
                "acceleration": 8,
            }
            config = ExperimentConfig.from_mapping(
                {
                    "experiment": {"name": "vista-acs", "seed": 1},
                    "dataset": {
                        "name": "inhouse_cine",
                        "manifest": "cohort.csv",
                        "split": "development",
                    },
                    "sampling": {
                        "mode": "retrospective",
                        "method": "vista",
                        "implementation": "precomputed",
                        "acceleration": 8,
                        "seed": 1,
                        "mask_root": str(mask_root),
                        "parameters": {"force_acs": True, "ncalib": 20},
                    },
                    "model": {
                        "name": "sdum",
                        "parameters": {"sampling_condition": sampling_condition},
                    },
                    "output": {"root": "outputs/test"},
                }
            )
            sensitivity_maps = np.ones((1, 1, 5, 24, 1), dtype=np.complex64)
            sample = CineMRISample(
                subject_id="Example0132",
                kspace=np.ones((1, 3, 5, 24, 1), dtype=np.complex64),
                sensitivity_maps=sensitivity_maps,
            )
            prepared = prepare_reconstruction_input(
                sample,
                mode="retrospective",
                protocol=create_sampling_protocol(config),
                acceleration=8,
                seed=1,
            )

        self.assertTrue(np.all(prepared.mask[:, :, :, 2:22]))
        self.assertTrue(np.all(prepared.mask[:, :, :, 0]))
        self.assertTrue(np.all(prepared.mask[:, :, :, 23]))
        np.testing.assert_array_equal(
            prepared.kspace, prepared.mask[..., None].astype(np.complex64)
        )
        np.testing.assert_array_equal(prepared.sensitivity_maps, sensitivity_maps)
        self.assertTrue(prepared.sampling.parameters["force_acs"])
        self.assertEqual(prepared.sampling.parameters["forced_acs_width"], 20)
        self.assertEqual(prepared.sampling.requested_acceleration, 8.0)
        self.assertEqual(
            config.model.parameters["sampling_condition"], sampling_condition
        )
        self.assertNotIn("epsilon", prepared.sampling.parameters)
        self.assertNotIn("external_smap", prepared.sampling.parameters)

    def test_generic_preparation_matches_validated_full_cine_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preprocess = root / "preprocess"
            masks = root / "masks"
            preprocess.mkdir()
            masks.mkdir()
            reference = (
                np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5, 1)
                + 1j
            ).astype(np.complex64)
            smaps = np.ones((2, 1, 4, 5, 2), dtype=np.complex64) / np.sqrt(2)
            np.save(preprocess / "norm_img_Example0132.txt.npy", reference)
            np.save(preprocess / "cc_smap_15_Example0132.txt.npy", smaps)
            manifest = root / "cohort.csv"
            manifest.write_text(
                "filename,split,cohort_type\nExample0132.h5,development,synthetic\n",
                encoding="utf-8",
            )
            compact = np.zeros((5, 3), dtype=np.uint8)
            compact[::2, :] = 1
            np.savetxt(
                masks / "mask_VISTA_5x3_acc2_1.txt",
                compact,
                fmt="%d",
                delimiter=",",
            )
            config = ExperimentConfig.from_mapping(
                {
                    "experiment": {"name": "parity", "seed": 1},
                    "dataset": {
                        "name": "inhouse_cine",
                        "manifest": str(manifest),
                        "split": "development",
                        "preprocess_root": str(preprocess),
                        "parameters": {"subject": "Example0132"},
                    },
                    "sampling": {
                        "mode": "retrospective",
                        "method": "vista",
                        "acceleration": 2,
                        "seed": 1,
                        "mask_root": str(masks),
                    },
                    "model": {"name": "fake"},
                    "output": {"root": str(root / "output")},
                }
            )
            generic_sample = create_cmr_dataset(config)[0]
            validated_sample = build_inhouse_cine_sample("Example0132", reference, smaps)
            generic_input = prepare_reconstruction_input(
                generic_sample,
                mode="retrospective",
                protocol=create_sampling_protocol(config),
                acceleration=2,
                seed=1,
            )
            validated_input = prepare_reconstruction_input(
                validated_sample,
                mode="retrospective",
                protocol=VISTASamplingProtocol(mask_root=masks),
                acceleration=2,
                seed=1,
            )
            zero_filled = ZeroFilledAdapter()
            generic_zf = zero_filled.reconstruct(generic_input).reconstruction
            validated_zf = zero_filled.reconstruct(validated_input).reconstruction

        pairs = (
            (generic_sample.reference, validated_sample.reference),
            (generic_sample.sensitivity_maps, validated_sample.sensitivity_maps),
            (generic_input.mask, validated_input.mask),
            (generic_input.kspace, validated_input.kspace),
            (generic_zf, validated_zf),
        )
        for generic, validated in pairs:
            with self.subTest(shape=np.asarray(generic).shape):
                self.assertTrue(np.array_equal(generic, validated))
                self.assertEqual(_array_sha256(generic), _array_sha256(validated))

    def test_sampling_factory_preserves_registry_parameters(self) -> None:
        config = make_config()
        registry = SamplingRegistry()
        calls = []

        def factory(**parameters):
            calls.append(parameters)
            return FakeSampling()

        registry.register("fake_sampling", factory)
        protocol = create_sampling_protocol(config, registry)
        self.assertIsInstance(protocol, FakeSampling)
        self.assertEqual(calls, [{}])

    def test_sampling_family_and_implementation_are_independent(self) -> None:
        raw = make_config().as_dict()
        raw["sampling"].update(
            {
                "method": "uniform",
                "implementation": None,
                "acceleration": 8,
                "parameters": {"ncalib": 16},
            }
        )
        protocol = create_sampling_protocol(ExperimentConfig.from_mapping(raw))
        result = protocol.generate((1, 25, 192, 156, 1), 8, seed=1)
        self.assertEqual(result.method, "uniform")
        self.assertEqual(result.parameters["family"], "uniform")
        self.assertEqual(result.parameters["implementation"], "canonical")
        self.assertEqual(result.parameters["generator_parameters"]["ncalib"], 16)

        raw["sampling"].update(
            {
                "method": "uniform",
                "implementation": "cardiomm",
                "parameters": {"ncalib": 15},
            }
        )
        source_specific = create_sampling_protocol(ExperimentConfig.from_mapping(raw))
        result = source_specific.generate((1, 7, 31, 30, 1), 8)
        self.assertEqual(result.parameters["implementation"], "cardiomm")

        raw["sampling"].update(
            {"method": "kt_gaussian", "implementation": "cardiomm"}
        )
        cardiomm = create_sampling_protocol(ExperimentConfig.from_mapping(raw))
        result = cardiomm.generate((1, 25, 192, 156, 1), 8, seed=1)
        self.assertEqual(result.parameters["implementation"], "cardiomm")

        raw["sampling"]["implementation"] = "invented"
        with self.assertRaisesRegex(ValueError, "requires implementation"):
            create_sampling_protocol(ExperimentConfig.from_mapping(raw))

    def test_native_vista_physical_mask_is_independent_of_model_identity(self) -> None:
        compact = np.zeros((8, 3), dtype=bool)
        compact[[0, 2, 4, 6], :] = True
        hashes = []
        with mock.patch(
            "cmrbench.sampling.vista.generate_vista_mask", return_value=compact
        ):
            for model_name in ("ram", "sdum", "cardiomm"):
                raw = make_config(model_name=model_name).as_dict()
                raw["sampling"].update(
                    {
                        "method": "vista",
                        "implementation": "cmrbench",
                        "acceleration": 2,
                        "seed": 7,
                    }
                )
                protocol = create_sampling_protocol(ExperimentConfig.from_mapping(raw))
                result = protocol.generate((2, 3, 5, 8, 4), 2, seed=7)
                hashes.append(result.parameters["binary_mask_sha256"])
        self.assertEqual(len(set(hashes)), 1)

    def test_random_cartesian_is_not_a_current_generic_selector(self) -> None:
        raw = make_config().as_dict()
        raw["sampling"].update(
            {
                "method": "random_cartesian",
                "implementation": None,
                "acceleration": 4,
            }
        )
        with self.assertRaisesRegex(KeyError, "unknown sampling method"):
            create_sampling_protocol(ExperimentConfig.from_mapping(raw))

    def test_cli_success_prints_sampling_implementation_and_both_r_values(self) -> None:
        result = ReconstructionResult(
            subject_id="Example0132",
            reconstruction=np.zeros((1, 1, 2, 2)),
            model_name="fake",
            metadata={
                "cmrbench_sampling": {
                    "sampling_method": "uniform",
                    "sampling_implementation": "sdum",
                    "requested_acceleration": 8.0,
                    "actual_acceleration": 4.105263157894737,
                }
            },
        )
        output = io.StringIO()
        with mock.patch(
            "cmrbench.reconstruct.ExperimentConfig.load", return_value=make_config()
        ), mock.patch(
            "cmrbench.reconstruct.run_reconstruction", return_value=[result]
        ), redirect_stdout(output):
            self.assertEqual(reconstruct_main(["--config", "unused.yaml"]), 0)
        rendered = output.getvalue()
        self.assertIn("sampling=uniform", rendered)
        self.assertIn("implementation=sdum", rendered)
        self.assertIn("requested_R=8.0", rendered)
        self.assertIn("actual_R=4.105263157894737", rendered)

    def test_legacy_nvidia_selector_resolves_to_sdum_without_model_branching(self) -> None:
        raw = make_config().as_dict()
        raw["sampling"].update(
            {"method": "kt_gaussian", "implementation": None, "acceleration": 8}
        )
        with self.assertWarns(FutureWarning):
            protocol = create_sampling_protocol(ExperimentConfig.from_mapping(raw))
        result = protocol.generate((1, 25, 192, 156, 1), 8, seed=1)
        self.assertEqual(result.parameters["implementation"], "sdum")

    def test_effective_config_records_final_implementation_without_model_inference(self) -> None:
        raw = make_config().as_dict()
        raw["sampling"].update(
            {"method": "uniform", "implementation": None, "acceleration": 8}
        )
        uniform = with_resolved_sampling_implementation(
            ExperimentConfig.from_mapping(raw)
        )
        self.assertEqual(uniform.sampling.implementation, "canonical")

        raw["sampling"].update(
            {"method": "kt_gaussian", "implementation": None, "acceleration": 24}
        )
        with self.assertWarns(FutureWarning):
            legacy = with_resolved_sampling_implementation(
                ExperimentConfig.from_mapping(raw)
            )
        self.assertEqual(legacy.sampling.implementation, "sdum")
        self.assertEqual(legacy.model.name, uniform.model.name)

    def test_model_yaml_composition_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.yaml").write_text(
                json.dumps(
                    {
                        "name": "ram",
                        "parameters": {
                            "implementation": "official_github",
                            "device": "cuda",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "run.yaml").write_text(
                json.dumps(
                    {
                        "experiment": {"name": "composed"},
                        "dataset": {
                            "name": "inhouse_cine",
                            "manifest": "cohort.csv",
                            "split": "development",
                        },
                        "sampling": {
                            "mode": "retrospective",
                            "method": "vista",
                            "acceleration": 4,
                        },
                        "model": {
                            "config": "model.yaml",
                            "parameters": {"implementation": "deepinv"},
                        },
                        "output": {"root": "outputs/composed"},
                    }
                ),
                encoding="utf-8",
            )
            fake_yaml = SimpleNamespace(safe_load=lambda handle: json.load(handle))
            with mock.patch.dict(sys.modules, {"yaml": fake_yaml}):
                config = ExperimentConfig.load(root / "run.yaml")
        self.assertEqual(config.model.name, "ram")
        self.assertEqual(config.model.parameters["implementation"], "deepinv")
        self.assertEqual(config.model.parameters["device"], "cuda")



    def test_entrypoint_source_has_no_evaluation_imports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "cmrbench" / "reconstruct.py").read_text(encoding="utf-8")
        self.assertNotIn("cmrbench.evaluation", source)
        self.assertNotIn("save_qc", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":
    unittest.main()
