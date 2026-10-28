"""Minimal configuration-driven reconstruction entry point for CMRBench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
import warnings

from cmrbench.data.base import CineMRIDataset
from cmrbench.data.fastmri import FastMRIMulticoilDataset
from cmrbench.data.inhouse_cine import InHouseCineDataset
from cmrbench.data.manifest import CohortManifest
from cmrbench.data.sample import CineMRISample
from cmrbench.experiment.config import ExperimentConfig, resolve_dataset_section
from cmrbench.experiment.runner import ExperimentRunner
from cmrbench.experiment.runtimes import reconstruction_runtime
from cmrbench.models.registry import ModelRegistry, default_model_registry
from cmrbench.sampling.registry import SamplingRegistry, default_sampling_registry

DatasetFactory = Callable[[ExperimentConfig], CineMRIDataset]


class DatasetRegistry:
    """Small reader extension point; readers return canonical whole subjects."""

    def __init__(self) -> None:
        self._factories: dict[str, DatasetFactory] = {}

    def register(self, name: str, factory: DatasetFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"dataset reader {name!r} is empty or already registered")
        self._factories[name] = factory

    def create(self, config: ExperimentConfig) -> CineMRIDataset:
        try:
            factory = self._factories[config.dataset.name]
        except KeyError as exc:
            raise ValueError(
                f"unsupported dataset reader {config.dataset.name!r}; "
                f"registered: {', '.join(sorted(self._factories))}"
            ) from exc
        return factory(config)


class _SubjectDataset(CineMRIDataset):
    """Select whole canonical subject samples without altering their geometry."""

    def __init__(
        self,
        source: InHouseCineDataset,
        *,
        subject: str | None,
        input_prewhitened: bool | None = None,
        input_prewhitening_source: str = "unknown",
    ) -> None:
        self.source = source
        self.input_prewhitened = input_prewhitened
        self.input_prewhitening_source = input_prewhitening_source
        if subject is None:
            self.indices = list(range(len(source)))
        else:
            subject_id = Path(subject).stem
            self.indices = [
                index
                for index, record in enumerate(source.records)
                if record.subject_id == subject_id
            ]
            if not self.indices:
                raise ValueError(
                    f"subject {subject_id!r} is not present in split {source.split!r}"
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> CineMRISample:
        sample = self.source[self.indices[index]]
        sample.metadata["input_preprocessing"] = {
            "prewhitened": self.input_prewhitened,
            "source": self.input_prewhitening_source,
            "prewhitening_applied_by_cmrbench": False,
        }
        return sample


def _effective_selection(config: ExperimentConfig) -> dict[str, Any]:
    if config.dataset.selection:
        return config.dataset.selection
    return resolve_dataset_section(
        {
            "name": config.dataset.name,
            "manifest": config.dataset.manifest,
            "split": config.dataset.split,
            "parameters": config.dataset.parameters,
            "domain": config.dataset.domain,
        }
    )["selection"]


def _create_inhouse_cine_dataset(config: ExperimentConfig) -> CineMRIDataset:
    parameters = dict(config.dataset.parameters)
    forbidden_geometry = {"slice", "phase", "slice_index", "phase_index"}.intersection(
        parameters
    )
    if forbidden_geometry:
        raise ValueError(
            "generic reconstruction preserves complete canonical S/T dimensions; "
            "pre-sampling slice/phase selection is not supported: "
            + ", ".join(sorted(forbidden_geometry))
        )
    supported = {
        "subject", "sensitivity_map_coils", "input_prewhitened", "prewhitened"
    }
    unknown = set(parameters).difference(supported)
    if unknown:
        raise ValueError(
            "unsupported inhouse_cine dataset parameters: " + ", ".join(sorted(unknown))
        )
    manifest = CohortManifest.load(config.dataset.manifest)
    source = InHouseCineDataset(
        manifest,
        split=config.dataset.split,
        preprocess_root=config.dataset.preprocess_root,
        sensitivity_map_coils=int(parameters.get("sensitivity_map_coils", 15)),
    )
    return _SubjectDataset(
        source,
        subject=_optional_string(
            _effective_selection(config).get("subject"), "dataset.selection.subject"
        ),
        input_prewhitened=config.dataset.input_prewhitened,
        input_prewhitening_source=config.dataset.resolution.get(
            "input_prewhitened", "unknown"
        ),
    )


def _create_fastmri_multicoil_dataset(
    config: ExperimentConfig,
) -> FastMRIMulticoilDataset:
    parameters = dict(config.dataset.parameters)
    supported = {
        "anatomy",
        "slice_index",
        "sensitivity_maps",
        "prewhitened",
        "input_prewhitened",
        "scaling",
    }
    unknown = set(parameters).difference(supported)
    if unknown:
        raise ValueError(
            "unsupported fastmri_multicoil dataset parameters: "
            + ", ".join(sorted(unknown))
        )
    if config.dataset.preprocess_root is None:
        raise ValueError("fastmri_multicoil requires dataset.preprocess_root")
    if parameters.get("scaling") != "none":
        raise ValueError(
            "fastmri_multicoil currently requires dataset.parameters.scaling='none'"
        )
    selection = _effective_selection(config)
    if "slices" not in selection:
        raise ValueError("fastmri_multicoil requires dataset.selection.slices")
    anatomy = _optional_string(
        config.dataset.domain or parameters.get("anatomy"), "dataset.domain"
    )
    if anatomy is None:
        raise ValueError("fastmri_multicoil requires dataset.parameters.anatomy")
    return FastMRIMulticoilDataset(
        config.dataset.preprocess_root,
        filename=config.dataset.manifest,
        slice_index=selection["slices"],
        split=config.dataset.split,
        anatomy=anatomy,
        input_prewhitened=config.dataset.input_prewhitened,
        input_prewhitening_source=config.dataset.resolution.get(
            "input_prewhitened", "unknown"
        ),
    )


default_dataset_registry = DatasetRegistry()
default_dataset_registry.register("inhouse_cine", _create_inhouse_cine_dataset)
default_dataset_registry.register("fastmri_multicoil", _create_fastmri_multicoil_dataset)


def create_cmr_dataset(
    config: ExperimentConfig, registry: DatasetRegistry = default_dataset_registry
) -> CineMRIDataset:
    """Resolve a registered canonical subject reader; shared by both stages."""

    return registry.create(config)


def create_sampling_protocol(
    config: ExperimentConfig,
    registry: SamplingRegistry = default_sampling_registry,
):
    if config.sampling.mode == "acquired":
        return None
    if config.sampling.method is None:
        raise ValueError("retrospective sampling requires sampling.method")
    method = config.sampling.method
    implementation = resolve_sampling_implementation(
        method, config.sampling.implementation
    )
    parameters = dict(config.sampling.parameters)
    if config.sampling.mask_root is not None:
        if method != "vista":
            raise ValueError("sampling.mask_root is supported only by the vista protocol")
        parameters.setdefault("mask_root", config.sampling.mask_root)
    if method == "uniform":
        if implementation == "canonical":
            return registry.create(method, **parameters)
        return registry.create(
            method, implementation_selector=implementation, **parameters
        )
    if registry.implementations(method):
        return registry.create(
            method, implementation_selector=implementation, **parameters
        )
    elif method == "external_mask":
        parameters.setdefault("implementation", implementation or "external")
    return registry.create(method, **parameters)


def resolve_sampling_implementation(
    method: str, configured: str | None
) -> str | None:
    """Resolve implementation independently of model identity and fail closed."""

    normalized = configured.strip().lower() if configured is not None else None
    if method == "uniform":
        if normalized not in {None, "canonical", "sdum", "cardiomm"}:
            raise ValueError(
                "uniform supports implementation='canonical', 'sdum', or 'cardiomm'"
            )
        return normalized or "canonical"
    if method in {"kt_uniform", "kt_gaussian", "kt_radial"}:
        if normalized is None:
            warnings.warn(
                f"sampling method {method!r} without implementation is ambiguous; "
                "temporarily resolving to 'sdum'. Set implementation: sdum or "
                "implementation: cardiomm explicitly.",
                FutureWarning,
                stacklevel=2,
            )
            return "sdum"
        if normalized not in {"sdum", "cardiomm"}:
            raise ValueError(
                f"sampling method {method!r} requires implementation='sdum' or "
                "implementation='cardiomm'"
            )
        return normalized
    if method == "random":
        if normalized not in {"sdum", "cardiomm"}:
            raise ValueError("random requires implementation='sdum' or 'cardiomm'")
        return normalized
    if method == "equispaced":
        if normalized not in {None, "sdum"}:
            raise ValueError("equispaced supports implementation='sdum'")
        return "sdum"
    if method == "radial":
        if normalized not in {None, "cardiomm"}:
            raise ValueError("radial supports implementation='cardiomm'")
        return "cardiomm"
    if method == "vista":
        if normalized in {None, "vista", "precomputed"}:
            return "precomputed"
        if normalized == "cmrbench":
            return "cmrbench"
        raise ValueError(
            "vista requires implementation='precomputed' or implementation='cmrbench'"
        )
    if method == "external_mask":
        return normalized or "external"
    return configured


def with_resolved_sampling_implementation(config: ExperimentConfig) -> ExperimentConfig:
    """Return the effective config persisted by a run."""

    if config.sampling.mode == "acquired" or config.sampling.method is None:
        return config
    implementation = resolve_sampling_implementation(
        config.sampling.method, config.sampling.implementation
    )
    if implementation == config.sampling.implementation:
        return config
    return replace(
        config,
        sampling=replace(config.sampling, implementation=implementation),
    )


def create_reconstruction_model(
    config: ExperimentConfig,
    registry: ModelRegistry = default_model_registry,
):
    """Create exactly the selected adapter; adapter validation remains authoritative."""

    parameters = dict(config.model.parameters)
    if config.model.name == "cardiomm" and "num_low_frequencies" not in parameters:
        # An omitted value means the adapter resolves the width from the
        # realized physical sampling policy once runtime Ky is known.
        parameters["num_low_frequencies"] = None
    return registry.create(config.model.name, **parameters)


def run_reconstruction(
    config: ExperimentConfig,
    *,
    dataset_factory: DatasetFactory = create_cmr_dataset,
    model_registry: ModelRegistry = default_model_registry,
    sampling_registry: SamplingRegistry = default_sampling_registry,
):
    """Run the existing preparation/adapter/output pipeline without evaluation."""

    config = with_resolved_sampling_implementation(config)
    model = create_reconstruction_model(config, model_registry)
    protocol = create_sampling_protocol(config, sampling_registry)
    runner = ExperimentRunner(
        config,
        dataset_factory=dataset_factory,
        model=model,
        sampling_protocol=protocol,
    )
    return runner.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one configured CMRBench reconstruction through the canonical "
            "dataset, sampling, adapter, and output interfaces."
        ),
        epilog=(
            "Current model choices: model.name=ram requires "
            "implementation=official_github|deepinv; model.name=sdum requires "
            "variant=small|base|large; model.name=cardiomm requires the pinned "
            "CardioMM source, local CLIP, and checkpoint assets. Select the matching isolated interpreter: "
            "RAM official -> ram; RAM DeepInv -> ram-deepinv035; SDUM -> sdum. "
            "CardioMM -> cardiomm. "
            "This command does not switch environments and does not calculate "
            "metrics or QC."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a reconstruction YAML")
    parser.add_argument(
        "--output-dir",
        help="Optional output-root override; the YAML remains the recorded source config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = ExperimentConfig.load(arguments.config)
    if arguments.output_dir:
        config = replace(config, output=replace(config.output, root=arguments.output_dir))
    try:
        results = run_reconstruction(config)
    except Exception as exc:
        hint = runtime_hint(config)
        parser.exit(
            2,
            f"CMRBench reconstruction failed for model {config.model.name!r}. "
            f"Expected runtime: {hint}. {type(exc).__name__}: {exc}\n",
        )
    for result in results:
        sampling = result.metadata.get("cmrbench_sampling", {})
        print(
            "PASS: subject={} model={} sampling={} implementation={} "
            "requested_R={} actual_R={}".format(
                result.subject_id,
                config.model.name,
                sampling.get("sampling_method", "unrecorded"),
                sampling.get("sampling_implementation", "unrecorded"),
                sampling.get("requested_acceleration"),
                sampling.get("actual_acceleration"),
            )
        )
    print(f"PASS: reconstructed {len(results)} subject(s); outputs: {config.output.root}")
    return 0


def runtime_hint(config: ExperimentConfig) -> str:
    try:
        return reconstruction_runtime(config)
    except ValueError:
        return "an interpreter containing the selected registered adapter"


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when specified")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
