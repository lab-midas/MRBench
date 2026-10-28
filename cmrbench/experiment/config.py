"""Small typed experiment configuration backed by ordinary YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_EVALUATION_METRIC_PROTOCOL = "cmrbench_recon_metrics_v2_subject_p995"
DEFAULT_QC_POLICY = "mid_slice_fixed_phases_v1"
DEFAULT_QC_PHASE_INDICES = (6, 22)


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    seed: int | None = None


@dataclass(frozen=True)
class DatasetSection:
    name: str
    manifest: str
    split: str
    preprocess_root: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    domain: str | None = None
    selection: dict[str, Any] = field(default_factory=dict)
    resolution: dict[str, str] = field(default_factory=dict)
    input_prewhitened: bool | None = None


@dataclass(frozen=True)
class SamplingSection:
    mode: str
    method: str | None = None
    implementation: str | None = None
    acceleration: float | None = None
    seed: int | None = None
    mask_root: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSection:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: str | None = None


@dataclass(frozen=True)
class EvaluationQCSection:
    enabled: bool = True
    policy: str = DEFAULT_QC_POLICY
    phase_indices: tuple[int, ...] = DEFAULT_QC_PHASE_INDICES


@dataclass(frozen=True)
class EvaluationGIFSection:
    enabled: bool = False
    mid_slice_reconstruction: bool = True


@dataclass(frozen=True)
class EvaluationSection:
    enabled: bool = True
    metric_protocol: str = DEFAULT_EVALUATION_METRIC_PROTOCOL
    qc: EvaluationQCSection = field(default_factory=EvaluationQCSection)
    gif: EvaluationGIFSection = field(default_factory=EvaluationGIFSection)


@dataclass(frozen=True)
class OutputSection:
    root: str
    save_reconstruction: bool = True
    save_metadata: bool = True
    save_qc: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    dataset: DatasetSection
    sampling: SamplingSection
    model: ModelSection
    output: OutputSection
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    source_path: Path | None = field(default=None, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Loading YAML requires PyYAML; install the project dependencies"
            ) from exc
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw = _resolve_model_reference(raw, source.parent, yaml)
        config = cls.from_mapping(raw)
        object.__setattr__(config, "source_path", source)
        return config

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("experiment configuration must be a mapping")
        required_sections = {"experiment", "dataset", "sampling", "model", "output"}
        missing = required_sections.difference(raw)
        if missing:
            raise ValueError("missing configuration sections: " + ", ".join(sorted(missing)))

        experiment_raw = _section(raw, "experiment")
        dataset_raw = _section(raw, "dataset")
        dataset_effective = resolve_dataset_section(dataset_raw)
        sampling_raw = _section(raw, "sampling")
        model_raw = _section(raw, "model")
        output_raw = _section(raw, "output")
        evaluation_raw = _optional_section(raw, "evaluation")
        evaluation_qc_raw = _optional_section(evaluation_raw, "qc")
        evaluation_gif_raw = _optional_section(evaluation_raw, "gif")

        config = cls(
            experiment=ExperimentSection(
                name=_required_string(experiment_raw, "name", "experiment"),
                seed=experiment_raw.get("seed"),
            ),
            dataset=DatasetSection(
                name=_required_string(dataset_effective, "name", "dataset"),
                manifest=_required_string(dataset_effective, "manifest", "dataset"),
                split=_required_string(dataset_effective, "split", "dataset"),
                preprocess_root=dataset_effective.get("preprocess_root"),
                parameters=dict(dataset_effective.get("parameters") or {}),
                domain=dataset_effective.get("domain"),
                selection=dict(dataset_effective.get("selection") or {}),
                resolution=dict(dataset_effective.get("resolution") or {}),
                input_prewhitened=dataset_effective.get("input_prewhitened"),
            ),
            sampling=SamplingSection(
                mode=_required_string(sampling_raw, "mode", "sampling"),
                method=sampling_raw.get("method"),
                implementation=sampling_raw.get("implementation"),
                acceleration=sampling_raw.get("acceleration"),
                seed=sampling_raw.get("seed"),
                mask_root=sampling_raw.get("mask_root"),
                parameters=dict(sampling_raw.get("parameters") or {}),
            ),
            model=ModelSection(
                name=_required_string(model_raw, "name", "model"),
                parameters=_model_parameters(model_raw),
                environment=_optional_string(
                    model_raw.get("environment"), "model.environment"
                ),
            ),
            output=OutputSection(
                root=_required_string(output_raw, "root", "output"),
                save_reconstruction=bool(output_raw.get("save_reconstruction", True)),
                save_metadata=bool(output_raw.get("save_metadata", True)),
                save_qc=bool(output_raw.get("save_qc", False)),
            ),
            evaluation=EvaluationSection(
                enabled=_boolean(evaluation_raw, "enabled", True, "evaluation"),
                metric_protocol=str(
                    evaluation_raw.get(
                        "metric_protocol", DEFAULT_EVALUATION_METRIC_PROTOCOL
                    )
                ),
                qc=EvaluationQCSection(
                    enabled=_boolean(
                        evaluation_qc_raw, "enabled", True, "evaluation.qc"
                    ),
                    policy=str(
                        evaluation_qc_raw.get("policy", DEFAULT_QC_POLICY)
                    ),
                    phase_indices=_phase_indices(
                        evaluation_qc_raw.get(
                            "phase_indices", DEFAULT_QC_PHASE_INDICES
                        )
                    ),
                ),
                gif=EvaluationGIFSection(
                    enabled=_boolean(
                        evaluation_gif_raw, "enabled", False, "evaluation.gif"
                    ),
                    mid_slice_reconstruction=_boolean(
                        evaluation_gif_raw,
                        "mid_slice_reconstruction",
                        True,
                        "evaluation.gif",
                    ),
                ),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.sampling.implementation is not None and (
            not isinstance(self.sampling.implementation, str)
            or not self.sampling.implementation.strip()
        ):
            raise ValueError(
                "sampling.implementation must be a non-empty string when specified"
            )
        if self.sampling.mode == "retrospective":
            if not self.sampling.method:
                raise ValueError("retrospective sampling requires sampling.method")
            if self.sampling.acceleration is None:
                raise ValueError("retrospective sampling requires sampling.acceleration")
            if self.sampling.acceleration < 1:
                raise ValueError("sampling.acceleration must be >= 1")
        elif self.sampling.mode == "acquired":
            if self.sampling.method is not None:
                raise ValueError("acquired sampling must not specify sampling.method")
            if self.sampling.implementation is not None:
                raise ValueError(
                    "acquired sampling must not specify sampling.implementation"
                )
            if self.sampling.acceleration is not None:
                raise ValueError("acquired sampling must not specify sampling.acceleration")
        else:
            raise ValueError("sampling.mode must be 'retrospective' or 'acquired'")
        if self.evaluation.metric_protocol != DEFAULT_EVALUATION_METRIC_PROTOCOL:
            raise ValueError(
                "evaluation.metric_protocol must remain "
                f"{DEFAULT_EVALUATION_METRIC_PROTOCOL!r}"
            )
        if self.evaluation.qc.policy != DEFAULT_QC_POLICY:
            raise ValueError(
                f"evaluation.qc.policy must be {DEFAULT_QC_POLICY!r}"
            )
        if (
            self.evaluation.gif.enabled
            and not self.evaluation.gif.mid_slice_reconstruction
        ):
            raise ValueError(
                "evaluation.gif.enabled requires mid_slice_reconstruction=true"
            )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        return _lists_for_tuples(result)


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return value


def resolve_dataset_section(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize old dataset selectors once, before either reconstruction stage.

    ``manifest`` keeps its historical meaning (cine cohort manifest or one
    fastMRI HDF5 basename).  The effective config additionally records the
    subject/slice selection and the source of each inferred value.
    """

    resolved = dict(raw)
    name = _required_string(raw, "name", "dataset")
    parameters = dict(raw.get("parameters") or {})
    selection = dict(_optional_section(raw, "selection"))
    provenance = dict(_optional_section(raw, "resolution"))
    domain = _optional_string(raw.get("domain"), "dataset.domain")
    input_prewhitened, prewhitening_source = _resolve_input_prewhitened(
        raw, parameters, name
    )
    provenance.setdefault("input_prewhitened", prewhitening_source)
    if name == "fastmri_multicoil":
        # Existing ESPIRiT provenance consumes this legacy key. Populate it
        # only in the effective config; source YAML need not repeat the fact.
        parameters["prewhitened"] = input_prewhitened
    if name == "inhouse_cine":
        legacy_subject = _optional_string(
            parameters.get("subject"), "dataset.parameters.subject"
        )
        explicit_subject = "subject" in selection
        explicit_slices = "slices" in selection
        if "subject" in selection and selection["subject"] != legacy_subject and legacy_subject is not None:
            raise ValueError("dataset.selection.subject conflicts with dataset.parameters.subject")
        selection.setdefault("subject", legacy_subject)
        provenance.setdefault(
            "subject",
            "dataset.selection.subject" if explicit_subject else
            "dataset.parameters.subject" if legacy_subject else "all manifest subjects",
        )
        selection.setdefault("slices", "all")
        provenance.setdefault(
            "slices", "dataset.selection.slices" if explicit_slices else
            "inhouse_cine complete-subject default",
        )
        if selection["slices"] != "all":
            raise ValueError("inhouse_cine generic experiments require selection.slices=all")
        if domain not in {None, "cardiac_cine"}:
            raise ValueError("inhouse_cine domain must be cardiac_cine")
        provenance.setdefault("domain", "dataset.domain" if domain else "inhouse_cine default")
        domain = "cardiac_cine"
    elif name == "fastmri_multicoil":
        filename = _required_string(raw, "manifest", "dataset")
        explicit_subject = "subject" in selection
        if "subject" in selection and selection["subject"] != filename:
            raise ValueError("dataset.selection.subject must match fastMRI manifest filename")
        selection.setdefault("subject", filename)
        provenance.setdefault(
            "subject", "dataset.selection.subject" if explicit_subject else "dataset.manifest"
        )
        legacy_slice = parameters.get("slice_index")
        if "slices" in selection:
            if legacy_slice is not None and selection["slices"] != legacy_slice:
                raise ValueError("dataset.selection.slices conflicts with dataset.parameters.slice_index")
            provenance.setdefault("slices", "dataset.selection.slices")
        elif legacy_slice is not None:
            selection["slices"] = legacy_slice
            provenance.setdefault("slices", "dataset.parameters.slice_index")
        else:
            raise ValueError("fastmri_multicoil requires selection.slices=all or a legacy slice_index")
        slices = selection["slices"]
        if slices != "all" and (isinstance(slices, bool) or not isinstance(slices, int) or slices < 0):
            raise ValueError("fastMRI selection.slices must be 'all' or a nonnegative integer")
        legacy_domain = _optional_string(parameters.get("anatomy"), "dataset.parameters.anatomy")
        domain = domain.lower() if domain is not None else None
        legacy_domain = legacy_domain.lower() if legacy_domain is not None else None
        if domain is not None and legacy_domain is not None and domain != legacy_domain:
            raise ValueError("dataset.domain conflicts with dataset.parameters.anatomy")
        domain = domain or legacy_domain
        if domain not in {"brain", "knee"}:
            raise ValueError("fastmri_multicoil requires domain=brain or knee")
        provenance.setdefault("domain", "dataset.domain" if raw.get("domain") else "dataset.parameters.anatomy")
    else:
        # A future registered reader owns its own selection semantics.  The
        # generic parser does not invent subject or dimension assumptions.
        provenance.setdefault("selection", "reader-defined; not inferred")
    resolved["domain"] = domain
    resolved["selection"] = selection
    resolved["parameters"] = parameters
    resolved["resolution"] = provenance
    resolved["input_prewhitened"] = input_prewhitened
    return resolved


def _resolve_input_prewhitened(
    raw: Mapping[str, Any], parameters: Mapping[str, Any], name: str
) -> tuple[bool | None, str]:
    """Describe input state; this never requests a preprocessing action."""

    declarations = [
        (label, source[key])
        for label, source, key in (
            ("dataset.input_prewhitened", raw, "input_prewhitened"),
            ("dataset.parameters.input_prewhitened", parameters, "input_prewhitened"),
            ("dataset.parameters.prewhitened", parameters, "prewhitened"),
        )
        if key in source
    ]
    for label, value in declarations:
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{label} must be Boolean or null")
    if declarations:
        source, value = declarations[0]
        for other_source, other_value in declarations[1:]:
            if other_value != value:
                raise ValueError(
                    f"{source} conflicts with {other_source}: "
                    "input prewhitening provenance must agree"
                )
        return value, source
    if name == "fastmri_multicoil":
        return False, "fastmri_multicoil raw-input default"
    return None, "unknown; reader/source has not declared noise prewhitening"


def _optional_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return value


def _required_string(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{key} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when specified")
    return value.strip()


def _boolean(
    section: Mapping[str, Any], key: str, default: bool, section_name: str
) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{section_name}.{key} must be Boolean")
    return value


def _phase_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("evaluation.qc.phase_indices must be a non-empty list")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(
                "evaluation.qc.phase_indices must contain nonnegative integers"
            )
        if item in indices:
            raise ValueError("evaluation.qc.phase_indices must not contain duplicates")
        indices.append(item)
    return tuple(indices)


def _lists_for_tuples(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _lists_for_tuples(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_lists_for_tuples(item) for item in value]
    if isinstance(value, list):
        return [_lists_for_tuples(item) for item in value]
    return value


def _model_parameters(section: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(section.get("parameters") or {})
    for key, value in section.items():
        if key in {"name", "parameters", "environment"}:
            continue
        if key in parameters:
            raise ValueError(
                f"model parameter {key!r} is specified both directly and under parameters"
            )
        parameters[key] = value
    return parameters


def _resolve_model_reference(
    raw: Any, source_directory: Path, yaml_module: Any
) -> Any:
    """Compose one small run config with an existing model YAML, if requested."""

    if not isinstance(raw, Mapping):
        return raw
    model = raw.get("model")
    if not isinstance(model, Mapping) or "config" not in model:
        return raw
    reference_value = model.get("config")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise ValueError("model.config must be a non-empty YAML path")
    reference = Path(reference_value)
    if not reference.is_absolute():
        reference = source_directory / reference
    with reference.open("r", encoding="utf-8") as handle:
        referenced = yaml_module.safe_load(handle)
    if not isinstance(referenced, Mapping):
        raise ValueError(f"referenced model configuration must be a mapping: {reference}")
    referenced_name = _required_string(referenced, "name", "referenced model")
    referenced_environment = _optional_string(
        referenced.get("environment"), "referenced model.environment"
    )
    referenced_parameters = dict(referenced.get("parameters") or {})
    overrides = dict(model.get("parameters") or {})
    direct_overrides = {
        key: value
        for key, value in model.items()
        if key not in {"config", "name", "parameters", "environment"}
    }
    duplicate_overrides = set(overrides).intersection(direct_overrides)
    if duplicate_overrides:
        raise ValueError(
            "model override specified twice: "
            + ", ".join(sorted(duplicate_overrides))
        )
    requested_name = model.get("name", referenced_name)
    if requested_name != referenced_name:
        raise ValueError(
            f"model.name {requested_name!r} disagrees with referenced model "
            f"name {referenced_name!r}"
        )
    composed = dict(raw)
    composed["model"] = {
        "name": referenced_name,
        "environment": model.get("environment", referenced_environment),
        "parameters": {
            **referenced_parameters,
            **overrides,
            **direct_overrides,
        },
    }
    return composed
