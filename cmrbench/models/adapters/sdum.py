"""Production adapter for pinned NVIDIA NV-Raw2Insights-MRI (SDUM)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import inspect
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping
import warnings

import numpy as np

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.errors import UpstreamModelInferenceError
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord
from cmrbench.sampling.result import (
    calculate_actual_acceleration,
    detect_complete_central_acs,
)


SDUM_ADAPTER_VERSION = "1.4"
SDUM_REPOSITORY = "https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI"
SDUM_COMMIT = "99be712b2e9fd34bf6a2f709373d311354fbda96"
SMALL_CONFIG_NAME = "nv_raw2insights_mri_small.json"
SMALL_MODEL_VARIANT = "nv_raw2insights_mri_small"
SMALL_CHECKPOINT_SHA256 = (
    "8ba66defef7c66738c1eac25f2ea6a5d164fb9e547c36d90b469d8d5287ed624"
)
BASE_CONFIG_NAME = "nv_raw2insights_mri_base.json"
BASE_MODEL_VARIANT = "nv_raw2insights_mri_base"
BASE_CHECKPOINT_SHA256 = (
    "667189bac5f159e52b2f1640b504b4efd549717c37720c2397e749d16ec056b8"
)
BASE_CASCADE_COUNT = 18
LARGE_CONFIG_NAME = "nv_raw2insights_mri_large.json"
LARGE_MODEL_VARIANT = "nv_raw2insights_mri_large"
LARGE_CHECKPOINT_SHA256 = (
    "0a1d06d74dcd3dbd0c3692edc203527370a796fbb6f61be4c58bdcbcb817c54b"
)
LARGE_CASCADE_COUNT = 34
BASE_ACCELERATION_CATEGORIES = (8, 16, 24)
BASE_MASK_FAMILIES = ("uniform", "kt_gaussian", "kt_radial")
BASE_ACQUISITIONS = (
    "BlackBlood",
    "Cine",
    "Flow2d",
    "LGE",
    "Mapping",
    "Perfusion",
    "T1rho",
    "T1w",
    "T2w",
)
LARGE_ACCELERATION_CATEGORIES = BASE_ACCELERATION_CATEGORIES
LARGE_MASK_FAMILIES = BASE_MASK_FAMILIES
LARGE_ACQUISITIONS = BASE_ACQUISITIONS
SMALL_ACCELERATION_CATEGORIES = BASE_ACCELERATION_CATEGORIES
SMALL_MASK_FAMILIES = BASE_MASK_FAMILIES
SMALL_ACQUISITIONS = ("Cine",)
SDUM_RUNTIME_MASK_FAMILIES = ("uniform", "gaussian", "radial")
SDUM_SMALL_RUNTIME_ACQUISITIONS = BASE_ACQUISITIONS + ("Aorta", "Tagging")
NUM_TEMPORAL_FRAMES = 5
INFERENCE_MODE = "direct_released_pretrained_inference"
OUTPUT_REPRESENTATION = "real_rss_magnitude"
SUPPORTED_POSITIVE_CONTROL = ("ktRadial16", 16, "Cine")
SDUM_DOCUMENTED_REFERENCE_NCALIB = 20
SUPPORTED_SMALL_CONDITIONS = tuple(
    (
        {"uniform": "Uniform", "kt_gaussian": "ktGaussian", "kt_radial": "ktRadial"}[
            family
        ]
        + str(acceleration),
        acceleration,
        acquisition,
    )
    for family in SMALL_MASK_FAMILIES
    for acceleration in SMALL_ACCELERATION_CATEGORIES
    for acquisition in SMALL_ACQUISITIONS
)


@dataclass(frozen=True)
class SDUMVariantSpec:
    name: str
    config_name: str
    model_variant: str
    checkpoint_sha256: str
    checkpoint_role: str
    cascade_count: int
    label_conditioning: bool


SDUM_VARIANTS = {
    "small": SDUMVariantSpec(
        name="small",
        config_name=SMALL_CONFIG_NAME,
        model_variant=SMALL_MODEL_VARIANT,
        checkpoint_sha256=SMALL_CHECKPOINT_SHA256,
        checkpoint_role="engineering / adapter parity smoke",
        cascade_count=6,
        label_conditioning=False,
    ),
    "base": SDUMVariantSpec(
        name="base",
        config_name=BASE_CONFIG_NAME,
        model_variant=BASE_MODEL_VARIANT,
        checkpoint_sha256=BASE_CHECKPOINT_SHA256,
        checkpoint_role="primary released scientific model",
        cascade_count=BASE_CASCADE_COUNT,
        label_conditioning=True,
    ),
    "large": SDUMVariantSpec(
        name="large",
        config_name=LARGE_CONFIG_NAME,
        model_variant=LARGE_MODEL_VARIANT,
        checkpoint_sha256=LARGE_CHECKPOINT_SHA256,
        checkpoint_role="released model under validation",
        cascade_count=LARGE_CASCADE_COUNT,
        label_conditioning=True,
    ),
}


@dataclass(frozen=True)
class SDUMConditioning:
    """Adapter-private released-model conditioning (not a common contract)."""

    mask_type: str
    acceleration: int
    acquisition: str

    def as_model_arguments(self) -> tuple[str, int, str]:
        return self.mask_type, self.acceleration, self.acquisition


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sdum_source(source_root: Path) -> str:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"SDUM source root does not exist: {root}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot verify SDUM source at {root}: {exc}") from exc
    if commit != SDUM_COMMIT:
        raise RuntimeError(
            f"SDUM checkout is {commit or 'UNKNOWN'}, but pinned commit {SDUM_COMMIT} is required"
        )
    if status:
        raise RuntimeError("pinned SDUM source checkout is dirty; upstream must remain unmodified")
    return commit


def verify_small_checkpoint(path: Path) -> str:
    """Backward-compatible Small-only checkpoint verifier."""

    return verify_sdum_checkpoint(path, variant="small")


def sdum_variant_spec(variant: str) -> SDUMVariantSpec:
    try:
        return SDUM_VARIANTS[str(variant).lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown SDUM variant {variant!r}; expected one of {sorted(SDUM_VARIANTS)}"
        ) from exc


def verify_sdum_checkpoint(path: Path, *, variant: str) -> str:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"SDUM checkpoint does not exist: {checkpoint}")
    spec = sdum_variant_spec(variant)
    actual = sha256_file(checkpoint)
    if actual != spec.checkpoint_sha256:
        raise RuntimeError(
            f"SDUM {spec.name.title()} checkpoint SHA-256 is {actual}, "
            f"expected {spec.checkpoint_sha256}"
        )
    return actual


def released_mask_family(mask_type: str) -> str | None:
    prefix = re.sub(r"\d+$", "", str(mask_type)).lower()
    return {
        "uniform": "uniform",
        "ktgaussian": "kt_gaussian",
        "ktradial": "kt_radial",
    }.get(prefix)


def _sampling_ncalib(parameters: Mapping[str, Any]) -> int | None:
    value = parameters.get("ncalib")
    generated = parameters.get("generator_parameters")
    if value is None and isinstance(generated, Mapping):
        value = generated.get("ncalib")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _condition_family(value: object) -> str | None:
    token = re.sub(r"\d+$", "", re.sub(r"[^a-z0-9]", "", str(value).lower()))
    return {
        "uniform": "uniform",
        "ktuniform": "uniform",
        "equispaced": "uniform",
        "gaussian": "gaussian",
        "ktgaussian": "gaussian",
        "radial": "radial",
        "ktradial": "radial",
    }.get(token)


def _physical_sdum_family(sampling: Any) -> tuple[str | None, str]:
    method = str(sampling.method).lower()
    parameters = dict(sampling.parameters)
    if method == "external_mask":
        declared = parameters.get("family", parameters.get("pattern"))
        return _condition_family(declared), "source_alias"
    mapping = {
        "uniform": ("uniform", "native_match"),
        "kt_uniform": ("uniform", "source_alias"),
        "equispaced": ("uniform", "source_alias"),
        "kt_gaussian": ("gaussian", "native_match"),
        "kt_radial": ("radial", "native_match"),
        "radial": ("radial", "source_alias"),
    }
    return mapping.get(method, (None, "ambiguous"))


def _sdum_mask_type(family: str, acceleration: int) -> str:
    return {
        "uniform": "Uniform",
        "gaussian": "ktGaussian",
        "radial": "ktRadial",
    }[family] + str(acceleration)


def resolve_sdum_conditioning(
    input: ReconstructionInput,
    *,
    sampling_condition: Mapping[str, Any] | None,
    legacy_conditioning: SDUMConditioning | None,
    default_acquisition: str,
) -> tuple[SDUMConditioning, dict[str, Any]]:
    """Resolve model-side conditioning independently from the physical mask."""

    sampling = input.sampling
    if sampling is None:
        raise ValueError("SDUM requires SamplingResult provenance")
    requested = sampling.requested_acceleration
    if requested is None or not float(requested).is_integer() or float(requested) < 1:
        raise ValueError("SDUM requires a positive integer physical nominal acceleration")
    physical_nominal = int(float(requested))
    physical_family, automatic_relation = _physical_sdum_family(sampling)

    if sampling_condition is None and legacy_conditioning is not None:
        family = _condition_family(legacy_conditioning.mask_type)
        if family is None:
            raise ValueError(
                "unmatched legacy SDUM mask_type requires sampling_condition.mode="
                "upstream_raw with allow_unmatched=true"
            )
        condition = legacy_conditioning
        return condition, {
            "model_conditioning_mode": "explicit_legacy",
            "model_conditioning_family": family,
            "conditioning_relation_to_physical_sampling": (
                automatic_relation
                if family is not None and family == physical_family
                else "user_forced"
            ),
            "raw_unmatched_opt_in": False,
        }

    request = dict(sampling_condition or {})
    mode = str(request.pop("mode", "auto")).lower()
    acquisition = str(request.pop("acquisition", default_acquisition))
    acceleration_value = request.pop("acceleration", physical_nominal)
    if isinstance(acceleration_value, bool) or not float(acceleration_value).is_integer():
        raise ValueError("SDUM model conditioning acceleration must be an integer")
    acceleration = int(float(acceleration_value))
    if acceleration < 1:
        raise ValueError("SDUM model conditioning acceleration must be positive")

    if mode == "auto":
        if request:
            raise ValueError(
                "SDUM sampling_condition mode=auto accepts only acceleration/acquisition"
            )
        if physical_family is None:
            raise ValueError(
                "SDUM physical sampling has no unambiguous automatic model family; "
                "set sampling_condition.mode=explicit or upstream_raw"
            )
        family = physical_family
        raw_mask_type = _sdum_mask_type(family, acceleration)
        relation = automatic_relation
        raw_opt_in = False
    elif mode == "explicit":
        family_value = request.pop("family", None)
        family = _condition_family(family_value)
        if family is None:
            raise ValueError(
                "SDUM explicit sampling_condition.family must be uniform, gaussian, or radial"
            )
        if request:
            raise ValueError(
                "unknown SDUM explicit sampling_condition keys: "
                + ", ".join(sorted(request))
            )
        raw_mask_type = _sdum_mask_type(family, acceleration)
        relation = "user_forced"
        raw_opt_in = False
    elif mode == "upstream_raw":
        raw_mask_type = request.pop("raw_mask_type", None)
        raw_opt_in = request.pop("allow_unmatched", False)
        if not isinstance(raw_mask_type, str) or not raw_mask_type.strip():
            raise ValueError("SDUM upstream_raw requires a non-empty raw_mask_type")
        if raw_opt_in is not True:
            raise ValueError(
                "SDUM upstream_raw requires allow_unmatched=true explicit opt-in"
            )
        if request:
            raise ValueError(
                "unknown SDUM upstream_raw sampling_condition keys: "
                + ", ".join(sorted(request))
            )
        family = _condition_family(raw_mask_type)
        relation = "unmatched_raw"
    else:
        raise ValueError("SDUM sampling_condition.mode must be auto, explicit, or upstream_raw")

    return SDUMConditioning(raw_mask_type, acceleration, acquisition), {
        "model_conditioning_mode": mode,
        "model_conditioning_family": family,
        "conditioning_relation_to_physical_sampling": relation,
        "raw_unmatched_opt_in": raw_opt_in,
    }


def preflight_conditioning(
    conditioning: SDUMConditioning, *, variant: str = "small"
) -> dict[str, Any]:
    spec = sdum_variant_spec(variant)
    resolved_upstream_family = _condition_family(conditioning.mask_type)
    mask_family = released_mask_family(conditioning.mask_type)
    suffix = re.search(r"(\d+)$", conditioning.mask_type)
    suffix_acceleration = int(suffix.group(1)) if suffix else None
    if spec.name == "small":
        supported = bool(
            resolved_upstream_family in {"uniform", "gaussian", "radial"}
            and conditioning.acceleration in SMALL_ACCELERATION_CATEGORIES
            and suffix_acceleration == conditioning.acceleration
            and conditioning.acquisition in SMALL_ACQUISITIONS
        )
        released_conditions = {
            "mask_families": list(SMALL_MASK_FAMILIES),
            "acceleration_categories": list(SMALL_ACCELERATION_CATEGORIES),
            "acquisitions": list(SMALL_ACQUISITIONS),
            "filename_suffix_must_match_acceleration": True,
        }
        scope = (
            "released Small source-defined uniform/kt-gaussian/kt-radial, "
            "R8/R16/R24 Cine vocabulary; the former CMRBench ktRadial8/16-only "
            "validated-subset whitelist was removed; Phase-2B bitwise parity "
            "remains anchored to ktRadial16/R16/Cine"
        )
    else:
        supported = bool(
            resolved_upstream_family in {"uniform", "gaussian", "radial"}
            and conditioning.acceleration in BASE_ACCELERATION_CATEGORIES
            and suffix_acceleration == conditioning.acceleration
            and conditioning.acquisition in BASE_ACQUISITIONS
        )
        released_conditions = {
            "mask_families": list(BASE_MASK_FAMILIES),
            "acceleration_categories": list(BASE_ACCELERATION_CATEGORIES),
            "acquisitions": list(BASE_ACQUISITIONS),
            "filename_suffix_must_match_acceleration": True,
        }
        scope = (
            f"released {spec.name.title()} source-defined mask family, nominal "
            "acceleration, and acquisition vocabularies; no unknown-family fallback"
        )
    return {
        "status": "SUPPORTED" if supported else "UNDOCUMENTED",
        "variant": spec.name,
        "conditioning": asdict(conditioning),
        "mask_family": mask_family,
        "resolved_upstream_mask_family": resolved_upstream_family,
        "mask_suffix_acceleration": suffix_acceleration,
        "released_condition": asdict(conditioning) if supported else None,
        "official_positive_control_anchor": {
            "mask_type": SUPPORTED_POSITIVE_CONTROL[0],
            "acceleration": SUPPORTED_POSITIVE_CONTROL[1],
            "acquisition": SUPPORTED_POSITIVE_CONTROL[2],
        },
        "released_conditions": released_conditions,
        "silent_fallback": False,
        "label_conditioning": spec.label_conditioning,
        "scope": scope,
    }


def sdum_sampling_compatibility(
    input: ReconstructionInput,
    conditioning: SDUMConditioning,
    *,
    variant: str,
    conditioning_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate physical truth and expose, rather than suppress, SDUM OOD resolution."""

    spec = sdum_variant_spec(variant)
    preflight = preflight_conditioning(conditioning, variant=variant)
    sampling = input.sampling
    if sampling is None:
        raise ValueError("SDUM requires SamplingResult provenance to validate conditioning")
    method = str(sampling.method).lower()
    parameters = dict(sampling.parameters)
    sampling_family, _automatic_relation = _physical_sdum_family(sampling)
    physical_family = sampling_family or method
    implementation = parameters.get("implementation")
    source_identity = parameters.get("source_identity") or parameters.get("generator")
    if implementation is None:
        raise ValueError(
            "SDUM requires explicit sampling implementation provenance; "
            "the model adapter never infers a generator from model identity"
        )
    conditioning_family = _condition_family(conditioning.mask_type)
    requested = sampling.requested_acceleration
    if requested is None or not float(requested).is_integer():
        raise ValueError("SDUM requires an integer requested nominal acceleration category")
    nominal = int(float(requested))
    ncalib = _sampling_ncalib(parameters)
    if ncalib is not None and ncalib < 1:
        raise ValueError("SDUM declared ncalib provenance must be positive")
    binary = strict_mask_support(np.asarray(input.mask))
    sampling_mask = strict_mask_support(np.asarray(sampling.mask))
    if sampling_mask.shape != binary.shape or not np.array_equal(sampling_mask, binary):
        raise ValueError("SDUM input mask differs from SamplingResult.mask")
    measured = calculate_actual_acceleration(binary)
    if measured is None or sampling.actual_acceleration is None or not np.isclose(
        float(sampling.actual_acceleration), float(measured), rtol=0.0, atol=1e-12
    ):
        raise ValueError("SDUM measured mask acceleration provenance is inconsistent")
    _slices, _phases, kx, ky = binary.shape
    if ncalib is not None and (ncalib > ky or ncalib > kx):
        raise ValueError(
            f"SDUM spatial dimensions cannot contain the declared ACS ncalib={ncalib}"
        )
    acs_detection = detect_complete_central_acs(binary)
    if ncalib is not None:
        strip_extent = int(acs_detection["detected_centered_ky_strip_extent"])
        square_extent = int(acs_detection["detected_centered_square_extent"])
        if max(strip_extent, square_extent) < ncalib:
            raise ValueError(
                "SDUM declared ncalib provenance conflicts with the measurable "
                f"centered calibration region: ncalib={ncalib}, "
                f"strip_extent={strip_extent}, square_extent={square_extent}"
            )
    acs_geometry = str(acs_detection["detected_acs_geometry"])

    runtime_acquisitions = (
        SDUM_SMALL_RUNTIME_ACQUISITIONS if spec.name == "small" else BASE_ACQUISITIONS
    )
    mask_idx = next(
        (i for i, token in enumerate(SDUM_RUNTIME_MASK_FAMILIES) if token in conditioning.mask_type.lower()),
        -1,
    )
    acc_idx = next(
        (i for i, value in enumerate(BASE_ACCELERATION_CATEGORIES) if value == conditioning.acceleration),
        -1,
    )
    acq_idx = next(
        (i for i, value in enumerate(runtime_acquisitions) if value.lower() == conditioning.acquisition.lower()),
        -1,
    )
    if acq_idx == -1:
        raise ValueError(
            f"SDUM acquisition {conditioning.acquisition!r} is malformed for the upstream runtime"
        )
    resolved_label = (
        mask_idx * len(SDUM_RUNTIME_MASK_FAMILIES) * len(runtime_acquisitions)
        + acc_idx * len(runtime_acquisitions)
        + acq_idx
    )
    collision = None
    for known_mask_idx, known_family in enumerate(SDUM_RUNTIME_MASK_FAMILIES):
        for known_acc_idx, known_acceleration in enumerate(BASE_ACCELERATION_CATEGORIES):
            for known_acq_idx, known_acquisition in enumerate(runtime_acquisitions):
                known_label = (
                    known_mask_idx * len(SDUM_RUNTIME_MASK_FAMILIES) * len(runtime_acquisitions)
                    + known_acc_idx * len(runtime_acquisitions)
                    + known_acq_idx
                )
                if known_label == resolved_label:
                    collision = {
                        "family": known_family,
                        "acceleration": known_acceleration,
                        "acquisition": known_acquisition,
                    }
                    break
            if collision is not None:
                break
        if collision is not None:
            break
    dc_index = mask_idx if mask_idx >= 0 else len(SDUM_RUNTIME_MASK_FAMILIES) - 1
    family_documented = conditioning_family in SDUM_RUNTIME_MASK_FAMILIES
    acceleration_documented = conditioning.acceleration in BASE_ACCELERATION_CATEGORIES
    acquisition_documented = conditioning.acquisition in (
        SMALL_ACQUISITIONS if spec.name == "small" else BASE_ACQUISITIONS
    )
    normalized_pattern = re.sub(
        r"[^a-z0-9]", "", str(parameters.get("pattern", parameters.get("family", ""))).lower()
    )
    physical_documented = (
        (implementation in {"sdum", "canonical"} and method == "uniform")
        or (implementation == "sdum" and method in {"kt_gaussian", "kt_radial"})
        or (
            implementation == "sdum"
            and method == "external_mask"
            and normalized_pattern in {"uniform", "ktgaussian", "ktradial"}
        )
    )
    model_condition_documented = (
        family_documented and acceleration_documented and acquisition_documented
    )
    documented = physical_documented and model_condition_documented
    runtime_resolution = (
        "fallback" if mask_idx == -1 else "unmatched" if acc_idx == -1 else "matched"
    )
    warning_messages = []
    if not documented:
        warning_messages.append(
            "SDUM physical sampling and/or model condition is not documented by the "
            "released training/reference config; "
            "undocumented does not prove unseen"
        )
    if acc_idx == -1:
        warning_messages.append("upstream SDUM resolved the requested acceleration to acc_idx=-1")
    if mask_idx == -1:
        warning_messages.append(
            "upstream SDUM resolved mask_idx=-1 and Python negative indexing selects the radial DC map"
        )
    relation = dict(conditioning_metadata or {}).get(
        "conditioning_relation_to_physical_sampling", "unrecorded"
    )
    if relation in {"user_forced", "unmatched_raw"}:
        warning_messages.append(
            "physical sampling and SDUM model-side sampling semantics were explicitly separated"
        )
    if not acs_detection["has_complete_central_acs"]:
        warning_messages.append(
            "SDUM physical mask has no complete central ACS; CMRBench passes the "
            "physically valid mask onward and upstream get_acs_region will reject "
            "it if the center sample is absent"
        )
    for message in warning_messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    compatibility_status = (
        "supported"
        if documented and implementation in {"sdum", "canonical"}
        else "ood_valid"
    )
    return {
        "cmrbench_method": sampling.method,
        "cmrbench_family": method,
        "physical_sampling_family": physical_family,
        "sampling_implementation": str(implementation),
        "requested_nominal_acceleration": nominal,
        "actual_acceleration": float(measured),
        "ncalib": ncalib,
        "documented_reference_ncalib": SDUM_DOCUMENTED_REFERENCE_NCALIB,
        "matches_documented_reference_ncalib": (
            ncalib == SDUM_DOCUMENTED_REFERENCE_NCALIB
        ),
        "physical_sampling_validity": "valid",
        "runtime_compatibility": (
            "valid"
            if acs_detection["has_complete_central_acs"]
            else "upstream_expected_rejection"
        ),
        "checkpoint_training_acs_exposure": "unknown",
        "acs_geometry": acs_geometry,
        "has_complete_central_acs": bool(
            acs_detection["has_complete_central_acs"]
        ),
        "detected_centered_ky_strip_extent": int(
            acs_detection["detected_centered_ky_strip_extent"]
        ),
        "detected_centered_square_extent": int(
            acs_detection["detected_centered_square_extent"]
        ),
        "acs_requirement_layer": "upstream_sdum_runtime",
        "no_acs_runtime_behavior": (
            "upstream get_acs_region raises when the center sample is absent"
        ),
        "conditioning_mask_type": conditioning.mask_type,
        "conditioning_family": conditioning_family,
        "conditioning_acceleration": conditioning.acceleration,
        "compatibility_status": compatibility_status,
        "source_identity": str(source_identity or "unrecorded_external_source"),
        "model_conditioning_mode": dict(conditioning_metadata or {}).get(
            "model_conditioning_mode", "explicit_legacy"
        ),
        "model_conditioning_family": conditioning_family,
        "model_conditioning_acceleration": conditioning.acceleration,
        "conditioning_relation_to_physical_sampling": relation,
        "documented_physical_sampling_family_match": physical_documented,
        "documented_model_conditioning_family_match": family_documented,
        "documented_training_family_match": family_documented,
        "documented_training_acceleration_match": acceleration_documented,
        "documented_training_acquisition_match": acquisition_documented,
        "checkpoint_training_exposure": "documented" if documented else "undocumented",
        "runtime_condition_resolution": runtime_resolution,
        "upstream_condition_unmatched": mask_idx == -1 or acc_idx == -1,
        "resolved_mask_idx": mask_idx,
        "resolved_acc_idx": acc_idx,
        "resolved_acq_idx": acq_idx,
        "resolved_label_value": resolved_label,
        "resolved_label_documented_collision": collision if not documented else None,
        "neural_label_conditioning_active": spec.label_conditioning,
        "neural_acceleration_conditioning_active": spec.label_conditioning,
        "neural_acceleration_conditioning_changed": (
            spec.label_conditioning and conditioning.acceleration != nominal
        ),
        "resolved_dc_weight_index": dc_index,
        "resolved_dc_weight_family": SDUM_RUNTIME_MASK_FAMILIES[dc_index],
        "negative_index_dc_fallback": mask_idx == -1,
        "silent_fallback": False,
        "warnings": warning_messages,
        "measured_actual_used_as_conditioning": False,
    }


def released_cascade_count(model: Any) -> int:
    """Read the released cascade ModuleList without architecture guessing."""

    recon_model = getattr(model, "recon_model", None)
    cascades = getattr(recon_model, "cascades", None)
    if cascades is None:
        cascades = getattr(model, "cascades", None)
    if cascades is None:
        raise RuntimeError("released SDUM model exposes no expected cascades ModuleList")
    return int(len(cascades))


def canonical_to_sdum_private(
    kspace_s_t_kx_ky_c: np.ndarray, mask_s_t_kx_ky: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-only map ``[S,T,Kx,Ky,C] -> [T,S,C,H,W]`` with H=Kx/W=Ky."""

    kspace = np.asarray(kspace_s_t_kx_ky_c)
    mask = np.asarray(mask_s_t_kx_ky)
    if kspace.ndim != 5:
        raise ValueError("SDUMAdapter k-space must be canonical [S,T,Kx,Ky,C]")
    if mask.ndim != 4:
        raise ValueError("SDUMAdapter mask must be canonical [S,T,Kx,Ky]")
    if mask.shape != kspace.shape[:4]:
        raise ValueError(
            f"SDUMAdapter mask shape {mask.shape} does not match k-space {kspace.shape[:4]}"
        )
    return np.transpose(kspace, (1, 0, 4, 2, 3)), np.transpose(mask, (1, 0, 2, 3))


def sdum_rss_to_canonical(stage_m_t_s_h_w: np.ndarray) -> np.ndarray:
    stage_m = np.asarray(stage_m_t_s_h_w)
    if stage_m.ndim != 4:
        raise ValueError("SDUM Stage-M output must have private [T,S,H,W] shape")
    return np.transpose(stage_m, (1, 0, 2, 3))


def strict_mask_support(mask_t_s_h_w: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_t_s_h_w)
    if mask.ndim != 4:
        raise ValueError("private SDUM mask must have [T,S,H,W] shape")
    if not np.isfinite(mask).all():
        raise ValueError("SDUMAdapter mask contains NaN or Inf")
    return mask > 0


def expand_private_mask(mask_t_s_h_w: np.ndarray, coils: int) -> np.ndarray:
    """Document/test the private broadcast to ``[T,S,C,H,W,2]``."""

    support = strict_mask_support(mask_t_s_h_w)
    return np.broadcast_to(support[:, :, None, :, :, None], (*support.shape[:2], coils, *support.shape[2:], 2))


def cyclic_temporal_indices(center: int, total_frames: int) -> list[int]:
    if total_frames < 1:
        raise ValueError("SDUM temporal input must contain at least one frame")
    return [int((center + offset) % total_frames) for offset in (-2, -1, 0, 1, 2)]


def complex_zscore_reference(value_c_h_w: np.ndarray) -> tuple[np.ndarray, complex, float]:
    """NumPy statement of the released formula; production calls released torch code."""

    value = np.asarray(value_c_h_w)
    if value.ndim != 3 or not np.iscomplexobj(value):
        raise ValueError("normalization reference expects complex [C,H,W]")
    mean = np.mean(value)
    sigma = float(np.sqrt(np.mean(np.abs(value - mean) ** 2)))
    return (value - mean) / sigma, complex(mean), sigma


@contextmanager
def temporary_torch_threads(torch_module: Any, count: int = 1) -> Iterator[dict[str, int]]:
    """Temporarily reproduce official worker reduction order, restoring on error."""

    original = int(torch_module.get_num_threads())
    evidence = {"before": original, "during": int(count), "after": original}
    torch_module.set_num_threads(int(count))
    try:
        yield evidence
    finally:
        torch_module.set_num_threads(original)
        evidence["after"] = int(torch_module.get_num_threads())


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


DebugHook = Callable[[str, int, Any], None]


class _DefaultSDUMRuntime:
    """Lazy, source-pinned runtime; all NVIDIA tensor conventions stay private."""

    def __init__(
        self,
        *,
        sdum_source_root: Path,
        checkpoint: Path,
        config: Path | None,
        device: str,
        variant: str = "small",
    ) -> None:
        self.source_root = Path(sdum_source_root)
        self.checkpoint = Path(checkpoint)
        self.config_path = Path(config) if config is not None else None
        self.device_requested = str(device)
        self.variant = sdum_variant_spec(variant)
        self.torch: Any = None
        self.device: Any = None
        self.config: Any = None
        self.model: Any = None
        self.upstream: dict[str, Any] = {}
        self.provenance: dict[str, Any] = {}

    def _import_released_symbols(self, scripts_dir: Path) -> dict[str, Any]:
        scripts_text = str(scripts_dir)
        if scripts_text not in sys.path:
            sys.path.insert(0, scripts_text)
        try:
            from models.latent_recon import create_mri_recon_model  # type: ignore
            from monai.apps.reconstruction.complex_utils import (  # type: ignore
                complex_abs,
                convert_to_tensor_complex,
            )
            from monai.data.fft_utils import ifftn_centered  # type: ignore
            from mri_data.data_utils import crop_k_space, rearrange_mri_data  # type: ignore
            from utils import complex_zscore, load_config, load_net, windowed_input  # type: ignore
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(f"dedicated SDUM environment is incomplete: {exc}") from exc
        released = {
            "create_mri_recon_model": create_mri_recon_model,
            "complex_abs": complex_abs,
            "convert_to_tensor_complex": convert_to_tensor_complex,
            "ifftn_centered": ifftn_centered,
            "crop_k_space": crop_k_space,
            "rearrange_mri_data": rearrange_mri_data,
            "complex_zscore": complex_zscore,
            "load_config": load_config,
            "load_net": load_net,
            "windowed_input": windowed_input,
        }
        for name in (
            "create_mri_recon_model",
            "crop_k_space",
            "rearrange_mri_data",
            "complex_zscore",
            "load_config",
            "load_net",
            "windowed_input",
        ):
            source = Path(inspect.getfile(released[name])).resolve()
            if not source.is_relative_to(scripts_dir):
                raise RuntimeError(
                    f"import collision: released symbol {name} came from {source}, not {scripts_dir}"
                )
        return released

    def load(self) -> dict[str, Any]:
        commit = verify_sdum_source(self.source_root)
        source_root = self.source_root.expanduser().resolve()
        checkpoint = self.checkpoint.expanduser().resolve()
        checkpoint_sha = verify_sdum_checkpoint(
            checkpoint, variant=self.variant.name
        )
        config_path = (
            self.config_path.expanduser().resolve()
            if self.config_path is not None
            else source_root / "configs" / self.variant.config_name
        )
        if not config_path.is_file():
            raise RuntimeError(f"SDUM config does not exist: {config_path}")
        try:
            import torch
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(f"dedicated SDUM environment cannot import PyTorch: {exc}") from exc
        device = torch.device(self.device_requested)
        if device.type != "cuda":
            raise RuntimeError("released SDUM parity requires CUDA BF16; device must be cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        upstream = self._import_released_symbols(source_root / "scripts")
        config = upstream["load_config"](config_path)
        if config is None:
            raise RuntimeError(f"released load_config failed for {config_path}")
        config.ddp = False
        config.debug = False
        config.model_ckpt = checkpoint
        config.is_multi_coil = True
        if str(config.model_variant) != self.variant.model_variant:
            raise RuntimeError(
                f"SDUM {self.variant.name.title()} requires released model_variant "
                f"{self.variant.model_variant}, found {config.model_variant}"
            )
        if int(config.num_cascades) != self.variant.cascade_count:
            raise RuntimeError(
                f"SDUM {self.variant.name.title()} requires "
                f"{self.variant.cascade_count} cascades, found {config.num_cascades}"
            )
        if self.variant.name == "small":
            if tuple(int(value) for value in config.accelerations) != SMALL_ACCELERATION_CATEGORIES:
                raise RuntimeError(
                    "released Small acceleration categories differ from [8,16,24]"
                )
            if tuple(str(value) for value in config.train_mask_types) != SMALL_MASK_FAMILIES:
                raise RuntimeError(
                    "released Small mask-family vocabulary differs from the pinned audit"
                )
        if self.variant.name in ("base", "large"):
            if tuple(int(value) for value in config.accelerations) != BASE_ACCELERATION_CATEGORIES:
                raise RuntimeError(
                    f"released {self.variant.name.title()} acceleration categories "
                    "differ from [8,16,24]"
                )
            if tuple(str(value) for value in config.train_mask_types) != BASE_MASK_FAMILIES:
                raise RuntimeError(
                    f"released {self.variant.name.title()} mask-family vocabulary "
                    "differs from the pinned audit"
                )
            if tuple(str(value) for value in config.acq_types) != BASE_ACQUISITIONS:
                raise RuntimeError(
                    f"released {self.variant.name.title()} acquisition vocabulary "
                    "differs from the pinned audit"
                )
            if not bool(config.label_cond):
                raise RuntimeError(
                    f"released {self.variant.name.title()} config must enable label conditioning"
                )
        if int(config.num_frames) != NUM_TEMPORAL_FRAMES or int(config.batch_size) != 1:
            raise RuntimeError("released parity requires num_frames=5 and forward batch_size=1")
        if not bool(config.amp):
            raise RuntimeError("released SDUM parity requires CUDA BF16 autocast")
        model = upstream["create_mri_recon_model"](config).to(device)
        loaded = upstream["load_net"](
            model,
            checkpoint,
            device,
            is_ddp=False,
            resume_rng_state=config.resume_rng_state,
        )
        model = loaded[0]
        model.eval()
        if model.training:
            raise RuntimeError("SDUM model remained in training mode after eval()")
        cascade_count = released_cascade_count(model)
        if cascade_count != self.variant.cascade_count:
            raise RuntimeError(
                f"loaded {self.variant.name.title()} model exposes {cascade_count} "
                f"cascades, expected {self.variant.cascade_count}"
            )
        self.torch = torch
        self.device = device
        self.config = config
        self.model = model
        self.upstream = upstream
        self.provenance = {
            "repository": SDUM_REPOSITORY,
            "source_root": str(source_root),
            "source_commit_expected": SDUM_COMMIT,
            "source_commit_actual": commit,
            "source_checkout_clean": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_role": self.variant.checkpoint_role,
            "variant": self.variant.name,
            "config_path": str(config_path),
            "config_name": config_path.name,
            "model_variant": str(config.model_variant),
            "cascade_count": cascade_count,
            "label_conditioning": bool(getattr(config, "label_cond", False)),
            "python": sys.version,
            "numpy": _distribution_version("numpy"),
            "torch": _distribution_version("torch"),
            "monai": _distribution_version("monai"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "precision": "CUDA BF16 autocast",
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        return dict(self.provenance)

    def _state_snapshot(self) -> dict[str, Any]:
        parameters = list(self.model.parameters())
        return {
            "parameter_versions": [int(getattr(item, "_version", -1)) for item in parameters],
            "parameters_with_grad": int(sum(item.grad is not None for item in parameters)),
        }

    def _preprocess(
        self, kspace_t_s_c_h_w: np.ndarray, mask_t_s_h_w: np.ndarray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self.torch
        support = strict_mask_support(mask_t_s_h_w)
        kspace = self.upstream["convert_to_tensor_complex"](
            np.asarray(kspace_t_s_c_h_w, dtype=np.complex64)
        )
        mask_singleton = torch.as_tensor(support.astype(np.float32))[:, :, None, :, :, None]
        masked_kspace = mask_singleton * kspace
        mask_expanded = mask_singleton.expand(masked_kspace.shape)
        zero_filled = self.upstream["ifftn_centered"](
            masked_kspace, spatial_dims=2, is_complex=True
        )
        flat_image, flat_mask = self.upstream["rearrange_mri_data"](
            [zero_filled[None, ...], mask_expanded[None, ...]], self.config
        )
        with temporary_torch_threads(torch, 1) as thread_evidence:
            normalized, mean, std = self.upstream["complex_zscore"](
                flat_image, dim=[1, 2, 3]
            )
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(
                "SDUM preprocessing produced nonfinite normalized data; zero variance or invalid input is likely"
            )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("SDUM preprocessing produced nonfinite normalization statistics")
        if bool((std <= 0).any()):
            raise ValueError("SDUM preprocessing requires strictly positive per-frame sigma")
        t, s, c, h, w = (int(item) for item in kspace_t_s_c_h_w.shape)
        private = {
            "normalized": normalized.contiguous(),
            "mask_flat": flat_mask.contiguous(),
            "mean": mean.contiguous(),
            "std": std.contiguous(),
            "final_shape": [t, s, c, h, w],
        }
        evidence = {
            "torch_num_threads": dict(thread_evidence),
            "thread_count_restored": thread_evidence["after"] == thread_evidence["before"],
            "mask_threshold": "mask > 0",
            "mask_soft_weights_used": False,
            "ifft": "released MONAI centered orthonormal IFFT (private)",
            "common_cmrbench_fft_used": False,
            "normalization": "complex mean/sigma over C,H,W; unbiased=false; no epsilon",
        }
        return private, evidence

    def infer_volume(
        self,
        kspace_t_s_c_h_w: np.ndarray,
        mask_t_s_h_w: np.ndarray,
        conditioning: SDUMConditioning,
        debug_hook: DebugHook | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("SDUM runtime has not been loaded")
        torch = self.torch
        private, preprocessing = self._preprocess(kspace_t_s_c_h_w, mask_t_s_h_w)
        normalized = private["normalized"]
        flat_mask = private["mask_flat"]
        mean = private["mean"].to(self.device)
        std = private["std"].to(self.device)
        final_shape = private["final_shape"]
        mask_type, acceleration, acquisition = conditioning.as_model_arguments()
        acceleration_t = torch.tensor(acceleration, dtype=torch.int64, device="cpu")
        conditioning_arguments = {
            "mask_type": {
                "python_type": "str",
                "value": mask_type,
            },
            "acc_factor": {
                "python_type": "torch.Tensor",
                "is_tensor": True,
                "shape": list(acceleration_t.shape),
                "dtype": str(acceleration_t.dtype),
                "device": str(acceleration_t.device),
                "value": int(acceleration_t.item()),
                "source_faithful_official_representation": True,
            },
            "acq_type": {
                "python_type": "str",
                "value": acquisition,
            },
        }
        outputs: list[np.ndarray] = []
        windows: list[list[int]] = []
        before = self._state_snapshot()
        old_cudnn = (
            bool(torch.backends.cudnn.deterministic),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.backends.cudnn.enabled),
        )
        self.model.eval()
        start = perf_counter()
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True
            with torch.no_grad():
                for flat_index in range(int(normalized.shape[0])):
                    model_input, window_index = self.upstream["windowed_input"](
                        normalized, [flat_index], final_shape, num_frames=NUM_TEMPORAL_FRAMES
                    )
                    model_mask_float = torch.Tensor(flat_mask[window_index])
                    model_mask = model_mask_float.bool()
                    windows.append([int(item) for item in window_index[0].cpu().tolist()])
                    if debug_hook is not None:
                        debug_hook("I", flat_index, model_input)
                        debug_hook("J", flat_index, model_mask)
                    model_input_cuda = model_input.to(self.device)
                    model_mask_cuda = model_mask.to(self.device)
                    with torch.amp.autocast(
                        "cuda", torch.bfloat16, enabled=bool(self.config.amp)
                    ):
                        model_output = self.model(
                            model_input_cuda,
                            model_mask_cuda,
                            mask_type,
                            acceleration_t,
                            acquisition,
                        )
                    center = model_output[:, NUM_TEMPORAL_FRAMES // 2]
                    if debug_hook is not None:
                        debug_hook("L1", flat_index, center)
                    denormalized = center * std[[flat_index]] + mean[[flat_index]]
                    magnitude = self.upstream["complex_abs"](
                        self.upstream["crop_k_space"](
                            denormalized, (final_shape[-2], final_shape[-1])
                        )
                    )
                    outputs.append(magnitude.data.cpu().numpy())
        finally:
            torch.backends.cudnn.deterministic = old_cudnn[0]
            torch.backends.cudnn.benchmark = old_cudnn[1]
            torch.backends.cudnn.enabled = old_cudnn[2]
        runtime_seconds = perf_counter() - start
        after = self._state_snapshot()
        state_unchanged = (
            before["parameter_versions"] == after["parameter_versions"]
            and before["parameters_with_grad"] == 0
            and after["parameters_with_grad"] == 0
        )
        if not state_unchanged:
            raise RuntimeError("SDUM parameters changed or gradients appeared during inference")
        restored = self.upstream["rearrange_mri_data"](
            [np.vstack(outputs)],
            self.config,
            is_complex=False,
            reverse=True,
            num_slices=final_shape[-4],
            num_coils=final_shape[-3],
            temporal_shuffle=None,
        )
        stage_m = np.sqrt(np.sum(restored[0] ** 2, axis=-3))
        if stage_m.shape != tuple(final_shape[:2] + final_shape[-2:]):
            raise RuntimeError(
                f"SDUM Stage-M shape {stage_m.shape} does not match expected "
                f"{tuple(final_shape[:2] + final_shape[-2:])}"
            )
        if not np.isfinite(stage_m).all():
            raise RuntimeError("SDUM Stage-M reconstruction contains NaN or Inf")
        return stage_m, {
            "runtime_seconds": runtime_seconds,
            "preprocessing": preprocessing,
            "window_indices_flat": windows,
            "model_eval_mode": not self.model.training,
            "inference_context": "torch.no_grad",
            "autocast": "CUDA bfloat16",
            "forward_batch_size": 1,
            "model_calls": len(outputs),
            "conditioning_arguments": conditioning_arguments,
            "state_verification": {
                "parameter_version_counters_unchanged": before["parameter_versions"] == after["parameter_versions"],
                "parameters_with_grad_before": before["parameters_with_grad"],
                "parameters_with_grad_after": after["parameters_with_grad"],
                "no_parameter_updates_or_gradients_verified": state_unchanged,
            },
            "cudnn_state_restored": (
                bool(torch.backends.cudnn.deterministic),
                bool(torch.backends.cudnn.benchmark),
                bool(torch.backends.cudnn.enabled),
            )
            == old_cudnn,
        }


RuntimeFactory = Callable[..., Any]


class SDUMAdapter(ReconstructionModel):
    """Released SDUM adapter for an explicitly selected validated variant."""

    def __init__(
        self,
        *,
        sdum_source_root: str | Path,
        checkpoint: str | Path,
        config: str | Path | None = None,
        device: str = "cuda",
        mask_type: str | None = None,
        acceleration: int | None = None,
        acquisition: str = "Cine",
        sampling_condition: Mapping[str, Any] | None = None,
        variant: str = "small",
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.sdum_source_root = Path(sdum_source_root)
        self.checkpoint = Path(checkpoint)
        self.config = Path(config) if config is not None else None
        self.device = str(device)
        self.variant = sdum_variant_spec(variant)
        if (mask_type is None) != (acceleration is None):
            raise ValueError("legacy SDUM mask_type and acceleration must be specified together")
        if sampling_condition is not None and mask_type is not None:
            raise ValueError(
                "SDUM conditioning must use either sampling_condition or legacy "
                "mask_type/acceleration, not both"
            )
        if sampling_condition is not None and not isinstance(sampling_condition, Mapping):
            raise ValueError("SDUM sampling_condition must be a mapping")
        self.default_acquisition = str(acquisition)
        self.sampling_condition = (
            dict(sampling_condition) if sampling_condition is not None else None
        )
        self.legacy_conditioning = (
            SDUMConditioning(
                mask_type=str(mask_type),
                acceleration=int(acceleration),
                acquisition=self.default_acquisition,
            )
            if mask_type is not None and acceleration is not None
            else None
        )
        self._runtime_factory = runtime_factory or _DefaultSDUMRuntime
        self._runtime: Any = None
        self._load_provenance: dict[str, Any] = {}

    def load(self) -> None:
        if self._runtime is not None:
            return
        runtime = self._runtime_factory(
            sdum_source_root=self.sdum_source_root,
            checkpoint=self.checkpoint,
            config=self.config,
            device=self.device,
            variant=self.variant.name,
        )
        provenance = runtime.load()
        if not isinstance(provenance, dict):
            raise RuntimeError("SDUM runtime load() must return provenance metadata")
        self._runtime = runtime
        self._load_provenance = dict(provenance)

    @staticmethod
    def _validate_input(input: ReconstructionInput) -> tuple[np.ndarray, np.ndarray]:
        kspace = np.asarray(input.kspace)
        mask = np.asarray(input.mask)
        if kspace.ndim != 5:
            raise ValueError("SDUMAdapter requires k-space [S,T,Kx,Ky,C]")
        if mask.ndim != 4:
            raise ValueError("SDUMAdapter requires mask [S,T,Kx,Ky]")
        if mask.shape != kspace.shape[:4]:
            raise ValueError("SDUMAdapter k-space/mask dimensions do not match")
        if not np.iscomplexobj(kspace):
            raise ValueError("SDUMAdapter requires complex canonical k-space")
        if kspace.shape[-1] < 1 or kspace.shape[2] < 1 or kspace.shape[3] < 1:
            raise ValueError("SDUMAdapter requires non-empty coil and spatial dimensions")
        if not (np.isfinite(kspace.real).all() and np.isfinite(kspace.imag).all()):
            raise ValueError("SDUMAdapter k-space contains NaN or Inf")
        strict_mask_support(mask)
        return np.asarray(kspace, dtype=np.complex64), mask

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        return self._reconstruct(input, debug_hook=None)

    def _reconstruct_for_regression(
        self, input: ReconstructionInput, debug_hook: DebugHook
    ) -> ReconstructionResult:
        """Private official-parity hook; private I/J/L1 never enter public results."""

        return self._reconstruct(input, debug_hook=debug_hook)

    def _reconstruct(
        self, input: ReconstructionInput, debug_hook: DebugHook | None
    ) -> ReconstructionResult:
        kspace, mask = self._validate_input(input)
        conditioning, conditioning_metadata = resolve_sdum_conditioning(
            input,
            sampling_condition=self.sampling_condition,
            legacy_conditioning=self.legacy_conditioning,
            default_acquisition=self.default_acquisition,
        )
        model_sampling = sdum_sampling_compatibility(
            input,
            conditioning,
            variant=self.variant.name,
            conditioning_metadata=conditioning_metadata,
        )
        if self._runtime is None:
            self.load()
        slices, phases, height, width, coils = kspace.shape
        temporal_warning = None
        if phases < NUM_TEMPORAL_FRAMES:
            temporal_warning = (
                f"T={phases} produces repeated cyclic indices in the five-frame SDUM window; "
                "execution is permitted but in-distribution support is not claimed"
            )
            warnings.warn(temporal_warning, RuntimeWarning, stacklevel=2)
        private_kspace, private_mask = canonical_to_sdum_private(kspace, mask)
        try:
            stage_m, runtime_record = self._runtime.infer_volume(
                private_kspace, private_mask, conditioning, debug_hook=debug_hook
            )
        except Exception as exc:
            raise UpstreamModelInferenceError("sdum", exc) from exc
        reconstruction = sdum_rss_to_canonical(stage_m)
        expected = (slices, phases, height, width)
        if reconstruction.shape != expected:
            raise UpstreamModelInferenceError(
                "sdum",
                RuntimeError(
                    f"canonical output shape {reconstruction.shape} does not match {expected}"
                ),
            )
        if np.iscomplexobj(reconstruction):
            raise UpstreamModelInferenceError(
                "sdum", RuntimeError("output must be real RSS magnitude, not complex")
            )
        if not np.isfinite(reconstruction).all():
            raise UpstreamModelInferenceError(
                "sdum", RuntimeError("output contains NaN or Inf")
            )
        preflight = preflight_conditioning(
            conditioning, variant=self.variant.name
        )
        transforms = [
            TransformRecord(
                "sdum_sampling_compatibility",
                dict(model_sampling),
            ),
            TransformRecord(
                "canonical_to_sdum_private_axes",
                {
                    "input": "[S,T,Kx,Ky,C]",
                    "private": "[T,S,C,H,W]",
                    "semantic_mapping": "H=Kx, W=Ky",
                    "axis_inference_from_size": False,
                },
            ),
            TransformRecord(
                "sdum_private_mask_preprocessing",
                {
                    "input": "[S,T,Kx,Ky]",
                    "private": "[T,S,H,W] -> broadcast coil/real-imag -> flatten T/S -> five-frame",
                    "threshold": "mask > 0",
                    "model_call_dtype": "bool",
                    "soft_weighting": False,
                },
            ),
            TransformRecord(
                "sdum_released_source_preprocessing",
                {
                    "ifft": "released MONAI centered orthonormal IFFT",
                    "common_cmrbench_fft_used": False,
                    "normalization": "per-frame complex mean/sigma over C,H,W; no epsilon",
                    "normalization_torch_threads": 1,
                },
            ),
            TransformRecord(
                "sdum_five_frame_cyclic_inference",
                {
                    "window": "[t-2,t-1,t,t+1,t+2] modulo T",
                    "forward_batch_size": 1,
                    "prediction": "center frame only",
                    "short_T_warning": temporal_warning,
                },
            ),
            TransformRecord(
                "sdum_denormalize_magnitude_rss",
                {
                    "denormalization": "center-frame sigma and complex mean",
                    "coil_output": "per-coil magnitude",
                    "combination": "root-sum-of-squares",
                    "final_shape": list(expected),
                    "complex_phase_fabricated": False,
                },
            ),
        ]
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=reconstruction,
            model_name="sdum",
            runtime_seconds=float(runtime_record["runtime_seconds"]),
            transforms=transforms,
            metadata={
                "adapter_version": SDUM_ADAPTER_VERSION,
                "inference_mode": INFERENCE_MODE,
                "direct_released_inference": True,
                "model_eval_mode": runtime_record["model_eval_mode"],
                "inference_context": runtime_record["inference_context"],
                "autocast": runtime_record["autocast"],
                "adaptation_performed": False,
                "fine_tuning_called": False,
                "optimizer_created": False,
                "backward_called": False,
                "conditioning": asdict(conditioning),
                "sampling_condition_request": dict(self.sampling_condition or {}),
                "model_sampling": dict(model_sampling),
                "variant": self.variant.name,
                "conditioning_preflight": preflight,
                "conditioning_model_arguments": runtime_record.get(
                    "conditioning_arguments",
                    {
                        "mask_type": {"python_type": "str", "value": conditioning.mask_type},
                        "acc_factor": {
                            "python_type": "torch.Tensor",
                            "is_tensor": True,
                            "shape": [],
                            "dtype": "torch.int64",
                            "device": "cpu",
                            "value": conditioning.acceleration,
                            "source_faithful_official_representation": True,
                        },
                        "acq_type": {"python_type": "str", "value": conditioning.acquisition},
                    },
                ),
                "external_smaps_present": input.sensitivity_maps is not None,
                "external_smaps_used": False,
                "output_representation": OUTPUT_REPRESENTATION,
                "output_is_complex": False,
                "output_axis_order": "[S,T,X,Y]",
                "challenge_stage_n_applied": False,
                "temporal_warning": temporal_warning,
                "runtime": runtime_record,
                "provenance": dict(self._load_provenance),
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="sdum",
            model_version=SDUM_ADAPTER_VERSION,
            checkpoint=str(self.checkpoint),
            checkpoint_identifier=self.variant.checkpoint_sha256,
            git_commit=SDUM_COMMIT,
            parameter_count=self._load_provenance.get("parameter_count"),
            device=self.device,
            precision="CUDA BF16 autocast",
            metadata={
                "repository": SDUM_REPOSITORY,
                "inference_mode": INFERENCE_MODE,
                "output_representation": OUTPUT_REPRESENTATION,
                "conditioning_request": (
                    dict(self.sampling_condition)
                    if self.sampling_condition is not None
                    else (
                        {"mode": "explicit_legacy", **asdict(self.legacy_conditioning)}
                        if self.legacy_conditioning is not None
                        else {"mode": "auto", "acquisition": self.default_acquisition}
                    )
                ),
                "variant": self.variant.name,
                "cascade_count_expected": self.variant.cascade_count,
                "label_conditioning": self.variant.label_conditioning,
                "external_smaps_used": False,
                "small_checkpoint_role": "engineering / adapter parity smoke",
                "base_checkpoint_role": "primary released scientific model",
                "large_checkpoint_role": "released model under validation",
                **self._load_provenance,
                "checkpoint_role": self.variant.checkpoint_role,
            },
        )
