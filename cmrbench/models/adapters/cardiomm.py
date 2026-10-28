"""Source-faithful CardioMM adapter with model-owned sampling validation."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.errors import UpstreamModelInferenceError
from cmrbench.models.input import ReconstructionInput
from cmrbench.experiment.calibration import resolve_requested_calibration_width
from cmrbench.models.result import ReconstructionResult, TransformRecord
from cmrbench.sampling.result import (
    calculate_actual_acceleration,
    detect_complete_central_acs,
)


CARDIOMM_ADAPTER_VERSION = "0.3.0"
CARDIOMM_REPOSITORY = "https://github.com/wangziblake/CardioMM_MMCMR-427K.git"
CARDIOMM_COMMIT = "abb6f8a4e293426736467c9859a5f5755c38c8aa"
CHECKPOINT_FILENAME = "epochbest.ckpt"
MODEL_ARCHITECTURE = "CardioMM_SEM"
NUM_CASCADES = 10
NUM_LOW_FREQUENCIES = 20
INFERENCE_BATCH_SIZE = 1
TEXT_CONDITIONING_CACHE_DEFAULT = True
INFERENCE_CONTEXT_DEFAULT = "no_grad"
MASK_SEMANTIC = "uniform"
NOMINAL_ACCELERATION = 8
SOURCE_IDENTITY = "pinned CardioMM ktMaskGenerator Uniform"
CARDIOMM_NOMINAL_ACCELERATIONS = (4, 8, 16, 24)
CARDIOMM_SAMPLING_METHODS = {
    "uniform": {
        "source_pattern": "Uniform",
        "semantic_family": "uniform",
        "acs_geometry": "centered_ky_strip",
        "temporal_behavior": "static",
    },
    "kt_uniform": {
        "source_pattern": "ktUniform",
        "semantic_family": "uniform",
        "acs_geometry": "centered_ky_strip",
        "temporal_behavior": "dynamic_allowed",
    },
    "kt_gaussian": {
        "source_pattern": "ktGaussian",
        "semantic_family": "random",
        "acs_geometry": "centered_ky_strip",
        "temporal_behavior": "dynamic_allowed",
    },
    "kt_radial": {
        "source_pattern": "ktRadial",
        "semantic_family": "radial",
        "acs_geometry": "centered_square",
        "temporal_behavior": "dynamic_allowed",
    },
    "random": {
        "source_pattern": "Random",
        "semantic_family": "random",
        "acs_geometry": "centered_ky_strip",
        "temporal_behavior": "static_single_source_call",
    },
    "radial": {
        "source_pattern": "Radial",
        "semantic_family": "radial",
        "acs_geometry": "centered_square",
        "temporal_behavior": "static_single_source_call",
    },
    "equispaced": {
        "source_pattern": "Equispaced",
        "semantic_family": "uniform",
        "acs_geometry": "centered_ky_strip",
        "temporal_behavior": "static",
    },
}
CARDIOMM_EXTERNAL_PATTERN_ALIASES = {
    "uniform": ("Uniform", "uniform", "supported", "centered_ky_strip", "static"),
    "ktuniform": (
        "ktUniform",
        "uniform",
        "ood_valid",
        "centered_ky_strip",
        "dynamic_allowed",
    ),
    "random": ("Random", "random", "ood_valid", "centered_ky_strip", "dynamic_allowed"),
    "gaussian": (
        "Gaussian",
        "random",
        "ood_valid",
        "centered_ky_strip",
        "dynamic_allowed",
    ),
    "ktgaussian": (
        "ktGaussian",
        "random",
        "supported",
        "centered_ky_strip",
        "dynamic_allowed",
    ),
    "radial": ("Radial", "radial", "ood_valid", "centered_square", "dynamic_allowed"),
    "ktradial": (
        "ktRadial",
        "radial",
        "supported",
        "centered_square",
        "dynamic_allowed",
    ),
}
METADATA_FIELDS = (
    "lifespan",
    "field_strength",
    "vendor",
    "scanner",
    "modality",
    "view",
)
UNDERSAMPLING_PROMPT = "Undersampling: 8x uniform."

MODEL_CONSTRUCTOR_ARGUMENTS = {
    "num_cascades": NUM_CASCADES,
    "num_adj_slices": 1,
    "n_feat0": 48,
    "feature_dim": [72, 96, 120],
    "prompt_dim": [24, 48, 72],
    "sens_n_feat0": 24,
    "sens_feature_dim": [36, 48, 60],
    "sens_prompt_dim": [12, 24, 36],
    "len_prompt": [3, 3, 3],
    "prompt_size": [64, 32, 16],
    "n_enc_cab": [2, 3, 3],
    "n_dec_cab": [2, 2, 3],
    "n_skip_cab": [1, 1, 1],
    "n_bottleneck_cab": 3,
    "no_use_ca": False,
}
LANGUAGE_HEAD_ARGUMENTS = {
    "llm_model_dim": 512,
    "llm_embd_dim": 256,
    "llm_nclasses": 3,
}

RuntimeFactory = Callable[..., Any]


def build_metadata_prompt(metadata: Mapping[str, Any]) -> str:
    """Construct the released prompt without fabricating missing metadata."""

    normalized = {
        field: str(metadata.get(field, "")).strip() for field in METADATA_FIELDS
    }
    missing = [field for field, value in normalized.items() if not value]
    if missing:
        raise ValueError(
            "CardioMM configuration error: missing required acquisition metadata "
            "fields: " + ", ".join(missing)
        )
    prompt = (
        "{} cardiac MRI reconstruction.Vendor: {}, {}, {}. Modality: {}, {}.".format(
            normalized["lifespan"],
            normalized["field_strength"],
            normalized["vendor"],
            normalized["scanner"],
            normalized["modality"],
            normalized["view"],
        )
    )
    return prompt


def build_undersampling_prompt(nominal_acceleration: float, family: str) -> str:
    value = float(nominal_acceleration)
    if not value.is_integer() or value < 1:
        raise ValueError("CardioMM nominal acceleration must be a positive integer")
    semantic = str(family).lower()
    if semantic not in {"uniform", "random", "radial"}:
        raise ValueError(
            "CardioMM sampling has no truthful conditioning semantic; expected "
            "uniform, random, or radial"
        )
    return "Undersampling: {}x {}.".format(int(value), semantic)


def _normalized_pattern_token(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _sampling_ncalib(parameters: Mapping[str, Any]) -> Optional[int]:
    direct = parameters.get("ncalib")
    generated = parameters.get("generator_parameters")
    if direct is None and isinstance(generated, Mapping):
        direct = generated.get("ncalib")
    if direct is None:
        return None
    try:
        return int(direct)
    except (TypeError, ValueError):
        return None


def _physical_cardiomm_semantic(sampling: Any) -> tuple[Optional[str], str]:
    method = str(sampling.method).lower()
    parameters = dict(sampling.parameters)
    if method == "external_mask":
        alias = CARDIOMM_EXTERNAL_PATTERN_ALIASES.get(
            _normalized_pattern_token(parameters.get("pattern", parameters.get("family")))
        )
        return (alias[1], "source_alias") if alias is not None else (None, "ambiguous")
    mapping = {
        "uniform": ("uniform", "native_match"),
        "equispaced": ("uniform", "source_alias"),
        "kt_uniform": ("uniform", "source_alias"),
        "random": ("random", "native_match"),
        "kt_gaussian": ("random", "source_alias"),
        "radial": ("radial", "native_match"),
        "kt_radial": ("radial", "source_alias"),
    }
    return mapping.get(method, (None, "ambiguous"))


def _resolve_cardiomm_conditioning(
    input: ReconstructionInput,
    sampling_condition: Optional[Mapping[str, Any]],
    default_num_low_frequencies: Optional[int],
) -> Dict[str, Any]:
    sampling = input.sampling
    if sampling is None:
        raise ValueError("CardioMM requires SamplingResult provenance")
    requested = sampling.requested_acceleration
    if requested is None or not float(requested).is_integer() or float(requested) < 1:
        raise ValueError("CardioMM requires a positive integer physical nominal acceleration")
    physical_semantic, automatic_relation = _physical_cardiomm_semantic(sampling)
    request = dict(sampling_condition or {})
    inferred_mode = "explicit" if "mask_type" in request else "auto"
    mode = str(request.pop("mode", inferred_mode)).lower()
    acceleration = request.pop("acceleration", int(float(requested)))
    if isinstance(acceleration, bool) or not float(acceleration).is_integer() or float(acceleration) < 1:
        raise ValueError("CardioMM model conditioning acceleration must be a positive integer")
    acceleration = int(float(acceleration))
    num_low = request.pop("num_low_frequencies", default_num_low_frequencies)
    if num_low is not None:
        if isinstance(num_low, bool) or not float(num_low).is_integer() or float(num_low) < 0:
            raise ValueError("CardioMM num_low_frequencies must be a nonnegative integer or null")
        num_low = int(float(num_low))
    if mode == "auto":
        if "mask_type" in request:
            raise ValueError("CardioMM mode=auto must not specify mask_type")
        if physical_semantic is None:
            raise ValueError(
                "CardioMM physical sampling has no unambiguous automatic mask_type; "
                "set sampling_condition.mode=explicit and choose uniform, random, or radial"
            )
        semantic = physical_semantic
        relation = automatic_relation
    elif mode == "explicit":
        semantic = str(request.pop("mask_type", "")).lower()
        if semantic not in {"uniform", "random", "radial"}:
            raise ValueError(
                "CardioMM explicit sampling_condition.mask_type must be uniform, random, or radial"
            )
        relation = "user_forced"
    else:
        raise ValueError("CardioMM sampling_condition.mode must be auto or explicit")
    if request:
        raise ValueError(
            "unknown CardioMM sampling_condition keys: " + ", ".join(sorted(request))
        )
    return {
        "model_conditioning_mode": mode,
        "model_conditioning_family": semantic,
        "model_conditioning_acceleration": acceleration,
        "model_num_low_frequencies": num_low,
        "conditioning_relation_to_physical_sampling": relation,
        "physical_semantic_family": physical_semantic,
    }


def cardiomm_sampling_compatibility(
    input: ReconstructionInput,
    sampling_condition: Optional[Mapping[str, Any]] = None,
    *,
    default_num_low_frequencies: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate physical truth and independently resolve CardioMM text/ACS inputs."""

    sampling = input.sampling
    if sampling is None:
        raise ValueError("CardioMM requires SamplingResult provenance")
    method = str(sampling.method).lower()
    parameters = dict(sampling.parameters)
    calibration = resolve_requested_calibration_width(
        sampling, tuple(input.kspace.shape)
    )
    requested_width = (
        calibration.requested_calibration_width if calibration is not None else None
    )
    explicit_condition_width = (
        sampling_condition.get("num_low_frequencies")
        if sampling_condition is not None else None
    )
    if requested_width is not None:
        for label, explicit_width in (
            ("model num_low_frequencies", default_num_low_frequencies),
            ("sampling_condition.num_low_frequencies", explicit_condition_width),
        ):
            if explicit_width is not None and explicit_width != requested_width:
                raise ValueError(
                    f"CardioMM {label}={explicit_width} conflicts with "
                    f"sampling-resolved requested calibration width={requested_width} "
                    f"from {calibration.calibration_source}"
                )
        if default_num_low_frequencies is None and explicit_condition_width is None:
            default_num_low_frequencies = requested_width
    source_identity = parameters.get("source_identity") or parameters.get("generator")
    implementation = parameters.get("implementation")
    if implementation is None:
        raise ValueError("CardioMM requires explicit physical sampling implementation provenance")
    if method == "external_mask":
        missing = [
            key
            for key in ("source_identity", "nominal_acceleration")
            if parameters.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "CardioMM external_mask provenance is missing: " + ", ".join(missing)
            )
    declaration = CARDIOMM_SAMPLING_METHODS.get(method, {})
    source_pattern = str(
        declaration.get("source_pattern", parameters.get("pattern", sampling.method))
    )
    physical_acs_geometry = str(
        declaration.get("acs_geometry", parameters.get("acs_geometry", "measured_center"))
    )
    temporal_behavior = str(
        declaration.get("temporal_behavior", parameters.get("temporal_behavior", "measured"))
    )
    if default_num_low_frequencies is None and not sampling_condition:
        default_num_low_frequencies = _sampling_ncalib(parameters)
    condition = _resolve_cardiomm_conditioning(
        input, sampling_condition, default_num_low_frequencies
    )
    semantic = str(condition["model_conditioning_family"])
    requested = sampling.requested_acceleration
    if requested is None or not float(requested).is_integer():
        raise ValueError("CardioMM requires an integer nominal acceleration category")
    nominal = int(float(requested))
    ncalib = _sampling_ncalib(parameters)
    if ncalib is not None and ncalib < 1:
        raise ValueError("CardioMM declared ncalib provenance must be positive")
    model_num_low = condition["model_num_low_frequencies"]
    if model_num_low is None:
        model_num_low = 0
    if semantic == "radial" and model_num_low == 0:
        raise ValueError("CardioMM radial/2D ACS requires explicit positive num_low_frequencies")
    if ncalib is not None and model_num_low not in {0, ncalib}:
        raise ValueError(
            "CardioMM model num_low_frequencies must equal physical ncalib, or be 0/None "
            "for source-supported 1D automatic ACS detection"
        )
    binary = np.asarray(input.mask).astype(bool, copy=False)
    sampling_mask = np.asarray(sampling.mask)
    if sampling_mask.shape != binary.shape or not np.array_equal(
        sampling_mask.astype(bool, copy=False), binary
    ):
        raise ValueError("CardioMM input mask differs from SamplingResult.mask")
    measured = calculate_actual_acceleration(binary)
    if measured is None or sampling.actual_acceleration is None or not np.isclose(
        float(sampling.actual_acceleration), float(measured), rtol=0.0, atol=1e-12
    ):
        raise ValueError("CardioMM measured mask acceleration provenance is inconsistent")
    slices, phases, kx, ky = binary.shape
    acs_geometry = "centered_square" if semantic == "radial" else "centered_ky_strip"
    extraction_width = int(model_num_low or ncalib or 0)
    if extraction_width and (
        ky < extraction_width
        or (acs_geometry == "centered_square" and kx < extraction_width)
    ):
        raise ValueError(
            "CardioMM spatial dimensions cannot contain the requested internal "
            f"ACS extraction width={extraction_width}"
        )
    acs_detection = detect_complete_central_acs(binary)
    detected_extent = int(
        acs_detection[
            "detected_centered_square_extent"
            if acs_geometry == "centered_square"
            else "detected_centered_ky_strip_extent"
        ]
    )
    if ncalib is not None and detected_extent < ncalib:
        raise ValueError(
            "CardioMM declared ncalib provenance conflicts with the measurable "
            f"{acs_geometry}: ncalib={ncalib}, detected_extent={detected_extent}"
        )
    documented_family = semantic in {"uniform", "random", "radial"}
    model_nominal = int(condition["model_conditioning_acceleration"])
    documented_acceleration = model_nominal in CARDIOMM_NOMINAL_ACCELERATIONS
    normalized_pattern = _normalized_pattern_token(
        parameters.get("pattern", parameters.get("family", ""))
    )
    physical_documented = (
        (implementation == "canonical" and method == "uniform")
        or (
            implementation == "cardiomm"
            and method
            in {
                "uniform",
                "kt_uniform",
                "kt_gaussian",
                "kt_radial",
                "random",
                "radial",
                "equispaced",
            }
        )
        or (
            implementation == "cardiomm"
            and method == "external_mask"
            and normalized_pattern
            in {
                "uniform",
                "ktuniform",
                "ktgaussian",
                "ktradial",
                "random",
                "radial",
                "equispaced",
            }
        )
    )
    model_condition_documented = documented_family and documented_acceleration
    documented = (
        physical_documented
        and model_condition_documented
        and ncalib == NUM_LOW_FREQUENCIES
    )
    warning_messages = []
    if not documented:
        warning_messages.append(
            "CardioMM physical sampling and/or model condition is not documented by "
            "released examples/training categories; "
            "undocumented does not prove unseen"
        )
    if condition["conditioning_relation_to_physical_sampling"] == "user_forced":
        warning_messages.append(
            "physical sampling and CardioMM model-side mask_type were explicitly separated"
        )
    if not acs_detection["has_complete_central_acs"]:
        warning_messages.append(
            "CardioMM physical mask has no complete central ACS; its released "
            "KspaceACSExtractor will receive the mask and may form an empty/partial "
            "center crop for internal sensitivity estimation"
        )
    for message in warning_messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    legacy_supported = bool(
        documented
        and (
            (implementation == "canonical" and method == "uniform")
            or (
                method == "external_mask"
                and implementation == "cardiomm"
                and "cardiomm" in str(source_identity or "").lower()
                and _normalized_pattern_token(parameters.get("pattern", ""))
                in {"uniform", "ktgaussian", "ktradial"}
            )
            or (
                implementation == "cardiomm"
                and method in {"uniform", "kt_gaussian", "kt_radial", "random", "radial"}
            )
        )
    )
    return {
        "cmrbench_method": sampling.method,
        "sampling_implementation": str(implementation),
        "source_pattern": source_pattern,
        "semantic_family": semantic,
        "conditioning_label": build_undersampling_prompt(model_nominal, semantic),
        "nominal_acceleration": nominal,
        "actual_acceleration": float(measured),
        "ncalib": ncalib,
        "calibration_source": (
            calibration.calibration_source if calibration is not None else None
        ),
        "requested_calibration_width": requested_width,
        "released_training_ncalib": NUM_LOW_FREQUENCIES,
        "model_internal_num_low_frequencies": model_num_low,
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
        "acs_requirement_layer": "upstream_cardiomm_runtime",
        "no_acs_runtime_behavior": (
            "explicit num_low_frequencies crops the requested central window "
            "without verifying full support; 1D auto mode forces at least width one"
        ),
        "physical_acs_geometry": physical_acs_geometry,
        "temporal_behavior": temporal_behavior,
        "compatibility_status": "supported" if legacy_supported else "ood_valid",
        "generator_is_cardiomm_native": implementation == "cardiomm",
        "generator_matches_verified_cardiomm_uniform_contract": (
            implementation == "canonical" and method == "uniform"
        ),
        "source_identity": str(source_identity or "unrecorded_generated_source"),
        "support_scope": (
            "released_condition_compatible"
            if documented
            else "technically_executable_with_truthful_conditioning_but_ood_or_unconfirmed"
        ),
        "physical_sampling_validity": "valid",
        "runtime_compatibility": (
            "valid"
            if acs_detection["has_complete_central_acs"]
            else "upstream_deferred"
        ),
        **condition,
        "documented_physical_sampling_family_match": physical_documented,
        "documented_model_conditioning_family_match": documented_family,
        "documented_training_family_match": documented_family,
        "documented_training_acceleration_match": documented_acceleration,
        "checkpoint_training_exposure": "documented" if documented else "undocumented",
        "runtime_condition_resolution": "matched",
        "warnings": warning_messages,
    }


def canonical_kspace_frame_to_cardiomm(frame: object) -> np.ndarray:
    """Convert ``[Kx,Ky,C]`` complex to ``[1,C,Kx,Ky,2]`` float32."""

    value = np.asarray(frame)
    if value.ndim != 3 or not np.iscomplexobj(value):
        raise ValueError("CardioMM frame must be complex [Kx,Ky,C]")
    if not (np.isfinite(value.real).all() and np.isfinite(value.imag).all()):
        raise ValueError("CardioMM frame contains NaN or Inf")
    two_channel = np.stack((value.real, value.imag), axis=-1).astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(two_channel.transpose(2, 0, 1, 3)[None])


def canonical_kspace_batch_to_cardiomm(frames: object) -> np.ndarray:
    """Convert ``[B,Kx,Ky,C]`` complex to ``[B,C,Kx,Ky,2]`` float32."""

    value = np.asarray(frames)
    if value.ndim != 4 or not np.iscomplexobj(value):
        raise ValueError("CardioMM batch must be complex [B,Kx,Ky,C]")
    if not (np.isfinite(value.real).all() and np.isfinite(value.imag).all()):
        raise ValueError("CardioMM batch contains NaN or Inf")
    two_channel = np.stack((value.real, value.imag), axis=-1).astype(
        np.float32, copy=False
    )
    return np.ascontiguousarray(two_channel.transpose(0, 3, 1, 2, 4))


def canonical_mask_frame_to_cardiomm(mask: object) -> np.ndarray:
    """Convert acquired=1 ``[Kx,Ky]`` to ``[1,1,Kx,Ky,1]`` Boolean."""

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("CardioMM mask frame must be [Kx,Ky]")
    if not np.isfinite(value).all() or not np.all((value == 0) | (value == 1)):
        raise ValueError("CardioMM mask must be finite and exactly binary")
    if not np.any(value):
        raise ValueError("CardioMM mask has no acquired samples")
    return np.ascontiguousarray(value.astype(bool, copy=False)[None, None, :, :, None])


def canonical_mask_batch_to_cardiomm(masks: object) -> np.ndarray:
    """Convert acquired=1 ``[B,Kx,Ky]`` to ``[B,1,Kx,Ky,1]`` Boolean."""

    value = np.asarray(masks)
    if value.ndim != 3:
        raise ValueError("CardioMM mask batch must be [B,Kx,Ky]")
    if not np.isfinite(value).all() or not np.all((value == 0) | (value == 1)):
        raise ValueError("CardioMM mask must be finite and exactly binary")
    if np.any(np.count_nonzero(value, axis=(1, 2)) == 0):
        raise ValueError("CardioMM mask batch contains an item with no acquired samples")
    return np.ascontiguousarray(value.astype(bool, copy=False)[:, None, :, :, None])


def conditioning_cache_key(
    metadata_prompt: str,
    undersampling_prompt: str,
    device: str,
    dtype: str,
    model_identity: Tuple[int, int, int],
) -> Tuple[str, str, str, str, Tuple[int, int, int]]:
    """Return the complete adapter-instance key for deterministic conditioning."""

    return (
        str(metadata_prompt),
        str(undersampling_prompt),
        str(device),
        str(dtype),
        tuple(int(value) for value in model_identity),
    )


def time_major_slice_minor_indices(slices: int, phases: int) -> Iterator[Tuple[int, int]]:
    for phase_index in range(phases):
        for slice_index in range(slices):
            yield slice_index, phase_index


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def _pinned_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    revision = completed.stdout.strip().lower()
    if completed.returncode != 0 or revision != CARDIOMM_COMMIT:
        raise RuntimeError("CardioMM source is not at the pinned commit")
    return revision


def _clear_official_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            sys.modules.pop(name, None)
        elif name == "text" or name.startswith("text."):
            sys.modules.pop(name, None)
        elif name == "fastmri" or name.startswith("fastmri."):
            sys.modules.pop(name, None)


def _import_official_modules(working_directory: Path) -> Tuple[Any, Any, Dict[str, str]]:
    working = working_directory.resolve()
    _clear_official_modules()
    sys.path.insert(0, str(working))
    try:
        importlib.invalidate_caches()
        model_module = importlib.import_module("models.CardioMM_model")
        text_module = importlib.import_module("text.text_models")
    finally:
        if sys.path and sys.path[0] == str(working):
            sys.path.pop(0)
    resolved = {}
    for name, module in (
        ("models.CardioMM_model", model_module),
        ("text.text_models", text_module),
    ):
        module_file = Path(str(getattr(module, "__file__", ""))).resolve()
        if not _is_within(module_file, working):
            raise RuntimeError("{} resolved outside pinned CardioMM source".format(name))
        resolved[name] = str(module_file)
    return model_module, text_module, resolved


def _partition_checkpoint(state_dict: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "loss.w" not in state_dict:
        raise KeyError("CardioMM checkpoint is missing source-required loss.w")
    partitions = {}
    for prefix, component in (
        ("metausunetlight.", "metausunetlight"),
        ("lm_head_meta.", "lm_head_meta"),
        ("lm_head_us.", "lm_head_us"),
    ):
        partition = {
            str(key).replace(prefix, ""): value
            for key, value in state_dict.items()
            if str(key).startswith(prefix)
        }
        if not partition:
            raise KeyError("CardioMM checkpoint partition is empty: {}".format(component))
        partitions[component] = partition
    return partitions


class _DefaultCardioMMRuntime:
    """Dependency-isolated wrapper around the exact pinned released classes."""

    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        clip_dir: Path,
        device: str,
    ) -> None:
        self.source_root = source_root
        self.working_directory = source_root / "CardioMM_reconstruction"
        self.checkpoint = checkpoint
        self.clip_dir = clip_dir
        self.device_name = device
        self.loaded = False
        self._conditioning_cache = {}  # type: Dict[Any, Tuple[Any, Any]]

    def load(self) -> Dict[str, Any]:
        if self.loaded:
            raise RuntimeError("CardioMM runtime load() was called more than once")
        if not self.source_root.is_dir() or not self.working_directory.is_dir():
            raise FileNotFoundError("pinned CardioMM source is absent")
        revision = _pinned_revision(self.source_root)
        if not self.checkpoint.is_file():
            raise FileNotFoundError("configured CardioMM checkpoint is absent")
        if self.checkpoint.name != CHECKPOINT_FILENAME:
            raise ValueError("CardioMM checkpoint filename must be epochbest.ckpt")
        expected_clip = (self.source_root / "clip-vit-base-patch16").resolve()
        if self.clip_dir.resolve() != expected_clip or not expected_clip.is_dir():
            raise FileNotFoundError("validated local CardioMM CLIP directory is absent")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch

        if not torch.cuda.is_available() or not self.device_name.startswith("cuda"):
            raise RuntimeError("CardioMM validated runtime requires a CUDA device")
        self.torch = torch
        self.device = torch.device(self.device_name)
        model_module, text_module, resolved = _import_official_modules(
            self.working_directory
        )
        components = {
            "metausunetlight": model_module.CardioMM_SEM(
                **MODEL_CONSTRUCTOR_ARGUMENTS
            ),
            "lm_head_meta": text_module.LMHead(**LANGUAGE_HEAD_ARGUMENTS),
            "lm_head_us": text_module.LMHead2(**LANGUAGE_HEAD_ARGUMENTS),
        }
        loaded = torch.load(str(self.checkpoint), map_location="cpu")
        if not isinstance(loaded, Mapping) or not isinstance(
            loaded.get("state_dict"), Mapping
        ):
            raise TypeError("CardioMM checkpoint must contain a state_dict mapping")
        partitions = _partition_checkpoint(loaded["state_dict"])
        for name, module in components.items():
            outcome = module.load_state_dict(partitions[name], strict=True)
            if getattr(outcome, "missing_keys", ()) or getattr(
                outcome, "unexpected_keys", ()
            ):
                raise RuntimeError("CardioMM strict checkpoint load was incomplete")
            module.eval()
            module.to(self.device)

        with _working_directory(self.working_directory):
            language_model = text_module.LanguageModel(
                llm_model_name="../clip-vit-base-patch16"
            )
        if any(parameter.requires_grad for parameter in language_model.parameters()):
            raise RuntimeError("released CardioMM LanguageModel is not frozen")
        clip_candidates = [
            module
            for module in language_model.modules()
            if module.__class__.__name__ == "CLIPModel"
        ]
        if len(clip_candidates) != 1:
            raise RuntimeError("released LanguageModel does not contain one CLIPModel")
        if getattr(clip_candidates[0], "vision_model", "MISSING") is not None:
            raise RuntimeError("released LanguageModel did not remove the vision tower")
        if bool(clip_candidates[0].training):
            raise RuntimeError("released internal CLIP model is not in eval mode")

        self.model = components["metausunetlight"]
        self.meta_head = components["lm_head_meta"]
        self.us_head = components["lm_head_us"]
        self.language_model = language_model
        self.loaded = True
        parameter_count = sum(
            int(parameter.numel()) for module in components.values() for parameter in module.parameters()
        )
        del loaded, partitions
        return {
            "source_commit": revision,
            "resolved_modules": resolved,
            "checkpoint_filename": self.checkpoint.name,
            "checkpoint_strict_load": True,
            "model_constructed_once": True,
            "checkpoint_loaded_once": True,
            "clip_offline": True,
            "clip_vision_tower_removed": True,
            "clip_parameters_frozen": True,
            "parameter_count": parameter_count,
        }

    def _conditioning(
        self,
        metadata_prompt: str,
        undersampling_prompt: str,
        *,
        batch_size: int,
        cache_enabled: bool,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        torch = self.torch
        key = conditioning_cache_key(
            metadata_prompt,
            undersampling_prompt,
            str(self.device),
            str(torch.float32),
            (id(self.language_model), id(self.meta_head), id(self.us_head)),
        )
        cache_hit = bool(cache_enabled and key in self._conditioning_cache)
        language_seconds = 0.0
        head_seconds = 0.0
        if cache_hit:
            metadata_adapted, undersampling_adapted = self._conditioning_cache[key]
        else:
            torch.cuda.synchronize(self.device)
            started = perf_counter()
            metadata_embedding = self.language_model([metadata_prompt])
            undersampling_embedding = self.language_model([undersampling_prompt])
            torch.cuda.synchronize(self.device)
            language_seconds = perf_counter() - started
            if tuple(metadata_embedding.shape) != (1, 512) or tuple(
                undersampling_embedding.shape
            ) != (1, 512):
                raise RuntimeError("CardioMM CLIP embeddings must be [1,512]")
            started = perf_counter()
            metadata_adapted, _ = self.meta_head(
                metadata_embedding.unsqueeze(0).to(self.device)
            )
            undersampling_adapted, _ = self.us_head(
                undersampling_embedding.unsqueeze(0).to(self.device)
            )
            torch.cuda.synchronize(self.device)
            head_seconds = perf_counter() - started
            metadata_adapted = metadata_adapted.detach()
            undersampling_adapted = undersampling_adapted.detach()
            if cache_enabled:
                self._conditioning_cache[key] = (
                    metadata_adapted,
                    undersampling_adapted,
                )

        def expand_batch(value: Any) -> Any:
            if int(value.shape[0]) != 1:
                raise RuntimeError("cached CardioMM conditioning must have batch dimension 1")
            if batch_size == 1:
                return value
            return value.repeat((batch_size,) + (1,) * (value.ndim - 1))

        return expand_batch(metadata_adapted), expand_batch(undersampling_adapted), {
            "conditioning_cache_hit": cache_hit,
            "text_conditioning_recomputed": not cache_hit,
            "tokenization_and_clip_text_forward_seconds": language_seconds,
            "language_heads_seconds": head_seconds,
        }

    def infer_batch(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        metadata_prompt: str,
        undersampling_prompt: str,
        expected_shape: Tuple[int, int],
        *,
        mask_semantic: str = MASK_SEMANTIC,
        num_low_frequencies: Optional[int] = NUM_LOW_FREQUENCIES,
        text_conditioning_cache: bool = False,
        inference_context: str = "no_grad",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.loaded:
            raise RuntimeError("CardioMM runtime is not loaded")
        torch = self.torch
        if inference_context not in {"no_grad", "inference_mode"}:
            raise ValueError("inference_context must be 'no_grad' or 'inference_mode'")
        batch_size = int(kspace.shape[0])
        if batch_size < 1 or tuple(mask.shape[:2]) != (batch_size, 1):
            raise ValueError("CardioMM batch and mask batch dimensions do not match")
        torch.cuda.synchronize(self.device)
        transfer_started = perf_counter()
        kspace_tensor = torch.from_numpy(kspace).to(self.device)
        mask_tensor = torch.from_numpy(mask).to(self.device)
        torch.cuda.synchronize(self.device)
        host_to_device_seconds = perf_counter() - transfer_started
        captured = {}
        sensitivity_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

        def sensitivity_start(_module: Any, _inputs: Sequence[Any]) -> None:
            sensitivity_events[0].record()

        def capture_sensitivity(_module: Any, _inputs: Sequence[Any], value: Any) -> None:
            sensitivity_events[1].record()
            captured["value"] = value.detach()

        pre_hook = self.model.sens_net.register_forward_pre_hook(sensitivity_start)
        post_hook = self.model.sens_net.register_forward_hook(capture_sensitivity)
        started = perf_counter()
        try:
            context = torch.no_grad if inference_context == "no_grad" else torch.inference_mode
            with context():
                metadata_adapted, undersampling_adapted, conditioning_record = self._conditioning(
                    metadata_prompt,
                    undersampling_prompt,
                    batch_size=batch_size,
                    cache_enabled=text_conditioning_cache,
                )
                torch.cuda.reset_peak_memory_stats(self.device)
                model_started = perf_counter()
                output_tensor = self.model(
                    kspace_tensor,
                    mask_tensor,
                    num_low_frequencies,
                    metadata_adapted,
                    undersampling_adapted,
                    [str(mask_semantic)] * batch_size,
                )
                torch.cuda.synchronize(self.device)
                model_forward_seconds = perf_counter() - model_started
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        finally:
            pre_hook.remove()
            post_hook.remove()
        elapsed = perf_counter() - started
        if "value" not in captured:
            raise RuntimeError("CardioMM internal sensitivity estimation was not observed")
        sensitivity = captured["value"]
        expected_smaps = (
            batch_size,
            int(kspace.shape[1]),
            expected_shape[0],
            expected_shape[1],
            2,
        )
        diagnostic_started = perf_counter()
        smaps_finite = bool(torch.isfinite(sensitivity).all())
        output_finite = bool(torch.isfinite(output_tensor).all())
        torch.cuda.synchronize(self.device)
        diagnostic_seconds = perf_counter() - diagnostic_started
        if tuple(sensitivity.shape) != expected_smaps or not smaps_finite:
            raise RuntimeError("CardioMM internal sensitivity maps are invalid")
        if not output_finite:
            raise RuntimeError("CardioMM native output contains NaN or Inf")
        transfer_started = perf_counter()
        output = output_tensor.detach().cpu().contiguous().numpy()
        device_to_host_seconds = perf_counter() - transfer_started
        expected_output = (batch_size, expected_shape[0], expected_shape[1])
        if output.shape != expected_output:
            raise RuntimeError("CardioMM output is not native [B,Kx,Ky]")
        sensitivity_seconds = float(sensitivity_events[0].elapsed_time(sensitivity_events[1])) / 1000.0
        return np.asarray(output, dtype=np.float32), {
            "runtime_seconds": elapsed,
            "internal_sensitivity_maps_shape": list(expected_smaps),
            "internal_sensitivity_maps_finite": True,
            "internal_sensitivity_estimation_seconds": sensitivity_seconds,
            "model_forward_seconds": model_forward_seconds,
            "cascade_and_output_seconds": max(0.0, model_forward_seconds - sensitivity_seconds),
            "host_to_device_seconds": host_to_device_seconds,
            "finite_shape_diagnostics_seconds": diagnostic_seconds,
            "device_to_host_seconds": device_to_host_seconds,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "batch_size": batch_size,
            "num_low_frequencies_passed_to_model": num_low_frequencies,
            **conditioning_record,
        }

    def infer_frame(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        metadata_prompt: str,
        undersampling_prompt: str,
        expected_shape: Tuple[int, int],
        mask_semantic: str = MASK_SEMANTIC,
        num_low_frequencies: Optional[int] = NUM_LOW_FREQUENCIES,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Frozen B=1/no-cache/no-grad compatibility path."""

        output, record = self.infer_batch(
            kspace,
            mask,
            metadata_prompt,
            undersampling_prompt,
            expected_shape,
            mask_semantic=mask_semantic,
            num_low_frequencies=num_low_frequencies,
            text_conditioning_cache=False,
            inference_context="no_grad",
        )
        return output[0], record


class CardioMMAdapter(ReconstructionModel):
    """Production adapter for source-interpretable CardioMM sampling conditions."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint: str | Path,
        clip_dir: str | Path,
        metadata: Mapping[str, Any],
        device: str = "cuda:0",
        num_low_frequencies: Optional[int] = NUM_LOW_FREQUENCIES,
        sampling_condition: Optional[Mapping[str, Any]] = None,
        inference_batch_size: int = INFERENCE_BATCH_SIZE,
        text_conditioning_cache: bool = TEXT_CONDITIONING_CACHE_DEFAULT,
        inference_context: str = INFERENCE_CONTEXT_DEFAULT,
        runtime_factory: Optional[RuntimeFactory] = None,
    ) -> None:
        if num_low_frequencies is not None and (
            isinstance(num_low_frequencies, bool)
            or int(num_low_frequencies) != float(num_low_frequencies)
            or int(num_low_frequencies) < 0
        ):
            raise ValueError("CardioMM num_low_frequencies must be nonnegative or null")
        if sampling_condition is not None and not isinstance(sampling_condition, Mapping):
            raise ValueError("CardioMM sampling_condition must be a mapping")
        if int(inference_batch_size) < 1:
            raise ValueError("inference_batch_size must be a positive integer")
        if inference_context not in {"no_grad", "inference_mode"}:
            raise ValueError("inference_context must be 'no_grad' or 'inference_mode'")
        self.source_root = Path(source_root)
        self.checkpoint = Path(checkpoint)
        self.clip_dir = Path(clip_dir)
        self.metadata = dict(metadata)
        self.metadata_prompt = build_metadata_prompt(self.metadata)
        self.device = str(device)
        self.num_low_frequencies = (
            None if num_low_frequencies is None else int(num_low_frequencies)
        )
        self.sampling_condition = (
            dict(sampling_condition) if sampling_condition is not None else None
        )
        self.inference_batch_size = int(inference_batch_size)
        self.text_conditioning_cache = bool(text_conditioning_cache)
        self.inference_context = str(inference_context)
        self._runtime_factory = runtime_factory or _DefaultCardioMMRuntime
        self._runtime = None  # type: Optional[Any]
        self._load_provenance = {}  # type: Dict[str, Any]

    def load(self) -> None:
        if self._runtime is not None:
            return
        runtime = self._runtime_factory(
            source_root=self.source_root,
            checkpoint=self.checkpoint,
            clip_dir=self.clip_dir,
            device=self.device,
        )
        provenance = runtime.load()
        if not isinstance(provenance, dict):
            raise RuntimeError("CardioMM runtime load() must return provenance metadata")
        self._runtime = runtime
        self._load_provenance = dict(provenance)

    @staticmethod
    def _validate_input(input: ReconstructionInput) -> Tuple[np.ndarray, np.ndarray]:
        kspace = np.asarray(input.kspace)
        mask = np.asarray(input.mask)
        if kspace.ndim != 5 or not np.iscomplexobj(kspace):
            raise ValueError("CardioMM requires complex canonical k-space [S,T,Kx,Ky,C]")
        if mask.ndim != 4 or mask.shape != kspace.shape[:4]:
            raise ValueError("CardioMM requires matching canonical mask [S,T,Kx,Ky]")
        if not (np.isfinite(kspace.real).all() and np.isfinite(kspace.imag).all()):
            raise ValueError("CardioMM k-space contains NaN or Inf")
        if not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("CardioMM mask must be finite, binary, and use 1=acquired")
        binary = mask.astype(bool, copy=False)
        if np.any(kspace[~binary] != 0):
            raise ValueError("CardioMM input k-space is not already masked by its canonical mask")
        return np.asarray(kspace, dtype=np.complex64), binary

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        kspace, mask = self._validate_input(input)
        model_sampling = cardiomm_sampling_compatibility(
            input,
            self.sampling_condition,
            default_num_low_frequencies=self.num_low_frequencies,
        )
        if self._runtime is None:
            self.load()
        slices, phases, kx, ky, coils = kspace.shape
        undersampling_prompt = str(model_sampling["conditioning_label"])
        mask_semantic = str(model_sampling["semantic_family"])
        output = np.empty((slices, phases, kx, ky), dtype=np.float32)
        frame_records = []
        batch_records = []
        conditioning_prompt_pairs = set()
        evaluated_conditioning_prompt_pairs = set()
        total_start = perf_counter()
        indices = list(time_major_slice_minor_indices(slices, phases))
        for batch_index, first in enumerate(range(0, len(indices), self.inference_batch_size)):
            selected = indices[first : first + self.inference_batch_size]
            extraction_started = perf_counter()
            canonical_kspace = np.stack(
                [kspace[slice_index, phase_index] for slice_index, phase_index in selected]
            )
            canonical_mask = np.stack(
                [mask[slice_index, phase_index] for slice_index, phase_index in selected]
            )
            extraction_seconds = perf_counter() - extraction_started
            conversion_started = perf_counter()
            official_kspace = canonical_kspace_batch_to_cardiomm(canonical_kspace)
            kspace_conversion_seconds = perf_counter() - conversion_started
            conversion_started = perf_counter()
            official_mask = canonical_mask_batch_to_cardiomm(canonical_mask)
            mask_conversion_seconds = perf_counter() - conversion_started
            infer_batch = getattr(self._runtime, "infer_batch", None)
            conditioning_prompt_pairs.add((self.metadata_prompt, undersampling_prompt))
            if callable(infer_batch):
                infer_keywords = {
                    "mask_semantic": mask_semantic,
                    "text_conditioning_cache": self.text_conditioning_cache,
                    "inference_context": self.inference_context,
                }
                if "num_low_frequencies" in inspect.signature(infer_batch).parameters:
                    infer_keywords["num_low_frequencies"] = model_sampling[
                        "model_internal_num_low_frequencies"
                    ]
                elif model_sampling["model_internal_num_low_frequencies"] != NUM_LOW_FREQUENCIES:
                    raise RuntimeError(
                        "configured CardioMM runtime cannot receive non-default num_low_frequencies"
                    )
                try:
                    frames, runtime_record = infer_batch(
                        official_kspace,
                        official_mask,
                        self.metadata_prompt,
                        undersampling_prompt,
                        (kx, ky),
                        **infer_keywords,
                    )
                except Exception as exc:
                    raise UpstreamModelInferenceError("cardiomm", exc) from exc
            elif len(selected) == 1:
                if mask_semantic != MASK_SEMANTIC:
                    raise RuntimeError(
                        "legacy CardioMM infer_frame cannot receive non-uniform mask semantics"
                    )
                if self.text_conditioning_cache:
                    raise RuntimeError(
                        "legacy CardioMM infer_frame fallback is available only with "
                        "text_conditioning_cache=false for regression/debug"
                    )
                try:
                    frame, runtime_record = self._runtime.infer_frame(
                        official_kspace,
                        official_mask,
                        self.metadata_prompt,
                        undersampling_prompt,
                        (kx, ky),
                    )
                except Exception as exc:
                    raise UpstreamModelInferenceError("cardiomm", exc) from exc
                frames = np.asarray(frame)[None]
                runtime_record = dict(runtime_record)
                runtime_record.setdefault("conditioning_cache_hit", False)
                runtime_record.setdefault("text_conditioning_recomputed", True)
            else:
                raise RuntimeError("configured CardioMM runtime does not support batching")
            if bool(runtime_record.get("text_conditioning_recomputed", False)):
                evaluated_conditioning_prompt_pairs.add(
                    (self.metadata_prompt, undersampling_prompt)
                )
            values = np.asarray(frames)
            expected_batch_shape = (len(selected), kx, ky)
            if values.shape != expected_batch_shape:
                raise UpstreamModelInferenceError(
                    "cardiomm",
                    RuntimeError(
                        "output shape {} does not match {} for batch {}".format(
                            values.shape, expected_batch_shape, batch_index
                        )
                    ),
                )
            if np.iscomplexobj(values):
                raise UpstreamModelInferenceError(
                    "cardiomm", RuntimeError("output must be real RSS magnitude")
                )
            if not np.isfinite(values).all() or np.any(values < 0):
                raise UpstreamModelInferenceError(
                    "cardiomm", RuntimeError("output batch is non-finite or negative")
                )
            reassembly_started = perf_counter()
            for item_index, (slice_index, phase_index) in enumerate(selected):
                flat_index = first + item_index
                output[slice_index, phase_index] = values[item_index].astype(
                    np.float32, copy=False
                )
                frame_records.append(
                    {
                        "slice": slice_index,
                        "phase": phase_index,
                        "time_major_frame_index": flat_index,
                        "batch_index": batch_index,
                        "batch_item_index": item_index,
                        "input_coils": coils,
                    }
                )
            reassembly_seconds = perf_counter() - reassembly_started
            batch_records.append(
                {
                    "batch_index": batch_index,
                    "first_time_major_frame_index": first,
                    "frames": len(selected),
                    "canonical_frame_extraction_seconds": extraction_seconds,
                    "kspace_layout_conversion_seconds": kspace_conversion_seconds,
                    "mask_layout_conversion_seconds": mask_conversion_seconds,
                    "canonical_output_reassembly_seconds": reassembly_seconds,
                    **dict(runtime_record),
                }
            )
        runtime_seconds = perf_counter() - total_start
        actual_acceleration = float(input.sampling.actual_acceleration)
        cache_hits = sum(
            bool(record.get("conditioning_cache_hit", False))
            for record in batch_records
        )
        conditioning_evaluations = sum(
            bool(record.get("text_conditioning_recomputed", False))
            for record in batch_records
        )
        transforms = [
            TransformRecord(
                "cardiomm_sampling_compatibility",
                dict(model_sampling),
            ),
            TransformRecord(
                "cardiomm_framewise_decomposition",
                {
                    "input_shape": list(kspace.shape),
                    "order": "time-major, slice-minor",
                    "batch_size": self.inference_batch_size,
                    "temporal_modeling": False,
                    "num_adj_slices": 1,
                },
            ),
            TransformRecord(
                "canonical_to_cardiomm_kspace",
                {
                    "canonical_frame": "[Kx,Ky,C] complex64",
                    "cardiomm_frame": "[B,C,Kx,Ky,2] float32",
                    "complex_channels": ["real", "imaginary"],
                    "spatial_transpose": False,
                    "coil_compression": False,
                    "input_coils": coils,
                },
            ),
            TransformRecord(
                "canonical_to_cardiomm_mask",
                {
                    "canonical_frame": "[Kx,Ky]",
                    "cardiomm_frame": "[B,1,Kx,Ky,1] bool",
                    "sampled_value": 1,
                    "mask_inverted": False,
                    "resampled_by_adapter": False,
                },
            ),
            TransformRecord(
                "cardiomm_internal_sensitivity_estimation",
                {
                    "source": "released sens_net",
                    "acs_lines": model_sampling["model_internal_num_low_frequencies"],
                    "per_frame": True,
                    "external_cmrbench_smaps_used": False,
                },
            ),
            TransformRecord(
                "cardiomm_rss_to_canonical",
                {
                    "native": "[B,Kx,Ky] real nonnegative RSS magnitude",
                    "canonical": "[S,T,X,Y] float32",
                    "spatial_transpose": False,
                    "phase_fabricated": False,
                    "normalization": None,
                },
            ),
        ]
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=output,
            model_name="cardiomm",
            runtime_seconds=runtime_seconds,
            transforms=transforms,
            metadata={
                "adapter_version": CARDIOMM_ADAPTER_VERSION,
                "implementation": "official_pinned_source",
                "source_commit": CARDIOMM_COMMIT,
                "architecture": MODEL_ARCHITECTURE,
                "num_cascades": NUM_CASCADES,
                "direct_pretrained_inference": True,
                "model_eval_mode": True,
                "inference_context": "torch.{}".format(self.inference_context),
                "adaptation_performed": False,
                "fine_tuning_called": False,
                "optimizer_created": False,
                "backward_called": False,
                "metadata_prompt": self.metadata_prompt,
                "undersampling_prompt": undersampling_prompt,
                "nominal_acceleration": model_sampling["nominal_acceleration"],
                "measured_actual_acceleration": actual_acceleration,
                "sampling_source_identity": model_sampling["source_identity"],
                "model_sampling": dict(model_sampling),
                "acs_lines": model_sampling["model_internal_num_low_frequencies"],
                "sampling_condition_request": dict(self.sampling_condition or {}),
                "input_coils": coils,
                "coil_compression": False,
                "training_coil_count_caveat": (
                    "CardioMM training standardized inputs to 10 coils; the runtime "
                    "coil count is preserved and may represent a distribution shift."
                ),
                "external_smaps_present": input.sensitivity_maps is not None,
                "external_smaps_used": False,
                "sensitivity_maps": "internally_estimated_per_frame",
                "inference_batch_size": self.inference_batch_size,
                "text_conditioning_cached": self.text_conditioning_cache,
                "unique_text_conditioning_inputs": len(conditioning_prompt_pairs),
                "unique_text_conditioning_evaluations": len(
                    evaluated_conditioning_prompt_pairs
                ),
                "text_conditioning_evaluations_this_run": conditioning_evaluations,
                "text_conditioning_cache_hits_this_run": cache_hits,
                "text_conditioning_cache_scope": "adapter_instance_memory_only",
                "batching_validation_status": (
                    "validated_production_b1"
                    if self.inference_batch_size == INFERENCE_BATCH_SIZE
                    else "experimental_non_frozen_b_greater_than_1"
                ),
                "framewise_independent_inference": True,
                "temporal_modeling": False,
                "internal_sensitivity_estimation": "internally_estimated_per_frame",
                "precision": "float32",
                "output_representation": "real nonnegative RSS magnitude",
                "output_axis_order": "[S,T,X,Y]",
                "frame_inference": frame_records,
                "batch_inference": batch_records,
                "checkpoint_loaded_once": bool(
                    self._load_provenance.get("checkpoint_loaded_once", False)
                ),
                "model_constructed_once": bool(
                    self._load_provenance.get("model_constructed_once", False)
                ),
                "clip_parameters_frozen": bool(
                    self._load_provenance.get("clip_parameters_frozen", False)
                ),
                "clip_offline": bool(self._load_provenance.get("clip_offline", False)),
                "runtime": dict(self._load_provenance),
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="cardiomm",
            model_version=CARDIOMM_ADAPTER_VERSION,
            checkpoint=self.checkpoint.name,
            checkpoint_identifier="confidential_author_checkpoint",
            git_commit=CARDIOMM_COMMIT,
            parameter_count=self._load_provenance.get("parameter_count"),
            device=self.device,
            precision="float32",
            metadata={
                "repository": CARDIOMM_REPOSITORY,
                "architecture": MODEL_ARCHITECTURE,
                "num_cascades": NUM_CASCADES,
                "inference_batch_size": self.inference_batch_size,
                "text_conditioning_cached": self.text_conditioning_cache,
                "inference_context": "torch.{}".format(self.inference_context),
                "sampling_condition_request": dict(self.sampling_condition or {}),
                "num_low_frequencies": self.num_low_frequencies,
                "batching_validation_status": (
                    "validated_production_b1"
                    if self.inference_batch_size == INFERENCE_BATCH_SIZE
                    else "experimental_non_frozen_b_greater_than_1"
                ),
                "external_smaps_used": False,
                "output_representation": "real nonnegative RSS magnitude",
                "checkpoint_filename": CHECKPOINT_FILENAME,
                "checkpoint_full_hash_serialized": False,
                **self._load_provenance,
            },
        )
