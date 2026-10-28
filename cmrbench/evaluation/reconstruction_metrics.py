"""Versioned CMRBench reconstruction metrics.

Metrics v1 is the preserved raw-amplitude diagnostic. Metrics v2 is the
primary cross-model protocol and independently divides each complete subject
magnitude volume by its own p99.5 before evaluating 2D slice-phase frames.
Neither protocol clips scientific arrays or fits a reference-dependent gain.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np


METRIC_PROTOCOL_V1 = "cmrbench_recon_metrics_v1"
METRIC_PROTOCOL_V2 = "cmrbench_recon_metrics_v2_subject_p995"
# Backward-compatible alias used by historical v1 imports and tests.
METRIC_PROTOCOL_VERSION = METRIC_PROTOCOL_V1
METRIC_PROTOCOL_SELECTIONS = ("v1_raw", "v2_subject_p995")
SSIM_WIN_SIZE = 7
SSIM_GAUSSIAN_WEIGHTS = False
PRIMARY_METRICS = ("nrmse", "psnr", "ssim")
CONDITION_IDENTITY_FIELDS = (
    "subject",
    "model",
    "acceleration_requested",
    "acceleration_actual",
    "sampling_method",
    "seed",
)
OPTIONAL_PROTOCOL_IDENTITY_FIELDS = (
    "metric_protocol_version",
    "reference_p99_5",
    "candidate_p99_5",
    "candidate_to_reference_p99_5_ratio",
    "reference_normalized_max",
    "candidate_normalized_max",
    "reference_fraction_above_1",
    "candidate_fraction_above_1",
    "scientific_clipping",
)


def resolve_metric_protocol(protocol: str) -> str:
    """Resolve a public protocol selector to its immutable protocol version."""

    aliases = {
        "v1_raw": METRIC_PROTOCOL_V1,
        METRIC_PROTOCOL_V1: METRIC_PROTOCOL_V1,
        "v2_subject_p995": METRIC_PROTOCOL_V2,
        METRIC_PROTOCOL_V2: METRIC_PROTOCOL_V2,
    }
    try:
        return aliases[str(protocol)]
    except KeyError as exc:
        raise ValueError(
            f"unknown reconstruction metric protocol {protocol!r}; expected one of "
            f"{sorted(aliases)}"
        ) from exc


def metric_protocol_provenance(
    protocol: str = METRIC_PROTOCOL_V1,
) -> dict[str, Any]:
    """Return one versioned protocol and installed evaluation package versions.

    The default remains v1 solely for backward compatibility with historical
    scripts. New benchmark evaluations must select ``v2_subject_p995``
    explicitly.
    """

    try:
        skimage_version = importlib_metadata.version("scikit-image")
    except importlib_metadata.PackageNotFoundError:
        skimage_version = "NOT INSTALLED"
    resolved = resolve_metric_protocol(protocol)
    common = {
        "protocol_version": resolved,
        "ssim_win_size": SSIM_WIN_SIZE,
        "ssim_gaussian_weights": SSIM_GAUSSIAN_WEIGHTS,
        "frame_unit": "2D slice-phase frame",
        "subject_aggregation": "equal_mean_over_SxT_frames",
        "cohort_aggregation": (
            "compute one equal-weight subject mean, then summarize across subjects"
        ),
        "gain_fitting": False,
        "clipping": False,
        "histogram_matching": False,
        "package_versions": {
            "numpy": np.__version__,
            "scikit-image": skimage_version,
        },
    }
    if resolved == METRIC_PROTOCOL_V1:
        return {
            **common,
            "selection": "v1_raw",
            "reporting_role": "secondary raw amplitude-sensitive diagnostic",
            "domain": "raw_magnitude",
            "nrmse_formula": "l2_error_over_l2_reference",
            "psnr_formula": "10 * log10(reference_data_range^2 / mse)",
            "psnr_data_range": "reference_max_minus_min",
            "ssim_data_range": "reference_max_minus_min",
            "candidate_specific_normalization": False,
            "reference_normalization": False,
            "normalization_scope": "none",
            "p99_5_primary_metric_usage": False,
        }
    return {
        **common,
        "selection": "v2_subject_p995",
        "reporting_role": "primary cross-model reconstruction metric",
        "domain": "independently_subject_p99_5_normalized_magnitude",
        "magnitude_conversion": "abs(volume)",
        "normalization_percentile": 99.5,
        "normalization_scope": "one scalar per complete [S,T,X,Y] subject source",
        "normalization_independent_per_source": True,
        "per_frame_normalization": False,
        "candidate_uses_reference_scale": False,
        "epsilon_scale_repair": False,
        "nrmse_formula": "l2_error_over_l2_normalized_reference",
        "psnr_formula": "10 * log10(1.0^2 / mse)",
        "psnr_data_range": 1.0,
        "ssim_data_range": 1.0,
        "candidate_specific_normalization": True,
        "reference_normalization": True,
        "p99_5_primary_metric_usage": True,
    }


def _validate_frame_pair(
    reference: np.ndarray, reconstruction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference)
    reconstruction_array = np.asarray(reconstruction)
    if reference_array.ndim != 2 or reconstruction_array.ndim != 2:
        raise ValueError(
            "primary frame metrics require reference and reconstruction [X,Y]"
        )
    if reference_array.shape != reconstruction_array.shape:
        raise ValueError(
            "reference and reconstruction frame shapes differ: "
            f"{reference_array.shape} vs {reconstruction_array.shape}"
        )
    reference_magnitude = np.abs(reference_array).astype(np.float64, copy=False)
    reconstruction_magnitude = np.abs(reconstruction_array).astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(reference_magnitude)):
        raise ValueError("reference magnitude contains NaN or Inf")
    if not np.all(np.isfinite(reconstruction_magnitude)):
        raise ValueError("reconstruction magnitude contains NaN or Inf")
    return reference_magnitude, reconstruction_magnitude


def normalize_subject_magnitude_p995(
    volume: np.ndarray,
    *,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize one complete cine magnitude by its own subject p99.5.

    The returned array is never clipped. A nonfinite or non-positive scale is
    an invalid evaluation condition; no epsilon or other repair is applied.
    """

    array = np.asarray(volume)
    if array.ndim != 4:
        raise ValueError(
            f"{source} subject normalization requires [S,T,X,Y]; "
            f"received {array.shape}"
        )
    magnitude = np.abs(array).astype(np.float64, copy=False)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError(f"{source} magnitude contains NaN or Inf")
    scale = float(np.percentile(magnitude, 99.5))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"{source} subject p99.5 scale must be positive and finite; "
            f"received {scale!r}"
        )
    normalized = magnitude / scale
    metadata = {
        "source": source,
        "p99_5": scale,
        "scope": "complete [S,T,X,Y] subject volume",
        "scientific_clipping": False,
        "epsilon_scale_repair": False,
        "normalized_max": float(np.max(normalized)),
        "fraction_of_values_above_1": float(np.mean(normalized > 1.0)),
    }
    return normalized, metadata


def compute_normalized_frame_metrics_v2(
    reference_normalized: np.ndarray,
    reconstruction_normalized: np.ndarray,
) -> dict[str, float]:
    """Compute Metrics v2 for one already subject-normalized 2D frame."""

    reference_array = np.asarray(reference_normalized, dtype=np.float64)
    reconstruction_array = np.asarray(
        reconstruction_normalized, dtype=np.float64
    )
    if reference_array.ndim != 2 or reconstruction_array.ndim != 2:
        raise ValueError(
            "Metrics v2 frame inputs must be normalized [X,Y] arrays"
        )
    if reference_array.shape != reconstruction_array.shape:
        raise ValueError(
            "Metrics v2 normalized frame shapes differ: "
            f"{reference_array.shape} vs {reconstruction_array.shape}"
        )
    if not np.all(np.isfinite(reference_array)):
        raise ValueError("Metrics v2 normalized reference contains NaN or Inf")
    if not np.all(np.isfinite(reconstruction_array)):
        raise ValueError("Metrics v2 normalized reconstruction contains NaN or Inf")
    if np.any(reference_array < 0.0) or np.any(reconstruction_array < 0.0):
        raise ValueError("Metrics v2 normalized magnitudes must be non-negative")

    reference_l2 = float(np.linalg.norm(reference_array))
    if reference_l2 == 0.0:
        raise ValueError("Metrics v2 NRMSE is undefined for a zero-reference frame")
    difference = reconstruction_array - reference_array
    nrmse = float(np.linalg.norm(difference) / reference_l2)
    mse = float(np.mean(difference**2, dtype=np.float64))
    data_range = 1.0
    psnr = (
        float("inf")
        if mse == 0.0
        else float(10.0 * np.log10((data_range**2) / mse))
    )
    if min(reference_array.shape) < SSIM_WIN_SIZE:
        raise ValueError(
            "Metrics v2 SSIM requires both spatial dimensions to be at least "
            f"win_size={SSIM_WIN_SIZE}; received {reference_array.shape}"
        )
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "CMRBench Metrics v2 SSIM requires the scikit-image dependency"
        ) from exc
    ssim = float(
        structural_similarity(
            reference_array,
            reconstruction_array,
            data_range=data_range,
            win_size=SSIM_WIN_SIZE,
            gaussian_weights=SSIM_GAUSSIAN_WEIGHTS,
        )
    )
    return {
        "nrmse": nrmse,
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse,
        "reference_l2": reference_l2,
        "reference_data_range": data_range,
    }


def compute_frame_metrics(
    reference: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """Compute frozen primary NRMSE, PSNR, and SSIM for one 2D frame."""

    reference_magnitude, reconstruction_magnitude = _validate_frame_pair(
        reference, reconstruction
    )
    reference_l2 = float(np.linalg.norm(reference_magnitude))
    if reference_l2 == 0.0:
        raise ValueError("primary NRMSE is undefined for a zero-reference frame")

    difference = reconstruction_magnitude - reference_magnitude
    nrmse = float(np.linalg.norm(difference) / reference_l2)
    mse = float(np.mean(difference**2, dtype=np.float64))
    data_range = float(
        np.max(reference_magnitude) - np.min(reference_magnitude)
    )
    if data_range <= 0.0:
        raise ValueError(
            "primary PSNR/SSIM require a positive reference magnitude data range"
        )
    psnr = (
        float("inf")
        if mse == 0.0
        else float(10.0 * np.log10((data_range**2) / mse))
    )
    if min(reference_magnitude.shape) < SSIM_WIN_SIZE:
        raise ValueError(
            "primary SSIM requires both spatial dimensions to be at least "
            f"win_size={SSIM_WIN_SIZE}; received {reference_magnitude.shape}"
        )
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "CMRBench primary SSIM requires the scikit-image dependency"
        ) from exc
    ssim = float(
        structural_similarity(
            reference_magnitude,
            reconstruction_magnitude,
            data_range=data_range,
            win_size=SSIM_WIN_SIZE,
            gaussian_weights=SSIM_GAUSSIAN_WEIGHTS,
        )
    )
    return {
        "nrmse": nrmse,
        "psnr": psnr,
        "ssim": ssim,
        "mse": mse,
        "reference_l2": reference_l2,
        "reference_data_range": data_range,
    }


def compute_cine_frame_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    subject: str,
    model: str,
    acceleration_requested: float | int | None,
    acceleration_actual: float | None,
    sampling_method: str,
    seed: int | None,
    protocol: str = METRIC_PROTOCOL_V1,
) -> list[dict[str, Any]]:
    """Compute one versioned metric row per canonical cine frame.

    The default is the historical v1 API. Callers producing new benchmark
    results must pass ``protocol="v2_subject_p995"`` explicitly.
    """

    reference_array = np.asarray(reference)
    reconstruction_array = np.asarray(reconstruction)
    if reference_array.ndim != 4 or reconstruction_array.ndim != 4:
        raise ValueError(
            "cine metrics require reference and reconstruction [S,T,X,Y]"
        )
    if reference_array.shape != reconstruction_array.shape:
        raise ValueError(
            "reference and reconstruction cine shapes differ: "
            f"{reference_array.shape} vs {reconstruction_array.shape}"
        )
    resolved_protocol = resolve_metric_protocol(protocol)
    reference_for_metrics: np.ndarray
    reconstruction_for_metrics: np.ndarray
    scale_fields: dict[str, Any] = {}
    if resolved_protocol == METRIC_PROTOCOL_V2:
        reference_for_metrics, reference_scale = normalize_subject_magnitude_p995(
            reference_array, source="reference"
        )
        reconstruction_for_metrics, candidate_scale = (
            normalize_subject_magnitude_p995(
                reconstruction_array, source=str(model)
            )
        )
        scale_fields = {
            "reference_p99_5": reference_scale["p99_5"],
            "candidate_p99_5": candidate_scale["p99_5"],
            "candidate_to_reference_p99_5_ratio": (
                candidate_scale["p99_5"] / reference_scale["p99_5"]
            ),
            "reference_normalized_max": reference_scale["normalized_max"],
            "candidate_normalized_max": candidate_scale["normalized_max"],
            "reference_fraction_above_1": reference_scale[
                "fraction_of_values_above_1"
            ],
            "candidate_fraction_above_1": candidate_scale[
                "fraction_of_values_above_1"
            ],
            "scientific_clipping": False,
        }
    else:
        reference_for_metrics = reference_array
        reconstruction_for_metrics = reconstruction_array

    rows: list[dict[str, Any]] = []
    slices, phases = reference_array.shape[:2]
    for slice_index in range(slices):
        for phase_index in range(phases):
            if resolved_protocol == METRIC_PROTOCOL_V2:
                metrics = compute_normalized_frame_metrics_v2(
                    reference_for_metrics[slice_index, phase_index],
                    reconstruction_for_metrics[slice_index, phase_index],
                )
            else:
                metrics = compute_frame_metrics(
                    reference_for_metrics[slice_index, phase_index],
                    reconstruction_for_metrics[slice_index, phase_index],
                )
            rows.append(
                {
                    "subject": subject,
                    "model": model,
                    "acceleration_requested": acceleration_requested,
                    "acceleration_actual": acceleration_actual,
                    "sampling_method": sampling_method,
                    "seed": seed,
                    "metric_protocol_version": resolved_protocol,
                    "slice": slice_index,
                    "phase": phase_index,
                    "flat_frame_index": slice_index * phases + phase_index,
                    "nrmse": metrics["nrmse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "reference_data_range": metrics["reference_data_range"],
                    **scale_fields,
                }
            )
    return rows


def _condition_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty frame-metric table")
    fields = list(CONDITION_IDENTITY_FIELDS)
    fields.extend(
        field
        for field in OPTIONAL_PROTOCOL_IDENTITY_FIELDS
        if field in rows[0]
    )
    identity = {field: rows[0][field] for field in fields}
    for row in rows[1:]:
        for field, expected in identity.items():
            if field not in row or row[field] != expected:
                raise ValueError(
                    "subject summary rows mix conditions: "
                    f"field {field!r} has {row[field]!r} and {expected!r}"
                )
    return identity


def _descriptive_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_subject_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one subject/model/sampling condition with equal frame weight."""

    identity = _condition_identity(rows)
    summary: dict[str, Any] = {
        **identity,
        "frame_count": len(rows),
        "primary_aggregation": "equal arithmetic mean over all S*T frames",
    }
    for metric in PRIMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        for statistic, value in _descriptive_statistics(values).items():
            summary[f"{statistic}_{metric}"] = value
    return summary


def summarize_frame_metric_distribution(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return secondary descriptive frame statistics for one condition."""

    identity = _condition_identity(rows)
    return {
        **identity,
        "frame_count": len(rows),
        "role": "secondary frame-level diagnostic; subject mean remains primary",
        "metrics": {
            metric: _descriptive_statistics(
                np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            )
            for metric in PRIMARY_METRICS
        },
    }


def compute_paired_frame_win_rates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare RAM with ZeroFilled on matched slice-phase frames.

    Frame wins are descriptive only and are not independent-subject evidence.
    """

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["subject"]), float(row["acceleration_requested"]))
        grouped.setdefault(key, []).append(row)
    results: list[dict[str, Any]] = []
    for (subject, acceleration), condition_rows in sorted(grouped.items()):
        by_model: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        for row in condition_rows:
            model = str(row["model"])
            frame_key = (int(row["slice"]), int(row["phase"]))
            if frame_key in by_model.setdefault(model, {}):
                raise ValueError(
                    f"duplicate {model} frame {frame_key} at R={acceleration:g}"
                )
            by_model[model][frame_key] = row
        required = {"ram", "zero_filled"}
        if set(by_model) != required:
            raise ValueError(
                f"paired wins require exactly {sorted(required)} at "
                f"R={acceleration:g}; received {sorted(by_model)}"
            )
        frame_keys = set(by_model["ram"])
        if frame_keys != set(by_model["zero_filled"]):
            raise ValueError(
                f"RAM and ZeroFilled frame keys differ at R={acceleration:g}"
            )
        total = len(frame_keys)
        if total == 0:
            raise ValueError(f"no paired frames at R={acceleration:g}")
        metric_results: dict[str, dict[str, Any]] = {}
        for metric in PRIMARY_METRICS:
            if metric == "nrmse":
                wins = sum(
                    float(by_model["ram"][key][metric])
                    < float(by_model["zero_filled"][key][metric])
                    for key in frame_keys
                )
                direction = "RAM < ZeroFilled"
            else:
                wins = sum(
                    float(by_model["ram"][key][metric])
                    > float(by_model["zero_filled"][key][metric])
                    for key in frame_keys
                )
                direction = "RAM > ZeroFilled"
            metric_results[metric] = {
                "ram_wins": int(wins),
                "total_paired_frames": total,
                "percentage": float(100.0 * wins / total),
                "win_definition": direction,
            }
        results.append(
            {
                "subject": subject,
                "acceleration_requested": acceleration,
                "role": "secondary paired-frame diagnostic; no significance test",
                "metrics": metric_results,
            }
        )
    return results


def summarize_phase_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe each phase across slices without changing primary endpoints."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in CONDITION_IDENTITY_FIELDS) + (
            int(row["phase"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, phase_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary: dict[str, Any] = {
            **dict(zip(CONDITION_IDENTITY_FIELDS, key[:-1])),
            "phase": key[-1],
            "slice_count": len(phase_rows),
            "role": "secondary phase-level diagnostic across slices",
        }
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in phase_rows], dtype=np.float64
            )
            for statistic, value in _descriptive_statistics(values).items():
                summary[f"{statistic}_{metric}"] = value
        summaries.append(summary)
    return summaries
