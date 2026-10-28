"""Robust display and historical normalized-magnitude diagnostic helpers.

These p99.5 utilities are display-only or explicitly historical diagnostics.
CMRBench primary Metrics v2 implement independent per-source subject-level
p99.5 normalization separately in ``reconstruction_metrics``; these helpers
must not be substituted into that scientific protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal

import numpy as np


ROBUST_MAGNITUDE_PERCENTILE = 99.5
DEFAULT_ZERO_SCALE_THRESHOLD = 1e-12
NormalizationUnit = Literal["subject", "single_frame"]

PSNR_SSIM_SPECIFICATION: dict[str, Any] = {
    "protocol_version": "cmrbench_recon_metrics_v1",
    "domain": "raw_magnitude",
    "data_range": "reference_max_minus_min",
    "psnr_formula": "10 * log10(reference_data_range^2 / MSE)",
    "ssim_data_range": "reference_max_minus_min",
    "candidate_specific_normalization": False,
    "p99_5_primary_metric_usage": False,
    "perfect_reconstruction_psnr": float("inf"),
    "perfect_reconstruction_ssim": 1.0,
    "implementation_status": "FROZEN AND IMPLEMENTED",
}


@dataclass(frozen=True)
class IndependentMagnitudePair:
    reference: np.ndarray
    candidate: np.ndarray
    reference_scale: float
    candidate_scale: float
    percentile: float
    normalization_unit: NormalizationUnit
    diagnostic_label: str


def robust_magnitude_scale(
    values: np.ndarray,
    *,
    percentile: float = ROBUST_MAGNITUDE_PERCENTILE,
    zero_threshold: float = DEFAULT_ZERO_SCALE_THRESHOLD,
) -> float:
    """Return one robust magnitude scale over every supplied array element."""

    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("cannot normalize an empty magnitude array")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0,100]")
    if not np.all(np.isfinite(array)):
        raise ValueError("magnitude normalization requires finite scientific input")
    scale = float(np.percentile(np.abs(array), percentile))
    if not np.isfinite(scale) or scale <= zero_threshold:
        raise ValueError(
            f"magnitude p{percentile:g} scale {scale!r} is at or below the "
            f"zero-scale threshold {zero_threshold:g}"
        )
    return scale


def normalize_magnitude_for_evaluation(
    values: np.ndarray,
    *,
    percentile: float = ROBUST_MAGNITUDE_PERCENTILE,
    zero_threshold: float = DEFAULT_ZERO_SCALE_THRESHOLD,
) -> tuple[np.ndarray, float]:
    """Return ``clip(abs(values)/p99.5, 0, 1)`` and its scalar scale."""

    scale = robust_magnitude_scale(
        values, percentile=percentile, zero_threshold=zero_threshold
    )
    normalized = np.abs(np.asarray(values)).astype(np.float64, copy=False) / scale
    return np.clip(normalized, 0.0, 1.0), scale


def _validate_normalization_unit(array: np.ndarray, unit: NormalizationUnit) -> None:
    expected_ndim = 4 if unit == "subject" else 2
    expected_shape = "[S,T,X,Y]" if unit == "subject" else "[X,Y]"
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{unit} normalization requires {expected_shape}, received {array.shape}"
        )


def prepare_independent_magnitude_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    normalization_unit: NormalizationUnit = "subject",
    percentile: float = ROBUST_MAGNITUDE_PERCENTILE,
    zero_threshold: float = DEFAULT_ZERO_SCALE_THRESHOLD,
) -> IndependentMagnitudePair:
    """Normalize reference/candidate independently using one scale per unit."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "reference and candidate shapes differ: "
            f"{reference_array.shape} vs {candidate_array.shape}"
        )
    _validate_normalization_unit(reference_array, normalization_unit)
    reference_eval, reference_scale = normalize_magnitude_for_evaluation(
        reference_array,
        percentile=percentile,
        zero_threshold=zero_threshold,
    )
    candidate_eval, candidate_scale = normalize_magnitude_for_evaluation(
        candidate_array,
        percentile=percentile,
        zero_threshold=zero_threshold,
    )
    diagnostic_label = (
        "subject-level independently normalized magnitude"
        if normalization_unit == "subject"
        else "single-frame normalized-magnitude diagnostic"
    )
    return IndependentMagnitudePair(
        reference=reference_eval,
        candidate=candidate_eval,
        reference_scale=reference_scale,
        candidate_scale=candidate_scale,
        percentile=percentile,
        normalization_unit=normalization_unit,
        diagnostic_label=diagnostic_label,
    )


def anatomical_cine_nrmse(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    normalization_unit: NormalizationUnit = "subject",
) -> dict[str, Any]:
    """Compute a historical independently normalized magnitude diagnostic.

    This function is not a CMRBench primary reconstruction metric.
    """

    pair = prepare_independent_magnitude_pair(
        reference, candidate, normalization_unit=normalization_unit
    )
    difference = pair.candidate - pair.reference
    denominator = float(np.linalg.norm(pair.reference.reshape(-1), ord=2))
    if denominator <= DEFAULT_ZERO_SCALE_THRESHOLD:
        raise ValueError("normalized reference has effectively zero L2 norm")
    nrmse = float(np.linalg.norm(difference.reshape(-1), ord=2) / denominator)
    return {
        "nrmse": nrmse,
        "reference_scale": pair.reference_scale,
        "candidate_scale": pair.candidate_scale,
        "percentile": pair.percentile,
        "normalization_unit": pair.normalization_unit,
        "diagnostic_label": (
            f"{pair.diagnostic_label}; NOT CMRBench primary metrics"
        ),
        "domain": "magnitude",
        "clipping": [0.0, 1.0],
    }


def array_sha256(values: np.ndarray) -> str:
    """Hash dtype, shape, and exact contiguous array bytes deterministically."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def compare_complex_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> dict[str, Any]:
    """Compare unnormalized scientific arrays, preserving complex information."""

    expected = np.asarray(reference)
    actual = np.asarray(candidate)
    result: dict[str, Any] = {
        "reference_shape": list(expected.shape),
        "candidate_shape": list(actual.shape),
        "reference_dtype": str(expected.dtype),
        "candidate_dtype": str(actual.dtype),
        "reference_sha256": array_sha256(expected),
        "candidate_sha256": array_sha256(actual),
        "shape_match": expected.shape == actual.shape,
    }
    if expected.shape != actual.shape:
        result.update({"array_equal": False, "allclose": False})
        return result
    difference = actual - expected
    error_energy = float(np.sum(np.abs(difference) ** 2, dtype=np.float64))
    reference_energy = float(np.sum(np.abs(expected) ** 2, dtype=np.float64))
    nmse = error_energy / reference_energy if reference_energy else float("inf")
    result.update(
        {
            "array_equal": bool(np.array_equal(expected, actual)),
            "number_of_different_entries": int(np.count_nonzero(expected != actual)),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "normalized_l2": float(np.sqrt(nmse)),
            "nmse": nmse,
            "rtol": rtol,
            "atol": atol,
            "allclose": bool(np.allclose(expected, actual, rtol=rtol, atol=atol)),
        }
    )
    return result


def magnitude_statistics(values: np.ndarray) -> dict[str, float]:
    magnitude = np.abs(np.asarray(values)).astype(np.float64, copy=False)
    if magnitude.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "min": float(np.min(magnitude)),
        "max": float(np.max(magnitude)),
        "mean": float(np.mean(magnitude)),
        "median": float(np.median(magnitude)),
        "p95": float(np.percentile(magnitude, 95)),
        "p99": float(np.percentile(magnitude, 99)),
        "p99_5": float(np.percentile(magnitude, 99.5)),
    }
