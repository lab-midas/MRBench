"""Sampling mask and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from cmrbench.data.sample import Tensor, _shape


def calculate_actual_acceleration(mask: Tensor) -> float | None:
    """Compute total grid locations divided by sampled locations."""

    mask_array = np.asarray(mask)
    acquired = int(np.count_nonzero(mask_array))
    if acquired == 0:
        return None
    return float(mask_array.size / acquired)


def binary_mask_provenance(mask: Tensor) -> dict[str, int | str]:
    """Return stable support counts and SHA256 without changing caller metadata."""

    binary = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8)
    return {
        "binary_mask_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
        "sampled_points": int(np.count_nonzero(binary)),
        "total_points": int(binary.size),
    }


def validate_physical_mask(mask: Tensor) -> np.ndarray:
    """Validate only model-independent Cartesian-mask properties.

    A complete central ACS is deliberately not part of this contract.
    """

    value = np.asarray(mask)
    if not (
        np.issubdtype(value.dtype, np.bool_)
        or np.issubdtype(value.dtype, np.number)
    ):
        raise TypeError("physical sampling mask must be Boolean or numeric")
    if not np.isfinite(value).all():
        raise ValueError("physical sampling mask contains NaN or Inf")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("physical sampling mask must contain only exact 0/1 values")
    binary = value.astype(bool, copy=False)
    if not np.any(binary):
        raise ValueError("physical sampling mask has no sampled support")
    return binary


def detect_complete_central_acs(mask: Tensor) -> dict[str, Any]:
    """Measure globally complete centered support without requiring it.

    Extents are the largest centered Ky strip and square that are fully sampled
    across every slice and time frame.  Zero means no such central support.
    """

    binary = validate_physical_mask(mask)
    if binary.ndim != 4:
        raise ValueError("ACS detection requires canonical [S,T,Kx,Ky] mask")
    _slices, _frames, kx, ky = binary.shape
    full_ky = np.all(binary, axis=(0, 1, 2))
    spatial_support = np.all(binary, axis=(0, 1))
    center_indices = {(ky - 1) // 2, ky // 2}
    strip_extent = 0
    index = 0
    while index < ky:
        if not full_ky[index]:
            index += 1
            continue
        end = index + 1
        while end < ky and full_ky[end]:
            end += 1
        if any(index <= center < end for center in center_indices):
            strip_extent = max(strip_extent, end - index)
        index = end

    square_extent = 0
    for extent in range(min(kx, ky), 0, -1):
        starts_x = {(kx - extent) // 2, (kx - extent + 1) // 2}
        starts_y = {(ky - extent) // 2, (ky - extent + 1) // 2}
        if any(
            np.all(spatial_support[x : x + extent, y : y + extent])
            for x in starts_x
            for y in starts_y
        ):
            square_extent = extent
            break

    geometry = (
        "centered_ky_strip"
        if strip_extent > 0
        else "centered_square"
        if square_extent > 0
        else "none"
    )
    return {
        "has_complete_central_acs": strip_extent > 0 or square_extent > 0,
        "detected_acs_geometry": geometry,
        "detected_centered_ky_strip_extent": int(strip_extent),
        "detected_centered_square_extent": int(square_extent),
        "acs_is_generic_requirement": False,
    }


def validate_declared_central_acs(
    mask: Tensor, *, ncalib: int, geometry: str, source: str
) -> dict[str, Any]:
    """Enforce an ACS region promised by one particular generator.

    This is deliberately separate from :func:`validate_physical_mask`: only a
    source that declares an ACS contract should call it.
    """

    if isinstance(ncalib, bool) or int(ncalib) != ncalib or int(ncalib) < 1:
        raise ValueError("declared generator ncalib must be a positive integer")
    measured = detect_complete_central_acs(mask)
    expected = int(ncalib)
    if geometry == "centered_ky_strip":
        observed = int(measured["detected_centered_ky_strip_extent"])
    elif geometry == "centered_square":
        observed = int(measured["detected_centered_square_extent"])
    else:
        raise ValueError(f"unsupported declared ACS geometry {geometry!r}")
    if observed < expected:
        raise RuntimeError(
            f"{source} declared {geometry} ncalib={expected}, but the generated "
            f"mask contains only extent={observed}"
        )
    return {
        "generator_acs_contract_enforced": True,
        "declared_acs_geometry": geometry,
        "declared_ncalib": expected,
        "detected_extent_for_declared_geometry": observed,
    }


@dataclass
class SamplingResult:
    mask: Tensor
    requested_acceleration: float | None
    actual_acceleration: float | None
    method: str
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    physical_mask_validation: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mask_shape = _shape(self.mask, "sampling mask")
        if len(mask_shape) != 4:
            raise ValueError(
                "sampling mask must have canonical [S,T,Kx,Ky] shape; "
                f"received {mask_shape}"
            )
        if not self.method:
            raise ValueError("sampling method must be non-empty")
        binary = validate_physical_mask(self.mask)
        measured = calculate_actual_acceleration(binary)
        if measured is None:
            raise ValueError("physical sampling mask has no sampled support")
        if self.actual_acceleration is None or not np.isclose(
            float(self.actual_acceleration), measured, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "sampling actual_acceleration is inconsistent with the physical mask"
            )
        self.physical_mask_validation = {
            "finite": True,
            "binary": True,
            "nonempty": True,
            "canonical_shape": list(mask_shape),
            "actual_acceleration_consistent": True,
            **detect_complete_central_acs(binary),
        }

    def as_metadata(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        parameters.setdefault(
            "physical_mask_validation", dict(self.physical_mask_validation)
        )
        return {
            "sampling_method": self.method,
            "sampling_implementation": self.parameters.get("implementation"),
            "requested_acceleration": self.requested_acceleration,
            "actual_acceleration": self.actual_acceleration,
            "sampling_seed": self.seed,
            "sampling_parameters": parameters,
        }
