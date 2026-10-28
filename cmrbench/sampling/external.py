"""Explicit user/external mask sampling path.

Compact layouts are accepted only when the caller names the layout.  No axis
is inferred from dimension sizes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from cmrbench.sampling.base import KSpaceShape, SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
)


CANONICAL_MASK_LAYOUT = "S,T,Kx,Ky"
EXPLICIT_COMPACT_LAYOUTS = ("Ky,T", "T,Ky", "Kx,Ky", "T,Kx,Ky")


def _strict_binary_mask(mask: object) -> np.ndarray:
    value = np.asarray(mask)
    if not (
        np.issubdtype(value.dtype, np.bool_)
        or np.issubdtype(value.dtype, np.number)
    ):
        raise TypeError("external mask must be Boolean or numeric")
    if not np.isfinite(value).all():
        raise ValueError("external mask contains NaN or Inf")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError(
            "external numeric mask must contain only exact 0/1 values; "
            "continuous density maps are not silently binarized"
        )
    binary = value.astype(bool, copy=True)
    if not np.any(binary):
        raise ValueError("external mask has no sampled support")
    return binary


def canonicalize_external_mask(
    mask: object,
    *,
    shape: KSpaceShape,
    layout: str = CANONICAL_MASK_LAYOUT,
) -> np.ndarray:
    """Convert one explicitly declared mask layout to ``[S,T,Kx,Ky]``."""

    if len(shape) != 5:
        raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
    slices, frames, kx, ky, _coils = (int(item) for item in shape)
    binary = _strict_binary_mask(mask)
    expected_canonical = (slices, frames, kx, ky)

    if layout == CANONICAL_MASK_LAYOUT:
        if binary.shape != expected_canonical:
            raise ValueError(
                f"external canonical mask must have shape {expected_canonical}; "
                f"received {binary.shape}"
            )
        return binary
    if layout == "Ky,T":
        expected = (ky, frames)
        if binary.shape != expected:
            raise ValueError(f"layout Ky,T requires shape {expected}; received {binary.shape}")
        native = binary.T[:, None, :]
        return np.broadcast_to(native[None], expected_canonical).copy()
    if layout == "T,Ky":
        expected = (frames, ky)
        if binary.shape != expected:
            raise ValueError(f"layout T,Ky requires shape {expected}; received {binary.shape}")
        native = binary[:, None, :]
        return np.broadcast_to(native[None], expected_canonical).copy()
    if layout == "Kx,Ky":
        expected = (kx, ky)
        if binary.shape != expected:
            raise ValueError(f"layout Kx,Ky requires shape {expected}; received {binary.shape}")
        return np.broadcast_to(binary[None, None], expected_canonical).copy()
    if layout == "T,Kx,Ky":
        expected = (frames, kx, ky)
        if binary.shape != expected:
            raise ValueError(
                f"layout T,Kx,Ky requires shape {expected}; received {binary.shape}"
            )
        return np.broadcast_to(binary[None], expected_canonical).copy()
    supported = (CANONICAL_MASK_LAYOUT, *EXPLICIT_COMPACT_LAYOUTS)
    raise ValueError(
        f"unsupported external mask layout {layout!r}; expected one of {supported}. "
        "Compact axes are never auto-detected."
    )


class ExternalMaskSamplingProtocol(SamplingProtocol):
    """Use a caller-supplied binary mask through the common sampling boundary."""

    def __init__(
        self,
        *,
        mask: object | None = None,
        mask_path: str | Path | None = None,
        layout: str = CANONICAL_MASK_LAYOUT,
        source: str = "external",
        implementation: str = "external",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if source not in {"external", "precomputed"}:
            raise ValueError("external mask source must be 'external' or 'precomputed'")
        if (mask is None) == (mask_path is None):
            raise ValueError("provide exactly one of external mask or mask_path")
        self.mask_path = Path(mask_path) if mask_path is not None else None
        self.mask = mask
        self.layout = str(layout)
        self.source = source
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError("external mask implementation must be a non-empty string")
        self.implementation = implementation.strip()
        self.parameters = dict(parameters or {})
        declared_actual = self.parameters.pop("actual_acceleration", None)
        if declared_actual is not None:
            self.parameters["declared_actual_acceleration_ignored"] = declared_actual

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if acceleration < 1:
            raise ValueError("requested acceleration must be >= 1")
        mask, file_metadata = self._load_mask()
        canonical = canonicalize_external_mask(mask, shape=shape, layout=self.layout)
        measured = calculate_actual_acceleration(canonical)
        if measured is None:
            raise ValueError("external mask has no sampled support")
        return SamplingResult(
            mask=canonical,
            requested_acceleration=float(acceleration),
            actual_acceleration=measured,
            method="external_mask",
            seed=seed,
            parameters={
                "mask_source": self.source,
                "input_layout": self.layout,
                "input_shape": list(np.asarray(mask).shape),
                "canonical_layout": "[S,T,Kx,Ky]",
                "canonical_shape": list(canonical.shape),
                "mask_dtype": str(canonical.dtype),
                "sampled_fraction": float(np.mean(canonical)),
                "binarization": "strict exact 0/1; no thresholding",
                **file_metadata,
                **self.parameters,
                **kwargs,
                "implementation": self.implementation,
                **binary_mask_provenance(canonical),
            },
        )

    def _load_mask(self) -> tuple[np.ndarray | object, dict[str, Any]]:
        if self.mask_path is None:
            return self.mask, {}
        path = self.mask_path.expanduser().resolve()
        if path.suffix.lower() != ".npy":
            raise ValueError("external mask_path must identify a .npy file")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"external mask file is absent or symlinked: {path}")
        value = np.load(path, allow_pickle=False)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return value, {
            "mask_path": str(path),
            "mask_file_bytes": int(path.stat().st_size),
            "mask_file_sha256": digest.hexdigest(),
        }
