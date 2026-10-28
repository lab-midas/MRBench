"""Source-faithful NVIDIA/MONAI k-t sampling generators.

Portions of the low-level generator logic are adapted from
``scripts/mri_data/ktSampling.py`` in NVIDIA NV-Raw2Insights-MRI commit
``99be712b2e9fd34bf6a2f709373d311354fbda96``.  The upstream file states:

    Copyright (c) MONAI Consortium
    Licensed under the Apache License, Version 2.0

The project-level upstream LICENSE is Apache-2.0.  The public protocol below
also preserves the released ``scripts/transforms.py`` wrapper semantics.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Any

import numpy as np

from cmrbench.sampling.base import KSpaceShape, SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
    validate_declared_central_acs,
)


UPSTREAM_REPOSITORY = "https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI"
UPSTREAM_COMMIT = "99be712b2e9fd34bf6a2f709373d311354fbda96"
UPSTREAM_SOURCE = "scripts/mri_data/ktSampling.py"
UPSTREAM_SOURCE_SHA256 = (
    "1c6a71ffe862a88182c36b2eddc955e854b047cfe6a98e1b0b0e2ef36f278db0"
)
UPSTREAM_WRAPPER = "scripts/transforms.py"
UPSTREAM_LICENSE = "Apache-2.0"
SUPPORTED_NVIDIA_GENERATORS = ("uniform", "kt_uniform", "kt_gaussian", "kt_radial")
UPSTREAM_TRANSFORMS_SHA256 = (
    "664a619d9869ccf46bbf1f1e80e97ed820396f2b7506e1af1a2bbdadbc7b9a2e"
)


def _center_crop(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    slices = tuple(
        slice((dim - target) // 2, (dim - target) // 2 + target)
        if target % 2 == 0
        else slice((dim - target) // 2 + 1, (dim - target) // 2 + 1 + target)
        for dim, target in zip(x.shape, target_shape)
    )
    return x[slices]


def _zero_pad(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    x = np.array(x)
    if x.shape == target_shape:
        return x
    result = np.zeros(target_shape, dtype=x.dtype)
    slices = []
    for original, target in zip(x.shape, target_shape):
        start = (target - original) // 2
        slices.append(slice(start, start + original))
    result[tuple(slices)] = x
    return result


def _ind_to_xy(ind: np.ndarray | int, x_size: int) -> tuple[np.ndarray, np.ndarray]:
    ind = np.asarray(ind)
    x = ((ind - 1) % x_size) + 1
    y = ((ind - 1) // x_size) + 1
    return x, y


def _nearest_vacant(duplicate: int, empty: np.ndarray, ny: int) -> int:
    x0, y0 = _ind_to_xy(np.array([duplicate]), ny)
    x0, y0 = x0[0], y0[0]
    x, y = _ind_to_xy(empty, ny)
    same_y = np.abs(y - y0) < 1e-10
    candidate_indices = np.where(same_y)[0] if np.any(same_y) else np.arange(len(empty))
    distances = np.abs(x[candidate_indices] - x0)
    return int(empty[candidate_indices[np.argmin(distances)]])


def _resolve_duplicate_indices(
    phase: np.ndarray, time: np.ndarray, ny: int, nt: int
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.array(phase, dtype=np.int64).copy()
    time = np.array(time, dtype=np.int64).copy()
    offset_x = int(ceil((ny + 1) / 2))
    offset_t = int(ceil((nt + 1) / 2))
    phase += offset_x
    time += offset_t
    linear = (time - 1) * ny + phase
    unique, counts = np.unique(linear, return_counts=True)
    duplicate_indices: list[int] = []
    for repeated in unique[counts != 1]:
        indices = np.where(linear == repeated)[0]
        if indices.size > 1:
            duplicate_indices.extend(indices[1:])
    empty = np.setdiff1d(np.arange(1, ny * nt + 1), linear)
    for index in np.array(duplicate_indices, dtype=np.int64):
        replacement = _nearest_vacant(int(linear[index]), empty, ny)
        linear[index] = replacement
        empty = empty[empty != replacement]
    new_phase, new_time = _ind_to_xy(linear, ny)
    new_phase -= offset_x
    new_time -= offset_t
    return new_phase, new_time


def _random_weighted_choice(
    probability: np.ndarray, seed: int, size: int | tuple[int, ...]
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64).flatten()
    generator = np.random.default_rng(seed)
    return generator.choice(
        np.arange(1, len(probability) + 1),
        size=size,
        p=probability / np.sum(probability),
    )


def _round_away_from_zero(value: float) -> int:
    return floor(value + 0.5) if value > 0 else ceil(value - 0.5)


def uniform_sampling(nx: int, ny: int, nt: int, ncalib: int, acceleration: int) -> np.ndarray:
    mask = _zero_pad(np.ones((nx, ncalib), dtype=np.float64), (nx, ny))
    mask[:, ::acceleration] = 1
    return mask[..., np.newaxis].repeat(nt, -1)


def kt_uniform_sampling(
    nx: int, ny: int, nt: int, ncalib: int, acceleration: int
) -> np.ndarray:
    phase_pattern = np.zeros(ny)
    phase_pattern[::acceleration] = 1
    time_pattern = np.zeros(nt)
    time_pattern[::acceleration] = 1
    row = np.arange(ny)[:, None]
    column = np.arange(nt)[None, :]
    difference = row - column
    toeplitz = np.empty_like(difference, dtype=phase_pattern.dtype)
    nonnegative = difference >= 0
    toeplitz[nonnegative] = phase_pattern[difference[nonnegative]]
    toeplitz[~nonnegative] = time_pattern[-difference[~nonnegative]]
    rows, columns = np.nonzero(toeplitz)
    phase = rows - ny // 2
    time = columns - nt // 2
    phase, time = _resolve_duplicate_indices(phase, time, ny, nt)
    compact = np.zeros((ny, nt))
    row_indices = np.clip(phase + ny // 2, 0, ny - 1)
    column_indices = np.clip(time + nt // 2, 0, nt - 1)
    compact[row_indices, column_indices] = 1
    acs = _zero_pad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    native = np.tile(compact, (nx, 1, 1))
    return (native + acs > 0).astype(np.float64)


def kt_gaussian_sampling(
    nx: int,
    ny: int,
    nt: int,
    ncalib: int,
    acceleration: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    positions = np.arange(-ny // 2, int(ceil(ny / 2)))
    lines_per_frame = int(round(ny / acceleration))
    sigma = ny / 5.0
    probability = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(
        -(positions**2) / (sigma**2)
    )
    generator = np.random.default_rng(seed)
    frame_seeds = np.round(generator.random(nt) * 1e6).astype(np.int64)
    phase_list: list[int] = []
    time_list: list[int] = []
    offset_t = nt // 2
    offset_ny = ny // 2
    for time_value in range(-int(floor(nt / 2)), int(ceil(nt / 2))):
        phase_values = _random_weighted_choice(
            probability,
            int(frame_seeds[time_value + offset_t]),
            (lines_per_frame,),
        ) - (offset_ny + 1)
        time_list.extend([time_value] * lines_per_frame)
        phase_list.extend(phase_values.tolist())
    phase, time = _resolve_duplicate_indices(
        np.array(phase_list, dtype=np.int64),
        np.array(time_list, dtype=np.int64),
        ny,
        nt,
    )
    compact = np.zeros((ny, nt))
    row_indices = np.clip(phase + offset_ny, 0, ny - 1)
    column_indices = np.clip(time + offset_t, 0, nt - 1)
    compact[row_indices, column_indices] = 1
    acs = _zero_pad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    native = np.tile(compact, (nx, 1, 1))
    return (native + acs > 0).astype(np.float64)


def kt_radial_sampling(
    nx: int,
    ny: int,
    nt: int,
    ncalib: int,
    acceleration: float,
    angle4next: float,
    cropcorner: bool,
) -> np.ndarray:
    try:
        from scipy.ndimage import rotate
    except ImportError as exc:
        raise RuntimeError("kt_radial generation requires scipy") from exc
    beams = floor((1.0 / acceleration) * 180)
    if beams < 1:
        raise ValueError("resolved radial acceleration produces zero beams")
    side = max(nx, ny) if cropcorner else int(ceil(np.sqrt(2) * max(nx, ny)))
    auxiliary = np.zeros((side, side))
    auxiliary[int(_round_away_from_zero(side / 2)) - 1, :] = 1
    angle_step = 180.0 / beams
    native = np.zeros((nx, ny, nt))
    for frame in range(nt):
        offset = angle4next * frame
        angles = np.arange(offset, 180 + offset + 1e-9, angle_step)
        image = np.zeros((nx, ny))
        for angle in angles:
            rotated = rotate(
                auxiliary,
                angle,
                reshape=False,
                order=0,
                mode="grid-constant",
                cval=0,
            )
            image += _center_crop(rotated, (nx, ny))
        native[:, :, frame] = image
    acs = _zero_pad(np.ones((ncalib, ncalib, nt)), (nx, ny, nt))
    return (native + acs > 0).astype(np.float64)


def nvidia_random_line_mask(
    num_cols: int, center_fraction: float, acceleration: int, seed: int
) -> np.ndarray:
    """Pinned ``RandomKspaceMask`` 2D-complex branch as a NumPy reference."""

    num_low_freqs = int(round(num_cols * center_fraction))
    if num_low_freqs < 1 or num_low_freqs >= num_cols:
        raise ValueError("center_fraction must resolve to 1..Ky-1 center lines")
    probability = (num_cols / acceleration - num_low_freqs) / (
        num_cols - num_low_freqs
    )
    if probability < 0 or probability > 1:
        raise ValueError(
            "center_fraction and acceleration imply an invalid outer sampling probability"
        )
    mask = np.random.RandomState(int(seed)).uniform(size=num_cols) < probability
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = True
    return mask


def nvidia_equispaced_line_mask(
    num_cols: int, center_fraction: float, acceleration: int, seed: int
) -> tuple[np.ndarray, float, int]:
    """Pinned ``EquispacedKspaceMask`` 2D-complex branch."""

    num_low_freqs = int(round(num_cols * center_fraction))
    if num_low_freqs < 1 or num_low_freqs >= num_cols:
        raise ValueError("center_fraction must resolve to 1..Ky-1 center lines")
    denominator = num_low_freqs * acceleration - num_cols
    if denominator == 0:
        raise ValueError("center_fraction creates a singular adjusted acceleration")
    adjusted = (acceleration * (num_low_freqs - num_cols)) / denominator
    if not np.isfinite(adjusted) or adjusted <= 0:
        raise ValueError("center_fraction creates an invalid adjusted acceleration")
    high = int(round(adjusted))
    if high < 1:
        raise ValueError("adjusted acceleration has no valid integer offset range")
    rng = np.random.RandomState(int(seed))
    offset = int(rng.randint(0, high))
    mask = np.zeros(num_cols, dtype=bool)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = True
    samples = np.around(np.arange(offset, num_cols - 1, adjusted)).astype(np.uint64)
    mask[samples] = True
    return mask, float(adjusted), offset


class NVIDIAStaticLineSamplingProtocol(SamplingProtocol):
    """Source-faithful NVIDIA random/equispaced, static 1D-PE masks."""

    def __init__(self, *, family: str, center_fraction: float) -> None:
        if family not in {"random", "equispaced"}:
            raise ValueError("NVIDIA static line family must be random or equispaced")
        if not 0 < float(center_fraction) < 1:
            raise ValueError("center_fraction must be strictly between zero and one")
        self.family = family
        self.center_fraction = float(center_fraction)

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if kwargs:
            raise ValueError(
                "unsupported NVIDIA static-line parameters: " + ", ".join(sorted(kwargs))
            )
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if seed is None:
            raise ValueError(f"NVIDIA {self.family} requires an explicit recorded seed")
        if not float(acceleration).is_integer() or float(acceleration) < 1:
            raise ValueError("NVIDIA random/equispaced acceleration must be a positive integer")
        slices, frames, kx, ky, _coils = (int(value) for value in shape)
        nominal = int(float(acceleration))
        if self.family == "random":
            line = nvidia_random_line_mask(
                ky, self.center_fraction, nominal, int(seed)
            )
            resolved_extra: dict[str, Any] = {
                "outer_probability": (
                    (ky / nominal - int(round(ky * self.center_fraction)))
                    / (ky - int(round(ky * self.center_fraction)))
                )
            }
        else:
            line, adjusted, offset = nvidia_equispaced_line_mask(
                ky, self.center_fraction, nominal, int(seed)
            )
            resolved_extra = {"adjusted_acceleration": adjusted, "offset": offset}
        canonical = np.broadcast_to(
            line[None, None, None, :], (slices, frames, kx, ky)
        ).copy()
        ncalib = int(round(ky * self.center_fraction))
        acs_contract = validate_declared_central_acs(
            canonical,
            ncalib=ncalib,
            geometry="centered_ky_strip",
            source=f"NVIDIA {self.family}",
        )
        measured = calculate_actual_acceleration(canonical)
        if measured is None:
            raise RuntimeError("generated NVIDIA line mask has no sampled support")
        return SamplingResult(
            mask=canonical,
            requested_acceleration=float(acceleration),
            actual_acceleration=measured,
            method=self.family,
            seed=int(seed),
            parameters={
                "mask_source": "generated",
                "family": self.family,
                "implementation": "sdum",
                "generator": (
                    "NV-Raw2Insights-MRI::RandomKspaceMask"
                    if self.family == "random"
                    else "NV-Raw2Insights-MRI::EquispacedKspaceMask"
                ),
                "generator_parameters": {
                    "center_fraction": self.center_fraction,
                    "resolved_num_low_frequencies": ncalib,
                    "acceleration": nominal,
                    "seed": int(seed),
                    **resolved_extra,
                },
                "ncalib": ncalib,
                "acs_parameter_semantics": "center_fraction (source-faithful)",
                "acs_geometry": "centered_ky_strip",
                **acs_contract,
                "native_layout": "[Ky]",
                "native_shape": [ky],
                "canonical_layout": "[S,T,Kx,Ky]",
                "canonical_shape": list(canonical.shape),
                "slice_broadcast": True,
                "temporal_broadcast": True,
                "deterministic": True,
                "rng": "numpy.random.RandomState(seed), matching MONAI self.R stream",
                "provenance_origin": "fastMRI-derived/modified by NVIDIA",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_source": UPSTREAM_WRAPPER,
                "upstream_source_sha256": UPSTREAM_TRANSFORMS_SHA256,
                "upstream_license": UPSTREAM_LICENSE,
                "sampling_capability_is_model_capability": False,
                **binary_mask_provenance(canonical),
            },
        )


class NVIDIASamplingProtocol(SamplingProtocol):
    """Canonical wrapper around the four pinned released generators."""

    def __init__(
        self,
        *,
        family: str,
        ncalib: int = 20,
        alpha: float = 0.2,
        angle4next: float = 137.5,
        cropcorner: bool = True,
        radial_acceleration_scale: float = 0.6,
        implementation: str = "sdum",
    ) -> None:
        if family not in SUPPORTED_NVIDIA_GENERATORS:
            raise ValueError(
                f"unknown NVIDIA generator {family!r}; expected {SUPPORTED_NVIDIA_GENERATORS}"
            )
        if ncalib < 1:
            raise ValueError("ncalib must be positive")
        self.family = family
        self.ncalib = int(ncalib)
        self.alpha = float(alpha)
        self.angle4next = float(angle4next)
        self.cropcorner = bool(cropcorner)
        self.radial_acceleration_scale = float(radial_acceleration_scale)
        if implementation not in {"sdum", "canonical"}:
            raise ValueError("NVIDIA sampling implementation label is invalid")
        if implementation == "canonical" and family != "uniform":
            raise ValueError("only Uniform has a verified canonical implementation")
        self.implementation = implementation

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if kwargs:
            raise ValueError(
                "unsupported NVIDIA sampling parameters: " + ", ".join(sorted(kwargs))
            )
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if acceleration < 1:
            raise ValueError("requested acceleration must be >= 1")
        slices, frames, kx, ky, _coils = (int(item) for item in shape)
        if self.ncalib > ky or (self.family == "kt_radial" and self.ncalib > kx):
            geometry = "Kx and Ky" if self.family == "kt_radial" else "Ky"
            raise ValueError(
                f"ncalib={self.ncalib} does not fit {self.family} {geometry} geometry"
            )
        if (
            self.implementation == "canonical"
            and ky % 2 == 0
            and self.ncalib % 2 == 1
        ):
            raise ValueError(
                "canonical Uniform is ambiguous for even Ky and odd ncalib: the pinned "
                "SDUM and CardioMM centering formulas differ by one Ky column; choose "
                "implementation='sdum' or implementation='cardiomm'"
            )
        integer_acceleration = int(acceleration)
        if self.family != "kt_radial" and not float(acceleration).is_integer():
            raise ValueError(f"{self.family} requires an integer acceleration")

        resolved: dict[str, Any] = {
            "nx": kx,
            "ny": ky,
            "nt": frames,
            "ncalib": self.ncalib,
        }
        deterministic = self.family != "kt_gaussian"
        if self.family == "uniform":
            native = uniform_sampling(kx, ky, frames, self.ncalib, integer_acceleration)
            resolved["R"] = integer_acceleration
        elif self.family == "kt_uniform":
            native = kt_uniform_sampling(kx, ky, frames, self.ncalib, integer_acceleration)
            resolved["R"] = integer_acceleration
        elif self.family == "kt_gaussian":
            if seed is None:
                raise ValueError("kt_gaussian requires an explicit recorded seed")
            native = kt_gaussian_sampling(
                kx,
                ky,
                frames,
                self.ncalib,
                integer_acceleration,
                self.alpha,
                seed,
            )
            resolved.update({"R": integer_acceleration, "alpha": self.alpha, "seed": seed})
        else:
            low_level_acceleration = float(acceleration) * self.radial_acceleration_scale
            native = kt_radial_sampling(
                kx,
                ky,
                frames,
                self.ncalib,
                low_level_acceleration,
                self.angle4next,
                self.cropcorner,
            )
            resolved.update(
                {
                    "nominal_acceleration": float(acceleration),
                    "wrapper_radial_acceleration_scale": self.radial_acceleration_scale,
                    "R": low_level_acceleration,
                    "angle4next": self.angle4next,
                    "cropcorner": self.cropcorner,
                    "beams": floor((1.0 / low_level_acceleration) * 180),
                }
            )

        native_binary = np.asarray(native) > 0
        expected_native = (kx, ky, frames)
        if native_binary.shape != expected_native:
            raise RuntimeError(
                f"upstream {self.family} returned {native_binary.shape}, expected {expected_native}"
            )
        time_kx_ky = np.transpose(native_binary, (2, 0, 1))
        canonical = np.broadcast_to(
            time_kx_ky[None], (slices, frames, kx, ky)
        ).copy()
        declared_geometry = (
            "centered_square" if self.family == "kt_radial" else "centered_ky_strip"
        )
        acs_contract = validate_declared_central_acs(
            canonical,
            ncalib=self.ncalib,
            geometry=declared_geometry,
            source=f"NVIDIA {self.family}",
        )
        measured = calculate_actual_acceleration(canonical)
        if measured is None:
            raise RuntimeError(f"generated {self.family} mask has no sampled support")
        return SamplingResult(
            mask=canonical,
            requested_acceleration=float(acceleration),
            actual_acceleration=measured,
            method=self.family,
            seed=seed,
            parameters={
                "mask_source": "generated",
                "family": self.family,
                "implementation": self.implementation,
                "generator": f"NV-Raw2Insights-MRI::{self.family}",
                "generator_parameters": resolved,
                "ncalib": self.ncalib,
                "acs_geometry": declared_geometry,
                **acs_contract,
                "native_layout": "[Kx,Ky,T]",
                "native_shape": list(native_binary.shape),
                "canonical_layout": "[S,T,Kx,Ky]",
                "canonical_shape": list(canonical.shape),
                "slice_broadcast": True,
                "temporal_broadcast": False,
                "mask_dtype": str(canonical.dtype),
                "sampled_fraction": float(np.mean(canonical)),
                "deterministic": deterministic,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_source": UPSTREAM_SOURCE,
                "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
                "upstream_wrapper": UPSTREAM_WRAPPER,
                "upstream_license": UPSTREAM_LICENSE,
                "sampling_capability_is_model_capability": False,
                "canonical_equivalence": (
                    {
                        "implementations": ["sdum", "cardiomm"],
                        "contract": (
                            "Uniform, integer R, positive ncalib<=Ky, excluding the "
                            "even-Ky/odd-ncalib centering boundary"
                        ),
                        "status": "verified",
                        "cardiomm_repository": (
                            "https://github.com/wangziblake/"
                            "CardioMM_MMCMR-427K.git"
                        ),
                        "cardiomm_commit": (
                            "abb6f8a4e293426736467c9859a5f5755c38c8aa"
                        ),
                        "cardiomm_source": (
                            "CardioMM_dataundersampling/Python_Reimpl_Version/"
                            "cardiomm_mask_generator/UniformSampling.py"
                        ),
                        "cardiomm_source_sha256": (
                            "274f0e483685139cba7abf7976bca583764efac0d9583cd205d8694e227033a0"
                        ),
                        "verified_nominal_accelerations": [4, 8, 16, 24],
                        "verified_ncalib": [8, 12, 15, 16, 20, 24],
                        "verified_geometries_kx_ky_t": [
                            [192, 156, 25],
                            [31, 29, 7],
                        ],
                        "comparisons": [
                            "array_equal",
                            "binary_sha256",
                            "sampled_points",
                            "actual_acceleration",
                            "acs_geometry",
                        ],
                    }
                    if self.implementation == "canonical"
                    else None
                ),
                **binary_mask_provenance(canonical),
            },
        )


def UniformSamplingProtocol(**parameters: Any) -> NVIDIASamplingProtocol:
    return NVIDIASamplingProtocol(family="uniform", **parameters)


def CanonicalUniformSamplingProtocol(**parameters: Any) -> NVIDIASamplingProtocol:
    return NVIDIASamplingProtocol(
        family="uniform", implementation="canonical", **parameters
    )


def KtUniformSamplingProtocol(**parameters: Any) -> NVIDIASamplingProtocol:
    return NVIDIASamplingProtocol(family="kt_uniform", **parameters)


def KtGaussianSamplingProtocol(**parameters: Any) -> NVIDIASamplingProtocol:
    return NVIDIASamplingProtocol(family="kt_gaussian", **parameters)


def KtRadialSamplingProtocol(**parameters: Any) -> NVIDIASamplingProtocol:
    return NVIDIASamplingProtocol(family="kt_radial", **parameters)


def RandomSamplingProtocol(**parameters: Any) -> NVIDIAStaticLineSamplingProtocol:
    return NVIDIAStaticLineSamplingProtocol(family="random", **parameters)


def EquispacedSamplingProtocol(**parameters: Any) -> NVIDIAStaticLineSamplingProtocol:
    return NVIDIAStaticLineSamplingProtocol(family="equispaced", **parameters)
