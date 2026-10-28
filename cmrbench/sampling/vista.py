"""Precomputed-first adapter for the established VISTA sampling convention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cmrbench.sampling.acs import add_centered_strip_acs, centered_strip_bounds
from cmrbench.sampling.base import KSpaceShape, SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
)
from cmrbench.vista import generate_vista_mask

VISTAGenerator = Callable[..., Any]
DEFAULT_VISTA_MASK_ROOT = Path("/path/to/cmrbench-assets/masks")


def precomputed_vista_mask_path(
    mask_root: str | Path,
    *,
    ky: int,
    frames: int,
    acceleration: float,
    seed: int,
) -> Path:
    """Build the exact established filename; precomputed R must be integral."""

    if not float(acceleration).is_integer():
        raise ValueError("precomputed VISTA masks require an integer acceleration")
    return Path(mask_root) / (
        f"mask_VISTA_{ky}x{frames}_acc{int(acceleration)}_{seed}.txt"
    )


def canonicalize_vista_mask(
    compact_mask: np.ndarray,
    *,
    slices: int,
    frames: int,
    kx: int,
    ky: int,
) -> np.ndarray:
    """Convert stored ``[Ky,T]`` to canonical ``[S,T,Kx,Ky]``."""

    compact = np.asarray(compact_mask)
    if compact.shape != (ky, frames):
        raise ValueError(
            f"stored VISTA mask must have [Ky,T] shape {(ky, frames)}; "
            f"received {compact.shape}"
        )
    time_ky = compact.T.astype(bool, copy=False)
    return np.broadcast_to(
        time_ky[None, :, None, :], (slices, frames, kx, ky)
    ).copy()


def centered_ky_strip_bounds(ky: int, ncalib: int) -> tuple[int, int]:
    """Return NVIDIA/SDUM-style centered Ky bounds as ``[start, stop)``."""

    if isinstance(ncalib, bool) or int(ncalib) != ncalib or int(ncalib) < 1:
        raise ValueError("VISTA configured ncalib must be a positive integer")
    width = int(ncalib)
    if width > int(ky):
        raise ValueError(f"VISTA configured ncalib={width} exceeds Ky={ky}")
    return centered_strip_bounds(int(ky), width)


class VISTASamplingProtocol(SamplingProtocol):
    """Resolve one VISTA compact source and optionally union Ky-strip ACS.

    Canonical data are ``[S,T,Kx,Ky,C]`` and VISTA always samples Ky (axis 3)
    for the current in-house cine dataset. The default leaves the source mask
    unchanged; registered sources use one shared canonicalization and ACS path.
    """

    def __init__(
        self,
        generator: VISTAGenerator | None = None,
        *,
        mask_root: str | Path | None = None,
        mask_path: str | Path | None = None,
        vista_type: str = "VISTA",
        force_acs: bool = False,
        ncalib: int | None = None,
        parameters: dict[str, Any] | None = None,
        implementation: str = "precomputed",
    ):
        if implementation not in {"precomputed", "cmrbench"}:
            raise ValueError(
                "VISTA implementation must be 'precomputed' or 'cmrbench'"
            )
        if implementation == "cmrbench" and (
            mask_root is not None or mask_path is not None
        ):
            raise ValueError(
                "VISTA implementation='cmrbench' does not accept mask_root or mask_path"
            )
        if implementation == "cmrbench" and generator is not None:
            raise ValueError(
                "VISTA implementation='cmrbench' uses generate_vista_mask directly "
                "and does not accept an injected generator"
            )
        if implementation == "cmrbench" and vista_type != "VISTA":
            raise ValueError(
                "VISTA implementation='cmrbench' requires vista_type='VISTA'"
            )
        if implementation == "cmrbench" and parameters:
            raise ValueError(
                "VISTA implementation='cmrbench' does not expose generator tuning "
                "parameters"
            )
        self.implementation = implementation
        self.generator = generator
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.mask_path = Path(mask_path) if mask_path is not None else None
        self.vista_type = vista_type
        if not isinstance(force_acs, bool):
            raise ValueError("VISTA force_acs must be Boolean")
        self.force_acs = force_acs
        if ncalib is not None and (
            isinstance(ncalib, bool) or int(ncalib) != ncalib or int(ncalib) < 1
        ):
            raise ValueError("VISTA configured ncalib must be a positive integer")
        self.configured_ncalib = int(ncalib) if ncalib is not None else None
        if self.force_acs and self.configured_ncalib is None:
            raise ValueError("VISTA force_acs=true requires an explicit ncalib")
        self.parameters = dict(parameters or {})

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if acceleration < 1:
            raise ValueError("acceleration must be >= 1")
        if seed is None:
            raise ValueError("VISTA sampling requires an explicit recorded seed")

        slices, frames, kx, ky, _coils = shape
        acs_start: int | None = None
        acs_stop: int | None = None
        if self.configured_ncalib is not None:
            acs_start, acs_stop = centered_ky_strip_bounds(
                int(ky), self.configured_ncalib
            )
        expected_path: Path | None
        resolution_mode: str | None
        if self.implementation == "cmrbench":
            expected_path = None
            resolution_mode = "native_cmrbench"
        elif self.mask_path is not None:
            expected_path = self.mask_path
            resolution_mode = "explicit_file"
        elif self.mask_root is not None:
            expected_path = precomputed_vista_mask_path(
                self.mask_root,
                ky=ky,
                frames=frames,
                acceleration=acceleration,
                seed=seed,
            )
            resolution_mode = "explicit_root"
        elif self.generator is None:
            expected_path = precomputed_vista_mask_path(
                DEFAULT_VISTA_MASK_ROOT,
                ky=ky,
                frames=frames,
                acceleration=acceleration,
                seed=seed,
            )
            resolution_mode = "default_root"
        else:
            expected_path = None
            resolution_mode = None

        if (
            expected_path is None
            and self.force_acs
            and self.implementation != "cmrbench"
        ):
            raise ValueError("VISTA force_acs is supported only for precomputed masks")
        if self.implementation == "cmrbench":
            if kwargs:
                raise ValueError(
                    "VISTA implementation='cmrbench' does not expose generator tuning "
                    "parameters"
                )
            compact = generate_vista_mask(
                p=ky,
                t=frames,
                acceleration=acceleration,
                seed=seed,
            )
            mask_source = "generated"
            mask_file = None
            generation_parameters = {
                "generator": "cmrbench.vista.generate_vista_mask",
                "generator_parameters": {
                    "p": ky,
                    "t": frames,
                    "acceleration": float(acceleration),
                    "seed": seed,
                    "defaults": "validated Phase 1 defaults",
                },
                "algorithm": "VISTA (Ahmad et al., MRM 2015)",
                "executable_reference": "OSU-CMR/VISTA@1ad7ea0130247aca220a3e761fc8a64b770900b2",
                "python_implementation": "CMRBench adaptation",
            }
        elif expected_path is not None:
            if not expected_path.is_file():
                raise FileNotFoundError(
                    "exact precomputed VISTA mask is missing; "
                    f"resolution_mode={resolution_mode}, requested Ky={ky}, T={frames}, "
                    f"R={acceleration:g}, seed={seed}, "
                    f"resolved root={expected_path.parent}, "
                    f"expected filename={expected_path.name}, resolved path={expected_path}. "
                    "No nearest-mask selection or generated fallback is permitted."
                )
            compact = np.loadtxt(expected_path, dtype=int, delimiter=",")
            mask_source = "precomputed"
            mask_file: str | None = str(expected_path.resolve())
            generation_parameters: dict[str, Any] = {}
        elif self.generator is not None:
            compact, generation_parameters = self._generate_compact(
                ky, frames, acceleration, seed, kwargs
            )
            mask_source = "generated"
            mask_file = None
        else:  # pragma: no cover - construction above guarantees one source
            raise RuntimeError("VISTA has neither a precomputed source nor a generator")

        base_compact = np.asarray(compact).copy()
        base_canonical = canonicalize_vista_mask(
            base_compact,
            slices=slices,
            frames=frames,
            kx=kx,
            ky=ky,
        )
        if self.force_acs:
            assert self.configured_ncalib is not None
            final_compact = add_centered_strip_acs(
                base_compact,
                axis=0,
                width=self.configured_ncalib,
            )
        else:
            final_compact = base_compact.copy()
        canonical = canonicalize_vista_mask(
            final_compact,
            slices=slices,
            frames=frames,
            kx=kx,
            ky=ky,
        )
        base_provenance = binary_mask_provenance(base_canonical)
        final_provenance = binary_mask_provenance(canonical)
        base_actual_acceleration = calculate_actual_acceleration(base_canonical)
        final_actual_acceleration = calculate_actual_acceleration(canonical)
        return SamplingResult(
            mask=canonical,
            requested_acceleration=float(acceleration),
            actual_acceleration=final_actual_acceleration,
            method="VISTA",
            seed=seed,
            parameters={
                "implementation": (
                    "precomputed"
                    if mask_source == "precomputed"
                    else "cmrbench"
                    if self.implementation == "cmrbench"
                    else "injected_generator"
                ),
                "mask_source": mask_source,
                "mask_file": mask_file,
                "resolved_mask_root": (
                    str(expected_path.resolve().parent)
                    if expected_path is not None
                    else None
                ),
                "resolved_mask_filename": (
                    expected_path.name if expected_path is not None else None
                ),
                "resolved_mask_path": mask_file,
                "resolution_mode": resolution_mode,
                "base_mask_path": mask_file,
                "base_mask_filename": (
                    expected_path.name if expected_path is not None else None
                ),
                "Ky": ky,
                "T": frames,
                "native_layout": "[Ky,T]",
                "canonical_layout": "[S,T,Kx,Ky]",
                "compact_mask_shape": list(base_compact.shape),
                "vista_type": self.vista_type,
                "force_acs": self.force_acs,
                "configured_ncalib": self.configured_ncalib,
                "ncalib_affects_precomputed_mask": (
                    self.force_acs if self.implementation == "precomputed" else False
                ),
                "forced_acs_width": (
                    self.configured_ncalib if self.force_acs else None
                ),
                "acs_native_geometry": (
                    "centered_ky_strip" if self.force_acs else None
                ),
                "acs_temporal_coverage": "all_T" if self.force_acs else None,
                "acs_overlay_start_ky": acs_start if self.force_acs else None,
                "acs_overlay_stop_ky_exclusive": (
                    acs_stop if self.force_acs else None
                ),
                "acs_overlay_ky_indices": (
                    list(range(acs_start, acs_stop)) if self.force_acs else []
                ),
                "base_mask_sha256": base_provenance["binary_mask_sha256"],
                "final_mask_sha256": final_provenance["binary_mask_sha256"],
                "base_sampled_count": base_provenance["sampled_points"],
                "final_sampled_count": final_provenance["sampled_points"],
                "added_sampled_count": (
                    final_provenance["sampled_points"]
                    - base_provenance["sampled_points"]
                ),
                "base_actual_acceleration": base_actual_acceleration,
                "final_actual_acceleration": final_actual_acceleration,
                **(
                    {"ncalib": self.configured_ncalib}
                    if self.force_acs
                    else {}
                ),
                **generation_parameters,
                **(
                    {
                        "ncalib_affects_native_mask": False,
                        "ncalib_affects_final_mask": self.force_acs,
                    }
                    if self.implementation == "cmrbench"
                    else {}
                ),
                **final_provenance,
            },
        )

    def _generate_compact(
        self,
        ky: int,
        frames: int,
        acceleration: float,
        seed: int,
        kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        parameters = {**self.parameters, **kwargs}
        n_iter = int(parameters.pop("n_iter", 120))
        alpha = float(parameters.pop("alpha", 0.28))
        grid_interval = int(parameters.pop("grid_interval", math.floor(n_iter / 6)))
        uniform_iteration = int(parameters.pop("uniform_iteration", math.floor(n_iter / 2)))
        step_size = float(parameters.pop("step_size", 0.25))
        fully_sampled_iteration = int(
            parameters.pop("fully_sampled_iteration", math.floor(n_iter * 5 / 6))
        )
        fully_sampled_time_average = int(parameters.pop("fully_sampled_time_average", 1))
        potential_exponent = float(parameters.pop("potential_exponent", 1.4))
        temporal_step = float(parameters.pop("temporal_step", 0.0))
        display_interval = int(parameters.pop("display_interval", 5))
        if parameters:
            raise ValueError(
                "unsupported VISTA parameters: " + ", ".join(sorted(parameters))
            )
        compact = self.generator(
            ky,
            frames,
            acceleration,
            self.vista_type,
            alpha,
            seed,
            n_iter,
            grid_interval,
            uniform_iteration,
            step_size,
            fully_sampled_iteration,
            fully_sampled_time_average,
            potential_exponent,
            temporal_step,
            display_interval,
        )
        provenance = {
            "alpha": alpha,
            "n_iter": n_iter,
            "grid_interval": grid_interval,
            "uniform_iteration": uniform_iteration,
            "step_size": step_size,
            "fully_sampled_iteration": fully_sampled_iteration,
            "fully_sampled_time_average": fully_sampled_time_average,
            "potential_exponent": potential_exponent,
            "temporal_step": temporal_step,
            "display_interval": display_interval,
        }
        return np.asarray(compact), provenance
