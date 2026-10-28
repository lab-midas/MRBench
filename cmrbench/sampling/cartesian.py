"""CMRBench-native random Cartesian line sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from cmrbench.sampling.base import KSpaceShape, SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
)


class RandomCartesianSampling(SamplingProtocol):
    """Sample full Cartesian lines along an explicitly selected spatial axis.

    This independently implemented sampler first chooses one set of 1-D line
    indices for each temporal frame (or one shared set when ``time_varying`` is
    false), then writes full orthogonal readout lines directly into canonical
    ``[S,T,Kx,Ky]`` storage.  Repository history contains no evidence that this
    code was ported or adapted from fastMRI.  In particular, it has no ACS or
    center-fraction rule and does not use fastMRI's outer-line Bernoulli scheme.
    """

    def __init__(self, *, phase_encoding_axis: int = 3, time_varying: bool = True):
        if phase_encoding_axis not in (2, 3):
            raise ValueError("phase_encoding_axis must be 2 (Kx) or 3 (Ky)")
        self.phase_encoding_axis = phase_encoding_axis
        self.time_varying = time_varying

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if kwargs:
            raise ValueError(
                "unsupported random_cartesian parameters: "
                + ", ".join(sorted(kwargs))
            )
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if acceleration < 1:
            raise ValueError("acceleration must be >= 1")

        slices, frames, kx, ky, _coils = shape
        spatial_shape = (kx, ky)
        phase_index = self.phase_encoding_axis - 2
        phase_size = spatial_shape[phase_index]
        lines_per_frame = max(1, min(phase_size, round(phase_size / acceleration)))
        generator = np.random.default_rng(seed)
        mask = np.zeros((slices, frames, kx, ky), dtype=bool)

        shared_lines = None
        if not self.time_varying:
            shared_lines = generator.choice(phase_size, size=lines_per_frame, replace=False)

        for frame in range(frames):
            lines = shared_lines
            if lines is None:
                lines = generator.choice(phase_size, size=lines_per_frame, replace=False)
            if self.phase_encoding_axis == 2:
                mask[:, frame, lines, :] = True
            else:
                mask[:, frame, :, lines] = True

        return SamplingResult(
            mask=mask,
            requested_acceleration=float(acceleration),
            actual_acceleration=calculate_actual_acceleration(mask),
            method="random_cartesian",
            seed=seed,
            parameters={
                "mask_source": "generated",
                "family": "random_cartesian",
                "implementation": "cmrbench_native",
                "generator": "CMRBench::RandomCartesianSampling",
                "origin": "independent CMRBench implementation",
                "origin_audit": {
                    "first_repository_commit": (
                        "18600427341155dc58c15451aa9ec5a56077d3d4"
                    ),
                    "fastmri_derived": False,
                    "basis": (
                        "creation history, source comments/docstrings, and separate "
                        "fastMRI helper scripts contain no port/adaptation evidence"
                    ),
                },
                "phase_encoding_axis": self.phase_encoding_axis,
                "sampled_axis": "Kx" if self.phase_encoding_axis == 2 else "Ky",
                "fully_sampled_orthogonal_axis": (
                    "Ky" if self.phase_encoding_axis == 2 else "Kx"
                ),
                "time_varying": self.time_varying,
                "lines_per_frame": lines_per_frame,
                "native_selection": "1-D selected line-index set per temporal frame",
                "native_selection_shape": [lines_per_frame],
                "native_mask_array_materialized": False,
                "canonical_layout": "[S,T,Kx,Ky]",
                "canonical_shape": list(mask.shape),
                "slice_broadcast": True,
                "independent_over_time": self.time_varying,
                "acs": None,
                **binary_mask_provenance(mask),
            },
        )
