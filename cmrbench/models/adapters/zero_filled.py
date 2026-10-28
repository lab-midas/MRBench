"""Established zero-filled cine reconstruction sanity baseline."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord
from cmrbench.utils.fft import ifft2c


class ZeroFilledAdapter(ReconstructionModel):
    """Inverse FFT followed by the established sensitivity-weighted sum."""

    def load(self) -> None:
        return None

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        sensitivity_maps = input.sensitivity_maps
        if sensitivity_maps is None:
            raise ValueError("ZeroFilledAdapter requires sensitivity_maps")
        sensitivity_maps = np.asarray(sensitivity_maps)
        kspace_shape = tuple(int(value) for value in input.kspace.shape)
        if sensitivity_maps.ndim != 5:
            raise ValueError("sensitivity_maps must have [S,1,X,Y,C] or [S,T,X,Y,C]")
        if sensitivity_maps.shape[0] != kspace_shape[0]:
            raise ValueError("sensitivity-map slice dimension does not match k-space")
        if sensitivity_maps.shape[1] not in (1, kspace_shape[1]):
            raise ValueError("sensitivity-map time dimension must be 1 or match T")
        if sensitivity_maps.shape[2:] != kspace_shape[2:]:
            raise ValueError("sensitivity-map X/Y/coil dimensions do not match k-space")

        start = perf_counter()
        coil_images = ifft2c(np.asarray(input.kspace))
        reconstruction = np.sum(
            coil_images * np.conj(sensitivity_maps),
            axis=-1,
        )
        runtime = perf_counter() - start
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=reconstruction,
            model_name="zero_filled",
            runtime_seconds=runtime,
            transforms=[
                TransformRecord(
                    "centered_inverse_fft",
                    {"axes": [2, 3], "scaling": "orthonormal"},
                ),
                TransformRecord(
                    "sensitivity_weighted_coil_sum",
                    {"conjugate_maps": True, "denominator": None},
                ),
            ],
            metadata={"baseline": True, "normalization": None},
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="zero_filled",
            model_version="0.1",
            device="cpu",
            precision="input_preserved",
            metadata={
                "scientific_model": False,
                "coil_combination": "sum(coil_images * conj(sensitivity_maps))",
            },
        )
