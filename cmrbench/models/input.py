"""Canonical measurements presented to every model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cmrbench.data.sample import Tensor, _shape
from cmrbench.sampling.result import SamplingResult, validate_physical_mask


@dataclass
class ReconstructionInput:
    """Actual measured input in ``[S,T,Kx,Ky,C]`` plus explicit mask."""

    subject_id: str
    kspace: Tensor
    mask: Tensor
    sensitivity_maps: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sampling: SamplingResult | None = None

    def __post_init__(self) -> None:
        kspace_shape = _shape(self.kspace, "kspace")
        mask_shape = _shape(self.mask, "mask")
        if len(kspace_shape) != 5:
            raise ValueError(
                "ReconstructionInput.kspace must have canonical [S,T,Kx,Ky,C] shape"
            )
        if len(mask_shape) != 4:
            raise ValueError(
                "ReconstructionInput.mask must have [S,T,Kx,Ky] shape without coils"
            )
        if mask_shape != kspace_shape[:4]:
            raise ValueError(
                f"input mask shape {mask_shape} does not match kspace {kspace_shape[:4]}"
            )
        binary = validate_physical_mask(self.mask)
        if self.sampling is not None and not np.array_equal(
            binary, np.asarray(self.sampling.mask).astype(bool, copy=False)
        ):
            raise ValueError("ReconstructionInput.mask differs from SamplingResult.mask")
