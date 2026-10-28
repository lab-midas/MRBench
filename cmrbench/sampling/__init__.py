"""Retrospective sampling protocols and provenance."""

from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.cardiomm import (
    CardioMMKtGaussianSamplingProtocol,
    CardioMMKtRadialSamplingProtocol,
    CardioMMKtUniformSamplingProtocol,
    CardioMMSamplingProtocol,
    CardioMMUniformSamplingProtocol,
    CardioMMFixedTrainingSamplingProtocol,
    CardioMMRandomSamplingProtocol,
    CardioMMRadialSamplingProtocol,
)
from cmrbench.sampling.external import (
    CANONICAL_MASK_LAYOUT,
    EXPLICIT_COMPACT_LAYOUTS,
    ExternalMaskSamplingProtocol,
    canonicalize_external_mask,
)
from cmrbench.sampling.nvidia import (
    CanonicalUniformSamplingProtocol,
    KtGaussianSamplingProtocol,
    KtRadialSamplingProtocol,
    KtUniformSamplingProtocol,
    NVIDIASamplingProtocol,
    UniformSamplingProtocol,
    NVIDIAStaticLineSamplingProtocol,
    RandomSamplingProtocol,
    EquispacedSamplingProtocol,
)
from cmrbench.sampling.registry import SamplingRegistry, default_sampling_registry
from cmrbench.sampling.result import SamplingResult, calculate_actual_acceleration
from cmrbench.sampling.vista import (
    VISTASamplingProtocol,
    canonicalize_vista_mask,
    precomputed_vista_mask_path,
)

__all__ = [
    "CanonicalUniformSamplingProtocol",
    "CardioMMKtGaussianSamplingProtocol",
    "CardioMMKtRadialSamplingProtocol",
    "CardioMMKtUniformSamplingProtocol",
    "CardioMMSamplingProtocol",
    "CardioMMUniformSamplingProtocol",
    "CardioMMFixedTrainingSamplingProtocol",
    "CardioMMRandomSamplingProtocol",
    "CardioMMRadialSamplingProtocol",
    "CANONICAL_MASK_LAYOUT",
    "EXPLICIT_COMPACT_LAYOUTS",
    "ExternalMaskSamplingProtocol",
    "KtGaussianSamplingProtocol",
    "KtRadialSamplingProtocol",
    "KtUniformSamplingProtocol",
    "NVIDIASamplingProtocol",
    "NVIDIAStaticLineSamplingProtocol",
    "RandomSamplingProtocol",
    "EquispacedSamplingProtocol",
    "SamplingProtocol",
    "SamplingRegistry",
    "SamplingResult",
    "UniformSamplingProtocol",
    "VISTASamplingProtocol",
    "canonicalize_external_mask",
    "canonicalize_vista_mask",
    "calculate_actual_acceleration",
    "default_sampling_registry",
    "precomputed_vista_mask_path",
]
