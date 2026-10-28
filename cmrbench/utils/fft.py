"""Centered NumPy FFT conventions used by the established cine pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Array = np.ndarray


def _spatial_axes(array: Array) -> tuple[int, int]:
    """Return X/Y axes for ``[..., X, Y, C]`` arrays."""

    if array.ndim < 3:
        raise ValueError(
            "centered 2D FFT expects an array ending in [X,Y,C], such as "
            "[S,T,X,Y,C] or [T,X,Y,C]"
        )
    return array.ndim - 3, array.ndim - 2


def fft2c(image: Array) -> Array:
    """Centered orthonormally scaled FFT over canonical X/Y spatial axes."""

    image = np.asarray(image)
    axes = _spatial_axes(image)
    scale = np.sqrt(float(image.shape[axes[0]] * image.shape[axes[1]]))
    return (
        np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes),
            axes=axes,
        )
        / scale
    )


def ifft2c(kspace: Array) -> Array:
    """Inverse of :func:`fft2c` over canonical Kx/Ky axes."""

    kspace = np.asarray(kspace)
    axes = _spatial_axes(kspace)
    scale = np.sqrt(float(kspace.shape[axes[0]] * kspace.shape[axes[1]]))
    return (
        np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes),
            axes=axes,
        )
        * scale
    )


def legacy_layout_fft2c(image: Array) -> Array:
    """Independent NumPy mirror of the trusted transpose-based legacy FFT2c.

    This exists only for server-side regression. The trusted legacy source is
    not changed or imported, avoiding its TensorFlow/merlintf side effects.
    """

    image = np.asarray(image)
    if image.ndim != 5:
        raise ValueError("legacy full-subject regression expects [S,T,X,Y,C]")
    coil_first = np.transpose(image, axes=(0, 1, 4, 2, 3))
    axes = (3, 4)
    scale = np.sqrt(float(np.prod(coil_first.shape[-2:])))
    kspace = (
        np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(coil_first, axes=axes), axes=axes),
            axes=axes,
        )
        / scale
    )
    return np.transpose(kspace, axes=(0, 1, 3, 4, 2))


def legacy_layout_ifft2c(kspace: Array) -> Array:
    """Independent NumPy mirror of the transpose-based legacy IFFT2c."""

    kspace = np.asarray(kspace)
    if kspace.ndim != 5:
        raise ValueError("legacy full-subject regression expects [S,T,Kx,Ky,C]")
    coil_first = np.transpose(kspace, axes=(0, 1, 4, 2, 3))
    axes = (3, 4)
    scale = np.sqrt(float(np.prod(coil_first.shape[-2:])))
    image = (
        np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(coil_first, axes=axes), axes=axes),
            axes=axes,
        )
        * scale
    )
    return np.transpose(image, axes=(0, 1, 3, 4, 2))


@dataclass(frozen=True)
class DifferenceMetrics:
    max_absolute_difference: float
    mean_absolute_difference: float
    max_relative_difference: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FFTRegressionResult:
    fft: DifferenceMetrics
    ifft: DifferenceMetrics
    rtol: float
    atol: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fft": self.fft.as_dict(),
            "ifft": self.ifft.as_dict(),
            "rtol": self.rtol,
            "atol": self.atol,
            "passed": self.passed,
        }


def difference_metrics(reference: Array, candidate: Array) -> DifferenceMetrics:
    difference = np.abs(np.asarray(candidate) - np.asarray(reference))
    denominator = np.abs(reference)
    meaningful = denominator > np.finfo(np.float64).eps
    if np.any(meaningful):
        max_relative = float(np.max(difference[meaningful] / denominator[meaningful]))
    else:
        max_relative = 0.0 if not np.any(difference) else float("inf")
    return DifferenceMetrics(
        max_absolute_difference=float(np.max(difference, initial=0.0)),
        mean_absolute_difference=float(np.mean(difference)),
        max_relative_difference=max_relative,
    )


def compare_fft_conventions(
    coil_images: Array,
    *,
    legacy_fft: Callable[[Array], Array] = legacy_layout_fft2c,
    legacy_ifft: Callable[[Array], Array] = legacy_layout_ifft2c,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> FFTRegressionResult:
    """Compare trusted-layout and simplified transforms; never hide failure."""

    legacy_kspace = legacy_fft(coil_images)
    simplified_kspace = fft2c(coil_images)
    legacy_image = legacy_ifft(legacy_kspace)
    simplified_image = ifft2c(legacy_kspace)
    fft_metrics = difference_metrics(legacy_kspace, simplified_kspace)
    ifft_metrics = difference_metrics(legacy_image, simplified_image)
    try:
        np.testing.assert_allclose(simplified_kspace, legacy_kspace, rtol=rtol, atol=atol)
        np.testing.assert_allclose(simplified_image, legacy_image, rtol=rtol, atol=atol)
    except AssertionError:
        passed = False
    else:
        passed = True
    return FFTRegressionResult(fft_metrics, ifft_metrics, rtol, atol, passed)

