"""Validated CMRBench-to-DeepInv representation bridge for complex MRI."""

from __future__ import annotations

import numpy as np


def complex_image_to_deepinv(image: np.ndarray) -> np.ndarray:
    """Convert canonical complex ``[X,Y]`` to float32 ``[1,2,X,Y]``."""

    value = np.asarray(image)
    if value.ndim != 2 or not np.iscomplexobj(value):
        raise ValueError("image must be a complex [X,Y] array")
    return np.stack((value.real, value.imag), axis=0)[None].astype(
        np.float32, copy=False
    )


def deepinv_image_to_complex(image: np.ndarray) -> np.ndarray:
    """Convert one DeepInv image ``[1,2,X,Y]`` to complex ``[X,Y]``."""

    value = np.asarray(image)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 2:
        raise ValueError("DeepInv image must have [1,2,X,Y] shape")
    return (value[0, 0] + 1j * value[0, 1]).astype(np.complex64, copy=False)


def complex_kspace_to_deepinv(kspace: np.ndarray) -> np.ndarray:
    """Convert complex coil-last ``[Kx,Ky,C]`` to ``[1,2,C,Kx,Ky]``."""

    value = np.asarray(kspace)
    if value.ndim != 3 or not np.iscomplexobj(value):
        raise ValueError("k-space must be a complex [Kx,Ky,C] array")
    coil_first = np.moveaxis(value, -1, 0)
    return np.stack((coil_first.real, coil_first.imag), axis=0)[None].astype(
        np.float32, copy=False
    )


def deepinv_kspace_to_complex(kspace: np.ndarray) -> np.ndarray:
    """Convert one DeepInv measurement to complex coil-last ``[Kx,Ky,C]``."""

    value = np.asarray(kspace)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 2:
        raise ValueError("DeepInv k-space must have [1,2,C,Kx,Ky] shape")
    coil_first = value[0, 0] + 1j * value[0, 1]
    return np.moveaxis(coil_first, 0, -1).astype(np.complex64, copy=False)


def mask_to_deepinv(mask: np.ndarray) -> np.ndarray:
    """Convert canonical binary ``[Kx,Ky]`` to float32 ``[1,2,Kx,Ky]``."""

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("mask must have [Kx,Ky] shape")
    binary = value.astype(bool, copy=False)
    return np.broadcast_to(binary[None, None], (1, 2, *binary.shape)).astype(
        np.float32, copy=True
    )


def deepinv_mask_to_canonical(mask: np.ndarray) -> np.ndarray:
    """Validate equal complex mask channels and return canonical ``[Kx,Ky]``."""

    value = np.asarray(mask)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 2:
        raise ValueError("DeepInv mask must have [1,2,Kx,Ky] shape")
    if not np.array_equal(value[0, 0], value[0, 1]):
        raise ValueError("DeepInv real/imag mask channels differ")
    return value[0, 0].astype(bool, copy=False)


def sensitivity_maps_to_deepinv(maps: np.ndarray) -> np.ndarray:
    """Convert complex coil-last maps ``[X,Y,C]`` to ``[1,C,X,Y]``."""

    value = np.asarray(maps)
    if value.ndim != 3 or not np.iscomplexobj(value):
        raise ValueError("sensitivity maps must be a complex [X,Y,C] array")
    return np.moveaxis(value, -1, 0)[None].astype(np.complex64, copy=False)


def deepinv_sensitivity_maps_to_canonical(maps: np.ndarray) -> np.ndarray:
    """Convert one DeepInv map batch ``[1,C,X,Y]`` to ``[X,Y,C]``."""

    value = np.asarray(maps)
    if value.ndim != 4 or value.shape[0] != 1 or not np.iscomplexobj(value):
        raise ValueError("DeepInv sensitivity maps must be complex [1,C,X,Y]")
    return np.moveaxis(value[0], 0, -1).astype(np.complex64, copy=False)


__all__ = [
    "complex_image_to_deepinv",
    "complex_kspace_to_deepinv",
    "deepinv_image_to_complex",
    "deepinv_kspace_to_complex",
    "deepinv_mask_to_canonical",
    "deepinv_sensitivity_maps_to_canonical",
    "mask_to_deepinv",
    "sensitivity_maps_to_deepinv",
]
