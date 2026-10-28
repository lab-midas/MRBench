"""MRBench wrappers for user-installed, pinned CardioMM sampling code.

The upstream algorithms are external dependencies and are not redistributed here.
See docs/models.md for source setup and terms. MRBench owns canonical conversion,
validation, provenance and the explicit seed policy below.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
from math import floor
from pathlib import Path
import random
from typing import Any

import numpy as np

from cmrbench.sampling.base import KSpaceShape, SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
    validate_declared_central_acs,
)

UPSTREAM_REPOSITORY = "https://github.com/wangziblake/CardioMM_MMCMR-427K.git"
UPSTREAM_COMMIT = "abb6f8a4e293426736467c9859a5f5755c38c8aa"
UPSTREAM_SOURCE_ROOT = (
    "CardioMM_dataundersampling/Python_Reimpl_Version/"
    "cardiomm_mask_generator"
)
UPSTREAM_DISPATCHER = f"{UPSTREAM_SOURCE_ROOT}/ktMaskGenerator.py"
UPSTREAM_DISPATCHER_SHA256 = (
    "866f2b9a5803aa50ff31d5c2260bc78f664b712b507298752e03d0f0e953a189"
)
UPSTREAM_LICENSE_STATUS = (
    "upstream README academic-use statement; no standard license file found "
    "in the audited commit"
)
SUPPORTED_CARDIOMM_GENERATORS = (
    "uniform",
    "kt_uniform",
    "kt_gaussian",
    "kt_radial",
)
SOURCE_FILES = {
    "uniform": f"{UPSTREAM_SOURCE_ROOT}/UniformSampling.py",
    "kt_uniform": f"{UPSTREAM_SOURCE_ROOT}/ktUniformSampling.py",
    "kt_gaussian": f"{UPSTREAM_SOURCE_ROOT}/ktGaussianSampling.py",
    "kt_radial": f"{UPSTREAM_SOURCE_ROOT}/ktRadialSampling.py",
}
SOURCE_SHA256 = {
    "uniform": "a154f3bbb9a78793c6974ab470da6344c122b7e6bd8986f7f50632079eebf01b",
    "kt_uniform": "c90f4890ba280ba91b91d1e5b8b7dcb19192dfaad95843442186f3aa8417ec6b",
    "kt_gaussian": "7a8a5c5bcced44dc54bee1da1457e38b1e85c480f0712c9dcd461b0bc525ccf0",
    "kt_radial": "9e8aac4b5d9d37716382967ea1cfc851123363d3300bd5e5ef326044f7b1d9a1",
}
SOURCE_TOKENS = {
    "uniform": "Uniform",
    "kt_uniform": "ktUniform",
    "kt_gaussian": "ktGaussian",
    "kt_radial": "ktRadial",
}
TRAINING_MASK_SOURCE = "CardioMM_reconstruction/data/subsample.py"
TRAINING_MASK_SOURCE_SHA256 = (
    "4218b7f6d88a2bc334bc914fc3eb15081ff24e17b1c81e2e8446b402eeb82583"
)


def _upstream(module_name: str, expected_sha256: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "CardioMM sampling requires the separately obtained upstream source "
            "and its dependencies on PYTHONPATH; see docs/models.md#cardiomm-sampling"
        ) from exc
    source = Path(module.__file__).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise RuntimeError(
            f"CardioMM sampling source mismatch for {module_name}; "
            f"use upstream revision {UPSTREAM_COMMIT} (see docs/models.md)"
        )
    return module


def _generator(family: str):
    token = SOURCE_TOKENS[family] + "Sampling"
    module = _upstream("cardiomm_mask_generator." + token, SOURCE_SHA256[family])
    return getattr(module, token)


def cardiomm_uniform_sampling(nx, ny, nt, ncalib, acceleration):
    native = _generator("uniform")(nx, ny, ncalib, acceleration)
    return np.repeat(np.asarray(native)[:, :, None], nt, axis=2)


def cardiomm_kt_uniform_sampling(nx, ny, nt, ncalib, acceleration):
    return _generator("kt_uniform")(nx, ny, nt, ncalib, acceleration)


def cardiomm_kt_gaussian_sampling(nx, ny, nt, ncalib, acceleration, alpha, seed):
    return _generator("kt_gaussian")(nx, ny, nt, ncalib, acceleration, alpha, seed)


def cardiomm_kt_radial_sampling(nx, ny, nt, ncalib, acceleration, angle4next, cropcorner):
    return _generator("kt_radial")(
        nx, ny, nt, ncalib, acceleration, angle4next, cropcorner
    )


def cardiomm_fixed_random_sampling(nx, ny, ncalib, acceleration, seed):
    if ncalib < 1 or ncalib > ny:
        raise ValueError("ncalib must fit Ky")
    module = _upstream(
        "CardioMM_reconstruction.data.subsample", TRAINING_MASK_SOURCE_SHA256
    )
    sampler = module.FixedLowRandomMaskFunc([ncalib], [acceleration], seed=int(seed))
    shape = (nx, ny, 2)
    center = sampler.calculate_center_mask(shape, ncalib)
    outer = sampler.calculate_acceleration_mask(ny, nx, acceleration, 0, ncalib)
    return np.broadcast_to(np.maximum(center, outer)[None, :], (nx, ny)).copy()


def cardiomm_fixed_radial_sampling(nx, ny, ncalib, acceleration, seed):
    if ncalib < 1 or ncalib > min(nx, ny):
        raise ValueError("ncalib must fit Kx and Ky")
    if int(np.floor(180 / (acceleration * 0.6))) < 1:
        raise ValueError("resolved radial acceleration produces no beams")
    module = _upstream(
        "CardioMM_reconstruction.data.subsample", TRAINING_MASK_SOURCE_SHA256
    )
    sampler = module.FixedLowRadialMaskFunc([ncalib], [acceleration], seed=int(seed))
    center = sampler.calculate_2Dcenter_mask((nx, ny, 2), ncalib)
    state = random.getstate()
    try:
        random.seed(int(seed))
        outer = sampler.calculate_acceleration_mask(ny, nx, acceleration, 0, ncalib)
    finally:
        random.setstate(state)
    return np.maximum(center, outer) > 0


class CardioMMFixedTrainingSamplingProtocol(SamplingProtocol):
    """CardioMM training-side fixed-ACS random or pseudo-radial generator."""

    def __init__(self, *, family: str, ncalib: int = 20) -> None:
        if family not in {"random", "radial"}:
            raise ValueError("CardioMM fixed training family must be random or radial")
        if ncalib < 1:
            raise ValueError("ncalib must be positive")
        self.family = family
        self.ncalib = int(ncalib)

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if kwargs:
            raise ValueError(
                "unsupported CardioMM fixed-training parameters: "
                + ", ".join(sorted(kwargs))
            )
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if seed is None:
            raise ValueError(f"CardioMM {self.family} requires an explicit recorded seed")
        if not float(acceleration).is_integer() or float(acceleration) < 1:
            raise ValueError("CardioMM fixed-training acceleration must be a positive integer")
        slices, frames, kx, ky, _coils = (int(value) for value in shape)
        nominal = int(float(acceleration))
        if self.family == "random":
            native = cardiomm_fixed_random_sampling(
                kx, ky, self.ncalib, nominal, int(seed)
            )
            geometry = "centered_ky_strip"
            generator = "FixedLowRandomMaskFunc"
            rng = "numpy.random.RandomState(seed)"
            resolved = {"outer_probability": 1.0 / nominal}
        else:
            native = cardiomm_fixed_radial_sampling(
                kx, ky, self.ncalib, nominal, int(seed)
            )
            geometry = "centered_square"
            generator = "FixedLowRadialMaskFunc"
            rng = "Python random.Random(seed); source uses unseeded global random"
            resolved = {
                "radial_acceleration_scale": 0.6,
                "angle4next": 137.5,
                "python_rng_explicitly_seeded_by_cmrbench": True,
            }
        canonical = np.broadcast_to(
            np.asarray(native, dtype=bool)[None, None],
            (slices, frames, kx, ky),
        ).copy()
        acs_contract = validate_declared_central_acs(
            canonical,
            ncalib=self.ncalib,
            geometry=geometry,
            source=f"CardioMM {self.family}",
        )
        measured = calculate_actual_acceleration(canonical)
        if measured is None:
            raise RuntimeError("generated CardioMM fixed training mask has no support")
        return SamplingResult(
            mask=canonical,
            requested_acceleration=float(acceleration),
            actual_acceleration=measured,
            method=self.family,
            seed=int(seed),
            parameters={
                "mask_source": "generated",
                "family": self.family,
                "implementation": "cardiomm",
                "generator": f"CardioMM::{generator}",
                "generator_parameters": {
                    "ncalib": self.ncalib,
                    "acceleration": nominal,
                    "seed": int(seed),
                    **resolved,
                },
                "ncalib": self.ncalib,
                "acs_geometry": geometry,
                **acs_contract,
                "temporal_behavior": "static_single_source_call",
                "native_layout": "[Kx,Ky]",
                "native_shape": [kx, ky],
                "canonical_layout": "[S,T,Kx,Ky]",
                "canonical_shape": list(canonical.shape),
                "slice_broadcast": True,
                "temporal_broadcast": True,
                "deterministic": True,
                "rng": rng,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_source": TRAINING_MASK_SOURCE,
                "upstream_source_sha256": TRAINING_MASK_SOURCE_SHA256,
                "upstream_license": UPSTREAM_LICENSE_STATUS,
                "sampling_capability_is_model_capability": False,
                **binary_mask_provenance(canonical),
            },
        )


class CardioMMSamplingProtocol(SamplingProtocol):
    """Canonical wrapper around the pinned CardioMM k-t generators."""

    def __init__(
        self,
        *,
        family: str,
        ncalib: int = 20,
        alpha: float = 0.2,
        angle4next: float = 137.5,
        cropcorner: bool = True,
        radial_acceleration_scale: float = 0.6,
    ) -> None:
        if family not in SUPPORTED_CARDIOMM_GENERATORS:
            raise ValueError(
                f"unknown CardioMM generator {family!r}; expected "
                f"{SUPPORTED_CARDIOMM_GENERATORS}"
            )
        if ncalib < 1:
            raise ValueError("ncalib must be positive")
        self.family = family
        self.ncalib = int(ncalib)
        self.alpha = float(alpha)
        self.angle4next = float(angle4next)
        self.cropcorner = bool(cropcorner)
        self.radial_acceleration_scale = float(radial_acceleration_scale)

    def generate(
        self,
        shape: KSpaceShape,
        acceleration: float,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if kwargs:
            raise ValueError(
                "unsupported CardioMM sampling parameters: "
                + ", ".join(sorted(kwargs))
            )
        if len(shape) != 5:
            raise ValueError("sampling shape must be canonical [S,T,Kx,Ky,C]")
        if acceleration < 1 or not float(acceleration).is_integer():
            raise ValueError("CardioMM generators require an integer nominal acceleration")
        slices, frames, kx, ky, _coils = (int(item) for item in shape)
        if self.ncalib > ky or (self.family == "kt_radial" and self.ncalib > kx):
            geometry = "Kx and Ky" if self.family == "kt_radial" else "Ky"
            raise ValueError(
                f"ncalib={self.ncalib} does not fit {self.family} {geometry} geometry"
            )
        integer_acceleration = int(acceleration)
        resolved: dict[str, Any] = {
            "nx": kx,
            "ny": ky,
            "nt": frames,
            "ncalib": self.ncalib,
        }
        deterministic = self.family != "kt_gaussian"
        if self.family == "uniform":
            native = cardiomm_uniform_sampling(
                kx, ky, frames, self.ncalib, integer_acceleration
            )
            resolved["R"] = integer_acceleration
        elif self.family == "kt_uniform":
            native = cardiomm_kt_uniform_sampling(
                kx, ky, frames, self.ncalib, integer_acceleration
            )
            resolved["R"] = integer_acceleration
        elif self.family == "kt_gaussian":
            if seed is None:
                raise ValueError("CardioMM kt_gaussian requires an explicit recorded seed")
            native = cardiomm_kt_gaussian_sampling(
                kx,
                ky,
                frames,
                self.ncalib,
                integer_acceleration,
                self.alpha,
                seed,
            )
            resolved.update(
                {"R": integer_acceleration, "alpha": self.alpha, "seed": seed}
            )
        else:
            low_level_acceleration = acceleration * self.radial_acceleration_scale
            native = cardiomm_kt_radial_sampling(
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
                    "rasterizer": "skimage.transform.rotate(order=0, mode=constant)",
                    "scikit_image_version": metadata.version("scikit-image"),
                }
            )

        native_binary = np.asarray(native) > 0
        if native_binary.ndim == 2 and frames == 1:
            native_binary = native_binary[:, :, np.newaxis]
        expected_native = (kx, ky, frames)
        if native_binary.shape != expected_native:
            raise RuntimeError(
                f"upstream {self.family} returned {native_binary.shape}, "
                f"expected {expected_native}"
            )
        canonical = np.broadcast_to(
            np.transpose(native_binary, (2, 0, 1))[None],
            (slices, frames, kx, ky),
        ).copy()
        declared_geometry = (
            "centered_square" if self.family == "kt_radial" else "centered_ky_strip"
        )
        acs_contract = validate_declared_central_acs(
            canonical,
            ncalib=self.ncalib,
            geometry=declared_geometry,
            source=f"CardioMM {self.family}",
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
                "implementation": "cardiomm",
                "generator": f"CardioMM::{SOURCE_TOKENS[self.family]}",
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
                "upstream_source": SOURCE_FILES[self.family],
                "upstream_source_sha256": SOURCE_SHA256[self.family],
                "upstream_dispatcher": UPSTREAM_DISPATCHER,
                "upstream_dispatcher_sha256": UPSTREAM_DISPATCHER_SHA256,
                "upstream_license": UPSTREAM_LICENSE_STATUS,
                "cmrbench_wrapper_policy": (
                    "external pinned upstream functions; explicit Gaussian seed; "
                    "canonical transpose and slice broadcast"
                ),
                "sampling_capability_is_model_capability": False,
                **binary_mask_provenance(canonical),
            },
        )


def CardioMMKtUniformSamplingProtocol(**parameters: Any) -> CardioMMSamplingProtocol:
    return CardioMMSamplingProtocol(family="kt_uniform", **parameters)


def CardioMMUniformSamplingProtocol(**parameters: Any) -> CardioMMSamplingProtocol:
    return CardioMMSamplingProtocol(family="uniform", **parameters)


def CardioMMKtGaussianSamplingProtocol(**parameters: Any) -> CardioMMSamplingProtocol:
    return CardioMMSamplingProtocol(family="kt_gaussian", **parameters)


def CardioMMKtRadialSamplingProtocol(**parameters: Any) -> CardioMMSamplingProtocol:
    return CardioMMSamplingProtocol(family="kt_radial", **parameters)


def CardioMMRandomSamplingProtocol(
    **parameters: Any,
) -> CardioMMFixedTrainingSamplingProtocol:
    return CardioMMFixedTrainingSamplingProtocol(family="random", **parameters)


def CardioMMRadialSamplingProtocol(
    **parameters: Any,
) -> CardioMMFixedTrainingSamplingProtocol:
    return CardioMMFixedTrainingSamplingProtocol(family="radial", **parameters)
