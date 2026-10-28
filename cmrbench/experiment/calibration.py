"""Model-independent requested ACS width from physical sampling policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from cmrbench.data.sensitivity import SensitivityMapConfig
from cmrbench.sampling.result import SamplingResult


@dataclass(frozen=True)
class CalibrationResolution:
    calibration_source: str
    requested_calibration_width: int
    runtime_ky: int
    requested_ncalib: int | None = None
    requested_center_fraction: float | None = None

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


def resolve_requested_calibration_width(
    sampling: Any,
    runtime_geometry: tuple[int, ...],
) -> CalibrationResolution | None:
    """Resolve physical requested ACS, never the measured contiguous extent.

    Accepts either a configured SamplingSection or its realized SamplingResult.
    Both have ``method`` and ``parameters``; a realized center-fraction sampler
    also carries its computed ncalib, but the center fraction remains its source.
    """

    if len(runtime_geometry) != 5:
        raise ValueError("calibration geometry must be canonical [S,T,Kx,Ky,C]")
    ky = int(runtime_geometry[3])
    if ky < 1:
        raise ValueError("calibration requires positive runtime Ky")
    parameters = dict(sampling.parameters)
    generator = parameters.get("generator_parameters")
    generated = dict(generator) if isinstance(generator, Mapping) else {}
    center_fraction = parameters.get("center_fraction")
    ncalib = parameters.get("ncalib")
    if center_fraction is None and (
        parameters.get("acs_parameter_semantics") == "center_fraction (source-faithful)"
        or (str(sampling.method) in {"random", "equispaced"} and "center_fraction" in generated)
    ):
        center_fraction = generated.get("center_fraction")
    if center_fraction is not None:
        if "center_fraction" in parameters and ncalib is not None:
            raise ValueError("sampling policy must not specify both ncalib and center_fraction")
        fraction = float(center_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("sampling center_fraction must be in (0, 1)")
        width = int(round(ky * fraction))
        if width < 1:
            raise ValueError("sampling center_fraction rounds to zero ACS lines")
        if ncalib is not None and int(ncalib) != width:
            raise ValueError(
                f"realized sampler ncalib={ncalib} disagrees with "
                f"round(runtime Ky={ky} * center_fraction={fraction})={width}"
            )
        return CalibrationResolution(
            "center_fraction", width, ky, requested_center_fraction=fraction
        )
    if ncalib is None:
        return None
    if isinstance(ncalib, bool) or not isinstance(ncalib, int) or ncalib < 1:
        raise ValueError("sampling ncalib must be a positive integer")
    if ncalib > ky:
        raise ValueError(f"sampling ncalib={ncalib} exceeds runtime Ky={ky}")
    return CalibrationResolution("ncalib", int(ncalib), ky, requested_ncalib=int(ncalib))


def resolve_input_calibration(
    sampling: SamplingResult,
    runtime_geometry: tuple[int, ...],
    *,
    configured_sampling: Any | None = None,
) -> CalibrationResolution | None:
    """Prefer configured physical policy and verify realized provenance agrees."""

    realized = resolve_requested_calibration_width(sampling, runtime_geometry)
    if configured_sampling is None:
        return realized
    configured = resolve_requested_calibration_width(configured_sampling, runtime_geometry)
    if configured is not None and realized is not None and (
        configured.calibration_source != realized.calibration_source
        or configured.requested_calibration_width != realized.requested_calibration_width
    ):
        raise ValueError(
            "sampling configuration and realized sampling provenance disagree on "
            "requested calibration width"
        )
    return configured or realized


def resolve_espirit_calibration_config(
    config: SensitivityMapConfig | Mapping[str, Any] | str,
    calibration: CalibrationResolution | None,
) -> SensitivityMapConfig:
    """Honor matching legacy widths; never infer width from realized ACS."""

    parsed = SensitivityMapConfig.from_mapping(config)
    if parsed.strategy != "espirit":
        return parsed
    if calibration is None:
        raise ValueError(
            "ESPIRiT requires sampling.parameters.ncalib or center_fraction "
            "as its calibration source of truth"
        )
    width = calibration.requested_calibration_width
    if parsed.calib_width is not None and int(parsed.calib_width) != width:
        raise ValueError(
            f"explicit ESPIRiT calib_width={parsed.calib_width} conflicts with "
            f"sampling-resolved requested calibration width={width} "
            f"from {calibration.calibration_source}"
        )
    return SensitivityMapConfig(
        **{**asdict(parsed), "calib_width": width}
    )


def calibration_provenance(
    sampling: SamplingResult,
    runtime_geometry: tuple[int, ...],
    *,
    configured_sampling: Any | None = None,
) -> dict[str, Any]:
    request = resolve_input_calibration(
        sampling, runtime_geometry, configured_sampling=configured_sampling
    )
    measured = sampling.physical_mask_validation
    geometry = str(sampling.parameters.get("acs_geometry", "centered_ky_strip"))
    width_key = (
        "detected_centered_square_extent"
        if geometry == "centered_square"
        else "detected_centered_ky_strip_extent"
    )
    return {
        "calibration_source": request.calibration_source if request else None,
        "requested_calibration_width": (
            request.requested_calibration_width if request else None
        ),
        "requested_ncalib": request.requested_ncalib if request else None,
        "requested_center_fraction": (
            request.requested_center_fraction if request else None
        ),
        "runtime_ky": int(runtime_geometry[3]),
        "realized_acs_width": int(measured[width_key]),
        "realized_acs_geometry": geometry,
        "reconstruction_espirit_calib_width": None,
        "evaluation_espirit_calib_width": None,
        "cardiomm_num_low_frequencies": None,
    }
