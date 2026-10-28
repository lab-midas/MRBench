"""The explicit dataset/sampling-to-model boundary."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np

from cmrbench.data.sample import CineMRISample
from cmrbench.experiment.calibration import (
    calibration_provenance,
    resolve_espirit_calibration_config,
    resolve_input_calibration,
)
from cmrbench.data.sensitivity import (
    EspiritEstimator,
    SensitivityMapConfig,
    prepare_sensitivity_maps,
)
from cmrbench.models.input import ReconstructionInput
from cmrbench.sampling.base import SamplingProtocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
)

SamplingMode = Literal["retrospective", "acquired"]


def prepare_reconstruction_input(
    sample: CineMRISample,
    *,
    mode: SamplingMode,
    protocol: SamplingProtocol | None = None,
    acceleration: float | None = None,
    seed: int | None = None,
) -> ReconstructionInput:
    """Prepare actual model measurements without model-specific preprocessing."""

    if mode == "retrospective":
        if sample.mask is not None:
            raise ValueError(
                "retrospective mode expects complete starting k-space with "
                "sample.mask=None; refusing to undersample acquired data again"
            )
        if protocol is None:
            raise ValueError("retrospective mode requires a sampling protocol")
        if acceleration is None:
            raise ValueError("retrospective mode requires requested acceleration")
        sampling = protocol.generate(tuple(sample.kspace.shape), acceleration, seed=seed)
        mask = sampling.mask
        masked_kspace = sample.kspace * _expand_mask_for_coils(mask, sample.kspace)
        metadata = dict(sample.metadata)
        metadata["cmrbench_calibration"] = calibration_provenance(
            sampling, tuple(sample.kspace.shape)
        )
        return ReconstructionInput(
            subject_id=sample.subject_id,
            kspace=masked_kspace,
            mask=mask,
            sensitivity_maps=sample.sensitivity_maps,
            metadata=metadata,
            sampling=sampling,
        )

    if mode == "acquired":
        if protocol is not None:
            raise ValueError("acquired mode bypasses SamplingProtocol; protocol must be None")
        if sample.mask is None:
            raise ValueError(
                "acquired mode requires CineMRISample.mask containing the acquisition mask"
            )
        sampling = SamplingResult(
            mask=sample.mask,
            requested_acceleration=None,
            actual_acceleration=calculate_actual_acceleration(sample.mask),
            method="acquired",
            seed=None,
            parameters={
                "implementation": "acquired",
                **binary_mask_provenance(sample.mask),
            },
        )
        metadata = dict(sample.metadata)
        metadata["cmrbench_calibration"] = calibration_provenance(
            sampling, tuple(sample.kspace.shape)
        )
        return ReconstructionInput(
            subject_id=sample.subject_id,
            kspace=sample.kspace,
            mask=sample.mask,
            sensitivity_maps=sample.sensitivity_maps,
            metadata=metadata,
            sampling=sampling,
        )

    raise ValueError(f"unknown sampling mode {mode!r}; expected retrospective or acquired")


def _expand_mask_for_coils(mask: object, kspace: object) -> object:
    """Append only a broadcast dimension, converting NumPy masks for torch if needed."""

    expanded = np.asarray(mask)[..., None]
    if type(kspace).__module__.split(".")[0] == "torch" and hasattr(kspace, "new_tensor"):
        return kspace.new_tensor(expanded)
    return expanded


def attach_prepared_sensitivity_maps(
    input: ReconstructionInput,
    *,
    config: SensitivityMapConfig | dict[str, object] | str,
    prewhitened: bool | None = None,
    estimator: EspiritEstimator | None = None,
    backend_version: str | None = None,
) -> ReconstructionInput:
    """Attach dataset-level maps after physical sampling has been realized.

    This is a separate opt-in preparation step so the generic reconstruction
    entry, sampling protocols, and model-adapter contracts remain unchanged.
    ESPIRiT therefore receives ``input.kspace`` and ``input.mask`` only after
    the existing retrospective/acquired preparation path has completed.
    """

    prepared = prepare_sensitivity_maps(
        kspace=np.asarray(input.kspace),
        mask=np.asarray(input.mask),
        provided_maps=(
            None
            if input.sensitivity_maps is None
            else np.asarray(input.sensitivity_maps)
        ),
        config=config,
        prewhitened=prewhitened,
        requested_center_fraction=_sampling_parameter(
            input, "center_fraction"
        ),
        requested_ncalib=_sampling_parameter(input, "ncalib"),
        estimator=estimator,
        backend_version=backend_version,
    )
    metadata = dict(input.metadata)
    metadata["cmrbench_sensitivity_maps"] = dict(prepared.provenance)
    return ReconstructionInput(
        subject_id=input.subject_id,
        kspace=input.kspace,
        mask=input.mask,
        sensitivity_maps=prepared.maps,
        metadata=metadata,
        sampling=input.sampling,
    )


def attach_dataset_sensitivity_maps(
    input: ReconstructionInput,
    parameters: Mapping[str, Any],
    *,
    configured_sampling: Any | None = None,
) -> ReconstructionInput:
    """Apply an explicitly configured dataset-level map strategy, if present."""

    if "sensitivity_maps" not in parameters:
        return input
    prewhitened = parameters.get("prewhitened")
    if prewhitened is not None and not isinstance(prewhitened, bool):
        raise ValueError("dataset.parameters.prewhitened must be Boolean or null")
    calibration = resolve_input_calibration(
        input.sampling, tuple(input.kspace.shape),
        configured_sampling=configured_sampling,
    ) if input.sampling is not None else None
    maps_config = resolve_espirit_calibration_config(
        parameters["sensitivity_maps"], calibration
    )
    prepared = attach_prepared_sensitivity_maps(
        input,
        config=maps_config,
        prewhitened=prewhitened,
    )
    if maps_config.strategy == "espirit":
        record = dict(prepared.metadata.get("cmrbench_calibration") or {})
        record["reconstruction_espirit_calib_width"] = maps_config.calib_width
        prepared.metadata["cmrbench_calibration"] = record
    return prepared


def _sampling_parameter(input: ReconstructionInput, name: str) -> object | None:
    if input.sampling is None:
        return None
    parameters = input.sampling.parameters
    if name in parameters:
        return parameters[name]
    generator = parameters.get("generator_parameters")
    if isinstance(generator, dict):
        if name in generator:
            return generator[name]
        if name == "ncalib" and "resolved_num_low_frequencies" in generator:
            return generator["resolved_num_low_frequencies"]
    return None
