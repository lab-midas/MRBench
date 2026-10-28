"""Evaluate one completed generic CMRBench reconstruction output directory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable
import warnings

import numpy as np

from cmrbench.data.base import CineMRIDataset
from cmrbench.data.sensitivity import EspiritEstimator
from cmrbench.evaluation.anatomical_cine import array_sha256
from cmrbench.evaluation.reconstruction_gif import (
    GIF_PROTOCOL,
    save_mid_slice_reconstruction_gif,
)
from cmrbench.evaluation.reconstruction_metrics import (
    METRIC_PROTOCOL_V2,
    compute_cine_frame_metrics,
    metric_protocol_provenance,
    summarize_subject_metrics,
)
from cmrbench.evaluation.reconstruction_qc import save_anatomy_qc, save_fidelity_qc
from cmrbench.experiment.config import (
    DEFAULT_QC_PHASE_INDICES,
    DEFAULT_QC_POLICY,
    ExperimentConfig,
)
from cmrbench.experiment.calibration import (
    resolve_espirit_calibration_config,
    resolve_input_calibration,
)
from cmrbench.experiment.prepare import (
    attach_dataset_sensitivity_maps,
    attach_prepared_sensitivity_maps,
    prepare_reconstruction_input,
)
from cmrbench.models.input import ReconstructionInput
from cmrbench.reconstruct import create_cmr_dataset, create_sampling_protocol
from cmrbench.sampling.result import (
    SamplingResult,
    binary_mask_provenance,
    calculate_actual_acceleration,
)


QC_PROTOCOL = "cmrbench_reconstruction_anatomy_fidelity_qc_v1"
QC_SELECTION_PROTOCOL = DEFAULT_QC_POLICY
FASTMRI_EVALUATION_ESPIRIT = {
    "source": "espirit",
    "implementation": "sigpy",
    "calibration": "sampling_acs",
    "kernel_width": 6,
    "calibration_matrix_threshold": 0.02,
    "eigenvalue_crop_threshold": 0.95,
    "max_iterations": 100,
}
METRICS_COLUMNS = (
    "subject",
    "model",
    "implementation",
    "variant",
    "sampling_method",
    "sampling_implementation",
    "requested_acceleration",
    "actual_acceleration",
    "ncalib",
    "seed",
    "metric_protocol",
    "frame_count",
    "zf_nrmse",
    "zf_psnr",
    "zf_ssim",
    "reconstruction_nrmse",
    "reconstruction_psnr",
    "reconstruction_ssim",
)
REQUIRED_RESULT_COLUMNS = {
    "subject_id",
    "model_name",
    "implementation",
    "variant",
    "sampling_method",
    "sampling_implementation",
    "requested_acceleration",
    "actual_acceleration",
    "sampling_seed",
    "sampling_parameters",
}

DatasetFactory = Callable[[ExperimentConfig], CineMRIDataset]
SamplingFactory = Callable[[ExperimentConfig], Any]
QCWriter = Callable[..., dict[str, Any]]
GIFWriter = Callable[..., dict[str, Any]]


def mid_slice_fixed_phase_qc_frames(
    shape: tuple[int, int, int, int],
    phase_indices: Sequence[int] = DEFAULT_QC_PHASE_INDICES,
) -> tuple[tuple[int, int], ...]:
    """Select zero-based phases on the middle slice of the actual cine shape."""

    slices, phases = shape[:2]
    if slices < 1 or phases < 1:
        raise ValueError("QC selection requires non-empty S and T dimensions")
    selected_phases = tuple(phase_indices)
    if not selected_phases:
        raise ValueError("QC phase_indices must not be empty")
    for phase_index in selected_phases:
        if phase_index < 0 or phase_index >= phases:
            raise ValueError(
                f"zero-based QC phase {phase_index} is outside T={phases}; "
                "set evaluation.qc.phase_indices explicitly for this cine"
            )
    return tuple((slices // 2, phase_index) for phase_index in selected_phases)


def evaluate_reconstruction(
    reconstruction_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    qc_frames: Sequence[tuple[int, int]] | None = None,
    gif_enabled: bool | None = None,
    dataset_factory: DatasetFactory = create_cmr_dataset,
    sampling_factory: SamplingFactory = create_sampling_protocol,
    anatomy_qc_writer: QCWriter = save_anatomy_qc,
    fidelity_qc_writer: QCWriter = save_fidelity_qc,
    gif_writer: GIFWriter = save_mid_slice_reconstruction_gif,
) -> list[dict[str, Any]]:
    """Evaluate all subjects declared by ``results.csv`` and fail closed."""

    source_root = Path(reconstruction_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"reconstruction directory not found: {source_root}")
    evaluation_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_root / "evaluation"
    )
    if evaluation_root == source_root:
        raise ValueError("evaluation output directory must differ from reconstruction root")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    qc_root = evaluation_root / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    _clear_previous_standard_outputs(evaluation_root, qc_root)

    metadata_path = evaluation_root / "evaluation_metadata.json"
    started_at = _utc_now()
    status: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "evaluation_protocol_identifier": METRIC_PROTOCOL_V2,
        "normalization_protocol_identifier": METRIC_PROTOCOL_V2,
        "qc_protocol_identifier": QC_PROTOCOL,
        "qc_selection_protocol_identifier": "pending_effective_config",
        "gif_protocol_identifier": None,
        "source_reconstruction_directory": str(source_root),
        "evaluation_output_directory": str(evaluation_root),
        "cmrbench_git_commit": _git_commit(),
        "number_of_subjects_evaluated": 0,
    }
    _write_json(metadata_path, status)

    try:
        config_path = _require_file(source_root / "config.yaml", "effective config")
        experiment_path = _require_file(
            source_root / "experiment_metadata.json", "experiment metadata"
        )
        results_path = _require_file(source_root / "results.csv", "results manifest")
        config = ExperimentConfig.load(config_path)
        if config.evaluation.metric_protocol != METRIC_PROTOCOL_V2:
            raise ValueError(
                "generic evaluation supports only frozen Metrics v2: "
                f"{METRIC_PROTOCOL_V2}"
            )
        qc_enabled = config.evaluation.qc.enabled
        if qc_frames is not None and not qc_enabled:
            raise ValueError(
                "--qc-frame cannot override evaluation.qc.enabled=false"
            )
        selected_gif_enabled = (
            config.evaluation.gif.enabled
            if gif_enabled is None
            else gif_enabled
        )
        qc_selection_identifier = (
            "disabled"
            if not qc_enabled
            else (
                QC_SELECTION_PROTOCOL
                if qc_frames is None
                else "explicit_cli_slice_phase_v1"
            )
        )
        status.update(
            {
                "qc_selection_protocol_identifier": qc_selection_identifier,
                "gif_protocol_identifier": (
                    GIF_PROTOCOL if selected_gif_enabled else None
                ),
            }
        )
        _write_json(metadata_path, status)
        if not config.output.save_reconstruction or not config.output.save_metadata:
            raise ValueError(
                "effective config must save both reconstructions and subject metadata"
            )
        experiment_metadata = _read_json_object(experiment_path)
        _validate_experiment_metadata(config, experiment_metadata)
        result_rows = _read_result_rows(results_path)
        _validate_run_level_results(config, experiment_metadata, result_rows)

        dataset = dataset_factory(config)
        samples: dict[str, Any] = {}
        for sample in dataset:
            if sample.subject_id in samples:
                raise ValueError(f"dataset contains duplicate subject {sample.subject_id!r}")
            samples[sample.subject_id] = sample
        declared_subjects = [row["subject_id"] for row in result_rows]
        if set(samples) != set(declared_subjects):
            raise ValueError(
                "results.csv subjects do not exactly match the effective-config dataset: "
                f"results={declared_subjects}, dataset={list(samples)}"
            )

        protocol = sampling_factory(config)
        zero_filled_adapter = _create_zero_filled_adapter()
        zero_filled_adapter.load()
        metric_rows: list[dict[str, Any]] = []
        subject_records: list[dict[str, Any]] = []

        for result_row in result_rows:
            subject = result_row["subject_id"]
            sample = samples[subject]
            candidate_path = _require_file(
                source_root / "subjects" / f"{subject}_reconstruction.npy",
                f"candidate reconstruction for {subject}",
            )
            subject_metadata_path = _require_file(
                source_root / "subjects" / f"{subject}_metadata.json",
                f"subject metadata for {subject}",
            )
            subject_metadata = _read_json_object(subject_metadata_path)
            recorded_sampling = _sampling_from_result_row(result_row)
            _validate_subject_metadata(
                subject,
                result_row,
                recorded_sampling,
                subject_metadata,
                experiment_model=experiment_metadata["model"],
            )
            subject_model_metadata = subject_metadata.get("metadata", {})
            model_conditioning_provenance = (
                subject_model_metadata.get("model_sampling")
                if isinstance(subject_model_metadata, dict)
                else None
            )

            prepared = prepare_reconstruction_input(
                sample,
                mode=config.sampling.mode,  # type: ignore[arg-type]
                protocol=protocol,
                acceleration=config.sampling.acceleration,
                seed=config.sampling.seed,
            )
            prepared = attach_dataset_sensitivity_maps(
                prepared,
                config.dataset.parameters,
                configured_sampling=config.sampling,
            )
            sampling = prepared.sampling
            if sampling is None:
                raise RuntimeError(f"{subject} recreated input has no sampling provenance")
            verified_sampling = _verify_sampling_parity(
                subject, sampling, recorded_sampling
            )
            reference = np.asarray(sample.reference)
            candidate = np.load(candidate_path, allow_pickle=False)
            _validate_candidate(subject, reference, candidate)
            zero_filled_input, zero_filled_smaps = prepare_zero_filled_input(
                prepared, config
            )
            calibration_record = dict(
                zero_filled_input.metadata.get("cmrbench_calibration") or {}
            )
            reconstruction_calibration = (
                subject_model_metadata.get("cmrbench_calibration")
                if isinstance(subject_model_metadata, dict)
                else None
            )
            if isinstance(reconstruction_calibration, dict):
                for key in (
                    "reconstruction_espirit_calib_width",
                    "cardiomm_num_low_frequencies",
                ):
                    calibration_record[key] = reconstruction_calibration.get(key)
            zero_filled = np.asarray(
                zero_filled_adapter.reconstruct(zero_filled_input).reconstruction
            )
            _validate_candidate(subject, reference, zero_filled, label="zero-filled")

            common_metric_arguments = {
                "subject": subject,
                "acceleration_requested": recorded_sampling[
                    "requested_acceleration"
                ],
                "acceleration_actual": recorded_sampling["actual_acceleration"],
                "sampling_method": recorded_sampling["sampling_method"],
                "seed": recorded_sampling["sampling_seed"],
                "protocol": METRIC_PROTOCOL_V2,
            }
            zf_frames = compute_cine_frame_metrics(
                reference,
                zero_filled,
                model="zero_filled",
                **common_metric_arguments,
            )
            reconstruction_frames = compute_cine_frame_metrics(
                reference,
                candidate,
                model=result_row["model_name"],
                **common_metric_arguments,
            )
            zf_summary = summarize_subject_metrics(zf_frames)
            reconstruction_summary = summarize_subject_metrics(
                reconstruction_frames
            )
            if zf_summary["reference_p99_5"] != reconstruction_summary[
                "reference_p99_5"
            ]:
                raise RuntimeError(f"{subject} Metrics v2 reference scales differ")

            ncalib = _extract_unique_ncalib(
                recorded_sampling["sampling_parameters"]
            )
            metric_row = {
                "subject": subject,
                "model": result_row["model_name"],
                "implementation": result_row["implementation"],
                "variant": result_row["variant"],
                "sampling_method": recorded_sampling["sampling_method"],
                "sampling_implementation": recorded_sampling[
                    "sampling_implementation"
                ],
                "requested_acceleration": recorded_sampling[
                    "requested_acceleration"
                ],
                "actual_acceleration": recorded_sampling["actual_acceleration"],
                "ncalib": ncalib,
                "seed": recorded_sampling["sampling_seed"],
                "metric_protocol": METRIC_PROTOCOL_V2,
                "frame_count": reconstruction_summary["frame_count"],
                "zf_nrmse": zf_summary["mean_nrmse"],
                "zf_psnr": zf_summary["mean_psnr"],
                "zf_ssim": zf_summary["mean_ssim"],
                "reconstruction_nrmse": reconstruction_summary["mean_nrmse"],
                "reconstruction_psnr": reconstruction_summary["mean_psnr"],
                "reconstruction_ssim": reconstruction_summary["mean_ssim"],
            }
            metric_rows.append(metric_row)

            reconstruction_label = _reconstruction_label(result_row)
            anatomy_path = qc_root / f"{subject}_anatomy_qc.png"
            fidelity_path = qc_root / f"{subject}_fidelity_qc.png"
            selected_frames = (
                tuple(qc_frames)
                if qc_frames is not None
                else mid_slice_fixed_phase_qc_frames(
                    tuple(int(v) for v in reference.shape),
                    config.evaluation.qc.phase_indices,
                )
            ) if qc_enabled else ()
            expected_frames = [
                {"slice": slice_index, "phase": phase_index}
                for slice_index, phase_index in selected_frames
            ]
            anatomy: dict[str, Any] | None = None
            fidelity: dict[str, Any] | None = None
            if qc_enabled:
                anatomy = anatomy_qc_writer(
                    anatomy_path,
                    subject=subject,
                    acceleration=_display_acceleration(
                        recorded_sampling["requested_acceleration"]
                    ),
                    reference=reference,
                    zero_filled=zero_filled,
                    reconstruction=candidate,
                    frames=selected_frames,
                    canonical_mask=sampling.mask,
                    actual_acceleration=recorded_sampling[
                        "actual_acceleration"
                    ],
                    error_vmax=0.2,
                    error_cmap="turbo",
                    reconstruction_label=reconstruction_label,
                )
                fidelity = fidelity_qc_writer(
                    fidelity_path,
                    subject=subject,
                    acceleration=_display_acceleration(
                        recorded_sampling["requested_acceleration"]
                    ),
                    reference=reference,
                    zero_filled=zero_filled,
                    reconstruction=candidate,
                    frames=selected_frames,
                    canonical_mask=sampling.mask,
                    actual_acceleration=recorded_sampling[
                        "actual_acceleration"
                    ],
                    error_vmax=1.0,
                    error_cmap="turbo",
                    reconstruction_label=reconstruction_label,
                )
                if anatomy.get("frames") != expected_frames or fidelity.get(
                    "frames"
                ) != expected_frames:
                    raise RuntimeError(
                        f"{subject} anatomy/fidelity QC frames differ"
                    )

            gif_path = qc_root / f"{subject}_mid_slice_reconstruction.gif"
            gif_metadata = (
                gif_writer(gif_path, reconstruction=candidate)
                if selected_gif_enabled
                else None
            )
            unique_slices = sorted({frame[0] for frame in selected_frames})
            phase_indices = [frame[1] for frame in selected_frames]
            qc_metadata_path = qc_root / f"{subject}_qc_metadata.json"
            _write_json(
                qc_metadata_path,
                {
                    "qc_protocol_identifier": QC_PROTOCOL,
                    "selection_protocol_identifier": qc_selection_identifier,
                    "subject": subject,
                    "qc_enabled": qc_enabled,
                    "frames": expected_frames,
                    "selected_slice_index": (
                        unique_slices[0] if len(unique_slices) == 1 else None
                    ),
                    "phase_indices": phase_indices,
                    "phase_indexing": "zero_based",
                    "display_provenance": (
                        {
                            "error_map_colormap": "turbo",
                            "colorbar_label": None,
                            "anatomy_error_display_range": [0.0, 0.2],
                            "fidelity_error_display_range": [0.0, 1.0],
                            "mask_source": (
                                "recreated verified canonical sampling mask"
                            ),
                            "mask_indices": expected_frames,
                            "actual_acceleration_displayed": recorded_sampling[
                                "actual_acceleration"
                            ],
                            "actual_acceleration_scope": (
                                "global complete canonical sampling mask"
                            ),
                        }
                        if qc_enabled
                        else None
                    ),
                    "anatomy": anatomy,
                    "fidelity": fidelity,
                    "gif": gif_metadata,
                    "zero_filled_sensitivity_maps": zero_filled_smaps,
                    "calibration": calibration_record,
                },
            )
            subject_records.append(
                {
                    "subject": subject,
                    "model": {
                        "name": result_row["model_name"],
                        "implementation": result_row["implementation"],
                        "variant": result_row["variant"],
                    },
                    "model_conditioning": model_conditioning_provenance,
                    "zero_filled_sensitivity_maps": zero_filled_smaps,
                    "calibration": calibration_record,
                    "sampling": {
                        "subject": subject,
                        **recorded_sampling,
                        "ncalib": ncalib,
                        "verified_binary_mask_sha256": verified_sampling.get(
                            "binary_mask_sha256"
                        ),
                    },
                    "canonical_shape": list(reference.shape),
                    "arrays": {
                        "reference_sha256": array_sha256(reference),
                        "zero_filled_sha256": array_sha256(zero_filled),
                        "candidate_sha256": array_sha256(candidate),
                    },
                    "qc_frames": expected_frames,
                    "qc_selection": {
                        "policy_identifier": qc_selection_identifier,
                        "selected_slice_index": (
                            unique_slices[0] if len(unique_slices) == 1 else None
                        ),
                        "phase_indices": phase_indices,
                        "phase_indexing": "zero_based",
                    },
                    "gif": gif_metadata,
                    "qc_metadata": str(qc_metadata_path),
                }
            )

        metrics_csv = evaluation_root / "metrics_v2.csv"
        metrics_json = evaluation_root / "metrics_v2.json"
        _write_csv(metrics_csv, metric_rows)
        _write_json(
            metrics_json,
            {
                "metric_protocol": metric_protocol_provenance(METRIC_PROTOCOL_V2),
                "primary_aggregation": "equal arithmetic mean over all S*T frames",
                "subject_count": len(metric_rows),
                "subjects": metric_rows,
            },
        )
        status.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "source_reconstruction_experiment_name": config.experiment.name,
                "source_model_provenance": experiment_metadata["model"],
                "sampling_provenance": [
                    record["sampling"] for record in subject_records
                ],
                "model_conditioning_provenance": [
                    record["model_conditioning"] for record in subject_records
                ],
                "zero_filled_sensitivity_map_provenance": [
                    record["zero_filled_sensitivity_maps"] for record in subject_records
                ],
                "calibration_provenance": [
                    record["calibration"] for record in subject_records
                ],
                "metric_protocol": metric_protocol_provenance(METRIC_PROTOCOL_V2),
                "qc_protocol": {
                    "identifier": QC_PROTOCOL,
                    "enabled": qc_enabled,
                    "anatomy_function": "save_anatomy_qc",
                    "fidelity_function": "save_fidelity_qc",
                    "anatomy_error_display_range": [0.0, 0.2],
                    "fidelity_error_display_range": [0.0, 1.0],
                    "error_map_colormap": "turbo",
                    "colorbar_label": None,
                    "mask_panel_source": (
                        "recreated verified canonical sampling mask at each "
                        "selected slice/phase"
                    ),
                    "actual_acceleration_label_scope": (
                        "global complete canonical sampling mask"
                    ),
                    "selection_policy_identifier": qc_selection_identifier,
                    "configured_phase_indices": list(
                        config.evaluation.qc.phase_indices
                    ),
                    "phase_indexing": "zero_based",
                },
                "gif_protocol": {
                    "identifier": GIF_PROTOCOL,
                    "enabled": selected_gif_enabled,
                    "source": "abs(reconstruction[S//2, 0:T])",
                    "fixed_temporal_display_scaling": True,
                },
                "number_of_subjects_evaluated": len(metric_rows),
                "subjects": subject_records,
                "outputs": {
                    "metrics_csv": str(metrics_csv),
                    "metrics_json": str(metrics_json),
                    "qc_directory": str(qc_root),
                },
            }
        )
        _write_json(metadata_path, status)
        return metric_rows
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json(metadata_path, status)
        raise


def _create_zero_filled_adapter():
    """Import only the canonical non-foundation-model baseline when evaluating."""

    from cmrbench.models.adapters.zero_filled import ZeroFilledAdapter

    return ZeroFilledAdapter()


def prepare_zero_filled_input(
    input: ReconstructionInput,
    config: ExperimentConfig,
    *,
    estimator: EspiritEstimator | None = None,
    backend_version: str | None = None,
) -> tuple[ReconstructionInput, dict[str, Any] | None]:
    """Supply maps only to the evaluation baseline, never to the model input."""

    if input.sensitivity_maps is not None:
        return input, {
            "purpose": "evaluation_zero_filled",
            "source": "preexisting_reconstruction_input",
            "reused_without_reestimation": True,
            "estimated_in_evaluation": False,
            "preparation": input.metadata.get("cmrbench_sensitivity_maps"),
            "calibration": input.metadata.get("cmrbench_calibration"),
        }

    shape = tuple(int(value) for value in input.kspace.shape)
    if config.dataset.name != "fastmri_multicoil" or shape[1] != 1 or shape[-1] < 2:
        return input, None
    prewhitened = config.dataset.parameters.get("prewhitened")
    if prewhitened is not None and not isinstance(prewhitened, bool):
        raise ValueError("dataset.parameters.prewhitened must be Boolean or null")

    # The reconstruction/evaluation sampling path already applies this mask.
    # Apply it again here so even a malformed acquired input cannot expose
    # unacquired source samples to ESPIRiT or the zero-filled baseline.
    acquired = replace(
        input,
        kspace=np.asarray(input.kspace) * np.asarray(input.mask)[..., None],
    )
    calibration = resolve_input_calibration(
        acquired.sampling, tuple(acquired.kspace.shape),
        configured_sampling=config.sampling,
    ) if acquired.sampling is not None else None
    maps_config = resolve_espirit_calibration_config(
        FASTMRI_EVALUATION_ESPIRIT, calibration
    )
    prepared = attach_prepared_sensitivity_maps(
        acquired,
        config=maps_config,
        prewhitened=prewhitened,
        estimator=estimator,
        backend_version=backend_version,
    )
    details = prepared.metadata["cmrbench_sensitivity_maps"]
    calibration_record = dict(prepared.metadata.get("cmrbench_calibration") or {})
    calibration_record["evaluation_espirit_calib_width"] = maps_config.calib_width
    prepared.metadata["cmrbench_calibration"] = calibration_record
    maps_finite = bool(np.all(np.isfinite(prepared.sensitivity_maps)))
    if not maps_finite:
        raise ValueError("evaluation-only ESPIRiT sensitivity maps contain NaN or Inf")
    return prepared, {
        "purpose": "evaluation_zero_filled",
        "strategy": "espirit",
        "backend": "sigpy",
        "backend_version": details["backend_version"],
        "calib_width": details["calibration"]["width"],
        "calibration_source": calibration.calibration_source,
        "requested_calibration_width": calibration.requested_calibration_width,
        "realized_acs_width": details["actual_acs_width"],
        "source": "realized acquired ACS from recreated masked multi-coil k-space",
        "model_input_used": False,
        "estimated_in_evaluation": True,
        "binary_mask_sha256": binary_mask_provenance(prepared.mask)["binary_mask_sha256"],
        "realized_acs_extent": details["actual_acs_width"],
        "calibration_bounds": {
            "ky_start": details["calibration"]["ky_start"],
            "ky_stop": details["calibration"]["ky_stop"],
        },
        "coil_count": details["coil_count"],
        "output_shape": details["output_shape"],
        "finite": maps_finite,
        "preparation": details,
    }


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_RESULT_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(
                "results.csv lacks required provenance columns: "
                + ", ".join(sorted(missing))
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("results.csv contains no successfully reconstructed subjects")
    rows: list[dict[str, Any]] = []
    subjects: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        subject = _required_text(raw.get("subject_id"), f"results.csv line {index} subject_id")
        if Path(subject).name != subject or Path(subject).stem != subject:
            raise ValueError(f"unsafe or non-canonical subject_id {subject!r}")
        if subject in subjects:
            raise ValueError(f"results.csv contains duplicate subject {subject!r}")
        subjects.add(subject)
        parameters = _parse_json_object(
            raw.get("sampling_parameters"),
            f"results.csv line {index} sampling_parameters",
        )
        sampling_implementation = _required_text(
            raw.get("sampling_implementation"),
            f"results.csv line {index} sampling_implementation",
        )
        rows.append(
            {
                **raw,
                "subject_id": subject,
                "model_name": _required_text(
                    raw.get("model_name"), f"results.csv line {index} model_name"
                ),
                "implementation": _optional_text(raw.get("implementation")),
                "variant": _optional_text(raw.get("variant")),
                "sampling_method": _required_text(
                    raw.get("sampling_method"),
                    f"results.csv line {index} sampling_method",
                ),
                "sampling_implementation": sampling_implementation,
                "requested_acceleration": _optional_float(
                    raw.get("requested_acceleration"),
                    f"results.csv line {index} requested_acceleration",
                ),
                "actual_acceleration": _required_float(
                    raw.get("actual_acceleration"),
                    f"results.csv line {index} actual_acceleration",
                ),
                "sampling_seed": _optional_int(
                    raw.get("sampling_seed"),
                    f"results.csv line {index} sampling_seed",
                ),
                "sampling_parameters": parameters,
            }
        )
    return rows


def _sampling_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampling_method": row["sampling_method"],
        "sampling_implementation": row["sampling_implementation"],
        "requested_acceleration": row["requested_acceleration"],
        "actual_acceleration": row["actual_acceleration"],
        "sampling_seed": row["sampling_seed"],
        "sampling_parameters": row["sampling_parameters"],
    }


def _validate_experiment_metadata(
    config: ExperimentConfig, metadata: dict[str, Any]
) -> None:
    required = {
        "experiment_name",
        "config",
        "sampling_mode",
        "sampling_method",
        "sampling_implementation",
        "requested_acceleration",
        "sampling_seed",
        "actual_accelerations",
        "model",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            "experiment metadata lacks required provenance: "
            + ", ".join(sorted(missing))
        )
    if metadata["config"] != config.as_dict():
        raise ValueError("experiment metadata config differs from effective config.yaml")
    expected = {
        "experiment_name": config.experiment.name,
        "sampling_mode": config.sampling.mode,
        "sampling_method": config.sampling.method,
        "sampling_implementation": config.sampling.implementation,
        "requested_acceleration": config.sampling.acceleration,
        "sampling_seed": config.sampling.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"experiment metadata {key}={metadata.get(key)!r} differs from "
                f"effective config value {value!r}"
            )
    model = metadata.get("model")
    if not isinstance(model, dict) or model.get("model_name") != config.model.name:
        raise ValueError("experiment metadata model differs from effective config")


def _validate_run_level_results(
    config: ExperimentConfig,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        if row["model_name"] != config.model.name:
            raise ValueError(
                f"results.csv model {row['model_name']!r} differs from effective "
                f"config model {config.model.name!r}"
            )
    recorded_actual = metadata.get("actual_accelerations")
    if not isinstance(recorded_actual, list) or len(recorded_actual) != len(rows):
        raise ValueError("experiment metadata actual_accelerations is incomplete")
    for index, (actual, row) in enumerate(zip(recorded_actual, rows)):
        if actual is None or not _same_float(
            float(actual), row["actual_acceleration"]
        ):
            raise ValueError(
                f"experiment metadata actual acceleration at row {index} differs "
                "from results.csv"
            )
    resolved = metadata.get("resolved_sampling_implementations")
    expected_implementations = sorted(
        {str(row["sampling_implementation"]) for row in rows}
    )
    if resolved != expected_implementations:
        raise ValueError(
            "experiment metadata resolved sampling implementations differ from "
            "results.csv"
        )


def _validate_subject_metadata(
    subject: str,
    result_row: dict[str, Any],
    recorded_sampling: dict[str, Any],
    metadata: dict[str, Any],
    *,
    experiment_model: dict[str, Any],
) -> None:
    if metadata.get("subject_id") != subject:
        raise ValueError(f"{subject} subject metadata subject_id mismatch")
    if metadata.get("model_name") != result_row["model_name"]:
        raise ValueError(f"{subject} subject metadata model_name mismatch")
    if metadata.get("model") != experiment_model:
        raise ValueError(
            f"{subject} subject model provenance differs from experiment metadata"
        )
    _compare_sampling_metadata(
        subject, metadata.get("sampling"), recorded_sampling, "subject metadata"
    )
    nested = metadata.get("metadata")
    if not isinstance(nested, dict):
        raise ValueError(f"{subject} subject metadata lacks adapter metadata")
    _compare_sampling_metadata(
        subject,
        nested.get("cmrbench_sampling"),
        recorded_sampling,
        "adapter metadata",
    )
    for field in ("implementation", "variant"):
        if _optional_text(nested.get(field)) != result_row[field]:
            raise ValueError(f"{subject} {field} differs between results and metadata")


def _verify_sampling_parity(
    subject: str,
    sampling: SamplingResult,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    recreated = sampling.as_metadata()
    mask_provenance = binary_mask_provenance(sampling.mask)
    parameters = recorded.get("sampling_parameters")
    recorded_hash = (
        parameters.get("binary_mask_sha256")
        if isinstance(parameters, dict)
        else None
    )
    recreated_hash = mask_provenance["binary_mask_sha256"]
    if recorded_hash is not None and recorded_hash != recreated_hash:
        raise ValueError(f"{subject} regenerated binary mask SHA256 mismatch")
    mask_sha_matches = recorded_hash is not None and recorded_hash == recreated_hash
    _compare_sampling_metadata(
        subject,
        recreated,
        recorded,
        "recreated sampling",
        allow_scikit_image_version_difference=mask_sha_matches,
    )
    recreated_version = _scikit_image_runtime_version(
        recreated.get("sampling_parameters")
    )
    recorded_version = _scikit_image_runtime_version(parameters)
    actual = calculate_actual_acceleration(sampling.mask)
    if actual is None or not _same_float(actual, recorded["actual_acceleration"]):
        raise ValueError(f"{subject} measured mask acceleration differs from results.csv")
    parameters = recorded["sampling_parameters"]
    for key, value in mask_provenance.items():
        if key in parameters and parameters[key] != value:
            raise ValueError(f"{subject} regenerated mask {key} mismatch")
    if mask_sha_matches and recreated_version != recorded_version:
        warnings.warn(
            "Sampling recreation runtime provenance differs: "
            f"scikit_image_version stored={recorded_version!r} "
            f"recreated={recreated_version!r}; physical mask SHA256 is identical, "
            "so validation passes.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mask_provenance


def _compare_sampling_metadata(
    subject: str,
    actual: Any,
    expected: dict[str, Any],
    source: str,
    *,
    allow_scikit_image_version_difference: bool = False,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{subject} {source} lacks sampling provenance")
    keys = (
        "sampling_method",
        "sampling_implementation",
        "requested_acceleration",
        "actual_acceleration",
        "sampling_seed",
        "sampling_parameters",
    )
    missing = [key for key in keys if key not in actual]
    if missing:
        raise ValueError(
            f"{subject} {source} lacks sampling fields: {', '.join(missing)}"
        )
    for key in keys:
        actual_value = actual[key]
        expected_value = expected[key]
        if key in {"requested_acceleration", "actual_acceleration"}:
            equal = _same_optional_float(actual_value, expected_value)
        elif key == "sampling_parameters" and allow_scikit_image_version_difference:
            equal = _without_scikit_image_runtime_version(
                actual_value
            ) == _without_scikit_image_runtime_version(expected_value)
        else:
            equal = actual_value == expected_value
        if not equal:
            raise ValueError(
                f"{subject} {source} {key} mismatch: "
                f"{actual_value!r} != {expected_value!r}"
            )


def _without_scikit_image_runtime_version(parameters: Any) -> Any:
    """Copy sampling parameters without the one runtime-only provenance field."""

    if not isinstance(parameters, dict):
        return parameters
    normalized = dict(parameters)
    generator_parameters = normalized.get("generator_parameters")
    if isinstance(generator_parameters, dict):
        normalized_generator = dict(generator_parameters)
        normalized_generator.pop("scikit_image_version", None)
        normalized["generator_parameters"] = normalized_generator
    return normalized


def _scikit_image_runtime_version(parameters: Any) -> Any:
    if not isinstance(parameters, dict):
        return None
    generator_parameters = parameters.get("generator_parameters")
    if not isinstance(generator_parameters, dict):
        return None
    return generator_parameters.get("scikit_image_version")


def _validate_candidate(
    subject: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    label: str = "candidate",
) -> None:
    if reference.ndim != 4:
        raise ValueError(f"{subject} canonical reference is not [S,T,X,Y]")
    if candidate.ndim != 4 or candidate.shape != reference.shape:
        raise ValueError(
            f"{subject} {label} shape {candidate.shape} differs from canonical "
            f"reference {reference.shape}"
        )
    if not np.all(np.isfinite(candidate)):
        raise ValueError(f"{subject} {label} contains NaN or Inf")


def _extract_unique_ncalib(parameters: dict[str, Any]) -> int | float | None:
    values: list[int | float] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "ncalib" and isinstance(item, (int, float)):
                    values.append(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(parameters)
    unique = list(dict.fromkeys(values))
    if len(unique) > 1:
        raise ValueError(f"sampling provenance contains inconsistent ncalib values: {unique}")
    return unique[0] if unique else None


def _reconstruction_label(row: dict[str, Any]) -> str:
    parts = [row["model_name"]]
    if row["implementation"]:
        parts.append(row["implementation"])
    if row["variant"]:
        parts.append(row["variant"])
    return " ".join(str(part) for part in parts)


def _display_acceleration(value: float | None) -> int:
    if value is None:
        raise ValueError("frozen QC requires a requested acceleration label")
    if not float(value).is_integer():
        raise ValueError("frozen QC requires an integer requested acceleration label")
    return int(value)


def _clear_previous_standard_outputs(evaluation_root: Path, qc_root: Path) -> None:
    for path in (
        evaluation_root / "metrics_v2.csv",
        evaluation_root / "metrics_v2.json",
    ):
        if path.is_file():
            path.unlink()
    for pattern in (
        "*_anatomy_qc.png",
        "*_fidelity_qc.png",
        "*_mid_slice_reconstruction.gif",
        "*_qc_metadata.json",
    ):
        for path in qc_root.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot serialize an empty Metrics v2 table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _parse_json_object(value: Any, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_float(value: Any, name: str) -> float:
    parsed = _optional_float(value, name)
    if parsed is None:
        raise ValueError(f"{name} must be present")
    return parsed


def _optional_float(value: Any, name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(parsed) != str(value).strip():
        raise ValueError(f"{name} must be an integer")
    return parsed


def _same_optional_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _same_float(float(left), float(right))


def _same_float(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _parse_qc_frame(value: str) -> tuple[int, int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("QC frame must use SLICE,PHASE")
    try:
        frame = (int(pieces[0]), int(pieces[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("QC frame indices must be integers") from exc
    if frame[0] < 0 or frame[1] < 0:
        raise argparse.ArgumentTypeError("QC frame indices must be nonnegative")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed generic CMRBench reconstruction with frozen "
            "Metrics v2 and anatomy/fidelity QC without running the model."
        )
    )
    parser.add_argument(
        "--reconstruction-dir",
        required=True,
        help="Completed output directory from python -m cmrbench.reconstruct",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional evaluation output; defaults to RECONSTRUCTION_DIR/evaluation",
    )
    parser.add_argument(
        "--qc-frame",
        action="append",
        type=_parse_qc_frame,
        help=(
            "Explicit display-only SLICE,PHASE selector; repeat as needed. "
            "Default uses actual S//2 with zero-based phases from config "
            "(normally 6 and 22)."
        ),
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        default=None,
        help="Enable the fixed-window mid-slice candidate reconstruction GIF.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        rows = evaluate_reconstruction(
            arguments.reconstruction_dir,
            output_dir=arguments.output_dir,
            qc_frames=arguments.qc_frame,
            gif_enabled=arguments.gif,
        )
    except Exception as exc:
        parser.exit(
            2,
            f"CMRBench evaluation failed: {type(exc).__name__}: {exc}\n",
        )
    output = (
        Path(arguments.output_dir).expanduser().resolve()
        if arguments.output_dir
        else Path(arguments.reconstruction_dir).expanduser().resolve() / "evaluation"
    )
    print(
        f"PASS: evaluated {len(rows)} subject(s) with {METRIC_PROTOCOL_V2}; "
        f"outputs: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
