"""Lightweight, dependency-injected experiment runner skeleton."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cmrbench.data.base import CineMRIDataset
from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.prepare import (
    attach_dataset_sensitivity_maps,
    prepare_reconstruction_input,
)
from cmrbench.models.base import ReconstructionModel
from cmrbench.models.errors import UpstreamModelInferenceError
from cmrbench.models.result import ReconstructionResult
from cmrbench.sampling.base import SamplingProtocol

DatasetFactory = Callable[[ExperimentConfig], CineMRIDataset]


class ExperimentRunner:
    """Coordinate already-registered dataset, sampling, and model components."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        dataset_factory: DatasetFactory,
        model: ReconstructionModel,
        sampling_protocol: SamplingProtocol | None = None,
    ):
        self.config = config
        self.dataset_factory = dataset_factory
        self.model = model
        self.sampling_protocol = sampling_protocol

    def run(self) -> list[ReconstructionResult]:
        output_root = Path(self.config.output.root)
        subjects_root = output_root / "subjects"
        output_root.mkdir(parents=True, exist_ok=True)
        subjects_root.mkdir(parents=True, exist_ok=True)
        failure_path = output_root / "reconstruction_failure.json"
        if failure_path.is_file():
            failure_path.unlink()
        try:
            dataset = self.dataset_factory(self.config)
        except Exception as exc:
            _write_reconstruction_failure(failure_path, "data_loading", "cmrbench", exc)
            raise
        try:
            self.model.load()
        except Exception as exc:
            _write_reconstruction_failure(failure_path, "model_load", "upstream_model_runtime", exc)
            raise
        results: list[ReconstructionResult] = []
        result_rows: list[dict[str, Any]] = []

        for sample in dataset:
            try:
                reconstruction_input = prepare_reconstruction_input(
                    sample,
                    mode=self.config.sampling.mode,  # type: ignore[arg-type]
                    protocol=self.sampling_protocol,
                    acceleration=self.config.sampling.acceleration,
                    seed=self.config.sampling.seed,
                )
                reconstruction_input = attach_dataset_sensitivity_maps(
                    reconstruction_input,
                    self.config.dataset.parameters,
                    configured_sampling=self.config.sampling,
                )
            except Exception as exc:
                _write_reconstruction_failure(
                    failure_path,
                    "physical_sampling_preparation",
                    "cmrbench",
                    exc,
                    subject_id=sample.subject_id,
                )
                raise
            try:
                result = self.model.reconstruct(reconstruction_input)
            except UpstreamModelInferenceError as exc:
                _write_reconstruction_failure(
                    failure_path,
                    "model_inference",
                    "upstream_model_runtime",
                    exc,
                    subject_id=sample.subject_id,
                    upstream_type=exc.upstream_exception_type,
                    upstream_message=exc.upstream_exception_message,
                )
                raise
            except Exception as exc:
                _write_reconstruction_failure(
                    failure_path,
                    "adapter_validation",
                    "cmrbench_model_adapter",
                    exc,
                    subject_id=sample.subject_id,
                )
                raise
            if result.subject_id != sample.subject_id:
                raise ValueError("model result subject_id does not match input subject_id")
            results.append(result)
            sampling_metadata = (
                reconstruction_input.sampling.as_metadata()
                if reconstruction_input.sampling is not None
                else {}
            )
            result.metadata["cmrbench_sampling"] = dict(sampling_metadata)
            if "cmrbench_sensitivity_maps" in reconstruction_input.metadata:
                result.metadata["cmrbench_sensitivity_maps"] = dict(
                    reconstruction_input.metadata["cmrbench_sensitivity_maps"]
                )
            calibration = dict(
                reconstruction_input.metadata.get("cmrbench_calibration") or {}
            )
            if result.model_name == "cardiomm":
                model_sampling = result.metadata.get("model_sampling")
                if isinstance(model_sampling, dict):
                    calibration["cardiomm_num_low_frequencies"] = model_sampling.get(
                        "model_internal_num_low_frequencies"
                    )
            result.metadata["cmrbench_calibration"] = calibration
            result_rows.append(
                {
                    "subject_id": result.subject_id,
                    "model_name": result.model_name,
                    "implementation": result.metadata.get("implementation"),
                    "variant": result.metadata.get("variant"),
                    "runtime_seconds": result.runtime_seconds,
                    **sampling_metadata,
                    "model_sampling": result.metadata.get("model_sampling"),
                }
            )
            if self.config.output.save_reconstruction:
                np.save(
                    subjects_root / f"{result.subject_id}_reconstruction.npy",
                    _to_numpy(result.reconstruction),
                    allow_pickle=False,
                )
            if self.config.output.save_metadata:
                subject_metadata = {
                    "subject_id": result.subject_id,
                    "model_name": result.model_name,
                    "dataset": {
                        "reader": self.config.dataset.name,
                        "domain": self.config.dataset.domain,
                        "split": self.config.dataset.split,
                        "selection": dict(self.config.dataset.selection),
                        "canonical_kspace_shape": list(sample.kspace.shape),
                        "canonical_reference_shape": (
                            list(sample.reference.shape)
                            if sample.reference is not None else None
                        ),
                        "canonical_sensitivity_map_shape": (
                            list(sample.sensitivity_maps.shape)
                            if sample.sensitivity_maps is not None else None
                        ),
                        "source_metadata": dict(sample.metadata),
                    },
                    "model": self.model.model_info().as_dict(),
                    "runtime_seconds": result.runtime_seconds,
                    "transforms": [asdict(record) for record in result.transforms],
                    "metadata": result.metadata,
                    "sampling": sampling_metadata,
                }
                _write_json(subjects_root / f"{result.subject_id}_metadata.json", subject_metadata)

        self._save_experiment_records(output_root, result_rows)
        return results

    def _save_experiment_records(
        self, output_root: Path, result_rows: list[dict[str, Any]]
    ) -> None:
        _write_config(output_root / "config.yaml", self.config.as_dict())
        sampling = self.config.sampling
        metadata = {
            "experiment_name": self.config.experiment.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config.as_dict(),
            "dataset": self.config.dataset.name,
            "split": self.config.dataset.split,
            "sampling_mode": sampling.mode,
            "sampling_method": sampling.method,
            "sampling_implementation": sampling.implementation,
            "requested_acceleration": sampling.acceleration,
            "sampling_seed": sampling.seed,
            "actual_accelerations": [row.get("actual_acceleration") for row in result_rows],
            "resolved_sampling_implementations": sorted(
                {
                    str(row["sampling_implementation"])
                    for row in result_rows
                    if row.get("sampling_implementation") is not None
                }
            ),
            "model": self.model.model_info().as_dict(),
            "model_sampling_conditions": [
                row.get("model_sampling") for row in result_rows
            ],
            "software": {"cmrbench_version": "0.1.0", "git_commit": _git_commit()},
        }
        _write_json(output_root / "experiment_metadata.json", metadata)
        fieldnames = [
            "subject_id",
            "model_name",
            "implementation",
            "variant",
            "runtime_seconds",
            "sampling_method",
            "sampling_implementation",
            "requested_acceleration",
            "actual_acceleration",
            "sampling_seed",
            "sampling_parameters",
            "model_sampling",
        ]
        with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in result_rows:
                serialized = dict(row)
                serialized["sampling_parameters"] = json.dumps(
                    serialized.get("sampling_parameters", {}), sort_keys=True
                )
                serialized["model_sampling"] = json.dumps(
                    serialized.get("model_sampling") or {}, sort_keys=True
                )
                writer.writerow(serialized)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _write_reconstruction_failure(
    path: Path,
    stage: str,
    source: str,
    error: Exception,
    *,
    subject_id: str | None = None,
    upstream_type: str | None = None,
    upstream_message: str | None = None,
) -> None:
    _write_json(
        path,
        {
            "status": "failed",
            "stage": stage,
            "outcome": "failed",
            "source": source,
            "subject_id": subject_id,
            "exception": {
                "type": upstream_type or type(error).__name__,
                "message": upstream_message if upstream_message is not None else str(error),
                "wrapped_type": type(error).__name__ if upstream_type else None,
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_config(path: Path, value: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Saving YAML requires PyYAML") from exc
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


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
