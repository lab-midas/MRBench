"""Thin cross-runtime orchestrator for reconstruction followed by evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.runtimes import (
    evaluation_runtime,
    reconstruction_runtime,
)


CommandRunner = Callable[..., Any]
RuntimeExists = Callable[[str], bool]
ReconstructionVerifier = Callable[[Path], list[dict[str, str]]]
EvaluationVerifier = Callable[
    [Path, list[dict[str, str]], bool, bool], dict[str, Any]
]


class ExperimentStageError(RuntimeError):
    """A subprocess or its standardized output failed one experiment stage."""


def run_experiment(
    config_path: str | Path,
    *,
    subprocess_runner: CommandRunner = subprocess.run,
    runtime_exists: RuntimeExists | None = None,
    reconstruction_verifier: ReconstructionVerifier | None = None,
    evaluation_verifier: EvaluationVerifier | None = None,
) -> dict[str, Any]:
    """Dispatch the two existing generic modules in strict stage order."""

    source_config = Path(config_path).expanduser().resolve()
    if not source_config.is_file():
        raise FileNotFoundError(f"experiment config not found: {source_config}")
    config = ExperimentConfig.load(source_config)
    reconstruction_root = Path(config.output.root).expanduser().resolve()
    reconstruction_root.mkdir(parents=True, exist_ok=True)
    _invalidate_previous_evaluation(reconstruction_root)
    run_metadata_path = reconstruction_root / "experiment_run_metadata.json"
    run_status: dict[str, Any] = {
        "status": "running",
        "stage": "preflight",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "experiment_name": config.experiment.name,
        "source_config": str(source_config),
        "reconstruction_runtime": None,
        "evaluation_runtime": None,
        "reconstruction_directory": str(reconstruction_root),
        "evaluation_enabled": config.evaluation.enabled,
    }
    _write_json(run_metadata_path, run_status)

    try:
        model_python = reconstruction_runtime(config)
        eval_python = evaluation_runtime()
        exists = runtime_exists or (lambda value: Path(value).is_file())
        _require_runtime(model_python, "reconstruction", exists)
        if config.evaluation.enabled:
            _require_runtime(eval_python, "evaluation", exists)
        if config.evaluation.enabled and (
            not config.output.save_reconstruction or not config.output.save_metadata
        ):
            raise ValueError(
                "end-to-end evaluation requires output.save_reconstruction=true and "
                "output.save_metadata=true"
            )
    except Exception as exc:
        _record_failure(run_metadata_path, run_status, "preflight", exc)
        raise
    run_status.update(
        {
            "stage": "reconstruction",
            "reconstruction_runtime": model_python,
            "evaluation_runtime": (
                eval_python if config.evaluation.enabled else None
            ),
        }
    )
    _write_json(run_metadata_path, run_status)

    reconstruction_command = [
        model_python,
        "-m",
        "cmrbench.reconstruct",
        "--config",
        str(source_config),
    ]
    try:
        subprocess_runner(reconstruction_command, check=True)
    except subprocess.CalledProcessError as exc:
        reconstruction_failure = _optional_failure_metadata(reconstruction_root)
        failure_stage = (
            str(reconstruction_failure.get("stage"))
            if reconstruction_failure is not None
            else "reconstruction"
        )
        _record_failure(
            run_metadata_path,
            run_status,
            failure_stage,
            exc,
            details=reconstruction_failure,
        )
        raise ExperimentStageError(
            f"reconstruction failed with exit code {exc.returncode}; "
            "evaluation was not launched"
        ) from exc
    except Exception as exc:
        _record_failure(run_metadata_path, run_status, "reconstruction", exc)
        raise

    verify_reconstruction = reconstruction_verifier or _verify_reconstruction_output
    try:
        reconstruction_rows = verify_reconstruction(reconstruction_root)
    except Exception as exc:
        _record_failure(
            run_metadata_path, run_status, "reconstruction_verification", exc
        )
        raise ExperimentStageError(
            f"reconstruction process returned success but output is incomplete: {exc}"
        ) from exc

    evaluation_metadata: dict[str, Any] | None = None
    if config.evaluation.enabled:
        run_status["stage"] = "evaluation"
        _write_json(run_metadata_path, run_status)
        evaluation_command = [
            eval_python,
            "-m",
            "cmrbench.evaluate",
            "--reconstruction-dir",
            str(reconstruction_root),
        ]
        try:
            subprocess_runner(evaluation_command, check=True)
        except subprocess.CalledProcessError as exc:
            _record_failure(run_metadata_path, run_status, "evaluation", exc)
            raise ExperimentStageError(
                f"evaluation failed with exit code {exc.returncode}; "
                f"valid reconstruction is preserved at {reconstruction_root}"
            ) from exc
        except Exception as exc:
            _record_failure(run_metadata_path, run_status, "evaluation", exc)
            raise

        verify_evaluation = evaluation_verifier or _verify_evaluation_output
        try:
            evaluation_metadata = verify_evaluation(
                reconstruction_root,
                reconstruction_rows,
                config.evaluation.qc.enabled,
                config.evaluation.gif.enabled,
            )
        except Exception as exc:
            _record_failure(
                run_metadata_path, run_status, "evaluation_verification", exc
            )
            raise ExperimentStageError(
                f"evaluation process returned success but output is incomplete: {exc}"
            ) from exc

    report = {
        "status": "complete",
        "stage": "complete",
        "started_at_utc": run_status["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "experiment_name": config.experiment.name,
        "source_config": str(source_config),
        "reconstruction_runtime": model_python,
        "evaluation_runtime": eval_python if config.evaluation.enabled else None,
        "reconstruction_directory": str(reconstruction_root),
        "evaluation_enabled": config.evaluation.enabled,
        "subjects": reconstruction_rows,
        "outputs": {
            "results_csv": str(reconstruction_root / "results.csv"),
            "candidate_reconstructions": [
                str(
                    reconstruction_root
                    / "subjects"
                    / f"{row['subject_id']}_reconstruction.npy"
                )
                for row in reconstruction_rows
            ],
            "metrics_v2_csv": (
                str(reconstruction_root / "evaluation" / "metrics_v2.csv")
                if config.evaluation.enabled
                else None
            ),
            "metrics_v2_json": (
                str(reconstruction_root / "evaluation" / "metrics_v2.json")
                if config.evaluation.enabled
                else None
            ),
            "qc_directory": (
                str(reconstruction_root / "evaluation" / "qc")
                if config.evaluation.enabled
                else None
            ),
            "qc_artifacts": (
                [
                    {
                        "subject_id": row["subject_id"],
                        "anatomy": (
                            str(
                                reconstruction_root
                                / "evaluation"
                                / "qc"
                                / f"{row['subject_id']}_anatomy_qc.png"
                            )
                            if config.evaluation.qc.enabled
                            else None
                        ),
                        "fidelity": (
                            str(
                                reconstruction_root
                                / "evaluation"
                                / "qc"
                                / f"{row['subject_id']}_fidelity_qc.png"
                            )
                            if config.evaluation.qc.enabled
                            else None
                        ),
                        "metadata": str(
                            reconstruction_root
                            / "evaluation"
                            / "qc"
                            / f"{row['subject_id']}_qc_metadata.json"
                        ),
                    }
                    for row in reconstruction_rows
                ]
                if config.evaluation.enabled
                else []
            ),
            "candidate_gifs": (
                [
                    str(
                        reconstruction_root
                        / "evaluation"
                        / "qc"
                        / f"{row['subject_id']}_mid_slice_reconstruction.gif"
                    )
                    for row in reconstruction_rows
                ]
                if config.evaluation.enabled and config.evaluation.gif.enabled
                else []
            ),
        },
        "evaluation_metadata_status": (
            evaluation_metadata.get("status")
            if evaluation_metadata is not None
            else "disabled"
        ),
    }
    _write_json(run_metadata_path, report)
    return report


def _require_runtime(
    runtime: str, stage: str, runtime_exists: RuntimeExists
) -> None:
    if not runtime_exists(runtime):
        raise FileNotFoundError(
            f"validated {stage} Python interpreter not found: {runtime}"
        )


def _verify_reconstruction_output(root: Path) -> list[dict[str, str]]:
    for name in ("config.yaml", "experiment_metadata.json", "results.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"missing generic reconstruction file: {root / name}")
    with (root / "results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("generic reconstruction results.csv is empty")
    for row in rows:
        subject = (row.get("subject_id") or "").strip()
        if not subject:
            raise ValueError("generic reconstruction results.csv has empty subject_id")
        for suffix in ("reconstruction.npy", "metadata.json"):
            path = root / "subjects" / f"{subject}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing generic reconstruction subject output: {path}"
                )
    return rows


def _verify_evaluation_output(
    root: Path,
    reconstruction_rows: list[dict[str, str]],
    qc_enabled: bool,
    gif_enabled: bool,
) -> dict[str, Any]:
    evaluation_root = root / "evaluation"
    for name in ("metrics_v2.csv", "metrics_v2.json", "evaluation_metadata.json"):
        if not (evaluation_root / name).is_file():
            raise FileNotFoundError(
                f"missing generic evaluation file: {evaluation_root / name}"
            )
    metadata = _read_json(evaluation_root / "evaluation_metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError(
            "generic evaluation metadata is not complete: "
            f"{metadata.get('status')!r}"
        )
    if metadata.get("number_of_subjects_evaluated") != len(reconstruction_rows):
        raise ValueError("evaluation subject count differs from reconstruction results")
    for row in reconstruction_rows:
        subject = str(row["subject_id"])
        qc_root = evaluation_root / "qc"
        required = [qc_root / f"{subject}_qc_metadata.json"]
        if qc_enabled:
            required.extend(
                [
                    qc_root / f"{subject}_anatomy_qc.png",
                    qc_root / f"{subject}_fidelity_qc.png",
                ]
            )
        if gif_enabled:
            required.append(qc_root / f"{subject}_mid_slice_reconstruction.gif")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"missing requested evaluation artifact: {path}")
    return metadata


def _invalidate_previous_evaluation(root: Path) -> None:
    evaluation_root = root / "evaluation"
    if not evaluation_root.is_dir():
        return
    for path in (
        evaluation_root / "metrics_v2.csv",
        evaluation_root / "metrics_v2.json",
    ):
        if path.is_file():
            path.unlink()
    qc_root = evaluation_root / "qc"
    if qc_root.is_dir():
        for pattern in (
            "*_anatomy_qc.png",
            "*_fidelity_qc.png",
            "*_mid_slice_reconstruction.gif",
            "*_qc_metadata.json",
        ):
            for path in qc_root.glob(pattern):
                if path.is_file():
                    path.unlink()
    _write_json(
        evaluation_root / "evaluation_metadata.json",
        {
            "status": "failed",
            "stage": "pre_reconstruction",
            "completed_at_utc": _utc_now(),
            "failure": {
                "type": "InvalidatedByReconstructionRerun",
                "message": (
                    "previous evaluation artifacts were invalidated before a new "
                    "reconstruction attempt"
                ),
            },
        },
    )


def _record_failure(
    path: Path,
    status: dict[str, Any],
    stage: str,
    error: Exception,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    status.update(
        {
            "status": "failed",
            "stage": stage,
            "completed_at_utc": _utc_now(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "reconstruction_failure": details,
            },
        }
    )
    _write_json(path, status)


def _optional_failure_metadata(root: Path) -> dict[str, Any] | None:
    path = root / "reconstruction_failure.json"
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except Exception as exc:
        return {
            "status": "failed",
            "stage": "reconstruction",
            "source": "cmrbench_failure_metadata_reader",
            "exception": {"type": type(exc).__name__, "message": str(exc)},
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run generic CMRBench reconstruction in its validated model runtime, "
            "then generic evaluation in the CMRBench evaluation runtime."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="One experiment YAML containing reconstruction and evaluation settings",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = run_experiment(arguments.config)
    except Exception as exc:
        parser.exit(
            2,
            f"CMRBench experiment failed: {type(exc).__name__}: {exc}\n",
        )
    print(f"PASS: experiment={report['experiment_name']}")
    for row in report["subjects"]:
        print(
            "  subject={} model={} model_implementation={} model_variant={} "
            "sampling={} sampling_implementation={} requested_R={} actual_R={}".format(
                row.get("subject_id"),
                row.get("model_name"),
                row.get("implementation") or "",
                row.get("variant") or "",
                row.get("sampling_method"),
                row.get("sampling_implementation"),
                row.get("requested_acceleration"),
                row.get("actual_acceleration"),
            )
        )
    print(f"  reconstruction_dir={report['reconstruction_directory']}")
    for path in report["outputs"]["candidate_reconstructions"]:
        print(f"  candidate={path}")
    print(f"  metrics_v2_csv={report['outputs']['metrics_v2_csv']}")
    print(f"  metrics_v2_json={report['outputs']['metrics_v2_json']}")
    for artifacts in report["outputs"]["qc_artifacts"]:
        print(f"  anatomy_qc={artifacts['anatomy']}")
        print(f"  fidelity_qc={artifacts['fidelity']}")
        print(f"  qc_metadata={artifacts['metadata']}")
    for path in report["outputs"]["candidate_gifs"]:
        print(f"  candidate_gif={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
