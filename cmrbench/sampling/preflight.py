"""Model-free sampling-generator preflight for isolated CMRBench runtimes."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from cmrbench.experiment.config import ExperimentConfig
from cmrbench.experiment.runtimes import (
    CARDIOMM_PYTHON,
    CMRBENCH_EVALUATION_PYTHON,
    RAM_DEEPINV_PYTHON,
    RAM_OFFICIAL_PYTHON,
    SDUM_PYTHON,
)
from cmrbench.reconstruct import create_sampling_protocol
from cmrbench.sampling.registry import default_sampling_registry
from cmrbench.sampling.result import (
    binary_mask_provenance,
    calculate_actual_acceleration,
    detect_complete_central_acs,
    validate_physical_mask,
)


PREFLIGHT_SHAPE = (1, 25, 192, 156, 1)
PREFLIGHT_ACCELERATION = 8
PREFLIGHT_NCALIB = 20
PREFLIGHT_CENTER_FRACTION = 0.04
PREFLIGHT_SEED = 7

RECONSTRUCTION_RUNTIMES = {
    "RAM official": RAM_OFFICIAL_PYTHON,
    "RAM DeepInv": RAM_DEEPINV_PYTHON,
    "SDUM": SDUM_PYTHON,
    "CardioMM": CARDIOMM_PYTHON,
}

ALL_KNOWN_RUNTIMES = {
    **RECONSTRUCTION_RUNTIMES,
    "Generic CMRBench": CMRBENCH_EVALUATION_PYTHON,
}

IMPLEMENTATION_DEPENDENCIES: dict[tuple[str, str], tuple[str, ...]] = {
    ("uniform", "canonical"): ("numpy",),
    ("uniform", "sdum"): ("numpy",),
    ("uniform", "cardiomm"): ("numpy", "scipy", "scikit-image", "external CardioMM source"),
    ("kt_uniform", "sdum"): ("numpy",),
    ("kt_uniform", "cardiomm"): ("numpy", "scipy", "scikit-image", "external CardioMM source"),
    ("kt_gaussian", "sdum"): ("numpy",),
    ("kt_gaussian", "cardiomm"): ("numpy", "scipy", "scikit-image", "external CardioMM source"),
    ("kt_radial", "sdum"): ("numpy", "scipy"),
    ("kt_radial", "cardiomm"): ("numpy", "scipy", "scikit-image", "external CardioMM source"),
    ("random", "sdum"): ("numpy",),
    ("random", "cardiomm"): ("numpy", "scipy", "torch", "fastmri", "external CardioMM source"),
    ("equispaced", "sdum"): ("numpy",),
    ("radial", "cardiomm"): ("numpy", "scipy", "torch", "fastmri", "external CardioMM source"),
    ("vista", "precomputed"): ("numpy", "precomputed text mask"),
    ("vista", "cmrbench"): ("numpy",),
    ("external_mask", "external"): ("numpy", "caller-supplied mask"),
}


def registered_sampling_implementations() -> list[dict[str, Any]]:
    """Enumerate all resolver-visible method/implementation combinations."""

    rows: list[dict[str, Any]] = []
    for method in default_sampling_registry.names():
        implementations = default_sampling_registry.implementations(method)
        if method == "uniform":
            implementations = ("canonical", *implementations)
        elif not implementations:
            implementations = (
                "precomputed"
                if method == "vista"
                else "external"
                if method == "external_mask"
                else "default",
            )
        for implementation in implementations:
            rows.append(
                {
                    "method": method,
                    "implementation": implementation,
                    "dependencies": list(
                        IMPLEMENTATION_DEPENDENCIES.get(
                            (method, implementation), ("unknown",)
                        )
                    ),
                }
            )
    return rows


def _no_acs_compact_mask() -> np.ndarray:
    """Return a valid deterministic [Ky,T] R8-like mask with center absent."""

    _slices, frames, _kx, ky, _coils = PREFLIGHT_SHAPE
    compact = np.zeros((ky, frames), dtype=np.uint8)
    compact[::PREFLIGHT_ACCELERATION, :] = 1
    compact[(ky - 1) // 2 : ky // 2 + 1, :] = 0
    return compact


def _sampling_config(
    method: str,
    implementation: str,
    parameters: Mapping[str, Any],
    *,
    mask_root: str | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "experiment": {"name": f"sampling-preflight-{method}-{implementation}"},
            "dataset": {
                "name": "inhouse_cine",
                "manifest": "unused-preflight-manifest.csv",
                "split": "development",
            },
            "sampling": {
                "mode": "retrospective",
                "method": method,
                "implementation": implementation,
                "acceleration": PREFLIGHT_ACCELERATION,
                "seed": seed,
                "mask_root": mask_root,
                "parameters": dict(parameters),
            },
            "model": {"name": "ram", "parameters": {}},
            "output": {"root": "unused-sampling-preflight-output"},
        }
    )


def _preflight_configs(mask_root: Path) -> Iterable[ExperimentConfig]:
    standard = {"ncalib": PREFLIGHT_NCALIB}
    seeded = {"kt_gaussian", "random", "equispaced", "radial"}
    for row in registered_sampling_implementations():
        method = str(row["method"])
        implementation = str(row["implementation"])
        if method == "vista":
            yield _sampling_config(
                method,
                implementation,
                {},
                mask_root=(
                    str(mask_root) if implementation == "precomputed" else None
                ),
                seed=PREFLIGHT_SEED,
            )
            continue
        if method == "external_mask":
            canonical = np.broadcast_to(
                _no_acs_compact_mask().T[None, :, None, :], PREFLIGHT_SHAPE[:4]
            ).copy()
            yield _sampling_config(
                method,
                implementation,
                {
                    "mask": canonical,
                    "layout": "S,T,Kx,Ky",
                    "source": "external",
                },
                seed=PREFLIGHT_SEED,
            )
            continue
        parameters: dict[str, Any]
        if method in {"random", "equispaced"} and implementation == "sdum":
            parameters = {"center_fraction": PREFLIGHT_CENTER_FRACTION}
        else:
            parameters = dict(standard)
        yield _sampling_config(
            method,
            implementation,
            parameters,
            seed=PREFLIGHT_SEED if method in seeded else None,
        )


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_observation(module: str, distribution: str) -> dict[str, Any]:
    """Observe an optional package without importing its runtime code."""

    try:
        importable = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        importable = False
    return {
        "importable": importable,
        "version": _version(distribution) if importable else None,
    }


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None:
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _failure_classification(error: BaseException) -> tuple[str, str | None]:
    """Classify one generator failure without hiding its original exception."""

    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    message = " ".join(str(item).lower() for item in chain)
    dependency = (
        "scikit-image"
        if "scikit-image" in message or "skimage" in message
        else "scipy"
        if "scipy" in message
        else None
    )
    if dependency is not None and (
        any(isinstance(item, ModuleNotFoundError) for item in chain)
        or "requires " in message
        or "no module named" in message
        or f"missing {dependency}" in message
    ):
        return "dependency_missing", dependency
    if any(isinstance(item, (ImportError, ModuleNotFoundError)) for item in chain):
        return "import_error", dependency
    if any(isinstance(item, FileNotFoundError) for item in chain):
        return "fixture_unavailable", dependency
    if any(isinstance(item, ValueError) for item in chain):
        return "invalid_source_parameter", dependency
    if any(isinstance(item, RuntimeError) for item in chain):
        return "generator_runtime_error", dependency
    return "unexpected_error", dependency


def _run_one(config: ExperimentConfig) -> dict[str, Any]:
    method = str(config.sampling.method)
    implementation = str(config.sampling.implementation)
    record: dict[str, Any] = {
        "method": method,
        "implementation": implementation,
        "case": f"{method}/{implementation}",
        "requested_acceleration": config.sampling.acceleration,
        "seed": config.sampling.seed,
        "configured_parameters": _json_safe(config.sampling.parameters),
        "status": "FAIL",
        "classification": "unexpected_error",
    }
    try:
        protocol = create_sampling_protocol(config)
        if protocol is None:
            raise RuntimeError("resolver unexpectedly returned no retrospective protocol")
        result = protocol.generate(
            PREFLIGHT_SHAPE,
            float(config.sampling.acceleration),
            seed=config.sampling.seed,
        )
        binary = validate_physical_mask(result.mask)
        measured = calculate_actual_acceleration(binary)
        if measured is None or result.actual_acceleration is None or not np.isclose(
            float(result.actual_acceleration), measured, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("preflight actual-R consistency check failed")
        if tuple(binary.shape) != PREFLIGHT_SHAPE[:4]:
            raise RuntimeError(
                f"preflight canonical shape mismatch: {binary.shape} != {PREFLIGHT_SHAPE[:4]}"
            )
        record.update(
            {
                "status": "PASS",
                "classification": "PASS",
                "mask_shape": list(binary.shape),
                "finite": bool(np.isfinite(np.asarray(result.mask)).all()),
                "binary": bool(
                    np.all(
                        (np.asarray(result.mask) == 0)
                        | (np.asarray(result.mask) == 1)
                    )
                ),
                "actual_acceleration": float(measured),
                **binary_mask_provenance(binary),
                "acs": detect_complete_central_acs(binary),
                "sampling_parameters": _json_safe(
                    result.as_metadata()["sampling_parameters"]
                ),
            }
        )
    except Exception as error:  # diagnostic boundary: preserve exact generator error
        classification, dependency = _failure_classification(error)
        record["classification"] = classification
        record["failure"] = {
            "classification": classification,
            "dependency": dependency,
            "exception_chain": _exception_chain(error),
        }
    return record


def run_current_runtime_preflight() -> dict[str, Any]:
    """Exercise every registered first-class pathway in the current interpreter."""

    with tempfile.TemporaryDirectory(prefix="cmrbench-sampling-preflight-") as root:
        root_path = Path(root)
        vista_path = root_path / (
            f"mask_VISTA_{PREFLIGHT_SHAPE[3]}x{PREFLIGHT_SHAPE[1]}_"
            f"acc{PREFLIGHT_ACCELERATION}_{PREFLIGHT_SEED}.txt"
        )
        np.savetxt(vista_path, _no_acs_compact_mask(), fmt="%d", delimiter=",")
        cases = [_run_one(config) for config in _preflight_configs(root_path)]

    scipy = _package_observation("scipy", "scipy")
    scikit_image = _package_observation("skimage", "scikit-image")
    return {
        "schema": "cmrbench_sampling_runtime_preflight_v1",
        "interpreter": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_importable": scipy["importable"],
        "scipy_version": scipy["version"],
        "scikit_image_importable": scikit_image["importable"],
        "scikit_image_version": scikit_image["version"],
        "geometry": {
            "S": PREFLIGHT_SHAPE[0],
            "T": PREFLIGHT_SHAPE[1],
            "Kx": PREFLIGHT_SHAPE[2],
            "Ky": PREFLIGHT_SHAPE[3],
            "C": PREFLIGHT_SHAPE[4],
            "nominal_acceleration": PREFLIGHT_ACCELERATION,
            "ncalib": PREFLIGHT_NCALIB,
            "fastmri_center_fraction": PREFLIGHT_CENTER_FRACTION,
            "effective_fastmri_center_lines": int(
                round(PREFLIGHT_SHAPE[3] * PREFLIGHT_CENTER_FRACTION)
            ),
        },
        "registered_implementations": registered_sampling_implementations(),
        "cases": cases,
        "pass_count": sum(case["status"] == "PASS" for case in cases),
        "fail_count": sum(case["status"] == "FAIL" for case in cases),
        "model_instantiated": False,
        "model_load_called": False,
        "checkpoint_loaded": False,
        "model_forward_called": False,
        "gpu_inference_run": False,
    }


def _runtime_accessible(path: str) -> bool:
    return Path(path).is_file() and os.access(path, os.X_OK)


def run_cross_runtime_preflight(
    runtimes: Mapping[str, str] = ALL_KNOWN_RUNTIMES,
) -> dict[str, Any]:
    """Invoke this module in each interpreter, without model/checkpoint loading."""

    runtime_results: dict[str, Any] = {}
    for name, interpreter in runtimes.items():
        if not _runtime_accessible(interpreter):
            runtime_results[name] = {
                "status": "FAIL",
                "interpreter": interpreter,
                "failure": {
                    "classification": "runtime_access",
                    "exception_chain": [
                        {
                            "type": "FileNotFoundError",
                            "message": (
                                "configured source-of-truth interpreter is not "
                                "accessible from the current host"
                            ),
                        }
                    ],
                },
                "cases": [],
                "model_instantiated": False,
                "model_load_called": False,
                "checkpoint_loaded": False,
                "model_forward_called": False,
            }
            continue
        completed = subprocess.run(
            [
                interpreter,
                "-m",
                "cmrbench.sampling.preflight",
                "--current-runtime",
                "--json-only",
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            runtime_results[name] = {
                "status": "FAIL",
                "interpreter": interpreter,
                "failure": {
                    "classification": "runtime_process",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                "cases": [],
                "model_instantiated": False,
                "model_load_called": False,
                "checkpoint_loaded": False,
                "model_forward_called": False,
            }
            continue
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            runtime_results[name] = {
                "status": "FAIL",
                "interpreter": interpreter,
                "failure": {
                    "classification": "runtime_report_parse",
                    "exception_chain": _exception_chain(error),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                "cases": [],
                "model_instantiated": False,
                "model_load_called": False,
                "checkpoint_loaded": False,
                "model_forward_called": False,
            }
            continue
        report["status"] = "PASS" if report["fail_count"] == 0 else "FAIL"
        report["stderr"] = completed.stderr
        runtime_results[name] = report

    implementation_rows = [
        f"{row['method']}/{row['implementation']}"
        for row in registered_sampling_implementations()
    ]
    matrix: dict[str, dict[str, Any]] = {}
    for implementation in implementation_rows:
        matrix[implementation] = {}
        for runtime_name, runtime in runtime_results.items():
            cases = {
                case["case"]: case for case in runtime.get("cases", [])
            }
            if implementation in cases:
                case = cases[implementation]
                matrix[implementation][runtime_name] = (
                    {"status": "PASS"}
                    if case["status"] == "PASS"
                    else {"status": "FAIL", "failure": case.get("failure")}
                )
            else:
                matrix[implementation][runtime_name] = {
                    "status": "FAIL",
                    "failure": runtime.get("failure"),
                }
    return {
        "schema": "cmrbench_cross_runtime_sampling_preflight_v1",
        "runtime_source_of_truth": dict(ALL_KNOWN_RUNTIMES),
        "runtime_results": runtime_results,
        "matrix": matrix,
        "model_instantiated": False,
        "model_load_called": False,
        "checkpoint_loaded": False,
        "model_forward_called": False,
        "reconstruction_experiment_run": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
            **binary_mask_provenance(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _console_value(value: Any, unavailable: str = "-") -> str:
    return unavailable if value is None else str(value)


def _acs_summary(case: Mapping[str, Any]) -> str:
    acs = case.get("acs")
    if not isinstance(acs, Mapping):
        return "-"
    geometry = str(acs.get("detected_acs_geometry", "none"))
    if geometry == "centered_ky_strip":
        return f"ky:{acs.get('detected_centered_ky_strip_extent', 0)}"
    if geometry == "centered_square":
        return f"square:{acs.get('detected_centered_square_extent', 0)}"
    return "none"


def format_console_report(report: Mapping[str, Any]) -> str:
    """Render a dependency-aware, full-matrix report for the PyCharm console."""

    lines = [
        "CMRBench Sampling Runtime Preflight",
        f"Interpreter : {report.get('interpreter', '-')}",
        f"Python      : {report.get('python_version', '-')}",
        f"NumPy       : {report.get('numpy_version', '-')}",
        "SciPy       : "
        + _console_value(
            report.get("scipy_version")
            if report.get("scipy_importable")
            else None,
            "NOT IMPORTABLE",
        ),
        "scikit-image: "
        + _console_value(
            report.get("scikit_image_version")
            if report.get("scikit_image_importable")
            else None,
            "NOT IMPORTABLE",
        ),
        "",
    ]
    columns = (
        ("method", 14),
        ("implementation", 14),
        ("status", 6),
        ("classification", 25),
        ("dependency", 14),
        ("shape", 18),
        ("points", 10),
        ("actual-R", 10),
        ("ACS", 12),
        ("SHA256", 12),
    )

    def row(values: Iterable[Any]) -> str:
        return " | ".join(
            str(value)[:width].ljust(width)
            for value, (_name, width) in zip(values, columns)
        ).rstrip()

    lines.append(row(name for name, _width in columns))
    lines.append("-+-".join("-" * width for _name, width in columns))
    failures: list[str] = []
    for case in report.get("cases", []):
        failure = case.get("failure", {})
        dependency = failure.get("dependency") if isinstance(failure, Mapping) else None
        shape = case.get("mask_shape")
        sha = case.get("binary_mask_sha256")
        lines.append(
            row(
                (
                    case.get("method", "-"),
                    case.get("implementation", "-"),
                    case.get("status", "-"),
                    case.get("classification", "-"),
                    _console_value(dependency),
                    "x".join(str(item) for item in shape) if shape else "-",
                    _console_value(case.get("sampled_points")),
                    (
                        f"{float(case['actual_acceleration']):.6g}"
                        if case.get("actual_acceleration") is not None
                        else "-"
                    ),
                    _acs_summary(case),
                    str(sha)[:12] if sha else "-",
                )
            )
        )
        if case.get("status") == "FAIL" and isinstance(failure, Mapping):
            chain = failure.get("exception_chain", [])
            exception = chain[0] if chain else {"type": "unknown", "message": ""}
            failures.append(
                f"FAIL {case.get('case', '-')}: {exception.get('type', 'unknown')}: "
                f"{exception.get('message', '')}"
            )
    if failures:
        lines.extend(("", *failures))
    lines.extend(
        (
            "",
            f"Summary: PASS={report.get('pass_count', 0)} FAIL={report.get('fail_count', 0)}",
            "Model safety: model_instantiated=false; checkpoint_loaded=false; "
            "model_forward_called=false",
        )
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise real CMRBench sampling resolver/generator paths only."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--current-runtime", action="store_true")
    group.add_argument("--all-runtimes", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON (used by the cross-runtime subprocess protocol).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = (
        run_cross_runtime_preflight()
        if arguments.all_runtimes
        else run_current_runtime_preflight()
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        serialized
        if arguments.json_only or arguments.all_runtimes
        else format_console_report(report)
    )
    if arguments.output is not None and not arguments.json_only:
        print(f"JSON report: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
