"""Validated interpreter selection for cross-runtime CMRBench orchestration."""

from __future__ import annotations

import os
import sys

from cmrbench.experiment.config import ExperimentConfig


RAM_OFFICIAL_PYTHON = os.environ.get("CMRBENCH_RAM_OFFICIAL_PYTHON", sys.executable)
RAM_DEEPINV_PYTHON = os.environ.get("CMRBENCH_RAM_DEEPINV_PYTHON", sys.executable)
SDUM_PYTHON = os.environ.get("CMRBENCH_SDUM_PYTHON", sys.executable)
CARDIOMM_PYTHON = os.environ.get("CMRBENCH_CARDIOMM_PYTHON", sys.executable)
CMRBENCH_EVALUATION_PYTHON = os.environ.get("CMRBENCH_EVALUATION_PYTHON", sys.executable)


def reconstruction_runtime(config: ExperimentConfig) -> str:
    """Return the validated model runtime without consulting sampling fields."""

    model = config.model.name
    if model == "ram":
        implementation = config.model.parameters.get(
            "implementation", "official_github"
        )
        if implementation == "official_github":
            expected = RAM_OFFICIAL_PYTHON
        elif implementation == "deepinv":
            expected = RAM_DEEPINV_PYTHON
        else:
            raise ValueError(
                "unknown RAM runtime combination: "
                f"implementation={implementation!r}; expected "
                "'official_github' or 'deepinv'"
            )
    elif model == "sdum":
        expected = SDUM_PYTHON
    elif model == "cardiomm":
        expected = CARDIOMM_PYTHON
    else:
        raise ValueError(f"no validated reconstruction runtime for model {model!r}")

    configured = config.model.environment
    if configured is not None and configured != expected:
        raise ValueError(
            f"model.environment {configured!r} disagrees with validated runtime "
            f"{expected!r} for model {model!r}"
        )
    return expected


def evaluation_runtime() -> str:
    """Return the one validated model-independent evaluation interpreter."""

    return CMRBENCH_EVALUATION_PYTHON
