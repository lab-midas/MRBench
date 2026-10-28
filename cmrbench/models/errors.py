"""Failure categories shared by generic model adapters and orchestration."""

from __future__ import annotations


class UpstreamModelInferenceError(RuntimeError):
    """A physically prepared input reached, then failed in, the model runtime."""

    def __init__(self, model_name: str, error: Exception) -> None:
        self.model_name = str(model_name)
        self.upstream_exception_type = type(error).__name__
        self.upstream_exception_message = str(error)
        super().__init__(
            f"{self.model_name} upstream inference failed: "
            f"{self.upstream_exception_type}: {self.upstream_exception_message}"
        )
