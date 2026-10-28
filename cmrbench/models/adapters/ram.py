"""Production RAM adapter for independent cardiac cine slice-phase inference."""

from __future__ import annotations

import hashlib
from importlib import metadata
import inspect
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np

from cmrbench.models.adapters.ram_bridge import (
    complex_kspace_to_deepinv,
    deepinv_image_to_complex,
    mask_to_deepinv,
    sensitivity_maps_to_deepinv,
)
from cmrbench.models.base import ModelInfo, ReconstructionModel
from cmrbench.models.input import ReconstructionInput
from cmrbench.models.result import ReconstructionResult, TransformRecord


RAM_ADAPTER_VERSION = "2.0"
RAM_IMPLEMENTATIONS = ("official_github", "deepinv")
RAM_REPOSITORY = "https://github.com/matthieutrs/ram"
RAM_COMMIT = "0de1617403291e5a28179097795752200e65b82c"
CHECKPOINT_REPOSITORY = "mterris/ram"
CHECKPOINT_FILENAME = "ram.pth.tar"
CHECKPOINT_SHA256 = (
    "1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a"
)
MODEL_STATE_SHA256 = (
    "c6fbc876f752e861a2b5aa68abc83bd6d2832e5d01ced248f97ed41e5d6b0e18"
)
OFFICIAL_GITHUB_DEEPINV_VERSION = "0.3.0"
OFFICIAL_GITHUB_DEEPINV_COMMIT_BASELINE = (
    "dca7d9419eef7bd4bf23102719a47adb78e036d4"
)
DEEPINV_VERSION = "0.3.5"
DEEPINV_CONSTRUCTOR = "deepinv.models.RAM(pretrained=True)"
PHYSICAL_GAUSSIAN_SIGMA = 0.0
RAM_EFFECTIVE_SIGMA_FLOOR = 1e-3
RAM_NOISE_GAMMA_FALLBACK = 5e-3
DEEPINV_SIGMA_THRESHOLD = 5e-3
DEEPINV_GAIN_THRESHOLD = 1e-4
RAM_SPATIAL_MULTIPLE = 8
INFERENCE_MODE = "direct_pretrained_inference"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_ram_source_commit(source_root: Path) -> str:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"RAM source root does not exist: {root}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot verify RAM source commit at {root}: {exc}") from exc
    actual = completed.stdout.strip()
    if actual != RAM_COMMIT:
        raise RuntimeError(
            f"RAM checkout is {actual or 'UNKNOWN'}, but pinned commit "
            f"{RAM_COMMIT} is required"
        )
    return actual


def verify_checkpoint_sha256(path: Path) -> str:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"RAM checkpoint does not exist: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 is {actual}, expected {CHECKPOINT_SHA256}"
        )
    return actual


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def verify_model_state_sha256(model: Any) -> str:
    actual = _model_state_sha256(model)
    if actual != MODEL_STATE_SHA256:
        raise RuntimeError(
            f"RAM model-state SHA-256 is {actual}, expected {MODEL_STATE_SHA256}"
        )
    return actual


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _construct_deepinv_ram(dinv: Any, device: Any) -> Any:
    """Construct the selected public model without adding adapter-side options."""

    model = dinv.models.RAM(pretrained=True)
    model.to(device)
    model.eval()
    return model


class _OfficialGitHubRAMRuntime:
    """Lazy pinned-GitHub runtime kept out of normal CMRBench imports/tests."""

    def __init__(
        self,
        *,
        ram_source_root: Path | None,
        checkpoint: Path | None,
        device: str,
    ) -> None:
        self.ram_source_root = (
            Path(ram_source_root) if ram_source_root is not None else None
        )
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.device_requested = str(device)
        self.torch: Any = None
        self.dinv: Any = None
        self.model: Any = None
        self.device: Any = None
        self.provenance: dict[str, Any] = {}

    def _resolve_checkpoint(self) -> tuple[Path, str]:
        if self.checkpoint is not None:
            path = self.checkpoint.expanduser().resolve()
            source = "explicit local path"
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required to obtain the official RAM checkpoint"
                ) from exc
            path = Path(
                hf_hub_download(
                    repo_id=CHECKPOINT_REPOSITORY,
                    filename=CHECKPOINT_FILENAME,
                )
            ).resolve()
            source = "official Hugging Face cache/download"
        verify_checkpoint_sha256(path)
        return path, source

    def load(self) -> dict[str, Any]:
        if self.ram_source_root is None:
            raise RuntimeError(
                "implementation='official_github' requires ram_source_root"
            )
        actual_commit = verify_ram_source_commit(self.ram_source_root)
        source_root = self.ram_source_root.expanduser().resolve()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            import torch
            import deepinv as dinv
            from ram import RAM
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                f"dedicated RAM environment is incomplete or incompatible: {exc}"
            ) from exc

        deepinv_version = _distribution_version("deepinv")
        if deepinv_version != OFFICIAL_GITHUB_DEEPINV_VERSION:
            raise RuntimeError(
                f"DeepInv {deepinv_version} is installed; exact version "
                f"{OFFICIAL_GITHUB_DEEPINV_VERSION} is required for "
                "implementation='official_github'"
            )
        if not hasattr(dinv.physics, "MultiCoilMRI") or not hasattr(
            dinv.physics, "GaussianNoise"
        ):
            raise RuntimeError("incompatible DeepInv API: MRI physics classes are missing")
        constructor_parameters = inspect.signature(RAM.__init__).parameters
        if "pretrained" not in constructor_parameters or "device" not in constructor_parameters:
            raise RuntimeError(
                "incompatible RAM API: RAM(pretrained=..., device=...) is required"
            )
        source_file = Path(inspect.getfile(RAM)).resolve()
        if not source_file.is_relative_to(source_root):
            raise RuntimeError(
                f"imported RAM from {source_file}, outside pinned root {source_root}"
            )

        device = torch.device(self.device_requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        checkpoint_path, checkpoint_source = self._resolve_checkpoint()
        model = RAM(pretrained=False, device=self.device_requested)
        load_kwargs: dict[str, Any] = {"map_location": self.device_requested}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        state_dict = torch.load(checkpoint_path, **load_kwargs)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        if model.training:
            raise RuntimeError("RAM remained in training mode after eval()")
        model_state_sha256 = verify_model_state_sha256(model)

        self.torch = torch
        self.dinv = dinv
        self.model = model
        self.device = device
        self.provenance = {
            "model": "RAM",
            "implementation": "official_github",
            "source_repository": RAM_REPOSITORY,
            "source_commit": actual_commit,
            "ram_repository": RAM_REPOSITORY,
            "ram_commit_expected": RAM_COMMIT,
            "ram_commit_actual": actual_commit,
            "ram_source_root": str(source_root),
            "ram_source_file": str(source_file),
            "model_class": f"{RAM.__module__}.{RAM.__qualname__}",
            "constructor": (
                "pinned external ram.RAM(pretrained=False) + strict checkpoint load"
            ),
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_source": checkpoint_source,
            "checkpoint_sha256": verify_checkpoint_sha256(checkpoint_path),
            "model_state_sha256": model_state_sha256,
            "checkpoint_loaded_strict": True,
            "python": sys.version,
            "numpy": _distribution_version("numpy"),
            "torch": _distribution_version("torch"),
            "torchvision": _distribution_version("torchvision"),
            "deepinv_version": deepinv_version,
            "deepinv": deepinv_version,
            "deepinv_commit_baseline": OFFICIAL_GITHUB_DEEPINV_COMMIT_BASELINE,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }
        return dict(self.provenance)

    @staticmethod
    def _conditioning_record(
        physical_sigma: float, noise_gamma: Any, noise_model: Any
    ) -> dict[str, Any]:
        return {
            "ram_effective_sigma": max(
                physical_sigma, RAM_EFFECTIVE_SIGMA_FLOOR
            ),
            "ram_noise_gamma": max(
                _scalar(noise_gamma), RAM_EFFECTIVE_SIGMA_FLOOR
            ),
            "ram_noise_gamma_source": (
                "physics.noise_model.gain"
                if hasattr(noise_model, "gain")
                else "official RAM generic fallback"
            ),
            "conditioning_semantics": "pinned official GitHub RAM.forward",
        }

    def state_snapshot(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("RAM runtime has not been loaded")
        parameters = list(self.model.parameters())
        return {
            "state_sha256": _model_state_sha256(self.model),
            "parameter_versions": [
                int(getattr(parameter, "_version", -1)) for parameter in parameters
            ],
            "parameters_with_grad": sum(
                parameter.grad is not None for parameter in parameters
            ),
        }

    def verify_unchanged(self, before: dict[str, Any]) -> dict[str, Any]:
        after = self.state_snapshot()
        unchanged = (
            before["state_sha256"] == after["state_sha256"]
            and before["parameter_versions"] == after["parameter_versions"]
            and before["parameters_with_grad"] == 0
            and after["parameters_with_grad"] == 0
        )
        return {
            "state_sha256_before": before["state_sha256"],
            "state_sha256_after": after["state_sha256"],
            "parameter_version_counters_unchanged": (
                before["parameter_versions"] == after["parameter_versions"]
            ),
            "parameters_with_grad_before": before["parameters_with_grad"],
            "parameters_with_grad_after": after["parameters_with_grad"],
            "no_parameter_or_buffer_updates_verified": unchanged,
        }

    def infer_frame(
        self,
        measurement: np.ndarray,
        mask: np.ndarray,
        sensitivity_maps: np.ndarray,
        spatial_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("RAM runtime has not been loaded")
        torch = self.torch
        dinv = self.dinv
        measurement_t = torch.from_numpy(np.ascontiguousarray(measurement)).to(
            self.device
        )
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(self.device)
        maps_t = torch.from_numpy(np.ascontiguousarray(sensitivity_maps)).to(
            self.device
        )
        noise_model = dinv.physics.GaussianNoise(sigma=PHYSICAL_GAUSSIAN_SIGMA)
        physics = dinv.physics.MultiCoilMRI(
            mask=mask_t,
            coil_maps=maps_t,
            img_size=spatial_shape,
            noise_model=noise_model,
            device=self.device_requested,
        )
        physical_sigma = _scalar(physics.noise_model.sigma)
        noise_gamma = getattr(
            physics.noise_model, "gain", RAM_NOISE_GAMMA_FALLBACK
        )
        self.model.eval()
        if self.model.training:
            raise RuntimeError("RAM remained in training mode after eval()")
        start = perf_counter()
        with torch.inference_mode():
            output_t = self.model(measurement_t, physics=physics)
        runtime_seconds = perf_counter() - start
        output = output_t.detach().cpu().numpy()
        return output, {
            "runtime_seconds": runtime_seconds,
            "model_eval_mode": not self.model.training,
            "inference_context": "torch.inference_mode",
            "physical_gaussian_sigma": physical_sigma,
            **self._conditioning_record(
                physical_sigma, noise_gamma, physics.noise_model
            ),
            "operator": "deepinv.physics.MultiCoilMRI",
            "output_tensor_shape": list(output.shape),
            "output_tensor_dtype": str(output.dtype),
        }


class _DeepInvRAMRuntime(_OfficialGitHubRAMRuntime):
    """Lazy DeepInv 0.3.5 public RAM runtime with native implementation logic."""

    def load(self) -> dict[str, Any]:
        if self.ram_source_root is not None:
            raise RuntimeError(
                "implementation='deepinv' does not use ram_source_root; remove it "
                "to avoid ambiguous source provenance"
            )
        if self.checkpoint is not None:
            raise RuntimeError(
                "implementation='deepinv' must load its public pretrained artifact "
                "through deepinv.models.RAM(pretrained=True); checkpoint= is not accepted"
            )
        try:
            import torch
            import deepinv as dinv
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                f"dedicated DeepInv RAM environment is incomplete or incompatible: {exc}"
            ) from exc

        deepinv_version = _distribution_version("deepinv")
        if deepinv_version != DEEPINV_VERSION:
            raise RuntimeError(
                f"DeepInv {deepinv_version} is installed; exact version "
                f"{DEEPINV_VERSION} is required for implementation='deepinv'"
            )
        if not hasattr(dinv, "models") or not hasattr(dinv.models, "RAM"):
            raise RuntimeError("incompatible DeepInv API: deepinv.models.RAM is missing")
        if not hasattr(dinv.physics, "MultiCoilMRI") or not hasattr(
            dinv.physics, "GaussianNoise"
        ):
            raise RuntimeError("incompatible DeepInv API: MRI physics classes are missing")
        constructor_parameters = inspect.signature(dinv.models.RAM.__init__).parameters
        if "pretrained" not in constructor_parameters:
            raise RuntimeError(
                "incompatible DeepInv RAM API: RAM(pretrained=...) is required"
            )

        device = torch.device(self.device_requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        model = _construct_deepinv_ram(dinv, device)
        if model.training:
            raise RuntimeError("DeepInv RAM remained in training mode after eval()")

        checkpoint_path = (
            Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_FILENAME
        ).resolve()
        checkpoint_sha256 = verify_checkpoint_sha256(checkpoint_path)
        model_state_sha256 = verify_model_state_sha256(model)
        model_class = type(model)
        model_source_file = Path(inspect.getfile(model_class)).resolve()

        self.torch = torch
        self.dinv = dinv
        self.model = model
        self.device = device
        self.provenance = {
            "model": "RAM",
            "implementation": "deepinv",
            "deepinv_version": deepinv_version,
            "deepinv": deepinv_version,
            "model_class": "deepinv.models.RAM",
            "model_class_resolved": (
                f"{model_class.__module__}.{model_class.__qualname__}"
            ),
            "model_source_file": str(model_source_file),
            "constructor": DEEPINV_CONSTRUCTOR,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_source": "deepinv.models.RAM(pretrained=True)",
            "checkpoint_sha256": checkpoint_sha256,
            "model_state_sha256": model_state_sha256,
            "checkpoint_loaded_by_public_constructor": True,
            "python": sys.version,
            "numpy": _distribution_version("numpy"),
            "torch": _distribution_version("torch"),
            "torchvision": _distribution_version("torchvision"),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }
        return dict(self.provenance)

    @staticmethod
    def _conditioning_record(
        physical_sigma: float, noise_gamma: Any, noise_model: Any
    ) -> dict[str, Any]:
        del noise_gamma, noise_model
        return {
            "ram_effective_sigma": max(physical_sigma, DEEPINV_SIGMA_THRESHOLD),
            "ram_noise_gamma": DEEPINV_GAIN_THRESHOLD,
            "deepinv_sigma_threshold": DEEPINV_SIGMA_THRESHOLD,
            "deepinv_gain_threshold": DEEPINV_GAIN_THRESHOLD,
            "conditioning_semantics": "native deepinv.models.RAM 0.3.5",
        }


RuntimeFactory = Callable[..., Any]


class RAMAdapter(ReconstructionModel):
    """Explicit public RAM backend over independent canonical cine frames."""

    def __init__(
        self,
        *,
        implementation: str | None = None,
        ram_source_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cuda",
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        if implementation not in RAM_IMPLEMENTATIONS:
            supported = ", ".join(RAM_IMPLEMENTATIONS)
            if implementation is None:
                raise ValueError(
                    "RAM implementation must be selected explicitly; choose one of: "
                    f"{supported}"
                )
            raise ValueError(
                f"unsupported RAM implementation {implementation!r}; choose one of: "
                f"{supported}"
            )
        self.implementation = implementation
        self.ram_source_root = (
            Path(ram_source_root) if ram_source_root is not None else None
        )
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        if self.implementation == "official_github" and self.ram_source_root is None:
            raise ValueError(
                "implementation='official_github' requires ram_source_root"
            )
        if self.implementation == "deepinv" and self.ram_source_root is not None:
            raise ValueError(
                "implementation='deepinv' does not accept ram_source_root"
            )
        if self.implementation == "deepinv" and self.checkpoint is not None:
            raise ValueError(
                "implementation='deepinv' loads its checkpoint through "
                "deepinv.models.RAM(pretrained=True); checkpoint= is not accepted"
            )
        self.device = str(device)
        default_runtime_factories = {
            "official_github": _OfficialGitHubRAMRuntime,
            "deepinv": _DeepInvRAMRuntime,
        }
        self._runtime_factory = (
            runtime_factory or default_runtime_factories[self.implementation]
        )
        self._runtime: Any = None
        self._load_provenance: dict[str, Any] = {}

    def load(self) -> None:
        if self._runtime is not None:
            return
        runtime = self._runtime_factory(
            ram_source_root=self.ram_source_root,
            checkpoint=self.checkpoint,
            device=self.device,
        )
        provenance = runtime.load()
        if not isinstance(provenance, dict):
            raise RuntimeError("RAM runtime load() must return provenance metadata")
        runtime_implementation = provenance.get("implementation")
        if runtime_implementation is not None and runtime_implementation != self.implementation:
            raise RuntimeError(
                "RAM runtime implementation provenance disagrees with the explicit "
                f"adapter selection: {runtime_implementation!r} != {self.implementation!r}"
            )
        self._runtime = runtime
        self._load_provenance = dict(provenance)

    def _implementation_provenance(self) -> dict[str, Any]:
        common = {
            "model": "RAM",
            "implementation": self.implementation,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_state_sha256": MODEL_STATE_SHA256,
        }
        if self.implementation == "official_github":
            return {
                **common,
                "source_repository": RAM_REPOSITORY,
                "source_commit": RAM_COMMIT,
                "deepinv_version": OFFICIAL_GITHUB_DEEPINV_VERSION,
                "constructor": "pinned external ram.RAM(pretrained=False) + strict checkpoint load",
            }
        return {
            **common,
            "deepinv_version": DEEPINV_VERSION,
            "model_class": "deepinv.models.RAM",
            "constructor": DEEPINV_CONSTRUCTOR,
        }

    @staticmethod
    def _validate_input(
        input: ReconstructionInput,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kspace = np.asarray(input.kspace)
        mask = np.asarray(input.mask)
        maps = input.sensitivity_maps
        if not np.iscomplexobj(kspace):
            raise ValueError("RAMAdapter requires complex canonical k-space")
        if not (
            np.all(np.isfinite(kspace.real)) and np.all(np.isfinite(kspace.imag))
        ):
            raise ValueError("RAMAdapter k-space contains NaN or Inf")
        if maps is None:
            raise ValueError("RAMAdapter requires trusted sensitivity_maps")
        sensitivity_maps = np.asarray(maps)
        if sensitivity_maps.ndim != 5 or not np.iscomplexobj(sensitivity_maps):
            raise ValueError(
                "sensitivity_maps must be complex [S,1,X,Y,C] or [S,T,X,Y,C]"
            )
        slices, phases, kx, ky, coils = kspace.shape
        if sensitivity_maps.shape[0] != slices:
            raise ValueError("sensitivity-map slice dimension does not match k-space")
        if sensitivity_maps.shape[1] not in (1, phases):
            raise ValueError("sensitivity-map time dimension must be 1 or match T")
        if sensitivity_maps.shape[2:] != (kx, ky, coils):
            raise ValueError("sensitivity-map X/Y/coil dimensions do not match k-space")
        if not (
            np.all(np.isfinite(sensitivity_maps.real))
            and np.all(np.isfinite(sensitivity_maps.imag))
        ):
            raise ValueError("RAMAdapter sensitivity maps contain NaN or Inf")
        if not np.all(np.isfinite(mask)):
            raise ValueError("RAMAdapter mask contains NaN or Inf")
        if not np.all(np.logical_or(mask == 0, mask == 1)):
            raise ValueError("RAMAdapter mask must be binary with 1=sampled")
        return kspace, mask.astype(bool, copy=False), sensitivity_maps

    def _padding_record(self, height: int, width: int) -> dict[str, Any]:
        return {
            "performed_by": (
                "pinned official GitHub RAM.forward"
                if self.implementation == "official_github"
                else "DeepInv 0.3.5 RAM.forward"
            ),
            "implementation": self.implementation,
            "spatial_multiple": RAM_SPATIAL_MULTIPLE,
            "padding_location": "top/left",
            "padding_top": (-height) % RAM_SPATIAL_MULTIPLE,
            "padding_left": (-width) % RAM_SPATIAL_MULTIPLE,
            "input_shape": [height, width],
            "output_restored_to_input_shape": True,
        }

    def reconstruct(self, input: ReconstructionInput) -> ReconstructionResult:
        if self._runtime is None:
            self.load()
        kspace, mask, sensitivity_maps = self._validate_input(input)
        slices, phases, height, width, coils = kspace.shape
        source_slice_origin = input.metadata.get("source_slice")
        source_phase_origin = input.metadata.get("source_phase")
        source_map_time_origin = input.metadata.get(
            "source_sensitivity_map_time_index"
        )
        output = np.empty((slices, phases, height, width), dtype=np.complex64)
        frame_records: list[dict[str, Any]] = []
        before = self._runtime.state_snapshot()
        total_start = perf_counter()
        for slice_index in range(slices):
            for phase_index in range(phases):
                map_time_index = 0 if sensitivity_maps.shape[1] == 1 else phase_index
                measurement_np = complex_kspace_to_deepinv(
                    kspace[slice_index, phase_index]
                )
                mask_np = mask_to_deepinv(mask[slice_index, phase_index])
                maps_np = sensitivity_maps_to_deepinv(
                    sensitivity_maps[slice_index, map_time_index]
                )
                output_np, runtime_record = self._runtime.infer_frame(
                    measurement_np,
                    mask_np,
                    maps_np,
                    (height, width),
                )
                frame = deepinv_image_to_complex(output_np)
                if frame.shape != (height, width):
                    raise RuntimeError(
                        "RAM output did not restore the original spatial geometry: "
                        f"received {frame.shape}, expected {(height, width)}"
                    )
                if not (
                    np.all(np.isfinite(frame.real))
                    and np.all(np.isfinite(frame.imag))
                ):
                    raise RuntimeError("RAM reconstruction contains NaN or Inf")
                output[slice_index, phase_index] = frame
                frame_records.append(
                    {
                        "slice": slice_index,
                        "phase": phase_index,
                        "local_slice": slice_index,
                        "local_phase": phase_index,
                        "source_slice": (
                            int(source_slice_origin)
                            if slices == 1
                            and isinstance(source_slice_origin, (int, np.integer))
                            else slice_index
                        ),
                        "source_phase": (
                            int(source_phase_origin)
                            if phases == 1
                            and isinstance(source_phase_origin, (int, np.integer))
                            else phase_index
                        ),
                        "flat_batch_index": slice_index * phases + phase_index,
                        "sensitivity_map_time_index": map_time_index,
                        "local_sensitivity_map_time_index": map_time_index,
                        "source_sensitivity_map_time_index": (
                            int(source_map_time_origin)
                            if sensitivity_maps.shape[1] == 1
                            and isinstance(source_map_time_origin, (int, np.integer))
                            else map_time_index
                        ),
                        "index_semantics": (
                            "local indices address this ReconstructionInput; source "
                            "indices identify the parent cine when supplied in input metadata"
                        ),
                        **runtime_record,
                    }
                )
        runtime_seconds = perf_counter() - total_start
        state_verification = self._runtime.verify_unchanged(before)
        if not state_verification.get(
            "no_parameter_or_buffer_updates_verified", False
        ):
            raise RuntimeError("RAM parameters or buffers changed during reconstruction")

        transforms = [
            TransformRecord(
                "canonical_input_validation",
                {
                    "kspace_shape": list(kspace.shape),
                    "mask_shape": list(mask.shape),
                    "sensitivity_maps_shape": list(sensitivity_maps.shape),
                    "complex_measurement_preserved": True,
                },
            ),
            TransformRecord(
                "flatten_slice_phase_to_batch",
                {
                    "slice_count": slices,
                    "phase_count": phases,
                    "order": "b = s*T + t",
                    "independent_inference_batch_size": 1,
                    "temporal_modeling": False,
                    "source_index_metadata": {
                        "source_slice": source_slice_origin,
                        "source_phase": source_phase_origin,
                        "source_sensitivity_map_time_index": source_map_time_origin,
                    },
                },
            ),
            TransformRecord(
                "permute_kspace_coil_last_to_first",
                {
                    "canonical_frame_shape": [height, width, coils],
                    "coil_first_shape": [coils, height, width],
                    "axis_swap_or_transpose": False,
                },
            ),
            TransformRecord(
                "complex_kspace_to_two_channel_real",
                {
                    "frame_output_shape": [1, 2, coils, height, width],
                    "dtype": "float32",
                    "channel_order": ["real", "imaginary"],
                },
            ),
            TransformRecord(
                "expand_mask_to_complex_channels",
                {
                    "canonical_frame_shape": [height, width],
                    "frame_output_shape": [1, 2, height, width],
                    "identical_real_imaginary_channels": True,
                    "sampled_value": 1,
                },
            ),
            TransformRecord(
                "broadcast_and_permute_sensitivity_maps",
                {
                    "source_time_dimension": int(sensitivity_maps.shape[1]),
                    "broadcast_over_phases": sensitivity_maps.shape[1] == 1,
                    "frame_output_shape": [1, coils, height, width],
                    "dtype": "complex64",
                    "normalization": None,
                    "conjugation": None,
                },
            ),
            TransformRecord(
                "ram_deepinv_operator_configuration",
                {
                    "implementation": self.implementation,
                    "operator": "deepinv.physics.MultiCoilMRI",
                    "physical_gaussian_sigma": PHYSICAL_GAUSSIAN_SIGMA,
                    "added_synthetic_noise": False,
                    "fft": "centered orthonormal",
                    "measurement_rescaling": None,
                    "sensitivity_map_normalization": None,
                },
            ),
            TransformRecord(
                "device_and_precision_transfer",
                {
                    "device": self.device,
                    "measurement_dtype": "float32",
                    "mask_dtype": "float32",
                    "sensitivity_map_dtype": "complex64",
                },
            ),
            TransformRecord(
                "ram_internal_spatial_padding",
                self._padding_record(height, width),
            ),
            TransformRecord(
                "two_channel_real_to_complex",
                {
                    "frame_input_shape": [1, 2, height, width],
                    "frame_output_shape": [height, width],
                    "dtype": "complex64",
                    "magnitude_conversion": False,
                },
            ),
            TransformRecord(
                "unflatten_batch_to_slice_phase",
                {
                    "order": "b = s*T + t",
                    "final_shape": [slices, phases, height, width],
                },
            ),
        ]
        return ReconstructionResult(
            subject_id=input.subject_id,
            reconstruction=output,
            model_name="ram",
            runtime_seconds=runtime_seconds,
            transforms=transforms,
            metadata={
                "adapter_version": RAM_ADAPTER_VERSION,
                "model": "RAM",
                "implementation": self.implementation,
                "implementation_provenance": self._implementation_provenance(),
                "inference_mode": INFERENCE_MODE,
                "direct_pretrained_checkpoint": True,
                "adaptation_performed": False,
                "fine_tuning_called": False,
                "optimizer_created": False,
                "backward_called": False,
                "model_eval_mode": True,
                "inference_context": "torch.inference_mode",
                "physical_gaussian_sigma": PHYSICAL_GAUSSIAN_SIGMA,
                "physical_sigma_interpretation": (
                    "no additional synthetic noise added during retrospective "
                    "undersampling; not a claim that acquired MRI is noiseless"
                ),
                "conditioning_protocol": (
                    {
                        "effective_sigma_floor": RAM_EFFECTIVE_SIGMA_FLOOR,
                        "noise_gamma_fallback": RAM_NOISE_GAMMA_FALLBACK,
                    }
                    if self.implementation == "official_github"
                    else {
                        "sigma_threshold": DEEPINV_SIGMA_THRESHOLD,
                        "gain_threshold": DEEPINV_GAIN_THRESHOLD,
                        "internal_realigning_and_scaling": (
                            "native DeepInv RAM 0.3.5 implementation"
                        ),
                    }
                ),
                "ram_effective_sigma_floor": (
                    RAM_EFFECTIVE_SIGMA_FLOOR
                    if self.implementation == "official_github"
                    else None
                ),
                "ram_noise_gamma_fallback": (
                    RAM_NOISE_GAMMA_FALLBACK
                    if self.implementation == "official_github"
                    else None
                ),
                "deepinv_sigma_threshold": (
                    DEEPINV_SIGMA_THRESHOLD
                    if self.implementation == "deepinv"
                    else None
                ),
                "deepinv_gain_threshold": (
                    DEEPINV_GAIN_THRESHOLD
                    if self.implementation == "deepinv"
                    else None
                ),
                "external_intensity_normalization": None,
                "measurement_rescaling": None,
                "measurement_rescaling_scope": (
                    "CMRBench adapter only; selected implementation internals are preserved"
                ),
                "implementation_internal_realigning_and_scaling": (
                    "pinned official GitHub RAM semantics"
                    if self.implementation == "official_github"
                    else "native DeepInv RAM 0.3.5 semantics"
                ),
                "sensitivity_map_normalization": None,
                "runtime": dict(self._load_provenance),
                "frame_inference": frame_records,
                "state_verification": state_verification,
                "warnings": [
                    "RAM reconstructs each cine phase independently and has no temporal prior.",
                    "Direct pretrained cardiac use is a documented domain-shift condition.",
                    (
                        "RAM implementation-specific conditioning is preserved and "
                        "must not be interpreted as benchmark-side measurement scaling."
                    ),
                ],
            },
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="ram",
            model_version=RAM_ADAPTER_VERSION,
            checkpoint=f"{CHECKPOINT_REPOSITORY}/{CHECKPOINT_FILENAME}",
            checkpoint_identifier=CHECKPOINT_SHA256,
            git_commit=(
                RAM_COMMIT if self.implementation == "official_github" else None
            ),
            parameter_count=self._load_provenance.get("parameter_count"),
            device=self.device,
            precision="float32/complex64",
            metadata={
                "model": "RAM",
                "implementation": self.implementation,
                "implementation_provenance": self._implementation_provenance(),
                "repository": (
                    RAM_REPOSITORY
                    if self.implementation == "official_github"
                    else None
                ),
                "inference_mode": INFERENCE_MODE,
                "adaptation_performed": False,
                "deepinv_version": (
                    OFFICIAL_GITHUB_DEEPINV_VERSION
                    if self.implementation == "official_github"
                    else DEEPINV_VERSION
                ),
                "physical_gaussian_sigma": PHYSICAL_GAUSSIAN_SIGMA,
                "external_runtime_required": True,
            },
        )


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_SHA256",
    "DEEPINV_CONSTRUCTOR",
    "DEEPINV_GAIN_THRESHOLD",
    "DEEPINV_SIGMA_THRESHOLD",
    "DEEPINV_VERSION",
    "INFERENCE_MODE",
    "MODEL_STATE_SHA256",
    "OFFICIAL_GITHUB_DEEPINV_VERSION",
    "RAMAdapter",
    "RAM_ADAPTER_VERSION",
    "RAM_COMMIT",
    "RAM_IMPLEMENTATIONS",
    "RAM_REPOSITORY",
    "verify_checkpoint_sha256",
    "verify_model_state_sha256",
    "verify_ram_source_commit",
]
