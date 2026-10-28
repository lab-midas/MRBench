"""Display-only fixed-window cine GIF for a candidate reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


GIF_PROTOCOL = "cmrbench_mid_slice_reconstruction_gif_fixed_max_v1"


def prepare_mid_slice_gif_frames(
    reconstruction: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map ``abs(candidate[S//2, 0:T])`` through one fixed linear window."""

    candidate = np.asarray(reconstruction)
    if candidate.ndim != 4:
        raise ValueError("GIF reconstruction must use canonical [S,T,X,Y] shape")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("GIF reconstruction contains NaN or Inf")
    slices, phases = candidate.shape[:2]
    if slices < 1 or phases < 1:
        raise ValueError("GIF reconstruction requires non-empty S and T dimensions")
    slice_index = slices // 2
    magnitude = np.abs(candidate[slice_index]).astype(np.float64, copy=False)
    display_max = float(np.max(magnitude))
    if not np.isfinite(display_max) or display_max <= 0.0:
        raise ValueError("GIF fixed temporal display maximum must be positive")
    scaled = np.clip(magnitude / display_max, 0.0, 1.0)
    frames = np.rint(255.0 * scaled).astype(np.uint8)
    return frames, {
        "protocol_identifier": GIF_PROTOCOL,
        "source": "abs(reconstruction[S//2, 0:T])",
        "slice_index": slice_index,
        "phase_indices": list(range(phases)),
        "phase_indexing": "zero_based",
        "frame_count": phases,
        "display_scaling": "one fixed linear window over complete mid-slice T,X,Y",
        "display_window": [0.0, display_max],
        "per_frame_autoscaling": False,
        "scientific_array_modified": False,
        "metrics_affected": False,
    }


def save_mid_slice_reconstruction_gif(
    path: Path,
    *,
    reconstruction: np.ndarray,
    duration_ms: int = 120,
) -> dict[str, Any]:
    """Write all prepared temporal frames using Pillow's GIF encoder."""

    if duration_ms <= 0:
        raise ValueError("GIF duration_ms must be positive")
    frames, metadata = prepare_mid_slice_gif_frames(reconstruction)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "mid-slice reconstruction GIF requires the existing Pillow dependency"
        ) from exc
    images = [Image.fromarray(frame) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return {
        **metadata,
        "path": str(path.resolve()),
        "duration_ms": duration_ms,
        "loop": 0,
        "format": "GIF",
    }
