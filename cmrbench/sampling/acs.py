"""Pure post-generation helpers for Cartesian ACS insertion."""

from __future__ import annotations

import numpy as np


def centered_strip_bounds(length: int, width: int) -> tuple[int, int]:
    """Return centered ``[start, stop)`` bounds for one array axis."""

    if isinstance(width, bool) or int(width) != width or int(width) < 1:
        raise ValueError("ACS width must be a positive integer")
    resolved_length = int(length)
    resolved_width = int(width)
    if resolved_width > resolved_length:
        raise ValueError(
            f"ACS width={resolved_width} exceeds axis length={resolved_length}"
        )
    start = (resolved_length - resolved_width) // 2
    return start, start + resolved_width


def add_centered_strip_acs(
    mask: np.ndarray,
    *,
    axis: int,
    width: int,
) -> np.ndarray:
    """Return a Boolean copy with a centered strip unioned along ``axis``."""

    binary = np.asarray(mask).astype(bool, copy=True)
    if binary.ndim == 0:
        raise ValueError("ACS insertion requires an array with at least one axis")
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += binary.ndim
    if resolved_axis < 0 or resolved_axis >= binary.ndim:
        raise ValueError(
            f"ACS axis={axis} is invalid for a {binary.ndim}-dimensional mask"
        )
    start, stop = centered_strip_bounds(binary.shape[resolved_axis], width)
    selection = [slice(None)] * binary.ndim
    selection[resolved_axis] = slice(start, stop)
    binary[tuple(selection)] = True
    return binary
