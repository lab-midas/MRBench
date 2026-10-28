"""Native NumPy implementation of the VISTA sampling algorithm.

Algorithm: Ahmad et al., Magnetic Resonance in Medicine 74:1266-1278 (2015).
Executable reference: OSU-CMR/VISTA, https://github.com/OSU-CMR/VISTA,
commit 1ad7ea0130247aca220a3e761fc8a64b770900b2.
This is a CMRBench Python adaptation. See ``NOTICE.md`` in this directory for
the upstream copyright, permission, warranty, and provenance notices.

The implementation uses centered coordinates for the optimization:

* phase coordinates are ``-floor(p/2), ..., ceil(p/2)-1``;
* time coordinates are ``-floor(t/2), ..., ceil(t/2)-1``;
* array indices are zero based and are formed only when a mask is emitted.

Keeping coordinates separate from array indices avoids translating MATLAB
linear-index expressions throughout the algorithm.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray


_EPS: Final[float] = float(np.spacing(1.0))


def _matlab_round(values: float | NDArray[np.floating]) -> int | NDArray[np.int64]:
    """Round halves away from zero, matching MATLAB rather than NumPy."""

    array = np.asarray(values, dtype=np.float64)
    rounded = np.copysign(np.floor(np.abs(array) + 0.5), array).astype(np.int64)
    if array.ndim == 0:
        return int(rounded)
    return rounded


def _centered_axis(length: int) -> NDArray[np.int64]:
    return np.arange(-(length // 2), (length + 1) // 2, dtype=np.int64)


def _wrap_centered_grid(values: NDArray[np.int64], length: int) -> NDArray[np.int64]:
    """Wrap integer coordinates onto the centered periodic grid."""

    lower = -(length // 2)
    return ((values - lower) % length + lower).astype(np.int64, copy=False)


def _weighted_random_indices(
    probabilities: NDArray[np.float64], seed: int, count: int
) -> NDArray[np.int64]:
    """Equivalent role to upstream ``randp.m`` (sampling with replacement)."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(probabilities))
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if np.any(probabilities < 0.0) or not np.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities must be finite, non-negative, and not all zero")
    draws = np.random.RandomState(seed).random_sample(count)
    cdf = np.cumsum(probabilities, dtype=np.float64) / total
    cdf[-1] = 1.0
    return np.searchsorted(cdf, draws, side="right").astype(np.int64)


def _initial_vrs_points(
    p: int,
    t: int,
    lines_per_frame: int,
    alpha: float,
    sigma: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create the variable-density random initialization used by VISTA."""

    phase_axis = _centered_axis(p)
    probabilities = 0.1 + alpha / (1.0 - alpha + 1e-10) * np.exp(
        -(phase_axis.astype(np.float64) ** 2) / (sigma**2)
    )
    seed_rng = np.random.RandomState(seed)
    frame_seeds = _matlab_round(1e6 * seed_rng.random_sample(t))
    assert isinstance(frame_seeds, np.ndarray)

    phase = np.empty(lines_per_frame * t, dtype=np.float64)
    time = np.repeat(_centered_axis(t), lines_per_frame).astype(np.float64)
    for frame_index, frame_seed in enumerate(frame_seeds):
        start = frame_index * lines_per_frame
        stop = start + lines_per_frame
        chosen = _weighted_random_indices(
            probabilities, int(frame_seed), lines_per_frame
        )
        phase[start:stop] = phase_axis[chosen]
    return phase, time


def _tile_points(
    phase: NDArray[np.float64], time: NDArray[np.float64], p: int, t: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the exact nine periodic tiles used by the MATLAB reference."""

    tiled_phase = np.concatenate(
        (
            phase,
            phase - p,
            phase,
            phase + p,
            phase - p,
            phase + p,
            phase - p,
            phase,
            phase + p,
        )
    )
    tiled_time = np.concatenate(
        (
            time,
            time - t,
            time - t,
            time - t,
            time,
            time,
            time + t,
            time + t,
            time + t,
        )
    )
    return tiled_phase, tiled_time


def _relocate_duplicates(
    phase: NDArray[np.float64], time: NDArray[np.float64], p: int, t: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Move duplicate points to the nearest empty PE location in their frame."""

    phase_grid = _wrap_centered_grid(
        np.asarray(_matlab_round(phase), dtype=np.int64), p
    )
    time_grid = _wrap_centered_grid(
        np.asarray(_matlab_round(time), dtype=np.int64), t
    )
    phase_indices = phase_grid + p // 2
    time_indices = time_grid + t // 2
    linear = time_indices * p + phase_indices

    duplicate_indices: list[int] = []
    unique_linear, counts = np.unique(linear, return_counts=True)
    for repeated in unique_linear[counts != 1]:
        duplicate_indices.extend(np.flatnonzero(linear == repeated)[1:].tolist())

    occupied = np.zeros(p * t, dtype=bool)
    occupied[np.unique(linear)] = True
    for duplicate_index in duplicate_indices:
        frame_index = int(linear[duplicate_index] // p)
        frame_start = frame_index * p
        empty_phase_indices = np.flatnonzero(
            ~occupied[frame_start : frame_start + p]
        )
        if empty_phase_indices.size == 0:
            raise RuntimeError("cannot relocate a duplicate: its frame is full")
        old_phase_index = int(linear[duplicate_index] % p)
        distance = np.abs(empty_phase_indices - old_phase_index)
        new_phase_index = int(empty_phase_indices[int(np.argmin(distance))])
        new_linear = frame_start + new_phase_index
        linear[duplicate_index] = new_linear
        occupied[new_linear] = True

    phase_grid = (linear % p) - p // 2
    time_grid = (linear // p) - t // 2
    return phase_grid.astype(np.float64), time_grid.astype(np.float64)


def _charges(
    phase: NDArray[np.float64], alpha: float, sigma: float
) -> NDArray[np.float64]:
    return 1.0 - alpha * np.exp(-(phase**2) / (2.0 * sigma**2))


def _compute_energy(
    phase: NDArray[np.float64],
    time: NDArray[np.float64],
    *,
    alpha: float,
    sigma: float,
    temporal_scale: float,
    potential_exponent: float,
) -> float:
    """Return the scalar Riesz energy used by upstream ``computeU.m``."""

    phase = np.asarray(phase, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    charge = _charges(phase, alpha, sigma)
    energy = 0.0
    for point in range(phase.size - 1):
        delta_phase = phase[point] - phase[point + 1 :]
        delta_time = temporal_scale * (time[point] - time[point + 1 :])
        distance_power = (delta_phase**2 + delta_time**2) ** (
            potential_exponent / 2.0
        )
        energy += float(
            np.sum(charge[point] * charge[point + 1 :] / distance_power)
        )
    return float(energy)


def _relocation_energy_delta(
    phase: NDArray[np.float64],
    time: NDArray[np.float64],
    point: int,
    new_phase: int,
    *,
    alpha: float,
    sigma: float,
    temporal_scale: float,
    potential_exponent: float,
) -> float:
    """Energy change for one fillK move, equivalent to recomputing all U."""

    keep = np.arange(phase.size) != point
    other_phase = phase[keep]
    other_time = time[keep]
    other_charge = _charges(other_phase, alpha, sigma)
    old_charge = float(_charges(phase[point : point + 1], alpha, sigma)[0])
    new_charge = float(
        _charges(np.asarray([new_phase], dtype=np.float64), alpha, sigma)[0]
    )
    time_delta = temporal_scale * (time[point] - other_time)
    old_distance = ((phase[point] - other_phase) ** 2 + time_delta**2) ** (
        potential_exponent / 2.0
    )
    new_distance = ((new_phase - other_phase) ** 2 + time_delta**2) ** (
        potential_exponent / 2.0
    )
    old_interaction = np.sum(old_charge * other_charge / old_distance)
    new_interaction = np.sum(new_charge * other_charge / new_distance)
    return float(new_interaction - old_interaction)


def _exclude_outer_holes(
    holes: NDArray[np.int64], p: int
) -> NDArray[np.int64]:
    """Apply the exact edge exception from upstream ``excludeOuter``."""

    remaining = np.sort(np.asarray(holes, dtype=np.int64))[::-1]
    if remaining.size == 0:
        return remaining
    gaps = np.abs(np.diff(np.concatenate(([math.ceil(p / 2)], remaining))))
    if np.max(gaps) == 1:
        return np.empty(0, dtype=np.int64)
    first_gap = int(np.flatnonzero(gaps > 1)[0])
    remaining = remaining[first_gap:]

    remaining = np.sort(remaining)
    if remaining.size == 0:
        return remaining
    gaps = np.abs(
        np.diff(np.concatenate(([-math.floor(p / 2) - 1], remaining)))
    )
    if np.max(gaps) == 1:
        return np.empty(0, dtype=np.int64)
    first_gap = int(np.flatnonzero(gaps > 1)[0])
    return remaining[first_gap:]


def _center_out_holes(
    sampled_phase: NDArray[np.float64], p: int, fraction: float
) -> NDArray[np.int64]:
    start = -math.floor(fraction * p / 2.0)
    stop = math.ceil(fraction * p / 2.0)
    target = np.arange(start, stop, dtype=np.int64)
    holes = np.setdiff1d(target, np.unique(sampled_phase.astype(np.int64)))
    if holes.size == 0:
        return holes
    order = np.argsort(np.abs(holes), kind="stable")
    return holes[order]


def _fill_time_average(
    subset_indices: NDArray[np.int64],
    phase: NDArray[np.float64],
    time: NDArray[np.float64],
    *,
    p: int,
    alpha: float,
    sigma: float,
    temporal_scale: float,
    potential_exponent: float,
    fraction: float,
) -> None:
    """Relocate boundary-side points until only allowed outer holes remain."""

    subset_phase = phase[subset_indices].copy()
    holes = _center_out_holes(subset_phase, p, fraction)
    while holes.size:
        hole = int(holes[0])
        direction = 1.0 if hole + _EPS > 0.0 else -1.0
        candidates = np.flatnonzero(direction * subset_phase > direction * hole)
        if candidates.size == 0:
            break
        signed_distances = direction * (subset_phase[candidates] - hole)
        boundary_distance = float(np.min(signed_distances))
        candidates = candidates[signed_distances == boundary_distance]

        selected_local = int(candidates[0])
        selected_delta = math.inf
        for local_index in candidates:
            global_index = int(subset_indices[int(local_index)])
            delta = _relocation_energy_delta(
                phase,
                time,
                global_index,
                hole,
                alpha=alpha,
                sigma=sigma,
                temporal_scale=temporal_scale,
                potential_exponent=potential_exponent,
            )
            # MATLAB uses <=, so the final candidate wins an exact tie.
            if delta <= selected_delta:
                selected_delta = delta
                selected_local = int(local_index)

        global_index = int(subset_indices[selected_local])
        phase[global_index] = hole
        subset_phase[selected_local] = hole
        remaining_holes = _exclude_outer_holes(
            _center_out_holes(subset_phase, p, fraction), p
        )
        if remaining_holes.size:
            center_order = np.argsort(np.abs(remaining_holes), kind="stable")
            holes = remaining_holes[center_order]
        else:
            holes = remaining_holes


def _validate_inputs(
    p: int,
    t: int,
    acceleration: float,
    seed: int,
    alpha: float,
    sigma: float,
    n_iter: int,
    grid_interval: int,
    uniform_iteration: int,
    fully_sampled_iteration: int,
    fully_sampled_time_average: int,
    fully_sampled_fraction: float,
    potential_exponent: float,
    temporal_scale: float,
    temporal_step: float,
) -> None:
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 2:
        raise ValueError("p must be an integer greater than one")
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 2:
        raise ValueError("t must be an integer greater than one")
    if not np.isfinite(acceleration) or acceleration < 1.0 or acceleration > p:
        raise ValueError("acceleration must be finite and between 1 and p")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not isinstance(n_iter, (int, np.integer)) or not 20 <= n_iter <= 1024:
        raise ValueError("n_iter must be an integer between 20 and 1024")
    for name, value in (
        ("grid_interval", grid_interval),
        ("uniform_iteration", uniform_iteration),
        ("fully_sampled_iteration", fully_sampled_iteration),
    ):
        if not isinstance(value, (int, np.integer)) or not 1 <= value <= n_iter:
            raise ValueError(f"{name} must be an integer in [1, n_iter]")
    if (
        isinstance(fully_sampled_time_average, bool)
        or not isinstance(fully_sampled_time_average, (int, np.integer))
        or fully_sampled_time_average < 0
    ):
        raise ValueError("fully_sampled_time_average must be a non-negative integer")
    if (
        fully_sampled_time_average not in (0, 1)
        and fully_sampled_time_average < acceleration
    ):
        raise ValueError(
            "values above one for fully_sampled_time_average must be at least R"
        )
    if not 0.0 < fully_sampled_fraction <= 1.0:
        raise ValueError("fully_sampled_fraction must be in (0, 1]")
    if not np.isfinite(potential_exponent) or potential_exponent <= 0.0:
        raise ValueError("potential_exponent must be finite and positive")
    if not np.isfinite(temporal_scale) or temporal_scale <= 0.0:
        raise ValueError("temporal_scale must be finite and positive")
    if temporal_step != 0.0:
        raise ValueError(
            "temporal_step must be zero: this generator guarantees constant "
            "temporal resolution"
        )


def generate_vista_mask(
    p: int,
    t: int,
    acceleration: float,
    seed: int,
    *,
    alpha: float = 0.28,
    sigma: float | None = None,
    n_iter: int = 120,
    grid_interval: int | None = None,
    uniform_iteration: int | None = None,
    step_size: float = 0.25,
    fully_sampled_iteration: int | None = None,
    fully_sampled_time_average: int = 1,
    fully_sampled_fraction: float = 1.0,
    potential_exponent: float = 1.4,
    temporal_scale: float | None = None,
    temporal_step: float = 0.0,
) -> NDArray[np.bool_]:
    """Generate a native ``[phase, time]`` VISTA mask.

    Defaults follow the released OSU-CMR MATLAB implementation where it
    differs from the paper. The generator has no ACS concept, no hidden global
    state, and no import-time work.

    Point updates within every iteration are sequential (Gauss-Seidel style),
    as in ``VISTA.m``. Vectorization is limited to the interaction calculation
    for the point currently being updated.
    """

    p = int(p) if isinstance(p, np.integer) else p
    t = int(t) if isinstance(t, np.integer) else t
    seed = int(seed) if isinstance(seed, np.integer) else seed
    acceleration = float(acceleration)
    sigma = p / 5.0 if sigma is None else float(sigma)
    grid_interval = (
        math.floor(n_iter / 6) if grid_interval is None else grid_interval
    )
    uniform_iteration = (
        int(_matlab_round(n_iter / 2.0))
        if uniform_iteration is None
        else uniform_iteration
    )
    fully_sampled_iteration = (
        int(_matlab_round(5.0 * n_iter / 6.0))
        if fully_sampled_iteration is None
        else fully_sampled_iteration
    )
    temporal_scale = (
        max(acceleration / 10.0 + 0.25, 1.0)
        if temporal_scale is None
        else float(temporal_scale)
    )
    _validate_inputs(
        p,
        t,
        acceleration,
        seed,
        alpha,
        sigma,
        n_iter,
        grid_interval,
        uniform_iteration,
        fully_sampled_iteration,
        fully_sampled_time_average,
        fully_sampled_fraction,
        potential_exponent,
        temporal_scale,
        temporal_step,
    )
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if acceleration == 1.0:
        return np.ones((p, t), dtype=bool)

    lines_per_frame = int(_matlab_round(p / acceleration))
    number_of_points = lines_per_frame * t
    phase, time = _initial_vrs_points(
        p, t, lines_per_frame, alpha, sigma, seed
    )
    displacement_extent = np.zeros(number_of_points, dtype=np.float64)
    iteration_step = 1.0
    time_scales = np.full(n_iter, temporal_scale, dtype=np.float64)

    for iteration_index in range(n_iter):
        iteration = iteration_index + 1
        tiled_phase, tiled_time = _tile_points(phase, time, p, t)
        raw_charge = _charges(tiled_phase, alpha, sigma)
        scale = float(time_scales[iteration_index])
        project_to_grid = iteration % grid_interval == 0 or iteration == n_iter

        # Deliberately sequential. Updating tiled_phase[point] immediately is
        # what lets later points observe earlier updates from this iteration.
        for point in range(number_of_points):
            delta_phase = tiled_phase[point] - tiled_phase
            delta_time = scale * (tiled_time - tiled_time[point])
            distance_squared = delta_phase**2 + delta_time**2
            distance_squared[distance_squared == 0.0] = np.inf
            inverse_s = distance_squared ** (-potential_exponent / 2.0)
            inverse_s_plus_two = inverse_s / distance_squared

            scaled_charge = raw_charge + (1.0 - raw_charge[0])
            derivative = (
                alpha
                * tiled_phase[point]
                * math.exp(-(tiled_phase[point] ** 2) / (2.0 * sigma**2))
                / sigma**2
            )
            force = float(
                np.sum(
                    potential_exponent
                    * delta_phase
                    * scaled_charge[point]
                    * scaled_charge
                    * inverse_s_plus_two
                    - derivative * scaled_charge * inverse_s
                )
            )
            displacement = iteration_step * force
            new_phase = tiled_phase[point] + float(
                np.clip(displacement, -acceleration / 4.0, acceleration / 4.0)
            )
            lower = -math.floor(p / 2.0) - 0.5
            upper = math.ceil(p / 2.0) - 0.5
            if new_phase < lower:
                new_phase += p
            elif new_phase > upper:
                new_phase -= p
            if project_to_grid:
                new_phase = float(_matlab_round(new_phase))
                new_phase = float(
                    _wrap_centered_grid(np.asarray([new_phase], dtype=np.int64), p)[0]
                )
            tiled_phase[point] = new_phase
            raw_charge[point] = float(
                _charges(np.asarray([new_phase]), alpha, sigma)[0]
            )
            if iteration == 2:
                displacement_extent[point] = abs(displacement)

        if iteration == 3:
            median_displacement = float(np.median(displacement_extent))
            if not np.isfinite(median_displacement) or median_displacement <= 0.0:
                raise RuntimeError(
                    "VISTA step normalization produced zero displacement"
                )
            iteration_step = (
                step_size * (1.0 + acceleration / 4.0) / median_displacement
            )

        phase = tiled_phase[:number_of_points].copy()
        time = tiled_time[:number_of_points].copy()
        if iteration == uniform_iteration:
            frame_sorted = np.empty((lines_per_frame, t), dtype=np.float64)
            for frame in range(t):
                start = frame * lines_per_frame
                frame_sorted[:, frame] = np.sort(
                    phase[start : start + lines_per_frame]
                )
            template = np.asarray(
                _matlab_round(np.mean(frame_sorted, axis=1)), dtype=np.float64
            )
            phase = np.tile(template, t)
            jitter_draws = np.random.RandomState(seed).random_sample(t)
            for frame, time_coordinate in enumerate(_centered_axis(t)):
                in_frame = time == time_coordinate
                jitter = int(
                    _matlab_round(
                        0.5 * acceleration * (float(jitter_draws[frame]) - 0.5)
                    )
                )
                jittered = np.asarray(
                    _matlab_round(phase[in_frame] + jitter), dtype=np.int64
                )
                phase[in_frame] = _wrap_centered_grid(jittered, p)

            future_iterations = np.arange(iteration + 1, n_iter + 1)
            if future_iterations.size:
                decay = math.ceil(n_iter / 60.0)
                time_scales[iteration:] *= 1.0 + np.exp(
                    -(future_iterations - (iteration + 1)) / decay
                )

        if project_to_grid:
            phase, time = _relocate_duplicates(phase, time, p, t)

        if project_to_grid and iteration >= fully_sampled_iteration:
            if fully_sampled_time_average == 1:
                _fill_time_average(
                    np.arange(number_of_points, dtype=np.int64),
                    phase,
                    time,
                    p=p,
                    alpha=alpha,
                    sigma=sigma,
                    temporal_scale=temporal_scale,
                    potential_exponent=potential_exponent,
                    fraction=fully_sampled_fraction,
                )
            elif fully_sampled_time_average > 1:
                block_size = lines_per_frame * fully_sampled_time_average
                for start in range(0, number_of_points - block_size + 1, block_size):
                    _fill_time_average(
                        np.arange(start, start + block_size, dtype=np.int64),
                        phase,
                        time,
                        p=p,
                        alpha=alpha,
                        sigma=sigma,
                        temporal_scale=temporal_scale,
                        potential_exponent=potential_exponent,
                        fraction=fully_sampled_fraction,
                    )

    phase, time = _relocate_duplicates(phase, time, p, t)
    phase_indices = phase.astype(np.int64) + p // 2
    time_indices = time.astype(np.int64) + t // 2
    if (
        np.any(phase_indices < 0)
        or np.any(phase_indices >= p)
        or np.any(time_indices < 0)
        or np.any(time_indices >= t)
    ):
        raise RuntimeError("VISTA produced a coordinate outside the Cartesian grid")
    mask = np.zeros((p, t), dtype=bool)
    mask[phase_indices, time_indices] = True
    if int(np.count_nonzero(mask)) != number_of_points:
        raise RuntimeError("VISTA produced duplicate final coordinates")
    return mask
