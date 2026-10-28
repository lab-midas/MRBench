"""Complementary display-only anatomy and amplitude-fidelity QC helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ERROR_MAP_COLORMAP = "turbo"
ANATOMY_ERROR_DISPLAY_RANGE = (0.0, 0.2)
FIDELITY_ERROR_DISPLAY_RANGE = (0.0, 1.0)


def _magnitude_pair(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference_magnitude = np.abs(np.asarray(reference)).astype(np.float64, copy=False)
    candidate_magnitude = np.abs(np.asarray(candidate)).astype(np.float64, copy=False)
    if reference_magnitude.shape != candidate_magnitude.shape:
        raise ValueError(
            "reference and candidate shapes differ: "
            f"{reference_magnitude.shape} vs {candidate_magnitude.shape}"
        )
    if not np.all(np.isfinite(reference_magnitude)) or not np.all(
        np.isfinite(candidate_magnitude)
    ):
        raise ValueError("QC error maps require finite magnitude arrays")
    return reference_magnitude, candidate_magnitude


def _safe_max_normalize(magnitude: np.ndarray) -> np.ndarray:
    maximum = float(np.max(magnitude))
    if maximum == 0.0:
        return np.zeros_like(magnitude, dtype=np.float64)
    return magnitude / maximum


def scale_invariant_magnitude_error(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return display-only ``abs(c/max(c) - r/max(r))`` structural error."""

    reference_magnitude, candidate_magnitude = _magnitude_pair(reference, candidate)
    return np.abs(
        _safe_max_normalize(candidate_magnitude)
        - _safe_max_normalize(reference_magnitude)
    )


def amplitude_sensitive_fidelity_error(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Return ``abs(abs(candidate)-abs(reference)) / reference_data_range``."""

    reference_magnitude, candidate_magnitude = _magnitude_pair(reference, candidate)
    data_range = float(
        np.max(reference_magnitude) - np.min(reference_magnitude)
    )
    if data_range <= 0.0:
        raise ValueError(
            "fidelity error requires a positive reference magnitude data range"
        )
    return np.abs(candidate_magnitude - reference_magnitude) / data_range


def _validate_qc_inputs(
    reference: np.ndarray,
    zero_filled: np.ndarray,
    reconstruction: np.ndarray,
    frames: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    arrays = tuple(np.asarray(value) for value in (reference, zero_filled, reconstruction))
    if any(value.ndim != 4 for value in arrays):
        raise ValueError("QC inputs must use canonical [S,T,X,Y] shape")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("reference, zero-filled, and reconstruction shapes must match")
    selected = list(frames)
    if not selected:
        raise ValueError("at least one deterministic QC frame is required")
    slices, phases = arrays[0].shape[:2]
    for slice_index, phase_index in selected:
        if not 0 <= slice_index < slices or not 0 <= phase_index < phases:
            raise ValueError(
                f"QC frame {(slice_index, phase_index)} is outside "
                f"S={slices}, T={phases}"
            )
    return arrays[0], arrays[1], arrays[2], selected


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("QC figures require matplotlib") from exc
    return plt


def _create_qc_layout(plt, frame_count: int):
    """Create side-by-side 2x3 phase blocks without unused generic panels."""

    figure = plt.figure(
        figsize=(8.0 * frame_count, 5.4),
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.78)
    grid = figure.add_gridspec(
        2,
        3 * frame_count,
        wspace=0.08,
        hspace=0.28,
    )
    axes = np.empty((2, 3 * frame_count), dtype=object)
    for row in range(2):
        for column in range(3 * frame_count):
            axes[row, column] = figure.add_subplot(grid[row, column])
    return figure, axes


def _validate_mask_context(
    canonical_mask: np.ndarray | None,
    actual_acceleration: float | None,
    reference_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Validate display context without deriving or renaming acceleration."""

    if canonical_mask is None and actual_acceleration is None:
        return None
    if canonical_mask is None or actual_acceleration is None:
        raise ValueError(
            "QC canonical_mask and global actual_acceleration must be supplied together"
        )
    mask = np.asarray(canonical_mask)
    if mask.ndim != 4 or tuple(mask.shape[:2]) != tuple(reference_shape[:2]):
        raise ValueError(
            "QC canonical_mask must use [S,T,Kx,Ky] with the evaluated S,T"
        )
    if not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("QC canonical_mask must contain finite exact binary values")
    if not np.isfinite(actual_acceleration) or actual_acceleration <= 0.0:
        raise ValueError("QC global actual_acceleration must be finite and positive")
    return mask


def _actual_r_label(actual_acceleration: float) -> str:
    return f"Mask\nglobal actual R = {actual_acceleration:.3g}"


def _selected_mask_frame(
    mask: np.ndarray, slice_index: int, phase_index: int
) -> np.ndarray:
    """Return the exact canonical mask plane used by one displayed QC frame."""

    return mask[slice_index, phase_index]


def _mask_panel_records(
    mask: np.ndarray | None,
    selected: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    if mask is None:
        return []
    records: list[dict[str, Any]] = []
    for slice_index, phase_index in selected:
        panel = np.ascontiguousarray(
            _selected_mask_frame(mask, slice_index, phase_index).astype(
                np.uint8, copy=False
            )
        )
        records.append(
            {
                "slice": slice_index,
                "phase": phase_index,
                "shape": list(panel.shape),
                "sampled_points": int(np.count_nonzero(panel)),
                "binary_sha256": hashlib.sha256(panel.tobytes()).hexdigest(),
            }
        )
    return records


def _plot_error_map(axis, error: np.ndarray, *, cmap: str, vmax: float):
    """Plot the original scientific error array with display-only saturation."""

    return axis.imshow(error, cmap=cmap, vmin=0.0, vmax=vmax)


def _add_phase_group_labels(
    figure,
    axes: np.ndarray,
    selected: list[tuple[int, int]],
) -> list[str]:
    """Center factual slice/phase labels over each 2x3 block."""

    figure.canvas.draw()
    labels: list[str] = []
    for frame_number, (slice_index, phase_index) in enumerate(selected):
        first_box = axes[0, 3 * frame_number].get_position()
        final_box = axes[0, 3 * frame_number + 2].get_position()
        label = f"Phase {phase_index} · Slice {slice_index} (zero-based)"
        figure.text(
            0.5 * (first_box.x0 + final_box.x1),
            first_box.y1 + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )
        labels.append(label)
    return labels


def _qc_alignment_geometry(
    figure,
    axes: np.ndarray,
    frame_count: int,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Measure image/error alignment after the figure layout has resolved."""

    figure.canvas.draw()
    candidates = (("zero_filled", 1), ("reconstruction", 2))
    measured: dict[str, Any] = {}
    for name, column in candidates:
        center_differences: list[float] = []
        width_differences: list[float] = []
        for frame_number in range(frame_count):
            block_column = 3 * frame_number + column
            image_box = axes[0, block_column].get_position()
            error_box = axes[1, block_column].get_position()
            image_center = 0.5 * (image_box.x0 + image_box.x1)
            error_center = 0.5 * (error_box.x0 + error_box.x1)
            center_differences.append(abs(image_center - error_center))
            width_differences.append(abs(image_box.width - error_box.width))
        maximum_center_difference = max(center_differences)
        maximum_width_difference = max(width_differences)
        measured[name] = {
            "maximum_center_x_difference": maximum_center_difference,
            "maximum_axes_width_difference": maximum_width_difference,
            "centers_aligned": maximum_center_difference <= tolerance,
            "widths_aligned": maximum_width_difference <= tolerance,
        }
    measured["tolerance"] = tolerance
    return measured


def _qc_colorbar_geometry(
    figure,
    axes: np.ndarray,
    colorbar_axis,
    *,
    reconstruction_label: str = "RAM",
    height_ratio_tolerance: float = 0.02,
    position_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Measure placement beside only the final reconstruction error-map panel."""

    figure.canvas.draw()
    colorbar_box = colorbar_axis.get_position()
    final_error_box = axes[1, -1].get_position()
    height_ratio = colorbar_box.height / final_error_box.height
    to_right = colorbar_box.x0 >= final_error_box.x1 - position_tolerance
    return {
        "orientation": "vertical",
        "anchor_panel": f"final {reconstruction_label} error map",
        "to_right_of_final_reconstruction_error": to_right,
        "to_right_of_final_ram_error": to_right,
        "horizontal_padding": colorbar_box.x0 - final_error_box.x1,
        "vertical_center_difference": abs(
            0.5 * (colorbar_box.y0 + colorbar_box.y1)
            - 0.5 * (final_error_box.y0 + final_error_box.y1)
        ),
        "height_to_error_panel_ratio": height_ratio,
        "height_ratio_tolerance": height_ratio_tolerance,
        "height_matches_error_panel": abs(height_ratio - 1.0)
        <= height_ratio_tolerance,
        "width_to_error_panel_width_ratio": (
            colorbar_box.width / final_error_box.width
        ),
        "label": None,
    }


def _add_final_error_colorbar_axis(
    figure,
    axes: np.ndarray,
    *,
    padding: float = 0.008,
    width: float = 0.012,
):
    """Add a narrow colorbar beside the final RAM error without resizing it."""

    figure.canvas.draw()
    target_box = axes[1, -1].get_position()
    colorbar_axis = figure.add_axes(
        [target_box.x1 + padding, target_box.y0, width, target_box.height]
    )
    colorbar_axis.set_in_layout(False)
    return colorbar_axis


def _qc_spacing_geometry(
    figure,
    axes: np.ndarray,
    *,
    position_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Measure clearance between each lower-row title and the row above."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse = figure.transFigure.inverted()
    clearances: list[float] = []
    for column in range(axes.shape[1]):
        title = axes[1, column].title
        if not title.get_text():
            continue
        title_box = title.get_window_extent(renderer=renderer).transformed(inverse)
        upper_box = axes[0, column].get_position()
        clearances.append(upper_box.y0 - title_box.y1)
    minimum_clearance = min(clearances) if clearances else float("nan")
    return {
        "grid_hspace": 0.28,
        "minimum_title_clearance": minimum_clearance,
        "titles_clear_of_preceding_panels": minimum_clearance
        > position_tolerance,
    }


def save_anatomy_qc(
    path: Path,
    *,
    subject: str,
    acceleration: int,
    reference: np.ndarray,
    zero_filled: np.ndarray,
    reconstruction: np.ndarray,
    frames: Iterable[tuple[int, int]],
    canonical_mask: np.ndarray | None = None,
    actual_acceleration: float | None = None,
    error_vmax: float = 0.2,
    error_cmap: str = ERROR_MAP_COLORMAP,
    reconstruction_label: str = "RAM",
) -> dict[str, Any]:
    """Save independently autoscaled anatomy panels and structural errors."""

    reference, zero_filled, reconstruction, selected = _validate_qc_inputs(
        reference, zero_filled, reconstruction, frames
    )
    if error_vmax <= 0.0:
        raise ValueError("anatomy QC error_vmax must be positive")
    mask = _validate_mask_context(
        canonical_mask, actual_acceleration, tuple(reference.shape)
    )
    plt = _matplotlib_pyplot()
    figure, axes = _create_qc_layout(plt, len(selected))
    error_image = None
    for frame_number, (slice_index, phase_index) in enumerate(selected):
        block_start = 3 * frame_number
        frames_for_display = (
            ("Reference", reference[slice_index, phase_index]),
            ("Zero-filled", zero_filled[slice_index, phase_index]),
            ("Reconstruction", reconstruction[slice_index, phase_index]),
        )
        for column, (label, values) in enumerate(frames_for_display):
            axis = axes[0, block_start + column]
            axis.imshow(np.abs(values), cmap="gray")
            axis.set_title(label)
            axis.axis("off")
        mask_axis = axes[1, block_start]
        if mask is None:
            mask_axis.axis("off")
        else:
            mask_axis.imshow(
                _selected_mask_frame(mask, slice_index, phase_index),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            mask_axis.set_title(_actual_r_label(float(actual_acceleration)))
            mask_axis.axis("off")
        errors = (
            scale_invariant_magnitude_error(
                reference[slice_index, phase_index],
                zero_filled[slice_index, phase_index],
            ),
            scale_invariant_magnitude_error(
                reference[slice_index, phase_index],
                reconstruction[slice_index, phase_index],
            ),
        )
        for column, error in enumerate(errors, start=1):
            error_axis = axes[1, block_start + column]
            error_image = _plot_error_map(
                error_axis, error, cmap=error_cmap, vmax=error_vmax
            )
            error_axis.set_title("Error map")
            error_axis.axis("off")
    if error_image is None:  # guarded by _validate_qc_inputs, retained for typing
        raise RuntimeError("anatomy QC did not create an error panel")
    colorbar_axis = _add_final_error_colorbar_axis(figure, axes)
    figure.colorbar(
        error_image,
        cax=colorbar_axis,
        orientation="vertical",
    )
    figure.suptitle(
        f"{subject} R={acceleration}: Anatomy QC\n"
        "Images independently autoscaled; structural error display "
        f"[0, {error_vmax:g}]."
    )
    phase_group_labels = _add_phase_group_labels(figure, axes, selected)
    alignment = _qc_alignment_geometry(figure, axes, len(selected))
    spacing_geometry = _qc_spacing_geometry(figure, axes)
    colorbar_geometry = _qc_colorbar_geometry(
        figure, axes, colorbar_axis, reconstruction_label=reconstruction_label
    )
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return {
        "path": str(path.resolve()),
        "mode": "anatomy_qc",
        "frames": [{"slice": s, "phase": t} for s, t in selected],
        "image_scaling": "independent panel autoscaling",
        "error": "scale-invariant magnitude error",
        "error_display_range": [0.0, error_vmax],
        "error_display_label": (
            f"Scale-invariant magnitude error; display range [0, {error_vmax:g}]"
        ),
        "numerical_error_clipping": False,
        "display_saturation_above_vmax": True,
        "error_colormap": error_cmap,
        "title": (
            f"{subject} R={acceleration}: Anatomy QC\n"
            "Images independently autoscaled; structural error display "
            f"[0, {error_vmax:g}]."
        ),
        "error_panel_title": "Error map",
        "colorbar_label": None,
        "mask": {
            "shown": mask is not None,
            "source": "verified canonical sampling mask" if mask is not None else None,
            "indices": (
                [
                    {"slice": slice_index, "phase": phase_index}
                    for slice_index, phase_index in selected
                ]
                if mask is not None
                else []
            ),
            "panels": _mask_panel_records(mask, selected),
            "actual_acceleration": actual_acceleration,
            "actual_acceleration_scope": (
                "global complete canonical sampling mask" if mask is not None else None
            ),
            "panel_title": (
                _actual_r_label(float(actual_acceleration))
                if mask is not None
                else None
            ),
        },
        "layout": {
            "rows": 2,
            "columns": 3 * len(selected),
            "outer_rows": 1,
            "outer_columns": len(selected),
            "phase_block_rows": 2,
            "phase_block_columns": 3,
            "phase_blocks_arranged": "side_by_side",
            "phase_group_labels": phase_group_labels,
            "row_hspace": 0.28,
            "shared_scientific_grid": True,
            "dedicated_external_colorbar_axis": True,
            "alignment": alignment,
            "spacing": spacing_geometry,
            "colorbar": colorbar_geometry,
            "frame_block": (
                "Reference/Zero-filled/Reconstruction above "
                "Mask/ZF Error map/Reconstruction Error map"
            ),
        },
        "reconstruction_label": reconstruction_label,
    }


def save_fidelity_qc(
    path: Path,
    *,
    subject: str,
    acceleration: int,
    reference: np.ndarray,
    zero_filled: np.ndarray,
    reconstruction: np.ndarray,
    frames: Iterable[tuple[int, int]],
    canonical_mask: np.ndarray | None = None,
    actual_acceleration: float | None = None,
    error_vmax: float = 1.0,
    error_cmap: str = ERROR_MAP_COLORMAP,
    reconstruction_label: str = "RAM",
) -> dict[str, Any]:
    """Save reference-windowed panels and amplitude-sensitive fidelity errors."""

    reference, zero_filled, reconstruction, selected = _validate_qc_inputs(
        reference, zero_filled, reconstruction, frames
    )
    if error_vmax <= 0.0:
        raise ValueError("fidelity QC error_vmax must be positive")
    mask = _validate_mask_context(
        canonical_mask, actual_acceleration, tuple(reference.shape)
    )
    plt = _matplotlib_pyplot()
    figure, axes = _create_qc_layout(plt, len(selected))
    windows: list[dict[str, float | int]] = []
    error_image = None
    for frame_number, (slice_index, phase_index) in enumerate(selected):
        block_start = 3 * frame_number
        reference_frame = reference[slice_index, phase_index]
        reference_vmax = float(np.max(np.abs(reference_frame)))
        if reference_vmax <= 0.0:
            raise ValueError("fidelity QC requires positive reference maximum")
        windows.append(
            {"slice": slice_index, "phase": phase_index, "vmin": 0.0, "vmax": reference_vmax}
        )
        frames_for_display = (
            ("Reference", reference_frame),
            ("Zero-filled", zero_filled[slice_index, phase_index]),
            ("Reconstruction", reconstruction[slice_index, phase_index]),
        )
        for column, (label, values) in enumerate(frames_for_display):
            axis = axes[0, block_start + column]
            axis.imshow(
                np.abs(values), cmap="gray", vmin=0.0, vmax=reference_vmax
            )
            axis.set_title(label)
            axis.axis("off")
        mask_axis = axes[1, block_start]
        if mask is None:
            mask_axis.axis("off")
        else:
            mask_axis.imshow(
                _selected_mask_frame(mask, slice_index, phase_index),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            mask_axis.set_title(_actual_r_label(float(actual_acceleration)))
            mask_axis.axis("off")
        errors = (
            amplitude_sensitive_fidelity_error(
                reference_frame, zero_filled[slice_index, phase_index]
            ),
            amplitude_sensitive_fidelity_error(
                reference_frame, reconstruction[slice_index, phase_index]
            ),
        )
        for column, error in enumerate(errors, start=1):
            error_axis = axes[1, block_start + column]
            error_image = _plot_error_map(
                error_axis, error, cmap=error_cmap, vmax=error_vmax
            )
            error_axis.set_title("Error map")
            error_axis.axis("off")
    if error_image is None:  # guarded by _validate_qc_inputs, retained for typing
        raise RuntimeError("fidelity QC did not create an error panel")
    colorbar_axis = _add_final_error_colorbar_axis(figure, axes)
    figure.colorbar(
        error_image,
        cax=colorbar_axis,
        orientation="vertical",
    )
    figure.suptitle(
        f"{subject} R={acceleration}: Fidelity QC\n"
        "Per-frame reference-only image window; fidelity error display "
        f"[0, {error_vmax:g}]."
    )
    phase_group_labels = _add_phase_group_labels(figure, axes, selected)
    alignment = _qc_alignment_geometry(figure, axes, len(selected))
    spacing_geometry = _qc_spacing_geometry(figure, axes)
    colorbar_geometry = _qc_colorbar_geometry(
        figure, axes, colorbar_axis, reconstruction_label=reconstruction_label
    )
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return {
        "path": str(path.resolve()),
        "mode": "fidelity_qc",
        "frames": [{"slice": s, "phase": t} for s, t in selected],
        "image_scaling": "per-frame reference-only window",
        "reference_windows": windows,
        "error": "amplitude-sensitive reference-normalized magnitude error",
        "error_display_range": [0.0, error_vmax],
        "error_display_label": (
            "Reference-normalized magnitude error; "
            f"display range [0, {error_vmax:g}]"
        ),
        "numerical_error_clipping": False,
        "display_saturation_above_vmax": True,
        "error_colormap": error_cmap,
        "title": (
            f"{subject} R={acceleration}: Fidelity QC\n"
            "Per-frame reference-only image window; fidelity error display "
            f"[0, {error_vmax:g}]."
        ),
        "error_panel_title": "Error map",
        "colorbar_label": None,
        "mask": {
            "shown": mask is not None,
            "source": "verified canonical sampling mask" if mask is not None else None,
            "indices": (
                [
                    {"slice": slice_index, "phase": phase_index}
                    for slice_index, phase_index in selected
                ]
                if mask is not None
                else []
            ),
            "panels": _mask_panel_records(mask, selected),
            "actual_acceleration": actual_acceleration,
            "actual_acceleration_scope": (
                "global complete canonical sampling mask" if mask is not None else None
            ),
            "panel_title": (
                _actual_r_label(float(actual_acceleration))
                if mask is not None
                else None
            ),
        },
        "layout": {
            "rows": 2,
            "columns": 3 * len(selected),
            "outer_rows": 1,
            "outer_columns": len(selected),
            "phase_block_rows": 2,
            "phase_block_columns": 3,
            "phase_blocks_arranged": "side_by_side",
            "phase_group_labels": phase_group_labels,
            "row_hspace": 0.28,
            "shared_scientific_grid": True,
            "dedicated_external_colorbar_axis": True,
            "alignment": alignment,
            "spacing": spacing_geometry,
            "colorbar": colorbar_geometry,
            "frame_block": (
                "Reference/Zero-filled/Reconstruction above "
                "Mask/ZF Error map/Reconstruction Error map"
            ),
        },
        "reconstruction_label": reconstruction_label,
    }
