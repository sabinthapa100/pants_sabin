"""Reusable, mask-guided visualization for PanTS NIfTI cases."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np

from .io import compare_geometry, get_geometry, load_nifti
from .labels import CLASS_MAP, NAME_TO_CLASS
from .paths import PROJECT_ROOT, get_case_paths


DEFAULT_NEGATIVE_STRUCTURES = (
    "pancreas",
    "pancreas_head",
    "pancreas_body",
    "pancreas_tail",
)
DEFAULT_POSITIVE_STRUCTURES = (
    "pancreas",
    "pancreatic_lesion",
)

PLANE_AXES = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}

PLANE_DISPLAY_AXES = {
    "sagittal": (1, 2),  # horizontal: posterior-anterior; vertical: inferior-superior
    "coronal": (0, 2),   # horizontal: left-right; vertical: inferior-superior
    "axial": (0, 1),     # horizontal: left-right; vertical: posterior-anterior
}

PLANE_DIRECTION_LABELS = {
    "sagittal": ("P  ←  →  A", "I  ←  →  S"),
    "coronal": ("L  ←  →  R", "I  ←  →  S"),
    "axial": ("L  ←  →  R", "P  ←  →  A"),
}

SPECIAL_COLORS = {
    "pancreas": "#00E5FF",
    "pancreatic_lesion": "#FF1744",
    "pancreas_head": "#FFD600",
    "pancreas_body": "#76FF03",
    "pancreas_tail": "#FF9100",
    "pancreatic_duct": "#E040FB",
    "veins": "#2979FF",
    "superior_mesenteric_artery": "#FF5252",
}


def select_max_area_slices(mask: np.ndarray) -> dict[str, int]:
    """Select one canonical sagittal, coronal, and axial mask-rich slice."""

    if mask.ndim != 3:
        raise ValueError(f"Selection mask must be 3D, got shape {mask.shape}.")
    if not np.any(mask):
        raise ValueError("Cannot select informative slices from an empty mask.")

    selected: dict[str, int] = {}
    for plane, axis in PLANE_AXES.items():
        in_plane_axes = tuple(index for index in range(3) if index != axis)
        areas = np.sum(mask, axis=in_plane_axes)
        maximizers = np.flatnonzero(areas == areas.max())
        selected[plane] = int(maximizers[len(maximizers) // 2])
    return selected


def extract_canonical_slice(
    volume: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """Extract a 2D anatomical view from a canonical RAS+ volume."""

    if plane == "sagittal":
        return volume[index, :, :].T
    if plane == "coronal":
        return volume[:, index, :].T
    if plane == "axial":
        return volume[:, :, index].T
    raise ValueError(f"Unknown anatomical plane: {plane}")


def _structure_color(name: str) -> Any:
    if name in SPECIAL_COLORS:
        return SPECIAL_COLORS[name]
    class_id = NAME_TO_CLASS[name]
    color_map = plt.get_cmap("tab20")
    return color_map(((class_id - 1) % 20) / 19)


def _validate_structure_names(structures: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(structures))
    unknown = sorted(set(requested) - set(NAME_TO_CLASS))
    if unknown:
        raise ValueError(
            f"Unknown PanTS structure names: {unknown}. "
            f"Valid names are: {sorted(NAME_TO_CLASS)}"
        )
    if not requested:
        raise ValueError("At least one structure must be requested.")
    if len(requested) > 6:
        warnings.warn(
            f"{len(requested)} overlays were requested; the figure may be "
            "difficult to interpret.",
            stacklevel=2,
        )
    return requested


def _require_aligned(
    ct_image: nib.Nifti1Image,
    mask_images: dict[str, nib.Nifti1Image],
    stage: str,
) -> None:
    mismatches = {}
    for name, image in mask_images.items():
        comparison = compare_geometry(ct_image, image)
        if not comparison["same_geometry"]:
            mismatches[name] = comparison
    if mismatches:
        raise ValueError(
            f"CT/mask geometry mismatch during {stage}: {mismatches}"
        )


def _plane_aspect(
    plane: str,
    spacing: tuple[float, float, float],
) -> float:
    horizontal_axis, vertical_axis = PLANE_DISPLAY_AXES[plane]
    return spacing[vertical_axis] / spacing[horizontal_axis]


def _display_ct(
    axis: plt.Axes,
    image: np.ndarray,
    aspect: float,
    window_min: float,
    window_max: float,
) -> None:
    axis.imshow(
        image,
        cmap="gray",
        origin="lower",
        vmin=window_min,
        vmax=window_max,
        interpolation="nearest",
        aspect=aspect,
    )


def _add_overlay(
    axis: plt.Axes,
    mask: np.ndarray,
    structure: str,
    aspect: float,
) -> None:
    if not np.any(mask):
        return

    color = _structure_color(structure)
    if structure != "pancreas":
        overlay = np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))
        axis.imshow(
            overlay,
            cmap=ListedColormap([color]),
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            alpha=0.28 if structure == "pancreatic_lesion" else 0.20,
            aspect=aspect,
        )

    axis.contour(
        mask.astype(np.uint8),
        levels=(0.5,),
        colors=(color,),
        linewidths=1.5 if structure == "pancreatic_lesion" else 1.0,
        origin="lower",
    )


def visualize_case(
    case_id: str,
    split: str = "train",
    structures: Sequence[str] | None = None,
    window_min: float = -150.0,
    window_max: float = 250.0,
    output_dir: Path | None = None,
    prediction: nib.Nifti1Image | None = None,
) -> dict[str, Any]:
    """Create a mask-guided three-plane PanTS quality-control figure.

    ``prediction`` is reserved for a later inference-QC milestone. Prediction
    panels are intentionally not implemented here.
    """

    if window_min >= window_max:
        raise ValueError(
            f"window_min ({window_min}) must be less than window_max "
            f"({window_max})."
        )
    if prediction is not None:
        raise NotImplementedError(
            "Prediction comparison belongs to the later inference-QC milestone."
        )

    paths = get_case_paths(case_id=case_id, split=split)
    ct_image, _ = load_nifti(paths["ct"])

    core_names = ("pancreas", "pancreatic_lesion")
    mask_images: dict[str, nib.Nifti1Image] = {}
    native_arrays: dict[str, np.ndarray] = {}
    for name in core_names:
        image, array = load_nifti(paths["segmentations"] / f"{name}.nii.gz")
        mask_images[name] = image
        native_arrays[name] = array

    lesion_voxels = int(np.count_nonzero(native_arrays["pancreatic_lesion"] > 0))
    lesion_positive = lesion_voxels > 0

    if structures is None:
        requested = (
            DEFAULT_POSITIVE_STRUCTURES
            if lesion_positive
            else DEFAULT_NEGATIVE_STRUCTURES
        )
    else:
        requested = _validate_structure_names(structures)

    requested = _validate_structure_names(requested)
    names_to_load = tuple(
        sorted(set(requested) | set(core_names), key=NAME_TO_CLASS.get)
    )
    for name in names_to_load:
        if name in mask_images:
            continue
        image, array = load_nifti(paths["segmentations"] / f"{name}.nii.gz")
        mask_images[name] = image
        native_arrays[name] = array

    _require_aligned(ct_image, mask_images, stage="native orientation")

    original_orientation = tuple(nib.aff2axcodes(ct_image.affine))
    canonical_ct_image = nib.as_closest_canonical(ct_image)
    canonical_mask_images = {
        name: nib.as_closest_canonical(image)
        for name, image in mask_images.items()
    }
    _require_aligned(
        canonical_ct_image,
        canonical_mask_images,
        stage="canonical orientation",
    )

    canonical_orientation = tuple(
        nib.aff2axcodes(canonical_ct_image.affine)
    )
    ct = np.asanyarray(canonical_ct_image.dataobj)
    masks = {
        name: np.asanyarray(image.dataobj) > 0
        for name, image in canonical_mask_images.items()
    }

    selection_structure = (
        "pancreatic_lesion" if lesion_positive else "pancreas"
    )
    selected_slices = select_max_area_slices(masks[selection_structure])
    spacing = tuple(
        float(value)
        for value in canonical_ct_image.header.get_zooms()[:3]
    )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lesion" if lesion_positive else "anatomy"
    output_path = output_dir / f"{case_id}_{suffix}.png"

    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 12),
        constrained_layout=True,
    )
    empty_structures = [
        name for name in requested if not np.any(masks[name])
    ]

    for row, plane in enumerate(("axial", "coronal", "sagittal")):
        slice_index = selected_slices[plane]
        ct_slice = extract_canonical_slice(ct, plane, slice_index)
        aspect = _plane_aspect(plane, spacing)

        for column in range(2):
            axis = axes[row, column]
            _display_ct(axis, ct_slice, aspect, window_min, window_max)
            horizontal_label, vertical_label = PLANE_DIRECTION_LABELS[plane]
            axis.set_xlabel(horizontal_label)
            axis.set_ylabel(vertical_label)
            axis.set_xticks([])
            axis.set_yticks([])

        axes[row, 0].set_title(
            f"{plane.capitalize()} CT | canonical index {slice_index}"
        )
        axes[row, 1].set_title(f"{plane.capitalize()} CT + annotations")

        for structure in requested:
            mask_slice = extract_canonical_slice(
                masks[structure], plane, slice_index
            )
            _add_overlay(axes[row, 1], mask_slice, structure, aspect)

    legend_handles = [
        Line2D(
            (0,),
            (0,),
            color=_structure_color(name),
            linewidth=2,
            label=name,
        )
        for name in requested
        if name not in empty_structures
    ]
    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncols=min(4, len(legend_handles)),
        )

    case_status = "lesion-positive" if lesion_positive else "lesion-negative"
    figure.suptitle(
        f"{case_id} | {case_status} | canonical RAS+ views\n"
        f"DISPLAY WINDOW ONLY [{window_min:g}, {window_max:g}]",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "case_id": case_id,
        "split": split,
        "lesion_positive": lesion_positive,
        "lesion_voxels": lesion_voxels,
        "original_orientation": original_orientation,
        "visualization_orientation": canonical_orientation,
        "canonical_shape": tuple(ct.shape),
        "canonical_spacing_mm": spacing,
        "selection_structure": selection_structure,
        "selected_slices": selected_slices,
        "requested_structures": requested,
        "empty_requested_structures": empty_structures,
        "display_window": (float(window_min), float(window_max)),
        "output_path": str(output_path.resolve()),
        "geometry_alignment": "native and canonical CT/masks aligned",
        "original_geometry": get_geometry(ct_image),
    }
