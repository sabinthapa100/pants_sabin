from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load a NIfTI file while preserving spatial metadata.

    Returns
    -------
    image:
        NiBabel image object containing header and affine.

    array:
        NumPy view of voxel data.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"NIfTI file not found: {path}"
        )

    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)

    return image, array


def get_geometry(image: nib.Nifti1Image) -> dict[str, Any]:
    """Return the important spatial properties of a NIfTI volume."""

    return {
        "shape": tuple(image.shape),
        "dtype": str(image.get_data_dtype()),
        "spacing_mm": tuple(
            float(x)
            for x in image.header.get_zooms()[:3]
        ),
        "orientation": tuple(
            nib.aff2axcodes(image.affine)
        ),
        "affine": image.affine.tolist(),
    }


def compare_geometry(
    first: nib.Nifti1Image,
    second: nib.Nifti1Image,
    atol: float = 1e-5,
) -> dict[str, bool]:
    """Compare voxel grids and their voxel-to-world transformations."""

    same_shape = first.shape == second.shape
    same_spacing = np.allclose(
        first.header.get_zooms()[:3],
        second.header.get_zooms()[:3],
        rtol=0.0,
        atol=atol,
    )
    same_orientation = (
        nib.aff2axcodes(first.affine)
        == nib.aff2axcodes(second.affine)
    )
    same_affine = np.allclose(
        first.affine,
        second.affine,
        rtol=0.0,
        atol=atol,
    )

    return {
        "same_shape": same_shape,
        "same_spacing": bool(same_spacing),
        "same_orientation": same_orientation,
        "same_affine": bool(same_affine),
        "same_geometry": bool(
            same_shape
            and same_spacing
            and same_orientation
            and same_affine
        ),
    }


def same_geometry(
    first: nib.Nifti1Image,
    second: nib.Nifti1Image,
    atol: float = 1e-5,
) -> bool:
    """
    Check whether two NIfTI images share the same voxel grid
    and voxel-to-world coordinate transformation.
    """

    return compare_geometry(first, second, atol)["same_geometry"]
