"""Scientific inspection of one PanTS CT and its pancreas-related labels."""

from typing import Any

import nibabel as nib
import numpy as np

from .io import compare_geometry, get_geometry, load_nifti
from .labels import (
    BACKGROUND,
    CLASS_MAP,
    PANCREAS,
    PANCREAS_BODY,
    PANCREAS_FAMILY,
    PANCREAS_HEAD,
    PANCREAS_TAIL,
    PANCREATIC_LESION,
)
from .paths import get_case_paths


PANCREAS_MASK_FILES = {
    "pancreas": "pancreas.nii.gz",
    "pancreas_body": "pancreas_body.nii.gz",
    "pancreas_head": "pancreas_head.nii.gz",
    "pancreas_tail": "pancreas_tail.nii.gz",
    "pancreatic_lesion": "pancreatic_lesion.nii.gz",
}


def _unique_values(array: np.ndarray) -> list[int | float]:
    """Return JSON-friendly unique values without changing the array."""

    values = np.unique(array)
    if np.issubdtype(values.dtype, np.integer):
        return [int(value) for value in values]
    return [float(value) for value in values]


def _is_integer_valued(array: np.ndarray) -> bool:
    """Return whether every finite value represents an integer."""

    if np.issubdtype(array.dtype, np.integer):
        return True
    return bool(
        np.all(np.isfinite(array))
        and np.all(array == np.rint(array))
    )


def _is_binary_after_scaling(values: list[int | float]) -> bool:
    """Accept NIfTI-scaled binary values that are numerically near 0 or 1."""

    return all(
        np.isclose(value, 0.0, rtol=0.0, atol=1e-6)
        or np.isclose(value, 1.0, rtol=0.0, atol=1e-6)
        for value in values
    )


def _mask_relation(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, int | bool]:
    """Describe equality and directional voxel differences of two masks."""

    return {
        "equal": bool(np.array_equal(reference, candidate)),
        "intersection_voxels": int(np.count_nonzero(reference & candidate)),
        "reference_only_voxels": int(np.count_nonzero(reference & ~candidate)),
        "candidate_only_voxels": int(np.count_nonzero(candidate & ~reference)),
    }


def _format_affine(affine: list[list[float]]) -> list[str]:
    return [
        "    [" + ", ".join(f"{value:10.6f}" for value in row) + "]"
        for row in affine
    ]


def inspect_case(
    case_id: str,
    split: str = "train",
) -> dict[str, Any]:
    """Inspect one PanTS CT and the pancreas-related label representations."""

    paths = get_case_paths(case_id=case_id, split=split)
    ct_image, ct = load_nifti(paths["ct"])
    combined_image, combined = load_nifti(paths["combined"])

    mask_images: dict[str, nib.Nifti1Image] = {}
    mask_arrays: dict[str, np.ndarray] = {}
    mask_unique_values: dict[str, list[int | float]] = {}
    mask_loaded_dtypes: dict[str, str] = {}
    mask_raw_unique_values: dict[str, list[int | float]] = {}
    mask_scaling: dict[str, tuple[float, float]] = {}

    for name, filename in PANCREAS_MASK_FILES.items():
        image, array = load_nifti(paths["segmentations"] / filename)
        mask_images[name] = image
        mask_loaded_dtypes[name] = str(array.dtype)
        mask_unique_values[name] = _unique_values(array)
        raw_array = np.asanyarray(image.dataobj.get_unscaled())
        mask_raw_unique_values[name] = _unique_values(raw_array)
        mask_scaling[name] = (
            float(image.dataobj.slope),
            float(image.dataobj.inter),
        )
        mask_arrays[name] = array > 0

    pancreas = mask_arrays["pancreas"]
    body = mask_arrays["pancreas_body"]
    head = mask_arrays["pancreas_head"]
    tail = mask_arrays["pancreas_tail"]
    lesion = mask_arrays["pancreatic_lesion"]

    parts_union = body | head | tail
    pancreas_not_in_parts = pancreas & ~parts_union
    combined_masks = {
        class_id: combined == class_id
        for class_id in (*PANCREAS_FAMILY, PANCREATIC_LESION)
    }
    combined_pancreas_family = np.isin(combined, PANCREAS_FAMILY)
    combined_pancreas_and_lesion = (
        combined_pancreas_family | combined_masks[PANCREATIC_LESION]
    )
    pancreas_family_gap = pancreas & ~combined_pancreas_family
    gap_values, gap_counts = np.unique(
        combined[pancreas_family_gap],
        return_counts=True,
    )
    pancreas_family_gap_labels = {
        int(value): int(count)
        for value, count in zip(gap_values, gap_counts, strict=True)
    }

    combined_unique = _unique_values(combined)
    expected_values = {BACKGROUND, *CLASS_MAP}
    unexpected_values = sorted(
        value for value in combined_unique if value not in expected_values
    )

    finite_ct = ct[np.isfinite(ct)]
    if finite_ct.size == 0:
        raise ValueError(f"CT contains no finite voxels: {paths['ct']}")

    geometry_matches = {
        "combined_labels": compare_geometry(ct_image, combined_image),
        **{
            name: compare_geometry(ct_image, image)
            for name, image in mask_images.items()
        },
    }

    standalone_masks = {
        name: {
            "stored_dtype": str(mask_images[name].get_data_dtype()),
            "loaded_dtype": mask_loaded_dtypes[name],
            "raw_unique_values": mask_raw_unique_values[name],
            "unique_values": mask_unique_values[name],
            "slope": mask_scaling[name][0],
            "intercept": mask_scaling[name][1],
            "binary_after_scaling": _is_binary_after_scaling(
                mask_unique_values[name]
            ),
            "foreground_voxels": int(mask_arrays[name].sum()),
        }
        for name in PANCREAS_MASK_FILES
    }

    relations = {
        "class28_vs_standalone_lesion": _mask_relation(
            lesion, combined_masks[PANCREATIC_LESION]
        ),
        "class17_vs_standalone_pancreas": _mask_relation(
            pancreas, combined_masks[PANCREAS]
        ),
        "class17_vs_pancreas_not_in_parts": _mask_relation(
            pancreas_not_in_parts, combined_masks[PANCREAS]
        ),
        "classes17_20_vs_standalone_pancreas": _mask_relation(
            pancreas, combined_pancreas_family
        ),
        "classes17_20_plus28_vs_standalone_pancreas": _mask_relation(
            pancreas, combined_pancreas_and_lesion
        ),
        "standalone_parts_union_vs_standalone_pancreas": _mask_relation(
            pancreas, parts_union
        ),
        "class18_vs_standalone_body": _mask_relation(
            body, combined_masks[PANCREAS_BODY]
        ),
        "class19_vs_standalone_head": _mask_relation(
            head, combined_masks[PANCREAS_HEAD]
        ),
        "class20_vs_standalone_tail": _mask_relation(
            tail, combined_masks[PANCREAS_TAIL]
        ),
    }

    overlaps = {
        "body_and_head": int(np.count_nonzero(body & head)),
        "body_and_tail": int(np.count_nonzero(body & tail)),
        "head_and_tail": int(np.count_nonzero(head & tail)),
        "lesion_inside_pancreas": int(np.count_nonzero(lesion & pancreas)),
        "lesion_outside_pancreas": int(np.count_nonzero(lesion & ~pancreas)),
        "parts_outside_pancreas": int(np.count_nonzero(parts_union & ~pancreas)),
        "pancreas_not_in_parts": int(np.count_nonzero(pancreas & ~parts_union)),
    }

    findings: list[str] = []
    if not all(item["same_geometry"] for item in geometry_matches.values()):
        findings.append("At least one label does not share the CT voxel geometry.")
    if not _is_integer_valued(combined):
        findings.append("combined_labels contains non-integer voxel values.")
    if unexpected_values:
        findings.append(
            f"combined_labels contains undocumented values: {unexpected_values}."
        )
    if standalone_masks["pancreatic_lesion"]["foreground_voxels"] == 0:
        findings.append(
            "The standalone lesion mask is empty; class-28 equality is "
            "therefore an empty-mask check, not positive-lesion validation."
        )
    if not relations["class28_vs_standalone_lesion"]["equal"]:
        findings.append("Combined class 28 differs from the standalone lesion mask.")
    if not relations["classes17_20_vs_standalone_pancreas"]["equal"]:
        findings.append(
            "The union of combined labels 17-20 is not exactly the standalone "
            "whole-pancreas mask; other combined classes occupy the missing "
            f"pancreas voxels: {pancreas_family_gap_labels}."
        )
    if relations["class17_vs_pancreas_not_in_parts"]["equal"]:
        findings.append(
            "Combined class 17 exactly represents whole-pancreas voxels not "
            "assigned to standalone head, body, or tail."
        )
    if any(overlaps[name] for name in ("body_and_head", "body_and_tail", "head_and_tail")):
        findings.append("Standalone head/body/tail masks overlap each other.")
    if overlaps["parts_outside_pancreas"]:
        findings.append("Some standalone head/body/tail voxels lie outside pancreas.nii.gz.")

    return {
        "case_id": case_id,
        "split": split,
        "paths": {name: str(path) for name, path in paths.items()},
        "ct": {
            "geometry": get_geometry(ct_image),
            "loaded_dtype": str(ct.dtype),
            "finite_voxels": int(finite_ct.size),
            "nonfinite_voxels": int(ct.size - finite_ct.size),
            "intensity": {
                "min": float(np.min(finite_ct)),
                "max": float(np.max(finite_ct)),
                "mean": float(np.mean(finite_ct)),
                "p01": float(np.percentile(finite_ct, 1)),
                "p50": float(np.percentile(finite_ct, 50)),
                "p99": float(np.percentile(finite_ct, 99)),
            },
        },
        "combined_labels": {
            "geometry": get_geometry(combined_image),
            "loaded_dtype": str(combined.dtype),
            "integer_valued": _is_integer_valued(combined),
            "unique_values": combined_unique,
            "unexpected_values": unexpected_values,
            "pancreas_class_counts": {
                class_id: int(combined_masks[class_id].sum())
                for class_id in (*PANCREAS_FAMILY, PANCREATIC_LESION)
            },
            "labels_inside_pancreas_outside_17_20": pancreas_family_gap_labels,
        },
        "geometry_matches": geometry_matches,
        "standalone_masks": standalone_masks,
        "relations": relations,
        "overlaps": overlaps,
        "findings": findings,
    }


def format_inspection(report: dict[str, Any]) -> str:
    """Format an inspection report for a human-readable terminal session."""

    ct = report["ct"]
    combined = report["combined_labels"]
    lines = [
        "PanTS first-case inspection",
        "===========================",
        f"Case:  {report['case_id']}",
        f"Split: {report['split']}",
        "",
        "CT volume",
        f"  shape:           {ct['geometry']['shape']}",
        f"  stored dtype:    {ct['geometry']['dtype']}",
        f"  loaded dtype:    {ct['loaded_dtype']}",
        f"  spacing (mm):    {ct['geometry']['spacing_mm']}",
        f"  orientation:     {ct['geometry']['orientation']}",
        f"  finite voxels:   {ct['finite_voxels']}",
        f"  nonfinite:       {ct['nonfinite_voxels']}",
        "  affine (voxel index -> physical coordinates):",
        *_format_affine(ct["geometry"]["affine"]),
        "  intensity statistics (stored CT values; no display window applied):",
        f"    min/mean/max:  {ct['intensity']['min']:.3f} / "
        f"{ct['intensity']['mean']:.3f} / {ct['intensity']['max']:.3f}",
        f"    p01/p50/p99:   {ct['intensity']['p01']:.3f} / "
        f"{ct['intensity']['p50']:.3f} / {ct['intensity']['p99']:.3f}",
        "",
        "Combined labels",
        f"  shape:           {combined['geometry']['shape']}",
        f"  stored dtype:    {combined['geometry']['dtype']}",
        f"  loaded dtype:    {combined['loaded_dtype']}",
        f"  integer-valued:  {combined['integer_valued']}",
        f"  unique values:   {combined['unique_values']}",
        f"  undocumented:    {combined['unexpected_values']}",
        "",
        "Geometry relative to CT",
    ]

    for name, comparison in report["geometry_matches"].items():
        lines.append(
            f"  {name:20s} same={comparison['same_geometry']} "
            f"shape={comparison['same_shape']} spacing={comparison['same_spacing']} "
            f"orientation={comparison['same_orientation']} "
            f"affine={comparison['same_affine']}"
        )

    lines.extend(["", "Standalone masks"])
    for name, details in report["standalone_masks"].items():
        lines.append(
            f"  {name:20s} voxels={details['foreground_voxels']:8d} "
            f"stored={details['stored_dtype']} loaded={details['loaded_dtype']}"
        )
        lines.append(
            f"    raw_values={details['raw_unique_values']} "
            f"scaled_values={details['unique_values']} "
            f"slope={details['slope']:.12g} intercept={details['intercept']:.12g} "
            f"binary_after_scaling={details['binary_after_scaling']}"
        )

    lines.extend(["", "Combined pancreas/lesion class voxel counts"])
    for class_id, count in combined["pancreas_class_counts"].items():
        lines.append(f"  class {class_id:2d}: {count:8d}  {CLASS_MAP[class_id]}")
    lines.append(
        "  labels inside standalone pancreas but outside classes 17-20: "
        f"{combined['labels_inside_pancreas_outside_17_20']}"
    )

    lines.extend(["", "Exact mask relationships"])
    for name, relation in report["relations"].items():
        lines.append(
            f"  {name}: equal={relation['equal']}, "
            f"intersection={relation['intersection_voxels']}, "
            f"standalone/reference-only={relation['reference_only_voxels']}, "
            f"combined/candidate-only={relation['candidate_only_voxels']}"
        )

    lines.extend(["", "Standalone overlap checks"])
    for name, count in report["overlaps"].items():
        lines.append(f"  {name:24s} {count}")

    lines.extend(["", "Interpretation flags"])
    if report["findings"]:
        lines.extend(f"  - {finding}" for finding in report["findings"])
    else:
        lines.append("  - No flagged inconsistency in the requested checks.")

    return "\n".join(lines)
