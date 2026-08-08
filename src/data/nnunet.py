"""Prepare and validate a small nnU-Net v2 dataset from immutable PanTS data."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .io import compare_geometry, load_nifti
from .labels import BACKGROUND, CLASS_MAP, PANCREATIC_LESION
from .paths import PROJECT_ROOT, get_case_paths, list_cases


NNUNET_ROOT = PROJECT_ROOT / "nnunet"
NNUNET_RAW = NNUNET_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_ROOT / "nnUNet_results"


@dataclass(frozen=True)
class CaseLabelSummary:
    """Unique semantic labels and lesion status for one PanTS-tr case."""

    case_id: str
    label_ids: frozenset[int]

    @property
    def foreground_labels(self) -> frozenset[int]:
        return self.label_ids - {BACKGROUND}

    @property
    def lesion_positive(self) -> bool:
        return PANCREATIC_LESION in self.label_ids


def _read_integer_label_ids(path: Path) -> tuple[Any, frozenset[int]]:
    """Load a segmentation and return its verified integer label IDs."""

    image, array = load_nifti(path)
    unique_values = np.unique(array)
    if not np.all(np.isfinite(unique_values)):
        raise ValueError(f"Segmentation contains nonfinite values: {path}")
    if not np.allclose(
        unique_values,
        np.rint(unique_values),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"Segmentation contains non-integer values: {path}")

    label_ids = frozenset(int(value) for value in unique_values)
    expected_ids = {BACKGROUND, *CLASS_MAP}
    unexpected = sorted(label_ids - expected_ids)
    if unexpected:
        raise ValueError(
            f"Segmentation contains undocumented label IDs {unexpected}: {path}"
        )
    return image, label_ids


def _discover_candidates(
    max_cases: int,
) -> tuple[list[CaseLabelSummary], dict[str, Any]]:
    """Scan sorted PanTS-tr labels until coverage and class-balance needs are met."""

    if not 20 <= max_cases <= 50:
        raise ValueError(
            "--max-cases must be between 20 and 50 for this smoke dataset."
        )

    expected_foreground = set(CLASS_MAP)
    target_positive = max_cases // 2
    target_negative = max_cases - target_positive
    candidates: list[CaseLabelSummary] = []
    observed: set[int] = set()
    positive = 0
    negative = 0

    for case_id in list_cases("train"):
        label_path = get_case_paths(case_id, "train")["combined"]
        _, label_ids = _read_integer_label_ids(label_path)
        summary = CaseLabelSummary(case_id=case_id, label_ids=label_ids)
        candidates.append(summary)
        observed.update(summary.foreground_labels)
        if summary.lesion_positive:
            positive += 1
        else:
            negative += 1

        if (
            observed == expected_foreground
            and positive >= target_positive
            and negative >= target_negative
        ):
            break

    missing = sorted(expected_foreground - observed)
    if missing or positive < target_positive or negative < target_negative:
        raise ValueError(
            "PanTS-tr candidate scan could not satisfy the smoke-subset "
            f"requirements. Missing labels={missing}, positive={positive}/"
            f"{target_positive}, negative={negative}/{target_negative}."
        )

    return candidates, {
        "cases_scanned": len(candidates),
        "last_case_scanned": candidates[-1].case_id,
        "candidate_positive": positive,
        "candidate_negative": negative,
    }


def _coverage_core(
    candidates: list[CaseLabelSummary],
) -> list[CaseLabelSummary]:
    """Greedily cover labels 1-28, breaking equal-gain ties by case ID."""

    uncovered = set(CLASS_MAP)
    remaining = list(candidates)
    selected: list[CaseLabelSummary] = []

    while uncovered:
        gains = [
            len(candidate.foreground_labels & uncovered)
            for candidate in remaining
        ]
        best_gain = max(gains, default=0)
        if best_gain == 0:
            raise ValueError(
                f"Candidate cases cannot cover labels {sorted(uncovered)}."
            )
        best_index = gains.index(best_gain)
        best = remaining.pop(best_index)
        selected.append(best)
        uncovered.difference_update(best.foreground_labels)

    return selected


def select_smoke_cases(
    max_cases: int,
) -> tuple[list[CaseLabelSummary], dict[str, Any]]:
    """Select a deterministic, balanced, full-label-coverage smoke subset."""

    candidates, scan_report = _discover_candidates(max_cases)
    selected = _coverage_core(candidates)
    if len(selected) > max_cases:
        raise ValueError(
            f"Full label coverage needs {len(selected)} cases, exceeding "
            f"--max-cases={max_cases}."
        )

    coverage_core_ids = [case.case_id for case in selected]
    selected_ids = set(coverage_core_ids)
    target_positive = max_cases // 2
    target_negative = max_cases - target_positive

    for desired_status, target in (
        (True, target_positive),
        (False, target_negative),
    ):
        while (
            sum(case.lesion_positive == desired_status for case in selected)
            < target
            and len(selected) < max_cases
        ):
            next_case = next(
                case
                for case in candidates
                if case.case_id not in selected_ids
                and case.lesion_positive == desired_status
            )
            selected.append(next_case)
            selected_ids.add(next_case.case_id)

    while len(selected) < max_cases:
        positive_count = sum(case.lesion_positive for case in selected)
        negative_count = len(selected) - positive_count
        desired_status = positive_count < negative_count
        matching = [
            case
            for case in candidates
            if case.case_id not in selected_ids
            and case.lesion_positive == desired_status
        ]
        pool = matching or [
            case for case in candidates if case.case_id not in selected_ids
        ]
        if not pool:
            raise ValueError(
                f"Only {len(selected)} eligible cases were available, fewer "
                f"than --max-cases={max_cases}."
            )
        selected.append(pool[0])
        selected_ids.add(pool[0].case_id)

    selected.sort(key=lambda case: case.case_id)
    coverage = set().union(
        *(case.foreground_labels for case in selected)
    )
    if coverage != set(CLASS_MAP):
        raise AssertionError("Selected cases lost required label coverage.")

    return selected, {
        **scan_report,
        "coverage_core_ids": coverage_core_ids,
        "selected_positive": sum(case.lesion_positive for case in selected),
        "selected_negative": sum(
            not case.lesion_positive for case in selected
        ),
        "covered_label_ids": sorted(coverage),
    }


def _validate_selected_sources(cases: list[CaseLabelSummary]) -> None:
    """Validate every selected source CT/label pair before creating output."""

    for case in cases:
        paths = get_case_paths(case.case_id, "train")
        ct_image, ct = load_nifti(paths["ct"])
        label_image, label_ids = _read_integer_label_ids(paths["combined"])
        if ct.ndim != 3 or label_image.ndim != 3:
            raise ValueError(
                f"Expected 3D CT and label for {case.case_id}; got "
                f"{ct.shape} and {label_image.shape}."
            )
        geometry = compare_geometry(ct_image, label_image)
        if not geometry["same_geometry"]:
            raise ValueError(
                f"CT/label geometry mismatch for {case.case_id}: {geometry}"
            )
        if label_ids != case.label_ids:
            raise ValueError(
                f"Labels changed during validation for {case.case_id}."
            )


def _dataset_folder_name(dataset_id: int, name: str) -> str:
    if not 1 <= dataset_id <= 999:
        raise ValueError("Dataset ID must be an integer from 1 through 999.")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Dataset name must be one safe directory-name component.")
    return f"Dataset{dataset_id:03d}_{name}"


def _create_relative_symlink(source: Path, destination: Path) -> None:
    """Create a relative link and verify that it resolves to the source."""

    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    relative_target = os.path.relpath(source, start=destination.parent.resolve())
    destination.symlink_to(relative_target)
    if not destination.is_symlink() or destination.resolve() != source:
        raise OSError(f"Symlink validation failed: {destination} -> {source}")


def _write_dataset_json(dataset_dir: Path, num_training: int) -> dict[str, Any]:
    labels = {"background": BACKGROUND}
    labels.update({name: class_id for class_id, name in CLASS_MAP.items()})
    content = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return content


def _validate_created_dataset(
    dataset_dir: Path,
    cases: list[CaseLabelSummary],
    expected_json: dict[str, Any],
) -> dict[str, Any]:
    """Validate layout, names, links, JSON, and one nnU-Net reader round trip."""

    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    expected_images = {f"{case.case_id}_0000.nii.gz" for case in cases}
    expected_labels = {f"{case.case_id}.nii.gz" for case in cases}
    actual_images = {path.name for path in images_dir.iterdir()}
    actual_labels = {path.name for path in labels_dir.iterdir()}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise ValueError("Created nnU-Net filenames do not match selected cases.")

    with (dataset_dir / "dataset.json").open(encoding="utf-8") as handle:
        actual_json = json.load(handle)
    if actual_json != expected_json:
        raise ValueError("Written dataset.json does not match requested metadata.")

    for case in cases:
        source_paths = get_case_paths(case.case_id, "train")
        image_link = images_dir / f"{case.case_id}_0000.nii.gz"
        label_link = labels_dir / f"{case.case_id}.nii.gz"
        expected_targets = (
            (image_link, source_paths["ct"]),
            (label_link, source_paths["combined"]),
        )
        for link, source in expected_targets:
            if not link.is_symlink() or not link.exists():
                raise ValueError(f"Missing or broken symlink: {link}")
            if link.resolve() != source.resolve(strict=True):
                raise ValueError(f"Symlink points to the wrong source: {link}")

    first = cases[0].case_id
    first_image = images_dir / f"{first}_0000.nii.gz"
    first_label = labels_dir / f"{first}.nii.gz"
    from nnunetv2.imageio.reader_writer_registry import (
        determine_reader_writer_from_dataset_json,
    )

    reader_class = determine_reader_writer_from_dataset_json(
        actual_json,
        example_file=str(first_image),
        verbose=False,
    )
    reader = reader_class()
    image_array, _ = reader.read_images((str(first_image),))
    label_array, _ = reader.read_seg(str(first_label))
    if image_array.shape != label_array.shape:
        raise ValueError(
            "nnU-Net reader returned different image/label shapes for "
            f"{first}: {image_array.shape} versus {label_array.shape}."
        )

    return {
        "image_links": len(actual_images),
        "label_links": len(actual_labels),
        "nnunet_reader": reader_class.__name__,
        "reader_test_case": first,
        "reader_image_shape": tuple(image_array.shape),
        "reader_label_shape": tuple(label_array.shape),
    }


def prepare_nnunet_smoke_dataset(
    dataset_id: int = 501,
    name: str = "PanTSSmoke",
    max_cases: int = 40,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create and fully validate a symlink-based PanTS-tr smoke dataset."""

    folder_name = _dataset_folder_name(dataset_id, name)
    dataset_dir = NNUNET_RAW / folder_name
    raw_collisions = [
        path
        for path in NNUNET_RAW.glob(f"Dataset{dataset_id:03d}_*")
        if path != dataset_dir
    ]
    if raw_collisions:
        raise FileExistsError(
            f"Dataset ID {dataset_id:03d} is already used in nnUNet_raw by: "
            f"{raw_collisions}"
        )
    downstream_collisions = [
        path
        for root in (NNUNET_PREPROCESSED, NNUNET_RESULTS)
        for path in root.glob(f"Dataset{dataset_id:03d}_*")
    ]
    if downstream_collisions:
        raise FileExistsError(
            f"Dataset ID {dataset_id:03d} has existing preprocessed/results "
            f"artifacts: {downstream_collisions}. Refusing to create stale or "
            "ambiguous state."
        )
    if dataset_dir.exists() or dataset_dir.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {dataset_dir}. Use --overwrite "
                "only if you intentionally want to replace this derived dataset."
            )
        if dataset_dir.is_symlink() or not dataset_dir.is_dir():
            raise ValueError(
                f"Refusing to overwrite a non-directory dataset path: {dataset_dir}"
            )

    staging_dir = NNUNET_RAW / f".{folder_name}.staging"
    backup_dir = NNUNET_RAW / f".{folder_name}.backup"
    for temporary_path in (staging_dir, backup_dir):
        if temporary_path.exists() or temporary_path.is_symlink():
            if not overwrite:
                raise FileExistsError(
                    f"Temporary dataset path already exists: {temporary_path}. "
                    "Inspect it before using --overwrite."
                )
            if temporary_path.is_symlink() or not temporary_path.is_dir():
                raise ValueError(
                    "Refusing to replace a non-directory temporary path: "
                    f"{temporary_path}"
                )

    cases, selection_report = select_smoke_cases(max_cases=max_cases)
    _validate_selected_sources(cases)

    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)
    for temporary_path in (staging_dir, backup_dir):
        if temporary_path.exists():
            shutil.rmtree(temporary_path)

    try:
        images_dir = staging_dir / "imagesTr"
        labels_dir = staging_dir / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir()

        for case in cases:
            source_paths = get_case_paths(case.case_id, "train")
            _create_relative_symlink(
                source_paths["ct"],
                images_dir / f"{case.case_id}_0000.nii.gz",
            )
            _create_relative_symlink(
                source_paths["combined"],
                labels_dir / f"{case.case_id}.nii.gz",
            )

        dataset_json = _write_dataset_json(staging_dir, len(cases))
        validation_report = _validate_created_dataset(
            staging_dir,
            cases,
            dataset_json,
        )

        if dataset_dir.exists():
            dataset_dir.rename(backup_dir)
        try:
            staging_dir.rename(dataset_dir)
        except Exception:
            if backup_dir.exists() and not dataset_dir.exists():
                backup_dir.rename(dataset_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if staging_dir.exists() and staging_dir.is_dir():
            shutil.rmtree(staging_dir)
        raise

    return {
        "dataset_id": dataset_id,
        "dataset_name": name,
        "dataset_dir": str(dataset_dir.resolve()),
        "cases": cases,
        "selection": selection_report,
        "validation": validation_report,
        "nnunet_paths": {
            "nnUNet_raw": str(NNUNET_RAW.resolve()),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED.resolve()),
            "nnUNet_results": str(NNUNET_RESULTS.resolve()),
        },
    }
