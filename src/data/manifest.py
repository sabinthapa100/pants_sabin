"""Authoritative PanTS-tr case manifest and the fixed development split.

The manifest is the single source of truth for which cases exist, where they
live relative to ``PANTS_DATA_ROOT``, and which of them contain a pancreatic
lesion. The split is derived from the manifest alone, so both artifacts are
reproducible from the raw dataset plus a fixed seed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .io import compare_geometry, load_nifti
from .labels import CLASS_MAP, PANCREATIC_LESION
from .paths import get_case_paths, get_data_root, get_relative_case_paths, list_cases


logger = logging.getLogger(__name__)

CASE_ID_PREFIX = "PanTS_"

# PanTS-tr is PanTS_00000001..PanTS_00009000; PanTS-te starts at 00009001.
TRAIN_INDEX_MIN = 1
TRAIN_INDEX_MAX = 9000
EXPECTED_TRAIN_CASES = 9000

VALID_LABEL_VALUES = frozenset({0, *CLASS_MAP})

MANIFEST_VERSION = "pants_tr_v1"
SPLIT_VERSION = "pants_cv_v1"

DEFAULT_SEED = 317
DEFAULT_FOLDS = 5


def parse_case_index(case_id: str) -> int:
    """Return the integer index encoded in a PanTS case identifier."""

    if not case_id.startswith(CASE_ID_PREFIX):
        raise ValueError(f"Case identifier lacks '{CASE_ID_PREFIX}' prefix: {case_id}")

    suffix = case_id[len(CASE_ID_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(f"Case identifier has a non-numeric index: {case_id}")

    return int(suffix)


def is_train_case(case_id: str) -> bool:
    """Return whether a case identifier belongs to PanTS-tr."""

    try:
        index = parse_case_index(case_id)
    except ValueError:
        return False

    return TRAIN_INDEX_MIN <= index <= TRAIN_INDEX_MAX


def scan_case(payload: tuple[str, str | None]) -> dict[str, Any]:
    """
    Read one PanTS-tr case and return its manifest entry.

    The CT is opened for its header only; the combined label is decompressed
    because lesion voxels can only be counted from the voxel data. The 28
    standalone masks are never read.
    """

    case_id, root = payload
    entry: dict[str, Any] = {"case_id": case_id, "problems": []}

    paths = get_case_paths(case_id, "train", root)
    relative = get_relative_case_paths(case_id, "train")
    entry["ct"] = relative["ct"]
    entry["label"] = relative["label"]

    if not paths["ct"].exists():
        entry["problems"].append("missing_ct")
    if not paths["combined"].exists():
        entry["problems"].append("missing_label")
    if entry["problems"]:
        return entry

    # The CT is opened for its header only. `load_nifti` would materialize the
    # voxel array, which means decompressing all 296 GB of ImageTr just to read
    # shape/spacing/affine - ruinous locally and unusable against Drive in
    # Colab. Only the ~0.9 MB label is actually decoded, because lesion voxels
    # cannot be counted from a header.
    ct_image = nib.load(str(paths["ct"]))
    label_image, label_array = load_nifti(paths["combined"])

    entry["shape"] = [int(size) for size in ct_image.shape]
    entry["spacing"] = [round(float(value), 6) for value in ct_image.header.get_zooms()[:3]]
    entry["orientation"] = "".join(nib.aff2axcodes(ct_image.affine))

    # Shape and spacing must agree: they establish that the label array is
    # voxel-for-voxel comparable with the CT. A disagreement there is fatal.
    #
    # The world-coordinate affine is treated separately. In 78 of the 9,000
    # PanTS-tr cases the combined_labels affine is degenerate (translation
    # zeroed, direction reset to LPS) even though the voxel array is stored
    # index-aligned with the CT. That was verified empirically: index-aligned
    # labels put liver at 15-143 HU and lung near -761 HU in all 78 cases,
    # whereas honouring the label affine mirrors the volume and yields
    # implausible values (liver median -113 HU). The quirk is therefore
    # recorded and corrected at load time, not treated as corruption.
    geometry = compare_geometry(ct_image, label_image)
    if not geometry["same_shape"]:
        entry["problems"].append("shape_mismatch")
    if not geometry["same_spacing"]:
        entry["problems"].append("spacing_mismatch")

    entry["label_affine_matches_ct"] = bool(geometry["same_affine"])
    entry["label_orientation_matches_ct"] = bool(geometry["same_orientation"])

    values = np.unique(label_array)
    if not np.all(np.isfinite(values)):
        entry["problems"].append("label_non_finite")
    elif not np.all(values == np.rint(values)):
        entry["problems"].append("label_non_integer")
    else:
        unexpected = sorted(int(value) for value in values if int(value) not in VALID_LABEL_VALUES)
        if unexpected:
            entry["problems"].append(f"label_out_of_range:{unexpected}")

    lesion_voxels = int(np.count_nonzero(label_array == PANCREATIC_LESION))
    entry["lesion_present"] = lesion_voxels > 0
    entry["lesion_voxel_count"] = lesion_voxels

    return entry


def _read_metadata_tumor_flags(root: Path) -> dict[str, int]:
    """
    Return the metadata ``tumor?`` flag per case, if metadata.xlsx is readable.

    This flag is recorded for QC comparison only. Ground truth for
    ``lesion_present`` is always the presence of class 28 in combined_labels.
    """

    metadata_path = root / "metadata.xlsx"
    if not metadata_path.exists():
        logger.warning("metadata.xlsx not found at %s; tumor flags omitted", metadata_path)
        return {}

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable; metadata tumor flags omitted")
        return {}

    try:
        frame = pd.read_excel(metadata_path, usecols=["PanTS ID", "tumor?"])
    except (ValueError, KeyError, OSError) as error:
        logger.warning("Could not read metadata.xlsx (%s); tumor flags omitted", error)
        return {}

    return {
        str(case_id): int(flag)
        for case_id, flag in zip(frame["PanTS ID"], frame["tumor?"], strict=True)
        if not pd.isna(flag)
    }


def build_manifest(
    root: str | Path | None = None,
    workers: int = 8,
    case_ids: Iterable[str] | None = None,
    expected_cases: int | None = EXPECTED_TRAIN_CASES,
) -> dict[str, Any]:
    """
    Scan PanTS-tr and return the manifest.

    Parameters
    ----------
    case_ids:
        Restrict the scan to these cases. Passing a subset relaxes the
        case-count check, since a subset is legitimately not the full dataset.
    expected_cases:
        Required number of cases for a full scan. Defaults to the official
        9,000 PanTS-tr cases; pass ``None`` to skip the check.

    Raises
    ------
    ValueError:
        If validation fails, listing every offending case.
    """

    data_root = get_data_root(root)
    root_argument = str(data_root)

    is_full_scan = case_ids is None
    cases = list_cases("train", data_root) if is_full_scan else sorted(case_ids)

    foreign = sorted(case for case in cases if not is_train_case(case))
    if foreign:
        raise ValueError(
            f"{len(foreign)} case(s) outside PanTS-tr "
            f"({CASE_ID_PREFIX}{TRAIN_INDEX_MIN:08d}..{CASE_ID_PREFIX}{TRAIN_INDEX_MAX:08d}) "
            f"were found: {foreign[:10]}"
        )

    logger.info("Scanning %d PanTS-tr cases with %d workers", len(cases), workers)
    payloads = [(case_id, root_argument) for case_id in cases]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            entries = list(executor.map(scan_case, payloads, chunksize=16))
    else:
        entries = [scan_case(payload) for payload in payloads]

    tumor_flags = _read_metadata_tumor_flags(data_root)
    for entry in entries:
        entry["metadata_tumor_flag"] = tumor_flags.get(entry["case_id"])

    # A full scan must account for every expected PanTS-tr case; an explicit
    # subset (tests, pilots) is validated in every other respect.
    validate_manifest_entries(
        entries,
        expected_cases=expected_cases if is_full_scan else None,
    )

    for entry in entries:
        entry.pop("problems", None)

    positives = sum(1 for entry in entries if entry["lesion_present"])
    disagreements = sum(
        1
        for entry in entries
        if entry["metadata_tumor_flag"] is not None
        and bool(entry["metadata_tumor_flag"]) != entry["lesion_present"]
    )
    affine_quirks = sum(1 for entry in entries if not entry["label_affine_matches_ct"])

    return {
        "meta": {
            "version": MANIFEST_VERSION,
            "split": "PanTS-tr",
            "case_count": len(entries),
            "lesion_positive": positives,
            "lesion_negative": len(entries) - positives,
            "metadata_tumor_flag_disagreements": disagreements,
            "label_affine_mismatch_cases": affine_quirks,
            "lesion_present_definition": (
                "presence of class 28 in combined_labels.nii.gz; "
                "metadata_tumor_flag is recorded for QC only and is never ground truth"
            ),
            "label_affine_note": (
                "Some combined_labels files carry a degenerate affine (zeroed "
                "translation, direction reset to LPS) while their voxel array is "
                "index-aligned with the CT. Shape and spacing always agree. The "
                "loader copies the CT affine onto the label so reorientation "
                "cannot mirror the annotation."
            ),
            "paths_relative_to": "PANTS_DATA_ROOT",
        },
        "cases": entries,
    }


def validate_manifest_entries(
    entries: list[dict[str, Any]],
    expected_cases: int | None = EXPECTED_TRAIN_CASES,
) -> None:
    """Raise ValueError describing every manifest problem found."""

    failures: list[str] = []

    case_ids = [entry["case_id"] for entry in entries]
    duplicates = sorted(case for case, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        failures.append(f"duplicate case identifiers: {duplicates[:10]}")

    if expected_cases is not None and len(entries) != expected_cases:
        failures.append(f"expected {expected_cases} cases, found {len(entries)}")

    non_train = sorted(case for case in case_ids if not is_train_case(case))
    if non_train:
        failures.append(f"non-PanTS-tr identifiers present: {non_train[:10]}")

    broken = [(entry["case_id"], entry["problems"]) for entry in entries if entry.get("problems")]
    if broken:
        preview = "; ".join(f"{case}={problems}" for case, problems in broken[:10])
        failures.append(f"{len(broken)} case(s) failed per-case validation: {preview}")

    if failures:
        raise ValueError("Manifest validation failed: " + " | ".join(failures))


def build_split(
    manifest: dict[str, Any],
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
) -> dict[str, Any]:
    """
    Build a deterministic, lesion-stratified k-fold split over PanTS-tr.

    The released PanTS metadata exposes no patient or study-group identifier
    (every ``PanTS ID`` is unique), so patient-level grouping cannot be
    enforced. The split is therefore case-level, and this is recorded in the
    output rather than left implicit.
    """

    if folds < 2:
        raise ValueError(f"folds must be at least 2; got {folds}")

    entries = manifest["cases"]
    if len(entries) < folds:
        raise ValueError(f"Cannot build {folds} folds from {len(entries)} cases")

    positives = sorted(entry["case_id"] for entry in entries if entry["lesion_present"])
    negatives = sorted(entry["case_id"] for entry in entries if not entry["lesion_present"])

    generator = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for stratum in (positives, negatives):
        order = generator.permutation(len(stratum))
        for position, index in enumerate(order):
            assignment[stratum[index]] = position % folds

    all_cases = sorted(assignment)
    fold_definitions = []
    for fold in range(folds):
        validation = [case for case in all_cases if assignment[case] == fold]
        training = [case for case in all_cases if assignment[case] != fold]
        fold_definitions.append({"train": training, "val": validation})

    lesion_lookup = {entry["case_id"]: entry["lesion_present"] for entry in entries}
    fold_summary = [
        {
            "fold": fold,
            "train": len(definition["train"]),
            "val": len(definition["val"]),
            "val_lesion_positive": sum(1 for case in definition["val"] if lesion_lookup[case]),
        }
        for fold, definition in enumerate(fold_definitions)
    ]

    return {
        "meta": {
            "version": SPLIT_VERSION,
            "source_manifest": manifest["meta"]["version"],
            "seed": seed,
            "folds": folds,
            "split_level": "case",
            "grouping_note": (
                "PanTS metadata exposes no patient or study identifier "
                "(all PanTS IDs are unique), so patient-level grouping cannot "
                "be enforced from released data"
            ),
            "stratified_by": "lesion_present (class 28 in combined_labels)",
            "population": "PanTS-tr only; PanTS-te is never used for development",
            "fold_summary": fold_summary,
        },
        "folds": fold_definitions,
    }


def to_nnunet_splits(split: dict[str, Any]) -> list[dict[str, list[str]]]:
    """
    Convert our split into nnU-Net's native ``splits_final.json`` structure.

    nnU-Net expects a bare list of ``{"train": [...], "val": [...]}`` objects.
    Emitting it from the same assignment keeps every experiment on identical
    case partitions instead of introducing a second split policy.
    """

    return [
        {"train": list(fold["train"]), "val": list(fold["val"])}
        for fold in split["folds"]
    ]


def write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON atomically so an interrupted run cannot leave a partial file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
