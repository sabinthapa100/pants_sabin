"""The on-disk form of a deterministically preprocessed PanTS case.

This module owns *storage*, not *rules*. Every preprocessing decision - affine
repair, RAS orientation, 1.5 mm resampling, the [-175, 250] HU window - lives
in :mod:`transforms` and is imported from there. Nothing here reimplements it,
so the cache can never drift from the raw-NIfTI path used for inference.

The pipeline this module sits in the middle of::

    raw ct.nii.gz            int16 Hounsfield units, native anisotropic lattice
      -> MONAI load          float32, channel-first
      -> affine repair       label borrows the CT's world transform
      -> RAS                 axis permutation/flip, lossless
      -> 1.5 mm resample     bilinear image / nearest label
      -> [-175,250] -> [0,1] clipped linear rescale, float32
      -> STORAGE             image float16, label uint8, zlib-compressed npz
      -> training            cast back to float32, crop, augment

Storage quantization is the only lossy step introduced here, and it is applied
*after* the intensity window. Because the signal has already been clipped to a
425 HU range and mapped to [0, 1], float16 (11-bit significand) resolves it to
a worst case of 0.104 HU - roughly two orders of magnitude below CT noise.

The cache is DERIVED data. It is rebuildable from the raw NIfTIs plus this
repository at any time, contains no geometry, and is never used for inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import zipfile

import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import Compose, EnsureTyped, MapTransform

from .labels import CLASS_MAP
from .paths import get_case_paths
from .transforms import (
    HU_MAX,
    HU_MIN,
    ORIENTATION,
    PATCH_SIZE,
    TARGET_SPACING,
    augmentation_transforms,
    deterministic_transforms,
    sampling_transforms,
)


logger = logging.getLogger(__name__)


CASES_DIRNAME = "cases"
METADATA_FILENAME = "preprocessing.json"
IMAGE_KEY = "image"
LABEL_KEY = "label"
IMAGE_STORAGE_DTYPE = np.float16
LABEL_STORAGE_DTYPE = np.uint8
MAX_CLASS = max(CLASS_MAP)

# float16 has an 11-bit significand, so the worst-case relative step over
# [0, 1] is 2^-11. Expressed in Hounsfield units over the stored window:
FLOAT16_HU_TOLERANCE = (HU_MAX - HU_MIN) * 2.0**-11


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #


def case_path(prepared_root: str | Path, case_id: str) -> Path:
    """Location of one prepared case. The filename carries the case identity."""
    return Path(prepared_root) / CASES_DIRNAME / f"{case_id}.npz"


def validate_arrays(image: np.ndarray, label: np.ndarray) -> None:
    """Reject anything that would poison training, loudly.

    Checks the properties that silent corruption would violate: storage dtype,
    finiteness, the [0, 1] contract the network's first layer assumes, exact
    integer class identity, and voxel-grid agreement between the two arrays.
    """
    if image.dtype != IMAGE_STORAGE_DTYPE:
        raise ValueError(f"image dtype {image.dtype}, expected {IMAGE_STORAGE_DTYPE}")
    if label.dtype != LABEL_STORAGE_DTYPE:
        raise ValueError(f"label dtype {label.dtype}, expected {LABEL_STORAGE_DTYPE}")
    if image.shape != label.shape:
        raise ValueError(f"image shape {image.shape} != label shape {label.shape}")
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {image.shape}")

    as_float = image.astype(np.float32)
    if not np.isfinite(as_float).all():
        raise ValueError("image contains non-finite values")
    if as_float.min() < 0.0 or as_float.max() > 1.0:
        raise ValueError(f"image outside [0, 1]: [{as_float.min()}, {as_float.max()}]")

    highest = int(label.max())
    if highest > MAX_CLASS:
        raise ValueError(f"label value {highest} outside 0..{MAX_CLASS}")


def read_prepared_case(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one prepared case. Never unpickles: the archive holds plain arrays."""
    with np.load(str(path), allow_pickle=False) as archive:
        return archive[IMAGE_KEY], archive[LABEL_KEY]


def write_prepared_case(
    path: str | Path,
    image: np.ndarray,
    label: np.ndarray,
) -> int:
    """Write one case atomically: temporary file -> read back -> validate -> rename.

    A partially written or corrupt archive never appears at the final path, so
    an interrupted run leaves behind only complete, valid cases and the resume
    logic can trust what it finds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Must still end in .npz: np.savez_compressed silently appends the
    # extension otherwise, and we would validate and rename the wrong file.
    temporary = path.with_suffix(".tmp.npz")

    try:
        np.savez_compressed(temporary, **{IMAGE_KEY: image, LABEL_KEY: label})

        # Force the bytes to the platter BEFORE the rename. ext4's default
        # data=ordered mode can otherwise make the rename durable while the
        # file contents are still in the page cache, so a power loss would
        # leave a correctly-named but truncated archive - the one corruption
        # our resume logic could not distinguish from a good file.
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())

        written_image, written_label = read_prepared_case(temporary)
        validate_arrays(written_image, written_label)
        if not np.array_equal(written_label, label):
            raise ValueError("label did not survive the round trip")
        size = temporary.stat().st_size

        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)  # make the rename itself durable
        finally:
            os.close(directory)
        return size
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def is_case_complete(path: str | Path, deep: bool = False) -> bool:
    """Has this case already been prepared successfully?

    The structural check (default) confirms both arrays are present with the
    right dtypes and matching shapes, reading only archive headers. ``deep``
    additionally decompresses and revalidates every voxel.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        if deep:
            validate_arrays(*read_prepared_case(path))
            return True
        with np.load(str(path), allow_pickle=False) as archive:
            if set(archive.files) != {IMAGE_KEY, LABEL_KEY}:
                return False
            image, label = archive[IMAGE_KEY], archive[LABEL_KEY]
            return (
                image.dtype == IMAGE_STORAGE_DTYPE
                and label.dtype == LABEL_STORAGE_DTYPE
                and image.shape == label.shape
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # An npz IS a zip archive. A truncated one raises BadZipFile, which
        # descends from Exception rather than OSError, so it has to be named
        # explicitly or a half-written file would crash the resume scan
        # instead of being queued for rebuild.
        return False


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #


def prepare_arrays(
    case_id: str,
    root: str | Path | None = None,
    transform: Compose | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the validated deterministic transform and quantize for storage.

    Returns ``(image float16 [D, H, W], label uint8 [D, H, W])``. The channel
    axis MONAI adds is dropped: it is always 1 and costs nothing to restore.
    """
    transform = transform or Compose(deterministic_transforms())
    paths = get_case_paths(case_id, "train", root)
    result = transform({"image": str(paths["ct"]), "label": str(paths["combined"])})

    image = np.asarray(result["image"].as_tensor()[0], dtype=np.float32)
    label = np.asarray(result["label"].as_tensor()[0], dtype=np.float32)

    if not np.array_equal(label, np.rint(label)):
        raise ValueError(f"{case_id}: non-integer label values after resampling")

    return image.astype(IMAGE_STORAGE_DTYPE), label.astype(LABEL_STORAGE_DTYPE)


def _prepare_one(job: tuple[str, str, str | None, bool]) -> dict[str, Any]:
    """Worker entry point. Must be module-level to be picklable."""
    case_id, prepared_root, root, overwrite = job
    destination = case_path(prepared_root, case_id)

    if not overwrite and is_case_complete(destination):
        return {"case_id": case_id, "status": "skipped", "bytes": destination.stat().st_size}

    image, label = prepare_arrays(case_id, root)
    validate_arrays(image, label)
    size = write_prepared_case(destination, image, label)
    return {
        "case_id": case_id,
        "status": "written",
        "bytes": size,
        "shape": list(image.shape),
        "lesion_voxels": int((label == MAX_CLASS).sum()),
    }


def prepare_dataset(
    case_ids: Sequence[str],
    prepared_root: str | Path,
    root: str | Path | None = None,
    workers: int = 4,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Prepare many cases with a simple process pool.

    Failures are collected rather than aborting the run, so one unreadable case
    cannot waste hours of completed work. The caller decides what to do with
    them; ``verify_dataset`` is the completeness gate.
    """
    prepared_root = Path(prepared_root)
    cases_dir = prepared_root / CASES_DIRNAME
    cases_dir.mkdir(parents=True, exist_ok=True)

    # A SIGKILL or power loss during np.savez_compressed leaves a .tmp.npz
    # behind. It is never mistaken for a finished case (the final name never
    # appeared), but sweeping it keeps the directory honest and reclaims disk.
    stale = list(cases_dir.glob("*.tmp.npz"))
    for leftover in stale:
        leftover.unlink(missing_ok=True)
    if stale:
        logger.warning("removed %d stale temporary file(s) from an interrupted run", len(stale))

    jobs = [(case_id, str(prepared_root), str(root) if root else None, overwrite)
            for case_id in case_ids]
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_prepare_one, job): job[0] for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                logger.error("%s failed: %s", case_id, error)
                results.append({"case_id": case_id, "status": "failed", "error": str(error)})
            if done % 25 == 0 or done == len(jobs):
                logger.info("prepared %d/%d", done, len(jobs))

    return results


def verify_dataset(
    case_ids: Sequence[str],
    prepared_root: str | Path,
    deep: bool = False,
) -> list[str]:
    """Return the case IDs that are missing or unusable."""
    return [
        case_id
        for case_id in case_ids
        if not is_case_complete(case_path(prepared_root, case_id), deep=deep)
    ]


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_preprocessing_metadata(
    prepared_root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    case_count: int,
    git_commit: str | None = None,
) -> Path:
    """Record exactly what these arrays are, with no machine-specific paths."""
    metadata = {
        "orientation": ORIENTATION,
        "spacing_mm": list(TARGET_SPACING),
        "hu_clip": [HU_MIN, HU_MAX],
        "scale_range": [0.0, 1.0],
        "image_storage_dtype": np.dtype(IMAGE_STORAGE_DTYPE).name,
        "label_dtype": np.dtype(LABEL_STORAGE_DTYPE).name,
        "class_count": MAX_CLASS + 1,
        "lesion_class": MAX_CLASS,
        "case_count": int(case_count),
        "float16_tolerance_hu": round(FLOAT16_HU_TOLERANCE, 6),
        "git_commit": git_commit,
        "manifest_sha256": file_sha256(manifest_path),
        "split_sha256": file_sha256(split_path),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destination = Path(prepared_root) / METADATA_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


# --------------------------------------------------------------------------- #
# training from the cache
# --------------------------------------------------------------------------- #


class LoadPreparedd(MapTransform):
    """Read a prepared case into the tensors the sampling stage expects.

    Output: ``image`` float32 ``[1, D, H, W]`` in [0, 1] and ``label`` float32
    ``[1, D, H, W]`` holding exact integers 0...28 - the same types the raw
    deterministic chain produces, so the downstream stages cannot tell the two
    paths apart.

    This transform deliberately does no reorientation, no resampling and no
    renormalization. Those already happened, once, on the laptop.
    """

    def __init__(self, keys: Sequence[str] = ("prepared_path",)) -> None:
        super().__init__(keys=list(keys))
        self.path_key = list(keys)[0]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        updated = dict(data)
        image, label = read_prepared_case(updated[self.path_key])
        updated[IMAGE_KEY] = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        updated[LABEL_KEY] = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        return updated


def prepared_train_transforms(
    patch_size: Sequence[int] = PATCH_SIZE,
    samples_per_case: int = 2,
    augment: bool = True,
) -> Compose:
    """Cache-fed training pipeline: load, crop, augment.

    Identical to ``transforms.train_transforms`` from the crop onwards - the
    same two stage builders are called - with the deterministic core replaced
    by a cache read.
    """
    return Compose(
        [
            LoadPreparedd(),
            *sampling_transforms(patch_size, samples_per_case),
            *(augmentation_transforms() if augment else []),
            EnsureTyped(keys=[IMAGE_KEY, LABEL_KEY], track_meta=False),
        ]
    )


def build_prepared_records(
    prepared_root: str | Path,
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Map case IDs to cache paths, failing loudly on anything absent."""
    records = []
    missing = []
    for case_id in case_ids:
        path = case_path(prepared_root, case_id)
        if not path.is_file():
            missing.append(case_id)
        records.append({"case_id": case_id, "prepared_path": str(path)})
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} case(s) absent from the prepared cache: {missing[:10]}"
        )
    return records


def build_prepared_dataloaders(
    split: dict[str, Any],
    prepared_root: str | Path,
    fold: int = 0,
    batch_size: int = 1,
    samples_per_case: int = 2,
    num_workers: int = 4,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Training and validation loaders backed by the cache.

    The manifest is not needed here: a prepared case's identity is its
    filename, and its geometry is fixed by ``preprocessing.json``. Validation
    uses unaugmented patches; whole-volume inference always runs from raw
    NIfTI, never from this cache.
    """
    if not 0 <= fold < len(split["folds"]):
        raise ValueError(f"fold {fold} outside 0..{len(split['folds']) - 1}")

    definition = split["folds"][fold]
    train_ids = list(definition["train"])[:limit]
    val_ids = list(definition["val"])[:limit]

    train_dataset = Dataset(
        data=build_prepared_records(prepared_root, train_ids),
        transform=prepared_train_transforms(samples_per_case=samples_per_case),
    )
    val_dataset = Dataset(
        data=build_prepared_records(prepared_root, val_ids),
        transform=prepared_train_transforms(samples_per_case=samples_per_case, augment=False),
    )

    common = {
        "num_workers": num_workers,
        "collate_fn": list_data_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_dataset, batch_size=1, shuffle=False, **common),
    )


# --------------------------------------------------------------------------- #
# pilot selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PilotSelection:
    case_ids: list[str]
    rationale: dict[str, Any]


def _stride_sample(ordered: list[str], count: int) -> list[str]:
    """Evenly spaced picks across an ordered list, endpoints included."""
    if count >= len(ordered):
        return list(ordered)
    positions = np.linspace(0, len(ordered) - 1, num=count).round().astype(int)
    return [ordered[index] for index in dict.fromkeys(positions.tolist())]


def select_pilot_cases(manifest: dict[str, Any], count: int = 100) -> PilotSelection:
    """Choose a deterministic, representative pilot cohort.

    Representativeness is engineered, not sampled: an even stride across
    lesion size spans the smallest to the largest tumor, an even stride across
    z-spacing spans thin to thick slices, and a final pass guarantees at least
    one case for every source orientation present in PanTS-tr. No random seed
    is involved, so the cohort is a pure function of the manifest.
    """
    cases = manifest["cases"]
    positives = sorted(
        (c for c in cases if c["lesion_present"]),
        key=lambda c: (c["lesion_voxel_count"], c["case_id"]),
    )
    negatives = sorted(
        (c for c in cases if not c["lesion_present"]),
        key=lambda c: (c["spacing"][2], c["case_id"]),
    )

    share = count // 5
    chosen = _stride_sample([c["case_id"] for c in positives], 2 * share)
    chosen += _stride_sample([c["case_id"] for c in negatives], 2 * share)

    by_case = {c["case_id"]: c for c in cases}
    covered = {by_case[case_id]["orientation"] for case_id in chosen}
    smallest_per_orientation: dict[str, dict[str, Any]] = {}
    for case in cases:
        orientation = case["orientation"]
        volume = int(np.prod(case["shape"]))
        best = smallest_per_orientation.get(orientation)
        if best is None or volume < int(np.prod(best["shape"])):
            smallest_per_orientation[orientation] = case

    for orientation in sorted(smallest_per_orientation):
        if orientation not in covered and len(chosen) < count:
            chosen.append(smallest_per_orientation[orientation]["case_id"])
            covered.add(orientation)

    # top up from the remaining lesion-positives: they are the scarce resource
    if len(chosen) < count:
        remaining = [c["case_id"] for c in positives if c["case_id"] not in set(chosen)]
        chosen += _stride_sample(remaining, count - len(chosen))

    chosen = list(dict.fromkeys(chosen))[:count]
    selected = [by_case[case_id] for case_id in chosen]
    z_spacings = [c["spacing"][2] for c in selected]
    lesion_counts = [c["lesion_voxel_count"] for c in selected if c["lesion_present"]]

    return PilotSelection(
        case_ids=sorted(chosen),
        rationale={
            "count": len(chosen),
            "lesion_positive": sum(1 for c in selected if c["lesion_present"]),
            "lesion_negative": sum(1 for c in selected if not c["lesion_present"]),
            "lesion_voxels_min_max": [min(lesion_counts), max(lesion_counts)] if lesion_counts else [],
            "z_spacing_mm_min_max": [round(min(z_spacings), 3), round(max(z_spacings), 3)],
            "orientations": sorted(covered),
        },
    )
