"""The prepared cache must be indistinguishable from the raw path.

If these tests fail, the laptop-side cache and the raw-NIfTI inference path
have drifted apart, and any model trained on the cache is being evaluated
under different preprocessing than it was trained with.
"""

import numpy as np
import pytest
import torch
from monai.transforms import Compose
from monai.utils import set_determinism

from src.data.labels import PANCREATIC_LESION
from src.data.paths import get_data_root
from src.data.prepared import (
    FLOAT16_HU_TOLERANCE,
    IMAGE_STORAGE_DTYPE,
    LABEL_STORAGE_DTYPE,
    case_path,
    is_case_complete,
    prepare_arrays,
    prepared_train_transforms,
    read_prepared_case,
    select_pilot_cases,
    validate_arrays,
    write_prepared_case,
)
from src.data.transforms import (
    HU_MAX,
    HU_MIN,
    deterministic_transforms,
    train_transforms,
)


CASE = "PanTS_00000003"  # lesion-positive, modest size

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir(), reason=f"PanTS-tr not readable at {data_root}"
)


# --------------------------------------------------------------------------- #
# storage contract
# --------------------------------------------------------------------------- #


def test_validate_arrays_rejects_corruption():
    good_image = np.zeros((4, 4, 4), dtype=IMAGE_STORAGE_DTYPE)
    good_label = np.zeros((4, 4, 4), dtype=LABEL_STORAGE_DTYPE)
    validate_arrays(good_image, good_label)  # baseline must pass

    with pytest.raises(ValueError, match="dtype"):
        validate_arrays(good_image.astype(np.float32), good_label)
    with pytest.raises(ValueError, match="dtype"):
        validate_arrays(good_image, good_label.astype(np.int16))
    with pytest.raises(ValueError, match="shape"):
        validate_arrays(good_image, np.zeros((4, 4, 5), dtype=LABEL_STORAGE_DTYPE))

    outside = good_image.copy()
    outside[0, 0, 0] = 1.5
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_arrays(outside, good_label)

    non_finite = good_image.copy()
    non_finite[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_arrays(non_finite, good_label)

    too_high = good_label.copy()
    too_high[0, 0, 0] = 29
    with pytest.raises(ValueError, match="outside 0..28"):
        validate_arrays(good_image, too_high)


def test_write_is_atomic_and_leaves_no_debris(tmp_path):
    destination = tmp_path / "cases" / "PanTS_00000001.npz"
    image = np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4).astype(IMAGE_STORAGE_DTYPE)
    label = np.arange(64, dtype=np.uint8).reshape(4, 4, 4) % 29

    write_prepared_case(destination, image, label)
    assert destination.is_file()
    assert not list(destination.parent.glob("*.tmp.npz")), "temporary file was left behind"

    stored_image, stored_label = read_prepared_case(destination)
    assert np.array_equal(stored_label, label), "labels must survive exactly"
    assert stored_image.dtype == IMAGE_STORAGE_DTYPE

    # A rejected write must not replace or create the destination.
    bad = label.copy()
    bad[0, 0, 0] = 200
    with pytest.raises(ValueError):
        write_prepared_case(tmp_path / "cases" / "PanTS_00000002.npz", image, bad)
    assert not (tmp_path / "cases" / "PanTS_00000002.npz").exists()
    assert not list(destination.parent.glob("*.tmp.npz"))


def test_incomplete_case_is_not_reported_complete(tmp_path):
    path = tmp_path / "PanTS_00000001.npz"
    assert not is_case_complete(path)

    path.write_bytes(b"not an npz archive")
    assert not is_case_complete(path), "a truncated file must never count as done"

    np.savez_compressed(path, image=np.zeros((2, 2, 2), dtype=IMAGE_STORAGE_DTYPE))
    assert not is_case_complete(path), "an archive missing the label must not count as done"


def test_truncated_archive_is_rejected_not_raised(tmp_path):
    """An npz is a zip: a half-written one raises BadZipFile, not OSError.

    Power loss can in principle leave a correctly-named but truncated archive.
    The resume scan must classify it as incomplete and rebuild it, never crash
    partway through a 9,000-case run.
    """
    path = tmp_path / "PanTS_00000001.npz"
    write_prepared_case(
        path,
        np.zeros((8, 8, 8), dtype=IMAGE_STORAGE_DTYPE),
        np.zeros((8, 8, 8), dtype=LABEL_STORAGE_DTYPE),
    )
    assert is_case_complete(path, deep=True)

    intact = path.read_bytes()
    path.write_bytes(intact[: len(intact) // 2])
    assert not is_case_complete(path), "truncated archive must not count as complete"
    assert not is_case_complete(path, deep=True)

    # and preparation can simply overwrite it
    write_prepared_case(
        path,
        np.zeros((8, 8, 8), dtype=IMAGE_STORAGE_DTYPE),
        np.zeros((8, 8, 8), dtype=LABEL_STORAGE_DTYPE),
    )
    assert is_case_complete(path, deep=True)


def test_archive_contains_only_plain_arrays(tmp_path):
    """No pickles, no objects: the cache must load with allow_pickle=False."""
    destination = tmp_path / "PanTS_00000001.npz"
    write_prepared_case(
        destination,
        np.zeros((3, 3, 3), dtype=IMAGE_STORAGE_DTYPE),
        np.zeros((3, 3, 3), dtype=LABEL_STORAGE_DTYPE),
    )
    with np.load(destination, allow_pickle=False) as archive:
        assert set(archive.files) == {"image", "label"}


# --------------------------------------------------------------------------- #
# equivalence with the validated raw path
# --------------------------------------------------------------------------- #


@requires_data
def test_cache_matches_the_deterministic_transform(tmp_path):
    """float16 storage is the ONLY difference between cache and raw."""
    reference = Compose(deterministic_transforms())(
        {
            "image": str(data_root / "ImageTr" / CASE / "ct.nii.gz"),
            "label": str(data_root / "LabelTr" / CASE / "combined_labels.nii.gz"),
        }
    )
    reference_image = np.asarray(reference["image"].as_tensor()[0], dtype=np.float32)
    reference_label = np.asarray(reference["label"].as_tensor()[0], dtype=np.float32)

    image, label = prepare_arrays(CASE)
    destination = case_path(tmp_path, CASE)
    write_prepared_case(destination, image, label)
    stored_image, stored_label = read_prepared_case(destination)

    assert stored_image.shape == reference_image.shape
    assert stored_image.dtype == IMAGE_STORAGE_DTYPE
    assert stored_label.dtype == LABEL_STORAGE_DTYPE

    # Labels are exact: nearest-neighbour resampling only ever copies existing
    # class IDs, and uint8 represents 0..28 without loss.
    assert np.array_equal(stored_label.astype(np.float32), reference_label)
    assert set(np.unique(stored_label)).issubset(set(range(29)))
    assert (stored_label == PANCREATIC_LESION).sum() == (
        reference_label == PANCREATIC_LESION
    ).sum(), "class-28 voxels must be preserved exactly"

    # The image differs only by float16 quantization of an already-clipped,
    # already-normalized signal.
    error = np.abs(stored_image.astype(np.float32) - reference_image)
    error_hu = float(error.max()) * (HU_MAX - HU_MIN)
    assert error_hu <= FLOAT16_HU_TOLERANCE, f"{error_hu:.4f} HU exceeds float16 tolerance"
    print(
        f"\n  cache vs raw: max {error_hu:.4f} HU "
        f"(tolerance {FLOAT16_HU_TOLERANCE:.4f} HU), shape {stored_image.shape}"
    )


@requires_data
def test_prepared_and_raw_training_paths_produce_the_same_patches(tmp_path):
    """Same seed, same case: the two front ends must agree downstream."""
    image, label = prepare_arrays(CASE)
    write_prepared_case(case_path(tmp_path, CASE), image, label)

    set_determinism(317)
    raw_patches = train_transforms(samples_per_case=2, augment=False)(
        {
            "image": str(data_root / "ImageTr" / CASE / "ct.nii.gz"),
            "label": str(data_root / "LabelTr" / CASE / "combined_labels.nii.gz"),
        }
    )

    set_determinism(317)
    cached_patches = prepared_train_transforms(samples_per_case=2, augment=False)(
        {"prepared_path": str(case_path(tmp_path, CASE))}
    )

    assert len(raw_patches) == len(cached_patches) == 2
    for raw, cached in zip(raw_patches, cached_patches, strict=True):
        assert tuple(raw["image"].shape) == tuple(cached["image"].shape) == (1, 96, 96, 96)
        assert torch.equal(
            cached["label"].as_tensor() if hasattr(cached["label"], "as_tensor") else cached["label"],
            raw["label"].as_tensor(),
        ), "the two paths sampled different crops or different labels"

        difference = (cached["image"].float() - raw["image"].as_tensor().float()).abs().max()
        assert float(difference) * (HU_MAX - HU_MIN) <= FLOAT16_HU_TOLERANCE


# --------------------------------------------------------------------------- #
# pilot cohort
# --------------------------------------------------------------------------- #


def test_pilot_selection_is_deterministic_and_representative():
    manifest = {
        "cases": [
            {
                "case_id": f"PanTS_{index:08d}",
                "lesion_present": index % 4 == 0,
                "lesion_voxel_count": index * 7 if index % 4 == 0 else 0,
                "spacing": [0.8, 0.8, 0.5 + (index % 20) * 0.5],
                "shape": [200 + index, 180, 100 + index],
                "orientation": ["RAS", "LPS", "LAI", "IPL"][index % 4],
            }
            for index in range(1, 401)
        ]
    }

    first = select_pilot_cases(manifest, count=100)
    second = select_pilot_cases(manifest, count=100)
    assert first.case_ids == second.case_ids, "pilot selection must be reproducible"
    assert len(first.case_ids) == 100
    assert len(set(first.case_ids)) == 100

    assert first.rationale["lesion_positive"] > 0
    assert first.rationale["lesion_negative"] > 0
    # spans the extremes rather than clustering
    assert first.rationale["z_spacing_mm_min_max"][0] < 1.0
    assert first.rationale["z_spacing_mm_min_max"][1] > 5.0
    assert set(first.rationale["orientations"]) == {"RAS", "LPS", "LAI", "IPL"}
