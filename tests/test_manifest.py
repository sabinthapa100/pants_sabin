"""Manifest validation and split determinism, on synthetic volumes."""

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from src.data.manifest import (
    build_manifest,
    build_split,
    is_train_case,
    parse_case_index,
    to_nnunet_splits,
)


def _write_case(
    root,
    case_id,
    label_values=(0, 17, 28),
    shape=(8, 8, 4),
    label_affine=None,
    split="train",
):
    """Create one synthetic PanTS case on disk."""
    image_dir = "ImageTr" if split == "train" else "ImageTe"
    label_dir = "LabelTr" if split == "train" else "LabelTe"

    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    ct_dir = root / image_dir / case_id
    lb_dir = root / label_dir / case_id
    ct_dir.mkdir(parents=True, exist_ok=True)
    lb_dir.mkdir(parents=True, exist_ok=True)

    ct = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    nib.save(nib.Nifti1Image(ct, affine), str(ct_dir / "ct.nii.gz"))

    label = np.zeros(shape, dtype=np.int16)
    flat = label.reshape(-1)
    for index, value in enumerate(label_values):
        flat[index] = value
    nib.save(
        nib.Nifti1Image(label, affine if label_affine is None else label_affine),
        str(lb_dir / "combined_labels.nii.gz"),
    )


@pytest.fixture
def dataset(tmp_path):
    for index in range(1, 11):
        # every third case carries a lesion (class 28)
        values = (0, 17, 28) if index % 3 == 0 else (0, 17)
        _write_case(tmp_path, f"PanTS_{index:08d}", label_values=values)
    return tmp_path


def test_parse_and_range():
    assert parse_case_index("PanTS_00000123") == 123
    assert is_train_case("PanTS_00009000")
    assert not is_train_case("PanTS_00009001")  # first PanTS-te case
    with pytest.raises(ValueError):
        parse_case_index("NotAPanTSCase")


def test_manifest_records_expected_fields(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    entry = next(c for c in manifest["cases"] if c["case_id"] == "PanTS_00000003")

    assert entry["ct"] == "ImageTr/PanTS_00000003/ct.nii.gz"
    assert entry["label"] == "LabelTr/PanTS_00000003/combined_labels.nii.gz"
    assert entry["shape"] == [8, 8, 4]
    assert entry["spacing"] == [1.5, 1.5, 2.0]
    assert entry["orientation"] == "RAS"
    assert entry["lesion_present"] is True
    assert entry["lesion_voxel_count"] == 1
    assert manifest["meta"]["lesion_positive"] == 3


def test_manifest_paths_are_relative(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    for entry in manifest["cases"]:
        assert not entry["ct"].startswith("/")
        assert str(dataset) not in entry["ct"]
        assert str(dataset) not in entry["label"]


def test_rejects_pants_te_case(dataset):
    _write_case(dataset, "PanTS_00009001", split="train")
    with pytest.raises(ValueError, match="outside PanTS-tr"):
        build_manifest(root=dataset, workers=1, expected_cases=None)


def test_rejects_out_of_range_label(dataset):
    _write_case(dataset, "PanTS_00000011", label_values=(0, 29))
    with pytest.raises(ValueError, match="label_out_of_range"):
        build_manifest(root=dataset, workers=1, expected_cases=None)


def test_rejects_spacing_mismatch(dataset):
    """Differing voxel size breaks index correspondence and is fatal."""
    _write_case(
        dataset,
        "PanTS_00000012",
        label_affine=np.diag([3.0, 3.0, 2.0, 1.0]),
    )
    with pytest.raises(ValueError, match="spacing_mismatch"):
        build_manifest(root=dataset, workers=1, expected_cases=None)


def test_rejects_shape_mismatch(dataset):
    """Differing array shape is fatal."""
    _write_case(dataset, "PanTS_00000013", shape=(8, 8, 4))
    label_dir = dataset / "LabelTr" / "PanTS_00000013"
    nib.save(
        nib.Nifti1Image(
            np.zeros((8, 8, 6), dtype=np.int16), np.diag([1.5, 1.5, 2.0, 1.0])
        ),
        str(label_dir / "combined_labels.nii.gz"),
    )
    with pytest.raises(ValueError, match="shape_mismatch"):
        build_manifest(root=dataset, workers=1, expected_cases=None)


def test_degenerate_label_affine_is_recorded_not_fatal(dataset):
    """
    Mirrors the real PanTS quirk: 78 of 9,000 cases carry a label affine whose
    translation is zeroed and direction reset, while the voxel array is still
    index-aligned with the CT. Shape and spacing agree, so the case is usable
    and the loader corrects the affine. It must be flagged, not rejected.
    """
    _write_case(
        dataset,
        "PanTS_00000014",
        # same voxel size, opposite direction and zeroed origin
        label_affine=np.diag([-1.5, -1.5, 2.0, 1.0]),
    )
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)

    entry = next(c for c in manifest["cases"] if c["case_id"] == "PanTS_00000014")
    assert entry["label_affine_matches_ct"] is False
    assert entry["label_orientation_matches_ct"] is False
    assert manifest["meta"]["label_affine_mismatch_cases"] == 1

    healthy = next(c for c in manifest["cases"] if c["case_id"] == "PanTS_00000001")
    assert healthy["label_affine_matches_ct"] is True


def test_rejects_missing_label(dataset):
    (dataset / "LabelTr" / "PanTS_00000001" / "combined_labels.nii.gz").unlink()
    with pytest.raises(ValueError, match="missing_label"):
        build_manifest(root=dataset, workers=1, expected_cases=None)


def test_split_is_deterministic(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    assert build_split(manifest, seed=317, folds=5) == build_split(manifest, seed=317, folds=5)


def test_split_changes_with_seed(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    a = build_split(manifest, seed=317, folds=5)["folds"]
    b = build_split(manifest, seed=318, folds=5)["folds"]
    assert a != b


def test_split_partitions_every_case_exactly_once(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    split = build_split(manifest, seed=317, folds=5)
    all_cases = {entry["case_id"] for entry in manifest["cases"]}

    seen_in_validation = []
    for fold in split["folds"]:
        assert not set(fold["train"]) & set(fold["val"]), "train/val leakage within a fold"
        assert set(fold["train"]) | set(fold["val"]) == all_cases
        seen_in_validation.extend(fold["val"])

    assert sorted(seen_in_validation) == sorted(all_cases)


def test_split_stratifies_lesion_positive_cases(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    split = build_split(manifest, seed=317, folds=3)
    counts = [row["val_lesion_positive"] for row in split["meta"]["fold_summary"]]
    # 3 lesion-positive cases dealt round-robin across 3 folds
    assert counts == [1, 1, 1]


def test_split_records_that_grouping_is_case_level(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    meta = build_split(manifest, seed=317, folds=5)["meta"]
    assert meta["split_level"] == "case"
    assert "patient" in meta["grouping_note"].lower()


def test_nnunet_split_format(dataset):
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    split = build_split(manifest, seed=317, folds=5)
    native = to_nnunet_splits(split)

    assert isinstance(native, list) and len(native) == 5
    for fold in native:
        assert set(fold) == {"train", "val"}
        assert all(isinstance(case, str) for case in fold["train"] + fold["val"])


def test_nnunet_split_matches_real_nnunet_file_structure(dataset):
    """
    nnU-Net must be able to consume our split verbatim.

    Compared against the splits_final.json nnU-Net itself generated for the
    existing Dataset501 smoke run, so the check is against real output rather
    than an assumption about the format.
    """
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "nnunet"
        / "nnUNet_preprocessed"
        / "Dataset501_PanTSSmoke"
        / "splits_final.json"
    )
    if not reference_path.exists():
        pytest.skip("Dataset501 splits_final.json not present")

    reference = json.loads(reference_path.read_text())
    manifest = build_manifest(root=dataset, workers=1, expected_cases=None)
    native = to_nnunet_splits(build_split(manifest, seed=317, folds=5))

    assert type(native) is type(reference)
    assert len(native) == len(reference)
    for ours, theirs in zip(native, reference, strict=True):
        assert set(ours) == set(theirs)
        assert type(ours["train"]) is type(theirs["train"])


def test_align_label_geometry_prevents_mirroring(dataset):
    """
    The safety property behind AlignLabelGeometryd.

    A label whose affine points the opposite way but whose array is
    index-aligned must survive reorientation without being mirrored relative
    to the image.
    """
    pytest.importorskip("monai")
    import torch
    from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd

    from src.data.transforms import AlignLabelGeometryd

    case = "PanTS_00000015"
    _write_case(dataset, case, label_affine=np.diag([-1.5, -1.5, 2.0, 1.0]))
    # put a unique marker in one corner of BOTH arrays at the same index
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    marked = np.zeros((8, 8, 4), dtype=np.int16)
    marked[0, 0, 0] = 17
    nib.save(
        nib.Nifti1Image(marked.astype(np.float32), affine),
        str(dataset / "ImageTr" / case / "ct.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(marked, np.diag([-1.5, -1.5, 2.0, 1.0])),
        str(dataset / "LabelTr" / case / "combined_labels.nii.gz"),
    )

    record = {
        "image": str(dataset / "ImageTr" / case / "ct.nii.gz"),
        "label": str(dataset / "LabelTr" / case / "combined_labels.nii.gz"),
    }

    aligned = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        EnsureTyped(keys=["image", "label"], track_meta=True),
        AlignLabelGeometryd(),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])(dict(record))

    unaligned = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
        EnsureTyped(keys=["image", "label"], track_meta=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])(dict(record))

    # With alignment the marker stays on top of the image marker.
    image_marker = (aligned["image"] == 17).nonzero()
    label_marker = (aligned["label"] == 17).nonzero()
    assert torch.equal(image_marker, label_marker), "aligned label drifted from the image"

    # Without alignment it is mirrored away from it.
    bad_marker = (unaligned["label"] == 17).nonzero()
    assert not torch.equal(
        (unaligned["image"] == 17).nonzero(), bad_marker
    ), "test is not exercising the mirroring hazard"
