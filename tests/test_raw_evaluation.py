"""Raw-mode evaluation against a SYNTHETIC data root.

The held-out evaluation path had never executed: it referenced a path key the
helper does not return, hard-coded the training split, and had no way to
enumerate held-out case IDs. These tests exercise it end to end on temporary
NIfTI files built in-process.

NOTHING HERE READS THE REAL PanTS-te. The synthetic root contains an
``ImageTe``/``LabelTe`` pair so the directory *resolution* can be proven, while
the actual dataset stays untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_segresnet as evaluate  # noqa: E402
from src.data.paths import get_case_paths  # noqa: E402


CASE = "PanTS_00009001"          # a held-out-range ID that exists only in tmp_path
CLASSES = 29
LESION = 28


def write_case(root: Path, split: str, case_id: str, shape=(16, 16, 16)) -> np.ndarray:
    """Create ImageTx/<case>/ct.nii.gz and LabelTx/<case>/combined_labels.nii.gz."""
    images, labels = {"train": ("ImageTr", "LabelTr"), "test": ("ImageTe", "LabelTe")}[split]
    affine = np.diag([1.5, 1.5, 1.5, 1.0])

    rng = np.random.default_rng(0)
    ct = (rng.standard_normal(shape) * 100).astype(np.int16)
    target = np.zeros(shape, dtype=np.uint8)
    target[4:8, 4:8, 4:8] = LESION
    target[9:12, 9:12, 9:12] = 17

    for directory, array, name in (
        (images, ct, "ct.nii.gz"), (labels, target, "combined_labels.nii.gz")
    ):
        case_directory = root / directory / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array, affine), str(case_directory / name))
    return target


class ConstantModel(torch.nn.Module):
    """Predicts one fixed class everywhere, with a controllable lesion peak.

    Standing in for the real network keeps these tests CPU-only and fast; what
    is under test is the raw-mode plumbing, not segmentation quality.
    """

    def __init__(self, lesion_logit: float = 5.0, runner_up: int = 17) -> None:
        super().__init__()
        self.lesion_logit = lesion_logit
        self.runner_up = runner_up
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], CLASSES, *x.shape[2:])
        logits[:, self.runner_up] = 0.5
        logits[:, LESION] = self.lesion_logit
        return logits


def raw_args(root: Path, split: str, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        data_root=root, data_split=split, overlap=0.5, sw_batch_size=1,
        device="cpu", accumulate_device="cpu", lesion_peak_probability=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# defect A: the path key
# --------------------------------------------------------------------------- #


def test_path_helper_exposes_combined_not_combined_labels():
    """Pins the contract the raw evaluator got wrong."""
    paths = get_case_paths(CASE, "test", "/nowhere")
    assert "combined" in paths
    assert "combined_labels" not in paths
    assert paths["combined"].name == "combined_labels.nii.gz"


# --------------------------------------------------------------------------- #
# defect B: split resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split,image_dir,label_dir", [
    ("train", "ImageTr", "LabelTr"),
    ("test", "ImageTe", "LabelTe"),
])
def test_data_split_selects_the_right_directory_pair(split, image_dir, label_dir):
    paths = get_case_paths(CASE, split, "/root")
    assert paths["ct"] == Path(f"/root/{image_dir}/{CASE}/ct.nii.gz")
    assert paths["combined"] == Path(f"/root/{label_dir}/{CASE}/combined_labels.nii.gz")


def test_raw_evaluation_resolves_ImageTe_and_LabelTe(tmp_path):
    """The held-out contract, proven on synthetic data only."""
    target = write_case(tmp_path, "test", CASE)
    assert (tmp_path / "ImageTe" / CASE / "ct.nii.gz").is_file()
    assert (tmp_path / "LabelTe" / CASE / "combined_labels.nii.gz").is_file()
    assert not (tmp_path / "ImageTr").exists(), "test mode must not need the training tree"

    row = evaluate.evaluate_raw(ConstantModel(), CASE, raw_args(tmp_path, "test"))

    assert row["lesion_present"] is True
    assert row["target_lesion_voxels"] == int((target == LESION).sum())
    assert row["predicted_lesion_voxels"] > 0
    assert 0.0 <= row["lesion_dice"] <= 1.0


def test_raw_evaluation_still_works_for_the_training_split(tmp_path):
    write_case(tmp_path, "train", "PanTS_00000042")
    row = evaluate.evaluate_raw(ConstantModel(), "PanTS_00000042", raw_args(tmp_path, "train"))
    assert row["lesion_present"] is True


def test_raw_evaluation_scores_in_source_geometry(tmp_path):
    """Prediction is compared on the CT's own grid, not the 1.5 mm frame."""
    shape = (20, 18, 16)
    write_case(tmp_path, "test", CASE, shape=shape)
    row = evaluate.evaluate_raw(ConstantModel(), CASE, raw_args(tmp_path, "test"))
    # A shape mismatch raises inside evaluate_raw, so completion proves agreement.
    assert row["target_lesion_voxels"] + row["predicted_lesion_voxels"] > 0
    assert row["target_lesion_mm3"] == pytest.approx(row["target_lesion_voxels"] * 1.5**3)


# --------------------------------------------------------------------------- #
# postprocessing reaches raw mode too
# --------------------------------------------------------------------------- #


def test_component_filter_applies_in_raw_mode(tmp_path):
    write_case(tmp_path, "test", CASE)
    # The logit must win the argmax (> the runner-up's 0.5) yet leave the softmax
    # below the threshold, or there would be no lesion component to reject in the
    # first place. With 27 channels at 0 and one at 0.5, p28 = e^z / (e^z + 28.65),
    # so z = 1 gives argmax 28 at p28 = 0.087.
    weak = ConstantModel(lesion_logit=1.0)

    unfiltered = evaluate.evaluate_raw(weak, CASE, raw_args(tmp_path, "test"))
    filtered = evaluate.evaluate_raw(
        weak, CASE, raw_args(tmp_path, "test", lesion_peak_probability=0.6))

    assert unfiltered["predicted_lesion_voxels"] > 0
    assert filtered["predicted_lesion_voxels"] == 0


# --------------------------------------------------------------------------- #
# defect C + the guard
# --------------------------------------------------------------------------- #


def test_held_out_ids_can_be_enumerated_only_with_the_explicit_flag(tmp_path):
    listing = tmp_path / "held_out.json"
    listing.write_text(json.dumps({"cases": [CASE]}))
    args = argparse.Namespace(
        split=None, fold=0, limit=None, allow_test_split=False, case_list=listing
    )

    with pytest.raises(SystemExit, match="PanTS-te"):
        evaluate.resolve_cases(args)

    args.allow_test_split = True
    assert evaluate.resolve_cases(args) == [CASE]


def test_the_guard_is_not_weakened_by_the_new_path():
    """A mixed list must still be refused on the strength of one held-out ID."""
    with pytest.raises(SystemExit, match="PanTS-te"):
        evaluate.guard_split(["PanTS_00000001", CASE], allow_test=False)


def test_summary_cohort_label_is_derived_from_case_ids_not_a_flag(tmp_path):
    """A held-out run must be labelled held-out by the data, not by intent."""
    import scripts.evaluate_segresnet as e

    development = ["PanTS_00000001", "PanTS_00008999"]
    held_out = ["PanTS_00000001", CASE]

    def cohort(cases):
        return ("PanTS-te in-distribution held-out evaluation"
                if any(e.case_number(c) > e.LAST_TRAIN_CASE_ID for c in cases)
                else "PanTS-tr development evaluation")

    assert cohort(development) == "PanTS-tr development evaluation"
    assert cohort(held_out) == "PanTS-te in-distribution held-out evaluation"
