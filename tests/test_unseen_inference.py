"""Inference must work on a bare CT with no PanTS context whatsoever.

This is the closest local simulation of an external evaluator: a renamed CT in
an unrelated directory, a trained checkpoint, and nothing else. The test copies
one PanTS-tr CT to a scratch location under a meaningless name so that no case
identifier, no sibling `LabelTr` path and no manifest entry can be reached.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.paths import get_case_paths, get_data_root
from src.data.transforms import inference_transforms
from src.evaluation.inference import predict_case_in_source_geometry
from src.models.segresnet import build_segresnet
from src.training.checkpoint import save_training_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASE = "PanTS_00000001"

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir(), reason=f"PanTS-tr not readable at {data_root}"
)


@pytest.fixture
def anonymous_ct(tmp_path) -> Path:
    """A CT with every trace of PanTS provenance stripped from its location."""
    source = get_case_paths(SOURCE_CASE, "train")["ct"]
    destination = tmp_path / "outside_cohort" / "scan_0001.nii.gz"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    return destination


@pytest.fixture
def trained_checkpoint(tmp_path) -> Path:
    """An untrained-but-valid checkpoint: this test is about plumbing, not accuracy."""
    model = build_segresnet("random")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "best.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        epoch=0, global_step=0, best_metric=0.0, config={"note": "test"},
    )
    return path


def test_inference_transform_needs_no_label():
    """The image-only chain must not carry a label key or a fabricated stand-in."""
    transform = inference_transforms()
    names = [type(step).__name__ for step in transform.transforms]
    assert "AlignLabelGeometryd" not in names, "label affine repair is meaningless without a label"
    for step in transform.transforms:
        keys = getattr(step, "keys", ())
        assert "label" not in tuple(keys), f"{type(step).__name__} still expects a label"


@requires_data
def test_preprocessing_reads_only_the_image(anonymous_ct):
    result = inference_transforms()({"image": str(anonymous_ct)})
    assert set(result) >= {"image"}
    assert "label" not in result, "inference produced a label key from nowhere"

    image = result["image"]
    assert image.ndim == 4 and image.shape[0] == 1
    values = image.as_tensor().float()
    assert torch.isfinite(values).all()
    assert 0.0 <= float(values.min()) and float(values.max()) <= 1.0


@requires_data
def test_unseen_ct_round_trips_to_source_geometry(anonymous_ct, trained_checkpoint):
    """End to end through the CLI, then verify the contract on the written files."""
    output = anonymous_ct.parent / "prediction"
    completed = subprocess.run(
        [sys.executable, "scripts/infer_segresnet.py",
         "--input", str(anonymous_ct),
         "--output", str(output),
         "--checkpoint", str(trained_checkpoint),
         "--lesion-probability",
         "--device", "cuda" if torch.cuda.is_available() else "cpu"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    source = nib.load(str(anonymous_ct))
    labels_image = nib.load(str(output / "combined_labels.nii.gz"))
    probability_image = nib.load(str(output / "pancreatic_lesion_probability.nii.gz"))

    # geometry is restored, not merely resampled to something plausible
    assert labels_image.shape == source.shape
    assert np.allclose(labels_image.affine, source.affine, atol=1e-4)
    assert nib.aff2axcodes(labels_image.affine) == nib.aff2axcodes(source.affine)

    labels = np.asanyarray(labels_image.dataobj)
    assert labels.dtype == np.uint8
    unique = np.unique(labels)
    assert unique.min() >= 0 and unique.max() <= 28, unique
    assert np.array_equal(unique, np.rint(unique)), "label map must be integer-valued"

    probability = np.asanyarray(probability_image.dataobj)
    assert probability.shape == source.shape
    assert np.isfinite(probability).all()
    assert 0.0 <= probability.min() and probability.max() <= 1.0

    # nothing but the CT and the checkpoint was consulted
    log = completed.stdout + completed.stderr
    for forbidden in ("LabelTr", "combined_labels.nii.gz\n  read", "manifest.json",
                      "pants_cv_v1.json", "supervised_suprem", ".npz"):
        assert forbidden not in log, f"inference appears to have touched {forbidden}"


@requires_data
def test_inference_does_not_open_any_pants_context(anonymous_ct, trained_checkpoint, monkeypatch):
    """Trip a wire on every path we must never read during external inference."""
    opened: list[str] = []
    real_open = open

    def watched_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", watched_open)
    model = build_segresnet("random")
    predict_case_in_source_geometry(
        model, str(anonymous_ct), sw_batch_size=1, accumulate_device="cpu",
        want_lesion_probability=True,
    )

    forbidden = ("LabelTr", "manifest", "pants_cv_v1", "prepared", ".npz", "suprem")
    touched = [p for p in opened if any(token in p.lower() for token in forbidden)]
    assert not touched, f"inference opened PanTS context files: {touched[:5]}"
