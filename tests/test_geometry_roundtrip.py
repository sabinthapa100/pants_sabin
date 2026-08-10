"""Whole-volume inference must return predictions on the SOURCE CT grid.

Covers the orientations that actually occur in PanTS-tr, including the
awkward ones (LAI is the most common, IPL is rare but present).
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.paths import get_case_paths, get_data_root
from src.evaluation.inference import predict_case_in_source_geometry
from src.models.segresnet import build_segresnet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "pants_tr_manifest.json"

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir() or not MANIFEST.exists(),
    reason="PanTS-tr or manifest unavailable",
)


def smallest_case_for(orientation: str) -> str | None:
    """Pick the smallest volume with a given source orientation, to keep this fast."""
    cases = json.loads(MANIFEST.read_text())["cases"]
    matching = [c for c in cases if c["orientation"] == orientation]
    if not matching:
        return None
    return min(matching, key=lambda c: int(np.prod(c["shape"])))["case_id"]


@requires_data
@pytest.mark.parametrize("orientation", ["RAS", "LPS", "LAI", "IPL"])
def test_prediction_returns_to_source_geometry(orientation):
    case_id = smallest_case_for(orientation)
    if case_id is None:
        pytest.skip(f"no PanTS-tr case with source orientation {orientation}")

    paths = get_case_paths(case_id, "train")
    source = nib.load(str(paths["ct"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_segresnet("random").to(device)

    result = predict_case_in_source_geometry(
        model,
        str(paths["ct"]),
        sw_batch_size=1,
        accumulate_device="cpu",
        want_lesion_probability=True,
    )
    labels = result["labels"]
    probability = result["lesion_probability"]

    assert tuple(labels.shape[1:]) == tuple(source.shape), (
        f"{case_id} ({orientation}): shape {tuple(labels.shape[1:])} != source {source.shape}"
    )
    assert tuple(probability.shape[1:]) == tuple(source.shape)

    restored_affine = labels.affine.numpy()
    assert np.allclose(restored_affine, source.affine, atol=1e-3), (
        f"{case_id} ({orientation}): affine not restored"
    )
    assert nib.aff2axcodes(restored_affine) == nib.aff2axcodes(source.affine)

    values = labels.numpy()
    assert np.all(values == np.rint(values)), "inverse produced fractional class ids"
    assert values.min() >= 0 and values.max() <= 28

    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0


@requires_data
def test_inference_accumulates_on_cpu_not_gpu():
    """
    The stitched 29-channel volume is what previously exhausted VRAM. Patches
    run on the GPU; the accumulator must stay on the host.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    case_id = smallest_case_for("RAS")
    paths = get_case_paths(case_id, "train")
    model = build_segresnet("random").cuda()

    torch.cuda.reset_peak_memory_stats()
    result = predict_case_in_source_geometry(
        model, str(paths["ct"]), sw_batch_size=1, accumulate_device="cpu"
    )
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    assert result["labels"].device.type == "cpu"
    # a single 96^3 window of 29-channel activations is far below 1 GB;
    # a GPU-resident full-volume accumulator would be several GB
    assert peak_gb < 4.0, f"peak VRAM {peak_gb:.2f} GB suggests GPU-side stitching"
    print(f"\n  peak VRAM with CPU stitching: {peak_gb:.2f} GB")
