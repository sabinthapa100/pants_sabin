"""End-to-end integration: raw NIfTI -> transforms -> batch -> loss.

This is an interface check, not training. It proves the data layer and both
model arms actually connect, on real PanTS volumes.

Skipped unless PANTS_DATA_ROOT resolves to a readable PanTS-tr tree. The
SuPreM half additionally needs SUPREM_CHECKPOINT.
"""

import os
import resource
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.manifest import build_manifest, build_split
from src.data.paths import get_data_root
from src.data.transforms import PATCH_SIZE, build_dataloaders
from src.models.segresnet import NUM_CLASSES, build_segresnet


PILOT_CASES = [f"PanTS_{index:08d}" for index in (1, 2, 3, 4)]

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir(),
    reason=f"PanTS-tr not readable at {data_root}; set PANTS_DATA_ROOT",
)

checkpoint_path = os.environ.get("SUPREM_CHECKPOINT")
requires_checkpoint = pytest.mark.skipif(
    not (checkpoint_path and Path(checkpoint_path).exists()),
    reason="SUPREM_CHECKPOINT is not set to an existing file",
)


def peak_cpu_memory_gb() -> float:
    """Peak resident set size of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


@pytest.fixture(scope="module")
def one_batch():
    """One real training batch built through the shared data layer."""
    from monai.utils import set_determinism

    # Augmentation is random; pin it so this integration test cannot flake.
    set_determinism(seed=317)

    manifest = build_manifest(case_ids=PILOT_CASES, workers=1)
    split = build_split(manifest, seed=317, folds=2)

    train_loader, val_loader = build_dataloaders(
        manifest,
        split,
        fold=0,
        batch_size=1,
        samples_per_case=2,
        num_workers=0,
    )
    batch = next(iter(train_loader))
    return batch, train_loader, val_loader


@requires_data
def test_batch_shapes_and_content(one_batch):
    batch, _, _ = one_batch
    image, label = batch["image"], batch["label"]

    # batch_size=1 with samples_per_case=2 yields 2 patches
    assert image.shape[0] == label.shape[0] == 2
    assert tuple(image.shape[1:]) == (1, *PATCH_SIZE)
    assert tuple(label.shape[1:]) == (1, *PATCH_SIZE)

    assert torch.isfinite(image).all(), "non-finite voxels in the CT patch"

    # Intensity augmentation (shift +/-0.10, noise sigma 0.01) runs after
    # normalization, so training patches legitimately leave [0, 1]. Only the
    # deterministic validation path is exactly bounded - see
    # test_validation_intensities_are_exactly_normalized.
    assert float(image.min()) >= -0.5 and float(image.max()) <= 1.5

    classes = np.unique(label.numpy()).astype(int)
    assert classes.min() >= 0 and classes.max() <= 28
    assert np.all(classes == np.rint(classes)), "labels must stay integral"

    print(f"\n  image  {tuple(image.shape)} {image.dtype} "
          f"range {float(image.min()):.3f}..{float(image.max()):.3f}")
    print(f"  label  {tuple(label.shape)} classes present: {classes.tolist()}")


@requires_data
def test_validation_loader_returns_whole_volumes(one_batch):
    _, _, val_loader = one_batch
    sample = next(iter(val_loader))
    image = sample["image"]

    assert image.ndim == 5
    # not cropped to the training patch size
    assert tuple(image.shape[2:]) != PATCH_SIZE
    print(f"\n  validation volume {tuple(image.shape)} (uncropped, sliding-window later)")


@requires_data
def test_validation_intensities_are_exactly_normalized(one_batch):
    """The deterministic path applies no augmentation, so [0, 1] is exact."""
    _, _, val_loader = one_batch
    image = next(iter(val_loader))["image"]

    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


@requires_data
def test_random_initialization_produces_finite_loss(one_batch):
    from monai.losses import DiceCELoss

    batch, _, _ = one_batch
    image, label = batch["image"], batch["label"]

    model = build_segresnet("random")
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)

    output = model(image)
    loss = criterion(output, label)

    assert tuple(output.shape) == (image.shape[0], NUM_CLASSES, *PATCH_SIZE)
    assert torch.isfinite(loss), "loss is not finite"

    print(f"\n  random  output {tuple(output.shape)} loss={loss.item():.4f} "
          f"peak CPU RAM={peak_cpu_memory_gb():.2f} GB")


@requires_data
@requires_checkpoint
def test_suprem_initialization_produces_finite_loss(one_batch):
    from monai.losses import DiceCELoss

    batch, _, _ = one_batch
    image, label = batch["image"], batch["label"]

    model = build_segresnet("suprem", checkpoint_path)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)

    output = model(image)
    loss = criterion(output, label)

    assert tuple(output.shape) == (image.shape[0], NUM_CLASSES, *PATCH_SIZE)
    assert torch.isfinite(loss), "loss is not finite"

    print(f"\n  suprem  output {tuple(output.shape)} loss={loss.item():.4f} "
          f"peak CPU RAM={peak_cpu_memory_gb():.2f} GB")


@requires_data
@requires_checkpoint
def test_both_arms_consume_the_identical_batch(one_batch):
    """
    The controlled-experiment guarantee: identical architecture and identical
    input, differing only in initialization.
    """
    batch, _, _ = one_batch
    image = batch["image"]

    random_model = build_segresnet("random").eval()
    suprem_model = build_segresnet("suprem", checkpoint_path).eval()

    with torch.no_grad():
        random_output = random_model(image)
        suprem_output = suprem_model(image)

    assert random_output.shape == suprem_output.shape
    # Same architecture, same input, different weights -> different predictions.
    assert not torch.allclose(random_output, suprem_output)


@requires_data
def test_gpu_memory_if_available(one_batch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    batch, _, _ = one_batch
    image = batch["image"].cuda()
    model = build_segresnet("random").cuda()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(image)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n  peak GPU VRAM for forward pass: {peak_gb:.2f} GB")
    assert peak_gb > 0
