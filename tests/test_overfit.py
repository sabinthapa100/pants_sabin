"""Tiny-overfit test: can the pipeline memorize one real deterministic patch?

This is a pipeline check, not a model result. If a network cannot drive the
loss down and reproduce the target on a single fixed sample, something in the
data path, the loss, or the optimizer is wrong and no amount of real training
will fix it.

Deliberately slow: a few hundred optimizer steps per arm, roughly 80 s each on
an RTX 4070 Laptop. Marked `slow` so it can be selected or skipped explicitly:

    pytest tests/test_overfit.py -q -s -m slow     # just this check
    pytest -q -m "not slow"                        # fast suite only
"""

import os
from pathlib import Path

import pytest
import torch
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.data.labels import PANCREATIC_LESION
from src.data.paths import get_case_paths, get_data_root
from src.data.transforms import train_transforms
from src.models.segresnet import build_segresnet


# A known lesion-positive PanTS-tr case (1,570 class-28 voxels after resampling).
#
# Step budget and learning rate were calibrated, not guessed. The lesion is only
# 0.177% of the patch, so class 28 is the last class to win the argmax: it stays
# at Dice 0 for a long warm-up and then rises sharply. Measured crossings of
# Dice 0.6 on this patch:
#
#     suprem, lr 1e-3 : step  200
#     random, lr 1e-3 : step 1000
#     random, lr 3e-3 : step  400
#
# A shorter budget would report a false failure. This is a pipeline check, so a
# brisk learning rate is fine here; it is NOT the training configuration.
OVERFIT_CASE = "PanTS_00000003"
STEPS = 500
LEARNING_RATE = 3e-3

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir(),
    reason=f"PanTS-tr not readable at {data_root}",
)
checkpoint_path = os.environ.get("SUPREM_CHECKPOINT")
requires_checkpoint = pytest.mark.skipif(
    not (checkpoint_path and Path(checkpoint_path).exists()),
    reason="SUPREM_CHECKPOINT not set",
)


def deterministic_lesion_patch():
    """One fixed lesion-containing patch, with augmentation disabled."""
    set_determinism(317)
    paths = get_case_paths(OVERFIT_CASE, "train")
    patches = train_transforms(samples_per_case=8, augment=False)(
        {"image": str(paths["ct"]), "label": str(paths["combined"])}
    )
    best = max(patches, key=lambda s: int((s["label"] == PANCREATIC_LESION).sum()))
    return best["image"].unsqueeze(0), best["label"].unsqueeze(0)


def lesion_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.argmax(logits, dim=1, keepdim=True) == PANCREATIC_LESION
    reference = target == PANCREATIC_LESION
    denominator = prediction.sum() + reference.sum()
    if denominator == 0:
        return float("nan")
    return float(2 * (prediction & reference).sum() / denominator)


def run_overfit(initialization: str, pretrained: str | None) -> dict:
    image, label = deterministic_lesion_patch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image, label = image.to(device), label.to(device)

    torch.manual_seed(317)
    model = build_segresnet(initialization, pretrained).to(device)
    criterion = DiceCELoss(
        include_background=False, to_onehot_y=True, softmax=True,
        lambda_dice=1.0, lambda_ce=1.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    model.train()
    first_loss = None
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), label)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = float(loss.detach())
    final_loss = float(loss.detach())

    model.eval()
    with torch.no_grad():
        logits = model(image)
    return {
        "initialization": initialization,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "lesion_dice": lesion_dice(logits, label),
        "target_lesion_voxels": int((label == PANCREATIC_LESION).sum()),
        "predicted_lesion_voxels": int(
            (torch.argmax(logits, 1, keepdim=True) == PANCREATIC_LESION).sum()
        ),
    }


@requires_data
def test_patch_is_deterministic_and_contains_lesion():
    first_image, first_label = deterministic_lesion_patch()
    second_image, second_label = deterministic_lesion_patch()

    assert torch.equal(first_image, second_image), "patch content is not reproducible"
    assert torch.equal(first_label, second_label)
    assert int((first_label == PANCREATIC_LESION).sum()) > 0, "patch has no lesion to learn"


@pytest.mark.slow
@requires_data
def test_random_initialization_overfits_one_patch():
    result = run_overfit("random", None)
    print(f"\n  random: loss {result['first_loss']:.4f} -> {result['final_loss']:.4f}, "
          f"class-28 Dice {result['lesion_dice']:.4f}, "
          f"{result['predicted_lesion_voxels']}/{result['target_lesion_voxels']} lesion voxels")

    assert result["final_loss"] < 0.5 * result["first_loss"], "loss did not fall substantially"
    assert result["lesion_dice"] > 0.5, "network cannot reproduce the lesion; debug before training"


@pytest.mark.slow
@requires_data
@requires_checkpoint
def test_suprem_initialization_overfits_one_patch():
    result = run_overfit("suprem", checkpoint_path)
    print(f"\n  suprem: loss {result['first_loss']:.4f} -> {result['final_loss']:.4f}, "
          f"class-28 Dice {result['lesion_dice']:.4f}, "
          f"{result['predicted_lesion_voxels']}/{result['target_lesion_voxels']} lesion voxels")

    assert result["final_loss"] < 0.5 * result["first_loss"]
    assert result["lesion_dice"] > 0.5
