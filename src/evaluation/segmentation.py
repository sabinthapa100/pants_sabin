"""Model-agnostic semantic segmentation metrics."""

from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    """Compute Dice for one integer semantic label.

    Empty/empty is defined as 1.0; empty/non-empty is 0.0.
    """
    pred_mask = pred == label
    target_mask = target == label
    denominator = int(pred_mask.sum()) + int(target_mask.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred_mask, target_mask).sum())
    return 2.0 * intersection / denominator


def per_class_dice(
    pred: np.ndarray,
    target: np.ndarray,
    labels: list[int] | tuple[int, ...],
) -> dict[int, float]:
    """Compute Dice for each requested label."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    return {label: dice_score(pred, target, label) for label in labels}
