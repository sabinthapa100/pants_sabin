"""Model-agnostic semantic segmentation metrics.

These are internal development metrics. They are deliberately *not* called
"official PanTS" metrics: the PanTS repository publishes no evaluator, and
patient-wise/tumor-wise sensitivity are defined only in prose, without an
overlap threshold, minimum lesion size, or AUC scoring rule.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from ..data.labels import PANCREATIC_LESION


def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    empty_score: float = math.nan,
) -> float:
    """
    Dice for one integer semantic label.

    When the label is absent from *both* prediction and target the score is
    undefined and ``empty_score`` (NaN by default) is returned rather than 1.0.

    This matters for the lesion class: 8,074 of 9,000 PanTS-tr cases contain no
    lesion, so scoring empty/empty as a perfect 1.0 and averaging over all
    cases would report near-perfect tumor Dice for a model that never predicts
    a tumor at all.
    """

    pred_mask = pred == label
    target_mask = target == label
    denominator = int(pred_mask.sum()) + int(target_mask.sum())
    if denominator == 0:
        return empty_score
    intersection = int(np.logical_and(pred_mask, target_mask).sum())
    return 2.0 * intersection / denominator


def per_class_dice(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    empty_score: float = math.nan,
) -> dict[int, float]:
    """Dice for each requested label."""

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    return {label: dice_score(pred, target, label, empty_score) for label in labels}


def all_class_dice(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    empty_score: float = math.nan,
) -> dict[int, float]:
    """
    Same result as :func:`per_class_dice`, computed from three label histograms.

    ``per_class_dice`` walks the full volume twice per label. Over 28 classes
    and ~8 M voxels that is 56 passes, roughly 0.45 s per case, which across
    1,801 validation cases is a quarter of an hour spent re-reading the same
    arrays. Counting once with ``bincount`` costs three passes instead.

    The identity is exact, not approximate: for label ``c``, the voxels where
    prediction and target agree *and* equal ``c`` are precisely the
    intersection, and the two histograms give the mask sizes. The empty/empty
    convention is inherited unchanged, so a class absent from both volumes is
    ``empty_score`` rather than a free 1.0.
    """

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    flat_pred = np.asarray(pred).reshape(-1).astype(np.int64, copy=False)
    flat_target = np.asarray(target).reshape(-1).astype(np.int64, copy=False)
    if flat_pred.size and (flat_pred.min() < 0 or flat_target.min() < 0):
        raise ValueError("negative label values cannot be histogrammed")

    size = 1 + max(
        int(flat_pred.max(initial=0)), int(flat_target.max(initial=0)), int(max(labels))
    )
    agreement = flat_pred == flat_target
    intersection = np.bincount(flat_pred[agreement], minlength=size)
    pred_counts = np.bincount(flat_pred, minlength=size)
    target_counts = np.bincount(flat_target, minlength=size)

    scores: dict[int, float] = {}
    for label in labels:
        denominator = int(pred_counts[label]) + int(target_counts[label])
        scores[label] = (
            empty_score if denominator == 0
            else 2.0 * int(intersection[label]) / denominator
        )
    return scores


def lesion_case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    label: int = PANCREATIC_LESION,
) -> dict[str, Any]:
    """
    Per-case lesion outcome, keeping positive and negative cases separable.

    Returns the ground-truth and predicted lesion voxel counts, whether the
    case is lesion-positive, and the Dice score (NaN on a true-negative case,
    where Dice is undefined).
    """

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    target_voxels = int((target == label).sum())
    pred_voxels = int((pred == label).sum())
    lesion_present = target_voxels > 0

    return {
        "lesion_present": lesion_present,
        "target_voxels": target_voxels,
        "predicted_voxels": pred_voxels,
        "dice": dice_score(pred, target, label),
        "detected": lesion_present and pred_voxels > 0,
        "false_positive": (not lesion_present) and pred_voxels > 0,
    }


def summarize_lesion_metrics(case_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-case lesion results without mixing the two populations.

    Lesion Dice is averaged over lesion-positive cases only. Lesion-negative
    cases are reported separately as a false-positive rate, so a model that
    predicts nothing cannot inflate the headline number.
    """

    positives = [case for case in case_metrics if case["lesion_present"]]
    negatives = [case for case in case_metrics if not case["lesion_present"]]

    positive_dice = [case["dice"] for case in positives if not math.isnan(case["dice"])]

    return {
        "lesion_positive_cases": len(positives),
        "lesion_negative_cases": len(negatives),
        "mean_dice_on_positive_cases": (
            float(np.mean(positive_dice)) if positive_dice else math.nan
        ),
        "case_detection_rate_on_positive_cases": (
            sum(case["detected"] for case in positives) / len(positives)
            if positives
            else math.nan
        ),
        "false_positive_rate_on_negative_cases": (
            sum(case["false_positive"] for case in negatives) / len(negatives)
            if negatives
            else math.nan
        ),
        "note": (
            "internal development metrics; not the official PanTS benchmark "
            "protocol, which is not published"
        ),
    }
