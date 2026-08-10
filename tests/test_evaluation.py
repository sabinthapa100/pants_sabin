"""Segmentation metrics and the model output contract."""

import math

import numpy as np
import pytest
import torch

from src.evaluation.inference import lesion_mask, lesion_probability, logits_to_labels
from src.evaluation.segmentation import (
    dice_score,
    lesion_case_metrics,
    per_class_dice,
    summarize_lesion_metrics,
)


def test_dice_perfect_overlap():
    target = np.array([[[0, 28], [28, 0]]])
    assert dice_score(target, target, 28) == 1.0


def test_dice_partial_overlap():
    target = np.array([[[28, 28], [0, 0]]])
    pred = np.array([[[28, 0], [28, 0]]])
    assert dice_score(pred, target, 28) == 0.5


def test_dice_no_overlap():
    target = np.array([[[28, 28], [0, 0]]])
    pred = np.array([[[0, 0], [28, 28]]])
    assert dice_score(pred, target, 28) == 0.0


def test_empty_prediction_on_positive_case_scores_zero():
    target = np.array([[[28, 28], [0, 0]]])
    pred = np.zeros_like(target)
    assert dice_score(pred, target, 28) == 0.0


def test_absent_in_both_is_undefined_not_perfect():
    """
    A model that predicts no tumor on a tumor-free case must not be credited
    with Dice 1.0. Most PanTS-tr cases are lesion-negative, so scoring
    empty/empty as perfect would make a do-nothing model look excellent.
    """
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    assert math.isnan(dice_score(empty, empty, 28))


def test_empty_score_is_configurable():
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    assert dice_score(empty, empty, 28, empty_score=1.0) == 1.0


def test_per_class_dice_rejects_shape_mismatch():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        per_class_dice(pred, target, [28])


def test_lesion_case_metrics_positive_case():
    target = np.array([[[28, 28], [0, 0]]])
    pred = np.array([[[28, 0], [0, 0]]])
    result = lesion_case_metrics(pred, target)

    assert result["lesion_present"] is True
    assert result["target_voxels"] == 2
    assert result["predicted_voxels"] == 1
    assert result["detected"] is True
    assert result["false_positive"] is False


def test_lesion_case_metrics_false_positive():
    target = np.zeros((2, 2, 2), dtype=np.uint8)
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0] = 28
    result = lesion_case_metrics(pred, target)

    assert result["lesion_present"] is False
    assert result["false_positive"] is True
    # Dice is defined here (0.0), because the prediction is non-empty.
    # Only the empty/empty case is undefined.
    assert result["dice"] == 0.0


def test_clean_negative_case_dice_is_undefined():
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    result = lesion_case_metrics(empty, empty)

    assert result["lesion_present"] is False
    assert result["false_positive"] is False
    assert math.isnan(result["dice"])


def test_summary_keeps_populations_separate():
    """
    Lesion Dice is averaged over positive cases only; negatives are reported
    as a false-positive rate rather than folded into the same mean.
    """
    positive = np.array([[[28, 28], [0, 0]]])
    negative = np.zeros((1, 2, 2), dtype=np.uint8)

    cases = [
        lesion_case_metrics(positive, positive),          # perfect positive
        lesion_case_metrics(np.zeros_like(positive), positive),  # missed positive
        lesion_case_metrics(negative, negative),          # clean negative
        lesion_case_metrics(np.full_like(negative, 28), negative),  # false positive
    ]
    summary = summarize_lesion_metrics(cases)

    assert summary["lesion_positive_cases"] == 2
    assert summary["lesion_negative_cases"] == 2
    assert summary["mean_dice_on_positive_cases"] == 0.5   # (1.0 + 0.0) / 2
    assert summary["case_detection_rate_on_positive_cases"] == 0.5
    assert summary["false_positive_rate_on_negative_cases"] == 0.5


def test_do_nothing_model_is_not_flattered():
    """A model predicting all background scores 0, not NaN-inflated 1.0."""
    positive = np.array([[[28, 28], [0, 0]]])
    negatives = [np.zeros((1, 2, 2), dtype=np.uint8) for _ in range(20)]

    cases = [lesion_case_metrics(np.zeros_like(positive), positive)]
    cases += [lesion_case_metrics(n, n) for n in negatives]
    summary = summarize_lesion_metrics(cases)

    assert summary["mean_dice_on_positive_cases"] == 0.0
    assert summary["false_positive_rate_on_negative_cases"] == 0.0


def _logits(batch=1, classes=29, size=4):
    return torch.randn(batch, classes, size, size, size)


def test_output_contract_shapes():
    """Full semantic labels, class-28 mask, and class-28 probability map."""
    logits = _logits()

    labels = logits_to_labels(logits)
    mask = lesion_mask(logits)
    probability = lesion_probability(logits)

    assert tuple(labels.shape) == (1, 1, 4, 4, 4)
    assert tuple(mask.shape) == (1, 1, 4, 4, 4)
    assert tuple(probability.shape) == (1, 1, 4, 4, 4)
    assert int(labels.min()) >= 0 and int(labels.max()) <= 28


def test_probability_is_a_valid_probability():
    probability = lesion_probability(_logits())
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0


def test_mask_is_consistent_with_labels():
    logits = _logits()
    assert torch.equal(lesion_mask(logits).bool(), (logits_to_labels(logits) == 28))


def test_inference_helpers_reject_wrong_rank():
    with pytest.raises(ValueError):
        logits_to_labels(torch.randn(1, 29, 4, 4))
    with pytest.raises(ValueError):
        lesion_probability(torch.randn(1, 29, 4, 4))
