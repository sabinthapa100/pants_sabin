import numpy as np

from src.evaluation.segmentation import dice_score, per_class_dice


def test_dice_perfect_overlap():
    target = np.array([[[0, 28], [28, 0]]])
    assert dice_score(target, target, 28) == 1.0


def test_dice_partial_overlap():
    target = np.array([[[28, 28], [0, 0]]])
    pred = np.array([[[28, 0], [28, 0]]])
    assert dice_score(pred, target, 28) == 0.5


def test_per_class_dice_rejects_shape_mismatch():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        per_class_dice(pred, target, [28])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched prediction geometry")
