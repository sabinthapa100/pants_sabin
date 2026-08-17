"""Synthetic CPU tests for the PanTS benchmark metric layer.

No medical data, no GPU, no checkpoint. Every fixture is a hand-built array
whose correct answer is known by construction, so a failure here is a bug in
the metric code rather than a property of the model.
"""

import numpy as np
import pytest

from src.evaluation.benchmark_metrics import (
    bootstrap_auc_ci,
    detection_rates,
    dice_from_counts,
    label_components,
    match_components_one_to_one,
    maximum_probability_at_isotropic_mm,
    micro_dice,
    roc_and_auc,
    spacing_from_affine,
    tumor_sensitivity,
)


# --------------------------------------------------------------------------- #
# P-Sen and specificity
# --------------------------------------------------------------------------- #


def test_confusion_arithmetic():
    truth = np.array([True, True, True, False, False, False, False])
    predicted = np.array([True, True, False, True, False, False, False])

    result = detection_rates(truth, predicted)

    assert (result["true_positive"], result["false_negative"]) == (2, 1)
    assert (result["false_positive"], result["true_negative"]) == (1, 3)
    assert result["patient_sensitivity"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(3 / 4)
    assert result["false_positive_rate"] == pytest.approx(1 / 4)
    # The four cells must exhaust the cohort.
    assert (result["true_positive"] + result["false_negative"]
            + result["false_positive"] + result["true_negative"]) == truth.size


def test_empty_positive_cohort_is_rejected():
    with pytest.raises(ValueError, match="P-Sen is undefined"):
        detection_rates(np.array([False, False]), np.array([True, False]))


def test_empty_negative_cohort_is_rejected():
    with pytest.raises(ValueError, match="specificity is undefined"):
        detection_rates(np.array([True, True]), np.array([True, False]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        detection_rates(np.array([True, False]), np.array([True]))


# --------------------------------------------------------------------------- #
# connected components
# --------------------------------------------------------------------------- #


def test_empty_mask_has_no_components():
    labels, count = label_components(np.zeros((4, 4, 4), dtype=bool))
    assert count == 0
    assert labels.max() == 0


def test_single_blob_is_one_component():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    _, count = label_components(mask)
    assert count == 1


def test_separated_blobs_are_distinct_components():
    mask = np.zeros((10, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[5, 1, 1] = True
    mask[8, 1, 1] = True
    _, count = label_components(mask)
    assert count == 3


def test_diagonal_neighbours_differ_between_6_and_26_connectivity():
    """Two corner-touching voxels: one blob under 26, two under 6."""
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True  # differs on all three axes -> corner contact only

    assert label_components(mask, connectivity=26)[1] == 1
    assert label_components(mask, connectivity=6)[1] == 2
    assert label_components(mask, connectivity=18)[1] == 2


def test_invalid_connectivity_is_rejected():
    with pytest.raises(ValueError, match="connectivity must be"):
        label_components(np.zeros((2, 2, 2), dtype=bool), connectivity=7)


# --------------------------------------------------------------------------- #
# one-to-one tumor matching
# --------------------------------------------------------------------------- #


def blobs(shape, regions):
    """Boolean volume with one blob per (x_slice, y_slice, z_slice) region."""
    mask = np.zeros(shape, dtype=bool)
    for x, y, z in regions:
        mask[x, y, z] = True
    return mask


def match_count(truth_mask, predicted_mask):
    truth_labels, truth_count = label_components(truth_mask)
    predicted_labels, predicted_count = label_components(predicted_mask)
    matched = match_components_one_to_one(
        truth_labels, truth_count, predicted_labels, predicted_count
    )
    return len(matched), truth_count, predicted_count


SHAPE = (8, 5, 14)


def test_one_truth_one_overlapping_prediction_matches():
    truth = blobs(SHAPE, [(slice(1, 2), slice(1, 2), slice(2, 5))])
    predicted = blobs(SHAPE, [(slice(1, 2), slice(1, 2), slice(3, 6))])
    assert match_count(truth, predicted) == (1, 1, 1)


def test_prediction_in_the_wrong_place_matches_nothing():
    truth = blobs(SHAPE, [(slice(1, 2), slice(1, 2), slice(2, 5))])
    predicted = blobs(SHAPE, [(slice(6, 7), slice(3, 4), slice(10, 13))])
    assert match_count(truth, predicted) == (0, 1, 1)


def test_one_prediction_spanning_two_truths_matches_only_one():
    """The core reason matching must be one-to-one."""
    truth = blobs(SHAPE, [
        (slice(1, 2), slice(1, 2), slice(2, 4)),
        (slice(1, 2), slice(1, 2), slice(8, 10)),
    ])
    predicted = blobs(SHAPE, [(slice(1, 2), slice(1, 2), slice(3, 9))])
    matched, truth_count, predicted_count = match_count(truth, predicted)
    assert (truth_count, predicted_count) == (2, 1)
    assert matched == 1


def test_two_predictions_on_one_truth_match_only_once():
    truth = blobs(SHAPE, [(slice(1, 2), slice(1, 2), slice(2, 9))])
    predicted = blobs(SHAPE, [
        (slice(1, 2), slice(1, 2), slice(2, 4)),
        (slice(1, 2), slice(1, 2), slice(6, 8)),
    ])
    matched, truth_count, predicted_count = match_count(truth, predicted)
    assert (truth_count, predicted_count) == (1, 2)
    assert matched == 1


def test_two_truths_with_two_correct_predictions_match_twice():
    truth = blobs(SHAPE, [
        (slice(1, 2), slice(1, 2), slice(2, 4)),
        (slice(1, 2), slice(1, 2), slice(8, 10)),
    ])
    predicted = blobs(SHAPE, [
        (slice(1, 2), slice(1, 2), slice(2, 4)),
        (slice(1, 2), slice(1, 2), slice(8, 10)),
    ])
    assert match_count(truth, predicted) == (2, 2, 2)


def test_matching_is_maximum_cardinality_not_greedy():
    """A layout where first-come-first-served loses a detection.

    Truth 1 touches predictions 1 and 2; truth 2 touches only prediction 1.
    Greedy in component order hands prediction 1 to truth 1 and then has
    nothing left for truth 2, scoring 1. The maximum matching pairs truth 1
    with prediction 2 and truth 2 with prediction 1, scoring 2.
    """
    truth = np.zeros(SHAPE, dtype=bool)
    truth[1:5, 1, 2:5] = True     # truth 1: spans x=1..4, reaches both predictions
    truth[1, 1, 8:11] = True      # truth 2: isolated in z

    predicted = np.zeros(SHAPE, dtype=bool)
    predicted[1, 1, 4:9] = True   # prediction 1: touches truth 1 (z=4) and truth 2 (z=8)
    predicted[4, 1, 2] = True     # prediction 2: touches truth 1 only

    truth_labels, truth_count = label_components(truth)
    predicted_labels, predicted_count = label_components(predicted)
    assert (truth_count, predicted_count) == (2, 2)

    matched = match_components_one_to_one(
        truth_labels, truth_count, predicted_labels, predicted_count
    )
    assert len(matched) == 2, "maximum-cardinality matching lost a detection"

    # Confirm the fixture really does defeat a greedy pass, so this test keeps
    # its meaning if the implementation is ever swapped.
    from src.evaluation.benchmark_metrics import overlap_edges

    edges = overlap_edges(truth_labels, predicted_labels)
    taken: set[int] = set()
    greedy = 0
    for truth_index in range(1, truth_count + 1):
        for predicted_index in sorted(p for t, p in edges if t == truth_index):
            if predicted_index not in taken:
                taken.add(predicted_index)
                greedy += 1
                break
    assert greedy == 1, "fixture no longer distinguishes greedy from maximum"


def test_matching_with_no_components_is_empty():
    empty = np.zeros((4, 4, 4), dtype=np.int32)
    assert match_components_one_to_one(empty, 0, empty, 0) == []
    assert match_components_one_to_one(empty, 0, empty, 3) == []
    assert match_components_one_to_one(empty, 2, empty, 0) == []


def test_tumor_sensitivity_arithmetic():
    assert tumor_sensitivity(3, 4) == pytest.approx(0.75)
    assert tumor_sensitivity(0, 5) == 0.0
    assert np.isnan(tumor_sensitivity(0, 0))


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #


def test_perfect_ranking_gives_auc_one():
    truth = np.array([False, False, True, True])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_and_auc(truth, score)["auc"] == pytest.approx(1.0)


def test_reversed_ranking_gives_auc_zero():
    truth = np.array([False, False, True, True])
    score = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_and_auc(truth, score)["auc"] == pytest.approx(0.0)


def test_all_scores_equal_gives_auc_one_half():
    truth = np.array([False, False, True, True])
    score = np.full(4, 0.42)
    assert roc_and_auc(truth, score)["auc"] == pytest.approx(0.5)


def test_tied_scores_are_handled_as_half_credit():
    """One tie between a positive and a negative costs exactly half a pair."""
    truth = np.array([False, False, True, True])
    score = np.array([0.1, 0.5, 0.5, 0.9])
    # pairs: (0.1,0.5)win (0.1,0.9)win (0.5,0.5)tie (0.5,0.9)win = 3.5/4
    assert roc_and_auc(truth, score)["auc"] == pytest.approx(0.875)


def test_roc_endpoints_are_valid():
    truth = np.array([False, False, True, True])
    score = np.array([0.1, 0.4, 0.6, 0.9])
    curve = roc_and_auc(truth, score)
    assert curve["fpr"][0] == 0.0 and curve["tpr"][0] == 0.0
    assert curve["fpr"][-1] == 1.0 and curve["tpr"][-1] == 1.0
    assert np.all(np.diff(curve["fpr"]) >= 0)
    assert np.all(np.diff(curve["tpr"]) >= 0)


def test_single_class_ground_truth_is_rejected():
    with pytest.raises(ValueError, match="both classes"):
        roc_and_auc(np.array([True, True]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="both classes"):
        roc_and_auc(np.array([False, False]), np.array([0.2, 0.8]))


def test_non_finite_scores_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        roc_and_auc(np.array([False, True]), np.array([0.2, np.nan]))


# --------------------------------------------------------------------------- #
# DSC
# --------------------------------------------------------------------------- #


def test_dice_perfect_overlap():
    assert dice_from_counts(100, 100, 100) == pytest.approx(1.0)


def test_dice_partial_overlap():
    # |P|=10, |G|=20, |P n G|=5  ->  2*5/30
    assert dice_from_counts(5, 10, 20) == pytest.approx(1 / 3)


def test_dice_empty_prediction_on_positive_target_is_zero():
    assert dice_from_counts(0, 0, 40) == 0.0


def test_dice_prediction_on_empty_target_is_zero():
    assert dice_from_counts(0, 40, 0) == 0.0


def test_dice_empty_and_empty_is_nan_not_one():
    assert np.isnan(dice_from_counts(0, 0, 0))


def test_micro_dice_pools_before_dividing():
    intersections = np.array([5, 0])
    predicted = np.array([10, 0])
    targets = np.array([20, 100])
    # macro would be mean(1/3, 0) = 0.1667; micro is 2*5/(10+120)
    assert micro_dice(intersections, predicted, targets) == pytest.approx(10 / 130)


def test_micro_dice_all_empty_is_nan():
    zeros = np.zeros(3)
    assert np.isnan(micro_dice(zeros, zeros, zeros))


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #


def cohort():
    generator = np.random.default_rng(0)
    truth = np.array([True] * 30 + [False] * 70)
    score = np.where(truth, generator.normal(0.7, 0.2, 100), generator.normal(0.3, 0.2, 100))
    return truth, np.clip(score, 0.0, 1.0)


def test_bootstrap_is_deterministic_under_seed_317():
    truth, score = cohort()
    first = bootstrap_auc_ci(truth, score, resamples=200, seed=317)
    second = bootstrap_auc_ci(truth, score, resamples=200, seed=317)
    assert first == second


def test_bootstrap_bounds_stay_inside_unit_interval_and_bracket_the_point():
    truth, score = cohort()
    point = roc_and_auc(truth, score)["auc"]
    interval = bootstrap_auc_ci(truth, score, resamples=400, seed=317)
    assert 0.0 <= interval["ci_low"] <= interval["ci_high"] <= 1.0
    assert interval["ci_low"] <= point <= interval["ci_high"]


def test_bootstrap_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_auc_ci(np.array([True, True]), np.array([0.1, 0.9]), resamples=10)


# --------------------------------------------------------------------------- #
# patient score
# --------------------------------------------------------------------------- #


def test_isotropic_maximum_is_unchanged_when_already_1mm():
    probability = np.zeros((6, 6, 6), dtype=np.float32)
    probability[3, 3, 3] = 0.83
    assert maximum_probability_at_isotropic_mm(probability, (1.0, 1.0, 1.0)) == pytest.approx(0.83)


def test_linear_resampling_cannot_exceed_the_input_maximum():
    generator = np.random.default_rng(317)
    probability = generator.random((12, 10, 8), dtype=np.float32)
    for spacing in [(2.5, 0.8, 0.8), (0.6, 0.6, 5.0), (1.5, 1.5, 1.5)]:
        value = maximum_probability_at_isotropic_mm(probability, spacing)
        assert value <= probability.max() + 1e-6
        assert 0.0 <= value <= 1.0


def test_degenerate_spacing_is_rejected():
    probability = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="invalid source spacing"):
        maximum_probability_at_isotropic_mm(probability, (1.0, 0.0, 1.0))


def test_spacing_from_affine_handles_rotation_and_flips():
    affine = np.diag([-1.5, 2.0, -3.0, 1.0])
    assert spacing_from_affine(affine) == pytest.approx((1.5, 2.0, 3.0))

    theta = np.pi / 3
    rotation = np.eye(4)
    rotation[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]) @ np.diag([0.7, 0.7, 2.5])
    assert spacing_from_affine(rotation) == pytest.approx((0.7, 0.7, 2.5))


def test_singular_affine_is_rejected():
    with pytest.raises(ValueError, match="degenerate affine"):
        spacing_from_affine(np.diag([1.0, 0.0, 1.0, 1.0]))
