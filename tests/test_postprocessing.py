"""Behaviour of the frozen lesion-component filter. No GPU, no data, no model."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.postprocessing import (  # noqa: E402
    LESION_PEAK_PROBABILITY,
    best_non_lesion_labels,
    filter_lesion_components,
    lesion_peak_probability_map,
)


CLASSES = 29
LESION = 28


def logits_with_lesion_blobs(
    shape: tuple[int, int, int],
    blobs: list[tuple[tuple[slice, slice, slice], float]],
    background_class: int = 0,
    runner_up_class: int = 17,
) -> torch.Tensor:
    """Build logits whose argmax is `background_class` except inside `blobs`.

    Each blob is a region plus the class-28 softmax probability it should reach.
    Inside a blob, `runner_up_class` is raised so it is the best NON-lesion
    explanation there; everywhere else all non-lesion channels stay at 0 and the
    argmax is `background_class`. That separation is what lets a test tell
    "fell back to the runner-up" apart from "was forced to background".

    With the other 28 channels contributing ``S`` to the softmax denominator,
    the lesion probability is ``e^z / (e^z + S)``; inverting it gives the logit
    for a requested probability, so the boundary tests rest on arithmetic rather
    than hand-tuned constants.
    """
    logits = torch.zeros(1, CLASSES, *shape, dtype=torch.float64)
    others = (CLASSES - 2) * math.exp(0.0) + math.exp(0.5)

    for region, probability in blobs:
        logits[(0, runner_up_class) + region] = 0.5
        z = math.log(probability * others / (1.0 - probability))
        logits[(0, LESION) + region] = z
    return logits


def test_probability_map_matches_softmax():
    torch.manual_seed(0)
    logits = torch.randn(1, CLASSES, 4, 4, 4, dtype=torch.float64) * 8
    reference = torch.softmax(logits, dim=1)[0, LESION]
    assert torch.allclose(lesion_peak_probability_map(logits), reference, atol=1e-12)


def test_dropping_the_lesion_channel_cannot_reorder_the_others():
    """The mathematical claim the fallback rests on, checked numerically.

    Softmax divides every channel by one shared denominator, so removing the
    lesion term rescales the rest by a positive constant and cannot change which
    of channels 0..27 is largest.
    """
    torch.manual_seed(1)
    logits = torch.randn(1, CLASSES, 5, 5, 5, dtype=torch.float64) * 6

    from_logits = best_non_lesion_labels(logits)
    from_full_softmax = torch.argmax(
        torch.softmax(logits, dim=1)[:, :LESION], dim=1, keepdim=True)
    from_reduced_softmax = torch.argmax(
        torch.softmax(logits[:, :LESION], dim=1), dim=1, keepdim=True)

    assert torch.equal(from_logits, from_full_softmax)
    assert torch.equal(from_logits, from_reduced_softmax)
    assert int(from_logits.max()) < LESION


def test_component_below_the_threshold_is_removed():
    logits = logits_with_lesion_blobs((8, 8, 8), [((slice(2, 5),) * 3, 0.59)])
    labels = torch.argmax(logits, dim=1, keepdim=True)
    assert (labels == LESION).sum() == 27

    result = filter_lesion_components(logits, labels, min_peak_probability=0.6)

    assert result["components_found"] == 1
    assert result["components_rejected"] == 1
    assert (result["labels"] == LESION).sum() == 0
    assert result["relabelled_voxels"] == 27


def test_component_exactly_on_the_threshold_is_retained():
    """Proves ``>=`` rather than ``>``.

    Rather than construct a peak that is exactly 0.6 in floating point -- which
    is not reliably representable through exp/log -- the component's own
    computed peak is used as the threshold. Equality then holds bit-exactly, and
    a ``>`` implementation would reject.
    """
    logits = logits_with_lesion_blobs((8, 8, 8), [((slice(2, 5),) * 3, 0.6)])
    labels = torch.argmax(logits, dim=1, keepdim=True)

    peak = float(lesion_peak_probability_map(logits)[labels[0, 0] == LESION].max())
    result = filter_lesion_components(logits, labels, min_peak_probability=peak)

    assert result["components_retained"] == 1
    assert result["components_rejected"] == 0
    assert (result["labels"] == LESION).sum() == 27
    assert result["relabelled_voxels"] == 0


def test_component_above_the_threshold_is_retained():
    logits = logits_with_lesion_blobs((8, 8, 8), [((slice(2, 5),) * 3, 0.61)])
    labels = torch.argmax(logits, dim=1, keepdim=True)

    result = filter_lesion_components(logits, labels, min_peak_probability=0.6)

    assert result["components_retained"] == 1
    assert (result["labels"] == LESION).sum() == 27


def test_two_components_are_judged_independently():
    weak = (slice(1, 3), slice(1, 3), slice(1, 3))
    strong = (slice(6, 9), slice(6, 9), slice(6, 9))
    logits = logits_with_lesion_blobs((12, 12, 12), [(weak, 0.4), (strong, 0.95)])
    labels = torch.argmax(logits, dim=1, keepdim=True)
    assert (labels == LESION).sum() == 8 + 27

    result = filter_lesion_components(logits, labels, min_peak_probability=0.6)

    assert result["components_found"] == 2
    assert result["components_retained"] == 1
    assert result["components_rejected"] == 1
    assert result["relabelled_voxels"] == 8
    filtered = result["labels"]
    assert filtered[0, 0][weak].eq(LESION).sum() == 0        # weak blob gone
    assert filtered[0, 0][strong].eq(LESION).sum() == 27     # strong blob intact


def test_rejected_voxels_become_the_best_non_lesion_class_not_background():
    """The whole point of the fallback: class 17 wins, not class 0."""
    logits = logits_with_lesion_blobs(
        (8, 8, 8), [((slice(2, 5),) * 3, 0.5)], runner_up_class=17)
    labels = torch.argmax(logits, dim=1, keepdim=True)

    result = filter_lesion_components(logits, labels, min_peak_probability=0.6)

    assert result["fallback_class_counts"] == {17: 27}
    assert result["labels"][0, 0][2:5, 2:5, 2:5].eq(17).all()
    assert result["labels"].eq(0).sum() == 8**3 - 27


def test_voxels_outside_rejected_components_are_bit_identical():
    weak = (slice(1, 3), slice(1, 3), slice(1, 3))
    strong = (slice(6, 9), slice(6, 9), slice(6, 9))
    logits = logits_with_lesion_blobs((12, 12, 12), [(weak, 0.3), (strong, 0.99)])
    labels = torch.argmax(logits, dim=1, keepdim=True)

    result = filter_lesion_components(logits, labels, min_peak_probability=0.6)

    untouched = torch.ones_like(labels, dtype=torch.bool)
    untouched[0, 0][weak] = False
    assert torch.equal(result["labels"][untouched], labels[untouched])
    assert not torch.equal(result["labels"], labels)      # something did change


def test_the_input_label_tensor_is_never_mutated():
    logits = logits_with_lesion_blobs((8, 8, 8), [((slice(2, 5),) * 3, 0.3)])
    labels = torch.argmax(logits, dim=1, keepdim=True)
    before = labels.clone()

    filter_lesion_components(logits, labels, min_peak_probability=0.6)

    assert torch.equal(labels, before)


def test_26_connectivity_merges_a_corner_touch_that_6_connectivity_splits():
    """Two cubes meeting only at a corner are one object under 26, two under 6.

    Made observable by giving the halves different peaks: under 26 they form a
    single component whose peak is the stronger one, so everything survives;
    under 6 they are judged separately and the weak half is removed.
    """
    logits = logits_with_lesion_blobs(
        (10, 10, 10),
        [((slice(2, 4), slice(2, 4), slice(2, 4)), 0.45),
         ((slice(4, 6), slice(4, 6), slice(4, 6)), 0.95)],
    )
    labels = torch.argmax(logits, dim=1, keepdim=True)
    assert (labels == LESION).sum() == 16

    merged = filter_lesion_components(
        logits, labels, min_peak_probability=0.6, connectivity=26)
    assert merged["components_found"] == 1
    assert merged["components_rejected"] == 0
    assert (merged["labels"] == LESION).sum() == 16

    split = filter_lesion_components(
        logits, labels, min_peak_probability=0.6, connectivity=6)
    assert split["components_found"] == 2
    assert split["components_rejected"] == 1
    assert (split["labels"] == LESION).sum() == 8


def test_no_lesion_prediction_is_a_no_op():
    logits = logits_with_lesion_blobs((6, 6, 6), [])
    labels = torch.argmax(logits, dim=1, keepdim=True)

    result = filter_lesion_components(logits, labels)

    assert result["components_found"] == 0
    assert result["relabelled_voxels"] == 0
    assert torch.equal(result["labels"], labels)


def test_frozen_default_threshold_is_documented_and_used():
    assert LESION_PEAK_PROBABILITY == 0.6
    logits = logits_with_lesion_blobs((8, 8, 8), [((slice(2, 5),) * 3, 0.55)])
    labels = torch.argmax(logits, dim=1, keepdim=True)

    result = filter_lesion_components(logits, labels)      # no threshold passed

    assert result["rule"]["min_peak_probability"] == 0.6
    assert result["rule"]["connectivity"] == 26
    assert result["rule"]["comparison"] == ">="
    assert result["components_rejected"] == 1


def test_lesion_channel_must_be_last():
    logits = torch.zeros(1, CLASSES, 4, 4, 4)
    with pytest.raises(ValueError, match="final channel"):
        best_non_lesion_labels(logits, lesion_class=5)


@pytest.mark.parametrize("bad", [7, 12, 0])
def test_unsupported_connectivity_is_rejected(bad):
    logits = logits_with_lesion_blobs((6, 6, 6), [])
    labels = torch.argmax(logits, dim=1, keepdim=True)
    with pytest.raises(ValueError, match="connectivity"):
        filter_lesion_components(logits, labels, connectivity=bad)
