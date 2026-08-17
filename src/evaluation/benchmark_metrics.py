"""Patient-level and tumor-level benchmark metrics for PanTS-style reporting.

Supplies the five quantities the PanTS leaderboard reports -- P-Sen, T-Sen,
specificity, AUC and DSC -- from per-case masks and one continuous patient
score. Nothing here changes a prediction; this module only measures.

Two definitions are ours rather than the benchmark's, because the benchmark
does not publish them:

* T-Sen needs a rule for deciding that a predicted blob *is* a given true
  tumor. The PanTS README says only "correctly localized". We use
  26-connected components and a maximum-cardinality one-to-one matching with
  any-overlap edges, which is stated wherever the number is reported.
* AUC needs one continuous score per patient. We use the maximum class-28
  softmax over the source-restored probability map resampled to 1 mm, adapting
  the maximum-probability score used by the public R-Super evaluation code.

One-to-one matching matters: without it, a single large false blob touching
three separate true tumors would count as three detections.
"""

from __future__ import annotations

from typing import Any

import numpy as np


LESION_COMPONENT_CONNECTIVITY = 26

# scipy indexes structuring elements by how many axes a neighbour may differ
# on: 1 -> faces (6), 2 -> faces+edges (18), 3 -> everything (26).
_CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


def label_components(
    mask: np.ndarray,
    connectivity: int = LESION_COMPONENT_CONNECTIVITY,
) -> tuple[np.ndarray, int]:
    """Label 3D connected components of a boolean mask.

    Returns an int32 label volume (0 = background, 1..n = components) and the
    component count. Same connectivity convention as the frozen postprocessing
    rule, so "component" means the same thing in both places.
    """
    from scipy import ndimage

    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask; got shape {mask.shape}")
    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError(f"connectivity must be one of {sorted(_CONNECTIVITY_RANK)}")

    structure = ndimage.generate_binary_structure(3, _CONNECTIVITY_RANK[connectivity])
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labels, int(count)


def overlap_edges(
    truth_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> set[tuple[int, int]]:
    """Pairs ``(truth_component, predicted_component)`` sharing >=1 voxel.

    Computed only where both label volumes are non-zero, so the cost scales
    with the size of the intersection rather than the volume.
    """
    both = (truth_labels > 0) & (predicted_labels > 0)
    if not both.any():
        return set()
    pairs = np.stack((truth_labels[both], predicted_labels[both]), axis=1)
    return {(int(a), int(b)) for a, b in np.unique(pairs, axis=0)}


def match_components_one_to_one(
    truth_labels: np.ndarray,
    truth_count: int,
    predicted_labels: np.ndarray,
    predicted_count: int,
) -> list[tuple[int, int]]:
    """Maximum-cardinality one-to-one matching between true and predicted tumors.

    An edge exists iff the two components share at least one voxel. The result
    is the largest possible set of disjoint pairs, so each true tumor is
    credited to at most one prediction and each prediction can vouch for at
    most one true tumor.

    Maximum-cardinality, not greedy: a greedy pass over components in index
    order can consume a prediction that was the only partner of a later true
    tumor, under-counting detections in a way that depends on labelling order.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    if truth_count == 0 or predicted_count == 0:
        return []

    edges = overlap_edges(truth_labels, predicted_labels)
    if not edges:
        return []

    rows = np.fromiter((t - 1 for t, _ in edges), dtype=np.int32, count=len(edges))
    columns = np.fromiter((p - 1 for _, p in edges), dtype=np.int32, count=len(edges))
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.uint8), (rows, columns)),
        shape=(truth_count, predicted_count),
    )
    # Values are ignored; only the sparsity pattern defines the graph.
    matched_column = maximum_bipartite_matching(graph, perm_type="column")
    return [
        (index + 1, int(column) + 1)
        for index, column in enumerate(matched_column)
        if column >= 0
    ]


def dice_from_counts(intersection: int, predicted: int, target: int) -> float:
    """``2|P n G| / (|P| + |G|)``; NaN when both sets are empty.

    Empty/empty is genuinely undefined, not perfect. Scoring it 1.0 would let
    lesion-negative scans inflate a mean taken over lesion-positive scans.
    """
    denominator = int(predicted) + int(target)
    if denominator == 0:
        return float("nan")
    return float(2 * int(intersection) / denominator)


def micro_dice(
    intersections: np.ndarray,
    predicted: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Pooled ("micro") Dice: sum the voxels first, divide once.

    Weights each scan by its lesion size, so one large tumor can dominate.
    The macro mean weights each patient equally instead. They answer different
    questions and are reported side by side rather than interchanged.
    """
    denominator = float(np.sum(predicted) + np.sum(targets))
    if denominator == 0.0:
        return float("nan")
    return float(2.0 * np.sum(intersections) / denominator)


def detection_rates(truth_positive: np.ndarray, predicted_positive: np.ndarray) -> dict[str, Any]:
    """Patient-level confusion counts, P-Sen and specificity.

    ``truth_positive`` and ``predicted_positive`` are boolean per patient.
    Both cohorts must be non-empty: a sensitivity with no positives, or a
    specificity with no negatives, is a division by zero dressed up as a rate.
    """
    truth = np.asarray(truth_positive, dtype=bool)
    predicted = np.asarray(predicted_positive, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {predicted.shape}")
    if truth.ndim != 1:
        raise ValueError("expected one boolean per patient")
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0:
        raise ValueError("no lesion-positive patients; P-Sen is undefined")
    if negatives == 0:
        raise ValueError("no lesion-negative patients; specificity is undefined")

    true_positive = int((truth & predicted).sum())
    false_negative = positives - true_positive
    false_positive = int((~truth & predicted).sum())
    true_negative = negatives - false_positive
    return {
        "patients": int(truth.size),
        "positives": positives,
        "negatives": negatives,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "patient_sensitivity": true_positive / positives,
        "specificity": true_negative / negatives,
        "false_positive_rate": false_positive / negatives,
    }


def tumor_sensitivity(matched: int, total: int) -> float:
    """Matched true tumors over all true tumors; NaN if there are none."""
    if total == 0:
        return float("nan")
    return float(matched) / float(total)


def roc_and_auc(truth_positive: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """ROC points and AUC from one continuous score per patient.

    A continuous score is what makes an ROC curve possible: sweeping a
    threshold over it traces out the whole sensitivity/specificity trade-off.
    A hard 0/1 prediction yields a single point, not a curve.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    truth = np.asarray(truth_positive, dtype=bool)
    values = np.asarray(score, dtype=float)
    if truth.shape != values.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {values.shape}")
    if truth.all() or not truth.any():
        raise ValueError("AUC needs both classes present")
    if not np.isfinite(values).all():
        raise ValueError("patient scores contain non-finite values")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, values)
    return {
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "thresholds": thresholds,
        "auc": float(roc_auc_score(truth, values)),
    }


def bootstrap_auc_ci(
    truth_positive: np.ndarray,
    score: np.ndarray,
    *,
    resamples: int = 2000,
    seed: int = 317,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Stratified patient-level bootstrap percentile interval for the AUC.

    Positives and negatives are resampled with replacement *separately*, so
    every replicate keeps the real 151/750 prevalence and both classes are
    always present. The interval describes sampling variability of this cohort
    under this frozen model; it is not a claim about other cohorts.
    """
    from sklearn.metrics import roc_auc_score

    truth = np.asarray(truth_positive, dtype=bool)
    values = np.asarray(score, dtype=float)
    if truth.shape != values.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {values.shape}")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    positive_index = np.flatnonzero(truth)
    negative_index = np.flatnonzero(~truth)
    if positive_index.size == 0 or negative_index.size == 0:
        raise ValueError("stratified bootstrap needs both classes present")

    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        chosen = np.concatenate(
            (
                generator.choice(positive_index, positive_index.size, replace=True),
                generator.choice(negative_index, negative_index.size, replace=True),
            )
        )
        draws[index] = roc_auc_score(truth[chosen], values[chosen])

    tail = (1.0 - confidence) / 2.0
    return {
        "auc_bootstrap_mean": float(draws.mean()),
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
    }


def maximum_probability_at_isotropic_mm(
    probability: np.ndarray,
    spacing: tuple[float, float, float],
    target_mm: float = 1.0,
) -> float:
    """Maximum voxel probability after linear resampling to isotropic ``target_mm``.

    Resampling before taking the maximum is what the public R-Super evaluation
    code does, and it matters: a scan sampled at 5 mm slices and one at 0.5 mm
    otherwise contribute maxima drawn from very different voxel supports.
    Linear interpolation cannot exceed the input maximum, so this score is
    conservative by construction.
    """
    from scipy import ndimage

    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D probability map; got shape {array.shape}")
    if array.size == 0:
        raise ValueError("empty probability map")
    if target_mm <= 0:
        raise ValueError("target spacing must be positive")
    if len(spacing) != 3 or not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"invalid source spacing {spacing}")

    zoom = tuple(float(s) / float(target_mm) for s in spacing)
    if np.allclose(zoom, 1.0):
        return float(array.max())
    resampled = ndimage.zoom(array, zoom, order=1)
    return float(resampled.max())


def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel size in mm along each array axis, from a 4x4 affine.

    The column norms give the physical step of one index increment, which is
    correct for rotated and flipped affines where the diagonal alone is not.
    """
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine; got {matrix.shape}")
    spacing = tuple(float(np.linalg.norm(matrix[:3, axis])) for axis in range(3))
    if not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"degenerate affine produced spacing {spacing}")
    return spacing
