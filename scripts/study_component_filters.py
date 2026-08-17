"""Offline study of predeclared component-level filters on fold-0 development data.

NO INFERENCE. Every number is recomputed arithmetically from the frozen
component table plus the frozen case table. This is valid because the component
decomposition was verified to reproduce the frozen evaluation exactly: component
voxel counts sum to each case's predicted class-28 voxels, and component overlaps
reconstruct each case's Dice to 1e-12.

This script is retained as the auditable record of how the frozen pmax >= 0.6
rule was chosen: it holds the predeclared candidate list and reconstructs every
metric from stored measurements, so the selection can be re-derived by anyone.
It is not part of the inference pipeline.

    python scripts/study_component_filters.py \
        --components evaluation/suprem_components/components.csv \
        --cases evaluation/suprem/evaluation_cases.csv \
        --frozen-summary evaluation/suprem/evaluation_summary.json \
        --output evaluation/postprocessing_development/

A candidate KEEPS or REJECTS whole components; it never edits voxels. Only
class-28 metrics are produced. This table cannot say what anatomical class a
rejected lesion voxel should become, so no anatomy metric is reported.

DEVELOPMENT ONLY. Thresholds are evaluated on the same fold that selected the
checkpoint, so every number here is optimistic by an unmeasured amount.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Predeclared candidates. Fixed before looking at any filtered result, round
# physical values rather than fitted quantiles, one free axis each.
VOLUME_THRESHOLDS = (50.0, 100.0, 200.0, 400.0)
PROBABILITY_THRESHOLDS = (0.4, 0.5, 0.6, 0.7)

# Reused verbatim from the frozen evaluation_summary.json. Recomputing tertiles
# after filtering would move the bins with the rule and make the size analysis
# meaningless.
FROZEN_TERTILE_EDGES = (2535.75, 7702.875000000002)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value == "True"


# --------------------------------------------------------------------------- #
# frozen inputs
# --------------------------------------------------------------------------- #


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases = {}
    for row in read_csv(path):
        dice = row["lesion_dice"]
        cases[row["case_id"]] = {
            "case_id": row["case_id"],
            "lesion_present": as_bool(row["lesion_present"]),
            "target_lesion_voxels": int(float(row["target_lesion_voxels"])),
            "target_lesion_mm3": float(row["target_lesion_mm3"]),
            "predicted_lesion_voxels": int(float(row["predicted_lesion_voxels"])),
            "frozen_dice": float("nan") if dice in ("", "nan") else float(dice),
            "frozen_detected": as_bool(row["detected"]),
            "frozen_false_positive": as_bool(row["false_positive"]),
        }
    return cases


def load_components(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv(path):
        by_case.setdefault(row["case_id"], []).append({
            "voxel_count": int(row["voxel_count"]),
            "volume_mm3": float(row["physical_volume_mm3"]),
            "prob_max": float(row["prob_max"]),
            "overlap_voxels": int(row["overlap_voxels"]),
            "overlap_with_gt_lesion": as_bool(row["overlap_with_gt_lesion"]),
        })
    return by_case


def verify_disjointness(
    cases: dict[str, dict[str, Any]],
    components: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Prove that summing component voxels and overlaps is legitimate.

    ``ndimage.label`` partitions the mask by construction, but the arithmetic in
    this study depends on it, so it is checked against the frozen evaluation
    rather than assumed. Two independent checks: voxel counts must sum to the
    case total (no voxel counted twice or dropped), and overlaps must never
    exceed the ground-truth lesion (which double counting would violate).
    """
    voxel_mismatch, overlap_excess, dice_mismatch = [], [], []
    for case_id, rows in components.items():
        case = cases[case_id]
        if sum(r["voxel_count"] for r in rows) != case["predicted_lesion_voxels"]:
            voxel_mismatch.append(case_id)
        overlap = sum(r["overlap_voxels"] for r in rows)
        if overlap > case["target_lesion_voxels"]:
            overlap_excess.append(case_id)
        if case["lesion_present"] and case["target_lesion_voxels"]:
            denominator = case["predicted_lesion_voxels"] + case["target_lesion_voxels"]
            if abs(2 * overlap / denominator - case["frozen_dice"]) > 1e-12:
                dice_mismatch.append(case_id)
    return {
        "cases_with_components": len(components),
        "voxel_sum_mismatches": voxel_mismatch,
        "overlap_exceeds_ground_truth": overlap_excess,
        "dice_reconstruction_mismatches": dice_mismatch,
        "disjointness_verified": not (voxel_mismatch or overlap_excess or dice_mismatch),
    }


# --------------------------------------------------------------------------- #
# candidate evaluation
# --------------------------------------------------------------------------- #


def apply_rule(
    cases: dict[str, dict[str, Any]],
    components: dict[str, list[dict[str, Any]]],
    keep: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Class-28 metrics for one KEEP/REJECT rule, computed arithmetically."""
    positives, negative_fp = [], 0
    true_kept = true_removed = false_kept = false_removed = 0

    for case_id, case in cases.items():
        rows = components.get(case_id, [])
        retained = [r for r in rows if keep(r)]

        for row in rows:
            kept = keep(row)
            if row["overlap_with_gt_lesion"]:
                true_kept += kept
                true_removed += not kept
            else:
                false_kept += kept
                false_removed += not kept

        if not case["lesion_present"]:
            negative_fp += bool(retained)
            continue

        accepted_voxels = sum(r["voxel_count"] for r in retained)
        accepted_overlap = sum(r["overlap_voxels"] for r in retained)
        denominator = accepted_voxels + case["target_lesion_voxels"]
        dice = (2 * accepted_overlap / denominator) if denominator else 0.0

        positives.append({
            "case_id": case_id,
            "target_lesion_mm3": case["target_lesion_mm3"],
            "dice": dice,
            "any_prediction": bool(retained),
            "positive_overlap": accepted_overlap > 0,
        })

    return summarize(positives, negative_fp, len(cases),
                     {"true_kept": true_kept, "true_removed": true_removed,
                      "false_kept": false_kept, "false_removed": false_removed})


def summarize(
    positives: list[dict[str, Any]],
    negative_fp: int,
    total_cases: int,
    components: dict[str, int],
) -> dict[str, Any]:
    dice = np.array([p["dice"] for p in positives])
    n_positive = len(positives)
    n_negative = total_cases - n_positive

    group_c = sum(p["positive_overlap"] for p in positives)
    group_b = sum(p["any_prediction"] and not p["positive_overlap"] for p in positives)
    group_a = n_positive - group_b - group_c

    result = {
        "positive_cases": n_positive,
        "mean_dice": float(dice.mean()),
        "median_dice": float(np.median(dice)),
        "p25_dice": float(np.percentile(dice, 25)),
        "p75_dice": float(np.percentile(dice, 75)),
        "zero_dice_count": int((dice == 0.0).sum()),
        "zero_dice_fraction": float((dice == 0.0).mean()),
        "A_no_retained_component": group_a,
        "B_retained_zero_overlap": group_b,
        "C_positive_overlap": group_c,
        "A_fraction": group_a / n_positive,
        "B_fraction": group_b / n_positive,
        "C_fraction": group_c / n_positive,
        "positive_overlap_rate": group_c / n_positive,
        "any_prediction_rate": sum(p["any_prediction"] for p in positives) / n_positive,
        "negative_cases": n_negative,
        "false_positive_cases": negative_fp,
        "false_positive_rate": negative_fp / n_negative,
        "internal_specificity": 1.0 - negative_fp / n_negative,
        "components": components,
        "size_bins": stratify(positives),
    }
    return result


def stratify(positives: list[dict[str, Any]]) -> dict[str, Any]:
    """Size behaviour using the FROZEN tertile edges, never recomputed."""
    low, high = FROZEN_TERTILE_EDGES
    bins: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for row in positives:
        volume = row["target_lesion_mm3"]
        name = "small" if volume <= low else ("medium" if volume <= high else "large")
        bins[name].append(row)

    out = {}
    for name, members in bins.items():
        if not members:
            out[name] = {"cases": 0}
            continue
        dice = np.array([m["dice"] for m in members])
        out[name] = {
            "cases": len(members),
            "mean_dice": float(dice.mean()),
            "positive_overlap_fraction": float(np.mean([m["positive_overlap"] for m in members])),
            "any_prediction_fraction": float(np.mean([m["any_prediction"] for m in members])),
            "zero_dice_fraction": float((dice == 0.0).mean()),
        }
    return out


def candidates() -> list[tuple[str, str, float | None, Callable[[dict], bool]]]:
    rules: list[tuple[str, str, float | None, Callable[[dict], bool]]] = [
        ("baseline", "none", None, lambda r: True)
    ]
    rules += [(f"volume>={t:g}", "volume_mm3", t,
               lambda r, t=t: r["volume_mm3"] >= t) for t in VOLUME_THRESHOLDS]
    rules += [(f"prob_max>={t:g}", "prob_max", t,
               lambda r, t=t: r["prob_max"] >= t) for t in PROBABILITY_THRESHOLDS]
    return rules


# --------------------------------------------------------------------------- #
# pareto
# --------------------------------------------------------------------------- #


def pareto(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Dominance on (C overlap cases, false-positive patients) only.

    A candidate is dominated when another has at least as many overlap cases AND
    no more false positives, with at least one strict improvement. No scalar
    score is formed: collapsing the two axes would hide exactly the tradeoff the
    study exists to expose.
    """
    names = list(results)
    dominated: dict[str, list[str]] = {}
    for name in names:
        here = results[name]
        by = [
            other for other in names
            if other != name
            and results[other]["C_positive_overlap"] >= here["C_positive_overlap"]
            and results[other]["false_positive_cases"] <= here["false_positive_cases"]
            and (results[other]["C_positive_overlap"] > here["C_positive_overlap"]
                 or results[other]["false_positive_cases"] < here["false_positive_cases"])
        ]
        if by:
            dominated[name] = by
    return {
        "axes": ["C_positive_overlap (maximize)", "false_positive_cases (minimize)"],
        "dominated": dominated,
        "non_dominated": [n for n in names if n not in dominated],
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Descriptive rank correlation. Reported, never used as a criterion."""
    from scipy import stats
    return float(stats.spearmanr(x, y).statistic)


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    components = load_components(args.components)

    disjoint = verify_disjointness(cases, components)
    if not disjoint["disjointness_verified"]:
        raise SystemExit(f"component additivity failed: {disjoint}")

    results = {name: apply_rule(cases, components, keep) for name, _, _, keep in candidates()}

    # The baseline must reproduce the frozen evaluation exactly, or nothing
    # downstream of it means anything.
    frozen = json.loads(args.frozen_summary.read_text())
    base = results["baseline"]
    expected = {
        "positive_cases": 177, "negative_cases": 1624,
        "A_no_retained_component": 63, "B_retained_zero_overlap": 19,
        "C_positive_overlap": 95, "false_positive_cases": 201,
    }
    failures = [f"{k}: got {base[k]}, expected {v}" for k, v in expected.items() if base[k] != v]
    frozen_mean = (frozen["results"]["primary_tumor_segmentation"]
                   ["class28_dice_on_positive_cases"]["mean"])
    if abs(base["mean_dice"] - frozen_mean) > 1e-12:
        failures.append(f"mean_dice: got {base['mean_dice']!r}, frozen {frozen_mean!r}")
    if failures:
        raise SystemExit("BASELINE RECONSTRUCTION FAILED:\n  " + "\n  ".join(failures))

    all_rows = [r for rows in components.values() for r in rows]
    volumes = np.array([r["volume_mm3"] for r in all_rows])
    probabilities = np.array([r["prob_max"] for r in all_rows])
    true_mask = np.array([r["overlap_with_gt_lesion"] for r in all_rows])
    correlation = {
        "note": "descriptive only; never used to select a threshold",
        "all_components": spearman(volumes, probabilities),
        "true_overlapping": spearman(volumes[true_mask], probabilities[true_mask]),
        "false": spearman(volumes[~true_mask], probabilities[~true_mask]),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    flat_fields = [
        "candidate", "mean_dice", "median_dice", "p25_dice", "p75_dice",
        "zero_dice_count", "zero_dice_fraction",
        "A_no_retained_component", "B_retained_zero_overlap", "C_positive_overlap",
        "positive_overlap_rate", "any_prediction_rate",
        "false_positive_cases", "false_positive_rate", "internal_specificity",
        "true_components_kept", "true_components_removed",
        "false_components_kept", "false_components_removed",
        "small_mean_dice", "small_overlap", "medium_mean_dice", "medium_overlap",
        "large_mean_dice", "large_overlap",
    ]
    with open(args.output / "candidate_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "candidate": name,
                **{k: result[k] for k in flat_fields[1:15] if k in result},
                "true_components_kept": result["components"]["true_kept"],
                "true_components_removed": result["components"]["true_removed"],
                "false_components_kept": result["components"]["false_kept"],
                "false_components_removed": result["components"]["false_removed"],
                **{f"{b}_mean_dice": result["size_bins"][b]["mean_dice"]
                   for b in ("small", "medium", "large")},
                **{f"{b}_overlap": result["size_bins"][b]["positive_overlap_fraction"]
                   for b in ("small", "medium", "large")},
            })

    report = pareto(results)
    summary = {
        "study": "offline predeclared component-filter candidates, fold-0 DEVELOPMENT",
        "warning": (
            "Thresholds are evaluated on the same fold that selected best.pt. "
            "These numbers are optimistic and are NOT an unbiased performance estimate."
        ),
        "no_anatomy_metrics": (
            "The component table cannot determine the replacement class for a rejected "
            "lesion voxel, so no anatomy or macro Dice is reported here."
        ),
        "inference_performed": False,
        "checkpoint_sha256": "54bbcf0ceb530fd929d352be11bc8d7b18d22c3925deb62d54fa3d6cfb4cef50",
        "tertile_edges_mm3": list(FROZEN_TERTILE_EDGES),
        "tertile_source": "reused verbatim from evaluation/suprem/evaluation_summary.json",
        "disjointness": disjoint,
        "baseline_reconstruction": "EXACT: A/B/C, FP counts and mean Dice all match frozen",
        "volume_probability_correlation": correlation,
        "candidates": results,
    }
    (args.output / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "pareto_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({"baseline_reconstruction": "EXACT",
                      "disjointness_verified": True,
                      "correlation": correlation,
                      "pareto": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
