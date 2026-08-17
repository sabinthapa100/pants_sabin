"""Aggregate per-case metric records into the five PanTS-style benchmark numbers.

Consumes what ``complete_pants_metrics.py`` wrote and produces P-Sen, T-Sen,
specificity, AUC (with a stratified bootstrap interval) and DSC, plus the ROC
points and figure. Pure arithmetic on a CSV: no model, no GPU, no medical
image is opened here, so it is cheap to re-run and easy to audit.

Hard gates compare the recomputed counts against the values the frozen
evaluation already reported. A mismatch means the two passes did not see the
same predictions, which invalidates everything downstream, so the script
refuses to write results rather than publishing a silently different number.

    python scripts/aggregate_pants_metrics.py \
        --input evaluation/pants_te_metric_completion \
        --gate-positives 151 --gate-negatives 750 \
        --gate-true-positive 104 --gate-false-positive 48 \
        --gate-overlap-cases 92 --gate-mean-dice 0.300721
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_metrics import (  # noqa: E402
    bootstrap_auc_ci,
    detection_rates,
    micro_dice,
    roc_and_auc,
    tumor_sensitivity,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    seen = {row["case_id"] for row in rows}
    if len(seen) != len(rows):
        raise SystemExit(f"{path} contains duplicate case IDs")
    return rows


def as_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    def column(name, cast):
        return np.array([cast(row[name]) for row in rows])

    boolean = lambda value: value == "True"  # noqa: E731
    return {
        "case_id": np.array([row["case_id"] for row in rows]),
        "truth": column("ground_truth_patient_positive", boolean),
        "predicted": column("frozen_hard_patient_prediction", boolean),
        "score": column("maximum_p28_patient_score", float),
        "target_voxels": column("target_lesion_voxels", int),
        "predicted_voxels": column("predicted_lesion_voxels", int),
        "intersection": column("intersection_voxels", int),
        "dice": column("lesion_dice", float),
        "gt_components": column("gt_tumor_components", int),
        "predicted_components": column("predicted_tumor_components", int),
        "matched": column("matched_gt_tumors", int),
    }


def check_gates(data: dict[str, np.ndarray], detection: dict[str, Any], args) -> list[str]:
    """Compare recomputed hard counts against the frozen evaluation."""
    overlap_cases = int(((data["truth"]) & (data["intersection"] > 0)).sum())
    positive_dice = data["dice"][data["truth"]]
    checks = [
        ("cases", len(data["case_id"]), args.gate_cases),
        ("positives", detection["positives"], args.gate_positives),
        ("negatives", detection["negatives"], args.gate_negatives),
        ("hard true positives", detection["true_positive"], args.gate_true_positive),
        ("hard false positives", detection["false_positive"], args.gate_false_positive),
        ("positive-overlap cases", overlap_cases, args.gate_overlap_cases),
    ]
    failures = [
        f"{name}: recomputed {actual}, expected {expected}"
        for name, actual, expected in checks
        if expected is not None and int(actual) != int(expected)
    ]
    if args.gate_mean_dice is not None:
        mean = float(np.nanmean(positive_dice))
        if abs(mean - args.gate_mean_dice) > args.gate_dice_tolerance:
            failures.append(
                f"mean positive-case DSC: recomputed {mean:.9f}, "
                f"expected {args.gate_mean_dice:.9f}"
            )
    return failures


def summarize(data: dict[str, np.ndarray], tumors: list[dict[str, Any]], args) -> dict[str, Any]:
    detection = detection_rates(data["truth"], data["predicted"])
    failures = check_gates(data, detection, args)
    if failures:
        raise SystemExit(
            "HARD CONSISTENCY GATE FAILED - refusing to report metrics:\n  "
            + "\n  ".join(failures)
        )

    truth = data["truth"]
    positive_dice = data["dice"][truth]
    total_tumors = int(data["gt_components"].sum())
    matched_tumors = int(data["matched"].sum())

    curve = roc_and_auc(truth, data["score"])
    interval = bootstrap_auc_ci(
        truth, data["score"], resamples=args.resamples, seed=args.seed
    )

    return {
        "cohort": {
            "cases": int(truth.size),
            "lesion_positive": detection["positives"],
            "lesion_negative": detection["negatives"],
        },
        "p_sen": {
            "value": detection["patient_sensitivity"],
            "numerator": detection["true_positive"],
            "denominator": detection["positives"],
            "definition": (
                "lesion-positive scans whose final postprocessed source-geometry label "
                "map contains at least one class-28 voxel, location not required"
            ),
        },
        "specificity": {
            "value": detection["specificity"],
            "numerator": detection["true_negative"],
            "denominator": detection["negatives"],
            "definition": (
                "lesion-negative scans whose final postprocessed label map contains "
                "no class-28 voxel"
            ),
        },
        "t_sen": {
            "value": tumor_sensitivity(matched_tumors, total_tumors),
            "numerator": matched_tumors,
            "denominator": total_tumors,
            "unmatched_ground_truth_tumors": total_tumors - matched_tumors,
            "unmatched_predicted_components": int(
                (data["predicted_components"] - data["matched"]).sum()
            ),
            "predicted_components_total": int(data["predicted_components"].sum()),
            "definition": (
                "26-connected components; maximum-cardinality one-to-one matching "
                "with an edge wherever a true and a predicted component share >=1 voxel"
            ),
        },
        "auc": {
            "value": curve["auc"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "bootstrap": {
                "resamples": interval["resamples"],
                "seed": interval["seed"],
                "confidence": interval["confidence"],
                "method": "stratified patient-level percentile bootstrap",
            },
            "patient_score": (
                "maximum class-28 softmax over the source-restored probability map, "
                "linearly resampled to 1 mm isotropic"
            ),
            "deployed_hard_point": {
                "fpr": detection["false_positive_rate"],
                "tpr": detection["patient_sensitivity"],
                "note": "frozen pmax>=0.6 rule; a single point, not part of the curve",
            },
        },
        "dsc": {
            "primary_macro_mean_on_positive_cases": float(np.nanmean(positive_dice)),
            "n": int(positive_dice.size),
            "median": float(np.nanmedian(positive_dice)),
            "std": float(np.nanstd(positive_dice, ddof=1)),
            "zero_dice_cases": int((positive_dice == 0.0).sum()),
            "zero_dice_fraction": float((positive_dice == 0.0).mean()),
            "micro_pooled": micro_dice(
                data["intersection"][truth],
                data["predicted_voxels"][truth],
                data["target_voxels"][truth],
            ),
            "definition": (
                "2|P n G| / (|P| + |G|) for class 28 per scan, averaged over "
                "ground-truth-positive scans; empty/empty never enters the mean"
            ),
        },
        "tumor_records": {
            "rows": len(tumors),
            "matched": sum(1 for row in tumors if row["matched"] == "True"),
        },
    }, curve


def write_outputs(summary: dict[str, Any], curve: dict[str, Any], args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "benchmark_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with open(args.output / "roc_points.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fpr", "tpr", "threshold"])
        for false_rate, true_rate, threshold in zip(
            curve["fpr"], curve["tpr"], curve["thresholds"]
        ):
            writer.writerow([f"{false_rate:.10f}", f"{true_rate:.10f}", f"{threshold:.10f}"])

    with open(args.output / "benchmark_metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "p_sen", "t_sen", "specificity", "auc", "dsc"])
        writer.writerow([
            args.model_label,
            f"{summary['p_sen']['value']:.4f}",
            f"{summary['t_sen']['value']:.4f}",
            f"{summary['specificity']['value']:.4f}",
            f"{summary['auc']['value']:.4f}",
            f"{summary['dsc']['primary_macro_mean_on_positive_cases']:.4f}",
        ])


def draw_roc(summary: dict[str, Any], curve: dict[str, Any], args) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auc = summary["auc"]
    point = auc["deployed_hard_point"]

    figure, axis = plt.subplots(figsize=(5.6, 5.4))
    axis.plot([0, 1], [0, 1], color="#9aa0a6", lw=1.0, ls="--", label="chance")
    axis.plot(
        curve["fpr"], curve["tpr"], color="#4c72b0", lw=2.0,
        label=f"max class-28 softmax (AUC {auc['value']:.3f})",
    )
    axis.plot(
        point["fpr"], point["tpr"], marker="o", ms=9, color="#c44e52", ls="none",
        markeredgecolor="white", markeredgewidth=1.2,
        label=f"deployed pmax>=0.6 rule ({point['fpr']:.3f}, {point['tpr']:.3f})",
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("1 - Specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_title(args.figure_title, fontsize=11)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    axis.text(
        0.02, -0.16,
        f"95% CI [{auc['ci_low']:.3f}, {auc['ci_high']:.3f}] "
        f"({auc['bootstrap']['resamples']} stratified bootstrap resamples, seed "
        f"{auc['bootstrap']['seed']}). n = {summary['cohort']['cases']} "
        f"({summary['cohort']['lesion_positive']} positive / "
        f"{summary['cohort']['lesion_negative']} negative).",
        transform=axis.transAxes, fontsize=7.5, va="top", wrap=True,
    )
    figure.tight_layout()
    destination = args.output / "figures" / "pants_te_roc.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-label", default="SegResNet (SuPreM init)")
    parser.add_argument("--figure-title", default="PanTS-te patient-level ROC")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--gate-cases", type=int, default=None)
    parser.add_argument("--gate-positives", type=int, default=None)
    parser.add_argument("--gate-negatives", type=int, default=None)
    parser.add_argument("--gate-true-positive", type=int, default=None)
    parser.add_argument("--gate-false-positive", type=int, default=None)
    parser.add_argument("--gate-overlap-cases", type=int, default=None)
    parser.add_argument("--gate-mean-dice", type=float, default=None)
    parser.add_argument("--gate-dice-tolerance", type=float, default=1e-5)
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output = args.output or args.input

    rows = read_rows(args.input / "per_case_metric_completion.csv")
    tumor_path = args.input / "per_tumor_matching.csv"
    with open(tumor_path, newline="") as handle:
        tumors = list(csv.DictReader(handle))

    summary, curve = summarize(as_arrays(rows), tumors, args)
    write_outputs(summary, curve, args)
    if not args.no_figure:
        print(f"figure -> {draw_roc(summary, curve, args)}")

    print(json.dumps({
        "P-Sen": f"{summary['p_sen']['numerator']}/{summary['p_sen']['denominator']}"
                 f" = {summary['p_sen']['value']:.4f}",
        "T-Sen": f"{summary['t_sen']['numerator']}/{summary['t_sen']['denominator']}"
                 f" = {summary['t_sen']['value']:.4f}",
        "Spe": f"{summary['specificity']['numerator']}/{summary['specificity']['denominator']}"
               f" = {summary['specificity']['value']:.4f}",
        "AUC": f"{summary['auc']['value']:.4f} "
               f"[{summary['auc']['ci_low']:.4f}, {summary['auc']['ci_high']:.4f}]",
        "DSC": f"{summary['dsc']['primary_macro_mean_on_positive_cases']:.6f}",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
