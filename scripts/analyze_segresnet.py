"""Plot the Random-vs-SuPreM comparison from the files training and evaluation already wrote.

    python scripts/analyze_segresnet.py \
        --suprem-run  /path/PanTS_runs/segresnet_suprem \
        --random-run  /path/PanTS_runs/segresnet_random \
        --suprem-eval /path/evaluation/suprem \
        --random-eval /path/evaluation/random \
        --output      /path/figures/

Every input is optional except ``--output``: a figure whose inputs are missing
is skipped and reported, never drawn from partial or invented data. Training
figures need only the run directories, so the first six can be produced as soon
as an arm finishes, long before evaluation exists.

Runs that were not full production runs - anything with a step cap or a case
limit - are refused rather than plotted, so a calibration artifact cannot end
up in a figure that looks like a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import CLASS_MAP, PANCREATIC_LESION  # noqa: E402

logger = logging.getLogger(__name__)

ARMS = ("suprem", "random")
LABEL = {"suprem": "SegResNet-SuPreM", "random": "SegResNet-Random"}
COLOR = {"suprem": "#0072B2", "random": "#E69F00"}     # colour-blind safe pair
FOREGROUND_CLASSES = sorted(CLASS_MAP)
DIAGNOSTIC = "DIAGNOSTIC ONLY - does not select best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suprem-run", type=Path, default=None)
    parser.add_argument("--random-run", type=Path, default=None)
    parser.add_argument("--suprem-eval", type=Path, default=None)
    parser.add_argument("--random-eval", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--allow-partial-runs",
        action="store_true",
        help="plot runs that used --max-steps-per-epoch or --limit-cases (calibration)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_history(run_dir: Path | None, allow_partial: bool) -> list[dict] | None:
    """Per-epoch records, refusing anything that is not a full production run."""
    if run_dir is None:
        return None
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        logger.warning("no history.json in %s", run_dir)
        return None

    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file() and not allow_partial:
        config = json.loads(provenance_path.read_text()).get("config", {})
        capped = config.get("max_steps_per_epoch")
        limited = config.get("limit_cases")
        if capped or limited:
            logger.error(
                "REFUSING %s: max_steps_per_epoch=%s limit_cases=%s. This is a "
                "calibration or smoke run, not a production result. Use "
                "--allow-partial-runs only if you know why you want it.",
                run_dir.name, capped, limited,
            )
            return None
    return json.loads(history_path.read_text())


def load_cases(eval_dir: Path | None) -> list[dict[str, Any]] | None:
    if eval_dir is None:
        return None
    path = eval_dir / "evaluation_cases.csv"
    if not path.is_file():
        logger.warning("no evaluation_cases.csv in %s", eval_dir)
        return None
    with open(path, newline="") as handle:
        rows = []
        for entry in csv.DictReader(handle):
            row: dict[str, Any] = {"case_id": entry["case_id"]}
            for key, value in entry.items():
                if key == "case_id":
                    continue
                if key in ("lesion_present", "detected", "false_positive"):
                    row[key] = value == "True"
                else:
                    row[key] = float(value) if value not in ("", "nan") else math.nan
            rows.append(row)
    return rows


def load_summary(eval_dir: Path | None) -> dict | None:
    if eval_dir is None:
        return None
    path = eval_dir / "evaluation_summary.json"
    return json.loads(path.read_text()) if path.is_file() else None


def selections(history: list[dict]) -> tuple[list[int], list[dict]]:
    """Only the epochs where deterministic whole-volume validation actually ran."""
    epochs = [r["epoch"] for r in history if "selection" in r]
    records = [r["selection"] for r in history if "selection" in r]
    return epochs, records


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _finish(fig, axis, output: Path, name: str, dpi: int) -> str:
    axis.grid(alpha=0.3, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    fig.tight_layout()
    path = output / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return name


def curve(histories, output, dpi, key, title, ylabel, name, logy=False) -> str | None:
    """One per-epoch quantity, both arms overlaid."""
    fig, axis = plt.subplots(figsize=(7, 4.2))
    drawn = False
    for arm in ARMS:
        history = histories.get(arm)
        if not history:
            continue
        epochs = [r["epoch"] for r in history]
        values = [r[key] for r in history if key in r]
        if len(values) != len(epochs):
            continue
        axis.plot(epochs, values, color=COLOR[arm], label=LABEL[arm], linewidth=1.8)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    if logy:
        axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    return _finish(fig, axis, output, name, dpi)


def selection_curve(histories, output, dpi, key, title, ylabel, name) -> str | None:
    """A quantity that exists only at deterministic-validation epochs."""
    fig, axis = plt.subplots(figsize=(7, 4.2))
    drawn = False
    for arm in ARMS:
        history = histories.get(arm)
        if not history:
            continue
        epochs, records = selections(history)
        values = [r.get(key, math.nan) for r in records]
        if not epochs:
            continue
        axis.plot(epochs, values, "o-", color=COLOR[arm], label=LABEL[arm],
                  linewidth=1.8, markersize=5)
        finite = [(e, v) for e, v in zip(epochs, values) if not math.isnan(v)]
        if finite and key == "mean_dice_on_positive_cases":
            best_epoch, best_value = max(finite, key=lambda pair: pair[1])
            axis.plot([best_epoch], [best_value], marker="*", markersize=16,
                      color=COLOR[arm], linestyle="none")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=11)
    return _finish(fig, axis, output, name, dpi)


def dice_distribution(cases, output, dpi) -> str | None:
    """Class-28 Dice over lesion-positive validation cases."""
    series = {
        arm: [r["lesion_dice"] for r in rows
              if r["lesion_present"] and not math.isnan(r["lesion_dice"])]
        for arm, rows in cases.items() if rows
    }
    series = {arm: values for arm, values in series.items() if values}
    if not series:
        return None

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, values in series.items():
        left.hist(values, bins=np.linspace(0, 1, 26), alpha=0.55,
                  color=COLOR[arm], label=f"{LABEL[arm]}  (n={len(values)})")
    left.set_xlabel("class-28 Dice"); left.set_ylabel("lesion-positive cases")
    left.set_title("Distribution", fontsize=11)
    left.legend(frameon=False, fontsize=9)
    left.grid(alpha=0.3, linewidth=0.6); left.spines[["top", "right"]].set_visible(False)

    order = [arm for arm in ARMS if arm in series]
    parts = right.boxplot([series[arm] for arm in order], patch_artist=True,
                          tick_labels=[LABEL[arm] for arm in order], widths=0.5)
    for patch, arm in zip(parts["boxes"], order):
        patch.set_facecolor(COLOR[arm]); patch.set_alpha(0.55)
    for arm_index, arm in enumerate(order, 1):
        values = series[arm]
        right.annotate(f"median {np.median(values):.3f}\nmean {np.mean(values):.3f}",
                       (arm_index, 1.02), ha="center", fontsize=8,
                       xycoords=("data", "axes fraction"))
    right.set_ylabel("class-28 Dice")
    right.set_title("Fold-0 validation, lesion-positive cases only", fontsize=11)
    right.grid(alpha=0.3, linewidth=0.6); right.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Pancreatic-lesion segmentation on fold-0 validation", fontsize=12)
    fig.tight_layout()
    name = "08_class28_dice_distribution.png"
    fig.savefig(output / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return name


def per_class_dice_plot(summaries, output, dpi) -> str | None:
    """Mean Dice for every foreground class, both arms."""
    available = {
        arm: summary["results"]["anatomy_aware_segmentation"]["per_class"]
        for arm, summary in summaries.items()
        if summary and summary.get("results")
    }
    if not available:
        return None

    fig, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(FOREGROUND_CLASSES))
    width = 0.38
    for offset, arm in enumerate(a for a in ARMS if a in available):
        per_class = available[arm]
        values = [per_class.get(str(c), {}).get("mean_dice", math.nan) for c in FOREGROUND_CLASSES]
        axis.bar(positions + (offset - 0.5) * width, values, width,
                 color=COLOR[arm], label=LABEL[arm], alpha=0.85)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{c} {CLASS_MAP[c]}" for c in FOREGROUND_CLASSES],
                         rotation=60, ha="right", fontsize=7)
    lesion_index = FOREGROUND_CLASSES.index(PANCREATIC_LESION)
    axis.axvspan(lesion_index - 0.5, lesion_index + 0.5, color="red", alpha=0.08)
    axis.set_ylabel("mean Dice (cases where the class is in the ground truth)")
    axis.set_title("Per-class Dice, fold-0 validation. Shaded: class 28 pancreatic lesion",
                   fontsize=11)
    axis.set_ylim(0, 1)
    return _finish(fig, axis, output, "09_per_class_dice.png", dpi)


def dice_vs_volume(cases, output, dpi) -> str | None:
    """EXPLORATORY: does 1.5 mm resampling cost the smallest lesions?"""
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    drawn = False
    for arm, rows in cases.items():
        if not rows:
            continue
        points = [(r["target_lesion_mm3"], r["lesion_dice"]) for r in rows
                  if r["lesion_present"] and r["target_lesion_mm3"] > 0
                  and not math.isnan(r["lesion_dice"])]
        if not points:
            continue
        volumes, dice = zip(*points)
        axis.scatter(volumes, dice, s=14, alpha=0.5, color=COLOR[arm], label=LABEL[arm],
                     edgecolors="none")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    axis.set_xscale("log")
    axis.set_xlabel("ground-truth lesion volume (mm$^3$, log scale)")
    axis.set_ylabel("class-28 Dice")
    axis.set_title("EXPLORATORY: lesion Dice vs lesion size (post-hoc, not tuned on)",
                   fontsize=11)
    axis.set_ylim(-0.02, 1.02)
    return _finish(fig, axis, output, "10_dice_vs_lesion_volume.png", dpi)


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    histories = {
        "suprem": load_history(args.suprem_run, args.allow_partial_runs),
        "random": load_history(args.random_run, args.allow_partial_runs),
    }
    cases = {"suprem": load_cases(args.suprem_eval), "random": load_cases(args.random_eval)}
    summaries = {"suprem": load_summary(args.suprem_eval), "random": load_summary(args.random_eval)}

    made: list[str] = []
    skipped: list[str] = []

    figures = [
        ("01_train_loss.png", lambda: curve(
            histories, args.output, args.dpi, "train_loss",
            "Training DiceCE loss", "DiceCE loss", "01_train_loss.png")),
        ("02_patch_val_loss.png", lambda: curve(
            histories, args.output, args.dpi, "patch_val_loss_diagnostic",
            f"Patch-validation DiceCE loss  ({DIAGNOSTIC})", "DiceCE loss (diagnostic)",
            "02_patch_val_loss.png")),
        ("03_learning_rate.png", lambda: curve(
            histories, args.output, args.dpi, "learning_rate",
            "Cosine learning-rate schedule", "learning rate", "03_learning_rate.png",
            logy=True)),
        ("04_monitoring_class28_dice.png", lambda: selection_curve(
            histories, args.output, args.dpi, "mean_dice_on_positive_cases",
            "Deterministic monitoring: class-28 Dice (SELECTS best.pt; star = best)",
            "mean class-28 Dice, 177 lesion-positive cases",
            "04_monitoring_class28_dice.png")),
        ("05_detection_rate.png", lambda: selection_curve(
            histories, args.output, args.dpi, "case_detection_rate_on_positive_cases",
            "INTERNAL positive-case detection rate on the monitoring subset",
            "detection rate (internal criterion)", "05_detection_rate.png")),
        ("06_false_positive_rate.png", lambda: selection_curve(
            histories, args.output, args.dpi, "false_positive_rate_on_negative_cases",
            "INTERNAL false-positive rate on lesion-negative monitoring cases",
            "false-positive rate (internal criterion)", "06_false_positive_rate.png")),
        ("07_epoch_seconds.png", lambda: curve(
            histories, args.output, args.dpi, "seconds",
            "Epoch wall time (engineering diagnostic)", "seconds", "07_epoch_seconds.png")),
        ("08_class28_dice_distribution.png", lambda: dice_distribution(cases, args.output, args.dpi)),
        ("09_per_class_dice.png", lambda: per_class_dice_plot(summaries, args.output, args.dpi)),
        ("10_dice_vs_lesion_volume.png", lambda: dice_vs_volume(cases, args.output, args.dpi)),
    ]

    for name, builder in figures:
        result = builder()
        (made if result else skipped).append(name)

    print(f"\n{len(made)} figure(s) -> {args.output}")
    for name in made:
        print(f"  + {name}")
    if skipped:
        print(f"\n{len(skipped)} skipped (inputs absent - nothing was invented):")
        for name in skipped:
            print(f"  - {name}")

    for arm in ARMS:
        history = histories.get(arm)
        if not history:
            continue
        epochs, records = selections(history)
        scores = [(e, r.get("mean_dice_on_positive_cases", math.nan))
                  for e, r in zip(epochs, records)]
        finite = [(e, v) for e, v in scores if not math.isnan(v)]
        best = max(finite, key=lambda pair: pair[1]) if finite else None
        print(f"\n{LABEL[arm]}: {len(history)} epochs, {len(epochs)} deterministic selections")
        print(f"  final train loss        {history[-1]['train_loss']:.4f}")
        if best:
            print(f"  best monitoring Dice    {best[1]:.4f} at epoch {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
