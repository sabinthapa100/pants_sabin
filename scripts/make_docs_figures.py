"""Regenerate the three presentation figures tracked under docs/figures/.

Reads the run histories and evaluation summaries, which are derived artifacts
kept outside Git, and writes only aggregate plots.

    python scripts/make_docs_figures.py --runs PanTS_run --evaluation evaluation

Random's monitoring history survives for only 5 of its 13 selection epochs
(a Colab reconnection lost the rest). Those epochs are plotted as isolated
markers with no connecting line; nothing is interpolated or reconstructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SUPREM = "#4c72b0"
RANDOM = "#c44e52"


def selection_points(history: list[dict]) -> tuple[list[int], list[float]]:
    """Epochs and deterministic whole-volume class-28 Dice, where recorded."""
    points = [(int(r["epoch"]), r["selection"]["mean_dice_on_positive_cases"])
              for r in history if r.get("selection")]
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def figure_model_selection(runs: Path, evaluation: Path, output: Path, dpi: int) -> Path:
    suprem = json.loads((runs / "segresnet_suprem" / "history.json").read_text())
    random_history = json.loads(
        (evaluation / "random_history" / "recovered_history.json").read_text())

    se, sd = selection_points(suprem)
    re_, rd = selection_points(random_history)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(se, sd, "-o", color=SUPREM, ms=5, lw=1.6, label=f"SuPreM ({len(se)} points)")
    axis.plot(re_, rd, "s", color=RANDOM, ms=6, ls="none",
              label=f"Random ({len(re_)} surviving points)")
    best = int(np.argmax(sd))
    axis.annotate(f"best.pt\nepoch {se[best]}  {sd[best]:.4f}", (se[best], sd[best]),
                  textcoords="offset points", xytext=(-64, -30), fontsize=8,
                  arrowprops=dict(arrowstyle="->", lw=0.8))
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean class-28 Dice, 177 positive cases")
    axis.set_title("Model selection: deterministic whole-volume monitoring metric", fontsize=11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=9)
    axis.text(0.99, 0.03,
              "Random history has gaps; no values were reconstructed.",
              transform=axis.transAxes, ha="right", fontsize=7.5, style="italic")
    figure.tight_layout()
    path = output / "01_model_selection.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def figure_failure_modes(summary: dict, output: Path, dpi: int) -> Path:
    groups = summary["results"]["primary_tumor_segmentation"]["outcome_groups"]
    counts = [groups["A_no_lesion_predicted"]["cases"],
              groups["B_predicted_but_zero_overlap"]["cases"],
              groups["C_positive_overlap"]["cases"]]
    total = sum(counts)
    labels = ["A\nno lesion\npredicted", "B\npredicted,\nzero overlap", "C\npositive\noverlap"]

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    bars = axis.bar(labels, counts, color=["#c44e52", "#dd8452", "#4c72b0"], width=0.6)
    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 1.2,
                  f"{count}\n{count / total:.1%}", ha="center", fontsize=9)
    axis.set_ylabel(f"lesion-positive cases (n = {total})")
    axis.set_ylim(0, max(counts) * 1.22)
    axis.set_title("PanTS-te held-out outcomes on lesion-positive cases", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output / "02_heldout_failure_modes.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def figure_dice_distribution(cases_csv: Path, output: Path, dpi: int) -> Path:
    import csv
    with open(cases_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    dice = np.array([
        0.0 if r["lesion_dice"] in ("", "nan") else float(r["lesion_dice"])
        for r in rows if r["lesion_present"] == "True"
    ])

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.hist(dice, bins=np.linspace(0, 1, 26), color=SUPREM, alpha=0.85, edgecolor="white")
    axis.axvline(dice.mean(), color="black", lw=1.4, label=f"mean {dice.mean():.4f}")
    axis.axvline(np.median(dice), color="black", lw=1.4, ls="--",
                 label=f"median {np.median(dice):.4f}")
    axis.set_xlabel("class-28 Dice")
    axis.set_ylabel(f"lesion-positive cases (n = {dice.size})")
    axis.set_title("PanTS-te held-out class-28 Dice distribution", fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output / "03_heldout_dice_distribution.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("PanTS_run"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    held_out = args.evaluation / "pants_te_final"
    summary = json.loads((held_out / "evaluation_summary.json").read_text())

    written = [
        figure_model_selection(args.runs, args.evaluation, args.output, args.dpi),
        figure_failure_modes(summary, args.output, args.dpi),
        figure_dice_distribution(held_out / "evaluation_cases.csv", args.output, args.dpi),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
