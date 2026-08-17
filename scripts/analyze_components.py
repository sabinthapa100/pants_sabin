"""Component-level measurement of one frozen checkpoint's class-28 predictions.

MEASUREMENT ONLY. Nothing here thresholds, filters, deletes or relabels a
component; the output is a description of what the model actually produced.

Inference settings are taken from ``scripts/evaluate_segresnet.py``'s prepared
path so the components measured here are the same objects that produced the
frozen case-level metrics: 96^3 windows, 0.5 overlap, sw_batch_size 1, gaussian
blending, windows on GPU, accumulation on the host.

    python scripts/analyze_components.py \
        --checkpoint PanTS_run/segresnet_suprem/best.pt \
        --prepared-root ../PanTS_prepared/segresnet \
        --cases-from evaluation/suprem/evaluation_cases.csv \
        --output evaluation/suprem_components/

Connectivity 26 is an INTERNAL development choice. The PanTS benchmark
publishes no component protocol, so nothing here is an official lesion count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.labels import PANCREATIC_LESION  # noqa: E402
from src.data.prepared import case_path, read_prepared_case  # noqa: E402
from src.evaluation.inference import logits_to_labels, predict_logits  # noqa: E402
from src.models.segresnet import build_segresnet  # noqa: E402
from src.training.checkpoint import load_training_checkpoint  # noqa: E402


logger = logging.getLogger(__name__)

PANCREAS_FAMILY = (17, 18, 19, 20, 21)      # pancreas head/body/tail + duct + PDAC-adjacent
SPACING_MM = (1.5, 1.5, 1.5)
VOXEL_MM3 = float(np.prod(SPACING_MM))       # 3.375
CONNECTIVITY = 26
LAST_TRAIN_CASE_ID = 9000

FIELDS = [
    "case_id",
    "lesion_present_in_ground_truth",
    "component_index",
    "components_in_case",
    "voxel_count",
    "physical_volume_mm3",
    "prob_max",
    "prob_mean",
    "prob_median",
    "prob_p90",
    "centroid_voxel_d",
    "centroid_voxel_h",
    "centroid_voxel_w",
    "centroid_prepared_mm_d",
    "centroid_prepared_mm_h",
    "centroid_prepared_mm_w",
    "distance_to_predicted_pancreas_mm",
    "predicted_pancreas_available",
    "distance_to_gt_pancreas_mm",
    "gt_pancreas_available",
    "overlap_with_gt_lesion",
    "overlap_voxels",
    "fraction_of_component_overlapping_gt",
    "gt_lesion_components_in_case",
]


# --------------------------------------------------------------------------- #
# probability
# --------------------------------------------------------------------------- #


def lesion_probability_stable(logits: torch.Tensor) -> torch.Tensor:
    """Class-28 softmax probability without materialising all 29 channels.

    ``softmax(x)[k] = exp(x_k - logsumexp(x))``. Computing it this way allocates
    one single-channel map instead of a second 29-channel float32 volume, which
    for a typical prepared case is the difference between ~40 MB and ~1.2 GB.
    Mathematically identical, and ``verify_probability_identity`` checks that
    claim numerically before the cohort pass rather than asserting it.
    """
    return torch.exp(logits[:, PANCREATIC_LESION] - torch.logsumexp(logits, dim=1))


def verify_probability_identity(seed: int = 0) -> dict[str, float]:
    """Check both float32 forms against a float64 reference.

    ``torch.softmax`` is not itself exact, so comparing the two float32 forms to
    each other cannot say which is right. Float64 is the arbiter.

    Scales are chosen to bracket what this model actually emits: measured logits
    on real prepared cases span roughly [-92, +17], and on those the two forms
    agree to ~3e-7. The tolerance below is set at 1e-5 -- two orders of magnitude
    above the observed disagreement, and far below any difference that could move
    a component-level statistic.
    """
    generator = torch.Generator().manual_seed(seed)
    softmax_error = logsumexp_error = 0.0
    for scale in (1.0, 10.0, 50.0):
        logits = torch.randn(1, 29, 8, 8, 8, generator=generator) * scale
        reference = torch.softmax(logits.double(), dim=1)[:, PANCREATIC_LESION]
        softmax_error = max(softmax_error, float(
            (torch.softmax(logits, dim=1)[:, PANCREATIC_LESION].double() - reference).abs().max()))
        logsumexp_error = max(logsumexp_error, float(
            (lesion_probability_stable(logits).double() - reference).abs().max()))
    return {
        "logsumexp_vs_float64": logsumexp_error,
        "softmax_vs_float64": softmax_error,
        "tolerance": 1e-5,
    }


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def distance_map(mask: np.ndarray) -> np.ndarray | None:
    """Euclidean distance in mm from every voxel to the nearest ``mask`` voxel.

    Returns ``None`` when the mask is empty: there is no nearest structure, and
    a fabricated distance would be indistinguishable from a measured one.
    """
    if not mask.any():
        return None
    return ndimage.distance_transform_edt(~mask, sampling=SPACING_MM)


def component_distance(
    distances: np.ndarray | None,
    components: np.ndarray,
    count: int,
) -> list[float]:
    """Minimum distance from each component to the reference structure."""
    if distances is None:
        return [float("nan")] * count
    index = list(range(1, count + 1))
    return [float(v) for v in np.atleast_1d(ndimage.minimum(distances, components, index))]


# --------------------------------------------------------------------------- #
# per-case measurement
# --------------------------------------------------------------------------- #


def measure_case(
    model: torch.nn.Module,
    case_id: str,
    prepared_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every predicted class-28 component of one case, with nothing removed."""
    image, target = read_prepared_case(case_path(prepared_root, case_id))
    volume = torch.from_numpy(image.astype(np.float32))[None, None]
    del image

    logits = predict_logits(
        model,
        volume,
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        sw_device=args.device,
        device=args.accumulate_device,
    )
    prediction = logits_to_labels(logits)[0, 0].numpy().astype(np.int16)
    probability = lesion_probability_stable(logits)[0].numpy().astype(np.float32)
    del logits, volume

    target = target.astype(np.int16)
    lesion_mask = prediction == PANCREATIC_LESION
    structure = ndimage.generate_binary_structure(3, 3)
    components, count = ndimage.label(lesion_mask, structure=structure)

    gt_lesion = target == PANCREATIC_LESION
    lesion_present = bool(gt_lesion.any())
    _, gt_component_count = ndimage.label(gt_lesion, structure=structure)

    predicted_pancreas = np.isin(prediction, PANCREAS_FAMILY)
    gt_pancreas = np.isin(target, PANCREAS_FAMILY)
    to_predicted = distance_map(predicted_pancreas)
    to_gt = distance_map(gt_pancreas)

    predicted_distances = component_distance(to_predicted, components, count)
    gt_distances = component_distance(to_gt, components, count)
    del to_predicted, to_gt, predicted_pancreas, gt_pancreas, prediction

    rows: list[dict[str, Any]] = []
    boxes = ndimage.find_objects(components)
    for order in range(count):
        # Work inside the component's bounding box: a 50-voxel speckle in a
        # 10-million-voxel volume should not cost a full-volume pass.
        box = boxes[order]
        local = components[box] == (order + 1)
        local_probability = probability[box][local]
        offset = np.array([s.start for s in box])
        centroid = offset + np.array(ndimage.center_of_mass(local))
        voxels = int(local.sum())

        overlap = int((gt_lesion[box] & local).sum())
        rows.append({
            "case_id": case_id,
            "lesion_present_in_ground_truth": lesion_present,
            "component_index": order + 1,
            "components_in_case": count,
            "voxel_count": voxels,
            "physical_volume_mm3": voxels * VOXEL_MM3,
            "prob_max": float(local_probability.max()),
            "prob_mean": float(local_probability.mean()),
            "prob_median": float(np.median(local_probability)),
            "prob_p90": float(np.percentile(local_probability, 90)),
            "centroid_voxel_d": float(centroid[0]),
            "centroid_voxel_h": float(centroid[1]),
            "centroid_voxel_w": float(centroid[2]),
            "centroid_prepared_mm_d": float(centroid[0] * SPACING_MM[0]),
            "centroid_prepared_mm_h": float(centroid[1] * SPACING_MM[1]),
            "centroid_prepared_mm_w": float(centroid[2] * SPACING_MM[2]),
            "distance_to_predicted_pancreas_mm": predicted_distances[order],
            "predicted_pancreas_available": not np.isnan(predicted_distances[order]),
            "distance_to_gt_pancreas_mm": gt_distances[order],
            "gt_pancreas_available": not np.isnan(gt_distances[order]),
            "overlap_with_gt_lesion": overlap > 0,
            "overlap_voxels": overlap,
            "fraction_of_component_overlapping_gt": overlap / voxels,
            "gt_lesion_components_in_case": gt_component_count,
        })
        del local, local_probability

    case_summary = {
        "case_id": case_id,
        "lesion_present": lesion_present,
        "components": count,
        "predicted_lesion_voxels": int(lesion_mask.sum()),
        "gt_lesion_voxels": int(gt_lesion.sum()),
        "gt_lesion_components": gt_component_count,
    }
    del components, lesion_mask, gt_lesion, probability, target
    return rows, case_summary


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #


def check_invariants(rows: list[dict[str, Any]], case: dict[str, Any]) -> None:
    """Fail loudly on anything that would make the measurement untrustworthy."""
    case_id = case["case_id"]
    if int(case_id.split("_")[1]) > LAST_TRAIN_CASE_ID:
        raise SystemExit(f"{case_id} is PanTS-te; this analysis must never read it")

    total = sum(r["voxel_count"] for r in rows)
    if total != case["predicted_lesion_voxels"]:
        raise SystemExit(
            f"{case_id}: component voxels {total} != predicted class-28 voxels "
            f"{case['predicted_lesion_voxels']}"
        )
    for row in rows:
        if abs(row["physical_volume_mm3"] - row["voxel_count"] * VOXEL_MM3) > 1e-9:
            raise SystemExit(f"{case_id}: volume does not equal voxel_count * {VOXEL_MM3}")
        for key in ("prob_max", "prob_mean", "prob_median", "prob_p90"):
            value = row[key]
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(f"{case_id}: {key} = {value} outside [0, 1]")
        if row["prob_mean"] > row["prob_max"] + 1e-6 or row["prob_median"] > row["prob_max"] + 1e-6:
            raise SystemExit(f"{case_id}: mean/median probability exceeds max")
        for key in ("distance_to_predicted_pancreas_mm", "distance_to_gt_pancreas_mm"):
            value = row[key]
            if not np.isnan(value) and value < 0.0:
                raise SystemExit(f"{case_id}: {key} = {value} is negative")
        if row["overlap_voxels"] > row["voxel_count"]:
            raise SystemExit(f"{case_id}: overlap exceeds component size")
        if row["overlap_with_gt_lesion"] != (row["overlap_voxels"] > 0):
            raise SystemExit(f"{case_id}: overlap flag disagrees with overlap voxels")
        if not case["lesion_present"] and row["overlap_voxels"] != 0:
            raise SystemExit(f"{case_id}: lesion-negative case reports lesion overlap")


# --------------------------------------------------------------------------- #
# cohort
# --------------------------------------------------------------------------- #


def load_cohort(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Positive cases plus the negatives the frozen evaluation called false positive.

    The CSV is read for case identity only; no metric in it is reused.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    positives = [r["case_id"] for r in rows if r["lesion_present"] == "True"]
    false_positives = [
        r["case_id"] for r in rows
        if r["lesion_present"] != "True" and int(float(r["predicted_lesion_voxels"])) > 0
    ]
    cohort = sorted(set(positives) | set(false_positives))
    held_out = [c for c in cohort if int(c.split("_")[1]) > LAST_TRAIN_CASE_ID]
    if held_out:
        raise SystemExit(f"cohort contains {len(held_out)} PanTS-te case(s): {held_out[:5]}")
    return cohort, {
        "source_csv": str(path),
        "rows_in_csv": len(rows),
        "lesion_positive": len(positives),
        "false_positive_negative": len(false_positives),
        "total": len(cohort),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def load_model(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_segresnet(initialization="random")
    checkpoint = load_training_checkpoint(checkpoint_path, model=model)
    missing = set(model.state_dict()) - set(checkpoint["model"])
    if missing:
        raise SystemExit(f"checkpoint does not define {len(missing)} model tensors")
    model.to(device).eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--cases-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N cases; validation only")
    parser.add_argument("--cases", nargs="*", default=None, help="explicit case IDs; validation only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--accumulate-device", default="cpu")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    identity = verify_probability_identity()
    if identity["logsumexp_vs_float64"] > identity["tolerance"]:
        raise SystemExit(f"logsumexp probability disagrees with float64 softmax: {identity}")
    logger.info("probability identity verified: logsumexp vs float64 %.3e (softmax vs float64 %.3e)",
                identity["logsumexp_vs_float64"], identity["softmax_vs_float64"])

    cohort, cohort_info = load_cohort(args.cases_from)
    if args.cases:
        cohort = [c for c in cohort if c in set(args.cases)]
    elif args.limit:
        cohort = cohort[: args.limit]
    logger.info("cohort %d case(s) of %d", len(cohort), cohort_info["total"])

    model, checkpoint = load_model(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    started = time.time()
    for position, case_id in enumerate(cohort, start=1):
        case_rows, case_summary = measure_case(model, case_id, args.prepared_root, args)
        check_invariants(case_rows, case_summary)
        rows.extend(case_rows)
        cases.append(case_summary)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if position % 25 == 0 or position == len(cohort):
            rate = (time.time() - started) / position
            logger.info("%d/%d cases, %d components, %.1f s/case, %.0f min remaining",
                        position, len(cohort), len(rows), rate,
                        rate * (len(cohort) - position) / 60)

    csv_path = args.output / "components.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "analysis": "predicted class-28 connected components; MEASUREMENT ONLY",
        "internal_metric_notice": (
            "Connectivity, spacing and component definitions here are internal "
            "development choices. PanTS publishes no component protocol."
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "epoch": checkpoint.get("epoch"),
            "best_metric": checkpoint.get("best_metric"),
            "training_git_commit": checkpoint.get("git_commit"),
        },
        "analysis_git_commit": git_commit(),
        "cohort": cohort_info | {"processed": len(cohort)},
        "components": {
            "connectivity": CONNECTIVITY,
            "structure": "scipy.ndimage.generate_binary_structure(3, 3)",
            "spacing_mm": list(SPACING_MM),
            "voxel_mm3": VOXEL_MM3,
            "frame": "prepared RAS 1.5 mm grid; NOT patient world coordinates",
            "pancreas_family_classes": list(PANCREAS_FAMILY),
            "total": len(rows),
            "cases_with_at_least_one": sum(1 for c in cases if c["components"]),
        },
        "inference": {
            "roi_size": [96, 96, 96],
            "overlap": args.overlap,
            "sw_batch_size": args.sw_batch_size,
            "blend_mode": "gaussian",
            "window_device": args.device,
            "accumulate_device": args.accumulate_device,
            "test_time_augmentation": False,
            "probability": "exp(logit_28 - logsumexp(logits, dim=1))",
            "probability_identity": identity,
            "probability_identity_on_real_logits": {
                "cases": ["PanTS_00007846", "PanTS_00004321"],
                "observed_logit_range": [-91.59, 17.32],
                "logsumexp_vs_float64_max_abs": 3.249e-07,
                "note": "measured once during validation, not recomputed per run",
            },
        },
        "runtime": {
            "seconds": round(time.time() - started, 1),
            "seconds_per_case": round((time.time() - started) / max(1, len(cohort)), 2),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cases": cases,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %d components from %d cases to %s", len(rows), len(cohort), csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
