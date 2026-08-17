"""Deterministic whole-volume evaluation of one trained PanTS SegResNet checkpoint.

Evaluates every case of one fold with sliding-window inference. There is no
random cropping, no augmentation and no tumor-aware sampling anywhere in this
path, so re-running it on the same checkpoint returns the same numbers.

    python scripts/evaluate_segresnet.py \
        --checkpoint best.pt --prepared-root /path/PanTS_prepared \
        --manifest /path/PanTS_prepared/manifest.json \
        --split pants_cv_v1.json --fold 0 --output evaluation/suprem/

Two modes share one metric implementation:

``--mode prepared`` (default)
    reads the prepared npz cache, so metrics are computed in the canonical
    training frame (RAS, 1.5 mm). This is the same frame the monitoring metric
    that selected ``best.pt`` uses, which is what makes the two comparable.

``--mode raw``
    reads the original NIfTI CT and label and evaluates in the *source* CT
    geometry via the unseen-scan inference path. Built for the later one-time
    held-out test; it is not used for development evaluation.

Every metric here is an INTERNAL development metric. The PanTS repository
publishes no evaluator, so nothing in this file is the official P-Sen, T-Sen,
Spe or AUC. See ``METRIC_DEFINITIONS``.
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
from typing import Any, Iterator

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.labels import CLASS_MAP, PANCREATIC_LESION  # noqa: E402
from src.data.paths import get_case_paths  # noqa: E402
from src.data.prepared import case_path, read_prepared_case  # noqa: E402
from src.evaluation.inference import (  # noqa: E402
    logits_to_labels,
    predict_case_in_source_geometry,
    predict_logits,
)
from src.evaluation.postprocessing import (  # noqa: E402
    LESION_COMPONENT_CONNECTIVITY,
    LESION_PEAK_PROBABILITY,
    filter_lesion_components,
)
from src.evaluation.segmentation import (  # noqa: E402
    all_class_dice,
    lesion_case_metrics,
    summarize_lesion_metrics,
)
from src.models.segresnet import build_segresnet  # noqa: E402
from src.training.checkpoint import load_training_checkpoint  # noqa: E402


logger = logging.getLogger(__name__)

FOREGROUND_CLASSES = sorted(CLASS_MAP)              # 1..28
ANATOMY_CLASSES = [c for c in FOREGROUND_CLASSES if c != PANCREATIC_LESION]
PANCREAS_FAMILY = [17, 18, 19, 20, 21, 28]
LAST_TRAIN_CASE_ID = 9000                           # PanTS-te starts at 9001

METRIC_DEFINITIONS = {
    "lesion_dice": (
        "2|P n G| / (|P| + |G|) for class 28 in the evaluation frame; NaN when "
        "the class is absent from both prediction and ground truth"
    ),
    "lesion_dice_on_positive_cases": (
        "PRIMARY. Mean lesion_dice over cases whose GROUND TRUTH contains class 28. "
        "This is the model-selection quantity; lesion-negative cases never enter it."
    ),
    "detected": (
        "INTERNAL: ground truth has >0 class-28 voxels AND prediction has >0. "
        "Presence only - the predicted voxels need not overlap the true lesion, "
        "so detected=True with lesion_dice=0.0 means a wrong-location prediction."
    ),
    "false_positive": "INTERNAL: ground truth has 0 class-28 voxels AND prediction has >0",
    "internal_specificity": "1 - false_positive_rate_on_negative_cases",
    "per_class_support_dice": (
        "Mean Dice per label over cases where the label occurs in the TARGET OR THE "
        "PREDICTION. A case with the label absent from both is NaN and excluded; a "
        "case where only the prediction has it contributes an exact 0.0. This is a "
        "DIFFERENT quantity from lesion_dice_on_positive_cases, which conditions on "
        "ground truth alone - compare class 28 across the two at your peril."
    ),
    "macro_foreground_dice": (
        "unweighted mean of the per-class support means over classes 1..28"
    ),
    "official_pants_metrics": (
        "NOT COMPUTED. P-Sen, T-Sen, Spe and AUC require JHU's unpublished "
        "component-matching, overlap-threshold and patient-scoring protocol. "
        "The continuous class-28 probability needed for their AUC is produced "
        "by scripts/infer_segresnet.py, not by this development evaluator."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="results directory")
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help=(
            "prepared-cache manifest. Required for --mode prepared (it supplies the "
            "voxel size); recorded for provenance only in --mode raw, where geometry "
            "comes from each CT's own header, so a held-out run need not reference "
            "the development cache at all."
        ),
    )
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "pants_cv_v1.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", choices=["prepared", "raw"], default="prepared")
    parser.add_argument("--prepared-root", type=Path, default=None, help="required for --mode prepared")
    parser.add_argument("--data-root", type=Path, default=None, help="raw PanTS root for --mode raw")
    parser.add_argument(
        "--data-split", choices=["train", "test"], default="train",
        help="raw-mode directory pair: train -> ImageTr/LabelTr, test -> ImageTe/LabelTe",
    )
    parser.add_argument(
        "--case-list", type=Path, default=None,
        help="explicit case IDs (JSON array/object or one per line); overrides --split/--fold",
    )
    parser.add_argument(
        "--lesion-peak-probability", type=float, default=None, metavar="P",
        help=(
            f"keep a class-28 component only if its peak softmax >= P "
            f"(frozen development rule: {LESION_PEAK_PROBABILITY}). Omitted by "
            "default so this script reproduces the unfiltered baseline exactly."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    parser.add_argument("--resume", action="store_true", help="skip cases already in evaluation_cases.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--accumulate-device",
        default="cpu",
        help="where full-volume logits are stitched; 'cuda' is faster but shifts logits by ~1e-3",
    )
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="permit case IDs above 9000 (PanTS-te). Off by default, on purpose.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def content_sha256(payload: Any) -> str:
    """The trainer's canonical content hash, so provenance can be cross-checked.

    Must stay byte-compatible with ``SegResNetTrainer._content_hash``; a run's
    checkpoint records the same function's output for the manifest and split.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def case_number(case_id: str) -> int:
    """Numeric part of ``PanTS_00001234``; the te guard depends on it."""
    return int(case_id.rsplit("_", 1)[1])


def guard_split(case_ids: list[str], allow_test: bool) -> None:
    """Refuse PanTS-te unless explicitly unlocked.

    The held-out test set is a one-shot resource: reading it during development
    silently converts it into a validation set. Making that require a flag is
    cheaper than remembering not to do it.
    """
    held_out = [c for c in case_ids if case_number(c) > LAST_TRAIN_CASE_ID]
    if held_out and not allow_test:
        raise SystemExit(
            f"{len(held_out)} case(s) above PanTS_{LAST_TRAIN_CASE_ID:08d} are PanTS-te "
            f"(first: {held_out[0]}). Pass --allow-test-split only for the final "
            "one-time held-out evaluation."
        )


def load_model(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the 29-class architecture and overwrite every tensor from the checkpoint.

    ``initialization="random"`` means only "do not read a SuPreM file"; the
    trained weights replace all of it. Evaluation never needs the pretraining
    checkpoint.
    """
    model = build_segresnet(initialization="random")
    checkpoint = load_training_checkpoint(checkpoint_path, model=model)
    missing = set(model.state_dict()) - set(checkpoint["model"])
    if missing:
        raise SystemExit(f"checkpoint does not define {len(missing)} model tensors: {sorted(missing)[:5]}")
    model.to(device).eval()
    return model, checkpoint


# --------------------------------------------------------------------------- #
# per-case evaluation
# --------------------------------------------------------------------------- #


def evaluate_prepared(model, case_id, args, voxel_mm3) -> dict[str, Any]:
    """Whole-volume inference in the canonical 1.5 mm training frame."""
    image, label = read_prepared_case(case_path(args.prepared_root, case_id))
    volume = torch.from_numpy(image.astype(np.float32))[None, None]
    logits = predict_logits(
        model, volume, overlap=args.overlap, sw_batch_size=args.sw_batch_size,
        sw_device=args.device, device=args.accumulate_device,
    )
    labels = logits_to_labels(logits)
    extra: dict[str, Any] = {}
    if args.lesion_peak_probability is not None:
        report = filter_lesion_components(
            logits, labels, min_peak_probability=args.lesion_peak_probability
        )
        labels = report.pop("labels")
        extra = {
            "components_found": report["components_found"],
            "components_retained": report["components_retained"],
            "components_rejected": report["components_rejected"],
            "relabelled_voxels": report["relabelled_voxels"],
            "fallback_class_counts": report["fallback_class_counts"],
        }
    prediction = labels[0, 0].to("cpu").numpy().astype(np.int16)
    del logits, volume, labels
    return _score(prediction, label.astype(np.int16), voxel_mm3) | extra


def evaluate_raw(model, case_id, args) -> dict[str, Any]:
    """Whole-volume inference in the ORIGINAL CT geometry.

    Uses the same image-only preprocessing and inverse transform as unseen-scan
    inference, so what is measured here is exactly what an external user gets.
    Reserved for the one-time held-out test; not used for development.
    """
    import nibabel as nib

    paths = get_case_paths(case_id, args.data_split, args.data_root)
    result = predict_case_in_source_geometry(
        model, str(paths["ct"]), overlap=args.overlap,
        sw_batch_size=args.sw_batch_size, sw_device=args.device,
        accumulate_device=args.accumulate_device,
        min_lesion_peak_probability=args.lesion_peak_probability,
    )
    prediction = np.asarray(result["labels"].detach().cpu())[0].astype(np.int16)
    reference = nib.load(str(paths["combined"]))
    target = np.asarray(reference.dataobj).astype(np.int16)
    if prediction.shape != target.shape:
        raise RuntimeError(f"{case_id}: prediction {prediction.shape} != label {target.shape}")
    voxel_mm3 = float(np.abs(np.linalg.det(reference.affine)))
    return _score(prediction, target, voxel_mm3)


def _score(prediction: np.ndarray, target: np.ndarray, voxel_mm3: float) -> dict[str, Any]:
    """The single scoring path both modes share."""
    lesion = lesion_case_metrics(prediction, target)
    dice = all_class_dice(prediction, target, FOREGROUND_CLASSES)
    row: dict[str, Any] = {
        "lesion_present": lesion["lesion_present"],
        "target_lesion_voxels": lesion["target_voxels"],
        "predicted_lesion_voxels": lesion["predicted_voxels"],
        "target_lesion_mm3": lesion["target_voxels"] * voxel_mm3,
        "predicted_lesion_mm3": lesion["predicted_voxels"] * voxel_mm3,
        "lesion_dice": lesion["dice"],
        "detected": lesion["detected"],
        "false_positive": lesion["false_positive"],
    }
    row.update({f"dice_{label:02d}": dice[label] for label in FOREGROUND_CLASSES})
    return row


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _stats(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return {"n": 0}
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def outcome_groups(positives: list[dict[str, Any]]) -> dict[str, Any]:
    """Split lesion-positive cases by *why* they scored what they scored.

    ``detected`` only asks whether any class-28 voxel was predicted; it does not
    ask whether those voxels touch the lesion. Group B is exactly the population
    that separates a missing prediction from a misplaced one, and without it a
    low mean Dice cannot be attributed to detection rather than localization.
    """
    no_prediction, zero_overlap, overlapping = [], [], []
    for row in positives:
        dice = row["lesion_dice"]
        if not row["predicted_lesion_voxels"]:
            no_prediction.append(row)
        elif np.isnan(dice) or dice == 0.0:
            zero_overlap.append(row)
        else:
            overlapping.append(row)

    total = len(positives) or 1
    return {
        "definition": (
            "lesion-positive cases partitioned by prediction outcome; A+B+C = all "
            "lesion-positive cases"
        ),
        "A_no_lesion_predicted": {
            "cases": len(no_prediction),
            "fraction": len(no_prediction) / total,
        },
        "B_predicted_but_zero_overlap": {
            "cases": len(zero_overlap),
            "fraction": len(zero_overlap) / total,
            "note": "prediction exists somewhere but shares no voxel with the lesion",
        },
        "C_positive_overlap": {
            "cases": len(overlapping),
            "fraction": len(overlapping) / total,
            "dice_among_overlapping": _stats([r["lesion_dice"] for r in overlapping]),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything reported, computed from the per-case rows only."""
    positives = [r for r in rows if r["lesion_present"]]
    # The CSV column is `lesion_dice`, but the shared aggregator speaks the
    # `lesion_case_metrics` vocabulary. Adapt rather than duplicate its rules:
    # the positive/negative separation is exactly what must not be reimplemented.
    lesion_summary = summarize_lesion_metrics(
        [{**row, "dice": row["lesion_dice"]} for row in rows]
    )

    positive_dice = [r["lesion_dice"] for r in positives]
    tumor = _stats(positive_dice)
    tumor["zero_dice_positive_cases"] = sum(
        1 for v in positive_dice if v is not None and not np.isnan(v) and v == 0.0
    )

    per_class: dict[str, Any] = {}
    for label in FOREGROUND_CLASSES:
        key = f"dice_{label:02d}"
        present = [r[key] for r in rows if r[key] is not None and not np.isnan(r[key])]
        per_class[str(label)] = {
            "name": CLASS_MAP[label],
            "cases_scored": len(present),
            "mean_dice": float(np.mean(present)) if present else float("nan"),
            "median_dice": float(np.median(present)) if present else float("nan"),
        }

    def macro(labels: list[int]) -> float:
        means = [per_class[str(c)]["mean_dice"] for c in labels]
        means = [m for m in means if not np.isnan(m)]
        return float(np.mean(means)) if means else float("nan")

    absolute_error = [abs(r["predicted_lesion_mm3"] - r["target_lesion_mm3"]) for r in positives]
    relative_error = [
        abs(r["predicted_lesion_mm3"] - r["target_lesion_mm3"]) / r["target_lesion_mm3"]
        for r in positives if r["target_lesion_mm3"] > 0
    ]

    seconds = [r["inference_seconds"] for r in rows if r.get("inference_seconds") is not None]

    return {
        "primary_tumor_segmentation": {
            "lesion_positive_cases": lesion_summary["lesion_positive_cases"],
            "class28_dice_on_positive_cases": tumor,
            "outcome_groups": outcome_groups(positives),
        },
        "internal_case_detection": {
            "INTERNAL": (
                "our own criterion (any predicted class-28 voxel), NOT the PanTS "
                "benchmark's P-Sen/Spe protocol, which is unpublished"
            ),
            "lesion_negative_cases": lesion_summary["lesion_negative_cases"],
            "positive_case_detection_rate": lesion_summary["case_detection_rate_on_positive_cases"],
            "negative_case_false_positive_rate": lesion_summary["false_positive_rate_on_negative_cases"],
            "internal_specificity": 1.0 - lesion_summary["false_positive_rate_on_negative_cases"],
        },
        "anatomy_aware_segmentation": {
            "per_class": per_class,
            "macro_foreground_dice_1_28": macro(FOREGROUND_CLASSES),
            "macro_anatomy_dice_1_27_excluding_lesion": macro(ANATOMY_CLASSES),
            "pancreas_family": {str(c): per_class[str(c)]["mean_dice"] for c in PANCREAS_FAMILY},
        },
        "lesion_volume": {
            "absolute_error_mm3": _stats(absolute_error),
            "relative_error_on_positive_cases": _stats(relative_error),
        },
        "efficiency": {
            "seconds_per_case": _stats(seconds),
            "cases_scored": len(rows),
        },
    }


def stratify_by_lesion_volume(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """EXPLORATORY: class-28 performance against ground-truth lesion size.

    Bins are tertiles of this evaluation set's own lesion-positive volume
    distribution, so they are defined by the data rather than chosen after
    seeing the scores. The cut points are recorded. At 1.5 mm the smallest
    lesions lose the most voxels, so this is where resampling cost shows up.
    """
    positives = [r for r in rows if r["lesion_present"] and r["target_lesion_mm3"] > 0]
    if len(positives) < 3:
        return {"note": "too few lesion-positive cases to stratify"}

    volumes = np.asarray([r["target_lesion_mm3"] for r in positives])
    cuts = [float(np.percentile(volumes, 100 / 3)), float(np.percentile(volumes, 200 / 3))]

    bins: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for row in positives:
        volume = row["target_lesion_mm3"]
        name = "small" if volume <= cuts[0] else ("medium" if volume <= cuts[1] else "large")
        bins[name].append(row)

    return {
        "EXPLORATORY": "post-hoc stratification; the model was not tuned on these bins",
        "bin_edges_mm3": cuts,
        "bins": {
            name: {
                "cases": len(members),
                "median_lesion_mm3": float(np.median([m["target_lesion_mm3"] for m in members]))
                if members else float("nan"),
                "mean_dice": float(np.nanmean([m["lesion_dice"] for m in members]))
                if members else float("nan"),
                "detection_rate": (sum(m["detected"] for m in members) / len(members))
                if members else float("nan"),
            }
            for name, members in bins.items()
        },
    }


# --------------------------------------------------------------------------- #


def summarize_filter_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the component filter actually did, aggregated over the cohort.

    ``fallback_classes`` answers the question the offline study could not: when a
    lesion component is rejected, what does the model call those voxels instead?
    A sensible distribution is dominated by pancreas-family and background; heavy
    mass on distant organs would mean the fallback is not behaving anatomically.
    """
    fallback: dict[int, int] = {}
    for row in rows:
        for label, count in (row.get("fallback_class_counts") or {}).items():
            fallback[int(label)] = fallback.get(int(label), 0) + int(count)
    relabelled = sum(int(row.get("relabelled_voxels", 0) or 0) for row in rows)
    return {
        "cases_with_any_component": sum(1 for r in rows if (r.get("components_found") or 0) > 0),
        "components_found": sum(int(r.get("components_found", 0) or 0) for r in rows),
        "components_retained": sum(int(r.get("components_retained", 0) or 0) for r in rows),
        "components_rejected": sum(int(r.get("components_rejected", 0) or 0) for r in rows),
        "relabelled_voxels": relabelled,
        "fallback_classes": {
            str(label): {
                "voxels": count,
                "fraction": count / relabelled if relabelled else float("nan"),
                "name": CLASS_MAP.get(label, "background" if label == 0 else "unknown"),
            }
            for label, count in sorted(fallback.items(), key=lambda kv: -kv[1])
        },
    }


def read_case_list(path: Path) -> list[str]:
    """Case IDs from a JSON array, a JSON object with ``cases``, or one per line.

    The cross-validation split only names PanTS-tr cases, so a held-out
    evaluation has no way to enumerate itself from it. This is the smallest
    mechanism that lets the caller say which cases to score without inventing a
    second split format.
    """
    text = path.read_text().strip()
    if text.startswith(("[", "{")):
        payload = json.loads(text)
        cases = payload["cases"] if isinstance(payload, dict) else payload
    else:
        cases = [line.strip() for line in text.splitlines()]
    cases = [case for case in cases if case]
    if not cases:
        raise SystemExit(f"{path} contains no case identifiers")
    return sorted(dict.fromkeys(cases))


def resolve_cases(args) -> list[str]:
    if args.case_list:
        cases = read_case_list(args.case_list)
    else:
        split = json.loads(args.split.read_text())
        if not 0 <= args.fold < len(split["folds"]):
            raise SystemExit(f"fold {args.fold} outside 0..{len(split['folds']) - 1}")
        cases = sorted(split["folds"][args.fold]["val"])
    guard_split(cases, args.allow_test_split)
    return cases[: args.limit] if args.limit else cases


def existing_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        return [], set()
    with open(path, newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for entry in raw:
        row: dict[str, Any] = {}
        for key, value in entry.items():
            if key == "case_id":
                row[key] = value
            elif value == "":
                row[key] = float("nan")
            elif key in ("lesion_present", "detected", "false_positive"):
                row[key] = value == "True"
            elif key.endswith("voxels"):
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows, {r["case_id"] for r in rows}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "prepared" and not args.prepared_root:
        raise SystemExit("--mode prepared needs --prepared-root")
    if args.mode == "prepared" and not args.manifest:
        raise SystemExit("--mode prepared needs --manifest")

    cases = resolve_cases(args)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    args.output.mkdir(parents=True, exist_ok=True)
    cases_csv = args.output / "evaluation_cases.csv"

    rows, done = existing_rows(cases_csv) if args.resume else ([], set())
    pending = [c for c in cases if c not in done]

    voxel_mm3 = 1.0
    if args.mode == "prepared":
        spacing = json.loads((Path(args.prepared_root) / "preprocessing.json").read_text())["spacing_mm"]
        voxel_mm3 = float(np.prod(spacing))

    model, checkpoint = load_model(args.checkpoint, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        "evaluating %s on %d/%d fold-%d validation cases (mode=%s, device=%s)",
        args.checkpoint.name, len(pending), len(cases), args.fold, args.mode, args.device,
    )

    fieldnames = (
        ["case_id", "lesion_present", "target_lesion_voxels", "predicted_lesion_voxels",
         "target_lesion_mm3", "predicted_lesion_mm3", "lesion_dice", "detected",
         "false_positive", "inference_seconds"]
        + [f"dice_{label:02d}" for label in FOREGROUND_CLASSES]
        # Only when filtering is active, so an unfiltered run's CSV keeps exactly
        # the columns it always had and stays comparable to earlier baselines.
        + (["components_found", "components_retained", "components_rejected",
            "relabelled_voxels"] if args.lesion_peak_probability is not None else [])
    )

    failures: list[dict[str, str]] = []
    started = time.time()
    handle = open(cases_csv, "a" if done else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    if not done:
        writer.writeheader()

    try:
        for index, case_id in enumerate(pending, 1):
            case_started = time.time()
            try:
                row = (
                    evaluate_prepared(model, case_id, args, voxel_mm3)
                    if args.mode == "prepared"
                    else evaluate_raw(model, case_id, args)
                )
            except (OSError, RuntimeError, ValueError) as error:
                logger.error("%s failed: %s", case_id, error)
                failures.append({"case_id": case_id, "error": str(error)})
                continue
            row["case_id"] = case_id
            row["inference_seconds"] = time.time() - case_started
            rows.append(row)
            writer.writerow(row)
            handle.flush()                       # a crash keeps everything scored so far
            if index % 50 == 0 or index == len(pending):
                logger.info("  %d/%d  %.2f s/case", index, len(pending),
                            (time.time() - started) / index)
    finally:
        handle.close()

    summary = {
        "checkpoint": {
            "filename": args.checkpoint.name,
            "sha256": file_sha256(args.checkpoint),
            "initialization": checkpoint.get("config", {}).get("config", {}).get("initialization"),
            "epoch": checkpoint.get("epoch"),
            "selection_metric_name": checkpoint.get("config", {}).get("selection_metric_name"),
            "selection_metric_value": checkpoint.get("config", {}).get("selection_metric_value"),
            "training_git_commit": checkpoint.get("git_commit"),
        },
        "evaluation": {
            # Derived from the case IDs actually scored, never from a flag the
            # caller sets, so a held-out run cannot be mislabelled development.
            "cohort": (
                "PanTS-te in-distribution held-out evaluation"
                if any(case_number(c) > LAST_TRAIN_CASE_ID for c in cases)
                else "PanTS-tr development evaluation"
            ),
            "git_commit": git_commit(),
            "mode": args.mode,
            "evaluation_frame": "prepared_RAS_1.5mm" if args.mode == "prepared" else "source_ct_geometry",
            "fold": args.fold,
            "data_split": args.data_split,
            "case_list": str(args.case_list) if args.case_list else None,
            "postprocessing": (
                {
                    "rule": "keep class-28 component iff peak softmax >= threshold",
                    "min_peak_probability": args.lesion_peak_probability,
                    "connectivity": LESION_COMPONENT_CONNECTIVITY,
                    "applied_in": "canonical RAS 1.5 mm frame, before any inversion",
                    "rejected_voxel_fallback": "argmax over logit channels 0..27",
                    "notice": (
                        "development rule selected on fold 0; a softmax threshold, "
                        "NOT a calibrated probability of malignancy"
                    ),
                }
                if args.lesion_peak_probability is not None
                else None
            ),
            "postprocessing_audit": (
                summarize_filter_effect(rows)
                if args.lesion_peak_probability is not None
                else None
            ),
            # Two hashes of the same split, because they answer different
            # questions. The file hash identifies these exact bytes on disk; the
            # content hash uses the trainer's canonical serialization
            # (json.dumps(sort_keys=True)) and is what checkpoint provenance
            # records, so it is the one that can be compared against a run.
            # Null when an explicit --case-list defined the cohort instead: a held-out
            # run is not selected by the development split, and recording a hash of a
            # file it never consulted would misstate its provenance.
            "split_file_sha256": (
                hashlib.sha256(args.split.read_bytes()).hexdigest()
                if args.split and not args.case_list else None
            ),
            "split_content_sha256": (
                content_sha256(json.loads(args.split.read_text()))
                if args.split and not args.case_list else None
            ),
            "case_list_sha256": (
                hashlib.sha256(args.case_list.read_bytes()).hexdigest()
                if args.case_list else None
            ),
            "manifest_content_sha256": content_sha256(manifest) if manifest else None,
            "manifest_version": manifest.get("meta", {}).get("version") if manifest else None,
            "cases_requested": len(cases),
            "cases_scored": len(rows),
            "cases_failed": len(failures),
            "failures": failures,
            "sliding_window": {
                "roi_size": [96, 96, 96],
                "overlap": args.overlap,
                "sw_batch_size": args.sw_batch_size,
                "accumulate_device": args.accumulate_device,
                "blend_mode": "gaussian",
            },
            "device": args.device,
            "peak_gpu_gb": (
                torch.cuda.max_memory_allocated() / 1024 ** 3
                if args.device.startswith("cuda") else None
            ),
            "total_seconds": time.time() - started,
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "results": aggregate(rows) if rows else {},
        "exploratory_lesion_size": stratify_by_lesion_volume(rows) if rows else {},
    }
    try:
        import monai

        summary["evaluation"]["software"]["monai"] = monai.__version__
    except ImportError:
        pass

    (args.output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if rows:
        tumor = summary["results"]["primary_tumor_segmentation"]["class28_dice_on_positive_cases"]
        detection = summary["results"]["internal_case_detection"]
        print(f"\n{len(rows)} cases scored, {len(failures)} failed, "
              f"{summary['evaluation']['total_seconds'] / 60:.1f} min")
        print(f"  class-28 Dice on {tumor['n']} lesion-positive cases: "
              f"mean {tumor.get('mean', float('nan')):.4f}  median {tumor.get('median', float('nan')):.4f}")
        print(f"  INTERNAL detection rate {detection['positive_case_detection_rate']:.4f}  "
              f"INTERNAL specificity {detection['internal_specificity']:.4f}")
        print(f"  macro foreground Dice (1..28) "
              f"{summary['results']['anatomy_aware_segmentation']['macro_foreground_dice_1_28']:.4f}")
    print(f"\n-> {cases_csv}\n-> {args.output / 'evaluation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
