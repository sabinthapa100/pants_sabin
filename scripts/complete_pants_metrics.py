"""Record the continuous and component-level information PanTS-style metrics need.

The frozen evaluator stores hard per-case counts only, which is enough for
P-Sen, specificity and DSC but not for T-Sen (needs tumor components and a
matching) or AUC (needs one continuous score per patient). This pass re-runs
the same frozen inference and the same pmax >= 0.6 rule and records what was
missing. It changes no prediction.

    # one-time prerequisite for T-Sen
    python scripts/complete_pants_metrics.py --verify-labels --data-split test \
        --case-list evaluation/pants_te_final/pants_te_cases.txt

    # fold-0 protocol validation, canonical 1.5 mm frame
    python scripts/complete_pants_metrics.py --mode prepared --fold 0 ...

    # held-out completion, source CT geometry
    python scripts/complete_pants_metrics.py --mode raw --data-split test ...

Large arrays are released per case; only scalars are kept.
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import PANCREATIC_LESION  # noqa: E402
from src.data.paths import get_case_paths  # noqa: E402
from src.evaluation.benchmark_metrics import (  # noqa: E402
    dice_from_counts,
    label_components,
    match_components_one_to_one,
    maximum_probability_at_isotropic_mm,
    spacing_from_affine,
)
from src.evaluation.inference import (  # noqa: E402
    lesion_probability,
    logits_to_labels,
    predict_case_in_source_geometry,
    predict_logits,
)
from src.evaluation.postprocessing import filter_lesion_components  # noqa: E402
from src.models.segresnet import build_segresnet  # noqa: E402
from src.training.checkpoint import load_training_checkpoint  # noqa: E402

logger = logging.getLogger("complete_pants_metrics")

ROW_FIELDS = [
    "case_id",
    "ground_truth_patient_positive",
    "frozen_hard_patient_prediction",
    "maximum_p28_patient_score",
    "target_lesion_voxels",
    "predicted_lesion_voxels",
    "intersection_voxels",
    "lesion_dice",
    "gt_tumor_components",
    "predicted_tumor_components",
    "matched_gt_tumors",
    "unmatched_gt_tumors",
    "unmatched_predicted_components",
    "inference_seconds",
]

TUMOR_FIELDS = [
    "case_id",
    "gt_component",
    "gt_voxels",
    "matched",
    "matched_predicted_component",
    "intersection_voxels",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_list(path: Path) -> list[str]:
    cases = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(set(cases)) != len(cases):
        raise ValueError(f"{path} contains duplicate case IDs")
    return cases


def load_ground_truth(case_id: str, args) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Standalone lesion mask, combined-label lesion mask, and voxel spacing.

    Both masks are returned so the caller can assert they agree; T-Sen is
    defined on the standalone mask and the frozen DSC on the combined map, and
    a silent disagreement would make the two metrics describe different truths.
    """
    import nibabel as nib

    paths = get_case_paths(case_id, args.data_split, args.data_root)
    combined_image = nib.load(str(paths["combined"]))
    combined = np.asarray(combined_image.dataobj)
    standalone_path = Path(paths["segmentations"]) / "pancreatic_lesion.nii.gz"
    standalone = np.asarray(nib.load(str(standalone_path)).dataobj)
    spacing = spacing_from_affine(combined_image.affine)
    return (standalone > 0), (combined == PANCREATIC_LESION), spacing


def verify_labels(cases: list[str], args) -> dict[str, Any]:
    """Check the standalone lesion mask equals ``combined_labels == 28`` everywhere."""
    mismatches = []
    positives = 0
    for index, case_id in enumerate(cases, start=1):
        standalone, combined, _ = load_ground_truth(case_id, args)
        if standalone.shape != combined.shape:
            mismatches.append({"case_id": case_id, "reason": "shape",
                               "standalone": list(standalone.shape),
                               "combined": list(combined.shape)})
        elif not np.array_equal(standalone, combined):
            differing = int(np.logical_xor(standalone, combined).sum())
            mismatches.append({"case_id": case_id, "reason": "voxels",
                               "differing_voxels": differing,
                               "standalone_voxels": int(standalone.sum()),
                               "combined_voxels": int(combined.sum())})
        positives += int(bool(combined.any()))
        if index % 100 == 0:
            logger.info("  verified %d/%d, %d mismatches", index, len(cases), len(mismatches))
    return {"cases": len(cases), "lesion_positive": positives,
            "mismatches": mismatches, "identical": not mismatches}


def measure_case_raw(model, case_id: str, args) -> dict[str, Any]:
    """One case in the ORIGINAL CT geometry, exactly as an external user sees it."""
    paths = get_case_paths(case_id, args.data_split, args.data_root)
    started = time.time()
    result = predict_case_in_source_geometry(
        model, str(paths["ct"]), overlap=args.overlap,
        sw_batch_size=args.sw_batch_size, sw_device=args.device,
        accumulate_device=args.accumulate_device,
        want_lesion_probability=True,
        min_lesion_peak_probability=args.lesion_peak_probability,
    )
    elapsed = time.time() - started

    predicted_labels = np.asarray(result["labels"].detach().cpu())[0]
    probability = np.asarray(result["lesion_probability"].detach().cpu())[0]
    standalone, combined, spacing = load_ground_truth(case_id, args)
    if predicted_labels.shape != combined.shape:
        raise RuntimeError(
            f"{case_id}: prediction {predicted_labels.shape} != label {combined.shape}"
        )
    if not np.array_equal(standalone, combined):
        raise RuntimeError(f"{case_id}: standalone lesion mask disagrees with combined_labels")

    predicted_mask = predicted_labels == PANCREATIC_LESION
    score = maximum_probability_at_isotropic_mm(probability, spacing, target_mm=args.score_mm)
    del result, predicted_labels, probability
    return summarize(case_id, standalone, predicted_mask, score, elapsed)


def measure_case_prepared(model, case_id: str, args) -> dict[str, Any]:
    """One case in the canonical 1.5 mm frame, for protocol validation only."""
    from src.data.prepared import case_path, read_prepared_case

    image, label = read_prepared_case(case_path(args.prepared_root, case_id))
    volume = torch.from_numpy(image.astype(np.float32))[None, None]
    started = time.time()
    logits = predict_logits(
        model, volume, overlap=args.overlap, sw_batch_size=args.sw_batch_size,
        sw_device=args.device, device=args.accumulate_device,
    )
    labels = logits_to_labels(logits)
    if args.lesion_peak_probability is not None:
        report = filter_lesion_components(
            logits, labels, min_peak_probability=args.lesion_peak_probability
        )
        labels = report.pop("labels")
    probability = lesion_probability(logits)[0, 0].to("cpu").numpy()
    elapsed = time.time() - started

    predicted_mask = labels[0, 0].to("cpu").numpy() == PANCREATIC_LESION
    truth_mask = np.asarray(label) == PANCREATIC_LESION
    score = maximum_probability_at_isotropic_mm(
        probability, (1.5, 1.5, 1.5), target_mm=args.score_mm
    )
    del logits, labels, volume, probability
    return summarize(case_id, truth_mask, predicted_mask, score, elapsed)


def summarize(
    case_id: str,
    truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    score: float,
    elapsed: float,
) -> dict[str, Any]:
    """Collapse two masks and one score into scalars plus per-tumor records."""
    target_voxels = int(truth_mask.sum())
    predicted_voxels = int(predicted_mask.sum())
    intersection = int(np.logical_and(truth_mask, predicted_mask).sum())

    truth_labels, truth_count = label_components(truth_mask)
    predicted_labels, predicted_count = label_components(predicted_mask)
    matched = match_components_one_to_one(
        truth_labels, truth_count, predicted_labels, predicted_count
    )
    matched_by_truth = dict(matched)

    tumors = []
    for component in range(1, truth_count + 1):
        component_mask = truth_labels == component
        partner = matched_by_truth.get(component)
        tumors.append({
            "case_id": case_id,
            "gt_component": component,
            "gt_voxels": int(component_mask.sum()),
            "matched": partner is not None,
            "matched_predicted_component": partner if partner is not None else "",
            "intersection_voxels": (
                int(np.logical_and(component_mask, predicted_labels == partner).sum())
                if partner is not None else 0
            ),
        })

    row = {
        "case_id": case_id,
        "ground_truth_patient_positive": target_voxels > 0,
        "frozen_hard_patient_prediction": predicted_voxels > 0,
        "maximum_p28_patient_score": float(score),
        "target_lesion_voxels": target_voxels,
        "predicted_lesion_voxels": predicted_voxels,
        "intersection_voxels": intersection,
        "lesion_dice": dice_from_counts(intersection, predicted_voxels, target_voxels),
        "gt_tumor_components": truth_count,
        "predicted_tumor_components": predicted_count,
        "matched_gt_tumors": len(matched),
        "unmatched_gt_tumors": truth_count - len(matched),
        "unmatched_predicted_components": predicted_count - len(matched),
        "inference_seconds": round(float(elapsed), 4),
    }
    return {"row": row, "tumors": tumors}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=["raw", "prepared"], default="raw")
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "pants_cv_v1.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-split", choices=["train", "test"], default="train")
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lesion-peak-probability", type=float, default=0.6)
    parser.add_argument("--score-mm", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--accumulate-device", default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test-split", action="store_true")
    parser.add_argument("--verify-labels", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_cases(args) -> list[str]:
    if args.case_list:
        return read_case_list(args.case_list)
    split = json.loads(Path(args.split).read_text())
    return list(split["folds"][args.fold]["val"])


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s"
    )
    args.output.mkdir(parents=True, exist_ok=True)

    if args.data_split == "test" and not args.allow_test_split:
        raise SystemExit("refusing to read the test split without --allow-test-split")

    cases = resolve_cases(args)
    logger.info("%d cases, split=%s, mode=%s", len(cases), args.data_split, args.mode)

    if args.verify_labels:
        report = verify_labels(cases, args)
        destination = args.output / "label_consistency.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        logger.info(
            "standalone lesion mask identical to combined_labels==28: %s (%d positives)",
            report["identical"], report["lesion_positive"],
        )
        if not report["identical"]:
            logger.error("%d MISMATCHES -> %s", len(report["mismatches"]), destination)
            return 1
        return 0

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --verify-labels is set")
    checkpoint_sha = sha256_of(args.checkpoint)
    logger.info("checkpoint %s sha256 %s", args.checkpoint.name, checkpoint_sha)

    # Same construction and load path as the frozen evaluator, so the weights
    # in memory here are the weights that produced the original result.
    model = build_segresnet(initialization="random")
    state = load_training_checkpoint(args.checkpoint, model=model)
    missing = set(model.state_dict()) - set(state["model"])
    if missing:
        raise SystemExit(f"checkpoint does not define {len(missing)} model tensors")
    model = model.to(args.device).eval()
    logger.info("epoch %s", state.get("epoch"))

    rows_path = args.output / "per_case_metric_completion.csv"
    tumors_path = args.output / "per_tumor_matching.csv"
    done: set[str] = set()
    if args.resume and rows_path.exists():
        with open(rows_path, newline="") as handle:
            done = {row["case_id"] for row in csv.DictReader(handle)}
        logger.info("resuming: %d cases already recorded", len(done))

    new_file = not rows_path.exists()
    measure = measure_case_raw if args.mode == "raw" else measure_case_prepared
    started = time.time()
    with open(rows_path, "a", newline="") as row_handle, \
            open(tumors_path, "a", newline="") as tumor_handle:
        row_writer = csv.DictWriter(row_handle, fieldnames=ROW_FIELDS)
        tumor_writer = csv.DictWriter(tumor_handle, fieldnames=TUMOR_FIELDS)
        if new_file:
            row_writer.writeheader()
            tumor_writer.writeheader()
        for index, case_id in enumerate(cases, start=1):
            if case_id in done:
                continue
            measured = measure(model, case_id, args)
            row_writer.writerow(measured["row"])
            tumor_writer.writerows(measured["tumors"])
            row_handle.flush()
            tumor_handle.flush()
            if index % 25 == 0:
                rate = (time.time() - started) / max(index - len(done), 1)
                logger.info("  %d/%d  %.2f s/case", index, len(cases), rate)

    elapsed = time.time() - started
    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": state.get("epoch"),
        "mode": args.mode,
        "data_split": args.data_split,
        "cases": len(cases),
        "case_list": str(args.case_list) if args.case_list else None,
        "case_list_sha256": sha256_of(args.case_list) if args.case_list else None,
        "lesion_peak_probability": args.lesion_peak_probability,
        "connectivity": 26,
        "patient_score": f"max class-28 softmax, linear resample to {args.score_mm} mm isotropic",
        "sliding_window": {
            "roi_size": [96, 96, 96], "overlap": args.overlap,
            "sw_batch_size": args.sw_batch_size,
            "accumulate_device": args.accumulate_device, "blend_mode": "gaussian",
        },
        "total_seconds": elapsed,
    }
    (args.output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("%d cases in %.1f min -> %s", len(cases), elapsed / 60.0, rows_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
