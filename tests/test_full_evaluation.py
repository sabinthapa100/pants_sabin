"""The post-training path: full-fold evaluation, plotting, and checkpoint usability.

Three things are protected here.

* The evaluator scores the *whole* fold with no stochastic component, and
  structurally refuses PanTS-te.
* A trained checkpoint fully defines an inference model, so the SuPreM
  initialization file is never needed again.
* The plotting script cannot silently turn a calibration run into a figure that
  looks like a result.
"""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.labels import CLASS_MAP
from src.evaluation.segmentation import all_class_dice, per_class_dice
from src.models.segresnet import build_segresnet
from src.training.checkpoint import save_training_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "pants_cv_v1.json"
prepared_root = os.environ.get("PANTS_PREPARED_ROOT")


def load_script(name: str):
    """`scripts/` is a directory of CLIs, not a package."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load_script("evaluate_segresnet")
analyze = load_script("analyze_segresnet")

requires_split = pytest.mark.skipif(not SPLIT.exists(), reason="pants_cv_v1.json absent")
requires_cache = pytest.mark.skipif(
    not (prepared_root and (Path(prepared_root) / "cases").is_dir() and SPLIT.exists()),
    reason="PANTS_PREPARED_ROOT not set to a prepared cache",
)


# --------------------------------------------------------------------------- #
# metric equivalence: the fast path must not be a second definition
# --------------------------------------------------------------------------- #


def test_all_class_dice_matches_per_class_dice():
    """The histogram shortcut must be exact, including the NaN convention."""
    rng = np.random.default_rng(317)
    labels = sorted(CLASS_MAP)
    for _ in range(5):
        target = rng.integers(0, 29, size=(16, 14, 12)).astype(np.int16)
        pred = np.where(rng.random(target.shape) < 0.5, target,
                        rng.integers(0, 29, size=target.shape)).astype(np.int16)
        slow = per_class_dice(pred, target, labels)
        fast = all_class_dice(pred, target, labels)
        for label in labels:
            if math.isnan(slow[label]):
                assert math.isnan(fast[label]), f"class {label}: NaN disagreement"
            else:
                assert slow[label] == pytest.approx(fast[label], abs=1e-12)


def test_all_class_dice_keeps_absent_in_both_undefined():
    """A class in neither volume must stay NaN, never a free 1.0."""
    empty = np.zeros((4, 4, 4), dtype=np.int16)
    scores = all_class_dice(empty, empty, sorted(CLASS_MAP))
    assert all(math.isnan(value) for value in scores.values())


def test_all_class_dice_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        all_class_dice(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)), [1])


# --------------------------------------------------------------------------- #
# PanTS-te must be unreachable without an explicit flag
# --------------------------------------------------------------------------- #


def test_guard_refuses_test_split_case_ids():
    with pytest.raises(SystemExit, match="PanTS-te"):
        evaluate.guard_split(["PanTS_00000005", "PanTS_00009001"], allow_test=False)


def test_guard_allows_training_ids():
    evaluate.guard_split(["PanTS_00000001", "PanTS_00009000"], allow_test=False)


def test_guard_can_be_unlocked_deliberately():
    evaluate.guard_split(["PanTS_00009001"], allow_test=True)


@requires_split
def test_production_split_contains_no_test_cases():
    split = json.loads(SPLIT.read_text())
    ids = [evaluate.case_number(case)
           for fold in split["folds"] for key in ("train", "val") for case in fold[key]]
    assert max(ids) <= evaluate.LAST_TRAIN_CASE_ID


# --------------------------------------------------------------------------- #
# full-fold selection
# --------------------------------------------------------------------------- #


@requires_split
def test_resolve_cases_returns_the_whole_fold_sorted(tmp_path):
    args = evaluate.argparse.Namespace(
        split=SPLIT, fold=0, limit=None, allow_test_split=False
    )
    cases = evaluate.resolve_cases(args)
    split = json.loads(SPLIT.read_text())
    assert len(cases) == len(split["folds"][0]["val"]) == 1801
    assert cases == sorted(cases), "evaluation order must be deterministic"
    assert set(cases).isdisjoint(split["folds"][0]["train"])


@requires_split
def test_resolve_cases_rejects_an_out_of_range_fold():
    args = evaluate.argparse.Namespace(split=SPLIT, fold=9, limit=None, allow_test_split=False)
    with pytest.raises(SystemExit, match="fold 9"):
        evaluate.resolve_cases(args)


# --------------------------------------------------------------------------- #
# scoring and aggregation
# --------------------------------------------------------------------------- #


def synthetic_rows() -> list[dict]:
    """Two positives (one perfect, one missed) and two negatives (one false positive)."""
    volume = np.zeros((8, 8, 8), dtype=np.int16)
    rows = []

    target = volume.copy(); target[0:2, 0:2, 0:2] = 28
    rows.append(dict(evaluate._score(target.copy(), target, 3.375), case_id="a"))

    pred = volume.copy()
    rows.append(dict(evaluate._score(pred, target, 3.375), case_id="b"))          # missed

    rows.append(dict(evaluate._score(volume.copy(), volume.copy(), 3.375), case_id="c"))
    fp = volume.copy(); fp[3, 3, 3] = 28
    rows.append(dict(evaluate._score(fp, volume.copy(), 3.375), case_id="d"))     # false positive

    for index, row in enumerate(rows):
        row["inference_seconds"] = 1.0 + index
    return rows


def test_score_reports_voxels_volume_and_outcome():
    rows = synthetic_rows()
    perfect, missed, clean, false_positive = rows

    assert perfect["lesion_present"] and perfect["lesion_dice"] == 1.0 and perfect["detected"]
    assert perfect["target_lesion_voxels"] == 8
    assert perfect["target_lesion_mm3"] == pytest.approx(8 * 3.375)

    assert missed["lesion_present"] and missed["lesion_dice"] == 0.0
    assert not missed["detected"] and not missed["false_positive"]

    assert not clean["lesion_present"] and math.isnan(clean["dice_28"])
    assert false_positive["false_positive"] and false_positive["predicted_lesion_voxels"] == 1


def test_aggregate_separates_populations_and_reports_the_documented_schema():
    summary = evaluate.aggregate(synthetic_rows())

    tumor = summary["primary_tumor_segmentation"]["class28_dice_on_positive_cases"]
    assert summary["primary_tumor_segmentation"]["lesion_positive_cases"] == 2
    assert tumor["mean"] == pytest.approx(0.5)          # 1.0 and 0.0, not inflated by negatives
    assert tumor["zero_dice_positive_cases"] == 1

    detection = summary["internal_case_detection"]
    assert detection["positive_case_detection_rate"] == pytest.approx(0.5)
    assert detection["negative_case_false_positive_rate"] == pytest.approx(0.5)
    assert detection["internal_specificity"] == pytest.approx(0.5)
    assert "INTERNAL" in detection

    anatomy = summary["anatomy_aware_segmentation"]
    assert set(anatomy["per_class"]) == {str(c) for c in sorted(CLASS_MAP)}
    assert anatomy["per_class"]["28"]["name"] == "pancreatic_lesion"
    assert "macro_foreground_dice_1_28" in anatomy
    assert set(anatomy["pancreas_family"]) == {"17", "18", "19", "20", "21", "28"}

    assert summary["lesion_volume"]["absolute_error_mm3"]["n"] == 2
    assert summary["efficiency"]["seconds_per_case"]["n"] == 4


def test_metric_definitions_refuse_to_claim_official_status():
    text = evaluate.METRIC_DEFINITIONS["official_pants_metrics"]
    assert "NOT COMPUTED" in text
    assert "unpublished" in text
    for key in ("detected", "false_positive", "internal_specificity"):
        assert "INTERNAL" in evaluate.METRIC_DEFINITIONS[key] or "internal" in key


def test_stratification_is_labelled_exploratory_and_shows_its_bin_edges():
    rng = np.random.default_rng(0)
    rows = [
        {"lesion_present": True, "target_lesion_mm3": float(v),
         "lesion_dice": float(rng.random()), "detected": True}
        for v in rng.integers(10, 100_000, size=40)
    ]
    result = evaluate.stratify_by_lesion_volume(rows)
    assert "EXPLORATORY" in result
    assert len(result["bin_edges_mm3"]) == 2
    assert result["bin_edges_mm3"][0] < result["bin_edges_mm3"][1]
    assert sum(b["cases"] for b in result["bins"].values()) == 40


def test_stratification_declines_when_there_is_nothing_to_stratify():
    assert "note" in evaluate.stratify_by_lesion_volume([])


# --------------------------------------------------------------------------- #
# a trained checkpoint is self-sufficient
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory) -> Path:
    """Stands in for best.pt: every tensor deliberately different from a fresh init."""
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = build_segresnet(initialization="random")
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(1.5)
    save_training_checkpoint(
        path, model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        scheduler=None, scaler=None, epoch=59, global_step=216_000, best_metric=0.42,
        config={"config": {"initialization": "suprem"},
                "selection_metric_name": "mean_dice_on_positive_cases",
                "selection_metric_value": 0.42},
        git_commit="afdb75f32075c5ed8e62f9dc631a199f0e70b9b6",
    )
    return path


def test_checkpoint_defines_every_tensor_of_the_29_class_model(trained_checkpoint):
    """No parameter may survive from the constructor's random draw."""
    model, checkpoint = evaluate.load_model(trained_checkpoint, "cpu")

    assert model.state_dict()["conv_final.2.conv.weight"].shape[0] == 29
    assert set(model.state_dict()) == set(checkpoint["model"]), "tensor sets differ"
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, checkpoint["model"][name]), f"{name} not restored"


def test_inference_model_ignores_the_suprem_file(trained_checkpoint, monkeypatch):
    """SUPREM_CHECKPOINT is irrelevant once training has finished."""
    monkeypatch.delenv("SUPREM_CHECKPOINT", raising=False)
    model, _ = evaluate.load_model(trained_checkpoint, "cpu")
    reference = build_segresnet(initialization="random").state_dict()
    differing = sum(
        1 for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and not torch.equal(tensor, reference[name])
    )
    assert differing > 0, "loaded weights are indistinguishable from a fresh init"


def test_load_model_rejects_a_checkpoint_missing_tensors(tmp_path):
    model = build_segresnet(initialization="random")
    incomplete = {k: v for k, v in model.state_dict().items() if "conv_final" not in k}
    torch.save({"model": incomplete, "epoch": 0, "global_step": 0, "best_metric": 0.0,
                "config": {}, "git_commit": None,
                "python_rng_state": None, "numpy_rng_state": None,
                "torch_rng_state": None, "cuda_rng_state": None},
               tmp_path / "broken.pt")
    with pytest.raises((SystemExit, RuntimeError, KeyError, TypeError)):
        evaluate.load_model(tmp_path / "broken.pt", "cpu")


# --------------------------------------------------------------------------- #
# plotting cannot invent or mislabel results
# --------------------------------------------------------------------------- #


def write_run(directory: Path, epochs: int = 10, **config) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        record = {"epoch": epoch, "train_loss": 2.0 - epoch * 0.1,
                  "patch_val_loss_diagnostic": 2.1 - epoch * 0.09,
                  "learning_rate": 1e-4, "seconds": 1900.0}
        if (epoch + 1) % 5 == 0:
            record["selection"] = {
                "mean_dice_on_positive_cases": 0.1 * epoch,
                "case_detection_rate_on_positive_cases": 0.5,
                "false_positive_rate_on_negative_cases": 0.2,
                "lesion_positive_cases": 177, "lesion_negative_cases": 50,
            }
        history.append(record)
    (directory / "history.json").write_text(json.dumps(history))
    base = {"max_steps_per_epoch": None, "limit_cases": None, "epochs": epochs}
    base.update(config)
    (directory / "provenance.json").write_text(json.dumps({"config": base}))
    return directory


def test_analysis_refuses_a_calibration_run(tmp_path):
    run = write_run(tmp_path / "calib", epochs=2, max_steps_per_epoch=80)
    assert analyze.load_history(run, allow_partial=False) is None
    assert analyze.load_history(run, allow_partial=True) is not None


def test_analysis_refuses_a_case_limited_run(tmp_path):
    run = write_run(tmp_path / "smoke", epochs=2, limit_cases=40)
    assert analyze.load_history(run, allow_partial=False) is None


def test_analysis_skips_figures_whose_inputs_are_absent(tmp_path, capsys):
    run = write_run(tmp_path / "segresnet_suprem")
    output = tmp_path / "figures"
    analyze.sys.argv = [
        "analyze_segresnet.py", "--suprem-run", str(run), "--output", str(output),
    ]
    assert analyze.main() == 0

    produced = {path.name for path in output.glob("*.png")}
    assert "01_train_loss.png" in produced
    assert "04_monitoring_class28_dice.png" in produced
    # evaluation-derived figures have no inputs here and must not be fabricated
    assert "08_class28_dice_distribution.png" not in produced
    assert "09_per_class_dice.png" not in produced
    assert "nothing was invented" in capsys.readouterr().out


def test_patch_validation_figure_is_labelled_diagnostic():
    assert "DIAGNOSTIC ONLY" in analyze.DIAGNOSTIC
    assert "does not select" in analyze.DIAGNOSTIC


# --------------------------------------------------------------------------- #
# end to end, when a real cache is available
# --------------------------------------------------------------------------- #


@requires_cache
def test_evaluator_writes_the_documented_files(tmp_path, trained_checkpoint):
    output = tmp_path / "evaluation"
    evaluate.sys.argv = [
        "evaluate_segresnet.py",
        "--checkpoint", str(trained_checkpoint),
        "--prepared-root", prepared_root,
        "--manifest", str(Path(prepared_root) / "manifest.json"),
        "--split", str(SPLIT), "--fold", "0", "--limit", "2",
        "--device", "cpu", "--output", str(output),
    ]
    assert evaluate.main() == 0

    import csv

    rows = list(csv.DictReader(open(output / "evaluation_cases.csv")))
    assert len(rows) == 2
    for column in ("case_id", "lesion_present", "target_lesion_voxels", "lesion_dice",
                   "detected", "false_positive", "inference_seconds", "dice_28"):
        assert column in rows[0]

    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["checkpoint"]["sha256"]
    assert summary["checkpoint"]["epoch"] == 59
    assert summary["checkpoint"]["training_git_commit"].startswith("afdb75f")
    assert summary["evaluation"]["evaluation_frame"] == "prepared_RAS_1.5mm"
    assert summary["evaluation"]["cases_failed"] == 0
    assert summary["evaluation"]["sliding_window"]["roi_size"] == [96, 96, 96]
    assert summary["evaluation"]["split_sha256"]
    assert "NOT COMPUTED" in summary["metric_definitions"]["official_pants_metrics"]
