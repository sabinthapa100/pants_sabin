"""best.pt must be chosen by a deterministic signal, and must survive a crash.

Two invariants:

* the number that selects the production checkpoint does not move when the
  model has not moved;
* a completed checkpoint reaches persistent storage during training, not in a
  later cell that a dead Colab runtime would never reach.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from src.training.trainer import SegResNetTrainer, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "pants_cv_v1.json"
prepared_root = os.environ.get("PANTS_PREPARED_ROOT")

requires_cache = pytest.mark.skipif(
    not (prepared_root and (Path(prepared_root) / "cases").is_dir() and SPLIT.exists()),
    reason="PANTS_PREPARED_ROOT not set to a prepared cache",
)


def make_trainer(tmp_path, **overrides) -> SegResNetTrainer:
    config = TrainingConfig(
        experiment="selection_test",
        initialization="random",
        prepared_root=prepared_root,
        fold=0,
        epochs=2,
        batch_size=1,
        samples_per_case=2,
        gradient_accumulation_steps=1,
        num_workers=0,
        limit_cases=4,
        max_steps_per_epoch=2,
        patch_validation_batches=2,
        validate_every_epochs=1,
        output_root=str(tmp_path / "runs"),
        **overrides,
    )
    manifest = json.loads((Path(prepared_root) / "manifest.json").read_text())
    split = json.loads(SPLIT.read_text())
    return SegResNetTrainer(config, manifest=manifest, split=split)


@requires_cache
def test_whole_volume_validation_is_repeatable(tmp_path):
    """The selection metric must not move on an unchanged model."""
    trainer = make_trainer(tmp_path)
    trainer.monitoring_cases = trainer.monitoring_cases[:2]

    first = trainer.validate_volumes()
    second = trainer.validate_volumes()

    for key in first:
        if key in ("seconds", "note"):
            continue
        assert first[key] == second[key], f"{key} changed on an unchanged model"
    print(f"\n  repeatable: class-28 Dice {first['mean_dice_on_positive_cases']}")


@requires_cache
def test_patch_validation_is_not_repeatable_and_never_selects(tmp_path):
    """Justifies the split: the cheap signal really is stochastic."""
    trainer = make_trainer(tmp_path)
    losses = {round(trainer.validate(), 8) for _ in range(4)}
    assert len(losses) > 1, "patch validation was expected to vary between calls"

    import inspect

    source = inspect.getsource(SegResNetTrainer.fit)
    selection_block = source[source.index("_should_select"):]
    assert "best.pt" in selection_block
    assert "patch_loss" not in selection_block.split("self.history.append")[0], (
        "patch loss must not participate in choosing best.pt"
    )


@requires_cache
def test_selection_metric_and_subset_are_recorded(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.monitoring_cases = trainer.monitoring_cases[:2]
    provenance = trainer.provenance()

    assert provenance["selection_metric_name"] == "mean_dice_on_positive_cases"
    assert provenance["monitoring_subset_fingerprint"] == trainer.monitoring_fingerprint
    assert "learning_rate_current" in provenance
    assert provenance["data_source"] == "prepared_cache"


@requires_cache
def test_checkpoints_persist_during_training(tmp_path):
    """A completed epoch must land in persistent storage before the next starts."""
    persistent = tmp_path / "drive" / "PanTS_runs"
    trainer = make_trainer(tmp_path, persistent_output_root=str(persistent))
    trainer.monitoring_cases = trainer.monitoring_cases[:1]
    summary = trainer.fit()

    run = persistent / "selection_test"
    assert (run / "latest.pt").is_file(), "latest.pt never reached persistent storage"
    assert (run / "provenance.json").is_file()
    assert (run / "history.json").is_file()
    assert (run / "summary.json").is_file()
    assert not list(run.glob("*.partial")), "an interrupted copy was left behind"

    local = torch.load(tmp_path / "runs" / "selection_test" / "latest.pt",
                       map_location="cpu", weights_only=False)
    mirrored = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert local["global_step"] == mirrored["global_step"]
    for key, tensor in local["model"].items():
        assert torch.equal(tensor, mirrored["model"][key])

    assert summary["selection_metric"] == "mean_dice_on_positive_cases"
    assert summary["monitoring_subset_fingerprint"]


@requires_cache
def test_resume_restores_selection_state(tmp_path):
    persistent = tmp_path / "drive" / "PanTS_runs"
    first = make_trainer(tmp_path, persistent_output_root=str(persistent))
    first.monitoring_cases = first.monitoring_cases[:1]
    first.fit()

    checkpoint_path = persistent / "selection_test" / "latest.pt"
    resumed = make_trainer(tmp_path)
    resumed.monitoring_cases = resumed.monitoring_cases[:1]
    checkpoint = resumed.resume(checkpoint_path)

    assert resumed.start_epoch == int(checkpoint["epoch"]) + 1
    assert resumed.global_step == first.global_step
    assert resumed.best_metric == first.best_metric
    assert resumed.selection_epoch == first.selection_epoch


@requires_cache
def test_resume_refuses_a_changed_monitoring_subset(tmp_path):
    """Comparing metrics computed on different cases would be meaningless."""
    persistent = tmp_path / "drive" / "PanTS_runs"
    trainer = make_trainer(tmp_path, persistent_output_root=str(persistent))
    trainer.monitoring_cases = trainer.monitoring_cases[:1]
    trainer.fit()

    other = make_trainer(tmp_path, monitoring_negatives=7)
    with pytest.raises(ValueError, match="Monitoring subset changed"):
        other.resume(persistent / "selection_test" / "latest.pt")
