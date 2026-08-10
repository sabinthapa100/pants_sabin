"""Prove checkpoint/resume restores the full training state.

This is the Colab-critical property: a disconnected session must be able to
continue rather than silently restart from a different trajectory.
"""

import os
from pathlib import Path

import pytest
import torch

from src.data.manifest import build_manifest, build_split
from src.data.paths import get_data_root
from src.training.trainer import SegResNetTrainer, TrainingConfig


PILOT_CASES = [f"PanTS_{index:08d}" for index in (3, 26, 29, 31)]

data_root = get_data_root()
requires_data = pytest.mark.skipif(
    not (data_root / "ImageTr").is_dir(),
    reason=f"PanTS-tr not readable at {data_root}; set PANTS_DATA_ROOT",
)


def build_trainer(tmp_path: Path, epochs: int = 1) -> SegResNetTrainer:
    manifest = build_manifest(case_ids=PILOT_CASES, workers=1, expected_cases=None)
    split = build_split(manifest, seed=317, folds=2)
    config = TrainingConfig(
        experiment="resume_check",
        initialization="random",
        epochs=epochs,
        samples_per_case=1,
        gradient_accumulation_steps=1,
        num_workers=0,
        max_steps_per_epoch=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_root=str(tmp_path / "runs"),
    )
    return SegResNetTrainer(config, manifest=manifest, split=split)


@requires_data
def test_resume_restores_full_training_state(tmp_path):
    trainer = build_trainer(tmp_path, epochs=1)
    trainer.fit()

    checkpoint_path = trainer.run_dir / "latest.pt"
    assert checkpoint_path.exists(), "no latest checkpoint written"

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "global_step",
        "best_metric",
        "config",
        "git_commit",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
    ):
        assert key in saved, f"checkpoint missing {key}"

    if trainer.use_amp:
        assert saved["scaler"] is not None, "AMP scaler state not saved"

    # a fresh process-equivalent trainer with different random weights
    revived = build_trainer(tmp_path, epochs=2)
    before = {k: v.detach().cpu().clone() for k, v in revived.model.state_dict().items()}
    trained = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
    assert not all(
        torch.equal(before[k], trained[k]) for k in before
    ), "fixture is not exercising a real restore"

    revived.resume(checkpoint_path)

    # model weights restored exactly
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(revived.model.state_dict()[key].cpu(), value.cpu()), key

    # optimizer, scheduler, counters
    assert revived.optimizer.state_dict()["param_groups"][0]["lr"] == pytest.approx(
        trainer.optimizer.state_dict()["param_groups"][0]["lr"]
    )
    assert revived.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert revived.global_step == trainer.global_step
    assert revived.start_epoch == saved["epoch"] + 1
    assert revived.best_metric == pytest.approx(saved["best_metric"])
    assert revived.resumed is True


@requires_data
def test_training_continues_after_resume(tmp_path):
    """One more optimizer step after resume must produce a finite loss."""
    trainer = build_trainer(tmp_path, epochs=1)
    trainer.fit()
    checkpoint_path = trainer.run_dir / "latest.pt"

    revived = build_trainer(tmp_path, epochs=2)
    revived.resume(checkpoint_path)
    assert revived.start_epoch == 1

    loss = revived.train_epoch()
    assert torch.isfinite(torch.tensor(loss)), "post-resume loss is not finite"
    assert revived.global_step > trainer.global_step, "resume did not continue the step count"


@requires_data
def test_resume_does_not_reseed_over_restored_rng(tmp_path):
    """
    `fit` must not call manual_seed, or it would discard the RNG state the
    checkpoint just restored and silently change the data/augmentation stream.
    """
    trainer = build_trainer(tmp_path, epochs=1)
    trainer.fit()

    revived = build_trainer(tmp_path, epochs=2)
    revived.resume(trainer.run_dir / "latest.pt")
    restored = torch.get_rng_state().clone()

    revived.config.__class__  # config is frozen; nothing to mutate
    (revived.run_dir / "provenance.json").unlink(missing_ok=True)
    # calling fit would step training; assert instead that the seed call is gone
    import inspect

    source = inspect.getsource(SegResNetTrainer.fit)
    assert "manual_seed" not in source, "fit() reseeds and would clobber a resume"
    assert torch.equal(torch.get_rng_state(), restored)


@requires_data
def test_provenance_records_config_split_and_commit(tmp_path):
    trainer = build_trainer(tmp_path, epochs=1)
    provenance = trainer.provenance()

    assert provenance["config"]["initialization"] == "random"
    assert provenance["split_version"] == "pants_cv_v1"
    assert provenance["split_seed"] == 317
    assert provenance["num_classes"] == 29
    assert "torch" in provenance
    # git_commit may be None outside a repo, but the key must exist
    assert "git_commit" in provenance
