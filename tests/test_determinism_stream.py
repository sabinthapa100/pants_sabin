"""The sampled crop/augmentation stream must follow the seed, not drift.

Statistics are hashed rather than tensors stored: the invariant is "the same
seed produces the same stream", which a signature captures without committing
megabytes of reference data to the repository.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from src.data.prepared import build_prepared_dataloaders, select_monitoring_cases
from src.training.trainer import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "pants_cv_v1.json"
prepared_root = os.environ.get("PANTS_PREPARED_ROOT")

requires_cache = pytest.mark.skipif(
    not (prepared_root and (Path(prepared_root) / "cases").is_dir() and SPLIT.exists()),
    reason="PANTS_PREPARED_ROOT not set to a prepared cache",
)


def stream_signature(seed: int, batches: int = 4, workers: int = 0) -> str:
    """Hash of the first few sampled patches: content, order and augmentation."""
    seed_everything(seed)
    split = json.loads(SPLIT.read_text())
    train_loader, _ = build_prepared_dataloaders(
        split, prepared_root, fold=0, batch_size=1, samples_per_case=2,
        num_workers=workers, limit=6,
    )

    digest = hashlib.sha256()
    for index, batch in enumerate(train_loader):
        if index >= batches:
            break
        image, label = batch["image"], batch["label"]
        # summary statistics are sensitive to which voxels were cropped and to
        # the intensity augmentation applied afterwards
        for value in (
            float(image.sum()), float(image.mean()), float(image.std()),
            float(label.sum()), float(label.max()),
        ):
            digest.update(f"{value:.6f}".encode())
    return digest.hexdigest()[:16]


@requires_cache
def test_same_seed_gives_the_same_stream():
    assert stream_signature(317) == stream_signature(317)


@requires_cache
def test_different_seed_gives_a_different_stream():
    assert stream_signature(317) != stream_signature(318)


@requires_cache
def test_worker_processes_do_not_break_determinism():
    """Multi-worker loading must stay reproducible, not just single-process."""
    assert stream_signature(317, workers=2) == stream_signature(317, workers=2)


@requires_cache
def test_monitoring_subset_is_fixed_and_ordered():
    """Model selection compares arms only if both score the identical cases."""
    split = json.loads(SPLIT.read_text())
    manifest = json.loads((Path(prepared_root) / "manifest.json").read_text())

    first = select_monitoring_cases(manifest, split, fold=0, negatives=100)
    second = select_monitoring_cases(manifest, split, fold=0, negatives=100)
    assert first == second == sorted(first)

    lesion = {case["case_id"]: case["lesion_present"] for case in manifest["cases"]}
    validation = set(split["folds"][0]["val"])
    assert set(first).issubset(validation), "monitoring cases must live inside fold-0 val"
    assert sum(1 for case in first if lesion[case]) == 177, "all lesion-positive cases included"

    # seeding must not perturb it: this subset is not a random sample
    seed_everything(999)
    assert select_monitoring_cases(manifest, split, fold=0, negatives=100) == first


def test_resume_path_does_not_reseed():
    """A resumed run must continue the RNG stream, not restart it."""
    import inspect

    from src.training.trainer import SegResNetTrainer

    fit_source = inspect.getsource(SegResNetTrainer.fit)
    assert "seed_everything" not in fit_source
    assert "manual_seed" not in fit_source

    resume_source = inspect.getsource(SegResNetTrainer.resume)
    assert "seed_everything" not in resume_source
    assert "manual_seed" not in resume_source

    # and the seed is established before the model is built
    init_source = inspect.getsource(SegResNetTrainer.__init__)
    assert init_source.index("seed_everything") < init_source.index("build_segresnet")
