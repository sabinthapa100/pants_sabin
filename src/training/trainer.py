"""One minimal SegResNet trainer, shared by both initialization arms.

The same class trains ``initialization="random"`` and ``initialization="suprem"``.
Nothing in this file branches on the initialization beyond constructing the
model, which is exactly what makes the pretraining comparison controlled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from ..data.prepared import (
    build_prepared_dataloaders,
    case_path,
    cases_fingerprint,
    read_prepared_case,
    select_monitoring_cases,
)
from ..data.transforms import PREPROCESSING, build_dataloaders
from ..evaluation.inference import logits_to_labels, predict_logits
from ..evaluation.segmentation import lesion_case_metrics, summarize_lesion_metrics
from ..models.segresnet import NUM_CLASSES, build_segresnet
from .checkpoint import load_training_checkpoint, save_training_checkpoint


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Everything needed to reproduce a run."""

    experiment: str
    initialization: str = "random"
    pretrained_checkpoint: str | None = None
    fold: int = 0
    epochs: int = 2
    batch_size: int = 1
    samples_per_case: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    seed: int = 317
    limit_cases: int | None = None
    max_steps_per_epoch: int | None = None
    # When set, training reads the prepared npz cache instead of raw NIfTI.
    # The deterministic preprocessing is identical either way; only where the
    # work happened differs (laptop, once, versus every epoch).
    prepared_root: str | None = None
    save_every_epochs: int = 1
    amp: bool = True
    device: str = "cuda"
    output_root: str = "outputs/runs"
    # Durable copy of the run artifacts, e.g. a mounted Drive directory. The
    # local root stays the fast disk; this is what survives a Colab disconnect.
    persistent_output_root: str | None = None
    # Deterministic whole-volume model selection.
    validate_every_epochs: int = 5
    monitoring_negatives: int = 50
    selection_metric_name: str = "mean_dice_on_positive_cases"
    # Patch validation is a diagnostic only, so it runs on a fixed prefix of the
    # validation loader rather than all 1,801 cases. Measured: the full pass
    # costs 155 s per epoch to produce a number that never selects anything.
    patch_validation_batches: int = 200
    preprocessing: dict[str, Any] = field(default_factory=lambda: dict(PREPROCESSING))


def git_commit() -> str | None:
    """Current commit, for run provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def seed_everything(seed: int) -> None:
    """Establish the run seed before anything stochastic is constructed.

    ``monai.utils.set_determinism`` seeds Python's ``random``, NumPy, torch CPU
    and all CUDA devices, and sets the global MONAI seed that ``Randomizable``
    transforms draw from. It must run before ``build_segresnet``: Conv3d weights
    are sampled at construction, so seeding afterwards leaves the initial
    weights - and therefore the whole run - outside the seed's control.
    """
    set_determinism(seed=int(seed))


def resolve_device(requested: str) -> torch.device:
    """Use CUDA when asked for and available; otherwise fall back to CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


class SegResNetTrainer:
    """Minimal train/validate/checkpoint loop for the PanTS SegResNet."""

    def __init__(
        self,
        config: TrainingConfig,
        manifest: dict[str, Any],
        split: dict[str, Any],
        data_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.split = split
        self.device = resolve_device(config.device)
        self.use_amp = bool(config.amp and self.device.type == "cuda")

        # FIRST, before anything stochastic exists. The model's Conv3d weights
        # are drawn during construction, so this line is what makes the two
        # initialization arms start from an identical PanTS output head.
        seed_everything(config.seed)

        self.run_dir = Path(config.output_root) / config.experiment
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_dir = (
            Path(config.persistent_output_root) / config.experiment
            if config.persistent_output_root
            else None
        )

        self.model = build_segresnet(
            initialization=config.initialization,
            checkpoint_path=config.pretrained_checkpoint,
        ).to(self.device)

        # Exclusive 29-class semantic target. Background is excluded from the
        # Dice term because it dominates the volume; cross-entropy still sees
        # it, so background is supervised without swamping the Dice signal.
        self.criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(config.epochs, 1)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if config.prepared_root:
            self.train_loader, self.val_loader = build_prepared_dataloaders(
                split,
                config.prepared_root,
                fold=config.fold,
                batch_size=config.batch_size,
                samples_per_case=config.samples_per_case,
                num_workers=config.num_workers,
                limit=config.limit_cases,
            )
        else:
            self.train_loader, self.val_loader = build_dataloaders(
                manifest,
                split,
                fold=config.fold,
                batch_size=config.batch_size,
                samples_per_case=config.samples_per_case,
                num_workers=config.num_workers,
                limit=config.limit_cases,
                root=data_root,
                val_patches=True,
            )

        # Fixed monitoring subset inside fold-0 validation. Available only from
        # the prepared cache; the raw path keeps patch validation alone.
        self.monitoring_cases: list[str] = (
            select_monitoring_cases(manifest, split, config.fold, config.monitoring_negatives)
            if config.prepared_root
            else []
        )
        self.monitoring_fingerprint = cases_fingerprint(self.monitoring_cases)

        self.start_epoch = 0
        self.global_step = 0
        # Selection metric is class-28 Dice: HIGHER is better.
        self.best_metric = float("-inf")
        self.history: list[dict[str, float]] = []
        self.resumed = False
        self.selection_metric = float("-inf")
        self.selection_epoch: int | None = None

        # NOTE: the run seed is set at the TOP of __init__, before the model and
        # the loaders exist. `fit` must never reseed, or a resume would overwrite
        # the RNG state just restored from the checkpoint.

    # ------------------------------------------------------------------ #

    @staticmethod
    def _content_hash(payload: Any) -> str:
        """Content-addressed identity, independent of file path or machine."""
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def provenance(self) -> dict[str, Any]:
        """Config, data identity, and code identity for this run."""
        return {
            "config": asdict(self.config),
            "data_source": "prepared_cache" if self.config.prepared_root else "raw_nifti",
            "manifest_version": self.manifest["meta"]["version"],
            "manifest_sha256": self._content_hash(self.manifest),
            "split_version": self.split["meta"]["version"],
            "split_sha256": self._content_hash(self.split),
            "split_seed": self.split["meta"]["seed"],
            "fold": self.config.fold,
            "num_classes": NUM_CLASSES,
            "selection_metric_name": self.config.selection_metric_name,
            "selection_metric_value": self.best_metric,
            "selection_epoch": self.selection_epoch,
            "monitoring_subset_fingerprint": self.monitoring_fingerprint,
            "monitoring_cases": len(self.monitoring_cases),
            "learning_rate_current": self.scheduler.get_last_lr()[0],
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "device": str(self.device),
            "amp": self.use_amp,
        }

    def _forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        image = batch["image"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model(image)
            return self.criterion(logits, label)

    def train_epoch(self) -> float:
        self.model.train()
        accumulation = max(1, self.config.gradient_accumulation_steps)
        total = 0.0
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(self.train_loader):
            loss = self._forward_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {self.global_step}")

            self.scaler.scale(loss / accumulation).backward()

            if (index + 1) % accumulation == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total += float(loss.detach())
            steps += 1
            self.global_step += 1

            if self.config.max_steps_per_epoch and steps >= self.config.max_steps_per_epoch:
                break

        # flush a partial accumulation window
        if steps % accumulation != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return total / max(steps, 1)

    @torch.no_grad()
    def validate_volumes(self) -> dict[str, Any]:
        """Deterministic whole-volume validation on the fixed monitoring subset.

        This is the production model-selection signal. Unlike patch validation
        it involves no random cropping: the case list, the sliding-window grid
        and the prepared volumes are all fixed, so re-running it on an unchanged
        model returns exactly the same number.

        Reads the prepared cache, which is already in the training frame, so no
        source-geometry inversion is needed here - that belongs to inference on
        raw CT. Logits are accumulated on CPU and released per case; only one
        case is ever resident.
        """
        if not self.monitoring_cases:
            raise RuntimeError("whole-volume validation requires a prepared cache")

        self.model.eval()
        lesion_rows: list[dict[str, Any]] = []
        started = time.time()

        for case_id in self.monitoring_cases:
            image, label = read_prepared_case(case_path(self.config.prepared_root, case_id))
            volume = torch.from_numpy(image.astype(np.float32))[None, None]
            logits = predict_logits(
                self.model, volume, sw_batch_size=1, sw_device=self.device, device="cpu"
            )
            prediction = logits_to_labels(logits)[0, 0].numpy().astype(np.int16)
            row = lesion_case_metrics(prediction, label.astype(np.int16))
            row["case_id"] = case_id
            lesion_rows.append(row)
            del logits, volume, prediction, image, label

        summary = summarize_lesion_metrics(lesion_rows)
        summary["seconds"] = time.time() - started
        summary["cases"] = len(self.monitoring_cases)
        summary["subset_fingerprint"] = self.monitoring_fingerprint
        return summary

    @torch.no_grad()
    def validate(self) -> float:
        """Patch-level validation loss. DIAGNOSTIC ONLY - it never selects best.pt.

        The crop is random even with augmentation disabled, so this number moves
        between calls on an unchanged model. It is a cheap per-epoch signal that
        optimization is progressing, not a model-selection criterion.
        """
        self.model.eval()
        total = 0.0
        steps = 0
        for batch in self.val_loader:
            total += float(self._forward_loss(batch).detach())
            steps += 1
            if self.config.max_steps_per_epoch and steps >= self.config.max_steps_per_epoch:
                break
            if self.config.patch_validation_batches and steps >= self.config.patch_validation_batches:
                break
        return total / max(steps, 1)

    # ------------------------------------------------------------------ #

    def save(self, path: Path, best_metric: float, epoch: int) -> Path:
        """Write a checkpoint locally, then mirror it to persistent storage.

        Mirroring happens here, immediately after the local file is complete and
        fsynced, rather than in a separate step after training. On Colab the
        training process holds the cell for hours; if the runtime dies during
        it, anything not already on Drive is gone.
        """
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            epoch=epoch,
            global_step=self.global_step,
            best_metric=best_metric,
            config=self.provenance(),
            git_commit=git_commit(),
        )
        self._persist(path)
        return path

    def _persist(self, path: Path) -> None:
        """Copy one completed artifact to the persistent run directory, atomically."""
        if not self.persistent_dir:
            return
        try:
            self.persistent_dir.mkdir(parents=True, exist_ok=True)
            destination = self.persistent_dir / path.name
            staging = destination.with_suffix(destination.suffix + ".partial")
            shutil.copyfile(path, staging)
            os.replace(staging, destination)
        except OSError as error:
            # A Drive hiccup must not destroy an otherwise healthy run; the
            # local checkpoint is already safe.
            logger.error("could not persist %s to %s: %s", path.name, self.persistent_dir, error)

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, scheduler, scaler, counters, and RNG."""
        checkpoint = load_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
        )
        self.model.to(self.device)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_metric = float(checkpoint["best_metric"])
        self.resumed = True

        # Restore the selection state too, or a resumed run would forget which
        # score best.pt already holds and could overwrite a better checkpoint.
        stored = checkpoint.get("config", {})
        self.selection_metric = self.best_metric
        self.selection_epoch = stored.get("selection_epoch")
        stored_fingerprint = stored.get("monitoring_subset_fingerprint")
        if stored_fingerprint and stored_fingerprint != self.monitoring_fingerprint:
            raise ValueError(
                "Monitoring subset changed since this checkpoint was written "
                f"({stored_fingerprint} -> {self.monitoring_fingerprint}). The "
                "selection metric would not be comparable across the resume."
            )

        # The scheduler's own state carries the T_max it was built with, so
        # restoring it overrides the value derived from the current --epochs.
        # Resuming with a different epoch budget therefore replays the ORIGINAL
        # cosine schedule, which is rarely what was intended.
        restored_t_max = self.scheduler.state_dict().get("T_max")
        if restored_t_max is not None and restored_t_max != max(self.config.epochs, 1):
            logger.warning(
                "Scheduler was built for %d epochs but this run configures %d. "
                "The restored cosine schedule wins; resume with the original "
                "--epochs to continue the intended learning-rate curve.",
                restored_t_max,
                self.config.epochs,
            )
        logger.info(
            "Resumed from %s at epoch %d, step %d, best %.4f",
            path,
            self.start_epoch,
            self.global_step,
            self.best_metric,
        )
        return checkpoint

    def _should_select(self, epoch: int) -> bool:
        """Deterministic validation cadence: fixed epochs, never data-dependent."""
        if not self.monitoring_cases:
            return False
        cadence = max(1, self.config.validate_every_epochs)
        return (epoch + 1) % cadence == 0 or epoch == self.config.epochs - 1

    def _write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        self._persist(path)

    def fit(self) -> dict[str, Any]:
        """Train, persisting `latest.pt` each epoch and `best.pt` on a real improvement.

        Two validation signals with different jobs:

        * patch loss, every epoch - cheap, noisy, DIAGNOSTIC. Never selects.
        * whole-volume class-28 Dice on the fixed monitoring subset, every
          ``validate_every_epochs`` - deterministic, and the ONLY thing that
          writes ``best.pt``.
        """
        self._write_json("provenance.json", self.provenance())

        started = time.time()
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_started = time.time()
            # the rate actually applied during this epoch, captured before stepping
            epoch_learning_rate = self.scheduler.get_last_lr()[0]
            train_loss = self.train_epoch()
            patch_loss = self.validate()
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "patch_val_loss_diagnostic": patch_loss,
                "learning_rate": epoch_learning_rate,
                "seconds": time.time() - epoch_started,
            }
            logger.info(
                "epoch %d  train %.4f  patch-val %.4f (diagnostic)  lr %.2e  %.1fs",
                epoch, train_loss, patch_loss, epoch_learning_rate, record["seconds"],
            )

            if self._should_select(epoch):
                selection = self.validate_volumes()
                score = selection[self.config.selection_metric_name]
                record["selection"] = selection
                logger.info(
                    "epoch %d  class-28 Dice %.4f over %d lesion-positive cases; "
                    "FP-case-rate %.3f over %d negatives; %.0fs",
                    epoch, score,
                    selection["lesion_positive_cases"],
                    selection["false_positive_rate_on_negative_cases"],
                    selection["lesion_negative_cases"],
                    selection["seconds"],
                )
                if not math.isnan(score) and score > self.best_metric:
                    self.best_metric = float(score)
                    self.selection_metric = float(score)
                    self.selection_epoch = epoch
                    self.save(self.run_dir / "best.pt", self.best_metric, epoch)
                    logger.info("epoch %d  new best.pt (%s %.4f)",
                                epoch, self.config.selection_metric_name, score)

            self.history.append(record)

            if (epoch + 1) % max(1, self.config.save_every_epochs) == 0:
                self.save(self.run_dir / "latest.pt", self.best_metric, epoch)
            self._write_json("history.json", self.history)

        summary = {
            "experiment": self.config.experiment,
            "initialization": self.config.initialization,
            "epochs_completed": self.config.epochs - self.start_epoch,
            "selection_metric": self.config.selection_metric_name,
            "best_selection_metric": self.best_metric,
            "best_epoch": self.selection_epoch,
            "monitoring_subset_fingerprint": self.monitoring_fingerprint,
            "monitoring_cases": len(self.monitoring_cases),
            "global_step": self.global_step,
            "total_seconds": time.time() - started,
            "history": self.history,
            "peak_gpu_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if self.device.type == "cuda"
                else None
            ),
            "provenance": self.provenance(),
        }
        self._write_json("summary.json", summary)
        return summary
