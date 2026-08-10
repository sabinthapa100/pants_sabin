"""One minimal SegResNet trainer, shared by both initialization arms.

The same class trains ``initialization="random"`` and ``initialization="suprem"``.
Nothing in this file branches on the initialization beyond constructing the
model, which is exactly what makes the pretraining comparison controlled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from monai.losses import DiceCELoss

from ..data.transforms import PREPROCESSING, build_dataloaders
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
    save_every_epochs: int = 1
    amp: bool = True
    device: str = "cuda"
    output_root: str = "outputs/runs"
    preprocessing: dict[str, Any] = field(default_factory=lambda: dict(PREPROCESSING))


def git_commit() -> str | None:
    """Current commit, for run provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


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

        self.run_dir = Path(config.output_root) / config.experiment
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.history: list[dict[str, float]] = []
        self.resumed = False

        # Seed the fresh run here. `fit` must NOT reseed, or a resume would
        # overwrite the RNG state just restored from the checkpoint.
        torch.manual_seed(config.seed)

    # ------------------------------------------------------------------ #

    def provenance(self) -> dict[str, Any]:
        """Config, data identity, and code identity for this run."""
        return {
            "config": asdict(self.config),
            "manifest_version": self.manifest["meta"]["version"],
            "split_version": self.split["meta"]["version"],
            "split_seed": self.split["meta"]["seed"],
            "fold": self.config.fold,
            "num_classes": NUM_CLASSES,
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
    def validate(self) -> float:
        """Patch-level validation loss (whole-volume inference is separate)."""
        self.model.eval()
        total = 0.0
        steps = 0
        for batch in self.val_loader:
            total += float(self._forward_loss(batch).detach())
            steps += 1
            if self.config.max_steps_per_epoch and steps >= self.config.max_steps_per_epoch:
                break
        return total / max(steps, 1)

    # ------------------------------------------------------------------ #

    def save(self, path: Path, best_metric: float, epoch: int) -> Path:
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
        return path

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

    def fit(self) -> dict[str, Any]:
        """Run training, saving `latest.pt` periodically and `best.pt` on improvement."""
        (self.run_dir / "provenance.json").write_text(
            json.dumps(self.provenance(), indent=2, default=str) + "\n", encoding="utf-8"
        )

        started = time.time()
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_started = time.time()
            # the rate actually applied during this epoch, captured before stepping
            epoch_learning_rate = self.scheduler.get_last_lr()[0]
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": epoch_learning_rate,
                "seconds": time.time() - epoch_started,
            }
            self.history.append(record)
            logger.info(
                "epoch %d  train %.4f  val %.4f  lr %.2e  %.1fs",
                epoch,
                train_loss,
                val_loss,
                record["learning_rate"],
                record["seconds"],
            )

            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.save(self.run_dir / "best.pt", self.best_metric, epoch)

            if (epoch + 1) % max(1, self.config.save_every_epochs) == 0:
                self.save(self.run_dir / "latest.pt", self.best_metric, epoch)

        summary = {
            "experiment": self.config.experiment,
            "initialization": self.config.initialization,
            "epochs_completed": self.config.epochs - self.start_epoch,
            "best_val_loss": self.best_metric,
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
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return summary
