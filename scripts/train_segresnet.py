"""Train the PanTS SegResNet from random or SuPreM initialization.

The same command runs both arms; only --initialization changes.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import read_json  # noqa: E402
from src.training.trainer import SegResNetTrainer, TrainingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialization",
        choices=["random", "suprem"],
        required=True,
        help="the only scientific difference between the two arms",
    )
    parser.add_argument("--experiment", default=None, help="run name (default segresnet_<init>)")
    parser.add_argument(
        "--pretrained-checkpoint",
        default=os.environ.get("SUPREM_CHECKPOINT"),
        help="SuPreM checkpoint; defaults to $SUPREM_CHECKPOINT",
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "pants_tr_manifest.json")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "pants_cv_v1.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--limit-cases", type=int, default=None, help="smoke cohort size")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs/runs"))
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    parser.add_argument("--data-root", type=Path, default=None, help="override PANTS_DATA_ROOT")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.initialization == "suprem" and not args.pretrained_checkpoint:
        raise SystemExit(
            "suprem initialization needs --pretrained-checkpoint or $SUPREM_CHECKPOINT"
        )

    config = TrainingConfig(
        experiment=args.experiment or f"segresnet_{args.initialization}",
        initialization=args.initialization,
        pretrained_checkpoint=args.pretrained_checkpoint if args.initialization == "suprem" else None,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_case=args.samples_per_case,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_cases=args.limit_cases,
        max_steps_per_epoch=args.max_steps_per_epoch,
        save_every_epochs=args.save_every_epochs,
        amp=not args.no_amp,
        device=args.device,
        output_root=args.output_root,
    )

    trainer = SegResNetTrainer(
        config,
        manifest=read_json(args.manifest),
        split=read_json(args.split),
        data_root=args.data_root,
    )
    if args.resume:
        trainer.resume(args.resume)

    summary = trainer.fit()

    print("\n=== run summary ===")
    print(f"experiment      {summary['experiment']} ({summary['initialization']})")
    print(f"epochs          {summary['epochs_completed']}")
    print(f"best val loss   {summary['best_val_loss']:.4f}")
    print(f"global steps    {summary['global_step']}")
    print(f"runtime         {summary['total_seconds']:.1f}s")
    if summary["peak_gpu_gb"] is not None:
        print(f"peak GPU VRAM   {summary['peak_gpu_gb']:.2f} GB")
    print(f"artifacts       {Path(config.output_root) / config.experiment}")
    print(json.dumps(summary["history"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
