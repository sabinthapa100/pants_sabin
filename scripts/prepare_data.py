"""Build the PanTS-tr manifest and the fixed development split."""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    build_manifest,
    build_split,
    read_json,
    to_nnunet_splits,
    write_json,
)
from src.data.paths import get_data_root  # noqa: E402


# The manifest is DERIVED: exactly reproducible from the raw data by this
# script, ~4 MB, and therefore gitignored rather than tracked.
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "pants_tr_manifest.json"

# The split is SOURCE TRUTH for the experiment: it defines which cases train
# and which validate, so it is small, tracked, and never regenerated casually.
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "pants_cv_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan PanTS-tr into an authoritative manifest and derive the fixed "
            "5-fold development split. PanTS-te is never read."
        )
    )
    parser.add_argument("--manifest", action="store_true", help="build the manifest")
    parser.add_argument("--split", action="store_true", help="build the split")
    parser.add_argument(
        "--emit-nnunet-splits",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the same assignment as an nnU-Net splits_final.json",
    )
    parser.add_argument("--root", type=Path, default=None, help="override PANTS_DATA_ROOT")
    parser.add_argument("--workers", type=int, default=8, help="parallel scan workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not (args.manifest or args.split or args.emit_nnunet_splits):
        print("Nothing to do. Pass --manifest, --split, and/or --emit-nnunet-splits.")
        return 1

    print(f"PanTS data root: {get_data_root(args.root)}")

    if args.manifest:
        manifest = build_manifest(root=args.root, workers=args.workers)
        write_json(manifest, args.manifest_path)
        meta = manifest["meta"]
        print(f"\nManifest -> {args.manifest_path}")
        print(f"  cases                    {meta['case_count']}")
        print(f"  lesion-positive          {meta['lesion_positive']}")
        print(f"  lesion-negative          {meta['lesion_negative']}")
        print(f"  metadata flag disagrees  {meta['metadata_tumor_flag_disagreements']}")

    if args.split:
        manifest = read_json(args.manifest_path)
        split = build_split(manifest, seed=args.seed, folds=args.folds)
        write_json(split, args.split_path)
        print(f"\nSplit -> {args.split_path}  (seed {args.seed}, {args.folds} folds)")
        for row in split["meta"]["fold_summary"]:
            print(
                f"  fold {row['fold']}: train={row['train']:5d} val={row['val']:5d} "
                f"val lesion-positive={row['val_lesion_positive']:4d}"
            )

    if args.emit_nnunet_splits:
        split = read_json(args.split_path)
        write_json(to_nnunet_splits(split), args.emit_nnunet_splits)
        print(f"\nnnU-Net splits_final.json -> {args.emit_nnunet_splits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
