"""Build the PanTS-tr manifest and the fixed development split."""

import argparse
import logging
import shutil
import sys
import time
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
from src.training.trainer import git_commit  # noqa: E402
from src.data.prepared import (  # noqa: E402
    prepare_dataset,
    select_pilot_cases,
    verify_dataset,
    write_preprocessing_metadata,
)


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
    parser.add_argument(
        "--prepare-segresnet",
        action="store_true",
        help="build the deterministic npz training cache (resumable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="prepared-cache root, OUTSIDE the repository",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="prepare only a deterministic N-case representative cohort",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild cases that already exist"
    )
    parser.add_argument(
        "--deep-verify",
        action="store_true",
        help="decompress and revalidate every voxel of existing cases",
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

    if not (args.manifest or args.split or args.emit_nnunet_splits or args.prepare_segresnet):
        print(
            "Nothing to do. Pass --manifest, --split, --emit-nnunet-splits "
            "and/or --prepare-segresnet."
        )
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

    if args.prepare_segresnet:
        if args.output is None:
            print("--prepare-segresnet requires --output <dir> outside the repository.")
            return 1
        return prepare_segresnet_cache(args)

    return 0


def prepare_segresnet_cache(args: argparse.Namespace) -> int:
    """Build (or resume) the deterministic npz cache, then verify completeness."""
    manifest = read_json(args.manifest_path)

    if args.pilot:
        selection = select_pilot_cases(manifest, count=args.pilot)
        case_ids = selection.case_ids
        print(f"\nPilot cohort ({len(case_ids)} cases), deterministic:")
        for key, value in selection.rationale.items():
            print(f"  {key:24s} {value}")
    else:
        case_ids = [case["case_id"] for case in manifest["cases"]]
        print(f"\nPreparing all {len(case_ids)} PanTS-tr cases")

    already = len(case_ids) - len(verify_dataset(case_ids, args.output, deep=args.deep_verify))
    print(f"  output          {args.output}")
    print(f"  already present {already}")
    print(f"  workers         {args.workers}")

    started = time.time()
    results = prepare_dataset(
        case_ids, args.output, root=args.root, workers=args.workers, overwrite=args.overwrite
    )
    elapsed = time.time() - started

    written = [r for r in results if r["status"] == "written"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    total_bytes = sum(r.get("bytes", 0) for r in results)

    print(f"\nWritten {len(written)}, skipped {len(skipped)}, failed {len(failed)}")
    print(f"  elapsed         {elapsed:.1f}s")
    if written:
        print(f"  seconds/case    {elapsed / len(written):.2f}")
    print(f"  cache bytes     {total_bytes / 1e9:.2f} GB")
    print(f"  mean MB/case    {total_bytes / max(len(results), 1) / 1e6:.2f}")

    for failure in failed[:10]:
        print(f"  FAILED {failure['case_id']}: {failure['error']}")

    missing = verify_dataset(case_ids, args.output)
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} case(s) missing, e.g. {missing[:5]}")
        print("Re-run the same command; completed cases are skipped.")
        return 1

    metadata_path = write_preprocessing_metadata(
        args.output,
        manifest_path=args.manifest_path,
        split_path=args.split_path,
        case_count=len(case_ids),
        git_commit=git_commit(),
    )
    # The cache travels with the manifest so a Colab session needs nothing from
    # the raw dataset. The split travels with the repository, which is why it
    # is tracked and this is not.
    shutil.copyfile(args.manifest_path, Path(args.output) / "manifest.json")

    print(f"\nComplete: {len(case_ids)}/{len(case_ids)} cases verified present")
    print(f"Metadata -> {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
