"""Create and validate a small nnU-Net v2 dataset from PanTS-tr."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import read_json, to_nnunet_splits, write_json  # noqa: E402
from src.data.nnunet import (  # noqa: E402
    NNUNET_PREPROCESSED,
    prepare_nnunet_production_dataset,
    prepare_nnunet_smoke_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a deterministic, symlink-based PanTS-tr nnU-Net v2 "
            "smoke dataset. No preprocessing or training is performed."
        )
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="build Dataset500_PanTS from all 9,000 manifest cases",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "pants_tr_manifest.json",
        help="authoritative case list, used with --production",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "pants_cv_v1.json",
        help="tracked CV split, emitted as the native splits_final.json",
    )
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--name", default="PanTSSmoke")
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the derived dataset directory if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.production:
        return main_production(args)
    try:
        report = prepare_nnunet_smoke_dataset(
            dataset_id=args.dataset_id,
            name=args.name,
            max_cases=args.max_cases,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    selection = report["selection"]
    validation = report["validation"]

    print("nnU-Net v2 PanTS smoke dataset prepared")
    print(f"  dataset:                    {report['dataset_dir']}")
    print(f"  candidate cases scanned:    {selection['cases_scanned']}")
    print(f"  last candidate scanned:     {selection['last_case_scanned']}")
    print(
        "  candidate positive/negative: "
        f"{selection['candidate_positive']}/"
        f"{selection['candidate_negative']}"
    )
    print(f"  coverage core:              {selection['coverage_core_ids']}")
    print(
        "  selected positive/negative:  "
        f"{selection['selected_positive']}/"
        f"{selection['selected_negative']}"
    )
    print(f"  covered label IDs:          {selection['covered_label_ids']}")
    print(f"  image/label links:          {validation['image_links']}/{validation['label_links']}")
    print(
        "  nnU-Net reader test:         "
        f"{validation['nnunet_reader']} on {validation['reader_test_case']}"
    )
    print(
        "  reader image/label shapes:   "
        f"{validation['reader_image_shape']}/"
        f"{validation['reader_label_shape']}"
    )
    print("  source split:                PanTS-tr only")
    print("  imagesTs created:            no")
    print("  raw PanTS files modified:    no")
    print("\nSelected cases")
    for case in report["cases"]:
        status = "positive" if case.lesion_positive else "negative"
        labels = ",".join(str(value) for value in sorted(case.label_ids))
        print(f"  {case.case_id}  {status:8s}  labels=[{labels}]")

    print("\nnnU-Net environment paths")
    for variable, path in report["nnunet_paths"].items():
        print(f'  export {variable}="{path}"')
    print("\nRaw dataset validation passed. Planning/preprocessing was not run.")


def main_production(args: argparse.Namespace) -> None:
    """Build Dataset500_PanTS and emit the tracked split in nnU-Net's format."""
    dataset_id = 500 if args.dataset_id == 501 else args.dataset_id
    name = "PanTS" if args.name == "PanTSSmoke" else args.name

    try:
        report = prepare_nnunet_production_dataset(
            manifest=read_json(args.manifest),
            dataset_id=dataset_id,
            name=name,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    validation = report["validation"]
    print("nnU-Net v2 PanTS production dataset prepared")
    print(f"  dataset:                    {report['dataset_dir']}")
    print(f"  cases:                      {len(report['cases'])}")
    print(f"  lesion-positive:            {report['lesion_positive']}")
    print(f"  image/label links:          {validation['image_links']}/{validation['label_links']}")
    print(f"  nnU-Net reader test:        {validation['nnunet_reader']}")
    print(f"  reader image/label shapes:  {validation['reader_image_shape']}/"
          f"{validation['reader_label_shape']}")
    print("  raw PanTS files modified:   no (symlinks only)")

    # The SAME fold assignment as the SegResNet arms. There is no second split
    # policy: this is a format conversion, not a new partition.
    split = read_json(args.split)
    destination = NNUNET_PREPROCESSED / Path(report["dataset_dir"]).name
    destination.mkdir(parents=True, exist_ok=True)
    write_json(to_nnunet_splits(split), destination / "splits_final.json")
    print(f"\n  splits_final.json ->        {destination / 'splits_final.json'}")
    print(f"  folds:                      {len(split['folds'])} (seed {split['meta']['seed']})")

    print("\nnnU-Net environment paths")
    for variable, path in report["nnunet_paths"].items():
        print(f'  export {variable}="{path}"')
    print("\nRaw dataset validation passed. Planning/preprocessing was not run.")


if __name__ == "__main__":
    main()
