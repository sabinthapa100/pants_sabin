"""Command-line entry point for PanTS visualization quality control."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import NAME_TO_CLASS  
from src.data.visualize import visualize_case 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mask-guided orthogonal QC views for one PanTS case."
    )
    parser.add_argument(
        "--case",
        required=True,
        dest="case_id",
        help="PanTS case identifier, for example PanTS_00000001.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split containing the case (default: train).",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=sorted(NAME_TO_CLASS),
        default=None,
        help=(
            "Standalone PanTS structures to overlay. Defaults to pancreas "
            "anatomy for negative cases and pancreas+lesion for positive cases."
        ),
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=-150.0,
        help="Lower CT display-window bound only (default: -150).",
    )
    parser.add_argument(
        "--window-max",
        type=float,
        default=250.0,
        help="Upper CT display-window bound only (default: 250).",
    )
    args = parser.parse_args()
    if args.window_min >= args.window_max:
        parser.error("--window-min must be less than --window-max")
    return args


def main() -> None:
    args = parse_args()
    result = visualize_case(
        case_id=args.case_id,
        split=args.split,
        structures=args.structures,
        window_min=args.window_min,
        window_max=args.window_max,
        prediction=None,
    )

    status = "positive" if result["lesion_positive"] else "negative"
    print("PanTS visualization complete")
    print(f"  case ID:                   {result['case_id']}")
    print(f"  split:                     {result['split']}")
    print(f"  lesion status:             {status}")
    print(f"  lesion voxels:             {result['lesion_voxels']}")
    print(f"  original orientation:      {result['original_orientation']}")
    print(f"  visualization orientation: {result['visualization_orientation']}")
    print(f"  canonical shape:           {result['canonical_shape']}")
    print(f"  canonical spacing (mm):    {result['canonical_spacing_mm']}")
    print(f"  slice-selection mask:      {result['selection_structure']}")
    print(f"  sagittal slice:            {result['selected_slices']['sagittal']}")
    print(f"  coronal slice:             {result['selected_slices']['coronal']}")
    print(f"  axial slice:               {result['selected_slices']['axial']}")
    print(f"  requested structures:      {result['requested_structures']}")
    print(f"  empty requested masks:     {result['empty_requested_structures']}")
    print(f"  display window:            {result['display_window']}")
    print(f"  geometry/alignment:        {result['geometry_alignment']}")
    print(f"  output figure:             {result['output_path']}")

if __name__ == "__main__":
    main()
