"""Command-line entry point for inspecting one PanTS case."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspect import format_inspection, inspect_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one PanTS CT and its pancreas-related labels."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_case(case_id=args.case_id, split=args.split)
    print(format_inspection(report))


if __name__ == "__main__":
    main()
