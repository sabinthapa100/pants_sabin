"""Merge the surviving per-epoch history segments of a run into one record.

A resume bug in the historical training code replaced ``history.json`` with only
the post-resume segment, so a completed run can leave several partial files. This
merges whatever segments exist and reports exactly what is still missing.

    python scripts/recover_history.py \
        --segment PanTS_run/.../history_until38.json \
        --segment PanTS_run/.../history.json \
        --epochs 64 \
        --output evaluation/random_history/

Nothing is interpolated, estimated, or inferred. Epochs with no surviving record
stay absent, and identical duplicates are collapsed only after being confirmed
identical - a genuine conflict is a hard stop, because two different values for
the same epoch means one of the files does not belong to this run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--segment", type=Path, action="append", required=True,
                        help="a history JSON file; repeat for each segment")
    parser.add_argument("--epochs", type=int, required=True,
                        help="configured epoch count, to report coverage against")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_segment(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise SystemExit(f"{path}: expected a list of epoch records")
    for record in records:
        if "epoch" not in record:
            raise SystemExit(f"{path}: a record has no 'epoch' field")
    return records


def merge(segments: dict[str, list[dict[str, Any]]]) -> tuple[dict[int, dict], list[dict]]:
    """Return epoch -> record, plus the list of conflicts found."""
    merged: dict[int, dict[str, Any]] = {}
    origin: dict[int, str] = {}
    conflicts: list[dict[str, Any]] = []

    for name, records in segments.items():
        for record in records:
            epoch = int(record["epoch"])
            if epoch not in merged:
                merged[epoch] = record
                origin[epoch] = name
                continue
            if merged[epoch] == record:
                continue                      # identical duplicate: harmless
            conflicts.append({
                "epoch": epoch,
                "first_seen_in": origin[epoch],
                "conflicting_file": name,
                "first_value": merged[epoch],
                "conflicting_value": record,
            })
    return merged, conflicts


def runs_of(missing: list[int]) -> list[str]:
    """Compress [24,25,26,40] into ['24-26', '40'] so gaps read at a glance."""
    spans: list[str] = []
    for epoch in sorted(missing):
        if spans and epoch == int(spans[-1].split("-")[-1]) + 1:
            start = spans[-1].split("-")[0]
            spans[-1] = f"{start}-{epoch}"
        else:
            spans.append(str(epoch))
    return spans


def main() -> int:
    args = parse_args()
    segments = {str(path): load_segment(path) for path in args.segment}
    merged, conflicts = merge(segments)

    if conflicts:
        print(f"CONFLICT: {len(conflicts)} epoch(s) have different values in different files.")
        for entry in conflicts[:5]:
            print(f"  epoch {entry['epoch']}: {entry['first_seen_in']} vs {entry['conflicting_file']}")
        print("Refusing to merge. One of these files does not belong to this run.")
        return 1

    expected = list(range(args.epochs))
    missing = [epoch for epoch in expected if epoch not in merged]
    records = [merged[epoch] for epoch in sorted(merged)]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recovered_history.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    selection_epochs = sorted(int(r["epoch"]) for r in records if "selection" in r)
    report = {
        "note": (
            "Merged from surviving segments. No value was interpolated, estimated "
            "or inferred; missing epochs are absent, not imputed."
        ),
        "segments_used": {
            name: {
                "records": len(records_in),
                "epochs": [int(r["epoch"]) for r in records_in],
            }
            for name, records_in in segments.items()
        },
        "configured_epochs": args.epochs,
        "epochs_recovered": len(records),
        "coverage_fraction": len(records) / args.epochs,
        "epochs_present": sorted(merged),
        "epochs_missing": missing,
        "missing_spans": runs_of(missing),
        "selection_epochs_recovered": selection_epochs,
        "duplicate_records_agreeing": sum(len(r) for r in segments.values()) - len(records),
        "conflicts": conflicts,
    }
    (args.output / "history_recovery_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"recovered {len(records)}/{args.epochs} epochs "
          f"({100 * len(records) / args.epochs:.1f}%)")
    print(f"missing: {', '.join(runs_of(missing)) if missing else 'none'}")
    print(f"selection epochs: {selection_epochs}")
    print(f"wrote {args.output / 'recovered_history.json'}")
    print(f"wrote {args.output / 'history_recovery_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
