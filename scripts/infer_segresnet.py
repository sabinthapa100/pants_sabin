"""Segment an unseen CT with a trained PanTS SegResNet.

Needs only a trained checkpoint and a NIfTI CT. It never reads labels, a
manifest, a split, the prepared training cache, or the SuPreM initialization
checkpoint, and it does not care which initialization the model started from -
after training those are simply learned PanTS weights.

    python scripts/infer_segresnet.py --input ct.nii.gz --output out/ --checkpoint best.pt
"""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.labels import PANCREATIC_LESION  # noqa: E402
from src.evaluation.inference import predict_case_in_source_geometry  # noqa: E402
from src.evaluation.postprocessing import LESION_PEAK_PROBABILITY  # noqa: E402
from src.models.segresnet import build_segresnet  # noqa: E402
from src.training.checkpoint import load_training_checkpoint  # noqa: E402


logger = logging.getLogger(__name__)

LABEL_FILENAME = "combined_labels.nii.gz"
PROBABILITY_FILENAME = "pancreatic_lesion_probability.nii.gz"
CT_SUFFIXES = (".nii.gz", ".nii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment unseen CT volumes with a trained PanTS SegResNet."
    )
    parser.add_argument("--input", type=Path, required=True, help="a CT NIfTI, or a directory of them")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained PanTS checkpoint")
    parser.add_argument(
        "--lesion-probability",
        action="store_true",
        help=f"also write {PROBABILITY_FILENAME} (class-28 softmax)",
    )
    parser.add_argument(
        "--lesion-peak-probability",
        type=float,
        default=LESION_PEAK_PROBABILITY,
        metavar="P",
        help=(
            "keep a class-28 component only if its peak softmax probability is >= P "
            f"(default {LESION_PEAK_PROBABILITY}, the frozen development rule). "
            "Pass a negative value to disable filtering and emit the raw argmax."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap")
    parser.add_argument("--sw-batch-size", type=int, default=1, help="windows evaluated at once")
    parser.add_argument(
        "--accumulate-device",
        default="cpu",
        help="where full-volume logits are stitched; cpu keeps VRAM flat",
    )
    return parser.parse_args()


def discover_inputs(path: Path) -> list[Path]:
    """One file, or every CT in a directory. Filenames carry no meaning here."""
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.name.endswith(CT_SUFFIXES)
        )
        if not found:
            raise SystemExit(f"No NIfTI volumes found in {path}")
        return found
    raise SystemExit(f"Input not found: {path}")


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Build the 29-class architecture and load trained PanTS weights into it.

    ``initialization="random"`` here only means "do not read a SuPreM file";
    every weight is immediately overwritten by the checkpoint.
    """
    model = build_segresnet(initialization="random")
    checkpoint = load_training_checkpoint(checkpoint_path, model=model)
    model.to(device).eval()

    provenance = checkpoint.get("config", {})
    logger.info(
        "checkpoint epoch %s, %s %s, trained from %s initialization at commit %s",
        checkpoint.get("epoch"),
        provenance.get("selection_metric_name", "metric"),
        provenance.get("selection_metric_value"),
        provenance.get("config", {}).get("initialization", "unknown"),
        (checkpoint.get("git_commit") or "unknown")[:12],
    )
    return model


def _as_nifti(array: np.ndarray, source: nib.Nifti1Image, dtype) -> nib.Nifti1Image:
    """Write `array` on the source grid, with an honest on-disk dtype.

    The source header is reused so voxel sizes, units and the qform/sform codes
    survive, but its data dtype describes the CT (int16) and would silently cast
    a uint8 label map or a float probability on save. It is overridden here, and
    the intensity scaling is cleared so nibabel does not re-apply the CT's
    slope/intercept to our values.
    """
    header = source.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(None, None)
    return nib.Nifti1Image(array, source.affine, header)


def segment_case(model: torch.nn.Module, ct_path: Path, destination: Path, args) -> dict:
    """Preprocess, sliding-window infer, invert to the source grid, save."""
    source = nib.load(str(ct_path))
    result = predict_case_in_source_geometry(
        model,
        str(ct_path),
        overlap=args.overlap,
        sw_batch_size=args.sw_batch_size,
        sw_device=args.device,
        accumulate_device=args.accumulate_device,
        want_lesion_probability=args.lesion_probability,
        min_lesion_peak_probability=(
            args.lesion_peak_probability if args.lesion_peak_probability >= 0 else None
        ),
    )

    labels = np.asarray(result["labels"].detach().cpu())[0]
    if labels.shape != source.shape:
        raise RuntimeError(
            f"{ct_path.name}: prediction {labels.shape} does not match source {source.shape}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    nib.save(_as_nifti(labels.astype(np.uint8), source, np.uint8), str(destination / LABEL_FILENAME))

    written = [LABEL_FILENAME]
    if args.lesion_probability:
        probability = np.asarray(result["lesion_probability"].detach().cpu())[0]
        nib.save(
            _as_nifti(probability.astype(np.float32), source, np.float32),
            str(destination / PROBABILITY_FILENAME),
        )
        written.append(PROBABILITY_FILENAME)

    return {
        "shape": tuple(int(v) for v in labels.shape),
        "classes": sorted(int(v) for v in np.unique(labels)),
        "lesion_voxels": int((labels == PANCREATIC_LESION).sum()),
        "written": written,
        "lesion_filter": result.get("lesion_filter"),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    volumes = discover_inputs(args.input)
    model = load_model(args.checkpoint, args.device)
    print(f"segmenting {len(volumes)} volume(s) on {args.device}\n")

    for ct_path in volumes:
        stem = ct_path.name
        for suffix in CT_SUFFIXES:
            stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
        destination = args.output if len(volumes) == 1 else args.output / stem

        report = segment_case(model, ct_path, destination, args)
        print(f"  {ct_path.name}")
        print(f"    shape           {report['shape']}")
        print(f"    classes present {report['classes']}")
        print(f"    class-28 voxels {report['lesion_voxels']}")
        if report["lesion_filter"]:
            audit = report["lesion_filter"]
            print(f"    lesion filter   peak softmax >= {audit['rule']['min_peak_probability']}, "
                  f"{audit['rule']['connectivity']}-connectivity: "
                  f"{audit['components_retained']} kept, {audit['components_rejected']} rejected, "
                  f"{audit['relabelled_voxels']} voxels relabelled")
        print(f"    -> {destination}/{', '.join(report['written'])}")

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"\npeak GPU VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
