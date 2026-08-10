"""PanTS filesystem layout and portable data-root resolution."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANTS_DATA_ROOT_ENV = "PANTS_DATA_ROOT"

SPLIT_DIRECTORIES = {
    "train": ("ImageTr", "LabelTr"),
    "test": ("ImageTe", "LabelTe"),
}


def get_data_root(root: str | Path | None = None) -> Path:
    """
    Resolve the PanTS data root.

    Resolution order:

    1. the explicit ``root`` argument;
    2. the ``PANTS_DATA_ROOT`` environment variable;
    3. the local fallback ``PROJECT_ROOT/PanTS/data``.

    The environment is read on every call so that a process which sets
    ``PANTS_DATA_ROOT`` after importing this module still sees the new value.
    """

    if root is not None:
        return Path(root).expanduser()

    environment_root = os.environ.get(PANTS_DATA_ROOT_ENV)
    if environment_root:
        return Path(environment_root).expanduser()

    return PROJECT_ROOT / "PanTS" / "data"


def get_split_roots(
    split: str,
    root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return image and label root directories for a split."""

    try:
        image_directory, label_directory = SPLIT_DIRECTORIES[split]
    except KeyError:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of {sorted(SPLIT_DIRECTORIES)}."
        ) from None

    data_root = get_data_root(root)

    return data_root / image_directory, data_root / label_directory


def get_case_paths(
    case_id: str,
    split: str = "train",
    root: str | Path | None = None,
) -> dict:
    """
    Return important PanTS paths for one case.
    """

    image_root, label_root = get_split_roots(split, root)

    image_case = image_root / case_id
    label_case = label_root / case_id

    return {
        "case_id": case_id,
        "ct": image_case / "ct.nii.gz",
        "combined": label_case / "combined_labels.nii.gz",
        "segmentations": label_case / "segmentations",
    }


def get_relative_case_paths(
    case_id: str,
    split: str = "train",
) -> dict[str, str]:
    """
    Return CT and combined-label paths relative to the data root.

    Manifests store these instead of absolute paths so that the same manifest
    is valid locally, in Colab, and on any other machine.
    """

    try:
        image_directory, label_directory = SPLIT_DIRECTORIES[split]
    except KeyError:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of {sorted(SPLIT_DIRECTORIES)}."
        ) from None

    return {
        "ct": f"{image_directory}/{case_id}/ct.nii.gz",
        "label": f"{label_directory}/{case_id}/combined_labels.nii.gz",
    }


def list_cases(
    split: str = "train",
    root: str | Path | None = None,
) -> list[str]:
    """Return sorted case identifiers."""

    image_root, _ = get_split_roots(split, root)

    if not image_root.is_dir():
        raise FileNotFoundError(
            f"PanTS image directory not found: {image_root}. "
            f"Set {PANTS_DATA_ROOT_ENV} to the directory containing "
            "ImageTr/ and LabelTr/."
        )

    return sorted(
        path.name
        for path in image_root.iterdir()
        if path.is_dir()
    )
