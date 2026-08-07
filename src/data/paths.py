from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANTS_DATA = PROJECT_ROOT / "PanTS" / "data"

IMAGE_TR = PANTS_DATA / "ImageTr"
LABEL_TR = PANTS_DATA / "LabelTr"

IMAGE_TE = PANTS_DATA / "ImageTe"
LABEL_TE = PANTS_DATA / "LabelTe"


def get_split_roots(split: str):
    """Return image and label root directories for a split."""

    if split == "train":
        return IMAGE_TR, LABEL_TR

    if split == "test":
        return IMAGE_TE, LABEL_TE

    raise ValueError(
        f"Unknown split '{split}'. "
        "Expected 'train' or 'test'."
    )


def get_case_paths(
    case_id: str,
    split: str = "train",
) -> dict:
    """
    Return important PanTS paths for one case.
    """

    image_root, label_root = get_split_roots(split)

    image_case = image_root / case_id
    label_case = label_root / case_id

    return {
        "case_id": case_id,
        "ct": image_case / "ct.nii.gz",
        "combined": label_case / "combined_labels.nii.gz",
        "segmentations": label_case / "segmentations",
    }


def list_cases(split: str = "train") -> list[str]:
    """Return sorted case identifiers."""

    image_root, _ = get_split_roots(split)

    return sorted(
        path.name
        for path in image_root.iterdir()
        if path.is_dir()
    )
