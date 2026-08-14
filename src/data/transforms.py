"""Shared PanTS preprocessing, augmentation, and tumor-aware patch sampling.

Both SegResNet experiments use this module unchanged. Preprocessing is part of
the controlled comparison: if the random-initialization and SuPreM arms saw
different inputs, the measured effect of pretraining would be confounded.

Design decisions and their evidence are documented in
``docs``/the milestone plan; the short version:

* **1.5 mm isotropic** target spacing. PanTS-tr z-spacing spans 0.4-10 mm and
  only 18% of cases are isotropic, so training on native grids would make a
  fixed-size patch cover wildly different anatomy per case.
* **96^3 patches**, which at 1.5 mm cover 144 mm per side - enough to contain
  the whole pancreas and the peripancreatic vessels in a single window.
* **[-175, 250] HU -> [0, 1]**, the exact window the SuPreM checkpoint was
  pretrained with, so transfer is tested on its own terms.

The chain is deliberately split into three named stages:

``deterministic_transforms``
    a pure function of the case. Identical for training, validation and
    inference, and therefore the part that can be computed once and cached
    (see ``prepared.py``).

``sampling_transforms`` / ``augmentation_transforms``
    stochastic. These must run at training time on every epoch, or the model
    would see one frozen realization of the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from .labels import PANCREAS_FAMILY, PANCREATIC_DUCT, PANCREATIC_LESION
from .paths import get_data_root


TARGET_SPACING = (1.5, 1.5, 1.5)
PATCH_SIZE = (96, 96, 96)
ORIENTATION = "RAS"

# The SuPreM *supervised-pretraining* input domain: [-175, 250] HU -> [0, 1].
#
# An earlier revision used [-100, 200], which comes from a later JHH pancreatic
# downstream experiment rather than from the pipeline that produced the
# checkpoint we initialize from. Matching the pretraining domain is what makes
# the transfer comparison fair; the wider window also retains more contrast for
# the bone and lung context classes.
HU_MIN = -175.0
HU_MAX = 250.0

SAMPLING_KEY = "sampling_map"

# Sampling classes. Deliberately *not* generic foreground: PanTS foreground
# includes femur, lung and spleen, so a foreground-centred crop would usually
# contain no pancreas at all.
SAMPLING_OTHER = 0
SAMPLING_PANCREAS = 1
SAMPLING_LESION = 2
SAMPLING_NUM_CLASSES = 3

# Pancreas parenchyma plus duct, excluding the lesion itself.
PANCREAS_SAMPLING_CLASSES = (*PANCREAS_FAMILY, PANCREATIC_DUCT)

# On a lesion-positive case this yields 50% tumor-centred, 33% pancreas,
# 17% general anatomy. On a lesion-negative case MONAI automatically zeroes the
# absent lesion ratio and renormalizes, giving 67% pancreas / 33% general.
SAMPLING_RATIOS = (1, 2, 3)

PREPROCESSING = {
    "orientation": ORIENTATION,
    "target_spacing_mm": TARGET_SPACING,
    "patch_size": PATCH_SIZE,
    "patch_field_of_view_mm": tuple(s * p for s, p in zip(TARGET_SPACING, PATCH_SIZE, strict=True)),
    "intensity_window_hu": (HU_MIN, HU_MAX),
    "intensity_range": (0.0, 1.0),
    "sampling_ratios_other_pancreas_lesion": SAMPLING_RATIOS,
    "spatial_flips": False,
    "spatial_flip_rationale": (
        "PanTS has laterality-paired classes (adrenal 1/2, femur 9/10, "
        "kidney 12/13, lung 15/16). A left-right flip would leave a mirrored "
        "left kidney labelled kidney_left, which is inconsistent supervision."
    ),
}


class AlignLabelGeometryd(MapTransform):
    """
    Force the label to share the CT's world-coordinate affine.

    80 of the 9,000 PanTS-tr labels carry a degenerate affine while their voxel
    array is index-aligned with the CT. Without this repair ``Orientationd``
    would reorient image and label by *different* affines and silently mirror
    the annotation, training on left/right-flipped anatomy with no error raised.

    A no-op for the other 8,920 cases, so it is applied unconditionally rather
    than from a per-case flag. Evidence in TECHNICAL_NOTES.md section 4.
    """

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=[label_key], allow_missing_keys=allow_missing_keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        updated = dict(data)
        image = updated[self.image_key]
        label = updated[self.label_key]

        image_affine = getattr(image, "affine", None)
        if image_affine is None or not hasattr(label, "affine"):
            return updated

        if tuple(image.shape) != tuple(label.shape):
            raise ValueError(
                f"CT and label voxel grids differ: {tuple(image.shape)} vs "
                f"{tuple(label.shape)}. Index alignment cannot be assumed."
            )

        label.affine = image_affine.clone()
        return updated


class DeriveSamplingMapd(MapTransform):
    """
    Build the three-class sampling map used to place training patches.

    Input:  ``label`` as ``[1, D, H, W]`` with integer values 0...28.
    Output: ``sampling_map`` of the same shape with values

    ``0`` everything else, ``1`` pancreas family and duct (lesion removed),
    ``2`` pancreatic lesion (class 28).

    The map exists only to choose crop centres and is discarded before the
    batch is collated; it is never a training target.
    """

    def __init__(
        self,
        label_key: str = "label",
        sampling_key: str = SAMPLING_KEY,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=[label_key], allow_missing_keys=allow_missing_keys)
        self.label_key = label_key
        self.sampling_key = sampling_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        updated = dict(data)
        label = updated[self.label_key]

        values = label.as_tensor() if hasattr(label, "as_tensor") else torch.as_tensor(label)
        values = values.long()

        sampling = torch.zeros_like(values, dtype=torch.uint8)
        pancreas = torch.isin(values, torch.tensor(PANCREAS_SAMPLING_CLASSES, dtype=values.dtype))
        sampling[pancreas] = SAMPLING_PANCREAS
        sampling[values == PANCREATIC_LESION] = SAMPLING_LESION

        updated[self.sampling_key] = sampling
        return updated


def deterministic_transforms(
    target_spacing: Sequence[float] = TARGET_SPACING,
    with_label: bool = True,
) -> list:
    """
    The preprocessing shared byte-for-byte by training, validation and inference.

    ``Orientationd`` reorients: it permutes and flips voxel axes so they align
    with Right/Anterior/Superior. This is pure index bookkeeping and loses no
    information.

    ``Spacingd`` resamples: it changes the physical sampling lattice and must
    interpolate. Labels use nearest-neighbour because linear interpolation
    would invent fractional class identifiers such as 17.4, which correspond to
    no anatomical structure.

    ``with_label=False`` yields the IMAGE-ONLY chain used to segment an unseen
    CT. The intensity rules are the same objects, not a copy: the only
    difference is that no label key participates, so inference never needs an
    annotation, a case identifier, or a fabricated placeholder label.
    """

    keys = ["image", "label"] if with_label else ["image"]

    chain: list = [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
        EnsureTyped(keys=keys, track_meta=True),
    ]
    if with_label:
        # Meaningful only when a label exists: it repairs the label's affine
        # against the CT's before either is reoriented.
        chain.append(AlignLabelGeometryd())

    chain += [
        Orientationd(keys=keys, axcodes=ORIENTATION),
        Spacingd(
            keys=keys,
            pixdim=tuple(float(value) for value in target_spacing),
            mode=("bilinear", "nearest") if with_label else ("bilinear",),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN,
            a_max=HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    return chain


def inference_transforms(target_spacing: Sequence[float] = TARGET_SPACING) -> Compose:
    """Image-only deterministic preprocessing for an unseen CT.

    Input: ``{"image": "/any/path/scan.nii.gz"}``. No label, no manifest, no
    split, no prepared cache, no case identifier. Output: ``image`` as a
    ``[1, D, H, W]`` MetaTensor in [0, 1] carrying the applied-operation stack
    that ``Invertd`` later replays to return predictions to the source grid.
    """
    return Compose(
        [
            *deterministic_transforms(target_spacing, with_label=False),
            EnsureTyped(keys=["image"]),
        ]
    )


def sampling_transforms(
    patch_size: Sequence[int] = PATCH_SIZE,
    samples_per_case: int = 2,
) -> list:
    """
    The stochastic tumor-aware cropping stage.

    Input: whole ``image``/``label`` volumes already in the canonical training
    frame - RAS, 1.5 mm, image scaled to [0, 1]. It does not matter whether
    they arrived from raw NIfTI or from the prepared cache; this stage never
    reorients, resamples or renormalizes.

    Output: ``samples_per_case`` patches of ``[1, 96, 96, 96]``.

    ``SpatialPadd`` runs first because 6% of PanTS-tr volumes are thinner than
    96 voxels on some axis after resampling and could not otherwise host a
    patch.
    """

    patch = tuple(int(value) for value in patch_size)

    return [
        DeriveSamplingMapd(),
        SpatialPadd(keys=["image", "label", SAMPLING_KEY], spatial_size=patch),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key=SAMPLING_KEY,
            num_classes=SAMPLING_NUM_CLASSES,
            ratios=list(SAMPLING_RATIOS),
            spatial_size=patch,
            num_samples=int(samples_per_case),
            warn=False,
        ),
        DeleteItemsd(keys=[SAMPLING_KEY]),
    ]


def augmentation_transforms() -> list:
    """
    Intensity-only augmentation.

    Spatial flips and 90-degree rotations are deliberately omitted: PanTS has
    laterality-paired classes, so a mirrored left kidney would still carry the
    ``kidney_left`` label. See ``PREPROCESSING``.
    """

    return [
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
        ),
    ]


def train_transforms(
    patch_size: Sequence[int] = PATCH_SIZE,
    target_spacing: Sequence[float] = TARGET_SPACING,
    samples_per_case: int = 2,
    augment: bool = True,
) -> Compose:
    """
    Training pipeline from **raw NIfTI**: deterministic core, crop, augment.

    Emits ``samples_per_case`` patches of ``image``/``label`` ``[1, 96, 96, 96]``.
    ``augment=False`` drops only the stochastic intensity stage, keeping the same
    deterministic core and the same tumor-aware crop; it is used for patch
    validation and for the tiny-overfit test.

    ``prepared.prepared_train_transforms`` is the cache-fed equivalent and reuses
    the two stochastic stages verbatim.
    """

    return Compose(
        [
            *deterministic_transforms(target_spacing),
            *sampling_transforms(patch_size, samples_per_case),
            *(augmentation_transforms() if augment else []),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def val_transforms(target_spacing: Sequence[float] = TARGET_SPACING) -> Compose:
    """
    Validation pipeline: the deterministic core only, at full volume.

    No cropping and no augmentation. Whole volumes are consumed later by
    sliding-window inference, and predictions are mapped back to the source
    geometry by inverting this pipeline.
    """

    return Compose(
        [
            *deterministic_transforms(target_spacing),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_case_records(
    manifest: dict[str, Any],
    case_ids: Sequence[str],
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Turn manifest entries into absolute-path records for MONAI.

    The manifest stores paths relative to the data root; absolute paths are
    resolved here, at call time, so the same manifest works locally and in
    Colab without modification.
    """

    data_root = get_data_root(root)
    lookup = {entry["case_id"]: entry for entry in manifest["cases"]}

    missing = [case_id for case_id in case_ids if case_id not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} case(s) absent from manifest: {missing[:10]}")

    return [
        {
            "case_id": case_id,
            "image": str(data_root / lookup[case_id]["ct"]),
            "label": str(data_root / lookup[case_id]["label"]),
            "lesion_present": lookup[case_id]["lesion_present"],
        }
        for case_id in case_ids
    ]


def build_dataloaders(
    manifest: dict[str, Any],
    split: dict[str, Any],
    fold: int = 0,
    batch_size: int = 1,
    samples_per_case: int = 2,
    num_workers: int = 4,
    limit: int | None = None,
    root: str | Path | None = None,
    val_patches: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Assemble training and validation loaders for one fold.

    Nothing model-specific belongs here: the same loaders feed the random and
    SuPreM SegResNet arms.

    ``limit`` truncates both case lists, for smoke tests and pilots.

    ``val_patches=True`` makes validation use unaugmented patches, which is what
    a per-epoch validation loss needs. The default returns whole volumes, for
    sliding-window inference.
    """

    if not 0 <= fold < len(split["folds"]):
        raise ValueError(f"fold {fold} outside 0..{len(split['folds']) - 1}")

    definition = split["folds"][fold]
    train_ids = list(definition["train"])[:limit]
    val_ids = list(definition["val"])[:limit]

    train_dataset = Dataset(
        data=build_case_records(manifest, train_ids, root),
        transform=train_transforms(samples_per_case=samples_per_case),
    )
    val_dataset = Dataset(
        data=build_case_records(manifest, val_ids, root),
        transform=(
            train_transforms(samples_per_case=samples_per_case, augment=False)
            if val_patches
            else val_transforms()
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
