"""Shared inference primitives for in-process 3D segmentation models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from monai.inferers import sliding_window_inference

from ..data.labels import PANCREATIC_LESION


def predict_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sw_device: str | torch.device | None = None,
    device: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Return full-volume logits using MONAI sliding-window inference.

    Memory model
    ------------
    The stitched output is the expensive object, not the patches: a
    ``512 x 512 x 300`` volume at 29 classes in float32 is roughly 9 GB, which
    is what previously exhausted VRAM. MONAI separates the two devices, so by
    default the windows are evaluated on the model's device (``sw_device``)
    while the accumulator lives on the host (``device="cpu"``). Only one patch
    of activations is ever resident in VRAM.

    Pass ``device=None`` to keep the accumulator wherever the input already is.

    Parameters
    ----------
    model:
        A model mapping ``[B, C, H, W, D]`` to class logits.
    image:
        Preprocessed 5D tensor ``[B, C, H, W, D]``.
    roi_size:
        Sliding-window spatial patch size.
    overlap:
        Fractional overlap between neighboring windows.
    sw_batch_size:
        Windows evaluated simultaneously. Keep at 1 unless VRAM is plentiful.
    sw_device:
        Where patches are executed. Defaults to the model's own device.
    device:
        Where the full-volume output is accumulated. Defaults to CPU.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] input; got shape {tuple(image.shape)}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    if sw_device is None:
        sw_device = next(model.parameters()).device

    model.eval()
    with torch.inference_mode():
        return sliding_window_inference(
            inputs=image,
            roi_size=tuple(int(v) for v in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
            mode="gaussian",
            sw_device=sw_device,
            device=device,
        )


def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert multiclass logits to integer semantic labels ``[B, 1, H, W, D]``."""
    if logits.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] logits; got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=1, keepdim=True)


def lesion_probability(
    logits: torch.Tensor,
    label: int = PANCREATIC_LESION,
) -> torch.Tensor:
    """
    Softmax probability of the pancreatic-lesion class, ``[B, 1, H, W, D]``.

    A continuous score is required for any future threshold-based or ranking
    metric (specificity at an operating point, AUC), which an argmax label map
    cannot provide.
    """
    if logits.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] logits; got {tuple(logits.shape)}")
    if not 0 <= label < logits.shape[1]:
        raise ValueError(f"label {label} outside the {logits.shape[1]} predicted channels")
    return torch.softmax(logits, dim=1)[:, label : label + 1]


def lesion_mask(
    logits: torch.Tensor,
    label: int = PANCREATIC_LESION,
) -> torch.Tensor:
    """
    Binary pancreatic-lesion mask from the argmax labels, ``[B, 1, H, W, D]``.

    Derived from the same argmax as :func:`logits_to_labels` so the mask is
    always consistent with the full semantic prediction.
    """
    return (logits_to_labels(logits) == label).to(torch.uint8)


def predict_case_in_source_geometry(
    model: torch.nn.Module,
    image_path: str,
    *,
    transform: Any = None,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sw_device: str | torch.device | None = None,
    accumulate_device: str | torch.device | None = "cpu",
    want_lesion_probability: bool = False,
) -> dict[str, Any]:
    """
    Predict one case and map the result back onto the source CT grid.

    Returns ``labels`` (integer 0...28) and optionally ``lesion_probability``,
    both as MetaTensors carrying the *source* affine and shape.

    The two outputs are inverted differently on purpose: the semantic label map
    must use nearest-neighbour resampling, or the inverse would interpolate
    class identifiers and produce values like 17.4 that name no structure.
    The probability map is continuous and is inverted with linear
    interpolation.

    Large intermediates are released before returning so that looping over
    patients does not accumulate memory.

    Nothing but the CT is read. The default transform is the image-only chain,
    so this function is exactly what runs on an unseen scan from an external
    cohort: no label, no manifest, no split, no prepared cache, no case ID.
    """
    from monai.data import MetaTensor
    from monai.transforms import Invertd

    from ..data.transforms import inference_transforms

    transform = transform or inference_transforms()

    prepared = transform({"image": image_path})
    reference = prepared["image"]

    logits = predict_logits(
        model,
        reference.unsqueeze(0),
        roi_size=roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        sw_device=sw_device,
        device=accumulate_device,
    )

    predicted_labels = logits_to_labels(logits)[0].to(torch.float32)
    probability = (
        lesion_probability(logits)[0].to(torch.float32) if want_lesion_probability else None
    )
    del logits

    def invert(values: torch.Tensor, nearest: bool) -> MetaTensor:
        payload = dict(prepared)
        payload["pred"] = MetaTensor(
            values.clone(),
            meta=reference.meta.copy(),
            applied_operations=list(reference.applied_operations),
        )
        restored = Invertd(
            keys="pred",
            transform=transform,
            orig_keys="image",
            nearest_interp=nearest,
            to_tensor=True,
        )(payload)["pred"]
        return restored

    result: dict[str, Any] = {"labels": invert(predicted_labels, nearest=True)}
    if probability is not None:
        result["lesion_probability"] = invert(probability, nearest=False)

    del predicted_labels, probability, prepared, reference
    return result
