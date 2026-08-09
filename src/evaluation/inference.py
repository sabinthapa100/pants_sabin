"""Shared inference primitives for in-process 3D segmentation models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.inferers import sliding_window_inference


def predict_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    roi_size: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
) -> torch.Tensor:
    """Return full-volume logits using MONAI sliding-window inference.

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
        Number of windows evaluated simultaneously.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] input; got shape {tuple(image.shape)}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    model.eval()
    with torch.inference_mode():
        return sliding_window_inference(
            inputs=image,
            roi_size=tuple(int(v) for v in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
            mode="gaussian",
        )


def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert multiclass logits to integer semantic labels."""
    if logits.ndim != 5:
        raise ValueError(f"Expected [B,C,H,W,D] logits; got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=1, keepdim=True)
