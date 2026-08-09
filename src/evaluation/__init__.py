"""Evaluation and inference utilities shared across model families."""

from .segmentation import dice_score

__all__ = ["dice_score"]
