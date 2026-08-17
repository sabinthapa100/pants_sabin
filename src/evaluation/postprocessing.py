"""Connected-component postprocessing of the predicted pancreatic-lesion class.

Rule: keep a hard class-28 connected component only if the maximum class-28
softmax probability inside it is >= 0.6. The threshold was selected from nine
predeclared candidates on fold 0 -- the same fold that selected the checkpoint,
so its development numbers are optimistic by an unmeasured amount.

0.6 is a threshold on the model's own softmax, not a calibrated probability of
malignancy; with 29 competing classes a voxel can win the argmax at 0.3.

Rejected voxels take ``argmax`` over channels 0..27 rather than background.
This is an exclusive 29-class segmenter, so forcing class 0 would assert
"outside body" in tissue the model believes is pancreas or vessel. Dropping
channel 28 cannot reorder the rest: softmax divides every channel by one shared
denominator, so removing a term rescales the survivors by a positive constant
and preserves their order -- hence a plain argmax on logits.

Only the hard label map is affected. The continuous class-28 probability map is
left untouched; see ``predict_case_in_source_geometry``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.labels import PANCREATIC_LESION


# Frozen development rule. Named rather than inlined so no caller can silently
# use a different value, and so every output can record what it ran with.
LESION_PEAK_PROBABILITY = 0.6
LESION_COMPONENT_CONNECTIVITY = 26
EVALUATION_FRAME = "canonical RAS 1.5 mm prepared grid, before inversion"

# scipy's structuring elements are indexed by how many axes a neighbour may
# differ on: 1 -> faces only (6), 2 -> faces+edges (18), 3 -> everything (26).
_CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


def lesion_peak_probability_map(
    logits: torch.Tensor,
    lesion_class: int = PANCREATIC_LESION,
) -> torch.Tensor:
    """Class-``lesion_class`` softmax probability, shape ``[D, H, W]``.

    Uses ``exp(z_k - logsumexp(z))`` rather than ``softmax(z)[k]`` to allocate
    one single-channel map instead of a second 29-channel float32 volume
    (~40 MB versus ~1.2 GB for a typical case). The two agree to ~3e-7 on this
    model's logit range.
    """
    if logits.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W] logits; got {tuple(logits.shape)}")
    if logits.shape[0] != 1:
        raise ValueError(f"expected batch size 1; got {logits.shape[0]}")
    return torch.exp(logits[0, lesion_class] - torch.logsumexp(logits[0], dim=0))


def best_non_lesion_labels(
    logits: torch.Tensor,
    lesion_class: int = PANCREATIC_LESION,
) -> torch.Tensor:
    """Argmax over every channel except the lesion class, ``[1, 1, D, H, W]``.

    Assumes the lesion channel is last, which it is for PanTS (classes 0..28
    with 28 = pancreatic lesion), so "all channels except the lesion" is the
    contiguous slice ``[:lesion_class]`` and no tensor is copied.
    """
    if lesion_class != logits.shape[1] - 1:
        raise ValueError(
            f"lesion class {lesion_class} is not the final channel of "
            f"{logits.shape[1]}; slicing would drop the wrong channels"
        )
    return torch.argmax(logits[:, :lesion_class], dim=1, keepdim=True)


def filter_lesion_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lesion_class: int = PANCREATIC_LESION,
    min_peak_probability: float = LESION_PEAK_PROBABILITY,
    connectivity: int = LESION_COMPONENT_CONNECTIVITY,
) -> dict[str, Any]:
    """Reject weak lesion components and relabel their voxels non-lesion.

    Parameters
    ----------
    logits : torch.Tensor
        Full-volume class scores, shape ``[1, C, D, H, W]``, in the frame the
        components are measured in. Must be the logits ``labels`` came from.
    labels : torch.Tensor
        Hard semantic labels, shape ``[1, 1, D, H, W]``.

    Returns
    -------
    dict
        Filtered ``labels`` (a new tensor; the input is never mutated), one
        record per component, the relabelled voxel count, and the fallback
        class distribution. Voxels outside a rejected component are unchanged.
    """
    from scipy import ndimage

    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError(f"expected [1,1,D,H,W] labels; got {tuple(labels.shape)}")
    if labels.shape[2:] != logits.shape[2:]:
        raise ValueError(
            f"labels {tuple(labels.shape[2:])} and logits {tuple(logits.shape[2:])} "
            "describe different volumes"
        )
    if connectivity not in _CONNECTIVITY_RANK:
        raise ValueError(f"connectivity must be one of {sorted(_CONNECTIVITY_RANK)}")

    filtered = labels.clone()
    lesion_mask = (labels[0, 0] == lesion_class).cpu().numpy()

    structure = ndimage.generate_binary_structure(3, _CONNECTIVITY_RANK[connectivity])
    components, count = ndimage.label(lesion_mask, structure=structure)

    record = {
        "labels": filtered,
        "components": [],
        "components_found": count,
        "components_retained": 0,
        "components_rejected": 0,
        "relabelled_voxels": 0,
        "fallback_class_counts": {},
        "rule": {
            "min_peak_probability": float(min_peak_probability),
            "connectivity": connectivity,
            "lesion_class": lesion_class,
            "comparison": ">=",
        },
    }
    if count == 0:
        return record

    probability = lesion_peak_probability_map(logits, lesion_class).cpu().numpy()
    index = list(range(1, count + 1))
    peaks = np.atleast_1d(ndimage.maximum(probability, components, index))
    sizes = np.atleast_1d(ndimage.sum_labels(np.ones_like(components), components, index))

    # ">=" and not ">": a component whose peak sits exactly on the threshold is
    # retained. Stated explicitly because the two differ on real data.
    retained = peaks >= min_peak_probability
    record["components"] = [
        {
            "component_index": i,
            "voxel_count": int(size),
            "peak_probability": float(peak),
            "retained": bool(keep),
        }
        for i, size, peak, keep in zip(index, sizes, peaks, retained)
    ]
    record["components_retained"] = int(retained.sum())
    record["components_rejected"] = int((~retained).sum())
    if retained.all():
        return record

    rejected_indices = [i for i, keep in zip(index, retained) if not keep]
    rejected = torch.from_numpy(np.isin(components, rejected_indices)).to(labels.device)

    fallback = best_non_lesion_labels(logits, lesion_class)[0, 0].to(labels.dtype)
    filtered[0, 0][rejected] = fallback[rejected]

    replacements = fallback[rejected]
    values, counts = torch.unique(replacements, return_counts=True)
    record["relabelled_voxels"] = int(rejected.sum())
    record["fallback_class_counts"] = {
        int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())
    }
    return record
