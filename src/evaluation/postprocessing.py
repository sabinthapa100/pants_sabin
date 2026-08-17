"""Connected-component postprocessing of the predicted pancreatic-lesion class.

The model over-predicts class 28: on fold-0 development data it flags 201 of
1,624 lesion-free patients, and the spurious components are systematically
smaller and less confident than genuine ones. This module removes the weakest
of them with a single-parameter rule chosen in an offline development study.

THE RULE
    Keep a hard class-28 connected component if and only if the maximum
    class-28 softmax probability anywhere inside it is >= 0.6.

``0.6`` is a threshold on the model's own softmax output. It is NOT a
calibrated 60% probability of malignancy: the network was never calibrated, and
with 29 competing classes a voxel can win the argmax at 0.3.

WHY REJECTED VOXELS ARE NOT SET TO BACKGROUND
    This is an exclusive 29-class segmenter, so a voxel the model called lesion
    is a voxel it believes is *something*. Forcing it to class 0 would assert
    "air/outside body" and inject an anatomy error into a region the model may
    confidently consider pancreas or vessel. Instead each rejected voxel takes
    ``argmax`` over channels 0..27 -- the best non-lesion explanation the model
    already computed.

    Dropping channel 28 cannot reorder channels 0..27. Softmax is
    ``p_k = e^{z_k} / Z`` with a denominator shared by every channel, so
    removing a term from ``Z`` rescales all remaining probabilities by one
    positive constant and preserves their order. The ranking of 0..27 is
    therefore identical whether computed from the 29-class softmax, the
    28-class softmax, or the raw logits -- which is why this is implemented as
    a plain ``argmax`` on logits.

    The rejected voxel does NOT return to the class it would have had if the
    model had never predicted lesion; it returns to the runner-up the model
    itself ranked second. Those are the same thing here, because argmax over a
    subset is exactly "the best remaining option".

SCOPE
    This defines the final hard semantic label map only. The continuous class-28
    probability map is deliberately left untouched -- see
    ``predict_case_in_source_geometry``.

DEVELOPMENT PROVENANCE
    The threshold was selected from nine predeclared candidates on fold 0, the
    same fold that selected the checkpoint. It is not a globally optimal value
    and its fold-0 numbers are optimistic by an unmeasured amount.
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
    """Class-``lesion_class`` softmax probability, ``[D, H, W]``.

    Uses ``exp(z_k - logsumexp(z))`` rather than ``softmax(z)[k]`` so only one
    single-channel map is allocated instead of a second 29-channel float32
    volume -- for a typical prepared case, ~40 MB instead of ~1.2 GB. The two
    are the same function; on real logits from this model (range about
    [-92, +17]) they agree to ~3e-7, well inside float32 noise.
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
    logits
        ``[1, C, D, H, W]`` float class scores in the frame the components are
        to be measured in. Must be the *same* logits ``labels`` came from.
    labels
        ``[1, 1, D, H, W]`` integer semantic labels, i.e. ``argmax`` over
        ``logits``.

    Returns a dict with the filtered ``labels`` (a new tensor; the input is
    never mutated), one record per component, the number of relabelled voxels,
    and how many of those went to each fallback class.

    Every voxel that was not part of a rejected component is bit-identical to
    the input, including voxels of other classes and retained lesion voxels.
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
