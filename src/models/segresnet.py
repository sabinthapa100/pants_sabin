"""The PanTS SegResNet and strict SuPreM weight transfer.

Both SegResNet experiments construct the identical architecture here. Only
``initialization`` differs, which is what makes the pretraining comparison a
controlled experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.labels import CLASS_MAP


# background + the 28 PanTS foreground classes
NUM_CLASSES = len(CLASS_MAP) + 1

# Architecture of the official SuPreM SegResNet backbone. Verified against
# supervised_suprem_segresnet_2100.pth: all 83 parameter names align, and only
# the class-specific output convolution differs in shape (SuPreM 32 -> PanTS 29).
BLOCKS_DOWN = (1, 2, 2, 4)
BLOCKS_UP = (1, 1, 1)
INIT_FILTERS = 16

# In MONAI's SegResNet, `conv_final` is a Sequential:
#   conv_final.0        GroupNorm      - task independent, transferable
#   conv_final.1        activation     - no parameters
#   conv_final.2.conv   Conv3d         - task specific, NOT transferable
# Excluding the whole `conv_final` block would needlessly discard the
# normalization parameters. The upstream SuPreM loader excludes exactly this
# substring, and asserts that everything else transfers.
OUTPUT_HEAD_MARKER = "conv_final.2.conv"

_CHECKPOINT_PREFIXES = ("module.", "backbone.", "model.")


def build_segresnet(
    initialization: str = "random",
    checkpoint_path: str | Path | None = None,
    out_channels: int = NUM_CLASSES,
    in_channels: int = 1,
    init_filters: int = INIT_FILTERS,
) -> torch.nn.Module:
    """
    Build the PanTS SegResNet.

    Parameters
    ----------
    initialization:
        ``"random"`` or ``"suprem"``. The architecture is identical either way.
    checkpoint_path:
        Path to the official SuPreM SegResNet checkpoint. Required when
        ``initialization="suprem"``.

    Returns
    -------
    A model mapping ``[B, 1, D, H, W]`` to ``[B, 29, D, H, W]`` logits.
    """

    from monai.networks.nets import SegResNet

    model = SegResNet(
        blocks_down=list(BLOCKS_DOWN),
        blocks_up=list(BLOCKS_UP),
        init_filters=int(init_filters),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        dropout_prob=0.0,
    )

    if initialization == "random":
        return model

    if initialization != "suprem":
        raise ValueError(
            f"Unknown initialization '{initialization}'. Expected 'random' or 'suprem'."
        )

    if checkpoint_path is None:
        raise ValueError("initialization='suprem' requires checkpoint_path.")

    load_suprem_weights(model, checkpoint_path)
    return model


def _strip_prefixes(key: str) -> str:
    """Remove wrapper prefixes such as the DataParallel ``module.``.

    ``removeprefix`` is used rather than ``replace`` because ``replace`` would
    strip the substring anywhere in the key, silently corrupting names.
    """

    for prefix in _CHECKPOINT_PREFIXES:
        key = key.removeprefix(prefix)
    return key


def suprem_transfer_report(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """
    Compare a SuPreM checkpoint against the model without mutating it.

    Returns counts and key lists for: keys intended to transfer, keys that
    actually match by name and shape, the intentionally excluded output head,
    keys missing from the checkpoint, unexpected checkpoint keys, and any
    shape mismatch outside the output head.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("net", checkpoint)
    source = {_strip_prefixes(key): value for key, value in raw_state.items()}
    target = model.state_dict()

    expected = [key for key in target if OUTPUT_HEAD_MARKER not in key]
    excluded_head = [key for key in target if OUTPUT_HEAD_MARKER in key]

    transferable: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    missing: list[str] = []

    for key in expected:
        if key not in source:
            missing.append(key)
        elif source[key].shape != target[key].shape:
            shape_mismatch.append(
                {
                    "key": key,
                    "model": tuple(target[key].shape),
                    "checkpoint": tuple(source[key].shape),
                }
            )
        else:
            transferable.append(key)

    unexpected = [key for key in source if key not in target]

    return {
        "checkpoint": str(checkpoint_path),
        "model_parameters": len(target),
        "expected_transferable": expected,
        "transferable": transferable,
        "excluded_output_head": excluded_head,
        "missing_from_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "shape_mismatch": shape_mismatch,
    }


def format_transfer_report(report: dict[str, Any]) -> str:
    """Render a transfer report for logs and run provenance."""

    return "\n".join(
        [
            "SuPreM transfer report",
            f"  checkpoint            {report['checkpoint']}",
            f"  model parameters      {report['model_parameters']}",
            f"  expected transferable {len(report['expected_transferable'])}",
            f"  transferred           {len(report['transferable'])}",
            f"  excluded output head  {len(report['excluded_output_head'])} "
            f"{report['excluded_output_head']}",
            f"  missing in checkpoint {len(report['missing_from_checkpoint'])} "
            f"{report['missing_from_checkpoint']}",
            f"  unexpected in ckpt    {len(report['unexpected_in_checkpoint'])} "
            f"{report['unexpected_in_checkpoint']}",
            f"  shape mismatches      {len(report['shape_mismatch'])} "
            f"{report['shape_mismatch']}",
        ]
    )


def load_suprem_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """
    Load every transferable SuPreM parameter into ``model``, in place.

    The class-specific output convolution is deliberately left at its random
    initialization because PanTS uses a different label ontology from the
    upstream pretraining task.

    Raises
    ------
    RuntimeError:
        If any intended backbone parameter fails to transfer. Partial loading
        is treated as a hard failure rather than a warning, because a quietly
        half-initialized backbone would invalidate the pretraining comparison.
    """

    report = suprem_transfer_report(model, checkpoint_path)

    expected_count = len(report["expected_transferable"])
    actual_count = len(report["transferable"])

    if actual_count != expected_count:
        raise RuntimeError(
            "SuPreM backbone did not transfer completely: "
            f"{actual_count}/{expected_count} parameters loaded.\n"
            + format_transfer_report(report)
        )

    if actual_count != report["model_parameters"] - len(report["excluded_output_head"]):
        raise RuntimeError(
            "Unexpected parameter accounting while loading SuPreM weights.\n"
            + format_transfer_report(report)
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("net", checkpoint)
    source = {_strip_prefixes(key): value for key, value in raw_state.items()}

    state = model.state_dict()
    for key in report["transferable"]:
        state[key] = source[key]
    model.load_state_dict(state)

    return report
