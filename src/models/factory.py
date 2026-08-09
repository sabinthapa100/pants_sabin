"""Model factory shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _load_suprem_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Load compatible SuPreM SegResNet backbone weights.

    The task-specific output layer is intentionally excluded because PanTS uses
    a different class ontology from the upstream pretraining task.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("net", checkpoint)

    def simplify(key: str) -> str:
        for prefix in ("module.", "features.", "backbone.", "model."):
            key = key.replace(prefix, "")
        return key

    source = {simplify(k): v for k, v in state.items()}
    target = model.state_dict()
    loaded = 0
    for key, value in target.items():
        if key.startswith("conv_final"):
            continue
        if key in source and source[key].shape == value.shape:
            target[key] = source[key]
            loaded += 1

    if loaded == 0:
        raise RuntimeError("No compatible SuPreM parameters were loaded.")
    model.load_state_dict(target)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Build a model from the `model` section of an experiment config."""
    name = config["name"]
    if name != "segresnet":
        raise ValueError(
            f"Model '{name}' is not constructed in-process. "
            "nnU-Net is invoked through its official CLI adapter."
        )

    from monai.networks.nets import SegResNet

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=int(config.get("init_filters", 16)),
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 29)),
        dropout_prob=0.0,
    )

    initialization = config.get("initialization", "random")
    if initialization == "suprem":
        path = config.get("pretrained_checkpoint")
        if not path:
            raise ValueError("SuPreM initialization requires pretrained_checkpoint.")
        _load_suprem_weights(model, path)
    elif initialization != "random":
        raise ValueError(f"Unknown initialization: {initialization}")

    return model
