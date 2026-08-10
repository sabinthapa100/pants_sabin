"""SegResNet construction and strict SuPreM weight transfer.

The transfer tests need the official checkpoint. Set SUPREM_CHECKPOINT to
supervised_suprem_segresnet_2100.pth (from https://huggingface.co/MrGiovanni/SuPreM)
to enable them.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from src.models.segresnet import (
    NUM_CLASSES,
    OUTPUT_HEAD_MARKER,
    build_segresnet,
    load_suprem_weights,
    suprem_transfer_report,
)


# Independently verified against the HuggingFace LFS object id.
SUPREM_SHA256 = "2db81dc05cd9ea7234ca75e921e53e32b8716dc4cba88a6710742bfc282589a3"

# The SuPreM SegResNet has 83 parameters; only the 2-tensor class-specific
# output convolution is task-dependent.
TOTAL_PARAMETERS = 83
EXPECTED_TRANSFERABLE = 81

checkpoint_path = os.environ.get("SUPREM_CHECKPOINT")
requires_checkpoint = pytest.mark.skipif(
    not (checkpoint_path and Path(checkpoint_path).exists()),
    reason="SUPREM_CHECKPOINT is not set to an existing file",
)


def test_pants_has_29_classes():
    """Background plus the 28 PanTS foreground classes."""
    assert NUM_CLASSES == 29


def test_random_initialization_forward_shape():
    model = build_segresnet("random").eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 64, 64, 64))
    assert tuple(output.shape) == (1, NUM_CLASSES, 64, 64, 64)


def test_random_initialization_needs_no_checkpoint():
    assert build_segresnet("random", checkpoint_path=None) is not None


def test_unknown_initialization_rejected():
    with pytest.raises(ValueError, match="Unknown initialization"):
        build_segresnet("imagenet")


def test_suprem_without_checkpoint_rejected():
    with pytest.raises(ValueError, match="requires checkpoint_path"):
        build_segresnet("suprem")


@requires_checkpoint
def test_checkpoint_hash_matches_official_release():
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    assert digest == SUPREM_SHA256


@requires_checkpoint
def test_every_backbone_parameter_transfers():
    """
    Regression test for the loading bug.

    The previous rule excluded everything starting with `conv_final`, which
    also discarded the transferable GroupNorm and loaded only 79 of 81
    parameters. Only `conv_final.2.conv` is class-specific.
    """
    model = build_segresnet("random")
    report = suprem_transfer_report(model, checkpoint_path)

    assert report["model_parameters"] == TOTAL_PARAMETERS
    assert len(report["expected_transferable"]) == EXPECTED_TRANSFERABLE
    assert len(report["transferable"]) == EXPECTED_TRANSFERABLE
    assert report["missing_from_checkpoint"] == []
    assert report["unexpected_in_checkpoint"] == []
    assert report["shape_mismatch"] == []


@requires_checkpoint
def test_pre_head_groupnorm_is_transferred():
    """conv_final.0 is a task-independent GroupNorm and must transfer."""
    model = build_segresnet("random")
    report = suprem_transfer_report(model, checkpoint_path)
    assert "conv_final.0.weight" in report["transferable"]
    assert "conv_final.0.bias" in report["transferable"]


@requires_checkpoint
def test_output_head_is_excluded_and_left_random():
    model = build_segresnet("suprem", checkpoint_path)
    report = suprem_transfer_report(model, checkpoint_path)

    assert report["excluded_output_head"] == [
        "conv_final.2.conv.weight",
        "conv_final.2.conv.bias",
    ]
    # PanTS predicts 29 classes; SuPreM's head predicted 32.
    head = model.state_dict()["conv_final.2.conv.weight"]
    assert head.shape[0] == NUM_CLASSES


@requires_checkpoint
def test_transferred_weights_equal_the_checkpoint():
    model = build_segresnet("suprem", checkpoint_path)
    report = suprem_transfer_report(model, checkpoint_path)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["net"]
    source = {key.removeprefix("module."): value for key, value in raw.items()}

    state = model.state_dict()
    for key in report["transferable"]:
        assert torch.equal(state[key], source[key]), f"{key} was not actually loaded"


@requires_checkpoint
def test_suprem_forward_shape():
    model = build_segresnet("suprem", checkpoint_path).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 64, 64, 64))
    assert tuple(output.shape) == (1, NUM_CLASSES, 64, 64, 64)


@requires_checkpoint
def test_partial_checkpoint_is_rejected(tmp_path):
    """
    A partially loaded backbone must fail loudly.

    Silently training on a half-initialized backbone would invalidate the
    pretraining comparison, so this is an error rather than a warning.
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["net"]
    truncated = {k: v for k, v in raw.items() if "down_layers.0" not in k}
    path = tmp_path / "truncated.pth"
    torch.save({"net": truncated}, path)

    with pytest.raises(RuntimeError, match="did not transfer completely"):
        build_segresnet("suprem", path)


@requires_checkpoint
def test_random_and_suprem_differ_only_by_weights():
    """Both arms must be architecturally identical."""
    random_model = build_segresnet("random")
    suprem_model = build_segresnet("suprem", checkpoint_path)

    random_state = random_model.state_dict()
    suprem_state = suprem_model.state_dict()

    assert list(random_state) == list(suprem_state)
    for key in random_state:
        assert random_state[key].shape == suprem_state[key].shape


def test_output_head_marker_targets_only_the_class_specific_conv():
    """The exclusion marker must not match the transferable normalization."""
    assert OUTPUT_HEAD_MARKER not in "conv_final.0.weight"
    assert OUTPUT_HEAD_MARKER in "conv_final.2.conv.weight"


@requires_checkpoint
def test_load_reports_are_consistent():
    model = build_segresnet("random")
    report = load_suprem_weights(model, checkpoint_path)
    assert len(report["transferable"]) + len(report["excluded_output_head"]) == TOTAL_PARAMETERS
