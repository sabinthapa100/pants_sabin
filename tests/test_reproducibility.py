"""The seed must control the experiment, and the two arms must start comparably.

These protect the controlled ablation itself. If the PanTS output head differed
between the random and SuPreM arms, any measured effect would be confounded by
a different starting point for the only layer that produces class predictions.
"""

import os
from pathlib import Path

import pytest
import torch
from monai.utils import set_determinism

from src.models.segresnet import (
    NUM_CLASSES,
    OUTPUT_HEAD_MARKER,
    build_segresnet,
    load_suprem_weights,
)
from src.training.trainer import seed_everything


SEED = 317

checkpoint_path = os.environ.get("SUPREM_CHECKPOINT")
requires_checkpoint = pytest.mark.skipif(
    not (checkpoint_path and Path(checkpoint_path).exists()),
    reason="SUPREM_CHECKPOINT not set",
)


def build_seeded(seed: int = SEED) -> torch.nn.Module:
    """Seed FIRST, then construct - the order the trainer now uses."""
    seed_everything(seed)
    return build_segresnet("random")


def test_seed_determines_initial_weights():
    first = build_seeded()
    second = build_seeded()
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), f"{name} not reproducible"


def test_different_seed_gives_different_weights():
    a = build_seeded(SEED)
    b = build_seeded(SEED + 1)
    differing = sum(
        1 for name, t in a.state_dict().items()
        if not torch.equal(t, b.state_dict()[name])
    )
    assert differing > 0, "changing the seed changed nothing; seeding is not wired up"


def test_seeding_after_construction_would_not_control_weights():
    """Regression guard for the exact bug that was fixed.

    Constructing before seeding leaves the weights at whatever the ambient RNG
    happened to hold, so two runs with the same configured seed disagree.
    """
    torch.manual_seed(999)
    first = build_segresnet("random")
    seed_everything(SEED)          # too late

    torch.manual_seed(12345)
    second = build_segresnet("random")
    seed_everything(SEED)          # too late

    identical = all(
        torch.equal(t, second.state_dict()[n]) for n, t in first.state_dict().items()
    )
    assert not identical, "expected seed-after-construction to be uncontrolled"


@requires_checkpoint
def test_controlled_ablation_shares_the_output_head():
    """The heart of the experiment: what differs between arms, and what must not."""
    random_arm = build_seeded()
    suprem_arm = build_seeded()          # same seed, same construction order

    # C. before loading SuPreM the two models are bit-identical
    for name, tensor in random_arm.state_dict().items():
        assert torch.equal(tensor, suprem_arm.state_dict()[name]), f"{name} differs pre-load"

    before = {name: tensor.clone() for name, tensor in suprem_arm.state_dict().items()}
    report = load_suprem_weights(suprem_arm, checkpoint_path)
    after = suprem_arm.state_dict()

    head_keys = [name for name in after if OUTPUT_HEAD_MARKER in name]
    assert sorted(head_keys) == [
        "conv_final.2.conv.bias",
        "conv_final.2.conv.weight",
    ], head_keys

    # D1. the PanTS-specific head is untouched, so both arms start it identically
    for name in head_keys:
        assert torch.equal(before[name], after[name]), f"{name} must not be overwritten"
        assert torch.equal(random_arm.state_dict()[name], after[name]), (
            f"{name} differs between arms; the ablation would be confounded"
        )
    assert after["conv_final.2.conv.weight"].shape[0] == NUM_CLASSES

    # D2. exactly the intended transferable tensors changed
    changed = [n for n, t in after.items() if not torch.equal(before[n], t)]
    unchanged_non_head = [
        n for n, t in after.items()
        if torch.equal(before[n], t) and OUTPUT_HEAD_MARKER not in n
    ]
    assert len(report["transferable"]) == 81, len(report["transferable"])
    assert len(changed) + len(head_keys) + len(unchanged_non_head) == len(after)
    assert not any(OUTPUT_HEAD_MARKER in name for name in changed)
    print(
        f"\n  {len(changed)} tensors took SuPreM values, "
        f"{len(unchanged_non_head)} coincided with their random init, "
        f"{len(head_keys)} PanTS head tensors shared between arms"
    )


def test_monai_transform_stream_follows_the_seed():
    """`set_determinism` must drive MONAI's Randomizable transforms, not just torch."""
    from monai.transforms import RandGaussianNoised

    def stream(seed: int) -> list[float]:
        set_determinism(seed=seed)
        transform = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.1)
        transform.set_random_state(seed=seed)
        return [
            float(transform({"image": torch.zeros(1, 4, 4, 4)})["image"].sum())
            for _ in range(3)
        ]

    assert stream(SEED) == stream(SEED), "same seed must give the same noise stream"
    assert stream(SEED) != stream(SEED + 1), "different seed must change the stream"
