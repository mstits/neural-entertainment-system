"""Tests for the batched reward dispatch entry point.

The trainer's per-step inner loop currently calls
`reward_fn.compute(ram, action)` once per genome. We exposed a
`nes_core.compute_rewards_batch(fns, rams, actions)` that runs the
same computation across all genomes in one PyO3 call, saving the
per-call argument-marshaling overhead.

These tests pin the batched path to behave identically to N
independent per-call invocations — same rewards, same done flags,
same level IDs, same accumulated breakdown.
"""

from __future__ import annotations

import nes_core


def _profile() -> dict:
    return {
        "name": "mario",
        "reward_weights": {
            "forward_progress": 1.0,
            "checkpoint_scale": 0.0,  # disable dense checkpoints for clean comparison
            "death_penalty": 0.0,
            "time_penalty": 0.0,
        },
    }


def _ram_with_x(world_x: int) -> bytes:
    buf = bytearray(2048)
    buf[0x006D] = (world_x >> 8) & 0xFF
    buf[0x0086] = world_x & 0xFF
    return bytes(buf)


def test_batched_matches_per_call() -> None:
    """Same inputs → same outputs."""
    fns_batch = [nes_core.build_reward_function(_profile()) for _ in range(4)]
    fns_per = [nes_core.build_reward_function(_profile()) for _ in range(4)]
    for f in fns_batch + fns_per:
        f.reset()

    # Initialize prev_x via a first step at x=0.
    rams_init = [_ram_with_x(0)] * 4
    actions_init = [0] * 4
    nes_core.compute_rewards_batch(fns_batch, rams_init, actions_init)
    for f, r, a in zip(fns_per, rams_init, actions_init):
        f.compute(r, action=a)

    # Now step each genome at a different forward x.
    rams = [_ram_with_x(x) for x in (100, 250, 500, 1000)]
    actions = [0, 1, 2, 3]
    batched = nes_core.compute_rewards_batch(fns_batch, rams, actions)
    per_call = [
        f.compute(r, action=a)
        for f, r, a in zip(fns_per, rams, actions)
    ]
    assert batched == per_call


def test_batched_updates_per_genome_state() -> None:
    """Each fn in the batch should accumulate into its OWN breakdown,
    not a shared one. After the batched call, fn[0]'s forward-progress
    should reflect only fn[0]'s ram input."""
    fns = [nes_core.build_reward_function(_profile()) for _ in range(3)]
    for f in fns:
        f.reset()
    nes_core.compute_rewards_batch(fns, [_ram_with_x(0)] * 3, [0] * 3)
    rams = [_ram_with_x(100), _ram_with_x(500), _ram_with_x(1000)]
    nes_core.compute_rewards_batch(fns, rams, [0, 0, 0])

    bd0 = dict(fns[0].breakdown)
    bd1 = dict(fns[1].breakdown)
    bd2 = dict(fns[2].breakdown)
    # Each genome should have its own forward total, scaling with its X.
    assert bd0["forward"] < bd1["forward"] < bd2["forward"]


def test_batched_validates_length_mismatch() -> None:
    fns = [nes_core.build_reward_function(_profile())]
    fns[0].reset()
    try:
        nes_core.compute_rewards_batch(fns, [b"\x00" * 2048] * 2, [0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_batched_handles_empty_input() -> None:
    """Empty input lists should return empty output (no crash)."""
    out = nes_core.compute_rewards_batch([], [], [])
    assert out == []
