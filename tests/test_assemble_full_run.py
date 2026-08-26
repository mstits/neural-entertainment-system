"""Tests for scripts/assemble_full_run.py's level-boundary settle.

The splice's settle must land on the same frame an entrance blob from
go_explore_chain.extract_next_entrance would have been snapshotted on:
that function settles until the (world, level) key holds stable for
many consecutive reads, not after a flat no-op count. No real Pool is
needed here — `settle_to_stable_wd` only calls a `step` callable and
reads (world, level) off the returned ram, so a scripted fake stands in
for the emulator.
"""

from __future__ import annotations

import scripts.assemble_full_run as afr


def _scripted_step(wd_sequence):
    """Returns a `step(mask)` callable that ignores `mask` and, on its
    Nth call, produces a ram-like buffer reporting wd_sequence[N]
    (clamped to the last entry once the sequence is exhausted) at the
    real (world, level) RAM addresses."""

    calls = {"n": 0}

    def step(mask):
        idx = min(calls["n"], len(wd_sequence) - 1)
        calls["n"] += 1
        w, l = wd_sequence[idx]
        ram = bytearray(0x800)
        ram[0x75F] = w
        ram[0x75C] = l
        return bytes(ram)

    step.calls = calls
    return step


def test_settle_rides_out_a_transient_key_past_the_old_fixed_8_step_point():
    """A wd reading that is still mid-transition at step 8 (what the old
    fixed-8-noop splice would have snapshotted as the entrance) must be
    rejected in favor of the value the read later settles on for
    STABLE_FOR consecutive steps — mirroring go_explore_chain's
    judge_transition settle instead of a flat no-op count."""
    # Steps 1-8 (the initial settle window) all read the transient value
    # (1, 3) — exactly what a flat 8-step settle would bank. From step 9
    # onward the key moves to (1, 4) and holds there for good.
    wd_sequence = [(1, 3)] * afr.SETTLE_NOOPS + [(1, 4)] * (afr.STABLE_FOR + 20)

    step = _scripted_step(wd_sequence)
    ram = afr.settle_to_stable_wd(step)

    settled_wd = (int(ram[0x75F]), int(ram[0x75C]))
    assert settled_wd == (1, 4), (
        "settle must land on the value that actually held stable, not the "
        "transient reading present at the old fixed 8-step mark"
    )
    # A flat 8-step settle would have called step() exactly 8 times. The
    # hardened settle must keep going until STABLE_FOR consecutive matches
    # are observed, so it takes meaningfully more steps than that.
    assert step.calls["n"] == afr.SETTLE_NOOPS + afr.STABLE_FOR + 1, (
        f"expected settle to consume settle + stable_for + 1 steps, got "
        f"{step.calls['n']}"
    )
