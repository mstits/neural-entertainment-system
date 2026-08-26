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

import json

import numpy as np
import pytest
import yaml

import scripts.assemble_full_run as afr
from src.training.profile_utils import action_space_to_bitmasks


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


def _write_profile(path, action_space):
    path.write_text(yaml.safe_dump({"action_space": action_space}))


def test_check_recorded_profile_catches_a_reordered_action_space(tmp_path):
    """A sidecar recording a different --profile than the one this run
    is decoding with must be rejected when that profile's action_space
    bitmasks actually disagree — the silent-mis-decode scenario: same
    length, differently-ordered button lists, so every action index is
    still in range and nothing raises IndexError, but the button masks
    those indices map to are wrong."""
    action_space = [
        [], ["right"], ["right", "A"], ["right", "B"],
        ["right", "A", "B"], ["A"], ["left"], ["left", "A"],
        ["down"], ["down", "right"], ["down", "left"],
    ]
    reordered = [
        [], ["right", "A"], ["right"], ["right", "A", "B"],
        ["right", "B"], ["left"], ["A"], ["down"],
        ["left", "A"], ["down", "left"], ["down", "right"],
    ]
    default_profile = tmp_path / "default.yaml"
    other_profile = tmp_path / "other.yaml"
    _write_profile(default_profile, action_space)
    _write_profile(other_profile, reordered)
    bm = action_space_to_bitmasks(action_space)
    assert bm != action_space_to_bitmasks(reordered), (
        "fixture is broken: the two profiles must actually disagree"
    )

    sols = tmp_path / "solutions"
    sols.mkdir()
    sol = sols / "sol_000.actions.npy"
    np.save(sol, np.arange(len(action_space), dtype=np.int64))
    (sols / "sol_000.json").write_text(
        json.dumps({"profile": str(other_profile)}))

    with pytest.raises(AssertionError):
        afr.check_recorded_profile(sol, bm, str(default_profile))


def test_check_recorded_profile_silent_when_bitmasks_agree(tmp_path):
    """A recorded profile at a different path but with an identical
    action_space (so identical bitmasks) is not a mismatch."""
    action_space = [[], ["right"], ["A"]]
    default_profile = tmp_path / "default.yaml"
    same_profile = tmp_path / "same.yaml"
    _write_profile(default_profile, action_space)
    _write_profile(same_profile, action_space)
    bm = action_space_to_bitmasks(action_space)

    sols = tmp_path / "solutions"
    sols.mkdir()
    sol = sols / "sol_000.actions.npy"
    np.save(sol, np.arange(len(action_space), dtype=np.int64))
    (sols / "sol_000.json").write_text(
        json.dumps({"profile": str(same_profile)}))

    afr.check_recorded_profile(sol, bm, str(default_profile))  # must not raise


def test_check_recorded_profile_silent_when_unrecorded(tmp_path):
    """Legacy sidecars (or no sidecar at all) that never recorded which
    profile solved them must not block assembly — there is nothing to
    check against, and closing that separate recording gap is out of
    scope for this check."""
    sols = tmp_path / "solutions"
    sols.mkdir()
    sol = sols / "sol_000.actions.npy"
    np.save(sol, np.arange(3, dtype=np.int64))

    afr.check_recorded_profile(sol, (0, 1, 2), "configs/anything.yaml")
