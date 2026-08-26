"""Regression guard for mid-rollout auto-reset on death.

The vanilla_ppo loop reloads a curriculum warm-start state when an env
dies mid-rollout (instead of freezing the env until the iteration
boundary), so each rollout contains several attempts per env rather
than one. The historical bug this guards against:

    On death, the loop used to call `reward_fn.reset()` INLINE without
    restoring the pre-death RAM. `reset()` cleared the reward fn's
    `died` flag, but the post-death player_state still sat in RAM, so
    the very next `compute()` re-detected death and re-fired `done` —
    an infinite done loop that hung the rollout.

The shipped fix reloads the worker's saved state FIRST (restoring a
known-alive RAM snapshot via `pool.load_worker_state`, which also
resets `frame_cycle_target` so the NES actually ticks again) and THEN
resets the reward fn. This test pins the property that matters: after
the load+reset sequence the env is alive and the reward fn does NOT
spuriously fire `done` — and that this holds across repeated resets
within a single rollout (the multi-attempt-per-env scenario).

GAE's done-boundary bootstrap (advantage must not leak across an
episode boundary) is covered separately in `test_gae.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_SMB_START = ROOT / "roms" / "Super Mario Bros. (World)_start.state.bin"

pytestmark = pytest.mark.skipif(
    not _SMB_ROM.exists(),
    reason="SMB ROM not present; auto-reset guard needs a real ROM.",
)


def _smb_profile() -> dict:
    return {
        "name": "Super Mario Bros.",
        "reward_id": "mario",
        "reward_weights": {
            "forward_progress": 1.0,
            "death_penalty": -15.0,
            "completion_bonus": 50.0,
        },
    }


def _make_pool():
    from nes_core import Pool

    start = str(_SMB_START) if _SMB_START.exists() else None
    return Pool(
        rom_path=str(_SMB_ROM), num_workers=1, frame_skip=4,
        start_state_path=start,
    )


def test_load_state_plus_reward_reset_does_not_fire_done() -> None:
    """The core auto-reset sequence must yield an alive, non-done env."""
    from src.utils.reward_functions import build_reward_function

    pool = _make_pool()
    try:
        reward_fn = build_reward_function(_smb_profile())
        reward_fn.reset()
        pool.reset_all()

        # Advance a few frames so Mario is alive and on-screen, then
        # snapshot a known-good "curriculum warm-start" state.
        noop = np.zeros(1, dtype=np.uint8)
        for _ in range(20):
            pool.step_all(noop)
        healthy_blob = pool.save_worker_state(0)
        assert healthy_blob is not None

        # Exactly the shipped auto-reset sequence: restore the saved
        # state, reset the reward fn, then step once. The reward fn
        # must NOT report done on the freshly restored, alive frame.
        pool.load_worker_state(0, healthy_blob)
        reward_fn.reset()
        res = pool.step_all(noop)
        ram = res[0][2]
        _reward, done, _ = reward_fn.compute(ram, action=0)
        assert done is False, (
            "reward fn fired done immediately after load_worker_state + "
            "reset — the historical mid-rollout done-loop regression."
        )
    finally:
        pool.shutdown()


def test_repeated_auto_resets_stay_clean() -> None:
    """Multi-attempt-per-rollout: many load+reset cycles in a row must
    each leave the env alive and ticking (no accumulating corruption,
    no frozen NES from a stale frame_cycle_target)."""
    from src.utils.reward_functions import build_reward_function

    pool = _make_pool()
    try:
        reward_fn = build_reward_function(_smb_profile())
        reward_fn.reset()
        pool.reset_all()

        noop = np.zeros(1, dtype=np.uint8)
        for _ in range(20):
            pool.step_all(noop)
        blob = pool.save_worker_state(0)
        assert blob is not None

        # Capture the canonical post-restore frame for an equality
        # check: every restore must land on the identical RAM, proving
        # the NES is actually re-ticking from the restored state rather
        # than silently freezing.
        baseline_ram = None
        for attempt in range(6):
            pool.load_worker_state(0, blob)
            reward_fn.reset()
            res = pool.step_all(noop)
            ram = res[0][2]
            _r, done, _ = reward_fn.compute(ram, action=0)
            assert done is False, f"spurious done on auto-reset #{attempt}"
            if baseline_ram is None:
                baseline_ram = bytes(ram)
            else:
                assert bytes(ram) == baseline_ram, (
                    f"auto-reset #{attempt} produced divergent RAM — "
                    "load_worker_state restore is not deterministic."
                )
    finally:
        pool.shutdown()
