"""Pin sticky-action override behavior in the trainer.

Sticky actions: each step, with probability `sticky_action_prob`,
the emulator executes the previous step's action instead of the
policy's freshly-sampled one. Targets the SMB jump-cutoff failure
where frame_skip=4 + entropy ~1.5 produces oscillating action
sequences that release A mid-jump.

These tests pin the helper that decides per-step whether to
override and what log-prob to record:

* When sticky_action_prob = 0, sampled action wins every time.
* When sticky_action_prob = 1, every step after step 0 sticks to
  last action (no override on step 0 — no meaningful "last" yet).
* Recorded log_prob matches the OVERRIDDEN action's log-prob when
  sticky fires (PPO importance ratio consistency).
* The `last_action_per_genome` slot tracks what was actually
  executed, not what was sampled.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.training.trainer import Trainer


def _stub(sticky_prob: float) -> Trainer:
    """Build just enough Trainer state to call _apply_sampled_actions."""
    t = Trainer.__new__(Trainer)
    t.sticky_action_prob = sticky_prob
    return t


def test_sticky_zero_passes_sampled_actions_through() -> None:
    t = _stub(0.0)
    active = [0, 1, 2]
    sampled = np.array([3, 5, 7], dtype=np.int64)
    lp = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    log_probs_all = np.full((3, 8), -2.08, dtype=np.float32)  # uniform 1/8
    actions = [0] * 4
    log_probs_old = [0.0] * 4
    last = [9, 9, 9, 9]
    t._apply_sampled_actions(
        active, sampled, lp, log_probs_all, actions, log_probs_old, last, step=10,
    )
    # No override: actions == sampled, log_probs from chosen_lp, last
    # tracks sampled.
    assert actions[:3] == [3, 5, 7]
    assert [round(x, 2) for x in log_probs_old[:3]] == [-1.0, -2.0, -3.0]
    assert last[:3] == [3, 5, 7]


def test_sticky_one_overrides_every_step_after_zero() -> None:
    t = _stub(1.0)
    active = [0, 1]
    sampled = np.array([2, 4], dtype=np.int64)
    lp = np.array([-1.0, -2.0], dtype=np.float32)
    # Make per-action log_probs distinguishable so we can verify the
    # right one is picked up under override.
    log_probs_all = np.array([
        [-10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0],
        [-20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0],
    ], dtype=np.float32)
    actions = [0] * 2
    log_probs_old = [0.0] * 2
    last = [5, 7]
    t._apply_sampled_actions(
        active, sampled, lp, log_probs_all, actions, log_probs_old, last, step=10,
    )
    # Both genomes should stick to their last_action (5 and 7).
    assert actions == [5, 7]
    # log_prob recorded should be the OVERRIDDEN action's log-prob from
    # log_probs_all, NOT the sampled action's lp — but FLOORED at -13.0.
    # A near-deterministic policy gives a stuck action a log-prob of
    # -30..-46; unclamped that explodes the PPO ratio to NaN, so the
    # recorded value is clamped (raw -15/-27 both floor to -13.0).
    assert log_probs_old[0] == pytest.approx(-13.0)  # clamp(log_probs_all[0,5])
    assert log_probs_old[1] == pytest.approx(-13.0)  # clamp(log_probs_all[1,7])
    # last unchanged because we re-stuck (5→5, 7→7).
    assert last == [5, 7]


def test_sticky_skipped_on_step_zero() -> None:
    """Even with sticky_prob=1.0, step 0 should NOT override — there's
    no meaningful 'last action' (defaults to 0=NOOP), and we don't want
    to forcibly inject NOOPs at the start of every episode."""
    t = _stub(1.0)
    active = [0]
    sampled = np.array([3], dtype=np.int64)
    lp = np.array([-1.5], dtype=np.float32)
    log_probs_all = np.full((1, 8), -2.08, dtype=np.float32)
    actions = [0]
    log_probs_old = [0.0]
    last = [0]
    t._apply_sampled_actions(
        active, sampled, lp, log_probs_all, actions, log_probs_old, last, step=0,
    )
    # Step 0: sampled wins regardless of sticky_prob.
    assert actions == [3]
    assert log_probs_old == [pytest.approx(-1.5)]
    assert last == [3]


def test_sticky_updates_last_to_executed_action() -> None:
    """When sticky fires, `last_action_per_genome` should still reflect
    the EXECUTED action (which equals the OVERRIDDEN action), so the
    next step's potential override re-sticks correctly."""
    t = _stub(1.0)
    active = [0]
    sampled = np.array([2], dtype=np.int64)
    lp = np.array([-1.0], dtype=np.float32)
    log_probs_all = np.full((1, 8), -2.08, dtype=np.float32)
    actions = [0]
    log_probs_old = [0.0]
    last = [4]  # last executed was 4
    t._apply_sampled_actions(
        active, sampled, lp, log_probs_all, actions, log_probs_old, last, step=5,
    )
    # Override fired → executed = 4 → last stays 4 (would chain on next step).
    assert actions == [4]
    assert last == [4]


def test_sticky_partial_probability_average_correct() -> None:
    """With sticky_prob=0.5 over many trials, ~50% of steps should
    stick. Stochastic test — uses a fixed seed and checks the ratio
    is in a reasonable bound."""
    np.random.seed(42)
    t = _stub(0.5)
    n_trials = 1000
    overrides = 0
    sampled = np.array([3], dtype=np.int64)
    lp = np.array([-1.0], dtype=np.float32)
    # Distinguishable log-probs so we can detect override events.
    log_probs_all = np.array([[-10.0] * 8], dtype=np.float32)
    # Unique marker ABOVE the -13.0 clamp floor so override detection is
    # unaffected by the clamp (the clamp itself is tested separately).
    log_probs_all[0, 5] = -12.0
    for _ in range(n_trials):
        actions = [0]
        log_probs_old = [0.0]
        last = [5]
        t._apply_sampled_actions(
            [0], sampled, lp, log_probs_all, actions, log_probs_old, last, step=10,
        )
        if actions[0] == 5:  # override fired
            overrides += 1
            assert log_probs_old[0] == pytest.approx(-12.0)
    # Allow ±5% slop on a 1000-trial fair coin.
    assert 450 <= overrides <= 550, f"sticky rate {overrides}/1000 out of [450, 550]"


def test_sticky_logprob_clamped_at_floor():
    """A near-deterministic policy gives a stuck action a very negative
    log-prob (-30..-46); recording it unclamped explodes the PPO ratio
    to NaN (the 2026-07-19 collapse). The GA sticky path must floor it
    at -13.0, matching the vanilla path."""
    t = _stub(1.0)  # always stick
    sampled = np.array([3], dtype=np.int64)
    lp = np.array([-1.0], dtype=np.float32)
    log_probs_all = np.array([[-2.0] * 8], dtype=np.float32)
    log_probs_all[0, 5] = -46.0  # stuck action ~0 probability
    actions = [0]
    log_probs_old = [0.0]
    last = [5]
    t._apply_sampled_actions(
        [0], sampled, lp, log_probs_all, actions, log_probs_old, last, step=10,
    )
    assert actions == [5]
    assert log_probs_old[0] == pytest.approx(-13.0), \
        "stuck-action log-prob must be floored at -13.0, not recorded raw"
