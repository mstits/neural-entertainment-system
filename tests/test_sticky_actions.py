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

The second block pins `StickyBoundary`, the vanilla-PPO path's
episode-boundary guard: a mid-rollout in-place restart begins a new
episode, and the carried action must not cross it (eval gates the roll
on `step > 0` per episode — scripts/eval_game.py). Opt-in, so the
disabled guard must stay bit-identical to the pre-flag roll.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.training.trainer import StickyBoundary, Trainer


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


# ===== StickyBoundary: no carry-over across an episode boundary =====


def _simulate_rollout(
    enabled: bool, death_step: int, n_steps: int = 5
) -> list[int]:
    """Replay the vanilla-PPO rollout's sticky bookkeeping, in order.

    Per step the loop rolls sticky (skipped at t == 0), writes the
    EXECUTED action into the carry slot, consumes the boundary
    suppression, steps the env, and — if the env died — restarts it in
    place. One env, sticky p = 1.0, so every post-t0 action either
    repeats the carry slot or proves it did not.

    Returns the executed action per step; the sampled action at step t
    is t + 1, so a repeat is visible by value.
    """
    guard = StickyBoundary(1, enabled)
    prev = np.zeros(1, dtype=np.int64)
    p_env = np.ones(1, dtype=np.float64)
    executed: list[int] = []
    for t in range(n_steps):
        act = np.array([t + 1], dtype=np.int64)
        if t > 0:
            rows = guard.override_rows(p_env)
            if rows.size:
                act[rows] = prev[rows]
        prev[:] = act
        guard.consume()
        executed.append(int(act[0]))
        if t == death_step:
            guard.mark_restart(0, prev)
    return executed


def test_no_sticky_override_on_first_step_after_restart() -> None:
    """The step after an in-place restart is an episode's FIRST step:
    it must run the freshly sampled action, not the input the previous
    life died holding. Matches eval's per-episode `step > 0` gate."""
    executed = _simulate_rollout(enabled=True, death_step=2)
    # t0 sampled (no roll at t == 0), t1/t2 stick to the carry slot,
    # t3 is the post-restart first step (sampled 4 wins), t4 sticks to it.
    assert executed == [1, 1, 1, 4, 4]


def test_boundary_suppression_lasts_exactly_one_step() -> None:
    """The guard suppresses one step, not the rest of the episode —
    sticky training is the point, the boundary is the exception."""
    guard = StickyBoundary(3, True)
    prev = np.array([5, 6, 7], dtype=np.int64)
    p_env = np.ones(3, dtype=np.float64)
    guard.mark_restart(1, prev)
    assert guard.override_rows(p_env).tolist() == [0, 2]
    guard.consume()
    assert guard.override_rows(p_env).tolist() == [0, 1, 2]


def test_restart_zeroes_the_carried_action() -> None:
    """Belt and braces: the carry slot itself is cleared at the restart,
    so no consumer downstream of the guard can replay a pre-death input."""
    guard = StickyBoundary(2, True)
    prev = np.array([5, 6], dtype=np.int64)
    guard.mark_restart(1, prev)
    assert prev.tolist() == [5, 0]


def test_disabled_guard_is_a_pure_noop() -> None:
    """Flag off = the pre-flag behavior, carry-over across the boundary
    included. Existing lineages and honest baselines must not move."""
    guard = StickyBoundary(2, False)
    prev = np.array([5, 6], dtype=np.int64)
    guard.mark_restart(1, prev)
    assert prev.tolist() == [5, 6], "disabled guard must not touch the slot"
    assert guard.override_rows(np.ones(2)).tolist() == [0, 1]
    assert _simulate_rollout(enabled=False, death_step=2) == [1, 1, 1, 1, 1]


def test_disabled_guard_draws_the_same_rng_as_the_inline_roll() -> None:
    """The roll this replaced was `np.random.random(n) < p_env`. Same
    generator, same draw count, same rows — a seeded run trained before
    the flag existed reproduces step for step."""
    p_env = np.full(6, 0.25)
    np.random.seed(1234)
    legacy = np.nonzero(np.random.random(6) < p_env)[0]
    np.random.seed(1234)
    got = StickyBoundary(6, False).override_rows(p_env)
    assert got.tolist() == legacy.tolist()
    np.random.seed(1234)
    guard_on = StickyBoundary(6, True)  # nothing marked → identical rows
    assert guard_on.override_rows(p_env).tolist() == legacy.tolist()


def test_override_rows_honors_per_env_probabilities() -> None:
    """CGSA gives each env its restart cell's noise; the guard must not
    flatten that vector."""
    guard = StickyBoundary(3, True)
    p_env = np.array([1.0, 0.0, 1.0])
    rows = guard.override_rows(p_env, rng_vals=np.array([0.5, 0.5, 0.5]))
    assert rows.tolist() == [0, 2]


def test_boundary_reset_key_is_registered() -> None:
    """A knob the trainer reads but the schema doesn't know about is the
    silent-default bug class — keep the registry honest."""
    from src.training.config_schema import (
        KNOWN_REINFORCE_KEYS,
        validate_profile,
    )

    assert "sticky_episode_boundary_reset" in KNOWN_REINFORCE_KEYS
    prof = {
        "name": "x",
        "reinforce": {
            "sticky_action_prob": 0.25,
            "sticky_episode_boundary_reset": True,
        },
    }
    assert validate_profile(prof) == []
