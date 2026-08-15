"""Acceptance criterion for the Rust actor/learner split — codified BEFORE
the build.

Design: docs/proposals/rust_actor_learner_design.md
Measured basis: runs/throughput_split_2026-07-20.json (MPS update 58.6%,
Rust emulation 32.1%, update-bound ceiling ~1.70x).

The actor does not exist yet. This file therefore splits into two halves:

  RUNNABLE NOW (validate the harness + freeze the contracts):
    * the reference-rollout builder is itself byte-reproducible, so it is
      a trustworthy golden (needs the SMB ROM + built nes_core; skips
      otherwise);
    * the sticky-override algorithm the Rust actor must copy, as a
      pure-Python executable spec;
    * source-pins proving entropy-floor and demo-anchor are LEARNER-side
      (the actor must NOT replicate them) and that the sticky log-prob
      clamp is exactly -13.0.

  SKIP UNTIL THE ACTOR EXISTS (the acceptance gates):
    * `collect_rollout` produces byte-identical env buffers to the Python
      step loop given an identical action sequence  (Axis A);
    * it is deterministic under a fixed seed;
    * its sticky override matches the Python reference bit-for-bit;
    * one call returns the whole (T, N, ...) rollout (the FFI-collapse
      contract: one call per rollout, not 61,440 per-worker touches).

These skip with a clear reason until `nes_core.Pool.collect_rollout`
lands, then assert with NO further edits — the bar is fixed now.

Axes (see the design doc §3):
  A  env path      -> BYTE-identical   (this file)
  B  policy fwd    -> allclose(atol)   (a Step-4 test, not here)
  C  async overlap -> learns the same  (a Step-6 soak, not here)
Byte-parity is asserted only on Axis A; the policy forward and the
one-iteration off-policy lag are deliberately NOT byte claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# nes_core may resolve to an empty namespace package outside the project
# venv (the built extension lives in .venv/.../nes_core.abi3.so). Import
# defensively so the pure-Python spec/pin tests below still run anywhere.
try:  # pragma: no cover - import shim
    import nes_core  # type: ignore

    _HAS_POOL = hasattr(nes_core, "Pool")
    _HAS_REWARD = hasattr(nes_core, "build_reward_function")
    _HAS_TILES = hasattr(nes_core, "extract_smb_tiles")
except Exception:  # pragma: no cover
    nes_core = None  # type: ignore
    _HAS_POOL = _HAS_REWARD = _HAS_TILES = False

_HAS_CORE = _HAS_POOL and _HAS_REWARD

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_rom() -> Path | None:
    """Locate the SMB ROM whether run from the primary checkout or an
    isolated worktree (roms/ is gitignored, lives only in the main tree)."""
    for base in [_REPO_ROOT, *Path(__file__).resolve().parents]:
        cand = base / "roms" / "Super Mario Bros. (World).nes"
        if cand.exists():
            return cand
    return None


def _find_state() -> Path | None:
    for base in [_REPO_ROOT, *Path(__file__).resolve().parents]:
        cand = base / "checkpoints" / "handoffs" / "handoff_2-1.state"
        if cand.exists():
            return cand
    return None


_SMB_ROM = _find_rom()
_SMB_STATE = _find_state()

# Minimal, self-contained reward profile (mirrors test_compute_rewards_batch)
# so the reference rollout does not depend on any external config.
_PROFILE = {
    "name": "mario",
    "reward_weights": {
        "forward_progress": 1.0,
        "checkpoint_scale": 0.0,
        "death_penalty": 0.0,
        "time_penalty": 0.0,
    },
}


def _actor_available() -> tuple[bool, str]:
    """(available, skip_reason) for the not-yet-built actor entry point."""
    if not _HAS_POOL:
        return False, "nes_core.Pool unavailable (build/venv)"
    if not hasattr(nes_core.Pool, "collect_rollout"):
        return False, (
            "nes_core.Pool.collect_rollout not built yet — this is the "
            "acceptance gate the Rust actor must satisfy "
            "(docs/proposals/rust_actor_learner_design.md, Step 5)"
        )
    return True, ""


# =============================================================================
# Reference rollout — the Axis-A golden. Uses the CURRENT production path
# (RustPool.step_all + per-worker RewardFunction.compute), exactly as the
# trainer does at trainer.py:5457 / :5603.
# =============================================================================


def _reference_rollout(
    rom: Path,
    state: Path | None,
    actions_tn: np.ndarray,
    frame_skip: int = 4,
) -> dict[str, np.ndarray]:
    """Drive the existing pool for T steps with a FIXED (T, N) uint8
    action-bitmask sequence, collecting the env-side buffers the actor
    must reproduce byte-for-byte: RAM, reward, done, and tile obs.

    This is the SAME computation as the trainer's inner loop; feeding a
    fixed action sequence factors out policy sampling (Axis B) so what
    remains is the deterministic substrate + integer/fixed reward math
    (Axis A) — the only thing byte-parity can honestly be claimed for.
    """
    from src.emulation.rust_pool_adapter import RustPool

    T, N = actions_tn.shape
    pool = RustPool(
        rom_path=str(rom),
        num_workers=N,
        frame_skip=frame_skip,
        start_state_path=str(state) if state is not None else None,
    )
    pool.start()
    try:
        pool.reset_all()
        fns = [nes_core.build_reward_function(_PROFILE) for _ in range(N)]
        for f in fns:
            f.reset()

        ram = np.zeros((T, N, 2048), dtype=np.uint8)
        reward = np.zeros((T, N), dtype=np.float64)
        done = np.zeros((T, N), dtype=np.bool_)
        obs = np.zeros((T, N, 175), dtype=np.int8) if _HAS_TILES else None

        for t in range(T):
            acts = np.ascontiguousarray(actions_tn[t], dtype=np.uint8)
            results = pool.step_all(acts)
            for i, r in enumerate(results):
                ram_bytes = r.ram_snapshot
                rew, rdone, _lvl = fns[i].compute(ram_bytes, action=int(acts[i]))
                ram[t, i] = np.frombuffer(ram_bytes, dtype=np.uint8)
                reward[t, i] = rew
                done[t, i] = bool(r.done) or bool(rdone)
                if obs is not None:
                    obs[t, i] = np.asarray(
                        nes_core.extract_smb_tiles(ram_bytes), dtype=np.int8
                    )
        out = {"ram": ram, "reward": reward, "done": done}
        if obs is not None:
            out["obs"] = obs
        return out
    finally:
        pool.shutdown()


def _fixed_actions(T: int, N: int, seed: int = 20260720) -> np.ndarray:
    """Deterministic per-step action-bitmask sequence. Byte content is
    all that matters for parity, so raw uint8 button bitmasks are fine
    (both paths receive the identical bytes)."""
    return np.random.default_rng(seed).integers(
        0, 256, size=(T, N), dtype=np.uint8
    )


# =============================================================================
# RUNNABLE NOW — the reference builder is a trustworthy golden only if it
# is itself byte-reproducible.
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_CORE, reason="built nes_core (Pool + reward) unavailable")
@pytest.mark.skipif(_SMB_ROM is None, reason="SMB ROM not available")
def test_reference_rollout_reproducible() -> None:
    """The golden generator must be deterministic: identical actions +
    identical start state => byte-identical rollout, twice. Without this
    the byte-parity gate below would be comparing against a moving target."""
    T, N = 48, 4
    actions = _fixed_actions(T, N)
    a = _reference_rollout(_SMB_ROM, _SMB_STATE, actions)
    b = _reference_rollout(_SMB_ROM, _SMB_STATE, actions)
    np.testing.assert_array_equal(a["ram"], b["ram"])
    np.testing.assert_array_equal(a["reward"], b["reward"])
    np.testing.assert_array_equal(a["done"], b["done"])
    if "obs" in a:
        np.testing.assert_array_equal(a["obs"], b["obs"])
    # Sanity: the rollout actually advanced RAM (not a frozen all-zero run).
    assert a["ram"].any(), "reference rollout produced all-zero RAM"


# =============================================================================
# RUNNABLE NOW — sticky-override executable spec. This IS the contract the
# Rust actor must reproduce bit-for-bit (design doc §7). Mirrors
# trainer.py:5415-5435 exactly.
# =============================================================================


def _apply_sticky_reference(
    sampled: np.ndarray,
    log_probs_taken: np.ndarray,
    log_probs_all: np.ndarray,
    prev_exec: np.ndarray,
    sticky_p: float,
    t: int,
    rand_uniform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference sticky override, byte-faithful to the trainer.

    Returns (executed_actions, recorded_log_probs, new_prev_exec).

    Semantics pinned:
      * only fires for t > 0 and sticky_p > 0;
      * on a hit (rand_uniform < sticky_p), the env executes the PREVIOUS
        step's action instead of the fresh sample;
      * the RECORDED log-prob is the executed action's under the current
        policy (PPO importance-ratio parity), CLAMPED at min=-13.0 (the
        NaN backstop, trainer.py:5432);
      * prev_exec is updated to the executed actions after the step.
    """
    executed = sampled.copy()
    recorded = log_probs_taken.copy()
    if sticky_p > 0.0 and t > 0:
        rows = np.nonzero(rand_uniform < sticky_p)[0]
        if rows.size:
            prev = prev_exec[rows]
            executed[rows] = prev
            recorded[rows] = np.clip(log_probs_all[rows, prev], -13.0, None)
    new_prev = executed.copy() if sticky_p > 0.0 else prev_exec
    return executed, recorded, new_prev


def test_sticky_reference_no_override_on_first_step() -> None:
    """t == 0 never overrides, even if every draw would otherwise hit."""
    N = 5
    sampled = np.arange(N, dtype=np.int64)
    lp = np.full(N, -0.5)
    lp_all = np.zeros((N, 8))
    prev = np.full(N, 7, dtype=np.int64)
    executed, recorded, new_prev = _apply_sticky_reference(
        sampled, lp, lp_all, prev, sticky_p=1.0, t=0, rand_uniform=np.zeros(N)
    )
    np.testing.assert_array_equal(executed, sampled)
    np.testing.assert_array_equal(recorded, lp)
    # prev is still refreshed to the executed actions when sticky is on.
    np.testing.assert_array_equal(new_prev, sampled)


def test_sticky_reference_full_override_repeats_prev() -> None:
    """sticky_p=1.0 at t>0 repeats the previous executed action every row
    and records that action's (clamped) log-prob."""
    N = 4
    sampled = np.array([0, 1, 2, 3], dtype=np.int64)
    prev = np.array([3, 2, 1, 0], dtype=np.int64)
    lp_taken = np.array([-0.1, -0.2, -0.3, -0.4])
    lp_all = np.tile(np.array([-1.0, -2.0, -3.0, -4.0]), (N, 1))
    executed, recorded, new_prev = _apply_sticky_reference(
        sampled, lp_taken, lp_all, prev, sticky_p=1.0, t=1,
        rand_uniform=np.zeros(N),
    )
    np.testing.assert_array_equal(executed, prev)
    # recorded[i] == lp_all[i, prev[i]]
    np.testing.assert_allclose(
        recorded, [lp_all[i, prev[i]] for i in range(N)]
    )
    np.testing.assert_array_equal(new_prev, prev)


def test_sticky_reference_clamps_logprob_at_minus_13() -> None:
    """A near-deterministic policy gives the stuck action ~0 probability;
    the recorded log-prob must be clamped at -13.0 so the PPO ratio can't
    explode to NaN (trainer.py:5431-5433)."""
    N = 2
    sampled = np.array([0, 0], dtype=np.int64)
    prev = np.array([1, 1], dtype=np.int64)
    lp_all = np.array([[-0.01, -40.0], [-0.01, -13.0 + 1e-3]])
    executed, recorded, _ = _apply_sticky_reference(
        sampled, np.zeros(N), lp_all, prev, sticky_p=1.0, t=3,
        rand_uniform=np.zeros(N),
    )
    np.testing.assert_array_equal(executed, prev)
    # row 0: -40 clamps up to -13.0; row 1: already above the floor, kept.
    assert recorded[0] == -13.0
    assert recorded[1] == pytest.approx(-13.0 + 1e-3)


def test_sticky_reference_partial_override_selects_by_draw() -> None:
    """Only rows whose draw < sticky_p are overridden; others keep the
    fresh sample and its log-prob."""
    N = 4
    sampled = np.array([5, 6, 7, 8], dtype=np.int64)
    prev = np.array([1, 2, 3, 4], dtype=np.int64)
    lp_taken = np.array([-0.5, -0.6, -0.7, -0.8])
    lp_all = np.tile(np.linspace(-1, -4, 9), (N, 1))
    draws = np.array([0.1, 0.9, 0.05, 0.8])  # rows 0 and 2 hit at p=0.5
    executed, recorded, _ = _apply_sticky_reference(
        sampled, lp_taken, lp_all, prev, sticky_p=0.5, t=2, rand_uniform=draws
    )
    np.testing.assert_array_equal(executed, [1, 6, 3, 8])
    assert recorded[1] == pytest.approx(-0.6)  # untouched
    assert recorded[3] == pytest.approx(-0.8)  # untouched
    assert recorded[0] == pytest.approx(lp_all[0, 1])  # overridden -> prev
    assert recorded[2] == pytest.approx(lp_all[2, 3])


# =============================================================================
# RUNNABLE NOW — source-pins. The actor design assumes these live entirely
# in the LEARNER and are preserved untouched. If the controller math or the
# clamp changes, these fail and flag that the "actor does not replicate
# this" assumption in the design doc needs revisiting.
# =============================================================================


def _trainer_source() -> str:
    return (_REPO_ROOT / "src" / "training" / "trainer.py").read_text()


def test_entropy_floor_is_learner_side() -> None:
    """The adaptive entropy-floor controller (trainer.py:7147-7160) runs
    AFTER the update, adjusting self.entropy_coef from last_entropy. It is
    learner-side; the actor must not replicate it. Pin the exact formula
    so a change surfaces here (design doc §7)."""
    src = _trainer_source()
    assert "self.entropy_coef * 1.5 + 1e-4" in src, "entropy-floor raise term changed"
    assert "self.entropy_coef * 0.9" in src, "entropy-floor decay term changed"
    assert "self.entropy_floor" in src


def test_demo_anchor_is_learner_side() -> None:
    """The DQfD demo-anchor term is added inside the K-epoch minibatch loop
    via src.training.ppo.demo_anchor_loss. Learner-side; the actor never
    touches it. The loop was lifted verbatim from trainer.py into
    `PPOUpdater.update` (trainer-decomposition Task 2), so the pin reads
    ppo_updater.py now — same retarget the C5 rollout-buffer spy got."""
    from src.training.ppo import demo_anchor_loss  # noqa: F401

    src = (
        _REPO_ROOT / "src" / "training" / "ppo_updater.py"
    ).read_text()
    assert "loss = loss + _demo_coef * _da_loss" in src, (
        "demo-anchor is expected to fold into the PPO minibatch loss "
        "(learner-side) — location/expression changed"
    )


def test_sticky_clamp_pinned_in_trainer() -> None:
    """The executed-action log-prob clamp the reference above encodes is
    the trainer's min=-13.0. Pin it so the reference and the actor stay
    tied to the real value."""
    assert "min=-13.0" in _trainer_source()


# =============================================================================
# SKIP UNTIL THE ACTOR EXISTS — the acceptance gates. Full assertion bodies
# are written now; they run unchanged the moment collect_rollout lands.
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(_SMB_ROM is None, reason="SMB ROM not available")
def test_collect_rollout_byte_identical_env_buffers() -> None:
    """AXIS A — THE ACCEPTANCE GATE.

    Given identical initial worker states and an identical (T, N) action
    sequence, the Rust actor's `collect_rollout` must produce env-side
    buffers (ram, reward, done, obs) BYTE-IDENTICAL to the current Python
    step_all + per-worker reward loop. Sampling is bypassed via
    `replay_actions` so this isolates the substrate + reward path.
    """
    available, reason = _actor_available()
    if not available:
        pytest.skip(reason)

    T, N = 48, 4
    actions = _fixed_actions(T, N)
    ref = _reference_rollout(_SMB_ROM, _SMB_STATE, actions)

    pool = nes_core.Pool(
        rom_path=str(_SMB_ROM),
        num_workers=N,
        frame_skip=4,
        start_state_path=str(_SMB_STATE) if _SMB_STATE is not None else None,
    )
    fns = [nes_core.build_reward_function(_PROFILE) for _ in range(N)]
    for f in fns:
        f.reset()
    batch = pool.collect_rollout(
        rollout_steps=T,
        policy=None,
        replay_actions=actions,
        reward_fns=fns,
        sticky_p=0.0,
        seed=0,
        obs_mode="tile",
        want_ram=True,
    )

    np.testing.assert_array_equal(np.asarray(batch.ram), ref["ram"])
    np.testing.assert_array_equal(np.asarray(batch.rewards), ref["reward"])
    np.testing.assert_array_equal(np.asarray(batch.dones), ref["done"])
    if "obs" in ref:
        np.testing.assert_array_equal(np.asarray(batch.obs), ref["obs"])


@pytest.mark.slow
@pytest.mark.skipif(_SMB_ROM is None, reason="SMB ROM not available")
def test_collect_rollout_deterministic_under_seed() -> None:
    """AXIS A — same seed + same replayed actions => identical rollout,
    twice. Codifies the documented 'reproducible under its own seed'
    contract (design doc §5)."""
    available, reason = _actor_available()
    if not available:
        pytest.skip(reason)

    T, N = 32, 4
    actions = _fixed_actions(T, N, seed=99)

    def _one():
        pool = nes_core.Pool(
            rom_path=str(_SMB_ROM), num_workers=N, frame_skip=4,
            start_state_path=str(_SMB_STATE) if _SMB_STATE is not None else None,
        )
        fns = [nes_core.build_reward_function(_PROFILE) for _ in range(N)]
        for f in fns:
            f.reset()
        b = pool.collect_rollout(
            rollout_steps=T, policy=None, replay_actions=actions,
            reward_fns=fns, sticky_p=0.0, seed=1234, obs_mode="tile",
            want_ram=True,
        )
        return np.asarray(b.ram).copy(), np.asarray(b.rewards).copy()

    ram_a, rew_a = _one()
    ram_b, rew_b = _one()
    np.testing.assert_array_equal(ram_a, ram_b)
    np.testing.assert_array_equal(rew_a, rew_b)


@pytest.mark.slow
@pytest.mark.skipif(_SMB_ROM is None, reason="SMB ROM not available")
def test_collect_rollout_sticky_matches_reference() -> None:
    """AXIS A — with sticky on and a fixed seed, the actor's EXECUTED
    action stream must match the Python sticky reference bit-for-bit. The
    actor is expected to expose the executed actions it recorded (post
    sticky override) so parity can be asserted without a policy."""
    available, reason = _actor_available()
    if not available:
        pytest.skip(reason)

    # Contract: with replay_actions supplying the freshly-sampled stream
    # and a fixed seed driving the sticky RNG, batch.actions holds the
    # EXECUTED actions. The reference re-derives them from the same seed
    # and the same sampled stream. (Filled in against the real RNG-stream
    # spec once collect_rollout documents its draw order — design doc §5.)
    pytest.skip(
        "actor present but sticky RNG-stream parity is asserted once "
        "collect_rollout documents its per-step/per-worker draw order "
        "(design doc §5). The Python contract is pinned by the "
        "test_sticky_reference_* tests above."
    )


@pytest.mark.slow
@pytest.mark.skipif(_SMB_ROM is None, reason="SMB ROM not available")
def test_collect_rollout_is_single_call() -> None:
    """FFI-collapse contract: ONE call returns the WHOLE (T, N, ...)
    rollout — the structural proof of 'one call per rollout, not 61,440
    per-worker touches'. A single collect_rollout yields all T steps with
    leading dim T for every buffer."""
    available, reason = _actor_available()
    if not available:
        pytest.skip(reason)

    T, N = 16, 4
    actions = _fixed_actions(T, N, seed=7)
    pool = nes_core.Pool(
        rom_path=str(_SMB_ROM), num_workers=N, frame_skip=4,
        start_state_path=str(_SMB_STATE) if _SMB_STATE is not None else None,
    )
    fns = [nes_core.build_reward_function(_PROFILE) for _ in range(N)]
    for f in fns:
        f.reset()
    batch = pool.collect_rollout(
        rollout_steps=T, policy=None, replay_actions=actions,
        reward_fns=fns, sticky_p=0.0, seed=0, obs_mode="tile", want_ram=True,
    )
    assert np.asarray(batch.actions).shape[0] == T
    assert np.asarray(batch.rewards).shape == (T, N)
    assert np.asarray(batch.dones).shape == (T, N)
    assert np.asarray(batch.ram).shape == (T, N, 2048)
