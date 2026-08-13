"""C5 characterization golden — the RolloutCollector anchor.

Task 0 / test C5 of `docs/proposals/trainer_decomposition_plan.md`. It is the
pre-extraction safety net for Task 5 (`RolloutCollector`, the loop's integration
point). Where C0 pins the whole-loop metric sequence and the final net, this pins
the *filled rollout buffers* the collector produces at the collector -> updater
seam, and the auto-reset-vs-freeze death branch (plan sec 4.5 — the
freeze-starvation / die-respawn-eval-inflation seam).

WHAT IS PINNED
--------------
1. The six per-step buffers the collector fills over one seeded tile+CPU
   rollout, captured non-invasively at the `batched_gae` seam (plus the raw
   pre-fold `reward_buf` captured at the `fold_intrinsic_into_rewards` boundary):
     * action_buf, done_buf, valid_buf  -> exact (integer/bool, bit-reproducible
       on any host) + a strict sha256 anchor;
     * reward_buf (RAW, the collector's product before the updater's RND fold),
       value_buf, log_prob_buf -> rel/abs 1e-6 (harmless-reassociation guard).
2. `valid_buf` semantics ("real executed step incl. death; post-done padding
   False"): the invariant `(~valid) => done`, per-env monotonic non-increasing
   validity, and — for THIS fresh stage-0 + start-state config — the die-respawn
   behavior (deaths reload inline, envs do NOT freeze, so every step is valid).
3. The auto-reset-vs-freeze branch SELECTION (plan sec 4.5): a pure replica of
   the trainer's death handler, its truth table over (a) warm-state reload,
   (b) stage-0 inline restart, (c) freeze; plus source-text anchors pinning the
   three branch conditions, their order, the `prev_completion_total[i] = 0.0`
   die-respawn re-arm, and the `valid_buf[t, i] = True` executed-step marker.

DETERMINISM BASIS
-----------------
Tile mode (`configs/mario_tiles_vanilla.yaml`) on `device=cpu` with a fixed seed
(`random`/`numpy`/`torch` + `seed=` to `Trainer`). The tile+CPU path is
bit-reproducible (validated for C0, and re-checked here: the buffer checksums are
identical across back-to-back runs). The pixel/MPS path is NOT (MPS multinomial
vs the CPU generator), which is why this golden is tile+CPU. Runtime is a few
seconds (one 32-step, 2-env iteration).

Regenerate the fixture deliberately (only when an intended behavior change is
being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_rollout_buffers.py

and review the fixture diff before committing.

EXPLICITLY NOT COVERED (documented so nobody over-trusts this net)
-----------------------------------------------------------------
  * A LIVE death does not occur within the tiny 32-step rollout at the 1-1 start
    (the agent stalls early and the value head saturates), so branches (a) and
    (c) are NOT exercised end-to-end and the pool mechanics
    (`load_worker_state` / `set_worker_done`) are NOT invoked in the recorded
    run — the pool is lazily built inside `run()` and is a native object this
    test does not wrap. The branch SELECTION is pinned by a source-anchored
    replica + truth table; the branch MECHANICS are not run against a live
    trajectory here (see `test_auto_reset_vs_freeze_branch_selection`).
  * The RAW `reward_buf` is all-zero in this window (no rewarded forward progress
    in ~2s from the start state); the biting reward signal therefore lives in the
    action/value/log_prob/done/valid buffers. A change that keeps this window
    zero is not caught by the reward pin alone.
  * The FOLDED reward (RND intrinsic added), the intrinsic/count folds and their
    obs-rms ordering are C2/C3's domain, not pinned here.
  * `final_values` (bootstrap V(s_T)), `bonus_buf` (count stream, disabled in
    this profile), `obs_buf`, the frame/tile stackers, sticky state, and the
    recurrent `h_rollout` path are not pinned here.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue as _queue
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"
_FIXTURE = ROOT / "tests" / "fixtures" / "char_rollout_buffers.json"
_TRAINER_SRC = ROOT / "src" / "training" / "trainer.py"

_HAVE_ROM = _SMB_ROM.exists() and _PROFILE.exists()

_SEED = 1234
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 200
_NUM_ACTIONS = 6  # mario_tiles_vanilla.yaml action_space length

# The six buffers the RolloutCollector fills; reward_buf here is the RAW,
# pre-fold stream (sec 2.1 — the collector's product; the updater folds RND in
# afterward). Integer/bool buffers get exact comparison, floats get a tolerance.
_INT_BUFS = ("action_buf", "done_buf", "valid_buf")
_FLOAT_BUFS = ("reward_buf", "value_buf", "log_prob_buf")


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _int_checksum(bufs: dict) -> str:
    """Strict, host-independent anchor over the integer/bool buffers."""
    h = hashlib.sha256()
    h.update(np.asarray(bufs["action_buf"], dtype=np.int32).tobytes())
    h.update(np.asarray(bufs["done_buf"], dtype=np.uint8).tobytes())
    h.update(np.asarray(bufs["valid_buf"], dtype=np.uint8).tobytes())
    return h.hexdigest()[:32]


def _capture_rollout_buffers() -> dict:
    """Run ONE real tile+CPU vanilla_ppo iteration and capture the filled
    rollout buffers at the collector -> updater seam, without touching source.

    * `fold_intrinsic_into_rewards` spy: its FIRST call receives the RAW
      reward_buf (the fold returns a new array and never mutates its input,
      so the reference stays raw) — that is the collector's own reward stream.
    * `batched_gae` spy: reaches one frame up into `PPOUpdater.update`'s locals
      to copy action/done/valid/value/log_prob as filled for this iter. (The
      fold + GAE calls were lifted from `_run_vanilla_ppo` into
      `PPOUpdater.update` by the Task-2 extraction, so both the patch target
      and the frame-up now resolve in `src/training/ppo_updater.py`.)
    Both are called exactly once per iteration.
    """
    _seed_all(_SEED)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"

    import src.training.ppo_updater as ppo_updater_mod
    from src.training.trainer import Trainer

    cap: dict = {}
    orig_fold = ppo_updater_mod.fold_intrinsic_into_rewards

    def _fold_spy(reward_buf, intrinsic, done_buf):
        cap.setdefault("reward_buf", np.array(reward_buf, copy=True))
        return orig_fold(reward_buf, intrinsic, done_buf)

    orig_gae = ppo_updater_mod.batched_gae

    def _gae_spy(reward_buf, value_buf, done_buf, *args, **kwargs):
        if "seam" not in cap:
            frame = sys._getframe(1).f_locals
            cap["seam"] = {
                name: np.array(frame[name], copy=True)
                for name in ("action_buf", "done_buf", "valid_buf",
                             "value_buf", "log_prob_buf")
            }
        return orig_gae(reward_buf, value_buf, done_buf, *args, **kwargs)

    ppo_updater_mod.fold_intrinsic_into_rewards = _fold_spy
    ppo_updater_mod.batched_gae = _gae_spy
    try:
        metrics_q: _queue.Queue = _queue.Queue()
        with tempfile.TemporaryDirectory(prefix="c5_rollout_") as tmp:
            trainer = Trainer(
                rom_path=str(_SMB_ROM),
                game_profile=profile,
                num_instances=_NUM_ENVS,
                population_size=_NUM_ENVS,
                checkpoint_dir=tmp,
                start_state_path=profile.get("start_state_path"),
                env_spec="nes_core",
                max_episode_steps=_MAX_EP_STEPS,
                metrics_queue=metrics_q,
                device_override="cpu",
                seed=_SEED,
            )
            trainer.run(num_generations=1, resume_from=None, fresh_start=True)
            rnd_built = trainer._rnd is not None
    finally:
        ppo_updater_mod.fold_intrinsic_into_rewards = orig_fold
        ppo_updater_mod.batched_gae = orig_gae

    assert rnd_built, "rnd_intrinsic_coef > 0 but RND never built — fold spy path"
    assert "reward_buf" in cap, (
        "fold_intrinsic_into_rewards never fired — the RAW reward_buf could not "
        "be captured (RND expected enabled in mario_tiles_vanilla)."
    )
    assert "seam" in cap, "batched_gae never called — the update did not run"

    seam = cap["seam"]
    out = {
        "action_buf": seam["action_buf"],
        "reward_buf": cap["reward_buf"],
        "done_buf": seam["done_buf"],
        "valid_buf": seam["valid_buf"],
        "value_buf": seam["value_buf"],
        "log_prob_buf": seam["log_prob_buf"],
    }
    return out


# Module-scoped so the (slow) real iteration runs at most once for the two
# ROM-gated tests below.
@pytest.fixture(scope="module")
def rollout_buffers() -> dict:
    if not _HAVE_ROM:
        pytest.skip("SMB ROM / mario_tiles_vanilla profile not present.")
    return _capture_rollout_buffers()


# ---------------------------------------------------------------------------
# C5.1 — the filled buffers match the recorded fixture (ROM-gated).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _HAVE_ROM, reason="SMB ROM / mario_tiles_vanilla profile not present."
)
def test_rollout_buffers_match_golden(rollout_buffers: dict) -> None:
    got = rollout_buffers

    # Shape + dtype contract of every buffer the collector hands the updater.
    exp_shape = (_ROLLOUT_STEPS, _NUM_ENVS)
    assert got["action_buf"].shape == exp_shape
    assert got["action_buf"].dtype == np.int32
    assert got["reward_buf"].shape == exp_shape
    assert got["reward_buf"].dtype == np.float32
    assert got["done_buf"].shape == exp_shape
    assert got["done_buf"].dtype == np.bool_
    assert got["valid_buf"].shape == exp_shape
    assert got["valid_buf"].dtype == np.bool_
    assert got["value_buf"].shape == exp_shape
    assert got["value_buf"].dtype == np.float32
    assert got["log_prob_buf"].shape == exp_shape
    assert got["log_prob_buf"].dtype == np.float32

    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        payload = {
            "rollout_steps": _ROLLOUT_STEPS,
            "num_envs": _NUM_ENVS,
            "int_checksum": _int_checksum(got),
            "action_buf": got["action_buf"].astype(int).tolist(),
            "done_buf": got["done_buf"].astype(int).tolist(),
            "valid_buf": got["valid_buf"].astype(int).tolist(),
            "reward_buf": got["reward_buf"].astype(float).tolist(),
            "value_buf": got["value_buf"].astype(float).tolist(),
            "log_prob_buf": got["log_prob_buf"].astype(float).tolist(),
        }
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")

    assert _FIXTURE.exists(), (
        f"golden fixture missing: {_FIXTURE}. Generate it with "
        f"CHAR_GOLDEN_REGEN=1 pytest {Path(__file__).name}"
    )
    want = json.loads(_FIXTURE.read_text())
    assert want["rollout_steps"] == _ROLLOUT_STEPS
    assert want["num_envs"] == _NUM_ENVS

    # Integer/bool buffers: exact, elementwise (host-independent bite).
    for name in _INT_BUFS:
        got_int = got[name].astype(int).tolist()
        assert got_int == want[name], (
            f"{name} diverged from the golden — the rollout's "
            f"sampling/episode logic changed. If intended, regenerate the "
            f"fixture (CHAR_GOLDEN_REGEN=1) and review the diff."
        )

    # Strict sha256 anchor over the integer/bool buffers.
    assert _int_checksum(got) == want["int_checksum"], (
        "integer-buffer checksum changed: the recorded rollout trajectory "
        "diverged. A pure structural extraction must be behavior-preserving."
    )

    # Float buffers: rel/abs 1e-6 (guards harmless reassociation; the target
    # host is bit-reproducible so this is a loose fence around exact values).
    for name in _FLOAT_BUFS:
        g = np.asarray(got[name], dtype=np.float64)
        w = np.asarray(want[name], dtype=np.float64)
        assert g.shape == w.shape, f"{name} shape drifted vs golden"
        assert g == pytest.approx(w, rel=1e-6, abs=1e-6), (
            f"{name} diverged beyond 1e-6 from the golden — the rollout's "
            f"value/log-prob/reward numerics changed. If intended, regenerate "
            f"the fixture (CHAR_GOLDEN_REGEN=1) and review the diff."
        )


# ---------------------------------------------------------------------------
# C5.2 — valid_buf semantics on the recorded rollout (ROM-gated).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _HAVE_ROM, reason="SMB ROM / mario_tiles_vanilla profile not present."
)
def test_valid_buf_semantics(rollout_buffers: dict) -> None:
    """Pin the collector -> updater `valid_buf` contract that gates advantage
    normalization and the minibatch permutation (sec 2.1)."""
    valid = rollout_buffers["valid_buf"]
    done = rollout_buffers["done_buf"]

    # Invariant 1: every NON-valid slot is post-done padding (done=True). A
    # real executed step is always valid; padding is always done.
    padding = ~valid
    assert bool(np.all(done[padding])) if padding.any() else True, (
        "a non-valid (padding) slot was not marked done — post-done padding "
        "must carry done=True so GAE masks it."
    )

    # Invariant 2: validity is monotonic non-increasing per env — once an env
    # freezes it stays frozen; a restarted env stays valid. Never valid again
    # after the first invalid step.
    for i in range(_NUM_ENVS):
        col = valid[:, i].astype(int)
        assert bool(np.all(np.diff(col) <= 0)), (
            f"env {i}: valid_buf went False then True again — validity must be "
            f"a contiguous executed prefix, never resurrected."
        )

    # This fresh stage-0 + start-state (tile) config takes the die-respawn
    # branch on death (inline restart, NOT freeze). No env freezes in the
    # recorded window, so every step is a real executed step: valid is all
    # True. This is exactly the "no freeze-starvation" behavior sec 4.5
    # protects — if a regression made envs freeze on death, padding (valid
    # False) would appear here and this bites.
    assert bool(valid.all()), (
        "recorded rollout has post-done padding — envs froze on death instead "
        "of restarting inline. That is the freeze-starvation regression sec 4.5 "
        "guards against. If this behavior change is intended, regenerate and "
        "update the branch expectation in this test deliberately."
    )
    assert not bool(done.any()), (
        "recorded rollout has a done step — the pinned no-death window changed; "
        "re-derive valid/done semantics against the new trajectory."
    )


# ---------------------------------------------------------------------------
# C5.3 — auto-reset-vs-freeze branch SELECTION (no ROM; pure logic + source).
# ---------------------------------------------------------------------------
def _death_branch(env_stage_i, smb_curriculum_states, start_bytes_present,
                  is_tile_mode) -> str:
    """Faithful replica of the death handler in `_run_vanilla_ppo` (the
    auto-reset-vs-freeze seam). Returns which branch a death takes:

        "warm_reload"    (a) — env has a warm-state for its stage
        "stage0_inline"  (b) — stage 0 with a configured tile start state
        "freeze"         (c) — else: set_worker_done, wait for iter boundary

    Mirror of, exactly:

        env_k = int(env_stage[i])
        current_stage_state_inline = (
            smb_curriculum_states[env_k]
            if 0 < env_k < len(smb_curriculum_states) else None)
        if current_stage_state_inline is not None:   # (a)
            ...
        elif _start_bytes is not None and self._is_tile_mode:  # (b)
            ...
        else:                                        # (c)
            ...
    """
    env_k = int(env_stage_i)
    current_stage_state_inline = (
        smb_curriculum_states[env_k]
        if 0 < env_k < len(smb_curriculum_states)
        else None
    )
    if current_stage_state_inline is not None:
        return "warm_reload"
    elif start_bytes_present and is_tile_mode:
        return "stage0_inline"
    else:
        return "freeze"


def test_auto_reset_vs_freeze_branch_selection() -> None:
    """Pin the branch truth table — the seam that produced freeze-starvation
    and die-respawn eval inflation (sec 4.5)."""
    # (b) — the C5 config: fresh stage 0 (env_stage 0), a configured start
    # state, tile mode. A death reloads INLINE (die-respawn), never freezes,
    # never warm-reloads (no curriculum state at stage 0).
    assert _death_branch(0, [None], True, True) == "stage0_inline"
    assert _death_branch(0, [b"base"], True, True) == "stage0_inline", (
        "stage 0 index is never a warm-reload: `0 < 0` is False, so the "
        "curriculum state at index 0 is deliberately not consulted."
    )

    # (a) — an advanced env with a warm-state for its stage reloads that state.
    assert _death_branch(2, [None, b"s1", b"s2"], True, True) == "warm_reload"
    assert _death_branch(1, [None, b"s1"], True, True) == "warm_reload"

    # (a) falls through to (b) when the stage's state is missing (None), and
    # to (c) when there is no start state / not tile.
    assert _death_branch(2, [None, b"s1", None], True, True) == "stage0_inline"
    assert _death_branch(2, [None, b"s1", None], False, True) == "freeze"

    # (c) — freeze: stage 0 with no configured start bytes, OR not tile mode.
    assert _death_branch(0, [None], False, True) == "freeze"
    assert _death_branch(0, [None], True, False) == "freeze", (
        "branch (b) requires BOTH a start state AND tile mode; a non-tile env "
        "with a start state still freezes."
    )
    # An out-of-range advanced stage (no state slot) with no start bytes freezes.
    assert _death_branch(5, [None, b"s1"], False, True) == "freeze"


def _vppo_body() -> str:
    src = _TRAINER_SRC.read_text()
    start = src.find("def _run_vanilla_ppo")
    assert start >= 0, "_run_vanilla_ppo not found"
    nxt = src.find("\n    def ", start + 1)
    assert nxt > start, "could not bound _run_vanilla_ppo body"
    return src[start:nxt]


def test_auto_reset_branch_source_anchors() -> None:
    """Anchor the three branch conditions, their order, and the die-respawn
    re-arm in the real `_run_vanilla_ppo` source, so the replica above cannot
    silently drift and a reorder/reword trips this test."""
    body = _vppo_body()

    gate = body.find("if 0 < env_k < len(smb_curriculum_states)")
    a = body.find("if current_stage_state_inline is not None:")
    b = body.find("elif _start_bytes is not None and self._is_tile_mode:")
    c = body.find("game): freeze until iter boundary.")

    assert gate >= 0, "the stage-state gate (`0 < env_k < len(...)`) is gone"
    assert a >= 0, "branch (a) warm-reload condition is gone or reworded"
    assert b >= 0, "branch (b) stage-0 inline-restart condition is gone/reworded"
    assert c >= 0, "branch (c) freeze fallback comment is gone"
    # Order: the stage gate defines the state, (a) tests it, (b) is its elif,
    # (c) is the trailing freeze. This is the exact branch order sec 4.5 says
    # must survive extraction verbatim.
    assert gate < a < b < c, (
        "the auto-reset-vs-freeze branch ORDER changed — (a) warm-reload must "
        "precede (b) inline-restart must precede (c) freeze."
    )

    # The die-respawn re-arm: prev_completion_total[i] = 0.0 after an inline
    # reload. Omitting it under-reports clears ~5-6x (sec 2.2). It appears in
    # BOTH the warm-reload and stage-0 branches.
    assert body.count("prev_completion_total[i] = 0.0") >= 2, (
        "the die-respawn clear-detector re-arm (prev_completion_total[i]=0.0) "
        "is missing from a reload branch — success rate would under-report."
    )

    # The freeze mechanic exists in the fallback path.
    assert "self.pool.set_worker_done(i, True)" in body, (
        "the freeze mechanic (set_worker_done) vanished from the death handler."
    )


def test_valid_buf_executed_step_marker_source_anchor() -> None:
    """Anchor the `valid_buf` executed-step marker and the post-done padding
    path so the semantics pinned in C5.2 stay tied to the real code."""
    body = _vppo_body()
    assert "valid_buf[t, i] = True  # real executed step (incl. death)" in body, (
        "the valid_buf executed-step marker changed — a step's validity is the "
        "collector -> updater contract that gates advantage-norm + the update."
    )
    # Post-done padding: an env already done this iter sets done_buf True and
    # `continue`s WITHOUT setting valid_buf, so the padding stays False.
    assert "if not active_in_iter[i]:" in body
    idx_pad = body.find("if not active_in_iter[i]:")
    pad_block = body[idx_pad:idx_pad + 500]
    assert "done_buf[t, i] = True" in pad_block, (
        "the post-done padding no longer marks done_buf True — GAE would "
        "bootstrap across the frozen boundary."
    )
    assert "continue" in pad_block, (
        "the post-done padding no longer short-circuits (valid_buf stays False "
        "only because this branch `continue`s before the executed-step marker)."
    )
