"""Characterization golden for the BACKWARD start-state curriculum.

This extends `docs/proposals/trainer_decomposition_plan.md`'s C0-C5
characterization net to a training-mode subsystem the C0-C5 goldens do NOT
exercise: `reinforce.backward_curriculum` (trainer.py ~5754, the reverse
Salimans & Chen (arXiv:1812.03381) curriculum — start near the end of a
solved tape and walk the restart cursor `tau` backward as the policy earns
each rung; the tape supplies START STATES only, never action labels). The
C0 golden's profile (`configs/mario_tiles_vanilla.yaml`) never sets
`backward_curriculum.states_dir`, so `bwd_on` is always False there and the
whole subsystem (`src/training/backward_curriculum.py`'s `TauScheduler`,
`RungBudget`, `draw_restart`) is net-uncovered by the existing goldens.

WHAT THIS PINS (outcome (a) — bit-reproducible, per the module's own
falsifier). `TauScheduler` is a pure, deterministic bookkeeping object (a
trailing-window success counter + a cursor) driven entirely by the
seeded RNG stream the rest of the tile+CPU loop already reproduces
bit-for-bit (see `test_char_vanilla_ppo_golden.py`'s determinism basis).
Enabling the curriculum adds exactly one more consumer of that same
`np.random` stream (`bwd.draw_restart`'s uniform draw) — it does not
introduce any new source of nondeterminism (MPS, thread pools, wall-clock).
Verified directly: three independent fresh-process runs of the exact
config below (seed 1234, 6 iterations) produced an IDENTICAL final policy
net checksum, an identical 6-row PPO metric sequence, AND an identical
`TauScheduler` trajectory (tau 757 -> unchanged -> unchanged -> 717 (one
real backward advance fires) -> unchanged -> unchanged). That three-way
match is `test_backward_two_run_reproducibility` below (run live, not just
asserted from memory) plus the pinned fixture comparison.

HOW THIS DIFFERS FROM `configs/mario_1_1_backward.yaml` (the production
recipe) — deliberately, to fit a sub-second characterization budget, same
spirit as C0 shrinking `rollout_steps` to 32 and `num_envs` to 2:

  * `min_attempts` 30 -> 3. At 30 (the registered gate's value) the
    trailing window never fills within 6 tiny iterations and `tau` never
    moves, which would leave "the curriculum activated but never
    ACTUALLY advanced" as the only obtainable evidence. 3 is still >= 1
    (a single lucky episode cannot fake an advance) and reliably fires a
    real `maybe_advance()` transition inside 6 iterations at this scale,
    which is the stronger, more honest proof this task asks for: not just
    "the code path ran" but "the cursor's own advance rule fired for real
    reasons under its own logic".
  * `advance_threshold` (0.2), `advance_actions` (40), `window_frames`
    (160), `entrance_weight` (1.0), `tau_init` (-1), `pin_entrance`
    (False), `count_truncations` (False) are all kept at the production
    recipe's values verbatim.
  * `num_envs` 60 -> 2, `rollout_steps` 1024 -> 32, `ppo_minibatch_size`
    256 -> 16, 3-6 iterations instead of an open-ended run. `smb_curriculum:
    false` is carried over UNCHANGED from the production recipe — it is a
    hard precondition (trainer.py's stage-0 inline-restart branch is the
    site `bwd_on` shares; with the ladder active a captured stage 1 state
    would win priority over the backward branch after the first capture).
  * `start_state_path` is pinned to the SAME tape root the minted states
    were captured from (`runs/live_show/smb_4_4_micro/entrance_start.state`
    -> `checkpoints/backward_states/1-1`, both checked into the repo) so
    the run is a real one-lineage backward curriculum, not the mismatched-
    lineage config `mario_1_1_backward.yaml`'s own header warns against.

ACTIVATION EVIDENCE (not just "the loop ran"). `TauScheduler.snapshot()`
is spied (subclassed, `monkeypatch.setattr` on the module attribute the
trainer re-imports every call) and its return value captured on every one
of the 6 per-iter `[backward] iter ...` telemetry calls (trainer.py
~7952 — unconditional, "every iter, not sampled" per its own comment).
The captured trace shows: attempts accruing (iter 2: 0 -> 1), a REAL
`maybe_advance()` transition (iter 3: tau 757 -> 717, `advances` 0 -> 1),
and the cursor holding at the new rung afterward — i.e. the mode did not
merely construct a `TauScheduler` and ignore it; the per-death record/
advance site (trainer.py ~6965-6992, shared with the CGSA/Go-Explore
inline-restart branch and reached only when `bwd_on`) executed for real.

WHAT IS EXPLICITLY NOT COVERED (honest gaps, same policy as the other C*
goldens): the recurrent/GRU path (`_recurrent_ppo_update` never touches
`bwd_on`); the backward cursor's checkpoint round-trip
(`_apply_pending_backward_state`, `TauScheduler.state_dict`/
`load_state_dict`) — that is C1/CheckpointManager territory, not this
subsystem's own determinism; `RungBudget` (opt-in `rung_step_budget`,
left unset here so it stays inert, matching a run without the feature
per the source's own "None => no episode is ever cut short" contract);
`BackwardEntropyGuard` (opt-in `entropy_guard`, same — left unset, inert);
`count_truncations=True` (the alternate truncation-as-failure convention,
untested here); the `entrance_weight`-driven entrance-vs-window draw
mix at scale (only ~a dozen total episodes run in this budget, so the
entrance branch may or may not be exercised in a given fixture regen —
harmless either way, since `draw_restart` itself is exercised regardless
of which branch it returns).

Regenerate the fixture deliberately (only when an intended behavior
change is being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_backward.py

and review the fixture diff before committing. A regen also invalidates
if `checkpoints/backward_states/1-1` is ever re-minted (a different tape
length changes `n_entries` and therefore every tau value below).
"""
from __future__ import annotations

import hashlib
import json
import os
import queue as _queue
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"
_STATES_DIR = ROOT / "checkpoints" / "backward_states" / "1-1"
_ENTRANCE_START = ROOT / "runs" / "live_show" / "smb_4_4_micro" / "entrance_start.state"
_FIXTURE = ROOT / "tests" / "fixtures" / "char_backward_golden.json"

_SEED = 1234
_NITERS = 6
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 300

_SEQ_KEYS = (
    "generation",
    "ppo_loss",
    "ppo_policy_loss",
    "ppo_value_loss",
    "ppo_entropy",
    "best_fitness",
    "avg_fitness",
)

_REQUIRED_PATHS = (_SMB_ROM, _PROFILE, _STATES_DIR, _ENTRANCE_START)
_SKIP_REASON = (
    "SMB ROM / mario_tiles_vanilla profile / minted backward_states/1-1 "
    "/ entrance_start.state not present."
)

import src.training.backward_curriculum as backward_curriculum_mod  # noqa: E402

_REAL_TAU_SCHEDULER = backward_curriculum_mod.TauScheduler


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _build_profile() -> dict:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"
    # Hard precondition (see module docstring): the stage-0 inline-restart
    # site is shared with the SMB ladder curriculum; without this a
    # captured stage-1 state could out-prioritize the backward branch.
    profile["reinforce"]["smb_curriculum"] = False
    # The enable-knob: `reinforce.backward_curriculum.states_dir` (grepped
    # from trainer.py ~5754-5761 — `bwd_on` requires `enabled` (default
    # True), `states_dir` truthy, tile mode, and a start state).
    profile["reinforce"]["backward_curriculum"] = {
        "enabled": True,
        "states_dir": str(_STATES_DIR),
        "tau_init": -1,
        "window_frames": 160,
        "advance_threshold": 0.2,
        "advance_actions": 40,
        "min_attempts": 3,  # shrunk from the production 30 — see docstring
        "entrance_weight": 1.0,
        "count_truncations": False,
        "pin_entrance": False,
    }
    # Same tape lineage the states were minted from (required for a sound
    # backward run, though not for bare activation — see docstring).
    profile["start_state_path"] = str(_ENTRANCE_START)
    return profile


def _net_checksum(trainer) -> str:
    net = trainer._ppo_net
    assert net is not None, "policy net was never built — the loop did not run"
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _run_backward_iters(monkeypatch, *, seed: int = _SEED, niters: int = _NITERS) -> dict:
    """Run the real backward-curriculum tile+CPU loop; return the
    fingerprint (net checksum, PPO metric sequence, per-iter TauScheduler
    trace) that proves both determinism and genuine activation."""
    captured_snapshots: list = []

    class _SpyTauScheduler(_REAL_TAU_SCHEDULER):  # type: ignore[misc,valid-type]
        def snapshot(self):
            snap = super().snapshot()
            captured_snapshots.append(dict(snap))
            return snap

    monkeypatch.setattr(backward_curriculum_mod, "TauScheduler", _SpyTauScheduler)

    _seed_all(seed)
    profile = _build_profile()

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="c_backward_") as tmp:
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
            seed=seed,
        )
        trainer.run(num_generations=niters, resume_from=None, fresh_start=True)
        checksum = _net_checksum(trainer)

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    seq = [{k: float(r[k]) for k in _SEQ_KEYS} for r in ppo_rows]

    assert captured_snapshots, (
        "TauScheduler.snapshot() was never called — the backward curriculum "
        "did not activate (bwd_on is False or bwd_sched is None). This is "
        "the outcome-(c) signal: check the [backward] disabled/INERT log."
    )

    return {
        "net_checksum": checksum,
        "n_ppo_rows": len(ppo_rows),
        "seq": seq,
        "bwd_snapshots": captured_snapshots,
    }


# ---------------------------------------------------------------------------
# Cheap falsifier — two independent fresh-process-state runs, same seed.
# This is the mandatory check BEFORE trusting any pinned golden below: if
# this test goes red, the golden fixture is lying and must not be trusted
# (regenerate only after finding out WHY reproducibility broke).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED_PATHS), reason=_SKIP_REASON
)
def test_backward_two_run_reproducibility(monkeypatch) -> None:
    """Same seed, two fresh runs: net checksum, PPO metric sequence, and
    the full TauScheduler per-iter trace (tau/attempts/successes/advances/
    entrance stats every iteration) must be bit-identical. Tile+CPU with a
    fixed seed is the project's established determinism basis (C0); this
    confirms the backward curriculum's one extra RNG consumer
    (`bwd.draw_restart`'s uniform draw) does not break it.
    """
    r1 = _run_backward_iters(monkeypatch, seed=_SEED, niters=_NITERS)
    r2 = _run_backward_iters(monkeypatch, seed=_SEED, niters=_NITERS)

    assert r1["net_checksum"] == r2["net_checksum"], (
        "backward-curriculum run is NOT bit-reproducible on tile+CPU "
        "(net checksum differs across two same-seed runs) — do not pin an "
        "exact golden; fall back to structural/behavioral invariants."
    )
    assert r1["seq"] == r2["seq"], (
        "PPO metric sequence differs across two same-seed backward-"
        "curriculum runs."
    )
    assert r1["bwd_snapshots"] == r2["bwd_snapshots"], (
        "TauScheduler trajectory differs across two same-seed runs — the "
        "cursor's own bookkeeping (tau/attempts/advances) is not "
        "reproducible."
    )


# ---------------------------------------------------------------------------
# Outcome (a): exact golden. Pinned only because the falsifier above (and
# a third, independent run during development — see module docstring)
# confirmed bit-reproducibility first.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED_PATHS), reason=_SKIP_REASON
)
def test_backward_curriculum_golden(monkeypatch) -> None:
    got = _run_backward_iters(monkeypatch)

    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")

    assert _FIXTURE.exists(), (
        f"golden fixture missing: {_FIXTURE}. Generate it with "
        f"CHAR_GOLDEN_REGEN=1 pytest {Path(__file__).name}"
    )
    want = json.loads(_FIXTURE.read_text())

    # --- activation: the mode ran on every iter, not just once ---
    assert got["n_ppo_rows"] == want["n_ppo_rows"] == _NITERS
    assert len(got["bwd_snapshots"]) == len(want["bwd_snapshots"]) == _NITERS, (
        "the [backward] telemetry line (and therefore the backward-"
        "curriculum branch) did not fire on every iteration."
    )

    # --- activation: a REAL advance happened under the cursor's own rule,
    # not merely that a TauScheduler object was constructed and unused ---
    tau_trace = [s["tau"] for s in got["bwd_snapshots"]]
    advances_trace = [s["advances"] for s in got["bwd_snapshots"]]
    assert tau_trace[0] > tau_trace[-1], (
        f"tau never moved backward across {_NITERS} iters ({tau_trace}) — "
        f"the curriculum activated but the cursor never advanced a rung."
    )
    assert advances_trace[-1] >= 1, "TauScheduler.maybe_advance() never fired"
    # tau only ever holds or walks backward — never forward.
    assert all(a >= b for a, b in zip(tau_trace, tau_trace[1:])), (
        f"tau moved FORWARD at some point in the trace: {tau_trace}"
    )
    # attempts were actually recorded (record()/record_entrance() ran).
    assert any(s["attempts"] > 0 or s["successes"] > 0 for s in got["bwd_snapshots"]), (
        "no attempts were ever recorded against the trailing window."
    )

    # --- exact-value golden: metric sequence ---
    assert len(got["seq"]) == len(want["seq"])
    for i, (g, w) in enumerate(zip(got["seq"], want["seq"])):
        for k in _SEQ_KEYS:
            assert g[k] == pytest.approx(w[k], rel=1e-6, abs=1e-6), (
                f"iter {i} metric {k!r}: got {g[k]!r} vs golden {w[k]!r} — "
                f"the backward-curriculum loop's behavior changed. If "
                f"intended, regenerate the fixture (CHAR_GOLDEN_REGEN=1) "
                f"and review the diff."
            )

    # --- exact-value golden: the full TauScheduler per-iter trace ---
    assert got["bwd_snapshots"] == want["bwd_snapshots"], (
        "the TauScheduler trajectory (tau/attempts/successes/advances/"
        "entrance stats per iter) diverged from the golden fixture."
    )

    # --- exact-value golden: final policy net checksum ---
    assert got["net_checksum"] == want["net_checksum"], (
        f"final policy-net checksum changed: {got['net_checksum']} vs "
        f"golden {want['net_checksum']}. Investigate before regenerating."
    )


# ---------------------------------------------------------------------------
# Source anchors — pin the enable-knob names and the shared-restart-site
# precondition so a rename/rework of the subsystem trips this file instead
# of silently going net-uncovered again.
# ---------------------------------------------------------------------------
def test_backward_curriculum_enable_knobs_present_in_trainer_source() -> None:
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert 'backward_curriculum' in src
    assert '_bwd_cfg.get("states_dir")' in src
    assert "TauScheduler(" in src
    assert "bwd_sched.maybe_advance()" in src


def test_backward_curriculum_module_has_deterministic_scheduler() -> None:
    """`TauScheduler` must stay a pure/stateless-except-fields object (no
    RNG, no wall-clock, no torch/np randomness of its own) for the golden
    above to remain a meaningful determinism anchor — the ONLY randomness
    in the backward path is the shared `np.random` stream `draw_restart`
    consumes, which the rest of the tile+CPU loop already pins."""
    src = (ROOT / "src" / "training" / "backward_curriculum.py").read_text()
    start = src.find("class TauScheduler")
    end = src.find("\nclass ", start + 1)
    body = src[start:end if end > 0 else None]
    for banned in ("random.random", "np.random", "torch.rand", "time.time"):
        assert banned not in body, (
            f"TauScheduler now uses {banned!r} — it is no longer a pure "
            f"deterministic scheduler; re-audit this golden's determinism "
            f"basis before trusting it."
        )
