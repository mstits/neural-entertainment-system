"""C-series golden — the CGSA-PPO subsystem (`reinforce.cgsa`).

Extends the trainer-decomposition characterization net
(`docs/proposals/trainer_decomposition_plan.md` §3) to a training-mode
subsystem the C0 master golden (`test_char_vanilla_ppo_golden.py`) does
NOT exercise: C0 runs the plain tile-vanilla path with `cgsa.enabled`
left unset, so the per-zone stochasticity-annealing block (trainer.py,
the "CGSA-PPO: Cell-Granular Stochasticity Annealing" section starting
~6117) never activates.

CGSA-PPO ("SMB 1-2 RL Consulting", 2026-07-23; see
`configs/mario_1_2_cgsa_s1.yaml`): every archived Go-Explore cell carries
its own sticky-noise probability p_sticky(c), annealed toward a target as
the policy masters the segment under that noise, with restarts drawn by
priority W(c) over the un-welded weld-frontier and cells retired
(welded) only by a Wald SPRT. Its activation gate is a hard AND, not
just its own `enabled` flag:

    cgsa_on = cgsa.enabled AND go_explore_on            (trainer.py ~6135)

`go_explore_on` itself requires `not self._smb_curriculum_active`
(trainer.py ~5665) — the SMB save-state curriculum and Go-Explore share
the same restart seam and are mutually exclusive. So enabling CGSA on
tile+CPU requires THREE profile edits on top of `mario_tiles_vanilla.yaml`:
`reinforce.smb_curriculum: false` (turns the curriculum off so
`go_explore_on` can be true), a minimal `reinforce.go_explore` block
(`enabled: true`), and a minimal `reinforce.cgsa` block (`enabled: true`).
See `configs/mario_1_2_cgsa_s1.yaml` for the real-world blocks this
config mirrors, and trainer.py:6134-6142 for the enable-knob names.

Determinism basis (the mandatory cheap falsifier, kept as a live
regression check below — `test_cgsa_two_runs_are_bit_identical`): two
independent fresh-seed 3-iteration tile+CPU runs of the config below
produced a BIT-IDENTICAL final policy-net checksum, full per-iteration
PPO/go_explore metric sequence, AND the full per-iteration CGSA internal
snapshot (per-zone stats dict, per-env restart-cell tags, per-env sticky
probabilities). Outcome (a) of the characterization brief — CGSA is
bit-reproducible on tile+CPU, same basis as C0 and C-prmdp (tile mode is
fully seedable on CPU; the pixel/MPS path is not, per the plan's
Risk-row-1). So this golden pins EXACT values, not just structural
invariants.

`cgsa.attempts_per_update` shrunk from the recipe's ~50 to 2 (a
documented deviation, same class as C0/prmdp shrinking rollout_steps):
with `attempts_per_update=50` a window needs 50 scored restarts to
close, far more than a 3-iteration/2-env smoke run produces — the
anneal/SPRT bookkeeping would never visibly execute and this golden
would only prove the archive populated, not that CGSA's own per-zone
control loop ran. At `attempts_per_update=2` (== num_envs), one window
closes at the very first iteration boundary that has a real restart
cell, which is what makes the "windows advanced" assertion meaningful
within the smoke-run horizon.

Mode activation is confirmed two ways, both pinned:
  1. `go_explore_archive` (`trainer._go_explore`) is non-empty at the
     end of the run — cells were recorded during rollout, the
     prerequisite CGSA rides on top of.
  2. the per-zone CGSA state (`cg_stats`, `cg_env_cell` — loop-locals of
     `_run_vanilla_ppo`, never exposed as `self.*` attributes, same
     situation as PR-MDP's `_adv_net`) shows: at least one non-entrance
     zone key exists, envs were actually TAGGED to a CGSA restart cell
     (`cg_env_cell` is not all `None`/unset), and that zone's stats dict
     carries `"windows" >= 1` — proof a full `attempts_per_update`-sized
     window of restarts was scored and put through the anneal/SPRT
     decision, not just that cells were archived.

Capture technique: as of this codebase revision, the trainer-decomposition
plan's Task 2 has landed (`docs/proposals/trainer_decomposition_plan.md`
§9 execution log still says "IN PROGRESS", but `PPOUpdater`
(`src/training/ppo_updater.py`) already owns the net-covered GAE/update
core; `_run_vanilla_ppo` now calls it as `_ppo_updater.update(...)`).
The old spy point other C-series tests use (patching
`trainer_mod.batched_gae`) no longer observes the protagonist path at
all — `batched_gae` is called from INSIDE `ppo_updater.py` now, against
its own separately-imported name, so patching the name bound in
`src.training.trainer`'s namespace is a silent no-op (verified: 0 spy
calls). CGSA itself was NOT moved (per the plan's Task-2 scope
correction, CGSA stays in the conductor), so `cg_stats`/`cg_env_cell`
are still `_run_vanilla_ppo` loop-locals. This module instead patches
`PPOUpdater.update` at the class level and reaches one frame up
(`sys._getframe(1)`) from inside the spy to the caller — `_run_vanilla_ppo`
itself, at its `_upd = _ppo_updater.update(...)` call site — which is
still the same technique `test_char_ppo_update.py`/
`test_char_rollout_buffers.py` established (frame-introspection on a
spied call), just re-pointed at the call site that actually exists on
this revision. Unlike PR-MDP's adversary net, CGSA has no periodic
checkpoint payload to fall back on: its `cgsa_stats.json` telemetry
write is gated at `it % 25 == 0`, far outside this smoke run's horizon.

Honest gap: within this 3-iteration/2-env horizon the per-zone sticky
probability itself (`p`) never crosses the +0.05 anneal step — the
random freshly-initialized policy's single-window success rate lands
below the 0.75 promotion bar (small-sample noise, expected — this is
exactly why the real recipe runs 350-1400+ iterations and gates on
`windows`-scale signposts, not single windows). What this golden proves
is that the CONTROL LOOP runs (tagging, scoring, window-close, SPRT
bookkeeping) end-to-end and reproducibly — not that training-scale
annealing behavior emerges in 3 iterations. Also not covered: the SPRT
weld path itself firing (`welded: True`), the priority-frontier cohort
logic beyond a single zone, and the `cgsa_stats.json` disk-persistence
path (gated at `it % 25 == 0`, never reached here).

Regenerate the fixture deliberately (only when an intended behavior
change is being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_cgsa.py

and review the fixture diff before committing.
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
_FIXTURE = ROOT / "tests" / "fixtures" / "char_cgsa_golden.json"

_SEED = 1234
_NITERS = 3
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 200

# Same protagonist keys C0 pins, plus the go_explore dashboard counters
# that are the observable, self.*-free proof go_explore (CGSA's
# prerequisite) is running every iteration.
_SEQ_KEYS = (
    "generation",
    "ppo_loss",
    "ppo_policy_loss",
    "ppo_value_loss",
    "ppo_entropy",
    "best_fitness",
    "avg_fitness",
    "go_explore_cells",
    "go_explore_frontier",
    "go_explore_records",
)


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _net_checksum(trainer) -> str:
    net = trainer._ppo_net
    assert net is not None, "policy net was never built — the loop did not run"
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _build_profile() -> dict:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"
    # Turn the SMB save-state curriculum OFF: it and Go-Explore share the
    # same restart seam and `go_explore_on` (trainer.py ~5665) requires
    # `not self._smb_curriculum_active`. Without this, cgsa_on can never
    # be True on this ("mario"-named) profile.
    profile["reinforce"]["smb_curriculum"] = False
    # Minimal go_explore block (CGSA's hard prerequisite, trainer.py
    # ~6135). Cell type mirrors configs/mario_1_2_cgsa_s1.yaml
    # (`smb_gx_phase`, 32x32 px cells, score=max_x). save_every is
    # pushed past this run's 3 iterations — archive persistence to disk
    # is not part of what this golden characterizes.
    profile["reinforce"]["go_explore"] = {
        "enabled": True,
        "inline_return_prob": 0.0,
        "cell": {"type": "smb_gx_phase", "gx_bucket": 32, "y_bucket": 32},
        "score": "max_x",
        "save_every": 1000,
    }
    # Minimal cgsa block (the real recipe's values per the task brief,
    # `attempts_per_update` shrunk — see module docstring).
    profile["reinforce"]["cgsa"] = {
        "enabled": True,
        "target_sticky": 0.25,
        "anneal_step": 0.05,
        "attempts_per_update": 2,
        "zone_px": 32,
        "cohort_k": 40,
    }
    return profile


def _run_three_iters() -> dict:
    """Run the real 3-iteration tile+CPU CGSA loop and fingerprint it.

    Spies on `PPOUpdater.update` (called once per iteration, immediately
    after the rollout+CGSA per-step bookkeeping, immediately before the
    PR-MDP adversary block) and reaches one frame up into
    `_run_vanilla_ppo`'s locals — its caller — to snapshot the CGSA
    state (`cg_stats`/`cg_env_cell`) as of that point in each iteration.
    See the module docstring's "Capture technique" note for why this
    call site, not the older `batched_gae` spy other C-series tests use,
    is the correct anchor on this revision.
    """
    _seed_all(_SEED)
    profile = _build_profile()

    import src.training.ppo_updater as ppo_updater_mod
    from src.training.trainer import Trainer

    cgsa_snapshots: list[dict] = []
    orig_update = ppo_updater_mod.PPOUpdater.update

    def _update_spy(self, **kwargs):
        fr = sys._getframe(1)
        lv = fr.f_locals
        cg_stats = lv.get("cg_stats", {})
        cgsa_snapshots.append(
            {
                "it": lv.get("it"),
                "cgsa_on": bool(lv.get("cgsa_on")),
                "cg_stats": {
                    str(key): dict(val) for key, val in sorted(
                        cg_stats.items(), key=lambda kv: str(kv[0])
                    )
                },
                "cg_env_cell": [str(c) for c in lv.get("cg_env_cell", [])],
                "sticky_p_env": [
                    float(p) for p in lv.get("_sticky_p_env", np.array([]))
                ],
            }
        )
        return orig_update(self, **kwargs)

    ppo_updater_mod.PPOUpdater.update = _update_spy
    try:
        metrics_q: _queue.Queue = _queue.Queue()
        with tempfile.TemporaryDirectory(prefix="c_cgsa_golden_") as tmp:
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
            trainer.run(num_generations=_NITERS, resume_from=None, fresh_start=True)
            checksum = _net_checksum(trainer)
            go_explore_archive = getattr(trainer, "_go_explore", None)
            ge_final_cells = (
                len(go_explore_archive) if go_explore_archive is not None else -1
            )
    finally:
        ppo_updater_mod.PPOUpdater.update = orig_update

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    seq = [{k: float(r[k]) for k in _SEQ_KEYS} for r in ppo_rows]

    return {
        "net_checksum": checksum,
        "n_ppo_rows": len(ppo_rows),
        "seq": seq,
        "go_explore_final_cells": ge_final_cells,
        "cgsa_snapshots": cgsa_snapshots,
    }


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_cgsa_activates_go_explore_and_per_zone_state() -> None:
    """Mode-activation gate: go_explore archives cells AND CGSA tags
    envs to per-zone restart cells AND runs a full anneal window through
    the scoring/SPRT bookkeeping. Independent of the fixture — this must
    hold even before/without a pinned golden. A regression here (e.g.
    `go_explore_on` false because `smb_curriculum` silently defaulted
    back on, or `cgsa_on` false because `go_explore_on` is false) means
    the subsystem this file characterizes did not run at all.
    """
    got = _run_three_iters()

    assert got["go_explore_final_cells"] > 0, (
        "go_explore archive is empty — CGSA's hard prerequisite "
        "(go_explore_on) never produced any cells"
    )
    assert len(got["cgsa_snapshots"]) == _NITERS
    assert all(s["cgsa_on"] for s in got["cgsa_snapshots"]), (
        "cgsa_on was False in at least one iteration — "
        "cgsa.enabled AND go_explore_on did not hold throughout"
    )

    # The last snapshot is the strongest: it reflects every restart+score
    # cycle that ran across all 3 iterations.
    last = got["cgsa_snapshots"][-1]
    assert any(c != "None" for c in last["cg_env_cell"]), (
        "no env was ever tagged to a CGSA restart cell "
        f"(cg_env_cell={last['cg_env_cell']!r}) — the per-zone curriculum "
        "never assigned a single restart"
    )
    non_entrance_zones = {
        k: v for k, v in last["cg_stats"].items() if k != "__ENTRANCE__"
    }
    assert non_entrance_zones, (
        "no non-entrance zone ever entered cg_stats — CGSA never scored "
        "a restart drawn from the go_explore archive"
    )
    assert any(v.get("windows", 0) >= 1 for v in non_entrance_zones.values()), (
        f"no zone completed a full attempts_per_update window "
        f"(cg_stats={non_entrance_zones!r}) — the anneal/SPRT scoring "
        f"loop never resolved a window; per-zone state exists but never "
        f"ADVANCED"
    )


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_cgsa_two_runs_are_bit_identical() -> None:
    """The mandatory cheap falsifier, kept as a live regression check.

    Two independent fresh-seed runs of the SAME 3-iteration config must
    produce identical net checksums, metric sequences, AND the full
    per-iteration CGSA internal-state trace (per-zone stats, per-env
    restart-cell tags, per-env sticky probabilities). This is what
    justifies pinning an EXACT golden below (outcome (a) of the
    characterization brief) rather than only structural invariants
    (outcome (b)).
    """
    r1 = _run_three_iters()
    r2 = _run_three_iters()

    assert r1["net_checksum"] == r2["net_checksum"]
    assert r1["seq"] == r2["seq"]
    assert r1["go_explore_final_cells"] == r2["go_explore_final_cells"]
    assert r1["cgsa_snapshots"] == r2["cgsa_snapshots"], (
        "CGSA's per-zone internal state diverged between two identically-"
        "seeded runs — the priority-sample draws "
        "(np.random.random()/np.random.randint inside _cgsa_select_state) "
        "are not reproducing bit-for-bit."
    )


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_cgsa_three_iter_golden() -> None:
    got = _run_three_iters()

    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")

    assert _FIXTURE.exists(), (
        f"golden fixture missing: {_FIXTURE}. Generate it with "
        f"CHAR_GOLDEN_REGEN=1 pytest {Path(__file__).name}"
    )
    want = json.loads(_FIXTURE.read_text())

    # Iteration count first (a changed loop length is the loudest signal).
    assert got["n_ppo_rows"] == want["n_ppo_rows"] == _NITERS, (
        f"expected {_NITERS} PPO rows, got {got['n_ppo_rows']} "
        f"(fixture {want['n_ppo_rows']})"
    )

    # Metric sequence — exact (tile+CPU is bit-reproducible on target;
    # the 1e-6 tolerance is a harmless-reassociation guard only, the
    # checksum below is the strict anchor).
    assert len(got["seq"]) == len(want["seq"])
    for i, (g, w) in enumerate(zip(got["seq"], want["seq"])):
        for k in _SEQ_KEYS:
            assert g[k] == pytest.approx(w[k], rel=1e-6, abs=1e-6), (
                f"iter {i} metric {k!r}: got {g[k]!r} vs golden {w[k]!r} — "
                f"the CGSA loop's behavior changed. If intended, "
                f"regenerate the fixture (CHAR_GOLDEN_REGEN=1) and review "
                f"the diff."
            )

    # Final-net checksum — the strict, whole-trajectory anchor.
    assert got["net_checksum"] == want["net_checksum"], (
        f"final policy-net checksum changed: {got['net_checksum']} vs "
        f"golden {want['net_checksum']}. The 3-iteration trajectory "
        f"diverged. Investigate before regenerating."
    )

    assert got["go_explore_final_cells"] == want["go_explore_final_cells"]

    # The full per-iteration CGSA internal-state trace — exact. This is
    # the subsystem-specific anchor (net checksum + PPO metrics alone
    # would pass even if CGSA's own bookkeeping silently broke, as long
    # as it didn't crash and didn't perturb the executed actions'
    # log-probs).
    assert got["cgsa_snapshots"] == want["cgsa_snapshots"], (
        "CGSA per-zone internal state (cg_stats / cg_env_cell / "
        "sticky_p_env) diverged from the golden. If intended, "
        "regenerate the fixture (CHAR_GOLDEN_REGEN=1) and review the "
        "diff."
    )
