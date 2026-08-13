"""C-series golden — the PR-MDP adversary subsystem (`reinforce.prmdp`).

Extends the trainer-decomposition characterization net
(`docs/proposals/trainer_decomposition_plan.md` §3) to a training-mode
subsystem the C0 master golden (`test_char_vanilla_ppo_golden.py`) does
NOT exercise: C0 runs the plain tile-vanilla path with `prmdp.enabled`
left unset, so the adversary net/GAE/update block (trainer.py, the
"PR-MDP: Probabilistic Action Robust MDP" section starting ~6035 and the
"PR-MDP ADVERSARY UPDATE" block ~7584) never activates. This module
pins that subsystem instead.

PR-MDP (Tessler, Efroni & Mannor, ICML 2019): a co-trained adversary net
forwards on the same rollout observations; on a Bernoulli(alpha) draw
per env per step its sampled action overrides the protagonist's, and the
adversary is trained by PPO on the SAME rollout with NEGATED protagonist
rewards, restricted to the steps its action executed (zero-sum). See
`configs/mario_1_2_prmdp.yaml` for the real-world block this config
mirrors, and trainer.py:6053-6085 for the enable-knob names.

Determinism basis (validated here before pinning anything, the mandatory
cheap falsifier): TWO independent fresh-seed 11-iteration tile+CPU runs
of the config below produced a BIT-IDENTICAL final protagonist-net
checksum, adversary-net checksum, and full per-iteration metric
sequence. Outcome (a) of the task brief — the mode is bit-reproducible
on tile+CPU, same basis as C0 (tile mode is fully seedable on CPU; the
pixel/MPS path is not, per the plan's Risk-row-1). So this golden pins
EXACT values, not just structural invariants.

Why 11 iterations, not 3 like C0: the adversary net + optimizer are only
persisted into the periodic vanilla_ppo checkpoint
(`vanilla_ppo_iter_{global_it:05d}.pt`), written every 10 iterations
(`if it > 0 and it % 10 == 0` in `CheckpointManager.save_iter`,
src/training/checkpoint_manager.py). Fewer iterations would leave this
golden unable to prove the adversary net was actually built AND trained
(as opposed to merely constructed and left untouched) from outside the
trainer — `_adv_net`/`_adv_opt` are loop-locals of `_run_vanilla_ppo`,
never exposed as `self.*` attributes, so the checkpoint payload is the
only externally-observable artifact of the adversary's existence.
Runtime cost is still small: ~0.11s/iter tile+CPU with num_envs=2,
rollout_steps=32 (~1.6s total for 11 iterations, measured on the M4
target), well under the suite's budget.

Mode activation is confirmed two ways, both pinned:
  1. the periodic checkpoint at iter 10 carries a
     `prmdp_adv_net_state_dict` (the adversary net was constructed —
     `_adv_is_net = prmdp_on and adversary_mode == "net"`, trainer.py
     ~6067) with the expected parameter count for this tile-MLP
     architecture;
  2. its paired `prmdp_adv_optimizer_state_dict`'s Adam `state` is
     non-empty and every tracked parameter's `step` counter is >= 1 —
     i.e. `_adv_opt.step()` (trainer.py ~7645, inside the "PR-MDP
     ADVERSARY UPDATE" block) actually ran, not just that the net was
     allocated and left untouched. This is `self._adv_net is not None`
     restated through the one channel that survives past `run()`
     returning (the brief's suggested `self._adv_net` attribute does
     not exist on `Trainer` — `_adv_net` is a loop-local of
     `_run_vanilla_ppo` — so the checkpoint payload is the load-bearing
     proxy for it here).

Regenerate the fixture deliberately (only when an intended behavior
change is being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_prmdp.py

and review the fixture diff before committing.
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
_FIXTURE = ROOT / "tests" / "fixtures" / "char_prmdp_golden.json"

_SEED = 1234
_NITERS = 11  # >= 11 so the it%10==0 checkpoint (carrying the adversary
              # net/optimizer) fires exactly once, at global_it=10.
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 200
_CKPT_ITER = 10

# Same keys C0 pins for the protagonist trajectory (the PR-MDP mode must
# not perturb the shape of the standard metrics contract).
_SEQ_KEYS = (
    "generation",
    "ppo_loss",
    "ppo_policy_loss",
    "ppo_value_loss",
    "ppo_entropy",
    "best_fitness",
    "avg_fitness",
)


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _state_dict_checksum(sd: dict) -> str:
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _build_profile() -> dict:
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"
    # PR-MDP training REPLACES random sticky with the alpha-mixture (see
    # configs/mario_1_2_prmdp.yaml's header) — force it off so the two
    # noise sources cannot interact and the golden isolates the PR-MDP
    # subsystem cleanly.
    profile["reinforce"]["sticky_action_prob"] = 0.0
    # The real config block (configs/mario_1_2_prmdp.yaml `reinforce.prmdp`),
    # shrunk to nothing except the tile+CPU determinism overrides above.
    profile["reinforce"]["prmdp"] = {
        "enabled": True,
        "alpha": 0.25,
        "adversary_epochs": 2,
        "adversary_clip": 0.1,
        "adversary_entropy_coef": 0.1,
        "adversary_lr": 2.5e-4,
        "adversary_mode": "net",
    }
    return profile


def _run_eleven_iters() -> dict:
    """Run the real 11-iteration tile+CPU PR-MDP loop and fingerprint it."""
    _seed_all(_SEED)
    profile = _build_profile()

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="c_prmdp_golden_") as tmp:
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

        protagonist_checksum = _state_dict_checksum(trainer._ppo_net.state_dict())

        ckpt_path = Path(tmp) / f"vanilla_ppo_iter_{_CKPT_ITER:05d}.pt"
        assert ckpt_path.exists(), (
            f"expected periodic checkpoint at iter {_CKPT_ITER} ({ckpt_path}) "
            f"— the PR-MDP adversary is only observable through this "
            f"payload (it is never a self.* attribute); no checkpoint means "
            f"activation cannot be confirmed at all."
        )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    seq = [{k: float(r[k]) for k in _SEQ_KEYS} for r in ppo_rows]

    assert "prmdp_adv_net_state_dict" in payload, (
        "checkpoint has no prmdp_adv_net_state_dict — the adversary net "
        "was never constructed (_adv_is_net false?), activation FAILED"
    )
    assert "prmdp_adv_optimizer_state_dict" in payload, (
        "checkpoint has no prmdp_adv_optimizer_state_dict — the adversary "
        "net exists but its optimizer was never persisted"
    )
    adv_net_sd = payload["prmdp_adv_net_state_dict"]
    adv_opt_sd = payload["prmdp_adv_optimizer_state_dict"]
    adv_opt_state = adv_opt_sd.get("state", {})

    return {
        "n_ppo_rows": len(ppo_rows),
        "seq": seq,
        "protagonist_checksum": protagonist_checksum,
        "adv_net_checksum": _state_dict_checksum(adv_net_sd),
        "adv_param_count": sum(v.numel() for v in adv_net_sd.values()),
        "adv_opt_state_entries": len(adv_opt_state),
        "adv_opt_min_step": (
            min(int(v["step"]) for v in adv_opt_state.values())
            if adv_opt_state
            else 0
        ),
    }


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_prmdp_adversary_activates_and_trains() -> None:
    """Mode-activation gate: the adversary net exists AND was trained.

    Independent of the fixture — this must hold even before/without a
    pinned golden, and is the concrete stand-in for "self._adv_net is
    not None" (which does not exist as an attribute; see module
    docstring). A regression here (e.g. `_adv_is_net` false, or the
    adversary update block silently skipped because `valid_indices` is
    empty) means the subsystem this file characterizes did not run.
    """
    got = _run_eleven_iters()

    assert got["adv_param_count"] > 0, "adversary net has zero parameters"
    assert got["adv_opt_state_entries"] > 0, (
        "adversary optimizer has no tracked parameter state — "
        "_adv_opt.step() never ran (PPO update block did not execute)"
    )
    assert got["adv_opt_min_step"] >= 1, (
        f"adversary optimizer step counter is {got['adv_opt_min_step']} for "
        f"its least-stepped parameter — expected >= 1 Adam step from the "
        f"PR-MDP ADVERSARY UPDATE block (trainer.py ~7591) running across "
        f"iterations 1-10"
    )


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_prmdp_two_runs_are_bit_identical() -> None:
    """The mandatory cheap falsifier, kept as a live regression check.

    Two independent fresh-seed runs of the SAME 11-iteration config must
    produce identical protagonist + adversary checksums and metric
    sequences. This is what justifies pinning an EXACT golden below
    (outcome (a) of the characterization brief) rather than only
    structural invariants (outcome (b)). If this test ever goes red
    while the exact-golden test also goes red with a DIFFERENT got-value
    each run, that is the signal the subsystem quietly picked up
    non-determinism and the golden strategy here needs to change.
    """
    r1 = _run_eleven_iters()
    r2 = _run_eleven_iters()

    assert r1["protagonist_checksum"] == r2["protagonist_checksum"]
    assert r1["adv_net_checksum"] == r2["adv_net_checksum"]
    assert r1["seq"] == r2["seq"]


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_prmdp_eleven_iter_golden() -> None:
    got = _run_eleven_iters()

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

    # Protagonist metric sequence — exact (tile+CPU is bit-reproducible on
    # target; the 1e-6 tolerance is a harmless-reassociation guard only,
    # the checksums below are the strict anchors).
    assert len(got["seq"]) == len(want["seq"])
    for i, (g, w) in enumerate(zip(got["seq"], want["seq"])):
        for k in _SEQ_KEYS:
            assert g[k] == pytest.approx(w[k], rel=1e-6, abs=1e-6), (
                f"iter {i} metric {k!r}: got {g[k]!r} vs golden {w[k]!r} — "
                f"the PR-MDP loop's protagonist behavior changed. If "
                f"intended, regenerate the fixture (CHAR_GOLDEN_REGEN=1) "
                f"and review the diff."
            )

    # Protagonist final-net checksum.
    assert got["protagonist_checksum"] == want["protagonist_checksum"], (
        f"protagonist net checksum changed: {got['protagonist_checksum']} "
        f"vs golden {want['protagonist_checksum']}. Investigate before "
        f"regenerating."
    )

    # Adversary net checksum at the iter-10 checkpoint — proves the
    # adversary's TRAINED weights (not just its architecture) are stable.
    assert got["adv_net_checksum"] == want["adv_net_checksum"], (
        f"adversary net checksum changed: {got['adv_net_checksum']} vs "
        f"golden {want['adv_net_checksum']}. The adversary's 10-iteration "
        f"training trajectory diverged. Investigate before regenerating."
    )
    assert got["adv_param_count"] == want["adv_param_count"]

    # Activation proof, pinned so a golden regen can't silently paper over
    # the adversary quietly stopping training (e.g. valid_indices going
    # empty under some future refactor).
    assert got["adv_opt_state_entries"] == want["adv_opt_state_entries"]
    assert got["adv_opt_min_step"] >= 1
