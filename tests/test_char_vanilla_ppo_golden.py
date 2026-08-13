"""C0 master golden — the value-level safety net for the trainer decomposition.

This is Task 0 / test C0 of `docs/proposals/trainer_decomposition_plan.md`:
the whole-loop behavior anchor that every structural extraction of
`Trainer._run_vanilla_ppo` must keep green. Where
`test_vanilla_ppo_characterization.py` pins shapes/types/guards, this pins the
*exact numbers*: the full PPO metric sequence over 3 seeded tile+CPU iterations
AND a checksum of the final policy net's weights. A pure structural move
reproduces both bit-for-bit; a move that changes behavior does not.

Determinism basis (validated 2026-08-12 across 3 back-to-back runs on the M4
target — see the plan's Risk-row-1): the tile path runs on CPU and is fully
seedable, so the trajectory is bit-reproducible. The pixel/MPS path is NOT
(MPS multinomial vs the CPU generator), which is why the golden is tile+CPU.

Regenerate the fixture deliberately (only when an intended behavior change is
being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_vanilla_ppo_golden.py

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
_FIXTURE = ROOT / "tests" / "fixtures" / "char_vanilla_ppo_golden.json"

_SEED = 1234
_NITERS = 3
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 200

# Metric keys pinned (exact-value, per the determinism basis). `generation` is
# the integer iter index; the rest are the PPO losses + entropy the dashboard
# consumes and that summarize the whole update.
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


def _net_checksum(trainer) -> str:
    net = trainer._ppo_net
    assert net is not None, "policy net was never built — the loop did not run"
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _run_three_iters() -> dict:
    """Run the real 3-iteration tile+CPU loop and return the fingerprint."""
    _seed_all(_SEED)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="c0_golden_") as tmp:
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

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    seq = [
        {k: float(r[k]) for k in _SEQ_KEYS}
        for r in ppo_rows
    ]
    return {"net_checksum": checksum, "n_ppo_rows": len(ppo_rows), "seq": seq}


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_vanilla_ppo_three_iter_golden() -> None:
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

    # Metric sequence — exact (the CPU path is bit-reproducible on target; a
    # 1e-6 tolerance is used only as a harmless-reassociation guard for a
    # non-target host, where the checksum below is the strict anchor).
    assert len(got["seq"]) == len(want["seq"])
    for i, (g, w) in enumerate(zip(got["seq"], want["seq"])):
        for k in _SEQ_KEYS:
            assert g[k] == pytest.approx(w[k], rel=1e-6, abs=1e-6), (
                f"iter {i} metric {k!r}: got {g[k]!r} vs golden {w[k]!r} — "
                f"the loop's behavior changed. If intended, regenerate the "
                f"fixture (CHAR_GOLDEN_REGEN=1) and review the diff."
            )

    # Final-net checksum — the strict, whole-trajectory anchor. A pure
    # structural move must leave every weight bit-identical.
    assert got["net_checksum"] == want["net_checksum"], (
        f"final policy-net checksum changed: {got['net_checksum']} vs golden "
        f"{want['net_checksum']}. The 3-iteration trajectory diverged — a "
        f"structural extraction must be behavior-preserving. Investigate "
        f"before regenerating."
    )
