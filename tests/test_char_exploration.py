"""C3 exploration golden — the ExplorationController behavior anchor.

This is Task 0 / test C3 of `docs/proposals/trainer_decomposition_plan.md`:
the safety net that pins the *current* behavior of the exploration machinery
`_run_vanilla_ppo` folds into the reward stream, so the planned Task-3
extraction of an `ExplorationController` (RND + count-bonus + Go-Explore) can
be proven behavior-preserving. It must stay green before and after that move.

The three exploration mechanisms this pins, with their real owners:

  * RND intrinsic fold — `src/models/tile_rnd.py::TileRND` +
    `src/training/ppo.py::fold_intrinsic_into_rewards`. Pinned: the folded
    reward vector, the per-step bonus, and — the "RND self-referential /
    f16 double-norm" bug surface (plan §2.5) — that the RND obs-rms is
    updated EXACTLY ONCE per iteration (asserted two ways: the isolated
    Welford count delta, and a source scan of `_run_vanilla_ppo`).
  * Count bonus keying — the inline `(wl_packed << 9) | (x_pos >> 6)` map at
    `trainer.py` ~6632-6653 that fills `self._gx_counts`. Pinned: the
    `_gx_counts` dict + per-step `beta / sqrt(n)` bonus for a fixed RAM
    sequence, plus a source anchor on the exact keying/decay expressions
    (the map is inline in the loop, not yet a callable — this test is its
    pre-extraction contract).
  * Go-Explore archive — `src/training/go_explore.py::GoExploreArchive` with
    the SMB cell fn `smb_gx_phase_cell`. Pinned: cells / frontier / records /
    domination after a scripted `record()` set, and the seeded
    `select_return_states` sequence.

Determinism basis (plan §3): all three run on CPU with fixed seeds and are
bit-reproducible on the M4 target — pure-Python parts (count keying,
Go-Explore) are exact; the TileRND floats are compared at rel/abs=1e-6 as a
harmless-reassociation guard, with the Welford counts held exact.

Regenerate the fixture deliberately (only when an intended behavior change is
being adopted) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_exploration.py

and review the fixture diff before committing. Regenerate on the target with
the SMB ROM present so the ROM-gated `gx_counts_real` block is written.

Explicitly NOT covered (documented honestly so nobody over-trusts this net):

  * The full loop wiring of these mechanisms together. The vanilla profile
    (`configs/mario_tiles_vanilla.yaml`) ships count-bonus OFF
    (`gx_count_bonus_coef` unset -> 0.0) and Go-Explore OFF (no `go_explore`
    block), so the live loop only exercises the RND fold. The count-bonus
    and Go-Explore paths are therefore pinned as ISOLATED component
    behaviors here (per the C3 brief), not as loop integrations. The one
    exception is the `gx_counts_real` block, which turns count-bonus ON in a
    real seeded trainer run to exercise the true inline keying end-to-end.
  * The Go-Explore unstick BURST (arm/tick/harvest, trainer ~6888-7015) and
    the per-minibatch RND predictor-loss cache (trainer ~6082-6105) — those
    ride on rollout state this component-level net does not reconstruct.
  * The pixel-CNN `RND` path (this profile is tile mode; the pixel/MPS path
    is not bit-reproducible on the target — plan §3).
"""

from __future__ import annotations

import json
import math
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
_FIXTURE = ROOT / "tests" / "fixtures" / "char_exploration_golden.json"

_SEED = 1234
# Real smb_tiles base feature dim (175) x the profile's 4-frame stack — the
# exact `feature_dim` the trainer hands TileRND (trainer.py ~4891).
_TILE_FEATURE_DIM = 175 * 4  # 700

# Fixed shape for the isolated RND fold.
_RND_T = 8
_RND_ENVS = 2
_RND_N = _RND_T * _RND_ENVS  # rows fed to update_normalization in one fold
_RND_COEF = 0.1  # rnd_intrinsic_coef in the vanilla profile
_GX_BETA = 0.5   # count-bonus coefficient used for the isolated keying pin


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _ram(**kv: int) -> bytes:
    """A 2 KB RAM snapshot with specific bytes set. Keys are hex address
    strings (e.g. "0x0760") so int(addr, 0) auto-detects the base."""
    buf = bytearray(2048)
    for addr, val in kv.items():
        buf[int(addr, 0)] = val
    return bytes(buf)


# ---------------------------------------------------------------------------
# Count-bonus keying — replicate the inline trainer map exactly.
# trainer.py ~6632/6636/6648-6652:
#   wl_packed = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
#   x_pos     = (int(ram[0x006D]) << 8) |  int(ram[0x0086])
#   _gxk      = (wl_packed << 9) | (x_pos >> 6)
#   _gxn      = counts.get(_gxk, 0) + 1 ; bonus = beta / sqrt(_gxn)
# It is inline in the loop, not a callable — this replica IS the C3
# pre-extraction contract, source-anchored below so a change to the real
# expressions trips `test_count_key_expressions_present_in_trainer_source`.
# ---------------------------------------------------------------------------
def _gx_count_key(ram: bytes) -> int:
    wl_packed = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
    x_pos = (int(ram[0x006D]) << 8) | int(ram[0x0086])
    return (wl_packed << 9) | (x_pos >> 6)


# A fixed RAM sequence chosen so keys repeat (counts accumulate) and both the
# world/level and x components move. (world_byte, level_byte, x_page, x_lo)
_GX_RAM_SEQUENCE = [
    (0, 0, 0, 100),   # wl=0,  x_pos=100 -> bucket 1 : new
    (0, 0, 0, 110),   # wl=0,  x_pos=110 -> bucket 1 : same key, count 2
    (0, 0, 0, 200),   # wl=0,  x_pos=200 -> bucket 3 : new
    (0, 1, 0, 100),   # wl=1,  x_pos=100 -> bucket 1 : new key (level moved)
    (0, 1, 0, 100),   # wl=1,  x_pos=100 -> bucket 1 : same key, count 2
    (0, 1, 1, 0),     # wl=1,  x_pos=256 -> bucket 4 : new
    (1, 0, 0, 100),   # wl=16, x_pos=100 -> bucket 1 : new key (world moved)
    (0, 0, 0, 105),   # wl=0,  x_pos=105 -> bucket 1 : back to first key, count 3
]


def _compute_count_bonus() -> dict:
    counts: dict[int, int] = {}
    bonus_seq: list[float] = []
    for world, level, x_page, x_lo in _GX_RAM_SEQUENCE:
        ram = _ram(
            **{
                "0x075F": world,
                "0x0760": level,
                "0x006D": x_page,
                "0x0086": x_lo,
            }
        )
        key = _gx_count_key(ram)
        n = counts.get(key, 0) + 1
        counts[key] = n
        bonus_seq.append(_GX_BETA / math.sqrt(n))
    return {
        "gx_counts": {str(k): v for k, v in sorted(counts.items())},
        "bonus_seq": bonus_seq,
    }


# ---------------------------------------------------------------------------
# RND intrinsic fold — the real TileRND + fold_intrinsic_into_rewards, driven
# exactly as `_run_vanilla_ppo` drives them (trainer.py ~7288-7311).
# ---------------------------------------------------------------------------
def _compute_rnd_intrinsic() -> dict:
    from src.models.tile_rnd import TileRND
    from src.training.ppo import fold_intrinsic_into_rewards

    _seed_all(_SEED)
    rnd = TileRND(feature_dim=_TILE_FEATURE_DIM)  # consumes the seeded RNG
    rnd.eval()
    # Tile obs are int8 features (trainer feeds them .float(); tile mode does
    # NOT divide by 255 — trainer.py ~7294).
    obs = torch.randint(
        -128, 128, (_RND_N, _TILE_FEATURE_DIM), dtype=torch.int8
    ).float()

    # Fixed extrinsic reward + done mask (a couple of terminal steps so the
    # done-masked zeroing of the intrinsic fold is exercised).
    reward_buf = (np.arange(_RND_N, dtype=np.float32) * 0.01).reshape(
        _RND_T, _RND_ENVS
    )
    done_buf = np.zeros((_RND_T, _RND_ENVS), dtype=np.bool_)
    done_buf[3, 0] = True
    done_buf[6, 1] = True

    with torch.no_grad():
        intrinsic_raw = rnd(obs)                      # raw per-sample MSE
        bonus_t = rnd.normalize_bonus(intrinsic_raw)  # / reward_rms.std
        # The ONE obs-rms / reward-rms update per iteration (plan §2.5).
        rnd.update_normalization(obs, intrinsic_raw)
        intrinsic_np = (
            bonus_t.cpu().numpy().astype(np.float32) * _RND_COEF
        ).reshape(_RND_T, _RND_ENVS)
    folded = fold_intrinsic_into_rewards(reward_buf, intrinsic_np, done_buf)

    return {
        "n_rows": _RND_N,
        "intrinsic_raw_mean": float(intrinsic_raw.mean()),
        "bonus_sum": float(bonus_t.sum()),
        "intrinsic_np": intrinsic_np.tolist(),
        "reward_buf": reward_buf.tolist(),
        "folded_reward_buf": folded.tolist(),
        # Exactly-once evidence: Welford count starts at 1e-4 and grows by the
        # batch size per update, so a single N-row update lands at 1e-4 + N.
        "obs_rms_count": float(rnd.obs_rms.count),
        "reward_rms_count": float(rnd.reward_rms.count),
        "obs_rms_mean_sum": float(rnd.obs_rms.mean.sum()),
        "obs_rms_var_sum": float(rnd.obs_rms.var.sum()),
    }


# ---------------------------------------------------------------------------
# Go-Explore archive — the real GoExploreArchive + smb_gx_phase_cell, driven
# by a scripted record() set that exercises new / dominate / non-dominate /
# equal-score-fewer-steps, then a seeded selection sequence.
# ---------------------------------------------------------------------------
def _phase_ram(area: int, gx: int, y: int) -> bytes:
    """RAM snapshot smb_gx_phase_cell reads: area 0x0760, gx (0x006D<<8|0x0086),
    phase 0x0009, y 0x00CE. Phase byte held 0 so the phase component is fixed."""
    return _ram(
        **{
            "0x0760": area,
            "0x006D": (gx >> 8) & 0xFF,
            "0x0086": gx & 0xFF,
            "0x0009": 0,
            "0x00CE": y,
        }
    )


def _compute_go_explore() -> dict:
    from src.training.go_explore import GoExploreArchive, smb_gx_phase_cell

    arc = GoExploreArchive(smb_gx_phase_cell(gx_bucket=16, y_bucket=32), seed=0)

    ram_a = _phase_ram(area=1, gx=100, y=50)   # key A
    ram_b = _phase_ram(area=1, gx=200, y=50)   # key B (different gx bucket)

    # new A ; higher-score dominate ; lower-score no-op ; equal-score fewer
    # steps dominate ; new B ; equal record on B (visit, no improvement).
    arc.record(ram_a, b"A-s10", score=10.0, steps=100)
    arc.record(ram_a, b"A-s20", score=20.0, steps=200)
    arc.record(ram_a, b"A-lo", score=5.0, steps=50)
    arc.record(ram_a, b"A-fast", score=20.0, steps=120)
    arc.record(ram_b, b"B-s8", score=8.0, steps=30)
    arc.record(ram_b, b"B-again", score=8.0, steps=30)

    stats = arc.stats()
    frontier_before = arc.frontier_size()
    cells = {
        repr(k): {
            "best_score": c.best_score,
            "best_steps": c.best_steps,
            "visits": c.visits,
            "explored": c.explored,
            "times_chosen": c.times_chosen,
            "state": c.state.decode() if c.state is not None else None,
        }
        for k, c in sorted(arc.cells.items())
    }

    # Seeded return-target selection (marks cells chosen/explored). Pin the
    # exact blob sequence the frontier-weighted RNG produces.
    selection = [
        s.decode() if s is not None else None
        for s in arc.select_return_states(6)
    ]

    return {
        "stats": stats,
        "frontier_before_selection": frontier_before,
        "cells": cells,
        "selection": selection,
    }


def _compute_component_golden() -> dict:
    return {
        "count_bonus": _compute_count_bonus(),
        "rnd_intrinsic": _compute_rnd_intrinsic(),
        "go_explore": _compute_go_explore(),
    }


# ===========================================================================
# ROM-gated: real trainer run with count-bonus ON, pinning `_gx_counts`.
# This is the end-to-end bite on the inline keying (the vanilla profile ships
# it OFF, so nothing else exercises the real map).
# ===========================================================================
_GX_REAL_ROLLOUT = 32
_GX_REAL_ENVS = 2
_GX_REAL_MINIBATCH = 16
_GX_REAL_MAXEP = 200
_GX_REAL_ITERS = 2


def _run_real_gx_counts() -> dict:
    _seed_all(_SEED)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _GX_REAL_ROLLOUT
    profile["reinforce"]["ppo_minibatch_size"] = _GX_REAL_MINIBATCH
    profile["reinforce"]["device"] = "cpu"
    # Turn the otherwise-dark count-bonus path ON so the real inline keying
    # fills `_gx_counts` along a bit-reproducible tile+CPU trajectory.
    profile["reinforce"]["gx_count_bonus_coef"] = _GX_BETA

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="c3_gx_real_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=_GX_REAL_ENVS,
            population_size=_GX_REAL_ENVS,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=_GX_REAL_MAXEP,
            metrics_queue=metrics_q,
            device_override="cpu",
            seed=_SEED,
        )
        trainer.run(num_generations=_GX_REAL_ITERS, resume_from=None, fresh_start=True)
        counts = dict(trainer._gx_counts)
    return {str(k): int(v) for k, v in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Fixture load / regenerate.
# ---------------------------------------------------------------------------
def _load_or_regen() -> dict:
    got = _compute_component_golden()
    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        if _SMB_ROM.exists() and _PROFILE.exists():
            got["gx_counts_real"] = _run_real_gx_counts()
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")
    assert _FIXTURE.exists(), (
        f"golden fixture missing: {_FIXTURE}. Generate it with "
        f"CHAR_GOLDEN_REGEN=1 .venv/bin/pytest {Path(__file__).name}"
    )
    return got


def _approx(x):
    return pytest.approx(x, rel=1e-6, abs=1e-6)


def _approx_nested(seq):
    return [[_approx(v) for v in row] for row in seq]


# ---------------------------------------------------------------------------
# Count-bonus keying.
# ---------------------------------------------------------------------------
def test_count_bonus_keying_and_decay() -> None:
    """Pin `_gx_counts` and the per-step beta/sqrt(n) decay for a fixed RAM
    sequence — the count-bonus map the future ExplorationController owns."""
    got = _load_or_regen()["count_bonus"]
    want = json.loads(_FIXTURE.read_text())["count_bonus"]

    assert got["gx_counts"] == want["gx_counts"], (
        "count-bonus keying changed: the (wl_packed<<9)|(x_pos>>6) map or the "
        "count accumulation drifted."
    )
    assert len(got["bonus_seq"]) == len(want["bonus_seq"])
    for i, (g, w) in enumerate(zip(got["bonus_seq"], want["bonus_seq"])):
        assert g == _approx(w), f"count bonus step {i} changed: {g} vs {w}"

    # Independent arithmetic check (not just fixture equality): the last RAM
    # in the sequence lands on the very first key for the 3rd time, so its
    # bonus is beta/sqrt(3).
    assert got["bonus_seq"][-1] == _approx(_GX_BETA / math.sqrt(3))


def test_count_key_expressions_present_in_source() -> None:
    """Anchor the count keying + decay so the replica above cannot silently
    drift from the real code. The Task-3 extraction split these: the loop still
    computes `wl_packed`/`x_pos` in `_run_vanilla_ppo` (they feed many trackers)
    and passes them to `ExplorationController.count_bonus`, which now owns the
    `(wl_packed << 9) | (x_pos >> 6)` key and the `beta / sqrt(n)` decay."""
    tsrc = (ROOT / "src" / "training" / "trainer.py").read_text()
    start = tsrc.find("def _run_vanilla_ppo")
    end = tsrc.find("def _save_checkpoint", start)
    assert start >= 0 and end > start
    body = tsrc[start:end]
    # The world/level + x-position packing still happens in the loop.
    assert "wl_packed = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)" in body
    assert "x_pos = (int(ram[0x006D]) << 8) | int(ram[0x0086])" in body

    # The cell key + decay moved into ExplorationController.count_bonus
    # (self.-> t. form; the trainer-owned `_gx_counts` / `_gx_count_beta`
    # are read through the `t = self.trainer` alias).
    esrc = (ROOT / "src" / "training" / "exploration_controller.py").read_text()
    cstart = esrc.find("def count_bonus")
    cend = esrc.find("\n    def ", cstart + 1)
    assert cstart >= 0 and cend > cstart
    cbody = esrc[cstart:cend]
    assert "(wl_packed << 9) | (x_pos >> 6)" in cbody, (
        "the count-bonus cell key expression changed — update this "
        "characterization deliberately."
    )
    assert "t._gx_count_beta / math.sqrt(_gxn)" in cbody, (
        "the count-bonus decay (beta/sqrt(n)) changed."
    )


# ---------------------------------------------------------------------------
# RND intrinsic fold + obs-rms-updated-exactly-once.
# ---------------------------------------------------------------------------
def test_rnd_intrinsic_fold_vector() -> None:
    """Pin the folded reward + per-step intrinsic bonus the RND fold produces
    on a fixed obs batch — the vector GAE consumes (plan §2.5)."""
    got = _load_or_regen()["rnd_intrinsic"]
    want = json.loads(_FIXTURE.read_text())["rnd_intrinsic"]

    assert got["n_rows"] == want["n_rows"] == _RND_N
    assert got["intrinsic_raw_mean"] == _approx(want["intrinsic_raw_mean"])
    assert got["bonus_sum"] == _approx(want["bonus_sum"])
    assert got["intrinsic_np"] == _approx_nested(want["intrinsic_np"])
    assert got["folded_reward_buf"] == _approx_nested(want["folded_reward_buf"])
    assert got["obs_rms_mean_sum"] == _approx(want["obs_rms_mean_sum"])
    assert got["obs_rms_var_sum"] == _approx(want["obs_rms_var_sum"])


def test_rnd_fold_zeroes_intrinsic_on_done_steps() -> None:
    """The single-stream fold must add ZERO intrinsic on terminal/padded
    steps (fold_intrinsic_into_rewards masks `done`) — else advantage leaks
    onto dead transitions. Pin that folded == extrinsic exactly where done."""
    r = _compute_rnd_intrinsic()
    folded = np.asarray(r["folded_reward_buf"], dtype=np.float32)
    reward = np.asarray(r["reward_buf"], dtype=np.float32)
    done = np.zeros((_RND_T, _RND_ENVS), dtype=np.bool_)
    done[3, 0] = True
    done[6, 1] = True
    # Done steps: no intrinsic added.
    assert folded[done] == pytest.approx(reward[done])
    # Non-done steps: intrinsic strictly added (bonus is positive here).
    assert np.all(folded[~done] >= reward[~done])
    assert np.any(folded[~done] > reward[~done])


def test_rnd_obs_rms_updated_exactly_once_per_iteration() -> None:
    """Pin the plan-§2.5 contract: one iteration folds RND intrinsic with
    EXACTLY ONE obs-rms / reward-rms update. Welford count starts at 1e-4 and
    grows by the batch size per update, so a single N-row update lands at
    1e-4 + N. A second (double-norm) update would land at 1e-4 + 2N; moving
    the update into the K-epoch minibatch loop would inflate it further."""
    r = _compute_rnd_intrinsic()
    assert r["obs_rms_count"] == _approx(1e-4 + _RND_N), (
        "RND obs-rms Welford count != 1e-4 + N — update_normalization did not "
        "run exactly once on the full N-row rollout."
    )
    assert r["reward_rms_count"] == _approx(1e-4 + _RND_N)


def test_rnd_update_normalization_called_once_in_updater_source() -> None:
    """Source-scan companion to the count check: the RND intrinsic fold must
    call `update_normalization` exactly once. Task 2 relocated the fold from
    `_run_vanilla_ppo` into `PPOUpdater.update`, so the fold-once contract is
    now anchored there (the GA path's call lives in a different file/method and
    must not be counted)."""
    src = (ROOT / "src" / "training" / "ppo_updater.py").read_text()
    n = src.count(".update_normalization(")
    assert n == 1, (
        f"expected exactly one RND update_normalization call in "
        f"PPOUpdater.update, found {n} — the fold-once contract changed."
    )


# ---------------------------------------------------------------------------
# Go-Explore archive.
# ---------------------------------------------------------------------------
def test_go_explore_archive_cells_and_domination() -> None:
    """Pin the archive stats + per-cell records after the scripted record()
    set: domination (higher score, then fewer steps at equal score) and the
    frontier count the future ExplorationController owns."""
    got = _load_or_regen()["go_explore"]
    want = json.loads(_FIXTURE.read_text())["go_explore"]

    assert got["stats"] == want["stats"], (
        "Go-Explore archive stats changed (cells/frontier/records/"
        "new_cells/improvements/best_score)."
    )
    assert got["frontier_before_selection"] == want["frontier_before_selection"]
    assert got["cells"] == want["cells"], (
        "Go-Explore per-cell records changed — domination rule or state "
        "retention drifted."
    )


def test_go_explore_selection_sequence() -> None:
    """Pin the seeded return-target selection — the frontier-weighted RNG
    that drives whole-pool warm-start returns must be reproducible."""
    got = _load_or_regen()["go_explore"]["selection"]
    want = json.loads(_FIXTURE.read_text())["go_explore"]["selection"]
    assert got == want, "Go-Explore select_return_states sequence changed."


# ---------------------------------------------------------------------------
# ROM-gated end-to-end: real trainer `_gx_counts` with count-bonus ON.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_real_trainer_gx_counts_end_to_end() -> None:
    """Drive the REAL inline count-bonus keying through a short seeded
    tile+CPU trainer run (count-bonus turned ON) and pin the resulting
    `_gx_counts`. This is the end-to-end bite the isolated replica cannot
    give — the vanilla profile ships this path OFF."""
    want_all = json.loads(_FIXTURE.read_text())
    assert "gx_counts_real" in want_all, (
        "fixture has no `gx_counts_real` block — regenerate with the SMB ROM "
        "present: CHAR_GOLDEN_REGEN=1 .venv/bin/pytest "
        f"{Path(__file__).name}"
    )
    want = want_all["gx_counts_real"]
    got = _run_real_gx_counts()

    # Structural invariants of the real keying (bite even off-target): every
    # count is a positive int and every key decomposes to (wl, x_bucket).
    assert got, "real run produced an empty _gx_counts — count-bonus path dark"
    for k, v in got.items():
        assert int(v) >= 1
        assert int(k) >= 0

    # Exact trajectory pin (bit-reproducible on the target; regenerate off it).
    assert got == want, (
        "real _gx_counts diverged: the count-bonus keying or the tile+CPU "
        "trajectory changed. If intended, regenerate the fixture "
        "(CHAR_GOLDEN_REGEN=1) with the ROM present and review the diff."
    )
