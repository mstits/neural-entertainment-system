"""Tests for the level-scoped consolidation mode ("weld ONE level").

Three layers, mirroring the campaign's other lanes:

  * pure gate helpers + the `gate_step` sequencer — the three named behaviours
    (rollback on protect regression, accept on target improvement, terminate on
    the sustained bar) with STUBBED eval rates, no emulator;
  * `LevelClearTracker` — the per-level "cleared the level it started in"
    predicate on synthetic RAM traces (1-1/1-2/1-3/1-4 starts, warp-guarded);
  * one real smoke (slow, ROM+champion-gated): consolidate-level "1-1" from the
    tile scout champion for ~15 iters — must boot, probe, and not regress 1-1
    (it starts at 100%; the smoke proves plumbing, not learning).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.training import oneshot_curriculum as oc
from src.training.smb_sequential import LevelClearTracker
from src.training.smb_substage_ladder import build_ladder

_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================================
# Pure gate helpers
# ==========================================================================

def test_level_rungs_and_assignment():
    ladder = build_ladder()
    assert oc.level_rungs(ladder, "1-1") == [0, 1, 2]
    assert oc.level_rungs(ladder, "1-2") == [3, 4, 5, 6]
    assert oc.level_rungs(ladder, "1-4") == [10, 11, 12, 13, 14, 15, 16, 17]
    assert oc.level_rungs(ladder, "9-9") == []
    # 100% target: round-robin across the level's rungs, deterministic.
    assert list(oc.consolidate_assignment(5, [3, 4, 5, 6])) == [3, 4, 5, 6, 3]
    # No rungs -> every env cold-boots (rung 0).
    assert list(oc.consolidate_assignment(4, [])) == [0, 0, 0, 0]
    assert list(oc.consolidate_assignment(0, [3, 4])) == []


def test_protect_regressed_only_on_drop_below_baseline():
    base = {"1-1": 1.0, "1-2": 0.5}
    # Held or improved -> no regression.
    assert oc.protect_regressed(base, {"1-1": 1.0, "1-2": 0.6}) is None
    # 1-2 dropped below baseline.
    assert oc.protect_regressed(base, {"1-1": 1.0, "1-2": 0.375}) == "1-2"
    # A failed protect probe (None) is skipped, never a false rollback.
    assert oc.protect_regressed(base, {"1-1": None, "1-2": 0.5}) is None
    # An identical clear count (within tol) is not a regression.
    assert oc.protect_regressed(base, {"1-1": 1.0, "1-2": 0.5}) is None
    # Empty baselines -> never regresses.
    assert oc.protect_regressed({}, {"1-1": 0.0}) is None


def test_target_improved_and_update_sustain():
    assert oc.target_improved(0.5, 0.625) is True
    assert oc.target_improved(0.5, 0.5) is False      # tie is not improvement
    assert oc.target_improved(0.5, None) is False     # failed probe
    # Sustain increments at/above the bar, resets otherwise / on failure.
    s = 0
    for r, exp in [(0.8, 1), (0.8, 2), (0.4, 0), (0.8, 1), (None, 0)]:
        s = oc.update_sustain(s, r, 0.75)
        assert s == exp


def test_gate_step_rollback_takes_priority():
    # A protect regression rolls back regardless of a strong target rate, and
    # never accepts or terminates this probe.
    d = oc.gate_step(regressed="1-1", target_rate=1.0, best_rate=0.5,
                     sustain=2, bar=0.75, need=3)
    # The decision is unchanged by the B6 fields (which are additive and
    # inert at their defaults); a rollback also never re-bases `best_rate`.
    assert {k: d[k] for k in ("action", "sustain", "done")} == {
        "action": "rollback", "sustain": 0, "done": False,
    }
    assert d["best_rate"] == 0.5 and d["target_lb"] is None


def test_gate_step_accept_on_target_improvement():
    d = oc.gate_step(regressed=None, target_rate=0.6, best_rate=0.5,
                     sustain=0, bar=0.75, need=3)
    assert d["action"] == "accept"
    assert d["done"] is False          # 0.6 < bar, sustain resets to 0


def test_gate_step_hold_when_no_improvement():
    d = oc.gate_step(regressed=None, target_rate=0.5, best_rate=0.5,
                     sustain=0, bar=0.75, need=3)
    assert d["action"] == "hold"


def test_gate_step_terminates_on_sustained_bar():
    # Three consecutive probes at/above the bar -> done, protect healthy.
    s = 0
    done = False
    for _ in range(3):
        d = oc.gate_step(regressed=None, target_rate=0.9, best_rate=1.0,
                         sustain=s, bar=0.75, need=3)
        s, done = d["sustain"], d["done"]
        assert d["action"] == "hold"    # no improvement over best 1.0
    assert done is True and s == 3


def test_gate_step_failed_target_probe_never_terminates():
    d = oc.gate_step(regressed=None, target_rate=None, best_rate=1.0,
                     sustain=2, bar=0.75, need=3)
    assert d["action"] == "hold"
    assert d["sustain"] == 0 and d["done"] is False


# ==========================================================================
# LevelClearTracker (per-level "cleared the level it started in")
# ==========================================================================

def _ram(world: int, area: int, display: int, x: int) -> bytearray:
    ram = bytearray(0x800)
    ram[0x075F] = world & 0xFF
    ram[0x0760] = area & 0xFF
    ram[0x075C] = display & 0xFF
    ram[0x006D] = (x >> 8) & 0xFF
    ram[0x0086] = x & 0xFF
    return ram


def _play(trace):
    t = LevelClearTracker()
    for r in trace:
        t.update(r)
    return t


def test_level_clear_1_1_start_reaches_1_2():
    t = _play([_ram(0, 0, 0, 100), _ram(0, 0, 0, 3200), _ram(0, 1, 1, 40)])
    assert t.start_level == (1, 1)
    assert t.level_cleared is True
    assert t.seq_clear is True         # proxied to level_cleared


def test_level_clear_1_2_start_reaches_1_3():
    # 1-2 spans area 1 (entrance) + area 2 (underground), both displayed level 2.
    t = _play([_ram(0, 2, 1, 40), _ram(0, 2, 1, 2000), _ram(0, 3, 2, 300)])
    assert t.start_level == (1, 2)
    assert t.level_cleared is True


def test_level_clear_1_2_intra_area_is_not_a_clear():
    # Advancing area 1 -> area 2 stays inside 1-2 (displayed level 2).
    t = _play([_ram(0, 1, 1, 40), _ram(0, 2, 1, 2000)])
    assert t.start_level == (1, 2)
    assert t.level_cleared is False


def test_level_clear_1_2_warp_is_not_a_clear():
    # A warp (world byte increment out of x-2, NOT the castle) must not clear.
    t = _play([_ram(0, 2, 1, 40), _ram(1, 0, 0, 40)])
    assert t.start_level == (1, 2)
    assert t.level_cleared is False
    assert t.warp_taken is True


def test_level_clear_1_4_castle_needs_world_increment():
    cleared = _play([_ram(0, 4, 3, 200), _ram(0, 4, 3, 2560), _ram(1, 0, 0, 40)])
    assert cleared.start_level == (1, 4)
    assert cleared.level_cleared is True
    stuck = _play([_ram(0, 4, 3, 200), _ram(0, 4, 3, 900)])
    assert stuck.start_level == (1, 4)
    assert stuck.level_cleared is False


def test_level_clear_reset():
    t = _play([_ram(0, 0, 0, 100), _ram(0, 1, 1, 40)])
    assert t.level_cleared is True
    t.reset()
    assert t.start_level is None
    assert t.level_cleared is False


# ==========================================================================
# Real smoke: consolidate-level "1-1" from the tile scout champion
# ==========================================================================

def _find(rel_candidates):
    """Locate gitignored data (ROM / start state / champion) across ancestors."""
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


@pytest.mark.slow
def test_consolidate_level_1_1_smoke(tmp_path):
    """Boot + probe + weld 1-1 from the scout champion; must not regress 1-1.

    Runs the real trainer for a handful of iters in an isolated CWD (a copy of
    the champion, absolute ROM/start paths), consolidating 1-1 with 1-1 itself
    on the protect list. The champion greedily clears 1-1, so the gate should
    sustain the bar and write a DONE marker with 1-1 held at its baseline.
    """
    rom = _find(["roms/Super Mario Bros. (World).nes"])
    start = _find(["roms/Super Mario Bros. (World)_start.state.bin"])
    champ = _find(
        ["checkpoints/super_mario_bros_one_shot_tiles/vanilla_ppo_iter_01580.pt"]
    )
    if rom is None or start is None or champ is None:
        pytest.skip("needs SMB ROM + start state + the tile scout champion")

    # Build a fast smoke profile from the committed template.
    cfg = yaml.safe_load(
        (_ROOT / "configs" / "smb_consolidate_level_tiles.yaml").read_text()
    )
    cfg["start_state_path"] = str(start)          # absolute -> CWD-independent
    rl = cfg["reinforce"]
    rl["num_envs"] = 8
    rl["rollout_steps"] = 128
    rl["steps"] = 3
    rl["device"] = "cpu"                          # deterministic + CI-portable
    cl = rl["consolidate_level"]
    cl["target"] = "1-1"
    cl["protect"] = [{"level": "1-1"}]            # exercises the protect path
    cl["seed_globs"] = []                         # 1-1 needs none (cold boot)
    cl["probe"] = {"every": 4, "episodes": 2, "max_steps": 800}
    cl["accept_bar"] = 0.75
    cl["accept_probes"] = 3
    cl["cooldown"] = 10
    cl["schedule"] = {"entropy": {"from": 0.01, "to": 0.002, "iters": 40},
                      "rnd": {"from": 0.1, "to": 0.0}}

    # Isolated CWD: the champion under its derived slug so auto-resume finds it.
    from src.training.profile_utils import profile_slug
    slug = profile_slug(cfg["name"])
    ckpt_dir = tmp_path / "checkpoints" / slug
    ckpt_dir.mkdir(parents=True)
    shutil.copy2(champ, ckpt_dir / "vanilla_ppo_iter_01580.pt")

    prof = tmp_path / "smoke.yaml"
    prof.write_text(yaml.safe_dump(cfg))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "train_game.py"),
         "--game", "mario", "--profile", str(prof), "--rom", str(rom),
         "--iters", "15", "--num-envs", "8", "--seed", "0"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]

    # Plumbing proof: best_1-1.pt written, and the run probed the gate.
    assert (ckpt_dir / "best_1-1.pt").exists(), sorted(
        p.name for p in ckpt_dir.iterdir()
    )
    assert "CONSOLIDATE PROBE" in out.stderr, out.stderr[-3000:]

    # The champion clears 1-1 greedily, so the gate should terminate with a
    # DONE marker; 1-1 must be held at (not below) its mode-start baseline.
    done = ckpt_dir / "consolidate_1-1.DONE"
    assert done.exists(), (
        "expected a DONE marker (target 1-1 sustains the bar); "
        + out.stderr[-3000:]
    )
    marker = json.loads(done.read_text())
    assert marker["target"] == "1-1"
    assert marker["final_rate"] >= marker["bar"]
    base = marker["protect_baselines"].get("1-1")
    last = marker["last_protect_rates"].get("1-1")
    assert base is not None and last is not None
    assert last >= base - 1e-9, f"1-1 regressed: {last} < baseline {base}"
    # No rollback should have fired on a non-regressing weld.
    assert "CONSOLIDATE ROLLBACK" not in out.stderr, out.stderr[-3000:]
