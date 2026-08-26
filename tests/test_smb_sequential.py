"""Unit + integration tests for the sequential World-1 predicate (Lane 1).

The synthetic-RAM cases prove the `SequentialTracker` clear/warp/furthest
semantics and the `smb_sequential_cell` warp-admission veto without touching the
emulator. The final (slow, ROM-gated) case proves `eval_game.py --sequential`
plays THROUGH the 1-1 flag instead of stopping on the `cleared_any` latch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.training.smb_sequential import (
    SEQ_AREAS,
    LevelClearTracker,
    SequentialTracker,
    level_label,
    smb_sequential_cell,
)

_ROOT = Path(__file__).resolve().parent.parent


def mk_ram(world: int, area: int, display: int, x: int) -> bytearray:
    """Minimal RAM blob with just the bytes the predicate reads.

    `world`/`area`/`display` are the RAW bytes ($075F/$0760/$075C): World 1 == 0,
    x-4 castle display == 3.
    """
    ram = bytearray(0x800)
    ram[0x075F] = world & 0xFF
    ram[0x0760] = area & 0xFF
    ram[0x075C] = display & 0xFF
    ram[0x006D] = (x >> 8) & 0xFF
    ram[0x0086] = x & 0xFF
    return ram


# --- SequentialTracker: full castle clear ---------------------------------

def test_full_world1_clear_sets_seq_clear_no_warp() -> None:
    # 1-1 -> 1-2 entrance -> 1-2 underground -> 1-3 -> 1-4 castle -> cross into
    # World 2 with the previous displayed level being the x-4 castle (disp 3).
    trace = [
        mk_ram(0, 0, 0, 100),    # 1-1
        mk_ram(0, 0, 0, 3200),   # 1-1 flagpole region
        mk_ram(0, 1, 1, 40),     # 1-2 entrance (area 1, x-2)
        mk_ram(0, 2, 1, 2000),   # 1-2 underground (area 2, x-2)
        mk_ram(0, 3, 2, 1500),   # 1-3 (area 3, x-3)
        mk_ram(0, 4, 3, 200),    # 1-4 castle (area 4, x-4)
        mk_ram(0, 4, 3, 2560),   # 1-4 axe
        mk_ram(1, 0, 0, 40),     # World 2, 2-1 (world byte 0 -> 1)
    ]
    t = SequentialTracker()
    for ram in trace:
        t.update(ram)

    assert t.seq_clear is True
    assert t.warp_taken is False
    assert t.furthest_seq == (1, 4)
    # furthest_any advanced "beyond" into World 2.
    assert t.furthest_any is not None and t.furthest_any[0] >= 2


# --- SequentialTracker: 1-2 warp ------------------------------------------

def test_warp_out_of_1_2_is_not_a_clear() -> None:
    # 1-1 -> 1-2 entrance -> 1-2 underground -> warp pipe crossing into World 2
    # with the previous displayed level being x-2 (disp 1), NOT the castle.
    trace = [
        mk_ram(0, 0, 0, 100),    # 1-1
        mk_ram(0, 1, 1, 40),     # 1-2 entrance
        mk_ram(0, 2, 1, 2200),   # 1-2 underground (warp zone)
        mk_ram(1, 0, 0, 40),     # World 2 via warp (world byte 0 -> 1, prev disp 1)
    ]
    t = SequentialTracker()
    for ram in trace:
        t.update(ram)

    assert t.seq_clear is False
    assert t.warp_taken is True
    # Warp does NOT credit sequential progress past where Mario actually was.
    assert t.furthest_seq == (1, 2)
    # But the warp IS reported "beyond" as any-route progress.
    assert t.furthest_any is not None and t.furthest_any[0] >= 2


# --- SequentialTracker: high-water monotonicity ---------------------------

def test_furthest_seq_is_monotone_high_water_and_warp_never_raises_it() -> None:
    t = SequentialTracker()
    t.update(mk_ram(0, 0, 0, 100))
    assert t.furthest_seq == (1, 1)
    t.update(mk_ram(0, 3, 2, 500))   # jumped ahead to 1-3
    assert t.furthest_seq == (1, 3)
    t.update(mk_ram(0, 0, 0, 100))   # backtracked to 1-1: high-water holds
    assert t.furthest_seq == (1, 3)
    t.update(mk_ram(1, 0, 0, 40))    # warp to World 2: must NOT raise furthest_seq
    assert t.furthest_seq == (1, 3)
    assert t.warp_taken is True
    assert t.seq_clear is False


def test_reset_clears_all_state() -> None:
    t = SequentialTracker()
    for ram in [mk_ram(0, 4, 3, 200), mk_ram(1, 0, 0, 40)]:
        t.update(ram)
    assert t.seq_clear is True
    t.reset()
    assert t.seq_clear is False
    assert t.warp_taken is False
    assert t.furthest_seq is None
    assert t.furthest_any is None


# --- smb_sequential_cell: admission + collision ---------------------------

def test_cell_rejects_world2() -> None:
    # World 2 (world byte 1) is off the World-1 chain -> None.
    assert smb_sequential_cell(mk_ram(1, 0, 0, 100)) is None


def test_cell_rejects_nonallowlisted_area() -> None:
    # An area byte outside the sequential set (e.g. a warp room) -> None.
    assert 5 not in SEQ_AREAS
    assert smb_sequential_cell(mk_ram(0, 5, 0, 100)) is None


def test_cell_value_for_1_3() -> None:
    # 1-3 (area 3) at x=800 with the default 8-wide bucket -> (1, 3, 100).
    assert smb_sequential_cell(mk_ram(0, 3, 2, 800), x_bucket=8) == (1, 3, 100)


def test_cell_distinguishes_levels_at_same_x_page() -> None:
    # The old go-explore cell keyed on x only, so 1-1 and 1-3 at the same x
    # collided. World/area in the key makes them distinct.
    x = 800
    c11 = smb_sequential_cell(mk_ram(0, 0, 0, x))
    c13 = smb_sequential_cell(mk_ram(0, 3, 2, x))
    assert c11 is not None and c13 is not None
    assert c11 != c13
    assert c11 == (1, 0, 100)
    assert c13 == (1, 3, 100)


def test_cell_x_bucket_guard() -> None:
    # A zero bucket must not divide-by-zero (degenerate but valid).
    assert smb_sequential_cell(mk_ram(0, 0, 0, 42), x_bucket=0) == (1, 0, 42)


# --- LevelClearTracker: castle start outside World 1 ----------------------

def test_level_clear_tracker_detects_castle_clear_outside_world_1() -> None:
    # A genuine 2-4 -> 3-1 castle clear: raw world=1/area=4/disp=3 (displayed
    # 2-4) crossing into raw world=2/area=0/disp=0 (displayed 3-1). The probe
    # warm-starts inside 2-4, so `start_level` resolves to (2, 4); only the
    # general (world-agnostic) castle-clear count can score this level as
    # cleared, since `SequentialTracker.seq_clear` is hard-gated to the
    # World-1 1-4->2-1 crossing and never fires here.
    trace = [
        mk_ram(1, 4, 3, 200),    # 2-4 castle (raw world 1, area 4, disp 3)
        mk_ram(1, 4, 3, 2560),   # 2-4 axe
        mk_ram(2, 0, 0, 40),     # World 3, 3-1 (world byte 1 -> 2)
    ]
    t = LevelClearTracker()
    for ram in trace:
        t.update(ram)

    assert t.start_level == (2, 4)
    assert t.level_cleared is True


def test_level_label() -> None:
    assert level_label((1, 4)) == "1-4"
    assert level_label(None) is None


# --- eval_game.py --sequential integration (ROM-gated) --------------------

_SMB_ROM = _ROOT / "roms" / "Super Mario Bros. (World).nes"
# Reach into the main working copy for the ROM + a known 1-1-clearing
# checkpoint (both are gitignored, so absent from an isolated worktree).
_MAIN_ROOT = Path("/Users/stits/Documents/macos-emulation-and-training")
_MAIN_ROM = _MAIN_ROOT / "roms" / "Super Mario Bros. (World).nes"
_CLEAR_CKPT = _MAIN_ROOT / "checkpoints" / "winners" / "smb_1-1_greedy_clear_iter70.pt"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_SMB_ROM.exists() or _MAIN_ROM.exists()) or not _CLEAR_CKPT.exists(),
    reason="needs the SMB ROM and a 1-1-clearing checkpoint",
)
def test_eval_sequential_plays_through_1_1_flag(tmp_path: Path) -> None:
    # Run from whichever repo root actually owns the ROM + the profile's
    # relative `start_state_path` (an isolated worktree has neither); the
    # worktree's eval_game.py puts its own root on sys.path, so `src...`
    # still resolves to the code under test regardless of cwd.
    if _SMB_ROM.exists():
        run_root, rom = _ROOT, _SMB_ROM
    else:
        run_root, rom = _MAIN_ROOT, _MAIN_ROM
    out = subprocess.run(
        [
            sys.executable, str(_ROOT / "scripts" / "eval_game.py"),
            "--game", "mario", "--sequential",
            "--episodes", "2", "--max-steps", "1600",
            "--rom", str(rom),
            "--checkpoint", str(_CLEAR_CKPT),
        ],
        capture_output=True, text=True, cwd=str(run_root), timeout=600,
    )
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["status"] == "ok", result
    # The sequential keys are present.
    for key in ("seq_clear_rate", "furthest_seq_level",
                "furthest_any_level", "warp_rate"):
        assert key in result, (key, result)
    # A 1-1 policy touches the flag near step ~415; --sequential must NOT stop
    # there — it plays on into 1-2 (past the flag) until death/timeout.
    assert result["mean_length"] > 450, result
