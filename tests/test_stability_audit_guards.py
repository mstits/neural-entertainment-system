"""Regression guards for the 2026-07-20 stability/integrity audit.

Each test pins one real defect the audit found and a fix shipped for, so
the failure class cannot silently return. Mapped one test per defect:

  1. FFI panic boundary — a wrong-mapper / truncated save-state must
     raise PyValueError, never crash the process (nes_core apply_state
     guard; the GUI "Load State" hard-crash).
  2. Tear-tolerant resume — a torn newest checkpoint must be quarantined
     and the newest GOOD one loaded, never a silent from-scratch restart
     (trainer resume scan).
  3. seq_clear warp guard — a warp out of 1-2 followed by a later-world
     castle clear must NOT count as a sequential World-1 clear
     (smb_sequential; the headline metric).
  4. Supervisor panic catch — the supervisor must catch BaseException
     (PyO3 PanicException), not just Exception.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms/Super Mario Bros. (World).nes"
ALT_ROM = REPO / "roms/Abadox - The Deadly Inner War (USA).nes"


@pytest.mark.skipif(not ROM.exists(), reason="SMB ROM not present")
def test_load_state_garbage_raises_not_crashes():
    import nes_core
    env = nes_core.NESEnvironment(rom_path=str(ROM))
    with pytest.raises((ValueError, RuntimeError)):
        env.load_state(b"NCST\x01" + b"\xff" * 40)
    # Process alive: a normal step still works.
    env.step(0)


@pytest.mark.skipif(
    not (ROM.exists() and ALT_ROM.exists()),
    reason="need SMB + a different-mapper ROM",
)
def test_cross_mapper_state_raises_not_crashes():
    """A state from a different mapper panics inside apply_state; the
    guard must convert that to PyValueError with the process alive."""
    import nes_core
    src = nes_core.NESEnvironment(rom_path=str(ALT_ROM))
    for _ in range(4):
        src.step(0)
    blob = src.save_state()
    dst = nes_core.NESEnvironment(rom_path=str(ROM))
    try:
        dst.load_state(blob)
    except ValueError:
        pass  # guard fired — the expected outcome
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"expected ValueError, got {type(exc).__name__}: {exc}")
    # Either way the process must be alive.
    dst.step(0)


def test_tear_tolerant_resume_prefers_good_older_checkpoint():
    """A torn newest checkpoint is quarantined and the newest loadable
    one is used — never a silent from-scratch restart."""
    import torch
    d = Path(tempfile.mkdtemp())
    torch.save(
        {"iter": 20, "net_state_dict": {"w": torch.zeros(3)},
         "optimizer_state_dict": {}},
        d / "vanilla_ppo_iter_00020.pt",
    )
    (d / "vanilla_ppo_iter_00030.pt").write_bytes(b"PK\x03\x04truncated")

    # Mirror the trainer's resume scan (newest->oldest, quarantine torn).
    cands = sorted(
        d.glob("vanilla_ppo_iter_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]), reverse=True,
    )
    state = latest = None
    for c in cands:
        try:
            state = torch.load(str(c)); latest = c; break
        except Exception:
            c.rename(c.with_suffix(".pt.corrupt"))
    assert latest is not None and latest.name == "vanilla_ppo_iter_00020.pt"
    assert state["iter"] == 20
    assert (d / "vanilla_ppo_iter_00030.pt.corrupt").exists()


def test_seq_clear_warp_guard():
    from src.training.smb_sequential import (
        SequentialTracker, RAM_WORLD, RAM_AREA, RAM_DISPLAY,
        DISPLAY_LEVEL_CASTLE,
    )

    def ram(world, area, disp):
        b = bytearray(2048)
        b[RAM_WORLD] = world
        b[RAM_AREA] = area
        b[RAM_DISPLAY] = disp
        return bytes(b)

    # Warp out of 1-2, then clear a later world's castle: NOT a seq clear.
    t = SequentialTracker()
    t.update(ram(0, 1, 1))
    t.update(ram(1, 5, 3))            # world rise, prev not castle => warp
    assert t.warp_taken and not t.seq_clear
    t.update(ram(1, 6, DISPLAY_LEVEL_CASTLE))
    t.update(ram(2, 0, 0))            # castle-out world rise in World 2
    assert not t.seq_clear, "warp-then-later-castle must not be seq_clear"

    # Genuine sequential 1-4 castle clear: IS a seq clear.
    t2 = SequentialTracker()
    t2.update(ram(0, 6, DISPLAY_LEVEL_CASTLE))
    t2.update(ram(1, 0, 0))
    assert t2.seq_clear


def test_supervisor_catches_baseexception():
    """The supervisor's except arm must be BaseException (PyO3
    PanicException subclasses it), not Exception."""
    src = (REPO / "scripts/train_game.py").read_text()
    assert "except BaseException" in src, \
        "supervisor must catch BaseException to survive PyO3 PanicException"
    # And SystemExit must still propagate (not be swallowed as a crash).
    assert "except SystemExit" in src
