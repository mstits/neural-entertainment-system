"""Regression test for scripts/emit_fm2.py's actions.npy interpretation.

emit_fm2.py's only real-world input is a solver "solution tape"
(`*.actions.npy`) — every producer of these files (go_explore_solve.py,
documented as the repo-wide convention in src/training/tape_replay.py,
consumed by assemble_full_run.py and tape_replay.py the same way) stores
per-step ACTION-SPACE INDICES, not raw RLDUTSBA controller bitmasks.
Treating an index as a bitmask silently renders the wrong buttons. This
pins the fix against the exact 1942 action_space entries the defect was
found against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import scripts.emit_fm2 as emit_fm2

REPO = Path(__file__).resolve().parent.parent
PROFILE = REPO / "configs" / "1942.yaml"


def _pad_rows(fm2_text: str) -> list[str]:
    """Pull the 8-char pad field out of each `|0|PAD|||` input row."""
    return [
        line.split("|")[2]
        for line in fm2_text.splitlines()
        if line.startswith("|0|")
    ]


def test_actions_npy_indices_resolve_through_action_space(tmp_path, monkeypatch):
    """Index 7 (["right", "A"]) and index 11 (["right", "A", "B"]) in
    1942's action_space must render as R+A and R+A+B pads — not the pad
    you get by misreading the index itself as a raw bitmask (index 7 ==
    0x07 == Select+B+A -> ".....SBA"; index 11 == 0x0B == Start+B+A ->
    "....T.BA")."""
    actions_path = tmp_path / "sol_000.actions.npy"
    np.save(actions_path, np.array([7, 11], dtype=np.int64))

    rom_path = tmp_path / "fake.nes"
    rom_path.write_bytes(b"\x00" * 16)  # emit_fm2 only md5-hashes the bytes

    out_path = tmp_path / "out.fm2"
    argv = [
        "emit_fm2.py", str(actions_path), str(out_path),
        "--rom", str(rom_path), "--profile", str(PROFILE),
        "--frame-skip", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert emit_fm2.main() == 0

    pads = _pad_rows(out_path.read_text())
    assert pads == ["R......A", "R.....BA"], (
        "index 7/11 must resolve through 1942's action_space "
        "(['right', 'A'] / ['right', 'A', 'B']), not be read as raw "
        f"bitmasks 0x07/0x0B; got {pads}"
    )
