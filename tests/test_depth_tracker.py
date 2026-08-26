"""Tests for the auto-curriculum depth tracker.

The RAM reader is chosen by `depth_id` — the profile's declared
`reward_id`. It used to be chosen by substring-matching the display
name, so every one of these `game="zelda"` fixtures had to grow an
explicit `depth_id="zelda"`. That churn is itself the receipt that
dispatch moved off the name.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.depth_tracker import DepthTracker


def _zelda_ram(dungeon: int, mx: int, my: int) -> bytes:
    """Build a RAM blob with just enough bytes for _zelda_depth to read."""
    buf = bytearray(0x1100)
    buf[0x10] = dungeon
    buf[0xEB] = mx
    buf[0xEC] = my
    return bytes(buf)


def test_zelda_tracker_records_first_and_deeper(tmp_path: Path) -> None:
    memo = tmp_path / "memo.jsonl"
    tr = DepthTracker(game="zelda", memo_path=memo, depth_id="zelda")

    first = tr.observe(_zelda_ram(0, 0x77, 0x77), 3, "Brutus", generation=0)
    assert first is not None
    assert first["key"] == [0, 0x77, 0x77]

    # Shallower positions don't record.
    assert tr.observe(_zelda_ram(0, 0x40, 0x40), 1, "Scout", 0) is None

    # A dungeon entry IS deeper than any overworld key.
    dungeon1 = tr.observe(_zelda_ram(1, 0x05, 0x07), 5, "Ajax", 1)
    assert dungeon1 is not None
    assert dungeon1["key"] == [1, 0x05, 0x07]

    # Memo file has both records, one per line.
    lines = memo.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["genome_name"] == "Brutus"
    assert parsed[1]["genome_name"] == "Ajax"
    assert "dungeon" in parsed[1]["caption"].lower()


def test_depth_tracker_handles_short_ram() -> None:
    """Truncated RAM must not crash the tracker — just no record."""
    tr = DepthTracker(game="zelda", depth_id="zelda")
    assert tr.observe(b"\x00\x01", 0, "Noop", 0) is None
    assert tr.best is None


def test_generic_depth_tracker_still_captions() -> None:
    """A profile that declares no depth id gets the generic reader."""
    tr = DepthTracker(game="some_obscure_rom")
    rec = tr.observe(bytes([5, 10]), 0, "Fluff", 0)
    assert rec is not None
    assert "depth key" in rec["caption"]


def test_dump_writes_full_history(tmp_path: Path) -> None:
    tr = DepthTracker(game="zelda", depth_id="zelda")
    tr.observe(_zelda_ram(0, 0x10, 0x10), 0, "A", 0)
    tr.observe(_zelda_ram(1, 0x00, 0x00), 0, "B", 0)
    path = tmp_path / "dump.jsonl"
    tr.dump(path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_display_name_does_not_select_the_depth_reader() -> None:
    """The defect, stated as an assertion.

    A profile titled "The Legend of Zelda" that declares no `reward_id`
    must read the GENERIC depth key (RAM bytes 0 and 1), not Zelda's
    hand-authored $10 / $EB / $EC offsets. Pre-change this returned
    (3, 34, 51) — the Zelda key — off the display name alone.
    """
    ram = bytearray(2048)
    ram[0x00] = 1
    ram[0x01] = 2
    ram[0x10] = 3
    ram[0xEB] = 0x22
    ram[0xEC] = 0x33
    tr = DepthTracker(game="The Legend of Zelda")  # label only, no depth_id
    rec = tr.observe(bytes(ram), 0, "g", 0)
    assert rec is not None
    assert rec["key"] == [0, 1, 2], (
        "the display name must not select the Zelda RAM reader"
    )

    # ...and the declared id is what does turn it on.
    tr2 = DepthTracker(game="anything at all", depth_id="zelda")
    rec2 = tr2.observe(bytes(ram), 0, "g", 0)
    assert rec2 is not None
    assert rec2["key"] == [3, 0x22, 0x33]


def test_unknown_depth_id_reads_generic_never_a_borrowed_map() -> None:
    """`tetris` is a valid reward id with no depth reader of its own.

    It must fall to the generic reader rather than borrowing Mario's or
    Zelda's offsets.
    """
    ram = bytearray(2048)
    ram[0x00] = 5
    ram[0x01] = 6
    ram[0x10] = 9
    tr = DepthTracker(game="Tetris", depth_id="tetris")
    rec = tr.observe(bytes(ram), 0, "g", 0)
    assert rec is not None
    assert rec["key"] == [0, 5, 6]
