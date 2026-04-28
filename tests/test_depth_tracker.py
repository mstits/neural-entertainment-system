"""Tests for the auto-curriculum depth tracker."""

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
    tr = DepthTracker(game="zelda", memo_path=memo)

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
    tr = DepthTracker(game="zelda")
    assert tr.observe(b"\x00\x01", 0, "Noop", 0) is None
    assert tr.best is None


def test_generic_depth_tracker_still_captions() -> None:
    """An unknown game falls back to a generic reader + caption."""
    tr = DepthTracker(game="some_obscure_rom")
    rec = tr.observe(bytes([5, 10]), 0, "Fluff", 0)
    assert rec is not None
    assert "depth key" in rec["caption"]


def test_dump_writes_full_history(tmp_path: Path) -> None:
    tr = DepthTracker(game="zelda")
    tr.observe(_zelda_ram(0, 0x10, 0x10), 0, "A", 0)
    tr.observe(_zelda_ram(1, 0x00, 0x00), 0, "B", 0)
    path = tmp_path / "dump.jsonl"
    tr.dump(path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
