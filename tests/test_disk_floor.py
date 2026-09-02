"""Unit tests for the shared free-disk floor (src/utils/disk_floor.py).

Extracted from scripts/run_online_campaign.py's disk_floor_breach so
scripts/go_explore_chain.py and scripts/night2_runner.py can share the
launch-time guard; see tests/test_disk_floor_and_save_escalation.py for
the per-caller wiring (train_game.py + run_online_campaign.py) this
module preserves byte-for-byte.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils import disk_floor  # noqa: E402


def test_refuses_below_floor():
    reason = disk_floor.disk_floor_breach(REPO, free_gb_fn=lambda p: 3.0)
    assert reason is not None
    assert "disk floor" in reason
    assert "3.0 GB free" in reason


def test_passes_with_headroom():
    reason = disk_floor.disk_floor_breach(
        REPO, free_gb_fn=lambda p: disk_floor.DISK_FLOOR_GB + 1)
    assert reason is None


def test_passes_with_override_even_when_low():
    reason = disk_floor.disk_floor_breach(
        REPO, free_gb_fn=lambda p: 3.0, allow_low_disk=True)
    assert reason is None, "allow_low_disk must bypass the refusal outright"


def test_fails_open_on_stat_error():
    def boom(p):
        raise OSError("statvfs hiccup")
    reason = disk_floor.disk_floor_breach(REPO, free_gb_fn=boom)
    assert reason is None, (
        "a stat error must never be the reason a run is refused")


def test_default_free_gb_fn_reads_real_disk(tmp_path):
    # No free_gb_fn override: exercises the real shutil.disk_usage path.
    free = disk_floor.disk_free_gb(tmp_path)
    assert free > 0
