"""Shared free-disk floor: refuse to launch a run whose checkpoint saves
are about to hit ENOSPC.

Extracted 2026-09-02 from scripts/run_online_campaign.py's
disk_floor_breach (same semantics, byte-for-byte on the reason string).
Before this, only run_online_campaign.py, scripts/train_game.py and
scripts/engine_driver.py checked free disk at launch; scripts/
go_explore_chain.py ("machine-busy and hands-off") and scripts/
night2_runner.py had no floor at all — either can run for hours-to-days
on a volume that fills mid-run, surfacing only as swallowed checkpoint-
save warnings inside the child (external audit 2026-08-29, volume at
91%).
"""
from __future__ import annotations

from pathlib import Path

DISK_FLOOR_GB = 40.0


def disk_free_gb(path: Path) -> float:
    import shutil as _shutil
    return _shutil.disk_usage(str(path)).free / 1e9


def disk_floor_breach(
    path: Path,
    *,
    floor_gb: float = DISK_FLOOR_GB,
    allow_low_disk: bool = False,
    free_gb_fn=disk_free_gb,
) -> str | None:
    """A reason string when free disk is below floor_gb, else None.

    Fails OPEN on a stat error (a stat hiccup must never be the reason a
    healthy run is refused) and CLOSED on measured low disk.
    `allow_low_disk` bypasses the refusal outright — a caller that
    exposes its own --allow-low-disk flag should log a warning before
    launching under it, since this returns None either way once
    overridden.
    """
    if allow_low_disk:
        return None
    try:
        free = free_gb_fn(path)
    except OSError:
        return None
    if free < floor_gb:
        return (f"disk floor: {free:.1f} GB free < {floor_gb:.0f} GB "
                f"— checkpoint durability is about to fail")
    return None
