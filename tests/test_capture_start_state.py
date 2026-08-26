"""Regression guard: is_controllable() must not mistake an animated,
directionally-navigable MENU screen for live gameplay.

History (2026-08-25): DuckTales 2's "WHERE TO, UNCLE SCROOGE?" world-map
land-select screen has a spinning-globe animation (mutates RAM every
frame independent of input) plus a genuinely directional cursor (LEFT
and RIGHT select different locations). scripts/capture_start_state.py's
generic mash schedule ran straight into it: the growth check compared
RAM against a frozen base snapshot, so the animation's own churn read
as "still changing" (not a saturated menu cursor), and the cursor's
real-but-bounded directionality satisfied the RIGHT-vs-LEFT check too —
so is_controllable() reported gameplay at frame 330 and the tool
silently wrote the land-select menu out as the game's start state,
exit 0, no error. A human happened to render the captured frame and
caught it (see runs/onboard_wave4/mint_ducktales_2.json +
mint_ducktales_2_frame.png) — nothing in the automated pipeline would
have.

This test replays the tool's own mash schedule against the real ROM up
to that exact frame and pins is_controllable() to reject it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from capture_start_state import _menu_input, is_controllable  # noqa: E402


def _find(rel_candidates: list[str]) -> Path | None:
    """Mirrors tests/test_cold_probe.py::_find — gitignored ROM lookup
    across this checkout and each ancestor (worktree-safe)."""
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


_DT2_ROM = _find(["roms/DuckTales 2 (USA).nes"])


@pytest.mark.skipif(_DT2_ROM is None, reason="needs the DuckTales 2 ROM")
def test_is_controllable_rejects_ducktales2_worldmap_menu():
    """Frame 330 of the tool's own mash schedule lands on the animated
    world-map land-select screen — not gameplay. Before the fix this
    was the confirmed false positive."""
    import nes_core

    env = nes_core.NESEnvironment(rom_path=str(_DT2_ROM), frame_skip=1)
    env.reset()
    for frame in range(331):
        env.step(_menu_input(frame))
    assert not is_controllable(env, thresh=25), (
        "is_controllable() accepted the DuckTales 2 world-map land-select "
        "menu as live gameplay -- the animated-menu false positive is back"
    )
