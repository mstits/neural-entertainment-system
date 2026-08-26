"""D4 (docs/research/CLEAR_GAP_CLOSURE_2026-08-26.md §1c row 6 / §9.3):
`configs/blaster_master.yaml` declared `area: 0x0020` as a "non-
saturating room/screen counter" discovered by `discover_observables.py`.
Both halves were wrong. $0020 is the high byte of the exact same
$0048|$0020<<8 pair the same receipt already flagged as a camera-clamp
saturator — not an independent room identity — and it saturates:
holding right from the start state runs it 32 -> 244 and then flat, the
same shape as its own pair. Using it as a cell-key `area:` spuriously
multiplied the archive 8.2x (9,287 banked cells vs 1,129 the identical
rollout data actually covers once the area dimension is collapsed to a
constant).

The fix is to stop declaring `area:` for this profile rather than
replace one guess with another — no independently-discovered
non-saturating room counter exists for this game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CONFIG_PATH = _ROOT / "configs" / "blaster_master.yaml"


def _find(rel_candidates: list[str]) -> Path | None:
    """Mirrors tests/test_cold_probe.py::_find — gitignored ROM lookup
    across this checkout and each ancestor (worktree-safe)."""
    for root in (_ROOT, *_ROOT.parents):
        for rel in rel_candidates:
            for hit in sorted(root.glob(rel)):
                if hit.exists():
                    return hit
    return None


_BM_ROM = _find(["roms/Blaster Master (USA).nes"])
_BM_STATE = _find(["roms/Blaster Master (USA)_start.state.bin"])


def test_area_key_is_not_declared() -> None:
    """Regression guard: no address should be promoted back to `area:`
    without fresh, passing non-saturation evidence attached — the bare
    absence of the key is what a correct, purity-respecting fix looks
    like here (see the module docstring)."""
    data = yaml.safe_load(CONFIG_PATH.read_text())
    assert "area" not in data["solve"], (
        "configs/blaster_master.yaml declares solve.area again — if this "
        "is $0020, it saturates (32 -> 244, reproduced live) and is the "
        "high byte of the already-saturating $0048|$0020<<8 progress "
        "pair, not an independent room counter; do not re-add it without "
        "a fresh, live-saturation check on the new candidate."
    )


@pytest.mark.skipif(
    _BM_ROM is None or _BM_STATE is None,
    reason="needs the Blaster Master ROM + start state",
)
def test_0020_saturates_under_a_forward_hold() -> None:
    """Live reproduction of the D4 finding: $0020 is not a non-saturating
    room counter. Holding right from the verified start state, it climbs
    then goes flat well before the hold ends — same shape as the
    $0048|$0020<<8 pair it is the high byte of."""
    import nes_core

    RIGHT = 0x80
    pool = nes_core.Pool(rom_path=str(_BM_ROM), num_workers=1, frame_skip=4)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, _BM_STATE.read_bytes())

    a = np.array([RIGHT], dtype=np.uint8)
    vals = np.empty(2000, dtype=np.uint8)
    for t in range(2000):
        ram = pool.step_all(a)[0][2]
        vals[t] = ram[0x0020]

    # Saturates well inside the hold, not merely by the very end —
    # distinguishes a real cap from a value still climbing when the
    # probe happened to stop.
    tail = vals[-200:]
    assert len(set(tail.tolist())) == 1, (
        f"$0020 is still changing in the final 200 of 2000 steps "
        f"({sorted(set(tail.tolist()))}) — it no longer saturates; "
        f"re-derive the D4 finding before trusting it as area: again."
    )
    assert int(tail[0]) < 255, (
        "saturated at the byte's own ceiling (255) rather than a "
        "game-specific cap — re-check this is still the same signal."
    )
