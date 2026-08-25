"""Deepest-cell diagnostic for the 1942 onboarding smoke (wave 3).

Purity-line tool: reads only Pool-exposed RAM/odometer surfaces already
certified elsewhere (odometer_cert.py) plus the profile's own discovered
`lives` byte (discover_observables.py, 5/5 agreement). No disassembly,
no external RAM maps.

Loads the highest-best_score cell from a go_explore_solve.py archive,
holds each candidate action for ~150 steps, and reports:
  - odometer (x, y) before/after (the certified progress surface)
  - the profile's `lives` byte before/after
  - raw RAM churn, to distinguish a frozen screen from a live one
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from nes_core import Pool  # noqa: E402

ROM = "roms/1942 (Japan, USA).nes"
ARCHIVE = "runs/onboard_wave3/smoke_1942/archive.pkl"
LIVES_ADDR = 0x00C9
HOLD_STEPS = 150

BIT = {"A": 0x01, "B": 0x02, "select": 0x04, "start": 0x08,
       "up": 0x10, "down": 0x20, "left": 0x40, "right": 0x80}
ACTIONS = {
    "noop": 0,
    "right": BIT["right"], "left": BIT["left"],
    "up": BIT["up"], "down": BIT["down"],
    "A": BIT["A"], "B": BIT["B"],
    "start": BIT["start"],
}


def main() -> None:
    with open(ARCHIVE, "rb") as f:
        arc = pickle.load(f)
    cells = list(arc.values())
    cells.sort(key=lambda c: (c.best_score, c.best_steps, c.visits), reverse=True)
    deep = cells[0]
    print(f"[deepest cell] key={deep.key} best_score={deep.best_score} "
          f"best_steps={deep.best_steps} visits={deep.visits} "
          f"explored={deep.explored} barren={deep.barren}")
    print(f"[archive] {len(cells)} total cells, all keys:")
    for c in cells:
        print(f"   key={c.key} score={c.best_score} steps={c.best_steps} "
              f"visits={c.visits}")

    pool = Pool(rom_path=ROM, num_workers=1, frame_skip=4)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.set_odometer_enabled(True)

    for name, mask in ACTIONS.items():
        pool.load_worker_state(0, deep.state)
        x0, y0 = pool.get_odometer_per_worker()[0]
        r0 = np.frombuffer(bytes(pool.step_all(
            np.array([mask], dtype=np.uint8))[0][2]), dtype=np.uint8)
        lives0 = int(r0[LIVES_ADDR])
        churn_total = 0
        lives_seq = [lives0]
        prev = r0
        for i in range(HOLD_STEPS):
            r = np.frombuffer(bytes(pool.step_all(
                np.array([mask], dtype=np.uint8))[0][2]), dtype=np.uint8)
            churn_total += int((r.astype(np.int16) != prev.astype(np.int16)).sum())
            prev = r
            if i % 30 == 29:
                lives_seq.append(int(r[LIVES_ADDR]))
        x1, y1 = pool.get_odometer_per_worker()[0]
        lives_end = int(prev[LIVES_ADDR])
        try:
            dbg = pool.odo_debug(0)
        except Exception as e:  # pragma: no cover
            dbg = f"<odo_debug error: {e}>"
        print(f"[hold {name:6s}] x {x0}->{x1} (d{x1-x0:+d})  "
              f"y {y0}->{y1} (d{y1-y0:+d})  "
              f"lives {lives0}->{lives_end} seq={lives_seq}  "
              f"churn/step={churn_total/HOLD_STEPS:.1f}  odo_debug={dbg}")


if __name__ == "__main__":
    main()
