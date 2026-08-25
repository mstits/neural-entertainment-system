"""Deepest-cell diagnostic for the Chip 'n Dale Rescue Rangers onboarding
smoke (wave 3).

Purity-line tool: reads only Pool-exposed RAM/odometer surfaces already
certified elsewhere (odometer_cert.py) plus the profile's own discovered
`area` (room/screen counter) and `lives` bytes (discover_observables.py
3-probe protocol, receipt runs/onboard_wave3/discover_chip_n_dale.json).
No disassembly, no external RAM maps.

Loads the highest-(best_score, best_steps, visits) cell from the
go_explore_solve.py smoke archive (frozen at max_area=40 for the whole
4-minute run), holds each candidate action for ~150 steps, and reports:
  - odometer (x, y) before/after (the certified progress surface)
  - the profile's `area` (0x0023, room/screen counter) before/after
  - the profile's `lives` (0x0052) byte before/after
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

ROM = "roms/Chip 'n Dale Rescue Rangers (USA).nes"
ARCHIVE = "runs/onboard_wave3/smoke_chip_n_dale/archive.pkl"
AREA_ADDR = 0x0023
LIVES_ADDR = 0x0052
HOLD_STEPS = 150

BIT = {"A": 0x01, "B": 0x02, "select": 0x04, "start": 0x08,
       "up": 0x10, "down": 0x20, "left": 0x40, "right": 0x80}
ACTIONS = {
    "noop": 0,
    "right": BIT["right"], "left": BIT["left"],
    "up": BIT["up"], "down": BIT["down"],
    "A": BIT["A"], "B": BIT["B"],
    "right+A": BIT["right"] | BIT["A"],
    "right+B": BIT["right"] | BIT["B"],
    "left+A": BIT["left"] | BIT["A"],
    "left+B": BIT["left"] | BIT["B"],
    "down+B": BIT["down"] | BIT["B"],
    "up+B": BIT["up"] | BIT["B"],
}


def main() -> None:
    with open(REPO / ARCHIVE, "rb") as f:
        arc = pickle.load(f)
    cells = list(arc.values())
    cells.sort(key=lambda c: (c.best_score, c.best_steps, c.visits), reverse=True)
    deep = cells[0]
    print(f"[deepest cell] key={deep.key} best_score={deep.best_score} "
          f"best_steps={deep.best_steps} visits={deep.visits} "
          f"explored={deep.explored} "
          f"barren={getattr(deep, 'barren', 'n/a')}")
    print(f"[archive] {len(cells)} total cells; top 10 by score:")
    for c in cells[:10]:
        print(f"   key={c.key} score={c.best_score} steps={c.best_steps} "
              f"visits={c.visits} times_chosen={c.times_chosen} "
              f"barren={getattr(c, 'barren', 'n/a')}")

    pool = Pool(rom_path=str(REPO / ROM), num_workers=1, frame_skip=4)
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
        area0 = int(r0[AREA_ADDR])
        churn_total = 0
        lives_seq = [lives0]
        area_seq = [area0]
        prev = r0
        for i in range(HOLD_STEPS):
            r = np.frombuffer(bytes(pool.step_all(
                np.array([mask], dtype=np.uint8))[0][2]), dtype=np.uint8)
            churn_total += int((r.astype(np.int16) != prev.astype(np.int16)).sum())
            prev = r
            if i % 30 == 29:
                lives_seq.append(int(r[LIVES_ADDR]))
                area_seq.append(int(r[AREA_ADDR]))
        x1, y1 = pool.get_odometer_per_worker()[0]
        lives_end = int(prev[LIVES_ADDR])
        area_end = int(prev[AREA_ADDR])
        try:
            dbg = pool.odo_debug(0)
        except Exception as e:  # pragma: no cover
            dbg = f"<odo_debug error: {e}>"
        print(f"[hold {name:8s}] odo_x {x0}->{x1} (d{x1-x0:+d})  "
              f"odo_y {y0}->{y1} (d{y1-y0:+d})  "
              f"lives {lives0}->{lives_end} seq={lives_seq}  "
              f"area@0x23 {area0}->{area_end} seq={area_seq}  "
              f"churn/step={churn_total/HOLD_STEPS:.1f}  odo_debug={dbg}")


if __name__ == "__main__":
    main()
