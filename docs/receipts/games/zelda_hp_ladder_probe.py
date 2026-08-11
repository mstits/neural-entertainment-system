"""Repro for configs/zelda.yaml `solve: lives: 0x0670`.

Two measurements, both from our own scripted rollouts with our own rendered
frame as the only ground truth (purity line: no RAM map, no disassembly).

  LADDER   drive into an enemy from the full-health root and record every
           distinct value of $0670 alongside the HUD heart box, sampling
           the HUD ONE STEP LATE. Establishes that $0670 is a
           half-heart-resolution HP register and that its integer ladder is
           NOT monotone (127 -> 254 while HP falls).

  LAG      the discriminator for "does is_dead fire at full health?". The
           instant $0670 leaves its root value, freeze the controller to
           no-op for 20 steps. A no-op cannot inflict new damage, so if the
           HUD then drops and stays down, the hit was already committed in
           RAM at the transition step and the byte simply LEADS the render
           by one frame_skip=4 step.

Run:  .venv/bin/python docs/receipts/games/zelda_hp_ladder_probe.py
Cost: ~3 min, 4 workers, no build, no training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402

ROM = str(REPO / "roms/Legend of Zelda, The (USA) (Rev A).nes")
FULL_PX = 123          # three full hearts in the HUD box, our own frame
DIRS = [0x80, 0x40, 0x10, 0x20]


def hearts(frame) -> int:
    """Red pixels in the HUD heart box. One full heart = 41 px."""
    box = frame[44:58, 170:208, :].astype(int)
    return int(((box[:, :, 0] > 150) & (box[:, :, 1] < 90)
                & (box[:, :, 2] < 90)).sum())


def ladder(pool, state_name: str) -> None:
    st = (REPO / "roms" / state_name).read_bytes()
    pool.load_worker_state(0, st)

    def step(mask):
        res = pool.step_all(np.array([mask] + [0] * (pool_workers - 1),
                                     dtype=np.uint8))
        return res[0][0], np.frombuffer(bytes(res[0][2]), dtype=np.uint8)[:0x800]

    frame, ram = step(0)
    root = int(ram[0x670])
    print(f"\n=== {state_name}: root $0670={root} $066F={int(ram[0x66F])} "
          f"hp={hearts(frame)} px")
    rows, prev, black = [], None, False
    for _ in range(900):
        frame, ram = step(0x40)                     # walk LEFT into the mob
        value = int(ram[0x670])
        if frame.mean() < 20:
            black = True
            break
        if value != prev:
            settled, _ = step(0)                    # HUD repaints one step later
            rows.append((value, int(ram[0x66F]), hearts(settled)))
            prev = value
    print("     $0670   $066F   hp px (one step later)")
    for value, coarse, px in rows:
        print(f"      {value:5d}   {coarse:5d}   {px:5d}")
    values = [r[0] for r in rows]
    damaged = [v for v in values[1:]]
    print(f"  death fade reached ......... {black}")
    print(f"  raw ladder monotone ........ "
          f"{all(b <= a for a, b in zip(values, values[1:]))}")
    print(f"  (v & 0x7F) monotone ........ "
          f"{all((b & 0x7F) <= (a & 0x7F) for a, b in zip(values, values[1:]))}")
    print(f"  every damaged value < root . "
          f"{all(v < root for v in damaged) if damaged else 'n/a'}"
          f"   <-- what GenericGame.is_dead actually needs")


def lag(pool, trials: int = 10) -> None:
    st = (REPO / "roms/zelda_start.state.bin").read_bytes()
    rng = np.random.default_rng(777)
    start_lives = 255
    same_frame = lagged = truly_full = 0
    for _ in range(trials):
        for w in range(pool_workers):
            pool.load_worker_state(w, st)
        cur = [int(DIRS[rng.integers(4)]) for _ in range(pool_workers)]
        frozen = [0] * pool_workers
        at, after = [None] * pool_workers, [[] for _ in range(pool_workers)]
        for k in range(700):
            if k % 45 == 0:
                cur = [int(DIRS[rng.integers(4)]) for _ in range(pool_workers)]
            acts = np.array(
                [0 if frozen[w] else cur[w] | (0x01 if k % 11 == 0 else 0)
                 for w in range(pool_workers)], dtype=np.uint8)
            res = pool.step_all(acts)
            for w in range(pool_workers):
                frame = res[w][0]
                ram = np.frombuffer(bytes(res[w][2]), dtype=np.uint8)[:0x800]
                if frozen[w]:
                    after[w].append(hearts(frame))
                    frozen[w] += 1
                elif int(ram[0x670]) < start_lives:   # exactly is_dead
                    at[w] = hearts(frame)
                    frozen[w] = 1
            if all(f >= 21 for f in frozen):
                break
        for w in range(pool_workers):
            if at[w] is None:
                continue
            tail = after[w][:20]
            if at[w] < FULL_PX:
                same_frame += 1
            elif tail and min(tail) < FULL_PX:
                lagged += 1
            else:
                truly_full += 1
    total = same_frame + lagged + truly_full
    print(f"\n=== no-op freeze test: {total} is_dead firings "
          f"({trials} trials x {pool_workers} workers)")
    print(f"  damage already visible on the same frame . {same_frame}")
    print(f"  HUD lagged one step (real hit) ........... {lagged}")
    print(f"  genuinely full health after 20 no-ops .... {truly_full}"
          f"   <-- the only real false deaths")


if __name__ == "__main__":
    pool_workers = 4
    p = Pool(rom_path=ROM, num_workers=pool_workers, frame_skip=4)
    p.set_headless(False)
    p.reset_all()
    try:
        ladder(p, "zelda_start.state.bin")
        ladder(p, "zelda_start_ctrl.state.bin")
        lag(p)
    finally:
        p.shutdown()
