#!/usr/bin/env python3
"""Repro driver for the Metroid death-observable probe (2026-08-24 T4,
receipt: docs/receipts/room_fp/metroid.md).

Differential analysis over OUR OWN rollouts, nothing external: three
scripted death runs + four live controls, full 2 KB system RAM per
solver step; a candidate byte must hold a novel sustained value after
the death moment in every death run and never take that value in any
control. The death moment itself is located from a hardware surface
(first rendered-lines==0 sample after the energy drain).

Run from the repo root:  .venv/bin/python docs/receipts/room_fp/metroid_death_probe.py
Expected: the receipt's table — $0020==4 (with co-witnesses $0614==31,
$0673==6, $001E==4) clean; $0610==140 struck for door-scroll false
fires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nes_core  # noqa: E402
from room_fp_calibrate import parse_script, script_actions  # noqa: E402

ROM = REPO / "roms" / "Metroid (USA).nes"
START = REPO / "roms" / "Metroid (USA)_start.state.bin"
ONPLAT = REPO / "tests" / "fixtures" / "roomgraph" / "metroid_onplat.state.bin"
R, L, A, B = 0x80, 0x40, 0x01, 0x02


def run(state: Path, actfn, n: int):
    pool = nes_core.Pool(rom_path=str(ROM), num_workers=1, frame_skip=4)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.set_odometer_enabled(True)
    pool.load_worker_state(0, state.read_bytes())
    rams, blank = [], None
    for s in range(n):
        out = pool.step_all(np.array([actfn(s)], dtype=np.uint8))
        rams.append(np.frombuffer(bytes(out[0][2]), dtype=np.uint8).copy())
        if blank is None and s > 50 and pool.odo_debug(0)[2] == 0:
            blank = s
    return np.stack(rams), blank


def candidates(ram: np.ndarray, death: int) -> dict:
    post, pre = ram[death + 20:], ram[:death - 10]
    out = {}
    for b in range(2048):
        vals = np.unique(post[:, b])
        if len(vals) == 1 and vals[0] not in pre[:, b]:
            out[b] = int(vals[0])
    return out


def main() -> int:
    deaths = [
        run(START, lambda s: (R if (s // 30) % 2 == 0 else L)
            | (A if (s // 4) % 2 == 0 else 0), 400),
        run(START, lambda s: (L if (s // 24) % 2 == 0 else R)
            | (A if (s // 3) % 2 == 0 else 0), 400),
        run(ONPLAT, lambda s: (R if (s // 30) % 2 == 0 else L)
            | (A if (s // 4) % 2 == 0 else 0), 500),
    ]
    for i, (_, blank) in enumerate(deaths):
        print(f"death run {i}: render-off at step {blank}")
        assert blank is not None, "a death script no longer kills Samus"
    common = None
    for ram, blank in deaths:
        c = candidates(ram, blank)
        common = c if common is None else {
            b: v for b, v in common.items() if c.get(b) == v}
    print(f"{len(common)} bytes consistent across all deaths")

    controls = [
        (START, "noop*1500"),
        (START, "noop*16,right*100,left*100,right*100,left*100,noop*24"),
        (ONPLAT, "noop*20,b%2*4,right*36,noop*30,right+a/4*52,b%2*10,"
                 "right*46,noop*50"),
    ]
    struck = set()
    for state, script in controls:
        acts = list(script_actions(parse_script(script)))
        ram, _ = run(state, lambda s: acts[s], len(acts))
        for b, v in common.items():
            if (ram[:, b] == v).any():
                struck.add(b)
    jig, _ = run(START, lambda s: R if (s // 30) % 2 == 0 else L, 400)
    for b, v in common.items():
        if (jig[:, b] == v).any():
            struck.add(b)

    print("clean candidates (byte -> sustained death value):")
    for b in sorted(set(common) - struck):
        print(f"  0x{b:04X} -> {common[b]}")
    if struck:
        print("struck for live false fires:",
              sorted(f"0x{b:04X}" for b in struck))
    assert common.get(0x0020) == 4 and 0x0020 not in struck, \
        "the receipted $0020==4 observable no longer reproduces"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
