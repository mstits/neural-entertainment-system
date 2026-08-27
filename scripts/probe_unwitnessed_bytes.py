#!/usr/bin/env python3
"""Re-measure the bytes that `nes_core/src/rewards.rs` asserts win/boss
semantics for, on ROMs where this repo has never witnessed the event.

Every annotation added by the 2026-08-27 Rust provenance pass that cites a
"MEASURED NULL" cites the receipt this script writes. The point is that the
annotation rests on an observation anybody can reproduce here rather than on
a claim about the game.

No game knowledge is used: the driver is uniform random actions (plus a NOOP
arm) from the repo's own shipped start state, and the script reports the
distinct values each address took. It cannot and does not decide what a byte
MEANS — only whether it ever moved.

    .venv/bin/python scripts/probe_unwitnessed_bytes.py
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (label, rom, start-state, {name: addr}, noop_steps, random_steps)
PROBES = [
    (
        "castlevania",
        "roms/Castlevania (USA).nes",
        "roms/Castlevania (USA)_start.state.bin",
        {
            "0x0044_player_health": 0x0044,
            "0x0045_contested": 0x0045,
            "0x0071_hearts": 0x0071,
            "0x01A9_boss_health": 0x01A9,
            "0x0028_stage": 0x0028,
        },
        0,
        20_000,
    ),
    (
        "kid_icarus",
        "roms/Kid Icarus (USA, Europe).nes",
        "roms/Kid Icarus (USA, Europe)_start.state.bin",
        {"0x0130_stage": 0x0130, "0x006B_boss": 0x006B},
        0,
        15_000,
    ),
    (
        "metroid",
        "roms/Metroid (USA).nes",
        "roms/Metroid (USA)_start.state.bin",
        {
            "0x0098_mb_state": 0x0098,
            "0x0099_mb_hits": 0x0099,
            "0x007A_ending_msg": 0x007A,
            "0x007B_credits": 0x007B,
        },
        0,
        15_000,
    ),
    (
        "zelda",
        "roms/Legend of Zelda, The (USA) (Rev A).nes",
        "roms/zelda_start_ctrl.state.bin",
        {"0x0609_song": 0x0609, "0x0672_ganon_defeated": 0x0672},
        600,
        60_000,
    ),
    (
        "kung_fu",
        "roms/Kung Fu (Japan, USA).nes",
        "roms/Kung Fu (Japan, USA)_start.state.bin",
        {"0x0058_floor": 0x0058},
        0,
        15_000,
    ),
    (
        "punch_out",
        "roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A).nes",
        "roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A)_start.state.bin",
        {"0x0001_match_id": 0x0001, "0x000A_losses": 0x000A,
         "0x0398_opp_hp": 0x0398},
        0,
        15_000,
    ),
    (
        "mega_man",
        "roms/Mega Man 2 (USA).nes",
        "roms/Mega Man 2 (USA)_start.state.bin",
        {"0x06C1_boss_health": 0x06C1, "0x06C0_player_health": 0x06C0,
         "0x00A8_lives": 0x00A8},
        0,
        15_000,
    ),
]


def _run(env, addrs: dict[str, int], steps: int, rng, noop: bool):
    seen = {k: set() for k in addrs}
    for _ in range(steps):
        env.step(0 if noop else rng.randrange(0, 256))
        for name, addr in addrs.items():
            seen[name].add(env.get_ram(addr))
    return seen


def probe_one(label, rom, state, addrs, noop_steps, random_steps, seed):
    from nes_core import NESEnvironment

    rom_p, state_p = REPO / rom, REPO / state
    if not rom_p.exists() or not state_p.exists():
        return {"label": label, "status": "SKIPPED_MISSING_FILE",
                "rom": rom, "state": state}

    env = NESEnvironment(str(rom_p), 1, str(state_p))
    start = {n: env.get_ram(a) for n, a in addrs.items()}
    rng = random.Random(seed)

    noop_seen = _run(env, addrs, noop_steps, rng, noop=True) if noop_steps else {}

    env = NESEnvironment(str(rom_p), 1, str(state_p))
    rand_seen = _run(env, addrs, random_steps, random.Random(seed), noop=False)

    out = {}
    for name in addrs:
        vals = sorted(rand_seen[name] | set(noop_seen.get(name, ())))
        out[name] = {
            "start_value": start[name],
            "distinct_values": len(vals),
            "values": vals if len(vals) <= 24 else
                      [vals[0], "...", vals[-1]],
            "min": vals[0],
            "max": vals[-1],
            "moved": len(vals) > 1,
        }
    return {
        "label": label, "status": "OK", "rom": rom, "start_state": state,
        "seed": seed, "noop_steps": noop_steps, "random_steps": random_steps,
        "addresses": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default="docs/receipts/purity/"
                                     "rust_unwitnessed_probe_2026-08-27.json")
    ap.add_argument("--only", default=None,
                    help="comma-separated labels to run")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results = []
    for spec in PROBES:
        if only and spec[0] not in only:
            continue
        print(f"[probe] {spec[0]} ...", flush=True)
        r = probe_one(*spec, seed=args.seed)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproduce the nulls the 2026-08-27 rewards.rs provenance "
                   "annotations cite. Uniform-random driving from the shipped "
                   "start states; no game knowledge used.",
        "driver": "uniform random action bitmask 0..255 (plus a NOOP arm "
                  "where noop_steps > 0)",
        "results": results,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
