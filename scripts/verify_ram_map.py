"""Observationally verify a game profile's ram_mapping from our own rollouts.

The solver may only key its cells on RAM bytes whose meaning has been
MEASURED, not assumed (CLAIMS.md provenance rule). This tool runs three
probes from the profile's start state and grades every claimed mapping:

  A. hold-right probe   — 900 steps of pure "right": progress bytes must
                          grow (monotone fraction + net delta); a 16-bit
                          (page, pixel) pair shows pixel wraps 255->0 in
                          the same step the page increments.
  B. jump probe         — alternating right/A: the y byte must swing.
  C. sticky-random probe— N steps of sticky random play: churn per 1k
                          steps for every byte (streaming scratchpads
                          churn 5-10/1k; real state bytes sit far below),
                          plus observed decrements of the lives byte.

Verdicts land in runs/ram_verify/<game>.json and print as a table.
Nothing here injects game knowledge: every number comes from the
emulator's own memory under our own inputs.

Usage:
  python scripts/verify_ram_map.py --profile configs/castlevania.yaml
  python scripts/verify_ram_map.py --profile configs/castlevania.yaml --steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

RAM_SIZE = 0x800


def find_rom(profile: dict) -> str:
    """Resolve the ROM from the start-state filename convention or name."""
    ss = profile.get("start_state_path", "")
    stem = Path(ss).name.replace("_start.state.bin", "")
    for cand in (REPO / "roms" / f"{stem}.nes",):
        if cand.exists():
            return str(cand)
    name = profile.get("name", "")
    hits = [p for p in (REPO / "roms").glob("*.nes")
            if name.lower() in p.name.lower()]
    if hits:
        return str(sorted(hits, key=lambda p: len(p.name))[0])
    raise SystemExit(f"cannot resolve ROM for profile {name!r}")


class Rig:
    def __init__(self, rom: str, state: bytes, frame_skip: int):
        self.pool = Pool(rom_path=rom, num_workers=1, frame_skip=frame_skip)
        self.pool.set_headless(True)
        self.pool.reset_all()
        self.state = state
        self.reload()

    def reload(self):
        self.pool.load_worker_state(0, self.state)

    def step(self, mask: int) -> np.ndarray:
        r = self.pool.step_all(np.array([mask], dtype=np.uint8))
        return np.frombuffer(bytes(r[0][2]), dtype=np.uint8)


def run_probe(rig: Rig, masks) -> np.ndarray:
    """Step through `masks`, return (steps, RAM_SIZE) uint8 log."""
    log = np.empty((len(masks), RAM_SIZE), dtype=np.uint8)
    for i, m in enumerate(masks):
        log[i] = rig.step(int(m))[:RAM_SIZE]
    return log


def analyze(log: np.ndarray) -> dict:
    """Per-byte deltas over a log: churn, net movement, monotone fraction."""
    d = np.diff(log.astype(np.int16), axis=0)
    changed = d != 0
    churn_1k = changed.sum(axis=0) / max(len(d), 1) * 1000.0
    # Treat wraparound (e.g. 255->0 while walking right) as +1 progress.
    dw = np.where(d < -128, d + 256, np.where(d > 128, d - 256, d))
    net = dw.sum(axis=0)
    nz = changed.sum(axis=0)
    mono = np.where(nz > 0, (dw > 0).sum(axis=0) / np.maximum(nz, 1), 0.0)
    return {"churn_1k": churn_1k, "net": net, "mono": mono, "diff": d}


def wrap_pairs(log: np.ndarray, pixel: int, page: int) -> dict:
    """Does `pixel` wrap 255->0 when `page` increments (+/-1 step)?"""
    d = np.diff(log.astype(np.int16), axis=0)
    wraps = np.where(d[:, pixel] < -128)[0]
    if len(wraps) == 0:
        return {"wraps": 0, "coupled": 0}
    coupled = sum(1 for w in wraps
                  if any(0 <= w + o < len(d) and d[w + o, page] == 1
                         for o in (-1, 0, 1)))
    return {"wraps": int(len(wraps)), "coupled": int(coupled)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, default=12000,
                    help="sticky-random probe length")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prof = yaml.safe_load(Path(args.profile).read_text())
    rom = find_rom(prof)
    state = (REPO / prof["start_state_path"]).read_bytes()
    fs = int(prof.get("frame_skip", 4))
    bm = action_space_to_bitmasks(prof["action_space"])
    mapping = {k: int(v) for k, v in prof.get("ram_mapping", {}).items()}
    print(f"[rig] {Path(rom).name}  frame_skip={fs}  "
          f"{len(mapping)} mapped bytes to grade")

    def mask_for(*buttons):
        want = set(buttons)
        for i, combo in enumerate(prof["action_space"]):
            if set(combo) == want:
                return int(bm[i])
        raise SystemExit(f"action space lacks {buttons}")

    rig = Rig(rom, state, fs)

    # --- A: hold right ------------------------------------------------
    right = mask_for("right")
    log_a = run_probe(rig, [right] * 900)
    a = analyze(log_a)

    # --- B: jump probe ------------------------------------------------
    rig.reload()
    jump = mask_for("A") if any(set(c) == {"A"} for c in prof["action_space"]) \
        else mask_for("right", "A")
    seq_b = ([right] * 6 + [jump] * 6) * 25
    log_b = run_probe(rig, seq_b)
    b = analyze(log_b)

    # --- C: sticky random ---------------------------------------------
    rig.reload()
    rng = np.random.default_rng(args.seed)
    cur = 0
    masks_c = []
    for _ in range(args.steps):
        if rng.random() > 0.75:
            cur = int(rng.integers(len(bm)))
        masks_c.append(int(bm[cur]))
    log_c = run_probe(rig, masks_c)
    c = analyze(log_c)

    # --- grade the profile's claims ------------------------------------
    verdicts = {}
    px = mapping.get("player_x")
    pg = mapping.get("player_x_hi", mapping.get("player_x_screen"))
    pg_name = "player_x_hi" if "player_x_hi" in mapping else "player_x_screen"

    def grade(name, addr, ok, why):
        verdicts[name] = {
            "addr": hex(addr), "ok": bool(ok), "why": why,
            "churn_1k_random": round(float(c["churn_1k"][addr]), 2),
            "net_holdright": int(a["net"][addr]),
            "mono_holdright": round(float(a["mono"][addr]), 2),
        }

    for name, addr in mapping.items():
        if name in ("player_x", "player_x_screen", "player_x_hi"):
            continue  # graded jointly below
        elif name == "player_y":
            swing = int(np.ptp(log_b[:, addr]))
            grade(name, addr, swing >= 8,
                  f"jump-probe swing {swing} (need >=8)")
        elif name in ("stage_number",):
            grade(name, addr, c["churn_1k"][addr] < 1.0,
                  f"stable under random play "
                  f"({c['churn_1k'][addr]:.2f} changes/1k)")
        elif name == "lives":
            drops = int((c["diff"][:, addr] < 0).sum())
            grade(name, addr, 0 < c["churn_1k"][addr] < 5.0 and drops > 0,
                  f"{drops} decrements over random play")
        elif name in ("player_health", "boss_health", "hearts"):
            grade(name, addr, c["churn_1k"][addr] < 50.0,
                  f"churn {c['churn_1k'][addr]:.1f}/1k (informational)")
        else:
            grade(name, addr, True,
                  f"unclassified; churn {c['churn_1k'][addr]:.1f}/1k")

    if px is not None:
        net_px = int(a["net"][px])
        mono_px = float(a["mono"][px])
        pair = wrap_pairs(log_a, px, pg) if pg is not None else None
        ok = net_px > 100 and mono_px > 0.8
        why = f"hold-right net {net_px}, mono {mono_px:.2f}"
        if pair:
            why += (f"; wraps {pair['wraps']}, page-coupled "
                    f"{pair['coupled']}")
        grade("player_x", px, ok, why)
        if pg is not None:
            grade(pg_name, pg,
                  pair["wraps"] > 0 and pair["coupled"] >= pair["wraps"] - 2,
                  f"page byte: {pair['coupled']}/{pair['wraps']} wraps coupled")

    # --- discover unmapped progress candidates -------------------------
    # Bytes that grew monotonically under hold-right but churn little
    # under random play — the signature of a real position/scroll counter.
    cand = [
        {"addr": hex(i), "net": int(a["net"][i]),
         "mono": round(float(a["mono"][i]), 2),
         "churn_1k_random": round(float(c["churn_1k"][i]), 1)}
        for i in range(RAM_SIZE)
        if a["net"][i] > 50 and a["mono"][i] > 0.9
        and i not in mapping.values()
    ]
    cand.sort(key=lambda x: -x["net"])

    out = {
        "profile": args.profile, "rom": Path(rom).name,
        "probe_steps": {"hold_right": 900, "jump": len(seq_b),
                        "sticky_random": args.steps},
        "verdicts": verdicts, "progress_candidates": cand[:20],
    }
    game = Path(args.profile).stem
    outp = REPO / "runs/ram_verify" / f"{game}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    width = max(len(k) for k in verdicts) + 2
    print(f"\n{'mapping':<{width}}{'addr':>8}  ok   evidence")
    for k, v in verdicts.items():
        print(f"{k:<{width}}{v['addr']:>8}  {'YES' if v['ok'] else 'NO ':<4} "
              f"{v['why']}")
    print(f"\n[candidates] top unmapped progress bytes: "
          f"{[x['addr'] for x in cand[:8]]}")
    print(f"[out] {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
