"""Fate probe: locate the RAM state that decides a maze lineage's fate,
without any transition-marker assumptions.

For a sweep of checkpoint coordinates X: collect many same-channel states
crossing X (one random-policy stream), then roll each forward under FIXED
scripted probe policies and record the lineage's FATE (deepest gx reached
before its first loop-warp). Where fates diverge, L1-logistic the RAM at X
against fate-class and mutation-validate the surviving bytes. The checkpoint
sweep localizes the decision zone; the Lasso isolates the deciding state;
the mutation test proves causality. (Methodology proven on 4-4/gate-779;
needs no $06DE/074E/0750 semantics — fate is defined purely by the observable
warp event.)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

RAM_OFF = 13  # RAM offset inside save_worker_state blobs


def _gx(ram) -> int:
    return (int(ram[0x6D]) << 8) | int(ram[0x86])


def probe_policy(step: int, variant: int, rng) -> int:
    """Fixed scripted continuations (right-biased with periodic jumps),
    deterministic per variant — fate differences then reflect STATE, not luck."""
    if variant == 0:
        return 4 if (step % 24) < 6 else 3        # run + periodic long jump
    if variant == 1:
        return 3 if (step % 16) < 12 else 2       # run + short hops
    return 4 if (step % 40) < 10 else (1 if (step % 7) else 5)


def collect_at(pool, bm, root, X, n_want, rng, max_steps=900):
    """Same-channel crossing states at gx==X (first crossing per lineage)."""
    out = []
    while len(out) < n_want:
        pool.reset_all()
        pool.load_worker_state(0, root)
        r = pool.step_all(np.zeros(1, dtype=np.uint8))
        prev = _gx(bytes(r[0][2]))
        for s in range(max_steps):
            u = rng.random()
            a = 3 if u < 0.45 else (4 if u < 0.6 else (8 if u < 0.72 else int(rng.integers(len(bm)))))
            r = pool.step_all(np.array([bm[a]], dtype=np.uint8))
            ram = bytes(r[0][2]); gx = _gx(ram)
            if gx < 7000:
                if prev - gx > 100:
                    break                       # looped before X: restart
                if prev < X <= gx:
                    out.append((ram, pool.save_worker_state(0)))
                    break
                prev = gx
            if bool(r[0][3]):
                break
    return out


def collect_from_archive(pool, cells, X, n_want, rng):
    """Validated, settle-tested states from archived cells with gx in
    [X-16, X+16] (low loop-phase preferred for lineage diversity)."""
    keys = [k for k, c in cells.items()
            if isinstance(k, tuple) and getattr(c, "state", None) is not None
            and abs(k[-1] * 16 - X) <= 16]
    rng.shuffle(keys)
    keys.sort(key=lambda k: k[0] if isinstance(k[0], int) else 0)
    out = []
    for k in keys:
        st = cells[k].state
        pool.reset_all()
        pool.load_worker_state(0, st)
        r = pool.step_all(np.zeros(1, dtype=np.uint8))
        ram = bytes(r[0][2]); g0 = _gx(ram); ok = True; prev = g0
        for _ in range(15):
            r = pool.step_all(np.zeros(1, dtype=np.uint8))
            ram = bytes(r[0][2]); gx = _gx(ram)
            if gx < 7000 and prev - gx > 100: ok = False; break
            if int(ram[0x0E]) in (6, 11) or bool(r[0][3]): ok = False; break
            prev = gx if gx < 7000 else prev
        if ok and abs(g0 - X) <= 24:
            out.append((ram, st))
        if len(out) >= n_want:
            break
    return out


def fate_of(pool, bm, blob, variant, horizon=700):
    """Deepest gx before the first loop-warp under a fixed probe policy."""
    pool.reset_all()
    pool.load_worker_state(0, blob)
    r = pool.step_all(np.zeros(1, dtype=np.uint8))
    ram = bytes(r[0][2]); prev = _gx(ram); mx = prev
    rng = np.random.default_rng(variant)
    for s in range(horizon):
        a = probe_policy(s, variant, rng)
        r = pool.step_all(np.array([bm[a]], dtype=np.uint8))
        ram = bytes(r[0][2]); gx = _gx(ram)
        if gx < 7000:
            if prev - gx > 100:
                return mx                        # first loop: fate = depth
            mx = max(mx, gx); prev = gx
        if bool(r[0][3]):
            return mx
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-state", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoints", default="600,1200,1800,2400,3000,3400")
    ap.add_argument("--per-checkpoint", type=int, default=240)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/fate_probe_8_4")
    ap.add_argument("--archive", default=None,
                    help="Optional solver archive: source checkpoint states "
                         "from archived cells near X (validated + settled) "
                         "instead of replaying from the root — from-root "
                         "random crossings get exponentially rare at depth.")
    args = ap.parse_args()
    prof = yaml.safe_load(Path(args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    root = Path(args.root_state).read_bytes()
    rng = np.random.default_rng(args.seed)
    pool = Pool(rom_path="roms/Super Mario Bros. (World).nes",
                num_workers=1, frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    report = {}
    arch_cells = None
    if args.archive:
        with open(args.archive, "rb") as f:
            _a = pickle.load(f)
        arch_cells = _a["cells"] if isinstance(_a, dict) and "cells" in _a else _a
        print(f"[fate] archive sourced: {len(arch_cells)} cells", flush=True)
    for X in [int(x) for x in args.checkpoints.split(",")]:
        if arch_cells is not None:
            samples = collect_from_archive(pool, arch_cells, X,
                                           args.per_checkpoint, rng)
        else:
            samples = collect_at(pool, bm, root, X, args.per_checkpoint, rng)
        fates = []
        for ram, blob in samples:
            f = np.mean([fate_of(pool, bm, blob, v) for v in range(args.variants)])
            fates.append(f)
        fates = np.array(fates)
        lo, hi = np.percentile(fates, 25), np.percentile(fates, 75)
        spread = hi - lo
        print(f"[fate] X={X}: n={len(samples)} fate p25={lo:.0f} p75={hi:.0f} "
              f"spread={spread:.0f} max={fates.max():.0f}", flush=True)
        report[X] = {"n": len(samples), "p25": float(lo), "p75": float(hi),
                     "spread": float(spread), "max": float(fates.max())}
        # persist for the Lasso stage: states + fates
        with open(outdir / f"X{X}.pkl", "wb") as f:
            pickle.dump({"samples": samples, "fates": fates.tolist()}, f)
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    pool.shutdown()
    # decision-zone verdict: the deepest X where fates still diverge is where
    # the route state is live; Lasso runs there (stage 2, scripts/route_probe
    # fit_probe on top/bottom fate quartiles).
    div = [x for x, r in report.items() if r["spread"] > 150]
    print(f"[fate] checkpoints with divergent fates: {div}")


if __name__ == "__main__":
    main()
