"""Observatory v1 — automated observable discovery over a solver archive.

The manual per-game discovery loop (differential probes, byte scans) costs
about an hour per game and misses mode bits until a wall diagnosis forces
them (the CV stair bit took 4.2M wasted steps to notice). This pass
automates the first rung: dispersion-guided abstraction refinement — find
bytes whose VALUE explains where short probe rollouts end up, beyond what
the existing cell coordinates already explain.

Method (v1, CPU only):
  1. Stratified-sample N archived cells (unique (y-band, gx-bucket) coords
     first, then fill randomly).
  2. Restore each cell state in a pool worker; run each of P scripted
     micro-probes (hold-right / hold-left / jump / up / down / noop) for
     T steps from the SAME state; record the outcome (progress delta
     bucket, y delta bucket, death).
  3. For each low-cardinality byte (2..MAX_CARD distinct values across the
     corpus), compute mutual information between the byte's value at the
     probe START and the outcome, conditioned per probe, summed over
     probes. Bytes already spanned by the cell key (progress/y/level/lives
     addresses) are excluded.
  4. Rank; emit a receipt JSON with per-byte value->outcome histograms so
     a human (or a later pass) can verify the byte is causal, not
     correlated furniture.

Acceptance test (run it after any change):
  python scripts/observatory.py --run runs/cv_chain_hw/lvl_02 \
      --profile configs/castlevania.yaml --cells 400 --expect 0x0004==8
  must rank $0004==8 (the receipted stair-mode bit, discovered manually
  2026-07-29 after the block-2 staircase seal) in the top 5 — WITHOUT
  being told about it.

  STATUS 2026-07-31: NOT YET MET (rank ~1276/3388). The predicate's
  signal is real and verified in isolation (differential up-probe dy:
  -15.6 mean on-stairs vs exactly 0 across 40 off-stair cells) but a
  global U-statistic dilutes it: most ==8 cells sit where up-motion is
  exhausted (stair tops). v2 needs the per-coordinate conditioning of
  the original manual method (within one aliased (y,gx) cell, does the
  byte separate outcomes?) — the true "dispersion-triggered
  abstraction refinement" structure. Iteration receipts: raw MI failed
  on cardinality bias (top-20 all counters); byte-level candidates
  failed on wide mode bytes (cardinality-filter excluded $0004
  entirely -> predicates); raw outcomes failed on gravity (differential
  vs per-cell noop outcome fixed that); distance buckets diluted
  enable-bits (sign classes fixed that).

Purity: probes run on our own archived states with generic inputs; no
disassembly, no external maps. Candidates it emits still need the
verify_ram_map receipt process before entering a production profile.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from go_explore_solve import (  # noqa: E402
    make_game,
    action_space_to_bitmasks,
)
from src.training.go_explore import GoExploreArchive  # noqa: E402

RAM_OFF = 13          # NCST\x01 magic (5) + bincode u64 len (8)
RAM_LEN = 2048
MAX_CARD = 32         # candidate bytes: 2..MAX_CARD distinct values
PROBE_STEPS = 24


def blob_ram(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob[RAM_OFF:RAM_OFF + RAM_LEN], dtype=np.uint8)


def build_probes(profile: dict) -> list[tuple[str, list[int]]]:
    """Scripted micro-probes from the profile's own action space."""
    names = [tuple(a) for a in profile["action_space"]]

    def find(want: set[str]):
        for i, a in enumerate(names):
            if set(a) == want:
                return i
        return None

    probes = []
    for label, want in (
        ("right", {"right"}), ("left", {"left"}), ("jump", {"A"}),
        ("up", {"up"}), ("down", {"down"}),
    ):
        idx = find(want)
        if idx is not None:
            probes.append((label, [idx] * PROBE_STEPS))
    jr = find({"right", "A"})
    if jr is not None:
        probes.append(("jump_right", [jr] * PROBE_STEPS))
    probes.append(("noop", [0] * PROBE_STEPS))
    return probes


def outcome_label(game, ram0, ram1, died: bool,
                  noop_dp: int = 0, noop_dy: int = 0) -> tuple:
    """Discretized probe outcome, DIFFERENTIAL against the same cell's
    no-op drift. Raw deltas drown mode signals in physics: a mid-air
    state "moves" under any input because gravity moves it (v1.2
    acceptance failure: on-stairs |dy| 19.6 vs off-stairs 14.7 under
    up-holds — barely separable raw, cleanly separable after
    subtracting the cell's own no-op outcome)."""
    if died:
        return ("dead",)
    dp = int(game.progress(ram1)) - int(game.progress(ram0)) - noop_dp
    dy = int(game.y(ram1)) - int(game.y(ram0)) - noop_dy
    # Sign classes, not distance buckets: a mode bit that ENABLES motion
    # produces varying distances (climb 8px or 45px depending on the
    # staircase); spreading those over buckets dilutes the byte's U.
    def sgn(d):
        return 0 if abs(d) <= 4 else (1 if d > 0 else -1)
    return (sgn(dp), sgn(dy))


def explained_fraction(pairs: list[tuple[int, tuple]]) -> float:
    """Bias-corrected uncertainty coefficient U(outcome|value): the
    fraction of outcome entropy the byte's value explains.

    Raw MI is cardinality-biased — a 30-value frame counter beats a
    2-value mode bit on finite samples even when the bit is causal
    (v1.0 acceptance failure: top-20 all card 16-31 counters). The
    normalization is the v6 recipe's own criterion ("byte must cut
    transition entropy"), and Miller-Madow corrects the small-sample
    inflation (~(|V|-1)(|O|-1)/(2N ln2))."""
    n = len(pairs)
    if n < 8:
        return 0.0
    cv, co, cj = Counter(), Counter(), Counter()
    for v, o in pairs:
        cv[v] += 1
        co[o] += 1
        cj[(v, o)] += 1
    h_o = -sum((c / n) * np.log2(c / n) for c in co.values())
    if h_o < 1e-9:
        return 0.0
    mi = 0.0
    for (v, o), c in cj.items():
        p = c / n
        mi += p * np.log2(p / ((cv[v] / n) * (co[o] / n)))
    bias = (len(cv) - 1) * (len(co) - 1) / (2.0 * n * np.log(2.0))
    return float(max(0.0, mi - bias) / h_o)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="solver run dir (archive.pkl)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cells", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="receipt JSON (default: <run>/observatory_v1.json)")
    ap.add_argument("--expect", default=None,
                    help="acceptance predicate 'ADDR==VAL' (hex addr); prints its rank")
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    game = make_game(profile)
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    probes = build_probes(profile)
    rng = np.random.default_rng(args.seed)

    arch = GoExploreArchive(game.cell_fn, seed=0)
    arch.load(Path(args.run) / "archive.pkl")
    cells = [c for c in arch.cells.values() if c.state]
    print(f"[observatory] archive: {len(arch.cells)} cells, {len(cells)} with states")

    # Stratified sample: one per (y-band, gx-bucket) first, then random fill.
    by_coord: dict[tuple, list] = defaultdict(list)
    for c in cells:
        by_coord[(c.key[-2], c.key[-1])].append(c)
    coords = list(by_coord)
    rng.shuffle(coords)
    sample = [by_coord[k][rng.integers(len(by_coord[k]))] for k in coords[: args.cells]]
    while len(sample) < min(args.cells, len(cells)):
        sample.append(cells[rng.integers(len(cells))])
    print(f"[observatory] sampled {len(sample)} cells over {len(coords)} coords, "
          f"{len(probes)} probes x {PROBE_STEPS} steps")

    # Candidate bytes: low cardinality across the corpus, excluding the
    # profile's own coordinate addresses (they explain outcomes trivially).
    corpus = np.stack([blob_ram(c.state) for c in sample])
    solve = profile.get("solve", {})
    excluded = set()
    if solve:
        p = solve.get("progress", {})
        excluded |= {int(p.get("lo", -1)), int(p.get("hi", -1)),
                     int(solve.get("y", -1)), int(solve.get("lives", -1))}
        excluded |= {int(a) for a in solve.get("level_key", [])}
    # Candidates are PREDICATES (addr, value) — the state_sig schema.
    # A mode byte can be wide (CV's $0004 spans ~70 animation values;
    # only ==8 means stairs) — filtering bytes by cardinality throws
    # the needle out with the hay (v1.1 acceptance failure). A value
    # qualifies with support in [3%, 97%] of the corpus.
    n_c = len(sample)
    lo_sup = max(8, int(0.03 * n_c))
    cand = []
    for b in range(RAM_LEN):
        if b in excluded:
            continue
        vals, counts = np.unique(corpus[:, b], return_counts=True)
        if len(vals) < 2:
            continue
        for v, k in zip(vals, counts):
            if lo_sup <= k <= n_c - lo_sup:
                cand.append((int(b), int(v)))
    print(f"[observatory] {len(cand)} candidate predicates (addr==value)")

    # Probe rollouts. One pool, sequential restores (v1 keeps it simple;
    # the pass is minutes, not the solver's problem size).
    pool = Pool(rom_path=game.rom, num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    act = np.zeros(1, dtype=np.uint8)

    def run_probe(state, action_seq):
        pool.load_worker_state(0, state)
        act[0] = 0
        ram0 = np.frombuffer(pool.step_all(act)[0][2], dtype=np.uint8).copy()
        lives0 = game.lives(ram0)
        ram1, died = ram0, False
        for a in action_seq:
            act[0] = bitmasks[int(a)]
            ram1 = np.frombuffer(pool.step_all(act)[0][2], dtype=np.uint8)
            if game.is_dead(ram1, lives0):
                died = True
                break
        return ram0, ram1, died

    records = []  # (cell_idx, probe_label, differential outcome)
    scored_probes = [(l, seq) for l, seq in probes if l != "noop"]
    noop_seq = next(seq for l, seq in probes if l == "noop")
    for ci, c in enumerate(sample):
        r0, r1, ndied = run_probe(c.state, noop_seq)
        noop_dp = int(game.progress(r1)) - int(game.progress(r0))
        noop_dy = int(game.y(r1)) - int(game.y(r0))
        for label, action_seq in scored_probes:
            ram0, ram1, died = run_probe(c.state, action_seq)
            records.append((ci, label,
                            outcome_label(game, ram0, ram1, died,
                                          noop_dp, noop_dy)))
        if (ci + 1) % 50 == 0:
            print(f"[observatory] probed {ci + 1}/{len(sample)} cells")
    pool.shutdown()

    # Per-byte explained-outcome fraction, conditioned on probe; a byte
    # scores by its BEST probe (a mode bit that only matters for one
    # input direction — the stair bit under up-holds — must not be
    # diluted by the six probes it says nothing about).
    scores = []
    by_probe = defaultdict(list)
    for ci, pl, out in records:
        by_probe[pl].append((ci, out))
    for b, v in cand:
        ind = (corpus[:, b] == v)
        best, best_probe = 0.0, None
        for label in by_probe:
            pairs = [(int(ind[ci]), out) for ci, out in by_probe[label]]
            u = explained_fraction(pairs)
            if u > best:
                best, best_probe = u, label
        scores.append((float(best), int(b), int(v), best_probe))
    scores.sort(reverse=True)

    top = []
    for s, b, v, bp in scores[:20]:
        ind = (corpus[:, b] == v)
        hist = {}
        for label in by_probe:
            per_val = defaultdict(Counter)
            for ci, out in by_probe[label]:
                per_val[int(ind[ci])][str(out)] += 1
            hist[label] = {str(k): dict(cnt) for k, cnt in sorted(per_val.items())}
        top.append({
            "predicate": f"0x{b:04X} == {v}",
            "explained_frac": round(s, 3),
            "support": int(ind.sum()),
            "best_probe": bp,
            "outcome_hist": hist,
        })
        print(f"  ${b:04X} == {v:3d}  U {s:6.3f}  support {int(ind.sum()):3d}  best probe: {bp}")

    if args.expect:
        a_s, v_s = args.expect.split("==")
        want = (int(a_s, 16), int(v_s))
        rank = next((i for i, (_, b, v, _bp) in enumerate(scores)
                     if (b, v) == want), None)
        u = next((sc for sc, b, v, _bp in scores if (b, v) == want), None)
        print(f"[observatory] ACCEPTANCE {args.expect}: rank "
              f"{rank if rank is not None else 'not-a-candidate'} "
              f"(U {u if u is not None else '-'}) of {len(scores)}")

    out_path = Path(args.out) if args.out else Path(args.run) / "observatory_v1.json"
    out_path.write_text(json.dumps({
        "run": str(args.run), "profile": str(args.profile),
        "cells_sampled": len(sample), "probes": [p for p, _ in probes],
        "probe_steps": PROBE_STEPS, "candidates": len(cand),
        "top": top,
    }, indent=1))
    print(f"[observatory] receipt -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
