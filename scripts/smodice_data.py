"""SMODICE offline-transition builder — unconditional recording, labeled windows.

Same DART replay mechanism and CLI as gen_iq_transitions.py (restore a
verified solution state, inject sticky-0.25 drift, expert-relabel a recovery
window from the time-resynced solution index) so it runs in the same attended
window, but corrects the two structural defects that left the IQ-Learn set
with done=0 on every row (HONEST_1_2_FINDINGS_2026-08-14 §C):

  1. Per-step transitions are recorded UNCONDITIONALLY while the window runs.
     A worker that dies keeps its terminal transition with done=1 (absorbing
     state) instead of having the whole window survivorship-filtered away.
  2. A window that survives all --recovery-window steps is a TIMEOUT: kept
     with done=0 and truncated=1 on its last row, so downstream losses
     bootstrap V(s') (Pardo 2018 partial-episode bootstrapping) instead of
     grounding it.

The forward-progress test (wavefront distance decreases) is a LABEL
(window_progressed), not a discard gate. The final output is stratified to a
~50/50 mix of successful-recovery windows vs failed windows (deaths + stalls)
by subsampling the majority class; composition counts are printed and stored
in the npz metadata.

Output npz schema (row-aligned, N rows):
  state              int8  (N, 712)  stacked tile observation s_t
  action             int64 (N,)      expert-relabeled action a_t
  next_state         int8  (N, 712)  s_{t+1}
  done               int8  (N,)      1 only on a death transition (absorbing)
  truncated          int8  (N,)      1 only on the last row of a timeout window
  is_expert_window   int8  (N,)      1 — every action here is expert-relabeled
  window_id          int64 (N,)      window index shared by all rows of a window
  window_progressed  int8  (N,)      window label: survived AND wavefront
                                     distance decreased by > 1
  meta_json          str   ()        json: composition counts + sha256
                                     provenance of the source solution files

Honest: no game internals; states from the search's own solutions, noise is
the eval's own sticky model, "progress" is the search-derived wavefront.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
R_PSTATE = 0x000E
DEATH = (6, 11)


def append_step(rows, s, a, ns, dead):
    """Append one transition row [s, a, ns, done, truncated] unconditionally."""
    rows.append([np.asarray(s, dtype=np.int8).copy(), int(a),
                 np.asarray(ns, dtype=np.int8).copy(), int(bool(dead)), 0])


def finalize_window(rows, died):
    """Mark a survived window's last row truncated=1 (timeout, done stays 0)."""
    if not died and rows:
        rows[-1][4] = 1
    return rows


def record_window(obs0, expert_actions, step_fn):
    """Record one recovery window against a stepper, stubbed or real.

    step_fn(action) -> (next_obs, dead). Records every step; a death keeps
    its terminal transition (done=1) and stops the window (absorbing state);
    surviving every step yields a timeout (done=0, truncated=1 on the last
    row). Returns (rows, died).
    """
    rows = []
    obs = np.asarray(obs0, dtype=np.int8)
    died = False
    for a in expert_actions:
        next_obs, dead = step_fn(int(a))
        next_obs = np.asarray(next_obs, dtype=np.int8)
        append_step(rows, obs, a, next_obs, dead)
        if dead:
            died = True
            break
        obs = next_obs
    return finalize_window(rows, died), died


def classify_window(died, d_start, d_end):
    """window_progressed label: survived AND wavefront distance decreased."""
    return (not died) and (d_end < d_start - 1.0)


def window_kind(died, progressed):
    if died:
        return "death"
    return "progressed" if progressed else "stall"


def stratify_windows(windows, rng):
    """Subsample the majority class to a ~50/50 progressed/failed mix.

    windows: dicts with keys id, rows, progressed, kind. Returns
    (kept_windows, composition_counts). Fails loud if either class is
    empty — a one-class dataset would silently reintroduce the
    survivorship bias this generator exists to remove.
    """
    prog = [w for w in windows if w["progressed"]]
    fail = [w for w in windows if not w["progressed"]]
    if not prog or not fail:
        raise ValueError(
            f"cannot stratify: {len(prog)} progressed vs {len(fail)} failed "
            "windows — both classes must be non-empty for the 50/50 mix")
    k = min(len(prog), len(fail))

    def _sub(group):
        if len(group) <= k:
            return list(group)
        idx = sorted(rng.choice(len(group), size=k, replace=False).tolist())
        return [group[i] for i in idx]

    kept_p, kept_f = _sub(prog), _sub(fail)
    kept = sorted(kept_p + kept_f, key=lambda w: w["id"])
    counts = {
        "windows_total": len(windows),
        "progressed_total": len(prog),
        "failed_total": len(fail),
        "deaths_total": sum(1 for w in windows if w["kind"] == "death"),
        "stalls_total": sum(1 for w in windows if w["kind"] == "stall"),
        "kept_progressed": len(kept_p),
        "kept_failed": len(kept_f),
        "kept_deaths": sum(1 for w in kept_f if w["kind"] == "death"),
        "kept_stalls": sum(1 for w in kept_f if w["kind"] == "stall"),
        "kept_transitions": sum(len(w["rows"]) for w in kept),
    }
    return kept, counts


def pack_windows(windows):
    """Flatten window records into the row-aligned npz arrays."""
    S, A, NS, D, T, WID, PROG = [], [], [], [], [], [], []
    for w in windows:
        for (s, a, ns, dn, tr) in w["rows"]:
            S.append(s); A.append(a); NS.append(ns); D.append(dn); T.append(tr)
            WID.append(w["id"]); PROG.append(int(w["progressed"]))
    return {
        "state": np.stack(S).astype(np.int8),
        "action": np.array(A, dtype=np.int64),
        "next_state": np.stack(NS).astype(np.int8),
        "done": np.array(D, dtype=np.int8),
        "truncated": np.array(T, dtype=np.int8),
        "is_expert_window": np.ones(len(A), dtype=np.int8),
        "window_id": np.array(WID, dtype=np.int64),
        "window_progressed": np.array(PROG, dtype=np.int8),
    }


def solution_provenance(paths):
    """sha256 of each source solution file, keyed by the path as given."""
    prov = {}
    for p in paths:
        h = hashlib.sha256()
        h.update(Path(p).read_bytes())
        prov[str(p)] = h.hexdigest()
    return prov


def save_dataset(out, windows, counts, provenance):
    arrays = pack_windows(windows)
    meta = {"counts": counts, "provenance": provenance}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, meta_json=np.array(json.dumps(meta, sort_keys=True)),
             **arrays)
    return arrays


def replay_states(pool, bitmasks, actions, wp, W):
    """Replay the solution on worker 0, snapshot every step + its distance."""
    def w0_step(a):
        x = np.zeros(W, dtype=np.uint8)
        x[0] = bitmasks[a]
        return pool.step_all(x)[0][2]

    ram = w0_step(0)
    states = [pool.save_worker_state(0)]
    dists = [wp._distance(ram)]
    for a in actions:
        ram = w0_step(int(a))
        states.append(pool.save_worker_state(0))
        dists.append(wp._distance(ram))
    return states, dists


def gen_for_solution(pool, args, bitmasks, extractor, stack_size, feature_dim,
                     wp, sol_path, rng, budget, next_window_id,
                     TileFeatureStacker):
    """Return (window records, next_window_id) for one solution."""
    actions = np.load(sol_path).astype(np.int64)
    N = len(actions)
    root = Path(args.root_state).read_bytes()
    W = args.workers

    pool.reset_all()
    pool.load_worker_state(0, root)
    states, dists = replay_states(pool, bitmasks, actions, wp, W)

    k_hi = max(1, N - args.recovery_window - args.max_drift)
    eligible = [k for k in range(k_hi)
                if (args.d_min is None or dists[k] >= args.d_min)
                and (args.d_max is None or dists[k] <= args.d_max)]
    if not eligible:
        print(f"[smodice] {sol_path.name}: no restore points in window; skip",
              flush=True)
        return [], next_window_id
    eligible = np.array(eligible)
    print(f"[smodice] {sol_path.name}: N={N} {len(eligible)}/{k_hi} restore "
          f"points, budget={budget}", flush=True)

    windows = []
    rows_total = 0
    n_batches = (args.samples + W - 1) // W
    for b in range(n_batches):
        if rows_total >= budget:
            break
        ks = eligible[rng.integers(0, len(eligible), size=W)]
        drifts = rng.integers(1, args.max_drift + 1, size=W)
        stalls = (rng.integers(0, args.stall_frames + 1, size=W)
                  if args.stall_frames > 0 else np.zeros(W, dtype=np.int64))
        for i in range(W):
            pool.load_worker_state(i, states[int(ks[i])])
        prev_a = rng.integers(0, len(bitmasks), size=W)
        maxd = int((drifts + stalls).max())
        for step in range(maxd):
            acts = np.zeros(W, dtype=np.uint8)
            for i in range(W):
                if step < drifts[i]:
                    a = (prev_a[i] if rng.random() < args.sticky
                         else rng.integers(0, len(bitmasks)))
                    prev_a[i] = a
                    acts[i] = bitmasks[int(a)]
                else:
                    acts[i] = bitmasks[0]
            pool.step_all(acts)
        # Expert relabel; record every transition per worker, unconditionally.
        stks = [TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
                for _ in range(W)]
        r = pool.step_all(np.zeros(W, dtype=np.uint8))
        obs = [stks[i].reset(extractor.extract(r[i][2])) for i in range(W)]
        d0 = [wp._distance(r[i][2]) for i in range(W)]
        rows = [[] for _ in range(W)]
        alive = [True] * W
        last_d = list(d0)
        for j in range(args.recovery_window):
            acts = np.zeros(W, dtype=np.uint8)
            idx = [min(N - 1, int(ks[i]) + int(drifts[i]) + j) for i in range(W)]
            s_before = [None] * W
            for i in range(W):
                a = int(actions[idx[i]])
                if alive[i]:
                    s_before[i] = np.asarray(obs[i], dtype=np.int8).copy()
                acts[i] = bitmasks[a]
            r = pool.step_all(acts)
            for i in range(W):
                ram = r[i][2]
                dead = int(ram[R_PSTATE]) in DEATH or bool(r[i][3])
                new_obs = stks[i].push(extractor.extract(ram))
                if alive[i]:
                    append_step(rows[i], s_before[i], int(actions[idx[i]]),
                                new_obs, dead)
                    last_d[i] = wp._distance(ram)
                if dead:
                    alive[i] = False
                obs[i] = new_obs
        for i in range(W):
            died = not alive[i]
            finalize_window(rows[i], died)
            if not rows[i]:
                continue
            progressed = classify_window(died, d0[i], last_d[i])
            windows.append({"id": next_window_id, "rows": rows[i],
                            "progressed": progressed,
                            "kind": window_kind(died, progressed)})
            next_window_id += 1
            rows_total += len(rows[i])
        if (b + 1) % 20 == 0:
            print(f"[smodice]   batch {b+1}/{n_batches}: {len(windows)} "
                  f"windows, {rows_total} transitions", flush=True)
    print(f"[smodice] {sol_path.name}: {len(windows)} windows, {rows_total} "
          f"transitions", flush=True)
    return windows, next_window_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solutions", nargs="+", required=True,
                    help="One or more .npy expert action seqs (diverse).")
    ap.add_argument("--root-state", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3000,
                    help="Drift samples attempted per solution.")
    ap.add_argument("--target", type=int, default=70000,
                    help="Stop once this many raw (pre-stratification) "
                         "transitions are collected.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--max-drift", type=int, default=5)
    ap.add_argument("--recovery-window", type=int, default=60)
    ap.add_argument("--stall-frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=None)
    ap.add_argument("--d-max", type=float, default=None)
    args = ap.parse_args()

    # Emulator imports live here so the commit-logic units above stay
    # importable in test environments without the built core.
    from nes_core import Pool
    from src.emulation.frame_utils import TileFeatureStacker
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder,
    )
    from src.utils.wavefront_reward import WavefrontPotential

    rng = np.random.default_rng(args.seed)
    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim
    wp = WavefrontPotential.load(args.dmap)

    pool = Pool(rom_path=ROM, num_workers=int(args.workers), frame_skip=fs)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)

    windows = []
    next_window_id = 0
    rows_so_far = 0
    n_sols = len(args.solutions)
    for si, sol in enumerate(args.solutions):
        remaining = args.target - rows_so_far
        if remaining <= 0:
            break
        budget = max(1, remaining // max(1, n_sols - si))
        ws, next_window_id = gen_for_solution(
            pool, args, bitmasks, extractor, stack_size, feature_dim, wp,
            Path(sol), rng, budget, next_window_id, TileFeatureStacker)
        windows.extend(ws)
        rows_so_far = sum(len(w["rows"]) for w in windows)
        print(f"[smodice] total so far: {len(windows)} windows, "
              f"{rows_so_far} transitions", flush=True)
    pool.shutdown()

    if not windows:
        raise SystemExit("[smodice] no windows collected")
    kept, counts = stratify_windows(windows, rng)
    prov = solution_provenance(list(args.solutions) + [args.root_state])
    arrays = save_dataset(args.out, kept, counts, prov)
    print(f"[smodice] composition: {json.dumps(counts, sort_keys=True)}",
          flush=True)
    print(f"[smodice] DONE: {len(arrays['action'])} transitions "
          f"(done={int(arrays['done'].sum())}, "
          f"truncated={int(arrays['truncated'].sum())}) -> {args.out}",
          flush=True)
    print(f"[smodice] action histogram: "
          f"{np.bincount(arrays['action'], minlength=len(bitmasks)).tolist()}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
