"""IQ-Learn transition builder — recovery transitions for offline soft-Bellman.

Same DART replay mechanism as generate_dars_recovery.py (restore a verified
solution state, inject sticky-0.25 drift to reach an off-equilibrium state, then
have the expert relabel a recovery window from the time-resynced solution
index), but instead of emitting (obs, action) demo pairs this records the full
Markov transition tuple every offline soft-Bellman method needs:

    (state_t, action_t, state_{t+1}, done_t, is_expert=1)

for each step of a surviving, forward-progressing recovery. States are the
stacked (712,) tile observations, int8. Only recoveries that SURVIVE the window
and progress toward the goal (wavefront distance decreases) are kept — the same
genuine-recovery filter as DARS, so the transition set is expert-quality
return-to-path dynamics, not doomed drifts.

Aggregates over several diverse Go-Explore solutions and stops at --target
transitions. Honest: no game internals; states from the search's own solutions,
noise is the eval's own sticky model, "progress" is the search-derived wavefront.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, resolve_encoder,
)
from src.utils.wavefront_reward import WavefrontPotential  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
R_PSTATE = 0x000E
DEATH = (6, 11)


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
                     wp, sol_path, rng, budget):
    """Return lists (state, action, next_state, done) for one solution."""
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
        print(f"[iq] {sol_path.name}: no restore points in window; skip",
              flush=True)
        return [], [], [], []
    eligible = np.array(eligible)
    print(f"[iq] {sol_path.name}: N={N} {len(eligible)}/{k_hi} restore points, "
          f"budget={budget}", flush=True)

    s_rows, a_rows, ns_rows, d_rows = [], [], [], []
    kept = 0
    n_batches = (args.samples + W - 1) // W
    for b in range(n_batches):
        if len(s_rows) >= budget:
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
        # Expert relabel; capture full transitions per surviving worker.
        stks = [TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
                for _ in range(W)]
        r = pool.step_all(np.zeros(W, dtype=np.uint8))
        obs = [stks[i].reset(extractor.extract(r[i][2])) for i in range(W)]
        d0 = [wp._distance(r[i][2]) for i in range(W)]
        pend = [[] for _ in range(W)]   # (s, a, s', done)
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
                    pend[i].append((s_before[i], int(actions[idx[i]]),
                                    np.asarray(new_obs, dtype=np.int8).copy(),
                                    bool(dead)))
                if dead:
                    alive[i] = False
                last_d[i] = wp._distance(ram)
                obs[i] = new_obs
        for i in range(W):
            if alive[i] and last_d[i] < d0[i] - 1 and pend[i]:
                for (s, a, ns, dn) in pend[i]:
                    s_rows.append(s); a_rows.append(a)
                    ns_rows.append(ns); d_rows.append(dn)
                kept += 1
        if (b + 1) % 20 == 0:
            print(f"[iq]   batch {b+1}/{n_batches}: kept {kept} recoveries, "
                  f"{len(s_rows)} transitions", flush=True)
    print(f"[iq] {sol_path.name}: kept {kept} recoveries, {len(s_rows)} "
          f"transitions", flush=True)
    return s_rows, a_rows, ns_rows, d_rows


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
                    help="Stop once this many transitions are collected.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--max-drift", type=int, default=5)
    ap.add_argument("--recovery-window", type=int, default=60)
    ap.add_argument("--stall-frames", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=None)
    ap.add_argument("--d-max", type=float, default=None)
    args = ap.parse_args()

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

    S, A, NS, D = [], [], [], []
    n_sols = len(args.solutions)
    for si, sol in enumerate(args.solutions):
        remaining = args.target - len(S)
        if remaining <= 0:
            break
        # Even-ish split of remaining target across the still-unvisited sols.
        budget = max(1, remaining // max(1, n_sols - si))
        s, a, ns, d = gen_for_solution(
            pool, args, bitmasks, extractor, stack_size, feature_dim, wp,
            Path(sol), rng, budget)
        S.extend(s); A.extend(a); NS.extend(ns); D.extend(d)
        print(f"[iq] total so far: {len(S)} transitions", flush=True)
    pool.shutdown()

    if not S:
        raise SystemExit("[iq] no transitions collected")
    state = np.stack(S).astype(np.int8)
    next_state = np.stack(NS).astype(np.int8)
    action = np.array(A, dtype=np.int64)
    done = np.array(D, dtype=np.int8)
    is_expert = np.ones(len(S), dtype=np.int8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, state=state, action=action, next_state=next_state,
             done=done, is_expert=is_expert)
    print(f"[iq] DONE: {len(S)} transitions (done={int(done.sum())}) -> {out}",
          flush=True)
    print(f"[iq] action histogram: "
          f"{np.bincount(action, minlength=len(bitmasks)).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
