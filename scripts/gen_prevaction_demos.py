"""DARS recovery demos with a_{t-1} — prev-executed-action augmentation.

A one-lever variant of generate_dars_recovery.py: alongside each surviving
(drifted_obs, expert_action) recovery pair it records the PREVIOUS EXECUTED
action and appends its one-hot (length = |action_space|) to the stacked tile
obs. The student therefore sees a_{t-1}, restoring the Markov property that
25%-sticky action noise breaks (the sole genuinely-untested "de-aliasing" lever
from the DR brief — vel_x/vel_y/on_ground/sub-tile-X are already in the tile
obs, so a_{t-1} is the only piece not already present).

Output width: stacked_dim (712) + num_actions (6) = 718. The previous executed
action at recovery step j is the action executed at step j-1; at j=0 it is the
last action the drift loop injected (prev_a after the drift phase). Everything
else — the DART restore/drift/expert-relabel loop and the survive+forward-
progress filter (alive AND last_d < d0-1) — is UNCHANGED from the DARS core.

Honest: no game internals; states come from the search's own solutions, noise
is the eval's own sticky model, "progress" is the wavefront distance (search-
derived), and a_{t-1} is the policy's own action — not a RAM read.
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
from src.utils.wavefront_reward import WavefrontPotential, wave_cell  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
R_PSTATE = 0x000E
R_LIVES = 0x075A
DEATH = (6, 11)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solution", required=True, help=".npy expert action seq.")
    ap.add_argument("--root-state", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dmap", required=True, help="wavefront .pkl distance map.")
    ap.add_argument("--out", required=True, help="output demo .npz.")
    ap.add_argument("--samples", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--max-drift", type=int, default=5)
    ap.add_argument("--recovery-window", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=None,
                    help="Only sample restore points whose wavefront distance "
                         "is >= this (targets a specific obstacle window).")
    ap.add_argument("--d-max", type=float, default=None,
                    help="Upper bound for the restore-point distance window.")
    ap.add_argument("--stall-frames", type=int, default=0,
                    help="After drift, hold NOOP for 0..this many steps so "
                         "Mario decelerates to STANDING before the expert "
                         "relabel — covers the stalled-at-obstacle states an "
                         "arriving policy actually occupies (the running-only "
                         "demo set has no coverage there).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    n_act = len(bitmasks)
    fs = int(profile.get("frame_skip", 4))
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim
    wp = WavefrontPotential.load(args.dmap)
    actions = np.load(args.solution).astype(np.int64)
    N = len(actions)
    root = Path(args.root_state).read_bytes()

    # We need, per solution state s_k, a saved emulator blob to restore. Replay
    # once on worker 0 and snapshot every step (cheap; µs restores).
    pool = Pool(rom_path=ROM, num_workers=int(args.workers), frame_skip=fs)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, root)

    def w0_step(a):
        x = np.zeros(args.workers, dtype=np.uint8)
        x[0] = bitmasks[a]
        return pool.step_all(x)[0][2]

    ram = w0_step(0)  # convention NOOP
    states = [pool.save_worker_state(0)]
    state_d = [wp._distance(ram)]
    for a in actions:
        ram = w0_step(int(a))
        states.append(pool.save_worker_state(0))
        state_d.append(wp._distance(ram))

    # Eligible restore indices: inside the distance window (if given) and far
    # enough from the end for a full recovery window.
    k_hi = max(1, N - args.recovery_window - args.max_drift)
    eligible = [k for k in range(k_hi)
                if (args.d_min is None or state_d[k] >= args.d_min)
                and (args.d_max is None or state_d[k] <= args.d_max)]
    if not eligible:
        raise SystemExit("[dars] no solution states inside the distance window")
    eligible = np.array(eligible)
    print(f"[dars] {len(eligible)}/{k_hi} restore points in window "
          f"d=[{args.d_min},{args.d_max}] stall<={args.stall_frames}",
          flush=True)

    obs_rows: list = []
    act_rows: list = []
    kept = 0
    # Parallel batches of `workers` samples.
    n_batches = (args.samples + args.workers - 1) // args.workers
    W = args.workers
    for b in range(n_batches):
        ks = eligible[rng.integers(0, len(eligible), size=W)]
        drifts = rng.integers(1, args.max_drift + 1, size=W)
        # Optional stall: 0..stall_frames extra NOOP steps after the drift so
        # Mario decelerates to standing (covers stalled-at-obstacle states).
        stalls = (rng.integers(0, args.stall_frames + 1, size=W)
                  if args.stall_frames > 0 else np.zeros(W, dtype=np.int64))
        for i in range(W):
            pool.load_worker_state(i, states[int(ks[i])])
        # Inject `drift` sticky-noise frames per worker (staggered lengths),
        # then hold NOOP for the worker's stall count.
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
                    acts[i] = bitmasks[0]  # stalling/settled workers hold NOOP
            pool.step_all(acts)
        # Expert-relabel recovery window: step the SOLUTION's actions from the
        # time-resynced index; capture (drifted obs, expert action). Per-worker
        # stacker seeded from the post-drift frame.
        stks = [TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
                for _ in range(W)]
        r = pool.step_all(np.zeros(W, dtype=np.uint8))
        obs = [stks[i].reset(extractor.extract(r[i][2])) for i in range(W)]
        d0 = [wp._distance(r[i][2]) for i in range(W)]
        pend = [[] for _ in range(W)]      # (obs_aug, action) buffer per worker
        alive = [True] * W
        last_d = list(d0)
        # Previous EXECUTED action per worker, threaded through the recovery
        # window so its one-hot (a_{t-1}) can be appended to the obs. At j=0 it
        # is the last action the drift loop injected (prev_a after the drift
        # phase); thereafter it is the expert action executed on the prior step.
        prev_exec = [int(prev_a[i]) for i in range(W)]
        for j in range(args.recovery_window):
            acts = np.zeros(W, dtype=np.uint8)
            idx = [min(N - 1, int(ks[i]) + int(drifts[i]) + j) for i in range(W)]
            for i in range(W):
                a = int(actions[idx[i]])
                if alive[i]:
                    onehot = np.zeros(n_act, dtype=np.int8)
                    onehot[prev_exec[i]] = 1
                    obs_aug = np.concatenate(
                        (np.asarray(obs[i], dtype=np.int8), onehot))
                    pend[i].append((obs_aug, a))
                acts[i] = bitmasks[a]
                prev_exec[i] = a  # executed this step -> previous for step j+1
            r = pool.step_all(acts)
            for i in range(W):
                ram = r[i][2]
                if int(ram[R_PSTATE]) in DEATH or bool(r[i][3]):
                    alive[i] = False
                last_d[i] = wp._distance(ram)
                obs[i] = stks[i].push(extractor.extract(ram))
        # Keep recoveries that survived AND progressed (distance decreased).
        for i in range(W):
            if alive[i] and last_d[i] < d0[i] - 1 and pend[i]:
                for o, a in pend[i]:
                    obs_rows.append(o)
                    act_rows.append(a)
                kept += 1
        if (b + 1) % 20 == 0:
            print(f"[dars] batch {b+1}/{n_batches}: kept {kept} recoveries, "
                  f"{len(obs_rows)} pairs", flush=True)
    pool.shutdown()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, n=1,
             obs_0=np.stack(obs_rows).astype(np.int8),
             act_0=np.array(act_rows, dtype=np.int64))
    print(f"[dars] DONE: {kept} recoveries, {len(obs_rows)} (obs+a_prev,action) "
          f"pairs, width={stacked_dim}+{n_act}={stacked_dim + n_act} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
