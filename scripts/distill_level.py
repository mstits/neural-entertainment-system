"""Demo-action-augmented backward-curriculum distillation.

Turns a Go-Explore SOLUTION (a receipted action sequence that clears a level)
into a LEARNED, sticky-robust tile policy — the depth track the plain
self-imitation robustifier stalled on (it couldn't self-clear precise obstacles,
so those segments got no training signal).

The fix: at every backward-curriculum rung K (action index along the solution,
deep→shallow), the BC dataset ALWAYS includes the SOLUTION's exact (obs, action)
pairs for the suffix seq[K:] — so even obstacles self-imitation can't cross get
the expert's precise actions — PLUS the policy's own sticky self-clears from the
rung's start state (recovery, for sticky robustness). Acceptance is under sticky
(matches the honest gate). Because the expert suffix guarantees a signal at every
rung, there is no stall; the self-collected recovery + backward schedule build
sticky robustness from the flag backward to the entrance.

Honest metric: eval the final net cold sticky+jitter from the level entrance
(eval_game --start-state <entrance> --sequential --level-clear --sticky-prob 0.25
--start-jitter 16).

Usage:
  python scripts/distill_level.py \
      --entrance checkpoints/.../stage_03.state \
      --solution runs/ge_1_2_solve/solutions/sol_003.actions.npy \
      --profile configs/mario_1_2_solo.yaml \
      --out checkpoints/mario_1_2_distilled/net.pt
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.models.tile_policy import (  # noqa: E402
    TilePolicyNetwork, build_tile_policy_from_checkpoint,
)
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, resolve_encoder,
)
from src.utils.reward_functions import build_reward_function  # noqa: E402
from eval_game import _TilePolicy  # noqa: E402
from robustify_level import collect_demos_parallel, _bc_fit  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entrance", required=True)
    ap.add_argument("--solution", required=True, help=".npy action indices.")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="Warm-start net (default: fresh).")
    ap.add_argument("--rung-stride", type=int, default=40,
                    help="Action-index gap between backward rungs.")
    ap.add_argument("--self-clears", type=int, default=4,
                    help="Self-collected sticky clears to add per rung (0=off).")
    ap.add_argument("--accept-sticky", type=float, default=0.25)
    ap.add_argument("--sticky-prob", type=float, default=0.25)
    ap.add_argument("--explore-eps", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bc-epochs", type=int, default=120)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--episode-cap", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    # Net: warm-start or fresh.
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        net, _ = build_tile_policy_from_checkpoint(
            sd, num_actions=len(bitmasks), feature_dim=stacked_dim)
        net.load_state_dict(sd["net_state_dict"], strict=False)
    else:
        net = TilePolicyNetwork(num_actions=len(bitmasks),
                                feature_dim=stacked_dim)
    net.eval()
    pol = _TilePolicy(net, TileFeatureStacker(stack_size=stack_size,
                                              feature_dim=feature_dim),
                      extractor, False)
    reward_fn = build_reward_function(profile)
    reward_factory = lambda: build_reward_function(profile)  # noqa: E731

    # 1-worker pool for _bc_fit's sticky eval; W-worker pool for collection.
    pool = Pool(rom_path=ROM, num_workers=1, frame_skip=4)
    pool.set_headless(True)
    cpool = Pool(rom_path=ROM, num_workers=int(args.workers), frame_skip=4)
    cpool.set_headless(True)
    cpool.set_skip_preprocess(True)

    # Replay the solution from the entrance -> full (obs, action) sequence +
    # per-step start states (the rung anchors). Convention: load, NOOP, act.
    actions = np.load(args.solution).astype(np.int64)
    entrance_bytes = Path(args.entrance).read_bytes()
    pool.reset_all()
    pool.load_worker_state(0, entrance_bytes)
    stk = TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
    r = pool.step_all(np.zeros(1, dtype=np.uint8))
    obs = stk.reset(extractor.extract(r[0][2]))
    seq = []
    states = [pool.save_worker_state(0)]  # state before applying action 0
    for a in actions:
        seq.append((np.array(obs, dtype=np.int8, copy=True), int(a)))
        r = pool.step_all(np.array([bitmasks[int(a)]], dtype=np.uint8))
        obs = stk.push(extractor.extract(r[0][2]))
        states.append(pool.save_worker_state(0))  # state after action i
    N = len(actions)
    print(f"[distill] solution replayed: {N} actions, {len(seq)} obs-action "
          f"pairs, {len(states)} rung states", flush=True)

    # _bc_fit reads these off `args`.
    args.success_gx = 0

    # Backward rungs: deep (near flag) -> shallow (entrance).
    Ks = list(range(max(0, N - args.rung_stride), -1, -args.rung_stride))
    if Ks[-1] != 0:
        Ks.append(0)
    print(f"[distill] {len(Ks)} backward rungs at stride {args.rung_stride}: "
          f"{Ks[0]} -> 0", flush=True)

    last_rate = 0.0
    for ri, K in enumerate(Ks):
        blob = states[K]
        suffix = seq[K:]  # expert (obs,action) suffix from K to the flag
        self_demos = []
        if args.self_clears > 0:
            self_demos, _seen = collect_demos_parallel(
                net, extractor, feature_dim, stack_size, cpool, bitmasks, blob,
                args.max_steps, reward_factory, args.self_clears,
                args.episode_cap, tag=f":K{K}", explore_eps=args.explore_eps,
                sticky_prob=args.sticky_prob, workers=int(args.workers),
                base_seed=ri + 1,
            )
        demos = [suffix] + [d for d in self_demos if d]
        rate, best_sd = _bc_fit(net, pol, pool, bitmasks, blob, demos, args,
                                device, reward_fn, tag=f":K{K}")
        if best_sd is not None:
            net.load_state_dict(best_sd)
            net.eval()
        last_rate = rate
        print(f"[distill] rung K={K} (of {N}) suffix={len(suffix)} "
              f"self_clears={len(self_demos)} sticky_rate={rate:.2f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "net_state_dict": {k: v.detach().cpu()
                           for k, v in net.state_dict().items()},
        "distilled": {"solution": args.solution, "entrance": args.entrance,
                      "rung_stride": args.rung_stride,
                      "entrance_sticky_rate": last_rate},
    }, str(out))
    print(f"[distill] DONE. entrance-rung sticky_rate={last_rate:.2f} -> {out}",
          flush=True)
    pool.shutdown()
    cpool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
