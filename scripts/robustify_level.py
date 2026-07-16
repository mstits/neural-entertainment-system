"""Self-imitation robustifier: distill a policy's own stochastic level
clears into its argmax path.

Phase A: run the checkpoint STOCHASTICALLY from a fixed entry state,
keeping the (obs, action) trajectories of episodes that clear the level
(LevelClearTracker predicate — same one the consolidation gate probes).
Phase B: behavior-clone those demos into the same net (CE on actions,
fine-tuned from the source checkpoint).
Phase C: greedy re-eval from the same entry; save the best clone.

Usage:
  python robustify.py --profile configs/smb_1_4_go_explore.yaml \
      --checkpoint checkpoints/mario_1_4_go_explore/vanilla_ppo_iter_02120.pt \
      --out checkpoints/mario_1_4_go_explore/robust_1_4.pt \
      [--start-state PATH] [--clears 8] [--episode-cap 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, resolve_encoder,
)
from src.training.smb_sequential import LevelClearTracker  # noqa: E402
from eval_game import _TilePolicy  # noqa: E402
from train_game import DEFAULT_ROMS  # noqa: E402


def make_policy(profile: dict, sd: dict, bitmasks) -> tuple:
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    net, is_recurrent = build_tile_policy_from_checkpoint(
        sd, num_actions=len(bitmasks), feature_dim=stacked_dim,
    )
    res = net.load_state_dict(sd["net_state_dict"], strict=False)
    assert not res.missing_keys, f"missing keys: {res.missing_keys}"
    net.eval()
    stacker = TileFeatureStacker(
        stack_size=stacked_dim // feature_dim, feature_dim=feature_dim,
    )
    return net, _TilePolicy(net, stacker, extractor, is_recurrent), is_recurrent


def run_episodes(pool, pol, bitmasks, blob, max_steps, n, mode, device,
                 seed0=0, collect=False):
    """mode: 'sample' | 'greedy'. Returns (clears, episodes, demos)."""
    demos, clears = [], 0
    for ep in range(n):
        g = torch.Generator().manual_seed(seed0 + ep)
        pool.reset_all()
        if blob is not None:
            pool.load_worker_state(0, blob)
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        obs = pol.reset(init[0])
        hidden = pol.initial_hidden(device)
        tracker = LevelClearTracker()
        tracker.update(init[0][2])
        traj = []
        for _ in range(max_steps):
            with torch.no_grad():
                logits, hidden = pol.logits(obs, hidden)
            if mode == "sample":
                probs = torch.softmax(logits[0], dim=-1)
                a = int(torch.multinomial(probs, 1, generator=g).item())
            else:
                a = int(torch.argmax(logits[0]).item())
            if collect:
                traj.append((np.array(obs, dtype=np.int8, copy=True), a))
            r = pool.step_all(np.array([bitmasks[a]], dtype=np.uint8))
            tracker.update(r[0][2])
            obs = pol.push(r[0])
            if tracker.seq_clear:
                clears += 1
                if collect:
                    demos.append(traj)
                break
            if bool(r[0][3]):
                break
    return clears, n, demos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--clears", type=int, default=8)
    ap.add_argument("--episode-cap", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--bc-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-episodes", type=int, default=6)
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    sd = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cpu")
    net, pol, is_recurrent = make_policy(profile, sd, bitmasks)
    assert not is_recurrent, "robustifier currently supports stateless nets"

    ss = args.start_state or profile.get("start_state_path")
    blob = Path(ss).read_bytes() if ss else None
    rom = DEFAULT_ROMS.get("mario", "roms/Super Mario Bros. (World).nes")
    pool = Pool(rom_path=rom, num_workers=1, frame_skip=4,
                start_state_path=None)

    # Phase A — collect stochastic clears.
    demos, seen, batch = [], 0, 40
    while len(demos) < args.clears and seen < args.episode_cap:
        c, n, d = run_episodes(
            pool, pol, bitmasks, blob, args.max_steps,
            min(batch, args.episode_cap - seen), "sample", device,
            seed0=seen, collect=True,
        )
        seen += n
        demos.extend(d)
        print(f"[collect] {seen} eps -> {len(demos)} clears", flush=True)
    if not demos:
        print("FAIL: no stochastic clears collected; nothing to distill")
        pool.shutdown()
        return 1

    X = torch.tensor(
        np.concatenate([np.stack([o for o, _ in t]) for t in demos]),
        dtype=torch.float32,
    ).squeeze(1) if demos[0][0][0].ndim > 1 else torch.tensor(
        np.concatenate([np.stack([o for o, _ in t]) for t in demos]),
        dtype=torch.float32,
    )
    if X.ndim == 3:  # (N, 1, D) from batched obs rows
        X = X.squeeze(1)
    Y = torch.tensor(
        np.concatenate([np.array([a for _, a in t]) for t in demos]),
        dtype=torch.long,
    )
    print(f"[bc] dataset: {X.shape[0]} steps from {len(demos)} clears")

    # Phase B — behavior-clone into the same net (fine-tune from source).
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_rate, best_sd = -1.0, None
    for epoch in range(args.bc_epochs):
        logits, _ = net.forward_ac(X)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == args.bc_epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net.forward_ac(X)[0].argmax(-1) == Y).float().mean()
            c, n, _ = run_episodes(
                pool, pol, bitmasks, blob, args.max_steps,
                args.eval_episodes, "greedy", device,
            )
            rate = c / n
            print(f"[bc] epoch {epoch+1} loss {loss.item():.4f} "
                  f"acc {acc.item():.3f} greedy {c}/{n}", flush=True)
            if rate > best_rate:
                best_rate = rate
                best_sd = {k: v.clone() for k, v in net.state_dict().items()}
            if rate == 1.0:
                break
            net.train()

    # Phase C — save the best greedy clone in trainer checkpoint format.
    net.load_state_dict(best_sd)
    out = dict(sd)
    out["net_state_dict"] = best_sd
    out["robustified"] = {
        "source": args.checkpoint, "demos": len(demos),
        "demo_steps": int(X.shape[0]), "greedy_rate": best_rate,
        "entry_state": str(ss),
    }
    torch.save(out, args.out)
    print(f"[done] greedy clear rate {best_rate:.2f} -> {args.out}")
    pool.shutdown()
    return 0 if best_rate > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
