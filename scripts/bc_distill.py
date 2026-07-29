"""Behavioral-cloning distillation of Go-Explore solution trajectories.

Stage 2 of the research-grounded pipeline (Blueprint §2C): take the (tile-obs,
action) demo pairs produced by replay_to_demos.py from the Go-Explore solver's
solution traces, and BC-pretrain a TilePolicyNetwork to reproduce them. This is
the first (imitation) half of distillation; sticky fine-tune / DAgger
robustification is a separate step.

Saves a checkpoint eval_game.py can load (shape-inferred), so the honest gate is
  eval_game.py --profile <p> --checkpoint <out>/vanilla_ppo_iter_00000.pt \
      --sequential --level-clear [--sticky-prob 0.25 --start-jitter 16]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.tile_policy import TilePolicyNetwork  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demos", required=True,
                    help="Glob of .npz demo files (obs_0, act_0).")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="Checkpoint dir to write.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--trunk-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    profile = yaml.safe_load(Path(args.profile).read_text())
    num_actions = len(profile["action_space"])

    files = sorted(glob.glob(args.demos))
    if not files:
        raise SystemExit(f"no demos matched {args.demos!r}")
    obs_list, act_list = [], []
    for f in files:
        d = np.load(f)
        obs_list.append(d["obs_0"].astype(np.float32))
        act_list.append(d["act_0"].astype(np.int64))
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    feature_dim = obs.shape[1]
    print(f"[bc] {len(files)} demos -> {obs.shape[0]} pairs, feature_dim="
          f"{feature_dim}, num_actions={num_actions}", flush=True)
    # Class balance (BC on a right-biased trajectory is imbalanced).
    binc = np.bincount(act, minlength=num_actions)
    print(f"[bc] action histogram: {binc.tolist()}", flush=True)

    X = torch.from_numpy(obs)
    Y = torch.from_numpy(act)
    net = TilePolicyNetwork(num_actions=num_actions, feature_dim=feature_dim,
                            hidden_dim=args.hidden_dim, trunk_dim=args.trunk_dim)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n = X.shape[0]
    net.train()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot_loss = tot_correct = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = X[idx], Y[idx]
            logits, _ = net.forward_ac(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss) * len(idx)
            tot_correct += int((logits.argmax(-1) == yb).sum())
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep:3d}  loss={tot_loss/n:.4f}  "
                  f"train_acc={tot_correct/n:.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "vanilla_ppo_iter_00000.pt"
    torch.save({"net_state_dict": {k: v.detach().cpu()
                                   for k, v in net.state_dict().items()},
                "iter": 0, "provenance": "bc_distill",
                "demos": files}, str(ckpt))
    print(f"[bc] saved {ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
