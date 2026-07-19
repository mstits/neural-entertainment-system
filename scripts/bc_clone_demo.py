"""BC-overfit a pilot net: perfect greedy replay of ONE verified demo.

The routed chain is deterministic and a gx-switch arrival state is
bit-identical run to run, so a seam stage needs one reliable path, not
robustness — and a single-trajectory clone at 100% action accuracy IS
that path (this is what closed 2-1's seam). Anchoring MANY solutions
from one root fails structurally: they teach contradictory actions near
the root and the anchor loss floors instead of converging. Clone ONE.

  python scripts/bc_clone_demo.py \
      --demo checkpoints/harvested_seeds/demos_seam_sol_020_ph4.npz \
      --init checkpoints/mario_2_1_runB/back_2_1_iter230_clears.pt \
      --out checkpoints/mario_2_1_seamweld/seam_pilot_sol020.pt

Route the result with noop_pad: 1 (the demo was minted under the
load -> one-noop -> act convention). Pick a demo with zero obs->action
conflicts (duplicate observation rows with different actions make 100%
accuracy impossible for a memoryless net — the tool checks and refuses).
Robustness remains the weld trainer's job; the clone is the pilot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="DemoBank .npz (obs_0/act_0)")
    ap.add_argument("--init", required=True,
                    help="Checkpoint whose net arch + weights initialize "
                         "the clone (good features converge far faster "
                         "than a cold net).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-actions", type=int, default=None,
                    help="Default: infer from the demo's max action + 1, "
                         "floor 7.")
    ap.add_argument("--epochs", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    d = np.load(args.demo)
    obs = torch.from_numpy(d["obs_0"].astype(np.float32))
    act = torch.from_numpy(d["act_0"].astype(np.int64))
    n_act = args.num_actions or max(7, int(act.max()) + 1)
    print(f"pairs {obs.shape[0]}, feature_dim {obs.shape[1]}, "
          f"num_actions {n_act}")

    # Refuse structurally unlearnable demos up front: duplicate obs rows
    # with different actions cap accuracy below 100% for a memoryless
    # net and the divergence point is exactly where the replay breaks.
    seen: dict = {}
    for i in range(obs.shape[0]):
        k = d["obs_0"][i].tobytes()
        if k in seen and seen[k] != int(act[i]):
            print(f"CONFLICT at pair {i}: identical obs, different "
                  f"action — pick another solution")
            return 1
        seen[k] = int(act[i])

    state = torch.load(args.init, map_location="cpu", weights_only=False)
    net, is_rec = build_tile_policy_from_checkpoint(
        state, num_actions=n_act, feature_dim=obs.shape[1],
    )
    net.load_state_dict(state["net_state_dict"], strict=False)
    assert not is_rec, "recurrent init nets are not supported here"
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5,
    )

    net.train()
    for epoch in range(args.epochs):
        logits, _ = net.forward_ac(obs)
        loss = F.cross_entropy(logits, act)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 500 == 0:
            acc = (logits.argmax(1) == act).float().mean().item()
            print(f"epoch {epoch}: loss {loss.item():.5f} acc {acc:.4f}")
            if acc == 1.0 and loss.item() < 1e-3:
                break

    net.eval()
    with torch.no_grad():
        logits, _ = net.forward_ac(obs)
        acc = (logits.argmax(1) == act).float().mean().item()
    print(f"final argmax accuracy: {acc:.4f}")
    if acc < 1.0:
        print("clone did not fully absorb the trajectory — NOT saved")
        return 1
    torch.save({"net_state_dict": net.state_dict(), "iter": 0}, args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
