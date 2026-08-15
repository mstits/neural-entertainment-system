"""Offline IQ-Learn (soft-Bellman) + DQfD large-margin distillation.

Trains a TilePolicyNetwork whose ACTION HEAD is treated as Q(s,a) on the
recovery-transition set built by gen_iq_transitions.py. Because eval_game's
greedy mode takes argmax over the same action-head output, an argmax over Q is
identical to the argmax over the logits eval already uses — so a Q-net trained
here drops straight into the honest gate with no eval change.

Objective (Garg et al. 2021 IQ-Learn soft-Bellman χ² form + Hester et al. 2018
DQfD large-margin term), per batch, over expert transitions:

    q       = Q(s);              q_sel = q[a]
    v_next  = α·logsumexp(Q(s')/α)          (no grad)
    target  = (1-done)·γ·v_next
    bellman = q_sel - target
    iq_loss = -mean(is_expert · bellman)                     # push expert Q up
    q_reg   = (1/(4α))·mean(bellman²)                        # χ² soft-Bellman reg
    margin  = mean(is_expert·(max_a'(Q + margin·[a'≠a]) - q_sel))  # DQfD eq.4
    total   = iq_loss + q_reg + margin

Q is clamped to [-20,20] before the log-sum-exp so α=0.1 (→ ×10 temperature)
cannot destabilize. Saved in the same checkpoint format as bc_distill.py.
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

from src.models.tile_policy import TilePolicyNetwork  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--transitions", required=True,
                    help=".npz with state,action,next_state,done,is_expert.")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="Checkpoint dir to write.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--margin-coef", type=float, default=1.0)
    ap.add_argument("--q-clamp", type=float, default=20.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--trunk-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    profile = yaml.safe_load(Path(args.profile).read_text())
    num_actions = len(profile["action_space"])

    d = np.load(args.transitions)
    # obs are small signed ints — do NOT divide by 255.
    S = torch.from_numpy(d["state"].astype(np.float32))
    NS = torch.from_numpy(d["next_state"].astype(np.float32))
    A = torch.from_numpy(d["action"].astype(np.int64))
    DONE = torch.from_numpy(d["done"].astype(np.float32))
    EXP = torch.from_numpy(d["is_expert"].astype(np.float32))
    feature_dim = S.shape[1]
    n = S.shape[0]
    print(f"[iq] {n} transitions, feature_dim={feature_dim}, "
          f"num_actions={num_actions}, done={int(DONE.sum())}, "
          f"expert={int(EXP.sum())}", flush=True)
    print(f"[iq] action histogram: "
          f"{torch.bincount(A, minlength=num_actions).tolist()}", flush=True)

    net = TilePolicyNetwork(num_actions=num_actions, feature_dim=feature_dim,
                            hidden_dim=args.hidden_dim, trunk_dim=args.trunk_dim)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    alpha, gamma, margin = args.alpha, args.gamma, args.margin
    clamp = args.q_clamp

    def q_of(x):
        # Action head as Q-values (critic head unused for control).
        logits, _ = net.forward_ac(x)
        return logits

    net.train()
    last_acc = 0.0
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot_loss = tot_iq = tot_reg = tot_mar = tot_val = 0.0
        tot_correct = 0
        unstable = False
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            s, ns, a = S[idx], NS[idx], A[idx]
            done, exp = DONE[idx], EXP[idx]
            aidx = a.unsqueeze(1)

            q = q_of(s).clamp(-clamp, clamp)
            q_sel = q.gather(1, aidx).squeeze(1)
            # Current-state soft value (spec's v_curr) — GRADED, it is the
            # IQ-Learn value baseline E_ρ[V(s) - γV(s')] that bounds Q from
            # growing unboundedly. Omitting it (as the bare 3-term form does)
            # lets Q run to the clamp uniformly and freezes the net — the
            # collapse observed on the first pass. Garg 2021 Eq.10 "value" form.
            v_curr = alpha * torch.logsumexp(q / alpha, dim=1)
            with torch.no_grad():
                q_next = q_of(ns).clamp(-clamp, clamp)
                v_next = alpha * torch.logsumexp(q_next / alpha, dim=1)
                target_v = (1.0 - done) * gamma * v_next
            bellman = q_sel - target_v
            iq_loss = -(exp * bellman).mean()
            value_loss = (v_curr - target_v).mean()
            q_reg = (1.0 / (4.0 * alpha)) * (bellman ** 2).mean()
            margin_mtx = torch.full_like(q, margin)
            margin_mtx.scatter_(1, aidx, 0.0)
            margin_loss = (exp * ((q + margin_mtx).max(dim=1).values - q_sel)).mean()
            total = iq_loss + value_loss + q_reg + args.margin_coef * margin_loss

            if not torch.isfinite(total):
                unstable = True
                break
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()

            bs = len(idx)
            tot_loss += float(total.detach()) * bs
            tot_iq += float(iq_loss.detach()) * bs
            tot_reg += float(q_reg.detach()) * bs
            tot_mar += float(margin_loss.detach()) * bs
            tot_val += float(value_loss.detach()) * bs
            tot_correct += int((q.argmax(1) == a).sum())

        if unstable:
            print(f"[iq] epoch {ep}: NON-FINITE loss — unstable; aborting. "
                  f"Lower alpha or tighten --q-clamp.", flush=True)
            break
        last_acc = tot_correct / n
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[iq] epoch {ep:3d}  loss={tot_loss/n:.4f} "
                  f"iq={tot_iq/n:.4f} val={tot_val/n:.4f} reg={tot_reg/n:.4f} "
                  f"mar={tot_mar/n:.4f} argmax_acc={last_acc:.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "vanilla_ppo_iter_00000.pt"
    torch.save({"net_state_dict": {k: v.detach().cpu()
                                   for k, v in net.state_dict().items()},
                "iter": 0, "provenance": "iq_distill"}, str(ckpt))
    print(f"[iq] saved {ckpt}  final_argmax_acc={last_acc:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
