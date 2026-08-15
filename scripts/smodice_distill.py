"""SMODICE offline distillation (Ma et al. 2022) — discrete-action form.

Three stages over the mixed offline transition set built by
scripts/smodice_data.py plus the verified expert solution demos:

  1. State discriminator c(s): expert-visited states (demo npz obs) vs the
     mixed offline states. R(s) = log(c(s)/(1-c(s))) is exactly the
     discriminator logit, scaled by --alpha.
  2. Value network V minimizing the Fenchel dual of the f-divergence
     occupancy-matching objective:
         (1-gamma)*E_init[V(s0)] + E_D[f*(R(s) + gamma*V(s') - V(s))]
     with the standard SMODICE implementation choices — chi-square
     f(x) = 0.5*(x-1)^2 so f*(y) = 0.5*relu(y+1)^2 - 0.5 (default), or KL
     via the log-mean-exp dual. E_init is over window-start states.
     done=1 transitions ground V(s')=0; truncated=1 transitions bootstrap
     (Pardo 2018) and are never treated as terminal.
  3. Policy extraction: BC weighted by the occupancy-ratio weights
     w = f*'(e_v) (chi: relu(e_v+1); KL: N*softmax(e_v)) into the standard
     TilePolicyNetwork tile head, saved in the trainer checkpoint layout
     (net_state_dict in vanilla_ppo_iter_00000.pt) that eval_game.py loads
     shape-inferred, so the honest gate is
       eval_game.py --profile <p> --checkpoint <out>/vanilla_ppo_iter_00000.pt \
           --sequential --level-clear [--sticky-prob 0.25 --start-jitter 16]

Ends by printing the pre-registered gate line: train argmax accuracy vs the
0.829 majority-classifier data ceiling measured on the diverse 1-2 funnel.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.tile_policy import TilePolicyNetwork  # noqa: E402


class StateMLP(nn.Module):
    """Scalar head over a state vector — discriminator logit or V(s)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def terminal_mask(done, truncated):
    """The done mask the Bellman residual may ground on.

    truncated rows bootstrap V(s') (they are timeouts, not terminals), so
    they must arrive with done=0 — a row flagged both ways means the data
    generator conflated the two and grounding would silently corrupt V.
    """
    done = np.asarray(done)
    truncated = np.asarray(truncated)
    both = (done != 0) & (truncated != 0)
    if both.any():
        raise ValueError(
            f"{int(both.sum())} rows are both done and truncated — timeouts "
            "must bootstrap (done=0, truncated=1), terminals must ground "
            "(done=1, truncated=0)")
    return done


def bellman_residual(reward, v_s, v_next, done, gamma):
    """e_v = R(s) + gamma*(1-done)*V(s') - V(s); done=1 grounds V(s')=0."""
    return reward + gamma * (1.0 - done) * v_next - v_s


def initial_state_mask(window_id):
    """True on the first row of each window (the window-start states s0)."""
    wid = np.asarray(window_id)
    mask = np.ones(len(wid), dtype=bool)
    mask[1:] = wid[1:] != wid[:-1]
    seen = {}
    for i in np.flatnonzero(mask):
        w = int(wid[i])
        if w in seen:
            mask[i] = False
        else:
            seen[w] = i
    return mask


def train_discriminator(expert_states, offline_states, hidden_dim=64,
                        epochs=100, lr=3e-4, batch=256, weight_decay=0.0):
    """Stage 1: c(s) separating expert-visited states from the offline mix.

    weight_decay regularizes the logit scale — the reference SMODICE
    implementation limits discriminator capacity/epochs so R(s) stays in a
    range the chi dual can absorb instead of saturating on disjoint support.
    """
    disc = StateMLP(offline_states.shape[1], hidden_dim)
    opt = torch.optim.Adam(disc.parameters(), lr=lr,
                           weight_decay=weight_decay)
    n = offline_states.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            eidx = torch.randint(0, expert_states.shape[0], (len(idx),))
            le = disc(expert_states[eidx])
            lo = disc(offline_states[idx])
            loss = (F.binary_cross_entropy_with_logits(
                        le, torch.ones_like(le))
                    + F.binary_cross_entropy_with_logits(
                        lo, torch.zeros_like(lo)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return disc


def clip_reward(r, c):
    """Symmetric clip of R(s) to [-c, c]; c=None is the identity.

    On near-disjoint expert/offline support the discriminator logit blows
    up (first pass: min=-78.85) and the Bellman residual it feeds drives
    the chi dual to zero-weight most of the data; clipping bounds the
    log-ratio like the reference implementation does.
    """
    if c is None:
        return r
    if c <= 0:
        raise ValueError(f"r-clip must be positive, got {c}")
    return r.clamp(-c, c)


def _f_star_chi(e_v):
    return 0.5 * F.relu(e_v + 1.0) ** 2 - 0.5


def train_value(init_states, states, next_states, rewards, done, gamma,
                f="chi", hidden_dim=64, epochs=100, lr=3e-4, batch=256):
    """Stage 2: V minimizing the f-divergence Fenchel dual.

    Returns (vnet, per-epoch mean loss history).
    """
    vnet = StateMLP(states.shape[1], hidden_dim)
    opt = torch.optim.Adam(vnet.parameters(), lr=lr)
    n = states.shape[0]
    history = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            iidx = torch.randint(0, init_states.shape[0], (len(idx),))
            v0 = vnet(init_states[iidx])
            e_v = bellman_residual(rewards[idx], vnet(states[idx]),
                                   vnet(next_states[idx]), done[idx], gamma)
            if f == "kl":
                dual = torch.logsumexp(e_v, dim=0) - math.log(len(idx))
            else:
                dual = _f_star_chi(e_v).mean()
            loss = (1.0 - gamma) * v0.mean() + dual
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        history.append(tot / n)
    return vnet, history


def smodice_weights(vnet, states, next_states, rewards, done, gamma, f="chi"):
    """Occupancy-ratio weights w = f*'(e_v) for the weighted-BC extraction."""
    with torch.no_grad():
        e_v = bellman_residual(rewards, vnet(states), vnet(next_states),
                               done, gamma)
        if f == "kl":
            return torch.softmax(e_v, dim=0) * len(e_v)
        return F.relu(e_v + 1.0)


def extract_policy(states, actions, weights, num_actions, hidden_dim=64,
                   trunk_dim=32, epochs=60, lr=1e-3, batch=256):
    """Stage 3: weighted BC into the standard tile head.

    Returns (net, final train argmax accuracy).
    """
    net = TilePolicyNetwork(num_actions=num_actions,
                            feature_dim=states.shape[1],
                            hidden_dim=hidden_dim, trunk_dim=trunk_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    w = weights / weights.mean().clamp_min(1e-8)
    n = states.shape[0]
    acc = 0.0
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot_loss = tot_correct = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits, _ = net.forward_ac(states[idx])
            ce = F.cross_entropy(logits, actions[idx], reduction="none")
            loss = (w[idx] * ce).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_correct += int((logits.argmax(-1) == actions[idx]).sum())
        acc = tot_correct / n
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[smodice] extract epoch {ep:3d}  loss={tot_loss/n:.4f}  "
                  f"argmax_acc={acc:.3f}", flush=True)
    return net, acc


def _apply_chunked(net, x, chunk=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            outs.append(net(x[i:i + chunk]))
    return torch.cat(outs)


def load_expert_states(paths):
    """Expert-visited states from demo npz banks (obs_i keys, int8 rows)."""
    O = []
    for p in paths:
        d = np.load(str(p))
        obs_keys = sorted(k for k in d.files if k.startswith("obs_"))
        if not obs_keys:
            raise ValueError(
                f"{p}: no obs_-prefixed keys — got {sorted(d.files)}; "
                "expert demos use the demo-bank schema, not the transition "
                "schema")
        for k in obs_keys:
            O.append(d[k].astype(np.float32))
    return torch.from_numpy(np.concatenate(O, axis=0))


def load_expert_transitions(paths):
    """(s, a, s') rows from consecutive steps WITHIN each demo window.

    SMODICE derives its occupancy ratio assuming the offline distribution
    d^O covers the expert's (the union assumption); our offline set is
    drift-recovery data that excludes the on-path expert transitions, so
    this rebuilds them from the demo banks for --union-expert. Window i of
    each bank contributes T-1 rows (obs_i[t], act_i[t], obs_i[t+1]); every
    row is done=0 (nothing terminal happened — the bank just ends) and the
    last row of each window is truncated=1 so V(s') bootstraps. Returns a
    dict of row-aligned arrays plus per-window lengths so the caller can
    assign fresh window ids.
    """
    S, A, NS, T, lengths = [], [], [], [], []
    for p in paths:
        d = np.load(str(p))
        obs_keys = sorted(k for k in d.files if k.startswith("obs_"))
        if not obs_keys:
            raise ValueError(
                f"{p}: no obs_-prefixed keys — got {sorted(d.files)}")
        for k in obs_keys:
            obs = d[k]
            act = d["act_" + k[len("obs_"):]]
            if len(obs) != len(act):
                raise ValueError(
                    f"{p}:{k}: obs/act length mismatch "
                    f"({len(obs)} vs {len(act)})")
            if len(obs) < 2:
                continue
            S.append(obs[:-1])
            A.append(act[:-1])
            NS.append(obs[1:])
            tr = np.zeros(len(obs) - 1, dtype=np.int8)
            tr[-1] = 1
            T.append(tr)
            lengths.append(len(obs) - 1)
    if not S:
        raise ValueError("no expert transitions derivable from demo banks")
    n = sum(lengths)
    return {
        "state": np.concatenate(S, axis=0),
        "action": np.concatenate(A, axis=0).astype(np.int64),
        "next_state": np.concatenate(NS, axis=0),
        "done": np.zeros(n, dtype=np.int8),
        "truncated": np.concatenate(T, axis=0),
        "window_lengths": lengths,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True,
                    help=".npz from smodice_data.py (state/action/next_state/"
                         "done/truncated/window_id).")
    ap.add_argument("--expert-demos", nargs="+", required=True,
                    help="Verified solution demo .npz paths (obs_i/act_i).")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True, help="Checkpoint dir to write.")
    ap.add_argument("--hidden", type=int, default=64,
                    help="Hidden width for disc/value MLPs and the tile head.")
    ap.add_argument("--trunk-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60,
                    help="Epochs per stage (disc, value, extraction).")
    ap.add_argument("--disc-epochs", type=int, default=None,
                    help="Stage-1 epochs override (default: --epochs). The "
                         "reference implementation trains the discriminator "
                         "briefly to keep R(s) from saturating.")
    ap.add_argument("--disc-weight-decay", type=float, default=0.0,
                    help="Adam weight decay on the discriminator (logit-"
                         "scale regularizer).")
    ap.add_argument("--r-clip", type=float, default=None,
                    help="Clip R(s) to [-C, C] after --alpha scaling.")
    ap.add_argument("--union-expert", action="store_true",
                    help="Append expert (s,a,s') transitions derived from "
                         "the demo banks into the offline set for stages "
                         "2-3, restoring the SMODICE union assumption "
                         "d^O ⊇ d^E.")
    ap.add_argument("--unweighted", action="store_true",
                    help="Skip stages 1-2 and extract with weights=1 "
                         "(plain-BC control for the DICE weighting).")
    ap.add_argument("--extract-epochs", type=int, default=None,
                    help="Stage-3 epochs override (default: --epochs).")
    ap.add_argument("--extract-lr", type=float, default=None,
                    help="Stage-3 learning rate override (default: --lr).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Scale on R(s) = alpha * log(c/(1-c)).")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--f", choices=("chi", "kl"), default="chi")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling", type=float, default=0.829,
                    help="Pre-registered majority-classifier data ceiling.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    profile = yaml.safe_load(Path(args.profile).read_text())
    num_actions = len(profile["action_space"])

    d = np.load(args.data)
    S_np = d["state"].astype(np.float32)
    NS_np = d["next_state"].astype(np.float32)
    A_np = d["action"].astype(np.int64)
    done_raw = np.asarray(d["done"])
    trunc_raw = np.asarray(d["truncated"])
    wid_np = np.asarray(d["window_id"])
    n_offline = S_np.shape[0]

    if args.union_expert:
        ex = load_expert_transitions(args.expert_demos)
        if ex["state"].shape[1] != S_np.shape[1]:
            raise SystemExit(
                f"[smodice] expert transition width {ex['state'].shape[1]} "
                f"!= offline state width {S_np.shape[1]}")
        wid0 = int(wid_np.max()) + 1 if len(wid_np) else 0
        ex_wid = np.concatenate(
            [np.full(L, wid0 + i, dtype=np.int64)
             for i, L in enumerate(ex["window_lengths"])])
        S_np = np.concatenate([S_np, ex["state"].astype(np.float32)])
        NS_np = np.concatenate([NS_np, ex["next_state"].astype(np.float32)])
        A_np = np.concatenate([A_np, ex["action"]])
        done_raw = np.concatenate([done_raw, ex["done"]])
        trunc_raw = np.concatenate([trunc_raw, ex["truncated"]])
        wid_np = np.concatenate([wid_np, ex_wid])
        print(f"[smodice] union-expert: +{len(ex['action'])} transitions "
              f"from {len(ex['window_lengths'])} demo windows", flush=True)

    S = torch.from_numpy(S_np)
    NS = torch.from_numpy(NS_np)
    A = torch.from_numpy(A_np)
    done_np = terminal_mask(done_raw, trunc_raw)
    DONE = torch.from_numpy(np.asarray(done_np).astype(np.float32))
    init_mask = initial_state_mask(wid_np)
    S0 = S[torch.from_numpy(init_mask)]
    n = S.shape[0]
    print(f"[smodice] {n} transitions, feature_dim={S.shape[1]}, "
          f"num_actions={num_actions}, done={int(DONE.sum())}, "
          f"truncated={int(trunc_raw.sum())}, windows={S0.shape[0]}",
          flush=True)
    print(f"[smodice] action histogram: "
          f"{torch.bincount(A, minlength=num_actions).tolist()}", flush=True)

    disc = vnet = None
    if args.unweighted:
        print("[smodice] unweighted control: skipping stages 1-2, "
              "weights=1", flush=True)
        W = torch.ones(n)
    else:
        expert = load_expert_states(args.expert_demos)
        if expert.shape[1] != S.shape[1]:
            raise SystemExit(
                f"[smodice] expert obs width {expert.shape[1]} != offline "
                f"state width {S.shape[1]}")
        disc_epochs = (args.disc_epochs if args.disc_epochs is not None
                       else args.epochs)
        # The discriminator's negative class stays the ORIGINAL offline
        # rows; --union-expert only widens the stage-2/3 expectation E_D.
        S_disc = S[:n_offline]
        print(f"[smodice] stage 1: discriminator on {expert.shape[0]} "
              f"expert states vs {S_disc.shape[0]} offline states "
              f"({disc_epochs} epochs, wd={args.disc_weight_decay})",
              flush=True)
        disc = train_discriminator(expert, S_disc, hidden_dim=args.hidden,
                                   epochs=disc_epochs, lr=args.lr,
                                   batch=args.batch,
                                   weight_decay=args.disc_weight_decay)
        R = clip_reward(args.alpha * _apply_chunked(disc, S), args.r_clip)
        print(f"[smodice] R(s): mean={float(R.mean()):.3f} "
              f"min={float(R.min()):.3f} max={float(R.max()):.3f}"
              + (f" (clipped to ±{args.r_clip:g})"
                 if args.r_clip is not None else ""), flush=True)

        print(f"[smodice] stage 2: value ({args.f} dual), "
              f"gamma={args.gamma}", flush=True)
        vnet, hist = train_value(S0, S, NS, R, DONE, args.gamma, f=args.f,
                                 hidden_dim=args.hidden, epochs=args.epochs,
                                 lr=args.lr, batch=args.batch)
        print(f"[smodice] value loss {hist[0]:.4f} -> {hist[-1]:.4f}",
              flush=True)

        W = smodice_weights(vnet, S, NS, R, DONE, args.gamma, f=args.f)
        print(f"[smodice] weights: mean={float(W.mean()):.3f} "
              f"zero_frac={float((W <= 0).float().mean()):.3f}", flush=True)

    extract_epochs = (args.extract_epochs if args.extract_epochs is not None
                      else args.epochs)
    extract_lr = args.extract_lr if args.extract_lr is not None else args.lr
    print("[smodice] stage 3: weighted-BC extraction", flush=True)
    net, acc = extract_policy(S, A, W, num_actions=num_actions,
                              hidden_dim=args.hidden,
                              trunk_dim=args.trunk_dim,
                              epochs=extract_epochs,
                              lr=extract_lr, batch=args.batch)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "vanilla_ppo_iter_00000.pt"
    torch.save({"net_state_dict": {k: v.detach().cpu()
                                   for k, v in net.state_dict().items()},
                "iter": 0, "provenance": "smodice_distill"}, str(ckpt))
    if disc is not None:
        torch.save({"disc_state_dict": disc.state_dict(),
                    "value_state_dict": vnet.state_dict(),
                    "f": args.f, "alpha": args.alpha, "gamma": args.gamma,
                    "hidden": args.hidden}, str(out / "smodice_stage12.pt"))
    print(f"[smodice] saved {ckpt}", flush=True)
    print(f"[smodice] GATE: argmax_acc={acc:.3f} vs ceiling "
          f"{args.ceiling:.3f} (gap={acc - args.ceiling:+.3f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
