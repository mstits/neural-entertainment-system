#!/usr/bin/env python3
"""Premise falsifier for the recurrent line (v26 Q4, run before any
salvage training): can a probe PREDICT whether the last action stuck,
from exactly what the policy sees?

v25's mechanism claim: sticky p=0.25 makes the env a POMDP because the
executed action is unobservable; a recurrent state carrying intent +
observed motion detects sticks and enables closed-loop correction. If
stick-detection is impossible even as a SUPERVISED task on the
policy's own observations, that premise is dead and no amount of
recurrent RL training will reach it.

Design: roll the BANKED feedforward control policy (honest 0.76
artifact) from the 1-1 entrance under sticky 0.25 applied in-script
(so the roll is observable). Per step t record:
  x_t = [stacked tile obs AFTER the transition (712), onehot(chosen_t)]
  y_t = 1 if executed_t != chosen_t (the sticky roll replaced it)
Train two probes on identical data, same budget:
  mlp:  logistic head on x_t                      (stateless baseline —
        the 4-frame stack already carries motion history)
  gru:  GRUCell(64) over x_t, episode-reset hidden, logistic head
Report accuracy + AUC vs the 0.25 base rate.

Readout: BOTH ~chance -> stick info absent from obs, v25 premise dead.
gru >> mlp -> recurrence adds real stick-detectability (premise lives).
both high -> info present even statelessly; the policy class argument
weakens (a feedforward net COULD have learned it).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/_preserved/backward_1_1_seed3_iter140.pt")
    ap.add_argument("--profile", default="configs/mario_1_1_backward.yaml")
    ap.add_argument("--start-state",
                    default="runs/live_show/smb_4_4_micro/entrance_start.state")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="runs/gru_ab/stick_probe.json")
    args = ap.parse_args()

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, torch, yaml, nes_core
    import torch.nn as nn
    import torch.nn.functional as F
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder)
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    prof = yaml.safe_load((REPO / args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    n_act = len(bm)
    extractor, feat_dim, stacked_dim = resolve_encoder(prof)
    net, _ = build_tile_policy_from_checkpoint(
        str(REPO / args.checkpoint), num_actions=n_act,
        feature_dim=stacked_dim)
    net.eval()

    pool = nes_core.Pool(rom_path=str(REPO / prof["rom_path"]),
                         num_workers=1,
                         frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    blob = (REPO / args.start_state).read_bytes()

    # ---- collect ----
    episodes = []            # list of (X [T,712+n_act], y [T]) per episode
    for ep in range(args.episodes):
        pool.load_worker_state(0, blob)
        stk = TileFeatureStacker(4, feat_dim)
        step = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
        obs = stk.reset(extractor.extract(step[2]))
        prev_exec = 0
        X, Y = [], []
        for t in range(args.max_steps):
            with torch.no_grad():
                logits = net.forward_ac(
                    torch.from_numpy(obs[None]).float())[0]
            chosen = int(logits.argmax(dim=-1).item())
            stuck = t > 0 and rng.random() < args.sticky
            executed = prev_exec if stuck else chosen
            step = pool.step_all(
                np.array([bm[executed]], dtype=np.uint8))[0]
            obs = stk.push(extractor.extract(step[2]))
            row = np.zeros(len(obs) + n_act, dtype=np.float32)
            row[:len(obs)] = obs
            row[len(obs) + chosen] = 1.0
            X.append(row)
            Y.append(1.0 if executed != chosen else 0.0)
            prev_exec = executed
            if bool(step[3]):
                break
        episodes.append((np.stack(X), np.array(Y, dtype=np.float32)))
    n_steps = sum(len(y) for _, y in episodes)
    pos = sum(float(y.sum()) for _, y in episodes)
    print(f"collected {n_steps} steps over {len(episodes)} eps, "
          f"stuck rate {pos/n_steps:.3f}")

    # ---- split by episode ----
    idx = rng.permutation(len(episodes))
    cut = max(1, int(len(episodes) * 0.8))
    train_eps = [episodes[i] for i in idx[:cut]]
    test_eps = [episodes[i] for i in idx[cut:]]
    in_dim = episodes[0][0].shape[1]

    class MlpProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(in_dim, 64), nn.SiLU(),
                                   nn.Linear(64, 1))
        def forward(self, x):                       # [T, in]
            return self.f(x).squeeze(-1)

    class GruProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(in_dim, 64)
            self.cell = nn.GRUCell(64, 64)
            self.head = nn.Linear(64, 1)
        def forward(self, x):                       # [T, in], one episode
            h = torch.zeros(1, 64)
            outs = []
            for t in range(x.shape[0]):
                h = self.cell(F.silu(self.inp(x[t:t+1])), h)
                outs.append(self.head(h).squeeze())
            return torch.stack(outs)

    def run_probe(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(args.epochs):
            for X, y in train_eps:
                xt = torch.from_numpy(X)
                yt = torch.from_numpy(y)
                opt.zero_grad()
                loss = lossf(model(xt), yt)
                loss.backward()
                opt.step()
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for X, y in test_eps:
                s = torch.sigmoid(model(torch.from_numpy(X))).numpy()
                scores.append(np.atleast_1d(s))
                labels.append(y)
        s = np.concatenate(scores); l = np.concatenate(labels)
        acc = float(((s > 0.5) == (l > 0.5)).mean())
        # AUC via rank statistic
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        n1 = l.sum(); n0 = len(l) - n1
        auc = (float(ranks[l > 0.5].sum()) - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
        return {"test_acc": round(acc, 4), "test_auc": round(auc, 4),
                "base_rate": round(float(l.mean()), 4),
                "test_steps": int(len(l))}

    torch.manual_seed(args.seed)
    r_mlp = run_probe(MlpProbe())
    torch.manual_seed(args.seed)
    r_gru = run_probe(GruProbe())
    out = {"mlp": r_mlp, "gru": r_gru, "steps": n_steps,
           "episodes": len(episodes),
           "checkpoint": args.checkpoint, "sticky": args.sticky}
    print(json.dumps(out, indent=2))
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
