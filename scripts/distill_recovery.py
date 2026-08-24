#!/usr/bin/env python3
"""Recovery distillation on 1-1 — the registered experiment.

Pre-registration: docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md
Fine-tunes the BANKED control (not from scratch) on short recovery
clips mined by scripts/mine_recovery_tapes.py, mixed 1:1 with
on-manifold anchor clips (the policy's own greedy play), at low LR
with a per-epoch honest mini-eval, a 0.70 drift-stop, and
preserve-on-peak. Scoring against the 0.80 gate is a separate, final
eval_game run on the preserved artifact.

  phase demos    build clip datasets (recovery via replay_to_demos on
                 each mined tape clipped to --clip steps; anchor by
                 rolling the control greedily from the entrance)
  phase train    fine-tune + per-epoch mini-eval + drift stop
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FUEL = REPO / "runs/recovery_distill/fuel"
OUT = REPO / "runs/recovery_distill"


def build_demos(args) -> int:
    import numpy as np
    tapes = sorted((FUEL / "tapes").glob("*.actions.npy"))
    print(f"{len(tapes)} tapes")
    (OUT / "demos").mkdir(parents=True, exist_ok=True)
    for t in tapes:
        acts = np.load(t)
        clip = t.with_suffix("").with_suffix("")  # strip .actions.npy
        clipped = OUT / "demos" / (t.stem + ".clipped.npy")
        np.save(clipped, acts[: args.clip])
        outp = OUT / "demos" / (t.stem + ".npz")
        cmd = [str(REPO / ".venv/bin/python"),
               str(REPO / "scripts/replay_to_demos.py"),
               "--start-state", str(t.with_suffix(".start.state")),
               "--actions", str(clipped),
               "--profile", args.profile,
               "--out", str(outp)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(f"  {outp.name}: {'ok' if outp.exists() else r.stderr[-200:]}")
    # ---- anchor demos: the control's own greedy play (self-clone) ----
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch, yaml, nes_core
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder)
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    prof = yaml.safe_load((REPO / args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    extractor, feat_dim, stacked_dim = resolve_encoder(prof)
    net, _ = build_tile_policy_from_checkpoint(
        str(REPO / args.checkpoint), num_actions=len(bm),
        feature_dim=stacked_dim)
    net.eval()
    pool = nes_core.Pool(rom_path=str(REPO / prof["rom_path"]),
                         num_workers=1, frame_skip=4,
                         start_state_path=prof["start_state_path"])
    pool.set_headless(True); pool.set_skip_preprocess(True)
    blob = (REPO / args.start_state).read_bytes()
    rng = np.random.default_rng(7)
    all_obs, all_act = [], []
    for ep in range(args.anchor_episodes):
        pool.reset_all(); pool.load_worker_state(0, blob)
        stk = TileFeatureStacker(4, feat_dim)
        step = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
        for _ in range(int(rng.integers(0, 17))):
            step = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
        obs = stk.reset(extractor.extract(step[2]))
        for t in range(args.anchor_steps):
            with torch.no_grad():
                logits = net.forward_ac(
                    torch.from_numpy(obs[None]).float())[0]
            a = int(logits.argmax(dim=-1).item())
            all_obs.append(obs.copy()); all_act.append(a)
            step = pool.step_all(np.array([bm[a]], dtype=np.uint8))[0]
            obs = stk.push(extractor.extract(step[2]))
            if bool(step[3]):
                break
    np.savez_compressed(OUT / "demos" / "anchor_selfplay.npz",
                        obs_0=np.stack(all_obs),
                        act_0=np.array(all_act, dtype=np.int64))
    print(f"anchor: {len(all_act)} pairs from {args.anchor_episodes} eps")
    return 0


def train(args) -> int:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, torch
    import torch.nn.functional as F
    import yaml
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder)
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    prof = yaml.safe_load((REPO / args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    _, _, stacked_dim = resolve_encoder(prof)
    net, _ = build_tile_policy_from_checkpoint(
        str(REPO / args.checkpoint), num_actions=len(bm),
        feature_dim=stacked_dim)

    # Variant A: frozen control for the KL anchor leash
    frozen, _ = build_tile_policy_from_checkpoint(
        str(REPO / args.checkpoint), num_actions=len(bm),
        feature_dim=stacked_dim)
    frozen.eval()
    for p_ in frozen.parameters():
        p_.requires_grad_(False)

    rec_obs, rec_act = [], []
    for f in sorted((OUT / "demos").glob("ep*.npz")):
        d = np.load(f)
        rec_obs.append(d["obs_0"]); rec_act.append(d["act_0"])
    a = np.load(OUT / "demos" / "anchor_selfplay.npz")
    rec_obs = np.concatenate(rec_obs); rec_act = np.concatenate(rec_act)
    anc_obs, anc_act = a["obs_0"], a["act_0"]
    print(f"recovery pairs {len(rec_act)}, anchor pairs {len(anc_act)}")
    # 1:1 mix per epoch: sample anchor subset the size of the recovery set
    dev = torch.device("cpu")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    (OUT / "ckpts").mkdir(parents=True, exist_ok=True)
    history, best_rate, best_path = [], -1.0, None
    for epoch in range(args.epochs):
        idx = rng.permutation(len(anc_act))[: len(rec_act)]
        obs = np.concatenate([rec_obs, anc_obs[idx]])
        act = np.concatenate([rec_act, anc_act[idx]])
        order = rng.permutation(len(act))
        net.train()
        n_rec = len(rec_act)
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            xb = torch.from_numpy(obs[b]).float().to(dev)
            logits, _ = net.forward_ac(xb)
            is_rec = torch.from_numpy(b < n_rec).to(dev)
            # recovery states: CE to the solver action
            ce = F.cross_entropy(
                logits, torch.from_numpy(act[b]).to(dev),
                reduction="none")
            # anchor states: KL leash to the FROZEN control (variant A)
            with torch.no_grad():
                ref_logits, _ = frozen.forward_ac(xb)
            kl = F.kl_div(F.log_softmax(logits, dim=-1),
                          F.log_softmax(ref_logits, dim=-1),
                          log_target=True, reduction="none").sum(-1)
            loss = torch.where(is_rec, ce, args.kl_beta * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        ck = OUT / "ckpts" / f"distill_epoch{epoch:02d}.pt"
        net.eval()
        torch.save({"net_state_dict": net.state_dict(),
                    "kind": "tile_mlp",
                    "num_actions": len(bm), "feature_dim": stacked_dim},
                   ck)
        # honest mini-eval (30 eps) — the drift stop and peak tracker
        cmd = [str(REPO / ".venv/bin/python"),
               str(REPO / "scripts/eval_game.py"),
               "--game", "mario", "--profile", args.profile,
               "--checkpoint", str(ck),
               "--start-state", args.start_state,
               "--episodes", "30", "--sticky-prob", "0.25",
               "--start-jitter", "16", "--eval-seed", "0",
               "--action-select", "greedy"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=3600)
        rate = None
        lines = r.stdout.splitlines()
        for i in reversed([j for j, l in enumerate(lines)
                           if l.startswith("{")]):
            try:
                rate = json.loads("\n".join(lines[i:])).get("clear_rate")
                break
            except json.JSONDecodeError:
                continue
        history.append({"epoch": epoch, "clear_rate": rate,
                        "loss": float(loss.item())})
        print(f"epoch {epoch}: mini-eval clear_rate={rate} "
              f"loss={loss.item():.4f}", flush=True)
        if rate is not None and rate > best_rate:
            best_rate, best_path = rate, str(ck)
        if rate is not None and rate < args.drift_stop:
            print(f"DRIFT STOP: {rate} < {args.drift_stop} — halting "
                  f"per registration", flush=True)
            break
    (OUT / "train_history.json").write_text(json.dumps(
        {"history": history, "best_rate": best_rate,
         "best_checkpoint": best_path}, indent=2) + "\n")
    print(f"best mini-eval {best_rate} at {best_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["demos", "train"])
    ap.add_argument("--checkpoint",
                    default="checkpoints/_preserved/backward_1_1_seed3_iter140.pt")
    ap.add_argument("--profile", default="configs/mario_1_1_backward.yaml")
    ap.add_argument("--start-state",
                    default="runs/live_show/smb_4_4_micro/entrance_start.state")
    ap.add_argument("--clip", type=int, default=60)
    ap.add_argument("--anchor-episodes", type=int, default=6)
    ap.add_argument("--anchor-steps", type=int, default=1400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--drift-stop", type=float, default=0.70)
    ap.add_argument("--kl-beta", type=float, default=1.0)
    a = ap.parse_args()
    import numpy as np  # noqa: F401  (demos phase uses np before import)
    sys.exit(build_demos(a) if a.phase == "demos" else train(a))
