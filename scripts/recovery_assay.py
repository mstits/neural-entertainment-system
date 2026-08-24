#!/usr/bin/env python3
"""Recovery assay — the registered follow-up from the v26 adjudication.

Decides whether the sticky wall is TRAINABLE RECOVERY or MECHANICAL
FATE: at the moment a divergent stick lands (executed != chosen), is
the episode already lost, or does a recovering continuation exist that
a policy could in principle learn?

Phase 1 (collect): roll the banked control policy from the 1-1
entrance under in-script sticky p=0.25 (greedy, jitter via eval
convention is omitted — the stick is the variable under test).
At every divergent stick, snapshot the post-stick savestate. Tag each
episode's outcome (clear / death) with the SMB adapter's own
predicates. Keep the snapshots from DEATH episodes: the last divergent
stick before death is the prime suspect, earlier ones are controls.

Phase 2 (adjudicate): for each sampled death-episode snapshot, run
go_explore_solve from that state for a few minutes. Solver finds the
flag => recovery EXISTED at that point (the policy failed a learnable
response). Solver cannot => the stick was mechanically fatal.

Verdict over N samples: recovery_rate = recovered / sampled.
High (>=0.8): the wall is a response-learning gap -> post-stick states
become curriculum fuel (solver-as-teacher). Low (<=0.2): sticks are
fate; the honest per-level ceilings are real.

    .venv/bin/python scripts/recovery_assay.py collect
    .venv/bin/python scripts/recovery_assay.py adjudicate --sample 16
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs/recovery_assay"


def collect(args) -> int:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, torch, yaml, nes_core
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder)
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint

    rng = np.random.default_rng(args.seed)
    prof = yaml.safe_load((REPO / args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    extractor, feat_dim, stacked_dim = resolve_encoder(prof)
    net, _ = build_tile_policy_from_checkpoint(
        str(REPO / args.checkpoint), num_actions=len(bm),
        feature_dim=stacked_dim)
    net.eval()

    pool = nes_core.Pool(rom_path=str(REPO / prof["rom_path"]),
                         num_workers=1,
                         frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    blob = (REPO / args.start_state).read_bytes()
    (OUT / "states").mkdir(parents=True, exist_ok=True)

    manifest = []
    for ep in range(args.episodes):
        pool.load_worker_state(0, blob)
        stk = TileFeatureStacker(4, feat_dim)
        step = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
        # Machado no-op start jitter — the other half of the honest
        # protocol. Without it, exact-entrance greedy is one
        # deterministic line and dies at one argmax-tie spot regardless
        # of sticks (measured: 60/60 deaths at step ~82).
        for _ in range(int(rng.integers(0, args.jitter + 1))):
            step = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
        obs = stk.reset(extractor.extract(step[2]))
        start_wd = (int(step[2][0x075F]), int(step[2][0x0760]))
        start_lives = int(step[2][0x075A])
        prev_exec = 0
        sticks = []          # (t, state_path) this episode
        outcome = "timeout"
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
            ram = step[2]
            if executed != chosen:
                sp = OUT / "states" / f"ep{ep:03d}_t{t:04d}.state"
                sp.write_bytes(bytes(pool.save_worker_state(0)))
                sticks.append((t, str(sp.relative_to(REPO))))
            prev_exec = executed
            wd = (int(ram[0x075F]), int(ram[0x0760]))
            if wd > start_wd:
                outcome = "clear"
                break
            # Death = lives decrement or the B5 pit fall (both verified
            # zero-false-positive in the solver). $000E player-state is
            # NOT used: value 11 fires during ordinary alive play here
            # (measured — 6/6 false deaths at x~314 on episodes eval_game
            # scores as clears), whatever its meaning in other contexts.
            if (int(ram[0x075A]) < start_lives
                    or int(ram[0x00B5]) >= 2):
                outcome = "death"
                break
        manifest.append({"episode": ep, "outcome": outcome,
                         "steps": t + 1, "sticks": sticks})
        # Snapshots from cleared episodes are controls we don't need in
        # bulk — drop all but the last two to bound disk.
        if outcome == "clear":
            for _, sp in sticks[:-2]:
                (REPO / sp).unlink(missing_ok=True)
                manifest[-1]["sticks"] = sticks[-2:]
        print(f"ep {ep}: {outcome} at step {t+1}, "
              f"{len(sticks)} divergent sticks", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "episodes": args.episodes,
        "sticky": args.sticky, "runs": manifest}, indent=2) + "\n")
    deaths = sum(1 for m in manifest if m["outcome"] == "death")
    clears = sum(1 for m in manifest if m["outcome"] == "clear")
    print(f"collected: {clears} clears, {deaths} deaths "
          f"of {args.episodes}")
    return 0


def adjudicate(args) -> int:
    # Manifest now comes from eval_game --dump-stick-states (the
    # verified honest harness; the standalone collector reimplemented
    # the loop and diverged — 0/60 vs the harness's 0.767).
    man = json.loads((OUT / "manifest.json").read_text())
    suspects = []
    for ep, m in enumerate(man["records"]):
        if not m["cleared"] and m["sticks"]:
            t, sp = m["sticks"][-1]
            suspects.append({"episode": ep, "t": t,
                             "death_step": m["length"], "state": sp})
    suspects = suspects[: args.sample]
    print(f"adjudicating {len(suspects)} death-preceding sticks "
          f"({args.minutes} solver-min each)")
    results = []
    for i, s in enumerate(suspects):
        out_dir = OUT / f"solve_ep{s['episode']:03d}_t{s['t']:04d}"
        cmd = [str(REPO / ".venv/bin/python"),
               str(REPO / "scripts/go_explore_solve.py"),
               "--profile", args.profile,
               "--root-state", s["state"],
               "--out", str(out_dir.relative_to(REPO)),
               "--workers", str(args.workers),
               "--minutes", str(args.minutes),
               "--want-solutions", "1",
               "--seed", str(1000 + i)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=(args.minutes * 60 + 600))
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        recovered = '"solutions": 0' not in tail and '"solutions"' in tail
        results.append({**s, "recovered": recovered, "solver_tail": tail})
        print(f"[{i+1}/{len(suspects)}] ep{s['episode']} t{s['t']} "
              f"(died at {s['death_step']}): "
              f"{'RECOVERED' if recovered else 'no recovery found'}",
              flush=True)
    n = len(results)
    rec = sum(1 for r in results if r["recovered"])
    verdict = {"sampled": n, "recovered": rec,
               "recovery_rate": round(rec / n, 3) if n else None,
               "solver_minutes_each": args.minutes,
               "results": results}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: verdict[k] for k in
                      ("sampled", "recovered", "recovery_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "adjudicate"])
    ap.add_argument("--checkpoint",
                    default="checkpoints/_preserved/backward_1_1_seed3_iter140.pt")
    ap.add_argument("--profile", default="configs/mario_1_1_backward.yaml")
    ap.add_argument("--start-state",
                    default="runs/live_show/smb_4_4_micro/entrance_start.state")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--jitter", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sys.exit(collect(a) if a.mode == "collect" else adjudicate(a))
