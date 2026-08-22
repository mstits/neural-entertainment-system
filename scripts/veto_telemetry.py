"""Measure what the hazard veto actually did, from outside the training run.

The mask's own counters live on the HazardMask object and die with the
process. They could not be logged mid-experiment: the control arm was
already executing trainer.py and the masked arm launched from the same
file, so editing the training path would have given the two arms
different code and voided the comparison.

So the veto is reconstructed instead. The hazard model is frozen and
deterministic, so replaying a policy's own visited states through it
yields exactly the vetoes that policy faced.

This is the measurement that separates two very different failures:

  INERT   — few actions vetoed, or the escape hatch firing nearly
            everywhere. A null result then says almost nothing about the
            veto, because the veto barely happened.
  HARMFUL — actions vetoed often, and specifically the ones that make
            progress. In a platformer the correct action is frequently
            the dangerous one: jumping a pit is high-hazard by
            construction, because the model learned hazard from what
            kills you. A veto that fires there and leaves standing still
            legal will reliably choose the action that never dies and
            never advances.

`--compare-progress` is what tells them apart: it reports the veto rate
separately for actions that INCREASE progress and those that do not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def summarize(vetoed: list, fully: list, prog_flags: list) -> dict:
    """PURE: turn per-step veto records into an interpretable verdict."""
    steps = len(fully)
    n_act = len(vetoed[0]) if vetoed else 0
    total = steps * n_act
    v_total = sum(sum(row) for row in vetoed)
    n_fully = sum(1 for f in fully if f)
    out = {
        "steps": steps,
        "veto_fraction": round(v_total / max(1, total), 4),
        "fully_vetoed_fraction": round(n_fully / max(1, steps), 4),
        "per_action_veto": [sum(r[i] for r in vetoed) for i in range(n_act)],
    }
    if prog_flags:
        pv = pn = pvc = pnc = 0
        for row, prog in zip(vetoed, prog_flags):
            for i, v in enumerate(row):
                if prog[i]:
                    pvc += 1
                    pv += int(v)
                else:
                    pnc += 1
                    pn += int(v)
        out["veto_rate_progressing_actions"] = round(pv / max(1, pvc), 4)
        out["veto_rate_other_actions"] = round(pn / max(1, pnc), 4)
        out["bias_against_progress"] = round(
            out["veto_rate_progressing_actions"]
            - out["veto_rate_other_actions"], 4)
    fv = out["fully_vetoed_fraction"]
    vf = out["veto_fraction"]
    if vf < 0.02:
        out["reading"] = ("INERT — almost nothing was vetoed; a null "
                          "result says little about the veto")
    elif fv > 0.5:
        out["reading"] = (f"MOSTLY INERT — the escape hatch dropped the "
                          f"mask in {fv:.0%} of states, so the masked arm "
                          f"approaches a second control")
    elif out.get("bias_against_progress", 0) > 0.05:
        out["reading"] = ("ACTIVE AND BIASED AGAINST PROGRESS — the veto "
                          "fires more often on actions that advance than "
                          "on those that do not, which is the mechanism "
                          "that makes standing still the safest policy")
    else:
        out["reading"] = "ACTIVE and not obviously biased against progress"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="configs/mario_1_2_phase3_masked.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--compare-progress", action="store_true", default=True)
    ap.add_argument("--out", default="runs/phase3/veto_telemetry.json")
    args = ap.parse_args(argv)

    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, torch, yaml, nes_core
    from src.training.hazard_mask import HazardMask
    from src.training.tile_policy import build_tile_policy_from_checkpoint

    prof = yaml.safe_load((REPO / args.profile).read_text())
    hz = (prof.get("reinforce", {}) or {}).get("hazard_mask") or {}
    mask = HazardMask.from_checkpoint(
        REPO / hz["checkpoint"], threshold=float(hz.get("threshold", 0.9)),
        enabled=True)
    net = build_tile_policy_from_checkpoint(str(REPO / args.checkpoint))
    net.eval()

    from scripts.eval_game import make_tile_pool  # reuse the eval stack
    pool, obs_fn, prog_fn = make_tile_pool(prof, n=args.episodes)

    vetoed, fully, prog_flags = [], [], []
    obs = obs_fn()
    for _ in range(args.steps):
        t = torch.tensor(np.asarray(obs), dtype=torch.float32)
        with torch.no_grad():
            logits, _ = net.forward_ac(t)
            risky = mask.death_probs(t) > mask.threshold
        allbad = risky.all(dim=-1)
        eff = (risky & ~allbad.unsqueeze(-1)).numpy()
        vetoed.extend(eff.tolist())
        fully.extend(allbad.numpy().tolist())
        if args.compare_progress:
            prog_flags.extend(prog_fn())
        obs = obs_fn(logits.masked_fill(
            torch.tensor(eff), float("-inf")).argmax(-1).numpy())

    v = summarize(vetoed, fully, prog_flags if args.compare_progress else [])
    print(json.dumps(v, indent=2))
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
