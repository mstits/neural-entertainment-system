"""Positive controls for experiments: prove the mechanism is ALIVE.

Every voided experiment this week passed its config checks — prereg,
single-variable assertions, smoke tests — and still measured nothing:
2-1 trained on another level's ladder, Phase 3's actor was NaN-frozen,
then BOTH Phase-3 arms and the options arm ran with an actor frozen by a
1e12 sentinel. The configs said the experiment existed; nothing proved
the treatment was operating or that ANY learning was occurring.

This tool is the missing positive control. Given the artifacts of a
short pilot run (checkpoints before/after ~2 iters, the training log),
it asserts from EVIDENCE, not configuration:

  A. LEARNING IS ALIVE — some non-critic parameter moved. A run where
     only the critic moves is the frozen-actor signature.
  B. THE MECHANISM IS ALIVE — the treatment left runtime evidence: its
     armed line in the log AND, where declared, an activity signal.
  C. NO SENTINEL OUTLASTS THE BUDGET — actor_freeze_steps (and kin)
     must not exceed the planned run's final step count.

An experiment that fails preflight never receives its budget.

    .venv/bin/python scripts/experiment_preflight.py \\
        --before ckpt_iter_01120.pt --after ckpt_iter_01122.pt \\
        --log pilot.log --profile configs/arm.yaml --iters 200
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CRITIC_PREFIXES = ("critic.",)
# Mechanisms and the runtime evidence each must leave. `armed` is a
# regex that must appear in the log; `activity` (optional) must match
# with a captured number > 0 somewhere after arming.
MECHANISMS = {
    "commitment_options": {
        "profile_path": ("reinforce", "commitment_options", "enabled"),
        "armed": r"\[commitment\] ARMED",
    },
    "hazard_mask": {
        "profile_path": ("reinforce", "hazard_mask", "enabled"),
        "armed": r"\[hazard-mask\] ARMED",
    },
}


def param_deltas(sd_before: dict, sd_after: dict) -> dict:
    """Max |delta| split into actor-side and critic-side groups. PURE."""
    import torch
    actor_max = critic_max = 0.0
    missing = []
    for k, v in sd_before.items():
        if k not in sd_after:
            missing.append(k)
            continue
        d = float((sd_after[k].float() - v.float()).abs().max())
        if any(k.startswith(p) or f".{p}" in k for p in CRITIC_PREFIXES):
            critic_max = max(critic_max, d)
        else:
            actor_max = max(actor_max, d)
    return {"actor_max_delta": actor_max, "critic_max_delta": critic_max,
            "missing_keys": missing}


def assess_learning(deltas: dict) -> tuple[bool, str]:
    a, c = deltas["actor_max_delta"], deltas["critic_max_delta"]
    if a > 0:
        return True, f"actor moved (max|Δ|={a:.2e}); critic {c:.2e}"
    if c > 0:
        return False, (f"FROZEN-ACTOR SIGNATURE: critic moved ({c:.2e}) "
                       f"but no non-critic parameter changed — the exact "
                       f"pattern that voided Phase 3 and options v1")
    return False, "NOTHING moved — optimizer not stepping at all"


def profile_flag(profile: dict, path: tuple) -> bool:
    node = profile
    for k in path:
        if not isinstance(node, dict):
            return False
        node = node.get(k)
    return bool(node)


def assess_mechanisms(profile: dict, log_text: str) -> tuple[bool, list]:
    notes, ok = [], True
    for name, spec in MECHANISMS.items():
        if not profile_flag(profile, spec["profile_path"]):
            continue
        if re.search(spec["armed"], log_text):
            notes.append(f"{name}: armed evidence present")
        else:
            ok = False
            notes.append(f"{name}: ENABLED IN PROFILE BUT NO ARMED "
                         f"EVIDENCE IN LOG — the mechanism never ran "
                         f"(Phase-3 eval made this exact mistake)")
    if not notes:
        notes.append("no optional mechanisms declared (control arm)")
    return ok, notes


def assess_sentinels(profile: dict, planned_iters: int,
                     steps_per_iter: int, resume_iter: int) -> tuple[bool, str]:
    freeze = float((profile.get("reinforce", {}) or {})
                   .get("actor_freeze_steps", 0) or 0)
    end_steps = (resume_iter + planned_iters) * steps_per_iter
    if freeze > end_steps:
        return False, (f"actor_freeze_steps {freeze:.3g} outlasts the "
                       f"entire planned run ({end_steps:.3g} steps) — "
                       f"controller-managed sentinel in a standalone run")
    return True, f"freeze {freeze:.3g} < run end {end_steps:.3g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--iters", type=int, required=True,
                    help="the FULL planned budget, not the pilot's")
    ap.add_argument("--steps-per-iter", type=int, default=92160)
    ap.add_argument("--resume-iter", type=int, default=0)
    args = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(REPO))
    import torch
    import yaml

    sd_b = torch.load(REPO / args.before, map_location="cpu",
                      weights_only=False)["net_state_dict"]
    sd_a = torch.load(REPO / args.after, map_location="cpu",
                      weights_only=False)["net_state_dict"]
    profile = yaml.safe_load((REPO / args.profile).read_text())
    log_text = (REPO / args.log).read_text()

    checks = []
    ok1, msg1 = assess_learning(param_deltas(sd_b, sd_a))
    checks.append(("learning alive", ok1, msg1))
    ok2, notes = assess_mechanisms(profile, log_text)
    checks.append(("mechanism alive", ok2, "; ".join(notes)))
    ok3, msg3 = assess_sentinels(profile, args.iters,
                                 args.steps_per_iter, args.resume_iter)
    checks.append(("no fatal sentinel", ok3, msg3))

    all_ok = all(c[1] for c in checks)
    for name, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    print(f"\npreflight: {'PASS — spend the budget' if all_ok else 'FAIL — DO NOT RUN THE EXPERIMENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
