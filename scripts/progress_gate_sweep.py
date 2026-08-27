"""Run scripts/progress_signal_gate.py across the whole roster and diff
the verdict each profile gets NOW against the verdict the pre-2026-08-26
gate gave it.

The roster is derived, never listed: every `configs/*.yaml` carrying a
`solve:` block (45 of them today). A hardcoded list would silently stop
covering a profile the day someone onboards one, which is the same class
of quiet under-coverage the gate itself was found to have.

Two things this measures, kept separate on purpose so one variable moves
at a time:

  * THE FIX. Same probe, same traces, `--min-window 0` (the old
    behaviour, exactly) against the default floor. Every verdict change
    in this column is attributable to the truncation-order fix and to
    nothing else.
  * THE PROBE. `--probe random` against `--probe hold`, both under the
    fixed assessor. This column measures how much of the roster's
    "unusable" was the undodging scripted hold walking into the first
    hazard, and it can only ever be read as evidence about the WINDOW —
    a random probe commands no direction, so it cannot certify an axis
    sign (see PROBE_DISARMS).

    .venv/bin/python scripts/progress_gate_sweep.py --probe hold \\
        --out docs/receipts/progress_gate_window_sweep_2026-08-26.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.progress_signal_gate import (  # noqa: E402
    MIN_ASSESSABLE_STEPS,
    assess_probe,
    run_probe,
)

#: The verdict `assess()` produced before 2026-08-26. `min_window=0`
#: puts every window above the floor, so the shortfall finding is always
#: issued as an instrument fault — bit-for-bit the old behaviour, which
#: is what makes the diff below a diff and not two different runs.
LEGACY_MIN_WINDOW = 0


def roster() -> list[str]:
    """Profiles the gate is defined on: those with a `solve:` block."""
    import yaml
    out = []
    for p in sorted((REPO / "configs").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("solve"):
            out.append(f"configs/{p.name}")
    return out


def probe_plan(profile: str, forward_policy: str = "default") -> dict:
    """How to drive THIS profile: which signal to trace and which
    singleton action counts as "forward".

    `forward_policy="default"` uses the gate's own `--forward right`
    default for every profile. That is deliberately the default HERE
    too, even though it is plainly the wrong hold for a vertical
    shooter, because it is what every banked verdict was measured with
    (docs/receipts/progress_gate_stasis_sweep_2026-08-26.json) — and the
    number this sweep exists to report is how many verdicts the
    TRUNCATION-ORDER FIX moves, which is only a clean reading if the
    probe does not move at the same time. 1942 is the case that proves
    the point: held `right` it rides the auto-scroll for all 1200 steps
    and passes; held `up` (its declared `-y` axis) it flies into the
    first enemy and dies at step 21. Both are facts about the probe.

    `forward_policy="axis"` derives the hold from the profile's own
    declared odometer axis (`-y` -> up, `+y` -> down, `-x` -> left, else
    right) — no game knowledge, only the profile's own declaration.

    A profile that declares no progress signal at all is not a gate
    failure and must not be scored as one — it is out of the gate's
    domain, reported as INAPPLICABLE.
    """
    import yaml
    d = yaml.safe_load((REPO / profile).read_text())
    solve = d.get("solve") or {}
    prog = solve.get("progress") or {}
    src = str(prog.get("source", "")).lower()
    odometer = src == "odometer"
    axis = str(prog.get("axis", "x")).lower()
    forward = "right"
    if forward_policy == "axis" and odometer:
        forward = {"-y": "up", "+y": "down", "y": "down",
                   "-x": "left"}.get(axis, "right")
    applicable = odometer or prog.get("lo") is not None
    return {"odometer": odometer, "forward": forward,
            "applicable": applicable,
            "reason": "" if applicable else
                      f"no solve.progress.lo and source={src or 'unset'} "
                      f"— this profile declares no scalar progress signal, "
                      f"so the gate has nothing to assess (not a failure)"}


def collect(profiles: list[str], *, probe: str, steps: int, episodes: int,
            seed: int, cache: Path | None,
            forward_policy: str = "default") -> dict:
    """Raw traces for every profile, from `cache` if it already holds
    them so the assessor can be re-run without re-emulating 54,000
    steps of NES."""
    if cache and cache.exists():
        with gzip.open(cache, "rt") as fh:
            return json.load(fh)
    out = {}
    for prof in profiles:
        plan = probe_plan(prof, forward_policy)
        if not plan["applicable"]:
            out[prof] = {"error": plan["reason"], "plan": plan}
            print(f"  {prof}: INAPPLICABLE — {plan['reason']}", flush=True)
            continue
        try:
            c = run_probe(prof, steps=steps, forward=plan["forward"],
                          odometer=plan["odometer"], probe=probe,
                          episodes=episodes, seed=seed)
        except SystemExit as e:
            out[prof] = {"error": f"probe could not run: {e}", "plan": plan}
            print(f"  {prof}: ERROR {e}", flush=True)
            continue
        c["plan"] = plan
        out[prof] = c
        print(f"  {prof}: live windows {c['live_steps_per_episode']}",
              flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt") as fh:
            json.dump(out, fh)
    return out


def _row(prof: str, collected: dict, min_window: int) -> dict:
    v = assess_probe(collected, min_window=min_window)
    return {"verdict": v["verdict"], "passed": v["passed"],
            "steps": v["steps"], "distinct": v["distinct"],
            "steps_to_min_distinct": v["steps_to_min_distinct"],
            "instrument_findings": v["instrument_findings"],
            "inconclusive_findings": v["inconclusive_findings"],
            "behaviour_findings": v["behaviour_findings"]}


def evaluate(traces: dict, *, min_window: int) -> list[dict]:
    rows = []
    for prof, c in traces.items():
        if "error" in c:
            rows.append({"profile": prof, "applicable": False,
                         "error": c["error"],
                         "before": {"verdict": "GATE INAPPLICABLE"},
                         "after": {"verdict": "GATE INAPPLICABLE"},
                         "changed": False})
            continue
        before = _row(prof, c, LEGACY_MIN_WINDOW)
        after = _row(prof, c, min_window)
        rows.append({
            "profile": prof, "applicable": True,
            "probe": c["probe"], "odometer": c["odometer"],
            "forward": c["plan"]["forward"],
            "requested_steps": c["steps"],
            "live_steps_per_episode": c["live_steps_per_episode"],
            "live_steps": after["steps"],
            "before": before, "after": after,
            "changed": before["verdict"] != after["verdict"],
            "passed_changed": before["passed"] != after["passed"],
        })
    return rows


def calibration(rows: list[dict]) -> dict:
    """The number MIN_ASSESSABLE_STEPS is set from, recomputed from this
    sweep's own data so it can be checked rather than believed.

    Population: every profile whose signal is DEMONSTRATED to reach
    MIN_DISTINCT levels — not just the ones that pass. Kung Fu is why.
    Its byte $0094 reaches 91 distinct levels and fails this gate on the
    unpaired-wrap check, which says nothing about resolution; it needed
    187 live steps to show 32 of them, and that is direct evidence about
    how long a real progress signal can take. Restricting the population
    to passing profiles would have thrown that evidence away and
    produced a floor of 102.

    The max is a LOWER BOUND and is honest about it: any signal that
    takes longer to reach MIN_DISTINCT than its own probe survived is
    censored out of this population entirely — which is precisely the
    defect being fixed. Erring low costs a false SIGNAL UNUSABLE; erring
    high costs an extra INCONCLUSIVE. Re-derive it when the roster grows
    or when the probe gets a longer window.
    """
    reached = [r for r in rows
               if r.get("applicable")
               and r["after"]["steps_to_min_distinct"] is not None]
    per = {r["profile"]: r["after"]["steps_to_min_distinct"] for r in reached}
    return {"profiles_reaching_min_distinct": len(reached),
            "steps_to_min_distinct": per,
            "max": max(per.values()) if per else None,
            "slowest_profile": max(per, key=per.get) if per else None}


def sensitivity(traces: dict, floors: list[int]) -> dict:
    """How many verdicts move at each candidate floor.

    A single headline number invites the question the calibration cannot
    fully answer — how much of it is the constant? — so the answer ships
    with it rather than waiting to be asked.
    """
    out = {}
    for f in floors:
        rows = evaluate(traces, min_window=f)
        out[str(f)] = {
            "verdicts_changed": sum(1 for r in rows if r["changed"]),
            "passed_changed": sum(1 for r in rows if r.get("passed_changed")),
            "profiles": [r["profile"] for r in rows if r["changed"]],
        }
    return out


def compose(hold_rows: list[dict], random_rows: list[dict]) -> list[dict]:
    """The roster-level reading across BOTH probes, which is not either
    column on its own.

    Each probe can demonstrate things the other cannot, and neither is
    strictly better:

      * only the directed hold can demonstrate a FAULT — an axis-sign
        error (1942), a wrap with no high byte (Kung Fu, whose random
        rollout never travels far enough to reach 200 and so never sees
        the fault at all), a shortfall, a static camera;
      * only a probe that survives can demonstrate RESOLUTION — Contra
        reaches 20 distinct in the hold's 69 live steps and 346 in the
        random probe's 721.

    So: faults are the UNION (a fault demonstrated once stays
    demonstrated), and resolution is the BEST evidence either probe
    produced. Anything else is a verdict picked to taste.
    """
    ran = {r["profile"]: r for r in random_rows}
    out = []
    for h in hold_rows:
        r = ran.get(h["profile"])
        if not h.get("applicable"):
            out.append({"profile": h["profile"], "verdict": "GATE INAPPLICABLE"})
            continue
        faults = list(h["after"]["instrument_findings"])
        if r is not None:
            for f in r["after"]["instrument_findings"]:
                if f not in faults:
                    faults.append(f)
        best = max([h, r] if r else [h],
                   key=lambda x: x["after"]["distinct"])
        resolved = best["after"]["distinct"] >= 32
        if faults:
            verdict = "SIGNAL UNUSABLE"
        elif resolved:
            verdict = "SIGNAL SOUND"
        else:
            verdict = "INCONCLUSIVE — probe died too early to assess"
        out.append({
            "profile": h["profile"], "verdict": verdict,
            "faults": faults,
            "hold": {"live_steps": h["live_steps"],
                     "distinct": h["after"]["distinct"],
                     "verdict": h["after"]["verdict"]},
            "random": None if r is None else {
                "live_steps": r["live_steps"],
                "distinct": r["after"]["distinct"],
                "verdict": r["after"]["verdict"]},
            "best_probe": best["probe"],
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", default="hold")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-window", type=int, default=MIN_ASSESSABLE_STEPS)
    ap.add_argument("--forward-policy", default="default",
                    choices=["default", "axis"],
                    help="default: hold `right` on every profile, as the "
                         "gate itself does and as every banked verdict "
                         "was measured. axis: derive the hold from the "
                         "profile's own declared odometer axis")
    ap.add_argument("--cache", default=None,
                    help="gzipped raw-trace cache; reused if present")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compose-with", default=None,
                    help="another sweep receipt (the other probe) to "
                         "compose this one with; writes the union-of-"
                         "faults / best-resolution roster reading")
    ap.add_argument("--only", default=None, help="comma-separated substrings")
    args = ap.parse_args(argv)

    profiles = roster()
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        profiles = [p for p in profiles if any(w in p for w in want)]
    print(f"roster: {len(profiles)} profiles, probe={args.probe}")
    traces = collect(profiles, probe=args.probe, steps=args.steps,
                     episodes=args.episodes, seed=args.seed,
                     cache=Path(args.cache) if args.cache else None,
                     forward_policy=args.forward_policy)
    rows = evaluate(traces, min_window=args.min_window)
    changed = [r for r in rows if r["changed"]]
    cal = calibration(rows)
    sens = sensitivity(traces, [32, 102, MIN_ASSESSABLE_STEPS, 289, 600, 1200])

    print(f"\n{'profile':40s} {'live':>5s}  before -> after")
    for r in rows:
        mark = "*" if r["changed"] else " "
        live = r.get("live_steps", "-")
        print(f"{mark}{r['profile'][8:]:39s} {str(live):>5s}  "
              f"{r['before']['verdict']} -> {r['after']['verdict']}")
    print(f"\nverdicts changed: {len(changed)} of {len(rows)}")
    print(f"passed changed:   {sum(1 for r in rows if r.get('passed_changed'))}"
          f"  (the fix relabels; it must never unblock)")
    print(f"calibration: the slowest signal that DOES reach 32 distinct "
          f"needed {cal['max']} live steps ({cal['slowest_profile']}); "
          f"MIN_ASSESSABLE_STEPS={args.min_window}")
    print("sensitivity (floor -> verdicts changed / passed changed):")
    for f, v in sens.items():
        print(f"  {f:>5s} -> {v['verdicts_changed']:2d} / "
              f"{v['passed_changed']}")

    composed = None
    if args.compose_with:
        other = json.loads((REPO / args.compose_with).read_text())
        hold_rows, rand_rows = ((rows, other["rows"])
                                if args.probe == "hold"
                                else (other["rows"], rows))
        composed = compose(hold_rows, rand_rows)
        tally = {}
        for c in composed:
            tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
        print("\ncomposed across both probes (faults union, best "
              "resolution):")
        for c in composed:
            print(f"  {c['profile'][8:]:36s} {c['verdict']}")
        print(f"  tally: {tally}")

    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "probe": args.probe, "steps": args.steps,
            "forward_policy": args.forward_policy,
            "episodes": args.episodes, "seed": args.seed,
            "min_window": args.min_window,
            "legacy_min_window": LEGACY_MIN_WINDOW,
            "n_profiles": len(rows), "verdicts_changed": len(changed),
            "changed_profiles": [r["profile"] for r in changed],
            "calibration": cal, "sensitivity": sens,
            "composed": composed,
            "rows": rows}, indent=2) + "\n")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
