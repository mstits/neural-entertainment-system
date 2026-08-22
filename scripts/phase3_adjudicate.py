"""Score Phase 3 against its pre-registered gate.

Written BEFORE either arm finished, deliberately. An analysis authored
after seeing the numbers is an analysis fitted to them, and this project
has a standing rule that a gate is registered before the run it judges.

THE GATE (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md, Phase 3):

    Strict honest clear rate over >=100 episodes must improve by >=20%
    RELATIVE to the unmasked control on a mid-difficulty level.

    Below that, Phase 3 FAILS and the synthesis's instruction is to
    terminate the substrate experiment — not to tune the threshold, not
    to try another level, not to re-run with a longer budget. Those are
    all available afterwards as new, separately registered experiments;
    none of them may be used to rescue this one.

RELATIVE, not absolute: 0.38 -> 0.456 passes, 0.38 -> 0.45 does not.
Computed as (masked - control) / control, which is undefined when the
control scores zero — in that case the comparison is reported as
UNSCORABLE rather than as an infinite improvement.

READING THE VETO'S OWN TELEMETRY. A masked arm where nearly every state
was FULLY vetoed had its mask dropped by the escape hatch nearly
everywhere, so it is close to a second control and a null result there
says little about the veto. That is reported alongside the verdict rather
than folded into it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GATE_RELATIVE_IMPROVEMENT = 0.20
MIN_EPISODES = 100
REQUIRED_SEEDS = 2


def pooled_rate(records: list[dict]) -> tuple[int, int, list]:
    """(clears, episodes, seeds) over honest-protocol records only."""
    clears = eps = 0
    seeds: list = []
    for r in records:
        n = int(r.get("n_episodes") or 0)
        if n < 50 or float(r.get("sticky_prob") or 0) <= 0:
            continue          # not the honest protocol; not admissible
        rate = r.get("clear_rate")
        if rate is None:
            continue
        clears += int(round(float(rate) * n))
        eps += n
        seeds.append(r.get("eval_seed"))
    return clears, eps, sorted(set(seeds), key=str)


def adjudicate(control: list[dict], masked: list[dict],
               veto_stats: dict | None = None) -> dict:
    c_cl, c_n, c_seeds = pooled_rate(control)
    m_cl, m_n, m_seeds = pooled_rate(masked)
    problems = []
    for label, n, seeds in (("control", c_n, c_seeds), ("masked", m_n, m_seeds)):
        if n < MIN_EPISODES:
            problems.append(f"{label} has {n} admissible episodes "
                            f"(< {MIN_EPISODES})")
        if len(seeds) < REQUIRED_SEEDS:
            problems.append(f"{label} used {len(seeds)} seed(s) "
                            f"(< {REQUIRED_SEEDS})")
    c_rate = c_cl / c_n if c_n else 0.0
    m_rate = m_cl / m_n if m_n else 0.0

    if problems:
        verdict, rel = "UNSCORABLE", None
    elif c_rate == 0.0:
        verdict, rel = "UNSCORABLE", None
        problems.append("control scored zero — relative improvement is "
                        "undefined and must not be reported as infinite")
    else:
        # Rounded BEFORE comparison, and to the same places the verdict
        # reports, so the decision always agrees with the number printed
        # beside it. Unrounded, (0.48 - 0.40) / 0.40 is 0.19999999999999998
        # and an exact-threshold run would print "relative_improvement:
        # 0.2" next to "verdict: FAIL". Clear rates are k/n at n>=100, so
        # nothing real lives below this resolution anyway.
        rel = round((m_rate - c_rate) / c_rate, 4)
        verdict = "PASS" if rel >= GATE_RELATIVE_IMPROVEMENT else "FAIL"

    out = {
        "gate": f">= {GATE_RELATIVE_IMPROVEMENT:.0%} relative improvement, "
                f">= {MIN_EPISODES} episodes over {REQUIRED_SEEDS} seeds",
        "control": {"clears": c_cl, "episodes": c_n, "rate": round(c_rate, 4),
                    "seeds": c_seeds},
        "masked": {"clears": m_cl, "episodes": m_n, "rate": round(m_rate, 4),
                   "seeds": m_seeds},
        "relative_improvement": rel,
        "verdict": verdict,
        "problems": problems,
    }
    if veto_stats:
        fv = float(veto_stats.get("fully_vetoed_fraction", 0.0))
        out["veto"] = dict(veto_stats)
        out["veto_caveat"] = (
            "the escape hatch dropped the mask in "
            f"{fv:.1%} of states; above ~50% the masked arm approaches a "
            "second control and a null result says little about the veto"
            if fv > 0.5 else None)
    if verdict == "FAIL":
        out["instruction"] = (
            "Per the synthesis: terminate the substrate experiment. Do NOT "
            "tune the threshold, change level, or extend the budget to "
            "rescue this result — any of those is a new experiment "
            "needing its own registration.")
    return out


def load_eval_records(ckpt_dir: Path) -> list[dict]:
    log = ckpt_dir / "eval.jsonl"
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control-dir",
                    default="checkpoints/mario_1_2_phase3_control")
    ap.add_argument("--masked-dir",
                    default="checkpoints/mario_1_2_phase3_masked")
    ap.add_argument("--veto-stats", default=None)
    ap.add_argument("--out", default="runs/phase3/verdict.json")
    args = ap.parse_args(argv)

    veto = None
    if args.veto_stats and (REPO / args.veto_stats).exists():
        veto = json.loads((REPO / args.veto_stats).read_text())
    v = adjudicate(load_eval_records(REPO / args.control_dir),
                   load_eval_records(REPO / args.masked_dir), veto)
    print(json.dumps(v, indent=2))
    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2) + "\n")
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
