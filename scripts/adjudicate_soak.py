"""Static content checks for a completed soak receipt trail.

Complements ``soak_harness.py --verify``: --verify proves the hash chain
is intact and the final receipt's counts match the trail; this checker
proves the trail's *contents* score cleanly, without re-running any
gameplay:

- manifest and final receipt are scoreable (never a selfcheck stub) and
  the final receipt says the soak completed;
- every segment receipt is scoreable and its outcome is a known class;
- every CLEAR's solution sidecar and action tape exist on disk, hash to
  the values recorded in the chained receipt, and the sidecar is
  replay-verified with a step count matching the receipt;
- no two CLEARs share an action tape (per-segment seeds differ by
  construction, so a duplicate tape means a copied solution);
- every BUDGET segment was a real attempt: solver exited cleanly, met a
  step floor, and banked zero solutions (a solution in a BUDGET tail is
  a missed clear);
- a game's config_sha256 never changes between two of that game's own
  rotation slots (a mid-soak edit to its profile config is a defect
  even when every receipt is internally hash-chained);
- the rotation strictly cycles the manifest's roster order;
- optionally, the roster file still hashes to the manifest's value.

With ``--baseline <prior_soak_dir>``, same-index segments are compared
by seed and tape hash: index-derived seeds make same-index segments
exact replays of a prior trail, which is reported as info (expected,
not a defect) so adjudication can count only novel-seed evidence.

Read-only. Prints a JSON report; exits nonzero if any problem is found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

KNOWN_OUTCOMES = {"CLEAR", "BUDGET", "STALL", "CRASH", "UNSCORABLE"}
BUDGET_STEP_FLOOR = 100_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _seed_of(detail: dict):
    argv = detail.get("argv") or []
    for i, a in enumerate(argv[:-1]):
        if a == "--seed":
            return argv[i + 1]
    return None


def _segment_index(seg_dir: Path) -> int:
    # Segment dirs are named seg_{count:04d}_{name}; count stops
    # zero-padding past 9999, so a plain string sort desyncs from
    # chronological order once a soak crosses that many segments
    # (e.g. "seg_10000_x" < "seg_1000_y" lexicographically).
    return int(seg_dir.name.split("_")[1])


def _segment_dirs(soak_dir: Path) -> list:
    return sorted((soak_dir / "segments").glob("seg_*"), key=_segment_index)


def check_trail(soak_dir: Path, roster_path: Path | None = None,
                budget_step_floor: int = BUDGET_STEP_FLOOR,
                in_flight: bool = False) -> dict:
    problems: list = []
    info: list = []
    report = {"soak_dir": str(soak_dir), "problems": problems, "info": info,
              "mode": "in_flight" if in_flight else "verdict"}

    manifest_path = soak_dir / "soak_manifest.json"
    if not manifest_path.exists():
        problems.append("soak_manifest.json missing")
        return report
    manifest = _load(manifest_path)
    if manifest.get("selfcheck"):
        problems.append("manifest is a selfcheck run: not scoreable")
        return report
    if not manifest.get("backend_scoreable"):
        problems.append("manifest backend is not scoreable")
        return report
    roster_names = [e["name"] for e in manifest.get("roster", [])]
    if not roster_names:
        problems.append("manifest roster is empty")
        return report

    if roster_path is not None:
        if not roster_path.exists():
            problems.append(f"roster file not found: {roster_path}")
        elif _sha256(roster_path) != manifest.get("roster_sha256"):
            problems.append("roster file sha256 does not match manifest")

    final_path = soak_dir / "final_receipt.json"
    final = None
    if final_path.exists():
        final = _load(final_path)
        if final.get("status") != "completed":
            problems.append(f"final receipt status={final.get('status')!r}")
        if final.get("selfcheck"):
            problems.append("final receipt marked selfcheck: not scoreable")
        if not final.get("backend_scoreable"):
            problems.append("final receipt backend not scoreable")
    elif in_flight:
        # A running soak has no final receipt yet. The per-segment
        # content checks below are still meaningful on the segments
        # that HAVE closed, which is what in-flight mode is for: catch
        # a content defect hours before the verdict run. It is never
        # itself a verdict — see the mode field and main()'s banner.
        info.append("final_receipt.json absent: soak still running — "
                    "content checks only, NOT a verdict")
    else:
        problems.append("final_receipt.json missing: soak did not complete")
        return report

    tally: Counter = Counter()
    per_game: dict = {}
    config_by_game: dict = {}
    clears: list = []
    tapes: dict = {}
    scored = 0
    seg_dirs = _segment_dirs(soak_dir)
    for idx, seg in enumerate(seg_dirs):
        rp = seg / "receipt.json"
        if not rp.exists():
            # Receipts are written at segment end, so in a running soak
            # the newest segment legitimately has none yet. Any EARLIER
            # segment without one is a real gap, in either mode.
            if in_flight and idx == len(seg_dirs) - 1:
                info.append(f"{seg.name}: in progress, no receipt yet")
                continue
            problems.append(f"{seg.name}: receipt.json missing")
            continue
        scored += 1
        r = _load(rp)
        name, outcome = r.get("name"), r.get("outcome")
        d = r.get("detail") or {}
        tally[outcome] += 1
        per_game.setdefault(name, Counter())[outcome] += 1
        cfg_sha = r.get("config_sha256")
        prior_cfg = config_by_game.setdefault(name, cfg_sha)
        if cfg_sha != prior_cfg:
            problems.append(
                f"{seg.name}: config_sha256 {cfg_sha!r} differs from an "
                f"earlier {name!r} segment's {prior_cfg!r} — config "
                "changed mid-soak")
        if not r.get("backend_scoreable"):
            problems.append(f"{seg.name}: receipt not backend_scoreable")
        if outcome not in KNOWN_OUTCOMES:
            problems.append(f"{seg.name}: unknown outcome {outcome!r}")
            continue
        if name != roster_names[idx % len(roster_names)]:
            problems.append(
                f"{seg.name}: rotation order broken (expected "
                f"{roster_names[idx % len(roster_names)]!r})")
        if outcome == "CLEAR":
            clears.append(_check_clear(seg, d, tapes, problems))
        elif outcome == "BUDGET":
            _check_budget(seg, d, budget_step_floor, problems)
        else:
            info.append(f"{seg.name}: outcome {outcome} — adjudicate by hand")

    if final is not None:
        for key in ("CLEAR", "BUDGET", "STALL", "CRASH", "UNSCORABLE"):
            if final.get("outcomes", {}).get(key, 0) != tally.get(key, 0):
                problems.append(
                    f"final receipt outcome count mismatch for {key}: "
                    f"receipt {final.get('outcomes', {}).get(key)} vs "
                    f"segments {tally.get(key, 0)}")
        if final.get("segments") != len(seg_dirs):
            problems.append(
                f"final receipt segments={final.get('segments')} but "
                f"{len(seg_dirs)} segment dirs on disk")
        # ZERO HUMAN INTERVENTIONS is the mission's central pass criterion,
        # so it is a problem, not a note. This was an info line until
        # adversary audit #3 built a fully self-consistent trail carrying 7
        # interventions and 7 diagnosis bundles that passed both --verify
        # and this checker with "0 problem(s)", exit 0.
        n_int = final.get("interventions", 0)
        if n_int != 0:
            problems.append(
                f"{n_int} human intervention(s) logged: the soak did NOT run "
                "unattended — see interventions.jsonl and its diagnosis "
                "bundles")
        if final.get("passed_zero_interventions") is not True:
            problems.append(
                "final receipt passed_zero_interventions is not true: the "
                "harness itself did not certify an unattended run")
        # A soak whose segments crashed or stalled has not failed by
        # definition — the oracle allows an honest stall that is diagnosed
        # and advanced past — but it must never be reported as an
        # uneventful "0 problem(s)" run. Surface it for hand adjudication.
        notable = {k: tally.get(k, 0)
                   for k in ("STALL", "CRASH", "UNSCORABLE") if tally.get(k)}
        if notable:
            info.append(
                "outcomes requiring hand adjudication: "
                + ", ".join(f"{k}={v}" for k, v in notable.items()))

    report["segments"] = len(seg_dirs)
    report["scored_segments"] = scored
    report["outcomes"] = dict(tally)
    report["per_game"] = {g: dict(c) for g, c in per_game.items()}
    report["clears"] = [c for c in clears if c]
    return report


def _check_clear(seg: Path, d: dict, tapes: dict, problems: list):
    for field in ("solution_sidecar", "sidecar_sha256", "tape_sha256",
                  "solution_steps"):
        if not d.get(field):
            problems.append(f"{seg.name}: CLEAR missing {field}")
            return None
    sidecar = Path(d["solution_sidecar"])
    tape = sidecar.parent / (sidecar.stem + ".actions.npy")
    if not sidecar.exists():
        problems.append(f"{seg.name}: sidecar missing on disk")
        return None
    if _sha256(sidecar) != d["sidecar_sha256"]:
        problems.append(f"{seg.name}: sidecar sha256 mismatch")
    if not tape.exists():
        problems.append(f"{seg.name}: action tape missing on disk")
    elif _sha256(tape) != d["tape_sha256"]:
        problems.append(f"{seg.name}: action tape sha256 mismatch")
    sc = _load(sidecar)
    if not sc.get("replay_verified"):
        problems.append(f"{seg.name}: sidecar not replay_verified")
    if sc.get("steps") != d["solution_steps"]:
        problems.append(
            f"{seg.name}: sidecar steps {sc.get('steps')} != receipt "
            f"solution_steps {d['solution_steps']}")
    prior = tapes.setdefault(d["tape_sha256"], seg.name)
    if prior != seg.name:
        problems.append(
            f"{seg.name}: action tape duplicates {prior} — copied "
            f"solution, not an independent solve")
    return {"seg": seg.name, "steps": d["solution_steps"],
            "seed": _seed_of(d), "tape_sha256": d["tape_sha256"]}


def _check_budget(seg: Path, d: dict, floor: int, problems: list) -> None:
    pl = d.get("progress_last") or {}
    if not pl:
        problems.append(f"{seg.name}: BUDGET with no progress sample")
        return
    if pl.get("solutions", 0) != 0:
        problems.append(
            f"{seg.name}: BUDGET banked {pl.get('solutions')} solution(s) "
            f"— missed clear")
    if (pl.get("steps") or 0) < floor:
        problems.append(
            f"{seg.name}: BUDGET steps {pl.get('steps')} below floor "
            f"{floor} — not a real attempt")
    if d.get("solver_exit", 0) != 0:
        problems.append(f"{seg.name}: solver_exit {d.get('solver_exit')}")


def compare_baseline(soak_dir: Path, baseline_dir: Path, info: list) -> None:
    base = {p.name: p for p in _segment_dirs(baseline_dir)}
    replayed = 0
    for seg in _segment_dirs(soak_dir):
        other = base.get(seg.name)
        if other is None:
            continue
        try:
            d = _load(seg / "receipt.json").get("detail") or {}
            bd = _load(other / "receipt.json").get("detail") or {}
        except (OSError, ValueError):
            continue
        if _seed_of(d) is not None and _seed_of(d) == _seed_of(bd):
            replayed += 1
    info.append(
        f"{replayed} same-index segment(s) share their seed with "
        f"{baseline_dir.name}: index-derived seeds replay the baseline; "
        f"count only later segments as novel-seed evidence")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("soak_dir", type=Path)
    ap.add_argument("--roster", type=Path, default=None,
                    help="roster file to hash-check against the manifest")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="prior soak dir; report same-seed segment overlap")
    ap.add_argument("--budget-step-floor", type=int,
                    default=BUDGET_STEP_FLOOR)
    ap.add_argument("--in-flight", action="store_true",
                    help="soak is still running: run the per-segment "
                         "content checks over the segments that have "
                         "closed, tolerating an absent final receipt and "
                         "an in-progress last segment. Never a verdict.")
    args = ap.parse_args(argv)

    report = check_trail(args.soak_dir, roster_path=args.roster,
                         budget_step_floor=args.budget_step_floor,
                         in_flight=args.in_flight)
    if args.baseline is not None:
        compare_baseline(args.soak_dir, args.baseline, report["info"])
    json.dump(report, sys.stdout, indent=1)
    print()
    n = len(report["problems"])
    if args.in_flight:
        print("adjudicate_soak: IN-FLIGHT SPOT-CHECK — NOT A VERDICT; "
              "re-run without --in-flight once the soak completes",
              file=sys.stderr)
    # Print the outcome mix alongside the problem count: "0 problem(s)" on
    # its own reads as "an uneventful run", which is exactly how an
    # all-CRASH trail slipped past in adversary audit #3.
    mix = report.get("outcomes") or {}
    mix_s = " ".join(f"{k}={mix[k]}" for k in sorted(mix)) or "no outcomes"
    print(f"adjudicate_soak: {report.get('scored_segments', 0)} scored / "
          f"{report.get('segments', 0)} segments, {n} problem(s) [{mix_s}]",
          file=sys.stderr)
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
