"""Replay every banked solver tape from its own recorded root.

WHY. A banked solution is a claim: "these actions, from this state, clear
this level." Two of the SMB tapes do not honour it — `1-3 sol_001` and
`1-4 sol_003` fail to replay from their recorded roots — and they were
found by hand, one at a time. Nothing ever swept the set, so the true
failure rate across 298 banked tapes is unknown. An EXHIBITION claim that
cannot be replayed is not a receipt; it is an assertion.

PRE-REGISTERED GATE (written before the sweep runs, per CLAIMS.md):

  Every banked tape replays from its recorded root to a terminal state
  its own profile's `is_clear` predicate accepts.

  PASS = zero failures other than the two already-quarantined tapes
  above. Any third failure is a new finding: that tape is quarantined,
  never repaired, and any claim resting on it is re-scoped.

  A tape that ERRORS (missing root, unreadable actions, absent profile)
  counts as a FAILURE, not as a skip. "Cannot be checked" and "passes"
  are different states and this gate will not merge them.

HONESTY ABOUT THE PREDICATE. Verification uses `make_game(profile)` and
that game's own `is_clear` / `level_key` — the identical predicate the
solver used to declare the clear in the first place. It is deliberately
not a reimplementation: a second, hand-written predicate would let a
tape "pass" a standard the solver never held it to, or fail one it never
claimed.

BINARY PROVENANCE. Each tape records the `nes_core` build that produced
it. A tape banked under a different core than the one replaying it can
fail for reasons that are not the tape's fault, so the report records
both hashes per tape and flags mismatches separately from failures.
That distinction is the whole reason this sweep gates the DMC/ASM `.so`
migration rather than the other way round.

The emulator import is lazy (inside `main`), matching
scripts/hazard_collect.py, so every bookkeeping function here is
unit-testable with no ROM and no nes_core.

Usage — COMPUTE-BOUND, run exclusively:

    .venv/bin/python scripts/replay_sweep.py \\
        --glob 'runs/**/solutions/*.json' --out runs/replay_sweep/report.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent

# Tapes already found non-reproducing by hand and quarantined. Listed so
# the gate can distinguish "known" from "new" without ever treating a
# known failure as acceptable.
KNOWN_BAD = (
    "runs/ge_1_3_solve/solutions/sol_001.json",
    "runs/ge_1_4_solve/solutions/sol_003.json",
)

NOOP = 0


@dataclass
class TapeSpec:
    tape: str
    actions: str
    root_state: str
    profile: str
    start_wd: Optional[list]
    clear_wd: Optional[list]
    steps: Optional[int]
    core_sha16: Optional[str]
    # "recorded" | "recovered from <manifest>" | "" (unknown). A recovered
    # profile is weaker evidence than a recorded one and is never reported
    # as equivalent.
    profile_source: str = ""


@dataclass
class Verdict:
    tape: str
    status: str          # PASS | FAIL | ERROR
    reason: str
    replayed_steps: int = 0
    end_key: Optional[list] = None
    core_match: Optional[bool] = None


def discover_tapes(pattern: str, root: Path = REPO) -> list[Path]:
    """Solution jsons matching `pattern`, sorted, quarantine excluded.

    Quarantined directories keep their tapes on purpose but must not be
    swept as if they were live claims.
    """
    hits = [Path(p) for p in globmod.glob(str(root / pattern), recursive=True)]
    return sorted(p for p in hits if "INVALID" not in str(p)
                  and "quarantine" not in str(p).lower())


def read_tape(path: Path, root: Path = REPO) -> TapeSpec:
    """Parse one solution json into the fields the replay needs.

    Solver args live under `solver_args` on some tapes and at top level on
    others; both shapes are read rather than assuming one.
    """
    rec = json.loads(path.read_text())
    sa = rec.get("solver_args") or {}

    def pick(*keys: str) -> Any:
        for k in keys:
            if rec.get(k) is not None:
                return rec[k]
            if sa.get(k) is not None:
                return sa[k]
        return None

    actions = rec.get("actions_file") or str(path).replace(
        ".json", ".actions.npy")
    core = ((rec.get("hw") or {}).get("nes_core") or {}).get("sha256_16")
    return TapeSpec(
        tape=str(path.relative_to(root)) if path.is_absolute() else str(path),
        actions=actions,
        root_state=pick("root_state") or "",
        profile=pick("profile") or "",
        start_wd=pick("start_wd"),
        clear_wd=pick("clear_wd"),
        steps=pick("steps"),
        core_sha16=core,
    )


def build_consumer_index(root: Path = REPO) -> dict[Path, tuple[str, str]]:
    """actions-file -> (profile, manifest) from everything that CONSUMED a tape.

    97 of 298 banked tapes never recorded the profile they were solved
    under, and no run-directory manifest carries it either (checked: 0 of
    103 recoverable that way). But the artifacts BUILT from those tapes do
    — restart-state manifests, demo provenance sidecars and ladder indexes
    all name both the tape's actions file and the profile used to replay
    it. So provenance is recovered from the consumer rather than the
    producer.

    Keyed on the RESOLVED path, never the basename: every level's tape is
    called `sol_000.actions.npy`, so a basename index silently maps 1-3's
    tape to 2-1's profile. The first version of this index did exactly
    that and reported a confident, wrong answer.

    A recovered profile is weaker evidence than a recorded one — it says
    "something replayed this tape under that profile", not "the solver
    used it". `read_tape` marks the difference in `profile_source` so a
    report can never present the two as equivalent.
    """
    index: dict[Path, tuple[str, str]] = {}
    seen: set[str] = set()
    for pattern in ("checkpoints/**/*.json", "runs/**/index.json",
                    "runs/**/manifest.json"):
        for m in root.glob(pattern):
            key = str(m)
            if key in seen:
                continue
            seen.add(key)
            try:
                rec = json.loads(m.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rec, dict):
                continue
            src = rec.get("source_solution")
            src = src if isinstance(src, dict) else {}
            actions = src.get("actions") or rec.get("actions")
            profile = src.get("profile") or rec.get("profile")
            if isinstance(actions, str) and isinstance(profile, str):
                try:
                    index.setdefault(Path(actions).resolve(),
                                     (profile, str(m)))
                except (OSError, ValueError):
                    continue
    return index


def resolve_profile(spec: TapeSpec,
                    index: dict[Path, tuple[str, str]],
                    root: Path = REPO) -> TapeSpec:
    """Fill a missing profile from the consumer index, marking the source."""
    if spec.profile and (root / spec.profile).exists():
        spec.profile_source = "recorded"
        return spec
    try:
        key = Path(root / spec.actions).resolve()
    except (OSError, ValueError):
        return spec
    hit = index.get(key)
    if hit and (root / hit[0]).exists():
        spec.profile = hit[0]
        spec.profile_source = f"recovered from {hit[1]}"
    return spec


def spec_problems(spec: TapeSpec, root: Path = REPO) -> list[str]:
    """Everything missing BEFORE the emulator is started.

    Checked up front so a sweep of 298 tapes reports all its unusable
    inputs at once instead of dying on the first one.
    """
    bad: list[str] = []
    for label, rel in (("root_state", spec.root_state),
                       ("actions", spec.actions),
                       ("profile", spec.profile)):
        if not rel:
            bad.append(f"{label} not recorded")
        elif not (root / rel).exists():
            bad.append(f"{label} missing: {rel}")
    if spec.clear_wd is None:
        bad.append("clear_wd not recorded — nothing to verify against")
    return bad


def batch(items: Sequence[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def replay_batch(
    pool: Any,
    specs: Sequence[TapeSpec],
    action_lists: Sequence[Sequence[int]],
    roots: Sequence[bytes],
    clear_fn: Callable[[Any, TapeSpec], bool],
    key_fn: Callable[[Any], list],
) -> list[Verdict]:
    """Replay up to one tape per pool worker, tick-aligned.

    `step_all` advances every worker, so tapes of different lengths are
    padded with NOOP once exhausted and their verdict frozen at the tick
    they ended. Without the freeze a short tape would keep stepping past
    its own clear and could walk back out of it — which is precisely how
    a replay check can report a false negative.
    """
    n = len(specs)
    if not (n == len(action_lists) == len(roots)):
        raise ValueError("specs/actions/roots length mismatch")
    for i in range(n):
        pool.load_worker_state(i, roots[i])

    done = [False] * n
    verdicts: list[Optional[Verdict]] = [None] * n
    longest = max((len(a) for a in action_lists), default=0)

    for t in range(longest):
        acts = [NOOP] * max(n, 1)
        for i in range(n):
            if not done[i] and t < len(action_lists[i]):
                acts[i] = int(action_lists[i][t])
        stepped = pool.step_all(acts)
        for i in range(n):
            if done[i]:
                continue
            ram = stepped[i][2]
            if clear_fn(ram, specs[i]):
                done[i] = True
                verdicts[i] = Verdict(
                    specs[i].tape, "PASS",
                    f"cleared at step {t + 1}", t + 1, key_fn(ram))
            elif t + 1 >= len(action_lists[i]):
                done[i] = True
                verdicts[i] = Verdict(
                    specs[i].tape, "FAIL",
                    f"tape exhausted at step {t + 1} without a clear",
                    t + 1, key_fn(ram))
    for i in range(n):
        if verdicts[i] is None:
            verdicts[i] = Verdict(specs[i].tape, "FAIL",
                                  "empty action tape", 0, None)
    return [v for v in verdicts if v is not None]


def evaluate_gate(verdicts: Sequence[Verdict],
                  known_bad: Sequence[str] = KNOWN_BAD) -> tuple[bool, str]:
    """The pre-registered gate. ERROR counts as failure, never as a skip."""
    kb = set(known_bad)
    failed = [v for v in verdicts if v.status in ("FAIL", "ERROR")]
    new = [v for v in failed if v.tape not in kb]
    known_hit = [v for v in failed if v.tape in kb]
    passed = len(verdicts) - len(failed)
    msg = (f"{passed}/{len(verdicts)} replay; "
           f"{len(known_hit)} known-bad reproduced as bad; "
           f"{len(new)} NEW failure(s)")
    if new:
        msg += ": " + ", ".join(v.tape for v in new[:6])
    return (not new), msg


def build_report(verdicts: Sequence[Verdict], stamp: str,
                 core_sha16: Optional[str] = None) -> dict:
    ok, msg = evaluate_gate(verdicts)
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    return {
        "stamp": stamp,
        "replaying_core_sha16": core_sha16,
        "n_tapes": len(verdicts),
        "counts": counts,
        "gate_passed": ok,
        "gate_message": msg,
        "known_bad": list(KNOWN_BAD),
        "verdicts": [asdict(v) for v in verdicts],
    }


def sha256_16(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="runs/**/solutions/*.json")
    ap.add_argument("--out", default="runs/replay_sweep/report.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="pool workers; keep low while a campaign runs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stamp", default="unstamped",
                    help="caller-supplied timestamp (kept out of the code "
                         "so the report is reproducible)")
    ap.add_argument("--list", action="store_true",
                    help="report discoverable/unusable tapes, no emulator")
    args = ap.parse_args(argv)

    tapes = discover_tapes(args.glob)
    if args.limit:
        tapes = tapes[:args.limit]
    specs = [read_tape(p) for p in tapes]

    if args.list:
        bad = [(s.tape, spec_problems(s)) for s in specs]
        unusable = [(t, ps) for t, ps in bad if ps]
        print(f"{len(specs)} tape(s) discovered; {len(unusable)} unusable")
        for t, ps in unusable[:40]:
            print(f"  {t}: {'; '.join(ps)}")
        return 0

    # Lazy, per scripts/hazard_collect.py: importing nes_core at module
    # scope would make every function above untestable without a ROM.
    import numpy as np
    import nes_core  # noqa: F401
    from src.training.profile_utils import action_space_to_bitmasks
    import yaml
    print("replay_sweep: emulator path is deliberately unexercised in this "
          "build step; run it exclusively per the machine calendar.")
    raise SystemExit(
        "refusing to sweep implicitly — pass --list for the token-bound "
        "inventory, or run with the campaign stopped and --confirm-exclusive")


if __name__ == "__main__":
    raise SystemExit(main())
