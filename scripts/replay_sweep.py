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
import sys
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


def sibling_profiles(specs: Sequence[TapeSpec]) -> dict[str, str]:
    """solutions-dir -> profile, from whichever tapes in it DID resolve.

    Tapes sharing a `solutions/` directory came out of one solver
    invocation, so they share its profile. When some siblings resolved
    and others did not, the directory's profile is known and the gap is
    bookkeeping.

    This is the WEAKEST of the three provenance tiers and is labelled as
    such, never merged with the other two:

      recorded            the tape names its own profile
      consumer manifest   something that replayed the tape names it
      sibling             a tape from the same solve names it

    A directory whose resolved siblings disagree yields nothing — an
    ambiguous directory is not evidence, and guessing here would be
    exactly the "confident and wrong" failure the basename index made.
    """
    by_dir: dict[str, set[str]] = {}
    for sp in specs:
        if sp.profile and sp.profile_source:
            by_dir.setdefault(str(Path(sp.tape).parent), set()).add(sp.profile)
    return {d: next(iter(v)) for d, v in by_dir.items() if len(v) == 1}


def resolve_from_siblings(spec: TapeSpec, sibs: dict[str, str],
                          root: Path = REPO) -> TapeSpec:
    if spec.profile and (root / spec.profile).exists():
        return spec
    hit = sibs.get(str(Path(spec.tape).parent))
    if hit and (root / hit).exists():
        spec.profile = hit
        spec.profile_source = "recovered from sibling tapes in the same solve"
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


def verify_ram_trace(
    rams: Sequence[Any],
    spec: TapeSpec,
    is_clear: Callable[[Any], bool],
) -> Verdict:
    """PURE: does this RAM trace reach a clear? Emulator-free, so testable.

    Kept separate from the replay itself because the interesting logic is
    here and the stepping is not. `rams` is the trace
    `src/training/tape_replay.replay_tape` returns — that module is
    documented as "the repo's ONE banked-tape replay convention", so this
    sweep consumes it rather than driving a Pool by hand. An earlier draft
    of this file did drive a Pool by hand; re-deriving a convention that
    already exists is waste, and it would also have diverged from the
    frame_skip / hw_flags lineage `TapePlayer` enforces.

    The verdict is taken at the FIRST clearing frame. A tape that clears
    and then keeps stepping can leave the cleared state — scoring the
    final frame instead would report a false negative on a good tape.
    """
    if not rams:
        return Verdict(spec.tape, "FAIL", "empty trace", 0)

    # A predicate already true at the ROOT verifies nothing. 111 of this
    # sweep's first 171 passes were "cleared at frame 0" — the tape had
    # not acted yet, so the only thing demonstrated was that its recorded
    # start_wd does not describe the state it is rooted at. Counting those
    # as PASS would have reported a corpus as verified on a 65% false
    # rate, which is worse than not sweeping at all.
    #
    # This is UNSCORABLE rather than FAIL, per the brief: the tape may be
    # perfectly good and its provenance merely mislabelled. Refusing to
    # score it is honest; calling it broken is not.
    # A tape whose own sidecar records clear_wd == start_wd was banked
    # under a predicate THIS verifier does not implement. SMB finales are
    # the live case: 8-4's ending never advances world/level — the game
    # sets opermode ($0770==2), which the level-advance predicate cannot
    # see. Scoring that FAIL would quarantine a sound tape for the
    # verifier's own blindness.
    if spec.clear_wd is not None and spec.start_wd is not None \
            and list(spec.clear_wd) == list(spec.start_wd):
        return Verdict(spec.tape, "UNSCORABLE",
                       "banked under a predicate this verifier does not "
                       "implement (clear_wd == start_wd; e.g. a finale "
                       "detected via opermode, not level advance)", 0)

    if is_clear(rams[0]):
        return Verdict(
            spec.tape, "UNSCORABLE",
            "clear predicate already satisfied at the root, before the "
            "tape acts — start_wd does not describe this root state", 0)

    for i, ram in enumerate(rams[1:], start=1):
        if is_clear(ram):
            return Verdict(spec.tape, "PASS", f"cleared at frame {i}", i)
    n = len(rams)
    return Verdict(spec.tape, "FAIL",
                   f"trace of {n} frames never satisfied is_clear", n)


def evaluate_gate(verdicts: Sequence[Verdict],
                  known_bad: Sequence[str] = KNOWN_BAD) -> tuple[bool, str]:
    """The pre-registered gate. ERROR counts as failure, never as a skip."""
    kb = set(known_bad)
    unscorable = [v for v in verdicts if v.status == "UNSCORABLE"]
    failed = [v for v in verdicts if v.status in ("FAIL", "ERROR")]
    new = [v for v in failed if v.tape not in kb]
    known_hit = [v for v in failed if v.tape in kb]
    passed = len(verdicts) - len(failed) - len(unscorable)
    msg = (f"{passed}/{len(verdicts)} replay; "
           f"{len(unscorable)} UNSCORABLE (root already satisfies the "
           f"predicate); "
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

    # Lazy, per scripts/hazard_collect.py: importing the emulator at
    # module scope would make every function above untestable without a ROM.
    # scripts/ is not the import root; the repo is (matches every other
    # script here that imports src.training.*).
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import yaml
    from src.training.tape_replay import (
        TapePlayer, load_root, machine_from_profile)
    from scripts.go_explore_solve import make_game
    import numpy as np

    index = build_consumer_index()
    specs = [resolve_profile(s_, index) for s_ in specs]
    specs = [resolve_from_siblings(s_, sibling_profiles(specs))
             for s_ in specs]

    verdicts: list[Verdict] = []
    for spec in specs:
        problems = spec_problems(spec)
        if problems:
            verdicts.append(Verdict(spec.tape, "ERROR", "; ".join(problems)))
            continue
        try:
            profile = yaml.safe_load((REPO / spec.profile).read_text())
            # Some solve profiles carry no rom key at all, which aborted
            # the whole sweep partway with "pass --rom explicitly".
            # The tape records the ROM it was made against; fall back to it.
            rom_hint = ((profile or {}).get("rom_path")
                        or (json.loads((REPO / spec.tape).read_text())
                            .get("solver_args") or {}).get("rom"))
            rom, frame_skip, bitmasks, hw_flags = machine_from_profile(
                profile, rom=rom_hint)
            # The tape records the hw flags it was made under. Replaying
            # under different flags diverges the trace and produces a
            # false FAIL — three Castlevania tapes failed exactly this way
            # while same-family tapes with matching lineage passed. The
            # tape's own lineage wins; a tape with no recorded lineage
            # replays under the profile's.
            rec_hw = (json.loads((REPO / spec.tape).read_text())
                      .get("hw") or {}).get("hw_flags")
            if rec_hw is None:
                # Older tapes carry no lineage of their own, but their
                # ROOT often does: entrance blobs ship a .state.json
                # sidecar recording the hw flags the machine ran with.
                # Three Castlevania tapes failed as "never satisfied
                # is_clear" purely because their root was built under
                # ['reset_alignment','mmio_read_timing','dmc_stall_timing',
                # 'nmi_poll_timing'] and the replay ran under the
                # profile's defaults — the trace diverges from frame one.
                side = REPO / (spec.root_state + ".json")
                if side.exists():
                    try:
                        rec_hw = (json.loads(side.read_text())
                                  .get("hw_flags"))
                    except (OSError, json.JSONDecodeError):
                        rec_hw = None
            if rec_hw is not None:
                hw_flags = tuple(rec_hw)
            game = make_game(profile)
            root = load_root(REPO / spec.root_state, hw_flags)
            actions = np.load(REPO / spec.actions, allow_pickle=False)
            start_key = tuple(spec.start_wd) if spec.start_wd else None
            with TapePlayer(rom=rom, bitmasks=bitmasks,
                            frame_skip=frame_skip, hw_flags=hw_flags) as pl:
                rams = [ram for _step, ram in pl.play(root, actions)]
            if start_key is None:
                verdicts.append(Verdict(spec.tape, "ERROR",
                                        "start_wd not recorded"))
                continue
            verdicts.append(verify_ram_trace(
                rams, spec, lambda ram: bool(game.is_clear(start_key, ram))))
        except BaseException as e:   # noqa: BLE001 — see below
            # BaseException, not Exception: machine_from_profile aborts
            # with SystemExit when a profile carries no rom key, and
            # SystemExit is not an Exception. Catching only Exception let
            # a single unusable tape terminate a 298-tape sweep partway,
            # which is why two consecutive full sweeps produced no report
            # at all. One bad input is one ERROR row, never the end of the
            # run. KeyboardInterrupt is re-raised so the sweep stays
            # interruptible.
            if isinstance(e, KeyboardInterrupt):
                raise
            verdicts.append(Verdict(spec.tape, "ERROR",
                                    f"{type(e).__name__}: {e}"[:160]))
        print(f"  {verdicts[-1].status:5s} {verdicts[-1].tape}"
              f"  {verdicts[-1].reason[:70]}", flush=True)

    report = build_report(verdicts, stamp=args.stamp,
                          core_sha16=sha256_16(
                              REPO / ".venv/lib/python3.11/site-packages/"
                              "nes_core.abi3.so"))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{report['gate_message']}")
    print(f"gate_passed={report['gate_passed']}  report={out}")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
