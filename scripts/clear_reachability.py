"""Refuse a solve profile that declares clear machinery which cannot fire.

WHY THIS EXISTS. `GenericGame.is_clear` opens with

    if self.level_key(ram) > tuple(start_key):

and 40 of the 45 profiles carrying a `solve:` block ship `level_key: []`
(two more omit the key entirely). For those, `level_key(ram)` returns `()`
and the test is `() > ()`, which is False in Python for every RAM state
that has ever existed or ever will. The branch is an algebraic identity,
not a measurement.

That is a deliberate, documented choice — "coverage baseline", the purity
line refusing to guess a stage byte nobody has watched advance. The defect
is not the empty key. The defect is that nothing downstream ever said so,
so `solutions: 0` in a progress.jsonl was read, for months and across ten
documents, as *the search looked for a win and did not find one* when it
actually meant *no question was ever asked*. Compute totals were multiplied
against that constant to make it look like corroboration.

The 2026-08-26 clear-detection census (docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md)
ran 29 profiles and returned 0 predicates confirmed. A 29-for-29 null is
also exactly what a broken instrument produces, and the census's own
positive controls never ran, so it could not tell the two apart.

WHAT THIS MODULE DOES. It answers one question about a profile dict, with
no ROM, no emulator, and no game knowledge:

    can this profile's declared clear machinery become True at all?

Three answers matter:

  REACHABLE  — some declared hook can fire. Whether it *does* is an
               empirical question this module does not touch.
  NONE       — no clear machinery is declared. Legitimate and common (the
               coverage baseline), but it means `solutions` is a compile-
               time 0 and `--want-solutions` is inert. Callers must say so
               out loud rather than let a reader infer a search result.
  UNFIREABLE — machinery IS declared and provably cannot fire. This is the
               refusal. A run under such a profile advertises a capability
               it does not have and burns its full wall-clock budget doing
               it (Gradius: 54+ min that day, and every minute after the
               2026-08-24 progress swap).
  DEGENERATE — machinery IS declared and is trivially true, which
               fabricates clears rather than missing them. Same refusal,
               opposite direction.

THE PURITY LINE (CLAIMS.md Tier 3) CONSTRAINS THIS FILE HARD. Every rule
below is derived from the repo's own code and from arithmetic. None of them
rests on knowing anything about a game. Specifically, this module does NOT
claim "game X has no end-of-level timer-to-score tally" — that is a fact
about a title, it can only be established by measuring, and asserting it
from memory is exactly the class of reasoning the purity line forbids. So a
profile whose `coord` half is alive is reported REACHABLE-but-unproven, not
refused, even when a human is fairly sure it will never fire. Refusing on a
hunch is its own defect.

WHAT WOULD THIS REPORT IF THE MECHANISM WERE ABSENT? The check is written
so a caller can answer that. `clear_reachability` on a profile with a live
`level_key` returns REACHABLE; delete the key and it returns NONE; wire a
confluence hook onto an odometer and it returns UNFIREABLE. Three inputs,
three different answers — the function is not a constant, which is the
property the guards it replaces did not have. tests/test_clear_reachability.py
asserts exactly that, including a mutation test that breaks each rule and
requires the verdict to move.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


class CoordConstantMissing(RuntimeError):
    """`COORD_RESET_DROP_MIN` was renamed or removed from clear_detect.

    Raised rather than defaulted. A guard that silently substitutes a
    plausible number when it loses sight of the real one is exactly the
    vacuous-check pattern this module exists to end: it would keep
    reporting PASS long after the thing it checks stopped existing."""


def _coord_drop_min() -> int:
    """`COORD_RESET_DROP_MIN` read from clear_detect.py, never copied.

    The arithmetic rule below is "a single RAM byte tops out at 255, which
    is less than the drop `coord` demands". Hardcoding 300 here would let
    the two drift the moment somebody retunes the detector, and the guard
    would then pass a profile that had silently become unfireable — the
    precise failure it exists to catch.

    Read by AST rather than by import on purpose: `import clear_detect`
    transitively imports `go_explore_solve`, which imports the compiled
    `nes_core` extension. A profile lint must be runnable in a bare
    checkout with no Rust build, and a guard nobody can run is a guard
    nobody runs. The AST walk reads the same single assignment the import
    would have bound, and tests/test_clear_reachability.py asserts the two
    agree whenever nes_core IS available."""
    import ast
    src = (REPO / "scripts" / "clear_detect.py").read_text()
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "COORD_RESET_DROP_MIN":
                return int(ast.literal_eval(node.value))
    raise CoordConstantMissing(
        "scripts/clear_detect.py no longer defines a module-level "
        "COORD_RESET_DROP_MIN; scripts/clear_reachability.py reads it to "
        "decide whether the `coord` vote can fire and refuses to guess.")


#: Streaming vote weight of the two signals `StreamingConfluenceDetector`
#: can actually derive from a RAM snapshot. The offline four-signal
#: detector also fuses `audio` and `lock`, but neither survives into the
#: live solver hook: there is no audio stream and no env handle for the
#: differential input probe (clear_detect.StreamingConfluenceDetector's own
#: "Availability note"). So the live ceiling is 2, plus `apu_weight` when a
#: profile opts the third signal in.
LIVE_CONFLUENCE_SIGNALS = 2

#: A single unsigned RAM byte.
MAX_RAM_BYTE = 255

REACHABLE = "REACHABLE"
NONE = "NONE"
UNFIREABLE = "UNFIREABLE"
DEGENERATE = "DEGENERATE"

#: Verdicts a solve run may proceed under. NONE is here on purpose: a
#: coverage baseline is a legitimate configuration, it just has to be
#: announced instead of inferred.
OK_VERDICTS = (REACHABLE, NONE)


@dataclass(frozen=True)
class Reachability:
    """`verdict` plus the evidence for it.

    `via` names the hook that can fire (REACHABLE) or the hook that cannot
    (UNFIREABLE / DEGENERATE). `reason` is written to be pasted into a
    receipt verbatim — the census's failure mode was a number travelling
    without its meaning, so the meaning travels attached."""

    verdict: str
    via: str | None
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict in OK_VERDICTS

    @property
    def can_bank_a_solution(self) -> bool:
        """True iff `progress.jsonl`'s `solutions` field is a measurement.

        When this is False the field is a compile-time constant 0 and must
        never be cited as evidence that a search tried and failed."""
        return self.verdict == REACHABLE


def _progress_block(solve: dict) -> dict:
    p = solve.get("progress")
    return dict(p) if isinstance(p, dict) else {}


def _coord_status(solve: dict) -> tuple[bool, str]:
    """Can the `coord` vote ever fire under this profile's progress source?

    `coord` (clear_detect.coord_entity_windows) requires the progress
    readout to fall by at least COORD_RESET_DROP_MIN inside one window AND
    land at/below COORD_RESET_ABS_MAX. Two profile shapes make that
    impossible, both established from this repo's code rather than from
    anything about a game:

      odometer  — nes_core/src/ppu.rs `odo_fold_frame` DROPS the anchor on
                  a mostly-blank frame (< 120 rendered lines) and
                  re-anchors on the next rendered one instead of
                  integrating across the discontinuity. A stage wipe or a
                  death fade therefore FREEZES the integral rather than
                  rewinding it, and `Solver._xram` additionally clamps
                  backward-of-origin to 0. The series is monotone
                  non-decreasing within a burst, and the detector's ctx
                  (and with it its rolling window) is rebuilt from scratch
                  by `Solver._assign` on every state restore, so a restore
                  cannot smuggle a drop into the window either.

      fight_gate — a cumulative-damage integral accumulated by
                  `fight_gate_step`; monotone by construction, same
                  consequence.

      single RAM byte — `progress: {lo: N}` with no `hi` and no `tiles`
                  composes to at most 255, which is below the required
                  drop. Pure arithmetic.

    Everything else (a 16-bit `{lo, hi}` pair, a decoded `tiles` HUD field)
    is reported ALIVE. Alive is not "fires" — it is only "not provably
    dead", which is as far as the purity line lets this function go."""
    p = _progress_block(solve)
    src = str(p.get("source", "")).lower()
    if src == "odometer":
        return False, (
            "progress.source: odometer — the camera integral freezes and "
            "re-anchors across a scene cut (nes_core/src/ppu.rs "
            "odo_fold_frame) and clamps backward-of-origin to 0 "
            "(Solver._xram), so the backwards drop of "
            f">= {_coord_drop_min()} px that `coord` requires never appears")
    if src == "fight_gate":
        return False, (
            "progress.source: fight_gate — a cumulative-damage integral, "
            "monotone by construction, so `coord`'s required drop of "
            f">= {_coord_drop_min()} never appears")
    if p.get("tiles"):
        return True, "progress is a decoded multi-tile HUD field"
    if p.get("hi") is not None:
        return True, "progress is a 16-bit {lo, hi} pair"
    if p.get("lo") is not None:
        return False, (
            f"progress is the single RAM byte {int(p['lo'])!r} "
            f"(max {MAX_RAM_BYTE}), which cannot drop by the "
            f">= {_coord_drop_min()} that `coord` requires")
    return True, "progress shape not recognised — assumed alive, not refused"


def _confluence(solve: dict, clear: dict) -> Reachability:
    """The 2-of-2 vote, checked against how many votes CAN be cast.

    `StreamingConfluenceDetector` fires when `tally + coord + W*apu >=
    min_signals`. `tally` is a claim about a game's RAM behaviour and this
    module never adjudicates it — it is assumed castable. `coord` is a
    claim about the profile's own progress plumbing and IS adjudicable.
    So the ceiling is:

        max_vote = 1 (tally) + (1 if coord alive else 0) + (W if W > 0)

    and the hook is refused only when that ceiling is below `min_signals`.
    A profile that arms the APU vote to compensate for a dead `coord`
    therefore passes, which is the intended escape hatch and the reason
    this is not simply "odometer + confluence is banned"."""
    coord_alive, coord_why = _coord_status(solve)
    try:
        apu_w = float(clear.get("apu_weight", 0.0) or 0.0)
    except (TypeError, ValueError):
        apu_w = 0.0
    raw_min = clear.get("min_signals")
    min_signals = 2.0 if raw_min is None else float(raw_min)

    max_vote = 1.0 + (1.0 if coord_alive else 0.0) + (apu_w if apu_w > 0 else 0.0)
    if max_vote + 1e-9 < min_signals:
        dead = "coord is dead" if not coord_alive else "too few signals are armed"
        return Reachability(
            UNFIREABLE, "confluence",
            f"clear: {{mode: confluence}} needs min_signals={min_signals:g} "
            f"but at most {max_vote:g} vote(s) can ever be cast "
            f"(tally=1, coord={'1' if coord_alive else '0'}, "
            f"apu={apu_w:g}) — {dead}. {coord_why}. "
            "Remedy: arm a third signal (apu_weight), point `progress` at "
            "an observable that really does reset across the terminal "
            "transition, or drop the hook and say the profile has no clear "
            "predicate.")
    return Reachability(
        REACHABLE, "confluence",
        f"clear: {{mode: confluence}} can cast up to {max_vote:g} of "
        f"min_signals={min_signals:g} ({coord_why}). Never observed to "
        "fire on a real clear for any profile — REACHABLE is not evidence "
        "that it works.")


def _finale(solve: dict, finale: dict) -> Reachability | None:
    """`is_finale` is `start_key == tuple(f['level_key']) and ram[addr] == value`.

    An arity mismatch between the finale's `level_key` and the profile's
    own makes that equality False for every state — the entrance key has
    len(solve.level_key) entries and can never equal a literal of a
    different length. Same length with literal values is a legitimate
    "only from this level" gate and is left alone."""
    lk = solve.get("level_key")
    lk = list(lk) if isinstance(lk, (list, tuple)) else []
    f_lk = finale.get("level_key")
    f_lk = list(f_lk) if isinstance(f_lk, (list, tuple)) else []
    if "addr" not in finale or "value" not in finale:
        return Reachability(
            UNFIREABLE, "finale",
            "finale: block is missing addr and/or value, so is_finale() "
            "cannot be evaluated and returns False for every state")
    if len(f_lk) != len(lk):
        return Reachability(
            UNFIREABLE, "finale",
            f"finale.level_key has {len(f_lk)} entr(ies) but solve.level_key "
            f"has {len(lk)}; is_finale() compares them for equality, and "
            "tuples of different length are never equal, so the hook is "
            "False for every state")
    return Reachability(
        REACHABLE, "finale",
        f"finale: ram[0x{int(finale['addr']):04X}] == {int(finale['value'])} "
        f"with a {len(lk)}-entry level_key match")


def clear_reachability(profile: dict) -> Reachability:
    """Classify one profile dict's clear machinery. No I/O, no ROM, no game.

    Checked in the order `GenericGame.is_clear` itself checks: the
    level_key advance first (it short-circuits every other hook), then the
    configured WIN-CONDITION, then the separate `finale` hook."""
    solve = profile.get("solve")
    if not isinstance(solve, dict):
        return Reachability(
            NONE, None,
            "no `solve:` block — this profile does not use the generic "
            "solver path")

    lk = solve.get("level_key")
    lk = list(lk) if isinstance(lk, (list, tuple)) else []
    if lk:
        return Reachability(
            REACHABLE, "level_key",
            f"level_key has {len(lk)} byte(s) "
            f"({', '.join(f'0x{int(a):04X}' for a in lk)}), so "
            "`level_key(ram) > start_key` can become True")

    clear = solve.get("clear")
    clear = dict(clear) if isinstance(clear, dict) else {}
    finale = solve.get("finale")
    finale = dict(finale) if isinstance(finale, dict) else {}

    mode = clear.get("mode")
    if mode == "confluence":
        r = _confluence(solve, clear)
        if r.verdict != REACHABLE:
            return r
        return r
    if mode == "byte_change":
        if clear.get("addr") is None:
            return Reachability(
                UNFIREABLE, "byte_change",
                "clear: {mode: byte_change} declares no addr, so "
                "`self._clear_addr is None` and the branch returns False "
                "for every state")
        return Reachability(
            REACHABLE, "byte_change",
            f"clear: byte_change on 0x{int(clear['addr']):04X} vs the "
            "entrance baseline latched by note_start()")
    if mode == "score_jump":
        try:
            thr = float(clear.get("threshold", 0))
        except (TypeError, ValueError):
            thr = 0.0
        if thr <= 0:
            return Reachability(
                DEGENERATE, "score_jump",
                f"clear: {{mode: score_jump, threshold: {thr:g}}} fires "
                "whenever progress does not decrease — a predicate that is "
                "true almost every step fabricates clears rather than "
                "missing them. threshold must be > 0.")
        return Reachability(
            REACHABLE, "score_jump",
            f"clear: score_jump of >= {thr:g} on the progress observable")
    if mode:
        return Reachability(
            UNFIREABLE, str(mode),
            f"clear: {{mode: {mode!r}}} is not a mode GenericGame.is_clear "
            "implements, so no branch can return True")

    if finale:
        r = _finale(solve, finale)
        if r is not None:
            return r

    return Reachability(
        NONE, None,
        "level_key is the empty coverage baseline and no clear:/finale: "
        "hook is declared, so `level_key(ram) > start_key` is `() > ()` "
        "= False for every state. This run CANNOT bank a solution: "
        "`solutions: 0` is a compile-time constant, not a search result, "
        "and --want-solutions is inert.")


def launch_banner(profile: dict, profile_path: str | None = None) -> str | None:
    """The line a solve run prints before it spends an hour, or None.

    Returned rather than printed so the caller decides the stream, and so
    tests can assert on it without capturing stdout."""
    r = clear_reachability(profile)
    if r.verdict == NONE:
        where = f" [{profile_path}]" if profile_path else ""
        return (f"[clear]{where} NO REACHABLE CLEAR PREDICATE — {r.reason} "
                "Frontier depth and cell counts from this run are real "
                "measurements; the solution count is not. Cite it only as "
                "\"no clear predicate wired\", never as \"searched and "
                "found none\".")
    return None


def enforce(profile: dict, profile_path: str | None = None) -> Reachability:
    """Refuse a run whose declared clear machinery cannot fire.

    Raises SystemExit for UNFIREABLE/DEGENERATE — matching how every other
    profile defect in go_explore_solve aborts before the pool is built, so
    a bad profile costs zero emulator seconds. Returns the Reachability
    otherwise, including for NONE: a coverage baseline is allowed to run,
    it is just required to announce itself."""
    r = clear_reachability(profile)
    if not r.ok:
        where = f" {profile_path}" if profile_path else ""
        raise SystemExit(
            f"[clear] REFUSED{where}: {r.verdict} clear machinery "
            f"({r.via}). {r.reason}")
    return r


def _load(path: Path) -> dict:
    import yaml
    d = yaml.safe_load(path.read_text())
    return d if isinstance(d, dict) else {}


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profiles", nargs="*", help="profile YAMLs to check")
    ap.add_argument("--all", action="store_true",
                    help="check every top-level configs/*.yaml")
    ap.add_argument("--quiet", action="store_true",
                    help="print refusals only")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.profiles]
    if args.all or not paths:
        paths = sorted((REPO / "configs").glob("*.yaml"))

    tally: dict[str, int] = {}
    bad = 0
    for p in paths:
        prof = _load(p)
        if "solve" not in prof:
            continue
        r = clear_reachability(prof)
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
        if not r.ok:
            bad += 1
        if r.ok and args.quiet:
            continue
        print(f"{r.verdict:<10} {p.name:<36} {r.via or '-':<12} {r.reason}")

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    if bad:
        print(f"\nREFUSED {bad} profile(s): they declare clear machinery "
              f"that cannot fire.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
