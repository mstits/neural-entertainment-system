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

TWO LAYERS, TWO QUESTIONS. `clear_reachability` answers the hook-level one
above. `clear_quorum` (added 2026-08-26, second half of this file) answers
it PER SIGNAL — which of a profile's signals can fire, which cannot and by
what mechanism, and whether what remains can reach the bar. That is the
distinction between VOID and FAIL, and it is the one the 41-profile
adjudication was missing: a hook can be REACHABLE in aggregate while the
specific signal a receipt depended on was dead the whole time. Its verdict
is UNREACHABLE rather than UNFIREABLE, deliberately — different vocabulary
for a different question, so a receipt can never blur the two.

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


class DetectorConstantMissing(RuntimeError):
    """A constant this module reads out of clear_detect.py has gone.

    Raised rather than defaulted. A guard that silently substitutes a
    plausible number when it loses sight of the real one is exactly the
    vacuous-check pattern this module exists to end: it would keep
    reporting PASS long after the thing it checks stopped existing."""


class CoordConstantMissing(DetectorConstantMissing):
    """`COORD_RESET_DROP_MIN` was renamed or removed from clear_detect."""


def _detect_const(name: str, exc: type[RuntimeError] | None = None):
    """One module-level constant read out of clear_detect.py by AST.

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
            if isinstance(tgt, ast.Name) and tgt.id == name:
                return ast.literal_eval(node.value)
    raise (exc or DetectorConstantMissing)(
        f"scripts/clear_detect.py no longer defines a module-level {name}; "
        "scripts/clear_reachability.py reads it to decide what the clear "
        "vote can reach and refuses to guess.")


def _coord_drop_min() -> int:
    """`COORD_RESET_DROP_MIN` read from clear_detect.py, never copied.

    The arithmetic rule below is "a single RAM byte tops out at 255, which
    is less than the drop `coord` demands". Hardcoding 300 here would let
    the two drift the moment somebody retunes the detector, and the guard
    would then pass a profile that had silently become unfireable — the
    precise failure it exists to catch."""
    return int(_detect_const("COORD_RESET_DROP_MIN", CoordConstantMissing))


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


# ===========================================================================
# THE ELIGIBILITY-GATED QUORUM
# ===========================================================================
#
# WHY A SECOND LAYER. `clear_reachability` above answers one question about
# a whole hook: can ANY declared machinery become True? That is the right
# question for "is this profile blind", and it is the wrong question for
# "did this measurement mean anything", because a hook can be REACHABLE in
# the aggregate while the specific signal a receipt depended on was dead the
# whole time.
#
# The 41-profile adjudication closed 4 CONFIRMED / 41 VOID / 0 FAIL. FAIL is
# exactly zero because not one gap profile was ever measured by an instrument
# demonstrated capable of returning a positive on that profile. Every null
# was a VOID wearing a FAIL's clothes, and two receipts in this repo show the
# mechanism by which that happens:
#
#   * runs/clear_control_2026-08-26/cv_odometer_swap.json — one witnessed
#     Castlevania clear, one detector, ONE key changed
#     (progress -> {source: odometer}). Hits 3/3 -> 0/3, coord checks
#     4/22 -> 0/22, largest single-step drop 592 -> 4 against the >= 300
#     that `coord` requires. The detector constructed happily and returned
#     a silent miss on a clear it had just found under the other arm.
#   * runs/clear_control_2026-08-26/bb_offline_r99.json — two Bubble Bobble
#     rows whose progress observable spans ONE unit, summarised as
#     `n_valid: 2, hit_rate: 0.0, hit_rate_pass: false`. A FAIL-shaped
#     number manufactured from a VOID-shaped measurement, inside the very
#     instrument built to end that confusion.
#
# So this section resolves, per signal and before the first observation:
# which signals CAN fire for this profile, which cannot and why, and whether
# what remains can reach the bar. A profile whose remaining signals cannot
# reach the bar is UNREACHABLE — not a miss, not a zero, and never a
# denominator.
#
# WHICH RULES ARE LIVE HERE AND WHICH ARE RECORDED ONLY. The vote redesign
# names five rules. Three are implemented as arithmetic below; two are
# deliberately metadata, and saying which is which is the point:
#
#   RULE 1 ADMISSIBILITY  — LIVE. Every signal is classified ALIVE / DEAD /
#     DEGENERATE / NOT_WIRED from the profile and from this repo's own code.
#     No rule rests on knowing anything about a game.
#   RULE 2 SEPARATION     — LIVE, and inert by default on purpose. A signal
#     whose MEASURED null fire-rate is at or above MAX_NULL_RATE is
#     DEGENERATE and casts nothing. `tally` fired on 22/22, 28/28 and 43/43
#     Castlevania checks and 30/30 Bubble Bobble checks, and the old ceiling
#     still counted it as a full vote. But no profile carries a measured
#     null today (scripts/clear_calibrate.py does not exist), and inventing
#     one from a hunch is the COORD_RESET_DROP_MIN mistake again. So absent
#     a measurement a signal is ALIVE-but-unseparated, and the receipt says
#     `null_rate: null` rather than a number somebody guessed.
#   RULE 3 SLOTS          — RECORDED, NOT ARITHMETIC. Every signal carries
#     its slot and the table prints it, but the ceiling is still a sum over
#     eligible signals. Reason: exactly one slot has two WIRED members today
#     (S_CADENCE = {tally, apu}) and the shipped vote gives each a full
#     vote, so collapsing the slot now would make this module's ceiling
#     SMALLER than what StreamingConfluenceDetector.push can actually reach
#     — it would refuse profiles that can fire. A ceiling that is not an
#     upper bound is worse than no ceiling. The moment a second member of
#     any other slot is wired, the sum stops being an upper bound in the
#     other direction and the arithmetic MUST become a slot-min;
#     test_a_second_wired_member_in_a_slot_forces_the_slot_arithmetic is the
#     tripwire that fails when that day arrives.
#   RULE 4 QUORUM FRACTION — NOT IMPLEMENTED. It is a function of the slot
#     count, so it waits on Rule 3.
#   RULE 5 REQUIRED CLASS — LIVE. At least one eligible signal must be
#     TRANSITION EVIDENCE ("a scene committed" / "the world did not come
#     back"). Corroborators alone cannot reach quorum however many of them
#     agree. This is the structural form of the Bubble Bobble frame-320
#     false positive: audio + tally + lock summed to exactly THRESHOLD with
#     coord = 0, 1736 frames before the true clear.
#
# THE SIX SHELF SIGNALS ARE REPORTED, NOT WIRED. entity_wipe,
# room_fp_transition, input_lock, lock_release_novelty, oam_quiesce and
# scene_cut were built and tested on 2026-08-26 and reach no production
# path. They appear in every table as NOT_WIRED with that reason attached,
# because "six signals wired to nothing" is a fact about this instrument's
# ceiling and belongs on the instrument's own readout rather than in a
# document nobody opens.

FIREABLE = "FIREABLE"
UNREACHABLE = "UNREACHABLE"

# ---------------------------------------------------------------------------
# THE FOUR-VALUED CLEAR VERDICT, replacing the bool everywhere a clear result
# is PRODUCED or RECORDED. (The bool contract at is_clear() is untouched --
# 43+ duck-typed call sites -- so the verdict rides in the receipts.)
#
#   CLEAR         quorum met, vetoes clean.
#   NO_CLEAR      quorum was REACHABLE, was evaluated, and was not met. The
#                 only value that carries a negative bit.
#   UNREACHABLE   the eligible signals cannot reach quorum for this profile,
#                 established before the first observation. Not a miss, not a
#                 zero, and never a denominator.
#   UNDER_WARMUP  the detector was not fed warmup_observations(), so its
#                 silence is a structural no-op rather than a result. The
#                 number already exists (StreamingConfluenceDetector.
#                 warmup_observations, GenericGame.clear_observation_budget);
#                 it just had no name in any output.
#   ERROR         the row never ran.
# ---------------------------------------------------------------------------
CLEAR = "CLEAR"
NO_CLEAR = "NO_CLEAR"
UNDER_WARMUP = "UNDER_WARMUP"
ERROR = "ERROR"

#: Rows that measured something. The rule this whole module exists for: an
#: instrument that could not have said yes contributes no denominator.
MEASURED_VERDICTS = (CLEAR, NO_CLEAR)

ALIVE = "ALIVE"
DEAD = "DEAD"
NOT_WIRED = "NOT_WIRED"
#: DEGENERATE is reused from the hook-level vocabulary above: machinery that
#: is trivially true fabricates clears rather than missing them.

#: A signal whose measured null fire-rate reaches this is DEGENERATE: it
#: carries no bits, so it cannot corroborate anything. Only ever applied to
#: a MEASURED rate — see Rule 2 above.
MAX_NULL_RATE = 0.05

#: Which question each signal answers. Correlated signals share a slot;
#: recorded today, arithmetic once a second member of a non-cadence slot is
#: wired (Rule 3).
SLOTS = {
    "S_TRANSITION": "a scene/room committed",
    "S_DESPAWN": "something emptied",
    "S_CADENCE": "the presentation changed",
    "S_IRREVERSIBLE": "the world did not come back",
    # Not a vote of its own, by its own docstring: InputLockSignal "must
    # never be the sole term in a clear vote". It arms tier-2 probing and
    # corroborates a transition somebody else observed. Its weight IS
    # counted in the offline ceiling, because clear_detect.run_episode
    # really does add 0.25 for it -- a ceiling has to bound what the vote
    # can reach, not what it ought to reach. Taking that vote away is
    # downstream work, and the frame-320 receipt is the argument for it.
    "S_ARMING": "an arming condition, never a vote of its own",
}

#: Which detector each roster describes. `live` is the solver's per-step
#: hook (StreamingConfluenceDetector: RAM only, plus the APU mask when a
#: profile arms it). `offline` is the four-signal weighted harness in
#: clear_detect.run_episode, which additionally has an audio stream and an
#: env handle for the differential input probe.
LIVE = "live"
OFFLINE = "offline"


@dataclass(frozen=True)
class SignalSpec:
    """What one signal is, independent of any profile.

    `weight` is the most it can contribute to its roster's vote. `wired`
    is whether it reaches a production path at all. `transition_evidence`
    is whether it can satisfy Rule 5 — deliberately False for `lock` and
    `input_lock`, which their own docstrings say must never be the sole
    term in a clear vote: they are arming conditions and corroborators,
    and the Bubble Bobble false positive counted one as a full vote."""

    name: str
    slot: str
    weight: float
    wired: bool
    transition_evidence: bool
    note: str = ""


_SHELF_NOTE = ("built and tested 2026-08-26, reaches no production path "
               "(the live vote is tally+coord+apu; the offline harness "
               "weights audio+tally+lock+coord)")

#: The six signals on the shelf. Present in every table so the ceiling
#: reads honestly, absent from every vote.
SHELF_SPECS = (
    SignalSpec("scene_cut", "S_TRANSITION", 0.0, False, True, _SHELF_NOTE),
    SignalSpec("room_fp_transition", "S_TRANSITION", 0.0, False, True,
               _SHELF_NOTE),
    SignalSpec("input_lock", "S_ARMING", 0.0, False, False,
               _SHELF_NOTE + "; arming condition only, never a sole term"),
    SignalSpec("oam_quiesce", "S_DESPAWN", 0.0, False, False, _SHELF_NOTE),
    SignalSpec("entity_wipe", "S_DESPAWN", 0.0, False, False, _SHELF_NOTE),
    SignalSpec("lock_release_novelty", "S_IRREVERSIBLE", 0.0, False, True,
               _SHELF_NOTE),
)


def _roster(roster: str, apu_weight: float) -> tuple[SignalSpec, ...]:
    """The signal set one detector really votes on, plus the shelf.

    Weights are read from clear_detect rather than copied: the offline
    harness fuses four signals at WEIGHTS against THRESHOLD, and a retune
    there must move this ceiling with it or the ceiling starts lying."""
    if roster == OFFLINE:
        w = _detect_const("WEIGHTS")
        return (
            SignalSpec("coord", "S_TRANSITION", float(w["coord"]), True, True),
            SignalSpec("tally", "S_CADENCE", float(w["tally"]), True, False),
            SignalSpec("audio", "S_CADENCE", float(w["audio"]), True, False),
            SignalSpec("lock", "S_ARMING", float(w["lock"]), True, False,
                       "differential input probe; corroborator only, never "
                       "a sole term"),
        ) + SHELF_SPECS
    return (
        SignalSpec("coord", "S_TRANSITION", 1.0, True, True),
        SignalSpec("tally", "S_CADENCE", 1.0, True, False),
        SignalSpec("apu", "S_CADENCE", float(apu_weight), True, False),
    ) + SHELF_SPECS


@dataclass(frozen=True)
class SignalState:
    """One signal's admissibility for one profile, with the reason.

    The reason is never "unknown". Every DEAD/DEGENERATE string names the
    mechanism — a file and a symbol from this repo, or arithmetic — so a
    receipt carrying this table carries why the instrument could not have
    said yes."""

    name: str
    state: str
    slot: str
    weight: float
    reason: str
    transition_evidence: bool = False
    null_rate: float | None = None

    @property
    def eligible(self) -> bool:
        return self.state == ALIVE


@dataclass(frozen=True)
class Quorum:
    """Can this profile's clear vote reach its bar, and on what.

    `verdict` is FIREABLE or UNREACHABLE. UNREACHABLE is the value the
    roster never had: it is not a miss, it does not go in a denominator,
    and it cannot be cited as "searched and found none"."""

    verdict: str
    roster: str
    required: float
    ceiling: float
    reason: str
    signal_state: dict

    @property
    def ok(self) -> bool:
        return self.verdict == FIREABLE

    @property
    def eligible(self) -> tuple[str, ...]:
        return tuple(n for n, s in self.signal_state.items() if s.eligible)

    @property
    def slots(self) -> dict:
        """slot -> the eligible signals in it. Every slot appears, empty
        ones included: a slot that lost its last member is exactly the
        thing that must be visible."""
        out = {k: [] for k in SLOTS}
        for name, st in self.signal_state.items():
            if st.eligible:
                out.setdefault(st.slot, []).append(name)
        return out

    @property
    def eligible_slots(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.slots.items() if v)

    def as_dict(self) -> dict:
        """Receipt form. Every writer of a clear result carries this."""
        return {
            "verdict": self.verdict,
            "roster": self.roster,
            "required": self.required,
            "ceiling": self.ceiling,
            "reason": self.reason,
            "eligible": list(self.eligible),
            "eligible_slots": list(self.eligible_slots),
            "signals": {
                n: {"state": s.state, "slot": s.slot, "weight": s.weight,
                    "null_rate": s.null_rate, "reason": s.reason}
                for n, s in self.signal_state.items()},
        }

    def table(self) -> str:
        """The per-signal eligibility table, for stdout at launch.

        The Gradius failure was not that the 2026-08-24 progress swap was
        undetectable. It is that nothing ever printed "coord is now DEAD,
        ceiling 1 of 2" while 54+ minutes a day burned. A table on stdout
        at every launch is the cheapest possible tripwire."""
        lines = [f"[clear] quorum {self.verdict} — roster={self.roster} "
                 f"ceiling={self.ceiling:g} required={self.required:g}"]
        for name, s in self.signal_state.items():
            rate = "-" if s.null_rate is None else f"{s.null_rate:.2f}"
            lines.append(
                f"[clear]   {name:<20} {s.state:<10} {s.slot:<14} "
                f"w={s.weight:<4g} null={rate:<5} {s.reason}")
        lines.append(f"[clear] {self.reason}")
        return "\n".join(lines)


def _null_rates(solve: dict, override: dict | None) -> dict:
    """Measured null fire-rates for this profile, from the profile then the
    caller. Never invented: a signal with no entry here has an UNMEASURED
    null, which is reported as such and is not grounds for refusal."""
    out = {}
    declared = solve.get("null_rates")
    if isinstance(declared, dict):
        out.update(declared)
    if override:
        out.update(override)
    clean = {}
    for k, v in out.items():
        try:
            clean[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return clean


def _classify(spec: SignalSpec, solve: dict, rates: dict) -> SignalState:
    """One signal's state for one profile. Rule 1, then Rule 2."""
    rate = rates.get(spec.name)
    common = dict(name=spec.name, slot=spec.slot, weight=spec.weight,
                  transition_evidence=spec.transition_evidence,
                  null_rate=rate)
    if not spec.wired:
        return SignalState(state=NOT_WIRED,
                           reason=spec.note or "not wired", **common)
    if spec.name == "apu" and spec.weight <= 0:
        return SignalState(
            state=DEAD,
            reason=("clear.apu_weight is 0, so no ApuActivitySignal is "
                    "constructed and the vote is the shipped integer path"),
            **common)
    why = ""
    if spec.name == "coord":
        alive, why = _coord_status(solve)
        if not alive:
            return SignalState(state=DEAD, reason=why, **common)
    elif spec.note:
        why = spec.note
    if rate is not None and rate >= MAX_NULL_RATE:
        return SignalState(
            state=DEGENERATE,
            reason=(f"measured null fire-rate {rate:.2f} >= "
                    f"{MAX_NULL_RATE:g}: it fires on ordinary play, so it "
                    "carries no bits and cannot corroborate anything"),
            **common)
    measured = ("null fire-rate UNMEASURED (scripts/clear_calibrate.py does "
                "not exist) — alive is not the same as separating")
    return SignalState(state=ALIVE,
                       reason=f"{why}; {measured}" if why else measured,
                       **common)


def clear_quorum(profile: dict, null_rates: dict | None = None,
                 roster: str = LIVE) -> Quorum:
    """Per-signal eligibility and the bar, for one profile dict.

    The question is "can the CONFLUENCE vote reach its bar for this
    profile", asked whether or not the profile DECLARES that hook —
    because the instrument asks it either way. clear_detect's offline
    harness fuses the same four signals over any profile's trace, which is
    how two Bubble Bobble rows (a byte_change profile) came to be scored by
    a detector whose `coord` half was arithmetically dead. The hook-level
    question — "does this profile declare machinery that can fire at all"
    — is `clear_reachability` above, and it is the one that consults
    clear.mode.

    No ROM, no emulator, no game knowledge — same constraint as
    `clear_reachability`, and for the same reason: a refusal that rests on
    a hunch about a title is its own defect.

    `null_rates` are MEASURED fire-rates over that profile's own ordinary
    play (Rule 2). They are read from `solve.null_rates` and may be
    overridden by the caller; a signal with no measurement is reported
    ALIVE-but-unseparated rather than guessed at in either direction."""
    solve = profile.get("solve")
    solve = solve if isinstance(solve, dict) else {}
    clear = solve.get("clear")
    clear = clear if isinstance(clear, dict) else {}
    rates = _null_rates(solve, null_rates)

    try:
        apu_w = float(clear.get("apu_weight", 0.0) or 0.0)
    except (TypeError, ValueError):
        apu_w = 0.0

    if roster == OFFLINE:
        required = float(_detect_const("THRESHOLD"))
        bar_src = "clear_detect.THRESHOLD"
    else:
        raw = clear.get("min_signals")
        required = 2.0 if raw is None else float(raw)
        bar_src = "clear.min_signals"

    states = {}
    for spec in _roster(roster, apu_w):
        states[spec.name] = _classify(spec, solve, rates)

    ceiling = sum(s.weight for s in states.values() if s.eligible)
    dead_why = "; ".join(
        f"{n}={s.state} ({s.reason})" for n, s in states.items()
        if s.state in (DEAD, DEGENERATE))

    if ceiling + 1e-9 < required:
        return Quorum(
            UNREACHABLE, roster, required, ceiling,
            f"UNREACHABLE: the eligible signals top out at {ceiling:g} "
            f"against {bar_src}={required:g}, so no observation sequence "
            f"can make this hook true. {dead_why}. A null from this "
            "configuration is VOID, not a miss: it must not enter a hit-rate "
            "denominator and must never be cited as \"searched and found "
            "none\".",
            states)

    has_transition = any(
        s.eligible and s.transition_evidence for s in states.values())
    if not has_transition:
        corroborators = ", ".join(sorted(sn for sn, s in states.items()
                                         if s.eligible))
        return Quorum(
            UNREACHABLE, roster, required, ceiling,
            "UNREACHABLE: no eligible signal is TRANSITION EVIDENCE, so the "
            f"bar can only be reached by corroborators ({corroborators}) "
            "agreeing with each other about a scene change none of them "
            "observed. Measured shape of that failure: "
            "runs/clear_control_2026-08-26/bb_offline_r99.json fired at "
            "frame 320 on audio+tally+lock with coord=0, 1736 frames before "
            f"the true clear at 2056. {dead_why}",
            states)

    return Quorum(
        FIREABLE, roster, required, ceiling,
        f"FIREABLE: eligible signals can cast up to {ceiling:g} against "
        f"{bar_src}={required:g}, with transition evidence among them. "
        "Whether the vote ever DOES fire on a real clear is an empirical "
        "question this module does not touch — FIREABLE is not evidence "
        "that it works.",
        states)


def _confluence(solve: dict, clear: dict) -> Reachability:
    """The confluence hook's verdict, delegated to `clear_quorum`.

    This used to compute its own ceiling as `1.0 + coord + apu` — `tally`
    an unconditional vote with no separation test anywhere in the tree,
    which is how a signal that fired on 22/22, 28/28, 43/43 Castlevania
    checks and 30/30 Bubble Bobble checks kept counting as a full
    corroborator. There is now exactly one adjudicator and both layers ask
    it, so the hook-level verdict and the per-signal table can never
    disagree about what this profile can reach."""
    q = clear_quorum({"solve": solve})
    if q.ok:
        return Reachability(REACHABLE, "confluence", q.reason)
    return Reachability(UNFIREABLE, "confluence", q.reason)


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
    where = f" [{profile_path}]" if profile_path else ""
    if r.verdict == NONE:
        return (f"[clear]{where} NO REACHABLE CLEAR PREDICATE — {r.reason} "
                "Frontier depth and cell counts from this run are real "
                "measurements; the solution count is not. Cite it only as "
                "\"no clear predicate wired\", never as \"searched and "
                "found none\".")
    if r.via == "confluence":
        # ENFORCEMENT POINT 3. The ceiling is a number on stdout at every
        # launch, so a silent disarm is structurally impossible. Gradius
        # ran eighteen days at a ceiling of 1 because nothing ever printed
        # the table; printing it costs one screenful and no emulator
        # seconds.
        return f"[clear]{where}\n" + clear_quorum(profile).table()
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
