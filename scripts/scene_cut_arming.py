"""Decide, by measurement, which odometer-cohort profiles may arm `scene_cut`.

THE ANSWER THIS SCRIPT ARRIVED AT (2026-08-27): **no cohort profile may
arm `scene_cut` today**, and the reason is C7 below -- not one of the
individual defects that prompted it. All 23 are disarmed with
`enabled: false` and a per-profile measured reason. The instrument
(`odo_blank`) is real and moves for 20 of the 25 profiles; what does not
exist is a WITNESSED positive on any of them to calibrate a gate
against. That is the same refusal, for the same reason, as
docs/receipts/rygar/clear_predicate_REFUTED.md.

WHY THIS EXISTS. On 2026-08-26 commit 27902e1 armed
`solve.clear.signals.scene_cut` on 23 profiles off a survey whose
reproducer was never committed, and review found three arming decisions
that could not survive contact with their own recorded evidence:

  * four profiles armed `kind: [fade]` -- which gates ONLY on the blank
    channel -- after that channel was measured at ZERO runs in 12,000
    steps. `tetris_usa` was DECLINED for exactly that evidence state.
  * one profile armed a death veto over a lives byte the survey itself
    recorded as `have_lives: false` (an admitted `lives: 0` placeholder),
    so the veto debounced RAM[0x0000]; another armed it over a byte this
    repo documents, the same day, as a 0<->255 flicker artifact.
  * seven profiles armed a gate at `blank_min: 1` while every blank run
    the survey observed for them was death-class, and the receipt's
    per-profile `reason` string called that "a null that measures zero by
    construction" -- next to its own field recording 33 runs.

The common root is that ARM was a JUDGEMENT rather than a MEASUREMENT.
This script makes it a measurement, and is committed so the number can be
re-derived rather than cited. Run as a measurement, the arming does not
survive: driven at 4 x 6,000 steps of their own ordinary play, **14 of
the 23 armed gates FIRE on play that cleared nothing** -- up to 207 false
positives on one profile. See the receipt.

WHAT IT MEASURES, per profile, from that profile's own ordinary play:

  RESIDUAL  the profile's OWN armed SceneCutSignal -- constructed through
            clear_detect.build_shelf_signals from the real YAML, not a
            synthetic stand-in -- fed every observation of a mixed-random
            rollout together with the profile's own declared lives byte.
            `n_triggers` is then the answer to the anti-vacuity question
            for this gate: WHAT DOES IT REPORT WHEN THE MECHANISM IS
            ABSENT? No level was cleared in these rollouts, so every
            trigger is a false positive and the only defensible residual
            is 0.
  NULL      the largest per-window (d_scene, d_blank) that ordinary play
            produces, over SceneCutSignal's own window/stride arithmetic,
            reported both with the death veto's coverage removed and --
            the number the gate is actually set from -- WITHOUT it. The
            gate a profile may arm is one integer above the second (the
            rule scripts/clear_calibrate.py already applies), never
            below it. Calibrating against the vetoed-out number instead
            is what let seven profiles ship `blank_min: 1` under a 5-9
            step death fade with the veto as their only guard.

...and the STATIC clauses that no amount of rollout can rescue, each with
the failure it exists to stop. C7 is the one that decides the roster
today, so it is stated first:

  C1 CHANNEL ALIVE       the channel `kind` actually gates on must have
                         been seen to move. A blank-only arm over a blank
                         counter measured at zero is indistinguishable
                         from an unwired odometer -- clear_calibrate.py's
                         own docstring says so in writing.
  C2 VALIDATED LIVES     a `death_debounce` may only be declared over a
                         lives byte that (a) exists and (b) is not on this
                         repo's own list of measured-false nominations.
                         GenericGame.lives is `int(ram[addr])` with no
                         None handling: `lives: 0` debounces a zero-page
                         scratch byte, and a byte that flickers 0<->255
                         drives the veto instead of guarding it.
  C7 SEPARABILITY         an ARM requires a WITNESSED POSITIVE -- one
     WITNESSED            blank run known to be a level transition, so a
                          length floor can be placed between it and the
                          death population. The probe clears nothing, so
                          a gate at or below its null fires on ordinary
                          play (demonstrated) and a gate above it has no
                          evidence it can ever fire. This clause is what
                          makes C6 decidable, and it refuses the whole
                          cohort today. Same refusal, same reason, as the
                          Rygar clear predicate.
  C6 GATE ABOVE THE       the gate must clear the veto-INDEPENDENT null
     FULL NULL            on its own. The probe clears nothing, so its
                          whole (d_scene, d_blank) distribution is null;
                          a gate at or below it fires on ordinary play
                          by construction and the death veto is then the
                          single point of failure. Checked against this
                          script's own committed receipt, so the suite
                          can guard it without a ROM.
  C3 SEPARABILITY UNWITNESSED (disclosure, not a veto)
                         when every blank run observed was death-class,
                         the profile has no witnessed non-death fire and
                         its residual is zero only because the veto ate
                         the entire population. That is still a real
                         measurement -- but it must travel WITH the
                         number, in the config, or it is a number without
                         its meaning.

WHAT WOULD THIS REPORT IF THE MECHANISM WERE ABSENT? A profile whose
odometer never moves reports n_blank_runs=0 / n_scene_runs=0 / residual=0
-- and residual=0 is the PASSING value, so a dead surface would sail
through the measured half. That is precisely why C1 is a separate static
clause evaluated against the raw run counts and why `verdict` is
DECLINE-CHANNEL-DEAD, never ARM, in that state. Reporting the residual
alone would be the same shape as the four vacuous gates this week.

USAGE
    .venv/bin/python scripts/scene_cut_arming.py --audit          # all
    .venv/bin/python scripts/scene_cut_arming.py --profile configs/gradius.yaml
    .venv/bin/python scripts/scene_cut_arming.py --check          # no ROMs
                    # static clauses + committed receipt vs the real YAML

`--check` is the cheap half and is what tests/test_scene_cut_arming.py
runs: it needs no emulator, no ROM and no receipt regeneration, so the
arming decision has a guard in the suite rather than only in a JSON file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Where the audit's own receipt lives -- the measurement of the configs
#: AS THEY SHIP. The roster tests read this one.
RECEIPT = REPO / "docs/receipts/clear_control/scene_cut_arming_2026-08-27.json"

#: The AS-FOUND measurement: the same audit run against the 2026-08-26
#: arming, before this commit disarmed it. Banked separately because it is
#: the EVIDENCE for the disarm decision, and after the disarm the receipt
#: above necessarily has no armed gate left to describe -- so without this
#: file the residual assertions would have nothing to iterate over and
#: would pass vacuously, which is the failure mode this whole campaign is
#: about. Regenerate only by checking out 27902e1's configs; it is a
#: historical measurement, not a live one.
ASFOUND = REPO / "docs/receipts/clear_control/scene_cut_arming_asfound_2026-08-27.json"

#: The survey this audit re-adjudicates. Read for the fields a static
#: check cannot re-derive without ROMs (raw run counts, death classes).
SURVEY = REPO / "docs/receipts/clear_control/odometer_cohort_scene_cut_survey_2026-08-26.json"

#: Lives nominations this repo has MEASURED to be false, with the receipt
#: that measured each. C2 reads this rather than a judgement call; adding
#: to it is how a newly-falsified byte disarms every veto that reads it.
#:
#: The three $-addresses below are the ones docs/research/
#: FALSE_DEATH_FANOUT_2026-08-26.md and tests/test_lives_behaviour_gate.py
#: name; only the first is currently declared by a cohort profile.
FALSIFIED_LIVES = {
    ("ninja_gaiden", 0x001F): (
        "docs/research/FALSE_DEATH_FANOUT_2026-08-26.md: $001F is a "
        "flicker artifact making 12-26 unpaired 0<->255 ticks per run, "
        "demoted to rank 15 of 22 by the fixed ranker"),
    ("bad_dudes", 0x00CD): (
        "docs/research/FALSE_DEATH_FANOUT_2026-08-26.md §6: an attack-"
        "animation counter cycling 2 -> 0 -> 2"),
    ("journey_to_silius", 0x0135): (
        "docs/research/FALSE_DEATH_FANOUT_2026-08-26.md §6: drops within "
        "four steps of ANY rightward hold -- movement onset, not death"),
}


#: Profiles with a WITNESSED level transition -- a blank run known, from a
#: real trajectory, to be a level transition rather than a death fade or a
#: boot fade. C7 reads this. It is deliberately near-empty: this is the
#: register that says which profiles have a positive to calibrate against,
#: and adding to it requires a receipt, not a judgement.
#:
#: `rygar` is NOT listed here even though it is the one profile that HAS a
#: witnessed transition (docs/receipts/rygar/clear_predicate_REFUTED.md:
#: death fade 14 blank frames 36/36, door 78-79 55/55, floor 40). It is
#: served by its own instrument, scripts/transition_witness.py, and
#: configs/rygar.yaml deliberately arms nothing -- a guard test asserts
#: that. Listing it here would invite arming the generic gate on top.
WITNESSED_TRANSITIONS: dict[str, str] = {}


def cohort(config_dir: Path | None = None) -> list[Path]:
    """The odometer cohort, DERIVED not listed: every profile whose
    progress scalar comes from the PPU scroll odometer, which is the exact
    shape that makes clear_reachability's `coord` check structurally
    dead and so the exact shape that needs a replacement transition
    signal."""
    out = []
    for p in sorted((config_dir or (REPO / "configs")).glob("*.yaml")):
        try:
            prof = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            continue
        solve = prof.get("solve")
        if not isinstance(solve, dict):
            continue
        prog = solve.get("progress")
        if isinstance(prog, dict) and prog.get("source") == "odometer":
            out.append(p)
    return out


def armed_block(prof: dict) -> dict | None:
    """This profile's scene_cut arming, or None for absent/declined."""
    solve = prof.get("solve") or {}
    clear = solve.get("clear") or {}
    sigs = clear.get("signals") or {}
    cfg = sigs.get("scene_cut")
    if not isinstance(cfg, dict) or cfg.get("enabled") is False:
        return None
    return cfg


# ---------------------------------------------------------------------------
# The static clauses. No ROM, no emulator -- this half runs in the suite.
# ---------------------------------------------------------------------------

def static_clauses(name: str, prof: dict, survey: dict | None,
                   path: Path | None = None,
                   banked: dict | None = None) -> list[dict]:
    """[{clause, ok, detail}] for one profile's arming, in policy order.

    `survey` is that profile's record from the 2026-08-26 survey (raw run
    counts per channel). Absent survey record => C1 cannot be evaluated
    and is reported UNEVALUATED, never ok -- a missing measurement is not
    a passing one.

    `banked` is that profile's `measured` record from this script's own
    committed receipt, which is what lets C6 -- the calibration clause --
    be checked with no ROM and no emulator, i.e. in the suite. Same rule:
    absent record is UNEVALUATED, never ok."""
    cfg = armed_block(prof)
    out: list[dict] = []
    if cfg is None:
        return [{"clause": "ARMED", "ok": True,
                 "detail": "not armed (absent or enabled: false) -- "
                           "no clause applies"}]

    kind = list(cfg.get("kind") or ())
    gates_blank = True                      # blank_min is always compared
    gates_scene = bool(set(kind) & {"pan", "warp"})

    # -- C1 CHANNEL ALIVE ---------------------------------------------------
    if survey is None:
        out.append({"clause": "C1_CHANNEL_ALIVE", "ok": False,
                    "detail": "no survey record -- the channel this "
                              "profile gates on has never been measured"})
    else:
        n_blank = int(survey.get("n_blank_runs") or 0)
        n_scene = int(survey.get("n_scene_runs") or 0)
        live = (n_blank if not gates_scene else max(n_blank, n_scene))
        ok = live > 0
        out.append({
            "clause": "C1_CHANNEL_ALIVE", "ok": ok,
            "detail": (
                f"kind={kind} gates on "
                f"{'blank+scene' if gates_scene else 'blank only'}; "
                f"measured n_blank_runs={n_blank}, n_scene_runs={n_scene}"
                + ("" if ok else
                   " -- the armed channel never moved, which is the "
                   "evidence state tetris_usa was DECLINED for"))})

    # -- C2 VALIDATED LIVES -------------------------------------------------
    if "death_debounce" not in cfg:
        out.append({"clause": "C2_VALIDATED_LIVES", "ok": True,
                    "detail": "no death veto declared -- nothing to validate"})
    else:
        addr = (prof.get("solve") or {}).get("lives")
        falsified = FALSIFIED_LIVES.get((name, int(addr))) if addr else None
        if not addr:
            out.append({
                "clause": "C2_VALIDATED_LIVES", "ok": False,
                "detail": f"declares death_debounce={cfg['death_debounce']} "
                          f"but solve.lives is {addr!r} -- GenericGame.lives "
                          "is int(ram[addr]) with no None handling, so the "
                          "veto would debounce RAM[0x0000], a zero-page "
                          "scratch byte"})
        elif falsified:
            out.append({
                "clause": "C2_VALIDATED_LIVES", "ok": False,
                "detail": f"declares death_debounce over lives=0x{int(addr):04X}, "
                          f"a MEASURED-FALSE nomination: {falsified}"})
        else:
            out.append({"clause": "C2_VALIDATED_LIVES", "ok": True,
                        "detail": f"lives=0x{int(addr):04X}, not on the "
                                  "measured-false list"})

    # -- C3 SEPARABILITY DISCLOSURE ----------------------------------------
    if survey is not None and survey.get("has_non_death_candidate") is False:
        # The disclosure surface is the YAML COMMENT beside the gate --
        # what a reader of the config actually sees. Not a key: a key
        # would have to be allow-listed by build_shelf_signals and would
        # travel into the signal's constructor.
        disclosed = path is not None and _comment_has_caveat(path)
        out.append({
            "clause": "C3_SEPARABILITY_DISCLOSED", "ok": bool(disclosed),
            "detail": (
                f"every one of the {survey.get('n_blank_runs')} blank runs "
                "observed for this profile was death-class, so the residual "
                "is zero only because the veto ate the whole population -- "
                "the config must carry that in a comment beside the gate"),
            "needs_config_comment": True})

    # -- C5 SCENE_MIN ABOVE THE SEAM ---------------------------------------
    # SceneCutSignal's own docstring: "the core's scene-cut heuristic fires
    # on ordinary camera clamp/seam noise near screen edges -- exactly
    # where real pans happen -- which is exactly why scene_min cannot be 1".
    # The floor is `warp_scene_min` (2), an ALREADY pre-registered constant
    # from the 2026-08-24 probe receipts, not a new number: below it a
    # scene bump cannot even be classified as a warp, so a gate at 1 is
    # asking to fire on a single seam bump by construction. The check is
    # unconditional because `kind: [fade]` still compares d_scene against
    # scene_min -- restricting the kind does NOT take the scene channel out
    # of the gate.
    warp_min = int(cfg.get("warp_scene_min", 2))
    sm = int(cfg.get("scene_min", 0))
    out.append({
        "clause": "C5_SCENE_MIN_ABOVE_SEAM", "ok": sm >= warp_min,
        "detail": f"scene_min={sm} against the seam floor warp_scene_min="
                  f"{warp_min}" + ("" if sm >= warp_min else
                                   " -- a single scene bump is the "
                                   "documented camera-clamp artifact")})

    # -- C7 SEPARABILITY WITNESSED -----------------------------------------
    # The clause that makes C6 decidable, and the one that turns out to
    # refuse the whole cohort.
    #
    # The audit probe CLEARS NOTHING -- it is undirected play on a roster
    # whose real level transitions gate behind the search this project
    # runs Go-Explore for. So every blank run it observes is a
    # non-transition, and the null therefore covers the ENTIRE observed
    # population. Two exhaustive cases follow, and neither is an ARM:
    #
    #   gate <= null : the gate fires on ordinary play, demonstrated, and
    #                  only the death veto holds it shut.
    #   gate >  null : the gate is above every blank run this profile has
    #                  ever been seen to produce, and NOTHING establishes
    #                  that a real transition would clear it.
    #
    # An ARM therefore requires a WITNESSED POSITIVE: at least one blank
    # run known to be a level transition, so a length floor can be placed
    # between it and the death population. Rygar has exactly that (death
    # fade 14 blank frames, door 78-79, floor 40, from its own banked
    # tape) and is deliberately handled by its own instrument,
    # scripts/transition_witness.py. No other cohort profile has one --
    # the 2026-08-26 survey says so itself, in its `scope_limit`: "'ARM'
    # records that the counter is alive ... NOT that a real transition has
    # been witnessed on tape."
    #
    # This is the SAME refusal as docs/receipts/rygar/
    # clear_predicate_REFUTED.md, for the same reason: a mechanism with no
    # witnessed positive can only ever be shown NOT to fire. Registering a
    # profile here is how that changes -- put a receipt in
    # WITNESSED_TRANSITIONS and the clause passes.
    if name not in WITNESSED_TRANSITIONS:
        out.append({
            "clause": "C7_SEPARABILITY_WITNESSED", "ok": False,
            "detail": (
                "no blank run on this profile has ever been WITNESSED to be "
                "a level transition, so there is nothing to place a length "
                "floor against. The audit probe clears nothing, so its whole "
                "observed distribution is null: a gate at or below it fires "
                "on ordinary play, and a gate above it has no evidence it "
                "can ever fire. Same refusal, and same reason, as the Rygar "
                "clear predicate")})
    else:
        out.append({"clause": "C7_SEPARABILITY_WITNESSED", "ok": True,
                    "detail": WITNESSED_TRANSITIONS[name]})

    # -- C6 GATE ABOVE THE VETO-INDEPENDENT NULL ---------------------------
    # The clause the 2026-08-26 arming had no equivalent of, and the one
    # that catches `blank_min: 1` sitting under a 5-9-step death fade. The
    # probe that produced `banked` cleared nothing, so every window in it
    # is a non-transition and its whole (d_scene, d_blank) distribution is
    # null. A gate at or below that null fires on the null by
    # construction; whether a death veto happens to cover it is a
    # different question and belongs to a different guard.
    bm = int(cfg.get("blank_min", 0))
    if banked is None:
        out.append({"clause": "C6_GATE_ABOVE_FULL_NULL", "ok": False,
                    "detail": "no banked measurement -- this gate's null "
                              "has never been measured with the death "
                              "veto's coverage removed"})
    else:
        nb = int(banked.get("null_max_d_blank_all", 0))
        ns = int(banked.get("null_max_d_scene_all", 0))
        ok = bm > nb and sm > ns
        out.append({
            "clause": "C6_GATE_ABOVE_FULL_NULL", "ok": ok,
            "detail": (
                f"gate (scene_min={sm}, blank_min={bm}) against the "
                f"veto-independent null (max d_scene={ns}, max d_blank="
                f"{nb}) over {banked.get('n_rollouts')}x"
                f"{banked.get('steps_per_rollout')} steps that cleared "
                "nothing" + ("" if ok else
                             " -- the gate is at or below its own null, "
                             "so it fires on ordinary play and only the "
                             "death veto is holding it shut"))})
    return out


def _comment_has_caveat(path: Path) -> bool:
    """Does the YAML text carry the death-class caveat as a comment?

    A YAML COMMENT is the disclosure surface, not a key -- a key would
    have to be allow-listed by build_shelf_signals and would then travel
    into the signal's constructor. So this is a text check on the file,
    which is exactly what the reader of the config sees."""
    txt = path.read_text()
    return "death-class" in txt and "SURVEY CAVEAT" in txt


# ---------------------------------------------------------------------------
# The measured half. Needs ROMs; run with --audit.
# ---------------------------------------------------------------------------

class NullTracker:
    """Largest (d_scene, d_blank) over SceneCutSignal's own window
    arithmetic, reported TWICE -- with the death veto's coverage removed
    and with it included.

    Separate from SceneCutSignal deliberately: asking a gated instrument
    what its null is would make the answer depend on the guess being
    calibrated away (scripts/clear_calibrate.py makes the same split for
    the same reason).

    WHY BOTH NUMBERS, AND WHY THE GATE IS SET FROM THE SECOND ONE
    (2026-08-27, after review). The 2026-08-26 arming set `blank_min: 1`
    on seven profiles whose every observed blank run was a death fade
    5-9 steps long -- i.e. the gate sat an order of magnitude BELOW its
    own null and fired on it by construction, and the only thing standing
    between that and a fabricated clear was the death veto. On
    configs/gradius.yaml the veto was covering 87.5% of all windows, so
    its "residual 0" was not a property of the gate at all.

    The probe clears nothing -- it is undirected play on a roster whose
    real transitions gate behind the search this project runs Go-Explore
    for. Therefore EVERY window it produces is a non-transition, and the
    whole observed d_blank distribution is null, veto or no veto. The
    rule scripts/clear_calibrate.py already writes down -- gate one
    integer above the observed null -- must therefore be applied to the
    UNVETOED-INCLUSIVE maximum. That demotes the death veto from sole
    guard to second guard, which is the entire lesson of this week."""

    def __init__(self, window: int, stride: int):
        self.window, self.stride = window, stride
        self._buf: deque = deque(maxlen=window)
        self._n = 0
        self.max_scene = 0
        self.max_blank = 0
        # ...and the same maxima with NO window excluded. This is the
        # number the gate is calibrated from.
        self.max_scene_all = 0
        self.max_blank_all = 0
        self.n_checks = 0
        self.n_vetoed = 0

    def push(self, scene: int, blank: int, dying: bool) -> None:
        self._n += 1
        self._buf.append((int(scene), int(blank)))
        if self._n % self.stride or len(self._buf) < 2:
            return
        d_scene = self._buf[-1][0] - self._buf[0][0]
        d_blank = self._buf[-1][1] - self._buf[0][1]
        self.max_scene_all = max(self.max_scene_all, d_scene)
        self.max_blank_all = max(self.max_blank_all, d_blank)
        if dying:
            self.n_vetoed += 1
            return
        self.n_checks += 1
        self.max_scene = max(self.max_scene, d_scene)
        self.max_blank = max(self.max_blank, d_blank)


def measure(path: Path, rollouts: int = 3, steps: int = 4000,
            seed: int = 0, forward_bias: float = 0.5) -> dict:
    """Drive this profile's own ordinary play through its OWN armed signal.

    The policy is the survey's: uniform-random over the profile's action
    space, biased toward its declared forward direction. It is NOT skilled
    play and cannot reach a real level transition for most of this roster
    -- which is the point. Everything it produces is, by construction, a
    non-transition, so every trigger is a false positive."""
    import nes_core
    from clear_detect import build_shelf_signals
    import clear_reachability
    from go_explore_solve import make_game
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load(path.read_text())
    name = path.stem
    cfg = armed_block(prof)
    q = clear_reachability.clear_quorum(prof, roster=clear_reachability.OFFLINE)
    shelf = build_shelf_signals(prof, q)
    sig = shelf.get("scene_cut")

    try:
        game = make_game(prof)
    except Exception as e:
        # A profile that cannot be constructed cannot be measured, and
        # `tetris_usa` is exactly that (`solve.constructible: false`). It
        # must come back as UNMEASURABLE with the reason attached -- NOT
        # as a row of zeros, which would read identically to a profile
        # whose gate is clean.
        return {"profile": name, "armed": cfg is not None,
                "gate": None if cfg is None else {k: cfg[k] for k in sorted(cfg)},
                "unmeasurable": f"{type(e).__name__}: {e}",
                "n_rollouts": 0, "steps_per_rollout": int(steps),
                "n_blank_runs": None, "n_scene_runs": None,
                "residual_triggers": None,
                "null_max_d_scene_all": None, "null_max_d_blank_all": None,
                "recommended_gate": None}
    fs = int(prof.get("frame_skip", 4))
    space = [list(a) for a in prof["action_space"]]
    masks = action_space_to_bitmasks(space)
    fwd_dir = ((prof.get("solve") or {}).get("progress") or {}).get("forward", "right")
    fwd = [i for i, a in enumerate(space) if fwd_dir in a] or [0]
    lives_addr = (prof.get("solve") or {}).get("lives")

    start = prof.get("start_state_path")
    blob = (REPO / start).read_bytes() if start and (REPO / start).exists() else None

    t0 = time.time()
    null = NullTracker(int((cfg or {}).get("window", 240)) if cfg else 240,
                       int((cfg or {}).get("stride", 20)) if cfg else 20)
    if sig is not None:
        null = NullTracker(sig.window, sig.stride)
    blank_runs: list[int] = []
    scene_runs: list[int] = []
    ends_dying = 0
    for r in range(rollouts):
        rng = random.Random(seed * 1000 + r)
        env = nes_core.NESEnvironment(game.rom, frame_skip=fs)
        env.reset()
        env.set_odometer_enabled(True)
        try:
            if blob is not None:
                env.load_state(blob)
            if sig is not None:
                sig.reset()
            pb, ps = env.get_odometer_blank(), env.get_odometer_scene()
            b_run = s_run = 0
            for _ in range(steps):
                a = rng.choice(fwd) if rng.random() < forward_bias \
                    else rng.randrange(len(space))
                env.step(int(masks[a]))
                b, s = env.get_odometer_blank(), env.get_odometer_scene()
                if b > pb:
                    b_run += 1
                elif b_run:
                    blank_runs.append(b_run)
                    b_run = 0
                if s > ps:
                    s_run += 1
                elif s_run:
                    scene_runs.append(s_run)
                    s_run = 0
                pb, ps = b, s
                lv = None
                if lives_addr:
                    lv = int(env.get_ram_range(int(lives_addr),
                                               int(lives_addr) + 1)[0])
                if sig is not None:
                    sig.push(env.get_odometer(), s, b, lives=lv)
                    null.push(s, b, sig.dying)
                else:
                    null.push(s, b, False)
            if b_run:
                blank_runs.append(b_run)
            if s_run:
                scene_runs.append(s_run)
            if sig is not None and sig.dying:
                ends_dying += 1
        finally:
            env.close()

    st = sig.stats() if sig is not None else {}
    return {
        "profile": name,
        "armed": cfg is not None,
        "gate": None if cfg is None else {k: cfg[k] for k in sorted(cfg)},
        "lives_addr": None if not lives_addr else f"0x{int(lives_addr):04X}",
        "n_rollouts": rollouts, "steps_per_rollout": steps,
        "n_blank_runs": len(blank_runs), "blank_run_lengths": sorted(blank_runs),
        "n_scene_runs": len(scene_runs), "scene_run_lengths": sorted(scene_runs),
        "residual_triggers": st.get("n_triggers"),
        "residual_events": st.get("n_events"),
        "n_checks": st.get("n_checks"),
        "n_death_vetoes": st.get("n_death_vetoes"),
        "n_lives_rebaselines": st.get("n_lives_rebaselines"),
        "rollouts_ending_dying": ends_dying,
        "null_max_d_scene": null.max_scene,
        "null_max_d_blank": null.max_blank,
        # The veto-independent null -- what the gate must clear on its
        # own, with no help from the death veto. See NullTracker.
        "null_max_d_scene_all": null.max_scene_all,
        "null_max_d_blank_all": null.max_blank_all,
        "null_checks_scored": null.n_checks,
        "null_checks_vetoed": null.n_vetoed,
        # LIVENESS. A veto that covers every check is a signal that cannot
        # fire, which is the same vacuity as a gate that always fires --
        # reported next to the residual so a 0 residual can never be read
        # as health on its own.
        "fraction_checks_vetoed": (
            None if not (null.n_checks + null.n_vetoed) else
            round(null.n_vetoed / (null.n_checks + null.n_vetoed), 3)),
        # One integer above the VETO-INDEPENDENT null, floored at the
        # pre-registered seam constant for the scene channel (a scene
        # gate below warp_scene_min cannot distinguish a cut from a
        # camera clamp, whatever the null says).
        "recommended_gate": {
            "scene_min": max(null.max_scene_all + 1,
                             int((cfg or {}).get("warp_scene_min", 2))),
            "blank_min": null.max_blank_all + 1},
        "wall_s": round(time.time() - t0, 1),
    }


def adjudicate(name: str, path: Path, prof: dict, survey: dict | None,
               measured: dict | None, banked: dict | None = None) -> dict:
    """One profile's verdict, from the clauses plus (when available) the
    residual. DECLINE always carries the clause that produced it.

    A live `measured` always outranks `banked` for C6 -- re-measuring is
    how the banked number is allowed to move."""
    clauses = static_clauses(name, prof, survey, path,
                             banked=measured if measured is not None else banked)
    cfg = armed_block(prof)
    if cfg is None:
        return {"verdict": "NOT_ARMED", "clauses": clauses}
    failed = [c for c in clauses if not c["ok"]]
    hard = [c for c in failed if c["clause"] != "C3_SEPARABILITY_DISCLOSED"]
    if hard:
        return {"verdict": "DECLINE", "blocking": [c["clause"] for c in hard],
                "clauses": clauses}
    soft = [c for c in failed]
    v = {"verdict": "ARM", "clauses": clauses}
    if soft:
        v["verdict"] = "ARM_UNDISCLOSED"
        v["blocking"] = [c["clause"] for c in soft]
    if measured is not None and measured.get("unmeasurable"):
        # Not a pass. A profile that cannot be driven cannot have its
        # gate measured, and an unmeasured gate is not a clean one.
        v["verdict"] = "DECLINE"
        v["blocking"] = (v.get("blocking") or []) + ["C4_UNMEASURABLE"]
        v["residual_detail"] = measured["unmeasurable"]
    elif measured is not None:
        res = measured.get("residual_triggers")
        v["residual_triggers"] = res
        if res:
            v["verdict"] = "DECLINE"
            v["blocking"] = (v.get("blocking") or []) + ["C4_RESIDUAL_ZERO"]
            v["residual_detail"] = (
                f"the armed gate fired {res} times in "
                f"{measured['n_rollouts'] * measured['steps_per_rollout']} "
                "steps of play that cleared nothing -- every one is a false "
                f"positive. Measured null wants "
                f"{measured['recommended_gate']}")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default=None, help="one configs/*.yaml")
    ap.add_argument("--audit", action="store_true",
                    help="drive every cohort profile (needs ROMs)")
    ap.add_argument("--check", action="store_true",
                    help="static clauses only -- no ROM, no emulator")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    survey = {}
    if SURVEY.exists():
        survey = (json.loads(SURVEY.read_text()) or {}).get("survey") or {}
    banked = {}
    if RECEIPT.exists():
        banked = (json.loads(RECEIPT.read_text()) or {}).get("profiles") or {}

    paths = [Path(a.profile)] if a.profile else cohort()
    rows, bad = {}, []
    dest = Path(a.out) if a.out else (None if a.check else RECEIPT)

    def emit():
        """Write the receipt as it stands. Called after EVERY profile, not
        once at the end: a ~50-minute measurement that loses everything to
        a crash on its last profile is how the first run of this audit was
        lost."""
        if dest is None:
            return
        out = {"what_this_is": __doc__.split("\n")[0],
               "policy": "C1 channel alive | C2 validated lives | "
                         "C3 separability disclosed | C4 residual zero at the "
                         "armed gate | C5 scene_min above the seam floor | "
                         "C6 gate above the veto-independent null | "
                         "C7 separability witnessed",
               "static_only": bool(a.check),
               "cohort_size": len(paths),
               "complete": len(rows) == len(paths),
               "generated_at": "2026-08-27",
               "profiles": rows}
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1) + "\n")

    for p in paths:
        prof = yaml.safe_load(p.read_text())
        name = p.stem
        m = None
        if not a.check:
            m = measure(p, rollouts=a.rollouts, steps=a.steps, seed=a.seed)
        v = adjudicate(name, p, prof, survey.get(name), m,
                       banked=(banked.get(name) or {}).get("measured"))
        rows[name] = {"verdict": v, "measured": m}
        if v["verdict"] not in ("ARM", "NOT_ARMED"):
            bad.append(name)
        print(f"{name:<26} {v['verdict']:<16} "
              f"{','.join(v.get('blocking') or []) or '-'}"
              + ("" if m is None or m.get("unmeasurable") else
                 f"  residual={m['residual_triggers']} "
                 f"null_all=(s{m['null_max_d_scene_all']},"
                 f"b{m['null_max_d_blank_all']})"
                 f" null_vetoed=(s{m['null_max_d_scene']},"
                 f"b{m['null_max_d_blank']})"),
              flush=True)
        emit()

    emit()
    if dest is not None:
        print(f"\nreceipt -> {dest}")
    print(f"\n{len(paths) - len(bad)}/{len(paths)} clean, "
          f"{len(bad)} blocked: {', '.join(bad) or '-'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
