"""Is a discovered progress byte actually usable by the solver?

discover_observables recommends a byte on a SHORT probe. The solver then
reads it over long rollouts, and the two can disagree badly:

  * Rygar. Recommended $0015 on net +2896 at monotone fraction 1.00, and
    reported wrapPair=0 in the same breath — no high byte found. The
    solver reads the low byte RAW, so it saw a value that saturates at
    209 and takes 21 distinct values in 1200 steps. 73 minutes of search
    produced 116 cells and zero progress, because the archive could not
    represent being further along.
  * Ninja Gaiden. Recommended $001F as the death byte on 5-of-5
    agreement, while that byte reads 0 at the start state, so
    "death = decrement" underflows to 255 and no death is detectable.

Both were reported as game-difficulty walls before anyone checked the
instrument. This gate is the check, and it runs BEFORE a profile is
allowed to drive a search.

And then this gate went and made the same mistake in its own turn
(2026-08-26). It reported "only 20 distinct values in 69 steps (< 32) —
too coarse to be a search gradient" about Contra, where the 69 steps
were all that survived D5 truncation of a 1200-step hold. A 69-sample
window cannot demonstrate a 32-distinct threshold: that sentence
measures how fast the PROBE died. Two things came out of it, and they
are the two axes this file now separates:

  * A verdict is only issued on a window that can carry it
    (MIN_ASSESSABLE_STEPS). Below that the honest answer is
    INCONCLUSIVE — blocked, but not condemned. VOID, not FAIL.
  * The probe is a variable, not a constant. `--probe hold` is the
    original undodging scripted forward hold and stays the default;
    `--probe random` buys a live window on games where holding forward
    walks into the first hazard (Contra: 69 live steps held forward,
    721 under uniform-random, 20 distinct against 346). Neither is
    strictly better and the file says which checks each one cannot make.

    .venv/bin/python scripts/progress_signal_gate.py --profile configs/rygar.yaml
    .venv/bin/python scripts/progress_signal_gate.py --profile configs/contra.yaml --probe random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MIN_DISTINCT = 32          # fewer levels than this is not a gradient
MIN_TAIL_FRACTION = 0.25   # progress must still be moving late in a roll

#: Live steps required before a SHORTFALL of distinct values may be
#: reported as an instrument fault.
#:
#: The defect this closes (2026-08-26). `assess()` computes its
#: resolution finding on the window that survives D5 truncation, and
#: then states it as a threshold claim: "only 20 distinct values in 69
#: steps (< 32) — too coarse to be a search gradient". A 69-sample
#: window cannot demonstrate a 32-distinct threshold. That sentence
#: measures how fast the scripted forward hold died, not the signal's
#: resolution — and it excluded Contra, whose progress pair
#: {lo:0x65, hi:0x64} is one of the better ones on the roster, on an
#: inference the evidence does not support.
#:
#: CALIBRATED, not chosen. Value = the largest number of live steps any
#: profile this gate has certified SIGNAL SOUND needed in order to
#: accumulate MIN_DISTINCT levels (`steps_to_min_distinct` on the
#: hold-probe sweep, docs/receipts/progress_gate_window_sweep_2026-08-26.json).
#: Below this window, a signal we have independently certified as sound
#: would itself have been called "too coarse" — which is the definition
#: of a window that cannot carry the claim. Raising it only converts
#: FAILs into VOIDs; it can never turn a FAIL into a PASS, because the
#: PASS direction is a positive demonstration and is not gated on it
#: (see `assess`).
MIN_ASSESSABLE_STEPS = 187

#: VOID, not FAIL. A profile that lands here has NOT been shown sound —
#: `passed` is False and it must not drive a search — but neither has it
#: been shown broken, and the two must never be reported as the same
#: thing. Same distinction CLAIMS.md draws between a FAILED eval and a
#: VOIDED one.
INCONCLUSIVE_VERDICT = "INCONCLUSIVE — probe died too early to assess"


def steps_to_distinct(trace: list, target: int = MIN_DISTINCT) -> int | None:
    """How many leading steps of `trace` it took to accumulate `target`
    distinct values, or None if it never got there.

    This is the measurement MIN_ASSESSABLE_STEPS is calibrated from, and
    it is on every receipt so the calibration can be re-derived from the
    sweep rather than taken on trust.
    """
    seen = set()
    for i, v in enumerate(trace, start=1):
        seen.add(v)
        if len(seen) >= target:
            return i
    return None


def verdict_label(instrument: list, inconclusive: list,
                  behaviour: list) -> str:
    """The one place the three finding lists become a verdict string.

    Precedence is deliberate and is the whole point of the INCONCLUSIVE
    band: a positively DEMONSTRATED instrument fault outranks "not
    enough window to say" (evidence beats absence of evidence), and
    "not enough window to say" outranks any SOUND certification (a
    window too short to condemn a signal is equally too short to bless
    it — the failure mode of a one-sided fix).
    """
    if instrument:
        return "SIGNAL UNUSABLE"
    if inconclusive:
        return INCONCLUSIVE_VERDICT
    if behaviour:
        return "SIGNAL SOUND — game stops"
    return "SIGNAL SOUND — still advancing"


#: Same fraction `assess()` already uses to call a PROGRESS trace's tail
#: flat; reused here on the LIVES trace so "exhaustion" is judged by the
#: one heuristic this file already trusts, not a second one invented to
#: match a single game.
EXHAUSTION_TAIL_FRACTION = 0.75


def first_exhaustion_index(lives_trace: list, start_lives: int) -> int | None:
    """First step at which the lives byte records a death (the exact
    modular check `GenericGame.is_dead` uses: `(start - cur) % 256` in
    1..8) that the game never comes back from — the trailing quarter of
    `lives_trace` is frozen at a single value from then on.

    D5 (docs/research/CLEAR_GAP_CLOSURE_2026-08-26.md §9.6): this gate's
    forward hold never watched the lives byte at all, so a 1-life game
    (Arkanoid: lives_at_start=1) that loses its only ball ~halfway
    through the default 1200-step hold spent the remaining ~600 steps
    stepping a non-interactive post-game-over screen, and one of those
    steps' incidental area-byte flip got read back as a real "room
    transition" in the discovery receipt. Reproducing it directly shows
    WHY a naive "stuck at 0" test is not enough: Arkanoid's own byte
    does not stay at 0 — it reads 1 -> 0 for a few hundred steps, then
    jumps to a constant 14 (evidently this ROM's attract/demo-mode
    placeholder) and sits there through the end of the hold. Requiring
    only "the trailing quarter is frozen at ONE value, whatever it is"
    after a real death catches that shape without needing to know what
    14 means.

    A single decrement with no such freeze is NOT treated as
    exhaustion — most games in this roster start with more than one
    life, and losing one of several while a forced, undodging forward
    hold runs into the first hazard is ordinary scripted-probe
    behaviour, usually followed by a real in-place or checkpoint
    respawn (the lives byte keeps varying afterward), not a
    non-interactive tail.

    A lives byte that reads 0 at the start state is NOT exempt. It used
    to be (`start_lives in (None, 0) -> return None`), which pinned the
    whole D5 fix out of reach on exactly the profiles the D5 sweep had
    flagged as contaminated: bad_dudes, ducktales_2, ninja_gaiden and
    paperboy all report `lives_at_start: 0` and all four show the
    died-then-frozen-tail shape (runs/onboard_wave6_d5_sweep_v3.json).
    The modular check handles that case perfectly well — a 0 -> 255
    underflow is `(0 - 255) % 256 == 1` — and `assess()` independently
    fails any profile whose death byte reads 0 at the start state, so
    the only effect of dropping the exemption is that a contaminated
    tail is now dropped instead of silently assessed.

    Returns None when there is nothing to exhaust (no lives byte, or no
    death is ever recorded) or the trace keeps varying after the death
    it does record.
    """
    if not lives_trace or start_lives is None:
        return None
    death = None
    for i, v in enumerate(lives_trace):
        if v is None:
            continue
        d = (int(start_lives) - int(v)) % 256
        if 1 <= d <= 8:
            death = i
            break
    if death is None:
        return None
    tail = lives_trace[int(len(lives_trace) * EXHAUSTION_TAIL_FRACTION):]
    return death if tail and len(set(tail)) == 1 else None


#: Below this many bytes of median per-step RAM churn a profile is too
#: quiet for the stasis test to separate "frozen" from "ordinary play",
#: so the test DISARMS with the measured reason recorded rather than
#: guessing. Same floor and the same refusal-to-guess discipline as the
#: solver's TerminalStasisSignal.
STASIS_MIN_MEDIAN_CHURN = 8
#: A frozen surface is one whose per-step churn sits at or under this
#: fraction of the profile's own live median.
STASIS_TOL_FRACTION = 0.05
#: ...and does so for at least this fraction of the hold. A brief pause
#: (a fade, a door transition, a boss intro) is not an absorbing state.
STASIS_MIN_TAIL_FRACTION = 0.25


def first_stasis_index(churn: list[int]) -> tuple[int | None, str]:
    """First step of a terminal STASIS tail: the point after which the
    whole RAM surface stops moving and never restarts.

    Why this exists next to `first_exhaustion_index`, which already
    detects exhaustion from the lives byte: defect D1 (2026-08-26)
    proved the lives byte can be BLIND to a terminal state. On
    ninja_gaiden_ii the byte at $004C sat flat at 1 straight through
    GAME OVER, so `(start - cur) % 256` was 0 the whole time and the
    modular check — which exists to catch the 0 -> 255 wrap — reported
    nothing. The solver grew `TerminalStasisSignal` for that. This gate
    re-imported the same blind predicate in the same commit wave and
    would happily assess a frozen GAME OVER screen as live play, which
    is the exact contamination D5 exists to remove.

    The claim here is deliberately weaker than "the player is dead": it
    is "this surface stopped moving and stayed stopped". Tolerance is
    derived from the profile's OWN median churn rather than a constant,
    so a quiet game and a busy one are held to the same relative
    standard, and a profile too quiet to discriminate disarms out loud.

    Returns `(index, reason)`. `index` is None when no stasis tail is
    found or the test disarmed; `reason` always carries the measurement,
    so a receipt records why the test said nothing.
    """
    n = len(churn)
    if n < 8:
        return None, f"trace too short for a stasis test ({n} steps)"
    live = sorted(churn[: max(n // 2, 1)])
    median = live[len(live) // 2]
    if median < STASIS_MIN_MEDIAN_CHURN:
        return None, (
            f"DISARMED: median live churn {median} bytes/step is under the "
            f"{STASIS_MIN_MEDIAN_CHURN}-byte floor — this profile is too "
            f"quiet for a frozen surface to be told apart from ordinary play")
    tol = max(1, int(median * STASIS_TOL_FRACTION))
    idx = n
    while idx > 0 and churn[idx - 1] <= tol:
        idx -= 1
    tail = n - idx
    if tail < max(int(n * STASIS_MIN_TAIL_FRACTION), 1):
        return None, (
            f"no stasis tail (median churn {median}, tol {tol}, longest "
            f"frozen tail {tail} of {n} steps)")
    return idx, (
        f"surface frozen from step {idx}: {tail} of {n} steps at or under "
        f"{tol} bytes/step against a live median of {median}")


def truncate_at_exhaustion(n: int, lives_trace, start_lives: int | None,
                           churn: list[int] | None = None) -> int:
    """How many LEADING steps of an `n`-step hold to keep before handing
    the trace to `assess()` — all of them, unless the hold ran off the
    end of live play, in which case the game-over/attract tail is
    dropped rather than silently believed.

    Two independent detectors, EARLIEST wins: the declared lives byte
    (`first_exhaustion_index`) and the surface itself
    (`first_stasis_index`). One is there because the other can be blind;
    see `first_stasis_index`.
    """
    idxs = []
    if lives_trace is not None and start_lives is not None:
        i = first_exhaustion_index(lives_trace, start_lives)
        if i is not None:
            idxs.append(i)
    if churn:
        i, _ = first_stasis_index(list(churn))
        if i is not None:
            idxs.append(i)
    return min(idxs) if idxs else n


def note_camera_static(v: dict, oam_churn: int, steps: int,
                       live_steps: int | None = None,
                       min_window: int | None = None,
                       directed: bool = True) -> dict:
    """Annotate (never launder) a verdict whose odometer axis had zero
    range (rx == ry == 0 at the call site).

    A camera that never moved means the rebased trace is constant —
    distinct=1, the exact degenerate case MIN_DISTINCT exists to catch.
    An earlier version of this function deleted that "too coarse"
    instrument finding and forced passed=True whenever OAM churn showed
    the agent was active, on the theory that a static camera is a fact
    about the GAME and not the instrument. That reasoning smuggled in a
    false conclusion: whether the flatness is a game-design fact or a
    route wall, the solver still cannot see a gradient on this axis —
    the archive would never grow past one cell no matter how active the
    agent was. "the agent moved" is not evidence "the odometer can ever
    report a positive here", and this instrument has never been
    demonstrated capable of doing so for a profile that lands here. So
    this must stay an instrument finding and keep blocking; only the
    OAM cross-check is new information, added as context.

    Regression case this guards (the legend_of_zelda receipt): distinct=1,
    min=0, max=0, oam_churn=967 — an agent that is visibly active on an
    axis that never moves. The old code certified that as
    "SIGNAL SOUND". It is not: it is the one shape of trace this gate
    exists to reject.

    `live_steps` is the window the zero range was actually observed over
    — NOT `steps`, which is the requested hold length and stays only in
    the human-readable text for context. The two came apart the moment
    D5 truncation shipped: double_dragon_ii's hold was requested for
    1200 steps and died at 257, and the old message said "camera never
    moved over 1200 steps" about 257 steps of evidence. Below
    MIN_ASSESSABLE_STEPS the zero range is the same unsupportable
    shortfall claim `assess` guards — a probe that died at step 22 has
    not shown the camera cannot move — so it is recorded as INCONCLUSIVE
    instead. It still blocks; it just stops asserting more than it saw.
    """
    min_window = (MIN_ASSESSABLE_STEPS if min_window is None
                  else int(min_window))
    live = steps if live_steps is None else int(live_steps)
    agent = ("agent active (OAM moving)" if oam_churn > live // 4
             else "agent inert (OAM static too)")
    if live >= min_window and directed:
        v["instrument_findings"].append(
            f"camera never moved over {live} live steps of a {steps}-step "
            f"hold under the odometer driver ({agent}) — a zero-range axis "
            f"cannot express progress regardless of agent activity; this "
            f"profile has not been demonstrated capable of returning a "
            f"positive on this axis and must not drive a search with it")
        label = "SIGNAL UNUSABLE — camera static"
    else:
        why = (f"under the {min_window}-step floor" if live < min_window
               else "under a probe that commanded no direction")
        v["inconclusive_findings"].append(
            f"camera showed zero range over {live} live steps of a "
            f"{steps}-step hold ({agent}) — {why}, so this is the probe's "
            f"survival time or its aimlessness, not a demonstration that "
            f"the axis cannot move")
        label = (f"{INCONCLUSIVE_VERDICT} — camera range zero on a short "
                 f"window" if live < min_window else
                 f"{INCONCLUSIVE_VERDICT} — camera range zero under an "
                 f"undirected probe")
    # NOT `not v["instrument_findings"]`. That idiom is what D6 was: a
    # verdict computed from the ABSENCE of findings is one reordering or
    # one early return away from certifying the thing it was meant to
    # reject, and this function's whole job is to reject. A camera-static
    # axis is disqualifying on its own terms, so say so directly.
    v["passed"] = False
    v["verdict"] = label
    return v


def resolve_odometer_axis(progress_cfg: dict, rx: int,
                           ry: int) -> tuple[int, int]:
    """Axis index (0=x, 1=y) and sign to trace under --odometer.

    A profile with `progress: {source: odometer, axis: ...}` has already
    committed to the axis go_explore_solve.py will actually search on
    (go_explore_solve.py:1908 trusts this field verbatim, sign included) —
    the gate must probe THAT axis, not whichever raw axis happens to show
    more incidental pixel range in this short forward-hold probe. Range
    auto-pick is only a fallback for the undeclared exploratory case:
    sizing up the odometer on a profile that still reads a RAM byte.
    """
    if str(progress_cfg.get("source", "")).lower() == "odometer":
        axis_raw = str(progress_cfg.get("axis", "x")).lower()
        sign = -1 if axis_raw.startswith("-") else 1
        axis = axis_raw[1:] if axis_raw and axis_raw[0] in "+-" else axis_raw
        if axis in ("x", "y"):
            return (0 if axis == "x" else 1), sign
    return (0 if rx >= ry else 1), 1


def resolve_forward_index(space: list, forward: str) -> int:
    """Index into action_space whose action is the bare singleton
    `[forward]` (e.g. `["right"]`).

    Raises SystemExit on no match rather than silently falling back to
    an unrelated action — a typo'd or unsupported --forward used to
    resolve to a hardcoded index 1 with no warning, running the whole
    probe under whatever action happened to sit there.
    """
    matches = [i for i, a in enumerate(space) if a == [forward]]
    if not matches:
        options = [a[0] for a in space if len(a) == 1]
        raise SystemExit(
            f"--forward {forward!r} matches no singleton action in "
            f"the profile's action_space; options: {options}")
    return matches[0]


def assess(trace: list[int], lives_at_start: int | None,
           has_high_byte: bool, raw_direction: int | None = None,
           min_window: int | None = None, directed: bool = True) -> dict:
    """PURE verdict over one long forward-hold trace.

    `min_window` is the live-window floor below which a SHORTFALL of
    distinct values is not a supportable claim; defaults to
    MIN_ASSESSABLE_STEPS, and `0` reproduces every verdict banked before
    2026-08-26 exactly, which is how the sweep diffs old against new.

    `directed` says the probe commanded forward motion. An UNDIRECTED
    probe cannot report a shortfall as a fault at all, at any window
    length, and the roster proves why: bionic_commando's odometer shows
    122 distinct levels over a 1200-step forward hold and 21 over a
    1200-step uniform-random rollout. Twenty-one levels in 1200 steps is
    a fact about a policy that wandered, not about the instrument — the
    same confound as the truncation-order defect, wearing the other hat.
    So an undirected probe can only ever ADD evidence (a positive
    >= MIN_DISTINCT demonstration, and a longer window); it can never
    subtract a certification.
    """
    # INSTRUMENT findings mean the signal itself is unusable — the profile
    # must not drive a search. BEHAVIOUR findings mean the signal is sound
    # and the GAME stops, which is a finding about the game and not a
    # reason to reject the profile. Conflating them would have rejected
    # Contra, whose progress pair is the best of any game here (163
    # distinct values over 0..635) and whose flat tail is the real,
    # receipted fixed-camera wall at gx 3072.
    #
    # INCONCLUSIVE findings are the third class, added 2026-08-26: the
    # measurement was attempted and the window could not carry it. They
    # block (nothing here is certified) but they are NOT a fault report,
    # and reporting them as one is what excluded Contra.
    instrument: list[str] = []
    behaviour: list[str] = []
    inconclusive: list[str] = []
    findings = instrument
    min_window = (MIN_ASSESSABLE_STEPS if min_window is None
                  else int(min_window))
    n = len(trace)
    distinct = len(set(trace))
    # raw_direction is the SIGNED net motion of the un-rebased odometer
    # reading (last - first, under the profile's declared/default axis
    # sign) — the ONE thing the rebase below (odometer callers pass
    # trace shifted to its own min) throws away. Without it, a raw axis
    # that counts DOWN going forward (1942's vertical scroll register)
    # produces a trace indistinguishable from a healthy increasing one:
    # distinct/max/tail-flatness are all order-blind. Solver._xram
    # clamps negative reads to 0 under the same sign assumption, so a
    # negative raw_direction means the solver would see a permanently
    # flat 0 the whole run, however clean this trace looks.
    if raw_direction is not None and raw_direction < 0:
        findings.append(
            "raw odometer reading net DECREASES while holding forward "
            "under the current axis sign — Solver._xram clamps negative "
            "reads to 0, so the solver would see a flat signal no matter "
            "how this rebased trace looks; declare a leading '-' on "
            "solve.progress.axis to match the hardware direction")
    # THE ASYMMETRY, and it is the whole fix. Observing >= MIN_DISTINCT
    # levels is a POSITIVE demonstration: 116 distinct in Rygar's 138
    # live steps proves the resolution is there, and no window floor can
    # take that back. Observing FEWER only demonstrates coarseness if the
    # window was long enough that a resolving signal would have shown
    # them — Contra's 20-in-69 does not, and the same 20-in-69 shape at
    # 1200 steps does. So the floor gates one direction and not the
    # other, and cannot convert a FAIL into a PASS.
    if distinct < MIN_DISTINCT:
        if n >= min_window and directed:
            findings.append(
                f"only {distinct} distinct values in {n} steps "
                f"(< {MIN_DISTINCT}) — too coarse to be a search gradient")
        elif not directed:
            inconclusive.append(
                f"only {distinct} distinct values over {n} live steps, but "
                f"the probe commanded no direction — under undirected play "
                f"a shortfall is a fact about the policy, not the "
                f"instrument (bionic_commando: 122 distinct held forward, "
                f"21 under uniform-random, same 1200 steps). Re-probe with "
                f"--probe hold to make this a claim about the signal")
        else:
            inconclusive.append(
                f"only {distinct} distinct values, but the live window is "
                f"{n} steps — under the {min_window}-step floor "
                f"(MIN_ASSESSABLE_STEPS: the longest any SIGNAL SOUND "
                f"profile on this roster took to accumulate "
                f"{MIN_DISTINCT} levels). A {n}-sample window cannot "
                f"demonstrate a {MIN_DISTINCT}-distinct threshold, so this "
                f"number measures how fast the probe died, not the "
                f"signal's resolution — re-probe for a longer live window "
                f"(--probe random) before calling this signal anything")
    if not has_high_byte and max(trace, default=0) >= 200:
        findings.append(
            "reaches >=200 with no paired high byte — a single byte wraps, "
            "and the solver reads it raw, so progress past one wrap is "
            "invisible")
    tail = trace[int(n * 0.75):]
    if tail and len(set(tail)) <= 1:
        behaviour.append(
            "flat for the last quarter of the roll — the signal stops "
            "moving, which is either saturation or the agent no longer "
            "advancing; either way the probe's net-forward number does not "
            "describe what the solver will see")
    if lives_at_start is not None and lives_at_start <= 0:
        findings.append(
            f"death byte reads {lives_at_start} at the start state — a "
            f"decrement underflows and no death can be detected")
    # Only instrument faults and unsupportable windows block. A sound
    # signal that flattens is reporting a wall, which is exactly what it
    # is for. Kept as one list so the vacuous-gate scanner still sees
    # this site (scripts/anti_vacuity_scan.py) and
    # tests/test_anti_vacuity_gates.py keeps re-proving both polarities.
    blocking = instrument + inconclusive
    return {
        "steps": n, "distinct": distinct,
        "min": min(trace) if trace else None,
        "max": max(trace) if trace else None,
        "tail_distinct": len(set(tail)) if tail else 0,
        "has_high_byte": has_high_byte,
        "lives_at_start": lives_at_start,
        "min_window": min_window,
        "window_supports_shortfall": n >= min_window,
        "steps_to_min_distinct": steps_to_distinct(trace),
        "passed": not blocking,
        "instrument_findings": instrument,
        "inconclusive_findings": inconclusive,
        "behaviour_findings": behaviour,
        "verdict": verdict_label(instrument, inconclusive, behaviour),
    }


def assess_hold(*, xy: list | None = None, ram_trace: list | None = None,
                lives_trace: list | None = None, lives0: int | None = None,
                churn: list[int] | None = None,
                oam_changed: list[bool] | None = None,
                progress_cfg: dict | None = None,
                has_high_byte: bool = False,
                requested_steps: int | None = None,
                min_window: int | None = None,
                directed: bool = True) -> dict:
    """EVERYTHING that happens between the last emulator step and the
    printed verdict, as one pure function over recorded traces.

    This exists because it used to live inline in `main()`, where no test
    could reach it. Verification of the D5/D6 work found the consequence:
    deleting either `xy = xy[:keep]` or `trace = trace[:keep]` restored
    the death-blind hold and failed nothing, and the `rx == 0 and ry == 0`
    branch that D6 fixed was likewise reachable only by booting a ROM.
    `main()` below now collects traces and calls this; it holds no
    decision of its own.

    Pass `xy` (odometer point list) or `ram_trace` (paired RAM reads),
    not both.

    `directed=False` says the probe commanded no direction (the random
    probe), so the raw-odometer-sign check is DISARMED rather than fed a
    number that cannot answer it: under uniform-random actions the sign
    of net motion is a property of the dice, and a 1942-class axis-sign
    fault would be reported or missed at random. Refusing to answer is
    the same discipline `first_stasis_index` already uses when a profile
    is too quiet to judge.
    """
    assert (xy is None) != (ram_trace is None), \
        "assess_hold takes exactly one of xy= / ram_trace="
    progress_cfg = progress_cfg or {}
    n_raw = len(xy if xy is not None else ram_trace)
    requested = requested_steps if requested_steps is not None else n_raw

    keep = truncate_at_exhaustion(n_raw, lives_trace, lives0, churn)
    dropped = n_raw - keep
    stasis_idx, stasis_reason = (first_stasis_index(list(churn))
                                 if churn else (None, "no churn recorded"))
    lives_idx = (first_exhaustion_index(lives_trace, lives0)
                 if lives_trace is not None and lives0 is not None else None)

    if xy is not None:
        xy = xy[:keep]
        oam_churn = sum(oam_changed[:keep]) if oam_changed else 0
        rx = max(p[0] for p in xy) - min(p[0] for p in xy) if xy else 0
        ry = max(p[1] for p in xy) - min(p[1] for p in xy) if xy else 0
        axis, sign = resolve_odometer_axis(progress_cfg, rx, ry)
        signed = [p[axis] * sign for p in xy]
        base = min(signed) if signed else 0
        trace = [int(s - base) for s in signed]
        raw_direction = ((signed[-1] - signed[0])
                         if (signed and directed) else None)
        v = assess(trace, lives0, True, raw_direction,
                   min_window=min_window, directed=directed)
        if rx == 0 and ry == 0:
            v = note_camera_static(v, oam_churn, requested,
                                   live_steps=len(xy), min_window=min_window,
                                   directed=directed)
        v["oam_churn"] = oam_churn
        v["odometer_range"] = {"x": rx, "y": ry}
        v["axis"] = "xy"[axis]
    else:
        v = assess(list(ram_trace[:keep]), lives0, has_high_byte,
                   min_window=min_window, directed=directed)

    # THE D1 CLASS, reported rather than absorbed. A surface that froze
    # while the declared death observable said nothing means the profile
    # cannot see its own terminal states — the same defect this file's
    # docstring already records for Ninja Gaiden's death byte, and an
    # instrument fault, not a fact about the game.
    if stasis_idx is not None and lives_idx is None:
        v["instrument_findings"].append(
            f"the hold ran into a frozen surface with NO death recorded by "
            f"the declared lives byte ({stasis_reason}) — the death "
            f"observable is blind to this terminal state (defect D1 class), "
            f"so everything after step {stasis_idx} was a non-interactive "
            f"screen this gate cannot certify")
        v["passed"] = False
        v["verdict"] = "SIGNAL UNUSABLE — death observable blind to a frozen tail"

    v["requested_steps"] = requested
    v["dropped_tail_steps"] = dropped
    v["exhaustion"] = {
        "lives_index": lives_idx,
        "stasis_index": stasis_idx,
        "stasis_reason": stasis_reason,
        "kept_steps": keep,
    }
    return v

# ==========================================================================
# THE PROBE. What drives the emulator while the traces above are recorded.
#
# `hold` is the original and stays the default so every verdict banked
# before 2026-08-26 remains reproducible byte for byte. It is also a
# measurement of the PROBE as much as of the game: an undodging scripted
# forward hold from a start state with `lives=1` walks into the first
# hazard and dies, and everything after that is a game-over screen the D5
# truncation (correctly) throws away. Rygar's "138 live steps" is that
# artifact — uniform-random survives a median 677 steps on the same start
# state and the solver runs 3,865-4,000 actions in ONE life.
#
# `random` exists because the live window is the binding constraint on
# what this gate can conclude (see MIN_ASSESSABLE_STEPS): a verdict about
# a signal's resolution needs a window, and the cheapest game-agnostic way
# to buy one is to stop steering into the hazard. It is NOT strictly
# better — it disarms the axis-sign check, which needs a commanded
# direction — so it is an alternative, never a replacement.
# ==========================================================================

PROBES = ("hold", "random")

#: Checks that a given probe structurally cannot make. Reported on the
#: verdict so a `random`-probe PASS can never be read as certifying
#: something the random probe never looked at.
PROBE_DISARMS = {
    "hold": (),
    "random": (
        "axis-sign (raw odometer direction): a uniform-random policy "
        "commands no direction, so net motion carries no information "
        "about whether the declared axis sign matches the hardware — "
        "run --probe hold for that check",
        "resolution SHORTFALL and camera-zero-range as FAULTS: an "
        "undirected policy that shows few levels may simply not have "
        "gone anywhere (bionic_commando: 122 distinct held forward, 21 "
        "under uniform-random over the same 1200 steps). This probe can "
        "only ADD a positive demonstration or a longer window; a "
        "shortfall it sees is reported INCONCLUSIVE, never UNUSABLE",
    ),
}


def probe_actions(n_actions: int, steps: int, forward_index: int,
                  probe: str = "hold", seed: int = 0) -> list[int]:
    """The action-index sequence a probe drives, as a PURE function of
    its arguments.

    Pure and seeded on purpose: the probe is the part of this gate that
    used to be unreachable from a test, and "the probe died at step 69"
    is now a claim a test can construct rather than one only a ROM can
    produce.
    """
    if probe == "hold":
        return [forward_index] * steps
    if probe == "random":
        import random
        rng = random.Random(seed)
        return [rng.randrange(n_actions) for _ in range(steps)]
    raise SystemExit(f"unknown probe {probe!r}; choose from {list(PROBES)}")


def select_longest_live_window(keeps: list[int]) -> int:
    """Index of the episode with the longest LIVE window; ties go to the
    earliest seed.

    Selecting on live-window length rather than on `distinct` is
    deliberate. `distinct` is the statistic under test, and picking the
    best-of-k on it would be winner's curse dressed up as a measurement.
    Window length is a different statistic and is exactly the quantity
    the probe defect is about, so selecting on it targets the defect
    without touching the verdict's own evidence. Every episode's window
    is recorded on the receipt either way.
    """
    if not keeps:
        raise ValueError("no episodes to select from")
    best = 0
    for i, k in enumerate(keeps):
        if k > keeps[best]:
            best = i
    return best


def _open_pool(prof: dict, rom: str, odometer: bool):
    """Boot a one-worker pool on the profile's ROM."""
    import nes_core
    pool = nes_core.Pool(rom_path=str(REPO / rom), num_workers=1,
                         frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    if odometer:
        pool.set_odometer_enabled(True)
    return pool


def _rollout(pool, prof: dict, bitmasks, actions: list[int], *,
             odometer: bool, lives_addr, lo, hi) -> dict:
    """One episode: restore the start state, drive `actions`, record
    traces. DECIDES NOTHING — every judgement lives in assess_hold().
    """
    import numpy as np
    # MUST precede load_worker_state, and is NOT redundant with it.
    # Restoring into a Pool that has never stepped leaves the mapper's
    # CPU-cycle counter at 0, which collides with MMC1's post-restore
    # `last_register_write_cycle = u64::MAX` sentinel (MAX + 1 wraps to
    # 0) and makes the RMW consecutive-write filter silently eat the
    # first bank-select write after the load — wrong PRG/CHR bank, dead
    # game, flat odometer. reset_all() runs a frame first, so the
    # counter is far off zero by the time the restored ROM writes.
    # Same ordering go_explore_solve.py and discover_observables.py
    # already use. See docs/research/ASM_CPU_STATUS_2026-08-25.md.
    pool.reset_all()
    pool.load_worker_state(0, (REPO / prof["start_state_path"]).read_bytes())

    first = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
    lives0 = int(first[lives_addr]) if lives_addr is not None else None
    prev_ram = np.frombuffer(bytes(first), dtype=np.uint8).astype(np.int16)

    xy: list = []
    oam_changed: list = []
    lives_trace: list = []
    churn: list = []
    trace: list = []
    prev_oam = None
    for ai in actions:
        a = np.array([bitmasks[ai]], dtype=np.uint8)
        ram = pool.step_all(a)[0][2]
        cur = np.frombuffer(bytes(ram), dtype=np.uint8).astype(np.int16)
        # Per-step RAM churn: the surface measurement the stasis test needs,
        # because the declared lives byte can be blind to a terminal state
        # (defect D1). One 2 KB compare per step.
        churn.append(int(np.count_nonzero(cur != prev_ram)))
        prev_ram = cur
        lives_trace.append(int(ram[lives_addr]) if lives_addr is not None else None)
        if odometer:
            xy.append(list(pool.get_odometer_per_worker()[0]))
            oam = bytes(pool.peek_oam(0))
            oam_changed.append(prev_oam is not None and oam != prev_oam)
            prev_oam = oam
        else:
            trace.append(int(ram[lo]) + (int(ram[hi]) << 8 if hi is not None else 0))
    return {"xy": xy, "oam_changed": oam_changed, "lives_trace": lives_trace,
            "churn": churn, "trace": trace, "lives0": lives0}


def run_probe(profile: str, *, steps: int = 1200, forward: str = "right",
              odometer: bool = False, probe: str = "hold",
              episodes: int = 5, seed: int = 0) -> dict:
    """Boot the ROM and record `episodes` rollouts of the chosen probe.

    Returns the raw traces plus the per-episode live-window lengths and
    the index this gate will assess. Nothing here is a verdict.
    """
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import yaml
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load((REPO / profile).read_text())
    solve = prof.get("solve") or {}
    progress_cfg = solve.get("progress") or {}
    lo = progress_cfg.get("lo")
    hi = progress_cfg.get("hi")
    lives_addr = solve.get("lives")
    rom = solve.get("rom") or prof.get("rom_path")
    if lo is None and not odometer:
        raise SystemExit("profile has no solve.progress.lo")

    space = prof["action_space"]
    fwd = resolve_forward_index(space, forward)
    bm = action_space_to_bitmasks(space)
    n_episodes = 1 if probe == "hold" else max(1, int(episodes))

    pool = _open_pool(prof, rom, odometer)
    eps = []
    for k in range(n_episodes):
        actions = probe_actions(len(space), steps, fwd, probe, seed + k)
        r = _rollout(pool, prof, bm, actions, odometer=odometer,
                     lives_addr=lives_addr, lo=lo, hi=hi)
        r["seed"] = seed + k
        r["live_steps"] = truncate_at_exhaustion(
            steps, r["lives_trace"] if lives_addr is not None else None,
            r["lives0"], r["churn"])
        eps.append(r)
    chosen = select_longest_live_window([e["live_steps"] for e in eps])
    return {"profile": profile, "probe": probe, "steps": steps,
            "forward": forward, "odometer": odometer,
            "progress_cfg": progress_cfg, "has_high_byte": hi is not None,
            "episodes": eps, "chosen": chosen,
            "live_steps_per_episode": [e["live_steps"] for e in eps]}


def assess_probe(collected: dict, *, min_window: int | None = None) -> dict:
    """Verdict over whichever episode `run_probe` selected."""
    e = collected["episodes"][collected["chosen"]]
    common = dict(lives_trace=e["lives_trace"] if e["lives0"] is not None else None,
                  lives0=e["lives0"], churn=e["churn"],
                  requested_steps=collected["steps"], min_window=min_window)
    if collected["odometer"]:
        v = assess_hold(xy=[tuple(p) for p in e["xy"]],
                        oam_changed=e["oam_changed"],
                        progress_cfg=collected["progress_cfg"],
                        # The axis-sign check needs a COMMANDED direction.
                        # A random probe has none, so it is disarmed by
                        # construction rather than answered with noise.
                        directed=collected["probe"] == "hold",
                        **common)
    else:
        v = assess_hold(ram_trace=e["trace"],
                        has_high_byte=collected["has_high_byte"], **common)
    v["probe"] = collected["probe"]
    v["probe_seed"] = e["seed"]
    v["live_steps_per_episode"] = collected["live_steps_per_episode"]
    v["disarmed_checks"] = list(PROBE_DISARMS.get(collected["probe"], ()))
    return v


def exit_code(v: dict) -> int:
    """0 PASS / 1 FAIL (SIGNAL UNUSABLE) / 2 VOID (INCONCLUSIVE).

    The VOID-vs-FAIL distinction, in the one place a caller can act on
    it. A script that reads only "nonzero" is unchanged; a script that
    wants to tell "this instrument is broken" from "this probe never got
    far enough to say" now can.
    """
    if v["passed"]:
        return 0
    return 2 if v["inconclusive_findings"] and not v["instrument_findings"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--forward", default="right")
    ap.add_argument("--odometer", action="store_true",
                    help="trace the PPU scroll odometer (dominant axis) "
                         "instead of a discovered RAM byte — the "
                         "hardware-surface progress signal for games "
                         "whose RAM bytes failed this gate")
    ap.add_argument("--probe", default="hold", choices=list(PROBES),
                    help="hold: the original scripted forward hold "
                         "(default, and what every banked verdict was "
                         "measured with). random: uniform-random actions, "
                         "--episodes rollouts, assess the one with the "
                         "longest live window — buys a window on games "
                         "where the forward hold walks straight into a "
                         "hazard, at the cost of the axis-sign check")
    ap.add_argument("--episodes", type=int, default=5,
                    help="rollouts for a stochastic probe (ignored by "
                         "--probe hold, which is deterministic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-window", type=int, default=None,
                    help=f"live steps required before a SHORTFALL of "
                         f"distinct values may be reported as an "
                         f"instrument fault (default "
                         f"{MIN_ASSESSABLE_STEPS}; pass 0 to reproduce "
                         f"pre-2026-08-26 verdicts exactly)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    collected = run_probe(args.profile, steps=args.steps,
                          forward=args.forward, odometer=args.odometer,
                          probe=args.probe, episodes=args.episodes,
                          seed=args.seed)
    v = assess_probe(collected, min_window=args.min_window)

    if args.odometer:
        print(f"odometer trace: axis={v['axis']} "
              f"(range x={v['odometer_range']['x']}, "
              f"y={v['odometer_range']['y']}) "
              f"oam_churn={v['oam_churn']}/{max(v['steps'] - 1, 0)}")
    if args.probe != "hold":
        print(f"probe={args.probe} seed={v['probe_seed']} — live windows "
              f"per episode: {v['live_steps_per_episode']} "
              f"(assessing the longest)")
        for d in v["disarmed_checks"]:
            print(f"  [DISARMED] {d}")
    dropped = v["dropped_tail_steps"]
    if dropped:
        print(f"[progress_signal_gate] hold ran off the end of live play at "
              f"step {v['exhaustion']['kept_steps']} of {args.steps} — "
              f"dropping {dropped} trailing steps (game-over/attract tail) "
              f"before assessing; lives_index="
              f"{v['exhaustion']['lives_index']}, stasis_index="
              f"{v['exhaustion']['stasis_index']} "
              f"({v['exhaustion']['stasis_reason']})")

    status = "PASS" if v["passed"] else ("VOID" if exit_code(v) == 2 else "FAIL")
    print(f"progress-signal gate: {status} — "
          f"{v['verdict']}  ({args.profile})")
    print(f"  {v['steps']} steps, {v['distinct']} distinct, "
          f"range {v['min']}..{v['max']}, high byte={v['has_high_byte']}, "
          f"lives@start={v['lives_at_start']}")
    for f in v["instrument_findings"]:
        print(f"  [INSTRUMENT]   {f}")
    for f in v["inconclusive_findings"]:
        print(f"  [INCONCLUSIVE] {f}")
    for f in v["behaviour_findings"]:
        print(f"  [BEHAVIOUR]    {f}")
    if dropped:
        print(f"  [D5] {dropped} trailing steps of the requested "
              f"{args.steps} were dropped — the hold ran off the end of "
              f"live play (lives byte exhausted and/or the surface froze "
              f"and never restarted), so everything above is assessed on "
              f"the {v['steps']} steps before that, not the full hold.")
    if v["passed"] and not v["behaviour_findings"]:
        print("  signal is usable: enough resolution, no unpaired wrap, "
              "still moving late, death detectable")
    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"profile": args.profile,
                                 "odometer": bool(args.odometer),
                                 **v}, indent=2) + "\n")
    return exit_code(v)


if __name__ == "__main__":
    raise SystemExit(main())
