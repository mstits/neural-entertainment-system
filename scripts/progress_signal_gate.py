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

    .venv/bin/python scripts/progress_signal_gate.py --profile configs/rygar.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MIN_DISTINCT = 32          # fewer levels than this is not a gradient
MIN_TAIL_FRACTION = 0.25   # progress must still be moving late in a roll


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


def note_camera_static(v: dict, oam_churn: int, steps: int) -> dict:
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
    """
    agent = ("agent active (OAM moving)" if oam_churn > steps // 4
             else "agent inert (OAM static too)")
    v["instrument_findings"].append(
        f"camera never moved over {steps} steps under the odometer "
        f"driver ({agent}) — a zero-range axis cannot express progress "
        f"regardless of agent activity; this profile has not been "
        f"demonstrated capable of returning a positive on this axis and "
        f"must not drive a search with it")
    # NOT `not v["instrument_findings"]`. That idiom is what D6 was: a
    # verdict computed from the ABSENCE of findings is one reordering or
    # one early return away from certifying the thing it was meant to
    # reject, and this function's whole job is to reject. A camera-static
    # axis is disqualifying on its own terms, so say so directly.
    v["passed"] = False
    v["verdict"] = "SIGNAL UNUSABLE — camera static"
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
           has_high_byte: bool, raw_direction: int | None = None) -> dict:
    """PURE verdict over one long forward-hold trace."""
    # INSTRUMENT findings mean the signal itself is unusable — the profile
    # must not drive a search. BEHAVIOUR findings mean the signal is sound
    # and the GAME stops, which is a finding about the game and not a
    # reason to reject the profile. Conflating them would have rejected
    # Contra, whose progress pair is the best of any game here (163
    # distinct values over 0..635) and whose flat tail is the real,
    # receipted fixed-camera wall at gx 3072.
    instrument: list[str] = []
    behaviour: list[str] = []
    findings = instrument
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
    if distinct < MIN_DISTINCT:
        findings.append(
            f"only {distinct} distinct values in {n} steps (< {MIN_DISTINCT}) "
            f"— too coarse to be a search gradient")
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
    return {
        "steps": n, "distinct": distinct,
        "min": min(trace) if trace else None,
        "max": max(trace) if trace else None,
        "tail_distinct": len(set(tail)) if tail else 0,
        "has_high_byte": has_high_byte,
        "lives_at_start": lives_at_start,
        # Only instrument faults block. A sound signal that flattens is
        # reporting a wall, which is exactly what it is for.
        "passed": not instrument,
        "instrument_findings": instrument,
        "behaviour_findings": behaviour,
        "verdict": ("SIGNAL UNUSABLE" if instrument
                    else "SIGNAL SOUND — game stops" if behaviour
                    else "SIGNAL SOUND — still advancing"),
    }


def assess_hold(*, xy: list | None = None, ram_trace: list | None = None,
                lives_trace: list | None = None, lives0: int | None = None,
                churn: list[int] | None = None,
                oam_changed: list[bool] | None = None,
                progress_cfg: dict | None = None,
                has_high_byte: bool = False,
                requested_steps: int | None = None) -> dict:
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
        v = assess(trace, lives0, True,
                   (signed[-1] - signed[0]) if signed else None)
        if rx == 0 and ry == 0:
            v = note_camera_static(v, oam_churn, requested)
        v["oam_churn"] = oam_churn
        v["odometer_range"] = {"x": rx, "y": ry}
        v["axis"] = "xy"[axis]
    else:
        v = assess(list(ram_trace[:keep]), lives0, has_high_byte)

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
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, yaml, nes_core
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load((REPO / args.profile).read_text())
    solve = prof.get("solve") or {}
    lo = (solve.get("progress") or {}).get("lo")
    hi = (solve.get("progress") or {}).get("hi")
    lives_addr = solve.get("lives")
    rom = solve.get("rom") or prof.get("rom_path")
    if lo is None and not args.odometer:
        raise SystemExit("profile has no solve.progress.lo")

    space = prof["action_space"]
    idx = resolve_forward_index(space, args.forward)
    bm = action_space_to_bitmasks(space)
    pool = nes_core.Pool(rom_path=str(REPO / rom), num_workers=1,
                         frame_skip=int(prof.get("frame_skip", 4)))
    pool.set_headless(True); pool.set_skip_preprocess(True)
    if args.odometer:
        pool.set_odometer_enabled(True)
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
    a = np.array([bm[idx]], dtype=np.uint8)

    first = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
    lives0 = int(first[lives_addr]) if lives_addr is not None else None
    prev_ram = np.frombuffer(bytes(first), dtype=np.uint8).astype(np.int16)

    # ---- collect, decide nothing ------------------------------------------
    # Every judgement below the loop lives in assess_hold(), which is a pure
    # function over these traces and is driven directly by
    # tests/test_progress_signal_gate.py. main() must stay dumb: an inline
    # decision here is one no test can reach, which is how the D5 truncation
    # and the D6 camera-static branch both shipped unguarded.
    xy: list = []
    oam_changed: list = []
    lives_trace: list = []
    churn: list = []
    trace: list = []
    prev_oam = None
    for _ in range(args.steps):
        ram = pool.step_all(a)[0][2]
        cur = np.frombuffer(bytes(ram), dtype=np.uint8).astype(np.int16)
        # Per-step RAM churn: the surface measurement the stasis test needs,
        # because the declared lives byte can be blind to a terminal state
        # (defect D1). One 2 KB compare per step.
        churn.append(int(np.count_nonzero(cur != prev_ram)))
        prev_ram = cur
        lives_trace.append(int(ram[lives_addr]) if lives_addr is not None else None)
        if args.odometer:
            xy.append(pool.get_odometer_per_worker()[0])
            oam = bytes(pool.peek_oam(0))
            oam_changed.append(prev_oam is not None and oam != prev_oam)
            prev_oam = oam
        else:
            trace.append(int(ram[lo]) + (int(ram[hi]) << 8 if hi is not None else 0))

    if args.odometer:
        v = assess_hold(xy=xy, lives_trace=lives_trace, lives0=lives0,
                        churn=churn, oam_changed=oam_changed,
                        progress_cfg=solve.get("progress") or {},
                        requested_steps=args.steps)
        print(f"odometer trace: axis={v['axis']} "
              f"(range x={v['odometer_range']['x']}, "
              f"y={v['odometer_range']['y']}) "
              f"oam_churn={v['oam_churn']}/{max(v['steps'] - 1, 0)}")
    else:
        v = assess_hold(ram_trace=trace, lives_trace=lives_trace, lives0=lives0,
                        churn=churn, has_high_byte=hi is not None,
                        requested_steps=args.steps)
    dropped = v["dropped_tail_steps"]
    if dropped:
        print(f"[progress_signal_gate] hold ran off the end of live play at "
              f"step {v['exhaustion']['kept_steps']} of {args.steps} — "
              f"dropping {dropped} trailing steps (game-over/attract tail) "
              f"before assessing; lives_index="
              f"{v['exhaustion']['lives_index']}, stasis_index="
              f"{v['exhaustion']['stasis_index']} "
              f"({v['exhaustion']['stasis_reason']})")

    print(f"progress-signal gate: {'PASS' if v['passed'] else 'FAIL'} — "
          f"{v['verdict']}  ({args.profile})")
    print(f"  {v['steps']} steps, {v['distinct']} distinct, "
          f"range {v['min']}..{v['max']}, high byte={v['has_high_byte']}, "
          f"lives@start={v['lives_at_start']}")
    for f in v["instrument_findings"]:
        print(f"  [INSTRUMENT] {f}")
    for f in v["behaviour_findings"]:
        print(f"  [BEHAVIOUR]  {f}")
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
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
