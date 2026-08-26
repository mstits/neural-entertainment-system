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
    trace = []
    if args.odometer:
        xy = []
        oam_churn = 0
        prev_oam = None
        for _ in range(args.steps):
            pool.step_all(a)
            xy.append(pool.get_odometer_per_worker()[0])
            oam = bytes(pool.peek_oam(0))
            if prev_oam is not None and oam != prev_oam:
                oam_churn += 1
            prev_oam = oam
        rx = max(p[0] for p in xy) - min(p[0] for p in xy)
        ry = max(p[1] for p in xy) - min(p[1] for p in xy)
        axis, sign = resolve_odometer_axis(solve.get("progress") or {}, rx, ry)
        signed = [p[axis] * sign for p in xy]
        base = min(signed)
        trace = [int(s - base) for s in signed]
        print(f"odometer trace: axis={'xy'[axis]} "
              f"(range x={rx}, y={ry}) oam_churn={oam_churn}/{args.steps-1}")
        # i64 integral: no wrap exists, so run assess as a paired signal.
        # `signed[-1] - signed[0]` is the net motion UNDER the resolved
        # sign, before the `- base` shift above folds it into a
        # nonnegative trace — that shift preserves order but assess()
        # never looks at order, only set-based distinct/tail counts, so
        # a raw axis that runs backward under this sign (undeclared or
        # wrong-signed `progress.axis`) renders exactly as clean a trace
        # as a genuinely forward-increasing one without this.
        v = assess(trace, lives0, True, signed[-1] - signed[0])
        # The odometer measures the CAMERA, and the build is certified
        # (scripts/odometer_cert.py) before this gate runs. A flat
        # odometer therefore reports a static camera — a fact about the
        # game under this driver, never about the instrument. Reclassify
        # the RAM-era coarseness fault and attach the OAM cross-check so
        # the verdict says whether the agent was even alive.
        if rx == 0 and ry == 0:
            v["instrument_findings"] = [
                f for f in v["instrument_findings"] if "too coarse" not in f]
            agent = ("agent active (OAM moving)" if oam_churn > args.steps // 4
                     else "agent inert (OAM static too)")
            v["behaviour_findings"].append(
                f"camera never moved over {args.steps} steps; {agent} — "
                f"the game does not scroll under this driver, which is a "
                f"skill/route wall, not an instrument fault")
            v["passed"] = not v["instrument_findings"]
            v["verdict"] = ("SIGNAL SOUND — camera static, " + agent
                            if v["passed"] else v["verdict"])
        v["oam_churn"] = oam_churn
    else:
        for _ in range(args.steps):
            ram = pool.step_all(a)[0][2]
            trace.append(int(ram[lo]) + (int(ram[hi]) << 8 if hi is not None else 0))
        v = assess(trace, lives0, hi is not None)
    print(f"progress-signal gate: {'PASS' if v['passed'] else 'FAIL'} — "
          f"{v['verdict']}  ({args.profile})")
    print(f"  {v['steps']} steps, {v['distinct']} distinct, "
          f"range {v['min']}..{v['max']}, high byte={v['has_high_byte']}, "
          f"lives@start={v['lives_at_start']}")
    for f in v["instrument_findings"]:
        print(f"  [INSTRUMENT] {f}")
    for f in v["behaviour_findings"]:
        print(f"  [BEHAVIOUR]  {f}")
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
