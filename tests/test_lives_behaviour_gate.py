"""Behavioural validation of a lives nomination
(docs/research/FALSE_DEATH_FANOUT_2026-08-26.md §6).

The fanout re-nominated `solve.lives` on eleven zero-start profiles. The
diagnosis held 11/11 and the static guard did its job, but THREE of the
tool's new top picks passed every static gate in `lives_from_death_drives`
and were caught only BEHAVIOURALLY, by driving the game and watching the
byte move:

  * Bad Dudes $00CD — an attack-animation counter cycling 2 -> 0 -> 2,
    five to eighteen times per 700-step drive, first toggle at step 3.
  * Ninja Gaiden $0386 — empties while the PPU scroll odometer keeps
    climbing 94 -> 426 px, i.e. the agent is demonstrably still alive.
  * Journey to Silius $0135 — drops within four steps of ANY rightward
    hold in a forced scroller: it fires on movement onset, not on death.

A wrong lives byte does not degrade death detection, it INVERTS it
(`GenericGame.is_dead` reads `(start - cur) % 256 in 1..8`), so each of
these would have fired "death" on ordinary play — two of them measurably
collapsed a search (1096 -> 24 cells, 774 -> 2 cells) when trial-wired.

These tests drive the gate on SYNTHETIC traces only — no ROM, no Pool,
no emulation — matching the discipline `tests/test_discover_observables.py`
and `tests/test_fight_gate.py` already use. Each of the three measured
failure modes gets a test, a clean stock that MUST pass gets one, and so
does every veto that keeps a real death from being read as liveness.

The thresholds these assert against were set by measurement on banked
arms from the six named profiles (three false positives, three verified
stocks) — see the constants' own comments in discover_observables.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.discover_observables import (
    ALIVE_MAX_STALL,
    ALIVE_MIN_PX,
    ALIVE_WITNESS_STEPS,
    MOVEMENT_ONSET_STEPS,
    OSC_MAX_CYCLES,
    OSC_RECOVER_STEPS,
    _arrival_at_zero,
    _first_drop_step,
    _longest_stall,
    _recovery_cycles,
    _wrap_levels,
    behavioural_lives_verdict,
    emit_solve_yaml,
    screen_lives_behaviourally,
)

RAM = 64                 # narrow synthetic RAM; the gate only indexes columns
LIVES = 0x0A             # the byte under test
OTHER = 0x0B             # a second nomination, for the ranking tests
N = 320                  # BEHAVIOUR_N, the arm length the tool drives


# ---------------------------------------------------------------------------
# Synthetic arms.
# ---------------------------------------------------------------------------
def _col(n: int = N, level: int = 2) -> np.ndarray:
    return np.full(n, level, dtype=np.uint8)


def _spend(n: int = N, start: int = 2, at=(62, 135)) -> np.ndarray:
    """A stock: `start`, stepping down by one at each row in `at`."""
    col = np.empty(n, dtype=np.uint8)
    v = start
    at = set(at)
    for t in range(n):
        if t in at:
            v -= 1
        col[t] = v % 256
    return col


def _toggle(n: int = N, level: int = 2, *, first: int = 3, period: int = 15,
            width: int = 2) -> np.ndarray:
    """Bad Dudes $00CD: parked at `level`, dipping to 0 and bouncing back."""
    col = np.full(n, level, dtype=np.uint8)
    for t in range(first, n, period):
        col[t:min(t + width, n)] = 0
    return col


def _empties_at(n: int = N, step: int = 32, start: int = 1) -> np.ndarray:
    col = np.full(n, start, dtype=np.uint8)
    col[step:] = 0
    return col


def _climb(n: int = N, *, px: int = 3, stall_at=None,
           stall_len: int = 0) -> np.ndarray:
    """A camera walking `px` per step, optionally frozen for a block —
    the death-animation freeze every real death leaves behind."""
    x = np.zeros(n, dtype=np.int64)
    v = 0
    for t in range(n):
        frozen = stall_at is not None and stall_at <= t < stall_at + stall_len
        if not frozen:
            v += px
        x[t] = v
    return np.stack([x, np.zeros(n, dtype=np.int64)], axis=1)


def _flat(n: int = N) -> np.ndarray:
    return np.zeros((n, 2), dtype=np.int64)


def _arm(cols: dict, *, odo=None, scene=None, thr: float = 350.0,
         n: int = N, churn_at=None) -> dict:
    """One validation arm: a RAM log with the named columns, an odometer
    trace, and the reset threshold the mass-reload veto reads."""
    log = np.zeros((n, RAM), dtype=np.uint8)
    for addr, col in cols.items():
        log[:, int(addr)] = col
    if churn_at is not None:
        # A level reload: every byte rewritten in one step, which is how
        # every other probe in this module detects one.
        log[churn_at, :] = np.arange(RAM, dtype=np.uint8) + 1
    return {"log": log, "steps": n,
            "odo": _flat(n) if odo is None else odo,
            "scene": np.zeros(n, dtype=np.int64) if scene is None else scene,
            "reset_threshold": thr}


def _probe(**arms) -> dict:
    return {"arms": arms, "odometer": True,
            "steps": len(next(iter(arms.values()))["log"])}


def _clean_stock_probe() -> dict:
    """The Contra $0032 shape, the positive control: two lives spent over
    a held-forward drive, the camera frozen for 20 steps by the death
    animation after the last one, nothing moving while the pad is idle."""
    return _probe(
        hold=_arm({LIVES: _spend()}, odo=_climb(stall_at=140, stall_len=20)),
        idle=_arm({LIVES: _col()}, odo=_flat()),
        mash=_arm({LIVES: _col()}, odo=_climb(px=2)),
    )


def _verdict(probe: dict, addr: int = LIVES, deaths=()) -> dict:
    return behavioural_lives_verdict({"addr": addr}, probe, death_logs=deaths)


# ---------------------------------------------------------------------------
# The positive control. Nothing else in this file means anything if a
# real stock does not survive the gate.
# ---------------------------------------------------------------------------
def test_a_clean_stock_passes_all_three_checks():
    v = _verdict(_clean_stock_probe())
    assert v["verdict"] == "PASS", v["reason"]
    assert v["failed"] == []
    assert all(c["verdict"] in ("pass", "inconclusive", "unavailable")
               for c in v["checks"].values())


def test_a_clean_stock_still_passes_with_its_death_drives_folded_in():
    """The death drives that nominated the byte are reused as extra
    oscillation evidence; a stock spent once per drive must not be
    convicted for being consistent across them (the Shatterhand $0199
    case: one 2 -> 0 -> 4 cycle in every one of five drives, verified
    real, 16 -> 232 cells when wired)."""
    deaths = []
    for k in range(5):
        log = np.zeros((700, RAM), dtype=np.uint8)
        col = np.full(700, 2, dtype=np.uint8)
        col[150 + 10 * k:153 + 10 * k] = 0
        col[153 + 10 * k:] = 4              # refilled once, per drive
        log[:, LIVES] = col
        deaths.append(log)
    v = _verdict(_clean_stock_probe(), deaths=deaths)
    assert v["verdict"] == "PASS", v["reason"]
    assert v["checks"]["oscillation"]["worst_drive"] <= OSC_MAX_CYCLES


# ---------------------------------------------------------------------------
# FAILURE MODE 1 — oscillation (Bad Dudes $00CD).
# ---------------------------------------------------------------------------
def test_an_oscillating_attack_counter_is_rejected():
    probe = _probe(
        hold=_arm({LIVES: _col()}, odo=_climb()),
        idle=_arm({LIVES: _col()}, odo=_flat()),
        # The attack-combo arm is the one that provokes it: measured 20
        # cycles in a single 320-step mash on Bad Dudes.
        mash=_arm({LIVES: _toggle()}, odo=_climb(px=2)),
    )
    v = _verdict(probe)
    assert v["verdict"] == "REJECT"
    assert "oscillation" in v["failed"]
    osc = v["checks"]["oscillation"]
    assert osc["worst_drive"] > OSC_MAX_CYCLES
    assert osc["first_cycle_step"] <= 5      # first toggle at step 3
    assert "not a stock" in v["reason"]


def test_the_oscillation_count_is_per_drive_not_summed():
    """One cycle per drive is a refill; many in ONE drive is a counter.
    Summing would convict the first for showing up in every rep."""
    once = [np.zeros((200, RAM), dtype=np.uint8) for _ in range(8)]
    for log in once:
        col = np.full(200, 2, dtype=np.uint8)
        col[100:103] = 0
        col[103:] = 2
        log[:, LIVES] = col
    v = _verdict(_clean_stock_probe(), deaths=once)
    assert v["checks"]["oscillation"]["cycles"] == 8      # eight drives
    assert v["checks"]["oscillation"]["worst_drive"] == 1
    assert v["checks"]["oscillation"]["verdict"] == "pass"


def test_a_slow_refill_is_not_counted_as_a_cycle():
    """A stock refilled at a continue comes back long after it emptied
    (1942 $0524: 1 -> 0 at t=65, refill at t=471). Only a bounce inside
    OSC_RECOVER_STEPS is an oscillation."""
    col = np.concatenate([np.full(60, 1), np.zeros(OSC_RECOVER_STEPS + 40),
                          np.full(60, 1)]).astype(np.uint8)
    assert _recovery_cycles(col) == (0, None)


def test_a_last_life_wrapping_0_to_255_is_not_a_recovery():
    """A counter that rolls 0 -> 255 to signal its last life reads as a
    huge RISE to a naive comparison; the wrap-unfolded level reads it as
    the step down it is."""
    col = np.concatenate([np.full(10, 1), np.zeros(10),
                          np.full(10, 255)]).astype(np.uint8)
    assert _wrap_levels(col)[-1] == -2
    assert _recovery_cycles(col) == (0, None)


# ---------------------------------------------------------------------------
# FAILURE MODE 2 — empties while alive (Ninja Gaiden $0386).
# ---------------------------------------------------------------------------
def test_a_byte_that_empties_while_the_odometer_climbs_is_rejected():
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=32)}, odo=_climb(px=3)),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
        mash=_arm({LIVES: _col(level=1)}, odo=_climb(px=2)),
    )
    v = _verdict(probe)
    assert v["verdict"] == "REJECT"
    assert "empties_while_alive" in v["failed"]
    ch = v["checks"]["empties_while_alive"]
    assert ch["empty_step"] == 32
    assert ch["witness_px"] >= ALIVE_MIN_PX
    assert ch["longest_stall"] == 0
    assert "still alive" in v["reason"]


def test_a_camera_that_freezes_after_the_empty_is_a_death_not_liveness():
    """The Contra discriminator, isolated. 67% of that window is moving —
    a moving-FRACTION bar low enough to spare it would have spared Bad
    Dudes' 97% too. The contiguous freeze is what says a death happened."""
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=100)},
                  odo=_climb(stall_at=105, stall_len=ALIVE_MAX_STALL + 14)),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "pass"
    assert ch["longest_stall"] > ALIVE_MAX_STALL
    assert 0.0 < ch["moving_fraction"] < 1.0


def test_a_level_reload_after_the_empty_vetoes_the_rejection():
    """A death that reloads the level rewrites RAM en masse. The camera
    may well keep rolling straight afterwards; that is a death, not a
    byte with nothing behind it."""
    # `thr` scaled to this 64-byte synthetic RAM: the live probe
    # calibrates it off the real 2 KB churn (`Discoverer.reset_threshold`).
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=100)}, odo=_climb(px=3),
                  churn_at=104, thr=RAM / 2),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "pass"
    assert ch["mass_ram_event"] is True


def test_a_scene_cut_after_the_empty_vetoes_the_rejection():
    scene = np.zeros(N, dtype=np.int64)
    scene[110:] = 1
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=100)}, odo=_climb(px=3),
                  scene=scene),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "pass"
    assert ch["scene_bump"] is True


def test_a_camera_re_anchored_backwards_vetoes_the_rejection():
    """A level restart drags the camera back to the start of the stage."""
    odo = _climb(px=3)
    odo[105:, 0] = odo[104, 0] - 200          # restart: back to the left
    odo[105:, 0] += np.arange(N - 105) * 3
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=100)}, odo=odo),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "pass"
    assert ch["regress_px"] > 0


def test_a_camera_static_game_is_inconclusive_not_a_rejection():
    """Galaga and Punch-Out never scroll (CAMERA_STATIC_AGENT_ACTIVE).
    There is no liveness witness there, and 'no witness' is not evidence
    of anything — it must not convict, and must not quietly acquit."""
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=100)}, odo=_flat()),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "inconclusive"
    assert "camera-static" in ch["note"]
    assert _verdict(probe)["passed"] is True


def test_a_byte_that_never_empties_in_an_arm_is_inconclusive():
    probe = _probe(hold=_arm({LIVES: _col()}, odo=_climb()),
                   idle=_arm({LIVES: _col()}, odo=_flat()))
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "inconclusive"
    assert "never arrived at zero" in ch["note"]


def test_an_empty_too_late_in_the_arm_to_witness_is_inconclusive():
    """Nothing is claimed from a witness window that does not exist."""
    probe = _probe(hold=_arm({LIVES: _empties_at(step=N - 4)}, odo=_climb()),
                   idle=_arm({LIVES: _col(level=1)}, odo=_flat()))
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["verdict"] == "inconclusive"
    assert "too" in ch["note"]


def test_no_odometer_binding_reports_unavailable_and_never_rejects():
    """On a build with no odometer the check has no witness at all. It
    says so instead of passing every candidate silently."""
    arm = _arm({LIVES: _empties_at(step=32)})
    arm["odo"] = None
    probe = {"arms": {"hold": arm}, "odometer": False, "steps": N}
    v = _verdict(probe)
    assert v["checks"]["empties_while_alive"]["verdict"] == "unavailable"
    assert v["passed"] is True


# ---------------------------------------------------------------------------
# FAILURE MODE 3 — fires on movement onset (Journey to Silius $0135).
# ---------------------------------------------------------------------------
def test_a_byte_that_fires_on_movement_onset_is_rejected():
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=4)}, odo=_climb(px=4)),
        idle=_arm({LIVES: _col(level=1)}, odo=_flat()),
        mash=_arm({LIVES: _empties_at(step=4)}, odo=_climb(px=2)),
    )
    v = _verdict(probe)
    assert v["verdict"] == "REJECT"
    assert "fires_on_movement" in v["failed"]
    ch = v["checks"]["fires_on_movement"]
    assert ch["hold_first_drop"] == 4 and ch["idle_first_drop"] is None
    assert "movement onset" in ch["note"]


def test_a_game_that_kills_an_idle_player_keeps_its_counter():
    """The idle arm is the control. If the drop shows up with the pad
    untouched too, it is not keyed to the input and the check must not
    fire — otherwise a game that kills a standing player would lose the
    one byte this whole pass exists to find."""
    probe = _probe(
        hold=_arm({LIVES: _empties_at(step=4)}, odo=_climb(px=4)),
        idle=_arm({LIVES: _empties_at(step=6)}, odo=_flat()),
    )
    ch = _verdict(probe)["checks"]["fires_on_movement"]
    assert ch["verdict"] == "pass"
    assert ch["idle_first_drop"] == 6


def test_a_drop_well_after_the_pad_goes_live_is_not_movement_triggered():
    probe = _probe(
        hold=_arm({LIVES: _spend(at=(MOVEMENT_ONSET_STEPS + 40, 200))},
                  odo=_climb()),
        idle=_arm({LIVES: _col()}, odo=_flat()),
    )
    assert _verdict(probe)["checks"]["fires_on_movement"]["verdict"] == "pass"


def test_no_held_direction_arm_reports_unavailable():
    probe = {"arms": {"idle": _arm({LIVES: _col()})}, "odometer": True,
             "steps": N}
    assert _verdict(probe)["checks"]["fires_on_movement"]["verdict"] == "unavailable"


def test_no_idle_control_arm_abstains_rather_than_convicting():
    """A drop at step 4 of a held drive is only evidence NEXT TO an idle
    arm that does not show it. Without the control the check must not
    reject on half the evidence."""
    probe = {"arms": {"hold": _arm({LIVES: _empties_at(step=4)},
                                   odo=_climb(px=4))},
             "odometer": True, "steps": N}
    v = _verdict(probe)
    assert v["checks"]["fires_on_movement"]["verdict"] == "unavailable"
    assert "fires_on_movement" not in v["failed"]


# ---------------------------------------------------------------------------
# Screening: the tool must nominate the highest-ranked candidate that
# PASSES, and must be honest when none do.
# ---------------------------------------------------------------------------
def _ranked_probe() -> dict:
    """Rank 0 is an oscillator, rank 1 is a real stock."""
    return _probe(
        hold=_arm({LIVES: _col(), OTHER: _spend()},
                  odo=_climb(stall_at=140, stall_len=20)),
        idle=_arm({LIVES: _col(), OTHER: _col()}, odo=_flat()),
        mash=_arm({LIVES: _toggle(), OTHER: _col()}, odo=_climb(px=2)),
    )


def test_the_pick_is_the_highest_ranked_candidate_that_passes():
    cands = [{"addr": LIVES}, {"addr": OTHER}]
    summary = screen_lives_behaviourally(cands, _ranked_probe())
    assert summary["picked"] == OTHER
    assert summary["passed"] == 1 and summary["rejected"] == 1
    assert summary["skipped_addrs"] == [LIVES]


def test_every_candidate_keeps_its_verdict_for_the_receipt():
    """A rejection is evidence. It stays attached to the candidate so the
    findings JSON shows what was thrown out and why, rather than a
    silently shorter list."""
    cands = [{"addr": LIVES}, {"addr": OTHER}]
    screen_lives_behaviourally(cands, _ranked_probe())
    assert cands[0]["behaviour"]["verdict"] == "REJECT"
    assert cands[0]["behaviour"]["reason"]
    assert cands[1]["behaviour"]["verdict"] == "PASS"
    assert set(cands[0]["behaviour"]["checks"]) == {
        "oscillation", "empties_while_alive", "fires_on_movement"}


def test_no_candidate_passing_is_reported_as_such_not_as_a_pick():
    """The correct answer for six of the fanout's eleven profiles. An
    empty pick with named reasons beats wiring a byte that inverts death
    detection."""
    probe = _probe(
        hold=_arm({LIVES: _toggle(), OTHER: _empties_at(step=4)},
                  odo=_climb(px=4)),
        idle=_arm({LIVES: _col(), OTHER: _col(level=1)}, odo=_flat()),
        mash=_arm({LIVES: _toggle(), OTHER: _col(level=1)}, odo=_climb(px=2)),
    )
    cands = [{"addr": LIVES}, {"addr": OTHER}]
    summary = screen_lives_behaviourally(cands, probe)
    assert summary["picked"] is None
    assert summary["passed"] == 0 and summary["rejected"] == 2
    assert {r["addr"] for r in summary["rejections"]} == {LIVES, OTHER}
    assert all(r["reason"] for r in summary["rejections"])


def test_an_empty_candidate_list_is_not_a_rejection():
    summary = screen_lives_behaviourally([], _clean_stock_probe())
    assert summary["watched"] == 0 and summary["rejected"] == 0
    assert summary["picked"] is None


# ---------------------------------------------------------------------------
# The emitted profile block must never hide a rejection.
# ---------------------------------------------------------------------------
def _findings(hp_lives: dict) -> dict:
    return {"rom": "roms/Fake (USA).nes", "forward": "right",
            "progress": {"forward": "right", "room_counter": None,
                         "candidates": [], "recommended": None},
            "room_counters": [], "y": [], "hp_lives": hp_lives}


def test_emit_says_no_lives_byte_when_every_nomination_was_rejected():
    gate = {"watched": 2, "passed": 0, "rejected": 2, "odometer": True,
            "steps_per_arm": N, "picked": None, "skipped_addrs": [LIVES],
            "rejections": [{"addr": LIVES, "failed": ["oscillation"],
                            "reason": "falls and recovers 20 times inside "
                                      "ONE drive"}]}
    text = emit_solve_yaml("roms/Fake (USA).nes", _findings(
        {"kind": "none", "addr": None, "note": "n/a", "behaviour_gate": gate}))
    assert "# lives: <none" in text and "REJECTED" in text
    assert f"rejected 0x{LIVES:04X}" in text
    assert "oscillation" in text
    # And it must NOT emit a live `lives:` line for the rejected byte.
    assert f"lives: 0x{LIVES:04X}" not in text


def test_emit_marks_a_wired_lives_byte_as_behaviourally_checked():
    """And it spells out the per-check verdicts rather than a bare PASS:
    an `inconclusive` liveness check (the byte never emptied inside the
    arm, or the camera never moved) is a real gap in the evidence and
    the reader must be able to see it in the profile itself."""
    gate = {"watched": 1, "passed": 1, "rejected": 0, "odometer": True,
            "steps_per_arm": N, "picked": OTHER, "skipped_addrs": [],
            "outranked_rejects": 0, "rejections": []}
    probe = _probe(hold=_arm({OTHER: _col()}, odo=_climb()),
                   idle=_arm({OTHER: _col()}, odo=_flat()))
    detail = {"addr": OTHER, "behaviour": behavioural_lives_verdict(
        {"addr": OTHER}, probe)}
    text = emit_solve_yaml("roms/Fake (USA).nes", _findings(
        {"kind": "lives", "addr": OTHER, "note": "n/a", "detail": detail,
         "behaviour_gate": gate}))
    assert f"lives: 0x{OTHER:04X}" in text
    assert "behaviour gate PASS" in text
    assert "osc pass" in text and "alive inconclusive" in text


# ---------------------------------------------------------------------------
# The primitives, directly.
# ---------------------------------------------------------------------------
def test_arrival_at_zero_needs_an_arrival_not_a_zero_start():
    assert _arrival_at_zero(np.array([0, 0, 0], dtype=np.uint8)) is None
    assert _arrival_at_zero(np.array([2, 1, 0, 0], dtype=np.uint8)) == 2


def test_first_drop_step_is_the_step_the_new_value_appears():
    assert _first_drop_step(np.array([2, 2, 1, 1], dtype=np.uint8)) == 2
    assert _first_drop_step(np.array([2, 2, 2], dtype=np.uint8)) is None


def test_longest_stall_counts_the_block_not_the_total():
    assert _longest_stall(np.array([1, 0, 1, 0, 1, 0])) == 1
    assert _longest_stall(np.array([1, 0, 0, 0, 1, 0])) == 3
    assert _longest_stall(np.zeros(5)) == 5


@pytest.mark.parametrize("witness", [ALIVE_WITNESS_STEPS])
def test_the_witness_window_is_the_named_budget(witness):
    """The window is a named constant, not a literal buried in the check
    — the next person to re-tune it against a new receipt edits one line."""
    probe = _probe(hold=_arm({LIVES: _empties_at(step=100)}, odo=_climb(px=3)),
                   idle=_arm({LIVES: _col(level=1)}, odo=_flat()))
    ch = _verdict(probe)["checks"]["empties_while_alive"]
    assert ch["witness_steps"] == witness
