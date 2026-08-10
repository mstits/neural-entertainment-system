"""Tests for scripts/discover_observables.py's pure gate machinery.

Kept to the parts that take arrays rather than an emulator: `Discoverer`
owns a real Pool and the four `find_*` passes drive real ROMs, which is
what `--selftest` (Contra + Kirby ground truth) is for. What is covered
here is the logic those passes delegate to, and — the reason this file
exists — the wiring that decides what `emit_solve_yaml` is allowed to
see.

The tape path (`find_progress_from_tapes`) ranks RAM bytes by
lexicographic weight over banked solver tapes. That ranking is a
SHORTLIST: it cannot tell a free-running timer from real progress, and
the objective behind it is anti-correlated with progress off its fit
range (held-out rho -0.771), so it must never reach a `solve:` block
unadjudicated. `gate_tape_candidates` is the choke point, and it is pure
by construction so that contract is testable without a ROM.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.discover_observables import (
    LEARNFUN_SHORTLIST,
    _col_stats,
    _flat_under_noop,
    _saturates_conclusively,
    _saturation_from_logs,
    _wrap_deltas,
    emit_solve_yaml,
    gate_summary,
    gate_tape_candidates,
)
from src.training.lexicographic_objectives import NEVER_WIRE_AS_REWARD

RAM = 0x800
PROGRESS, TIMER, CAMERA, ROOM = 0x0075, 0x07C7, 0x0086, 0x004F

#: Keys `find_progress` puts on every candidate. The tape path must
#: produce the same shape or downstream readers (emit_solve_yaml,
#: _print_report, the findings JSON) quietly break.
FIND_PROGRESS_CANDIDATE_KEYS = {
    "lo", "hi", "kind", "label", "wrap_coupled", "gate1_rises_forward",
    "gate1_flat_under_noop", "reversible_bonus", "net_forward",
    "mono_forward", "net_noop", "net_reverse", "saturates", "camera_clamp",
    "sat_saturates", "sat_within_room_cap", "sat_cap_tail_len",
    "sat_rebases_at_transitions", "sat_n_transitions",
    "sat_player_active_in_tail", "sat_observed_max",
}


# ---------------------------------------------------------------------
# Shared delta math.
# ---------------------------------------------------------------------

def test_wrap_deltas_folds_8bit_wrap_both_ways():
    log = np.array([[254], [255], [0], [1], [0], [255]], dtype=np.uint8)
    d = _wrap_deltas(log)[:, 0]
    assert list(d) == [1, 1, 1, -1, -1]      # 255->0 is +1, 0->255 is -1
    keep = np.array([True, False, True, True, True])
    assert list(_wrap_deltas(log, keep)[:, 0]) == [1, 1, -1, -1]


# ---------------------------------------------------------------------
# GATE 1 — NOOP flatness. The only thing that rejects a timer.
# ---------------------------------------------------------------------

def test_gate1_rejects_a_free_running_timer_and_passes_a_parked_byte():
    """SMB's $07C7 is the named case: it takes lead-rank 6 with the
    heaviest mass in the top ten of the banked 1-1 tape and scores the
    same on a shuffled tape, so only idling can see through it."""
    n = 240
    idle = np.zeros((n, RAM), dtype=np.uint8)
    idle[:, TIMER] = np.arange(n).astype(np.uint8)
    idle[:, PROGRESS] = 40
    sn = _col_stats(idle)
    assert not _flat_under_noop(sn, TIMER)
    assert _flat_under_noop(sn, PROGRESS)


def test_gate1_churn_term_means_no_movement_at_all_on_the_idle_probe():
    """The rule is |net| <= 4 AND churn < 2 changes per 1000 steps, and
    on the 240-step NOOP probe the second term binds hard: 1 change is
    4.2/1k, so ANY movement while idling is a rejection. Worth knowing
    before widening the probe — the threshold is calibrated to its
    length, not to an absolute jitter budget."""
    n = 240
    log = np.zeros((n, RAM), dtype=np.uint8)
    log[:, 0x10] = 50                          # never moves
    log[:, 0x11] = 50
    log[100, 0x11] = 52                        # one twitch, returns
    sn = _col_stats(log)
    assert _flat_under_noop(sn, 0x10)
    assert float(sn["churn"][0x11]) > 2.0
    assert not _flat_under_noop(sn, 0x11)


def test_gate1_net_term_rejects_a_slow_creep_the_churn_term_would_miss():
    """On a long drive a byte can tick rarely enough to clear the churn
    term and still walk away from its resting value; |net| catches it."""
    n = 6000
    log = np.zeros((n, RAM), dtype=np.uint8)
    log[:, 0x12] = 50 + (np.arange(n) // 2000).astype(np.uint8)  # +2, ok
    log[:, 0x13] = 50 + (np.arange(n) // 1000).astype(np.uint8)  # +5, drift
    sn = _col_stats(log)
    assert float(sn["churn"][0x13]) < 2.0      # too rare for the churn term
    assert _flat_under_noop(sn, 0x12)
    assert not _flat_under_noop(sn, 0x13)      # ... but net 9 > 4


# ---------------------------------------------------------------------
# GATE 2 — saturation, over whichever forward drive it is handed.
# ---------------------------------------------------------------------

def _drive(values: np.ndarray, *, room: np.ndarray | None = None,
           churn: bool = True) -> np.ndarray:
    """A RAM log whose CAMERA byte follows `values`, with zero-page churn
    so the gate can tell "player still active" from "frozen"."""
    n = values.size
    log = np.zeros((n, RAM), dtype=np.uint8)
    log[:, CAMERA] = values.astype(np.uint8)
    if churn:
        log[:, 0x20] = (np.arange(n) * 7 % 251).astype(np.uint8)
    if room is not None:
        log[:, ROOM] = room.astype(np.uint8)
    return log


def test_gate2_flags_a_byte_that_caps_early_while_play_continues():
    n = 900
    v = np.concatenate([np.linspace(0, 240, 300), np.full(600, 240.0)])
    log = _drive(v)
    sat = _saturation_from_logs(log, log, np.zeros(n - 1, bool),
                                CAMERA, None, None)
    assert sat["within_room_cap"] and sat["saturates"]
    assert sat["cap_tail_len"] >= 150
    assert sat["player_active_in_tail"]
    assert not sat["never_rose"]
    assert _saturates_conclusively(sat)


def test_gate2_passes_a_byte_still_climbing_when_the_drive_ends():
    n = 900
    log = _drive(np.linspace(0, 250, n))
    sat = _saturation_from_logs(log, log, np.zeros(n - 1, bool),
                                CAMERA, None, None)
    assert not sat["saturates"] and not sat["within_room_cap"]
    assert not sat["never_rose"]


def test_gate2_flags_a_coordinate_that_rebases_at_room_transitions():
    """The other camera-clamp signature: a room-local coordinate resets
    at each door, a global position does not."""
    n = 900
    room = np.repeat(np.arange(3), n // 3)
    v = np.tile(np.linspace(0, 200, n // 3), 3)
    log = _drive(v, room=room)
    sat = _saturation_from_logs(log, log, np.zeros(n - 1, bool),
                                CAMERA, None, ROOM)
    assert sat["rebases_at_transitions"] >= 1
    assert sat["n_transitions"] >= 2
    assert sat["saturates"] and _saturates_conclusively(sat)


def test_gate2_calls_a_byte_that_never_moved_inconclusive_not_capped():
    """A world/stage byte does not change inside one entrance's probe.
    Its tail is flat for the boring reason, so the cap signature must not
    be read as a camera clamp — otherwise the gates reject exactly the
    coarse progress bytes the tape ranking is good at finding."""
    n = 900
    log = _drive(np.zeros(n))
    sat = _saturation_from_logs(log, log, np.zeros(n - 1, bool),
                                CAMERA, None, None)
    assert sat["never_rose"]
    assert sat["within_room_cap"]              # the raw signature fires...
    assert not _saturates_conclusively(sat)    # ...but it is not evidence


# ---------------------------------------------------------------------
# The choke point: what emit_solve_yaml is allowed to see.
# ---------------------------------------------------------------------

def _ranked(addr: int, rank: int, *, lead: float, mass: float,
            up: int = 50, down: int = 0) -> dict:
    return {"addr": addr, "rank": rank, "lead": lead, "mass": mass,
            "disc": lead, "n_obj": 4, "best_pos": 0, "n_up": up,
            "n_down": down, "span": 200, "singleton_vf": 0.5}


def _noop_stats(*, ticking: tuple = (TIMER,)) -> dict:
    n = 240
    idle = np.zeros((n, RAM), dtype=np.uint8)
    for a in ticking:
        idle[:, a] = np.arange(n).astype(np.uint8)
    return _col_stats(idle)


def _gate2_stub(saturating=(), never_rose=()):
    def gate2(addr: int) -> dict:
        return {"saturates": addr in saturating,
                "within_room_cap": addr in saturating,
                "cap_tail_len": 400 if addr in saturating else 0,
                "rebases_at_transitions": 0, "n_transitions": 3,
                "player_active_in_tail": True, "observed_max": 240,
                "never_rose": addr in never_rose}
    return gate2


def test_a_top_ranked_timer_is_rejected_and_never_recommended():
    """The whole point of routing the shortlist through the gates: the
    timer wins the ranking (rank 1, heaviest mass, climbs every frame)
    and still cannot become the recommendation."""
    ranked = [_ranked(TIMER, 1, lead=44.0, mass=241.0, up=239),
              _ranked(PROGRESS, 2, lead=26.0, mass=112.0, up=60)]
    cands, rec = gate_tape_candidates(
        ranked, probe_stats={"noop": _noop_stats()}, gate2=_gate2_stub())

    assert rec is not None and rec["lo"] == PROGRESS
    assert rec["kind"] == "learnfun_shortlist"
    assert rec["shortlist_only"] is True
    assert rec["never_wire_as_reward"] == NEVER_WIRE_AS_REWARD

    timer = next(c for c in cands if c["lo"] == TIMER)
    assert timer["gate1_flat_under_noop"] is False
    assert timer["gates_passed"] is False
    assert timer["rejected_by"] == "gate1_noop_timer"
    # ... and it sorts BELOW the byte that passed, despite outranking it.
    assert [c["lo"] for c in cands] == [PROGRESS, TIMER]


def test_a_saturator_is_rejected_even_with_the_top_lead_weight():
    ranked = [_ranked(CAMERA, 1, lead=90.0, mass=300.0),
              _ranked(PROGRESS, 2, lead=10.0, mass=40.0)]
    cands, rec = gate_tape_candidates(
        ranked, probe_stats={"noop": _noop_stats()},
        gate2=_gate2_stub(saturating=(CAMERA,)))
    cam = next(c for c in cands if c["lo"] == CAMERA)
    assert cam["saturates"] and cam["camera_clamp"]
    assert cam["rejected_by"] == "gate2_camera_clamp"
    assert rec["lo"] == PROGRESS


def test_a_byte_the_drives_never_moved_is_held_back_as_inconclusive():
    ranked = [_ranked(PROGRESS, 1, lead=90.0, mass=300.0)]
    cands, rec = gate_tape_candidates(
        ranked, probe_stats={"noop": _noop_stats()},
        gate2=_gate2_stub(never_rose=(PROGRESS,)))
    assert cands[0]["gate2_conclusive"] is False
    assert cands[0]["rejected_by"] == "gate2_inconclusive"
    assert rec is None, "an unadjudicated byte must not be recommended"


def test_a_byte_that_only_falls_on_the_tape_fails_gate1s_rise_half():
    ranked = [_ranked(PROGRESS, 1, lead=9.0, mass=40.0, up=2, down=60)]
    cands, rec = gate_tape_candidates(
        ranked, probe_stats={"noop": _noop_stats()}, gate2=_gate2_stub())
    assert cands[0]["gate1_rises_on_tape"] is False
    assert cands[0]["rejected_by"] == "gate1_no_forward_rise"
    assert rec is None


def test_tape_candidates_keep_find_progress_shape():
    """`discover_all` / `emit_solve_yaml` / the findings JSON all read
    `find_progress`'s candidate keys; the tape path must not drift."""
    cands, _ = gate_tape_candidates(
        [_ranked(PROGRESS, 1, lead=9.0, mass=40.0)],
        probe_stats={"noop": _noop_stats()}, gate2=_gate2_stub())
    missing = FIND_PROGRESS_CANDIDATE_KEYS - set(cands[0])
    assert not missing, f"tape candidate lost {sorted(missing)}"
    assert cands[0]["kind"] == "single" and cands[0]["hi"] is None
    assert cands[0]["source"] == "learnfun_tape_chain"


def test_probe_forward_stats_can_supply_gate1s_rise_half():
    """A byte the probe itself sees climbing passes the rise half even
    when the tape barely moves it."""
    n = 240
    fwd = np.zeros((n, RAM), dtype=np.uint8)
    fwd[:, PROGRESS] = (np.arange(n) // 2).astype(np.uint8)
    stats = {"noop": _noop_stats(), "forward": _col_stats(fwd)}
    cands, rec = gate_tape_candidates(
        [_ranked(PROGRESS, 1, lead=9.0, mass=40.0, up=0, down=0)],
        probe_stats=stats, gate2=_gate2_stub())
    assert cands[0]["gate1_rises_forward"] is True
    assert cands[0]["gates_passed"] is True and rec is not None


def test_shortlist_size_is_a_named_budget():
    assert 8 <= LEARNFUN_SHORTLIST <= 64


def test_gate_summary_separates_untested_bytes_from_rejected_ones():
    """The useful column on a chain: a byte can be timer-clean and still
    have no saturation verdict, because the live probe never moves it
    from one entrance. Measured on the SMB world-1 chain, that is where
    $075F (world) lands — it must not be filed alongside the timers."""
    ranked = [_ranked(TIMER, 1, lead=44.0, mass=241.0),
              _ranked(CAMERA, 2, lead=30.0, mass=90.0),
              _ranked(0x075F, 3, lead=20.0, mass=241.0),
              _ranked(PROGRESS, 4, lead=10.0, mass=40.0)]
    cands, rec = gate_tape_candidates(
        ranked, probe_stats={"noop": _noop_stats()},
        gate2=_gate2_stub(saturating=(CAMERA,), never_rose=(0x075F,)))
    s = gate_summary(cands)
    assert s["n_gated"] == 4
    assert s["passed"] == [f"${PROGRESS:04X}"]
    assert s["rejected_as_timer"] == [f"${TIMER:04X}"]
    assert s["rejected_as_camera_clamp"] == [f"${CAMERA:04X}"]
    assert s["gate1_clean_gate2_untested"] == ["$075F"]
    assert s["verdicts"]["passed"] == 1
    assert rec["lo"] == PROGRESS


# ---------------------------------------------------------------------
# emit_solve_yaml: the only writer of a progress line.
# ---------------------------------------------------------------------

def _findings(progress: dict) -> dict:
    return {"rom": "roms/Fake (USA).nes", "forward": "right",
            "progress": progress, "room_counters": [], "y": [],
            "hp_lives": {"kind": "none", "addr": None, "note": "n/a"}}


def test_emit_marks_a_tape_shortlist_as_unconfirmed_and_not_a_reward():
    cands, rec = gate_tape_candidates(
        [_ranked(TIMER, 1, lead=44.0, mass=241.0),
         _ranked(PROGRESS, 2, lead=26.0, mass=112.0)],
        probe_stats={"noop": _noop_stats()}, gate2=_gate2_stub())
    text = emit_solve_yaml("roms/Fake (USA).nes", _findings(
        {"forward": "right", "room_counter": None, "candidates": cands,
         "recommended": rec}))
    assert f"progress: {{lo: 0x{PROGRESS:04X}}}" in text
    assert "SHORTLIST" in text and "CONFIRM" in text
    assert NEVER_WIRE_AS_REWARD in text
    # The byte the ranking liked best never reaches the block.
    assert f"0x{TIMER:04X}" not in text


def test_emit_writes_no_progress_line_when_nothing_cleared_the_gates():
    cands, rec = gate_tape_candidates(
        [_ranked(TIMER, 1, lead=44.0, mass=241.0)],
        probe_stats={"noop": _noop_stats()}, gate2=_gate2_stub())
    assert rec is None
    text = emit_solve_yaml("roms/Fake (USA).nes", _findings(
        {"forward": "right", "room_counter": None, "candidates": cands,
         "recommended": rec}))
    assert "# progress: <none isolated" in text
    assert f"0x{TIMER:04X}" not in text


def test_emit_handles_a_single_byte_saturator_in_the_room_fallback():
    """A fine byte with no page partner has hi=None; the room-counter
    fallback used to format it as `hi: 0x{None}` and crash."""
    saturator = {"lo": CAMERA, "hi": None, "label": f"${CAMERA:04X}",
                 "saturates": True, "sat_observed_max": 3300}
    text = emit_solve_yaml("roms/Fake (USA).nes", _findings({
        "forward": "right", "room_counter": ROOM, "candidates": [saturator],
        "recommended": {"lo": ROOM, "hi": None, "label": f"${ROOM:04X}",
                        "kind": "room_counter_fallback", "saturates": False,
                        "as_progress": "room counter",
                        "spatial_saturator": f"${CAMERA:04X}",
                        "spatial_cap": 3300}}))
    assert f"progress: {{lo: 0x{ROOM:04X}}}" in text
    assert f"#   progress: {{lo: 0x{CAMERA:04X}}}" in text
    assert "None" not in text


@pytest.mark.parametrize("bad", [
    {"noop": None},
])
def test_gate_tape_candidates_requires_the_noop_probe(bad):
    """Gate 1's rejecting half is not optional — without the idle probe
    there is nothing standing between a timer and a `solve:` block."""
    with pytest.raises((TypeError, KeyError, IndexError)):
        gate_tape_candidates([_ranked(TIMER, 1, lead=1.0, mass=1.0)],
                             probe_stats=bad, gate2=_gate2_stub())
