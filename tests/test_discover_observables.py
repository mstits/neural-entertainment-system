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
    DEATH_MIN_AGREE,
    LEARNFUN_SHORTLIST,
    LIVES_MAX_DROP,
    LIVES_MAX_TICKS,
    SETTLE_CAP,
    _col_stats,
    _decrement_consensus,
    _flat_under_noop,
    _saturates_conclusively,
    _saturation_from_logs,
    _settled_start,
    _wrap_deltas,
    emit_solve_yaml,
    gate_summary,
    gate_tape_candidates,
    lives_from_death_drives,
    settle_index,
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


def test_gate2_checks_the_maneuver_drive_before_calling_a_plateau_a_clamp():
    """clean_forward is hold-forward only; it can stall on ordinary terrain
    (a pipe, a gap) that needs a jump to clear, long before any camera
    limit. advance() — already computed in the same call — drives
    jump/attack/door macros and is the cross-check: if IT keeps climbing
    well past clean_forward's plateau, the plateau is a hold-forward
    artifact, not a hardware scroll cap, and must not be flagged."""
    n = 900
    cf = _drive(np.concatenate([np.linspace(0, 40, 100), np.full(800, 40.0)]))
    adv = _drive(np.linspace(0, 240, n))
    sat = _saturation_from_logs(cf, adv, np.zeros(n - 1, bool),
                                CAMERA, None, None)
    assert not sat["within_room_cap"]
    assert not sat["saturates"]


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


# ---------------------------------------------------------------------
# The settle window. A start state is not always captured mid-play, and
# the level load that follows one that is not lands INSIDE every probe.
# ---------------------------------------------------------------------

def _idle_churn(*, steady: float, load_at: int | None, load: float = 320.0,
                n: int = 150) -> np.ndarray:
    """Per-step changed-byte counts for an idle scan: a quiet intro at a
    fraction of the steady rate, one load burst, then ordinary play."""
    ch = np.full(n, steady, dtype=float)
    if load_at is not None:
        ch[:load_at] = steady * 0.2
        ch[load_at] = load
    return ch


def test_settle_skips_a_level_load_the_probe_would_otherwise_measure():
    """The defect this exists for: a state captured on a level-intro
    card makes the player-position byte read 0 across the intro and jump
    to its spawn value when the level appears. Measured with the pad
    untouched, that is movement, and Gate 1 rejects the position byte as
    a free-running counter."""
    assert settle_index(_idle_churn(steady=45, load_at=40)) == 42


def test_a_state_already_in_play_settles_immediately():
    rng = np.random.default_rng(0)
    ch = 45 + rng.integers(-6, 7, size=150)
    assert settle_index(ch) == 0


def test_an_event_in_the_second_half_is_play_not_start_up():
    """Only the HEAD of the scan is start-up. A burst later on is the
    game doing something, and skipping past it would throw away the
    probe rather than its transient."""
    ch = _idle_churn(steady=45, load_at=None)
    ch[110] = 400.0
    assert settle_index(ch) == 0


def test_settle_is_capped_so_a_noisy_scan_cannot_eat_the_probe():
    ch = _idle_churn(steady=10, load_at=None, n=400)
    ch[199] = 900.0
    assert settle_index(ch) == SETTLE_CAP


def test_settle_needs_no_scan_at_all_to_answer():
    assert settle_index([]) == 0
    assert settle_index([10.0, 12.0]) == 0


# ---------------------------------------------------------------------
# The life counter, over uninterrupted runs.
#
# The two recall failures this replaces: a zero-page-only scan cannot
# even nominate a counter kept in high RAM, and a scan that reads the
# maneuver drive sees nothing because that drive MASKS the diff rows
# spanning a death — which is the only place a life counter moves.
# ---------------------------------------------------------------------

LIVES_HIGH = 0x075A          # a counter above the zero page
TIMER_HIGH = 0x07A0          # a countdown that ticks with the pad idle
SCRATCH = 0x006E


def _falling(n: int, start: int, at, *, step: int = 1, refill_at=None,
             refill_to: int = 0) -> np.ndarray:
    """A byte holding `start`, dropping by `step` at each row in `at`,
    optionally refilled to `refill_to` later. 8-bit wrapped."""
    col = np.empty(n, dtype=np.uint8)
    v = start
    at, refill_at = set(at), set(refill_at or ())
    for t in range(n):
        if t in at:
            v -= step
        if t in refill_at:
            v = refill_to
        col[t] = v % 256
    return col


def _life_drive(n: int = 300, **cols) -> dict:
    """One `death_drives` entry: a RAM log with the named columns set."""
    log = np.zeros((n, RAM), dtype=np.uint8)
    for addr, col in cols.items():
        log[:, int(addr)] = col
    return {"log": log, "steps": n}


def _idle_stats(*ticking: int) -> dict:
    churn = np.zeros(RAM)
    for a in ticking:
        churn[a] = 40.0
    return {"churn": churn}


def _runs(n_runs: int = 4, **kw) -> list:
    """`n_runs` drives whose deaths land in different places, the way
    different seeds make them."""
    out = []
    for k in range(n_runs):
        cols = {a: fn(k) for a, fn in kw.items()}
        out.append(_life_drive(**{str(a): c for a, c in cols.items()}))
    return out


def test_a_counter_above_the_zero_page_is_recalled():
    """The scope half of the defect. $075A is nowhere near the zero page
    the old scan read, and no amount of ranking can rescue a byte that
    was never a candidate."""
    cands, ev = lives_from_death_drives(_runs(
        **{str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k))}))
    assert ev["usable"] == 4
    assert [c["addr"] for c in cands] == [LIVES_HIGH]
    assert cands[0]["drop"] == 1
    assert cands[0]["runs_agreeing"] == 4
    assert cands[0]["reaches_empty"] and cands[0]["spends_its_stock"]


def test_a_countdown_that_ticks_while_idle_loses_to_the_counter():
    """A clock passes every other test here — it falls by one, reaches
    zero, accounts for its stock, and does it far more often. Idling is
    what separates them, exactly as it does for progress."""
    cands, _ = lives_from_death_drives(
        _runs(**{
            str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k)),
            str(TIMER_HIGH): lambda k: _falling(300, 6, range(20, 200, 30)),
        }),
        idle_stats=_idle_stats(TIMER_HIGH))
    assert [c["addr"] for c in cands] == [LIVES_HIGH, TIMER_HIGH]
    assert cands[1]["moves_while_idle"] is True
    # Demoted, never dropped: a game that kills an idle player would
    # otherwise disqualify its own counter.
    assert cands[1]["addr"] == TIMER_HIGH


def test_the_counter_that_moves_at_every_ending_beats_debris_that_does_not():
    """Both spend a stock to empty. Only one is charged for every run
    that ends."""
    cands, _ = lives_from_death_drives(_runs(**{
        str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k)),
        str(SCRATCH): lambda k: _falling(300, 1, (60 + 9 * k,)),
    }))
    assert cands[0]["addr"] == LIVES_HIGH
    assert cands[0]["spends"] > cands[1]["spends"]


def test_a_refill_after_the_stock_is_spent_is_not_a_disqualification():
    """A run that outlives the counter gets a fresh one — measured on
    the SMB control as 1 -> 0 -> 255 and then back up to 2. Reading
    that as 'this byte goes both ways' loses the counter outright."""
    cands, _ = lives_from_death_drives(_runs(**{
        str(LIVES_HIGH): lambda k: _falling(
            300, 1, (60 + 9 * k, 130 + 9 * k),
            refill_at=(200 + 9 * k,), refill_to=2),
    }))
    assert [c["addr"] for c in cands] == [LIVES_HIGH]
    assert cands[0]["refilled_runs"] == 4


def test_a_byte_refilled_without_ever_emptying_ranks_below_a_real_stock():
    """The refill allowance is what lets a counter survive a game over,
    so it cannot be a free pass. A byte that dips once, never reaches
    empty, and is topped up to more than it started with has not spent
    a stock, and must not outrank something that has."""
    cands, _ = lives_from_death_drives(_runs(**{
        str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k)),
        str(SCRATCH): lambda k: _falling(300, 3, (40 + 9 * k,),
                                         refill_at=(80 + 9 * k,), refill_to=9),
    }))
    assert [c["addr"] for c in cands] == [LIVES_HIGH, SCRATCH]
    assert not cands[1]["reaches_empty"]
    assert not cands[1]["spends_its_stock"]


def test_a_run_that_never_ended_abstains_instead_of_disqualifying():
    """Not every seeded run spends a life inside its budget. An
    unfinished run is missing evidence, not counter-evidence."""
    drives = _runs(3, **{
        str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k))})
    drives.append(_life_drive(**{str(LIVES_HIGH): np.full(300, 2, dtype=np.uint8)}))
    cands, ev = lives_from_death_drives(drives)
    assert [c["addr"] for c in cands] == [LIVES_HIGH]
    assert cands[0]["runs_agreeing"] == 3
    assert cands[0]["runs_watched"] == 4


def test_a_byte_that_disagrees_across_runs_is_not_a_counter():
    """A stock is spent at one price. One run charging a different
    amount is not a counter with an exception — it is debris that
    happened to fall the same way three times."""
    drives = _runs(3, **{
        str(SCRATCH): lambda k: _falling(300, 4, (60 + 9 * k, 150 + 9 * k))})
    drives.append(_life_drive(**{str(SCRATCH): _falling(300, 4, (70,), step=3)}))
    cands, _ = lives_from_death_drives(drives)
    assert [c["addr"] for c in cands] == []


def test_a_drop_wider_than_a_life_is_not_admitted():
    cands, _ = lives_from_death_drives(_runs(**{
        str(SCRATCH): lambda k: _falling(300, LIVES_MAX_DROP + 2, (60 + 9 * k,),
                                         step=LIVES_MAX_DROP + 1)}))
    assert cands == []


def test_a_byte_that_ticks_more_than_a_stock_could_is_not_admitted():
    cands, _ = lives_from_death_drives(_runs(**{
        str(SCRATCH): lambda k: _falling(
            300, 200, range(20, 20 + 12 * (LIVES_MAX_TICKS + 2), 12))}))
    assert cands == []


def test_a_mirror_is_folded_under_the_byte_it_copies():
    """A HUD copy of the counter carries the identical signature, and
    would otherwise fill the shortlist with the same finding."""
    cands, _ = lives_from_death_drives(_runs(**{
        str(LIVES_HIGH): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k)),
        str(LIVES_HIGH + 1): lambda k: _falling(300, 2, (60 + 9 * k, 150 + 9 * k)),
    }))
    assert [c["addr"] for c in cands] == [LIVES_HIGH]
    assert cands[0]["mirrors"] == [LIVES_HIGH + 1]


def test_nothing_watched_reports_that_rather_than_no_counter():
    cands, ev = lives_from_death_drives([])
    assert cands == []
    assert ev["usable"] == 0


# ---------------------------------------------------------------------
# _settled_start / _decrement_consensus's `same_start` gate
# (FIGHTGATE_MECHANISM_2026-08-25.md's attack_mash/approach_retreat
# probes) — the capture-timing edge case: `attack_mash` logs the RAM
# state AFTER a rep's own first (independently-seeded, per-rep) action
# already ran, so `log[0, addr]` is not guaranteed to be at rest. A
# health-bar/tally byte that chips away over several frames instead of
# jumping straight to its new value puts that transient in `log[0]`
# whenever a rep's opening action happens to land — and a naive
# frame-0 read then tells two reps that truly began the bout at the
# SAME stock apart, purely on capture timing.
# ---------------------------------------------------------------------

FIGHT_HP = 0x0398   # Punch-Out's own mac_hp, named in _decrement_consensus


def _tweening(n: int, sequence, hold: int) -> np.ndarray:
    """A byte that plays back `sequence` frame by frame — an
    in-flight HP-bar/tally animation still resolving — then holds at
    `hold` for the rest of the log. `sequence` need not include `hold`
    itself; playback just runs out and holding takes over."""
    col = np.empty(n, dtype=np.uint8)
    seq = list(sequence)
    for t in range(n):
        col[t] = seq[t] if t < len(seq) else hold
    return col


def test_settled_start_waits_out_an_in_flight_tween():
    """The bug this replaces: trusting `col[0]` outright. Here frame 0
    is mid-animation (40), not the value the bout actually rests at
    (96) — `_settled_start` must wait for the read to hold before
    reporting it, rather than repeating the first thing it saw."""
    tweening = _tweening(20, [40, 70, 88], hold=96)
    assert int(tweening[0]) == 40          # the naive frame-0 read: wrong
    assert _settled_start(tweening) == 96  # the settled read: correct


def test_settled_start_is_immediate_when_there_is_no_tween_at_all():
    """A rep whose opening action never touched the byte has nothing
    to wait out — frame 0 already IS the settled value."""
    already_stable = np.full(20, 96, dtype=np.uint8)
    assert _settled_start(already_stable) == 96


def test_settled_start_needs_min_stable_consecutive_reads_not_two():
    """A single repeat is not enough to call it settled — the byte
    could just be pausing mid-tween on its way to somewhere else.
    `min_stable` (DEATH_MIN_AGREE by default) sets the bar."""
    col = _tweening(20, [40, 40, 70, 88], hold=96)
    assert _settled_start(col, min_stable=2) == 40   # two reads is enough here
    assert _settled_start(col, min_stable=DEATH_MIN_AGREE) == 96


def test_settled_start_falls_back_to_the_raw_first_read_if_it_never_settles():
    """A column that keeps changing for the whole window has nothing
    to report but the old behaviour: whatever frame 0 was."""
    col = np.arange(20, dtype=np.uint8)  # never repeats
    assert _settled_start(col) == int(col[0]) == 0


def test_settled_start_is_bounded_so_a_later_plateau_is_not_mistaken_for_it():
    """The scan is capped at `horizon` frames (SETTLE_CAP, the same
    settle-window bound `settle_steps` uses) so a real, later combat
    plateau many frames in is never read back as the probe's start."""
    col = _tweening(200, list(range(40, 40 + SETTLE_CAP + 10)), hold=200)
    # never stable inside the first `horizon` frames -> falls back raw
    assert _settled_start(col, horizon=SETTLE_CAP) == int(col[0])


def test_decrement_consensus_same_start_survives_an_in_flight_tween():
    """Two reps that truly started the bout at the same stock (96)
    must not be told apart just because one of them was captured while
    its own opening action's chip-away tween was still resolving."""
    n = 300

    def _rep(sequence):
        log = np.zeros((n, RAM), dtype=np.uint8)
        log[:, FIGHT_HP] = _tweening(n, sequence, hold=96)
        return log

    logs = [_rep([40, 70, 88]),   # frame 0 mid-tween
            _rep([96])]           # frame 0 already at rest
    result = _decrement_consensus(logs, FIGHT_HP)
    assert result["same_start"] is True
    assert result["start"] == 96


def test_decrement_consensus_same_start_still_rejects_genuinely_different_starts():
    """The fix must not paper over a REAL disagreement — two reps
    that settle at different resting values are not the same start,
    tween or no tween."""
    n = 300

    def _rep(sequence, hold):
        log = np.zeros((n, RAM), dtype=np.uint8)
        log[:, FIGHT_HP] = _tweening(n, sequence, hold=hold)
        return log

    logs = [_rep([40, 70, 88], hold=96),
            _rep([10, 20], hold=64)]
    result = _decrement_consensus(logs, FIGHT_HP)
    assert result["same_start"] is False


# ---------------------------------------------------------------------------
# Zero-start lives guard (2026-08-25).
#
# A mechanism-coverage sweep across 43 onboarded profiles found 11 whose
# nominated `lives` byte read 0 at their own root state. Root cause, read
# directly from source: at start == 0 BOTH acceptance gates in
# `lives_from_death_drives` are structurally vacuous --
#   * `reaches_empty` sliced `log[:end+1]`, which includes the start sample,
#     so a byte that merely BEGAN at zero satisfied "reaches empty";
#   * `spends_its_stock` reduces to `0 <= spent <= -first`, satisfied by any
#     single decrement.
# Such a byte underflows 0 -> 255 on its first change and is misread as a
# death in every episode's first frames. Measured on Mega Man (USA): the
# search frontier froze at 10 cells; a correctly-nominated byte moved it to
# 16 with progress 16px -> 24px.
# ---------------------------------------------------------------------------

def _lives_log(col_by_addr, n_addr=8, n_steps=40):
    """Build a (steps, addrs) uint8 log from {addr: [values]}."""
    import numpy as np
    log = np.zeros((n_steps, n_addr), dtype=np.uint8)
    for a, vals in col_by_addr.items():
        for t, v in enumerate(vals[:n_steps]):
            log[t, a] = v
    return log


def test_zero_start_candidate_cannot_outrank_a_real_stock():
    """A byte that starts at 0 must never be nominated over one that
    carries an actual stock, however clean its decrement looks."""
    import numpy as np
    from scripts.discover_observables import lives_from_death_drives

    # addr 1: a REAL stock -- starts at 3, steps down to 0 one at a time.
    real = [3] * 10 + [2] * 10 + [1] * 10 + [0] * 10
    # addr 5: starts at 0, ticks once. Vacuously "reaches empty" and
    # vacuously "spends its stock" under the pre-fix gates.
    fake = [0] * 20 + [255] * 20

    logs = [_lives_log({1: real, 5: fake}) for _ in range(5)]
    drives = [{"log": lg} for lg in logs]
    cands, _ev = lives_from_death_drives(drives)

    assert cands, "expected at least one lives candidate"
    ranked = [c["addr"] for c in cands]
    assert 1 in ranked, f"the real stock at addr 1 was not surfaced: {ranked}"
    if 5 in ranked:
        assert ranked.index(1) < ranked.index(5), (
            f"zero-start byte (addr 5) outranked the real stock (addr 1): {ranked}"
        )
    # The top-ranked candidate must not be the zero-start byte.
    assert ranked[0] != 5, "nominated a byte that reads 0 at the start state"


def test_starts_nonzero_is_reported_per_candidate():
    """The new discriminator is visible in the receipt, not just in the sort."""
    from scripts.discover_observables import lives_from_death_drives
    real = [3] * 10 + [2] * 10 + [1] * 10 + [0] * 10
    logs = [_lives_log({1: real}) for _ in range(5)]
    cands, _ev = lives_from_death_drives([{"log": lg} for lg in logs])
    assert cands and "starts_nonzero" in cands[0], (
        "candidates must record starts_nonzero so a reader can audit the rank"
    )
    assert cands[0]["starts_nonzero"] is True


# ---------------------------------------------------------------------------
# Regime-truncation vacuity (2026-08-26) -- loophole 2, the same failure
# shape as the zero-start guard above through a different door.
#
# `_regime_split` ends the first regime at the first RISE, because a refill
# legitimately starts a new regime (a continue, a new bout). It had no
# minimum-duration guard and no cap on how OFTEN that may happen, so a byte
# that falls and bounces back within a handful of steps got a handful-of-steps
# "first regime" holding exactly ONE down-tick -- which satisfies both
# `reaches_empty` and `spends_its_stock` trivially (spent == start).
#
# Measured on Bad Dudes (docs/research/FALSE_DEATH_FANOUT_2026-08-26.md): the
# tool's top pick $00CD toggles 2 -> 0 -> 2 five to eighteen times per 700-step
# drive, first toggle at step 3, tied to the attack macros. Wiring it fires
# `GenericGame.is_dead` on ordinary A/B presses dozens of times an episode --
# strictly worse than leaving the profile alone, because a wrong lives
# nomination does not degrade death detection, it INVERTS it.
# ---------------------------------------------------------------------------

BAD_DUDES_N = 700          # DEATH_MAX_N, the real drive budget
OSC = 0x00CD               # the measured address, kept for readability


def _toggling(n: int, level: int, *, first: int, period: int,
              width: int = 2, low: int = 0) -> np.ndarray:
    """The Bad Dudes $00CD shape: a byte parked at `level` that drops to
    `low` and bounces back `width` steps later, over and over."""
    col = np.full(n, level, dtype=np.uint8)
    for t in range(first, n, period):
        col[t:min(t + width, n)] = low
    return col


def _blip_once(n: int, level: int, *, first: int, width: int = 4) -> np.ndarray:
    """The same shape with the repetition removed: ONE fall-and-bounce,
    early, then flat forever. The general form of the loophole -- an
    oscillation count alone cannot see it, only the duration guard can."""
    col = np.full(n, level, dtype=np.uint8)
    col[first:first + width] = 0
    return col


def test_an_attack_animation_toggle_is_not_a_life_counter():
    """Bad Dudes $00CD, at its measured rate. Every rep toggles far more
    often than a stock can be refilled, so it must not be nominated at
    all -- an honest empty result beats a nomination that inverts death."""
    drives = [{"log": _life_drive(BAD_DUDES_N,
                                  **{str(OSC): _toggling(BAD_DUDES_N, 2,
                                                         first=3 + k,
                                                         period=90)})["log"]}
              for k in range(5)]
    cands, _ev = lives_from_death_drives([{"log": d["log"]} for d in drives])
    assert [c["addr"] for c in cands] == [], (
        f"an attack-animation toggle was nominated as a stock: "
        f"{[(c['addr'], c['start'], c['spends']) for c in cands]}")


def test_the_toggle_never_outranks_a_real_stock_in_the_same_drive():
    """The ranking half: where a game DOES carry a real counter, the
    oscillator must not sit above it."""
    n = BAD_DUDES_N
    drives = []
    for k in range(5):
        drives.append(_life_drive(
            n,
            **{str(OSC): _toggling(n, 2, first=3 + k, period=90),
               str(LIVES_HIGH): _falling(n, 2, (200 + 9 * k, 460 + 9 * k))}))
    cands, _ev = lives_from_death_drives(drives)
    ranked = [c["addr"] for c in cands]
    assert LIVES_HIGH in ranked, f"the real stock was lost: {ranked}"
    assert ranked[0] == LIVES_HIGH, f"oscillator outranked the stock: {ranked}"
    assert OSC not in ranked, f"oscillator still nominated: {ranked}"


def test_a_single_early_bounce_is_not_a_spent_stock():
    """The general shape, with the repetition removed: one fall-and-bounce
    a few steps in. The first regime is too short to have HELD a life, so
    the stock it appears to spend is an artifact of the truncation."""
    n = BAD_DUDES_N
    drives = [{"log": _life_drive(
        n, **{str(OSC): _blip_once(n, 2, first=3 + k)})["log"]}
        for k in range(5)]
    cands, _ev = lives_from_death_drives(drives)
    assert [c["addr"] for c in cands] == [], (
        f"a 4-step bounce was credited as a spent stock: "
        f"{[(c['addr'], c['start'], c['spends']) for c in cands]}")


def test_a_continue_refill_hundreds_of_steps_in_is_still_a_life_counter():
    """The guard must not eat the case the split exists FOR: a counter
    spent to nothing and handed a fresh stock at a continue, measured on
    the SMB control as 1 -> 0 -> 255 and back up to 2."""
    n = BAD_DUDES_N
    drives = [_life_drive(n, **{str(LIVES_HIGH): _falling(
        n, 1, (150 + 9 * k, 330 + 9 * k),
        refill_at=(480 + 9 * k,), refill_to=2)}) for k in range(4)]
    cands, _ev = lives_from_death_drives(drives)
    assert [c["addr"] for c in cands] == [LIVES_HIGH]
    assert cands[0]["refilled_runs"] == 4


def test_a_slow_oscillator_is_caught_even_when_its_first_regime_is_long():
    """The two halves of the guard are independently load-bearing. Here the
    first bounce lands 300 steps in, so the duration half sees nothing --
    only counting refills across the WHOLE trace catches it. (The Bad Dudes
    measurement spans both: 5-18 toggles per drive, first at step 3.)"""
    n = BAD_DUDES_N
    drives = [_life_drive(n, **{str(OSC): _toggling(n, 2, first=300 + k,
                                                    period=90)})
              for k in range(5)]
    cands, _ev = lives_from_death_drives(drives)
    assert [c["addr"] for c in cands] == [], (
        f"a slow oscillator was nominated as a stock: "
        f"{[(c['addr'], c['start'], c['spends']) for c in cands]}")
