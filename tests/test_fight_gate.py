"""Fight-gate mechanism tests (FIGHTGATE_MECHANISM_2026-08-25.md).

Covers the generic mechanism at both ends of the pipeline, entirely with
synthetic RAM traces / plain integers — no live emulation, no ROM, no
Pool, matching the discipline `tests/test_discover_observables.py`
already uses for `lives_from_death_drives` (the "directly parallel but
inverted" sibling this mechanism reuses):

  * discover_observables.py — the pure adjudication core of
    `find_fight_health` (`fight_health_from_drives`, Gates FH1/FH2/FH3)
    and `find_round_gate` (`round_gate_from_drives`), plus the shared
    `_regime_split` / `_decrement_consensus` / `_mass_reset_boundaries`
    / `_corroborates_tally` helpers each is built from.
  * go_explore_solve.py — the pure `fight_gate_step` / `fight_gate_mass_
    reset` integral math, `GenericGame`'s `source: fight_gate` profile
    parsing, and `Solver._xram`'s pseudo-RAM extension wiring (via
    SimpleNamespace + MethodType stand-ins, the pattern
    tests/test_room_fp.py and tests/test_go_explore_atomic_peek.py
    already use to test Solver internals without a Pool).
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts.discover_observables import (
    DEATH_MIN_AGREE,
    RAM_SIZE,
    _corroborates_tally,
    _decrement_consensus,
    _mark_mirror_siblings,
    _mass_reset_boundaries,
    _range_plausibility_penalty,
    _regime_split,
    FIGHT_MASH_WEIGHTS,
    fight_health_from_drives,
    round_gate_from_drives,
    emit_solve_yaml,
)
from scripts.go_explore_solve import (
    FIGHT_HI,
    FIGHT_LO,
    FIGHT_RESET_THRESHOLD,
    ODO_ALT,
    ODO_LO,
    ROOM_HI,
    ROOM_LO,
    GenericGame,
    Solver,
    fight_gate_mass_reset,
    fight_gate_step,
)

OPP, MAC, ROUND = 0x0398, 0x0392, 0x0006


# ===========================================================================
# discover_observables.py — pure gate machinery
# ===========================================================================

def _drive(n: int, cols: dict) -> dict:
    """One attack_mash/approach_retreat rep: a RAM log with the named
    columns set, the rest zero. Shape-compatible with what
    `Discoverer.attack_mash`/`approach_retreat` return."""
    log = np.zeros((n, RAM_SIZE), dtype=np.uint8)
    for addr, col in cols.items():
        log[:, addr] = col
    return {"log": log, "steps": n}


def _stepped_col(n: int, start: int, events: dict) -> np.ndarray:
    """A byte holding `start`, jumping to `events[t]` at each row `t`."""
    col = np.empty(n, dtype=np.int64)
    cur = start
    for t in range(n):
        if t in events:
            cur = events[t]
        col[t] = cur
    return (col % 256).astype(np.uint8)


# --- _regime_split -------------------------------------------------------

def test_regime_split_stops_at_the_first_rise():
    mv = np.array([0, -3, 0, -2, 0, 5, -9, 0])
    assert _regime_split(mv) == 5           # index of the +5


def test_regime_split_is_the_whole_series_when_it_never_rises():
    mv = np.array([0, -1, 0, -1, 0])
    assert _regime_split(mv) == len(mv)


def test_regime_split_on_an_all_zero_series():
    assert _regime_split(np.zeros(6)) == 6


# --- _decrement_consensus -------------------------------------------------

def test_decrement_consensus_passes_a_byte_that_agrees_across_reps():
    logs = []
    for k in range(5):
        log = np.zeros((200, RAM_SIZE), dtype=np.uint8)
        log[:, OPP] = _stepped_col(200, 96, {50 + k: 80 - k, 150 + k: 60 - k})
        logs.append(log)
    res = _decrement_consensus(logs, OPP, min_agree=DEATH_MIN_AGREE)
    assert res["passes"] is True
    assert res["watched"] == 5 and res["agreeing"] == 5
    assert res["start"] == 96


def test_decrement_consensus_abstains_on_a_run_that_never_moved():
    logs = [{"log": np.zeros((100, RAM_SIZE), dtype=np.uint8)}]
    log = np.zeros((100, RAM_SIZE), dtype=np.uint8)
    log[:, OPP] = 96
    res = _decrement_consensus([log], OPP)
    assert res["watched"] == 1 and res["agreeing"] == 0
    assert res["passes"] is False


def test_decrement_consensus_rejects_a_byte_that_starts_empty():
    """Found live against the real Punch-Out ROM: several zero-page
    bytes tied to Little Mac's OWN punch animation (ticking on every
    A/B mash regardless of whether it lands, flat under the defense-
    only control) satisfy the direction+agreement test just as well as
    the true opp_hp byte — and every one of them starts the probe at 0,
    wrapping DOWN through 255 rather than draining from a full value.
    A stock has something in it when the probe begins; start=0 is
    disqualifying regardless of how clean the agreement otherwise is."""
    logs = []
    for k in range(5):
        log = np.zeros((200, RAM_SIZE), dtype=np.uint8)
        log[:, 0x04B4] = _stepped_col(200, 0, {50 + k: -20 % 256,
                                               150 + k: -40 % 256})
        logs.append(log)
    res = _decrement_consensus(logs, 0x04B4, min_agree=DEATH_MIN_AGREE)
    assert res["start"] == 0
    assert res["passes"] is False


def test_decrement_consensus_rejects_disagreeing_starts():
    log_a = np.zeros((50, RAM_SIZE), dtype=np.uint8)
    log_a[:, OPP] = _stepped_col(50, 96, {10: 80})
    log_b = np.zeros((50, RAM_SIZE), dtype=np.uint8)
    log_b[:, OPP] = _stepped_col(50, 40, {10: 20})   # different start value
    res = _decrement_consensus([log_a, log_b], OPP, min_agree=2)
    assert res["same_start"] is False
    assert res["passes"] is False


# --- fight_health_from_drives: the self/foe discriminator (§3.3) ---------

def _fight_gate_scene(n=800, reps=5, dmg_opp=8, dmg_mac=6, defense_dmg=4):
    """Pilot-Evidence-shaped scene: attack_mash lands hits on BOTH opp_hp
    and mac_hp (offense draws return fire); approach_retreat (no attack
    input) never touches opp_hp but still chips mac_hp a little."""
    atk, apr = [], []
    for k in range(reps):
        atk.append(_drive(n, {
            OPP: _stepped_col(n, 96, {100 + k: 80 - k, 300 + k: 60 - k,
                                      600 + k: 40 - k}),
            MAC: _stepped_col(n, 96, {50 + k: 70 - k, 200 + k: 50 - k,
                                      400 + k: 30 - k, 550 + k: 10 - k}),
        }))
        apr.append(_drive(n, {
            OPP: np.full(n, 96, dtype=np.uint8),
            MAC: _stepped_col(n, 96, {80 + k: 96 - defense_dmg,
                                      400 + k: 96 - 2 * defense_dmg}),
        }))
    return atk, apr


def test_fight_health_separates_foe_hp_from_self_hp():
    """Pilot Evidence #1's exact failure, closed: an offense-only signal
    alone cannot tell mac_hp from opp_hp; the defense-only control can."""
    atk, apr = _fight_gate_scene()
    res = fight_health_from_drives(atk, apr, self_hp_addr=MAC)
    assert res["kind"] == "foe_hp"
    assert res["addr"] == OPP
    foe_addrs = [c["addr"] for c in res["foe_hp_candidates"]]
    self_addrs = [c["addr"] for c in res["self_hp_candidates"]]
    assert OPP in foe_addrs and OPP not in self_addrs
    assert MAC in self_addrs and MAC not in foe_addrs


def test_fight_health_flags_self_hp_conflict_only_when_adjacent():
    atk, apr = _fight_gate_scene()
    res = fight_health_from_drives(atk, apr, self_hp_addr=MAC)
    best = res["detail"]
    assert best["self_hp_conflict"] is False       # OPP is far from MAC
    res_close = fight_health_from_drives(atk, apr, self_hp_addr=OPP + 1)
    assert res_close["detail"]["self_hp_conflict"] is True


def test_fight_health_reports_self_hp_conflict_field_from_hp_lives():
    atk, apr = _fight_gate_scene()
    res = fight_health_from_drives(atk, apr, self_hp_addr=MAC)
    assert res["self_hp_conflict"] == MAC


def test_fight_health_gate_fh3_survives_a_mid_probe_refill():
    """Probe 3's own finding: mac_hp (here modeled as the foe byte) hits
    0 and RISES back to 72 inside one continuous probe, no death, no
    round change. The regime split must not reject it."""
    n = 800
    atk = [_drive(n, {OPP: _stepped_col(
        n, 96, {150 + k: 60, 300 + k: 20, 450 + k: 0, 500 + k: 96,
                650 + k: 40})}) for k in range(5)]
    apr = [_drive(n, {OPP: np.full(n, 96, dtype=np.uint8)}) for _ in range(5)]
    res = fight_health_from_drives(atk, apr)
    assert res["kind"] == "foe_hp" and res["addr"] == OPP


def test_fight_health_insufficient_probe_when_attack_mash_is_empty():
    res = fight_health_from_drives([], [{"log": np.zeros((10, RAM_SIZE),
                                                         dtype=np.uint8)}])
    assert res["kind"] == "insufficient_probe" and res["addr"] is None


def test_fight_health_none_when_nothing_clears_fh1():
    n = 100
    atk = [_drive(n, {}) for _ in range(5)]     # nothing ever moves
    apr = [_drive(n, {}) for _ in range(5)]
    res = fight_health_from_drives(atk, apr)
    assert res["kind"] == "none"
    assert res["foe_hp_candidates"] == []


def test_fight_health_ranking_prefers_varying_outcomes_over_identical_ones():
    """Found live against the real Punch-Out ROM: a handful of zero-page
    bytes decremented by the EXACT SAME amount in all 3 differently-
    seeded attack_mash reps — the signature of a byte driven by elapsed
    STEP COUNT alone (every rep runs the same number of steps), not by
    how many punches actually landed (which varies with the random
    sequence each rep drew). The true opp_hp byte's nets in that trace
    were three DIFFERENT numbers. A candidate with more spread across
    reps must outrank an equally-agreeing one with none, even when the
    identical-outcome byte's raw magnitude is larger."""
    n = 300
    # reactive: three DIFFERENT outcomes across seeds (spread=55)
    atk = []
    for k, drop in enumerate([58, 19, 74]):
        atk.append(_drive(n, {OPP: _stepped_col(n, 96, {50 + k: 96 - drop})}))
    for k in range(3):
        # input-invariant: IDENTICAL -105 every rep despite different
        # random seeds, and a bigger raw magnitude than opp_hp's worst
        # single net (-74) — a naive "biggest drop wins" rank would
        # pick this instead. Nonzero start so `start > 0` alone can't
        # explain the outcome — only the spread tiebreak can.
        atk[k]["log"][:, 0x04B4] = _stepped_col(n, 200, {80: 200 - 105})
    apr = [_drive(n, {}) for _ in range(3)]
    res = fight_health_from_drives(atk, apr)
    assert res["addr"] == OPP


def _punchout_receipt_scene():
    """The exact attack-probe numbers from the live Punch-Out receipt
    (`runs/fight_gate/discover_punchout.json`, the FIGHTGATE_MECHANISM
    validation run): the true opp_hp byte (0x0398) and its two adjacent
    display mirrors (0x0399/0x039A), alongside three zero-page decoys
    that ALSO clear FH1+FH2 in that receipt with real (non-identical)
    cross-rep spread — one of them wider than opp_hp's own. Rebuilt here
    as a synthetic scene (one wrap-aware jump per rep reproducing each
    candidate's reported `start`/`attack_nets`) so the regression needs
    neither the ROM nor the receipt file to run.
    """
    reps = 5
    addrs = {
        0x0398: (96, [-50, -19, -74, -64, -96]),
        0x0399: (96, [-50, -19, -74, -64, -96]),   # mirror
        0x039A: (96, [-50, -19, -74, -64, -96]),   # mirror
        0x009E: (253, [-198, -176, -176, -176, -253]),
        0x00EE: (73, [-11, -89, -65, -73, -16]),
        0x0711: (82, [-113, -113, -126, -113, -126]),
    }
    n = 200
    atk = []
    for k in range(reps):
        cols = {addr: _stepped_col(n, start, {50: start + nets[k]})
                for addr, (start, nets) in addrs.items()}
        atk.append(_drive(n, cols))
    apr = [_drive(n, {}) for _ in range(reps)]
    return atk, apr


def test_fight_health_ranks_the_true_byte_first_on_the_punchout_receipt():
    """Registered qualification (FIGHTGATE_MECHANISM_2026-08-25.md): on
    the live Punch-Out receipt, the true opp_hp byte (0x0398) cleared
    FH1+FH2 but ranked #3 of 5 distinct addresses under the spread
    tiebreak alone — beaten by two zero-page decoys with equal or wider
    cross-rep spread. Two further, general corroboration terms must
    move it to #1: `_mark_mirror_siblings` credits it for having two
    byte-for-byte adjacent mirrors (a HUD/shadow-copy signature neither
    decoy has), and `_range_plausibility_penalty` demotes both decoys
    for a net magnitude that exceeds their own starting value (a
    free-running-counter signature the true byte does not show)."""
    atk, apr = _punchout_receipt_scene()
    res = fight_health_from_drives(atk, apr)
    assert res["kind"] == "foe_hp"
    assert res["addr"] == 0x0398
    top3 = [c["addr"] for c in res["foe_hp_candidates"][:3]]
    assert set(top3) == {0x0398, 0x0399, 0x039A}


def test_fight_health_tally_corroboration_is_advisory_not_a_gate():
    """A candidate with NO periodic partner still wins on FH1+FH2 alone
    (§3.2 point 6: disagreement never disqualifies)."""
    atk, apr = _fight_gate_scene()
    res = fight_health_from_drives(atk, apr, self_hp_addr=MAC)
    assert res["kind"] == "foe_hp"
    # tally_corroborated is reported, whichever way it falls, but never
    # gates the winner:
    assert "tally_corroborated" in res["detail"]


# --- _corroborates_tally ---------------------------------------------------

def test_corroborates_tally_true_when_the_candidate_decreases_in_window():
    hist = np.zeros((100, RAM_SIZE), dtype=np.uint8)
    hist[:, OPP] = np.concatenate([np.full(20, 96), np.full(80, 80)])
    assert _corroborates_tally(hist, [(0, 40)], OPP) is True


def test_corroborates_tally_false_with_no_windows_or_no_movement():
    hist = np.zeros((100, RAM_SIZE), dtype=np.uint8)
    hist[:, OPP] = 96
    assert _corroborates_tally(hist, [], OPP) is False
    assert _corroborates_tally(hist, [(0, 40)], OPP) is False


# --- _mark_mirror_siblings / _range_plausibility_penalty (ranking) -------

def test_mark_mirror_siblings_credits_an_exact_adjacent_match():
    cands = [
        {"addr": 0x0398, "start": 96, "attack_nets": [-50, -19, -74]},
        {"addr": 0x0399, "start": 96, "attack_nets": [-50, -19, -74]},
        {"addr": 0x00EE, "start": 73, "attack_nets": [-11, -89, -65]},
    ]
    _mark_mirror_siblings(cands)
    assert cands[0]["mirror_siblings"] == 1
    assert cands[1]["mirror_siblings"] == 1
    assert cands[2]["mirror_siblings"] == 0


def test_mark_mirror_siblings_ignores_a_near_address_that_disagrees():
    cands = [
        {"addr": 0x0398, "start": 96, "attack_nets": [-50, -19, -74]},
        {"addr": 0x0399, "start": 96, "attack_nets": [-11, -19, -74]},  # close but not equal
        {"addr": 0x0500, "start": 96, "attack_nets": [-50, -19, -74]},  # equal but too far
    ]
    _mark_mirror_siblings(cands, window=4)
    assert cands[0]["mirror_siblings"] == 0
    assert cands[1]["mirror_siblings"] == 0
    assert cands[2]["mirror_siblings"] == 0


def test_range_plausibility_penalty_is_zero_within_start():
    c = {"start": 96, "attack_nets": [-50, -19, -74, -96]}
    assert _range_plausibility_penalty(c) == 0.0


def test_range_plausibility_penalty_rises_past_start():
    c = {"start": 73, "attack_nets": [-11, -89, -65]}
    assert _range_plausibility_penalty(c) == pytest.approx((89 - 73) / 73)


# --- _mass_reset_boundaries / round_gate_from_drives (§3.4) ---------------

def _reset_log(n: int, resets: list, round_vals: list) -> np.ndarray:
    log = np.zeros((n, RAM_SIZE), dtype=np.uint8)
    rv, ri = round_vals[0], 0
    for t in range(n):
        if ri < len(resets) and t == resets[ri]:
            log[t, 0x10:0x200] = (t * 37) % 256    # mass RAM rewrite
            ri += 1
            rv = round_vals[ri]
        log[t, ROUND] = rv
    return log


def test_mass_reset_boundaries_finds_every_boundary_once():
    log = _reset_log(700, [200, 450], [1, 2, 3])
    bounds = _mass_reset_boundaries(log, FIGHT_RESET_THRESHOLD)
    assert len(bounds) == 2
    assert all(abs(b - r) <= 2 for b, r in zip(bounds, [200, 450]))


def test_mass_reset_boundaries_needs_no_events_at_all():
    assert _mass_reset_boundaries(np.zeros((100, RAM_SIZE), dtype=np.uint8),
                                  FIGHT_RESET_THRESHOLD) == []


def test_round_gate_nominates_the_monotone_round_byte():
    drives = [{"log": _reset_log(700, [200, 450], [1, 2, 3]), "steps": 700}
              for _ in range(3)]
    res = round_gate_from_drives(drives, threshold=FIGHT_RESET_THRESHOLD)
    assert res["kind"] == "round_gate" and res["addr"] == ROUND


def test_round_gate_is_void_not_fail_when_no_boundary_is_ever_crossed():
    """§2's VOID/FAIL split: an under-powered probe is an INSTRUMENT
    finding, not a behaviour verdict."""
    drives = [{"log": np.zeros((100, RAM_SIZE), dtype=np.uint8), "steps": 100}]
    res = round_gate_from_drives(drives, threshold=FIGHT_RESET_THRESHOLD)
    assert res["kind"] == "insufficient_probe"
    assert res["void"] is True


def test_round_gate_rejects_an_attempt_counter_that_resets_to_start():
    """A lost-bout reload of the SAME opponent must not be mistaken for
    a round advance — the check a room counter never needed."""
    n = 700
    log = np.zeros((n, RAM_SIZE), dtype=np.uint8)
    ATTEMPT = 0x0007
    resets = [200, 450]
    ri, val = 0, 1
    for t in range(n):
        if ri < len(resets) and t == resets[ri]:
            log[t, 0x10:0x200] = (t * 41) % 256
            ri += 1
            val = 1                        # same opponent: resets to start
        log[t, ATTEMPT] = val
    drives = [{"log": log, "steps": n} for _ in range(3)]
    res = round_gate_from_drives(drives, threshold=FIGHT_RESET_THRESHOLD)
    assert ATTEMPT not in [c["addr"] for c in res["candidates"]]


def test_round_gate_rejects_a_byte_that_drops_at_a_boundary():
    n = 700
    log = np.zeros((n, RAM_SIZE), dtype=np.uint8)
    DROPPER = 0x0008
    log[:200, DROPPER] = 5
    log[200:450, DROPPER] = 2               # dropped at the reset: rejected
    log[450:, DROPPER] = 3
    log[200, 0x10:0x200] = 11
    log[450, 0x10:0x200] = 13
    drives = [{"log": log, "steps": n} for _ in range(3)]
    res = round_gate_from_drives(drives, threshold=FIGHT_RESET_THRESHOLD)
    assert DROPPER not in [c["addr"] for c in res["candidates"]]


# --- FIGHT_MASH_WEIGHTS (attack_mash's action distribution, §3.1) --------

def test_fight_mash_weights_sum_to_one_and_favor_offense():
    assert FIGHT_MASH_WEIGHTS.sum() == pytest.approx(1.0)
    # order: A, B, A|B, fwd, rev, NOOP — offense mass is the first three
    assert FIGHT_MASH_WEIGHTS[:3].sum() == pytest.approx(0.60)


# --- emit_solve_yaml: the fight-gate progress block ------------------------

def _base_findings(**extra) -> dict:
    return {"rom": "roms/Fake (USA).nes", "forward": "right",
            "progress": {"forward": "right", "room_counter": None,
                        "candidates": [], "recommended": None},
            "room_counters": [], "y": [],
            "hp_lives": {"kind": "none", "addr": None, "note": "n/a"},
            **extra}


def test_emit_solve_yaml_writes_fight_gate_progress_when_no_spatial_signal():
    fh = {"kind": "foe_hp", "addr": OPP,
          "detail": {"start": 96, "attack_agree": 5, "attack_watched": 5,
                     "defense_watched": 5, "self_hp_conflict": False}}
    rg = {"kind": "round_gate", "addr": ROUND}
    text = emit_solve_yaml("roms/Fake (USA).nes",
                           _base_findings(fight_health=fh, round_gate=rg))
    assert (f"progress: {{source: fight_gate, foe_hp: 0x{OPP:04X}, "
            f"foe_hp_start: 0x60, round: 0x{ROUND:04X}}}") in text
    assert "WARNING" not in text


def test_emit_solve_yaml_warns_on_a_self_hp_conflict():
    fh = {"kind": "foe_hp", "addr": OPP,
          "detail": {"start": 96, "attack_agree": 5, "attack_watched": 5,
                     "defense_watched": 5, "self_hp_conflict": True}}
    text = emit_solve_yaml("roms/Fake (USA).nes",
                           _base_findings(fight_health=fh, round_gate=None))
    assert "WARNING" in text and "self/foe aliasing" in text


def test_emit_solve_yaml_reports_insufficient_round_probe_as_a_comment():
    fh = {"kind": "foe_hp", "addr": OPP,
          "detail": {"start": 96, "attack_agree": 5, "attack_watched": 5,
                     "defense_watched": 5, "self_hp_conflict": False}}
    rg = {"kind": "insufficient_probe", "addr": None, "void": True,
          "note": "no bout boundary observed"}
    text = emit_solve_yaml("roms/Fake (USA).nes",
                           _base_findings(fight_health=fh, round_gate=rg))
    assert "round: 0x" not in text.split("progress:")[1].split("\n")[0]
    assert "# round_gate: insufficient_probe" in text


def test_emit_solve_yaml_never_touched_when_fight_gate_was_not_requested():
    """A profile that never runs --fight-gate gets byte-identical output
    (§3.5) — no fight_health/round_gate keys present at all."""
    text = emit_solve_yaml("roms/Fake (USA).nes", _base_findings())
    assert "fight_gate" not in text
    assert "fight_health" not in text
    assert "round_gate" not in text


# ===========================================================================
# go_explore_solve.py — the consumption side
# ===========================================================================

# --- fight_gate_step (pure integral math, §4.1) ---------------------------

def test_fight_gate_step_first_observation_is_a_rearm_not_a_delta():
    cum, prev = fight_gate_step(None, 96, is_transition=False, cum=0)
    assert cum == 0 and prev == 96


def test_fight_gate_step_banks_a_drop():
    cum, prev = fight_gate_step(96, 80, is_transition=False, cum=0)
    assert cum == 16 and prev == 80


def test_fight_gate_step_ignores_a_rise():
    cum, prev = fight_gate_step(40, 96, is_transition=False, cum=10)
    assert cum == 10 and prev == 96          # unchanged: no credit for a rise


def test_fight_gate_step_accumulates_across_many_steps():
    cum, prev = 0, 96
    for now in (90, 90, 70, 70, 40):
        cum, prev = fight_gate_step(prev, now, is_transition=False, cum=cum)
    assert cum == 96 - 40 and prev == 40


def test_fight_gate_step_transition_rearms_with_no_windfall():
    """The design's named failure mode: comparing a fresh opponent's
    full HP against the previous opponent's near-empty reading (or a
    load frame's garbage) must never be read as a giant hit."""
    cum, prev = fight_gate_step(3, 96, is_transition=True, cum=93)
    assert cum == 93                          # no +93 windfall from the jump
    assert prev == 96
    # the FIRST step after the re-arm behaves normally again:
    cum2, prev2 = fight_gate_step(prev, 80, is_transition=False, cum=cum)
    assert cum2 == 93 + 16 and prev2 == 80


def test_fight_gate_step_clamps_to_the_16_bit_cap():
    cum, _ = fight_gate_step(60000, 0, is_transition=False, cum=0xFFFE,
                             cap=0xFFFF)
    assert cum == 0xFFFF


# --- fight_gate_mass_reset --------------------------------------------------

def test_fight_gate_mass_reset_false_on_first_observation():
    ram = np.zeros(RAM_SIZE, dtype=np.uint8)
    assert fight_gate_mass_reset(None, ram, threshold=350.0) is False


def test_fight_gate_mass_reset_false_on_ordinary_play_churn():
    prev = np.zeros(RAM_SIZE, dtype=np.uint8)
    ram = prev.copy()
    ram[0x10:0x30] = 7                       # 32 bytes changed
    assert fight_gate_mass_reset(prev, ram, threshold=350.0) is False


def test_fight_gate_mass_reset_true_on_a_bout_boundary_rewrite():
    prev = np.zeros(RAM_SIZE, dtype=np.uint8)
    ram = prev.copy()
    ram[0x00:0x400] = 9                      # 1024 bytes changed
    assert fight_gate_mass_reset(prev, ram, threshold=350.0) is True


# --- GenericGame: `progress: {source: fight_gate, ...}` parsing ----------

def _fg_profile(**progress):
    p = {"source": "fight_gate", "foe_hp": OPP}
    p.update(progress)
    return {"solve": {"rom": "roms/x.nes", "progress": p, "y": 0x003F,
                      "level_key": [0x0028], "lives": 0x002A}}


def test_fight_gate_source_is_off_by_default():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"lo": 0x0040, "hi": 0x0041},
                      "y": 0x003F, "level_key": [0x0028], "lives": 0x002A}}
    g = GenericGame(prof)
    assert g.foe_hp_addr is None
    assert g.fight_round_addr is None


def test_fight_gate_source_sets_foe_hp_and_remaps_lo_hi():
    g = GenericGame(_fg_profile())
    assert g.foe_hp_addr == OPP
    assert g.fight_round_addr is None
    assert g.foe_hp_start == 0
    ram = bytearray(FIGHT_HI + 1)
    ram[FIGHT_LO], ram[FIGHT_HI] = 0x34, 0x12
    assert g.progress(ram) == 0x1234          # reads through FIGHT_LO/HI


def test_fight_gate_source_reads_optional_round_and_start():
    g = GenericGame(_fg_profile(round=ROUND, foe_hp_start=0x60))
    assert g.fight_round_addr == ROUND
    assert g.foe_hp_start == 0x60


def test_fight_gate_source_requires_foe_hp():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"source": "fight_gate"},
                      "y": 0x3F, "level_key": [0x28], "lives": 0x2A}}
    with pytest.raises(SystemExit, match="foe_hp"):
        GenericGame(prof)


# --- Solver._xram wiring (SimpleNamespace + MethodType, no Pool) ---------

def _fight_solver(*, foe_hp=OPP, round_addr=None, odo_on=False,
                  room_fp=None, nw=2):
    fake = SimpleNamespace(
        _odo=odo_on, _fight=True,
        _odo_now=[(0, 0)] * nw, _odo_scene=[0] * nw,
        game=SimpleNamespace(odometer_axis=None if not odo_on else "x",
                             odometer_sign=1,
                             foe_hp_addr=foe_hp,
                             fight_round_addr=round_addr),
        room_fp=room_fp,
        _room_ord=np.zeros(nw, dtype=np.uint16),
        _fight_prev_hp=[None] * nw,
        _fight_cum=np.zeros(nw, dtype=np.uint32),
        _fight_round_prev=[None] * nw,
        _fight_prev_ram=[None] * nw,
    )
    fake._fight_step = MethodType(Solver._fight_step, fake)
    return fake


def _ram_with(foe_val: int, round_val: int | None = None,
             n: int = RAM_SIZE) -> np.ndarray:
    ram = np.zeros(n, dtype=np.uint8)
    ram[OPP] = foe_val
    if round_val is not None:
        ram[ROUND] = round_val
    return ram


def test_xram_fight_only_extends_9_bytes_and_writes_fight_lo_hi():
    fake = _fight_solver()
    ext = Solver._xram(fake, _ram_with(96), 0)
    assert len(ext) == RAM_SIZE + 9
    # first observation: re-arm, no damage banked yet
    assert ext[FIGHT_LO] == 0 and ext[FIGHT_HI] == 0
    ext2 = Solver._xram(fake, _ram_with(80), 0)
    assert ext2[FIGHT_LO] == 16 and ext2[FIGHT_HI] == 0


def test_xram_fight_only_leaves_the_odometer_room_fp_block_zeroed():
    fake = _fight_solver()
    ext = Solver._xram(fake, _ram_with(96), 0)
    assert all(ext[a] == 0 for a in (ODO_LO, ODO_LO + 1, ODO_LO + 2,
                                     ODO_LO + 3, ROOM_LO, ROOM_HI, ODO_ALT))


def test_xram_off_when_neither_odometer_nor_fight_configured():
    fake = SimpleNamespace(_odo=False)     # no _fight attribute at all
    ram = bytes(RAM_SIZE)
    assert Solver._xram(fake, ram, 0) is ram


def test_xram_fight_gate_per_worker_state_is_independent():
    fake = _fight_solver(nw=2)
    Solver._xram(fake, _ram_with(96), 0)
    Solver._xram(fake, _ram_with(96), 1)
    ext0 = Solver._xram(fake, _ram_with(70), 0)
    ext1 = Solver._xram(fake, _ram_with(90), 1)
    assert ext0[FIGHT_LO] == 26                # worker 0: 96 -> 70
    assert ext1[FIGHT_LO] == 6                 # worker 1: 96 -> 90


def test_xram_fight_gate_round_transition_rearms_without_a_windfall():
    fake = _fight_solver(round_addr=ROUND)
    Solver._xram(fake, _ram_with(20, round_val=1), 0)   # near-dead opponent
    # round advances to a fresh opponent at full HP -> no fake windfall
    ext = Solver._xram(fake, _ram_with(96, round_val=2), 0)
    assert ext[FIGHT_LO] == 0 and ext[FIGHT_HI] == 0
    # the NEXT hit banks normally again
    ext2 = Solver._xram(fake, _ram_with(80, round_val=2), 0)
    assert ext2[FIGHT_LO] == 16


def test_xram_fight_gate_mass_reset_fallback_rearms_with_no_round_byte():
    fake = _fight_solver(round_addr=None)
    ram0 = _ram_with(20)
    Solver._xram(fake, ram0, 0)
    ram1 = _ram_with(96)
    ram1[0x10:0x600] = 9                       # a bout-boundary-sized rewrite
    ext = Solver._xram(fake, ram1, 0)
    assert ext[FIGHT_LO] == 0 and ext[FIGHT_HI] == 0


def test_xram_fight_gate_coexists_with_odometer_and_room_fp():
    fake = _fight_solver(odo_on=True, room_fp={"anything": True})
    fake._odo_now[0] = (0x123456, 0x0ABCDE)
    fake._odo_scene[0] = 7
    fake._room_ord[0] = 0x0102
    ext = Solver._xram(fake, _ram_with(96), 0)
    assert len(ext) == RAM_SIZE + 9
    assert (ext[ODO_LO], ext[ODO_LO + 1], ext[ODO_LO + 2]) == (0x56, 0x34, 0x12)
    assert ext[ODO_LO + 3] == 7
    assert (ext[ROOM_LO], ext[ROOM_HI]) == (0x02, 0x01)
    assert ext[FIGHT_LO] == 0 and ext[FIGHT_HI] == 0


# --- Solver._fight_reset ---------------------------------------------------

def test_fight_reset_clears_all_per_worker_state():
    fake = _fight_solver(nw=1)
    Solver._xram(fake, _ram_with(96), 0)
    Solver._xram(fake, _ram_with(70), 0)
    assert fake._fight_cum[0] == 26
    Solver._fight_reset(fake, 0)
    assert fake._fight_prev_hp[0] is None
    assert fake._fight_cum[0] == 0
    assert fake._fight_round_prev[0] is None
    assert fake._fight_prev_ram[0] is None
    # the next observation is a fresh re-arm, not a delta against the
    # PREVIOUS worker-slot occupant's leftover HP:
    ext = Solver._xram(fake, _ram_with(50), 0)
    assert ext[FIGHT_LO] == 0


def test_fight_reset_is_a_noop_when_fight_gate_is_not_configured():
    fake = SimpleNamespace(_fight=False)
    Solver._fight_reset(fake, 0)              # must not raise
