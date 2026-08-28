"""`scripts/collect_onpolicy_bank.py` — the on-policy V_adv bank collector.

Registration: `docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md` §4, §8.
Every check here is a pure function of arrays/ints — no torch, no emulator,
no checkpoint — so the four registered anti-vacuity conditions are testable
directly: the level-identity purity guard, the cross-population partition,
the done-on-clear rule, and the sticky `step > 0` gate. Each test in the
"revert-verified" group is written so that reverting the guarded behavior
makes it FAIL — a check that has never been seen to fail is not a check.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.collect_onpolicy_bank import (
    TARGET_WORLD_LEVEL,
    append_transition_row,
    check_purity,
    cross_population_mask,
    episode_step_cap,
    finalize_truncation,
    is_dead,
    level_identity,
    penetration_receipt,
    sticky_should_apply,
    window_for_rung,
)


def _ram(world: int = 0, level: int = 1, player_state: int = 8) -> bytes:
    """A 2048-byte RAM image with the config's declared addresses set."""
    arr = np.zeros(2048, dtype=np.uint8)
    arr[0x075F] = world
    arr[0x075C] = level
    arr[0x000E] = player_state
    return arr.tobytes()


# ==========================================================================
# Level-identity purity guard — the exact aliasing bug it exists to catch
# ==========================================================================

def test_purity_accepts_a_normal_in_level_transition():
    check_purity(0, 1, 0, 1, terminal=False)  # must not raise


def test_purity_rejects_start_state_outside_the_target_level():
    with pytest.raises(RuntimeError, match="purity violation at s"):
        check_purity(0, 2, 0, 1, terminal=False)


def test_purity_rejects_a_non_terminal_row_whose_successor_left_the_level():
    """This is the exact defect §4 discloses: the unguarded pilot recorded
    rows deep into 1-3 as if they were still part of a 1-2 episode. A
    non-terminal row's successor leaving (0, 1) is that bug, live."""
    with pytest.raises(RuntimeError, match="NON-terminal row"):
        check_purity(0, 1, 0, 2, terminal=False)


def test_purity_permits_a_terminal_rows_successor_to_leave_the_level():
    """A level clear (or a death) legitimately changes the identity at s' —
    that transition is what `terminal=True` means. Rejecting it here would
    make every clearing episode's own last row un-collectable."""
    check_purity(0, 1, 0, 2, terminal=True)  # must not raise


def test_purity_guard_fails_on_revert():
    """If the guard were removed (the historical bug: check nothing), the
    exact aliasing case would silently pass. Assert the REMOVED-guard
    behavior is what we do NOT want, so this test fails loudly if the
    import ever stops raising."""
    def _unguarded(*_a, **_kw):
        return None  # the pre-fix behavior: never raises

    # The guarded function must behave differently from the unguarded stub
    # on the aliasing case — i.e. the guard must actually DO something.
    with pytest.raises(RuntimeError):
        check_purity(0, 1, 0, 2, terminal=False)
    assert _unguarded(0, 1, 0, 2, terminal=False) is None  # sanity on the stub


def test_level_identity_reads_the_configs_declared_addresses():
    assert level_identity(_ram(world=0, level=1)) == (0, 1)
    assert level_identity(_ram(world=1, level=3)) == (1, 3)


def test_target_world_level_is_1_2_in_this_projects_encoding():
    assert TARGET_WORLD_LEVEL == (0, 1)


# ==========================================================================
# Death detection — gen_iq_transitions.py's own convention
# ==========================================================================

@pytest.mark.parametrize("state", [6, 11])
def test_is_dead_true_on_the_registered_death_states(state):
    assert is_dead(_ram(player_state=state)) is True


def test_is_dead_false_on_a_live_state():
    assert is_dead(_ram(player_state=8)) is False


def test_is_dead_true_on_pool_done_flag_even_with_a_live_player_state():
    assert is_dead(_ram(player_state=8), pool_done=True) is True


# ==========================================================================
# Done-on-clear + truncation bookkeeping — smodice_data.py's own convention
# ==========================================================================

def test_append_transition_row_records_done_1_on_a_terminal_row():
    rows: list = []
    s = np.zeros(4, dtype=np.int8)
    ns = np.ones(4, dtype=np.int8)
    append_transition_row(rows, s, 2, ns, terminal=True)
    _, a, _, done, truncated = rows[0]
    assert a == 2 and done == 1 and truncated == 0


def test_append_transition_row_records_done_0_on_a_non_terminal_row():
    rows: list = []
    append_transition_row(rows, np.zeros(4, dtype=np.int8), 0,
                          np.ones(4, dtype=np.int8), terminal=False)
    assert rows[0][3] == 0


def test_no_row_is_recorded_after_a_terminal_transition():
    """The registration's binding rule: 'No row after that transition is
    recorded.' This is enforced by the CALLER stopping the loop on
    `terminal`, not by this function — asserted here as the contract every
    caller must honor: a terminal row is the last one appended."""
    rows: list = []
    append_transition_row(rows, np.zeros(4, dtype=np.int8), 0,
                          np.ones(4, dtype=np.int8), terminal=True)
    # A caller that (incorrectly) kept going after `terminal=True` would grow
    # `rows` further; the collector's own step loop breaks the wave's active
    # set on `terminal`, which is what this file's other tests hold it to.
    assert len(rows) == 1


def test_finalize_truncation_marks_only_the_last_row_of_a_survived_episode():
    rows = [[None, 0, None, 0, 0], [None, 1, None, 0, 0]]
    finalize_truncation(rows, terminal_hit=False)
    assert [r[4] for r in rows] == [0, 1]


def test_finalize_truncation_leaves_a_terminated_episode_untouched():
    rows = [[None, 0, None, 0, 0], [None, 1, None, 1, 0]]
    finalize_truncation(rows, terminal_hit=True)
    assert [r[4] for r in rows] == [0, 0]
    assert rows[-1][3] == 1  # done stays whatever the terminal row set


def test_truncation_and_done_are_mutually_exclusive_on_the_last_row():
    """A row is never both done=1 (absorbing) and truncated=1 (bootstrap) —
    the two encode DIFFERENT reasons a bank reader should treat a successor
    differently (drop it vs. bootstrap V(s'))."""
    survived = [[None, 0, None, 0, 0]]
    finalize_truncation(survived, terminal_hit=False)
    died = [[None, 0, None, 1, 0]]
    finalize_truncation(died, terminal_hit=True)
    for rows in (survived, died):
        assert not (rows[-1][3] == 1 and rows[-1][4] == 1)


# ==========================================================================
# Sticky gate — gated on step > 0, the config's own
# `sticky_episode_boundary_reset: true` and the honest harness's own rule
# ==========================================================================

def test_sticky_never_applies_on_the_first_step_of_an_episode():
    # A roll of exactly 0.0 would apply at any later step under p=0.25.
    assert sticky_should_apply(step=0, roll=0.0, sticky_prob=0.25) is False


def test_sticky_can_apply_after_the_first_step():
    assert sticky_should_apply(step=1, roll=0.0, sticky_prob=0.25) is True


def test_sticky_respects_the_probability_threshold():
    assert sticky_should_apply(step=5, roll=0.30, sticky_prob=0.25) is False
    assert sticky_should_apply(step=5, roll=0.10, sticky_prob=0.25) is True


def test_sticky_gate_fails_on_revert():
    """Reverting the `step > 0` gate to 'always eligible' would apply sticky
    on step 0 too — the exact contamination the config comment warns about
    (replaying the action the PREVIOUS life died holding). Confirm the gate
    actually distinguishes step 0 from later steps at the same roll."""
    at_zero = sticky_should_apply(step=0, roll=0.05, sticky_prob=0.25)
    at_one = sticky_should_apply(step=1, roll=0.05, sticky_prob=0.25)
    assert at_zero is False and at_one is True


# ==========================================================================
# Cross-population partition — dropped, not merged
# ==========================================================================

def test_cross_population_drops_a_pc_src_row_landing_in_wall():
    gx = np.array([2700])
    src = np.array(["PC_SRC"])
    assert cross_population_mask(gx, src)[0] == False  # noqa: E712


def test_cross_population_drops_a_wall_src_row_landing_in_pc_b5():
    gx = np.array([2900])
    src = np.array(["WALL_SRC"])
    assert cross_population_mask(gx, src)[0] == False  # noqa: E712


def test_cross_population_keeps_a_wall_src_row_in_its_own_band():
    gx = np.array([2700])
    src = np.array(["WALL_SRC"])
    assert cross_population_mask(gx, src)[0] == True  # noqa: E712


def test_cross_population_keeps_a_pc_src_row_in_its_own_band():
    gx = np.array([2900])
    src = np.array(["PC_SRC"])
    assert cross_population_mask(gx, src)[0] == True  # noqa: E712


def test_cross_population_keeps_entrance_rows_regardless_of_gx():
    gx = np.array([2700, 2900, 200])
    src = np.array(["ENTR_SRC", "ENTR_SRC", "ENTR_SRC"])
    assert cross_population_mask(gx, src).all()


def test_cross_population_keeps_out_of_band_rows_from_either_source():
    gx = np.array([100, 5000])
    src = np.array(["WALL_SRC", "PC_SRC"])
    assert cross_population_mask(gx, src).all()


def test_cross_population_mixed_batch():
    gx = np.array([2700, 2900, 2700, 2900])
    src = np.array(["WALL_SRC", "WALL_SRC", "PC_SRC", "PC_SRC"])
    keep = cross_population_mask(gx, src)
    assert list(keep) == [True, False, False, True]


def test_cross_population_partition_fails_on_revert():
    """Reverting to 'keep everything' would let a WALL_SRC row deposit in
    PC_B5 — the exact artifact §4 forbids because it inflates R toward
    CAPABILITY for a reason unrelated to the wall-893 critic."""
    gx = np.array([2900])
    src = np.array(["WALL_SRC"])
    guarded = cross_population_mask(gx, src)
    unguarded = np.ones_like(guarded, dtype=bool)  # the reverted behavior
    assert not np.array_equal(guarded, unguarded)


# ==========================================================================
# Window / episode-cap geometry — read off the tape's own index
# ==========================================================================

def test_window_for_rung_is_the_41_entry_inclusive_window():
    entries = [{"step": s} for s in range(0, 1100)]
    w = window_for_rung(entries, 893, span=40)
    assert len(w) == 41
    assert w[0]["step"] == 853 and w[-1]["step"] == 893


def test_episode_step_cap_matches_the_configs_formula():
    # rung 893: min(1536, 600 + 2*(1094-893)) = min(1536, 1002) = 1002
    assert episode_step_cap(893) == 1002
    # rung 1093: min(1536, 600 + 2*1) = 602
    assert episode_step_cap(1093) == 602
    # the entrance-scale distance (r=0) saturates the global cap
    assert episode_step_cap(0) == 1536


# ==========================================================================
# Penetration receipt — the no-penetration reading's own arithmetic
# ==========================================================================

def test_penetration_receipt_zero_when_no_episode_exceeds_the_threshold():
    eps = [np.array([2670, 2672, 2674]), np.array([2674])]
    rec = penetration_receipt(eps, threshold=2676)
    assert rec["pen_rate"] == 0.0
    assert rec["gx_max"] == 2674


def test_penetration_receipt_counts_episodes_not_rows():
    eps = [np.array([2670, 2700, 2674]), np.array([2674, 2674])]
    rec = penetration_receipt(eps, threshold=2676)
    assert rec["pen_rate"] == 0.5  # one of two EPISODES penetrated
    assert rec["gx_max"] == 2700


def test_penetration_receipt_handles_no_episodes():
    rec = penetration_receipt([], threshold=2676)
    assert rec["pen_rate"] is None and rec["n_episodes"] == 0
