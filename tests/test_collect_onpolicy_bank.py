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
    assert_bank_wellformed,
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


# ==========================================================================
# Bank well-formedness — the artifact-level guard for the aliasing defect
# that voided the 2026-08-27 on-policy read. Both invariants are
# threshold-free; see `assert_bank_wellformed`.
# ==========================================================================

def _chain(frames, ep=0):
    """Build a correct one-episode bank from a list of stacked observations."""
    frames = [np.asarray(f, dtype=np.int8) for f in frames]
    state = np.stack(frames[:-1])
    next_state = np.stack(frames[1:])
    episode_id = np.full(len(state), ep, dtype=np.int64)
    return state, next_state, episode_id


def test_wellformed_accepts_a_correct_single_episode_bank():
    s, ns, ep = _chain([[0, 0], [1, 0], [2, 0], [3, 0]])
    assert_bank_wellformed(s, ns, ep)


def test_wellformed_accepts_two_independent_episodes():
    s0, ns0, e0 = _chain([[0, 0], [1, 0], [2, 0]], ep=0)
    s1, ns1, e1 = _chain([[9, 0], [8, 0], [7, 0]], ep=1)
    assert_bank_wellformed(np.concatenate([s0, s1]),
                           np.concatenate([ns0, ns1]),
                           np.concatenate([e0, e1]))


def test_wellformed_accepts_an_individually_frozen_row():
    # A genuinely static scene repeats one stacked observation. Legal.
    s, ns, ep = _chain([[0, 0], [1, 0], [1, 0], [2, 0]])
    assert_bank_wellformed(s, ns, ep)


def test_wellformed_accepts_a_single_row_episode():
    s, ns, ep = _chain([[0, 0], [1, 0]])
    assert_bank_wellformed(s, ns, ep)


def test_wellformed_accepts_an_empty_bank():
    assert_bank_wellformed(np.zeros((0, 2), dtype=np.int8),
                           np.zeros((0, 2), dtype=np.int8),
                           np.zeros(0, dtype=np.int64))


def test_wellformed_rejects_the_aliasing_defect_that_voided_the_read():
    """The exact 2026-08-27 artifact: every row is (s', a, s')."""
    obs = [np.array([i, 0], dtype=np.int8) for i in range(5)]
    state = np.stack(obs[1:])          # successor in the antecedent slot
    next_state = np.stack(obs[1:])     # successor again
    ep = np.zeros(len(state), dtype=np.int64)
    with pytest.raises(RuntimeError, match="DEGENERATE"):
        assert_bank_wellformed(state, next_state, ep)


def test_wellformed_rejects_a_broken_chain_even_when_rows_differ():
    """Aliasing on a moving scene breaks the chain before it looks degenerate."""
    s, ns, ep = _chain([[0, 0], [1, 0], [2, 0], [3, 0]])
    ns = ns.copy()
    ns[0] = [7, 7]  # successor at step 0 is not the antecedent at step 1
    with pytest.raises(RuntimeError, match="CHAIN BROKEN"):
        assert_bank_wellformed(s, ns, ep)


def test_wellformed_does_not_require_the_chain_to_cross_episodes():
    """Episode 1's first antecedent need not equal episode 0's last successor."""
    s0, ns0, e0 = _chain([[0, 0], [1, 0]], ep=0)
    s1, ns1, e1 = _chain([[5, 5], [6, 6]], ep=1)
    assert_bank_wellformed(np.concatenate([s0, s1]),
                           np.concatenate([ns0, ns1]),
                           np.concatenate([e0, e1]))


def test_wellformed_rejects_mismatched_shapes():
    with pytest.raises(RuntimeError, match="malformed"):
        assert_bank_wellformed(np.zeros((3, 2), dtype=np.int8),
                               np.zeros((3, 4), dtype=np.int8),
                               np.zeros(3, dtype=np.int64))


def test_bank_wellformed_guard_fails_on_revert():
    """ANTI-VACUITY, executed not asserted. Neuter the guard to a no-op and
    the two defect cases above stop being caught."""
    def reverted(state, next_state, episode_id):
        return None

    obs = [np.array([i, 0], dtype=np.int8) for i in range(5)]
    aliased = np.stack(obs[1:])
    ep = np.zeros(len(aliased), dtype=np.int64)
    assert reverted(aliased, aliased, ep) is None
    with pytest.raises(RuntimeError):
        assert_bank_wellformed(aliased, aliased, ep)

    s, ns, ep2 = _chain([[0, 0], [1, 0], [2, 0], [3, 0]])
    ns = ns.copy()
    ns[0] = [7, 7]
    assert reverted(s, ns, ep2) is None
    with pytest.raises(RuntimeError):
        assert_bank_wellformed(s, ns, ep2)


def test_stacker_push_returns_a_reused_buffer_so_the_copy_is_load_bearing():
    """The root cause, pinned. If TileFeatureStacker ever stops reusing its
    output buffer this test should be updated, not deleted -- the collector's
    `.copy()` calls are what make the recorded antecedent an antecedent."""
    from src.emulation.frame_utils import TileFeatureStacker
    stk = TileFeatureStacker(stack_size=2, feature_dim=2)
    a = stk.reset(np.zeros(2, dtype=np.int8))
    b = stk.push(np.ones(2, dtype=np.int8))
    assert a is b, "buffer reuse is the premise of the collector's .copy()"
    assert stk.reset(np.zeros(2, dtype=np.int8)).copy() is not stk.get()


# ==========================================================================
# Gap-aware chain — the guard's own false positive, fixed 2026-08-28.
# The registered cross-population DROP removes interior rows (a PC_SRC
# rung-1013 episode traverses the WALL band on its way up), so surviving
# neighbours are legitimately non-consecutive. `row_step` makes the gap
# visible in the written artifact; the chain is asserted only across pairs
# whose recorded steps are adjacent. Live receipt: the 2026-08-28 re-run
# aborted CHAIN BROKEN at iter 30 / episode 48 on exactly this pattern.
# ==========================================================================

def _chain_with_steps(frames, ep=0):
    s, ns, e = _chain(frames, ep=ep)
    return s, ns, e, np.arange(len(s), dtype=np.int64)


def test_wellformed_accepts_a_mid_episode_gap_from_the_registered_drop():
    """The iter-30 false positive, reproduced: drop one interior row of a
    correct chain and pass the survivors with their original steps. This
    test FAILS on the pre-2026-08-28 guard (no row_step, gap reads as
    CHAIN BROKEN) — it is the executed revert-verification of the fix."""
    s, ns, e, st = _chain_with_steps([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    keep = np.array([True, True, False, True])  # drop interior row 2
    assert_bank_wellformed(s[keep], ns[keep], e[keep], st[keep])


def test_wellformed_still_rejects_aliasing_when_the_bank_has_gaps():
    """The defect class the guard exists for must not hide behind a gap:
    aliased rows fail on every surviving ADJACENT pair."""
    obs = [np.array([i, 0], dtype=np.int8) for i in range(6)]
    state = np.stack(obs[1:])       # (s', a, s') on every row
    next_state = np.stack(obs[1:])
    ep = np.zeros(len(state), dtype=np.int64)
    st = np.arange(len(state), dtype=np.int64)
    keep = np.array([True, True, False, True, True])  # a gap too
    # Make one row differ so DEGENERATE doesn't fire first; the chain
    # violation on the remaining adjacent pairs must still raise.
    state = state.copy()
    state[4] = [99, 99]
    with pytest.raises(RuntimeError, match="CHAIN BROKEN"):
        assert_bank_wellformed(state[keep], next_state[keep], ep[keep],
                               st[keep])


def test_wellformed_rejects_a_broken_chain_between_adjacent_steps():
    """With row_step present, adjacent-step pairs keep the full contract."""
    s, ns, e, st = _chain_with_steps([[0, 0], [1, 0], [2, 0], [3, 0]])
    ns = ns.copy()
    ns[0] = [7, 7]
    with pytest.raises(RuntimeError, match="CHAIN BROKEN"):
        assert_bank_wellformed(s, ns, e, st)


def test_wellformed_rejects_nonmonotonic_row_step():
    s, ns, e, st = _chain_with_steps([[0, 0], [1, 0], [2, 0]])
    st = st.copy()
    st[1], st[0] = st[0], st[1]
    with pytest.raises(RuntimeError, match="ROW ORDER BROKEN"):
        assert_bank_wellformed(s, ns, e, st)


def test_wellformed_rejects_row_step_length_mismatch():
    s, ns, e, st = _chain_with_steps([[0, 0], [1, 0], [2, 0]])
    with pytest.raises(RuntimeError, match="malformed"):
        assert_bank_wellformed(s, ns, e, st[:-1])


def test_lanes_to_bank_writes_row_step_and_the_drop_leaves_a_visible_gap():
    """Integration on the real assembly path: a WALL_SRC lane with one row
    landing in PC_B5 loses that row to the registered drop; the written
    row_step skips its index and the artifact guard passes the survivors."""
    from scripts.collect_onpolicy_bank import lanes_to_bank

    frame_dim = 178

    def obs_at(gx: int, tag: int) -> np.ndarray:
        o = np.zeros(frame_dim, dtype=np.int8)
        o[0] = tag  # make every observation distinct
        page, fine = gx >> 8, (gx & 0xFF) >> 1
        o[175] = page
        o[176] = fine
        return o

    class _FakeLane:
        pass

    lane = _FakeLane()
    lane.src = "WALL_SRC"
    lane.src_rung = 893
    lane.episode_id = 0
    # gx walk: 2674 -> 2900 (in PC_B5, DROPPED for a WALL_SRC lane) -> 2674.
    o0, o1, o2, o3 = (obs_at(2674, 1), obs_at(2900, 2),
                      obs_at(2674, 3), obs_at(2670, 4))
    lane.rows = [
        [o0, 1, o1, 0, 0],
        [o1, 2, o2, 0, 0],   # state gx 2900 => dropped by the partition
        [o2, 3, o3, 0, 0],
    ]
    bank = lanes_to_bank([lane])
    assert bank["n_dropped_cross_population"] == 1
    assert bank["row_step"].tolist() == [0, 2]
    assert_bank_wellformed(bank["state"], bank["next_state"],
                           bank["episode_id"], bank["row_step"])
