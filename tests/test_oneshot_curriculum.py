"""Tests for the SMB one-shot Lane-4 curriculum glue
(`src/training/oneshot_curriculum.py`).

These cover the four decision behaviours the campaign spec names for the
contended `_run_vanilla_ppo` edit — factored out as pure functions so they
are exercised here without booting an emulator:

  * ladder advances on synthetic success records (`region_past_fraction`
    + `advance_ready`, measured on `region_of` orders);
  * the three-way warm-start partition sums correctly (`warm_start_partition`);
  * the forgetting alarm fires on a forced cold regression (`update_forgetting`);
  * the reversible consolidation schedule (`lerp_coef`) and its 1-4 trigger
    (`reached_1_4` / `furthest_rank`).
"""

from __future__ import annotations

import numpy as np

from src.training.oneshot_curriculum import (
    Partition,
    advance_ready,
    cleared_level_entries,
    forgetting_action,
    furthest_rank,
    lerp_coef,
    level_entry_rungs,
    reached_1_4,
    region_past_fraction,
    update_forgetting,
    warm_start_partition,
)
from src.training.smb_substage_ladder import LADDER_SIZE, OFF_LADDER, build_ladder, region_of


def _ram(*, world: int = 0, area: int = 0, x: int = 0) -> bytes:
    buf = bytearray(2048)
    buf[0x075F] = world & 0xFF
    buf[0x0760] = area & 0xFF
    buf[0x006D] = (x >> 8) & 0xFF
    buf[0x0086] = x & 0xFF
    return bytes(buf)


# ---- ladder advances on synthetic success records -------------------------


def test_advance_gate_fires_when_half_the_pool_clears_the_frontier():
    ladder = build_ladder()
    frontier = 0  # rung 0 = 1-1 entry
    # 8 envs assigned to the frontier rung; half of them push into rung 1+.
    env_regions = np.zeros(8, dtype=int)  # all at the frontier
    # max region per env this iter: 4 envs reach rung 1 (past frontier 0),
    # 4 stay on rung 0.
    rams_past = [region_of(_ram(area=0, x=1500), ladder) for _ in range(4)]  # -> 1
    rams_stay = [region_of(_ram(area=0, x=200), ladder) for _ in range(4)]   # -> 0
    max_regions = np.array(rams_past + rams_stay, dtype=int)
    assert list(max_regions) == [1, 1, 1, 1, 0, 0, 0, 0]

    frac = region_past_fraction(max_regions, env_regions, frontier)
    assert frac == 0.5

    # 5-iter rolling window at >= 50% -> advance is ready.
    history = [frac] * 5
    assert advance_ready(history, window=5, pct=0.50) is True
    # A single strong iter is not enough (window not full).
    assert advance_ready([frac], window=5, pct=0.50) is False
    # Below threshold never advances.
    assert advance_ready([0.4] * 5, window=5, pct=0.50) is False


def test_region_past_fraction_ignores_non_frontier_envs_and_warps():
    ladder = build_ladder()
    frontier = 4  # deep 1-2
    # env 0,1 at the frontier; env 2 at an earlier rung (not counted).
    env_regions = np.array([4, 4, 1], dtype=int)
    # env0 pushed past (rung 5), env1 warped (OFF_LADDER), env2 irrelevant.
    max_regions = np.array([5, OFF_LADDER, 9], dtype=int)
    # Denominator = 2 frontier envs; numerator = 1 (env0). env1's warp can
    # never count (OFF_LADDER < frontier), env2 is off-frontier.
    assert region_past_fraction(max_regions, env_regions, frontier) == 0.5


def test_warp_region_never_satisfies_the_gate():
    ladder = build_ladder()
    # A warp lands off-chain -> OFF_LADDER, strictly below any frontier.
    assert region_of(_ram(world=1, area=0, x=50), ladder) == OFF_LADDER
    env_regions = np.array([3, 3], dtype=int)
    max_regions = np.array([OFF_LADDER, OFF_LADDER], dtype=int)
    assert region_past_fraction(max_regions, env_regions, 3) == 0.0


# ---- retention partition sums correctly -----------------------------------


def test_partition_sums_to_num_envs_and_splits_by_fraction():
    ladder = build_ladder()
    num_envs = 24
    # Frontier deep in 1-3 (rung 8): levels 1-1 and 1-2 are fully cleared.
    part = warm_start_partition(
        num_envs, 8, ladder, frontier_frac=0.50, retention_frac=0.25
    )
    assert isinstance(part, Partition)
    assert len(part.assignment) == num_envs
    assert part.n_frontier + part.n_retention + part.n_spread == num_envs
    assert part.n_frontier == 12   # round(24 * 0.5)
    assert part.n_retention == 6   # round(24 * 0.25), cleared levels exist
    assert part.n_spread == 6
    # Every env lands on a valid rung below-or-at the frontier.
    assert all(0 <= int(r) <= 8 for r in part.assignment)


def test_partition_frontier_holds_F_and_F_minus_1():
    ladder = build_ladder()
    part = warm_start_partition(24, 8, ladder, frontier_frac=0.50, retention_frac=0.25)
    front = part.assignment[: part.n_frontier].tolist()
    assert set(front) == {8, 7}  # frontier holds F and F-1


def test_partition_retention_pins_cleared_level_entries():
    ladder = build_ladder()
    # Frontier at rung 8 (1-3): cleared levels are 1-1 (entry 0) and 1-2
    # (entry 3). Retention envs must sit on exactly those entry rungs.
    part = warm_start_partition(24, 8, ladder, frontier_frac=0.50, retention_frac=0.25)
    ret = part.assignment[part.n_frontier : part.n_frontier + part.n_retention]
    assert set(ret.tolist()) <= {0, 3}
    assert cleared_level_entries(ladder, 8) == [0, 3]


def test_partition_no_cleared_levels_yields_zero_retention():
    ladder = build_ladder()
    # Frontier still inside 1-1 (rung 1): nothing cleared yet.
    part = warm_start_partition(24, 1, ladder, frontier_frac=0.50, retention_frac=0.25)
    assert part.n_retention == 0
    assert part.n_frontier + part.n_spread == 24


def test_partition_frontier_zero_is_all_cold_1_1():
    ladder = build_ladder()
    part = warm_start_partition(24, 0, ladder)
    assert set(part.assignment.tolist()) == {0}
    assert part.n_retention == 0


def test_partition_retention_bump_raises_retention_share():
    ladder = build_ladder()
    base = warm_start_partition(24, 8, ladder, frontier_frac=0.50, retention_frac=0.25)
    bumped = warm_start_partition(24, 8, ladder, frontier_frac=0.50, retention_frac=0.40)
    assert bumped.n_retention > base.n_retention
    assert bumped.n_frontier + bumped.n_retention + bumped.n_spread == 24


def test_level_entry_rungs_maps_first_rung_of_each_level():
    ladder = build_ladder()
    entries = level_entry_rungs(ladder)
    assert entries == {"1-1": 0, "1-2": 3, "1-3": 7, "1-4": 10}


# ---- forgetting alarm fires on a forced cold regression -------------------


def test_forgetting_alarm_after_two_consecutive_regressions():
    # High-water lifts to 1-4 (rank 104), then two probes regress to 1-2.
    hw, strikes = 0, 0
    hw, strikes, alarm = update_forgetting(hw, strikes, "1-4", probes=2)
    assert hw == 104 and strikes == 0 and alarm is False
    hw, strikes, alarm = update_forgetting(hw, strikes, "1-2", probes=2)
    assert strikes == 1 and alarm is False  # one regression, no alarm yet
    hw, strikes, alarm = update_forgetting(hw, strikes, "1-2", probes=2)
    assert strikes == 2 and alarm is True   # two in a row -> alarm


def test_forgetting_recovery_resets_strikes_and_lifts_highwater():
    hw, strikes = 104, 1  # already one regression on record
    hw, strikes, alarm = update_forgetting(hw, strikes, "1-4", probes=2)
    assert strikes == 0 and alarm is False and hw == 104
    # A deeper furthest lifts the high-water further.
    hw, strikes, alarm = update_forgetting(hw, strikes, [2, 1], probes=2)
    assert hw == furthest_rank([2, 1]) and hw > 104


def test_forgetting_ignores_probes_with_no_rank():
    hw, strikes, alarm = update_forgetting(50, 1, None, probes=2)
    assert (hw, strikes, alarm) == (50, 1, False)


def test_forgetting_action_bumps_then_rolls_back_on_persistence():
    # No alarm -> no action.
    assert forgetting_action(False, 100, 0) == "none"
    # First alarm while Retention is NOT bumped (global_it >= bump_until=0):
    # escalate to a Retention bump.
    assert forgetting_action(True, 100, 0) == "bump"
    # A subsequent alarm WHILE the bump window is still open
    # (global_it < bump_until) -> roll back to best_cold.pt.
    assert forgetting_action(True, 150, 200) == "rollback"
    # Once the bump window has elapsed, a fresh alarm bumps again first.
    assert forgetting_action(True, 200, 200) == "bump"


def test_forced_regression_drives_a_rollback_end_to_end():
    """The full escalation the trainer runs: two probes regress -> alarm ->
    Retention bump; the regression persists inside the bump window -> the
    next alarm resolves to a rollback."""
    hw, strikes = 0, 0
    bump_until, bump_iters = 0, 100
    actions = []
    # Probe 0: reaches 1-4 (high-water). Every later probe forces a regression
    # to 1-2, so the alarm re-fires every 2 probes.
    for probe_i, furthest in enumerate(["1-4", "1-2", "1-2", "1-2", "1-2"]):
        global_it = probe_i * 25
        hw, strikes, alarm = update_forgetting(hw, strikes, furthest, probes=2)
        action = forgetting_action(alarm, global_it, bump_until)
        if action == "bump":
            bump_until = global_it + bump_iters
            strikes = 0
        elif action == "rollback":
            strikes = 0
        actions.append(action)
    # probe2: 2 strikes -> alarm -> bump (bump window opens to iter 150).
    # probe4: 2 more strikes -> alarm, still inside the bump window -> rollback.
    assert actions == ["none", "none", "bump", "none", "rollback"]


def test_furthest_rank_parses_labels_pairs_and_none():
    assert furthest_rank("1-4") == 104
    assert furthest_rank([1, 4]) == 104
    assert furthest_rank("2-1") == 201
    assert furthest_rank(None) is None
    assert furthest_rank("garbage") is None
    # Monotone in progression.
    assert furthest_rank("1-1") < furthest_rank("1-4") < furthest_rank("2-1")


# ---- reversible consolidation schedule + trigger --------------------------


def test_reached_1_4_trigger():
    assert reached_1_4("1-4") is True
    assert reached_1_4([1, 4]) is True
    assert reached_1_4("2-1") is True   # beyond 1-4 also qualifies
    assert reached_1_4("1-3") is False
    assert reached_1_4(None) is False


def test_lerp_coef_linear_decay_and_clamp():
    # entropy 0.01 -> 0.002 over 80 iters.
    assert lerp_coef(0.01, 0.002, 0, 80) == 0.01
    assert lerp_coef(0.01, 0.002, 80, 80) == 0.002
    assert lerp_coef(0.01, 0.002, 160, 80) == 0.002   # clamped past the end
    assert abs(lerp_coef(0.01, 0.002, 40, 80) - 0.006) < 1e-9  # midpoint
    # rnd 0.1 -> 0.0.
    assert lerp_coef(0.1, 0.0, 0, 80) == 0.1
    assert lerp_coef(0.1, 0.0, 80, 80) == 0.0
    # Degenerate schedule pins the target.
    assert lerp_coef(0.01, 0.002, 5, 0) == 0.002
