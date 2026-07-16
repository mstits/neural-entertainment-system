"""Pure sub-stage-curriculum glue for the SMB one-shot campaign (Lane 4).

The contended `_run_vanilla_ppo` edit is deliberately thin: all of the
*decisions* it makes each iteration live here as small pure functions so
they are unit-testable without booting an emulator. The trainer wires
them into the existing curriculum machinery (warm-start / capture /
advance / anti-collapse rollback); every game- and RAM-specific fact
lives in `smb_substage_ladder` / `smb_sequential`, and this module knows
only about rung orders, env counts, and cold-probe scalars.

Six behaviours are factored out:

  * `warm_start_partition` — the three-way Frontier / Retention / Spread
    split of the env pool, recomputed each iter from the frontier rung.
  * `advance_ready` / `region_past_fraction` — the reused 50%/5-iter
    rolling advance gate, measured on `region_of` sub-stage orders.
  * `update_forgetting` — the cold-probe regression alarm (N consecutive
    high-water regressions on an already-cleared early level).
  * `lerp_coef` — the linear entropy/RND consolidation schedule.
  * `furthest_rank` / `reached_1_4` — parse a cold-probe "furthest_seq"
    (a "1-4" label, a [world, level] pair, or None) into a comparable
    rank so forgetting and the consolidation trigger can reason about it.
  * `stall_ready` / `burst_quota` / `burst_tick` / `harvest_burst_seed` —
    the DEFERRED, bounded Go-Explore unstick burst (Lane 5): when a rung
    stalls, arm a capped, self-retracting archive burst that harvests at
    most one deeper seed. Pure decisions only; the trainer owns the
    `GoExploreArchive` and the env diversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

# Rank of the "1-4 reached" milestone, the consolidation gate (§Q4).
_LEVEL_1_4_RANK = 1 * 100 + 4


@dataclass(frozen=True)
class Partition:
    """A single-iter warm-start assignment of the env pool to ladder rungs.

    `assignment[i]` is the rung order env `i` warm-starts at this iter.
    `n_frontier` / `n_retention` / `n_spread` are the three-way split
    sizes and always sum to `len(assignment)`.
    """

    assignment: np.ndarray
    n_frontier: int
    n_retention: int
    n_spread: int


def level_entry_rungs(ladder: Sequence) -> dict:
    """Map each displayed level label to its lowest (entry) rung order."""
    entry: dict = {}
    for r in ladder:
        if r.level not in entry or r.order < entry[r.level]:
            entry[r.level] = r.order
    return entry


def cleared_level_entries(ladder: Sequence, frontier: int) -> list:
    """Entry-rung orders of every level the frontier has fully passed.

    A level is "cleared" once the frontier order is strictly greater than
    the level's highest rung order — i.e. training has moved entirely
    beyond it. Returns the sorted entry rungs (the anti-forgetting floor
    sits on exactly these).
    """
    spans: dict = {}
    for r in ladder:
        lo, hi = spans.get(r.level, (r.order, r.order))
        spans[r.level] = (min(lo, r.order), max(hi, r.order))
    return sorted(lo for (lo, hi) in spans.values() if hi < int(frontier))


def warm_start_partition(
    num_envs: int,
    frontier: int,
    ladder: Sequence,
    *,
    frontier_frac: float = 0.50,
    retention_frac: float = 0.25,
) -> Partition:
    """Three-way Frontier / Retention / Spread split of the env pool (§Q2).

    * **Frontier** (`frontier_frac`): rungs `F` and `F-1` (alternating;
      `F-1` only when `F > 0`) — where new progress is earned.
    * **Retention** (`retention_frac`): the entry rung of every
      already-cleared level, round-robined — the fixed anti-forgetting
      floor. Zero when no level has been cleared yet.
    * **Spread** (the remainder): round-robin over all rungs `[0, F)` —
      uniform coverage below the frontier.

    Sizes are integer-rounded with the frontier taking priority and the
    spread absorbing the remainder, so the three always sum to `num_envs`.
    At `F == 0` every env lands on rung 0 (a cold 1-1 boot). The rung
    assignment is deterministic (round-robin, no RNG) so a resumed run and
    a fresh run place the pool identically.
    """
    n = int(num_envs)
    F = int(frontier)
    assign = np.full(n, F, dtype=int)
    if n <= 0:
        return Partition(assign, 0, 0, 0)

    n_front = min(n, int(round(n * float(frontier_frac))))
    ret_rungs = cleared_level_entries(ladder, F)
    n_ret = int(round(n * float(retention_frac))) if ret_rungs else 0
    n_ret = max(0, min(n_ret, n - n_front))
    n_spread = n - n_front - n_ret

    front_rungs = [F] if F <= 0 else [F, F - 1]
    spread_rungs = list(range(F)) if F > 0 else [F]

    idx = 0
    for j in range(n_front):
        assign[idx] = front_rungs[j % len(front_rungs)]
        idx += 1
    for j in range(n_ret):
        assign[idx] = ret_rungs[j % len(ret_rungs)]
        idx += 1
    for j in range(n_spread):
        assign[idx] = spread_rungs[j % len(spread_rungs)]
        idx += 1
    return Partition(assign, n_front, n_ret, n_spread)


def region_past_fraction(
    max_regions: np.ndarray, env_regions: np.ndarray, frontier: int
) -> float:
    """Fraction of current-frontier envs that reached past the frontier.

    Denominator: envs assigned to the frontier rung this iter
    (`env_regions == frontier`). Numerator: those whose max region order
    this iter strictly exceeds the frontier (`region_of(ram) >= F+1`). A
    warp yields region `OFF_LADDER` (-1) so it can never be counted.
    """
    F = int(frontier)
    at_current = env_regions == F
    n_at = int(at_current.sum())
    if n_at <= 0:
        return 0.0
    n_past = int(((max_regions > F) & at_current).sum())
    return n_past / n_at


def advance_ready(pastfrac_history: Sequence, window: int, pct: float) -> bool:
    """The proven rolling-mean advance gate, unchanged from the scalar path.

    Fires once the window is full AND the mean past-fraction over the last
    `window` iters reaches `pct` — a rolling mean (not "N consecutive
    iters above") because the per-iter fraction is intrinsically noisy.
    """
    if len(pastfrac_history) < int(window):
        return False
    return (sum(pastfrac_history) / len(pastfrac_history)) >= float(pct)


def furthest_rank(furthest: Union[str, Sequence, None]) -> Optional[int]:
    """Rank a cold-probe furthest-level as `world*100 + level`, or None.

    Accepts eval_game's string label ("1-4"), a `[world, level]` pair, or
    None (predicate did not produce a value). Monotone in progression so a
    later state always outranks an earlier one.
    """
    if furthest is None:
        return None
    try:
        if isinstance(furthest, str):
            world_s, level_s = furthest.split("-", 1)
            world, level = int(world_s), int(level_s)
        else:
            world, level = int(furthest[0]), int(furthest[1])
    except (ValueError, IndexError, TypeError):
        return None
    return world * 100 + level


def reached_1_4(furthest: Union[str, Sequence, None]) -> bool:
    """True iff the cold probe's furthest sequential state reached 1-4."""
    rank = furthest_rank(furthest)
    return rank is not None and rank >= _LEVEL_1_4_RANK


def update_forgetting(
    highwater: int, strikes: int, furthest: Union[str, Sequence, None], probes: int
) -> tuple:
    """Advance the cold-probe forgetting alarm by one probe (§Q2).

    Maintains a monotone high-water rank of `cold_furthest_seq`. A probe
    whose rank falls below the high-water mark is a regression and adds a
    strike; any probe that holds or improves resets the strike counter and
    lifts the high-water. Probes with no rank (predicate absent) are
    inert. Returns `(highwater, strikes, alarm)` where `alarm` is True once
    `strikes` reaches `probes` consecutive regressions.

    Pure: the trainer owns the escalation (retention bump → rollback) and
    resets `strikes` after acting on an alarm.
    """
    rank = furthest_rank(furthest)
    if rank is None:
        return highwater, strikes, False
    if rank < highwater:
        strikes += 1
    else:
        strikes = 0
        highwater = max(highwater, rank)
    return highwater, strikes, strikes >= int(probes)


def forgetting_action(
    alarm: bool, global_it: int, retention_bump_until: int
) -> str:
    """Escalation decision for a forgetting alarm (§Q2 step b/c).

    Returns one of:
      * ``"none"``     — no alarm this probe;
      * ``"bump"``     — first alarm (Retention not currently bumped): raise
        the Retention floor for a window;
      * ``"rollback"`` — the regression persists while Retention is already
        bumped: roll back to ``best_cold.pt`` and reset the optimizer.

    Pure: the trainer owns the side effects; this is the branch it takes.
    """
    if not alarm:
        return "none"
    return "bump" if int(global_it) >= int(retention_bump_until) else "rollback"


# ---------------------------------------------------------------------------
# Level-scoped consolidation gate (SMB one-shot "weld one level" mode).
#
# Whole-distribution consolidation collapsed the cold chain twice on the tile
# scout (CONSOLIDATE ABORT #1/#2). The level-scoped mode instead welds exactly
# ONE level at a time under a hard cold no-regression gate: it trains 100% of
# the pool inside the TARGET level's rungs while every probe cold-evals the
# target AND every already-welded PROTECT level from their entry states. These
# pure decisions are the gate; the trainer owns the eval subprocess, the
# snapshot save, and the rollback side effects.
# ---------------------------------------------------------------------------


def level_rungs(ladder: Sequence, level: str) -> list:
    """Rung orders belonging to `level` (e.g. ``"1-2"``), in ascending order.

    A level can span several rungs (its x-buckets) and — for 1-2 — two area
    bytes that share a displayed label. The consolidation warm-start spreads
    the pool across exactly these rungs; an empty list means the level is not
    represented in this ladder (the caller falls back to a cold boot).
    """
    return sorted(r.order for r in ladder if r.level == level)


def consolidate_assignment(num_envs: int, rung_orders: Sequence) -> np.ndarray:
    """Round-robin ALL envs across the target level's rungs (100% target).

    No Frontier/Retention/Spread split and no advance — this mode trains the
    single target level only, deterministically (round-robin, no RNG) so a
    resumed weld places the pool identically. With no rungs (level absent from
    the ladder) every env lands on rung 0, a cold boot into the game's start.
    """
    n = int(num_envs)
    if n <= 0:
        return np.zeros(0, dtype=int)
    if not rung_orders:
        return np.zeros(n, dtype=int)
    rungs = [int(r) for r in rung_orders]
    return np.array([rungs[i % len(rungs)] for i in range(n)], dtype=int)


def protect_regressed(
    baselines: dict, rates: dict, *, tol: float = 1e-9
) -> Optional[str]:
    """First protect level whose cold clear rate fell below its baseline.

    `baselines` is the per-level cold greedy clear rate captured at mode start;
    `rates` is this probe's. Returns the name of the FIRST level whose current
    rate is strictly below its baseline (beyond `tol`), or None if every
    protect level held. A level absent from `rates` (its probe failed) is
    skipped rather than treated as a regression — a failed probe must never
    trigger a false rollback. `tol` absorbs eval quantization (a rate is
    ``k/episodes``) so an identical clear count never reads as a regression.
    """
    for level in baselines:
        cur = rates.get(level)
        if cur is None:
            continue
        if float(cur) < float(baselines[level]) - float(tol):
            return str(level)
    return None


def target_improved(best_rate: float, rate: float, *, tol: float = 1e-9) -> bool:
    """True iff the target cold clear rate strictly improves on the accepted best.

    A `rate` of None (probe failed) or one within `tol` of `best_rate` is not
    an improvement — the accepted snapshot is only replaced on real progress.
    """
    if rate is None:
        return False
    return float(rate) > float(best_rate) + float(tol)


def update_sustain(sustain: int, rate: float, bar: float) -> int:
    """Advance the "target >= bar for N consecutive probes" counter.

    Increments when `rate` meets `bar`, resets to 0 otherwise (or on a failed
    probe, `rate is None`). The trainer terminates once the returned count
    reaches the configured `accept_probes`.
    """
    if rate is None or float(rate) < float(bar):
        return 0
    return int(sustain) + 1


def gate_step(
    *,
    regressed: Optional[str],
    target_rate: Optional[float],
    best_rate: float,
    sustain: int,
    bar: float,
    need: int,
    tol: float = 1e-9,
) -> dict:
    """Sequence ONE probe of the level-scoped consolidation gate (pure).

    `regressed` is the protect level that fell below baseline this probe (from
    :func:`protect_regressed`), or None. Returns a decision dict:

      * ``action``  — ``"rollback"`` (a protect level regressed: restore the
        last accepted snapshot + freeze entropy; takes strict priority and
        never accepts or terminates this probe), ``"accept"`` (protect held and
        the target strictly improved: snapshot the new best), or ``"hold"``
        (protect held, no target improvement: keep training).
      * ``sustain`` — the updated "target >= bar" consecutive-probe count, reset
        to 0 on a rollback and advanced only while protect is healthy.
      * ``done``    — True once `sustain` reaches `need` with protect healthy:
        the target held its bar long enough to weld; the caller writes the
        final snapshot + DONE marker and exits.

    This is the whole gate policy in one place so its three named behaviours —
    rollback on protect regression, accept on target improvement, terminate on
    the sustained bar — are unit-testable with stubbed eval rates.
    """
    if regressed is not None:
        return {"action": "rollback", "sustain": 0, "done": False}
    action = "accept" if target_improved(best_rate, target_rate, tol=tol) else "hold"
    new_sustain = update_sustain(sustain, target_rate, bar)
    return {"action": action, "sustain": new_sustain, "done": new_sustain >= int(need)}


def lerp_coef(from_v: float, to_v: float, step: int, iters: int) -> float:
    """Linear coefficient schedule for the reversible consolidation decay.

    `step` counts iters since consolidation armed; at `step >= iters` the
    value is pinned at `to_v`. Used for both `entropy_coef` (0.01→0.002)
    and `rnd_intrinsic_coef` (0.1→0.0). `iters <= 0` returns `to_v`.
    """
    if int(iters) <= 0:
        return float(to_v)
    frac = min(1.0, max(0.0, float(step) / float(iters)))
    return float(from_v) + (float(to_v) - float(from_v)) * frac


# ---------------------------------------------------------------------------
# Go-Explore unstick burst (Lane 5, DEFERRED — §Q3).
#
# When a rung's advance gate stalls for `patience` iters while the frontier is
# genuinely being played, the trainer arms a BOUNDED archive burst: it diverts
# a small, capped env quota to `GoExploreArchive` return states to spread
# exploration across the stalled rung, harvests at most one deeper seed, then
# self-retracts. These four pure functions are the burst's *decisions*; the
# trainer owns the archive, the env diversion, and the seed injection. Nothing
# here ever flips the curriculum flag — the burst is a subroutine, not a mode.
# ---------------------------------------------------------------------------


def stall_ready(
    iters_since_advance: int,
    patience: int,
    *,
    enabled: bool,
    reaches: bool,
    frontier: int,
    ladder_size: int,
    blocked: bool = False,
) -> bool:
    """Decide whether to arm a Go-Explore unstick burst this iter (§Q3).

    Fires iff ALL hold:
      * `enabled` — the profile set the `go_explore_fallback` knob;
      * not `blocked` — never concurrent with consolidation (or an
        already-running burst; the trainer passes both);
      * `reaches` — the frontier rung is genuinely being reached this iter
        (the block is "find the next state," NOT "can't play this rung"; a
        rung the pool can't even reach is not a Go-Explore candidate);
      * there is a deeper rung to seed (`frontier < ladder_size - 1`);
      * the stall has lasted at least `patience` iters
        (`iters_since_advance >= patience` — fires at N, never at N-1).
    """
    if not enabled or blocked or not reaches:
        return False
    if int(frontier) >= int(ladder_size) - 1:
        return False
    return int(iters_since_advance) >= int(patience)


def burst_quota(num_envs: int, frac: float, cap: int) -> int:
    """Size the capped env quota a burst diverts to archive returns.

    A `frac` of the pool, but never more than `cap` envs (nor the whole
    pool) — the hard cap is what keeps the burst a *diversion* and not a
    permanent Go-Explore takeover of the curriculum. At least 1 env when
    the pool is non-empty (a zero-env burst would be a no-op arm); 0 for an
    empty pool.
    """
    n = int(num_envs)
    if n <= 0:
        return 0
    q = max(1, int(round(n * float(frac))))
    return max(0, min(q, int(cap), n))


def burst_tick(remaining: int) -> tuple:
    """Advance the burst clock one iter; return `(new_remaining, retract)`.

    Consuming the last remaining iter (`remaining <= 1`) sets `retract`
    True — the burst has run its bounded course and must retract to normal
    curriculum operation. A burst armed with `burst_iters == N` therefore
    ticks N times and retracts on exactly the Nth (never earlier, never
    permanently), so it self-terminates even if it never harvested a seed.
    """
    r = int(remaining) - 1
    return max(0, r), r <= 0


def harvest_burst_seed(cells, frontier: int) -> Optional[tuple]:
    """The harvest-one-seed rule: pick at most ONE deeper seed from a burst.

    `cells` is an iterable of `(region_order, state_blob)` pairs the archive
    produced (the trainer reads each cell's region off its stored score).
    Returns the SINGLE deepest `(region_order, state_blob)` whose region is
    strictly past the stalled `frontier`, or `None` when the burst found
    nothing deeper (a state at/below the frontier is no help, and a cell
    with no state blob can't seed). At most one seed ever leaves a burst —
    the burst harvests a single frontier state, not a whole sub-curriculum.
    """
    best: Optional[tuple] = None
    f = int(frontier)
    for region_order, state_blob in cells:
        if state_blob is None:
            continue
        r = int(region_order)
        if r > f and (best is None or r > int(best[0])):
            best = (r, state_blob)
    return best
