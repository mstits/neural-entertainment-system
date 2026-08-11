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
  * `resolve_probe_settings` / `require_selection_seed` /
    `wilson_lower_bound` / `probe_failed` / `decay_best` — the B6
    consolidation-gate arithmetic: an honest (sticky + jitter + seeded)
    probe distribution whose selection seed is kept off the reporting
    seeds, a Wilson lower-bounded accept, and the ratchet break that
    stops one lucky probe pinning an unreachable bar. All default-inert.
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


# ---------------------------------------------------------------------------
# B6 gate arithmetic: honest-probe distribution + Wilson-bounded acceptance.
#
# Receipts this exists to close (checkpoints/mario_1_1_backward_v4/):
#   * `run.log` — of 16 gate probes on target 1-1, fifteen read 0.000 and one
#     read 1.000. None of `sticky_prob`, `start_jitter` or `eval_seed` reached
#     the eval, so all of a probe's episodes were one replay repeated and the
#     accept rested on a single sample.
#   * `consolidate_1-1.json` — closed at {"best_tgt_rate": 1.0,
#     "target_rate": 0.0, "sustain": 0, "iter": 260}. Strict improvement over
#     a best of 1.0 cannot be satisfied, so nothing could be accepted over the
#     run's last 144 iterations.
#
# Everything below is default-inert: with the new profile keys absent the
# resolver returns the pre-B6 settings and the gate arithmetic reduces to the
# point-estimate comparison it always did.
# ---------------------------------------------------------------------------

# Accept-rule names. `point_gt_best` is the historical (pre-B6) rule.
ACCEPT_RULE_POINT = "point_gt_best"
ACCEPT_RULE_WILSON_LB = "wilson_lb_gt_best_point"
ACCEPT_RULES = (ACCEPT_RULE_POINT, ACCEPT_RULE_WILSON_LB)

# Seeds reserved for REPORTING (the published deliverable pair). The gate's
# SELECTION seed must never be one of them: a probe that both selects the
# snapshot and reports its score would draw from the very stream the number is
# later published on. The 1-1 consolidation deliverable is reported at
# eval_seed 0 and 1 (the two seeds the banked v4 control was measured on, so
# the comparison is paired), and the gate selects outside that set.
REPORTING_EVAL_SEEDS: tuple[int, ...] = (0, 1)

# `scripts/eval_game.py`'s own `--eval-seed` default. An UNSET probe seed does
# not mean "no seed" — `cold_probe.probe` simply omits the flag and the eval
# subprocess falls back to this, which is REPORTING_EVAL_SEEDS[0]. Resolving
# the effective seed is what makes the reporting-seed guard real: a profile
# that arms sticky + jitter and leaves `eval_seed` unset silently selects on
# reporting seed 0.
EVAL_GAME_DEFAULT_SEED = 0

# Two-sided z quantiles for the confidences a gate realistically uses. Kept as
# a table so the module stays dependency-free (no scipy) and the arithmetic is
# reproducible from the receipt; anything else falls back to the nearest
# tabulated confidence, which is logged by the caller.
_Z_FOR_CONFIDENCE: dict = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


#: The confidences the table can serve exactly. A caller asking for anything
#: else silently gets the nearest one, so the trainer checks membership and
#: says so in the log rather than letting a "0.975 gate" quietly be a 0.98 one.
TABULATED_CONFIDENCES: frozenset = frozenset(_Z_FOR_CONFIDENCE)


def nearest_tabulated_confidence(confidence: float) -> float:
    """The tabulated confidence :func:`z_for_confidence` will actually use."""
    c = float(confidence)
    if c in _Z_FOR_CONFIDENCE:
        return c
    return min(_Z_FOR_CONFIDENCE, key=lambda k: abs(k - c))


def z_for_confidence(confidence: float) -> float:
    """Two-sided normal quantile for `confidence` (nearest tabulated value)."""
    return _Z_FOR_CONFIDENCE[nearest_tabulated_confidence(confidence)]


def wilson_lower_bound(
    successes: float, n: int, *, confidence: float = 0.95
) -> float:
    """Wilson score-interval LOWER bound for `successes`/`n` (pure).

    The score interval, not the normal approximation: at the counts a gate
    probe actually produces (n = 30, p̂ near 0 or 1) the Wald interval is
    degenerate — 30/30 gives a Wald half-width of exactly 0 — while Wilson
    stays inside (0, 1) and shrinks correctly with n. `successes` is accepted
    as a float because the caller reconstructs it from a rounded rate.

    n <= 0 returns 0.0 (an unmeasured probe bounds nothing).
    """
    n_i = int(n)
    if n_i <= 0:
        return 0.0
    k = min(max(float(successes), 0.0), float(n_i))
    z = z_for_confidence(confidence)
    p = k / n_i
    denom = 1.0 + (z * z) / n_i
    centre = p + (z * z) / (2.0 * n_i)
    margin = z * ((p * (1.0 - p) / n_i + (z * z) / (4.0 * n_i * n_i)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


def probe_failed(target_rate: Optional[float],
                 target_n: Optional[int]) -> bool:
    """True iff this probe produced no usable measurement (pure).

    Three shapes of "no measurement", all of which must leave the incumbent
    and the sustain counter exactly where they were:

      * ``target_rate is None`` — the pure-API spelling of a failed probe;
      * ``target_rate < 0`` — the TRAINER's spelling. `_gate_probe` returns
        None on a failed eval subprocess and the caller converts it to the
        ``-1.0`` sentinel before it reaches this module, so a guard that only
        checks ``is None`` is unreachable from production;
      * ``target_n == 0`` — the eval ran but scored zero episodes, so the rate
        is not an estimate of anything and no bound can be taken over it.

    ``target_n is None`` is NOT a failure: every pre-B6 caller omits it, and
    those callers must keep their exact behaviour.
    """
    if target_rate is None:
        return True
    if float(target_rate) < 0.0:
        return True
    return target_n is not None and int(target_n) <= 0


def decay_best(best_rate: float, observed: Optional[float],
               decay: float) -> float:
    """Re-estimate the incumbent best toward the live measurement (pure).

    `decay` 0.0 (the default everywhere) returns `best_rate` unchanged — the
    pre-B6 permanent high-water mark. `decay` 1.0 replaces the incumbent with
    this probe's estimate outright; values between the two are an exponential
    re-estimation.

    `observed is None` never moves the bar, but note that this alone is NOT
    the failed-probe guarantee: the trainer converts a failed probe to the
    ``-1.0`` sentinel, which would drag the incumbent DOWNWARD (best 0.60 at
    decay 0.25 -> 0.20) if it reached here. :func:`gate_step` screens for that
    with :func:`probe_failed` before calling this, and that is where the
    guarantee is tested.
    """
    d = float(decay)
    if d <= 0.0 or observed is None:
        return float(best_rate)
    d = min(1.0, d)
    return float(best_rate) + d * (float(observed) - float(best_rate))


@dataclass(frozen=True)
class ProbeSettings:
    """Resolved `reinforce.consolidate_level.probe.*` settings (pure).

    `sticky_prob` / `start_jitter` / `eval_seed` default to the pre-B6 values
    (0.0 / 0 / None), which is what makes an absent probe block byte-identical
    to v4: `cold_probe.probe` appends `--sticky-prob` / `--start-jitter` /
    `--eval-seed` only when they are non-default, so the emitted eval command
    line does not change.
    """

    every: int
    episodes: int
    max_steps: int
    sticky_prob: float
    start_jitter: int
    eval_seed: Optional[int]
    #: The seed the eval subprocess will ACTUALLY use — `eval_seed` when the
    #: profile set one, else :data:`EVAL_GAME_DEFAULT_SEED`. Never None, so a
    #: receipt written from it can never carry a null protocol field.
    effective_eval_seed: int
    eval_workers: int
    eval_rng: str
    stochastic: bool
    seed_collision: bool

    def probe_kwargs(self) -> dict:
        """The `cold_probe.probe(...)` keyword arguments for these settings."""
        return {
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "sticky_prob": self.sticky_prob,
            "start_jitter": self.start_jitter,
            "eval_seed": self.eval_seed,
            "eval_workers": self.eval_workers,
            "eval_rng": self.eval_rng,
        }


def resolve_probe_settings(probe_cfg: Optional[dict]) -> ProbeSettings:
    """Parse the consolidation probe block into a resolved settings object.

    The gate probes to SELECT; the published deliverable is measured on
    different seeds (:data:`REPORTING_EVAL_SEEDS`).

    `seed_collision` is keyed on the EFFECTIVE seed, not the configured one.
    An unset `eval_seed` is not "no seed": `cold_probe.probe` omits the flag
    and `eval_game.py` falls back to :data:`EVAL_GAME_DEFAULT_SEED`, which is
    a reporting seed — so a profile that arms sticky + jitter and forgets
    `eval_seed` selects on the reporting stream while looking configured. It
    is reported only for a stochastic probe, because that is the only case in
    which a stream is drawn from at all: a deterministic probe consumes no
    randomness (its own, louder defect is reported separately).

    `eval_rng` is forced to ``per-episode`` whenever more than one eval worker
    is asked for on a stochastic probe: `eval_game.py` refuses that
    combination under the shared stream (episode i's draws depend on how many
    draws episodes 0..i-1 consumed), and a refused probe is a dead gate.
    """
    cfg = dict(probe_cfg or {})
    episodes = max(1, int(cfg.get("episodes", 8)))
    sticky = float(cfg.get("sticky_prob", 0.0) or 0.0)
    jitter = int(cfg.get("start_jitter", 0) or 0)
    raw_seed = cfg.get("eval_seed", None)
    seed = None if raw_seed is None else int(raw_seed)
    effective_seed = EVAL_GAME_DEFAULT_SEED if seed is None else seed
    workers = max(1, int(cfg.get("eval_workers", 1) or 1))
    # A run "consumes randomness" (eval_game.run_consumes_randomness) iff any
    # perturbation is armed; the gate always draws greedily.
    stochastic = bool(sticky > 0.0 or jitter > 0)
    rng = str(cfg.get("eval_rng", "shared-stream"))
    if workers > 1 and stochastic:
        rng = "per-episode"
    return ProbeSettings(
        every=max(1, int(cfg.get("every", 25))),
        episodes=episodes,
        max_steps=int(cfg.get("max_steps", 1500)),
        sticky_prob=sticky,
        start_jitter=jitter,
        eval_seed=seed,
        effective_eval_seed=effective_seed,
        eval_workers=workers,
        eval_rng=rng,
        stochastic=stochastic,
        seed_collision=bool(
            stochastic and effective_seed in REPORTING_EVAL_SEEDS
        ),
    )


def require_selection_seed(settings: ProbeSettings) -> None:
    """Raise if the gate would select on a seed the deliverable is reported on.

    Enforcement, not advice. A log line does not stop the probe, and
    ``--strict-config`` cannot catch this case at all: every key involved is
    registered and individually valid, and the commonest way to hit it is to
    arm sticky + jitter and simply omit ``eval_seed`` — which lands on
    :data:`EVAL_GAME_DEFAULT_SEED`. Called once at consolidation-mode start so
    a misconfigured attended block fails before its first iteration.
    """
    if not settings.seed_collision:
        return
    raise ValueError(
        f"reinforce.consolidate_level.probe.eval_seed resolves to "
        f"{settings.effective_eval_seed} (configured: "
        f"{settings.eval_seed!r}), which is one of the reporting seeds "
        f"{list(REPORTING_EVAL_SEEDS)}. The gate would select the snapshot on "
        f"the same random stream the deliverable is later reported on. Set an "
        f"explicit selection seed outside {list(REPORTING_EVAL_SEEDS)} — note "
        f"that OMITTING the key is not neutral: eval_game defaults "
        f"--eval-seed to {EVAL_GAME_DEFAULT_SEED}."
    )


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
    target_n: Optional[int] = None,
    use_wilson_bound: bool = False,
    wilson_confidence: float = 0.95,
    accept_rule: str = ACCEPT_RULE_POINT,
    best_decay: float = 0.0,
    best_floor: Optional[float] = None,
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
      * ``target_lb`` — the probe's Wilson lower bound, or None when no bound
        was asked for (or the probe produced no measurement to bound).
      * ``gate_metric`` — the scalar actually compared against `bar` (the
        Wilson LB when armed, else the point rate).
      * ``accept_lhs`` — the scalar actually compared against `best_rate`.
      * ``best_rate`` — the value the caller must store as the new incumbent
        (the probe's POINT estimate on an accept, the decayed/re-estimated
        incumbent on a hold, unchanged on a rollback or a failed probe).
      * ``best_floor`` — the value the caller must store as the new floor under
        `best_rate` (see ``best_decay`` below), or None while no floor has been
        established. None and inert unless a bound is armed.

    This is the whole gate policy in one place so its named behaviours —
    rollback on protect regression, accept on target improvement, terminate on
    the sustained bar — are unit-testable with stubbed eval rates.

    THE RATCHET BREAK (v4 receipt: ``consolidate_1-1.json`` closed at
    ``best_tgt_rate 1.0``, set by ONE deterministic replay at iter 116; from
    there the accept condition was unsatisfiable for 144 iterations). Two
    changes, both default-inert, and INTENDED TO BE ARMED TOGETHER — the
    bound stops a lucky probe raising the bar, the decay stops an honest but
    unrepeatable high-water holding it (see the reachability note below):

      * ``accept_rule="wilson_lb_gt_best_point"`` accepts only when the new
        probe's Wilson LOWER bound beats the incumbent's POINT estimate — a
        single lucky episode has a low LB, so it can no longer raise the bar;
        and the incumbent is only ever re-set from that probe's point estimate
        over the configured (>= 30) episode count, never from an n=1 replay.
      * ``best_decay`` in (0, 1] re-estimates the incumbent toward the live
        measurement on every non-accepting MEASURED probe, so a stale
        high-water erodes instead of locking the gate out forever.

    REACHABILITY, stated so nobody has to rediscover it mid-run. The bound
    alone does not make the bar reachable: it only makes it honest. Setting
    ``best`` to an accepted probe's point estimate means the next accept needs
    ``LB(new) > point(best)``, i.e. roughly +0.18 of point estimate at n = 30
    and 95%. Concretely, accepting at 20/30 (0.667) puts the next accept at
    26/30 (0.867), and a 30/30 probe pins ``best`` at 1.0 against a maximum
    attainable LB of 0.886 — which is the v4 lockout exactly. ``best_decay``
    is what dissolves that: an incumbent the current policy cannot reproduce
    erodes toward what the policy actually does, so the bar tracks the
    measurement instead of the luckiest sample ever drawn.

    TWO CLAMPS make the decay safe, and both are no-ops at ``best_decay=0``:

      * the decay may only move the bar DOWN. The bar describes the ACCEPTED
        SNAPSHOT, which a hold did not change, so a hold must never raise it.
        (Reachable under the bound rule: a probe can have a higher point
        estimate than ``best`` and still hold on its lower bound.)
      * the decay may not fall below ``best_floor``, which the caller carries
        forward from ``best_floor`` in this dict (None until the first bounded
        accept establishes one). On an accept the floor becomes the accepting
        probe's own Wilson LOWER bound: the run is 95% confident the snapshot
        it just saved is at least that good, so a candidate that cannot beat
        that number is not credibly better and must not be allowed to
        overwrite it. Without the floor, a late collapse (v4's final iterate
        read 0.00) would drag the bar to ~0 and let a clearly worse network
        replace the deliverable.

    FAILING CLOSED. Under ``accept_rule="wilson_lb_gt_best_point"`` the accept
    is gated on the BOUND: if the bound cannot be computed the gate holds, and
    it never silently falls back to comparing the point estimate (which would
    restore the very ratchet the rule exists to break under a profile that
    believes it is protected). A caller that asks for the rule without
    threading ``target_n`` at all is a wiring bug, not a runtime condition, and
    raises.

    With the defaults (``use_wilson_bound=False``, ``accept_rule`` point,
    ``best_decay=0.0``) every returned field reproduces the pre-B6 gate
    exactly: action/sustain/done are unchanged and ``best_rate`` is the point
    rate on an accept and the untouched incumbent otherwise.
    """
    if regressed is not None:
        return {
            "action": "rollback", "sustain": 0, "done": False,
            "target_lb": None, "gate_metric": target_rate,
            "accept_lhs": target_rate, "best_rate": float(best_rate),
            "best_floor": best_floor,
        }
    rule = str(accept_rule)
    if rule not in ACCEPT_RULES:
        raise ValueError(
            f"accept_rule must be one of {list(ACCEPT_RULES)}, got {rule!r}"
        )
    wilson_rule = rule == ACCEPT_RULE_WILSON_LB
    want_lb = bool(use_wilson_bound) or wilson_rule
    if want_lb and target_n is None:
        # Wiring bug: the bound has no sample size, so a "bounded" gate would
        # be comparing point estimates while its log said otherwise.
        raise ValueError(
            "gate_step: a Wilson-bounded gate needs target_n (the episode "
            f"count the probe actually scored); got accept_rule={rule!r}, "
            f"use_wilson_bound={bool(use_wilson_bound)}, target_n=None"
        )

    if probe_failed(target_rate, target_n):
        # No measurement: hold, reset the sustain counter, and leave the
        # incumbent exactly where it was. In particular the -1.0 sentinel the
        # trainer passes for a failed eval must never be fed to `decay_best`,
        # which would drag `best` below every rate the policy can produce.
        return {
            "action": "hold", "sustain": 0, "done": False,
            "target_lb": None, "gate_metric": target_rate,
            "accept_lhs": target_rate, "best_rate": float(best_rate),
            "best_floor": best_floor,
        }

    lb = None
    if want_lb:
        lb = wilson_lower_bound(
            float(target_rate) * int(target_n), int(target_n),
            confidence=wilson_confidence,
        )
    gate_metric = lb if lb is not None else target_rate
    accept_lhs = lb if wilson_rule else target_rate
    accept = target_improved(best_rate, accept_lhs, tol=tol)
    action = "accept" if accept else "hold"
    new_floor = None if best_floor is None else float(best_floor)
    if accept:
        # Re-estimate the incumbent from THIS probe's point estimate (an
        # n>=30 measurement under B6), not from the bound that authorised it.
        new_best = float(target_rate)
        if lb is not None:
            new_floor = float(lb) if new_floor is None \
                else max(new_floor, float(lb))
    else:
        # Monotone-down, floored. See "TWO CLAMPS" above; both are no-ops at
        # the default best_decay=0.0, where `decayed` IS `best_rate`, and the
        # floor is skipped entirely until a bounded accept establishes one.
        decayed = decay_best(best_rate, target_rate, best_decay)
        new_best = min(float(best_rate), decayed)
        if new_floor is not None:
            new_best = max(new_best, new_floor)
    new_sustain = update_sustain(sustain, gate_metric, bar)
    return {
        "action": action, "sustain": new_sustain,
        "done": new_sustain >= int(need),
        "target_lb": lb, "gate_metric": gate_metric,
        "accept_lhs": accept_lhs, "best_rate": new_best,
        "best_floor": new_floor,
    }


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
