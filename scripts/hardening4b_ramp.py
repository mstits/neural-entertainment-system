"""Hardening-4b budget-ramp decision logic — pure functions, no I/O.

This module is the pre-registered arbiter for the ANCHORED hardening
package (configs/overrides/online_1_2_hardening4b.yaml). It owns exactly
one thing: given the stream of honest-probe medians (the same
median_max_x the campaign controller already logs per probe,
scripts/run_online_campaign.py probe_summary), decide at each probe
whether the adversary's budget-penalty ramp HOLDS at the current rung,
ADVANCES to the next rung, RESETS its consecutive-hold counter, or
KILLS the hardening attempt with a rollback to the pre-hardening
checkpoint.

It is NOT a runner and launches nothing. The operator (or a future
controller patch) feeds it probe medians and applies the corresponding
catalog block by hand-merge + resume, exactly like the attempt-5..8
re-registrations. Keeping the decision rule here — pure, imported by
tests/test_hardening4b_catalog.py, exercised on stub streams — means
the gate semantics are pinned by the suite before any compute is spent,
and the yaml's prose and this code cannot drift apart unnoticed (the
suite cross-checks the constants against the catalog's blocks).

Registered semantics (2026-08-15, hardening phase 4b):

  * BUDGET_RAMP = (0.3, 0.2, 0.1) — adversary budget_penalty rungs,
    strongest price first: the adversary starts expensive-to-intervene
    and is made cheaper (stronger) only as the policy proves it holds.
  * HOLD gate: an honest probe (entrance, greedy, sticky 0.25,
    jitter 16 — the controller's standing protocol) whose median
    max-x is >= HOLD_MEDIAN (800). HOLDS_TO_ADVANCE (2) CONSECUTIVE
    holds are required before the next rung arms; any probe in
    [KILL_MEDIAN, HOLD_MEDIAN) resets the counter and stays put.
  * KILL: any probe median < KILL_MEDIAN (500) — respond with the K1
    rollback procedure (restore pre-hardening checkpoint, quarantine
    the degraded iters), not merely an abort. Killed state is
    absorbing: further probes are no-ops.
  * Completion: HOLDS_TO_ADVANCE consecutive holds at the final rung
    (budget 0.1) completes the ramp; the kill line stays armed and
    healthy probes read as plain holds thereafter.

Grounding for the 800/500 lines: attempt 7's un-anchored collapse read
honest median 899 -> 671 -> 196 across two probe cycles
(runs/online_1_2_attempt_ledger.md) — 671 would have RESET this ramp
(no advance) and 196 would have killed it two cycles before the
controller's own 150 competence floor formality.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

BUDGET_RAMP: tuple[float, ...] = (0.3, 0.2, 0.1)
HOLD_MEDIAN: float = 800.0
KILL_MEDIAN: float = 500.0
HOLDS_TO_ADVANCE: int = 2

# The H1 anchor stack — the exact reinforce keys every hardening-4b
# launch profile must carry (cross-checked against the catalog's blocks
# by tests/test_hardening4b_catalog.py).
ANCHOR_CHECKPOINT: str = (
    "checkpoints/_preserved/online_v2_FINAL_consolidated.pt")
ANCHOR_LOSS_COEF: float = 0.3
HARDENING_ENTROPY_COEF: float = 0.01
HARDENING_EPOCHS: int = 10  # 10:10 — reinforce.steps : adversary.epochs


def check_armed_profile(
    profile: dict, expected_budget: float | None = None,
) -> list[str]:
    """Pre-launch guard: verify a LAUNCH-EFFECTIVE profile still carries
    the registered hardening-4b stack. Returns a list of violations —
    empty means armed; any non-empty result is a registered NO-LAUNCH.

    This exists because the anchor can be disarmed silently AFTER the
    hand-merge: scripts/run_online_campaign.py's phase-4 and phase-5
    override rows both pin ``kl_anchor_loss_coef: 0.0`` and
    build_phase_profile deep-merges them ONTO whatever base it is given
    (and unconditionally overwrites entropy_coef from the cumulative
    schedule). Launching a hand-merged hardening-4b base through
    ``--start-phase 4`` therefore re-runs an UN-ANCHORED adversary — the
    exact attempt-7 collapse this package exists to prevent. Run this
    check on the EXACT profile file handed to the trainer, after all
    merging, and record the empty verdict in the decision log BEFORE
    launch (the catalog's receipts require it).

    Checks, in two tiers:
      * anchor tier (always): kl_anchor_checkpoint == the preserved
        consolidated head, kl_anchor_loss_coef == 0.3, reward-level
        betas 0, actor unfrozen, and sam_rho <= 0 (ppo_updater applies
        the loss tether only when SAM is off).
      * adversary tier (when ``expected_budget`` is given, or the
        profile itself arms an adversary ``mode``): mode kernel_sticky,
        budget_penalty == expected_budget (or any registered rung),
        10:10 epochs, entropy 0.01 both sides.
    """
    violations: list[str] = []
    rl = (profile or {}).get("reinforce")
    if not isinstance(rl, dict):
        return ["profile has no reinforce block"]

    def _want(key: str, expected) -> None:
        got = rl.get(key)
        if got != expected:
            violations.append(
                f"reinforce.{key} must be {expected!r}, got {got!r}")

    _want("kl_anchor_checkpoint", ANCHOR_CHECKPOINT)
    _want("kl_anchor_loss_coef", ANCHOR_LOSS_COEF)
    _want("kl_beta_start", 0.0)
    _want("kl_beta_end", 0.0)
    _want("actor_freeze_steps", 0)
    sam_rho = rl.get("sam_rho", 0.0) or 0.0
    if float(sam_rho) > 0.0:
        violations.append(
            f"reinforce.sam_rho is {sam_rho!r} — the loss-level tether "
            f"is inert when sam_rho > 0 (ppo_updater), so the anchor "
            f"would be silently off")

    adv = rl.get("adversary") or {}
    armed = expected_budget is not None or adv.get("mode") is not None
    if armed:
        if adv.get("mode") != "kernel_sticky":
            violations.append(
                f"reinforce.adversary.mode must be 'kernel_sticky', "
                f"got {adv.get('mode')!r}")
        budget = adv.get("budget_penalty")
        if expected_budget is not None:
            if budget != expected_budget:
                violations.append(
                    f"reinforce.adversary.budget_penalty must be "
                    f"{expected_budget!r}, got {budget!r}")
        elif budget not in BUDGET_RAMP:
            violations.append(
                f"reinforce.adversary.budget_penalty {budget!r} is not "
                f"a registered rung {BUDGET_RAMP}")
        if adv.get("epochs") != HARDENING_EPOCHS:
            violations.append(
                f"reinforce.adversary.epochs must be "
                f"{HARDENING_EPOCHS}, got {adv.get('epochs')!r}")
        if adv.get("entropy_coef") != HARDENING_ENTROPY_COEF:
            violations.append(
                f"reinforce.adversary.entropy_coef must be "
                f"{HARDENING_ENTROPY_COEF}, got "
                f"{adv.get('entropy_coef')!r}")
        if rl.get("steps") != HARDENING_EPOCHS:
            violations.append(
                f"reinforce.steps must be {HARDENING_EPOCHS} (10:10 "
                f"epochs), got {rl.get('steps')!r}")
        if rl.get("entropy_coef") != HARDENING_ENTROPY_COEF:
            violations.append(
                f"reinforce.entropy_coef must be pinned to "
                f"{HARDENING_ENTROPY_COEF} (the controller's schedule "
                f"overwrites it on a phase launch), got "
                f"{rl.get('entropy_coef')!r}")
    return violations


@dataclass(frozen=True)
class RampState:
    """Immutable ramp position: rung index into BUDGET_RAMP, the
    consecutive-hold counter at that rung, and the two terminal flags."""

    rung: int = 0
    holds: int = 0
    killed: bool = False
    complete: bool = False


def budget_for(state: RampState) -> float:
    """The adversary budget_penalty the current rung prescribes."""
    return BUDGET_RAMP[state.rung]


def step_ramp(state: RampState, probe_median: float) -> tuple[RampState, str]:
    """Fold one honest-probe median into the ramp state.

    Returns ``(new_state, action)`` with action one of:
      * ``"kill_rollback"`` — median < KILL_MEDIAN: apply K1 (restore
        pre-hardening checkpoint); terminal.
      * ``"noop"``          — state already killed; nothing to decide.
      * ``"hold"``          — median >= HOLD_MEDIAN, gate not yet met
        (or ramp already complete): stay at the rung.
      * ``"advance"``       — second consecutive hold with a next rung
        available: move to it (counter resets for the new rung).
      * ``"complete"``      — second consecutive hold at the FINAL
        rung: ramp finished at budget 0.1.
      * ``"reset"``         — median in [KILL_MEDIAN, HOLD_MEDIAN):
        consecutive counter back to zero, rung unchanged.
    """
    if state.killed:
        return state, "noop"
    m = float(probe_median)
    if m < KILL_MEDIAN:
        return replace(state, killed=True), "kill_rollback"
    if m < HOLD_MEDIAN:
        return replace(state, holds=0), "reset"
    # A hold.
    if state.complete:
        return state, "hold"
    holds = state.holds + 1
    if holds < HOLDS_TO_ADVANCE:
        return replace(state, holds=holds), "hold"
    if state.rung < len(BUDGET_RAMP) - 1:
        return replace(state, rung=state.rung + 1, holds=0), "advance"
    return replace(state, holds=0, complete=True), "complete"


def run_stream(
    probe_medians: Iterable[float], state: RampState | None = None,
) -> list[dict]:
    """Fold a whole stub/replayed probe stream; one row per probe.

    Each row carries the median consumed, the action taken, and the
    post-step position (rung, budget, killed, complete) — the exact
    trace an operator writes into the decision log next to the probe
    receipts.
    """
    s = state if state is not None else RampState()
    rows: list[dict] = []
    for m in probe_medians:
        s, action = step_ramp(s, m)
        rows.append({
            "median": float(m),
            "action": action,
            "rung": s.rung,
            "budget": budget_for(s),
            "killed": s.killed,
            "complete": s.complete,
        })
    return rows
