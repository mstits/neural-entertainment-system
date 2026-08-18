"""SMB online-campaign phase controller (1-2 by default; any level via
`--campaign-config`).

Drives the six-phase schedule over the four-mechanism trainer stack
(configs/mario_1_2_online_v2.yaml): one trainer SUBPROCESS per phase, each
launched via scripts/train_game.py with phase-specific overrides layered
onto the base profile. Every phase keeps the same profile `name`, so every
phase auto-resumes the previous phase's checkpoint in
checkpoints/mario_1_2_online_v2/ — the trainer's own tear-tolerant resume
path is the hand-off mechanism.

Phases:
  0 critic_warmup   zero-noise; A7 actor frozen (critic-only warmup)
  1 local_clear     zero-noise; earn deterministic clears from the x2500 rung
  2 sticky_local    house sticky 0.25; robustify the local clear
  3 reverse_walk    SPDL-style backward expansion: tau walks x2500 -> x0
  4 hardening       kernel_sticky adversary + cold entrance starts
  5 consolidation   adversary off, sticky 0.25, low entropy

Every 5M env-steps the controller probes the latest checkpoint with the
HONEST protocol (30 episodes, cold from the 1-2 entrance, greedy, sticky
0.25, jitter 16) via a scripts/eval_game.py subprocess, and logs median
max-x + bottleneck survival (fraction of episodes reaching the x~2674
approach) + clear rate to runs/online_1_2/campaign.jsonl.

Pre-registered kill criteria (CONFIG below; mirrored into the manifest):
  * kl_anchor_div > 0.15 sustained for 2M env-steps during phase 1;
  * value-loss spike > 500% of the trailing baseline, unrecovered for 1M
    env-steps (any phase);
  * 20M phase-1 env-steps without a single SIL clear.
On a kill: SIGINT the trainer, write the reason to campaign.jsonl, exit
nonzero. A phase that exhausts its step budget without meeting its
advance gate aborts the same way (phases 4 and 5 are budget-gated and
complete on exhaustion).

`--dry-run` validates the whole assembly without training: base + phase
profiles pass the strict schema, the A7 anchor shape-infers into the
configured net dims, the minted restart states load and match their
manifest sha256s, and the probe command assembles against real paths.

Long runs are launched by the operator, not by edits to this file: every
threshold lives in CONFIG at the top and is mirrored into
runs/online_1_2/manifest.json at campaign start.

RETARGETING TO ANOTHER LEVEL (`--campaign-config <yaml>`): the same
controller drives a different level package by loading a small override
document over CONFIG and PHASES — no fork, no second copy of the phase
logic. The doc carries `config:` (keys that must ALREADY exist in
CONFIG — an unregistered key is a typo and raises) and optionally
`phases:` (a full replacement schedule; every phase is validated for a
known gate and a sequential index). Without the flag, CONFIG and PHASES
are exactly the pinned 1-2 values. See configs/campaign_1_3.yaml and
docs/receipts/one_three_coordination_2026-08-16.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# CONFIG — every threshold the campaign runs on. Mirrored verbatim into
# runs/online_1_2/manifest.json (build_manifest) so the pre-registration
# and the run can never drift apart.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "base_profile": "configs/mario_1_2_online_v2.yaml",
    "run_dir": "runs/online_1_2",
    "campaign_log": "runs/online_1_2/campaign.jsonl",
    "restart_states_dir": "checkpoints/online_1_2/restart_states",
    # Identity: `campaign_name` labels the manifest, `probe_game` is the
    # logical game name eval_game.py echoes into its JSON (the profile
    # and ROM are passed explicitly, so this is a label, not a lookup).
    "campaign_name": "smb_1_2_online_v2",
    "probe_game": "mario_1_2_online_v2",
    # The level this campaign trains. The dry run refuses a restart set
    # whose ladder was minted for a different level, or from a root that
    # is not the base profile's own `start_state_path`. Schema-compatible
    # is NOT the same as provenance-compatible: the on-disk
    # checkpoints/backward_states/1-2 ladder is a valid index.json whose
    # meta names root entrance_after_1-1.state and profile
    # smb_4_4_micro.yaml, so consuming a ladder from another lane without
    # this check seeds the campaign from a foreign room, silently.
    "campaign_level": "1-2",

    # Honest probe protocol (Machado sticky + start jitter, greedy).
    "probe_every_env_steps": 5_000_000,
    "probe_episodes": 30,
    "probe_sticky": 0.25,
    "probe_jitter": 16,
    "probe_max_steps": 3000,
    "probe_eval_workers": 5,
    "probe_seed": 20260814,
    "probe_timeout_s": 3600.0,

    # The 1-2 wall: x~2674 off-manifold-drift barrier. An episode whose
    # max gx reaches bottleneck_survival_x has survived the approach.
    "bottleneck_x": 2674,
    "bottleneck_survival_x": 2600,

    # Kill criteria (pre-registered).
    # Attempt-4 re-registration (logged BEFORE launch; attempt-3 receipt:
    # runs/online_1_2_attempt3/). Attempt 3's loss-level tether produced a
    # STABLE plateau at KL~0.31 (iters 51-73: oscillation 0.19-0.50 settling
    # flat at 0.30-0.32, vs attempt 2's monotone unspool to ~1.4) — the 0.15
    # threshold, registered before any tether existed, killed the healthiest
    # dynamics yet. The failure mode this kill exists to catch is monotone
    # divergence, and the two attempts' telemetry separates the modes at
    # ~0.6. A bounded plateau is not forgetting; competence collapse is —
    # so a second, direct kill on probe competence is added below.
    "kill_kl_threshold": 0.60,
    "kill_kl_sustain_steps": 2_000_000,       # phase 1 only
    # Competence kill (any phase, any probe): if the honest probe's median
    # max-x falls below this floor, the anchor has failed at the only level
    # that matters — the policy lost even the early-hazard competence the
    # frozen A7 prior scores (~187 median on the probe seed). Fires
    # immediately, no sustain window.
    "kill_probe_median_floor": 150.0,
    "kill_vloss_spike_ratio": 5.0,            # >500% of trailing baseline
    "kill_vloss_recovery_steps": 1_000_000,
    "kill_vloss_baseline_rows": 20,           # rows before the spike arms
    "kill_phase1_no_sil_clear_steps": 20_000_000,
    # Rung-progress kill (audit-7). The backward curriculum's cursor is
    # the thing being falsified: if tau has not walked back past the
    # ladder midpoint after this many CUMULATIVE campaign env-steps, the
    # walk is not walking and the campaign aborts instead of spending
    # the rest of its budget to report the same thing. Cumulative, and
    # checked in every phase that does not pin the entrance, because the
    # curriculum sits in the BASE profile and advances from phase 0 —
    # scoping it to the reverse-walk phase would arm it where 1-2's
    # precedent proves it cannot fire (see `rung_progress_kill`).
    # PROVISIONAL wherever it is armed: no banked run measures a
    # reverse-walk rate, so a value here is a bound, not a calibration.
    # 0 == DISARMED, which is the 1-2 default (its ledger predates the
    # criterion; retargeted campaigns register a value in their
    # --campaign-config).
    "kill_rung_halfway_steps": 0,

    # Advance gates.
    # Attempt-1 calibration: raw huber value loss on this reward scale
    # oscillated 30-95 (CV ~0.3) at 8M steps and never approached 10%.
    # 0.25 over a 50-iter trail plus a minimum-steps floor replaces the
    # old actor_freeze_steps coupling (freeze is now a config sentinel).
    "gate_critic_cv": 0.25,
    "gate_critic_window_iters": 50,
    # 5M floor guarded a FRESH critic; on resumed runs the critic
    # arrives with 50M+ steps of training and passes instantly, so the
    # floor is pure re-warmup overhead. 2M still forces a real trailing
    # window before the gate can fire. (Attempt-6 re-registration.)
    "gate_min_phase0_steps": 2_000_000,
    "gate_det_clears": 10,                    # phase 1: 10/10 deterministic
    "gate_sticky_clear": 0.80,                # phase 2: sticky-local > 80%
    # phase 3 gate: backward tau == 0 (read from the latest checkpoint's
    # persisted backward_curriculum state).

    # Entropy schedule (applied per phase launch from cumulative steps —
    # the trainer has no global env-step entropy schedule).
    "entropy_start": 0.01,
    "entropy_end": 0.002,
    "entropy_decay_steps": 100_000_000,

    # Controller cadence.
    "poll_secs": 15.0,
    "trainer_stop_grace_s": 180.0,
}

# Phase schedule. `budget_env_steps` bounds the phase; gate types:
#   critic_warmup      — trailing critic-loss CV < gate_critic_cv AND the
#                        actor-freeze window has elapsed;
#   det_local_clears   — probe (sticky 0, jitter 0, greedy) clears
#                        gate_det_clears / gate_det_clears episodes;
#   sticky_clear       — honest probe clear rate > gate_sticky_clear;
#   restart_at_entrance— backward tau == 0 in the latest checkpoint;
#   budget             — phase completes when the budget is spent.
PHASES: tuple[dict, ...] = (
    # Phase 0 keeps the base profile's sentinel actor_freeze_steps: the
    # actor stays frozen until the critic-stability gate passes, however
    # long that takes — unfreeze is gate-coupled, never timer-coupled
    # (attempt-1 post-mortem). Phases 1+ unfreeze (freeze 0) and phase 1
    # additionally drops lr to 3e-5 so the fresh-unfrozen actor takes
    # small steps while the KL anchor (beta 0.5 at unfreeze) holds.
    # Attempt 3 adds the loss-level tether (kl_anchor_loss_coef): the
    # reward-level beta alone let KL blow 0 -> 0.574 in ONE PPO iteration
    # at unfreeze; the per-update term pulls every minibatch back toward
    # the prior. 1.0 at unfreeze, relaxed to 0.3 while robustifying,
    # 0.0 from hardening on (anchor fully removed, per the accepted spec).
    {"idx": 0, "name": "critic_warmup", "gate": "critic_warmup",
     "budget_env_steps": 15_000_000,
     "overrides": {"reinforce": {"sticky_action_prob": 0.0}}},
    {"idx": 1, "name": "local_clear", "gate": "det_local_clears",
     "budget_env_steps": 40_000_000,
     "overrides": {"reinforce": {"sticky_action_prob": 0.0,
                                 "actor_freeze_steps": 0,
                                 "lr": 0.00003,
                                 "kl_anchor_loss_coef": 1.0}}},
    # Attempt-6 re-registration (evidence: runs/online_1_2_attempt5/).
    # Phase 2's sticky_clear gate (>0.8 rung-local under sticky+jitter)
    # was measured 5 times at 0.7/0.2/0.2/0.37/0.1 — mean ~0.31, no
    # trend, median max-x pinned at 2675 on every probe — while the
    # campaign's true metric moved: entrance honest clears emerged
    # (5/150 pooled) and bottleneck survival from the rung was 30/30 on
    # every single probe. The gate measured the minted tape state's
    # entry conditions (frozen velocity/enemy phase + jitter), not the
    # capability the mission needs; as a gated phase it would have
    # budget-ABORTED the campaign before the reverse walk ever ran.
    # Re-registered as budget-complete at 2M (25M+ of sticky wall
    # practice already banked across attempts); the rate-building work
    # belongs to phase 3.
    {"idx": 2, "name": "sticky_local", "gate": "budget",
     "budget_env_steps": 2_000_000,
     "overrides": {"reinforce": {"sticky_action_prob": 0.25,
                                 "actor_freeze_steps": 0,
                                 "kl_anchor_loss_coef": 0.3}}},
    {"idx": 3, "name": "reverse_walk", "gate": "restart_at_entrance",
     "budget_env_steps": 60_000_000,
     "overrides": {"reinforce": {"sticky_action_prob": 0.25,
                                 "actor_freeze_steps": 0,
                                 "kl_anchor_loss_coef": 0.3}}},
    # Phase 4's perturbation is the kernel-matched adversary (house sticky
    # off — the two would double-perturb the same channel); restarts pin
    # to the true entrance for cold hardening.
    {"idx": 4, "name": "hardening", "gate": "budget",
     "budget_env_steps": 20_000_000,
     "overrides": {"reinforce": {
         "sticky_action_prob": 0.0,
         "actor_freeze_steps": 0,
         "kl_anchor_loss_coef": 0.0,
         "adversary": {"mode": "kernel_sticky"},
         "backward_curriculum": {"pin_entrance": True}}}},
    {"idx": 5, "name": "consolidation", "gate": "budget",
     "budget_env_steps": 10_000_000,
     "overrides": {"reinforce": {
         "sticky_action_prob": 0.25,
         "actor_freeze_steps": 0,
         "kl_anchor_loss_coef": 0.0,
         "backward_curriculum": {"pin_entrance": True}}}},
)


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested with stubbed streams — tests/test_online_campaign.py)
# ---------------------------------------------------------------------------


# Every gate `_gate_met` implements. Literal (not derived) so a
# retargeted campaign's schedule is validated against a registry that a
# refactor of `_gate_met` cannot silently widen.
KNOWN_GATES: frozenset[str] = frozenset({
    "budget", "critic_warmup", "det_local_clears", "sticky_clear",
    "restart_at_entrance",
})

# Top-level keys a --campaign-config document may carry.
CAMPAIGN_DOC_KEYS: frozenset[str] = frozenset({"config", "phases", "notes"})

# Keys every phase entry must carry.
PHASE_REQUIRED_KEYS: tuple[str, ...] = (
    "idx", "name", "gate", "budget_env_steps")


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursive dict merge; returns a new dict, mutates neither input."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _same_kind(old: Any, new: Any) -> bool:
    """Type compatibility for a CONFIG override (int<->float is fine)."""
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return True
    return type(old) is type(new)


def merge_campaign_config(
    base_cfg: dict, base_phases: tuple[dict, ...], doc: dict,
) -> tuple[dict, tuple[dict, ...]]:
    """PURE: the (CONFIG, PHASES) a --campaign-config document produces.

    Neither input is mutated. Validation is deliberately strict — this
    document is a pre-registration, and a silently-ignored key here is a
    threshold that does not exist at 3 a.m.:

    * only `config` / `phases` / `notes` at the top level;
    * every `config` key must ALREADY exist in the controller's CONFIG,
      with a compatible type (a typo'd threshold raises, it does not
      quietly join the dict);
    * `phases`, if present, REPLACES the schedule wholesale (a partial
      phase merge would make "which budget is live?" unanswerable). Each
      entry needs idx/name/gate/budget_env_steps, a gate in KNOWN_GATES,
      a sequential idx from 0, and a dict `overrides` if it has one.
    """
    if not isinstance(doc, dict):
        raise ValueError(
            f"campaign config must be a YAML mapping, got {type(doc).__name__}")
    unknown_top = sorted(set(doc) - CAMPAIGN_DOC_KEYS)
    if unknown_top:
        raise ValueError(
            f"unknown campaign-config key(s) {unknown_top}; allowed: "
            f"{sorted(CAMPAIGN_DOC_KEYS)}")

    cfg = copy.deepcopy(base_cfg)
    over = doc.get("config") or {}
    if not isinstance(over, dict):
        raise ValueError("campaign-config `config` must be a mapping")
    for k, v in over.items():
        if k not in cfg:
            raise ValueError(
                f"campaign-config sets unknown CONFIG key {k!r} — it would "
                f"be ignored by the controller (typo? renamed?)")
        if not _same_kind(cfg[k], v):
            raise ValueError(
                f"campaign-config key {k!r}: expected "
                f"{type(cfg[k]).__name__}, got {type(v).__name__}")
        cfg[k] = copy.deepcopy(v)

    raw_phases = doc.get("phases")
    if raw_phases is None:
        return cfg, copy.deepcopy(base_phases)
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("campaign-config `phases` must be a non-empty list")
    phases: list[dict] = []
    for i, p in enumerate(raw_phases):
        if not isinstance(p, dict):
            raise ValueError(f"phase {i} must be a mapping")
        missing = [k for k in PHASE_REQUIRED_KEYS if k not in p]
        if missing:
            raise ValueError(f"phase {i} is missing {missing}")
        if int(p["idx"]) != i:
            raise ValueError(
                f"phase indices must be sequential from 0; entry {i} "
                f"declares idx {p['idx']}")
        if str(p["gate"]) not in KNOWN_GATES:
            raise ValueError(
                f"phase {i} ({p['name']}) declares unknown gate "
                f"{p['gate']!r}; known gates: {sorted(KNOWN_GATES)}")
        if int(p["budget_env_steps"]) <= 0:
            raise ValueError(
                f"phase {i} ({p['name']}) needs a positive budget_env_steps")
        ov = p.get("overrides", {})
        if not isinstance(ov, dict):
            raise ValueError(f"phase {i} `overrides` must be a mapping")
        phases.append(copy.deepcopy(p))
    return cfg, tuple(phases)


def apply_campaign_config(path) -> dict:
    """Load a campaign-override doc and install it into CONFIG/PHASES.

    CONFIG is updated IN PLACE (callers hold references to it) and
    PHASES is rebound at module scope. Returns the raw document so the
    caller can log what was applied.
    """
    doc = _load_yaml(path) or {}
    cfg, phases = merge_campaign_config(CONFIG, PHASES, doc)
    CONFIG.clear()
    CONFIG.update(cfg)
    globals()["PHASES"] = phases
    return doc


def entropy_coef_at(env_steps: float) -> float:
    """Linear entropy_start -> entropy_end over entropy_decay_steps."""
    frac = min(1.0, max(0.0, float(env_steps)) / CONFIG["entropy_decay_steps"])
    return CONFIG["entropy_start"] + frac * (
        CONFIG["entropy_end"] - CONFIG["entropy_start"])


def build_phase_profile(base: dict, phase: dict, cum_env_steps: float) -> dict:
    """The profile a phase launches with: base + overrides + entropy.

    Always layered onto the BASE profile (not the previous phase's), so a
    phase can only be what it declares — e.g. the adversary `mode` exists
    in phase 4's profile and nowhere else. The `name` is never touched:
    every phase must resume the same checkpoint dir.
    """
    prof = deep_merge(base, phase.get("overrides", {}))
    prof.setdefault("reinforce", {})["entropy_coef"] = round(
        entropy_coef_at(cum_env_steps), 6)
    return prof


def steps_per_iter(profile: dict) -> int:
    rl = profile.get("reinforce", {}) or {}
    return int(rl.get("rollout_steps", 512)) * int(rl.get("num_envs", 60))


def critic_cv(vlosses) -> float:
    """Coefficient of variation of a critic-loss window (inf when empty
    or degenerate — an empty window must never pass a variance gate)."""
    vals = [float(v) for v in vlosses if v is not None]
    if len(vals) < 2:
        return float("inf")
    mean = statistics.fmean(vals)
    if mean <= 0.0:
        return float("inf")
    return statistics.pstdev(vals) / mean


def probe_summary(eval_json: dict, survival_x: int) -> dict:
    """Median max-x / bottleneck survival / clear rates off eval_game JSON.

    PREDICATE DISCIPLINE (2026-08-15 forensics,
    docs/receipts/eval_rng_regimes_2026-08-15.md): sequential probes
    carry TWO clear predicates that disagreed 13:1 on real data —
    seq_clear_rate (level-chain advance, fires on reaching the flag
    area) and clear_rate (strict episode_success: flagpole/castle).
    The headline `clear_rate` keeps its historical chain-preferring
    semantics so pinned gates/kills are unchanged, but both predicates
    are now carried explicitly so no row can be quoted without one.
    """
    gxs = [int(g) for g in (eval_json.get("max_gx_per_episode") or [])]
    chain = eval_json.get("seq_clear_rate")
    strict = eval_json.get("clear_rate")
    clear = chain if chain is not None else strict
    return {
        "n_episodes": int(eval_json.get("n_episodes", len(gxs)) or 0),
        "median_max_x": float(statistics.median(gxs)) if gxs else 0.0,
        "bottleneck_survival": (
            sum(1 for g in gxs if g >= int(survival_x)) / len(gxs)
            if gxs else 0.0),
        "clear_rate": float(clear or 0.0),
        "clear_rate_chain": (float(chain) if chain is not None else None),
        "clear_rate_strict": (float(strict) if strict is not None else None),
    }


class KillMonitor:
    """The three pre-registered kill criteria over a metrics.jsonl stream.

    Fed one metrics row per iter via `observe(row)`; returns None while
    healthy, or a human-readable kill reason the moment a criterion
    fires. Pure over the row stream — no filesystem, no subprocess — so
    the whole surface is testable with synthetic rows.
    """

    def __init__(self, cfg: dict, steps_per_iter: int) -> None:
        self.cfg = dict(cfg)
        self.spi = int(steps_per_iter)
        self._phase = -1
        self._phase_start_steps = 0.0
        self._sil_baseline: Optional[float] = None
        self._sil_cleared = False
        self._kl_high_since: Optional[float] = None
        self._vloss_hist: deque = deque(
            maxlen=int(cfg["kill_vloss_baseline_rows"]))
        self._spike_since: Optional[float] = None
        self._spike_baseline: Optional[float] = None

    def start_phase(self, phase_idx: int, env_steps: float,
                    sil_clears_total: Optional[float]) -> None:
        self._phase = int(phase_idx)
        self._phase_start_steps = float(env_steps)
        self._sil_baseline = (
            float(sil_clears_total) if sil_clears_total is not None else None)
        self._sil_cleared = False
        self._kl_high_since = None
        # Value-loss baseline deliberately survives phase changes: the
        # criterion is a run-level health check, and a phase boundary is
        # not a spike amnesty.

    def _env_steps(self, row: dict) -> float:
        return (float(row.get("generation", 0)) + 1.0) * self.spi

    def observe(self, row: dict) -> Optional[str]:
        steps = self._env_steps(row)

        # --- KL anchor divergence (phase 1 only) ---
        kl = row.get("kl_anchor_div")
        if self._phase == 1 and kl is not None:
            if float(kl) > self.cfg["kill_kl_threshold"]:
                if self._kl_high_since is None:
                    self._kl_high_since = steps
                elif steps - self._kl_high_since >= (
                        self.cfg["kill_kl_sustain_steps"]):
                    return (
                        f"KILL kl_anchor_div>{self.cfg['kill_kl_threshold']} "
                        f"sustained {steps - self._kl_high_since:.0f} steps "
                        f"(>= {self.cfg['kill_kl_sustain_steps']}) in phase 1")
            else:
                self._kl_high_since = None

        # --- value-loss spike, unrecovered ---
        v = row.get("ppo_value_loss")
        if v is not None:
            v = float(v)
            if self._spike_since is None:
                if (len(self._vloss_hist) >=
                        self.cfg["kill_vloss_baseline_rows"]):
                    baseline = statistics.median(self._vloss_hist)
                    if baseline > 0 and v > (
                            self.cfg["kill_vloss_spike_ratio"] * baseline):
                        # Freeze the pre-spike baseline: the spike must
                        # not launder itself into its own reference.
                        self._spike_since = steps
                        self._spike_baseline = baseline
                if self._spike_since is None:
                    self._vloss_hist.append(v)
            else:
                if v <= (self.cfg["kill_vloss_spike_ratio"] *
                         self._spike_baseline):
                    self._spike_since = None
                    self._spike_baseline = None
                    self._vloss_hist.append(v)
                elif steps - self._spike_since >= (
                        self.cfg["kill_vloss_recovery_steps"]):
                    return (
                        f"KILL value-loss spike >{self.cfg['kill_vloss_spike_ratio']:.0f}x "
                        f"baseline {self._spike_baseline:.4f} unrecovered for "
                        f"{steps - self._spike_since:.0f} steps "
                        f"(>= {self.cfg['kill_vloss_recovery_steps']})")

        # --- phase-1 SIL-clear drought ---
        sil = row.get("sil_clears_total")
        if self._phase == 1 and sil is not None:
            if self._sil_baseline is None:
                self._sil_baseline = float(sil)
            if float(sil) > self._sil_baseline:
                self._sil_cleared = True
            phase_steps = steps - self._phase_start_steps
            if (not self._sil_cleared and phase_steps >=
                    self.cfg["kill_phase1_no_sil_clear_steps"]):
                return (
                    f"KILL {phase_steps:.0f} phase-1 steps "
                    f"(>= {self.cfg['kill_phase1_no_sil_clear_steps']}) "
                    f"without a single SIL clear")
        return None


def restart_provenance_ok(
    meta: Optional[dict], base_profile: dict, cfg: dict,
) -> tuple[bool, str]:
    """PURE: does this restart set belong to THIS campaign?

    `mint_backward_states.py` stamps every ladder's `index.json` meta
    with the level it minted, the root blob it replayed from and the
    profile it replayed under, and `select_restart_states.py` copies
    that meta forward into the selection. Schema checks read none of it,
    so a ladder minted by another lane for another level from another
    root passes every other check in this dry run.

    That is a live on-disk situation, not a hypothetical: the banked
    `checkpoints/backward_states/1-2` ladder's meta names root
    `runs/live_show/smb_4_4_micro/entrance_after_1-1.state` and profile
    `configs/smb_4_4_micro.yaml` over a 1215-action tape, while the 1-2
    campaign profile starts at `stage_03.state` and the restart set it
    actually shipped came from an 871-action stage_03-rooted tape.
    Consuming the former as a `--ladder` input would have seeded the
    campaign from a foreign room with every check printing PASS.

    Two things must agree, and both are compared by resolved path /
    exact string so a near-miss is a failure:

      * `meta["level"]` == `cfg["campaign_level"]`;
      * `meta["root_state"]` == the base profile's `start_state_path`.

    A ladder with no `level`/`root_state` in its meta FAILS. Unstamped
    provenance is not a pass — it is a ladder whose origin cannot be
    established, which is the case this gate exists to refuse.
    """
    if not isinstance(meta, dict):
        return False, f"no ladder meta to check (got {type(meta).__name__})"
    want_level = str(cfg.get("campaign_level") or "")
    got_level = meta.get("level")
    want_root = base_profile.get("start_state_path")
    got_root = meta.get("root_state")
    if got_level is None or got_root is None:
        return False, (
            f"ladder meta is missing provenance (level={got_level!r}, "
            f"root_state={got_root!r}); re-mint with "
            f"scripts/mint_backward_states.py")
    if not want_level:
        return False, "campaign_level is unset — nothing to check against"
    if not want_root:
        return False, "base profile has no start_state_path to check against"
    bad = []
    if str(got_level) != want_level:
        bad.append(f"level {got_level!r} != campaign_level {want_level!r}")
    if Path(str(got_root)).resolve() != Path(str(want_root)).resolve():
        bad.append(f"root_state {got_root!r} != profile start_state_path "
                   f"{want_root!r}")
    if bad:
        return False, "; ".join(bad) + " — this ladder is another lane's"
    return True, (f"level {got_level}, root {got_root}, profile "
                  f"{meta.get('profile')}, {meta.get('n_actions')} actions")


def preflight_restart_ladders(
    base_profile: dict, cfg: dict,
) -> tuple[bool, list[str]]:
    """PURE: do the ladder the TRAINER reads and the ladder the GATES
    read exist, agree with each other, and belong to this campaign?

    `restart_provenance_ok` asks the right question of the wrong path.
    It reads `cfg["restart_states_dir"]` — the key `_top_rung_state`
    uses to launch gate probes — while training restarts come from the
    base profile's `backward_curriculum.states_dir`. Those are two
    independent strings, and nothing compared them.

    Attempt 1 on 2-1 is what that costs: the campaign config named
    `checkpoints/online_2_1/restart_states` and the profile named
    `checkpoints/online_1_3/restart_states`, cloned from the 1-3 lane
    and never repointed. The dry run validated the campaign config's
    ladder and printed PASS, gate probes launched from 2-1's gx-2700
    rung, and 20M steps of backward-curriculum training ran against
    1-3's rungs. Every instrument read plausibly; the deterministic gate
    simply never passed, stalling at gx 2910 against a stretch of 2-1
    the policy had never been restarted into.

    So this checks three things, and any failure aborts the launch:

      * the profile's `states_dir` resolves to the same directory as
        `cfg["restart_states_dir"]` — disagreement means gates and
        training are measuring different ladders, which is silent;
      * that directory's meta passes `restart_provenance_ok`, i.e. it
        is this level's ladder from this profile's root;
      * a profile with no `backward_curriculum.states_dir` at all is a
        PASS only when the campaign config also names none, since a
        campaign can legitimately run without a ladder.
    """
    from src.training.backward_curriculum import load_index

    notes: list[str] = []
    prof_dir = (base_profile.get("reinforce", {})
                .get("backward_curriculum", {}) or {}).get("states_dir")
    cfg_dir = cfg.get("restart_states_dir")

    if not prof_dir and not cfg_dir:
        return True, ["no restart ladder configured on either side"]
    if bool(prof_dir) != bool(cfg_dir):
        return False, [
            f"only one side names a ladder (profile={prof_dir!r}, "
            f"campaign={cfg_dir!r}) — gates and training would disagree"]

    prof_path = (REPO / str(prof_dir)).resolve()
    cfg_path = (REPO / str(cfg_dir)).resolve()
    if prof_path != cfg_path:
        return False, [
            f"profile trains on {prof_dir!r} but gate probes read "
            f"{cfg_dir!r} — these must be the same ladder"]
    notes.append(f"trainer and gates agree on {prof_dir}")

    try:
        _, meta = load_index(prof_path)
    except Exception as e:  # unreadable ladder is a failure, not a skip
        return False, [f"cannot read ladder index at {prof_dir}: {e!r}"]

    ok, why = restart_provenance_ok(meta, base_profile, cfg)
    notes.append(why)
    return ok, notes


def phase_pins_entrance(phase: dict) -> bool:
    """True when this phase's overrides pin restarts to the entrance.

    A pinned phase has no cursor to falsify — tau is held at 0 by the
    schedule, not by learning — so the rung-progress kill must not read
    it. Everything else in the schedule leaves the base profile's
    backward curriculum live, and therefore advancing.
    """
    bwd = (((phase.get("overrides") or {}).get("reinforce") or {})
           .get("backward_curriculum") or {})
    return bool(bwd.get("pin_entrance"))


def rung_progress_kill(
    *, state: Optional[dict], curriculum_env_steps: float, cfg: dict,
) -> Optional[str]:
    """Audit-7 rung-progress kill over the persisted backward cursor.

    PURE over the curriculum's `state_dict()` (the trainer persists it
    under `backward_curriculum` in every checkpoint). Fires only when
    ALL of these hold:

      * the criterion is armed (`kill_rung_halfway_steps` > 0);
      * the CURRICULUM has burned at least that many env-steps;
      * the ladder is real (>= 2 entries) and the cursor is readable;
      * tau has NOT yet walked past the ladder midpoint — tau counts
        DOWN from the deepest rung to 0 (the entrance), so "past
        halfway" is `tau <= (n_entries - 1) // 2`.

    SCOPE — `curriculum_env_steps` is CUMULATIVE campaign steps, not
    phase-local, and the caller evaluates this in every phase that does
    not pin the entrance. That is not a refinement, it is the whole
    criterion: tau is a campaign-global cursor. The backward curriculum
    sits in the BASE profile, so it is live and advancing from phase 0,
    and the only banked precedent shows it finishing there — 1-2 attempt
    7 entered its reverse-walk phase and passed `restart_at_entrance` at
    the FIRST check because tau had already walked to 0 across the
    earlier phases (runs/online_1_2_attempt_ledger.md, attempt 7). A
    kill scoped to the reverse-walk phase alone, measured on that
    phase's own steps, would have had ~5M steps of a 25M threshold to
    fire in and could never have fired; meanwhile a cursor that genuinely
    stalls, stalls in phase 1 or 2, where nothing would have read it.

    CALIBRATION — the threshold is PROVISIONAL and must be treated like
    `kill_probe_median_floor`. No banked artifact measures a reverse-walk
    RATE: attempt 7 resumed from a checkpoint whose cursor was already at
    0, so the ledger bounds the walk only from above (tau reached 0 by
    the ~70.0M cumulative steps at which phase 3 opened) and never
    timestamps the midpoint. The threshold therefore has to sit far
    enough out that the one precedent survives it.

    NOT a post-hoc rationalisation of the 1-2 ledger. The ledger records
    no attempt lost to a stalled cursor — attempts 1-3 were unfreeze and
    tether calibration, 4 a miswired gate probe, 5 a miscalibrated gate,
    6 a resume re-litigating earned gates, 7 an un-anchored adversary.
    This criterion is a NEW pre-registration against a failure mode that
    has not been observed, which is why it is armed generously rather
    than tightly.

    Returns the kill reason, or None while healthy. A missing cursor is
    never a kill: an unreadable checkpoint is a probe problem, and a
    kill criterion that fires on missing data is a coin flip.
    """
    limit = float(cfg.get("kill_rung_halfway_steps") or 0)
    if limit <= 0 or float(curriculum_env_steps) < limit:
        return None
    if not isinstance(state, dict):
        return None
    tau, n = state.get("tau"), state.get("n_entries")
    if tau is None or n is None:
        return None
    tau, n = int(tau), int(n)
    if n < 2:
        return None
    halfway = (n - 1) // 2
    if tau <= halfway:
        return None
    return (
        f"KILL rung-progress: tau={tau}/{n - 1} after "
        f"{curriculum_env_steps:.0f} cumulative env-steps (>= {limit:.0f}) "
        f"— the backward curriculum has not reached the ladder midpoint "
        f"(tau <= {halfway})")


def build_manifest(base_profile: dict) -> dict:
    """The campaign manifest: CONFIG + phase schedule, mirrored verbatim."""
    return {
        "campaign": CONFIG["campaign_name"],
        "profile_name": base_profile.get("name"),
        "config": copy.deepcopy(CONFIG),
        "phases": copy.deepcopy(list(PHASES)),
        "created_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Runtime pieces (subprocess / filesystem; exercised by --dry-run and the
# real campaign, not by the unit tests)
# ---------------------------------------------------------------------------


def _load_yaml(path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: dict, path: Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _tail_jsonl(path: Path, offset: int) -> tuple[list[dict], int]:
    """New complete JSON rows past byte `offset` (tolerates a torn tail)."""
    if not path.exists():
        return [], offset
    if path.stat().st_size < offset:
        # The trainer truncates metrics.jsonl at the top of a fresh run;
        # a stale offset past EOF would blind the monitor for the whole
        # phase. Restart from the top of the truncated file.
        offset = 0
    rows: list[dict] = []
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    if not chunk:
        return [], offset
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    for line in chunk[:end].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, offset + end + 1


def checkpoint_dir_for(profile: dict) -> Path:
    from src.training.profile_utils import derive_checkpoint_dir
    return REPO / derive_checkpoint_dir("./checkpoints", profile.get("name"))


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    cands = sorted(ckpt_dir.glob("vanilla_ppo_iter_*.pt"), reverse=True)
    return cands[0] if cands else None


def load_checkpoint_payload(ckpt_dir: Path) -> Optional[dict]:
    """Newest loadable checkpoint payload (walks back over torn files)."""
    import torch
    for p in sorted(ckpt_dir.glob("vanilla_ppo_iter_*.pt"), reverse=True):
        try:
            return torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception:
            continue
    return None


def read_backward_state(payload: Optional[dict]) -> Optional[dict]:
    """The persisted TauScheduler state_dict from a checkpoint payload."""
    if not isinstance(payload, dict):
        return None
    state = payload.get("backward_curriculum")
    return state if isinstance(state, dict) else None


def read_backward_tau(payload: Optional[dict]) -> Optional[int]:
    state = read_backward_state(payload)
    if state is not None and "tau" in state:
        return int(state["tau"])
    return None


def _top_rung_state() -> Path:
    """The highest-gx minted restart state (the bottleneck approach rung).

    Gate probes are RUNG-LOCAL by design: phase 1/2 gates measure clears
    of the bottleneck from the x~2502 rung, not entrance-to-flag runs.
    (Attempt-4 harness defect: gate probes launched from the entrance,
    making the deterministic gate impossible — proven by a det gate probe
    median of 668 < 2502, arithmetically impossible from the rung.)
    Honest probes stay entrance-based; only gates use this.
    """
    manifest = json.loads(
        (REPO / CONFIG["restart_states_dir"] / "manifest.json").read_text())
    entries = manifest if isinstance(manifest, list) else (
        manifest.get("rungs") or manifest.get("states") or [])
    top = max(entries, key=lambda e: e.get("gx", -1))
    return REPO / CONFIG["restart_states_dir"] / top["file"]


def build_probe_command(
    *, checkpoint: Path, episodes: int, sticky: float, jitter: int,
    max_steps: int, eval_seed: int, start_state: Optional[Path] = None,
) -> list[str]:
    """The honest-probe eval_game argv (greedy; sticky/jitter as given).

    Stochastic probes get parallel lanes + per-episode RNG (eval_game
    refuses shared-stream parallel stochastic runs); the deterministic
    phase-1 gate probe runs a single serial lane. start_state overrides
    the profile entrance (rung-local gate probes); None = entrance.
    """
    base = _load_yaml(REPO / CONFIG["base_profile"])
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_game.py"),
        "--game", str(CONFIG["probe_game"]),
        "--profile", str(REPO / CONFIG["base_profile"]),
        "--rom", str(REPO / base["rom_path"]),
        "--checkpoint", str(checkpoint),
        "--episodes", str(int(episodes)),
        "--max-steps", str(int(max_steps)),
        "--sequential", "--level-clear",
        "--start-state", str(start_state if start_state is not None
                             else REPO / base["start_state_path"]),
        "--eval-seed", str(int(eval_seed)),
    ]
    if sticky > 0.0:
        cmd += ["--sticky-prob", str(float(sticky))]
    if jitter > 0:
        cmd += ["--start-jitter", str(int(jitter))]
    if sticky > 0.0 or jitter > 0:
        cmd += ["--eval-workers", str(int(CONFIG["probe_eval_workers"])),
                "--eval-rng", "per-episode"]
    return cmd


def run_probe(*, checkpoint: Path, episodes: int, sticky: float, jitter: int,
              eval_seed: int, start_state: Optional[Path] = None) -> dict:
    """Run one probe subprocess; returns eval_game's JSON (or an error dict)."""
    cmd = build_probe_command(
        checkpoint=checkpoint, episodes=episodes, sticky=sticky,
        jitter=jitter, max_steps=CONFIG["probe_max_steps"],
        eval_seed=eval_seed, start_state=start_state)
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True,
            timeout=CONFIG["probe_timeout_s"])
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout"}
    if proc.returncode != 0:
        return {"status": "probe_failed",
                "detail": proc.stderr.strip()[-500:]}
    text = proc.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"status": "probe_failed", "detail": "no JSON on stdout"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return {"status": "probe_failed", "detail": f"bad JSON: {e}"}


def _stop_trainer(proc: subprocess.Popen) -> None:
    """SIGINT -> clean trainer stop; escalate only if it hangs."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=CONFIG["trainer_stop_grace_s"])
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run() -> int:
    """Validate the whole assembly without training. Returns exit code."""
    from src.training.config_schema import check_profile
    from src.training.profile_utils import resolve_encoder

    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" — {detail}" if detail else ""), flush=True)

    # 1. Base profile parses + strict schema.
    base_path = REPO / CONFIG["base_profile"]
    try:
        base = _load_yaml(base_path)
        warns = check_profile(base, strict=False)
        report("base profile schema", not warns,
               "; ".join(warns) if warns else str(base_path))
    except Exception as e:
        report("base profile schema", False, repr(e))
        return 1

    # 2. Every phase profile passes the strict schema too.
    for phase in PHASES:
        prof = build_phase_profile(base, phase, cum_env_steps=0)
        warns = check_profile(prof, strict=False)
        report(f"phase {phase['idx']} ({phase['name']}) profile schema",
               not warns, "; ".join(warns))

    # 3. A7 anchor loads with shape inference into the configured dims.
    try:
        import torch
        from src.models.tile_policy import build_tile_policy_from_checkpoint
        rl = base["reinforce"]
        _, _, obs_width = resolve_encoder(base)
        payload = torch.load(str(REPO / rl["kl_anchor_checkpoint"]),
                             map_location="cpu", weights_only=False)
        sd = payload.get("net_state_dict", payload)
        prior, is_recurrent = build_tile_policy_from_checkpoint(
            payload, num_actions=len(base["action_space"]),
            feature_dim=obs_width)
        missing, _ = prior.load_state_dict(sd, strict=False)
        fc1 = tuple(sd["fc1.weight"].shape)
        fc2 = tuple(sd["fc2.weight"].shape)
        dims_ok = (
            not is_recurrent and not missing
            and fc1 == (int(rl["tile_hidden_dim"]), obs_width)
            and fc2 == (int(rl["tile_trunk_dim"]), int(rl["tile_hidden_dim"]))
        )
        report("KL anchor shape-infers into the configured net", dims_ok,
               f"fc1={fc1} fc2={fc2} obs={obs_width} recurrent={is_recurrent}"
               f" missing={missing}")
    except Exception as e:
        report("KL anchor shape-infers into the configured net", False,
               repr(e))

    # 3b. The wavefront distance map exists and unpickles (a missing dmap
    # is a launch-time crash inside the trainer subprocess, i.e. a phase
    # log nobody reads until the window is gone).
    wave = (base.get("reinforce", {}) or {}).get("wavefront_reward") or {}
    if wave.get("enabled") and wave.get("dmap"):
        try:
            import pickle
            with open(REPO / wave["dmap"], "rb") as f:
                dmap = pickle.load(f)
            ds = [int(v) for v in dmap.values()]
            report("wavefront dmap loads", bool(ds),
                   f"{len(dmap)} cells, dist range [{min(ds)}, {max(ds)}]"
                   if ds else "empty dmap")
        except Exception as e:
            report("wavefront dmap loads", False, repr(e))

    # 4. Restart states load and match their manifest sha256s.
    try:
        from src.training.backward_curriculum import load_blobs, load_index
        states_dir = REPO / CONFIG["restart_states_dir"]
        entries, meta = load_index(states_dir)
        blobs = load_blobs(states_dir, entries)
        manifest = json.loads((states_dir / "manifest.json").read_text())
        by_file = {s["file"]: s for s in manifest["states"]}
        sha_ok = all(
            hashlib.sha256(blob).hexdigest() == by_file[e.file]["sha256"]
            for e, blob in zip(entries, blobs))
        gxs = [e.gx for e in entries]
        report(
            "restart states load + sha256 match manifest",
            sha_ok and len(entries) == len(manifest["states"])
            and bool(meta.get("reached_clear")),
            f"{len(entries)} rungs, gx {gxs}, verified clear="
            f"{meta.get('reached_clear')}")
    except Exception as e:
        report("restart states load + sha256 match manifest", False, repr(e))

    # 4b. PROVENANCE, not schema. The check above passes on any
    # well-formed ladder, including one another lane minted for another
    # level from another root — see CONFIG["campaign_level"]. This one
    # asks whose ladder it actually is.
    try:
        ok_p, notes = preflight_restart_ladders(base, CONFIG)
        report("restart states are THIS level's, from THIS root, and the "
               "trainer and gates read the SAME ladder",
               ok_p, "; ".join(notes))
    except Exception as e:
        report("restart states are THIS level's, from THIS root", False,
               repr(e))

    # 5. Probe command assembles against real paths.
    try:
        fake_ckpt = REPO / base["reinforce"]["kl_anchor_checkpoint"]
        cmd = build_probe_command(
            checkpoint=fake_ckpt, episodes=CONFIG["probe_episodes"],
            sticky=CONFIG["probe_sticky"], jitter=CONFIG["probe_jitter"],
            max_steps=CONFIG["probe_max_steps"],
            eval_seed=CONFIG["probe_seed"])
        missing_paths = [
            a for a in cmd
            if ("/" in a and not a.startswith("-") and not Path(a).exists())]
        report("probe command assembles", not missing_paths,
               " ".join(cmd) if not missing_paths
               else f"missing paths: {missing_paths}")
    except Exception as e:
        report("probe command assembles", False, repr(e))

    # 6. Manifest mirrors CONFIG + phases.
    man = build_manifest(base)
    report("manifest mirrors CONFIG + phases",
           man["config"] == CONFIG and len(man["phases"]) == len(PHASES))

    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The campaign loop
# ---------------------------------------------------------------------------


def _gate_met(phase: dict, *, base: dict, vloss_window: deque,
              phase_env_steps: float, last_probe: Optional[dict],
              ckpt_dir: Path, log_path: Path, probe_seed: int) -> bool:
    gate = phase["gate"]
    if gate == "budget":
        return phase_env_steps >= phase["budget_env_steps"]
    if gate == "critic_warmup":
        if phase_env_steps < CONFIG["gate_min_phase0_steps"]:
            return False
        cv = critic_cv(list(vloss_window))
        return cv < CONFIG["gate_critic_cv"]
    if gate == "det_local_clears":
        # Deterministic gate probe: greedy, sticky 0, jitter 0, FROM THE
        # TOP RUNG (x~2502) — N consecutive local bottleneck clears ==
        # clear_rate 1.0 over N episodes. Rung-local by design; the
        # entrance-to-flag path is phase 3's job.
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None:
            return False
        res = run_probe(checkpoint=ckpt,
                        episodes=CONFIG["gate_det_clears"], sticky=0.0,
                        jitter=0, eval_seed=probe_seed,
                        start_state=_top_rung_state())
        summ = probe_summary(res, CONFIG["bottleneck_survival_x"])
        _append_jsonl(log_path, {
            "type": "gate_probe", "phase": phase["idx"], "protocol": {
                "sticky": 0.0, "jitter": 0, "start": "rung_top",
                "episodes": CONFIG["gate_det_clears"]},
            **summ, "status": res.get("status")})
        return (res.get("status") == "ok"
                and summ["n_episodes"] >= CONFIG["gate_det_clears"]
                and summ["clear_rate"] >= 1.0)
    if gate == "sticky_clear":
        # Rung-local sticky gate: clears from the top rung under the
        # honest noise profile (sticky 0.25 + jitter). The entrance-based
        # honest probe is the campaign metric, NOT this gate — requiring
        # >80% honest entrance clears here would gate phase 2 on a bar
        # above the campaign's own end goal (attempt-4 harness defect).
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None:
            return False
        res = run_probe(checkpoint=ckpt,
                        episodes=CONFIG["probe_episodes"],
                        sticky=CONFIG["probe_sticky"],
                        jitter=CONFIG["probe_jitter"],
                        eval_seed=probe_seed,
                        start_state=_top_rung_state())
        summ = probe_summary(res, CONFIG["bottleneck_survival_x"])
        _append_jsonl(log_path, {
            "type": "gate_probe", "phase": phase["idx"], "protocol": {
                "sticky": CONFIG["probe_sticky"],
                "jitter": CONFIG["probe_jitter"], "start": "rung_top",
                "episodes": CONFIG["probe_episodes"]},
            **summ, "status": res.get("status")})
        return (res.get("status") == "ok"
                and summ["clear_rate"] > CONFIG["gate_sticky_clear"])
    if gate == "restart_at_entrance":
        tau = read_backward_tau(load_checkpoint_payload(ckpt_dir))
        return tau == 0
    raise ValueError(f"unknown gate {gate!r}")


def run_campaign(start_phase: int = 0) -> int:
    base = _load_yaml(REPO / CONFIG["base_profile"])
    run_dir = REPO / CONFIG["run_dir"]
    log_path = REPO / CONFIG["campaign_log"]
    ckpt_dir = checkpoint_dir_for(base)
    metrics_path = ckpt_dir / "metrics.jsonl"
    spi = steps_per_iter(base)
    run_dir.mkdir(parents=True, exist_ok=True)

    ok_ladder, ladder_notes = preflight_restart_ladders(base, CONFIG)
    if not ok_ladder:
        sys.stderr.write(
            "ABORT: restart-ladder preflight failed —\n  "
            + "\n  ".join(ladder_notes) + "\n")
        _append_jsonl(log_path, {"type": "abort",
                                 "reason": "restart_ladder_preflight",
                                 "notes": ladder_notes})
        return 2

    manifest = build_manifest(base)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _append_jsonl(log_path, {"type": "campaign_start", "config": CONFIG,
                             "phases": list(PHASES),
                             "steps_per_iter": spi,
                             "start_phase": start_phase})

    monitor = KillMonitor(CONFIG, steps_per_iter=spi)
    cum_env_steps = 0.0
    payload = load_checkpoint_payload(ckpt_dir) if ckpt_dir.exists() else None
    if payload is not None:
        cum_env_steps = float(payload.get("iter", 0)) * spi

    probe_idx = 0
    for phase in PHASES[start_phase:]:
        phase_profile = build_phase_profile(base, phase, cum_env_steps)
        prof_path = run_dir / "phase_configs" / f"phase_{phase['idx']}.yaml"
        _dump_yaml(phase_profile, prof_path)
        iters = max(1, int(phase["budget_env_steps"] // spi) + 1)
        phase_log = run_dir / f"phase_{phase['idx']}.log"
        _append_jsonl(log_path, {
            "type": "phase_start", "phase": phase["idx"],
            "name": phase["name"], "profile": str(prof_path),
            "iters": iters, "cum_env_steps": cum_env_steps,
            "entropy_coef": phase_profile["reinforce"]["entropy_coef"]})

        cmd = [sys.executable, str(REPO / "scripts" / "train_game.py"),
               "--profile", str(prof_path), "--iters", str(iters),
               "--no-supervise", "--strict-config"]
        with open(phase_log, "ab") as lf:
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf,
                                    stderr=subprocess.STDOUT)
        offset = metrics_path.stat().st_size if metrics_path.exists() else 0
        phase_start_steps: Optional[float] = None
        vloss_window: deque = deque(
            maxlen=CONFIG["gate_critic_window_iters"])
        last_probe: Optional[dict] = None
        next_probe_at = cum_env_steps + CONFIG["probe_every_env_steps"]
        started_monitor = False
        gate_passed = False

        def _abort(reason: str) -> int:
            _stop_trainer(proc)
            _append_jsonl(log_path, {"type": "abort", "phase": phase["idx"],
                                     "reason": reason,
                                     "env_steps": cum_env_steps})
            print(f"[campaign] ABORT: {reason}", flush=True)
            return 2

        while True:
            rows, offset = _tail_jsonl(metrics_path, offset)
            for row in rows:
                env_steps = (float(row.get("generation", 0)) + 1.0) * spi
                cum_env_steps = max(cum_env_steps, env_steps)
                if not started_monitor:
                    phase_start_steps = env_steps - spi
                    monitor.start_phase(
                        phase["idx"], phase_start_steps,
                        row.get("sil_clears_total"))
                    started_monitor = True
                if row.get("ppo_value_loss") is not None:
                    vloss_window.append(float(row["ppo_value_loss"]))
                reason = monitor.observe(row)
                if reason:
                    return _abort(reason)

            phase_env_steps = (
                cum_env_steps - phase_start_steps
                if phase_start_steps is not None else 0.0)

            # Scheduled honest probe every probe_every_env_steps.
            if cum_env_steps >= next_probe_at:
                next_probe_at += CONFIG["probe_every_env_steps"]
                ckpt = latest_checkpoint(ckpt_dir)
                if ckpt is not None:
                    probe_idx += 1
                    res = run_probe(
                        checkpoint=ckpt,
                        episodes=CONFIG["probe_episodes"],
                        sticky=CONFIG["probe_sticky"],
                        jitter=CONFIG["probe_jitter"],
                        eval_seed=CONFIG["probe_seed"] + probe_idx)
                    summ = probe_summary(
                        res, CONFIG["bottleneck_survival_x"])
                    last_probe = {**summ, "status": res.get("status")}
                    _append_jsonl(log_path, {
                        "type": "probe", "phase": phase["idx"],
                        "checkpoint": str(ckpt),
                        "env_steps": cum_env_steps, "protocol": {
                            "sticky": CONFIG["probe_sticky"],
                            "jitter": CONFIG["probe_jitter"],
                            "episodes": CONFIG["probe_episodes"],
                            "greedy": True},
                        **summ, "status": res.get("status")})
                    # Competence kill: an OK probe whose median max-x is
                    # below the floor means the policy lost even the
                    # frozen prior's early-hazard competence — the direct
                    # signature of anchor failure. No sustain window.
                    if (res.get("status") == "ok"
                            and summ["n_episodes"] > 0
                            and summ["median_max_x"]
                            < CONFIG["kill_probe_median_floor"]):
                        return _abort(
                            f"KILL probe median_max_x "
                            f"{summ['median_max_x']:.0f} < floor "
                            f"{CONFIG['kill_probe_median_floor']:.0f} "
                            f"in phase {phase['idx']}")

                # Audit-7 rung-progress kill. Armed in every phase whose
                # restarts are NOT entrance-pinned, because the backward
                # curriculum lives in the base profile and its cursor
                # advances from phase 0 — 1-2's reverse-walk gate passed
                # at its first check on a cursor the earlier phases had
                # already walked to 0. Measured on CUMULATIVE env-steps
                # for the same reason. Probe cadence only: this is the
                # one cycle slow enough to pay for a checkpoint load.
                if not phase_pins_entrance(phase):
                    reason = rung_progress_kill(
                        state=read_backward_state(
                            load_checkpoint_payload(ckpt_dir)),
                        curriculum_env_steps=cum_env_steps, cfg=CONFIG)
                    if reason:
                        return _abort(reason)

                # Gate check rides the probe cadence (cheap gates too —
                # they only read state the probe cycle already loads).
                if started_monitor and _gate_met(
                        phase, base=base, vloss_window=vloss_window,
                        phase_env_steps=phase_env_steps,
                        last_probe=last_probe, ckpt_dir=ckpt_dir,
                        log_path=log_path,
                        probe_seed=CONFIG["probe_seed"] + probe_idx):
                    gate_passed = True
                    _append_jsonl(log_path, {
                        "type": "gate_pass", "phase": phase["idx"],
                        "gate": phase["gate"],
                        "env_steps": cum_env_steps})
                    _stop_trainer(proc)
                    break

            if proc.poll() is not None:
                # Trainer exited on its own: budget exhausted or crash.
                rows, offset = _tail_jsonl(metrics_path, offset)
                for row in rows:
                    reason = monitor.observe(row)
                    if reason:
                        return _abort(reason)
                    cum_env_steps = max(
                        cum_env_steps,
                        (float(row.get("generation", 0)) + 1.0) * spi)
                if proc.returncode != 0:
                    return _abort(
                        f"trainer subprocess exited {proc.returncode} "
                        f"in phase {phase['idx']}")
                if phase["gate"] == "budget":
                    gate_passed = True
                break

            time.sleep(CONFIG["poll_secs"])

        if not gate_passed:
            # Budget-gated phases were handled above; a gated phase whose
            # trainer ran out of iters never earned its advance.
            if phase["gate"] == "budget":
                gate_passed = True
            else:
                return _abort(
                    f"phase {phase['idx']} ({phase['name']}) exhausted its "
                    f"{phase['budget_env_steps']:.0f}-step budget without "
                    f"meeting gate {phase['gate']!r}")

        _append_jsonl(log_path, {"type": "phase_complete",
                                 "phase": phase["idx"],
                                 "name": phase["name"],
                                 "env_steps": cum_env_steps})

    # Final honest probe for the record.
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is not None:
        res = run_probe(checkpoint=ckpt, episodes=CONFIG["probe_episodes"],
                        sticky=CONFIG["probe_sticky"],
                        jitter=CONFIG["probe_jitter"],
                        eval_seed=CONFIG["probe_seed"] + probe_idx + 1)
        summ = probe_summary(res, CONFIG["bottleneck_survival_x"])
        _append_jsonl(log_path, {"type": "final_probe",
                                 "checkpoint": str(ckpt), **summ,
                                 "status": res.get("status")})
    _append_jsonl(log_path, {"type": "campaign_complete",
                             "env_steps": cum_env_steps})
    print("[campaign] complete.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate config, anchor, restart states and the "
                         "probe command without training.")
    ap.add_argument("--start-phase", type=int, default=0,
                    help="Resume the schedule at this phase (checkpoints "
                         "carry the trainer state; default 0).")
    ap.add_argument("--campaign-config", default=None,
                    help="YAML overriding CONFIG (and optionally the whole "
                         "phase schedule) to retarget this controller at "
                         "another level package, e.g. "
                         "configs/campaign_1_3.yaml. Omit for the pinned "
                         "1-2 campaign.")
    args = ap.parse_args()
    if args.campaign_config:
        apply_campaign_config(REPO / args.campaign_config
                              if not Path(args.campaign_config).is_absolute()
                              else Path(args.campaign_config))
        print(f"[campaign] config: {args.campaign_config} -> "
              f"{CONFIG['campaign_name']} ({len(PHASES)} phases)", flush=True)
    if args.dry_run:
        return dry_run()
    if not 0 <= args.start_phase < len(PHASES):
        raise SystemExit(f"--start-phase must be 0..{len(PHASES) - 1}")
    return run_campaign(start_phase=args.start_phase)


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
