"""
Main training orchestrator.

Wires together:
- RustPool:            N parallel emulator instances (in-process rayon)
- PolicyNetwork:       shared CNN policy
- GeneticAlgorithm:    evolves a population of network weights
- CurriculumManager:   progressively introduces harder levels
- RewardFunction:      game-specific reward signals from RAM

Can be run standalone via CLI or driven from the GUI.

Baseline evaluation strategy: for each genome in the population, run one
episode on a single worker of the parallel pool. The other workers are used
to parallelize evaluation across genomes (evaluate up to `num_instances`
genomes in parallel). This is a simple but adequate v1; a proper batched
evaluator with asynchronous genome dispatch is a follow-up.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import queue as _queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.emulation.frame_utils import FrameStacker, TileFeatureStacker
from src.emulation.rust_pool_adapter import RustPool, StepResult


def _make_pool(env_spec: str, **kwargs):
    """Pool factory. `nes_core:NESEnvironment` is the only supported spec."""
    if not env_spec.startswith("nes_core"):
        raise ValueError(
            f"env_spec {env_spec!r} is not supported; only nes_core envs remain"
        )
    return RustPool(env_spec=env_spec, **kwargs)
from src.models.policy_network import PolicyNetwork, get_best_device
from src.models.rnd import RND
from src.training.bc_seed_cache import (
    bc_seed_cache_path as _bc_seed_cache_path_for,
    resolve_bc_demo_paths,
)
from src.training.behavior_cloning import build_dataset, pretrain, seed_population_from_weights
from src.training.checkpoint_manager import CheckpointManager
from src.training.exploration_controller import ExplorationController
from src.training.ppo_updater import PPOUpdater
from src.training.checkpointing import (
    archive_previous_run,
    find_latest_checkpoint as _find_latest_checkpoint,
    maybe_export_elite_to_coreml,
    rotate_old_checkpoints,
    save_checkpoint_atomic,
    save_winner,
)
from src.training.curriculum import Curriculum, CurriculumManager, CurriculumStage
from src.training.gae import gae as _gae
from src.training.genetic_algorithm import GeneticAlgorithm, Genome
from src.training.ppo import (
    batched_gae, demo_anchor_loss, fold_intrinsic_into_rewards, ppo_losses,
)
from src.training.redo import maybe_check_and_recycle as _redo_maybe_check
from src.training.redo import dose_fraction as _redo_dose_fraction
from src.training.redo import dose_ceiling_trips as _redo_dose_ceiling_trips
from src.training.redo import SELECT_BOTTOM_K as _REDO_SELECT_BOTTOM_K
from src.training.redo import SELECT_THRESHOLD as _REDO_SELECT_THRESHOLD
from src.training.redo import SELECT_MODES as _REDO_SELECT_MODES
from src.training.metrics_sink import MetricsSink
from src.training.notifications import StallNotifier
from src.utils.reward_functions import build_reward_function


log = logging.getLogger(__name__)

# Torch's intra-op pool size as the process started (one thread per
# P-core — 12 on the target M4 Max), captured at import time before any
# Trainer caps it. CPU-device trainers pin the pool to 1 thread (see
# __init__) because for the tiny tensors this trainer pushes on CPU the
# pool's sync overhead exceeds the math; the once-per-iter full-rollout
# RND pass is the one op big enough to profit from threads, so it
# temporarily restores this value.
_TORCH_DEFAULT_NUM_THREADS = torch.get_num_threads()

# Iterations an armed ReDo run may go without a single recycle before it
# is declared VOID (V30 registration, abort A2). Deliberately a module
# constant and not a config key — see the raise site in the PPO loop.
# RAISED 25 -> 40 for V31_REDO_SURGICAL_2026-08-27.md §4.3: 25 was
# calibrated for tau=0.25/0.50, which fire by iter 1 (24-iteration
# margin). At the surgical operating point tau=0.10 the measured first
# crossing is iter 16, so 25 would leave only 9 iterations of margin
# against seed-to-seed variation in the crossing iteration. 40 restores
# the same 24-iteration margin. Still a hardcoded module constant, not a
# config key (see the raise site below).
_REDO_ARM_DEADLINE_ITERS = 40

# In-run dose ceiling (V31_REDO_SURGICAL_2026-08-27.md §3, abort A4). The
# SAME numeral scripts/redo_arm_gate.py's post-hoc V6 uses, so the in-run
# early-abort and the verdict-time gate can never disagree; V6 remains
# authoritative at verdict time, this is an early-abort against a
# tail that is measured to drift DOWN across training, not a substitute.
_REDO_DOSE_CEILING = 0.25


def _safe_sample_from_logits(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per row from policy logits without crashing on
    NaN/Inf inputs. Returns (sampled_actions, chosen_log_probs, log_probs_all,
    n_bad_rows).

    The PPO loss occasionally produces a network update that drives
    weights to NaN/Inf — usually when an outsized advantage estimate
    couples with a policy near zero entropy. The downstream
    `torch.multinomial` then raises `RuntimeError: probability tensor
    contains either inf, nan or element < 0` and kills the training
    process mid-run. This helper sanitises logits with `nan_to_num` +
    a clamp, falls back to a uniform distribution for any row that's
    still pathological, and logs a warning so the divergence is
    visible without taking the run down.

    Sampling runs on CPU. On MPS `torch.multinomial` decomposes into
    ~10 serial primitive kernels (~0.83 ms/step, flat with batch —
    dispatch-bound, more than the CNN forward), the `bad_rows.any()`
    guard forces a device→host sync, and callers pull three tensors
    back with separate `.cpu()` transfers. Moving the tiny
    (rows, num_actions) logits device→host once folds all of that into
    a single small transfer and runs the sanitise / sample / gather on
    CPU (~0.008 ms). `.cpu()` is a no-op when logits already live on
    the CPU (tile / CPU-device runs), and the returned tensors are on
    CPU so the callers' downstream `.cpu()` calls become no-ops. The
    log-probs stay paired with the distribution actually sampled from
    (`chosen_lp == log_probs_all[row, sampled]`).
    """
    logits = logits.cpu()
    safe_logits = torch.nan_to_num(
        logits, nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(-50.0, 50.0)
    log_probs_all = F.log_softmax(safe_logits, dim=-1)
    probs = log_probs_all.exp()

    bad_rows = (~torch.isfinite(probs).all(dim=-1)) | (probs.sum(dim=-1) <= 0)
    n_bad_rows = int(bad_rows.sum())
    if n_bad_rows:
        n_actions = probs.size(-1)
        uniform_p = 1.0 / float(n_actions)
        probs[bad_rows] = uniform_p
        log_probs_all[bad_rows] = float(np.log(uniform_p))
        log.warning(
            "policy logits contained NaN/Inf — substituted uniform "
            "distribution for %d row(s). Network may have diverged.",
            n_bad_rows,
        )

    sampled = torch.multinomial(probs, 1).squeeze(-1)
    chosen_lp = log_probs_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
    # Returning the full log-probs tensor (in addition to the chosen
    # action's log-prob) lets callers implement sticky-action overrides
    # while keeping PPO's importance ratio correct: when sticky replaces
    # the sampled action with the previous one, we need the log-prob of
    # the OVERRIDDEN action under the CURRENT policy distribution, not
    # the sampled-action's log-prob. `n_bad_rows` lets callers accumulate
    # a per-generation NaN/Inf divergence count for metrics.jsonl instead
    # of the event being visible only in the text log.
    return sampled, chosen_lp, log_probs_all, n_bad_rows


# The 1-2 "gauntlet": the residual hard core of the level in global-x
# pixels. Every CGSA report scopes its noise figure to this window.
CGSA_GAUNTLET_GX: tuple[int, int] = (1600, 2200)


def cgsa_zone_summary(
    cg_stats: dict,
    gauntlet_gx: tuple[int, int] = CGSA_GAUNTLET_GX,
) -> dict[str, float]:
    """Summarise a CGSA curriculum's per-zone state for telemetry.

    Returns the frontier-only figures the run logs have always carried
    AND their uncensored counterparts (every tracked zone, welded or
    not), plus the welded fraction.

    The frontier-only averages are structurally misleading as a progress
    signal. A zone leaves the frontier exactly when the SPRT accepts it
    as locally sticky-robust at its annealed noise, so a curriculum that
    is *succeeding* retires its high-`p` zones and drives
    `frontier_avg_p` toward zero — the same trace a curriculum that
    never annealed anything produces. `avg_p_all` and
    `gauntlet_avg_p_all` do not censor the welds; `welded_frac` says how
    much of the population the frontier figure is ignoring. At the END OF
    the two archived 1-2 protocol runs (seed 1 iter 5975, seed 2 iter
    13695) the gauntlet reads 0.022 / 0.009 frontier-only and 0.165 /
    0.148 uncensored, with 64% / 73% of zones already welded. Earlier in
    those runs the two censorings coincided — at seed 1's it800 signpost
    no gauntlet zone had welded yet, so its 0.054 FAIL was uncensored.

    Pure and side-effect free so the archived `cgsa_stats.json` files
    can be re-scored offline with the exact code the run logged with.
    """
    lo, hi = gauntlet_gx
    tracked = [s for s in cg_stats.values() if "gx" in s]
    welded = [s for s in tracked if s["welded"]]
    frontier = [s for s in tracked if not s["welded"]]
    gaunt_all = [s for s in tracked if lo <= s.get("gx", -1) <= hi]
    gaunt_front = [s for s in frontier if lo <= s.get("gx", -1) <= hi]
    # Rate distribution over MEASURED zones (>=1 completed window) — the
    # leading indicator the v1/v2 runs lacked: tells apart "windows never
    # fill" from "rates below bar".
    measured = [s for s in tracked if s.get("windows", 0) > 0]
    rates = sorted(s["rate"] for s in measured)

    def _mean_p(zones: list) -> float:
        return float(np.mean([s["p"] for s in zones])) if zones else 0.0

    return {
        "cells": len(tracked),
        "welded": len(welded),
        "welded_frac": len(welded) / max(1, len(tracked)),
        "frontier_avg_p": _mean_p(frontier),
        "avg_p_all": _mean_p(tracked),
        "gauntlet_n": len(gaunt_front),
        "gauntlet_avg_p": _mean_p(gaunt_front),
        "gauntlet_n_all": len(gaunt_all),
        "gauntlet_avg_p_all": _mean_p(gaunt_all),
        "gx_max": max((s.get("gx", 0) for s in tracked), default=0),
        "measured": len(measured),
        "rate_p50": rates[len(rates) // 2] if rates else 0.0,
        "rate_p90": rates[int(len(rates) * 0.9)] if rates else 0.0,
        "zones_p_gt0": sum(1 for s in tracked if s["p"] > 0.0),
    }


def _select_winner_metric(
    in_training_rate: float,
    *,
    cold_rate: Optional[float] = None,
    bwd_snapshot: Optional[dict] = None,
) -> tuple[Optional[float], str]:
    """Pick the winner-retention metric for the non-ladder / non-consolidate
    branch of the retention block.

    Precedence: the backward-curriculum entrance rate (or pre-entrance
    suppression) keeps its deliberate re-keying; otherwise an honest
    cold-probe rate in scope (PLR mode's `last_cold_metrics`) keys the
    winner; only when neither exists does the RAW in-training clear_rate
    remain — loudly flagged, because it is inflated relative to the honest
    protocol (PR-MDP incident: winners/ retained a "best" at in-training
    0.25 while the honest cold probe read 0.0 at all nine sampled
    checkpoints). Negative cold rates are the probe's "no episodes scored"
    sentinel, not an honest measurement.

    Returns `(metric_value_or_None, metric_name)`; None means "do not
    save a winner this round".
    """
    if bwd_snapshot is not None:
        if bwd_snapshot["at_entrance"]:
            return float(bwd_snapshot["rate"]), "entrance_trailing_rate"
        return None, "entrance_trailing_rate"
    if cold_rate is not None and float(cold_rate) >= 0.0:
        return float(cold_rate), "cold_seq_clear_rate"
    log.warning(
        "[vanilla_ppo] WINNER METRIC = in-training clear_rate (INFLATED; "
        "not honest-probe verified): %.3f — no honest cold-probe result in "
        "scope for this mode",
        float(in_training_rate),
    )
    return float(in_training_rate), "clear_rate"


class StickyBoundary:
    """One-step sticky-action suppression across an episode boundary.

    Sticky training repeats the previous step's EXECUTED action with
    probability p. The vanilla-PPO rollout restarts a dead env IN PLACE
    mid-rollout (curriculum warm-state, stage-0 start bytes, CGSA /
    Go-Explore cell), and the carried action survives that restart: the
    first step of the fresh episode can execute the input the previous
    life died holding. The honest eval harness never does that — it
    gates the roll on `step > 0` per episode (scripts/eval_game.py) — so
    train and eval disagreed exactly at the boundary the restart makes,
    and every restarted episode started from a slightly different
    distribution than the one the gate measures.

    Opt-in via `reinforce.sticky_episode_boundary_reset`. Disabled it is
    a no-op: `override_rows` draws the same one `np.random.random(n)`
    the inline roll always drew, so existing lineages are bit-identical.
    """

    __slots__ = ("enabled", "_restarted")

    def __init__(self, num_envs: int, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._restarted = np.zeros(int(num_envs), dtype=bool)

    def mark_restart(self, i: int, prev_exec_action: np.ndarray) -> None:
        """Note that env `i` began a fresh episode on this step.

        Zeroes the carried action too, so nothing downstream can replay
        a pre-death input even if the suppression is consumed elsewhere.
        """
        if not self.enabled:
            return
        self._restarted[i] = True
        prev_exec_action[i] = 0

    def override_rows(
        self,
        sticky_p_env: np.ndarray,
        rng_vals: np.ndarray | None = None,
    ) -> np.ndarray:
        """Env rows whose action the sticky protocol overrides this step."""
        if rng_vals is None:
            rng_vals = np.random.random(len(sticky_p_env))
        rows = rng_vals < sticky_p_env
        if self.enabled:
            rows = rows & ~self._restarted
        return np.nonzero(rows)[0]

    def consume(self) -> None:
        """Drop the suppression once the roll that owed it has run."""
        if self.enabled:
            self._restarted[:] = False


# Transitions returned by `BackwardEntropyGuard.observe`. Returned rather
# than logged inside the guard so the decision stays pure and the caller
# owns the log line (and the iteration number it needs).
GUARD_ARM = "arm"
GUARD_DISARM = "disarm"


class BackwardEntropyGuard:
    """Trailing-entropy floor scoped to the backward start-state curriculum.

    THE RECEIPT (B4 v1, 2026-08-08). On the 1-1 reverse curriculum the
    policy entropy fell 0.19 -> 0.04 after ~iter 130, and the collapsed
    argmax then refused the final flag jump the sampled policy still
    took: cold-entrance greedy 0.02 against sampled 0.50/0.63, with the
    greedy episodes stalling at gx 3154-3157 — ten pixels short of the
    pole. A reverse curriculum manufactures exactly that. The deep rungs
    it starts on are nearly solved, so their advantage signal is small
    and sharply peaked and the policy sharpens there, long before tau
    reaches the entrance where the same policy still needs to explore.

    Neither existing mechanism covers it. The anti-collapse rollback
    (`SMB_ENTROPY_COLLAPSE_FRAC`) fires on entropy near ln(A) — a
    MELTING policy, the opposite failure. The adaptive `entropy_floor`
    controller does chase a low floor, but it is a whole-run knob: set
    high enough to stop the entrance-era collapse it also fights the
    early near-flag rungs, where sharpening is what the curriculum
    wants. This guard is scoped to backward mode, keyed on a TRAILING
    mean (one noisy minibatch entropy must never arm it), and hysteretic
    (it disarms on a real recovery above `floor * recover_mult`, not on
    the first iter that poked back over the line).

    Pure and side-effect free: it decides, the caller applies the
    multiplier to `entropy_coef` and writes the log line. Opt-in via
    ``reinforce.backward_curriculum.entropy_guard: {floor, boost}``; with
    the key absent no instance is built and the loop is byte-identical
    to a run without the feature.
    """

    __slots__ = ("floor", "boost", "trailing", "min_samples",
                 "recover_mult", "_window", "_armed", "_armed_hist",
                 "_arms")

    def __init__(
        self,
        floor: float,
        boost: float,
        *,
        trailing: int = 10,
        min_samples: int = 5,
        recover_mult: float = 1.25,
        armed_history: int = 50,
    ) -> None:
        floor = float(floor)
        boost = float(boost)
        recover_mult = float(recover_mult)
        trailing = int(trailing)
        min_samples = int(min_samples)
        armed_history = int(armed_history)
        if not floor > 0.0:
            raise ValueError(f"entropy_guard.floor must be > 0, got {floor}")
        if boost < 1.0:
            raise ValueError(
                f"entropy_guard.boost must be >= 1.0 (a boost below 1 would "
                f"SHARPEN a collapsing policy), got {boost}")
        if trailing < 1:
            raise ValueError(
                f"entropy_guard.trailing must be >= 1, got {trailing}")
        if min_samples < 1:
            raise ValueError(
                f"entropy_guard.min_samples must be >= 1, got {min_samples}")
        if recover_mult < 1.0:
            raise ValueError(
                f"entropy_guard.recover_mult must be >= 1.0 (below 1 the "
                f"disarm band sits under the arm band and the guard "
                f"flaps), got {recover_mult}")
        if armed_history < 1:
            raise ValueError(
                f"entropy_guard.armed_history must be >= 1, got "
                f"{armed_history}")
        self.floor = floor
        self.boost = boost
        self.trailing = trailing
        # A sample floor longer than the window can never be met.
        self.min_samples = min(min_samples, trailing)
        self.recover_mult = recover_mult
        self._window: deque = deque(maxlen=trailing)
        self._armed = False
        # Rolling armed/disarmed history — the kill criterion registered
        # for B5 reads "armed for more than half of 50 consecutive
        # iters", so the run log must be able to state that directly.
        self._armed_hist: deque = deque(maxlen=armed_history)
        self._arms = 0

    @classmethod
    def from_config(cls, cfg) -> Optional["BackwardEntropyGuard"]:
        """Build from an `entropy_guard` sub-block, or None when absent.

        None means "not configured", and the caller must then leave
        `entropy_coef` alone — that is the byte-identical default path.
        `enabled: false` is honored so a profile can keep the block
        documented while turning it off. A block that IS present but
        malformed raises: a pre-registered run must not spend hours on a
        knob that silently did nothing (this project's dead-knob bug
        class, see config_schema.py).
        """
        if not isinstance(cfg, dict) or not cfg:
            return None
        if not bool(cfg.get("enabled", True)):
            return None
        missing = [k for k in ("floor", "boost") if k not in cfg]
        if missing:
            raise ValueError(
                f"entropy_guard is configured but missing {missing}; it "
                f"needs both `floor` and `boost`")
        return cls(
            floor=float(cfg["floor"]),
            boost=float(cfg["boost"]),
            trailing=int(cfg.get("trailing", 10)),
            min_samples=int(cfg.get("min_samples", 5)),
            recover_mult=float(cfg.get("recover_mult", 1.25)),
            armed_history=int(cfg.get("armed_history", 50)),
        )

    # -- state --------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def multiplier(self) -> float:
        """What `entropy_coef` should be multiplied by right now."""
        return self.boost if self._armed else 1.0

    @property
    def samples(self) -> int:
        return len(self._window)

    @property
    def trailing_mean(self) -> float:
        """Mean entropy over the trailing window (0.0 before any sample).

        Only meaningful once `samples >= min_samples`; the arm test is
        gated on that, so the empty-window 0.0 can never arm the guard.
        """
        return (sum(self._window) / len(self._window)) if self._window else 0.0

    @property
    def recover_floor(self) -> float:
        """Trailing mean the policy must clear to disarm the guard."""
        return self.floor * self.recover_mult

    @property
    def arms(self) -> int:
        """How many times the guard has armed over the whole run."""
        return self._arms

    @property
    def armed_recent(self) -> int:
        """Armed iters inside the rolling history window."""
        return sum(self._armed_hist)

    @property
    def history_len(self) -> int:
        return len(self._armed_hist)

    @property
    def armed_frac(self) -> float:
        """Armed fraction over the rolling history (0.0 when empty)."""
        return (self.armed_recent / len(self._armed_hist)
                if self._armed_hist else 0.0)

    # -- the decision -------------------------------------------------

    def observe(self, entropy: float) -> Optional[str]:
        """Feed one iteration's policy entropy; return the transition.

        Returns `GUARD_ARM` on the iter the guard arms, `GUARD_DISARM` on
        the iter it releases, and None when nothing changed (including
        every iter it stays armed). Idempotent in the sense that only
        transitions are reported — the caller applies `multiplier` every
        iter regardless.
        """
        self._window.append(float(entropy))
        event: Optional[str] = None
        if len(self._window) >= self.min_samples:
            mean = self.trailing_mean
            if not self._armed and mean < self.floor:
                self._armed = True
                self._arms += 1
                event = GUARD_ARM
            elif self._armed and mean >= self.recover_floor:
                self._armed = False
                event = GUARD_DISARM
        self._armed_hist.append(self._armed)
        return event


class Trainer:
    """Top-level training loop."""

    def __init__(
        self,
        rom_path: str,
        game_profile: dict,
        num_instances: int = 16,
        population_size: int = 16,
        checkpoint_dir: str = "./checkpoints",
        # Halved from 4000. Atari-RL literature uses 1000-2000 steps; the
        # diminishing-returns curve is flat past 1000 for Zelda-scale games.
        max_episode_steps: int = 1000,
        # Optional in-process callback. When set, the trainer invokes
        # it with the list[StepResult] after every step_all so the GUI
        # (now in the same process) can paint live frames without any
        # IPC. Signature: (results: list[StepResult]) -> None.
        frame_sink=None,
        metrics_queue=None,
        reward_queue=None,
        audio_queue=None,
        # Optional queue onto which the narrator pushes caption events
        # (dict with worker_id / genome_name / kind / caption / ...).
        # The GUI polls this at ~20 Hz and renders the most recent
        # caption over each tile. Events are ephemeral; a full queue
        # drops rather than stalls.
        narrator_queue=None,
        env_spec: str = "nes_core:NESEnvironment",
        start_state_path: str | None = None,
        bc_demo_path: str | None = None,
        bc_epochs: int = 8,
        seed: int | None = None,
        device_override: str | None = None,
    ) -> None:
        self.rom_path = rom_path
        self.game_profile = game_profile
        self.num_instances = num_instances
        self.population_size = population_size
        # Per-game checkpoint directory. If the caller passes the
        # default `./checkpoints`, append the profile's slug as a
        # subdirectory so each game gets its own subtree (mario,
        # zelda, contra, etc. don't share artifacts). Explicit
        # overrides (test fixtures, debug runs) are honored verbatim.
        # See docs/proposals/unified_learning_thesis.md §5.
        from src.training.profile_utils import derive_checkpoint_dir
        self.checkpoint_dir = derive_checkpoint_dir(
            checkpoint_dir, game_profile.get("name"),
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Profile override wins. Lets zelda.yaml ship with a far larger
        # episode budget (hours of game time) without baking it into
        # the Trainer class default.
        self.max_episode_steps = int(
            game_profile.get("max_episode_steps", max_episode_steps)
        )
        # tracemalloc is OPT-IN (default OFF) because the 10-frame
        # call-stack capture on every Python allocation is a 30-50%
        # throughput hit. Enable with NES_EVOLVE_TRACEMALLOC=1 when
        # you specifically want per-source-line allocation tracking
        # for a leak hunt; `_log_gen_memory` below reports both the
        # absolute top-N allocation sites and the per-gen diff when
        # tracemalloc is active.
        import os as _os
        if _os.environ.get("NES_EVOLVE_TRACEMALLOC") == "1":
            try:
                import tracemalloc as _tm
                if not _tm.is_tracing():
                    _tm.start(10)
                    log.info("[trainer] tracemalloc started (10 frames/alloc)")
            except Exception as exc:
                log.debug("tracemalloc unavailable: %s", exc)
        self._last_tm_snapshot = None

        # macOS libmalloc arena release: tracemalloc confirmed the
        # Python layer is clean between generations (253 MB flat
        # gen 0 → gen 1), but RSS grows +6-8 GB per gen because macOS's
        # default malloc retains freed arenas instead of returning
        # pages to the OS. Bind the private malloc_zone_pressure_relief
        # C API via ctypes so we can explicitly ask the default zone to
        # drop retained pages at gen boundary. goal=0 means "release
        # everything you can". No-op on non-macOS platforms.
        self._malloc_zone_pressure_relief = None
        if _os.uname().sysname == "Darwin":
            try:
                import ctypes
                libc = ctypes.CDLL("libSystem.dylib", use_errno=False)
                libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
                libc.malloc_zone_pressure_relief.argtypes = [
                    ctypes.c_void_p, ctypes.c_size_t,
                ]

                def _relief() -> int:
                    # Pass NULL for the zone arg so pressure relief
                    # applies to ALL registered zones (default + numpy's
                    # custom allocator zone + torch's allocator zone +
                    # any other C extension's private zone). 0 goal =
                    # "release all that's releasable". Returns the total
                    # bytes actually returned to the kernel across all
                    # zones.
                    return int(libc.malloc_zone_pressure_relief(None, 0))

                self._malloc_zone_pressure_relief = _relief
                log.info("[trainer] macOS malloc_zone_pressure_relief bound (all zones)")
            except Exception as exc:
                log.warning(
                    "failed to bind malloc_zone_pressure_relief (arena "
                    "retention diagnostic unavailable): %s", exc,
                )

        # Memory safety: trajectory obs buffers are pre-allocated at
        # (num_instances, max_episode_steps, *obs_shape). At 16 workers
        # and max_episode_steps=937000 that's 423 GB of VSZ per buffer
        # in pixel mode, and TWO buffers coexist (pop-wide in
        # _run_one_generation + per-batch inside _evaluate_batch) — so
        # a naive cap of 8 GB still peaks at ~17 GB resident mid-gen.
        # Keep each buffer ≤ 2 GB so the two-buffer peak stays under 5 GB.
        #
        # The per-step size depends on encoder. We peek at the profile
        # here (before the rl_cfg pull at line ~300) so the clamp uses
        # the RIGHT obs size: pixel mode = 4×84×84×float16 (56448
        # bytes), tile mode = feature_dim × int8 (175 bytes for SMB).
        # Without this guard, every tile-mode run with max_episode_steps
        # > ~1188 was being silently clamped down by 322× over-estimate
        # of buffer size, costing learning signal at the trajectory tail
        # — caught while bug-hunting the persistent BC cache failure.
        _TRAJ_OBS_BUDGET_BYTES = 2 * 1024 ** 3
        _enc = (game_profile.get("reinforce") or {}).get("encoder", "")
        if str(_enc).startswith("smb_tile") or _enc == "tile":
            # Tile mode: int8 features, ~175 dim for SMB. Worst case 256
            # to leave headroom for future games with larger tile grids.
            _bytes_per_step = num_instances * 256 * 1
            _shape_desc = "feature_dim × int8"
        else:
            # Pixel mode: 4-channel grayscale stack at 84×84, float16
            # when preprocess_f16 is on (the worst case for memory).
            _bytes_per_step = num_instances * 4 * 84 * 84 * 2
            _shape_desc = "4×84×84 × float16"
        _safe_max = max(1024, _TRAJ_OBS_BUDGET_BYTES // _bytes_per_step)
        if self.max_episode_steps > _safe_max:
            log.warning(
                "max_episode_steps=%d requires %.1f GB per trajectory obs "
                "buffer (%d workers × %s). Clamping to %d to keep per-buffer "
                "memory ≤ %.0f GB; REINFORCE only uses the last "
                "reinforce_max_steps (default 1500) of each trajectory, so "
                "this does not change learning signal.",
                self.max_episode_steps,
                num_instances * self.max_episode_steps * _bytes_per_step / num_instances / 1e9,
                num_instances, _shape_desc, _safe_max,
                _TRAJ_OBS_BUDGET_BYTES / 1e9,
            )
            self.max_episode_steps = int(_safe_max)
        self.env_spec = env_spec
        # Validate start_state_path up front. A CONFIGURED-but-missing
        # start state is a hard startup error: the old behavior (warn +
        # fall back to cold boot) silently trained the title-screen
        # attract demo / the wrong level, wasting entire runs before
        # anyone noticed. Raising here — before any pool spawns — names
        # the bad path while the operator is still watching the launch;
        # the GUI surfaces trainer-thread exceptions, so the historical
        # "Training finished with no explanation" failure mode this
        # used to guard against no longer applies. A path that is None
        # or empty (nothing configured) still cold-boots below.
        if start_state_path:
            _ss = Path(start_state_path)
            if not _ss.exists():
                raise FileNotFoundError(
                    f"start_state_path={start_state_path!r} does not exist. "
                    "Refusing to start: falling back to cold boot would "
                    "silently train the title-screen demo / the wrong level "
                    "instead of the configured start state. Fix the path in "
                    "the profile or GUI (or capture the state with "
                    "scripts/capture_start_state.py), or remove "
                    "start_state_path entirely to cold-boot deliberately."
                )
            if not _ss.is_file():
                raise FileNotFoundError(
                    f"start_state_path={start_state_path!r} is not a regular "
                    "file. Refusing to start: point it at a save-state file, "
                    "or remove start_state_path entirely to cold-boot "
                    "deliberately."
                )
        if not start_state_path:
            # No start state => the emulator cold-boots to the title
            # screen. For these NES games the title screen auto-plays an
            # attract-mode DEMO that ignores controller input, so every
            # env runs the identical scripted sequence and the policy
            # gets zero learning signal. This silently wasted whole runs
            # (operMode never leaves 0; entropy pins at ln(num_actions)).
            log.warning(
                "NO start_state_path — training will cold-boot to the "
                "title screen, where the attract-mode demo IGNORES agent "
                "input. The policy cannot learn from this. Set "
                "start_state_path in the profile to a live-gameplay state.",
            )
        self.start_state_path = start_state_path
        # bc_demo_path: explicit ctor arg from GUI/CLI; the YAML fallback
        # is applied later once `rl_cfg` is defined (see below).
        self.bc_demo_path = bc_demo_path
        self.bc_epochs = bc_epochs
        self.seed = seed

        if seed is not None:
            import random as _random
            _random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
            log.info("Deterministic seed set: %d", seed)

        # ROM integrity check. Profiles can pin `expected_md5:
        # <hex>` in YAML; mismatched dumps silently shift RAM
        # addresses and break reward functions — historically the
        # single biggest source of "agent learned something weird"
        # confusion. Warn loudly (not fatal, since test harnesses
        # can run without a profile lock) but log the live MD5 so
        # the user can copy it into the profile.
        try:
            import nes_core as _nc
            if hasattr(_nc, "rom_info"):
                md5, mapper, _sub, is_nes20 = _nc.rom_info(self.rom_path)
                expected = game_profile.get("expected_md5")
                if expected and expected.lower() != md5.lower():
                    log.warning(
                        "ROM MD5 MISMATCH for %s: expected %s, got %s "
                        "(mapper=%d, nes2.0=%s). Reward functions may "
                        "operate on the wrong RAM layout.",
                        self.rom_path, expected, md5, mapper, is_nes20,
                    )
                elif not expected:
                    log.info(
                        "ROM %s loaded: md5=%s mapper=%d nes2.0=%s. Add "
                        "'expected_md5: %s' to the profile to lock.",
                        self.rom_path, md5, mapper, is_nes20, md5,
                    )
        except Exception as exc:
            log.debug("rom_info unavailable (%s) — skipping integrity check", exc)

        # Hybrid GA + policy-gradient knobs (overridable from the profile).
        rl_cfg = game_profile.get("reinforce", {})
        # YAML fallback for bc_demo_path. Done HERE (after rl_cfg
        # exists) so a profile can self-declare its demo without
        # requiring the GUI's file-picker or the CLI's --bc-demo
        # flag — canonical needs a demo seed to escape the "policy
        # never commits, can't learn from anything" trap. Explicit
        # ctor arg still wins.
        if not self.bc_demo_path:
            self.bc_demo_path = rl_cfg.get("bc_demo_path")
        self.reinforce_enabled: bool = rl_cfg.get("enabled", True)
        self.reinforce_top_k: int = rl_cfg.get("top_k", 3)
        self.reinforce_lr: float = rl_cfg.get("lr", 1e-4)
        self.reinforce_steps: int = rl_cfg.get("steps", 3)
        self.reinforce_gamma: float = rl_cfg.get("gamma", 0.99)
        self.reinforce_grad_clip: float = rl_cfg.get("grad_clip", 1.0)
        # Sticky-action probability. Each agent step, with this
        # probability, the emulator executes the previous step's action
        # instead of the policy's freshly-sampled one. Mirrors the
        # standard "sticky actions" mechanism from Machado et al. 2018
        # for Atari and gym-super-mario-bros's optional stochastic
        # action repeat.
        #
        # Why we need it: SMB jumps require holding A for ~30 NES
        # frames to reach max height. With frame_skip=4 and 8 actions,
        # the policy makes a fresh decision every 4 NES frames; if it
        # picks an A-action then a non-A-action on the next decision,
        # A is RELEASED 4 frames into the jump and Mario falls early.
        # At entropy ~1.5, the chance of picking A-actions for 7
        # consecutive decisions (needed for a full jump arc) is low,
        # so most jump attempts get cut short. Sticky actions
        # smooth this out by encouraging temporal coherence: once
        # the policy commits to an A-action, there's a 25% chance
        # the next 4-frame window keeps holding A even if the policy
        # tried to release.
        #
        # When the override fires, the recorded log_prob in
        # `log_probs_old` is the log-probability of the OVERRIDDEN
        # action under the current policy (not the sampled action),
        # so PPO's importance-ratio computation stays consistent
        # with what the env actually executed. 0.0 disables.
        self.sticky_action_prob: float = float(rl_cfg.get("sticky_action_prob", 0.0))
        # Sticky at an episode boundary. The vanilla-PPO rollout restarts
        # dead envs in place mid-rollout, and the carried action crosses
        # that restart: the fresh episode's first step can execute the
        # input the previous life died holding. Eval never does that (it
        # gates the roll on step > 0 per episode), so the training
        # distribution and the honest gate disagree exactly at the
        # boundary. Opt-in — off, the roll is bit-identical to every
        # lineage trained before this flag existed.
        self.sticky_episode_boundary_reset: bool = bool(
            rl_cfg.get("sticky_episode_boundary_reset", False)
        )
        # Live BC replay: when a genome achieves a successful (level-clear)
        # episode during training, capture its trajectory and periodically
        # retrain a fresh policy via supervised behavior cloning on the
        # accumulated success buffer. Solves the "lucky-clear-then-regress"
        # plateau where one genome touches the flag at gen 152 and the
        # population loses the trajectory by gen 180. With BC replay, the
        # successful trajectory persists in memory and gets re-imitated
        # every `bc_replay_every_gens` generations, anchoring the policy
        # to behaviors that actually clear the level.
        self.bc_replay_enabled: bool = bool(rl_cfg.get("bc_replay_enabled", False))
        self.bc_replay_every_gens: int = max(1, int(rl_cfg.get("bc_replay_every_gens", 20)))
        self.bc_replay_epochs: int = max(1, int(rl_cfg.get("bc_replay_epochs", 5)))
        self.bc_replay_max_buffer: int = max(1, int(rl_cfg.get("bc_replay_max_buffer", 16)))
        # How many of the most recent buffered trajectories actually feed
        # BC training. Buffer storage stays at max_buffer for archival
        # (a successful clear that took 8 h to discover should not be
        # discarded just because BC training prefers fresher data), but
        # `_run_bc_replay` trains only on the latest train_window entries.
        # Diagnosis 2026-05-12: training on all 16 buffer entries
        # aggregates clears from 16 different source genomes across the
        # run's history. The resulting action labels averaged ~uniform
        # across the 6-action space (14-18% each vs uniform=16.67%), so
        # BC loss plateaued at 1.68 ≈ ln(6)=1.79 (random baseline) and
        # the BC-injected genome failed to reproduce any single clear.
        # Restricting training to 3 recent trajectories gives BC a
        # coherent single-or-few-policy target it can actually fit.
        self.bc_replay_train_window: int = max(
            1, int(rl_cfg.get("bc_replay_train_window", 3))
        )
        # === DEMO-ANCHORED PPO (DQfD-style) ===
        # Unlike bc_replay (out-of-loop separate-net BC, which caps at the
        # demo-fit ceiling and never sees the reward gradient), the demo
        # anchor adds a supervised CE(+large-margin) term on a FIXED bank
        # of winning demo (obs, action) pairs INSIDE every PPO minibatch:
        # the anchor holds the policy on the demo manifold through a hard
        # seam while the reward gradient sculpts what BC alone cannot fit.
        # The coefficient decays linearly so PPO can eventually exceed the
        # demos. Tile mode only (the bank stores tile feature vectors).
        self.demo_anchor_enabled: bool = bool(
            rl_cfg.get("demo_anchor_enabled", False)
        )
        self.demo_anchor_paths: list = list(
            rl_cfg.get("demo_anchor_paths", []) or []
        )
        self.demo_anchor_coef0: float = float(
            rl_cfg.get("demo_anchor_coef", 1.0)
        )
        self.demo_anchor_final: float = float(
            rl_cfg.get("demo_anchor_coef_final", 0.02)
        )
        # Global iteration at which the anchor schedule STARTS. Lets a
        # mid-run bank swap re-anchor fresh (coef0 at the swap iter, not
        # back-dated to iter 0) while staying honest under global_it.
        self.demo_anchor_decay_start: int = int(
            rl_cfg.get("demo_anchor_decay_start", 0)
        )
        self.demo_anchor_decay_iters: int = max(
            1, int(rl_cfg.get("demo_anchor_decay_iters", 400))
        )
        self.demo_anchor_margin: float = float(
            rl_cfg.get("demo_anchor_margin", 0.0)
        )
        self.demo_anchor_mb: int = max(
            1, int(rl_cfg.get("demo_anchor_minibatch", 256))
        )
        self._demo_bank = None
        # In-memory buffer of (obs, actions, rewards, length) tuples from
        # successful episodes. Capped at bc_replay_max_buffer to bound
        # memory; new successes evict the oldest.
        #
        # Persisted to disk at `checkpoints/bc_success_cache.npz` after
        # every new capture so a successful clear is never lost across
        # restarts. On trainer init, if that cache exists, we re-hydrate
        # the in-memory buffer from it — turning the BC replay system
        # into a permanent "remember every clear we've ever seen"
        # anchor instead of a per-session-ephemeral one. Critical given
        # how rare clears are: a 1-in-200-episode success that took 8
        # hours of training to discover should not vanish on restart.
        self._bc_replay_buffer: list[tuple] = []
        # Track buffer size at the end of the previous gen so we can fire
        # BC replay IMMEDIATELY on the next gen if a new clear was
        # captured this gen — instead of waiting up to bc_replay_every_gens
        # for the modulo trigger. Modulo timing alone gives PPO 1..N gens
        # to drift the policy AWAY from the freshly-discovered clear
        # before BC anchors it; immediate anchoring eliminates that gap.
        self._last_bc_buffer_size: int = 0
        self._bc_success_cache_path: Path = self.checkpoint_dir / "bc_success_cache.npz"
        # NOTE: deferred — the load needs `self.num_actions` and
        # `self._obs_buffer_shape()` for schema validation, both of
        # which aren't set yet. Calling here and then a cache existing
        # on disk would AttributeError. Triggered later in __init__
        # once num_actions and the obs-shape helpers are available.
        # Number of episodes to roll for each genome per generation.
        # 1 (default) = original behavior. Higher values average out the
        # per-episode stochasticity from the policy's action sampling +
        # the GA selecting genomes on noisy single-episode scores. With
        # entropy ≈ 0.4 and 8 actions, a single episode's score can
        # easily 5x for the same weights — observed empirically: gen 10
        # best=582, gen 11 best=147, both running the SAME elite weights
        # (the freeze snapshot guarantees this). Mean over N episodes
        # bounds the variance to σ²/N. For SMB tile mode, 4 episodes
        # cuts noise to 1/4 at 4× wall-clock; reasonable trade.
        self.episodes_per_genome: int = max(1, int(rl_cfg.get("episodes_per_genome", 1)))
        # Coefficient on the value-loss term in PPO's combined loss.
        # The standard PPO paper uses 0.5 assuming normalized value
        # targets; our value targets are unnormalized GAE returns
        # which can span 100s of units, so the value loss dominates
        # the shared-trunk gradient and washes out the policy signal.
        # Drop to 0.1-0.25 for unnormalized targets.
        self.value_coef: float = float(rl_cfg.get("value_coef", 0.5))
        # Huber (smooth L1) value loss instead of MSE. Bounded gradient
        # for large value-target outliers — critical when returns are
        # unnormalized and span [0, 5000+] over a 1500-step trajectory.
        # MSE on those targets gives loss in the hundreds; Huber bounds
        # it to the 10s range and keeps the critic from steamrolling
        # the actor through the shared trunk.
        self.value_loss_kind: str = str(
            rl_cfg.get("value_loss", "mse")
        ).lower()
        if self.value_loss_kind not in ("mse", "huber", "smooth_l1"):
            raise ValueError(
                f"reinforce.value_loss must be 'mse' or 'huber'/'smooth_l1', "
                f"got {self.value_loss_kind!r}"
            )
        # GA-only warmup: skip the PPO gradient step for the first N
        # generations so the BC-imitated weights have time to spread
        # through the population via crossover/mutation before PPO's
        # noisy advantage signal starts pulling them toward whatever
        # the reward function thinks is good. Without this, the strong
        # BC seed gets washed out by the first few PPO updates and the
        # agent regresses to "random RL" performance — the "started
        # strong then tanked" pattern. Default 0 = disabled.
        self.warmup_gens_ga_only: int = int(rl_cfg.get("warmup_gens_ga_only", 0))
        # ReDo — Recycling Dormant neurons (Sokar et al. 2023). Registered:
        # docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md AMENDMENT 1
        # (B3 mechanism, B4 knobs, B6 armed-evidence log lines). Default
        # OFF: an absent block or redo_enabled: false leaves every
        # existing run bit-identical — the end-of-iteration hook returns
        # before touching any RNG stream. Scope is the feedforward tile
        # policy's hidden units (fc1/fc2) in the vanilla_ppo loop only;
        # heads and the RND predictor are never recycled. The cumulative
        # recycle counter is telemetry (B5 read #4), in-memory per run.
        self.redo_enabled: bool = bool(rl_cfg.get("redo_enabled", False))
        self.redo_tau: float = float(rl_cfg.get("redo_tau", 0.025))
        self.redo_check_every_iters: int = max(
            1, int(rl_cfg.get("redo_check_every_iters", 1))
        )
        self.redo_sample_batch: int = max(
            1, int(rl_cfg.get("redo_sample_batch", 4096))
        )
        self.redo_reset_optimizer_moments: bool = bool(
            rl_cfg.get("redo_reset_optimizer_moments", True)
        )
        # V32 additions (V32_REDO_BOTTOM_K_2026-08-28.md §2, §12.1) —
        # the selection rule. Default "threshold" is the pre-v32
        # behaviour exactly, so every existing config is untouched and a
        # redo-off run stays byte-identical. Under "bottom_k" the rank
        # rule recycles the k lowest-scoring fc2 units per check and
        # `redo_tau` is NOT READ; the v32 configs pin tau at its schema
        # default and say so, because a live numeral on a path that
        # never reads it is a dead knob.
        self.redo_mode: str = str(
            rl_cfg.get("redo_mode", _REDO_SELECT_THRESHOLD)
        )
        if self.redo_mode not in _REDO_SELECT_MODES:
            raise ValueError(
                f"[redo] redo_mode: {self.redo_mode!r} is not one of "
                f"{list(_REDO_SELECT_MODES)}"
            )
        self.redo_bottom_k: int = max(0, int(rl_cfg.get("redo_bottom_k", 0)))
        if (
            self.redo_enabled
            and self.redo_mode == _REDO_SELECT_BOTTOM_K
            and self.redo_bottom_k < 1
        ):
            raise ValueError(
                "[redo] redo_mode: bottom_k requires redo_bottom_k >= 1 "
                f"(got {self.redo_bottom_k}); a rank rule with k=0 "
                "recycles nothing and would look armed while being inert."
            )
        self._redo_cum_recycled: int = 0
        # V31 additions — trailing-window dose ceiling input (ALL checks,
        # including zero-recycle ones, per dose_ceiling_trips' contract),
        # plus the arming-floor telemetry the F1/F2/F3 conditions and the
        # run-manifest patch at end-of-run are computed from.
        self._redo_dose_ceiling_history: list[float] = []
        self._redo_fracs_on_fire: list[float] = []
        self._redo_recycle_events: int = 0
        self._redo_first_recycle_iter: Optional[int] = None
        self._redo_agree_log: list[float] = []
        self._redo_fc2_index_counts: dict[int, int] = {}
        # BC pretraining epoch count, threadable from YAML so different
        # games can use different schedules. Constructor still accepts
        # bc_epochs and YAML overrides it when set. Higher values mean
        # the BC seed is more entrenched (resistant to PPO drift) but
        # take longer at startup.
        self._bc_epochs_yaml: Optional[int] = rl_cfg.get("bc_epochs")
        if self._bc_epochs_yaml is not None:
            # YAML wins when present so per-game tuning works without
            # GUI changes. Constructor's bc_epochs (default 8) is the
            # fallback when the profile is silent.
            self.bc_epochs = int(self._bc_epochs_yaml)
        # Cap trajectory length per genome during REINFORCE to bound MPS
        # memory (each step = 4*84*84 uint8; on MPS the float cast triples).
        self.reinforce_max_steps: int = rl_cfg.get("max_steps_per_traj", 1500)
        # fp16 autocast — ~2x faster forward/backward on MPS with negligible
        # quality impact for this network size. Can disable via profile if
        # it causes numerical issues on a specific ROM.
        self.autocast_enabled: bool = rl_cfg.get("autocast_fp16", True)
        # Emit observations from the Rust pool as float16 already
        # normalized to [0, 1], skipping the per-batch MPS `.float().div_(255.0)`
        # kernel launch. Off by default — changes numerical semantics
        # (uint8 255-valued grid vs fp16-rounded normalized floats),
        # so existing training runs keep bit-identical observation math
        # unless the profile opts in.
        self.preprocess_f16: bool = rl_cfg.get("preprocess_f16", False)
        # Batched PPU rendering (ppu_neon Replace mode): skip-per-pixel
        # fast path driven by prev-frame clean-scanline history. Measured
        # +10-27% single-env / +2-4% parallel (2026-04-21 bench). Off by
        # default: games with mid-scanline raster tricks can see a
        # 1-frame-stale row on mispredict, so profiles opt in per-game
        # ("off" | "verify" | "replace"). Applied only in headless runs —
        # the GUI grid needs exact frames.
        self.batched_render: str = str(rl_cfg.get("batched_render", "off")).lower()
        # PPO clipping + entropy bonus. Vanilla REINFORCE collapses policies
        # onto single actions too fast (we saw the Zelda agent spam "down"
        # for 80% of steps). PPO's clipped surrogate limits per-step policy
        # drift; the entropy term keeps exploration alive. Set
        # ppo_clip_eps=0 to degrade to vanilla REINFORCE for A/B testing.
        self.ppo_clip_eps: float = rl_cfg.get("ppo_clip_eps", 0.2)
        self.entropy_coef: float = rl_cfg.get("entropy_coef", 0.01)
        # Adaptive entropy FLOOR (opt-in; 0 = off). The fixed coef can't
        # stop a policy from collapsing to near-deterministic under
        # sticky/jitter training — the 2026-07-19 1-1 calibration crashed
        # entropy 0.57 -> 0.05 and then died 0/50 under sticky eval,
        # because a start-locked deterministic policy is exactly what
        # sticky perturbations break. When last_entropy drops below
        # entropy_floor we raise the effective coef (toward
        # entropy_coef_max); when it recovers we decay back toward the
        # base. SAC-style automatic-temperature logic, simplified to a
        # per-iteration multiplicative controller.
        self.entropy_floor: float = float(rl_cfg.get("entropy_floor", 0.0))
        self._entropy_coef_base: float = self.entropy_coef
        self.entropy_coef_max: float = float(
            rl_cfg.get("entropy_coef_max", 0.05)
        )
        # torch.compile can lower per-step dispatch overhead on MPS; it's
        # opt-out because compile can fail on non-trivial graphs.
        self.compile_nets: bool = rl_cfg.get("torch_compile", True)
        # Vmap across genomes: parameter-stacked forward = single GPU
        # dispatch for all N genomes per step. 3-5x speedup when it works.
        # If the runtime trips on MPS ops, it silently falls back to the
        # per-genome loop (still has the batched transfer win).
        self.vmap_forward: bool = rl_cfg.get("vmap_forward", True)
        # CNN architecture knobs. Defaults preserve the historical
        # Nature-DQN encoder so existing checkpoints + benchmarks remain
        # apples-to-apples; switch to `impala` in the profile for the
        # bigger ResNet encoder.
        self.encoder_kind: str = str(rl_cfg.get("encoder", "nature_dqn"))
        # Tile mode: use a small MLP on RAM-decoded tiles instead of a
        # CNN on stacked pixels. Dramatically smaller policy parameter
        # count (14k vs 1.7M) makes GA mutation meaningful and PPO
        # gradients large enough to actually steer learning. Set
        # `reinforce.encoder: smb_tiles` (or another tile encoder name)
        # to enable. Tile mode also implies a different obs shape, so
        # the trainer's trajectory buffers, frame stacker, and PPO
        # preprocessing all branch on it.
        self._is_tile_mode: bool = self.encoder_kind in (
            "smb_tiles", "smb_tiles_pos",
        )
        # Opt-in recurrent (GRU) tile policy for the vanilla_ppo path —
        # the SMB-past-1-2 lever (memory over the trajectory). Only valid
        # with the tile encoder; the proven feedforward path is untouched
        # when this is off. Wired in _run_vanilla_ppo (rollout threads the
        # hidden state, the update replays sequences with truncated BPTT).
        self._recurrent: bool = self._resolve_recurrent(rl_cfg)
        # Envs per BPTT minibatch on the recurrent path. Each minibatch
        # replays this many full env trajectories through the GRU, so the
        # gradient batch is env_mb sequences. The per-sample
        # ppo_minibatch_size doesn't map to sequence minibatches (it
        # underflows to 1 env when rollout_steps > it), so recurrence gets
        # its own env-count knob.
        self.recurrent_env_minibatch: int = int(
            rl_cfg.get("recurrent_env_minibatch", 8)
        )
        self._tile_extractor = None
        self._tile_feature_dim: int = 0
        # The vanilla_ppo save-state curriculum is built around SMB's
        # area-byte progression ($075F world / $0760 area), so it ONLY
        # applies to Super Mario Bros. Other games (Contra, etc.) run
        # vanilla_ppo as plain single-stage PPO from their start state —
        # without this gate the capture/advance logic would read garbage
        # from their RAM at those addresses and fabricate bogus stages.
        # A generalized per-game curriculum is future work.
        # Name-gated (reward dispatch also substring-matches "mario", so
        # the name must keep it), but with an explicit off-switch:
        # `reinforce.smb_curriculum: false` runs single-stage from the
        # profile's start_state without the capture/advance machinery.
        # Used to warm-start directly at a specific area byte (e.g. the
        # 1-2 underground main) for a clean, curriculum-free measurement.
        self._smb_curriculum_active: bool = (
            "mario" in str(game_profile.get("name", "")).lower()
            and bool(rl_cfg.get("smb_curriculum", True))
        )
        # A fresh run (GUI "Resume" unticked / headless --no-resume) resets
        # the model weights but, historically, silently inherited whatever
        # SMB curriculum stage a prior run had reached on disk — a validated
        # confusion where a "fresh" run resumed at stage 2. Fresh runs now
        # start the curriculum at stage 0; set this True to restore the old
        # inherit-on-fresh behavior. Resume (non-fresh) runs always inherit
        # the saved curriculum regardless of this knob. The stage_NN.state
        # files are never deleted either way, so paused runs stay resumable.
        self._inherit_curriculum_on_fresh: bool = bool(
            rl_cfg.get("inherit_curriculum_on_fresh", False)
        )
        # Frame stacking for tile observations. yumouwei (2022) showed
        # empirically that stack=1 fails to clear SMB 1-1 while stack=4
        # succeeds — point-in-time RAM features lack temporal context
        # for jump timing and enemy avoidance. Reads `reinforce.tile_frame_stack`
        # with default 4 (the canonical yumouwei value). Set to 1 in
        # the YAML to disable stacking.
        self._tile_frame_stack: int = int(
            rl_cfg.get("tile_frame_stack", 4 if self._is_tile_mode else 1)
        )
        # Tile MLP width. The defaults (64/32) are tuned for a SINGLE level;
        # a multi-level generalist needs more capacity in the trunk (the
        # 32-d bottleneck squeezes two levels' policies into a mediocre
        # average — the measured multi-level collapse). Config-overridable so
        # single-level runs stay byte-identical.
        self._tile_hidden_dim: int = int(rl_cfg.get("tile_hidden_dim", 64))
        self._tile_trunk_dim: int = int(rl_cfg.get("tile_trunk_dim", 32))
        if self._is_tile_mode:
            from src.emulation.tile_observations import get_extractor
            self._tile_extractor = get_extractor(self.encoder_kind)
            base_dim = self._tile_extractor.feature_dim
            self._tile_feature_dim = base_dim * max(1, self._tile_frame_stack)
            log.info(
                "Tile mode active: encoder=%s, base_feature_dim=%d, "
                "frame_stack=%d, total_feature_dim=%d",
                self.encoder_kind, base_dim,
                self._tile_frame_stack, self._tile_feature_dim,
            )
        # Preserve elite diversity: when True, PPO updates only `best`
        # and leaves the other elites untouched. Default False keeps
        # the historical elite-clone behavior; tile-mode profiles set
        # True to prevent the post-PPO clone from collapsing
        # population diversity to a single policy.
        self.preserve_elite_diversity: bool = bool(
            rl_cfg.get("preserve_elite_diversity", False)
        )
        # Freeze a pre-PPO snapshot of the top-fitness genome and
        # inject it into the population so the next generation's GA
        # elitism keeps BOTH the post-PPO `best` (which PPO may have
        # regressed) AND the original pre-PPO weights. Without this,
        # a single bad PPO step on a fragile winning policy permanently
        # corrupts the elite — observed empirically: SMB tile mode
        # gen 151 hit fitness 7716 (cleared 1-1) then collapsed to
        # 1567 in gen 152 because PPO modified the winning weights
        # in-place and the GA had no fallback.
        self.freeze_pre_ppo_elite: bool = bool(
            rl_cfg.get("freeze_pre_ppo_elite", True)
        )
        # Pure-PPO mode short-circuits the GA: evolve() clones the
        # post-PPO elite into every slot, which silently overwrites
        # any state_dict the pre-PPO snapshot or BC-replay injection
        # parked on the weakest slot earlier in the gen. Force both
        # safety nets off and warn once if the user explicitly opted
        # in — the only honest pure-PPO recipe is "PPO is the only
        # update source" (uvipen/yumouwei). Without this guard, a
        # canonical-profile run reads `freeze_pre_ppo_elite: true`
        # and `preserve_elite_diversity: true` from YAML and silently
        # ignores both.
        _pure_ppo_active = (
            str(rl_cfg.get("trainer_mode", "vanilla_ppo")).lower() == "pure_ppo"
        )
        if _pure_ppo_active:
            if rl_cfg.get("freeze_pre_ppo_elite") is True:
                log.warning(
                    "freeze_pre_ppo_elite=true is a no-op under "
                    "trainer_mode=pure_ppo (evolve() clones the elite "
                    "into every slot, overwriting the snapshot). "
                    "Disabling."
                )
            if rl_cfg.get("preserve_elite_diversity") is True:
                log.warning(
                    "preserve_elite_diversity=true is a no-op under "
                    "trainer_mode=pure_ppo (evolve() flattens diversity "
                    "every generation). Disabling."
                )
            self.freeze_pre_ppo_elite = False
            self.preserve_elite_diversity = False
        # vanilla_ppo: the literature recipe (yumouwei/uvipen) —
        # ONE policy network, N parallel envs, batched GAE, K-epoch
        # PPO. NO GA, NO BC injection, NO population. The existing
        # ga_ppo / pure_ppo modes treat 30 envs as a "population of
        # policies" with mutation/crossover/elite-broadcast, which
        # mixes data from many policies into PPO's gradient and breaks
        # PPO's stable-policy-across-updates assumption. Empirical:
        # after 2+ days of incrementally fixing the hybrid we have
        # ~1 lucky-walk clear per ~80 gens on canonical / mario_tiles;
        # vanilla PPO is what actually converges on SMB 1-1 in the
        # published recipes. This mode reuses the same parallel-env
        # pool but as N rollout collectors for a single policy.
        # Default to vanilla_ppo — the supported, documented path that
        # actually converges. The legacy GA hybrid (ga_ppo / pure_ppo) is
        # now opt-in per profile and warned about loudly below, so
        # `make train GAME=<x>` never silently runs the plateaued path
        # (the two intentional GA recipes, mario_tiles + mario, pin
        # trainer_mode: ga_ppo explicitly).
        _trainer_mode = str(rl_cfg.get("trainer_mode", "vanilla_ppo")).lower()
        self.vanilla_ppo_mode: bool = (_trainer_mode == "vanilla_ppo")
        if self.redo_enabled and not self.vanilla_ppo_mode:
            # Mechanism-armed-but-inert guard: ReDo hooks the vanilla_ppo
            # iteration loop only. A treatment run in any other mode never
            # prints the `[redo] ENABLED` line, so the V2 preflight voids
            # it rather than reading a silent no-op as a result.
            log.warning(
                "[redo] configured but INERT: redo_enabled requires "
                "trainer_mode=vanilla_ppo (got %s) — no dormancy checks "
                "will run", _trainer_mode,
            )
        if _trainer_mode in ("ga_ppo", "pure_ppo"):
            log.warning(
                "trainer_mode=%s is the LEGACY GA path (plateaus on most "
                "games); the supported path is trainer_mode=vanilla_ppo. "
                "Set it explicitly in the game profile to silence this.",
                _trainer_mode,
            )
        # Rollout length per PPO iteration (per env). 512 × 30 envs =
        # 15,360 timesteps per update — comparable to OpenAI baselines'
        # default n_steps=128 × num_envs=8 = 1024 but scaled to our
        # 30-env pool for more gradient stability.
        self.rollout_steps: int = int(rl_cfg.get("rollout_steps", 512))
        # PPO minibatch size for the K-epoch SGD pass over the rollout.
        # 256 is a round number; full batch (rollout_steps * num_envs)
        # is too large for MPS memory on a 1.7M-param CNN.
        self.ppo_minibatch_size: int = int(rl_cfg.get("ppo_minibatch_size", 256))
        # GAE-λ. 0.95 is the canonical PPO value (Schulman+2017).
        self.gae_lambda: float = float(rl_cfg.get("gae_lambda", 0.95))
        # LayerNorm on the trunk FC. Always-on default — cheap and
        # measurably stabilizes the value head when reward magnitudes
        # span orders of magnitude (which they do in this project).
        # Loaded checkpoints that don't have the LN params init at γ=1,
        # β=0 (PyTorch default) and converge fast.
        self.use_layernorm: bool = bool(rl_cfg.get("layernorm", True))
        # Symlog reward transform (sign(r) * log(1+|r|)) applied to the
        # raw reward stream before GAE/PPO. Compresses the dynamic range
        # so a 200k-scale exploration reward doesn't drown a 1-scale
        # score reward in the value loss. Off by default — flipping it
        # changes existing fitness numerics. New runs on sparse-reward
        # games (Zelda, Metroid) should opt in.
        self.symlog_rewards: bool = bool(rl_cfg.get("symlog_rewards", False))
        # DrQ-style random-shift augmentation applied during the PPO
        # update only (not at action selection). Pads the 84x84 input
        # with replicate padding to 88x88, then random-crops back to
        # 84x84 each gradient step. Off by default — adds a small
        # amount of compute to the gradient step.
        self.drq_aug: bool = bool(rl_cfg.get("drq_aug", False))
        # DrQ pad pixels each side; total padded size = frame + 2*pad.
        self.drq_pad: int = int(rl_cfg.get("drq_pad", 4))
        # RND intrinsic motivation (sparse-reward exploration). Two
        # weights:
        #   * rnd_intrinsic_coef — bonus added to per-step extrinsic
        #     reward. Should be small relative to typical extrinsic
        #     reward magnitudes; the paper recommends ≈1.0 with reward
        #     normalization, lower without.
        #   * rnd_loss_coef — weight on the predictor MSE in the PPO
        #     total loss. Typical 1.0 (single shared optimizer) or
        #     anything that brings the predictor's grad magnitude into
        #     range with the policy/value losses.
        # Setting rnd_intrinsic_coef=0 disables RND entirely (no module
        # is built and no bonus is added).
        self.rnd_intrinsic_coef: float = float(rl_cfg.get("rnd_intrinsic_coef", 0.0))
        self.rnd_loss_coef: float = float(rl_cfg.get("rnd_loss_coef", 1.0))
        # Count-based frontier bonus on (world/level, 64px gx-bucket)
        # visitation: bonus = beta / sqrt(visits). Unlike RND, whose
        # novelty re-inflates on distribution shift, the count table is
        # cumulative across the whole run — trodden ground decays
        # permanently and only a genuine frontier keeps paying, so the
        # bonus is a cross-episode monotone gradient toward rare depth.
        # 0.0 (default) = off, no behavior change for other profiles.
        self._gx_count_beta: float = float(rl_cfg.get("gx_count_bonus_coef", 0.0))
        self._gx_counts: dict[int, int] = {}
        # Fraction of vanilla-PPO minibatches on which the RND *predictor*
        # is distilled toward the frozen target. 1.0 (default) trains it on
        # every minibatch — byte-for-byte today's behaviour. f in (0,1)
        # subsamples the predictor update on a DETERMINISTIC schedule
        # (minibatch index i carries RND grads iff i % round(1/f) == 0,
        # counted over processed minibatches across all K epochs); the
        # skipped minibatches backprop policy/value losses only, so the
        # predictor's Adam params stay frozen on those steps. The intrinsic
        # reward pass and update_normalization are unaffected — only the
        # distillation cadence changes — so the reward signal is identical
        # and this stays an opt-in learning A/B, not a default change. Only
        # the feedforward vanilla path honours it (GA + recurrent untouched).
        _rnd_pred_frac = float(rl_cfg.get("rnd_predictor_update_fraction", 1.0))
        if not (0.0 < _rnd_pred_frac <= 1.0):
            raise ValueError(
                "reinforce.rnd_predictor_update_fraction must be in (0, 1]; "
                f"got {_rnd_pred_frac!r}"
            )
        self.rnd_predictor_update_fraction: float = _rnd_pred_frac
        # Built lazily on first PPO step so the RND module lands on the
        # correct device once Trainer.run sets it up.
        self._rnd: Optional["RND"] = None
        # RND state (predictor weights + obs/reward running stats) read
        # from a resumed checkpoint. Applied at lazy-build time so resume
        # keeps the "what have I already explored" novelty memory instead
        # of re-rewarding every visited state as maximally novel.
        self._pending_rnd_state: Optional[dict] = None
        # Backward-curriculum cursor (tau + trailing window + entrance
        # counters) read from a resumed checkpoint. Applied when the
        # TauScheduler is built in _run_vanilla_ppo — the states dir has
        # to load first, since tau only means something against a tape.
        self._pending_backward_curriculum: Optional[dict] = None
        # Go-Explore archive — opt-in via reinforce.go_explore.enabled and
        # built per-run in _run_vanilla_ppo only when the SMB curriculum is
        # off (mutually exclusive). Stays None otherwise; exposed for tests.
        self._go_explore = None

        # Persistent PPO network and optimizer to maintain Adam's momentum
        # state across generations. Recreating them every gen resets the
        # learning rate adaptation and destroys "relearning" capability.
        self._ppo_net: Optional["PolicyNetwork"] = None
        self._ppo_optimizer: Optional[torch.optim.Optimizer] = None

        # Profile YAML can pin a specific device (e.g. `reinforce.device: cpu`)
        # which is the right call for tile mode — at 14k params, MPS kernel
        # launch overhead (~150-300 us per call) dwarfs the ~14k FLOPS of
        # actual compute, so CPU inference is faster despite the GPU being
        # available. The constructor `device_override` arg still wins (used
        # by tests + headless harnesses); falling back to YAML; falling back
        # to auto-detect in priority order.
        profile_device = rl_cfg.get("device")
        if device_override:
            self.device = torch.device(device_override)
        elif profile_device:
            self.device = torch.device(profile_device)
        else:
            self.device = get_best_device()
        log.info("Using device: %s", self.device)
        # CPU-device runs (tile mode) push tensors so small that torch's
        # default intra-op pool is a net loss: at mb=256×700 through the
        # tile MLP + TileRND, one minibatch measures 2399 us with 12
        # threads vs 1191 us with 1 (sweep is monotonic: 1T beats 2/4/8/
        # 12T), and the per-step rollout inference chain shows the same
        # pathology (240 us vs 92 us). Pin the pool to a single thread —
        # the update bucket roughly halves. MPS/pixel runs keep the
        # default; rayon's emulation threads are unaffected either way
        # (torch threads only spin during torch ops). The full-rollout
        # RND pass temporarily lifts the cap — see the update loop.
        if self.device.type == "cpu":
            torch.set_num_threads(1)
            log.info(
                "CPU device: capped torch intra-op threads to 1 "
                "(default %d) — small-tensor thread sync costs more "
                "than the math",
                _TORCH_DEFAULT_NUM_THREADS,
            )

        self.action_space = game_profile.get("action_space", [])
        self.num_actions = len(self.action_space)
        self._bitmask_table = self._build_bitmask_table()
        # uint8 lookup table for vectorized action->bitmask gather in the
        # rollout hot loop (avoids a per-env Python loop every step).
        self._bitmask_lut = np.array(self._bitmask_table, dtype=np.uint8)
        # Deferred from earlier in __init__: now that num_actions is
        # set and the _obs_buffer_shape helper is bound, it's safe to
        # rehydrate the persistent BC success cache (which validates
        # cache_version + num_actions + obs_shape before loading).
        self._load_bc_success_cache()
        # Reserved: when a future rewrite of _evaluate_batch actually
        # overlaps GPU inference for action_{t+1} with worker
        # emulation of action_t (which requires action_{t+1} to be
        # computed from obs_{t-1} — 1-step observation lag), this
        # flag will gate it. Today it's a no-op because inference and
        # emulation are still strictly serial. See
        # `docs/rust_nes_core.md` ("Future perf wins") for the design
        # sketch. Keeping the profile knob so downstream configs don't
        # have to change when the feature lands.
        # Async pipeline reads from `reinforce.async_pipeline` first
        # (where everything PPO-shaped now lives) and falls back to the
        # top-level `async_pipeline` for backwards compat with profiles
        # written before the section move.
        self.async_pipeline = bool(
            rl_cfg.get("async_pipeline", game_profile.get("async_pipeline", False))
        )
        # Panic isolation around per-worker rayon bodies. True (default)
        # is safe — worker panics don't crash the process. False is the
        # production fast path: skips `catch_unwind`, gains ~0.5-1%
        # throughput, but a panic anywhere in the worker pool can take
        # the trainer down. Opt out per-profile after the ROMs in use
        # have been validated.
        self.panic_isolation = bool(rl_cfg.get("panic_isolation", True))
        # Spectator pacing speed multiplier. Scales the pool-level
        # realtime pacing budget (frame_skip / (60 × m) wall-clock per
        # step): 1.0 = realtime, 2.0 = double speed. Default 1.0 = off
        # (knob not pushed to the pool at all — today's behavior).
        # Applied in _apply_pool_knobs only when a frame sink exists;
        # headless training must never pace. Pacing itself engages only
        # when some worker is paced (mixer solo mode today); the
        # multiplier just sets how fast paced stepping runs.
        self.pace_multiplier = float(rl_cfg.get("pace_multiplier", 1.0))
        # The multiplier actually applied to the live pool (1.0 until
        # _apply_pool_knobs pushes one). The adapter stamps it onto
        # StepResult.audio_rate so the mixer resampler pitch-ups
        # instead of overflowing its ring.
        self._active_pace_multiplier: float = 1.0
        # ASM bulk-step budget (CPU cycles per ASM invocation) for the
        # batch-safe mappers (MMC1: Zelda/Metroid/Mega Man 2...; UxROM:
        # Contra/Castlevania...). Default 1 = one instruction per
        # invocation — the shipped path, timing unchanged (parity-safe).
        # Raising it (ladder: 8, then 16) amortizes per-invocation ASM
        # setup over 2-5 instructions but coarsens tick granularity
        # visible to timed $2002 sprite-0 polling and mid-batch APU IRQ
        # service. Opt in per-profile via `reinforce.asm_bulk_cycles`
        # ONLY after the lockstep + Mesen-oracle + parity gate passes
        # for that game at that budget. Other mappers ignore the call.
        self.asm_bulk_cycles = int(rl_cfg.get("asm_bulk_cycles", 1))
        # Frames emulated between decisions. 16 is ~2.8× faster in
        # game-time throughput than 8 with no measurable learning loss on
        # slow NES games. Profiles can override: twitchy games (Contra
        # boss fights, frame-perfect platformers) may benefit from 8.
        self.frame_skip = int(game_profile.get("frame_skip", 16))
        if self.num_actions == 0:
            raise ValueError("Game profile must define a non-empty action_space")

        profile_name = str(game_profile.get("name", "")).lower()
        # SMB-only flag for the peak-x patching path in
        # _evaluate_batch. Detected from the profile name so any
        # Mario-derived profile (mario_tiles, mario_pixel, mario_smb1,
        # …) opts in. Reward function dispatch is independent — the
        # patching is just a no-op if x bytes happen to be game-
        # specific noise on a non-SMB ROM.
        self._is_smb_profile: bool = "mario" in profile_name or "smb" in profile_name
        rom_basename = Path(rom_path).stem.lower() if rom_path else ""
        if profile_name and rom_basename:
            profile_tokens = [t for t in profile_name.replace(".", " ").split() if t]
            if profile_tokens and not any(t in rom_basename for t in profile_tokens):
                log.warning(
                    "ROM/profile mismatch: profile=%r rom=%r. The reward "
                    "function reads game-specific RAM addresses; a mismatch "
                    "silently produces nonsense rewards (immediate 'death' on "
                    "step 1, identical fitness across the population). "
                    "Double-check the profile selection.",
                    profile_name, rom_basename,
                )

        self.reward_fn_factory = lambda: build_reward_function(game_profile)

        curriculum_spec = game_profile.get("curriculum", {})
        stages = self._build_curriculum_stages(curriculum_spec)
        # top_k_gate (optional): when set, the ga_ppo advance gate measures
        # only the top-k highest-fitness genomes' episodes instead of the
        # whole-population mean, which a mutated GA population can never push
        # above the 0.8 threshold (see CurriculumManager). Absent → legacy
        # whole-population gate, unchanged for every existing config.
        top_k_gate = (
            curriculum_spec.get("top_k_gate")
            if isinstance(curriculum_spec, dict)
            else None
        )
        if top_k_gate is not None:
            top_k_gate = int(top_k_gate)
        self.curriculum = CurriculumManager(stages=stages, top_k_gate=top_k_gate)

        ga_params = game_profile.get("ga_params", {})
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            network_factory=self._make_network,
            mutation_rate=ga_params.get("mutation_rate", 0.2),
            mutation_std=ga_params.get("mutation_std", 0.05),
            elite_fraction=ga_params.get("elite_fraction", 0.2),
            tournament_size=ga_params.get("tournament_size", 5),
            stale_gens_before_restart=int(ga_params.get("stale_gens_before_restart", 10)),
            restart_fraction=float(ga_params.get("restart_fraction", 0.5)),
            adaptive_mutation_scale=bool(ga_params.get("adaptive_mutation_scale", False)),
            # Pure-PPO mode: skip GA mutation/crossover, sync all
            # genomes to the post-PPO elite each generation. Reads
            # `reinforce.trainer_mode` so it's set at the same scope
            # as the rest of the PPO knobs (encoder, frame_stack, etc.).
            pure_ppo_mode=(
                str(rl_cfg.get("trainer_mode", "vanilla_ppo")).lower() == "pure_ppo"
            ),
            # Thread the trainer-level seed into the GA so tournament
            # selection, mutation noise, and crossover masks are
            # reproducible. Without this, --seed only set the global
            # python/numpy/torch RNGs but the GA's `random.sample` and
            # `torch.randn_like` would still pick up whatever state
            # MPS init / torch.compile had drifted into.
            seed=seed,
        )

        self.pool: Optional[RustPool] = None
        self._running = False

        # In-process GUI frame sink. Invoked with the list[StepResult]
        # after every step_all. None in headless mode.
        self._frame_sink = frame_sink
        # Set once if the frame sink ever raises, so we warn a single
        # time instead of either spamming or (the old behavior) silently
        # swallowing every failure and losing the GUI live-view with no
        # signal that anything broke.
        self._frame_sink_warned = False
        self._metrics_queue = metrics_queue
        self._reward_queue = reward_queue  # live reward-weight updates
        self._audio_queue = audio_queue    # GUI -> trainer: mixer mode/volume
        self._audio_mixer = None  # built lazily on first push
        # Background drainer for the audio queue — without it, mixer
        # commands (mute / solo / volume) only get applied at generation
        # boundaries, which can be minutes apart. See _start_audio_drainer.
        self._audio_drainer_stop: Optional[threading.Event] = None
        self._audio_drainer_thread: Optional[threading.Thread] = None
        # The audio-drainer thread records the latest requested mixer
        # mode here; the TRAINER thread applies the worker-pace change
        # (a pool mutation) at a safe point between step_all calls. The
        # pace call must NOT run on the drainer thread: set_worker_pace
        # mints &mut Worker, and step_all releases the GIL during its
        # rayon dispatch, so an off-thread pace change could alias a
        # worker mid-step (the Pool's documented "never overlap with
        # step_all" invariant). GUI-only path; None in headless.
        self._pending_pace_mode: Optional[str] = None
        self._narrator_queue = narrator_queue
        # Narrator instance: detects "first dungeon entry", combo kills,
        # deaths, new items, triforce — anything worth captioning. Seeded
        # from the trainer seed so reproducible runs produce the same
        # caption template choices on the same events.
        from src.training.narrator import Narrator
        self._narrator = Narrator(rng_seed=seed) if narrator_queue is not None else None
        # Per-generation timing accumulator. Every `timing_*_ms` key in
        # metrics.jsonl comes from this — a perf regression on any
        # section surfaces immediately.
        from src.training.timing import GenTimer
        self._gen_timer = GenTimer()
        # Count of policy-logit rows `_safe_sample_from_logits` had to
        # substitute a uniform distribution for (NaN/Inf — network
        # divergence) since the last emit. Snapshotted into
        # `nan_rows_this_gen` and reset alongside `_gen_timer` so the
        # divergence event is visible in metrics.jsonl, not just the
        # text log.
        self._nan_rows_this_gen = 0
        # Depth tracker — watches per-step RAM for new-all-time-deep
        # positions (dungeon+room for Zelda, world+level+x for Mario,
        # etc.). New records are captioned through the narrator and
        # appended to checkpoints/depth_memo.jsonl for post-run analysis
        # and future auto-curriculum promotion.
        from src.training.depth_tracker import DepthTracker
        self._depth_tracker = DepthTracker(
            game=game_profile.get("name", "unknown"),
            memo_path=self.checkpoint_dir / "depth_memo.jsonl",
            # The RAM reader follows the profile's DECLARED reward arm,
            # not its display name. A profile with no reward_id gets the
            # generic reader.
            depth_id=game_profile.get("reward_id"),
        )
        # Auto-curriculum: checkpoints of worker save-states at new
        # depth records land here so the NEXT training run can start
        # deeper. Populated by _on_new_depth_record when the env
        # backend supports save_state (nes_core does).
        self._auto_curriculum_dir = self.checkpoint_dir / "auto_curriculum"
        # Queue of (worker_id, depth_key, caption) pending save-state
        # RPCs. Drained at gen boundary (off the hot step path).
        self._pending_state_snapshots: list[tuple[int, tuple, str]] = []

        # Metrics fan-out: JSONL on disk + GUI queue + lazy TensorBoard.
        # The path is kept on `self` because `_archive_previous_run`
        # needs to stat + move it before the sink would have a chance
        # to truncate it.
        self._metrics_path = self.checkpoint_dir / "metrics.jsonl"
        self._metrics_sink = MetricsSink(
            metrics_path=self._metrics_path,
            tb_log_dir=self.checkpoint_dir / "tb",
            queue=self._metrics_queue,
            tb_enabled=game_profile.get("tensorboard", True),
        )

    def _archive_previous_run(self) -> None:
        archive_previous_run(
            checkpoint_dir=self.checkpoint_dir,
            metrics_path=self._metrics_path,
            game_profile=self.game_profile,
        )

    def _attach_run_file_logger(self) -> None:
        """Add a FileHandler so log.info / log.warning hit disk.

        Without this, `log.info(...)` lands only on stderr — the moment
        the launching terminal closes (or the GUI is run from a
        click-launched binary), the entire log stream is gone and
        post-hoc forensic analysis becomes impossible. The file lives
        at `<checkpoint_dir>/run.log` and is moved into the run-
        archive on the next Start by `_archive_previous_run`.
        """
        try:
            import logging as _logging
            run_log = self.checkpoint_dir / "run.log"
            # Truncate by opening in 'w' mode; the previous run's log
            # was just moved into the archive directory.
            with open(run_log, "w") as _f:
                _f.write("")
            # Detach any prior FileHandler we attached on a previous
            # Start (idempotent across stop/start cycles in one
            # process — without this the handlers stack up).
            root = _logging.getLogger()
            for h in list(root.handlers):
                if getattr(h, "_nes_run_handler", False):
                    root.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass
            handler = _logging.FileHandler(str(run_log), mode="a")
            handler.setLevel(_logging.INFO)
            handler.setFormatter(_logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            ))
            handler._nes_run_handler = True  # tag for the cleanup path above
            root.addHandler(handler)
        except Exception as exc:
            log.warning("[archive] failed to attach run.log handler: %s", exc)

    def _make_seed_content_probe(self):
        """Build `f(path) -> (world, area, gx) | None` that classifies a
        save-state blob by loading it into a scratch single-worker pool and
        reading RAM — the authoritative alternative to filename parsing for
        ladder seed binding. The scratch pool is created lazily on first
        probe and torn down by GC; results are cached per path. Any load
        failure (corrupt blob, mapper-state drift) returns None so the
        candidate is simply skipped."""
        from nes_core import Pool as _RawPool

        state = {"pool": None}
        cache: dict = {}

        def _probe(path):
            if path in cache:
                return cache[path]
            if state["pool"] is None:
                state["pool"] = _RawPool(
                    rom_path=self.rom_path, num_workers=1, frame_skip=4,
                )
            _p = state["pool"]
            try:
                blob = Path(path).read_bytes()
                _p.reset_all()
                _p.load_worker_state(0, blob)
                r = _p.step_all(np.zeros(1, dtype=np.uint8))
                ram = r[0][2]
                gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
                out = (int(ram[0x075F]), int(ram[0x0760]), gx)
                # Viability: a blob can be a DOOMED capture — saved on the
                # frame of death (player_state 0x06/0x0B, lives 0xFF), from
                # which every episode ends in ~3 steps regardless of policy
                # (stage_1_4_bw_x700.state is such a capture). Advance a few
                # frames and reject states that are dying or out of lives.
                for _ in range(8):
                    r = _p.step_all(np.zeros(1, dtype=np.uint8))
                ram = r[0][2]
                if int(ram[0x000E]) in (0x06, 0x0B) or int(ram[0x075A]) >= 0x80:
                    out = None
            except Exception:
                out = None
            cache[path] = out
            return out

        return _probe

    def _resolve_recurrent(self, rl_cfg: dict) -> bool:
        """True only for `recurrent: true` riding a tile encoder. A
        recurrent knob on a pixel encoder is DROPPED — loudly, because
        the policy class is the experimental variable in policy-class
        A/Bs and a silent coercion to feedforward would be discoverable
        only by autopsy (the run would train the CNN and leave no
        [policy] evidence the knob was ignored)."""
        requested = bool(rl_cfg.get("recurrent", False))
        if requested and not self._is_tile_mode:
            log.warning(
                "[policy] reinforce.recurrent=true IGNORED: encoder %r "
                "is not a tile encoder — the feedforward pixel policy "
                "will train", self.encoder_kind,
            )
        return requested and self._is_tile_mode

    def _make_network(self, num_actions: int | None = None):
        """Build the policy, wrapping it in the Phase-3 hazard veto if armed.

        Wrapped HERE rather than at each of the five construction sites,
        and outside the encoder branches so every network kind gets the
        same treatment. The wrapper delegates state_dict, so a checkpoint
        written under the veto still loads into a plain policy — the mask
        is an experiment arm, not a change to the artifact.

        Default off: with no `reinforce.hazard_mask` block this returns
        exactly what it always returned, which is what makes the Phase-3
        control arm a true control.
        """
        net = self._make_network_raw(num_actions)
        # getattr: characterization tests build bare Trainer.__new__
        # instances without game_profile; for them (and any caller
        # without a profile) the experiment-arm wrappers below are
        # simply off, which is also the correct default.
        rl_cfg = (getattr(self, "game_profile", None) or {}).get(
            "reinforce", {}) or {}

        commit_cfg = rl_cfg.get("commitment_options") or {}
        if commit_cfg.get("enabled"):
            # Action-commitment options (OPTIONS_PREREG_2026-08-22): the
            # policy's head is over (primitive, duration) pairs; the
            # commitment TIMER lives in the rollout loop, which holds the
            # chosen pair for k steps and marks held rows invalid so the
            # update only ever sees decisions. Tile, non-recurrent only —
            # the pseudo-recurrent eval surface reuses the recurrent
            # plumbing, so a genuinely recurrent base would collide.
            if (rl_cfg.get("hazard_mask") or {}).get("enabled"):
                raise ValueError("commitment_options and hazard_mask are "
                                 "mutually exclusive")
            if self._recurrent or not self._is_tile_mode:
                raise ValueError("commitment_options requires the "
                                 "non-recurrent tile policy")
            from src.training.commitment_policy import CommitmentPolicy
            durations = tuple(commit_cfg.get("durations", (1, 2, 4)))
            wrapped = CommitmentPolicy.from_flat_policy(
                net, trunk_dim=net.fc2.out_features,
                num_primitives=len(self.action_space),
                durations=durations)
            self._commit_durations = durations
            print(f"[commitment] ARMED durations={durations} "
                  f"pairs={wrapped.num_pairs}", flush=True)
            return wrapped

        cfg = rl_cfg.get("hazard_mask") or {}
        if not cfg.get("enabled"):
            return net
        if self._recurrent:
            # HazardMaskedPolicy masks inside forward_ac only; the
            # recurrent rollout/update call forward_ac_recurrent, which
            # __getattr__ delegates to the RAW net — the veto would
            # print ARMED and then never execute.
            raise ValueError(
                "hazard_mask requires the non-recurrent policy (the "
                "veto wraps forward_ac; the recurrent path bypasses it)"
            )
        from src.training.hazard_mask import HazardMask, HazardMaskedPolicy
        mask = HazardMask.from_checkpoint(
            cfg["checkpoint"],
            threshold=float(cfg.get("threshold", 0.90)),
            enabled=True)
        self._hazard_mask = mask
        print(f"[hazard-mask] ARMED threshold={mask.threshold} "
              f"from {cfg['checkpoint']}", flush=True)
        return HazardMaskedPolicy(net, mask)

    def _make_network_raw(self, num_actions: int | None = None):
        """Construct the policy network appropriate for the configured
        encoder. Pixel encoders (`nature_dqn`, `impala`) use the CNN-
        backed `PolicyNetwork`; tile encoders (`smb_tiles`) use the
        small MLP-backed `TilePolicyNetwork`. `num_actions` overrides the
        head width (default: the game's action space) — the kernel
        adversary's 2-action head shares everything else."""
        head_actions = self.num_actions if num_actions is None else int(num_actions)
        if self._is_tile_mode:
            if self._recurrent:
                from src.models.tile_policy import TileRecurrentPolicyNetwork
                net = TileRecurrentPolicyNetwork(
                    num_actions=head_actions,
                    feature_dim=self._tile_feature_dim,
                    hidden_dim=self._tile_hidden_dim,
                    gru_dim=self._tile_trunk_dim,
                )
            else:
                from src.models.tile_policy import TilePolicyNetwork
                net = TilePolicyNetwork(
                    num_actions=head_actions,
                    feature_dim=self._tile_feature_dim,
                    hidden_dim=self._tile_hidden_dim,
                    trunk_dim=self._tile_trunk_dim,
                )
            # Armed-evidence line: the policy CLASS is the experimental
            # variable in policy-class A/Bs, and the preflight
            # (scripts/experiment_preflight.py) needs positive log proof
            # of which class actually trained — a silently-ignored
            # `recurrent:` knob must not be discoverable only by autopsy.
            log.info(
                "[policy] class=%s params=%d (hidden=%d, trunk/gru=%d, "
                "features=%d, actions=%d)",
                type(net).__name__,
                sum(p.numel() for p in net.parameters()),
                self._tile_hidden_dim, self._tile_trunk_dim,
                self._tile_feature_dim, head_actions,
            )
            return net
        net = PolicyNetwork(
            num_actions=head_actions,
            encoder=self.encoder_kind,
            use_layernorm=self.use_layernorm,
        )
        # Same armed-evidence contract as the tile branch: every run
        # leaves positive log proof of the policy class that trained.
        log.info(
            "[policy] class=%s params=%d (encoder=%s, actions=%d)",
            type(net).__name__,
            sum(p.numel() for p in net.parameters()),
            self.encoder_kind, head_actions,
        )
        return net

    def _obs_buffer_shape(self, n: int, max_steps: int) -> tuple[int, ...]:
        """Per-genome trajectory buffer shape. Pixel mode stores the
        4-stacked frames as `(n, max_steps, 4, 84, 84)`; tile mode
        stores feature vectors as `(n, max_steps, feature_dim)`. Both
        are uint8/int8 to bound memory."""
        if self._is_tile_mode:
            return (n, max_steps, self._tile_feature_dim)
        return (n, max_steps, 4, 84, 84)

    def _obs_buffer_dtype(self):
        """Per-step obs storage dtype. Tile features are signed (have
        -1 enemy markers + signed velocities) so int8; pixels are
        unsigned uint8 (or float16 in the f16 preprocess fast path)."""
        if self._is_tile_mode:
            return np.int8
        return np.float16 if self.preprocess_f16 else np.uint8

    def _step_obs_shape(self) -> tuple[int, ...]:
        """Single-step obs shape, used for batch buffers."""
        if self._is_tile_mode:
            return (self._tile_feature_dim,)
        return (4, 84, 84)

    def _build_curriculum_stages(self, spec: dict) -> list[CurriculumStage]:
        """Parse the YAML curriculum spec. Two formats accepted:

        1. Legacy flat: `stage_1: ["1-1"]`
        2. Rich dict:   `stage_1: {levels: ["1-1"], start_state: "roms/mario_1-1.state.bin"}`
        """
        stages: list[CurriculumStage] = []
        stage_names = sorted(k for k in spec.keys() if k.startswith("stage_"))
        for name in stage_names:
            entry = spec[name]
            start_state = None
            if isinstance(entry, dict):
                levels = entry.get("levels", ["*"])
                start_state = entry.get("start_state")
            elif isinstance(entry, str):
                levels = [entry]
            else:
                levels = entry
            if isinstance(levels, str):
                levels = [levels]
            stages.append(
                CurriculumStage(
                    name=name,
                    levels=levels,
                    advance_threshold=0.8,
                    min_episodes=50,
                    start_state_path=start_state,
                )
            )
        if not stages:
            stages.append(
                CurriculumStage(
                    name="default", levels=["full_game"], advance_threshold=0.8
                )
            )
        return stages

    def _action_to_bitmask(self, action_idx: int) -> int:
        # Precomputed table indexed by action_idx. Previously this method
        # re-imported button constants, rebuilt a name→bit dict, and
        # re-walked the action-space list on every call — 16× T times per
        # generation of dict/list churn in the hot path. The table is
        # built once in _build_bitmask_table() during __init__ and never
        # changes after that.
        return self._bitmask_table[action_idx]

    def _apply_sampled_actions(
        self,
        active: list[int],
        sampled_cpu: np.ndarray,
        lp_cpu: np.ndarray,
        log_probs_all_cpu: np.ndarray,
        actions: list[int],
        log_probs_old: list[float],
        last_action_per_genome: list[int],
        step: int,
    ) -> None:
        """Write sampled (or sticky-overridden) actions into per-genome
        slots, recording the corresponding log-prob under the current
        policy.

        Sticky-action override: with probability `sticky_action_prob`,
        replace the freshly sampled action with the previous step's
        executed action. Smooths the policy's action sequences so jumps
        get held for full duration instead of being released mid-arc by
        random action transitions. Skipped on step 0 because there's no
        meaningful "last action" yet (we'd just stick to NOOP).
        """
        sticky = self.sticky_action_prob
        roll_sticky = sticky > 0 and step > 0
        # Per-genome RNG draw. Originally argued this was "unfair" (genome
        # A free while genome B forced to stick), and the fix landed in
        # cb509b0. Reverted because the run regressed: depth pinned at
        # 1887 for 32 gens with the synced version, vs prior runs
        # reaching 2596+ by gen 5. Theory: in GA selection,
        # action-stickiness variance ACROSS genomes is exploration —
        # outlier-lucky genomes occasionally breakthrough walls and
        # become elites. Synchronizing it removed that source of
        # behavioral diversity and the population converged on safe
        # strategies. The "fairness" framing was correct in isolation
        # but missed the GA dynamics that actually use the variance.
        rng_vals = (
            np.random.random(len(active)) if roll_sticky else None
        )
        for batch_idx, genome_idx in enumerate(active):
            if roll_sticky and rng_vals[batch_idx] < sticky:
                # Override with last-executed action; record its
                # log-prob under the current policy distribution so
                # PPO's importance ratio stays consistent with what
                # the env actually executes.
                a = last_action_per_genome[genome_idx]
                # Clamp at ~2e-6 probability: a near-deterministic policy
                # gives a stuck action a log-prob of -30..-46, and an
                # unclamped value explodes the PPO ratio into NaN. Same
                # floor the vanilla sticky path uses; this GA/ga_ppo path
                # ships live via mario_tiles.yaml (sticky 0.5).
                lp = max(float(log_probs_all_cpu[batch_idx, a]), -13.0)
            else:
                a = int(sampled_cpu[batch_idx])
                lp = float(lp_cpu[batch_idx])
            actions[genome_idx] = a
            log_probs_old[genome_idx] = lp
            last_action_per_genome[genome_idx] = a

    def _build_bitmask_table(self) -> tuple[int, ...]:
        # Conversion lives in profile_utils so the trainer hot path and
        # the headless eval/launch scripts share exactly one mapping.
        # The string-entry / unknown-button guards (this exact bug bit
        # the canonical profile in commit d76c4ab) moved there too.
        from src.training.profile_utils import action_space_to_bitmasks
        table = action_space_to_bitmasks(self.action_space)
        commit_cfg = ((self.game_profile.get("reinforce", {}) or {})
                      .get("commitment_options") or {})
        if commit_cfg.get("enabled"):
            # Pair index -> its primitive's bitmask, pair-major order
            # matching CommitmentPolicy.pair_of. The pool sees only
            # primitives; durations exist purely in the policy/rollout.
            durations = tuple(commit_cfg.get("durations", (1, 2, 4)))
            table = tuple(table[p_ // len(durations)]
                          for p_ in range(len(table) * len(durations)))
        return table


    def run(
        self,
        num_generations: int = 1000,
        resume_from: str | None = None,
        fresh_start: bool = False,
    ) -> None:
        """Main training loop. Blocks until complete or stopped.

        If resume_from points to an existing GA checkpoint, load it and
        continue from that generation; otherwise start fresh.

        `fresh_start` is the explicit from-scratch signal for the
        vanilla_ppo auto-resume (GUI "Resume" checkbox unticked / headless
        `--no-resume`): when True the vanilla_ppo_iter_*.pt scan is
        skipped. Default False preserves today's auto-resume behavior for
        callers that pass no signal.
        """
        import time as _time
        _t0 = _time.monotonic()
        def _stage(label: str) -> None:
            nonlocal _t0
            now = _time.monotonic()
            log.info("[startup] %s: %.2fs", label, now - _t0)
            _t0 = now

        self._running = True
        # Auto-archive previous run before truncating the metrics log.
        # Without this, every Start silently nukes the prior session's
        # data (we lost a 466-gen overnight run this way). Snapshot
        # everything that could be useful for post-hoc analysis into
        # `runs/<timestamp>/` — metrics, curriculum, BC success cache,
        # the latest checkpoint snapshot, and the current YAML profile
        # so we can later answer "what config produced these numbers".
        self._archive_previous_run()
        self._metrics_sink.truncate()
        # Persistent Python log: every log.info / log.warning that
        # would have only hit stderr now ALSO lands in
        # `<checkpoint_dir>/run.log` for forensic analysis after a
        # crash. The file is truncated per run start (matching the
        # metrics log lifecycle); the auto-archive above moved the
        # previous one into the runs/<ts>/ snapshot.
        self._attach_run_file_logger()
        _stage("metrics log reset + previous run archived")

        # Spin up the audio mixer (emits song changes derived from per-step
        # RAM snapshots; no-op when sounddevice is missing or mode=mute).
        try:
            from src.audio.ram_music import AudioMixer
            # Size the mixer to the worker count.
            self._audio_mixer = AudioMixer(
                num_instances=self._ga_batch_size(),
                game=self.game_profile.get("name", "unknown"),
                audio_root="audio",
            )
            self._audio_mixer.start()
        except Exception as exc:
            log.info("audio mixer unavailable: %s", exc)
            self._audio_mixer = None
        _stage("audio mixer built")

        # Start the audio-command drainer. Polls the GUI->trainer mixer
        # queue on its own thread so mute/solo/volume changes apply
        # within ~50 ms instead of waiting for the next generation
        # boundary (which can be minutes).
        self._start_audio_drainer()

        _stage("pre-GA")
        if resume_from and Path(resume_from).exists():
            self.ga.load_checkpoint(resume_from)
            # Architecture sanity-check: load the first genome's weights
            # into a freshly-built network. If the profile's
            # action_space length (and therefore the policy's `fc2`
            # output dim) changed since the checkpoint was written, this
            # raises a useful error here at the start instead of mid
            # _evaluate_batch (where the traceback is harder to diagnose).
            if self.ga.population:
                probe = self._make_network()
                # strict=False lets us absorb backward-compat gaps:
                # when PPO+GAE added a value_head to PolicyNetwork,
                # older checkpoints (pre-2026-04-21) have no
                # value_head.weight / bias in their state_dict. The
                # missing tensors stay at their fresh-init random
                # values; the critic relearns them from scratch on
                # the first few PPO updates while the actor trunk
                # carries forward from the checkpoint.
                #
                # Genuine mismatches (action_space length changed —
                # actor head out-dim differs) still surface as
                # "size mismatch" errors which strict=False does NOT
                # silence, so the safety net is preserved.
                try:
                    probe.load_state_dict(
                        self.ga.population[0].state_dict, strict=False
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Checkpoint {resume_from} does not match the current "
                        f"profile's network architecture (likely action_space "
                        f"length changed). Either restore the original profile "
                        f"or start a fresh run. Underlying error: {exc}"
                    ) from exc
            curr_path = self.checkpoint_dir / "curriculum.json"
            if curr_path.exists():
                try:
                    with open(curr_path, "r") as f:
                        self.curriculum.load_state_dict(json.load(f))
                except Exception as exc:
                    log.warning("curriculum state unreadable, starting fresh: %s", exc)
            log.info("Resumed from %s at generation %d", resume_from, self.ga.generation)
        else:
            self.ga.initialize()
            # Fresh run: if a demo file was provided, pre-train a network
            # on it and seed every genome in the population from those
            # weights (plus small per-genome noise for diversity).
            if self.bc_demo_path:
                self._behavior_clone_seed()
        _stage("GA init / resume")

        self.pool = _make_pool(
            env_spec=self.env_spec,
            rom_path=self.rom_path,
            num_workers=self.num_instances,
            frame_skip=self.frame_skip,
            start_state_path=self.start_state_path,
            seed=self.seed,
        )
        self.pool.start()
        self._apply_pool_knobs()
        _stage("pool.start (workers spawned)")

        # Poke every worker with a reset so they boot their emulator now.
        # Otherwise workers sit idle until the first generation's step loop,
        # making the grid look frozen on first render.
        self.pool.reset_all()
        _stage("reset_all (first frames published)")
        log.info("[startup] ready — beginning training")

        # Vanilla PPO dispatch: bypass the GA loop entirely. Single
        # policy network, N parallel envs as rollout collectors,
        # batched GAE, K-epoch PPO update. Matches yumouwei/uvipen's
        # literature recipe (the only PPO setup that empirically
        # converges on SMB 1-1 in our compute budget).
        if self.vanilla_ppo_mode:
            # try/finally so a Stop, an exhausted iter budget, or an
            # uncaught exception all still release the pool / audio
            # thread / TB writer. Without it the GUI leaked a full pool
            # + a live daemon thread on every Start→Stop→Start.
            try:
                self._run_vanilla_ppo(
                    num_iters=num_generations, fresh_start=fresh_start,
                )
            finally:
                self._teardown()
            return

        try:
            import gc as _gc
            for gen in range(num_generations):
                if not self._running:
                    break
                self._run_one_generation(gen)
                # Use the GA's cumulative generation so resumed runs don't
                # overwrite earlier checkpoints at the same loop index.
                if gen % 10 == 0:
                    self._save_checkpoint(self.ga.generation)
                # Force cycle collection between generations so Python
                # objects captured in refcount cycles (PyTorch computation
                # graphs, closures over tensors, traceback retention in
                # handled exceptions) release the large native buffers
                # they hold alive. Measured during the overnight-OOM
                # debug session 2026-04-21: without this the step loop
                # accumulates ~6500 cyclic objects per second and RSS
                # grows ~800 MB per 5 s even after the trajectory obs
                # buffer is clamped.
                _freed = _gc.collect()
                if _freed > 50:
                    log.debug("gen %d: gc.collect freed %d cycles", gen, _freed)
                # macOS arena relief across all registered zones. Log
                # unconditionally — a zero return is itself data (means
                # either nothing retained, OR the memory lives outside
                # the allocator zones entirely, which rules out libmalloc
                # retention as the RSS-growth cause).
                if self._malloc_zone_pressure_relief is not None:
                    try:
                        _released = self._malloc_zone_pressure_relief()
                        log.info(
                            "gen %d: malloc_zone_pressure_relief returned %d bytes (%d MB)",
                            gen, _released, _released // (1024 * 1024),
                        )
                    except Exception as exc:
                        log.warning("malloc_zone_pressure_relief failed: %s", exc)
                self._log_gen_memory(gen, gc_freed=_freed)
        finally:
            import gc as _gc
            self._stop_audio_drainer()
            if self.pool:
                self.pool.shutdown()
            self.pool = None
            # Drop the GUI callback so the AppController ↔ Trainer cycle
            # (training_thread target → Trainer → _frame_sink bound
            # method → AppController → training_thread) breaks without
            # waiting for the cycle collector. This single reference
            # chain kept ~17 GB of pool buffers resident between Stop
            # and the next Start.
            self._frame_sink = None
            if self._audio_mixer is not None:
                self._audio_mixer.stop()
            # Flush + close the TensorBoard writer if one was opened —
            # without it the last few generations of scalars buffered
            # in the writer never hit disk and the TB UI shows a
            # truncated tail.
            self._metrics_sink.close()
            self._save_checkpoint(self.ga.generation)
            # Final cycle sweep so any post-shutdown refs the caller
            # clears next (pool=None already done above) actually
            # release their native buffers before this thread exits.
            _gc.collect()

    def _log_gen_memory(self, gen: int, gc_freed: int = 0) -> None:
        """Emit a structured memory snapshot so leak sources are visible
        per generation. RSS from `ps`, Python heap from stdlib, torch MPS
        from the allocator, and a live-numpy-array scan that aggregates
        size by shape prefix so the largest retained buffers are obvious
        in the log.

        If tracemalloc is running (enable with NES_EVOLVE_TRACEMALLOC=1
        env var), also emits the top-5 Python allocation sites by total
        bytes. tracemalloc costs ~10-20% throughput so it's opt-in. It
        catches what gc.get_objects misses — notably bytes objects
        returned from PyO3 (Rust Vec<u8> → Python bytes) which aren't
        cycle-tracked and therefore invisible to the numpy scan below.

        Best-effort: any probe that fails is silently skipped.
        """
        import os as _os
        import subprocess as _sp
        try:
            rss_mb = -1
            r = _sp.run(
                ["ps", "-p", str(_os.getpid()), "-o", "rss="],
                capture_output=True, text=True, timeout=1.0,
            )
            if r.stdout.strip():
                rss_mb = int(r.stdout.strip()) // 1024
        except Exception:
            rss_mb = -1
        try:
            import sys as _sys
            py_blocks = _sys.getallocatedblocks()
        except Exception:
            py_blocks = -1
        mps_mb = -1
        mps_driver_mb = -1
        try:
            if torch.backends.mps.is_available():
                mps_mb = torch.mps.current_allocated_memory() // (1024 * 1024)
                if hasattr(torch.mps, "driver_allocated_memory"):
                    mps_driver_mb = (
                        torch.mps.driver_allocated_memory() // (1024 * 1024)
                    )
        except Exception:
            pass
        # Aggregate live Python objects by category so the largest
        # retained bytes show up clearly. Single O(N) walk; a few
        # thousand arrays is negligible vs the gen boundary work.
        np_count = 0
        np_bytes = 0
        bytes_count = 0
        bytes_total = 0
        tensor_count = 0
        tensor_bytes = 0
        top_arrays: list[tuple[str, int, int]] = []
        try:
            import gc as _gc
            import warnings as _warnings
            from collections import defaultdict
            bucket: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
            # gc.get_objects() walks every live Python object, which trips
            # torch's deprecation guard on `torch.distributed.reduce_op`
            # every iteration when the module is loaded. The guard is
            # informational, not actionable for us — torch will remove
            # the alias on its own schedule. Suppress its FutureWarning
            # for the duration of this single walk so the trainer log
            # isn't dominated by it.
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message=".*torch.distributed.reduce_op.*",
                    category=FutureWarning,
                )
                for obj in _gc.get_objects():
                    if isinstance(obj, np.ndarray):
                        np_count += 1
                        np_bytes += obj.nbytes
                        key = (obj.shape, str(obj.dtype))
                        bucket[key][0] += 1
                        bucket[key][1] += obj.nbytes
                    elif isinstance(obj, (bytes, bytearray)):
                        bytes_count += 1
                        bytes_total += len(obj)
                    elif isinstance(obj, torch.Tensor):
                        tensor_count += 1
                        try:
                            tensor_bytes += obj.numel() * obj.element_size()
                        except Exception:
                            pass
            ranked = sorted(
                bucket.items(), key=lambda kv: kv[1][1], reverse=True
            )[:3]
            for (shape, dtype), (count, nbytes) in ranked:
                top_arrays.append((f"{dtype}{list(shape)}", count, nbytes // (1024 * 1024)))
        except Exception:
            pass
        tops = "  ".join(f"{name}×{n}={mb}MB" for (name, n, mb) in top_arrays)
        # If RSS >> sum of tracked categories, the gap is
        # Rust/native/unmanaged (nes_core pool, torch driver reserve,
        # rayon stacks, mmap'd ROMs, etc.) OR untracked Python objects
        # like bytes that aren't cycle-GC tracked. tracemalloc (opt-in)
        # closes that gap by seeing ALL Python allocations.
        tracked_mb = (np_bytes + bytes_total + tensor_bytes) // (1024 * 1024)
        native_gap_mb = rss_mb - tracked_mb - max(0, mps_driver_mb)
        log.info(
            "[memstat gen %d] rss=%d MB  mps=%d/%d MB  py_blocks=%d  gc_freed=%d | "
            "np=%dMB(%d arrs)  bytes=%dMB(%d objs)  tensors=%dMB(%d)  native_gap=%dMB | top: %s",
            gen, rss_mb, mps_mb, mps_driver_mb, py_blocks, gc_freed,
            np_bytes // (1024 * 1024), np_count,
            bytes_total // (1024 * 1024), bytes_count,
            tensor_bytes // (1024 * 1024), tensor_count,
            native_gap_mb, tops,
        )
        # tracemalloc top-N — the definitive Python allocation tracker.
        # Unlike gc.get_objects above (which misses bytes, small ints,
        # etc.), tracemalloc sees every Python alloc with source-line
        # attribution. If the leak is Python-side this will point
        # directly at the guilty file:line. If the leak is Rust-side,
        # tracemalloc total will be SMALL and the native_gap above will
        # still be large — either way we learn which direction to dig.
        try:
            import tracemalloc as _tm
            if _tm.is_tracing():
                snap = _tm.take_snapshot()
                # Absolute top-N allocations (for orientation).
                stats = snap.statistics("lineno")[:8]
                tm_total_mb = sum(s.size for s in stats) // (1024 * 1024)
                lines = []
                for s in stats:
                    frame = s.traceback[0]
                    path = frame.filename
                    parts = path.replace("\\", "/").split("/")
                    short = "/".join(parts[-2:]) if len(parts) >= 2 else path
                    lines.append(
                        f"{short}:{frame.lineno}={s.size // (1024 * 1024)}MB({s.count})"
                    )
                log.info(
                    "[memstat gen %d] tm_top%d=%dMB | %s",
                    gen, len(stats), tm_total_mb, "  ".join(lines),
                )
                # Snapshot DIFF vs previous gen — this is the definitive
                # leak signal. Anything that's allocated between gen N-1
                # and gen N and NOT freed shows up as a positive size_diff.
                # The #1 entry here is the leak source, with file:lineno
                # attribution. Added 2026-04-21 because the absolute
                # top-N above was flat (~250 MB) while RSS grew 6.7 GB
                # per gen — the leak is invisible without the diff.
                prev = self._last_tm_snapshot
                if prev is not None:
                    deltas = snap.compare_to(prev, "lineno")
                    # Positive-only, sorted by size_diff descending.
                    growing = [d for d in deltas if d.size_diff > 0][:8]
                    delta_lines = []
                    for d in growing:
                        frame = d.traceback[0]
                        path = frame.filename
                        parts = path.replace("\\", "/").split("/")
                        short = "/".join(parts[-2:]) if len(parts) >= 2 else path
                        delta_lines.append(
                            f"{short}:{frame.lineno}=+{d.size_diff // (1024 * 1024)}MB(+{d.count_diff})"
                        )
                    total_growth_mb = sum(
                        d.size_diff for d in growing
                    ) // (1024 * 1024)
                    log.info(
                        "[memstat gen %d] tm_diff vs prev gen: +%dMB top=%s",
                        gen, total_growth_mb, "  ".join(delta_lines) or "(none)",
                    )
                self._last_tm_snapshot = snap
        except Exception as exc:
            log.debug("tracemalloc stats failed: %s", exc)

    def _respawn_pool(self, start_state_path: Optional[str]) -> bool:
        """Rebuild the worker pool with a new start state, exception-safe.

        Builds + starts the NEW pool BEFORE shutting down the old one, so
        a build failure (bad ROM / corrupt start state / OOM spinning up
        N workers) leaves the existing pool intact and training
        continues, instead of stranding `self.pool` as a dead/half-built
        reference. Returns True on success (pool swapped, old shut down,
        knobs reapplied); False on failure (current pool untouched)."""
        try:
            new_pool = _make_pool(
                env_spec=self.env_spec,
                rom_path=self.rom_path,
                num_workers=self.num_instances,
                frame_skip=self.frame_skip,
                start_state_path=start_state_path,
                seed=self.seed,
            )
            new_pool.start()
        except Exception as exc:
            log.error(
                "pool respawn (start_state=%s) failed: %s; keeping the "
                "current pool + start state.", start_state_path, exc,
            )
            return False
        old = self.pool
        self.pool = new_pool
        if old is not None:
            try:
                old.shutdown()
            except Exception as exc:
                log.warning("old pool shutdown after respawn failed: %s", exc)
        self._apply_pool_knobs()
        return True

    def _maybe_rebuild_pool_for_stage(self) -> None:
        """If the active curriculum stage specifies a per-stage start state
        different from the pool's current state, tear down and respawn the
        pool so workers boot into the new state. No-op if the stage has no
        override."""
        stage_state = self.curriculum.current_stage.start_state_path
        if stage_state is None:
            return
        pool = self.pool
        if pool is None or pool.start_state_path == stage_state:
            return
        log.info(
            "Curriculum stage %s specifies start_state=%s, rebuilding pool",
            self.curriculum.current_stage.name, stage_state,
        )
        self._respawn_pool(stage_state)

    def _drain_audio_updates(self) -> None:
        """Apply any audio-mixer control messages waiting on the GUI
        queue. Non-blocking; mixer is no-op if absent."""
        if self._audio_queue is None or self._audio_mixer is None:
            return
        while True:
            try:
                upd = self._audio_queue.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break
            if not isinstance(upd, dict):
                continue
            if "mode" in upd:
                # Mixer mode is mixer-local state (safe off-thread). The
                # pool worker-pace change is NOT — defer it to the
                # trainer thread via _pending_pace_mode (see __init__).
                self._audio_mixer.set_mode(upd["mode"])
                self._pending_pace_mode = upd["mode"]
            if "volume" in upd:
                self._audio_mixer.set_volume(upd["volume"])

    def _drain_pending_pace(self) -> None:
        """Apply a pending audio-mode worker-pace change. MUST be called
        from the trainer thread between step_all calls (where it has
        exclusive pool access), never from the audio-drainer thread."""
        mode = self._pending_pace_mode
        if mode is None:
            return
        self._pending_pace_mode = None
        self._apply_pace_for_mode(mode)

    def _apply_pace_for_mode(self, mode: str) -> None:
        """Map the audio mixer mode onto worker pacing/audio state.

        - ``solo-X``: today's behavior — pace worker X (audio source
          produces continuously at realtime) and unpace everyone else;
          audio explicitly on for X only, so entering solo from ``all``
          actually silences the rest.
        - ``all``: enable audio production on EVERY worker, pacing
          untouched — the Rust mixer already sums all rings with
          1/sqrt(n) normalization in Mode::All. Pool-level pacing
          (solo leftovers / pace-multiplier knob) stays whatever it is.
        - ``mute`` (and unknown modes): today's behavior — unpace
          everyone (training back to max throughput) and audio off
          everywhere.

        Backends without per-worker audio control fall back to the
        historical pace-welded behavior (solo paces X / others off;
        all+mute unpace everyone, which welds audio off)."""
        pool = self.pool
        if pool is None:
            return
        set_audio = getattr(pool, "set_worker_audio", None)
        if mode == "all":
            if set_audio is None:
                # Legacy backend: no audio-without-pacing control.
                # Unpace everyone (welds audio off) — 'all' stays
                # silent, exactly today's behavior.
                for worker_id in range(self.num_instances):
                    pool.set_worker_pace(worker_id, False)
                return
            for worker_id in range(self.num_instances):
                set_audio(worker_id, True)
            # A config that sets reinforce.pace_multiplier declares this
            # a spectator pool: pace every worker so N audio streams are
            # produced at ~playback rate (the pool-level pacing costs one
            # sleep per step_all regardless of worker count). Without the
            # knob, 'all' keeps today's semantics — audible but unpaced,
            # training at full speed.
            if self._active_pace_multiplier != 1.0:
                for worker_id in range(self.num_instances):
                    self.pool.set_worker_pace(worker_id, True)
                # Re-pin audio ON everywhere: the pace weld just turned
                # it on anyway, but be explicit — last call wins.
                for worker_id in range(self.num_instances):
                    set_audio(worker_id, True)
            return
        soloed: Optional[int] = None
        if mode.startswith("solo-"):
            try:
                soloed = int(mode.split("-", 1)[1])
            except ValueError:
                soloed = None
        for worker_id in range(self.num_instances):
            pool.set_worker_pace(worker_id, soloed == worker_id)
        # The pace weld only fires on a pace *transition*, so a worker
        # left audio-on by a previous 'all' mode (unpaced then, unpaced
        # now) would keep sounding through mute/solo. Pin audio
        # explicitly: soloed on, everyone else off. Last call wins on
        # the Rust side.
        if set_audio is not None:
            for worker_id in range(self.num_instances):
                set_audio(worker_id, soloed == worker_id)

    def _start_audio_drainer(self) -> None:
        if self._audio_queue is None or self._audio_mixer is None:
            return
        if self._audio_drainer_thread is not None:
            return
        stop_evt = threading.Event()

        def _loop() -> None:
            while not stop_evt.is_set():
                try:
                    self._drain_audio_updates()
                except Exception:
                    pass
                stop_evt.wait(0.05)

        self._audio_drainer_stop = stop_evt
        self._audio_drainer_thread = threading.Thread(
            target=_loop, name="audio-drainer", daemon=True,
        )
        self._audio_drainer_thread.start()

    def _stop_audio_drainer(self) -> None:
        if self._audio_drainer_stop is not None:
            self._audio_drainer_stop.set()
        self._audio_drainer_thread = None
        self._audio_drainer_stop = None

    # Reserved namespaced keys the reward queue accepts as control
    # messages instead of treating them as reward weights. Lets a single
    # queue carry both kinds of update without spawning a second IPC
    # channel for every new live-tunable knob.
    _CONTROL_KEY_START_STATE = "__start_state_path__"
    _CONTROL_KEY_GA = "__ga_params__"

    def _ga_batch_size(self) -> int:
        """Parallel slots available for GA genome evaluation."""
        return self.num_instances

    def _ga_worker_offset(self) -> int:
        """Index of the first GA worker slot in the pool."""
        return 0

    def _drain_pending_state_snapshots(self) -> int:
        """Turn the gen's depth-record notes into persisted
        .state.bin files under `checkpoints/auto_curriculum/`. Calls
        `pool.save_worker_state()` for each pending entry. Returns count
        actually written.

        Filename format:
            <checkpoint_dir>/auto_curriculum/depth_<d0>_<d1>_<d2>_<ts>.state.bin

        where d0, d1, d2 are the depth key components (e.g.
        dungeon_level, map_x, map_y for Zelda) and ts is a UNIX
        timestamp so multiple records in one run don't collide."""
        if self.pool is None or not self._pending_state_snapshots:
            return 0
        self._auto_curriculum_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        deepest_path: Optional[Path] = None
        deepest_key: Optional[tuple] = None
        pending, self._pending_state_snapshots = self._pending_state_snapshots, []
        for worker_id, depth_key, caption in pending:
            try:
                blob = self.pool.save_worker_state(worker_id, timeout=5.0)
            except Exception as exc:
                log.debug("save_worker_state(%d) failed: %s", worker_id, exc)
                continue
            if blob is None:
                continue
            key_str = "_".join(str(int(x)) for x in depth_key)
            ts_int = int(time.time())
            out_path = self._auto_curriculum_dir / (
                f"depth_{key_str}_{ts_int}.state.bin"
            )
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            try:
                tmp.write_bytes(blob)
                tmp.replace(out_path)
                written += 1
                log.info(
                    "auto-curriculum: saved depth snapshot key=%s -> %s (%s)",
                    depth_key, out_path.name, caption,
                )
                # Track the deepest-key snapshot this batch for
                # frontier-bootstrap below.
                if deepest_key is None or tuple(depth_key) > deepest_key:
                    deepest_key = tuple(depth_key)
                    deepest_path = out_path
            except OSError as exc:
                log.warning("failed to write %s: %s", out_path, exc)

        # Frontier bootstrap: if the agent reached a new-deepest-ever
        # state this generation, promote that snapshot to the pool's
        # start_state_path so subsequent episodes resume from the
        # frontier instead of cold-booting. Replaces the binary
        # "advance_threshold" curriculum signal (which CurriculumManager
        # can false-positive on local-minimum farming) with a direct
        # depth-record wire. Only applies when the profile opted into
        # auto-curriculum (self.start_state_path is None/empty OR was
        # set by a previous auto-curriculum snapshot).
        if (
            deepest_path is not None
            and getattr(self, "auto_curriculum_enabled", True)
            and (
                self.start_state_path is None
                or (
                    isinstance(self.start_state_path, str)
                    and str(self._auto_curriculum_dir) in self.start_state_path
                )
            )
        ):
            prev = self.start_state_path
            self.start_state_path = str(deepest_path)
            log.info(
                "auto-curriculum: frontier bootstrap %s -> %s (depth key=%s)",
                prev, deepest_path.name, deepest_key,
            )
        # Rotate old depth snapshots — keep the most recent N by mtime so
        # the dir doesn't grow unbounded over a long run (one file is
        # written per new depth record). Keep-newest preserves the active
        # frontier state (just promoted above, the newest); guard it
        # explicitly too so it can never be unlinked while in use.
        try:
            snaps = sorted(
                self._auto_curriculum_dir.glob("depth_*.state.bin"),
                key=lambda p: p.stat().st_mtime,
            )
            _keep = 30
            if len(snaps) > _keep:
                active = str(self.start_state_path or "")
                for old in snaps[:-_keep]:
                    if str(old) == active:
                        continue
                    try:
                        old.unlink()
                    except OSError:
                        pass
        except Exception:
            pass
        return written


    def _drain_reward_updates(self) -> None:
        """Apply any live updates the GUI pushed since the last
        generation. Handles three kinds of payload in a single queue:

          * `{weight_name: value, ...}` — reward weight updates.
          * `{"__start_state_path__": "/abs/path.bin"}` — swap the
            active start state. Triggers a pool rebuild so subsequent
            episodes boot into the new state. Pass `None` to clear.
          * `{"__ga_params__": {"mutation_std": 0.02, ...}}` — live
            adjustment of GA hyperparameters (mutation, tournament,
            elite fraction). Applied to `self.ga` immediately.
        """
        if self._reward_queue is None:
            return
        import queue as _queue
        applied_weights = False
        new_start_state = self._START_STATE_UNSET
        ga_changes: dict = {}
        while True:
            try:
                updates = self._reward_queue.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break
            if not isinstance(updates, dict):
                continue
            # Pull control-namespaced entries out before treating the
            # rest as reward weights.
            if self._CONTROL_KEY_START_STATE in updates:
                new_start_state = updates.pop(self._CONTROL_KEY_START_STATE)
            if self._CONTROL_KEY_GA in updates:
                ga_payload = updates.pop(self._CONTROL_KEY_GA)
                if isinstance(ga_payload, dict):
                    ga_changes.update(ga_payload)
            if updates:
                weights = self.game_profile.setdefault("reward_weights", {})
                for k, v in updates.items():
                    weights[k] = v
                applied_weights = True

        if applied_weights:
            # Rebuild the factory so the NEXT batch's reward_fn uses the
            # new weights. The running batch completes with the old weights
            # — avoids mid-episode reward discontinuities.
            self.reward_fn_factory = lambda: build_reward_function(self.game_profile)
            log.info("Reward weights updated live: %s", self.game_profile["reward_weights"])

        for k, v in ga_changes.items():
            if hasattr(self.ga, k):
                old = getattr(self.ga, k)
                setattr(self.ga, k, v)
                log.info("GA param updated live: %s = %s (was %s)", k, v, old)
            else:
                log.warning("Ignored unknown GA param: %s", k)

        if new_start_state is not self._START_STATE_UNSET:
            self._apply_new_start_state(new_start_state)

    # Sentinel distinct from None so the user can clear the start state
    # by sending an explicit None.
    _START_STATE_UNSET = object()

    def _apply_new_start_state(self, new_path) -> None:
        """Tear down + respawn the worker pool so subsequent episodes
        boot into the new start state. Does NOT interrupt the current
        batch — the next generation's resets will land on the new state.
        Same external transports are reattached so the GUI grid keeps
        rendering without seams.
        """
        new_path_str = str(new_path) if new_path else None
        if new_path_str == self.start_state_path:
            log.info(
                "Start-state swap requested but path is unchanged (%s); ignoring.",
                new_path_str,
            )
            return
        if new_path_str and not Path(new_path_str).exists():
            log.warning(
                "Start-state swap requested but file %s does not exist; ignoring.",
                new_path_str,
            )
            return
        log.info(
            "Start-state swap: %s -> %s; rebuilding pool.",
            self.start_state_path, new_path_str,
        )
        if self.pool is None:
            # No live pool yet — just record; the next build uses it.
            self.start_state_path = new_path_str
            return
        # Commit the new start state only if the respawn actually
        # succeeds, so a failed swap leaves pool + start_state_path
        # consistent at the old (working) values.
        if self._respawn_pool(new_path_str):
            self.start_state_path = new_path_str

    def _apply_pool_knobs(self) -> None:
        """Push trainer-side configuration onto the freshly-started pool.

        Accesses the inner ``nes_core.Pool`` through ``RustPool._inner``
        so we can reach optional methods (``set_preprocess_f16``,
        ``set_headless``) without widening the adapter's public surface.
        Silently skips when the adapter doesn't expose an inner pool
        (e.g. legacy mock pool used by a few tests).
        """
        pool = self.pool
        if pool is None:
            return
        inner = getattr(pool, "_inner", None)
        if inner is None:
            return
        if self.preprocess_f16:
            setter = getattr(inner, "set_preprocess_f16", None)
            if setter is not None:
                setter(True)
                log.info("Pool preprocess_f16=ON — observations arrive as np.float16 in [0, 1].")
        # Headless: when there's no GUI frame sink, nobody reads the full
        # 256x240x3 RGB frame — the policy uses RAM (tile) or the 84x84
        # `preprocessed` (pixel). Skip rendering/allocating the RGB frame
        # per worker per step. Measured ~2% on step_all and removes the
        # per-worker RGB alloc; preprocessed is unaffected. Headless mode
        # returns a 1x1x3 dummy frame, so only enable it when no sink
        # needs real frames.
        if self._frame_sink is None:
            setter = getattr(inner, "set_headless", None)
            if setter is not None:
                setter(True)
                log.info(
                    "Pool headless=ON (no frame sink) — skipping RGB frame "
                    "render; policy obs (RAM / preprocessed) unaffected."
                )
        # Tile-encoder games read `ram_snapshot`, never the 84x84
        # `preprocessed` obs, so the per-worker gray+resize kernel is
        # pure wasted CPU in headless training (60 workers x 1024
        # steps/iter). Skip it for tile mode; pixel paths still get it.
        if self._is_tile_mode and self._frame_sink is None:
            setter = getattr(inner, "set_skip_preprocess", None)
            if setter is not None:
                setter(True)
                log.info(
                    "Pool skip_preprocess=ON (tile mode) — skipping the 84x84 "
                    "gray+resize kernel; policy reads RAM (ram_snapshot)."
                )
        # Panic isolation: when False, rayon worker bodies run without
        # a `catch_unwind` wrap. Saves 0.5-1% throughput per the Rust
        # docs (pool.rs:528-538). Cost: a panic in any worker (e.g. a
        # bad ROM, a mapper bug, a corrupt save state) unwinds past
        # rayon and may abort the whole training process. Acceptable
        # trade in production after ROMs/states are validated. Default
        # remains True for safety; opt out per-profile via
        # `reinforce.panic_isolation: false`.
        if not self.panic_isolation:
            setter = getattr(inner, "set_panic_isolation", None)
            if setter is not None:
                setter(False)
                log.info(
                    "Pool panic_isolation=OFF — production fast path. "
                    "Worker panics may abort the process."
                )
        # Batched PPU (skip-per-pixel via clean-scanline history).
        # Headless-only: under Replace, raster-effect games can paint a
        # 1-frame-stale row — invisible to a 4-frame-stacked policy but
        # wrong for the GUI grid / recordings, so a live frame sink
        # forces exact per-pixel rendering regardless of the profile.
        if self.batched_render in ("verify", "replace"):
            if self._frame_sink is None:
                setter = getattr(inner, "set_batched_render_mode", None)
                if setter is not None:
                    setter(self.batched_render)
                    log.info(
                        "Pool batched_render=%s — skip-per-pixel PPU "
                        "fast path enabled.",
                        self.batched_render,
                    )
            else:
                log.info(
                    "Pool batched_render=%s requested but a frame sink "
                    "is active — keeping exact rendering (off).",
                    self.batched_render,
                )
        elif self.batched_render != "off":
            log.warning(
                "Unknown reinforce.batched_render=%r (expected off | "
                "verify | replace) — keeping off.",
                self.batched_render,
            )
        # ASM bulk budget: opt-in per-profile, honored only by the
        # batch-safe mappers (MMC1/UxROM). Default 1 leaves emulation
        # timing unchanged from the shipped path; see the config
        # comment where `self.asm_bulk_cycles` is read.
        if self.asm_bulk_cycles > 1:
            setter = getattr(inner, "set_asm_bulk_cycles", None)
            if setter is not None:
                setter(self.asm_bulk_cycles)
                log.info(
                    "Pool asm_bulk_cycles=%d — ASM CPU batches up to %d "
                    "cycles per invocation on MMC1/UxROM workers. Only "
                    "enable after the lockstep/Mesen/parity gate passes "
                    "for this game at this budget.",
                    self.asm_bulk_cycles,
                    self.asm_bulk_cycles,
                )
            else:
                log.warning(
                    "reinforce.asm_bulk_cycles=%d requested but the pool "
                    "binary has no set_asm_bulk_cycles — running at the "
                    "default budget of 1. The installed nes_core .so is "
                    "likely stale; rebuild and reinstall it.",
                    self.asm_bulk_cycles,
                )
        # Spectator pacing speed (reinforce.pace_multiplier). Goes
        # through the ADAPTER method — not `inner` — because the
        # adapter records the applied multiplier and stamps it onto
        # StepResult.audio_rate (BASE_AUDIO_RATE × m) so the mixer's
        # resampler pitch-ups instead of overflowing its ring. Only
        # applied when a frame sink exists: headless training must
        # never pace. Old adapter/.so => silent no-op (pool keeps
        # pacing at 1.0×, audio_rate stays 1.0× — consistent).
        if self.pace_multiplier != 1.0 and self._frame_sink is not None:
            setter = getattr(pool, "set_pace_multiplier", None)
            if setter is not None:
                setter(self.pace_multiplier)
                self._active_pace_multiplier = getattr(
                    pool, "_pace_multiplier", self.pace_multiplier
                )
                log.info(
                    "Pool pace_multiplier=%.2f — paced stepping targets "
                    "%.2fx realtime; audio_rate scales to match.",
                    self._active_pace_multiplier,
                    self._active_pace_multiplier,
                )

    def _record_curriculum_episodes(
        self,
        pop: list,
        records: list[tuple[str, bool, int]],
    ) -> None:
        """Flush a generation's buffered episodes into the curriculum.

        `records` is a list of (level_id, success, global_genome_index) in the
        same order the episodes were evaluated. When the curriculum has a
        top_k_gate, rank the whole population by THIS generation's mean fitness
        (already finalized by the caller) and tag each episode with whether its
        genome is one of the top-k — the advance/regress gate then measures
        only the best k policies' clear rate, mirroring how the vanilla trainer
        gates on its single deployed policy rather than a population average.

        Ranking uses the current generation's fitness, not the previous one's:
        at generation 0 there is no prior fitness, and immediately after a
        stage advance the fitness scale resets to the new level set, so only
        the freshly-measured fitnesses identify the current best policies.

        With no gate (top_k_gate is None) the eligibility flag is passed as
        None and ignored downstream — every episode is recorded exactly as
        before, in the original order.
        """
        gate = self.curriculum.top_k_gate
        topk_idx: set[int] | None = None
        if gate is not None and gate > 0 and pop:
            # k highest by mean fitness. Stable sort → deterministic ties.
            ranked = sorted(
                range(len(pop)), key=lambda i: pop[i].fitness, reverse=True
            )
            topk_idx = set(ranked[: min(gate, len(pop))])
        n_topk_clears = 0
        for level_id, success, g_idx in records:
            eligible = None if topk_idx is None else (g_idx in topk_idx)
            self.curriculum.record_episode(
                level_id, success, top_k_eligible=eligible
            )
            if eligible and success:
                n_topk_clears += 1
        if topk_idx is not None:
            # Cheap per-gen telemetry so a run can be inspected for whether the
            # gate is moving (top-k clears > 0) without re-deriving it from raw
            # episode logs.
            log.info(
                "  curriculum top_k_gate=%d: %d/%d episodes from top-k genomes, "
                "%d top-k clears; stage_success_rate=%.4f",
                gate,
                sum(1 for _, _, gi in records if gi in topk_idx),
                len(records),
                n_topk_clears,
                self.curriculum.stage_success_rate(),
            )

    def _run_one_generation(self, gen: int) -> None:
        log.info("=== Generation %d (stage: %s) ===", gen, self.curriculum.current_stage.name)
        self._drain_reward_updates()
        # Apply any pending GUI audio-mode worker-pace change on the
        # trainer thread (the drainer thread only records it).
        self._drain_pending_pace()
        # Audio-mixer updates are drained by a dedicated daemon thread
        # (see _start_audio_drainer) so mute/solo/volume apply within
        # ~50 ms instead of once per generation.

        # Evaluate genomes in batches of `num_instances`.
        pop = self.ga.population
        # Flat trajectory storage: per-population-position slice of the
        # pre-allocated arrays. Indexed by genome index (0..pop_size).
        pop_obs_dtype = self._obs_buffer_dtype()
        pop_traj_obs = np.zeros(
            self._obs_buffer_shape(len(pop), self.max_episode_steps),
            dtype=pop_obs_dtype,
        )
        pop_traj_actions = np.zeros((len(pop), self.max_episode_steps), dtype=np.int32)
        pop_traj_rewards = np.zeros((len(pop), self.max_episode_steps), dtype=np.float32)
        pop_traj_log_probs = np.zeros((len(pop), self.max_episode_steps), dtype=np.float32)
        pop_traj_lens = np.zeros(len(pop), dtype=np.int32)
        gen_breakdown: dict[str, float] = {}
        # Curriculum episode records for this generation, flushed AFTER every
        # genome's mean fitness is known (below). Each entry is
        # (level_id, success, global_genome_index). Deferring the flush lets
        # us tag each episode with whether its genome landed in this
        # generation's top-k by fitness — the top_k_gate advance gate then
        # measures only those. The order of collection matches the old inline
        # record order (batch-major, then episode, then genome), so the legacy
        # (gate-off) path appends to the curriculum in the exact same sequence
        # it always did.
        gen_curriculum_records: list[tuple[str, bool, int]] = []
        ga_batch_size = self._ga_batch_size()
        for batch_start in range(0, len(pop), ga_batch_size):
            if not self._running:
                return
            batch = pop[batch_start : batch_start + ga_batch_size]
            nb = len(batch)
            # Multi-episode evaluation: run `episodes_per_genome` rollouts
            # and average fitness so the GA selects on expected return,
            # not single-episode luck. Per-episode trajectory is kept
            # only for the BEST-fitness episode of each genome, so PPO
            # trains on the agent's good runs (which actually reach
            # interesting states) instead of a randomly-picked one.
            #
            # The curriculum records EVERY episode individually (not OR-ed
            # across) so the rolling success-rate window over all episodes
            # remains unbiased — multi-eval doesn't artificially inflate
            # apparent skill by treating "1 success in N tries" as a
            # success at the genome level.
            best_fits = [-float("inf")] * nb
            sum_fits = [0.0] * nb
            best_traj_obs = None
            best_traj_actions = None
            best_traj_rewards = None
            best_traj_log_probs = None
            best_traj_lens = None
            for _ep in range(self.episodes_per_genome):
                fitnesses, successes, level_ids, traj_flat, batch_bd = \
                    self._evaluate_batch(batch)
                for g_i in range(nb):
                    sum_fits[g_i] += float(fitnesses[g_i])
                    if fitnesses[g_i] > best_fits[g_i]:
                        best_fits[g_i] = float(fitnesses[g_i])
                        # Lazy-allocate the best-trajectory holders on
                        # first improvement; per-genome slice copy when
                        # this episode is the new best.
                        if best_traj_obs is None:
                            best_traj_obs = np.zeros_like(traj_flat["obs"])
                            best_traj_actions = np.zeros_like(traj_flat["actions"])
                            best_traj_rewards = np.zeros_like(traj_flat["rewards"])
                            best_traj_log_probs = np.zeros_like(traj_flat["log_probs"])
                            best_traj_lens = np.zeros_like(traj_flat["lens"])
                        best_traj_obs[g_i] = traj_flat["obs"][g_i]
                        best_traj_actions[g_i] = traj_flat["actions"][g_i]
                        best_traj_rewards[g_i] = traj_flat["rewards"][g_i]
                        best_traj_log_probs[g_i] = traj_flat["log_probs"][g_i]
                        best_traj_lens[g_i] = traj_flat["lens"][g_i]
                # Curriculum tracks every episode regardless of best/avg
                # so the rolling success-rate window is unbiased. Buffered
                # here (not recorded inline) so the flush below can tag each
                # episode with its genome's top-k membership once this
                # generation's fitnesses are final. global index = the
                # genome's position in `pop` = batch_start + g_i.
                for g_i, (success, level_id) in enumerate(zip(successes, level_ids)):
                    gen_curriculum_records.append(
                        (level_id, success, batch_start + g_i)
                    )
                # Capture successful trajectories to the BC replay buffer.
                # Done HERE (inside the per-episode loop) rather than after
                # multi-eval averaging because we need the per-episode
                # success flag — the mean-fitness aggregation discards
                # per-episode metadata. Each captured trajectory is the
                # exact (obs, action) sequence that produced a level
                # clear, perfect for behavior cloning to imitate.
                if self.bc_replay_enabled:
                    for g_i in range(nb):
                        if not successes[g_i]:
                            continue
                        traj_len = int(traj_flat["lens"][g_i])
                        if traj_len < 2:
                            continue
                        # Slice down to actual length to avoid storing
                        # zero-padded tail data; copy so the in-memory
                        # buffer survives the next iteration's
                        # traj_flat reuse. The 5th tuple element is
                        # the SOURCE genome_id — diagnostics-only for
                        # now (BC training currently picks by recency
                        # not by id), but it makes the buffer's
                        # provenance inspectable post-hoc.
                        self._bc_replay_buffer.append((
                            traj_flat["obs"][g_i, :traj_len].copy(),
                            traj_flat["actions"][g_i, :traj_len].copy(),
                            traj_flat["rewards"][g_i, :traj_len].copy(),
                            float(fitnesses[g_i]),
                            int(batch[g_i].genome_id),
                        ))
                        # FIFO eviction once over capacity.
                        if len(self._bc_replay_buffer) > self.bc_replay_max_buffer:
                            self._bc_replay_buffer.pop(0)
                        log.info(
                            "  BC replay: captured success trajectory "
                            "(genome_id=%d, slot=%d, len=%d, fitness=%.1f, buffer=%d/%d)",
                            int(batch[g_i].genome_id), g_i, traj_len,
                            fitnesses[g_i],
                            len(self._bc_replay_buffer),
                            self.bc_replay_max_buffer,
                        )
                        # Immediately persist to disk so a clear never
                        # gets lost across crashes/restarts/Ctrl-C.
                        self._save_bc_success_cache()
                for k, v in batch_bd.items():
                    gen_breakdown[k] = gen_breakdown.get(k, 0.0) + v

            # Mean fitness across episodes — the GA selects on this.
            for g_i, g in enumerate(batch):
                g.fitness = sum_fits[g_i] / self.episodes_per_genome

            # Copy best-episode trajectories into the population-wide slot.
            assert best_traj_obs is not None  # always set after first ep
            pop_traj_obs[batch_start:batch_start + nb] = best_traj_obs[:nb]
            pop_traj_actions[batch_start:batch_start + nb] = best_traj_actions[:nb]
            pop_traj_rewards[batch_start:batch_start + nb] = best_traj_rewards[:nb]
            pop_traj_log_probs[batch_start:batch_start + nb] = best_traj_log_probs[:nb]
            pop_traj_lens[batch_start:batch_start + nb] = best_traj_lens[:nb]

        # Flush this generation's episodes into the curriculum now that every
        # genome's mean fitness (set above) is final. Done BEFORE the PPO step
        # so top-k ranking reads the pure eval fitnesses, not any post-PPO
        # clone-overwrite. No-op ordering change for the legacy gate.
        self._record_curriculum_episodes(pop, gen_curriculum_records)

        best = self.ga.best_genome()
        avg = sum(g.fitness for g in pop) / len(pop)
        log.info("Gen %d: best=%.2f avg=%.2f", gen, best.fitness, avg)
        raw_fits = [g.fitness for g in pop]
        log.info(
            "Gen %d raw fitnesses: [%s]  distinct=%d/%d",
            gen,
            ", ".join(f"{f:.4f}" for f in raw_fits),
            len(set(raw_fits)),
            len(raw_fits),
        )
        try:
            # Per-genome length-aware concatenation. The pop_traj_actions
            # buffer is pre-allocated zero-init at (n, max_episode_steps);
            # any genome that died before the max length has its tail
            # filled with zeros. The previous slice `[:, :max_len]` was
            # counting those zeros as legitimate action-0 (NOOP)
            # selections, which made the histogram report 60-70% noop
            # when actual policy distributions were much more diverse.
            # Now we only ravel the prefix [0:traj_lens[i]] of each
            # genome's row.
            real_actions = np.concatenate([
                pop_traj_actions[i, :int(pop_traj_lens[i])]
                for i in range(len(pop_traj_lens))
                if int(pop_traj_lens[i]) > 0
            ]) if pop_traj_lens.size and int(pop_traj_lens.max()) > 0 else np.array([], dtype=pop_traj_actions.dtype)
            counts = np.bincount(real_actions, minlength=self.num_actions)
            total = int(counts.sum()) or 1
            hist = ", ".join(
                f"{i}:{c}({100.0*c/total:.1f}%)"
                for i, c in enumerate(counts.tolist())
            )
            log.info("Gen %d action histogram [idx:count(pct)]: %s", gen, hist)
        except Exception as exc:
            log.debug("action histogram skipped: %s", exc)

        # REINFORCE step on the elite genome using the generation's best
        # trajectories. Gradient descent pushes the elite toward actions that
        # earned above-average returns; next-gen mutation+crossover build on
        # the improved weights.
        ppo_stats: dict[str, float] = {}
        # GA-only warmup: the first `warmup_gens_ga_only` generations
        # skip the PPO step entirely. Lets a strong BC seed propagate
        # through the population via crossover before PPO's noisy
        # advantage signal pulls the elite away from it. Once warmup
        # ends, PPO resumes normally.
        in_warmup = (
            self.warmup_gens_ga_only > 0
            and self.ga.generation < self.warmup_gens_ga_only
        )
        if in_warmup:
            log.info(
                "Gen %d in GA-only warmup (%d/%d) — skipping PPO update",
                gen, self.ga.generation, self.warmup_gens_ga_only,
            )
            if self.redo_enabled:
                # Registered skip line (AMENDMENT 1 B3.3/B6): a GA-only
                # warmup generation performs no gradient step, so no
                # dormancy check runs and there is no optimizer state
                # to handle.
                log.info(
                    "[redo] iter %d: skipped (no gradient step)", gen
                )
        if self.reinforce_enabled and not in_warmup:
            # Rank by fitness; pick top_k with non-trivial trajectories.
            indexed = list(enumerate(pop))
            indexed.sort(key=lambda ig: ig[1].fitness, reverse=True)
            top_k = max(1, int(self.reinforce_top_k))
            elite_idx = [
                idx for idx, _ in indexed[:top_k] if pop_traj_lens[idx] > 1
            ]
            # Snapshot best's pre-PPO weights so a regressive update
            # can't permanently destroy the generation's winning
            # policy. Re-injected into the weakest population slot
            # below — keeps both versions alive for the next eval.
            pre_ppo_snapshot = self._snapshot_pre_ppo_elite(best, elite_idx)
            if elite_idx:
                try:
                    loss, ppo_stats = self._reinforce_update(
                        best,
                        pop_traj_obs,
                        pop_traj_actions,
                        pop_traj_rewards,
                        pop_traj_log_probs,
                        pop_traj_lens,
                        elite_idx,
                    )
                    num_preserved = max(
                        1, int(self.ga.elite_fraction * len(pop))
                    )
                    # Copy `best`'s post-update weights into the other
                    # top-ranked genomes the GA will keep. Skip index
                    # 0 (that's `best` itself, already updated).
                    #
                    # When `preserve_elite_diversity` is True, skip
                    # the clone-overwrite entirely. The other elites
                    # keep their own structurally-distinct weights —
                    # crucial for tile mode, where collapsing all 4
                    # elites onto `best`'s policy after every PPO
                    # step destroys the diversity GA needs to escape
                    # local minima. PPO still updates `best`; the
                    # other elites just don't inherit those updates.
                    if not self.preserve_elite_diversity:
                        for idx, _ in indexed[1:num_preserved]:
                            pop[idx].state_dict = {
                                k: v.clone() for k, v in best.state_dict.items()
                            }
                            # Fitness must move with the state_dict; the
                            # genome now contains best's exact weights so
                            # its true fitness == best.fitness. Without
                            # this, GA's next sort uses the genome's
                            # stale (pre-overwrite) score, which can
                            # demote the freshly-cloned elite below
                            # rank-1 and break the elitism guarantee.
                            pop[idx].fitness = best.fitness
                    log.info(
                        "  REINFORCE loss: %.4f (n=%d, applied to %d elites, preserve_diversity=%s)",
                        loss, len(elite_idx),
                        1 if self.preserve_elite_diversity else num_preserved,
                        self.preserve_elite_diversity,
                    )
                    # Inject the pre-PPO snapshot into the weakest
                    # population slot so GA elitism keeps it for one
                    # more gen. If PPO regressed `best`, the next
                    # eval will re-score the snapshot ≈ pre-PPO
                    # fitness and the GA will resurface it as elite.
                    # If PPO improved best, the snapshot drops out
                    # naturally on the next sort.
                    self._inject_pre_ppo_snapshot(pre_ppo_snapshot, pop, best)
                except Exception as exc:
                    # Don't let a bad gradient step kill an overnight run —
                    # the GA alone will still make progress.
                    log.warning("  REINFORCE step failed: %s", exc)

        # MPS allocator caches transient tensors aggressively; without a
        # periodic flush, an overnight run on a 128 GB machine can still
        # accumulate several GB of unreclaimed device memory.
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        advanced = self.curriculum.maybe_advance()
        regressed = False
        if not advanced:
            regressed = self.curriculum.maybe_regress()
        if advanced:
            log.info("Advanced -> %s", self.curriculum.current_stage.name)
        if regressed:
            log.info("Regressed -> %s", self.curriculum.current_stage.name)

        # If the new stage specifies a different start-state file than the
        # pool was spawned with, rebuild the pool so subsequent episodes
        # reset to the stage-specific state. Expensive but rare (stage
        # changes happen at most once per dozens of generations).
        if advanced or regressed:
            self._maybe_rebuild_pool_for_stage()

        # Drain narrator events accumulated over the generation and
        # push them onto the GUI's caption queue. Non-blocking;
        # ephemeral drop on backpressure.
        if self._narrator is not None and self._narrator_queue is not None:
            from src.training.narrator import push_events_to_queue
            push_events_to_queue(self._narrator.drain(), self._narrator_queue)

        # Auto-curriculum: write .state.bin snapshots for any depth
        # records the gen's step loop queued.
        self._drain_pending_state_snapshots()

        # Normalize breakdown: per-genome average so values are comparable
        # across pop sizes. Prefix keys with "reward_" so the metrics
        # window can easily filter them.
        pop_size = max(1, len(pop))
        breakdown_metrics = {f"reward_{k}": v / pop_size for k, v in gen_breakdown.items()}
        # Per-section timing breakdown (milliseconds) so perf regressions
        # surface in the next `jq '.timing_*' metrics.jsonl`.
        timing_metrics = self._gen_timer.snapshot()
        self._gen_timer.reset()

        # Policy-divergence count accumulated by `_safe_sample_from_logits`
        # across this generation's step loop. Defaults to 0 (not
        # null/missing) when no NaN/Inf substitution happened.
        nan_rows_this_gen = self._nan_rows_this_gen
        self._nan_rows_this_gen = 0

        # Current all-time depth record key (per game-specific lex order).
        # Surfaces in the dashboard as a "deepest reached" curve so the
        # user can watch curriculum progress at a glance even when raw
        # fitness numbers are noisy across stages.
        depth_metrics: dict[str, float] = {}
        try:
            best_depth = self._depth_tracker.best
            if best_depth is not None:
                # Tuple shape varies per game; encode as a single
                # monotone-increasing scalar by treating elements as
                # base-256 digits (matches the tracker's lex order).
                depth_scalar = 0
                for d in best_depth:
                    depth_scalar = depth_scalar * 256 + int(d)
                depth_metrics["depth_scalar"] = float(depth_scalar)
                # Also surface the last-element (typically the most
                # actionable axis: x-pos for Mario, room for Zelda).
                depth_metrics["depth_leaf"] = float(best_depth[-1])
        except Exception:
            pass

        self._emit_metrics(
            generation=gen,
            best_fitness=best.fitness,
            avg_fitness=avg,
            stage=self.curriculum.current_stage.name,
            success_rate=self.curriculum.stage_success_rate(),
            episodes=self.curriculum.episodes_in_stage,
            nan_rows_this_gen=nan_rows_this_gen,
            **breakdown_metrics,
            **timing_metrics,
            **ppo_stats,
            **depth_metrics,
        )

        # Log the breakdown for quick human inspection — which signals
        # actually fired this generation?
        if gen_breakdown:
            top = sorted(gen_breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            log.info("  reward breakdown: %s", " ".join(f"{k}={v:.1f}" for k, v in top))

        # BC replay triggers on EITHER:
        #   * a new trajectory was captured this gen (buffer grew) — fire
        #     immediately so PPO doesn't drift the policy away from the
        #     freshly-discovered clear before BC can anchor it
        #   * the modulo schedule (every bc_replay_every_gens) — keeps
        #     anchoring active even on stable gens with no new clears,
        #     resampling the latest train_window trajectory
        cur_bc_buf = len(self._bc_replay_buffer)
        new_capture_this_gen = cur_bc_buf > self._last_bc_buffer_size
        modulo_fire = gen > 0 and gen % self.bc_replay_every_gens == 0
        if (
            self.bc_replay_enabled
            and cur_bc_buf > 0
            and (new_capture_this_gen or modulo_fire)
        ):
            if new_capture_this_gen:
                log.info(
                    "  BC replay: triggered immediately (buffer grew %d -> %d this gen)",
                    self._last_bc_buffer_size, cur_bc_buf,
                )
            self._run_bc_replay(pop)
        self._last_bc_buffer_size = len(self._bc_replay_buffer)

        self.ga.evolve()

    def _evaluate_batch(
        self, genomes: list[Genome]
    ) -> tuple[list[float], list[bool], list[str], dict, dict]:
        """Run one episode per genome in parallel. Unused workers idle.

        Returns `(fitnesses, successes, level_ids, traj_flat, batch_breakdown)`
        where `traj_flat` is the flat ndarray bundle PPO/REINFORCE consumes.

        Three phases — grep `=== PHASE` to jump between them:
          1. SETUP  — build per-genome nets (+ optional vmap stacked-fn),
                      reward fns, frame stackers; reset all workers; pre-
                      allocate per-genome trajectory storage; optionally
                      start the async-pipeline step executor.
          2. STEP LOOP — per-step forward → action dispatch → pool.step_all
                         → reward+depth bookkeeping → trajectory write.
                         Runs up to `max_episode_steps` or until every
                         genome is done.
          3. FINALIZE — drain the async executor, fold per-genome reward
                        breakdowns into the batch-level dict, drive the
                        audio mixer's intensity by z-scored fitness, and
                        bundle the trajectory arrays for the return.
        """
        # === PHASE 1: SETUP ===
        assert self.pool is not None
        n = len(genomes)

        # Load each genome's weights into a fresh network; keep them on the
        # main process (MPS tensors don't cross process boundaries cleanly).
        nets: list[PolicyNetwork] = []
        for g in genomes:
            net = self._make_network()
            # strict=False: old checkpoints may predate value_head (PPO+GAE
            # added post-hoc); missing keys stay at their fresh-init random
            # values and PPO retrains them. See trainer.py:424 for the same
            # rationale at probe-load time.
            net.load_state_dict(g.state_dict, strict=False)
            net.to(self.device)
            net.eval()
            # Try to compile on MPS for a per-step dispatch speedup.
            # torch.compile can fail silently on some MPS graphs — wrap in
            # try/except so it never breaks a run. Effect stacks on top of
            # batched-forward and fp16.
            if self.compile_nets:
                try:
                    net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
                except Exception as exc:
                    log.debug("torch.compile failed, running eager: %s", exc)
                    self.compile_nets = False  # give up for this run
            nets.append(net)

        # Prepare the vmap stacked-parameter path. One-time cost per batch;
        # per-step forward is a single dispatch instead of N.
        stacked_fn = None
        if self.vmap_forward and len(nets) > 1:
            # Setup can legitimately fail on very old PyTorch (missing
            # torch.func). That's a build-env issue, not a bug in our
            # code. Log at WARNING so it's visible, then fall back.
            try:
                from torch.func import functional_call, stack_module_state, vmap
                raw_nets = [
                    n._orig_mod if hasattr(n, "_orig_mod") else n for n in nets
                ]
                stacked_params, stacked_buffers = stack_module_state(raw_nets)
                base_net = raw_nets[0]
                def _one(p, b, x):
                    return functional_call(base_net, (p, b), (x,))
                stacked_fn = vmap(_one, in_dims=(0, 0, 0))
            except ImportError as exc:
                log.warning(
                    "vmap unavailable (missing torch.func — PyTorch too old?): %s; "
                    "falling back to serial forward.",
                    exc,
                )
                stacked_fn = None

        reward_fns = [self.reward_fn_factory() for _ in range(n)]
        for fn in reward_fns:
            fn.reset()

        obs_dtype = self._obs_buffer_dtype()
        # Frame stackers — pixel mode uses the 84×84 grayscale stacker,
        # tile mode uses the 1D feature-vector stacker (yumouwei recipe
        # for sparse RAM observations). When `tile_frame_stack == 1`
        # the tile stacker still wraps the feature vector but adds no
        # temporal context, matching the prior no-stack behavior.
        if self._is_tile_mode:
            base_dim = self._tile_extractor.feature_dim
            stackers = []
            tile_stackers = [
                TileFeatureStacker(
                    stack_size=self._tile_frame_stack,
                    feature_dim=base_dim,
                ) for _ in range(n)
            ]
        else:
            stackers = [FrameStacker(dtype=obs_dtype) for _ in range(n)]
            tile_stackers = []
        # Sticky-action state: per-genome record of the most-recently
        # executed action. Initialized to 0 (NOOP) at episode start so
        # the very first step can never roll into a "stick to NOOP"
        # state — even if sticky fires, it'd repeat NOOP which is a
        # safe default. After each step this is updated to whatever
        # action the env actually executed (post-sticky-override).
        last_action_per_genome: list[int] = [0] * n

        # Reset every worker via the pool's public surface. Returns
        # StepResult-per-worker sorted by worker_id. The Rust rayon pool
        # returns all workers' results in one PyO3 call per generation.
        offset = self._ga_worker_offset()
        all_init = self.pool.reset_all()
        self._emit_frame_sink(all_init)
        init_results = [r for r in all_init if r.worker_id >= offset]
        # Build per-genome initial observation. Pixel mode resets the
        # FrameStacker with the first frame; tile mode decodes RAM
        # directly into a feature vector. The resulting `stacked_obs`
        # variable is used identically by the rest of the loop below
        # — only its shape differs.
        if self._is_tile_mode:
            stacked_obs: list[np.ndarray] = [
                tile_stackers[i].reset(
                    self._tile_extractor.extract(r.ram_snapshot)
                )
                for i, r in enumerate(init_results)
            ]
        else:
            stacked_obs = [
                stackers[i].reset(r.frame, getattr(r, "preprocessed", None))
                for i, r in enumerate(init_results)
            ]
        # GUI reads frames from the trainer's frame_sink callback directly.

        cumulative = [0.0] * n
        done_flags = [False] * n
        success_flags = [False] * n
        # Pin the level_id to the FIRST observation per genome instead of
        # overwriting every step. Otherwise the curriculum success record
        # gets attributed to whatever screen the agent died on, not the
        # screen they were trying to clear — noisy stats and false
        # advancement signals.
        level_ids: list[str] = [""] * n
        level_id_locked: list[bool] = [False] * n
        # Pre-allocated trajectory storage. Historically this was a
        # list-of-list-of-tuples where every step allocated a fresh
        # 4×84×84 uint8 copy via np.ascontiguousarray(...).copy().
        # At pop=16 × max_steps=1000 that's 16,000 small mallocs per
        # generation = ~450 MB of churn + GC pressure, occasional
        # OOM kills on overnight runs. One pre-alloc per batch cuts
        # allocs to exactly four and lets the per-step writes land as
        # in-place slice assignments.
        _obs_shape = self._obs_buffer_shape(n, self.max_episode_steps)
        traj_obs = np.zeros(_obs_shape, dtype=obs_dtype)
        traj_actions = np.zeros((n, self.max_episode_steps), dtype=np.int32)
        traj_rewards = np.zeros((n, self.max_episode_steps), dtype=np.float32)
        traj_log_probs = np.zeros((n, self.max_episode_steps), dtype=np.float32)
        traj_lens = np.zeros(n, dtype=np.int32)
        # Trajectories live in the flat arrays above; _reinforce_update
        # slices them via traj_lens[i]. No list-of-tuples / list-of-views
        # allocated per step — eliminates ~96k tuple + view objects/gen.
        # Pre-allocate the forward-pass batch buffer. Previously
        # `np.stack([stacked_obs[i] for i in active])` allocated a
        # fresh (num_active, 4, 84, 84) uint8 array every step —
        # ~4,000 small mallocs × pop=16 ≈ 64k mallocs + ~1.8 GB of
        # Numpy allocation churn per generation. One flat buffer
        # reused across the episode costs effectively zero.
        batch_np_buffer = np.zeros((n,) + self._step_obs_shape(), dtype=obs_dtype)
        # Per-worker action bitmasks passed to `pool.step_all` every
        # step. Previously rebuilt as `[0] * num_workers` each iteration
        # — PyO3 had to unbox each PyLong and alloc a fresh Rust
        # `Vec<u8>` per call (~4000/gen). A reused numpy uint8 array
        # gives Rust a zero-copy buffer handle via `PyReadonlyArray1`.
        step_actions = np.zeros(self.pool.num_workers, dtype=np.uint8)
        # Per-batch reward signal breakdown (sum over all genomes in batch).
        batch_breakdown: dict[str, float] = {}

        # Async-pipeline executor with 1-step observation lag.
        # When `self.async_pipeline` is True:
        #   * At iter t, compute forward(stacked_obs) — uses obs_{t-1}
        #     because stacked_obs hasn't been updated by this iter's
        #     wait yet. Produces actions_t, log_probs_t.
        #   * Submit pool.step_all(actions_t) via ThreadPoolExecutor.
        #     Rayon workers begin emulating step t in the background.
        #   * Wait on the PENDING future (last iter's submission,
        #     which was step t-1). Main thread was free during that
        #     step_all — ready to do the next forward. Net: forward
        #     of iter t overlaps with step_all of iter t-1 on rayon.
        # The observation lag is intrinsic: actions_t is computed from
        # obs_{t-1} (the observation available pre-wait), applied
        # during step t, producing obs_t returned by pending.result().
        # Trajectory tuples still pair obs_that_fed_forward with
        # action+reward, so PPO / GAE math is consistent.
        step_executor = None
        pending_future = None
        pending_ctx: Optional[dict] = None
        if self.async_pipeline:
            import concurrent.futures as _cf
            step_executor = _cf.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="step-pipe"
            )

        # === PHASE 2: STEP LOOP ===
        import gc as _step_gc
        for step in range(self.max_episode_steps):
            if not self._running:
                break
            if all(done_flags):
                break
            # In-loop cycle sweep. Without this, ~64 s of step-loop
            # activity accumulates ~5 GB of cyclic refs (PyTorch forward
            # tensors, rust-frame numpy wrappers, per-step closures)
            # before the next gen-boundary gc runs — more than enough
            # to OOM over a few generations. gc.collect(0) targets
            # only the youngest cycle generation, which is where these
            # short-lived cycles live; measured cost on this loop
            # is <1 ms at step-count ~500.
            if step and step % 500 == 0:
                _step_gc.collect(0)
            if step and step % 200 == 0:
                import os as _os
                if _os.environ.get("NES_RSS_TRACE") == "1":
                    import resource as _res
                    _rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                    log.info("[rss-trace] gen step=%d rss=%.0f MB", step, _rss)

            actions = [0] * n
            active = [i for i in range(n) if not done_flags[i]]
            # Old log-probs captured per-step so PPO can compute the ratio
            # against the updated policy later.
            log_probs_old = [0.0] * n
            # `pre_obs` holds the observation each action was computed from —
            # needed later for trajectory logging. Instead of allocating N
            # separate copies we reuse the contiguous stack we build for the
            # forward pass (one allocation instead of N).
            pre_obs_stack: Optional[np.ndarray] = None
            pre_obs_idx: dict[int, int] = {}
            if active:
                with self._gen_timer.section("inference"):
                    # In-place write into the pre-allocated buffer; no
                    # np.stack malloc per step. Slice the prefix so
                    # the torch tensor shape matches `len(active)`.
                    # NOTE: tried full-batch vmap (always feed n rows)
                    # — broke training. Best fitness regressed 5× over
                    # 40 gens with the change in. The PRNG-advance
                    # difference (sampling n actions vs len(active))
                    # changed the entire training trajectory and the
                    # original gating was load-bearing. Keeping the
                    # original len(active)==len(nets) gate.
                    for b_i, a_i in enumerate(active):
                        batch_np_buffer[b_i] = stacked_obs[a_i]
                    batch_np = batch_np_buffer[: len(active)]
                    pre_obs_stack = batch_np
                    pre_obs_idx = {g: b for b, g in enumerate(active)}
                    if self.preprocess_f16:
                        # Host→device transfer at fp16 (half the bytes
                        # of fp32 and no CPU-side divide by 255). The
                        # on-device `.float()` expansion is cheap — a
                        # bandwidth-bound copy with no arithmetic unit
                        # involvement — but saves torch from running
                        # the GPU `/ 255.0` kernel we'd otherwise need,
                        # and ensures the conv weights' fp32 layout
                        # matches the input dtype outside autocast
                        # (vmap + functional_call bypass the autocast
                        # context so we can't rely on it to promote).
                        batch_t = (
                            torch.from_numpy(batch_np).to(self.device).float()
                        )
                    else:
                        batch_t = (
                            torch.from_numpy(batch_np).to(self.device).float().div_(255.0)
                        )
                    autocast_ctx = (
                        torch.autocast(device_type="mps", dtype=torch.float16)
                        if self.autocast_enabled and self.device.type == "mps"
                        else contextlib.nullcontext()
                    )

                # Fast path: parameter-stacked vmap forward. One GPU
                # dispatch for all len(active) genomes. CRITICAL: we do a
                # SINGLE bulk transfer to CPU at the end — each per-genome
                # .item() call would force an MPS stream sync, which
                # serialized the whole inference pipeline and was the
                # biggest remaining perf cost.
                # Fast path: parameter-stacked vmap forward. One GPU
                # dispatch for all len(active) genomes. CRITICAL: we do a
                # SINGLE bulk transfer to CPU at the end — each per-genome
                # .item() call would force an MPS stream sync, which
                # serialized the whole inference pipeline and was the
                # biggest remaining perf cost.
                used_vmap = False
                _forward_t0 = time.perf_counter_ns()
                if stacked_fn is not None and len(active) == len(nets):
                    # NO silent fallback here — if vmap fails mid-run
                    # it's a shape/autocast/architecture bug that
                    # silently 3-5×'d training slowness (historically
                    # caught by the old `try: ... except: stacked_fn=None`
                    # wrapper, which logged only at DEBUG and
                    # permanently disabled the fast path). Let the
                    # exception propagate so real bugs surface at dev
                    # time instead of quietly burning GPU cycles.
                    with torch.no_grad(), autocast_ctx:
                        x = batch_t.unsqueeze(1)
                        stacked_logits = stacked_fn(
                            stacked_params, stacked_buffers, x
                        )  # (N, 1, num_actions)
                        stacked_logits = stacked_logits.squeeze(1).float()
                        sampled, chosen_lp, log_probs_all, n_bad_rows = _safe_sample_from_logits(stacked_logits)
                    self._nan_rows_this_gen += n_bad_rows
                    sampled_cpu = sampled.cpu().numpy()
                    lp_cpu = chosen_lp.cpu().numpy()
                    log_probs_all_cpu = log_probs_all.cpu().numpy()
                    self._apply_sampled_actions(
                        active, sampled_cpu, lp_cpu, log_probs_all_cpu,
                        actions, log_probs_old, last_action_per_genome, step,
                    )
                    used_vmap = True

                if not used_vmap:
                    # Slow path — still bulk-transfers to avoid per-genome syncs.
                    with torch.no_grad(), autocast_ctx:
                        per_row_logits = []
                        for batch_idx in range(len(active)):
                            row = batch_t[batch_idx : batch_idx + 1]
                            per_row_logits.append(nets[active[batch_idx]](row))
                        # Concat on device, sample once via the defensive
                        # helper, transfer once.
                        cat_logits = torch.cat(per_row_logits, dim=0).float()
                        sampled, chosen_lp, log_probs_all, n_bad_rows = _safe_sample_from_logits(cat_logits)
                    self._nan_rows_this_gen += n_bad_rows
                    sampled_cpu = sampled.cpu().numpy()
                    lp_cpu = chosen_lp.cpu().numpy()
                    log_probs_all_cpu = log_probs_all.cpu().numpy()
                    self._apply_sampled_actions(
                        active, sampled_cpu, lp_cpu, log_probs_all_cpu,
                        actions, log_probs_old, last_action_per_genome, step,
                    )
                self._gen_timer.add(
                    "forward", time.perf_counter_ns() - _forward_t0
                )

            # Build the actions array for EVERY worker slot. Done GA
            # genomes get action=0 (idle step); unused workers in a
            # smaller-than-pool batch also get 0. Reuse the numpy
            # buffer allocated outside the loop — zero-copy handoff to
            # Rust's `PyReadonlyArray1<u8>` in `pool.step_all`.
            bitmasks: dict[int, int] = {}
            step_actions[:] = 0
            for i in range(n):
                if done_flags[i]:
                    continue
                bitmask = self._action_to_bitmask(actions[i])
                bitmasks[i] = bitmask
                step_actions[i + offset] = bitmask
            demo_pending = bool(offset)

            _wait_t0 = time.perf_counter_ns()
            if step_executor is not None:
                # Async pipeline with 1-step observation lag.
                # Snapshot the forward state just produced — it pairs
                # with THIS iter's step submission, and the
                # processing for those results will happen NEXT iter.
                new_ctx = {
                    "bitmasks": bitmasks,
                    "pre_obs_stack": pre_obs_stack,
                    "pre_obs_idx": pre_obs_idx,
                    "actions": actions,
                    "log_probs_old": log_probs_old,
                    "active_snapshot": list(active),
                    "done_snapshot": list(done_flags),
                }
                # Snapshot the 16-byte action buffer for the async
                # submission — the reusable `step_actions` is mutated
                # next iter before this future's worker thread enters
                # Rust. Copy is O(16) and still cheaper than the old
                # per-step list alloc.
                new_future = step_executor.submit(
                    self.pool.step_all, step_actions.copy()
                )
                if pending_future is None:
                    # First iter: nothing to process yet. Stash and go.
                    pending_future = new_future
                    pending_ctx = new_ctx
                    self._gen_timer.add(
                        "emulation_wait", time.perf_counter_ns() - _wait_t0
                    )
                    continue
                # Wait for the PREVIOUS step. Its results pair with
                # pending_ctx (forward state from one iter ago).
                results = pending_future.result()
                # Overwrite current-iter locals with pending's state
                # so the downstream processing loop uses the correct
                # action/log-prob/obs pairing.
                bitmasks = pending_ctx["bitmasks"]
                pre_obs_stack = pending_ctx["pre_obs_stack"]
                pre_obs_idx = pending_ctx["pre_obs_idx"]
                actions = pending_ctx["actions"]
                log_probs_old = pending_ctx["log_probs_old"]
                # Roll pending to the just-submitted step.
                pending_future = new_future
                pending_ctx = new_ctx
            else:
                results = self.pool.step_all(step_actions)
            self._gen_timer.add(
                "emulation_wait", time.perf_counter_ns() - _wait_t0
            )
            self._emit_frame_sink(results)

            # PRE-PASS: collect inputs for batched reward dispatch so
            # we can do one PyO3 round-trip instead of N. Audit pegged
            # this loop at 14.4k Python reward calls per gen; batching
            # saves the per-call argument-marshaling overhead (~5µs
            # each ≈ 50 ms/gen at 16 workers × 600 steps). Marginal
            # individually but the architecture is cleaner.
            #
            # Also capture prev_breakdown snapshots BEFORE the batched
            # compute mutates the per-genome state, so the narrator
            # diff (old → new) below works the same as the per-call
            # version.
            narrator_on = self._narrator is not None
            batch_genome_indices: list[int] = []
            batch_rams: list[bytes] = []
            batch_actions: list[int] = []
            batch_fns: list = []
            prev_breakdowns: dict[int, dict] = {}
            for r in results:
                worker_id = r.worker_id
                if worker_id < offset:
                    continue
                genome_i = worker_id - offset
                if genome_i >= n or done_flags[genome_i]:
                    continue
                batch_genome_indices.append(genome_i)
                batch_rams.append(r.ram_snapshot)
                batch_actions.append(bitmasks.get(genome_i, 0))
                batch_fns.append(reward_fns[genome_i])
                if narrator_on:
                    prev_breakdowns[genome_i] = dict(reward_fns[genome_i].breakdown)

            # Single Rust call for all active genomes' reward computation.
            if batch_rams:
                import nes_core as _nc
                # SMB peak-x tracking: the worker advanced frame_skip
                # NES frames inside one step_all call, but only the
                # final frame's RAM made it back to Python. A Mario
                # who reached x=1810 mid-jump and then died at x=1700
                # would silently lose the credit for crossing 1800
                # because compute() reads x from the final-frame RAM.
                # Pool now tracks peak x during the frame_skip window;
                # we read it here and overwrite the x bytes in the
                # RAM copies so MarioReward sees the peak. Cheap
                # (60 KB of byte-level copy across 30 workers per
                # step), and only relevant for SMB — for other games
                # the peak equals the final-frame x anyway.
                if self._is_smb_profile and self.pool is not None:
                    try:
                        max_xs = self.pool.peek_max_x_per_worker()
                    except Exception:
                        max_xs = None
                    if max_xs is not None:
                        patched_rams = []
                        for ram, gi in zip(batch_rams, batch_genome_indices):
                            # genome_i → worker_id == genome_i + offset.
                            # offset is 0 today (see comment ~line 2305).
                            wid = gi + offset
                            if wid < len(max_xs):
                                mx = int(max_xs[wid])
                                # Only patch when the peak is strictly
                                # higher than the final-frame x — no
                                # point burning a copy when nothing
                                # would change.
                                final_x = (
                                    256 * ram[0x006D] + ram[0x0086]
                                )
                                if mx > final_x:
                                    ba = bytearray(ram)
                                    ba[0x006D] = (mx >> 8) & 0xFF
                                    ba[0x0086] = mx & 0xFF
                                    patched_rams.append(bytes(ba))
                                    continue
                            patched_rams.append(ram)
                        batch_rams = patched_rams
                batch_results = _nc.compute_rewards_batch(
                    batch_fns, batch_rams, batch_actions,
                )
                rewards_by_genome = {
                    gi: res
                    for gi, res in zip(batch_genome_indices, batch_results)
                }
            else:
                rewards_by_genome = {}

            for r in results:
                _book_t0 = time.perf_counter_ns()
                worker_id = r.worker_id
                # `offset` is 0 today but preserved for any future
                # reserved-slot use case.
                if worker_id < offset:
                    continue
                genome_i = worker_id - offset
                if genome_i >= n or done_flags[genome_i]:
                    # Worker beyond the active batch — skip, its step
                    # was a no-op idle step.
                    continue
                i = genome_i
                if self._is_tile_mode:
                    # Tile mode: re-decode RAM into a tile feature vector,
                    # then push through the per-genome TileFeatureStacker
                    # so the policy sees a temporal stack (yumouwei recipe).
                    # When tile_frame_stack=1 this collapses to a single
                    # frame and the policy gets the same point-in-time
                    # vector as the pre-stack code did.
                    raw = self._tile_extractor.extract(r.ram_snapshot)
                    stacked_obs[i] = tile_stackers[i].push(raw)
                else:
                    stacked_obs[i] = stackers[i].push(r.frame, r.preprocessed)
                done = r.done
                # PERF: skip audio mixer push entirely when muted — the
                # most common state in headless runs.
                if self._audio_mixer is not None and self._audio_mixer.mode != "mute":
                    # Audio mixer slots are 1:1 with GA genomes, not
                    # pool worker_ids — pass genome_i so slot 0 in the
                    # mixer corresponds to the first evaluated genome.
                    self._audio_mixer.push_ram(i, r.ram_snapshot)
                    if r.audio.size > 0 and r.audio_rate > 0:
                        # r.audio_rate is stamped by the adapter as
                        # BASE_AUDIO_RATE × active pace multiplier, so
                        # the mixer's resampler consumes exactly as
                        # fast as paced workers produce (pitch-up at
                        # >1x instead of ring overflow). Do NOT scale
                        # it again here.
                        self._audio_mixer.push_audio(i, r.audio, r.audio_rate)
                # Pre-captured snapshot from the pre-pass above (None in
                # the headless / narrator-off case to skip the dict copy).
                prev_breakdown = prev_breakdowns.get(i) if narrator_on else None
                # Cached batched-compute result; identical semantics to
                # the prior per-call `reward_fns[i].compute(...)`.
                reward, rew_done, level_id = rewards_by_genome[i]
                cumulative[i] += reward
                # Cache the genome name (attribute lookup x2 -> x1) and
                # the success flag (called at most once per done step).
                genome_name = (
                    genomes[i].name if i < len(genomes) else f"worker-{worker_id}"
                )
                # `r.done` is the env-side done flag (Mario fell off the
                # bottom, hardware game-over screen). `rew_done` is the
                # reward function's done — fires on completion bonus,
                # death-state RAM read, or lives drop. For SMB-style
                # successes, completion fires via rew_done only; gating
                # ep_success on r.done alone misses every flag-touch and
                # the success counter (which drives BC capture + curriculum
                # promotion + narrator events) silently stays at zero.
                ep_success = (
                    reward_fns[i].episode_success()
                    if (bool(r.done) or rew_done)
                    else False
                )
                if narrator_on:
                    self._narrator.observe(
                        worker_id=worker_id,
                        genome_name=genome_name,
                        prev_breakdown=prev_breakdown,
                        new_breakdown=reward_fns[i].breakdown,
                        done=bool(r.done),
                        success=ep_success,
                    )
                # Depth progression check: did this step push the run's
                # deepest-point record? If so, narrate it (when enabled).
                depth_record = self._depth_tracker.observe(
                    ram=r.ram_snapshot,
                    worker_id=worker_id,
                    genome_name=genome_name,
                    generation=self.ga.generation,
                )
                if depth_record is not None:
                    if narrator_on:
                        from src.training.narrator import NarratorEvent
                        caption = f"🏁 {genome_name}: {depth_record['caption']}"
                        # Use time.monotonic() to match the narrator's own
                        # event timestamps (line 124 in narrator.py). Mixing
                        # wall-clock time.time() with monotonic produced
                        # inconsistent ordering metadata across the event
                        # stream; the difference is invisible most of the
                        # time but breaks any downstream consumer that
                        # treats timestamps as comparable.
                        self._narrator._events.append(NarratorEvent(
                            worker_id=worker_id,
                            genome_name=genome_name,
                            kind="depth_record",
                            caption=caption,
                            first_ever=True,  # always treat as banner-worthy
                            timestamp=time.monotonic(),
                        ))
                    # Queue a save-state RPC for after this gen's step
                    # loop completes. Can't do it inline: SaveStateCommand
                    # blocks the worker and we'd deadlock the hot step
                    # cycle. Deferred drain in _run_one_generation.
                    self._pending_state_snapshots.append((
                        int(worker_id),
                        tuple(depth_record["key"]),
                        str(depth_record["caption"]),
                    ))
                if not level_id_locked[i]:
                    level_ids[i] = level_id
                    level_id_locked[i] = True
                # In-place write into the pre-allocated trajectory
                # buffers. The trajectories[i] list stores VIEWS into
                # traj_obs (no per-step copy) + scalars — total alloc
                # per step across all genomes = 0. Elite culling later
                # on grabs slices [0:traj_lens[i]] and works with them
                # without additional copies (dropping non-elites just
                # drops the reference to the view — the underlying
                # buffer is reused next batch).
                step_idx = int(traj_lens[i])
                if step_idx < self.max_episode_steps:
                    if pre_obs_stack is not None and i in pre_obs_idx:
                        traj_obs[i, step_idx] = pre_obs_stack[pre_obs_idx[i]]
                    else:
                        traj_obs[i, step_idx] = np.ascontiguousarray(stacked_obs[i])
                    traj_actions[i, step_idx] = actions[i]
                    traj_rewards[i, step_idx] = reward
                    traj_log_probs[i, step_idx] = log_probs_old[i]
                    # No tuple/view append — data is already in the
                    # flat arrays. _reinforce_update consumes them
                    # directly. Bumps traj_lens[i] so slice accessors
                    # ([:traj_lens[i]]) see the correct length.
                    traj_lens[i] = step_idx + 1
                else:
                    # Hit the max-episode-steps budget — stop growing
                    # the trajectory but keep draining the step loop
                    # so downstream bookkeeping (sprite 0 hit, done
                    # flag) completes cleanly.
                    done_flags[i] = True
                if r.done or rew_done:
                    done_flags[i] = True
                    # Reuse the ep_success computed above (episode_success
                    # call is non-trivial for some reward fns; don't
                    # repeat it).
                    success_flags[i] = ep_success or reward_fns[i].episode_success()
                # Tell the Rust pool to short-circuit this worker on
                # subsequent step_all calls. Without this, a worker
                # that died at step 50 of 1500 burns ~5800 NES frames
                # of NOOP emulation across the rest of the episode.
                # Cleared automatically on next reset_all.
                if done_flags[i]:
                    self.pool.set_worker_done(i, True)
                self._gen_timer.add(
                    "bookkeeping", time.perf_counter_ns() - _book_t0
                )

        # === PHASE 3: FINALIZE ===
        # Async pipeline drain: the final iter's step_all submission
        # is still in flight (no subsequent iter pulled it). Wait for
        # it so the Rust pool state settles, but skip the Python-side
        # processing — loses at most one trajectory tuple per genome
        # vs the sync path (~0.1% of a 1000-step episode; invisible
        # to PPO's gradient since advantages normalize per-batch).
        # Executor is shut down so the worker thread doesn't linger.
        if step_executor is not None:
            if pending_future is not None:
                try:
                    pending_future.result(timeout=10.0)
                except Exception as exc:
                    log.warning(
                        "async pipeline drain failed: %s (pool state may "
                        "lag by one step; next batch's reset_all fixes it)",
                        exc,
                    )
                pending_future = None
                pending_ctx = None
            step_executor.shutdown(wait=False)

        # Merge every genome's breakdown into the batch-level dict.
        for fn in reward_fns:
            for k, v in getattr(fn, "breakdown", {}).items():
                batch_breakdown[k] = batch_breakdown.get(k, 0.0) + v

        # Drive the audio mixer's per-instance intensity from fitness
        # momentum: rank the batch's cumulative rewards and map the
        # top to the loudest, the bottom to the quietest. Neutral 1.0
        # when every genome scored identically so flat mixes sound
        # natural. This is "reactive scoring" for the stream — the
        # music scales with which genome is doing well right now.
        if self._audio_mixer is not None and n > 0:
            import numpy as _np
            rewards = _np.asarray(cumulative, dtype=_np.float32)
            if float(rewards.std() or 0.0) > 1e-3:
                # z-score into roughly [-2, 2], then map to [0.4, 1.5].
                z = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
                z = _np.clip(z, -2.0, 2.0)
                intensities = 0.95 + 0.3 * z  # -> roughly [0.35, 1.55]
                for i, val in enumerate(intensities):
                    self._audio_mixer.set_instance_intensity(i, float(val))
            else:
                for i in range(n):
                    self._audio_mixer.set_instance_intensity(i, 1.0)

        # Bundle the flat trajectory arrays for _reinforce_update.
        # All per-genome slices are sized via traj_lens[i], so
        # consumers don't need max_episode_steps pre-knowledge.
        traj_flat = {
            "obs": traj_obs,
            "actions": traj_actions,
            "rewards": traj_rewards,
            "log_probs": traj_log_probs,
            "lens": traj_lens,
        }
        return cumulative, success_flags, level_ids, traj_flat, batch_breakdown

    def _bc_demo_paths(self) -> list[Path]:
        return resolve_bc_demo_paths(self.bc_demo_path)

    def _bc_seed_cache_path(self) -> Optional[Path]:
        return _bc_seed_cache_path_for(
            demos=self._bc_demo_paths(),
            rom_path=self.rom_path,
            game_name=self.game_profile.get("name", "unknown"),
            action_space=self.action_space,
            frame_skip=self.frame_skip,
            encoder_kind=self.encoder_kind,
            checkpoint_dir=self.checkpoint_dir,
        )

    def _behavior_clone_seed(self) -> None:
        """Build a BC dataset from the demo, pre-train a net, seed the pop.

        Applies the game's reward function to the recording so imitation
        is reward-weighted (the expert's best moments dominate the loss,
        worst moments barely contribute). Caches the resulting seed to
        `bc_seed_<hash>.pt` so subsequent fresh runs with the same demo
        reuse it without re-training."""
        cache_path = self._bc_seed_cache_path()

        # Reuse a cached seed if the demo, ROM and profile all match.
        if cache_path is not None and cache_path.exists():
            try:
                seed = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                # Validate shapes against a freshly-constructed network
                # of the current architecture. If they don't match, the
                # cache is stale (e.g. an older run used a different
                # encoder) — log and fall through to rebuild rather
                # than crashing in load_state_dict.
                probe = self._make_network()
                missing, unexpected = probe.load_state_dict(seed, strict=False)
                # Surface unexpected keys as evidence of arch drift but
                # don't fail the run on them — strict=False already
                # tolerated them. The actual failure mode this guards
                # against is `RuntimeError: size mismatch`.
                if unexpected:
                    log.warning(
                        "BC cache had unexpected keys (likely stale): %s — rebuilding",
                        unexpected[:3],
                    )
                    raise RuntimeError("stale BC cache")
                log.info("Loaded cached BC seed from %s (skipping pretrain)", cache_path)
                seed_population_from_weights(self.ga.population, seed, noise_std=0.02)
                return
            except Exception as exc:
                log.warning(
                    "Failed to load BC cache %s (%s); re-pretraining.",
                    cache_path, exc,
                )
                # Move the bad cache out of the way so future runs of
                # the same arch can write a fresh one without us
                # tripping over this on every restart.
                try:
                    cache_path.rename(cache_path.with_suffix(".pt.stale"))
                except OSError:
                    pass

        reward_fn = self.reward_fn_factory()
        # Pass the trainer's frame_skip so the BC observation distribution
        # matches what the policy sees at runtime. Default in build_dataset
        # was 4 — using that here while training at frame_skip=16 produced
        # a stack of 4 frames spaced 4 apart (≈64ms apart) at BC time but
        # 4 frames spaced 16 apart (≈256ms apart) at runtime: literally
        # different observations from the policy's POV, so the warm-start
        # weights were applied to a distribution they had never seen.
        # Pass the resolved list of demos (may be multiple). build_dataset
        # concatenates per-demo (state, action, reward) tuples so BC
        # sees the full state distribution across all recordings.
        demos = self._bc_demo_paths()
        log.info("Behavioral cloning from %d demo(s): %s",
                 len(demos), ", ".join(p.name for p in demos))
        states, actions, rewards, demo_boundaries = build_dataset(
            rom_path=self.rom_path,
            demo_path=demos if len(demos) != 1 else demos[0],
            action_space=self.action_space,
            frame_skip=self.frame_skip,
            reward_fn=reward_fn,
            # Tile mode threads the extractor through so BC obs match
            # what the policy sees at runtime — same RAM-decoded
            # feature vector. Also thread the frame-stack size so BC
            # emits stacked features when the policy expects them
            # (without this, BC samples are 175-dim while the net
            # expects 700-dim and pretrain crashes on the mat-mul).
            tile_extractor=self._tile_extractor,
            tile_frame_stack=(
                self._tile_frame_stack if self._is_tile_mode else 1
            ),
        )
        if states.shape[0] < 16:
            log.warning("BC demo too short (%d pairs); skipping pretrain.", states.shape[0])
            return
        log.info(
            "BC dataset: %d pairs, total reward in demo = %.2f",
            states.shape[0], float(rewards.sum().item()),
        )
        net = self._make_network()
        # Hold out 10 % for validation so the BC log shows train/val
        # divergence — useful early signal for overfitting on small
        # demo recordings.
        final_loss = pretrain(
            net,
            (states, actions, rewards),
            epochs=self.bc_epochs,
            device=self.device,
            use_reward_weighting=True,
            val_fraction=0.1,
            # Tile-mode obs are signed int8 in [-128, 127], not pixel
            # uint8 — skip the /255 normalize so the network sees the
            # raw feature scale.
            normalize_obs=not self._is_tile_mode,
            # Per-demo boundaries so AWR weights don't leak rewards
            # backwards across demo breaks (multi-demo BC bug).
            episode_boundaries=demo_boundaries,
        )
        log.info("BC pretrain done; final loss=%.4f; seeding population.", final_loss)

        seed = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        if cache_path is not None:
            try:
                torch.save(seed, str(cache_path))
                log.info("Cached BC seed at %s", cache_path)
            except Exception as exc:
                log.warning("Could not write BC cache: %s", exc)
        seed_population_from_weights(self.ga.population, seed, noise_std=0.02)

    def _snapshot_pre_ppo_elite(
        self, best: Genome, elite_idx: list[int]
    ) -> Optional[tuple[float, dict]]:
        """Capture `best.state_dict` before PPO mutates it.

        Returns `(fitness, cloned_state_dict)` or `None` when freezing
        is disabled / PPO won't run. Tensors are detached and cloned
        on CPU so they're safe to keep around past the next forward
        pass that might dirty the live state_dict in place.
        """
        if not elite_idx or not self.freeze_pre_ppo_elite:
            return None
        return (
            float(best.fitness),
            {k: v.detach().cpu().clone() for k, v in best.state_dict.items()},
        )

    def _inject_pre_ppo_snapshot(
        self,
        snapshot: Optional[tuple[float, dict]],
        pop: list[Genome],
        best: Genome,
    ) -> None:
        """Write a pre-PPO snapshot into the weakest non-best slot.

        After PPO modifies `best.state_dict` in place, the only way
        the GA can keep the original winning policy across an
        evolve() call is to have a separate genome holding those
        same pre-PPO weights. We park the snapshot on the genome
        with the lowest current fitness — typically a fresh-mutation
        child that contributes nothing — and assign it the snapshot's
        original fitness so GA elitism preserves it on the next sort.
        On the following generation, the snapshot is re-evaluated
        like any other genome; if PPO regressed `best`, the snapshot
        re-scores high and resurfaces as elite, otherwise it drops
        out naturally.
        """
        if snapshot is None:
            return
        snap_fit, snap_state = snapshot
        weakest_i = min(range(len(pop)), key=lambda i: pop[i].fitness)
        if pop[weakest_i] is best:
            # Population must be size 1 — nowhere to park it. Skip
            # silently; the GA itself will preserve `best` regardless.
            return
        pop[weakest_i].state_dict = snap_state
        pop[weakest_i].fitness = snap_fit
        log.info(
            "  Frozen pre-PPO elite copied to slot %d "
            "(fitness %.1f preserved for next-gen elitism)",
            weakest_i, snap_fit,
        )

    def _save_bc_success_cache(self) -> None:
        """Persist the BC replay buffer to a single .npz on disk.

        Called every time a new success is captured. Atomic-write via
        tmp + rename so a Ctrl-C mid-save can never leave a half-written
        file. The encoded format is a flat dict where each trajectory is
        stored as four arrays prefixed by index:
            traj_{i}_obs, traj_{i}_actions, traj_{i}_rewards, traj_{i}_fitness
        """
        if not self._bc_replay_buffer:
            return
        try:
            data: dict = {}
            for i, entry in enumerate(self._bc_replay_buffer):
                # Backward compat: 4-tuple entries (no genome_id) still
                # appear when loading old caches into a buffer that
                # never re-recorded them. Normalize on write so v2 cache
                # always carries genome_id.
                obs, acts, rews, fit = entry[:4]
                gid = int(entry[4]) if len(entry) > 4 else -1
                data[f"traj_{i}_obs"] = obs
                data[f"traj_{i}_actions"] = acts
                data[f"traj_{i}_rewards"] = rews
                data[f"traj_{i}_fitness"] = np.array([fit], dtype=np.float32)
                data[f"traj_{i}_genome_id"] = np.array([gid], dtype=np.int32)
            data["count"] = np.array([len(self._bc_replay_buffer)], dtype=np.int32)
            # Schema metadata so the load path can validate that the
            # cache is compatible with the current trainer configuration
            # before injecting potentially-stale trajectories. Bump
            # cache_version on any backward-incompatible format change.
            # v1 = obs+actions+rewards+fitness + num_actions+obs_shape.
            # v2 = adds traj_{i}_genome_id (sentinel -1 means unknown).
            data["cache_version"] = np.array([2], dtype=np.int32)
            data["num_actions"] = np.array([self.num_actions], dtype=np.int32)
            obs_shape_tuple = self._obs_buffer_shape(1, 1)[2:]
            data["obs_shape"] = np.array(obs_shape_tuple, dtype=np.int64)
            # np.savez_compressed appends '.npz' to the filename if it
            # doesn't already end in '.npz' — so a temp path ending in
            # '.tmp' would write to <name>.tmp.npz on disk, breaking
            # the atomic-rename pattern. Anchor the temp path with .npz
            # itself so numpy writes to the literal path we name.
            tmp = self._bc_success_cache_path.with_suffix(".tmp.npz")
            try:
                np.savez_compressed(str(tmp), **data)
                tmp.replace(self._bc_success_cache_path)
            except Exception:
                # Clean up the partial tmp file so it doesn't accumulate
                # across crashes — np.savez_compressed leaves a half-
                # written .npz on disk if the process dies mid-write.
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                raise
        except Exception as exc:
            log.warning("BC success cache save failed: %s", exc)

    def _load_bc_success_cache(self) -> None:
        """Re-hydrate `_bc_replay_buffer` from disk on trainer init.

        Silent no-op when no cache exists. On format mismatch (e.g., an
        older cache from a different observation shape) we log and skip
        rather than crashing, so the cache file format can evolve
        without bricking the trainer.
        """
        if not self._bc_success_cache_path.exists():
            return
        try:
            arr = np.load(str(self._bc_success_cache_path), allow_pickle=False)
            count = int(arr["count"][0])
            # Schema validation: a cache from a previous run with a
            # DIFFERENT action space or observation feature_dim would
            # produce trajectories whose action indices are invalid
            # against the current policy network (e.g. cache has
            # action=7 from an 8-action run, current run has 6 actions
            # → BC pretrain trains policy to predict an output index
            # that no longer exists, silently corrupting training).
            # `cache_version` was added to the save format below; old
            # caches without it lack the shape metadata and we skip
            # them defensively rather than load-and-crash later.
            cache_version = int(arr.get("cache_version", np.array([0]))[0])
            if cache_version < 1:
                log.warning(
                    "BC success cache lacks schema version (legacy save). "
                    "Skipping load — will rebuild on next clear."
                )
                return
            # v1 → v2 added per-trajectory genome_id. v1 caches load fine
            # with sentinel -1; the buffer will start emitting v2 once new
            # clears arrive.
            cache_has_genome_id = cache_version >= 2
            saved_num_actions = int(arr["num_actions"][0])
            saved_obs_shape = tuple(arr["obs_shape"].tolist())
            cur_obs_shape = self._obs_buffer_shape(1, 1)[2:]
            if saved_num_actions != self.num_actions:
                log.warning(
                    "BC success cache action_space mismatch (saved=%d, "
                    "current=%d) — skipping load. Trajectories from a "
                    "different action layout would corrupt BC training.",
                    saved_num_actions, self.num_actions,
                )
                return
            if saved_obs_shape != cur_obs_shape:
                log.warning(
                    "BC success cache obs shape mismatch (saved=%s, "
                    "current=%s) — skipping load.",
                    saved_obs_shape, cur_obs_shape,
                )
                return
            for i in range(count):
                obs = arr[f"traj_{i}_obs"]
                acts = arr[f"traj_{i}_actions"]
                rews = arr[f"traj_{i}_rewards"]
                fit = float(arr[f"traj_{i}_fitness"][0])
                gid = (
                    int(arr[f"traj_{i}_genome_id"][0])
                    if cache_has_genome_id else -1
                )
                self._bc_replay_buffer.append((obs, acts, rews, fit, gid))
            log.info(
                "BC success cache loaded: %d trajectories (num_actions=%d, "
                "obs_shape=%s) from %s",
                count, saved_num_actions, saved_obs_shape,
                self._bc_success_cache_path,
            )
        except Exception as exc:
            log.warning(
                "BC success cache load failed (%s); starting with empty buffer",
                exc,
            )

    def _run_bc_replay(self, pop: list[Genome]) -> None:
        """Train a fresh policy via BC on the success-trajectory buffer
        and inject it as a new genome.

        Stitches every (obs, actions, rewards) trajectory in the buffer
        into a single supervised dataset, then runs `pretrain` to fit
        a fresh network. The result replaces the lowest-fitness slot
        in the population — the next generation's evaluation gives the
        BC-trained genome a real fitness, and if it's good (which it
        should be, since we trained on level-clear trajectories), the
        GA preserves it as elite.
        """
        if not self._bc_replay_buffer:
            return
        # In pure_ppo mode evolve() clones the highest-fitness slot to
        # every other slot — which is exactly what we WANT for BC if the
        # BC-trained network's claimed fitness beats the post-PPO elite's.
        # An earlier version of this code skipped BC replay entirely under
        # pure_ppo on the reasoning that "the weakest-slot injection would
        # be wiped" — wrong, because the injected fitness lifts the slot
        # to the top of the sort, then evolve() broadcasts it. Keeping
        # BC replay active in both modes; the success-buffer is the
        # mechanism we rely on to anchor clears across PPO drift.
        # Train BC only on the most-recent `bc_replay_train_window`
        # trajectories — the buffer itself archives more for safety,
        # but BC needs a COHERENT single-policy target. Aggregating
        # across all 16 buffer slots (each from a different gen's
        # elite) produced near-uniform action labels and BC loss
        # plateau'd at the random-policy entropy floor. See the
        # bc_replay_train_window comment in __init__.
        train_set = self._bc_replay_buffer[-self.bc_replay_train_window:]
        gids_used = [int(t[4]) if len(t) > 4 else -1 for t in train_set]
        fits_used = [float(t[3]) for t in train_set]
        log.info(
            "  BC replay: training on %d/%d buffered trajectories "
            "(genome_ids=%s, fitnesses=%s)",
            len(train_set), len(self._bc_replay_buffer),
            gids_used,
            [f"{f:.0f}" for f in fits_used],
        )
        # Concatenate the selected trajectories. Track per-trajectory
        # start indices so AWR weighting resets its discounted-return
        # accumulator at episode boundaries instead of leaking returns
        # backwards across them.
        all_obs = np.concatenate([t[0] for t in train_set], axis=0)
        all_acts = np.concatenate([t[1] for t in train_set], axis=0)
        all_rews = np.concatenate([t[2] for t in train_set], axis=0)
        episode_boundaries: list[int] = []
        running = 0
        for t in train_set[:-1]:  # last one's "end" isn't a boundary
            running += len(t[1])
            episode_boundaries.append(running)
        n = all_obs.shape[0]
        if n < 16:
            log.info("  BC replay skipped: only %d state-action pairs (need ≥16)", n)
            return
        # Tile-mode obs are int8 in [-128, 127]; cast to float for BC.
        # Pixel mode is uint8; pretrain handles the /255 normalize.
        if self._is_tile_mode:
            states = torch.from_numpy(np.ascontiguousarray(all_obs)).float()
            normalize_obs = False
        else:
            states = torch.from_numpy(np.ascontiguousarray(all_obs))
            normalize_obs = True
        actions = torch.from_numpy(all_acts.astype(np.int64))
        rewards = torch.from_numpy(all_rews.astype(np.float32))

        # Build a fresh network and run a few BC epochs.
        net = self._make_network()
        try:
            loss = pretrain(
                net,
                (states, actions, rewards),
                epochs=self.bc_replay_epochs,
                batch_size=64,
                lr=1e-3,
                device=self.device,
                use_reward_weighting=True,
                normalize_obs=normalize_obs,
                episode_boundaries=episode_boundaries,
            )
        except Exception as exc:
            log.warning("  BC replay pretrain failed: %s", exc)
            return

        # Inject into the weakest non-best slot, mirroring the freeze
        # snapshot pattern. Fitness is set to the train-set's mean
        # (not the whole buffer's) so the GA's first-gen evaluation
        # picks a number anchored to the policy that BC actually
        # fit, not to old/unrelated successes still archived in the
        # buffer.
        weakest_i = min(range(len(pop)), key=lambda i: pop[i].fitness)
        avg_train_fitness = sum(t[3] for t in train_set) / len(train_set)
        pop[weakest_i].state_dict = {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()
        }
        pop[weakest_i].fitness = float(avg_train_fitness)
        log.info(
            "  BC replay: trained on %d trajectories (%d state-action pairs), "
            "BC loss %.4f, injected to slot %d (fitness %.1f for next-gen elitism)",
            len(train_set), n, loss, weakest_i, avg_train_fitness,
        )
        # Reset the persistent Adam optimizer. The next _reinforce_update
        # will load the BC-injected genome's state_dict into self._ppo_net,
        # but the optimizer's accumulated m/v moments are tied to the
        # previous policy's gradient history. Applying that stale momentum
        # to freshly-trained BC weights pulls them right back toward the
        # pre-BC policy — undoing the anchor. Clearing _ppo_optimizer
        # forces a fresh build with zero moments, so Adam learns
        # parameter-specific learning rates against the new policy from
        # step 1 instead of fighting it.
        if self._ppo_optimizer is not None:
            self._ppo_optimizer = None
            log.info(
                "  BC replay: cleared persistent Adam optimizer "
                "(fresh momentum on next PPO update so BC weights "
                "aren't pulled back to the pre-BC policy)"
            )

    def _reinforce_update(
        self,
        genome: Genome,
        traj_obs: np.ndarray,
        traj_actions: np.ndarray,
        traj_rewards: np.ndarray,
        traj_log_probs: np.ndarray,
        traj_lens: np.ndarray,
        elite_indices: list[int],
    ) -> tuple[float, dict[str, float]]:
        """Policy-gradient update on the elite's weights using recorded rollouts.

        Consumes flat pre-allocated arrays (shapes: `traj_obs` =
        `(pop, max_steps, 4, 84, 84)`, `traj_actions/rewards/log_probs`
        = `(pop, max_steps)`, `traj_lens` = `(pop,)`). `elite_indices`
        picks which rows of those arrays to train on. No per-step tuple
        allocation — slices directly.

        Uses PPO's clipped surrogate with entropy bonus when
        `ppo_clip_eps > 0`; vanilla REINFORCE when clip=0.
        """
        if self._ppo_net is None:
            self._ppo_net = self._make_network()
            self._ppo_net.to(self.device)

        net = self._ppo_net
        # See trainer.py:424 for the strict=False rationale (backward compat
        # with checkpoints that predate the value_head).
        net.load_state_dict(genome.state_dict, strict=False)
        net.train()

        # Lazy-build the RND module on first use so it lands on the
        # current device. Predictor params are added to the same Adam
        # optimizer as the policy so a single backward+step covers both.
        # Encoder dispatches on observation kind: pixel mode uses the
        # CNN-backed `RND`; tile mode uses the MLP-backed `TileRND`.
        # Lifecycle owned by ExplorationController (Task 3); the GA path's
        # log prefix differs from the vanilla path's, preserved verbatim.
        ExplorationController(self).build_rnd(
            log_msg=(
                "RND intrinsic motivation enabled (%s): predictor=%d params, "
                "intrinsic_coef=%.3f, loss_coef=%.3f"
            )
        )

        if self._ppo_optimizer is None:
            self._ppo_optimizer = self._build_ppo_optimizer(net)

        optimizer = self._ppo_optimizer

        last_loss_scalar = 0.0
        # Running averages across the inner reinforce_steps loop. Only
        # the final epoch's stats survive; earlier epochs' values are
        # overwritten. This matches `last_loss_scalar` semantics — the
        # dashboard charts the LAST PPO step's stats per generation.
        ppo_stats_accum: dict[str, float] = {}
        for _ in range(self.reinforce_steps):
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=self.device)
            total_traj_items = 0
            # Per-component sums (over trajectories in this epoch). At
            # the end of the epoch we average to give the dashboard a
            # "what is the agent learning" telemetry per gen.
            sum_policy_loss = 0.0
            sum_value_loss = 0.0
            sum_entropy = 0.0
            sum_rnd_loss = 0.0
            sum_rnd_intrinsic = 0.0

            for g_idx in elite_indices:
                traj_len = int(traj_lens[g_idx])
                if traj_len < 2:
                    continue
                gamma = self.reinforce_gamma
                # Flat reward stream for GAE. Optionally symlog-transformed
                # (sign(r)*log1p(|r|)) to compress dynamic range — keeps
                # the critic from being dominated by a single large
                # exploration reward. Applied to a copy so the original
                # buffer is untouched (the dashboard still reports raw
                # reward components).
                full_r = traj_rewards[g_idx, :traj_len]
                if self.symlog_rewards:
                    full_r = np.sign(full_r) * np.log1p(np.abs(full_r))

                # Most recent window for the actual gradient step —
                # bounded so MPS memory stays sane.
                win_start = max(0, traj_len - self.reinforce_max_steps)
                window_offset = win_start
                states_np = traj_obs[g_idx, win_start:traj_len]
                actions = torch.from_numpy(
                    traj_actions[g_idx, win_start:traj_len].astype(np.int64)
                ).to(self.device)
                log_probs_old = torch.from_numpy(
                    traj_log_probs[g_idx, win_start:traj_len].copy()
                ).to(self.device).float()

                if self._is_tile_mode:
                    # Tile features are int8 in [-128, 127], small
                    # signed integers. Cast to float; do NOT divide by
                    # 255 (that's a pixel-mode normalization). The
                    # network's first LayerNorm + Linear handle scale.
                    states_t = (
                        torch.from_numpy(np.ascontiguousarray(states_np))
                        .to(self.device).float()
                    )
                elif self.preprocess_f16:
                    # fp16 HtoD transfer (half the bytes of fp32, zero
                    # /255 divide). On-device `.float()` promotes back
                    # to fp32 so the PPO forward/backward hits the
                    # optimizer's fp32 weight update path; autocast
                    # downcasts internally where safe.
                    states_t = torch.from_numpy(
                        np.ascontiguousarray(states_np)
                    ).to(self.device).float()
                else:
                    states_t = (
                        torch.from_numpy(np.ascontiguousarray(states_np))
                        .to(self.device).float().div_(255.0)
                    )
                # DrQ-style random-shift augmentation. Pads each frame
                # with replicate edges by `drq_pad` pixels then crops
                # back to the original size at a random offset. The crop
                # offset is shared across the trajectory so temporal
                # consistency within a frame stack is preserved (a stack
                # element shifted vs. its predecessors would teach the
                # network spurious motion). Tile mode skips this — DrQ
                # is a pixel-shift augmentation that doesn't apply to
                # discrete tile grids.
                if self.drq_aug and not self._is_tile_mode:
                    states_t = self._random_shift(states_t, pad=self.drq_pad)
                autocast_ctx = (
                    torch.autocast(device_type="mps", dtype=torch.float16)
                    if self.autocast_enabled and self.device.type == "mps"
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    # Actor-critic forward: one pass, both heads.
                    logits, values_pred = net.forward_ac(states_t)
                log_probs_all = F.log_softmax(logits.float(), dim=-1)
                log_probs_new = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
                values_pred = values_pred.float()

                # RND intrinsic-motivation bonus. Compute per-state
                # prediction error against the frozen target, mix the
                # detached error into the reward stream as exploration
                # bonus, and add the predictor MSE to the loss so the
                # predictor learns to mimic the target on visited states
                # (driving the bonus down on familiar inputs over time).
                rnd_loss_term = None
                rnd_intrinsic_t = None
                if self._rnd is not None:
                    rnd_per_sample = self._rnd(states_t)  # (T,) raw per-sample MSE
                    # Bonus = raw error / running std (detached). Compute
                    # BEFORE update_normalization so the divisor reflects
                    # prior batches, not the current one.
                    rnd_intrinsic_t = (
                        self._rnd.normalize_bonus(rnd_per_sample)
                        * self.rnd_intrinsic_coef
                    )
                    # Feed the RAW error (not the normalized bonus) to the
                    # running stats so reward_rms tracks the true error
                    # scale instead of err/std (self-referential).
                    self._rnd.update_normalization(
                        states_t.detach(),
                        rnd_per_sample.detach(),
                    )
                    rnd_loss_term = rnd_per_sample.mean()  # raw MSE — trains predictor

                # GAE-λ over the critic's value baseline — see
                # `src/training/gae.py` for the truncation-vs-natural-
                # termination bootstrap discussion and the load-bearing
                # MPSGraph-leak workaround.
                rewards_t = torch.from_numpy(
                    full_r[window_offset:]
                ).to(self.device).float()
                if rnd_intrinsic_t is not None:
                    # Length match is guaranteed: rnd_intrinsic_t has
                    # the same T as states_t which was sliced from
                    # `[win_start:traj_len]`, identical to the window
                    # applied to `full_r`.
                    rewards_t = rewards_t + rnd_intrinsic_t
                advantages, value_targets = _gae(
                    rewards=rewards_t,
                    values_pred=values_pred,
                    traj_len_full=int(traj_lens[g_idx]),
                    max_episode_steps=self.max_episode_steps,
                    gamma=gamma,
                    gae_lambda=self.gae_lambda,
                )

                # Per-trajectory MEAN (not sum) so a 1500-step elite
                # doesn't dominate the gradient over a 50-step elite.
                # The previous `policy_loss = -X.sum()` summed across the
                # window, then divided by the grand total of items across
                # all trajectories — meaning longer trajectories
                # contributed proportionally MORE to the gradient. Mean
                # here normalizes per-step within each trajectory; the
                # `total_loss / num_trajectories` step below averages
                # across trajectories. Net effect: each trajectory has
                # equal weight regardless of length.
                window_n = max(1, log_probs_new.numel())
                if self.ppo_clip_eps > 0 and log_probs_old is not None:
                    ratio = torch.exp(log_probs_new - log_probs_old)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps
                    )
                    policy_obj = torch.min(ratio * advantages, clipped * advantages)
                    policy_loss = -policy_obj.sum() / window_n
                else:
                    # Vanilla REINFORCE — either configured (clip<=0) or this
                    # trajectory was missing log_probs_old.
                    policy_loss = -(log_probs_new * advantages).sum() / window_n

                # Entropy bonus: encourages diverse action selection until
                # the advantage signal becomes strong enough to dominate.
                probs = log_probs_all.exp()
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                # Critic loss: predicted V vs GAE target (advantage +
                # baseline = one-step-TD return). MSE is the textbook
                # PPO choice but assumes normalized targets — when our
                # targets span [0, 5000+] (unnormalized symlogged
                # returns over 1500 steps), MSE produces loss in the
                # hundreds and the value gradient through the shared
                # trunk drowns the actor's policy gradient. Smooth-L1
                # (Huber) is the structural fix: quadratic for small
                # errors, linear for outliers, bounded gradient.
                if self.value_loss_kind == "mse":
                    value_loss = F.mse_loss(values_pred, value_targets)
                else:
                    value_loss = F.smooth_l1_loss(values_pred, value_targets)

                loss_traj = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                if rnd_loss_term is not None:
                    loss_traj = loss_traj + self.rnd_loss_coef * rnd_loss_term

                total_loss = total_loss + loss_traj
                total_traj_items += 1  # count trajectories, not items
                # Detach + .item() each component so the gradient graph
                # stays unaffected. Cheap; one host transfer per traj.
                sum_policy_loss += float(policy_loss.detach().item())
                sum_value_loss += float(value_loss.detach().item())
                sum_entropy += float(entropy.detach().item())
                if rnd_loss_term is not None:
                    sum_rnd_loss += float(rnd_loss_term.detach().item())
                    sum_rnd_intrinsic += float(rnd_intrinsic_t.detach().mean().item())

            if total_traj_items == 0:
                break
            # Average across trajectories. Combined with the per-step mean
            # above, this gives each trajectory equal voice regardless of
            # length.
            loss = total_loss / total_traj_items
            # NaN backstop (see vanilla path): skip a non-finite step
            # rather than let clip_grad_norm_ splash NaN across weights.
            if not torch.isfinite(loss):
                log.error("[ga_ppo] non-finite loss — skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.reinforce_grad_clip)
            optimizer.step()
            last_loss_scalar = float(loss.item())
            ppo_stats_accum = {
                "ppo_loss": last_loss_scalar,
                "ppo_policy_loss": sum_policy_loss / total_traj_items,
                "ppo_value_loss": sum_value_loss / total_traj_items,
                "ppo_entropy": sum_entropy / total_traj_items,
            }
            if self._rnd is not None and total_traj_items > 0:
                ppo_stats_accum["rnd_loss"] = sum_rnd_loss / total_traj_items
                ppo_stats_accum["rnd_intrinsic_avg"] = (
                    sum_rnd_intrinsic / total_traj_items
                )

        # Copy the updated weights back to the genome so the GA keeps them.
        genome.state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        return last_loss_scalar, ppo_stats_accum

    def _recurrent_ppo_update(
        self, net, optimizer, obs_buf, action_buf, log_prob_buf,
        advantages_norm, value_targets, done_buf, num_envs, rollout_steps,
    ) -> tuple[float, float, float, float, float]:
        """K-epoch PPO update for the recurrent (GRU) tile policy.

        The stateless update shuffles (T×N) samples randomly; a recurrent
        policy cannot — the hidden state must flow in time order. So we
        minibatch over ENVS (columns), replay each env's full T-step
        trajectory through `forward_ac_recurrent` (truncated BPTT), and
        reset the hidden state on every `done` using the SAME `done_buf`
        the rollout used — so the replayed hidden trajectory matches the
        one that produced the actions. Returns the last
        (policy_loss, value_loss, entropy, total_loss, rnd_loss).
        """
        dev = self.device
        obs_seq = torch.from_numpy(obs_buf).to(dev).float()              # (T,N,feat)
        done_seq = torch.from_numpy(done_buf.astype(np.float32)).to(dev)  # (T,N)
        act_seq = torch.from_numpy(action_buf.astype(np.int64)).to(dev)   # (T,N)
        logp_old_seq = torch.from_numpy(log_prob_buf).to(dev).float()     # (T,N)
        adv_seq = torch.from_numpy(advantages_norm).to(dev).float()       # (T,N)
        tgt_seq = torch.from_numpy(value_targets).to(dev).float()         # (T,N)
        feat = obs_seq.shape[-1]
        # Envs per minibatch = gradient batch of env_mb full trajectories,
        # each replayed in full through the GRU (the rollout_steps-step
        # unroll graph is retained for backward). Dedicated env-count knob
        # (see recurrent_env_minibatch); clamped to the env count.
        env_mb = max(1, min(num_envs, int(self.recurrent_env_minibatch)))
        lp = lv = le = ll = lr = 0.0
        for _epoch in range(self.reinforce_steps):
            env_perm = np.random.permutation(num_envs)
            for s in range(0, num_envs, env_mb):
                envs = env_perm[s:s + env_mb]
                if envs.size < 1:
                    continue
                envs_t = torch.from_numpy(envs).to(dev)
                h = net.initial_hidden(int(envs.size), dev)
                logits_steps, value_steps = [], []
                for t in range(rollout_steps):
                    ot = obs_seq[t].index_select(0, envs_t)        # (mb,feat)
                    lg, v, h_next = net.forward_ac_recurrent(ot, h)
                    logits_steps.append(lg)
                    value_steps.append(v)
                    # Reset hidden for envs whose episode ended at step t
                    # (mirrors the rollout's done-mask exactly).
                    keep = (1.0 - done_seq[t].index_select(0, envs_t)).unsqueeze(-1)
                    h = h_next * keep
                # Flatten time-major (matches the *_seq.index_select(1).reshape).
                logits_b = torch.cat(logits_steps, dim=0)          # (T*mb, A)
                values_b = torch.cat(value_steps, dim=0)           # (T*mb,)
                act_b = act_seq.index_select(1, envs_t).reshape(-1)
                logp_old_b = logp_old_seq.index_select(1, envs_t).reshape(-1)
                adv_b = adv_seq.index_select(1, envs_t).reshape(-1)
                tgt_b = tgt_seq.index_select(1, envs_t).reshape(-1)
                loss, policy_loss, value_loss, entropy = ppo_losses(
                    logits_b, values_b, act_b, logp_old_b, adv_b, tgt_b,
                    clip_eps=self.ppo_clip_eps,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                    value_loss_kind=self.value_loss_kind,
                )
                if self._rnd is not None:
                    states_b = obs_seq.index_select(1, envs_t).reshape(-1, feat)
                    rnd_loss = self._rnd(states_b).mean()
                    loss = loss + self.rnd_loss_coef * rnd_loss
                    lr = float(rnd_loss.detach().item())
                optimizer.zero_grad()
                # NaN backstop (see vanilla path).
                if not torch.isfinite(loss):
                    log.error("[ga_ppo] non-finite loss — skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.reinforce_grad_clip
                )
                optimizer.step()
                lp = float(policy_loss.detach().item())
                lv = float(value_loss.detach().item())
                le = float(entropy.detach().item())
                ll = float(loss.detach().item())
        return lp, lv, le, ll, lr

    def _maybe_resume_vanilla_ppo(
        self, net, optimizer, *, fresh_start: bool = False,
    ) -> int:
        """Thin shim: the resume scan/load lives in
        `CheckpointManager.resume` (trainer-decomposition plan, Task 1).

        Kept so pre-existing direct callers stay valid. Delegates to a
        CheckpointManager built on this trainer, which stages the same
        `_vppo_resumed_from_iter` / `_pending_*` / `_gx_counts` attributes on
        the trainer (via its `self.trainer` ref) and returns the absolute
        iteration offset to continue checkpoint numbering from (0 = nothing
        loaded). `fresh_start` skips the scan so a from-scratch experiment
        isn't silently continued off a stale checkpoint.
        """
        return CheckpointManager(
            self, checkpoint_dir=self.checkpoint_dir, device=self.device,
        ).resume(net, optimizer, fresh_start=fresh_start)

    def _load_smb_curriculum_from_disk(
        self, *, fresh_start: bool = False,
    ) -> tuple[list, list, int]:
        """Thin shim: the SMB save-state curriculum disk-load now lives in
        `Curriculum.load_from_disk` (trainer-decomposition plan, Task 4). The
        body relocated there verbatim; this shim stays so pre-existing
        direct callers (`test_char_curriculum_glue`, `test_vanilla_ppo_fresh_
        curriculum`, and the conductor) reach the identical logic. See
        `Curriculum.load_from_disk` for the full contract.
        """
        return Curriculum(self).load_from_disk(fresh_start=fresh_start)

    def _run_vanilla_ppo(self, num_iters: int, fresh_start: bool = False) -> None:
        """Vanilla PPO with N parallel envs — the literature recipe.

        Single policy network, N envs collecting parallel rollouts of
        fixed length, batched GAE across all envs, K-epoch minibatched
        PPO update on the aggregated data. No GA, no population, no
        BC injection. Matches yumouwei/uvipen.

        Why a separate path from `_run_one_generation`: the GA modes
        treat the 30 workers as a population of 30 DIFFERENT policies
        whose data gets mixed into PPO's gradient — that mix violates
        PPO's "stable policy across updates" assumption and is the
        empirical reason ga_ppo and pure_ppo plateaued in our 2-day
        investigation. Here all 30 workers run the SAME policy and
        their data is one big homogeneous batch.

        Iteration shape:
          1. Reset all envs (fresh episode boundary for the batch)
          2. Rollout: for rollout_steps × num_envs steps, forward the
             single policy and step all envs in parallel; record
             (obs, action, reward, value, log_prob, done) for each
             (env, step).
          3. Bootstrap V(s_T) from the final observation per env.
          4. GAE-λ sweep backward per env, masking across done boundaries.
          5. Globally normalize advantages across the (env × step) batch.
          6. K epochs of minibatched PPO update on the flattened data.
          7. Periodic checkpoint + metrics emission.
        """
        assert self.pool is not None
        num_envs = self.num_instances
        rollout_steps = self.rollout_steps

        # Exploration lifecycle owner (RND build, count bonus, generic
        # Go-Explore archive). Built once; used for the RND build below and
        # the per-step count-bonus / go_explore record hooks in the loop.
        _exploration = ExplorationController(self)

        # Persistent net + optimizer (same machinery the GA-mode PPO
        # update uses; reused so the existing fix infrastructure
        # — Adam state, lr, etc. — applies uniformly).
        if self._ppo_net is None:
            self._ppo_net = self._make_network()
            self._ppo_net.to(self.device)
        net = self._ppo_net
        # Demo-anchor bank (DQfD-style; see knobs at __init__). Built once
        # per run — validates obs width against the net's input dim and
        # rejects buffer-aliased banks (all-identical rows).
        if self.demo_anchor_enabled and self._is_tile_mode:
            from src.training.demo_bank import DemoBank
            # Provenance gate (CLAIMS.md): Learned-ledger training only
            # consumes allowlisted demo banks. The allowlist is
            # authoritative — sidecars are advisory and have mislabeled
            # once already. Fail LOUD: silently training on a tainted
            # bank would poison the ledger's central claim.
            _allow_path = Path("configs/demo_allowlist.txt")
            if _allow_path.exists():
                _allowed = {
                    ln.strip() for ln in _allow_path.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")
                }
                _bad = [p for p in self.demo_anchor_paths
                        if str(p) not in _allowed]
                if _bad:
                    raise ValueError(
                        "demo_anchor_paths not on configs/demo_allowlist"
                        f".txt (see CLAIMS.md): {_bad}"
                    )
            self._demo_bank = DemoBank.from_npz(
                self.demo_anchor_paths, self._tile_feature_dim,
                self.num_actions, self.device,
            )
            log.info(
                "[vanilla_ppo] DEMO ANCHOR on: %d pairs from %s; coef "
                "%.3f -> %.3f over %d iters, margin %.2f, mb %d",
                self._demo_bank.n, self.demo_anchor_paths,
                self.demo_anchor_coef0, self.demo_anchor_final,
                self.demo_anchor_decay_iters, self.demo_anchor_margin,
                self.demo_anchor_mb,
            )
        # RND intrinsic exploration (opt-in via reinforce.rnd_intrinsic_coef).
        # Same module + knobs the GA path uses; lazily built so it lands
        # on the current device. Predictor params train in the same Adam
        # optimizer as the policy. Canonical SMB-PPO is exploration-
        # limited (entropy pins near ln(A), policy settles on a single
        # safe-failing behavior); RND's novelty bonus keeps advantages
        # alive past that wall.
        _exploration.build_rnd(
            log_msg=(
                "[vanilla_ppo] RND enabled (%s): predictor=%d params, "
                "intrinsic_coef=%.3f, loss_coef=%.3f"
            )
        )
        if self._ppo_optimizer is None:
            self._ppo_optimizer = self._build_ppo_optimizer(net)
        optimizer = self._ppo_optimizer

        # ===== ReDo startup (V27_FRESH_RECOVERY AMENDMENT 1, B6) =====
        # The two startup lines below are exact grep targets for the V2
        # preflight: a treatment run must show `[redo] ENABLED tau=...`
        # and is VOID if its log contains `[redo] disabled`. Do not
        # reword. An armed-but-unsupported architecture therefore prints
        # `[redo] disabled` (plus a loud warning) so the preflight voids
        # the run instead of a silent no-op passing as a treatment.
        redo_on = self.redo_enabled
        if redo_on and not (
            self._is_tile_mode and not self._recurrent
            and all(
                hasattr(net, a)
                for a in ("fc1", "norm1", "fc2", "norm2", "actor", "critic")
            )
        ):
            log.warning(
                "[redo] configured but UNSUPPORTED here: ReDo needs the "
                "feedforward tile policy (fc1/norm1/fc2/norm2 + actor/"
                "critic heads); encoder=%s recurrent=%s net=%s",
                self.encoder_kind, self._recurrent, type(net).__name__,
            )
            redo_on = False
        if redo_on:
            log.info(
                "[redo] ENABLED tau=%g every_iters=%d scope=fc1,fc2 "
                "sample=%d reset_moments=%s mode=%s k=%d recycle_scope=%s",
                self.redo_tau, self.redo_check_every_iters,
                self.redo_sample_batch,
                "true" if self.redo_reset_optimizer_moments else "false",
                # V32 §3 B1 grep target. `scope=fc1,fc2` above is the
                # module's scope and is pinned by the B6 evidence test;
                # `recycle_scope` is the rule's EFFECTIVE scope, which is
                # fc2-only under the rank rule. Under mode=bottom_k the
                # `tau=` field above is not read by the selection path —
                # it is logged for provenance, not as an operating point.
                self.redo_mode, self.redo_bottom_k,
                "fc2" if self.redo_mode == _REDO_SELECT_BOTTOM_K
                else "fc1,fc2",
            )
        else:
            log.info("[redo] disabled")

        # Absolute iteration offset for resume-safe checkpoint naming.
        # The loop counter `it` always restarts at 0, so without an
        # offset a resumed run rewrites vanilla_ppo_iter_00010.pt,
        # _00020.pt, … on top of the prior run's same-named files —
        # which is how a collapsed resume silently destroyed a good
        # run's checkpoints. Continue numbering from the resumed iter
        # so checkpoints BEFORE the resume point are never overwritten.
        # `fresh_start` (GUI "Resume" unticked / headless --no-resume)
        # skips the scan entirely so a from-scratch experiment isn't
        # silently continued off a stale checkpoint.
        ckpt = CheckpointManager(
            self,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
            game_name=self.game_profile.get("name"),
        )
        iter_offset = ckpt.resume(
            net, optimizer, fresh_start=fresh_start,
        )
        # RND is built lazily ABOVE (_exploration.build_rnd), before this
        # resume() call populates _pending_rnd_state from the checkpoint —
        # so build_rnd's own inline apply_pending_rnd_state() ran too early
        # and no-opped. t._rnd is non-None now, so apply_pending_rnd_state's
        # build-guard won't fire again on its own; call it explicitly here
        # so a resumed run's staged predictor/target weights and obs/reward
        # running stats actually land on the live RND module instead of
        # being silently discarded for the rest of the process.
        _exploration.apply_pending_rnd_state()

        # Surface an auto-resume for the whole run: the log line alone is
        # easy to miss, and a silent resume of a supposedly-fresh run has
        # invalidated real experiments. Carried on every metrics emission
        # so the GUI dashboard keeps showing "Resumed from iter N";
        # absent entirely on a genuinely fresh run.
        resume_metrics = (
            {"resumed_from_iter": self._vppo_resumed_from_iter}
            if self._vppo_resumed_from_iter else {}
        )

        # Reproducibility manifest: ROM + start state + seed + git commit
        # + pinned hyperparameters, written to the checkpoint dir. With
        # the code at that commit and the run's metrics.jsonl, the result
        # is reproducible/citable. Best-effort — never block training.
        ckpt.write_manifest(
            game=self.game_profile.get("name"),
            rom_path=self.rom_path,
            start_state_path=self.start_state_path,
            seed=self.seed,
            profile=self.game_profile,
            num_envs=num_envs,
            frame_skip=self.frame_skip,
        )

        # Per-env reward functions and stackers.
        reward_fns = [self.reward_fn_factory() for _ in range(num_envs)]
        for fn in reward_fns:
            fn.reset()

        # SMB multi-level "save-state curriculum".
        #
        # Background: vanilla_ppo's iter boundary unconditionally cold-
        # boots every env back to 1-1 via pool.reset_all(). Even when
        # the reward fn permits the episode to continue past the 1-1
        # flagpole (so SMB's level-transition cutscene advances the
        # game to 1-2 naturally), that 1-2 state never persists into
        # subsequent rollouts — the next iter's reset_all snaps every
        # worker back to 1-1 cold boot. Empirically (iter 150-250
        # multi-level run) Mario clears 1-1 ~70% of the time but the
        # policy never trains a single 1-2 frame.
        #
        # Fix: the moment ANY env's RAM shows W1-L2 mid-rollout,
        # snapshot the worker state and persist it to disk as
        # `world1-2_spawn.state`. On every iter boundary, AFTER the
        # cold-boot reset_all, restore half the env pool from that
        # save state. Half the population trains 1-1 from cold boot,
        # half trains 1-2 from the captured spawn. Both data streams
        # feed the same PPO update.
        #
        # This is the canonical "save-state curriculum" recipe used
        # in NES RL when natural-progression exploration burden is
        # prohibitive (clearing 1-1 + surviving cutscene + playing
        # an unseen level is a multi-stage exploration problem; the
        # save state collapses it to two single-stage problems that
        # the same policy network can learn in parallel).
        # SMB curriculum: whole-pool level progression.
        #
        # Stage gating: at every iter boundary, all 30 envs start
        # together at the current stage's anchor area-byte. Once the
        # pool *consistently* clears the current stage (>=80% of envs
        # reach the NEXT area byte for `consec_iters` consecutive
        # iters), capture a save state from one of those envs at the
        # next level and advance the stage. Then ALL future iter
        # resets warm-start from that state. Continue until the game
        # is finished.
        #
        # Why area bytes instead of "level N+1": SMB's $0760 byte is
        # an internal area index, not displayed level. byte=0 is 1-1
        # main; byte=2 is 1-2 underground main; byte=5 is 1-3 etc.
        # The curriculum advances on byte changes (any time an env's
        # max-byte exceeds the current stage anchor, we count it as
        # "made progress past the stage").
        #
        # The initial `states`/`anchors`/`stage` come from
        # `_load_smb_curriculum_from_disk` below (a fresh run starts at
        # stage 0 unless opted in); it seeds the stage-0 cold-boot entry.
        smb_stage_clear_history: list[int] = []  # rolling: clears past current anchor
        # `smb_pending_capture` is updated mid-rollout the moment an
        # env transitions to area-byte > current_anchor while alive.
        # If the advance threshold is met at iter boundary, this is
        # what we promote — guarantees the captured state is from a
        # LIVING env at level entry, not a dead env's frozen state.
        smb_pending_capture: Optional[tuple[int, bytes]] = None
        # Advance when the ROLLING-MEAN fraction of current-stage envs
        # clearing past the anchor reaches SMB_ADVANCE_PCT over the last
        # SMB_ADVANCE_WINDOW iters. A rolling mean (not "N consecutive
        # iters >= pct") is deliberate: the per-iter clear fraction is
        # noisy (observed oscillating 35-55% even after the agent has
        # mastered the level), so a consecutive-all-above gate almost
        # never fires and the curriculum stalls despite real mastery.
        # The mean smooths that variance. 0.5 is a clear majority; the
        # mixed-stage warm-start keeps ~40% of envs on earlier stages so
        # advancing at 50% reliability is safe (the gated+mixed+anti-
        # collapse trio replaced the original eager "first env" advance
        # that caused the entropy-melt collapse).
        SMB_ADVANCE_PCT = 0.50
        SMB_ADVANCE_WINDOW = 5
        smb_pastfrac_history: list[float] = []  # recent per-iter clear fractions
        # Mixed-stage warm-start: fraction of the pool that warm-starts
        # at the CURRENT (hardest) stage each iter; the rest spread over
        # earlier stages down to cold-boot stage 0. Whole-pool warm-start
        # (frac=1.0) lands every env on a level it may not reliably play,
        # so a single hard stage yields uniform failure that the entropy
        # bonus turns into policy collapse. Keeping a spread preserves
        # earlier-level competence and keeps advantages informative.
        SMB_CURRENT_STAGE_FRAC = 0.6
        # Which curriculum stage each env is warm-started to this iter,
        # and that env's stage-start observation (for mid-rollout auto-
        # reset re-seeding). Populated at the iter-boundary warm-start.
        env_stage = np.zeros(num_envs, dtype=int)
        stage_seed_results: list = [None] * num_envs
        # Anti-collapse guard state. The failure mode that ate the last
        # run was an entropy melt: ppo_entropy climbing toward the
        # uniform-random max (ln num_actions) while fitness flatlined,
        # never recovering. Detect it by ENTROPY (stage-independent,
        # unlike fitness which legitimately drops on a harder stage) and
        # roll back to the last healthy snapshot + reset the optimizer.
        _entropy_max = math.log(max(2, len(self.action_space)))
        SMB_ENTROPY_COLLAPSE_FRAC = 0.90  # entropy above this frac of max = melting
        SMB_ENTROPY_HEALTHY_FRAC = 0.75   # snapshot only when entropy below this
        SMB_COLLAPSE_PATIENCE = 5         # consecutive melting iters before rollback
        best_net_snapshot: Optional[dict] = None
        best_snapshot_fitness = float("-inf")
        collapse_strikes = 0
        # Restore the anti-collapse rollback baseline from a resumed
        # checkpoint so a post-restart entropy melt can still be rolled
        # back, instead of measuring collapse against -inf until a new
        # healthy snapshot re-forms.
        _ac = getattr(self, "_pending_anticollapse", None)
        if _ac:
            best_net_snapshot = _ac.get("best_net_snapshot") or None
            best_snapshot_fitness = float(
                _ac.get("best_snapshot_fitness", float("-inf"))
            )
            collapse_strikes = int(_ac.get("collapse_strikes", 0))
            self._pending_anticollapse = None
        # Load any previously-saved curriculum states from disk so a
        # restart resumes the curriculum mid-game instead of starting
        # over from 1-1. Anchor byte is stored in a sidecar JSON so the
        # capture gate (`byte > anchor + 1`) doesn't degrade on reload.
        # A fresh run (fresh_start) ignores saved state and starts at
        # stage 0 unless reinforce.inherit_curriculum_on_fresh is set;
        # the saved files are left untouched either way. `smb_curriculum_dir`
        # is still needed below for writing newly captured stages.
        smb_curriculum_dir = self.checkpoint_dir / "smb_curriculum"
        (
            smb_curriculum_states,
            smb_curriculum_anchors,
            smb_curriculum_stage,
        ) = self._load_smb_curriculum_from_disk(fresh_start=fresh_start)
        if smb_curriculum_stage > 0:
            log.info(
                "[vanilla_ppo] curriculum resumed at stage %d (%d save "
                "states loaded from %s)",
                smb_curriculum_stage, smb_curriculum_stage,
                smb_curriculum_dir,
            )
        else:
            log.info(
                "[vanilla_ppo] curriculum starting fresh at stage 0 "
                "(all envs train 1-1; advance when rolling-mean clear "
                "fraction over %d iters >= %.0f%%)",
                SMB_ADVANCE_WINDOW, SMB_ADVANCE_PCT * 100,
            )

        # === PRIORITIZED LEVEL REPLAY (generalist, Phase-B) ===
        # ONE policy over a FIXED set of level entrances, each env sampled by
        # inverse-recent-success weight. All decision logic is in the isolated,
        # unit-tested src/training/plr.py; this is thin glue. Mutually exclusive
        # with the staged curriculum / ladder / consolidate modes (PLR requires
        # reinforce.smb_curriculum: false). See configs/mario_generalist_w1.yaml.
        from src.training import plr as _plr_mod
        plr_ctx = _plr_mod.build_plr_context(self.game_profile, num_envs)
        plr_on = plr_ctx is not None
        if plr_on:
            if self._smb_curriculum_active:
                raise ValueError(
                    "reinforce.plr_enabled requires smb_curriculum: false "
                    "(PLR replaces the staged curriculum / ladder / consolidate)."
                )
            # Route each env's warm-start through the existing per-env plumbing:
            # env_stage[i] indexes smb_curriculum_states, so populating it with
            # the PLR entry blobs (index 0 == None cold boot) makes the iter-
            # boundary warm-start AND the mid-rollout auto-reset reload each
            # env's assigned level with zero new load bookkeeping.
            smb_curriculum_states = list(plr_ctx.states)
            log.info(
                "[vanilla_ppo] PLR ENABLED: train=%s index_map=%s holdout=%s",
                plr_ctx.train_labels, plr_ctx.level_to_idx,
                sorted(plr_ctx.holdout.keys()),
            )

        # === SUB-STAGE LADDER (SMB one-shot campaign, Lane 4) ===
        # When the profile declares `reinforce.substage_ladder.enabled`,
        # the flat scalar area-byte curriculum above is REPLACED by an
        # ordered (world, area, x-bucket) rung ladder: `smb_curriculum_stage`
        # becomes the FRONTIER rung order, `smb_curriculum_states` is
        # re-indexed by rung order (disk seeds loaded up front, deeper rungs
        # live-filled by capture), the advance gate measures `region_of(ram)`
        # sub-stage orders, and warm-start is the three-way Frontier /
        # Retention / Spread partition. Everything is gated on `ladder_on`,
        # so non-campaign SMB profiles are byte-for-byte untouched. All the
        # DECISIONS route through the isolated, unit-tested
        # `oneshot_curriculum` module; this block is thin glue.
        _rl_cfg = self.game_profile.get("reinforce", {}) or {}
        # Per-episode metrics sidecar (episodes.jsonl via
        # MetricsSink.emit_episode()). OFF by default: the call site is
        # inside the innermost per-worker step loop, right next to the
        # hot rollout path, so an unconditional emit here would add a
        # file write per completed episode (tens of times a second at
        # num_envs~60-121) to every run instead of only the ones that
        # asked for per-episode/per-worker granularity. metrics.jsonl's
        # per-generation aggregates remain the default source of truth;
        # opt in with `reinforce.episode_metrics: true` when you need to
        # ask "which worker produced the outlier" instead of just
        # "generation N looked off."
        episode_metrics_on = bool(_rl_cfg.get("episode_metrics", False))
        _substage_cfg = dict(_rl_cfg.get("substage_ladder", {}) or {})
        # Level-scoped consolidation ("weld ONE level"), a DISTINCT mode from
        # the frontier ladder — see the setup block below. It trains 100% of the
        # pool inside a single target level under a hard cold no-regression gate
        # on a protect list; there is no advance / spread / Go-Explore. Whole-
        # distribution consolidation collapsed the cold chain twice on the tile
        # scout; this scopes the weld to one level with the cold probe as a
        # hard gate. Mutually exclusive with the frontier ladder — if a profile
        # enables both, consolidation wins and the ladder advance is OFF.
        _clevel_cfg = dict(_rl_cfg.get("consolidate_level", {}) or {})
        consolidate_on = bool(
            self._smb_curriculum_active and _clevel_cfg.get("enabled", False)
        )
        ladder_on = bool(
            self._smb_curriculum_active and _substage_cfg.get("enabled", False)
            and not consolidate_on
        )
        if consolidate_on and _substage_cfg.get("enabled", False):
            log.warning(
                "[vanilla_ppo] both substage_ladder and consolidate_level are "
                "enabled; consolidate_level wins (frontier advance is OFF)."
            )
        ladder = None
        oc = None            # oneshot_curriculum module (bound when ladder_on)
        region_of = None     # bound from smb_substage_ladder when ladder_on
        smb_sequential_cell = None
        OFF_LADDER = -1
        LADDER_SIZE = 0
        _adv_cfg = dict(_rl_cfg.get("advance", {}) or {})
        _ws_cfg = dict(_rl_cfg.get("warm_start", {}) or {})
        _cold_cfg = dict(_rl_cfg.get("cold_eval", {}) or {})
        _cons_cfg = dict(_rl_cfg.get("consolidate", {}) or {})
        _forget_cfg = dict(_cold_cfg.get("forgetting", {}) or {})
        # Non-cheating wavefront potential reward (PBRS). Loads a distance-to-goal
        # map built from Go-Explore SOLUTION traces (search output, not the ROM);
        # densifies the long-horizon gradient without changing the optimal
        # policy. Gated on reinforce.wavefront_reward.enabled + a dmap path.
        _wave_cfg = dict(_rl_cfg.get("wavefront_reward", {}) or {})
        wave_pot = None
        if _wave_cfg.get("enabled", False) and _wave_cfg.get("dmap"):
            from src.utils.wavefront_reward import WavefrontPotential
            wave_pot = WavefrontPotential.load(
                _wave_cfg["dmap"],
                phi_target=float(_wave_cfg.get("phi_target", 100.0)),
                gamma=float(_wave_cfg.get("gamma", _rl_cfg.get("gamma", 0.99))),
            )
            log.info("[vanilla_ppo] WAVEFRONT reward ON (NON-FARMABLE): %d cells, "
                     "phi_target=%.1f, D_start=%.0f (positive-shifted PBRS, "
                     "search-derived, no game internals)",
                     len(wave_pot.dmap), wave_pot.phi_target, wave_pot.d_start)
        # Invariance-preserving terminal rule (reinforce.wave_terminal_rule
        # "monotone"): shaping pays only on new episode peaks, true deaths
        # charge -peak (Grzes zero-terminal on the peak-augmented potential,
        # so non-clearing episodes telescope to exactly zero net shaping),
        # and the lost-cut becomes a TRUNCATION whose GAE bootstraps V
        # instead of 0 (Pardo). Absent = the legacy -peak-on-any-non-clear
        # behavior, byte-identical.
        from src.training.wave_shaping import (
            monotone_wave_step, resolve_wave_terminal_rule,
            wave_terminal_charge,
        )
        wave_monotone = resolve_wave_terminal_rule(
            _rl_cfg.get("wave_terminal_rule")
        )
        if wave_monotone and wave_pot is None:
            log.warning(
                "[vanilla_ppo] wave_terminal_rule 'monotone' set but the "
                "wavefront reward is OFF — the rule has nothing to act on."
            )
        if wave_monotone and wave_pot is not None:
            log.info(
                "[vanilla_ppo] wave terminal rule: MONOTONE (peak-only "
                "shaping, death charges -peak, lost-cut truncates + "
                "bootstraps V)"
            )
        # KL-anchored warm start + critic warmup
        # (reinforce.kl_anchor_checkpoint + kl_beta_* + actor_freeze_steps).
        # The anchor's actor weights seed the live net only on a FRESH run —
        # a resume keeps its trained weights and just continues the beta
        # schedule (env-steps derive from the absolute iter).
        self._kl_anchor = None
        _kl_ckpt = _rl_cfg.get("kl_anchor_checkpoint")
        if _kl_ckpt:
            if not self._is_tile_mode:
                raise ValueError(
                    "reinforce.kl_anchor_checkpoint requires tile mode "
                    "(the anchor prior is a tile MLP)"
                )
            if self._recurrent:
                raise ValueError(
                    "reinforce.kl_anchor_checkpoint does not support the "
                    "recurrent path (the freeze/penalty hooks live in the "
                    "feedforward update)"
                )
            from src.training.kl_anchor import KLAnchor
            self._kl_anchor = KLAnchor(
                checkpoint_path=str(_kl_ckpt),
                beta_start=float(_rl_cfg.get("kl_beta_start", 0.05)),
                beta_end=float(_rl_cfg.get("kl_beta_end", 0.01)),
                beta_decay_steps=float(_rl_cfg.get("kl_beta_decay_steps", 50e6)),
                actor_freeze_steps=float(_rl_cfg.get("actor_freeze_steps", 5e6)),
                num_actions=self.num_actions,
                feature_dim=self._tile_feature_dim,
                device=self.device,
            )
            if iter_offset == 0:
                _kl_loaded = self._kl_anchor.load_actor_into(net)
                log.info(
                    "[vanilla_ppo] KL ANCHOR: loaded %d actor weight(s) "
                    "from %s (critic stays fresh)", len(_kl_loaded), _kl_ckpt,
                )
            _run_end_steps = (iter_offset + num_iters) * (
                self.num_instances * self.rollout_steps)
            if self._kl_anchor.actor_freeze_steps > _run_end_steps:
                log.warning(
                    "[vanilla_ppo] KL ANCHOR: actor_freeze_steps %.3g "
                    "EXCEEDS this run's final step count %.3g — the actor "
                    "will NEVER unfreeze and only the critic will train. "
                    "This is how two experiments (phase-3 hazard, "
                    "commitment options) silently trained nothing: the "
                    "value was inherited from a campaign base profile "
                    "whose controller overrides it per phase, but a "
                    "standalone train_game run has no controller.",
                    self._kl_anchor.actor_freeze_steps, _run_end_steps)
            log.info(
                "[vanilla_ppo] KL ANCHOR on: beta %.3f -> %.3f over %.0f "
                "steps, actor frozen for %.0f steps",
                self._kl_anchor.beta_start, self._kl_anchor.beta_end,
                self._kl_anchor.beta_decay_steps,
                self._kl_anchor.actor_freeze_steps,
            )
        # Loss-level anchor tether (reinforce.kl_anchor_loss_coef): each PPO
        # minibatch adds coef * mean KL(prior(.|s) || pi_theta(.|s)) on its
        # own states directly to the update loss (ppo_updater), reusing the
        # frozen prior above. 0.0 (the default) = off, byte-identical
        # update; composable with the reward-level beta penalty.
        self._kl_anchor_loss_coef = float(
            _rl_cfg.get("kl_anchor_loss_coef", 0.0) or 0.0
        )
        if self._kl_anchor_loss_coef > 0.0:
            if self._kl_anchor is None:
                log.warning(
                    "[vanilla_ppo] reinforce.kl_anchor_loss_coef %.3f set "
                    "but no kl_anchor_checkpoint — the loss-level tether "
                    "has no prior and stays OFF.",
                    self._kl_anchor_loss_coef,
                )
            else:
                log.info(
                    "[vanilla_ppo] KL ANCHOR loss-level tether on: "
                    "coef %.3f per minibatch", self._kl_anchor_loss_coef,
                )
        # Self-imitation buffer (reinforce.sil): full trajectories of
        # level-clearing episodes feed a BC term in the PPO update. The
        # updater reads self._sil_buffer / self._sil_bc_coef; None = inert.
        _sil_cfg = dict(_rl_cfg.get("sil", {}) or {})
        sil_on = bool(_sil_cfg.get("enabled", False))
        self._sil_buffer = None
        self._sil_bc_coef = 0.0
        if sil_on and not self._is_tile_mode:
            log.warning(
                "[vanilla_ppo] reinforce.sil requires tile mode (pixel "
                "trajectories would be ~GBs at buffer_size clears) — OFF."
            )
            sil_on = False
        if sil_on and self._recurrent:
            log.warning(
                "[vanilla_ppo] reinforce.sil is feedforward-only (the BC "
                "term lives in the K-epoch minibatch loop) — OFF."
            )
            sil_on = False
        if sil_on:
            from src.training.sil import SelfImitationBuffer
            self._sil_buffer = SelfImitationBuffer(
                capacity=int(_sil_cfg.get("buffer_size", 64))
            )
            self._sil_bc_coef = float(_sil_cfg.get("bc_coef", 0.5))
            if float(_rl_cfg.get("sam_rho", 0.0) or 0.0) > 0.0:
                log.warning(
                    "[vanilla_ppo] sil + sam_rho: the BC term is not "
                    "applied on the SHAPO three-pass path."
                )
            log.info(
                "[vanilla_ppo] SELF-IMITATION on: buffer %d clears, "
                "bc_coef %.2f", int(_sil_cfg.get("buffer_size", 64)),
                self._sil_bc_coef,
            )
        frontier_frac = float(_ws_cfg.get("frontier", 0.50))
        base_retention_frac = float(_ws_cfg.get("retention", 0.25))
        retention_frac = base_retention_frac
        retention_bump = float(_forget_cfg.get("retention_bump", 0.40))
        retention_bump_until = 0
        forget_probes = int(_forget_cfg.get("probes", 2))
        forget_bump_iters = int(_forget_cfg.get("bump_iters", 100))
        cold_every = max(1, int(_cold_cfg.get("every", 25)))
        cold_curve_eps = max(1, int(_cold_cfg.get("curve_episodes", 8)))
        cold_winner_name = str(_cold_cfg.get("winner", "best_cold.pt"))
        cold_highwater = 0
        forget_strikes = 0
        best_cold_rate = -1.0
        best_cold_key = (-1.0, -1.0, -1.0)
        best_cold_snapshot: Optional[dict] = None
        last_cold_metrics: dict = {}
        # Alarm/winner state survives resumes via a sidecar. Without it,
        # every resume re-baselined the forgetting high-water to whatever
        # regressed level the run started at (regression stopped counting
        # as regression) and reset the winner baseline (see the winner-key
        # comment at the probe block). Best-effort IO — a missing/corrupt
        # sidecar degrades to fresh state, never a crash.
        _alarm_sidecar = self.checkpoint_dir / "oneshot_alarm.json"
        try:
            if _alarm_sidecar.exists():
                _sc = json.loads(_alarm_sidecar.read_text())
                cold_highwater = int(_sc.get("highwater", 0))
                best_cold_key = tuple(_sc.get("best_cold_key", (-1.0,) * 3))
                best_cold_rate = float(best_cold_key[0])
                log.info(
                    "[vanilla_ppo] alarm sidecar loaded: highwater=%d "
                    "best_cold_key=%s", cold_highwater, best_cold_key,
                )
        except Exception as _e:
            log.warning("[vanilla_ppo] alarm sidecar unreadable (%s) — "
                        "fresh alarm state", _e)
        # Consolidation (gated, reversible) state.
        _cons_ent = dict(_cons_cfg.get("entropy", {}) or {})
        _cons_rnd = dict(_cons_cfg.get("rnd", {}) or {})
        _cons_abort = dict(_cons_cfg.get("abort_if", {}) or {})
        cons_ent_from = float(_cons_ent.get("from", self.entropy_coef))
        cons_ent_to = float(_cons_ent.get("to", self.entropy_coef))
        cons_ent_iters = int(_cons_ent.get("iters", 80))
        cons_rnd_from = float(_cons_rnd.get("from", self.rnd_intrinsic_coef))
        cons_rnd_to = float(_cons_rnd.get("to", 0.0))
        cons_no_gain_iters = int(_cons_abort.get("no_gain_iters", 40))
        cons_fallback = str(_cons_cfg.get("fallback", "cyclic"))
        consolidating = False
        cons_step = 0
        cons_best_rate = -1.0
        cons_no_gain = 0
        cons_aborts = 0
        cyclic_mode = False
        cyclic_phase = 0
        # Go-Explore unstick burst (Lane 5, DEFERRED). A bounded, reversible
        # archive burst armed ONLY when a rung stalls for `stall_patience`
        # iters while the frontier is genuinely being played (§Q3). Off unless
        # the profile sets `go_explore_fallback.enabled`; when off, every `ge_*`
        # branch below is dead, so non-campaign runs are byte-for-byte untouched.
        _geb_cfg = dict(_rl_cfg.get("go_explore_fallback", {}) or {})
        ge_burst_on = bool(ladder_on and _geb_cfg.get("enabled", False))
        ge_stall_patience = max(1, int(_geb_cfg.get("stall_patience", 60)))
        ge_burst_iters = max(1, int(_geb_cfg.get("burst_iters", 30)))
        ge_burst_frac = float(_geb_cfg.get("burst_env_frac", 0.25))
        ge_burst_cap = int(_geb_cfg.get("burst_env_cap", 8))
        ge_archive = None            # a GoExploreArchive, live only mid-burst
        ge_burst_active = False
        ge_burst_remaining = 0
        ge_burst_quota = 0
        ge_iters_since_advance = 0   # stall clock; reset on advance / arm / retract
        ge_bursts_done = 0
        # Human-notify latch (pure addition — no effect on control flow).
        # `ge_iters_since_advance` is bounded by design: the burst arm/
        # retract housekeeping above resets it to 0 every ~(stall_patience
        # + burst_iters) iters as long as the self-correcting Go-Explore
        # fallback keeps engaging, so it only climbs PAST `stall_patience`
        # when that automated recovery itself can't engage (blocked by an
        # active consolidation freeze, or the frontier genuinely isn't
        # being reached at all) — i.e. exactly the case where the built-in
        # fallback has nothing left to try and a human should look. A
        # threshold near `stall_patience` (or the much smaller, purely
        # noise-smoothing `SMB_ADVANCE_WINDOW` rolling-mean window used by
        # the ordinary advance gate a few hundred lines below) would fire
        # on every routine burst cycle instead, so it's sized as a
        # multiple of `stall_patience` — the existing "how long is too
        # long" threshold for this exact counter — rather than of that
        # unrelated, fast-smoothing window. 3x gives the fallback two full
        # arm/retract cycles to unstick things on its own before paging a
        # human on the third.
        ge_stall_notify_multiplier = max(
            1, int(_geb_cfg.get("stall_notify_multiplier", 3))
        )
        ge_stall_notify_threshold = ge_stall_patience * ge_stall_notify_multiplier
        stall_notifier = StallNotifier(
            threshold=ge_stall_notify_threshold,
            title="Training stalled",
            message_fn=lambda n: (
                f"[{self.game_profile.get('name', 'unknown')}] "
                f"{self.checkpoint_dir.name}: frontier has not advanced in "
                f"{n} iters (notify threshold {ge_stall_notify_threshold})."
            ),
        )
        # Restore the rolling advance history + burst bookkeeping from a
        # resumed checkpoint (staged by CheckpointManager.resume). Without
        # it every resume re-earned the advance window from an empty
        # history and zeroed the stall clock. The mid-burst `ge_archive`
        # is intentionally NOT persisted — every consumer already guards
        # on `ge_archive is not None`, so a restored active burst simply
        # ticks down without archive returns and retracts normally.
        _cr = getattr(self, "_pending_curriculum_resume", None)
        if _cr:
            smb_stage_clear_history = [
                int(x) for x in _cr.get("smb_stage_clear_history", [])
            ]
            smb_pastfrac_history = [
                float(x) for x in _cr.get("smb_pastfrac_history", [])
            ]
            ge_burst_active = bool(_cr.get("ge_burst_active", False))
            ge_burst_remaining = int(_cr.get("ge_burst_remaining", 0))
            ge_burst_quota = int(_cr.get("ge_burst_quota", 0))
            ge_iters_since_advance = int(_cr.get("ge_iters_since_advance", 0))
            ge_bursts_done = int(_cr.get("ge_bursts_done", 0))
            # Restore the notify latch too — without it, resuming mid-stall
            # (ge_iters_since_advance already past threshold) would re-fire
            # a notification for the same stall episode that already
            # notified before the restart.
            stall_notifier.notified = bool(_cr.get("ge_stall_notified", False))
            self._pending_curriculum_resume = None
            log.info(
                "[vanilla_ppo] curriculum resume state RESTORED: "
                "pastfrac_window=%d clear_history=%d ge_burst_active=%s "
                "ge_burst_remaining=%d ge_stall_iters=%d ge_bursts_done=%d",
                len(smb_pastfrac_history), len(smb_stage_clear_history),
                ge_burst_active, ge_burst_remaining, ge_iters_since_advance,
                ge_bursts_done,
            )
        elif self._vppo_resumed_from_iter is not None:
            log.info(
                "[vanilla_ppo] curriculum resume state: fresh reset "
                "(checkpoint predates the curriculum_resume blob) — rolling "
                "history + burst state start empty"
            )
        if ladder_on:
            from src.training import oneshot_curriculum as oc
            from src.training.smb_substage_ladder import (
                LADDER_SIZE, OFF_LADDER, build_ladder, region_of,
                smb_sequential_cell,
            )
            ladder = build_ladder(_substage_cfg.get("seed_globs"))
            # Preserve any campaign-captured rung states from a resume: they
            # were written with meta anchor == rung order (see the advance
            # block below), so re-index them into the rung array first.
            _resumed_states = smb_curriculum_states
            _resumed_anchors = smb_curriculum_anchors
            smb_curriculum_states = [None] * LADDER_SIZE
            _resumed_frontier = 0
            for _st, _an in zip(_resumed_states, _resumed_anchors):
                if _st is not None and 0 <= int(_an) < LADDER_SIZE:
                    smb_curriculum_states[int(_an)] = _st
                    _resumed_frontier = max(_resumed_frontier, int(_an))
            # Then bind each rung's disk seed (never clobber a resumed live
            # capture, which is a truer warm-start than the base blob).
            _n_seeded = 0
            for _rung in ladder:
                if smb_curriculum_states[_rung.order] is not None:
                    continue
                if _rung.seed_blob:
                    try:
                        smb_curriculum_states[_rung.order] = (
                            Path(_rung.seed_blob).read_bytes()
                        )
                        _n_seeded += 1
                    except Exception as _e:
                        log.warning(
                            "[vanilla_ppo] ladder seed load failed "
                            "(rung %d, %s): %s", _rung.order, _rung.seed_blob, _e,
                        )
            # anchors[k] == k so the reused gate reads the frontier order
            # directly; `smb_curriculum_stage` is now the frontier rung.
            smb_curriculum_anchors = list(range(LADDER_SIZE))
            smb_curriculum_stage = min(
                max(_resumed_frontier, 0), LADDER_SIZE - 1
            )
            SMB_ADVANCE_PCT = float(_adv_cfg.get("pct", SMB_ADVANCE_PCT))
            SMB_ADVANCE_WINDOW = int(_adv_cfg.get("window", SMB_ADVANCE_WINDOW))
            log.info(
                "[vanilla_ppo] SUB-STAGE LADDER on: %d rungs, %d disk seeds "
                "bound, frontier=%d, advance %.0f%%/%d-iter, warm-start "
                "F/R/S=%.2f/%.2f/%.2f, cold probe every %d iters.",
                LADDER_SIZE, _n_seeded, smb_curriculum_stage,
                SMB_ADVANCE_PCT * 100, SMB_ADVANCE_WINDOW,
                frontier_frac, base_retention_frac,
                float(_ws_cfg.get("spread", 0.25)), cold_every,
            )

        # === LEVEL-SCOPED CONSOLIDATION ("weld one level") setup ===
        # Trains 100% of the pool inside ONE target level's rungs and, every
        # probe, cold-evals the target AND every already-welded PROTECT level
        # from their entry states (eval_game.py --sequential --level-clear:
        # greedy from each level's entry, "cleared the level it started in").
        # A protect regression below its mode-start baseline rolls back to the
        # last accepted snapshot + freezes entropy for a cooldown; a target
        # improvement snapshot-accepts (best_<level>.pt). Terminates when the
        # target cold clear rate holds >= accept_bar for accept_probes probes
        # (writes best_<level>.pt + a DONE marker + exits cleanly, so a driver
        # can chain levels). All gate DECISIONS route through the unit-tested
        # oneshot_curriculum module; this block is thin glue.
        clevel_target = ""
        clevel_target_rungs: list = []
        clevel_target_entry: Optional[str] = None
        clevel_protect_entries: dict = {}
        _clevel_probe = dict(_clevel_cfg.get("probe", {}) or {})
        # THE B6 REPAIR (v4 receipt: checkpoints/mario_1_1_backward_v4/run.log
        # + consolidate_1-1.json). None of the perturbation settings reached
        # v4's gate probe, so `eval_game.run_consumes_randomness` was False and
        # its `probe.episodes: 1` was not a shortcut: any N would have produced
        # N copies of one trajectory. The 1.000 that set the ratchet was that
        # single trajectory, and strict improvement over best=1.0 then held the
        # gate shut for 144 iterations.
        #
        # `resolve_probe_settings` resolves the probe distribution in ONE
        # place, mirroring the PLR cold probe (which has always passed
        # sticky_prob=0.25 / start_jitter=16). Absent keys resolve to the
        # pre-B6 values, so a profile without them emits the identical
        # eval_game command line and the gate is byte-identical to v4.
        from src.training import oneshot_curriculum as _oc_gate
        clevel_probe_cfg = _oc_gate.resolve_probe_settings(_clevel_probe)
        clevel_every = clevel_probe_cfg.every
        clevel_eps = clevel_probe_cfg.episodes
        clevel_max_steps = clevel_probe_cfg.max_steps
        clevel_probe_kwargs = clevel_probe_cfg.probe_kwargs()
        clevel_bar = float(_clevel_cfg.get("accept_bar", 0.75))
        clevel_need = max(1, int(_clevel_cfg.get("accept_probes", 3)))
        clevel_cooldown = max(0, int(_clevel_cfg.get("cooldown", 40)))
        clevel_tol = float(_clevel_cfg.get("tol", 1e-9))
        # Accept arithmetic. Defaults reproduce the pre-B6 point-estimate
        # ratchet exactly; `wilson_lb_gt_best_point` is the break.
        clevel_use_wilson = bool(_clevel_cfg.get("use_wilson_bound", False))
        clevel_wilson_conf = float(_clevel_cfg.get("wilson_confidence", 0.95))
        clevel_accept_rule = str(
            _clevel_cfg.get("accept_rule", _oc_gate.ACCEPT_RULE_POINT)
        )
        clevel_best_decay = float(_clevel_cfg.get("best_decay", 0.0))
        if clevel_accept_rule not in _oc_gate.ACCEPT_RULES:
            raise ValueError(
                f"reinforce.consolidate_level.accept_rule must be one of "
                f"{list(_oc_gate.ACCEPT_RULES)}, got {clevel_accept_rule!r}"
            )
        if not (0.0 <= clevel_best_decay <= 1.0):
            raise ValueError(
                f"reinforce.consolidate_level.best_decay must be in [0, 1] "
                f"(0 = the pre-B6 permanent high-water mark, 1 = replace the "
                f"incumbent with every measurement); got {clevel_best_decay}"
            )
        clevel_probe_n = 0   # episodes the last target probe actually scored
        # Lower bound under `best_tgt_rate`, set from the Wilson LB of the
        # probe that last accepted. None until a bounded accept establishes
        # one, which is why the pre-B6 path never sees it.
        clevel_best_floor: Optional[float] = None
        # The gate's protocol, built ONCE and written verbatim into both the
        # sidecar and the DONE marker, so the run's two receipts cannot
        # disagree about how their numbers were produced. `eval_seed` is the
        # EFFECTIVE seed, never the configured-or-null one: recording null for
        # an unset key would reintroduce, inside the new receipt block, the
        # very defect that writing sticky/jitter/seed into eval.jsonl exists
        # to close: a number whose protocol is missing from its own receipt.
        clevel_protocol = {
            "episodes": clevel_probe_cfg.episodes,
            "sticky_prob": clevel_probe_cfg.sticky_prob,
            "start_jitter": clevel_probe_cfg.start_jitter,
            "eval_seed": clevel_probe_cfg.effective_eval_seed,
            "eval_seed_configured": clevel_probe_cfg.eval_seed,
            "eval_workers": clevel_probe_cfg.eval_workers,
            "eval_rng": clevel_probe_cfg.eval_rng,
            "stochastic": clevel_probe_cfg.stochastic,
            "accept_rule": clevel_accept_rule,
            "use_wilson_bound": clevel_use_wilson,
            "wilson_confidence": clevel_wilson_conf,
            "wilson_confidence_effective": _oc_gate.nearest_tabulated_confidence(
                clevel_wilson_conf
            ),
            "best_decay": clevel_best_decay,
            "accept_bar": clevel_bar,
            "accept_probes": clevel_need,
            "reporting_eval_seeds": list(_oc_gate.REPORTING_EVAL_SEEDS),
        }
        _clevel_sched = dict(_clevel_cfg.get("schedule", {}) or {})
        _clevel_ent = dict(_clevel_sched.get("entropy", {}) or {})
        _clevel_rnd = dict(_clevel_sched.get("rnd", {}) or {})
        clevel_ent_from = float(_clevel_ent.get("from", self.entropy_coef))
        clevel_ent_to = float(_clevel_ent.get("to", self.entropy_coef))
        clevel_ent_iters = int(_clevel_ent.get("iters", 80))
        clevel_rnd_from = float(_clevel_rnd.get("from", self.rnd_intrinsic_coef))
        clevel_rnd_to = float(_clevel_rnd.get("to", 0.0))
        clevel_step = 0
        clevel_cooldown_until = 0
        clevel_sustain = 0
        clevel_done = False
        clevel_baselines: Optional[dict] = None
        clevel_protect_rates: dict = {}
        clevel_tgt_rate = -1.0
        best_tgt_rate = -1.0
        accepted_snapshot: Optional[dict] = None
        clevel_winner_name = "best_level.pt"
        clevel_sidecar = self.checkpoint_dir / "consolidate.json"
        clevel_done_marker = self.checkpoint_dir / "consolidate.DONE"
        if consolidate_on:
            from src.training import oneshot_curriculum as oc
            from src.training.smb_substage_ladder import (
                LADDER_SIZE, build_ladder,
            )
            _seed_globs = (
                _clevel_cfg.get("seed_globs") or _substage_cfg.get("seed_globs")
            )
            # Content-verified seed binding: classify every candidate blob by
            # LOADING it and reading (world, level, area, gx) from RAM, never
            # by filename/meta. Filename schemes have drifted (bw_x* blobs
            # that are really post-win World-2 states; one-shot metas storing
            # the rung order, not the packed anchor) and a mis-bound seed
            # silently poisons a rung's warm-starts: World-2 starts terminate
            # off-target within ~12 steps, and depth captures claiming x=0
            # leave every rung training the level entry.
            ladder = build_ladder(
                _seed_globs, content_probe=self._make_seed_content_probe(),
            )
            # Rung-indexed warm-start seed array (disk seeds only — this mode
            # never captures or advances). Reuses the same smb_curriculum_states
            # slot the auto-reset + warm-start plumbing already reads.
            smb_curriculum_states = [None] * LADDER_SIZE
            for _rung in ladder:
                if _rung.seed_blob:
                    try:
                        smb_curriculum_states[_rung.order] = (
                            Path(_rung.seed_blob).read_bytes()
                        )
                    except Exception as _e:
                        log.warning(
                            "[vanilla_ppo] consolidate seed load failed "
                            "(rung %d, %s): %s", _rung.order, _rung.seed_blob, _e,
                        )
            smb_curriculum_anchors = list(range(LADDER_SIZE))
            smb_curriculum_stage = 0  # unused (no advance); keeps indexing safe

            clevel_target = str(_clevel_cfg.get("target", "")).strip()
            clevel_target_rungs = oc.level_rungs(ladder, clevel_target)

            def _resolve_entry(_level, _explicit):
                """Entry-state PATH for a level: explicit config path wins, else
                the level's entry-rung disk seed, else the profile cold boot
                (correct only for 1-1). Used for both probing and warm-start."""
                if _explicit:
                    _p = Path(str(_explicit))
                    if not _p.is_absolute():
                        _p = Path.cwd() / _p
                    if _p.exists():
                        return str(_p)
                    log.warning(
                        "[vanilla_ppo] consolidate entry state for %s not "
                        "found: %s — falling back.", _level, _p,
                    )
                _rungs = oc.level_rungs(ladder, _level)
                if _rungs and ladder[_rungs[0]].seed_blob:
                    return ladder[_rungs[0]].seed_blob
                return self.start_state_path  # profile start == the 1-1 boot

            clevel_target_entry = _resolve_entry(
                clevel_target, _clevel_cfg.get("target_entry_state")
            )
            for _pe in (_clevel_cfg.get("protect", []) or []):
                if isinstance(_pe, str):
                    _plevel, _ppath = _pe, None
                else:
                    _plevel = str(_pe.get("level", "")).strip()
                    _ppath = _pe.get("entry_state")
                if _plevel:
                    clevel_protect_entries[_plevel] = _resolve_entry(_plevel, _ppath)

            # Train on what you probe: the resolved target entry state
            # replaces the entry rung's disk seed, so the entry-rung envs
            # warm-start from the exact frame the gate cold-probes. A
            # mid-chain handoff differs from the scout's native capture by
            # enemy/timer phase — a gate probing a state the pool never
            # trains from can sit at 0.000 indefinitely.
            if clevel_target_entry and clevel_target_rungs:
                try:
                    smb_curriculum_states[clevel_target_rungs[0]] = Path(
                        clevel_target_entry
                    ).read_bytes()
                    log.info(
                        "[vanilla_ppo] consolidate entry rung %d warm-start "
                        "bound to target entry state %s",
                        clevel_target_rungs[0], clevel_target_entry,
                    )
                except Exception as _e:
                    log.warning(
                        "[vanilla_ppo] consolidate entry-state warm-start "
                        "bind failed (%s): %s — keeping disk seed.",
                        clevel_target_entry, _e,
                    )

            clevel_winner_name = str(
                _clevel_cfg.get("winner", f"best_{clevel_target}.pt")
            )
            clevel_sidecar = self.checkpoint_dir / f"consolidate_{clevel_target}.json"
            clevel_done_marker = (
                self.checkpoint_dir / f"consolidate_{clevel_target}.DONE"
            )

            log.info(
                "[vanilla_ppo] LEVEL CONSOLIDATION on: target=%s rungs=%s "
                "entry=%s protect=%s; probe every %d (%d eps, %d steps), "
                "accept_bar=%.2f x%d probes, cooldown=%d, entropy "
                "%.3f->%.3f/%d iters.",
                clevel_target, clevel_target_rungs, clevel_target_entry,
                list(clevel_protect_entries), clevel_every, clevel_eps,
                clevel_max_steps, clevel_bar, clevel_need, clevel_cooldown,
                clevel_ent_from, clevel_ent_to, clevel_ent_iters,
            )
            # The probe DISTRIBUTION and the accept ARITHMETIC, logged as one
            # receipt line: without it a run's log cannot say whether its gate
            # measured a replay or a distribution (the v4 ambiguity).
            # `eval_seed` is logged as the EFFECTIVE seed (what the subprocess
            # will use) with the configured value beside it, because an unset
            # seed is not "no seed" — eval_game defaults it to 0.
            log.info(
                "[vanilla_ppo] CONSOLIDATE GATE PROTOCOL: episodes=%d "
                "sticky=%.2f jitter=%d eval_seed=%d (configured=%s) "
                "eval_workers=%d eval_rng=%s stochastic=%s | accept_rule=%s "
                "wilson=%s conf=%.2f best_decay=%.2f bar=%.2f x%d",
                clevel_probe_cfg.episodes, clevel_probe_cfg.sticky_prob,
                clevel_probe_cfg.start_jitter,
                clevel_probe_cfg.effective_eval_seed,
                clevel_probe_cfg.eval_seed,
                clevel_probe_cfg.eval_workers, clevel_probe_cfg.eval_rng,
                clevel_probe_cfg.stochastic, clevel_accept_rule,
                clevel_use_wilson, clevel_wilson_conf, clevel_best_decay,
                clevel_bar, clevel_need,
            )
            if clevel_wilson_conf not in _oc_gate.TABULATED_CONFIDENCES:
                log.warning(
                    "[vanilla_ppo] CONSOLIDATE GATE: wilson_confidence=%.4f "
                    "is not tabulated; the bound is taken at the nearest "
                    "available confidence %.2f. Tabulated: %s.",
                    clevel_wilson_conf,
                    _oc_gate.nearest_tabulated_confidence(clevel_wilson_conf),
                    sorted(_oc_gate.TABULATED_CONFIDENCES),
                )
            if not clevel_probe_cfg.stochastic:
                log.warning(
                    "[vanilla_ppo] CONSOLIDATE GATE: probe is DETERMINISTIC "
                    "(sticky=0, jitter=0) — every one of its %d episodes is a "
                    "replay of the same trajectory, so however large the "
                    "episode count, the accept rests on one sample (the v4 "
                    "failure mode). Set reinforce.consolidate_level.probe."
                    "{sticky_prob,start_jitter} to measure a distribution.",
                    clevel_probe_cfg.episodes,
                )
            # SELECTION seed vs REPORTING seed. The gate selects the snapshot;
            # the published deliverable is scored on oneshot_curriculum.
            # REPORTING_EVAL_SEEDS. Sharing a seed between the two lets the
            # gate pick the network that happens to win on the very stream it
            # will later be graded on — the probe-overfitting hole.
            #
            # ENFORCED, not warned. A warning does not stop the probe, and
            # `--strict-config` cannot catch this at all (every key involved is
            # registered and individually valid — the defect is the value, and
            # the commonest way to hit it is to arm sticky + jitter and simply
            # OMIT `eval_seed`, which lands on eval_game's default of 0).
            # `seed_collision` is therefore computed from the EFFECTIVE seed,
            # and a collision stops the run at mode start, before a single
            # iteration of an attended block is spent.
            _oc_gate.require_selection_seed(clevel_probe_cfg)

            # Baseline capture at mode start: probe the pristine champion once
            # for the target AND every protect level from their entry states.
            # These baselines are the hard no-regression floor; the initial
            # accepted snapshot is the champion itself.
            from src.training import cold_probe as _cold_probe

            def _clevel_probe_level(_entry):
                """Baseline probe. `clevel_probe_kwargs` carries the resolved
                honest-probe distribution (episodes / sticky / jitter / seed /
                workers) — the same threading the PLR cold probe has always
                done. Returns (rate, furthest, n_episodes_scored)."""
                if _entry is None:
                    return None, None, 0
                _r = _cold_probe.probe(
                    net, self.game_profile,
                    device=self.device, sequential=True, level_clear=True,
                    start_state=_entry,
                    rom_path=self.rom_path,
                    game=str(self.game_profile.get("name", "mario")),
                    **clevel_probe_kwargs,
                )
                _rate = _r.get("cold_seq_clear_rate")
                return (
                    float(_rate) if _rate is not None else None,
                    _r.get("cold_furthest_seq"),
                    int(_r.get("cold_n_episodes") or 0),
                )
            clevel_baselines = {}
            for _plevel, _pentry in clevel_protect_entries.items():
                _b, _, _ = _clevel_probe_level(_pentry)
                if _b is not None:
                    clevel_baselines[_plevel] = _b
            _tb, _, clevel_probe_n = _clevel_probe_level(clevel_target_entry)
            best_tgt_rate = _tb if _tb is not None else -1.0
            accepted_snapshot = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }
            # Persist the accepted champion immediately so best_<level>.pt
            # exists even if the run dies before the first accept.
            try:
                torch.save(
                    {"net_state_dict": accepted_snapshot, "iter": iter_offset,
                     "metric_name": f"consolidate_{clevel_target}_clear_rate",
                     "metric_value": max(best_tgt_rate, 0.0)},
                    str(self.checkpoint_dir / clevel_winner_name),
                )
            except Exception as _e:
                log.warning("[vanilla_ppo] best_%s init save failed: %s",
                            clevel_target, _e)
            log.info(
                "[vanilla_ppo] CONSOLIDATE baselines: target %s=%.3f "
                "(n=%d, wilson_lb=%.3f), protect=%s",
                clevel_target, best_tgt_rate, clevel_probe_n,
                _oc_gate.wilson_lower_bound(
                    max(best_tgt_rate, 0.0) * clevel_probe_n, clevel_probe_n,
                    confidence=clevel_wilson_conf,
                ),
                clevel_baselines,
            )

        if self._is_tile_mode:
            base_dim = self._tile_extractor.feature_dim
            tile_stackers = [
                TileFeatureStacker(
                    stack_size=self._tile_frame_stack, feature_dim=base_dim,
                )
                for _ in range(num_envs)
            ]
            stackers: list = []
        else:
            obs_dtype = self._obs_buffer_dtype()
            stackers = [FrameStacker(dtype=obs_dtype) for _ in range(num_envs)]
            tile_stackers = []

        obs_shape = self._step_obs_shape()
        obs_dtype = self._obs_buffer_dtype()

        # Per-iter rollout buffers — pre-allocated, reused across iters.
        # Shape: (rollout_steps, num_envs, *obs_shape) for observations.
        obs_buf = np.zeros((rollout_steps, num_envs) + obs_shape, dtype=obs_dtype)
        action_buf = np.zeros((rollout_steps, num_envs), dtype=np.int32)
        reward_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        # Count-bonus stream kept separate from extrinsic rewards so it
        # folds once per iter, done-masked, alongside RND's intrinsic.
        bonus_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        log_prob_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        # Truncation mask (wave_terminal_rule "monotone" only): marks dones
        # that are CUTS (the lost-cut), not real terminals, so GAE
        # bootstraps V(s) there instead of 0. Stays all-False — and is
        # passed to the updater as None — under the legacy rule.
        trunc_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        # Per-step validity mask: True only where an env actually executed
        # that step (including its death step). Post-done frozen-padding
        # slots stay False and are excluded from advantage normalization
        # and the PPO update — otherwise, for non-curriculum games where
        # envs freeze after the first death, a large fraction of the batch
        # is spurious (advantage = -V(stale), value_target = 0) padding.
        valid_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        step_actions = np.zeros(self.pool.num_workers, dtype=np.uint8)

        # Per-env episode tracking (for metrics: mean episode reward / length).
        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_lengths = np.zeros(num_envs, dtype=np.int32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        # Per-iter "still active" mask: True until that env hits done.
        # Once an env is done for this iter we (a) stop processing
        # reward_fn results from it (the rust pool keeps stepping the
        # worker until reset_all but its post-death frames are garbage
        # for our policy gradient), (b) tell the pool to short-circuit
        # the worker via set_worker_done so it stops eating CPU, and
        # (c) zero-pad the rollout buffer so GAE sees done=True for
        # all subsequent steps. Resolves the "inline reward_fn.reset()
        # causes re-firing done on every step" bug — that path inflated
        # completed_eps to 1000+ per iter and mean_len to 3-5.
        active_in_iter = np.ones(num_envs, dtype=bool)
        # Stage-0 inline restart: deferred stacker re-seed flags (the
        # restored state produces RAM only on the NEXT step_all).
        _stage0_reseed = np.zeros(num_envs, dtype=bool)
        _start_bytes = None
        if self.start_state_path:
            try:
                _start_bytes = Path(self.start_state_path).read_bytes()
            except Exception:
                _start_bytes = None
        # Count per-env flag-touches across the iter. Each time the
        # reward fn's `completion` breakdown key grows, an SMB level
        # was cleared by that env. Tracks last-seen value per env so
        # we can diff per-step. Quantitative confirmation of clears
        # alongside the visual GUI signal.
        prev_completion_total = np.zeros(num_envs, dtype=np.float32)
        # Per-env "already recorded this episode's outcome to PLR" flag. A
        # level clear is recorded True the instant the completion-breakdown
        # diff fires (below) — NOT at the `done` boundary, because a clear
        # often does NOT fire done (the env rolls past the flag into the next
        # level and only `done`s on a later death), so done-time recording
        # systematically drops clears. A death records False at done IFF the
        # episode wasn't already recorded as a clear. Dedupe via this flag so
        # each episode contributes exactly one PLR sample.
        episode_recorded = np.zeros(num_envs, dtype=bool)
        # Wavefront PBRS: previous-step potential per env (None = episode start,
        # no shaping across the boundary).
        wave_prev_phi: list = [None] * num_envs
        # Per-episode peak Phi + consecutive off-envelope steps. Non-clear
        # terminals charge -PEAK (not -Phi(final)): charging the final state
        # let the policy bank progress then retreat to a Phi=0 region (the 1-2
        # warp room, off every solution) and time out for FREE — the shaping's
        # last sanctuary. Peak-charging makes every non-completing episode net
        # <= 0 shaping regardless of HOW it ends. The lost-cut terminates
        # episodes that linger off the solution envelope (Phi~0 for LOST_K
        # consecutive steps) — reclaims the ~70% of rollout steps burned
        # wandering the warp room. Search-derived (envelope = solution
        # trajectories), no game internals.
        wave_peak_phi = np.zeros(num_envs, dtype=np.float32)
        wave_lost_count = np.zeros(num_envs, dtype=np.int32)
        WAVE_LOST_K = 150  # ~10 s game time; area-load blips are ~6 steps
        # Self-imitation episode accumulators: per-env (obs, action) lists
        # for the CURRENT episode, appended on every executed step. Flushed
        # into the buffer at the completion-diff clear; dropped on any
        # non-clearing end (death / iter boundary).
        sil_ep_obs: list = [[] for _ in range(num_envs)] if sil_on else []
        sil_ep_act: list = [[] for _ in range(num_envs)] if sil_on else []

        def _sil_flush_clear(i: int) -> None:
            """Store the accumulated episode — it just cleared the level."""
            if not sil_on or not sil_ep_act[i]:
                return
            self._sil_buffer.add(
                np.stack(sil_ep_obs[i]),
                np.asarray(sil_ep_act[i], dtype=np.int64),
            )
            sil_ep_obs[i].clear()
            sil_ep_act[i].clear()

        def _sil_drop_episode(i: int) -> None:
            """Discard a non-clearing episode's accumulator."""
            if sil_on:
                sil_ep_obs[i].clear()
                sil_ep_act[i].clear()

        n_clears_this_iter = 0
        # Per-env max world+level reached this iter, packed as
        # world*16+level so a single uint8 max comparator works
        # ordinally (W=0 L=0 → 0; W=0 L=1 → 1; W=1 L=0 → 16; ...).
        # SMB world byte is at $075F (0=World 1), level at $0760
        # (0=Level 1, 1=Level 2, ...). Initialized to zero (= W1-L1)
        # which is the reset spawn for every env.
        max_world_level_packed = np.zeros(num_envs, dtype=np.uint8)
        end_world_level_packed = np.zeros(num_envs, dtype=np.uint8)
        # Per-env furthest forward x reached this iter (page<<8 | low).
        # Independent of the reward, so it measures real progress within
        # a level/area even when the packed world-level byte doesn't move
        # — e.g. distinguishing "stalled at x~980 in the 1-2 underground"
        # from "pushed past it" without the dense-checkpoint reward (which
        # would otherwise inflate the return and confound the comparison).
        max_x_reached = np.zeros(num_envs, dtype=np.int32)
        # Per-env furthest sub-stage rung reached this iter (ladder mode).
        # Init OFF_LADDER (-1) so an env that only ever warps stays strictly
        # below frontier 0 and can never satisfy the advance gate.
        max_region_reached = np.full(num_envs, -1, dtype=np.int32)

        # ============== GO-EXPLORE ARCHIVE (opt-in, generic) ==============
        # First-return-then-explore (Ecoffet et al. 2021): a game-agnostic
        # alternative to the SMB-hardcoded save-state curriculum. Built ONLY
        # when the curriculum is OFF — the two are mutually exclusive
        # exploration mechanisms that share the same warm-start/auto-reset
        # plumbing, which go_explore reuses when the curriculum is absent. All
        # hooks below are None-guarded, so with go_explore.enabled=false
        # (default) this is entirely inert.
        _ge_cfg = dict(
            (self.game_profile.get("reinforce", {}) or {}).get("go_explore", {})
            or {}
        )
        go_explore_on = (
            bool(_ge_cfg.get("enabled", False)) and not self._smb_curriculum_active
        )
        go_explore_archive = None
        go_explore_save_every = max(1, int(_ge_cfg.get("save_every", 10)))
        # Per-env frontier-cell blob each env was returned to this iter.
        env_return_state: list = [None] * num_envs

        def _ge_score(_i: int) -> float:  # replaced below when enabled
            return 0.0

        # Probability that an INLINE (mid-rollout death) restart draws its
        # start from the archive instead of the profile start state. 0.0
        # (default) keeps every inline restart at the configured start —
        # archive returns then happen only at iter boundaries.
        _ge_inline_p = float(_ge_cfg.get("inline_return_prob", 0.0))
        if go_explore_on:
            # Archive build/load + score-kind selection live in
            # ExplorationController (Task 3); it sets `self._go_explore` and
            # returns `_use_max_x` so the score closure (which captures the
            # rollout trackers) stays here in the conductor.
            go_explore_archive, _use_max_x = _exploration.build_go_explore(
                _ge_cfg, fresh_start=fresh_start
            )

            def _ge_score(_i: int) -> float:  # noqa: F811
                return (
                    float(max_x_reached[_i]) if _use_max_x
                    else float(ep_returns[_i])
                )

        # ========= BACKWARD START-STATE CURRICULUM (opt-in) =========
        # Salimans & Chen (arXiv:1812.03381): a solved tape supplies START
        # STATES, never labels. Episodes restart from a trailing window
        # ending at a cursor `tau` that walks BACKWARD along the tape as
        # the policy earns each rung; the reward is the only learning
        # signal at every rung. That is what separates it from behavior
        # cloning, which this stack already eliminated on these very tapes
        # (Dossier v3, 2026-07-23: clone accuracy 1.0 -> 0.00 honest
        # success). The true entrance always rides in the per-episode draw
        # with length-balanced mass, so (a) the honest start rate is
        # measurable throughout the run and (b) the terminal curriculum
        # (tau 0) IS the honest start distribution — no hand-off step.
        #
        # Inert unless `reinforce.backward_curriculum.states_dir` is set;
        # mutually exclusive with CGSA and the Go-Explore inline return,
        # which own the same restart site.
        _bwd_cfg = dict(_rl_cfg.get("backward_curriculum", {}) or {})
        bwd_on = bool(
            _bwd_cfg.get("enabled", True)
            and _bwd_cfg.get("states_dir")
            and self._is_tile_mode
            and _start_bytes is not None
        )
        bwd_sched = None
        bwd_entries: list = []
        bwd_blobs: list = []
        bwd_tape: list = []           # entry indices; the start_window seq
        bwd_window_frames = int(_bwd_cfg.get("window_frames", 160))
        bwd_frames_per_entry = 4
        bwd_entrance_w = float(_bwd_cfg.get("entrance_weight", 1.0))
        # Episodes still running when the iter boundary calls reset_all —
        # or cut off by the per-rung budget below — are TRUNCATED, and a
        # truncated attempt is censored data: the policy neither cleared
        # nor died. Dropping it (default) keeps the rung's rate an honest
        # "of the attempts that resolved, how many cleared", but censors
        # long episodes — and an episode from a deep rung cannot resolve
        # at all when the remaining tape is longer than `rollout_steps`
        # (see the reachability note logged below).
        #
        # Counting truncations as failures is the Salimans & Chen timeout
        # convention, and B5 run 2 is the receipt for why it matters: the
        # cursor sat 156 iters at rung 893 with a trailing window of 0/0
        # while 15,914 attempts were truncated and dropped, so the advance
        # gate's attempt floor could never be met. `count_truncations` is
        # the registered name; `truncation_is_failure` is the original one
        # and stays a live alias so B4's profiles keep their meaning.
        bwd_trunc_fail = bool(_bwd_cfg.get(
            "count_truncations",
            _bwd_cfg.get("truncation_is_failure", False),
        ))
        bwd_trunc_dropped = 0
        bwd_trunc_scored = 0
        # Dev/guard knob: freeze every restart at the profile start state.
        # With sticky off this reproduces the pre-flag run bit-for-bit (no
        # extra RNG is drawn), which is how the zero-diff gate is checked.
        bwd_pin = bool(_bwd_cfg.get("pin_entrance", False))
        # OPT-IN trailing-entropy guard (see BackwardEntropyGuard for the
        # B4 v1 receipt it exists for). Built only when the curriculum is
        # actually armed; an absent block yields None and the entropy_coef
        # path below is byte-identical to a run without the feature. A
        # block that is present but malformed raises HERE, seconds into
        # the run, rather than after hours on a dead knob.
        bwd_guard = (
            BackwardEntropyGuard.from_config(_bwd_cfg.get("entropy_guard"))
            if bwd_on else None
        )
        # OPT-IN per-rung episode budget (see backward_curriculum.RungBudget
        # for the same run-2 receipt). The object needs the tape length and
        # the rollout cap, so it is BUILT below once the index has loaded;
        # the block's SHAPE is validated here, seconds into the run, so a
        # malformed knob raises instead of silently leaving every episode
        # uncapped. None => no episode is ever cut short (the historical
        # path, and the one every pre-08-09 lineage was measured on).
        bwd_budget = None
        if bwd_on:
            from src.training.backward_curriculum import RungBudget
            RungBudget.parse_config(_bwd_cfg.get("rung_step_budget"))
        # The exact coefficient the guard last wrote, and the unboosted
        # value it was computed from. Both None while disarmed. Storing
        # the written value (not just the multiplier) lets the strip
        # below confirm nobody else moved the coefficient in between — a
        # consolidation schedule ASSIGNS entropy_coef outright — instead
        # of blindly dividing a boost back out of an unrelated number.
        _bwd_guard_base: Optional[float] = None
        _bwd_guard_applied: Optional[float] = None
        # Per-env provenance of the LIVE episode: the cursor it started
        # under, whether it was drawn from the window (1), the entrance
        # (0) or not by this curriculum at all (-1), and whether it has
        # cleared. Attempts the draw did not create are never scored
        # against a rung — an iter-boundary warm start is a different
        # distribution and would drag the rung's rate down.
        _bwd_env_tau = np.zeros(num_envs, dtype=np.int64)
        _bwd_env_src = np.full(num_envs, -1, dtype=np.int8)
        _bwd_env_clear = np.zeros(num_envs, dtype=bool)
        # Per-env episode step cap (0 = uncapped) and "this episode ended
        # because the cap fired", which is what separates a truncation
        # from a death at the scoring site.
        _bwd_env_budget = np.zeros(num_envs, dtype=np.int64)
        _bwd_env_trunc = np.zeros(num_envs, dtype=bool)
        if bwd_on:
            try:
                from src.training import backward_curriculum as bwd
                from src.training.go_explore import start_window as _bwd_window
                bwd_entries, _bwd_meta = bwd.load_index(_bwd_cfg["states_dir"])
                bwd_blobs = bwd.load_blobs(_bwd_cfg["states_dir"], bwd_entries)
                bwd_tape = list(range(len(bwd_entries)))
                bwd_frames_per_entry = max(
                    1, int(_bwd_meta.get("every_frames", 4))
                )
                _bwd_stride = max(1, int(_bwd_meta.get("stride_steps", 1)))
                bwd_sched = bwd.TauScheduler(
                    len(bwd_entries),
                    tau_init=int(_bwd_cfg.get("tau_init", -1)),
                    advance_entries=max(1, int(_bwd_cfg.get(
                        "advance_actions", bwd.DEFAULT_ADVANCE_ACTIONS
                    )) // _bwd_stride),
                    advance_threshold=float(_bwd_cfg.get(
                        "advance_threshold", bwd.DEFAULT_ADVANCE_THRESHOLD
                    )),
                    min_attempts=int(_bwd_cfg.get(
                        "min_attempts", bwd.DEFAULT_MIN_ATTEMPTS
                    )),
                    count_truncations=bwd_trunc_fail,
                )
                # The global cap is the rollout length: the iter boundary
                # truncates every live episode there regardless, so a
                # budget above it would never fire.
                bwd_budget = bwd.RungBudget.from_config(
                    _bwd_cfg.get("rung_step_budget"),
                    n_entries=len(bwd_entries),
                    max_steps=rollout_steps,
                )
                log.info(
                    "[backward] ENABLED: %d states (%d MB) from %s | tau0=%d "
                    "window=%d frames (%d entries) advance=+%d entries at "
                    ">=%.2f over %d attempts | entrance_weight=%.2f%s",
                    len(bwd_entries),
                    sum(len(b) for b in bwd_blobs) >> 20,
                    _bwd_cfg["states_dir"], bwd_sched.tau, bwd_window_frames,
                    bwd_window_frames // bwd_frames_per_entry,
                    bwd_sched.advance_entries, bwd_sched.advance_threshold,
                    bwd_sched.min_attempts, bwd_entrance_w,
                    "  [PINNED AT ENTRANCE]" if bwd_pin else "",
                )
                # A resumed run continues its ladder from where it stopped
                # instead of re-walking rungs it already earned. Restored
                # AFTER the line above (so `tau0=` keeps meaning "what the
                # config asked for") but BEFORE the per-env cursor is
                # seeded, or every env would carry the config's tau while
                # the scheduler sat elsewhere and dropped their attempts
                # as stale.
                self._apply_pending_backward_state(bwd_sched)
                _bwd_env_tau[:] = bwd_sched.tau
                if bwd_budget is not None:
                    log.info(
                        "[backward] BUDGET: an episode restarted at rung r "
                        "gets min(%d, %d + %.2f * (%d - r)) steps — %d at "
                        "the current rung %d, %d at the ladder top, %d at "
                        "the entrance. Over budget = truncated, %s.",
                        bwd_budget.max_steps, bwd_budget.base,
                        bwd_budget.per_entry, bwd_budget.n_entries,
                        bwd_budget.steps_for(bwd_sched.tau), bwd_sched.tau,
                        bwd_budget.steps_for(bwd_budget.n_entries - 1),
                        bwd_budget.steps_for(0),
                        "scored as a failure" if bwd_trunc_fail else "dropped",
                    )
                if bwd_guard is not None:
                    log.info(
                        "[backward-guard] configured: arm when the %d-iter "
                        "trailing policy entropy falls below %.3f (needs "
                        "%d samples), entropy_coef x%.2f while armed, "
                        "disarm above %.3f.",
                        bwd_guard.trailing, bwd_guard.floor,
                        bwd_guard.min_samples, bwd_guard.boost,
                        bwd_guard.recover_floor,
                    )
                # Reachability: an attempt only resolves if the tape's
                # remaining tail fits inside one rollout, because the iter
                # boundary truncates everything still running. Name the
                # deepest rung this rollout length can actually score, so
                # a cursor that stalls there is diagnosed in one line
                # instead of one overnight run.
                _bwd_reach = len(bwd_entries) - (rollout_steps // _bwd_stride)
                if _bwd_reach > 0:
                    log.warning(
                        "[backward] REACHABILITY: rollout_steps=%d covers "
                        "the last %d of %d rungs. Below tau=%d an episode "
                        "cannot finish the tape inside one rollout and is "
                        "truncated (%s). Raise rollout_steps to walk "
                        "further back.",
                        rollout_steps, len(bwd_entries) - _bwd_reach,
                        len(bwd_entries), _bwd_reach,
                        "scored as failure" if bwd_trunc_fail else "dropped",
                    )
                # Every rung restart is an episode boundary, and this
                # curriculum makes hundreds of them per rollout. With
                # sticky on and the boundary guard off, the roll on the
                # first step of a fresh attempt replays the action the
                # previous life died holding — contaminating the very
                # success rate the advance gate consumes, at its most
                # decisive step. Not fatal (a p=0 lineage is fine), so
                # this is loud rather than hard.
                if (
                    self.sticky_action_prob > 0.0
                    and not self.sticky_episode_boundary_reset
                ):
                    log.warning(
                        "[backward] CONTAMINATED: sticky_action_prob=%.2f "
                        "with sticky_episode_boundary_reset=false. Every "
                        "rung restart can replay the dead life's held "
                        "action on the fresh attempt's first step, and "
                        "that lands in the rate the advance gate reads. "
                        "Set reinforce.sticky_episode_boundary_reset: "
                        "true.", self.sticky_action_prob,
                    )
            except Exception as e:
                log.warning(
                    "[backward] disabled — could not load %s: %s",
                    _bwd_cfg.get("states_dir"), e,
                )
                bwd_on = False
                bwd_guard = None
                bwd_budget = None
        elif _bwd_cfg.get("states_dir"):
            log.warning(
                "[backward] configured but INERT: needs a tile-mode encoder "
                "(is=%s) and a start_state_path (have=%s).",
                self._is_tile_mode, _start_bytes is not None,
            )

        def _bwd_score_truncation(idx: int, src: int) -> None:
            """Book one TRUNCATED attempt from env `idx` (source `src`).

            Both truncation sites — the per-rung budget mid-rollout and
            the iter boundary — land here, so the two can never drift
            apart on what a timeout means. The scheduler owns the
            scored-or-dropped rule (`count_truncations`); these counters
            are what the per-iter line reports as `truncated D (S
            scored)`, i.e. how much of this rung's evidence is being
            censored versus counted.
            """
            nonlocal bwd_trunc_scored, bwd_trunc_dropped
            if (bwd_sched.record_truncation(tau=int(_bwd_env_tau[idx]))
                    if src == 1 else
                    bwd_sched.record_entrance_truncation()):
                bwd_trunc_scored += 1
            else:
                bwd_trunc_dropped += 1

        # Initial reset.
        init_results = self.pool.reset_all()
        self._emit_frame_sink(init_results)
        if self._is_tile_mode:
            stacked_obs: list[np.ndarray] = [
                tile_stackers[i].reset(self._tile_extractor.extract(r.ram_snapshot))
                for i, r in enumerate(init_results)
            ]
        else:
            stacked_obs = [
                stackers[i].reset(r.frame, getattr(r, "preprocessed", None))
                for i, r in enumerate(init_results)
            ]

        # `env_stage` / `stage_seed_results` (initialized above) are
        # populated at each iter-boundary warm-start. The initial reset
        # above is a cold boot (all envs at stage 0), so they stay at
        # their defaults (stage 0, no seed) for iteration 0.

        # Narrator: only active when a GUI passed a caption queue. In
        # headless training it stays None and the per-step observe below
        # is skipped entirely — zero overhead on the winning path.
        narrator_on = self._narrator is not None

        # Sticky-action training (Machado et al. 2018 protocol, ported
        # from the GA path's _apply_sampled_actions): with probability
        # sticky_action_prob the env executes the PREVIOUS step's action
        # instead of the fresh sample, and the recorded log-prob is the
        # EXECUTED action's under the current policy, keeping PPO's
        # importance ratio consistent with what actually ran. Training
        # with the same stochasticity the honest eval applies is the
        # published cure for deterministic-replay collapse (Go-Explore,
        # Nature 2021, robustification phase).
        _sticky_p = float(self.sticky_action_prob)
        _prev_exec_action = np.zeros(num_envs, dtype=np.int64)
        # Episode-boundary guard for the carried action (opt-in). Every
        # mid-rollout restart site calls _sticky_restart(i); the roll on
        # the following step skips those envs, matching eval's per-episode
        # `step > 0` gate. Disabled it is a no-op on both paths.
        _sticky_boundary = StickyBoundary(
            num_envs, self.sticky_episode_boundary_reset
        )
        if _sticky_p > 0.0:
            log.info(
                "[vanilla_ppo] STICKY TRAINING on: p=%.2f (executed-action "
                "log-probs; eval protocol parity) boundary_reset=%s",
                _sticky_p, _sticky_boundary.enabled,
            )

        def _sticky_restart(i: int) -> None:
            """Mark env `i` as having just begun a fresh episode."""
            _sticky_boundary.mark_restart(i, _prev_exec_action)

        # ===== PR-MDP: Probabilistic Action Robust MDP =====
        # (Tessler, Efroni & Mannor, ICML 2019; deep-research candidate 1,
        # pre-registration 2026-07-27.) Two-player zero-sum: a co-trained
        # adversary's action executes with prob alpha instead of the
        # protagonist's — the sticky protocol with the noise made
        # worst-case. During training this REPLACES random sticky (set
        # sticky_action_prob 0 in prmdp configs); the honest eval protocol
        # (cold greedy sticky-0.25 jitter-16) is untouched. The
        # protagonist trains on executed actions with clamped log-probs
        # (the house sticky pattern); the adversary trains by PPO on the
        # SAME rollout with negated rewards, restricted to the steps its
        # action executed. Update alternation 10:2 = reinforce_steps
        # protagonist epochs vs adversary_epochs.
        _prmdp_cfg = dict(_rl_cfg.get("prmdp", {}) or {})
        prmdp_on = bool(_prmdp_cfg.get("enabled", False))
        _prmdp_alpha = float(_prmdp_cfg.get("alpha", 0.25))
        _prmdp_adv_epochs = int(_prmdp_cfg.get("adversary_epochs", 2))
        _prmdp_adv_clip = float(_prmdp_cfg.get("adversary_clip", 0.1))
        _prmdp_adv_ent = float(_prmdp_cfg.get("adversary_entropy_coef", 0.1))
        # v7 report (2026-07-30): the co-trained adversary collapsed to a
        # uniform noise generator (entropy pinned at ln|A| for the whole
        # 130M-step run — GDA limit cycle from the 10:2 epoch imbalance +
        # 0.1 entropy coef). adversary_mode "uniform" keeps the exact
        # alpha-override noise profile the protagonist adapted to, with
        # no adversary net, no adversary updates — the recovered compute
        # pays for SHAPO below.
        _prmdp_adv_mode = str(_prmdp_cfg.get("adversary_mode", "net"))
        _adv_is_net = prmdp_on and _prmdp_adv_mode == "net"
        # SHAPO / SAM-PPO (v7 report): pessimistic policy gradient
        # evaluated at θ+ε over the ACTOR set (trunk + actor head; the
        # critic head and RND train standard at θ — three-pass variant,
        # documented deviation for the shared-trunk architecture).
        # 0.0 disables — byte-identical update path.
        _sam_rho = float(_rl_cfg.get("sam_rho", 0.0) or 0.0)
        if _sam_rho > 0.0 and self._recurrent:
            log.warning("[shapo] sam_rho ignored on the recurrent path")
            _sam_rho = 0.0
        _adv_net = None
        _adv_opt = None
        if _adv_is_net:
            # RAW build: the experiment-arm wrappers in _make_network
            # (hazard veto, commitment head) are protagonist arms — a
            # hazard-masked ADVERSARY would be vetoed from exactly the
            # death-causing actions it exists to force.
            _adv_net = self._make_network_raw()
            _adv_net.to(self.device)
            _adv_opt = torch.optim.Adam(
                _adv_net.parameters(),
                lr=float(_prmdp_cfg.get("adversary_lr", 2.5e-4)),
            )
            _pending_adv = getattr(self, "_pending_prmdp_adv", None)
            if _pending_adv is not None:
                _adv_sd, _adv_opt_sd = _pending_adv
                _adv_net.load_state_dict(_adv_sd, strict=False)
                if _adv_opt_sd is not None:
                    try:
                        _adv_opt.load_state_dict(_adv_opt_sd)
                    except Exception as exc:
                        log.warning(
                            "[prmdp] adversary optimizer state load failed "
                            "(continuing with fresh Adam state): %s", exc,
                        )
                self._pending_prmdp_adv = None
                log.info(
                    "[prmdp] RESUMED adversary net + optimizer from "
                    "checkpoint — arms race continues, not reset"
                )
            log.info(
                "[prmdp] ON: alpha=%.2f adversary=%s params, %d epochs, "
                "clip=%.2f ent=%.2f (sticky_p=%.2f should be 0)",
                _prmdp_alpha,
                sum(p.numel() for p in _adv_net.parameters()),
                _prmdp_adv_epochs, _prmdp_adv_clip, _prmdp_adv_ent,
                _sticky_p,
            )
        adv_action_buf = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        adv_logp_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        adv_value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        exec_mask_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)

        # ===== Kernel-matched binary adversary (reinforce.adversary) =====
        # The PR-MDP machinery refit to the honest-eval noise kernel: a
        # 2-action head {pass, repeat-previous-EXECUTED}, deciding every
        # step, paid -protagonist_reward - budget_penalty*I(repeat), and
        # trained epochs-matched to the protagonist (10:10 — the v7 10:2
        # imbalance drove the GDA limit cycle) at entropy_coef 0.01
        # against the ln 2 binary ceiling. Absent block = legacy PR-MDP
        # behavior untouched (both enabled = config error, fail loud).
        _kadv_cfg = dict(_rl_cfg.get("adversary", {}) or {})
        _kadv = None
        _kadv_mode = str(_kadv_cfg.get("mode", "") or "")
        if _kadv_mode and _kadv_mode != "kernel_sticky":
            raise ValueError(
                f"reinforce.adversary.mode {_kadv_mode!r} is not "
                f"'kernel_sticky' (the only implemented mode)"
            )
        if _kadv_mode == "kernel_sticky":
            if prmdp_on:
                raise ValueError(
                    "reinforce.adversary (kernel_sticky) and reinforce."
                    "prmdp.enabled are mutually exclusive — both would "
                    "override the same executed-action channel"
                )
            if self._recurrent:
                raise ValueError(
                    "reinforce.adversary (kernel_sticky) does not support "
                    "the recurrent protagonist path"
                )
            from src.training.kernel_adversary import KernelStickyAdversary
            # RAW build (same rule as the PR-MDP adversary above): the
            # hazard veto's 6-action risky mask would masked_fill this
            # 2-action head — a shape crash at the first decide().
            _kadv_net = self._make_network_raw(num_actions=2)
            _kadv_net.to(self.device)
            _kadv = KernelStickyAdversary(
                net=_kadv_net,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                budget_penalty=float(_kadv_cfg.get("budget_penalty", 0.1)),
                entropy_coef=float(_kadv_cfg.get("entropy_coef", 0.01)),
                epochs=int(_kadv_cfg.get("epochs", 10)),
                clip=float(_kadv_cfg.get("clip", 0.1)),
                lr=float(_kadv_cfg.get("lr", 2.5e-4)),
                gamma=self.reinforce_gamma,
                gae_lambda=self.gae_lambda,
                value_coef=self.value_coef,
                value_loss_kind=self.value_loss_kind,
                grad_clip=self.reinforce_grad_clip,
                device=self.device,
                preprocess_f16=self.preprocess_f16,
                is_tile_mode=self._is_tile_mode,
            )
            log.info(
                "[kernel_adv] ON: budget=%.3f ent=%.3f epochs=%d clip=%.2f "
                "(%d params; protagonist epochs=%d — matched)",
                _kadv.budget_penalty, _kadv.entropy_coef, _kadv.epochs,
                _kadv.clip, sum(p.numel() for p in _kadv_net.parameters()),
                self.reinforce_steps,
            )

        # ===== CGSA-PPO: Cell-Granular Stochasticity Annealing =====
        # (research recipe 2026-07-23, "SMB 1-2 RL Consulting"). Each archived
        # Go-Explore cell carries its OWN sticky probability p_sticky(c),
        # initialized 0.0 and annealed toward the eval target as the policy
        # masters the segment under the current noise:
        #   every cg_window attempts from c: succ_rate >= 0.75 -> p += 0.05;
        #   succ_rate < 0.30 and p > 0 -> p -= 0.05.  (success = advancing
        #   >= one cell zone from the restart gx.)
        # Restart cells are drawn from the archive with priority
        #   W(c) = (1-p_hat_c)^2 + 0.1 on the un-welded frontier, 0.01
        # maintenance on welded cells (anti-forgetting).  A cell at the
        # target noise is WELDED only by passing Wald's SPRT
        # (H0: p<=0.15 vs H1: p>=0.60, alpha=.01, beta=.05) — the sound
        # replacement for the small-sample "gate-mirage" acceptance that
        # inflated every prior weld claim.  Training-time noise curricula
        # do not alter the eval protocol (cold sticky-0.25 from the
        # entrance, unchanged).
        _cgsa_cfg = dict(_rl_cfg.get("cgsa", {}) or {})
        cgsa_on = bool(_cgsa_cfg.get("enabled", False)) and go_explore_on
        cg_target = float(_cgsa_cfg.get("target_sticky", _sticky_p or 0.25))
        cg_step = float(_cgsa_cfg.get("anneal_step", 0.05))
        cg_window = int(_cgsa_cfg.get("attempts_per_update", 50))
        cg_zone_px = int(_cgsa_cfg.get("zone_px", 32))
        cg_alpha = float(_cgsa_cfg.get("priority_alpha", 2.0))
        cg_floor = float(_cgsa_cfg.get("priority_floor", 0.1))
        cg_maint = float(_cgsa_cfg.get("maintenance_weight", 0.01))
        # SPRT log-likelihood increments for H0 p=.15 vs H1 p=.60.
        _SPRT_S, _SPRT_F = 1.386, -0.754
        _SPRT_ACC, _SPRT_REJ = 4.554, -2.996
        # key -> {p, att, succ, rate, lam, welded}; "__ENTRANCE__" covers
        # non-archive (start-state) episodes so the entrance anneals too.
        cg_stats: dict = {}
        cg_env_cell: list = [None] * num_envs
        cg_env_start_gx = np.full(num_envs, -1, dtype=np.int64)
        # Steps lived since the tag — episodes cut younger than this at a
        # boundary are DROPPED, not scored: a late-rollout restart gets <30
        # steps before the boundary and would be counted a failure it never
        # had time to avoid, biasing every zone's rate downward.
        cg_env_steps = np.zeros(num_envs, dtype=np.int32)
        CG_MIN_SCORE_STEPS = 20
        # Per-env sticky prob consulted by the rollout. Without CGSA it stays
        # at the scalar (behavior unchanged); with CGSA each episode runs at
        # its restart cell's current curriculum noise.
        _sticky_p_env = np.full(num_envs, _sticky_p, dtype=np.float64)
        _cgsa_sel_cache: dict = {"it": -1, "keys": None, "cum": None}

        def _cg_entry(key):
            st = cg_stats.get(key)
            if st is None:
                st = {"p": 0.0, "att": 0, "succ": 0, "rate": 0.0,
                      "lam": 0.0, "welded": False}
                cg_stats[key] = st
            return st

        def _cg_zone_of_key(k):
            """Archive key (area, phase, yband, gxbucket) -> spatial ZONE
            (area, gxbucket, yband). The curriculum attaches to zones — the
            researcher's 32x32 px (gx, gy) bins — NOT to phase-augmented
            archive keys: the 8x phase multiplier diluted the v1 run to ~4
            attempts/cell against a 50-attempt window (anneal never fired)."""
            try:
                return (k[0], k[3], k[2])
            except Exception:
                return k

        # Active weld-frontier cohort size: the K lowest-(area, gx) un-welded
        # zones get the priority mass; everything deeper waits, welded zones
        # get maintenance. Without this, attempts spread uniformly over ~800
        # zones (~0.15/zone/iter — a 25-attempt window takes 150+ iters) and
        # the curriculum starves; with it, the weld proceeds entrance-forward
        # the same way the honest eval composes.
        CG_COHORT_K = int(_cgsa_cfg.get("cohort_k", 40))

        def _cgsa_select_state():
            """Priority-sample a restart ZONE by W(z) over the weld-frontier
            cohort, then a random archive cell within it; returns
            (zone_key, state_bytes). The entrance is a pseudo-zone whose
            member state is the profile start state — once welded it drops
            to maintenance like any other zone (no fixed restart share)."""
            cells = go_explore_archive.cells
            if not cells:
                return ("__ENTRANCE__", _start_bytes)
            if _cgsa_sel_cache["it"] != it:  # refresh once per iter
                zone_cells: dict = {}
                for _k in cells.keys():
                    zone_cells.setdefault(_cg_zone_of_key(_k), []).append(_k)
                # entrance pseudo-zone sorts FIRST (area 0 < any real area)
                zone_cells["__ENTRANCE__"] = None
                def _zsort(z):
                    return (0, 0) if z == "__ENTRANCE__" else (z[0], z[1])
                unwelded = sorted(
                    (z for z in zone_cells
                     if not cg_stats.get(z, {}).get("welded", False)),
                    key=_zsort,
                )
                cohort = set(unwelded[:CG_COHORT_K])
                zones = list(zone_cells.keys())
                wts = np.empty(len(zones), dtype=np.float64)
                for _j, _z in enumerate(zones):
                    st = cg_stats.get(_z)
                    if st is not None and st["welded"]:
                        wts[_j] = cg_maint
                    elif _z in cohort:
                        rate = st["rate"] if st is not None else 0.0
                        wts[_j] = (1.0 - rate) ** cg_alpha + cg_floor
                    else:
                        wts[_j] = cg_maint  # beyond the frontier: wait
                _cgsa_sel_cache.update(
                    it=it, zones=zones, zone_cells=zone_cells,
                    cum=np.cumsum(wts / wts.sum()),
                )
            zones = _cgsa_sel_cache["zones"]
            _z = zones[int(np.searchsorted(
                _cgsa_sel_cache["cum"], np.random.random()
            ))]
            if _z == "__ENTRANCE__":
                return ("__ENTRANCE__", _start_bytes)
            _members = _cgsa_sel_cache["zone_cells"][_z]
            _k = _members[np.random.randint(len(_members))]
            cell = cells.get(_k)
            return (_z, cell.state) if cell is not None else (None, None)

        def _cg_finish_episode(i: int, min_steps: int = 0) -> None:
            """Score the ended episode against its restart cell and run the
            annealing + SPRT bookkeeping."""
            key = cg_env_cell[i]
            if key is None:
                return
            if cg_env_steps[i] < min_steps:
                # Too young to judge (boundary-cut) — drop, don't score.
                cg_env_cell[i] = None
                cg_env_start_gx[i] = -1
                cg_env_steps[i] = 0
                return
            st = _cg_entry(key)
            success = (
                cg_env_start_gx[i] >= 0
                and int(max_x_reached[i]) >= int(cg_env_start_gx[i]) + cg_zone_px
            )
            st["att"] += 1
            st["succ"] += int(success)
            if st["p"] >= cg_target and not st["welded"]:
                st["lam"] += _SPRT_S if success else _SPRT_F
                if st["lam"] >= _SPRT_ACC:
                    st["welded"] = True
                elif st["lam"] <= _SPRT_REJ:
                    st["lam"] = 0.0
                    st["p"] = max(0.0, st["p"] - cg_step)  # reject: back off
            if st["att"] >= cg_window:
                st["rate"] = st["succ"] / st["att"]
                st["windows"] = st.get("windows", 0) + 1
                if st["rate"] >= 0.75:
                    st["p"] = min(cg_target, st["p"] + cg_step)
                elif st["rate"] < 0.30 and st["p"] > 0.0:
                    st["p"] = max(0.0, st["p"] - cg_step)
                st["att"] = 0
                st["succ"] = 0
            cg_env_cell[i] = None
            cg_env_start_gx[i] = -1
            cg_env_steps[i] = 0

        if cgsa_on:
            _sticky_p_env[:] = 0.0  # every cell (incl. entrance) starts at 0
            # Resume the curriculum: without this, a process restart zeroes
            # every zone's p/rate/weld and the anneal starts over.
            try:
                import json as _json
                import ast as _ast
                _cg_path = self.checkpoint_dir / "cgsa_stats.json"
                if _cg_path.exists():
                    with open(_cg_path) as _f:
                        _raw = _json.load(_f)
                    for _ks, _v in _raw.items():
                        try:
                            _key = (_ks if _ks == "__ENTRANCE__"
                                    else tuple(_ast.literal_eval(_ks)))
                        except Exception:
                            _key = _ks
                        cg_stats[_key] = _v
                    log.info(
                        "[vanilla_ppo] CGSA: resumed curriculum (%d zones, "
                        "%d welded)", len(cg_stats),
                        sum(1 for s in cg_stats.values() if s.get("welded")),
                    )
            except Exception as e:
                log.warning("[vanilla_ppo] CGSA stats reload failed: %s", e)
            log.info(
                "[vanilla_ppo] CGSA ENABLED: target=%.2f step=%.2f window=%d "
                "zone=%dpx alpha=%.1f (per-cell noise curriculum + SPRT welds)",
                cg_target, cg_step, cg_window, cg_zone_px, cg_alpha,
            )

        # Core PPO-update owner (trainer-decomposition plan, Task 2): the
        # RND/count fold -> GAE-lambda -> valid-mask advantage norm ->
        # K-epoch minibatch PPO update lives in PPOUpdater. Built once so
        # its lazily-resolved SHAPO actor set persists across iters (it
        # replaces the old `_sam_actor_params = None` run-scoped local).
        # The un-net-covered PR-MDP / CGSA / backward blocks stay below.
        _ppo_updater = PPOUpdater(self)

        for it in range(num_iters):
            if not self._running:
                break
            global_it = it + iter_offset
            _iter_t0 = time.perf_counter()
            # Per-iter pool-max screen for screen_XX reward games (Contra).
            # Surfaces progress in metrics even before a stage-end is known.
            iter_max_screen = 0
            # KL anchor: refresh beta + the freeze flag from the global
            # env-step count (absolute iter, so a resume continues the
            # schedule), and reset the per-iter D_KL accumulator.
            if self._kl_anchor is not None:
                self._kl_anchor.set_env_steps(
                    global_it * rollout_steps * num_envs
                )
            _kl_div_sum = 0.0
            _kl_div_n = 0
            _kl_step = None
            _kl_pen_step = None
            # Apply any GUI audio-mode pace change here (trainer thread,
            # between rollouts) rather than from the drainer thread.
            self._drain_pending_pace()

            # ============== ROLLOUT COLLECTION ==============
            net.eval()
            # Recurrent: GRU hidden state threaded across rollout steps,
            # reset per env on each episode boundary (done). None for the
            # feedforward path (left fully unchanged).
            h_rollout = (
                net.initial_hidden(num_envs, self.device)
                if self._recurrent else None
            )
            # Buffers are reused across iters; reset the validity mask so a
            # slot is only counted valid if this iter's rollout wrote it.
            valid_buf[:] = False
            # Per-step critic values / taken log-probs stay on-device
            # during the rollout and drain to value_buf / log_prob_buf in
            # ONE fused transfer after the loop (see below). Neither is
            # consumed until GAE / the update, so the old per-step .cpu()
            # calls were two avoidable MPS queue drains per step.
            value_steps: list[torch.Tensor] = []
            log_prob_steps: list[torch.Tensor] = []
            adv_value_steps: list[torch.Tensor] = []
            exec_mask_buf[:] = False
            trunc_buf[:] = False
            # Commitment options: the timer lives HERE, not in the module,
            # so the update-pass network stays stateless. Reset at iter
            # start — a commitment crossing the iter boundary is truncated
            # once per 1,536 steps, a negligible and unbiased edge.
            _commit_on = getattr(self, "_commit_durations", None) is not None
            if _commit_on:
                _cd = np.asarray(self._commit_durations, dtype=np.int64)
                _commit_pair = np.zeros(num_envs, dtype=np.int64)
                _commit_left = np.zeros(num_envs, dtype=np.int64)
                _commit_held_buf = np.zeros((rollout_steps, num_envs),
                                            dtype=np.bool_)
                _entw_buf = np.ones((rollout_steps, num_envs),
                                    dtype=np.float32)
            if _kadv is not None:
                _kadv.begin_iter()
            with torch.no_grad():
                for t in range(rollout_steps):
                    if not self._running:
                        break
                    # Pack current observations straight into the
                    # preallocated rollout slot (no intermediate array),
                    # then build the policy-input tensor from that slot.
                    # `.float()` copies, so obs_buf[t] keeps the raw obs
                    # for the update phase.
                    np.stack(stacked_obs, axis=0, out=obs_buf[t])
                    batch_t = (
                        torch.from_numpy(obs_buf[t])
                        .to(self.device).float()
                    )
                    # /255 only for uint8 pixel obs. With preprocess_f16 the
                    # pool already delivers float16 in [0,1] (the GA path
                    # gates the same way) — dividing again scales obs by 255x.
                    if not self._is_tile_mode and not self.preprocess_f16:
                        batch_t = batch_t.div_(255.0)

                    # Single forward pass: actor + critic.
                    _fwd_t0 = time.perf_counter_ns()
                    if self._recurrent:
                        logits, values, h_rollout_next = net.forward_ac_recurrent(
                            batch_t, h_rollout
                        )
                    else:
                        logits, values = net.forward_ac(batch_t)
                    # Sample the categorical action on CPU. On MPS
                    # `torch.multinomial` on the tiny (num_envs,
                    # num_actions) logits decomposes into ~10 serial
                    # primitive kernels (~0.83 ms/step, flat with batch —
                    # dispatch-bound), costing more than the CNN forward
                    # itself; the CPU draw on the same shape is ~0.008 ms.
                    # Pulling the tiny logits device→host REPLACES the
                    # mandatory per-step actions sync a few lines down (it
                    # does not add a transfer — the old `actions.cpu()`
                    # drained the same MPS queue). `values` stays on device
                    # for the fused post-loop drain. log_softmax / exp /
                    # gather run on the SAME CPU distribution we sample
                    # from, so `log_probs_taken == log_probs_all[i, action]`.
                    # Note: draws now consume the CPU default generator
                    # instead of the MPS Philox stream — statistically
                    # identical, but a seeded run does not bit-reproduce
                    # pre-change trajectories.
                    logits_cpu = logits.float().cpu()
                    log_probs_all = F.log_softmax(logits_cpu, dim=-1)
                    probs = log_probs_all.exp()
                    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    if _commit_on:
                        # Hold the committed pair for its duration. Held
                        # rows keep executing the SAME primitive (their
                        # pair maps to it via the expanded bitmask table)
                        # and are marked held so the result loop leaves
                        # them invalid — the update sees only decisions.
                        _acts_np = actions.numpy()
                        _held = _commit_left > 0
                        _acts_np[_held] = _commit_pair[_held]
                        _new = ~_held
                        _commit_pair[_new] = _acts_np[_new]
                        _commit_left[_new] = _cd[_acts_np[_new] % len(_cd)]
                        _commit_left -= 1
                        _commit_held_buf[t] = _held
                        _entw_buf[t] = np.where(
                            _held, 1.0,
                            _cd[_acts_np % len(_cd)].astype(np.float32))
                    log_probs_taken = log_probs_all.gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)

                    if self._kl_anchor is not None:
                        # State-based D_KL(prior || pi) on the SAME
                        # distribution the sample came from (pre-override —
                        # sticky/adversary rewrites change the executed
                        # action, not pi). The reward-level penalty is
                        # applied per env in the result loop below, only
                        # after the actor unfreezes (while frozen pi IS the
                        # prior, so the KL is identically ~0 anyway).
                        _kl_lp = log_probs_all
                        if _commit_on:
                            # The anchor prior is over PRIMITIVES; compare
                            # it to the policy's primitive MARGINAL:
                            # logsumexp over each primitive's duration
                            # group. Pair-major layout makes that a
                            # reshape. Same anchor semantics ("stay near
                            # the banked primitive distribution"), so the
                            # arms stay comparable — durations are exactly
                            # the treatment's added freedom and are
                            # correctly NOT penalized by the anchor.
                            _kl_lp = torch.logsumexp(
                                log_probs_all.reshape(
                                    log_probs_all.shape[0], -1, len(_cd)),
                                dim=-1)
                        _kl_step = self._kl_anchor.kl_divergence(
                            batch_t, _kl_lp
                        )
                        _kl_pen_step = (
                            None if self._kl_anchor.frozen
                            else self._kl_anchor.beta * _kl_step
                        )

                    if (_sticky_p > 0.0 or cgsa_on) and t > 0:
                        # Per-env sticky: with CGSA each env rolls at its
                        # restart cell's curriculum noise; without it the
                        # vector holds the scalar everywhere (unchanged).
                        # Envs that restarted on the previous step are
                        # dropped from the roll when the boundary guard is
                        # on (same single RNG draw either way).
                        _stick_rows = _sticky_boundary.override_rows(
                            _sticky_p_env
                        )
                        if _stick_rows.size:
                            _rows_t = torch.from_numpy(_stick_rows)
                            _prev_t = torch.from_numpy(
                                _prev_exec_action[_stick_rows]
                            )
                            actions[_rows_t] = _prev_t
                            # Clamp: a near-deterministic policy gives a
                            # stuck action ~0 probability, and an
                            # unclamped log-prob of -30..-46 explodes the
                            # PPO ratio into NaN weights (observed: warm-
                            # started 1-1 collapsed to NaN in 56 iters).
                            # Floor at ~2e-6 probability.
                            log_probs_taken[_rows_t] = torch.clamp(
                                log_probs_all[_rows_t, _prev_t], min=-13.0
                            )
                    if _sticky_p > 0.0 or cgsa_on:
                        _prev_exec_action[:] = actions.numpy()
                        # The roll above (or, at t == 0, the absence of
                        # one) has spent the suppression these envs were
                        # owed — clear it so the NEXT step sticks normally.
                        _sticky_boundary.consume()

                    if prmdp_on:
                        # Adversary forward on the same obs; its action
                        # executes on alpha of the rows. Protagonist's
                        # taken log-prob follows the executed action
                        # (clamped — the house sticky pattern; unclamped
                        # near-zero-prob overrides NaN the PPO ratio).
                        # Uniform mode (v7): same override channel, the
                        # actions are uniform draws, nothing to train.
                        if _adv_net is not None:
                            adv_logits, adv_values = _adv_net.forward_ac(batch_t)
                            adv_lp_all = F.log_softmax(
                                adv_logits.float().cpu(), dim=-1
                            )
                            adv_actions = torch.multinomial(
                                adv_lp_all.exp(), 1
                            ).squeeze(1)
                        else:
                            adv_actions = torch.randint(
                                0, int(log_probs_all.shape[1]),
                                (num_envs,), dtype=torch.long,
                            )
                        _ov_rows = np.nonzero(
                            np.random.random(num_envs) < _prmdp_alpha
                        )[0]
                        if _ov_rows.size:
                            _ov_t = torch.from_numpy(_ov_rows)
                            _ov_a = adv_actions[_ov_t]
                            actions[_ov_t] = _ov_a
                            log_probs_taken[_ov_t] = torch.clamp(
                                log_probs_all[_ov_t, _ov_a], min=-13.0
                            )
                            exec_mask_buf[t, _ov_rows] = True
                        if _adv_net is not None:
                            adv_action_buf[t] = adv_actions.numpy()
                            adv_logp_buf[t] = torch.clamp(
                                adv_lp_all.gather(
                                    1, adv_actions.unsqueeze(1)
                                ).squeeze(1), min=-13.0,
                            ).numpy()
                            adv_value_steps.append(adv_values)

                    if _kadv is not None:
                        # Kernel-matched adversary: pass/repeat decision on
                        # every env; REPEAT swaps in the previous EXECUTED
                        # action and the tracker advances post-override
                        # (decide() owns both — see kernel_adversary.py).
                        _kadv.decide(
                            t, batch_t, actions, log_probs_all,
                            log_probs_taken, _prev_exec_action,
                        )

                    actions_np = actions.numpy().astype(np.int32)
                    action_buf[t] = actions_np
                    # Accumulate as device tensors; the fused post-loop
                    # drain writes value_buf/log_prob_buf. Plain lists +
                    # torch.stack only — per-step indexed writes into a
                    # preallocated MPS tensor leak via the MPSGraph
                    # kernel cache.
                    value_steps.append(values)
                    log_prob_steps.append(log_probs_taken)
                    self._gen_timer.add(
                        "rollout_forward", time.perf_counter_ns() - _fwd_t0
                    )

                    # Apply actions to envs. Vectorized bitmask gather
                    # (one numpy LUT lookup) replaces a per-env Python
                    # loop of ~num_envs int-unbox + table lookups, ×1024
                    # steps/iter.
                    step_actions[:] = 0
                    step_actions[:num_envs] = self._bitmask_lut[actions_np]
                    _emu_t0 = time.perf_counter_ns()
                    step_results = self.pool.step_all(step_actions)
                    self._gen_timer.add(
                        "emulation", time.perf_counter_ns() - _emu_t0
                    )
                    self._emit_frame_sink(step_results)
                    # Live audio: forward drained APU samples to the GUI
                    # mixer. This push existed only in the GA loop, so
                    # every vanilla_ppo run was structurally silent no
                    # matter what the mixer window said. Same gate as the
                    # GA path: skip entirely when muted (the common
                    # headless state — r.audio is empty then anyway
                    # because no worker has audio enabled).
                    if (self._audio_mixer is not None
                            and self._audio_mixer.mode != "mute"):
                        for r in step_results:
                            if r.audio.size > 0 and r.audio_rate > 0:
                                self._audio_mixer.push_audio(
                                    r.worker_id, r.audio, r.audio_rate)

                    # Process each env's result, compute reward, advance stacker.
                    # Envs already done in this iter contribute zero reward
                    # and done=True (frozen) — GAE will mask them and the
                    # rollout buffer carries valid post-done padding.
                    for i, r in enumerate(step_results):
                        if not active_in_iter[i]:
                            # Already done this iter; keep the rollout
                            # buffer's done flag set so GAE doesn't
                            # bootstrap across the boundary. Zero reward
                            # (nothing earned in this "dead" step).
                            reward_buf[t, i] = 0.0
                            bonus_buf[t, i] = 0.0
                            done_buf[t, i] = True
                            continue
                        ram = r.ram_snapshot
                        # Pack world+level for ordinal max tracking. The
                        # game's world/level bytes are zero-indexed; we
                        # display as 1-indexed "W-L" at log time.
                        wl_packed = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
                        if wl_packed > max_world_level_packed[i]:
                            max_world_level_packed[i] = wl_packed
                        end_world_level_packed[i] = wl_packed
                        x_pos = (int(ram[0x006D]) << 8) | int(ram[0x0086])
                        if x_pos > max_x_reached[i]:
                            max_x_reached[i] = x_pos
                        # CGSA: fill the restart gx lazily on the first REAL
                        # frame after a restore (the restored RAM only arrives
                        # one step later; x reads 0 on the load frame).
                        if cgsa_on and cg_env_cell[i] is not None:
                            cg_env_steps[i] += 1
                            if cg_env_start_gx[i] < 0 and x_pos > 0:
                                cg_env_start_gx[i] = x_pos
                                _cg_entry(cg_env_cell[i]).setdefault("gx", x_pos)
                        bonus_buf[t, i] = _exploration.count_bonus(
                            wl_packed, x_pos
                        )
                        # Ladder mode: track the furthest sub-stage rung this
                        # env reached. `region_of` credits only sequential-path
                        # states (a warp yields OFF_LADDER), so it never rises
                        # on a warp and the frontier can't be gamed off-chain.
                        reg_here = None
                        if ladder_on:
                            reg_here = region_of(ram, ladder)
                            if reg_here > max_region_reached[i]:
                                max_region_reached[i] = reg_here
                                # Go-Explore burst (Lane 5): archive each new
                                # on-ladder region a rollout reaches while a
                                # burst is in flight, so the burst can return
                                # to and explore forward from the deepest
                                # frontier cells and harvest one deeper seed.
                                # The cell's stored score IS its region order,
                                # so the harvest reads depth off it directly.
                                # Warp states (cell == None) are never archived
                                # (§Q5). Guarded by `ge_burst_active`, so this
                                # save_worker_state cost is paid only mid-burst
                                # (expected 0-1 bursts per campaign).
                                if (ge_burst_active and ge_archive is not None
                                        and smb_sequential_cell(ram) is not None):
                                    try:
                                        _geb = self.pool.save_worker_state(i)
                                        if _geb is not None:
                                            ge_archive.record(
                                                bytes(ram), bytes(_geb),
                                                float(reg_here), int(t),
                                            )
                                    except Exception:
                                        pass
                        # Mid-rollout curriculum capture: FIRST detection
                        # per stage wins (no overwrites), capture on ANY
                        # byte strictly greater than the current anchor.
                        #
                        # Previously this required `byte > cur_anchor + 1`
                        # to skip the cutscene transition byte between
                        # 1-1 and 1-2 (sequence 0 → 1 → 2). But NOT every
                        # level transition has a cutscene byte: 1-2 → 1-3
                        # goes directly from byte=2 to byte=3 with no
                        # intermediate. The `+1` skip prevented the
                        # curriculum from ever capturing 1-3 — agents
                        # were reaching byte=3 (real 1-3 gameplay per
                        # GUI observation) but the gate required byte=4+,
                        # leaving the curriculum stuck at stage 1.
                        #
                        # Cost of the looser gate: the rare 1-1→1-2-like
                        # transition may capture the brief cutscene byte
                        # first. That's acceptable — loading from a mid-
                        # cutscene state still plays out naturally
                        # post-restore (cutscene is autopiloted by the
                        # game engine, agent inputs ignored) and Mario
                        # lands in the new level within ~30 RL steps.
                        cur_anchor = (
                            smb_curriculum_anchors[smb_curriculum_stage]
                            if smb_curriculum_stage < len(smb_curriculum_anchors)
                            else 0
                        )
                        # Also gate on lives >= 1 (= 2 in-game lives).
                        # Without this gate, if Mario crosses into the
                        # next level on his LAST life, the captured save
                        # state has lives=0 and every warm-start has
                        # only one attempt before in-game game-over. With
                        # auto-reset on death (below) this still works
                        # but wastes the ~15-step in-game GAME OVER
                        # screen between attempts.
                        if ladder_on:
                            # Warp guard (§Q5 layer 2): a state the admission
                            # predicate rejects (world != 1 or an off-chain
                            # area byte) never seeds the ladder. Capture the
                            # first live state to reach PAST the frontier rung;
                            # its region order is the anchor, and it live-fills
                            # that rung's warm-start seed (disk seeds win).
                            if (smb_pending_capture is None
                                    and reg_here is not None
                                    and reg_here > smb_curriculum_stage
                                    and smb_sequential_cell(ram) is not None
                                    and int(ram[0x075A]) >= 1):
                                try:
                                    blob = self.pool.save_worker_state(i)
                                    if blob is not None:
                                        smb_pending_capture = (
                                            int(reg_here), bytes(blob),
                                        )
                                        if smb_curriculum_states[reg_here] is None:
                                            smb_curriculum_states[reg_here] = bytes(blob)
                                except Exception:
                                    pass
                        elif (self._smb_curriculum_active
                                and not consolidate_on
                                and smb_pending_capture is None
                                and wl_packed > cur_anchor
                                and int(ram[0x075A]) >= 1):
                            try:
                                blob = self.pool.save_worker_state(i)
                                if blob is not None:
                                    smb_pending_capture = (
                                        int(wl_packed), bytes(blob),
                                    )
                            except Exception:
                                pass
                        # Snapshot the cumulative reward breakdown BEFORE
                        # compute so the narrator can diff it against the
                        # post-step breakdown to caption new events (death,
                        # item, clear). GUI-only; None in headless.
                        prev_bd = dict(reward_fns[i].breakdown) if narrator_on else None
                        reward, rew_done, level_id = reward_fns[i].compute(ram, action=int(step_actions[i]))
                        # screen_XX reward games (Contra) expose progress
                        # via the level_id. Track the per-iter pool max so
                        # advancement is visible in metrics, and so the
                        # real stage-1-end screen can be read off a run to
                        # arm ContraReward's clear_screen threshold.
                        if level_id[:7] == "screen_":
                            try:
                                _sc = int(level_id[7:], 16)
                                if _sc > iter_max_screen:
                                    iter_max_screen = _sc
                            except ValueError:
                                pass
                        done = bool(r.done) or bool(rew_done)
                        # ===== BACKWARD CURRICULUM: the per-rung budget =====
                        # The one cut the vanilla_ppo loop never had. An
                        # attempt that neither dies nor clears otherwise runs
                        # to the iter boundary, where it is censored — B5 run
                        # 2 burned 156 iters at one rung that way. End it HERE
                        # instead, at a cap sized by how much tape is left
                        # ahead of its restart, so the attempt resolves inside
                        # the rollout and the next one starts immediately.
                        # Marked as a truncation (not a death) so the scoring
                        # site can tell them apart; ending it BEFORE the
                        # wavefront block below is deliberate — a non-clearing
                        # terminal is charged -peak(Phi) there, so a budgeted
                        # episode cannot bank approach shaping by idling.
                        if (
                            bwd_budget is not None
                            and not done
                            and _bwd_env_src[i] >= 0
                            and _bwd_env_budget[i] > 0
                            and ep_lengths[i] + 1 >= _bwd_env_budget[i]
                        ):
                            done = True
                            _bwd_env_trunc[i] = True
                        # NON-FARMABLE wavefront PBRS (research fix 2026-07-22).
                        # Positive-shifted Phi in [0, phi_target]; Phi(terminal)=0.
                        #   LIVE : F = gamma*Phi(s') - Phi(s)
                        #   DEATH: F = gamma*0 - Phi(s) = -Phi(s)  <- cancels the
                        #          earned approach shaping, so progress-then-die
                        #          telescopes to <= 0 and CANNOT be farmed.
                        #   CLEAR: no shaping term (the +50 completion is the
                        #          reward; a positive terminal Phi would double it).
                        # The prior form skipped the terminal term on death, which
                        # let greedy bank +168 of un-canceled approach shaping then
                        # die at gx 183 (confirmed by the 3-mode diagnostic).
                        if wave_pot is not None:
                            if not done:
                                _phi = wave_pot.potential(ram)
                                if wave_monotone:
                                    # Monotone rule: F pays only on a NEW
                                    # episode peak of Phi (the peak-augmented
                                    # potential); retreat/idle steps pay 0 at
                                    # gamma=1. Peak starts at 0 per episode,
                                    # so the death charge below refunds the
                                    # whole stream exactly (Grzes).
                                    _wf, _wpk = monotone_wave_step(
                                        float(wave_peak_phi[i]), _phi,
                                        wave_pot.gamma,
                                    )
                                    reward += _wf
                                    wave_peak_phi[i] = _wpk
                                    wave_prev_phi[i] = _phi
                                else:
                                    if wave_prev_phi[i] is not None:
                                        reward += wave_pot.gamma * _phi - wave_prev_phi[i]
                                    wave_prev_phi[i] = _phi
                                    if _phi > wave_peak_phi[i]:
                                        wave_peak_phi[i] = _phi
                                # Lost-cut: linger off the solution envelope
                                # (Phi~0) for WAVE_LOST_K consecutive steps ->
                                # terminate as a non-clear (charge -peak below
                                # via the done branch on THIS step). Under the
                                # monotone rule the cut is a TRUNCATION: no
                                # terminal charge, GAE bootstraps V(s).
                                if _phi <= 0.5:
                                    wave_lost_count[i] += 1
                                    if wave_lost_count[i] >= WAVE_LOST_K:
                                        done = True
                                        rew_done = True
                                        if wave_monotone:
                                            trunc_buf[t, i] = True
                                else:
                                    wave_lost_count[i] = 0
                            if done:
                                # A clear grows the `completion` breakdown this
                                # step (same diff the clear-counter uses below).
                                # CLEAR -> no shaping term. Any OTHER terminal
                                # (death, timeout, lost-cut) -> charge -PEAK Phi:
                                # the telescoped sum over any non-completing
                                # episode is <= 0 no matter where it ends, so
                                # neither dying, idling, nor hiding in a Phi=0
                                # region can bank shaping.
                                _cur_comp = float(
                                    reward_fns[i].breakdown.get("completion", 0.0)
                                )
                                _is_clear = _cur_comp > prev_completion_total[i] + 1e-6
                                if wave_monotone:
                                    # TRUE deaths charge -peak; clears and
                                    # truncations (the lost-cut) charge 0.
                                    reward += wave_terminal_charge(
                                        float(wave_peak_phi[i]),
                                        is_clear=_is_clear,
                                        truncated=bool(trunc_buf[t, i]),
                                    )
                                elif not _is_clear and wave_peak_phi[i] > 0.0:
                                    reward += -float(wave_peak_phi[i])
                                wave_prev_phi[i] = None
                                wave_peak_phi[i] = 0.0
                                wave_lost_count[i] = 0
                        # A budget cut is a TIMEOUT for the critic too:
                        # flag it so GAE bootstraps V(s) instead of 0 at
                        # the cap (the Pardo time-limit correction the
                        # lost-cut already gets), matching the scheduler's
                        # censored-timeout accounting. Written AFTER the
                        # wavefront block so the terminal charge above
                        # still treats the cut as a non-clear (-peak) —
                        # idling into the cap banks no shaping either way.
                        if done and _bwd_env_trunc[i]:
                            trunc_buf[t, i] = True
                        # Caption new events for the live stream. Per-env
                        # (one policy across N envs); the narrator's
                        # min-event-gap rate-limits so the grid doesn't spam.
                        if narrator_on:
                            # Pass the COMBINED done: SMB/Contra deaths and
                            # clears fire via the reward fn (rew_done), not
                            # the env-side r.done — so the narrator's
                            # death/success captions need both, or it stays
                            # silent for non-Zelda games.
                            _ep_done = bool(r.done) or bool(rew_done)
                            self._narrator.observe(
                                worker_id=i,
                                genome_name=f"agent-{i}",
                                prev_breakdown=prev_bd,
                                new_breakdown=reward_fns[i].breakdown,
                                done=_ep_done,
                                success=(
                                    reward_fns[i].episode_success()
                                    if _ep_done else False
                                ),
                            )
                        if _kl_pen_step is not None:
                            # Reward-level KL penalty (post-unfreeze): keep
                            # pi near the anchor prior where it has no
                            # better evidence, at the decayed beta.
                            reward -= float(_kl_pen_step[i])
                        if _kl_step is not None:
                            _kl_div_sum += float(_kl_step[i])
                            _kl_div_n += 1
                        reward_buf[t, i] = reward
                        done_buf[t, i] = done
                        # Commitment options: held (non-decision) rows stay
                        # INVALID — their reward/value/done still feed the
                        # per-step GAE recursion (dense critic view), but
                        # the update never treats them as sampled actions.
                        # A done also cuts the commitment so the next step
                        # of that env decides fresh.
                        if _commit_on:
                            valid_buf[t, i] = not _commit_held_buf[t, i]
                            if done:
                                _commit_left[i] = 0
                        else:
                            valid_buf[t, i] = True  # real executed step (incl. death)
                        ep_returns[i] += reward
                        ep_lengths[i] += 1
                        if sil_on:
                            # Copy: obs_buf slots are reused across iters.
                            sil_ep_obs[i].append(obs_buf[t, i].copy())
                            sil_ep_act[i].append(int(action_buf[t, i]))
                        # Detect a completion bonus firing on this step
                        # by diffing the reward fn's cumulative
                        # `completion` breakdown against the last seen
                        # value. A positive delta = the agent touched a
                        # flagpole this step (or a level was cleared by
                        # whatever other completion mechanism applies).
                        try:
                            # `.breakdown` is already a dict — read the
                            # key directly instead of copying it into a
                            # new dict every live-env step.
                            cur_comp = float(
                                reward_fns[i].breakdown.get("completion", 0.0)
                            )
                            if cur_comp > prev_completion_total[i] + 1e-6:
                                n_clears_this_iter += 1
                                # Same trusted signal for the backward
                                # cursor's rung statistics (scored at the
                                # restart, which is where the episode's
                                # provenance is still known).
                                _bwd_env_clear[i] = True
                                # Flag touched this step = this episode cleared
                                # its level. Record the PLR success NOW (a clear
                                # may never reach `done`), once per episode.
                                if plr_on and not episode_recorded[i]:
                                    try:
                                        plr_ctx.record(plr_ctx.env_level[i], True)
                                    except Exception:
                                        pass
                                    episode_recorded[i] = True
                                # Same trusted signal feeds the SIL buffer:
                                # this episode CLEARED — bank its trajectory.
                                # Last in the block so a flush failure can
                                # never eat the clear/PLR bookkeeping above
                                # (this try swallows exceptions).
                                _sil_flush_clear(i)
                            prev_completion_total[i] = cur_comp
                        except Exception:
                            pass

                        # ===== GO-EXPLORE RECORD =====
                        # Discretize this state into a cell and remember the
                        # best-known way to reach it. Bound cost by only
                        # paying for pool.save_worker_state on a NEW cell or a
                        # strict improvement (peek the archive first). Skip
                        # death states — they make useless return targets.
                        if go_explore_archive is not None and not done:
                            _exploration.record_cell(
                                i, ram, _ge_score(i), int(ep_lengths[i]),
                            )

                        # Set True when an auto-reset re-seeds the
                        # stacker this step, so the post-step push below
                        # is skipped (the seed already IS the new obs).
                        reseeded = False
                        if done:
                            if episode_metrics_on:
                                # Per-episode identity (worker i, this
                                # episode's own return/length/final_x)
                                # is only ever available HERE, for the
                                # rest of this one iteration of the loop
                                # — completed_returns/completed_lengths
                                # right below fold it into an unlabeled
                                # list, and the generation-level emit
                                # further down folds those into a mean/
                                # max that no longer says which worker
                                # contributed what. Every field is read,
                                # not derived: `x_pos` is this exact
                                # step's position (already computed
                                # above for max_x_reached), and the two
                                # done flags are logged RAW and
                                # separately rather than folded into one
                                # `died` boolean. That split is load-
                                # bearing, not pedantry: per the
                                # "r.done vs rew_done" note near the
                                # narrator call above, SMB/Contra deaths
                                # and clears fire through the reward fn
                                # (`rew_done`) while the env-side
                                # `r.done` stays False, so a lone
                                # `died=r.done` column would read False
                                # on every SMB episode ever logged. A
                                # consumer reconstructs what it needs:
                                # a done with neither flag set is the
                                # backward-curriculum budget truncation.
                                # A level-clear flag is left out here on
                                # purpose: the closest existing signal
                                # (`_bwd_env_clear`) isn't reset per-
                                # episode outside the backward-
                                # curriculum path, so on a multi-
                                # episode-per-slot rollout (auto-reset
                                # on death) it can still read True from
                                # an earlier episode in the same slot —
                                # shipping that would misattribute a
                                # clear to an episode that didn't clear.
                                self._metrics_sink.emit_episode(
                                    generation=global_it,
                                    worker_id=i,
                                    episode_return=float(ep_returns[i]),
                                    episode_length=int(ep_lengths[i]),
                                    final_x=int(x_pos),
                                    env_done=bool(r.done),
                                    reward_done=bool(rew_done),
                                )
                            completed_returns.append(float(ep_returns[i]))
                            completed_lengths.append(int(ep_lengths[i]))
                            ep_returns[i] = 0.0
                            ep_lengths[i] = 0
                            # SIL: a done that reaches here without having
                            # flushed at the clear diff above did NOT clear
                            # — drop the episode's accumulator.
                            _sil_drop_episode(i)
                            if plr_on and not episode_recorded[i]:
                                # Episode ended WITHOUT clearing (a death) and
                                # wasn't already recorded at a flag-touch above
                                # — record the failure so PLR up-weights this
                                # level. env_stage[i] is unchanged, so the
                                # reload below re-seeds the SAME level (a fresh
                                # level is drawn at the next iter boundary).
                                try:
                                    plr_ctx.record(plr_ctx.env_level[i], False)
                                except Exception:
                                    pass
                            # Start the next episode's accounting fresh.
                            episode_recorded[i] = False
                            # AUTO-RESET on death within a rollout. If the
                            # curriculum has a warm-start state for the
                            # current stage, immediately reload it so this
                            # env gets another attempt at the level within
                            # the same rollout. Each rollout now contains
                            # 5-6 attempts per env instead of 1. Standard
                            # PPO baseline behavior — done is just an
                            # episode boundary in GAE, the rollout buffer
                            # accumulates multi-episode data naturally.
                            #
                            # If we're at stage 0 (no warm-state), fall
                            # back to the old freeze-until-iter-boundary
                            # behavior — there's no per-env cold-boot
                            # reset API in the pool. With mixed-stage
                            # warm-start, each env reloads ITS OWN stage
                            # (env_stage[i]), not the global current one.
                            env_k = int(env_stage[i])
                            current_stage_state_inline = (
                                smb_curriculum_states[env_k]
                                if 0 < env_k < len(smb_curriculum_states)
                                else None
                            )
                            if current_stage_state_inline is not None:
                                try:
                                    self.pool.load_worker_state(
                                        i, current_stage_state_inline,
                                    )
                                    _sticky_restart(i)
                                    reward_fns[i].reset()
                                    # reset() zeroes the cumulative
                                    # `completion` breakdown, so re-arm the
                                    # clear detector too — otherwise the
                                    # stale (higher) prev value makes every
                                    # post-reset clear fail the diff at
                                    # ~4333 and success_rate under-reports
                                    # by up to ~5-6x at curriculum stage>=1.
                                    prev_completion_total[i] = 0.0
                                    # Re-seed the stacker from the
                                    # canonical stage-start observation
                                    # captured at the iter-boundary
                                    # warm-start. load_worker_state
                                    # restores RAM but produces no frame
                                    # until the next step_all, so pushing
                                    # the pre-reset death frame here would
                                    # desync the observation from the
                                    # restored env (the policy would pick
                                    # the next action from a death frame
                                    # while the env is at level start).
                                    # Seeding makes the first post-reset
                                    # obs identical to a fresh episode's.
                                    seed_i = stage_seed_results[i]
                                    if seed_i is not None:
                                        if self._is_tile_mode:
                                            stacked_obs[i] = tile_stackers[i].reset(
                                                self._tile_extractor.extract(
                                                    seed_i.ram_snapshot
                                                )
                                            )
                                        else:
                                            stacked_obs[i] = stackers[i].reset(
                                                seed_i.frame,
                                                getattr(
                                                    seed_i,
                                                    "preprocessed", None,
                                                ),
                                            )
                                        reseeded = True
                                    else:
                                        # No stage-start seed yet (it=0
                                        # on resume, before the first
                                        # warm-start). Freeze rather than
                                        # re-seed from a stale frame.
                                        active_in_iter[i] = False
                                        try:
                                            self.pool.set_worker_done(i, True)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    log.warning(
                                        "[vanilla_ppo] auto-reset env %d "
                                        "failed: %s — falling back to "
                                        "freeze-on-done.", i, e,
                                    )
                                    active_in_iter[i] = False
                                    try:
                                        self.pool.set_worker_done(i, True)
                                    except Exception:
                                        pass
                            elif _start_bytes is not None and self._is_tile_mode:
                                # Stage 0 with a configured start state:
                                # inline restart instead of freezing. The
                                # old freeze-on-done wasted ~95% of env
                                # slots at rollout 4096 (one ~200-step
                                # episode then frozen padding) — this is
                                # the single biggest attempts-per-hour
                                # multiplier on the level. Stacker re-seed
                                # is deferred one step (post-restore RAM
                                # arrives on the next step_all).
                                _restart_bytes = _start_bytes
                                if cgsa_on:
                                    # Score the episode that just ended vs
                                    # its restart cell, then draw the next
                                    # restart from the weld-frontier selector
                                    # and adopt that zone's curriculum noise.
                                    _cg_finish_episode(i)
                                    max_x_reached[i] = 0
                                    _k, _ge_blob = _cgsa_select_state()
                                    if _ge_blob is not None:
                                        _restart_bytes = _ge_blob
                                        cg_env_cell[i] = _k
                                        _sticky_p_env[i] = _cg_entry(_k)["p"]
                                    else:
                                        cg_env_cell[i] = "__ENTRANCE__"
                                        _sticky_p_env[i] = _cg_entry(
                                            "__ENTRANCE__"
                                        )["p"]
                                    cg_env_start_gx[i] = -1
                                elif bwd_on:
                                    # Reverse curriculum: score the life
                                    # that just ended against the rung it
                                    # started on, walk the cursor back if
                                    # that rung is earned, then draw this
                                    # life's restart UNIFORMLY from the
                                    # window behind the cursor — with the
                                    # true entrance in the pool, so the
                                    # honest start is sampled all run and
                                    # tau 0 needs no hand-off. (The sticky
                                    # boundary is handled once for every
                                    # branch by _sticky_restart below.)
                                    _bwd_ok = bool(_bwd_env_clear[i])
                                    if _bwd_env_trunc[i] and not _bwd_ok:
                                        # Cut by the per-rung budget: a
                                        # timeout, not a death.
                                        _bwd_score_truncation(
                                            i, int(_bwd_env_src[i]))
                                    elif _bwd_env_src[i] == 1:
                                        bwd_sched.record(
                                            _bwd_ok, tau=int(_bwd_env_tau[i]),
                                        )
                                    elif _bwd_env_src[i] == 0:
                                        bwd_sched.record_entrance(_bwd_ok)
                                    _bwd_env_clear[i] = False
                                    _bwd_env_trunc[i] = False
                                    bwd_sched.maybe_advance()
                                    _bwd_pick = bwd.ENTRANCE
                                    if not bwd_pin:
                                        _bwd_win = _bwd_window(
                                            bwd_tape, bwd_sched.tau,
                                            window_frames=bwd_window_frames,
                                            frames_per_step=(
                                                bwd_frames_per_entry
                                            ),
                                        )
                                        _bwd_pick = bwd.draw_restart(
                                            len(_bwd_win), bwd_entrance_w,
                                            float(np.random.random()),
                                        )
                                        if _bwd_pick != bwd.ENTRANCE:
                                            _restart_bytes = bwd_blobs[
                                                _bwd_win[_bwd_pick]
                                            ]
                                    _bwd_env_tau[i] = bwd_sched.tau
                                    _bwd_env_src[i] = (
                                        0 if _bwd_pick == bwd.ENTRANCE else 1
                                    )
                                    # Budget the fresh attempt by the rung it
                                    # actually restarts from (the entrance is
                                    # rung 0), not by the cursor: the window
                                    # spans entries behind tau.
                                    _bwd_env_budget[i] = (
                                        bwd_budget.steps_for(
                                            0 if _bwd_pick == bwd.ENTRANCE
                                            else int(_bwd_win[_bwd_pick])
                                        ) if bwd_budget is not None else 0
                                    )
                                elif (
                                    go_explore_archive is not None
                                    and _ge_inline_p > 0.0
                                    and len(go_explore_archive) > 0
                                    and np.random.random() < _ge_inline_p
                                ):
                                    # First-return-then-explore on the
                                    # inline path: restart this life from
                                    # an archive frontier cell instead of
                                    # the level entrance, concentrating
                                    # attempt density where progress
                                    # actually stalls. The checkpoint
                                    # cursor arms at the restored x
                                    # (rewards.rs first-step arming), so
                                    # deep restores pay nothing for the
                                    # ground behind them.
                                    _ge_blob = (
                                        go_explore_archive
                                        .select_return_states(1)[0]
                                    )
                                    if _ge_blob is not None:
                                        _restart_bytes = _ge_blob
                                try:
                                    self.pool.load_worker_state(i, _restart_bytes)
                                    _sticky_restart(i)
                                    reward_fns[i].reset()
                                    prev_completion_total[i] = 0.0
                                    _stage0_reseed[i] = True
                                except Exception:
                                    # The restart never happened: drop the
                                    # rung provenance assigned above, or
                                    # the iter boundary scores a phantom
                                    # truncation for an attempt with zero
                                    # executed steps (the boundary path
                                    # assigns src only after a successful
                                    # load — same discipline).
                                    _bwd_env_src[i] = -1
                                    active_in_iter[i] = False
                                    try:
                                        self.pool.set_worker_done(i, True)
                                    except Exception:
                                        pass
                            else:
                                # Stage 0, no start bytes (cold-boot-only
                                # game): freeze until iter boundary.
                                active_in_iter[i] = False
                                try:
                                    self.pool.set_worker_done(i, True)
                                except Exception:
                                    pass

                        # Advance stacker with the post-step frame/RAM.
                        # Only when still active and NOT just re-seeded by
                        # an auto-reset (the seed already is the new obs;
                        # pushing the pre-reset death frame on top would
                        # re-introduce the desync). Once done+frozen, the
                        # stacker's contents represent the final pre-done
                        # observation — no point contaminating it.
                        if active_in_iter[i] and _stage0_reseed[i] and self._is_tile_mode:
                            # Deferred stage-0 restart seed: this step's
                            # RAM is the first post-restore frame — fresh
                            # stack, exactly like an episode start.
                            stacked_obs[i] = tile_stackers[i].reset(
                                self._tile_extractor.extract(ram)
                            )
                            _stage0_reseed[i] = False
                        elif active_in_iter[i] and not reseeded:
                            if self._is_tile_mode:
                                stacked_obs[i] = tile_stackers[i].push(
                                    self._tile_extractor.extract(ram)
                                )
                            else:
                                stacked_obs[i] = stackers[i].push(
                                    r.frame, getattr(r, "preprocessed", None)
                                )

                    # Recurrent: reset the GRU hidden state for envs that
                    # ended an episode this step (done-mask), so neither
                    # the next rollout step nor the update-replay carries
                    # hidden state across an episode boundary. Uses the
                    # SAME done_buf the replay uses, so the two match.
                    if self._recurrent:
                        done_mask_t = torch.from_numpy(
                            done_buf[t].astype(np.float32)
                        ).to(self.device).unsqueeze(-1)
                        h_rollout = h_rollout_next * (1.0 - done_mask_t)

                # Fused device-to-host drain of the deferred per-step
                # values / log-probs: one MPS sync for the whole rollout
                # instead of two per step. On an early Stop the loop
                # breaks after len(value_steps) steps — only those rows
                # are written and the untouched tail stays zero (the
                # update is skipped for truncated rollouts anyway).
                _drain_t0 = time.perf_counter_ns()
                n_rolled = len(value_steps)
                if n_rolled:
                    value_buf[:n_rolled] = (
                        torch.stack(value_steps, dim=0).cpu().numpy()
                    )
                    log_prob_buf[:n_rolled] = (
                        torch.stack(log_prob_steps, dim=0).cpu().numpy()
                    )
                    if prmdp_on and adv_value_steps:
                        adv_value_buf[:len(adv_value_steps)] = (
                            torch.stack(adv_value_steps, dim=0).cpu().numpy()
                        )
                    if _kadv is not None:
                        _kadv.drain_values()
                self._gen_timer.add(
                    "rollout_forward", time.perf_counter_ns() - _drain_t0
                )

                # Bootstrap V(s_T) from the final observation per env.
                final_batch_np = np.stack(stacked_obs, axis=0)
                final_batch_t = (
                    torch.from_numpy(final_batch_np)
                    .to(self.device).float()
                )
                if not self._is_tile_mode and not self.preprocess_f16:
                    final_batch_t = final_batch_t.div_(255.0)
                if self._recurrent:
                    _, final_values, _ = net.forward_ac_recurrent(
                        final_batch_t, h_rollout
                    )
                else:
                    _, final_values = net.forward_ac(final_batch_t)
                final_values_np = final_values.cpu().numpy()
                if _adv_net is not None:
                    _, adv_final_values = _adv_net.forward_ac(final_batch_t)
                    adv_final_values_np = adv_final_values.cpu().numpy()
                if _kadv is not None:
                    _kadv.compute_final_values(final_batch_t)

            # Stop pressed mid-rollout: the rollout loop broke early, so the
            # buffer is only partially filled (valid_buf is sparse). Running
            # the RND/GAE/PPO update on that truncated rollout would corrupt
            # the policy with a bad gradient step and pollute the optimizer
            # state — the net a user keeps after Stop would be worse than the
            # one before. Bail out of the iter loop before the update. A
            # completed rollout is unaffected (_running stays True); final
            # teardown after the loop still runs.
            if not self._running:
                break

            # ============== CORE PPO UPDATE ==============
            # Intrinsic/count fold -> GAE-lambda -> advantage norm over the
            # valid mask -> K-epoch minibatch PPO update (RND target cache,
            # demo anchor, SHAPO, non-finite-loss backstop). Lifted verbatim
            # into PPOUpdater (Task 2). The updater hands back the folded
            # reward_buf, the valid mask, the shared obs tensors + mb_size,
            # and the reported scalars so the PR-MDP adversary block below
            # (disabled in the golden profile) reuses the identical
            # intermediates -- byte-identical behavior on that path.
            _upd = _ppo_updater.update(
                net=net, optimizer=optimizer,
                obs_buf=obs_buf, action_buf=action_buf, reward_buf=reward_buf,
                value_buf=value_buf, log_prob_buf=log_prob_buf,
                done_buf=done_buf, valid_buf=valid_buf, bonus_buf=bonus_buf,
                final_values_np=final_values_np,
                rollout_steps=rollout_steps, num_envs=num_envs,
                obs_shape=obs_shape, global_it=global_it, sam_rho=_sam_rho,
                trunc_buf=(trunc_buf if wave_monotone else None),
                entropy_weight_buf=(_entw_buf if _commit_on else None),
            )
            reward_buf = _upd["reward_buf"]
            valid_flat = _upd["valid_flat"]
            valid_indices = _upd["valid_indices"]
            rnd_intrinsic_mean = _upd["rnd_intrinsic_mean"]
            count_bonus_mean = _upd["count_bonus_mean"]
            last_policy_loss = _upd["last_policy_loss"]
            last_value_loss = _upd["last_value_loss"]
            last_entropy = _upd["last_entropy"]
            last_loss = _upd["last_loss"]
            last_rnd_loss = _upd["last_rnd_loss"]
            last_clip_fraction = _upd["last_clip_fraction"]
            last_approx_kl = _upd["last_approx_kl"]
            last_grad_norm = _upd["last_grad_norm"]
            adv_mean = _upd["adv_mean"]
            adv_std = _upd["adv_std"]
            explained_variance = _upd["explained_variance"]
            _demo_coef = _upd["demo_coef"]
            _demo_loss_accum = _upd["demo_loss_accum"]
            _demo_loss_n = _upd["demo_loss_n"]
            obs_all = _upd["obs_all"]
            obs_flat = _upd["obs_flat"]
            mb_size = _upd["mb_size"]

            # ===== ReDo dormancy check + recycle (AMENDMENT 1, B3) =====
            # End-of-gradient-iteration hook: immediately after the PPO
            # update, before checkpointing (B3.3). With fewer than 2
            # valid rollout rows the minibatch loop above trained
            # nothing, so the check is skipped with the registered line.
            # The hook consumes no RNG when redo is off; a later anti-
            # collapse rollback (snapshot restore + full optimizer
            # rebuild) supersedes any recycle this iter, by design.
            if redo_on and valid_indices.size < 2:
                log.info(
                    "[redo] iter %d: skipped (no gradient step)", global_it
                )
            else:
                _rd = _redo_maybe_check(
                    enabled=redo_on, net=net, optimizer=optimizer,
                    obs_all=obs_all, valid_indices=valid_indices,
                    tau=self.redo_tau,
                    sample_batch=self.redo_sample_batch,
                    check_every_iters=self.redo_check_every_iters,
                    global_it=global_it,
                    reset_optimizer_moments=(
                        self.redo_reset_optimizer_moments
                    ),
                    mode=self.redo_mode,
                    bottom_k=self.redo_bottom_k,
                )
                if _rd is not None:
                    self._redo_cum_recycled += _rd.recycled
                    log.info(
                        "[redo] iter %d: dormant fc1 %d/%d fc2 %d/%d "
                        "recycled %d cum %d agree %.4f max_dlogit %.6f "
                        "tail fc1 %.4f/%.4f/%.4f fc2 %.4f/%.4f/%.4f",
                        global_it, _rd.dormant_fc1, _rd.hidden_dim,
                        _rd.dormant_fc2, _rd.trunk_dim, _rd.recycled,
                        self._redo_cum_recycled, _rd.agree, _rd.max_dlogit,
                        _rd.fc1_tail[0], _rd.fc1_tail[1], _rd.fc1_tail[2],
                        _rd.fc2_tail[0], _rd.fc2_tail[1], _rd.fc2_tail[2],
                    )
                    if _rd.recycled:
                        # Promoted DEBUG -> INFO (V31 §12.3): the F3
                        # distinctness gate (redo_arm_gate.py) parses this
                        # line, and DEBUG-level output is dropped by the
                        # launcher's default logging config, which would
                        # make F3 silently unverifiable on every real run.
                        # Pure logging — no RNG, no behavior change;
                        # `redo_enabled: false` byte-identity is untouched
                        # since this line is unreachable on that path.
                        log.info(
                            "[redo] recycled unit indices: fc1=%s fc2=%s",
                            _rd.fc1_indices, _rd.fc2_indices,
                        )
                        # V32 §3 B2 artifact-match input: the FULL fc2
                        # score vector this check selected from, so a
                        # gate can recompute bottom-k offline from the
                        # logged bytes and compare against the logged
                        # indices. Lane A's 2.13 h voided because eight
                        # gates all tested the pipeline and none tested
                        # the artifact; this line is what lets the v32
                        # gate test the artifact. Pure logging — no RNG,
                        # no behaviour change, unreachable when redo is
                        # off, so `redo_enabled: false` byte-identity is
                        # untouched.
                        log.info(
                            "[redo] fc2 scores: %s",
                            [round(_s, 6) for _s in _rd.fc2_scores],
                        )
                        self._redo_recycle_events += 1
                        if self._redo_first_recycle_iter is None:
                            self._redo_first_recycle_iter = global_it
                        self._redo_agree_log.append(_rd.agree)
                        _frac_fire = _redo_dose_fraction(
                            _rd.dormant_fc1, _rd.hidden_dim,
                            _rd.dormant_fc2, _rd.trunk_dim,
                        )
                        self._redo_fracs_on_fire.append(_frac_fire)
                        for _idx in _rd.fc2_indices:
                            self._redo_fc2_index_counts[_idx] = (
                                self._redo_fc2_index_counts.get(_idx, 0) + 1
                            )

                    # ===== ReDo in-run dose ceiling (V31 §3, abort A4) ===
                    # Appended for EVERY executed check, including
                    # zero-recycle ones (dose_ceiling_trips' contract): a
                    # median over firing events only is blind to how
                    # OFTEN the treatment fires, and this ceiling exists
                    # specifically to see that. The tail is measured to
                    # drift DOWN across training (v30 §1.3/§5), so a tau
                    # that is surgical at iter 30 can be a partial network
                    # reset by iter 200 — this catches that live instead
                    # of discovering it after burning to iter 250.
                    self._redo_dose_ceiling_history.append(
                        _redo_dose_fraction(
                            _rd.dormant_fc1, _rd.hidden_dim,
                            _rd.dormant_fc2, _rd.trunk_dim,
                        )
                    )
                    if _redo_dose_ceiling_trips(
                        self._redo_dose_ceiling_history,
                        ceiling=_REDO_DOSE_CEILING,
                    ):
                        raise RuntimeError(
                            "[redo] VOID-OVERDOSE: trailing-"
                            f"{len(self._redo_dose_ceiling_history[-10:])}"
                            "-check median dose of the worst-hit layer "
                            f"exceeded {_REDO_DOSE_CEILING:.2f} at iter "
                            f"{global_it} (tau={self.redo_tau:g}). This is "
                            "NOT a FAIL: it is a positive mechanism "
                            "finding — a fixed dormancy threshold that "
                            "was surgical early became a partial network "
                            "reset by this iteration. Receipts: "
                            "docs/proposals/V31_REDO_SURGICAL_2026-08-27"
                            ".md §3.2. This run is VOID-OVERDOSE; the "
                            "whole seed sequence aborts. Do not issue a "
                            "verdict."
                        )

            # ===== ReDo arming deadline (V30 registration, A2) =====
            # A run in which ReDo never fires is VOID, not FAIL. v27 and
            # v28 each burned 7h10m x 4 seeds at tau=0.025 — an order of
            # magnitude below the firing threshold of this architecture —
            # and then reported a FAIL that the treatment could not have
            # produced. The deadline turns that class of run into a hard
            # abort at ~25 iterations. The bound is a module constant, NOT
            # a config key: 20 declared keys in the flagship recipe never
            # executed, so a new declared key is a new way to be inert.
            # Placed AFTER the whole hook (not inside the `_rd is not
            # None` branch) so an off-cadence or skipped iteration cannot
            # buy the run past the deadline.
            if (
                redo_on
                and global_it >= _REDO_ARM_DEADLINE_ITERS
                and self._redo_cum_recycled == 0
            ):
                raise RuntimeError(
                    f"[redo] VOID: armed at tau={self.redo_tau:g} but "
                    f"cum_recycled==0 after {global_it + 1} iterations. "
                    "On this Linear->LayerNorm->SiLU trunk the fc2 "
                    "dormancy score bottoms out near 0.08-0.13 once "
                    "training is past ~iter 5 and never approaches 0.025; "
                    "read the per-layer min/p5/p10 on the `[redo] iter` "
                    "lines above to see where this run's tail actually "
                    "sits and retarget tau by measurement. Receipts: "
                    "runs/v30_premise_falsifier_2026-08-27/. This run is "
                    "VOID, not FAIL. Do not issue a verdict."
                )

            # ============== PR-MDP ADVERSARY UPDATE ==============
            # Plain PPO on the SAME rollout with negated rewards and its
            # own critic, restricted to steps where the adversary's
            # action executed (its behavior distribution). Zero-sum:
            # what kills the protagonist is the adversary's return.
            if _adv_net is not None and valid_indices.size:
                _prm_t0 = time.perf_counter_ns()
                adv_advantages, adv_targets = batched_gae(
                    -reward_buf, adv_value_buf, done_buf,
                    adv_final_values_np,
                    self.reinforce_gamma, self.gae_lambda,
                    trunc_buf=(trunc_buf if wave_monotone else None),
                )
                exec_flat = (exec_mask_buf & valid_buf).reshape(-1)
                adv_valid = np.where(exec_flat)[0]
                _prm_override_frac = float(exec_flat.sum()) / max(
                    1, int(valid_flat.sum()))
                _prm_last = {}
                if adv_valid.size >= 2:
                    _a_adv = adv_advantages.reshape(-1)[adv_valid]
                    _a_mean = float(_a_adv.mean())
                    _a_std = float(_a_adv.std()) + 1e-8
                    adv_adv_all = torch.from_numpy(
                        ((adv_advantages.reshape(-1) - _a_mean) / _a_std)
                        .astype(np.float32)).to(self.device)
                    adv_tgt_all = torch.from_numpy(
                        adv_targets.reshape(-1).astype(np.float32)
                    ).to(self.device)
                    adv_act_all = torch.from_numpy(
                        adv_action_buf.reshape(-1)).to(self.device)
                    adv_lp_old_all = torch.from_numpy(
                        adv_logp_buf.reshape(-1)).to(self.device)
                    for _ in range(_prmdp_adv_epochs):
                        _perm = np.random.permutation(adv_valid)
                        for _mb0 in range(0, _perm.shape[0], mb_size):
                            _mb = _perm[_mb0:_mb0 + mb_size]
                            if _mb.size < 2:
                                continue
                            _mb_t = torch.from_numpy(_mb).to(self.device)
                            if obs_all is not None:
                                _st = obs_all[_mb_t]
                            else:
                                _st = torch.from_numpy(np.ascontiguousarray(
                                    obs_flat[_mb])).to(self.device).float()
                                if not self.preprocess_f16:
                                    _st = _st.div_(255.0)
                            _lg, _vp = _adv_net.forward_ac(_st)
                            _l, _pl, _vl, _en = ppo_losses(
                                _lg, _vp, adv_act_all[_mb_t],
                                adv_lp_old_all[_mb_t], adv_adv_all[_mb_t],
                                adv_tgt_all[_mb_t],
                                clip_eps=_prmdp_adv_clip,
                                value_coef=self.value_coef,
                                entropy_coef=_prmdp_adv_ent,
                                value_loss_kind=self.value_loss_kind,
                            )
                            if not torch.isfinite(_l):
                                _adv_opt.zero_grad(set_to_none=True)
                                continue
                            _adv_opt.zero_grad()
                            _l.backward()
                            torch.nn.utils.clip_grad_norm_(
                                _adv_net.parameters(),
                                self.reinforce_grad_clip,
                            )
                            _adv_opt.step()
                            _prm_last = {"pl": _pl.detach(),
                                         "en": _en.detach()}
                if _prm_last:
                    log.info(
                        "[prmdp] iter %d: override_frac=%.3f "
                        "adv_policy=%.4f adv_entropy=%.4f (%d exec steps)",
                        global_it, _prm_override_frac,
                        float(_prm_last["pl"].item()),
                        float(_prm_last["en"].item()), adv_valid.size,
                    )
                self._gen_timer.add(
                    "prmdp_adversary", time.perf_counter_ns() - _prm_t0
                )
            elif prmdp_on and valid_indices.size:
                log.info(
                    "[prmdp] iter %d: uniform adversary, "
                    "override_frac=%.3f", global_it,
                    float(exec_mask_buf.mean()),
                )

            # ===== KERNEL ADVERSARY UPDATE (reinforce.adversary) =====
            # PPO on the SAME rollout with the adversary's own reward
            # stream (negated protagonist reward minus the repeat budget)
            # and its own critic, over EVERY valid step — pass is a
            # decision too, so its behavior distribution is the whole
            # rollout, not an alpha draw. Epoch count matches the
            # protagonist's (10:10). `_mech_metrics` also carries the KL
            # anchor + SIL surfaces; empty when all mechanisms are off so
            # the metrics rows are unchanged byte-for-byte.
            _mech_metrics: dict = {}
            if _kadv is not None and valid_indices.size:
                _kadv_t0 = time.perf_counter_ns()
                _kadv_stats = _kadv.update(
                    reward_buf=reward_buf, done_buf=done_buf,
                    valid_buf=valid_buf, obs_all=obs_all, obs_flat=obs_flat,
                    mb_size=mb_size,
                    trunc_buf=(trunc_buf if wave_monotone else None),
                )
                _mech_metrics.update(_kadv_stats)
                log.info(
                    "[kernel_adv] iter %d: repeat_frac=%.3f entropy=%.4f "
                    "(ln2=%.4f) policy=%.4f", global_it,
                    _kadv_stats["adversary_repeat_frac"],
                    _kadv_stats["adversary_entropy"], math.log(2.0),
                    _kadv_stats["adversary_policy_loss"],
                )
                self._gen_timer.add(
                    "kernel_adversary", time.perf_counter_ns() - _kadv_t0
                )
            if self._kl_anchor is not None:
                _kl_mean = _kl_div_sum / max(1, _kl_div_n)
                _mech_metrics["kl_anchor_div"] = float(_kl_mean)
                _mech_metrics["kl_anchor_beta"] = float(self._kl_anchor.beta)
                _mech_metrics["kl_anchor_actor_frozen"] = int(
                    self._kl_anchor.frozen
                )
                if _upd["kl_loss_n"]:
                    # Loss-level tether magnitude: coef * mean minibatch
                    # KL(prior || pi) — the exact term the update loss
                    # carried this iter, alongside the rollout-time div.
                    _mech_metrics["kl_anchor_loss"] = float(
                        _upd["kl_loss_coef"]
                        * _upd["kl_loss_accum"] / _upd["kl_loss_n"]
                    )
                    _mech_metrics["kl_anchor_loss_coef"] = float(
                        _upd["kl_loss_coef"]
                    )
                # Logged every iteration — the campaign kill criterion
                # (KL > 0.15 sustained 2M steps) reads this series.
                log.info(
                    "[kl_anchor] iter %d: D_KL(prior||pi)=%.4f beta=%.4f "
                    "frozen=%d", global_it, _kl_mean,
                    float(self._kl_anchor.beta), int(self._kl_anchor.frozen),
                )
            if self._sil_buffer is not None:
                _mech_metrics["sil_buffer_trajs"] = len(self._sil_buffer)
                _mech_metrics["sil_buffer_steps"] = self._sil_buffer.n_steps
                _mech_metrics["sil_clears_total"] = (
                    self._sil_buffer.total_clears
                )
                if _upd["sil_loss_n"]:
                    _mech_metrics["sil_loss"] = (
                        _upd["sil_loss_accum"] / _upd["sil_loss_n"]
                    )
            if wave_monotone:
                # Lost-cut truncation rate — how much rollout the
                # off-envelope cut is reclaiming under the monotone rule.
                _mech_metrics["wave_truncations"] = int(trunc_buf.sum())

            # ============== LOGGING + METRICS ==============
            # Attribute the per-iter bookkeeping (episode stats, reward
            # breakdown, curriculum/advance detection, metric assembly) —
            # previously part of the unbucketed ~17% iter-wall gap. Closed
            # just before the snapshot below so it lands in THIS gen's row.
            # Diagnostic only.
            _bookkeeping_t0 = time.perf_counter_ns()
            mean_ep_return = (
                float(np.mean(completed_returns)) if completed_returns else 0.0
            )
            mean_ep_length = (
                float(np.mean(completed_lengths)) if completed_lengths else 0.0
            )
            n_complete = len(completed_returns)
            # Also surface in-progress (still-alive) episodes' partial
            # returns + lengths — once the policy gets good enough to
            # survive past rollout_steps, n_complete drops to 0 and the
            # "completed-only" mean_return reads as 0.0 which looks like
            # a regression. The in-progress numbers reflect actual
            # ongoing reward accumulation.
            n_in_progress = int(active_in_iter.sum())
            mean_inprogress_return = (
                float(ep_returns[active_in_iter].mean()) if n_in_progress else 0.0
            )
            mean_inprogress_length = (
                float(ep_lengths[active_in_iter].mean()) if n_in_progress else 0.0
            )
            # World/level distribution — answer "did agents reach 1-2?".
            # Count envs whose MAX W-L this iter == each unique value.
            # SMB's $0760 byte is an INTERNAL AREA INDEX, not the
            # displayed level number. Verified empirically (screenshots
            # at /tmp/level_check on 2026-05-14): byte=0 -> "1-1",
            # byte=1 is a transient transition state (status bar already
            # shows next level but visuals are still the previous
            # level's cutscene), byte=2 -> "1-2" underground gameplay.
            # Higher byte values correspond to 1-2 sub-areas, then 1-3,
            # then 1-4. Map honestly so the user can read the column at
            # a glance instead of trusting a naive `byte+1` offset that
            # over-reports progress by one level.
            def _byte_to_label(byte_val: int) -> str:
                w = (byte_val >> 4) + 1  # world byte is at upper nibble
                area = byte_val & 0x0F
                # Within world 1 (and structurally similar elsewhere):
                #   area 0 = 1-1
                #   area 1 = transition (status bar already says "1-2"
                #            but Mario is mid-cutscene; few frames)
                #   area 2 = 1-2 underground main
                #   area 5 = 1-3 overworld (post-1-2-clear)
                #   area 6 = 1-4 castle
                # If Mario never clears 1-2, areas 3-4 are unseen.
                # Map to displayed level via the canonical SMB area
                # table — fall back to "byte=N" for unknown values
                # so we don't silently mislabel.
                AREA_TO_LEVEL = {0: "1", 1: "1->2", 2: "2", 5: "3", 6: "4"}
                lvl = AREA_TO_LEVEL.get(area, f"area{area}")
                return f"{w}-{lvl}"

            def _fmt_wl_dist(packed: np.ndarray) -> str:
                unique, counts = np.unique(packed, return_counts=True)
                return " ".join(
                    f"{_byte_to_label(int(v))}={int(c)}"
                    for v, c in zip(unique, counts)
                )
            max_wl_str = _fmt_wl_dist(max_world_level_packed)
            end_wl_str = _fmt_wl_dist(end_world_level_packed)

            # Stage-advance detection. Count envs whose max-area-byte
            # this iter exceeds the current stage's anchor byte. If the
            # rolling-mean fraction reaches SMB_ADVANCE_PCT over
            # SMB_ADVANCE_WINDOW iters, capture the next stage's save
            # state from one of those envs and bump the curriculum.
            current_anchor = (
                smb_curriculum_anchors[smb_curriculum_stage]
                if smb_curriculum_stage < len(smb_curriculum_anchors)
                else 0
            )
            # Gated advance: only promote the curriculum when the envs
            # ASSIGNED to the current stage RELIABLY clear past it —
            # rolling-mean clear fraction >= SMB_ADVANCE_PCT over
            # SMB_ADVANCE_WINDOW iters. The previous
            # "first env past anchor advances the whole pool" gate
            # advanced onto stages the policy couldn't play, producing
            # uniform failure → entropy melt → permanent collapse
            # (root-caused from the iter_02700 run). Measuring the
            # fraction among current-stage envs (not the whole mixed
            # pool) is the correct denominator now that earlier-stage
            # envs are intentionally present.
            # Ladder mode measures `region_of(ram)` sub-stage orders; the
            # scalar path measures the packed world<<4|area byte. Both feed
            # the identical rolling-mean gate below.
            past_metric = max_region_reached if ladder_on else max_world_level_packed
            at_current = (env_stage == smb_curriculum_stage)
            n_at_current = int(at_current.sum())
            past_mask = (past_metric > current_anchor)
            n_past_stage = int(past_mask.sum())
            n_past_current = int((past_mask & at_current).sum())
            frac_past = n_past_current / max(1, n_at_current)
            smb_pastfrac_history.append(frac_past)
            if len(smb_pastfrac_history) > SMB_ADVANCE_WINDOW:
                smb_pastfrac_history.pop(0)
            rolling_mean = sum(smb_pastfrac_history) / len(smb_pastfrac_history)
            should_advance = (
                smb_pending_capture is not None
                and len(smb_pastfrac_history) >= SMB_ADVANCE_WINDOW
                and rolling_mean >= SMB_ADVANCE_PCT
                # Freeze the frontier while consolidating (§Q4) and never
                # advance past the ladder's final rung. Level-scoped
                # consolidation keeps the ladder read-only: never advance.
                and not consolidating
                and not consolidate_on
                and (not ladder_on or smb_curriculum_stage < LADDER_SIZE - 1)
            )
            if self._smb_curriculum_active:
                log.info(
                    "[vanilla_ppo] curriculum diag: stage=%d, current-stage "
                    "envs past anchor=%d/%d (%.0f%%), rolling-mean(%d)=%.0f%% "
                    "(need %.0f%%), pending_capture=%s, should_advance=%s",
                    smb_curriculum_stage, n_past_current, n_at_current,
                    frac_past * 100, SMB_ADVANCE_WINDOW, rolling_mean * 100,
                    SMB_ADVANCE_PCT * 100,
                    "set" if smb_pending_capture is not None else "None",
                    should_advance,
                )
            if should_advance and ladder_on:
                # Ladder advance: the frontier climbs exactly ONE rung. The
                # capture already live-filled its own (deeper) rung's seed, so
                # here we only move the pointer, persist the new frontier
                # rung's seed for resume (anchor == rung order), and re-arm.
                new_frontier = smb_curriculum_stage + 1
                _seed = smb_curriculum_states[new_frontier]
                if _seed is not None:
                    stage_path = smb_curriculum_dir / f"stage_{new_frontier:02d}.state"
                    _tmp_stage = stage_path.with_suffix(".state.tmp")
                    _tmp_stage.write_bytes(_seed)
                    _tmp_stage.replace(stage_path)
                    meta_path = smb_curriculum_dir / f"stage_{new_frontier:02d}.meta.json"
                    _tmp_meta = meta_path.with_suffix(".json.tmp")
                    _tmp_meta.write_text(json.dumps({"anchor": int(new_frontier)}))
                    _tmp_meta.replace(meta_path)
                _rung = ladder[new_frontier]
                smb_curriculum_stage = new_frontier
                smb_stage_clear_history = []
                smb_pastfrac_history = []   # new rung must re-earn advance
                smb_pending_capture = None
                log.info(
                    "[vanilla_ppo] *** LADDER ADVANCE *** frontier %d -> %d "
                    "(%s, area=%d, x[%d,%d)); seed=%s.",
                    new_frontier - 1, new_frontier, _rung.level, _rung.area,
                    _rung.x_lo, _rung.x_hi,
                    "live/disk" if _seed is not None else "cold",
                )
                # An advance resets the stall clock; a burst in flight has
                # done its job the moment the frontier moves, so retract it
                # early (no harvest — the capture already seeded the new rung).
                if ge_burst_on:
                    ge_iters_since_advance = 0
                    stall_notifier.reset()  # genuine advance — new stall episode
                    if ge_burst_active:
                        ge_burst_active = False
                        ge_archive = None
                        ge_bursts_done += 1
                        log.warning(
                            "[vanilla_ppo] *** GO-EXPLORE BURST SUCCESS *** "
                            "frontier advanced during burst #%d; retracting "
                            "early.", ge_bursts_done,
                        )
            elif should_advance and smb_pending_capture is not None:
                # Use the mid-rollout capture: a LIVING env's state
                # taken the moment it first crossed into the next
                # area-byte. The previous design used end-of-iter
                # candidates which were already dead/frozen.
                new_anchor, blob = smb_pending_capture
                new_stage = smb_curriculum_stage + 1
                stage_path = smb_curriculum_dir / (
                    f"stage_{new_stage:02d}.state"
                )
                # Atomic write (tmp + replace): a crash mid-write would
                # otherwise leave a truncated .state that the resume
                # loader reads and chokes on, breaking curriculum resume.
                _tmp_stage = stage_path.with_suffix(".state.tmp")
                _tmp_stage.write_bytes(blob)
                _tmp_stage.replace(stage_path)
                # Persist anchor in sidecar so the capture gate works
                # correctly after a restart.
                meta_path = smb_curriculum_dir / (
                    f"stage_{new_stage:02d}.meta.json"
                )
                _tmp_meta = meta_path.with_suffix(".json.tmp")
                _tmp_meta.write_text(json.dumps({"anchor": int(new_anchor)}))
                _tmp_meta.replace(meta_path)
                while len(smb_curriculum_anchors) <= new_stage:
                    smb_curriculum_anchors.append(0)
                    smb_curriculum_states.append(None)
                smb_curriculum_anchors[new_stage] = new_anchor
                smb_curriculum_states[new_stage] = blob
                smb_curriculum_stage = new_stage
                smb_stage_clear_history = []
                smb_pastfrac_history = []  # new stage must re-earn advance
                smb_pending_capture = None  # reset for the next stage
                log.info(
                    "[vanilla_ppo] *** CURRICULUM ADVANCE *** "
                    "stage %d -> %d: anchor area-byte=%d (mid-rollout "
                    "capture from alive env). Next iter, all %d envs "
                    "warm-start from %s.",
                    new_stage - 1, new_stage, new_anchor, num_envs,
                    stage_path,
                )
            log.info(
                "[vanilla_ppo] iter %d: completed_eps=%d  mean_return=%.1f  "
                "mean_len=%.1f  in_progress=%d  ip_return=%.1f  ip_len=%.1f  "
                "clears=%d  loss=%.4f  policy=%.4f  value=%.4f  entropy=%.4f",
                it, n_complete, mean_ep_return, mean_ep_length,
                n_in_progress, mean_inprogress_return, mean_inprogress_length,
                n_clears_this_iter,
                last_loss, last_policy_loss, last_value_loss, last_entropy,
            )
            if self._smb_curriculum_active:
                log.info(
                    "[vanilla_ppo] iter %d: max-W-L: %s  |  end-W-L: %s  "
                    "|  curriculum stage=%d (anchor area=%d, %d/%d envs "
                    "past stage this iter)",
                    it, max_wl_str, end_wl_str,
                    smb_curriculum_stage, current_anchor,
                    n_past_stage, num_envs,
                )
            # best_fitness uses the larger of completed-max and in-progress-max
            # so the dashboard reflects real progress when episodes are
            # outlasting rollout_steps.
            best_completed = (
                float(max(completed_returns)) if completed_returns else 0.0
            )
            best_in_progress = (
                float(ep_returns[active_in_iter].max())
                if n_in_progress else 0.0
            )
            # Aggregate per-env reward breakdown into `reward_<key>` metrics
            # the dashboard's reward-signal-stack panel reads. Sum each
            # breakdown component across all 30 envs' reward fns for this
            # iter, normalize per-env so the magnitudes are comparable to
            # the ga_ppo path's breakdown emission.
            reward_breakdown_emit: dict[str, float] = {}
            try:
                for rfn in reward_fns:
                    for k, v in dict(rfn.breakdown).items():
                        reward_breakdown_emit[f"reward_{k}"] = (
                            reward_breakdown_emit.get(f"reward_{k}", 0.0) + float(v)
                        )
                for k in list(reward_breakdown_emit):
                    reward_breakdown_emit[k] /= max(1, num_envs)
            except Exception:
                pass
            # PPO metric keys MUST be `ppo_*` (not `vanilla_ppo_*`) to
            # match the dashboard's panel-config lookup. The ga_ppo path
            # emits with these exact names; the chart uses them
            # uniformly across both modes.
            # vanilla_ppo never feeds the GA-oriented CurriculumManager,
            # so its stage_success_rate() is permanently 0.0 — which
            # reads on the dashboard as "never clears" even when the
            # agent is clearing the level repeatedly. Derive the
            # displayed success_rate from the real per-iter clear
            # counter instead: fraction of completed episodes that
            # cleared (or per-env when episodes outlast the rollout).
            if n_complete > 0:
                vppo_success_rate = n_clears_this_iter / n_complete
            else:
                vppo_success_rate = n_clears_this_iter / max(1, num_envs)
            vppo_success_rate = min(1.0, float(vppo_success_rate))
            # Throughput: env-steps/s, NES frames/s (× frame_skip), and
            # realtime multiple (vs the NES's 60 fps). Makes the M4's
            # utilization visible and turns "weekend per game" from a
            # guess into an estimate (samples-to-target / samples_per_sec).
            _iter_dt = max(1e-6, time.perf_counter() - _iter_t0)
            samples_per_sec = (rollout_steps * num_envs) / _iter_dt
            nes_fps = samples_per_sec * self.frame_skip
            realtime_x = nes_fps / 60.0
            log.info(
                "[vanilla_ppo] iter %d throughput: %.0f env-steps/s | "
                "%.0f NES-fps | %.0fx realtime | %.2f s/iter",
                global_it, samples_per_sec, nes_fps, realtime_x, _iter_dt,
            )
            # ===== BACKWARD CURRICULUM telemetry =====
            # Every iter, not sampled: tau is the run's whole story, and
            # the entrance columns are the only in-training number that
            # speaks the same language as the honest gate.
            if bwd_on and bwd_sched is not None:
                _bs = bwd_sched.snapshot()
                _bt = bwd_entries[_bs["tau"]]
                # The guard suffix is empty unless entropy_guard is
                # configured, so an unguarded run's log line is
                # character-identical to before the guard existed. Armed
                # fraction is here because the registered B5 kill
                # criterion ("armed >50% of 50 consecutive iters") is
                # read off it directly.
                #
                # The guard observes at the END of an iter (below, after
                # the entropy-floor controller), so what this line prints
                # is the guard state and coefficient that GOVERNED THIS
                # ITER'S UPDATE — decided one iter ago. That is the
                # honest pairing: the `entropy=` figure on the
                # `[vanilla_ppo] iter` line above was produced under
                # exactly this coefficient. The arm/disarm `[backward-
                # guard]` lines are emitted at the iter they happen.
                # Truncation accounting rides the same suffix rule: silent
                # unless one of the two run-2 repairs is armed, so an
                # untouched lineage's log line is character-identical.
                # `scored` is the half of `truncated` that reached the
                # window as a failure — with both off it is always 0 and
                # `truncated N` alone means "N attempts censored".
                _btr_sfx = ""
                if bwd_trunc_fail:
                    _btr_sfx += " (%d scored)" % bwd_trunc_scored
                if bwd_budget is not None:
                    _btr_sfx += " | budget %d steps" % (
                        bwd_budget.steps_for(_bs["tau"]),
                    )
                _bg_sfx = ""
                if bwd_guard is not None:
                    _bg_sfx = (
                        " | guard %s ent~%.4f armed %d/%d coef %.5f" % (
                            "ARMED" if bwd_guard.armed else "off",
                            bwd_guard.trailing_mean, bwd_guard.armed_recent,
                            bwd_guard.history_len, self.entropy_coef,
                        )
                    )
                log.info(
                    "[backward] iter %d: tau=%d/%d (step %d frame %d gx %d) "
                    "trailing %d/%d=%.2f (advance at >=%.2f over %d) "
                    "advances=%d%s | entrance %d/%d=%.3f | truncated %d%s%s",
                    global_it, _bs["tau"], _bs["n_entries"] - 1,
                    _bt.step, _bt.frame, _bt.gx,
                    _bs["successes"], _bs["attempts"], _bs["rate"],
                    bwd_sched.advance_threshold, bwd_sched.min_attempts,
                    _bs["advances"],
                    "  AT-ENTRANCE" if _bs["at_entrance"] else "",
                    _bs["entrance_successes"], _bs["entrance_attempts"],
                    _bs["entrance_rate"], bwd_trunc_dropped, _btr_sfx,
                    _bg_sfx,
                )

            # ===== CGSA telemetry + research signposts =====
            if cgsa_on and (it % 25 == 0 or it in (350, 800, 1150)):
                _cg = cgsa_zone_summary(cg_stats)
                _gaunt_p = _cg["gauntlet_avg_p"]
                _gaunt_p_all = _cg["gauntlet_avg_p_all"]
                _gx_arch = _cg["gx_max"]
                # Both censorings on one line. The `frontier_*` figures
                # exclude welded zones by construction, so a curriculum
                # that is succeeding reads identically to one that never
                # annealed; the `uncensored` group is the honest
                # population average and `welded_frac` is how much of it
                # the frontier figures drop.
                log.info(
                    "[cgsa] iter %d: cells=%d welded=%d frontier_avg_p=%.3f "
                    "gauntlet(n=%d)_avg_p=%.3f archive_gx_max>=%d | "
                    "measured=%d rate_p50=%.2f rate_p90=%.2f zones_p>0=%d | "
                    "uncensored avg_p_all=%.3f "
                    "gauntlet_all(n=%d)_avg_p=%.3f welded_frac=%.3f",
                    global_it, _cg["cells"], _cg["welded"],
                    _cg["frontier_avg_p"],
                    _cg["gauntlet_n"], _gaunt_p, _gx_arch,
                    _cg["measured"], _cg["rate_p50"], _cg["rate_p90"],
                    _cg["zones_p_gt0"],
                    _cg["avg_p_all"], _cg["gauntlet_n_all"], _gaunt_p_all,
                    _cg["welded_frac"],
                )
                # Persist curriculum state for post-run analysis (the v1
                # run's stats died with the process).
                try:
                    import json as _json
                    _cg_path = self.checkpoint_dir / "cgsa_stats.json"
                    with open(_cg_path, "w") as _f:
                        _json.dump(
                            {str(k): v for k, v in cg_stats.items()}, _f
                        )
                except Exception:
                    pass
                if it == 350:
                    _ok = _gx_arch >= 2000
                    log.info(
                        "[cgsa] SIGNPOST 1 (it350) frontier-expansion: "
                        "gx_max=%d %s (abandon if <1800)", _gx_arch,
                        "PASS" if _ok else ("FAIL" if _gx_arch < 1800
                                            else "MARGINAL"),
                    )
                if it == 800:
                    # Verdict stays on the frontier-only figure the
                    # signpost was pre-registered against — changing the
                    # criterion mid-record would be moving the goalposts.
                    # The uncensored figure rides alongside so a future
                    # low frontier reading can be told apart from a real
                    # stall: if welded_frac is high the frontier average
                    # is just reporting the zones that have not retired
                    # yet. (On the 2026-07 seeds it was NOT that — seed 1
                    # had zero welded gauntlet zones at it800, so its
                    # 0.054 FAIL was already uncensored.)
                    log.info(
                        "[cgsa] SIGNPOST 2 (it800) gauntlet noise: "
                        "avg_p=%.3f %s (need >=0.15; terminate if <0.10) "
                        "| uncensored gauntlet_avg_p_all=%.3f "
                        "welded_frac=%.3f "
                        "(report-only, verdict is frontier-only)",
                        _gaunt_p,
                        "PASS" if _gaunt_p >= 0.15 else (
                            "FAIL" if _gaunt_p < 0.10 else "MARGINAL"),
                        _gaunt_p_all, _cg["welded_frac"],
                    )
                if it == 1150:
                    log.info(
                        "[cgsa] SIGNPOST 3 (it1150): run the 30-episode "
                        "gauntlet-traversal probe from gx~1500 at sticky "
                        "0.25 externally (need >=10%% to gx>=2200).",
                    )
            # Per-section timing (rollout_forward / emulation / gae /
            # update / rnd_intrinsic / bookkeeping / iter_reset) so the
            # active pixel path stops flying blind: a regression shows up
            # as `jq '.timing_*' metrics.jsonl`.
            self._gen_timer.add(
                "bookkeeping", time.perf_counter_ns() - _bookkeeping_t0
            )
            timing_metrics = self._gen_timer.snapshot()
            self._gen_timer.reset()
            # Progress telemetry so every run is interpretable: Contra's
            # max screen reached, and SMB's max area-byte + real
            # curriculum stage (the `stage` field below is the GA
            # curriculum, which reads "default" on the vanilla path).
            progress_metrics: dict[str, float] = {}
            if iter_max_screen > 0:
                progress_metrics["vanilla_ppo_max_screen"] = int(iter_max_screen)
            # SMB progress proxies are gated by GAME (the $075F/$0760/x
            # RAM bytes are only meaningful for Mario), NOT by the
            # curriculum flag — a curriculum-off SMB run (e.g. a fixed
            # area-warm-start A/B) still wants these.
            if "mario" in str(self.game_profile.get("name", "")).lower():
                progress_metrics["vanilla_ppo_max_world_level"] = int(
                    max_world_level_packed.max()
                )
                progress_metrics["vanilla_ppo_max_x"] = int(max_x_reached.max())
            if go_explore_archive is not None:
                _ges = go_explore_archive.stats()
                progress_metrics["go_explore_cells"] = _ges["cells"]
                progress_metrics["go_explore_frontier"] = _ges["frontier"]
                progress_metrics["go_explore_best_score"] = _ges["best_score"]
                progress_metrics["go_explore_records"] = _ges["records"]
            if self._smb_curriculum_active:
                progress_metrics["vanilla_ppo_curriculum_stage"] = int(
                    smb_curriculum_stage
                )

            # === COLD-EVAL PROBE (primary metric + winner selector, §Q2) ===
            # Subprocess eval_game.py --sequential (greedy from the 1-1 start,
            # sequential predicate reconstructed from RAM) every `cold_every`
            # iters. The result is the DoD number: it drives best_cold.pt, the
            # forgetting alarm, and the consolidation trigger — training
            # telemetry never selects the deliverable.
            cold_metrics_emit: dict = {}
            if plr_on and it % cold_every == 0:
                from src.training import cold_probe as _cold_probe
                # Cold-eval EACH training level from its entrance AND the
                # held-out level under the honest protocol (sticky-0.25 +
                # jitter-16) so the winner metric matches the Phase-A gate.
                # Winner key = the WEAKEST training level (a generalist must
                # hold ALL of them), tie-broken by the mean. Holdout is
                # report-only — the transfer measurement.
                _per_level: dict = {}
                for _lvl in plr_ctx.train_labels:
                    _ss = plr_ctx.level_state_path.get(_lvl)  # None == cold boot
                    _c = _cold_probe.probe(
                        net, self.game_profile, episodes=cold_curve_eps,
                        device=self.device, sequential=True, level_clear=True,
                        start_state=_ss, sticky_prob=0.25, start_jitter=16,
                        rom_path=self.rom_path,
                        game=str(self.game_profile.get("name", "mario")),
                    )
                    _lvl_rate = _c.get("cold_seq_clear_rate")
                    _per_level[_lvl] = (
                        float(_lvl_rate) if _lvl_rate is not None else -1.0
                    )
                _hold: dict = {}
                for _lvl, _ss in plr_ctx.holdout.items():
                    _c = _cold_probe.probe(
                        net, self.game_profile, episodes=cold_curve_eps,
                        device=self.device, sequential=True, level_clear=True,
                        start_state=_ss, sticky_prob=0.25, start_jitter=16,
                        rom_path=self.rom_path,
                        game=str(self.game_profile.get("name", "mario")),
                    )
                    _hold_rate = _c.get("cold_seq_clear_rate")
                    _hold[_lvl] = (
                        float(_hold_rate) if _hold_rate is not None else -1.0
                    )
                _weakest = min(_per_level.values()) if _per_level else -1.0
                _mean = (
                    sum(_per_level.values()) / len(_per_level)
                    if _per_level else -1.0
                )
                cold_metrics_emit = {
                    "cold_seq_clear_rate": _weakest,
                    "cold_plr_per_level": _per_level,
                    "cold_plr_mean": _mean,
                    "cold_plr_holdout": _hold,
                }
                last_cold_metrics = cold_metrics_emit
                log.info(
                    "[vanilla_ppo] PLR COLD PROBE iter %d: per_level=%s "
                    "weakest=%.2f mean=%.2f holdout=%s (sticky0.25+jitter16)",
                    global_it,
                    {k: round(v, 2) for k, v in _per_level.items()},
                    _weakest, _mean,
                    {k: round(v, 2) for k, v in _hold.items()},
                )
                _pkey = (round(_weakest, 4), round(_mean, 4), 0.0)
                if _pkey > best_cold_key:
                    best_cold_key = _pkey
                    best_cold_rate = _weakest
                    best_cold_snapshot = {
                        k: v.detach().cpu().clone()
                        for k, v in net.state_dict().items()
                    }
                    try:
                        torch.save(
                            {"net_state_dict": best_cold_snapshot,
                             "iter": global_it,
                             "metric_name": "cold_plr_weakest_level",
                             "metric_value": best_cold_rate,
                             "per_level": _per_level, "holdout": _hold},
                            str(self.checkpoint_dir / cold_winner_name),
                        )
                        log.info(
                            "[vanilla_ppo] *** PLR WINNER *** best_cold.pt "
                            "weakest=%.2f mean=%.2f", _weakest, _mean,
                        )
                    except Exception as _e:
                        log.warning(
                            "[vanilla_ppo] PLR best_cold save failed: %s", _e
                        )
            if ladder_on:
                progress_metrics["vanilla_ppo_frontier"] = int(smb_curriculum_stage)
                progress_metrics["vanilla_ppo_retention_frac"] = float(retention_frac)
                progress_metrics["vanilla_ppo_consolidating"] = int(consolidating)
            if ladder_on and it % cold_every == 0:
                from src.training import cold_probe as _cold_probe
                _cold = _cold_probe.probe(
                    net, self.game_profile,
                    episodes=cold_curve_eps, device=self.device,
                    sequential=True, rom_path=self.rom_path,
                    game=str(self.game_profile.get("name", "mario")),
                )
                last_cold_metrics = _cold
                cold_metrics_emit = dict(_cold)
                _rate = _cold.get("cold_seq_clear_rate")
                _rate = float(_rate) if _rate is not None else -1.0
                _furthest = _cold.get("cold_furthest_seq")
                log.info(
                    "[vanilla_ppo] COLD PROBE iter %d: seq_clear=%s furthest=%s "
                    "warp=%s clear_1_1=%.2f status=%s",
                    global_it, _cold.get("cold_seq_clear_rate"), _furthest,
                    _cold.get("cold_warp_rate"),
                    _cold.get("cold_clear_1_1", 0.0), _cold.get("cold_status"),
                )
                # best_cold.pt: keep the greedy-sequential winner. Snapshot +
                # persist on a strict improvement; this is the rollback target.
                # The winner key is LEXICOGRAPHIC (seq_clear_rate, furthest
                # rank, 1-1 clear rate) — comparing the rate alone let a
                # regressed policy clobber the champion whenever every rate
                # was still 0.0 (all-night failure mode, 2026-07-16: the
                # first probe of each resumed run beat the fresh -1.0
                # baseline and best_cold.pt stopped being the best).
                _winner_key = (
                    float(_rate),
                    float(oc.furthest_rank(_furthest) or 0),
                    float(_cold.get("cold_clear_1_1") or 0.0),
                )
                if _winner_key > best_cold_key:
                    best_cold_key = _winner_key
                    best_cold_rate = _rate
                    best_cold_snapshot = {
                        k: v.detach().cpu().clone()
                        for k, v in net.state_dict().items()
                    }
                    try:
                        torch.save(
                            {"net_state_dict": best_cold_snapshot,
                             "iter": global_it,
                             "metric_name": "cold_seq_clear_rate",
                             "metric_value": best_cold_rate},
                            str(self.checkpoint_dir / cold_winner_name),
                        )
                    except Exception as _e:
                        log.warning("[vanilla_ppo] best_cold save failed: %s", _e)
                # Forgetting alarm on an already-cleared early level (§Q2): 2
                # consecutive high-water regressions -> bump Retention to 40%
                # for 100 iters; if it persists -> roll back to best_cold.pt.
                _alarm = False
                if _cold.get("cold_sequential"):
                    cold_highwater, forget_strikes, _alarm = oc.update_forgetting(
                        cold_highwater, forget_strikes, _furthest, forget_probes,
                    )
                # Persist alarm/winner state so resumes cannot re-baseline
                # regression as normal (see the init-side sidecar comment).
                try:
                    _alarm_sidecar.write_text(json.dumps({
                        "highwater": int(cold_highwater),
                        "best_cold_key": list(best_cold_key),
                    }))
                except Exception:
                    pass
                _action = oc.forgetting_action(
                    _alarm, global_it, retention_bump_until
                )
                if _action == "bump":
                    retention_frac = retention_bump
                    retention_bump_until = global_it + forget_bump_iters
                    log.warning(
                        "[vanilla_ppo] *** FORGETTING *** cold furthest "
                        "regressed %d probes; Retention -> %.0f%% for %d iters.",
                        forget_probes, retention_bump * 100, forget_bump_iters,
                    )
                    forget_strikes = 0
                elif _action == "rollback" and best_cold_snapshot is not None:
                    net.load_state_dict(best_cold_snapshot)
                    self._ppo_optimizer = self._build_ppo_optimizer(net)
                    optimizer = self._ppo_optimizer
                    log.warning(
                        "[vanilla_ppo] *** FORGETTING ROLLBACK *** persisted "
                        "regression; restored best_cold.pt (rate=%.3f) + reset "
                        "optimizer.", best_cold_rate,
                    )
                    forget_strikes = 0
                # Consolidation trigger / abort / re-arm (§Q4).
                # De-circularized trigger (2026-07-16): the original gate
                # required the COLD probe to reach 1-4 before consolidating —
                # but cold 1-4 needs the welded chain consolidation itself
                # produces, so the weld waited on its own output all night.
                # Arm when the ladder is COMPLETE (frontier at the top rung)
                # — the June recipe consolidated from stochastic competence —
                # or on cold-1-4 if it ever arrives first.
                if (not consolidating
                        and (oc.reached_1_4(_furthest)
                             or smb_curriculum_stage >= LADDER_SIZE - 1)
                        and smb_curriculum_stage >= 15):
                    consolidating = True
                    cons_step = 0
                    cons_best_rate = _rate
                    cons_no_gain = 0
                    log.info(
                        "[vanilla_ppo] *** CONSOLIDATE ARMED *** cold reached "
                        "1-4 at frontier %d; entropy %.3f->%.3f, rnd %.3f->%.3f "
                        "over %d iters (reversible).",
                        smb_curriculum_stage, cons_ent_from, cons_ent_to,
                        cons_rnd_from, cons_rnd_to, cons_ent_iters,
                    )
                elif consolidating:
                    if _rate > cons_best_rate + 1e-6:
                        cons_best_rate = _rate
                        cons_no_gain = 0
                    if _alarm or cons_no_gain >= cons_no_gain_iters:
                        # Abort + rearm (reversibility): restore exploration.
                        self.entropy_coef = cons_ent_from
                        self.rnd_intrinsic_coef = cons_rnd_from
                        consolidating = False
                        cons_aborts += 1
                        if cons_aborts >= 2 and cons_fallback == "cyclic":
                            cyclic_mode = True
                        log.warning(
                            "[vanilla_ppo] *** CONSOLIDATE ABORT #%d *** (%s); "
                            "restored entropy=%.3f rnd=%.3f%s.",
                            cons_aborts, "forgetting" if _alarm else "no gain",
                            cons_ent_from, cons_rnd_from,
                            " -> cyclic fallback" if cyclic_mode else "",
                        )
            # Reset a stale Retention bump once its window elapses.
            if retention_bump_until and global_it >= retention_bump_until:
                retention_frac = base_retention_frac
                retention_bump_until = 0

            # === LEVEL-SCOPED CONSOLIDATION GATE (the whole point) ===
            # Every `clevel_every` iters, cold-eval the TARGET level and EVERY
            # protect level from their entry states (greedy, per-level
            # "cleared the level it started in"). HARD RULE: any protect level
            # below its mode-start baseline -> immediate rollback to the last
            # accepted snapshot + entropy freeze for a cooldown. Protect held +
            # target improved -> snapshot-accept (best_<level>.pt). Target held
            # >= accept_bar for accept_probes probes -> DONE + clean exit.
            clevel_metrics_emit: dict = {}
            if consolidate_on:
                # `consolidate_sustain` is emitted post-gate via
                # clevel_metrics_emit below (on probe iters), so it is NOT
                # duplicated here.
                progress_metrics["consolidate_step"] = int(clevel_step)
                progress_metrics["consolidate_cooldown"] = int(
                    max(0, clevel_cooldown_until - global_it)
                )
            if consolidate_on and it % clevel_every == 0:
                from src.training import cold_probe as _cold_probe

                def _gate_probe(_entry):
                    """One gate probe. Identical threading to the mode-start
                    baseline probe and to the PLR cold probe: the resolved
                    honest distribution rides in `clevel_probe_kwargs`, so the
                    gate measures the protocol the deliverable is graded under
                    (v4's did not). Returns (rate, furthest, n_scored)."""
                    if _entry is None:
                        return None, None, 0
                    _r = _cold_probe.probe(
                        net, self.game_profile,
                        device=self.device, sequential=True, level_clear=True,
                        start_state=_entry,
                        rom_path=self.rom_path,
                        game=str(self.game_profile.get("name", "mario")),
                        **clevel_probe_kwargs,
                    )
                    _rate = _r.get("cold_seq_clear_rate")
                    return (
                        float(_rate) if _rate is not None else None,
                        _r.get("cold_furthest_seq"),
                        int(_r.get("cold_n_episodes") or 0),
                    )
                _probe_t0 = time.time()
                _tr, _, clevel_probe_n = _gate_probe(clevel_target_entry)
                clevel_tgt_rate = _tr if _tr is not None else -1.0
                clevel_protect_rates = {}
                for _plevel, _pentry in clevel_protect_entries.items():
                    _pr, _, _ = _gate_probe(_pentry)
                    if _pr is not None:
                        clevel_protect_rates[_plevel] = _pr
                _probe_secs = time.time() - _probe_t0
                _pmin = (
                    min(clevel_protect_rates.values())
                    if clevel_protect_rates else 1.0
                )
                # The probe's own Wilson LB, reported whether or not the gate
                # is armed with it, so the log is a receipt for the accept
                # arithmetic rather than a bare rate.
                clevel_tgt_lb = _oc_gate.wilson_lower_bound(
                    max(clevel_tgt_rate, 0.0) * clevel_probe_n, clevel_probe_n,
                    confidence=clevel_wilson_conf,
                )
                log.info(
                    "[vanilla_ppo] CONSOLIDATE PROBE iter %d: target %s=%.3f "
                    "(n=%d, wilson_lb=%.3f @%.2f, best=%.3f) protect=%s "
                    "sustain=%d/%d probe_secs=%.1f%s",
                    global_it, clevel_target, clevel_tgt_rate, clevel_probe_n,
                    clevel_tgt_lb, clevel_wilson_conf, best_tgt_rate,
                    clevel_protect_rates, clevel_sustain, clevel_need,
                    _probe_secs,
                    " [cooldown]" if global_it < clevel_cooldown_until else "",
                )
                _regressed = oc.protect_regressed(
                    clevel_baselines or {}, clevel_protect_rates, tol=clevel_tol
                )
                # Pure gate sequencing: rollback (protect regressed) > accept
                # (target strictly improved) > hold, plus the sustained-bar
                # termination. A rollback with no accepted snapshot yet degrades
                # to a hold (nothing to restore to — the champion IS the floor).
                if _regressed is not None and accepted_snapshot is None:
                    _regressed = None
                _gate = oc.gate_step(
                    regressed=_regressed, target_rate=(
                        clevel_tgt_rate if clevel_tgt_rate >= 0.0 else None
                    ),
                    best_rate=best_tgt_rate, sustain=clevel_sustain,
                    bar=clevel_bar, need=clevel_need, tol=clevel_tol,
                    target_n=clevel_probe_n,
                    use_wilson_bound=clevel_use_wilson,
                    wilson_confidence=clevel_wilson_conf,
                    accept_rule=clevel_accept_rule,
                    best_decay=clevel_best_decay,
                    best_floor=clevel_best_floor,
                )
                clevel_sustain = int(_gate["sustain"])
                clevel_done = bool(_gate["done"])
                if _gate["action"] == "rollback":
                    # Roll back to the last accepted snapshot, reset the
                    # optimizer, and freeze entropy high for a cooldown so the
                    # protect level can recover before the weld resumes.
                    net.load_state_dict(accepted_snapshot)
                    self._ppo_optimizer = self._build_ppo_optimizer(net)
                    optimizer = self._ppo_optimizer
                    clevel_cooldown_until = global_it + clevel_cooldown
                    self.entropy_coef = clevel_ent_from
                    self.rnd_intrinsic_coef = clevel_rnd_from
                    log.warning(
                        "[vanilla_ppo] *** CONSOLIDATE ROLLBACK *** protect "
                        "level %s regressed (%.3f < baseline %.3f); restored "
                        "%s + reset optimizer + froze entropy for %d iters.",
                        _regressed,
                        clevel_protect_rates.get(_regressed, -1.0),
                        (clevel_baselines or {}).get(_regressed, -1.0),
                        clevel_winner_name, clevel_cooldown,
                    )
                else:
                    # Hold or accept: the gate owns the incumbent from here.
                    # On a hold this is `best_decay`'s re-estimation (a no-op
                    # at the default 0.0, and never upward); on an accept it is
                    # THIS probe's point estimate, so the bar a future probe
                    # must clear is a measured rate over `probe.episodes`,
                    # never the n=1 high-water v4 pinned at 1.000. A FAILED
                    # probe returns the incumbent untouched — the -1.0 sentinel
                    # must never reach the re-estimation.
                    best_tgt_rate = float(_gate["best_rate"])
                    clevel_best_floor = _gate["best_floor"]
                if _gate["action"] == "accept":
                    # Protect held + target strictly improved: pin the new best.
                    accepted_snapshot = {
                        k: v.detach().cpu().clone()
                        for k, v in net.state_dict().items()
                    }
                    try:
                        torch.save(
                            {"net_state_dict": accepted_snapshot,
                             "iter": global_it,
                             "metric_name":
                                 f"consolidate_{clevel_target}_clear_rate",
                             "metric_value": best_tgt_rate},
                            str(self.checkpoint_dir / clevel_winner_name),
                        )
                    except Exception as _e:
                        log.warning(
                            "[vanilla_ppo] best_%s save failed: %s",
                            clevel_target, _e,
                        )
                    log.info(
                        "[vanilla_ppo] *** CONSOLIDATE ACCEPT *** target %s "
                        "improved to %.3f (rule=%s, lhs=%.3f, n=%d); "
                        "snapshot -> %s.",
                        clevel_target, best_tgt_rate, clevel_accept_rule,
                        float(_gate["accept_lhs"] or 0.0), clevel_probe_n,
                        clevel_winner_name,
                    )
                # Persist gate state for resume / inspection. The PROTOCOL
                # travels with the numbers: a sidecar holding only a rate
                # cannot say whether the gate measured a replay (v4) or a
                # distribution, and the accept arithmetic is unreadable
                # without the rule it used.
                try:
                    clevel_sidecar.write_text(json.dumps({
                        "target": clevel_target,
                        "best_tgt_rate": best_tgt_rate,
                        "target_rate": clevel_tgt_rate,
                        "target_lb": clevel_tgt_lb,
                        "target_n": int(clevel_probe_n),
                        "best_floor": clevel_best_floor,
                        "baselines": clevel_baselines or {},
                        "last_protect_rates": clevel_protect_rates,
                        "sustain": int(clevel_sustain),
                        "iter": int(global_it),
                        "probe_secs": round(float(_probe_secs), 2),
                        "protocol": dict(clevel_protocol),
                    }))
                except Exception:
                    pass
                clevel_metrics_emit = {
                    "consolidate_target_rate": float(clevel_tgt_rate),
                    "consolidate_target_lb": float(clevel_tgt_lb),
                    "consolidate_best_rate": float(best_tgt_rate),
                    "consolidate_protect_min": float(_pmin),
                    "consolidate_sustain": int(clevel_sustain),
                    "consolidate_probe_secs": float(_probe_secs),
                }
                if clevel_done:
                    # Final accepted snapshot + a DONE marker so a driver script
                    # can detect completion and chain the next level. The break
                    # (after the metrics emit below) is the clean exit.
                    try:
                        if accepted_snapshot is not None:
                            torch.save(
                                {"net_state_dict": accepted_snapshot,
                                 "iter": global_it,
                                 "metric_name":
                                     f"consolidate_{clevel_target}_clear_rate",
                                 "metric_value": best_tgt_rate},
                                str(self.checkpoint_dir / clevel_winner_name),
                            )
                        clevel_done_marker.write_text(json.dumps({
                            "target": clevel_target,
                            "final_rate": best_tgt_rate,
                            "final_lb": clevel_tgt_lb,
                            "final_n": int(clevel_probe_n),
                            "bar": clevel_bar,
                            "probes_sustained": int(clevel_sustain),
                            "winner": clevel_winner_name,
                            "protect_baselines": clevel_baselines or {},
                            "last_protect_rates": clevel_protect_rates,
                            "iter": int(global_it),
                            # The DONE marker is the run's terminal receipt;
                            # it carries the SAME protocol object the sidecar
                            # does, so a downstream driver never has to
                            # reconstruct it from the config and the two
                            # receipts can never disagree.
                            "protocol": dict(clevel_protocol),
                        }, indent=2))
                        log.info(
                            "[vanilla_ppo] *** CONSOLIDATE DONE *** target %s "
                            "held %.3f >= %.2f for %d probes; wrote %s + %s. "
                            "Exiting cleanly.", clevel_target, best_tgt_rate,
                            clevel_bar, clevel_need, clevel_winner_name,
                            clevel_done_marker.name,
                        )
                    except Exception as _e:
                        log.warning(
                            "[vanilla_ppo] DONE marker write failed: %s", _e,
                        )

            # === GO-EXPLORE UNSTICK BURST (Lane 5, DEFERRED — §Q3) ===
            # A bounded, reversible archive burst that fires ONLY when the
            # frontier rung stalls for `stall_patience` iters while being
            # genuinely reached. It diverts a small, capped env quota (below,
            # in the warm-start block) to Go-Explore return states to spread
            # exploration across the stalled rung, then harvests AT MOST ONE
            # deeper seed and RETRACTS. It is a direct `GoExploreArchive`
            # call — never the `go_explore_on` branch — so `smb_curriculum_
            # active` never flips; it touches neither entropy nor RND; and it
            # self-caps after `burst_iters`, so it can never become a
            # permanent Go-Explore takeover. All decisions route through the
            # unit-tested `oneshot_curriculum` module.
            if ge_burst_on:
                if not ge_burst_active:
                    # "Reaches" = the pool actually got to the frontier rung
                    # this iter (the block is "find the NEXT state," not
                    # "can't play this rung"). `consolidating` blocks it.
                    _reaches = bool(
                        (max_region_reached >= smb_curriculum_stage).any()
                    )
                    if oc.stall_ready(
                        ge_iters_since_advance, ge_stall_patience,
                        enabled=True, reaches=_reaches,
                        frontier=smb_curriculum_stage, ladder_size=LADDER_SIZE,
                        blocked=consolidating,
                    ):
                        from src.training.go_explore import GoExploreArchive
                        _stalled = ge_iters_since_advance
                        ge_archive = GoExploreArchive(
                            smb_sequential_cell, seed=global_it
                        )
                        ge_burst_active = True
                        ge_burst_remaining = ge_burst_iters
                        ge_burst_quota = oc.burst_quota(
                            num_envs, ge_burst_frac, ge_burst_cap
                        )
                        ge_iters_since_advance = 0
                        log.warning(
                            "[vanilla_ppo] *** GO-EXPLORE BURST ARMED *** "
                            "frontier %d stalled %d iters; diverting %d/%d "
                            "envs to archive returns for <=%d iters "
                            "(reversible; curriculum flag untouched).",
                            smb_curriculum_stage, _stalled, ge_burst_quota,
                            num_envs, ge_burst_iters,
                        )
                else:
                    ge_burst_remaining, _retract = oc.burst_tick(
                        ge_burst_remaining
                    )
                    # Consolidation wins over an in-flight burst: the
                    # arm-side `blocked=consolidating` gate is one-way, so
                    # force the retract here or the two could overlap for
                    # up to burst_iters (the "never concurrent" invariant).
                    if consolidating and not _retract:
                        _retract = True
                        log.warning(
                            "[vanilla_ppo] *** GO-EXPLORE BURST RETRACT *** "
                            "consolidation armed mid-burst; retracting early "
                            "(%d iters unused).", ge_burst_remaining,
                        )
                    if _retract:
                        # Harvest at most ONE deeper seed. A cell's stored
                        # score IS its region order, so read depth off it.
                        _cells = (
                            [(int(round(c.best_score)), c.state)
                             for c in ge_archive.cells.values()]
                            if ge_archive is not None else []
                        )
                        _harvest = oc.harvest_burst_seed(
                            _cells, smb_curriculum_stage
                        )
                        ge_burst_active = False
                        ge_iters_since_advance = 0
                        ge_bursts_done += 1
                        if _harvest is not None:
                            _hreg, _hstate = _harvest
                            # Inject as the NEXT rung's live-fill seed (§Q3):
                            # region is recomputed from RAM on warm-start, so
                            # a deeper-than-F+1 blob simply makes rung F+1
                            # immediately "beatable".
                            _target = min(
                                smb_curriculum_stage + 1, LADDER_SIZE - 1
                            )
                            smb_curriculum_states[_target] = _hstate
                            try:
                                _sp = (smb_curriculum_dir
                                       / f"stage_{_target:02d}.state")
                                _tp = _sp.with_suffix(".state.tmp")
                                _tp.write_bytes(_hstate)
                                _tp.replace(_sp)
                                _mp = (smb_curriculum_dir
                                       / f"stage_{_target:02d}.meta.json")
                                _tmp = _mp.with_suffix(".json.tmp")
                                _tmp.write_text(
                                    json.dumps({"anchor": int(_target)})
                                )
                                _tmp.replace(_mp)
                            except Exception as _e:
                                log.warning(
                                    "[vanilla_ppo] burst seed persist "
                                    "failed: %s", _e,
                                )
                            log.warning(
                                "[vanilla_ppo] *** GO-EXPLORE BURST HARVEST "
                                "*** burst #%d found a region-%d state (%d "
                                "cells); injected as rung %d seed. Retracting "
                                "to curriculum.",
                                ge_bursts_done, _hreg, len(_cells), _target,
                            )
                        else:
                            log.warning(
                                "[vanilla_ppo] *** GO-EXPLORE BURST RETRACT "
                                "*** burst #%d found no state past frontier %d "
                                "(%d cells); no seed harvested. Retracting to "
                                "curriculum.",
                                ge_bursts_done, smb_curriculum_stage,
                                len(_cells),
                            )
                        ge_archive = None
                progress_metrics["vanilla_ppo_ge_bursting"] = int(
                    ge_burst_active
                )
                progress_metrics["vanilla_ppo_ge_bursts_done"] = int(
                    ge_bursts_done
                )
                progress_metrics["vanilla_ppo_ge_stall_iters"] = int(
                    ge_iters_since_advance
                )
                progress_metrics["vanilla_ppo_ge_stall_notified"] = int(
                    stall_notifier.notified
                )

            # Apply the consolidation coefficient schedule (or the cyclic
            # fallback oscillation) for the NEXT update.
            if consolidating:
                self.entropy_coef = oc.lerp_coef(
                    cons_ent_from, cons_ent_to, cons_step, cons_ent_iters
                )
                self.rnd_intrinsic_coef = oc.lerp_coef(
                    cons_rnd_from, cons_rnd_to, cons_step, cons_ent_iters
                )
                cons_step += 1
                cons_no_gain += 1
            elif cyclic_mode:
                if it % cold_every == 0:
                    cyclic_phase ^= 1
                self.entropy_coef = cons_ent_from if cyclic_phase else cons_ent_to

            # Level-scoped consolidation schedule (the weld itself): decay
            # entropy/RND from->to over N iters via the shared lerp_coef. During
            # a post-rollback cooldown, FREEZE — hold exploration high and pause
            # the decay so the protect level recovers before the weld resumes.
            if consolidate_on:
                if global_it < clevel_cooldown_until:
                    self.entropy_coef = clevel_ent_from
                    self.rnd_intrinsic_coef = clevel_rnd_from
                else:
                    self.entropy_coef = oc.lerp_coef(
                        clevel_ent_from, clevel_ent_to,
                        clevel_step, clevel_ent_iters,
                    )
                    self.rnd_intrinsic_coef = oc.lerp_coef(
                        clevel_rnd_from, clevel_rnd_to,
                        clevel_step, clevel_ent_iters,
                    )
                    clevel_step += 1

            # Either schedule above can raise rnd_intrinsic_coef off a
            # zero baseline (RND started disabled, the consolidate/
            # consolidate_level ramp turns it on). build_rnd's build is
            # gated on the CURRENT coefficient, not just the pre-loop
            # value, so it must be re-offered here — otherwise `_rnd`
            # stays None forever and the coefficient climbs in the logs
            # while zero intrinsic reward or predictor loss ever runs.
            _exploration.build_rnd(
                log_msg=(
                    "[vanilla_ppo] RND enabled (%s): predictor=%d params, "
                    "intrinsic_coef=%.3f, loss_coef=%.3f"
                )
            )

            self._emit_metrics(
                generation=global_it,
                best_fitness=max(best_completed, best_in_progress),
                avg_fitness=(
                    mean_ep_return if n_complete else mean_inprogress_return
                ),
                stage=self.curriculum.current_stage.name,
                success_rate=vppo_success_rate,
                episodes=n_complete,
                ppo_loss=last_loss,
                ppo_policy_loss=last_policy_loss,
                ppo_value_loss=last_value_loss,
                ppo_entropy=last_entropy,
                vanilla_ppo_in_progress=n_in_progress,
                vanilla_ppo_clears=n_clears_this_iter,
                vanilla_ppo_rnd_loss=last_rnd_loss,
                vanilla_ppo_intrinsic_mean=rnd_intrinsic_mean,
                vanilla_ppo_count_bonus_mean=count_bonus_mean,
                # V29_STABILITY_2026-08-25.md F0: the five previously
                # missing PPO-update scalars. Pure observations — none
                # of the five feed back into the update itself.
                vanilla_ppo_clip_fraction=last_clip_fraction,
                vanilla_ppo_approx_kl=last_approx_kl,
                vanilla_ppo_grad_norm=last_grad_norm,
                vanilla_ppo_adv_mean=adv_mean,
                vanilla_ppo_adv_std=adv_std,
                vanilla_ppo_explained_variance=explained_variance,
                demo_anchor_coef=_demo_coef,
                demo_anchor_loss=(
                    float(_demo_loss_accum) / _demo_loss_n
                    if _demo_loss_n else 0.0
                ),
                vanilla_ppo_samples_per_sec=samples_per_sec,
                vanilla_ppo_realtime_x=realtime_x,
                **resume_metrics,
                **progress_metrics,
                **timing_metrics,
                **reward_breakdown_emit,
                **cold_metrics_emit,
                **clevel_metrics_emit,
                **_mech_metrics,
            )

            # Level-scoped consolidation reached its termination bar: the DONE
            # marker + final snapshot were written in the gate block, and the
            # final metrics were just emitted. Break for a clean exit so a
            # driver script can chain the next level.
            if clevel_done:
                log.info(
                    "[vanilla_ppo] consolidation of %s complete — exiting the "
                    "training loop.", clevel_target,
                )
                break

            # Drain the narrator events accumulated this iter onto the
            # GUI caption queue so the live stream is narrated (captions
            # + gold milestone banners). Non-blocking; GUI-only — in
            # headless training narrator_on is False and this is skipped.
            if narrator_on and self._narrator_queue is not None:
                from src.training.narrator import push_events_to_queue
                push_events_to_queue(self._narrator.drain(), self._narrator_queue)

            # ============== ANTI-COLLAPSE GUARD ==============
            # Snapshot the policy while it's healthy (low entropy +
            # improving fitness); roll back if it melts (entropy near
            # the uniform-random max for several iters running). Entropy
            # is the right trigger because it's stage-independent —
            # fitness legitimately drops when the curriculum reaches a
            # harder stage, but a healthy policy keeps entropy moderate;
            # only a genuine collapse drives entropy toward ln(A).
            iter_fitness = (
                mean_ep_return if n_complete else mean_inprogress_return
            )
            if (last_entropy < SMB_ENTROPY_HEALTHY_FRAC * _entropy_max
                    and iter_fitness > best_snapshot_fitness):
                best_snapshot_fitness = iter_fitness
                best_net_snapshot = {
                    k: v.detach().cpu().clone()
                    for k, v in net.state_dict().items()
                }
                collapse_strikes = 0
            elif last_entropy > SMB_ENTROPY_COLLAPSE_FRAC * _entropy_max:
                collapse_strikes += 1
                if (collapse_strikes >= SMB_COLLAPSE_PATIENCE
                        and best_net_snapshot is not None):
                    net.load_state_dict(best_net_snapshot)
                    # The melted Adam moments would re-melt the restored
                    # weights immediately; start fresh. Use the shared
                    # builder so the RND predictor stays in the optimizer
                    # (a hand-rolled rebuild silently dropped it).
                    self._ppo_optimizer = self._build_ppo_optimizer(net)
                    optimizer = self._ppo_optimizer
                    collapse_strikes = 0
                    log.warning(
                        "[vanilla_ppo] *** ANTI-COLLAPSE ROLLBACK *** "
                        "entropy %.3f > %.0f%% of max (%.3f) for %d iters; "
                        "restored best snapshot (fitness %.1f) + reset "
                        "optimizer.",
                        last_entropy, SMB_ENTROPY_COLLAPSE_FRAC * 100,
                        _entropy_max, SMB_COLLAPSE_PATIENCE,
                        best_snapshot_fitness,
                    )
            else:
                collapse_strikes = 0

            # Backward entropy guard, half 1 of 2: STRIP last iter's boost
            # so every other controller below reads the unboosted
            # coefficient and the boost can never compound across iters.
            # Only strips a value this guard itself wrote — if a
            # consolidation schedule reassigned entropy_coef in between,
            # the record is simply dropped.
            #
            # ORDER IS LOAD-BEARING: this block must stay ABOVE the
            # entropy-floor controller. That controller rescales
            # entropy_coef, which would break the float-equality check
            # below, silently drop the record, and leave the boost
            # multiplying an already-boosted coefficient every iter,
            # without bound. Enforced by
            # tests/test_vanilla_ppo_backward_guard_smoke.py::
            # test_guard_survives_an_active_entropy_floor_controller
            # (the profile that ships the guard runs the controller at
            # entropy_floor 0.02; the 1-1 profile has it off, so the
            # other smokes in that file cannot see this).
            if bwd_guard is not None and _bwd_guard_applied is not None:
                if self.entropy_coef == _bwd_guard_applied:
                    self.entropy_coef = _bwd_guard_base
                _bwd_guard_base = _bwd_guard_applied = None

            # Adaptive entropy-floor controller (opt-in via entropy_floor).
            # Keeps the policy from collapsing to a brittle deterministic
            # trajectory under sticky/jitter training — the mechanism that
            # makes a policy actually survive the honest sticky eval rather
            # than memorizing one start-locked path.
            if self.entropy_floor > 0.0:
                if last_entropy < self.entropy_floor:
                    self.entropy_coef = min(
                        self.entropy_coef * 1.5 + 1e-4, self.entropy_coef_max
                    )
                elif last_entropy > 1.5 * self.entropy_floor:
                    self.entropy_coef = max(
                        self.entropy_coef * 0.9, self._entropy_coef_base
                    )

            # Backward entropy guard, half 2 of 2: OBSERVE this iter's
            # entropy and re-apply the multiplier on top of whatever the
            # controllers above just decided. Runs last so the boost is
            # the outermost factor; note the effective ceiling while
            # armed is therefore entropy_coef_max * boost, by design —
            # the floor controller's clamp bounds the base, not the
            # emergency response to a collapse.
            if bwd_guard is not None:
                _bg_event = bwd_guard.observe(last_entropy)
                if _bg_event == GUARD_ARM:
                    log.warning(
                        "[backward-guard] ARMED at iter %d: trailing "
                        "entropy %.4f over %d iters < floor %.3f "
                        "(B4 v1 collapsed 0.19 -> 0.04 here). "
                        "entropy_coef %.5f -> %.5f (x%.2f) until the "
                        "trailing mean recovers above %.3f.",
                        global_it, bwd_guard.trailing_mean,
                        bwd_guard.samples, bwd_guard.floor,
                        self.entropy_coef,
                        self.entropy_coef * bwd_guard.boost,
                        bwd_guard.boost, bwd_guard.recover_floor,
                    )
                elif _bg_event == GUARD_DISARM:
                    log.info(
                        "[backward-guard] disarmed at iter %d: trailing "
                        "entropy %.4f >= %.3f. entropy_coef back to "
                        "%.5f (armed %d/%d of the last iters, %d arms "
                        "this run).",
                        global_it, bwd_guard.trailing_mean,
                        bwd_guard.recover_floor, self.entropy_coef,
                        bwd_guard.armed_recent, bwd_guard.history_len,
                        bwd_guard.arms,
                    )
                if bwd_guard.armed:
                    _bwd_guard_base = self.entropy_coef
                    _bwd_guard_applied = self.entropy_coef * bwd_guard.boost
                    self.entropy_coef = _bwd_guard_applied

            # Clear per-iteration episode-completion buffer so the next
            # iter reports against fresh episode data only.
            completed_returns.clear()
            completed_lengths.clear()

            # Reset the entire env pool between iterations: every iter
            # starts from N fresh cold-boot episodes. Avoids workers
            # idling in a post-done state when our reward-fn / stacker
            # logic doesn't match the rust pool's auto-step semantics.
            # Attribute the iter-boundary reset + warm-start — previously
            # part of the unbucketed ~17% iter-wall gap. This runs AFTER
            # the snapshot above, so it lands in the NEXT gen's row (a
            # constant one-gen offset; magnitude is what matters).
            # Diagnostic only.
            _iter_reset_t0 = time.perf_counter_ns()
            init_results = self.pool.reset_all()
            self._emit_frame_sink(init_results)
            # reset_all revives dead workers; a still-nonzero count means a
            # worker keeps re-panicking (a bad ROM/state edge). Log it loudly
            # so an overnight run never silently trains on a dead cohort.
            _ndead = getattr(self.pool, "num_dead", 0)
            if _ndead:
                log.warning(
                    "[vanilla_ppo] %d worker(s) failed to revive after reset "
                    "— they emit zero-frames into PPO; check for a bad ROM/state.",
                    _ndead,
                )
            # SMB mixed-stage curriculum warm-start. Past stage 0, keep
            # a majority of envs (SMB_CURRENT_STAGE_FRAC) at the current
            # (hardest) stage for focused training, and spread the rest
            # across earlier stages down to cold-boot stage 0. Whole-pool
            # warm-start landed every env on a level it might not play,
            # so a single hard stage produced uniform failure that the
            # entropy bonus turned into policy collapse; the spread keeps
            # earlier levels fresh and advantages informative.
            env_stage[:] = 0
            stage_seed_results = [None] * num_envs
            env_return_state = [None] * num_envs
            # CGSA: score the OLD episodes BEFORE the warm-start chain below
            # re-tags every env. (First version scored after re-tagging: the
            # fresh tags — start_gx still -1 — were counted as instant
            # failures, ~60 per boundary, and every zone's window filled with
            # zeros; the entrance read 0/1975.)
            if cgsa_on:
                for _ci in range(num_envs):
                    _cg_finish_episode(_ci, min_steps=CG_MIN_SCORE_STEPS)
            if plr_on:
                # PLR: sample each env's level by inverse-recent-success weight
                # and warm-start it from that level's entrance. Index 0 == the
                # cold-boot level (already at its entrance from reset_all above,
                # so no per-worker load); index>0 loads the level's entry blob.
                # Structurally identical to the mixed-stage branch below, with
                # PLR sampling in place of the current/spread split and NO
                # advance / capture / Go-Explore.
                for i in range(num_envs):
                    _lvl = plr_ctx.sample()
                    plr_ctx.env_level[i] = _lvl
                    env_stage[i] = plr_ctx.index_of(_lvl)
                any_warm = False
                for i in range(num_envs):
                    k = int(env_stage[i])
                    if k > 0 and smb_curriculum_states[k] is not None:
                        try:
                            self.pool.load_worker_state(
                                i, smb_curriculum_states[k]
                            )
                            any_warm = True
                        except Exception as e:
                            log.warning(
                                "[vanilla_ppo] PLR warm-start env %d level %s "
                                "failed: %s — cold boot.", i,
                                plr_ctx.env_level[i], e,
                            )
                            env_stage[i] = 0
                            plr_ctx.env_level[i] = plr_ctx.idx_to_level[0]
                if any_warm:
                    noop_actions = np.zeros(
                        self.pool.num_workers, dtype=np.uint8
                    )
                    init_results = self.pool.step_all(noop_actions)
                    self._emit_frame_sink(init_results)
                stage_seed_results = [
                    init_results[i] if int(env_stage[i]) > 0 else None
                    for i in range(num_envs)
                ]
                if it % 25 == 0:
                    log.info(
                        "[vanilla_ppo] PLR warm-start dist=%s success=%s",
                        plr_ctx.distribution(),
                        {k: round(v, 2)
                         for k, v in plr_ctx.success_rates().items()},
                    )
            elif consolidate_on:
                # 100% of the pool inside the TARGET level's rungs (round-robin
                # over its x-buckets). No Frontier/Retention/Spread, no advance,
                # no Go-Explore — the ladder is read-only here. On death, the
                # auto-reset path above reloads each env's OWN target rung
                # (env_stage[i]), so a rollout still runs multiple attempts.
                env_stage[:] = oc.consolidate_assignment(
                    num_envs, clevel_target_rungs
                )
                any_warm = False
                for i in range(num_envs):
                    k = int(env_stage[i])
                    _seed = smb_curriculum_states[k] if k > 0 else None
                    if _seed is not None:
                        try:
                            self.pool.load_worker_state(i, _seed)
                            any_warm = True
                        except Exception as e:
                            log.warning(
                                "[vanilla_ppo] consolidate warm-start env %d "
                                "rung %d failed: %s — cold boot.", i, k, e,
                            )
                            env_stage[i] = 0
                if any_warm:
                    noop_actions = np.zeros(self.pool.num_workers, dtype=np.uint8)
                    init_results = self.pool.step_all(noop_actions)
                    self._emit_frame_sink(init_results)
                stage_seed_results = [
                    init_results[i]
                    if (int(env_stage[i]) > 0
                        and smb_curriculum_states[int(env_stage[i])] is not None)
                    else None
                    for i in range(num_envs)
                ]
                _dist: dict = {}
                for k in env_stage.tolist():
                    _dist[k] = _dist.get(k, 0) + 1
                log.info(
                    "[vanilla_ppo] consolidate warm-start target=%s rung "
                    "distribution %s", clevel_target, dict(sorted(_dist.items())),
                )
            elif ladder_on:
                # Three-way warm-start partition (§Q2): Frontier F/F-1,
                # Retention on every cleared level-entry (the anti-forgetting
                # floor, bumped to 40% while a forgetting alarm is active),
                # Spread uniform below F. All rung math in oneshot_curriculum;
                # here we just load each env's rung seed.
                _part = oc.warm_start_partition(
                    num_envs, smb_curriculum_stage, ladder,
                    frontier_frac=frontier_frac, retention_frac=retention_frac,
                )
                env_stage[:] = _part.assignment
                # Go-Explore burst (Lane 5): while a burst is in flight, divert
                # a capped quota of envs off their ladder rung onto archive
                # return states (spreads exploration across the stalled rung).
                # The rest of the pool stays on the curriculum, so the burst is
                # a diversion, never a takeover. `_burst_seeds[i]` overrides
                # env i's warm-start blob for this iter only.
                _burst_seeds: dict = {}
                if (ge_burst_active and ge_archive is not None
                        and ge_burst_quota > 0 and len(ge_archive) > 0):
                    _returns = ge_archive.select_return_states(ge_burst_quota)
                    for _j, _rs in enumerate(_returns):
                        if _j < num_envs and _rs is not None:
                            _burst_seeds[_j] = _rs
                    if _burst_seeds:
                        log.info(
                            "[vanilla_ppo] GO-EXPLORE burst diverting %d/%d "
                            "envs to archive returns (%d cells).",
                            len(_burst_seeds), num_envs, len(ge_archive),
                        )
                any_warm = False
                for i in range(num_envs):
                    k = int(env_stage[i])
                    _seed = _burst_seeds.get(i)
                    if _seed is None and k > 0:
                        _seed = smb_curriculum_states[k]
                    if _seed is not None:
                        try:
                            self.pool.load_worker_state(i, _seed)
                            any_warm = True
                        except Exception as e:
                            log.warning(
                                "[vanilla_ppo] ladder warm-start env %d rung "
                                "%d failed: %s — cold boot.", i, k, e,
                            )
                            env_stage[i] = 0
                            _burst_seeds.pop(i, None)
                if any_warm:
                    noop_actions = np.zeros(self.pool.num_workers, dtype=np.uint8)
                    init_results = self.pool.step_all(noop_actions)
                    self._emit_frame_sink(init_results)
                stage_seed_results = [
                    init_results[i]
                    if (i in _burst_seeds
                        or (int(env_stage[i]) > 0
                            and smb_curriculum_states[int(env_stage[i])] is not None))
                    else None
                    for i in range(num_envs)
                ]
                _dist: dict = {}
                for k in env_stage.tolist():
                    _dist[k] = _dist.get(k, 0) + 1
                log.info(
                    "[vanilla_ppo] ladder warm-start F=%d, F/R/S=%d/%d/%d, "
                    "rung distribution %s", smb_curriculum_stage,
                    _part.n_frontier, _part.n_retention, _part.n_spread,
                    dict(sorted(_dist.items())),
                )
            elif smb_curriculum_stage > 0:
                n_current = max(
                    1, int(round(num_envs * SMB_CURRENT_STAGE_FRAC))
                )
                env_stage[:n_current] = smb_curriculum_stage
                # Round-robin the remaining envs over earlier stages
                # [0 .. current-1] for even coverage.
                for j, i in enumerate(range(n_current, num_envs)):
                    env_stage[i] = j % smb_curriculum_stage
                any_warm = False
                for i in range(num_envs):
                    k = int(env_stage[i])
                    if k > 0 and smb_curriculum_states[k] is not None:
                        try:
                            self.pool.load_worker_state(
                                i, smb_curriculum_states[k]
                            )
                            any_warm = True
                        except Exception as e:
                            log.warning(
                                "[vanilla_ppo] warm-start env %d stage "
                                "%d failed: %s — cold boot.", i, k, e,
                            )
                            env_stage[i] = 0
                if any_warm:
                    # Re-step a no-op to flush post-restore frames into
                    # init_results (stage-0 envs just advance one frame).
                    noop_actions = np.zeros(
                        self.pool.num_workers, dtype=np.uint8
                    )
                    init_results = self.pool.step_all(noop_actions)
                    self._emit_frame_sink(init_results)
                # Per-env stage-start observation for mid-rollout auto-
                # reset re-seeding. Only warm-started (stage>0) envs get
                # a seed; cold stage-0 envs keep None (auto-reset freezes
                # them, as there's no per-env cold-boot reset).
                stage_seed_results = [
                    init_results[i] if int(env_stage[i]) > 0 else None
                    for i in range(num_envs)
                ]
                _dist: dict = {}
                for k in env_stage.tolist():
                    _dist[k] = _dist.get(k, 0) + 1
                log.info(
                    "[vanilla_ppo] mixed warm-start stage distribution "
                    "(stage:count): %s", dict(sorted(_dist.items())),
                )
            elif go_explore_archive is not None and len(go_explore_archive) > 0:
                # ===== GO-EXPLORE RETURN (iter boundary) =====
                # Warm-start each env from a frontier cell the archive picks
                # (weighted toward under-explored cells) instead of leaving
                # every env at cold boot. Reuses the curriculum's no-op flush
                # + the stacked_obs rebuild below, so there is zero extra
                # re-seed bookkeeping and no desync risk.
                if cgsa_on:
                    # CGSA priority restarts: ALL restarts flow through the
                    # weld-frontier selector (the entrance is a pseudo-zone in
                    # the same distribution — no fixed share; once welded it
                    # drops to maintenance like everything else).
                    _states = []
                    for i in range(num_envs):
                        _k, _blob2 = _cgsa_select_state()
                        _states.append(_blob2)
                        if _blob2 is not None:
                            cg_env_cell[i] = _k
                            cg_env_start_gx[i] = -1
                            _sticky_p_env[i] = _cg_entry(_k)["p"]
                else:
                    _states = go_explore_archive.select_return_states(num_envs)
                _any_return = False
                for i in range(num_envs):
                    _blob = _states[i]
                    if _blob is not None:
                        try:
                            self.pool.load_worker_state(i, _blob)
                            env_return_state[i] = _blob
                            _any_return = True
                        except Exception as e:
                            log.warning(
                                "[vanilla_ppo] go_explore return env %d "
                                "failed: %s — cold boot.", i, e,
                            )
                if _any_return:
                    noop_actions = np.zeros(
                        self.pool.num_workers, dtype=np.uint8
                    )
                    init_results = self.pool.step_all(noop_actions)
                    self._emit_frame_sink(init_results)
                stage_seed_results = [
                    init_results[i] if env_return_state[i] is not None else None
                    for i in range(num_envs)
                ]
            for fn in reward_fns:
                fn.reset()
            ep_returns[:] = 0
            ep_lengths[:] = 0
            # reward_fns.reset() above zeroed each env's `completion`
            # breakdown, so the last-seen tracker must zero too — otherwise a
            # stale-high prior value masks the FIRST clear of the new iter
            # (cur_comp == stale prev → no positive diff), under-counting both
            # n_clears and the PLR clear flag. Sync them here.
            prev_completion_total[:] = 0.0
            episode_recorded[:] = False
            if sil_on:
                # The reset_all above truncated every live episode; a
                # censored attempt is not a clear — drop its accumulator.
                for _si in range(num_envs):
                    _sil_drop_episode(_si)
            # ===== BACKWARD CURRICULUM: the iter-boundary restart =====
            # `reset_all` above truncated every live episode and put each
            # env back at the profile start state. Two consequences the
            # curriculum has to handle:
            #
            # 1. A truncated attempt is not evidence either way, so its
            #    rung provenance is DROPPED — except when it had already
            #    cleared, which is a completed outcome and keeps its
            #    rung's credit.
            # 2. Left alone, every env would begin the next rollout at the
            #    entrance, and tau would describe a distribution the
            #    policy hardly ever trains on. So the draw owns this
            #    restart exactly as it owns the mid-rollout one. Only
            #    stage-0 envs are touched — a curriculum/PLR warm start
            #    (env_stage > 0) keeps its own level, same rule the inline
            #    branch follows.
            if bwd_on:
                for _bi in range(num_envs):
                    _bwd_src = int(_bwd_env_src[_bi])
                    if _bwd_src < 0:
                        continue
                    if not _bwd_env_clear[_bi]:
                        # Cut by the iter boundary: same censored-data
                        # rule as a budget cut, same accounting.
                        _bwd_score_truncation(_bi, _bwd_src)
                        continue
                    if _bwd_src == 1:
                        bwd_sched.record(True, tau=int(_bwd_env_tau[_bi]))
                    else:
                        bwd_sched.record_entrance(True)
                bwd_sched.maybe_advance()
            _bwd_env_src[:] = -1
            _bwd_env_clear[:] = False
            _bwd_env_trunc[:] = False
            if bwd_on:
                _bwd_any = False
                for _bi in range(num_envs):
                    if int(env_stage[_bi]) != 0:
                        continue
                    _bwd_pick = bwd.ENTRANCE
                    if not bwd_pin:
                        _bwd_win = _bwd_window(
                            bwd_tape, bwd_sched.tau,
                            window_frames=bwd_window_frames,
                            frames_per_step=bwd_frames_per_entry,
                        )
                        _bwd_pick = bwd.draw_restart(
                            len(_bwd_win), bwd_entrance_w,
                            float(np.random.random()),
                        )
                    if _bwd_pick != bwd.ENTRANCE:
                        try:
                            self.pool.load_worker_state(
                                _bi, bwd_blobs[_bwd_win[_bwd_pick]]
                            )
                            _sticky_restart(_bi)
                            _bwd_any = True
                        except Exception as e:
                            log.warning(
                                "[backward] env %d restart at tau %d "
                                "failed: %s — entrance.", _bi,
                                bwd_sched.tau, e,
                            )
                            _bwd_pick = bwd.ENTRANCE
                    _bwd_env_tau[_bi] = bwd_sched.tau
                    _bwd_env_src[_bi] = (
                        0 if _bwd_pick == bwd.ENTRANCE else 1
                    )
                    _bwd_env_budget[_bi] = (
                        bwd_budget.steps_for(
                            0 if _bwd_pick == bwd.ENTRANCE
                            else int(_bwd_win[_bwd_pick])
                        ) if bwd_budget is not None else 0
                    )
                if _bwd_any:
                    # Flush the post-restore frames into init_results, the
                    # same no-op step the curriculum warm start takes; the
                    # stacker re-seed at the end of this block reads it.
                    init_results = self.pool.step_all(
                        np.zeros(self.pool.num_workers, dtype=np.uint8)
                    )
                    self._emit_frame_sink(init_results)
            for _wi in range(num_envs):
                wave_prev_phi[_wi] = None  # fresh episodes, no shaping carryover
            wave_peak_phi[:] = 0.0
            wave_lost_count[:] = 0
            # Re-arm every env for the next iter. `set_worker_done(i, True)`
            # calls from this iter are cleared automatically by reset_all
            # (per the rust pool's contract).
            active_in_iter[:] = True
            # Reset per-iter world/level trackers. After the warm-start
            # block above, half the envs are at W1-L2 and half at W1-L1
            # — but neither has stepped yet, so initial position == max
            # position for this iter. The first rollout step will
            # update these from the post-restore RAM.
            max_world_level_packed[:] = 0
            end_world_level_packed[:] = 0
            # (CGSA episode scoring moved ABOVE the warm-start chain — it must
            # run before re-tagging, see the comment there.)
            # Per-iter reset (was missing): without this, vanilla_ppo_max_x
            # is a cumulative running max that preserves an old peak even
            # after the policy regresses/collapses — which masked a 1-4
            # collapse (metric held at 2432 while the policy fell to 814).
            max_x_reached[:] = 0
            # Per-iter reset of the sub-stage rung tracker to OFF_LADDER, so a
            # non-progressing / warping env can never satisfy the advance gate.
            max_region_reached[:] = -1
            # Go-Explore stall clock (Lane 5): count iters since the frontier
            # last advanced (reset on advance / arm / retract). Incremented
            # once per completed iter so `stall_ready` fires at exactly
            # `stall_patience` iters of no advance.
            if ge_burst_on:
                ge_iters_since_advance += 1
                # Pure addition: surface a prolonged stall to a human via a
                # macOS notification. Fires at most once per stall episode
                # (see StallNotifier) and can never affect training —
                # notify_macos swallows every failure (missing osascript,
                # non-macOS host, etc.) and logs a warning instead.
                try:
                    stall_notifier.maybe_notify(ge_iters_since_advance)
                except Exception as _e:
                    log.warning(
                        "[vanilla_ppo] stall notification check failed "
                        "(non-fatal): %s", _e,
                    )
            # Clear per-iter clear-counter + per-env completion-baseline.
            # Reward fns were just reset() above, so their `completion`
            # breakdowns are back to 0 — match that here.
            prev_completion_total[:] = 0
            n_clears_this_iter = 0
            if self._is_tile_mode:
                stacked_obs = [
                    tile_stackers[i].reset(self._tile_extractor.extract(r.ram_snapshot))
                    for i, r in enumerate(init_results)
                ]
            else:
                stacked_obs = [
                    stackers[i].reset(r.frame, getattr(r, "preprocessed", None))
                    for i, r in enumerate(init_results)
                ]
            self._gen_timer.add(
                "iter_reset", time.perf_counter_ns() - _iter_reset_t0
            )

            # Checkpoint every 10 iters (the poison guard + atomic write now
            # live in `CheckpointManager.save_iter`). Returns the finiteness
            # verdict so the winner-retention gate below stays identical to
            # the inline block.
            _params_finite = ckpt.save_iter(
                net=net,
                optimizer=optimizer,
                adv_net=_adv_net,
                adv_opt=_adv_opt,
                bwd_on=bwd_on,
                bwd_sched=bwd_sched,
                anticollapse=(
                    best_net_snapshot,
                    best_snapshot_fitness,
                    collapse_strikes,
                ),
                it=it,
                global_it=global_it,
                curriculum_resume={
                    "smb_stage_clear_history": [
                        int(x) for x in smb_stage_clear_history
                    ],
                    "smb_pastfrac_history": [
                        float(x) for x in smb_pastfrac_history
                    ],
                    "ge_burst_active": bool(ge_burst_active),
                    "ge_burst_remaining": int(ge_burst_remaining),
                    "ge_burst_quota": int(ge_burst_quota),
                    "ge_stall_notified": bool(stall_notifier.notified),
                    "ge_iters_since_advance": int(ge_iters_since_advance),
                    "ge_bursts_done": int(ge_bursts_done),
                },
            )
            if it > 0 and it % 10 == 0 and _params_finite:
                # Retain the best-ever policy under winners/ (excluded from
                # rotation) so `make demo`/`make eval` stay pointed at a real
                # win even if the run later self-collapses. save_winner only
                # overwrites when the metric strictly beats the stored best.
                # LADDER MODE re-keys this on the COLD sequential clear rate
                # (the DoD number) and pins the exact net that scored it — so
                # training telemetry (the 1-1-latched clear_rate) never selects
                # the deliverable (§Q2).
                try:
                    if consolidate_on:
                        # The mode's deliverable is the gate-accepted
                        # best_<level>.pt (written on accept in the gate block);
                        # the shared winners/ slot is left untouched so a
                        # per-level weld can't clobber the cold champion under a
                        # different metric.
                        pass
                    elif ladder_on:
                        if best_cold_snapshot is not None and best_cold_rate >= 0.0:
                            save_winner(
                                best_cold_snapshot,
                                game=str(self.game_profile.get("name", "game")),
                                metric_value=float(best_cold_rate),
                                out_dir=self.checkpoint_dir,
                                metric_name="cold_seq_clear_rate",
                                source_iter=global_it,
                            )
                    else:
                        # BACKWARD MODE re-keys the winner on the TRAILING
                        # entrance success rate, and only once tau has
                        # reached the entrance: the raw in-training
                        # clear_rate is inflated by near-flag rung restarts
                        # (B4 v1 receipt: best.pt landed at iter 80, mid-
                        # ladder, and cold-entrance greedy scored 0.02
                        # while the never-snapshotted entrance-era policy
                        # trained at 0.25). Pre-entrance nets are never
                        # the deliverable.
                        # Elsewhere the honest cold probe (PLR mode's
                        # `last_cold_metrics`) re-keys the winner the same
                        # way — the raw rate is a loudly-flagged last
                        # resort (see _select_winner_metric).
                        _wm_val, _wm_name = _select_winner_metric(
                            vppo_success_rate,
                            cold_rate=last_cold_metrics.get(
                                "cold_seq_clear_rate"
                            ),
                            bwd_snapshot=(
                                bwd_sched.snapshot()
                                if bwd_on and bwd_sched is not None
                                else None
                            ),
                        )
                        if _wm_val is not None:
                            save_winner(
                                {k: v.detach().cpu() for k, v in net.state_dict().items()},
                                game=str(self.game_profile.get("name", "game")),
                                metric_value=_wm_val,
                                out_dir=self.checkpoint_dir,
                                metric_name=_wm_name,
                                source_iter=global_it,
                            )
                except Exception as exc:
                    log.warning("[vanilla_ppo] winner retention failed: %s", exc)

            # Persist the Go-Explore archive on its OWN cadence (dedented out
            # of the 10-iter checkpoint block above) so save_every actually
            # takes effect instead of being masked by lcm(10, save_every).
            if go_explore_archive is not None and it % go_explore_save_every == 0:
                _exploration.save_go_explore()

    def _save_checkpoint(self, gen: int, keep_last: int = 5) -> None:
        path = self.checkpoint_dir / f"gen_{gen:05d}.pt"
        save_checkpoint_atomic(
            path=path,
            ga=self.ga,
            curriculum=self.curriculum,
            checkpoint_dir=self.checkpoint_dir,
        )
        log.info("Saved checkpoint: %s", path)
        maybe_export_elite_to_coreml(
            checkpoint_path=path,
            ga=self.ga,
            make_network=self._make_network,
            num_actions=len(self.action_space),
            is_tile_mode=self._is_tile_mode,
        )
        rotate_old_checkpoints(self.checkpoint_dir, keep_last=keep_last)

    @staticmethod
    def _random_shift(states: torch.Tensor, pad: int = 4) -> torch.Tensor:
        """DrQ random-shift augmentation.

        `states` shape: (T, C, H, W). Pads spatially with replicate
        boundaries by `pad` on each side, then samples a single random
        crop offset and applies it across the entire trajectory so the
        frame-stack channels stay temporally consistent.
        """
        if pad <= 0:
            return states
        # F.pad on a 4D tensor takes (left, right, top, bottom). Replicate
        # mode keeps edge intensity rather than introducing a black border
        # the encoder would learn to special-case.
        padded = F.pad(states, (pad, pad, pad, pad), mode="replicate")
        h_off = int(torch.randint(0, 2 * pad + 1, ()).item())
        w_off = int(torch.randint(0, 2 * pad + 1, ()).item())
        _, _, h, w = states.shape
        return padded[:, :, h_off:h_off + h, w_off:w_off + w]

    def _emit_metrics(self, **metrics) -> None:
        self._metrics_sink.emit(**metrics)

    def _apply_pending_backward_state(self, sched) -> None:
        """Load a resumed checkpoint's backward cursor into `sched`.

        Called once, right after the TauScheduler is built (tau only means
        something against a loaded tape, so this cannot happen at
        checkpoint-load time). No-op when nothing was stashed — which is
        the case for every checkpoint written before the cursor was
        persisted, so old checkpoints resume exactly as they did before:
        at the configured `tau_init`.

        A state the scheduler rejects (re-minted tape, corrupt counters)
        is a warning, not a crash: the run continues from the configured
        rung, which is strictly what it did before this restore existed.
        """
        state = getattr(self, "_pending_backward_curriculum", None)
        self._pending_backward_curriculum = None
        if not state:
            return
        before = sched.tau
        try:
            sched.load_state_dict(state)
        except Exception as exc:
            log.warning(
                "[backward] cursor NOT restored from checkpoint (%s) — "
                "starting at tau=%d as configured", exc, before,
            )
            return
        snap = sched.snapshot()
        log.info(
            "[backward] RESUMED cursor from checkpoint: tau=%d/%d "
            "(config would have started at %d) | trailing %d/%d | "
            "advances=%d | entrance %d/%d%s",
            snap["tau"], snap["n_entries"] - 1, before,
            snap["successes"], snap["attempts"], snap["advances"],
            snap["entrance_successes"], snap["entrance_attempts"],
            "  AT-ENTRANCE" if snap["at_entrance"] else "",
        )

    def _build_ppo_optimizer(self, net) -> "torch.optim.Optimizer":
        """Build the PPO Adam over the policy net + (when RND is enabled)
        the RND predictor. SINGLE source of truth so the three build
        sites — initial GA-path build, initial vanilla build, and the
        anti-collapse rollback rebuild — can never drift. They did: the
        rollback rebuilt over net.parameters() only, silently dropping
        the RND predictor from the optimizer (predictor then never
        trained again and accumulated grads unbounded)."""
        params = list(net.parameters())
        if self._rnd is not None:
            params += list(self._rnd.predictor.parameters())
        # CPU-device runs (tile mode) get PyTorch's fused Adam: on the
        # ~14k-param tile net + RND predictor the per-parameter Python
        # loop over the moment updates is optimizer overhead, not math,
        # and fusing it into one C++ kernel trims ~18% off the tile
        # minibatch (~0.09 ms/mb). Same algorithm, identical moment
        # math — a drop-in impl swap. MPS has no fused Adam, so the
        # pixel path is unaffected. Fall back to the default impl if
        # this torch build rejects fused (older build, unsupported
        # param dtype/device).
        if self.device.type == "cpu":
            try:
                return torch.optim.Adam(
                    params, lr=self.reinforce_lr, fused=True
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                log.warning(
                    "fused Adam unavailable on CPU (%s) — falling back "
                    "to the default optimizer impl", exc,
                )
        return torch.optim.Adam(params, lr=self.reinforce_lr)

    def _teardown(self) -> None:
        """Shared run() cleanup: stop the audio drainer, shut down the
        pool, break the frame_sink reference cycle, stop the mixer, and
        flush+close the metrics writer. Called from BOTH trainer modes'
        finally. vanilla_ppo used to `return` before the GA path's
        finally and so skipped all of this — leaking the pool's native
        buffers, a live daemon thread, the audio stream, and the TB tail
        on every GUI Start→Stop (the exact ~17 GB retention cycle the GA
        path already guards). Best-effort: each step is independently
        guarded so one failure can't strand the rest."""
        import gc as _gc
        try:
            self._stop_audio_drainer()
        except Exception as exc:
            log.warning("audio drainer stop failed: %s", exc)
        if self.pool:
            try:
                self.pool.shutdown()
            except Exception as exc:
                log.warning("pool shutdown failed: %s", exc)
        self.pool = None
        # Drop the GUI callback so the AppController ↔ Trainer cycle
        # releases the pool buffers without waiting for the cycle GC.
        self._frame_sink = None
        if self._audio_mixer is not None:
            try:
                self._audio_mixer.stop()
            except Exception:
                pass
        try:
            self._metrics_sink.close()
        except Exception:
            pass
        _gc.collect()

    def _emit_frame_sink(self, results) -> None:
        """Push step results to the optional frame sink (GUI live-view).

        No-op in headless mode. If the sink raises, warn ONCE and keep
        training — a broken GUI sink shouldn't kill the run, but it also
        shouldn't vanish silently (the old `except: pass` lost the live
        view with zero signal).
        """
        if self._frame_sink is None:
            return
        try:
            self._frame_sink(results)
        except Exception as e:
            if not self._frame_sink_warned:
                self._frame_sink_warned = True
                log.warning(
                    "frame_sink raised (%s); suppressing further frame-sink "
                    "errors this run. GUI live-view may be stale.", e,
                )

    def stop(self) -> None:
        self._running = False


def load_game_profile(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Re-export so existing `from src.training.trainer import find_latest_checkpoint`
# call sites in scripts/, GUI, and tests keep working without churn.
find_latest_checkpoint = _find_latest_checkpoint


def main() -> None:
    # Become our own process group leader so Ctrl-C / SIGTERM kills the
    # whole tree with one signal. Rust workers run in-process via rayon
    # so there's no child-process tree to tear down, but CLI users still
    # get the expected "one Ctrl-C ends it all" behavior.
    import os as _os
    try:
        _os.setpgrp()
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="Neural Entertainment System headless trainer")
    parser.add_argument("--rom", required=True, help="Path to .nes ROM file")
    parser.add_argument("--profile", required=True, help="Path to game profile YAML")
    # 24 workers measured as the post-PGO throughput peak on M4 Max
    # (12P + 4E cores). Pre-PGO the sweet spot was 20; PGO shrinks
    # per-worker cost so we can pack more workers before context-
    # switching overhead dominates. See
    # docs/proposals/archive/hot_path_baseline.md "Worker-count scaling".
    # Users on smaller chips should drop to their physical core count;
    # past ~24 on M4 Max throughput plateaus.
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--max-episode-steps", type=int, default=4000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed threaded through numpy/random/torch for reproducible runs",
    )
    parser.add_argument("--start-state", default=None, help="Path to a .state.bin recorded via the GUI")
    parser.add_argument(
        "--auto-start-state",
        action="store_true",
        help=(
            "Instead of --start-state, pick the deepest auto-saved "
            "state from <checkpoint_dir>/auto_curriculum/. Each training "
            "run then starts from where the previous run's best "
            "genome reached — compound curriculum without manual YAML edits."
        ),
    )
    parser.add_argument(
        "--reward-overrides",
        default=None,
        help="Path to a YAML with reward_weights / reinforce / ga_params overrides, merged on top of --profile",
    )
    parser.add_argument(
        "--bc-demo",
        default=None,
        help="Path to a .state.bin to use as a behavioral-cloning demo (pre-trains genomes to imitate it before GA starts)",
    )
    parser.add_argument(
        "--env-backend",
        choices=("nes_core",),
        default="nes_core",
        help=(
            "Emulator core. `nes_core` is the in-tree Rust NES core — "
            "real APU audio, per-worker panic-safe, NEON-SIMD preprocess, "
            "Rust rewards. The only supported backend."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help='Resume from a specific checkpoint path, or "auto" to pick the latest in --checkpoint-dir',
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from src.utils.logging_config import configure as _configure_logging
    _configure_logging(verbose=args.verbose, log_dir=args.checkpoint_dir)

    profile = load_game_profile(args.profile)
    if args.reward_overrides:
        with open(args.reward_overrides) as _fh:
            override = yaml.safe_load(_fh) or {}
        for section in ("reward_weights", "reinforce", "ga_params"):
            if section in override:
                merged = dict(profile.get(section, {}))
                merged.update(override[section])
                profile[section] = merged

    resume_from: Optional[str] = None
    if args.resume == "auto":
        latest = find_latest_checkpoint(args.checkpoint_dir)
        resume_from = str(latest) if latest else None
    elif args.resume:
        resume_from = args.resume

    env_spec = "nes_core:NESEnvironment"

    # Resolve --auto-start-state: pick the deepest snapshot currently
    # in the auto_curriculum dir. Depth keys are encoded in the
    # filename (depth_<d0>_<d1>_<d2>_<ts>.state.bin) so we can rank
    # lexicographically and the deepest wins.
    start_state = args.start_state
    if args.auto_start_state and not start_state:
        auto_dir = Path(args.checkpoint_dir) / "auto_curriculum"
        candidates = sorted(auto_dir.glob("depth_*.state.bin")) if auto_dir.exists() else []
        if candidates:
            start_state = str(candidates[-1])
            log.info("--auto-start-state resolved to %s", start_state)
        else:
            log.info(
                "--auto-start-state requested but %s has no snapshots; "
                "starting from cold reset", auto_dir,
            )
    trainer = Trainer(
        rom_path=args.rom,
        game_profile=profile,
        num_instances=args.instances,
        population_size=args.population,
        checkpoint_dir=args.checkpoint_dir,
        max_episode_steps=args.max_episode_steps,
        start_state_path=start_state,
        bc_demo_path=args.bc_demo,
        env_spec=env_spec,
        seed=args.seed,
    )
    trainer.run(num_generations=args.generations, resume_from=resume_from)


if __name__ == "__main__":
    main()
