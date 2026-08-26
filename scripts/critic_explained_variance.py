#!/usr/bin/env python3
"""F1 — critic explained variance against realized returns.

The kill switch registered in `docs/proposals/V29_STABILITY_2026-08-25.md`
(§"F1 — the cheap mechanism falsifier"). V29's top-ranked mechanism claims
the CRITIC DEGRADES over the back half of a run. The only evidence on disk
is that the raw Huber `ppo_value_loss` rises — which is equally consistent
with the critic being fine and the RETURNS becoming harder to predict as
the policy sharpens and outcomes go bimodal. Raw value loss cannot tell
those apart; a scale-free fit statistic can:

    EV = 1 - Var(R - V_t(s)) / Var(R)

EV is invariant to any additive constant in `V` and to the scale of `R`,
so "the targets got noisier" moves the denominator and the numerator
together and leaves EV flat, while "the fit degraded" drives EV down. That
is the whole discriminator.

Two variants, and the contrast between them is the point:

* ``EV_onpolicy(t)`` — checkpoint `t`'s critic scored on a rollout that
  checkpoint `t`'s own policy collected. Confounded by design: a
  collapsing policy visits a different state distribution, so a decline
  here is not attributable to fit.
* ``EV_fixed(t)`` — checkpoint `t`'s critic scored on a FROZEN reference
  batch collected once per run from that run's corrected-peak checkpoint
  `P`. Same states, same targets, every `t`. This is the registered gating
  quantity.

WHERE R AND V COME FROM (the load-bearing design decision)
----------------------------------------------------------
An archived `vanilla_ppo_iter_*.pt` carries weights, not data: there is no
banked rollout in this tree to score against, so the states and their
realized returns have to be produced here. `V_t(s)` is the critic head of
the shared actor-critic trunk (`TilePolicyNetwork.forward_ac`), loaded
through the same `build_tile_policy_from_checkpoint` dispatch the eval and
demo scripts use. `R` is a MONTE-CARLO discounted return to episode end,
computed over the SAME reward stream the trainer feeds GAE:

    R_t   = r_t + gamma * R_{t+1},  R_T = r_T at a natural terminal

and states whose episode does not terminate inside the rollout are DROPPED
rather than bootstrapped. That is deliberate and registered: MC targets
carry no value-function bootstrap, so the frozen batch cannot flatter the
critic that collected it. (`src/training/gae.py` is therefore NOT reused —
its `value_targets` are `advantages + values_pred`, i.e. the critic's own
opinion folded into its own target, which would make EV partly circular.
The live `vanilla_ppo_explained_variance` scalar in `ppo_updater.py` is
built on exactly those GAE targets, so it is a DIFFERENT quantity from
this one and cannot substitute for it at the F1 gate.)

`r_t` is the RAW extrinsic reward, NOT symlog-compressed. `symlog` is
applied in `trainer._reinforce_update` (the GA path, `trainer.py:4505`)
and NOWHERE in the vanilla_ppo path or `PPOUpdater` — so a config's
`reinforce.symlog_rewards: true` is inert under `trainer_mode:
vanilla_ppo`, and symlogging here would put `R` on a scale the critic was
never fit to (measured: mean V ~801 against symlogged returns ~128, an
EV of -0.21 that is entirely an artifact of the transform).

The rollout that produces those states reproduces the TRAINING collection
path, not the eval path, because the training distribution is the one the
critic was fit on:

* start states are drawn by `backward_curriculum.draw_restart` over the
  window behind the cursor `tau` that the checkpoint itself recorded
  (`checkpoint["backward_curriculum"]["tau"]`), with the true entrance in
  the pool — the same draw `trainer.py` makes at every restart;
* actions are SAMPLED from the policy's own softmax (the trainer samples;
  greedy is an eval-only convention), under the config's
  `sticky_action_prob` with `sticky_episode_boundary_reset` honored via
  the trainer's own `StickyBoundary`;
* an episode that ends mid-rollout restarts IN PLACE and the stacker
  re-seed is deferred to the next step's post-restore RAM — the trainer's
  exact behavior, quirk included (see `_stage0_reseed` in trainer.py);
* `done = pool_done or reward_done`, matching `trainer.py:7181`.

RND
---
`reinforce.rnd_intrinsic_coef` folds a novelty bonus into the reward
stream before GAE (`trainer.py:4604`), so the critic's true regression
target is `symlog(extrinsic) + coef * normalized_rnd_error`. Both readings
are emitted:

* `ev_*` (PRIMARY, gating) uses the extrinsic-only symlog stream. The
  intrinsic term is a property of a network that is itself still training,
  so folding it in makes the "fixed" target set depend on which
  checkpoint's RND computed it.
* `ev_*_with_intrinsic` (secondary) folds in the bonus computed from the
  COLLECTING checkpoint's frozen `rnd_state_dict`, which is the closest
  faithful reproduction of the trainer's target.

Their biases point in opposite directions (excluding the bonus flatters
late checkpoints, whose novelty has decayed; including a frozen early
bonus penalizes them), so a Δ that agrees across both is a Δ that is not
an artifact of this choice. Disagreement is itself a finding and must be
escalated, not averaged.

USAGE
-----
Per run (the registered F1 grid, P from F0's `ladder.csv`):

    .venv/bin/python scripts/critic_explained_variance.py \\
      --profile configs/mario_1_1_v28_seed3.yaml --peak-iter 120 \\
      --out-dir runs/v29_stability/f1_explained_variance

Any single archived checkpoint (no F0 peak needed):

    .venv/bin/python scripts/critic_explained_variance.py \\
      --profile configs/mario_1_1_v28_seed3.yaml --iters 100,120 \\
      --reference-iter 120 --out-dir /tmp/f1_smoke

The pre-registered three-way call, once all 8 runs are on disk:

    .venv/bin/python scripts/critic_explained_variance.py \\
      --verdict runs/v29_stability/f1_explained_variance

Receipts: one `<run>_it<NNN>.json` per checkpoint, one `f1_<run>.json` per
run carrying the grid table and `delta_fixed`, one `<run>_reference_it<NNN>.npz`
frozen batch, and `f1_verdict.json` from the verdict pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.models.tile_policy import (  # noqa: E402
    build_tile_policy_from_checkpoint, checkpoint_is_recurrent,
)
from src.training import backward_curriculum as bwd  # noqa: E402
from src.training.go_explore import start_window  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, derive_checkpoint_dir, resolve_encoder,
)
from src.training.trainer import StickyBoundary  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402

INSTRUMENT = "critic_explained_variance"
SCHEMA_VERSION = 1

# The registered grid: {P-20, P-10, P, P+10, P+20, P+40, P+60} n {available}.
DEFAULT_GRID_OFFSETS: tuple[int, ...] = (-20, -10, 0, 10, 20, 40, 60)
# 16,384 states subsampled uniformly from one rollout_steps x num_envs rollout.
DEFAULT_BATCH_SIZE = 16384
# Pre-registered three-way kill rule (V29 §F1).
LIVE_DELTA = -0.15
DEAD_DELTA = -0.10
REGISTERED_RUNS = 8
REGISTERED_MAJORITY = 5


# ======================================================================
# Pure quantities — no emulator, no torch, unit-testable in isolation.
# ======================================================================


def explained_variance(
    returns: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> Optional[float]:
    """``EV = 1 - Var(R - V) / Var(R)``.

    Returns None when Var(R) == 0, where EV is undefined (any constant
    predictor would score -inf or 1 depending on rounding). Both variances
    are population variances over the SAME sample, so the ddof convention
    cancels in the ratio.

    Reference points the callers rely on: a perfect predictor scores 1.0,
    a predictor pinned at the mean of R scores 0.0, and a predictor
    anti-correlated with R scores below 0 (V = -R gives -3.0). EV is
    unbounded below on purpose — "worse than predicting the mean" is a
    real and reportable state for a critic.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if r.shape != v.shape:
        raise ValueError(
            f"returns and values must have the same length, got "
            f"{r.shape[0]} and {v.shape[0]}"
        )
    if r.size == 0:
        raise ValueError("explained_variance needs at least one sample")
    var_r = float(r.var())
    if not np.isfinite(var_r) or var_r <= 0.0:
        return None
    return float(1.0 - (float((r - v).var()) / var_r))


def mc_returns(
    rewards: np.ndarray,
    terminal: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo discounted returns to episode end, plus a validity mask.

    `rewards` and `terminal` are `(T,)` or `(T, num_envs)` and share a
    shape; `terminal[t]` is True on the step where the episode ENDED (that
    step's reward belongs to the finished episode, nothing after it does).

    Returns `(R, valid)`. `R[t] = rewards[t] + gamma * R[t+1]`, restarting
    the accumulator at every terminal. `valid[t]` is False when no terminal
    follows `t` inside the buffer — those states are censored, not
    bootstrapped, and callers MUST drop them. Bootstrapping them with
    `V(s_T)` is exactly the circularity this instrument exists to avoid.
    """
    r = np.asarray(rewards, dtype=np.float64)
    term = np.asarray(terminal, dtype=bool)
    if r.shape != term.shape:
        raise ValueError(
            f"rewards {r.shape} and terminal {term.shape} must match"
        )
    if r.ndim not in (1, 2):
        raise ValueError(f"rewards must be 1-D or 2-D, got {r.ndim}-D")
    squeeze = r.ndim == 1
    if squeeze:
        r = r[:, None]
        term = term[:, None]
    n_steps, n_envs = r.shape
    out = np.zeros_like(r)
    valid = np.zeros(r.shape, dtype=bool)
    running = np.zeros(n_envs, dtype=np.float64)
    running_valid = np.zeros(n_envs, dtype=bool)
    for t in range(n_steps - 1, -1, -1):
        term_t = term[t]
        running = np.where(term_t, 0.0, running)
        running_valid = np.where(term_t, True, running_valid)
        running = r[t] + float(gamma) * running
        out[t] = running
        valid[t] = running_valid
    if squeeze:
        return out[:, 0], valid[:, 0]
    return out, valid


def subsample_indices(n_available: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """`k` indices drawn uniformly WITHOUT replacement from `range(n_available)`,
    sorted. Returns everything (sorted) when `k >= n_available` — a batch
    smaller than the target is reported, never silently padded."""
    if n_available <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n_available:
        return np.arange(n_available, dtype=np.int64)
    return np.sort(rng.choice(n_available, size=int(k), replace=False)).astype(np.int64)


def classify_delta(delta: float) -> str:
    """One run's `Δ_fixed` against the registered bands: 'live'
    (`<= -0.15`), 'dead' (`> -0.10`), or 'band' (the -0.15..-0.10 gap)."""
    if delta <= LIVE_DELTA:
        return "live"
    if delta > DEAD_DELTA:
        return "dead"
    return "band"


def f1_verdict(
    deltas: dict[str, float],
    *,
    expected_runs: int = REGISTERED_RUNS,
    majority: int = REGISTERED_MAJORITY,
) -> dict:
    """The pre-registered three-way kill rule (V29 §F1).

    LIVE when `Δ_fixed <= -0.15` in `majority` of `expected_runs` runs;
    DEAD when `Δ_fixed > -0.10` in `majority` of them; INCONCLUSIVE for a
    split field or a majority sitting in the -0.15..-0.10 band.

    A field with fewer than `expected_runs` runs is INCONCLUSIVE by
    construction: "an ambiguous preflight may not be read as a green
    light," and a partial field is ambiguous no matter which way the runs
    it does contain lean.
    """
    per_run = {name: classify_delta(float(d)) for name, d in deltas.items()}
    n_live = sum(1 for v in per_run.values() if v == "live")
    n_dead = sum(1 for v in per_run.values() if v == "dead")
    n_band = sum(1 for v in per_run.values() if v == "band")
    if len(deltas) < expected_runs:
        verdict = "INCONCLUSIVE"
        reason = (
            f"only {len(deltas)} of {expected_runs} registered runs measured; "
            f"a partial field is not a launch"
        )
    elif n_live >= majority:
        verdict = "LIVE"
        reason = (
            f"{n_live}/{len(deltas)} runs at delta_fixed <= {LIVE_DELTA}: "
            f"critic fit degrades on a fixed target set. Proceed to F2."
        )
    elif n_dead >= majority:
        verdict = "DEAD"
        reason = (
            f"{n_dead}/{len(deltas)} runs at delta_fixed > {DEAD_DELTA}: "
            f"critic fit HELD while behavior collapsed. The value-loss rise "
            f"is target noise. Rank 1 is refuted; V29 does not launch."
        )
    else:
        verdict = "INCONCLUSIVE"
        reason = (
            f"split field (live={n_live}, band={n_band}, dead={n_dead}); "
            f"escalate to the owner with the per-run table. Not a launch."
        )
    return {
        "instrument": INSTRUMENT,
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "reason": reason,
        "rule": {
            "live": f"delta_fixed <= {LIVE_DELTA} in >= {majority} of {expected_runs}",
            "dead": f"delta_fixed > {DEAD_DELTA} in >= {majority} of {expected_runs}",
            "inconclusive": "anything else",
        },
        "counts": {"live": n_live, "band": n_band, "dead": n_dead,
                   "runs_measured": len(deltas)},
        "per_run": {name: {"delta_fixed": float(deltas[name]), "band": per_run[name]}
                    for name in sorted(deltas)},
    }


# ======================================================================
# Run spec — everything the rollout needs, read from the profile.
# ======================================================================


@dataclass
class RunSpec:
    profile_path: Path
    profile: dict
    run_name: str
    rom_path: str
    ckpt_dir: Path
    start_state_path: Path
    start_bytes: bytes
    bitmasks: Any
    extractor: Any
    feature_dim: int
    stacked_dim: int
    stack_size: int
    gamma: float
    symlog_rewards: bool
    sticky_prob: float
    sticky_boundary_reset: bool
    rnd_intrinsic_coef: float
    rollout_steps: int
    num_envs: int
    max_episode_steps: int
    bwd_states_dir: Optional[str]
    bwd_window_frames: int
    bwd_entrance_weight: float
    bwd_pin_entrance: bool

    @staticmethod
    def from_profile(profile_path: Path) -> "RunSpec":
        with open(profile_path) as fh:
            profile = yaml.safe_load(fh)
        rl = profile.get("reinforce", {}) or {}
        if rl.get("trainer_mode") != "vanilla_ppo":
            raise ValueError(
                f"{profile_path}: reinforce.trainer_mode is "
                f"{rl.get('trainer_mode')!r}; this instrument reproduces the "
                f"vanilla_ppo collection path only."
            )
        extractor, feature_dim, stacked_dim = resolve_encoder(profile)
        stack_size = stacked_dim // feature_dim
        rom_path = profile.get("rom_path")
        if not rom_path or not Path(rom_path).exists():
            raise FileNotFoundError(f"{profile_path}: rom_path {rom_path!r} not found")
        start_state = profile.get("start_state_path")
        if not start_state or not Path(start_state).exists():
            raise FileNotFoundError(
                f"{profile_path}: start_state_path {start_state!r} not found — the "
                f"entrance blob is the restart pool's anchor and cannot be defaulted."
            )
        bwd_cfg = dict(rl.get("backward_curriculum", {}) or {})
        return RunSpec(
            profile_path=profile_path,
            profile=profile,
            run_name=derive_checkpoint_dir("./checkpoints", profile.get("name")).name,
            rom_path=str(rom_path),
            ckpt_dir=derive_checkpoint_dir("./checkpoints", profile.get("name")),
            start_state_path=Path(start_state),
            start_bytes=Path(start_state).read_bytes(),
            bitmasks=action_space_to_bitmasks(profile["action_space"]),
            extractor=extractor,
            feature_dim=int(feature_dim),
            stacked_dim=int(stacked_dim),
            stack_size=int(stack_size),
            gamma=float(rl.get("gamma", 0.99)),
            symlog_rewards=bool(rl.get("symlog_rewards", False)),
            sticky_prob=float(rl.get("sticky_action_prob", 0.0)),
            sticky_boundary_reset=bool(rl.get("sticky_episode_boundary_reset", False)),
            rnd_intrinsic_coef=float(rl.get("rnd_intrinsic_coef", 0.0)),
            rollout_steps=int(rl.get("rollout_steps", 512)),
            num_envs=int(rl.get("num_envs", 1)),
            max_episode_steps=int(profile.get("max_episode_steps", 2400)),
            bwd_states_dir=(bwd_cfg.get("states_dir")
                            if bwd_cfg.get("enabled", True) else None),
            bwd_window_frames=int(bwd_cfg.get("window_frames", 160)),
            bwd_entrance_weight=float(bwd_cfg.get("entrance_weight", 1.0)),
            bwd_pin_entrance=bool(bwd_cfg.get("pin_entrance", False)),
        )


def available_iters(ckpt_dir: Path) -> list[int]:
    """Sorted iteration numbers of the archived `vanilla_ppo_iter_*.pt`."""
    out = []
    for p in ckpt_dir.glob("vanilla_ppo_iter_*.pt"):
        stem = p.stem.rsplit("_", 1)[-1]
        try:
            out.append(int(stem))
        except ValueError:
            continue
    return sorted(out)


def checkpoint_path(ckpt_dir: Path, it: int) -> Path:
    return ckpt_dir / f"vanilla_ppo_iter_{it:05d}.pt"


def load_policy(ckpt: dict, spec: RunSpec):
    """The real network-loading path — the same `build_tile_policy_from_checkpoint`
    dispatch `scripts/eval_game.py` uses, so a recurrent checkpoint cannot
    silently misload into the stateless net."""
    if checkpoint_is_recurrent(ckpt):
        raise ValueError(
            "this checkpoint is recurrent (tile_gru); its critic must be scored "
            "with hidden state threaded through the trajectory. The V27/V28 "
            "cohort F1 gates are stateless tile MLPs — refusing to score a "
            "recurrent critic through the stateless forward, which would "
            "silently zero the GRU state at every sample."
        )
    net, is_recurrent = build_tile_policy_from_checkpoint(
        ckpt, num_actions=len(spec.bitmasks), feature_dim=spec.stacked_dim,
    )
    assert not is_recurrent
    net.eval()
    return net


def load_rnd(ckpt: dict, spec: RunSpec):
    """The checkpoint's frozen RND, or None. `rnd_state_dict` round-trips the
    target, the predictor, `obs_rms` AND `reward_rms`, so the intrinsic bonus
    a checkpoint produced is exactly reproducible offline — no normalization
    state is updated here."""
    sd = ckpt.get("rnd_state_dict")
    if not sd or spec.rnd_intrinsic_coef <= 0.0:
        return None
    from src.models.tile_rnd import TileRND
    rnd = TileRND(feature_dim=spec.stacked_dim)
    rnd.load_state_dict(sd)
    rnd.eval()
    return rnd


def checkpoint_tau(ckpt: dict) -> Optional[int]:
    """The backward-curriculum cursor the checkpoint recorded, or None."""
    state = ckpt.get("backward_curriculum") or {}
    tau = state.get("tau")
    return int(tau) if tau is not None else None


# ======================================================================
# Forward-only reproduction of the trainer's collection path.
# ======================================================================


@dataclass
class Rollout:
    """One forward-only rollout, shaped exactly like the trainer's buffers."""
    obs: np.ndarray          # (T, N, stacked_dim) int8
    rewards: np.ndarray      # (T, N) float32 — raw extrinsic, pre-symlog
    terminal: np.ndarray     # (T, N) bool — episode ended on this step
    restart_age: np.ndarray  # (T, N) uint16 — steps since this env restarted
    n_episodes: int
    n_clears: int
    tau: int
    seconds: float


class TrainingRollout:
    """Emulator-backed reproduction of `trainer.py`'s vanilla_ppo collection.

    Owns the pool for its lifetime; call `close()`. Forward-only: no
    gradients, no optimizer, no RND normalization updates, nothing written
    back to any checkpoint.
    """

    def __init__(self, spec: RunSpec, num_envs: int, seed: int) -> None:
        from nes_core import Pool
        self.spec = spec
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.pool = Pool(
            rom_path=spec.rom_path, num_workers=self.num_envs, frame_skip=4,
            start_state_path=str(spec.start_state_path),
        )
        self.reward_fns = [build_reward_function(spec.profile)
                           for _ in range(self.num_envs)]
        self.stackers = [
            TileFeatureStacker(stack_size=spec.stack_size,
                               feature_dim=spec.feature_dim)
            for _ in range(self.num_envs)
        ]
        self.blobs: list[bytes] = []
        self.tape: list[int] = []
        self.frames_per_entry = 4
        if spec.bwd_states_dir:
            entries, meta = bwd.load_index(spec.bwd_states_dir)
            self.blobs = bwd.load_blobs(spec.bwd_states_dir, entries)
            self.tape = list(range(len(entries)))
            self.frames_per_entry = max(1, int(meta.get("every_frames", 4)))

    def close(self) -> None:
        try:
            self.pool.shutdown()
        except Exception:
            pass

    def _draw_restart(self, tau: int, rand: float) -> bytes:
        """The trainer's restart draw: uniform over the window behind `tau`
        with the true entrance carrying `entrance_weight` entries' worth of
        mass (`trainer.py:9617`, `backward_curriculum.draw_restart`)."""
        spec = self.spec
        if not self.tape or spec.bwd_pin_entrance:
            return spec.start_bytes
        window = start_window(
            self.tape, tau, window_frames=spec.bwd_window_frames,
            frames_per_step=self.frames_per_entry,
        )
        pick = bwd.draw_restart(len(window), spec.bwd_entrance_weight, rand)
        if pick == bwd.ENTRANCE:
            return spec.start_bytes
        return self.blobs[window[pick]]

    def collect(self, net, tau: int, n_steps: int) -> Rollout:
        spec = self.spec
        n_envs = self.num_envs
        rng = np.random.default_rng(self.seed)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.seed))

        obs_buf = np.zeros((n_steps, n_envs, spec.stacked_dim), dtype=np.int8)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        term_buf = np.zeros((n_steps, n_envs), dtype=bool)
        age_buf = np.zeros((n_steps, n_envs), dtype=np.uint16)

        boundary = StickyBoundary(n_envs, spec.sticky_boundary_reset)
        sticky_p_env = np.full(n_envs, spec.sticky_prob, dtype=np.float64)
        prev_exec = np.zeros(n_envs, dtype=np.int64)

        # Iter-boundary warm start: reset, draw every env's restart, then
        # flush the post-restore frame with one no-op step — the pattern
        # trainer.py:9654 uses, because load_worker_state restores RAM but
        # produces no frame until the next step.
        self.pool.reset_all()
        for i in range(n_envs):
            self.pool.load_worker_state(i, self._draw_restart(tau, float(rng.random())))
            boundary.mark_restart(i, prev_exec)
            self.reward_fns[i].reset()
        init = self.pool.step_all(np.zeros(n_envs, dtype=np.uint8))
        stacked = np.zeros((n_envs, spec.stacked_dim), dtype=np.int8)
        for i in range(n_envs):
            stacked[i] = self.stackers[i].reset(spec.extractor.extract(init[i][2]))

        age = np.zeros(n_envs, dtype=np.int64)
        n_episodes = 0
        n_clears = 0
        t0 = time.perf_counter()
        for t in range(n_steps):
            obs_buf[t] = stacked
            age_buf[t] = np.minimum(age, np.iinfo(np.uint16).max)
            with torch.no_grad():
                logits, _ = net.forward_ac(torch.from_numpy(stacked).float())
                probs = torch.softmax(logits.float(), dim=-1)
                actions = torch.multinomial(
                    probs, num_samples=1, generator=gen,
                ).squeeze(-1).numpy().astype(np.int64)
            # Sticky-action training (Machado et al. 2018) with the trainer's
            # own boundary guard: the roll is gated on `t > 0` and suppressed
            # for one step on any env that just restarted (trainer.py:6904).
            # The uniforms come from THIS instrument's own generator, not the
            # global numpy stream the trainer draws from — same distribution,
            # and nothing here can perturb a concurrent job's RNG.
            if spec.sticky_prob > 0.0:
                if t > 0:
                    rows = boundary.override_rows(sticky_p_env, rng.random(n_envs))
                    actions[rows] = prev_exec[rows]
                prev_exec = actions
                boundary.consume()
            else:
                prev_exec = actions
            step_actions = np.array(
                [spec.bitmasks[a] for a in actions], dtype=np.uint8,
            )
            results = self.pool.step_all(step_actions)

            for i in range(n_envs):
                ram = results[i][2]
                reward, rew_done, _ = self.reward_fns[i].compute(
                    ram, action=int(step_actions[i]),
                )
                rew_buf[t, i] = float(reward)
                done = bool(results[i][3]) or bool(rew_done)
                term_buf[t, i] = done
                if done:
                    n_episodes += 1
                    if self.reward_fns[i].episode_success():
                        n_clears += 1
                    # AUTO-RESET in place: draw a fresh restart, suppress the
                    # sticky carry, reset the reward fn — then RE-SEED the
                    # stacker from THIS step's RAM. That last part looks
                    # wrong and is deliberate: trainer.py sets
                    # `_stage0_reseed[i]` at :7663 and consumes it at :7694
                    # inside the SAME per-env iteration, so the fresh
                    # episode's stack is seeded with the pre-restore frame
                    # and flushes over the next `stack_size` steps. The
                    # critic was fit on exactly those observations, so
                    # reproducing the quirk is the faithful choice; the
                    # `restart_age` buffer records how far each state is
                    # from its restart so a consumer can exclude them.
                    self.pool.load_worker_state(
                        i, self._draw_restart(tau, float(rng.random())),
                    )
                    boundary.mark_restart(i, prev_exec)
                    self.reward_fns[i].reset()
                    stacked[i] = self.stackers[i].reset(spec.extractor.extract(ram))
                    age[i] = 0
                else:
                    stacked[i] = self.stackers[i].push(spec.extractor.extract(ram))
                    age[i] += 1
        seconds = time.perf_counter() - t0

        return Rollout(
            obs=obs_buf, rewards=rew_buf, terminal=term_buf,
            restart_age=age_buf, n_episodes=n_episodes, n_clears=n_clears,
            tau=int(tau), seconds=float(seconds),
        )


# ======================================================================
# Batch construction + scoring.
# ======================================================================


def build_batch(
    roll: Rollout,
    spec: RunSpec,
    *,
    rnd,
    batch_size: int,
    rng: np.random.Generator,
    min_restart_age: int = 0,
) -> dict:
    """Flatten a rollout into (obs, R_extrinsic, R_with_intrinsic) over the
    states whose episode terminated inside the buffer, subsampled uniformly
    to `batch_size`.

    Censoring is the only filter applied by default: a state whose episode
    ran past the end of the buffer has no realized return and is dropped
    (never bootstrapped). `min_restart_age > 0` additionally drops the
    first N states after each restart, where the stacker still carries
    pre-restore frames — off by default, because the critic was fit on
    those states too and excluding them silently changes the population
    the EV describes.
    """
    # RAW extrinsic rewards. See the module docstring: symlog lives only in
    # the GA path, so the vanilla_ppo critic was fit on this scale.
    rewards = roll.rewards.astype(np.float64)
    n_steps, n_envs, dim = roll.obs.shape
    flat_obs = roll.obs.reshape(n_steps * n_envs, dim)

    bonus = None
    if rnd is not None:
        bonus_flat = intrinsic_bonus(rnd, flat_obs, spec.rnd_intrinsic_coef)
        # Zeroed on terminal steps, exactly as
        # `ppo.fold_intrinsic_into_rewards` does before GAE.
        bonus = bonus_flat.reshape(n_steps, n_envs) * (~roll.terminal)

    ret_ext, valid = mc_returns(rewards, roll.terminal, spec.gamma)
    ret_int = (
        mc_returns(rewards + bonus, roll.terminal, spec.gamma)[0]
        if bonus is not None else None
    )

    age_flat = roll.restart_age.reshape(-1)
    keep = valid.reshape(-1)
    if min_restart_age > 0:
        keep = keep & (age_flat >= int(min_restart_age))
    idx_valid = np.flatnonzero(keep)
    if idx_valid.size == 0:
        raise RuntimeError(
            f"no state in this rollout has a realized return: "
            f"{roll.n_episodes} episodes finished in "
            f"{n_steps} steps x {n_envs} envs, so every state is censored by "
            f"the rollout boundary. Raise --rollout-steps (the registered "
            f"value is the profile's reinforce.rollout_steps) — an EV over "
            f"zero samples is not a measurement."
        )
    pick = subsample_indices(idx_valid.size, batch_size, rng)
    idx = idx_valid[pick]
    return {
        "obs": np.ascontiguousarray(flat_obs[idx]),
        "returns": ret_ext.reshape(-1)[idx].astype(np.float32),
        "returns_with_intrinsic": (
            ret_int.reshape(-1)[idx].astype(np.float32)
            if ret_int is not None else None
        ),
        "restart_age": age_flat[idx],
        "n_valid": int(idx_valid.size),
        "n_total": int(keep.size),
        "n_sampled": int(idx.size),
        "n_uncensored": int(valid.sum()),
    }


def intrinsic_bonus(rnd, obs: np.ndarray, coef: float, chunk: int = 4096) -> np.ndarray:
    """`normalize_bonus(rnd(obs)) * coef` — the trainer's per-step novelty
    bonus (`trainer.py:4579`) computed from a FROZEN RND: no
    `update_normalization` call, so `reward_rms` stays exactly where the
    checkpoint left it."""
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), chunk):
            x = torch.from_numpy(np.ascontiguousarray(obs[i:i + chunk])).float()
            err = rnd(x)
            out.append((rnd.normalize_bonus(err) * float(coef)).cpu().numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def critic_values(net, obs: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """`V(s)` from the critic head of the shared trunk, in eval mode, no grad.
    Tile features are int8 and are cast to float WITHOUT a /255 divide —
    the trainer's tile branch (`trainer.py:4532`) does the same."""
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), chunk):
            x = torch.from_numpy(np.ascontiguousarray(obs[i:i + chunk])).float()
            _, v = net.forward_ac(x)
            out.append(v.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def score(returns: np.ndarray, values: np.ndarray) -> dict:
    ev = explained_variance(returns, values)
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    return {
        "ev": ev,
        "n": int(r.size),
        "var_returns": float(r.var()),
        "var_residual": float((r - v).var()),
        "mean_returns": float(r.mean()),
        "mean_values": float(v.mean()),
        "std_returns": float(r.std()),
        "std_values": float(v.std()),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_reference(path: Path, batch: dict, meta: dict) -> None:
    payload = {
        "obs": batch["obs"],
        "returns": batch["returns"],
        "restart_age": batch["restart_age"],
        "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    if batch.get("returns_with_intrinsic") is not None:
        payload["returns_with_intrinsic"] = batch["returns_with_intrinsic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_reference(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    batch = {
        "obs": z["obs"],
        "returns": z["returns"],
        "restart_age": z["restart_age"],
        "returns_with_intrinsic": (
            z["returns_with_intrinsic"] if "returns_with_intrinsic" in z.files else None
        ),
        "n_valid": int(meta.get("n_valid", len(z["returns"]))),
        "n_total": int(meta.get("n_total", len(z["returns"]))),
        "n_sampled": int(len(z["returns"])),
    }
    return batch, meta


# Harness fields that determine the CONTENT of a frozen reference batch.
# Prose keys (symlog_note, rng, restart_distribution) and `torch_threads`
# are excluded: the first three are documentation, and thread count does
# not change which states or returns land in the batch.
REFERENCE_BATCH_KEYS: tuple[str, ...] = (
    "num_envs", "rollout_steps", "batch_size", "gamma", "symlog_applied",
    "sticky_prob", "sticky_boundary_reset", "rnd_intrinsic_coef",
    "action_select", "target", "unterminated_states", "min_restart_age",
    "seed",
)


def reference_mismatches(
    ref_meta: dict, harness: dict, run_name: str, ref_iter: int,
) -> list[str]:
    """Fields where a cached frozen batch disagrees with this invocation.

    The whole point of `EV_fixed` is that every grid point is scored on
    the SAME states and the SAME targets, so the cached `.npz` is the
    gating quantity's ground truth. But the cache key is only
    `(run_name, ref_iter)` — rerun F1 into a populated `--out-dir` with a
    different `--num-envs`/`--rollout-steps`/`--batch-size`/`--seed` and
    the old batch is silently reused while the emitted `harness` block
    describes the NEW settings, i.e. a receipt that misdescribes its own
    evidence. It also breaks the `EV_onpolicy(P) == EV_fixed(P)`
    self-check, since the on-policy rollout would follow the new shape
    and the frozen batch the old one. Observed in practice; this is the
    guard.
    """
    out: list[str] = []
    if ref_meta.get("run") not in (None, run_name):
        out.append(f"run: cached {ref_meta.get('run')!r} != {run_name!r}")
    if int(ref_meta.get("iter", ref_iter)) != int(ref_iter):
        out.append(f"iter: cached {ref_meta.get('iter')} != {ref_iter}")
    cached = ref_meta.get("harness") or {}
    for key in REFERENCE_BATCH_KEYS:
        if key not in cached or key not in harness:
            continue
        if cached[key] != harness[key]:
            out.append(f"{key}: cached {cached[key]!r} != {harness[key]!r}")
    return out


# ======================================================================
# Driver
# ======================================================================


def resolve_grid(
    spec: RunSpec,
    peak_iter: Optional[int],
    offsets: Sequence[int],
    explicit: Optional[Sequence[int]],
) -> list[int]:
    have = set(available_iters(spec.ckpt_dir))
    if explicit:
        missing = [i for i in explicit if i not in have]
        if missing:
            raise FileNotFoundError(
                f"{spec.ckpt_dir}: no archived checkpoint for iters {missing}"
            )
        return sorted(set(int(i) for i in explicit))
    if peak_iter is None:
        raise ValueError("one of --peak-iter or --iters is required")
    grid = sorted({int(peak_iter) + int(o) for o in offsets} & have)
    if int(peak_iter) not in grid:
        raise FileNotFoundError(
            f"{spec.ckpt_dir}: the peak checkpoint iter {peak_iter} is not on "
            f"disk; delta_fixed is defined relative to it and cannot be formed."
        )
    return grid


def run_one(args: argparse.Namespace) -> int:
    spec = RunSpec.from_profile(Path(args.profile))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_envs = int(args.num_envs or spec.num_envs)
    n_steps = int(args.rollout_steps or spec.rollout_steps)
    ref_iter = int(args.reference_iter if args.reference_iter is not None
                   else args.peak_iter)
    grid = resolve_grid(spec, args.peak_iter, args.grid_offsets, args.iters)

    harness = {
        "num_envs": n_envs,
        "rollout_steps": n_steps,
        "batch_size": int(args.batch_size),
        "gamma": spec.gamma,
        "symlog_rewards_config": spec.symlog_rewards,
        "symlog_applied": False,
        "symlog_note": (
            "reinforce.symlog_rewards is inert under trainer_mode: vanilla_ppo "
            "— symlog is applied only in trainer._reinforce_update (the GA "
            "path, trainer.py:4505), never in PPOUpdater. R is on the raw "
            "reward scale the critic was fit to."
        ),
        "sticky_prob": spec.sticky_prob,
        "sticky_boundary_reset": spec.sticky_boundary_reset,
        "rnd_intrinsic_coef": spec.rnd_intrinsic_coef,
        "action_select": "sampled",
        "target": "monte_carlo_discounted_return_to_episode_end",
        "unterminated_states": "dropped",
        "min_restart_age": int(args.min_restart_age),
        "restart_distribution": (
            "backward_curriculum.draw_restart at the checkpoint's recorded tau"
            if spec.bwd_states_dir else "profile start_state_path"
        ),
        "seed": int(args.seed),
        "torch_threads": int(args.torch_threads),
        "rng": (
            "common random numbers: every grid point's on-policy rollout is "
            "re-seeded from --seed, so two checkpoints see the same restart "
            "draws and the same sticky rolls and differ only in their own "
            "action distribution"
        ),
    }

    # ---- the frozen reference batch (collected ONCE, from P) ----------
    ref_path = Path(args.reference_batch) if args.reference_batch else (
        out_dir / f"{spec.run_name}_reference_it{ref_iter:05d}.npz"
    )
    if ref_path.exists():
        ref_batch, ref_meta = load_reference(ref_path)
        bad = reference_mismatches(ref_meta, harness, spec.run_name, ref_iter)
        if bad:
            detail = "\n  ".join(bad)
            if args.reference_batch:
                # Explicitly named by the caller: they asked for THIS
                # file, so honor it — but the receipt must not claim the
                # batch was collected under the current settings.
                print(f"[f1] WARNING: --reference-batch {ref_path} was "
                      f"collected under different settings:\n  {detail}\n"
                      f"[f1] proceeding as instructed; EV_fixed is scored "
                      f"on THAT batch, and the mismatch is recorded under "
                      f"reference_batch.mismatch in the emitted JSON.",
                      flush=True)
            else:
                raise SystemExit(
                    f"[f1] refusing to reuse the cached frozen batch at "
                    f"{ref_path}: it was collected under different "
                    f"settings, so EV_fixed would be scored on states "
                    f"this invocation does not describe:\n  {detail}\n"
                    f"Delete it to recollect, or pass it explicitly with "
                    f"--reference-batch to reuse it anyway."
                )
        print(f"[f1] reusing frozen reference batch {ref_path} "
              f"(n={ref_batch['n_sampled']}, from iter {ref_meta.get('iter')})",
              flush=True)
    else:
        ref_ckpt_path = checkpoint_path(spec.ckpt_dir, ref_iter)
        ref_ckpt = torch.load(str(ref_ckpt_path), map_location="cpu", weights_only=False)
        ref_net = load_policy(ref_ckpt, spec)
        ref_rnd = load_rnd(ref_ckpt, spec)
        ref_tau = checkpoint_tau(ref_ckpt)
        if ref_tau is None:
            ref_tau = 0
        print(f"[f1] collecting frozen reference batch from iter {ref_iter} "
              f"(tau={ref_tau}, {n_envs} envs x {n_steps} steps)", flush=True)
        roller = TrainingRollout(spec, n_envs, seed=int(args.seed))
        try:
            roll = roller.collect(ref_net, ref_tau, n_steps)
        finally:
            roller.close()
        ref_batch = build_batch(
            roll, spec, rnd=ref_rnd, batch_size=int(args.batch_size),
            rng=np.random.default_rng(int(args.seed) + 1),
            min_restart_age=int(args.min_restart_age),
        )
        ref_meta = {
            "instrument": INSTRUMENT,
            "schema_version": SCHEMA_VERSION,
            "run": spec.run_name,
            "profile": str(spec.profile_path),
            "iter": ref_iter,
            "checkpoint": str(ref_ckpt_path),
            "tau": ref_tau,
            "n_valid": ref_batch["n_valid"],
            "n_uncensored": ref_batch["n_uncensored"],
            "n_total": ref_batch["n_total"],
            "n_sampled": ref_batch["n_sampled"],
            "n_episodes": roll.n_episodes,
            "n_clears": roll.n_clears,
            "min_restart_age": int(args.min_restart_age),
            "median_restart_age": float(np.median(ref_batch["restart_age"])),
            "rollout_seconds": roll.seconds,
            "with_intrinsic": ref_batch.get("returns_with_intrinsic") is not None,
            "harness": harness,
        }
        save_reference(ref_path, ref_batch, ref_meta)
        print(f"[f1] frozen batch written: {ref_path} "
              f"(n={ref_batch['n_sampled']} of {ref_batch['n_valid']} valid / "
              f"{ref_batch['n_total']} collected, {roll.n_episodes} episodes, "
              f"{roll.seconds:.1f}s)", flush=True)

    ref_fingerprint = {
        "path": str(ref_path),
        "sha256": sha256_file(ref_path),
        "iter": int(ref_meta.get("iter", ref_iter)),
        "n": int(ref_batch["n_sampled"]),
    }
    # Only ever non-empty on the explicit --reference-batch escape hatch
    # (the auto-discovered path hard-fails above). Carried into the
    # receipt so a reader can see EV_fixed was scored on a batch the
    # top-level `harness` block does not describe.
    _ref_bad = reference_mismatches(ref_meta, harness, spec.run_name, ref_iter)
    if _ref_bad:
        ref_fingerprint["mismatch"] = _ref_bad
        ref_fingerprint["collected_under"] = ref_meta.get("harness")

    # ---- score the grid ----------------------------------------------
    rows: list[dict] = []
    roller: Optional[TrainingRollout] = None
    try:
        for it in grid:
            t0 = time.perf_counter()
            ck_path = checkpoint_path(spec.ckpt_dir, it)
            ckpt = torch.load(str(ck_path), map_location="cpu", weights_only=False)
            net = load_policy(ckpt, spec)
            tau = checkpoint_tau(ckpt)
            if tau is None:
                tau = 0

            v_fixed = critic_values(net, ref_batch["obs"])
            fixed = score(ref_batch["returns"], v_fixed)
            fixed_i = (
                score(ref_batch["returns_with_intrinsic"], v_fixed)
                if ref_batch.get("returns_with_intrinsic") is not None else None
            )

            onp = onp_i = None
            onp_meta: dict = {}
            if not args.no_onpolicy:
                if roller is None:
                    roller = TrainingRollout(spec, n_envs, seed=int(args.seed))
                roll = roller.collect(net, tau, n_steps)
                batch = build_batch(
                    roll, spec, rnd=load_rnd(ckpt, spec),
                    batch_size=int(args.batch_size),
                    rng=np.random.default_rng(int(args.seed) + 1),
                    min_restart_age=int(args.min_restart_age),
                )
                v_on = critic_values(net, batch["obs"])
                onp = score(batch["returns"], v_on)
                onp_i = (
                    score(batch["returns_with_intrinsic"], v_on)
                    if batch.get("returns_with_intrinsic") is not None else None
                )
                onp_meta = {
                    "n_valid": batch["n_valid"], "n_total": batch["n_total"],
                    "n_uncensored": batch["n_uncensored"],
                    "n_sampled": batch["n_sampled"],
                    "n_episodes": roll.n_episodes, "n_clears": roll.n_clears,
                    "clear_rate": (roll.n_clears / roll.n_episodes
                                   if roll.n_episodes else None),
                    "rollout_seconds": roll.seconds,
                }

            receipt = {
                "instrument": INSTRUMENT,
                "schema_version": SCHEMA_VERSION,
                "run": spec.run_name,
                "profile": str(spec.profile_path),
                "checkpoint": str(ck_path),
                "iter": int(it),
                "tau": int(tau),
                "reference_batch": ref_fingerprint,
                "ev_fixed": fixed["ev"],
                "ev_fixed_with_intrinsic": fixed_i["ev"] if fixed_i else None,
                "ev_onpolicy": onp["ev"] if onp else None,
                "ev_onpolicy_with_intrinsic": onp_i["ev"] if onp_i else None,
                "fixed_stats": fixed,
                "fixed_stats_with_intrinsic": fixed_i,
                "onpolicy_stats": onp,
                "onpolicy_stats_with_intrinsic": onp_i,
                "onpolicy_rollout": onp_meta,
                "harness": harness,
                "seconds": time.perf_counter() - t0,
            }
            path = out_dir / f"{spec.run_name}_it{it:05d}.json"
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            rows.append(receipt)
            print(
                f"[f1] iter {it:>4}  tau={tau:<4} "
                f"EV_fixed={_fmt(fixed['ev'])}  "
                f"EV_fixed+int={_fmt(fixed_i['ev'] if fixed_i else None)}  "
                f"EV_onpolicy={_fmt(onp['ev'] if onp else None)}  "
                f"-> {path.name}",
                flush=True,
            )
    finally:
        if roller is not None:
            roller.close()

    # ---- run-level summary + delta_fixed ------------------------------
    by_iter = {int(r["iter"]): r for r in rows}
    peak = int(args.peak_iter) if args.peak_iter is not None else int(ref_iter)
    last = max(by_iter)
    delta = None
    delta_i = None
    if peak in by_iter and by_iter[peak]["ev_fixed"] is not None \
            and by_iter[last]["ev_fixed"] is not None:
        delta = float(by_iter[last]["ev_fixed"] - by_iter[peak]["ev_fixed"])
        if by_iter[peak]["ev_fixed_with_intrinsic"] is not None:
            delta_i = float(by_iter[last]["ev_fixed_with_intrinsic"]
                            - by_iter[peak]["ev_fixed_with_intrinsic"])
    summary = {
        "instrument": INSTRUMENT,
        "schema_version": SCHEMA_VERSION,
        "run": spec.run_name,
        "profile": str(spec.profile_path),
        "peak_iter": peak,
        "last_grid_iter": last,
        "grid": grid,
        "reference_batch": ref_fingerprint,
        "delta_fixed": delta,
        "delta_fixed_with_intrinsic": delta_i,
        "band": classify_delta(delta) if delta is not None else None,
        "table": [
            {
                "iter": int(r["iter"]), "tau": int(r["tau"]),
                "ev_fixed": r["ev_fixed"],
                "ev_fixed_with_intrinsic": r["ev_fixed_with_intrinsic"],
                "ev_onpolicy": r["ev_onpolicy"],
                "onpolicy_clear_rate": r["onpolicy_rollout"].get("clear_rate"),
            }
            for r in rows
        ],
        "harness": harness,
        "note": (
            "delta_fixed = EV_fixed(last available grid point) - EV_fixed(P). "
            "Gating band per V29 F1: <= -0.15 live, > -0.10 dead, between = "
            "inconclusive. This file is one run; the verdict is a majority "
            "over all 8 (scripts/critic_explained_variance.py --verdict)."
        ),
    }
    spath = out_dir / f"f1_{spec.run_name}.json"
    spath.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[f1] {spec.run_name}: delta_fixed={_fmt(delta)} "
          f"({summary['band']}) -> {spath}", flush=True)
    return 0


def _fmt(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{x:+.4f}"


def run_verdict(args: argparse.Namespace) -> int:
    d = Path(args.verdict)
    deltas: dict[str, float] = {}
    for p in sorted(d.glob("f1_*.json")):
        if p.name == "f1_verdict.json":
            continue
        data = json.loads(p.read_text())
        if data.get("delta_fixed") is None:
            print(f"[f1] {p.name}: delta_fixed is null — skipped", flush=True)
            continue
        deltas[str(data.get("run", p.stem))] = float(data["delta_fixed"])
    verdict = f1_verdict(
        deltas, expected_runs=int(args.expected_runs),
        majority=int(args.majority),
    )
    out = d / "f1_verdict.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"[f1] verdict -> {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--profile", help="training profile YAML for the run")
    p.add_argument("--peak-iter", type=int, default=None,
                   help="corrected peak P from the F0 ladder")
    p.add_argument("--iters", type=_int_list, default=None,
                   help="explicit comma-separated iters instead of P+offsets")
    p.add_argument("--grid-offsets", type=_int_list, default=DEFAULT_GRID_OFFSETS,
                   help=f"default {','.join(str(o) for o in DEFAULT_GRID_OFFSETS)}")
    p.add_argument("--reference-iter", type=int, default=None,
                   help="checkpoint that collects the frozen batch (default: P)")
    p.add_argument("--reference-batch", default=None,
                   help="reuse an existing frozen-batch .npz instead of collecting")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--rollout-steps", type=int, default=None,
                   help="default: the profile's reinforce.rollout_steps")
    p.add_argument("--num-envs", type=int, default=None,
                   help="default: the profile's reinforce.num_envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-restart-age", type=int, default=0,
                   help="drop states within N steps of a restart (default 0 = "
                        "keep them; the critic was fit on them)")
    p.add_argument("--no-onpolicy", action="store_true",
                   help="skip EV_onpolicy (one rollout per grid point)")
    p.add_argument("--torch-threads", type=int, default=2,
                   help="torch intra-op threads; keep low when other jobs run")
    p.add_argument("--out-dir", default="runs/v29_stability/f1_explained_variance")
    p.add_argument("--verdict", default=None,
                   help="apply the F1 kill rule to a directory of f1_<run>.json")
    p.add_argument("--expected-runs", type=int, default=REGISTERED_RUNS)
    p.add_argument("--majority", type=int, default=REGISTERED_MAJORITY)
    return p


def _int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(s).replace(" ", "").split(",") if x)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_grad_enabled(False)
    if args.verdict:
        return run_verdict(args)
    if not args.profile:
        build_parser().error("--profile is required unless --verdict is given")
    if args.peak_iter is None and not args.iters:
        build_parser().error("one of --peak-iter or --iters is required")
    if args.reference_iter is None and args.peak_iter is None:
        build_parser().error(
            "--reference-iter is required when --iters is used without "
            "--peak-iter: the frozen batch must name the checkpoint that "
            "collected it."
        )
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
