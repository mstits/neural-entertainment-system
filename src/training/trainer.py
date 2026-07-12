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
import queue as _queue
import threading
import time
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
from src.training.checkpointing import (
    archive_previous_run,
    find_latest_checkpoint as _find_latest_checkpoint,
    maybe_export_elite_to_coreml,
    rotate_old_checkpoints,
    save_checkpoint_atomic,
    save_winner,
)
from src.training.curriculum import CurriculumManager, CurriculumStage
from src.training.gae import gae as _gae
from src.training.genetic_algorithm import GeneticAlgorithm, Genome
from src.training.ppo import batched_gae, fold_intrinsic_into_rewards, ppo_losses
from src.training.metrics_sink import MetricsSink
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


def _safe_sample_from_logits(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per row from policy logits without crashing on
    NaN/Inf inputs. Returns (sampled_actions, chosen_log_probs).

    The PPO loss occasionally produces a network update that drives
    weights to NaN/Inf — usually when an outsized advantage estimate
    couples with a policy near zero entropy. The downstream
    `torch.multinomial` then raises `RuntimeError: probability tensor
    contains either inf, nan or element < 0` and kills the training
    process mid-run. This helper sanitises logits with `nan_to_num` +
    a clamp, falls back to a uniform distribution for any row that's
    still pathological, and logs a warning so the divergence is
    visible without taking the run down.
    """
    safe_logits = torch.nan_to_num(
        logits, nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(-50.0, 50.0)
    log_probs_all = F.log_softmax(safe_logits, dim=-1)
    probs = log_probs_all.exp()

    bad_rows = (~torch.isfinite(probs).all(dim=-1)) | (probs.sum(dim=-1) <= 0)
    if bad_rows.any():
        n_actions = probs.size(-1)
        uniform_p = 1.0 / float(n_actions)
        probs[bad_rows] = uniform_p
        log_probs_all[bad_rows] = float(np.log(uniform_p))
        log.warning(
            "policy logits contained NaN/Inf — substituted uniform "
            "distribution for %d row(s). Network may have diverged.",
            int(bad_rows.sum()),
        )

    sampled = torch.multinomial(probs, 1).squeeze(-1)
    chosen_lp = log_probs_all.gather(1, sampled.unsqueeze(1)).squeeze(1)
    # Returning the full log-probs tensor (in addition to the chosen
    # action's log-prob) lets callers implement sticky-action overrides
    # while keeping PPO's importance ratio correct: when sticky replaces
    # the sampled action with the previous one, we need the log-prob of
    # the OVERRIDDEN action under the CURRENT policy distribution, not
    # the sampled-action's log-prob.
    return sampled, chosen_lp, log_probs_all


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
        # Validate start_state_path up front — if the file doesn't exist
        # or is non-readable, warn and proceed without it rather than
        # letting the Pool constructor crash the training thread with a
        # hard RuntimeError. Observed 2026-04-21/22: a prior session
        # left a stale path in the GUI config; every Start click
        # produced "Training finished" because the thread died with
        # FileNotFoundError before gen 0 even began.
        if start_state_path:
            _ss = Path(start_state_path)
            if not _ss.exists():
                log.warning(
                    "start_state_path=%s does not exist — proceeding without "
                    "a start state (fresh boot). Clear the field in the GUI "
                    "or profile to silence this warning.",
                    start_state_path,
                )
                start_state_path = None
            elif not _ss.is_file():
                log.warning(
                    "start_state_path=%s is not a regular file — proceeding "
                    "without a start state.",
                    start_state_path,
                )
                start_state_path = None
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
        self._is_tile_mode: bool = self.encoder_kind in ("smb_tiles",)
        # Opt-in recurrent (GRU) tile policy for the vanilla_ppo path —
        # the SMB-past-1-2 lever (memory over the trajectory). Only valid
        # with the tile encoder; the proven feedforward path is untouched
        # when this is off. Wired in _run_vanilla_ppo (rollout threads the
        # hidden state, the update replays sequences with truncated BPTT).
        self._recurrent: bool = (
            bool(rl_cfg.get("recurrent", False)) and self._is_tile_mode
        )
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
        # Frame stacking for tile observations. yumouwei (2022) showed
        # empirically that stack=1 fails to clear SMB 1-1 while stack=4
        # succeeds — point-in-time RAM features lack temporal context
        # for jump timing and enemy avoidance. Reads `reinforce.tile_frame_stack`
        # with default 4 (the canonical yumouwei value). Set to 1 in
        # the YAML to disable stacking.
        self._tile_frame_stack: int = int(
            rl_cfg.get("tile_frame_stack", 4 if self._is_tile_mode else 1)
        )
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
        # Built lazily on first PPO step so the RND module lands on the
        # correct device once Trainer.run sets it up.
        self._rnd: Optional["RND"] = None
        # RND state (predictor weights + obs/reward running stats) read
        # from a resumed checkpoint. Applied at lazy-build time so resume
        # keeps the "what have I already explored" novelty memory instead
        # of re-rewarding every visited state as maximally novel.
        self._pending_rnd_state: Optional[dict] = None
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

        stages = self._build_curriculum_stages(game_profile.get("curriculum", {}))
        self.curriculum = CurriculumManager(stages=stages)

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
        # Depth tracker — watches per-step RAM for new-all-time-deep
        # positions (dungeon+room for Zelda, world+level+x for Mario,
        # etc.). New records are captioned through the narrator and
        # appended to checkpoints/depth_memo.jsonl for post-run analysis
        # and future auto-curriculum promotion.
        from src.training.depth_tracker import DepthTracker
        self._depth_tracker = DepthTracker(
            game=game_profile.get("name", "unknown"),
            memo_path=self.checkpoint_dir / "depth_memo.jsonl",
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

    def _make_network(self):
        """Construct the policy network appropriate for the configured
        encoder. Pixel encoders (`nature_dqn`, `impala`) use the CNN-
        backed `PolicyNetwork`; tile encoders (`smb_tiles`) use the
        small MLP-backed `TilePolicyNetwork`."""
        if self._is_tile_mode:
            if self._recurrent:
                from src.models.tile_policy import TileRecurrentPolicyNetwork
                return TileRecurrentPolicyNetwork(
                    num_actions=self.num_actions,
                    feature_dim=self._tile_feature_dim,
                )
            from src.models.tile_policy import TilePolicyNetwork
            return TilePolicyNetwork(
                num_actions=self.num_actions,
                feature_dim=self._tile_feature_dim,
            )
        return PolicyNetwork(
            num_actions=self.num_actions,
            encoder=self.encoder_kind,
            use_layernorm=self.use_layernorm,
        )

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
                lp = float(log_probs_all_cpu[batch_idx, a])
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
        return action_space_to_bitmasks(self.action_space)


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
                # so the rolling success-rate window is unbiased.
                for success, level_id in zip(successes, level_ids):
                    self.curriculum.record_episode(level_id, success)
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
                        sampled, chosen_lp, log_probs_all = _safe_sample_from_logits(stacked_logits)
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
                        sampled, chosen_lp, log_probs_all = _safe_sample_from_logits(cat_logits)
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
        if self.rnd_intrinsic_coef > 0.0 and self._rnd is None:
            if self._is_tile_mode:
                from src.models.tile_rnd import TileRND
                self._rnd = TileRND(
                    feature_dim=self._tile_feature_dim
                ).to(self.device)
            else:
                self._rnd = RND(in_channels=4).to(self.device)
            log.info(
                "RND intrinsic motivation enabled (%s): predictor=%d params, "
                "intrinsic_coef=%.3f, loss_coef=%.3f",
                "tile_mlp" if self._is_tile_mode else "pixel_cnn",
                self._rnd.num_params,
                self.rnd_intrinsic_coef,
                self.rnd_loss_coef,
            )
            self._apply_pending_rnd_state()

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
        """Load the newest vanilla_ppo_iter_*.pt on top of the freshly
        built net + optimizer and return the absolute iteration offset to
        continue checkpoint numbering from (0 = nothing loaded).

        The GA-path resume at trainer.run() loads `gen_*.pt` and is a
        no-op for our `vanilla_ppo_iter_*.pt` format, so this is the only
        resume a vanilla_ppo run gets. `fresh_start` is the user's
        explicit from-scratch request — the GUI "Resume" checkbox
        unticked, or headless `--no-resume` — and skips the scan so the
        run starts from random weights even when a checkpoint is sitting
        in the dir. Default False preserves the historical auto-resume so
        headless scripts that rely on it are unaffected.

        Sets `self._vppo_resumed_from_iter` to the resumed iter (or None
        when nothing was loaded) so the caller can surface it to the GUI.
        """
        self._vppo_resumed_from_iter = None
        if fresh_start:
            # Only worth a line when there was actually a checkpoint to
            # ignore — otherwise it is noise on a genuinely fresh dir.
            existing = list(
                self.checkpoint_dir.glob("vanilla_ppo_iter_*.pt")
            )
            if existing:
                log.info(
                    "[vanilla_ppo] fresh start requested — ignoring %d "
                    "existing checkpoint(s) in %s",
                    len(existing), self.checkpoint_dir,
                )
            return 0

        iter_offset = 0
        try:
            latest_ckpts = sorted(
                self.checkpoint_dir.glob("vanilla_ppo_iter_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            if latest_ckpts:
                latest = latest_ckpts[-1]
                state = torch.load(str(latest), map_location=self.device)
                if isinstance(state, dict) and "net_state_dict" in state:
                    net.load_state_dict(state["net_state_dict"], strict=False)
                    try:
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                    except Exception as exc:
                        log.warning(
                            "[vanilla_ppo] optimizer state load failed "
                            "(continuing with fresh Adam state): %s", exc,
                        )
                    iter_offset = int(state.get("iter", 0) or 0)
                    if "rnd_state_dict" in state:
                        # RND builds lazily on the first update (after this
                        # resume block); stash and apply at build time.
                        self._pending_rnd_state = state["rnd_state_dict"]
                    if "anticollapse" in state:
                        # Anti-collapse locals init after this block; stash
                        # and restore them there.
                        self._pending_anticollapse = state["anticollapse"]
                    self._vppo_resumed_from_iter = iter_offset
                    log.info(
                        "[vanilla_ppo] RESUMED net + optimizer from %s "
                        "(saved at iter %d; continuing checkpoint "
                        "numbering from there)",
                        latest, iter_offset,
                    )
        except Exception as exc:
            log.warning(
                "[vanilla_ppo] auto-resume scan failed (starting fresh): %s",
                exc,
            )
        return iter_offset

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

        # Persistent net + optimizer (same machinery the GA-mode PPO
        # update uses; reused so the existing fix infrastructure
        # — Adam state, lr, etc. — applies uniformly).
        if self._ppo_net is None:
            self._ppo_net = self._make_network()
            self._ppo_net.to(self.device)
        net = self._ppo_net
        # RND intrinsic exploration (opt-in via reinforce.rnd_intrinsic_coef).
        # Same module + knobs the GA path uses; lazily built so it lands
        # on the current device. Predictor params train in the same Adam
        # optimizer as the policy. Canonical SMB-PPO is exploration-
        # limited (entropy pins near ln(A), policy settles on a single
        # safe-failing behavior); RND's novelty bonus keeps advantages
        # alive past that wall.
        if self.rnd_intrinsic_coef > 0.0 and self._rnd is None:
            if self._is_tile_mode:
                from src.models.tile_rnd import TileRND
                self._rnd = TileRND(
                    feature_dim=self._tile_feature_dim
                ).to(self.device)
            else:
                self._rnd = RND(in_channels=4).to(self.device)
            log.info(
                "[vanilla_ppo] RND enabled (%s): predictor=%d params, "
                "intrinsic_coef=%.3f, loss_coef=%.3f",
                "tile_mlp" if self._is_tile_mode else "pixel_cnn",
                self._rnd.num_params, self.rnd_intrinsic_coef,
                self.rnd_loss_coef,
            )
            self._apply_pending_rnd_state()
        if self._ppo_optimizer is None:
            self._ppo_optimizer = self._build_ppo_optimizer(net)
        optimizer = self._ppo_optimizer

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
        iter_offset = self._maybe_resume_vanilla_ppo(
            net, optimizer, fresh_start=fresh_start,
        )

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
        try:
            from src.training.run_manifest import write_run_manifest
            mpath = write_run_manifest(
                self.checkpoint_dir,
                game=self.game_profile.get("name"),
                rom_path=self.rom_path,
                start_state_path=self.start_state_path,
                seed=self.seed,
                profile=self.game_profile,
                num_envs=num_envs,
                frame_skip=self.frame_skip,
            )
            log.info("[vanilla_ppo] wrote run manifest: %s", mpath)
        except Exception as exc:
            log.warning("[vanilla_ppo] run manifest write failed: %s", exc)

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
        smb_curriculum_anchors: list[int] = [0]  # area-byte for each stage
        smb_curriculum_states: list[Optional[bytes]] = [None]  # save state per stage
        smb_curriculum_stage = 0  # index into anchors
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
        smb_curriculum_dir = self.checkpoint_dir / "smb_curriculum"
        smb_curriculum_dir.mkdir(parents=True, exist_ok=True)
        # Non-Mario games skip the SMB area-byte curriculum entirely
        # (stage stays 0 -> no warm-start, no capture, no advance): plain
        # single-stage vanilla PPO from the profile start state.
        _curr_range = range(1, 32) if self._smb_curriculum_active else range(0)
        for n in _curr_range:
            stage_path = smb_curriculum_dir / f"stage_{n:02d}.state"
            meta_path = smb_curriculum_dir / f"stage_{n:02d}.meta.json"
            if not stage_path.exists():
                break
            try:
                blob = stage_path.read_bytes()
                # Read the anchor byte from the sidecar metadata. If
                # absent (state written before this fix), fall back to
                # reading ram[$0760] from the save state by briefly
                # loading it into worker 0. Pool exists at this point.
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    anchor = int(meta["anchor"])
                else:
                    # Legacy state without sidecar: derive anchor by
                    # peeking at the saved RAM. Load into worker 0,
                    # read byte, then re-reset so we don't pollute the
                    # cold-boot initial state for the actual run.
                    self.pool.load_worker_state(0, blob)
                    peek = self.pool.step_all(
                        np.zeros(self.pool.num_workers, dtype=np.uint8)
                    )
                    ram = peek[0][2]
                    anchor = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
                    log.info(
                        "[vanilla_ppo] derived anchor=%d for legacy "
                        "stage_%02d.state (no sidecar); writing it now.",
                        anchor, n,
                    )
                    meta_path.write_text(json.dumps({"anchor": anchor}))
                    # Reset before continuing setup.
                    self.pool.reset_all()
                smb_curriculum_states.append(blob)
                smb_curriculum_anchors.append(anchor)
                smb_curriculum_stage = n
            except Exception as e:
                log.warning(
                    "[vanilla_ppo] failed to load stage_%02d: %s", n, e,
                )
                break
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
        value_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        log_prob_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
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
        # Count per-env flag-touches across the iter. Each time the
        # reward fn's `completion` breakdown key grows, an SMB level
        # was cleared by that env. Tracks last-seen value per env so
        # we can diff per-step. Quantitative confirmation of clears
        # alongside the visual GUI signal.
        prev_completion_total = np.zeros(num_envs, dtype=np.float32)
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

        if go_explore_on:
            from src.training.go_explore import (
                GoExploreArchive, ram_bytes_cell, ram_downsample_cell,
            )
            _cell_cfg = dict(_ge_cfg.get("cell", {}) or {})
            _cell_type = str(_cell_cfg.get("type", "ram_downsample"))
            if _cell_type == "ram_bytes":
                _addrs = [
                    int(a) for a in _cell_cfg.get(
                        "addresses", [0x075F, 0x0760, 0x006D]
                    )
                ]
                cell_fn = ram_bytes_cell(
                    _addrs, bucket=int(_cell_cfg.get("bucket", 1))
                )
            else:
                cell_fn = ram_downsample_cell(
                    stride=int(_cell_cfg.get("stride", 64)),
                    bucket=int(_cell_cfg.get("bucket", 16)),
                )
            go_explore_archive = GoExploreArchive(
                cell_fn, seed=int(_ge_cfg.get("seed", self.seed) or 0)
            )
            self._go_explore = go_explore_archive
            _ge_path = self.checkpoint_dir / "go_explore" / "archive.pkl"
            if _ge_path.exists():
                try:
                    go_explore_archive.load(_ge_path)
                    log.info(
                        "[vanilla_ppo] go_explore: resumed archive (%d cells)",
                        len(go_explore_archive),
                    )
                except Exception as e:
                    log.warning("[vanilla_ppo] go_explore reload failed: %s", e)
            _score_kind = str(_ge_cfg.get("score", "auto"))
            _is_mario = "mario" in str(self.game_profile.get("name", "")).lower()
            _use_max_x = _score_kind == "max_x" or (
                _score_kind == "auto" and _is_mario
            )

            def _ge_score(_i: int) -> float:  # noqa: F811
                return (
                    float(max_x_reached[_i]) if _use_max_x
                    else float(ep_returns[_i])
                )

            log.info(
                "[vanilla_ppo] go_explore ENABLED: cell=%s, score=%s "
                "(SMB curriculum OFF)",
                _cell_type, "max_x" if _use_max_x else "ep_return",
            )

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

        for it in range(num_iters):
            if not self._running:
                break
            global_it = it + iter_offset
            _iter_t0 = time.perf_counter()
            # Per-iter pool-max screen for screen_XX reward games (Contra).
            # Surfaces progress in metrics even before a stage-end is known.
            iter_max_screen = 0
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
                    log_probs_all = F.log_softmax(logits.float(), dim=-1)
                    probs = log_probs_all.exp()
                    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    log_probs_taken = log_probs_all.gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)

                    actions_np = actions.cpu().numpy().astype(np.int32)
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
                        if (self._smb_curriculum_active
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
                        reward_buf[t, i] = reward
                        done_buf[t, i] = done
                        valid_buf[t, i] = True  # real executed step (incl. death)
                        ep_returns[i] += reward
                        ep_lengths[i] += 1
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
                            _gk = go_explore_archive.cell_fn(ram)
                            _gc = go_explore_archive.cells.get(_gk)
                            _gs = _ge_score(i)
                            _gsteps = int(ep_lengths[i])
                            _dom = (
                                _gc is None
                                or _gs > _gc.best_score + 1e-9
                                or (abs(_gs - _gc.best_score) <= 1e-9
                                    and _gsteps < _gc.best_steps)
                            )
                            if _dom:
                                try:
                                    _blob = self.pool.save_worker_state(i)
                                except Exception:
                                    _blob = None
                                if _blob is not None:
                                    go_explore_archive.record(
                                        ram, _blob, _gs, _gsteps
                                    )
                            else:
                                go_explore_archive.record(ram, None, _gs, _gsteps)

                        # Set True when an auto-reset re-seeds the
                        # stacker this step, so the post-step push below
                        # is skipped (the seed already IS the new obs).
                        reseeded = False
                        if done:
                            completed_returns.append(float(ep_returns[i]))
                            completed_lengths.append(int(ep_lengths[i]))
                            ep_returns[i] = 0.0
                            ep_lengths[i] = 0
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
                            else:
                                # Stage 0: no warm-state, freeze and wait
                                # for the iter-boundary reset_all.
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
                        if active_in_iter[i] and not reseeded:
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

            # ============== RND INTRINSIC REWARD ==============
            # Fold the per-state novelty bonus into the reward stream
            # BEFORE GAE so it bootstraps like any reward (single-stream
            # RND). Computed once on the full rollout obs with the
            # current predictor (no grad); the predictor is then trained
            # in the update below, driving the bonus down on familiar
            # states. Zeroed on done/padded steps by the fold helper.
            rnd_intrinsic_mean = 0.0
            if self._rnd is not None:
                # This full-rollout pass (rollout_steps × num_envs rows in
                # one op) is the only torch call in the loop big enough to
                # profit from the intra-op pool (62 ms at 1T vs 24 ms at
                # default threads). Lift the CPU 1-thread cap from
                # __init__ just for this block, then restore it for the
                # small-tensor minibatch loop below.
                _rnd_threads_raised = (
                    self.device.type == "cpu"
                    and torch.get_num_threads() < _TORCH_DEFAULT_NUM_THREADS
                )
                if _rnd_threads_raised:
                    torch.set_num_threads(_TORCH_DEFAULT_NUM_THREADS)
                try:
                    with torch.no_grad():
                        rnd_obs_t = (
                            torch.from_numpy(
                                obs_buf.reshape((rollout_steps * num_envs,) + obs_shape)
                            ).to(self.device).float()
                        )
                        if not self._is_tile_mode and not self.preprocess_f16:
                            rnd_obs_t = rnd_obs_t.div_(255.0)
                        intrinsic_raw = self._rnd(rnd_obs_t)  # raw per-sample MSE
                        bonus_t = self._rnd.normalize_bonus(intrinsic_raw)
                        # Update running stats with the RAW error, not the
                        # normalized bonus (avoids the reward_rms self-loop).
                        self._rnd.update_normalization(rnd_obs_t, intrinsic_raw)
                        intrinsic_np = (
                            bonus_t.cpu().numpy().astype(np.float32)
                            * self.rnd_intrinsic_coef
                        ).reshape(rollout_steps, num_envs)
                finally:
                    if _rnd_threads_raised:
                        torch.set_num_threads(1)
                rnd_intrinsic_mean = float(intrinsic_np.mean())
                reward_buf = fold_intrinsic_into_rewards(
                    reward_buf, intrinsic_np, done_buf
                )

            # ============== GAE-λ BACKWARD SWEEP ==============
            # Batched, per-step-done-masked GAE (src/training/ppo.py).
            # A done at step t zeroes both the bootstrapped next-value
            # and the running accumulator, breaking the episode
            # boundary so advantage never leaks across death/clear —
            # required now that auto-reset gives multiple dones per env
            # per rollout.
            _gae_t0 = time.perf_counter_ns()
            advantages, value_targets = batched_gae(
                reward_buf, value_buf, done_buf, final_values_np,
                self.reinforce_gamma, self.gae_lambda,
            )
            self._gen_timer.add("gae", time.perf_counter_ns() - _gae_t0)

            # Global advantage normalization across the entire (env × step)
            # batch. This is the canonical PPO advantage normalization
            # — preserves magnitude information between trajectories,
            # unlike per-trajectory z-scoring which would erase the
            # "this env had a high-return episode" signal.
            # Normalize over VALID steps only. Post-done frozen padding
            # carries advantage = -V(stale); including it would skew the
            # batch mean/std (and it is excluded from the minibatch below).
            valid_flat = valid_buf.reshape(-1)
            _valid_adv = advantages.reshape(-1)[valid_flat]
            if _valid_adv.size > 1:
                adv_mean = float(_valid_adv.mean())
                adv_std = float(_valid_adv.std()) + 1e-8
            else:
                adv_mean, adv_std = 0.0, 1.0
            advantages_norm = (advantages - adv_mean) / adv_std

            # ============== K-EPOCH PPO UPDATE ==============
            # Flatten (rollout_steps, num_envs, ...) → (rollout_steps * num_envs, ...)
            total_n = rollout_steps * num_envs
            obs_flat = obs_buf.reshape((total_n,) + obs_shape)
            action_flat = action_buf.reshape(-1)
            log_prob_old_flat = log_prob_buf.reshape(-1)
            adv_flat = advantages_norm.reshape(-1)
            target_flat = value_targets.reshape(-1)
            # Only real (valid) steps are trained on — drop frozen padding.
            valid_indices = np.where(valid_flat)[0]

            net.train()
            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            last_loss = 0.0
            last_rnd_loss = 0.0
            # Hold the final minibatch's loss tensors and convert to Python
            # floats ONCE after the update loop, instead of .item()-syncing
            # the MPS stream on every minibatch (only the last is logged).
            # Each .item() drains the GPU queue, serializing CPU-side
            # minibatch prep against GPU compute — costliest for the tiny
            # tile MLP. None-guarded so the recurrent path (which skips this
            # loop and sets last_* itself) is unaffected.
            _last_policy_t = _last_value_t = _last_entropy_t = None
            _last_loss_t = _last_rnd_t = None
            mb_size = max(1, self.ppo_minibatch_size)
            # Build the full-rollout tensors ONCE per iter and index them
            # with a torch permutation inside the minibatch loop, instead
            # of rebuilding 5 tensors from numpy (cast + host-copy) for
            # every one of ~K*total_n/mb minibatches. The four scalar
            # vectors are tiny; the obs tensor is materialized up front
            # only for tile mode (~170 MB float32) — for the pixel path
            # that would be multi-GB, so pixel obs stays per-minibatch.
            actions_all = torch.from_numpy(
                action_flat.astype(np.int64)
            ).to(self.device)
            log_probs_old_all = torch.from_numpy(log_prob_old_flat).to(self.device).float()
            adv_all = torch.from_numpy(adv_flat).to(self.device).float()
            target_all = torch.from_numpy(target_flat).to(self.device).float()
            obs_all = (
                torch.from_numpy(obs_flat).to(self.device).float()
                if (self._is_tile_mode and not self._recurrent) else None
            )
            _upd_t0 = time.perf_counter_ns()
            if self._recurrent:
                # Recurrent sequence-BPTT update. The feedforward epoch
                # loop below is skipped (its range is zeroed when
                # recurrent) so the proven path stays byte-for-byte intact.
                (last_policy_loss, last_value_loss, last_entropy,
                 last_loss, last_rnd_loss) = self._recurrent_ppo_update(
                    net, optimizer, obs_buf, action_buf, log_prob_buf,
                    advantages_norm, value_targets, done_buf,
                    num_envs, rollout_steps,
                )
            for epoch in range(0 if self._recurrent else self.reinforce_steps):
                perm = np.random.permutation(valid_indices)
                n_valid = perm.shape[0]
                for mb_start in range(0, n_valid, mb_size):
                    mb_end = min(mb_start + mb_size, n_valid)
                    mb_np = perm[mb_start:mb_end]
                    if mb_np.size < 2:
                        continue
                    mb_idx = torch.from_numpy(mb_np).to(self.device)

                    if obs_all is not None:
                        states_t = obs_all[mb_idx]
                    else:
                        states_t = torch.from_numpy(
                            np.ascontiguousarray(obs_flat[mb_np])
                        ).to(self.device).float()
                        # Pixel path: /255 only when the pool delivered
                        # uint8; preprocess_f16 obs are already [0,1].
                        if not self.preprocess_f16:
                            states_t = states_t.div_(255.0)
                    actions_t = actions_all[mb_idx]
                    log_probs_old_t = log_probs_old_all[mb_idx]
                    adv_t = adv_all[mb_idx]
                    target_t = target_all[mb_idx]

                    logits, values_pred = net.forward_ac(states_t)
                    # PPO clipped-surrogate + value + entropy loss
                    # (src/training/ppo.py). Forward pass and optimizer
                    # step stay here; the pure loss math is shared so
                    # the Phase 1 RND intrinsic term has one plug point.
                    loss, policy_loss, value_loss, entropy = ppo_losses(
                        logits, values_pred, actions_t, log_probs_old_t,
                        adv_t, target_t,
                        clip_eps=self.ppo_clip_eps,
                        value_coef=self.value_coef,
                        entropy_coef=self.entropy_coef,
                        value_loss_kind=self.value_loss_kind,
                    )
                    # RND predictor loss: train the predictor to mimic
                    # the frozen target on visited states (its params are
                    # in this optimizer). Forward with grad here, unlike
                    # the no-grad intrinsic-reward pass above.
                    if self._rnd is not None:
                        rnd_loss = self._rnd(states_t).mean()
                        loss = loss + self.rnd_loss_coef * rnd_loss
                        _last_rnd_t = rnd_loss.detach()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), self.reinforce_grad_clip
                    )
                    optimizer.step()

                    _last_policy_t = policy_loss.detach()
                    _last_value_t = value_loss.detach()
                    _last_entropy_t = entropy.detach()
                    _last_loss_t = loss.detach()
            # Single MPS sync for the final minibatch's scalars.
            if _last_policy_t is not None:
                last_policy_loss = float(_last_policy_t.item())
                last_value_loss = float(_last_value_t.item())
                last_entropy = float(_last_entropy_t.item())
                last_loss = float(_last_loss_t.item())
            if _last_rnd_t is not None:
                last_rnd_loss = float(_last_rnd_t.item())
            self._gen_timer.add("update", time.perf_counter_ns() - _upd_t0)

            # ============== LOGGING + METRICS ==============
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
            at_current = (env_stage == smb_curriculum_stage)
            n_at_current = int(at_current.sum())
            past_mask = (max_world_level_packed > current_anchor)
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
            if should_advance and smb_pending_capture is not None:
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
            # Per-section timing (rollout_forward / emulation / gae /
            # update) so the active pixel path stops flying blind: a
            # regression shows up as `jq '.timing_*' metrics.jsonl`.
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
                vanilla_ppo_samples_per_sec=samples_per_sec,
                vanilla_ppo_realtime_x=realtime_x,
                **resume_metrics,
                **progress_metrics,
                **timing_metrics,
                **reward_breakdown_emit,
            )

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

            # Clear per-iteration episode-completion buffer so the next
            # iter reports against fresh episode data only.
            completed_returns.clear()
            completed_lengths.clear()

            # Reset the entire env pool between iterations: every iter
            # starts from N fresh cold-boot episodes. Avoids workers
            # idling in a post-done state when our reward-fn / stacker
            # logic doesn't match the rust pool's auto-step semantics.
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
            if smb_curriculum_stage > 0:
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
            # Per-iter reset (was missing): without this, vanilla_ppo_max_x
            # is a cumulative running max that preserves an old peak even
            # after the policy regresses/collapses — which masked a 1-4
            # collapse (metric held at 2432 while the policy fell to 814).
            max_x_reached[:] = 0
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

            # Checkpoint every 10 iters. No GA to save — just the policy
            # network's state_dict (plus the ABSOLUTE iter counter for
            # resume). Filenames use global_it so a resumed run never
            # overwrites checkpoints saved before the resume point.
            if it > 0 and it % 10 == 0:
                ckpt_path = (
                    self.checkpoint_dir
                    / f"vanilla_ppo_iter_{global_it:05d}.pt"
                )
                try:
                    _ckpt_payload = {
                        "iter": global_it,
                        "net_state_dict": {
                            k: v.detach().cpu()
                            for k, v in net.state_dict().items()
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    # Persist RND (predictor weights + obs/reward running
                    # stats) so resume keeps the novelty memory instead of
                    # re-rewarding every already-explored state.
                    if self._rnd is not None:
                        _ckpt_payload["rnd_state_dict"] = {
                            k: v.detach().cpu()
                            for k, v in self._rnd.state_dict().items()
                        }
                    # Persist the anti-collapse rollback baseline (best-ever
                    # healthy snapshot + its fitness + strike count) so a
                    # resumed run keeps its collapse-recovery memory. The
                    # snapshot is already cpu-cloned at capture time.
                    if best_net_snapshot is not None:
                        _ckpt_payload["anticollapse"] = {
                            "best_net_snapshot": best_net_snapshot,
                            "best_snapshot_fitness": float(best_snapshot_fitness),
                            "collapse_strikes": int(collapse_strikes),
                        }
                    torch.save(_ckpt_payload, str(ckpt_path))
                    log.info("[vanilla_ppo] saved checkpoint: %s", ckpt_path)
                except Exception as exc:
                    log.warning("[vanilla_ppo] checkpoint save failed: %s", exc)

                # Retain the best-ever policy by clear rate under winners/
                # (excluded from rotation). This is what keeps `make demo`/
                # `make eval` pointed at a real win even if the curriculum
                # run later self-collapses. save_winner only overwrites when
                # this iter's clear rate strictly beats the stored best.
                try:
                    save_winner(
                        {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        game=str(self.game_profile.get("name", "game")),
                        metric_value=float(vppo_success_rate),
                        out_dir=self.checkpoint_dir,
                        metric_name="clear_rate",
                        source_iter=global_it,
                    )
                except Exception as exc:
                    log.warning("[vanilla_ppo] winner retention failed: %s", exc)

            # Persist the Go-Explore archive on its OWN cadence (dedented out
            # of the 10-iter checkpoint block above) so save_every actually
            # takes effect instead of being masked by lcm(10, save_every).
            if go_explore_archive is not None and it % go_explore_save_every == 0:
                try:
                    go_explore_archive.save(
                        self.checkpoint_dir / "go_explore" / "archive.pkl"
                    )
                except Exception as exc:
                    log.warning(
                        "[vanilla_ppo] go_explore archive save failed: %s", exc
                    )

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

    def _apply_pending_rnd_state(self) -> None:
        """Load RND state stashed from a resumed checkpoint into the
        freshly-built module. RND builds lazily on the first update —
        after resume runs — so the state can't be applied at load time.
        No-op when nothing is pending. Called right after each lazy build."""
        if self._pending_rnd_state is None or self._rnd is None:
            return
        try:
            self._rnd.load_state_dict(self._pending_rnd_state, strict=False)
            log.info(
                "[vanilla_ppo] restored RND state (predictor + normalization) "
                "from checkpoint — novelty memory preserved across resume"
            )
        except Exception as exc:
            log.warning(
                "[vanilla_ppo] RND state restore failed (fresh init): %s", exc
            )
        self._pending_rnd_state = None

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
