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
import queue as _queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.emulation.frame_utils import FrameStacker
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
from src.training.behavior_cloning import build_dataset, pretrain, seed_population_from_weights
from src.training.curriculum import CurriculumManager, CurriculumStage
from src.training.genetic_algorithm import GeneticAlgorithm, Genome
from src.utils.reward_functions import build_reward_function


log = logging.getLogger(__name__)


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
        self.checkpoint_dir = Path(checkpoint_dir)
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
        self.start_state_path = start_state_path
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
        self._tile_extractor = None
        self._tile_feature_dim: int = 0
        if self._is_tile_mode:
            from src.emulation.tile_observations import get_extractor
            self._tile_extractor = get_extractor(self.encoder_kind)
            self._tile_feature_dim = self._tile_extractor.feature_dim
            log.info(
                "Tile mode active: encoder=%s, feature_dim=%d",
                self.encoder_kind, self._tile_feature_dim,
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

        self.action_space = game_profile.get("action_space", [])
        self.num_actions = len(self.action_space)
        self._bitmask_table = self._build_bitmask_table()
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
        # Frames emulated between decisions. 16 is ~2.8× faster in
        # game-time throughput than 8 with no measurable learning loss on
        # slow NES games. Profiles can override: twitchy games (Contra
        # boss fights, frame-perfect platformers) may benefit from 8.
        self.frame_skip = int(game_profile.get("frame_skip", 16))
        if self.num_actions == 0:
            raise ValueError("Game profile must define a non-empty action_space")

        profile_name = str(game_profile.get("name", "")).lower()
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
        self._metrics_queue = metrics_queue
        self._reward_queue = reward_queue  # live reward-weight updates
        self._audio_queue = audio_queue    # GUI -> trainer: mixer mode/volume
        self._audio_mixer = None  # built lazily on first push
        # Background drainer for the audio queue — without it, mixer
        # commands (mute / solo / volume) only get applied at generation
        # boundaries, which can be minutes apart. See _start_audio_drainer.
        self._audio_drainer_stop: Optional[threading.Event] = None
        self._audio_drainer_thread: Optional[threading.Thread] = None
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

        # Metrics log
        self._metrics_path = self.checkpoint_dir / "metrics.jsonl"

        # TensorBoard writer (lazy-init so tests that don't care don't
        # import torch.utils.tensorboard transitively and slow down.)
        self._tb_writer = None
        self._tb_enabled = game_profile.get("tensorboard", True)

    def _make_network(self):
        """Construct the policy network appropriate for the configured
        encoder. Pixel encoders (`nature_dqn`, `impala`) use the CNN-
        backed `PolicyNetwork`; tile encoders (`smb_tiles`) use the
        small MLP-backed `TilePolicyNetwork`."""
        if self._is_tile_mode:
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
        from src.emulation.frame_utils import (
            BUTTON_A, BUTTON_B, BUTTON_DOWN, BUTTON_LEFT, BUTTON_NOOP,
            BUTTON_RIGHT, BUTTON_SELECT, BUTTON_START, BUTTON_UP,
        )
        name_to_bit = {
            "NOOP": BUTTON_NOOP, "A": BUTTON_A, "B": BUTTON_B,
            "up": BUTTON_UP, "down": BUTTON_DOWN,
            "left": BUTTON_LEFT, "right": BUTTON_RIGHT,
            "start": BUTTON_START, "select": BUTTON_SELECT,
        }
        table = []
        for buttons in self.action_space:
            bitmask = 0
            for b in buttons:
                bitmask |= name_to_bit[b]
            table.append(bitmask)
        return tuple(table)


    def run(self, num_generations: int = 1000, resume_from: str | None = None) -> None:
        """Main training loop. Blocks until complete or stopped.

        If resume_from points to an existing GA checkpoint, load it and
        continue from that generation; otherwise start fresh.
        """
        import time as _time
        _t0 = _time.monotonic()
        def _stage(label: str) -> None:
            nonlocal _t0
            now = _time.monotonic()
            log.info("[startup] %s: %.2fs", label, now - _t0)
            _t0 = now

        self._running = True
        # Fresh metrics log per run — otherwise plots stack prior sessions.
        self._metrics_path.write_text("")
        _stage("metrics log reset")

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
            # Flush + close the TensorBoard writer if one was opened.
            # Without this, the last few generations of scalars buffered
            # in the writer never hit disk and the TB UI shows a
            # truncated tail.
            if self._tb_writer is not None:
                try:
                    self._tb_writer.flush()
                    self._tb_writer.close()
                except Exception:
                    pass
                self._tb_writer = None
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
        pool.shutdown()
        self.pool = _make_pool(
            env_spec=self.env_spec,
            rom_path=self.rom_path,
            num_workers=self.num_instances,
            start_state_path=stage_state,
            seed=self.seed,
        )
        self.pool.start()
        self._apply_pool_knobs()

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
                self._audio_mixer.set_mode(upd["mode"])
                self._apply_pace_for_mode(upd["mode"])
            if "volume" in upd:
                self._audio_mixer.set_volume(upd["volume"])

    def _apply_pace_for_mode(self, mode: str) -> None:
        """Toggle realtime pacing on workers based on the audio
        mixer mode. Solo X mode paces worker X (so its audio source
        produces continuously at realtime); all other modes (mute,
        all) unpace every worker so training runs at max throughput.
        Backends that don't support pacing silently no-op."""
        if self.pool is None:
            return
        soloed: Optional[int] = None
        if mode.startswith("solo-"):
            try:
                soloed = int(mode.split("-", 1)[1])
            except ValueError:
                soloed = None
        for worker_id in range(self.num_instances):
            self.pool.set_worker_pace(worker_id, soloed == worker_id)

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
        self.start_state_path = new_path_str
        if self.pool is not None:
            self.pool.shutdown()
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

    def _run_one_generation(self, gen: int) -> None:
        log.info("=== Generation %d (stage: %s) ===", gen, self.curriculum.current_stage.name)
        self._drain_reward_updates()
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
            # `successes` is OR-ed across episodes — any clear counts as
            # a curriculum success. Curriculum records every episode so
            # the success-rate window stays calibrated.
            best_fits = [-float("inf")] * nb
            sum_fits = [0.0] * nb
            best_traj_obs = None
            best_traj_actions = None
            best_traj_rewards = None
            best_traj_log_probs = None
            best_traj_lens = None
            best_level_ids: list[str] = ["?"] * nb
            for _ep in range(self.episodes_per_genome):
                fitnesses, successes, level_ids, traj_flat, batch_bd = \
                    self._evaluate_batch(batch)
                for g_i in range(nb):
                    sum_fits[g_i] += float(fitnesses[g_i])
                    if fitnesses[g_i] > best_fits[g_i]:
                        best_fits[g_i] = float(fitnesses[g_i])
                        best_level_ids[g_i] = level_ids[g_i]
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
                        # traj_flat reuse.
                        self._bc_replay_buffer.append((
                            traj_flat["obs"][g_i, :traj_len].copy(),
                            traj_flat["actions"][g_i, :traj_len].copy(),
                            traj_flat["rewards"][g_i, :traj_len].copy(),
                            float(fitnesses[g_i]),
                        ))
                        # FIFO eviction once over capacity.
                        if len(self._bc_replay_buffer) > self.bc_replay_max_buffer:
                            self._bc_replay_buffer.pop(0)
                        log.info(
                            "  BC replay: captured success trajectory "
                            "(genome=%d, len=%d, fitness=%.1f, buffer=%d/%d)",
                            g_i, traj_len, fitnesses[g_i],
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

        # BC replay: every N gens, train a fresh policy on the success
        # buffer and inject as a new genome. Keeps "what an actual clear
        # looks like" anchored in the population even as PPO drifts.
        if (
            self.bc_replay_enabled
            and len(self._bc_replay_buffer) > 0
            and gen > 0
            and gen % self.bc_replay_every_gens == 0
        ):
            self._run_bc_replay(pop)

        self.ga.evolve()

    def _evaluate_batch(
        self, genomes: list[Genome]
    ) -> tuple[list[float], list[bool], list[str], dict, dict]:
        """Run one episode per genome in parallel. Unused workers idle.

        Returns (fitnesses, successes, level_ids, trajectories) where
        trajectories[i] is a list of (state_uint8, action_idx, reward) tuples
        for genome i. Trajectories are used downstream for the REINFORCE
        gradient update on the generation's best genome.
        """
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
        # Frame stackers are only used in pixel mode. Tile mode
        # doesn't stack frames — each step's observation is the latest
        # tile feature vector decoded from current RAM.
        stackers = (
            [FrameStacker(dtype=obs_dtype) for _ in range(n)]
            if not self._is_tile_mode else []
        )
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
        if self._frame_sink is not None:
            try:
                self._frame_sink(all_init)
            except Exception:
                pass
        init_results = [r for r in all_init if r.worker_id >= offset]
        # Build per-genome initial observation. Pixel mode resets the
        # FrameStacker with the first frame; tile mode decodes RAM
        # directly into a feature vector. The resulting `stacked_obs`
        # variable is used identically by the rest of the loop below
        # — only its shape differs.
        if self._is_tile_mode:
            stacked_obs: list[np.ndarray] = [
                self._tile_extractor.extract(r.ram_snapshot)
                for r in init_results
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
            if self._frame_sink is not None:
                try:
                    self._frame_sink(results)
                except Exception:
                    pass

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
                    # Tile mode: re-decode the latest RAM snapshot into
                    # a tile feature vector. No frame stacking — the
                    # tile grid + scalar state already encodes
                    # everything the policy needs.
                    stacked_obs[i] = self._tile_extractor.extract(r.ram_snapshot)
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
                ep_success = (
                    reward_fns[i].episode_success() if bool(r.done) else False
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
                        self._narrator._events.append(NarratorEvent(
                            worker_id=worker_id,
                            genome_name=genome_name,
                            kind="depth_record",
                            caption=caption,
                            first_ever=True,  # always treat as banner-worthy
                            timestamp=time.time(),
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
                self._gen_timer.add(
                    "bookkeeping", time.perf_counter_ns() - _book_t0
                )

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
        """Resolve `self.bc_demo_path` to a concrete list of demo files.

        Accepts:
          * a single .state.bin path
          * a directory — every `*.state.bin` inside (sorted by name)
          * a colon-separated string of paths
          * already a list/tuple of paths

        Multi-demo BC is the only way to give the policy enough state
        coverage to matter on a long game like Zelda. A single 5-min
        recording is ~1k (state, action) pairs; six varied playthroughs
        is ~10k, which lets the network generalise instead of memorise.
        """
        if not self.bc_demo_path:
            return []
        paths: list[Path] = []
        spec = self.bc_demo_path
        if isinstance(spec, (list, tuple)):
            paths = [Path(p) for p in spec]
        elif isinstance(spec, str) and ":" in spec:
            # Colon-separated multi-demo spec from the GUI. The earlier
            # implementation called `Path(spec).exists()` to disambiguate
            # "single path that happens to contain ':'" from "joined
            # multi-demo list" — but on macOS that calls os.stat on the
            # full string, which raises ENAMETOOLONG (errno 63) at ~1024
            # chars. With 30+ demo files joined, the string easily
            # exceeds that. Guard the existence check by length first,
            # AND swallow the OSError defensively so an exotic FS error
            # never crashes BC seeding.
            looks_single = len(spec) < 1000
            if looks_single:
                try:
                    looks_single = Path(spec).exists()
                except OSError:
                    looks_single = False
            if looks_single:
                paths = [Path(spec)]
            else:
                paths = [Path(p) for p in spec.split(":") if p]
        else:
            p = Path(spec)
            if p.is_dir():
                paths = sorted(p.glob("*.state.bin"))
            else:
                paths = [p]
        # Filter to existing paths. Wrap `.exists()` defensively in case
        # any individual path is also too long for some reason.
        out: list[Path] = []
        for p in paths:
            try:
                if p.exists():
                    out.append(p)
            except OSError:
                log.warning("BC demo path skipped (stat failed): %s", p)
        return out

    def _bc_seed_cache_path(self) -> Optional[Path]:
        """Return the on-disk cache path for the BC seed tied to the
        current (demo file contents, ROM, game name, action space,
        frame_skip). Hashes ALL demos when more than one is configured
        so a single demo being added/replaced invalidates the cache.

        frame_skip is load-bearing: a seed trained at frame_skip=4
        produces a different observation distribution than one trained
        at frame_skip=16 — re-using the wrong seed silently warm-starts
        the policy on the wrong distribution.
        """
        demos = self._bc_demo_paths()
        if not demos:
            return None
        import hashlib
        h = hashlib.sha1()
        for demo in demos:
            h.update(demo.read_bytes())
            h.update(b"||")
        h.update(str(self.rom_path).encode())
        h.update(b"|")
        h.update(self.game_profile.get("name", "unknown").encode())
        h.update(b"|")
        for entry in self.action_space:
            h.update(("+".join(entry) + ",").encode())
        h.update(b"|")
        h.update(f"fs={self.frame_skip}".encode())
        # Encoder kind is load-bearing: a tile_mlp seed has shape
        # (175,) input, a nature_dqn seed has shape (4, 84, 84) input.
        # Loading one into the other crashes with a state_dict shape
        # mismatch. Including the encoder name in the cache key gives
        # each architecture its own seed file.
        h.update(b"|")
        h.update(f"enc={self.encoder_kind}".encode())
        digest = h.hexdigest()[:12]
        return self.checkpoint_dir / f"bc_seed_{digest}.pt"

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
            # feature vector, not stacked frames.
            tile_extractor=self._tile_extractor,
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
            for i, (obs, acts, rews, fit) in enumerate(self._bc_replay_buffer):
                data[f"traj_{i}_obs"] = obs
                data[f"traj_{i}_actions"] = acts
                data[f"traj_{i}_rewards"] = rews
                data[f"traj_{i}_fitness"] = np.array([fit], dtype=np.float32)
            data["count"] = np.array([len(self._bc_replay_buffer)], dtype=np.int32)
            # Schema metadata so the load path can validate that the
            # cache is compatible with the current trainer configuration
            # before injecting potentially-stale trajectories. Version
            # 1 = "carries num_actions + obs_shape." Bump on any
            # backward-incompatible format change.
            data["cache_version"] = np.array([1], dtype=np.int32)
            data["num_actions"] = np.array([self.num_actions], dtype=np.int32)
            obs_shape_tuple = self._obs_buffer_shape(1, 1)[2:]
            data["obs_shape"] = np.array(obs_shape_tuple, dtype=np.int64)
            # np.savez_compressed appends '.npz' to the filename if it
            # doesn't already end in '.npz' — so a temp path ending in
            # '.tmp' would write to <name>.tmp.npz on disk, breaking
            # the atomic-rename pattern. Anchor the temp path with .npz
            # itself so numpy writes to the literal path we name.
            tmp = self._bc_success_cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), **data)
            tmp.replace(self._bc_success_cache_path)
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
                self._bc_replay_buffer.append((obs, acts, rews, fit))
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
        # Concatenate all buffered trajectories into one big dataset.
        # Track per-trajectory start indices so AWR weighting resets
        # its discounted-return accumulator at episode boundaries
        # instead of leaking returns backwards across them.
        all_obs = np.concatenate([t[0] for t in self._bc_replay_buffer], axis=0)
        all_acts = np.concatenate([t[1] for t in self._bc_replay_buffer], axis=0)
        all_rews = np.concatenate([t[2] for t in self._bc_replay_buffer], axis=0)
        episode_boundaries: list[int] = []
        running = 0
        for t in self._bc_replay_buffer[:-1]:  # last one's "end" isn't a boundary
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
        # snapshot pattern. Fitness is set to the buffer's mean so the
        # GA gives the new genome a fair shot at elitism for one gen
        # before re-evaluation.
        weakest_i = min(range(len(pop)), key=lambda i: pop[i].fitness)
        avg_buffer_fitness = sum(t[3] for t in self._bc_replay_buffer) / len(self._bc_replay_buffer)
        pop[weakest_i].state_dict = {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()
        }
        pop[weakest_i].fitness = float(avg_buffer_fitness)
        log.info(
            "  BC replay: trained on %d successful trajectories (%d state-action pairs), "
            "BC loss %.4f, injected to slot %d (fitness %.1f for next-gen elitism)",
            len(self._bc_replay_buffer), n, loss, weakest_i, avg_buffer_fitness,
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
        net = self._make_network()
        # See trainer.py:424 for the strict=False rationale (backward compat
        # with checkpoints that predate the value_head).
        net.load_state_dict(genome.state_dict, strict=False)
        net.to(self.device)
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

        opt_params = list(net.parameters())
        if self._rnd is not None:
            opt_params += list(self._rnd.predictor.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=self.reinforce_lr)

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
                    rnd_per_sample = self._rnd(states_t)  # (T,) normalized
                    # Update obs + reward running stats with the latest
                    # batch BEFORE detaching, so subsequent batches see
                    # an up-to-date normalization scale.
                    self._rnd.update_normalization(
                        states_t.detach(),
                        rnd_per_sample.detach(),
                    )
                    rnd_intrinsic_t = (
                        rnd_per_sample.detach() * self.rnd_intrinsic_coef
                    )
                    rnd_loss_term = rnd_per_sample.mean()

                # Generalized Advantage Estimation (GAE-λ) using the
                # critic's value baseline. This replaces the raw
                # (G - mean)/std normalization that was mathematically
                # invalid under PPO clipping (high-variance advantages
                # cause the clip to suppress learning signal).
                # Bootstrap: the final state's value is approximated
                # by the critic's prediction (truncated-episode case
                # — most training trajectories hit max_steps without
                # a natural done). For naturally-terminated episodes
                # this slightly over-weights the tail but converges
                # well in practice.
                with torch.no_grad():
                    rewards_t = torch.from_numpy(
                        full_r[window_offset:]
                    ).to(self.device).float()
                    if rnd_intrinsic_t is not None:
                        # Length match is guaranteed: rnd_intrinsic_t
                        # has the same T as states_t which was sliced
                        # from `[win_start:traj_len]`, identical to the
                        # window applied to `full_r`.
                        rewards_t = rewards_t + rnd_intrinsic_t
                    T = rewards_t.numel()
                    # Next-state values: shift values_pred by 1 to
                    # produce V(s_{t+1}). For the LAST timestep:
                    #   * If the episode ended naturally (death,
                    #     level-clear) — i.e. traj_len < max_steps —
                    #     V(s_{T+1}) = 0 (no future after terminal).
                    #   * Otherwise (truncated by max_steps), bootstrap
                    #     with V(s_T) so the GAE return uses the
                    #     critic's estimate of "what comes next."
                    #
                    # The previous unconditional bootstrap was a major
                    # learning bug: at death, true delta_T = r_T - V(s_T),
                    # but bootstrap gives r_T + (γ-1)·V(s_T). With γ=0.99
                    # the death signal was muted ~100× — the policy
                    # could not learn to avoid pits/enemies because the
                    # gradient at terminal states was dominated by the
                    # value-bootstrap, not the death penalty.
                    values_next = torch.empty_like(values_pred)
                    if T > 1:
                        values_next[:-1] = values_pred[1:]
                    traj_len_full = int(traj_lens[g_idx])
                    terminated_naturally = (
                        traj_len_full < self.max_episode_steps
                    )
                    if terminated_naturally:
                        values_next[-1] = 0.0  # no future after terminal
                    else:
                        values_next[-1] = values_pred[-1]  # bootstrap for truncation
                    gae_lambda = 0.95  # standard PPO default
                    deltas = rewards_t + gamma * values_next - values_pred
                    # GAE recurrence runs CPU-side: each iteration's
                    # `advantages[t] = python_scalar` setitem on an MPS
                    # tensor would otherwise trigger
                    # `THPVariable_setitem → fill_scalar_mps`, which
                    # creates a fresh MPSGraph cached graph PER CALL. The
                    # graphs accumulate ~5 MB each and the cache key
                    # mis-hits because the index `t` rotates them out;
                    # measured at >150 MB/s leak in 16-worker training
                    # (rounds went from ~1 min to ~20 min as RSS climbed
                    # past 30 GB). Computing the running sum on a
                    # numpy view, then a single `torch.from_numpy().to()`
                    # transfer for `advantages` keeps the MPS-side tensor
                    # alloc to one per trajectory.
                    deltas_np = deltas.detach().cpu().numpy()
                    advantages_np = np.empty_like(deltas_np)
                    running_adv = 0.0
                    for t in range(T - 1, -1, -1):
                        running_adv = float(deltas_np[t]) + gamma * gae_lambda * running_adv
                        advantages_np[t] = running_adv
                    advantages = torch.from_numpy(advantages_np).to(self.device)
                    # Targets for the critic: advantage + baseline =
                    # the discounted return estimate. Learn V toward this.
                    value_targets = advantages + values_pred.detach()
                    # Normalize advantages (standard PPO practice).
                    if advantages.numel() > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

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

    def _save_checkpoint(self, gen: int, keep_last: int = 5) -> None:
        # Atomic write: serialize to `<path>.tmp`, fsync, then rename.
        # A torch.save directly to `path` that's interrupted mid-write
        # (Ctrl-C, OOM, power loss) leaves a truncated unpickleable
        # file and the next resume crashes. Rename is atomic on POSIX,
        # so either the old or the new checkpoint is visible — never
        # a half-written one.
        import os
        path = self.checkpoint_dir / f"gen_{gen:05d}.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.ga.save_checkpoint(str(tmp_path))
        try:
            fd = os.open(str(tmp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # fsync is best-effort — on some filesystems it errors out,
            # but the rename alone still gives us a consistent file.
            pass
        os.replace(str(tmp_path), str(path))

        curr_path = self.checkpoint_dir / "curriculum.json"
        curr_tmp = curr_path.with_suffix(curr_path.suffix + ".tmp")
        with open(curr_tmp, "w") as f:
            json.dump(self.curriculum.state_dict(), f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(str(curr_tmp), str(curr_path))

        log.info("Saved checkpoint: %s", path)

        # Also export the elite genome's policy to a Core ML .mlpackage
        # alongside the checkpoint. The replay viewer prefers CoreML/ANE
        # at batch=1 — 8× faster than PyTorch MPS per
        # scripts/bench_coreml_ane.py. Exporting here (once per
        # checkpoint) amortizes the ~100ms JIT+convert cost out of the
        # interactive replay-launch path; the viewer now just loads.
        # Best-effort: any export failure is logged and ignored; the
        # viewer falls back to MPS automatically if the .mlpackage is
        # missing.
        #
        # Tile mode: skipped. The 14k-param MLP is fast enough on CPU
        # that ANE acceleration is irrelevant; the export was profiling
        # at ~13% of trainer wall-time on tile runs (every 10 gens for
        # ~0.8s each in a subprocess). Net cost > replay-viewer benefit.
        if os.environ.get("NES_DISABLE_COREML_EXPORT") == "1":
            log.debug("CoreML export disabled via NES_DISABLE_COREML_EXPORT=1")
        elif self._is_tile_mode:
            log.debug("CoreML export skipped for tile mode (network too small to benefit)")
        else:
            try:
                elite = self.ga.best_genome()
                if elite is not None and elite.state_dict is not None:
                    from src.models.coreml_export import maybe_export
                    net = self._make_network()
                    net.load_state_dict(elite.state_dict, strict=False)
                    mlpath = path.with_suffix(".mlpackage")
                    maybe_export(
                        net,
                        str(mlpath),
                        num_actions=len(self.action_space),
                    )
                    del net
                    import gc as _gc
                    _gc.collect()
            except Exception as exc:
                log.debug("checkpoint CoreML export skipped: %s", exc)

        # Rotate old checkpoints so overnight runs don't fill the disk. A
        # single checkpoint is ~100 MB for population=16, Nature-DQN.
        # Sort by MODIFICATION TIME, not filename, so a fresh run that
        # restarts the gen counter at 0 doesn't get its checkpoints
        # silently deleted by an older, higher-named cohort still
        # sitting in the dir. The previous name-sort had a real
        # data-loss case: tile-mode runs writing gen_00010, 00020,
        # ... were instantly pruned because gen_01094 (pixel-era) was
        # alphabetically higher. mtime is the only signal that
        # actually tracks "most recent" across resumes.
        ckpts = sorted(
            self.checkpoint_dir.glob("gen_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if keep_last > 0 and len(ckpts) > keep_last:
            for old in ckpts[:-keep_last]:
                try:
                    old.unlink()
                except OSError:
                    pass
                # Remove the paired .mlpackage too, if present.
                mlpath = old.with_suffix(".mlpackage")
                if mlpath.exists():
                    import shutil
                    try:
                        shutil.rmtree(mlpath)
                    except OSError:
                        pass

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
        metrics["timestamp"] = time.time()
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._metrics_queue is not None:
            try:
                self._metrics_queue.put_nowait(metrics)
            except Exception:
                pass  # queue full — drop this update

        # TensorBoard (optional, lazy init so import cost is only paid when
        # actually used and never during tests or headless CI runs that
        # don't care about rich plots).
        if self._tb_enabled:
            if self._tb_writer is None:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._tb_writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "tb"))
                except Exception as exc:
                    log.debug("TensorBoard unavailable, disabling: %s", exc)
                    self._tb_enabled = False
                    return
            gen = metrics.get("generation", 0)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("generation", "timestamp"):
                    try:
                        self._tb_writer.add_scalar(k, float(v), gen)
                    except Exception:
                        pass

    def stop(self) -> None:
        self._running = False


def load_game_profile(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return the newest gen_*.pt in `checkpoint_dir`, or None if empty."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    ckpts = sorted(d.glob("gen_*.pt"))
    return ckpts[-1] if ckpts else None


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
