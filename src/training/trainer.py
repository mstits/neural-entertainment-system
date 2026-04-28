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
    return sampled, chosen_lp


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
        # (num_instances, max_episode_steps, 4, 84, 84). At 16 workers
        # and max_episode_steps=937000 that's 423 GB of VSZ per buffer,
        # and TWO buffers coexist (pop-wide in _run_one_generation +
        # per-batch inside _evaluate_batch) — so a naive cap of 8 GB
        # still peaks at ~17 GB resident mid-generation. Keep each
        # buffer ≤ 2 GB so the two-buffer peak stays under 5 GB.
        # REINFORCE only looks at the last reinforce_max_steps (default
        # 1500) of each trajectory, so anything past ~2000 steps of
        # recording is dead weight anyway.
        _TRAJ_OBS_BUDGET_BYTES = 2 * 1024 ** 3
        _bytes_per_step = num_instances * 4 * 84 * 84 * 2  # float16 worst case
        _safe_max = max(1024, _TRAJ_OBS_BUDGET_BYTES // _bytes_per_step)
        if self.max_episode_steps > _safe_max:
            log.warning(
                "max_episode_steps=%d requires %.1f GB per trajectory obs "
                "buffer (%d workers × 4×84×84 × float16). Clamping to %d "
                "to keep per-buffer memory ≤ %.0f GB; REINFORCE only uses "
                "the last reinforce_max_steps (default 1500) of each "
                "trajectory, so this does not change learning signal.",
                self.max_episode_steps,
                num_instances * self.max_episode_steps * 4 * 84 * 84 * 2 / 1e9,
                num_instances, _safe_max,
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

        self.device = (
            torch.device(device_override) if device_override else get_best_device()
        )
        log.info("Using device: %s", self.device)

        self.action_space = game_profile.get("action_space", [])
        self.num_actions = len(self.action_space)
        self._bitmask_table = self._build_bitmask_table()
        # Reserved: when a future rewrite of _evaluate_batch actually
        # overlaps GPU inference for action_{t+1} with worker
        # emulation of action_t (which requires action_{t+1} to be
        # computed from obs_{t-1} — 1-step observation lag), this
        # flag will gate it. Today it's a no-op because inference and
        # emulation are still strictly serial. See
        # `docs/rust_nes_core.md` ("Future perf wins") for the design
        # sketch. Keeping the profile knob so downstream configs don't
        # have to change when the feature lands.
        self.async_pipeline = bool(game_profile.get("async_pipeline", False))
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

    def _make_network(self) -> PolicyNetwork:
        return PolicyNetwork(num_actions=self.num_actions)

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
        pop_obs_dtype = np.float16 if self.preprocess_f16 else np.uint8
        pop_traj_obs = np.zeros(
            (len(pop), self.max_episode_steps, 4, 84, 84), dtype=pop_obs_dtype
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
            fitnesses, successes, level_ids, traj_flat, batch_bd = \
                self._evaluate_batch(batch)
            for g, f in zip(batch, fitnesses):
                g.fitness = f
            for success, level_id in zip(successes, level_ids):
                self.curriculum.record_episode(level_id, success)
            # Copy this batch's flat arrays into the population-wide slot.
            # Slice by [:len(batch)] in case the last batch is partial.
            nb = len(batch)
            pop_traj_obs[batch_start:batch_start + nb] = traj_flat["obs"][:nb]
            pop_traj_actions[batch_start:batch_start + nb] = traj_flat["actions"][:nb]
            pop_traj_rewards[batch_start:batch_start + nb] = traj_flat["rewards"][:nb]
            pop_traj_log_probs[batch_start:batch_start + nb] = traj_flat["log_probs"][:nb]
            pop_traj_lens[batch_start:batch_start + nb] = traj_flat["lens"][:nb]
            for k, v in batch_bd.items():
                gen_breakdown[k] = gen_breakdown.get(k, 0.0) + v

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
            counts = np.bincount(
                pop_traj_actions[:, :pop_traj_lens.max() if pop_traj_lens.size else 0].ravel(),
                minlength=self.num_actions,
            )
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
        if self.reinforce_enabled:
            # Rank by fitness; pick top_k with non-trivial trajectories.
            indexed = list(enumerate(pop))
            indexed.sort(key=lambda ig: ig[1].fitness, reverse=True)
            top_k = max(1, int(self.reinforce_top_k))
            elite_idx = [
                idx for idx, _ in indexed[:top_k] if pop_traj_lens[idx] > 1
            ]
            if elite_idx:
                try:
                    loss = self._reinforce_update(
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
                    for idx, _ in indexed[1:num_preserved]:
                        pop[idx].state_dict = {
                            k: v.clone() for k, v in best.state_dict.items()
                        }
                    log.info(
                        "  REINFORCE loss: %.4f (n=%d, applied to %d elites)",
                        loss, len(elite_idx), num_preserved,
                    )
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

        self._emit_metrics(
            generation=gen,
            best_fitness=best.fitness,
            avg_fitness=avg,
            stage=self.curriculum.current_stage.name,
            success_rate=self.curriculum.stage_success_rate(),
            episodes=self.curriculum.episodes_in_stage,
            **breakdown_metrics,
            **timing_metrics,
        )

        # Log the breakdown for quick human inspection — which signals
        # actually fired this generation?
        if gen_breakdown:
            top = sorted(gen_breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
            log.info("  reward breakdown: %s", " ".join(f"{k}={v:.1f}" for k, v in top))

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

        obs_dtype = np.float16 if self.preprocess_f16 else np.uint8
        stackers = [FrameStacker(dtype=obs_dtype) for _ in range(n)]

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
        stacked_obs: list[np.ndarray] = [
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
        _obs_shape = (n, self.max_episode_steps, 4, 84, 84)
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
        batch_np_buffer = np.zeros((n, 4, 84, 84), dtype=obs_dtype)
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
                        sampled, chosen_lp = _safe_sample_from_logits(stacked_logits)
                    sampled_cpu = sampled.cpu().numpy()
                    lp_cpu = chosen_lp.cpu().numpy()
                    for batch_idx, genome_idx in enumerate(active):
                        actions[genome_idx] = int(sampled_cpu[batch_idx])
                        log_probs_old[genome_idx] = float(lp_cpu[batch_idx])
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
                        sampled, chosen_lp = _safe_sample_from_logits(cat_logits)
                    sampled_cpu = sampled.cpu().numpy()
                    lp_cpu = chosen_lp.cpu().numpy()
                    for batch_idx, genome_idx in enumerate(active):
                        actions[genome_idx] = int(sampled_cpu[batch_idx])
                        log_probs_old[genome_idx] = float(lp_cpu[batch_idx])
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
                # PERF: only snapshot the breakdown when a narrator is
                # attached. Skipping the dict copy saves ~60ns * steps *
                # workers per gen in headless training.
                narrator_on = self._narrator is not None
                prev_breakdown = (
                    dict(reward_fns[i].breakdown) if narrator_on else None
                )
                reward, rew_done, level_id = reward_fns[i].compute(
                    r.ram_snapshot, action=bitmasks.get(i, 0)
                )
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
        elif isinstance(spec, str) and ":" in spec and not Path(spec).exists():
            paths = [Path(p) for p in spec.split(":") if p]
        else:
            p = Path(spec)
            if p.is_dir():
                paths = sorted(p.glob("*.state.bin"))
            else:
                paths = [p]
        return [p for p in paths if p.exists()]

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
                log.info("Loaded cached BC seed from %s (skipping pretrain)", cache_path)
                seed_population_from_weights(self.ga.population, seed, noise_std=0.02)
                return
            except Exception as exc:
                log.warning("Failed to load BC cache %s (%s); re-pretraining.", cache_path, exc)

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
        states, actions, rewards = build_dataset(
            rom_path=self.rom_path,
            demo_path=demos if len(demos) != 1 else demos[0],
            action_space=self.action_space,
            frame_skip=self.frame_skip,
            reward_fn=reward_fn,
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

    def _reinforce_update(
        self,
        genome: Genome,
        traj_obs: np.ndarray,
        traj_actions: np.ndarray,
        traj_rewards: np.ndarray,
        traj_log_probs: np.ndarray,
        traj_lens: np.ndarray,
        elite_indices: list[int],
    ) -> float:
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

        optimizer = torch.optim.Adam(net.parameters(), lr=self.reinforce_lr)

        last_loss_scalar = 0.0
        for _ in range(self.reinforce_steps):
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=self.device)
            total_traj_items = 0

            for g_idx in elite_indices:
                traj_len = int(traj_lens[g_idx])
                if traj_len < 2:
                    continue
                gamma = self.reinforce_gamma
                # Flat reward stream for GAE.
                full_r = traj_rewards[g_idx, :traj_len]

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

                if self.preprocess_f16:
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
                    T = rewards_t.numel()
                    # Next-state values: shift values_pred by 1 and
                    # repeat the last (bootstrap) — equivalent to
                    # V(s_{T+1}) ≈ V(s_T) for truncated tails.
                    values_next = torch.empty_like(values_pred)
                    if T > 1:
                        values_next[:-1] = values_pred[1:]
                    values_next[-1] = values_pred[-1]  # bootstrap
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

                # Critic loss: MSE between predicted values and GAE
                # targets (advantages + baseline = one-step-TD return
                # estimate). 0.5 coefficient is the PPO standard.
                value_loss = F.mse_loss(values_pred, value_targets)

                loss_traj = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                total_loss = total_loss + loss_traj
                total_traj_items += 1  # count trajectories, not items

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

        # Copy the updated weights back to the genome so the GA keeps them.
        genome.state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        return last_loss_scalar

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
        if os.environ.get("NES_DISABLE_COREML_EXPORT") == "1":
            log.debug("CoreML export disabled via NES_DISABLE_COREML_EXPORT=1")
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
        ckpts = sorted(self.checkpoint_dir.glob("gen_*.pt"))
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
