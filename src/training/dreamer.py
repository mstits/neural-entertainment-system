"""DreamerV3-style world-model trainer.

Self-contained alternative to the GA + PPO trainer. Picked at startup
based on `profile.training_mode == "dreamer"`. Reuses the same
`nes_core.Pool` for data collection but runs an entirely different
update loop:

  1. Collect: act in the pool with the current actor on the current
     world-model latent, append `(obs, action, reward, done)` to a
     sequential replay buffer.
  2. Train world model: sample `(B, L)` sequence batches from replay,
     compute reconstruction + reward + continue + KL losses, backward,
     optimizer step.
  3. Train actor + critic: imagine `H` steps of latent rollout from
     each sequence's posterior latents, compute λ-returns from the
     reward + value heads + target critic, backprop policy loss
     (advantage) + entropy + critic regression.
  4. Polyak-update the target critic.

This is a structurally-correct scaffold of DreamerV3. Tuning to actually
beat PPO on Zelda is a separate research effort; the architecture is in
place to make that experimentation possible. See `docs/world_model_rl.md`
for the long-term plan.

Compared to the PPO trainer this one is intentionally minimal — no
genetic algorithm, no curriculum manager, no behavior cloning seeding.
Those layers can be added back if they prove useful, but starting clean
makes the world-model side easier to debug.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.emulation.frame_utils import FrameStacker
from src.emulation.rust_pool_adapter import RustPool, StepResult
from src.models.dreamer_ac import DreamerActor, DreamerCritic, lambda_returns
from src.models.policy_network import get_best_device
from src.models.world_model import RSSMState, WorldModel, WorldModelConfig
from src.training.replay_buffer import SequentialReplayBuffer
from src.utils.reward_functions import build_reward_function


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class DreamerConfig:
    """All Dreamer-side knobs in one place. Profile YAML maps keys
    under `dreamer:` to fields here."""

    # Data collection
    num_workers: int = 16
    replay_capacity: int = 200_000
    train_every_steps: int = 16
    warmup_steps: int = 5_000
    # World-model training
    batch_size: int = 16
    seq_len: int = 64
    wm_lr: float = 3e-4
    wm_grad_clip: float = 1.0
    # Actor-critic training
    imag_horizon: int = 15
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    entropy_coef: float = 3e-4
    discount: float = 0.99
    lambd: float = 0.95
    target_tau: float = 0.02
    # World-model architecture knobs (passed through to WorldModelConfig).
    latent_kind: str = "categorical"
    stoch_dim: int = 32
    stoch_classes: int = 32
    deter_dim: int = 256
    kl_free_nats: float = 1.0
    # Bookkeeping
    checkpoint_every_gens: int = 50
    log_every_steps: int = 100
    resume: bool = True


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------


class DreamerTrainer:
    """End-to-end world-model trainer.

    Construct with the same `(rom_path, game_profile, …)` shape as the
    classic Trainer; `main.py` dispatches based on
    `profile['training_mode']`. The trainer owns the pool, replay
    buffer, world model, actor, critic, and metrics emission.
    """

    def __init__(
        self,
        rom_path: str,
        game_profile: dict,
        num_instances: int,
        frame_sink=None,
        metrics_queue=None,
        env_spec: str = "nes_core:NESEnvironment",
        start_state_path: Optional[str] = None,
        max_episode_steps: int = 100_000,
        seed: Optional[int] = None,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.rom_path = rom_path
        self.game_profile = dict(game_profile)
        self.num_instances = num_instances
        self.env_spec = env_spec
        self.start_state_path = start_state_path
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self._frame_sink = frame_sink
        self._metrics_queue = metrics_queue

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

        self.action_space = game_profile.get("action_space", [])
        self.num_actions = len(self.action_space)
        if self.num_actions == 0:
            raise ValueError("Game profile must define a non-empty action_space")
        self._bitmask_table = self._build_bitmask_table()
        self.frame_skip = int(game_profile.get("frame_skip", 16))

        # Read Dreamer config from the profile, falling back to defaults.
        d_cfg = game_profile.get("dreamer", {}) or {}
        self.cfg = DreamerConfig(
            num_workers=num_instances,
            replay_capacity=int(d_cfg.get("replay_capacity", 200_000)),
            train_every_steps=int(d_cfg.get("train_every_steps", 16)),
            warmup_steps=int(d_cfg.get("warmup_steps", 5_000)),
            batch_size=int(d_cfg.get("batch_size", 16)),
            seq_len=int(d_cfg.get("seq_len", 64)),
            wm_lr=float(d_cfg.get("wm_lr", 3e-4)),
            wm_grad_clip=float(d_cfg.get("wm_grad_clip", 1.0)),
            imag_horizon=int(d_cfg.get("imag_horizon", 15)),
            actor_lr=float(d_cfg.get("actor_lr", 8e-5)),
            critic_lr=float(d_cfg.get("critic_lr", 8e-5)),
            entropy_coef=float(d_cfg.get("entropy_coef", 3e-4)),
            discount=float(d_cfg.get("discount", 0.99)),
            lambd=float(d_cfg.get("lambd", 0.95)),
            target_tau=float(d_cfg.get("target_tau", 0.02)),
            latent_kind=str(d_cfg.get("latent_kind", "categorical")),
            stoch_dim=int(d_cfg.get("stoch_dim", 32)),
            stoch_classes=int(d_cfg.get("stoch_classes", 32)),
            deter_dim=int(d_cfg.get("deter_dim", 256)),
            kl_free_nats=float(d_cfg.get("kl_free_nats", 1.0)),
            checkpoint_every_gens=int(d_cfg.get("checkpoint_every_gens", 50)),
            log_every_steps=int(d_cfg.get("log_every_steps", 100)),
            resume=bool(d_cfg.get("resume", True)),
        )

        self.device = get_best_device()
        log.info("DreamerTrainer device: %s", self.device)

        # Build world model + actor + critic. Sized from the action
        # space so the actor's output dim matches.
        wm_cfg = WorldModelConfig(
            obs_channels=4,
            obs_size=84,
            num_actions=self.num_actions,
            latent_kind=self.cfg.latent_kind,
            stoch_dim=self.cfg.stoch_dim,
            stoch_classes=self.cfg.stoch_classes,
            deter_dim=self.cfg.deter_dim,
            kl_free_nats=self.cfg.kl_free_nats,
        )
        self.wm = WorldModel(wm_cfg).to(self.device)
        latent_dim = wm_cfg.deter_dim + wm_cfg.stoch_flat
        self.actor = DreamerActor(latent_dim, self.num_actions).to(self.device)
        self.critic = DreamerCritic(latent_dim).to(self.device)

        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=self.cfg.wm_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(
            self.critic.online.parameters(), lr=self.cfg.critic_lr
        )

        self.replay = SequentialReplayBuffer(
            capacity=self.cfg.replay_capacity,
            obs_shape=(4, 84, 84),
            seed=seed,
        )

        self.pool: Optional[RustPool] = None
        self._stackers: list[FrameStacker] = []
        self._reward_fns: list = []
        # Per-worker RSSM state for online action selection. Reset on
        # episode boundaries.
        self._rssm_states: list[RSSMState] = []
        self._prev_actions = np.zeros(num_instances, dtype=np.int64)

        # Bookkeeping for metrics + checkpointing.
        self._running = False
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.dreamer_ckpt_dir = self.checkpoint_dir / "dreamer"
        self.dreamer_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.checkpoint_dir / "metrics.jsonl"
        # NB: don't truncate the metrics log if we plan to resume — the
        # dashboard's history is the only thing the user can see across
        # sessions. Truncate only on a fresh start (no checkpoint to
        # resume from).
        self._step_count = 0
        self._train_count = 0
        self._wall_start = time.time()

    # ----- bitmask helpers (mirror Trainer for action dispatch) ------

    def _build_bitmask_table(self) -> list[int]:
        # nes-py bitmask order. Profile YAMLs use names like "right",
        # "right_a", etc. — same convention used by the PPO trainer.
        BUTTON_BITS = {
            "right": 0x80, "left": 0x40, "down": 0x20, "up": 0x10,
            "start": 0x08, "select": 0x04, "b": 0x02, "a": 0x01,
            "noop": 0x00,
        }
        table: list[int] = []
        for combo in self.action_space:
            mask = 0
            for tok in combo.split("_"):
                tok = tok.lower().strip()
                if tok in BUTTON_BITS:
                    mask |= BUTTON_BITS[tok]
            table.append(mask)
        return table

    def _action_to_bitmask(self, idx: int) -> int:
        return self._bitmask_table[int(idx)]

    # ----- main entry -------------------------------------------------

    def stop(self) -> None:
        self._running = False

    # ----- checkpointing ---------------------------------------------

    def _save_checkpoint(self) -> Path:
        """Atomic save of world_model + actor + critic + step counters.

        Atomic = write to `.tmp`, fsync, rename. A SIGINT mid-write
        therefore cannot leave a half-serialized file behind that would
        crash the next resume. Keeps only the last 5 checkpoints to
        bound disk usage on overnight runs.
        """
        path = self.dreamer_ckpt_dir / f"dreamer_{self._train_count:06d}.pt"
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "world_model": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_online": self.critic.online.state_dict(),
            "critic_target": self.critic.target.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "step_count": self._step_count,
            "train_count": self._train_count,
            "cfg": self.cfg.__dict__,
            "num_actions": self.num_actions,
        }
        torch.save(payload, str(tmp))
        try:
            import os
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        import os
        os.replace(str(tmp), str(path))

        # Prune to last 5.
        existing = sorted(self.dreamer_ckpt_dir.glob("dreamer_*.pt"))
        for old in existing[:-5]:
            try:
                old.unlink()
            except OSError:
                pass
        log.info("Saved Dreamer checkpoint: %s", path.name)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        """Restore from a previous run."""
        log.info("Loading Dreamer checkpoint: %s", path)
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.wm.load_state_dict(payload["world_model"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.online.load_state_dict(payload["critic_online"])
        self.critic.target.load_state_dict(payload["critic_target"])
        self.opt_wm.load_state_dict(payload["opt_wm"])
        self.opt_actor.load_state_dict(payload["opt_actor"])
        self.opt_critic.load_state_dict(payload["opt_critic"])
        self._step_count = int(payload.get("step_count", 0))
        self._train_count = int(payload.get("train_count", 0))

    def _find_latest_checkpoint(self) -> Optional[Path]:
        files = sorted(self.dreamer_ckpt_dir.glob("dreamer_*.pt"))
        return files[-1] if files else None

    def _dump_reconstruction(self) -> Optional[Path]:
        """Pull a tiny batch from replay, run encoder→RSSM→decoder, and
        save the (original, reconstructed) frames as a single numpy
        array. The dashboard renders this via QImage so no image library
        is needed in the trainer.

        Output layout is a `(2, 4, 84, 84)` uint8 array — top row is
        the four most-recent frames from one trajectory, bottom row is
        the world-model's reconstruction of those same frames. Used to
        eyeball whether the world model has learned anything beyond a
        blurry mean.
        """
        if len(self.replay) < self.cfg.seq_len:
            return None
        try:
            obs_np, act_np, _, _ = self.replay.sample(
                batch_size=1, seq_len=self.cfg.seq_len,
            )
        except RuntimeError:
            return None

        obs = torch.from_numpy(obs_np).to(self.device).float() / 255.0
        actions_oh = F.one_hot(
            torch.from_numpy(act_np).to(self.device).long(),
            num_classes=self.num_actions,
        ).float()
        with torch.no_grad():
            B, L = obs.shape[:2]
            flat_obs = obs.view(B * L, *obs.shape[2:])
            embeds = self.wm.encoder(flat_obs).view(B, L, -1)
            states = self.wm.rssm.observe(embeds, actions_oh)
            # Use the LAST step's posterior latent for reconstruction —
            # that's the single frame the user most wants to see (most
            # recent observation).
            latent = self.wm.latent(states[-1])  # (B=1, latent_dim)
            recon = self.wm.decoder(latent)      # (1, 4, 84, 84)

        # Original = the last 4-stack from the sampled sequence.
        orig_uint8 = obs_np[0, -1]  # (4, 84, 84)
        recon_uint8 = (
            recon.detach().cpu().numpy()[0].clip(0, 1) * 255.0
        ).astype(np.uint8)  # (4, 84, 84)
        # Stack into one (2, 4, 84, 84) — top row originals, bottom recon.
        out_arr = np.stack([orig_uint8, recon_uint8], axis=0)

        out_path = self.dreamer_ckpt_dir / "reconstruction.npy"
        # Atomic write so the dashboard never reads a half-written file.
        # Write through an open file handle so np.save doesn't auto-append
        # `.npy` to our `.tmp` path (which would skip the rename target).
        tmp = out_path.with_suffix(".tmp")
        with open(str(tmp), "wb") as f:
            np.save(f, out_arr, allow_pickle=False)
        import os
        os.replace(str(tmp), str(out_path))
        return out_path

    def run(self, num_steps: int = 1_000_000) -> None:
        """Main training loop. Runs until `num_steps` env steps are
        taken or `stop()` is called."""
        from src.training.trainer import _make_pool  # avoid circular import

        # Resume if requested. When resuming, leave metrics.jsonl
        # untouched so the dashboard's history persists. Truncate
        # fresh-start runs.
        latest = self._find_latest_checkpoint() if self.cfg.resume else None
        if latest is not None:
            try:
                self._load_checkpoint(latest)
                log.info(
                    "Resumed Dreamer at step=%d train=%d",
                    self._step_count, self._train_count,
                )
            except Exception as exc:
                log.warning("Could not resume from %s (%s); starting fresh", latest, exc)
                self._metrics_path.write_text("")
        else:
            self._metrics_path.write_text("")

        self._running = True
        self.pool = _make_pool(
            env_spec=self.env_spec,
            rom_path=self.rom_path,
            num_workers=self.num_instances,
            frame_skip=self.frame_skip,
            start_state_path=self.start_state_path,
            seed=self.seed,
        )
        self.pool.start()

        self._stackers = [FrameStacker() for _ in range(self.num_instances)]
        self._reward_fns = [
            build_reward_function(self.game_profile) for _ in range(self.num_instances)
        ]
        self._rssm_states = [
            self.wm.rssm.initial_state(1, self.device)
            for _ in range(self.num_instances)
        ]
        self._prev_actions[:] = 0
        self._episode_steps = np.zeros(self.num_instances, dtype=np.int64)

        try:
            results = self.pool.reset_all()
            stacked = np.zeros(
                (self.num_instances, 4, 84, 84), dtype=np.uint8
            )
            for i, r in enumerate(results):
                stacked[i] = self._stackers[i].reset(r.frame, r.preprocessed)
                self._reward_fns[i].reset()

            while self._running and self._step_count < num_steps:
                stacked = self._collect_step(stacked)
                self._step_count += 1
                if (
                    self._step_count >= self.cfg.warmup_steps
                    and self._step_count % self.cfg.train_every_steps == 0
                ):
                    self._train_step()
                if self._step_count % self.cfg.log_every_steps == 0:
                    self._emit_metrics_step()
                # Train-step-indexed checkpointing — happens at most
                # once per train step regardless of env-step pacing.
                if (
                    self._train_count > 0
                    and self._train_count % self.cfg.checkpoint_every_gens == 0
                    and getattr(self, "_last_ckpt_train", -1) != self._train_count
                ):
                    try:
                        self._save_checkpoint()
                    except Exception as exc:
                        log.warning("checkpoint save failed: %s", exc)
                    self._last_ckpt_train = self._train_count
                # Reconstruction-image dump for the dashboard. Cheaper
                # than a checkpoint (one tiny forward pass), so do it
                # more frequently — every 10 train steps gives a useful
                # animation rate without measurable perf cost.
                if (
                    self._train_count > 0
                    and self._train_count % 10 == 0
                    and getattr(self, "_last_recon_train", -1) != self._train_count
                ):
                    try:
                        self._dump_reconstruction()
                    except Exception as exc:
                        log.debug("reconstruction dump skipped: %s", exc)
                    self._last_recon_train = self._train_count
        finally:
            # Save one final checkpoint on graceful shutdown so a
            # Stop button click doesn't lose the last train-cycle's
            # progress.
            if self._train_count > 0:
                try:
                    self._save_checkpoint()
                except Exception as exc:
                    log.warning("final checkpoint save failed: %s", exc)
            if self.pool is not None:
                self.pool.shutdown()
            self.pool = None
            self._frame_sink = None

    # ----- collection -------------------------------------------------

    def _collect_step(self, stacked: np.ndarray) -> np.ndarray:
        """One step of action selection + env step + replay append.

        Returns the next stacked-obs tensor (numpy, uint8) for use on
        the next iteration.
        """
        # Encode current obs via the world model. Convert uint8 → float
        # in [0,1] on device. Single-step RSSM advance using the
        # previous action so the latent reflects history up to "now".
        obs_t = (
            torch.from_numpy(stacked).to(self.device).float() / 255.0
        )
        with torch.no_grad():
            embed = self.wm.encoder(obs_t)  # (N, F)
            # Advance one step. observe() expects (B, L, ...) so add
            # the seq dim.
            prev_action_oh = (
                F.one_hot(
                    torch.from_numpy(self._prev_actions).to(self.device),
                    num_classes=self.num_actions,
                ).float()
            )
            new_states: list[RSSMState] = []
            for i in range(self.num_instances):
                states_i = self.wm.rssm.observe(
                    embed[i:i + 1].unsqueeze(1),
                    prev_action_oh[i:i + 1].unsqueeze(1),
                    init_state=self._rssm_states[i],
                )
                new_states.append(states_i[-1])
            self._rssm_states = new_states

            latents = torch.stack(
                [self.wm.latent(s) for s in new_states], dim=0
            )  # (N, latent_dim)
            actions, _logp, _ent = self.actor.sample(latents)
            action_idx = actions.argmax(dim=-1).cpu().numpy()

        # Step the pool. Use bitmask dispatch identical to the PPO path.
        bitmasks = np.array(
            [self._action_to_bitmask(int(a)) for a in action_idx],
            dtype=np.uint8,
        )
        results: list[StepResult] = self.pool.step_all(bitmasks)
        if self._frame_sink is not None:
            try:
                self._frame_sink(results)
            except Exception:
                pass

        next_stacked = np.empty_like(stacked)
        for i, r in enumerate(results):
            next_stacked[i] = self._stackers[i].push(r.frame, r.preprocessed)
            reward, rew_done, _level = self._reward_fns[i].compute(
                r.ram_snapshot,
                action=int(bitmasks[i]),
            )
            done_flag = bool(r.done) or rew_done or (
                self._episode_steps[i] >= self.max_episode_steps
            )
            self.replay.add(
                obs=stacked[i],
                action=int(action_idx[i]),
                reward=float(reward),
                done=done_flag,
            )
            self._episode_steps[i] += 1
            if done_flag:
                # End-of-episode: reset the worker's stacker, RSSM
                # state, and reward function state. The pool itself
                # auto-resets on `done`.
                self._stackers[i].reset(r.frame, r.preprocessed)
                self._rssm_states[i] = self.wm.rssm.initial_state(1, self.device)
                self._reward_fns[i].reset()
                self._episode_steps[i] = 0

        self._prev_actions[:] = action_idx
        return next_stacked

    # ----- training ---------------------------------------------------

    def _train_step(self) -> None:
        """One world-model batch + one actor-critic batch."""
        wm_stats = self._train_world_model()
        ac_stats = self._train_actor_critic()
        self.critic.update_target(self.cfg.target_tau)
        self._train_count += 1
        self._last_wm_stats = wm_stats
        self._last_ac_stats = ac_stats

    def _train_world_model(self) -> dict[str, float]:
        obs_np, act_np, rew_np, done_np = self.replay.sample(
            batch_size=self.cfg.batch_size,
            seq_len=self.cfg.seq_len,
        )
        # uint8 → float [0,1] on device. The world-model decoder
        # reconstructs into the same range, so MSE is well-scaled.
        obs = torch.from_numpy(obs_np).to(self.device).float() / 255.0
        # Symlog transform on rewards before regression — matches the
        # paper's high-dynamic-range handling.
        rewards = torch.from_numpy(rew_np).to(self.device).float()
        rewards = torch.sign(rewards) * torch.log1p(rewards.abs())
        continues = 1.0 - torch.from_numpy(done_np).to(self.device).float()
        actions_oh = F.one_hot(
            torch.from_numpy(act_np).to(self.device).long(),
            num_classes=self.num_actions,
        ).float()

        loss, stats = self.wm.loss(obs, actions_oh, rewards, continues)
        self.opt_wm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.wm_grad_clip)
        self.opt_wm.step()
        stats["wm_total"] = float(loss.detach().item())
        return stats

    def _train_actor_critic(self) -> dict[str, float]:
        """Imagine `H` steps from a fresh batch of posterior latents,
        compute λ-returns, regress critic, climb actor."""
        obs_np, act_np, _, _ = self.replay.sample(
            batch_size=self.cfg.batch_size,
            seq_len=self.cfg.seq_len,
        )
        obs = torch.from_numpy(obs_np).to(self.device).float() / 255.0
        actions_oh = F.one_hot(
            torch.from_numpy(act_np).to(self.device).long(),
            num_classes=self.num_actions,
        ).float()
        B, L = obs.shape[:2]

        # Compute posterior latents for each step in the sequence
        # (through the same RSSM forward used in WorldModel.loss).
        # No grads through the world model — it was just trained.
        with torch.no_grad():
            flat_obs = obs.view(B * L, *obs.shape[2:])
            embeds = self.wm.encoder(flat_obs).view(B, L, -1)
            states = self.wm.rssm.observe(embeds, actions_oh)
        # Flatten the time dim into the batch dim so each (B, t) is
        # treated as an independent imagination start.
        start_state = RSSMState(
            deter=torch.cat([s.deter for s in states], dim=0),
            stoch=torch.cat([s.stoch for s in states], dim=0),
        )

        # Imagine H steps. The actor decides each step using the
        # current latent; gradients flow through reparameterized stoch.
        H = self.cfg.imag_horizon
        latents_seq: list[torch.Tensor] = []
        log_probs_seq: list[torch.Tensor] = []
        entropy_seq: list[torch.Tensor] = []
        actions_seq: list[torch.Tensor] = []

        latent_t = self.wm.latent(start_state)
        cur_state = start_state
        for _ in range(H):
            action_oh, log_prob, entropy = self.actor.sample(latent_t)
            # Step through the prior — we have no observations in
            # imagination space.
            new_states = self.wm.rssm.imagine(cur_state, action_oh.unsqueeze(1))
            cur_state = new_states[-1]
            latent_t = self.wm.latent(cur_state)
            latents_seq.append(latent_t)
            log_probs_seq.append(log_prob)
            entropy_seq.append(entropy)
            actions_seq.append(action_oh)

        # (B*L, H, latent_dim)
        latents = torch.stack(latents_seq, dim=1)
        log_probs = torch.stack(log_probs_seq, dim=1)
        entropies = torch.stack(entropy_seq, dim=1)

        # Predict reward + continue along the imagined trajectory.
        flat_lat = latents.view(-1, latents.size(-1))
        rewards_pred = self.wm.reward_head(flat_lat).view(B * L, H)
        continue_logits = self.wm.continue_head(flat_lat).view(B * L, H)
        continues_pred = torch.sigmoid(continue_logits)

        # Critic value estimates (with grad for the actor's expected-
        # return objective and detached for the critic regression
        # target).
        values = self.critic(flat_lat).view(B * L, H)
        with torch.no_grad():
            target_values = self.critic.target_value(flat_lat).view(B * L, H)
            bootstrap = self.critic.target_value(latents[:, -1])

        returns = lambda_returns(
            rewards=rewards_pred.detach(),
            values=target_values,
            bootstrap=bootstrap,
            continues=continues_pred.detach(),
            discount=self.cfg.discount,
            lambd=self.cfg.lambd,
        )

        # Critic regresses online value to detached lambda-returns.
        critic_loss = F.mse_loss(values, returns.detach())
        self.opt_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.online.parameters(), 1.0)
        self.opt_critic.step()

        # Actor maximizes returns (with detached advantage as the
        # baseline) plus an analytic categorical entropy bonus for
        # exploration. `entropies` is the proper Shannon entropy
        # `-Σ p log p` returned by `DreamerActor.sample()`, not the
        # `-log p(a)` proxy: the difference matters because the proxy
        # only sees the chosen action's surprise, while entropy reflects
        # the full distribution. Subtracting `coef * entropy` from the
        # loss means we maximize entropy.
        advantage = (returns - values.detach())
        actor_loss = -(log_probs * advantage.detach()).mean()
        entropy_bonus = entropies.mean()
        actor_total = actor_loss - self.cfg.entropy_coef * entropy_bonus

        self.opt_actor.zero_grad()
        actor_total.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_entropy": float(entropy_bonus.detach().item()),
            "imag_return_mean": float(returns.detach().mean().item()),
            "imag_value_mean": float(values.detach().mean().item()),
        }

    # ----- metrics ---------------------------------------------------

    def _emit_metrics_step(self) -> None:
        """Emit one metrics row roughly every `log_every_steps` env
        steps. Generation index is the train-step count so the
        dashboard's existing x-axis (Generation) maps cleanly."""
        wm_stats = getattr(self, "_last_wm_stats", {}) or {}
        ac_stats = getattr(self, "_last_ac_stats", {}) or {}
        elapsed = time.time() - self._wall_start
        rec = {
            "generation": self._train_count,
            "step": self._step_count,
            "elapsed": elapsed,
            "replay_size": len(self.replay),
            "replay_capacity": self.cfg.replay_capacity,
            "best_fitness": ac_stats.get("imag_return_mean", 0.0),
            "avg_fitness": ac_stats.get("imag_value_mean", 0.0),
            "stage": "dreamer",
            "success_rate": 0.0,
            "episodes": 0,
            **{f"{k}": v for k, v in wm_stats.items()},
            **{f"{k}": v for k, v in ac_stats.items()},
            "timestamp": time.time(),
        }
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self._metrics_queue is not None:
            try:
                self._metrics_queue.put_nowait(rec)
            except Exception:
                pass
        log.info(
            "[dreamer] step %d train %d replay %d/%d | wm %.3f / ac %.3f / V %.2f",
            self._step_count, self._train_count,
            len(self.replay), self.cfg.replay_capacity,
            wm_stats.get("wm_total", float("nan")),
            ac_stats.get("actor_loss", float("nan")),
            ac_stats.get("imag_value_mean", float("nan")),
        )
