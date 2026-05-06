"""
Behavioral cloning from a recorded play trace.

The play window writes one byte per emulator frame (frame_skip=1) — each
byte is the NES button bitmask held that frame. We turn that sequence into
a supervised (state, action-index) dataset and pre-train the policy network
on it. The pre-trained weights then seed every genome in the population, so
the GA starts from "roughly imitates the human" instead of random.

Even an imperfect demo helps: the agent picks up the "press Start on the
title", "navigate menus with A", and general button-use priors that a pure
random walk takes thousands of generations to stumble onto.

Pipeline:
  1. build_dataset(rom, demo, action_space [, start_state]) runs a single
     nes_core.NESEnvironment frame-by-frame, maintains a FrameStacker, and
     at every frame_skip-boundary emits a (stacked_state, action_index)
     pair. The raw bitmask is mapped to the nearest action in action_space
     by Hamming distance (exact match preferred).
  2. pretrain(network, dataset, ...) runs standard cross-entropy gradient
     descent on the dataset.
  3. The trainer calls both and seeds GA genomes from the result.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import nes_core

from src.emulation.frame_utils import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_NOOP,
    BUTTON_RIGHT,
    BUTTON_SELECT,
    BUTTON_START,
    BUTTON_UP,
    FrameStacker,
)

NESEnvironment = nes_core.NESEnvironment


log = logging.getLogger(__name__)


_NAME_TO_BIT = {
    "NOOP": BUTTON_NOOP,
    "A": BUTTON_A,
    "B": BUTTON_B,
    "up": BUTTON_UP,
    "down": BUTTON_DOWN,
    "left": BUTTON_LEFT,
    "right": BUTTON_RIGHT,
    "start": BUTTON_START,
    "select": BUTTON_SELECT,
}


def _action_space_bitmasks(action_space: list) -> list[int]:
    """Pre-compute the bitmask of each action_space entry once."""
    out: list[int] = []
    for entry in action_space:
        mask = 0
        for name in entry:
            mask |= _NAME_TO_BIT[name]
        out.append(mask)
    return out


def _nearest_action_index(raw_mask: int, action_bitmasks: list[int]) -> int:
    """Map a recorded bitmask to the closest action_space entry.

    Prefer exact match; fall back to Hamming distance on the 8-bit button
    register. Ties break to the lowest index (which is typically NOOP,
    the safest default)."""
    for idx, m in enumerate(action_bitmasks):
        if m == raw_mask:
            return idx
    best_idx, best_dist = 0, 9
    for idx, m in enumerate(action_bitmasks):
        d = bin((m ^ raw_mask) & 0xFF).count("1")
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def build_dataset(
    rom_path: str | Path,
    demo_path,
    action_space: list,
    frame_skip: int = 4,
    max_pairs: int = 20000,
    reward_fn=None,
    tile_extractor=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce (states, actions, rewards) tensors from a `.state.bin` recording.

    states:  (N, 4, 84, 84) uint8 — ready for division by 255 at train time
    actions: (N,) int64 — indices into `action_space`
    rewards: (N,) float32 — per-sample reward as the game-specific reward
        function scored it WHILE the expert was driving. Zero everywhere
        when `reward_fn` is None (caller wants uniform-weight BC).

    When `reward_fn` is supplied (expected to be a freshly-constructed
    `RewardFunction` instance — this function `reset()`s it), we call
    `reward_fn.compute(ram, action=mask)` at every emulator step and
    accumulate its output across the frame_skip window, so the emitted
    sample's reward is "what the expert actually earned during the chunk
    that produced this (state, action)". Downstream code uses that to
    weight imitation: actions taken during high-return moments matter
    more than idle or mistake moments.

    The recording is always replayed from a cold reset — the demo IS the
    action sequence from power-on, so applying a start-state before it
    would put the env at the wrong place and the captured states wouldn't
    match what the recording's actions were originally taken under. The
    trainer uses start_state for runtime resets; BC uses the raw demo.

    `demo_path` may be a single path OR an iterable of paths. When it's
    a list, each demo is replayed independently from its own cold-reset
    env, and the resulting (state, action, reward) tuples are
    concatenated. This is the only path that gives BC enough state
    coverage to matter on a long game like Zelda — a single 5-min
    recording is ~1k pairs, but six varied playthroughs is closer to
    10k, which lets the policy generalise instead of memorising.
    """
    # Normalise to a list of paths so the rest of the function is
    # multi-demo-native.
    if isinstance(demo_path, (str, Path)):
        demo_list = [Path(demo_path)]
    else:
        demo_list = [Path(p) for p in demo_path]

    action_bitmasks = _action_space_bitmasks(action_space)

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []

    from collections import Counter

    for demo_idx, single_demo in enumerate(demo_list):
        demo_bytes = single_demo.read_bytes()
        if not demo_bytes:
            log.warning("BC demo %s is empty; skipping.", single_demo)
            continue

        # Fresh reward fn per demo — its internal "first step" anchors
        # bake in the demo's own initial state, not state from the
        # previous demo.
        if reward_fn is not None:
            reward_fn.reset()

        # Each demo gets its own cold-boot env so (state, action) pairs
        # are coherent within a single replay.
        env = NESEnvironment(rom_path=str(rom_path), frame_skip=1)
        first_frame = env.reset()
        # Pixel mode uses a frame stacker; tile mode decodes RAM
        # directly via the supplied extractor. Only one is active per
        # demo replay; the other variable stays at its default.
        if tile_extractor is None:
            stacker = FrameStacker(stack_size=4)
            stacker.reset(first_frame)
        else:
            stacker = None

        window_actions: list[int] = []
        window_reward = 0.0
        log.info(
            "BC demo %d/%d: %s (%d frames)",
            demo_idx + 1, len(demo_list), single_demo.name, len(demo_bytes),
        )
        try:
            for frame_idx, raw_byte in enumerate(demo_bytes):
                raw = int(raw_byte)
                window_actions.append(raw)
                frame, done = env.step(raw)
                # In pixel mode, push into the stacker each NES frame so
                # the stack always reflects the latest 4. In tile mode,
                # we don't precompute per-frame state — we'll decode RAM
                # at the chunk boundary below since RAM is already
                # final-state by then.
                if tile_extractor is None:
                    current_stack = stacker.push(frame)
                # Reward function reads RAM regardless of mode.
                ram_snapshot = env.get_ram_range(0, 2048).tobytes() if (
                    reward_fn is not None or tile_extractor is not None
                ) else None
                if reward_fn is not None:
                    r, rew_done, _ = reward_fn.compute(ram_snapshot, action=raw)
                    window_reward += r
                    if rew_done:
                        # Treat reward-signalled termination the same as env
                        # termination — wrap the chunk, cold-reset, move on.
                        done = True

                if (frame_idx + 1) % frame_skip == 0:
                    # Majority-rule the action across the frame_skip window
                    # so one noisy single-frame flick doesn't pollute the
                    # label.
                    most_common = Counter(window_actions).most_common(1)[0][0]
                    action_idx = _nearest_action_index(most_common, action_bitmasks)
                    if tile_extractor is not None:
                        # Tile mode: decode the chunk's final RAM state.
                        # `ram_snapshot` is already the latest because we
                        # always re-read it for the reward function above
                        # (or compute it fresh on the boundary).
                        if ram_snapshot is None:
                            ram_snapshot = env.get_ram_range(0, 2048).tobytes()
                        states.append(tile_extractor.extract(ram_snapshot).copy())
                    else:
                        states.append(current_stack.copy())
                    actions.append(action_idx)
                    rewards.append(window_reward)
                    window_actions.clear()
                    window_reward = 0.0
                    if len(states) >= max_pairs:
                        break
                if done:
                    # Expected during recorded menu/intro: cold reset and
                    # continue. This only happens for broken recordings.
                    env.reset()
                    if tile_extractor is None:
                        stacker.reset(env.get_frame())
                    window_actions.clear()
                    window_reward = 0.0
                    if reward_fn is not None:
                        reward_fn.reset()
        finally:
            env.close()
        if len(states) >= max_pairs:
            break  # done across all demos — pair budget exhausted

    log.info(
        "BC dataset built: %d demos, %d (state,action) pairs, "
        "%d total demo bytes",
        len(demo_list), len(states), sum(p.stat().st_size for p in demo_list if p.exists()),
    )
    if states:
        states_t = torch.from_numpy(np.stack(states, axis=0))
    elif tile_extractor is not None:
        # Empty fallback shape matches tile-mode policy input dim.
        states_t = torch.zeros((0, tile_extractor.feature_dim), dtype=torch.int8)
    else:
        states_t = torch.zeros((0, 4, 84, 84), dtype=torch.uint8)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    return states_t, actions_t, rewards_t


def _compute_return_weights(
    rewards: torch.Tensor,
    gamma: float = 0.99,
    awr_beta: float = 1.0,
    weight_clip: tuple[float, float] = (0.1, 10.0),
) -> torch.Tensor:
    """Turn a per-step reward tensor into a per-sample imitation weight.

    Pipeline:
      1. Discounted returns G_t = r_t + γ · G_{t+1}.
      2. Normalise: (G - G.mean()) / (G.std() + ε).
      3. Advantage-weighted regression: w_t = exp(G_norm_t / β).
      4. Clip to `weight_clip` so no single sample dominates or vanishes.

    The result is strictly positive — we never flip the sign of the
    cross-entropy loss. Bad moments in the demo get down-weighted but
    still contribute; good moments are amplified.
    """
    n = rewards.numel()
    if n == 0:
        return rewards.clone()
    # Discounted return, walked backwards in numpy (cheap for ≤ 20k steps).
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(rewards[t].item()) + gamma * running
        returns[t] = running
    mean = returns.mean()
    std = returns.std()
    if float(std.item()) < 1e-8:
        # Demo had flat reward — fall back to uniform imitation.
        return torch.ones_like(returns)
    g_norm = (returns - mean) / (std + 1e-8)
    weights = torch.exp(g_norm / awr_beta)
    lo, hi = weight_clip
    return torch.clamp(weights, min=lo, max=hi)


def pretrain(
    network,
    dataset,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    use_reward_weighting: bool = True,
    # Fraction held out as a validation split for overfitting checks.
    # Default 0.0 keeps backwards-compatible behavior for any caller
    # (and tests) that don't ask for one. Trainer enables it explicitly.
    val_fraction: float = 0.0,
    # When True (default), divide states by 255 — appropriate for
    # uint8 pixel observations. Tile-mode states are small signed
    # ints (-128..127) and need no normalization; the trainer sets
    # this False for tile policies so the LayerNorm + first Linear
    # learn the right scale on raw values.
    normalize_obs: bool = True,
) -> float:
    """Supervised cross-entropy pre-training. Returns final TRAIN epoch loss.

    `dataset` is `(states, actions)` or `(states, actions, rewards)`. When
    rewards are provided AND `use_reward_weighting` is on, each sample's
    loss is scaled by an AWR-style weight derived from its discounted
    return — imitate the expert's best moments hard, their worst moments
    only gently. Falls back to uniform cross-entropy otherwise.

    A small held-out validation split (default 10 %) is logged each epoch
    so divergence between train and val loss surfaces overfitting on the
    demo. The split is deterministic per call (seeded shuffle), so
    repeated runs on the same dataset compare apples-to-apples.

    Shuffles the train split each epoch.
    """
    if len(dataset) == 3:
        states, actions, rewards = dataset
    else:
        states, actions = dataset
        rewards = None
    n = states.shape[0]
    if n == 0:
        log.warning("BC dataset is empty; skipping pretrain.")
        return float("inf")

    # Deterministic train/val split.
    val_n = max(1, int(n * val_fraction)) if n >= 20 else 0
    g = torch.Generator(device="cpu").manual_seed(0)
    perm_all = torch.randperm(n, generator=g)
    val_idx = perm_all[:val_n]
    train_idx = perm_all[val_n:]
    train_n = int(train_idx.numel())

    weighted = use_reward_weighting and rewards is not None and rewards.numel() == n
    if weighted:
        sample_weights = _compute_return_weights(rewards).to(device)
        log.info(
            "BC weighted-imitation enabled: reward range=[%.2f, %.2f], "
            "weight range=[%.3f, %.3f]",
            float(rewards.min().item()),
            float(rewards.max().item()),
            float(sample_weights.min().item()),
            float(sample_weights.max().item()),
        )

    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    def _val_loss() -> float:
        if val_n == 0:
            return float("nan")
        network.eval()
        with torch.no_grad():
            s = states[val_idx].to(device).float()
            if normalize_obs:
                s = s.div_(255.0)
            a = actions[val_idx].to(device)
            logits = network(s)
            return float(F.cross_entropy(logits, a).item())

    last_loss = float("inf")
    for epoch in range(epochs):
        network.train()
        perm = train_idx[torch.randperm(train_n)]
        losses = []
        for i in range(0, train_n, batch_size):
            idx = perm[i : i + batch_size]
            # Skip a singleton final batch — cross_entropy on n=1 is
            # high-variance noise that's worse than dropping the sample.
            if idx.numel() < 2:
                continue
            s = states[idx].to(device).float()
            if normalize_obs:
                s = s.div_(255.0)
            a = actions[idx].to(device)
            logits = network(s)
            if weighted:
                per_sample = F.cross_entropy(logits, a, reduction="none")
                w = sample_weights[idx]
                loss = (per_sample * w).sum() / (w.sum() + 1e-8)
            else:
                loss = F.cross_entropy(logits, a)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        last_loss = sum(losses) / max(1, len(losses))
        v = _val_loss()
        log.info(
            "BC epoch %d/%d  train_loss=%.4f  val_loss=%.4f  n_train=%d  weighted=%s",
            epoch + 1, epochs, last_loss, v, train_n, weighted,
        )

    return last_loss


def seed_population_from_weights(
    population: list,
    source_state_dict: dict,
    noise_std: float = 0.01,
) -> None:
    """Copy pre-trained weights into every genome in `population`, adding a
    small dose of Gaussian noise so the GA has diversity to select over.
    The first genome gets the weights unchanged (elite seed)."""
    for idx, genome in enumerate(population):
        new_state: dict = {}
        for k, v in source_state_dict.items():
            if idx == 0:
                new_state[k] = v.detach().clone()
            else:
                new_state[k] = v.detach().clone() + torch.randn_like(v) * noise_std
        genome.state_dict = new_state
