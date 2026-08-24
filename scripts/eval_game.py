"""Per-game eval script: load latest checkpoint, run N episodes,
report clear rate + furthest stage reached.

Usage:
    python scripts/eval_game.py --game mario
    python scripts/eval_game.py --game mario --episodes 30

Reads the per-game checkpoint directory derived from the profile,
finds the highest-numbered `vanilla_ppo_iter_*.pt`, loads the
policy network, instantiates a single-worker pool from the same
profile, and runs N eval episodes (deterministic seed). Reports:
  - clear rate (fraction of episodes that ended with the game's
    success criterion satisfied)
  - max area-byte reached across episodes (proxy for "furthest
    stage")
  - mean episode return + length

Dispatches on the checkpoint's architecture family: a recurrent
(tile_gru) checkpoint is loaded into the recurrent policy and its GRU
hidden state is threaded through the step loop (reset on episode
boundaries), so a recurrent-trained agent is actually scored instead of
silently misloading into the stateless net.

Output is a single JSON line on stdout and a row appended to
`checkpoints/<game_slug>/eval.jsonl` for later scoreboarding.

Episodes run serially in one worker by default. `--eval-workers K` runs K
episodes concurrently in a K-worker pool with per-lane bookkeeping; see
`_run_episodes_parallel` and the `--eval-rng` note on why the parallel branch
needs a per-episode RNG derivation to stay reproducible.

The effective worker count — and ONLY that — picks the executor: an effective
1 is always `_run_episodes_serial` (the historical loop), anything wider is
always `_run_episodes_parallel`. Both honor both `--eval-rng` modes, so
`--eval-workers 1 --eval-rng per-episode` vs
`--eval-workers K --eval-rng per-episode` is a genuine serial-vs-parallel
differential — that is what the ROM-gated test in tests/test_eval_parallel.py
asserts on, and it is why the stub suite there can catch a bug in the parallel
branch's honest-protocol constants instead of applying it to both sides of its
own comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import FrameStacker, TileFeatureStacker  # noqa: E402
from src.models.policy_network import PolicyNetwork  # noqa: E402
from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, derive_checkpoint_dir, resolve_encoder,
)
from src.training.checkpointing import (  # noqa: E402
    find_latest_trained_checkpoint, find_playable_checkpoint,
)
from src.training.smb_sequential import (  # noqa: E402
    LevelClearTracker, SequentialTracker, level_label,
)

# Reuse the same logical → profile + ROM tables as the launcher so
# eval and train always agree on which files to read.
sys.path.insert(0, str(ROOT / "scripts"))
from train_game import DEFAULT_PROFILES, DEFAULT_ROMS, resolve_profile_path  # noqa: E402


# Pixel encoders the trainer's PolicyNetwork can build. resolve_encoder
# (the tile path) rejects these by design — they have their own CNN path
# in the trainer — so eval routes them through the pixel path below. Kept
# in lockstep with src/models/policy_network.PolicyNetwork's accepted
# encoders and the trainer's `_is_tile_mode = encoder in ("smb_tiles",)`
# split: any encoder that is neither a tile encoder nor one of these is
# genuinely unsupported.
PIXEL_ENCODERS: tuple[str, ...] = ("nature_dqn", "impala")

# How an action is drawn from the policy's logits. `greedy` is argmax — the
# only behavior this script ever had, and still the default, so every existing
# caller is byte-identical. `sampled` draws from softmax(logits / T), which is
# what the honest protocol actually mandates reporting ALONGSIDE greedy: a
# policy whose argmax clears a level while its own action distribution does
# not is a Brute, not a learner, and greedy-only eval cannot see the
# difference.
_ACTION_SELECT_MODES: tuple[str, ...] = ("greedy", "sampled")

# How the stochastic-eval randomness (Machado no-op starts, sticky repeats,
# sampled action draws) is derived.
#
#   shared-stream  the historical behavior and the DEFAULT: ONE
#                  `np.random.RandomState(eval_seed)` and ONE
#                  `torch.Generator(eval_seed)` are threaded through every
#                  episode in order. Episode i's stream position therefore
#                  depends on how many draws episodes 0..i-1 consumed, which
#                  depends on how long they ran — data-dependent, so episode i
#                  is NOT a function of (eval_seed, i) alone. Every receipt
#                  ever written by this script is a shared-stream receipt and
#                  reproduces bit-for-bit under this mode.
#
#   per-episode    each episode's numpy + torch streams are seeded from
#                  `np.random.SeedSequence([eval_seed, episode_index])`, so
#                  episode i is a pure function of (eval_seed, i). This is the
#                  ONLY mode in which the answer is invariant to how the
#                  episodes are distributed over workers — and therefore the
#                  only mode `--eval-workers > 1` can run a stochastic eval in.
#                  Its numbers are NOT comparable byte-for-byte to
#                  shared-stream receipts (same distribution, different draws).
#
# When the run draws no randomness at all (greedy + no sticky + no jitter) the
# two modes are provably identical because neither stream is ever touched;
# `run_consumes_randomness` is the predicate that decides that, and it is what
# lets `--eval-workers > 1` run under the default mode in that case.
#
# BOTH executors implement BOTH modes. The mode does not select an executor —
# the worker count does — so the per-episode mode at one worker is the serial
# loop with its streams re-seeded per episode, which is exactly the reference a
# K-worker run must reproduce.
_EVAL_RNG_MODES: tuple[str, ...] = ("shared-stream", "per-episode")


def run_consumes_randomness(
    sticky_prob: float, start_jitter: int, action_select: str,
) -> bool:
    """Does the episode loop draw from either RNG stream at all?

    Mirrors the exact guards in the step loop: the jitter draw is gated on
    `start_jitter > 0`, the sticky draw short-circuits on `sticky_prob > 0.0`,
    and `select_action` never touches the generator in greedy mode. If all
    three are off, both streams stay untouched for the whole run and every
    episode is a deterministic replay of the same trajectory.
    """
    return (
        float(sticky_prob) > 0.0
        or int(start_jitter) > 0
        or str(action_select) != "greedy"
    )


def episode_rng_seeds(eval_seed: int, episode_index: int) -> tuple[int, int]:
    """Per-episode `(numpy_seed, torch_seed)` derived from (seed, index).

    `SeedSequence` gives two decorrelated 32-bit seeds per episode from the
    pair, so episode i's jitter/sticky stream and its sampled-action stream are
    independent of each other AND of every other episode — the property the
    parallel executor needs, since a lane cannot know what the episodes
    scheduled before it consumed.
    """
    ss = np.random.SeedSequence([int(eval_seed), int(episode_index)])
    np_seed, torch_seed = (int(x) for x in ss.generate_state(2, dtype=np.uint32))
    return np_seed, torch_seed


def select_action(
    logits: torch.Tensor,
    mode: str = "greedy",
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Pick one action index from a (1, n_actions) logits tensor.

    `greedy` is argmax (bit-identical to the historical inline call).
    `sampled` draws once from softmax(logits / temperature) via
    `torch.multinomial`, taking its randomness from `generator` so a sampled
    eval is reproducible from `--eval-seed` and never touches global torch RNG.
    """
    if mode == "greedy":
        return int(torch.argmax(logits[0]).item())
    if mode != "sampled":
        raise ValueError(f"unknown action_select mode: {mode!r}")
    probs = torch.softmax(logits[0].float() / float(temperature), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


def resolve_pixel_encoder(profile: dict) -> Optional[str]:
    """Return `reinforce.encoder` iff it names a pixel encoder the trainer's
    PolicyNetwork supports (nature_dqn / impala); else None."""
    rl = profile.get("reinforce", {}) or {}
    name = rl.get("encoder")
    return name if name in PIXEL_ENCODERS else None


class _TilePolicy:
    """Obs assembly + greedy forward for a tile (RAM-feature) policy.

    Wraps the exact operations the pre-pixel eval loop performed inline so the
    tile path stays byte-identical. `step` is a pool `step_all`/`reset_all`
    tuple `(frame, preprocessed, ram, done)`; tile obs reads `ram` (index 2)
    and decodes it into a feature vector.
    """

    def __init__(
        self,
        net,
        stacker,
        extractor,
        is_recurrent: bool,
        prev_action_feature: bool = False,
        num_actions: int = 0,
    ) -> None:
        self.net = net
        self.stacker = stacker
        self.extractor = extractor
        self.is_recurrent = is_recurrent
        # a_{t-1} de-aliasing: when enabled, `logits` appends the one-hot of the
        # last EXECUTED action (length num_actions) to the tile obs so the net
        # sees the same 718-dim input the prev-action BC checkpoint trained on.
        # Off by default => obs is untouched and the path is byte-identical.
        self.prev_action_feature = bool(prev_action_feature)
        self.num_actions = int(num_actions)

    def reset(self, step) -> np.ndarray:
        return self.stacker.reset(self.extractor.extract(step[2]))

    def push(self, step) -> np.ndarray:
        return self.stacker.push(self.extractor.extract(step[2]))

    def initial_hidden(self, device):
        return self.net.initial_hidden(1, device) if self.is_recurrent else None

    def logits(self, obs: np.ndarray, hidden, prev_action: int = 0):
        if self.prev_action_feature:
            onehot = np.zeros(self.num_actions, dtype=obs.dtype)
            onehot[int(prev_action)] = 1
            obs = np.concatenate((obs, onehot))
        x = torch.from_numpy(obs[None, :]).float()
        if self.is_recurrent:
            logits, _, hidden = self.net.forward_ac_recurrent(x, hidden)
        else:
            logits, _ = self.net.forward_ac(x)
        return logits, hidden


class _PixelPolicy:
    """Obs assembly + greedy forward for a pixel (CNN) policy.

    Reproduces the trainer's collection path bit-for-bit: a FrameStacker over
    the pool's 84x84 `preprocessed` obs (index 1 of the step tuple), then
    `torch.from_numpy(obs).float()` with a `/255` divide ONLY for uint8 obs.
    When the profile sets `preprocess_f16`, the pool delivers observations
    already normalized to [0, 1] in float16, so dividing again would scale
    them 255x and silently wreck the policy — the divide is gated exactly as
    the trainer gates it (`if not preprocess_f16`). Non-recurrent; CPU device.
    """

    is_recurrent = False

    def __init__(self, net, stacker, preprocess_f16: bool) -> None:
        self.net = net
        self.stacker = stacker
        self.preprocess_f16 = preprocess_f16

    def _pp(self, step) -> np.ndarray:
        pp = step[1]
        if self.preprocess_f16:
            # The raw pool in f16 mode ships the 84x84 float16 obs as a
            # (84, 168) uint8 array of raw IEEE-754 half bytes — identical to
            # RustPool._materialize; reinterpret it back to (84, 84) float16
            # so the FrameStacker sees the same [0, 1] normalized obs the
            # trainer feeds its CNN.
            if pp.dtype == np.uint8 and pp.shape == (84, 168):
                return pp.view(np.float16).reshape((84, 84))
            return np.asarray(pp, dtype=np.float16)
        return pp

    def reset(self, step) -> np.ndarray:
        return self.stacker.reset(step[0], self._pp(step))

    def push(self, step) -> np.ndarray:
        return self.stacker.push(step[0], self._pp(step))

    def initial_hidden(self, device):
        return None

    def logits(self, obs: np.ndarray, hidden, prev_action: int = 0):
        # `prev_action` is accepted for a uniform call signature with the tile
        # policy and ignored: the a_{t-1} feature is tile-path only.
        x = torch.from_numpy(obs[None, :]).float()
        if not self.preprocess_f16:
            x = x.div_(255.0)
        logits, _ = self.net.forward_ac(x)
        return logits, hidden


class _Lane:
    """One pool worker's slice of the parallel eval: the episode it is
    currently playing plus every piece of per-episode state the serial loop
    keeps in local variables.

    The serial loop's locals map 1:1 onto these fields — `obs`/`hidden`,
    `ep_return`/`ep_max_byte`/`ep_max_gx`/`ep_cleared`, `step`,
    `_prev_action_idx` — which is what makes the two branches comparable line
    by line. `noops_left` is the one addition: the serial loop can afford to
    burn its no-op frames in a tight pre-loop because it owns the pool, while a
    lane must interleave its no-ops with lanes that are already acting.
    """

    __slots__ = (
        "episode", "worker", "pol", "reward_fn", "tracker", "rng", "gen",
        "noops_left", "obs", "hidden", "step", "prev_action", "action_idx",
        "ret", "max_byte", "max_gx", "cleared", "done",
    )

    def __init__(self, episode: int, worker: int) -> None:
        self.episode = episode
        self.worker = worker
        self.pol: Any = None
        self.reward_fn: Any = None
        self.tracker: Any = None
        self.rng: Any = None
        self.gen: Any = None
        self.noops_left = 1
        self.obs: Any = None
        self.hidden: Any = None
        self.step = 0
        self.prev_action = 0
        self.action_idx = 0
        self.ret = 0.0
        self.max_byte = 0
        self.max_gx = 0
        self.cleared = False
        self.done = False

    def record(self) -> dict:
        """The finished episode's contribution to the aggregate stats.

        Same fields the serial loop folds into its accumulators, in the same
        units, so the two branches share one aggregation site.
        """
        tr = self.tracker
        return {
            "ep_return": self.ret,
            "length": self.step + 1,
            "max_byte": self.max_byte,
            "max_gx": self.max_gx,
            "cleared": self.cleared,
            "seq_clear": bool(tr.seq_clear) if tr is not None else False,
            "warp_taken": bool(tr.warp_taken) if tr is not None else False,
            "furthest_seq": tr.furthest_seq if tr is not None else None,
            "furthest_any": tr.furthest_any if tr is not None else None,
        }


def _run_episodes_parallel(
    pool,
    *,
    lanes: int,
    n_episodes: int,
    max_steps: int,
    bitmasks,
    make_policy: Callable[[], Any],
    make_reward_fn: Callable[[], Any],
    make_tracker: Callable[[], Any],
    stage_blob: Optional[bytes],
    device,
    sticky_prob: float,
    start_jitter: int,
    eval_seed: int,
    action_select: str,
    temperature: float,
) -> list[dict]:
    """Play `n_episodes` across `lanes` pool workers; return one record per
    episode, in episode order.

    Scheduling is by WAVE: episodes are handed out in blocks of `lanes`, and a
    wave starts with one `reset_all()` (plus the same per-worker
    `load_worker_state` warm-start the serial loop does). Nothing carries from
    one wave to the next, so an episode's start state is bit-identical to the
    state the serial loop would have handed it — the alternative (refilling a
    finished lane in place from a saved blob) would make an episode's start
    depend on save/load fidelity rather than on `reset_all`.

    Each lane owns its policy wrapper (hence its own frame/tile stacker), its
    own reward function, its own tracker and its own pair of RNG streams keyed
    on (eval_seed, episode index). Because none of that is shared and the pool
    workers are independent emulators, an episode's trajectory does not depend
    on which lane ran it or on what the other lanes were doing: the returned
    records are identical for any `lanes`, and identical to what
    `_run_episodes_serial` returns for the same arguments under
    `eval_rng="per-episode"`. That last claim is the one that matters and it is
    NOT self-evident, so it is asserted directly — stubbed in
    tests/test_eval_parallel.py and on real emulation in the ROM-gated
    differential there. Under `eval_rng="shared-stream"` there is no such
    claim to make: the serial loop's draws depend on episode order, which is
    why `eval_one_game` refuses that combination.

    Finished lanes are flagged `set_worker_done` so the pool skips their
    emulation for the rest of the wave instead of burning cycles on a
    trajectory nobody reads.
    """
    records: list[Optional[dict]] = [None] * n_episodes
    n_workers = int(pool.num_workers)

    def _finish(lane: _Lane) -> None:
        lane.done = True
        records[lane.episode] = lane.record()
        pool.set_worker_done(lane.worker, True)

    for wave_start in range(0, n_episodes, lanes):
        batch = list(range(wave_start, min(wave_start + lanes, n_episodes)))
        pool.reset_all()
        active: list[_Lane] = []
        for worker, ep in enumerate(batch):
            if stage_blob is not None:
                pool.load_worker_state(worker, stage_blob)
            lane = _Lane(ep, worker)
            np_seed, torch_seed = episode_rng_seeds(eval_seed, ep)
            lane.rng = np.random.RandomState(np_seed)
            lane.gen = torch.Generator(device="cpu")
            lane.gen.manual_seed(torch_seed)
            lane.pol = make_policy()
            lane.reward_fn = make_reward_fn()
            lane.reward_fn.reset()
            lane.tracker = make_tracker()
            # 1 mandatory no-op step to flush the post-reset frame, then the
            # Machado no-op start. Same count, same order, same draw as the
            # serial pre-loop.
            jitter = (
                int(lane.rng.randint(0, start_jitter + 1)) if start_jitter > 0 else 0
            )
            lane.noops_left = 1 + jitter
            active.append(lane)
        # Trailing workers with no episode this wave (final short wave): park
        # them so the pool does not emulate dead slots.
        for worker in range(len(batch), n_workers):
            pool.set_worker_done(worker, True)

        while any(not lane.done for lane in active):
            actions = np.zeros(n_workers, dtype=np.uint8)
            for lane in active:
                if lane.done or lane.noops_left > 0:
                    continue  # no-op: action 0, exactly as the serial pre-loop
                with torch.no_grad():
                    logits, lane.hidden = lane.pol.logits(
                        lane.obs, lane.hidden, lane.prev_action,
                    )
                    action_idx = select_action(
                        logits, action_select, temperature, lane.gen,
                    )
                if (
                    sticky_prob > 0.0
                    and lane.step > 0
                    and lane.rng.random() < sticky_prob
                ):
                    action_idx = lane.prev_action
                lane.prev_action = action_idx
                lane.action_idx = action_idx
                actions[lane.worker] = bitmasks[action_idx]
            r = pool.step_all(actions)
            for lane in active:
                if lane.done:
                    continue
                out = r[lane.worker]
                if lane.noops_left > 0:
                    lane.noops_left -= 1
                    if lane.noops_left == 0:
                        lane.obs = lane.pol.reset(out)
                        lane.hidden = lane.pol.initial_hidden(device)
                        if lane.tracker is not None:
                            lane.tracker.update(out[2])
                        if max_steps <= 0:
                            # `for step in range(0)` never runs; the serial
                            # loop still reports length 1 off its pre-loop
                            # `step = 0`.
                            _finish(lane)
                    continue
                ram = out[2]
                bitmask = bitmasks[lane.action_idx]
                reward, rew_done, _ = lane.reward_fn.compute(ram, action=int(bitmask))
                lane.ret += float(reward)
                byte = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
                if byte > lane.max_byte:
                    lane.max_byte = byte
                gx_now = (int(ram[0x006D]) << 8) | int(ram[0x0086])
                if gx_now > lane.max_gx:
                    lane.max_gx = gx_now
                lane.obs = lane.pol.push(out)
                if lane.reward_fn.episode_success():
                    lane.cleared = True
                if lane.tracker is not None:
                    lane.tracker.update(ram)
                    ended = rew_done or bool(out[3]) or lane.tracker.seq_clear
                else:
                    ended = rew_done or bool(out[3]) or lane.cleared
                if ended or lane.step + 1 >= max_steps:
                    _finish(lane)
                else:
                    lane.step += 1

    missing = [i for i, rec in enumerate(records) if rec is None]
    if missing:  # pragma: no cover - defensive; the wave loop covers every index
        raise RuntimeError(f"parallel eval lost episodes {missing}")
    return [rec for rec in records if rec is not None]


def _run_episodes_serial(
    pool,
    *,
    n_episodes: int,
    max_steps: int,
    bitmasks,
    make_policy: Callable[[], Any],
    make_reward_fn: Callable[[], Any],
    make_tracker: Callable[[], Any],
    stage_blob: Optional[bytes],
    device,
    sticky_prob: float,
    start_jitter: int,
    eval_seed: int,
    action_select: str,
    temperature: float,
    eval_rng: str = "shared-stream",
    dump_stick_dir=None,
) -> list[dict]:
    """Play `n_episodes` one after another in worker 0; return one record per
    episode, in episode order.

    This IS the historical eval loop — the pre-parallel code, lifted out of
    `eval_one_game` so that the thing `_run_episodes_parallel` claims to
    reproduce is a callable object that can be diffed against it. The stub
    suite drives BOTH through the same fake pool and asserts the records are
    byte-identical; without that, every "invariance" test is
    parallel-vs-parallel and an off-by-one in the parallel branch's jitter
    range or sticky guard (i.e. the two numbers that DEFINE the honest
    protocol) passes unnoticed.

    `eval_rng` is the only behavioral addition over the pre-parallel loop:

      shared-stream  one `RandomState(eval_seed)` plus one
                     `Generator(eval_seed)` threaded through every episode in
                     order — bit-identical to every receipt this script has
                     ever written, and the default.
      per-episode    both streams re-seeded from
                     `episode_rng_seeds(eval_seed, ep)` at the top of each
                     episode, so episode i is a pure function of
                     (eval_seed, i). That is what makes the answer independent
                     of how the episodes are scheduled, and therefore what
                     makes this loop and the parallel one comparable at all.

    One policy wrapper and one reward function are built here and reused
    across episodes (reset per episode), exactly as before — the parallel
    branch builds one per lane instead, and the stub/ROM differentials are
    what prove those two are the same thing.
    """
    pol = make_policy()
    reward_fn = make_reward_fn()
    _sticky_rng = np.random.RandomState(eval_seed)
    # Sampled-action selection draws from its OWN torch generator, seeded from
    # --eval-seed, so (a) sampled runs are reproducible and (b) the sticky /
    # jitter numpy stream is byte-identical to the greedy path — a greedy and
    # a sampled run at the same seed see the same no-op starts and the same
    # sticky-repeat decisions, so the pair differs only in the policy draw.
    _action_gen = torch.Generator(device="cpu")
    _action_gen.manual_seed(int(eval_seed))
    records: list[dict] = []
    for ep in range(n_episodes):
        if eval_rng == "per-episode":
            _np_seed, _torch_seed = episode_rng_seeds(eval_seed, ep)
            _sticky_rng = np.random.RandomState(_np_seed)
            _action_gen.manual_seed(_torch_seed)
        pool.reset_all()
        if stage_blob is not None:
            # Warm-start into the curriculum stage. load_worker_state
            # restores RAM but produces no frame until the next step,
            # so the noop step below flushes the post-restore frame for
            # the stacker seed (same pattern the trainer uses).
            pool.load_worker_state(0, stage_blob)
        reward_fn.reset()
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        # Machado no-op starts: hold up to start_jitter no-op frames before
        # control begins, desynchronizing enemy/timer phase from the trained
        # trajectory (the other half of the honest stochastic-eval protocol).
        if start_jitter > 0:
            for _ in range(int(_sticky_rng.randint(0, start_jitter + 1))):
                init = pool.step_all(np.zeros(1, dtype=np.uint8))
        obs = pol.reset(init[0])
        # Fresh hidden state per episode — the GRU must not carry memory
        # across episode boundaries.
        hidden = pol.initial_hidden(device)
        # In --sequential mode reconstruct World-1 progression from RAM
        # independently of episode_success(). episode_success() latches on
        # `cleared_any` at the FIRST 1-1 flagpole, so it can never observe a
        # sequential 1-4 clear; the tracker's `won` semantics can. With
        # --level-clear the predicate is instead "cleared the level it STARTED
        # in" (a forward area/level transition out of the warm-start level,
        # warp-guarded) — the notion the level-scoped consolidation gate probes
        # each level from its entry with. The tracker is seeded with the
        # warm-start frame (init RAM) so its start level locks to where the
        # episode actually began, not the first post-action frame.
        tracker = make_tracker()
        if tracker is not None:
            tracker.update(init[0][2])
        ep_return = 0.0
        ep_max_byte = 0
        ep_max_gx = 0
        ep_cleared = False
        step = 0
        _prev_action_idx = 0
        _ep_sticks: list = []
        for step in range(max_steps):
            # Obs -> action. The policy object supplies the (obs -> logits)
            # mapping; everything else in this loop (reward, RAM byte proxy,
            # sequential tracker, JSON) is obs-agnostic and identical for tile
            # and pixel. `action_select` picks argmax (default) or a seeded
            # draw from the policy's own softmax.
            with torch.no_grad():
                logits, hidden = pol.logits(obs, hidden, _prev_action_idx)
                action_idx = select_action(
                    logits, action_select, temperature, _action_gen,
                )
            # Sticky-actions eval (Machado et al. 2018): with prob
            # sticky_prob repeat the previous executed action. On the
            # TRUSTWORTHY-obs harness (eval_game's obs matches training,
            # unlike eval_composite's tile adapter) this is the honest
            # perturbation test.
            _chosen_idx = action_idx
            if sticky_prob > 0.0 and step > 0 and _sticky_rng.random() < sticky_prob:
                action_idx = _prev_action_idx
            _prev_action_idx = action_idx
            bitmask = bitmasks[action_idx]
            r = pool.step_all(np.array([bitmask], dtype=np.uint8))
            # Recovery-assay hook (opt-in, no effect otherwise): at every
            # DIVERGENT stick (executed != chosen) snapshot the post-stick
            # savestate. The assay adjudicates later whether a recovering
            # continuation existed from exactly this state.
            if dump_stick_dir is not None and action_idx != _chosen_idx:
                _sp = dump_stick_dir / f"ep{ep:03d}_t{step:05d}.state"
                _sp.write_bytes(bytes(pool.save_worker_state(0)))
                _ep_sticks.append([int(step), str(_sp)])
            ram = r[0][2]
            reward, rew_done, _ = reward_fn.compute(ram, action=int(bitmask))
            ep_return += float(reward)
            # SMB world/level packing as a coarse "furthest stage"
            # proxy. Not a success signal — that's episode_success().
            byte = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
            if byte > ep_max_byte:
                ep_max_byte = byte
            # Death-gx telemetry (SMB addresses; benign reads elsewhere,
            # like the byte proxy above): where each episode's horizontal
            # frontier stalled — THE localization signal when a level
            # trains but a cold probe still reads 0.0.
            gx_now = (int(ram[0x006D]) << 8) | int(ram[0x0086])
            if gx_now > ep_max_gx:
                ep_max_gx = gx_now
            obs = pol.push(r[0])
            if reward_fn.episode_success():
                ep_cleared = True
            if tracker is not None:
                tracker.update(ram)
                # Sequential mode: do NOT stop at the 1-1 flag (ep_cleared).
                # Run until a real World-1 castle clear (seq_clear), a death /
                # pool-done, or max_steps.
                if rew_done or bool(r[0][3]) or tracker.seq_clear:
                    break
            # End the episode on the reward fn's terminal signal, the
            # pool's done flag, or once the game is cleared.
            elif rew_done or bool(r[0][3]) or ep_cleared:
                break
        records.append({
            "sticks": _ep_sticks,
            "ep_return": ep_return,
            "length": step + 1,
            "max_byte": ep_max_byte,
            "max_gx": ep_max_gx,
            "cleared": ep_cleared,
            "seq_clear": bool(tracker.seq_clear) if tracker is not None else False,
            "warp_taken": bool(tracker.warp_taken) if tracker is not None else False,
            "furthest_seq": tracker.furthest_seq if tracker is not None else None,
            "furthest_any": tracker.furthest_any if tracker is not None else None,
        })
    return records


def resolve_eval_checkpoint(
    ckpt_dir: Path,
    game: str,
    latest: bool = False,
    iter_: Optional[int] = None,
    checkpoint: Optional[str] = None,
) -> Optional[Path]:
    """Pick which checkpoint to eval.

    Default prefers the retained winner (`winners/best.pt`), then the best
    eval-history clear_rate, then the latest — so `make eval` measures a
    real win rather than whatever the latest (possibly collapsed) iter is.
    `--checkpoint PATH` pins an exact file by path (used by the cold-eval
    probe, which writes a temp checkpoint and subprocesses this script);
    `--iter N` pins an exact `vanilla_ppo_iter_NNNNN.pt`; `--latest` forces
    the freshest trained checkpoint (to measure current training progress).
    """
    if checkpoint is not None:
        pinned = Path(checkpoint)
        return pinned if pinned.exists() else None
    if iter_ is not None:
        pinned = ckpt_dir / f"vanilla_ppo_iter_{iter_:05d}.pt"
        return pinned if pinned.exists() else None
    if latest:
        return find_latest_trained_checkpoint(ckpt_dir)
    return find_playable_checkpoint(game, ckpt_dir)


def eval_one_game(
    game: str,
    profile_path: Path,
    rom_path: str,
    n_episodes: int,
    max_steps: int,
    stage: Optional[int] = None,
    latest: bool = False,
    iter_: Optional[int] = None,
    checkpoint: Optional[str] = None,
    sequential: bool = False,
    start_state: Optional[str] = None,
    level_clear: bool = False,
    sticky_prob: float = 0.0,
    start_jitter: int = 0,
    eval_seed: int = 0,
    action_select: str = "greedy",
    temperature: float = 1.0,
    eval_workers: int = 1,
    eval_rng: str = "shared-stream",
    prev_action_feature: bool = False,
    dump_stick_states: Optional[str] = None,
) -> dict:
    """Run N episodes; return a dict of eval stats."""
    if action_select not in _ACTION_SELECT_MODES:
        raise ValueError(
            f"action_select must be one of {sorted(_ACTION_SELECT_MODES)}, "
            f"got {action_select!r}"
        )
    if not (float(temperature) > 0.0):
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    if eval_rng not in _EVAL_RNG_MODES:
        raise ValueError(
            f"eval_rng must be one of {sorted(_EVAL_RNG_MODES)}, got {eval_rng!r}"
        )
    if int(eval_workers) < 1:
        raise ValueError(f"eval_workers must be >= 1, got {eval_workers!r}")
    # Concurrency is only sound where the answer does not depend on the
    # schedule. Under shared-stream a stochastic episode's draws depend on how
    # many draws the episodes BEFORE it consumed — a data-dependent count no
    # lane can know — so running such a run on >1 worker cannot reproduce the
    # serial number and must not silently return a different one.
    _stochastic = run_consumes_randomness(sticky_prob, start_jitter, action_select)
    if int(eval_workers) > 1 and eval_rng == "shared-stream" and _stochastic:
        raise ValueError(
            "eval_workers > 1 needs eval_rng='per-episode' for a stochastic "
            "run (sticky_prob > 0, start_jitter > 0, or action_select="
            "'sampled'): the shared-stream RNG is threaded through the "
            "episodes in order, so episode i's draws depend on how many draws "
            "episodes 0..i-1 consumed and cannot be re-derived per episode. "
            "Pass eval_rng='per-episode' (results are invariant to worker "
            "count but are NOT byte-comparable to shared-stream receipts), or "
            "keep eval_workers=1."
        )
    # ONE thing decides which executor runs: the EFFECTIVE worker count, after
    # the episodes/CPU clamp below. `--eval-rng` only changes where the
    # randomness comes from, and BOTH executors honor both modes — so a receipt
    # that reads `eval_workers: 1` always came out of the serial loop, and the
    # 1-worker-vs-K-worker comparison is a real serial-vs-parallel differential
    # rather than the parallel branch grading its own homework.
    _wide_requested = int(eval_workers) > 1
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    ckpt_dir = derive_checkpoint_dir("./checkpoints", profile.get("name"))
    ckpt = resolve_eval_checkpoint(
        ckpt_dir, game, latest=latest, iter_=iter_, checkpoint=checkpoint,
    )
    if ckpt is None:
        return {
            "game": game,
            "status": "no_checkpoint",
            "ckpt_dir": str(ckpt_dir),
            "detail": (
                f"no vanilla_ppo_iter_{iter_:05d}.pt in {ckpt_dir}"
                if iter_ is not None
                else "no winner, eval history, or trained checkpoint found"
            ),
        }

    # Profile-driven action space + observation encoder/policy. No game-
    # specific constants live in this script — adding a game is a profile +
    # a trained checkpoint, never an eval-script edit. Tile encoders route
    # through resolve_encoder (unchanged, byte-identical). A pixel encoder
    # (nature_dqn / impala) that resolve_encoder rejects gets the CNN path
    # below; anything neither tile nor pixel stays unsupported_profile.
    try:
        bitmasks = action_space_to_bitmasks(profile["action_space"])
    except (KeyError, ValueError) as e:
        return {
            "game": game,
            "status": "unsupported_profile",
            "detail": str(e),
            "ckpt_dir": str(ckpt_dir),
        }
    pixel_encoder: Optional[str] = None
    try:
        extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    except ValueError as tile_err:
        pixel_encoder = resolve_pixel_encoder(profile)
        if pixel_encoder is None:
            return {
                "game": game,
                "status": "unsupported_profile",
                "detail": str(tile_err),
                "ckpt_dir": str(ckpt_dir),
            }

    state = torch.load(str(ckpt), map_location="cpu")
    # CPU device: eval is deliberately CPU-only so the cold probe can
    # subprocess it without perturbing the trainer's MPS context/RNG.
    device = torch.device("cpu")
    # Whether the pool must emit float16-normalized [0, 1] observations
    # (pixel f16 fast path). False for tile — tile reads RAM, not pixels.
    pool_preprocess_f16 = False

    if pixel_encoder is None:
        # === TILE PATH (RAM-feature MLP / GRU) — unchanged behavior ===
        stack_size = stacked_dim // feature_dim
        # With --prev-action-feature the policy input is the stacked tile obs
        # PLUS a one-hot of a_{t-1} (length num_actions), so the net must be
        # built one action-space wider or the 718-dim checkpoint fails to load.
        # Flag off => net_feature_dim == stacked_dim, byte-identical.
        net_feature_dim = stacked_dim + (
            len(bitmasks) if prev_action_feature else 0
        )
        # Dispatch on the checkpoint's architecture family. A recurrent
        # (tile_gru) checkpoint MUST load into the recurrent policy and have
        # its GRU hidden state threaded through the step loop below; loading
        # it into the stateless net would leave a non-empty `missing` set
        # (the GRU weights) and eval a half-random policy.
        _commit_cfg = ((profile.get("reinforce", {}) or {})
                       .get("commitment_options") or {})
        if _commit_cfg.get("enabled"):
            # Commitment-options checkpoint: an 18-way (primitive,
            # duration) head whose timer rides the RECURRENT plumbing —
            # hidden state (pair, remaining) is zeroed at episode starts
            # by the same machinery that resets a GRU, giving fresh-
            # decision semantics with no harness changes. Pair choice at
            # decisions is greedy inside the driver; the emitted primitive
            # logits are near-one-hot so both harness selection modes
            # follow the commitment.
            from src.training.commitment_policy import CommitmentPolicy
            _sd = state["net_state_dict"]
            _flat_sd = {k[len("trunk.flat."):]: v for k, v in _sd.items()
                        if k.startswith("trunk.flat.")}
            _flat, _ = build_tile_policy_from_checkpoint(
                {"net_state_dict": _flat_sd},
                num_actions=len(bitmasks), feature_dim=net_feature_dim,
            )
            net = CommitmentPolicy.from_flat_policy(
                _flat, trunk_dim=_flat.fc2.out_features,
                num_primitives=len(bitmasks),
                durations=tuple(_commit_cfg.get("durations", (1, 2, 4))))
            is_recurrent = True          # rides the hidden-state plumbing
            net_kind = "CommitmentPolicy"
        else:
            net, is_recurrent = build_tile_policy_from_checkpoint(
                state, num_actions=len(bitmasks), feature_dim=net_feature_dim,
            )
            net_kind = type(net).__name__
        # Load non-strict so older checkpoints with extra aux heads still
        # load, but FAIL LOUD if core weights are missing — a silent
        # partial load would eval a half-random policy and still print
        # status "ok", which is worse than an error.
        load_res = net.load_state_dict(state["net_state_dict"], strict=False)
        # PHASE 3: if the profile arms the hazard veto, EVALUATE with it.
        # The first Phase-3 run did not, so the treatment arm was scored as
        # though unmasked — the arm's entire intervention was absent from
        # its own measurement, and both arms returned identical numbers
        # because they were, in effect, the same policy. Wrapped after the
        # state dict loads so the weights land on the real net.
        _hz = (profile.get("reinforce", {}) or {}).get("hazard_mask") or {}
        if _hz.get("enabled") and not load_res.missing_keys:
            from src.training.hazard_mask import (
                HazardMask, HazardMaskedPolicy)
            _m = HazardMask.from_checkpoint(
                _hz["checkpoint"],
                threshold=float(_hz.get("threshold", 0.90)), enabled=True)
            net = HazardMaskedPolicy(net, _m)
            net_kind = f"{net_kind}+hazard_veto"
            print(f"[eval] hazard veto ARMED threshold={_m.threshold}",
                  flush=True)
        if load_res.missing_keys:
            return {
                "game": game,
                "status": "checkpoint_mismatch",
                "checkpoint": str(ckpt),
                "recurrent": bool(is_recurrent),
                "missing_keys": list(load_res.missing_keys),
                "unexpected_keys": list(load_res.unexpected_keys),
                "detail": (
                    "checkpoint does not provide all network weights for "
                    f"{net_kind}(num_actions={len(bitmasks)}, "
                    f"feature_dim={net_feature_dim}); refusing to eval a "
                    "partially-initialized policy."
                ),
            }
        net.eval()

        def make_policy() -> _TilePolicy:
            # One stacker PER policy wrapper: the stacker is the only stateful
            # piece, so a parallel lane must not share one. Constructed with
            # the same arguments the serial path always used. The stacker still
            # produces the stacked_dim (712) tile obs; the a_{t-1} one-hot is
            # appended inside `logits` only when prev_action_feature is set.
            return _TilePolicy(
                net,
                TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim),
                extractor,
                is_recurrent,
                prev_action_feature=prev_action_feature,
                num_actions=len(bitmasks),
            )
    else:
        # === PIXEL PATH (CNN) ===
        # Build the SAME PolicyNetwork the trainer builds — frame_stack=4 /
        # frame_size=84 defaults, encoder + layernorm from the profile — and
        # load its net_state_dict (the exact key the trainer saves). Obs are
        # assembled by a FrameStacker over the pool's 84x84 preprocessed obs,
        # matching the trainer's collection path bit-for-bit (see _PixelPolicy).
        rl = profile.get("reinforce", {}) or {}
        pool_preprocess_f16 = bool(rl.get("preprocess_f16", False))
        frame_stack = int(rl.get("frame_stack", 4))
        use_layernorm = bool(rl.get("layernorm", True))
        net = PolicyNetwork(
            num_actions=len(bitmasks),
            frame_stack=frame_stack,
            frame_size=84,
            encoder=pixel_encoder,
            use_layernorm=use_layernorm,
        )
        is_recurrent = False
        net_kind = type(net).__name__
        load_res = net.load_state_dict(state["net_state_dict"], strict=False)
        if load_res.missing_keys:
            return {
                "game": game,
                "status": "checkpoint_mismatch",
                "checkpoint": str(ckpt),
                "recurrent": False,
                "missing_keys": list(load_res.missing_keys),
                "unexpected_keys": list(load_res.unexpected_keys),
                "detail": (
                    "checkpoint does not provide all network weights for "
                    f"{net_kind}(num_actions={len(bitmasks)}, "
                    f"encoder={pixel_encoder!r}, frame_stack={frame_stack}); "
                    "refusing to eval a partially-initialized policy."
                ),
            }
        net.eval()
        obs_dtype = np.float16 if pool_preprocess_f16 else np.uint8

        def make_policy() -> _PixelPolicy:  # type: ignore[misc]
            return _PixelPolicy(
                net,
                FrameStacker(stack_size=frame_stack, dtype=obs_dtype),
                pool_preprocess_f16,
            )

    # The profile's start state seeds the POOL (cold-boot default). It
    # must not clobber the `start_state` CLI param: this line previously
    # assigned to `start_state` itself, which silently killed every
    # --start-state override — all "arbitrary state" probes warm-started
    # from the profile's state instead (the 2-2 probes cold-booted into
    # the title demo and scored an artifact 0.0).
    profile_start = profile.get("start_state_path") or (
        Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
    )
    # Serial keeps the historical single worker. Parallel takes the smallest of
    # what was asked for, what there is to run, and what the box has: more
    # workers than episodes would idle, and oversubscribing cores slows the
    # whole pool down without changing the answer. If the clamp lands on 1 the
    # serial loop runs — asking for 8 workers to play 1 episode should not put
    # a "parallel" run in the receipt.
    lanes = 1
    if _wide_requested:
        lanes = max(1, min(
            int(eval_workers), int(n_episodes), int(os.cpu_count() or 1),
        ))
    _parallel = lanes > 1
    pool = Pool(
        rom_path=rom_path, num_workers=lanes, frame_skip=4,
        start_state_path=(
            str(profile_start) if Path(str(profile_start)).exists() else None
        ),
    )
    # Pixel f16 fast path: tell the pool to emit the 84x84 obs as float16
    # already normalized to [0, 1] (shipped as (84, 168) uint8 half-bytes),
    # exactly as the trainer configures it via RustPool._apply_pool_knobs.
    # _PixelPolicy reinterprets and the /255 divide is skipped. No-op for the
    # tile path (pool_preprocess_f16 stays False — tile reads RAM, not pixels).
    if pool_preprocess_f16:
        pool.set_preprocess_f16(True)

    # Optional mid-chain warm-start. By default eval boots from the profile
    # start state (live level-1 gameplay), so it only measures the FIRST
    # level. Two ways to warm-start each episode elsewhere:
    #   --stage N        load smb_curriculum/stage_NN.state (curriculum stage).
    #   --start-state P  load an ARBITRARY save-state file P (a ladder rung's
    #                    seed, a captured entry blob, etc.) — the level-scoped
    #                    consolidation gate probes each level from its entry.
    # The two are mutually exclusive; --start-state takes precedence.
    stage_blob: Optional[bytes] = None
    if start_state is not None:
        ss_path = Path(start_state)
        if not ss_path.exists():
            pool.shutdown()
            return {
                "game": game,
                "status": "no_such_start_state",
                "start_state": str(start_state),
                "detail": f"{ss_path} not found",
            }
        stage_blob = ss_path.read_bytes()
    elif stage is not None:
        stage_path = ckpt_dir / "smb_curriculum" / f"stage_{stage:02d}.state"
        if not stage_path.exists():
            pool.shutdown()
            return {
                "game": game,
                "status": "no_such_stage",
                "stage": stage,
                "detail": f"{stage_path} not found",
            }
        stage_blob = stage_path.read_bytes()

    # The reward function is the SAME Rust object the trainer uses, so
    # the eval's notion of "cleared" (episode_success) and "return"
    # (sum of per-step rewards) matches training exactly — no separate
    # heuristic to drift out of sync. A factory, not an instance: the serial
    # loop builds one and resets it per episode (as it always did), a parallel
    # lane builds its own.
    def _make_reward_fn():
        return build_reward_function(profile)

    def _make_tracker():
        """Same tracker choice the serial loop makes per episode; hoisted so
        the parallel executor can give every lane its own."""
        if sequential and level_clear:
            return LevelClearTracker()
        if sequential:
            return SequentialTracker()
        return None

    returns: list[float] = []
    lengths: list[int] = []
    max_bytes: list[int] = []
    max_gxs: list[int] = []
    clears = 0
    # --sequential accumulators (per-episode). Only populated in sequential
    # mode; left empty otherwise so the back-compat result is unchanged.
    seq_clears = 0
    warps = 0
    best_seq: Optional[tuple] = None
    best_any: Optional[tuple] = None

    # ONE call site, ONE fold. Both executors take the same arguments and
    # return the same per-episode records, which is what lets the stub suite
    # (and the ROM differential) assert that swapping one for the other cannot
    # change the answer.
    _executor_kwargs = dict(
        n_episodes=n_episodes,
        max_steps=max_steps,
        bitmasks=bitmasks,
        make_policy=make_policy,
        make_reward_fn=_make_reward_fn,
        make_tracker=_make_tracker,
        stage_blob=stage_blob,
        device=device,
        sticky_prob=sticky_prob,
        start_jitter=start_jitter,
        eval_seed=eval_seed,
        action_select=action_select,
        temperature=temperature,
    )
    if dump_stick_states:
        # The recovery-assay hook lives only in the serial loop; parallel
        # lanes would interleave snapshots across workers.
        _dsd = Path(dump_stick_states)
        _dsd.mkdir(parents=True, exist_ok=True)
        _records = _run_episodes_serial(
            pool, eval_rng=eval_rng, dump_stick_dir=_dsd, **_executor_kwargs)
    elif _parallel:
        _records = _run_episodes_parallel(pool, lanes=lanes, **_executor_kwargs)
    else:
        _records = _run_episodes_serial(pool, eval_rng=eval_rng, **_executor_kwargs)

    for _rec in _records:
        returns.append(_rec["ep_return"])
        lengths.append(_rec["length"])
        max_bytes.append(_rec["max_byte"])
        max_gxs.append(_rec["max_gx"])
        if _rec["cleared"]:
            clears += 1
        if sequential:
            if _rec["seq_clear"]:
                seq_clears += 1
            if _rec["warp_taken"]:
                warps += 1
            if _rec["furthest_seq"] is not None and (
                best_seq is None or _rec["furthest_seq"] > best_seq
            ):
                best_seq = _rec["furthest_seq"]
            if _rec["furthest_any"] is not None and (
                best_any is None or _rec["furthest_any"] > best_any
            ):
                best_any = _rec["furthest_any"]

    pool.shutdown()

    # Recovery-assay manifest: when dumping stick states, persist the
    # per-episode records (stick lists + outcome) next to the snapshots
    # so the adjudicator can map states -> episode fate without parsing
    # stdout.
    if dump_stick_states:
        import json as _json
        (Path(dump_stick_states).parent / "manifest.json").write_text(
            _json.dumps({"records": [
                {k: r.get(k) for k in
                 ("sticks", "length", "cleared", "ep_return")}
                for r in _records]}, indent=2) + "\n")

    result = {
        "game": game,
        "status": "ok",
        "checkpoint": str(ckpt),
        "recurrent": bool(is_recurrent),
        "stage": stage if stage is not None else "start",
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_byte_seen": int(max(max_bytes)) if max_bytes else 0,
        "mean_max_byte": float(np.mean(max_bytes)) if max_bytes else 0.0,
        "max_gx_per_episode": max_gxs,
        "clear_rate": clears / max(1, n_episodes),
        # Provenance for the honest protocol: a clear rate is only readable
        # next to HOW the actions were drawn. Always emitted so a receipt can
        # never be mistaken for the other mode's number.
        "action_select": str(action_select),
        "temperature": float(temperature),
        # ...and next to WHAT WAS PERTURBED. These three were absent from the
        # row entirely until now, which is how the v4 consolidation receipts
        # (checkpoints/mario_1_1_backward_v4/eval.jsonl) came to carry a
        # sticky-0.25 + jitter-16 number with no record that it was one: the
        # protocol lived only in the config header and the invoking command
        # line. A rate whose protocol is not in its own row cannot be cited,
        # and cannot be compared against another row. Always emitted, never
        # null — 0.0 / 0 / the seed actually used are the honest values for an
        # unperturbed run, and `run_consumes_randomness` is the one-field
        # answer to "was this a deterministic replay?".
        "sticky_prob": float(sticky_prob),
        "start_jitter": int(start_jitter),
        "eval_seed": int(eval_seed),
        "stochastic": bool(
            run_consumes_randomness(sticky_prob, start_jitter, action_select)
        ),
        # How the episodes were scheduled and where their randomness came
        # from. `eval_workers` is the EFFECTIVE lane count (after the
        # episodes/cpu clamp), not what was asked for. Both belong next to the
        # clear rate for the same reason `action_select` does: a per-episode
        # number and a shared-stream number are different draws of the same
        # protocol and must never be compared as if they were the same run.
        "eval_workers": int(lanes),
        "eval_rng": str(eval_rng),
        "timestamp": time.time(),
    }
    if start_state is not None:
        result["start_state"] = str(start_state)
    if sequential:
        # THE primary campaign metric: sequential World-1 clear rate + how far
        # the greedy policy actually reaches, with warps reported "beyond" but
        # never counted as a clear. `clear_rate` above stays the 1-1-latch rate
        # for back-compat; `seq_clear_rate` is the DoD number every phase gates
        # on. With --level-clear it is instead the per-level rate: the fraction
        # of episodes that cleared the level they warm-started in — the number
        # the level-scoped consolidation gate accepts/rolls back on.
        # `furthest_*_level` are the deepest (world, level) across episodes.
        result["sequential"] = True
        result["level_clear"] = bool(level_clear)
        result["seq_clear_rate"] = seq_clears / max(1, n_episodes)
        result["warp_rate"] = warps / max(1, n_episodes)
        result["furthest_seq_level"] = level_label(best_seq)
        result["furthest_any_level"] = level_label(best_any)
        result["furthest_seq"] = list(best_seq) if best_seq is not None else None
        result["furthest_any"] = list(best_any) if best_any is not None else None
    # Append to per-game eval log for scoreboard consumption. The dir may
    # not exist when --checkpoint points outside the profile's default
    # checkpoint dir (e.g. probing a seed-suffixed run dir).
    eval_log = ckpt_dir / "eval.jsonl"
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_log, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True,
                        help=f"Game logical name: {sorted(set(DEFAULT_PROFILES))}")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--rom", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--stage", type=int, default=None,
                        help="Curriculum stage to warm-start each episode "
                             "from (smb_curriculum/stage_NN.state). Default: "
                             "boot from the profile start state (first stage).")
    parser.add_argument("--latest", action="store_true",
                        help="Eval the freshest trained checkpoint instead of "
                             "the retained winner (measures current training "
                             "progress).")
    parser.add_argument("--iter", type=int, default=None, dest="iter_",
                        metavar="N",
                        help="Eval an exact vanilla_ppo_iter_NNNNN.pt by "
                             "iteration number.")
    parser.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                        help="Eval an exact checkpoint file by path (used by "
                             "the cold-eval probe, which writes a temp "
                             "checkpoint and subprocesses this script).")
    parser.add_argument("--sequential", action="store_true",
                        help="SMB one-shot mode: reconstruct sequential World-1 "
                             "progression from RAM (play THROUGH the 1-1 flag) "
                             "and report seq_clear_rate / furthest_seq_level / "
                             "furthest_any_level / warp_rate. The DoD metric.")
    parser.add_argument("--start-state", type=str, default=None, metavar="PATH",
                        dest="start_state",
                        help="Warm-start every episode from an ARBITRARY "
                             "save-state file (a ladder rung seed or captured "
                             "entry blob). The level-scoped consolidation gate "
                             "probes each level from its entry with this. "
                             "Mutually exclusive with --stage; takes precedence.")
    parser.add_argument("--sticky-prob", type=float, default=0.0, dest="sticky_prob",
                        help="Machado sticky-actions eval: prob of repeating the previous action.")
    parser.add_argument("--start-jitter", type=int, default=0, dest="start_jitter",
                        help="Machado no-op starts: up to this many no-op frames before control.")
    parser.add_argument("--eval-seed", type=int, default=0, dest="eval_seed")
    parser.add_argument("--action-select", type=str, default="greedy",
                        dest="action_select", choices=list(_ACTION_SELECT_MODES),
                        help="How to draw each action from the policy logits: "
                             "'greedy' (argmax, the default and the historical "
                             "behavior) or 'sampled' (one draw from "
                             "softmax(logits/T), seeded from --eval-seed). The "
                             "honest protocol reports BOTH.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        dest="temperature",
                        help="Softmax temperature for --action-select sampled. "
                             "Ignored when greedy. Must be > 0.")
    parser.add_argument("--dump-stick-states", type=str, default=None,
                        metavar="DIR",
                        help="recovery assay: save the post-stick savestate "
                             "at every DIVERGENT sticky application "
                             "(executed != chosen) into DIR; per-episode "
                             "stick lists ride the episode records. Forces "
                             "the serial loop.")
    parser.add_argument("--eval-workers", type=int, default=1, dest="eval_workers",
                        metavar="K",
                        help="Run K episodes concurrently in a K-worker pool "
                             "(clamped to --episodes and the CPU count). "
                             "Default 1 = the historical serial path. A "
                             "stochastic run (sticky / jitter / sampled) also "
                             "needs --eval-rng per-episode.")
    parser.add_argument("--eval-rng", type=str, default="shared-stream",
                        dest="eval_rng", choices=list(_EVAL_RNG_MODES),
                        help="Where the jitter/sticky/sampling randomness "
                             "comes from. 'shared-stream' (default, the "
                             "historical behavior) threads one stream through "
                             "the episodes in order, so episode i depends on "
                             "how long episodes 0..i-1 ran. 'per-episode' "
                             "seeds each episode from (--eval-seed, episode "
                             "index), which makes the result invariant to "
                             "--eval-workers but NOT byte-comparable to "
                             "shared-stream receipts.")
    parser.add_argument("--level-clear", action="store_true", dest="level_clear",
                        help="With --sequential, score 'cleared the level it "
                             "STARTED in' (a forward area/level transition out "
                             "of the warm-start level, warp-guarded) instead of "
                             "the full World-1 chain. The per-level gate metric.")
    parser.add_argument("--prev-action-feature", action="store_true",
                        dest="prev_action_feature",
                        help="Tile path only: append the one-hot of the last "
                             "EXECUTED action (a_{t-1}, length num_actions) to "
                             "the obs before the forward pass, matching a policy "
                             "trained on the 718-dim prev-action augmented obs. "
                             "The net is built one action-space wider. Off by "
                             "default (obs untouched, byte-identical path).")
    args = parser.parse_args()

    if not (args.temperature > 0.0):
        # argparse cannot express "positive float"; surface it as the same
        # single-JSON-line contract every other early exit uses rather than a
        # traceback the cold probe would have to parse out of stderr.
        print(json.dumps({
            "game": args.game, "status": "bad_args",
            "detail": f"--temperature must be > 0, got {args.temperature}",
        }))
        return 1

    if args.eval_workers < 1:
        print(json.dumps({
            "game": args.game, "status": "bad_args",
            "detail": f"--eval-workers must be >= 1, got {args.eval_workers}",
        }))
        return 1

    if (
        args.eval_workers > 1
        and args.eval_rng == "shared-stream"
        and run_consumes_randomness(
            args.sticky_prob, args.start_jitter, args.action_select,
        )
    ):
        # Refuse rather than quietly returning a different (but equally valid)
        # set of draws: a receipt whose numbers depend on the worker count is
        # not a receipt. Same single-JSON-line contract as above.
        print(json.dumps({
            "game": args.game, "status": "bad_args",
            "detail": (
                "--eval-workers > 1 needs --eval-rng per-episode for a "
                "stochastic run (--sticky-prob > 0, --start-jitter > 0, or "
                "--action-select sampled): the default shared-stream RNG is "
                "threaded through the episodes in order, so episode i's draws "
                "depend on how many draws episodes 0..i-1 consumed and cannot "
                "be re-derived per episode. Per-episode results are invariant "
                "to worker count but are NOT byte-comparable to shared-stream "
                "receipts."
            ),
        }))
        return 1

    profile_path = resolve_profile_path(args.game, args.profile)
    rom_path = args.rom or DEFAULT_ROMS.get(args.game.lower().strip())
    if rom_path is None or not Path(rom_path).exists():
        print(json.dumps({
            "game": args.game, "status": "no_rom",
            "rom_path": rom_path,
        }))
        return 1

    result = eval_one_game(
        args.game, profile_path, rom_path,
        args.episodes, args.max_steps, stage=args.stage,
        latest=args.latest, iter_=args.iter_,
        checkpoint=args.checkpoint, sequential=args.sequential,
        start_state=args.start_state, level_clear=args.level_clear,
        sticky_prob=args.sticky_prob, start_jitter=args.start_jitter,
        eval_seed=args.eval_seed, action_select=args.action_select,
        temperature=args.temperature, eval_workers=args.eval_workers,
        eval_rng=args.eval_rng, prev_action_feature=args.prev_action_feature,
        dump_stick_states=args.dump_stick_states,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
