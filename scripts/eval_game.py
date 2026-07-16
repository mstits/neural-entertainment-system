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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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

    def __init__(self, net, stacker, extractor, is_recurrent: bool) -> None:
        self.net = net
        self.stacker = stacker
        self.extractor = extractor
        self.is_recurrent = is_recurrent

    def reset(self, step) -> np.ndarray:
        return self.stacker.reset(self.extractor.extract(step[2]))

    def push(self, step) -> np.ndarray:
        return self.stacker.push(self.extractor.extract(step[2]))

    def initial_hidden(self, device):
        return self.net.initial_hidden(1, device) if self.is_recurrent else None

    def logits(self, obs: np.ndarray, hidden):
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

    def logits(self, obs: np.ndarray, hidden):
        x = torch.from_numpy(obs[None, :]).float()
        if not self.preprocess_f16:
            x = x.div_(255.0)
        logits, _ = self.net.forward_ac(x)
        return logits, hidden


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
) -> dict:
    """Run N episodes; return a dict of eval stats."""
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
        # Dispatch on the checkpoint's architecture family. A recurrent
        # (tile_gru) checkpoint MUST load into the recurrent policy and have
        # its GRU hidden state threaded through the step loop below; loading
        # it into the stateless net would leave a non-empty `missing` set
        # (the GRU weights) and eval a half-random policy.
        net, is_recurrent = build_tile_policy_from_checkpoint(
            state, num_actions=len(bitmasks), feature_dim=stacked_dim,
        )
        net_kind = type(net).__name__
        # Load non-strict so older checkpoints with extra aux heads still
        # load, but FAIL LOUD if core weights are missing — a silent
        # partial load would eval a half-random policy and still print
        # status "ok", which is worse than an error.
        load_res = net.load_state_dict(state["net_state_dict"], strict=False)
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
                    f"feature_dim={stacked_dim}); refusing to eval a "
                    "partially-initialized policy."
                ),
            }
        net.eval()
        stacker = TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
        pol = _TilePolicy(net, stacker, extractor, is_recurrent)
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
        stacker = FrameStacker(stack_size=frame_stack, dtype=obs_dtype)
        pol = _PixelPolicy(net, stacker, pool_preprocess_f16)

    start_state = profile.get("start_state_path") or (
        Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
    )
    pool = Pool(
        rom_path=rom_path, num_workers=1, frame_skip=4,
        start_state_path=str(start_state) if Path(str(start_state)).exists() else None,
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
    # heuristic to drift out of sync.
    reward_fn = build_reward_function(profile)

    returns: list[float] = []
    lengths: list[int] = []
    max_bytes: list[int] = []
    clears = 0
    # --sequential accumulators (per-episode). Only populated in sequential
    # mode; left empty otherwise so the back-compat result is unchanged.
    seq_clears = 0
    warps = 0
    best_seq: Optional[tuple] = None
    best_any: Optional[tuple] = None

    for ep in range(n_episodes):
        pool.reset_all()
        if stage_blob is not None:
            # Warm-start into the curriculum stage. load_worker_state
            # restores RAM but produces no frame until the next step,
            # so the noop step below flushes the post-restore frame for
            # the stacker seed (same pattern the trainer uses).
            pool.load_worker_state(0, stage_blob)
        reward_fn.reset()
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
        if sequential and level_clear:
            tracker = LevelClearTracker()
        elif sequential:
            tracker = SequentialTracker()
        else:
            tracker = None
        if tracker is not None:
            tracker.update(init[0][2])
        ep_return = 0.0
        ep_max_byte = 0
        ep_cleared = False
        step = 0
        for step in range(max_steps):
            # Obs -> greedy argmax action. The policy object supplies the
            # (obs -> logits) mapping; everything else in this loop (reward,
            # RAM byte proxy, sequential tracker, JSON) is obs-agnostic and
            # identical for tile and pixel.
            with torch.no_grad():
                logits, hidden = pol.logits(obs, hidden)
                action_idx = int(torch.argmax(logits[0]).item())
            bitmask = bitmasks[action_idx]
            r = pool.step_all(np.array([bitmask], dtype=np.uint8))
            ram = r[0][2]
            reward, rew_done, _ = reward_fn.compute(ram, action=int(bitmask))
            ep_return += float(reward)
            # SMB world/level packing as a coarse "furthest stage"
            # proxy. Not a success signal — that's episode_success().
            byte = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
            if byte > ep_max_byte:
                ep_max_byte = byte
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
        returns.append(ep_return)
        lengths.append(step + 1)
        max_bytes.append(ep_max_byte)
        if ep_cleared:
            clears += 1
        if tracker is not None:
            if tracker.seq_clear:
                seq_clears += 1
            if tracker.warp_taken:
                warps += 1
            if tracker.furthest_seq is not None and (
                best_seq is None or tracker.furthest_seq > best_seq
            ):
                best_seq = tracker.furthest_seq
            if tracker.furthest_any is not None and (
                best_any is None or tracker.furthest_any > best_any
            ):
                best_any = tracker.furthest_any

    pool.shutdown()

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
        "clear_rate": clears / max(1, n_episodes),
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
    # Append to per-game eval log for scoreboard consumption.
    eval_log = ckpt_dir / "eval.jsonl"
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
    parser.add_argument("--level-clear", action="store_true", dest="level_clear",
                        help="With --sequential, score 'cleared the level it "
                             "STARTED in' (a forward area/level transition out "
                             "of the warm-start level, warp-guarded) instead of "
                             "the full World-1 chain. The per-level gate metric.")
    args = parser.parse_args()

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
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
