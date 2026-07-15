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
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, derive_checkpoint_dir, resolve_encoder,
)
from src.training.checkpointing import (  # noqa: E402
    find_latest_trained_checkpoint, find_playable_checkpoint,
)
from src.training.smb_sequential import (  # noqa: E402
    SequentialTracker, level_label,
)

# Reuse the same logical → profile + ROM tables as the launcher so
# eval and train always agree on which files to read.
sys.path.insert(0, str(ROOT / "scripts"))
from train_game import DEFAULT_PROFILES, DEFAULT_ROMS, resolve_profile_path  # noqa: E402


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

    # Profile-driven action space + observation encoder. No game-
    # specific constants live in this script — adding a 7th game is a
    # profile + a trained checkpoint, never an eval-script edit.
    try:
        bitmasks = action_space_to_bitmasks(profile["action_space"])
        extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    except (KeyError, ValueError) as e:
        return {
            "game": game,
            "status": "unsupported_profile",
            "detail": str(e),
            "ckpt_dir": str(ckpt_dir),
        }
    stack_size = stacked_dim // feature_dim

    state = torch.load(str(ckpt), map_location="cpu")
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

    start_state = profile.get("start_state_path") or (
        Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
    )
    pool = Pool(
        rom_path=rom_path, num_workers=1, frame_skip=4,
        start_state_path=str(start_state) if Path(str(start_state)).exists() else None,
    )

    # Optional curriculum-stage warm-start. By default eval boots from
    # the profile start state (live level-1 gameplay), so it only
    # measures the FIRST stage. A curriculum-trained policy spends most
    # of its training on later stages; pass --stage N to load
    # smb_curriculum/stage_NN.state and measure that stage instead.
    stage_blob: Optional[bytes] = None
    if stage is not None:
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

    stacker = TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
    # The reward function is the SAME Rust object the trainer uses, so
    # the eval's notion of "cleared" (episode_success) and "return"
    # (sum of per-step rewards) matches training exactly — no separate
    # heuristic to drift out of sync.
    reward_fn = build_reward_function(profile)

    device = torch.device("cpu")
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
        obs = stacker.reset(extractor.extract(init[0][2]))
        # Fresh hidden state per episode — the GRU must not carry memory
        # across episode boundaries.
        hidden = net.initial_hidden(1, device) if is_recurrent else None
        # In --sequential mode reconstruct World-1 progression from RAM
        # independently of episode_success(). episode_success() latches on
        # `cleared_any` at the FIRST 1-1 flagpole, so it can never observe a
        # sequential 1-4 clear; the tracker's `won` semantics can.
        tracker = SequentialTracker() if sequential else None
        ep_return = 0.0
        ep_max_byte = 0
        ep_cleared = False
        step = 0
        for step in range(max_steps):
            x = torch.from_numpy(obs[None, :]).float()
            with torch.no_grad():
                if is_recurrent:
                    logits, _, hidden = net.forward_ac_recurrent(x, hidden)
                else:
                    logits, _ = net.forward_ac(x)
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
            obs = stacker.push(extractor.extract(ram))
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
    if sequential:
        # THE primary campaign metric: sequential World-1 clear rate + how far
        # the greedy policy actually reaches, with warps reported "beyond" but
        # never counted as a clear. `clear_rate` above stays the 1-1-latch rate
        # for back-compat; `seq_clear_rate` is the DoD number every phase gates
        # on. `furthest_*_level` are the deepest (world, level) across episodes.
        result["sequential"] = True
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
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
