"""Play back the best trained policy for a game and record it.

`make demo GAME=mario` loads the most *playable* checkpoint (the
retained winner first — see src/training/checkpointing.py — then the
best eval clear_rate, then the latest), plays one deterministic
(argmax) episode from the profile's start state, records the rendered
frames to `demos/<game>_iter<N>.gif`, and reports the outcome (furthest
stage + whether the game/level was cleared, via the same Rust reward fn
the trainer uses).

This is the "show me it works" command: it must play a REAL win, not
whatever the latest (possibly self-collapsed) checkpoint happens to be.
`--latest` forces the freshest checkpoint and `--iter N` pins an exact
one, for debugging current training progress.

Works for both policy families: tile-encoder games (SMB) use the MLP +
RAM-decoded obs; pixel games (Contra/Castlevania/Mega Man) use the CNN +
84x84 preprocessed obs.

GIF (not MP4) so it works without ffmpeg; downsampled to keep it small.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import FrameStacker, TileFeatureStacker  # noqa: E402
from src.models.policy_network import PolicyNetwork  # noqa: E402
from src.models.tile_policy import TilePolicyNetwork  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, derive_checkpoint_dir, resolve_encoder,
)
from src.training.checkpointing import (  # noqa: E402
    find_latest_trained_checkpoint, find_playable_checkpoint,
)
from train_game import DEFAULT_PROFILES, DEFAULT_ROMS, resolve_profile_path  # noqa: E402


def _is_tile(profile: dict) -> bool:
    return (profile.get("reinforce", {}) or {}).get("encoder") in ("smb_tiles",)


def resolve_demo_checkpoint(
    ckpt_dir: Path,
    game: str,
    latest: bool = False,
    iter_: Optional[int] = None,
) -> Optional[Path]:
    """Pick which checkpoint to play back.

    Default prefers the retained winner (`winners/best.pt`), then the best
    eval-history clear_rate, then the latest — so the demo plays a real
    win. `--iter N` pins an exact `vanilla_ppo_iter_NNNNN.pt`; `--latest`
    forces the freshest trained checkpoint.
    """
    if iter_ is not None:
        pinned = ckpt_dir / f"vanilla_ppo_iter_{iter_:05d}.pt"
        return pinned if pinned.exists() else None
    if latest:
        return find_latest_trained_checkpoint(ckpt_dir)
    return find_playable_checkpoint(game, ckpt_dir)


def demo(game: str, profile_path: Path, rom_path: str, max_steps: int,
         out_dir: Path, fps: int, stride: int,
         latest: bool = False, iter_: Optional[int] = None) -> dict:
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    ckpt_dir = derive_checkpoint_dir("./checkpoints", profile.get("name"))
    ckpt = resolve_demo_checkpoint(ckpt_dir, game, latest=latest, iter_=iter_)
    if ckpt is None:
        return {"game": game, "status": "no_checkpoint", "ckpt_dir": str(ckpt_dir)}

    bitmasks = action_space_to_bitmasks(profile["action_space"])
    tile = _is_tile(profile)
    if tile:
        extractor, feat_dim, stacked = resolve_encoder(profile)
        stacker = TileFeatureStacker(stack_size=stacked // feat_dim, feature_dim=feat_dim)
        net = TilePolicyNetwork(num_actions=len(bitmasks), feature_dim=stacked)
    else:
        stacker = FrameStacker(stack_size=4, target_size=84)
        net = PolicyNetwork(
            num_actions=len(bitmasks),
            encoder=(profile.get("reinforce", {}) or {}).get("encoder", "nature_dqn"),
        )
    state = torch.load(str(ckpt), map_location="cpu")
    # Same fail-loud contract as eval: load non-strict so extra aux heads
    # are tolerated, but refuse to demo a policy missing core weights —
    # a silently half-random policy would "look broken" and misreport.
    load_res = net.load_state_dict(state["net_state_dict"], strict=False)
    if load_res.missing_keys:
        return {"game": game, "status": "checkpoint_mismatch",
                "missing_keys": list(load_res.missing_keys), "checkpoint": str(ckpt)}
    net.eval()

    start = profile.get("start_state_path")
    # NOT headless — we need the real RGB frames to record.
    pool = Pool(rom_path=rom_path, num_workers=1, frame_skip=int(profile.get("frame_skip", 4)),
                start_state_path=str(start) if start and Path(str(start)).exists() else None)
    reward_fn = build_reward_function(profile)
    reward_fn.reset()
    pool.reset_all()
    init = pool.step_all(np.zeros(1, dtype=np.uint8))
    if tile:
        obs = stacker.reset(extractor.extract(init[0][2]))
    else:
        obs = stacker.reset(init[0][0], init[0][1])

    frames = [np.asarray(init[0][0], dtype=np.uint8)]
    cleared = False
    max_byte = 0
    step = 0
    for step in range(max_steps):
        x = torch.from_numpy(obs[None, :]).float()
        if not tile:
            x = x.div_(255.0)
        with torch.no_grad():
            logits, _ = net.forward_ac(x)
            a = int(torch.argmax(logits[0]).item())
        r = pool.step_all(np.array([bitmasks[a]], dtype=np.uint8))
        ram = r[0][2]
        _, rew_done, _ = reward_fn.compute(ram, action=int(bitmasks[a]))
        frames.append(np.asarray(r[0][0], dtype=np.uint8))
        byte = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)
        max_byte = max(max_byte, byte)
        if reward_fn.episode_success():
            cleared = True
        obs = stacker.push(extractor.extract(ram)) if tile else stacker.push(r[0][0], r[0][1])
        if rew_done or bool(r[0][3]) or cleared:
            break
    pool.shutdown()

    out_dir.mkdir(parents=True, exist_ok=True)
    it = int(Path(ckpt).stem.rsplit("_", 1)[-1]) if "_" in Path(ckpt).stem else 0
    gif_path = out_dir / f"{game}_iter{it:05d}.gif"
    try:
        import imageio.v2 as imageio
        sub = frames[::max(1, stride)]
        imageio.mimsave(str(gif_path), sub, duration=1.0 / max(1, fps), loop=0)
        wrote = str(gif_path)
    except Exception as e:
        wrote = f"(gif write failed: {e})"

    return {
        "game": game, "status": "ok", "checkpoint": str(ckpt),
        "cleared": cleared, "steps": step + 1, "max_byte_seen": int(max_byte),
        "frames": len(frames), "gif": wrote,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True,
                   help=f"Game logical name: {sorted(set(DEFAULT_PROFILES))}")
    p.add_argument("--profile", default=None)
    p.add_argument("--rom", default=None)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--out", default="demos")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame in the gif")
    p.add_argument("--latest", action="store_true",
                   help="Play the freshest trained checkpoint instead of the "
                        "retained winner.")
    p.add_argument("--iter", type=int, default=None, dest="iter_", metavar="N",
                   help="Play an exact vanilla_ppo_iter_NNNNN.pt by iteration number.")
    args = p.parse_args()

    profile_path = resolve_profile_path(args.game, args.profile)
    rom = args.rom or DEFAULT_ROMS.get(args.game.lower().strip())
    if not rom or not Path(rom).exists():
        print(f"no ROM for {args.game!r}: {rom}", file=sys.stderr)
        return 1
    res = demo(args.game, profile_path, rom, args.max_steps, Path(args.out),
               args.fps, args.stride, latest=args.latest, iter_=args.iter_)
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
