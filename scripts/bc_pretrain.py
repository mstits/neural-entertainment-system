"""Behavior-cloning pretrain -> vanilla_ppo seed checkpoint.

Replays a recorded BC tape (`.state.bin` per-frame button masks) into
(obs, action, reward) pairs, supervised-pretrains a fresh policy on them,
and writes the result as a `vanilla_ppo_iter_00000.pt` checkpoint that the
vanilla_ppo auto-resume loads as its initial policy. Decouples BC (which
the trainer only wires into the GA population) from the vanilla_ppo path.

Usage:
  python scripts/bc_pretrain.py --profile configs/smb_1_3.yaml \
      --rom "roms/Super Mario Bros. (World).nes" \
      --demo "roms/<your_recording>.bc_demo.state.bin" \
      --out checkpoints/mario_1_3/vanilla_ppo_iter_00000.pt --epochs 40
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.behavior_cloning import build_dataset, pretrain  # noqa: E402
from src.models.tile_policy import TilePolicyNetwork  # noqa: E402
from src.models.policy_network import PolicyNetwork  # noqa: E402
from src.training.profile_utils import resolve_encoder  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rom", required=True)
    ap.add_argument("--demo", required=True, help="BC tape .state.bin (or colon-joined list)")
    ap.add_argument("--out", required=True, help="output checkpoint path")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--start-state", default=None,
                    help="warm-start save-state the demo was recorded from "
                         "(e.g. a stage_1_4.state for a focused 1-4 demo); "
                         "BC replays the tape from this state, not cold-boot")
    args = ap.parse_args()

    profile = yaml.safe_load(open(args.profile))
    action_space = profile["action_space"]
    is_tile = profile["reinforce"].get("encoder") == "smb_tiles"
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    tile_frame_stack = int(profile["reinforce"].get("tile_frame_stack", 4)) if is_tile else 1
    reward_fn = build_reward_function(profile)

    demos = args.demo.split(":") if ":" in args.demo else args.demo
    states, actions, rewards, boundaries = build_dataset(
        rom_path=args.rom,
        demo_path=demos,
        action_space=action_space,
        frame_skip=args.frame_skip,
        reward_fn=reward_fn,
        tile_extractor=extractor if is_tile else None,
        tile_frame_stack=tile_frame_stack,
        start_state_path=args.start_state,
    )
    print(f"BC dataset: {states.shape[0]} pairs, shape={tuple(states.shape)}, "
          f"demo total reward={float(rewards.sum()):.1f}, boundaries={boundaries}")
    # Action distribution — confirms the demo isn't all one button.
    import numpy as np
    print("action histogram:", np.bincount(actions.numpy(), minlength=len(action_space)).tolist())

    if is_tile:
        net = TilePolicyNetwork(num_actions=len(action_space), feature_dim=stacked_dim)
    else:
        net = PolicyNetwork(num_actions=len(action_space))

    loss = pretrain(
        net, (states, actions, rewards),
        epochs=args.epochs, device=torch.device("cpu"),
        use_reward_weighting=True, val_fraction=0.1,
        normalize_obs=not is_tile, episode_boundaries=boundaries,
    )
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net_state_dict": net.state_dict()}, str(out))
    print(f"BC pretrain done; final loss={loss:.4f}; saved seed -> {out}")

    # Verify it reloads into a fresh net (the shape contract vanilla_ppo needs).
    chk = TilePolicyNetwork(num_actions=len(action_space), feature_dim=stacked_dim) if is_tile \
        else PolicyNetwork(num_actions=len(action_space))
    res = chk.load_state_dict(torch.load(str(out))["net_state_dict"], strict=False)
    print("reload check:", "OK" if not res.missing_keys else f"MISSING {res.missing_keys}")


if __name__ == "__main__":
    main()
