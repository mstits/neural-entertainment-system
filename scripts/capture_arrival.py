"""Capture a live arrival state (and optionally the action stream) by
running a checkpoint greedily from a start state.

The level-cracking loop's first tool: seams fail on ARRIVAL LINEAGE
(scroll-history enemy state), so harvesters and clones must root at the
state the chain actually produces — not at a previously banked handoff.
Typical uses:

  # the exact state a gx-routed switch hands the incoming net
  python scripts/capture_arrival.py --profile configs/smb_oneshot_tiles.yaml \
      --checkpoint checkpoints/mario_2_1/robust_2_1_front.pt \
      --start-state runs/chain_handoffs/handoff_2-1.state \
      --at-gx 1440 --out checkpoints/handoffs/seam_2_1_chain_arrival.state

  # a policy prefix for the harvester (record actions until death)
  python scripts/capture_arrival.py --profile configs/smb_2_2_water.yaml \
      --checkpoint checkpoints/mario_2_2_water/winners/best.pt \
      --start-state runs/chain_handoffs/handoff_2-2.state \
      --record-actions checkpoints/go_explore_2_2/prefix_2_2.npy \
      --trim-tail 8

The switch condition mirrors the composite router exactly (gx >= at_gx,
2-frame confirm) so a captured arrival is bit-identical to what the
routed chain hands the incoming stage. Replay convention downstream is
load -> ONE noop -> act (replay_to_demos.py / eval_game --start-state);
route clone stages with noop_pad: 1 to reproduce it in-chain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.composite_policy import build_level_net  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
DEATH_STATES = (6, 11)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--at-gx", type=int, default=None,
                    help="Capture --out at this gx (2-frame confirm, "
                         "matching the composite router's gx_switches).")
    ap.add_argument("--out", default=None,
                    help="Arrival state path (requires --at-gx).")
    ap.add_argument("--record-actions", default=None,
                    help="Record the greedy action-index stream here "
                         "(.npy); runs until death or --max-steps.")
    ap.add_argument("--trim-tail", type=int, default=8,
                    help="Drop this many trailing actions from a "
                         "recorded stream (the death approach) so a "
                         "harvester prefix ends on a live state.")
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()
    if args.at_gx is not None and not args.out:
        ap.error("--at-gx requires --out")
    if args.at_gx is None and not args.record_actions:
        ap.error("nothing to do: pass --at-gx/--out or --record-actions")

    profile = yaml.safe_load(Path(args.profile).read_text())
    net = build_level_net(profile, args.checkpoint, torch.device("cpu"),
                          label="capture")
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    pool = Pool(rom_path=ROM, num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.reset_all()
    pool.load_worker_state(0, Path(args.start_state).read_bytes())
    sr = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
    adapter = net.new_adapter()
    obs = adapter.reset(sr[2], sr[1])
    hidden = net.initial_hidden()

    actions: list[int] = []
    confirm = 0
    best_gx = 0
    for step in range(args.max_steps):
        t = adapter.to_tensor(obs, torch.device("cpu"))
        bitmask, hidden = net.forward_greedy(t, hidden)
        sr = pool.step_all(np.array([bitmask], dtype=np.uint8))[0]
        ram = sr[2]
        gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
        best_gx = max(best_gx, gx)
        if args.record_actions:
            actions.append(bitmasks.index(bitmask)
                           if bitmask in bitmasks else 0)
        if args.at_gx is not None:
            confirm = confirm + 1 if gx >= args.at_gx else 0
            if confirm >= 2:
                phase = (int(ram[0x0009]) >> 2) & 7
                Path(args.out).write_bytes(pool.save_worker_state(0))
                print(f"captured {args.out}: gx={gx} phase={phase} "
                      f"step={step}")
                pool.shutdown()
                return 0
        if int(ram[0x000E]) in DEATH_STATES or bool(sr[3]):
            print(f"died at step {step}, gx {gx} (best {best_gx})")
            break
        obs = adapter.push(ram, sr[1])

    if args.record_actions:
        trim = max(0, len(actions) - args.trim_tail)
        np.save(args.record_actions,
                np.array(actions[:trim], dtype=np.int64))
        print(f"saved {trim} actions -> {args.record_actions} "
              f"(best gx {best_gx})")
        pool.shutdown()
        return 0
    print(f"never crossed gx {args.at_gx} (best {best_gx})")
    pool.shutdown()
    return 1


if __name__ == "__main__":
    sys.exit(main())
