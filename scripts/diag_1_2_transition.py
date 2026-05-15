"""Diagnostic: does Mario ever transition from 1-1 to 1-2?

Loads the latest vanilla_ppo checkpoint, runs the policy on a single
env headless for `max_steps` rollout steps, and logs the world/level
RAM bytes (and `Player_State` at $001D) every step around interesting
events.

The trainer's per-iter telemetry showed `max-W-L: 1-1=30` even with
21+ clears per iter — meaning the policy touches the flagpole but
the game never appears to advance to 1-2. This script answers:
  1. Does float_state==3 ever fire during the rollout? (flagpole)
  2. After that fires, does ram[$0760] ever become 1 (1-2)?
  3. If not, what's the game state around the moment we'd expect
     the transition?

Usage: `.venv/bin/python scripts/diag_1_2_transition.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nes_core import Pool  # type: ignore[import-not-found]
from src.models.tile_policy import TilePolicyNetwork
from src.emulation.tile_observations import get_extractor
from src.emulation.frame_utils import TileFeatureStacker


CKPT = ROOT / "checkpoints" / "vanilla_ppo_iter_00420.pt"
ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
START_STATE = ROOT / "roms" / "Super Mario Bros. (World)_start.state.bin"

MAX_STEPS = 3000  # plenty of time for clear + cutscene + 1-2 entry
NUM_WORKERS = 1
FRAME_SKIP = 4

# SMB action space (must match what training used). Standard 7-action.
ACTIONS: list[int] = [
    0x00,        # NOOP
    0x80,        # RIGHT
    0x80 | 0x01, # RIGHT + A   (right + jump)
    0x80 | 0x02, # RIGHT + B   (right + run)
    0x80 | 0x01 | 0x02,  # RIGHT + A + B (run-jump)
    0x01,        # A
    0x40,        # LEFT
]


def main() -> None:
    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(str(CKPT), map_location="cpu")
    state_dict = ckpt["net_state_dict"]

    # Infer feature dim from the checkpoint (700 = 13*13*4 stacked tiles).
    base_dim = 175  # SMB tile extractor base
    stack = 4
    total_dim = base_dim * stack

    net = TilePolicyNetwork(
        num_actions=len(ACTIONS),
        feature_dim=total_dim,
    )
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    print(f"Building pool with rom={ROM.name}, start_state={START_STATE.name}")
    pool = Pool(
        rom_path=str(ROM),
        num_workers=NUM_WORKERS,
        frame_skip=FRAME_SKIP,
        start_state_path=str(START_STATE),
    )
    pool.reset_all()

    extractor = get_extractor("smb_tiles")
    stacker = TileFeatureStacker(stack_size=stack, feature_dim=base_dim)

    init = pool.reset_all()
    # Pool returns list of (frame, preprocessed, ram, done) tuples.
    ram0 = init[0][2]
    obs = stacker.reset(extractor.extract(ram0))

    # Trackers
    first_flagpole_step = -1
    last_world_level_logged = (-1, -1)
    flagpole_window_logged = 0

    print(f"\nRunning {MAX_STEPS} steps (= {MAX_STEPS * FRAME_SKIP} NES frames)...")
    print(f"{'step':>5}  {'W':>2}  {'L':>2}  {'p_state':>7}  {'x':>4}  {'note'}")

    actions_np = np.zeros(NUM_WORKERS, dtype=np.uint8)
    for step in range(MAX_STEPS):
        batch = torch.from_numpy(obs[None, :]).float()
        with torch.no_grad():
            logits, _ = net.forward_ac(batch)
            probs = torch.softmax(logits, dim=-1)
            action_idx = int(torch.multinomial(probs[0], 1).item())
        actions_np[0] = ACTIONS[action_idx]

        results = pool.step_all(actions_np)
        ram = results[0][2]
        world = int(ram[0x075F])
        level = int(ram[0x0760])
        p_state = int(ram[0x001D])
        x = 256 * int(ram[0x006D]) + int(ram[0x0086])

        # Log every world/level change.
        if (world, level) != last_world_level_logged:
            print(f"{step:>5}  {world+1:>2}  {level+1:>2}  {p_state:>7}  "
                  f"{x:>4}  ** W-L CHANGE **")
            last_world_level_logged = (world, level)

        # Catch first flagpole entry.
        if p_state == 3 and first_flagpole_step < 0:
            first_flagpole_step = step
            print(f"{step:>5}  {world+1:>2}  {level+1:>2}  {p_state:>7}  "
                  f"{x:>4}  ** first p_state=3 (flagpole) **")

        # Capture Mario *playing 1-2 with a non-stale PPU*:
        # World 1, Level 2, p_state in {0,1,2}, AND x in [40, 150] so
        # the title-card sequence has fully cleared and the PPU
        # nametable has been updated with the actual 1-2 level pixels
        # (not the stale "WORLD 1-1" title-card text from cold boot).
        # Without the x >= 40 gate the GUI renders the previous PPU
        # frame which shows the old title-card text and looks like
        # the warm-start did nothing.
        if (world == 0 and level == 1 and p_state in (0, 1, 2)
                and 40 <= x <= 150 and first_flagpole_step >= 0
                and step > first_flagpole_step + 200):
            out_path = ROOT / "checkpoints" / "world1-2_spawn.state"
            blob = pool.save_worker_state(0)
            if blob is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(bytes(blob))
                print(f"\n*** CAPTURED 1-2 SPAWN STATE ***")
                print(f"    step={step} W=1 L=2 x={x} p_state={p_state}")
                print(f"    bytes={len(blob)} -> {out_path}")
                print(f"    (trainer's vanilla_ppo will warm-start half the "
                      f"pool from this state on next launch)")
                pool.shutdown()
                return

        # Log a window of 300 steps around the first flagpole event so
        # we can see whether the cutscene + level advance actually fires.
        if first_flagpole_step >= 0 and flagpole_window_logged < 300:
            if step % 10 == 0 or p_state != 0:
                print(f"{step:>5}  {world+1:>2}  {level+1:>2}  {p_state:>7}  "
                      f"{x:>4}")
            flagpole_window_logged += 1

        obs = stacker.push(extractor.extract(ram))

    print(f"\nDone. first_flagpole_step={first_flagpole_step}, "
          f"final W-L={last_world_level_logged[0]+1}-"
          f"{last_world_level_logged[1]+1}")
    pool.shutdown()


if __name__ == "__main__":
    main()
