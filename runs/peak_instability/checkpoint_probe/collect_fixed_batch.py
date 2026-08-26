"""Collect a fixed batch of real tile-mode observations for checkpoint probing.

Every v27/v28 seed config shares the same rom_path, start_state_path,
frame_skip, encoder (smb_tiles_pos) and action_space (6-way). So one batch,
collected once with a fixed action seed, is reused to probe every checkpoint
of every run -- the batch never depends on any checkpoint under test, which
keeps the entropy/logit-saturation comparison across iters uncontaminated by
which policy generated the states.

Actions are drawn uniform-random (not from any trained policy) from a fixed
RandomState so the batch is deterministic and not biased toward whichever
run's own behavior produced it. Episodes reset on death/pool-done so the
batch keeps covering fresh states instead of stalling on a post-death screen.

Output: fixed_batch.npy, shape (N, 712) int8 (raw stacked tile features,
matching the stacker's native dtype -- consumers cast to float themselves).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.emulation.tile_observations import get_extractor  # noqa: E402

ROM_PATH = str(REPO / "roms" / "Super Mario Bros. (World).nes")
START_STATE = str(REPO / "runs/live_show/smb_4_4_micro/entrance_start.state")
# 6-way action space shared verbatim by every v27/v28 seed config.
ACTION_SPACE = [
    [],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
]
BUTTON_BIT = {"A": 0x01, "B": 0x02, "up": 0x10, "down": 0x20, "left": 0x40, "right": 0x80}


def bitmask(buttons):
    m = 0
    for b in buttons:
        m |= BUTTON_BIT[b]
    return m


BITMASKS = [bitmask(a) for a in ACTION_SPACE]


def collect(n_obs: int = 800, max_steps_per_episode: int = 400, seed: int = 20260825):
    extractor = get_extractor("smb_tiles_pos")
    feature_dim = int(extractor.feature_dim)
    assert feature_dim == 178, feature_dim
    stacker = TileFeatureStacker(stack_size=4, feature_dim=feature_dim, dtype=np.int8)

    rng = np.random.RandomState(seed)
    pool = Pool(rom_path=ROM_PATH, num_workers=1, frame_skip=4, start_state_path=START_STATE)

    obs_list = []
    episode = 0
    while len(obs_list) < n_obs:
        pool.reset_all()
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        obs = stacker.reset(extractor.extract(init[0][2]))
        for step in range(max_steps_per_episode):
            a_idx = int(rng.randint(0, len(BITMASKS)))
            r = pool.step_all(np.array([BITMASKS[a_idx]], dtype=np.uint8))
            obs = stacker.push(extractor.extract(r[0][2]))
            obs_list.append(obs.copy())
            if len(obs_list) >= n_obs:
                break
            if bool(r[0][3]):
                break
        episode += 1

    batch = np.stack(obs_list, axis=0)
    print(f"collected {batch.shape} over {episode} episodes, seed={seed}")
    return batch


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    batch = collect()
    np.save(out_dir / "fixed_batch.npy", batch)
    print("saved", out_dir / "fixed_batch.npy")
