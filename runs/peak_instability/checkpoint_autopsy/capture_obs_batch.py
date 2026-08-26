"""Capture a FIXED batch of real tile-mode observations for the
checkpoint autopsy (dimension: direct policy/weight measurement).

Why a captured (not synthesized) batch: the eval harness's tile obs is a
stack of 4 real SMBTileObservationV2 feature vectors decoded from real
NES RAM (see src/emulation/tile_observations/smb.py +
src/emulation/frame_utils.py:TileFeatureStacker). Hand-rolling random
712-dim vectors would put every checkpoint's net on inputs miles off
its training manifold (wrong feature ranges, wrong scalar/grid
correlations) and any entropy/logit reading from that would be
meaningless. This script drives the REAL pool from the REAL entrance
start-state that all 8 runs (v27 seed0-3, v28 seed0-3) trained and were
gated from, with a fixed pseudo-random action sequence (numpy
RandomState, hard seed), and records the stacked observation seen
before every action for ~1500 steps across a few episodes (resetting to
the same start state on death). This is captured ONCE. Every checkpoint
in the autopsy is then run forward on the exact same array -- so any
difference in output entropy / action distribution / logit stats
between checkpoints is attributable ONLY to the weights, not to a
different input distribution.

The action sequence is policy-agnostic (not drawn from any trained
checkpoint) specifically so the batch isn't pre-biased toward the
states any one run's own policy prefers to visit -- it's mostly
right-biased (SMB requires moving right to see anything past the
starting screen) with jump mixed in, which is the same qualitative
bias every one of these 8 runs' reward functions imposes anyway
(rightward progress is the terminal reward).

Output: obs_batch.npy, shape (N, 712) float32; and actions_used.npy,
shape (N,) uint8 (the bitmask executed at that step) for provenance.

Run:
    .venv/bin/python runs/peak_instability/checkpoint_autopsy/capture_obs_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.emulation.tile_observations import get_extractor  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = REPO_ROOT / "configs" / "mario_1_1_v28_seed3.yaml"
N_TARGET = 1536
MAX_STEPS_PER_EPISODE = 1500
SEED = 20260825


def main() -> None:
    with open(PROFILE_PATH) as f:
        profile = yaml.safe_load(f)
    rom_path = REPO_ROOT / profile["rom_path"]
    start_state_path = REPO_ROOT / profile["start_state_path"]
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    encoder_name = profile["reinforce"]["encoder"]
    extractor = get_extractor(encoder_name)
    feature_dim = extractor.feature_dim
    stack_size = 4
    print(f"rom={rom_path}")
    print(f"start_state={start_state_path}")
    print(f"bitmasks={bitmasks} feature_dim={feature_dim} stack_size={stack_size}")

    pool = Pool(
        rom_path=str(rom_path), num_workers=1, frame_skip=4,
        start_state_path=str(start_state_path),
    )

    rng = np.random.RandomState(SEED)
    stacker = TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []

    # Action-index probabilities over the 6-way SMB action space
    # ([], [right], [right,A], [right,B], [right,A,B], [A]) -- weighted
    # toward forward progress (this is what every one of the 8 runs'
    # reward shaping does too) but with enough NOOP/A-only mass that the
    # batch also contains idle and pure-jump states, not just one groove.
    n_actions = len(bitmasks)
    if n_actions == 6:
        p = np.array([0.05, 0.30, 0.25, 0.10, 0.25, 0.05])
    else:
        p = np.full(n_actions, 1.0 / n_actions)

    episode = 0
    while len(obs_list) < N_TARGET:
        pool.reset_all()
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        obs = stacker.reset(extractor.extract(init[0][2]))
        for step in range(MAX_STEPS_PER_EPISODE):
            if len(obs_list) >= N_TARGET:
                break
            obs_list.append(obs.astype(np.float32).copy())
            action_idx = int(rng.choice(n_actions, p=p))
            act_list.append(int(bitmasks[action_idx]))
            r = pool.step_all(np.array([bitmasks[action_idx]], dtype=np.uint8))
            ram = r[0][2]
            done = bool(r[0][3])
            obs = stacker.push(extractor.extract(ram))
            if done:
                break
        episode += 1
        print(f"episode {episode}: collected {len(obs_list)}/{N_TARGET}")

    pool.shutdown()

    obs_arr = np.stack(obs_list, axis=0)
    act_arr = np.array(act_list, dtype=np.uint8)
    np.save(OUT_DIR / "obs_batch.npy", obs_arr)
    np.save(OUT_DIR / "actions_used.npy", act_arr)
    print(f"saved obs_batch.npy shape={obs_arr.shape} dtype={obs_arr.dtype}")
    print(f"saved actions_used.npy shape={act_arr.shape}")
    print(f"obs_batch stats: mean={obs_arr.mean():.4f} std={obs_arr.std():.4f} "
          f"min={obs_arr.min():.4f} max={obs_arr.max():.4f}")


if __name__ == "__main__":
    main()
