"""Live win-predicate verification — drive the REAL game to a REAL win and
assert episode_success fires.

This is the antidote to the self-referential-test trap that let two
Castlevania win bugs ship this project (a unit test that writes the same
RAM address the code reads passes even when the address is wrong/dead).
These tests reach an actual game event on the real emulator and confirm
the reward's win predicate fires — no synthetic RAM.

Skip-guarded on the local trained checkpoint + curriculum states (ROMs and
save-states are not committed), so they run only where those artifacts
exist and never break a fresh clone's suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_CKPT = ROOT / "checkpoints" / "mario_1_4_go_explore" / "vanilla_ppo_iter_00790.pt"
_STATE = (
    ROOT / "checkpoints" / "super_mario_bros" / "smb_curriculum"
    / "stage_1_4_bw_x2300.state"  # real world-0, in 1-4 near Bowser
)
_PROFILE = ROOT / "configs" / "smb_1_4_go_explore.yaml"

pytestmark = pytest.mark.skipif(
    not (_ROM.exists() and _CKPT.exists() and _STATE.exists() and _PROFILE.exists()),
    reason="SMB ROM / trained 1-4 checkpoint / near-Bowser state not present (local artifacts).",
)


def test_smb_castle_clear_fires_episode_success_on_a_real_bowser_clear() -> None:
    """Warm-start a real 1-4 state, drive the trained policy to beat Bowser,
    and assert MarioReward.episode_success() fires when the world byte
    increments — verifying the F52 $075C-gated castle-clear predicate on the
    actual game, not synthetic RAM."""
    import torch
    import yaml
    from nes_core import Pool
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    from src.utils.reward_functions import build_reward_function
    from src.training.profile_utils import action_space_to_bitmasks, resolve_encoder

    profile = yaml.safe_load(open(_PROFILE))
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    extractor, feat_dim, stacked = resolve_encoder(profile)
    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt.get("net_state_dict", ckpt)
    net, _ = build_tile_policy_from_checkpoint(
        {"net_state_dict": sd}, num_actions=len(bitmasks), feature_dim=stacked
    )
    net.load_state_dict(sd, strict=False)
    net.eval()

    blob = _STATE.read_bytes()
    pool = Pool(rom_path=str(_ROM), num_workers=1, frame_skip=4, start_state_path=None)
    stacker = TileFeatureStacker(stack_size=stacked // feat_dim, feature_dim=feat_dim)
    reward_fn = build_reward_function(profile)
    g = torch.Generator().manual_seed(0)

    crossed = 0
    fired = 0
    for _ in range(8):
        pool.reset_all()
        pool.load_worker_state(0, blob)
        reward_fn.reset()
        init = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
        # Sanity: this MUST be a real in-1-4 world-0 state, else the test is
        # meaningless (a world-2 state trivially "crosses" — the trap that
        # first fooled this verification).
        assert init[0x075F] == 0 and init[0x075C] == 3, "not a real world-0 1-4 state"
        obs = stacker.reset(extractor.extract(init))
        for _ in range(400):
            x = torch.from_numpy(obs[None, :]).float()
            with torch.no_grad():
                logits = net.forward_ac(x)[0][0]
                a = int(torch.multinomial(torch.softmax(logits, 0), 1, generator=g).item())
            r = pool.step_all(np.array([bitmasks[a]], dtype=np.uint8))
            ram = r[0][2]
            reward_fn.compute(ram, action=int(bitmasks[a]))
            obs = stacker.push(extractor.extract(ram))
            if int(ram[0x075F]) > 0:  # beat Bowser -> world 2
                crossed += 1
                fired += int(reward_fn.episode_success())
                break
            if bool(r[0][3]):
                break
    pool.shutdown()

    assert crossed >= 3, f"policy did not reach enough real Bowser clears ({crossed})"
    assert fired == crossed, (
        f"episode_success fired {fired}/{crossed} real castle clears — the "
        f"win predicate does NOT fire on an actual Bowser clear"
    )
