"""Diagnose where Mario dies in 1-3 under the trained policy.

Loads the latest vanilla_ppo checkpoint + stage_02.state (the 1-3
warm-start save state), runs one rollout, records the trajectory,
saves a screenshot at the death frame, and reports:
  - x at death (vs the observed mean_return ≈ 830)
  - player_state at death (DYING/DEAD/lives-lost)
  - y at death (was Mario in the air? fell in a pit?)
  - last 10 actions before death
  - frames at t-5, t-2, t-1, t (death) saved as PNGs

Run a few times to see if the death point is stochastic or
deterministic.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nes_core import Pool
from src.models.tile_policy import TilePolicyNetwork
from src.emulation.tile_observations import get_extractor
from src.emulation.frame_utils import TileFeatureStacker

OUT_DIR = Path("/tmp/death_diag")
OUT_DIR.mkdir(exist_ok=True)

ACTIONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40]
ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]


def find_latest_ckpt() -> Path:
    ckpts = sorted(
        Path("checkpoints").glob("vanilla_ppo_iter_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not ckpts:
        raise RuntimeError("no vanilla_ppo_iter_*.pt found")
    return ckpts[-1]


def run_one(run_id: int) -> None:
    ckpt_path = find_latest_ckpt()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    net = TilePolicyNetwork(num_actions=len(ACTIONS), feature_dim=700)
    net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()

    pool = Pool(
        rom_path="roms/Super Mario Bros. (World).nes",
        num_workers=1, frame_skip=4,
        start_state_path="roms/Super Mario Bros. (World)_start.state.bin",
    )
    pool.reset_all()

    stage_path = Path("checkpoints/smb_curriculum/stage_02.state")
    blob = stage_path.read_bytes()
    pool.load_worker_state(0, blob)

    extractor = get_extractor("smb_tiles")
    stacker = TileFeatureStacker(stack_size=4, feature_dim=175)

    r = pool.step_all(np.zeros(1, dtype=np.uint8))
    obs = stacker.reset(extractor.extract(r[0][2]))

    history = []  # (step, x, y, p_state, lives, byte, action_idx, frame)

    for step in range(1024):
        batch = torch.from_numpy(obs[None, :]).float()
        with torch.no_grad():
            logits, _ = net.forward_ac(batch)
            action_idx = int(torch.multinomial(
                torch.softmax(logits, dim=-1)[0], 1
            ).item())
        r = pool.step_all(np.array([ACTIONS[action_idx]], dtype=np.uint8))
        ram = r[0][2]
        frame = r[0][0]
        x = 256 * int(ram[0x006D]) + int(ram[0x0086])
        y = int(ram[0x00CE])
        p_state = int(ram[0x000E])
        float_state = int(ram[0x001D])
        lives = int(ram[0x075A])
        wbyte = int(ram[0x075F])
        lbyte = int(ram[0x0760])
        history.append((step, x, y, p_state, float_state, lives, wbyte, lbyte,
                        action_idx, frame))
        obs = stacker.push(extractor.extract(ram))

        # Death detected: player_state DYING(0x0B) or DEAD(0x06), or
        # lives dropped. Also break on lives==0 (game over).
        dying = p_state in (0x0B, 0x06)
        lives_lost = step > 0 and lives < history[-2][5]
        if dying or lives_lost:
            print(f"[run {run_id}] DEATH at step {step}: x={x} y={y} "
                  f"p_state=0x{p_state:02x} lives={lives} W-L byte=({wbyte},{lbyte})")
            # Save the last 5 frames + the death frame.
            for off in (5, 2, 1, 0):
                idx = max(0, len(history) - 1 - off)
                s, hx, hy, hps, hfs, hl, hwb, hlb, ha, hf = history[idx]
                tag = f"run{run_id}_step{s:04d}_x{hx:04d}_a{ha}_{ACTION_NAMES[ha]}"
                if off == 0:
                    tag += "_DEATH"
                Image.fromarray(hf, mode="RGB").save(str(OUT_DIR / f"{tag}.png"))
            # Print last 10 actions.
            recent = history[-10:]
            print(f"  last 10 actions: " + " ".join(
                f"{ACTION_NAMES[h[8]]}@x{h[1]}" for h in recent
            ))
            break
    else:
        # Survived rollout end.
        last = history[-1]
        s, hx, hy, hps, hfs, hl, hwb, hlb, ha, hf = last
        print(f"[run {run_id}] SURVIVED rollout: step={s} x={hx} W-L=({hwb},{hlb})")
        Image.fromarray(hf, mode="RGB").save(
            str(OUT_DIR / f"run{run_id}_survived_x{hx:04d}.png")
        )

    pool.shutdown()


def main() -> None:
    print(f"Latest ckpt: {find_latest_ckpt()}")
    for run_id in range(5):
        run_one(run_id)
    print(f"\nFrames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
