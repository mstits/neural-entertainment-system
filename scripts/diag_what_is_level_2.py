"""Capture screen frames at each $0760 (level byte) transition."""
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

OUT_DIR = Path("/tmp/level_check")
OUT_DIR.mkdir(exist_ok=True)

ACTIONS = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40]


def main() -> None:
    ckpt = torch.load("checkpoints/vanilla_ppo_iter_00420.pt", map_location="cpu")
    net = TilePolicyNetwork(num_actions=len(ACTIONS), feature_dim=700)
    net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()

    pool = Pool(rom_path="roms/Super Mario Bros. (World).nes",
                num_workers=1, frame_skip=4,
                start_state_path="roms/Super Mario Bros. (World)_start.state.bin")
    extractor = get_extractor("smb_tiles")
    stacker = TileFeatureStacker(stack_size=4, feature_dim=175)
    init = pool.reset_all()
    obs = stacker.reset(extractor.extract(init[0][2]))

    last_level = 0
    actions_np = np.zeros(1, dtype=np.uint8)
    seen_levels: dict[int, bool] = {0: True}

    for step in range(3000):
        batch = torch.from_numpy(obs[None, :]).float()
        with torch.no_grad():
            logits, _ = net.forward_ac(batch)
            action_idx = int(torch.multinomial(
                torch.softmax(logits, dim=-1)[0], 1).item())
        actions_np[0] = ACTIONS[action_idx]
        r = pool.step_all(actions_np)
        ram = r[0][2]
        world = int(ram[0x075F])
        level = int(ram[0x0760])
        x = 256 * int(ram[0x006D]) + int(ram[0x0086])
        ps = int(ram[0x001D])
        if level != last_level:
            last_level = level
            for offset_label, delay_steps in [
                ("imm", 0), ("after_60", 60), ("after_180", 180)
            ]:
                if delay_steps > 0:
                    for _ in range(delay_steps):
                        r = pool.step_all(np.array([ACTIONS[3]], dtype=np.uint8))
                ram = r[0][2]
                w = int(ram[0x075F])
                l = int(ram[0x0760])
                x = 256 * int(ram[0x006D]) + int(ram[0x0086])
                ps = int(ram[0x001D])
                png = OUT_DIR / (
                    f"step{step:05d}_{offset_label}_"
                    f"W{w+1}L{l+1}_x{x:04d}_ps{ps}.png"
                )
                Image.fromarray(r[0][0], mode="RGB").save(str(png))
                print(f"step={step:5d} {offset_label:8s}  "
                      f"W-L={w+1}-{l+1}  x={x:4d}  pstate={ps}  -> {png.name}")
            seen_levels[level] = True
            if 2 in seen_levels:
                break
        obs = stacker.push(extractor.extract(ram))
    pool.shutdown()
    print(f"\nFrames -> {OUT_DIR}")


if __name__ == "__main__":
    main()
