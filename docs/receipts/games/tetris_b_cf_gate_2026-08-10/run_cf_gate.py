"""Run the REAL counterfactual gate (Solver.counterfactual_probe, unmodified,
imported from scripts/go_explore_solve.py) against a measured B-TYPE winning
trace. Nothing in the solver is touched; the probe is bound to a shim carrying
exactly the attributes it reads.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import go_explore_solve as G   # noqa: E402

PROFILE = REPO / "configs/tetris_b.yaml"
ROOT = REPO / "roms/Tetris (USA)_btype_start.state.bin"


def build_shim(profile_path, root_path):
    prof = yaml.safe_load(Path(profile_path).read_text())
    shim = types.SimpleNamespace()
    shim.bitmasks = G.action_space_to_bitmasks(prof["action_space"])
    shim.game = G.make_game(prof)
    shim.hw_flags = G.resolve_hw_flags(prof, None)
    shim.frame_skip = int(prof.get("frame_skip", 4))
    shim.roots = {"entrance": {"path": str(root_path)}}
    shim.cf_branches, shim.cf_perturb_p = 8, 0.25
    shim.cf_agree, shim.cf_seed, shim.cf_pre_steps = 0.5, 0, 32
    shim._needs_apu = False
    # exactly what Solver.seed() records
    pool = G.Pool(rom_path=shim.game.rom, num_workers=1,
                  frame_skip=shim.frame_skip)
    G.apply_hw_flags(pool, shim.hw_flags)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, Path(root_path).read_bytes())
    ram = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
    shim.start_wd = shim.game.level_key(ram)
    shim.start_lives = shim.game.lives(ram)
    shim.game.note_start(ram)
    pool.shutdown()
    return shim


def main():
    trace = np.load(sys.argv[1]).tolist()
    pre = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    shim = build_shim(PROFILE, ROOT)
    shim.cf_pre_steps = pre
    print(f"start_wd={shim.start_wd} start_lives={shim.start_lives} "
          f"baseline=0x{shim.game._clear_baseline:02X} "
          f"margin={shim.game.clear_verify_margin()} "
          f"budget={shim.game.clear_observation_budget()} "
          f"n_actions={len(trace)} pre={pre}", flush=True)
    probe = G.Solver.counterfactual_probe.__get__(shim, type(shim))
    res = probe("entrance", trace)
    print(json.dumps(res, indent=1, default=str))
    print("\nVERDICT:", res["verdict"], "ok=", res["ok"],
          "agreement=", res["agreement"], "threshold=", res["threshold"])


if __name__ == "__main__":
    main()
