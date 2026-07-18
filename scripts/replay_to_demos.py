"""Deterministic action replay -> DemoBank-compatible npz.

(root save-state, action sequence) in; (obs, action) anchor pairs out,
rendered through the profile's encoder exactly as the trainer's
collection path builds them (resolve_encoder + TileFeatureStacker).
This is the ONE renderer downstream of every 2-1 demo generator
(go_explore_2_1.py solutions, searched sequences, banked prefixes), so
every bank the DQfD anchor consumes shares one obs-assembly path.

Replay convention (matches robustify_level.run_episode and the banked
ep231 prefix — verified: the prefix lands at gx1966 only under it):
load the root blob, take ONE NOOP step to materialize RAM, seed the
stacker from it, then step the actions in order. Pair i is (stacked
features BEFORE action i, action i).

Rows are stored as copies: the stacker reuses a single output buffer,
and a collector that stores views produces a bank of identical rows —
the aliasing failure DemoBank.from_npz's uniqueness guard rejects.

NOTE: the profile must name the encoder the consuming net was trained
on. The Run B anchor bank is v2 (smb_tiles_pos, 178 x 4 = 712 wide);
configs/smb_oneshot_tiles.yaml is the v1 175-dim encoder and will NOT
produce 712-wide rows.

Usage:
  python scripts/replay_to_demos.py \
      --start-state checkpoints/harvested_seeds/seed_2_1_gx1421_runway.state \
      --actions checkpoints/harvested_seeds/ep231_prefix_actions.npy \
      --profile checkpoints/mario_2_1_runB/weld_2_1_runB.yaml \
      --out checkpoints/harvested_seeds/demos_2_1_ep231.npz \
      [--root-id gx1421_runway]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, resolve_encoder,
)

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start-state", required=True,
                    help="Root save-state the actions replay from. Must be "
                         "the sequence's OWN root — traces do not transplant "
                         "across roots.")
    ap.add_argument("--actions", required=True,
                    help=".npy int action-index sequence (indices into the "
                         "profile's action_space).")
    ap.add_argument("--profile", required=True,
                    help="Profile yaml: action_space, frame_skip and the "
                         "encoder the pairs are rendered under.")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--root-id", default=None,
                    help="Provenance tag for the root state; written to the "
                         "sidecar json next to --out.")
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stacker = TileFeatureStacker(
        stack_size=stacked_dim // feature_dim, feature_dim=feature_dim,
    )
    actions = np.load(args.actions).astype(np.int64)
    assert actions.ndim == 1 and int(actions.max()) < len(bitmasks), \
        f"actions {actions.shape} max {actions.max()} vs {len(bitmasks)}-way"

    pool = Pool(rom_path=ROM, num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.load_worker_state(0, Path(args.start_state).read_bytes())
    r = pool.step_all(np.zeros(1, dtype=np.uint8))  # materialize RAM
    obs = stacker.reset(extractor.extract(r[0][2]))

    rows = []
    for a in actions:
        # .copy(): obs is a view into the stacker's reused ring buffer.
        rows.append((np.array(obs, dtype=np.int8, copy=True), int(a)))
        r = pool.step_all(np.array([bitmasks[int(a)]], dtype=np.uint8))
        obs = stacker.push(extractor.extract(r[0][2]))
    ram = r[0][2]
    final_gx = (int(ram[0x006D]) << 8) | int(ram[0x0086])
    pool.shutdown()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, n=1,
        obs_0=np.stack([o for o, _ in rows]).astype(np.int8),
        act_0=np.array([a for _, a in rows], dtype=np.int64),
    )
    sidecar = {
        "provenance": "search",
        "root_id": args.root_id or Path(args.start_state).stem,
        "root_state": str(args.start_state),
        "actions": str(args.actions),
        "profile": str(args.profile),
        "pairs": len(rows),
        "feature_dim": stacked_dim,
        "final_gx": final_gx,
        "final_player_state": int(ram[0x000E]),
        "final_world_disp": [int(ram[0x075F]), int(ram[0x075C])],
    }
    out.with_suffix(".provenance.json").write_text(
        json.dumps(sidecar, indent=2) + "\n")
    print(f"[replay_to_demos] {len(rows)} pairs ({stacked_dim}-wide) -> "
          f"{out}  (final gx {final_gx}, ps {sidecar['final_player_state']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
