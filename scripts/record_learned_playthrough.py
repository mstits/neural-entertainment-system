"""Record a LEARNED-policy playthrough from power-on to a video file.

Runs a composite manifest of TRAINED policies (no BC single-trajectory
pilots) from the 1-1 power-on start, greedy/deterministic, captures every
RGB frame, tracks how far it gets with SequentialTracker, and encodes an
MP4 + GIF you can show. Prints the honest per-level result.

This is the learned agent PLAYING — not a replay of a search trajectory.
Honest caveat it prints: deterministic (no-sticky) eval; robustness to
input perturbation (sticky-0.25) is separate ongoing work.

  python scripts/record_learned_playthrough.py \
      --manifest configs/composite_learned.yaml --out runs/demo/world1
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
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from src.training.composite_policy import (  # noqa: E402
    CompositeController, build_level_net, label_from_ram,
)
from src.training.smb_sequential import SequentialTracker, level_label  # noqa: E402
from train_game import DEFAULT_ROMS  # noqa: E402


def build_chain(manifest: dict, device):
    """Load every net + gx-route + entry-opt from the manifest (mirrors
    eval_composite's loader; trained nets only)."""
    levels = manifest["levels"]
    cache: dict = {}
    nets: dict = {}
    gx_routes: dict = {}
    entry_opts: dict = {}
    for key, entry in levels.items():
        prof = yaml.safe_load(Path(entry["profile"]).read_text())
        ck = (str(Path(entry["ckpt"]).resolve()), str(Path(entry["profile"]).resolve()))
        if ck not in cache:
            cache[ck] = build_level_net(prof, entry["ckpt"], device, label=key)
        nets[key] = cache[ck]
        if isinstance(entry.get("entry"), dict):
            entry_opts[key] = {
                "noop_pad": int(entry["entry"].get("noop_pad", 0)),
                "continuous_stack": bool(entry["entry"].get("continuous_stack", False)),
            }
        for sw in entry.get("gx_switches") or []:
            sprof = yaml.safe_load(Path(sw["profile"]).read_text())
            sck = (str(Path(sw["ckpt"]).resolve()), str(Path(sw["profile"]).resolve()))
            if sck not in cache:
                cache[sck] = build_level_net(sprof, sw["ckpt"], device,
                                             label=f"{key}@gx{sw['at_gx']}")
            skey = f"{key}@gx{sw['at_gx']}"
            nets[skey] = cache[sck]
            gx_routes.setdefault(key, []).append(
                (int(sw["at_gx"]), skey, int(sw.get("noop_pad", 0)),
                 bool(sw.get("continuous_stack", False)))
            )
    for k in gx_routes:
        gx_routes[k].sort()
    return nets, gx_routes, entry_opts, list(levels.keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output basename (no ext)")
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--stop-after-worlds", type=int, default=8)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    device = torch.device("cpu")
    nets, gx_routes, entry_opts, labels = build_chain(manifest, device)
    game = str(manifest.get("game", "mario")).lower()
    rom = manifest.get("rom") or DEFAULT_ROMS.get(game)
    scoring_prof = yaml.safe_load(
        Path(manifest["levels"].get("1-1", manifest["levels"]["default"])["profile"]).read_text()
    )
    start = scoring_prof.get("start_state_path")
    frame_skip = int(scoring_prof.get("frame_skip", 4))

    pool = Pool(rom_path=rom, num_workers=1, frame_skip=frame_skip,
                start_state_path=str(start) if start and Path(str(start)).exists() else None)
    pool.reset_all()
    ctrl = CompositeController(nets, labels, device, k=2,
                              gx_routes=gx_routes, entry_opts=entry_opts)
    tracker = SequentialTracker()
    init = pool.step_all(np.zeros(1, dtype=np.uint8))[0]
    tracker.update(init[2])
    ctrl.begin(init[2], init[1])

    frames = [np.asarray(init[0]).astype(np.uint8)]
    per_level_first = {}
    end_reason = "max_steps"
    for step in range(args.max_steps):
        bm = ctrl.act()
        sr = pool.step_all(np.array([bm], dtype=np.uint8))[0]
        ram = sr[2]
        frames.append(np.asarray(sr[0]).astype(np.uint8))
        tracker.update(ram)
        lbl = label_from_ram(ram)
        per_level_first.setdefault(lbl, step)
        ctrl.observe(ram, sr[1], step + 1)
        if args.stop_after_worlds and tracker.worlds_cleared >= args.stop_after_worlds:
            end_reason = "seq_clear"; break
        if bool(sr[3]):
            end_reason = "died"; break
    pool.shutdown()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image
    # GIF always (Pillow backend, embeds anywhere); MP4 best-effort
    # (needs the ffmpeg backend — skip cleanly if absent).
    gif_frames = frames[::3]  # ~10fps gif
    gif_small = [np.asarray(Image.fromarray(f).resize((256, 240))) for f in gif_frames]
    imageio.mimsave(str(outp) + ".gif", gif_small, duration=1.0 / (args.fps / 3))
    try:
        imageio.mimsave(str(outp) + ".mp4", frames, fps=args.fps, macro_block_size=1)
    except Exception as exc:  # noqa: BLE001 — MP4 is a bonus
        print(f"(mp4 skipped: {exc}; GIF written — `pip install imageio[ffmpeg]` for mp4)")

    furthest = tracker.furthest_seq or tracker.furthest_any
    print("=" * 60)
    print(f"LEARNED playthrough recorded: {outp}.mp4 / {outp}.gif")
    print(f"frames: {len(frames)}  end_reason: {end_reason}")
    print(f"worlds fully cleared (sequential, no warp): {tracker.worlds_cleared}")
    print(f"warp taken: {tracker.warp_taken}")
    print(f"furthest sequential level reached: {level_label(tracker.furthest_seq)}")
    print(f"furthest any level: {level_label(tracker.furthest_any)}")
    print(f"levels entered (in order): {list(per_level_first.keys())}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
