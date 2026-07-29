"""Assemble and verify the complete power-on -> princess run of SMB.

Stitches the clean-chain solutions (runs/ge_chain_final_a: 1-1..4-3,
runs/ge_chain_final_b: 4-4..8-3, + the 8-4 finale) into ONE controller tape
of per-step button masks, starting from an actual cold boot (reset, title,
START) with zero state loads anywhere. Replays the tape end-to-end on a
fresh pool, verifying every level boundary (world/level bytes) and the
finale (operating mode 2, the THANK YOU MARIO screen), and writes:

  runs/full_run/full_tape.npy      -- uint8 button mask per frame-skip step
  runs/full_run/receipts.json      -- per-level boundary step + wd + verify
  runs/full_run/final_frame.png    -- the ending screen from THIS replay

Transition contract (mirrors go_explore_chain.extract_next_entrance): each
level's solution actions are consumed only UP TO the first wd-change step,
then 8 settle no-ops, then one convention no-op before the next level's
actions (go_explore_solve steps one no-op after rooting). The 8-4 finale
runs its full trace (the ending never changes wd) plus a tail of no-ops so
the princess scene plays out on tape.

Optional: --video renders the verified tape to an mp4 via ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
START_MASK = 0x08

# Boot prefix (mirrors the power_on_1-1.state creation exactly): one
# convention step, 60 title-wait no-ops, 3 START presses, 36 settle no-ops.
BOOT = [0] * 1 + [0] * 60 + [START_MASK] * 3 + [0] * 37  # byte-exact vs power_on_1-1.state


def level_dirs():
    a = sorted(Path("runs/ge_chain_final_a").glob("lvl_*"))
    b = sorted(Path("runs/ge_chain_final_b").glob("lvl_*"))
    b = [d for d in b if "8-4" not in d.name or "retry" in d.name]
    dirs = a + [d for d in b if "retry" not in d.name] + \
        [Path("runs/ge_chain_final_b/lvl_16_8-4_retry")]
    out = []
    for d in dirs:
        sols = sorted(d.glob("solutions/sol_*.actions.npy"))
        if sols:
            out.append((d.name, sols[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="configs/smb_4_4_micro.yaml")
    ap.add_argument("--out", default="runs/full_run")
    ap.add_argument("--video", default=None, help="mp4 output path")
    ap.add_argument("--tail-noops", type=int, default=400)
    args = ap.parse_args()
    prof = yaml.safe_load(Path(args.profile).read_text())
    bm = action_space_to_bitmasks(prof["action_space"])
    fs = int(prof.get("frame_skip", 4))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    levels = level_dirs()
    print(f"[assemble] {len(levels)} level solutions on the chain")
    assert len(levels) == 32, f"expected 32 levels, found {len(levels)}"

    pool = Pool(rom_path=ROM, num_workers=1, frame_skip=fs)
    pool.set_headless(args.video is None)
    pool.reset_all()                      # COLD BOOT — no state loads, ever

    tape: list[int] = []
    receipts = []
    video_proc = None
    if args.video:
        video_proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "256x240", "-r", str(60 // fs), "-i", "-",
             "-vf", "scale=768:720:flags=neighbor",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             args.video],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=open("/tmp/ffmpeg_err.log", "w"))

    def step(mask: int):
        r = pool.step_all(np.array([mask], dtype=np.uint8))
        tape.append(mask)
        if video_proc is not None and r[0][0] is not None:
            video_proc.stdin.write(np.asarray(r[0][0]).astype(np.uint8).tobytes())
        return bytes(r[0][2])

    for m in BOOT:
        ram = step(m)
    wd = (int(ram[0x75F]), int(ram[0x75C]))
    print(f"[assemble] after boot: wd={wd}")
    assert wd == (0, 0), f"boot did not reach 1-1: {wd}"
    ram = step(0)                         # solver rooting convention no-op

    for name, sol in levels:
        acts = np.load(sol).astype(int)
        start_wd = (int(ram[0x75F]), int(ram[0x75C]))
        is_finale = (start_wd == (7, 3))
        transitioned = False
        for a in acts:
            ram = step(bm[int(a)])
            wd = (int(ram[0x75F]), int(ram[0x75C]))
            if not is_finale and wd != start_wd:
                transitioned = True
                break
        if is_finale:
            for _ in range(args.tail_noops):
                ram = step(0)
            ok = int(ram[0x770]) == 2
            receipts.append({"level": name, "end_step": len(tape),
                             "verify": "opermode==2", "ok": bool(ok)})
            print(f"[assemble] {name}: FINALE opermode={int(ram[0x770])} "
                  f"{'VERIFIED' if ok else '*** FAILED ***'}")
            assert ok, "finale did not reach the victory mode"
            break
        assert transitioned, f"{name}: solution did not transition on-chain"
        for _ in range(8):                # settle (mirrors entrance extraction)
            ram = step(0)
        receipts.append({"level": name, "end_step": len(tape),
                         "wd_after": [int(ram[0x75F]), int(ram[0x75C])],
                         "ok": True})
        print(f"[assemble] {name}: cleared -> wd={(int(ram[0x75F]), int(ram[0x75C]))} "
              f"@step {len(tape)}")
        ram = step(0)                     # convention no-op (solver rooting)

    if video_proc is not None:
        video_proc.stdin.close(); video_proc.wait()
    # final frame from THIS replay
    try:
        from PIL import Image
        r = pool.step_all(np.zeros(1, dtype=np.uint8))
        if r[0][0] is not None:
            Image.fromarray(np.asarray(r[0][0]).astype(np.uint8)).save(
                outdir / "final_frame.png")
    except Exception:
        pass
    pool.shutdown()

    tape_arr = np.array(tape, dtype=np.uint8)
    np.save(outdir / "full_tape.npy", tape_arr)
    sha = hashlib.sha256(tape_arr.tobytes()).hexdigest()
    (outdir / "receipts.json").write_text(json.dumps({
        "rom": "Super Mario Bros. (World).nes",
        "frame_skip": fs,
        "cold_boot": True, "state_loads": 0,
        "steps": len(tape), "frames": len(tape) * fs,
        "duration_s": len(tape) * fs / 60.0,
        "tape_sha256": sha,
        "levels": receipts}, indent=2))
    print(f"[assemble] COMPLETE: {len(tape)} steps ({len(tape)*fs/60.0:.0f}s "
          f"of gameplay), tape sha256 {sha[:16]}…")


if __name__ == "__main__":
    main()
