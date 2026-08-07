"""Rebuild the CV power-on input tape from the banked cv_chain_hw2 lineage.

The original /tmp/cv_tape.bin (20,061 frames; Mesen receipts: stage
changes at frames 2872/9322, all-flags trace fork ~4118) was transient
and deleted. This reassembles it from the surviving artifacts using the
EXACT original conventions, recovered from the 2026-07-31 session
record (the hw2-chain driver + the make_cv_tape prefix generator):

  - PREFIX (900 frames): periodic START taps — 0x08 while (f % 90) < 8,
    else no-op — replayed on a frame_skip=1 Pool with the FOUR pool
    hw flags of the era (reset_alignment, mmio_read, dmc_stall,
    nmi_poll; mmio_write existed but was NOT set by the driver, and
    frame_anchor is env-only). pool.reset_all() contributes the 1-frame
    cold-reset advance = tape byte 0 (0x00); root lands at power-on
    frame 901 (the blob's own PPU frame counter).
  - BLOCKS 0-2: from the root, per block: ONE frame_skip=4 no-op pool
    step (4 tape bytes), then each solver action held 4 frames,
    truncating at the FIRST level_key ($0028) change — the transit must
    land at action 492/1604/1252 per the verdict log or the lineage has
    drifted and we ABORT — then EIGHT fs=4 no-op settle steps (32
    bytes). The settled state must match the banked
    entrance_after_N.state anchor.

Blocks-only total: 14,401 frames — covering the 240 s (14,400-frame)
lockstep gate horizon. The optional block-3 hall segment (~5,660 more
frames, reproducing the original 20,061) is appended with --hall.

Output: /tmp/cv_tape.bin + runs/cv_chain_hw2/cv_tape_rebuilt.bin and a
JSON receipt (sha256, segment table) next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms" / "Castlevania (USA).nes")
CHAIN = REPO / "runs" / "cv_chain_hw2"
LEVEL_KEY = 0x0028
# Verdict-log fingerprints ("transit at act 492/1604/1252") are 1-based
# action counts; stored 0-based for the enumerate() comparison below.
EXPECTED_TRANSITS = [491, 1603, 1251]


def set_hw(pool) -> None:
    """The hw2-chain driver's exact flag set (2026-07-31): four pool
    flags; mmio_write deliberately absent, frame_anchor env-only."""
    pool.set_hw_reset_alignment(True)
    pool.set_hw_mmio_read_timing(True)
    pool.set_hw_dmc_stall_timing(True)
    pool.set_hw_nmi_poll_timing(True)


def prefix_byte(f: int) -> int:
    return 0x08 if (f % 90) < 8 else 0x00


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hall", action="store_true",
                    help="append the block-3 hall trace segment "
                         "(reproduces the original 20,061-frame total)")
    args = ap.parse_args()
    t0 = time.time()
    profile = yaml.safe_load((REPO / "configs" / "castlevania.yaml").read_text())
    bm = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))

    # ---- prefix: fs=1 pool, byte-for-frame ----
    pool = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=1)
    pool.set_headless(True)
    set_hw(pool)
    pool.reset_all()
    tape = [0x00]                       # byte 0: reset_all's 1-frame advance
    x = np.zeros(1, dtype=np.uint8)
    for f in range(900):
        x[0] = prefix_byte(f)
        ram = pool.step_all(x)[0][2]
        tape.append(prefix_byte(f))
    print(f"[prefix] done: stage {int(ram[0x28])} lives {int(ram[0x2A])} "
          f"({len(tape)} tape frames)", flush=True)
    rebuilt_root = bytes(pool.save_worker_state(0))
    root_blob = (CHAIN / "poweron_root.state").read_bytes()
    root_exact = rebuilt_root == root_blob
    print(f"[prefix] root blob byte-match: {root_exact}", flush=True)
    pool.shutdown()

    # Certify the root behaviorally regardless of blob-layout equality:
    # continue 120 no-op frames from both states in twin fs=1 pools and
    # require identical RAM every frame (CV's free-running counters
    # diverge immediately on any cycle/RNG mismatch).
    pa = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=1)
    pb = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=1)
    for p in (pa, pb):
        p.set_headless(True)
        set_hw(p)
        p.reset_all()
    pa.load_worker_state(0, rebuilt_root)
    pb.load_worker_state(0, root_blob)
    for i in range(120):
        ra = pa.step_all(np.zeros(1, dtype=np.uint8))[0][2]
        rb = pb.step_all(np.zeros(1, dtype=np.uint8))[0][2]
        if bytes(ra) != bytes(rb):
            raise SystemExit(f"[prefix] rebuilt root DIVERGES from banked "
                             f"blob at continuation frame {i} — ABORT")
    pa.shutdown()
    pb.shutdown()
    print("[prefix] root certified: 120-frame continuation identical to "
          "the banked blob", flush=True)
    segments = [{"segment": "prefix", "frames": len(tape),
                 "root_blob_byte_match": bool(root_exact)}]

    # ---- blocks 0-2: fs=4 pool, 4 tape bytes per step ----
    p2 = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=fs)
    p2.set_headless(True)
    p2.reset_all()
    set_hw(p2)
    p2.load_worker_state(0, root_blob)
    x = np.zeros(1, dtype=np.uint8)

    def step4(mask: int):
        x[0] = mask
        r = p2.step_all(x)[0][2]
        tape.extend([mask] * fs)
        return r

    for n in range(3):
        acts = np.load(CHAIN / f"lvl_{n:02d}" / "solutions" /
                       "sol_000.actions.npy").astype(int)
        ram = step4(0x00)               # rooting no-op step
        start_key = int(ram[LEVEL_KEY])
        transit_at = None
        for i, a in enumerate(acts):
            ram = step4(int(bm[int(a)]))
            if int(ram[LEVEL_KEY]) != start_key:
                transit_at = i
                break
        if transit_at != EXPECTED_TRANSITS[n]:
            raise SystemExit(
                f"[block {n}] transit at action {transit_at}, expected "
                f"{EXPECTED_TRANSITS[n]} — lineage drift, ABORT")
        for _ in range(8):              # settle
            step4(0x00)
        anchor = (CHAIN / "entrances" / f"entrance_after_{n}.state").read_bytes()
        mine = bytes(p2.save_worker_state(0))
        if mine != anchor:
            # Behavioral fallback, as for the root above.
            qa = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=1)
            qb = nes_core.Pool(rom_path=ROM, num_workers=1, frame_skip=1)
            for p in (qa, qb):
                p.set_headless(True)
                set_hw(p)
                p.reset_all()
            qa.load_worker_state(0, mine)
            qb.load_worker_state(0, anchor)
            for i in range(120):
                ra = qa.step_all(np.zeros(1, dtype=np.uint8))[0][2]
                rb = qb.step_all(np.zeros(1, dtype=np.uint8))[0][2]
                if bytes(ra) != bytes(rb):
                    raise SystemExit(f"[block {n}] settled state diverges "
                                     f"from entrance anchor at continuation "
                                     f"frame {i} — ABORT")
            qa.shutdown()
            qb.shutdown()
        print(f"[block {n}] verified: transit at action {transit_at}, "
              f"anchor {'byte-exact' if mine == anchor else 'behaviorally certified'}, "
              f"tape now {len(tape)} frames", flush=True)
        segments.append({"segment": f"block_{n}", "transit_action": transit_at,
                         "frames": len(tape)})

    if args.hall:
        import pickle
        traces = pickle.loads(
            (CHAIN / "lvl_03_trace" / "traces.pkl").read_bytes())
        # Deepest banked hall trace: max progress key; values are
        # (root_id, action_bytes, ...) per the archive convention.
        key = max(traces, key=lambda k: k[-1])
        acts = traces[key][1]
        step4(0x00)
        for a in acts:
            step4(int(bm[int(a)]))
        segments.append({"segment": "hall", "actions": len(acts),
                         "frames": len(tape)})
        print(f"[hall] appended {len(acts)} actions, tape now "
              f"{len(tape)} frames", flush=True)
    p2.shutdown()

    blob = bytes(tape)
    Path("/tmp/cv_tape.bin").write_bytes(blob)
    durable = CHAIN / "cv_tape_rebuilt.bin"
    durable.write_bytes(blob)
    receipt = {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "frames": len(blob),
        "segments": segments,
        "built_s": round(time.time() - t0, 1),
    }
    (CHAIN / "cv_tape_rebuilt.receipt.json").write_text(
        json.dumps(receipt, indent=2))
    print(f"[done] {len(blob)} frames, sha256={receipt['sha256'][:16]}..., "
          f"{receipt['built_s']}s -> /tmp/cv_tape.bin + {durable}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
