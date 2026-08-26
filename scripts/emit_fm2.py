"""Emit an FCEUX-verifiable `.fm2` movie from a recorded action sequence.

The inverse of `convert_fm2.fceux_movie_to_bytes`: our evals step the
pool at frame_skip N with HELD buttons, so each recorded pool-step
bitmask expands to exactly N identical per-frame input rows — the
reconstruction is exact, not approximate. The emitted movie starts from
power-on (`|1|........|||` reset command on frame 0), so an eval whose
episode begins at the cold boot replays end-to-end in FCEUX with no
savestate anchor: TAS-grade receipts for a claimed run.

Usage:
  python scripts/emit_fm2.py actions.npy out.fm2 --rom "roms/Super Mario Bros. (World).nes" --profile configs/smb.yaml [--frame-skip 4]

`actions.npy`: int64 action-space INDICES, one per pool step — the
repo-wide solver "solution tape" convention (see
`src/training/tape_replay.py`), not raw controller bitmasks. `--profile`
names the YAML config whose `action_space` resolves each index to its
RLDUTSBA bitmask via `action_space_to_bitmasks`, the same mapping every
other tape consumer (`assemble_full_run.py`, `tape_replay.py`) uses.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import uuid
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from convert_fm2 import _FM2_BIT_AT_POS, fceux_movie_to_bytes  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402


def mask_to_pad(mask: int) -> str:
    """Render one controller bitmask as the fm2 8-char pad field."""
    chars = "RLDUTSBA"
    return "".join(
        chars[i] if mask & bit else "."
        for i, bit in enumerate(_FM2_BIT_AT_POS)
    )


def emit_fm2(actions: np.ndarray, rom_path: Path, frame_skip: int = 4,
             rom_name: str | None = None) -> str:
    """Build the .fm2 text for a pool-step action sequence."""
    md5 = base64.b64encode(hashlib.md5(rom_path.read_bytes()).digest())
    lines = [
        "version 3",
        "emuVersion 20500",
        "rerecordCount 0",
        "palFlag 0",
        f"romFilename {rom_name or rom_path.name}",
        f"romChecksum base64:{md5.decode()}",
        f"guid {uuid.uuid4()}",
        "fourscore 0",
        "microphone 0",
        "port0 1",
        "port1 0",
        "port2 0",
        "FDS 0",
        "NewPPU 0",
    ]
    # Frame 0 carries the soft-reset command (bit 1) with no input — the
    # movie is anchored at power-on, not a savestate.
    lines.append("|1|........|||")
    for mask in actions:
        pad = mask_to_pad(int(mask))
        lines.extend(f"|0|{pad}|||" for _ in range(frame_skip))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("actions", help="npy of int64 action-space indices "
                                     "(the solver 'solution tape' convention)")
    ap.add_argument("out", help="output .fm2 path")
    ap.add_argument("--rom", required=True)
    ap.add_argument("--profile", required=True,
                     help="YAML config whose action_space resolves each "
                          "index to its controller bitmask")
    ap.add_argument("--frame-skip", type=int, default=4)
    args = ap.parse_args()

    indices = np.load(args.actions).astype(np.int64)
    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    actions = np.array([bitmasks[int(a)] for a in indices], dtype=np.uint8)
    text = emit_fm2(actions, Path(args.rom), frame_skip=args.frame_skip)
    Path(args.out).write_text(text)

    # Round-trip self-check: parse the emitted movie back with the
    # project's own fm2 reader and verify the per-frame masks match the
    # frame-skip-expanded input exactly (frame 0 is the reset row).
    parsed = fceux_movie_to_bytes(Path(args.out))
    expected = np.repeat(actions, args.frame_skip)
    got = np.frombuffer(parsed, dtype=np.uint8)[1:]  # drop reset row
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert np.array_equal(got, expected), "round-trip mismatch"
    print(f"wrote {args.out}: {len(actions)} pool steps -> "
          f"{len(actions) * args.frame_skip + 1} frames, round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
