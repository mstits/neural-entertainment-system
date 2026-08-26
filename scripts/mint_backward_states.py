"""Mint a backward start-state curriculum from a solver tape.

Deterministically replays one banked solution and snapshots the emulator
every N frames, producing the state ladder a reverse curriculum
(Salimans & Chen, arXiv:1812.03381) restarts from. The tape supplies
START STATES ONLY — no action labels leave this script, so nothing here
feeds imitation (naive BC on these same tapes was eliminated in Dossier
v3: clone accuracy 1.0 -> 0.00 honest success).

Replay convention is the repo's one convention, and lives in exactly one
place: `src.training.tape_replay` (load the root blob, take ONE noop step
to materialize RAM, then step the tape's actions in order). The state
minted for step i is the emulator BEFORE action i runs — the frame
`TapePlayer.play()` pauses on when it yields step i — so restoring it and
running the tape's tail reproduces the clear exactly.

The output dir is self-describing: `index.json` carries every entry's
(step, frame, gx, area) plus the tape's provenance, and
`src/training/backward_curriculum.load_index` is the only reader.

Usage:
  python scripts/mint_backward_states.py --level 1-1 \
      --run runs/live_show/smb_4_4_micro \
      --profile configs/smb_4_4_micro.yaml \
      --out checkpoints/backward_states/1-1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.go_explore_solve import hw_provenance  # noqa: E402
from src.training.backward_curriculum import (  # noqa: E402
    DEFAULT_EVERY_FRAMES, DEFAULT_GX_TOLERANCE, StateEntry, gx_report,
    snapshot_steps, state_filename, stride_steps, write_index,
)
from src.training.tape_replay import (  # noqa: E402
    TapePlayer, load_root, machine_from_profile, root_state_for,
    solution_paths,
)

R_WORLD = 0x075F
R_LEVEL = 0x075C
R_AREA = 0x0760
R_X_PAGE = 0x006D
R_X_LOW = 0x0086

# `solution_paths` / `root_state_for` moved to src.training.tape_replay
# (the tape-provenance half of the one replay convention) and are
# re-exported here so existing callers keep importing them from the
# script they were written against.
__all__ = ["mint", "root_state_for", "solution_paths"]


def mint(
    *,
    rom: str,
    root_bytes: bytes,
    actions: np.ndarray,
    bitmasks: list,
    out_dir: Path,
    frame_skip: int,
    every_frames: int,
    hw_flags: tuple = (),
) -> tuple[list[StateEntry], dict]:
    """Replay the tape, writing a savestate at each planned step.

    Returns (entries, tail-facts). Tail facts describe where the tape
    ENDS — the caller gates on them, because a tape that no longer
    reaches its clear on this machine has silently changed lineage and
    every state minted from it is worthless as a curriculum.
    """
    steps = set(snapshot_steps(len(actions), frames_per_step=frame_skip,
                               every_frames=every_frames))
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[StateEntry] = []
    ram = None
    with TapePlayer(rom=rom, bitmasks=bitmasks, frame_skip=frame_skip,
                    hw_flags=hw_flags) as player:
        # `play` yields the RAM BEFORE action `i` runs and holds the
        # emulator there until the loop body returns, so the blob saved
        # here is exactly the machine that action i is about to act on.
        for i, ram in player.play(root_bytes, actions):
            if i not in steps:
                continue
            name = state_filename(i)
            (out_dir / name).write_bytes(player.save_state(i))
            entries.append(StateEntry(
                step=i,
                frame=i * frame_skip,
                gx=(int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW]),
                area=int(ram[R_AREA]),
                file=name,
            ))
    tail = {
        "end_wd": [int(ram[R_WORLD]), int(ram[R_LEVEL])],
        "end_area": int(ram[R_AREA]),
        "end_gx": (int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW]),
    }
    return entries, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", required=True, help='e.g. "1-1"')
    ap.add_argument("--run", default="runs/live_show/smb_4_4_micro",
                    help="Solver run dir holding lvl_<level>/solutions/.")
    ap.add_argument("--profile", default="configs/smb_4_4_micro.yaml",
                    help="Profile the tape was SOLVED under — its "
                         "action_space indexes the tape and its frame_skip "
                         "sets the replay cadence.")
    ap.add_argument("--out", default=None,
                    help="Output dir (default checkpoints/backward_states/"
                         "<level>).")
    ap.add_argument("--solution", type=int, default=0,
                    help="Which banked solution to replay (default 0).")
    ap.add_argument("--actions", default=None, help="Override tape path.")
    ap.add_argument("--start-state", default=None,
                    help="Override the tape's root blob. Must be the "
                         "sequence's OWN root.")
    ap.add_argument("--every-frames", type=int, default=DEFAULT_EVERY_FRAMES,
                    help="NES frames between minted states (default 4 = one "
                         "state per action step at frame_skip 4).")
    ap.add_argument("--hw-flags", default=None,
                    help="Comma-separated nes_core hw flags the tape was "
                         'produced under. Omit to take the profile\'s '
                         '`solve.hw_flags`; pass "none" to force empty.')
    ap.add_argument("--gx-tolerance", type=int, default=DEFAULT_GX_TOLERANCE,
                    help="Same-area gx dip (px) tolerated as sub-tile "
                         "jitter by the monotonicity gate (default one tile).")
    ap.add_argument("--allow-nonmonotone", action="store_true",
                    help="Mint even when the per-area gx gate fails.")
    ap.add_argument("--rom", default=None,
                    help="ROM path. Required when the profile carries no "
                         "rom_path key (solve profiles pass the ROM on the "
                         "solver CLI instead).")
    args = ap.parse_args()

    run_dir = Path(args.run)
    act_path, sidecar = solution_paths(run_dir, args.level, args.solution)
    if args.actions:
        act_path = Path(args.actions)
    profile = yaml.safe_load(Path(args.profile).read_text())
    rom, frame_skip, bitmasks, hw_flags = machine_from_profile(
        profile, rom=args.rom, hw_flags=args.hw_flags,
        profile_name=args.profile, who="mint")

    root_path = (Path(args.start_state) if args.start_state
                 else root_state_for(sidecar, run_dir))
    root_bytes = load_root(root_path, hw_flags)
    actions = np.load(act_path).astype(np.int64)
    if actions.ndim != 1 or (actions.size and int(actions.max()) >= len(bitmasks)):
        raise SystemExit(
            f"[mint] {act_path}: actions {actions.shape} max "
            f"{actions.max() if actions.size else '-'} vs "
            f"{len(bitmasks)}-way action space in {args.profile}")

    out_dir = Path(args.out) if args.out else (
        REPO / "checkpoints" / "backward_states" / args.level)
    t0 = time.time()
    entries, tail = mint(
        rom=rom, root_bytes=root_bytes, actions=actions,
        bitmasks=bitmasks, out_dir=out_dir, frame_skip=frame_skip,
        every_frames=args.every_frames, hw_flags=hw_flags,
    )
    dt = time.time() - t0

    rec = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    want_wd = list(rec.get("clear_wd", [])) or None
    if want_wd is None:
        raise SystemExit(
            f"[mint] no clear_wd recorded to verify against ({sidecar} "
            f"missing or has no clear_wd) — refusing to mint an unverified "
            f"ladder. Point --run at the solve that banked this tape, or "
            f"pass --actions/--start-state from one whose sidecar records "
            f"clear_wd.")
    cleared = tail["end_wd"] == want_wd
    report = gx_report(entries, tolerance=int(args.gx_tolerance))

    meta = {
        "level": args.level,
        "provenance": "solver_tape",
        "root_state": str(root_path),
        "actions": str(act_path),
        "profile": str(args.profile),
        "rom": rom,
        "frame_skip": frame_skip,
        "every_frames": int(args.every_frames),
        "stride_steps": stride_steps(frame_skip, int(args.every_frames)),
        "n_actions": int(actions.size),
        "start_wd": list(rec.get("start_wd", [])),
        "clear_wd_expected": want_wd,
        "hw": hw_provenance(hw_flags, frame_skip),
        "tail": tail,
        "reached_clear": bool(cleared),
        "gx": report,
        "minted_at": time.time(),
    }
    if not cleared:
        raise SystemExit(
            f"[mint] tape did NOT reach its banked clear: end_wd "
            f"{tail['end_wd']} != {want_wd}. The root blob or the machine "
            f"lineage has drifted — mint aborted, {out_dir} left partial.")
    if not report["monotone"] and not args.allow_nonmonotone:
        raise SystemExit(
            f"[mint] per-area gx gate FAILED: "
            f"{report['decreases_over_tolerance']} same-area drops beyond "
            f"{report['tolerance']} px (max drop {report['max_drop']}). Pass "
            f"--allow-nonmonotone to mint anyway.")

    write_index(out_dir, entries, meta)
    print(f"[mint] {args.level}: {len(entries)} states "
          f"(stride {meta['stride_steps']} step(s) = {args.every_frames} "
          f"frames) from {actions.size} actions in {dt:.1f}s -> {out_dir}\n"
          f"[mint]   gx {report['gx_first']}..{report['gx_last']} "
          f"areas={report['areas']} segments={report['segments']} "
          f"monotone={report['monotone']} (strict="
          f"{report['strict_monotone']}, max_drop {report['max_drop']}px) "
          f"end_wd={tail['end_wd']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
