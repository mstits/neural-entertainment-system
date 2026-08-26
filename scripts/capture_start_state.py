"""Capture a live-gameplay start state for a game.

Without a start state the emulator cold-boots to the TITLE SCREEN, where
the attract-mode demo auto-plays and ignores controller input — training
there gives zero learning signal (see the start-state-demo bug). This
tool boots a ROM, mashes through the title/menus, and snapshots the
emulator state the moment the player becomes CONTROLLABLE, writing
`roms/<rom_stem>_start.state.bin` (the sidecar the launcher/profile use).

"Controllable" is detected game-agnostically: from a candidate state,
hold RIGHT for K frames vs LEFT for K frames and compare RAM. If the two
diverge by more than a few bytes, directional input is moving the player
(timers advance identically in both branches, so they cancel) — i.e.
we're in gameplay, not a demo/menu. Verberbose `--probe` tuning available.

Usage:
    python scripts/capture_start_state.py --game contra
    python scripts/capture_start_state.py --game megaman --max-frames 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nes_core

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_game import DEFAULT_ROMS  # noqa: E402

# nes-py bit layout (matches frame_utils / pool).
NOOP, A, B, SELECT, START = 0x00, 0x01, 0x02, 0x04, 0x08
UP, DOWN, LEFT, RIGHT = 0x10, 0x20, 0x40, 0x80


def _menu_input(frame: int) -> int:
    """Input schedule that mashes through most title/menu flows.

    Pulses START (rising edges register on title + 'press start' menus),
    then layers in A (confirm) and DOWN (move a default cursor) so stage-
    select / file-select screens advance too. NOOP gaps between pulses so
    toggling menus settle on a stable choice.
    """
    if frame < 24:
        return NOOP  # let the title screen settle / logo finish
    phase = frame % 16
    pressing = phase < 6  # 6-on / 10-off pulse
    if not pressing:
        return NOOP
    # Alternate which button the pulse sends so we clear multi-screen
    # flows (title -> press start -> stage/file select -> go).
    cycle = (frame // 16) % 4
    return {0: START, 1: START, 2: A, 3: START}[cycle]


def _ram(env: nes_core.NESEnvironment) -> bytes:
    return bytes(env.get_ram_range(0, 2048))


def _diff_bytes(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i])


def _ram_after(env: nes_core.NESEnvironment, snap: bytes, button: int, n: int) -> bytes:
    env.load_state(snap)
    for _ in range(n):
        env.step(button)
    return _ram(env)


def is_controllable(env: nes_core.NESEnvironment, thresh: int) -> bool:
    """Distinguish live GAMEPLAY from a menu / cutscene.

    A menu IS "controllable" (LEFT/RIGHT move a cursor), so a one-shot
    RIGHT-vs-LEFT diff false-positives on stage-select / name-entry
    screens. A menu with its OWN animation (e.g. an animated world-map
    cursor screen) also false-positives on growth measured against a
    frozen base snapshot: the animation mutates RAM whether or not a
    direction is held, so a RIGHT-only diff against base keeps growing
    for reasons that have nothing to do with the held direction. Gameplay
    is told apart by two properties a menu (animated or not) lacks:

      * GROWTH — measured on the RIGHT-vs-LEFT split itself, not against
        base, so shared animation (identical on both branches) cancels
        out of the diff. A cursor reaches its rest position within a
        handful of frames and the split stops widening; real movement
        keeps separating the two branches further apart the longer
        input is held.
      * DIRECTIONALITY — RIGHT and LEFT lead to different RAM at all. A
        scripted cutscene / autoscroll that ignores input fails this.
    """
    snap = bytes(env.save_state())
    base = _ram(env)
    r20 = _ram_after(env, snap, RIGHT, 20)
    r90 = _ram_after(env, snap, RIGHT, 90)
    l20 = _ram_after(env, snap, LEFT, 20)
    l90 = _ram_after(env, snap, LEFT, 90)
    env.load_state(snap)  # restore caller's position
    d_late = _diff_bytes(r90, base)
    d_dir_early = _diff_bytes(r20, l20)
    d_dir_late = _diff_bytes(r90, l90)
    return (d_late > thresh
            and d_dir_late > thresh                     # RIGHT != LEFT => input-driven, not a cutscene
            and d_dir_late > d_dir_early + thresh // 2)  # split still widening => real movement, not a settled cursor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, help=f"one of {sorted(set(DEFAULT_ROMS))}")
    p.add_argument("--rom", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--probe-thresh", type=int, default=25)
    p.add_argument("--check-every", type=int, default=15)
    p.add_argument("--settle", type=int, default=40,
                   help="frames to advance (NOOP) after detecting gameplay, "
                        "so the captured frame is fully rendered + stable")
    args = p.parse_args()

    rom = args.rom or DEFAULT_ROMS.get(args.game.lower().strip())
    if not rom or not Path(rom).exists():
        print(f"ROM not found for game={args.game!r}: {rom}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(rom).with_name(Path(rom).stem + "_start.state.bin")

    env = nes_core.NESEnvironment(rom_path=rom, frame_skip=1)
    env.reset()

    for frame in range(args.max_frames):
        env.step(_menu_input(frame))
        if frame >= 60 and frame % args.check_every == 0:
            if is_controllable(env, args.probe_thresh):
                # Settle into stable, fully-rendered gameplay before
                # snapshotting (the detection frame may be mid-fade).
                for _ in range(args.settle):
                    env.step(NOOP)
                if not is_controllable(env, args.probe_thresh):
                    continue  # settled back into a menu/transition; keep going
                state = bytes(env.save_state())
                out.write_bytes(state)
                print(f"[capture] {args.game}: gameplay confirmed at frame "
                      f"{frame} (+{args.settle} settle); wrote {len(state)} "
                      f"bytes -> {out}")
                return 0

    print(f"[capture] {args.game}: never became controllable within "
          f"{args.max_frames} frames. Game may need manual menu navigation "
          f"(e.g. Zelda save-file registration).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
