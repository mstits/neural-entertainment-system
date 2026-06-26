"""Record a BC demo tape, optionally warm-started at a save-state.

Opens the keyboard-driven PlayWindow so a human can record a focused
demonstration. With --state, the emulator boots straight into that
save-state (e.g. drop into 1-4 and record only that level instead of a
full 1-1->1-4 run). The recorded `.state.bin` tape pairs with the SAME
--state when fed to scripts/bc_pretrain.py --start-state.

NOTE: this is an OPTIONAL human-in-the-loop path. The project's primary
goal is fully autonomous learning (no demos); keep this for cases where
a level resists every autonomous lever.

Usage:
  python scripts/record_demo.py --rom "roms/Super Mario Bros. (World).nes" \
      --state checkpoints/super_mario_bros/smb_curriculum/stage_1_4.state
  # play (arrows=D-pad, Z=A/jump, X=B/run), click "Save BC Tape", note the path
  # then:
  python scripts/bc_pretrain.py --profile configs/smb_1_4.yaml \
      --rom "roms/Super Mario Bros. (World).nes" --demo <tape>.state.bin \
      --start-state checkpoints/super_mario_bros/smb_curriculum/stage_1_4.state \
      --out checkpoints/mario_1_4_bc/vanilla_ppo_iter_00000.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--state", default=None,
                    help="save-state to warm-start the recording from "
                         "(omit to record from power-on)")
    args = ap.parse_args()

    if args.state and not Path(args.state).exists():
        print(f"start-state not found: {args.state}", file=sys.stderr)
        return 2

    from PyQt6.QtWidgets import QApplication
    from src.gui.play_window import PlayWindow

    app = QApplication(sys.argv)
    win = PlayWindow(rom_path=args.rom, start_state_path=args.state)
    win.show()
    where = f"warm-started at {Path(args.state).name}" if args.state else "from power-on"
    print(f"[record_demo] PlayWindow open ({where}). "
          f"Controls: arrows=D-pad, Z=A(jump), X=B(run). "
          f"Click 'Save BC Tape' when done.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
