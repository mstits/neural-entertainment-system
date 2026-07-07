"""
Interactive play window.

Open from the main menu to drive a single emulator instance with the
keyboard. Records the action bitmask held on every frame, with two
distinct save targets:

  * **Save State** (`S`) — writes a NCST snapshot of the current
    emulator state. Hand to `--start-state` so workers reset to this
    exact spot every episode.
  * **Save BC Tape** — writes the raw per-frame action mask sequence
    from cold boot to now. Hand to `--bc-demo` so the policy is
    behavior-cloned against your playthrough before RL takes over.

Both write `.state.bin` files; the trainer dispatches by which flag
points at them.

Keyboard mapping:
    Arrow keys   -> D-pad
    Z            -> A
    X            -> B
    Enter        -> Start
    Right Shift  -> Select
    P            -> pause / resume
    S            -> Save State (same as the button)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import nes_core as _nes_core

from src.emulation.frame_utils import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_SELECT,
    BUTTON_START,
    BUTTON_UP,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

# Backend selection. nes_core is the project's native Rust emulator;
# nes-py is a known-correct C++ reference (LaiNES) that we fall back
# to while the nes_core core CPU/PPU timing bug (see today's session
# memory: SMB Mario falls through floor, Zelda Link can't transition
# screens) is being investigated. Toggle via NES_PY_BACKEND=1.
import os as _os
if _os.environ.get("NES_PY_BACKEND") == "1":
    from src.emulation.nespy_adapter import NESpyEnvironment as _NESEnvironment
    print("[playwindow] backend=nes-py (NES_PY_BACKEND=1)",
          file=__import__("sys").stderr, flush=True)
else:
    _NESEnvironment = _nes_core.NESEnvironment
    print("[playwindow] backend=nes_core (set NES_PY_BACKEND=1 for nes-py)",
          file=__import__("sys").stderr, flush=True)


# Qt key -> NES button bitmask
_KEYMAP = {
    Qt.Key.Key_Up: BUTTON_UP,
    Qt.Key.Key_Down: BUTTON_DOWN,
    Qt.Key.Key_Left: BUTTON_LEFT,
    Qt.Key.Key_Right: BUTTON_RIGHT,
    Qt.Key.Key_Z: BUTTON_A,
    Qt.Key.Key_X: BUTTON_B,
    Qt.Key.Key_Return: BUTTON_START,
    Qt.Key.Key_Enter: BUTTON_START,
    Qt.Key.Key_Shift: BUTTON_SELECT,  # right shift on macOS reports as Shift
}

_KEYMAP_HELP = (
    "Arrows = D-pad   Z = A   X = B   Enter = Start   R-Shift = Select\n"
    "P = pause      S = save state      close the window to discard"
)

# NES native is 256×240. Default 3× display scale (768×720 logical) so
# the play window is comfortably usable on Retina displays without the
# user having to resize after every launch. 2× was a tiny postage stamp
# at 2x device pixel ratio.
_DISPLAY_SCALE = 3


class PlayWindow(QMainWindow):
    """Single emulator instance, keyboard-driven, records action sequence."""

    def __init__(self, rom_path: str, parent=None, start_state_path: str | None = None) -> None:
        super().__init__(parent)
        title = f"NES — Play: {Path(rom_path).name}"
        if start_state_path:
            title += f"  [warm-start: {Path(start_state_path).name}]"
        self.setWindowTitle(title)

        # frame_skip=1 so one user keypress per frame. With a
        # start_state_path the env boots at that save-state (e.g. drop
        # straight into 1-4 to record a focused demo of just that level);
        # reset() restores to it rather than cold-booting (verified).
        if start_state_path:
            self.env = _NESEnvironment(rom_path=rom_path, frame_skip=1,
                                       start_state_path=start_state_path)
        else:
            self.env = _NESEnvironment(rom_path=rom_path, frame_skip=1)
        self.env.reset()
        self._start_state_path = start_state_path
        # nes_core.NESEnvironment is a PyO3 pyclass and does NOT accept
        # arbitrary attribute assignment — trying to set `self.env.rom_path`
        # raises AttributeError, which silently killed PlayWindow
        # construction. Store the rom path on `self` directly instead;
        # the save dialog reads `self._rom_path`, not `self.env.rom_path`.
        self._rom_path = rom_path
        self._current_mask: int = 0
        self._recorded: list[int] = []
        # How many frames were committed by the most recent save. Used
        # by closeEvent to decide whether to prompt about unsaved data.
        self._saved_at_len: int = 0
        self._paused: bool = False

        self._build_ui()

        # ~60 fps tick
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // 60)

        # Scale up the first blank frame so the window sizes correctly
        self._paint(self.env.get_frame())

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        help_label = QLabel(_KEYMAP_HELP)
        help_label.setStyleSheet(
            "color: #ccc; background: #222; padding: 6px; font-family: Menlo"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setFixedSize(
            FRAME_WIDTH * _DISPLAY_SCALE,
            FRAME_HEIGHT * _DISPLAY_SCALE,
        )
        self.frame_label.setStyleSheet("background: black")
        layout.addWidget(self.frame_label)

        btn_row = QHBoxLayout()
        self.status_label = QLabel("recording: 0 frames")
        self.status_label.setStyleSheet("color: #8f8; font-family: Menlo")
        btn_row.addWidget(self.status_label, stretch=1)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.pause_btn)

        self.save_btn = QPushButton("Save State")
        self.save_btn.clicked.connect(self._save_state)
        btn_row.addWidget(self.save_btn)

        # "Save State" writes a NCST snapshot of the current emulator
        # state — used by --start-state to land workers at this exact
        # spot on every reset. "Save BC Tape" writes the raw per-frame
        # action mask sequence from cold boot — used by --bc-demo to
        # warm-start a policy via behavior cloning of the playthrough.
        # Both produce .state.bin files; the trainer dispatches by
        # which flag points at them.
        self.save_tape_btn = QPushButton("Save BC Tape")
        self.save_tape_btn.clicked.connect(self._save_bc_tape)
        btn_row.addWidget(self.save_tape_btn)
        layout.addLayout(btn_row)

        # The window must have keyboard focus to receive key events.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _tick(self) -> None:
        if self._paused:
            return
        # Contain emulator-step failures: this runs from a QTimer slot, and
        # an uncaught exception here aborts the whole app on PyQt6. Stop the
        # loop and surface the error instead of crashing the Play window.
        try:
            frame, done = self.env.step(self._current_mask)
        except Exception as exc:
            self._timer.stop()
            self._paused = True
            import logging as _logging
            _logging.getLogger(__name__).exception("play emulator step failed")
            try:
                self.status_label.setText(
                    f"emulator error: {type(exc).__name__}: {exc}"
                )
            except Exception:
                pass
            return
        self._dbg_last_mask = getattr(self, "_dbg_last_mask", -1)
        # Every-frame mask + RAM dump to /tmp/playwindow_trace.txt — used
        # to verify GUI ↔ headless parity. Replay the dumped masks
        # through a fresh NESEnvironment headlessly and compare RAM at
        # the same frame index. Any divergence pins a real GUI-vs-Pool
        # parity bug. The file is overwritten per Play-window session
        # (truncated on first frame). Trace I/O is best-effort — a
        # non-writable /tmp must never take down the window.
        if not hasattr(self, "_trace_fh"):
            try:
                self._trace_fh = open("/tmp/playwindow_trace.txt", "w")
                self._trace_fh.write(
                    "# frame mask $10 $11 $84 $70 $86 $CE $1D $657 $770 $772\n"
                )
            except OSError:
                self._trace_fh = None
        if self._trace_fh is not None:
            try:
                r = self.env.get_ram_range(0, 0x800)
                self._trace_fh.write(
                    f"{len(self._recorded):>6d} {self._current_mask:02X} "
                    f"{r[0x10]:02X} {r[0x11]:02X} {r[0x84]:02X} {r[0x70]:02X} "
                    f"{r[0x86]:02X} {r[0xCE]:02X} {r[0x1D]:02X} {r[0x657]:02X} "
                    f"{r[0x770]:02X} {r[0x772]:02X}\n"
                )
                if len(self._recorded) % 60 == 0:
                    self._trace_fh.flush()
            except Exception:
                pass
        if self._current_mask != self._dbg_last_mask and self._current_mask != 0:
            try:
                r = self.env.get_ram_range(0, 0x800)
                print(
                    f"[playwindow] mask=0x{self._current_mask:02X} "
                    f"$10={r[0x10]:02X} $11={r[0x11]:02X} "
                    f"$84={r[0x84]:02X} $70={r[0x70]:02X} "
                    f"$86={r[0x86]:02X} $CE={r[0xCE]:02X}",
                    file=__import__("sys").stderr, flush=True,
                )
            except Exception:
                pass
        self._dbg_last_mask = self._current_mask
        self._recorded.append(self._current_mask)
        self._paint(frame)
        self.status_label.setText(
            f"recording: {len(self._recorded)} frames "
            f"({len(self._recorded) / 60:.1f}s)"
        )
        if done:
            self._paused = True
            self.pause_btn.setText("Resume (game over)")

    def _paint(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        contig = np.ascontiguousarray(frame)
        qimg = QImage(contig.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            w * _DISPLAY_SCALE,
            h * _DISPLAY_SCALE,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.frame_label.setPixmap(pix)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        key = event.key()
        if key == Qt.Key.Key_P:
            self._toggle_pause()
            return
        if key == Qt.Key.Key_S:
            self._save_state()
            return
        bit = _KEYMAP.get(key)
        if bit is not None:
            self._current_mask |= bit

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        bit = _KEYMAP.get(event.key())
        if bit is not None:
            self._current_mask &= (~bit) & 0xFF

    def focusOutEvent(self, event) -> None:
        # Focus loss (Cmd+Tab, opening another window, switching macOS
        # Spaces) suppresses keyReleaseEvent for any key currently held.
        # Without clearing the mask, Link continues "walking right" even
        # though the user already let go — and the recording captures
        # those phantom presses. Reset to NOOP on every focus loss so
        # both the live emulator and the saved start state are honest.
        self._current_mask = 0
        super().focusOutEvent(event)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")

    def _save_state(self) -> None:
        if not self._recorded:
            self.status_label.setText("nothing recorded yet")
            return

        from src.gui.file_dialogs import remembered_save
        default_name = Path(self._rom_path).stem + "_start.state.bin"
        path = remembered_save(
            self,
            key="play_save_state",
            title="Save start state",
            default_filename=default_name,
            file_filter="Start state (*.state.bin);;All files (*)",
            default_dir=str(Path(self._rom_path).parent),
        )
        if not path:
            return

        # `NESEnvironment.save_state()` already returns the
        # `b"NCST\x01" + bincode(State)` blob — the prefix is baked in
        # by the PyO3 layer (`python.rs::save_state` line 315-317).
        # Historical bug: this path used to ALSO add its own prefix,
        # producing a double-NCST file that load_start_state stripped
        # one layer of → bincode saw leftover `NCST\x01` bytes in the
        # body → "start-state snapshot unreadable: io error" at every
        # training Start. Write the blob straight through.
        #
        # The nes-py adapter (`src/emulation/nespy_adapter.py`) does
        # not have a `save_state()` — fall back to the legacy
        # action-replay format (one mask byte per frame). nes_core's
        # start-state loader recognises this and replays through the
        # tape on reset.
        import os
        if hasattr(self.env, "save_state"):
            try:
                data = bytes(self.env.save_state())
            except Exception as exc:
                self.status_label.setText(f"save_state failed: {exc!s}")
                return
        else:
            data = bytes(self._recorded)
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except OSError as exc:
            self.status_label.setText(f"save failed: {exc!s}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return
        self.status_label.setText(
            f"saved: {Path(path).name} ({len(data)} frames)"
        )
        self._saved_at_len = len(self._recorded)

    def _save_bc_tape(self) -> None:
        """Write the recorded action sequence as a raw per-frame mask
        tape, the format expected by `--bc-demo` / `build_dataset` in
        `src/training/behavior_cloning.py`. One byte per frame from
        cold-boot (NES button bitmask). Distinct from `Save State`
        which writes a NCST snapshot for `--start-state`.
        """
        if not self._recorded:
            self.status_label.setText("nothing recorded yet")
            return

        from src.gui.file_dialogs import remembered_save
        default_name = Path(self._rom_path).stem + ".bc_demo.state.bin"
        path = remembered_save(
            self,
            key="play_save_bc_tape",
            title="Save BC demo tape",
            default_filename=default_name,
            file_filter="BC demo tape (*.state.bin);;All files (*)",
            default_dir=str(Path(self._rom_path).parent),
        )
        if not path:
            return

        import os
        data = bytes(self._recorded)
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except OSError as exc:
            self.status_label.setText(f"save failed: {exc!s}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return
        self.status_label.setText(
            f"saved BC tape: {Path(path).name} ({len(data)} frames)"
        )
        self._saved_at_len = len(self._recorded)

    def closeEvent(self, event) -> None:
        # Prompt before discarding an unsaved recording. Comparing
        # against `_saved_at_len` lets a Save-then-keep-playing flow
        # close cleanly as long as no new frames were recorded since
        # the last save.
        recorded_len = len(self._recorded)
        saved_len = getattr(self, "_saved_at_len", 0)
        if recorded_len > saved_len:
            from PyQt6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("Unsaved recording")
            box.setText(
                f"You have {recorded_len - saved_len} unsaved recorded "
                "frames. Close anyway?"
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            box.setDefaultButton(QMessageBox.StandardButton.Save)
            choice = box.exec()
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.StandardButton.Save:
                self._save_state()
                # If user cancelled the save dialog, abort the close
                # too — better than silently dropping the recording.
                if len(self._recorded) > getattr(self, "_saved_at_len", 0):
                    event.ignore()
                    return
        # Teardown must never raise out of closeEvent — that would abort
        # the app on window close. Each step is independently guarded.
        try:
            self._timer.stop()
        except Exception:
            pass
        trace_fh = getattr(self, "_trace_fh", None)
        if trace_fh is not None:
            try:
                trace_fh.close()
            except Exception:
                pass
            self._trace_fh = None
        try:
            self.env.close()
        except Exception:
            pass
        super().closeEvent(event)
