"""
Genome replay viewer.

Load a checkpoint, pick the best (or N-th best) genome, run its policy in a
single emulator window so you can actually watch the agent play. This is
the tool that tells you whether your overnight training produced competent
behavior or random thrashing — metrics alone lie.

Runs the same frame-skip + preprocessing + stacked-observation pipeline
the trainer uses, then calls `net.act(obs, deterministic=True)` per step
and feeds the chosen action bitmask back into the emulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import nes_core

from src.emulation.frame_utils import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_DOWN,
    BUTTON_LEFT,
    BUTTON_NOOP,
    BUTTON_RIGHT,
    BUTTON_SELECT,
    BUTTON_START,
    BUTTON_UP,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FrameStacker,
)

NESEnvironment = nes_core.NESEnvironment
from src.models.policy_network import PolicyNetwork, get_best_device
from src.utils.video import VideoWriter


_DISPLAY_SCALE = 2
_TARGET_FPS = 60


def _action_to_bitmask(buttons: list, name_to_bit: dict) -> int:
    bitmask = 0
    for b in buttons:
        bitmask |= name_to_bit[b]
    return bitmask


_NAME_TO_BIT = {
    "NOOP": BUTTON_NOOP,
    "A": BUTTON_A,
    "B": BUTTON_B,
    "up": BUTTON_UP,
    "down": BUTTON_DOWN,
    "left": BUTTON_LEFT,
    "right": BUTTON_RIGHT,
    "start": BUTTON_START,
    "select": BUTTON_SELECT,
}


class ReplayWindow(QMainWindow):
    """Single emulator window, driven by a loaded policy genome."""

    def __init__(
        self,
        rom_path: str,
        profile_path: str,
        checkpoint_path: str,
        start_state_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"NES — Replay: {Path(checkpoint_path).name}")

        with open(profile_path, "r") as f:
            self.profile = yaml.safe_load(f)
        self.action_space = self.profile["action_space"]

        self.env = NESEnvironment(
            rom_path=rom_path,
            frame_skip=4,
            start_state_path=start_state_path,
        )
        # Track init success so an exception below releases the env we
        # just opened. Without this guard, a checkpoint with a bad
        # action-space arch (or any other init failure) leaks the
        # emulator handle until Python GC eventually runs.
        try:
            self._init_replay_state(
                profile_path, checkpoint_path, rom_path, start_state_path,
            )
        except Exception:
            try:
                self.env.close()
            except Exception:
                pass
            raise

    def _init_replay_state(
        self,
        profile_path: str,
        checkpoint_path: str,
        rom_path: str,
        start_state_path: Optional[str],
    ) -> None:
        # Load checkpoint + find the fittest genome.
        # Prefer weights_only=True (safe — pickle restricted to tensors +
        # primitive containers) so a malicious .pt cannot execute code on
        # open. Some legacy checkpoints contain non-tensor objects (e.g.
        # custom Genome dataclasses, optimizer state) that the safe loader
        # rejects; in that case fall back to the old behavior with a
        # logged warning so the user knows what they just trusted.
        try:
            data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "checkpoint %s did not load with weights_only=True (%s); "
                "falling back to unsafe loader. Only do this with files "
                "you trust.", checkpoint_path, exc,
            )
            data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        genomes = data["population"]
        # Sort by fitness descending; -inf genomes (unevaluated children) go last.
        genomes.sort(key=lambda g: g.get("fitness", float("-inf")), reverse=True)
        self._genomes = genomes
        self._selected_idx = 0
        # Remembered so _load_selected can look for a pre-exported
        # Core ML .mlpackage alongside the .pt.
        self._checkpoint_path = checkpoint_path

        self.device = get_best_device()
        self.net = PolicyNetwork(num_actions=len(self.action_space))
        self._load_selected()

        self._stacker = FrameStacker()
        first_frame = self.env.reset()
        self._stacker.reset(first_frame)
        self._deterministic = True
        self._paused = False
        self._step_count = 0
        self._recorder: Optional[VideoWriter] = None

        self._build_ui()
        self._paint(first_frame)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // _TARGET_FPS)

    # ------- UI -----------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Genome selector
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Genome:"))
        self.genome_box = QComboBox()
        for i, g in enumerate(self._genomes):
            fit = g.get("fitness", float("-inf"))
            fit_str = f"{fit:.2f}" if fit != float("-inf") else "unevaluated"
            # Defensive .get on genome_id — older checkpoints sometimes
            # omit it; index alone is enough to identify the genome.
            self.genome_box.addItem(
                f"#{i}  id={g.get('genome_id', '?')}  fitness={fit_str}"
            )
        self.genome_box.currentIndexChanged.connect(self._on_genome_changed)
        top_row.addWidget(self.genome_box, stretch=1)

        self.det_checkbox = QCheckBox("Deterministic (argmax)")
        self.det_checkbox.setChecked(True)
        self.det_checkbox.toggled.connect(self._on_det_toggled)
        top_row.addWidget(self.det_checkbox)
        layout.addLayout(top_row)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setFixedSize(
            FRAME_WIDTH * _DISPLAY_SCALE,
            FRAME_HEIGHT * _DISPLAY_SCALE,
        )
        self.frame_label.setStyleSheet("background: black")
        layout.addWidget(self.frame_label)

        # Status + controls
        btn_row = QHBoxLayout()
        self.status_label = QLabel("step: 0  reward: 0.00")
        self.status_label.setStyleSheet("color: #8f8; font-family: Menlo")
        btn_row.addWidget(self.status_label, stretch=1)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("Reset Episode")
        self.reset_btn.clicked.connect(self._reset_episode)
        btn_row.addWidget(self.reset_btn)

        self.record_btn = QPushButton("Record MP4")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self._on_record_toggled)
        btn_row.addWidget(self.record_btn)
        layout.addLayout(btn_row)

    # ------- policy loop --------------------------------------------------

    def _load_selected(self) -> None:
        g = self._genomes[self._selected_idx]
        try:
            self.net.load_state_dict(g["state_dict"])
        except RuntimeError as exc:
            # Architecture mismatch: the checkpoint was trained with a
            # different num_actions / network spec than the current
            # profile. Surface this in the status bar instead of
            # crashing the window when the user changes the combo box.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Genome %s load_state_dict failed: %s",
                g.get("genome_id", "?"), exc,
            )
            if hasattr(self, "status_label"):
                self.status_label.setText(
                    f"genome {g.get('genome_id', '?')} arch mismatch — "
                    "retrain or pick a different checkpoint"
                )
            return
        self.net.to(self.device)
        self.net.eval()
        # Prefer the Core ML / ANE policy for single-frame inference —
        # 8× faster than PyTorch MPS at batch=1 on M4 Max (bench via
        # scripts/bench_coreml_ane.py: 0.081 ms/fwd vs 0.649 ms/fwd).
        # The trainer pre-exports each checkpoint to a paired
        # `<checkpoint>.mlpackage`; we just load it here. When ONLY the
        # elite was exported (most cases), genomes other than the fittest
        # fall back to PyTorch MPS.
        #
        # Any failure is silent — viewer falls back to MPS via self.net.
        self._coreml_policy = None
        try:
            from src.models.coreml_export import maybe_export, CoreMLPolicy
            # Re-derive the checkpoint path from the genome's data.
            ckpt_path = getattr(self, "_checkpoint_path", None)
            if ckpt_path is None:
                self._coreml_policy = None
            else:
                from pathlib import Path as _P
                mlpkg = _P(ckpt_path).with_suffix(".mlpackage")
                if mlpkg.exists() and self._selected_idx == 0:
                    # Pre-exported elite — instant load, no conversion.
                    self._coreml_policy = CoreMLPolicy(mlpkg)
                else:
                    # Non-elite genome or no pre-export — convert on
                    # the fly to a temp file. ~100 ms one-time per
                    # genome pick; amortized across all subsequent
                    # frames of replay.
                    import tempfile, os
                    tmpdir = tempfile.gettempdir()
                    tmp_ml = os.path.join(
                        tmpdir,
                        f"nes_replay_{g.get('genome_id', 'x')}.mlpackage",
                    )
                    exported = maybe_export(
                        self.net,
                        tmp_ml,
                        num_actions=len(self.action_space),
                    )
                    if exported is not None:
                        self._coreml_policy = CoreMLPolicy(exported)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "CoreML policy unavailable, using MPS: %s", exc
            )

    def _tick(self) -> None:
        if self._paused:
            return
        # This runs from a QTimer slot; an uncaught exception here aborts
        # the app on PyQt6. Contain any inference/emulator failure, stop
        # the loop, and report it in the status bar.
        try:
            self._tick_step()
        except Exception as exc:
            self._timer.stop()
            self._paused = True
            import logging as _logging
            _logging.getLogger(__name__).exception("replay step failed")
            try:
                self.status_label.setText(
                    f"replay error: {type(exc).__name__}: {exc}"
                )
                self.pause_btn.setText("Error — Resume resets")
            except Exception:
                pass

    def _tick_step(self) -> None:
        obs = self._stacker.push(self.env.get_frame())
        # Prefer Core ML / ANE — 8× faster than PyTorch MPS at batch=1
        # on M4 Max. Falls back to MPS if compilation failed.
        if self._coreml_policy is not None:
            obs_t = torch.from_numpy(obs).float().div_(255.0)
            try:
                action_idx = self._coreml_policy.act(
                    obs_t, deterministic=self._deterministic
                )
            except Exception:
                # Runtime failure — drop Core ML for the rest of this
                # genome and fall back to MPS. Next genome change will
                # re-attempt compilation.
                self._coreml_policy = None
                obs_t = obs_t.to(self.device)
                action_idx = self.net.act(obs_t, deterministic=self._deterministic)
        else:
            obs_t = torch.from_numpy(obs).float().div_(255.0).to(self.device)
            action_idx = self.net.act(obs_t, deterministic=self._deterministic)
        bitmask = _action_to_bitmask(self.action_space[action_idx], _NAME_TO_BIT)

        frame, done = self.env.step(bitmask)
        self._step_count += 1
        if self._recorder is not None:
            self._recorder.write(frame)
        self._paint(frame)
        self.status_label.setText(
            f"step: {self._step_count}  action_idx: {action_idx}  "
            f"buttons: {self.action_space[action_idx] or 'NOOP'}"
        )

        if done:
            self._paused = True
            self.pause_btn.setText("Game Over — Resume resets")

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

    # ------- handlers -----------------------------------------------------

    def _on_genome_changed(self, idx: int) -> None:
        self._selected_idx = idx
        self._load_selected()
        self._reset_episode()

    def _on_det_toggled(self, checked: bool) -> None:
        self._deterministic = checked

    def _on_record_toggled(self, checked: bool) -> None:
        if checked:
            from src.gui.file_dialogs import remembered_save
            path = remembered_save(
                self,
                key="replay_record_mp4",
                title="Save replay as MP4",
                default_filename="replay.mp4",
                file_filter="MP4 video (*.mp4);;All files (*)",
            )
            if not path:
                self.record_btn.setChecked(False)
                return
            # The QTimer drives the emulator at _TARGET_FPS; matching the
            # writer's fps avoids 2× playback speed (writer claimed 30 while
            # frames arrived at 60). A missing codec / unwritable path
            # raises here — surface it instead of crashing the window.
            try:
                self._recorder = VideoWriter(path, fps=_TARGET_FPS)
            except Exception as exc:
                self._recorder = None
                # Re-enters this handler with checked=False, which resets
                # the button label; set the status afterward so it sticks.
                self.record_btn.setChecked(False)
                try:
                    self.status_label.setText(
                        f"Recording failed: {type(exc).__name__}: {exc}"
                    )
                except Exception:
                    pass
                return
            self.record_btn.setText("Stop Recording")
            self.status_label.setText(f"Recording to {Path(path).name}")
        else:
            if self._recorder is not None:
                self._recorder.close()
                n = self._recorder.frames_written
                self._recorder = None
                self.status_label.setText(f"Recording saved ({n} frames)")
            self.record_btn.setText("Record MP4")

    def _toggle_pause(self) -> None:
        if self.env.done:
            self._reset_episode()
            return
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")

    def _reset_episode(self) -> None:
        try:
            first_frame = self.env.reset()
        except Exception as exc:
            # A reset failure (e.g. bad start-state) must not crash the
            # window from a button click. Report and leave it paused.
            self._paused = True
            import logging as _logging
            _logging.getLogger(__name__).exception("replay episode reset failed")
            try:
                self.status_label.setText(
                    f"reset failed: {type(exc).__name__}: {exc}"
                )
            except Exception:
                pass
            return
        self._stacker.reset(first_frame)
        self._step_count = 0
        self._paused = False
        self.pause_btn.setText("Pause")
        self._paint(first_frame)

    def closeEvent(self, event) -> None:
        # Teardown must never raise out of closeEvent — that would abort
        # the app on window close. Guard each step independently.
        try:
            self._timer.stop()
        except Exception:
            pass
        if self._recorder is not None:
            try:
                self._recorder.close()
            except Exception:
                pass
            self._recorder = None
        try:
            self.env.close()
        except Exception:
            pass
        super().closeEvent(event)
