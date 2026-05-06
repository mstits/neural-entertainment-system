"""
PyQt6 main window for the Neural Entertainment System (NES).

Provides:
- ROM file picker
- Game profile dropdown (auto-populated from configs/ directory)
- Parallel instance count slider (1 to 64)
- Start / Stop / Pause buttons
- Training metrics display
- Opens the emulator grid window on Start

The trainer runs in a separate subprocess. GUI talks to it via queues.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui.file_dialogs import remembered_open, remembered_open_many

_REPO_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = _REPO_ROOT / "configs"
ROMS_DIR = _REPO_ROOT / "roms"
SAMPLES_DIR = _REPO_ROOT / "samples"
CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
RECENTS_PATH = Path.home() / ".nes_recents.json"


def _load_recents() -> dict:
    """Read persisted last-used picks from disk. Tolerant of missing/corrupt."""
    try:
        with open(RECENTS_PATH, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_recents(data: dict) -> None:
    """Best-effort persistence — never raises."""
    try:
        RECENTS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


class MainWindow(QMainWindow):
    """Main control window."""

    start_requested = pyqtSignal(dict)  # emits training config
    stop_requested = pyqtSignal()
    play_requested = pyqtSignal(str)    # emits the ROM path for play mode
    replay_requested = pyqtSignal(dict) # emits {rom, profile, checkpoint, state}
    tune_rewards_requested = pyqtSignal()
    audio_mixer_requested = pyqtSignal()
    # Emitted when the user picks a new start-state file. AppController
    # routes this to the running trainer (if any) via the reward queue;
    # the trainer rebuilds the worker pool with the new path so future
    # episodes boot into the new state — no Stop / Start dance needed.
    start_state_change_requested = pyqtSignal(str)  # absolute path or "" to clear
    dashboard_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NES")
        self.resize(720, 540)

        self._rom_path: str = ""
        self._start_state_path: str = ""
        self._reward_overrides_path: str = ""
        self._bc_demo_path: str = ""
        self._build_ui()
        # Auto-populate from last session if the files still exist.
        self._apply_recents(_load_recents(), silent=True)
        # Restore window position + size from last session.
        from PyQt6.QtCore import QSettings
        geom = QSettings("NeuralEntertainmentSystem", "gui").value("main_window/geometry")
        if geom is not None:
            self.restoreGeometry(geom)

    def closeEvent(self, event):
        # If training is live (stop_btn is the canonical "running"
        # signal — enabled iff the trainer accepted Start and hasn't
        # been stopped), confirm before quit. Closing the GUI tears
        # down the trainer subprocess + drops the in-flight generation,
        # so an accidental cmd-W mid-run loses real work.
        if self.stop_btn.isEnabled():
            from PyQt6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("Training is running")
            box.setText("A training run is in progress.")
            box.setInformativeText(
                "Quitting now will stop training and discard any in-flight "
                "generation. The most recent saved checkpoint is kept."
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Discard
            )
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            box.button(QMessageBox.StandardButton.Discard).setText("Stop and quit")
            if box.exec() == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Persist window geometry so the next launch opens at the
        # same place + size. Done in closeEvent so resize/move done
        # at any point during the session is captured.
        from PyQt6.QtCore import QSettings
        QSettings("NeuralEntertainmentSystem", "gui").setValue(
            "main_window/geometry", self.saveGeometry()
        )
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- ROM picker ---
        rom_row = QHBoxLayout()
        rom_row.addWidget(QLabel("ROM:"))
        self.rom_label = QLabel("(no ROM selected)")
        # Use font-style + system palette instead of hardcoded colors —
        # otherwise "color: #fff" disappears against a light-mode background.
        self.rom_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        rom_row.addWidget(self.rom_label, stretch=1)
        rom_btn = QPushButton("Choose ROM…")
        rom_btn.clicked.connect(self._on_pick_rom)
        rom_row.addWidget(rom_btn)
        self.play_btn = QPushButton("Play / Record…")
        self.play_btn.setToolTip(
            "Play the selected ROM with your keyboard and save a start state."
        )
        self.play_btn.clicked.connect(self._on_play)
        rom_row.addWidget(self.play_btn)
        layout.addLayout(rom_row)

        # --- Start state picker (optional) ---
        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("Start state:"))
        self.state_label = QLabel("(none — training starts from cold reset)")
        self.state_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        state_row.addWidget(self.state_label, stretch=1)
        state_btn = QPushButton("Choose State…")
        state_btn.clicked.connect(self._on_pick_state)
        state_row.addWidget(state_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear_state)
        state_row.addWidget(clear_btn)
        layout.addLayout(state_row)

        # --- Game profile dropdown ---
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Game profile:"))
        self.profile_dropdown = QComboBox()
        self._populate_profiles()
        profile_row.addWidget(self.profile_dropdown, stretch=1)
        profile_row.addSpacing(12)
        # Trainer mode picker. `(profile)` falls through to whatever
        # the YAML's `training_mode` key says (or `ga_ppo` if absent),
        # so existing profiles keep their default behavior. Selecting
        # an explicit value overrides the profile at run time.
        profile_row.addWidget(QLabel("Trainer:"))
        self.trainer_mode_dropdown = QComboBox()
        self.trainer_mode_dropdown.addItem("(profile)", "")
        self.trainer_mode_dropdown.addItem("GA + PPO", "ga_ppo")
        self.trainer_mode_dropdown.addItem("DreamerV3", "dreamer")
        self.trainer_mode_dropdown.setToolTip(
            "GA + PPO is the classic genetic algorithm + policy gradient "
            "trainer. DreamerV3 is the experimental world-model trainer "
            "(better sample efficiency on sparse-reward games like Zelda)."
        )
        profile_row.addWidget(self.trainer_mode_dropdown)
        layout.addLayout(profile_row)

        # --- Reward overrides (optional) ---
        # User can supply a small YAML with just a `reward_weights:` block
        # (and optionally `reinforce:` overrides). Merges on top of the
        # profile at trainer-init time. Lets you A/B reward variants
        # without editing the canonical game YAML.
        rw_row = QHBoxLayout()
        rw_row.addWidget(QLabel("Reward overrides:"))
        self.rw_label = QLabel("(profile defaults)")
        self.rw_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        rw_row.addWidget(self.rw_label, stretch=1)
        rw_btn = QPushButton("Load Reward File…")
        rw_btn.clicked.connect(self._on_pick_reward_overrides)
        rw_row.addWidget(rw_btn)
        rw_clear_btn = QPushButton("Clear")
        rw_clear_btn.clicked.connect(self._on_clear_reward_overrides)
        rw_row.addWidget(rw_clear_btn)
        layout.addLayout(rw_row)

        # --- Instance count slider ---
        # Default 24: post-PGO M4 Max throughput peak per
        # docs/proposals/archive/hot_path_baseline.md. Users on smaller chips
        # should drop to their physical core count.
        inst_row = QHBoxLayout()
        inst_row.addWidget(QLabel("Parallel instances:"))
        self.instance_spin = QSpinBox()
        self.instance_spin.setRange(1, 64)
        self.instance_spin.setValue(24)
        inst_row.addWidget(self.instance_spin)
        self.instance_slider = QSlider(Qt.Orientation.Horizontal)
        self.instance_slider.setRange(1, 64)
        self.instance_slider.setValue(24)
        self.instance_slider.valueChanged.connect(self.instance_spin.setValue)
        self.instance_spin.valueChanged.connect(self.instance_slider.setValue)
        inst_row.addWidget(self.instance_slider, stretch=1)
        layout.addLayout(inst_row)

        # --- Population size ---
        pop_row = QHBoxLayout()
        pop_row.addWidget(QLabel("Population size:"))
        self.population_spin = QSpinBox()
        self.population_spin.setRange(4, 256)
        self.population_spin.setValue(24)
        pop_row.addWidget(self.population_spin)
        pop_row.addStretch()
        layout.addLayout(pop_row)

        # --- Episode budget (agent-steps per episode) ---
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("Max episode steps:"))
        self.max_ep_spin = QSpinBox()
        self.max_ep_spin.setRange(100, 1_000_000)
        self.max_ep_spin.setSingleStep(500)
        self.max_ep_spin.setValue(36000)
        self.max_ep_spin.setToolTip(
            "Hard cap on agent-steps per episode. At frame_skip=16 and "
            "60 fps, N × 16 / 60 ≈ seconds of NES game time. 36 000 "
            "≈ 2.7 hours. Only fires if the game hasn't terminated and "
            "the activity timeout hasn't fired first."
        )
        ep_row.addWidget(self.max_ep_spin)
        ep_row.addSpacing(20)
        ep_row.addWidget(QLabel("Activity timeout steps:"))
        self.activity_spin = QSpinBox()
        self.activity_spin.setRange(0, 100_000)
        self.activity_spin.setSingleStep(60)
        self.activity_spin.setValue(900)
        self.activity_spin.setToolTip(
            "End the episode after this many agent-steps without any "
            "positive reward signal (new screen / kill / item / heart). "
            "0 disables. 900 ≈ 4 minutes of Zelda at frame_skip=16 — "
            "generous enough to ride out a sword fight, tight enough "
            "that idle/wall-bump genomes give up their slot fast."
        )
        ep_row.addWidget(self.activity_spin)
        ep_row.addStretch()
        layout.addLayout(ep_row)

        # --- Resume toggle ---
        resume_row = QHBoxLayout()
        self.resume_checkbox = QCheckBox("Resume from latest checkpoint if present")
        self.resume_checkbox.setChecked(True)
        self.resume_checkbox.setToolTip(
            "If ./checkpoints/gen_*.pt exists, continue training from the "
            "newest one. Uncheck to force a fresh run."
        )
        resume_row.addWidget(self.resume_checkbox)
        resume_row.addStretch()
        layout.addLayout(resume_row)

        # --- Behavioral cloning toggle + demo picker ---
        bc_row = QHBoxLayout()
        self.bc_checkbox = QCheckBox("Learn from a play recording (BC)")
        self.bc_checkbox.setChecked(False)
        self.bc_checkbox.setToolTip(
            "Pre-train the network to imitate a recorded .state.bin play "
            "trace before the GA starts. Gives agents a warm start instead "
            "of random weights. Only applies on fresh runs (not resume)."
        )
        bc_row.addWidget(self.bc_checkbox)
        self.bc_label = QLabel("(will reuse start-state if no file picked)")
        self.bc_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        bc_row.addWidget(self.bc_label, stretch=1)
        bc_btn = QPushButton("Choose Demo…")
        bc_btn.setToolTip(
            "Pick a longer .state.bin playthrough recording to imitate.\n"
            "Convert FCEUX/TASVideos .fm2 files with scripts/convert_fm2.py;\n"
            "see samples/README.md for sources."
        )
        bc_btn.clicked.connect(self._on_pick_bc_demo)
        bc_row.addWidget(bc_btn)
        bc_clear_btn = QPushButton("Clear")
        bc_clear_btn.clicked.connect(self._on_clear_bc_demo)
        bc_row.addWidget(bc_clear_btn)
        layout.addLayout(bc_row)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.load_last_btn = QPushButton("Load Last")
        self.load_last_btn.setToolTip(
            "Populate ROM, start state, profile, and settings from your last session."
        )
        self.load_last_btn.clicked.connect(self._on_load_last)
        btn_row.addWidget(self.load_last_btn)

        self.replay_btn = QPushButton("Replay Checkpoint…")
        self.replay_btn.setToolTip(
            "Load a gen_*.pt checkpoint and watch the best genome play the game."
        )
        self.replay_btn.clicked.connect(self._on_replay)
        btn_row.addWidget(self.replay_btn)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        self.tune_btn = QPushButton("Tune Rewards…")
        self.tune_btn.setToolTip(
            "Open a live slider panel for the active game's reward weights. "
            "Changes push to the trainer without restarting."
        )
        self.tune_btn.clicked.connect(self._on_tune_rewards)
        self.tune_btn.setEnabled(False)  # only while training runs
        btn_row.addWidget(self.tune_btn)

        self.audio_btn = QPushButton("Audio Mixer…")
        self.audio_btn.setToolTip(
            "Open the audio mixer: mute, solo one instance, or mix all.\n"
            "Music is driven by each instance's RAM 'current-song' byte.\n"
            "Drop pre-rendered loops under audio/<game>/<song_id>.ogg for real music;\n"
            "otherwise a deterministic tone stands in."
        )
        self.audio_btn.clicked.connect(self._on_audio_mixer)
        self.audio_btn.setEnabled(False)  # only while training runs
        btn_row.addWidget(self.audio_btn)

        self.dashboard_btn = QPushButton("Dashboard…")
        self.dashboard_btn.setToolTip(
            "Reopen the unified training dashboard if it was closed. "
            "Always opens automatically on Start."
        )
        self.dashboard_btn.clicked.connect(self.dashboard_requested.emit)
        self.dashboard_btn.setEnabled(False)
        btn_row.addWidget(self.dashboard_btn)
        layout.addLayout(btn_row)

        # --- Metrics display ---
        layout.addWidget(QLabel("Training metrics:"))
        self.metrics_label = QLabel("(not running)")
        self.metrics_label.setStyleSheet(
            "background: #1a1a1a; color: #8f8; font-family: Menlo; padding: 8px"
        )
        self.metrics_label.setMinimumHeight(120)
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.metrics_label, stretch=1)

    def _populate_profiles(self) -> None:
        import yaml as _yaml
        self.profile_dropdown.clear()
        if not CONFIGS_DIR.exists():
            return
        for yaml_file in sorted(CONFIGS_DIR.glob("*.yaml")):
            try:
                data = _yaml.safe_load(yaml_file.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if not data.get("action_space"):
                # Override-only profile (e.g. zelda_gui_tuned): hide from
                # the picker so a user can't select it as a standalone
                # profile and crash the trainer with empty action_space.
                continue
            self.profile_dropdown.addItem(yaml_file.stem, str(yaml_file))

    def _on_pick_rom(self) -> None:
        path = remembered_open(
            self,
            key="rom",
            title="Select NES ROM",
            file_filter="NES ROMs (*.nes);;All files (*)",
            default_dir=str(ROMS_DIR),
        )
        if path:
            self._rom_path = path
            self.rom_label.setText(Path(path).name)
            # Bold + default palette color — readable in both light and dark mode.
            self.rom_label.setStyleSheet("font-weight: bold")

    def _on_pick_state(self) -> None:
        # Default to the picked ROM's directory if there is one;
        # otherwise the helper falls back to its remembered key dir.
        rom_dir = str(Path(self._rom_path).parent) if self._rom_path else str(ROMS_DIR)
        path = remembered_open(
            self,
            key="start_state",
            title="Select start state",
            file_filter="Start state (*.state.bin);;All files (*)",
            default_dir=rom_dir,
        )
        if path:
            self._start_state_path = path
            self.state_label.setText(Path(path).name)
            self.state_label.setStyleSheet("font-weight: bold")
            # Tell anyone listening (AppController if a run is live) so a
            # mid-run pick can take effect without a Stop / Start cycle.
            self.start_state_change_requested.emit(path)

    def _on_clear_state(self) -> None:
        self._start_state_path = ""
        self.state_label.setText("(none — training starts from cold reset)")
        self.state_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")
        # Empty string signals "no start state"; trainer cold-resets
        # instead of cold-replaying any recording.
        self.start_state_change_requested.emit("")

    def _on_pick_reward_overrides(self) -> None:
        path = remembered_open(
            self,
            key="reward_overrides",
            title="Select reward overrides YAML",
            file_filter="YAML (*.yaml *.yml);;All files (*)",
            default_dir=str(CONFIGS_DIR),
        )
        if path:
            self._reward_overrides_path = path
            self.rw_label.setText(Path(path).name)
            self.rw_label.setStyleSheet("font-weight: bold")

    def _on_clear_reward_overrides(self) -> None:
        self._reward_overrides_path = ""
        self.rw_label.setText("(profile defaults)")
        self.rw_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")

    def _on_pick_bc_demo(self) -> None:
        # Default to samples/ if it exists; otherwise the ROM dir; the
        # helper falls back further to the remembered key's last dir.
        default_dir = (
            str(SAMPLES_DIR) if SAMPLES_DIR.exists()
            else (str(Path(self._rom_path).parent) if self._rom_path else str(ROMS_DIR))
        )
        # Multi-select: pass several .fm2 / .state.bin files at once and
        # the trainer's BC pipeline will replay each from a fresh cold-
        # boot env, concatenating the (state, action) pairs. Diverse
        # recordings give the policy much better state coverage than a
        # single TAS run on a long game.
        paths = remembered_open_many(
            self,
            key="bc_demo",
            title="Select behavioral-cloning demo recording(s)",
            # Accept both FCEUX movies (.fm2) and our native .state.bin.
            # FM2s are auto-converted on the fly so users don't need the CLI.
            file_filter="Play recording (*.state.bin *.fm2);;"
            "FCEUX movie (*.fm2);;"
            "Native play recording (*.state.bin);;"
            "All files (*)",
            default_dir=default_dir,
        )
        if not paths:
            return

        # Convert any .fm2s to .state.bin alongside the source. Collect
        # the resolved paths (post-conversion for fm2s, unchanged for
        # native recordings) into a flat list.
        resolved: list[str] = []
        for path in paths:
            if path.lower().endswith(".fm2"):
                resolved_path = self._convert_fm2(path)
                if resolved_path is None:
                    return  # critical error already surfaced via QMessageBox
                resolved.append(resolved_path)
            else:
                resolved.append(path)

        # The trainer accepts a colon-joined string of paths and splits
        # them in `_bc_demo_paths()`. Joining here keeps the single
        # `bc_demo_path` config field unchanged downstream.
        joined = ":".join(resolved)
        self._bc_demo_path = joined
        if len(resolved) == 1:
            self.bc_label.setText(Path(resolved[0]).name)
        else:
            self.bc_label.setText(
                f"{len(resolved)} files: {', '.join(Path(p).name for p in resolved[:3])}"
                + ("…" if len(resolved) > 3 else "")
            )
        self.bc_label.setStyleSheet("font-weight: bold")
        # Picking a demo file implies the user wants BC on.
        self.bc_checkbox.setChecked(True)
        QMessageBox.information(
            self,
            "BC demos selected",
            f"{len(resolved)} demo file(s) ready for BC pretraining.\n\n"
            "Behavioral cloning is now enabled. Uncheck "
            "'Resume from latest checkpoint' on Start so the GA seeds "
            "from the BC weights instead of an existing checkpoint.",
        )

    def _convert_fm2(self, path: str) -> str | None:
        """Convert an FCEUX `.fm2` recording to the trainer's native
        `.state.bin` format. Writes the converted file alongside the
        source. Returns the converted path on success, None on error
        (a QMessageBox is shown to the user)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_fm2",
            str(Path(__file__).parent.parent.parent / "scripts" / "convert_fm2.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            data = mod.fm2_to_bytes(Path(path))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "FM2 conversion failed",
                f"Could not read {Path(path).name}:\n\n{exc}",
            )
            return None
        converted = Path(path).with_suffix(".state.bin")
        try:
            converted.write_bytes(data)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Cannot write converted file",
                f"Failed to write {converted}:\n\n{exc}",
            )
            return None
        return str(converted)

    def _on_clear_bc_demo(self) -> None:
        self._bc_demo_path = ""
        self.bc_label.setText("(will reuse start-state if no file picked)")
        self.bc_label.setStyleSheet("font-style: italic; color: palette(placeholder-text)")

    def _on_play(self) -> None:
        if not self._rom_path:
            self._show_status("Choose a ROM first.")
            return
        self.play_requested.emit(self._rom_path)

    def _on_replay(self) -> None:
        if not self._rom_path:
            self._show_status("Choose a ROM first.")
            return
        profile_path = self.profile_dropdown.currentData()
        if not profile_path:
            self._show_status("No game profile selected.")
            return
        path = remembered_open(
            self,
            key="checkpoint",
            title="Pick a checkpoint to replay",
            file_filter="Checkpoint (*.pt);;All files (*)",
            default_dir=str(CHECKPOINTS_DIR) if CHECKPOINTS_DIR.exists() else None,
        )
        if not path:
            return
        self.replay_requested.emit({
            "rom_path": self._rom_path,
            "profile_path": profile_path,
            "checkpoint_path": path,
            "start_state_path": self._start_state_path or None,
        })

    def _on_start(self) -> None:
        if not self._rom_path:
            self._show_status("Choose a ROM first.")
            return

        profile_path = self.profile_dropdown.currentData()
        if not profile_path:
            self._show_status("No game profile selected.")
            return

        # BC demo resolution: explicit picker first, then fall back to
        # the start-state file (common quick-start case), then None.
        if self.bc_checkbox.isChecked():
            bc_path = self._bc_demo_path or self._start_state_path or None
        else:
            bc_path = None

        config = {
            "rom_path": self._rom_path,
            "profile_path": profile_path,
            "num_instances": self.instance_spin.value(),
            "population_size": self.population_spin.value(),
            "start_state_path": self._start_state_path or None,
            "resume": self.resume_checkbox.isChecked(),
            "bc_demo_path": bc_path,
            "reward_overrides_path": self._reward_overrides_path or None,
            "max_episode_steps": int(self.max_ep_spin.value()),
            "activity_timeout_steps": int(self.activity_spin.value()),
            # Empty string means "use whatever the profile YAML says".
            "trainer_mode_override": self.trainer_mode_dropdown.currentData() or "",
        }
        self._persist_recents(config)
        self.start_requested.emit(config)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tune_btn.setEnabled(True)
        self.audio_btn.setEnabled(True)
        self.dashboard_btn.setEnabled(True)
        self._show_status("Starting training…")

    def _on_stop(self) -> None:
        self.stop_requested.emit()
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.tune_btn.setEnabled(False)
        self.audio_btn.setEnabled(False)
        self.dashboard_btn.setEnabled(False)
        self._show_status("Stopping…")

    def _on_tune_rewards(self) -> None:
        self.tune_rewards_requested.emit()

    def _on_audio_mixer(self) -> None:
        self.audio_mixer_requested.emit()

    def _show_status(self, msg: str) -> None:
        self.metrics_label.setText(msg)

    # ------ recent-files convenience ---------------------------------------

    def _on_load_last(self) -> None:
        data = _load_recents()
        if not data:
            self._show_status("No previous session found.")
            return
        applied = self._apply_recents(data, silent=False)
        if applied:
            self._show_status(f"Loaded last session: {Path(self._rom_path).name}")
        else:
            self._show_status("Last session's files no longer exist on disk.")

    def _apply_recents(self, data: dict, silent: bool) -> bool:
        """Populate the UI from persisted recents. Skips fields whose files
        no longer exist so stale entries don't put us in a half-configured
        state. Returns True if at least one field was applied."""
        applied = False

        rom_path = data.get("rom_path", "")
        if rom_path and Path(rom_path).exists():
            self._rom_path = rom_path
            self.rom_label.setText(Path(rom_path).name)
            self.rom_label.setStyleSheet("font-weight: bold")
            applied = True

        state_path = data.get("start_state_path") or ""
        if state_path and Path(state_path).exists():
            self._start_state_path = state_path
            self.state_label.setText(Path(state_path).name)
            self.state_label.setStyleSheet("font-weight: bold")
            applied = True
        elif not silent and state_path:
            # User asked to load and the recents named a start-state, but
            # the file vanished from disk — clear the in-memory pointer
            # so we don't pass a non-existent path to the trainer later.
            self._on_clear_state()

        rw_path = data.get("reward_overrides_path") or ""
        if rw_path and Path(rw_path).exists():
            self._reward_overrides_path = rw_path
            self.rw_label.setText(Path(rw_path).name)
            self.rw_label.setStyleSheet("font-weight: bold")
            applied = True
        elif not silent and rw_path:
            # Same thing for the reward-overrides file. Previously this
            # branch wrongly called _on_clear_state(), wiping the start
            # state when the reward file was the one missing — load a
            # session whose reward overrides got deleted and you'd lose
            # your start state too.
            self._on_clear_reward_overrides()

        bc_path = data.get("bc_demo_path") or ""
        if bc_path:
            # Multi-demo path is colon-joined; single is just one path.
            # Validate every component still exists; if any are gone,
            # treat the whole thing as invalid (better than partially
            # restoring a broken set).
            candidates = bc_path.split(":") if ":" in bc_path else [bc_path]
            if all(Path(p).exists() for p in candidates):
                self._bc_demo_path = bc_path
                if len(candidates) == 1:
                    self.bc_label.setText(Path(candidates[0]).name)
                else:
                    self.bc_label.setText(
                        f"{len(candidates)} files: "
                        + ", ".join(Path(p).name for p in candidates[:3])
                        + ("…" if len(candidates) > 3 else "")
                    )
                self.bc_label.setStyleSheet("font-weight: bold")
                applied = True

        profile = data.get("profile")
        if profile:
            idx = self.profile_dropdown.findText(profile)
            if idx >= 0:
                self.profile_dropdown.setCurrentIndex(idx)
                applied = True

        if "num_instances" in data:
            self.instance_spin.setValue(int(data["num_instances"]))
        if "population_size" in data:
            self.population_spin.setValue(int(data["population_size"]))
        if "resume" in data:
            self.resume_checkbox.setChecked(bool(data["resume"]))
        if "bc" in data:
            self.bc_checkbox.setChecked(bool(data["bc"]))
        if "max_episode_steps" in data:
            self.max_ep_spin.setValue(int(data["max_episode_steps"]))
        if "activity_timeout_steps" in data:
            self.activity_spin.setValue(int(data["activity_timeout_steps"]))
        tm = data.get("trainer_mode_override", "")
        if tm:
            for i in range(self.trainer_mode_dropdown.count()):
                if self.trainer_mode_dropdown.itemData(i) == tm:
                    self.trainer_mode_dropdown.setCurrentIndex(i)
                    break

        return applied

    def _persist_recents(self, config: dict) -> None:
        _save_recents({
            "rom_path": config["rom_path"],
            "start_state_path": config.get("start_state_path"),
            "reward_overrides_path": config.get("reward_overrides_path"),
            "bc_demo_path": self._bc_demo_path or None,
            "profile": self.profile_dropdown.currentText(),
            "num_instances": config["num_instances"],
            "population_size": config["population_size"],
            "resume": config.get("resume", False),
            "bc": config.get("bc_demo_path") is not None,
            "max_episode_steps": config.get("max_episode_steps"),
            "activity_timeout_steps": config.get("activity_timeout_steps"),
            "trainer_mode_override": config.get("trainer_mode_override", ""),
        })

    def update_metrics(self, data: dict) -> None:
        """Called periodically by the training bridge to show live metrics."""
        # Success rate previously defaulted to 0.0 before the isinstance
        # check, so a metrics dict that omitted the field would still
        # render as "Success rate: 0.0%" — visually indistinguishable
        # from a real zero rate. Use sentinel None and check membership
        # explicitly so missing fields show as "-".
        sr = data.get('success_rate')
        lines = [
            f"Generation:       {data.get('generation', '-')}",
            f"Best fitness:     {data.get('best_fitness', '-'):.2f}" if isinstance(data.get('best_fitness'), (int, float)) else "Best fitness:     -",
            f"Avg fitness:      {data.get('avg_fitness', '-'):.2f}" if isinstance(data.get('avg_fitness'), (int, float)) else "Avg fitness:      -",
            f"Curriculum stage: {data.get('stage', '-')}",
            f"Success rate:     {sr:.1%}" if isinstance(sr, (int, float)) else "Success rate:     -",
            f"Episodes:         {data.get('episodes', '-')}",
        ]
        self.metrics_label.setText("\n".join(lines))
