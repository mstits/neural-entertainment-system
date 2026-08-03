#!/usr/bin/env python3
"""THE SHOW CONTROL PANEL — a persistent settings surface for the live show.

The front door to everything the search system can do and has already done,
AND the control desk you keep open next to the stream. In one window it:

  - browses every configured game (short name + a live-rendered start-state
    thumbnail) on the left; selecting a game LOADS its profile into a full
    settings editor;
  - inspects the selected game's start state and its BANKED demos -- the
    verified SMB power-on -> 8-4 full clear, the Castlevania block chain,
    each game's Go-Explore bootstrap frontier -- every entry read off a
    real receipt;
  - exposes EVERY show + profile knob as an editable widget grouped into
    Game / Display / Audio / Show Panels / Solver, so the stream is a
    controlled experience rather than a fixed command line;
  - LOADS and SAVES self-contained profiles: any configs/*.yaml round-trips
    (name, rom, start_state_path, action_space, solve:/reinforce:, ram_mapping
    all preserved) with a top-level `show:` block carrying the runtime setup
    (workers/scale/volume/view/audio/panels...), so one file restores the
    whole stream;
  - picks a MODE (Live Solve = the search plays it live now, victory lap on
    each clear; Replay = lap a banked solution in the hero cam) and LAUNCHES
    the show as a NON-BLOCKING subprocess, so the panel stays open and you
    keep steering / relaunch.

Nothing here beats a game itself: it composes the argv and spawns
scripts/live_solve_show.py, which is the working show. The panel is a
separate control surface -- the show is its own window.

Usage:
    make launcher
    make control-panel
    python scripts/show_launcher.py
    python scripts/show_launcher.py --list        # print the catalog and exit
    QT_QPA_PLATFORM=offscreen python scripts/show_launcher.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402
from scripts import demo_catalog  # noqa: E402

SHOW_SCRIPT = REPO / "scripts" / "live_solve_show.py"
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"
# A scratch profile the panel writes on launch when the editor's solver knobs
# diverge from the backing config -- so edits reach the show WITHOUT clobbering
# the repo's canonical configs/<game>.yaml.
WORKING_PROFILE = RUNS / "launcher" / "active_profile.yaml"

# The interpreter running the launcher is the venv python under `make
# launcher`; the show inherits it so the same nes_core .so is used.
PYTHON = sys.executable or "python"

FRAME_W, FRAME_H = 256, 240

# live_solve_show.py's own argparse defaults. A runtime flag is only emitted
# when the panel's value differs from these, so a launch stays byte-identical
# to the base show unless the operator changed something.
SHOW_DEFAULTS = {
    "workers": 12,
    "scale": 3,
    "volume": 0.6,
    "view": "swarm",
    "minutes_per_level": 120.0,
    "audio": "both",
    "chorus_pitch": "ff",
    "chorus_voices": -1,
    "fx": "off",
    "heatmap": False,
    "clips": False,
    "spectator_lite": False,
    "state": None,
}

# Profile knobs surfaced in the Solver group when the loaded config actually
# carries them (we never invent a knob a profile lacks). Each: (dotted path,
# leaf key, kind, (lo, hi)). Leaf keys are unique per config, so a save can
# rewrite exactly the one line without disturbing the rest of the file.
SOLVER_KNOBS = [
    ("frame_skip", "frame_skip", "int", (1, 64)),
    ("max_episode_steps", "max_episode_steps", "int", (100, 1_000_000)),
    ("solve.progress_cap", "progress_cap", "int", (0, 10_000_000)),
    ("solve.kill_key_local", "kill_key_local", "bool", None),
    ("reinforce.rollout_steps", "rollout_steps", "int", (16, 8192)),
    ("reinforce.gamma", "gamma", "float", (0.0, 1.0)),
    ("reinforce.lr", "lr", "float", (0.0, 1.0)),
    ("reinforce.entropy_coef", "entropy_coef", "float", (0.0, 1.0)),
    ("reinforce.rnd_intrinsic_coef", "rnd_intrinsic_coef", "float", (0.0, 10.0)),
]


# ---------------------------------------------------------------------------
# pure helpers (unit-testable without Qt)
# ---------------------------------------------------------------------------

def _placeholder_frame() -> np.ndarray:
    """A flat slate frame with a diagonal marker, for when a real thumbnail
    can't be rendered (missing ROM/state or a boot failure)."""
    f = np.full((FRAME_H, FRAME_W, 3), 32, dtype=np.uint8)
    for i in range(min(FRAME_H, FRAME_W)):
        f[i, i] = (70, 70, 78)
        if i + 1 < FRAME_W:
            f[i, i + 1] = (70, 70, 78)
    return f


def render_thumbnail(rom_path: str, start_state_path: Optional[str],
                     settle: int = 60) -> np.ndarray:
    """Render a game's start-state frame as a (240,256,3) uint8 array.

    Loads the ROM in a throwaway env; if a start state is given, restores it
    and steps one frame to render it; otherwise settles the boot/title for a
    few frames (SMB has no captured state -- its title screen is the thumb).
    Any failure returns a placeholder rather than raising, so a bad ROM never
    takes down the launcher."""
    env = None
    try:
        env = nes_core.NESEnvironment(rom_path=rom_path, frame_skip=1)
        env.reset()
        loaded = False
        if start_state_path:
            p = Path(start_state_path)
            if not p.is_absolute():
                p = REPO / p
            if p.is_file():
                env.load_state(p.read_bytes())
                env.step(0)
                loaded = True
        if not loaded:
            for _ in range(max(1, settle)):
                env.step(0)
        frame = np.asarray(env.get_frame(), dtype=np.uint8)
        if frame.shape != (FRAME_H, FRAME_W, 3):
            return _placeholder_frame()
        return frame.copy()
    except Exception:
        return _placeholder_frame()
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


def replay_target_of(entry: dict) -> Optional[str]:
    """The --replay target for a banked entry, or None if it isn't a
    replayable single-lap solution. Prefers the exact actions file; falls
    back to the run-dir (live_solve_show.resolve_replay picks its newest
    solution)."""
    if not isinstance(entry, dict):
        return None
    if entry.get("actions_path"):
        return entry["actions_path"]
    # A run-dir is replayable only if it directly holds a solutions/ dir.
    run_dir = entry.get("run_dir")
    if run_dir and entry.get("kind") == "solution":
        if (REPO / run_dir / "solutions").is_dir():
            return run_dir
    return None


def _emit(argv: list[str], flag: str, value, default) -> None:
    """Append `flag value` only when value is set and differs from the show's
    own default (keeps the launched command minimal + faithful)."""
    if value is not None and value != default:
        argv += [flag, str(value)]


def build_launch_argv(python_exe: str, *, game: Optional[str] = None,
                      profile: Optional[str] = None, mode: str = "solve",
                      state: Optional[str] = None,
                      replay: Optional[str] = None,
                      workers: Optional[int] = None,
                      scale: Optional[int] = None,
                      volume: Optional[float] = None,
                      view: Optional[str] = None,
                      minutes_per_level: Optional[float] = None,
                      audio: Optional[str] = None,
                      chorus_pitch: Optional[str] = None,
                      chorus_voices: Optional[int] = None,
                      heatmap: bool = False, clips: bool = False,
                      fx: str = "off",
                      spectator_lite: bool = False) -> list[str]:
    """Compose the live_solve_show.py argv for a launch.

    --game is preferred over --profile (a convenience the show already
    resolves to configs/<name>.yaml). Mode is always explicit; --replay is
    only emitted in replay mode. Runtime knobs are emitted only when they
    differ from the show's defaults, so the invocation stays byte-identical
    to the base show unless the operator changed a setting."""
    argv = [python_exe, str(SHOW_SCRIPT)]
    if game:
        argv += ["--game", game]
    elif profile:
        argv += ["--profile", profile]
    if state:
        argv += ["--state", state]
    argv += ["--mode", mode]
    if mode == "replay" and replay:
        argv += ["--replay", replay]
    if workers is not None:
        argv += ["--workers", str(int(workers))]
    if scale is not None:
        argv += ["--scale", str(int(scale))]
    _emit(argv, "--volume", volume, SHOW_DEFAULTS["volume"])
    _emit(argv, "--view", view, SHOW_DEFAULTS["view"])
    _emit(argv, "--minutes-per-level", minutes_per_level,
          SHOW_DEFAULTS["minutes_per_level"])
    _emit(argv, "--audio", audio, SHOW_DEFAULTS["audio"])
    _emit(argv, "--chorus-pitch", chorus_pitch, SHOW_DEFAULTS["chorus_pitch"])
    _emit(argv, "--chorus-voices", chorus_voices,
          SHOW_DEFAULTS["chorus_voices"])
    if heatmap:
        argv += ["--heatmap"]
    if clips:
        argv += ["--clips"]
    if fx and fx != "off":
        argv += ["--fx", fx]
    if spectator_lite:
        argv += ["--spectator-lite"]
    return argv


def catalog_summary(catalog: dict) -> dict:
    """Totals for the 'what we've done' banner: games, banked wins (cleared
    entries), and the headline (the SMB full-clear chain if present)."""
    games = catalog.get("games", [])
    wins = 0
    headline = None
    for g in games:
        for b in g.get("banked", []):
            if b.get("cleared"):
                wins += 1
            if (g.get("name") == "mario" and b.get("kind") == "chain"
                    and b.get("cleared") and headline is None):
                headline = b.get("label")
    return {"games": len(games), "wins": wins, "headline": headline}


def display_name(name: str) -> str:
    return demo_catalog.PRETTY_NAMES.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# profile I/O helpers (text-preserving; ruamel-free)
#
# ruamel.yaml isn't installed here, so a full re-serialize would drop the
# configs' extensive comments/order. Instead we keep the original file text
# verbatim and only touch the exact lines that changed: a scalar solver line
# is rewritten in place (indent + inline comment preserved), and the panel's
# `show:` block is stripped and re-appended. Everything else -- the whole
# solve:/reinforce:/ram_mapping structure -- is byte-preserved.
# ---------------------------------------------------------------------------

def parse_profile(path) -> tuple[dict, str]:
    """(parsed doc, raw text) for a profile file. Missing/garbage -> ({}, '')
    so the editor degrades to defaults rather than raising."""
    try:
        text = Path(path).read_text()
    except Exception:
        return {}, ""
    try:
        doc = yaml.safe_load(text) if text.strip() else {}
    except Exception:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    return doc, text


def knob_get(doc: dict, dotted: str):
    """Value at a dotted path (e.g. 'reinforce.lr'), or None if absent."""
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def knob_is_scalar(v) -> bool:
    return isinstance(v, (bool, int, float)) and not isinstance(v, (list, dict))


def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _top_level_block_span(lines: list[str], key: str) -> Optional[tuple[int, int]]:
    """(start, end) line indices of a top-level `key:` block: the `key:` line
    plus every following line until the next column-0, non-blank, non-comment
    line (or EOF). None if the key isn't a top-level block."""
    kp = re.compile(rf"^{re.escape(key)}\s*:")
    start = None
    for i, ln in enumerate(lines):
        if kp.match(ln):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "":
            continue
        # a new top-level key starts at column 0 and isn't a comment.
        if ln[:1] not in (" ", "\t") and not ln.lstrip().startswith("#"):
            end = j
            break
    return start, end


def extract_top_block(text: str, key: str) -> str:
    """The verbatim text of a top-level block, or '' if absent."""
    lines = text.split("\n")
    span = _top_level_block_span(lines, key)
    if span is None:
        return ""
    return "\n".join(lines[span[0]:span[1]])


def remove_top_block(text: str, key: str) -> str:
    """Drop a top-level block from the text (used to replace `show:`)."""
    lines = text.split("\n")
    span = _top_level_block_span(lines, key)
    if span is None:
        return text
    del lines[span[0]:span[1]]
    return "\n".join(lines)


def replace_scalar_line(text: str, leaf_key: str, value) -> tuple[str, bool]:
    """Rewrite the single `leaf_key: <scalar>` line's value, preserving its
    indentation and any inline comment. Returns (text, changed). If the key
    doesn't occur exactly once, nothing changes (guards against editing the
    wrong line when a name is ambiguous)."""
    lines = text.split("\n")
    pat = re.compile(
        rf"^(?P<indent>[ \t]*)(?P<key>{re.escape(leaf_key)})[ \t]*:"
        rf"[ \t]*(?P<val>[^#\n]*?)(?P<pad>[ \t]*)(?P<comment>#.*)?$")
    idxs = [i for i, ln in enumerate(lines) if pat.match(ln)]
    if len(idxs) != 1:
        return text, False
    i = idxs[0]
    m = pat.match(lines[i])
    comment = m.group("comment")
    tail = ("  " + comment) if comment else ""
    lines[i] = f"{m.group('indent')}{m.group('key')}: {_yaml_scalar(value)}{tail}"
    return "\n".join(lines), True


def compose_profile_text(base_text: str, solver_edits: dict,
                         show_block: dict) -> tuple[str, list[str]]:
    """The saved profile text: base file verbatim, with changed scalar solver
    lines rewritten in place and the `show:` block replaced/appended.
    Returns (text, list of solver leaf-keys that couldn't be rewritten)."""
    text = base_text if base_text.strip() else ""
    fails: list[str] = []
    for leaf, val in solver_edits.items():
        text, ok = replace_scalar_line(text, leaf, val)
        if not ok:
            fails.append(leaf)
    text = remove_top_block(text, "show")
    show_yaml = yaml.safe_dump({"show": show_block}, sort_keys=False,
                               default_flow_style=False, allow_unicode=True)
    body = text.rstrip("\n")
    text = (body + "\n\n" + show_yaml) if body else show_yaml
    if not text.endswith("\n"):
        text += "\n"
    return text, fails


# ---------------------------------------------------------------------------
# Qt (imported after the pure helpers so --list/--self-test paths that don't
# need a display still import cleanly)
# ---------------------------------------------------------------------------

from PyQt6.QtCore import Qt, QSize  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap, QIcon  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QRadioButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

_MONO = "font-family: Menlo, monospace;"


def _frame_to_pixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    fh, fw = frame.shape[0], frame.shape[1]
    img = QImage(np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
                 fw, fh, 3 * fw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        w, h, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation)


class LauncherWindow(QMainWindow):
    """The show control panel: game roster + banked-win gallery + a full,
    persistent settings editor that spawns the show non-blocking."""

    def __init__(self, catalog: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.catalog = catalog or demo_catalog.build_catalog()
        self.games = self.catalog.get("games", [])
        self._thumb_cache: dict[str, np.ndarray] = {}
        self._current_game: Optional[dict] = None
        self._replay_entry: Optional[dict] = None
        self._procs: list[subprocess.Popen] = []

        # profile-editor state
        self._roster_game: Optional[str] = None
        self._profile_path: Optional[Path] = None
        self._profile_doc: dict = {}
        self._base_text: str = ""
        self._solver_widgets: dict = {}   # dotted path -> (widget, kind, leaf, orig)
        self._solver_dirty = False
        self._loading = False

        self.setWindowTitle("NES Show Control Panel")

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_summary_bar())

        body = QHBoxLayout()
        body.addWidget(self._build_game_list(), 0)
        body.addWidget(self._build_right_panel(), 1)
        outer.addLayout(body)
        self.setCentralWidget(central)

        if self.games:
            idx = next((i for i, g in enumerate(self.games)
                        if g.get("name") == "mario"), 0)
            self.game_list.setCurrentRow(idx)

    # -- summary + roster -----------------------------------------------
    def _build_summary_bar(self) -> QWidget:
        s = catalog_summary(self.catalog)
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(bar)
        title = QLabel("Beat the Game — Live: show control panel")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 2px;")
        head = s["headline"] or "no full-clear chain banked yet"
        sub = QLabel(
            f"{s['games']} games  •  {s['wins']} banked wins on disk  •  "
            f"headline: {head}")
        sub.setStyleSheet(_MONO + "font-size: 11px; padding: 2px;")
        sub.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(sub)
        return bar

    def _build_game_list(self) -> QWidget:
        self.game_list = QListWidget()
        self.game_list.setIconSize(QSize(112, 105))
        self.game_list.setFixedWidth(240)
        self.game_list.setStyleSheet("font-size: 12px;")
        for g in self.games:
            frame = self._thumb_for(g)
            item = QListWidgetItem(QIcon(_frame_to_pixmap(frame, 112, 105)),
                                   f" {display_name(g['name'])}")
            item.setData(Qt.ItemDataRole.UserRole, g)
            self.game_list.addItem(item)
        self.game_list.currentItemChanged.connect(self._on_game_changed)
        return self.game_list

    # -- right panel (detail + settings editor) -------------------------
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)

        self.detail_title = QLabel("")
        self.detail_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 2px;")
        lay.addWidget(self.detail_title)

        top = QHBoxLayout()
        self.detail_thumb = QLabel()
        self.detail_thumb.setFixedSize(256, 240)
        self.detail_thumb.setStyleSheet("background:#111;")
        top.addWidget(self.detail_thumb, 0)

        self.detail_info = QLabel("")
        self.detail_info.setStyleSheet(_MONO + "font-size: 11px; padding: 4px;")
        self.detail_info.setWordWrap(True)
        self.detail_info.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        top.addWidget(self.detail_info, 1)
        lay.addLayout(top)

        # settings editor lives in a scroll area so the full knob set fits.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._settings_lay = QVBoxLayout(inner)
        self._settings_lay.addWidget(self._build_game_group())
        self._settings_lay.addWidget(self._build_display_group())
        self._settings_lay.addWidget(self._build_audio_group())
        self._settings_lay.addWidget(self._build_panels_group())
        self._settings_lay.addWidget(self._build_solver_group())
        self._settings_lay.addWidget(self._build_banked_group())
        self._settings_lay.addStretch(1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        lay.addWidget(self._build_profile_io())
        lay.addWidget(self._build_launch_controls())
        return panel

    # -- settings groups ------------------------------------------------
    def _hook(self, w):
        """Wire a runtime widget so any change refreshes the command line."""
        for sig in ("valueChanged", "currentTextChanged", "textChanged",
                    "toggled"):
            s = getattr(w, sig, None)
            if s is not None:
                try:
                    s.connect(self._on_runtime_changed)
                except Exception:
                    pass
                break
        return w

    def _combo(self, options, default):
        c = QComboBox()
        c.addItems(options)
        i = c.findText(str(default))
        c.setCurrentIndex(i if i >= 0 else 0)
        return self._hook(c)

    def _build_game_group(self) -> QWidget:
        box = QGroupBox("Game")
        form = QFormLayout(box)
        self.lbl_game = QLabel("—")
        self.lbl_game.setStyleSheet(_MONO + "font-size: 11px;")
        form.addRow("selected", self.lbl_game)

        state_row = QHBoxLayout()
        self.ed_state = QLineEdit()
        self.ed_state.setPlaceholderText("(profile default start state)")
        self.ed_state.textChanged.connect(self._on_runtime_changed)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_state)
        self.cb_state = QCheckBox("use selected banked entrance")
        self.cb_state.setEnabled(False)
        self.cb_state.toggled.connect(self._on_use_entrance)
        state_row.addWidget(self.ed_state, 1)
        state_row.addWidget(btn_browse)
        holder = QWidget()
        holder.setLayout(state_row)
        form.addRow("start-state --state", holder)
        form.addRow("", self.cb_state)
        return box

    def _build_display_group(self) -> QWidget:
        box = QGroupBox("Display")
        form = QFormLayout(box)
        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(1, 64)
        self._hook(self.sp_workers)
        form.addRow("workers", self.sp_workers)
        self.sp_scale = QSpinBox()
        self.sp_scale.setRange(1, 6)
        self._hook(self.sp_scale)
        form.addRow("scale", self.sp_scale)
        self.cmb_view = self._combo(["swarm", "solo"], "swarm")
        form.addRow("view", self.cmb_view)
        self.dsp_minutes = QDoubleSpinBox()
        self.dsp_minutes.setRange(0.1, 600.0)
        self.dsp_minutes.setDecimals(1)
        self.dsp_minutes.setSingleStep(5.0)
        self._hook(self.dsp_minutes)
        form.addRow("minutes-per-level", self.dsp_minutes)
        return box

    def _build_audio_group(self) -> QWidget:
        box = QGroupBox("Audio")
        form = QFormLayout(box)
        self.dsp_volume = QDoubleSpinBox()
        self.dsp_volume.setRange(0.0, 1.0)
        self.dsp_volume.setDecimals(2)
        self.dsp_volume.setSingleStep(0.05)
        self._hook(self.dsp_volume)
        form.addRow("volume", self.dsp_volume)
        self.cmb_audio = self._combo(["both", "chorus", "hero", "off"], "both")
        form.addRow("audio", self.cmb_audio)
        self.cmb_pitch = self._combo(["ff", "native"], "ff")
        form.addRow("chorus-pitch", self.cmb_pitch)
        self.sp_voices = QSpinBox()
        self.sp_voices.setRange(-1, 64)
        self.sp_voices.setSpecialValueText("all (-1)")
        self._hook(self.sp_voices)
        form.addRow("chorus-voices", self.sp_voices)
        return box

    def _build_panels_group(self) -> QWidget:
        box = QGroupBox("Show Panels")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.cb_heatmap = QCheckBox("heatmap")
        self.cb_clips = QCheckBox("clips")
        self.cb_spectator = QCheckBox("spectator-lite")
        for cb in (self.cb_heatmap, self.cb_clips, self.cb_spectator):
            cb.toggled.connect(self._on_runtime_changed)
            row.addWidget(cb)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("lanes", holder)
        self.cmb_fx = self._combo(["off", "crt"], "off")
        form.addRow("fx", self.cmb_fx)
        return box

    def _build_solver_group(self) -> QWidget:
        self.solver_box = QGroupBox("Solver (from profile)")
        self.solver_form = QFormLayout(self.solver_box)
        return self.solver_box

    def _rebuild_solver_group(self, doc: dict) -> None:
        """Populate the Solver group from the loaded profile: one editable
        widget per knob the profile actually carries (nothing invented)."""
        while self.solver_form.rowCount():
            self.solver_form.removeRow(0)
        self._solver_widgets = {}
        found = False
        for dotted, leaf, kind, rng in SOLVER_KNOBS:
            val = knob_get(doc, dotted)
            if val is None or not knob_is_scalar(val):
                continue
            found = True
            if kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
                w.toggled.connect(self._on_solver_changed)
            elif kind == "float":
                w = QDoubleSpinBox()
                w.setRange(rng[0], rng[1])
                w.setDecimals(6)
                w.setSingleStep(0.0001)
                w.setValue(float(val))
                w.valueChanged.connect(self._on_solver_changed)
            else:  # int
                w = QSpinBox()
                w.setRange(int(rng[0]), int(rng[1]))
                w.setValue(int(val))
                w.valueChanged.connect(self._on_solver_changed)
            self.solver_form.addRow(dotted, w)
            self._solver_widgets[dotted] = (w, kind, leaf, val)
        if not found:
            lbl = QLabel("(this profile exposes no editable solver scalars)")
            lbl.setStyleSheet(_MONO + "font-size: 11px; color:#999;")
            self.solver_form.addRow(lbl)

    def _build_banked_group(self) -> QWidget:
        box = QGroupBox("Banked demos — what it has already done")
        lay = QVBoxLayout(box)
        self.banked_list = QListWidget()
        self.banked_list.setStyleSheet(_MONO + "font-size: 11px;")
        self.banked_list.setMinimumHeight(120)
        self.banked_list.currentItemChanged.connect(self._on_banked_changed)
        lay.addWidget(self.banked_list)
        return box

    # -- profile I/O + launch controls ----------------------------------
    def _build_profile_io(self) -> QWidget:
        box = QGroupBox("Profile")
        lay = QVBoxLayout(box)
        self.lbl_profile = QLabel("—")
        self.lbl_profile.setStyleSheet(_MONO + "font-size: 10px; color:#68a;")
        self.lbl_profile.setWordWrap(True)
        lay.addWidget(self.lbl_profile)
        row = QHBoxLayout()
        b_load = QPushButton("Load Profile…")
        b_load.clicked.connect(self._on_load_profile)
        b_save = QPushButton("Save Profile")
        b_save.clicked.connect(self._on_save_profile)
        b_saveas = QPushButton("Save As…")
        b_saveas.clicked.connect(self._on_save_profile_as)
        for b in (b_load, b_save, b_saveas):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        return box

    def _build_launch_controls(self) -> QWidget:
        box = QGroupBox("Launch")
        lay = QVBoxLayout(box)
        mode_row = QHBoxLayout()
        self.rb_solve = QRadioButton("Live Solve")
        self.rb_replay = QRadioButton("Replay a banked win")
        self.rb_solve.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_solve)
        self.mode_group.addButton(self.rb_replay)
        self.rb_solve.toggled.connect(self._refresh_command)
        self.rb_replay.toggled.connect(self._refresh_command)
        mode_row.addWidget(self.rb_solve)
        mode_row.addWidget(self.rb_replay)
        mode_row.addStretch(1)
        self.launch_btn = QPushButton("Launch ▶")
        self.launch_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.launch_btn.clicked.connect(self._on_launch)
        mode_row.addWidget(self.launch_btn)
        lay.addLayout(mode_row)

        self.cmd_label = QLabel("")
        self.cmd_label.setStyleSheet(_MONO + "font-size: 10px; color:#5a5;"
                                     "padding: 2px;")
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.cmd_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(_MONO + "font-size: 10px; padding:2px;")
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)
        return box

    # -- thumbnails ------------------------------------------------------
    def _thumb_for(self, game: dict) -> np.ndarray:
        name = game.get("name", "?")
        cached = self._thumb_cache.get(name)
        if cached is not None:
            return cached
        rom = game.get("rom") or ""
        rom_path = str(REPO / rom) if rom and not Path(rom).is_absolute() else rom
        frame = render_thumbnail(rom_path, game.get("start_state"))
        self._thumb_cache[name] = frame
        return frame

    # -- population ------------------------------------------------------
    def _show_get(self, show: dict, key):
        v = show.get(key) if isinstance(show, dict) else None
        return v if v is not None else SHOW_DEFAULTS[key]

    def _set_combo(self, combo: QComboBox, value) -> None:
        i = combo.findText(str(value))
        combo.setCurrentIndex(i if i >= 0 else 0)

    def _populate_editor(self, doc: dict) -> None:
        """Fill every runtime + solver widget from the loaded profile (its
        `show:` block if present, else sensible show defaults)."""
        self._loading = True
        try:
            show = doc.get("show") if isinstance(doc.get("show"), dict) else {}
            self.sp_workers.setValue(int(self._show_get(show, "workers")))
            self.sp_scale.setValue(int(self._show_get(show, "scale")))
            self._set_combo(self.cmb_view, self._show_get(show, "view"))
            self.dsp_minutes.setValue(
                float(self._show_get(show, "minutes_per_level")))
            self.dsp_volume.setValue(float(self._show_get(show, "volume")))
            self._set_combo(self.cmb_audio, self._show_get(show, "audio"))
            self._set_combo(self.cmb_pitch, self._show_get(show, "chorus_pitch"))
            self.sp_voices.setValue(int(self._show_get(show, "chorus_voices")))
            self.cb_heatmap.setChecked(bool(self._show_get(show, "heatmap")))
            self.cb_clips.setChecked(bool(self._show_get(show, "clips")))
            self.cb_spectator.setChecked(
                bool(self._show_get(show, "spectator_lite")))
            self._set_combo(self.cmb_fx, self._show_get(show, "fx"))
            st = show.get("state") if isinstance(show, dict) else None
            self.ed_state.setText(str(st) if st else "")
            default_state = doc.get("start_state_path")
            self.ed_state.setPlaceholderText(
                str(default_state) if default_state else "(power-on boot)")
            self._rebuild_solver_group(doc)
        finally:
            self._loading = False
        self._solver_dirty = False

    # -- selection handlers ---------------------------------------------
    def _on_game_changed(self, current, _previous=None):
        if current is None:
            return
        try:
            game = current.data(Qt.ItemDataRole.UserRole)
            self._current_game = game
            self._roster_game = game.get("name")
            prof_rel = game.get("profile") or f"configs/{game.get('name')}.yaml"
            prof_path = Path(prof_rel)
            if not prof_path.is_absolute():
                prof_path = REPO / prof_path
            self._profile_path = prof_path
            self._profile_doc, self._base_text = parse_profile(prof_path)

            self.detail_title.setText(
                f"{display_name(game['name'])}   ({game['name']})")
            self.detail_thumb.setPixmap(
                _frame_to_pixmap(self._thumb_for(game), 256, 240))
            adapter = "yes" if game.get("has_adapter") else \
                "no (Live Solve once a solve adapter exists)"
            self.detail_info.setText(
                f"profile:     {game.get('profile')}\n"
                f"rom:         {game.get('rom')}\n"
                f"start state: {game.get('start_state') or '(power-on boot)'}\n"
                f"solve adapter: {adapter}\n"
                f"banked demos: {len(game.get('banked', []))}")
            self.lbl_game.setText(display_name(game['name']))
            self.lbl_profile.setText(f"backing: {game.get('profile')}")

            self._fill_banked(game)
            self._populate_editor(self._profile_doc)

            self._replay_entry = None
            self.rb_solve.setChecked(True)
            self.rb_replay.setEnabled(False)
            self.cb_state.setEnabled(False)
            self.cb_state.setChecked(False)
            self._refresh_command()
        except Exception as e:  # never let a bad profile crash the panel
            self.status_label.setText(f"load error: {e}")

    def _fill_banked(self, game: dict) -> None:
        self.banked_list.clear()
        banked = game.get("banked", [])
        if not banked:
            item = QListWidgetItem("no banked demos yet — Live Solve available")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.banked_list.addItem(item)
            return
        for b in banked:
            flag = "CLEARED " if b.get("cleared") else "frontier"
            depth = b.get("depth")
            depth_s = f"depth={depth}" if depth is not None else ""
            replay = "  [replayable]" if replay_target_of(b) else ""
            text = f"[{b.get('kind', '?'):8s} {flag}] {b.get('label')}"
            if depth_s:
                text += f"  ({depth_s})"
            text += replay
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, b)
            self.banked_list.addItem(item)

    def _on_banked_changed(self, current, _previous=None):
        try:
            entry = current.data(Qt.ItemDataRole.UserRole) if current else None
            self._replay_entry = entry if isinstance(entry, dict) else None
            replayable = bool(self._replay_entry
                              and replay_target_of(self._replay_entry))
            self.rb_replay.setEnabled(replayable)
            if not replayable and self.rb_replay.isChecked():
                self.rb_solve.setChecked(True)
            has_root = bool(self._replay_entry
                            and self._replay_entry.get("root_state"))
            self.cb_state.setEnabled(has_root)
            if not has_root and self.cb_state.isChecked():
                self.cb_state.setChecked(False)
            self._refresh_command()
        except Exception as e:
            self.status_label.setText(f"banked select error: {e}")

    def _on_use_entrance(self, checked: bool):
        try:
            if checked and self._replay_entry \
                    and self._replay_entry.get("root_state"):
                self.ed_state.setText(str(self._replay_entry["root_state"]))
            elif not checked:
                # only clear if it still holds the entrance path
                if self._replay_entry and \
                        self.ed_state.text() == str(
                            self._replay_entry.get("root_state")):
                    self.ed_state.setText("")
            self._refresh_command()
        except Exception as e:
            self.status_label.setText(f"entrance error: {e}")

    def _on_browse_state(self):
        try:
            start_dir = str(REPO / "roms")
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose a start-state file", start_dir,
                "State files (*.state *.bin *.state.bin);;All files (*)")
            if path:
                self.ed_state.setText(path)
        except Exception as e:
            self.status_label.setText(f"browse error: {e}")

    def _on_runtime_changed(self, *_):
        if self._loading:
            return
        self._refresh_command()

    def _on_solver_changed(self, *_):
        if self._loading:
            return
        self._solver_dirty = True
        self._refresh_command()

    # -- solver-widget reads --------------------------------------------
    def _read_solver_widget(self, w, kind):
        if kind == "bool":
            return bool(w.isChecked())
        if kind == "float":
            return float(w.value())
        return int(w.value())

    def _solver_edits(self) -> dict:
        """{leaf_key: value} for solver knobs whose widget diverges from the
        loaded profile (floats compared with tolerance to avoid spurious
        rewrites)."""
        edits = {}
        for dotted, (w, kind, leaf, orig) in self._solver_widgets.items():
            cur = self._read_solver_widget(w, kind)
            if kind == "float":
                if abs(float(cur) - float(orig)) > 1e-9:
                    edits[leaf] = cur
            elif cur != orig:
                edits[leaf] = cur
        return edits

    # -- runtime + show-block collection --------------------------------
    def _collect_runtime(self) -> dict:
        return {
            "workers": self.sp_workers.value(),
            "scale": self.sp_scale.value(),
            "view": self.cmb_view.currentText(),
            "minutes_per_level": round(self.dsp_minutes.value(), 3),
            "volume": round(self.dsp_volume.value(), 3),
            "audio": self.cmb_audio.currentText(),
            "chorus_pitch": self.cmb_pitch.currentText(),
            "chorus_voices": self.sp_voices.value(),
            "heatmap": self.cb_heatmap.isChecked(),
            "clips": self.cb_clips.isChecked(),
            "spectator_lite": self.cb_spectator.isChecked(),
            "fx": self.cmb_fx.currentText(),
        }

    def _collect_show_block(self) -> dict:
        rt = self._collect_runtime()
        block = {"game": self._roster_game or ""}
        block.update(rt)
        st = self.ed_state.text().strip()
        if st:
            block["state"] = st
        return block

    # -- command building ------------------------------------------------
    def _launch_target(self) -> tuple[Optional[str], Optional[str], bool]:
        """(game, profile, needs_write). Use --game for a clean roster launch
        (byte-identical to the base show); otherwise --profile a file that
        reflects the editor's solver edits, writing a scratch working copy
        when those edits aren't yet on disk."""
        game = self._roster_game
        canonical = (CONFIGS / f"{game}.yaml") if game else None
        prof = self._profile_path
        clean_roster = (
            game and canonical and prof
            and prof.resolve() == canonical.resolve()
            and not self._solver_dirty)
        if clean_roster:
            return game, None, False
        if self._solver_dirty or prof is None:
            return None, str(WORKING_PROFILE), True
        return None, str(prof), False

    def _build_argv(self, game, profile) -> Optional[list[str]]:
        if not self._current_game and game is None and profile is None:
            return None
        replay_mode = self.rb_replay.isChecked() and self.rb_replay.isEnabled()
        mode = "replay" if replay_mode else "solve"
        replay = replay_target_of(self._replay_entry) if replay_mode else None
        st = self.ed_state.text().strip()
        state = st or None
        rt = self._collect_runtime()
        return build_launch_argv(
            PYTHON, game=game, profile=profile, mode=mode,
            state=(None if replay_mode else state), replay=replay,
            workers=rt["workers"], scale=rt["scale"], volume=rt["volume"],
            view=rt["view"], minutes_per_level=rt["minutes_per_level"],
            audio=rt["audio"], chorus_pitch=rt["chorus_pitch"],
            chorus_voices=rt["chorus_voices"], heatmap=rt["heatmap"],
            clips=rt["clips"], fx=rt["fx"],
            spectator_lite=rt["spectator_lite"])

    def _refresh_command(self, *_):
        try:
            game, profile, _needs = self._launch_target()
            argv = self._build_argv(game, profile)
            self.cmd_label.setText(shlex.join(argv) if argv else "")
            if self._solver_dirty:
                self.status_label.setText(
                    "solver knobs edited — Launch writes a working profile so "
                    "they apply (Save to keep them).")
        except Exception as e:
            self.cmd_label.setText(f"(command error: {e})")

    # -- profile save / load --------------------------------------------
    def _compose_text(self) -> tuple[str, list[str]]:
        base = self._base_text
        if not base.strip():
            try:
                base = yaml.safe_dump(self._profile_doc, sort_keys=False,
                                      default_flow_style=False)
            except Exception:
                base = ""
        return compose_profile_text(base, self._solver_edits(),
                                    self._collect_show_block())

    def _save_profile_to(self, path: Path, adopt: bool = True) -> list[str]:
        """Write the current editor state to `path`. When adopt, the written
        file becomes the editor's backing profile (edits are no longer
        pending). Returns solver leaf-keys that couldn't be written."""
        text, fails = self._compose_text()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if adopt:
            self._profile_path = path
            self._base_text = text
            self._profile_doc, _ = parse_profile(path)
            # solver widgets now match disk
            new = {}
            for dotted, (w, kind, leaf, _orig) in self._solver_widgets.items():
                new[dotted] = (w, kind, leaf, self._read_solver_widget(w, kind))
            self._solver_widgets = new
            self._solver_dirty = False
        return fails

    def _on_save_profile(self):
        try:
            if not self._profile_path:
                return self._on_save_profile_as()
            fails = self._save_profile_to(self._profile_path)
            self._after_save(self._profile_path, fails)
        except Exception as e:
            self.status_label.setText(f"save failed: {e}")

    def _on_save_profile_as(self):
        try:
            default = str((CONFIGS / f"{self._roster_game or 'profile'}.yaml"))
            path, _ = QFileDialog.getSaveFileName(
                self, "Save profile as", default, "YAML (*.yaml *.yml)")
            if not path:
                return
            if not path.endswith((".yaml", ".yml")):
                path += ".yaml"
            fails = self._save_profile_to(Path(path))
            self.lbl_profile.setText(f"backing: {path}")
            self._after_save(Path(path), fails)
        except Exception as e:
            self.status_label.setText(f"save-as failed: {e}")

    def _after_save(self, path: Path, fails: list[str]):
        msg = f"saved {path} (show block written)"
        if fails:
            msg += f"  — could not rewrite solver keys: {', '.join(fails)}"
        self.status_label.setText(msg)
        self._refresh_command()

    def _on_load_profile(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load profile", str(CONFIGS), "YAML (*.yaml *.yml)")
            if not path:
                return
            p = Path(path)
            doc, text = parse_profile(p)
            self._profile_path = p
            self._profile_doc = doc
            self._base_text = text
            self._roster_game = p.stem if p.parent.resolve() == \
                CONFIGS.resolve() else None
            self.lbl_profile.setText(f"backing: {path}")
            self.lbl_game.setText(doc.get("name") or p.stem)
            self._populate_editor(doc)
            self._refresh_command()
            self.status_label.setText(f"loaded {path}")
        except Exception as e:
            self.status_label.setText(f"load failed: {e}")

    # -- launch ----------------------------------------------------------
    def _on_launch(self):
        try:
            game, profile, needs_write = self._launch_target()
            if needs_write:
                try:
                    self._save_profile_to(WORKING_PROFILE, adopt=False)
                    profile = str(WORKING_PROFILE)
                    game = None
                except Exception as e:
                    self.status_label.setText(
                        f"working-profile write failed ({e}); launching roster "
                        f"config without unsaved solver edits")
                    game, profile = self._roster_game, None
            argv = self._build_argv(game, profile)
            if not argv:
                return
            self.cmd_label.setText(shlex.join(argv))
            proc = subprocess.Popen(argv, cwd=str(REPO))
            self._procs.append(proc)
            self.status_label.setText(
                f"launched pid {proc.pid} — panel stays open for the next")
        except Exception as e:  # pragma: no cover - spawn failure surface
            self.status_label.setText(f"launch failed: {e}")

    # -- test hook -------------------------------------------------------
    def select_game(self, name: str) -> bool:
        for i in range(self.game_list.count()):
            g = self.game_list.item(i).data(Qt.ItemDataRole.UserRole)
            if isinstance(g, dict) and g.get("name") == name:
                self.game_list.setCurrentRow(i)
                return True
        return False


# ---------------------------------------------------------------------------
# self-test (offscreen; no windowed show is spawned)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print("[self-test] building catalog ...")
    catalog = demo_catalog.build_catalog()
    games = catalog.get("games", [])
    assert games, "catalog produced no games"
    summary = catalog_summary(catalog)
    print(f"[self-test] catalog: {summary['games']} games, "
          f"{summary['wins']} banked wins")
    print(f"[self-test] headline: {summary['headline']}")
    assert summary["games"] == 17, summary["games"]

    print("[self-test] rendering a thumbnail ...")
    contra = next((g for g in games if g["name"] == "contra"), games[0])
    rom = str(REPO / contra["rom"])
    frame = render_thumbnail(rom, contra.get("start_state"))
    assert frame.shape == (FRAME_H, FRAME_W, 3), frame.shape

    print("[self-test] constructing control panel (offscreen) ...")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = LauncherWindow(catalog=catalog)
    assert win.game_list.count() == len(games), "game list not populated"

    # -- editor reflects each profile's frame_skip + solve knobs ---------
    assert win.select_game("contra"), "contra not in roster"
    fs = win._solver_widgets.get("frame_skip")
    assert fs is not None and fs[0].value() == 4, "contra frame_skip not 4"
    kkl = win._solver_widgets.get("solve.kill_key_local")
    assert kkl is not None and kkl[0].isChecked() is True, \
        "contra kill_key_local not reflected"
    print("[self-test] contra editor: frame_skip=4, kill_key_local=True, "
          f"solver knobs surfaced={sorted(win._solver_widgets)}")

    assert win.select_game("kirby"), "kirby not in roster"
    fsk = win._solver_widgets.get("frame_skip")
    assert fsk is not None and fsk[0].value() == 4, "kirby frame_skip not 4"
    pc = win._solver_widgets.get("solve.progress_cap")
    assert pc is not None and pc[0].value() == 4000, \
        "kirby progress_cap not 4000"
    print("[self-test] kirby editor: frame_skip=4, progress_cap=4000, "
          f"solver knobs surfaced={sorted(win._solver_widgets)}")

    # -- Save/Load round-trip (show block persists; solve: byte-preserved) -
    print("[self-test] Save/Load round-trip on contra ...")
    win.select_game("contra")
    contra_text = (CONFIGS / "contra.yaml").read_text()
    solve_block = extract_top_block(contra_text, "solve")
    assert solve_block, "could not extract contra solve: block"
    win.sp_workers.setValue(8)
    win.cb_heatmap.setChecked(True)
    out = Path("/tmp/contra_streamtest.yaml")
    fails = win._save_profile_to(out, adopt=False)
    assert not fails, f"solver-key write failures: {fails}"
    saved_text = out.read_text()
    reloaded = yaml.safe_load(saved_text)
    assert isinstance(reloaded.get("show"), dict), "no show block persisted"
    assert reloaded["show"]["workers"] == 8, reloaded["show"].get("workers")
    assert reloaded["show"]["heatmap"] is True, reloaded["show"].get("heatmap")
    assert solve_block in saved_text, "solve: block was NOT byte-preserved"
    # and the structural sections survived untouched
    assert reloaded.get("solve") == yaml.safe_load(contra_text).get("solve")
    assert reloaded.get("reinforce") == \
        yaml.safe_load(contra_text).get("reinforce")
    print(f"[self-test] round-trip OK: {out} — show.workers=8, "
          "show.heatmap=True, solve:/reinforce: preserved")

    # -- argv for a configured game --------------------------------------
    contra_argv = build_launch_argv(
        PYTHON, game="contra", mode="solve", workers=8, scale=4,
        volume=0.8, view="solo", minutes_per_level=45.0, audio="hero",
        chorus_pitch="native", chorus_voices=6, heatmap=True, fx="crt",
        spectator_lite=True)
    print("\n[self-test] argv (configured contra):")
    print("  " + shlex.join(contra_argv))
    for tok in ("--game", "contra", "--workers", "8", "--scale", "4",
                "--volume", "0.8", "--view", "solo", "--minutes-per-level",
                "45.0", "--audio", "hero", "--chorus-pitch", "native",
                "--chorus-voices", "6", "--heatmap", "--fx", "crt",
                "--spectator-lite"):
        assert tok in contra_argv, f"missing {tok} in argv"
    # default-valued knobs stay off the command line
    default_argv = build_launch_argv(PYTHON, game="contra", mode="solve",
                                     workers=12, scale=3, volume=0.6,
                                     view="swarm", audio="both")
    assert "--volume" not in default_argv and "--audio" not in default_argv, \
        "default runtime knobs should be omitted"

    # -- banked SMB replay still works -----------------------------------
    mario = next((g for g in games if g["name"] == "mario"), None)
    assert mario is not None, "no SMB game in catalog"
    replay_entry = next(
        (b for b in mario["banked"] if replay_target_of(b)), None)
    assert replay_entry is not None, "no replayable SMB banked win found"
    target = replay_target_of(replay_entry)
    replay_argv = build_launch_argv(PYTHON, game="mario", mode="replay",
                                    replay=target)
    print("\n[self-test] argv (banked SMB replay):")
    print(f"  banked entry: {replay_entry['label']}")
    print("  " + shlex.join(replay_argv))
    assert "--mode" in replay_argv and "replay" in replay_argv
    assert "--replay" in replay_argv and target in replay_argv

    out_png = REPO / "runs" / "launcher_thumb_sample.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pm = _frame_to_pixmap(frame, FRAME_W * 2, FRAME_H * 2)
    ok = pm.save(str(out_png), "PNG")
    print(f"[self-test] thumbnail sample saved: {out_png} (ok={ok})")

    print("\n[self-test] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="print the demo catalog and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="offscreen construction + argv/round-trip smoke, then "
                         "exit (no windowed show is spawned)")
    args = ap.parse_args()

    if args.list:
        demo_catalog._pretty_print(demo_catalog.build_catalog())
        return 0
    if args.self_test:
        return _self_test()

    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.resize(1180, 860)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
