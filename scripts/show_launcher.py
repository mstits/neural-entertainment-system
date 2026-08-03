#!/usr/bin/env python3
"""THE DEMO CONSOLE — a launcher for the live show.

The front door to everything the search system can do and has already done.
It reads the on-disk catalog (scripts/demo_catalog.build_catalog) and, in one
window, lets you:

  - browse every configured game (short name + a live-rendered start-state
    thumbnail) on the left;
  - inspect the selected game's start state and its BANKED demos on the right
    -- the verified SMB power-on -> 8-4 full clear, the Castlevania block
    chain, each game's Go-Explore bootstrap frontier -- as a selectable
    gallery of "what it has done", every entry read off a real receipt;
  - pick a MODE (Live Solve = the search plays it live now, victory lap on
    each clear; Replay = lap a banked solution in the hero cam, no swarm) and
    LAUNCH the show as a subprocess, so the launcher stays open to fire off
    the next one.

Nothing here beats a game itself: it composes the argv and spawns
scripts/live_solve_show.py, which is the working show. The launcher never
changes the show's default behavior.

Usage:
    make launcher
    python scripts/show_launcher.py
    python scripts/show_launcher.py --list        # print the catalog and exit
    QT_QPA_PLATFORM=offscreen python scripts/show_launcher.py --self-test
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402
from scripts import demo_catalog  # noqa: E402

SHOW_SCRIPT = REPO / "scripts" / "live_solve_show.py"
# The interpreter running the launcher is the venv python under `make
# launcher`; the show inherits it so the same nes_core .so is used.
PYTHON = sys.executable or "python"

FRAME_W, FRAME_H = 256, 240


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


def build_launch_argv(python_exe: str, *, game: Optional[str] = None,
                      profile: Optional[str] = None, mode: str = "solve",
                      state: Optional[str] = None,
                      replay: Optional[str] = None,
                      workers: Optional[int] = None,
                      scale: Optional[int] = None,
                      heatmap: bool = False, clips: bool = False,
                      fx: str = "off",
                      spectator_lite: bool = False) -> list[str]:
    """Compose the live_solve_show.py argv for a launch.

    --game is preferred over --profile (a convenience the show already
    resolves to configs/<name>.yaml). Mode is always explicit; --replay is
    only emitted in replay mode; the show-lane toggles are omitted when off
    so the invocation stays byte-identical to the base show unless a lane is
    turned on."""
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
# Qt (imported after the pure helpers so --list/--self-test paths that don't
# need a display still import cleanly)
# ---------------------------------------------------------------------------

from PyQt6.QtCore import Qt, QSize  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap, QIcon  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QButtonGroup, QCheckBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout, QWidget)

_MONO = "font-family: Menlo, monospace;"


def _frame_to_pixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    fh, fw = frame.shape[0], frame.shape[1]
    img = QImage(np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
                 fw, fh, 3 * fw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        w, h, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation)


class LauncherWindow(QMainWindow):
    """The demo console: game roster + banked-win gallery + launch controls."""

    def __init__(self, catalog: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.catalog = catalog or demo_catalog.build_catalog()
        self.games = self.catalog.get("games", [])
        self._thumb_cache: dict[str, np.ndarray] = {}
        self._current_game: Optional[dict] = None
        self._replay_entry: Optional[dict] = None
        self._procs: list[subprocess.Popen] = []

        self.setWindowTitle("NES Demo Console — Launcher")

        central = QWidget()
        outer = QVBoxLayout(central)

        outer.addWidget(self._build_summary_bar())

        body = QHBoxLayout()
        body.addWidget(self._build_game_list(), 0)
        body.addWidget(self._build_detail_panel(), 1)
        outer.addLayout(body)

        self.setCentralWidget(central)

        if self.games:
            # Head-line SMB first if present, else the first game.
            idx = next((i for i, g in enumerate(self.games)
                        if g.get("name") == "mario"), 0)
            self.game_list.setCurrentRow(idx)

    # -- widgets ---------------------------------------------------------
    def _build_summary_bar(self) -> QWidget:
        s = catalog_summary(self.catalog)
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(bar)
        title = QLabel("Beat the Game — Live: demo console")
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

    def _build_detail_panel(self) -> QWidget:
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

        lay.addWidget(QLabel("Banked demos — what it has already done:"))
        self.banked_list = QListWidget()
        self.banked_list.setStyleSheet(_MONO + "font-size: 11px;")
        self.banked_list.setMinimumHeight(150)
        self.banked_list.currentItemChanged.connect(self._on_banked_changed)
        lay.addWidget(self.banked_list, 1)

        lay.addWidget(self._build_controls())
        return panel

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Launch")
        lay = QVBoxLayout(box)

        # mode toggle
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
        lay.addLayout(mode_row)

        # workers / scale
        num_row = QHBoxLayout()
        num_row.addWidget(QLabel("workers"))
        self.sp_workers = QSpinBox()
        self.sp_workers.setRange(1, 64)
        self.sp_workers.setValue(12)
        self.sp_workers.valueChanged.connect(self._refresh_command)
        num_row.addWidget(self.sp_workers)
        num_row.addSpacing(12)
        num_row.addWidget(QLabel("scale"))
        self.sp_scale = QSpinBox()
        self.sp_scale.setRange(1, 6)
        self.sp_scale.setValue(3)
        self.sp_scale.valueChanged.connect(self._refresh_command)
        num_row.addWidget(self.sp_scale)
        num_row.addStretch(1)
        lay.addLayout(num_row)

        # show-lane toggles
        tog_row = QHBoxLayout()
        self.cb_heatmap = QCheckBox("heatmap")
        self.cb_clips = QCheckBox("clips")
        self.cb_fx = QCheckBox("CRT fx")
        self.cb_spectator = QCheckBox("spectator-lite")
        self.cb_state = QCheckBox("solve from selected entrance")
        self.cb_state.setEnabled(False)
        for cb in (self.cb_heatmap, self.cb_clips, self.cb_fx,
                   self.cb_spectator, self.cb_state):
            cb.toggled.connect(self._refresh_command)
            tog_row.addWidget(cb)
        tog_row.addStretch(1)
        lay.addLayout(tog_row)

        # launch + command line
        launch_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch ▶")
        self.launch_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.launch_btn.clicked.connect(self._on_launch)
        launch_row.addWidget(self.launch_btn)
        launch_row.addStretch(1)
        lay.addLayout(launch_row)

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

    # -- selection handlers ---------------------------------------------
    def _on_game_changed(self, current, _previous=None):
        if current is None:
            return
        game = current.data(Qt.ItemDataRole.UserRole)
        self._current_game = game
        self.detail_title.setText(
            f"{display_name(game['name'])}   ({game['name']})")
        self.detail_thumb.setPixmap(
            _frame_to_pixmap(self._thumb_for(game), 256, 240))

        adapter = "yes" if game.get("has_adapter") else "no (Live Solve only "\
            "once a solve adapter exists)"
        self.detail_info.setText(
            f"profile:     {game.get('profile')}\n"
            f"rom:         {game.get('rom')}\n"
            f"start state: {game.get('start_state') or '(SMB power-on boot)'}\n"
            f"solve adapter: {adapter}\n"
            f"banked demos: {len(game.get('banked', []))}")

        self.banked_list.clear()
        banked = game.get("banked", [])
        if not banked:
            item = QListWidgetItem(
                "no banked demos yet — Live Solve available")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.banked_list.addItem(item)
        else:
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

        # reset mode to Live Solve until a replayable banked entry is picked
        self._replay_entry = None
        self.rb_solve.setChecked(True)
        self.rb_replay.setEnabled(False)
        self.cb_state.setEnabled(False)
        self.cb_state.setChecked(False)
        self._refresh_command()

    def _on_banked_changed(self, current, _previous=None):
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
        if not has_root:
            self.cb_state.setChecked(False)
        self._refresh_command()

    # -- command building ------------------------------------------------
    def _current_argv(self) -> Optional[list[str]]:
        game = self._current_game
        if not game:
            return None
        replay_mode = self.rb_replay.isChecked() and self.rb_replay.isEnabled()
        mode = "replay" if replay_mode else "solve"
        replay = replay_target_of(self._replay_entry) if replay_mode else None
        state = None
        if (not replay_mode and self.cb_state.isChecked()
                and self._replay_entry and self._replay_entry.get("root_state")):
            state = self._replay_entry["root_state"]
        return build_launch_argv(
            PYTHON, game=game["name"], mode=mode, state=state, replay=replay,
            workers=self.sp_workers.value(), scale=self.sp_scale.value(),
            heatmap=self.cb_heatmap.isChecked(), clips=self.cb_clips.isChecked(),
            fx="crt" if self.cb_fx.isChecked() else "off",
            spectator_lite=self.cb_spectator.isChecked())

    def _refresh_command(self, *_):
        argv = self._current_argv()
        if argv:
            self.cmd_label.setText(shlex.join(argv))
        else:
            self.cmd_label.setText("")

    # -- launch ----------------------------------------------------------
    def _on_launch(self):
        argv = self._current_argv()
        if not argv:
            return
        try:
            proc = subprocess.Popen(argv, cwd=str(REPO))
            self._procs.append(proc)
            self.status_label.setText(
                f"launched pid {proc.pid} — launcher stays open for the next")
        except Exception as e:  # pragma: no cover - spawn failure surface
            self.status_label.setText(f"launch failed: {e}")


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

    print("[self-test] rendering a thumbnail ...")
    contra = next((g for g in games if g["name"] == "contra"), games[0])
    rom = str(REPO / contra["rom"])
    frame = render_thumbnail(rom, contra.get("start_state"))
    assert frame.shape == (FRAME_H, FRAME_W, 3), frame.shape

    print("[self-test] constructing launcher window (offscreen) ...")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = LauncherWindow(catalog=catalog)
    assert win.game_list.count() == len(games), "game list not populated"

    out = REPO / "runs" / "launcher_thumb_sample.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pm = _frame_to_pixmap(frame, FRAME_W * 2, FRAME_H * 2)
    ok = pm.save(str(out), "PNG")
    print(f"[self-test] thumbnail sample saved: {out} (ok={ok})")

    contra_argv = build_launch_argv(PYTHON, game="contra", mode="solve",
                                    workers=12, scale=3)
    print("\n[self-test] argv (game=contra, mode=solve):")
    print("  " + shlex.join(contra_argv))
    assert "--game" in contra_argv and "contra" in contra_argv
    assert "--mode" in contra_argv and "solve" in contra_argv

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

    print("\n[self-test] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="print the demo catalog and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="offscreen construction + argv smoke, then exit "
                         "(no windowed show is spawned)")
    args = ap.parse_args()

    if args.list:
        demo_catalog._pretty_print(demo_catalog.build_catalog())
        return 0
    if args.self_test:
        return _self_test()

    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.resize(1040, 720)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
