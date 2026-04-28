"""Wrapped QFileDialog calls that remember the user's last-picked
directory per logical key, so the next launch lands at the right
place instead of `~/`.

Backed by `QSettings` (organization=`NeuralEntertainmentSystem`, app=`gui`),
which on macOS persists to `~/Library/Preferences/com.nes-training-
lab.gui.plist`. Keys are scoped per dialog purpose (`rom`, `bc_demo`,
`start_state`, `reward_overrides`, `checkpoint`, `play_save_state`,
`play_save_bc_tape`) so each picker has its own memory and the user
isn't bounced across unrelated trees just because they last touched
a different one.

If no setting is stored yet, falls back to `default_dir` (the workspace
default — typically `roms/`, `samples/`, `configs/`, or `checkpoints/`),
and to `~` only if that doesn't exist either. After every successful
pick, the parent directory of the chosen file is written back so the
next call resumes from the right place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog, QWidget


_SETTINGS_ORG = "NeuralEntertainmentSystem"
_SETTINGS_APP = "gui"
_KEY_PREFIX = "last_dir/"


def _settings() -> QSettings:
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def _resolve_start_dir(key: str, default_dir: Optional[str]) -> str:
    """Resolve the directory the dialog should open at.

    Order: stored setting (if the path still exists) → caller-provided
    default_dir (if it exists) → user's home dir.
    """
    stored = _settings().value(f"{_KEY_PREFIX}{key}", "", type=str)
    if stored and Path(stored).exists():
        return stored
    if default_dir and Path(default_dir).exists():
        return default_dir
    return str(Path.home())


def _remember(key: str, picked_path: str) -> None:
    """Persist the parent directory of a successful pick."""
    if picked_path:
        _settings().setValue(f"{_KEY_PREFIX}{key}", str(Path(picked_path).parent))


def remembered_open(
    parent: QWidget,
    key: str,
    title: str,
    file_filter: str,
    default_dir: Optional[str] = None,
) -> str:
    """Open-file dialog that remembers its last directory.

    Returns the picked path, or empty string on cancel.
    """
    start_dir = _resolve_start_dir(key, default_dir)
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir, file_filter)
    _remember(key, path)
    return path


def remembered_save(
    parent: QWidget,
    key: str,
    title: str,
    default_filename: str,
    file_filter: str,
    default_dir: Optional[str] = None,
) -> str:
    """Save-file dialog that remembers its last directory.

    `default_filename` is the suggested filename (without path); the
    dialog opens with `<remembered_dir>/<default_filename>` selected.
    Returns the picked path, or empty string on cancel.
    """
    start_dir = _resolve_start_dir(key, default_dir)
    suggestion = str(Path(start_dir) / default_filename)
    path, _ = QFileDialog.getSaveFileName(parent, title, suggestion, file_filter)
    _remember(key, path)
    return path
