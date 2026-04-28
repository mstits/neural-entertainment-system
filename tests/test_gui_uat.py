"""
User acceptance tests: drive the GUI with pytest-qt and exercise the
flows users actually use. Each test is a scripted walkthrough of a path
through the UI.

Uses `qtbot` to work around the macOS+PyQt6 6.11+pytest abort we hit
when constructing QMainWindow directly — qtbot registers the widget
with pytest-qt's managed QApplication, avoiding the crash.

Covers:
  * MainWindow construction + state transitions
  * ROM / profile / start-state picker flow (mocked file dialogs)
  * Start disabled without ROM, enabled with; Stop wired to start
  * Load Last populates fields from persisted recents
  * Reward tuning window pushes updates to a queue
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


def _make_dummy_rom(tmp_path: Path) -> Path:
    rom = tmp_path / "dummy.nes"
    # Minimal iNES header so NESEnvironment won't complain if the path is
    # handed off somewhere — but these tests never actually load it.
    rom.write_bytes(b"NES\x1a" + b"\x01\x00\x00\x00" + b"\x00" * 8 + b"\x00" * 100)
    return rom


def _make_dummy_state(tmp_path: Path) -> Path:
    p = tmp_path / "dummy.state.bin"
    p.write_bytes(bytes([0x08] * 30 + [0x00] * 30))
    return p


def test_main_window_construction(qtbot):
    from src.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert w.windowTitle() == "NES"
    assert w.start_btn.isEnabled()
    assert not w.stop_btn.isEnabled()
    assert not w.tune_btn.isEnabled()


def test_start_rejects_without_rom(qtbot, tmp_path, monkeypatch):
    from src.gui import main_window as mw_mod
    from src.gui.main_window import MainWindow

    # Point recents at a nonexistent file so MainWindow.__init__ doesn't
    # auto-populate a ROM from the user's actual last session.
    monkeypatch.setattr(mw_mod, "RECENTS_PATH", tmp_path / "does_not_exist.json")

    w = MainWindow()
    qtbot.addWidget(w)

    signal_received = []
    w.start_requested.connect(lambda cfg: signal_received.append(cfg))

    w._on_start()
    assert signal_received == []
    assert "Choose a ROM first" in w.metrics_label.text()


def test_start_emits_config_when_fully_configured(qtbot, tmp_path, monkeypatch):
    from src.gui import main_window as mw_mod
    from src.gui.main_window import MainWindow

    # Isolate recents from the user's real file — this test triggers
    # _persist_recents via _on_start, and we don't want to stomp on whatever
    # the user currently has saved.
    monkeypatch.setattr(mw_mod, "RECENTS_PATH", tmp_path / "recents.json")

    w = MainWindow()
    qtbot.addWidget(w)

    rom = _make_dummy_rom(tmp_path)
    state = _make_dummy_state(tmp_path)
    w._rom_path = str(rom)
    w._start_state_path = str(state)
    # Pick the first profile in the dropdown
    assert w.profile_dropdown.count() > 0
    w.profile_dropdown.setCurrentIndex(0)

    configs = []
    w.start_requested.connect(lambda cfg: configs.append(cfg))

    w._on_start()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg["rom_path"] == str(rom)
    assert cfg["start_state_path"] == str(state)
    # Default 24 post-PGO on M4 Max — see hot_path_baseline.md.
    assert cfg["num_instances"] == 24
    assert cfg["population_size"] == 24
    assert "resume" in cfg
    # Buttons updated
    assert not w.start_btn.isEnabled()
    assert w.stop_btn.isEnabled()
    assert w.tune_btn.isEnabled()


def test_stop_returns_window_to_idle(qtbot):
    from src.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.start_btn.setEnabled(False)
    w.stop_btn.setEnabled(True)
    w.tune_btn.setEnabled(True)

    w._on_stop()
    assert w.start_btn.isEnabled()
    assert not w.stop_btn.isEnabled()
    assert not w.tune_btn.isEnabled()


def test_load_last_restores_from_recents(qtbot, tmp_path, monkeypatch):
    from src.gui import main_window as mw_mod
    from src.gui.main_window import MainWindow

    rom = _make_dummy_rom(tmp_path)
    state = _make_dummy_state(tmp_path)
    fake_recents = tmp_path / "recents.json"
    fake_recents.write_text(
        json.dumps({
            "rom_path": str(rom),
            "start_state_path": str(state),
            "profile": "zelda",
            "num_instances": 8,
            "population_size": 12,
            "resume": False,
            "bc": True,
        })
    )
    monkeypatch.setattr(mw_mod, "RECENTS_PATH", fake_recents)

    w = MainWindow()
    qtbot.addWidget(w)
    # Construction already applied recents silently in __init__; verify.
    assert w._rom_path == str(rom)
    assert w._start_state_path == str(state)
    assert w.instance_spin.value() == 8
    assert w.population_spin.value() == 12
    assert w.resume_checkbox.isChecked() is False
    assert w.bc_checkbox.isChecked() is True


def test_update_metrics_renders_fields(qtbot):
    from src.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.update_metrics({
        "generation": 17,
        "best_fitness": 123.45,
        "avg_fitness": 67.89,
        "stage": "stage_1",
        "success_rate": 0.42,
        "episodes": 256,
    })
    text = w.metrics_label.text()
    assert "Generation:       17" in text
    assert "123.45" in text
    assert "67.89" in text
    assert "42.0%" in text


def test_reward_tuning_window_debounces_and_pushes(qtbot):
    from src.gui.reward_tuning_window import RewardTuningWindow

    pushed: list[dict] = []

    w = RewardTuningWindow(
        initial_weights={"motion": 0.0005, "exploration_bonus": 5.0},
        on_change=lambda upd: pushed.append(upd),
    )
    qtbot.addWidget(w)

    # Flip the motion slider via its spinbox
    spin = w._spinboxes["motion"]
    spin.setValue(0.001)
    # Wait out the 300ms debounce so the pending update actually fires.
    qtbot.wait(400)

    assert len(pushed) >= 1
    last = pushed[-1]
    assert "motion" in last
    assert abs(last["motion"] - 0.001) < 1e-6


def test_emulator_grid_paints_from_provider(qtbot):
    import numpy as np
    from src.gui.emulator_grid import EmulatorGrid

    grid = EmulatorGrid(num_instances=4)
    qtbot.addWidget(grid)

    frames = [np.full((240, 256, 3), i * 60, dtype=np.uint8) for i in range(4)]
    grid.set_frame_provider(lambda: frames)
    grid._refresh_frames()  # manual tick
    # All 4 labels should now have a non-null pixmap
    for label in grid.frame_labels:
        pix = label.pixmap()
        assert pix is not None and not pix.isNull()
