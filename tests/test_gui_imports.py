"""
GUI import-surface test.

We do NOT instantiate QMainWindow under pytest: on macOS + PyQt6 6.11,
constructing any QWidget while pytest has redirected stdio causes an
immediate SIGABRT inside Qt's platform plugin. Widget construction is
covered instead by tests/gui_selftest.py, which is a standalone script
you run outside pytest.

This pytest-friendly test just confirms the modules import and the static
attributes we depend on exist.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_module_imports() -> None:
    from src.gui import main_window

    assert hasattr(main_window, "MainWindow")
    assert hasattr(main_window, "CONFIGS_DIR")


def test_emulator_grid_module_imports() -> None:
    from src.gui import emulator_grid

    assert hasattr(emulator_grid, "EmulatorGrid")
    assert emulator_grid.NES_WIDTH == 256
    assert emulator_grid.NES_HEIGHT == 240


def test_metrics_window_module_imports() -> None:
    from src.gui import metrics_window

    assert hasattr(metrics_window, "MetricsWindow")


def test_main_module_imports() -> None:
    from src.gui import main

    assert hasattr(main, "AppController")
    assert hasattr(main, "_run_trainer")
