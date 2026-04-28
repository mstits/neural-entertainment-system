"""
Standalone GUI widget-construction test.

Run outside pytest — construction of QMainWindow under pytest's stdio
redirection aborts on macOS + PyQt6 6.11.

Usage:
    QT_QPA_PLATFORM=offscreen python tests/gui_selftest.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication


def run() -> None:
    # Make `src.` imports resolve when run from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    app = QApplication.instance() or QApplication(sys.argv)

    from src.gui.emulator_grid import EmulatorGrid
    from src.gui.main_window import MainWindow
    from src.gui.metrics_window import MetricsWindow

    mw = MainWindow()
    mw.update_metrics({
        "generation": 3,
        "best_fitness": 12.5,
        "avg_fitness": 8.0,
        "stage": "stage_1",
        "success_rate": 0.4,
        "episodes": 50,
    })
    assert "Generation" in mw.metrics_label.text()

    grid = EmulatorGrid(num_instances=4)
    grid.set_frame_provider(
        lambda: [np.full((240, 256, 3), i * 60, dtype=np.uint8) for i in range(4)]
    )
    grid._refresh_frames()
    assert len(grid.frame_labels) == 4

    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "metrics.jsonl"
        log.write_text(
            '{"generation": 0, "best_fitness": 1.0, "avg_fitness": 0.5, "success_rate": 0.1}\n'
        )
        metrics = MetricsWindow(metrics_path=str(log))
        metrics._refresh()
        assert metrics._generations == [0]

    print("GUI selftest OK")


if __name__ == "__main__":
    run()
