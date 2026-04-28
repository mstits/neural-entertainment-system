"""Live training metrics window.

Uses pyqtgraph for fast updates. Reads the trainer's JSONL metrics log every
second and repaints best/avg fitness, curriculum success-rate, and a
per-signal reward breakdown so the user can see WHICH rewards are firing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget


# Deterministic colors per reward signal so plots stay consistent across
# refreshes. Color palette chosen for dark backgrounds.
_SIGNAL_COLORS = {
    "exploration":       "#4caf50",  # green — primary signal
    "dungeon_enter":     "#00bcd4",  # cyan
    "new_item":          "#ff9800",  # orange
    "item_tier_upgrade": "#ffc107",  # amber
    "magic_key":         "#e91e63",  # pink
    "map":               "#9c27b0",  # purple
    "compass":           "#673ab7",  # deep purple
    "heart_container":   "#f44336",  # red
    "heart_recovery":    "#ff5722",  # deep orange
    "triforce":          "#ffd700",  # gold
    "rupee":             "#cddc39",  # lime
    "enemy_kill":        "#2196f3",  # blue
    "item_used":         "#03a9f4",  # light blue
    "item_cycle":        "#9e9e9e",  # gray
    "motion":            "#607d8b",  # blue gray
    "time_penalty":      "#555555",  # dark gray
    "damage":            "#bf360c",  # dark red
    "death":             "#3f0000",  # darker red
}


class MetricsWindow(QMainWindow):
    def __init__(self, metrics_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("NES — Training Metrics")
        self.resize(1000, 720)

        self.metrics_path = Path(metrics_path)
        self._build_ui()

        self._generations: list[int] = []
        self._best: list[float] = []
        self._avg: list[float] = []
        self._success: list[float] = []
        # {signal_name: list_of_values_per_generation}
        self._reward_signals: dict[str, list[float]] = {}
        self._signal_curves: dict[str, pg.PlotDataItem] = {}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)

    def closeEvent(self, event) -> None:
        # Stop the refresh timer so a pending tick can't fire into
        # destroyed pyqtgraph widgets after the window closes.
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.fitness_plot = pg.PlotWidget(title="Fitness (best / avg)")
        self.fitness_plot.setLabel("bottom", "Generation")
        self.fitness_plot.setLabel("left", "Cumulative reward")
        self.fitness_plot.addLegend()
        self.best_curve = self.fitness_plot.plot(
            [], [], pen=pg.mkPen("#4caf50", width=2), name="best"
        )
        self.avg_curve = self.fitness_plot.plot(
            [], [], pen=pg.mkPen("#2196f3", width=2), name="avg"
        )
        layout.addWidget(self.fitness_plot)

        self.signal_plot = pg.PlotWidget(
            title="Reward signal breakdown (avg per-genome contribution per gen)"
        )
        self.signal_plot.setLabel("bottom", "Generation")
        self.signal_plot.setLabel("left", "Reward contribution")
        self.signal_plot.addLegend(offset=(10, 10))
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.signal_plot)

        self.success_plot = pg.PlotWidget(title="Curriculum stage success rate")
        self.success_plot.setLabel("bottom", "Generation")
        self.success_plot.setLabel("left", "Success rate")
        self.success_plot.setYRange(0.0, 1.0)
        self.success_curve = self.success_plot.plot(
            [], [], pen=pg.mkPen("#ff9800", width=2)
        )
        layout.addWidget(self.success_plot)

    def _refresh(self) -> None:
        if not self.metrics_path.exists():
            return

        self._generations.clear()
        self._best.clear()
        self._avg.clear()
        self._success.clear()
        self._reward_signals.clear()

        with open(self.metrics_path, "r") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._generations.append(m.get("generation", 0))
                self._best.append(m.get("best_fitness", 0.0))
                self._avg.append(m.get("avg_fitness", 0.0))
                self._success.append(m.get("success_rate", 0.0))
                # Collect every reward_* key so new signals added to
                # the reward function surface automatically.
                for k, v in m.items():
                    if isinstance(k, str) and k.startswith("reward_"):
                        signal = k[len("reward_"):]
                        self._reward_signals.setdefault(signal, []).append(float(v))

        # Pad missing rows so short curves align on x-axis.
        n = len(self._generations)
        for sig, values in self._reward_signals.items():
            if len(values) < n:
                values[:0] = [0.0] * (n - len(values))

        self.best_curve.setData(self._generations, self._best)
        self.avg_curve.setData(self._generations, self._avg)
        self.success_curve.setData(self._generations, self._success)

        # Draw/update each signal's curve. Create on first appearance,
        # update thereafter. Order the plot by most-contribution-first
        # so legend is useful at a glance.
        total_abs = sorted(
            self._reward_signals.items(),
            key=lambda kv: sum(abs(x) for x in kv[1]),
            reverse=True,
        )
        for signal, values in total_abs:
            color = _SIGNAL_COLORS.get(signal, "#aaaaaa")
            curve = self._signal_curves.get(signal)
            if curve is None:
                curve = self.signal_plot.plot(
                    self._generations, values,
                    pen=pg.mkPen(color, width=2), name=signal,
                )
                self._signal_curves[signal] = curve
            else:
                curve.setData(self._generations, values)
