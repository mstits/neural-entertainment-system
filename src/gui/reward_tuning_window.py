"""
Live reward-weight tuning panel.

Auto-generates a slider per reward_weights entry in the active game
profile. Each slider change is debounced and pushed onto an in-process
Queue the trainer drains at every generation boundary. No restart needed —
the reward function gets rebuilt with the new weights on the next episode.

This is the shortest path from "my agent is exploiting motion reward" to
"I dropped motion_weight 10x and see the curve bend" without losing the
population you've already trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# Range heuristics per reward-weight key name. We scale the slider to the
# magnitude of the initial value so mouse precision is useful whether the
# weight is 0.0005 (motion) or 50.0 (triforce). Penalties flip the sign.
def _slider_scale_for(name: str, current: float) -> tuple[float, float, float]:
    """Return (min, max, single_step_size) for a reward-weight slider."""
    mag = max(abs(current), 1.0)
    if "penalty" in name or current < 0:
        return (-mag * 5.0, 0.0, mag / 100.0)
    if "bonus" in name or "piece" in name or "heart_container" in name:
        return (0.0, mag * 5.0, mag / 100.0)
    return (0.0, mag * 10.0, max(mag / 100.0, 1e-4))


# Slider int encoding scale. Qt sliders use 32-bit ints (~2.1B max), and
# we encode floats by multiplying. The fixed 10000× scale used previously
# overflowed for any weight whose stretched range exceeded ~21,400. We
# pick the largest scale that keeps the entire range within int32 for
# every slider, so very large reward weights (e.g. user-tuned
# stage_cleared bonuses in the thousands) don't truncate.
def _slider_int_scale(lo: float, hi: float) -> int:
    span = max(abs(lo), abs(hi), 1.0)
    # Reserve a 10× safety margin under int32 max so cumulative rounding
    # in setRange / setValue can never cross.
    max_scale = (2_000_000_000 // int(span)) if span > 0 else 10_000
    return max(1, min(10_000, max_scale))


class RewardTuningWindow(QMainWindow):
    """Live slider panel for reward_weights keys in the active profile."""

    def __init__(
        self,
        initial_weights: dict,
        on_change: Callable[[dict], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NES — Live Reward Tuning")
        self.resize(520, 480)

        self._on_change = on_change
        self._spinboxes: dict[str, QDoubleSpinBox] = {}
        self._sliders: dict[str, QSlider] = {}
        # Per-slider int encoding scale. See _slider_int_scale: large
        # weights need a smaller scale so int(value * scale) doesn't
        # overflow Qt's int32 slider value.
        self._slider_scales: dict[str, int] = {}

        # Debounce repeated slider changes into one push per 300ms.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._emit_changes)

        self._pending: dict[str, float] = {}
        self._build_ui(initial_weights)

    def _build_ui(self, weights: dict) -> None:
        central = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        root = QVBoxLayout(central)
        help_label = QLabel(
            "Sliders push to the live trainer on change.\n"
            "Changes take effect on the next episode reset."
        )
        help_label.setStyleSheet(
            "color: #ccc; background: #222; padding: 6px; font-family: Menlo"
        )
        root.addWidget(help_label)

        grid = QGridLayout()
        for row, (name, value) in enumerate(sorted(weights.items())):
            try:
                current = float(value)
            except (TypeError, ValueError):
                continue
            lo, hi, step = _slider_scale_for(name, current)

            grid.addWidget(QLabel(name), row, 0)

            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(current)
            spin.valueChanged.connect(
                lambda v, n=name: self._on_spin_changed(n, v)
            )
            grid.addWidget(spin, row, 1)

            # Slider uses integer values internally; encoding scale is
            # adaptive so the int value stays within int32 even for very
            # large reward weights.
            scale = _slider_int_scale(lo, hi)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(lo * scale), int(hi * scale))
            slider.setValue(int(current * scale))
            slider.valueChanged.connect(
                lambda v, n=name: self._on_slider_changed(n, v)
            )
            grid.addWidget(slider, row, 2)

            self._spinboxes[name] = spin
            self._sliders[name] = slider
            self._slider_scales[name] = scale

        root.addLayout(grid)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8f8; font-family: Menlo")
        root.addWidget(self.status_label)

        reset_btn = QPushButton("Reset to profile defaults")
        reset_btn.clicked.connect(lambda: self._reset_to(weights))
        root.addWidget(reset_btn)

    def _on_spin_changed(self, name: str, value: float) -> None:
        slider = self._sliders.get(name)
        scale = self._slider_scales.get(name, 10000)
        if slider and slider.value() != int(value * scale):
            slider.blockSignals(True)
            slider.setValue(int(value * scale))
            slider.blockSignals(False)
        self._pending[name] = value
        self._debounce.start()

    def _on_slider_changed(self, name: str, value: int) -> None:
        spin = self._spinboxes.get(name)
        scale = self._slider_scales.get(name, 10000)
        float_val = value / scale
        if spin and abs(spin.value() - float_val) > 1e-6:
            spin.blockSignals(True)
            spin.setValue(float_val)
            spin.blockSignals(False)
        self._pending[name] = float_val
        self._debounce.start()

    def _emit_changes(self) -> None:
        if not self._pending:
            return
        updates = dict(self._pending)
        self._pending.clear()
        # Catch on_change failures (queue closed, callback errored) so a
        # pending debounce that fires after the window has been torn
        # down doesn't leak an exception out of the QTimer callback and
        # silently kill the timer.
        try:
            self._on_change(updates)
        except Exception:
            return
        try:
            self.status_label.setText(
                f"sent: " + ", ".join(f"{k}={v:g}" for k, v in updates.items())
            )
        except RuntimeError:
            # Status label was deleted (window closed mid-debounce); fine.
            pass

    def closeEvent(self, event) -> None:
        # Drop any pending debounce so a tick that arrives after we're
        # closed doesn't try to push to a now-dead queue.
        try:
            if self._debounce is not None:
                self._debounce.stop()
        except Exception:
            pass
        self._pending.clear()
        super().closeEvent(event)

    def _reset_to(self, defaults: dict) -> None:
        for name, spin in self._spinboxes.items():
            if name in defaults:
                try:
                    spin.setValue(float(defaults[name]))
                except (TypeError, ValueError):
                    continue
