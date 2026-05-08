"""Unified training dashboard — single observer view for a live run.

Displays everything you need to "watch learning happen" without bouncing
between MetricsWindow, RewardTuningWindow, and the highlights folder:

* Header: ROM, profile, generation, run elapsed, current stage,
  best/avg fitness, episodes-in-stage, success rate.
* Fitness plot: best vs avg cumulative reward per generation.
* Reward signal breakdown: per-signal contribution per gen, color-coded.
* PPO learning telemetry: loss, policy loss, value loss, entropy —
  a quick read on whether the network is actually updating.
* Depth + success rate plot: are we getting deeper / clearing the stage?
* Recent depth records: lex-greater per-game depth events (room/level/x).
* Recent highlights: MP4 captures the highlight recorder wrote.

Reads `metrics.jsonl`, `depth_memo.jsonl`, and `highlights/` directly —
no extra trainer-side queues. Refreshes once a second.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


# Per-signal colors. Named entries are stable across runs; unknown
# signals get a deterministic color derived from the signal name (hash
# → HSV rotation) so every reward shaper renders in its own color
# without anyone having to extend this dict.
_SIGNAL_COLORS = {
    # Mario
    "forward":           "#8bc34a",
    "air":               "#03a9f4",
    "checkpoint":        "#ffeb3b",
    "completion":        "#ffd700",
    "score":             "#ff5252",
    "death":             "#3f0000",
    "pit_fall":          "#bf360c",
    "enemy_death":       "#d84315",
    "survival":          "#26a69a",
    "jump_clear":        "#7e57c2",
    "time_penalty":      "#555555",
    # Zelda
    "exploration":       "#4caf50",
    "dungeon_enter":     "#00bcd4",
    "new_item":          "#ff9800",
    "first_sword":       "#ffeb3b",
    "weapon_upgrade":    "#ffc107",
    "magic_key":         "#e91e63",
    "map":               "#9c27b0",
    "compass":           "#673ab7",
    "heart_container":   "#f44336",
    "heart_recovery":    "#ff5722",
    "triforce":          "#ffd700",
    "rupee":             "#cddc39",
    "enemy_kill":        "#2196f3",
    "item_used":         "#03a9f4",
    "item_cycle":        "#9e9e9e",
    "motion":            "#607d8b",
    "wall_bump":         "#795548",
    "stagnation":        "#5d4037",
    "damage":            "#bf360c",
    # Other games
    "boss_killed":       "#ff1744",
    "stage_cleared":     "#00e676",
    "y_progress":        "#9ccc65",
}


def _color_for_signal(signal: str) -> str:
    """Stable color for any signal name, including unknown ones.

    Known signals come from the curated palette above. Unknowns get a
    hue rotated by the signal-name hash through a golden-ratio step so
    adjacent unknown signals don't collide. Saturation/value held
    constant so colors look like they belong to the same family.
    """
    if signal in _SIGNAL_COLORS:
        return _SIGNAL_COLORS[signal]
    # Stable hash → hue. Python's str hash is salted across runs, so
    # use a content-based fold instead so the same signal renders the
    # same color across sessions.
    h = 0
    for ch in signal:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = (h * 0.61803398875) % 1.0  # golden-ratio rotation
    # HSV → RGB at S=0.65, V=0.85.
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _fmt_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_depth_key(key) -> str:
    """Render a depth key as a short tuple-like string."""
    try:
        return ".".join(str(int(x)) for x in key)
    except Exception:
        return str(key)


class TrainingDashboardWindow(QMainWindow):
    """Single-pane observer view. Open one per run."""

    def __init__(
        self,
        metrics_path: str | Path,
        depth_memo_path: str | Path | None = None,
        highlights_dir: str | Path = "highlights",
        reconstruction_path: str | Path | None = None,
        rom_label: str = "",
        profile_label: str = "",
        instances: int = 0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("NES — Training Dashboard")
        self.resize(1280, 820)

        self.metrics_path = Path(metrics_path)
        # Default depth memo lives next to metrics.jsonl in the
        # checkpoint dir. Tolerant of None — feed simply stays empty.
        if depth_memo_path is None:
            depth_memo_path = self.metrics_path.parent / "depth_memo.jsonl"
        self.depth_memo_path = Path(depth_memo_path)
        self.highlights_dir = Path(highlights_dir)
        # Dreamer dumps a `(2, 4, 84, 84)` uint8 array of orig+recon
        # frames here. Default location matches DreamerTrainer's
        # `dreamer_ckpt_dir / reconstruction.npy`.
        if reconstruction_path is None:
            reconstruction_path = (
                self.metrics_path.parent / "dreamer" / "reconstruction.npy"
            )
        self.reconstruction_path = Path(reconstruction_path)
        self._recon_mtime: float = 0.0

        self._rom_label = rom_label
        self._profile_label = profile_label
        self._instances = instances
        self._start_wall = time.time()

        self._build_ui()

        # Per-gen series. Lists grow with metrics file length.
        self._gens: list[int] = []
        self._best: list[float] = []
        self._avg: list[float] = []
        self._success: list[float] = []
        self._depth_scalar: list[float] = []
        self._ppo_loss: list[float] = []
        self._ppo_policy: list[float] = []
        self._ppo_value: list[float] = []
        self._ppo_entropy: list[float] = []
        # World-model loss series (Dreamer only). All-NaN in PPO mode.
        self._wm_total: list[float] = []
        self._wm_recon: list[float] = []
        self._wm_kl: list[float] = []
        self._wm_reward: list[float] = []
        self._wm_continue: list[float] = []
        self._reward_signals: dict[str, list[float]] = {}
        # Per-signal generation tracking — each signal gets the SUBSET of
        # generation numbers where its reward fired, so plotting stays
        # aligned even when a signal stops firing mid-run.
        self._signal_gens: dict[str, list[int]] = {}
        self._signal_curves: dict[str, pg.PlotDataItem] = {}

        self._depth_seen: set[str] = set()
        self._highlight_seen: set[str] = set()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        # First paint immediately so the user doesn't see a blank pane
        # for the first second after opening the window.
        self._refresh()

    def closeEvent(self, event) -> None:
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        # Drop the sigResized connection so the curric_plot's right
        # viewbox doesn't try to fire into us after we're destroyed.
        try:
            sig = getattr(self, "_curric_resized_signal", None)
            if sig is not None:
                sig.disconnect(self._sync_success_vb)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)

    # ----- layout --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        self.setCentralWidget(central)

        outer.addWidget(self._build_header())

        # Plots in a 2x2 grid, side feeds in a right pane.
        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._build_plot_grid())
        body.addWidget(self._build_side_pane())
        # Plots:side ≈ 3:1. Side gets ~320px at default 1280 width which
        # is wide enough for a short genome name + depth tuple without
        # eliding mid-word.
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)
        outer.addWidget(body, stretch=1)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        h = QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(18)

        # Left: identity (ROM, profile, instances).
        left = QVBoxLayout()
        rom = QLabel(self._rom_label or "(no ROM)")
        rom_font = QFont()
        rom_font.setPointSize(13)
        rom_font.setBold(True)
        rom.setFont(rom_font)
        left.addWidget(rom)
        sub = QLabel(
            f"profile: {self._profile_label or '-'}  •  workers: {self._instances or '-'}"
        )
        sub.setStyleSheet("color: palette(placeholder-text)")
        left.addWidget(sub)
        h.addLayout(left)

        h.addStretch(1)

        # Right: live counters. Built once; refreshed in _refresh.
        self._lbl_gen = self._make_kpi("Gen", "-")
        self._lbl_elapsed = self._make_kpi("Elapsed", "0s")
        self._lbl_stage = self._make_kpi("Stage", "-")
        self._lbl_best = self._make_kpi("Best", "-")
        self._lbl_avg = self._make_kpi("Avg", "-")
        self._lbl_episodes = self._make_kpi("Episodes", "-")
        self._lbl_success = self._make_kpi("Success", "-")
        for w in (
            self._lbl_gen, self._lbl_elapsed, self._lbl_stage,
            self._lbl_best, self._lbl_avg,
            self._lbl_episodes, self._lbl_success,
        ):
            h.addWidget(w)
        return frame

    def _make_kpi(self, label: str, initial: str) -> QWidget:
        wrap = QFrame()
        wrap.setFrameShape(QFrame.Shape.NoFrame)
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        cap = QLabel(label.upper())
        cap.setStyleSheet(
            "color: palette(placeholder-text); font-size: 9pt; letter-spacing: 1px"
        )
        cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        val = QLabel(initial)
        val_font = QFont("Menlo")
        val_font.setPointSize(13)
        val_font.setBold(True)
        val.setFont(val_font)
        val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(cap)
        v.addWidget(val)
        # Stash the value label for later access without poking at children.
        wrap._value = val  # type: ignore[attr-defined]
        return wrap

    @staticmethod
    def _set_kpi(widget: QWidget, text: str) -> None:
        try:
            widget._value.setText(text)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _build_plot_grid(self) -> QWidget:
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        self.fitness_plot = pg.PlotWidget(title="Fitness (best / avg)")
        self.fitness_plot.setLabel("bottom", "Generation")
        self.fitness_plot.setLabel("left", "Cumulative reward")
        self.fitness_plot.addLegend()
        self.fitness_plot.showGrid(x=True, y=True, alpha=0.2)
        self.best_curve = self.fitness_plot.plot(
            [], [], pen=pg.mkPen("#4caf50", width=2), name="best"
        )
        self.avg_curve = self.fitness_plot.plot(
            [], [], pen=pg.mkPen("#2196f3", width=2), name="avg"
        )
        grid.addWidget(self.fitness_plot, 0, 0)

        self.signal_plot = pg.PlotWidget(
            title="Reward signal breakdown (per-genome avg)"
        )
        self.signal_plot.setLabel("bottom", "Generation")
        self.signal_plot.setLabel("left", "Reward contribution")
        self.signal_plot.addLegend(offset=(10, 10))
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)
        grid.addWidget(self.signal_plot, 0, 1)

        self.ppo_plot = pg.PlotWidget(title="PPO learning telemetry")
        self.ppo_plot.setLabel("bottom", "Generation")
        self.ppo_plot.setLabel("left", "Loss / Entropy")
        self.ppo_plot.addLegend()
        self.ppo_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ppo_loss_curve = self.ppo_plot.plot(
            [], [], pen=pg.mkPen("#e91e63", width=2), name="total"
        )
        self.ppo_policy_curve = self.ppo_plot.plot(
            [], [], pen=pg.mkPen("#ff9800", width=1, style=Qt.PenStyle.DashLine),
            name="policy",
        )
        self.ppo_value_curve = self.ppo_plot.plot(
            [], [], pen=pg.mkPen("#03a9f4", width=1, style=Qt.PenStyle.DashLine),
            name="value",
        )
        self.ppo_entropy_curve = self.ppo_plot.plot(
            [], [], pen=pg.mkPen("#cddc39", width=2), name="entropy"
        )
        grid.addWidget(self.ppo_plot, 1, 0)

        # Depth + success share an x-axis but have different y-scales.
        # Use a dual-axis ViewBox so they overlay legibly.
        self.curric_plot = pg.PlotWidget(title="Depth & curriculum success")
        self.curric_plot.setLabel("bottom", "Generation")
        self.curric_plot.setLabel("left", "Depth (game-specific)")
        self.curric_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curric_plot.addLegend()
        self.depth_curve = self.curric_plot.plot(
            [], [], pen=pg.mkPen("#ffd700", width=2), name="depth"
        )
        # Right-axis success curve.
        self._success_vb = pg.ViewBox()
        self.curric_plot.scene().addItem(self._success_vb)
        right_axis = self.curric_plot.getAxis("right")
        right_axis.linkToView(self._success_vb)
        self.curric_plot.showAxis("right")
        self.curric_plot.setLabel("right", "Success rate")
        self._success_vb.setXLink(self.curric_plot)
        self._success_vb.setYRange(0.0, 1.0)
        self.success_curve = pg.PlotDataItem(
            pen=pg.mkPen("#ff5722", width=2, style=Qt.PenStyle.DashLine),
        )
        self._success_vb.addItem(self.success_curve)
        # Keep the right viewbox glued to the main plot's geometry.
        # Stash the signal so closeEvent can disconnect — without that,
        # closing and reopening the dashboard accumulates connections to
        # destroyed widgets, which raise RuntimeError on the next emit.
        self._curric_resized_signal = self.curric_plot.getViewBox().sigResized
        self._curric_resized_signal.connect(self._sync_success_vb)
        grid.addWidget(self.curric_plot, 1, 1)

        # World-model losses panel. Only meaningful in Dreamer training
        # mode (the metrics keys are written by DreamerTrainer); empty
        # otherwise. Stays in the layout so the grid doesn't reflow
        # when modes switch.
        self.wm_plot = pg.PlotWidget(title="World model losses (Dreamer only)")
        self.wm_plot.setLabel("bottom", "Generation")
        self.wm_plot.setLabel("left", "Loss")
        self.wm_plot.addLegend()
        self.wm_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wm_total_curve = self.wm_plot.plot(
            [], [], pen=pg.mkPen("#e91e63", width=2), name="total"
        )
        self.wm_recon_curve = self.wm_plot.plot(
            [], [], pen=pg.mkPen("#4caf50", width=1, style=Qt.PenStyle.DashLine),
            name="recon",
        )
        self.wm_kl_curve = self.wm_plot.plot(
            [], [], pen=pg.mkPen("#9c27b0", width=1, style=Qt.PenStyle.DashLine),
            name="kl",
        )
        self.wm_reward_curve = self.wm_plot.plot(
            [], [], pen=pg.mkPen("#ff9800", width=1, style=Qt.PenStyle.DashLine),
            name="reward",
        )
        self.wm_continue_curve = self.wm_plot.plot(
            [], [], pen=pg.mkPen("#03a9f4", width=1, style=Qt.PenStyle.DashLine),
            name="continue",
        )
        # Spans both columns of the bottom row — the curves are useful
        # at full width and there's no second WM-related plot to pair
        # against.
        grid.addWidget(self.wm_plot, 2, 0, 1, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return grid_widget

    def _sync_success_vb(self) -> None:
        # Keep the dashed success-rate viewbox aligned with the main
        # depth plot's coordinate system (else it won't pan/zoom together).
        self._success_vb.setGeometry(self.curric_plot.getViewBox().sceneBoundingRect())
        self._success_vb.linkedViewChanged(
            self.curric_plot.getViewBox(), self._success_vb.XAxis
        )

    def _build_side_pane(self) -> QWidget:
        side = QFrame()
        side.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(side)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # Replay buffer fill — Dreamer-only; in PPO mode the metric is
        # absent and the bar stays at 0/0. Useful primary signal for
        # whether the trainer has accumulated enough experience to
        # start learning (Dreamer's `warmup_steps` gate).
        replay_lbl = QLabel("Replay buffer")
        replay_lbl.setStyleSheet("font-weight: bold")
        v.addWidget(replay_lbl)
        self.replay_bar = QProgressBar()
        self.replay_bar.setRange(0, 100)
        self.replay_bar.setValue(0)
        self.replay_bar.setFormat("%p% (waiting)")
        self.replay_bar.setStyleSheet("QProgressBar { font-family: Menlo; font-size: 10pt }")
        v.addWidget(self.replay_bar)

        # Reconstruction preview — the latest world-model orig vs recon
        # 4-frame strip the trainer dumps every 10 train steps. Greyscale,
        # upscaled 2x for legibility. Empty in PPO mode (no .npy file).
        recon_lbl = QLabel("World-model recon (orig / recon)")
        recon_lbl.setStyleSheet("font-weight: bold; margin-top: 6px")
        v.addWidget(recon_lbl)
        self.recon_image_label = QLabel()
        self.recon_image_label.setMinimumHeight(120)
        self.recon_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recon_image_label.setText("(no reconstruction yet)")
        self.recon_image_label.setStyleSheet(
            "color: palette(placeholder-text); font-style: italic;"
            " border: 1px solid palette(mid)"
        )
        v.addWidget(self.recon_image_label)

        depth_lbl = QLabel("Depth records")
        depth_lbl.setStyleSheet("font-weight: bold; margin-top: 6px")
        v.addWidget(depth_lbl)
        self.depth_list = QListWidget()
        self.depth_list.setStyleSheet("font-family: Menlo; font-size: 10pt")
        # Items longer than the row width get an ellipsis instead of
        # being truncated mid-character — keeps the pane readable when
        # the user drags the splitter narrow.
        self.depth_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        v.addWidget(self.depth_list, stretch=2)

        hl_lbl = QLabel("Recent highlights")
        hl_lbl.setStyleSheet("font-weight: bold; margin-top: 6px")
        v.addWidget(hl_lbl)
        self.highlights_list = QListWidget()
        self.highlights_list.setStyleSheet("font-family: Menlo; font-size: 10pt")
        self.highlights_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        v.addWidget(self.highlights_list, stretch=1)
        return side

    # ----- refresh -------------------------------------------------------

    def _refresh(self) -> None:
        self._reload_metrics()
        self._reload_depth_memo()
        self._reload_highlights()
        self._reload_reconstruction()
        self._update_header()

    def _reload_reconstruction(self) -> None:
        """Load the latest world-model reconstruction strip if the
        trainer has updated the .npy file since we last rendered.

        The file is `(2, 4, 84, 84)` uint8 — 4 columns of frames, top
        row originals and bottom row reconstructions. We compose them
        into a single greyscale QImage at 2× upscale (336×168) and set
        it on the side-pane label.
        """
        if not self.reconstruction_path.exists():
            return
        try:
            mtime = self.reconstruction_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._recon_mtime:
            return
        try:
            arr = np.load(str(self.reconstruction_path))
        except (OSError, ValueError):
            return
        if arr.shape != (2, 4, 84, 84) or arr.dtype != np.uint8:
            return
        self._recon_mtime = mtime

        # Compose orig (top row) and recon (bottom row) into a 2x4 grid.
        # 2× upscale via numpy broadcasting (cheap, no library needed).
        upscale = 2
        rows = []
        for r in range(2):
            row = np.concatenate(list(arr[r]), axis=1)  # (84, 4*84)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)  # (2*84, 4*84)
        grid_up = grid.repeat(upscale, axis=0).repeat(upscale, axis=1)
        h, w = grid_up.shape
        # ascontiguousarray so QImage's stride math works cleanly.
        grid_up = np.ascontiguousarray(grid_up)
        img = QImage(grid_up.data, w, h, w, QImage.Format.Format_Grayscale8)
        # QPixmap takes ownership of a copy internally — safe to drop
        # `grid_up` after this line.
        self.recon_image_label.setPixmap(QPixmap.fromImage(img))
        self.recon_image_label.setText("")
        self.recon_image_label.setStyleSheet(
            "border: 1px solid palette(mid)"
        )

    def _reload_metrics(self) -> None:
        if not self.metrics_path.exists():
            return

        # Always rebuild from scratch. metrics.jsonl is reset on every
        # run start (trainer truncates it), so a tail-style follow
        # would risk reading partial lines mid-write. The file is
        # short (one row per gen, well under MB-scale).
        self._gens.clear()
        self._best.clear()
        self._avg.clear()
        self._success.clear()
        self._depth_scalar.clear()
        self._ppo_loss.clear()
        self._ppo_policy.clear()
        self._ppo_value.clear()
        self._ppo_entropy.clear()
        self._wm_total.clear()
        self._wm_recon.clear()
        self._wm_kl.clear()
        self._wm_reward.clear()
        self._wm_continue.clear()
        self._reward_signals.clear()
        self._signal_gens.clear()
        self._latest_metric: dict = {}

        try:
            with open(self.metrics_path, "r") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._latest_metric = m
                    self._gens.append(int(m.get("generation", 0)))
                    self._best.append(float(m.get("best_fitness", 0.0)))
                    self._avg.append(float(m.get("avg_fitness", 0.0)))
                    self._success.append(float(m.get("success_rate", 0.0)))
                    self._depth_scalar.append(float(m.get("depth_scalar", 0.0)))
                    # PPO stats may be absent on gens that didn't run a
                    # gradient step. Use NaN so the curve breaks rather
                    # than dropping to zero (which looks like "loss=0",
                    # i.e. converged — the opposite of what's true).
                    self._ppo_loss.append(float(m.get("ppo_loss", float("nan"))))
                    self._ppo_policy.append(
                        float(m.get("ppo_policy_loss", float("nan")))
                    )
                    self._ppo_value.append(float(m.get("ppo_value_loss", float("nan"))))
                    self._ppo_entropy.append(
                        float(m.get("ppo_entropy", float("nan")))
                    )
                    self._wm_total.append(float(m.get("wm_total", float("nan"))))
                    self._wm_recon.append(float(m.get("wm_recon", float("nan"))))
                    self._wm_kl.append(float(m.get("wm_kl", float("nan"))))
                    self._wm_reward.append(float(m.get("wm_reward", float("nan"))))
                    self._wm_continue.append(
                        float(m.get("wm_continue", float("nan")))
                    )
                    for k, v in m.items():
                        if isinstance(k, str) and k.startswith("reward_"):
                            sig = k[len("reward_"):]
                            self._reward_signals.setdefault(sig, []).append(float(v))
                            # Per-signal x-axis tracking. Without this, the
                            # later "pad-prepend zeros" logic silently
                            # misaligned signals that STOPPED firing mid-run
                            # (the missing samples are at the END, not the
                            # start; prepending zeros shifts visible data
                            # rightward by N gens).
                            self._signal_gens.setdefault(sig, []).append(
                                int(m.get("generation", 0))
                            )
        except OSError:
            return

        # Repaint plots.
        self.best_curve.setData(self._gens, self._best)
        self.avg_curve.setData(self._gens, self._avg)
        self.success_curve.setData(self._gens, self._success)
        self.depth_curve.setData(self._gens, self._depth_scalar)
        self.ppo_loss_curve.setData(self._gens, self._ppo_loss)
        self.ppo_policy_curve.setData(self._gens, self._ppo_policy)
        self.ppo_value_curve.setData(self._gens, self._ppo_value)
        self.ppo_entropy_curve.setData(self._gens, self._ppo_entropy)
        self.wm_total_curve.setData(self._gens, self._wm_total)
        self.wm_recon_curve.setData(self._gens, self._wm_recon)
        self.wm_kl_curve.setData(self._gens, self._wm_kl)
        self.wm_reward_curve.setData(self._gens, self._wm_reward)
        self.wm_continue_curve.setData(self._gens, self._wm_continue)

        # Reward signal stack — sort by abs-contribution so the legend
        # leads with the signals that actually move the needle.
        ordered = sorted(
            self._reward_signals.items(),
            key=lambda kv: sum(abs(x) for x in kv[1]),
            reverse=True,
        )
        for sig, vals in ordered:
            color = _color_for_signal(sig)
            curve = self._signal_curves.get(sig)
            xs = self._signal_gens.get(sig, self._gens)
            if curve is None:
                curve = self.signal_plot.plot(
                    xs, vals,
                    pen=pg.mkPen(color, width=2), name=sig,
                )
                self._signal_curves[sig] = curve
            else:
                curve.setData(xs, vals)

    def _reload_depth_memo(self) -> None:
        if not self.depth_memo_path.exists():
            return
        try:
            with open(self.depth_memo_path, "r") as f:
                lines = f.readlines()
        except OSError:
            return
        # Show newest first; the file is append-only and short
        # (one row per all-time depth record, typically dozens to
        # low-hundreds across an overnight run).
        for line in lines:
            line = line.strip()
            if not line or line in self._depth_seen:
                continue
            self._depth_seen.add(line)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _fmt_depth_key(rec.get("key", ()))
            gen = rec.get("generation", "?")
            name = rec.get("genome_name", "?")
            text = f"gen {gen:>4} · {key:<20} · {name}"
            self.depth_list.insertItem(0, QListWidgetItem(text))
        # Cap the list so an overnight run doesn't grow it unbounded.
        while self.depth_list.count() > 200:
            self.depth_list.takeItem(self.depth_list.count() - 1)

    def _reload_highlights(self) -> None:
        if not self.highlights_dir.exists():
            return
        try:
            files = sorted(
                self.highlights_dir.glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for p in files[:50]:
            name = p.name
            if name in self._highlight_seen:
                continue
            self._highlight_seen.add(name)
            self.highlights_list.insertItem(0, QListWidgetItem(name))
        while self.highlights_list.count() > 50:
            self.highlights_list.takeItem(self.highlights_list.count() - 1)

    def _update_header(self) -> None:
        m = getattr(self, "_latest_metric", {}) or {}
        self._set_kpi(self._lbl_gen, str(m.get("generation", "-")))
        self._set_kpi(self._lbl_elapsed, _fmt_elapsed(time.time() - self._start_wall))
        self._set_kpi(self._lbl_stage, str(m.get("stage", "-")))
        bf = m.get("best_fitness")
        af = m.get("avg_fitness")
        self._set_kpi(
            self._lbl_best,
            f"{bf:.1f}" if isinstance(bf, (int, float)) else "-",
        )
        self._set_kpi(
            self._lbl_avg,
            f"{af:.1f}" if isinstance(af, (int, float)) else "-",
        )
        self._set_kpi(self._lbl_episodes, str(m.get("episodes", "-")))
        sr = m.get("success_rate")
        self._set_kpi(
            self._lbl_success,
            f"{sr:.1%}" if isinstance(sr, (int, float)) else "-",
        )
        # Replay-buffer fill (Dreamer only). Read both `replay_size`
        # and the configured capacity if present; fall back to a soft
        # max derived from observed values otherwise.
        replay_size = m.get("replay_size")
        replay_cap = m.get("replay_capacity")
        if isinstance(replay_size, (int, float)) and isinstance(
            replay_cap, (int, float)
        ):
            cap = max(1, int(replay_cap))
            pct = min(100, int(100 * float(replay_size) / cap))
            self.replay_bar.setValue(pct)
            self.replay_bar.setFormat(f"{int(replay_size):,} / {int(cap):,}")
        elif isinstance(replay_size, (int, float)):
            # Older Dreamer checkpoints predate the capacity emit; show
            # the absolute size with no percent.
            self.replay_bar.setValue(0)
            self.replay_bar.setFormat(f"{int(replay_size):,} steps")
        else:
            self.replay_bar.setValue(0)
            self.replay_bar.setFormat("(PPO mode — N/A)")
