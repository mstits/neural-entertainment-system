"""
Live grid of emulator views — watch all N instances play simultaneously.

The trainer thread writes each worker's latest frame into a shared
buffer on every step_all. A QTimer here polls at ~30 Hz via the
frame-provider callback and repaints the grid.

Rendering:
  * Window is freely resizable. Tiles auto-scale and keep NES aspect
    ratio so Zelda doesn't stretch into widescreen.
  * "Sharp bilinear" upscaling by default: integer nearest-neighbor
    pre-scale to the largest multiple that fits in the tile, then
    bilinear to fill the residual. Preserves pixel-art character and
    avoids mushy edges at large sizes.
  * Skip-on-unchanged: we cache the last frame ptr per tile; identical
    frame skips the QImage + QPixmap work entirely. At 16 tiles this
    is a meaningful main-thread cost reduction when the trainer is
    between steps.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QWidget,
)


NES_WIDTH = 256
NES_HEIGHT = 240


def _wrap_text(text: str, fm, max_px: int, max_lines: int = 2) -> list[str]:
    """Greedy word-wrap for caption overlays. Returns up to `max_lines`
    lines each no wider than `max_px` (uses QFontMetrics)."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = w if not current else current + " " + w
        if fm.horizontalAdvance(candidate) <= max_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = [text[: max_px // 8]]  # last-resort truncate
    return lines


class ScaleMode(Enum):
    """How to upscale NES pixels into the tile's display rect.

    The first three modes preserve the NES 256:240 aspect (letterboxing
    if the cell isn't a 256:240 rectangle). `STRETCH_FILL` ignores
    aspect and fills the entire cell — useful at high instance counts
    where every pixel of screen real-estate matters and you don't mind
    Link looking slightly squashed or stretched."""
    SHARP_BILINEAR = "Sharp bilinear (default, keep aspect)"
    PIXEL_PERFECT = "Pixel-perfect nearest (keep aspect)"
    SMOOTH = "Smooth bilinear (keep aspect)"
    STRETCH_FILL = "Stretch to fill cell (no letterbox)"


class _TileLabel(QLabel):
    """One tile of the grid. Knows how to render a raw NES frame into
    its current widget rect using the selected ScaleMode without
    allocating a QImage each paint.

    A caption overlay can be set via `set_caption(text, ttl_ms)`. The
    caption is drawn as a bottom-aligned text strip on top of the
    frame and auto-fades after `ttl_ms` milliseconds."""

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background: black; border: 1px solid #333")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Grow to fill the grid cell; the grid layout drives sizing.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(NES_WIDTH // 2, NES_HEIGHT // 2)
        self._scale_mode = ScaleMode.SHARP_BILINEAR
        # Sentinel that no real frame_id will ever equal. -1 is reserved
        # by the legacy provider path ("invalidate every tick"); using
        # None as the sentinel keeps those paths from accidentally
        # matching on the very first call.
        self._last_frame_id: int | None = None
        # Caption overlay state. None = no caption active. expiry is a
        # monotonic-ms timestamp after which the caption is cleared.
        self._caption_text: str | None = None
        self._caption_expiry_ms: int = 0
        self._caption_is_banner: bool = False
        # Persistent name tag rendered in the top-right corner so the
        # user can tell at a glance which genome occupies this tile.
        # None = no name yet.
        self._name_tag: str | None = None

    def set_name_tag(self, name: str | None) -> None:
        """Set (or clear) the top-right genome name badge."""
        if self._name_tag != name:
            self._name_tag = name
            self.update()  # schedule a repaint

    def set_caption(self, text: str, ttl_ms: int = 4500, banner: bool = False) -> None:
        """Show a short caption overlay at the bottom of the tile for
        `ttl_ms` milliseconds. `banner=True` highlights it in gold for
        first-of-its-kind events."""
        from PyQt6.QtCore import QTime
        self._caption_text = text
        self._caption_expiry_ms = QTime.currentTime().msecsSinceStartOfDay() + ttl_ms
        self._caption_is_banner = banner
        self.update()

    def set_scale_mode(self, mode: ScaleMode) -> None:
        self._scale_mode = mode
        # Force next render even if the frame ID hasn't changed.
        self._last_frame_id = None

    def sizeHint(self) -> QSize:
        return QSize(NES_WIDTH, NES_HEIGHT)

    def update_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """Called by the grid poll. `frame_id` is a monotonic per-tile
        counter (transport seq) — skip the render entirely if identical.
        Legacy providers pass -1 meaning "always invalidate" and we
        honor that by never short-circuiting on -1."""
        if frame_id != -1 and frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id

        # Ensure contiguous memory so QImage doesn't need a copy of
        # unaligned data. NES frames from the transport are already
        # contiguous but defensive .copy() is cheap.
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        # Build the QImage. Format_RGB888 maps uint8 RGB directly —
        # zero-copy view into the numpy buffer. We .copy() at the end
        # because the provider's buffer may be reused on the next poll.
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        label_w = self.width()
        label_h = self.height()
        if label_w <= 0 or label_h <= 0:
            # Widget not yet shown / sized — fall back to sizeHint so the
            # test harness (which calls _refresh_frames before .show())
            # still produces a paintable pixmap instead of silently skipping.
            hint = self.sizeHint()
            label_w = max(label_w, hint.width())
            label_h = max(label_h, hint.height())
            if label_w <= 0 or label_h <= 0:
                return

        # Compute the target rect that preserves NES aspect (256:240).
        aspect = NES_WIDTH / NES_HEIGHT
        if label_w / label_h > aspect:
            tgt_h = label_h
            tgt_w = int(tgt_h * aspect)
        else:
            tgt_w = label_w
            tgt_h = int(tgt_w / aspect)

        if self._scale_mode is ScaleMode.STRETCH_FILL:
            # Fill the ENTIRE cell, ignoring NES aspect. Useful at high
            # instance counts where every pixel of screen matters more
            # than perfect rendering geometry.
            pix = QPixmap.fromImage(qimg).scaled(
                label_w, label_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        elif self._scale_mode is ScaleMode.PIXEL_PERFECT:
            # Nearest-neighbor at the largest integer multiple that fits.
            scale = max(1, min(tgt_w // NES_WIDTH, tgt_h // NES_HEIGHT))
            pix = QPixmap.fromImage(qimg).scaled(
                NES_WIDTH * scale, NES_HEIGHT * scale,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        elif self._scale_mode is ScaleMode.SHARP_BILINEAR:
            # Two-stage: nearest-neighbor to the largest integer multiple
            # that still fits, then bilinear to the final rect. Residual
            # bilinear pass is ≤1 pixel per source pixel so it only
            # smooths subpixel offsets, not the pixel art itself.
            scale = max(1, min(tgt_w // NES_WIDTH, tgt_h // NES_HEIGHT))
            intermediate = QPixmap.fromImage(qimg).scaled(
                NES_WIDTH * scale, NES_HEIGHT * scale,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            if intermediate.width() == tgt_w and intermediate.height() == tgt_h:
                pix = intermediate
            else:
                pix = intermediate.scaled(
                    tgt_w, tgt_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

        else:  # SMOOTH
            pix = QPixmap.fromImage(qimg).scaled(
                tgt_w, tgt_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Overlay the name tag + caption onto the pixmap BEFORE setting
        # it. Drawing directly on the pixmap means the QLabel renders
        # in a single paint, no separate layered widgets needed.
        self._draw_overlays(pix)
        self.setPixmap(pix)

    def _draw_overlays(self, pix) -> None:
        """Draw the optional genome name tag (top-right) and caption
        strip (bottom, autofading) onto `pix` in place. Called after
        upscaling so overlay text sits at display resolution —
        readable regardless of NES pixel density."""
        from PyQt6.QtCore import QRect, Qt as _Qt, QTime
        from PyQt6.QtGui import QColor, QFont, QPainter, QPen

        if self._name_tag is None and self._caption_text is None:
            return

        # Expire stale captions before drawing. Avoids the next paint
        # showing text that's already past its TTL.
        if self._caption_text is not None:
            now_ms = QTime.currentTime().msecsSinceStartOfDay()
            if now_ms >= self._caption_expiry_ms:
                self._caption_text = None

        p = QPainter(pix)
        try:
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            if self._name_tag:
                # Top-right badge: semi-transparent pill with the
                # genome name. Keeps it readable over busy pixel art
                # without covering the play area.
                font = QFont("Menlo", 10, QFont.Weight.Bold)
                p.setFont(font)
                fm = p.fontMetrics()
                text = self._name_tag
                tw = fm.horizontalAdvance(text) + 10
                th = fm.height() + 4
                margin = 4
                badge = QRect(pix.width() - tw - margin, margin, tw, th)
                p.fillRect(badge, QColor(0, 0, 0, 170))
                p.setPen(QColor(230, 230, 230))
                p.drawText(badge, int(_Qt.AlignmentFlag.AlignCenter), text)

            if self._caption_text:
                # Bottom strip: full-width caption. Gold background for
                # first-of-its-kind banners, otherwise translucent black.
                font = QFont("Menlo", 11, QFont.Weight.Bold)
                p.setFont(font)
                fm = p.fontMetrics()
                # Word-wrap manually to the pixmap width; keep at most
                # two lines so the caption fits without covering Link.
                max_w = pix.width() - 8
                text = self._caption_text
                lines = _wrap_text(text, fm, max_w, max_lines=2)
                line_h = fm.height()
                strip_h = line_h * len(lines) + 6
                strip = QRect(0, pix.height() - strip_h, pix.width(), strip_h)
                if self._caption_is_banner:
                    p.fillRect(strip, QColor(190, 140, 30, 215))
                    p.setPen(QColor(255, 245, 200))
                else:
                    p.fillRect(strip, QColor(0, 0, 0, 185))
                    p.setPen(QColor(240, 240, 240))
                y = pix.height() - strip_h + line_h - 2
                for line in lines:
                    p.drawText(4, y, line)
                    y += line_h
        finally:
            p.end()


class EmulatorGrid(QMainWindow):
    """Window showing a grid of live emulator frames."""

    def __init__(self, num_instances: int) -> None:
        super().__init__()
        # Defensive floor: a zero-instance grid would ZeroDivisionError
        # in the layout math below. Caller bugs (e.g. config with 0
        # workers) should fail loudly elsewhere, but the grid window
        # itself is harmless to construct empty.
        if num_instances < 1:
            num_instances = 1
        self.setWindowTitle(f"NES — {num_instances} instances")

        self.num_instances = num_instances
        self.cols = math.ceil(math.sqrt(num_instances))
        self.rows = math.ceil(num_instances / self.cols)

        self._build_ui()
        self._build_menu()

        # Poll at ~30 fps. Each unchanged frame is a cheap no-op because
        # _TileLabel.update_frame short-circuits on frame_id match.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_frames)
        self._timer.start(33)

        # Provider is a callable returning a list of (frame, frame_id)
        # pairs, one per instance — or (frame, None) if no ID is tracked.
        self._frame_provider = None

        # Initial size: 2× NES per tile, but clamp to ~85 % of the
        # primary screen so a 64-instance grid doesn't open at 4096×3840
        # and disappear off the right of the monitor. Aspect-preserving
        # tile scaling means smaller initial windows still look right;
        # the user can resize freely.
        from PyQt6.QtWidgets import QApplication as _QApp
        screen = _QApp.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            max_w = int(geom.width() * 0.85)
            max_h = int(geom.height() * 0.85)
        else:
            max_w, max_h = 1600, 1200
        target_w = min(self.cols * NES_WIDTH * 2, max_w)
        target_h = min(self.rows * NES_HEIGHT * 2, max_h)
        self.resize(target_w, target_h)

    def closeEvent(self, event) -> None:
        # Stop the refresh timer on close so we never poll the frame
        # provider after AppController has unlinked the SharedMemory
        # blocks underneath us. Without this, every Stop click could
        # trigger a "FileNotFoundError: /dev/shm/..." or a torn read
        # in the brief window between close() and the QTimer's last
        # scheduled tick.
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        # Drop the provider too so any in-flight tick that already
        # entered _refresh_frames doesn't dereference a stale closure.
        self._frame_provider = None
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)
        # Uniform stretch — every column and row grows equally when the
        # user resizes the window, so tiles stay the same size relative
        # to each other.
        for c in range(self.cols):
            layout.setColumnStretch(c, 1)
        for r in range(self.rows):
            layout.setRowStretch(r, 1)

        self.frame_labels: list[_TileLabel] = []
        for i in range(self.num_instances):
            label = _TileLabel()
            row = i // self.cols
            col = i % self.cols
            layout.addWidget(label, row, col)
            self.frame_labels.append(label)

    def _build_menu(self) -> None:
        """View menu: choose scaling mode. Saved per session in-memory;
        sharp bilinear is the default because it balances crispness with
        smoothness at arbitrary resize ratios."""
        view_menu = self.menuBar().addMenu("&View")
        self._scale_actions: dict[ScaleMode, QAction] = {}
        for mode in ScaleMode:
            action = QAction(mode.value, self, checkable=True)
            action.setChecked(mode is ScaleMode.SHARP_BILINEAR)
            action.triggered.connect(
                lambda _checked=False, m=mode: self._set_scale_mode(m)
            )
            view_menu.addAction(action)
            self._scale_actions[mode] = action

    def _set_scale_mode(self, mode: ScaleMode) -> None:
        for m, act in self._scale_actions.items():
            act.setChecked(m is mode)
        for tile in self.frame_labels:
            tile.set_scale_mode(mode)

    def set_frame_provider(self, provider) -> None:
        """
        `provider` is a callable returning either:
            list[np.ndarray]                — legacy, frame IDs missing
            list[tuple[np.ndarray, int]]    — (frame, seq) per instance

        In the seq form we skip re-rendering unchanged frames entirely.
        """
        self._frame_provider = provider

    # -------------- caption / name overlay API ---------------------

    def set_tile_name(self, worker_id: int, name: str | None) -> None:
        """Set the top-right genome name badge on tile `worker_id`.
        No-op for out-of-range indices (demo tile, shrunk grids)."""
        if 0 <= worker_id < len(self.frame_labels):
            self.frame_labels[worker_id].set_name_tag(name)

    def show_caption(
        self,
        worker_id: int,
        text: str,
        banner: bool = False,
        ttl_ms: int = 4500,
    ) -> None:
        """Flash a caption over tile `worker_id`. `banner=True` styles
        it gold for first-of-its-kind moments."""
        if 0 <= worker_id < len(self.frame_labels):
            self.frame_labels[worker_id].set_caption(
                text, ttl_ms=ttl_ms, banner=banner
            )

    def _refresh_frames(self) -> None:
        if self._frame_provider is None:
            return
        items = self._frame_provider()
        for i, tile in enumerate(self.frame_labels):
            if i >= len(items):
                continue
            item = items[i]
            if item is None:
                continue
            if isinstance(item, tuple):
                frame, frame_id = item
            else:
                # Legacy providers — invalidate every poll.
                frame, frame_id = item, -1
            if frame is None:
                continue
            tile.update_frame(frame, frame_id if frame_id is not None else -1)
