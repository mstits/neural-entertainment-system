"""LIVE EXPLORATION HEATMAP — the search's coverage, painted in real time.

The swarm's wall-grinding is invisible on the worker tiles: identical-
looking frames while the archive quietly fills a level's state space. This
module turns that hidden progress into the show's centerpiece visual — a
side-panel heatmap of every cell the Go-Explore search has reached,
projected into level space (horizontal = forward progress, vertical =
height band), with:

  * VISIT INTENSITY   — persistent cold map (blue->cyan->white) of how
                        thoroughly each region has been explored.
  * RECENT-BURST TRAILS — cells discovered in the last few seconds glow
                        hot (white/orange), then cool. This is the live
                        pulse of the frontier being pushed.
  * FRONTIER BAND     — the current deepest progress column, highlighted,
                        so the eye tracks where the search is actually
                        breaking new ground.
  * DOOR / CHOKEPOINT markers (optional) — articulation cells, if the
                        caller supplies them.

Performance contract: per-frame cost is O(dirty cells + a decay pass over
the small fixed grid), NEVER O(archive). The archive (652k cells,
multi-GB) is read at most ONCE at cold start — and even then only its
cell KEYS, via a capped, state-skipping unpickler (the 21 KB emulator
state blob on every cell is discarded on the way in). Live growth arrives
as a stream of (cell_key, timestamp) events through `observe`.

Standalone demo / receipt:
  python scripts/show_heatmap.py \
      --archive runs/breadth_contra/stage1_v6_resume/archive.pkl \
      --cells 50000 --frames 600
It loads the Contra archive READ-ONLY, replays the first N discovered
cells as a simulated live feed, renders `--frames` frames offscreen (pure
numpy + PIL, no Qt required), writes 3 representative PNGs and a measured
receipt (load time, peak RSS, sustained render fps) under
docs/receipts/show_lane/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import struct
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = REPO / "runs/breadth_contra/stage1_v6_resume/archive.pkl"
RECEIPT_DIR = REPO / "docs/receipts/show_lane"


# ---------------------------------------------------------------------------
# Cold-start snapshot loader: read cell KEYS only, from a multi-GB archive.
#
# The archive is `pickle.dump(dict[CellKey, Cell])`. Each Cell carries a
# ~21 KB emulator state blob; 652k of them make the file multi-GB. For the
# heatmap we need only the keys (a few ints each). This unpickler:
#   * skips every bytes payload (the state blobs) — reads and discards the
#     stream bytes, substituting a tiny stub, so they never accumulate in
#     memory nor in the unpickler memo;
#   * remaps the Cell class to a slotted key-only shell;
#   * ABORTS after `cap` items, so a 50k-cell replay reads ~1% of the file
#     instead of unpickling all of it.
# Dict insertion order is preserved by pickle, and archive insertion order
# is first-discovery order — so iterating the result IS discovery order.
# ---------------------------------------------------------------------------

class _EarlyStop(Exception):
    pass


class _KeyOnlyCell:
    """A Cell stripped to what the heatmap projects. The 21 KB state blob is
    dropped during unpickling; only key + a couple of light scalars remain."""
    __slots__ = ("key", "best_score", "best_steps", "visits")

    def __setstate__(self, st: dict) -> None:
        self.key = st.get("key")
        self.best_score = st.get("best_score", 0.0)
        self.best_steps = st.get("best_steps", 0)
        self.visits = st.get("visits", 1)


_STATE_STUB = b""


class _CellKeyUnpickler(pickle._Unpickler):
    """Pure-Python unpickler that skips state blobs and stops after `cap`
    dict items. Pure-Python is required to override opcode handlers; at ~50k
    items it is still well under a second (the C loader can't be subclassed
    and would materialize the whole multi-GB file)."""

    def __init__(self, f, cap: int):
        super().__init__(f)
        self._cap = cap
        self._result: Optional[dict] = None

    def find_class(self, module, name):
        if name == "Cell":
            return _KeyOnlyCell
        return super().find_class(module, name)

    # -- discard bytes payloads (the emulator state blobs) ------------------
    def load_short_binbytes(self):
        n = self.read(1)[0]
        self.read(n)                     # consume, keep nothing
        self.append(_STATE_STUB)

    def load_binbytes(self):
        (n,) = struct.unpack("<I", self.read(4))
        self.read(n)
        self.append(_STATE_STUB)

    def load_binbytes8(self):
        (n,) = struct.unpack("<Q", self.read(8))
        self.read(n)
        self.append(_STATE_STUB)

    # -- abort once we have enough cells -----------------------------------
    def load_setitems(self):
        super().load_setitems()
        d = self.stack[-1]
        if len(d) >= self._cap:
            self._result = d
            raise _EarlyStop

    def load_setitem(self):
        super().load_setitem()
        d = self.stack[-1]
        if isinstance(d, dict) and len(d) >= self._cap:
            self._result = d
            raise _EarlyStop

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.SHORT_BINBYTES[0]] = load_short_binbytes
    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8
    dispatch[pickle.SETITEMS[0]] = load_setitems
    dispatch[pickle.SETITEM[0]] = load_setitem

    def load(self):
        try:
            return super().load()
        except _EarlyStop:
            return self._result


def load_archive_keys(path, cap: int = 50000) -> list[tuple]:
    """Cold-start snapshot: return up to `cap` cell keys, in discovery order,
    reading only the head of a multi-GB archive.pkl. READ-ONLY."""
    with open(path, "rb") as f:
        cells = _CellKeyUnpickler(f, cap).load()
    return list((cells or {}).keys())


# ---------------------------------------------------------------------------
# Colour lookup tables (256-entry gradients, built once).
# ---------------------------------------------------------------------------

def _gradient_lut(anchors: Sequence[tuple]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 256)
    xs = np.array([a[0] for a in anchors], dtype=np.float32)
    lut = np.zeros((256, 3), dtype=np.float32)
    for ch in range(3):
        ys = np.array([a[1][ch] for a in anchors], dtype=np.float32)
        lut[:, ch] = np.interp(t, xs, ys)
    return lut


# Cold = cumulative visit density. Dark blue -> teal -> cyan -> white.
_COLD_LUT = _gradient_lut([
    (0.00, (12, 18, 42)),
    (0.25, (24, 64, 130)),
    (0.55, (34, 150, 175)),
    (0.80, (120, 220, 230)),
    (1.00, (235, 248, 255)),
])
# Hot = recent-burst glow, ADDED on top. Black (adds nothing) -> red ->
# orange -> yellow -> white-hot.
_HOT_LUT = _gradient_lut([
    (0.00, (0, 0, 0)),
    (0.30, (120, 30, 0)),
    (0.60, (230, 110, 0)),
    (0.82, (255, 210, 70)),
    (1.00, (255, 255, 240)),
])
_BASE_RGB = np.array([6, 7, 12], dtype=np.float32)        # unexplored panel
_FRONTIER_RGB = np.array([90, 255, 190], dtype=np.float32)  # frontier band
_DOOR_RGB = np.array([255, 60, 225], dtype=np.float32)      # chokepoint mark


class HeatmapRenderer:
    """Incremental exploration heatmap -> RGBA surface.

    Feed discovery/visit events with `observe(cell_key, ts)`; call
    `render(now)` (up to 60 fps) to get an (H, W, 4) uint8 RGBA array. All
    per-frame work is on the fixed grid, independent of archive size.

    Cell-key convention (Go-Explore solver): key[0] = section index,
    key[-2] = y-band, key[-1] = gx bucket (forward progress). Horizontal
    grid position = section stride + gx bucket, auto-scrolling so the
    frontier stays on-screen without ever re-touching archived cells.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 240,
        *,
        gx_cols: int = 420,
        y_bands: int = 32,
        grid_h: int = 32,
        sect_stride: int = 400,
        trail_halflife: float = 1.2,
        visit_ref: float = 200.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.gw = int(gx_cols)
        self.gh = int(grid_h)
        self.y_bands = int(y_bands)
        self.sect_stride = int(sect_stride)
        self.trail_halflife = float(trail_halflife)
        self._log_ref = float(np.log1p(max(1.0, visit_ref)))

        self.visits = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.heat = np.zeros((self.gh, self.gw), dtype=np.float32)
        self._x_offset = 0            # columns scrolled off the left edge
        self._frontier_raw = -1       # max raw_x ever observed
        self._doors_raw: list[tuple[int, int]] = []   # (raw_x, row)

        self._base = np.empty((self.gh, self.gw, 3), dtype=np.float32)
        self._base[:] = _BASE_RGB
        # nearest-neighbour upscale index maps (grid -> surface), built once.
        self._ys = (np.arange(self.height) * self.gh // self.height)
        self._xs = (np.arange(self.width) * self.gw // self.width)
        self._rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        self._rgba[..., 3] = 255
        self._last_render_t: Optional[float] = None
        self.events_seen = 0

    # -- projection --------------------------------------------------------
    def _row_col(self, key) -> tuple[int, int]:
        yb = int(key[-2])
        gxb = int(key[-1])
        sect = int(key[0]) if len(key) else 0
        raw_x = sect * self.sect_stride + gxb
        row = yb if self.gh == self.y_bands else min(
            int(yb * self.gh / max(1, self.y_bands)), self.gh - 1)
        row = min(max(row, 0), self.gh - 1)
        return row, raw_x

    def _scroll(self, delta: int) -> None:
        delta = min(int(delta), self.gw)
        if delta <= 0:
            return
        self.visits = np.roll(self.visits, -delta, axis=1)
        self.visits[:, -delta:] = 0.0
        self.heat = np.roll(self.heat, -delta, axis=1)
        self.heat[:, -delta:] = 0.0
        self._x_offset += delta

    # -- feed API ----------------------------------------------------------
    def observe(self, key, ts: Optional[float] = None, *, hot: bool = True) -> None:
        """Record one discovery/visit event (O(1)). `ts` is accepted for
        API symmetry; recency is driven by wall time at render. `hot=False`
        pre-fills a cold-start snapshot cell without a live burst glow."""
        row, raw_x = self._row_col(key)
        col = raw_x - self._x_offset
        if col >= self.gw:
            self._scroll(col - (self.gw - 1))
            col = self.gw - 1
        if col < 0:
            return
        self.visits[row, col] += 1.0
        if hot:
            self.heat[row, col] = 1.0
        if raw_x > self._frontier_raw:
            self._frontier_raw = raw_x
        self.events_seen += 1

    def observe_many(self, keys: Iterable, *, hot: bool = True) -> None:
        for k in keys:
            self.observe(k, hot=hot)

    def prefill(self, keys: Iterable) -> None:
        """Bulk-load a cold-start snapshot as already-explored (cold) cells."""
        self.observe_many(keys, hot=False)

    def set_doors(self, keys: Iterable) -> None:
        """Optional articulation / chokepoint markers to overlay."""
        self._doors_raw = []
        for k in keys:
            row, raw_x = self._row_col(k)
            self._doors_raw.append((raw_x, row))

    # -- render ------------------------------------------------------------
    def render(self, now: Optional[float] = None) -> np.ndarray:
        """Decay the burst trails and compose the RGBA surface. Cost is
        O(grid), independent of archive size. Returns an (H, W, 4) uint8
        array (buffer is reused; copy if you need to retain it)."""
        if now is None:
            now = time.monotonic()
        if self._last_render_t is not None:
            dt = now - self._last_render_t
            if dt > 0.0:
                self.heat *= 0.5 ** (dt / self.trail_halflife)
                self.heat[self.heat < 0.01] = 0.0
        self._last_render_t = now

        grid = self._base.copy()
        mask = self.visits > 0.0
        if mask.any():
            vidx = np.clip(
                np.log1p(self.visits) / self._log_ref, 0.0, 1.0)
            vidx = (vidx * 255.0).astype(np.int32)
            grid[mask] = _COLD_LUT[vidx[mask]]
            hidx = (np.clip(self.heat, 0.0, 1.0) * 255.0).astype(np.int32)
            grid += _HOT_LUT[hidx]

        # frontier band (highlight the deepest active column).
        fc = self._frontier_raw - self._x_offset
        if 0 <= fc < self.gw:
            grid[:, fc] = 0.45 * grid[:, fc] + 0.55 * _FRONTIER_RGB
            if fc + 1 < self.gw:
                grid[:, fc + 1] = 0.8 * grid[:, fc + 1] + 0.2 * _FRONTIER_RGB

        # chokepoint markers.
        for raw_x, row in self._doors_raw:
            c = raw_x - self._x_offset
            if 0 <= c < self.gw:
                grid[row, c] = _DOOR_RGB

        np.clip(grid, 0.0, 255.0, out=grid)
        surf = grid[self._ys][:, self._xs].astype(np.uint8)   # (H, W, 3)
        self._rgba[..., :3] = surf
        return self._rgba

    # -- Qt convenience (lazy import; not needed for the numpy/PIL path) ----
    def qimage(self, now: Optional[float] = None):
        from PyQt6.QtGui import QImage
        arr = self.render(now)
        return QImage(arr.tobytes(), self.width, self.height,
                      4 * self.width, QImage.Format.Format_RGBA8888)


# ---------------------------------------------------------------------------
# Standalone demo + receipt.
# ---------------------------------------------------------------------------

def _percentile(vals: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(vals), p)) if vals else 0.0


def _demo(args) -> int:
    import resource

    archive = Path(args.archive)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[heatmap] cold-start snapshot: reading up to {args.cells} keys "
          f"from {archive} ...")
    t0 = time.time()
    keys = load_archive_keys(archive, cap=args.cells)
    load_s = time.time() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[heatmap] loaded {len(keys)} cell keys in {load_s:.2f}s "
          f"(peak RSS {rss_after_load:.0f} MB)")

    if not keys:
        print("[heatmap] no cells loaded — aborting")
        return 1

    order = args.order
    feed = list(keys)
    if order == "gx":
        feed.sort(key=lambda k: (k[0], k[-1], k[-2]))
    elif order == "random":
        np.random.default_rng(0).shuffle(feed)
    # "discovery" (default): keep archive insertion order.

    gxs = [k[-1] for k in feed]
    ybs = [k[-2] for k in feed]
    areas = sorted({k[-5] for k in feed})
    gx_lo, gx_hi = min(gxs), max(gxs)

    r = HeatmapRenderer(width=args.width, height=args.height,
                        gx_cols=max(args.width, gx_hi + 8))

    # Chokepoint stand-in for the demo: the highest-traffic columns read as
    # natural bottlenecks (real integration would pass the solver's R4
    # articulation cells here). Cheap, purely illustrative.
    col_traffic: dict[int, int] = {}
    for k in feed:
        col_traffic[k[-1]] = col_traffic.get(k[-1], 0) + 1
    hot_cols = sorted(col_traffic, key=col_traffic.get, reverse=True)[:4]
    r.set_doors([(0, 0, 0, (), 0, (), 0, 12, 0, 12, c) for c in hot_cols])

    frames = int(args.frames)
    per_frame = max(1, len(feed) // frames)
    png_at = {max(1, frames // 10): "early", frames // 2: "mid",
              frames: "late"}
    png_paths: dict[str, str] = {}

    render_times: list[float] = []
    feed_times: list[float] = []
    idx = 0
    sim_t = 0.0
    sim_dt = 1.0 / 60.0
    loop0 = time.time()
    for fi in range(1, frames + 1):
        tf = time.time()
        end = min(len(feed), idx + per_frame)
        while idx < end:
            r.observe(feed[idx], ts=sim_t)
            idx += 1
        feed_times.append(time.time() - tf)

        sim_t += sim_dt
        tr = time.time()
        surf = r.render(now=sim_t)             # sim clock -> deterministic decay
        render_times.append(time.time() - tr)

        if fi in png_at:
            from PIL import Image
            p = out_dir / f"heatmap_{png_at[fi]}.png"
            Image.fromarray(surf, "RGBA").save(p)
            png_paths[png_at[fi]] = str(p)
    loop_s = time.time() - loop0

    render_total = sum(render_times)
    render_fps = frames / render_total if render_total > 0 else 0.0
    combined_fps = frames / loop_s if loop_s > 0 else 0.0
    frame_ms = [t * 1000.0 for t in render_times]
    rss_final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    receipt = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "cells_requested": args.cells,
        "cells_loaded": len(keys),
        "feed_order": order,
        "cold_start_load_seconds": round(load_s, 3),
        "peak_rss_mb_after_load": round(rss_after_load, 1),
        "peak_rss_mb_final": round(rss_final, 1),
        "grid": [r.gw, r.gh],
        "surface": [r.width, r.height],
        "gx_bucket_range": [int(gx_lo), int(gx_hi)],
        "y_band_range": [int(min(ybs)), int(max(ybs))],
        "areas": [int(a) for a in areas],
        "frames_rendered": frames,
        "events_replayed": r.events_seen,
        "events_per_frame": per_frame,
        "render_only_seconds": round(render_total, 4),
        "render_only_fps": round(render_fps, 1),
        "avg_render_ms": round(render_total / frames * 1000.0, 4),
        "p50_render_ms": round(_percentile(frame_ms, 50), 4),
        "p99_render_ms": round(_percentile(frame_ms, 99), 4),
        "max_render_ms": round(max(frame_ms), 4),
        "combined_feed_plus_render_fps": round(combined_fps, 1),
        "target_fps": 60,
        "meets_target": render_fps >= 60.0,
        "pngs": png_paths,
    }
    receipt_path = out_dir / "heatmap_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"[heatmap] rendered {frames} frames | render-only "
          f"{render_fps:.0f} fps (avg {receipt['avg_render_ms']} ms, "
          f"p99 {receipt['p99_render_ms']} ms) | combined "
          f"{combined_fps:.0f} fps")
    print(f"[heatmap] target 60 fps: "
          f"{'PASS' if receipt['meets_target'] else 'FAIL'}")
    print(f"[heatmap] receipt -> {receipt_path}")
    for tag, p in png_paths.items():
        print(f"[heatmap] png[{tag}] -> {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE),
                    help="Go-Explore archive.pkl (READ-ONLY)")
    ap.add_argument("--cells", type=int, default=50000,
                    help="cap on cells to load/replay (default 50000)")
    ap.add_argument("--frames", type=int, default=600,
                    help="frames to render for the fps measurement")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--order", choices=("discovery", "gx", "random"),
                    default="discovery",
                    help="feed order for the simulated live stream")
    ap.add_argument("--out", default=str(RECEIPT_DIR),
                    help="receipt + PNG output directory")
    return _demo(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
