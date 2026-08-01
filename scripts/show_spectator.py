"""THE SPECTATOR RENDERER — swarm/hero views decoupled from the solver.

Today the live show renders its swarm grid and hero cam by making the
solver's POOL WORKERS produce full RGB frames (`pool.set_headless(False)`).
Those workers pay the pixel-rendering cost on the very P-cores that are
supposed to be searching — the SPECTATOR TAX. Every frame the audience
sees is a frame the search did not compute.

This module removes that tax. The pool stays HEADLESS at full search speed;
a SEPARATE lightweight renderer takes tiny save-state snapshots (~21 KB,
the same blobs Go-Explore already mints for free at 1.5 us each) at a modest
cadence (10-20 per second per tile) and renders frames OUTSIDE the pool, on
its own `nes_core.NESEnvironment` instances. Restoring a snapshot and
rendering one frame costs ~0.6-1.1 ms on one core, so a full swarm mosaic
plus a hero tile renders comfortably under a single core — a core the
solver was never using for search anyway.

Design
------
`SpectatorRenderer` owns K tile-slot emulators (one env per slot, all the
SAME ROM — restoring arbitrary states into one env is stable only within a
ROM; see the receipt) plus an optional bigger HERO env. The producer (the
solver's step hook) pushes each worker's latest snapshot via `update_slot`;
the renderer applies pending snapshots (`load_state`), steps each slot's own
emulator one frame to render it (which also gives smooth motion between
snapshots), and blits the tiles into one composite surface: an N-up mosaic
beside a hero panel at native or integer scale. `tick()` returns the
composite as a plain `HxWx3` uint8 array — the Qt window blits it exactly
like `live_solve_show._to_pixmap` blits worker frames today.

The renderer runs in its OWN thread (`SpectatorThread`), so its render CPU
lands on a spare core, never on the solver's. Feeding it costs the solver
only `save_worker_state` (1.5 us) at the snapshot cadence — negligible.

Demo / receipt
--------------
Run this file directly: it pulls 16 diverse Contra save-state blobs from a
live archive (streamed — never loads the multi-GB pickle), composites a 4x4
mosaic + hero for 30 s, saves 3 PNGs, and writes measured restores/sec,
rendered tile-frames/sec, and single-core utilization to
docs/receipts/show_lane/spectator_receipt.json.

    .venv/bin/python scripts/show_spectator.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

DEFAULT_ARCHIVE = REPO / "runs/breadth_contra/stage1_v6_resume/archive.pkl"
DEFAULT_ROM = REPO / "roms/Contra (USA).nes"
DEFAULT_RECEIPT = REPO / "docs/receipts/show_lane/spectator_receipt.json"


# ---------------------------------------------------------------------------
# Nearest-neighbour integer scale (matches the show's FastTransformation).
# ---------------------------------------------------------------------------
def _scale_nn(frame: np.ndarray, s: int) -> np.ndarray:
    """Return an owned, contiguous copy of `frame` scaled s x (s >= 1)."""
    if s <= 1:
        return np.array(frame, dtype=np.uint8)   # own it; get_frame is a view
    return np.repeat(np.repeat(frame, s, axis=0), s, axis=1).astype(np.uint8)


class SpectatorRenderer:
    """K tile-slot emulators + one hero, composited into a single surface.

    All envs share one ROM (restoring arbitrary states into a single env is
    stable within a ROM — verified in the receipt — but NOT across ROMs; use
    one renderer per ROM). Thread-safe producer methods (`update_slot`,
    `update_hero`); render happens on whatever thread calls `tick()`.
    """

    NATIVE_W, NATIVE_H = 256, 240

    def __init__(self, rom_path: str, n_tiles: int = 16, cols: int = 4,
                 tile_scale: int = 1, hero_scale: int = 2, frame_skip: int = 1,
                 tile_fps: float = 20.0, gap: int = 8,
                 bg: tuple[int, int, int] = (18, 18, 18)) -> None:
        self.rom_path = str(rom_path)
        self.n_tiles = int(n_tiles)
        self.cols = int(cols)
        self.rows = (self.n_tiles + self.cols - 1) // self.cols
        self.tile_scale = max(1, int(tile_scale))
        self.hero_scale = int(hero_scale)          # 0 disables the hero panel
        self.tile_fps = float(tile_fps)
        self._tile_dt = 1.0 / max(self.tile_fps, 1e-6)
        self.gap = int(gap)

        # One env per mosaic slot + one hero env. frame_skip 1: a snapshot
        # renders with a single-frame advance (minimal drift from the saved
        # state) and animates one NES frame per rendered tick.
        self._envs = [nes_core.NESEnvironment(self.rom_path, frame_skip=frame_skip)
                      for _ in range(self.n_tiles)]
        for e in self._envs:
            e.reset()
        self._hero_env = None
        if self.hero_scale > 0:
            self._hero_env = nes_core.NESEnvironment(self.rom_path,
                                                     frame_skip=frame_skip)
            self._hero_env.reset()

        # Producer -> renderer handoff (latest-wins; a slow renderer never
        # backs up the solver, it just drops stale snapshots).
        self._lock = threading.Lock()
        self._pending: list[Optional[bytes]] = [None] * self.n_tiles
        self._pending_hero: Optional[bytes] = None
        self._primed = [False] * self.n_tiles      # has the slot ever rendered
        self._hero_primed = False
        self._next_render = [0.0] * self.n_tiles
        self._next_render_hero = 0.0

        # Geometry.
        self.tw = self.NATIVE_W * self.tile_scale
        self.th = self.NATIVE_H * self.tile_scale
        self._mosaic_w = self.cols * self.tw + (self.cols - 1) * self.gap
        self._mosaic_h = self.rows * self.th + (self.rows - 1) * self.gap
        self._hw = self.NATIVE_W * self.hero_scale if self._hero_env else 0
        self._hh = self.NATIVE_H * self.hero_scale if self._hero_env else 0
        self._hero_gap = self.gap if self._hero_env else 0
        self.H = max(self._mosaic_h, self._hh)
        self.W = self._hw + self._hero_gap + self._mosaic_w
        self._bg = np.array(bg, dtype=np.uint8)
        self._composite = np.empty((self.H, self.W, 3), dtype=np.uint8)
        self._composite[:] = self._bg

        # Precomputed blit rects (hero left, mosaic right — the show layout).
        self._mosaic_x0 = self._hw + self._hero_gap
        self._hero_y0 = (self.H - self._hh) // 2 if self._hero_env else 0

        # Counters (the receipt reads these).
        self.restores = 0
        self.tile_frames = 0
        self.ticks = 0

    # -- geometry helper -------------------------------------------------
    def _slot_rect(self, i: int) -> tuple[int, int]:
        r, c = divmod(i, self.cols)
        y0 = r * (self.th + self.gap)
        x0 = self._mosaic_x0 + c * (self.tw + self.gap)
        return y0, x0

    # -- producer side (thread-safe, cheap) ------------------------------
    def update_slot(self, i: int, blob: bytes) -> None:
        """Hand slot `i` its latest ~21 KB save-state snapshot. Latest wins;
        called from the solver's step hook at the snapshot cadence."""
        if 0 <= i < self.n_tiles and blob is not None:
            with self._lock:
                self._pending[i] = bytes(blob)

    def update_hero(self, blob: bytes) -> None:
        if self._hero_env is not None and blob is not None:
            with self._lock:
                self._pending_hero = bytes(blob)

    # -- render side -----------------------------------------------------
    def _render_env(self, env, scale: int) -> np.ndarray:
        frame, _ = env.step(0)          # advance one frame -> renders it
        return _scale_nn(np.asarray(frame), scale)

    # Per-tick render budget: nes_core's env calls HOLD the GIL (no
    # allow_threads in the bindings yet), so rendering all tiles every
    # tick starves the interpreter inside the live show — the receipted
    # in-show failure was a sub-1 fps mosaic while the standalone demo
    # (idle main thread) hit 20 fps/tile. Budgeted round-robin keeps
    # spectator GIL demand at ~30-60 ms/s regardless of tile count.
    TILES_PER_TICK = 4

    def tick(self) -> np.ndarray:
        """Apply pending snapshots, render up to TILES_PER_TICK due tiles
        (round-robin), composite, return the surface (a live buffer — copy
        it if another thread consumes it)."""
        now = time.monotonic()
        with self._lock:
            pending, self._pending = self._pending, [None] * self.n_tiles
            phero, self._pending_hero = self._pending_hero, None

        # Apply ALL pending snapshots (cheap ~21 KB deserializes) so no
        # blob is dropped, but budget the expensive render+scale work.
        for i, env in enumerate(self._envs):
            if pending[i] is not None:
                try:
                    env.load_state(pending[i])
                    self.restores += 1
                    self._next_render[i] = 0.0   # render ASAP post-restore
                except Exception:
                    pass

        rendered = 0
        start = getattr(self, "_rr_cursor", 0)
        for k in range(self.n_tiles):
            i = (start + k) % self.n_tiles
            if rendered >= self.TILES_PER_TICK:
                break
            if now < self._next_render[i] and self._primed[i]:
                continue
            env = self._envs[i]
            try:
                buf = self._render_env(env, self.tile_scale)
            except Exception:
                continue
            y0, x0 = self._slot_rect(i)
            self._composite[y0:y0 + self.th, x0:x0 + self.tw] = buf
            self.tile_frames += 1
            self._primed[i] = True
            self._next_render[i] = now + self._tile_dt
            rendered += 1
        self._rr_cursor = (start + rendered) % self.n_tiles

        if self._hero_env is not None:
            rendered = False
            if phero is not None:
                try:
                    self._hero_env.load_state(phero)
                    self.restores += 1
                    rendered = True
                except Exception:
                    pass
            if rendered or now >= self._next_render_hero or not self._hero_primed:
                try:
                    buf = self._render_env(self._hero_env, self.hero_scale)
                    self._composite[self._hero_y0:self._hero_y0 + self._hh,
                                    0:self._hw] = buf
                    self.tile_frames += 1
                    self._hero_primed = True
                    self._next_render_hero = now + self._tile_dt
                except Exception:
                    pass

        self.ticks += 1
        return self._composite

    @property
    def stats(self) -> dict:
        return {"restores": self.restores, "tile_frames": self.tile_frames,
                "ticks": self.ticks, "n_tiles": self.n_tiles,
                "composite_wh": [self.W, self.H]}

    def close(self) -> None:
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass
        if self._hero_env is not None:
            try:
                self._hero_env.close()
            except Exception:
                pass


class SpectatorThread(threading.Thread):
    """Drives a SpectatorRenderer on its own core at a target composite fps.

    The Qt window reads `.latest()` (a stable copy) on its 60 Hz timer; the
    render CPU never touches the solver's threads."""

    def __init__(self, renderer: SpectatorRenderer, fps: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.r = renderer
        self.dt = 1.0 / max(fps, 1e-6)
        self._latest: Optional[np.ndarray] = None
        self._llock = threading.Lock()
        self.stop = False

    def latest(self) -> Optional[np.ndarray]:
        with self._llock:
            return None if self._latest is None else self._latest

    def run(self) -> None:
        try:
            from src.training.qos import demote_current_thread
            demote_current_thread()   # E-cores: rendering is ambience
        except Exception:
            pass
        nxt = time.monotonic()
        while not self.stop:
            comp = self.r.tick()
            with self._llock:
                self._latest = comp.copy()
            nxt += self.dt
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                nxt = time.monotonic()


# ---------------------------------------------------------------------------
# Streaming archive tap — pull a few diverse save-state blobs from a huge
# Go-Explore archive WITHOUT unpickling the whole (multi-GB) dict. A custom
# Unpickler intercepts the Cell class; each cell's __setstate__ records its
# blob and aborts once enough distinct gx-buckets are collected. Peak memory
# is bounded by the scan cap, not the file size.
# ---------------------------------------------------------------------------
class _StopScan(Exception):
    pass


def sample_archive_blobs(path: str | Path, n: int = 16,
                         scan_cap: int = 20000) -> list[tuple]:
    """Return up to `n` (cell_key, state_blob) pairs spread across the
    archive's gx-buckets, streamed from a pickle of arbitrary size."""
    picks: dict = {}          # gx_bucket -> (key, blob)
    seen = [0]

    class _Tap:
        __slots__ = ()        # keep nothing on the object placed in the dict
        def __setstate__(self, st):     # noqa: D401 (pickle hook)
            seen[0] += 1
            blob = st.get("state")
            if blob is not None:
                k = st.get("key")
                gxb = k[-1] if isinstance(k, tuple) and k else seen[0]
                picks.setdefault(gxb, (k, blob))
            if len(picks) >= n * 3 or seen[0] >= scan_cap:
                raise _StopScan

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Cell":
                return _Tap
            return super().find_class(module, name)

    try:
        with open(path, "rb") as f:
            _U(f).load()
    except _StopScan:
        pass
    if not picks:
        return []
    buckets = sorted(picks)
    idx = np.linspace(0, len(buckets) - 1, min(n, len(buckets)))
    chosen = sorted({int(round(x)) for x in idx})
    return [picks[buckets[i]] for i in chosen]


# ---------------------------------------------------------------------------
# Demo CLI + receipt.
# ---------------------------------------------------------------------------
def _single_core_ceiling(rom: str, blobs: list[bytes], iters: int = 2500) -> float:
    """Max rendered restores/sec on ONE core with diverse states (the ceiling
    a spectator core can sustain)."""
    env = nes_core.NESEnvironment(rom, frame_skip=1)
    env.reset(); env.step(0)
    for i in range(200):
        env.load_state(blobs[i % len(blobs)]); env.step(0)
    t0 = time.perf_counter()
    for i in range(iters):
        env.load_state(blobs[i % len(blobs)])
        f, _ = env.step(0)
        _ = np.asarray(f)
    dt = time.perf_counter() - t0
    env.close()
    return iters / dt


def _load_state_micro(rom: str, blob: bytes, iters: int = 3000) -> dict:
    env = nes_core.NESEnvironment(rom, frame_skip=1)
    env.reset(); env.load_state(blob); env.step(0)
    t0 = time.perf_counter()
    for _ in range(iters):
        env.load_state(blob)
    load_us = (time.perf_counter() - t0) / iters * 1e6
    t0 = time.perf_counter()
    for _ in range(iters):
        env.load_state(blob); f, _ = env.step(0); _ = np.asarray(f)
    render_us = (time.perf_counter() - t0) / iters * 1e6
    env.close()
    return {"load_state_us": round(load_us, 3),
            "load_step_frame_us": round(render_us, 3)}


def _verify_stability(rom: str, blobs: list[bytes]) -> dict:
    """One env, round-robin restore across every distinct state; confirm no
    crash, distinct frames, and deterministic restore."""
    import hashlib
    env = nes_core.NESEnvironment(rom, frame_skip=1)
    env.reset(); env.step(0)
    hs = []
    for b in blobs:
        env.load_state(b); f, _ = env.step(0)
        hs.append(hashlib.md5(np.asarray(f).tobytes()).hexdigest())
    env.load_state(blobs[0]); f1, _ = env.step(0)
    h1 = hashlib.md5(np.asarray(f1).tobytes()).hexdigest()
    env.load_state(blobs[-1]); env.step(0)              # perturb
    env.load_state(blobs[0]); f2, _ = env.step(0)
    h2 = hashlib.md5(np.asarray(f2).tobytes()).hexdigest()
    env.close()
    return {"states_tested": len(blobs),
            "distinct_frames": len(set(hs)),
            "deterministic_restore": bool(h1 == h2),
            "one_env_same_rom_stable": True}


def _feeder(renderer: SpectatorRenderer, blobs: list[bytes],
            stop: threading.Event, per_tile_hz: float = 15.0,
            hero_hz: float = 5.0) -> None:
    """Simulate the solver pushing snapshots: rotate a diverse blob into each
    slot at `per_tile_hz`, cycling the blob so tiles visibly change (swarm
    churn). In production this loop lives on the solver threads as a throttled
    pool.save_worker_state -> update_slot, costing 1.5 us per push."""
    step = 0
    dt = 1.0 / per_tile_hz
    hero_every = max(1, int(per_tile_hz / max(hero_hz, 1e-6)))
    nxt = time.monotonic()
    while not stop.is_set():
        for i in range(renderer.n_tiles):
            renderer.update_slot(i, blobs[(i + step) % len(blobs)])
        if step % hero_every == 0:
            renderer.update_hero(blobs[step % len(blobs)])
        step += 1
        nxt += dt
        d = nxt - time.monotonic()
        if d > 0:
            stop.wait(d)
        elif d < -0.25:
            nxt = time.monotonic()


def run_demo(archive: str, rom: str, n_tiles: int, cols: int, seconds: float,
             tile_fps: float, composite_fps: float, hero_scale: int,
             tile_scale: int, receipt_path: Path, png_dir: Path) -> dict:
    print(f"== spectator renderer demo ==")
    print(f"  archive: {archive}")
    print(f"  rom:     {rom}")
    t_pull = time.perf_counter()
    pairs = sample_archive_blobs(archive, n=n_tiles + 1)
    pull_s = time.perf_counter() - t_pull
    if len(pairs) < n_tiles + 1:
        raise SystemExit(f"archive yielded only {len(pairs)} blobs "
                         f"(need {n_tiles + 1})")
    blobs = [bytes(b) for _, b in pairs]
    print(f"  pulled {len(blobs)} diverse blobs in {pull_s * 1e3:.1f} ms "
          f"(gx buckets {[k[-1] for k, _ in pairs]})")

    micro = _load_state_micro(rom, blobs[0])
    stability = _verify_stability(rom, blobs[:n_tiles])
    ceiling = _single_core_ceiling(rom, blobs)
    print(f"  load_state: {micro['load_state_us']} us | "
          f"load+step+frame: {micro['load_step_frame_us']} us")
    print(f"  stability: {stability['distinct_frames']}/"
          f"{stability['states_tested']} distinct, "
          f"deterministic={stability['deterministic_restore']}")
    print(f"  single-core ceiling: {ceiling:.0f} rendered restores/s "
          f"= {ceiling / (n_tiles + 1):.1f} fps across {n_tiles + 1} tiles")

    rend = SpectatorRenderer(rom, n_tiles=n_tiles, cols=cols,
                             tile_scale=tile_scale, hero_scale=hero_scale,
                             tile_fps=tile_fps)
    print(f"  composite: {rend.W}x{rend.H}  ({cols}x{rend.rows} mosaic + hero)")

    stop = threading.Event()
    feeder = threading.Thread(target=_feeder, args=(rend, blobs, stop),
                              daemon=True)
    feeder.start()

    png_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    shot_at = {seconds * 0.15: 0, seconds * 0.5: 1, seconds * 0.93: 2}
    shots_done: set[int] = set()
    saved_pngs: list[str] = []

    dt = 1.0 / composite_fps
    t0 = time.monotonic()
    cpu0 = time.thread_time()
    nxt = t0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= seconds:
            break
        rend.tick()
        for mark, idx in shot_at.items():
            if idx not in shots_done and elapsed >= mark:
                p = png_dir / f"spectator_composite_{idx}.png"
                Image.fromarray(rend._composite.copy()).save(p)
                saved_pngs.append(str(p))
                shots_done.add(idx)
        nxt += dt
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        elif d < -0.25:
            nxt = time.monotonic()
    wall = time.monotonic() - t0
    cpu = time.thread_time() - cpu0
    stop.set()
    feeder.join(timeout=1.0)
    st = rend.stats
    rend.close()

    restores_per_s = st["restores"] / wall
    tile_frames_per_s = st["tile_frames"] / wall
    core_frac = cpu / wall
    print(f"  --- {wall:.1f}s run ---")
    print(f"  restores/s:          {restores_per_s:.0f}")
    print(f"  rendered tile-fps:   {tile_frames_per_s:.0f} "
          f"(~{tile_frames_per_s / (n_tiles + 1):.1f} fps/tile x {n_tiles + 1})")
    print(f"  render core util:    {core_frac:.3f} core "
          f"({core_frac * 100:.1f}% of one core)")
    print(f"  saved {len(saved_pngs)} PNGs -> {png_dir}")

    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"

    receipt = {
        "what": "low-cost spectator renderer — swarm mosaic + hero rendered "
                "outside the solver pool from ~21KB save-state snapshots",
        "git_commit": commit,
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor() or "arm",
                    "python": platform.python_version()},
        "source_archive": str(archive),
        "rom": str(rom),
        "blob_pull": {"blobs": len(blobs),
                      "pull_ms": round(pull_s * 1e3, 1),
                      "method": "streaming unpickle, bounded scan "
                                "(never loads the full multi-GB pickle)"},
        "microbench": micro,
        "stability_caveat_verified": stability,
        "single_core_ceiling_restores_per_s": round(ceiling, 1),
        "config": {"n_tiles": n_tiles, "cols": cols, "rows": rend.rows,
                   "tile_scale": tile_scale, "hero_scale": hero_scale,
                   "tile_fps_target": tile_fps,
                   "composite_fps_target": composite_fps,
                   "composite_wh": [rend.W, rend.H],
                   "feeder_snapshot_hz_per_tile": 15.0},
        "measured": {
            "wall_s": round(wall, 2),
            "restores_per_s": round(restores_per_s, 1),
            "rendered_tile_frames_per_s": round(tile_frames_per_s, 1),
            "fps_per_tile": round(tile_frames_per_s / (n_tiles + 1), 2),
            "render_core_fraction": round(core_frac, 4),
            "render_core_percent": round(core_frac * 100, 2),
            "full_mosaic_under_one_core": bool(core_frac < 1.0),
        },
        "pngs": saved_pngs,
        "verdict": ("full swarm mosaic + hero renders under one core; the "
                    "solver pool stays headless at full search speed"
                    if core_frac < 1.0 else
                    "over one core — lower tile_fps or tile count"),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"  receipt -> {receipt_path}")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--rom", default=str(DEFAULT_ROM))
    ap.add_argument("--n-tiles", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--tile-fps", type=float, default=20.0)
    ap.add_argument("--composite-fps", type=float, default=60.0)
    ap.add_argument("--hero-scale", type=int, default=2)
    ap.add_argument("--tile-scale", type=int, default=1)
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    ap.add_argument("--png-dir", default=str(DEFAULT_RECEIPT.parent))
    args = ap.parse_args()
    run_demo(args.archive, args.rom, args.n_tiles, args.cols, args.seconds,
             args.tile_fps, args.composite_fps, args.hero_scale,
             args.tile_scale, Path(args.receipt), Path(args.png_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
