"""THE POLISH TIER: pure surface transforms for the show's rendering path.

Three composable effects, each a plain function/class over numpy arrays —
none of them know about Qt, the solver, or the mixer. Any surface (hero
cam, swarm tile, a future stats HUD) calls these directly on the frames
it already has:

  crt(frame_rgb, strength)        -- scanlines + slight barrel/vignette +
                                      phosphor bloom, vectorized (LUTs
                                      cached per input shape; no per-pixel
                                      python loops).
  upscale(frame, factor)          -- integer nearest-neighbor upscale
                                      (pixel-purity default) with an
                                      optional scanline-aware variant that
                                      pairs with crt() at low cost when a
                                      full shader pass isn't warranted.
  Sparkline(history_len).render() -- a small RGBA strip for stat overlays
                                      (sps, cells/min, frontier progress):
                                      min/max autoscale + current-value
                                      marker, NES-palette-friendly colors.

crt() takes ONLY (frame, strength) in the common case — it infers the
display scale from the frame's own height (round(h / 240)) so it applies
correctly whether the caller upscaled first (the intended composition:
`crt(upscale(frame, 3), strength)`) or handed it a native 240x256 frame
directly. Pass `scale=` explicitly to skip the inference or to force a
scanline period that doesn't match the frame's actual size.

Demo + receipt: `python scripts/show_fx.py` replays real SMB solution
frames through nes_core, runs crt+upscale at 2x/3x (numpy, and torch/MPS
if available, for a head-to-head), renders sparklines from a synthetic
sps series, saves before/after PNGs, and writes measured per-frame ms to
docs/receipts/show_lane/fx_receipt.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import torch
    _TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    _TORCH_DEVICE = None


# ---------------------------------------------------------------------------
# upscale()
# ---------------------------------------------------------------------------

def upscale(frame: np.ndarray, factor: int, scanline_aware: bool = False) -> np.ndarray:
    """Integer nearest-neighbor upscale. Vectorized via np.repeat (a C-level
    op, not a python pixel loop).

    scanline_aware=True bakes a cheap scanline gap into the upscale itself
    (dims the last sub-row of every `factor`-row block) — a lighter-weight
    alternative to a full crt() pass for surfaces that don't need the
    barrel/vignette/bloom shader (e.g. the swarm grid tiles at full machine
    speed).
    """
    frame = np.asarray(frame)
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        out = frame.copy()
    else:
        out = np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)
    if scanline_aware and factor > 1:
        dim_rows = np.arange(factor - 1, out.shape[0], factor)
        out[dim_rows, :, :3] = (
            out[dim_rows, :, :3].astype(np.float32) * 0.55
        ).astype(out.dtype)
    return out


# ---------------------------------------------------------------------------
# crt() -- numpy backend (default)
# ---------------------------------------------------------------------------
# LUTs are cached per (h, w) so repeated calls at a steady resolution (the
# common case: one surface, one scale, 60 times a second) pay the meshgrid
# cost once. `strength` only scales the cached unit fields, so re-keying
# the cache on strength is unnecessary.

_RADIAL_CACHE: dict[tuple[int, int], np.ndarray] = {}
_SCANLINE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _normalized_grid(h: int, w: int):
    """Cached per-shape (nx, ny, r2) normalized coordinate fields, shared
    by the barrel warp and the vignette -- the mgrid build (the expensive
    part) happens once per (h, w), not once per frame."""
    cached = _RADIAL_CACHE.get((h, w))
    if cached is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ny = (ys - cy) / (h / 2.0)
        nx = (xs - cx) / (w / 2.0)
        cached = (nx, ny, nx * nx + ny * ny)
        _RADIAL_CACHE[(h, w)] = cached
    return cached


_BARREL_IDX_CACHE: dict = {}


def _barrel_flat_index(h: int, w: int, k: float) -> np.ndarray:
    """Flat gather indices for a slight pincushion/barrel warp, nearest-
    sampled. k=0 is exactly the identity permutation, so crt(strength=0)
    is a true passthrough.

    Cached per (h, w, k) -- k only changes when the caller's `strength`
    changes, which in the show is a fixed config value, so in steady
    state this whole function is a dict lookup and only the gather itself
    (real per-frame work, since pixel content changes) runs every call.

    Returned as a flat (h*w,) int index for `arr.reshape(-1, c)[idx]` --
    measured ~7x faster than the equivalent `arr[src_y, src_x]` 2D fancy
    index (4.0ms -> 0.6ms at 480x512): numpy's advanced indexing has much
    lower per-call overhead gathering along a single flattened axis than
    pairing two full-shape index arrays.
    """
    key = (h, w, round(k, 6))
    idx = _BARREL_IDX_CACHE.get(key)
    if idx is not None:
        return idx
    nx, ny, r2 = _normalized_grid(h, w)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    factor = 1.0 + k * r2
    src_x = np.clip(np.round(nx * factor * (w / 2.0) + cx), 0, w - 1)
    src_y = np.clip(np.round(ny * factor * (h / 2.0) + cy), 0, h - 1)
    idx = (src_y.astype(np.int64) * w + src_x.astype(np.int64)).ravel()
    _BARREL_IDX_CACHE[key] = idx
    return idx


def _scanline_mask(h: int, period: int, dim: float) -> np.ndarray:
    idx = _SCANLINE_CACHE.get((h, period))
    if idx is None:
        idx = (np.arange(h) % period) == (period - 1)
        _SCANLINE_CACHE[(h, period)] = idx
    mask = np.ones(h, dtype=np.float32)
    mask[idx] = 1.0 - dim
    return mask[:, None, None]


def _box_blur(img_f32: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via shifted-slice summation -- no per-pixel
    python loop (only a `2*radius+1`-iteration loop over whole-array
    slices, a small constant since bloom radii stay small). Two passes
    (vertical then horizontal) approximate a gaussian well enough for a
    bloom highlight.

    Measured faster than the equivalent cumsum-based box blur at the
    small radii (1-3) this module uses: 1.34ms -> 0.43ms at radius 1 on a
    240x256x3 array -- cumsum's pad+concatenate overhead dominates when
    the kernel itself is this small; it only pays off at large radii."""
    if radius < 1:
        return img_f32
    n = 2 * radius + 1

    def _pass(a: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (radius, radius)
        p = np.pad(a, pad_width, mode="edge")
        acc = None
        for i in range(n):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            term = p[tuple(sl)]
            acc = term if acc is None else acc + term
        return acc / n

    return _pass(_pass(img_f32, 0), 1)


def _downsample(f: np.ndarray, scale: int) -> np.ndarray:
    return f[::scale, ::scale] if scale > 1 else f


def _upsample_like(small: np.ndarray, scale: int, h: int, w: int) -> np.ndarray:
    if scale == 1:
        return small
    big = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    return big[:h, :w]  # crop off any stride rounding overshoot


_SHADE_CACHE: dict = {}


def _shade_mask(h: int, w: int, strength: float) -> np.ndarray:
    """Vignette x scanlines fused into one cached (h, w, 1) multiplier so
    applying both costs a single elementwise pass instead of two. Always
    built at the (h, w) it's given -- see the native-resolution note on
    `_crt_numpy` for why that's a native-res shape, not the output one."""
    key = (h, w, round(strength, 4))
    m = _SHADE_CACHE.get(key)
    if m is None:
        _, _, r2 = _normalized_grid(h, w)
        vmask = np.clip(1.0 - (0.35 * strength) * r2, 0.35, 1.0)
        smask = _scanline_mask(h, 2, 0.45 * strength)  # (h, 1, 1)
        m = (vmask[..., None] * smask).astype(np.float32)
        _SHADE_CACHE[key] = m
    return m


def _crt_numpy(frame: np.ndarray, strength: float, scale: int) -> np.ndarray:
    """The whole shader pipeline (barrel warp, bloom, vignette, scanlines)
    runs on a NATIVE-resolution working buffer, upscaled to the output
    size exactly once at the end. Every one of these effects is a smooth,
    low-frequency, or explicitly-quantized-to-scanlines operation, so
    computing them at 240x256 and nearest-upscaling loses nothing visible
    while cutting the element count of every pass by scale^2. Measured:
    two separate repeat-upscales (one for the warp, one for bloom) plus a
    full-res shade multiply cost ~1.9ms combined at 2x on 480x512; one
    shared native-res pipeline + a single final upscale costs ~0.6ms for
    the equivalent work."""
    h, w = frame.shape[:2]
    small = _downsample(frame, scale).astype(np.float32)
    sh, sw = small.shape[:2]

    barrel_k = 0.06 * strength
    if barrel_k > 0:
        flat_idx = _barrel_flat_index(sh, sw, barrel_k)
        small = small.reshape(-1, small.shape[-1])[flat_idx].reshape(sh, sw, -1)

    if strength > 0:
        bright = np.clip(small - 170.0, 0.0, None)
        small = small + _box_blur(bright, 1) * (0.35 * strength)

    small = small * _shade_mask(sh, sw, strength)

    f = _upsample_like(small, scale, h, w)
    np.clip(f, 0, 255, out=f)
    return f.astype(np.uint8)


# ---------------------------------------------------------------------------
# crt() -- torch/MPS backend (measured alternative; see main() for the
# numpy-vs-torch comparison this module was benchmarked with)
# ---------------------------------------------------------------------------

_TORCH_R2_CACHE: dict = {}


def _crt_torch(frame: np.ndarray, strength: float, scale: int) -> np.ndarray:
    h, w = frame.shape[:2]
    device = _TORCH_DEVICE
    t = torch.from_numpy(np.ascontiguousarray(frame)).to(
        device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W

    key = (h, w, device)
    r2 = _TORCH_R2_CACHE.get(key)
    if r2 is None:
        ny, nx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device), indexing="ij")
        r2 = nx * nx + ny * ny
        _TORCH_R2_CACHE[key] = (nx, ny, r2)
    nx, ny, r2 = _TORCH_R2_CACHE[key]

    barrel_k = 0.06 * strength
    if barrel_k > 0:
        factor = 1.0 + barrel_k * r2
        grid = torch.stack([nx * factor, ny * factor], dim=-1).unsqueeze(0)
        t = torch.nn.functional.grid_sample(
            t, grid, mode="nearest", align_corners=True, padding_mode="border")

    if strength > 0:
        bright = torch.clamp(t - 170.0, min=0.0)
        k = max(1, int(round(scale)))
        bloom = torch.nn.functional.avg_pool2d(
            bright, kernel_size=2 * k + 1, stride=1, padding=k)
        t = t + bloom * (0.35 * strength)

    vmask = torch.clamp(1.0 - (0.35 * strength) * r2, min=0.35, max=1.0)
    t = t * vmask

    period = max(2, int(round(scale)))
    rows = torch.arange(h, device=device)
    smask = torch.where((rows % period) == (period - 1),
                        1.0 - 0.45 * strength, 1.0).view(1, 1, h, 1)
    t = t * smask

    t = torch.clamp(t, 0, 255)
    return t.squeeze(0).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()


def crt(frame_rgb: np.ndarray, strength: float = 1.0, scale: int | None = None,
        backend: str = "numpy") -> np.ndarray:
    """scanlines + slight barrel/vignette + phosphor bloom approximation.

    strength in [0, 1]; 0 is an exact passthrough. `scale` sets the
    scanline period / bloom radius to match the frame's actual display
    scale -- if omitted it's inferred as round(height / 240), so calling
    crt(frame, strength) on either a native 240x256 frame or one already
    upscaled by upscale() does the right thing.
    """
    frame = np.asarray(frame_rgb)
    strength = float(np.clip(strength, 0.0, 1.0))
    if scale is None:
        scale = max(1, round(frame.shape[0] / 240.0))
    if backend == "torch":
        if not _TORCH_OK:
            raise RuntimeError("torch backend requested but torch is unavailable")
        return _crt_torch(frame, strength, int(scale))
    return _crt_numpy(frame, strength, int(scale))


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

class Sparkline:
    """A small stat strip: min/max-autoscaled trace + current-value marker.

    Two usage modes, both supported:
      spark = Sparkline(history_len=120)
      spark.push(value)              # each tick
      img = spark.render(size=(w,h)) # reads the internal ring buffer

    or, stateless, pass the series directly:
      img = Sparkline(0).render(values=my_array, size=(w, h))

    Returns an (h, w, 4) uint8 RGBA array. Colors lean on a small
    NES-adjacent palette (SMB sky blue trace, coin-gold marker) rather
    than an arbitrary chart palette, so it drops into the show's frame
    without clashing.
    """

    BG = (16, 20, 24, 170)
    FILL = (28, 60, 96, 200)
    LINE = (92, 148, 252, 255)
    MARKER = (252, 188, 0, 255)

    def __init__(self, history_len: int = 120):
        self.history_len = max(1, int(history_len))
        self._buf: deque[float] = deque(maxlen=self.history_len)

    def push(self, value: float) -> None:
        self._buf.append(float(value))

    update = push  # alias -- either name reads naturally at a call site

    def render(self, values=None, size: tuple[int, int] = (120, 28)) -> np.ndarray:
        w, h = size
        img = np.empty((h, w, 4), dtype=np.uint8)
        for ch in range(4):
            img[..., ch] = self.BG[ch]

        vals = np.asarray(values if values is not None else self._buf,
                          dtype=np.float32)
        if vals.size == 0:
            return img
        if vals.size == 1:
            vals = np.array([vals[0], vals[0]], dtype=np.float32)

        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        if span < 1e-6:
            span = max(abs(hi), 1.0)  # flat series: still draw a centered line
        norm = np.clip((vals - lo) / span, 0.0, 1.0)

        xs_src = np.linspace(0.0, 1.0, norm.size)
        xs_dst = np.linspace(0.0, 1.0, w)
        line = np.interp(xs_dst, xs_src, norm)

        pad = max(1, int(h * 0.08))
        y = (1.0 - line) * (h - 1 - 2 * pad) + pad  # autoscaled plot y, per column

        rows = np.arange(h)[:, None]
        col_y = y[None, :]
        thickness = max(1.0, h * 0.05)
        line_mask = np.abs(rows - col_y) <= thickness
        fill_mask = rows > col_y

        for ch in range(4):
            img[..., ch] = np.where(fill_mask, self.FILL[ch], img[..., ch])
            img[..., ch] = np.where(line_mask, self.LINE[ch], img[..., ch])

        mr, mc = int(round(y[-1])), w - 1  # current-value marker, last column
        rad = max(1, int(h * 0.09))
        r0, r1 = max(0, mr - rad), min(h, mr + rad + 1)
        c0 = max(0, mc - rad)
        for ch in range(4):
            img[r0:r1, c0:w, ch] = self.MARKER[ch]

        return img


# ---------------------------------------------------------------------------
# Demo CLI + receipt
# ---------------------------------------------------------------------------

def _find_replay_source(profile_path: Path):
    """Locate a real entrance state + a matching solved-action trace to
    replay for the demo. Prefers the SMB 1-1 pair from the live show's own
    run directory; falls back to the first solution found anywhere under
    runs/live_show with any entrance state sitting alongside it, so the
    demo still works if that exact level was re-run or evicted."""
    show_dir = REPO / "runs/live_show" / profile_path.stem
    preferred_entrance = show_dir / "entrance_start.state"
    preferred_sol = show_dir / "lvl_1-1/solutions/sol_000.actions.npy"
    if preferred_entrance.exists() and preferred_sol.exists():
        return preferred_entrance, preferred_sol

    for sol in sorted((REPO / "runs/live_show").glob("*/lvl_*/solutions/sol_*.actions.npy")):
        lvl_show_dir = sol.parents[2]
        states = sorted(lvl_show_dir.glob("entrance*.state"))
        if states:
            return states[0], sol
    raise FileNotFoundError(
        "no entrance state + solved trace found under runs/live_show/ -- "
        "run the show at least once (make show) before the fx demo")


def _replay_frames(profile: dict, entrance: Path, sol: Path, n_frames: int) -> list[np.ndarray]:
    import nes_core
    from src.training.profile_utils import action_space_to_bitmasks

    rom = str(REPO / profile["rom_path"])
    fs = int(profile.get("frame_skip", 4))
    bm = action_space_to_bitmasks(profile["action_space"])
    actions = np.load(sol).astype(int)

    env = nes_core.NESEnvironment(rom, frame_skip=fs)
    env.load_state(entrance.read_bytes())
    for _ in range(fs):  # settle, matching the show's own rooting convention
        env.step(0)

    frames = []
    n = min(n_frames, len(actions))
    for a in actions[:n]:
        frame, _done = env.step(int(bm[int(a)]))
        frames.append(np.array(frame))
    return frames


def _timeit_ms(fn, n_repeat: int = 1):
    t0 = time.perf_counter()
    out = None
    for _ in range(n_repeat):
        out = fn()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / n_repeat
    return out, dt_ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(REPO / "configs/smb_4_4_micro.yaml"))
    ap.add_argument("--frames", type=int, default=200,
                    help="how many replayed SMB frames to benchmark over")
    ap.add_argument("--scales", default="2,3")
    ap.add_argument("--out-dir", default=str(REPO / "docs/receipts/show_lane"))
    ap.add_argument("--sparkline-len", type=int, default=150)
    args = ap.parse_args()

    import yaml
    from PIL import Image

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [int(s) for s in args.scales.split(",")]

    profile_path = Path(args.profile)
    profile = yaml.safe_load(profile_path.read_text())
    entrance, sol = _find_replay_source(profile_path)
    print(f"[fx] replay source: entrance={entrance} solution={sol}")

    frames = _replay_frames(profile, entrance, sol, args.frames)
    print(f"[fx] replayed {len(frames)} real SMB frames "
         f"({frames[0].shape[1]}x{frames[0].shape[0]})")

    receipt = {
        "profile": str(profile_path.relative_to(REPO)),
        "entrance": str(entrance.relative_to(REPO)) if REPO in entrance.parents else str(entrance),
        "solution": str(sol.relative_to(REPO)) if REPO in sol.parents else str(sol),
        "n_frames_replayed": len(frames),
        "native_shape": list(frames[0].shape),
        "torch_available": _TORCH_OK,
        "torch_device": _TORCH_DEVICE,
        "target_ms_crt_plus_2x": 4.0,
        "effects": {},
        "images": {},
    }

    mid = frames[len(frames) // 2]
    Image.fromarray(mid, mode="RGB").save(out_dir / "before_native.png")
    receipt["images"]["before_native"] = str((out_dir / "before_native.png").relative_to(REPO))

    for scale in scales:
        # warmup: prime the shape-keyed LUT caches (and, for torch, the
        # device/graph warmup) before timing.
        up0 = upscale(frames[0], scale)
        _ = crt(up0, 1.0, scale=scale)
        if _TORCH_OK:
            try:
                _ = crt(up0, 1.0, scale=scale, backend="torch")
            except Exception as e:
                print(f"[fx] torch backend unavailable at {scale}x: {e}")

        up_ms, crt_ms, combined_ms = [], [], []
        for f in frames:
            t0 = time.perf_counter()
            up = upscale(f, scale)
            t1 = time.perf_counter()
            out = crt(up, 1.0, scale=scale)
            t2 = time.perf_counter()
            up_ms.append((t1 - t0) * 1000.0)
            crt_ms.append((t2 - t1) * 1000.0)
            combined_ms.append((t2 - t0) * 1000.0)

        crt_torch_ms = []
        torch_error = None
        if _TORCH_OK:
            try:
                for f in frames:
                    up = upscale(f, scale)
                    t0 = time.perf_counter()
                    _ = crt(up, 1.0, scale=scale, backend="torch")
                    crt_torch_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                torch_error = str(e)

        def stats(ms_list):
            a = np.asarray(ms_list, dtype=np.float64)
            return {"mean_ms": round(float(a.mean()), 4),
                    "p95_ms": round(float(np.percentile(a, 95)), 4),
                    "max_ms": round(float(a.max()), 4)}

        scale_key = f"{scale}x"
        entry = {
            "upscale_numpy": stats(up_ms),
            "crt_numpy": stats(crt_ms),
            "crt_plus_upscale_numpy_combined": stats(combined_ms),
        }
        entry["crt_plus_upscale_numpy_combined"]["meets_4ms_target"] = bool(
            entry["crt_plus_upscale_numpy_combined"]["mean_ms"] < 4.0)
        if crt_torch_ms:
            entry["crt_torch"] = stats(crt_torch_ms)
            entry["crt_torch"]["faster_than_numpy"] = bool(
                entry["crt_torch"]["mean_ms"] < entry["crt_numpy"]["mean_ms"])
        elif torch_error:
            entry["crt_torch_error"] = torch_error
        receipt["effects"][scale_key] = entry
        print(f"[fx] {scale_key}: upscale {entry['upscale_numpy']['mean_ms']:.3f} ms, "
             f"crt {entry['crt_numpy']['mean_ms']:.3f} ms, "
             f"combined {entry['crt_plus_upscale_numpy_combined']['mean_ms']:.3f} ms"
             + (f", torch {entry['crt_torch']['mean_ms']:.3f} ms" if crt_torch_ms else ""))

        after = crt(upscale(mid, scale), 1.0, scale=scale)
        p = out_dir / f"after_crt_{scale_key}.png"
        Image.fromarray(after, mode="RGB").save(p)
        receipt["images"][f"after_crt_{scale_key}"] = str(p.relative_to(REPO))

    # -- Sparkline demo -----------------------------------------------------
    rng = np.random.default_rng(0)
    n = args.sparkline_len
    walk = np.cumsum(rng.normal(0, 60, size=n))
    sps_series = np.clip(3000 + walk - walk.mean(), 200, None)
    sps_series[n // 3: n // 3 + 8] *= 0.15  # a stall, like the campaign's real telemetry

    spark = Sparkline(history_len=n)
    for v in sps_series:
        spark.push(v)

    (wide, wide_ms) = _timeit_ms(lambda: spark.render(size=(240, 48)), n_repeat=20)
    (small, small_ms) = _timeit_ms(lambda: spark.render(size=(120, 24)), n_repeat=20)
    (stateless, _) = _timeit_ms(
        lambda: Sparkline(0).render(values=sps_series[-60:], size=(160, 32)), n_repeat=20)

    Image.fromarray(wide, mode="RGBA").save(out_dir / "sparkline_wide.png")
    Image.fromarray(small, mode="RGBA").save(out_dir / "sparkline_small.png")
    Image.fromarray(stateless, mode="RGBA").save(out_dir / "sparkline_stateless.png")

    receipt["sparkline"] = {
        "history_len": n,
        "render_ms_240x48": round(wide_ms, 4),
        "render_ms_120x24": round(small_ms, 4),
        "synthetic_series": "sps: base ~3000 + random walk + one simulated stall",
    }
    receipt["images"]["sparkline_wide"] = str((out_dir / "sparkline_wide.png").relative_to(REPO))
    receipt["images"]["sparkline_small"] = str((out_dir / "sparkline_small.png").relative_to(REPO))
    receipt["images"]["sparkline_stateless"] = str((out_dir / "sparkline_stateless.png").relative_to(REPO))
    print(f"[fx] sparkline render: 240x48 {wide_ms:.4f} ms, 120x24 {small_ms:.4f} ms")

    receipt_path = out_dir / "fx_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"[fx] receipt written: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
