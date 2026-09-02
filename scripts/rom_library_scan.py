#!/usr/bin/env python3
"""
ROM library compatibility scanner.

Walks a directory of .nes files, tries to load + run each through
nes_core.NESEnvironment for N frames, and emits a per-ROM result
(JSON + CSV). Used to answer "what fraction of the full NES library
works today?" and to identify the long-tail mappers that need
hardening before nes_core can claim general-purpose emulator status.

Each ROM is tested in a subprocess so a Rust panic on one cart doesn't
take down the whole scan. Parallel by default via multiprocessing
(one ROM at a time per worker; --workers N controls concurrency).

Usage:
  scripts/rom_library_scan.py --roms-dir /path/to/library \
                              --out scan_results \
                              --workers 8 \
                              --frames 300 \
                              --timeout 15

Outputs:
  <out>.json   — full per-ROM records (list of dicts)
  <out>.csv    — tabular: path,bytes,md5,mapper,sub_mapper,nes20,
                 status,frames_run,wall_ms,error,motion,
                 static_distinct_hashes,static_first_change_frame
  <out>.md     — summary: per-mapper compatibility, top panic classes,
                 failure buckets, static-screen check.

Status values:
  ok                — loaded, reset, ran `frames` frames cleanly
  header_parse_err  — iNES header malformed or mapper unknown
  load_err          — header OK but env construction failed
  reset_err         — reset() panicked
  step_panic        — step() panicked (records frame index)
  timeout           — ran past --timeout seconds without finishing
  io_err            — couldn't read file

`ok` alone only means "no panic or timeout": a ROM that boots to a
frozen screen is `ok` too. --static-check (on by default) runs a
second pass on every `ok` ROM: reset, then `--static-frames` frames
under a periodic Start/A-burst-plus-late-random input schedule,
hashing the rendered framebuffer every frame. If the hash never
changes from the first frame, the ROM is classified `static` instead
of `live`. This mirrors the ground-truth executor's hang re-probe
method (2026-09-01, receipts/hang_reprobe.py: 3000 frames, Start/A
bursts + random input, 1 distinct hash over the run = static) so a
scan run this way can't repeat the "Jackal reports as a pass" mistake.

Motion values (only set on `ok` records when --static-check is on):
  live       : framebuffer changed at least once during the check
  static     : framebuffer hash never changed (frozen screen)
  check_err  : the static-check pass itself panicked/errored
  ""         : --static-check was off, or status != ok

The scan is read-only on the ROM directory. No side effects outside
the --out artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------------------------------------------------------
# Per-ROM probe (runs in a subprocess so panics are isolated).
# -----------------------------------------------------------------------------

def probe_one(args: tuple) -> dict:
    """Probe a single ROM. Must be top-level for multiprocessing."""
    rom_path, frames, static_cfg = args
    rom_path = str(rom_path)
    t0 = time.perf_counter()

    rec = {
        "path": rom_path,
        "name": os.path.basename(rom_path),
        "bytes": 0,
        "md5": "",
        "mapper": -1,
        "sub_mapper": 0,
        "nes20": False,
        "status": "pending",
        "frames_run": 0,
        "wall_ms": 0.0,
        "error": "",
        "motion": "",
        "static_distinct_hashes": "",
        "static_first_change_frame": "",
    }

    try:
        rec["bytes"] = os.path.getsize(rom_path)
    except OSError as e:
        rec["status"] = "io_err"
        rec["error"] = f"stat: {e}"
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    # nes_core import isolated per-process (avoids sharing GIL-held state
    # across probes; also means a panic in one probe can't corrupt
    # the shared interpreter of another).
    try:
        import nes_core  # type: ignore
    except Exception as e:
        rec["status"] = "load_err"
        rec["error"] = f"import nes_core: {e}"
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    # Header parse — fast, no emulator spin-up.
    try:
        md5, mapper, sub_mapper, nes20 = nes_core.rom_info(rom_path)
        rec["md5"] = md5
        rec["mapper"] = int(mapper)
        rec["sub_mapper"] = int(sub_mapper)
        rec["nes20"] = bool(nes20)
    except Exception as e:
        rec["status"] = "header_parse_err"
        rec["error"] = _short(str(e))
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    # Construct environment — this triggers mapper::new() which may
    # panic on obscure carts. Caught here.
    try:
        env = nes_core.NESEnvironment(rom_path, frame_skip=1)
    except BaseException as e:
        rec["status"] = "load_err"
        rec["error"] = _short(f"{type(e).__name__}: {e}")
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    # Reset — PPU warm-up + initial vector fetch. Panics on some
    # broken mappers.
    try:
        env.reset()
    except BaseException as e:
        rec["status"] = "reset_err"
        rec["error"] = _short(f"{type(e).__name__}: {e}")
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return rec

    # Run frames with neutral input. Most games idle at title without
    # any input; a smaller set boot into demo mode that exercises
    # mapper/PPU/APU paths immediately. 300 frames ≈ 5 seconds real
    # time at 60 fps — enough to surface early-boot panics.
    for i in range(frames):
        try:
            env.step(0)
            rec["frames_run"] = i + 1
        except BaseException as e:
            rec["status"] = "step_panic"
            rec["error"] = _short(f"frame={i} {type(e).__name__}: {e}")
            rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            return rec

    rec["status"] = "ok"

    # Static-screen check: this ROM never panicked or timed out, but
    # that alone can't tell a live boot from a frozen one (Jackal,
    # among others). Re-run under a mashing input schedule and hash
    # every frame, see classify_motion. Uses a FRESH env, not the one
    # that just ran `frames` neutral frames above: env.reset() is a
    # soft reset (some mappers keep bank/IRQ state across it, same as
    # hitting the console's reset button vs power-cycling), and the
    # executor's hang_reprobe.py template this mirrors always starts
    # from a brand-new NESEnvironment. Reusing the stepped env here
    # produced 3 false "static" verdicts on ROMs the census confirmed
    # live (John Elway's Quarterback, The Punisher, Rescue - The
    # Embassy Mission), a soft-reset artifact, not a real freeze.
    if static_cfg:
        try:
            static_env = nes_core.NESEnvironment(rom_path, frame_skip=1)
            motion = classify_motion(
                static_env,
                frames=static_cfg["frames"],
                period=static_cfg["period"],
                start_burst_len=static_cfg["start_burst_len"],
                a_burst_start=static_cfg["a_burst_start"],
                a_burst_len=static_cfg["a_burst_len"],
                random_from=static_cfg["random_from"],
                random_prob=static_cfg["random_prob"],
                seed=static_cfg["seed"],
                reset_first=True,
            )
            rec["motion"] = motion["motion"]
            rec["static_distinct_hashes"] = motion["distinct_hashes"]
            rec["static_first_change_frame"] = (
                motion["first_change_frame"]
                if motion["first_change_frame"] is not None else ""
            )
        except BaseException as e:
            rec["motion"] = "check_err"
            rec["error"] = _short(f"static-check: {type(e).__name__}: {e}")

    rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    return rec


def classify_motion(env, *, frames: int, period: int, start_burst_len: int,
                     a_burst_start: int, a_burst_len: int, random_from: int,
                     random_prob: float, seed: int,
                     reset_first: bool = True) -> dict:
    """Run `env` for `frames` frames under a periodic Start/A-burst
    input schedule (random input mixed in from `random_from` on) and
    hash the rendered framebuffer every frame. If the hash never
    changes from the first frame, the screen is static, mashing
    Start/A and eventually everything else for thousands of frames
    produced no visible response.

    Mirrors the ground-truth executor's re-probe method exactly at
    default parameters (period=120, start_burst_len=8, a_burst_start=60,
    a_burst_len=4, random_from=1500, random_prob=0.5): see
    reports/macos-emulation-and-training/2026-09-01-ground-truth-execution/
    receipts/hang_reprobe.py.

    `env` needs only `.reset()` (called first when `reset_first`) and
    `.step(action)` + `.get_frame()`, this takes a plain fake in
    tests, a real nes_core.NESEnvironment in the scanner. No nes_core
    or numpy import at module scope, so this stays unit-testable
    without either installed.
    """
    if reset_first:
        env.reset()

    rng = random.Random(seed)
    hashes: set[str] = set()
    first_hash = None
    changed_at = None

    for f in range(frames):
        m = f % period
        if m < start_burst_len:
            action = 0x08  # Start
        elif a_burst_start <= m < a_burst_start + a_burst_len:
            action = 0x01  # A
        else:
            action = 0
        if f >= random_from and rng.random() < random_prob:
            action = rng.randrange(256)

        env.step(action)
        frame = env.get_frame()
        try:
            frame_bytes = frame.tobytes()
        except AttributeError:
            # Fake envs in tests may hand back plain bytes already.
            frame_bytes = bytes(frame)
        h = hashlib.md5(frame_bytes).hexdigest()
        hashes.add(h)
        if first_hash is None:
            first_hash = h
        elif changed_at is None and h != first_hash:
            changed_at = f

    return {
        "distinct_hashes": len(hashes),
        "first_change_frame": changed_at,
        "frames_checked": frames,
        "motion": "static" if changed_at is None else "live",
    }


def _short(s: str, n: int = 240) -> str:
    """Trim for CSV / summary; full error stays in JSON if needed."""
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[:n] + "…"


# -----------------------------------------------------------------------------
# Scan orchestration.
# -----------------------------------------------------------------------------

def discover(roms_dir: Path, dedup: bool = True,
             min_bytes: int = 16 + 16 * 1024,  # header + 1 PRG bank
             max_bytes: int = 16 * 1024 * 1024,  # 16 MB hard ceiling
             ) -> list[Path]:
    """Walk the directory, filter obvious non-ROMs, optionally dedup
    by file MD5. Dedup here is BYTE-EXACT (whole file including header)
    rather than PRG+CHR — cheaper and still catches the common case
    of repeated downloads in the same collection.
    """
    candidates: list[Path] = []
    for root, _dirs, files in os.walk(roms_dir):
        for f in files:
            if f.lower().endswith(".nes"):
                p = Path(root) / f
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                if sz < min_bytes or sz > max_bytes:
                    continue
                candidates.append(p)
    candidates.sort()

    if not dedup or not candidates:
        return candidates

    # File-MD5 dedup. Use the OS stat size as a cheap first pass —
    # only hash files that share a size with another file.
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in candidates:
        by_size[p.stat().st_size].append(p)
    dedup_out: list[Path] = []
    seen_hashes: set[str] = set()
    for sz, group in by_size.items():
        if len(group) == 1:
            dedup_out.append(group[0])
            continue
        # Hash each; keep only first occurrence of each unique hash.
        for p in group:
            try:
                h = _md5_file(p)
            except OSError:
                continue
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            dedup_out.append(p)
    dedup_out.sort()
    return dedup_out


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scan(roms: list[Path], workers: int, frames: int, timeout: float,
         progress_every: int, static_cfg: dict | None = None,
         static_timeout: float = 0.0) -> list[dict]:
    if not roms:
        return []

    args = [(str(p), frames, static_cfg) for p in roms]
    # The static-check pass runs inside the same probe_one call, after
    # the initial `frames`-frame check, so its budget has to be added
    # to the per-probe timeout rather than replacing it.
    probe_timeout = timeout + (static_timeout if static_cfg else 0.0)
    results: list[dict] = []

    # Per-task timeout via pool.apply_async + wait(). The probe_one
    # function itself is synchronous; if it wedges on a mapper that
    # spins internally we cap wall time to `timeout` and mark it
    # as a timeout.
    if workers <= 1:
        for i, a in enumerate(args):
            rec = _probe_with_timeout(a, probe_timeout)
            results.append(rec)
            if progress_every and (i + 1) % progress_every == 0:
                _print_progress(i + 1, len(args), results)
    else:
        # Use spawn so each child has a pristine nes_core import; fork
        # would share the parent's nes_core module state and defeat
        # the isolation we want.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            # imap_unordered for streaming progress; wrap each task
            # in a per-child guard so a stuck probe doesn't hold the
            # whole pool. Pool workers themselves handle SIGALRM.
            async_results = [
                pool.apply_async(_probe_with_timeout, (a, probe_timeout))
                for a in args
            ]
            for i, ar in enumerate(async_results):
                try:
                    rec = ar.get(timeout=probe_timeout * 2 + 10)
                except mp.TimeoutError:
                    rec = {
                        "path": args[i][0],
                        "name": os.path.basename(args[i][0]),
                        "bytes": 0, "md5": "", "mapper": -1, "sub_mapper": 0,
                        "nes20": False,
                        "status": "timeout",
                        "frames_run": 0,
                        "wall_ms": probe_timeout * 1000.0,
                        "error": "worker exceeded timeout",
                        "motion": "", "static_distinct_hashes": "",
                        "static_first_change_frame": "",
                    }
                results.append(rec)
                if progress_every and (i + 1) % progress_every == 0:
                    _print_progress(i + 1, len(args), results)

    return results


def _probe_with_timeout(a: tuple, timeout: float) -> dict:
    """Set a SIGALRM to kill a wedged probe; falls back to the record
    emitted by probe_one if it returned normally."""
    def _alarm(_sig, _frm):
        raise TimeoutError(f"probe exceeded {timeout}s")

    had_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return probe_one(a)
    except TimeoutError as e:
        return {
            "path": a[0],
            "name": os.path.basename(a[0]),
            "bytes": 0, "md5": "", "mapper": -1, "sub_mapper": 0,
            "nes20": False,
            "status": "timeout",
            "frames_run": 0,
            "wall_ms": timeout * 1000.0,
            "error": str(e),
            "motion": "", "static_distinct_hashes": "",
            "static_first_change_frame": "",
        }
    except BaseException as e:
        return {
            "path": a[0],
            "name": os.path.basename(a[0]),
            "bytes": 0, "md5": "", "mapper": -1, "sub_mapper": 0,
            "nes20": False,
            "status": "step_panic",
            "frames_run": 0,
            "wall_ms": 0.0,
            "error": _short(f"{type(e).__name__}: {e}"),
            "motion": "", "static_distinct_hashes": "",
            "static_first_change_frame": "",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, had_handler)


def _print_progress(done: int, total: int, results: list[dict]) -> None:
    status_counts = Counter(r["status"] for r in results)
    pct = 100.0 * done / total
    ok = status_counts.get("ok", 0)
    print(
        f"[{done:>5}/{total}  {pct:5.1f}%]  ok={ok}  "
        f"fail={done - ok}  latest={results[-1]['name'][:40]}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Output.
# -----------------------------------------------------------------------------

def write_json(results: list[dict], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))


def write_csv(results: list[dict], path: Path) -> None:
    fields = ["path", "name", "bytes", "md5", "mapper", "sub_mapper",
              "nes20", "status", "frames_run", "wall_ms", "error",
              "motion", "static_distinct_hashes", "static_first_change_frame"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})


def write_summary(results: list[dict], path: Path,
                  supported_mappers: set[int] | None = None) -> None:
    total = len(results)
    by_status = Counter(r["status"] for r in results)
    # Mapper coverage: per-mapper ok% (only counts ROMs where
    # header_parse succeeded, i.e. mapper is known).
    by_mapper: dict[int, Counter] = defaultdict(Counter)
    for r in results:
        if r["mapper"] >= 0:
            by_mapper[r["mapper"]][r["status"]] += 1

    # Error buckets: group by first 80 chars of error to spot
    # recurring panic patterns.
    err_buckets: Counter = Counter()
    for r in results:
        if r["status"] != "ok":
            err_buckets[(r["status"], _short(r["error"], 80))] += 1

    # Unsupported-mapper gap — what's in the user's library that we
    # don't handle yet. Sorted by cart count so the top item is the
    # highest-leverage mapper to add next.
    unsupported_gap: Counter = Counter()
    if supported_mappers is not None:
        for m, c in by_mapper.items():
            if m not in supported_mappers:
                unsupported_gap[m] = sum(c.values())

    lines: list[str] = []
    lines.append("# ROM library compatibility scan\n")
    lines.append(f"Total ROMs: **{total}**  \n")
    lines.append(f"Pass: **{by_status.get('ok', 0)}** "
                 f"({100.0 * by_status.get('ok', 0) / max(total, 1):.1f}%)  \n\n")

    lines.append("## Status breakdown\n")
    lines.append("| status | count | % |\n|---|---:|---:|\n")
    for s, c in by_status.most_common():
        lines.append(f"| `{s}` | {c} | {100.0 * c / max(total, 1):.1f}% |\n")

    # Static-screen check, only meaningful for `ok` records, and only
    # populated when the scan ran with --static-check (the default).
    motion_counts = Counter(r.get("motion", "") for r in results if r["status"] == "ok")
    checked = sum(v for k, v in motion_counts.items() if k in ("live", "static", "check_err"))
    if checked:
        live = motion_counts.get("live", 0)
        static = motion_counts.get("static", 0)
        check_err = motion_counts.get("check_err", 0)
        lines.append("\n## Static-screen check\n")
        lines.append(f"Of {checked} `ok` ROMs re-probed with Start/A bursts + random "
                     f"input: **{live} live**, **{static} static** (frozen screen, "
                     f"`ok` but not actually booting), {check_err} check_err.  \n\n")
        if static:
            lines.append("| ROM | mapper | distinct hashes |\n|---|---:|---:|\n")
            for r in sorted(results, key=lambda r: r["name"]):
                if r["status"] == "ok" and r.get("motion") == "static":
                    lines.append(f"| {r['name']} | {r['mapper']} | "
                                 f"{r.get('static_distinct_hashes', '')} |\n")

    if unsupported_gap:
        lines.append("\n## Missing mapper support (top-leverage gaps)\n")
        lines.append("Carts in your library whose mapper is **not** in "
                     "`nes_core.supported_mappers()`. Adding the top entry "
                     "unblocks the most ROMs per unit of work.\n\n")
        lines.append("| mapper | affected carts |\n|---:|---:|\n")
        for m, c in unsupported_gap.most_common(20):
            lines.append(f"| {m} | {c} |\n")

    lines.append("\n## Mapper coverage (known-header ROMs)\n")
    lines.append("| mapper | total | ok | ok% | fail | supported? |\n"
                 "|---:|---:|---:|---:|---:|:---:|\n")
    for m in sorted(by_mapper.keys()):
        c = by_mapper[m]
        t = sum(c.values())
        ok = c.get("ok", 0)
        supp = "✓" if (supported_mappers and m in supported_mappers) else "—"
        lines.append(f"| {m} | {t} | {ok} | {100.0 * ok / t:.1f}% | {t - ok} | {supp} |\n")

    lines.append("\n## Top error buckets\n")
    lines.append("| status | error prefix | count |\n|---|---|---:|\n")
    for (status, err), c in err_buckets.most_common(25):
        lines.append(f"| `{status}` | `{err or '(empty)'}` | {c} |\n")

    path.write_text("".join(lines))


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[1]
    p.add_argument("--roms-dir", type=Path, default=repo / "roms",
                   help="directory to walk for .nes files (recursive)")
    p.add_argument("--out", type=Path, default=repo / "scan_results",
                   help="output path prefix (writes .json, .csv, .md)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="parallel probe workers (default: half-of-cpus)")
    p.add_argument("--frames", type=int, default=300,
                   help="frames to run per ROM after reset (default: 300)")
    p.add_argument("--timeout", type=float, default=15.0,
                   help="seconds before a probe is killed and marked timeout")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N ROMs (0=all)")
    p.add_argument("--progress-every", type=int, default=25,
                   help="print progress every N probes (0=off)")
    p.add_argument("--no-dedup", action="store_true",
                   help="skip MD5-based deduplication (default: dedup on)")
    p.add_argument("--static-check", dest="static_check", action="store_true",
                   default=True,
                   help="re-probe every `ok` ROM for a frozen screen (default: on)")
    p.add_argument("--no-static-check", dest="static_check", action="store_false",
                   help="skip the static-screen re-probe (restores old `ok`-only behavior)")
    p.add_argument("--static-frames", type=int, default=3000,
                   help="frames to run in the static-screen re-probe (default: 3000)")
    p.add_argument("--static-timeout", type=float, default=20.0,
                   help="extra seconds allowed for the static-check pass, "
                        "added to --timeout (default: 20)")
    p.add_argument("--static-period", type=int, default=120,
                   help="input-schedule period in frames (default: 120)")
    p.add_argument("--static-start-burst-len", type=int, default=8,
                   help="Start held for this many frames at the top of each period (default: 8)")
    p.add_argument("--static-a-burst-start", type=int, default=60,
                   help="A burst begins this many frames into each period (default: 60)")
    p.add_argument("--static-a-burst-len", type=int, default=4,
                   help="A held for this many frames per period (default: 4)")
    p.add_argument("--static-random-from", type=int, default=1500,
                   help="frame index after which random input can fire (default: 1500)")
    p.add_argument("--static-random-prob", type=float, default=0.5,
                   help="per-frame probability of a random action once past "
                        "--static-random-from (default: 0.5)")
    p.add_argument("--static-seed", type=int, default=314159265,
                   help="RNG seed for the random-input portion of the schedule, "
                        "same for every ROM (default: 314159265, matches the "
                        "ground-truth executor's hang_reprobe.py)")
    args = p.parse_args()

    if not args.roms_dir.is_dir():
        print(f"error: not a directory: {args.roms_dir}", file=sys.stderr)
        return 2

    static_cfg = None
    if args.static_check:
        static_cfg = {
            "frames": args.static_frames,
            "period": args.static_period,
            "start_burst_len": args.static_start_burst_len,
            "a_burst_start": args.static_a_burst_start,
            "a_burst_len": args.static_a_burst_len,
            "random_from": args.static_random_from,
            "random_prob": args.static_random_prob,
            "seed": args.static_seed,
        }

    print(f"scanning {args.roms_dir} …", flush=True)
    roms = discover(args.roms_dir, dedup=not args.no_dedup)
    if args.limit > 0:
        roms = roms[:args.limit]
    static_msg = (f"static-check={args.static_frames}f" if static_cfg
                  else "static-check=off")
    print(f"found {len(roms)} unique .nes files; workers={args.workers} "
          f"frames={args.frames} timeout={args.timeout}s {static_msg}", flush=True)
    if not roms:
        return 0

    t0 = time.perf_counter()
    results = scan(roms, args.workers, args.frames, args.timeout,
                   args.progress_every, static_cfg=static_cfg,
                   static_timeout=args.static_timeout)
    wall = time.perf_counter() - t0

    # Pull the supported-mapper set from nes_core so the summary can
    # flag library gaps. Safe if the module fails to import — just
    # skips the gap section.
    supported: set[int] | None = None
    try:
        import nes_core  # type: ignore
        supported = set(nes_core.supported_mappers())
    except Exception:
        pass

    out_base = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    write_json(results, out_base.with_suffix(".json"))
    write_csv(results, out_base.with_suffix(".csv"))
    write_summary(results, out_base.with_suffix(".md"), supported_mappers=supported)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\ndone in {wall:.1f}s — {ok}/{len(results)} ok "
          f"({100.0 * ok / len(results):.1f}%)", flush=True)
    if static_cfg:
        live = sum(1 for r in results if r.get("motion") == "live")
        static = sum(1 for r in results if r.get("motion") == "static")
        print(f"  static-check: {live} live, {static} static")
    print(f"  {out_base.with_suffix('.json')}")
    print(f"  {out_base.with_suffix('.csv')}")
    print(f"  {out_base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
