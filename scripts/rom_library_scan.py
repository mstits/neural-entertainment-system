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
                 status,frames_run,wall_ms,error
  <out>.md     — summary: per-mapper compatibility, top panic classes,
                 failure buckets.

Status values:
  ok                — loaded, reset, ran `frames` frames cleanly
  header_parse_err  — iNES header malformed or mapper unknown
  load_err          — header OK but env construction failed
  reset_err         — reset() panicked
  step_panic        — step() panicked (records frame index)
  timeout           — ran past --timeout seconds without finishing
  io_err            — couldn't read file

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
    rom_path, frames = args
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
    rec["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    return rec


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
         progress_every: int) -> list[dict]:
    if not roms:
        return []

    args = [(str(p), frames) for p in roms]
    results: list[dict] = []

    # Per-task timeout via pool.apply_async + wait(). The probe_one
    # function itself is synchronous; if it wedges on a mapper that
    # spins internally we cap wall time to `timeout` and mark it
    # as a timeout.
    if workers <= 1:
        for i, a in enumerate(args):
            rec = _probe_with_timeout(a, timeout)
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
                pool.apply_async(_probe_with_timeout, (a, timeout)) for a in args
            ]
            for i, ar in enumerate(async_results):
                try:
                    rec = ar.get(timeout=timeout * 2 + 10)
                except mp.TimeoutError:
                    rec = {
                        "path": args[i][0],
                        "name": os.path.basename(args[i][0]),
                        "bytes": 0, "md5": "", "mapper": -1, "sub_mapper": 0,
                        "nes20": False,
                        "status": "timeout",
                        "frames_run": 0,
                        "wall_ms": timeout * 1000.0,
                        "error": "worker exceeded timeout",
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
              "nes20", "status", "frames_run", "wall_ms", "error"]
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
    args = p.parse_args()

    if not args.roms_dir.is_dir():
        print(f"error: not a directory: {args.roms_dir}", file=sys.stderr)
        return 2

    print(f"scanning {args.roms_dir} …", flush=True)
    roms = discover(args.roms_dir, dedup=not args.no_dedup)
    if args.limit > 0:
        roms = roms[:args.limit]
    print(f"found {len(roms)} unique .nes files; workers={args.workers} "
          f"frames={args.frames} timeout={args.timeout}s", flush=True)
    if not roms:
        return 0

    t0 = time.perf_counter()
    results = scan(roms, args.workers, args.frames, args.timeout,
                   args.progress_every)
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
    print(f"  {out_base.with_suffix('.json')}")
    print(f"  {out_base.with_suffix('.csv')}")
    print(f"  {out_base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
