"""Library-wide lockstep RAM-divergence sweep across every .nes ROM.

For each ROM we:
  1. Try to instantiate both nes_core and nes-py.
  2. Run `--frames` cold-boot idle frames on each (button=0).
  3. Compare 2 KB of CPU RAM.
  4. Bucket the result.

Output:
  - JSON: one record per ROM with its divergence count + bucket + classification.
  - Console summary table: how many ROMs in each bucket.

Buckets:
  "byte_exact"     RAM matches nes-py byte-for-byte. Strongest correctness guarantee.
  "tight"          1-5 bytes drift. Cycle-accuracy edge cases; gameplay typically fine.
  "moderate"       6-50 bytes drift.
  "loose"          51-500 bytes drift.
  "wide"           >500 bytes drift.
  "ours_panic"     nes_core failed to load/step (mapper panic, ROM corruption).
  "theirs_unsupported"  nes-py unsupported mapper (UxROM/CNROM/SxROM/NROM only).
  "both_failed"

The harness mirrors `tests/parity/lockstep.py::_load_theirs` — nes-py
gets a compensating extra `step(0)` after reset to align phase with
nes_core's reset-warmup behavior.

Usage:
    python scripts/parity_sweep.py --frames 120 --out parity_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROMS_DIR = REPO / "roms"


def classify(diff: int | None, error: str | None) -> str:
    if error == "ours_panic":
        return "ours_panic"
    if error == "theirs_unsupported":
        return "theirs_unsupported"
    if error == "both":
        return "both_failed"
    if diff is None:
        return "unknown"
    if diff == 0:
        return "byte_exact"
    if diff <= 5:
        return "tight"
    if diff <= 50:
        return "moderate"
    if diff <= 500:
        return "loose"
    return "wide"


def _suppress_cpp_stderr():
    """LaiNES (nes-py's C++ core) prints 'failed to execute opcode: ff' to
    stderr for ROMs with unimplemented opcodes (illegal/undocumented).
    These aren't crashes — emulation continues — but they spam the log.
    Redirect fd 2 once at sweep start so the per-ROM run stays clean.
    Caller restores via the returned dup'd fd if needed."""
    import os, sys
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


def run_one(rom_path: Path, frames: int) -> dict:
    """Run lockstep on a single ROM. Returns a result record."""
    rec: dict = {"rom": rom_path.name}

    # nes_core
    try:
        import nes_core
        ours = nes_core.NESEnvironment(str(rom_path), frame_skip=1)
        ours.reset()
    except Exception as e:
        rec["error"] = "ours_panic"
        rec["error_msg"] = str(e)[:140]
        rec["bucket"] = classify(None, "ours_panic")
        return rec

    # nes-py
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import nes_py
        theirs = nes_py.NESEnv(str(rom_path))
        theirs.reset()
        theirs.step(0)  # phase compensation; matches lockstep harness
    except Exception as e:
        rec["error"] = "theirs_unsupported"
        rec["error_msg"] = str(e)[:140]
        rec["bucket"] = classify(None, "theirs_unsupported")
        try:
            ours.close()
        except Exception:
            pass
        return rec

    # Step both
    try:
        for _ in range(frames):
            ours.step(0)
            theirs.step(0)
        ours_ram = bytes(ours.get_ram_range(0, 0x0800))
        theirs_ram = bytes(theirs.ram)
        diff = sum(1 for i in range(0x800) if ours_ram[i] != theirs_ram[i])
        rec["diff"] = diff
        rec["bucket"] = classify(diff, None)
    except Exception as e:
        rec["error"] = "step_failure"
        rec["error_msg"] = str(e)[:140]
        rec["bucket"] = "ours_panic"
    finally:
        try:
            ours.close()
        except Exception:
            pass
        try:
            theirs.close()
        except Exception:
            pass

    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120,
                        help="Cold-boot idle frames to step before comparing.")
    parser.add_argument("--out", type=Path, default=REPO / "parity_sweep.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only sweep first N ROMs (for quick test).")
    parser.add_argument("--rom", action="append", default=None,
                        help="Sweep only the specified ROM(s) (can repeat).")
    args = parser.parse_args()

    if args.rom:
        rom_paths = [ROMS_DIR / r for r in args.rom]
    else:
        rom_paths = sorted(ROMS_DIR.glob("*.nes"))
    if args.limit > 0:
        rom_paths = rom_paths[: args.limit]
    # ROMs known to C-level segfault nes-py. Rather than have them take
    # down the whole sweep process (Python try/except can't catch SIGSEGV),
    # record them as 'theirs_unsupported' and move on. Add ROMs here as
    # they're discovered; each one is a nes-py bug, not ours.
    NESPY_SEGFAULT_ROMS = {
        "Sesame Street ABC & 123 (USA).nes",
        "Rygar (USA).nes",
    }

    print(f"sweeping {len(rom_paths)} ROMs for {args.frames} idle frames each",
          flush=True)
    _suppress_cpp_stderr()  # LaiNES opcode spam
    t0 = time.perf_counter()
    results: list[dict] = []
    # Resume support: if output exists, pick up where we left off.
    done_roms: set[str] = set()
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
            results.extend(prev)
            done_roms = {r["rom"] for r in prev}
            print(f"resuming: {len(done_roms)} ROMs already done")
        except Exception:
            pass
    for i, rom in enumerate(rom_paths, 1):
        if rom.name in done_roms:
            continue
        if rom.name in NESPY_SEGFAULT_ROMS:
            rec = {
                "rom": rom.name,
                "error": "theirs_unsupported",
                "error_msg": "nes-py C-level segfault on this ROM; skipped",
                "bucket": "theirs_unsupported",
            }
        else:
            rec = run_one(rom, args.frames)
        results.append(rec)
        # Checkpoint EVERY ROM: nes-py has latent C-level bugs (observed
        # SIGSEGV at various points in long runs). Cheaper to write a
        # 100KB JSON per ROM than re-run 700 ROMs after a segfault.
        args.out.write_text(json.dumps(results, indent=2))
        if i % 25 == 0 or i == len(rom_paths):
            el = time.perf_counter() - t0
            rate = (i - len(done_roms)) / el if el > 0 else 0
            eta = (len(rom_paths) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(rom_paths)}] {rec['rom'][:50]:50s} "
                  f"bucket={rec['bucket']:18s} ({rate:.1f}/s, eta {eta:.0f}s)",
                  flush=True)

    args.out.write_text(json.dumps(results, indent=2))
    el = time.perf_counter() - t0
    print(f"\nwrote {args.out} in {el:.1f}s")

    # Summary
    from collections import Counter
    buckets = Counter(r["bucket"] for r in results)
    print("\nbucket summary:")
    for b in [
        "byte_exact", "tight", "moderate", "loose", "wide",
        "theirs_unsupported", "ours_panic", "both_failed", "unknown",
    ]:
        n = buckets.get(b, 0)
        if n:
            print(f"  {b:22s} {n:4d}")
    n_compared = sum(1 for r in results if "diff" in r)
    n_byte_exact = buckets.get("byte_exact", 0)
    n_tight = buckets.get("tight", 0)
    if n_compared:
        print(f"\nof {n_compared} compared:")
        print(f"  byte-exact:   {n_byte_exact:4d} ({100*n_byte_exact/n_compared:.1f}%)")
        print(f"  byte-exact + tight: {n_byte_exact + n_tight:4d} "
              f"({100*(n_byte_exact + n_tight)/n_compared:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
