"""Single-tape runner. Exit 0 on pass, 1 on fail.

    python -m tests.parity.run --tape <path> [--verbose]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tests.parity.diff import (
    dump_failure_cross,
    dump_failure_golden,
    frame_hash,
    scanline_divergence,
)
from tests.parity.drivers import NESCoreDriver, NESPyDriver
from tests.parity.tape import Tape, TapeError

REPO = Path(__file__).resolve().parents[2]
FAILURE_DIR = Path(__file__).parent / "failures"


def run(tape_path: Path, verbose: bool = False) -> int:
    try:
        tape = Tape.load(tape_path)
    except TapeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    buttons = tape.button_sequence()
    t0 = time.perf_counter()

    if tape.mode == "cross_emulator":
        return _run_cross(tape, buttons, verbose, t0)
    else:
        return _run_golden(tape, buttons, verbose, t0)


def _run_cross(tape: Tape, buttons: np.ndarray, verbose: bool, t0: float) -> int:
    ours_driver = NESCoreDriver(tape.rom)
    theirs_driver = NESPyDriver(tape.rom)
    tol_scan = tape.tolerance_scanlines_diverged
    tol_px = tape.tolerance_pixels_per_scanline

    for i, btn in enumerate(buttons):
        ours_driver.step(int(btn))
        theirs_driver.step(int(btn))
        if i < tape.warmup_frames:
            continue
        a = ours_driver.frame()
        b = theirs_driver.frame()
        div = scanline_divergence(a, b)
        diverged_scans = int((div > tol_px).sum())
        if diverged_scans > tol_scan:
            diff_path = dump_failure_cross(FAILURE_DIR, f"{tape.name}_{i:04d}", a, b)
            elapsed = time.perf_counter() - t0
            print(
                f"{tape.name} [cross_emulator]: frame {i}: "
                f"{diverged_scans} scanlines diverged "
                f"(max {int(div.max())} px/scanline) → {diff_path} ({elapsed:.1f}s)"
            )
            return 1
        if verbose and (i + 1) % 60 == 0:
            print(f"[{i + 1}/{tape.frames}] OK")

    elapsed = time.perf_counter() - t0
    print(f"{tape.name} [cross_emulator]: PASS ({tape.frames} frames, {elapsed:.1f}s)")
    return 0


def _run_golden(tape: Tape, buttons: np.ndarray, verbose: bool, t0: float) -> int:
    assert tape.golden_hashes is not None
    ours_driver = NESCoreDriver(tape.rom)

    for i, btn in enumerate(buttons):
        ours_driver.step(int(btn))
        if i < tape.warmup_frames:
            continue
        a = ours_driver.frame()
        actual = frame_hash(a)
        expected = int(tape.golden_hashes[i])
        if actual != expected:
            ours_path = dump_failure_golden(
                FAILURE_DIR, f"{tape.name}_{i:04d}", a, expected, actual,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"{tape.name} [golden_hash]: frame {i}: "
                f"hash mismatch (expected {expected:016x}, got {actual:016x}) "
                f"→ {ours_path} ({elapsed:.1f}s)"
            )
            return 1
        if verbose and (i + 1) % 60 == 0:
            print(f"[{i + 1}/{tape.frames}] OK")

    elapsed = time.perf_counter() - t0
    print(f"{tape.name} [golden_hash]: PASS ({tape.frames} frames, {elapsed:.1f}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a parity tape.")
    parser.add_argument("--tape", required=True, type=Path, help="Tape JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return run(args.tape, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
