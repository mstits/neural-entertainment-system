#!/usr/bin/env python3
"""End-to-end pool test: PlayWindow-style binary state file loaded by
ParallelPool workers via the standard env_spec mechanism."""

import os
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nes_core
from src.emulation.parallel_pool import ParallelPool


def main() -> None:
    # Build an NCST file by playing 100 frames of pressing Start
    env = nes_core.NESEnvironment("roms/zelda.nes", frame_skip=1)
    env.reset()
    for _ in range(100):
        env.step(0x10)
    blob = bytes(env.save_state())
    print(f"saved state blob: {len(blob)} bytes")

    fd, path = tempfile.mkstemp(suffix=".state.bin")
    os.close(fd)
    Path(path).write_bytes(b"NCST\x01" + blob)

    try:
        pool = ParallelPool(
            rom_path="roms/zelda.nes",
            num_workers=4,
            frame_skip=4,
            env_spec="nes_core:NESEnvironment",
            start_state_path=path,
        )
        pool.start()
        try:
            results = pool.reset_all()
            for r in results:
                assert r.frame.shape == (240, 256, 3)
                assert len(r.ram_snapshot) == 2048
                assert not r.done
            print(f"all 4 workers loaded NCST state and emitted first frame")

            t0 = time.perf_counter()
            for _ in range(100):
                pool.step_all([0, 0, 0, 0])
            dur = time.perf_counter() - t0
            print(
                f"100 pool steps in {dur:.2f}s = {100*4/dur:.1f} steps/s combined "
                f"— no crashes"
            )
        finally:
            pool.shutdown()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
