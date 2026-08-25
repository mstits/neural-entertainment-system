"""IS-1a byte-identity driver: runs go_explore_solve.py's real main()
against a MONKEYPATCHED, deterministic fake clock (fixed delta per
time.time() call) so the wall-clock deadline crosses after an EXACT,
reproducible number of engine steps -- independent of real CPU
contention from any other process on this machine (v28 training).
This is the only way to get a true step-for-step comparable run
between the pre-graft and post-graft code without relying on wall-clock
timing, which is not deterministic under system load.

Not part of the shipped engine; a one-shot IS-1a receipt driver.
Not committed (orchestrator commits).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent


class FakeTime:
    """Drop-in replacement for the stdlib `time` module inside the
    target script's namespace ONLY (rebound via `module.time = ...`
    after import, never a global monkeypatch of the real `time`
    module) -- every call to time.time() advances a fixed virtual
    delta, so the wall-clock deadline the real explore() loop checks
    crosses after a fixed, real-CPU-speed-independent number of calls.
    """

    def __init__(self, dt: float = 0.01):
        self.t = 0.0
        self.dt = dt
        self.n_calls = 0

    def time(self):
        self.t += self.dt
        self.n_calls += 1
        return self.t

    def sleep(self, s):
        self.t += s

    def strftime(self, fmt):
        import time as _real_time
        return _real_time.strftime(fmt, _real_time.gmtime(self.t))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(path: Path, name: str, out_dir: Path, *, minutes: float, dt: float) -> dict:
    mod = load_module(path, name)
    fake = FakeTime(dt=dt)
    mod.time = fake  # rebind the module-global `time` name only
    argv = [
        str(path),
        "--out", str(out_dir),
        "--root-state", "roms/zelda_start_ctrl.state.bin",
        "--profile", "configs/zelda_roomfp.yaml",
        "--workers", "1", "--minutes", str(minutes),
        "--seed", "0", "--want-solutions", "0",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = mod.main()
    finally:
        sys.argv = old_argv
    return {"rc": rc, "fake_time_calls": fake.n_calls, "fake_t_final": fake.t}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("pre", "post"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.01)
    args = ap.parse_args()

    if args.which == "pre":
        path = REPO / "scripts" / "_is1a_pregraft_baseline.py"
        name = "is1a_pregraft"
    else:
        path = REPO / "scripts" / "go_explore_solve.py"
        name = "is1a_postgraft"
    import os
    os.chdir(REPO)
    info = run(path, name, Path(args.out), minutes=args.minutes, dt=args.dt)
    print(f"[byteid] {args.which}: {info}")


if __name__ == "__main__":
    main()
