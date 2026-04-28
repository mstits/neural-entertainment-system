"""Watchdog sampler for training sessions.

Runs on a daemon thread inside the GUI process and appends JSONL
samples every `interval_s` seconds. Fields are best-effort — a probe
that fails (torch not available, ps returns garbage) is silently
omitted and sampling continues.

When RSS crosses `rss_dump_threshold_mb` or jumps by more than
`rss_growth_dump_mb` between samples, a dump of every Python thread's
stack is appended to the companion `.stacks.txt` file, so we have a
post-mortem frame even if the process later freezes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional


class Watchdog:
    def __init__(
        self,
        log_dir: str | os.PathLike = "logs",
        interval_s: float = 5.0,
        rss_dump_threshold_mb: int = 60_000,
        rss_growth_dump_mb: int = 4_000,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._interval = float(interval_s)
        self._threshold_mb = int(rss_dump_threshold_mb)
        self._growth_mb = int(rss_growth_dump_mb)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sample_path: Optional[Path] = None
        self._stack_path: Optional[Path] = None
        self._last_rss_mb: int = 0
        self._last_stack_dump_ts: float = 0.0
        self._pid = os.getpid()

    def start(self) -> Optional[Path]:
        if self._thread is not None:
            return self._sample_path
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._sample_path = self._log_dir / f"watchdog-{stamp}.jsonl"
        self._stack_path = self._log_dir / f"watchdog-{stamp}.stacks.txt"
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="watchdog", daemon=True
        )
        self._thread.start()
        return self._sample_path

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        start_ts = time.time()
        while not self._stop.is_set():
            sample = self._sample(start_ts)
            try:
                assert self._sample_path is not None
                with self._sample_path.open("a") as fh:
                    fh.write(json.dumps(sample) + "\n")
            except Exception:
                pass
            rss = int(sample.get("rss_mb", 0))
            growth = rss - self._last_rss_mb if self._last_rss_mb else 0
            crossed = rss >= self._threshold_mb
            big_jump = growth >= self._growth_mb and self._last_rss_mb > 0
            if crossed or big_jump:
                self._dump_stacks(
                    reason="threshold" if crossed else "growth",
                    rss=rss,
                    growth=growth,
                )
            if rss:
                self._last_rss_mb = rss
            self._stop.wait(self._interval)

    def _sample(self, start_ts: float) -> dict:
        out: dict = {
            "ts": time.time(),
            "uptime_s": round(time.time() - start_ts, 1),
            "pid": self._pid,
        }
        try:
            proc = subprocess.run(
                ["ps", "-p", str(self._pid), "-o", "rss=,vsz=,%cpu="],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            parts = proc.stdout.strip().split()
            if len(parts) >= 3:
                out["rss_mb"] = int(parts[0]) // 1024
                out["vsz_mb"] = int(parts[1]) // 1024
                out["cpu_pct"] = float(parts[2])
        except Exception:
            pass
        try:
            proc = subprocess.run(
                ["ps", "-M", "-p", str(self._pid)],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            out["threads"] = max(0, len(proc.stdout.strip().splitlines()) - 1)
        except Exception:
            pass
        try:
            out["py_alloc_blocks"] = sys.getallocatedblocks()
        except Exception:
            pass
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                mps = getattr(torch, "mps", None)
                if mps is not None and torch.backends.mps.is_available():
                    out["torch_mps_alloc_mb"] = (
                        mps.current_allocated_memory() // (1024 * 1024)
                    )
                    if hasattr(mps, "driver_allocated_memory"):
                        out["torch_mps_driver_mb"] = (
                            mps.driver_allocated_memory() // (1024 * 1024)
                        )
            except Exception:
                pass
        return out

    def _dump_stacks(self, *, reason: str, rss: int, growth: int) -> None:
        now = time.time()
        if now - self._last_stack_dump_ts < 30.0:
            return
        self._last_stack_dump_ts = now
        try:
            assert self._stack_path is not None
            with self._stack_path.open("a") as fh:
                fh.write(
                    f"\n===== {time.strftime('%H:%M:%S')}  reason={reason}  "
                    f"rss={rss}MB  growth={growth}MB =====\n"
                )
                frames = sys._current_frames()
                names = {t.ident: t.name for t in threading.enumerate()}
                for tid, frame in frames.items():
                    name = names.get(tid, "?")
                    fh.write(f"\n--- thread id={tid} name={name} ---\n")
                    traceback.print_stack(frame, file=fh)
        except Exception:
            pass
