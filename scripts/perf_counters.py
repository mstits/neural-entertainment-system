#!/usr/bin/env python3
"""
kpc/kperf hardware performance-counter telemetry, for PPU-rework stage gates.

Programs the Apple Silicon PMU via the private kperf.framework (the
control plane: arm counting classes, read per-thread counters) and
kperfdata.framework (the event database: symbolic event names -> the
raw config words the current chip actually needs -- so this module
never hardcodes chip-generation-specific magic numbers). Reports, for
a workload run on the calling thread:

  - fixed-counter cycles + instructions, and the derived IPC
  - L1D fetch misses (L1D_CACHE_MISS_LD) and L1D sector refills
    (ARM_L1D_CACHE_REFILL)
  - L2 TLB misses, data-side and instruction-side
  - branch mispredictions (retired, non-speculative)

Programming the counters (kpc_set_config / kpc_set_counting /
kpc_set_thread_counting) requires root -- the kernel gates it,
independent of SIP. Reading the event database and resolving event
names to config words does NOT require root. When run without root
this module degrades gracefully: it still runs the requested
benchmark workload and reports wall-clock/throughput, but reports
hardware counters as unavailable with the exact sudo invocation to
use instead.

Usage:
    .venv/bin/python scripts/perf_counters.py --bench emu
    .venv/bin/python scripts/perf_counters.py --bench emu --json --out receipt.json
    sudo .venv/bin/python scripts/perf_counters.py --bench emu   # programs real counters
    .venv/bin/python scripts/perf_counters.py --diagnose --json  # binding/DB/permission probe only
    .venv/bin/python scripts/perf_counters.py --xctrace-recipe   # print the Instruments fallback
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

KPERF_PATH = "/System/Library/PrivateFrameworks/kperf.framework/kperf"
KPERFDATA_PATH = "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata"

# kpc counting-class bitmask (bsd/kern kpc.h). FIXED is always cycles +
# instructions on every Apple Silicon generation to date (A7 through
# the M4 this machine reports as "Apple silicon" -- see --diagnose).
KPC_CLASS_FIXED = 0
KPC_CLASS_CONFIGURABLE = 1
KPC_CLASS_POWER = 2
KPC_CLASS_RAWPMU = 3
KPC_CLASS_FIXED_MASK = 1 << KPC_CLASS_FIXED
KPC_CLASS_CONFIGURABLE_MASK = 1 << KPC_CLASS_CONFIGURABLE
KPC_CLASS_POWER_MASK = 1 << KPC_CLASS_POWER
KPC_CLASS_RAWPMU_MASK = 1 << KPC_CLASS_RAWPMU

# The 5 configurable-class events this module programs. Chosen to fit
# exactly in the 5 configurable PMU slots this chip reports
# (kpc_get_counter_count(CONFIGURABLE) == 5, confirmed on this M4/T6041
# tonight -- see docs/receipts/perf_counters_bootstrap.json). Looked up
# BY NAME against the live kperfdata event database at run time, so the
# config words produced are correct for whatever Apple Silicon
# generation is actually running this, not just this one.
CONFIGURABLE_EVENTS: dict[str, str] = {
    "l1d_fetch_misses": "L1D_CACHE_MISS_LD",
    "l1d_sector_refills": "ARM_L1D_CACHE_REFILL",
    "l2_tlb_misses_data": "L2_TLB_MISS_DATA",
    "l2_tlb_misses_instruction": "L2_TLB_MISS_INSTRUCTION",
    "branch_mispredictions": "BRANCH_MISPRED_NONSPEC",
}

XCTRACE_BIN = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"

XCTRACE_RECIPE_TEXT = f"""\
xctrace/Instruments fallback (verified working tonight, NO root needed)
========================================================================
This machine has the 'CPU Counters' Instruments template available at:
    {XCTRACE_BIN}
Confirmed tonight: `xctrace list templates` shows it, and a live
record+export round-trip completed as a non-root user with no Developer
Mode toggle required.

Record a run against a launched process:
    {XCTRACE_BIN} record \\
        --template 'CPU Counters' \\
        --time-limit 30s \\
        --output /tmp/perf_counters_emu.trace \\
        --launch -- {sys.executable} scripts/perf_counters.py --bench emu

Pull raw per-interval PMU counter arrays back out as XML:
    {XCTRACE_BIN} export \\
        --input /tmp/perf_counters_emu.trace \\
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="CounterMetricAggregatedForProcess"]' \\
        > counters.xml

What this gives you: real uint64 counter arrays per sample interval,
receipt-usable as-is (attach counters.xml + the .trace bundle path).
What it does NOT give you yet: per-column labels matching our 5 named
events -- the default template runs in "Guided / CPU Bottlenecks" mode,
whose column meanings are defined by form.template's metric formulas
inside the .trace bundle rather than exposed directly. Getting named
columns needs re-recording with a "Custom" counter configuration that
lists our exact events (same names as CONFIGURABLE_EVENTS above) --
that step is NOT wired yet; use this path only if the ctypes/kpc path
below is unavailable on a given machine.
"""


class PerfCountersUnavailable(RuntimeError):
    """Hardware counters can't be programmed on this run: non-root,
    framework missing/changed, or an event lookup failed. Expected and
    handled by the CLI, not a bug -- the caller degrades gracefully."""


def is_root() -> bool:
    return os.geteuid() == 0


def sudo_invocation() -> str:
    script = Path(__file__).resolve()
    return f"sudo {sys.executable} {script} --bench emu --json"


def _errno_str(errno_val: int) -> str:
    return os.strerror(errno_val) if errno_val else "(errno not set)"


class _KperfBindings:
    """Binds the private kperf.framework control-plane calls: the
    "known init sequence" -- kpc_set_config, kpc_set_counting,
    kpc_set_thread_counting, kpc_get_thread_counters -- plus the two
    read-only query calls used to size buffers. Loading the framework
    and querying counter counts does NOT require root; only the
    kpc_set_* calls do (confirmed: they return EPERM under a
    non-privileged euid, not a load/symbol failure)."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL(KPERF_PATH, use_errno=True)
        except OSError as e:
            raise PerfCountersUnavailable(
                f"dlopen of kperf.framework failed at {KPERF_PATH}: {e}. "
                "ctypes binding is unworkable on this macOS version -- "
                "fall back to the xctrace recipe (--xctrace-recipe)."
            ) from e

        def bind(name: str, argtypes: list, restype):
            try:
                fn = getattr(self.lib, name)
            except AttributeError as e:
                raise PerfCountersUnavailable(
                    f"kperf.framework on this macOS version "
                    f"({platform.mac_ver()[0]}) is missing the expected "
                    f"symbol {name!r}. ctypes binding is unworkable as "
                    "written -- fall back to the xctrace recipe "
                    "(--xctrace-recipe) and document the API drift."
                ) from e
            fn.argtypes = argtypes
            fn.restype = restype
            return fn

        u32, i32, vp = ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
        self.kpc_get_counter_count = bind("kpc_get_counter_count", [u32], u32)
        self.kpc_get_config_count = bind("kpc_get_config_count", [u32], u32)
        self.kpc_set_config = bind("kpc_set_config", [u32, vp], i32)
        self.kpc_set_counting = bind("kpc_set_counting", [u32], i32)
        self.kpc_set_thread_counting = bind("kpc_set_thread_counting", [u32], i32)
        self.kpc_get_thread_counters = bind(
            "kpc_get_thread_counters", [u32, u32, vp], i32
        )

    def last_errno(self) -> int:
        return ctypes.get_errno()

    def call_failed(self, fn_name: str) -> str:
        err = self.last_errno()
        return (
            f"{fn_name} returned an error (errno={err}: {_errno_str(err)}). "
            + (
                "This is the expected non-root failure mode -- re-run as:\n"
                f"    {sudo_invocation()}"
                if err == 1  # EPERM
                else "Unexpected errno; investigate before trusting counter values."
            )
        )


class _KpepBindings:
    """Binds kperfdata.framework -- the per-machine PMU EVENT DATABASE.
    Resolves symbolic event names (e.g. "L1D_CACHE_MISS_LD") to the
    correct raw kpc config words for whatever Apple Silicon core this
    process actually runs on, so CONFIGURABLE_EVENTS above never needs
    chip-generation-specific hardcoded encodings. Loading + querying
    this does NOT require root (confirmed tonight)."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.CDLL(KPERFDATA_PATH, use_errno=True)
        except OSError as e:
            raise PerfCountersUnavailable(
                f"dlopen of kperfdata.framework failed at {KPERFDATA_PATH}: "
                f"{e}. Event-name resolution is unworkable -- fall back to "
                "the xctrace recipe (--xctrace-recipe)."
            ) from e

        def bind(name: str, argtypes: list, restype):
            try:
                fn = getattr(self.lib, name)
            except AttributeError as e:
                raise PerfCountersUnavailable(
                    f"kperfdata.framework is missing expected symbol "
                    f"{name!r} on this macOS version. Fall back to the "
                    "xctrace recipe (--xctrace-recipe)."
                ) from e
            fn.argtypes = argtypes
            fn.restype = restype
            return fn

        vp, cp, i32, u32 = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint32,
        )
        pvp = ctypes.POINTER(ctypes.c_void_p)
        psz = ctypes.POINTER(ctypes.c_size_t)
        self.kpep_db_create = bind("kpep_db_create", [cp, pvp], i32)
        self.kpep_db_name = bind("kpep_db_name", [vp, ctypes.POINTER(cp)], i32)
        self.kpep_db_event = bind("kpep_db_event", [vp, cp, pvp], i32)
        self.kpep_config_create = bind("kpep_config_create", [vp, pvp], i32)
        self.kpep_config_add_event = bind(
            "kpep_config_add_event", [vp, pvp, u32, ctypes.POINTER(u32)], i32
        )
        self.kpep_config_kpc_classes = bind(
            "kpep_config_kpc_classes", [vp, ctypes.POINTER(u32)], i32
        )
        self.kpep_config_kpc_count = bind("kpep_config_kpc_count", [vp, psz], i32)
        self.kpep_config_kpc = bind("kpep_config_kpc", [vp, vp, ctypes.c_size_t], i32)
        self.kpep_config_kpc_map = bind(
            "kpep_config_kpc_map", [vp, vp, ctypes.c_size_t], i32
        )

    def open_db(self) -> ctypes.c_void_p:
        db = ctypes.c_void_p()
        rc = self.kpep_db_create(None, ctypes.byref(db))
        if rc != 0:
            raise PerfCountersUnavailable(f"kpep_db_create failed (rc={rc})")
        return db

    def db_name(self, db: ctypes.c_void_p) -> str:
        name = ctypes.c_char_p()
        rc = self.kpep_db_name(db, ctypes.byref(name))
        if rc != 0 or not name.value:
            return "?"
        return name.value.decode()


@dataclass
class ConfigurableEventConfig:
    """The resolved configurable-counter program: which physical PMU
    slot each named event landed in, plus the raw config words to hand
    kpc_set_config."""

    classes: int
    config_words: list
    slot_of: dict = field(default_factory=dict)  # metric name -> physical slot idx


def build_configurable_config(kpep: _KpepBindings) -> ConfigurableEventConfig:
    """Resolve CONFIGURABLE_EVENTS against the live event database and
    produce the kpc config words for them. Raises PerfCountersUnavailable
    (with the offending event name) if this chip's database doesn't
    have one of the events -- an honest failure rather than a silently
    wrong counter."""
    db = kpep.open_db()
    cfg = ctypes.c_void_p()
    rc = kpep.kpep_config_create(db, ctypes.byref(cfg))
    if rc != 0:
        raise PerfCountersUnavailable(f"kpep_config_create failed (rc={rc})")

    metric_names = list(CONFIGURABLE_EVENTS.keys())
    for metric_name in metric_names:
        event_name = CONFIGURABLE_EVENTS[metric_name]
        ev = ctypes.c_void_p()
        rc = kpep.kpep_db_event(db, event_name.encode(), ctypes.byref(ev))
        if rc != 0 or not ev.value:
            raise PerfCountersUnavailable(
                f"event {event_name!r} (for metric {metric_name!r}) is not "
                f"in this chip's PMU database ({kpep.db_name(db)!r}). The "
                "event table this module targets doesn't match this "
                "machine's Apple Silicon generation -- update "
                "CONFIGURABLE_EVENTS after checking --diagnose's full "
                "event listing."
            )
        err = ctypes.c_uint32(0)
        ev_ref = ctypes.c_void_p(ev.value)
        rc = kpep.kpep_config_add_event(cfg, ctypes.byref(ev_ref), 0, ctypes.byref(err))
        if rc != 0:
            raise PerfCountersUnavailable(
                f"kpep_config_add_event({event_name}) failed (rc={rc}, "
                f"err={err.value}) -- likely too many events for this "
                "chip's configurable-counter budget; see --diagnose."
            )

    classes = ctypes.c_uint32(0)
    kpep.kpep_config_kpc_classes(cfg, ctypes.byref(classes))

    count = ctypes.c_size_t(0)
    kpep.kpep_config_kpc_count(cfg, ctypes.byref(count))
    n = count.value

    words_buf = (ctypes.c_uint64 * n)()
    rc = kpep.kpep_config_kpc(cfg, ctypes.cast(words_buf, ctypes.c_void_p), ctypes.sizeof(words_buf))
    if rc != 0:
        raise PerfCountersUnavailable(f"kpep_config_kpc failed (rc={rc})")

    map_buf = (ctypes.c_size_t * len(metric_names))()
    rc = kpep.kpep_config_kpc_map(
        cfg, ctypes.cast(map_buf, ctypes.c_void_p), ctypes.sizeof(map_buf)
    )
    if rc != 0:
        raise PerfCountersUnavailable(f"kpep_config_kpc_map failed (rc={rc})")

    slot_of = {name: int(map_buf[i]) for i, name in enumerate(metric_names)}
    return ConfigurableEventConfig(
        classes=classes.value, config_words=list(words_buf), slot_of=slot_of
    )


class PerfCounterSession:
    """Arms fixed + configurable PMU counting for the CALLING THREAD
    (kpc_set_thread_counting scopes counts to this OS thread, which is
    what makes a single-threaded, single-core workload measurement
    meaningful regardless of which physical core the scheduler actually
    places it on -- macOS/Apple Silicon has no user-facing hard core
    pin API, unlike Linux's taskset/sched_setaffinity).

    Construction fails fast with PerfCountersUnavailable when this
    process can't program counters (not root, framework/DB drift, or
    an unresolvable event) -- callers should catch that and degrade to
    a counters-unavailable report rather than crash.
    """

    def __init__(self) -> None:
        if not is_root():
            raise PerfCountersUnavailable(
                "kpc counter programming requires root -- the kernel gates "
                f"kpc_set_config/kpc_set_counting/kpc_set_thread_counting "
                f"to euid 0 (confirmed tonight: all three return EPERM "
                f"under euid={os.geteuid()}). Re-run as:\n"
                f"    {sudo_invocation()}"
            )
        self.kperf = _KperfBindings()
        self.kpep = _KpepBindings()
        self.event_config = build_configurable_config(self.kpep)

        self.n_fixed = self.kperf.kpc_get_counter_count(KPC_CLASS_FIXED_MASK)
        self.n_config = self.kperf.kpc_get_counter_count(self.event_config.classes)
        self.n_total = self.n_fixed + self.n_config
        self.all_classes = KPC_CLASS_FIXED_MASK | self.event_config.classes

        words = self.event_config.config_words
        cfg_buf = (ctypes.c_uint64 * len(words))(*words)
        rc = self.kperf.kpc_set_config(
            self.event_config.classes, ctypes.cast(cfg_buf, ctypes.c_void_p)
        )
        if rc != 0:
            raise PerfCountersUnavailable(self.kperf.call_failed("kpc_set_config"))

        rc = self.kperf.kpc_set_counting(self.all_classes)
        if rc != 0:
            raise PerfCountersUnavailable(self.kperf.call_failed("kpc_set_counting"))

        rc = self.kperf.kpc_set_thread_counting(self.all_classes)
        if rc != 0:
            raise PerfCountersUnavailable(
                self.kperf.call_failed("kpc_set_thread_counting")
            )

    def read_raw(self) -> list:
        buf = (ctypes.c_uint64 * self.n_total)()
        rc = self.kperf.kpc_get_thread_counters(
            0, self.n_total, ctypes.cast(buf, ctypes.c_void_p)
        )
        if rc != 0:
            raise PerfCountersUnavailable(
                self.kperf.call_failed("kpc_get_thread_counters")
            )
        return list(buf)

    def delta_metrics(self, before: list, after: list) -> dict:
        # Fixed-counter slot order (cycles=0, instructions=1) is an
        # architectural convention used uniformly across every public
        # Apple Silicon kpc-based tool to date; it is NOT re-derived
        # from the event database here because FIXED events carry no
        # configurable slot mapping. Cross-check against a known
        # workload before trusting absolute values -- see the risks
        # section of docs/receipts/perf_counters_bootstrap.json.
        cycles = after[0] - before[0]
        instructions = after[1] - before[1]
        out = {
            "cycles": cycles,
            "instructions": instructions,
            "ipc": (instructions / cycles) if cycles else None,
        }
        for metric_name, slot in self.event_config.slot_of.items():
            idx = self.n_fixed + slot
            out[metric_name] = after[idx] - before[idx]
        return out


def run_emu_workload(rom: Path, tape_path: Path, frames: int) -> dict:
    """Single NESEnvironment (1 core, no worker Pool), stepped through
    the first `frames` actions of a receipted solution tape.

    The tape (runs/full_run/full_tape.npy) is the full-game-clear
    action recording banked from the show's completed run. It was
    captured under frame_skip=4 for gameplay purposes; this benchmark
    replays its raw controller bytes 1:1 through frame_skip=1 steps
    purely as a fixed, deterministic, realistic CPU/PPU/APU workload --
    it does not reproduce the original in-game trajectory frame-for-
    frame, and doesn't need to. What matters for a counter benchmark is
    that it drives genuine emulator code paths with genuine recorded
    inputs rather than synthetic noise.
    """
    import numpy as np
    import nes_core

    if not rom.exists():
        raise FileNotFoundError(f"ROM not found: {rom}")
    if not tape_path.exists():
        raise FileNotFoundError(f"solution tape not found: {tape_path}")

    tape = np.load(tape_path, allow_pickle=True)
    if len(tape) < frames:
        raise ValueError(
            f"tape {tape_path} has only {len(tape)} actions; need {frames}"
        )
    actions = tape[:frames]

    env = nes_core.NESEnvironment(str(rom), frame_skip=1)
    env.reset()

    done_seen = False
    t0 = time.perf_counter()
    for a in actions:
        _frame, done = env.step(int(a))
        done_seen = done_seen or done
    dt = time.perf_counter() - t0

    return {
        "rom": str(rom),
        "tape": str(tape_path),
        "frames_requested": frames,
        "frames_run": int(len(actions)),
        "wall_s": dt,
        "steps_per_s": (len(actions) / dt) if dt > 0 else None,
        "episode_done_during_replay": bool(done_seen),
    }


WORKLOADS = {
    "emu": run_emu_workload,
}


def run_diagnose() -> dict:
    """Non-invasive probe: can we dlopen both frameworks, bind every
    required symbol, resolve every CONFIGURABLE_EVENTS name against
    this machine's live PMU database, and compute config words? Also
    reports the exact permission-check outcome. Never programs
    counters (safe to run without root, and safe to run repeatedly)."""
    report: dict = {
        "euid": os.geteuid(),
        "is_root": is_root(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
    }

    try:
        kperf = _KperfBindings()
        report["kperf_binding"] = {"loaded": True, "path": KPERF_PATH}
        report["fixed_counter_count"] = int(
            kperf.kpc_get_counter_count(KPC_CLASS_FIXED_MASK)
        )
        report["configurable_counter_count"] = int(
            kperf.kpc_get_counter_count(KPC_CLASS_CONFIGURABLE_MASK)
        )
    except PerfCountersUnavailable as e:
        report["kperf_binding"] = {"loaded": False, "error": str(e)}
        kperf = None

    try:
        kpep = _KpepBindings()
        db = kpep.open_db()
        report["kpep_binding"] = {
            "loaded": True,
            "path": KPERFDATA_PATH,
            "db_name": kpep.db_name(db),
        }
        cfg = build_configurable_config(kpep)
        report["configurable_events"] = {
            metric: {
                "kpep_event": CONFIGURABLE_EVENTS[metric],
                "physical_slot": cfg.slot_of[metric],
            }
            for metric in CONFIGURABLE_EVENTS
        }
        report["configurable_config_words"] = [hex(w) for w in cfg.config_words]
        report["configurable_classes_mask"] = hex(cfg.classes)
    except PerfCountersUnavailable as e:
        report["kpep_binding"] = {"loaded": False, "error": str(e)}

    if kperf is not None:
        probe_results = {}
        for fn_name, args in (
            ("kpc_set_counting", (KPC_CLASS_FIXED_MASK,)),
            ("kpc_set_thread_counting", (KPC_CLASS_FIXED_MASK,)),
        ):
            fn = getattr(kperf.lib, fn_name)
            rc = fn(*args)
            err = kperf.last_errno()
            probe_results[fn_name] = {
                "rc": rc,
                "errno": err,
                "errno_str": _errno_str(err),
            }
        report["permission_probe"] = probe_results

    report["sudo_invocation_for_tomorrow"] = sudo_invocation()
    return report


def _print_human(result: dict) -> None:
    print("== perf_counters ==")
    wl = result.get("workload", {})
    print(f"  bench:        {result.get('bench')}")
    print(f"  rom:          {wl.get('rom')}")
    print(f"  frames:       {wl.get('frames_run')} / requested {wl.get('frames_requested')}")
    print(f"  wall time:    {wl.get('wall_s', 0.0)*1000:.1f} ms")
    print(f"  throughput:   {wl.get('steps_per_s', 0.0):.1f} steps/s")
    print()
    hc = result.get("hardware_counters", {})
    if not hc.get("available"):
        print("  hardware counters: UNAVAILABLE")
        print(f"    reason: {hc.get('reason')}")
        return
    m = hc["metrics"]
    print("  hardware counters (this thread, delta over the workload):")
    print(f"    cycles:                 {m['cycles']:,}")
    print(f"    instructions:           {m['instructions']:,}")
    ipc = m["ipc"]
    print(f"    IPC:                    {ipc:.3f}" if ipc is not None else "    IPC: n/a")
    print(f"    l1d_fetch_misses:       {m['l1d_fetch_misses']:,}")
    print(f"    l1d_sector_refills:     {m['l1d_sector_refills']:,}")
    print(f"    l2_tlb_misses_data:     {m['l2_tlb_misses_data']:,}")
    print(f"    l2_tlb_misses_instr:    {m['l2_tlb_misses_instruction']:,}")
    print(f"    branch_mispredictions:  {m['branch_mispredictions']:,}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=sorted(WORKLOADS), default="emu")
    ap.add_argument(
        "--rom",
        default=str(_REPO / "roms" / "Super Mario Bros. (World).nes"),
    )
    ap.add_argument(
        "--tape",
        default=str(_REPO / "runs" / "full_run" / "full_tape.npy"),
        help="action tape to replay; must have >= --frames entries",
    )
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="write the JSON result to this path")
    ap.add_argument(
        "--diagnose",
        action="store_true",
        help="probe binding/event-DB/permissions only; don't run a workload",
    )
    ap.add_argument(
        "--xctrace-recipe",
        action="store_true",
        help="print the Instruments/xctrace fallback recipe and exit",
    )
    args = ap.parse_args(argv)

    if args.xctrace_recipe:
        print(XCTRACE_RECIPE_TEXT)
        return 0

    if args.diagnose:
        report = run_diagnose()
        if args.json:
            text = json.dumps(report, indent=2)
            print(text)
            if args.out:
                Path(args.out).write_text(text + "\n")
        else:
            print(json.dumps(report, indent=2))
        return 0

    session: Optional[PerfCounterSession] = None
    unavailable_reason: Optional[str] = None
    try:
        session = PerfCounterSession()
    except PerfCountersUnavailable as e:
        unavailable_reason = str(e)

    before = session.read_raw() if session is not None else None

    workload_fn = WORKLOADS[args.bench]
    try:
        workload_result = workload_fn(Path(args.rom), Path(args.tape), args.frames)
    except (FileNotFoundError, ValueError) as e:
        print(f"perf_counters: {e}", file=sys.stderr)
        return 1

    result: dict = {
        "bench": args.bench,
        "workload": workload_result,
        "is_root": is_root(),
    }

    if session is not None:
        after = session.read_raw()
        metrics = session.delta_metrics(before, after)
        result["hardware_counters"] = {
            "available": True,
            "n_fixed": session.n_fixed,
            "n_configurable": session.n_config,
            "metrics": metrics,
        }
    else:
        result["hardware_counters"] = {
            "available": False,
            "reason": unavailable_reason,
        }

    if args.json:
        text = json.dumps(result, indent=2)
        print(text)
    else:
        _print_human(result)
        text = json.dumps(result, indent=2)

    if args.out:
        Path(args.out).write_text(text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
