"""Solver-backed SegmentBackend for scripts/soak_harness.py.

Wires the harness's segment seam to the real search stack: each segment
launches scripts/go_explore_solve.py as a subprocess for the roster
entry's profile and wall-clock budget, watches its progress.jsonl and
solutions/ output, and maps what actually happened onto the harness's
outcome vocabulary:

- CLEAR      a solutions/sol_*.json landed, its sidecar records
             replay_verified (the solver's verify-bank re-simulates
             every candidate from its root in a fresh pool before
             banking; DEFAULT ON), start_wd != clear_wd, and the
             profile's clear signal is a deterministic byte predicate
             (SMB engine adapter, level_key, byte_change, finale).
             The receipt carries the sidecar and action-tape hashes.
- UNSCORABLE a solution landed but the clear signal is windowed /
             heuristic (confluence) or the artifact fails validation —
             queued for adjudication, never counted either way.
             Confluence-mode segments are launched with the
             counterfactual gate armed, so the artifact adjudicators
             see includes the K-branch probe verdict.
- STALL      the solver's own stall watchdog (stall_flat_windows in
             progress.jsonl, one record per 60 s) reported at least
             stall_flat_windows_limit consecutive flat windows.
- BUDGET     wall-clock budget elapsed with the solver still healthy
             (including the solver reaching its own --minutes deadline
             and exiting cleanly).
- CRASH      the solver exited early without output, or stopped
             emitting progress records while still alive (hang) — the
             child is killed, the segment diagnosed, rotation continues.

The backend never blocks past entry.budget_s plus a bounded teardown
grace: the child gets SIGTERM (the solver traps it, flushes, and exits
cleanly) and SIGKILL after term_grace_s. The solver's worker pool is
audio-silent by construction, satisfying the roster's null-sink
requirement without any audio plumbing here.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from soak_harness import (  # noqa: E402
    OUTCOME_BUDGET,
    OUTCOME_CLEAR,
    OUTCOME_CRASH,
    OUTCOME_STALL,
    OUTCOME_UNSCORABLE,
    RosterEntry,
    Roster,
    SegmentBackend,
    SegmentResult,
)

REPO = Path(__file__).resolve().parent.parent

# Matches the SMB-engine default ROM in scripts/go_explore_solve.py
# (profiles without a `solve:` block use the byte-exact SMB adapter,
# which falls back to this when the profile has no `rom:` override).
DEFAULT_SMB_ROM = "roms/Super Mario Bros. (World).nes"


def clear_mode_for_profile(profile: dict) -> str:
    """The clear-signal mode a profile's solve run will use.

    Mirrors make_game() + GenericGame's clear wiring in
    go_explore_solve.py: no `solve:` block means the byte-exact SMB
    engine adapter; a `solve.clear.mode` wins; a non-empty
    `solve.level_key` without a clear block is the level_key predicate;
    otherwise no clear signal is wired at all ("none") and a solution
    artifact appearing anyway is inherently unscorable.
    """
    if "solve" not in profile:
        return "smb_engine"
    s = profile.get("solve") or {}
    cl = s.get("clear") or {}
    mode = cl.get("mode")
    if mode:
        return str(mode)
    if s.get("level_key"):
        return "level_key"
    return "none"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class SolverBackend(SegmentBackend):
    """Launches go_explore_solve.py per segment; see module docstring."""

    scoreable = True

    solver = REPO / "scripts" / "go_explore_solve.py"
    python = sys.executable
    workers = int(os.environ.get("SOAK_WORKERS", "8"))
    max_steps = 4000
    flush_secs = 60.0
    poll_secs = 5.0
    # Solver emits one progress.jsonl record per 60 s; no record for
    # this long while the process is alive means it is wedged.
    hang_timeout_s = 210.0
    # Solver-side watchdog convention: one flat window == 60 s with no
    # archive growth. Three in a row ends the segment as STALL.
    stall_flat_windows_limit = 3
    term_grace_s = 25.0
    # Shave the solver's own deadline under the segment budget so the
    # normal budget path is the solver flushing and exiting cleanly.
    solver_deadline_margin_s = 15.0
    check_imports = True

    #: Deterministic byte-predicate clear signals; anything else is
    #: adjudicated, never auto-scored.
    SCOREABLE_MODES = ("smb_engine", "level_key", "byte_change", "finale")

    # -- harness hooks ------------------------------------------------------

    def preflight(self) -> None:
        problems = []
        if not Path(self.solver).exists():
            problems.append(f"solver not found: {self.solver}")
        if not (REPO / "roms").exists():
            problems.append(
                f"{REPO / 'roms'} missing — ROM/state assets are resolved "
                "relative to the repo containing this script")
        if self.check_imports:
            r = subprocess.run(
                [str(self.python), "-c", "import nes_core, numpy, yaml"],
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                problems.append(
                    f"interpreter {self.python} cannot import solver deps: "
                    f"{r.stderr.strip()[-300:]}")
        if problems:
            raise RuntimeError(
                "SolverBackend preflight failed:\n  " + "\n  ".join(problems))

    def validate_roster(self, roster: Roster) -> list:
        problems = []
        for e in roster.entries:
            cfg = e.resolved_config()
            try:
                profile = yaml.safe_load(cfg.read_text())
            except Exception as ex:  # noqa: BLE001 — reported, not raised
                problems.append(f"{e.name}: profile {cfg} unreadable: {ex}")
                continue
            if not isinstance(profile, dict):
                problems.append(f"{e.name}: profile {cfg} is not a mapping")
                continue
            root = self._root_state(e, profile)
            if root is None:
                problems.append(
                    f"{e.name}: no root_state in roster entry and no "
                    f"start_state_path in {cfg.name}")
            elif not root.exists():
                problems.append(f"{e.name}: root state missing: {root}")
            rom_problem = self._rom_problem(e.name, profile)
            if rom_problem:
                problems.append(rom_problem)
        return problems

    def run_segment(self, entry: RosterEntry, seg_dir: Path) -> SegmentResult:
        profile = yaml.safe_load(entry.resolved_config().read_text())
        root = self._root_state(entry, profile)
        if root is None or not root.exists():
            raise RuntimeError(
                f"segment {entry.name}: root state unavailable: {root}")
        mode = clear_mode_for_profile(profile)
        solve_dir = seg_dir / "solve"
        log_path = seg_dir / "solver.log"
        argv = self._argv(entry, root, mode, solve_dir)

        t0 = time.monotonic()
        deadline = t0 + float(entry.budget_s)
        end_reason = None
        last_progress_t = t0
        last_rec: dict = {}
        prog_pos = 0
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                argv, stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(REPO), start_new_session=True)
        try:
            progress = solve_dir / "progress.jsonl"
            while True:
                rc = proc.poll()
                if sorted(solve_dir.glob("solutions/sol_*.json")):
                    end_reason = "solution"
                    break
                recs, prog_pos = self._read_new_progress(progress, prog_pos)
                now = time.monotonic()
                if recs:
                    last_progress_t = now
                    last_rec = recs[-1]
                    flat = int(last_rec.get("stall_flat_windows", 0) or 0)
                    if flat >= self.stall_flat_windows_limit:
                        end_reason = "stall"
                        break
                if rc is not None:
                    end_reason = "exited"
                    break
                if now >= deadline:
                    end_reason = "budget"
                    break
                if now - last_progress_t > self.hang_timeout_s:
                    end_reason = "hang"
                    break
                time.sleep(self.poll_secs)
        finally:
            self._stop(proc)

        elapsed = time.monotonic() - t0
        # A solution may land during teardown (the solver banks on its
        # way out); it always outranks the loop's exit reason.
        sols = sorted(solve_dir.glob("solutions/sol_*.json"))
        detail = {
            "game": entry.name,
            "clear_mode": mode,
            "argv": [str(a) for a in argv],
            "root_state": str(root),
            "elapsed_s": round(elapsed, 2),
            "solver_exit": proc.returncode,
            "end_reason": end_reason,
            "progress_last": {
                k: last_rec.get(k)
                for k in ("cells", "steps", "sps", "solutions",
                          "stall_flat_windows")
                if k in last_rec
            },
            "loadavg": self._loadavg(),
            "log_tail": self._tail(log_path),
        }
        if sols:
            return self._classify_solution(mode, sols, detail)
        if end_reason == "stall":
            detail["diagnosis"] = (
                "solver stall watchdog: "
                f">={self.stall_flat_windows_limit} consecutive flat 60s "
                "windows (no archive growth)")
            return SegmentResult(OUTCOME_STALL, detail)
        if end_reason == "budget":
            return SegmentResult(OUTCOME_BUDGET, detail)
        if end_reason == "hang":
            detail["diagnosis"] = (
                f"no progress record for >{self.hang_timeout_s:g}s with "
                "the solver process still alive; killed")
            return SegmentResult(OUTCOME_CRASH, detail)
        # exited: rc==0 at (or near) the solver's own --minutes deadline
        # is the normal budget path; anything else is an early death.
        if proc.returncode == 0 and elapsed >= 0.85 * float(entry.budget_s):
            return SegmentResult(OUTCOME_BUDGET, detail)
        detail["diagnosis"] = (
            f"solver exited rc={proc.returncode} after {elapsed:.0f}s "
            f"(budget {entry.budget_s:g}s) without banking a solution")
        return SegmentResult(OUTCOME_CRASH, detail)

    # -- solution classification -------------------------------------------

    def _classify_solution(self, mode: str, sols: list,
                           detail: dict) -> SegmentResult:
        side_p = Path(sols[0])
        tape_p = side_p.with_name(side_p.stem + ".actions.npy")
        problems = []
        sidecar = {}
        try:
            sidecar = json.loads(side_p.read_text())
        except Exception as ex:  # noqa: BLE001 — reported, not raised
            problems.append(f"sidecar unreadable: {ex}")
        if sidecar and sidecar.get("replay_verified") is not True:
            problems.append("sidecar.replay_verified is not true")
        start_wd = sidecar.get("start_wd")
        clear_wd = sidecar.get("clear_wd")
        if start_wd is not None and start_wd == clear_wd:
            problems.append(
                f"start_wd == clear_wd ({start_wd}): no level transition")
        if not tape_p.exists():
            problems.append(f"action tape missing: {tape_p.name}")
        detail = {
            **detail,
            "solution_sidecar": str(side_p),
            "sidecar_sha256": _sha256_file(side_p),
            "tape_sha256": _sha256_file(tape_p) if tape_p.exists() else None,
            "start_wd": start_wd,
            "clear_wd": clear_wd,
            "solution_steps": sidecar.get("steps"),
            "counterfactual_gate": sidecar.get("counterfactual_gate"),
            "extra_solutions": [str(p) for p in sols[1:]],
        }
        if problems:
            detail["problems"] = problems
            detail["reason"] = "solution artifact failed validation"
            return SegmentResult(OUTCOME_UNSCORABLE, detail)
        if mode in self.SCOREABLE_MODES:
            return SegmentResult(OUTCOME_CLEAR, detail)
        detail["reason"] = (
            "confluence clear signal is windowed/heuristic; queued for "
            "adjudication with its counterfactual-gate verdict"
            if mode == "confluence" else
            f"clear signal mode {mode!r} is not auto-scoreable; queued "
            "for adjudication")
        return SegmentResult(OUTCOME_UNSCORABLE, detail)

    # -- launch plumbing ----------------------------------------------------

    def _argv(self, entry: RosterEntry, root: Path, mode: str,
              solve_dir: Path) -> list:
        seed = int.from_bytes(
            hashlib.sha256(solve_dir.parent.name.encode()).digest()[:4],
            "big")
        minutes = max(
            0.25,
            (float(entry.budget_s) - self.solver_deadline_margin_s) / 60.0)
        argv = [str(self.python), str(self.solver),
                "--out", str(solve_dir),
                "--root-state", str(root),
                "--profile", str(entry.resolved_config()),
                "--minutes", f"{minutes:g}",
                "--workers", str(self.workers),
                "--want-solutions", "1",
                "--max-steps", str(self.max_steps),
                "--flush-secs", f"{self.flush_secs:g}",
                "--seed", str(seed)]
        if mode == "confluence":
            argv.append("--counterfactual-gate")
        return argv

    def _root_state(self, entry: RosterEntry, profile: dict):
        p = entry.resolved_root_state()
        if p is not None:
            return p
        ssp = profile.get("start_state_path")
        if not ssp:
            return None
        sp = Path(ssp)
        return sp if sp.is_absolute() else REPO / sp

    def _rom_problem(self, name: str, profile: dict):
        if "solve" in profile:
            rom = (profile.get("solve") or {}).get("rom")
            if not rom:
                return f"{name}: profile has a solve block but no solve.rom"
        else:
            rom = profile.get("rom") or DEFAULT_SMB_ROM
        rom_p = Path(rom)
        rom_p = rom_p if rom_p.is_absolute() else REPO / rom_p
        if not rom_p.exists():
            return f"{name}: rom missing: {rom_p}"
        return None

    def _read_new_progress(self, path: Path, pos: int):
        if not path.exists():
            return [], pos
        recs = []
        with path.open("rb") as f:
            f.seek(pos)
            while True:
                line = f.readline()
                if not line or not line.endswith(b"\n"):
                    break  # partial write: retry from same pos next poll
                pos = f.tell()
                try:
                    recs.append(json.loads(line))
                except Exception:  # noqa: BLE001 — skip torn/garbled record
                    continue
        return recs, pos

    def _stop(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=self.term_grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)

    def _tail(self, log_path: Path, n: int = 30, max_chars: int = 6000):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            return None
        tail = "\n".join(text.splitlines()[-n:])
        return tail[-max_chars:]

    def _loadavg(self):
        try:
            return list(os.getloadavg())
        except OSError:
            return None
