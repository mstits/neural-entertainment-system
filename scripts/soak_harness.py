"""Unattended rotation soak harness.

Rotates over a roster of game segments read from a config file, running
each segment under a wall-clock budget, and records everything needed to
audit the run afterwards without having watched it:

  * a per-segment receipt trail (hash-chained JSON receipts; verify with
    `--verify <soak_dir>`),
  * an intervention log — every event that would have required a human
    touch is recorded as a FAILURE with a diagnosis bundle path,
  * an UNSCORABLE queue — ambiguous events are queued for later manual
    adjudication, never silently counted as clears or non-clears,
  * a line-oriented JSONL event log suitable for detached (nohup) runs.

Rotation advances on segment clear, stall, or budget exhaustion; a
segment crash is contained (logged as an intervention with a bundle,
outcome CRASH) and the rotation continues — a dying segment never kills
the soak.

Audio: soaks default to the null audio sink. A roster requesting a real
audio sink is refused unless both --attended and --allow-real-audio are
passed (real audio is validated in short attended runs, never unattended
soaks).

WHAT IS NOT WIRED YET (honest list): actually launching an emulator
rollout for a roster entry, real clear detection / replay verification
against banked receipts, and the stall watchdog all live behind
`SegmentBackend`. The default backend is `UnwiredBackend`, which fails
loudly at preflight with instructions — this harness cannot silently
pretend to soak. Unit tests inject stub backends; `--selfcheck` uses a
built-in stub whose receipts are marked non-scoreable.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

OUTCOME_CLEAR = "CLEAR"
OUTCOME_STALL = "STALL"
OUTCOME_BUDGET = "BUDGET"
OUTCOME_UNSCORABLE = "UNSCORABLE"
OUTCOME_CRASH = "CRASH"
OUTCOMES = (
    OUTCOME_CLEAR,
    OUTCOME_STALL,
    OUTCOME_BUDGET,
    OUTCOME_UNSCORABLE,
    OUTCOME_CRASH,
)

NULL_AUDIO = "null"


class IntegrationPointNotWired(RuntimeError):
    """Raised where real emulator/detector machinery must be plugged in.

    Anything that catches this and continues is lying about what ran.
    """


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RosterEntry:
    name: str
    config: str  # profile path, relative to repo root or absolute
    budget_s: float
    # Optional explicit root state for backends that launch a solver;
    # when unset, backends fall back to the profile's start_state_path.
    root_state: Optional[str] = None

    def resolved_config(self) -> Path:
        p = Path(self.config)
        return p if p.is_absolute() else REPO / p

    def resolved_root_state(self) -> Optional[Path]:
        if self.root_state is None:
            return None
        p = Path(self.root_state)
        return p if p.is_absolute() else REPO / p


@dataclass(frozen=True)
class Roster:
    path: Path
    sha256: str
    audio_sink: str
    entries: tuple


def load_roster(path: Path, default_budget_s: float = 480.0) -> Roster:
    """Parse and validate a roster config.

    Schema (YAML):
        audio_sink: "null"          # anything else needs attended flags
        games:
          - name: <unique id>
            config: <profile path>
            budget_s: <per-segment wall-clock budget, optional>
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"roster config not found: {path}")
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not isinstance(data.get("games"), list):
        raise ValueError(f"roster {path}: expected mapping with a 'games' list")
    audio_sink = str(data.get("audio_sink", NULL_AUDIO))
    entries = []
    seen = set()
    for i, g in enumerate(data["games"]):
        if not isinstance(g, dict) or "name" not in g or "config" not in g:
            raise ValueError(
                f"roster {path}: games[{i}] must have 'name' and 'config'")
        name = str(g["name"])
        if name in seen:
            raise ValueError(f"roster {path}: duplicate game name {name!r}")
        seen.add(name)
        entry = RosterEntry(
            name=name,
            config=str(g["config"]),
            budget_s=float(g.get("budget_s", default_budget_s)),
            root_state=(str(g["root_state"])
                        if g.get("root_state") is not None else None),
        )
        if not entry.resolved_config().exists():
            raise FileNotFoundError(
                f"roster {path}: games[{i}] ({name}) config does not exist: "
                f"{entry.resolved_config()}")
        entries.append(entry)
    if not entries:
        raise ValueError(f"roster {path}: empty games list")
    return Roster(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        audio_sink=audio_sink,
        entries=tuple(entries),
    )


# ---------------------------------------------------------------------------
# Segment backends (THE integration point)
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    outcome: str
    detail: dict = field(default_factory=dict)
    # Ambiguous events observed during the segment; each is queued
    # UNSCORABLE regardless of the segment's own outcome.
    unscorable_events: list = field(default_factory=list)


class SegmentBackend:
    """Interface a real (or stub) segment runner must implement."""

    #: Receipts record this; adjudication must only score receipts from
    #: backends explicitly marked scoreable.
    scoreable = False

    def preflight(self) -> None:
        """Raise if this backend cannot honestly run segments."""
        raise NotImplementedError

    def validate_roster(self, roster: "Roster") -> list:
        """Return a list of problems that make roster entries unrunnable.

        Called by the harness after preflight() and before any output is
        created; a non-empty return refuses the whole soak up front
        rather than burning segments on per-entry crashes."""
        return []

    def run_segment(self, entry: RosterEntry, seg_dir: Path) -> SegmentResult:
        """Run one segment within entry.budget_s; never block past budget."""
        raise NotImplementedError


class UnwiredBackend(SegmentBackend):
    """Default backend: refuses loudly. This is deliberate.

    To make the harness run real segments, implement a SegmentBackend
    that (1) launches the solver/eval process for entry.resolved_config()
    with a null audio sink, (2) detects clears ONLY via replay
    verification against banked solution receipts (plus the
    counterfactual gate for first-time clears), (3) raises no outcome it
    cannot back with a receipt — ambiguous means UNSCORABLE, and
    (4) enforces the stall watchdog to end the segment at budget.
    Then pass it via --backend module:Class.
    """

    _MSG = (
        "soak backend not wired: no emulator launch / clear verification / "
        "stall watchdog is connected. Implement a SegmentBackend (see "
        "UnwiredBackend docstring in scripts/soak_harness.py) and pass "
        "--backend module:Class. Refusing to pretend to soak."
    )

    def preflight(self) -> None:
        raise IntegrationPointNotWired(self._MSG)

    def run_segment(self, entry: RosterEntry, seg_dir: Path) -> SegmentResult:
        raise IntegrationPointNotWired(self._MSG)


class SelfCheckStubBackend(SegmentBackend):
    """Built-in stub for --selfcheck ONLY. Produces fake outcomes so the
    receipt/rotation/logging plumbing can be exercised without an
    emulator. Never scoreable; receipts carry backend=SelfCheckStubBackend
    and the soak manifest carries selfcheck=true.
    """

    scoreable = False

    def preflight(self) -> None:
        return None

    def run_segment(self, entry: RosterEntry, seg_dir: Path) -> SegmentResult:
        return SegmentResult(
            outcome=OUTCOME_BUDGET,
            detail={"note": "selfcheck stub — no emulator was run"},
        )


# A backend that claims scoreable=True can mint CLEAR receipts an
# adjudicator will count. Restrict that power to reviewed backends: a
# rogue --backend evil:Backend with scoreable=True and no emulator is
# refused here (adversary M2). Non-scoreable backends are unrestricted —
# they cannot produce a scoreable receipt anyway.
SCOREABLE_BACKEND_ALLOWLIST = frozenset({
    "soak_backend:SolverBackend",
})


def resolve_backend(spec: str) -> SegmentBackend:
    """Import 'module:Class' and instantiate it as a SegmentBackend."""
    mod_name, _, cls_name = spec.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"--backend must be module:Class, got {spec!r}")
    import importlib

    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, SegmentBackend)):
        raise TypeError(f"{spec} is not a SegmentBackend subclass")
    inst = cls()
    if getattr(inst, "scoreable", False) and spec not in SCOREABLE_BACKEND_ALLOWLIST:
        raise TypeError(
            f"backend {spec} claims scoreable=True but is not on the "
            "scoreable-backend allowlist (soak_harness.SCOREABLE_BACKEND_"
            "ALLOWLIST); refusing to let an unreviewed backend mint "
            "scoreable receipts")
    return inst


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def _git_rev(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1).encode()


def verify_receipt_trail(soak_dir: Path) -> list:
    """Recompute the receipt hash chain; return a list of problems.

    Empty list == trail verifies. Each receipt embeds the sha256 of the
    previous receipt file (genesis links to the soak manifest), so
    deleting, reordering, or editing any receipt breaks the chain.
    """
    soak_dir = Path(soak_dir)
    problems = []
    recomputed_outcomes = {k: 0 for k in OUTCOMES}
    manifest_p = soak_dir / "soak_manifest.json"
    if not manifest_p.exists():
        return [f"missing soak_manifest.json in {soak_dir}"]
    prev_hash = _sha256_file(manifest_p)
    seg_root = soak_dir / "segments"
    seg_dirs = sorted(seg_root.iterdir()) if seg_root.exists() else []
    for seg_dir in seg_dirs:
        rp = seg_dir / "receipt.json"
        if not rp.exists():
            problems.append(f"segment without receipt: {seg_dir.name}")
            continue
        try:
            receipt = json.loads(rp.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"unparseable receipt {seg_dir.name}: {e}")
            continue
        if receipt.get("prev_sha256") != prev_hash:
            problems.append(
                f"receipt chain broken at {seg_dir.name}: "
                f"prev_sha256={receipt.get('prev_sha256')!r} != {prev_hash!r}")
        body = dict(receipt)
        claimed = body.pop("receipt_sha256", None)
        actual = _sha256_bytes(_canonical_json(body))
        if claimed != actual:
            problems.append(
                f"receipt body hash mismatch at {seg_dir.name}: "
                f"claimed {claimed!r} actual {actual!r}")
        oc = receipt.get("outcome")
        if oc not in OUTCOMES:
            problems.append(
                f"receipt {seg_dir.name}: unknown outcome {oc!r}")
        else:
            recomputed_outcomes[oc] += 1
        prev_hash = _sha256_file(rp)

    # Final receipt: the oracle's pass bit lives in the hash chain, not
    # in the loose summary.json (adversary M2). Present on any run that
    # reached its end (completed or terminated); absent only on a run
    # killed before flush. When present it is chained AND its outcome
    # counts must equal what the segment receipts actually say, and its
    # intervention count must match the intervention log and bundle dirs
    # — so neither the receipt nor the loose logs can lie unilaterally.
    final_p = soak_dir / "final_receipt.json"
    if final_p.exists():
        try:
            final = json.loads(final_p.read_text())
        except json.JSONDecodeError as e:
            return problems + [f"unparseable final_receipt.json: {e}"]
        if final.get("prev_sha256") != prev_hash:
            problems.append(
                f"final receipt chain broken: prev_sha256="
                f"{final.get('prev_sha256')!r} != {prev_hash!r}")
        body = dict(final)
        claimed = body.pop("receipt_sha256", None)
        if claimed != _sha256_bytes(_canonical_json(body)):
            problems.append("final receipt body hash mismatch")
        if final.get("outcomes") != recomputed_outcomes:
            problems.append(
                f"final receipt outcome counts {final.get('outcomes')!r} "
                f"disagree with the chained segment receipts "
                f"{recomputed_outcomes!r} (tampering)")
        claimed_interv = final.get("interventions")
        interv_log = _count_lines(soak_dir / "interventions.jsonl")
        bundles = _count_bundles(soak_dir / "diagnosis")
        if claimed_interv != interv_log:
            problems.append(
                f"final receipt interventions={claimed_interv} disagrees "
                f"with interventions.jsonl line count {interv_log} "
                "(a scrubbed intervention log)")
        if claimed_interv != bundles:
            problems.append(
                f"final receipt interventions={claimed_interv} disagrees "
                f"with diagnosis bundle count {bundles} "
                "(a scrubbed diagnosis directory)")
    return problems


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text().splitlines() if ln.strip())


def _count_bundles(diag_dir: Path) -> int:
    if not diag_dir.exists():
        return 0
    return sum(1 for d in diag_dir.iterdir()
               if d.is_dir() and d.name.startswith("bundle_"))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class SoakHarness:
    def __init__(
        self,
        roster: Roster,
        out_dir: Path,
        backend: SegmentBackend,
        duration_s: float,
        max_segments: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        selfcheck: bool = False,
    ):
        self.roster = roster
        self.out_dir = Path(out_dir)
        self.backend = backend
        self.duration_s = float(duration_s)
        self.max_segments = max_segments
        self.clock = clock
        self.selfcheck = selfcheck
        self._t0 = None
        self._prev_receipt_hash = None
        self._seg_count = 0
        self._intervention_count = 0
        self._unscorable_count = 0
        self._outcome_counts = {k: 0 for k in OUTCOMES}
        self._terminated = False

    # -- paths ------------------------------------------------------------
    @property
    def events_path(self) -> Path:
        return self.out_dir / "soak_log.jsonl"

    @property
    def interventions_path(self) -> Path:
        return self.out_dir / "interventions.jsonl"

    @property
    def unscorable_path(self) -> Path:
        return self.out_dir / "unscorable.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "soak_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"

    # -- plumbing ----------------------------------------------------------
    def _emit(self, event: str, **fields) -> dict:
        rec = {"t": time.time(), "event": event, **fields}
        with self.events_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"[soak] {event} {json.dumps(fields, sort_keys=True)}",
              flush=True)
        return rec

    def _loadavg(self):
        try:
            return list(os.getloadavg())
        except OSError:
            return None

    def _tail_events(self, n: int = 40) -> list:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text().splitlines()
        return lines[-n:]

    # -- intervention log ---------------------------------------------------
    def log_intervention(self, reason: str, segment: Optional[dict] = None,
                         exc: Optional[BaseException] = None) -> Path:
        """A would-be human touch. Always a logged FAILURE with a bundle."""
        self._intervention_count += 1
        bundle = self.out_dir / "diagnosis" / f"bundle_{self._intervention_count:04d}"
        bundle.mkdir(parents=True, exist_ok=True)
        context = {
            "reason": reason,
            "segment": segment,
            "traceback": (
                "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)) if exc else None),
            "loadavg": self._loadavg(),
            "recent_events": self._tail_events(),
            "t": time.time(),
        }
        (bundle / "context.json").write_text(json.dumps(context, indent=1))
        rec = {
            "t": time.time(),
            "reason": reason,
            "segment": segment,
            "bundle": str(bundle),
            "counted_as": "failure",
        }
        with self.interventions_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._emit("intervention", reason=reason, bundle=str(bundle))
        return bundle

    def _queue_unscorable(self, segment: dict, event) -> None:
        self._unscorable_count += 1
        rec = {
            "t": time.time(),
            "segment": segment,
            "event": event,
            "disposition": "queued_for_adjudication_not_counted",
        }
        with self.unscorable_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._emit("unscorable_queued", segment=segment["name"],
                   index=segment["index"])

    # -- receipts ------------------------------------------------------------
    def _write_receipt(self, seg_dir: Path, info: dict) -> None:
        if self._prev_receipt_hash is None:
            self._prev_receipt_hash = _sha256_file(self.manifest_path)
        body = dict(info)
        body["prev_sha256"] = self._prev_receipt_hash
        body["receipt_sha256"] = _sha256_bytes(_canonical_json(body))
        rp = seg_dir / "receipt.json"
        rp.write_text(json.dumps(body, sort_keys=True, indent=1))
        self._prev_receipt_hash = _sha256_file(rp)

    def _write_summary(self, status: str) -> None:
        summary = {
            "status": status,
            "segments": self._seg_count,
            "outcomes": dict(self._outcome_counts),
            "interventions": self._intervention_count,
            "unscorable": self._unscorable_count,
            "scoreable_backend": bool(self.backend.scoreable),
            "selfcheck": self.selfcheck,
            "passed_zero_interventions": (
                status == "completed" and self._intervention_count == 0
                # a run that never ran a segment proves nothing and can
                # never claim the pass bit (e.g. --max-segments 0)
                and self._seg_count > 0),
            "t": time.time(),
        }
        self.summary_path.write_text(json.dumps(summary, indent=1))

    def _handle_signal(self, signum, frame) -> None:
        self._terminated = True
        self._emit("terminate_requested", signum=signum)

    # -- main loop -------------------------------------------------------------
    def run(self) -> dict:
        self.backend.preflight()
        roster_problems = self.backend.validate_roster(self.roster)
        if roster_problems:
            raise SystemExit(
                "backend refuses roster:\n  "
                + "\n  ".join(str(p) for p in roster_problems))
        if self.roster.audio_sink != NULL_AUDIO:
            raise SystemExit(
                f"roster audio_sink={self.roster.audio_sink!r}: harness-level "
                "guard requires the null sink for unattended runs "
                "(caller must gate real audio behind --attended "
                "--allow-real-audio)")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": "soak_manifest",
            "argv": sys.argv,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "git_rev": _git_rev(REPO),
            "harness_sha256": _sha256_file(Path(__file__)),
            "roster_path": str(self.roster.path),
            "roster_sha256": self.roster.sha256,
            "roster": [
                {"name": e.name, "config": e.config, "budget_s": e.budget_s,
                 "root_state": e.root_state}
                for e in self.roster.entries
            ],
            "audio_sink": self.roster.audio_sink,
            "duration_s": self.duration_s,
            "backend": type(self.backend).__name__,
            "backend_scoreable": bool(self.backend.scoreable),
            "selfcheck": self.selfcheck,
            "started_at": time.time(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=1))
        (self.out_dir / "pid").write_text(str(os.getpid()))
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle_signal)
            except ValueError:
                pass  # not on the main thread (tests)
        self._t0 = self.clock()
        self._emit("soak_start", roster=[e.name for e in self.roster.entries],
                   duration_s=self.duration_s,
                   backend=type(self.backend).__name__)
        idx = 0
        while not self._terminated:
            if self.clock() - self._t0 >= self.duration_s:
                break
            if self.max_segments is not None and self._seg_count >= self.max_segments:
                break
            entry = self.roster.entries[idx % len(self.roster.entries)]
            self._run_one_segment(entry)
            idx += 1
        status = "terminated" if self._terminated else "completed"
        self._emit("soak_end", status=status, segments=self._seg_count,
                   interventions=self._intervention_count,
                   unscorable=self._unscorable_count)
        self._write_final_receipt(status)
        self._write_summary(status)
        return json.loads(self.summary_path.read_text())

    def _write_final_receipt(self, status: str) -> None:
        """The pass bit, inside the hash chain. summary.json is a loose
        convenience copy; final_receipt.json is the authority /adjudicate
        must read (it is chained and its counts are cross-checked against
        the segment receipts + intervention log by verify_receipt_trail).
        """
        if self._prev_receipt_hash is None:
            # no segments ran: genesis is the manifest
            self._prev_receipt_hash = _sha256_file(self.manifest_path)
        body = {
            "kind": "soak_final_receipt",
            "status": status,
            "segments": self._seg_count,
            "outcomes": dict(self._outcome_counts),
            "interventions": self._intervention_count,
            "unscorable": self._unscorable_count,
            "backend": type(self.backend).__name__,
            "backend_scoreable": bool(self.backend.scoreable),
            "selfcheck": self.selfcheck,
            "passed_zero_interventions": (
                status == "completed" and self._intervention_count == 0
                and self._seg_count > 0),
            "prev_sha256": self._prev_receipt_hash,
        }
        body["receipt_sha256"] = _sha256_bytes(_canonical_json(body))
        (self.out_dir / "final_receipt.json").write_text(
            json.dumps(body, sort_keys=True, indent=1))

    def _run_one_segment(self, entry: RosterEntry) -> None:
        self._seg_count += 1
        seg_dir = (self.out_dir / "segments"
                   / f"seg_{self._seg_count:04d}_{entry.name}")
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_ref = {"index": self._seg_count, "name": entry.name}
        started = time.time()
        self._emit("segment_start", **seg_ref, config=entry.config,
                   budget_s=entry.budget_s)
        try:
            result = self.backend.run_segment(entry, seg_dir)
            if result.outcome not in OUTCOMES:
                raise ValueError(
                    f"backend returned unknown outcome {result.outcome!r}")
        except BaseException as e:  # noqa: BLE001 — crash-only: contain, log, advance
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            bundle = self.log_intervention(
                f"segment crashed: {type(e).__name__}: {e}",
                segment=seg_ref, exc=e)
            result = SegmentResult(
                outcome=OUTCOME_CRASH,
                detail={"error": f"{type(e).__name__}: {e}",
                        "bundle": str(bundle)},
            )
        self._outcome_counts[result.outcome] += 1
        if result.outcome == OUTCOME_UNSCORABLE:
            self._queue_unscorable(seg_ref, result.detail or
                                   {"note": "segment-level ambiguity"})
        for ev in result.unscorable_events:
            self._queue_unscorable(seg_ref, ev)
        cfg = entry.resolved_config()
        receipt = {
            "kind": "segment_receipt",
            "segment_index": self._seg_count,
            "name": entry.name,
            "config": entry.config,
            "config_sha256": _sha256_file(cfg) if cfg.exists() else None,
            "budget_s": entry.budget_s,
            "outcome": result.outcome,
            "detail": result.detail,
            "backend": type(self.backend).__name__,
            "backend_scoreable": bool(self.backend.scoreable),
            "started_at": started,
            "ended_at": time.time(),
            "loadavg": self._loadavg(),
        }
        self._write_receipt(seg_dir, receipt)
        self._emit("segment_end", **seg_ref, outcome=result.outcome)
        self._write_summary("running")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_out_dir(root: Path, selfcheck: bool) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = "selfcheck_" if selfcheck else "soak_"
    return root / f"{prefix}{stamp}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", type=Path, default=None,
                    help="roster config (yaml). Required unless --verify "
                         "or --selfcheck.")
    ap.add_argument("--hours", type=float, default=1.0,
                    help="total soak duration in hours")
    ap.add_argument("--out-root", type=Path, default=REPO / "runs" / "soak",
                    help="parent directory for this soak's output dir")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="optional hard cap on segments (smoke runs)")
    ap.add_argument("--segment-budget", type=float, default=None,
                    help="override every roster entry's budget_s")
    ap.add_argument("--backend", type=str, default=None,
                    help="module:Class implementing SegmentBackend; "
                         "default refuses (UnwiredBackend)")
    ap.add_argument("--attended", action="store_true",
                    help="a human is watching this run")
    ap.add_argument("--allow-real-audio", action="store_true",
                    help="permit a non-null audio sink (needs --attended)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="exercise plumbing with the built-in stub backend "
                         "(receipts marked non-scoreable)")
    ap.add_argument("--verify", type=Path, default=None, metavar="SOAK_DIR",
                    help="verify a soak dir's receipt chain and exit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.verify is not None:
        problems = verify_receipt_trail(args.verify)
        for p in problems:
            print(f"FAIL: {p}")
        print("receipt trail: " + ("OK" if not problems else
                                   f"{len(problems)} problem(s)"))
        return 0 if not problems else 1

    if args.selfcheck:
        roster = Roster(
            path=Path("<selfcheck>"), sha256="0" * 64, audio_sink=NULL_AUDIO,
            entries=(
                RosterEntry("stub_a", str(Path(__file__).relative_to(REPO)), 1.0),
                RosterEntry("stub_b", str(Path(__file__).relative_to(REPO)), 1.0),
            ),
        )
        out_dir = _default_out_dir(args.out_root, selfcheck=True)
        harness = SoakHarness(roster, out_dir, SelfCheckStubBackend(),
                              duration_s=60.0, max_segments=4, selfcheck=True)
        summary = harness.run()
        problems = verify_receipt_trail(out_dir)
        ok = (not problems and summary["segments"] == 4
              and summary["interventions"] == 0)
        print(f"selfcheck: {'OK' if ok else 'FAIL'} out={out_dir} "
              f"receipt_problems={problems}")
        return 0 if ok else 1

    if args.roster is None:
        ap.error("--roster is required (or use --selfcheck / --verify)")
    roster = load_roster(args.roster)
    if args.segment_budget is not None:
        roster = Roster(
            path=roster.path, sha256=roster.sha256,
            audio_sink=roster.audio_sink,
            entries=tuple(
                RosterEntry(e.name, e.config, float(args.segment_budget),
                            root_state=e.root_state)
                for e in roster.entries),
        )
    if roster.audio_sink != NULL_AUDIO:
        if not (args.attended and args.allow_real_audio):
            print(f"REFUSED: roster requests audio_sink="
                  f"{roster.audio_sink!r}; unattended soaks use the null "
                  "sink. Re-run with --attended --allow-real-audio only "
                  "for a short, watched dry run.")
            return 2
    backend: SegmentBackend
    backend = (resolve_backend(args.backend) if args.backend
               else UnwiredBackend())
    out_dir = _default_out_dir(args.out_root, selfcheck=False)
    harness = SoakHarness(
        roster, out_dir, backend,
        duration_s=args.hours * 3600.0,
        max_segments=args.max_segments,
    )
    summary = harness.run()
    print(json.dumps(summary, indent=1))
    return 0 if summary["interventions"] == 0 else 1


if __name__ == "__main__":
    # Delegate to the canonical module so classes resolved by
    # --backend (which imports 'soak_harness') are identical to the
    # ones this process runs with — running the file as __main__ would
    # otherwise create a second SegmentBackend class and fail the
    # resolve_backend issubclass check for every real backend.
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    import soak_harness as _canonical

    sys.exit(_canonical.main())
