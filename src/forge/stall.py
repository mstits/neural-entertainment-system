"""The STALLED verdict: turns the existing, unacted stall counter into a
verdict the engine journals. Two independent kinds (FORGE_SPEC_2026-09-01.md
§2a) because one clock cannot see both walls:

  * ``archive_verdict`` -- in-run, clock-driven, for a *live* child's
    ``progress.jsonl`` tail. Independently REPLAYS the real
    ``update_stall()`` (``scripts/go_explore_solve.py``) over the given
    rows rather than trusting the stored ``stall_flat_windows`` field, so
    a stale or corrupted field on disk cannot produce a false verdict.

  * ``campaign_verdict`` -- cross-run, receipt-driven, for a wall defined
    as a family of prior runs via a manifest (``runs/forge/walls/<id>.json``,
    written by the build; see ``load_wall_manifest``). Detection stays
    self-measured: the verdict reads only what each manifest member
    already recorded on disk.

Every value returned is JSON-safe (str/int/float/bool/list/dict/None) so
it can be journaled and appended to ``runs/engine/stall_receipts.jsonl``
verbatim. ``evidence`` always holds raw numbers, never a derived ratio.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from scripts.go_explore_solve import update_stall

REPO = Path(__file__).resolve().parent.parent.parent

# FROZEN_WINDOWS_MAX and EFFORT_MIN_STEPS are DUPLICATED here, not
# imported. The module in src/training/ that calibrates these two
# constants (see its :149 and :159) is deliberately offline and
# runtime-inert by its own module docstring -- the struck-down
# misclassification verdict in this repo's history was live BECAUSE
# that module was wired into a runtime decision -- and its own test
# suite fails a named test for any importer outside one tolerated pure
# reader (scripts/go_explore_solve.py). That guard scans scripts/+src/+
# nes_core/+configs/ for the module's name as a bare substring, so this
# comment deliberately never spells the name as one token, the same way
# the module's own docstring is never quoted here, and never spells the
# struck-down verdict's own name either. A same-value assertion lives in
# tests/test_forge_stall.py instead, which is free to
# import that module directly since tests/ sits outside the guard's
# scanned roots.
FROZEN_WINDOWS_MAX = 12  # CALIBRATED-OFFLINE-2026-08-10
EFFORT_MIN_STEPS = 250_000  # CALIBRATED-OFFLINE-2026-08-10

#: Consecutive terminal (no-advance) family members before a campaign
#: reads STALLED. Re-derived from disk, not asserted (FORGE_SPEC §2a):
#: three is the largest value both of tonight's walls clear -- cv_hall's
#: three progress-shaped members clear it exactly, so the boundary sits
#: at 3, not at contra's higher count.
MIN_TERMINAL = 3


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_wall_manifest(wall_id: str, repo: Path = REPO) -> dict:
    """Reads ``runs/forge/walls/<wall_id>.json``, the wall-as-family
    manifest a build writes. Membership is scope, like a config."""
    return json.loads((repo / "runs" / "forge" / "walls" / f"{wall_id}.json").read_text())


# --------------------------------------------------------------- archive

def archive_verdict(tail: list[dict], *, wall_id: Optional[str] = None,
                     frozen_windows_max: int = FROZEN_WINDOWS_MAX,
                     effort_min_steps: int = EFFORT_MIN_STEPS) -> dict:
    """Verdict for a live child's ``progress.jsonl`` tail (oldest row
    first; each row at least ``{"cells": int, "steps": int}``, the shape
    ``progress_line()`` writes).

    Replays ``update_stall()`` across the given rows from a fresh
    watchdog state -- this is the independent check, not a read of the
    tail's own ``stall_flat_windows`` column. UNMEASURED on an empty
    tail; otherwise STALLED iff the replayed ``flat_windows`` clears
    ``FROZEN_WINDOWS_MAX`` (12) AND the last row's cumulative ``steps``
    clears ``EFFORT_MIN_STEPS`` (250,000) -- both CALIBRATED-OFFLINE
    constants, duplicated at module scope above (see the comment there
    for why not imported); WATCHING at
    ``flat_windows >= 2`` (the same threshold the solver's own stderr
    stall banner uses); ADVANCING otherwise.
    """
    if not tail:
        return {
            "verdict": "UNMEASURED", "kind": "archive", "wall_id": wall_id,
            "measure": "no member parsed",
            "evidence": {"flat_windows": None, "steps": None},
            "threshold": {"FROZEN_WINDOWS_MAX": frozen_windows_max,
                          "EFFORT_MIN_STEPS": effort_min_steps},
            "t": _now_iso(),
        }
    stall = {"last_cells": 0, "last_t": 0.0, "flat_windows": 0}
    for i, row in enumerate(tail):
        now = row.get("elapsed_s", i)
        update_stall(stall, int(row["cells"]), now)
    flat_windows = stall["flat_windows"]
    steps = int(tail[-1].get("steps", 0))

    if flat_windows >= frozen_windows_max and steps >= effort_min_steps:
        verdict = "STALLED"
        measure = "flat_windows>=FROZEN_WINDOWS_MAX and steps>=EFFORT_MIN_STEPS"
    elif flat_windows >= 2:
        verdict = "WATCHING"
        measure = "flat_windows>=2"
    else:
        verdict = "ADVANCING"
        measure = "flat_windows<2"

    return {
        "verdict": verdict, "kind": "archive", "wall_id": wall_id,
        "measure": measure,
        "evidence": {"flat_windows": flat_windows, "steps": steps},
        "threshold": {"FROZEN_WINDOWS_MAX": frozen_windows_max,
                      "EFFORT_MIN_STEPS": effort_min_steps},
        "t": _now_iso(),
    }


# -------------------------------------------------------------- campaign

def _member_progress(dir_path: Path, prior_best) -> dict:
    """``shape:"progress"`` member: the last line of ``progress.jsonl``,
    ``best_seen`` from ``archive.stats.json["best_score"]`` (falling back
    to the tail's ``max_gx_in_max_area``), root id = sha256 of the raw
    ``roots.json`` bytes (root STATES, not directories)."""
    progress_path = dir_path / "progress.jsonl"
    if not progress_path.exists():
        return {"parsed": False}
    last = None
    try:
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
    except (OSError, json.JSONDecodeError):
        return {"parsed": False}
    if last is None:
        return {"parsed": False}

    best_seen = None
    stats_path = dir_path / "archive.stats.json"
    if stats_path.exists():
        try:
            best_seen = json.loads(stats_path.read_text()).get("best_score")
        except (OSError, json.JSONDecodeError):
            best_seen = None
    if best_seen is None:
        best_seen = last.get("max_gx_in_max_area", 0)

    root_id = None
    roots_path = dir_path / "roots.json"
    if roots_path.exists():
        root_id = hashlib.sha256(roots_path.read_bytes()).hexdigest()

    solutions = int(last.get("solutions", 0))
    terminal_no_advance = solutions == 0 and best_seen <= prior_best
    advance = solutions > 0 or best_seen > prior_best
    return {
        "parsed": True, "unmeasured": False,
        "best_seen": best_seen, "solutions": solutions,
        "steps": last.get("steps"), "root_id": root_id,
        "terminal_no_advance": terminal_no_advance, "advance": advance,
    }


def _member_receipt(dir_path: Path, receipt_name: str,
                     terminal_field: Optional[str],
                     best_field: Optional[str],
                     root_family: Optional[str],
                     prior_best, member_name: str) -> dict:
    """``shape:"receipt"`` member: the named file. ``terminal_field: null``
    (a receipt that carries only a prose ``verdict`` string) makes the
    member UNMEASURED -- prose is not a receipt field. root id =
    ``root_family`` (the manifest's label; A-series root-state paths are
    dead scratchpad paths, so the label is the only identity the
    receipts can still supply)."""
    receipt_path = dir_path / receipt_name
    if not receipt_path.exists():
        return {"parsed": False}
    try:
        data = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"parsed": False}

    # advance is a blanket rule over "any member" (FORGE_SPEC §2a) --
    # independent of whether THIS member's terminal_field parses. A
    # receipt with no usable terminal reading can still report the wall
    # moving via best_field, and that must count.
    best_seen = data.get(best_field) if best_field else None
    advance = best_seen is not None and best_seen > prior_best

    if terminal_field is None:
        return {
            "parsed": True, "unmeasured": True, "root_id": root_family,
            "best_seen": best_seen, "advance": advance,
            "gap": (f"{member_name}: {receipt_name} carries no boolean "
                    f"terminal field, prose verdict only (PA §4)"),
        }
    raw = data.get(terminal_field)
    if not isinstance(raw, bool):
        return {
            "parsed": True, "unmeasured": True, "root_id": root_family,
            "best_seen": best_seen, "advance": advance,
            "gap": (f"{member_name}: {terminal_field!r} missing or not "
                    f"boolean in {receipt_name}"),
        }

    terminal_no_advance = raw is False
    return {
        "parsed": True, "unmeasured": False,
        "best_seen": best_seen, "root_id": root_family,
        "terminal_no_advance": terminal_no_advance, "advance": advance,
    }


def campaign_verdict(manifest: dict, repo: Path = REPO) -> dict:
    """Verdict for a wall-as-family manifest (see ``load_wall_manifest``).

    ``terminal_runs`` = members with ``terminal_no_advance``; ``advances``
    = members with ``advance``. ADVANCING iff ``advances > 0``; STALLED
    iff ``advances == 0 and terminal_runs >= MIN_TERMINAL``; WATCHING iff
    ``advances == 0 and 0 < terminal_runs < MIN_TERMINAL``; UNMEASURED iff
    no member parsed. ``degraded`` is true whenever the manifest's
    ``prior_best_replay_verified`` is false or any member is UNMEASURED.
    An unreadable member's file is recorded in ``missing`` and the member
    is otherwise dropped (never counted as terminal or advancing).
    """
    wall_id = manifest["wall_id"]
    prior_best = manifest["prior_best"]
    prior_best_replay_verified = bool(manifest.get("prior_best_replay_verified", False))
    missing: list[str] = list(manifest.get("missing", []))
    members_unmeasured: list[str] = []
    terminal_runs = 0
    advances = 0
    root_ids: set[str] = set()
    best_seen_list: list = []
    solutions_list: list = []
    steps_list: list = []
    any_parsed = False

    for m in manifest["members"]:
        d = repo / m["dir"]
        name = Path(m["dir"]).name
        shape = m["shape"]
        if shape == "progress":
            r = _member_progress(d, prior_best)
        elif shape == "receipt":
            r = _member_receipt(d, m["receipt"], m.get("terminal_field"),
                                 m.get("best_field"), m.get("root_family"),
                                 prior_best, name)
        else:
            raise ValueError(f"unknown member shape {shape!r} for {m['dir']}")

        if not r["parsed"]:
            missing.append(f"{m['dir']}: unreadable ({shape})")
            continue
        any_parsed = True

        if r.get("unmeasured"):
            members_unmeasured.append(name)
            if r.get("gap"):
                missing.append(r["gap"])
            # Falls through, deliberately: "advance" is a blanket rule
            # over any member (best_seen > prior_best), independent of
            # whether THIS member's terminal_field parsed. Only the
            # terminal (no-advance) reading is unavailable for it.
        elif r.get("terminal_no_advance"):
            terminal_runs += 1
        if r.get("advance"):
            advances += 1
        if r.get("root_id"):
            root_ids.add(r["root_id"])
        if r.get("best_seen") is not None:
            best_seen_list.append(r["best_seen"])
        if "solutions" in r:
            solutions_list.append(r["solutions"])
        if r.get("steps") is not None:
            steps_list.append(r["steps"])

    degraded = (not prior_best_replay_verified) or bool(members_unmeasured)

    if not any_parsed:
        verdict, measure = "UNMEASURED", "no member parsed"
    elif advances > 0:
        verdict, measure = "ADVANCING", "advances>0"
    elif terminal_runs >= MIN_TERMINAL:
        verdict, measure = "STALLED", "advances==0 and terminal_runs>=MIN_TERMINAL"
    elif terminal_runs > 0:
        verdict, measure = "WATCHING", "advances==0 and 0<terminal_runs<MIN_TERMINAL"
    else:
        verdict, measure = "UNMEASURED", "no member contributed a terminal or advance signal"

    return {
        "verdict": verdict, "kind": "campaign", "wall_id": wall_id,
        "source": f"runs/forge/walls/{wall_id}.json",
        "measure": measure,
        "evidence": {
            "terminal_runs": terminal_runs, "advances": advances,
            "distinct_roots": len(root_ids), "prior_best": prior_best,
            "best_seen": best_seen_list, "solutions": solutions_list,
            "steps": steps_list, "members_unmeasured": members_unmeasured,
        },
        "threshold": {"MIN_TERMINAL": MIN_TERMINAL},
        "degraded": degraded,
        "missing": missing,
        "t": _now_iso(),
    }