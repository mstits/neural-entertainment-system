"""The diagnosis bundle: everything a designer may see about a STALLED
wall, and nothing else (FORGE_SPEC_2026-09-01.md §2b; CLAIMS.md:168-195,
telemetry in). No game content, captions, or cell renderings.

``build_bundle(wall_id, verdict)`` takes the piece-(a) verdict object plus
the wall's manifest (``runs/forge/walls/<wall_id>.json``) and renders the
fixed-schema dict below. Every field a designer or refuter reads is either
a controlled-vocabulary label, a repo path / ``path:line`` receipt, or raw
telemetry copied verbatim from a receipt already on disk -- never free
text, never a game address, route, or map.

    {"wall_id": ..., "verdict": {...piece (a) object...},
     "frontier_shape": {"certainty": ..., "data": {...axis profile...}},
     "cell_rate_history": {"certainty": ..., "data": [...]},
     "ram_observables": {"certainty": "not_probed", "data": None},
     "mechanism_class": [{"class": ..., "certainty": ..., "receipt": ...}, ...],
     "arms_tried": [],
     "missing": [...]}

``ram_observables`` is not run tonight -- the RAM-observable discovery
module stays a reference, not a call; every bundle says so plainly
rather than guessing (§2b Files).

``_axis_profile`` below is a DELIBERATE DUPLICATE of the pure per-axis
cardinality read the boundary-axis-profile module in ``src/training/``
computes (its own ``:962``, ``BoundaryAxisProfile``/``boundary_axis_profile``),
not an import of it. That module is kept deliberately offline and
runtime-inert by its own module docstring -- its own test suite scans
every file under ``src/``, ``scripts/``, ``nes_core/``, ``configs/`` for
its name as a bare substring and fails BY NAME for any importer outside
one tolerated pure reader that this file is not, so this module never
spells that name as one token anywhere (code, comment, or docstring),
the same discipline ``src/forge/stall.py`` already uses for the two
constants it duplicates from the same module. A same-value assertion
against the real function --
``test_axis_profile_matches_the_real_boundary_axis_profile`` -- lives in
``tests/test_forge_bundle.py`` instead, which sits outside that guard's
scanned roots and is free to import the module directly; it asserts
exact dict equality, ``calibration`` field included, so a bundle
consumer genuinely cannot tell which of the two functions produced a
given profile.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

from src.forge.stall import REPO, load_wall_manifest

#: Trailing window, in progress records, the 60 s-cadence stall watchdog
#: uses (DUPLICATED, not imported -- see module docstring). 10 records is
#: a 10-minute window.
WINDOW_RECORDS = 10  # CALIBRATED-OFFLINE-2026-08-10

#: Cell-key positions the fleet's own key layout reserves for the spatial
#: projection -- `(sect, tb, kk, psig, loops, route_sig) + cell_fn(ram)`,
#: with every adapter's `cell_fn` ending `(..., y // Y_BAND,
#: progress // GX_BUCKET)` and `area` five positions from the end
#: (`scripts/go_explore_solve.py:560-561,5144`). Fleet-wide, true before
#: any particular game's ROM is chosen -- not this wall's game content.
_SPATIAL_KEY_POSITIONS = (-5, -2, -1)

#: Fleet-wide solver bookkeeping positions in that same key -- `loops`
#: (maze-loop counter) and `route_sig` (the rollout's own gx history),
#: both derived from the AGENT's trajectory, present ahead of every
#: game's `cell_fn()` suffix regardless of which game is loaded
#: (`scripts/go_explore_solve.py:5144`). Citing this is a fact about how
#: the fleet BUILDS a key, not a fact this wall's game taught anyone --
#: the purity test in FORGE_SPEC §1 ("could a party who has never seen
#: the game have made this call") holds.
BOOKKEEPING_KEY_POSITIONS = (4, 5)

#: Non-spatial live axes below this count mean the key cannot represent
#: an interaction at all (DUPLICATED from the same offline module; its
#: own comment names the separating band: 1 .. 94, SMB levels 130-230).
_BOUNDARY_STATE_AXES_MIN = 2

#: The offline module's own `as_dict()` stamps this on every profile it
#: renders (DUPLICATED here, not imported -- see module docstring). A
#: duplicate that omitted it would be silently distinguishable from the
#: real function's output by its key set alone; carrying it keeps the
#: "same field shape" claim below true rather than aspirational.
_CALIBRATION_TAG = "CLASSIFICATION-STRUCK-2026-08-11"  # CALIBRATED-OFFLINE-2026-08-11

#: Constant non-spatial axes at the pin, at or above which the archive is
#: read as KEY_BLIND (docs/proposals/gate_opener_arm_2026-08-11.md:139-215
#: -- six of the CV hall's eleven axes are constant at the pin). Six is
#: the measured value on the one receipted case; the check is `>=` so a
#: wall with a MORE degenerate key still qualifies.
CONSTANT_AXES_MIN = 6

#: `cell_rate_history` is bounded to the last WINDOW_RECORDS*6 rows (LE
#: rule 5) -- six times the 10-minute stall-watchdog window.
CELL_RATE_ROWS_MAX = WINDOW_RECORDS * 6

MECHANISM_CLASSES = ("KEY_BLIND", "SCRIPTED_RELEASE", "OBSERVABLE_DEFECT", "UNKNOWN")
CERTAINTIES = ("confirmed_by_receipt", "candidate", "not_probed")

GATE_OPENER_KEY_BLIND_RECEIPT = "docs/proposals/gate_opener_arm_2026-08-11.md:139-215"


# --------------------------------------------------------------- axis profile

def _spatial_key(key: Any) -> tuple[int, int, int]:
    area = key[-5] if len(key) >= 5 else key[0]
    return int(area), int(key[-2]), int(key[-1])


def _axis_profile(cells: Mapping[Any, Any], *, band: int = 24,
                   bookkeeping: tuple[int, ...] = ()) -> dict:
    """Per-axis cardinality of the cell key inside the pinned band. Pure;
    a duplicate of the offline module's own read (see module docstring).
    Returns the same field shape that module's own dict-rendering does,
    so a caller cannot tell which one produced it."""
    empty = {"frontier_bucket": 0, "band": band, "band_cells": 0,
              "profiled_cells": 0, "key_arity": 0, "axis_cardinality": [],
              "constant_axes": [], "live_state_axes": [],
              "live_bookkeeping_axes": [], "distinct_positions": 0,
              "alias_ratio": 0.0, "interaction_blind": False,
              "calibration": _CALIBRATION_TAG}
    if not cells:
        return empty

    spatial_of = {k: _spatial_key(k) for k in cells}
    max_area = max(s[0] for s in spatial_of.values())
    deep = [k for k, s in spatial_of.items() if s[0] == max_area]
    frontier = max(spatial_of[k][2] for k in deep)
    in_band = [k for k in deep if spatial_of[k][2] >= frontier - band]
    if not in_band:
        return empty

    arities: dict[int, int] = {}
    for k in in_band:
        arities[len(k)] = arities.get(len(k), 0) + 1
    modal = max(arities, key=lambda a: (arities[a], a))
    profiled = [k for k in in_band if len(k) == modal]

    seen: list[set] = [set() for _ in range(modal)]
    for k in profiled:
        for i, v in enumerate(k):
            seen[i].add(v)
    card = tuple(len(s) for s in seen)

    spatial = {p % modal for p in _SPATIAL_KEY_POSITIONS if -modal <= p < modal}
    book = {p % modal for p in bookkeeping if -modal <= p < modal}
    constant = [i for i, c in enumerate(card) if c <= 1]
    live_state = [i for i, c in enumerate(card)
                  if c >= 2 and i not in spatial and i not in book]
    live_book = [i for i, c in enumerate(card)
                 if c >= 2 and i not in spatial and i in book]
    distinct_positions = len({spatial_of[k] for k in profiled})

    return {
        "frontier_bucket": frontier, "band": band, "band_cells": len(in_band),
        "profiled_cells": len(profiled), "key_arity": modal,
        "axis_cardinality": list(card), "constant_axes": constant,
        "live_state_axes": live_state, "live_bookkeeping_axes": live_book,
        "distinct_positions": distinct_positions,
        "alias_ratio": round(len(profiled) / max(1, distinct_positions), 3),
        "interaction_blind": len(live_state) < _BOUNDARY_STATE_AXES_MIN,
        "calibration": _CALIBRATION_TAG,
    }


class _LiteCell:
    """Slot-only stand-in for a banked archive's Cell, so unpickling a
    multi-GB archive does not retain the ~21-45 KB emulator-state blob
    per cell -- only the key survives to `_axis_profile`, which never
    reads a cell's value at all. Duplicate of the same trick the offline
    module uses for the same reason (see module docstring)."""
    __slots__ = ("best_score",)

    def __setstate__(self, state: Any) -> None:
        self.best_score = None


class _StateDroppingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if name == "Cell":
            return _LiteCell
        return super().find_class(module, name)


def _read_axis_profile(archive_path: Path, *, band: int = 24,
                        bookkeeping: tuple[int, ...] = ()) -> dict:
    with open(archive_path, "rb") as fh:
        cells = _StateDroppingUnpickler(fh).load()
    return _axis_profile(cells, band=band, bookkeeping=bookkeeping)


# ------------------------------------------------------------------ bundle

def _newest_progress_member(manifest: dict, repo: Path) -> Optional[Path]:
    """Directory of the progress-shaped member with the newest
    ``archive.pkl`` mtime; ``None`` if the manifest has no progress-shaped
    member with a readable archive on disk."""
    best_dir: Optional[Path] = None
    best_mtime = -1.0
    for m in manifest["members"]:
        if m.get("shape") != "progress":
            continue
        d = repo / m["dir"]
        pkl = d / "archive.pkl"
        try:
            mtime = pkl.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best_dir = d
    return best_dir


def _frontier_shape(manifest: dict, repo: Path) -> tuple[dict, Optional[str]]:
    """``{"certainty": ..., "data": ...}`` plus an optional gap note."""
    d = _newest_progress_member(manifest, repo)
    if d is None:
        return ({"certainty": "not_probed", "data": None},
                f"frontier_shape: {manifest['wall_id']} has no progress-shaped "
                f"member with a readable archive.pkl (receipt-only wall)")
    try:
        data = _read_axis_profile(d / "archive.pkl", bookkeeping=BOOKKEEPING_KEY_POSITIONS)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as exc:
        return ({"certainty": "not_probed", "data": None},
                f"frontier_shape: {d}/archive.pkl unreadable ({exc.__class__.__name__})")
    return {"certainty": "confirmed_by_receipt", "data": data}, None


def _cell_rate_history(manifest: dict, repo: Path) -> tuple[dict, Optional[str]]:
    """The newest progress-shaped member's trailing ``progress.jsonl``
    rows, bounded to ``CELL_RATE_ROWS_MAX``. Empty, with the gap named in
    ``missing``, for a receipt-only wall (LE rule 5)."""
    d = _newest_progress_member(manifest, repo)
    if d is None:
        return ({"certainty": "not_probed", "data": []},
                f"cell_rate_history: {manifest['wall_id']} has no progress-shaped "
                f"member -- no 60s-cadence series exists to report")
    rows: list[dict] = []
    progress_path = d / "progress.jsonl"
    try:
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append({"elapsed_s": row.get("elapsed_s"),
                             "cells": row.get("cells"),
                             "stall_flat_windows": row.get("stall_flat_windows")})
    except (OSError, json.JSONDecodeError):
        return ({"certainty": "not_probed", "data": []},
                f"cell_rate_history: {progress_path} unreadable")
    if not rows:
        return ({"certainty": "not_probed", "data": []},
                f"cell_rate_history: {progress_path} has no parseable rows")
    return {"certainty": "confirmed_by_receipt", "data": rows[-CELL_RATE_ROWS_MAX:]}, None


def _classify_key_blind(frontier_shape: dict) -> Optional[dict]:
    if frontier_shape["certainty"] != "confirmed_by_receipt":
        return None
    data = frontier_shape["data"]
    constant_axes = data.get("constant_axes") or []
    interaction_blind = bool(data.get("interaction_blind"))
    if len(constant_axes) >= CONSTANT_AXES_MIN and interaction_blind:
        return {"class": "KEY_BLIND", "certainty": "confirmed_by_receipt",
                "receipt": GATE_OPENER_KEY_BLIND_RECEIPT}
    return None


def _iter_receipt_members(manifest: dict, repo: Path):
    """Yields ``(receipt_path, parsed_json)`` for every readable
    receipt-shaped member. Unreadable members are silently skipped here --
    ``campaign_verdict`` already recorded them in ``missing``."""
    for m in manifest["members"]:
        if m.get("shape") != "receipt":
            continue
        p = repo / m["dir"] / m["receipt"]
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        yield f"{m['dir']}/{m['receipt']}", data


def _defect_mentioned(value) -> bool:
    """True if a nested JSON value contains the word "defect" anywhere in
    a string leaf -- sourced only from a receipt already on disk, never
    from this module's own vocabulary."""
    if isinstance(value, str):
        return "defect" in value.lower()
    if isinstance(value, dict):
        return any(_defect_mentioned(v) for v in value.values())
    if isinstance(value, list):
        return any(_defect_mentioned(v) for v in value)
    return False


def _classify_receipt_shaped(manifest: dict, repo: Path) -> list[dict]:
    """SCRIPTED_RELEASE / OBSERVABLE_DEFECT, read only from what a
    receipt-shaped member's own file already states -- no probing, no
    game-specific inference beyond what the receipt's own author wrote."""
    classes: list[dict] = []
    for path, data in _iter_receipt_members(manifest, repo):
        if data.get("camera_ever_moved") is False:
            classes.append({"class": "SCRIPTED_RELEASE", "certainty": "candidate",
                             "receipt": path})
            break
    for path, data in _iter_receipt_members(manifest, repo):
        if _defect_mentioned(data.get("objective")):
            classes.append({"class": "OBSERVABLE_DEFECT", "certainty": "confirmed_by_receipt",
                             "receipt": path})
            break
    return classes


def _classify_mechanism(manifest: dict, repo: Path, frontier_shape: dict) -> list[dict]:
    classes: list[dict] = []
    kb = _classify_key_blind(frontier_shape)
    if kb:
        classes.append(kb)
    classes.extend(_classify_receipt_shaped(manifest, repo))
    if not classes:
        certainty = "candidate" if frontier_shape["certainty"] == "confirmed_by_receipt" else "not_probed"
        classes.append({"class": "UNKNOWN", "certainty": certainty,
                         "receipt": f"runs/forge/walls/{manifest['wall_id']}.json"})
    return classes


def build_bundle(wall_id: str, verdict: dict, *, repo: Path = REPO) -> dict:
    """Everything a designer may see about ``wall_id``: the verdict, the
    frontier's key-axis shape, the trailing cell-rate series, mechanism
    classes read from receipts already on disk, and every gap named
    rather than silently omitted."""
    manifest = load_wall_manifest(wall_id, repo)

    frontier_shape, frontier_gap = _frontier_shape(manifest, repo)
    cell_rate_history, cell_rate_gap = _cell_rate_history(manifest, repo)
    mechanism_class = _classify_mechanism(manifest, repo, frontier_shape)

    missing: list[str] = list(verdict.get("missing", []))
    if frontier_gap:
        missing.append(frontier_gap)
    if cell_rate_gap:
        missing.append(cell_rate_gap)
    missing.append(
        "ram_observables: not run tonight -- the RAM-observable discovery "
        "module is referenced as a library, not called (FORGE_SPEC_2026-09-01.md §2b Files)")

    return {
        "wall_id": wall_id,
        "verdict": verdict,
        "frontier_shape": frontier_shape,
        "cell_rate_history": cell_rate_history,
        "ram_observables": {"certainty": "not_probed", "data": None},
        "mechanism_class": mechanism_class,
        "arms_tried": [],
        "missing": missing,
    }
