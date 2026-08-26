"""Honest Go-Explore level solver for SMB (research-grounded pipeline, Phase 1).

Runs first-return-then-explore (Ecoffet et al. 2021) PURELY from a level's
ENTRANCE — no hand-crafted prefixes, no handoff seeds, no policy net. Search
(not gradient descent) does the hard exploration that model-free PPO cannot:
the deterministic Rust pool + microsecond save/restore lets the archive return
to any frontier cell for free, so a long correct path is discovered
incrementally instead of re-explored from scratch every episode.

This is the SOLVER half of the Go-Explore -> distillation pipeline (Blueprint
§2C). Solved trajectories are dumped to <out>/solutions/ as action traces +
provenance; the distillation step (BC + DAgger + 25% sticky) teaches a policy
to reproduce them robustly, and the honest metric is cold sticky+jitter from
the entrance. The search is a TEACHER; the deployed agent is the learned net.

Differences from the 2-1 campaign harvester (scripts/go_explore_2_1.py), which
crossed into hand-authored territory:
  * root ONLY at the level entrance (--root-state); NO prefix/handoff lineages.
  * lives-based death detection ($075A) — robust across enemy/pit/time deaths
    and multi-area levels, vs the fragile gx-drop heuristic.
  * (area-order, gx) progress so multi-area levels (1-2's entrance->underground)
    keep pushing the true frontier, not just the entrance area.
  * WARP GUARD: a level "clear" counts only when the world/level advance in the
    natural forward sequence (world unchanged, displayed level +1, OR a castle
    world-advance) — a warp-zone pipe (world jumps) is NOT a legit clear.

Usage:
  python scripts/go_explore_solve.py --out runs/ge_1_2 \
      --root-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
      --profile configs/mario_1_2_solo.yaml --workers 8 --minutes 20
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clear_reachability  # noqa: E402  (profile lint; no nes_core dependency)
from nes_core import Pool  # noqa: E402
from src.training import interaction_basis as ib  # noqa: E402
from src.training.go_explore import GoExploreArchive, keep_exploring  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
DEATH_STATES = (6, 11)  # RAM $000E player-state: dying / death-pit
NOOP = 0

# RAM addresses (SMB).
R_X_PAGE, R_X_LOW = 0x006D, 0x0086
R_PSTATE = 0x000E
R_YPOS = 0x00CE
R_PHASE = 0x0009
R_AREA = 0x0760
R_WORLD = 0x075F
R_LEVEL = 0x075C
R_LIVES = 0x075A


# ---------------------------------------------------------------------
# Hardware-flag selection + lineage provenance.
#
# A save-state blob records the machine's STATE, never the machine's
# TIMING CONFIGURATION — `Nes::State` carries {ram, mapper, cpu, ppu,
# apu, input, cycles} and nothing else, and every `set_hw_*` flag is
# sticky config that survives `load_worker_state` untouched. So loading
# someone else's entrance blob into a stock Pool silently runs a
# DIFFERENT machine than the one that produced it, and the divergence
# looks like a game/AI mystery rather than a config mismatch.
#
# That is not hypothetical: the Castlevania `cv_chain_hw2` lineage was
# built by a scratch driver under four hw flags that no committed code
# path reproduced, and replaying its block-3 entrance on a stock Pool
# dies in the hall around frame ~15,900 where the lineage survives — the
# whole "bat-wake mystery". Flags are opt-in here (DEFAULT EMPTY, so
# every existing seeded solve is bit-identical) and the resolved set is
# recorded under the key `hw_provenance` in roots.json,
# archive.stats.json and every solution receipt, plus a flattened
# `<blob>.state.json` sidecar next to every state this code writes.
#
# `hw_provenance`, NOT `provenance`: `provenance` is RESERVED for the
# honest-origin string marker (`"search"`) that solution receipts carry
# and CLAIMS.md audits compare against literally. Do not reuse it here
# — a dict literal that binds one key twice keeps only the last value,
# which silently deleted that marker once already (2026-08-07).
# ---------------------------------------------------------------------

def available_hw_flags() -> tuple:
    """Flag names the INSTALLED nes_core exposes, derived from `Pool`'s
    `set_hw_*` methods so this list can never go stale against the
    wheel — a name that needs a rebuild (e.g. `frame_anchor` before the
    Pool-side port lands) is reported as unavailable rather than dying
    in an AttributeError halfway through a run."""
    return tuple(sorted(n[len("set_hw_"):] for n in dir(Pool)
                        if n.startswith("set_hw_")))


def resolve_hw_flags(profile: dict, cli: str | None = None) -> list:
    """Resolve the hw-flag set for this run: `--hw-flags` wins over the
    profile's `solve.hw_flags:` list, and the DEFAULT IS EMPTY.

    `--hw-flags none` (or an empty string) forces the empty set even
    when the profile pins one. Names are validated against the
    installed core and de-duplicated with order preserved."""
    if cli is not None:
        raw = cli.strip()
        names = ([] if raw.lower() in ("", "none")
                 else [p.strip() for p in raw.split(",") if p.strip()])
    else:
        names = list(profile.get("solve", {}).get("hw_flags") or [])
    out, seen = [], set()
    for n in names:
        n = str(n).strip()
        if n.startswith("set_hw_"):        # tolerate the method spelling
            n = n[len("set_hw_"):]
        if n in seen:
            continue
        if n not in available_hw_flags():
            raise SystemExit(
                f"[go_explore_solve] unknown hw flag {n!r}. The installed "
                f"nes_core exposes: {', '.join(available_hw_flags())}. "
                f"(A flag that exists in nes_core/src but not here needs a "
                f"wheel rebuild + the stale-.so copy step.)")
        seen.add(n)
        out.append(n)
    return out


def apply_hw_flags(pool, flags) -> None:
    """Apply resolved hw flags to a freshly-constructed Pool.

    ORDER IS LOAD-BEARING: call this after `Pool(...)` and BEFORE
    `reset_all()`. `reset_alignment` changes boot cycle accounting, so
    setting it after the reset boots a different power-on lineage
    (measured 2026-08-07: RAM diverges at frame 11, $006F/$01FE). The
    other flags are order-insensitive; they are applied together here
    so there is exactly one ordering rule to remember. This is the same
    recipe scripts/tracing/build_cv_tape.py documents for the CV tape."""
    for name in flags:
        getattr(pool, f"set_hw_{name}")(True)


def _core_build_id() -> dict:
    """Identify the nes_core binary actually loaded. The dist version is
    static across rebuilds, so the extension module's digest is what
    actually pins the machine a lineage was produced on."""
    if _core_build_id._cache is None:
        import importlib.metadata as md

        import nes_core
        info = {"dist_version": None, "module": None, "sha256_16": None}
        try:
            info["dist_version"] = md.version("nes_core")
        except Exception:
            pass
        try:
            so = Path(nes_core.__file__).with_name("nes_core.abi3.so")
            if not so.exists():
                so = Path(nes_core.nes_core.__file__)
            info["module"] = so.name
            info["sha256_16"] = hashlib.sha256(so.read_bytes()).hexdigest()[:16]
        except Exception:
            pass
        _core_build_id._cache = info
    return dict(_core_build_id._cache)


_core_build_id._cache = None


def hw_provenance(flags, frame_skip: int) -> dict:
    """The machine description that must travel with any state blob."""
    return {"hw_flags": list(flags), "frame_skip": int(frame_skip),
            "nes_core": _core_build_id()}


def sidecar_path(state_path) -> Path:
    """`foo.state` -> `foo.state.json`. Sidecar, NOT the blob: the
    savestate format stays frozen and config never enters it."""
    p = Path(state_path)
    return p.with_name(p.name + ".json")


def write_state_sidecar(state_path, provenance: dict, extra: dict | None = None):
    """Record the machine a state blob was produced on, next to it."""
    rec = dict(provenance)
    if extra:
        rec.update(extra)
    rec["blob"] = Path(state_path).name
    path = sidecar_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def check_state_sidecar(state_path, flags) -> bool:
    """Warn LOUDLY when a root blob's recorded lineage disagrees with the
    flags this run is about to use. Returns True when they agree (or no
    sidecar exists to check against). Deliberately non-fatal: an
    unlabelled blob is the common case until backfill catches up, and a
    warning that names both sets is what turns an invisible config
    mismatch into a five-second diagnosis."""
    path = sidecar_path(state_path)
    if not path.exists():
        return True
    try:
        want = list(json.loads(path.read_text()).get("hw_flags", []))
    except (OSError, ValueError) as e:
        print(f"[go_explore_solve] unreadable state sidecar {path}: {e}",
              flush=True)
        return True
    if sorted(want) == sorted(flags):
        return True
    print(f"[go_explore_solve] *** HW-FLAG LINEAGE MISMATCH *** "
          f"{Path(state_path).name} was built with {want or '[]'} but this "
          f"run uses {list(flags) or '[]'}. The restored machine is not the "
          f"machine that produced this state; expect divergence that looks "
          f"like nondeterminism. Pass --hw-flags {','.join(want) or 'none'} "
          f"to match the lineage.", flush=True)
    return False


def stamp_stats_provenance(archive_path, provenance: dict,
                           key_config: dict | None = None) -> None:
    """Merge run provenance into the archive's stats sidecar.

    `GoExploreArchive.save()` owns archive.stats.json; this re-opens it
    right after and adds the machine description under `hw_provenance`,
    so a banked run can be re-created without guessing which flags it
    ran under. Uses its own `.prov.tmp` scratch name so it can never
    race the archive's own `.stats.json.tmp`.

    The key is `hw_provenance`, never `provenance`: `provenance` is the
    reserved honest-origin marker (`"search"`) on solution receipts, and
    one token must not mean a string in one artifact and a dict in its
    sibling.

    `key_config` (optional) records the CELL-KEY SCHEMA the archive was
    built under — see `key_config_axes`. Without it, a later
    `--resume-archive` can only check the machine and has to take the
    key on faith, which is exactly the hole that let a 1-flag lineage
    be resumed into a 4-flag run over a disjoint tb/kk key subspace
    (GATE_OPENER_CAMPAIGN_2026-08-11 §12)."""
    p = Path(archive_path).with_suffix(".stats.json")
    try:
        stats = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    want = dict(key_config) if key_config else None
    if (stats.get("hw_provenance") == provenance
            and (want is None or stats.get("key_config") == want)):
        return
    stats["hw_provenance"] = provenance
    if want is not None:
        stats["key_config"] = want
    tmp = p.with_name(p.name + ".prov.tmp")
    try:
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, p)
    except OSError as e:
        print(f"[go_explore_solve] could not stamp provenance into {p}: {e}",
              flush=True)


# ---------------------------------------------------------------------
# --resume-archive LINEAGE CHECK.
#
# `--resume-archive` loads someone else's archive.pkl straight into this
# run's archive. Cells are only comparable if they were keyed the same
# way AND produced on the same machine, and nothing checked either: the
# D3 lineage check (GATE_OPENER_CAMPAIGN_2026-08-11 §12) found
# runs/cv_chain_hw/lvl_03_overnight — 560,410 cells, cited as the
# deepest banked attack on the wall — was built under ONE hw flag
# against four, with a disjoint tb/kk key subspace, so 560,410 cells
# collapsed to 88,212 on a like-for-like recount. That resume would
# have run silently.
#
# The comparison is two-sided and deliberately narrow: the axes below
# are the ones that change what a cell KEY MEANS (so the resumed cells
# partition a different space) or what a state BLOB restores into (so
# the resumed savestates run a different machine). Cosmetic differences
# — out-dir, seed, workers, minutes — are not lineage.
# ---------------------------------------------------------------------

#: Sidecar axes compared on resume, with the human name used in the report.
LINEAGE_KEY_AXES = (
    ("tb", "time-bin key prefix (--time-bins)"),
    ("kk", "kill-count key cardinality (--kill-key)"),
    ("sig_arity", "state_sig bit count (profile + --gate-axes sidecar)"),
    ("sig_sha", "state_sig contents (addresses/matches/mods)"),
    ("gx_bucket", "gx bucket px (--gx-bucket)"),
    ("y_band", "y band px (--y-band)"),
    ("cell_fn", "cell-key adapter"),
    # Room-fingerprint schema (solve.room_fp): with it on, room ordinals
    # ride the psig/area key slots, so two archives interned under
    # different masks/settle/classifier constants — or one with and one
    # without the feature — do not partition the same space. "" = off.
    # ABSENT reads as "" (see resume_lineage_diff): every archive banked
    # before the feature existed was necessarily built with it off, and
    # refusing all of them as UNVERIFIABLE would break flags-off resume
    # compatibility for a schema they could not have carried.
    ("room_fp", "room fingerprint config (solve.room_fp)"),
)


def state_sig_sha(spec) -> str:
    """A stable 8-hex digest of a `state_sig` spec (empty for no sig).

    `sig_arity` alone cannot separate two sigs with the same BIT COUNT
    over different bytes, which is exactly what the --gate-axes sidecar
    produces: it merges axes into `state_sig` before the adapter reads
    it, so one arm's 1-bit contact sig and another's 1-bit mode sig
    would compare equal and their cells would be silently mixed. Axis
    ORDER is part of the digest (axis i owns bit i, so a reordering is a
    different key), each axis's `match` set is canonicalised (sorted) so
    a cosmetic profile re-listing is not a new lineage."""
    if not spec:
        return ""
    canon = [[int(addr), sorted(int(v) for v in match), int(mod or 0)]
             for addr, match, mod in spec]
    return hashlib.sha256(
        json.dumps(canon, separators=(",", ":")).encode()).hexdigest()[:8]


def key_config_axes(args, game) -> dict:
    """The cell-key SCHEMA this run will build, as JSON scalars.

    `tb`/`kk` are recorded as CARDINALITIES rather than as the flags
    that set them, because that is what actually decides whether two
    archives partition the same space: --time-bins off contributes one
    constant slot, on contributes `bit_length` bins; --kill-key off is
    one slot, on is the 0..15 cap in observe().

    `smooth` is RECORDED BUT NOT COMPARED (it is deliberately absent
    from LINEAGE_KEY_AXES): the torn-read filter changes which samples
    become cells, not what a key means, and resuming an unfiltered
    archive into a filtered run is the repair §12 asks for — refusing it
    would refuse the fix. It still has to travel with the archive,
    because those banked cells were screened by a different rule than
    the ones this run will add; `resume_lineage_diff` reports the
    difference as UNVERIFIABLE rather than as a mismatch."""
    return {
        "tb": 0 if not bool(getattr(args, "time_bins", False)) else 1,
        "kk": 1 if not bool(getattr(args, "kill_key", False)) else 16,
        "sig_arity": int(getattr(game, "state_sig_arity", 0) or 0),
        "sig_sha": str(getattr(game, "state_sig_sha", "") or ""),
        "gx_bucket": int(getattr(args, "gx_bucket", GX_BUCKET)),
        "y_band": int(getattr(args, "y_band", Y_BAND)),
        "cell_fn": type(game).__name__,
        "smooth": str(getattr(game, "progress_smooth", "off") or "off"),
        # sha8 over the identity-bearing solve.room_fp knobs (mask,
        # settle, classifier constants, palette_cokey, max_rooms);
        # "" when the profile has no room_fp block. Compared as a
        # LINEAGE axis: interned ordinals ride the psig/area slots, so
        # two intern schemas never partition the same space.
        "room_fp": str(getattr(game, "room_fp_sha", "") or ""),
    }


def resume_lineage_diff(prev_stats: dict, provenance: dict,
                        key_config: dict) -> dict:
    """Compare a resumed archive's recorded lineage against this run's.

    Pure. Returns `{"mismatch": [...], "unverifiable": [...],
    "caveats": [...]}`, all lists of one-line human-readable strings.

      MISMATCH      a recorded axis that DISAGREES.
      UNVERIFIABLE  an axis the banked sidecar never recorded, so no
                    comparison could be made. NOT agreement — this is
                    the tier the §12 archive lands in
                    (runs/cv_chain_hw/lvl_03_overnight records neither
                    hw_provenance nor key_config, so the check that
                    warned and resumed returned mismatch=[] on the exact
                    archive it was written for). The caller refuses on
                    it; `recover_lineage` is what turns as much of it as
                    the disk can answer into a real comparison first.
      CAVEATS       recorded, different, and NOT lineage: the torn-read
                    filter changes which samples become cells, not what
                    a key means, and resuming an unfiltered archive into
                    a filtered run is the repair §12 asks for. Warned
                    about, never refused — a refusal here would refuse
                    the fix.
    """
    prev_stats = prev_stats or {}
    mismatch, unverifiable, caveats = [], [], []

    prov = prev_stats.get("hw_provenance")
    if not isinstance(prov, dict):
        unverifiable.append(
            "hw_provenance: absent from the resumed archive.stats.json — "
            "the machine that produced those state blobs is unrecorded")
    else:
        got, want = list(prov.get("hw_flags") or []), list(
            provenance.get("hw_flags") or [])
        if sorted(got) != sorted(want):
            mismatch.append(
                f"hw_flags: archive {got or '[]'} vs run {want or '[]'} — "
                f"restored savestates run a different machine")
        if int(prov.get("frame_skip", -1)) != int(
                provenance.get("frame_skip", -1)):
            mismatch.append(
                f"frame_skip: archive {prov.get('frame_skip')} vs run "
                f"{provenance.get('frame_skip')}")
        a = (prov.get("nes_core") or {}).get("sha256_16")
        b = (provenance.get("nes_core") or {}).get("sha256_16")
        if a and b and a != b:
            mismatch.append(f"nes_core: archive {a} vs run {b} — different "
                            f"emulator binary")

    prev_key = prev_stats.get("key_config")
    if not isinstance(prev_key, dict):
        unverifiable.append(
            "key_config: absent from the resumed archive.stats.json — the "
            "cell-key schema those cells were built under is unrecorded "
            "(archives flushed before this field existed)")
    else:
        for axis, label in LINEAGE_KEY_AXES:
            if axis not in prev_key:
                if axis == "room_fp":
                    # Predates the room-fingerprint layer => built with
                    # it off, which is a FACT, not a gap: match a
                    # room_fp-off run silently (flags-off resume stays
                    # byte-identical to pre-roomgraph behavior), refuse
                    # a room_fp-on run as a real schema mismatch (its
                    # cells embed no ordinals to compare against).
                    if key_config.get(axis, ""):
                        mismatch.append(
                            f"{label}: archive predates room "
                            f"fingerprinting (off) vs run "
                            f"{key_config.get(axis)!r} — the resumed "
                            f"cells carry no room ordinals in their "
                            f"psig/area slots")
                    continue
                unverifiable.append(f"key_config.{axis} ({label}): not "
                                    f"recorded by the resumed archive")
                continue
            if prev_key[axis] != key_config.get(axis):
                mismatch.append(
                    f"{label}: archive {prev_key[axis]!r} vs run "
                    f"{key_config.get(axis)!r} — the two runs' cells do not "
                    f"partition the same space")

    # Torn-read filter: a WARNING, never a refusal (see key_config_axes).
    # Absent reads as "off" on both sides, so the default path and every
    # legacy archive stay silent; the line only appears once someone
    # actually arms the filter, which is when the caveat matters.
    prev_smooth = str((prev_key if isinstance(prev_key, dict) else {})
                      .get("smooth", "off") or "off")
    run_smooth = str(key_config.get("smooth", "off") or "off")
    if prev_smooth != run_smooth:
        caveats.append(
            f"progress smoothing: archive {prev_smooth!r} vs run "
            f"{run_smooth!r} — the resumed cells were recorded under a "
            f"different torn-read rule, so banked frontier cells are not "
            f"screened the way this run screens new ones (§12: the gx-767 "
            f"phantom is already banked in every hall archive)")
    return {"mismatch": mismatch, "unverifiable": unverifiable,
            "caveats": caveats}


def recover_lineage(prev_dir, prev_stats: dict) -> tuple:
    """Backfill a resumed archive's `hw_provenance` from the rest of its
    own out-dir. Returns `(provenance or None, [source notes])`.

    THE PROOF IS USUALLY ON DISK, UNREAD. archive.stats.json only grew
    an `hw_provenance` field partway through the campaign, so the §12
    archive — the one the whole check exists for — records none, and a
    comparison against nothing returned "no mismatch". But that run's
    roots.json names its root blob, and the blob's own sidecar
    (`foo.state.json`, written by `write_state_sidecar`) records the
    flags it was built under: for runs/cv_chain_hw/lvl_03_overnight that
    is ["mmio_read_timing"] — one flag against the arms' four, which is
    exactly the §12 verdict, sitting one file away from the check that
    could not reach it.

    Sources, most authoritative first: the stats sidecar itself, then
    any root's recorded `hw_provenance` in roots.json, then the root
    blob's own state sidecar. Every recovery is NAMED in the notes,
    because a recovered lineage is weaker evidence than a recorded one
    and a receipt has to say which it had.
    """
    prev_stats = prev_stats or {}
    if isinstance(prev_stats.get("hw_provenance"), dict):
        return prev_stats["hw_provenance"], []
    d = Path(prev_dir)
    try:
        roots = json.loads((d / "roots.json").read_text())
    except (OSError, ValueError):
        return None, []
    if not isinstance(roots, dict):
        return None, []
    for rid, info in roots.items():
        if isinstance(info, dict) and isinstance(info.get("hw_provenance"),
                                                 dict):
            return info["hw_provenance"], [
                f"hw_provenance RECOVERED from roots.json[{rid}] (the "
                f"archive.stats.json predates the field)"]
    for rid, info in roots.items():
        path = (info or {}).get("path") if isinstance(info, dict) else None
        if not path:
            continue
        side = sidecar_path(path)
        try:
            rec = json.loads(Path(side).read_text())
        except (OSError, ValueError):
            continue
        if "hw_flags" not in rec:
            continue
        prov = {"hw_flags": list(rec.get("hw_flags") or []),
                "frame_skip": int(rec.get("frame_skip", -1)),
                "nes_core": dict(rec.get("nes_core") or {})}
        note = (f"hw_provenance RECOVERED from the root blob's sidecar "
                f"{Path(side).name} (roots.json[{rid}])")
        if rec.get("provenance_source"):
            note += f" — {rec['provenance_source']}"
        return prov, [note]
    return None, []


#: Cell-key arity the frontier readers index against:
#: (sect, tb, kk, psig, loops, route_sig) + cell_fn(ram)[5].
KEY_ARITY = 11


def key_schema_from_keys(keys) -> dict:
    """What the archive's OWN keys say about the schema they were built
    under. Pure.

    The recorded half of the lineage check is unverifiable for every
    archive on disk (no `key_config` was ever flushed), but the keys
    themselves are not silent: `tb` is a bit-length slot that is >= 1
    whenever --time-bins is on and exactly 0 when it is off, `kk` is a
    capped kill count that can only be nonzero with --kill-key on, and
    the state_sig slot is a bitmask whose width cannot exceed the
    profile's declared arity. seed() already walks every loaded key
    twice (frontier trackers, then local coverage); this rides along.

    Returns {"n", "arity", "tb_max", "kk_max", "sig_bits"}, with arity
    None on an empty archive.
    """
    n = tb = kk = sig = 0
    arity = None
    for k in keys:
        n += 1
        if arity is None:
            arity = len(k)
        elif len(k) != arity:
            arity = -1
        if len(k) >= KEY_ARITY:
            tb = max(tb, int(k[1]))
            kk = max(kk, int(k[2]))
            sig = max(sig, int(k[-3]))
    return {"n": n, "arity": arity, "tb_max": tb, "kk_max": kk,
            "sig_bits": int(sig).bit_length()}


def key_schema_conflicts(observed: dict, key_config: dict,
                         arity: int | None = None) -> dict:
    """Compare `key_schema_from_keys` against the schema this run builds.

    Same two tiers as `resume_lineage_diff`: a CONTRADICTION is a key
    the run could not have produced, an UNVERIFIABLE is an axis whose
    "off" reading is also what an "on" run with no events looks like
    (--kill-key with nothing killed yet). Pure.
    """
    mismatch, unverifiable = [], []
    if not observed.get("n"):
        return {"mismatch": mismatch, "unverifiable": unverifiable}
    want = int(arity if arity is not None else KEY_ARITY)
    got = observed.get("arity")
    if got == -1:
        mismatch.append("cell-key arity: the resumed archive mixes key "
                        "lengths — its cells were not all built by one "
                        "schema")
    elif got is not None and int(got) != want:
        mismatch.append(
            f"cell-key arity: archive keys are {got}-tuples, this run "
            f"builds {want}-tuples — every positional frontier read "
            f"(area, y band, gx bucket) lands on a different slot")
    if observed["tb_max"] and not int(key_config.get("tb", 0)):
        mismatch.append(
            f"time-bin key prefix: archive keys carry a nonzero time bin "
            f"(max {observed['tb_max']}) and this run builds none — the "
            f"two runs' cells do not partition the same space")
    if not observed["tb_max"] and int(key_config.get("tb", 0)):
        mismatch.append(
            "time-bin key prefix: this run bins by time and every "
            "resumed key has bin 0, which --time-bins never emits")
    if observed["kk_max"] and int(key_config.get("kk", 1)) <= 1:
        mismatch.append(
            f"kill-count key: archive keys carry a nonzero kill slot "
            f"(max {observed['kk_max']}) and this run builds none")
    if not observed["kk_max"] and int(key_config.get("kk", 1)) > 1:
        unverifiable.append(
            "kill-count key: this run keys on kills and no resumed key "
            "records one — consistent with an archive that killed "
            "nothing, and with one built without the flag")
    if observed["sig_bits"] > int(key_config.get("sig_arity", 0) or 0):
        mismatch.append(
            f"state_sig bit count: archive keys use {observed['sig_bits']} "
            f"bit(s) and this run declares "
            f"{int(key_config.get('sig_arity', 0) or 0)}")
    elif observed["sig_bits"] < int(key_config.get("sig_arity", 0) or 0):
        unverifiable.append(
            f"state_sig bit count: this run declares "
            f"{int(key_config.get('sig_arity', 0) or 0)} bit(s) and the "
            f"resumed keys only ever set {observed['sig_bits']} — the "
            f"unset bits cannot be told from absent ones, and the sig "
            f"CONTENTS (which bytes, which matches) are unrecoverable "
            f"from a key at all")
    return {"mismatch": mismatch, "unverifiable": unverifiable}


def format_lineage_report(path, diff: dict, allowed: bool,
                          notes=(), unverified_allowed: bool = False) -> str:
    """The operator-facing text for `resume_lineage_diff`."""
    head = ("*** RESUME LINEAGE MISMATCH ***" if diff["mismatch"]
            else "*** RESUME LINEAGE UNVERIFIABLE ***"
            if diff.get("unverifiable") else "resume lineage: note")
    lines = [f"[go_explore_solve] {head} {path}"]
    lines += [f"    recovered {n}" for n in notes]
    lines += [f"    MISMATCH  {m}" for m in diff["mismatch"]]
    lines += [f"    UNCHECKED {u}" for u in diff.get("unverifiable", ())]
    lines += [f"    note      {c}" for c in diff.get("caveats", ())]
    if diff["mismatch"]:
        lines.append(
            "    Resuming across lineages silently mixes cells that were "
            "never comparable (campaign §12: 560,410 cells -> 88,212 on "
            "collapse). Re-run with matching flags, or pass "
            "--allow-lineage-mismatch to proceed anyway."
            if not allowed else
            "    --allow-lineage-mismatch given: proceeding, and this "
            "run's receipts inherit a mixed-lineage archive.")
    elif diff.get("unverifiable"):
        lines.append(
            "    An axis nothing recorded is not an axis that matches. The "
            "§12 archive records neither field and its ROOT SIDECAR says "
            "one hw flag against four, so 'no mismatch' was a hole and not "
            "a pass. Re-run with an archive that carries its lineage, or "
            "pass --allow-unverified-lineage to resume on faith."
            if not unverified_allowed else
            "    --allow-unverified-lineage given: proceeding, and this "
            "run's receipts inherit an UNVERIFIED lineage.")
    return "\n".join(lines)


def _gx(ram) -> int:
    return (int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW])


def _wd(ram) -> tuple:
    """Displayed (world, level), both 0-indexed."""
    return (int(ram[R_WORLD]), int(ram[R_LEVEL]))


#: Pseudo-addresses where Solver._xram appends the scroll-odometer
#: integral (little-endian, 3 bytes, clamped to [0, 0xFFFFFF]) past the
#: 2KB RAM snapshot. Indexing them on an unextended snapshot is a LOUD
#: IndexError by design — a path that forgot the extension must not
#: silently read zeros as progress.
ODO_LO = 0x800

# ---------------------------------------------------------------------
# ROOM-GRAPH ENGINE — identity layer (ROOMGRAPH_ENGINE_2026-08-24 §2).
#
# Room identity = a SETTLED, masked blake2b-64 hash of the 2 KB physical
# nametable VRAM (Pool.peek_nametables — a hardware surface, v23-legal),
# interned to a discovery-order ordinal and written into the pseudo-RAM
# extension at 0x804/0x805 beside the odometer bytes above. Profiles
# then point `solve.area` / `solve.room_sig` / `solve.room_advance` at
# these pseudo-addresses, so room-keyed cells, sect/psig transits and
# door-entry macros all come from SHIPPED machinery — exactly the way
# the scene ordinal already rides 0x803. Scene is CLASSIFIER EVIDENCE,
# never identity: the probes measured it noisy in Metroid (spurious
# bumps at clamp/seam) and blind to Zelda fades / Rygar blank doors.
#
# Everything below is pure and unit-testable without a Solver, a ROM or
# a Pool (tests/test_room_fp.py); the hot-loop wiring is T2's.
# ---------------------------------------------------------------------

#: Pseudo-addresses of the room ordinal (little-endian uint16) in the
#: extended snapshot, and the ORTHOGONAL odometer axis byte — the axis
#: _xram's 0x800..0x802 integral does NOT carry, compressed to
#: (clamp(v, 0, 0xFFFFFF) >> 4) & 0xFF so a Metroid profile can key
#: y-bands on it (`y: 0x806`). Only present when `solve.room_fp` is
#: configured: the extension is 7 bytes then, 4 otherwise, so every
#: non-fp profile keeps its exact snapshot identity.
ROOM_LO, ROOM_HI = 0x804, 0x805
ODO_ALT = 0x806
#: The "no settled room yet" sentinel a worker carries from _assign
#: until its first fingerprint settles. Never interned (the intern
#: table caps at 4096), so it can never alias a real ordinal.
ROOM_UNKNOWN = 0xFFFF

# ---------------------------------------------------------------------
# FIGHT-GATE — cumulative-damage progress for camera-static combat games
# with no spatial frontier at all (FIGHTGATE_MECHANISM_2026-08-25.md §4).
# Punch-Out is the validation target: both odometer axes read flat/noise
# (the CAMERA_STATIC_AGENT_ACTIVE receipt quoted in that document's §1),
# so a discovered foe-HP byte (`scripts/discover_observables.py`'s new
# `find_fight_health`) is integrated into a monotone "total damage dealt"
# frontier the SAME way the odometer's camera integral is: written into
# the pseudo-RAM extension so cells, the glitch filter and macros all
# read it through the ordinary lo/hi path, with NO new cell/archive code.
# ---------------------------------------------------------------------

#: Pseudo-address pair for the cumulative-damage integral (little-endian
#: uint16, clamped [0, 0xFFFF]) — the next free slot after the
#: odometer/room-fp block above. A fight_gate profile always reserves
#: the FULL 0x800..0x808 range (9 bytes), even on a profile with no
#: odometer/room_fp of its own, so this pair's ABSOLUTE address never
#: moves: FIGHT_LO/FIGHT_HI are indices into the extended snapshot, not
#: offsets from wherever the odometer block happened to end. Every
#: non-fight-gate profile is untouched — this range is only allocated
#: when `self._fight` is True (see Solver._xram).
FIGHT_LO, FIGHT_HI = 0x807, 0x808

#: Mass-RAM-rewrite churn floor for the "no round byte" fallback
#: (a fixed-ring game like Punch-Out — §4.2: no round/opponent-index
#: byte exists, so a bout boundary is detected the same way
#: discover_observables.Discoverer.reset_threshold's floor calibrates
#: it offline: a death/level-reload/bout-transition rewrites far more
#: of RAM than one frame of ordinary play). A profile with a `round`
#: byte never touches this path at all.
FIGHT_RESET_THRESHOLD = 350.0


def fight_gate_step(prev_hp, now_hp: int, *, is_transition: bool, cum: int,
                    cap: int = 0xFFFF) -> tuple[int, int]:
    """One step of the fight-gate cumulative-damage integral (design doc
    §4.1). Pure: no RAM array, no Solver, no emulator — every branch is
    exercised directly with plain integers
    (tests/test_fight_gate.py).

    `prev_hp` is the foe-HP byte's PREVIOUS reading (None the first time
    a fresh worker is observed — there is nothing to diff against yet).
    `now_hp` is this step's reading. `is_transition` is True on the step
    a round-gate byte advances (or, with no round byte, the mass-RAM-
    reset detector fires) — the opponent-refill boundary.

    On a transition (or the very first observation), the running
    baseline is RE-ARMED to `now_hp` and NO delta is banked for that
    step: comparing a fresh opponent's full HP against whatever the
    PREVIOUS opponent's near-empty reading happened to be (or a load
    frame's transient garbage) is exactly the `foe_hp_start - 0` fake
    windfall the design names — the fix is to never compute a delta
    across the boundary at all, only re-baseline from it.

    Every other step banks `max(0, prev_hp - now_hp)` — a DROP is
    damage; a rise (a refill mid-probe with no round/reset signal at
    all, or read noise the wrong way) contributes nothing. Same non-
    negative-clamp discipline the PPU scroll odometer integral already
    uses ("regress is not negative progress, it is no progress").

    Returns `(new_cum, new_prev_hp)`, both written back by the caller.
    """
    if prev_hp is None or is_transition:
        return cum, int(now_hp)
    dropped = int(prev_hp) - int(now_hp)
    if dropped > 0:
        cum = min(cap, cum + dropped)
    return cum, int(now_hp)


def fight_gate_mass_reset(prev_ram, ram, *, threshold: float) -> bool:
    """The generic mass-RAM-rewrite signature
    `discover_observables.Discoverer._first_reset` already uses to find
    a death or level reload, reused here to flag a bout boundary for a
    fixed-ring fight-gate profile with no round byte. `prev_ram` is the
    previous step's raw snapshot (None on a worker's first observed
    step — never a reset). Pure: `threshold` is whatever the caller
    calibrated."""
    if prev_ram is None:
        return False
    n = min(len(prev_ram), len(ram))
    changed = int(np.count_nonzero(
        np.asarray(prev_ram[:n]).astype(np.int16)
        != np.asarray(ram[:n]).astype(np.int16)))
    return changed > threshold


def room_fp_mask(ranges) -> np.ndarray:
    """np.uint8[2048] KEEP-mask over the nametable snapshot: 1 keeps a
    byte, 0 zeroes it before hashing. `ranges` = the profile's volatile
    [lo, hi) byte spans (animated tiles, HUD counters), emitted by the
    calibration script from variance over OUR OWN idle/walk frames —
    no human reads the screen. Attribute bytes participate unless a
    range says otherwise."""
    m = np.ones(2048, dtype=np.uint8)
    for lo, hi in ranges:
        m[int(lo):int(hi)] = 0
    return m


def nt_fingerprint(nt, mask, palette=None) -> int:
    """blake2b-64 of the masked 2 KB nametable snapshot, as an int.

    blake2b (stdlib C) rather than a pure-Python FNV: 2 KB x workers x
    steps makes the hash a hot-loop cost — an instrumentation choice,
    not doctrine (noted as a deviation in the design). `palette` folds
    the 32-byte palette RAM in as an optional co-key (default off:
    fades must not fork rooms)."""
    a = (np.frombuffer(nt, dtype=np.uint8)
         if isinstance(nt, (bytes, bytearray)) else nt)
    d = hashlib.blake2b(digest_size=8)
    d.update((a * mask).tobytes())
    if palette is not None:
        d.update(bytes(palette))
    return int.from_bytes(d.digest(), "little")


def fp_settle(pend, h, settled_h, odo_xy, scene, step, settle):
    """One worker's fingerprint-settle state transition (pure).

    `pend` is the burst ctx's `fp_pend`: `(hash, n_consecutive,
    onset_odo_xy, onset_scene, onset_step) | None`. A pend exists only
    while the sampled hash differs from the worker's settled hash; its
    ONSET fields are captured at the first diverging sample and
    PRESERVED across intra-churn hash changes — a Zelda pan churns ~64
    straight frames of partially-drawn nametables, and the classifier
    needs the odometer integrated over the WHOLE churn window, not from
    the final hash's first appearance (which lands near pan end, where
    almost nothing is left to integrate).

    Returns `(pend', fired)`; `fired` is None until `h` has repeated
    `settle` consecutive samples, then `(h, (dx, dy), d_scene, steps)`
    measured from onset. A sample matching `settled_h` cancels the
    churn (false alarm) — mid-pan settles cannot fire because churning
    frames never repeat `settle` times (probe-receipted, §2). Blank
    frames never reach here: the caller gates on odo_debug rendered
    lines and resets pend to None instead (`min_lines`, §2)."""
    if settled_h is not None and h == settled_h:
        return None, None
    if pend is None:
        pend = (h, 1, (int(odo_xy[0]), int(odo_xy[1])), int(scene),
                int(step))
    elif pend[0] != h:
        pend = (h, 1, pend[2], pend[3], pend[4])
    else:
        pend = (h, pend[1] + 1, pend[2], pend[3], pend[4])
    if pend[1] >= int(settle):
        d_odo = (int(odo_xy[0]) - pend[2][0], int(odo_xy[1]) - pend[2][1])
        return None, (h, d_odo, int(scene) - pend[3], int(step) - pend[4])
    return pend, None


def classify_transition(d_odo, d_scene, pan_odo=(128, 384),
                        warp_scene_min=2):
    """Transition-kind classifier from integrated Δodo + Δscene over the
    churn window (pure; constants pre-registered from the 2026-08-24
    probe receipts, see §2).

      pan:  |Δodo| in [pan_min, pan_max] on EXACTLY one axis ->
            ("pan", dir) with dir E/W (+x/-x) or S/N (+y/-y; NES y
            grows downward). Both games measured ~256 px. Δscene is
            NOT gated here (2026-08-25 hardening finding): the core's
            scene-cut heuristic fires on ordinary camera clamp/seam
            noise near screen edges — exactly where real pans happen
            — so RG-0's own fixture proves a genuine door pan can
            co-occur with a spurious scene bump in the same churn
            window (test_rg0_4's +254px/Δscene+1 real door still reads
            as a clean pan at Δscene+2). A pan-sized odometer delta on
            exactly one axis is definitionally incompatible with the
            warp branch below (which requires BOTH axes near-zero), so
            dropping the old `Δscene <= 1` gate here cannot smuggle a
            real warp through as a pan.
      warp: Δscene >= warp_scene_min AND |Δodo| < 32 on both axes ->
            ("warp", None). Measured Zelda death: odometer modal
            16->272->16 (flat at settle), scene +2. Warp settles ADOPT
            the new room identity but mint NO adjacency edge and are
            never routable — the game-agnostic death-edge guard
            (Metroid has no probed death observable and must not need
            one).
      fade: otherwise -> ("fade", None). Hash flip, odo ~flat, scene
            0..1 — Zelda caves/dungeons, Rygar doors: the class the
            scene core is blind to by design."""
    dx, dy = int(d_odo[0]), int(d_odo[1])
    ds = int(d_scene)
    lo, hi = int(pan_odo[0]), int(pan_odo[1])
    in_x = lo <= abs(dx) <= hi
    in_y = lo <= abs(dy) <= hi
    if in_x != in_y:
        if in_x:
            return ("pan", "E" if dx > 0 else "W")
        return ("pan", "S" if dy > 0 else "N")
    if ds >= int(warp_scene_min) and abs(dx) < 32 and abs(dy) < 32:
        return ("warp", None)
    return ("fade", None)


def room_fp_config_sha(cfg: dict) -> str:
    """Stable 8-hex digest of the identity-bearing room_fp knobs — the
    lineage axis (`key_config["room_fp"]`) and room_index.json's
    `config_sha`. Mask ranges are canonicalised (sorted) so a cosmetic
    re-listing is not a new lineage; `sample_every` is EXCLUDED — a
    perf fallback changes when the detector looks, not what a settled
    ordinal means (the RG-1d re-measure path must not fork the
    schema)."""
    canon = {
        "mask": sorted([int(lo), int(hi)] for lo, hi in cfg["mask"]),
        "settle": int(cfg["settle"]),
        "min_lines": int(cfg["min_lines"]),
        "pan_odo": [int(cfg["pan_odo"][0]), int(cfg["pan_odo"][1])],
        "warp_scene_min": int(cfg["warp_scene_min"]),
        "palette_cokey": bool(cfg["palette_cokey"]),
        "max_rooms": int(cfg["max_rooms"]),
    }
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()[:8]


def _deep_tuple(v):
    """JSON round-trip repair for archive cell keys: lists back to the
    nested tuples observe() built (an exemplar cell key must compare
    equal to the live archive's keys or the edge-replay audit can never
    find its cell)."""
    if isinstance(v, list):
        return tuple(_deep_tuple(x) for x in v)
    return v


# Exemplar-action ring bounds (RG-1e, docs/receipts/room_graph/
# RG1_zelda_2026-08-25.md): a flat 32-step ring is fine for a pan/warp
# (measured well under it) but truncates a Zelda `fade` — a scripted,
# largely input-independent wipe whose real duration can run past 32
# steps — cutting off the actions between onset and the truncated
# window's start; replaying "exemplar_cell + exemplar_actions" then
# never reaches the settle it is supposed to reproduce. FLOOR keeps
# every already-working short traversal at exactly its old length;
# CEILING bounds the worst case (a pathological/never-settling churn)
# so per-edge storage and the live per-worker ring buffer both stay
# bounded instead of growing with churn length.
EXEMPLAR_RING_FLOOR = 32
EXEMPLAR_RING_CEILING = 128


class RoomIndex:
    """Global room-identity intern table + kind-tagged directed
    adjacency (ROOMGRAPH_ENGINE_2026-08-24 §2).

    APPEND-ONLY/MONOTONE by construction — ordinals are discovery
    order, edges only accumulate — so the structure is restore-order
    independent (the §2 lockstep invariant: global state never has to
    be rewound when a worker restores an older cell). Every mutator
    takes `self.lock` (the R4 _door_lock shape); readers of a single
    ordinal take it too, so the async flush can serialize a coherent
    snapshot.

    At `cap` the table HOLDS: a new hash returns None (caller keeps its
    last ordinal), `cap_hits` counts it, nothing crashes — the
    false-split failure mode (§7.2) degrades to telemetry, never to a
    138 GB archive."""

    VERSION = 1

    def __init__(self, cap: int = 1024, config_sha: str = "") -> None:
        self.hashes: dict = {}       # blake2b64 -> ordinal
        self.ordinals: list = []     # ordinal -> blake2b64
        self.meta: dict = {}         # ordinal -> {visits, bbox, aliased}
        self.adj: dict = {}          # src -> {dst -> EdgeStat}; DIRECTED
        self.warps: list = []        # recent warp settles (telemetry)
        self.warp_count = 0          # ALL warp settles, ever
        self.cap = int(cap)
        self.cap_hits = 0
        self.config_sha = str(config_sha)
        self.lock = threading.Lock()

    def intern(self, h: int, odo_xy=None):
        """hash -> ordinal, minting the next discovery-order ordinal for
        a new hash; updates visits and the odometer bbox (the
        false-merge audit's evidence). Returns None at cap (hold-last).
        """
        with self.lock:
            o = self.hashes.get(h)
            if o is None:
                if len(self.ordinals) >= self.cap:
                    self.cap_hits += 1
                    return None
                o = len(self.ordinals)
                self.hashes[h] = o
                self.ordinals.append(h)
                self.meta[o] = {"visits": 0, "bbox": None, "aliased": False}
            m = self.meta[o]
            m["visits"] += 1
            if odo_xy is not None:
                x, y = int(odo_xy[0]), int(odo_xy[1])
                b = m["bbox"]
                m["bbox"] = ([x, y, x, y] if b is None else
                             [min(b[0], x), min(b[1], y),
                              max(b[2], x), max(b[3], y)])
            return o

    def lookup(self, h: int):
        """FROZEN read (replay paths): ordinal or None, never interns —
        a replay must read the graph the live search built, not grow
        it."""
        with self.lock:
            return self.hashes.get(h)

    def hash_of(self, ordinal: int):
        """FROZEN reverse read (restore seeding): the hash an ordinal
        names, or None when no such room has been interned — which is
        how a stale psig tail (ROOM_UNKNOWN bytes, a truncated resume)
        is refused at _assign instead of aliasing a future room."""
        with self.lock:
            o = int(ordinal)
            return self.ordinals[o] if 0 <= o < len(self.ordinals) else None

    def record_edge(self, src: int, dst: int, kind: str, direction,
                    frames: int, exemplar_cell=None,
                    exemplar_actions=(), cap_sig: int = 0,
                    exemplar_state: "bytes | None" = None) -> None:
        """Commit one traversed adjacency edge (kind pan|fade). The
        FIRST exemplar is kept (archive cell key at onset + a trailing
        action ring, EXEMPLAR_RING_FLOOR..EXEMPLAR_RING_CEILING actions
        long and grounded in this settle's own measured `frames` — see
        the module-level EXEMPLAR_RING_FLOOR comment, RG-1e) so the
        sticky-replay edge-validity audit has a stable trajectory; later
        traversals only accumulate count/frames_mean. Warp-classified
        settles must never reach here — refused loudly, because a warp
        minted as navigable is the death-edge failure the classifier
        exists to close (§7.5).

        SELF-HEAL (2026-08-25 hardening finding): a fade recorded with
        dir=None never used to heal even when a LATER traversal of the
        same edge classified cleanly as pan — count/frames_mean were
        the only fields a repeat traversal touched, so one unlucky
        scene-noise co-occurrence permanently blinded the router to an
        already-mapped exit. A fade upgrades to pan (adopting the new
        direction) on any later pan traversal; a pan is never
        downgraded by a later fade — noise on one visit must not erase
        a clean read from another.

        ITEM-SIG GRAFT (ITEM_SEMANTICS_ENGINE_2026-08-25 §2A/§3 row 1):
        `cap_hist` buckets this traversal's `cap_sig` (the profile's
        `state_sig` bit-vector at the moment this edge staged, 0 when
        item-sig tracking is off or unset). Default `cap_sig=0` means
        every pre-existing caller — and every caller while
        `--item-sig-report` is off — writes bucket "0" exactly as
        before this graft existed: no new observable, no behavior
        change, additive only.

        FROZEN EXEMPLAR (RG1_zelda_2026-08-25 §RG-1a/RG-1e finding):
        `exemplar_cell` is a Go-Explore archive KEY, and `GoExploreArchive`
        is domination-based — a later, better-scoring visit to that same
        cell overwrites its `Cell.state` in place, so reading the archive
        back through `exemplar_cell` after the fact is not guaranteed to
        reproduce the state that was live when this edge was recorded
        (measured 86.6% stability, root-caused to exactly this). Fix:
        `exemplar_state`, when the caller has one, is a bytes COPY of the
        `pool.save_worker_state()` blob taken at record time — bytes are
        immutable, so once stored here no later domination of the source
        archive cell can change it. `exemplar_cell` is kept unchanged
        alongside it (still useful as a human-readable locator / for
        pre-existing consumers), it just stops being the ONLY way to get
        the state back. A caller that passes nothing (or an archive-less
        stand-in) gets `exemplar_state=None`, and a reader must then fall
        back to the old mutable-key lookup — this is additive, not a
        replacement, so archives banked before this graft still load.
        """
        if kind not in ("pan", "fade"):
            raise ValueError(
                f"room edge kind must be pan|fade, got {kind!r} — warp "
                f"settles are telemetry (record_warp), never edges")
        with self.lock:
            e = self.adj.setdefault(int(src), {}).get(int(dst))
            if e is None:
                e = self.adj[int(src)][int(dst)] = {
                    "kind": kind, "dir": direction, "count": 0,
                    "frames_mean": 0.0, "exemplar_cell": None,
                    "exemplar_actions": [], "exemplar_state": None,
                    "validated": False,
                    "validate_attempts": 0, "cap_hist": {}}
            elif e["kind"] == "fade" and kind == "pan" and direction is not None:
                e["kind"] = "pan"
                e["dir"] = direction
            e["count"] += 1
            e["frames_mean"] += (float(frames) - e["frames_mean"]) / e["count"]
            k = str(int(cap_sig))
            e.setdefault("cap_hist", {})
            e["cap_hist"][k] = e["cap_hist"].get(k, 0) + 1
            if e["exemplar_cell"] is None and exemplar_cell is not None:
                e["exemplar_cell"] = exemplar_cell
                # Ring length grounded in the MEASURED transition
                # duration (`frames`, already steps*frame_skip from the
                # settle that fired this edge) rather than a flat
                # constant (RG-1e). `frames` is always >= the true
                # step count the live ring accumulated, so clamping to
                # it never under-covers a real transition; the floor
                # keeps every pan/warp-sized (short) traversal at
                # exactly the pre-fix 32, and the ceiling bounds a
                # pathological one.
                ring_len = max(EXEMPLAR_RING_FLOOR,
                               min(int(frames), EXEMPLAR_RING_CEILING))
                e["exemplar_actions"] = [int(a) for a
                                         in exemplar_actions][-ring_len:]
                e["exemplar_state"] = (bytes(exemplar_state)
                                       if exemplar_state is not None
                                       else None)

    def record_warp(self, src, dst: int, d_scene: int, d_odo) -> None:
        """A warp-classified settle: telemetry ONLY — no adjacency, ever
        (bounded ring of recent events + a total count)."""
        with self.lock:
            self.warp_count += 1
            if len(self.warps) < 256:
                self.warps.append({
                    "src": (None if src is None else int(src)),
                    "dst": int(dst), "d_scene": int(d_scene),
                    "d_odo": [int(d_odo[0]), int(d_odo[1])]})

    def n_rooms(self) -> int:
        with self.lock:
            return len(self.ordinals)

    @staticmethod
    def _edge_to_json(e: dict) -> dict:
        """Shallow-copy one edge record, base64-encoding `exemplar_state`
        (raw state bytes are not JSON-serializable). Never mutates `e` —
        the live adjacency dict a search is still writing to must not be
        touched by a concurrent stats-flush."""
        d = dict(e)
        st = d.get("exemplar_state")
        d["exemplar_state"] = (base64.b64encode(st).decode("ascii")
                               if st is not None else None)
        return d

    def to_json(self) -> dict:
        with self.lock:
            return {
                "version": self.VERSION,
                "config_sha": self.config_sha,
                "cap": self.cap,
                "cap_hits": self.cap_hits,
                "hashes": [f"{h:016x}" for h in self.ordinals],
                "meta": {str(o): m for o, m in self.meta.items()},
                "adj": {str(s): {str(d): self._edge_to_json(e)
                                for d, e in dsts.items()}
                        for s, dsts in self.adj.items()},
                "warp_count": self.warp_count,
                "warps": list(self.warps),
            }

    def save(self, path) -> None:
        """Atomic tmp+rename on the archive.stats.json cadence + exit.
        An OSError is loud, never fatal — a full disk must not kill a
        live search over its telemetry sidecar."""
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.to_json()) + "\n")
            os.replace(tmp, p)
        except OSError as e:
            print(f"[go_explore_solve] could not save room index {p}: {e}",
                  flush=True)

    @classmethod
    def load(cls, path) -> "RoomIndex":
        """Rebuild from room_index.json. Raises ValueError on a schema
        version this build does not speak (the caller turns that into a
        resume refusal, never a silent reinterpretation)."""
        data = json.loads(Path(path).read_text())
        if int(data.get("version", -1)) != cls.VERSION:
            raise ValueError(
                f"room_index version {data.get('version')!r} != "
                f"{cls.VERSION}")
        idx = cls(cap=int(data.get("cap", 1024)),
                  config_sha=str(data.get("config_sha", "")))
        for o, hx in enumerate(data.get("hashes") or []):
            h = int(hx, 16)
            idx.hashes[h] = o
            idx.ordinals.append(h)
        for o_s, m in (data.get("meta") or {}).items():
            idx.meta[int(o_s)] = {
                "visits": int(m.get("visits", 0)),
                "bbox": (list(m["bbox"]) if m.get("bbox") else None),
                "aliased": bool(m.get("aliased", False))}
        for s_s, dsts in (data.get("adj") or {}).items():
            row = idx.adj[int(s_s)] = {}
            for d_s, e in dsts.items():
                # exemplar_state is base64 text on disk, absent entirely
                # on an archive banked before the RG-1a/RG-1e frozen-copy
                # fix. Absent/null => None, which is NOT a crash: it is
                # the documented fallback signal telling a reader (e.g.
                # scripts/rg1e_edge_validity.py) to go back to the OLD
                # mutable-key behavior (look `exemplar_cell` up in the
                # live archive) for this one edge, same as it always did.
                st_b64 = e.get("exemplar_state")
                row[int(d_s)] = {
                    "kind": str(e["kind"]), "dir": e.get("dir"),
                    "count": int(e.get("count", 0)),
                    "frames_mean": float(e.get("frames_mean", 0.0)),
                    "exemplar_cell": _deep_tuple(e.get("exemplar_cell")),
                    "exemplar_actions": [int(a) for a in
                                         e.get("exemplar_actions") or []],
                    "exemplar_state": (base64.b64decode(st_b64)
                                      if st_b64 else None),
                    "validated": bool(e.get("validated", False)),
                    "validate_attempts": int(e.get("validate_attempts",
                                                   0)),
                    "cap_hist": {str(k): int(v) for k, v in
                                (e.get("cap_hist") or {}).items()}}
        idx.warp_count = int(data.get("warp_count", 0))
        idx.warps = list(data.get("warps") or [])
        idx.cap_hits = int(data.get("cap_hits", 0))
        return idx


GX_BUCKET = 16   # overridable via --gx-bucket (micro-search: 8)
Y_BAND = 32      # overridable via --y-band (micro-search: 16)


def cell_fn(ram) -> tuple:
    # (area, step-phase, y-band, gx bucket). Phase = (frame>>2)&7 is
    # step-granular at frame_skip 4 (all 8 classes reachable via step-count
    # variance). gx bucket is LAST so the archive's horizontal_neighbors
    # frontier bonus applies unmodified.
    # vx sign disambiguates travel direction: doubling back through
    # previously-visited coordinates is a DISTINCT cell, so the archive
    # explores backtracking maneuvers instead of pruning them as loops
    # (heuristic-inversion recipe, maze consultation 2026-07-24).
    # $0057 is a signed velocity byte; take the two's-complement in pure
    # Python so an out-of-int8-range value (e.g. 252 = -4) casts cleanly
    # (np.int8(252) now raises a NumPy out-of-bound DeprecationWarning).
    vx = int(ram[0x57])
    if vx >= 128:
        vx -= 256
    vsign = 0 if vx == 0 else (1 if vx > 0 else 2)
    return (int(ram[R_AREA]), (int(ram[R_PHASE]) >> 2) & 7, vsign,
            int(ram[R_YPOS]) // Y_BAND, _gx(ram) // GX_BUCKET)


# ---------------------------------------------------------------------
# TORN MULTI-BYTE PROGRESS READS (the gx-767 phantom).
#
# A 16-bit progress observable is two RAM bytes, and the game updates
# them with two instructions. Sample the pair on the frame BETWEEN them
# and you read a position the game was never in. On Castlevania block 3
# ($0040 lo / $0041 hi) the lo byte wraps 0x00 -> 0xFF one frame before
# the hi byte borrows 2 -> 1, so a player walking LEFT past x=512 reads
# 0x02FF = 767 for exactly one frame.
#
# That single frame is not a curiosity: `score = sect*10000 + gx + ...`
# means the phantom DOMINATES its cell outright, so every archive built
# on that wall put a torn read at the top of its frontier. All 35
# band-95 cells across five Castlevania archives — 10.7 h and ~77M
# steps of "pinned at 767" — were the same one-frame artifact, and the
# true frontier was gx 751 (campaign §12; the restores read 507-511).
#
# The filter below is GENERIC (it knows only that the observable is
# built from two bytes), OPT-IN per profile (`progress: {smooth: ...}`)
# and REJECT-ONLY: a sample it dislikes is dropped exactly the way
# observe() already drops a `progress_cap` garbage read, so nothing
# downstream — score, cell key, max_gx, the pin clock — is ever handed
# a value the filter invented.
#
# It is deliberately ONLINE (one or two prior samples, no lookahead):
# observe() has to classify the frame it is holding. The price is one
# dropped observation at a genuine discontinuity (a warp, a room load,
# a re-root); the same precision-over-recall trade the room_veto doc
# above makes, and a dropped observation costs a re-record, never a
# missed clear (is_dead/is_clear/is_finale all resolve BEFORE this).
# ---------------------------------------------------------------------

#: `progress: {smooth: ...}` modes. `off` = the shipped behaviour.
PROGRESS_SMOOTH_MODES = ("off", "borrow", "median3", "hampel")

#: Default `progress: {jump: N}`: how far the observable may move in one
#: observation before a filter is allowed to call the sample torn. 128 is
#: half a page — comfortably above any per-frame velocity a position byte
#: pair can express, comfortably below the ~255 a torn pair fabricates.
PROGRESS_JUMP = 128

#: `hampel`'s rolling window: the median/MAD are taken over this many raw
#: samples (the current one plus the trailing HAMPEL_WINDOW - 1), and the
#: persistence check that lets a sustained jump through reads one sample
#: further back still — see `progress_glitch` and `HAMPEL_KEEP` below.
HAMPEL_WINDOW = 5

#: How many raw samples `observe()` must keep in `hist` for `hampel` to see
#: both its window and the one-sample-further-back persistence check. One
#: more than the window: the window's oldest slot is exactly the sample the
#: persistence check needs as ITS window's newest slot.
HAMPEL_KEEP = HAMPEL_WINDOW + 1


def _median_mad(values) -> tuple:
    """(median, MAD) of `values`. Pure; ties broken the same way for both
    odd and even counts (average of the two middle order statistics)."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    devs = sorted(abs(v - median) for v in s)
    m = len(devs) // 2
    mad = devs[m] if len(devs) % 2 else (devs[m - 1] + devs[m]) / 2
    return median, mad


def progress_glitch(mode: str, hist, sample, jump: int = PROGRESS_JUMP) -> bool:
    """True when `sample` is a torn read rather than a state the game
    was in. Pure.

    `sample` and every entry of `hist` are `(value, hi, lo)` — the
    composed observable plus the two raw bytes it came from, `hi`/`lo`
    None when the profile's progress is not a two-byte read (a single
    byte, or a decoded HUD field), in which case `borrow` is inert by
    construction. `hist` holds the RAW previous samples, most recent
    last, INCLUDING ones this function rejected: the rule is about the
    byte stream, not about the accepted series.

    borrow  — the low byte wrapped across the 0xFF/0x00 boundary while
              the high byte did not move. On a real 16-bit counter a lo
              wrap is always accompanied by a carry/borrow into hi, so
              "wrapped, hi unchanged" is a torn pair and nothing else.
    median3 — the sample is not the median of the last three raw
              readings and misses it by more than `jump`: a one-sample
              spike. Catches a tear whose bytes are not adjacent (and
              any other single-frame impulse) at the cost of dropping
              the first sample of a genuine jump too.
    hampel  — a robust generalisation of median3: the sample is compared
              against the median of the trailing HAMPEL_WINDOW raw values
              (current one included), and flagged only when it clears BOTH
              3 * MAD (the Hampel identifier's usual threshold) and `jump`.
              median3 always drops the first sample of a genuine warp; this
              mode does too, UNLESS the immediately preceding raw sample
              (which the window rule would still be outvoting 3-2 by the
              old baseline) was itself a comparable outlier against the
              window one step further back AND `sample` sits close to it
              rather than reverting — two consecutive readings that agree
              on a new level are a level, not a spike, and the second one
              is admitted retroactively (<=1 sample of delay: the first
              reading of a warp still costs an observation, exactly like
              median3, but the second one is never dropped).
    """
    if mode in (None, "", "off") or not hist:
        return False
    val, hi, lo = sample
    pval, phi, plo = hist[-1]
    if mode == "borrow":
        if hi is None or phi is None:
            return False
        # With `hi` unchanged the composed delta IS the low-byte delta,
        # so one test covers both: the observable moved most of a page
        # in one observation without the page byte moving with it.
        # Clamped into a byte — this threshold is about the 0xFF/0x00
        # boundary whatever scale `jump` was set for, and 255 would make
        # the rule unfireable rather than merely permissive.
        thresh = max(1, min(int(jump), 254))
        return bool(int(hi) == int(phi) and abs(int(lo) - int(plo)) > thresh)
    if mode == "median3":
        if len(hist) < 2:
            return False
        median = sorted((int(hist[-2][0]), int(pval), int(val)))[1]
        return abs(int(val) - median) > jump
    if mode == "hampel":
        if len(hist) < 2:
            return False
        window = [int(h[0]) for h in hist[-(HAMPEL_WINDOW - 1):]] + [int(val)]
        median, mad = _median_mad(window)
        dev = abs(int(val) - median)
        if not (dev > 3 * mad and dev > jump):
            return False
        prev_val = int(pval)
        if abs(int(val) - prev_val) > jump:
            return True   # not sitting near the previous sample: no
                           # persistence signal, this is a lone spike
        prior = [int(h[0]) for h in hist[-HAMPEL_WINDOW:-1]]
        if len(prior) < 2:
            return True
        pmedian, pmad = _median_mad(prior)
        pdev = abs(prev_val - pmedian)
        if pdev > 3 * pmad and pdev > jump:
            return False  # two consecutive outliers agreeing on a new
                           # level: a sustained jump, admitted
        return True
    return False


#: How many refuted borrow rejections it takes to disarm the rule. Three,
#: and only while they outnumber the confirmed tears: one coincidence
#: (an unrelated hi move on the frame after a real wrap) must not disarm
#: a filter that is working, and a pair that is not a pair produces a
#: refutation at EVERY crossing, so three arrive quickly.
BORROW_REFUTE_LIMIT = 3


def borrow_followup(rejected, nxt) -> str:
    """What the frame AFTER a borrow rejection says the rejected frame
    was. Pure. Returns "tear", "not_a_pair" or "unknown".

    THE RULE ASSUMES hi IS lo's PAGE BYTE AND NOTHING CHECKED IT. The
    only guard was that `hi` exists, and 6 of the 11 two-byte progress
    profiles in configs/ are (screen, x) composites — kid_icarus
    (0x750/0x4D1), megaman (0x460/0x440), kirby, excitebike,
    ghosts_n_goblins, double_dragon — whose bytes are not adjacent and
    whose low byte wraps without any carry into the high one. On those,
    "lo wrapped, hi flat" is ORDINARY MOTION and the rule deletes a real
    observation at every crossing.

    It cannot be settled on the frame the filter must decide (that is
    why the filter is online), but it is settled one frame later: a torn
    read is a sample taken BETWEEN two instructions, so the carry lands
    on the next frame and hi moves. A pair that is not a 16-bit pair
    leaves hi exactly where it was. The caller counts both and disarms
    the rule when the refutations win — measured, not declared.
    """
    if not rejected or not nxt:
        return "unknown"
    _rv, rhi, _rlo = rejected
    _nv, nhi, _nlo = nxt
    if rhi is None or nhi is None:
        return "unknown"
    return "tear" if int(nhi) != int(rhi) else "not_a_pair"


def push_progress_sample(hist, sample, keep: int = 2) -> None:
    """Append a raw reading to a bounded rolling history, in place."""
    hist.append(sample)
    if len(hist) > keep:
        del hist[:-keep]


#: A resumed frontier group is a torn-read ISLAND when at least this many
#: gx buckets below it are empty. The §12 signature exactly: bucket 95
#: occupied, bucket 94 empty in all five archives, the body at 93.
PHANTOM_GAP = 1
#: ...and when it holds no more than this share of the body below it. 14
#: cells over ortho_ctrl's 2,805 is 0.005; a real frontier advance
#: arrives with a population, a one-frame artifact cannot.
PHANTOM_MAX_FRAC = 0.02


def phantom_top_buckets(keys, gap: int = PHANTOM_GAP,
                        max_frac: float = PHANTOM_MAX_FRAC,
                        rounds: int = 3) -> set:
    """{(area, gx bucket)} at the top of each area that are a banked
    torn read rather than a reached position. Pure.

    THE FILTER ONLY SCREENS NEW SAMPLES; THE PHANTOM IS ALREADY BANKED.
    Every Castlevania hall archive carries the gx-767 artifact as real
    cells, and both frontier readers — seed()'s max_gx_in_area walk and
    _refresh_sel_cache's `_sel_topgx = max(key[-1])` — take a maximum, so
    14 cells out of 149,153 defined the frontier of every resumed run.
    §12 re-froze the root pool on bucket 93 by hand and no code did it.

    The signature is structural and needs no game knowledge: an occupied
    bucket separated from the body of the frontier by an EMPTY gap and
    holding a negligible share of it. A search advances by occupying the
    buckets it passes through; a non-atomic 16-bit read jumps the gap and
    lands nowhere near a population. Repeated up to `rounds` times per
    area, because one tear can strand another just below it.
    """
    per_area: dict = {}
    for k in keys:
        per_area.setdefault(int(k[-5]), {})
        b = int(k[-1])
        per_area[int(k[-5])][b] = per_area[int(k[-5])].get(b, 0) + 1
    out: set = set()
    for area, hist in per_area.items():
        live = dict(hist)
        for _ in range(max(1, int(rounds))):
            if len(live) < 2:
                break
            top = max(live)
            group = []
            b = top
            while b in live:                     # the contiguous top group
                group.append(b)
                b -= 1
            below = [x for x in live if x < min(group)]
            if not below:
                break
            if min(group) - max(below) - 1 < int(gap):
                break
            body = sum(live[x] for x in below)
            island = sum(live[x] for x in group)
            if body <= 0 or island > float(max_frac) * body:
                break
            out.update((area, x) for x in group)
            for x in group:
                live.pop(x, None)
    return out


# ---------------------------------------------------------------------
# LOCAL COVERAGE (c_local) — the derivative the receipts kept asking for.
#
# `c_local` = the number of distinct (area, y_band, gx_bucket) triples
# the archive has reached. It is the archive's SPATIAL footprint, as
# opposed to `cells`, which counts key-space entries and therefore grows
# with every extra key axis.
#
# It is emitted per progress line because the signal is a DERIVATIVE and
# a banked archive holds exactly one reading of it, at the flush: on the
# Castlevania hall the footprint went 932 -> 1,102 (+18%) while cells
# went 28,929 -> 560,410 (19x), and no cross-sectional statistic
# recovers that (docs/receipts/dispatch/size_decoupled_statistic_2026-
# 08-11.md §11: 22 candidates, 0 separate; "every candidate is a
# surrogate for a derivative nobody recorded"). The elasticity
# d log(c_local) / d log(cells) is scale-free and is the form that
# receipt names as the one worth banking.
#
# NOTHING HERE CLASSIFIES A WALL — but do not read that as "inert".
# `c_local` is the first entry on wall_taxonomy.MISSING_TELEMETRY (the
# module's own shopping list: "count of DISTINCT spatial buckets in the
# archive ... per progress line"), and emitting it MOVES VERDICTS. The
# same 90-line hall log reads INDETERMINATE / missing=('c_local', ...)
# without the field and COVERAGE_LIMITED ("C_local saturation 0.049 <
# 0.85: the map footprint is still expanding") with it — measured on
# runs/cv_hall_ortho_ctrl/progress.jsonl, both ways, 2026-08-11. That is
# evidence arriving, not a regression: the discriminator applies the
# same tests to the C_local series and to the archive snapshot
# symmetrically, so a run does not flip verdicts merely because the
# emitter learned a new field — it stops abstaining for want of one.
# What stays struck is PROMOTION: the GATED class this series was
# originally meant to feed is REFUTED, and c_local may only refute a
# plateau (COVERAGE_LIMITED), name a stagnant map (BARREN) or name a
# blind key (KEY_BLIND). Nothing here may certify a wall as opened.
# ---------------------------------------------------------------------

#: gx buckets counted as "the frontier band" for the band-local series.
LOCAL_BAND_BUCKETS = 24


def local_coverage(keys) -> set:
    """Distinct (area, y_band, gx_bucket) triples over archive keys.

    Positional, like every other frontier reader in this file: a cell
    key is `(sect, tb, kk, psig, loops, route_sig) + cell_fn(ram)` and
    cell_fn's tail is `(area, ..., y_band, gx_bucket)`."""
    return {(k[-5], k[-2], k[-1]) for k in keys}


def local_coverage_band(triples, band: int = LOCAL_BAND_BUCKETS) -> int:
    """How many of those triples sit within `band` buckets of the
    deepest one reached. The frontier-local half of the series: total
    footprint keeps climbing while a search wanders, so a wall shows up
    as the BAND term flattening while the total does not."""
    triples = list(triples)
    if not triples:
        return 0
    floor = max(int(t[-1]) for t in triples) - int(band)
    return sum(1 for t in triples if int(t[-1]) >= floor)


def coverage_elasticity(prev_local, prev_cells, local, cells):
    """d log(c_local) / d log(cells) across one progress window.

    None when it is undefined (first window, no new cells, or either
    count still at zero) rather than 0.0 — "not measurable yet" and
    "measured flat" are the two readings this series exists to tell
    apart, and a silent zero is exactly the plateau it would fake."""
    try:
        a, b = int(prev_local), int(local)
        m, n = int(prev_cells), int(cells)
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0 or m <= 0 or n <= 0 or n == m:
        return None
    return (math.log(b) - math.log(a)) / (math.log(n) - math.log(m))


def is_forward_clear(start_wd: tuple, ram) -> bool:
    """True iff (world, level) advanced in the NATURAL forward sequence.

    Legit 1-2 -> 1-3: world unchanged, level +1. Legit castle world advance
    (e.g. 1-4 -> 2-1): world +1 (level resets to 0). A WARP-ZONE pipe jumps
    the world by >1 or advances the world without having been on a castle —
    guarded out so warps never count as a clear (honesty requirement)."""
    sw, sl = start_wd
    w, l = _wd(ram)
    if w == sw and l == sl + 1:
        return True   # same-world level advance (the common case, incl. 1-2->1-3)
    if w == sw + 1 and l == 0 and sl >= 2:
        return True   # castle clear -> next world (only from an x-4, level idx>=2... )
    return False


def resolve_verify_bank(args) -> bool:
    """Whether a clear candidate must reproduce from its root, in a fresh
    pool, before it is written to solutions/ (--verify-bank /
    --no-verify-bank).

    DEFAULTS TRUE, including when the attribute is absent — in-process
    constructions that predate the flag (the live show's hand-built args
    namespace) get verification too. That direction is deliberate and the
    opposite of the hw-flag/ortho knobs' "absent means off": those change what
    the search DOES, this only decides whether a claimed win is checked before
    it becomes a receipt, and banking trust must not be something a stale call
    site opts out of by omission."""
    return bool(getattr(args, "verify_bank", True))


def resolve_counterfactual_gate(args) -> bool:
    """Whether a clear candidate must additionally survive the K-branch
    counterfactual perturbation probe before it is banked
    (--counterfactual-gate / --no-counterfactual-gate).

    DEFAULTS FALSE, including when the attribute is absent — the opposite of
    resolve_verify_bank, and deliberately so. Replay verification is
    milliseconds per action and asks "did this happen at all"; this gate is
    SECONDS (K short branches re-simulated from a pre-clear state, measured
    4-40 s per candidate) and asks the strictly harder question "would this
    still have happened if the player had played slightly differently". A
    check with that cost is an opt-in instrument, not something a call site
    inherits by omission."""
    return bool(getattr(args, "counterfactual_gate", False))


#: How much deeper Solver.counterfactual_probe re-snapshots when its CONTROL
#: branch could not see the clear (the DEEP RETRY). One retry at 4x the
#: requested pre-steps: far enough back to step over a commit point that
#: predates the first snapshot (the case the retry exists to rescue), small
#: enough that the probe stays a seconds-scale banking-path check instead of
#: K full replays of a 1,700-action candidate.
CF_RETRY_FACTOR = 4


def resolve_apu_sampling(game, pool) -> bool:
    """Whether the per-step 5-bit APU channel-activity mask must be sampled
    and threaded into the clear hook's ctx.

    True only when BOTH hold: the adapter asked for it (`clear: {apu_weight:
    > 0}`, default absent) and the linked nes_core exports
    `Pool.apu_activity_all` (added 2026-08-09). A profile that opts in against
    an older wheel degrades to the RAM-only detector with one loud line rather
    than raising per step — and says so, because silently dropping a vote a
    profile asked for is how a receipt ends up describing a detector that
    never ran."""
    if not bool(getattr(game, "needs_apu", lambda: False)()):
        return False
    if not hasattr(pool, "apu_activity_all"):
        print("[go_explore_solve] profile asked for the APU channel-activity "
              "vote but this nes_core has no Pool.apu_activity_all — running "
              "the RAM-only detector instead", flush=True)
        return False
    return True


def warn_if_burst_starves_clear_hook(game, burst: int) -> int:
    """One loud line when a live BURST is shorter than the clear hook's own
    warm-up. Returns the budget it asked the adapter for (0 = stateless hook,
    nothing to warn about).

    The live hook's state is per burst, not per run: _assign() hands every
    worker a fresh ctx each time it re-roots, and GenericGame.is_clear builds
    the streaming detector lazily inside that ctx. A hook that needs more
    observations than `--burst` gives it can therefore NEVER fire during the
    search, however long the run lasts — the same structural no-op the
    counterfactual gate was measured committing, one level up and with no
    receipt to notice it in. Worth exactly one line at construction: it is
    the difference between "the audio vote disagreed" and "the audio vote was
    never able to exist".

    Diagnostic only — it changes no behavior and never raises, so unlike
    Solver.counterfactual_probe (whose verdict depends on the number) it
    tolerates an adapter that predates the hook."""
    budget = int(getattr(game, "clear_observation_budget", lambda: 0)())
    if budget > 0 and burst > 0 and burst < budget:
        print(f"[go_explore_solve] WARNING: this profile's clear hook needs "
              f"{budget} observations of warm-up but --burst is {burst}, and "
              f"the hook's state is rebuilt every burst — it can never fire "
              f"during the search. Raise --burst above {budget} or relax the "
              f"clear knobs (stride/persist_checks/apu min_baseline).",
              flush=True)
    return budget


def resolve_inversion_pin_secs(args) -> float:
    """Seconds the deep frontier must sit pinned before the heuristic-
    inversion arm engages (--inversion-pin-secs).

    Absent attribute — in-process constructions (the live show) that
    predate the flag — resolves to 180.0, the constant the receipted
    32-level SMB clear ran under, so those callers are unchanged. A
    NEGATIVE value disables the arm outright: the sampler then never
    flips to inverted weights, whatever the frontier does."""
    v = getattr(args, "inversion_pin_secs", None)
    return 180.0 if v is None else float(v)


def inversion_armed(pin_secs: float, pin_gx: int, gx: int, floor: float,
                    elapsed: float) -> bool:
    """Whether a worker sitting at `gx` should sample from the INVERTED
    weights this step.

    Four independent conditions, all measured off our own telemetry:
    the arm is enabled (`pin_secs` non-negative — a negative value is the
    --inversion-pin-secs disable sentinel, and it short-circuits FIRST
    because `elapsed >= -1` would otherwise always hold); the run has a
    real frontier to be pinned at (`pin_gx > 400`); the worker is inside
    the saturation window [floor, pin_gx+60] on a non-garbage gx; and the
    frontier has sat unmoved for at least `pin_secs`.

    A free function, not an inline expression, so the disable sentinel
    and the window boundaries are testable without a ROM and a Pool —
    the wiring is the whole deliverable of the flag."""
    return (pin_secs >= 0.0 and pin_gx > 400 and gx >= 0
            and floor <= gx <= pin_gx + 60 and elapsed >= pin_secs)


def finisher_extension_ok(status: str, c: dict) -> bool:
    """Burst-side eligibility for the finisher extension explore() grants at
    the deepest frontier (the positional band check stays in the loop). One
    grant per burst, and NEVER during a dead-blip: observe() returns "live"
    for the first two consecutive dead reads, so `status == "live"` alone
    would extend a lineage already dying at burst end — and its confirmed
    death 1-2 steps into the extension would DOA-retire the deepest frontier
    cell, the extension's own root. A free function so the blip guard is
    testable without a ROM and a Pool."""
    return (status == "live" and c["left"] <= 0
            and not c.get("extended") and not c.get("_dead_mm"))


def inverted_weights(action_space) -> list:
    """Exploration bias for the saturation window: reward the maneuvers the
    forward heuristic structurally prunes (leftward, downward)."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "left" in b:
            w += 3.0
        if "down" in b:
            w += 3.0
        if "A" in b:
            w += 1.0
        if "right" in b:
            w -= 0.5
        ws.append(max(w, 0.2))
    return ws


def action_weights(action_space) -> list:
    """Right-biased sampling weights over the profile's action space."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "right" in b:
            w += 2.0
        if "A" in b:
            w += 2.0  # jumps carry gaps/pipes
        if "B" in b:
            w += 1.0
        ws.append(w)
    return ws


class SmbGame:
    @staticmethod
    def score_bonus(ram) -> int:
        return 0

    """SMB-engine adapter — wraps the module-level helpers verbatim, so
    solver behavior on SMB is byte-identical to the pre-adapter code
    (regression: seeded 1-1 run reproduces the same solution sha).

    Works for any game on the SMB1 ENGINE, not just SMB1 itself — the
    RAM map ($006D/$0086 x, $0760 area, $075F/$075C world/level, $075A
    lives) is engine-level. The Lost Levels (Super Mario Bros. 2 Japan /
    FDS) is the same engine with harder layouts, so a profile pointing
    `rom:` at it (and carrying NO solve: section) routes here and gets
    the full byte-exact SMB solver for free. A profile's optional `rom:`
    overrides the default SMB1 World ROM."""
    progress_cap = 7000   # transition-frame garbage guard (8-1 lesson)

    def __init__(self, profile: dict = None) -> None:
        r = (profile or {}).get("rom")
        self.rom = str(REPO / r) if r else ROM

    def progress(self, ram) -> int:
        return _gx(ram)

    def level_key(self, ram) -> tuple:
        return _wd(ram)

    def lives(self, ram) -> int:
        return int(ram[R_LIVES])

    def area(self, ram) -> int:
        return int(ram[R_AREA])

    def y(self, ram) -> int:
        return int(ram[R_YPOS])

    def swim(self, ram) -> int:
        return int(ram[0x1D])

    def cell_fn(self, ram) -> tuple:
        return cell_fn(ram)

    def is_clear(self, start_key: tuple, ram, ctx: dict | None = None) -> bool:
        # ctx accepted for call-site uniformity with GenericGame; SMB uses the
        # warp-guarded level_key advance only (behavior byte-identical).
        return is_forward_clear(start_key, ram)

    def is_dead(self, ram, start_lives: int) -> bool:
        # $00B5 = vertical screen page: 1 on-screen, 0 above (vines — legit),
        # >= 2 fallen below the screen. A pit fall reads B5=2..4 for ~120
        # frames BEFORE lives decrements (measured live, Lost Levels 2-2), so
        # without this check every frame of the fall is archivable as a
        # live cell — at 2-2 the deepest-gx frontier was ~entirely such
        # corpses (observatory receipt runs/lost_levels/ll_2_2_sectcap_fix/
        # observatory_v2.json: top predicate = the death timer $0795), and
        # every burst selected from them was doomed on arrival. Verified
        # zero false positives across the full receipted 8-4 clear replay
        # (1,735 actions incl. hold-Down pipe entries + the water section):
        # legit play never reads B5 >= 2.
        return (int(ram[R_LIVES]) < start_lives
                or int(ram[R_PSTATE]) in DEATH_STATES
                or int(ram[0xB5]) >= 2)

    def is_finale(self, start_key: tuple, ram) -> bool:
        # GAME-COMPLETE (8-4): the ending never advances the world/level
        # bytes — victory is operating mode $0770 == 2 with inputs locked.
        return int(ram[0x770]) == 2 and tuple(start_key) == (7, 3)

    def room_id(self, ram) -> tuple:
        # rid = (area type, swim flag, Bowser-slot) — verified 2026-07-26:
        # $074E changes 0.67/1k steps, only at full-screen transitions.
        bw = 1 if 0x2D in bytes(ram[0x14:0x1C]) else 0
        return (int(ram[0x74E]), int(ram[0x1D]), bw)

    def clear_verify_margin(self) -> int:
        """No NOOP margin: SMB's clear hooks (a level_key advance, and the
        $0770==2 finale) are STATELESS — pure functions of the current frame
        — so a replay's verdict lands on exactly the frame the live one did.
        See GenericGame.clear_verify_margin for the windowed case and why it
        needs one. Every adapter must define this; Solver.replay_verify calls
        it unconditionally, and its broad except would otherwise turn a
        missing method into a blanket 'error' verdict that rejects every real
        clear the game ever finds."""
        return 0

    def needs_apu(self) -> bool:
        """SMB's clear hooks are RAM-only; the APU modality is never sampled
        for this adapter. Defined here for the same reason
        clear_verify_margin is: the callers ask every adapter."""
        return False

    def clear_observation_budget(self) -> int:
        """Zero: SMB's clear hooks are stateless, so a replay of ANY length
        can see the clear on the frame it happens. Defined here for the same
        reason clear_verify_margin and needs_apu are — the callers ask every
        adapter, unconditionally, and a missing method would turn into a
        blanket 'error' verdict inside their broad excepts."""
        return 0

    @staticmethod
    def label(key: tuple) -> str:
        return f"{key[0] + 1}-{key[1] + 1}"


class GenericGame:
    """Profile-driven adapter: every address comes from the game profile's
    `solve:` section, whose bytes must be observationally verified first
    (scripts/verify_ram_map.py; receipts under docs/receipts/ram_verify/).

    Required solve keys: rom, progress {lo[, hi]}, y, level_key (list of
    addrs; any lexicographic advance = clear), lives (decrement = death).
    Optional: area, progress_cap, player_state + death_states, finale
    {addr, value, level_key} for the game's ending; clear {mode: ...} = an
    OPTIONAL second WIN-CONDITION signal for games whose level_key never
    advances on a clear (see __init__ / is_clear; default OFF = level_key
    only). The discrete-transition gate (room_advance) is a Solver-level
    option (see derive_transition_macros)."""

    def __init__(self, profile: dict) -> None:
        s = profile["solve"]
        self.rom = str(REPO / s["rom"])
        p = s["progress"]
        # PPU SCROLL ODOMETER progress (optional): `progress: {source:
        # odometer, axis: x|y}`. The progress observable is the in-core
        # camera integral (nes_core odometer, certified by
        # scripts/odometer_cert.py) instead of a discovered RAM byte —
        # the hardware-surface signal for games whose RAM bytes failed
        # the progress-signal gate (Rygar's saturating $0015, Ninja
        # Gaiden's dud). The Solver appends the clamped 24-bit integral
        # to every worker's RAM snapshot at pseudo-addresses
        # 0x800..0x802 (see Solver._xram), so every existing consumer —
        # cells, glitch filter, macros — reads it through the ordinary
        # lo/hi path. verify_ram_map receipts do not apply: the
        # instrument is certified by odometer_cert + the gate, not by
        # RAM verification.
        self.odometer_axis = None
        self.odometer_sign = 1
        if str(p.get("source", "")).lower() == "odometer":
            axis_raw = str(p.get("axis", "x")).lower()
            # SIGNED axis (2026-08-25, the 1942 vertical-scroller find):
            # `_xram` clamps the raw camera integral to >= 0 on the
            # assumption that "forward" is the increasing direction —
            # true for every prior odometer profile (all horizontal
            # scrollers moving right). 1942's PPU scroll register counts
            # DOWN while the plane flies forward (measured directly:
            # every one of noop/right/left/up/down/A/B held from the
            # verified-live start state moves raw y strictly negative,
            # runs/onboard_wave3/diag_1942_root.log), so the unsigned
            # read clamped every real frame of flight to 0 and the
            # solver saw a permanently flat progress column (7 cells,
            # best_score 0, over a full 4-minute smoke —
            # runs/onboard_wave3/smoke_1942/archive.stats.json). `axis:
            # -y` (leading sign, default +) flips the raw reading before
            # the clamp so "forward" is monotone increasing again,
            # exactly like every other profile.
            sign = -1 if axis_raw.startswith("-") else 1
            axis = axis_raw[1:] if axis_raw and axis_raw[0] in "+-" else axis_raw
            if axis not in ("x", "y"):
                raise SystemExit(
                    f"[go_explore_solve] progress.axis must be x, y, -x or "
                    f"-y, got {axis_raw!r}")
            self.odometer_axis = axis
            self.odometer_sign = sign
            p = dict(p)
            p["lo"], p["hi"] = ODO_LO, ODO_LO + 1
        # FIGHT-GATE progress (optional): `progress: {source: fight_gate,
        # foe_hp: <addr>[, foe_hp_start: <val>][, round: <addr>]}` — a
        # camera-static combat game with no spatial frontier at all
        # (Punch-Out, FIGHTGATE_MECHANISM_2026-08-25.md §4). `foe_hp` and
        # `round` come from `scripts/discover_observables.py --fight-gate`
        # (`find_fight_health` / `find_round_gate`), never hand-picked from
        # a RAM map — the same purity discipline every other solve address
        # in this file already follows. The progress observable is a
        # cumulative-damage integral the Solver accumulates into the
        # pseudo-RAM extension at FIGHT_LO/FIGHT_HI (fight_gate_step, one
        # call per worker per step in Solver._xram) — exactly the odometer
        # branch above's relationship to ODO_LO, so every existing lo/hi
        # consumer (cells, the glitch filter, macros, progress_cap) needs
        # no changes. `round` is optional: with none (Punch-Out has none,
        # §4.2), a bout boundary is instead detected by the mass-RAM-reset
        # signature (`fight_gate_mass_reset`) the Solver already applies
        # for deaths/reloads elsewhere.
        self.foe_hp_addr = None
        self.foe_hp_start = 0
        self.fight_round_addr = None
        if str(p.get("source", "")).lower() == "fight_gate":
            if "foe_hp" not in p:
                raise SystemExit(
                    "[go_explore_solve] progress: {source: fight_gate} "
                    "needs foe_hp: <addr> (from find_fight_health).")
            self.foe_hp_addr = int(p["foe_hp"])
            self.foe_hp_start = int(p.get("foe_hp_start", 0))
            self.fight_round_addr = (int(p["round"]) if "round" in p
                                     else None)
            p = dict(p)
            p["lo"], p["hi"] = FIGHT_LO, FIGHT_HI
        # Decoded HUD-field progress (optional): score/money games (DuckTales)
        # have no monotone spatial frontier — the objective is a multi-tile
        # decimal HUD field. `progress: {tiles: [addrs MSB-first], blank: B,
        # scale: N}` decodes those tiles (tile value = literal digit, blank
        # tile skipped) into an integer // scale. The monotone objective then
        # IS the frontier: banking higher-score states drives the solver
        # toward treasure-rich paths through the level. `lo`/`hi` is the
        # plain 1-2 byte mode (spatial games).
        self._ptiles = [int(a) for a in p["tiles"]] if "tiles" in p else None
        self._pblank = int(p.get("blank", 0x24))
        self._pscale = int(p.get("scale", 1))
        self._plo = int(p["lo"]) if "lo" in p else None
        self._phi = int(p["hi"]) if "hi" in p else None
        # TORN-READ FILTER (optional, default off — see progress_glitch).
        # `progress: {smooth: borrow|median3[, jump: N]}`. Opt-in per
        # profile because it costs one dropped observation at every real
        # discontinuity, which only pays for itself on a game whose
        # progress pair is actually non-atomic.
        self.progress_smooth = str(p.get("smooth", "off") or "off").lower()
        if self.progress_smooth not in PROGRESS_SMOOTH_MODES:
            raise SystemExit(
                f"[go_explore_solve] unknown progress smooth mode "
                f"{self.progress_smooth!r}; choose from "
                f"{', '.join(PROGRESS_SMOOTH_MODES)}.")
        if self.progress_smooth == "borrow" and self._phi is None:
            raise SystemExit(
                "[go_explore_solve] progress.smooth: borrow needs a two-byte "
                "progress read (`progress: {lo: .., hi: ..}`) — there is no "
                "carry to be torn on a single byte. Use median3 instead.")
        self.progress_jump = int(p.get("jump", PROGRESS_JUMP))
        if self.progress_jump < 1:
            raise SystemExit(
                f"[go_explore_solve] progress.jump must be >= 1, got "
                f"{self.progress_jump}: a zero-width tolerance calls every "
                f"observation torn and the search records nothing.")
        # ATOMIC PEEK (optional, default off). `progress: {atomic: true}`
        # routes the two-byte sample through Pool.peek_u16_consistent
        # instead of composing it from a step snapshot: the scratch-
        # instance adjudicator resolves the same 0xFF/0x00 boundary
        # `smooth` merely screens for, so it supersedes `smooth` rather
        # than stacking with it (Solver.observe skips the smoothing
        # block whenever atomic is on). Needs a two-byte read for the
        # same reason `borrow` does — there is no pair to adjudicate on
        # a single byte or a decoded HUD field.
        self.progress_atomic = bool(p.get("atomic", False))
        if self.progress_atomic and self._phi is None:
            raise SystemExit(
                "[go_explore_solve] progress.atomic: true needs a two-byte "
                "progress read (`progress: {lo: .., hi: ..}`) — there is no "
                "pair for peek_u16_consistent to adjudicate on a single "
                "byte.")
        self._y = int(s["y"])
        self._lk = [int(a) for a in s["level_key"]]
        self._lives = int(s["lives"])
        self._area = int(s["area"]) if "area" in s else None
        self.progress_cap = int(s.get("progress_cap", 30000))
        self._pstate = int(s["player_state"]) if "player_state" in s else None
        self._death_states = tuple(s.get("death_states", ()))
        self._finale = s.get("finale")
        # Boss-fight progress (optional): a fixed-camera boss room pins the
        # gx frontier and position cells saturate in minutes (block-3 lesson:
        # 4.2M steps, 5,235 cells, zero gradient). `boss: {hp, start}` keys
        # cells on the boss-HP byte and scores damage dealt, giving the
        # archive a fight dimension. The byte must be discovered by
        # differential analysis of our own rollouts and receipted like every
        # other solve address.
        b = s.get("boss") or {}
        self._boss_hp = int(b["hp"]) if "hp" in b else None
        self._boss_start = int(b.get("start", 255))
        # Type-conditioned boss health (object-array engines): slots are
        # SHARED between fight objects and respawning mobs (Contra base:
        # wall guns type 54/68 + sensor 55 alternate with soldiers 33/44
        # in $0311-$0314; per-slot HP at $04BF-$04C2, causal fire receipt
        # 2026-07-31). hp = sum of hp_addrs[i] where type_addrs[i] holds
        # a listed type; reads as `start` when no fight object is live
        # (pre-wall AND post-destruction — the transit machinery owns the
        # post state).
        bt = s.get("boss_typed") or {}
        self._bt_types = frozenset(int(t) for t in bt.get("types", ()))
        self._bt_addrs = list(zip((int(a) for a in bt.get("type_addrs", ())),
                                  (int(a) for a in bt.get("hp_addrs", ()))))
        self._bt_start = int(bt.get("start", 12))
        # Movement-mode signature (optional): `state_sig: [{addr, match}]`.
        # Each entry contributes one bit: ram[addr] in match. Separates
        # aliased player modes at the same coordinates — the CV block-3
        # lesson: stair-climb states share (gx, y-band) with jump apexes,
        # tie-break to fewer steps, and the airborne dead-end wins the
        # cell; the staircase stays structurally invisible to the archive.
        # Optional 'mod': the bit tests (ram[addr] % mod) in match —
        # for PHASE observables (e.g. Contra's core open-window: frame
        # counter $001A % 16 in 7..14, discovered by pixel-phase mining
        # 2026-07-31, MI 0.53, accuracy 0.997). mod=0/absent = raw value.
        self._state_sig = [(int(e["addr"]),
                            frozenset(int(v) for v in e.get("match", ())),
                            int(e.get("mod", 0)))
                           for e in s.get("state_sig", ())]
        #: Public, because it is a CELL-KEY axis: `key_config_axes` reads
        #: it so --resume-archive can refuse an archive whose sig had a
        #: different bit count (the merged --gate-axes sidecar is already
        #: folded into `state_sig` by the time this runs, which is the
        #: whole reason the arity has to travel with the archive).
        self.state_sig_arity = len(self._state_sig)
        #: The same axis, by CONTENT (see state_sig_sha): two 1-bit sigs
        #: over different bytes have equal arity and unequal meaning,
        #: and the sidecar makes that the ordinary case, not an exotic
        #: one — arity alone would call those two archives one lineage.
        self.state_sig_sha = state_sig_sha(self._state_sig)
        # Room signature (optional): `room_sig: [addr,...]` — bytes stable
        # within a room and different across rooms (found by before/after
        # transition diff of our own climbs). Feeds room_id, so the sect/
        # psig transit machinery (built on SMB's $074E) counts CV room
        # progress even though gx resets in every room.
        self._room_sig = tuple(int(a) for a in s.get("room_sig", ()))
        # ROOM FINGERPRINTING (optional, default OFF => byte-identical;
        # ROOMGRAPH_ENGINE_2026-08-24 §4). `solve: room_fp: {mask,
        # settle, min_lines, pan_odo, warp_scene_min, palette_cokey,
        # max_rooms, sample_every}` arms the identity layer: the Solver
        # settles a masked NT-hash per worker, interns it to a
        # discovery ordinal and writes it at 0x804/0x805 in the
        # pseudo-RAM extension, where `area` / `room_sig` /
        # `room_advance.addr` / `y` may point (they read ram[addr]
        # blindly, so 0x800-0x806 references need no code). The mask is
        # emitted by scripts/room_fp_calibrate.py from OUR OWN
        # idle/walk frames (auto volatility mask, receipt required per
        # game under docs/receipts/room_fp/); the classifier constants
        # are probe-receipted, game-agnostic. Validation is loud and at
        # construction — a bad range must fail before any pool exists.
        rf = s.get("room_fp")
        self.room_fp = None
        self.room_fp_sha = ""
        if rf is not None:
            cfg = {
                "mask": [[int(lo), int(hi)] for lo, hi in
                         (rf.get("mask") or ())],
                "settle": int(rf.get("settle", 3)),
                "min_lines": int(rf.get("min_lines", 200)),
                "pan_odo": [int(v) for v in rf.get("pan_odo", (128, 384))],
                "warp_scene_min": int(rf.get("warp_scene_min", 2)),
                "palette_cokey": bool(rf.get("palette_cokey", False)),
                "max_rooms": int(rf.get("max_rooms", 1024)),
                "sample_every": int(rf.get("sample_every", 1)),
            }
            for lo, hi in cfg["mask"]:
                if not (0 <= lo < hi <= 2048):
                    raise SystemExit(
                        f"[go_explore_solve] room_fp.mask range "
                        f"[{lo}, {hi}) is not inside [0, 2048) — the "
                        f"snapshot is the 2 KB physical nametable VRAM "
                        f"and a mask outside it masks nothing.")
            if cfg["settle"] < 1:
                raise SystemExit(
                    f"[go_explore_solve] room_fp.settle must be >= 1, "
                    f"got {cfg['settle']}: a zero-sample settle would "
                    f"adopt every churning mid-transition frame as a "
                    f"room.")
            if not (1 <= cfg["max_rooms"] <= 4096):
                raise SystemExit(
                    f"[go_explore_solve] room_fp.max_rooms must be in "
                    f"[1, 4096], got {cfg['max_rooms']} — the ordinal "
                    f"rides a uint16 slot and the cap is the false-"
                    f"split blast shield.")
            if (len(cfg["pan_odo"]) != 2
                    or not 0 < cfg["pan_odo"][0] <= cfg["pan_odo"][1]):
                raise SystemExit(
                    f"[go_explore_solve] room_fp.pan_odo must be "
                    f"[min, max] px with 0 < min <= max, got "
                    f"{cfg['pan_odo']}.")
            if cfg["warp_scene_min"] < 1:
                raise SystemExit(
                    f"[go_explore_solve] room_fp.warp_scene_min must be "
                    f">= 1, got {cfg['warp_scene_min']}: at 0 every "
                    f"odometer-flat settle classifies warp and no fade "
                    f"edge can ever be banked.")
            if not (0 <= cfg["min_lines"] <= 262):
                raise SystemExit(
                    f"[go_explore_solve] room_fp.min_lines must be in "
                    f"[0, 262] (rendered scanlines), got "
                    f"{cfg['min_lines']}.")
            if cfg["sample_every"] < 1:
                raise SystemExit(
                    f"[go_explore_solve] room_fp.sample_every must be "
                    f">= 1, got {cfg['sample_every']}.")
            self.room_fp = cfg
            self.room_fp_sha = room_fp_config_sha(cfg)
        # WIN-CONDITION hook (optional, default OFF). Many games have no clean
        # level_key that advances on a clear (contra/kirby/ducktales all ship
        # `level_key: []` = coverage baseline: deep frontier cells accrue but a
        # clear NEVER fires). `solve: clear: {mode: ...}` lets is_clear ALSO
        # fire on a second, configured signal. level_key advance stays the sole
        # default clear, so a profile without a `clear:` key is byte-identical.
        # Modes:
        #   score_jump  {threshold}  — a single-step rise in the decoded
        #     progress (solve.progress, which for DuckTales already IS the
        #     money odometer) >= threshold. DuckTales's boss main-treasure is
        #     one atomic $1,000,000 pickup (>= the $50k
        #     largest gem) so a jump this large can ONLY be the clear; being
        #     per-step, gem farming can never fake it. threshold is in the same
        #     units as solve.progress (DuckTales scale /100 -> $1M = 10000).
        #   byte_change {addr[, direction: up|down|any][, target]} — ram[addr]
        #     moved off its level-start value (captured in note_start) in the
        #     given direction (default up): a stage/level counter incrementing
        #     on a real clear, without making it the level_key (which also gates
        #     the warp/dead machinery). `target` requires a specific landing.
        #   confluence  {[window][stride][min_signals][persist_checks]
        #                [progress_median][apu_weight][apu: {...}]
        #                [room_veto: {steps[, addrs][, min_progress]}]} — the
        #     multi-signal clear_detect confluence, streamed over a bounded
        #     rolling window (clear_detect.StreamingConfluenceDetector; see its
        #     docstring for which signals are RAM-derivable in the hot loop).
        #
        #     DETECTOR v2 (2026-08-08) adds three knobs on top of the
        #     2026-08-06 lives-drop veto. ALL DEFAULT-INERT, so an existing
        #     `clear: {mode: confluence}` profile (contra, gradius) is
        #     byte-identical until it opts in:
        #       progress_median: K — median-filter the progress series over K
        #         trailing samples before the coord test. THE combat-blip fix
        #         (Double Dragon: 72 -> 846 -> 88 in 5 steps read as a level
        #         load). K=5 erases any impulse under ~2 samples and leaves a
        #         real load's step change intact. Default 1 = off.
        #       persist_checks: N — the confluence must additionally hold N
        #         consecutive checks before the latch closes. Second, cheaper
        #         filter; NOT sufficient alone (see the detector docstring for
        #         the window arithmetic). Default 1 = shipped behavior.
        #     DETECTOR v3 (2026-08-09) adds ONE more, also default-inert:
        #       apu_weight: W — arm the APU channel-activity vote
        #         (clear_detect.ApuActivitySignal) with weight W, fed the
        #         5-bit $4015 length-counter mask the solver samples next to
        #         each RAM snapshot. The check becomes `tally + coord +
        #         W*apu >= min_signals`; at W = 0 (the default) no signal is
        #         built, no mask is sampled and the arithmetic is the shipped
        #         integer path. The vote is a self-calibrated, content-free
        #         change detector — it knows this game's own rolling
        #         per-channel activity rates and nothing whatsoever about
        #         fanfares. Being ADDITIVE it can only make firing easier at
        #         the default min_signals: 2; to make the audio modality
        #         REQUIRED, raise the bar with it (`min_signals: 3,
        #         apu_weight: 1.0` = all three signals must agree). `apu:
        #         {...}` passes the signal's own knobs through
        #         (short_window, baseline_window, min_baseline, sustain,
        #         gate_k, gate_floor, hold).
        #         MIND THE WARM-UP, it is not free: the vote needs
        #         short_window + min_baseline + sustain - 1 = 93 observations
        #         in a FRESH detector before it can be 1, so with the audio
        #         vote REQUIRED the first check that can pass sits at
        #         observation 100 (clear_observation_budget computes it from
        #         the knobs). The live hook's state is per BURST, so --burst
        #         must exceed that — the default 64 does NOT, and the solver
        #         prints a loud line when it doesn't
        #         (warn_if_burst_starves_clear_hook). Bounded replays budget
        #         for it themselves: the counterfactual gate raises its
        #         pre-steps to cover it.
        #
        #     WHAT A WINDOWED HOOK COSTS YOU AT BANKING TIME (2026-08-10).
        #     Every mode in this block except `confluence` is STATELESS — it
        #     reads the current frame, so a replay of any length sees exactly
        #     what the live hook saw. A rolling detector can instead fire on
        #     evidence it accumulated over the whole trajectory, and that
        #     fire is reproduced perfectly by --verify-bank (same actions,
        #     same root, same accumulated state) while corresponding to
        #     nothing the game did. --counterfactual-gate is the check that
        #     catches it: a clear no snapshot INSIDE the trajectory can
        #     reproduce is refused with verdict `state_artifact` (receipted
        #     on Double Dragon, runs/detector_gate_20260810/). Configure a
        #     confluence hook expecting that gate to be on.
        #       room_veto: {steps: M[, addrs: [..]][, min_progress: P]} — a
        #         change of the room observable within the last M observations
        #         VETOES a fire unless `progress` has advanced by >= P (default
        #         1) measured from just before that change. This is the Kirby
        #         fix: a room-based game loads a fresh room far more often than
        #         it finishes a stage, and "a fresh room loading in" is
        #         literally what the coord signal fingerprints, so the only
        #         generic discriminator is whether the game's own progress
        #         observable actually moved forward across the transition.
        #         `addrs` names the room bytes explicitly; omitted, it uses the
        #         adapter's already-verified room identity (level_key + area +
        #         room_sig) — no new addresses either way. steps: 0 = disarmed
        #         (default). Progress samples above progress_cap are dropped
        #         rather than believed: a room load is exactly when the page
        #         byte reads transitional garbage, and one such sample used to
        #         satisfy the advance test in the very call that armed the
        #         veto — permanently disarming it. Set progress_cap for the
        #         game if the default 30000 is not right for its scale.
        #         PRECISION OVER RECALL, deliberately: on a game whose TERMINAL
        #         transition is observationally identical to an internal one
        #         (same room-byte change, same progress reset) this suppresses
        #         the terminal one too, and the hook degenerates toward "never
        #         fires". That is the intended failure direction — a missed
        #         clear costs one re-search, a fabricated one poisons
        #         go_explore_chain's next-entrance extraction and the receipt
        #         corpus. Raise recall by pointing `progress` at an observable
        #         that really does advance across the terminal transition, not
        #         by disarming the veto.
        cl = s.get("clear") or {}
        self._clear_mode = cl.get("mode")
        self._clear_threshold = float(cl.get("threshold", 0))
        self._clear_addr = int(cl["addr"]) if "addr" in cl else None
        self._clear_dir = str(cl.get("direction", "up"))
        self._clear_target = int(cl["target"]) if "target" in cl else None
        self._clear_baseline = None          # byte_change: set by note_start
        self._conf_window = int(cl.get("window", 240))
        self._conf_stride = int(cl.get("stride", 20))
        self._conf_min_signals = cl.get("min_signals")
        self._conf_persist = cl.get("persist_checks")
        self._conf_median = cl.get("progress_median")
        # APU CHANNEL-ACTIVITY VOTE (v3, opt-in). `apu_weight: W` (0 = off,
        # the default) arms clear_detect.ApuActivitySignal as a third vote
        # inside the streaming detector, fed the 5-bit $4015 length-counter
        # mask the solver samples alongside each RAM snapshot. `apu: {...}`
        # passes that signal's own knobs through verbatim (short_window,
        # baseline_window, min_baseline, sustain, gate_k, gate_floor, hold).
        # At weight 0 no signal object is built and no mask is ever sampled,
        # so an existing profile is byte-identical and pays nothing.
        self._conf_apu_weight = float(cl.get("apu_weight", 0.0) or 0.0)
        self._conf_apu = dict(cl.get("apu") or {})
        # Cache for clear_observation_budget: the answer is a pure function
        # of the knobs above, so it is computed once from a throwaway
        # detector and reused (callers ask it once per banked candidate).
        self._conf_budget: int | None = None
        rv = cl.get("room_veto") or {}
        self._rv_steps = int(rv.get("steps", 0))
        self._rv_addrs = tuple(int(a) for a in rv.get("addrs", ()))
        self._rv_min_progress = int(rv.get("min_progress", 1))

    def clear_verify_margin(self) -> int:
        """Extra NOOP observations a replay must feed the clear hook, past the
        end of the trace, before "it did not reproduce" means anything.

        Zero for every STATELESS hook (level_key, finale, byte_change,
        score_jump): those fire on the exact frame the RAM satisfies them, so
        the replay's verdict lands on the same frame the live one did.

        NON-ZERO only for `confluence`, which is windowed AND phase-dependent.
        The streaming detector evaluates only when its own observation counter
        satisfies `_n % stride == 0`, and that counter starts when the ctx is
        created. Live, _assign() builds a fresh ctx per BURST, so the fire
        lands at (F - burst_start) % stride == 0; a replay from the ROOT builds
        its ctx at index 0, so its checks sit at multiples of stride from
        there. The two phases agree only when burst_start % stride == 0, and
        because the banked trace is cut exactly at F there are no steps left
        for the phase to re-align in — the replay's last check falls BEFORE the
        level-load signature and returns 'no_clear'. Measured over the real
        detector on a genuine no-life-lost clear: 136 of 200 burst alignments
        (68%) were rejected with no margin, 0 of 200 with a margin of
        stride-1 or more. Without this, --verify-bank being ON by default
        would have made the four-game confluence gate report zero solutions,
        with 'this is a fabricated clear' printed over every real win.

        stride * persist_checks, so the replay gets a full check period per
        consecutive confluence the live latch required. At the defaults that
        is 20 actions (~54 ms), against candidates of 500-1,700."""
        if self._clear_mode != "confluence":
            return 0
        persist = max(1, int(self._conf_persist or 1))
        return max(0, self._conf_stride) * persist

    def needs_apu(self) -> bool:
        """Whether this adapter's clear hook wants the per-step APU
        channel-activity mask in its ctx (`ctx["_apu_mask"]`).

        False for every profile that has not set `clear: {apu_weight: > 0}`,
        which is all of them by default — the caller then never calls
        pool.apu_activity_all() at all, so the audio modality costs exactly
        nothing on the paths that do not ask for it."""
        return self._clear_mode == "confluence" and self._conf_apu_weight > 0

    def _new_clear_detector(self):
        """Build the streaming confluence detector this profile's knobs
        describe.

        One constructor shared by the live hook and by
        clear_observation_budget, so a warm-up budget can never be computed
        from a differently-configured detector than the one that actually
        runs."""
        import sys as _sys
        _sd = str(Path(__file__).resolve().parent)   # scripts/ on path
        if _sd not in _sys.path:
            _sys.path.insert(0, _sd)
        from clear_detect import StreamingConfluenceDetector
        return StreamingConfluenceDetector(
            self.progress, window=self._conf_window,
            stride=self._conf_stride,
            min_signals=self._conf_min_signals,
            persist_checks=self._conf_persist,
            progress_median=self._conf_median,
            apu_weight=self._conf_apu_weight,
            apu_params=self._conf_apu)

    def clear_observation_budget(self) -> int:
        """Observations a FRESH clear-hook ctx must be fed before this hook is
        CAPABLE of firing — the warm-up counterpart to clear_verify_margin's
        phase allowance, and the second number any bounded replay owes a
        windowed detector.

        Zero for every STATELESS hook (level_key, finale, byte_change,
        score_jump): they read the current frame, so one observation is
        enough. For `confluence` the answer is the streaming detector's own
        `warmup_observations()` — a detector built from these very knobs is
        ASKED, rather than the arithmetic being copied here, so the two can
        never drift apart the way a duplicated constant would.

        WHY IT EXISTS. A replay that feeds the hook fewer observations than
        this cannot tell "the clear did not reproduce" from "the detector
        never got to look", and both come back as False. Measured on the
        counterfactual gate before this hook existed: its branches see
        cf_pre_steps (32) + clear_verify_margin (20) = 52 observations, while
        the `min_signals: 3, apu_weight: 1.0` configuration recommended for
        RAM-shaped false positives needs 100 before an APU-backed check can
        pass (90 to have a null at all, +3 to sustain it, rounded up to the
        next check point). Every branch INCLUDING the control therefore
        returned no_clear, the probe reported 'inconclusive' with ok=True on
        every candidate forever, and the gate refused nothing while still
        paying 4-40 s a candidate — a silent no-op that looked like a working
        gate in the receipts. Any caller with a fixed observation budget must
        ask for this number and either meet it or say out loud that it could
        not (Solver.counterfactual_probe now does both)."""
        if self._clear_mode != "confluence":
            return 0
        if self._conf_budget is None:
            self._conf_budget = int(
                self._new_clear_detector().warmup_observations())
        return self._conf_budget

    def note_start(self, ram) -> None:
        """Record the level-entrance baseline the byte_change WIN-CONDITION
        compares against (called once from Solver.seed at the entrance). No-op
        unless a byte_change clear hook is configured."""
        if self._clear_mode == "byte_change" and self._clear_addr is not None:
            self._clear_baseline = int(ram[self._clear_addr])

    def progress(self, ram) -> int:
        if self._ptiles is not None:
            v = 0
            for a in self._ptiles:          # MSB-first digit tiles
                t = int(ram[a])
                if t < 10:                   # blank tile (e.g. 0x24) skipped
                    v = v * 10 + t
            return v // self._pscale
        v = int(ram[self._plo])
        if self._phi is not None:
            v |= int(ram[self._phi]) << 8
        return v

    def progress_pair(self, ram) -> tuple:
        """`(value, hi, lo)` — the composed observable plus the raw bytes
        it was composed from, for the torn-read filter. `hi`/`lo` are
        None whenever there is no two-byte read to tear (single byte, or
        a decoded HUD field), which makes `borrow` inert by construction
        rather than by a caller remembering to check."""
        v = self.progress(ram)
        if self._ptiles is not None or self._phi is None:
            return (v, None, None)
        return (v, int(ram[self._phi]), int(ram[self._plo]))

    def progress_atomic_read(self, pool, wid: int) -> tuple:
        """`(value, consistent)` for `progress: {atomic: true}` — the
        composed 16-bit progress pair read through the Rust torn-read-
        guarded peek (`Pool.peek_u16_consistent`) rather than composed
        from a step snapshot. `consistent` is False on an unadjudicated
        boundary (mid-OAM-DMA, a missing/panicked scratch instance); the
        caller must treat that exactly like a transition-frame read, not
        trust `value`. Raises SystemExit rather than silently falling
        back to the torn read this flag exists to avoid — the profile
        parser already refused `atomic: true` on a single-byte progress
        read, so the only way here is a pool built before the binding
        shipped."""
        if not hasattr(pool, "peek_u16_consistent"):
            raise SystemExit(
                "[go_explore_solve] progress.atomic: true but this "
                "nes_core build has no peek_u16_consistent binding — "
                "rebuild it (`make build`) and refresh the venv .so "
                "(maturin does not update it in place) before using "
                "this flag.")
        return pool.peek_u16_consistent(wid, self._plo, self._phi)

    def level_key(self, ram) -> tuple:
        return tuple(int(ram[a]) for a in self._lk)

    def lives(self, ram) -> int:
        return int(ram[self._lives])

    def area(self, ram) -> int:
        return int(ram[self._area]) if self._area is not None else 0

    def y(self, ram) -> int:
        return int(ram[self._y])

    def swim(self, ram) -> int:
        return 0

    def _typed_hp(self, ram) -> int:
        live = [int(ram[h]) for t, h in self._bt_addrs
                if int(ram[t]) in self._bt_types]
        return sum(live) if live else self._bt_start

    def cell_fn(self, ram) -> tuple:
        # Same arity as SMB's cell (selection caches index key[-5]/key[-1];
        # boss HP and the mode bits ride in free middle slots so those
        # stay intact).
        if self._bt_addrs:
            hp = self._typed_hp(ram)
        elif self._boss_hp is not None:
            hp = int(ram[self._boss_hp])
        else:
            hp = 0
        sig = 0
        for i, (a, m, md) in enumerate(self._state_sig):
            v = int(ram[a]) % md if md else int(ram[a])
            if v in m:
                sig |= 1 << i
        return (self.area(ram), hp, sig,
                self.y(ram) // Y_BAND, self.progress(ram) // GX_BUCKET)

    def score_bonus(self, ram) -> int:
        # Damage dealt dominates within-room gx differences (a whip hit is
        # worth more frontier than any sidestep) without touching the
        # cross-room score ordering scale.
        if self._bt_addrs:
            return max(0, self._bt_start - self._typed_hp(ram)) * 2000
        if self._boss_hp is None:
            return 0
        return max(0, self._boss_start - int(ram[self._boss_hp])) * 2000

    def room_veto_key(self, ram) -> tuple:
        """The observable whose CHANGE means 'a different room/level just
        loaded'. `clear.room_veto.addrs` when the profile names bytes
        explicitly; otherwise this adapter's own already-verified room
        identity (level_key + area + room_sig). No new addresses either way."""
        if self._rv_addrs:
            return tuple(int(ram[a]) for a in self._rv_addrs)
        return self.room_id(ram)

    # Room-veto verdicts (see room_veto_step).
    RV_NONE, RV_HOLD, RV_DISCARD = "none", "hold", "discard"

    def room_veto_step(self, ram, ctx: dict) -> str:
        """Advance this worker's room/progress bookkeeping by ONE observation
        and say what the room veto wants done with a clear vote right now.

        Must be called on every observation, not only the ones that fire: it
        IS the history.

          RV_NONE     nothing to veto — no room change is pending, or progress
                      has since advanced past where it stood just before one.
          RV_HOLD     a room change is pending and progress has not advanced
                      yet: suppress a fire but KEEP the evidence; the advance
                      may still arrive inside the window.
          RV_DISCARD  the window expired with no advance, so that was an
                      ordinary room transition: suppress AND throw the
                      evidence away. Fires exactly once per room change.

        The three-state shape is load-bearing, and a plain boolean is not
        enough in either direction. Suppress-only leaks: the samples that
        produced the fire sit in the detector's rolling window, so the same
        false fire simply lands again the moment the window expires.
        Discard-immediately over-blocks: it destroys the evidence before the
        progress advance that would have EXONERATED the transition has had its
        `steps` observations to show up, which would make the escape clause
        unreachable and the veto a blanket mute.

        GARBAGE-READ GUARD (2026-08-08): every progress sample is filtered
        through progress_cap before it is used or stored. A room load is
        precisely when the page byte reads transitional garbage, and the escape
        clause runs on the arming frame too — so an unguarded read let ONE
        garbage sample satisfy `prog - prog_before >= min_progress` in the very
        call that armed the veto, permanently disarming it and re-banking the
        Kirby false positive. Replay verification cannot catch that: the
        emulator is deterministic, so the replay reproduces the same garbage
        read and the same fabricated verdict. Every other consumer of
        progress() in this file already guards it the same way (observe()'s
        `gx > game.progress_cap` skip, explore()'s cap clamp); this is the one
        that did not."""
        n = ctx["_rv_n"] = ctx.get("_rv_n", 0) + 1
        key = self.room_veto_key(ram)
        raw = self.progress(ram)
        # None == "not an observation of progress at all", not "zero": a
        # garbage sample must neither advance the escape test nor become the
        # baseline a later escape is measured from.
        prog = None if raw > self.progress_cap else raw
        prev_key = ctx.get("_rv_key")
        prev_prog = ctx.get("_rv_prog_prev")
        ctx["_rv_key"] = key
        if prog is not None:
            ctx["_rv_prog_prev"] = prog     # last GOOD sample, never garbage
        if prev_key is not None and key != prev_key:
            ctx["_rv_at"] = n
            # Progress as it read on the step BEFORE the room changed — the
            # transition frame itself already reads the new room's value.
            # Both may be garbage (None); the veto then arms with no baseline
            # and simply cannot be escaped, which is the safe direction.
            ctx["_rv_prog_before"] = prev_prog if prev_prog is not None else prog
        at = ctx.get("_rv_at")
        if at is None:
            return self.RV_NONE
        before = ctx.get("_rv_prog_before")
        if (prog is not None and before is not None
                and prog - int(before) >= self._rv_min_progress):
            ctx["_rv_at"] = None          # the transition earned its keep
            return self.RV_NONE
        if n - at <= self._rv_steps:
            return self.RV_HOLD
        ctx["_rv_at"] = None              # act once per room change
        return self.RV_DISCARD

    def is_clear(self, start_key: tuple, ram, ctx: dict | None = None) -> bool:
        # Forward = lexicographic advance of the level key (stage counters
        # increment; a game-over reset reads backward and lands in `dead`).
        if self.level_key(ram) > tuple(start_key):
            return True
        # Optional configured WIN-CONDITION (default OFF). ctx is the caller's
        # per-worker state dict; None (e.g. the seed observe) short-circuits the
        # stateful modes so only the level_key advance above can fire.
        m = self._clear_mode
        if not m or ctx is None:
            return False
        if m == "score_jump":
            v = self.progress(ram)
            prev = ctx.get("_clear_prev")
            ctx["_clear_prev"] = v
            return prev is not None and (v - prev) >= self._clear_threshold
        if m == "byte_change":
            if self._clear_addr is None or self._clear_baseline is None:
                return False
            cur = int(ram[self._clear_addr])
            if self._clear_target is not None:
                return cur == self._clear_target and cur != self._clear_baseline
            if self._clear_dir == "up":
                return cur > self._clear_baseline
            if self._clear_dir == "down":
                return cur < self._clear_baseline
            return cur != self._clear_baseline
        if m == "confluence":
            det = ctx.get("_clear_det")
            if det is None:
                det = ctx["_clear_det"] = self._new_clear_detector()
            # The APU mask rides in ctx rather than in is_clear's signature:
            # every caller (observe, replay_verify, the counterfactual gate,
            # the show) already threads ctx, and every duck-typed game
            # adapter in the corpus implements the 3-argument form.
            #
            # The one-argument push() is kept literally, not defaulted to
            # None, on the disarmed path: `det` is whatever the ctx holds,
            # and the corpus is full of duck-typed one-argument detector
            # stubs. A default-off knob must not widen the call it makes.
            fired = (det.push(ram, ctx.get("_apu_mask"))
                     if self._conf_apu_weight > 0 else det.push(ram))
            # ROOM-TRANSITION VETO (v2, 2026-08-08). Bookkeeping runs on
            # every observation, so it is evaluated before the early
            # returns below; disarmed profiles (room_veto.steps 0, the
            # default) skip it entirely and stay byte-identical.
            rv = (self.room_veto_step(ram, ctx) if self._rv_steps > 0
                  else self.RV_NONE)
            # Lives-drop veto (2026-08-06): confirmed empirically on
            # Gradius that a real death (explosion + entity-slot wipe +
            # respawn) trips this detector's "coord" signal identically
            # to a real level load. Solver.observe() now resolves
            # is_dead() BEFORE is_clear() (v2), so this is belt-and-
            # braces for that path — but it is also the only guard for
            # any caller that checks the clear hook on its own. Track
            # this worker's own lives across steps (not the archive-wide
            # start_lives baseline, which only bounds a whole lineage,
            # not a single frame) and refuse to fire the same step lives
            # just dropped.
            cur_lives = self.lives(ram)
            prev_lives = ctx.get("_clear_prev_lives")
            ctx["_clear_prev_lives"] = cur_lives
            # ---- every per-step bookkeeping side effect is now done; only
            # verdicts below this line, so no early return can leave a
            # tracker one step stale.
            if rv == self.RV_DISCARD:
                # Drop the evidence with the verdict — including evidence
                # that has not fired yet. The samples spanning an ordinary
                # room load ARE the coord signature, and they stay in the
                # detector's rolling window for another ~window/stride
                # checks; leaving them there only postpones the same false
                # fire past the veto's own horizon.
                det.reset()
                return False
            if not fired:
                return False
            if prev_lives is not None and cur_lives < prev_lives:
                return False
            if rv == self.RV_HOLD:
                # Suppress, but keep the window: a progress advance inside
                # the remaining veto steps still exonerates this fire.
                return False
            return True
        return False

    def is_dead(self, ram, start_lives: int) -> bool:
        # MODULAR decrement (2026-08-23, the Ninja Gaiden underflow):
        # a lives counter that displays REMAINING lives can read 0 in
        # normal play, so death decrements 0 -> 255 and a plain `<`
        # never fires — NG banked a 5888-px corpse frontier because
        # every death was invisible. `(start - cur) % 256 in 1..8` reads
        # any small wrap-aware decrement as death, keeps Rygar's 1 -> 0
        # and SMB-class 3 -> 2, and stays safe on extra-life pickups
        # (cur = start+1 gives delta 255, not 1..8). Transition blips
        # through adjacent values are absorbed by observe()'s 3-step
        # death debounce, same as before.
        d = (int(start_lives) - int(self.lives(ram))) % 256
        if 1 <= d <= 8:
            return True
        return (self._pstate is not None
                and int(ram[self._pstate]) in self._death_states)

    def is_finale(self, start_key: tuple, ram) -> bool:
        f = self._finale
        return (bool(f) and tuple(start_key) == tuple(f["level_key"])
                and int(ram[int(f["addr"])]) == int(f["value"]))

    def room_id(self, ram) -> tuple:
        return (self.level_key(ram) + (self.area(ram),)
                + tuple(int(ram[a]) for a in self._room_sig))

    @staticmethod
    def label(key: tuple) -> str:
        return "-".join(str(x) for x in key)


class ProfileNotConstructible(RuntimeError):
    """Raised by make_game() for a `solve:` block that declares
    `constructible: false` — a profile behaviourally proven to be
    missing something GenericGame hard-requires (Tetris has no
    jump/ballistic `y` to discover; it is a static-playfield puzzle
    game, not a platformer), rather than one that simply has not been
    onboarded yet. Before this existed, both cases surfaced identically
    as a bare `KeyError('y')` deep inside `GenericGame.__init__`, and
    nothing upstream (the roster-construction check, `clear_reachability.py`)
    could tell them apart. `reason` carries the profile's own recorded
    evidence forward into the exception message."""


def make_game(profile: dict):
    """SMB-engine profiles carry no `solve:` section — they get the
    byte-exact SMB adapter (which reads an optional `rom:` override, so
    Lost Levels / any SMB1-engine game works). A profile with `solve:`
    opts into the generic path, unless it explicitly declares
    `constructible: false` (see ProfileNotConstructible)."""
    if "solve" not in profile:
        return SmbGame(profile)
    solve = profile["solve"]
    if solve.get("constructible") is False:
        reason = solve.get("reason", "no reason recorded")
        raise ProfileNotConstructible(
            f"{profile.get('name', '<unnamed>')!r} declares "
            f"solve.constructible: false — {reason}")
    return GenericGame(profile)


def derive_transition_macros(action_space: list, room_advance: dict | None) -> list:
    """Discrete-transition gate (the Kirby lesson). Room-based games stall
    because forward progress cannot gradient the player into a DISCRETE
    'grounded + UP' door/room entry — the camera clamps at the scroll limit so
    world-X gives no pull toward the door. When `solve: room_advance:` is
    configured this AUTO-DERIVES the door-entry maneuver macros (up / up+jump /
    up+right holds) straight from the profile's own action space — no
    hand-authored hold_macros needed — for the solver to inject near the deep
    frontier. Returns [(action_idx, hold_steps)]; empty when the space has no
    UP move (the gate is then inert). This is the exact settle-then-HOLD
    mechanism the existing hold_macros use (Solver.explore), just auto-built and
    frontier-biased rather than fired uniformly.

    Config: `room_advance: {addr[, steps=20][, p=0.05][, near=24][, buttons]}`.
    `addr` = the room/area-id byte whose advance we want to provoke (also
    surfaced as telemetry). `buttons` overrides the auto-derivation with an
    explicit list of button combos."""
    if not room_advance:
        return []
    steps = int(room_advance.get("steps", 20))
    want = room_advance.get("buttons")
    combos = ([set(b) for b in want] if want
              else [set(c) for c in action_space if "up" in set(c)])
    out = []
    for combo in combos:
        idx = next((i for i, c in enumerate(action_space) if set(c) == combo),
                   None)
        if idx is not None:
            out.append((idx, steps))
    return out


# --- room-graph router (T3, ROOMGRAPH_ENGINE_2026-08-24 §3 rows 7-9) --
# Selection-side ONLY: these helpers read the archive's own cells and
# the RoomIndex adjacency — both derived purely from our rollouts — and
# never touch score, keys, or the transit machinery. All geometry is in
# CELL units (gx buckets / y bands, key[-1] / key[-2]): one coordinate
# system for the boundary sublists, the router's routing and the
# route-follow macro gate, whatever the profile's progress source is.


def derive_direction_macros(action_space: list, steps: int = 20) -> dict:
    """Direction-hold macros for the router's route-follow gate, derived
    STRUCTURALLY from the profile's own action space — the
    generalization of derive_transition_macros' up-only rule to all four
    room sides. {side: [(action_idx, hold_steps), ...]}: every combo
    containing that side's d-pad button qualifies (a Metroid door wants
    right- or left-holds, some with shoot folded in; Zelda's stairs want
    up — the injection samples uniformly among them). A side the space
    cannot express gets an empty list and its rolls are inert."""
    btn = {"E": "right", "W": "left", "N": "up", "S": "down"}
    return {side: [(i, int(steps))
                   for i, c in enumerate(action_space) if b in set(c)]
            for side, b in btn.items()}


def room_cell_ord(key, psig_off):
    """The room ordinal a cell key carries in its threaded psig tail
    (key[3]), or None. EXACTLY _room_seed's extraction (`psig_off` =
    Solver._room_psig_off: negative lo/hi offsets from the tail's end +
    the room_sig arity), so the router's pools group cells by the same
    identity a restored worker re-seeds from. A lineage that has not
    transited yet has psig () and belongs to no room; the ROOM_UNKNOWN
    sentinel (a transit recorded before the worker's fingerprint had
    settled) is likewise unroutable."""
    if psig_off is None:
        return None
    sig = key[3]
    lo_i, hi_i, n = psig_off
    if not isinstance(sig, tuple) or len(sig) < n:
        return None
    try:
        o = (int(sig[lo_i]) & 0xFF) | ((int(sig[hi_i]) & 0xFF) << 8)
    except (TypeError, ValueError):
        return None
    return None if o == ROOM_UNKNOWN else o


def aliased_rooms(adj: dict, min_traversals: int = 3) -> set:
    """Fingerprint-aliasing audit (§7.1, D1 graft): one (src, kind, dir)
    exit reaching >= 2 distinct dsts, each traversed >= min_traversals
    times, means src's fingerprint is carrying more than one physical
    room (enemies are OAM — invisible to the hash — so structurally
    identical rooms CAN merge). Identity is left alone; the router
    down-weights the marked node x0.25. Thresholds are D1's: a single
    stray traversal is noise, two established destinations are not."""
    out: set = set()
    for src, dsts in adj.items():
        fan: dict = {}
        for dst, e in dsts.items():
            if int(e.get("count", 0)) >= int(min_traversals):
                fan.setdefault((e.get("kind"), e.get("dir")), set()).add(dst)
        if any(len(v) >= 2 for v in fan.values()):
            out.add(src)
    return out


def room_boundaries(cells, near: int):
    """Per-side boundary sublists of one room's selectable cells: the
    cells within `near` gx buckets (E/W) or y bands (N/S) of the room's
    own cell-extent bbox — the states a worker would push PAST to leave
    through that side. Returns ({side: [cells]}, (gx_lo, gx_hi, y_lo,
    y_hi)), cell units. NES y grows downward, so N is the LOW band. The
    bbox here is the archive's footprint, deliberately not the odometer
    bbox in RoomIndex.meta — cells are keyed in profile progress/y
    units and the two scales need never be reconciled (the meta bbox
    serves the RG-1a false-merge audit, nothing here)."""
    gxs = [int(c.key[-1]) for c in cells]
    ybs = [int(c.key[-2]) for c in cells]
    gx0, gx1, y0, y1 = min(gxs), max(gxs), min(ybs), max(ybs)
    n = int(near)
    sides = {
        "E": [c for c in cells if int(c.key[-1]) >= gx1 - n],
        "W": [c for c in cells if int(c.key[-1]) <= gx0 + n],
        "S": [c for c in cells if int(c.key[-2]) >= y1 - n],
        "N": [c for c in cells if int(c.key[-2]) <= y0 + n],
    }
    return sides, (gx0, gx1, y0, y1)


def room_weight(in_artic: bool, aliased: bool, u: int, visits: float,
                artic_w: float, exit_w: float) -> float:
    """Router room weight (§3 row 8): w(r) = (1 + artic_w·[r ∈ artic] +
    exit_w·U(r)) / sqrt(V(r)+1), aliased rooms x0.25. V(r) = the room's
    summed times_chosen — the same count-prior family as the count
    arm's 1/sqrt(times_chosen+1), and deliberately NO score term: the
    score is exactly what makes an off-axis room look worthless."""
    w = ((1.0 + (float(artic_w) if in_artic else 0.0)
          + float(exit_w) * float(u))
         / (float(visits) + 1.0) ** 0.5)
    return w * 0.25 if aliased else w


def room_frontier(ords, recent_k: int, degree: dict, artic: set,
                  aliased: set, u: dict) -> list:
    """The router's frontier set F (§3 row 8): recently-discovered rooms
    (ordinal within recent_k of the newest — ordinals ARE discovery
    order, so recency is a comparison, not a clock), graph leaves
    (undirected degree <= 1, isolated rooms included), articulation
    rooms, and un-aliased rooms with an unexplored boundary side
    (U(r) > 0). Sorted so the weighted draw over F is deterministic
    under a fixed seed regardless of pool-dict insertion order."""
    ords = list(ords)
    if not ords:
        return []
    mx = max(ords)
    return sorted(o for o in ords
                  if o >= mx - int(recent_k)
                  or int(degree.get(o, 0)) <= 1
                  or o in artic
                  or (o not in aliased and int(u.get(o, 0)) > 0))


def route_near_side(direction, bbox, gx_bucket: int, y_band: int,
                    near: int) -> bool:
    """Is a worker at (gx_bucket, y_band) within `near` cell units of
    `bbox`'s `direction` side? The route-follow macro gate — the same
    band room_boundaries uses, so a routed worker rolls its
    direction-hold exactly where the router sampled its target cells."""
    gx0, gx1, y0, y1 = bbox
    n = int(near)
    if direction == "E":
        return gx_bucket >= gx1 - n
    if direction == "W":
        return gx_bucket <= gx0 + n
    if direction == "S":
        return y_band >= y1 - n
    if direction == "N":
        return y_band <= y0 + n
    return False


def update_stall(stall: dict, n_cells: int, now: float) -> None:
    """Pure state transition for the flat-archive stall watchdog: bumps
    `flat_windows` when the archive hasn't grown since the last call,
    resets it otherwise. Caller decides the polling cadence and what to
    do once flat_windows crosses its own threshold (>=2 == 2+ minutes
    with zero new cells, matching live_solve_show.py's convention)."""
    stall["flat_windows"] = (stall["flat_windows"] + 1
                             if n_cells <= stall["last_cells"] else 0)
    stall["last_cells"], stall["last_t"] = n_cells, now


def column_extremes(keys, mode: str) -> dict:
    """Per-gx-column ceiling (mode 'up') or floor (mode 'down') of the
    ORTHOGONAL axis: {gx bucket -> extreme y-band} over the given cell
    keys. key[-1] is the gx bucket, key[-2] the y-band (both SmbGame and
    GenericGame cells end (..., y_band, gx_bucket)). 'up' is min because
    the NES y axis grows downward — a smaller band is higher up."""
    pick = min if mode == "up" else max
    ext: dict = {}
    for k in keys:
        col, yb = k[-1], k[-2]
        cur = ext.get(col)
        ext[col] = yb if cur is None else pick(cur, yb)
    return ext


def ortho_pool(cells, mode: str, band: int) -> list:
    """The orthogonal frontier: cells within `band` y-bands of THEIR OWN
    column's extreme. Per-column, not global — a global cutoff would let
    one tall column's ceiling define the whole level and silently drop
    every other column's top-of-stack cell (the aliasing trap)."""
    ext = column_extremes([c.key for c in cells], mode)
    return [c for c in cells if abs(c.key[-2] - ext[c.key[-1]]) <= band]


def ortho_armed(mode: str, pin_time: float, now: float,
                pin_secs: float) -> bool:
    """Whether the orthogonal arm is live: a mode is selected AND the
    primary (x) frontier has been pinned at least `pin_secs`. Reuses the
    solver's own self-measured `_pin_time` — the same saturation signal
    the heuristic-inversion window already runs on, no new machinery and
    no external map."""
    return mode not in (None, "", "off") and now - pin_time >= pin_secs


def count_wmax(door_weight: float, ortho_weight: float,
               gate_weight: float = 1.0) -> float:
    """Exact Wmax for the count arm's O(1) rejection sampling. The prior
    is W = 1/sqrt(times_chosen+1) * (score_norm + 0.1) <= 1.1, times any
    armed multiplier (R4 doors, orthogonal frontier, gate-opener).
    Under-stating it would silently truncate the prior; 1.1 is the legacy
    value every multiplier reduces to when off.

    `gate_weight` is a DEFAULT ARG so the 2-argument call contract every
    existing caller and test uses keeps holding exactly."""
    return (1.1 * max(door_weight, 1.0) * max(ortho_weight, 1.0)
            * max(gate_weight, 1.0))


# ---------------------------------------------------------------------
# GATE-OPENER ARM (default off; byte-identical when off).
#
# The saturated-boundary hypothesis this arm exists to test: the archive
# has visited every POSITION at the wall many times and knows nothing
# about which INTERACTIONS were tried there. The arm enumerates the
# interactions off archive savestates, ranks the RAM they move against a
# paired NOOP control, and carries the survivors as a SHADOW LEDGER —
# candidates never touch the live cell key, so `cells`/`new_cells` stay
# comparable between an armed run and its control. Promotion to a real
# `state_sig` bit happens BETWEEN runs, through the --gate-axes sidecar.
#
# Everything below is pure and unit-testable without a Solver, a ROM or
# a Pool, in the same shape as ortho_armed/count_wmax above.
# ---------------------------------------------------------------------

def resolve_gate_pin_secs(args) -> float:
    """`--gate-pin-secs`, EXPLICIT and required whenever the arm is on.

    v1 does not derive it. The obvious derivation (a multiple of the
    median inter-advance interval) has no data source: `_pin_time` is
    overwritten in place at every frontier advance, no interval history
    is recorded anywhere, and a wall that logs ZERO advances in-session
    leaves that median undefined — which is exactly the case the arm is
    for. Absent (or negative) resolves to -1.0 = never arms."""
    v = getattr(args, "gate_pin_secs", None)
    return -1.0 if v is None else float(v)


def gate_armed(mode: str, pin_time: float, now: float, pin_secs: float,
               typed: bool, band_growth_ok: bool) -> bool:
    """Whether the gate-opener arm is live. Mirrors ortho_armed's shape.

    ALL of: a mode is selected; the pin clock has run at least `pin_secs`
    (a NEGATIVE pin_secs disables the arm outright — the same disable
    sentinel --inversion-pin-secs uses, and the reason the elapsed test
    alone is not enough, since `now - pin_time >= -1` is true forever);
    the operator attested a typed corpus row with --gate-target-typed;
    and the band has stopped growing.
    """
    return (mode not in (None, "", "off")
            and float(pin_secs) >= 0.0
            and now - pin_time >= float(pin_secs)
            and bool(typed)
            and bool(band_growth_ok))


def gate_arm_floor_secs(cadence: float, need: int = 3) -> float:
    """The earliest a sweep can possibly arm, in seconds.

    `band_growth_stalled` needs `need + 1` checkpoints before it stops
    returning False, so the checkpoint cadence multiplied by that count
    IS the arming floor — a structural property of the conjunct, not a
    tuning choice. Stated as a function because it was invisible while
    the cadence was a hardcoded 60 s inside the progress line: nothing
    could arm inside four minutes, on runs whose whole grant was fifteen
    (repair D-ARM, measured arming times 240-300 s in the K0 receipts).
    """
    return max(0.0, float(cadence)) * (int(need) + 1)


def band_cell_count(keys, top_gx: int | None = None, band: int = 24) -> int:
    """Cells within `band` gx buckets of the frontier — the progress-line
    field the band-growth conjunct reads. One pass, no allocation beyond
    the count; `top_gx` is recomputed from the keys when not supplied."""
    keys = list(keys)
    if not keys:
        return 0
    if top_gx is None:
        top_gx = max(int(k[-1]) for k in keys)
    floor = int(top_gx) - int(band)
    return sum(1 for k in keys if int(k[-1]) >= floor)


def band_growth_stalled(history, need: int = 3, tol: float = 0.05) -> bool:
    """True when the last `need` band-cell checkpoints each grew by less
    than `tol` of the series' own peak.

    Self-measured and scale-free: a wall whose band still gains 5% of its
    peak per minute is not saturated, however long the pin clock has run.
    Returns False while the history is too short to judge — an arm that
    fires on one checkpoint is arming on noise.
    """
    hist = [float(h) for h in history]
    if len(hist) < need + 1:
        return False
    peak = max(hist)
    if peak <= 0:
        return False
    window = hist[-(need + 1):]
    return all((b - a) < tol * peak for a, b in zip(window, window[1:]))


def gate_counters() -> dict:
    """The gate arm's telemetry counters, at their start values.

    A FACTORY, not a literal in __init__: the progress line, the sweep
    receipt and every duck-typed test solver read this same dict, so a
    counter added in one place and missed in another is a KeyError in a
    live run rather than a gap in a fixture.
    """
    return {
        "sweeps": 0, "programs": 0, "candidates": 0, "admitted": 0,
        "cross": 0, "injections": 0, "inexpressible": 0, "steps": 0,
        # K1's null. None = NEVER MEASURED, and it starts there: a 0.0
        # default reads as "the sham roots found nothing, the ranking is
        # clean" to the kill that consumes it, and the sweep can be
        # structurally unable to draw a sham root at all (the region
        # below the band floor is empty on a one-column target — Bubble
        # Bobble r99_1, spatial_span 1). The two counts behind the ratio
        # ride beside it so a reader never has to trust the quotient.
        "sham_yield": None, "sham_roots": 0, "sham_hits": 0,
        "wall_hits": 0, "lift_active": 0, "lift_ctrl": 0,
        # Workers the sweep could not hand back intact (see _gate_sweep's
        # finally block). Nonzero voids the run's A/B comparability, so
        # it rides in the progress line rather than only in stderr.
        "restore_failed": 0,
        # Undirected baseline windows closed early by a death. Large
        # relative to `programs` means the prior and the K4 screen are
        # being measured over a fraction of the horizon they were
        # budgeted for, which is a fact about the roots.
        "baseline_truncated": 0,
        # Undirected frames REFUSED as replays of a trajectory this run
        # already folded (see _gate_baseline_len). Receipted because it
        # is the size of the pseudo-replication the K4 floor used to be
        # paid in: on a 120-pattern basis it is ~97% of every frame the
        # sweep steps through undirected.
        "baseline_dupes": 0,
    }


#: "not sampled yet", distinct from a liveness reading of None (which is
#: what a profile that declares no lives byte returns on every frame).
_UNSET = object()

#: How many roots' last undirected frame are kept for change PAIRING
#: (2 KB each). Only the pairing needs the frame; the fold ledger that
#: refuses replays is integers and is never evicted, so an eviction can
#: cost one pair and can never re-admit a duplicate.
GATE_BASELINE_CACHE = 64


def _first_rank_by_addr(rows) -> dict:
    """{"0xNNNN": 1-based rank of that address's BEST row}.

    D-TRUNC's minimum: with this on the receipt, "the answer ranked 137th
    of 2,135" and "the answer never ranked" are different readings of the
    same artifact. Keyed by address because that is what a grade and the
    sidecar both speak; the row that carried it is in the full ranked
    table beside it.
    """
    out: dict = {}
    for i, r in enumerate(rows, start=1):
        key = "0x%04X" % int(r["addr"])
        if key not in out:
            out[key] = i
    return out


def gate_axes_sidecar_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def merge_gate_axes(profile: dict, path, contact_bits: int = 3) -> tuple:
    """Merge a --gate-axes sidecar into `profile['solve']['state_sig']`.

    APPENDED LAST, always: a resumed archive's cells were keyed with the
    bit indices the previous run used, so inserting an axis anywhere but
    the end silently re-numbers every existing bit.

    REFUSALS (all fatal, none silent): a sidecar entry without a
    probe-receipt sha is refused at merge, so no agent can hand-inject an
    address that no probe ever measured; more than `contact_bits` entries
    is refused; a merged signature longer than 8 bits is refused; and a
    profile with no `solve:` section is refused rather than being
    promoted to the generic adapter as a side effect of the merge.

    Returns (entries, sha) — ([], None) when no sidecar was given, so the
    caller's byte-identical default path is one falsy check away.
    """
    if not path:
        return [], None
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except Exception as exc:                       # noqa: BLE001
        raise SystemExit(f"[gate] unreadable --gate-axes {p}: {exc}")
    entries = raw.get("axes", []) if isinstance(raw, dict) else list(raw)
    if "solve" not in profile:
        raise SystemExit(
            f"[gate] --gate-axes {p} needs a profile with a `solve:` "
            f"section (state_sig lives there); this profile has none.")
    out = []
    for i, e in enumerate(entries):
        sha = str(e.get("receipt_sha") or e.get("probe_receipt_sha") or "")
        if not sha:
            raise SystemExit(
                f"[gate] REFUSED --gate-axes {p} entry {i} "
                f"(addr {e.get('addr')!r}): no probe-receipt sha. An axis "
                f"nothing measured is an injected address, not a finding.")
        if "addr" not in e:
            raise SystemExit(f"[gate] REFUSED --gate-axes {p} entry {i}: "
                             f"no addr")
        out.append({"addr": int(e["addr"]),
                    "match": [int(v) for v in e.get("match", ())],
                    "mod": int(e.get("mod", 0)),
                    "receipt_sha": sha})
    if len(out) > int(contact_bits):
        raise SystemExit(
            f"[gate] REFUSED --gate-axes {p}: {len(out)} axes over the "
            f"--contact-bits cap {int(contact_bits)}")
    sig = list(profile["solve"].get("state_sig", ()))
    merged = sig + [{"addr": e["addr"], "match": e["match"], "mod": e["mod"]}
                    for e in out]
    if len(merged) > 8:
        raise SystemExit(
            f"[gate] REFUSED --gate-axes {p}: merged state_sig would be "
            f"{len(merged)} bits (cap 8)")
    profile["solve"]["state_sig"] = merged
    return out, gate_axes_sidecar_sha(p)


def macro_slot_owner(in_flight: bool, gate_ready: bool,
                     transition_ready: bool, hold_ready: bool):
    """Who gets the ONE shared macro slot this step.

    Precedence at SLOT ACQUISITION only — gate > transition > hold — and
    an in-flight macro is never preempted, so Castlevania's declared
    up-hold keeps running to completion once it has started.

    explore() implements this INLINE (three branches in this order, each
    guarded by the same `macro_left <= 0`) so the hot loop stays
    call-free; this is the statement of the rule those branches are
    tested against, and the only place it is written down once.
    """
    if in_flight:
        return "in_flight"
    if gate_ready:
        return "gate"
    if transition_ready:
        return "transition"
    if hold_ready:
        return "hold"
    return None


def gate_suppress_trace(trace, marks, kind: str = "pattern"):
    """T2's DETERMINISTIC ABLATION: the same trace with the injected
    pattern frames replaced by NOOP.

    `marks` are the run's own gate_marks — (step, cand, kind, span)
    tuples recorded at injection time — so the ablation masks exactly the
    frames the arm claims responsibility for, and nothing else.

    SPAN IS THE LOAD-BEARING FIELD. An injected program owns
    macro_hold + 6 frames (six settle NOOPs, then the hold), not one; a
    one-frame mask on a 108-step hold leaves 113 of the 114 injected
    frames in place, so the "ablated" trace still contains the whole
    interaction and T2's necessity test compares a program against
    itself. Marks from before the field existed carry span 1 and are
    honoured as written rather than guessed at.

    Returns a NEW list; the input trace is never mutated (it is shared
    with the archive)."""
    out = list(trace)
    for m in marks:
        step, k = int(m[0]), (m[2] if len(m) > 2 else kind)
        span = max(1, int(m[3])) if len(m) > 3 else 1
        if k != kind:
            continue
        for s in range(max(0, step), min(step + span, len(out))):
            out[s] = NOOP
    return out


def gate_run_header(args, *, commit, hw_flags, root_sha, sidecar_sha,
                    axes, active_predicate: str) -> dict:
    """The A/B header every gate receipt is judged against. Absent any
    one of these fields the arms are not comparable after the fact, which
    is the entire failure mode §8-iii exists to prevent."""
    return {
        "argv": list(sys.argv),
        "commit": commit,
        "seed": int(getattr(args, "seed", 0)),
        "workers": int(getattr(args, "workers", 0)),
        "burst": int(getattr(args, "burst", 0)),
        "hw_flags": list(hw_flags),
        "root_sha": root_sha,
        "sidecar_sha": sidecar_sha,
        "gate_axes": list(axes),
        "gate_opener": str(getattr(args, "gate_opener", "off")),
        "gate_pin_secs": resolve_gate_pin_secs(args),
        "gate_target_typed": bool(getattr(args, "gate_target_typed", False)),
        "gate_band": int(getattr(args, "gate_band", 24)),
        # The arming floor, DERIVED and stated rather than left implicit
        # in a logging cadence (repair D-ARM).
        "gate_arm_cadence_secs": float(
            getattr(args, "gate_arm_cadence_secs", 60.0) or 60.0),
        "gate_arm_floor_secs": gate_arm_floor_secs(
            getattr(args, "gate_arm_cadence_secs", 60.0) or 60.0),
        # The arbitrary half of the ranking's total order. Seeded, so it
        # is reproducible and so two seeds are a real replication of the
        # ranking; named here so no citation of a rank can omit it.
        "rank_tie_seed": int(getattr(args, "seed", 0)),
        "rank_sort_key": ("score desc, effect_size desc, addr_families "
                          "desc, seeded tie_key, (addr, family) — the "
                          "address is never a tie-break (D-TIE)"),
        # Where the prior and the K4 screen are measured. Both samplers
        # are named because the graded run's failure was that only the
        # first existed and it sampled a population disjoint from the
        # sweep's roots (D-PRIOR).
        "prior_and_farm_sources": {
            "live_band": {"stride": ib.BAND_SAMPLE_STRIDE,
                          "population": "in-band cells the search re-enters"},
            "sweep_undirected": {
                "stride": ib.SWEEP_SAMPLE_STRIDE,
                "population": ("frames of a sweep program before it "
                               "presses anything: the settle prefix of "
                               "every program and the whole body of the "
                               "ladder's all-NOOP controls")},
            "pooled_in": "steps",
            "min_steps": ib.FARM_MIN_STEPS,
        },
        # The control-pairing rule, in full: duration matching (which
        # window a control may speak for) AND liveness (whether it may
        # speak at all).
        "control_pairing": {
            "duration": "matched on (settle phase, pattern length)",
            "liveness": ("an observation whose profile-declared lives "
                         "reading changes inside its own differential "
                         "window is excluded as control and as candidate; "
                         "a root whose controls all died is dropped whole; "
                         "a root that ran no control is unchanged"),
        },
        # The contact admission, in full: the observable, its two
        # constants, and the WINDOW those constants were applied over.
        # Without the window the same eps/K describe several different
        # tests (settle-only, settle slid forward by a frame, settle plus
        # a closing frame), and two arms that ran different ones would
        # produce headers that agree.
        "contact": {"observable": "profile gx/y telemetry",
                    "eps": ib.CONTACT_EPS, "k": ib.CONTACT_K,
                    "window": ("last k+1 settle samples, closed by the "
                               "pattern's first frame; extends into that "
                               "frame only when settle alone cannot "
                               "supply k+1")},
        "active_predicate": active_predicate,
        "loadavg": list(os.getloadavg()),
    }


def _git_commit() -> str | None:
    """Current commit, read straight off .git (no subprocess in a hot
    script, and no failure when the tree is not a checkout)."""
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = REPO / ".git" / head[5:]
            if ref.exists():
                return ref.read_text().strip()[:12]
            packed = (REPO / ".git" / "packed-refs").read_text().splitlines()
            for line in packed:
                if line.endswith(head[5:]):
                    return line.split()[0][:12]
            return None
        return head[:12]
    except Exception:                              # noqa: BLE001
        return None


class Solver:
    def __init__(self, args) -> None:
        # Cell granularity globals are consumed by cell_fn at call time.
        # main() sets them for subprocess runs; IN-PROCESS constructions
        # (the live show) previously bypassed that, silently ignoring
        # args.gx_bucket/y_band — the escalation ladder's micro-search
        # arm would have been a no-op. Apply here so both paths agree
        # (subprocess path re-applies the same values: no change).
        global GX_BUCKET, Y_BAND
        GX_BUCKET = int(getattr(args, "gx_bucket", GX_BUCKET))
        Y_BAND = int(getattr(args, "y_band", Y_BAND))
        self.args = args
        self.out = Path(args.out)
        (self.out / "solutions").mkdir(parents=True, exist_ok=True)
        profile = yaml.safe_load(Path(args.profile).read_text())
        # CLEAR-REACHABILITY PRE-FLIGHT (2026-08-26). Runs before the pool
        # exists, like every other profile check here, so a bad profile
        # costs zero emulator seconds.
        #
        # Two outcomes, and the SECOND one is the reason this exists.
        # `enforce` aborts a run whose declared clear machinery provably
        # cannot fire (Gradius shipped exactly that from 2026-08-24 to
        # 2026-08-26). But the far commoner case is a profile with NO clear
        # predicate at all — 37 of 45 today — where `solutions` in
        # progress.jsonl is a compile-time 0 and `--want-solutions` is
        # inert. That is a legitimate coverage baseline and is allowed to
        # run; what is NOT allowed is letting it run silently, because ten
        # separate documents went on to cite that constant as though a
        # search had asked a question and got "no"
        # (docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md). The
        # banner puts the disclaimer in the run's own log, where every
        # receipt is written from.
        _reach = clear_reachability.enforce(profile, str(args.profile))
        _banner = clear_reachability.launch_banner(profile, str(args.profile))
        if _banner:
            print(_banner, flush=True)
        self.clear_reachability = _reach
        self.bitmasks = action_space_to_bitmasks(profile["action_space"])
        self.weights = np.array(action_weights(profile["action_space"]))
        self.weights /= self.weights.sum()
        self.inv_weights = np.array(inverted_weights(profile["action_space"]))
        self.inv_weights /= self.inv_weights.sum()
        # GATE-OPENER AXIS SIDECAR — merged HERE, after the profile is
        # loaded and BEFORE make_game reads state_sig, because the
        # adapter snapshots that list once in __init__ and consumes it
        # per cell_fn call: there is no runtime admission path, by
        # design. Independent of --gate-opener so the CONTROL arm can
        # (and must) load the same sidecar — otherwise the two arms
        # partition cells differently and every A/B metric dies.
        self.gate_axes, self.gate_axes_sha = merge_gate_axes(
            profile, getattr(args, "gate_axes", None),
            int(getattr(args, "contact_bits", 3)))
        if self.gate_axes:
            print(f"[gate] merged {len(self.gate_axes)} axis/axes from "
                  f"{args.gate_axes} (sha {self.gate_axes_sha}): "
                  f"state_sig={profile['solve']['state_sig']}", flush=True)
        self.game = make_game(profile)
        # Hardware-flag selection (opt-in; empty by default, so a run
        # with neither --hw-flags nor solve.hw_flags: is bit-identical
        # to every pre-existing seeded solve). Resolved BEFORE the pool
        # exists so a bad name fails before any work is done.
        self.hw_flags = resolve_hw_flags(profile, getattr(args, "hw_flags", None))
        self.frame_skip = int(profile.get("frame_skip", 4))
        self.pool = Pool(rom_path=self.game.rom, num_workers=args.workers,
                         frame_skip=self.frame_skip)
        # MUST precede reset_all() — see apply_hw_flags' docstring.
        apply_hw_flags(self.pool, self.hw_flags)
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        # `solve.room_fp` FORCES the odometer on even for RAM-progress
        # profiles (Zelda): the rendered-lines gate, the Δodo classifier
        # and the scene evidence all read it (§3 row 2). Nothing else
        # changes for such a profile — progress still comes from its
        # declared RAM bytes; only the pseudo-RAM extension appears.
        self.room_fp = getattr(self.game, "room_fp", None)
        self._odo = (getattr(self.game, "odometer_axis", None) is not None
                     or self.room_fp is not None)
        if self._odo:
            if not hasattr(self.pool, "set_odometer_enabled"):
                raise SystemExit(
                    "[go_explore_solve] progress.source: odometer (or "
                    "solve.room_fp, which needs it) but this nes_core "
                    "build has no odometer binding — rebuild "
                    "(make build) before solving with this profile.")
            self.pool.set_odometer_enabled(True)
            if getattr(self.game, "odometer_axis", None) is not None:
                print(f"[go_explore_solve] progress source: PPU scroll "
                      f"odometer (axis {self.game.odometer_axis})",
                      flush=True)
        if self.room_fp is not None:
            if not (hasattr(self.pool, "peek_nametables")
                    and hasattr(self.pool, "odo_debug")):
                raise SystemExit(
                    "[go_explore_solve] solve.room_fp needs the "
                    "peek_nametables + odo_debug bindings — rebuild "
                    "nes_core (make build) and refresh the venv .so "
                    "(maturin does not update it in place) before "
                    "solving with this profile.")
            if (self.room_fp["palette_cokey"]
                    and not hasattr(self.pool, "palette_ram")):
                raise SystemExit(
                    "[go_explore_solve] room_fp.palette_cokey: true but "
                    "this nes_core build has no palette_ram binding — "
                    "rebuild (make build) before solving with this "
                    "profile.")
            print(f"[go_explore_solve] room fingerprinting ON: "
                  f"settle={self.room_fp['settle']} "
                  f"min_lines={self.room_fp['min_lines']} "
                  f"max_rooms={self.room_fp['max_rooms']} "
                  f"config_sha={self.game.room_fp_sha}", flush=True)
        self._odo_now: list = []
        self._odo_scene: list = []
        # FIGHT-GATE cumulative-damage integral (§4.1). Unlike the
        # odometer, whose camera-integral state lives INSIDE nes_core and
        # round-trips through load_worker_state for free, this total is
        # Python-side per-worker state: `_fight_prev_hp`/`_fight_cum` are
        # reset only here and on every `_assign()` load (see there), never
        # re-derived from an archived cell's own history. That is correct
        # for the pre-registered single-worker validation smoke
        # (FIGHTGATE_MECHANISM_2026-08-25.md §5.2, a continuous root-to-
        # entrance lineage) and for a fresh worker seeded at the entrance;
        # it is a KNOWN, documented gap for the general multi-worker
        # archive case, where a Go-Explore restore to an ARCHIVED cell
        # (as opposed to a fresh entrance load) does not yet re-derive the
        # total from that cell's own banked damage — future work, not
        # silently pretended away.
        self._fight = getattr(self.game, "foe_hp_addr", None) is not None
        nw = int(args.workers)
        self._fight_prev_hp: list = [None] * nw
        self._fight_cum = np.zeros(nw, dtype=np.uint32)
        self._fight_round_prev: list = [None] * nw
        self._fight_prev_ram: list = [None] * nw
        if self._fight:
            rd = getattr(self.game, "fight_round_addr", None)
            tail = (f"round=0x{rd:04X}" if rd is not None
                    else "no round byte -> mass-reset fallback")
            print(f"[go_explore_solve] progress source: fight-gate "
                  f"cumulative damage (foe_hp=0x{self.game.foe_hp_addr:04X}, "
                  f"{tail})", flush=True)
        self.pool.reset_all()
        self.provenance = hw_provenance(self.hw_flags, self.frame_skip)
        if self.hw_flags:
            print(f"[go_explore_solve] hw flags: {self.hw_flags}", flush=True)
        # AUDIO MODALITY (v3, opt-in). Resolved ONCE: the sample is a
        # per-step pool call, and `hasattr` per step on the hot loop is not
        # free. False unless the profile's clear hook asked for it AND the
        # linked core actually exports the accessor (an older wheel must
        # degrade to the RAM-only detector with a loud line, not AttributeError
        # every step).
        self._needs_apu = resolve_apu_sampling(self.game, self.pool)
        # Diagnostic only: a windowed clear hook whose warm-up exceeds one
        # burst can never fire live, because the ctx holding its state is
        # rebuilt at every re-root.
        warn_if_burst_starves_clear_hook(self.game,
                                         int(getattr(args, "burst", 0) or 0))
        self.rng = np.random.default_rng(args.seed)
        # Sustained-hold macros (generic mechanism, profile-selected): at a
        # small per-step probability a worker settles briefly then HOLDS one
        # input for N steps — producing the long consecutive holds that
        # stochastic sampling almost never emits (stair mounts, pipe
        # entries). The 8-4 pipe-entry macro was this mechanism hardcoded;
        # macro steps are recorded verbatim in traces like any other action.
        self.macros = []          # (action_idx, hold_steps, weight)
        self.macro_p = 0.0
        for m in profile.get("solve", {}).get("hold_macros", []):
            want = set(m["buttons"])
            idx = next((i for i, c in enumerate(profile["action_space"])
                        if set(c) == want), None)
            if idx is None:
                print(f"[solver] hold_macro {m['buttons']} not in "
                      f"action_space — skipped", flush=True)
                continue
            p = float(m.get("p", 0.02))
            self.macros.append((idx, int(m.get("steps", 20)), p))
            self.macro_p += p
        if self.macros:
            w = np.array([m[2] for m in self.macros])
            self._macro_weights = w / w.sum()
        # Discrete-transition gate (the Kirby lesson). When solve.room_advance
        # is configured, auto-derive door/room-entry maneuver macros from the
        # action space (derive_transition_macros) and inject them with elevated
        # probability ONLY when a worker sits at the deep frontier — where the
        # coordinate frontier has stalled and the next progress is a discrete
        # 'grounded + UP' door entry, not another step rightward. Reuses the
        # hold_macros settle-then-hold machinery; empty (inert) for every SMB /
        # non-room profile, so the default path is byte-identical.
        ra = (profile.get("solve", {}) or {}).get("room_advance")
        self.transition_macros = derive_transition_macros(
            profile["action_space"], ra)
        self.room_advance_addr = (int(ra["addr"]) if ra and "addr" in ra
                                  else None)
        self.transition_p = float(ra.get("p", 0.05)) if ra else 0.0
        self.transition_near = int(ra.get("near", 24)) if ra else 24
        self._transition_injections = 0   # telemetry: door macros injected
        self.max_room = 0                 # telemetry: deepest room_advance byte
        if ra and not self.transition_macros:
            print("[solver] room_advance configured but action_space has no "
                  "'up' maneuver — discrete-transition gate inert", flush=True)
        self.archive = GoExploreArchive(self.game.cell_fn, seed=args.seed)
        self.traces: dict = {}            # cell key -> (root_id, trace bytes)
        self.roots: dict = {}             # root_id -> {path, start_wd, lives}
        self.start_wd = (0, 0)
        self.start_lives = 0
        # Progress = (area order, gx). We track the DEEPEST area reached and
        # the max gx within it, so the deep-frontier bias follows Mario into
        # the underground instead of pinning on the entrance area.
        self.max_area = 0
        self.max_gx_in_area: dict = {}    # area -> max gx seen
        # {(area, gx bucket)} the resume refreeze refused as banked torn
        # reads (§12's gx-767 phantom). Empty on every non-resumed run,
        # so the default path is untouched; both frontier readers
        # (max_gx_in_area, _sel_topgx) skip what is in it.
        self._gx_phantoms: set = set()
        self.resume_refreeze = bool(
            getattr(args, "resume_refreeze", True))
        self._pin_time = time.time()      # last frontier advance (inversion gate)
        # Resolved once (the gate is read per worker per step).
        self._inv_pin_secs = resolve_inversion_pin_secs(args)
        self._loop_dest_min = None        # min gx observed right after a loop
        self.max_sect = 0                 # deepest section-transit count seen
        self.n_solutions = 0
        self.sol_counter = 0
        self.best_sol_len = 10 ** 9
        # REPLAY-VERIFIED BANKING (v2, 2026-08-08). Default ON — measured
        # cost is ~2.7 ms per action of one candidate (see replay_verify),
        # i.e. seconds against a 25-45 minute solve. Absent attribute (the
        # live show's in-process SimpleNamespace) also resolves ON: banking
        # trust is not something an older call site should silently opt out
        # of. --no-verify-bank is the explicit escape.
        self.verify_bank = resolve_verify_bank(args)
        self.verify_checks = 0
        self.verify_rejections = 0
        # COUNTERFACTUAL GATE (v3, 2026-08-09) — default OFF, see
        # resolve_counterfactual_gate for why this one does NOT inherit the
        # verify_bank "absent means on" rule.
        self.cf_gate = resolve_counterfactual_gate(args)
        self.cf_branches = int(getattr(args, "cf_branches", 8))
        self.cf_pre_steps = int(getattr(args, "cf_pre_steps", 32))
        self.cf_perturb_p = float(getattr(args, "cf_perturb_p", 0.25))
        self.cf_agree = float(getattr(args, "cf_agree", 0.5))
        # Seeded off the run's own seed so two runs of the same invocation
        # probe identically and a receipt's verdict is reproducible.
        _cfs = getattr(args, "cf_seed", None)
        self.cf_seed = int(getattr(args, "seed", 0) if _cfs is None else _cfs)
        self.cf_checks = 0
        self.cf_rejections = 0
        # Split out of cf_rejections (2026-08-10): a candidate refused as a
        # STATE ARTIFACT is not a contingent clear that lost a vote, it is a
        # clear hook firing on its own accumulated state — the number a
        # detector-safety review actually wants, and the one that says the
        # profile's hook needs explaining rather than the search needs luck.
        self.cf_state_artifacts = 0
        self.cf_retries = 0        # deep re-snapshots actually run
        self.steps_done = 0
        self.t0 = time.time()
        self.stop = False
        # Maze-coverage recipes (research round 2), flag-gated so the
        # default path stays byte-identical to the receipted campaign:
        # sel_mode "count" swaps the uniform arm for Go-Explore's native
        # count-based prior W = 1/sqrt(times_chosen+1) * (score_norm +
        # 0.1) via O(1) rejection sampling (the archive's own weighted
        # selection was bypassed for being an O(N) scan — this restores
        # the prior without the scan). frontier_throttle N > 0 arms
        # DD-RRT-style boundary suppression: a cell whose bursts yield
        # nothing novel N times in a row is excluded from the deep-
        # frontier band (stop throwing bursts at the physical wall).
        self.sel_mode = str(getattr(args, "sel_mode", "legacy"))
        self.frontier_throttle = int(getattr(args, "frontier_throttle", 0))
        # R4 door discovery (research round 2): cut vertices of the archive
        # transition graph are the maze's doors — every route between the
        # regions they separate passes through them. Edges are recorded from
        # our own rollouts (cell-to-cell transitions, no game internals); an
        # async pass re-derives articulation points every door_interval and
        # the count arm up-weights door cells by door_weight. 0 = off, and
        # the default path stays byte-identical to the receipted campaign.
        self.door_weight = float(getattr(args, "door_weight", 0.0))
        self.door_interval = float(getattr(args, "door_interval", 45.0))
        # ORTHOGONAL-FRONTIER arm (CV hall diagnosis, 2026-08-08), flag-
        # gated, default off => byte-identical to the receipted SMB
        # lineage. The solver's only reward axis is x (score = sect*10000
        # + gx, and CV's sect never moves), so the deep arm samples only
        # the deepest gx buckets while every climbing cell sits at LOW gx:
        # a vertical frontier is unreachable by construction, and the
        # count arm's score term penalizes it a second time. When the x
        # frontier has been pinned for `ortho_pin_secs` (the same self-
        # measured _pin_time the inversion window uses), a third arm
        # samples the top-of-column cells of the orthogonal axis with a
        # PURE count prior — no score term, so the x-axis score cannot
        # suppress a climb. SELECTION-SIDE ONLY: `score` and the cell key
        # are untouched, so domination, solution receipts, cross-run
        # comparability and --resume-archive all keep working.
        self.ortho_mode = str(getattr(args, "ortho", "off"))
        self.ortho_pin_secs = float(getattr(args, "ortho_pin_secs", 120.0))
        self.ortho_bias = float(getattr(args, "ortho_bias", 0.30))
        self.ortho_band = int(getattr(args, "ortho_band", 1))
        self.ortho_weight = float(getattr(args, "ortho_weight", 4.0))
        self.ortho_macro_p = float(getattr(args, "ortho_macro_p", 0.0))
        self._ortho_pool: list = []       # cached top-of-column cells
        self._ortho_ids: set = set()      # their keys, for O(1) membership
        self._ortho_ext: dict = {}        # gx bucket -> column extreme
        self._ortho_best = None           # best y-band any worker reached
        self._ortho_time = time.time()    # last orthogonal improvement
        self._ortho_deep_yband = None     # extreme y-band AT the x frontier
        self._ortho_selections = 0
        self._ortho_cols_improved = 0
        # --- ROOM-GRAPH ENGINE (default off => byte-identical) --------
        # Identity + persistence live here (T1); the settle/classify hot
        # loop, edge commits and the router arm are wired separately
        # (T2/T3). All of it is inert unless the profile declares
        # `solve.room_fp` (self.room_fp, resolved above with the
        # odometer force); the router arm additionally needs
        # --room-bias > 0. Per-worker ordinal state is DERIVED, never
        # accumulated across restores (§2 lockstep invariants):
        # _assign seeds _room_ord from cell metadata, and _xram only
        # ever reads it.
        self.room_bias = float(getattr(args, "room_bias", 0.0))
        self.room_artic_weight = float(getattr(args, "room_artic_weight",
                                               2.0))
        self.room_exit_weight = float(getattr(args, "room_exit_weight",
                                              1.0))
        self.room_recent_k = int(getattr(args, "room_recent_k", 4))
        self.room_index = None
        self._room_mask = None
        self._room_ord = np.full(int(args.workers), ROOM_UNKNOWN,
                                 dtype=np.uint16)
        self._room_settle_rejects = 0     # telemetry: min_lines/cap holds
        self._room_router_picks = 0       # telemetry: router-arm picks
        self._room_route_injections = 0   # telemetry: route-follow macros
        # Router caches (T3), rebuilt by _refresh_sel_cache on its own
        # cadence — the same single scan, §3 row 7. Empty and never
        # touched unless BOTH the profile fingerprints and --room-bias
        # arms the arm.
        self._room_pools: dict = {}       # ordinal -> selectable cells
        self._room_sides: dict = {}       # ordinal -> {side: boundary cells}
        self._room_bounds: dict = {}      # ordinal -> cell-unit bbox
        self._room_U: dict = {}           # ordinal -> unexplored-side count
        self._room_V: dict = {}           # ordinal -> sum times_chosen
        self._room_out_dirs: dict = {}    # ordinal -> dirs with out-edges
        self._room_degree: dict = {}      # undirected adjacency degree
        self._room_artic: set = set()     # articulation ordinals (inline)
        self._room_aliased: set = set()   # aliasing-audit marks (monotone)
        self._room_dir_macros: dict = {}  # side -> [(action_idx, hold)]
        self._route_pick = None           # router arm -> _assign handoff
        self._room_edges_committed = 0    # transits that banked a staged edge
        self._room_edges_dropped = 0      # staged edges no transit consumed
        self._room_restore_transits = 0   # transits at burst step <= 1 (tripwire: 0)
        self._room_adoptions = 0          # first settles from ROOM_UNKNOWN
        # Where the ordinal bytes sit in a threaded psig tail (negative
        # offsets from the end + the room_sig arity), so _assign can
        # seed _room_ord from the restored cell's lineage (§2 lockstep
        # invariants). None when room_sig does not carry both bytes —
        # then every burst seeds ROOM_UNKNOWN and re-settles.
        self._room_psig_off = None
        if self.room_fp is not None:
            _rs = list(getattr(self.game, "_room_sig", ()) or ())
            if ROOM_LO in _rs and ROOM_HI in _rs:
                self._room_psig_off = (_rs.index(ROOM_LO) - len(_rs),
                                       _rs.index(ROOM_HI) - len(_rs),
                                       len(_rs))
            else:
                print(f"[go_explore_solve] room_fp is armed but "
                      f"solve.room_sig {_rs} does not carry both ordinal "
                      f"bytes (0x{ROOM_LO:03X}/0x{ROOM_HI:03X}): ordinal "
                      f"changes cannot fire sect/psig transits, every "
                      f"staged edge will be dropped and restores cannot "
                      f"seed — point room_sig at [0x804, 0x805] "
                      f"(ROOMGRAPH_ENGINE_2026-08-24 §4).", flush=True)
        # ITEM-SIG GRAFT (ITEM_SEMANTICS_ENGINE_2026-08-25 §3 row 5):
        # report-only — armed only when --item-sig-report is passed AND
        # the profile declares a state_sig (getattr, never a bare
        # attribute read: SmbGame carries no state_sig_arity at all,
        # and args.item_sig_report defaults False, so this is a no-op
        # on every existing profile/flags-off path).
        self._item_sig_armed = (bool(getattr(args, "item_sig_report", False))
                                and getattr(self.game, "state_sig_arity",
                                           0) > 0)
        if self.room_fp is not None:
            self._room_mask = room_fp_mask(self.room_fp["mask"])
            self.room_index = RoomIndex(cap=self.room_fp["max_rooms"],
                                        config_sha=self.game.room_fp_sha)
            # Route-follow macros (§3 row 9): structural, from the
            # action space alone — the same derivation family as the
            # room_advance door macros, one list per room side. The
            # hold length rides room_advance's own `steps` knob.
            self._room_dir_macros = derive_direction_macros(
                profile["action_space"],
                int(ra.get("steps", 20)) if ra else 20)
        # --- GATE-OPENER ARM (default off => byte-identical) ----------
        # Every knob comes off its own argparse dest; the mode alone is
        # what holds the arm down, and it is checked FIRST in every hot
        # path below so a disarmed run pays one string compare and draws
        # no randomness whatsoever.
        self.gate_mode = str(getattr(args, "gate_opener", "off"))
        self.gate_pin_secs = resolve_gate_pin_secs(args)
        self.gate_target_typed = bool(getattr(args, "gate_target_typed",
                                              False))
        self.gate_band = int(getattr(args, "gate_band", 24))
        self.gate_sweep_frac = float(getattr(args, "gate_sweep_frac", 0.10))
        self.gate_sweep_roots = int(getattr(args, "gate_sweep_roots", 16))
        self.gate_sweep_repeats = int(getattr(args, "gate_sweep_repeats", 2))
        self.gate_sham_roots = int(getattr(args, "gate_sham_roots", 4))
        # THE ARMING FLOOR, AS A PARAMETER (repair D-ARM). The band-growth
        # conjunct needs `need + 1` = 4 checkpoints before it can return
        # anything but False, and those checkpoints used to be minted by
        # the 60 s progress cadence alone — so no sweep could fire before
        # t = 240 s no matter how short the run was, and the K0 grades
        # measured arming at 240-300 s out of a ~15 min budget. The
        # cadence is now stated; the DEFAULT IS UNCHANGED at 60 s, so a
        # run that does not pass the flag checkpoints exactly as the
        # receipted ones did.
        self.gate_arm_cadence = float(
            getattr(args, "gate_arm_cadence_secs", 60.0) or 60.0)
        self.contact_bits = int(getattr(args, "contact_bits", 3))
        # Reserved multiplier for the count arm's exact Wmax. v1 never
        # up-weights (no flag declares one), so the prior is unchanged;
        # it is threaded through count_wmax now so the pair can never be
        # split later — the receipted failure mode is a multiplier that
        # ships without its ceiling.
        self.gate_weight = 1.0
        # The sweep's OWN stream. Seeded off the run seed so a sweep is
        # reproducible, but SEPARATE from self.rng so scheduling a sweep
        # cannot shift a single draw of the search.
        self._gate_rng = np.random.default_rng(
            (int(args.seed) ^ int.from_bytes(b"gate", "little"))
            & 0xFFFFFFFF)
        self._gate_basis = (ib.interaction_basis(profile["action_space"])
                            if self.gate_mode != "off" else [])
        self._gate_phases = ib.SETTLE_PHASES_PASS_A
        self._gate_swept: dict = {}       # cell key -> times swept
        self._gate_admitted: list = []    # admitted candidate addrs
        self._gate_shadow: dict = {}      # addr -> {(vbucket, yb, gxb)}
        self._gate_positions: set = set()  # in-band (yb, gxb) seen
        self._gate_band_hist: list = []   # band_cells at each checkpoint
        self._gate_marks_kind = "pattern"
        self._gate_obs_n = 0
        # The BH denominator the LAST sweep charged its wall and its
        # sham null against, kept so the receipt (a null one especially)
        # can state the bar rather than leave it to be re-derived.
        self._gate_fdr_m = None
        self._boundary_hist = None        # 2048x256 uint32, lazy (2 MB)
        self._boundary_hist_total = 0
        self._boundary_rows = None
        # K4's farmability input. The histogram above records a value
        # DISTRIBUTION and is structurally silent about how often a byte
        # CHANGES, which is the half of K4 that refuses a farmable axis
        # — so the change rate is counted here, off the same samples.
        self._gate_change = None          # per-addr change count, uint32
        self._gate_change_n = 0           # sample PAIRS behind it
        self._gate_prev_vals = None
        # ...AND THE SWEEP'S OWN COPY OF BOTH (repair D-PRIOR). The two
        # counters above are fed by the LIVE search's in-band sampling,
        # and the sweep roots in the band tail a pinned search essentially
        # never re-enters: on the graded Castlevania run they stayed at
        # zero for 24 minutes, novelty defaulted to 1.0 for every row and
        # the admissible list was empty by construction. These are fed by
        # the sweep's own UNDIRECTED frames — every frame of a program in
        # which nothing has been pressed yet — at stride 1, and the two
        # sources are pooled in step units by farm_rate_pooled.
        self._gate_change_sweep = None    # per-addr change count, uint32
        self._gate_change_sweep_n = 0     # consecutive frame PAIRS (stride 1)
        self._gate_baseline_frames = 0    # DISTINCT undirected frames folded
        # ...AND THE DE-REPLICATOR THAT MAKES THOSE COUNTS HONEST. Every
        # program at one root replays the SAME all-NOOP trajectory until
        # its first press: the settle prefixes of all 120 patterns x 2
        # phases x N repeats are byte-identical, and the six all-NOOP
        # controls are that trajectory in full. Counting each replay as
        # fresh evidence turned ~2k distinct frames into ~78k "observed
        # steps" and walked the K4 refusal floor (FARM_MIN_STEPS) with
        # pseudo-replication, so a slow mover could read 0.0 events/1k as
        # MEASURED off a 21-frame horizon. Frames are therefore folded at
        # most once per (root trajectory, frame index).
        self._gate_baseline_len: dict = {}   # root sig -> frames folded
        self._gate_baseline_prev: dict = {}  # root sig -> last folded frame
        self._gate_root_sig: dict = {}       # cell key -> root blob digest
        # Arm C's queue: (action index, hold, candidate addr), filled at
        # admission, drained round-robin at burst assignment.
        self._gate_inject: list = []
        self._gate_inject_i = 0
        self._gate_boundary_hit = False   # a burst just ended this step
        self._gate_next_sweep = 0
        self._gate_last_pin = self._pin_time
        self._gate_disarmed = False       # set on admission; fresh pin clears
        self._gate_armed_secs = 0.0
        self._gate_armed_since = None
        # Band-growth checkpoint clock, started HERE rather than at 0.0 so
        # the first checkpoint lands one full cadence into the run, which
        # is exactly where the old progress-line-driven one landed.
        self._gate_last_ckpt = time.time()
        self._gate_band_last = 0
        # What the last differential DECLINED to read (dead controls,
        # dead patterns, roots left with no live baseline). Receipted, so
        # a shrinking candidate pool is a reported fact rather than an
        # invisible one.
        self._gate_rank_stats: dict = {}
        self._gate_counters = gate_counters()
        self._gate_axes_live = None       # boundary_axis_profile snapshot
        # v6-report recipes (2026-07-30), flag-gated, default byte-identical:
        # time_bins: append floor(log2(steps+1)) to the key prefix (after
        # sect, so key[0]/key[-N] indexing is untouched). Time-Myopic
        # Go-Explore: slow structural trajectories (stairs, waits) stop
        # losing the fewer-steps domination tie-break to drive-by states,
        # and wait-gated triggers become explorable at archive scale.
        # kill_key: append a capped cumulative entity-kill count (observed
        # 1->0 transitions of profile-configured entity-slot flags) —
        # prerequisite/kill-count gates become a searchable dimension.
        self.time_bins = bool(getattr(args, "time_bins", False))
        self.kill_key = bool(getattr(args, "kill_key", False))
        self.kill_key_local = bool(
            (profile.get("solve", {}) or {}).get("kill_key_local", False))
        eslots = (profile.get("solve", {}) or {}).get("entity_slots")
        # Optional stride: object-array engines (Contra: state bytes at
        # $0368 + k*$20) space their per-entity flags; contiguous ranges
        # (CV: $0450..$0457) are stride 1, the default.
        self.entity_slots = (range(int(eslots["lo"]), int(eslots["hi"]),
                                   int(eslots.get("stride", 1)))
                             if eslots else None)
        self._key_ids: dict = {}          # cell key -> int id (interned)
        self._adj: dict = {}              # id -> set of neighbor ids
        self._doors: frozenset = frozenset()
        self._door_lock = threading.Lock()
        self._door_thread: threading.Thread | None = None
        self._last_door_t = time.time()
        self._flush_thread: threading.Thread | None = None
        self._recorded_new = False
        # Torn multi-byte progress reads (see progress_glitch). Resolved
        # ONCE off the adapter so observe()'s hot path is one string
        # compare, and defaulted so an adapter without the knob (SmbGame,
        # every duck-typed stand-in) is the shipped behaviour.
        self.progress_smooth = str(
            getattr(self.game, "progress_smooth", "off") or "off")
        self.progress_jump = int(
            getattr(self.game, "progress_jump", PROGRESS_JUMP))
        self._prog_glitches = 0
        # ATOMIC PEEK preflight (see GenericGame.progress_atomic_read):
        # fail at construction, not on the first observation, when the
        # profile asks for `progress.atomic` but the loaded nes_core
        # build predates peek_u16_consistent.
        if (getattr(self.game, "progress_atomic", False)
                and not hasattr(self.pool, "peek_u16_consistent")):
            raise SystemExit(
                "[go_explore_solve] progress.atomic: true but this "
                "nes_core build has no peek_u16_consistent binding — "
                "rebuild it (`make build`) and refresh the venv .so "
                "(maturin does not update it in place) before using "
                "this flag.")
        # THE BORROW RULE AUDITS ITSELF (see borrow_followup). It assumes
        # `hi` is `lo`'s page byte, which is true of Castlevania's
        # $0040/$0041 and false of the six (screen, x) composites in
        # configs/ — and the profile cannot be trusted to know which it
        # declared. Every rejection is checked against the NEXT frame,
        # and the rule disarms itself if the refutations win.
        self._borrow_tears = 0
        self._borrow_flat = 0
        self._borrow_off = False
        # c_local bookkeeping: the archive's distinct (area, y_band,
        # gx_bucket) footprint, maintained incrementally on the record
        # path (a full re-scan at the 60 s cadence would be a 560k-key
        # pass on a hall archive; this is one tuple per NEW cell).
        self._c_local: set = set()
        self._c_local_prev: tuple | None = None
        # Optional spectator hook: called every pool step with
        # (results, solver) — the FULL per-worker results list, so the
        # live show can render one worker or the whole swarm. None in
        # headless runs (zero overhead).
        self.step_hook = None

    def _odo_refresh(self, pool=None) -> None:
        """Snapshot every worker's odometer straight after a step_all
        (state read, no side effects) so _xram can extend RAM without a
        per-worker Rust round-trip."""
        if self._odo:
            src = pool or self.pool
            self._odo_now = src.get_odometer_per_worker()
            self._odo_scene = src.get_odometer_scene_per_worker()

    def _fight_step(self, base: np.ndarray, wid: int) -> int:
        """Advance the fight-gate cumulative-damage integral for one
        worker by one step, from this call's raw (pre-extension) RAM
        snapshot. Owns per-worker state indexing and transition
        detection only — the actual accounting is the pure
        `fight_gate_step` helper (tests/test_fight_gate.py exercises it
        directly with synthetic traces)."""
        now = int(base[self.game.foe_hp_addr])
        round_addr = self.game.fight_round_addr
        if round_addr is not None:
            r_now = int(base[round_addr])
            r_prev = self._fight_round_prev[wid]
            self._fight_round_prev[wid] = r_now
            is_transition = r_prev is not None and r_now != r_prev
        else:
            is_transition = fight_gate_mass_reset(
                self._fight_prev_ram[wid], base,
                threshold=FIGHT_RESET_THRESHOLD)
            self._fight_prev_ram[wid] = np.array(base, copy=True)
        cum, prev = fight_gate_step(
            self._fight_prev_hp[wid], now, is_transition=is_transition,
            cum=int(self._fight_cum[wid]))
        self._fight_prev_hp[wid] = prev
        self._fight_cum[wid] = cum
        return cum

    def _xram(self, ram, wid: int):
        """Extend one worker's 2KB RAM snapshot with the scroll-odometer
        integral at ODO_LO..ODO_LO+2 (little-endian, clamped to 24 bits;
        backward-of-origin clamps to 0 — regress is not negative
        progress, it is no progress) and/or the fight-gate cumulative-
        damage integral at FIGHT_LO/FIGHT_HI (§4.1). No-op when the
        profile uses neither, so every other solve keeps zero overhead
        and its exact snapshot identity.

        With `solve.room_fp` the extension carries 3 more bytes: the
        settled room ordinal (uint16 LE) at ROOM_LO/ROOM_HI and the
        ORTHOGONAL odometer axis at ODO_ALT ((v>>4)&0xFF). With
        `progress.source: fight_gate` the extension is ALWAYS the full
        9 bytes (0x800..0x808), independent of odometer/room_fp,
        because FIGHT_LO/HI are fixed absolute addresses past the whole
        odometer/room-fp block — any bytes in that block a fight_gate
        profile doesn't otherwise populate (no odometer axis, no
        room_fp) read as zero, never uninitialized. The length switch
        is keyed on room_fp/fight_gate presence alone, so every profile
        declaring neither — SMB included — keeps its exact 4-byte
        extension and snapshot identity (RG-1c control)."""
        fight = getattr(self, "_fight", False)
        if not (self._odo or fight):
            return ram
        base = (np.frombuffer(ram, dtype=np.uint8)
                if isinstance(ram, (bytes, bytearray)) else ram)
        rf = self.room_fp is not None
        ext_len = 9 if fight else (7 if rf else 4)
        ext = np.zeros(len(base) + ext_len, dtype=np.uint8)
        ext[:len(base)] = base
        if self._odo:
            x, y = self._odo_now[wid]
            axis = self.game.odometer_axis
            # axis is None only when room_fp forced the odometer on for a
            # RAM-progress profile; the primary slot then carries x and
            # the ODO_ALT byte y. For declared odometer profiles ("x"/
            # "y", validated at parse) this expression is byte-identical
            # to the pre-roomgraph `x if axis == "x" else y`.
            v = y if axis == "y" else x
            v *= getattr(self.game, "odometer_sign", 1)
            v = 0 if v < 0 else (0xFFFFFF if v > 0xFFFFFF else int(v))
            ext[ODO_LO] = v & 0xFF
            ext[ODO_LO + 1] = (v >> 8) & 0xFF
            ext[ODO_LO + 2] = (v >> 16) & 0xFF
            # Scene ordinal (mod 256) at ODO_LO+3: profiles point
            # solve.area here so cells key on (scene, within-scene x) —
            # camera-locked rooms and stage wipes become area
            # transitions, exactly the (area-order, gx) machinery the
            # solver already has.
            ext[ODO_LO + 3] = self._odo_scene[wid] & 0xFF
            if rf:
                o = int(self._room_ord[wid])
                ext[ROOM_LO] = o & 0xFF
                ext[ROOM_HI] = (o >> 8) & 0xFF
                alt = x if axis == "y" else y
                alt = 0 if alt < 0 else (0xFFFFFF if alt > 0xFFFFFF
                                         else int(alt))
                ext[ODO_ALT] = (alt >> 4) & 0xFF
        if fight:
            cum = self._fight_step(base, wid)
            ext[FIGHT_LO] = cum & 0xFF
            ext[FIGHT_HI] = (cum >> 8) & 0xFF
        return ext

    def _xram_local(self, ram, pool, hold=None):
        """_xram against a FRESH single-worker pool (replay paths). The
        replay pool's odometer was enabled by the v3 blob it loaded, so
        the read is coherent with what the live search saw.

        With `solve.room_fp` the room ordinal is RE-DERIVED by hashing
        the replay env's nametables against the FROZEN index — see
        _replay_room_ord; `hold` is the caller's per-replay dict (its
        clear-hook ctx) carrying the hold-last ordinal across steps. A
        replay whose hash diverges therefore composes different
        room_sig bytes and is marked UNVERIFIED by the existing
        comparison, never silently passed.

        With `progress.source: fight_gate`, the cumulative-damage
        integral is tracked in `hold` under `_fight_*` keys instead of a
        Solver-owned per-worker array — a replay is one continuous
        lineage with no restore-to-an-archived-cell step, so `hold`
        alone is enough state (mirrors `_replay_room_ord`'s use of it)."""
        fight = getattr(self, "_fight", False)
        base = (np.frombuffer(ram, dtype=np.uint8)
                if isinstance(ram, (bytes, bytearray)) else ram)
        rf = self.room_fp is not None
        ext_len = 9 if fight else (7 if rf else 4)
        ext = np.zeros(len(base) + ext_len, dtype=np.uint8)
        ext[:len(base)] = base
        if self._odo:
            x, y = pool.get_odometer_per_worker()[0]
            axis = self.game.odometer_axis
            v = y if axis == "y" else x
            v *= getattr(self.game, "odometer_sign", 1)
            v = 0 if v < 0 else (0xFFFFFF if v > 0xFFFFFF else int(v))
            ext[ODO_LO] = v & 0xFF
            ext[ODO_LO + 1] = (v >> 8) & 0xFF
            ext[ODO_LO + 2] = (v >> 16) & 0xFF
            ext[ODO_LO + 3] = pool.get_odometer_scene_per_worker()[0] & 0xFF
            if rf:
                o = self._replay_room_ord(pool, hold)
                ext[ROOM_LO] = o & 0xFF
                ext[ROOM_HI] = (o >> 8) & 0xFF
                alt = x if axis == "y" else y
                alt = 0 if alt < 0 else (0xFFFFFF if alt > 0xFFFFFF
                                         else int(alt))
                ext[ODO_ALT] = (alt >> 4) & 0xFF
        if fight:
            h = hold if hold is not None else {}
            now = int(base[self.game.foe_hp_addr])
            round_addr = self.game.fight_round_addr
            if round_addr is not None:
                r_now = int(base[round_addr])
                r_prev = h.get("_fight_round_prev")
                h["_fight_round_prev"] = r_now
                is_transition = r_prev is not None and r_now != r_prev
            else:
                is_transition = fight_gate_mass_reset(
                    h.get("_fight_prev_ram"), base,
                    threshold=FIGHT_RESET_THRESHOLD)
                h["_fight_prev_ram"] = np.array(base, copy=True)
            cum, prev = fight_gate_step(
                h.get("_fight_prev_hp"), now, is_transition=is_transition,
                cum=int(h.get("_fight_cum", 0)))
            h["_fight_prev_hp"] = prev
            h["_fight_cum"] = cum
            ext[FIGHT_LO] = cum & 0xFF
            ext[FIGHT_HI] = (cum >> 8) & 0xFF
        return ext

    def _replay_room_ord(self, pool, hold=None) -> int:
        """The room ordinal for a REPLAY pool frame, derived against the
        FROZEN index — lookup only, never intern: a replay must read
        the graph the live search built, not grow it.

        Blank frames (rendered lines under min_lines) and unknown
        hashes HOLD the last known ordinal (`hold` is the caller's
        per-replay dict; without one, ROOM_UNKNOWN). No settle loop
        here: mid-transition churn hashes are simply unknown and hold,
        and the first known settled-room hash snaps to its ordinal —
        the same value the live worker carried."""
        last = (ROOM_UNKNOWN if hold is None
                else int(hold.get("_room_ord", ROOM_UNKNOWN)))
        try:
            lines = pool.odo_debug(0)[2]
        except Exception:                          # noqa: BLE001
            return last
        if lines < self.room_fp["min_lines"]:
            return last
        pal = (bytes(pool.palette_ram(0))
               if self.room_fp["palette_cokey"] else None)
        h = nt_fingerprint(pool.peek_nametables(0), self._room_mask, pal)
        o = self.room_index.lookup(h) if self.room_index is not None else None
        if o is None:
            return last
        if hold is not None:
            hold["_room_ord"] = int(o)
        return int(o)

    # ---- room-fingerprint hot loop (T2 wiring, §2 + §3 rows 4-5/9) ---

    def _room_seed(self, wid: int, c: dict, psig=()) -> None:
        """DERIVE (never accumulate) one worker's fingerprint state at
        assignment — the §2 restore-lockstep invariant. The ordinal
        comes from the restored cell's threaded psig tail (the room_sig
        bytes room_id() folded at the lineage's last transit); a root
        burst, an empty psig, or a tail naming no interned ordinal
        seeds ROOM_UNKNOWN and self-heals by re-settle.

        `fp_live` starts False: an edge may only be staged once THIS
        burst has confirmed the settled hash live (a steady-state
        sample, or its own settle) — so a stale seed can never bridge
        a restore into a false adjacency edge, only re-settle."""
        o, h = ROOM_UNKNOWN, None
        if self._room_psig_off is not None and psig:
            lo_i, hi_i, n = self._room_psig_off
            if len(psig) >= n:
                v = int(psig[lo_i]) | (int(psig[hi_i]) << 8)
                hv = self.room_index.hash_of(v)
                if hv is not None:
                    o, h = v, hv
        self._room_ord[wid] = o
        c["fp_pend"] = None
        c["fp_base"] = None           # onset baseline: derived, never
        c["fp_ring"] = deque(maxlen=EXEMPLAR_RING_CEILING)  # restore-carried
        c["fp_edge"] = None
        c["fp_onset_key"] = None
        c["fp_live"] = False
        c["fp_h"] = h

    def _room_step(self, wid: int, c: dict) -> None:
        """Per-worker settle+classify, run BEFORE this step's _xram
        (§3 row 4) so the ordinal the pseudo-RAM carries is this
        frame's — the same ordinal flip then changes room_id() through
        the profile's room_sig bytes and the EXISTING transit machinery
        fires sect/psig exactly like SMB's $074E.

        A staged edge (`c["fp_edge"]`) lives exactly one step: staged
        here, committed by _room_transit inside this same iteration's
        transit block. A survivor found on the NEXT sampled step means
        the ordinal flip fired no transit (mis-wired room_sig) — it is
        dropped and counted, never committed late against an unrelated
        transit."""
        rf = self.room_fp
        ring = c.get("fp_ring")
        if ring is None:      # ctx minted before arming (defensive)
            ring = c["fp_ring"] = deque(maxlen=EXEMPLAR_RING_CEILING)
        ring.append(int(c["pending"]))
        if c.get("fp_edge") is not None:
            c["fp_edge"] = None
            self._room_edges_dropped += 1
        se = rf["sample_every"]
        if se > 1 and int(c["burst_step"]) % se:
            return
        if self.pool.odo_debug(wid)[2] < rf["min_lines"]:
            c["fp_pend"] = None       # blanks/fades are never sampled
            c["fp_base"] = None       # ... and break the onset baseline
            self._room_settle_rejects += 1
            return
        pal = (bytes(self.pool.palette_ram(wid))
               if rf["palette_cokey"] else None)
        h = nt_fingerprint(self.pool.peek_nametables(wid),
                           self._room_mask, pal)
        was = c.get("fp_pend")
        # ONSET-BASELINE CONVENTION (RG-0-falsified, receipts under
        # docs/receipts/room_fp/): a churn onset is stamped with the
        # PREVIOUS rendered sample's odo/scene — the last pre-churn
        # baseline — because a transition's odometer/scene movement
        # lands in the same sampled step as its first NT flip (the
        # measured Zelda death flash bumps the scene in the very frames
        # of its first attribute rewrite; a current-sample onset reads
        # Δscene +1-of-2 and the death mis-classifies fade -> edge). On
        # non-minting calls the current sample is the settle endpoint.
        base = c.get("fp_base")
        if was is None and base is not None:
            onset_odo, onset_scene = base
        else:
            onset_odo, onset_scene = (self._odo_now[wid],
                                      self._odo_scene[wid])
        pend, fired = fp_settle(was, h, c.get("fp_h"), onset_odo,
                                onset_scene, c["burst_step"],
                                rf["settle"])
        c["fp_base"] = (self._odo_now[wid], self._odo_scene[wid])
        c["fp_pend"] = pend
        if was is None and (pend is not None or fired is not None):
            # Churn onset: anchor the exemplar at the last cell observed
            # BEFORE the transition began — the edge-replay audit's
            # restore point (§2 graft 3). fp_settle preserves this onset
            # across intra-churn hash flips, so it stays put mid-pan.
            c["fp_onset_key"] = c.get("cur_key")
        if fired is None:
            if pend is None:
                # Steady state (h == settled hash): the carried identity
                # is confirmed live — edges may bridge from it now.
                c["fp_live"] = True
            return
        h2, d_odo, d_scene, steps = fired
        kind, direction = classify_transition(d_odo, d_scene,
                                              rf["pan_odo"],
                                              rf["warp_scene_min"])
        prev_ord = int(self._room_ord[wid])
        o = self.room_index.intern(h2, self._odo_now[wid])
        if o is None:
            # At cap: HOLD the last ordinal AND the last settled hash.
            # Adopting the uncharted hash would let a later steady
            # sample re-confirm fp_live against an identity the held
            # ordinal does not name, and the next settle would bridge a
            # false edge ACROSS the uncharted room. Held instead: the
            # pend re-fires every `settle` samples while the worker
            # stays there (cap_hits + settle_rejects count it), fp_live
            # stays down, and no edge can mint — §7.2 degrades to
            # telemetry, never to false adjacency.
            c["fp_live"] = False
            self._room_settle_rejects += 1
            return
        if kind == "warp":
            # Telemetry only, NEVER adjacency (§2 graft 2): a death /
            # teleport must not mint a navigable edge. Identity still
            # adopts below, so cells key to the room the warp landed in.
            self.room_index.record_warp(
                None if prev_ord == ROOM_UNKNOWN else prev_ord, o,
                d_scene, d_odo)
        elif (prev_ord != ROOM_UNKNOWN and prev_ord != o
                and c.get("fp_live")):
            # ITEM-SIG GRAFT (ITEM_SEMANTICS_ENGINE_2026-08-25 §3 row 2):
            # cap_sig is the already-computed sig slot (index -3) of
            # the last observed cell key, read only when armed by
            # --item-sig-report AND the profile declares state_sig.
            # getattr, not a bare attribute access: duck-typed Solver
            # stand-ins in the tests carry _room_step and nothing else,
            # like every other arm read here, and SmbGame carries no
            # state_sig_arity at all so _item_sig_armed must never be
            # assumed present. Unarmed, absent, or cur_key-less => 0,
            # the pre-graft value, on every path.
            cap_sig = (int(c["cur_key"][-3])
                      if (getattr(self, "_item_sig_armed", False)
                          and c.get("cur_key"))
                      else 0)
            c["fp_edge"] = (prev_ord, o, kind, direction,
                            int(steps) * self.frame_skip,
                            c.get("fp_onset_key"), list(ring), cap_sig)
        if prev_ord == ROOM_UNKNOWN:
            # Adoption != transit (§2): a first settle from UNKNOWN is
            # identity acquisition, not a traversal — nulling p0750
            # makes it transit-free (and, via the guards above,
            # edge-free) for root bursts and blind restores alike.
            c["p0750"] = None
            self._room_adoptions += 1
        c["fp_h"] = h2
        c["fp_live"] = True
        self._room_ord[wid] = o

    def _room_transit(self, c: dict) -> None:
        """Room-side bookkeeping inside the EXISTING transit point (§3
        row 5; the R4 recorder shape): commit the edge this step's
        settle staged — kind-tagged, warp-vetoed upstream (a warp never
        stages), exemplar-carrying — under the index lock. Transit
        logic itself is untouched; this only consumes the stage.

        Also counts transits on a burst's first step, which must stay
        zero: p0750 is None at burst start, so a restore can never mint
        a transit. The Zelda smoke asserts both properties.

        FROZEN EXEMPLAR (RG1_zelda_2026-08-25 §RG-1a/RG-1e finding):
        `e[5]` (the staged exemplar) is an ARCHIVE CELL KEY, and
        `GoExploreArchive.record` mutates `Cell.state` in place on a
        later dominating visit — so the key alone can go stale. Look the
        key up in `self.archive` right here, at commit time, and hand
        `record_edge` a bytes COPY of whatever state is live under it
        RIGHT NOW; that copy can never be touched by a later domination
        (bytes are immutable). `getattr` guards `self.archive` because
        the T2 hot-loop unit tests drive this method off a duck-typed
        stand-in that carries no archive at all — that path simply gets
        `exemplar_state=None`, same as before this graft existed."""
        if c["burst_step"] <= 1:
            self._room_restore_transits += 1
        e = c.get("fp_edge")
        if e is not None:
            c["fp_edge"] = None
            archive = getattr(self, "archive", None)
            cell = (archive.cells.get(e[5])
                    if archive is not None and e[5] is not None else None)
            exemplar_state = (bytes(cell.state)
                              if cell is not None and cell.state is not None
                              else None)
            self.room_index.record_edge(e[0], e[1], e[2], e[3], e[4],
                                        exemplar_cell=e[5],
                                        exemplar_actions=e[6],
                                        cap_sig=e[7],
                                        exemplar_state=exemplar_state)
            self._room_edges_committed += 1

    def _step0(self, a: int):
        acts = np.zeros(self.args.workers, dtype=np.uint8)
        acts[0] = self.bitmasks[a]
        out = self.pool.step_all(acts)
        self._odo_refresh()
        return self._xram(out[0][2], 0)

    # ---- record path -------------------------------------------------

    def _borrow_audit(self, verdict: str) -> None:
        """Count one borrow rejection's follow-up verdict, and DISARM the
        rule once the refutations outnumber the confirmations.

        The rule is only valid when `hi` is `lo`'s page byte. Six of the
        eleven two-byte profiles in configs/ pair a screen number with an
        in-screen x, where a lo wrap with hi flat is ordinary motion and
        every crossing loses a real observation. The verdict is measured
        one frame later (`borrow_followup`), never declared by the
        profile, and disarming is LOUD: a filter that turned itself off
        is a fact about the profile's address pair, and the run that
        finds it out should say so where the operator will see it.
        """
        if verdict == "tear":
            self._borrow_tears = getattr(self, "_borrow_tears", 0) + 1
            return
        if verdict != "not_a_pair":
            return
        self._borrow_flat = getattr(self, "_borrow_flat", 0) + 1
        if (self._borrow_flat >= BORROW_REFUTE_LIMIT
                and self._borrow_flat > getattr(self, "_borrow_tears", 0)
                and getattr(self, "progress_smooth", "off") == "borrow"):
            self.progress_smooth = "off"
            self._borrow_off = True
            print(f"[go_explore_solve] *** progress.smooth: borrow DISARMED "
                  f"*** {self._borrow_flat} rejections were followed by a "
                  f"frame in which the high byte had NOT moved, against "
                  f"{getattr(self, '_borrow_tears', 0)} that had. This "
                  f"profile's `hi` is not `lo`'s page byte (a screen/x "
                  f"composite reads exactly this way), so the rule was "
                  f"deleting real observations at every wrap. Filtering is "
                  f"off for the rest of the run; use smooth: median3 if the "
                  f"observable still needs screening.", flush=True)

    def observe(self, wid: int, ram, trace: list, steps: int,
                root_id: str, loops: int = 0,
                route_sig: tuple = (), sect: int = 0,
                psig: tuple = (), ctx: dict | None = None,
                kills: int = 0) -> str:
        """Record one reached state. Returns 'dead' | 'clear' | 'live'."""
        game = self.game
        # DEATH IS RESOLVED FIRST (v2, 2026-08-08). Any life lost, or an
        # explicit dying state = death; lives-based detection is robust
        # across enemy/pit/time deaths and multi-area levels.
        #
        # This used to run LAST, after is_finale and is_clear, which made
        # every clear-detector false positive win the race unconditionally:
        # on Gradius a real death (lives 3->2, progress reset to 0) was
        # banked as a win at 205 actions, and go_explore_chain.py would then
        # have extracted the next level's "entrance" out of a corpse. A step
        # that reads as BOTH dead and clear now resolves dead — a missed real
        # clear costs one re-search, a fabricated one poisons the chain and
        # the receipt corpus.
        #
        # Verified against every banked receipt this reorder could touch
        # before shipping it, so it is unconditional rather than flag-gated:
        # replaying regress_pre 1-1, ge_1_2/1_3/1_4/2_1 sol_000, the 4-4 maze
        # solve (1,104 actions) and the 8-4 FINALE (1,735 actions) from their
        # own roots reproduces the clear at exactly the recorded action count
        # with is_dead() FALSE at that frame in all 7 cases. Nothing banked
        # depends on the old order.
        # DEBOUNCED (2026-08-23, the Rygar door lesson — same class as
        # the level-key blip below): area/room transitions can blip the
        # lives byte through 0 while the screen is blanked (Rygar $0303
        # reads 0 for exactly 2 steps crossing the first door; receipt:
        # the 45-min odometer run walled at gx 1536 = that door, every
        # crossing burst killed by the blip). A momentary blip survives
        # (nothing is recorded during it); only a PERSISTENT dead state
        # (>= 3 consecutive observations) terminates the lineage. Real
        # deaths confirm 2 steps later at zero cost — corpse frames are
        # never banked because each blip step returns before recording.
        if game.is_dead(ram, self.start_lives):
            if ctx is not None:
                n = ctx["_dead_mm"] = ctx.get("_dead_mm", 0) + 1
                if n < 3:
                    return "live"   # blip: keep stepping, record nothing
                ctx["_dead_cause"] = "lives"
            return "dead"
        elif ctx is not None:
            ctx["_dead_mm"] = 0
        # GAME-COMPLETE check (8-4 finale): the ending never advances the
        # world/level bytes (there is no next level) — the victory state is
        # operating mode $0770 == 2 with inputs locked (verified 2026-07-27,
        # THANK YOU MARIO screen). Without this the winning trajectory sits
        # in the archive invisible, as it did for 1.5 hours on the night the
        # game was first beaten.
        #
        # Then the warp-guarded clear. ctx carries the per-worker state the
        # optional WIN-CONDITION hook needs (score_jump prev value, streaming-
        # confluence window); None at seed = level_key-only. Both go through
        # _dump_solution, which replay-verifies the candidate before anything
        # is written; a candidate that fails to reproduce is NOT a clear, so
        # the lineage is abandoned as dead rather than left to re-fire on the
        # same latched evidence next step.
        if game.is_finale(self.start_wd, ram):
            return ("clear" if self._dump_solution(root_id, trace, ram, steps)
                    else "dead")
        if game.is_clear(self.start_wd, ram, ctx):
            return ("clear" if self._dump_solution(root_id, trace, ram, steps)
                    else "dead")
        # A level-key change that ISN'T a forward clear = a warp (or a
        # backward reload / game-over reset): do not record it (would
        # poison the archive with off-level cells). DEBOUNCED (2026-08-06):
        # transition animations can blip the key byte through stale values
        # — Bubble Bobble's umbrella-warp counter dips to the PREVIOUS
        # round mid-animation, and the un-debounced kill executed every
        # lineage ~6 steps into the warp (round 52: 1,666 kills, 0 cells,
        # 0 solutions across 3 runs). A momentary blip now survives
        # (nothing is recorded during it); only a PERSISTENT off-key state
        # (a real reset/warp, >= 3 consecutive observations) still dies.
        if game.level_key(ram) != self.start_wd:
            if ctx is not None:
                n = ctx["_key_mm"] = ctx.get("_key_mm", 0) + 1
                if n < 3:
                    return "live"   # blip: keep stepping, record nothing
                # Cause marker for _assign()'s DOA retirement: a key-change
                # "dead" is a warp/reset — action-dependent, not proof the
                # ROOT state was doomed — so it must never retire the source
                # cell the way an instant lives-death does.
                ctx["_dead_cause"] = "key"
            return "dead"
        elif ctx is not None:
            ctx["_key_mm"] = 0
        # ATOMIC PEEK (opt-in, `progress: {atomic: true}`, see
        # GenericGame.progress_atomic_read): reads the two-byte pair
        # through the Rust torn-read-guarded scratch adjudicator instead
        # of composing it from this step's snapshot. `consistent=False`
        # is an unadjudicated boundary — same disposition as every other
        # garbage read on this path, drop the sample, record nothing.
        # (getattr: duck-typed Solver stand-ins in the tests and the
        # live show predate the knob.)
        if getattr(game, "progress_atomic", False):
            gx, gx_consistent = game.progress_atomic_read(self.pool, wid)
            if not gx_consistent:
                return "live"
        else:
            gx = game.progress(ram)
        if gx > game.progress_cap:
            # Transition-frame garbage read (page byte mid-load reads huge);
            # real SMB levels reach ~6,300 px (8-1) — the old 3900 cap silently
            # froze the 8-1 frontier at 3900 (states past it never archived).
            return "live"
        # TORN-READ GUARD, same disposition as the cap above: a sample
        # the filter rejects is not recorded at all, so `score`, the cell
        # key, max_gx and the pin clock never see it. Needs the burst's
        # own ctx to hold the history — one stream per worker, reset for
        # free at every re-root, which is exactly where a positional
        # observable legitimately teleports. See progress_glitch.
        # Inert when `atomic` is on: peek_u16_consistent already resolves
        # the boundary the filter can only screen for, so stacking a
        # heuristic on top of an adjudicated read could only add false
        # rejections of a genuine warp, never fix anything the peek
        # missed.
        # (getattr, not a bare attribute: every duck-typed Solver
        # stand-in in the tests and in the live show predates the knob,
        # and an observation path that only runs under the real
        # constructor is a path the unit tests cannot reach.)
        smooth = getattr(self, "progress_smooth", "off")
        if (smooth != "off" and ctx is not None
                and not getattr(game, "progress_atomic", False)):
            sample = (game.progress_pair(ram)
                      if hasattr(game, "progress_pair") else (gx, None, None))
            hist = ctx.setdefault("_prog_hist", [])
            # THE PAIR AUDIT (see borrow_followup). The frame after a
            # borrow rejection says whether the pair really is a
            # page/offset pair; the rejection itself is never undone, but
            # a rule that keeps being refuted disarms itself instead of
            # quietly deleting a real observation at every wrap.
            # (getattr, like the mode read above: the duck-typed Solver
            # stand-ins in the tests and in the live show carry observe()
            # and nothing else.)
            pend = ctx.pop("_borrow_pending", None)
            audit = getattr(self, "_borrow_audit", None)
            if pend is not None and audit is not None:
                audit(borrow_followup(pend, sample))
            torn = progress_glitch(smooth, hist, sample,
                                   getattr(self, "progress_jump",
                                           PROGRESS_JUMP))
            push_progress_sample(hist, sample,
                                 keep=(HAMPEL_KEEP if smooth == "hampel"
                                       else 2))
            if torn:
                self._prog_glitches = getattr(self, "_prog_glitches", 0) + 1
                if smooth == "borrow":
                    ctx["_borrow_pending"] = sample
                return "live"
        area = game.area(ram)
        if gx > self.max_gx_in_area.get(area, 0):
            self.max_gx_in_area[area] = gx
            self._pin_time = time.time()  # frontier moved: inversion stays off
        if area > self.max_area:
            self.max_area = area
        # Orthogonal frontier: a NEW best y-band forces a selection-cache
        # rebuild immediately. The cache otherwise waits for the 2%-growth
        # trigger (~2 min at CV's ~700 cells/min), so a freshly-climbed
        # cell would sit unselectable for exactly as long as the climb
        # takes to decay — the arm would never compound its own progress.
        if self.ortho_mode != "off":
            yb = game.y(ram) // Y_BAND
            best = self._ortho_best
            if best is None or (yb < best if self.ortho_mode == "up"
                                else yb > best):
                self._ortho_best = yb
                self._ortho_time = time.time()
                self._sel_cells = None
        # Domination score = gx within the cell; deeper-x wins, ties to fewer
        # steps (elites keep shortening). Peek-then-record: only pay
        # save_worker_state for a new/dominating cell.
        # MAZE FIX (2026-07-24): the trajectory's loop-count is the LEADING
        # key component. Castle mazes (4-4/7-4/8-4) loop wrong paths back —
        # observable as a discontinuous gx->0 collapse in our own rollouts —
        # so gx alone aliases first-pass and looped-back states and the
        # frontier saturates (4-4: pinned at gx 2064, 0 solutions, 1.5M
        # records). With the counter, the same coordinates on different
        # maze phases are DIFFERENT cells and search explores each pass.
        # Search-derived (a property of the agent's own path), no internals.
        #
        # SECOND MAZE FIX (same day, v4): the loop-count alone still aliased
        # right-route and wrong-route states at the same coordinates — the
        # game tracks the current pass's route in internal state, and the
        # wall-2102 diagnostic showed the deepest lineages were wrong-route
        # spirals from which EVERY action loops back. (A raw RAM-hash key
        # separated routes but exploded the archive to 585k one-visit cells
        # — timers/enemies churn every frame.) The ROUTE SIGNATURE is the
        # compact middle path: the trajectory's own y-band at each 512-px
        # gx-boundary crossing since the last loop event — the observable
        # footprint of the route the current pass has taken. Derived purely
        # from our own rollout; bounded cardinality (<=4 entries, 4 bands).
        # v5: key on the DISCOVERED route-tracker bytes. Differential analysis
        # of our own rollouts (32 passes, /tmp/ram_diff2.py) found $0742 and
        # $07F8 change rarely, at single consistent fork positions — the
        # empirical signature of the game's route-tracking state. Keying on
        # their VALUES separates right-route from wrong-route states with
        # bounded cardinality (vs the raw-hash explosion / sig blindness).
        # Archive-eligibility gate (8-4 lesson): lineages beyond loop-phase 6
        # are wrong-route spirals — they reach deep gx but are unwinnable, and
        # archiving them bloated the archive to 3.6M cells (sps 3300->1700)
        # while poisoning deep-cell selection. They may still EXPLORE (their
        # forks feed discovery) but do not enter the selection pool.
        if loops > 6:
            return "live"
        if sect > self.max_sect:
            self.max_sect = sect
            self._pin_time = time.time()
        # v7: CONTENT-aware cells. Hypothesis shift — the seam wrap (gx->0)
        # may fire on the CORRECT route too, loading the NEXT section's
        # layout; coordinate keys then alias post-seam progress with
        # wrong-route repeats and every gx-based metric is blind to success
        # by construction. The tile buffer ($0500-$06BF, the drawn layout) is
        # the observable content signature: same coords + same section hash
        # alike, same coords + ADVANCED section hash differently. Bounded
        # cardinality (distinct layouts only), unlike the v4 full-RAM hash.
        # SECTION-aware key (8-4 fix, verified empirically 2026-07-25):
        # correct pipe transits change the section pointer $0750 while the
        # area byte $0760 stays constant — confirmed by differential
        # comparison of our own pre/post-gate archive states (2 -> 229).
        # The transit count leads the key and dominates the score, so
        # section progress is the frontier even when gx jumps backward.
        # psig = the last-4 section-pointer values = PIPE-PATH IDENTITY.
        # sect alone aliases different pipe sequences at equal transit
        # counts; 8-4's final checks discriminate by the route taken, so
        # each path is its own frontier (fix 2026-07-26).
        tb = int(steps + 1).bit_length() if self.time_bins else 0
        kk = min(int(kills), 15) if self.kill_key else 0
        key = (sect, tb, kk, psig, loops, route_sig) + game.cell_fn(ram)
        # GATE-OPENER SHADOW LEDGER + boundary histogram. Arity is FROZEN
        # — the key above is built and consumed unchanged, so cells and
        # new_cells stay A/B-comparable between the armed run and its
        # control. The mode compare is the whole cost when off.
        if self.gate_mode != "off":
            self._gate_observe(ram, key)
        # R4 edge recording: a cell-to-cell transition in OUR OWN rollout is
        # an edge of the maze's traversal graph. Interned ids keep the
        # adjacency compact at castle-archive scale (1M+ cells).
        # cur_key is also kept fresh under room_fp (door arm off): the
        # settle loop anchors each edge EXEMPLAR at the last cell
        # observed before the churn began (§2 graft 3). Both gates off
        # => the exact pre-roomgraph behavior, byte-identical.
        if ctx is not None and (self.door_weight > 0
                                or getattr(self, "room_fp", None) is not None):
            if self.door_weight > 0:
                pk = ctx.get("cur_key")
                if pk is not None and pk != key:
                    ids = self._key_ids
                    ia = ids.get(pk)
                    if ia is None:
                        ia = ids[pk] = len(ids)
                    ib = ids.get(key)
                    if ib is None:
                        ib = ids[key] = len(ids)
                    with self._door_lock:
                        self._adj.setdefault(ia, set()).add(ib)
                        self._adj.setdefault(ib, set()).add(ia)
            ctx["cur_key"] = key
        score = sect * 10000 + gx + game.score_bonus(ram)
        cur = self.archive.cells.get(key)
        dom = (cur is None or score > cur.best_score + 1e-9
               or (abs(score - cur.best_score) <= 1e-9 and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, score, steps,
                                                        key=key):
                rec = (root_id, bytes(trace), loops, route_sig,
                       sect, psig, kills)
                if self.gate_mode != "off":
                    # gate_marks = the 8th trace element: (step, cand,
                    # kind, span) for every gate injection this lineage
                    # carries, so T2's deterministic ablation can mask
                    # exactly the frames the arm claims — SPAN included,
                    # because an injected program owns macro_hold + 6
                    # frames and a step index alone would ablate one of
                    # them. Appended ONLY when armed, so a default-path
                    # traces.pkl is byte-identical and a 7-tuple archive
                    # resumes unchanged either way. (Marks banked before
                    # the field existed are 3-tuples and are honoured as
                    # written; see gate_suppress_trace.)
                    rec = rec + (tuple(ctx.get("gate_marks", ()))
                                 if ctx is not None else (),)
                self.traces[key] = rec
                self._recorded_new = True
                # c_local: the archive's SPATIAL footprint, kept
                # incrementally here (the only place a key can enter the
                # archive) rather than rescanned at report time.
                # Reporting only — nothing branches on it.
                cl = getattr(self, "_c_local", None)
                if cl is not None:
                    cl.add((key[-5], key[-2], key[-1]))
                # R2 credit signal (fix 2026-08-09): _assign() debits the
                # source cell's `barren` counter for every burst and reads
                # `prev["yielded"]` to credit the productive ones — but
                # NOTHING ever wrote that key, so the credit branch was
                # dead and `barren` was really "times ever selected". With
                # --frontier-throttle N every archive cell went permanently
                # barren after N selections, which silently retired the
                # deep-frontier band AND the orthogonal arm (its candidate
                # loop skips barren cells and then has no pick to return).
                # Receipt: runs/bubble_bobble/r68_retry_ortho.log — armed
                # 1,800 s, ortho_pool 18, ortho_selections 0. The burst
                # taught the archive something (a new cell or a dominating
                # improvement), so its root is not a wall.
                if ctx is not None:
                    ctx["yielded"] = True
        else:
            self.archive.record(ram, None, score, steps, key=key)
        return "live"

    def replay_verify(self, root_id: str, trace) -> dict:
        """Re-simulate `trace` from its own root in a FRESH single-worker pool
        and report whether the clear reproduces.

        The whole banking chain — go_explore_chain.py's next-entrance
        extraction, CLAIMS.md receipts, the show ledger — trusts one bit
        produced by a detector that has fabricated wins on three of seven
        games (2026-08-06: a Gradius death, a Double Dragon combat blip, a
        Kirby room transition). A clear that cannot be reproduced from the
        root is not a clear, and the check is cheap enough to be
        unconditional: measured on this machine, a full replay of a banked
        solution costs ~2.7 ms per action (887 actions 2.4 s, 1,104 actions
        3.3 s, 1,735 actions 4.7 s; pool construction is not measurable
        against it), against solve runs of 25-45 minutes that dump a handful
        of candidates. Hence --verify-bank defaults ON, with --no-verify-bank
        as the escape hatch.

        FRESH pool, never the search pool: the search pool's workers hold live
        lineage state and its worker 0 is mid-burst. Verdict is judged against
        the SAME start_wd/start_lives baseline the live hook used, so this
        reproduces the live judgement rather than inventing a second one.
        A fresh ctx is used so the streaming detector must re-earn its own
        evidence rather than inheriting the latch that produced the fire.

        A windowed clear hook also gets a NOOP MARGIN past the end of the
        trace (GenericGame.clear_verify_margin, 0 for every stateless hook, so
        SMB/CV replays are unchanged). The live and replayed detectors count
        their evaluation stride from different origins — per-burst vs. root —
        and the trace ends on the live fire frame, so without a margin a
        genuine confluence clear fails to reproduce ~68% of the time purely on
        phase. Dead-before-clear holds on every frame INCLUDING the margin:
        the live hook can never adjudicate clear on a dying frame (observe()
        returns before its clear checks), and a tail-firing replay must not
        be more permissive — a Gradius-class death that trips the detector
        after the lives decrement would otherwise verify in the tail. A real
        clear never requires the player to be persistently dead at
        adjudication time.

        Returns {"ok": bool, "verdict": str, "at": int|None, "elapsed_s":
        float, "n_actions": int, "margin": int[, "error": str]} — never
        raises: an infrastructure failure is reported as a non-reproduction
        (fail closed), because banking an unverified clear is the outcome this
        exists to prevent. `at` > n_actions means the clear reproduced inside
        the margin (a phase difference, not a different trajectory)."""
        t0 = time.time()
        out = {"ok": False, "verdict": "error", "at": None,
               "n_actions": len(trace), "margin": 0, "elapsed_s": 0.0}
        pool = None
        try:
            root = self.roots.get(root_id)
            if root is None:
                raise KeyError(f"unknown root_id {root_id!r}")
            blob = Path(root["path"]).read_bytes()
            pool = Pool(rom_path=self.game.rom, num_workers=1,
                        frame_skip=self.frame_skip)
            # Same recipe (and the same load-bearing ordering) as the
            # search pool, so the replay runs the SAME machine.
            apply_hw_flags(pool, self.hw_flags)
            pool.set_headless(True)
            pool.set_skip_preprocess(True)
            pool.reset_all()
            pool.load_worker_state(0, blob)
            acts = np.zeros(1, dtype=np.uint8)
            pool.step_all(acts)          # rooting NOOP, exactly as seed() does
            ctx: dict = {}
            n = len(trace)
            margin = self.game.clear_verify_margin()
            out["margin"] = margin
            # NOOP tail, exactly as clear_detect.run_episode pads an episode
            # for the same reason. The trajectory is not extended; the hook is
            # only given the observations it needs to reach the verdict it
            # already reached live.
            needs_apu = bool(getattr(self, "_needs_apu", False))
            _dmm = 0
            for i, a in enumerate(list(trace) + [0] * margin):
                acts[0] = self.bitmasks[int(a)]
                ram = pool.step_all(acts)[0][2]
                if getattr(self, "_odo", False) or getattr(self, "_fight", False):
                    ram = self._xram_local(ram, pool, ctx)
                # Same modality the live hook saw, or the replay judges a
                # different detector than the one that fired.
                if needs_apu:
                    ctx["_apu_mask"] = pool.apu_activity_all()[0]
                # Tail included: the margin exists so a phase-lagged clear
                # can fire, and it must fire under the SAME dead-before-
                # clear ordering the live hook enforced.
                if self.game.is_dead(ram, self.start_lives):
                    _dmm += 1
                    if _dmm >= 3:   # transition-blip debounce (observe())
                        out.update(verdict="dead", at=i + 1)
                        break
                    continue  # blip: no clear adjudication on a dying frame
                else:
                    _dmm = 0
                if (self.game.is_finale(self.start_wd, ram)
                        or self.game.is_clear(self.start_wd, ram, ctx)):
                    out.update(ok=True, verdict="clear", at=i + 1)
                    break
            else:
                out.update(verdict="no_clear", at=None)
        except Exception as exc:                      # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if pool is not None:
                try:
                    pool.shutdown()
                except Exception:                     # noqa: BLE001
                    pass
        out["elapsed_s"] = round(time.time() - t0, 3)
        return out

    def counterfactual_probe(self, root_id: str, trace) -> dict:
        """K-branch counterfactual perturbation probe: does this candidate's
        clear survive the player having played the last few seconds slightly
        differently?

        THE QUESTION, and why it is different from replay verification.
        replay_verify asks "did this happen at all" — it re-simulates the
        SAME inputs and the emulator is deterministic, so a clear hook that
        fabricates a win on a transient RAM shape fabricates the identical
        win in the replay (this is exactly how the Kirby garbage-read false
        positive survived verification). This probe asks the strictly harder
        question: "is the transition something the GAME committed to, or
        something that hung off the exact frames we happened to press". A
        real stage clear is committed — a flag slide, an exit pipe animation,
        a tally cutscene all run themselves out whatever the pad does, which
        is precisely the premise of the offline detector's `lock` signal. A
        combat blip, a respawn or a one-off RAM coincidence is contingent:
        perturb the inputs going into it and it does not recur.

        This is that lock signal generalized from 2 branches (hold-a-direction
        vs. NOOP, judged by a RAM diff) to K branches judged by the clear hook
        itself, and run at its TRUE cost. It is NEVER in-loop: the measured
        4-40 s per candidate is fine on the banking path (a handful of
        candidates per 25-45 minute solve) and catastrophic anywhere near the
        hot loop, which is why there is no per-step form of it anywhere in
        this file.

        MECHANISM. Replay the trace from its root in a FRESH single-worker
        pool up to `cf_pre_steps` actions before the end, and snapshot that
        pre-clear state. From that snapshot:

          * one CONTROL branch replays the remaining actions verbatim;
          * K branches replay them with each action independently replaced,
            with probability `cf_perturb_p`, by a uniformly random action
            from the profile's own space (seeded per branch, so the whole
            probe is reproducible from the receipt).

        Every branch gets a fresh ctx and the same NOOP margin replay
        verification uses, and is judged by the same is_dead / is_finale /
        is_clear hooks the live search used.

        WHY RANDOM PERTURBATION AND NOT K CONSTANT HOLDS. Constant holds are
        the literal shape of the lock probe and were tried first on paper:
        they refuse REAL clears whose last second contains a required input.
        An exit-pipe clear needs `down` held at the pipe mouth; a branch
        holding NOOP or `right` simply never enters the pipe, so a genuine
        clear would score 1-2 branches out of 8 and be thrown away. Sparse
        random perturbation keeps the trajectory's own shape (a committed
        transition is untouched by a few changed inputs) while still
        destroying anything that balanced on exact frames.

        THE CONTROL BRANCH IS A GUARD, NOT A VOTE. If the unperturbed branch
        does not reproduce the clear from the snapshot, this probe has no
        discriminating power on this candidate — the harness itself could not
        see the clear even with the real inputs (a windowed hook whose
        evidence predates the snapshot is the obvious way that happens). The
        verdict is then `inconclusive` and ok=True: it does NOT refuse, it
        reports that it could not judge. Measuring one's own instrument and
        calling the result a fabricated clear is how a gate turns into a
        blanket mute.

        ...WHICH WAS ALSO AN OPEN FABRICATION ESCAPE (fix, 2026-08-10). A
        blind control has two causes and they are not the same event:

          * the clear COMMITTED BEFORE the snapshot — a flag slide or tally
            that latched 40 actions back is simply not inside the last 32, so
            no branch, control included, can watch it happen; or
          * there was never a game transition at all and the "clear" is an
            artifact of the DETECTOR'S OWN ROLLING STATE — a windowed
            confluence that exists only when the detector has been fed this
            candidate's entire observation history from the root, at that
            exact phase.

        Reporting `inconclusive` on both banked the second one. Receipted on
        Double Dragon (runs/detector_gate_20260810/double_dragon_cf.log): an
        88-action candidate whose control saw nothing, whose eight perturbed
        branches saw nothing, and which was banked anyway. Replay
        verification cannot catch that class BY CONSTRUCTION — it replays the
        same actions from the same root and therefore rebuilds the identical
        detector state, so the artifact reproduces perfectly every time.

        THE DEEP RETRY separates the two. On `control_blind` the probe
        re-snapshots ONCE at CF_RETRY_FACTOR x the requested pre-steps
        (capped at the trace) and re-runs the control and the branches from
        there. A commit that predates the first snapshot is INSIDE the deeper
        one: the deeper control replays through the commit point, sees the
        clear, and the candidate is then judged normally on branch agreement
        — Kirby-class candidates survive. A rolling-state artifact does not
        come back: no bounded fresh-context replay reproduces it, so the
        verdict becomes `state_artifact` with ok=False and the gate REFUSES.
        Every OTHER inconclusive reason still refuses nothing.

        WHEN THE DEEPER SNAPSHOT WOULD BE THE ROOT ITSELF (the trace is
        shorter than the retry depth, which is the Double Dragon case: 88
        actions against 4 x 32) the retry is not run, and the verdict is
        `state_artifact` with reason `root_pivot`. That is not a shortcut, it
        is the whole point: a control replaying from the root IS replay
        verification, which already passed by construction on exactly the
        candidates this refuses. Re-running it and calling the result a
        second, independent opinion would launder one check's known blind
        spot into two green ticks. The only replay that can exonerate a blind
        control is one that starts from a state INSIDE the trajectory, where
        the hook has to re-earn its evidence.

        PRECISION OVER RECALL, deliberately, and in the same direction as the
        detector's own room_veto: a genuine clear on a trace too short to
        snapshot into can be refused here, costing one re-search. Note what
        it takes to land in that case — a hook that fires from the root but
        not from a snapshot 32 actions before the end, which no STATELESS
        hook can do (level_key, byte_change, score_jump and finale all read
        the current frame, so their control sees exactly what the live hook
        saw). It is a windowed-confluence-only failure mode, which is
        precisely where the fabricated clears live.

        EVERY BRANCH GETS THE HOOK'S OWN WARM-UP (fix, 2026-08-10). A branch
        is a bounded replay into a FRESH ctx, so a windowed hook starts from
        zero evidence in it; `cf_pre_steps` alone therefore does not describe
        what the hook can see. `pre` is raised to
        `clear_observation_budget() - clear_verify_margin()` whenever that is
        larger, exactly the way clear_verify_margin itself is derived from
        stride*persist — the adapter is asked, nothing is hardcoded here.
        Before that floor existed the probe was a STRUCTURAL NO-OP against the
        one configuration it is most wanted for: 32 pre-steps + a 20-action
        margin = 52 observations against the 100 that
        `min_signals: 3, apu_weight: 1.0` needs before an APU-backed check can
        pass, so the control could never reproduce ANY clear, every candidate
        came back 'inconclusive' with ok=True, and the gate refused nothing
        while still spending 4-40 s each time. When the trace itself is too
        short to cover the budget the floor cannot help; the probe then says
        so out loud on stderr and stamps `warmup_short` on the receipt rather
        than reporting a silent 'inconclusive'. The floor is also this
        probe's cost knob — k+1 branches of (pre + margin) steps — so a hook
        needing 100 observations costs about twice one needing 52. Still
        seconds, still banking-path only, and a cheap probe that cannot
        conclude anything is not the cheaper option.

        Returns {"ok", "verdict", "agree", "k", "agreement", "threshold",
        "pre_steps", "pre_steps_requested", "warmup_budget", "margin",
        "perturb_p", "n_actions", "control", "branches", "elapsed_s"[,
        "reason"][, "warmup_short"][, "retry"][, "first_pass"][, "error"]}.
        `verdict` is one of commits / contingent / state_artifact /
        inconclusive / error. The top-level control, branches, agree,
        agreement and pre_steps always describe the pass that PRODUCED the
        verdict, so a receipt never has to be read in two places to know what
        was measured; when a deep retry ran, the superseded shallow pass is
        kept verbatim under `first_pass` and `retry` records the depth it was
        re-snapshotted at (and, when it was not run, why). Never raises; an
        infrastructure failure is reported as ok=False (fail closed, same rule
        as replay_verify)."""
        t0 = time.time()
        k = max(1, int(getattr(self, "cf_branches", 8)))
        p = float(getattr(self, "cf_perturb_p", 0.25))
        need = float(getattr(self, "cf_agree", 0.5))
        seed = int(getattr(self, "cf_seed", 0))
        trace = [int(a) for a in trace]
        n = len(trace)
        req = max(1, int(getattr(self, "cf_pre_steps", 32)))
        pre = min(req, n)
        out = {"ok": False, "verdict": "error", "agree": 0, "k": k,
               "agreement": 0.0, "threshold": need, "pre_steps": pre,
               "pre_steps_requested": req, "warmup_budget": None,
               "margin": None,
               "perturb_p": p, "n_actions": n, "seed": seed,
               "control": None, "branches": [], "elapsed_s": 0.0}
        pool = None
        try:
            root = self.roots.get(root_id)
            if root is None:
                raise KeyError(f"unknown root_id {root_id!r}")
            # WARM-UP FLOOR, resolved before a single emulator step is spent:
            # ask the adapter what its clear hook needs to SEE, and give the
            # branches at least that much (see the docstring). Both hooks are
            # called unconditionally, exactly the way replay_verify calls
            # clear_verify_margin — an adapter missing either one is an
            # infrastructure error that must fail closed, not a 0 quietly
            # substituted for a number nobody measured.
            margin = self.game.clear_verify_margin()
            budget = int(self.game.clear_observation_budget())
            pre = min(max(req, budget - margin), n)
            out["pre_steps"], out["warmup_budget"] = pre, budget
            out["margin"] = margin
            if pre + margin < budget:
                # The trace itself is shorter than the hook's own warm-up, so
                # no branch — control included — can reach an evaluation. Say
                # it, instead of returning a silent 'inconclusive' that reads
                # in a receipt like a check that ran.
                out["warmup_short"] = True
                sys.stderr.write(
                    f"[go_explore_solve] *** COUNTERFACTUAL GATE CANNOT JUDGE "
                    f"*** root={root_id}: this clear hook needs {budget} "
                    f"observations before it can fire at all, and a branch of "
                    f"this candidate gets only {pre}+{margin}="
                    f"{pre + margin} (the trace is {n} actions). The probe "
                    f"will report 'inconclusive' and refuse nothing — that is "
                    f"a MISSING check, not a passing one.\n")
                sys.stderr.flush()
            blob = Path(root["path"]).read_bytes()
            pool = Pool(rom_path=self.game.rom, num_workers=1,
                        frame_skip=self.frame_skip)
            apply_hw_flags(pool, self.hw_flags)   # same machine as the search
            pool.set_headless(True)
            pool.set_skip_preprocess(True)
            pool.reset_all()
            pool.load_worker_state(0, blob)
            acts = np.zeros(1, dtype=np.uint8)
            pool.step_all(acts)                   # rooting NOOP, as seed() does
            needs_apu = bool(getattr(self, "_needs_apu", False))
            n_act = len(self.bitmasks)

            def _branch(pivot: bytes, actions: list) -> dict:
                pool.load_worker_state(0, pivot)
                ctx: dict = {}
                verdict, at = "no_clear", None
                _dmm = 0
                for i, a in enumerate(list(actions) + [0] * margin):
                    acts[0] = self.bitmasks[int(a)]
                    ram = pool.step_all(acts)[0][2]
                    if getattr(self, "_odo", False) or getattr(self, "_fight", False):
                        ram = self._xram_local(ram, pool, ctx)
                    if needs_apu:
                        ctx["_apu_mask"] = pool.apu_activity_all()[0]
                    # Tail included, same ordering as replay_verify.
                    if self.game.is_dead(ram, self.start_lives):
                        _dmm += 1
                        if _dmm >= 3:   # transition-blip debounce
                            verdict, at = "dead", i + 1
                            break
                        continue  # blip: a dying frame must never
                                  # adjudicate clear (dead-before-clear)
                    else:
                        _dmm = 0
                    if (self.game.is_finale(self.start_wd, ram)
                            or self.game.is_clear(self.start_wd, ram, ctx)):
                        verdict, at = "clear", i + 1
                        break
                return {"verdict": verdict, "at": at}

            def _pass(pre: int, reroot: bool) -> dict:
                """One full probe at snapshot depth `pre`: replay the head,
                snapshot it, then run the control and the K perturbed
                branches from that snapshot.

                `reroot` re-loads the root state first — the deep retry
                snapshots EARLIER than the pass before it, so it cannot walk
                forward from where that one left the emulator. The first
                pass never re-roots, so the no-retry path is exactly the
                sequence of pool calls it has always been."""
                if reroot:
                    pool.load_worker_state(0, blob)
                    acts[0] = 0
                    pool.step_all(acts)           # rooting NOOP, as above
                head, tail = trace[: n - pre], trace[n - pre:]
                for a in head:
                    acts[0] = self.bitmasks[a]
                    pool.step_all(acts)
                pivot = pool.save_worker_state(0)
                control = _branch(pivot, tail)
                control["perturbed"] = 0
                branches = []
                for b in range(k):
                    rng = np.random.default_rng([seed, b, n])
                    branch = [int(rng.integers(n_act))
                              if (n_act > 1 and rng.random() < p) else a
                              for a in tail]
                    res = _branch(pivot, branch)
                    res["branch"] = b
                    res["perturbed"] = sum(1 for x, y in zip(branch, tail)
                                           if x != y)
                    branches.append(res)
                agree = sum(1 for r in branches if r["verdict"] == "clear")
                return {"pre_steps": pre, "control": control,
                        "branches": branches, "agree": agree,
                        "agreement": round(agree / float(k), 4)}

            def _judge(res: dict) -> None:
                """Promote a pass to the top level and read its verdict off
                the branch agreement (only ever called with a control that
                saw the clear, i.e. with a probe that could discriminate)."""
                out.update(pre_steps=res["pre_steps"], control=res["control"],
                           branches=res["branches"], agree=res["agree"],
                           agreement=res["agreement"])
                if res["agreement"] >= need - 1e-9:
                    out.update(ok=True, verdict="commits")
                else:
                    out.update(ok=False, verdict="contingent")

            first = _pass(pre, reroot=False)
            if first["control"]["verdict"] == "clear":
                _judge(first)
            elif out.get("warmup_short"):
                # The hook never got enough observations to look — a hole in
                # the harness, already printed above, and NOT evidence about
                # this candidate. Refuses nothing, as it always has. There is
                # nothing to retry either: warmup_short means the whole trace
                # is already inside the snapshot.
                out.update(pre_steps=first["pre_steps"],
                           control=first["control"],
                           branches=first["branches"], agree=first["agree"],
                           agreement=first["agreement"],
                           ok=True, verdict="inconclusive",
                           reason="warmup_short")
            else:
                # CONTROL BLIND -> DEEP RETRY (see the docstring). Either the
                # clear committed before this snapshot, or it is an artifact
                # of the detector's rolling state; one snapshot at
                # CF_RETRY_FACTOR x the depth tells the two apart.
                out.update(pre_steps=first["pre_steps"],
                           control=first["control"],
                           branches=first["branches"], agree=first["agree"],
                           agreement=first["agreement"])
                req2 = req * CF_RETRY_FACTOR
                pre2 = min(max(req2, budget - margin), n)
                retry = {"pre_steps_requested": req2, "pre_steps": pre2,
                         "ran": False}
                out["retry"] = retry
                if pre2 >= n:
                    # The deeper snapshot IS the root: its control would be
                    # replay verification, which passed by construction on
                    # exactly this class of candidate. Not run, not believed.
                    retry["reason"] = "root_pivot"
                elif pre2 <= first["pre_steps"]:
                    # The warm-up floor already pinned the snapshot as deep
                    # as it goes; a second identical pass would re-measure a
                    # blind control and learn nothing.
                    retry["reason"] = "no_deeper"
                else:
                    retry["ran"] = True
                    deep = _pass(pre2, reroot=True)
                    # The superseded pass is kept verbatim: a reader has to
                    # be able to see that the shallow control was blind and
                    # the deeper one was not, which is the whole argument.
                    out["first_pass"] = {f: first[f] for f in (
                        "pre_steps", "control", "branches", "agree",
                        "agreement")}
                    if deep["control"]["verdict"] == "clear":
                        # The commit predated the first snapshot; the deeper
                        # control replayed through it. Judge normally.
                        _judge(deep)
                    else:
                        out.update(pre_steps=deep["pre_steps"],
                                   control=deep["control"],
                                   branches=deep["branches"],
                                   agree=deep["agree"],
                                   agreement=deep["agreement"])
                        retry["reason"] = "control_blind_at_depth"
                # `reason` is set in exactly the three cases the retry could
                # not exonerate the candidate — and in none of the ones it
                # could.
                if retry.get("reason"):
                    out.update(ok=False, verdict="state_artifact",
                               reason=retry["reason"])
        except Exception as exc:                      # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if pool is not None:
                try:
                    pool.shutdown()
                except Exception:                     # noqa: BLE001
                    pass
        out["elapsed_s"] = round(time.time() - t0, 3)
        return out

    def _dump_solution(self, root_id: str, trace: list, ram, steps) -> bool:
        """Bank a solution. Returns whether the event counts as a REAL clear.

        False means the candidate failed replay verification — it was not a
        clear at all, so observe() abandons the lineage instead of reporting
        one. True with nothing written is the ordinary "re-clear that isn't
        materially shorter" case: real, just not worth another receipt."""
        if len(trace) >= self.best_sol_len - 8:
            return True  # keep only materially-shorter re-clears
        if self.verify_bank:
            v = self.replay_verify(root_id, list(trace))
            self.verify_checks += 1
            if not v["ok"]:
                self.verify_rejections += 1
                sys.stderr.write(
                    f"[go_explore_solve] *** CLEAR REJECTED (replay "
                    f"verification) *** root={root_id} {len(trace)} actions "
                    f"did NOT reproduce: {json.dumps(v)} — NOTHING banked. "
                    f"This is a fabricated clear, not a near miss: treat the "
                    f"profile's clear hook as unsafe until it is explained.\n")
                sys.stderr.flush()
                print(f"[go_explore_solve] CLEAR REJECTED {json.dumps(v)}",
                      flush=True)
                return False
            print(f"[go_explore_solve] replay-verified clear "
                  f"{json.dumps(v)}", flush=True)
        # COUNTERFACTUAL GATE (v3, opt-in, default OFF). Runs beside replay
        # verification and answers the question determinism cannot: would the
        # transition still have happened if the player had played the last
        # couple of seconds differently. Refusal happens ONLY here — with the
        # gate off, nothing on this path changes at all.
        cf = None
        if getattr(self, "cf_gate", False):
            cf = self.counterfactual_probe(root_id, list(trace))
            self.cf_checks = getattr(self, "cf_checks", 0) + 1
            if (cf.get("retry") or {}).get("ran"):
                self.cf_retries = getattr(self, "cf_retries", 0) + 1
            print(f"[go_explore_solve] counterfactual probe "
                  f"{json.dumps(cf)}", flush=True)
            if not cf["ok"]:
                self.cf_rejections = getattr(self, "cf_rejections", 0) + 1
                if cf.get("verdict") == "state_artifact":
                    # A DIFFERENT failure from a contingent clear, and it is
                    # worth its own words: nothing about the game happened
                    # here, so quoting a branch count would misdescribe it.
                    self.cf_state_artifacts = getattr(
                        self, "cf_state_artifacts", 0) + 1
                    sys.stderr.write(
                        f"[go_explore_solve] *** CLEAR REJECTED "
                        f"(counterfactual gate: STATE ARTIFACT) *** "
                        f"root={root_id} {len(trace)} actions: no bounded "
                        f"replay from inside the trajectory reproduces this "
                        f"clear ({cf.get('reason')}, retry "
                        f"{json.dumps(cf.get('retry'))}) — NOTHING banked. "
                        f"It is not a transition the game committed to but "
                        f"an artifact of the clear hook's own accumulated "
                        f"state, and replay verification passes it BY "
                        f"CONSTRUCTION because it rebuilds that state from "
                        f"the same root. Treat this profile's clear hook as "
                        f"unsafe until it is explained.\n")
                else:
                    sys.stderr.write(
                        f"[go_explore_solve] *** CLEAR REJECTED "
                        f"(counterfactual gate) *** root={root_id} "
                        f"{len(trace)} actions: the clear did NOT survive "
                        f"input perturbation ({cf['agree']}/{cf['k']} "
                        f"branches, threshold {cf['threshold']}) — NOTHING "
                        f"banked. The transition was contingent on the exact "
                        f"inputs, which is what a combat blip or a one-off "
                        f"RAM coincidence looks like and what a committed "
                        f"stage transition does not.\n")
                sys.stderr.flush()
                return False
        self.best_sol_len = len(trace)
        n = self.sol_counter
        self.sol_counter += 1
        self.n_solutions += 1
        base = self.out / "solutions" / f"sol_{n:03d}"
        np.save(str(base) + ".actions.npy", np.array(trace, dtype=np.int64))
        rec = {
            "provenance": "search",
            # Full effective invocation (audit finding: receipts recorded
            # neither workers nor profile nor buckets, making parity
            # claims between runs unverifiable after the fact).
            "solver_args": {k: v for k, v in vars(self.args).items()
                            if isinstance(v, (str, int, float, bool))},
            "root_id": root_id,
            "root_state": self.roots[root_id]["path"],
            # The machine this trace was produced on. A solution replayed
            # under different hw flags is not the same trajectory.
            #
            # NAMED `hw_provenance`, NOT `provenance`: `"provenance":
            # "search"` above is the honest-origin marker every banked
            # receipt carries and that CLAIMS.md audits compare against
            # the literal string (docs/proposals/smb_oneshot_campaign.md
            # 543/596). Reusing the key here shadowed it silently — a
            # dict literal keeps the LAST binding, so the marker was
            # deleted with no error. The token `provenance` means
            # exactly one thing corpus-wide: origin. The machine
            # description travels under `hw_provenance` everywhere
            # (receipts, roots.json, archive.stats.json).
            "hw_provenance": self.provenance,
            "start_wd": list(self.start_wd),
            "clear_wd": list(self.game.level_key(ram)),
            "steps": steps, "actions": len(trace),
            # Whether this receipt's clear was re-simulated from the root in a
            # fresh pool before it was written. Recorded so an audit can tell a
            # verified receipt from a pre-v2 (or --no-verify-bank) one without
            # re-running anything.
            "replay_verified": bool(self.verify_bank),
        }
        # Present ONLY when the gate ran, so a default-path receipt is
        # byte-identical to every receipt already in the corpus and an audit
        # can tell "passed the probe" from "was never probed" by the key's
        # presence rather than by its value.
        if cf is not None:
            rec["counterfactual_gate"] = cf
        (base.parent / (base.name + ".json")).write_text(
            json.dumps(rec, indent=2) + "\n")
        print(f"[go_explore_solve] *** SOLUTION {n} *** root={root_id} "
              f"{len(trace)} actions, {self.start_wd}->"
              f"{self.game.level_key(ram)}", flush=True)
        return True

    # ---- seeding: root ONLY (honest) ---------------------------------

    def seed(self) -> None:
        path = self.args.root_state
        # The root blob carries no timing config of its own; if a
        # sidecar records the lineage it was built under, say so before
        # a mismatch turns into hours of "the game behaves differently".
        check_state_sidecar(path, self.hw_flags)
        self.pool.load_worker_state(0, Path(path).read_bytes())
        r = self._step0(NOOP)  # convention: load root, one NOOP, then actions
        self.start_wd = self.game.level_key(r)
        self.start_lives = self.game.lives(r)
        self.max_area = self.game.area(r)
        # WIN-CONDITION byte_change: capture the entrance baseline the hook
        # compares against (no-op for SMB / non-configured profiles).
        if hasattr(self.game, "note_start"):
            self.game.note_start(r)
        self.roots["entrance"] = {"path": str(path),
                                  "start_wd": list(self.start_wd),
                                  "lives": self.start_lives,
                                  "hw_provenance": self.provenance}
        self.observe(0, r, [], 0, "entrance")
        prev = getattr(self.args, "resume_archive", None)
        if prev:
            prev = Path(prev)
            # Room-index half FIRST (§3 row 11): an archive whose cells
            # embed room ordinals is only meaningful next to the intern
            # table that minted them, and that check must not wait for
            # a multi-GB unpickle either.
            self._resume_room_index(prev)
            self.check_resume_lineage(prev)
            # The entrance observation above is already a cell of THIS
            # run's schema; the key audit below has to speak about the
            # RESUMED cells only, or a legacy archive's arity would come
            # back as "the archive mixes key lengths".
            own = set(self.archive.cells)
            self.archive.load(prev / "archive.pkl")
            with open(prev / "traces.pkl", "rb") as f:
                self.traces.update(pickle.load(f))
            saved_roots = json.loads((prev / "roots.json").read_text())
            for rid, info in saved_roots.items():
                self.roots.setdefault(rid, info)
            keys = [k for k in self.archive.cells if k not in own]
            # The half of the lineage check that only the LOADED keys can
            # answer (§12's disjoint tb/kk subspace).
            self.audit_resumed_keys(prev, keys)
            # THE BANKED PHANTOMS, REFROZEN. The torn-read filter screens
            # new samples; these cells are already in the pickle, and
            # both frontier readers take a maximum over them.
            if self.resume_refreeze:
                self._gx_phantoms = phantom_top_buckets(keys)
                if self._gx_phantoms:
                    n = sum(1 for k in keys
                            if (int(k[-5]), int(k[-1])) in self._gx_phantoms)
                    print(f"[seed] REFROZEN frontier: {n} resumed cells in "
                          f"{sorted(self._gx_phantoms)} are a torn-read "
                          f"island (empty gap below, negligible population) "
                          f"and are excluded from max_gx/topgx. They stay in "
                          f"the archive; only the frontier readers ignore "
                          f"them (--no-resume-refreeze to keep them).",
                          flush=True)
            # Rebuild the frontier trackers the loaded cells imply.
            for c in self.archive.cells.values():
                area, gx = c.key[-5], c.key[-1] * GX_BUCKET
                sect = c.key[0]
                if area > self.max_area:
                    self.max_area = area
                # Only the gx read is refused: the tear is in the
                # two-byte position pair, and the cell's transit count is
                # a different byte that the same frame recorded honestly.
                if (gx > self.max_gx_in_area.get(area, 0)
                        and (int(area), int(c.key[-1]))
                        not in self._gx_phantoms):
                    self.max_gx_in_area[area] = gx
                if sect > self.max_sect:
                    self.max_sect = sect
            # One pass, once, so the incremental c_local series continues
            # the resumed archive's footprint instead of restarting at 0.
            self._c_local |= local_coverage(self.archive.cells.keys())
            print(f"[seed] RESUMED archive from {prev}: "
                  f"{len(self.archive.cells)} cells, "
                  f"{len(self.traces)} traces, max_area={self.max_area}, "
                  f"max_sect={self.max_sect}, "
                  f"c_local={len(self._c_local)}", flush=True)
        print(f"[seed] rooted at {path} wd={self.start_wd} lives="
              f"{self.start_lives} area={self.max_area}; archive="
              f"{json.dumps(self.archive.stats())}", flush=True)
        if self.gate_mode != "off" or self.gate_axes:
            self._write_gate_header(path)

    def key_config(self) -> dict:
        """This run's cell-key schema, as banked into archive.stats.json
        and compared on resume."""
        return key_config_axes(self.args, self.game)

    def _resume_room_index(self, prev) -> None:
        """Room-graph half of the --resume-archive check (§3 row 11).

        A room_fp archive's cells carry interned ordinals in their
        psig/area key slots; those ordinals are an ALPHABET defined by
        room_index.json. Resuming without that file, or across a
        different room_fp schema, is not a weaker lineage that an
        override flag can accept — the resumed keys would be
        reinterpreted under a different intern table — so every refusal
        here is HARD (no --allow-* escape). When everything matches,
        the loaded index becomes this run's, so new settles continue
        the same discovery-order ordinals. Per-worker state needs no
        repair: it self-heals from cell psig + re-settle (§2
        invariants).

        The on/off axis itself (archive fingerprinted, run not — or the
        reverse) is the generic lineage diff's job (`room_fp` in
        LINEAGE_KEY_AXES) with its usual overrides; this method owns
        only the index file's existence and schema."""
        prev = Path(prev)
        try:
            stats = json.loads((prev / "archive.stats.json").read_text())
        except (OSError, ValueError):
            stats = {}
        banked_sha = str((stats.get("key_config") or {}).get("room_fp", "")
                         or "")
        run_sha = str(getattr(self.game, "room_fp_sha", "") or "")
        if not banked_sha and not run_sha:
            return
        idx_path = prev / "room_index.json"
        if not banked_sha:
            # Archive off, run on: the lineage diff refuses it (absent
            # reads as off). Nothing to load here.
            return
        if not idx_path.exists():
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: its "
                f"archive was built with room fingerprinting (room_fp "
                f"{banked_sha}) but {idx_path.name} is missing — the "
                f"cells' room ordinals are uninterpretable without the "
                f"intern table that minted them.")
        try:
            idx = RoomIndex.load(idx_path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: "
                f"{idx_path.name} is unreadable or speaks a different "
                f"schema ({e}).")
        if idx.config_sha != banked_sha:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: "
                f"{idx_path.name} carries config_sha {idx.config_sha!r} "
                f"but the archive's lineage records room_fp "
                f"{banked_sha!r} — the index on disk is not the one the "
                f"cells were interned under.")
        if run_sha and run_sha != banked_sha:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: this "
                f"run's solve.room_fp hashes to {run_sha!r} but the "
                f"archive was interned under {banked_sha!r} — a "
                f"different mask/settle/classifier schema mints "
                f"different ordinals for the same rooms.")
        if run_sha:
            self.room_index = idx
            print(f"[seed] RESUMED room index from {idx_path}: "
                  f"{idx.n_rooms()} rooms, "
                  f"{sum(len(d) for d in idx.adj.values())} edges, "
                  f"{idx.warp_count} warps (config_sha {idx.config_sha})",
                  flush=True)

    def check_resume_lineage(self, prev) -> dict:
        """Refuse a --resume-archive whose lineage disagrees with this
        run's, or cannot be checked at all.

        Runs BEFORE archive.pkl is loaded: a multi-GB unpickle is not
        the right place to discover the cells were never comparable.

        UNVERIFIABLE IS A REFUSAL NOW, AND THE RECOVERY IS WHY IT CAN BE.
        The first build warned and resumed on an unrecorded axis, on the
        grounds that a legacy archive is the common case — which meant
        the exact §12 archive (runs/cv_chain_hw/lvl_03_overnight: no
        hw_provenance, no key_config in its stats sidecar) produced
        mismatch=[] and resumed silently, one flag against four. But its
        lineage was never actually unknown: `recover_lineage` reads it
        out of roots.json and the root blob's own state sidecar, which
        records hw_flags=["mmio_read_timing"]. So the check first
        recovers everything the disk can answer and only then refuses
        what is left, with --allow-unverified-lineage as the explicit
        way past it (--allow-lineage-mismatch, the stronger override,
        implies it).

        The key-schema half cannot be answered here — a key is only
        readable once the archive is loaded — and is audited in seed()
        by `audit_resumed_keys`.
        """
        prev = Path(prev)
        try:
            stats = json.loads((prev / "archive.stats.json").read_text())
        except (OSError, ValueError) as e:
            stats = {}
            print(f"[go_explore_solve] no readable archive.stats.json in "
                  f"{prev} ({e}): falling back to the run's own root "
                  f"sidecars for its lineage.", flush=True)
        recovered, notes = recover_lineage(prev, stats)
        if recovered is not None and not isinstance(
                stats.get("hw_provenance"), dict):
            stats = dict(stats, hw_provenance=recovered)
        diff = resume_lineage_diff(stats, self.provenance, self.key_config())
        allowed = bool(getattr(self.args, "allow_lineage_mismatch", False))
        unver_ok = allowed or bool(
            getattr(self.args, "allow_unverified_lineage", False))
        if diff["mismatch"] or diff["unverifiable"] or diff["caveats"]:
            report = format_lineage_report(prev, diff, allowed, notes,
                                           unver_ok)
            (sys.stderr if diff["mismatch"] or diff["unverifiable"]
             else sys.stdout).write(report + "\n")
            sys.stderr.flush()
        if diff["mismatch"] and not allowed:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: "
                f"{len(diff['mismatch'])} lineage mismatch(es) above. Pass "
                f"--allow-lineage-mismatch to override.")
        if diff["unverifiable"] and not unver_ok:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: "
                f"{len(diff['unverifiable'])} lineage axis/axes could not be "
                f"checked at all (above). Pass --allow-unverified-lineage to "
                f"resume anyway.")
        return diff

    def audit_resumed_keys(self, prev, keys) -> dict:
        """The key-schema half of the resume check, read off the LOADED
        keys — the only place it can be read.

        `key_config` was never flushed by any archive on disk, so the
        recorded comparison is unverifiable for all of them, including
        the §12 resume base. The keys are not silent though: tb, kk, the
        state_sig width and the tuple arity are all recoverable from
        them (see `key_schema_from_keys`), and a disjoint tb/kk subspace
        is precisely what §12 found.

        IT REFUSES ONLY CONTRADICTIONS. What the keys cannot settle is
        already refused upstream — `check_resume_lineage` reads an
        unrecorded `key_config` as UNVERIFIABLE and exits on it — so a
        second refusal here would be double jeopardy, and it would also
        fire on archives that DID record their schema and matched it (a
        sig bit the archive never happened to set is not evidence of
        anything). The ambiguous tier is printed and returned, never
        enforced.
        """
        observed = key_schema_from_keys(keys)
        conflicts = key_schema_conflicts(observed, self.key_config())
        allowed = bool(getattr(self.args, "allow_lineage_mismatch", False))
        if conflicts["mismatch"] or conflicts["unverifiable"]:
            out = [f"[go_explore_solve] *** RESUMED KEY SCHEMA *** {prev} "
                   f"({observed['n']} keys, arity {observed['arity']})"]
            out += [f"    MISMATCH  {m}" for m in conflicts["mismatch"]]
            out += [f"    unchecked {u}" for u in conflicts["unverifiable"]]
            sys.stderr.write("\n".join(out) + "\n")
            sys.stderr.flush()
        if conflicts["mismatch"] and not allowed:
            raise SystemExit(
                f"[go_explore_solve] refusing to resume {prev}: the loaded "
                f"cells were keyed under a different schema (above). Pass "
                f"--allow-lineage-mismatch to override.")
        return conflicts

    def _write_gate_header(self, root_path) -> None:
        """The A/B header. Written once, next to the archive, and echoed
        to stdout so a teed log carries it too."""
        hdr = gate_run_header(
            self.args, commit=_git_commit(), hw_flags=self.hw_flags,
            root_sha=hashlib.sha256(
                Path(root_path).read_bytes()).hexdigest()[:16],
            sidecar_sha=self.gate_axes_sha, axes=self.gate_axes,
            active_predicate=(
                "gate_armed(mode!=off, now-_pin_time>=--gate-pin-secs>=0, "
                "--gate-target-typed, band growth <5% of peak over 3 "
                "checkpoints)"))
        hdr["basis"] = {"patterns": len(self._gate_basis),
                        "program_len": ib.PROGRAM_LEN_PASS_A,
                        "phases": list(self._gate_phases),
                        "tail": ib.TAIL_PASS_A}
        (self.out / "gate_header.json").write_text(
            json.dumps(hdr, indent=2) + "\n")
        print(f"[gate] header {json.dumps(hdr)}", flush=True)

    # ---- frontier selection ------------------------------------------

    def _refresh_sel_cache(self) -> None:
        """Rebuild the selection caches. A full archive scan ran on EVERY
        worker reassignment (every episode end) — O(330k) three times over
        (state filter, deep filter, weighted fallback) — starving the Rust
        pool at 1.6/16 cores with sps decayed 2378->475. Rebuild only when
        the archive grows 2% or the deepest area advances."""
        self._sel_cells = [c for c in self.archive.cells.values()
                           if c.state is not None]
        deep_area = self._sel_cells
        if self.ortho_mode != "off":
            # Restrict the ORTHO POOL — and only it — to the DEEPEST area:
            # an earlier area's column ceilings are not the wall being
            # climbed, and mixing them would let a tall stale column
            # define this one's frontier. Keyed on the MODE (static), not
            # on armed-ness (time-varying) — a cache whose contents depend
            # on when it happened to be rebuilt is not reproducible.
            #
            # SCOPED TO THE ARM (fix): this filter used to overwrite
            # self._sel_cells, i.e. it restricted the ENTIRE selection
            # pool. record() bumps self.max_area BEFORE the `loops > 6`
            # archive-eligibility early-out, so a castle-maze spiral
            # (4-4/7-4/8-4 — the exact case the loop counter exists for)
            # that crosses an area boundary advances max_area with ZERO
            # state-bearing cells in that area. The pool then emptied,
            # select() returned None, and _assign reset EVERY worker to
            # the entrance root — permanently, because entrance bursts
            # archive into the earlier area that the same filter discards.
            # Reproduced against snapshot e0f59e9 (6 cells in area 3,
            # max_area 4: mode=off -> 6 cells; mode=up -> 0, select None).
            # The ortho arm is additive: an empty deep-area subset now
            # just makes the arm fall through, exactly as an evicted pool
            # already did, and the count/legacy arms keep the full pool.
            deep_area = [c for c in self._sel_cells
                         if c.key[-5] == self.max_area]
            ext = column_extremes([c.key for c in deep_area],
                                  self.ortho_mode)
            better = min if self.ortho_mode == "up" else max
            prev = self._ortho_ext
            # Cumulative (monotone) count of column ceilings this run has
            # pushed — the arm's own partial-progress signal, independent
            # of whether a solution ever lands.
            self._ortho_cols_improved += sum(
                1 for col, yb in ext.items()
                if col in prev and yb != prev[col]
                and better(yb, prev[col]) == yb)
            self._ortho_ext = ext
            self._ortho_pool = ortho_pool(deep_area, self.ortho_mode,
                                          self.ortho_band)
            self._ortho_ids = {c.key for c in self._ortho_pool}
        self._sel_n = len(self.archive.cells)
        self._sel_area = self.max_area
        self._sel_maxscore = max(
            (c.best_score for c in self._sel_cells), default=1.0) or 1.0
        deep = [c for c in self._sel_cells
                if c.key[0] == self.max_sect and c.key[-5] == self.max_area]
        if self._gx_phantoms:
            # THE REFROZEN FRONTIER (§12). `_sel_topgx` is a maximum, so
            # a banked torn read defines the band every selection arm
            # samples: ortho_ctrl's 14 bucket-95 cells restore at gx
            # 507-511 and pinned topgx at 95 for every resumed run.
            deep = [c for c in deep
                    if (int(c.key[-5]), int(c.key[-1]))
                    not in self._gx_phantoms]
        self._sel_deep = deep
        if deep:
            minl = min(c.key[0] for c in deep)
            lowl = [c for c in deep if c.key[0] <= minl + 1]
            self._sel_topgx = max(c.key[-1] for c in deep)
            # Near-frontier bands precomputed (the per-call band filter over
            # `deep` was itself O(N) in one-area castle levels).
            _f24 = self._sel_topgx - 24
            self._sel_band24 = [c for c in deep if c.key[-1] >= _f24]
            self._sel_lowl_band24 = [c for c in lowl if c.key[-1] >= _f24]
        else:
            self._sel_topgx = 0
            self._sel_band24 = []
            self._sel_lowl_band24 = []
        if self.ortho_mode != "off":
            # The literal figure the pre-registered gate names: the
            # extreme y-band reached inside the deep band the primary arm
            # actually samples (gx buckets >= topgx-24), inside the
            # deepest area (the arm's own subset, not the whole pool).
            ybs = [c.key[-2] for c in deep_area
                   if c.key[-1] >= self._sel_topgx - 24]
            self._ortho_deep_yband = (
                (min(ybs) if self.ortho_mode == "up" else max(ybs))
                if ybs else None)
        # ROOM ROUTER CACHE (T3, §3 row 7): the same single scan feeds
        # the room pools. The ortho lesson holds — _sel_cells is never
        # overwritten, and empty pools just make the arm fall through.
        # Guarded on the ARM, not just the feature (getattr: duck-typed
        # stand-ins in the tests predate the room attrs), so a
        # fingerprinting run with --room-bias 0 pays nothing here.
        if (getattr(self, "room_bias", 0.0) > 0
                and getattr(self, "room_index", None) is not None):
            poff = getattr(self, "_room_psig_off", None)
            idx = self.room_index
            pools: dict = {}
            with idx.lock:
                idx_n = len(idx.ordinals)
            for c in self._sel_cells:
                o = room_cell_ord(c.key, poff)
                if o is not None and o < idx_n:
                    pools.setdefault(o, []).append(c)
            with idx.lock:
                und: dict = {}
                out_dirs: dict = {}
                for s, dsts in idx.adj.items():
                    for d, e in dsts.items():
                        und.setdefault(s, set()).add(d)
                        und.setdefault(d, set()).add(s)
                        if e.get("dir"):
                            out_dirs.setdefault(s, set()).add(e["dir"])
                # Aliasing audit (§7.1): marks are MONOTONE — set, never
                # cleared — like every other RoomIndex mutation, so the
                # flag survives save/load and restore ordering.
                for o in aliased_rooms(idx.adj):
                    m = idx.meta.get(o)
                    if m is not None:
                        m["aliased"] = True
                aliased = {o for o, m in idx.meta.items()
                           if m.get("aliased")}
            self._room_pools = pools
            self._room_out_dirs = out_dirs
            self._room_degree = {o: len(nb) for o, nb in und.items()}
            # Inline, not the async door thread: the room graph is at
            # most max_rooms nodes (§3 row 7) — three orders of
            # magnitude below the cell graph the R4 daemon scans.
            self._room_artic = self._articulation_points(und)
            self._room_aliased = aliased
            near = int(getattr(self, "transition_near", 24))
            sides_all: dict = {}
            bounds: dict = {}
            u: dict = {}
            vis: dict = {}
            for o, pool in pools.items():
                sides, bbox = room_boundaries(pool, near)
                sides_all[o] = sides
                bounds[o] = bbox
                have = out_dirs.get(o, set())
                # U(r): sides with boundary cells but no out-edge of
                # that direction — the unexplored-boundary exit term.
                u[o] = sum(1 for dr, cs in sides.items()
                           if cs and dr not in have)
                vis[o] = float(sum(int(c.times_chosen) for c in pool))
            self._room_sides = sides_all
            self._room_bounds = bounds
            self._room_U = u
            self._room_V = vis

    def _ortho_armed(self) -> bool:
        return ortho_armed(self.ortho_mode, self._pin_time, time.time(),
                           self.ortho_pin_secs)

    def select(self):
        if (getattr(self, "_sel_cells", None) is None
                or len(self.archive.cells) > self._sel_n * 1.02
                or self._sel_area != self.max_area):
            self._refresh_sel_cache()
        cells = self._sel_cells
        if not cells:
            return None
        # Deep-frontier arm: bias toward cells in the DEEPEST area reached,
        # near its max gx — this follows Mario through area transitions.
        if self.rng.random() < self.args.deep_bias:
            pool_band = (self._sel_lowl_band24
                         if (self.rng.random() < 0.7 and self._sel_lowl_band24)
                         else self._sel_band24)
            if pool_band and self.frontier_throttle > 0:
                # DD-RRT boundary suppression: cells whose bursts came
                # back empty `throttle` times in a row are walls — stop
                # sampling them; if the whole band is walls, fall through
                # to the count arm (least-visited exploration).
                pool_band = [c for c in pool_band
                             if getattr(c, "barren", 0) < self.frontier_throttle]
            if pool_band:
                floor = self._sel_topgx - int(self.rng.integers(0, 24))
                band = [c for c in pool_band if c.key[-1] >= floor]
                if band:
                    cell = band[int(self.rng.integers(len(band)))]
                    cell.times_chosen += 1
                    cell.explored = True
                    return cell
        # Orthogonal-frontier arm (between the deep and count arms so
        # deep_bias semantics are unchanged): once the x frontier is
        # pinned, spend `ortho_bias` of the remaining budget on the
        # top-of-column cells with a PURE count prior W =
        # 1/sqrt(times_chosen+1) (Wmax 1.0) — deliberately no score term,
        # since the score is exactly what makes a climb look worthless.
        # Same O(1) rejection sampling and same barren skip as the other
        # arms; an empty pool, or one that is all wall, falls through.
        armed = self._ortho_armed()
        if armed and self._ortho_pool and self.rng.random() < self.ortho_bias:
            opool = self._ortho_pool
            pick = None
            for _ in range(64):
                cand = opool[int(self.rng.integers(len(opool)))]
                if (self.frontier_throttle > 0
                        and getattr(cand, "barren", 0) >= self.frontier_throttle):
                    continue
                pick = cand
                if self.rng.random() < 1.0 / (cand.times_chosen + 1) ** 0.5:
                    break
            if pick is not None:
                pick.times_chosen += 1
                pick.explored = True
                self._ortho_selections += 1
                return pick
        # ROOM-GRAPH ROUTER ARM (T3, §3 row 8) — between the ortho and
        # count arms, gated on --room-bias (default 0.0): with the arm
        # off there is no draw and no branch, so the default stream is
        # byte-identical (the ortho gate-ordering lesson). Selection-
        # side only — pure count prior, score untouched. Sample a room
        # from the frontier set F by w(r), then a cell via the ortho
        # arm's exact rejection loop + barren skip; an empty F, an
        # unpopulated pool set, or an all-wall room falls through to
        # the count arm exactly like an evicted ortho pool.
        if getattr(self, "room_bias", 0.0) > 0:
            pools = self._room_pools
            if pools and self.rng.random() < self.room_bias:
                front = room_frontier(pools.keys(), self.room_recent_k,
                                      self._room_degree, self._room_artic,
                                      self._room_aliased, self._room_U)
                if front:
                    w = np.array([room_weight(
                        o in self._room_artic, o in self._room_aliased,
                        self._room_U.get(o, 0), self._room_V.get(o, 0.0),
                        self.room_artic_weight, self.room_exit_weight)
                        for o in front])
                    r = front[int(self.rng.choice(len(front),
                                                  p=w / w.sum()))]
                    sides = self._room_sides.get(r, {})
                    have = self._room_out_dirs.get(r, set())
                    # U(r)'s support: sides with boundary cells and no
                    # out-edge yet. One of them (drawn uniformly) rides
                    # to _assign as the burst's route_dir.
                    open_sides = [d for d in ("E", "W", "N", "S")
                                  if sides.get(d) and d not in have]
                    route_dir = (open_sides[int(self.rng.integers(
                        len(open_sides)))] if open_sides else None)
                    pool = pools[r]
                    # Boundary cells at p=0.40: the routed side's when a
                    # side was drawn, else any side's.
                    if self.rng.random() < 0.40:
                        bpool = (sides.get(route_dir) if route_dir
                                 else [c for cs in sides.values()
                                       for c in cs])
                        if bpool:
                            pool = bpool
                    pick = None
                    for _ in range(64):
                        cand = pool[int(self.rng.integers(len(pool)))]
                        if (self.frontier_throttle > 0
                                and getattr(cand, "barren", 0)
                                >= self.frontier_throttle):
                            continue
                        pick = cand
                        if self.rng.random() < 1.0 / (cand.times_chosen
                                                      + 1) ** 0.5:
                            break
                    if pick is not None:
                        pick.times_chosen += 1
                        pick.explored = True
                        self._room_router_picks += 1
                        self._route_pick = (int(r), route_dir,
                                            self._room_bounds.get(r))
                        return pick
        if self.sel_mode == "count":
            # Count-based prior via O(1) rejection sampling: accept cell
            # with prob W/Wmax, W = 1/sqrt(times_chosen+1) * (score_norm
            # + 0.1), Wmax = 1.1. Expected O(few) draws; bounded at 64.
            # R4: door cells (articulation points of the transition graph)
            # get door_weight x — Wmax scales so the sampling stays exact.
            ms = self._sel_maxscore
            doors = self._doors
            dw = self.door_weight if doors else 0.0
            # Orthogonal frontier gets the same treatment as doors: an
            # up-weight inside the count arm, with Wmax scaled so the
            # rejection sampling stays EXACT when both multipliers arm.
            ow = self.ortho_weight if armed else 1.0
            # v1 never up-weights (gw == 1.0, no flag declares one), but
            # the multiplier and its ceiling travel together by
            # construction so they can never be split later.
            gw = self.gate_weight
            wmax = count_wmax(dw, ow, gw)
            pick = None
            for _ in range(64):
                pick = cells[int(self.rng.integers(len(cells)))]
                # R2 extension (block-3 doomed-tip lesson): a cell whose
                # bursts die yielding nothing `throttle` times in a row is
                # skipped here too — deterministic death states (e.g. a
                # committed bat swoop) otherwise drain the count arm's
                # budget forever, since depth keeps their score high.
                if (self.frontier_throttle > 0
                        and getattr(pick, "barren", 0) >= self.frontier_throttle):
                    continue
                w = ((1.0 / (pick.times_chosen + 1) ** 0.5)
                     * (pick.best_score / ms + 0.1))
                if dw > 0 and self._key_ids.get(pick.key) in doors:
                    w *= dw
                if ow > 1.0 and pick.key in self._ortho_ids:
                    w *= ow
                if self.rng.random() < w / wmax:
                    break
            pick.times_chosen += 1
            pick.explored = True
            return pick
        # Legacy: uniform over the cached list (the archive's weighted
        # selection was an O(N) scan; the deep arm supplies direction).
        return cells[int(self.rng.integers(len(cells)))]

    # ---- R4: door discovery (articulation points, async) --------------

    @staticmethod
    def _articulation_points(adj: dict) -> set:
        """Iterative Hopcroft-Tarjan cut vertices of an undirected graph
        given as {node: iterable-of-neighbors}. Pure derived-from-rollouts
        structure — no game internals."""
        disc: dict = {}
        low: dict = {}
        ap: set = set()
        timer = 0
        for start in adj:
            if start in disc:
                continue
            disc[start] = low[start] = timer
            timer += 1
            root_children = 0
            stack = [(start, None, iter(adj.get(start, ())))]
            while stack:
                node, parent, it = stack[-1]
                pushed = False
                for nb in it:
                    if nb == parent:
                        continue
                    if nb in disc:
                        if disc[nb] < low[node]:
                            low[node] = disc[nb]
                    else:
                        disc[nb] = low[nb] = timer
                        timer += 1
                        if node == start:
                            root_children += 1
                        stack.append((nb, node, iter(adj.get(nb, ()))))
                        pushed = True
                        break
                if not pushed:
                    stack.pop()
                    if stack:
                        pnode = stack[-1][0]
                        if low[node] < low[pnode]:
                            low[pnode] = low[node]
                        if pnode != start and low[node] >= disc[pnode]:
                            ap.add(pnode)
            if root_children > 1:
                ap.add(start)
        return ap

    def _door_scan_qos(self, *a, **k):
        # Housekeeping: run the periodic articulation-point scan on the
        # E-cores (QOS_BACKGROUND) so it never evicts worker cache lines.
        from src.training.qos import demote_current_thread
        demote_current_thread()
        return self._door_scan(*a, **k)

    def _door_scan(self) -> None:
        """Snapshot the adjacency and publish the current door set. Runs in
        a daemon thread; the snapshot copy holds the edge lock briefly (the
        4-4 frozen-show lesson: never scan a live million-entry structure
        from the hot loop)."""
        try:
            with self._door_lock:
                snap = {k: tuple(v) for k, v in self._adj.items()}
            self._doors = frozenset(self._articulation_points(snap))
        except Exception as exc:
            print(f"[door_scan] failed (doors unchanged): {exc}", flush=True)

    def _maybe_scan_doors(self, now: float) -> None:
        if self.door_weight <= 0:
            return
        if now - self._last_door_t < self.door_interval:
            return
        if self._door_thread is not None and self._door_thread.is_alive():
            return
        self._last_door_t = now
        self._door_thread = threading.Thread(target=self._door_scan_qos,
                                             daemon=True)
        self._door_thread.start()

    def _fight_reset(self, wid: int) -> None:
        """Re-baseline one worker's fight-gate state after a fresh
        `load_worker_state` (entrance or an archived cell) — a no-op
        when `progress.source: fight_gate` isn't configured. The
        cumulative-damage total starts over at 0 for whatever this
        worker does from here, exactly the `_fight_prev_hp is None`
        first-observation case `fight_gate_step` already handles: the
        very first post-load read is a re-arm, never a delta against
        stale pre-load state. See the KNOWN GAP note above
        `self._fight` in `__init__` — this is the honest, documented
        limit of that choice, not a silent one."""
        if not self._fight:
            return
        self._fight_prev_hp[wid] = None
        self._fight_cum[wid] = 0
        self._fight_round_prev[wid] = None
        self._fight_prev_ram[wid] = None

    def _assign(self, wid: int, prev: dict | None = None) -> dict:
        # R2 bookkeeping: credit or debit the cell the finished burst
        # was rooted at. A burst that recorded nothing novel increments
        # the source cell's barren counter; any novelty resets it.
        if prev is not None and prev.get("key") is not None:
            src = self.archive.cells.get(prev["key"])
            if src is not None:
                if prev.get("yielded"):
                    src.barren = 0
                else:
                    src.barren = getattr(src, "barren", 0) + 1
                    # DEAD-ON-ARRIVAL retirement (the NG scene-9 corpse
                    # lesson): a burst that died within its first few
                    # steps was rooted at a state that was already doomed
                    # when banked — the death byte lags the fatal hit, so
                    # no bank-time check can screen these. One instant
                    # death is proof enough; retire the cell immediately
                    # instead of draining `throttle` bursts on it. Armed
                    # with --frontier-throttle like the rest of the
                    # barren machinery. Lives-deaths only: a key-change
                    # "dead" (warp/reset, _dead_cause == "key") depends on
                    # the burst's sampled actions, not the root state, so
                    # one unlucky warp entry must not retire a live cell.
                    if (self.frontier_throttle > 0
                            and prev.get("_dead_cause") != "key"
                            and prev.get("died_at_burst_step") is not None
                            and prev["died_at_burst_step"] <= 5):
                        src.barren = max(src.barren, self.frontier_throttle)
        # T3 route handoff: cleared BEFORE every select() so a stale
        # router stash can never survive a deep/count-arm pick (those
        # arms return without touching it) and tag the wrong burst.
        self._route_pick = None
        cell = self.select()
        if cell is None:
            # Fall back to the entrance root.
            self.pool.load_worker_state(wid, Path(self.args.root_state).read_bytes())
            if getattr(self, "_fight", False):
                self._fight_reset(wid)
            c = {"key": None, "root": "entrance", "trace": [], "steps": 0,
                 "left": self.args.burst, "burst_step": 0,
                 "loops": 0, "prev_gx": -1,
                 "sig": (), "sect": 0, "p0750": None, "psig": (),
                 "kills": 0, "eslots": None, "ortho": False,
                 "prev": int(self.rng.choice(len(self.weights), p=self.weights))}
            # getattr: duck-typed Solver stand-ins in the tests carry
            # _assign and nothing else, like every other arm read here.
            if getattr(self, "room_fp", None) is not None:
                # Root path: ROOM_UNKNOWN — the first settle adopts
                # transit-free and edge-free (§2 lockstep invariants).
                self._room_seed(wid, c, ())
            return c
        self.pool.load_worker_state(wid, cell.state)
        if getattr(self, "_fight", False):
            self._fight_reset(wid)
        rec = self.traces[cell.key]
        root_id, tb, loops, sig = rec[0], rec[1], rec[2], rec[3]
        sect = rec[4] if len(rec) > 4 else 0
        psig = rec[5] if len(rec) > 5 else ()
        # prev_gx -1: no loop detection on the restore step (the load frame
        # reads transitional garbage; the first real step re-arms it).
        c = {"key": cell.key, "root": root_id, "trace": list(tb),
                "steps": cell.best_steps, "left": self.args.burst,
                "burst_step": 0, "loops": loops, "prev_gx": -1, "sig": sig,
                "sect": sect, "p0750": None, "psig": psig, "cur_key": cell.key,
                "kills": rec[6] if len(rec) > 6 else 0, "eslots": None,
                # 8th element (gate_marks) is present only on lineages a
                # gate-armed run recorded; a resumed 7-tuple archive
                # loads with an empty mark list and nothing else changes.
                "gate_marks": list(rec[7]) if len(rec) > 7 else [],
                # Burst rooted at an orthogonal-frontier cell: explore()
                # rolls the hold-macro at ortho_macro_p for these.
                "ortho": cell.key in self._ortho_ids,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights))}
        if getattr(self, "room_fp", None) is not None:
            # Cell path: seed the ordinal from the restored lineage's
            # threaded psig tail; per-worker fingerprint state is
            # DERIVED here, never carried across restores (§2).
            self._room_seed(wid, c, psig)
        # T3 route attach (§3 row 9): a router-arm pick tags its burst
        # with the target room and — when that room still has an
        # unexplored boundary side — the side to push, plus the room's
        # cell-unit bbox for the route-follow near gate. Attached ONLY
        # on router picks, so a --room-bias 0 run's ctx dicts are
        # byte-identical.
        rp = self._route_pick
        if rp is not None:
            self._route_pick = None
            c["route_room"] = rp[0]
            if rp[1] is not None and rp[2] is not None:
                c["route_dir"], c["route_bbox"] = rp[1], rp[2]
        # ARM C: hand this burst a queued candidate program. It happens at
        # ASSIGNMENT and nowhere else, so an injection can never arrive
        # mid-burst or preempt an in-flight macro; only bursts rooted
        # inside the same --gate-band the sweep rooted in are eligible;
        # and the round-robin costs no RNG draw, which is what keeps arm
        # C's search stream comparable with arm B's. Mode first, so a
        # default run pays one string compare.
        if self.gate_mode != "off" and self._gate_inject:
            if int(cell.key[-1]) >= (getattr(self, "_sel_topgx", 0)
                                     - self.gate_band):
                ai, hold, addr = self._gate_inject[
                    self._gate_inject_i % len(self._gate_inject)]
                self._gate_inject_i += 1
                c["gate_macro"] = (ai, hold)
                c["gate_cand"] = addr
        return c

    # ---- gate-opener arm ----------------------------------------------

    def _gate_armed(self, now: float | None = None) -> bool:
        """Live gate for the sweep. Self-disarms on a fresh frontier
        advance (the pin clock reset) and after a candidate admission;
        re-arms only once the frontier pins again."""
        if self._pin_time != self._gate_last_pin:
            self._gate_last_pin = self._pin_time
            self._gate_disarmed = False
        if self._gate_disarmed:
            return False
        return gate_armed(self.gate_mode, self._pin_time,
                          time.time() if now is None else now,
                          self.gate_pin_secs, self.gate_target_typed,
                          band_growth_stalled(self._gate_band_hist))

    def _gate_checkpoint(self, now: float) -> int:
        """One band-growth checkpoint: count the band, append it to the
        history the arming conjunct reads, and roll the armed-secs clock.

        THE CADENCE IS THE ARMING FLOOR (repair D-ARM). `band_growth
        _stalled` needs four checkpoints before it can return True, so
        the interval between checkpoints multiplied by four is the
        earliest any sweep can fire — and while this lived inside the 60 s
        progress line that floor was 240 s, hardcoded, on runs whose whole
        budget was fifteen minutes. Now the interval is
        --gate-arm-cadence-secs and the floor is stated rather than
        inherited from a logging cadence. The default is unchanged.
        """
        band = band_cell_count(self.archive.cells.keys(),
                               band=self.gate_band)
        self._gate_band_hist.append(band)
        self._gate_band_last = band
        armed = self._gate_armed(now)
        if armed and self._gate_armed_since is None:
            self._gate_armed_since = now
        elif not armed and self._gate_armed_since is not None:
            self._gate_armed_secs += now - self._gate_armed_since
            self._gate_armed_since = None
        self._gate_last_ckpt = now
        return band

    def _gate_hist_alloc(self) -> None:
        """Allocate the 2 MB boundary histogram on first use. Its own
        method because BOTH samplers feed it now — the live in-band
        observations and the sweep's undirected frames — and a lazily
        allocated array with two writers needs one place that creates
        it."""
        if self._boundary_hist is None:
            self._boundary_hist = np.zeros((ib.RAM_SIZE, 256), dtype=np.uint32)
            self._boundary_rows = np.arange(ib.RAM_SIZE)

    def _gate_baseline_sample(self, vals, prev):
        """Fold ONE undirected frame into the prior and the farm counter.

        `vals` is the RAM after a frame in which the program had not yet
        pressed anything (module docstring of interaction_basis, D-PRIOR:
        the settle prefix of every program, and the whole body of the
        ladder's all-NOOP controls). `prev` is the previous such frame of
        the SAME program, or None at its first — consecutive frames of one
        program are a stride-1 change measurement; frames from different
        programs are not consecutive and are never paired.

        Post-onset frames are deliberately excluded. Counting them would
        let the instrument cancel its own discoveries: the value a pattern
        just created would score as "already seen at this boundary" and
        the novelty term would shrink toward zero exactly where it is
        supposed to be largest.

        Returns the frame, to be carried as the next `prev`.
        """
        self._gate_hist_alloc()
        if self._gate_change_sweep is None:
            self._gate_change_sweep = np.zeros(ib.RAM_SIZE, dtype=np.uint32)
        self._boundary_hist[self._boundary_rows[:len(vals)], vals] += 1
        self._boundary_hist_total += 1
        self._gate_baseline_frames += 1
        if prev is not None and len(prev) == len(vals):
            self._gate_change_sweep[:len(vals)] += (vals != prev)
            self._gate_change_sweep_n += 1
        return vals

    def _gate_root_signature(self, cell) -> str:
        """The identity of a root's UNDIRECTED TRAJECTORY: a digest of the
        state blob every program at that root is restored from.

        Not the cell key. Two archive keys holding the same blob restore
        the same machine and, under the same all-NOOP input, produce the
        same frames — so they are one trajectory and one piece of
        evidence, whatever the archive calls them. Memoised per cell key,
        since a root is swept many times over a run.
        """
        key = getattr(cell, "key", None)
        sig = self._gate_root_sig.get(key)
        if sig is None:
            blob = getattr(cell, "state", None) or b""
            sig = hashlib.blake2b(bytes(blob), digest_size=16).hexdigest()
            if key is not None:
                self._gate_root_sig[key] = sig
        return sig

    def _gate_baseline_fold(self, sig: str, t: int, vals) -> bool:
        """Fold frame `t` of root `sig`'s undirected trajectory into the
        prior and the K4 screen, ONCE.

        Returns True when the frame was new. Every program at a root
        replays the same frames from the same blob, so the second and
        every later sighting of frame `t` is a replay and is counted as
        `baseline_dupes` instead of as evidence. The pair (t-1, t) is
        formed only when frame t-1 was itself folded and still cached, so
        a pair is never invented across a gap — the direction that can
        cost a change count, never fabricate one.
        """
        if t <= int(self._gate_baseline_len.get(sig, 0)):
            self._gate_counters["baseline_dupes"] += 1
            return False
        prev = (self._gate_baseline_prev.get(sig)
                if int(self._gate_baseline_len.get(sig, 0)) == t - 1 else None)
        self._gate_baseline_sample(vals, prev)
        self._gate_baseline_len[sig] = t
        if len(self._gate_baseline_prev) > GATE_BASELINE_CACHE \
                and sig not in self._gate_baseline_prev:
            self._gate_baseline_prev.pop(next(iter(self._gate_baseline_prev)))
        self._gate_baseline_prev[sig] = vals
        return True

    def _gate_liveness(self, ram):
        """The profile's OWN liveness telemetry (its declared lives byte),
        or None when the profile declares none.

        Read through the same adapter the search reads it through, so the
        death-aware control pairing uses no address this run did not
        already depend on. A profile with no lives address yields None and
        every observation is treated as live, i.e. exactly the behaviour
        that predates the repair."""
        try:
            return int(self.game.lives(ram))
        except Exception:                                  # noqa: BLE001
            return None

    def _gate_observe(self, ram, key) -> None:
        """Boundary histogram + shadow ledger. Called from observe() only
        while the arm is on. NEVER touches the cell key."""
        self._gate_obs_n += 1
        if self._gate_obs_n % ib.BAND_SAMPLE_STRIDE:
            return
        top = getattr(self, "_sel_topgx", 0)
        gxb = int(key[-1])
        if gxb < top - self.gate_band:
            return
        self._gate_hist_alloc()
        if self._gate_change is None:
            # Its own guard rather than a rider on the histogram's: the
            # two are read by different consumers, and coupling the
            # allocations means a caller that seeds one leaves the hot
            # path indexing None.
            self._gate_change = np.zeros(ib.RAM_SIZE, dtype=np.uint32)
        vals = ib.as_ram(ram)[:ib.RAM_SIZE]
        self._boundary_hist[self._boundary_rows[:len(vals)], vals] += 1
        self._boundary_hist_total += 1
        # K4's farmability numerator: how often this address DIFFERS
        # between consecutive band samples. Counted here rather than
        # derived from the histogram, which knows the values a byte took
        # and nothing about how often it moved between them.
        prev = self._gate_prev_vals
        if prev is not None and len(prev) == len(vals):
            self._gate_change[:len(vals)] += (vals != prev)
            self._gate_change_n += 1
        self._gate_prev_vals = np.array(vals, dtype=np.uint8)
        yb = int(key[-2])
        self._gate_positions.add((yb, gxb))
        for addr in self._gate_admitted:
            self._gate_shadow.setdefault(addr, set()).add(
                (int(vals[addr]) // ib.VALUE_BUCKET, yb, gxb))

    def _gate_novelty(self, addr: int, value: int) -> float:
        """Surprise, in bits, of `value` at `addr` under this run's own
        UNDIRECTED prior — see interaction_basis.novelty_score.

        THE SUPPORT IS THE REPAIR. The prior holds pre-press frames only
        (post-press frames would let a pattern cancel its own discovery),
        so the post-press value of every LATCHED candidate is unseen by
        construction and the count-only estimator hands all of them the
        same log2(total+1): on the probe, 40 latched rows, one novelty,
        one score, and a plateau no tie-break can be blamed for. The
        address's own SUPPORT — how many distinct values it took across
        the same undirected frames — is measured from the same histogram
        and separates a byte that has been pinned all run from one that
        free-runs, which is exactly the distinction the plateau erased.
        """
        if self._boundary_hist is None:
            return 1.0
        row = self._boundary_hist[int(addr)]
        seen = int(row[int(value) & 0xFF])
        return ib.novelty_score(seen, self._boundary_hist_total,
                                support=int(np.count_nonzero(row)))

    def _gate_farm_sources(self) -> list:
        """The (changes, samples, stride) triples behind every farmability
        number this run reports. Pulled out of the estimator so the
        receipt can state them SEPARATELY: "the live band sampler saw
        nothing and the sweep carried the reading" and "both agreed" are
        different facts, and the first is the one the graded run needed to
        be able to say."""
        src = []
        if self._gate_change is not None and self._gate_change_n:
            src.append(("live_band", self._gate_change,
                        self._gate_change_n, ib.BAND_SAMPLE_STRIDE))
        if self._gate_change_sweep is not None and self._gate_change_sweep_n:
            src.append(("sweep_undirected", self._gate_change_sweep,
                        self._gate_change_sweep_n, ib.SWEEP_SAMPLE_STRIDE))
        return src

    def _gate_farmability(self, addr: int):
        """K4's farmability for one address, MEASURED off this run's own
        sampling: value-change events per 1,000 observed steps, POOLED
        over the live in-band sampler and the sweep's own undirected
        frames.

        The pooling is the D-PRIOR repair's other half. The live sampler
        alone cannot answer K4 at a pinned wall — it only fires on band
        cells the search re-enters, and the sweep is rooted precisely in
        the tail it does not — so on the graded Castlevania run it never
        produced a single sample and every significant row was refused
        `farm_unmeasured`. The sweep's undirected frames are consecutive
        and are counted exactly (stride 1); pooling in step units is what
        makes the two combinable.

        Returns None — never 0.0 — when the pooled sources have not
        observed enough steps to resolve the 1-event/1k threshold. An
        unmeasured axis is then REFUSED by the ranker rather than admitted
        on a number nothing produced; a constant 0.0 in a receipt column
        called "farmability" is a fabricated measurement, and it would
        also make K4's farmable half unable to fire at all.
        """
        src = self._gate_farm_sources()
        if not src:
            return None
        a = int(addr)
        return ib.farm_rate_pooled(
            [(int(counts[a]), n, stride) for _name, counts, n, stride in src])

    def shadow_yield(self) -> float:
        """Cells the admitted axes WOULD create, per distinct in-band
        position — the crossing-free endpoint. Zero without a candidate,
        so it can never be flattered by coverage alone."""
        denom = max(1, len(self._gate_positions))
        return sum(len(v) for v in self._gate_shadow.values()) / denom

    def _gate_roots(self) -> list:
        """The sweep's roots: the least-swept band cells, plus sham roots
        drawn from the freely-advancing region (K1's null). Archive blobs
        are a free reset, so a root is just a cell that still has one.

        THE LEAST-SWEPT TIE IS BROKEN AT RANDOM (repair D-ROOTSEL). The
        key used to be `(times_swept, key[-1])`, and on a fresh sweep
        every band cell has times_swept 0 — so the operative key was
        ASCENDING GX and the selector returned the single most-rearward
        slice of the band, the same correlated cluster every time. Its
        receipt: 200 random cells of a 15,852-cell Contra band held an
        OCCUPIED hp slot 123 times (61.5%); the 32 cells the selector
        actually drew held one ZERO times (0.0%), at bands 24, 96 and
        192 alike, and 19 of those 32 roots died identically under the
        all-NOOP control. An answer byte that is unoccupied at every
        root cannot be measured at any parameter setting, so no
        gate_sweep_roots / gate_band value could reach the miss — this
        is D-TIE's family (an arbitrary ordering key deciding an
        outcome) one stage earlier, deciding what gets MEASURED rather
        than what ranks.

        The tie value comes from the SWEEP'S OWN Generator, which is
        seeded off the run seed: a sweep is reproducible from its
        header, the search stream is untouched (see _gate_sweep), and
        no address, column or key term is privileged. Sorting by
        (times_swept, coin) is a uniform sample without replacement
        from the least-swept stratum, then the next, so the
        least-swept-first contract is unchanged — only its ties move.
        Cross-root invariance presumes >=3 INDEPENDENT roots, and it is
        now fed a spread of the band rather than one corner of it.
        """
        cells = [c for c in self.archive.cells.values() if c.state is not None]
        if not cells:
            return []
        top = max(int(c.key[-1]) for c in cells)
        floor = top - self.gate_band
        band = [c for c in cells if int(c.key[-1]) >= floor]
        free = [c for c in cells if int(c.key[-1]) < floor]
        coin = self._gate_rng.random(len(band))
        order = sorted(range(len(band)),
                       key=lambda i: (self._gate_swept.get(band[i].key, 0),
                                      float(coin[i]), i))
        roots = [(band[i], False) for i in order[:self.gate_sweep_roots]]
        n_sham = min(self.gate_sham_roots, len(free))
        if n_sham:
            idx = self._gate_rng.choice(len(free), size=n_sham, replace=False)
            roots += [(free[int(i)], True) for i in idx]
        return roots

    def _gate_maybe_sweep(self, ctx: list, now: float,
                          deadline: float | None = None) -> None:
        # THE RUN'S DEADLINE BINDS THE SWEEP TOO. A full pass-A sweep is
        # tens of thousands of worker-steps and its pass-B tail is
        # 654-step programs; started one tick before --minutes expires it
        # overruns by an entire sweep. T1 scores three arms at EQUAL
        # WALL-CLOCK, so an arm that runs minutes long is not the arm
        # that was registered — and the arm that overruns is always the
        # armed one, which is the deciding metric's own direction.
        if deadline is not None and now >= deadline:
            return
        if not self._gate_armed(now):
            return
        if self.steps_done < self._gate_next_sweep:
            return
        roots = self._gate_roots()
        if not roots:
            # Empty (or state-free) archive: nothing to sweep. Fall
            # through silently rather than reassigning anything — the
            # search owns worker state, the sweep only borrows it.
            return
        cost = ib.sweep_step_cost(len(self._gate_basis), len(roots),
                                  self.gate_sweep_repeats,
                                  len(self._gate_phases))
        interval = ib.sweep_interval_steps(cost, self.gate_sweep_frac)
        self._gate_sweep(ctx, roots)
        # Scheduled off the post-sweep counter: the sweep's own steps go
        # into steps_done too (which is exactly why sps is blind to the
        # tax), so anchoring on the pre-sweep count would spend
        # cost/(cost+interval-cost) instead of the requested fraction.
        self._gate_next_sweep = self.steps_done + interval

    def _gate_sweep(self, ctx: list, roots=None) -> list:
        """One full interaction sweep off archive blobs.

        STATE ISOLATION IS THE CONTRACT. Before anything is loaded, every
        worker's savestate is banked and its ctx deep-copied; afterwards
        both are restored verbatim. `_assign` is NEVER called from here
        (it calls select(), draws from self.rng and debits `barren`), no
        cell's barren/yielded bookkeeping is touched, and the sweep draws
        exclusively from its own Generator — so scheduling a sweep cannot
        shift a single draw of the search stream.

        The restore is ATOMIC PER WORKER: the machine and the ctx go back
        together or neither does. Handing back a ctx over a machine that
        was never rewound is worse than dropping the burst, because the
        worker then writes a trace that does not describe the frames it
        ran and observe() banks it as a cell.
        """
        if roots is None:
            roots = self._gate_roots()
        if not roots or not self._gate_basis:
            return []
        n = int(self.args.workers)
        saved = [(i, self.pool.save_worker_state(i), copy.deepcopy(ctx[i]))
                 for i in range(n)]
        jobs = [(cell, sham, slot, pat, phase, rep)
                for cell, sham in roots
                for rep in range(max(1, self.gate_sweep_repeats))
                for phase in self._gate_phases
                for slot, pat in enumerate(self._gate_basis)]
        obs: list = []
        try:
            for start in range(0, len(jobs), n):
                obs += self._gate_wave(jobs[start:start + n])
            # ONE BH DENOMINATOR FOR THE WHOLE SWEEP, charged to both
            # calls below. The grid is addr x family, and a call left to
            # derive it sees only the families ITS OWN half of the
            # observations carried: the wall roots can raise a CONTACT
            # relabel that the sham roots, by construction away from the
            # wall, never do. The null would then be graded against a
            # 3-family grid while the wall it is the null FOR was graded
            # against a 4-family one — K1 compares two yields, so the two
            # bars have to be the same bar, and a receipt has to be able
            # to name it.
            fams = {o["family"] for o in obs if not o.get("control")}
            self._gate_fdr_m = fdr_m = ib.comparison_grid(len(fams))
            # ONE TIE SEED FOR THE WHOLE SWEEP, and it is the RUN's seed.
            # The tie-break has to be arbitrary (that is the point — see
            # tie_break_key), but it must not be arbitrary DIFFERENTLY for
            # the wall and its null, or K1 compares two shuffles.
            tie = int(getattr(self.args, "seed", 0))
            stats: dict = {}
            # K1: wall roots and sham roots are ranked SEPARATELY by the
            # same instrument. A sham yield that approaches the wall
            # yield means the ranking is reading noise, whatever it found
            # at the wall.
            ranked = self._gate_rank([o for o in obs if not o["sham"]],
                                     novelty=self._gate_novelty,
                                     farmability=self._gate_farmability,
                                     m=fdr_m, tie_seed=tie, stats=stats)
            sham = self._gate_rank([o for o in obs if o["sham"]],
                                   novelty=self._gate_novelty,
                                   farmability=self._gate_farmability,
                                   m=fdr_m, tie_seed=tie)
            self._gate_rank_stats = stats
            ranked = self._gate_pass_b(obs, ranked, roots)
        finally:
            # Whatever brought us into this finally is the report that
            # matters. The reassignment below is a REPAIR, and a repair
            # that throws inside a finally replaces the failure it was
            # repairing — the sweep's real traceback would be swapped for
            # a select() error raised while cleaning up after it. So the
            # repair is caught per worker, the rest of the restore still
            # runs, and it is re-raised (chained) only when nothing else
            # is in flight; when something is, it goes to stderr in full
            # and the original propagates untouched.
            in_flight = sys.exc_info()[1]
            repairs: list = []
            for i, blob, snap in saved:
                if blob is None:
                    # BOTH OR NEITHER. Restoring the ctx over a machine
                    # that was never rewound is the worst of the three
                    # outcomes: the worker keeps stepping a sweep-end
                    # emulator while the search believes it is mid-burst
                    # at `snap`, so every action it appends lands in a
                    # trace that does not describe the frames that ran —
                    # and observe() banks that trace as a cell. A failed
                    # savestate therefore resyncs the pair from a real
                    # archive root instead. That costs an RNG draw and
                    # voids byte-identity for the run, which is exactly
                    # why it is counted and shouted rather than absorbed.
                    self._gate_counters["restore_failed"] += 1
                    sys.stderr.write(
                        f"[gate] worker {i}: save_worker_state returned "
                        f"nothing before the sweep — its burst is "
                        f"ABANDONED and the worker reassigned. The search "
                        f"stream has diverged; this run is no longer "
                        f"byte-comparable with its A/B partner.\n")
                    try:
                        # prev=snap, because this burst is OVER: the
                        # search credits or debits the source cell of
                        # every burst it finishes (R2's barren signal),
                        # and an abandoned burst is still a burst that
                        # ran. Dropping prev here would silently exempt
                        # exactly the roots the sweep interrupted.
                        ctx[i] = self._assign(i, prev=snap)
                    except Exception as exc:              # noqa: BLE001
                        repairs.append((i, exc))
                        sys.stderr.write(
                            f"[gate] worker {i}: the reassignment that "
                            f"replaces its abandoned burst ALSO failed — "
                            f"the worker has no valid ctx.\n"
                            f"{traceback.format_exc()}")
                    continue
                self.pool.load_worker_state(i, blob)
                ctx[i] = snap
            if repairs and in_flight is None:
                i, exc = repairs[0]
                raise RuntimeError(
                    f"gate sweep: worker {i} could not be restored (its "
                    f"savestate was never banked) and the reassignment "
                    f"that replaces it failed") from exc
        for cell, _sham in roots:
            self._gate_swept[cell.key] = self._gate_swept.get(cell.key, 0) + 1
        self._gate_counters["sweeps"] += 1
        self._gate_counters["programs"] += len(jobs)
        # K1'S NULL IS A MEASUREMENT OR IT IS NOTHING. The sham arm draws
        # from the region BELOW the band floor, and that region can be
        # empty by construction (a one-column target has no freely
        # advancing region at all), in which case no sham program ever
        # ran. Reporting 0.0 there hands the kill — "sham yield >= 50% of
        # wall yield" — a clean bill of health off a null that does not
        # exist. None says UNMEASURED, and the two counts behind the
        # ratio are published beside it.
        n_sham = sum(1 for _c, s in roots if s)
        wall_hits = sum(1 for r in ranked if r["significant"])
        sham_hits = sum(1 for r in sham if r["significant"])
        self._gate_counters["sham_roots"] = n_sham
        self._gate_counters["sham_hits"] = sham_hits
        self._gate_counters["wall_hits"] = wall_hits
        self._gate_counters["sham_yield"] = (
            sham_hits / max(1, wall_hits) if n_sham else None)
        self._gate_admit(ranked, roots)
        return ranked

    def _gate_pass_b(self, obs, ranked, roots) -> list:
        """Re-run the surviving PATTERNS at four phases and the long tail
        from three confirm roots, to fix the persistence class.

        A 24-step tail misreads a slow-reverting LATCHED change as TIMED
        — the tail is a ladder rung (24 -> 512), not a tuned constant —
        and the class is the term that carries the most weight in the
        rank. Cheap by construction: only pass-A survivors come back.
        """
        keep = {r["addr"] for r in ranked if r["significant"]}
        if not keep:
            return ranked
        addrs = np.fromiter(sorted(keep), dtype=np.int64)
        slots: dict = {}
        for o in obs:
            if o.get("control") or o["sham"] or o["slot"] in slots:
                continue
            r0, r2 = ib.as_ram(o["ram0"]), ib.as_ram(o["ram2"])
            if bool(np.any(r0[addrs] != r2[addrs])) or bool(
                    np.any(ib.as_ram(o["ram1"])[addrs] != r0[addrs])):
                slots[o["slot"]] = o["pattern"]
        keep_slots = sorted(slots)[:16]
        if not keep_slots:
            return ranked
        # THE LENGTH TWINS. A survivor cannot come back for its long-tail
        # confirmation without the all-NOOP program of its own length
        # coming with it: subtraction is only valid between programs that
        # measured the same window, and pass B's window is 20x longer, so
        # an unmatched control here would hand every slow clock in the
        # machine a confirmed LATCHED class.
        ctl = ib.control_slots(self._gate_basis)
        run_slots = sorted({0} | set(keep_slots) | {
            ctl[len(self._gate_basis[s].masks)] for s in keep_slots
            if len(self._gate_basis[s].masks) in ctl})
        confirm = [(c, s) for c, s in roots if not s][:3]
        jobs = [(cell, sham, slot, self._gate_basis[slot], phase, 0)
                for cell, sham in confirm
                for phase in ib.SETTLE_PHASES_PASS_B
                for slot in run_slots]
        n = int(self.args.workers)
        bobs: list = []
        for start in range(0, len(jobs), n):
            bobs += self._gate_wave(jobs[start:start + n],
                                    tail=ib.TAIL_PASS_B,
                                    length=ib.PROGRAM_LEN_PASS_B,
                                    baseline=False)
        self._gate_counters["programs"] += len(jobs)
        confirmed = {(r["addr"], r["family"]): r
                     for r in self._gate_rank(
                         bobs, novelty=self._gate_novelty,
                         farmability=self._gate_farmability,
                         tie_seed=int(getattr(self.args, "seed", 0)))}
        for r in ranked:
            c = confirmed.get((r["addr"], r["family"]))
            r["pass_b"] = c is not None
            if c is not None:
                farm = r["farmability"]
                r["class"] = c["class"]
                r["score"] = (r["novelty"]
                              * ib.PERSIST_WEIGHT.get(c["class"], 0.0)
                              * (r["roots"] / max(1, r["family_roots"]))
                              * (1.0 if farm is None
                                 else max(0.0, 1.0 - farm)))
        # ONE sort key, defined once in interaction_basis. A pass-B
        # re-sort with its own key is how an address-ordered tie-break
        # walks back in through the confirmation pass after the ranking
        # itself was repaired.
        ranked.sort(key=ib.rank_sort_key)
        return ranked

    def _gate_wave(self, jobs, tail: int = ib.TAIL_PASS_A,
                   length: int = ib.PROGRAM_LEN_PASS_A,
                   baseline: bool = True) -> list:
        """Step one lockstep wave of <= workers programs and capture RAM
        at each program's own t0/t1/t2.

        CONTACT is admitted HERE and not in the basis: whether a hold is
        pressed against something is a property of the state it runs
        from, so it cannot be a function of the action space alone. The
        test is the profile's OWN gx/y telemetry over the settle window,
        closed by the pattern's first frame (eps, K and the window rule
        declared in interaction_basis and echoed in the run header) — no
        collision map, no game internals. `n_settle=t0` is what keeps the
        K-step freeze anchored to settle: the extra frame is a separate
        closing conjunct at every phase, and a window EXTENSION only at
        phase 8, where a settle of eight samples cannot supply the nine a
        K=8 freeze needs (the root frame is not readable). See the
        sampling loop below and interaction_basis.contact_admitted.

        THE SWEEP FEEDS ITS OWN PRIOR AND ITS OWN SCREEN (repair
        D-PRIOR). Every frame of a program in which nothing has been
        pressed yet is an undirected observation at a sweep root, and
        those frames go into the same boundary histogram and the same
        farmability counter the live search feeds. That is the population
        the ranking is actually about; the live sampler alone measures a
        different one (the band cells the search re-enters) and, at a
        pinned wall, measured nothing at all.

        `baseline=False` FOR PASS B, and the reason is the same one the
        BH denominator is charged the full grid for. Pass B re-runs the
        slots that SURVIVED pass A (and their length twins) from the
        first three roots — a sub-population chosen by looking at the
        data. Letting it feed the prior would make the horizon the prior
        is measured over a function of what the sweep happened to find,
        so two sweeps of the same wall would screen against different
        priors. Pass A is the pre-declared population: every root, every
        pattern in the basis, the same horizon every time.

        THE BASELINE WINDOW IS DEATH-AWARE TOO. It closes at the frame
        the program's liveness reading changes, because a death cascade
        is neither a quiescent prior nor an undirected change rate — and
        on the staged Castlevania band roots a 153-step NOOP program dies
        at step 31-115, so without the close the screen would read every
        address the death touches as wildly farmable and K4 would refuse
        the entire map.

        AND IT MEASURES LIVENESS. The profile's own lives byte is read at
        t0/t1/t2, so an observation whose differential window contains a
        death is marked and excluded by the ranker — as a control (whose
        death cascade would otherwise be subtracted out of every real
        candidate) and as a pattern (whose death cascade would otherwise
        BE the candidate). On the staged Castlevania band roots, none of
        the 16 survives the 153-step all-NOOP program, so this is not a
        corner case at that wall: it is the common case.
        """
        progs, points = [], []
        for cell, _sham, _slot, pat, phase, _rep in jobs:
            progs.append(ib.build_program(pat, phase, tail, length))
            points.append(ib.capture_points(pat, phase, tail))
        for wid, job in enumerate(jobs):
            self.pool.load_worker_state(wid, job[0].state)
        caps: list = [{} for _ in jobs]
        settle_pos: list = [[] for _ in jobs]
        # How many leading frames of each program press nothing — the
        # window in which a sample is still an UNDIRECTED observation.
        undirected = ([ib.noop_prefix_len(p) for p in progs] if baseline
                      else [0] * len(jobs))
        # The trajectory each worker's undirected frames belong to. All
        # programs at one root share it, which is exactly why the frames
        # are folded once (see _gate_baseline_fold).
        root_sig = [self._gate_root_signature(j[0]) for j in jobs]
        base_live: list = [_UNSET for _ in jobs]
        lives: list = [{} for _ in jobs]
        acts = np.zeros(int(self.args.workers), dtype=np.uint8)
        for t in range(1, length + 1):
            acts[:] = 0
            for wid, prog in enumerate(progs):
                acts[wid] = prog[t - 1]
            results = self.pool.step_all(acts)
            if getattr(self, "_odo", False):
                self._odo_refresh()
            self.steps_done += int(self.args.workers)
            self._gate_counters["steps"] += int(self.args.workers)
            for wid, (t0, t1, t2) in enumerate(points):
                # SETTLE, PLUS ONE FRAME OF THE PATTERN. CONTACT_K = 8
                # frozen steps needs nine samples and a settle of t0
                # steps yields only t0 of them (the root frame itself is
                # unreadable: the pool exposes RAM only as the result of
                # a step), so at the shortest registered phase, 8, the
                # freeze window is one sample short and contact could
                # never be admitted there at all. The extra sample is
                # taken one frame INTO the pattern, where a genuine
                # contact is still frozen and open ground is already
                # moving. Which of the two jobs it does is decided by
                # contact_admitted, NOT here: at phase 8 it extends the
                # window, and at every phase it is an independent closing
                # check that can only refuse. The sampler's job is just
                # to record it, labelled by n_settle=t0 below.
                if t <= t0 + 1:
                    ram = (self._xram(results[wid][2], wid)
                           if getattr(self, "_odo", False)
                           or getattr(self, "_fight", False)
                           else results[wid][2])
                    settle_pos[wid].append((self.game.progress(ram),
                                            self.game.y(ram)))
                if t <= undirected[wid]:
                    # Nothing has been pressed in this program yet, so
                    # this frame belongs to the prior and to the K4
                    # screen — ONCE per root trajectory. Every program at
                    # a root replays the same undirected frames from the
                    # same blob, so a second sighting of frame t is a
                    # replay, not a second observation, and pairing is
                    # done inside the trajectory rather than inside the
                    # program (which would also invent changes across
                    # two programs' unrelated frames).
                    lv = self._gate_liveness(results[wid][2])
                    if base_live[wid] is _UNSET:
                        base_live[wid] = lv
                    if lv != base_live[wid]:
                        # The window CLOSES here and does not reopen: a
                        # respawn is not a return to the state that was
                        # being measured.
                        undirected[wid] = 0
                        self._gate_counters["baseline_truncated"] += 1
                    else:
                        self._gate_baseline_fold(
                            root_sig[wid], t,
                            ib.as_ram(results[wid][2])[:ib.RAM_SIZE].copy())
                if t in (t0, t1, t2):
                    caps[wid][t] = ib.as_ram(results[wid][2]).copy()
                    lives[wid][t] = self._gate_liveness(results[wid][2])
        out = []
        for wid, (cell, sham, slot, pat, phase, rep) in enumerate(jobs):
            t0, t1, t2 = points[wid]
            if not all(k in caps[wid] for k in (t0, t1, t2)):
                continue
            # CONTACT ADMISSION: a longest hold whose settle window left
            # the position observable frozen was pressed against
            # something, and that is a DIFFERENT interaction from the
            # same hold across open ground — so it is ranked as its own
            # family rather than diluted into the hold family's
            # cross-root count. The pattern name is unchanged, so the
            # receipt still names the program that ran.
            contact = ib.contact_admitted(settle_pos[wid], n_settle=t0)
            family = pat.family
            if (contact and pat.family == ib.HOLD
                    and len(pat.masks) == max(ib.HOLD_STEPS)
                    and any(pat.masks)):
                family = ib.CONTACT
            out.append({
                # The cell key IS the root identity: stable across
                # processes, unlike an object id, so two sweeps of the
                # same archive produce comparable receipts.
                "root": cell.key,
                "sham": bool(sham),
                "slot": int(slot),
                "pattern": pat.name, "family": family, "phase": phase,
                "repeat": rep,
                # The paired control is every all-NOOP program in the
                # ladder, not just slot 0, and (phase, plen) is the
                # window it certifies. The ranker pairs each observation
                # with the control that measured the SAME window.
                "control": not any(pat.masks),
                "plen": len(pat.masks),
                "contact": contact,
                # DEATH-AWARE PAIRING. `alive` is False when the
                # profile's own liveness reading changed anywhere inside
                # THIS observation's differential window — which is the
                # only window it can speak for. Unreadable liveness (a
                # profile that declares no lives byte) leaves every
                # sample None, the comparison is vacuously equal, and the
                # behaviour is the pre-repair one.
                "alive": len({v for v in (lives[wid].get(t0),
                                          lives[wid].get(t1),
                                          lives[wid].get(t2))}) == 1,
                "lives": [lives[wid].get(t0), lives[wid].get(t1),
                          lives[wid].get(t2)],
                "ram0": caps[wid][t0], "ram1": caps[wid][t1],
                "ram2": caps[wid][t2],
            })
        return out

    @staticmethod
    def _gate_rank(observations, novelty=None, farmability=None,
                   m=None, tie_seed=0, stats=None) -> list:
        """PURE differential ranking — see interaction_basis.rank_candidates
        for the pipeline (NOOP subtraction, cross-root invariance,
        persistence class, BH-FDR, novelty x controllability x
        (1-farmability)). `m` is the sweep-wide BH denominator, so the
        wall and its sham null are charged against the same grid. A
        staticmethod so it is testable with hand-built captures and
        cannot reach any live solver state."""
        return ib.rank_candidates(observations, novelty=novelty,
                                  farmability=farmability, m=m,
                                  tie_seed=tie_seed, stats=stats)

    def _gate_admit(self, ranked, roots) -> None:
        """Record the sweep's verdict: counters, the K1 sham null, and a
        per-candidate receipt — WHETHER OR NOT anything was admitted.
        Admission DISARMS the sweep until the frontier pins again — an
        instrument that keeps firing after it has found something is
        spending the budget it was granted to find it.

        THE NULL SWEEP'S RECEIPT IS THE PRE-REGISTERED PRIMARY OUTCOME,
        so it is written on exactly the same path as a hit. A sweep that
        ran every pattern in the basis, at every phase, from every root,
        and admitted nothing IS the registered result of K0's blind
        grade; skipping the write left it as a counter in a progress line
        and nothing else, which after the fact is indistinguishable from
        "the instrument never ran", "the sweep threw" and "somebody
        deleted the receipt". Same filename series, same fields, empty
        `admitted` — a null that can be read, checked and cited.

        AND THE RANKED LIST IS RECEIPTED IN FULL (repair D-TRUNC). The
        first build wrote ranked[:32]. A grade whose whole question is
        "where did this address rank" cannot be settled from a truncated
        list: the graded Castlevania sweep produced 2,135 candidate rows
        and the receipt could only say that $0004 was not in the top 32 —
        it could not distinguish "ranked 33rd" from "never ranked at all",
        and the two are opposite verdicts about the instrument. So the
        head stays inline for a human reader, and beside it go the total,
        the rank of EVERY address that ranked, the full admissible list
        with its own ranks, and the complete ranked table gzipped next to
        the receipt with its sha.
        """
        self._gate_counters["candidates"] += len(ranked)
        self._gate_counters["cross"] += sum(1 for r in ranked
                                            if r["roots"] >= 3)
        admitted = ib.admitted_candidates(ranked)
        for r in admitted:
            if r["addr"] not in self._gate_admitted:
                self._gate_admitted.append(r["addr"])
        if admitted:
            self._gate_counters["admitted"] += len(admitted)
            self._gate_queue_injections(admitted)
            self._gate_disarmed = True
        d = self.out / "gate"
        d.mkdir(parents=True, exist_ok=True)
        n = self._gate_counters["sweeps"]
        full_name = f"ranked_{n:03d}.json.gz"
        # mtime=0 so the sha below is a function of the RANKING and not
        # of the wall clock: two runs that ranked identically must
        # produce identical receipt shas, or the sha cannot be used to
        # check that they did.
        blob = gzip.compress(
            (json.dumps([dict(r) for r in ranked]) + "\n").encode(), mtime=0)
        (d / full_name).write_bytes(blob)
        admissible = ib.admissible_rows(ranked)
        (d / f"candidates_{n:03d}.json").write_text(json.dumps({
            "sweep": n, "roots": len(roots),
            "sham_roots": sum(1 for _c, s in roots if s),
            # K1's null, as it will be read by the kill: null (never
            # measured) is not 0.0 (measured clean), and the counts the
            # ratio is built from are on the receipt beside it.
            "sham_yield": self._gate_counters["sham_yield"],
            "sham_hits": self._gate_counters["sham_hits"],
            "wall_hits": self._gate_counters["wall_hits"],
            "basis": len(self._gate_basis),
            "phases": list(self._gate_phases),
            "program_len": ib.PROGRAM_LEN_PASS_A,
            "fdr_q": ib.FDR_Q, "rank_cutoff": ib.RANK_CUTOFF,
            # The BH denominator every verdict above was charged against.
            # It rides on each ranked row too, but a NULL sweep has no
            # rows — and the one thing a reader of a null has to be able
            # to check is the bar it was judged against.
            "fdr_m": getattr(self, "_gate_fdr_m", None),
            # The arbitrary half of the total order, named. Without it
            # the ranking is not reproducible from the receipt, and the
            # whole point of seeding the tie-break rather than fixing it
            # is that it is a declared property of the run.
            "tie_seed": int(getattr(self.args, "seed", 0)),
            "sort_key": ("score desc, significant first, unrefused first, "
                         "effect_size desc, seeded tie_key, (addr, family)"),
            # How the farmability column was measured, so a reader can
            # reproduce it (and can tell a null from a zero). BOTH
            # sources, separately: "the live band sampler never fired and
            # the sweep carried the reading" is the fact the graded run
            # could not state.
            "farm_samples": int(self._gate_change_n),
            "farm_stride": ib.BAND_SAMPLE_STRIDE,
            "farm_min_samples": ib.FARM_MIN_SAMPLES,
            "farm_min_steps": ib.FARM_MIN_STEPS,
            "farm_sources": [
                {"name": name, "samples": int(cnt), "stride": int(stride),
                 "steps": int(cnt) * int(stride)}
                for name, _counts, cnt, stride in self._gate_farm_sources()],
            "farm_steps_total": sum(
                int(cnt) * int(stride)
                for _n, _c, cnt, stride in self._gate_farm_sources()),
            # The prior, and where it came from. novelty == 1.0 across a
            # whole receipt is the D-PRIOR signature and it has to be
            # readable off the artifact.
            "boundary_hist_total": int(self._boundary_hist_total),
            "boundary_frames_sweep": int(self._gate_baseline_frames),
            # DISTINCT vs STEPPED. Every program at a root replays the
            # same undirected frames, so the second number is the size of
            # the pseudo-replication the refusal floor is NOT charged in.
            # A reader who sees frames_sweep below FARM_MIN_STEPS knows
            # the farmability column had to be refused, and can see how
            # much of the old denominator was replay.
            "boundary_frames_replayed": int(
                self._gate_counters["baseline_dupes"]),
            "boundary_roots": len(self._gate_baseline_len),
            # What the differential declined to read, and why.
            "differential": dict(getattr(self, "_gate_rank_stats", {}) or {}),
            # D-TRUNC: the head, the total, and the pointers that make
            # "absent" distinguishable from "ranked low".
            "ranked": [dict(r) for r in ranked[:32]],
            "ranked_total": len(ranked),
            "ranked_head": 32,
            "ranked_full": full_name,
            "ranked_full_sha256": hashlib.sha256(blob).hexdigest(),
            "ranked_addr_rank": _first_rank_by_addr(ranked),
            "admissible_total": len(admissible),
            "admissible_addr_rank": _first_rank_by_addr(admissible),
            # ADDRESSES, not rows (D-DUP): `rank <= 5` is defined on
            # addresses and the sidecar promotes addresses.
            "admitted": [r["addr"] for r in admitted],
            "queued_injections": [list(e) for e in self._gate_inject],
        }, indent=2) + "\n")

    def _gate_queue_injections(self, admitted) -> None:
        """Arm C's WRITER: turn admitted candidates into programs the
        search will actually run.

        Without it, the injection branch in explore(), the gate_marks
        trace element and T2's ablation are reachable only from tests,
        arm C is arm B plus a step tax, and T3 has nothing to be
        sufficient about. The policy is the smallest one that is not
        invented: every admitted candidate whose winning pattern fits the
        shared macro slot is queued once, in rank order, and band-rooted
        bursts take them round-robin at assignment. It draws no
        randomness, so arm C's search stream stays comparable with arm
        B's, and §1's precedence (gate > transition > hold, in-flight
        never preempted) is enforced where the slot is acquired.

        The channel is (action index, hold), so it can only carry a
        constant hold of a DECLARED action. Every TAP duty and every
        COMBO mask is refused here and counted: the sweep can measure 120
        interactions on Castlevania and the search can be handed 60 of
        them. That gap is reported (gate_inexpressible), never absorbed.
        """
        for r in admitted:
            shot = None
            for slot in r.get("slots", ()):
                if 0 <= slot < len(self._gate_basis):
                    shot = ib.macro_injection(self._gate_basis[slot],
                                              self.bitmasks)
                    if shot is not None:
                        break
            r["inject"] = list(shot) if shot is not None else None
            if shot is None:
                self._gate_counters["inexpressible"] += 1
                continue
            entry = (int(shot[0]), int(shot[1]), int(r["addr"]))
            if entry not in self._gate_inject:
                self._gate_inject.append(entry)

    def explore(self) -> None:
        args = self.args
        game = self.game
        cap = game.progress_cap
        ctx = [self._assign(i) for i in range(args.workers)]
        acts = np.zeros(args.workers, dtype=np.uint8)
        self.t0 = last_progress = last_flush = time.time()
        self._stall = {"last_cells": 0, "last_t": self.t0, "flat_windows": 0}
        deadline = self.t0 + args.minutes * 60 if args.minutes > 0 else None
        while not self.stop:
            # Pin clock read ONCE per step rather than once per worker:
            # the inversion gate is a call now, so it can no longer
            # short-circuit the time.time() away. Against a 180 s
            # threshold a one-step-stale reading is noise, and this is
            # strictly cheaper than the per-worker read it replaces.
            _pin_elapsed = time.time() - self._pin_time
            for i, c in enumerate(ctx):
                # Heuristic inversion inside the self-measured saturation
                # window [pin-300, pin+60]: the frontier pin is where OUR
                # search saturates (live telemetry, not any external map);
                # inside it, sample from inverted weights so leftward /
                # downward entries get explored instead of pruned.
                # SATURATION-TRIGGERED inversion (fix 2026-07-25): always-on
                # inversion sabotaged standard levels — at 5-3's frontier the
                # solver sampled left/down exactly where a full-speed rightward
                # jump was needed (chain stall). The maze maneuver hunt now
                # arms only after the frontier has been pinned for
                # --inversion-pin-secs (default 180 s = the receipted
                # constant; negative disables the arm outright).
                _pin = self.max_gx_in_area.get(self.max_area, 0)
                _floor = (self._loop_dest_min
                          if self._loop_dest_min is not None
                          else _pin - 300)
                _w = (self.inv_weights
                      if inversion_armed(self._inv_pin_secs, _pin,
                                         c.get("gx", -1), _floor,
                                         _pin_elapsed)
                      else self.weights)
                # Discrete-transition gate (Kirby lesson): at the deep frontier,
                # inject an auto-derived door-entry maneuver (settle then HOLD
                # up) so a room/area advance can fire where coordinate progress
                # has stalled. Checked before the ordinary macro roll so it wins
                # the (shared) macro slot at the frontier; inert everywhere else.
                # GATE INJECTION takes the shared macro slot FIRST when a
                # sweep has queued a program for this worker (precedence
                # gate > transition > hold, see macro_slot_owner). It
                # costs no RNG draw — the program is a queued, receipted
                # candidate, not a sampled one — and it can never preempt
                # an in-flight macro, because every arm below is guarded
                # by the same `macro_left <= 0`.
                if (self.gate_mode != "off" and c.get("macro_left", 0) <= 0
                        and c.get("gate_macro")):
                    gi, gh = c.pop("gate_macro")
                    c["macro_a"], c["macro_hold"] = int(gi), int(gh)
                    c["macro_left"] = c["macro_hold"] + 6
                    self._gate_counters["injections"] += 1
                    # The mark carries its own SPAN — the whole
                    # macro_left window it is about to own — because T2's
                    # ablation has no other way to know how many frames
                    # this injection wrote. A step index alone ablates
                    # one frame of a 114-frame program.
                    c.setdefault("gate_marks", []).append(
                        (int(c["steps"]), c.pop("gate_cand", None),
                         self._gate_marks_kind, int(c["macro_left"])))
                if (self.transition_macros and c.get("macro_left", 0) <= 0
                        and c.get("at_frontier")
                        and self.rng.random() < self.transition_p):
                    ti = int(self.rng.integers(len(self.transition_macros)))
                    c["macro_a"], c["macro_hold"] = self.transition_macros[ti]
                    c["macro_left"] = c["macro_hold"] + 6
                    self._transition_injections += 1
                # T3 route-follow OR-term (§3 row 9): a ROUTED worker
                # standing near its target side rolls that side's
                # direction-hold macro at the same room_advance p as
                # the door gate — the sustained push through an
                # unexplored boundary that stochastic sampling almost
                # never emits. route_dir exists only on router-tagged
                # bursts (--room-bias > 0), so the default path pays
                # one dict get and draws nothing.
                if (c.get("route_dir") is not None
                        and c.get("route_near")
                        and c.get("macro_left", 0) <= 0
                        and self._room_dir_macros.get(c["route_dir"])
                        and self.rng.random() < self.transition_p):
                    dm = self._room_dir_macros[c["route_dir"]]
                    c["macro_a"], c["macro_hold"] = dm[int(
                        self.rng.integers(len(dm)))]
                    c["macro_left"] = c["macro_hold"] + 6
                    self._room_route_injections += 1
                # Bursts rooted at an orthogonal-frontier cell roll the
                # hold macro at ortho_macro_p instead of the profile's own
                # rate: a stair mount IS a sustained up/up+right hold, and
                # the profile-declared macros fire at p=0.02, so the one
                # maneuver the climb needs is the rarest thing the sampler
                # emits exactly where it matters. Cheap checks first —
                # with the arm off (macro_p 0) this costs one comparison.
                _mp = (self.ortho_macro_p
                       if (self.ortho_macro_p > 0 and c.get("ortho")
                           and self._ortho_armed())
                       else self.macro_p)
                if (c.get("macro_left", 0) <= 0 and self.macros
                        and self.rng.random() < _mp):
                    mi = int(self.rng.choice(len(self.macros),
                                             p=self._macro_weights))
                    c["macro_a"], c["macro_hold"] = self.macros[mi][:2]
                    c["macro_left"] = c["macro_hold"] + 6   # settle, then hold
                if c.get("macro_left", 0) > 0:
                    c["macro_left"] -= 1
                    a = NOOP if c["macro_left"] >= c["macro_hold"] \
                        else c["macro_a"]
                else:
                    a = c["prev"] if self.rng.random() < args.sticky else \
                        int(self.rng.choice(len(_w), p=_w))
                c["prev"] = a
                c["pending"] = a
                acts[i] = self.bitmasks[a]
            results = self.pool.step_all(acts)
            if getattr(self, "_odo", False):
                self._odo_refresh()
            # One byte per worker, read straight after the step (state read,
            # no stepping, no side effects). None unless a profile's clear
            # hook asked for the audio modality, so the default hot loop is
            # untouched.
            apu_all = self.pool.apu_activity_all() if self._needs_apu else None
            self.steps_done += args.workers
            if self.step_hook is not None:
                try:
                    self.step_hook(results, self)
                except Exception:
                    pass
            for i, c in enumerate(ctx):
                # ROOM FINGERPRINTING (§3 row 4, the only hot-loop
                # insertion): settle+classify BEFORE _xram, which reads
                # _room_ord — so the pseudo-RAM ordinal, room_id() and
                # the transit below all see this frame's identity. One
                # attribute test when the profile has no room_fp.
                if self.room_fp is not None:
                    self._room_step(i, c)
                ram = (self._xram(results[i][2], i)
                       if getattr(self, "_odo", False)
                       or getattr(self, "_fight", False)
                       else results[i][2])
                if apu_all is not None:
                    c["_apu_mask"] = apu_all[i]
                c["trace"].append(c["pending"])
                c["steps"] += 1
                c["left"] -= 1
                c["burst_step"] += 1
                # Loop-back detection: a discontinuous backward gx jump on a
                # non-garbage frame advances the trajectory's maze phase
                # (capped so wrong-path spirals can't explode the archive).
                # Route signature: the y-band recorded at each 512-px gx
                # boundary the pass crosses (reset on loop-back) — the
                # observable footprint of THIS pass's route choice.
                _lgx = game.progress(ram)
                c["gx"] = _lgx if _lgx <= cap else c.get("gx", -1)
                # Discrete-transition gate: is this worker sitting AT the deep
                # frontier (deepest area reached, within `near` gx buckets of
                # its max)? If so the next step may inject a door-entry macro.
                # Only computed when the gate is armed (non-SMB room profiles).
                if self.transition_macros:
                    _fgx = self.max_gx_in_area.get(self.max_area, 0)
                    c["at_frontier"] = (
                        game.area(ram) == self.max_area and c["gx"] >= 0
                        and c["gx"] >= _fgx - self.transition_near * GX_BUCKET)
                    if self.room_advance_addr is not None:
                        _rv = int(ram[self.room_advance_addr])
                        if _rv > self.max_room:
                            self.max_room = _rv
                # T3 route-follow (§3 row 9): a routed burst tracks
                # whether it now stands within `near` cell units of its
                # target side's bbox edge — read by the next pre-step
                # macro roll exactly like at_frontier. route_dir exists
                # only on router-tagged bursts (--room-bias > 0), so
                # the default path pays one dict get here.
                if c.get("route_dir") is not None:
                    c["route_near"] = (
                        c["gx"] >= 0
                        and route_near_side(
                            c["route_dir"], c["route_bbox"],
                            c["gx"] // GX_BUCKET,
                            game.y(ram) // Y_BAND,
                            self.transition_near))
                # ROOM IDENTITY (SMB: verified 2026-07-26, $074E changes
                # 0.67/1k steps, only at full-screen transitions). A rid
                # CHANGE is a true room transition.
                if self.kill_key and self.entity_slots is not None:
                    _es = tuple(int(ram[a]) for a in self.entity_slots)
                    pes = c.get("eslots")
                    if pes is not None:
                        # nonzero->zero: CV's slots are 0/1 flags (same
                        # result); object-array engines (Contra) hold
                        # state values, so ==1 missed most despawns.
                        c["kills"] = c.get("kills", 0) + sum(
                            1 for p, q in zip(pes, _es) if p != 0 and q == 0)
                    c["eslots"] = _es
                    # Local mode: reset the counter whenever the
                    # trajectory sets a new progress high-water mark.
                    # Run-and-gun games kill constantly — cumulative kk
                    # saturates its cap long before the wall (Contra v3:
                    # 92% of cells at kk=15, ALL wall cells capped =
                    # zero gradient exactly where the fight is). With
                    # the reset, kk counts kills SINCE progress froze —
                    # at a fixed-camera fight, that IS the fight.
                    if self.kill_key_local:
                        if _lgx <= cap and _lgx > c.get("kk_max_gx", -1):
                            c["kk_max_gx"] = _lgx
                            c["kills"] = 0
                _rid = game.room_id(ram)
                _transit = (c["p0750"] is not None and _rid != c["p0750"])
                if _transit and c["sect"] < self.args.sect_cap:
                    c["sect"] += 1
                    c["psig"] = _rid
                # Room-graph edge commit (§3 row 5): transit logic above
                # is UNTOUCHED — this only consumes the edge the settle
                # staged this same step. Not gated on sect_cap: the
                # graph keeps learning adjacency after psig saturates.
                if _transit and self.room_fp is not None:
                    self._room_transit(c)
                c["p0750"] = _rid
                if _lgx <= cap:
                    if c["prev_gx"] >= 0:
                        if _transit:
                            c["prev_gx"] = _lgx   # section change: re-arm, no loop
                        elif _lgx < c["prev_gx"] - 100:
                            if c["loops"] < 8:
                                c["loops"] += 1
                            c["sig"] = ()   # new pass, fresh route
                        elif _lgx // 512 != c["prev_gx"] // 512:
                            c["sig"] = (c["sig"] + (game.y(ram) // 64,))[-4:]
                    c["prev_gx"] = _lgx
                if (self.args.swim_gx_ceiling > 0 and game.swim(ram) == 1
                        and _lgx <= cap
                        and _lgx > self.args.swim_gx_ceiling):
                    ctx[i] = self._assign(i, prev=c)
                    continue
                status = self.observe(i, ram, c["trace"], c["steps"],
                                      c["root"], c["loops"], c["sig"],
                                      c["sect"], c["psig"], ctx=c,
                                      kills=c.get("kills", 0))
                # Finisher extension: the level-END transition (exit pipe /
                # flag slide) can run many steps with gx frozen, so a burst
                # from the deepest cell can end just short of the wd advance.
                # A burst ending in the deepest-area top band gets one +200
                # extension so it can actually complete the clear.
                if (finisher_extension_ok(status, c)
                        and game.area(ram) == self.max_area
                        and _lgx // 16 >= self.max_gx_in_area.get(self.max_area, 0) // 16 - 3):
                    c["left"] += 200
                    c["extended"] = True
                if status != "live" or c["left"] <= 0 or c["steps"] >= args.max_steps:
                    if status == "dead":
                        # burst_step, never `args.burst - left`: the finisher
                        # extension above adds 200 to `left`, which drives
                        # that derivation negative and DOA-retires the
                        # extension's own root cell in _assign().
                        c["died_at_burst_step"] = c["burst_step"]
                    # A burst just completed: this is the only moment the
                    # gate sweep may fire (nothing in flight is discarded).
                    self._gate_boundary_hit = True
                    ctx[i] = self._assign(i, prev=c)
            now = time.time()
            # The band-growth checkpoint runs on its OWN cadence and
            # BEFORE the sweep poll, so the arming conjunct it feeds is
            # current on the step the sweep is decided (repair D-ARM: the
            # checkpoints used to be minted by the 60 s progress line, so
            # nothing could arm before four minutes had passed).
            if (self.gate_mode != "off"
                    and now - self._gate_last_ckpt >= self.gate_arm_cadence):
                self._gate_checkpoint(now)
            if self.gate_mode != "off" and self._gate_boundary_hit:
                self._gate_boundary_hit = False
                self._gate_maybe_sweep(ctx, now, deadline)
                # Re-read the clock: a sweep is thousands of steps long,
                # so every check below (progress cadence, flush cadence
                # and the deadline itself) would otherwise be deciding
                # on a reading taken before it started.
                now = time.time()
            self._maybe_scan_doors(now)
            if now - last_progress >= 60:
                last_progress = now
                self.progress_line(now - self.t0)
            if now - last_flush >= args.flush_secs:
                last_flush = now
                self._maybe_flush_async()
            if deadline and now >= deadline:
                break
            if not keep_exploring(self.n_solutions, args.want_solutions):
                break

    # ---- reporting / persistence -------------------------------------

    def progress_line(self, elapsed: float) -> None:
        # Flat-archive stall watchdog (ported from live_solve_show.py,
        # 2026-08-06): this standalone driver runs most of the fleet's
        # actual compute but had no equivalent — a run in this exact
        # 60s-cadence path died silently and went unnoticed for 32+ min
        # during the review that flagged this gap. Two consecutive 60s
        # windows (this method's own cadence) with zero new cells warns
        # loudly; it cannot detect the process dying outright (nothing
        # runs after that), only an in-process hang/freeze while alive.
        n_cells = len(self.archive)
        update_stall(self._stall, n_cells, time.time())
        if self._stall["flat_windows"] >= 2:
            sys.stderr.write(
                f"[go_explore_solve] STALL WARNING: stuck at {n_cells} "
                f"cells for {self._stall['flat_windows']} min straight — "
                f"possible frozen/dead state\n")
        line = {
            "schema_version": 1,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed),
            "cells": n_cells,
            "max_area": self.max_area,
            "max_gx_in_max_area": self.max_gx_in_area.get(self.max_area, 0),
            "max_sect": self.max_sect,
            "solutions": self.n_solutions,
            "best_sol_actions": (self.best_sol_len if self.n_solutions else None),
            "steps": self.steps_done,
            "sps": round(self.steps_done / max(elapsed, 1e-9)),
            "stall_flat_windows": self._stall["flat_windows"],
        }
        # LOCAL COVERAGE (c_local) — unconditional, unlike every arm's
        # telemetry below, and deliberately so: it is a derivative, the
        # archive on disk holds exactly ONE reading of it (at the flush),
        # and the whole finding of size_decoupled_statistic_2026-08-11 §11
        # is that no cross-sectional statistic substitutes for the series
        # (22 candidates, 0 separate). It cannot be back-filled from a
        # banked run, so it has to be in every run's line from now on.
        # Peer of `cells`, not arm telemetry. It is READ by the wall
        # discriminator (it is MISSING_TELEMETRY entry #1) and adding it
        # to the line moves runs out of abstention — see the section
        # header for the measured before/after and for what stays struck.
        triples = getattr(self, "_c_local", None)
        if triples is not None:
            c_local = len(triples)
            c_band = local_coverage_band(triples)
            prev = getattr(self, "_c_local_prev", None)
            line["c_local"] = c_local
            line["c_local_band"] = c_band
            line["c_local_delta"] = None if prev is None else c_local - prev[0]
            line["c_local_band_delta"] = (None if prev is None
                                          else c_band - prev[1])
            elast = (None if prev is None else
                     coverage_elasticity(prev[0], prev[2], c_local, n_cells))
            line["c_local_elasticity"] = (None if elast is None
                                          else round(elast, 4))
            self._c_local_prev = (c_local, c_band, n_cells)
        if self.ortho_mode != "off":
            # ortho_deep_yband is the pre-registered gate figure: the
            # extreme y-band inside the band the PRIMARY arm samples, so
            # a climb that only happens off-frontier cannot flatter it.
            # pinned_secs is the arm's own arming clock, printed so a log
            # reader can tell "inert" from "armed and not working".
            line["ortho_best_yband"] = self._ortho_best
            line["ortho_deep_yband"] = self._ortho_deep_yband
            line["ortho_pool"] = len(self._ortho_pool)
            line["ortho_selections"] = self._ortho_selections
            line["ortho_cols_improved"] = self._ortho_cols_improved
            line["pinned_secs"] = round(time.time() - self._pin_time)
        if getattr(self, "gate_mode", "off") != "off":
            # band_cells is the bookkeeping the self-arming band-growth
            # conjunct reads (§3): one pass over the archive keys, and the
            # last few records are the whole input. READ-ONLY here. The history the arming conjunct reads and
            # the armed-secs clock are both rolled by _gate_checkpoint on
            # its own cadence (--gate-arm-cadence-secs); a progress line
            # that also appended would double-count the series the
            # conjunct measures growth over and make the arming floor a
            # function of how often the run happens to log.
            band = band_cell_count(self.archive.cells.keys(),
                                   band=self.gate_band)
            armed = self._gate_armed()
            secs = self._gate_armed_secs + (
                time.time() - self._gate_armed_since
                if self._gate_armed_since is not None else 0.0)
            line["band_cells"] = band
            line["gate_armed"] = bool(armed)
            line["gate_armed_secs"] = round(secs)
            line["gate_sweep_steps"] = self._gate_counters["steps"]
            for k in ("sweeps", "programs", "candidates", "admitted",
                      "cross", "injections", "inexpressible", "sham_yield",
                      "sham_roots", "sham_hits", "wall_hits",
                      "lift_active", "lift_ctrl", "restore_failed"):
                line[f"gate_{k}"] = self._gate_counters[k]
            line["gate_shadow_yield"] = round(self.shadow_yield(), 4)
            # The farmability denominators, so a reader can tell "measured
            # and clean" from "never measured" without opening a receipt
            # — and can tell WHICH sampler carried it. gate_farm_samples
            # pinned at 0 for 24 minutes was D-PRIOR's visible symptom,
            # and it was visible only in the receipt.
            line["gate_farm_samples"] = int(self._gate_change_n)
            line["gate_farm_samples_sweep"] = int(self._gate_change_sweep_n)
            line["gate_prior_frames"] = int(self._boundary_hist_total)
            prof = self._gate_axes_live
            if prof is not None:
                line["boundary_state_axes"] = prof.get("live_state_axes_n")
                line["alias_ratio"] = prof.get("alias_ratio")
        if self.door_weight > 0:
            line["doors"] = len(self._doors)
            line["edges"] = sum(len(v) for v in self._adj.values()) // 2
        if self.transition_macros:
            line["door_macros_injected"] = self._transition_injections
            if self.room_advance_addr is not None:
                line["max_room"] = self.max_room
        # ROOM-GRAPH telemetry (§3 row 12): printed whenever the profile
        # fingerprints, zero or not — "armed and never fired" and "never
        # armed" are different readings (the torn-read lesson above).
        # getattr guards: duck-typed progress_line stand-ins in the
        # tests predate the room attrs.
        if (getattr(self, "room_fp", None) is not None
                and getattr(self, "room_index", None) is not None):
            idx = self.room_index
            with idx.lock:
                kinds = [e["kind"] for dsts in idx.adj.values()
                         for e in dsts.values()]
                line["rooms"] = len(idx.ordinals)
                line["edges_pan"] = sum(1 for k in kinds if k == "pan")
                line["edges_fade"] = sum(1 for k in kinds if k == "fade")
                line["warps_vetoed"] = idx.warp_count
                line["aliased"] = sum(1 for m in idx.meta.values()
                                      if m.get("aliased"))
                line["room_cap_hits"] = idx.cap_hits
            line["settle_rejects"] = getattr(self, "_room_settle_rejects", 0)
            line["artic"] = len(getattr(self, "_room_artic", ()))
            line["router_picks"] = getattr(self, "_room_router_picks", 0)
            line["route_macros_injected"] = getattr(
                self, "_room_route_injections", 0)
            line["room_edges_committed"] = getattr(
                self, "_room_edges_committed", 0)
            line["room_edges_dropped"] = getattr(
                self, "_room_edges_dropped", 0)
            line["room_restore_transits"] = getattr(
                self, "_room_restore_transits", 0)
        # Fabricated-clear telemetry: silent while nothing has been rejected
        # (so ordinary runs' progress lines are unchanged), loud the moment a
        # candidate fails to reproduce — the four-game detector gate reads
        # exactly this number.
        # Torn-read telemetry: silent unless the filter is on, so an
        # unfiltered run's line is unchanged, and printed even at zero
        # once it IS on — "armed and never fired" and "never armed" are
        # different readings and the phantom cost 10.7 h of not knowing.
        if (getattr(self, "progress_smooth", "off") != "off"
                or getattr(self, "_borrow_off", False)):
            line["progress_smooth"] = self.progress_smooth
            line["progress_glitches"] = self._prog_glitches
            # The pair audit, whenever the borrow rule has ruled on
            # anything: "the filter is armed and firing" and "the filter
            # is armed and wrong about this profile's address pair" are
            # different runs, and only the second one disarms itself.
            if (getattr(self, "_borrow_tears", 0)
                    or getattr(self, "_borrow_flat", 0)
                    or getattr(self, "_borrow_off", False)):
                line["borrow_tears"] = self._borrow_tears
                line["borrow_refuted"] = self._borrow_flat
                line["borrow_disarmed"] = bool(self._borrow_off)
        if getattr(self, "verify_rejections", 0):
            line["verify_checks"] = self.verify_checks
            line["verify_rejections"] = self.verify_rejections
        # Counterfactual-gate telemetry: printed as soon as the gate has run
        # at all (unlike the replay counters, which stay silent until
        # something is rejected) — the gate is opt-in, so a reader who turned
        # it on wants to see that it is actually being exercised.
        if getattr(self, "cf_checks", 0):
            line["cf_checks"] = self.cf_checks
            line["cf_rejections"] = self.cf_rejections
            # Of those rejections, the ones where the hook fired on its own
            # rolling state rather than on a game event (verdict
            # state_artifact) — and how many candidates needed the deep
            # re-snapshot at all, so "the retry never runs" is visible
            # instead of inferred.
            line["cf_state_artifacts"] = getattr(self, "cf_state_artifacts", 0)
            line["cf_retries"] = getattr(self, "cf_retries", 0)
        with open(self.out / "progress.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[go_explore_solve] {json.dumps(line)}", flush=True)

    def flush(self) -> None:
        # A background _flush_async_qos daemon (see _maybe_flush_async) may
        # still be mid-write to the SAME archive.pkl.tmp path this writes
        # to — join it first so the two writers never race the same inode
        # (found by adversarial review, 2026-08-06: explore() can break
        # with a flush thread in flight, and this synchronous flush is the
        # last-chance persist on exit, so it's always correct to wait).
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._flush_thread.join()
        self.archive.save(self.out / "archive.pkl")
        stamp_stats_provenance(self.out / "archive.pkl", self.provenance,
                               self.key_config())
        with open(self.out / "traces.pkl", "wb") as f:
            pickle.dump(self.traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        (self.out / "roots.json").write_text(json.dumps(self.roots, indent=2) + "\n")
        # The intern table travels with the archive (§2 persistence):
        # ordinals embedded in the flushed keys are meaningless without
        # it, and _resume_room_index refuses a resume that lacks it.
        if self.room_index is not None:
            self.room_index.save(self.out / "room_index.json")

    def _maybe_flush_async(self) -> None:
        """Non-blocking periodic flush: a multi-GB archive.pkl pickle dump
        run synchronously on the hot loop is the receipted cause of a
        1563->322 sps stall (src/training/qos.py's docstring). Take a
        cheap shallow-copy snapshot HERE on the solve thread (a dict()
        copy is a single bulk operation under the GIL — safe against the
        'dict changed size during iteration' race a live multi-GB dict
        would hit if handed directly to another thread), then do the slow
        pickling on a QOS_CLASS_BACKGROUND daemon thread so it lands on
        the E-cores and never evicts a P-core worker's cache lines. Skips
        (rather than queuing) if the previous flush hasn't finished yet —
        the search keeps running either way; the next interval tries
        again with fresher data."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        cells_snapshot = dict(self.archive.cells)
        traces_snapshot = dict(self.traces)
        roots_snapshot = dict(self.roots)
        counters = (self.archive.total_records, self.archive.total_new_cells,
                    self.archive.total_improvements)
        self._flush_thread = threading.Thread(
            target=self._flush_async_qos,
            args=(cells_snapshot, traces_snapshot, roots_snapshot, counters),
            daemon=True)
        self._flush_thread.start()

    def _flush_async_qos(self, cells_snapshot, traces_snapshot,
                         roots_snapshot, counters) -> None:
        from src.training.qos import demote_current_thread
        demote_current_thread()
        try:
            snap = GoExploreArchive(self.game.cell_fn, seed=self.args.seed)
            snap._cells = cells_snapshot
            snap.total_records, snap.total_new_cells, snap.total_improvements = counters
            snap.save(self.out / "archive.pkl")
            stamp_stats_provenance(self.out / "archive.pkl", self.provenance,
                                   self.key_config())
            if getattr(self, "gate_mode", "off") != "off":
                # Taxonomy is KEYED, never WIRED: only the PURE per-axis
                # profile runs, on the flush SNAPSHOT, on this background
                # thread — never in the hot loop. Nothing that CLASSIFIES
                # a wall is reachable from runtime; the verdict stays an
                # operator-read artifact between sessions.
                from src.training.wall_taxonomy import boundary_axis_profile
                p = boundary_axis_profile(cells_snapshot,
                                          band=self.gate_band,
                                          bookkeeping=(4, 5))
                self._gate_axes_live = {
                    "live_state_axes_n": p.live_state_axis_count,
                    "alias_ratio": round(p.alias_ratio, 3),
                    "interaction_blind": p.interaction_blind,
                }
            with open(self.out / "traces.pkl", "wb") as f:
                pickle.dump(traces_snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            (self.out / "roots.json").write_text(
                json.dumps(roots_snapshot, indent=2) + "\n")
            # RoomIndex serializes a coherent snapshot under its own
            # lock (append-only + to_json holds it), so the background
            # flush needs no extra coordination with the hot loop.
            if self.room_index is not None:
                self.room_index.save(self.out / "room_index.json")
        except Exception as e:
            print(f"[go_explore_solve] background flush failed (search "
                  f"continues, will retry next interval): {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-state", required=True,
                    help="Level ENTRANCE save-state (honest root; no prefix).")
    ap.add_argument("--profile", required=True,
                    help="Source of action_space + frame_skip only.")
    ap.add_argument("--hw-flags", type=str, default=None, metavar="a,b,c",
                    help="Comma-separated nes_core hw timing flags to set on "
                         "the pool (e.g. mmio_read_timing,nmi_poll_timing), "
                         "overriding the profile's solve.hw_flags. "
                         "'none' forces the empty set. DEFAULT: whatever the "
                         "profile pins, else NONE — a state blob records no "
                         "timing config, so a lineage built under flags must "
                         "be re-solved under the same flags or the restored "
                         "machine is not the machine that produced it.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--burst", type=int, default=64)
    ap.add_argument("--sticky", type=float, default=0.5)
    ap.add_argument("--deep-bias", type=float, default=0.4)
    ap.add_argument("--minutes", type=float, default=0,
                    help="Wall-clock budget (0 = until solution quota).")
    ap.add_argument("--want-solutions", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--flush-secs", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-bank", dest="verify_bank", action="store_true",
                    default=True,
                    help="Re-simulate every clear candidate from its root in "
                         "a fresh pool and refuse to bank it unless the clear "
                         "reproduces (DEFAULT ON; measured ~2.7 ms/action, "
                         "i.e. seconds per candidate against a 25-45 min "
                         "solve).")
    ap.add_argument("--no-verify-bank", dest="verify_bank",
                    action="store_false",
                    help="Bank clear candidates unverified (pre-2026-08-08 "
                         "behavior). Receipts written this way record "
                         "replay_verified: false.")
    ap.add_argument("--counterfactual-gate", dest="counterfactual_gate",
                    action="store_true", default=False,
                    help="Additionally require every clear candidate to "
                         "survive a K-branch input-perturbation probe before "
                         "banking (DEFAULT OFF). Re-simulates K short "
                         "branches from a pre-clear state with perturbed "
                         "inputs: a committed stage transition reproduces "
                         "across most of them, a combat blip or one-off RAM "
                         "coincidence does not. Costs seconds per candidate "
                         "(measured 4-40 s), which is why it lives on the "
                         "banking path and nowhere near the hot loop.")
    ap.add_argument("--no-counterfactual-gate", dest="counterfactual_gate",
                    action="store_false",
                    help="Explicitly disable the counterfactual gate (the "
                         "default).")
    ap.add_argument("--cf-branches", type=int, default=8,
                    help="K: perturbed branches per candidate (default 8).")
    ap.add_argument("--cf-pre-steps", type=int, default=32,
                    help="Actions before the end of the trace at which the "
                         "pre-clear snapshot is taken; the branches perturb "
                         "exactly this tail (default 32 ~= 2.1 s at "
                         "frame_skip 4). Raised automatically to cover a "
                         "windowed hook's warm-up, and re-taken ONCE at 4x "
                         "this depth if the unperturbed control could not "
                         "see the clear from it: a clear that no snapshot "
                         "inside the trajectory can reproduce is refused as "
                         "a state_artifact.")
    ap.add_argument("--cf-perturb-p", type=float, default=0.25,
                    help="Per-action probability that a branch replaces the "
                         "trace's action with a random one (default 0.25). "
                         "Higher = harsher; high enough and it deletes "
                         "inputs a real clear needed.")
    ap.add_argument("--cf-agree", type=float, default=0.5,
                    help="Fraction of branches that must still reach the "
                         "clear for the candidate to be banked (default 0.5 "
                         "= most of them).")
    ap.add_argument("--cf-seed", type=int, default=None,
                    help="Seed for the perturbation streams (default: the "
                         "run's --seed, so the probe is reproducible).")
    ap.add_argument("--swim-gx-ceiling", type=int, default=0,
                    help="If >0: in swim rooms ($001D=1), lineages crossing "
                         "this gx are terminated — forces attempt density "
                         "onto the section's floor pipes instead of the "
                         "scroll-buffer wrap (8-4 water separating expt).")
    ap.add_argument("--sect-cap", type=int, default=16,
                    help="Cap on the room-transition counter (`sect`, part "
                         "of the cell key) a lineage can accumulate; room_id() "
                         "stops updating `psig` past this count. Sized for "
                         "SMB1's 8-4 (~5 rooms, room_id churns 0.67/1k steps) "
                         "— default preserves the exact byte-identical "
                         "behavior of the verified 32-level SMB clear. Raise "
                         "this for ROMs where room_id() churns faster (e.g. "
                         "Lost Levels measured 1.15-8.8/1k steps), or the cap "
                         "saturates within the first burst and every "
                         "post-saturation state collapses onto one frozen "
                         "cell-key slot for the rest of the run.")
    ap.add_argument("--gx-bucket", type=int, default=16,
                    help="Cell gx granularity px (micro-search: 8).")
    ap.add_argument("--y-band", type=int, default=32,
                    help="Cell y granularity px (micro-search: 16).")
    ap.add_argument("--sel-mode", choices=("legacy", "count"),
                    default="legacy",
                    help="count = Go-Explore count-based selection prior "
                         "via O(1) rejection sampling (maze-coverage R1).")
    ap.add_argument("--frontier-throttle", type=int, default=0,
                    help="If >0: exclude cells whose bursts came back "
                         "empty this many times in a row from the deep-"
                         "frontier band (DD-RRT boundary suppression, R2).")
    ap.add_argument("--door-weight", type=float, default=0.0,
                    help="If >0: up-weight archive cells that are cut "
                         "vertices (doors) of the rollout transition graph "
                         "by this factor in the count arm (Hopcroft-Tarjan "
                         "sub-goals, R4). Needs --sel-mode count. 5 = "
                         "research-recipe default.")
    ap.add_argument("--door-interval", type=float, default=45.0,
                    help="Seconds between async door (articulation-point) "
                         "recomputations (R4).")
    ap.add_argument("--room-bias", type=float, default=0.0,
                    help="If >0: probability the selection router picks a "
                         "ROOM first (frontier / articulation / unexplored-"
                         "boundary weighted, pure count prior) and then a "
                         "cell inside it (room-graph engine; needs "
                         "solve.room_fp). 0 = router arm off, the default — "
                         "the identity layer still runs whenever the "
                         "profile declares room_fp.")
    ap.add_argument("--room-artic-weight", type=float, default=2.0,
                    help="Router weight bonus for rooms that are "
                         "articulation points of the room adjacency graph "
                         "(every route between the regions they separate "
                         "passes through them).")
    ap.add_argument("--room-exit-weight", type=float, default=1.0,
                    help="Router weight per unexplored side U(r): sides "
                         "with boundary cells but no out-edge of that "
                         "direction.")
    ap.add_argument("--room-recent-k", type=int, default=4,
                    help="Frontier set includes rooms with ordinal >= "
                         "max_ord - K (discovery-recency band).")
    ap.add_argument("--item-sig-report", action="store_true",
                    help="Report-only (ITEM_SEMANTICS_ENGINE_2026-08-25): "
                         "tag each room-graph edge's cap_hist with the "
                         "profile's state_sig bit-vector at the moment "
                         "the edge staged, for offline discover_item_"
                         "bits.py analysis. No effect on selection, "
                         "scoring, or cell keys. Needs solve.state_sig "
                         "(GenericGame profiles only). Default off.")
    ap.add_argument("--time-bins", action="store_true",
                    help="Append floor(log2(steps)) to the key prefix "
                         "(Time-Myopic Go-Explore, v6 recipe 4): slow "
                         "structural paths survive domination; wait-gates "
                         "become searchable. Old-format archives resume "
                         "with duplicated (still-valid) cells.")
    ap.add_argument("--kill-key", action="store_true",
                    help="Append capped cumulative entity kills (profile "
                         "solve.entity_slots {lo,hi} flag bytes, observed "
                         "1->0) to the key prefix — prerequisite/kill-count "
                         "gates become a searchable dimension (v6 mech B).")
    ap.add_argument("--resume-archive", type=str, default=None, metavar="DIR",
                    help="Resume from a prior run's flushed out-dir: load "
                         "archive.pkl + traces.pkl + roots.json and continue "
                         "exploring instead of starting from one root cell. "
                         "Iterating on a single wall keeps every hard-won "
                         "frontier cell (e.g. CV block-3's stair funnel). "
                         "The resumed archive's recorded lineage (hw flags, "
                         "frame_skip, core sha, cell-key schema) is compared "
                         "against this run's before anything is loaded.")
    ap.add_argument("--allow-unverified-lineage", action="store_true",
                    help="Proceed with --resume-archive when an axis could "
                         "not be CHECKED at all (the archive records no "
                         "hw_provenance and none can be recovered from its "
                         "roots.json or root sidecars, or its keys leave a "
                         "schema axis ambiguous). OFF by default: 'nothing "
                         "recorded' is not 'matches', and the §12 archive "
                         "was resumed on exactly that reading. Implied by "
                         "--allow-lineage-mismatch.")
    ap.add_argument("--no-resume-refreeze", dest="resume_refreeze",
                    action="store_false", default=True,
                    help="Keep banked torn-read frontier cells in the "
                         "frontier readers on --resume-archive. By default "
                         "a resumed gx-bucket ISLAND — occupied, separated "
                         "from the body of the frontier by an empty bucket, "
                         "holding <2%% of it — is excluded from max_gx and "
                         "_sel_topgx (it stays in the archive). That is the "
                         "gx-767 phantom: 14 of ortho_ctrl's 149,153 cells "
                         "pinned every resumed run's frontier at a position "
                         "the game was never in (§12).")
    ap.add_argument("--allow-lineage-mismatch", action="store_true",
                    help="Proceed with --resume-archive even when the "
                         "resumed archive's hw_provenance or cell-key "
                         "schema disagrees with this run. OFF by default: "
                         "the silent version of this cost the campaign a "
                         "resume across a 1-flag lineage into a 4-flag run "
                         "over a disjoint key subspace, where 560,410 cells "
                         "collapsed to 88,212 on a like-for-like recount "
                         "(GATE_OPENER_CAMPAIGN_2026-08-11 §12).")
    ap.add_argument("--inversion-pin-secs", type=float, default=180.0,
                    metavar="SECS",
                    help="Seconds the deep frontier must sit pinned before "
                         "the heuristic-inversion arm engages (leftward/"
                         "downward sampling inside the self-measured "
                         "saturation window). DEFAULT 180 = the constant "
                         "the verified 32-level SMB clear ran under, so the "
                         "banked receipts stay reproducible. Lower it on "
                         "walls that saturate early; -1 (any negative) "
                         "disables the arm entirely — the sampler keeps the "
                         "forward weights no matter how long the frontier "
                         "stays pinned.")
    # --- orthogonal (vertical) progress arm ------------------------------
    # The inversion arm above is horizontal: it un-prunes left/down when
    # gx saturates. On vertically-structured walls (shafts, climbs) the
    # search needs the mirror image — treat one vertical direction as the
    # progress axis. Every knob below is inert while --ortho is off (the
    # default), so a run that omits them samples exactly as before and
    # every banked SMB receipt's solver_args replays unchanged.
    ap.add_argument("--ortho", choices=("off", "up", "down"), default="off",
                    help="Orthogonal (vertical) progress axis for the "
                         "saturation arm: 'up' or 'down' names the "
                         "direction to treat as forward once horizontal "
                         "progress is pinned. DEFAULT off = no vertical "
                         "arm, sampling is bit-identical to the receipted "
                         "campaign.")
    ap.add_argument("--ortho-pin-secs", type=float, default=120.0,
                    metavar="SECS",
                    help="Seconds the frontier must sit pinned before the "
                         "ortho arm engages (the vertical counterpart of "
                         "--inversion-pin-secs; shorter by default because "
                         "a shaft saturates faster than a run of ground).")
    ap.add_argument("--ortho-bias", type=float, default=0.30, metavar="P",
                    help="Probability that an armed selection restarts a "
                         "worker from the ortho pool (top-of-column cells) "
                         "instead of falling through to the ordinary arms. "
                         "Sampled with a PURE count prior — no score term, "
                         "since the score is what buries a climb. 0 = "
                         "never.")
    ap.add_argument("--ortho-band", type=int, default=1, metavar="N",
                    help="Half-width, in y-bands (--y-band px each), of the "
                         "window around EACH gx column's own ortho extreme "
                         "that enters the pool. 1 = the column's frontier "
                         "band plus its two neighbours.")
    ap.add_argument("--ortho-weight", type=float, default=4.0, metavar="W",
                    help="Multiplier applied to pooled cells inside the "
                         "count arm's rejection sampling while the ortho "
                         "arm is armed (Wmax scales with it, so the prior "
                         "stays exact). 1.0 = no bias. Needs --sel-mode "
                         "count to have any effect.")
    ap.add_argument("--ortho-macro-p", type=float, default=0.0, metavar="P",
                    help="Per-step probability of injecting a profile-"
                         "declared sustained-hold macro in bursts ROOTED at "
                         "an ortho-pool cell (the long consecutive holds a "
                         "climb needs and stochastic sampling almost never "
                         "emits; profile rates are ~0.02). DEFAULT 0 = off, "
                         "so the macro slot behaves exactly as today.")
    # --- gate-opener arm --------------------------------------------------
    # The interaction enumerator. Every knob is inert while
    # --gate-opener is off (the default), so a run that omits them all
    # samples exactly as the receipted campaign did. --gate-axes is
    # DELIBERATELY independent of the mode: the A/B control arm must
    # merge the same sidecar as the armed arm, or the two runs partition
    # cells differently and nothing they report is comparable.
    ap.add_argument("--gate-opener", choices=("off", "enumerate"),
                    default="off",
                    help="Gate-opener arm: 'enumerate' runs the "
                         "deterministic interaction basis off archive "
                         "blobs at the pinned boundary and ledgers the "
                         "candidates it finds. DEFAULT off = no sweep, "
                         "sampling bit-identical to the receipted "
                         "campaign.")
    ap.add_argument("--gate-sweep-frac", type=float, default=0.10,
                    metavar="F",
                    help="Fraction of the step budget the sweep may "
                         "consume; sets the cadence between sweeps. The "
                         "spend is also counted directly "
                         "(gate_sweep_steps), because sweep steps "
                         "increment steps_done and sps is therefore "
                         "blind to the tax.")
    ap.add_argument("--gate-sweep-roots", type=int, default=16, metavar="N",
                    help="Least-swept band cells used as sweep roots.")
    ap.add_argument("--gate-sweep-repeats", type=int, default=2, metavar="N",
                    help="Repeats of the whole basis per root (pass A).")
    ap.add_argument("--gate-pin-secs", type=float, default=None,
                    metavar="SECS",
                    help="Seconds the frontier must sit pinned before the "
                         "gate arm engages. REQUIRED whenever "
                         "--gate-opener is not off: v1 derives nothing "
                         "(no advance-interval history exists to derive "
                         "from, and a saturated wall can log zero "
                         "advances in a session). Mined defaults: 600 for "
                         "the CV hall, 120 for Bubble Bobble. Any "
                         "negative value disables the arm outright.")
    ap.add_argument("--gate-arm-cadence-secs", type=float, default=60.0,
                    metavar="SECS",
                    help="Seconds between band-growth checkpoints. Four "
                         "are needed before the band-growth conjunct can "
                         "return True, so this value x 4 is the earliest "
                         "any sweep can fire — 240 s at the default, "
                         "which is most of a short bounded run. Lower it "
                         "to arm a K0-class run inside its budget. "
                         "DEFAULT 60 = the receipted behaviour.")
    ap.add_argument("--gate-target-typed", action="store_true",
                    help="Operator attestation that this target has a "
                         "TYPED wall-corpus row (the corpus is markdown; "
                         "there is no runtime surface to read it from). "
                         "Echoed in the run header. Without it the arm "
                         "never arms.")
    ap.add_argument("--gate-band", type=int, default=24, metavar="N",
                    help="Width, in gx buckets behind the frontier, of "
                         "the band the sweep roots and the boundary "
                         "histogram are drawn from.")
    ap.add_argument("--gate-sham-roots", type=int, default=4, metavar="N",
                    help="Roots drawn from the FREELY-ADVANCING region "
                         "as K1's null: a ranking that yields as much "
                         "there as at the wall is reading noise.")
    ap.add_argument("--gate-axes", type=str, default=None, metavar="PATH",
                    help="Axis sidecar (docs/receipts/probes/"
                         "gate_axes_<target>.json) merged into the "
                         "profile's solve.state_sig, APPENDED last so a "
                         "resumed archive's bit indices survive. Entries "
                         "without a probe-receipt sha are REFUSED at "
                         "merge. BOTH A/B arms must pass the same file.")
    ap.add_argument("--contact-bits", type=int, default=3, metavar="N",
                    help="Cap on merged sidecar axes (the merged "
                         "state_sig is capped at 8 bits regardless).")
    args = ap.parse_args()
    if args.gate_opener != "off" and args.gate_pin_secs is None:
        ap.error("--gate-pin-secs is REQUIRED when --gate-opener is not "
                 "off (mined defaults: 600 CV / 120 BB; negative "
                 "disables). v1 derives nothing.")
    global GX_BUCKET, Y_BAND
    GX_BUCKET, Y_BAND = args.gx_bucket, args.y_band

    s = Solver(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(s, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(s, "stop", True))
    s.seed()
    s.flush()
    s.explore()
    s.flush()
    s.progress_line(time.time() - s.t0)
    print(f"[go_explore_solve] done: {json.dumps(s.archive.stats())}, "
          f"{s.n_solutions} solutions -> {s.out}", flush=True)
    s.pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
