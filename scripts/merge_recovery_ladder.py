#!/usr/bin/env python3
"""Phase 0 of docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md — mint the
merged 1-1 ladder: the banked 758-rung solver-tape ladder PLUS the 27
mined post-stick recovery states, interleaved at their gx-mapped tape
positions, into one states dir the existing `backward_curriculum`
machinery consumes unchanged.

    .venv/bin/python scripts/merge_recovery_ladder.py \
      --ladder checkpoints/backward_states/1-1 \
      --recovery checkpoints/backward_states/1-1-recovery \
      --out checkpoints/backward_states/1-1-v27

Contract (registered; do not drift):

1. The recovery index's `gx: 0` values are PLACEHOLDERS. For each
   recovery state, boot a 1-worker pool from the v27 profile (same ROM,
   frame_skip 4), `load_worker_state(blob)`, and read
   gx = x_position_page*256 + x_position_low via the profile's own
   `ram_mapping` — the observable the reward already uses; nothing new
   crosses the purity line. The area byte is recorded alongside.
2. Each recovery state maps to the ladder entry nearest by gx.
   Insertion `step` = that entry's step (duplicate steps are legal;
   `load_index`'s sort is stable). If the merged sequence dips > 16 px
   at the insertion point, shift the insertion by up to +-2 rungs; if
   it still dips, DROP the state and record why.
3. Merged index: entries sorted by step; recovery blobs byte-copied as
   r_000.state..r_026.state; recovery entries carry
   `frame: 900000 + i` — `entry.frame` is telemetry-only (the window
   draw counts entries), so the marker is inert to training and makes
   a recovery rung loudly identifiable in the `[backward] iter ...`
   log line. Meta: `every_frames: 4`, `stride_steps: 1` (preserves
   advance_entries = 40), provenance of both sources, and the
   `recovery_map` merge manifest.
4. Self-checks, abort (and delete the written index — no manifest, no
   launch) on failure:
     a. entry count == 758 + kept
     b. every kept recovery gx inside [40, 3266]
     c. gx_report(merged) monotone at tolerance 16 / reset_max 256
     d. every blob loads in the emulator without error
     e. sha256 of every recovery blob matches its source in
        runs/recovery_distill/fuel/tapes/*.start.state
   V1 threshold: < 24 kept (> 3 dropped) is VOID — treatment too thin.

The recovery ACTION tapes are never read. Start states only — the
tape-as-teacher family is adjudicated dead (Dossier v3; Variant A).
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.backward_curriculum import (  # noqa: E402
    DEFAULT_GX_RESET_MAX, DEFAULT_GX_TOLERANCE, INDEX_NAME, StateEntry,
    gx_report, load_index, write_index,
)
from src.utils.run_lock import acquire as _acquire_run_lock  # noqa: E402

GX_SPAN = (40, 3266)     # tape gx_first .. tape gx_last (self-check b)
VOID_MIN_KEPT = 24       # V1: < 24 kept recovery states = treatment too thin
SHIFT_RUNGS = 2          # max insertion shift when the mapped spot dips
AREA_ADDR = 0x0760       # SMB internal area index (segments the gx gate)


def merge_lock_path(out_dir) -> Path:
    """Run-lock path for a mint --out target (see src/utils/run_lock.py).
    Lives BESIDE out_dir, not inside it, so it survives the --force
    rmtree of out_dir."""
    out_dir = Path(out_dir)
    return out_dir.parent / f".{out_dir.name}.run.lock"


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md5_path(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def boot_pool(profile: dict):
    import nes_core
    pool = nes_core.Pool(
        rom_path=str(REPO / profile["rom_path"]),
        num_workers=1,
        frame_skip=int(profile.get("frame_skip", 4)),
    )
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    return pool


def peek_byte(pool, addr: int) -> int:
    val, _ = pool.peek_u16_consistent(0, addr, addr)
    return val & 0xFF


def measure_recovery(pool, ram_mapping: dict, blob: bytes) -> tuple[int, int, bool]:
    """gx and area of a savestate, read from the loaded machine without
    stepping it (peek_u16_consistent reads system RAM in place)."""
    pool.load_worker_state(0, blob)
    lo = int(ram_mapping["x_position_low"])
    hi = int(ram_mapping["x_position_page"])
    gx, consistent = pool.peek_u16_consistent(0, lo, hi)
    area = peek_byte(pool, AREA_ADDR)
    return int(gx), area, bool(consistent)


def try_insert(merged: list[StateEntry], ladder_entry: StateEntry,
               gx_r: int, tolerance: int) -> tuple[int | None, dict]:
    """Insertion position for a recovery entry at `ladder_entry`'s step,
    or (None, dips) if the merged sequence would dip > tolerance there.

    The side (before/after the rung) follows gx order; the dip check runs
    against the CURRENT merged sequence, so recovery entries already
    inserted nearby count as neighbors.
    """
    idx = merged.index(ladder_entry)
    pos = idx + 1 if gx_r >= ladder_entry.gx else idx
    prev_e = merged[pos - 1] if pos > 0 else None
    next_e = merged[pos] if pos < len(merged) else None
    drop_in = (prev_e.gx - gx_r) if prev_e is not None else 0
    drop_out = (gx_r - next_e.gx) if next_e is not None else 0
    dips = {"drop_in": int(drop_in), "drop_out": int(drop_out)}
    if drop_in > tolerance or drop_out > tolerance:
        return None, dips
    return pos, dips


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mint the merged v27 ladder (758 tape rungs + mined "
                    "recovery states, gx-interleaved).")
    ap.add_argument("--ladder", default="checkpoints/backward_states/1-1")
    ap.add_argument("--recovery",
                    default="checkpoints/backward_states/1-1-recovery")
    ap.add_argument("--out", default="checkpoints/backward_states/1-1-v27")
    ap.add_argument("--profile", default="configs/mario_1_1_v27_seed0.yaml",
                    help="v27 profile the gx measurement pool boots from "
                         "(ROM + frame_skip + ram_mapping).")
    ap.add_argument("--tapes", default="runs/recovery_distill/fuel/tapes",
                    help="Source dir the recovery blobs were banked from; "
                         "self-check (e) verifies sha256 against "
                         "*.start.state here.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing --out dir.")
    args = ap.parse_args()

    import yaml

    ladder_dir = REPO / args.ladder
    recovery_dir = REPO / args.recovery
    out_dir = REPO / args.out
    tapes_dir = REPO / args.tapes

    # Run-lock: two mints racing the same --out target can both pass the
    # exists()/--force check, both do the (expensive) gx measurement
    # pass, and then race the rmtree + mkdir — the same defect class as
    # the 2026-08-29 duplicate-chain-watcher incident, just with a
    # bigger blast radius here (a stomped or half-copied ladder dir). The
    # lock lives beside out_dir, not inside it, so it survives the
    # --force rmtree below. A stale lock from a dead process is
    # reclaimed; a live one refuses. See src/utils/run_lock.py.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    _lock = merge_lock_path(out_dir)
    _holder = _acquire_run_lock(_lock, extra=str(out_dir))
    if _holder is not None:
        sys.exit(f"[merge_recovery_ladder] {out_dir} is locked by live "
                 f"PID {_holder.pid} ({_lock}). Refusing to run two "
                 "mints against one --out target.")
    atexit.register(lambda: _lock.exists() and _lock.unlink())

    if out_dir.exists():
        if not args.force:
            sys.exit(f"ABORT: {out_dir} already exists — pass --force to "
                     "re-mint (the merge is deterministic).")
        shutil.rmtree(out_dir)

    profile = yaml.safe_load((REPO / args.profile).read_text())
    ram_mapping = profile["ram_mapping"]

    # ---- load both indexes (V1 count gate) --------------------------
    ladder_entries, ladder_meta = load_index(ladder_dir)
    recovery_entries, recovery_meta = load_index(recovery_dir)
    n_ladder, n_recovery = len(ladder_entries), len(recovery_entries)
    print(f"[merge] ladder {n_ladder} entries "
          f"(gx {ladder_entries[0].gx}..{ladder_entries[-1].gx}), "
          f"recovery {n_recovery} entries")
    if n_ladder != 758 or n_recovery != 27:
        sys.exit(f"ABORT (V1 count): expected 758 ladder + 27 recovery, "
                 f"got {n_ladder} + {n_recovery}")

    # ---- machine identity: same ROM, same frame_skip ----------------
    prof_fs = int(profile.get("frame_skip", 4))
    ladder_fs = int(ladder_meta.get("frame_skip", 4))
    if prof_fs != 4 or ladder_fs != 4:
        sys.exit(f"ABORT: frame_skip mismatch — profile {prof_fs}, "
                 f"ladder {ladder_fs}, contract says 4")
    rom_path = REPO / profile["rom_path"]
    rom_md5 = md5_path(rom_path)
    ladder_rom = Path(ladder_meta.get("rom", rom_path))
    if ladder_rom.exists() and md5_path(ladder_rom) != rom_md5:
        sys.exit(f"ABORT: ROM md5 mismatch — profile ROM {rom_md5} vs "
                 f"ladder ROM {md5_path(ladder_rom)}")

    # ---- sha256 -> source tape (needed for the recovery_map) --------
    tape_by_sha: dict[str, str] = {}
    for t in sorted(tapes_dir.glob("*.start.state")):
        h = sha256_path(t)
        if h in tape_by_sha:
            sys.exit(f"ABORT: duplicate tape sha256 — {t.name} and "
                     f"{tape_by_sha[h]} are byte-identical; source "
                     "attribution would be ambiguous")
        tape_by_sha[h] = t.name

    # ---- measure gx/area of every recovery blob (placeholders!) ----
    pool = boot_pool(profile)
    measured = []   # per recovery entry: dict with gx/area/sha/source
    try:
        for e in recovery_entries:
            blob_path = recovery_dir / e.file
            blob = blob_path.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            source = tape_by_sha.get(sha)
            if source is None:
                sys.exit(f"ABORT (self-check e): recovery blob {e.file} "
                         f"sha256 {sha[:16]}... matches no tape in "
                         f"{tapes_dir}")
            gx, area, consistent = measure_recovery(pool, ram_mapping, blob)
            measured.append({
                "recovery_index": e.step, "recovery_file": e.file,
                "source_frame": e.frame, "gx": gx, "area": area,
                "peek_consistent": consistent, "sha256": sha,
                "source_tape": source,
            })
            print(f"[measure] {e.file} -> gx {gx} area {area} "
                  f"(source {source})")
    finally:
        pool.shutdown()

    # ---- map + interleave ------------------------------------------
    merged: list[StateEntry] = list(ladder_entries)
    recovery_map: list[dict] = []
    dropped: list[dict] = []
    for m in measured:
        i, gx_r = m["recovery_index"], m["gx"]
        nearest = min(range(n_ladder),
                      key=lambda j: (abs(ladder_entries[j].gx - gx_r), j))
        candidates = sorted(
            (j for j in range(max(0, nearest - SHIFT_RUNGS),
                              min(n_ladder, nearest + SHIFT_RUNGS + 1))),
            key=lambda j: (abs(ladder_entries[j].gx - gx_r), j))
        placed = None
        attempts = []
        for j in candidates:
            pos, dips = try_insert(merged, ladder_entries[j], gx_r,
                                   DEFAULT_GX_TOLERANCE)
            attempts.append({"rung": j, "step": ladder_entries[j].step,
                             "rung_gx": ladder_entries[j].gx, **dips})
            if pos is not None:
                entry = StateEntry(step=ladder_entries[j].step,
                                   frame=900000 + i, gx=gx_r,
                                   area=m["area"], file=f"r_{i:03d}.state")
                merged.insert(pos, entry)
                placed = {"file": entry.file,
                          "source_tape": m["source_tape"],
                          "measured_gx": gx_r,
                          "mapped_step": entry.step,
                          "sha256": m["sha256"],
                          "area": m["area"],
                          "frame": entry.frame,
                          "recovery_index": i,
                          "source_file": m["recovery_file"],
                          "source_frame": m["source_frame"],
                          "nearest_rung": nearest,
                          "shift": j - nearest,
                          "peek_consistent": m["peek_consistent"]}
                recovery_map.append(placed)
                print(f"[map] {entry.file} gx {gx_r} -> step "
                      f"{entry.step} (rung {j}, shift {j - nearest:+d})")
                break
        if placed is None:
            dropped.append({"recovery_index": i,
                            "source_file": m["recovery_file"],
                            "source_tape": m["source_tape"],
                            "measured_gx": gx_r, "area": m["area"],
                            "sha256": m["sha256"],
                            "why": "merged sequence dips > "
                                   f"{DEFAULT_GX_TOLERANCE} px at the "
                                   f"insertion point and at every shift "
                                   f"within +-{SHIFT_RUNGS} rungs",
                            "attempts": attempts})
            print(f"[drop] recovery {i} gx {gx_r}: no insertion within "
                  f"+-{SHIFT_RUNGS} rungs avoids a dip "
                  f"> {DEFAULT_GX_TOLERANCE} px")

    kept = len(recovery_map)
    if kept < VOID_MIN_KEPT:
        sys.exit(f"ABORT (V1 VOID): only {kept} of {n_recovery} recovery "
                 f"states kept (< {VOID_MIN_KEPT}) — treatment too thin; "
                 "no manifest written, no launch")

    # ---- write the merged dir --------------------------------------
    out_dir.mkdir(parents=True)
    for e in ladder_entries:
        shutil.copyfile(ladder_dir / e.file, out_dir / e.file)
    for r in recovery_map:
        shutil.copyfile(recovery_dir / r["source_file"],
                        out_dir / r["file"])

    report = gx_report(merged, tolerance=DEFAULT_GX_TOLERANCE,
                       reset_max=DEFAULT_GX_RESET_MAX)
    meta = {
        "level": ladder_meta.get("level", "1-1"),
        "provenance": "merged_recovery_ladder",
        "registered": "docs/proposals/V27_FRESH_RECOVERY_2026-08-24.md",
        "minted_by": "scripts/merge_recovery_ladder.py",
        "root_state": ladder_meta.get("root_state"),
        "actions": ladder_meta.get("actions"),
        "profile": args.profile,
        "rom": str(rom_path),
        "rom_md5": rom_md5,
        "frame_skip": 4,
        "every_frames": 4,
        "stride_steps": 1,
        "n_actions": ladder_meta.get("n_actions"),
        "hw": ladder_meta.get("hw"),
        "tail": ladder_meta.get("tail"),
        "sources": {
            "ladder": {"dir": str(ladder_dir), "entries": n_ladder,
                       "provenance": ladder_meta.get("provenance"),
                       "minted_at": ladder_meta.get("minted_at")},
            "recovery": {"dir": str(recovery_dir), "entries": n_recovery,
                         "kept": kept, "dropped": len(dropped),
                         "tapes_dir": str(tapes_dir),
                         "note": recovery_meta.get("note")},
        },
        "recovery_map": recovery_map,
        "recovery_dropped": dropped,
        "gx": report,
        "minted_at": time.time(),
    }
    write_index(out_dir, merged, meta)

    # ---- self-checks on the WRITTEN artifact; abort = no manifest --
    failures: list[str] = []
    checks: dict[str, str] = {}

    written_entries, written_meta = load_index(out_dir)
    n_expected = n_ladder + kept

    # (a) entry count
    if len(written_entries) == n_expected:
        checks["a_entry_count"] = f"PASS ({len(written_entries)} == 758 + {kept})"
    else:
        failures.append(f"(a) entry count {len(written_entries)} != {n_expected}")

    # (b) recovery gx span
    rec_written = [e for e in written_entries if e.frame >= 900000]
    bad_span = [e for e in rec_written
                if not (GX_SPAN[0] <= e.gx <= GX_SPAN[1])]
    if len(rec_written) == kept and not bad_span:
        gxs = [e.gx for e in rec_written]
        checks["b_gx_span"] = (f"PASS ({kept} recovery gx in "
                               f"[{min(gxs)}, {max(gxs)}] within "
                               f"[{GX_SPAN[0]}, {GX_SPAN[1]}])")
    else:
        failures.append(f"(b) recovery gx span: {len(rec_written)} marked "
                        f"entries, out-of-span {[e.gx for e in bad_span]}")

    # (c) merged monotonicity
    wreport = gx_report(written_entries, tolerance=DEFAULT_GX_TOLERANCE,
                        reset_max=DEFAULT_GX_RESET_MAX)
    if wreport["monotone"]:
        checks["c_monotone"] = (f"PASS (decreases {wreport['decreases']}, "
                                f"over-tolerance "
                                f"{wreport['decreases_over_tolerance']}, "
                                f"max_drop {wreport['max_drop']} px, "
                                f"segments {wreport['segments']})")
    else:
        failures.append(f"(c) gx_report not monotone: {wreport}")

    # (d) every blob loads in the emulator
    pool = boot_pool(profile)
    load_errors = []
    try:
        for e in written_entries:
            try:
                pool.load_worker_state(0, (out_dir / e.file).read_bytes())
            except Exception as exc:  # noqa: BLE001 — scored, not raised
                load_errors.append(f"{e.file}: {exc}")
    finally:
        pool.shutdown()
    if not load_errors:
        checks["d_blob_load"] = f"PASS (all {len(written_entries)} blobs load)"
    else:
        failures.append(f"(d) blob load failures: {load_errors[:5]} "
                        f"({len(load_errors)} total)")

    # (e) recovery sha256 vs tape sources
    sha_bad = []
    for r in written_meta["recovery_map"]:
        got = sha256_path(out_dir / r["file"])
        want = sha256_path(tapes_dir / r["source_tape"])
        if got != want or got != r["sha256"]:
            sha_bad.append(r["file"])
    if not sha_bad:
        checks["e_sha256"] = (f"PASS (all {kept} recovery blobs match "
                              "their fuel-tape sources)")
    else:
        failures.append(f"(e) sha256 mismatches: {sha_bad}")

    for k in sorted(checks):
        print(f"[self-check] {k}: {checks[k]}")
    if failures:
        (out_dir / INDEX_NAME).unlink(missing_ok=True)
        for f in failures:
            print(f"[self-check] FAIL {f}", file=sys.stderr)
        sys.exit("ABORT: self-check failed — index.json removed "
                 "(no merge manifest, no launch)")

    print(f"\n[merge] OK: {len(written_entries)} entries "
          f"({n_ladder} ladder + {kept} recovery, {len(dropped)} dropped) "
          f"-> {out_dir}")
    print(f"[merge] gx {wreport['gx_first']}..{wreport['gx_last']} "
          f"monotone={wreport['monotone']} max_drop={wreport['max_drop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
