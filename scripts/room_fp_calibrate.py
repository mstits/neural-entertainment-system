#!/usr/bin/env python3
"""Room-fingerprint calibration + probe-fixture tooling (room-graph T4).

Three subcommands, all built on the same hardware-surface capture:

  capture   Run ONE scripted rollout from a savestate and record, per
            solver step: the 2 KB physical nametable snapshot
            (`Pool.peek_nametables`), the PPU scroll odometer (x, y),
            the scene ordinal, and the odo_debug rendered-line count.
            Written as an .npz with a JSON provenance record (rom +
            state sha256, frame_skip, action script, date) so any
            fixture under tests/fixtures/roomgraph/ can be re-minted
            from one command line.

  mask      Auto volatility mask (ROOMGRAPH_ENGINE_2026-08-24 §4): from
            one or more SAME-ROOM captures (idle and/or in-room walk
            jiggles — scripts that never cross a room boundary), a
            nametable byte is VOLATILE iff it takes more than one value
            within any single capture's rendered frames. Volatile bytes
            are merged into [lo, hi) zero-ranges — the profile's
            `room_fp.mask`. The receipt (docs/receipts/room_fp/<game>.md)
            records the capture provenance, the volatile set, and the
            pre/post-mask distinct-hash counts per capture. No human
            reads the screen anywhere in this pipeline: the mask is a
            variance computation over our own rollouts.

  replay    Replay the T1 detector (`fp_settle` + `classify_transition`
            + `RoomIndex`) over a capture with a given room_fp config
            and print the settled rooms / edges / warps. This is the
            same routine the RG-0 offline falsifier imports
            (`replay_room_stream`), so the pytest and the receipt
            numbers can never drift apart.

PURITY: every input is a hardware surface (nametable VRAM, PPU scroll
odometer, scene ordinal, rendered-line vote, palette RAM) — the classes
v23 ruled legal — plus, under --record-ram only, the 2 KB system RAM
that `step_all` already returns, used exclusively for probe-side
validation of experiment-discovered observables (e.g. locating a death
event to trim a capture window). RAM never feeds the mask or the hash.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.go_explore_solve import (  # noqa: E402
    RoomIndex,
    classify_transition,
    fp_settle,
    nt_fingerprint,
    room_fp_mask,
)

BUTTON_BITS = {
    "a": 0x01, "b": 0x02, "select": 0x04, "start": 0x08,
    "up": 0x10, "down": 0x20, "left": 0x40, "right": 0x80,
    "noop": 0x00,
}


def parse_script(spec: str):
    """`spec` = comma-joined phases, each `<buttons>*<steps>`.

    `<buttons>` is `+`-joined button tokens; a token may carry `%N`
    ("held only on steps where (step_within_phase % N) == 0" — e.g.
    `right+b%2*400` holds RIGHT and pulses B every other step, the
    Metroid door-opening walk) or `/N` (duty cycle: held for N steps,
    released for N — `right+a/4*34` is the Metroid item-room hop).
    `noop*75` idles."""
    phases = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        btns, _, steps = raw.rpartition("*")
        if not btns or not steps.isdigit() or int(steps) < 1:
            raise SystemExit(f"[room_fp_calibrate] bad phase {raw!r} — "
                             f"want <buttons>*<steps>")
        toks = []
        for tok in btns.split("+"):
            name, mode, period = tok, "hold", 1
            if "%" in tok:
                name, _, p = tok.partition("%")
                mode, period = "pulse", int(p)
            elif "/" in tok:
                name, _, p = tok.partition("/")
                mode, period = "duty", int(p)
            name = name.strip().lower()
            if name not in BUTTON_BITS:
                raise SystemExit(f"[room_fp_calibrate] unknown button "
                                 f"{name!r} in {raw!r}")
            toks.append((BUTTON_BITS[name], mode, period))
        phases.append((toks, int(steps)))
    if not phases:
        raise SystemExit("[room_fp_calibrate] empty action script")
    return phases


def script_actions(phases):
    """Yield one controller bitmask per solver step."""
    for toks, steps in phases:
        for s in range(steps):
            act = 0
            for bit, mode, period in toks:
                on = (s % period == 0 if mode == "pulse" else
                      (s // period) % 2 == 0 if mode == "duty" else True)
                if on:
                    act |= bit
            yield act


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(rom: str, state: str, frame_skip: int, script: str,
            out: Path, record_ram: bool = False) -> dict:
    import nes_core
    rom_p, state_p = Path(rom), Path(state)
    pool = nes_core.Pool(rom_path=str(rom_p), num_workers=1,
                         frame_skip=int(frame_skip))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.set_odometer_enabled(True)
    pool.load_worker_state(0, state_p.read_bytes())
    nts, odos, scenes, lines, rams = [], [], [], [], []
    for act in script_actions(parse_script(script)):
        out_step = pool.step_all(np.array([act], dtype=np.uint8))
        x, y = pool.get_odometer_per_worker()[0]
        scene = pool.get_odometer_scene_per_worker()[0]
        _, _, nlines = pool.odo_debug(0)
        nts.append(np.frombuffer(pool.peek_nametables(0),
                                 dtype=np.uint8).copy())
        odos.append((int(x), int(y)))
        scenes.append(int(scene))
        lines.append(int(nlines))
        if record_ram:
            rams.append(np.frombuffer(bytes(out_step[0][2]),
                                      dtype=np.uint8).copy())
    meta = {
        "tool": "scripts/room_fp_calibrate.py capture",
        "date": _dt.date.today().isoformat(),
        "rom": str(rom_p), "rom_sha256": _sha256(rom_p),
        "state": str(state_p), "state_sha256": _sha256(state_p),
        "frame_skip": int(frame_skip), "script": script,
        "steps": len(nts),
    }
    arrays = {
        "nt": np.stack(nts).astype(np.uint8),
        "odo": np.array(odos, dtype=np.int64),
        "scene": np.array(scenes, dtype=np.int64),
        "lines": np.array(lines, dtype=np.int64),
        "meta": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    }
    if record_ram:
        arrays["ram"] = np.stack(rams).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[capture] {out}: {len(nts)} steps fs{frame_skip} "
          f"scene {scenes[0]}->{scenes[-1]} odo {odos[0]}->{odos[-1]}")
    return meta


def mint_state(rom: str, state: str, frame_skip: int, script: str,
               out: Path) -> None:
    """Run a scripted rollout and save the END worker state — the
    receipted way to mint fixture root states (e.g. the Zelda
    north-screen state the east-exit capture starts from). The saved
    blob is a full versioned savestate envelope (odometer included),
    loadable at any pool frame_skip."""
    import nes_core
    pool = nes_core.Pool(rom_path=str(rom), num_workers=1,
                         frame_skip=int(frame_skip))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.set_odometer_enabled(True)
    pool.load_worker_state(0, Path(state).read_bytes())
    n = 0
    for act in script_actions(parse_script(script)):
        pool.step_all(np.array([act], dtype=np.uint8))
        n += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(pool.save_worker_state(0)))
    print(f"[mint] {out}: {n} steps fs{frame_skip} from {state}")


def load_capture(path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z else {}
    return z["nt"], z["odo"], z["scene"], z["lines"], meta


def volatile_bytes(captures, min_lines: int = 200):
    """Union over captures of {bytes taking >1 value within rendered
    frames of that capture}. Captures must be same-room by script
    construction — cross-room variance is signal, not volatility."""
    vol = np.zeros(2048, dtype=bool)
    per_capture = []
    for path in captures:
        nt, _, _, lines, _ = load_capture(path)
        keep = lines >= int(min_lines)
        frames = nt[keep]
        if len(frames) == 0:
            raise SystemExit(f"[room_fp_calibrate] {path}: no rendered "
                             f"frames above min_lines={min_lines}")
        v = (frames != frames[0]).any(axis=0)
        per_capture.append((str(path), int(v.sum()), len(frames)))
        vol |= v
    return vol, per_capture


def to_ranges(vol: np.ndarray):
    """Boolean volatility vector -> merged [lo, hi) mask ranges."""
    ranges, start = [], None
    for i in range(2048):
        if vol[i] and start is None:
            start = i
        elif not vol[i] and start is not None:
            ranges.append([start, i])
            start = None
    if start is not None:
        ranges.append([start, 2048])
    return ranges


def hash_stats(path, mask_ranges, min_lines: int = 200,
               palette_cokey: bool = False):
    """Pre/post-mask distinct hash counts over rendered frames."""
    nt, _, _, lines, _ = load_capture(path)
    keep = lines >= int(min_lines)
    frames = nt[keep]
    raw = room_fp_mask([])
    masked = room_fp_mask(mask_ranges)
    pre = len({nt_fingerprint(f, raw) for f in frames})
    post = len({nt_fingerprint(f, masked) for f in frames})
    return pre, post, len(frames)


def replay_room_stream(nt, odo, scene, lines, cfg, index=None):
    """Offline replay of the T1 settle/classify/intern loop over one
    captured stream — the §2 pseudo-code, executed on fixture data.
    Feeds RG-0 (tests/test_rg0_roomgraph.py) and the receipt numbers.

    Returns (RoomIndex, events); each event is a dict
    {step, kind, dir, src, dst, d_odo, d_scene, steps} — one per FIRED
    settle. Adoption from unknown (src None) mints no edge; warp mints
    no edge ever (telemetry only); pan/fade from a known room records a
    directed edge, exactly like the live hot loop (T2).

    ONSET-BASELINE CONVENTION (measured 2026-08-24, receipts under
    docs/receipts/room_fp/): when a churn onset is minted (pend is
    None), the odo/scene passed to `fp_settle` are the PREVIOUS
    rendered sample's — the last pre-churn baseline — not the current
    sample's. A transition's first NT flip lands in the same solver
    step as (or after) its odometer/scene movement: Zelda's death
    flash bumps the scene ordinal in the very frames its first
    attribute rewrite appears, so an onset stamped with the CURRENT
    sample reads Δscene +1-of-2 (or 0) and the warp classifies fade.
    The baseline sample is the correct integration origin for the
    whole churn window. On non-minting calls the CURRENT sample is
    passed (it is only read as the settle endpoint). T2's hot loop
    must follow the same convention. Degenerate `settle: 1` would
    fire on the minting call and read the baseline as the endpoint —
    profiles must use settle >= 2 (the parse enforces >= 1 only)."""
    mask = room_fp_mask(cfg["mask"])
    settle = int(cfg.get("settle", 3))
    min_lines = int(cfg.get("min_lines", 200))
    pan_odo = cfg.get("pan_odo", (128, 384))
    warp_scene_min = int(cfg.get("warp_scene_min", 2))
    if index is None:
        index = RoomIndex(cap=int(cfg.get("max_rooms", 1024)))
    room = None
    settled_h = None
    pend = None
    events = []
    prev_odo, prev_scene = None, None
    for t in range(len(nt)):
        if lines[t] < min_lines:
            pend = None
            prev_odo, prev_scene = None, None   # blank breaks the baseline
            continue
        h = nt_fingerprint(nt[t], mask)
        if pend is None and prev_odo is not None:
            base_odo, base_scene = prev_odo, prev_scene
        else:
            base_odo, base_scene = odo[t], scene[t]
        pend, fired = fp_settle(pend, h, settled_h, base_odo, base_scene,
                                t, settle)
        prev_odo, prev_scene = odo[t], scene[t]
        if fired is None:
            continue
        fh, d_odo, d_scene, steps = fired
        kind, direction = classify_transition(d_odo, d_scene, pan_odo,
                                              warp_scene_min)
        o = index.intern(fh, odo[t])
        if o is None:            # at cap: hold-last, keep old identity
            settled_h = fh       # but do not re-fire on every sample
            continue
        src = room
        if src is not None and o != src:
            if kind == "warp":
                index.record_warp(src, o, d_scene, d_odo)
            else:
                index.record_edge(src, o, kind, direction, steps)
        elif src is None and kind == "warp":
            index.record_warp(None, o, d_scene, d_odo)
        events.append({"step": int(t), "kind": kind, "dir": direction,
                       "src": src, "dst": int(o),
                       "d_odo": [int(d_odo[0]), int(d_odo[1])],
                       "d_scene": int(d_scene), "steps": int(steps)})
        room = o
        settled_h = fh
    return index, events


def _cfg_from_args(a) -> dict:
    return {"mask": json.loads(a.mask), "settle": a.settle,
            "min_lines": a.min_lines,
            "pan_odo": json.loads(a.pan_odo),
            "warp_scene_min": a.warp_scene_min,
            "max_rooms": a.max_rooms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--rom", required=True)
    c.add_argument("--state", required=True)
    c.add_argument("--frame-skip", type=int, default=4)
    c.add_argument("--script", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--record-ram", action="store_true")

    mi = sub.add_parser("mint")
    mi.add_argument("--rom", required=True)
    mi.add_argument("--state", required=True)
    mi.add_argument("--frame-skip", type=int, default=4)
    mi.add_argument("--script", required=True)
    mi.add_argument("--out", required=True)

    m = sub.add_parser("mask")
    m.add_argument("captures", nargs="+")
    m.add_argument("--min-lines", type=int, default=200)
    m.add_argument("--game", required=True)
    m.add_argument("--receipt", default=None,
                   help="write/overwrite docs/receipts/room_fp/<game>.md")

    r = sub.add_parser("replay")
    r.add_argument("capture")
    r.add_argument("--mask", default="[]")
    r.add_argument("--settle", type=int, default=3)
    r.add_argument("--min-lines", type=int, default=200)
    r.add_argument("--pan-odo", default="[128, 384]")
    r.add_argument("--warp-scene-min", type=int, default=2)
    r.add_argument("--max-rooms", type=int, default=1024)

    a = ap.parse_args(argv)
    if a.cmd == "capture":
        capture(a.rom, a.state, a.frame_skip, a.script, Path(a.out),
                record_ram=a.record_ram)
        return 0
    if a.cmd == "mint":
        mint_state(a.rom, a.state, a.frame_skip, a.script, Path(a.out))
        return 0
    if a.cmd == "mask":
        vol, per = volatile_bytes(a.captures, a.min_lines)
        ranges = to_ranges(vol)
        print(f"[mask] {a.game}: {int(vol.sum())} volatile bytes -> "
              f"{len(ranges)} ranges: {ranges}")
        for path, nvol, nfr in per:
            pre, post, _ = hash_stats(path, ranges, a.min_lines)
            print(f"  {path}: {nvol} volatile / {nfr} rendered frames; "
                  f"distinct hashes pre-mask {pre} -> post-mask {post}")
        print(f"  yaml: mask: {json.dumps(ranges)}")
        return 0
    if a.cmd == "replay":
        nt, odo, scene, lines, meta = load_capture(a.capture)
        idx, events = replay_room_stream(nt, odo, scene, lines,
                                         _cfg_from_args(a))
        print(f"[replay] {a.capture} ({meta.get('script', '?')}): "
              f"{idx.n_rooms()} rooms, "
              f"{sum(len(v) for v in idx.adj.values())} edges, "
              f"{idx.warp_count} warps")
        for e in events:
            print(f"  step {e['step']:4d}: {e['kind']:<4} "
                  f"dir={e['dir']} {e['src']}->{e['dst']} "
                  f"d_odo={e['d_odo']} d_scene={e['d_scene']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
