"""Semi-automated observable discovery for a fresh game bootstrap.

Codifies the hard-won methodology for finding the RAM bytes a Go-Explore
`solve:` adapter (scripts/go_explore_solve.py, GenericGame) needs — progress,
room/screen counter, y, health/lives — from THIS core's own scripted rollouts.
No external RAM maps, no disassembly: every verdict is measured from the
emulator's own memory under our own inputs (CLAIMS.md provenance rule), the
same discipline as scripts/verify_ram_map.py.

What it finds, and the gates each candidate must clear
------------------------------------------------------
PROGRESS (find_progress). A byte, or a lo|hi<<8 pair, that climbs with real
forward travel. TWO independent gates, because either alone is fooled:

  Gate 1 — GAMEPLAY-LOCKED-WRAP + REVERSIBILITY.
    * rises under forward input, with a high wrap-aware monotone fraction;
    * is FLAT under NOOP — this is what rejects a free-running FRAME COUNTER,
      which advances on a fixed cadence whether or not the player moves
      (the trap the MM2 $001C bootstrap fell into);
    * for a lo|hi pair, lo wraps 255->0 in the same step hi increments;
    * (bonus, not required) falls under reverse input. A forced-scroller
      like Contra cannot travel backward past the left edge, so reversibility
      is supporting evidence only, never a rejection.

  Gate 2 — SATURATION.
    A camera/scroll byte can pass Gate 1 and still be useless: it caps at the
    scroll-window limit while the level keeps going (the Kirby $0083|$0095
    lesson). We drive forward across many hundreds of frames / multiple
    rooms and ask whether the candidate KEEPS CLIMBING or CAPS. A capper is
    flagged as a camera-clamp saturator and is NOT world position; the true
    progress for such (room-based) games is the ROOM/SCREEN counter.

ROOM COUNTER (find_room_counter). A byte stable within a room and stepping
(monotone up) at every screen/door/level transition — the real progress for
room-based games (Kirby, Metroid, Zelda, Ghosts'n Goblins) and the reference
the saturation test uses to detect a spatial byte re-basing at transitions.

Y (find_y). Ballistic/jump signature: a dominant resting value with jump
excursions that return to it; flat-ish under NOOP.

HEALTH / LIVES (find_hp_lives). A HUD-legible lives byte that decrements on
death; or, when the opening is too forgiving to take damage, a stable health
field (with HUD mirror tiles) used as a death proxy (is_dead on health drop).

Usage
-----
  python scripts/discover_observables.py --rom "roms/Contra (USA).nes" \
      --state "roms/Contra (USA)_start.state.bin"
  python scripts/discover_observables.py --rom <rom> --state <state> \
      --profile configs/kirby.yaml --emit-solve
  python scripts/discover_observables.py --selftest   # Contra + Kirby ground truth

The importable functions — find_progress / find_room_counter / find_y /
find_hp_lives — each take (rom, state) and return ranked candidates, so a
bootstrap can call them directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402

# --- controller bit layout (matches src.emulation.frame_utils / the pool) ---
NOOP, A, B, SELECT, START = 0x00, 0x01, 0x02, 0x04, 0x08
UP, DOWN, LEFT, RIGHT = 0x10, 0x20, 0x40, 0x80
_DIR = {"right": RIGHT, "left": LEFT, "up": UP, "down": DOWN}
_OPP = {"right": "left", "left": "right", "up": "down", "down": "up"}
RAM_SIZE = 0x800

# Probe budgets (single worker, headless — a few thousand steps total per
# game, seconds of wall clock, well under the swarm's 6-minute solve cap).
CLEAN_N = 900        # pure directional probes (truncated at first reset)
NOOP_N = 240         # idle probe (frame-counter / flatness reference)
ADVANCE_N = 1800     # maneuver + death-recovery drive (depth / transitions)


def _resolve_dirs(forward: str) -> tuple[int, int, str, str]:
    forward = forward.lower()
    if forward not in _DIR:
        raise SystemExit(f"--forward must be one of {sorted(_DIR)}; got {forward!r}")
    rev = _OPP[forward]
    return _DIR[forward], _DIR[rev], forward, rev


# --------------------------------------------------------------------------
# Rollout engine: one pool, replayed from the start state for each probe.
# --------------------------------------------------------------------------
class Discoverer:
    """Owns a single headless worker and caches every probe's RAM log so the
    four find_* passes share one boot. Probes always start from `state`."""

    def __init__(self, rom: str, state, frame_skip: int = 4,
                 forward: str = "right", seed: int = 1) -> None:
        self.rom = rom
        # Accept either raw state bytes or a path to the *_start.state.bin so
        # the importable find_* helpers can be called with a filename directly.
        if isinstance(state, (str, Path)):
            state = Path(state).read_bytes()
        self.state = state
        self.frame_skip = int(frame_skip)
        self.fwd, self.rev, self.fwd_name, self.rev_name = _resolve_dirs(forward)
        self.seed = int(seed)
        self.pool = Pool(rom_path=rom, num_workers=1, frame_skip=self.frame_skip)
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self._cache: dict = {}
        # Baseline per-frame churn (median changed-byte count during ordinary
        # play) sets the reset threshold: a death/level reload rewrites far
        # more of RAM than one frame of motion does.
        self._reset_thr: Optional[float] = None

    # ---- low-level driving ------------------------------------------------
    def _reload(self) -> None:
        self.pool.load_worker_state(0, self.state)

    def _step(self, mask: int) -> np.ndarray:
        r = self.pool.step_all(np.array([mask], dtype=np.uint8))
        return np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]

    def _run(self, schedule, n: int) -> np.ndarray:
        self._reload()
        log = np.empty((n, RAM_SIZE), dtype=np.uint8)
        for t in range(n):
            log[t] = self._step(int(schedule(t)))
        return log

    def reset_threshold(self, log: np.ndarray) -> float:
        if self._reset_thr is None:
            d = (np.diff(log[:60].astype(np.int16), axis=0) != 0).sum(1)
            base = float(np.median(d)) if len(d) else 120.0
            self._reset_thr = max(350.0, base * 2.2)
        return self._reset_thr

    # ---- cached probes ----------------------------------------------------
    def clean_forward(self) -> np.ndarray:
        """Pure forward hold, truncated at the first mass-reset (a death/
        game-over reload would otherwise fold a fake backward jump into the
        deltas). Clean monotone travel for a forced-scroller's page/fine
        counters."""
        if "cf" not in self._cache:
            log = self._run(lambda t: self.fwd, CLEAN_N)
            self._cache["cf"] = log[: self._first_reset(log)]
        return self._cache["cf"]

    def clean_reverse(self) -> np.ndarray:
        if "cr" not in self._cache:
            log = self._run(lambda t: self.rev, CLEAN_N)
            self._cache["cr"] = log[: self._first_reset(log)]
        return self._cache["cr"]

    def noop(self) -> np.ndarray:
        if "noop" not in self._cache:
            self._cache["noop"] = self._run(lambda t: NOOP, NOOP_N)
        return self._cache["noop"]

    def _first_reset(self, log: np.ndarray, warmup: int = 25) -> int:
        thr = self.reset_threshold(log)
        d = (np.diff(log.astype(np.int16), axis=0) != 0).sum(1)
        idx = np.where(d[warmup:] > thr)[0]
        return int(idx[0]) + warmup if len(idx) else len(log)

    def advance(self) -> tuple[np.ndarray, np.ndarray]:
        """Aggressive 'make progress' drive with death recovery.

        Mostly holds forward, but injects generic maneuver macros — jumps,
        attacks, and COMPOUND door moves (back-up-then-hold-up, hold-up) that
        carry room-based games through a door the way pure forward never
        will. On a mass-reset (death / game-over) it reloads the start state
        and keeps pushing, so we accumulate deep forward travel across many
        lives. Returns (log, reset_mask) where reset_mask[i] is True for a
        diff row that spans a reload — those rows are excluded from every
        delta metric so a reload's RAM rewrite never reads as motion."""
        if "adv" in self._cache:
            return self._cache["adv"]
        rng = np.random.default_rng(self.seed)
        fwd, rev = self.fwd, self.rev
        # Each macro is a list of (mask, hold) segments played back to back.
        macros = [[(fwd, 8)], [(fwd | A, 10)], [(fwd | B, 10)], [(A, 4)],
                  [(rev, 5), (UP, 30)], [(fwd, 3), (UP, 30)], [(UP, 25)]]
        mw = np.array([6, 3, 3, 2, 2.5, 2.5, 2], float)
        mw /= mw.sum()
        n = ADVANCE_N
        log = np.empty((n, RAM_SIZE), dtype=np.uint8)
        reset_rows = np.zeros(max(n - 1, 1), dtype=bool)
        self._reload()
        prev = None
        seg, si, left = [(fwd, 1)], 0, 1
        thr = None
        for t in range(n):
            if left <= 0:
                si += 1
                if si >= len(seg):
                    seg = (macros[int(rng.choice(len(macros), p=mw))]
                           if rng.random() < 0.14 else [(fwd, 1)])
                    si = 0
                left = seg[si][1]
            mask = seg[si][0]
            left -= 1
            log[t] = self._step(mask)
            if prev is not None:
                if thr is None and t > 55:
                    thr = self.reset_threshold(log[:60])
                nch = int((log[t] != prev).sum())
                if thr is not None and nch > thr:
                    reset_rows[t - 1] = True   # diff (t-1 -> t) spans the death
                    self._reload()
                    if t < n - 1:
                        reset_rows[t] = True    # diff (t -> t+1) spans the reload
                    seg, si, left = [(fwd, 1)], 0, 1
                    prev = None
                    continue
            prev = log[t].copy()
        self._cache["adv"] = (log, reset_rows)
        return self._cache["adv"]

    def jump(self) -> tuple[np.ndarray, np.ndarray]:
        """Reload-per-rep ballistic probe: settle, then a forward-jump burst
        and a settle, repeated. Returns (baseline_ram, concatenated_log)."""
        if "jump" in self._cache:
            return self._cache["jump"]
        self._reload()
        for _ in range(12):
            self._step(NOOP)
        base = self._step(NOOP).astype(int)
        reps = []
        # Jump in place: a clean ballistic arc (dip and return) reads the y
        # byte without the forward drift a running jump adds.
        seq = [NOOP] * 4 + [A] * 8 + [NOOP] * 26
        for _ in range(6):
            self._reload()
            for m in seq:
                reps.append(self._step(m).astype(int))
        self._cache["jump"] = (base, np.array(reps))
        return self._cache["jump"]

    def close(self) -> None:
        try:
            self.pool.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Shared delta math.
# --------------------------------------------------------------------------
def _wrap_deltas(log: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-step signed deltas with 8-bit wrap folded in (255->0 counts as +1,
    0->255 as -1). Optional row mask zeroes reset/reload rows."""
    d = np.diff(log.astype(np.int16), axis=0)
    dw = np.where(d < -128, d + 256, np.where(d > 128, d - 256, d))
    if mask is not None:
        dw = dw[mask]
    return dw


def _col_stats(log: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """net (wrap-aware), monotone fraction, churn/1k, pos/neg step counts,
    raw 255->0 wrap count, per byte."""
    dw = _wrap_deltas(log, mask)
    raw = np.diff(log.astype(np.int16), axis=0)
    if mask is not None:
        raw = raw[mask]
    nz = (dw != 0)
    n = max(dw.shape[0], 1)
    pos = (dw > 0).sum(0)
    neg = (dw < 0).sum(0)
    denom = np.maximum(pos + neg, 1)
    return {
        "net": dw.sum(0),
        "mono": pos / denom,
        "churn": nz.sum(0) / n * 1000.0,
        "pos": pos, "neg": neg,
        "wraps": (raw < -128).sum(0),
    }


# --------------------------------------------------------------------------
# Coupled spatial analysis: the fine byte and the two counters that step on
# its discontinuities — the PAGE byte (steps when the fine wraps 255->0, so
# fine|page<<8 is a continuous 16-bit world position) and the ROOM byte
# (steps when the fine RE-BASES to a mid value at a door/room load, the
# signature of a camera-relative coordinate). This one pass feeds both
# find_room_counter and find_progress; cached on the Discoverer.
# --------------------------------------------------------------------------
def _spatial(disc: Discoverer) -> dict:
    if "spatial" in disc._cache:
        return disc._cache["spatial"]
    cf = disc.clean_forward()
    cr = disc.clean_reverse()
    noop = disc.noop()
    adv, rmask = disc.advance()
    keep = ~rmask
    sf, sr, sn = _col_stats(cf), _col_stats(cr), _col_stats(noop)
    sa = _col_stats(adv, keep)
    raw_cf = np.diff(cf.astype(np.int16), axis=0)
    raw_ad = np.diff(adv.astype(np.int16), axis=0)
    raw_ad[~keep] = 0     # never read a reload's RAM rewrite as a step

    def flat_noop(i: int) -> bool:
        return abs(int(sn["net"][i])) <= 4 and float(sn["churn"][i]) < 2.0

    def _coincide(a, b, tol: int = 2) -> int:
        sb = set(int(x) for x in b)
        return sum(1 for f in a if any((int(f) + o) in sb for o in range(-tol, tol + 1)))

    # FINE candidates (zero page): flat under NOOP, monotone forward travel,
    # ranked by how hard they churn under motion (a position byte moves every
    # step; a page/room counter barely moves). Lower address breaks near-ties
    # so a canonical byte wins over a zero-page mirror.
    fines = []
    for i in range(0x100):
        if not flat_noop(i):
            continue
        if int(sf["net"][i]) < 6 or float(sf["mono"][i]) < 0.9:
            continue
        churn = max(float(sf["churn"][i]), float(sa["churn"][i]))
        if churn >= 4 or int(sf["wraps"][i]) >= 1 or int(sf["net"][i]) >= 64:
            fines.append((i, churn))
    fines.sort(key=lambda x: (-round(x[1]), x[0]))
    fine_addrs = [a for a, _ in fines]

    def find_page(F: int):
        """The zero-page byte that increments exactly when F wraps 255->0."""
        best = None
        for pg in range(0x100):
            if pg == F or not flat_noop(pg):
                continue
            pos = int(sf["pos"][pg]) + int(sa["pos"][pg])
            neg = int(sf["neg"][pg]) + int(sa["neg"][pg])
            # A page byte is a SLOW counter: it changes on only a small
            # fraction of forward steps (per-1k churn is misleading on a short
            # probe, so measure the change fraction directly).
            cf_frac = float((raw_cf[:, pg] != 0).mean()) if raw_cf.shape[0] else 1.0
            if pos < 1 or neg > max(0, pos * 0.34) or cf_frac >= 0.12:
                continue
            coupled = 0
            for raw, log in ((raw_cf, cf), (raw_ad, adv)):
                if raw.shape[0] < 2:
                    continue
                fw = [w for w in np.where(raw[:, F] < -128)[0]
                      if int(log[w + 1, F]) <= 24]        # a true wrap toward 0
                inc = np.where((raw[:, pg] >= 1) & (raw[:, pg] <= 4))[0]
                coupled += _coincide(fw, inc)
            if coupled >= 1 and (best is None or coupled > best[1]):
                best = (pg, coupled)
        return best

    def find_rooms(F: int):
        """Zero-page bytes that step when F RE-BASES (a big partial drop that
        is NOT a wrap to ~0) — a room/screen load re-basing a camera-local
        coordinate."""
        reb = [f for f in np.where(raw_ad[:, F] < -40)[0] if int(adv[f + 1, F]) > 24]
        out = []
        for c in range(0x100):
            if c == F or not flat_noop(c) or float(sa["churn"][c]) >= 3.0:
                continue
            pos, neg = int(sa["pos"][c]), int(sa["neg"][c])
            if pos < 1 or neg > max(0, pos * 0.34):
                continue
            inc = np.where(raw_ad[:, c] > 0)[0]
            hits = _coincide(list(inc), reb)
            if hits >= 1 and hits >= 0.5 * len(inc):
                out.append({"addr": c, "steps": pos, "rebase_hits": hits,
                            "distinct": int(len(np.unique(adv[:, c]))),
                            "churn": round(float(sa["churn"][c]), 2),
                            "values": [int(v) for v in np.unique(adv[:, c])[:12]]})
        out.sort(key=lambda d: (d["rebase_hits"], d["steps"], d["distinct"]),
                 reverse=True)
        return out

    F0 = fine_addrs[0] if fine_addrs else None
    result = {"fines": fine_addrs, "F0": F0,
              "page0": find_page(F0) if F0 is not None else None,
              "rooms": find_rooms(F0) if F0 is not None else [],
              "find_page": find_page, "find_rooms": find_rooms,
              "sf": sf, "sr": sr, "sn": sn, "sa": sa}
    disc._cache["spatial"] = result
    return result


# --------------------------------------------------------------------------
# ROOM / SCREEN counter.
# --------------------------------------------------------------------------
def find_room_counter(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
                      frame_skip: int = 4, forward: str = "right",
                      top: int = 6) -> list[dict]:
    """A byte stable within a room that steps at every screen/door/level
    transition — the true progress for room-based games and the reference the
    saturation test needs to catch a spatial byte re-basing at transitions.

    Found as the zero-page counter whose steps coincide with the progress
    fine byte re-basing (a room-load re-base). A forced-scroller has no such
    re-basing room id; there the progress PAGE byte IS the screen counter
    (Contra $0064), returned as the fallback."""
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        sp = _spatial(disc)
        if sp["rooms"]:
            return sp["rooms"][:top]
        if sp["page0"] is not None:
            pg = sp["page0"][0]
            adv, rmask = disc.advance()
            cf = disc.clean_forward()
            return [{"addr": pg, "rebase_hits": 0,
                     "steps": int(_col_stats(adv, ~rmask)["pos"][pg]),
                     "distinct": int(len(np.unique(cf[:, pg]))),
                     "role": "progress_page_as_screen_counter",
                     "values": [int(v) for v in np.unique(cf[:, pg])[:12]]}]
        return []
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# PROGRESS.
# --------------------------------------------------------------------------
def _combined(log: np.ndarray, lo: int, hi: Optional[int]) -> np.ndarray:
    v = log[:, lo].astype(np.int64)
    if hi is not None:
        v = v + (log[:, hi].astype(np.int64) << 8)
    return v


def _saturation_verdict(disc: Discoverer, lo: int, hi: Optional[int],
                        room_addr: Optional[int]) -> dict:
    """Gate 2. Under forward travel, does lo|hi<<8 keep climbing, or cap while
    play continues? Two camera-clamp signatures, either sufficient:

      * WITHIN-ROOM CAP (primary). In the forward-dominant clean probe the
        value reaches its ceiling EARLY and then sits FLAT for a long tail
        while other RAM keeps churning (the player is active, not frozen) —
        the camera won't scroll further though the level continues. A true
        world position is instead still climbing when the probe ends (it only
        stops because the player died or the probe ran out).
      * RE-BASING. The value drops sharply at a room-counter transition — a
        camera/room-local coordinate re-bases each room; a global position
        does not.
    """
    cf = disc.clean_forward()
    adv, rmask = disc.advance()
    comb_cf = _combined(cf, lo, hi).astype(float)
    # Drop transition-frame garbage (a page byte read mid-load spikes the pair).
    if len(comb_cf):
        gcap = float(np.percentile(comb_cf, 99.0)) * 1.5 + 32
        comb_cf = np.clip(comb_cf, 0, gcap)

    # --- primary: caps early with a long flat tail while active -------------
    within_cap = False
    cap_len = 0
    active_flag = False
    n = len(comb_cf)
    if n >= 120:
        rmax = np.maximum.accumulate(comb_cf)
        news = np.where(np.diff(rmax) > 0)[0]
        last_high = int(news[-1]) + 1 if len(news) else 0
        flat_tail = n - last_high
        tail = comb_cf[last_high:] if last_high < n else comb_cf[-1:]
        seg = cf[max(last_high, 1):, :0x100]
        active_flag = (int((np.diff(seg.astype(np.int16), axis=0) != 0).sum()) > 0
                       if len(seg) > 1 else False)
        if (last_high < 0.5 * n and flat_tail >= 150 and flat_tail >= 0.30 * n
                and float(np.std(tail)) < 8.0 and active_flag):
            within_cap = True
            cap_len = int(flat_tail)

    # --- corroborating: re-basing at room transitions (advance drive) -------
    rebases = 0
    transitions = 0
    comb_ad = _combined(adv, lo, hi).astype(float)
    acap = (float(np.percentile(comb_ad, 99.0)) * 1.5 + 32) if len(comb_ad) else 1e9
    if room_addr is not None:
        rc = adv[:, room_addr].astype(int)
        m = len(comb_ad)
        for t in range(5, m - 5):
            # Only a FORWARD room step (entering the next room) re-bases a
            # camera-local coordinate; a backward scroll of the counter would
            # drop the value for a mundane reason.
            if rmask[t - 1] or rc[t] <= rc[t - 1]:
                continue
            transitions += 1
            before = float(np.median(comb_ad[t - 4:t - 1]))
            after = float(np.median(comb_ad[t + 2:t + 5]))
            if before > acap or after > acap:
                continue
            if after < before - max(32.0, 0.25 * before):
                rebases += 1

    observed_max = int(max(comb_cf.max() if len(comb_cf) else 0,
                           float(np.clip(comb_ad, 0, acap).max()) if len(comb_ad) else 0))
    return {
        "saturates": bool(within_cap or rebases >= 1),
        "within_room_cap": bool(within_cap),
        "cap_tail_len": int(cap_len),
        "rebases_at_transitions": int(rebases),
        "n_transitions": int(transitions),
        "player_active_in_tail": bool(active_flag),
        "observed_max": observed_max,
    }


def find_progress(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
                  frame_skip: int = 4, forward: str = "right",
                  top: int = 8) -> dict:
    """Rank progress candidates and apply BOTH gates.

    Returns {"forward": name, "room_counter": addr|None, "candidates":[...],
    "recommended": {...}}. Each candidate carries the gates it passed and,
    for a saturator, an explicit camera-clamp flag."""
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        sp = _spatial(disc)
        sf, sr, sn, sa = sp["sf"], sp["sr"], sp["sn"], sp["sa"]
        rooms = find_room_counter(rom, state, disc=disc, forward=forward)
        room_addr = rooms[0]["addr"] if rooms else None
        F0 = sp["F0"]

        cands = []

        def add(lo, hi, coupled):
            netf = int(sf["net"][lo])
            monof = float(sf["mono"][lo])
            reversible = int(sr["net"][lo]) < 0 or (hi is not None and int(sr["net"][hi]) < 0)
            sat = _saturation_verdict(disc, lo, hi, room_addr)
            strength = (netf + (int(sa["net"][hi]) * 256 if hi is not None else 0))
            cands.append({
                "lo": lo, "hi": hi, "kind": "pair" if hi is not None else "single",
                "label": (f"${lo:04X}|${hi:04X}<<8" if hi is not None else f"${lo:04X}"),
                "wrap_coupled": int(coupled),
                "gate1_rises_forward": bool(netf >= 6 and monof >= 0.9),
                "gate1_flat_under_noop": True,
                "reversible_bonus": bool(reversible),
                "net_forward": netf, "mono_forward": round(monof, 3),
                "net_noop": int(sn["net"][lo]),
                "net_reverse": int(sr["net"][lo]),
                **{f"sat_{k}": v for k, v in sat.items()},
                "saturates": sat["saturates"],
                "camera_clamp": sat["saturates"],
                "_strength": strength,
            })

        for F in sp["fines"][:top]:
            pg = sp["find_page"](F)
            add(F, pg[0] if pg else None, pg[1] if pg else 0)

        # The PRIMARY candidate is the top fine byte (heaviest churn = the real
        # world position). Its verdict decides the recommendation; the rest are
        # informational. Saturators sort last so the report leads with anything
        # that cleared both gates.
        primary = next((c for c in cands if c["lo"] == F0), None)
        cands.sort(key=lambda c: (c["saturates"], -c["_strength"]))
        for c in cands:
            c.pop("_strength", None)
        cands = cands[:top]

        recommended = None
        if primary is not None and not primary["saturates"]:
            recommended = {"lo": primary["lo"], "hi": primary["hi"],
                           "label": primary["label"], "kind": "spatial",
                           "as_progress": primary["label"], "saturates": False}
        elif primary is not None and room_addr is not None:
            # The world-position byte is camera-relative (saturates): the room
            # counter IS the true progress; the capped spatial is a secondary.
            recommended = {
                "lo": room_addr, "hi": None, "label": f"${room_addr:04X}",
                "as_progress": f"${room_addr:04X} (room counter — spatial "
                               f"{primary['label']} SATURATES / camera-clamp)",
                "spatial_saturator": primary["label"],
                "spatial_cap": primary.get("sat_observed_max"),
                "saturates": False, "kind": "room_counter_fallback",
            }
        return {"forward": disc.fwd_name, "room_counter": room_addr,
                "candidates": cands, "recommended": recommended}
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# Y (jump signature).
# --------------------------------------------------------------------------
def find_y(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
           frame_skip: int = 4, forward: str = "right",
           exclude: tuple = (), top: int = 6) -> list[dict]:
    """Ballistic byte: a dominant resting value (the ground) with a wide
    vertical jump excursion that returns to it.

    Restricted to low RAM (player/position state lives there, not the sprite
    buffers). Returns a ranked shortlist — several bytes move with a jump
    (position, velocity, jump timer), so this narrows the field for the human
    to confirm rather than asserting a single byte. Scored by how wide the
    mode-anchored swing is (a position sweeps a bounded band and rests at the
    ground); a full-range 0<->255 wrapper is a velocity accumulator, not y."""
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        base, J = disc.jump()
        excl = set(exclude)
        # Player position lives in zero page or the low $03xx work RAM; skip
        # the stack ($0100-$01FF) and the OAM sprite shadow ($0200-$02FF),
        # which swing wildly with every on-screen object.
        addrs = list(range(0x100)) + list(range(0x300, 0x340))
        out = []
        for i in addrs:
            if i in excl:
                continue
            col = J[:, i]
            maxdev = int(np.abs(col - base[i]).max())
            rng = int(col.max() - col.min())
            mode = int(np.bincount(col.astype(np.int64)).argmax())
            fmode = float((col == mode).mean())
            distinct = int(len(np.unique(col)))
            # A bounded position that rests at a ground mode and traces a real
            # arc (>=6 distinct sampled heights, not a 2-value flag).
            if (16 <= rng <= 200 and 12 <= maxdev <= 190
                    and 0.30 <= fmode <= 0.97 and distinct >= 6):
                out.append({"addr": i, "base": int(base[i]), "mode": mode,
                            "max_dev": maxdev, "range": rng, "distinct": distinct,
                            "frac_at_mode": round(fmode, 2),
                            "score": round(fmode * min(rng, 180), 1)})
        out.sort(key=lambda c: c["score"], reverse=True)
        return out[:top]
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# HEALTH / LIVES.
# --------------------------------------------------------------------------
def find_hp_lives(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
                  frame_skip: int = 4, forward: str = "right") -> dict:
    """A lives byte that decrements on death, or — when the opening is too
    forgiving to take damage — a stable HUD health field used as a death
    proxy (is_dead when it drops below its start value)."""
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        adv, rmask = disc.advance()
        keep = ~rmask
        dw = _wrap_deltas(adv, keep)
        churn = _col_stats(adv, keep)["churn"]

        lives = []
        for i in range(0x100):
            start = int(adv[0, i])
            # Games start with >=2 lives; a start-1 byte that dips once is
            # almost always a transient flag, not a lives counter.
            if not (2 <= start <= 9):
                continue
            col = adv[:, i]
            rng = int(col.max() - col.min())
            dec = int((dw[:, i] < 0).sum())
            # A real lives byte changes only a handful of times (very low
            # churn); a churny position/velocity byte that happens to dip is
            # not lives.
            if dec >= 1 and rng <= 3 and float(churn[i]) < 2.0:
                lives.append({"addr": i, "start": start, "decrements": dec,
                              "range": rng, "churn": round(float(churn[i]), 2)})
        # Prefer a canonical small lives count (2-5) with the most decrements.
        lives.sort(key=lambda c: (2 <= c["start"] <= 5, c["decrements"],
                                  -c["range"]), reverse=True)

        # HP fallback: a stable low-value byte whose value is redrawn as a run
        # of consecutive HUD tiles (a health meter is N identical tiles in a
        # row), preferred over a value that merely happens to recur.
        hp = []
        buf0 = adv[0]
        hud = buf0[0x500:0x800]
        for i in range(0x100):
            v0 = int(adv[0, i])
            if not (2 <= v0 <= 16) or float(churn[i]) >= 2.0:
                continue
            eq = (hud == v0).astype(int)
            mirrors = int(eq.sum())
            run = 0
            best_run = 0
            for e in eq:
                run = run + 1 if e else 0
                best_run = max(best_run, run)
            if mirrors >= 2 and best_run >= 2:
                hp.append({"addr": i, "value": v0, "mirrors": mirrors,
                           "hud_run": best_run, "churn": round(float(churn[i]), 2)})
        hp.sort(key=lambda c: (c["hud_run"], c["mirrors"]), reverse=True)

        if lives:
            best = lives[0]
            return {"kind": "lives", "addr": best["addr"], "start": best["start"],
                    "detail": best, "lives_candidates": lives[:5],
                    "hp_candidates": hp[:5],
                    "note": "is_dead on lives decrement"}
        if hp:
            best = hp[0]
            return {"kind": "hp_death_proxy", "addr": best["addr"],
                    "value": best["value"], "detail": best,
                    "lives_candidates": [], "hp_candidates": hp[:5],
                    "note": ("no decrementing lives byte in this forgiving "
                             "opening; is_dead on health < start value")}
        return {"kind": "none", "addr": None, "lives_candidates": [],
                "hp_candidates": [], "note": "no health/lives byte isolated"}
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# Full discovery + emitters.
# --------------------------------------------------------------------------
def discover_all(rom: str, state_path: str, *, frame_skip: int = 4,
                 forward: str = "right", seed: int = 1) -> dict:
    state = Path(state_path).read_bytes()
    disc = Discoverer(rom, state, frame_skip, forward, seed)
    try:
        prog = find_progress(rom, state, disc=disc, forward=forward)
        rooms = find_room_counter(rom, state, disc=disc, forward=forward)
        excl = set()
        if prog["recommended"]:
            excl.add(prog["recommended"].get("lo"))
            excl.add(prog["recommended"].get("hi"))
        for r in rooms:
            excl.add(r["addr"])
        y = find_y(rom, state, disc=disc, forward=forward,
                   exclude=tuple(a for a in excl if a is not None))
        hl = find_hp_lives(rom, state, disc=disc, forward=forward)
        return {"rom": rom, "state": state_path, "forward": forward,
                "progress": prog, "room_counters": rooms, "y": y,
                "hp_lives": hl}
    finally:
        disc.close()


def emit_solve_yaml(rom: str, findings: dict) -> str:
    """A ready-to-paste `solve:` block built purely from the findings."""
    prog = findings["progress"]
    rooms = findings["room_counters"]
    y = findings["y"]
    hl = findings["hp_lives"]
    rec = prog["recommended"]
    lines = ["solve:"]
    try:
        rel = str(Path(rom).resolve().relative_to(REPO))
    except ValueError:
        rel = rom
    lines.append(f'  rom: "{rel}"')

    room_addr = prog["room_counter"]
    saturating = [c for c in prog["candidates"] if c["saturates"]]

    if rec and rec.get("kind") == "room_counter_fallback":
        lines.append(f"  # PROGRESS WARNING: spatial {rec['spatial_saturator']} "
                     f"SATURATES (camera-clamp: caps at the scroll-window limit")
        lines.append("  #   and re-bases at room transitions). True progress = "
                     "the room/screen counter below.")
        lo = rec["lo"]
        lines.append(f"  progress: {{lo: 0x{lo:04X}}}   # room/screen counter "
                     "(steps at transitions; monotone, non-saturating)")
        sp = saturating[0] if saturating else None
        if sp is not None:
            cap = sp.get("sat_observed_max") or 0
            capval = int(cap * 1.2) if cap else 4000
            lines.append(f"  # capped spatial secondary (fine within-room X, "
                         f"DO NOT use as sole progress): {sp['label']}")
            lines.append(f"  #   progress: {{lo: 0x{sp['lo']:04X}, "
                         f"hi: 0x{sp['hi']:04X}}}  # progress_cap: {capval}")
    elif rec:
        lo, hi = rec["lo"], rec["hi"]
        if hi is not None:
            lines.append(f"  progress: {{lo: 0x{lo:04X}, hi: 0x{hi:04X}}}   "
                         "# fine | page; passes wrap + saturation (keeps climbing)")
        else:
            lines.append(f"  progress: {{lo: 0x{lo:04X}}}   # passes wrap + saturation")
    else:
        lines.append("  # progress: <none isolated — inspect candidates in the receipt>")

    if room_addr is not None:
        lines.append(f"  area: 0x{room_addr:04X}   # room/screen counter "
                     "(drives sect transitions)")
    if y:
        lines.append(f"  y: 0x{y[0]['addr']:04X}   # jump/ballistic signature")
    lines.append("  level_key: []   # coverage baseline; wire a stage-advance "
                 "byte here once a real clear is captured")
    if hl.get("addr") is not None:
        if hl["kind"] == "lives":
            lines.append(f"  lives: 0x{hl['addr']:04X}   # decrements on death")
        else:
            lines.append(f"  lives: 0x{hl['addr']:04X}   # HUD health "
                         f"(=={hl.get('value')}); death proxy: is_dead on health<start")
    return "\n".join(lines) + "\n"


def _print_report(findings: dict, emit: bool) -> None:
    prog = findings["progress"]
    print(f"\n=== {Path(findings['rom']).name} — forward='{findings['forward']}' ===")
    print(f"[room counter] {'0x%04X' % prog['room_counter'] if prog['room_counter'] is not None else 'none'}"
          f"   (candidates: {[hex(r['addr']) for r in findings['room_counters']]})")
    print("[progress candidates]  gate1=rises+flat-noop, sat=saturation gate:")
    for c in prog["candidates"]:
        flag = "  <<< CAMERA-CLAMP SATURATOR" if c["saturates"] else "  (both gates OK)"
        rb = c.get("sat_rebases_at_transitions", 0)
        wc = c.get("sat_within_room_cap", False)
        print(f"   {c['label']:>16}  kind={c['kind']:<6} netF={c['net_forward']:>6} "
              f"mono={c['mono_forward']:.2f} wrapPair={c['wrap_coupled']} "
              f"rev={int(c['reversible_bonus'])} | sat: rebase={rb} cap={int(wc)}{flag}")
    rec = prog["recommended"]
    print(f"[recommended progress] {rec['as_progress'] if rec else 'none'}")
    y = findings["y"]
    print(f"[y] {('0x%04X' % y[0]['addr']) if y else 'none'}  "
          f"(candidates: {[hex(c['addr']) for c in y]})")
    hl = findings["hp_lives"]
    print(f"[hp/lives] kind={hl['kind']} addr="
          f"{('0x%04X' % hl['addr']) if hl.get('addr') is not None else 'none'} — {hl['note']}")
    if emit:
        print("\n--- paste-ready solve: ---")
        print(emit_solve_yaml(findings["rom"], findings))


# --------------------------------------------------------------------------
# Self-test against two already-adapted games (ground truth).
# --------------------------------------------------------------------------
def _selftest() -> int:
    import yaml
    cases = [
        {"game": "Contra", "profile": "configs/contra.yaml",
         "expect_progress": (0x0065, 0x0064), "expect_saturates": False,
         "expect_room": 0x0064},
        {"game": "Kirby", "profile": "configs/kirby.yaml",
         "expect_progress": (0x0083, 0x0095), "expect_saturates": True,
         "expect_room": 0x004F},
    ]
    results = []
    ok_all = True
    for c in cases:
        prof = yaml.safe_load((REPO / c["profile"]).read_text())
        rom = str(REPO / prof["solve"]["rom"])
        state_path = str(REPO / prof["start_state_path"])
        fs = int(prof.get("frame_skip", 4))
        findings = discover_all(rom, state_path, frame_skip=fs)
        _print_report(findings, emit=True)

        prog = findings["progress"]
        lo, hi = c["expect_progress"]
        # Did the known pair (or its fine byte) surface as a candidate, and is
        # its saturation verdict what we expect?
        match = None
        for cand in prog["candidates"]:
            if cand["lo"] == lo and (cand["hi"] == hi or cand["hi"] is None):
                match = cand
                break
        found = match is not None
        sat_ok = bool(match and match["saturates"] == c["expect_saturates"])
        room_hit = any(r["addr"] == c["expect_room"] for r in findings["room_counters"])
        # Contra must recommend the climbing pair; Kirby must fall back to the
        # room counter with the spatial flagged as a saturator.
        rec = prog["recommended"] or {}
        if c["expect_saturates"]:
            rec_ok = rec.get("kind") == "room_counter_fallback" and rec.get("lo") == c["expect_room"]
        else:
            rec_ok = rec.get("lo") == lo and rec.get("hi") == hi
        passed = bool(found and sat_ok and room_hit and rec_ok)
        ok_all = ok_all and passed
        results.append({
            "game": c["game"], "rom": Path(rom).name,
            "expect_progress": f"${lo:04X}|${hi:04X}<<8",
            "candidate_found": found,
            "candidate_saturates": bool(match["saturates"]) if match else None,
            "expect_saturates": c["expect_saturates"],
            "saturation_verdict_correct": sat_ok,
            "saturation_evidence": ({k: match[k] for k in match
                                     if k.startswith("sat_")} if match else None),
            "room_counter_found": room_hit,
            "expect_room": f"${c['expect_room']:04X}",
            "found_room_counters": [f"${r['addr']:04X}" for r in findings["room_counters"]],
            "recommendation_correct": rec_ok,
            "recommended": rec.get("as_progress"),
            "y_top": f"${findings['y'][0]['addr']:04X}" if findings["y"] else None,
            "hp_lives": {"kind": findings["hp_lives"]["kind"],
                         "addr": (f"${findings['hp_lives']['addr']:04X}"
                                  if findings["hp_lives"].get("addr") is not None else None)},
            "passed": passed,
        })
    receipt = {
        "tool": "scripts/discover_observables.py",
        "purpose": "self-test observable discovery against two ground-truth games",
        "provenance": "own scripted rollouts (clean directional + maneuver "
                      "death-recovery + reload jump probes); no external RAM maps",
        "probe_budget": {"clean_forward": CLEAN_N, "clean_reverse": CLEAN_N,
                         "noop": NOOP_N, "advance": ADVANCE_N},
        "all_passed": ok_all,
        "cases": results,
    }
    out = REPO / "runs/discover_observables_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"\n[selftest] {'ALL PASS' if ok_all else 'FAILURES PRESENT'} "
          f"-> receipt {out}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rom", help="Path to the .nes ROM.")
    ap.add_argument("--state", help="Path to the *_start.state.bin start state.")
    ap.add_argument("--profile", help="Optional profile (reads frame_skip / a "
                    "discover.forward hint).")
    ap.add_argument("--forward", default=None,
                    help="Forward-travel direction: right|left|up|down "
                    "(default right; Kid-Icarus-style vertical games use up).")
    ap.add_argument("--frame-skip", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emit-solve", action="store_true",
                    help="Also print a paste-ready solve: YAML block.")
    ap.add_argument("--out", default=None, help="Write findings JSON here.")
    ap.add_argument("--selftest", action="store_true",
                    help="Verify against Contra + Kirby ground truth and write "
                    "runs/discover_observables_selftest.json.")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.rom or not args.state:
        ap.error("--rom and --state are required (or use --selftest)")

    fs, forward = args.frame_skip, args.forward
    if args.profile:
        import yaml
        prof = yaml.safe_load(Path(args.profile).read_text())
        if fs is None:
            fs = int(prof.get("frame_skip", 4))
        if forward is None:
            forward = (prof.get("discover") or {}).get("forward")
    fs = fs or 4
    forward = forward or "right"

    findings = discover_all(args.rom, args.state, frame_skip=fs, forward=forward,
                            seed=args.seed)
    _print_report(findings, emit=args.emit_solve)
    if args.out:
        Path(args.out).write_text(json.dumps(findings, indent=2, default=int) + "\n")
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
