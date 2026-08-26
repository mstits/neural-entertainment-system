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

PROGRESS FROM BANKED TAPES (find_progress_from_tapes). Same output shape,
different evidence: instead of driving the game live it replays a CHAIN of
tapes we have already solved and learns lexicographic orderings over the RAM
trace (Tom7's learnfun; src/training/lexicographic_objectives.py) to rank all
2048 bytes. That ranking is a SHORTLIST ONLY — it routes through the same
Gate 1 and Gate 2 above, because the ranking alone cannot see a free-running
timer, and the learned objective must never become a reward or archive score
(held-out Spearman rho -0.771 off the fit range). See the section header
above the function for the full ledger.

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
from typing import Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from scripts.clear_detect import score_tally_windows  # noqa: E402
from src.training.lexicographic_objectives import (  # noqa: E402
    NEVER_WIRE_AS_REWARD, WHOLE_MOVIE_REPS, concentration,
    concentration_verdict, learn_objectives, rank_locations,
    skip_until_first_input,
)
from src.training.tape_replay import (  # noqa: E402
    machine_from_profile, replay_segments, resolve_chain,
)

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
SETTLE_SCAN = 150    # idle steps used to find where the machine starts running
SETTLE_CAP = 120     # never skip more than this before a probe begins
IDLE_REPS = 4        # idle probe repeats, when a death cuts one short
DEATH_REPS = 5       # uninterrupted runs, one per seed
DEATH_MAX_N = 700    # steps per run — long enough to spend a life or two
#: How many runs must show the same drop. A run in which the player never
#: died simply abstains; none may contradict.
DEATH_MIN_AGREE = 3

#: Fight-gate probe budgets (FIGHTGATE_MECHANISM_2026-08-25.md §3.1/§5.4).
#: A fixed-rhythm mash never reaches live combat (Pilot Evidence #1/#2 on
#: Punch-Out: 400 and 600 steps of a deterministic cycle never moved
#: opp_hp at all); a RANDOMIZED mash landed a hit within 1000 steps
#: (Pilot Evidence #3). The shipped default sits meaningfully above that
#: floor rather than at it (§5.4 failure mode 2's named mitigation), at
#: the same ADVANCE_N order of magnitude every other maneuver drive uses.
ATTACK_N = 2400      # randomized offense-heavy mash (Gate FH1)
APPROACH_N = 2400    # pure-defense control, no attack input at all (Gate FH2)
FIGHT_REPS = 5        # one seed per rep, same discipline as DEATH_REPS

#: attack_mash's action distribution over the SAME six-symbol vocabulary
#: every game in this roster already defines for its action space — A, B,
#: A|B (both at once), forward, reverse, NOOP — weighted toward offense
#: (0.60 combined on A/B/A|B) rather than any fixed rhythm. Order matches
#: the mask tuple built in `attack_mash`.
FIGHT_MASH_WEIGHTS = np.array([0.25, 0.25, 0.10, 0.15, 0.15, 0.10])

#: The lives/health scan reads all of system RAM. Persistent counters are
#: routinely kept OUTSIDE the zero page (the zero page is scratch a game
#: rewrites every frame), so a zero-page-only scan cannot even nominate
#: them — it reports "no death signal" for a game that has one in plain
#: sight. Bytes are ADJUDICATED by measurement below, not by address.
LIVES_SCAN_TOP = RAM_SIZE

#: Health candidates are matched against a HUD reference window, so they
#: are drawn from the RAM BELOW that window — a byte inside the reference
#: trivially "mirrors" itself and would pass for free.
HUD_REF_LO = 0x500
HUD_REF_HI = 0x800


def _resolve_dirs(forward: str) -> tuple[int, int, str, str]:
    forward = forward.lower()
    if forward not in _DIR:
        raise SystemExit(f"--forward must be one of {sorted(_DIR)}; got {forward!r}")
    rev = _OPP[forward]
    return _DIR[forward], _DIR[rev], forward, rev


def settle_index(churn: Sequence[float] | np.ndarray) -> int:
    """How many leading steps of an idle scan are start-up, not play.

    `churn[i]` is the number of RAM bytes that changed between idle rows
    i and i+1. A machine that is still loading writes a burst far larger
    than a frame of ordinary play; a machine already playing never does.
    So: take the steady churn from the second half of the scan, and end
    the transient just past the LAST burst in the first half. No burst
    means no transient and the probe starts immediately.

    Pure, so the rule can be tested without a ROM.
    """
    ch = np.asarray(churn, dtype=float)
    if ch.size < 4:
        return 0
    half = ch.size // 2
    steady = float(np.median(ch[half:]))
    thr = max(3.0 * steady, steady + 60.0, 64.0)
    spikes = np.where(ch[:half] > thr)[0]
    if not len(spikes):
        return 0
    return int(min(int(spikes[-1]) + 2, SETTLE_CAP))


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
        # How many idle steps a probe must burn after loading the state
        # before the machine is actually playing (see settle_steps).
        self._settle: Optional[int] = None

    # ---- low-level driving ------------------------------------------------
    def settle_steps(self) -> int:
        """Idle steps to burn after loading the state before a probe counts.

        A start state is not always captured mid-play. It is routinely
        captured on a level-intro card, a fade-in or a room load — the
        machine is displaying, not yet simulating — and the level LOAD
        then lands a few dozen steps INSIDE every probe. That one event
        is enough to break the idle gate outright: the player-position
        byte reads 0 across the intro and jumps to its spawn value when
        the level appears, so the idle probe scores it as having MOVED
        while the pad was untouched, and Gate 1 rejects the true
        position byte as a free-running counter. (Measured on the SMB
        control: $0086 idles 0 -> 40 at step 41, net 40, and the whole
        paged position pair was rejected because of it.)

        Found by measurement, not by assumption: hold NOOP, take the
        per-step changed-byte count, and call the transient over after
        the LAST load-sized spike in the first half of the scan, judged
        against the steady churn of the second half. A state already
        in play has no such spike and settles at 0.
        """
        if self._settle is None:
            self.pool.load_worker_state(0, self.state)
            log = np.empty((SETTLE_SCAN, RAM_SIZE), dtype=np.uint8)
            for t in range(SETTLE_SCAN):
                log[t] = self._step(NOOP)
            ch = (np.diff(log.astype(np.int16), axis=0) != 0).sum(1)
            self._settle = settle_index(ch)
        return self._settle

    def _reload(self) -> None:
        self.pool.load_worker_state(0, self.state)
        for _ in range(self.settle_steps()):
            self._step(NOOP)

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

    def idle(self) -> tuple[np.ndarray, np.ndarray]:
        """The idle probe, VALIDATED — `(log, keep)`.

        Gate 1 asks "does this byte move while the pad is untouched?", so
        the answer is only worth anything if the rows it is measured over
        really are untouched play. Standing still is not always safe:
        plenty of games kill an idle player, and the death — plus the
        level reload behind it — teleports the position bytes, which is
        exactly the movement the gate is looking for. Read naively that
        reads as "the position byte is a free-running counter" and the
        gate rejects the one byte it exists to protect.

        So the probe is truncated at the first mass event and, if that
        left too little to judge on, repeated from a fresh load. `keep`
        masks every diff row that spans a rep boundary, so no delta is
        ever taken across a reload.
        """
        if "idle" not in self._cache:
            segs: list[np.ndarray] = []
            rows = 0
            for _ in range(IDLE_REPS):
                log = self._run(lambda t: NOOP, NOOP_N)
                seg = log[: self._first_reset(log, warmup=2)]
                if len(seg) >= 2:
                    segs.append(seg)
                    rows += len(seg) - 1
                if rows >= NOOP_N - 1:
                    break
            if not segs:                      # nothing survived: judge on one rep
                segs = [self._run(lambda t: NOOP, NOOP_N)]
            log = np.concatenate(segs) if len(segs) > 1 else segs[0]
            keep = np.ones(max(len(log) - 1, 1), dtype=bool)
            edge = -1
            for s in segs[:-1]:
                edge += len(s)
                keep[edge] = False
            self._cache["idle"] = (log, keep)
        return self._cache["idle"]

    def noop(self) -> np.ndarray:
        return self.idle()[0]

    def noop_stats(self) -> dict:
        """Gate 1's reference statistics, over validated idle rows only."""
        if "idle_stats" not in self._cache:
            log, keep = self.idle()
            self._cache["idle_stats"] = _col_stats(log, keep)
        return self._cache["idle_stats"]

    def _first_reset(self, log: np.ndarray, warmup: int = 25) -> int:
        thr = self.reset_threshold(log)
        d = (np.diff(log.astype(np.int16), axis=0) != 0).sum(1)
        idx = np.where(d[warmup:] > thr)[0]
        return int(idx[0]) + warmup if len(idx) else len(log)

    def _reset_boundaries(self, log: np.ndarray, warmup: int = 25) -> list[int]:
        """EVERY mass-RAM-rewrite boundary in `log`, not just the first.

        `_first_reset` stops at the first one because every other probe
        here only needs to know where to TRUNCATE. `find_round_gate`
        needs the whole sequence — a long attack_mash/approach_retreat
        rep can plausibly end more than one bout — so this calls the
        pure boundary finder with this Discoverer's calibrated
        threshold. See `_mass_reset_boundaries` for the math."""
        return _mass_reset_boundaries(log, self.reset_threshold(log), warmup)

    def _macros(self) -> tuple[list, np.ndarray]:
        """The maneuver vocabulary the driving probes share.

        Generic pad moves only — hold forward, jump, attack, and COMPOUND
        door moves that hold a vertical direction after a step of setup.
        BOTH verticals are in the table: a "door" is as often below the
        player (a pipe, a stairwell, a hatch) as above one, and a table
        that only ever pressed UP could not take any of them.
        """
        fwd, rev = self.fwd, self.rev
        macros = [[(fwd, 8)], [(fwd | A, 10)], [(fwd | B, 10)], [(A, 4)],
                  [(rev, 5), (UP, 30)], [(fwd, 3), (UP, 30)], [(UP, 25)],
                  [(fwd, 3), (DOWN, 30)], [(DOWN, 25)]]
        mw = np.array([6, 3, 3, 2, 2.5, 2.5, 2, 2.5, 2], float)
        return macros, mw / mw.sum()

    def death_drives(self, reps: int = DEATH_REPS) -> list[dict]:
        """Play until the run ends, `reps` times, ERASING NOTHING.

        Every other driving probe here protects itself from a death.
        `clean_forward` truncates at one; `advance` reloads and then masks
        the two diff rows that span it, so a reload's RAM rewrite is never
        read as motion. That is right for a position byte and fatal for a
        life counter, because a counter's whole signature IS what happens
        at a death. Two ways it gets lost:

          * a game that decrements in the same step that reloads the
            level puts its one informative delta inside the masked rows,
            and the scan reads `decrements == 0` (measured on the SMB
            control: $075A steps 1 -> 0 on the very row whose churn is
            what flagged the reload);
          * a game whose death is QUIET never trips a reload threshold at
            all, so nothing marks it — Contra's death rewrites ~150 bytes
            against an ordinary median of ~104, which no outlier test can
            separate.

        So this probe stops trying to find the death. It holds one seeded
        drive, never reloads, and hands back the whole log; every change
        the counter makes is in there, and the adjudication is left to
        `lives_from_death_drives`, which needs no event boundary. Each
        rep gets its own seed so the runs end differently — agreement
        across them is what separates a counter from debris.
        """
        if "deaths" in self._cache:
            return self._cache["deaths"]
        macros, mw = self._macros()
        fwd = self.fwd
        out: list[dict] = []
        for rep in range(reps):
            rng = np.random.default_rng(self.seed * 977 + rep * 31 + 1)
            self._reload()
            log = np.empty((DEATH_MAX_N, RAM_SIZE), dtype=np.uint8)
            seg, si, left = [(fwd, 1)], 0, 1
            for t in range(DEATH_MAX_N):
                if left <= 0:
                    si += 1
                    if si >= len(seg):
                        seg = (macros[int(rng.choice(len(macros), p=mw))]
                               if rng.random() < 0.20 else [(fwd, 1)])
                        si = 0
                    left = seg[si][1]
                left -= 1
                log[t] = self._step(seg[si][0])
            out.append({"rep": rep, "log": log, "steps": DEATH_MAX_N})
        self._cache["deaths"] = out
        return out

    def attack_mash(self, reps: int = FIGHT_REPS, n: int = ATTACK_N) -> list[dict]:
        """Randomized attack-heavy drive for the fight-gate probe
        (FIGHTGATE_MECHANISM_2026-08-25.md §3.1).

        Modeled on `death_drives`: never reloads mid-run, seeds
        differently per rep, hands back the whole log — the adjudication
        (`find_fight_health`) needs the raw sequence, not an event
        boundary. Each step draws INDEPENDENTLY from the six-symbol
        vocabulary every game in this roster's action space already
        defines (A, B, A|B, forward, reverse, NOOP), weighted toward
        offense (FIGHT_MASH_WEIGHTS: 0.60 combined on A/B/A|B).

        A FIXED short cycle is deliberately never used here: Pilot
        Evidence #1 (alternating A/B, 400 steps) and #2 (a fixed
        dodge+attack cycle, 600 steps) both left opp_hp completely
        unmoved on Punch-Out — the same failure `configs/punchout.yaml`
        already names ("the generic masher does NOT reach live gameplay
        on this dump"). Only Pilot Evidence #3's independent-per-step
        random draws moved it, which is why every action here is drawn
        fresh from `rng`, not sequenced through a macro table.
        """
        if "atk" in self._cache:
            return self._cache["atk"]
        fwd, rev = self.fwd, self.rev
        masks = np.array([A, B, A | B, fwd, rev, NOOP], dtype=np.int64)
        p = FIGHT_MASH_WEIGHTS
        out: list[dict] = []
        for rep in range(reps):
            rng = np.random.default_rng(self.seed * 977 + rep * 31 + 101)
            self._reload()
            log = np.empty((n, RAM_SIZE), dtype=np.uint8)
            draws = rng.choice(len(masks), size=n, p=p)
            for t in range(n):
                log[t] = self._step(int(masks[draws[t]]))
            out.append({"rep": rep, "log": log, "steps": n})
        self._cache["atk"] = out
        return out

    def approach_retreat(self, reps: int = FIGHT_REPS,
                         n: int = APPROACH_N) -> list[dict]:
        """Alternating forward/reverse bursts with NO attack input at
        all — the pure-defense control condition (§3.1).

        Burst lengths are drawn from a small distribution (3-10 steps)
        per rep so the cadence itself is not a fixed rhythm either. Its
        only job is to give `find_fight_health`'s Gate FH2 a probe where
        the player is not landing (or even attempting) a hit, so a
        foe-HP byte is provably distinguishable from the player's own
        HP (which takes damage under an offense-only probe too, per
        Pilot Evidence #1) and from a byte that merely tracks "the
        player is doing anything."
        """
        if "apr" in self._cache:
            return self._cache["apr"]
        fwd, rev = self.fwd, self.rev
        out: list[dict] = []
        for rep in range(reps):
            rng = np.random.default_rng(self.seed * 977 + rep * 31 + 202)
            self._reload()
            log = np.empty((n, RAM_SIZE), dtype=np.uint8)
            going_fwd = True
            left = int(rng.integers(3, 11))
            for t in range(n):
                if left <= 0:
                    going_fwd = not going_fwd
                    left = int(rng.integers(3, 11))
                left -= 1
                log[t] = self._step(fwd if going_fwd else rev)
            out.append({"rep": rep, "log": log, "steps": n})
        self._cache["apr"] = out
        return out

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
        fwd = self.fwd
        macros, mw = self._macros()
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


def _flat_under_noop(noop_stats: dict, addr: int) -> bool:
    """GATE 1's rejecting half: is `addr` FLAT while the player idles?

    A free-running counter — the frame counter, a music/animation timer,
    the MM2 $001C byte an early bootstrap fell for — advances on a fixed
    cadence whether or not the player moves, so it looks like beautiful
    monotone progress to anything that only watches a forward rollout.
    Idling is what separates it from real progress: hold NOOP and a
    timer keeps ticking while a position/room/world byte sits still.

    This is the ONLY thing that rejects the trap, and nothing inside the
    lexicographic objective learner can substitute for it: on SMB 1-1
    the cadence byte $07C7 takes lead-rank 6 with the heaviest mass in
    the top ten (43.7), outranking every hand-authored observable but
    $006D, and it scores the same on a SHUFFLED tape — its weight is
    cadence-invariant, so no amount of trajectory-quality reasoning can
    see through it. Only idling can.
    """
    return (abs(int(noop_stats["net"][addr])) <= 4
            and float(noop_stats["churn"][addr]) < 2.0)


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
    adv, rmask = disc.advance()
    keep = ~rmask
    sf, sr = _col_stats(cf), _col_stats(cr)
    sn = disc.noop_stats()
    sa = _col_stats(adv, keep)
    raw_cf = np.diff(cf.astype(np.int16), axis=0)
    raw_ad = np.diff(adv.astype(np.int16), axis=0)
    raw_ad[~keep] = 0     # never read a reload's RAM rewrite as a step

    def flat_noop(i: int) -> bool:
        return _flat_under_noop(sn, i)

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


def _saturation_from_logs(cf: np.ndarray, adv: np.ndarray,
                          rmask: np.ndarray, lo: int, hi: Optional[int],
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

    Pure over the logs so the same gate can be fed either drive: the
    scripted probe (`_saturation_verdict`, `cf` = clean forward hold,
    `adv` = maneuver drive) or a banked solver TAPE, which is the long
    multi-room forward drive this gate has always wanted
    (`find_progress_from_tapes` passes the tape as both, with an
    all-false `rmask` — a tape has no death-reload rows to exclude).
    """
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
    # Did this drive move the value FORWARD at all? A byte that never rose
    # has a flat tail for the boring reason, so "capped early then flat"
    # is absence of evidence rather than evidence of a camera clamp. The
    # scripted probe pre-filters candidates on forward motion so this is
    # always False there; the tape path shortlists bytes that may not move
    # within one entrance's probe at all, and reads it as INCONCLUSIVE.
    never_rose = not (bool((np.diff(comb_cf) > 0).any()) if len(comb_cf) > 1
                      else False) and not (
        bool((np.diff(comb_ad) > 0).any()) if len(comb_ad) > 1 else False)
    return {
        "saturates": bool(within_cap or rebases >= 1),
        "within_room_cap": bool(within_cap),
        "cap_tail_len": int(cap_len),
        "rebases_at_transitions": int(rebases),
        "n_transitions": int(transitions),
        "player_active_in_tail": bool(active_flag),
        "observed_max": observed_max,
        "never_rose": bool(never_rose),
    }


def _saturation_verdict(disc: Discoverer, lo: int, hi: Optional[int],
                        room_addr: Optional[int]) -> dict:
    """Gate 2 over the scripted probe's own drives."""
    adv, rmask = disc.advance()
    return _saturation_from_logs(disc.clean_forward(), adv, rmask, lo, hi,
                                 room_addr)


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
# PROGRESS FROM BANKED TAPES (learnfun shortlist -> the SAME two gates).
# --------------------------------------------------------------------------
#
# `find_progress` drives the game itself and reasons about what it sees.
# `find_progress_from_tapes` starts from trajectories we have ALREADY
# solved: it replays a chain of banked solver tapes, learns maximal
# lexicographic orderings over the RAM trace (Tom7's learnfun, ported in
# src/training/lexicographic_objectives.py), and hands the top-ranked
# locations to this file's existing gates.
#
# WHAT IT IS: a byte SHORTLISTING instrument. It turns 2048 addresses
# into ~30 worth adjudicating. On the 32-tape SMB chain it surfaces 15 of
# the 17 addresses in the hand-authored SMB table and puts $075F (world)
# first by mass over all 2048 bytes (241.0, lead-rank 5); on the 30-round
# Bubble Bobble chain it finds $0401 at lead-rank 28 while the known-bad
# $0021 scratch byte never enters the ranking.
#
# WHAT IT IS NOT, EVER: a reward, an archive score, a cell key, or any
# other training/search signal. The learned objective is fit to the
# trajectories it was shown and is ANTI-correlated with progress off
# that range — Spearman rho vs elapsed time is +0.999 in-sample on
# Bubble Bobble rounds 69..79 and -0.771 on held-out rounds 80..84.
# Optimizing it would train the agent backwards. The output of this
# function is a list of ADDRESSES for the existing gates and a human to
# rule on; the objective itself stays in the receipt.
#
# Two adjudications from the validation run are baked in below:
#   1. Every candidate routes through Gate 1 (NOOP flatness) and Gate 2
#      (saturation) before it can be recommended, because the raw
#      lexicographic ranking cannot see a free-running timer: on the SMB
#      1-1 tape $07C7 takes lead-rank 6 with the heaviest mass in the top
#      ten and scores identically on shuffled tapes. Only idling rejects
#      it. (It weakens on long chains — rank 1402 over all 32 tapes — but
#      a new game's first chain is short, so the gate is not optional.)
#   2. The quality statistic is lead-mass CONCENTRATION against a
#      caller-supplied reference — never total objective weight / VF.
#      The negative control inverts the latter: the same SMB 1-1 tape
#      replayed from the WRONG root scores 205.2 against the correct
#      run's 82.1, i.e. the broken trajectory earns 2.5x MORE weight.

#: How many learnfun-ranked locations to put through the gates.
LEARNFUN_SHORTLIST = 32


def _saturates_conclusively(sat: dict) -> bool:
    """Gate 2 with "the drive never moved this byte" held apart from "the
    byte capped". Re-basing at a room transition is always evidence; a
    flat tail is only evidence when the value actually rose first."""
    return bool(sat["rebases_at_transitions"] >= 1
                or (sat["within_room_cap"] and not sat["never_rose"]))


def gate_tape_candidates(ranked: list, *, probe_stats: dict, gate2,
                         mono_min: float = 0.9) -> tuple:
    """Route learnfun-ranked locations through Gate 1 and Gate 2.

    Pure, so the wiring is testable without a ROM:
      * `ranked`   — `lexicographic_objectives.rank_locations(res,
                     memories=trace)` records (addr, rank, lead, mass,
                     n_up, n_down, ...).
      * `probe_stats` — `{"noop": _col_stats(disc.noop()), "forward":
                     ..., "reverse": ...}`; only "noop" is required, and
                     it is the half of Gate 1 that kills the timer trap.
      * `gate2(addr) -> dict` — a `_saturation_from_logs` verdict.

    Gate 1 has two halves and the tape supplies one of them. "Rises
    under forward input" is answered by the TAPE (a location only earns
    lexicographic weight by increasing along a solved trajectory);
    "flat under NOOP" can only be answered by idling the emulator, and
    is mandatory. A candidate that fails either half, or that Gate 2
    calls a camera-clamp saturator, can never become `recommended` —
    which is the only thing `emit_solve_yaml` writes a progress line
    from.

    Returns `(candidates, recommended)` in `find_progress`'s shapes, the
    candidates sorted best-adjudicated first and NOT truncated — the
    caller decides how many to publish, but the gate summary is over all
    of them.
    """
    sn = probe_stats["noop"]
    sf = probe_stats.get("forward")
    sr = probe_stats.get("reverse")
    cands = []
    for rec in ranked:
        addr = int(rec["addr"])
        up, down = int(rec.get("n_up", 0)), int(rec.get("n_down", 0))
        tape_mono = up / max(up + down, 1)
        rises_tape = bool(up > 0 and tape_mono >= mono_min)
        netf = int(sf["net"][addr]) if sf is not None else 0
        monof = float(sf["mono"][addr]) if sf is not None else 0.0
        rises_probe = bool(netf >= 6 and monof >= mono_min)
        flat = _flat_under_noop(sn, addr)
        sat = gate2(addr)
        saturates = _saturates_conclusively(sat)
        conclusive = bool(sat["rebases_at_transitions"] >= 1
                          or not sat["never_rose"])
        gate1 = bool(flat and (rises_tape or rises_probe))
        rejected = (None if gate1 and conclusive and not saturates else
                    "gate1_noop_timer" if not flat else
                    "gate1_no_forward_rise" if not (rises_tape or rises_probe)
                    else "gate2_camera_clamp" if saturates else
                    "gate2_inconclusive")
        cands.append({
            # --- find_progress candidate shape ---
            "lo": addr, "hi": None, "kind": "single",
            "label": f"${addr:04X}",
            "wrap_coupled": 0,
            "gate1_rises_forward": rises_probe,
            "gate1_flat_under_noop": flat,
            "reversible_bonus": bool(sr is not None
                                     and int(sr["net"][addr]) < 0),
            "net_forward": netf, "mono_forward": round(monof, 3),
            "net_noop": int(sn["net"][addr]),
            "net_reverse": int(sr["net"][addr]) if sr is not None else 0,
            **{f"sat_{k}": v for k, v in sat.items()},
            "saturates": saturates,
            "camera_clamp": saturates,
            # --- tape / learnfun provenance ---
            "source": "learnfun_tape_chain",
            "gate1_rises_on_tape": rises_tape,
            "tape_up": up, "tape_down": down,
            "tape_mono": round(tape_mono, 3),
            "tape_span": int(rec.get("span", 0)),
            "learnfun_rank": int(rec["rank"]),
            "learnfun_lead": float(rec["lead"]),
            "learnfun_mass": float(rec["mass"]),
            "learnfun_disc": float(rec["disc"]),
            "n_obj": int(rec["n_obj"]), "best_pos": int(rec["best_pos"]),
            "singleton_vf": float(rec.get("singleton_vf", 0.0)),
            "gate1": gate1, "gate2_conclusive": conclusive,
            "gates_passed": bool(gate1 and conclusive and not saturates),
            "rejected_by": rejected,
        })

    # Gate failures sort last so the report — and `emit_solve_yaml`'s
    # "first saturator" pick — always leads with something adjudicated.
    cands.sort(key=lambda c: (not c["gates_passed"], c["saturates"],
                              -c["learnfun_lead"]))
    passing = next((c for c in cands if c["gates_passed"]), None)
    recommended = None
    if passing is not None:
        recommended = {
            "lo": passing["lo"], "hi": None, "label": passing["label"],
            "kind": "learnfun_shortlist",
            "as_progress": (
                f"{passing['label']} (learnfun lead-rank "
                f"{passing['learnfun_rank']} over banked tapes; gates 1+2 "
                f"pass) — SHORTLIST ONLY, confirm before wiring"),
            "saturates": False,
            "learnfun_rank": passing["learnfun_rank"],
            "shortlist_only": True,
            "never_wire_as_reward": NEVER_WIRE_AS_REWARD,
        }
    return cands, recommended


def gate_summary(cands: list) -> dict:
    """What the gates concluded over the WHOLE shortlist.

    `gate1_clean_gate2_untested` is the interesting column on a chain:
    bytes that are flat while idling and climb along solved play — so
    not timers — but that the live probe never moves from one entrance,
    so the saturation gate has no verdict on them. On the SMB world-1
    chain that is where $075F (world) lands. They are candidates for a
    human, not recommendations.
    """
    verdicts: dict = {}
    for c in cands:
        key = c["rejected_by"] or "passed"
        verdicts[key] = verdicts.get(key, 0) + 1
    return {
        "n_gated": len(cands),
        "verdicts": verdicts,
        "passed": [c["label"] for c in cands if c["gates_passed"]],
        "gate1_clean_gate2_untested": [
            c["label"] for c in cands
            if c["gate1"] and c["rejected_by"] == "gate2_inconclusive"],
        "rejected_as_timer": [c["label"] for c in cands
                              if c["rejected_by"] == "gate1_noop_timer"],
        "rejected_as_camera_clamp": [
            c["label"] for c in cands
            if c["rejected_by"] == "gate2_camera_clamp"],
    }


def find_progress_from_tapes(
    chain, *, profile, rom: Optional[str] = None,
    hw_flags: Optional[str] = None, state=None,
    disc: Optional[Discoverer] = None, forward: str = "right",
    top: int = 8, shortlist: int = LEARNFUN_SHORTLIST,
    whole: int = WHOLE_MOVIE_REPS, decrease_tol: float = 0.0,
    reference=None, reference_seed: int = 1234, solution: int = 0,
    verify: bool = False,
) -> dict:
    """Shortlist progress bytes from ALREADY-SOLVED tapes, then gate them.

    Returns `find_progress`'s shape — `{"forward", "room_counter",
    "candidates", "recommended"}` — so `discover_all` / `emit_solve_yaml`
    consume it unchanged, plus `"learnfun"` (the ranking and its
    provenance) and `"quality"` (the concentration verdict).

    `chain` is a solver run dir, a chain JSON, or a list of
    `{"root", "actions"}` segments — see `tape_replay.resolve_chain`.
    CHAINS, NOT SINGLE TAPES: the Bubble Bobble round counter $0401 is
    invisible on one round's 214-memory tape and lands at lead-rank 28
    on the 30-round chain.

    `profile` is the SOLVE profile the tapes were produced under (its
    `action_space` indexes them). `state` is the start state the two
    gates probe from; it defaults to the chain's first root blob, which
    is the entrance the tapes themselves start at.

    `reference` is what the trajectory-quality statistic is read
    against, and it must be MATCHED IN LENGTH:
      * `"shuffled"` — replay the same tapes with their actions permuted
        (`reference_seed`). Matched by construction, and it is the exact
        negative control the statistic was validated on.
      * a chain spec, a `learn_objectives` result, or a `concentration`
        dict — used directly.
      * `None` (default) — concentration is still reported, with verdict
        "no_reference".

    NEVER read total objective weight as quality: the broken controls
    score 2.5x MORE of it than the real solve. See `NEVER_WIRE_AS_REWARD`
    for why nothing here may become a reward term.
    """
    if isinstance(profile, (str, Path)):
        import yaml
        profile = yaml.safe_load(Path(profile).read_text())
    rom_abs, frame_skip, bitmasks, hw = machine_from_profile(
        profile, rom=rom, hw_flags=hw_flags, who="discover")

    segments = resolve_chain(chain, solution=solution, hw_flags=hw)
    mem, masks, seams = replay_segments(
        segments, rom=rom_abs, bitmasks=bitmasks, frame_skip=frame_skip,
        hw_flags=hw)
    mem, skipped = skip_until_first_input(mem, masks)

    res = learn_objectives(mem, whole=whole, verify=verify,
                           decrease_tol=decrease_tol)
    ranked = rank_locations(res, by="lead", memories=mem)
    # Top-N by lead UNION top-N by mass: $0401 sits at lead-rank 28 with
    # unremarkable mass, while $075F is only 5th by lead but 1st of all
    # 2048 bytes by mass. Neither view alone catches both.
    by_mass = {r["addr"] for r in
               sorted(ranked, key=lambda r: -r["mass"])[:shortlist]}
    short = ranked[:shortlist] + [r for r in ranked[shortlist:]
                                  if r["addr"] in by_mass]

    quality = _tape_quality(
        res, mem, reference, chain=chain, profile=profile, rom=rom_abs,
        bitmasks=bitmasks, frame_skip=frame_skip, hw=hw,
        reference_seed=reference_seed, solution=solution, whole=whole,
        decrease_tol=decrease_tol, verify=verify)

    own = disc is None
    if own:
        probe_state = state if state is not None else segments[0].root
        disc = Discoverer(rom_abs, probe_state, frame_skip, forward)
    try:
        fwd_name = disc.fwd_name
        sp = _spatial(disc)
        rooms = find_room_counter(rom_abs, disc.state, disc=disc,
                                  forward=forward)
        room_addr = rooms[0]["addr"] if rooms else None

        # GATE 2 IS THE EXISTING GATE, UNMODIFIED: the scripted probe's
        # own drives, the same reference room counter, the same verdict
        # `find_progress` gets. Two things were tried and rejected while
        # wiring this up, both recorded so they are not re-attempted:
        #
        #   * Feeding the banked TAPE in as the forward drive. It looks
        #     like an upgrade (a real multi-room drive instead of a
        #     900-step hold) but the within-room cap test is calibrated
        #     to ONE continuous drive, and a chain of concatenated tapes
        #     is not one — a per-level byte that maxes out in 1-2 and
        #     sits still through 1-3/1-4 reads as a camera clamp. On the
        #     SMB world-1 chain it rejected every conclusive candidate.
        #   * Swapping the reference room counter when
        #     `find_room_counter` falls back to the progress PAGE byte.
        #     That would make the gate REJECT LESS, which is the wrong
        #     direction for a safety gate to be tuned in.
        #
        # What the tape does supply is Gate 1's other half — "rises under
        # forward travel" — which is what a location earning
        # lexicographic weight means in the first place.
        def gate2(addr: int) -> dict:
            return _saturation_verdict(disc, addr, None, room_addr)

        cands, recommended = gate_tape_candidates(
            short, probe_stats={"noop": sp["sn"], "forward": sp["sf"],
                                "reverse": sp["sr"]},
            gate2=gate2)
        gates = gate_summary(cands)
    finally:
        if own:
            disc.close()

    return {
        "forward": fwd_name,
        "room_counter": room_addr,
        "candidates": cands[:top],
        "recommended": recommended,
        "quality": quality,
        "gates": gates,
        "learnfun": {
            "source": "banked solver tapes replayed on our own core",
            "purpose": "byte SHORTLIST for the gates below — never a "
                       "reward, archive score or cell key",
            "never_wire_as_reward": NEVER_WIRE_AS_REWARD,
            "segments": [{"label": s.label, "n_actions": len(s)}
                         for s in segments],
            "n_segments": len(segments), "seams": seams,
            "n_memories": int(mem.shape[0]),
            "skipped_pre_input": int(skipped),
            "n_orderings": res["n_unique"],
            "n_positive": res["n_positive"],
            "decrease_tol": float(decrease_tol),
            "invalid_orderings": res["invalid_orderings"],
            "shortlist": [
                {"addr": r["addr"], "rank": r["rank"], "lead": r["lead"],
                 "mass": r["mass"], "disc": r["disc"],
                 "best_pos": r["best_pos"], "n_up": r.get("n_up"),
                 "n_down": r.get("n_down")} for r in short],
        },
    }


def _tape_quality(res: dict, mem: np.ndarray, reference, *, chain, profile,
                  rom, bitmasks, frame_skip, hw, reference_seed, solution,
                  whole, decrease_tol, verify) -> dict:
    """Lead-mass concentration for the fit chain, read against a reference.

    Deliberately does NOT report total objective weight as quality: the
    negative control inverts it (wrong-root replay of the SMB 1-1 tape
    scores 205.2 vs the correct run's 82.1). Total weight is carried in
    `diagnostics` with that warning attached, because it is still useful
    for spotting an empty enumeration.
    """
    fit = concentration(res)
    ref_conc, n_ref, ref_kind = None, 0, "none"
    if isinstance(reference, str) and reference == "shuffled":
        segs = resolve_chain(chain, solution=solution, hw_flags=hw,
                             shuffle_seed=reference_seed)
        rmem, rmasks, _ = replay_segments(segs, rom=rom, bitmasks=bitmasks,
                                          frame_skip=frame_skip, hw_flags=hw)
        rmem, _ = skip_until_first_input(rmem, rmasks)
        ref_conc = concentration(learn_objectives(
            rmem, whole=whole, verify=verify, decrease_tol=decrease_tol))
        n_ref, ref_kind = int(rmem.shape[0]), f"shuffled(seed={reference_seed})"
    elif isinstance(reference, dict) and "n_leaders" in reference:
        ref_conc = reference
        n_ref = int(reference.get("n_memories", mem.shape[0]))
        ref_kind = "caller_supplied_concentration"
    elif isinstance(reference, dict) and "objectives" in reference:
        ref_conc = concentration(reference)
        n_ref, ref_kind = int(reference["n_memories"]), "caller_supplied_result"
    elif reference is not None:
        segs = resolve_chain(reference, solution=solution, hw_flags=hw)
        rmem, rmasks, _ = replay_segments(segs, rom=rom, bitmasks=bitmasks,
                                          frame_skip=frame_skip, hw_flags=hw)
        rmem, _ = skip_until_first_input(rmem, rmasks)
        ref_conc = concentration(learn_objectives(
            rmem, whole=whole, verify=verify, decrease_tol=decrease_tol))
        n_ref, ref_kind = int(rmem.shape[0]), "caller_supplied_chain"

    verdict = concentration_verdict(fit, ref_conc, n_fit=int(mem.shape[0]),
                                    n_reference=n_ref)
    verdict["reference_kind"] = ref_kind
    verdict["diagnostics"] = {
        "total_weight": float(sum(r["weight"] for r in res["objectives"])),
        "warning": "total objective weight / VF is NOT a validity gate — "
                   "the negative control INVERTS it (a broken trajectory "
                   "scored 2.5x more: 205.2 wrong-root vs 82.1 correct on "
                   "the same SMB 1-1 tape). Read `verdict` instead.",
    }
    return verdict


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
#: A life counter steps DOWN by a small fixed amount. Wider drops are how
#: a health bar or a score reads, and those are adjudicated separately.
LIVES_MAX_DROP = 8
#: ...and it steps a handful of times in a run, not continuously. This is
#: what separates it from a countdown: a level timer only ever falls, by
#: one, and would otherwise satisfy every other test here.
LIVES_MAX_TICKS = 6


def _regime_split(mv: np.ndarray) -> int:
    """Index of the end of the FIRST regime in a wrap-aware delta column.

    A run that outlives its stock gets it REFILLED — a new game hands
    the player a fresh life, a fixed-ring fight hands the loser a fresh
    opponent — in however many stores the game feels like. Everything
    from the first RISE on is a different regime (a different life, a
    different bout), so a stock's spend is judged only up to there, not
    across it.

    Shared by `lives_from_death_drives` (a life counter refilled at a
    continue) and `find_fight_health` (an opponent's HP bar refilled by
    a round change, FIGHTGATE_MECHANISM_2026-08-25.md Gate FH3, measured
    live on Punch-Out's own mac_hp: 0 -> 72 inside a single 1000-step
    probe with no death and no round change) — same shape of event,
    same fix, reused rather than re-derived twice.
    """
    nz = np.nonzero(mv)[0]
    if not nz.size:
        return len(mv)
    ups = nz[mv[nz] > 0]
    return int(ups[0]) if ups.size else len(mv)


def lives_from_death_drives(drives: Sequence[Mapping], *,
                            idle_stats: Optional[Mapping] = None,
                            top: int = 6) -> tuple[list, dict]:
    """Rank life-counter candidates over uninterrupted runs.

    Pure — `drives` is whatever `Discoverer.death_drives` produced, but
    any sequence of mappings with a `log` array does, so the adjudication
    is testable without a ROM.

    Deliberately makes no attempt to locate the deaths. Detecting them
    was the old failure: a game whose death is loud gets its evidence
    masked as a reload, and a game whose death is quiet is never marked
    at all. What a life counter looks like over a whole run needs no such
    boundary:

      * every change it makes is DOWN, and all of them by the same small
        amount (wrap folded, so a counter that rolls 0 -> 255 to signal
        the last life still reads as one step down);
      * it changes only a few times in the whole run — the test that
        rejects a countdown timer, which passes everything above;
      * it starts from the same value in every rep, which it must, since
        every rep starts from the same state;
      * and the runs AGREE. Seeds end differently, so a byte the level
        layout happened to knock down once does not repeat.

    Whether it also moves while the pad is idle is reported but not
    disqualifying: a game that kills an idle player would otherwise
    disqualify its own counter.

    Returns `(candidates, evidence)`.
    """
    logs = [np.asarray(d["log"]) for d in drives if len(np.asarray(d["log"])) > 1]
    ev = {"drives": len(drives), "usable": len(logs),
          "steps": [int(len(np.asarray(d["log"]))) for d in drives]}
    if not logs:
        return [], ev

    width = min(log.shape[1] for log in logs)
    scan = min(LIVES_SCAN_TOP, width)
    deltas = [_wrap_deltas(log) for log in logs]
    starts = np.stack([log[0, :scan].astype(np.int64) for log in logs])
    idle_moved = (np.asarray(idle_stats["churn"])[:scan] > 0
                  if idle_stats is not None else np.zeros(scan, dtype=bool))

    need = min(DEATH_MIN_AGREE, len(logs))
    out = []
    for i in range(scan):
        start = int(starts[0, i])
        if not bool((starts[:, i] == start).all()):
            continue
        step: Optional[int] = None
        agreeing = refills = spends = 0
        floor = True
        stock = True
        ok = True
        for log, dw in zip(logs, deltas):
            mv = dw[:, i]
            nz = np.nonzero(mv)[0]
            if not nz.size:
                continue                       # this run never ended: abstain
            end = _regime_split(mv)
            down = mv[:end][mv[:end] < 0]
            if not down.size or down.size > LIVES_MAX_TICKS:
                ok = False
                break
            first = int(down[0])
            if not (-LIVES_MAX_DROP <= first < 0) or not bool((down == first).all()):
                ok = False
                break
            if step is None:
                step = first
            elif first != step:
                ok = False
                break
            agreeing += 1
            spends += int(down.size)
            refills += int(end < len(mv))         # a rise was found: refilled
            # A stock is spent to nothing, exactly once: it hits empty,
            # and what came out of it accounts for what was in it. Debris
            # knocked down by a level reload satisfies neither.
            # The zero must be ARRIVED AT, not merely present at index 0.
            # At start == 0 the old `log[:end+1]` slice included the start
            # sample itself, so `reaches_empty` was vacuously true for any
            # byte that simply began at zero -- which, paired with the
            # equally-vacuous stock check below, nominated a non-counter as
            # the lives byte on 11 of 43 onboarded profiles (measured
            # 2026-08-25). A byte that starts at 0 then underflows 0 -> 255
            # on its first change reads as a death in every episode's first
            # frames; on Mega Man (USA) that froze the search frontier at 10
            # cells, and substituting a correctly-nominated byte moved it to
            # 16 with progress 16px -> 24px.
            floor = floor and bool((log[1:end + 1, i] == 0).any())
            spent = int(down.size) * -first
            stock = stock and (start <= spent <= start - first)
        if not ok or step is None or agreeing < need:
            continue
        out.append({
            "addr": i, "start": start, "drop": -step,
            "runs_agreeing": agreeing, "runs_watched": len(logs),
            "spends": spends, "refilled_runs": refills,
            "reaches_empty": floor, "spends_its_stock": stock,
            "starts_nonzero": bool(start > 0),
            "moves_while_idle": bool(idle_moved[i]),
        })

    # Idling comes FIRST, for the same reason it leads Gate 1: a byte
    # that ticks down with the pad untouched is a clock, and a clock
    # beats a life counter on every other test here — it falls by one,
    # reaches zero, accounts for its stock, and does it far more often.
    # (Measured on the SMB control: $07A0 spends 30 against $075A's 9.)
    # Demotion, not rejection: a game that kills an idle player would
    # otherwise disqualify its own counter.
    #
    # Then: a stock spent to empty that accounts for itself, one death at
    # a time, and spent the MOST times — every run ends by costing a
    # life, while a byte the ending merely knocked down is not there for
    # all of them. Then the canonical (lowest) address, so a duplicate
    # never outranks the byte it mirrors.
    # A stock that reads 0 at the start state is not a stock. Both
    # `reaches_empty` and `spends_its_stock` are structurally vacuous at
    # start == 0 (see the arrival-at-zero note above), so neither can
    # demote such a byte on its own. Demotion rather than rejection,
    # matching the idle-clock convention directly above: the candidate
    # stays visible in the report, it just can never outrank a byte that
    # actually carries a stock.
    out.sort(key=lambda c: (c["moves_while_idle"],
                            not c["starts_nonzero"],
                            not (c["reaches_empty"] and c["spends_its_stock"]),
                            not c["reaches_empty"], abs(c["drop"] - 1),
                            -c["spends"], -c["runs_agreeing"], c["addr"]))

    # Mirror fold: a HUD copy of the counter carries the identical
    # signature. Keep the first, record the rest against it.
    kept: list[dict] = []
    _sig = lambda c: (c["start"], c["drop"], c["runs_agreeing"], c["spends"],
                      c["refilled_runs"], c["reaches_empty"],
                      c["spends_its_stock"], c["moves_while_idle"],
                      c["starts_nonzero"])
    for c in out:
        sig = _sig(c)
        prior = next((k for k in kept if _sig(k) == sig), None)
        if prior is not None:
            if len(prior["mirrors"]) < 8:
                prior["mirrors"].append(c["addr"])
            continue
        c["mirrors"] = []
        kept.append(c)
    ev["nominated"] = len(out)
    return kept[:top], ev


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
        churn = _col_stats(adv, keep)["churn"]

        lives, evidence = lives_from_death_drives(
            disc.death_drives(), idle_stats=disc.noop_stats())

        # HP fallback: a stable low-value byte whose value is redrawn as a run
        # of consecutive HUD tiles (a health meter is N identical tiles in a
        # row), preferred over a value that merely happens to recur. Drawn
        # from the RAM below the reference window — a byte inside it mirrors
        # itself for free.
        hp = []
        hud = adv[0][HUD_REF_LO:HUD_REF_HI]
        for i in range(HUD_REF_LO):
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
                    "hp_candidates": hp[:5], "death_evidence": evidence,
                    "note": f"is_dead on lives decrement (-{best['drop']}, "
                            f"agreed by {best['runs_agreeing']} of "
                            f"{best['runs_watched']} runs)"}
        if hp:
            best = hp[0]
            return {"kind": "hp_death_proxy", "addr": best["addr"],
                    "value": best["value"], "detail": best,
                    "lives_candidates": [], "hp_candidates": hp[:5],
                    "death_evidence": evidence,
                    "note": ("no decrementing lives byte in this forgiving "
                             "opening; is_dead on health < start value")}
        return {"kind": "none", "addr": None, "lives_candidates": [],
                "hp_candidates": [], "death_evidence": evidence,
                "note": ("no health/lives byte isolated "
                         f"({evidence['usable']} uninterrupted runs watched)")}
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# FIGHT-GATE — foe HP + round/opponent index for camera-static combat games
# with no spatial frontier at all (FIGHTGATE_MECHANISM_2026-08-25.md).
# Punch-Out is the validation target: both odometer axes read flat/noise
# (§1's CAMERA_STATIC_AGENT_ACTIVE receipt), so there is no fallback
# progress signal to lean on here the way a room-based game always has.
# --------------------------------------------------------------------------
def _settled_start(col: np.ndarray, *, min_stable: int = DEATH_MIN_AGREE,
                   horizon: int = SETTLE_CAP) -> int:
    """The first value `col` holds for `min_stable` consecutive reads.

    `attack_mash`/`approach_retreat` log the RAM state AFTER each rep's
    own first drawn action already ran — `Discoverer._reload()` only
    settles the machine up to the point play begins, not up to the
    point every HUD element has finished animating. A health bar that
    chips away over several frames rather than jumping straight to its
    new value (a real, common NES pattern, not a display-only effect)
    means `log[0, addr]` can land mid-chip: two reps that truly started
    the bout from the SAME reloaded stock then read as different
    "starts" purely because one rep's first action happened to land a
    hit whose tween was still in flight at frame 0 and the other's
    didn't touch the byte at all. `same_start` below would wrongly
    reject a real self/foe-HP candidate on that basis alone.

    Same debounce discipline `DEATH_MIN_AGREE` already encodes
    elsewhere in this module (a reading only counts once it has held,
    not on first appearance): scan forward for the first run of
    `min_stable` identical consecutive reads and report that value,
    bounded to the first `horizon` frames — the same settle-window cap
    `settle_steps` uses — so a real, later combat plateau many frames
    in is never mistaken for the start. A column that never settles
    inside that window falls back to the raw first read, i.e. the old
    behaviour, since there is nothing better to fall back to.
    """
    head = col[:max(int(horizon), 1)]
    n = len(head)
    if n == 0:
        raise ValueError("col must be non-empty")
    k = min(min_stable, n)
    for i in range(n - k + 1):
        window = head[i:i + k]
        if bool((window == window[0]).all()):
            return int(window[0])
    return int(head[0])


def _decrement_consensus(logs: Sequence[np.ndarray], addr: int, *,
                         min_agree: int = DEATH_MIN_AGREE) -> dict:
    """How many differently-seeded reps show `addr` net DECREASING
    within its first regime (`_regime_split`) — the same
    DEATH_MIN_AGREE-style agreement discipline `lives_from_death_drives`
    uses, generalized from a life counter's fixed -1 tick to a
    variable-magnitude HP bar. A punch does not always deal the same
    damage the way a lost life always costs exactly one, so direction
    and net magnitude are judged; the per-event size is NOT required to
    match across (or even within) reps, unlike the stock-counter test.

    Pure — `logs` is whatever a live probe or a synthetic trace hands
    it, so both Gate FH1 (offense probe) and Gate FH2 (defense-only
    probe) in `fight_health_from_drives` are the SAME call against two
    different log sets.

    `fight_health_from_drives` calls this once per candidate ADDRESS
    (up to 2048 of them) per probe set, so this diffs only the ONE
    column it needs rather than calling `_wrap_deltas` on the whole
    (n, 2048) log — that full-matrix diff is fine at `_col_stats`'
    call rate (once per probe) but is a quadratic blow-up at this call
    rate (once per address per probe).

    Requires `start > 0`: a stock has something in it when the probe
    begins. Measured live on Punch-Out (roms/Mike Tyson's Punch-Out!!,
    single-worker attack_mash/approach_retreat probe): several zero-
    page bytes tied to Little Mac's OWN punch animation (not to whether
    it lands) pass FH1+FH2 exactly as well as the true opp_hp byte —
    they decrement hard under attack_mash (pressing A/B drives the
    animation) and sit flat under approach_retreat (no A/B at all) —
    and every one of them starts the probe at 0, wrapping DOWN through
    255 instead of draining from a full value. A real HP/stock reading
    starts non-empty; this is a generic plausibility bound, not a
    Punch-Out special case, and it is what separates $0398 (start 96)
    from its animation-counter aliases in that exact live trace.

    `starts` is read through `_settled_start`, not `log[0, addr]`
    directly — see that function's docstring for the capture-timing
    edge case this closes.
    """
    agreeing = 0
    nets: list[int] = []
    starts: list[int] = []
    for log in logs:
        if len(log) < 2:
            continue
        col = log[:, addr].astype(np.int16)
        d = np.diff(col)
        dw = np.where(d < -128, d + 256, np.where(d > 128, d - 256, d))
        end = _regime_split(dw)
        regime = dw[:end]
        starts.append(_settled_start(col))
        net = int(regime.sum())
        nets.append(net)
        if net < 0 and bool((regime < 0).any()):
            agreeing += 1
    watched = len(nets)
    same_start = bool(len(set(starts)) <= 1) if starts else False
    start = starts[0] if starts else None
    need = min(min_agree, max(watched, 1))
    return {"addr": addr, "agreeing": agreeing, "watched": watched,
            "nets": nets, "same_start": same_start, "start": start,
            "passes": bool(watched > 0 and agreeing >= need and same_start
                          and start is not None and start > 0)}


def _corroborates_tally(hist: np.ndarray, windows: Sequence[tuple], addr: int
                        ) -> bool:
    """Does `addr` net-decrease inside a window `score_tally_windows`
    already flagged as a periodic anti-correlated timer/tally exchange
    (§3.2 point 6 — corroboration, not a gate)? A foe-HP byte owes
    nothing periodic, so disagreement never disqualifies a candidate;
    this only ever RAISES confidence when it fires."""
    for start, end in windows:
        if end - start < 2:
            continue
        seg = hist[start:end, addr].astype(np.int16)
        if int(np.diff(seg).sum()) < 0:
            return True
    return False


#: How many bytes away a genuine display-mirror copy of a stat can sit.
#: Chosen to catch an adjacent-byte HUD/shadow-copy mirror (a stat
#: written a second time, unchanged, to drive another digit or render
#: pass) while staying far too narrow to let two independently-moving
#: zero-page counters that just happen to sit near each other pass as
#: siblings. Not tuned to any one game's memory map.
FIGHT_MIRROR_WINDOW = 4


def _mark_mirror_siblings(cands: list[dict],
                          window: int = FIGHT_MIRROR_WINDOW) -> None:
    """Annotate each candidate in `cands` with `mirror_siblings`: how
    many OTHER candidates within `window` bytes carry the EXACT same
    `start` and the EXACT same per-rep `attack_nets` sequence.

    A stat that drives more than one on-screen readout — a HUD tile,
    a duplicate used by a different render pass, one half of a BCD
    pair — is often written byte-for-byte identically to more than one
    nearby address every frame, so a neighbour that matches perfectly
    is not independent evidence against a candidate, it is a second
    witness FOR it. An animation-frame or elapsed-step counter has no
    reason to be copied to a neighbouring address at all. Measured live
    on Punch-Out (`runs/fight_gate/discover_punchout.json`): the true
    opp_hp byte (start=96, nets=[-50,-19,-74,-64,-96]) has two
    neighbours carrying that identical start+nets pair; neither
    zero-page decoy that also clears FH1+FH2 in that same receipt has
    any. Purely corroborating — a candidate with zero siblings is left
    unboosted, never rejected, the same discipline `_corroborates_tally`
    already uses for its periodic-tally check.
    """
    for c in cands:
        c["mirror_siblings"] = sum(
            1 for other in cands
            if other is not c
            and abs(other["addr"] - c["addr"]) <= window
            and other["start"] == c["start"]
            and other["attack_nets"] == c["attack_nets"])


def _range_plausibility_penalty(c: dict) -> float:
    """>= 0; higher means less physically plausible as a single-regime
    stock reading, used as a ranking demotion (never a hard gate).

    `_decrement_consensus` only ends a regime at the first RISE
    (`_regime_split`); a byte that free-wheels downward through
    repeated 0->255 wraps without ever rising is never cut off, so its
    reported cumulative net can run past its own `start`. A real
    HP/stock byte can lose AT MOST what it started with before hitting
    zero — losing MORE than that within one regime means it wrapped
    at least once more, the signature of a free-running counter, not a
    bigger hit. Measured live on Punch-Out
    (`runs/fight_gate/discover_punchout.json`): the true opp_hp byte's
    worst net (-96) never exceeds its start (96) — ratio 0.0 — while
    two zero-page decoys that also clear FH1+FH2 in that same receipt
    worst out at 89/73 and 126/82, both well over 1.0. Continuous, not
    a cliff: a real byte a few counts past its start from some other
    cause is demoted, not disqualified, the same discipline the spread
    tiebreak already uses for suspiciously-identical outcomes.
    """
    nets = c["attack_nets"]
    start = c["start"] or 0
    if not nets or start <= 0:
        return 0.0
    worst = max(abs(n) for n in nets)
    return max(0.0, (worst - start) / start)


def fight_health_from_drives(attack_drives: Sequence[Mapping],
                             defense_drives: Sequence[Mapping], *,
                             self_hp_addr: Optional[int] = None,
                             noop_stats: Optional[Mapping] = None,
                             top: int = 6) -> dict:
    """Pure core of `find_fight_health` (§3.2/§3.3). Ranks foe-HP
    candidates from two sequences of RAM logs shaped exactly like
    `Discoverer.attack_mash` / `approach_retreat` produce — or a
    synthetic trace built the same way — so the adjudication is
    testable without a ROM.

    Directly parallel to `lives_from_death_drives` but inverted: a
    foe's stock empties from OUR offense, not from an event we can't
    locate either, so there is no reset boundary to find — just the
    same "watch the whole run, let cross-rep agreement do the work"
    discipline, applied to two probes instead of one:

      * Gate FH0 — the NOOP-oscillation veto (2026-08-25 follow-up,
        named from the Kung Fu finding below). `noop_stats` is
        whatever `Discoverer.noop_stats()` already produces for the
        SAME idle-prefilter machinery `_flat_under_noop` uses in the
        spatial/progress pipeline (§ `_spatial`) — no new probe is
        collected here, the pure-NOOP window this mechanism needs
        (NOOP_N=240 steps, well inside the 200-400 range a spawn/
        despawn or animation cycle needs to show itself) was already
        being driven for Gate 1's own free-running-counter check. A
        candidate that is NOT `_flat_under_noop` — moves away from its
        own idle value and/or churns — under ZERO player input is a
        background/animation/timer signature by construction (nothing
        else can make RAM move while the pad is untouched), so it is
        REJECTED here, the same hard-gate treatment as FH1/FH2, not
        merely reordered by `_rank` below. A soft demotion is not
        enough: on the real Kung Fu receipt
        (`runs/fight_gate/discover_kungfu.json`) the offending byte
        (0x00B1) was the ONLY candidate with 5/5 attack agreement, so a
        tiebreak that only reorders survivors would still have crowned
        it — only dropping it outright surfaces the honest "nothing
        plausible was found" result. Skipped entirely (no candidate is
        rejected) when `noop_stats` is omitted, so every existing
        caller/test that predates this gate is unaffected.
      * Gate FH1 — decrements under `attack_drives` (offense-heavy,
        randomized).
      * Gate FH2 — the self/foe discriminator, the one truly new idea
        this mechanism needs (§3.3): a real foe-HP byte must NOT show
        the same decrement-consensus under `defense_drives` (no attack
        input at all). Pilot Evidence #1 is the failure this closes —
        an offense-only probe alone makes the PLAYER's own HP look
        exactly like what Gate FH1 is searching for.
      * Gate FH3 — the refill regime split is Gate FH1/FH2's
        `_regime_split` call, already reused, not a separate pass.

    `self_hp_addr` (`find_hp_lives`'s own nominated address, or None)
    is used only to ANNOTATE a `self_hp_conflict` flag on every foe_hp
    candidate — never to filter. The actual self/foe split is Gate FH2;
    a collision with `find_hp_lives` is reported as a named finding
    (§5.4 failure mode 1), never silently resolved either way.

    Passing FH0+FH1+FH2 only narrows the field; among survivors the
    ranking (`_rank` below) adds two further corroboration terms on top
    of the original agreement/spread score, both measured against the
    live Punch-Out receipt where spread alone was not enough to put the
    true byte first: `_mark_mirror_siblings` rewards an adjacent byte
    that mirrors a candidate's start+nets exactly, and
    `_range_plausibility_penalty` demotes a candidate whose net
    magnitude exceeds its own starting value (a free-running-counter
    signature a real single-regime stock can't show). Neither is a
    gate — both only reorder candidates that already cleared FH0+FH1+FH2.
    """
    atk_logs = [np.asarray(d["log"]) for d in attack_drives
               if len(np.asarray(d["log"])) > 1]
    apr_logs = [np.asarray(d["log"]) for d in defense_drives
               if len(np.asarray(d["log"])) > 1]
    ev = {"attack_reps": len(attack_drives), "attack_usable": len(atk_logs),
         "defense_reps": len(defense_drives), "defense_usable": len(apr_logs)}
    if not atk_logs:
        return {"kind": "insufficient_probe", "addr": None,
                "foe_hp_candidates": [], "self_hp_candidates": [],
                "self_hp_conflict": self_hp_addr, "probe_evidence": ev,
                "note": "no usable attack_mash rep (every run was too "
                        "short to judge) — instrument finding"}

    width = min(l.shape[1] for l in atk_logs)
    if apr_logs:
        width = min(width, min(l.shape[1] for l in apr_logs))
    scan = min(LIVES_SCAN_TOP, width)
    if noop_stats is not None:
        scan = min(scan, len(np.asarray(noop_stats["net"])))

    foe_cands: list[dict] = []
    self_cands: list[dict] = []
    noop_vetoed: list[int] = []
    for i in range(scan):
        fh1 = _decrement_consensus(atk_logs, i, min_agree=DEATH_MIN_AGREE)
        if not fh1["passes"]:
            continue
        flat_under_noop = (_flat_under_noop(noop_stats, i)
                           if noop_stats is not None else None)
        if flat_under_noop is False:
            noop_vetoed.append(i)
            continue
        fh2 = _decrement_consensus(apr_logs, i, min_agree=DEATH_MIN_AGREE)
        entry = {
            "addr": i, "start": fh1["start"],
            "attack_nets": fh1["nets"], "attack_agree": fh1["agreeing"],
            "attack_watched": fh1["watched"],
            "defense_nets": fh2["nets"], "defense_agree": fh2["agreeing"],
            "defense_watched": fh2["watched"],
            "fh1_decrements_under_offense": True,
            "fh2_flat_under_defense": not fh2["passes"],
            "flat_under_noop": flat_under_noop,
            "self_hp_conflict": bool(self_hp_addr is not None
                                     and abs(i - self_hp_addr) <= 1),
        }
        (self_cands if fh2["passes"] else foe_cands).append(entry)
    if noop_vetoed:
        ev["noop_vetoed"] = noop_vetoed

    # Full-pool corroboration passes, run BEFORE ranking/truncation so a
    # candidate's siblings are whatever cleared FH0+FH1+FH2, not whatever
    # survived an earlier, weaker ranking's cut to `top`.
    _mark_mirror_siblings(foe_cands)
    _mark_mirror_siblings(self_cands)

    def _rank(c: dict) -> tuple:
        # Prefer a WIDER spread of net damage across the differently-
        # seeded reps over a narrower one. Measured live on Punch-Out
        # (single-worker attack_mash probe): several zero-page bytes
        # decremented by the EXACT SAME amount in all 3 reps regardless
        # of which random actions each rep took — the signature of a
        # byte driven by ELAPSED STEP COUNT alone (a background/attract
        # timer), not by how many punches actually landed, which DOES
        # vary run to run. The true opp_hp byte's nets in that same
        # trace were [-58, -19, -74]: three different outcomes from
        # three different random sequences, exactly what a combat-
        # reactive stock should show. This is the same "different seeds
        # should decorrelate a coincidence" logic
        # `lives_from_death_drives` already leans on for agreement,
        # applied in reverse: SUSPICIOUSLY IDENTICAL outcomes across
        # seeds are down-ranked, never hard-rejected (a real byte could
        # legitimately tie by chance) — demoted, not dropped, the same
        # discipline `moves_while_idle` already uses there.
        #
        # Spread alone is not enough: measured live on the real
        # Punch-Out receipt (runs/fight_gate/discover_punchout.json),
        # TWO zero-page decoys clear FH1+FH2 with real (non-identical)
        # cross-rep spread wider than the true opp_hp byte's, and both
        # out-rank it under spread alone. Two further, independent
        # corroboration terms (each pre-registered as a general rule,
        # not a per-game special case) come first: `mirror_siblings`
        # rewards a candidate for having an adjacent byte that mirrors
        # its start+nets exactly (a HUD/shadow-copy signature the true
        # byte has and the decoys do not), and the plausibility penalty
        # demotes a candidate whose net magnitude free-wheels past its
        # own starting value (a free-running counter signature the two
        # decoys have and the true byte does not). Together they move
        # the true byte from rank #3 of 5 to rank #1 on that receipt.
        nets = c["attack_nets"]
        spread = (max(nets) - min(nets)) if nets else 0
        worst_net = min(nets) if nets else 0
        return (-c["attack_agree"], _range_plausibility_penalty(c),
                -c["mirror_siblings"], -spread, worst_net, c["addr"])
    foe_cands.sort(key=_rank)
    self_cands.sort(key=_rank)
    foe_cands, self_cands = foe_cands[:top], self_cands[:top]

    tally_hist = np.concatenate(atk_logs)
    tally_windows = score_tally_windows(tally_hist)
    for c in foe_cands:
        c["tally_corroborated"] = _corroborates_tally(tally_hist, tally_windows,
                                                       c["addr"])

    if foe_cands:
        best = foe_cands[0]
        return {"kind": "foe_hp", "addr": best["addr"], "detail": best,
                "foe_hp_candidates": foe_cands,
                "self_hp_candidates": self_cands,
                "self_hp_conflict": self_hp_addr,
                "tally_windows_found": len(tally_windows),
                "probe_evidence": ev,
                "note": (f"is_foe_hp on offense-driven decrement (agreed "
                         f"by {best['attack_agree']} of "
                         f"{best['attack_watched']} attack reps; flat "
                         f"under {best['defense_watched']} defense-only "
                         "reps)")}
    return {"kind": "none", "addr": None, "foe_hp_candidates": [],
            "self_hp_candidates": self_cands, "self_hp_conflict": self_hp_addr,
            "tally_windows_found": len(tally_windows), "probe_evidence": ev,
            "note": (
                f"{len(self_cands)} address(es) decremented under BOTH "
                "probes (self_hp_candidates: self/foe aliasing, §5.4 "
                "failure mode 1) and none cleared FH2"
                if self_cands else
                "no byte cleared FH1 (decrements under offense) at all — "
                f"a behaviour finding given {ev['attack_usable']} usable "
                "attack-mash reps")}


def find_fight_health(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
                      frame_skip: int = 4, forward: str = "right",
                      hp_lives: Optional[dict] = None, top: int = 6) -> dict:
    """Live-probe wrapper: runs `attack_mash` + `approach_retreat` (and
    `find_hp_lives`, unless the caller already has it) and hands them to
    `fight_health_from_drives`. See that function for the gates.

    `noop_stats=disc.noop_stats()` reuses the SAME idle-prefilter probe
    `_flat_under_noop`/`_spatial` already drive this Discoverer through
    for Gate 1 — no separate NOOP probe is run here, it is whatever the
    Discoverer already has cached or drives on first access."""
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        if hp_lives is None:
            hp_lives = find_hp_lives(rom, state, disc=disc, forward=forward)
        return fight_health_from_drives(
            disc.attack_mash(), disc.approach_retreat(),
            self_hp_addr=hp_lives.get("addr"), noop_stats=disc.noop_stats(),
            top=top)
    finally:
        if own:
            disc.close()


def _mass_reset_boundaries(log: np.ndarray, threshold: float,
                           warmup: int = 25) -> list[int]:
    """Every index in `log` where the frame-to-frame changed-byte count
    exceeds `threshold` — the same mass-RAM-rewrite signature
    `Discoverer._first_reset` uses for the FIRST such boundary, applied
    repeatedly with a short cooldown so one multi-frame rewrite is never
    counted twice. Pure: `threshold` is whatever the caller calibrated
    (`Discoverer.reset_threshold` live, a fixed value in a test)."""
    if len(log) < warmup + 2:
        return []
    d = (np.diff(log.astype(np.int16), axis=0) != 0).sum(1)
    idx = np.where(d[warmup:] > threshold)[0] + warmup
    out: list[int] = []
    last = -10
    for i in idx:
        if int(i) - last > 5:
            out.append(int(i))
        last = int(i)
    return out


def round_gate_from_drives(drives: Sequence[Mapping], *, threshold: float,
                           top: int = 6) -> dict:
    """Pure core of `find_round_gate` (§3.4). `drives` is whatever
    `Discoverer.attack_mash`/`approach_retreat` produced (or a synthetic
    trace shaped the same way); `threshold` is the mass-reset churn
    calibration (`Discoverer.reset_threshold` live, a fixed value in a
    test) — pure, so this is testable without a ROM.

    A candidate must be: stable within a bout (flat aside from the
    boundary itself), monotone non-decreasing across every observed
    boundary, and — the check a room counter never needed — NOT reset
    back to its own start value at a boundary the way a life-counter-
    triggered level reload is (a lost bout on a multi-life game reloads
    the SAME opponent; a won bout advances to the NEXT one).

    A probe that never crosses a single boundary returns
    `insufficient_probe` — an INSTRUMENT finding (raise the budget),
    never conflated with `no_round_signal`, the BEHAVIOUR finding
    reserved for a probe that ran long enough and genuinely found
    nothing (§2's VOID/FAIL split).
    """
    logs = [np.asarray(d["log"]) for d in drives if len(np.asarray(d["log"])) > 1]
    bounds = [_mass_reset_boundaries(log, threshold) for log in logs]
    total_bounds = sum(len(b) for b in bounds)
    total_steps = sum(len(l) for l in logs)
    ev = {"drives": len(drives), "usable": len(logs),
         "probe_steps": total_steps, "bouts_observed": total_bounds}
    if total_bounds == 0:
        return {"kind": "insufficient_probe", "addr": None, "candidates": [],
                "void": True, **ev,
                "note": (f"no bout boundary observed across {total_steps} "
                         "attack_mash/approach_retreat steps — INSTRUMENT "
                         "finding (raise the probe budget), not evidence "
                         "against a round-gate byte")}

    scan = min(LIVES_SCAN_TOP, min(l.shape[1] for l in logs))
    cands = []
    for i in range(scan):
        ok = True
        resets_seen = steps_up = reset_to_start = 0
        start_val: Optional[int] = None
        for log, bnd in zip(logs, bounds):
            if not bnd:
                continue
            col = log[:, i].astype(np.int64)
            if start_val is None:
                start_val = int(col[0])
            prev_b = 0
            for b in bnd:
                seg = col[prev_b:b]
                if len(seg) > 3 and int(np.ptp(seg[1:])) > 2:
                    ok = False
                    break
                before, after = int(col[b - 1]), int(col[min(b + 1, len(col) - 1)])
                resets_seen += 1
                if after < before:
                    ok = False
                    break
                if after == start_val:
                    reset_to_start += 1
                if after > before:
                    steps_up += 1
                prev_b = b
            if not ok:
                break
        if not ok or resets_seen == 0 or steps_up == 0 or reset_to_start > 0:
            continue
        cands.append({"addr": i, "start": start_val,
                     "resets_seen": resets_seen, "steps_up": steps_up})
    cands.sort(key=lambda c: (-c["steps_up"], c["addr"]))
    cands = cands[:top]
    return {"kind": "round_gate" if cands else "no_round_signal",
            "addr": cands[0]["addr"] if cands else None,
            "candidates": cands, "void": False, **ev,
            "note": (
                f"steps at every one of {total_bounds} observed bout "
                "boundary(ies), never back to its own start value"
                if cands else
                f"{total_bounds} bout boundary(ies) observed across "
                f"{total_steps} steps, but no byte was both stable within "
                "a bout and monotone across boundaries without resetting "
                "to start — behaviour finding")}


def find_round_gate(rom: str, state: bytes, *, disc: Optional[Discoverer] = None,
                    frame_skip: int = 4, forward: str = "right",
                    top: int = 6) -> dict:
    """Live-probe wrapper: runs `attack_mash` + `approach_retreat` (reused
    from `find_fight_health`'s probes when driven through `discover_all` —
    `Discoverer` caches both) and hands them to `round_gate_from_drives`.
    """
    own = disc is None
    if own:
        disc = Discoverer(rom, state, frame_skip, forward)
    try:
        atk = disc.attack_mash()
        apr = disc.approach_retreat()
        drives = atk + apr
        ref_log = np.asarray(drives[0]["log"]) if drives else np.zeros((1, RAM_SIZE))
        thr = disc.reset_threshold(ref_log)
        return round_gate_from_drives(drives, threshold=thr, top=top)
    finally:
        if own:
            disc.close()


# --------------------------------------------------------------------------
# Full discovery + emitters.
# --------------------------------------------------------------------------
def discover_all(rom: str, state_path: str, *, frame_skip: int = 4,
                 forward: str = "right", seed: int = 1,
                 fight_gate: bool = False) -> dict:
    """Run the four standard passes; with `fight_gate=True` (the CLI's
    `--fight-gate`), also run `find_fight_health` + `find_round_gate`
    (§3) and fold their output into the findings dict under
    `fight_health` / `round_gate`. Those two keys are ABSENT — not
    `None` — when `fight_gate` is left at its default, so a profile
    that never asks for `--fight-gate` gets a byte-identical findings
    dict to before this mechanism existed (§3.5)."""
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
        out = {"rom": rom, "state": state_path, "forward": forward,
              "progress": prog, "room_counters": rooms, "y": y,
              "hp_lives": hl}
        if fight_gate:
            fh = find_fight_health(rom, state, disc=disc, forward=forward,
                                   hp_lives=hl)
            out["fight_health"] = fh
            out["round_gate"] = find_round_gate(rom, state, disc=disc,
                                                forward=forward)
        return out
    finally:
        disc.close()


def emit_solve_yaml(rom: str, findings: dict) -> str:
    """A ready-to-paste `solve:` block built purely from the findings."""
    prog = findings["progress"]
    rooms = findings["room_counters"]
    y = findings["y"]
    hl = findings["hp_lives"]
    rec = prog["recommended"]
    round_gate_inlined = False
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
            # A fine byte with no page partner has hi=None; print the
            # single-byte spelling rather than crashing on the format.
            pair = (f"lo: 0x{sp['lo']:04X}" if sp["hi"] is None else
                    f"lo: 0x{sp['lo']:04X}, hi: 0x{sp['hi']:04X}")
            lines.append(f"  # capped spatial secondary (fine within-room X, "
                         f"DO NOT use as sole progress): {sp['label']}")
            lines.append(f"  #   progress: {{{pair}}}  # progress_cap: {capval}")
    elif rec and rec.get("kind") == "learnfun_shortlist":
        # Came from `find_progress_from_tapes`: a lexicographic ranking
        # over banked tapes that then cleared Gate 1 and Gate 2. It is a
        # shortlist, so it is emitted COMMENTED — a human confirms the
        # byte before it becomes a solve adapter's progress signal.
        lines.append(f"  # PROGRESS SHORTLIST (learnfun over banked tapes, "
                     f"lead-rank {rec.get('learnfun_rank')}): passed Gate 1 "
                     f"(flat under NOOP) + Gate 2 (no camera clamp).")
        lines.append(f"  #   {NEVER_WIRE_AS_REWARD}")
        lines.append(f"  progress: {{lo: 0x{rec['lo']:04X}}}   "
                     f"# CONFIRM before trusting: shortlisted from tapes, "
                     f"not driven live")
    elif rec:
        lo, hi = rec["lo"], rec["hi"]
        if hi is not None:
            lines.append(f"  progress: {{lo: 0x{lo:04X}, hi: 0x{hi:04X}}}   "
                         "# fine | page; passes wrap + saturation (keeps climbing)")
        else:
            lines.append(f"  progress: {{lo: 0x{lo:04X}}}   # passes wrap + saturation")
    elif findings.get("fight_health", {}).get("kind") == "foe_hp":
        # FIGHT-GATE (FIGHTGATE_MECHANISM_2026-08-25.md §3.5): only reached
        # when NO spatial frontier was isolated above — a camera-static
        # combat game (Punch-Out) has none to fall back to, so this is
        # the honest degraded case, not a blocker (§4.2).
        fh = findings["fight_health"]
        rg = findings.get("round_gate") or {}
        d = fh["detail"]
        lines.append(f"  # FIGHT-GATE PROGRESS: no spatial frontier isolated; "
                     f"foe_hp cleared Gate FH1+FH2 (agreed by "
                     f"{d['attack_agree']} of {d['attack_watched']} "
                     "randomized attack-mash reps, flat under "
                     f"{d['defense_watched']} pure-defense reps).")
        if d.get("self_hp_conflict"):
            lines.append("  # WARNING: this address is within 1 of the "
                         "find_hp_lives candidate — self/foe aliasing "
                         "(§5.4 failure mode 1); CONFIRM by hand before "
                         "trusting.")
        round_gate_inlined = rg.get("kind") == "round_gate"
        round_field = f", round: 0x{rg['addr']:04X}" if round_gate_inlined else ""
        lines.append(f"  progress: {{source: fight_gate, "
                     f"foe_hp: 0x{fh['addr']:04X}, "
                     f"foe_hp_start: 0x{d['start']:02X}{round_field}}}")
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

    # Fight-gate discovery is reported even when it did NOT become the
    # active progress source (a spatial frontier already won above, or
    # the probe came back empty) — a finding is never silently dropped.
    if "fight_health" in findings and rec:
        fh = findings["fight_health"]
        if fh.get("kind") == "foe_hp":
            lines.append(f"  # fight_health (unused: spatial progress "
                         f"already isolated): foe_hp candidate "
                         f"0x{fh['addr']:04X}")
        else:
            lines.append(f"  # fight_health: {fh['kind']} — {fh.get('note', '')}")
    if findings.get("round_gate") is not None and not round_gate_inlined:
        rg = findings["round_gate"]
        lines.append(f"  # round_gate: {rg['kind']} — {rg.get('note', '')}")
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
    if "fight_health" in findings:
        fh = findings["fight_health"]
        print(f"[fight_health] kind={fh['kind']} addr="
              f"{('0x%04X' % fh['addr']) if fh.get('addr') is not None else 'none'} "
              f"self_hp_conflict={fh.get('self_hp_conflict')} — {fh.get('note', '')}")
    if "round_gate" in findings:
        rg = findings["round_gate"]
        print(f"[round_gate] kind={rg['kind']} addr="
              f"{('0x%04X' % rg['addr']) if rg.get('addr') is not None else 'none'} "
              f"— {rg.get('note', '')}")
    if emit:
        print("\n--- paste-ready solve: ---")
        print(emit_solve_yaml(findings["rom"], findings))


# --------------------------------------------------------------------------
# Self-test against two already-adapted games (ground truth).
# --------------------------------------------------------------------------
def _selftest() -> int:
    import yaml
    cases = [
        # `expect_lives` is the POSITIVE control the death half of the
        # protocol had none of: without one, a scan that silently
        # recalls nothing still reports all_passed. Contra spends a
        # visible stock in the opening, so it can carry the assertion;
        # Kirby's opening is documented as too forgiving to take damage
        # at all, so there is nothing there to recall and asserting one
        # would only encode a wish.
        {"game": "Contra", "profile": "configs/contra.yaml",
         "expect_progress": (0x0065, 0x0064), "expect_saturates": False,
         "expect_room": 0x0064, "expect_lives": 0x0032},
        {"game": "Kirby", "profile": "configs/kirby.yaml",
         "expect_progress": (0x0083, 0x0095), "expect_saturates": True,
         "expect_room": 0x004F, "expect_lives": None},
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
        # Lives recall: the known counter must be nominated. Rank is not
        # asserted — the shortlist is what the protocol owes; picking the
        # head of it is a heuristic and would make this a tuning test.
        want_lives = c.get("expect_lives")
        hl = findings["hp_lives"]
        lives_cands = hl.get("lives_candidates") or []
        lives_ok = want_lives is None or any(
            cand["addr"] == want_lives or want_lives in cand.get("mirrors", [])
            for cand in lives_cands)
        passed = bool(found and sat_ok and room_hit and rec_ok and lives_ok)
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
            "hp_lives": {"kind": hl["kind"],
                         "addr": (f"${hl['addr']:04X}"
                                  if hl.get("addr") is not None else None)},
            "expect_lives": (f"${want_lives:04X}" if want_lives is not None
                             else None),
            "lives_recalled": lives_ok,
            "lives_candidates": [f"${cand['addr']:04X}" for cand in lives_cands],
            "death_evidence": hl.get("death_evidence"),
            "passed": passed,
        })
    receipt = {
        "tool": "scripts/discover_observables.py",
        "purpose": "self-test observable discovery against two ground-truth games",
        "provenance": "own scripted rollouts (settle scan + clean directional "
                      "+ validated idle + maneuver death-recovery + reload jump "
                      "+ uninterrupted death drives); no external RAM maps",
        "probe_budget": {"clean_forward": CLEAN_N, "clean_reverse": CLEAN_N,
                         "noop": NOOP_N, "advance": ADVANCE_N,
                         "settle_scan": SETTLE_SCAN,
                         "death_drives": DEATH_REPS * DEATH_MAX_N},
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
    ap.add_argument("--fight-gate", action="store_true",
                    help="Also run find_fight_health + find_round_gate "
                    "(FIGHTGATE_MECHANISM_2026-08-25.md §3) for camera-static "
                    "combat games with no spatial frontier at all (Punch-Out). "
                    "No effect on any other flag's output.")
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
                            seed=args.seed, fight_gate=args.fight_gate)
    _print_report(findings, emit=args.emit_solve)
    if args.out:
        Path(args.out).write_text(json.dumps(findings, indent=2, default=int) + "\n")
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
