#!/usr/bin/env python3
"""Count room transitions from the PPU blank-fold counter, and say which of
them arrived somewhere new.

WHY THIS EXISTS
---------------
`docs/research/RYGAR_CAMPAIGN_2026-08-26.md` §6 records: "At least 3 visually
distinct areas from power-on, counted by hand from our own rendered frames.
**No instrument in the pipeline can count them.**"

That conclusion was true of `odometer_scene` and false of the pipeline. The
scene counter is structurally blind on this profile -- `ppu.rs::odo_fold_frame`
takes the *blank* branch at every Rygar door and `return`s before the
scene-cut test ever runs -- but the very branch that shadows it increments
`odo_blank`, and that counter reads all 55 of the tape's transitions.

This module edge-detects that counter into runs, rejects the runs that are not
transitions, and reports whether the arriving area is one the trajectory has
already occupied.

WHAT IT IS NOT
--------------
**It is not a clear predicate, and it must never be wired as one.** See
`docs/receipts/rygar/clear_predicate_REFUTED.md`. On Rygar's own deepest banked
tape every one of the 55 transitions is a *reversible corridor door*: the
measured area key alternates perfectly between two values for all 56 segments,
27 round trips through one door. A predicate of the form "a transition
happened" fires 55 times on that tape and a `level_key`-style lexicographic
advance fires 28 times, on a trajectory whose first-visit frontier never moved
past 4,608 px. Either one would have banked 28-55 fabricated wins.

What this instrument is *for* is the step before a clear predicate: no Rygar
clear has ever been witnessed, so there is no positive to calibrate a predicate
against. This is the thing that will recognise and record the first genuinely
new area the search reaches -- the witness that would make minting one
possible.

CALIBRATION IS PER GAME AND MUST BE RE-EARNED
---------------------------------------------
`min_blank_frames` is not a property of the counter. It is a property of one
ROM's fade timings, measured from that ROM's own rollouts. Rygar's numbers
(`RYGAR_MIN_BLANK_FRAMES`) are derived in `docs/receipts/rygar/`; another game
gets its own or gets nothing.

WHAT IT REPORTS WHEN THE MECHANISM IS ABSENT
--------------------------------------------
`summary()["verdict"]` is `UNAVAILABLE` when the caller declares the odometer
off, and `UNINSTRUMENTED` when the counter never moved once across a whole
trajectory. Neither is reported as "0 transitions": a zero from a counter that
was never observed to move is not evidence of zero transitions, and this class
refuses to let a consumer read it as one.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

# --- per-observation verdicts -------------------------------------------
UNAVAILABLE = "unavailable"   # caller declared the odometer off
NONE = "none"                 # nothing closed this observation
IN_BLANK = "in_blank"         # a blank run is open
SPLICE = "splice"             # the counter stopped describing one timeline
SHORT = "short"               # a blank run closed below the transition floor
DEATH = "death"               # a blank run closed with a debounced death in it
REVISIT = "revisit"           # transition into an already-occupied area
NOVEL = "novel"               # transition into an area not yet occupied

TRANSITIONS = (REVISIT, NOVEL)

# --- Rygar's own calibration, from its own rollouts ----------------------
# Measured 2026-08-26 on `roms/Rygar (USA).nes` from its own start state.
# Death fade: 14 blank frames, 36 of 36 across 18 rollouts (2 per rollout --
# the fade and the game-over reload), holding out to 9,000 steps.
# Door transition: 78-79 blank frames, 55 of 55 on the banked R1 tape.
# Boot fade: 9 blank frames.
# The floor sits between the two clusters with a 2.9x margin below and a
# 2.0x margin above. It is NOT a constant of the counter.
RYGAR_MIN_BLANK_FRAMES = 40
# Rygar's lives byte blips to 0 for exactly 2 observations at every door,
# 55 of 55, and pins for hundreds to thousands at a real death. `>= 3` is
# the project-wide debounce and it sits on that boundary with a full
# observation of margin in both directions.
RYGAR_DEATH_DEBOUNCE = 3
# Area identity, derived blind: the segment-stability search in `discover`
# below, run over the banked tape, returns exactly 8 addresses that are
# constant inside every one of the 56 inter-blackout segments and vary
# across them. $0014 alone reads 2 areas; adding $001C reads 3, which is
# what §6 counted by eye. No disassembly, no RAM map, no walkthrough.
RYGAR_AREA_KEY_ADDRS = (0x0014, 0x001C)


class TransitionWitness:
    """Stream `odo_blank` in, get room transitions out.

    Feed every observation. The run history *is* the instrument; a consumer
    that pushes only on the frames it finds interesting gets nonsense.

    Parameters
    ----------
    min_blank_frames:
        A blank run shorter than this is not a transition. Per game, from
        that game's own rollouts. There is deliberately no default.
    frames_per_step:
        How many PPU frames one `push` covers -- `frame_skip`. The counter
        cannot legitimately advance by more than this in one observation, so
        a larger jump is a spliced timeline, not a blackout.
    death_debounce:
        Consecutive dead observations required before a run is called a
        death. Load-bearing in BOTH directions on Rygar: too low and all 55
        real transitions misfile as deaths (the 2-observation door blip);
        absent, and a death fade long enough to clear `min_blank_frames`
        would read as a transition.
    seen:
        Areas already occupied. `None` seeds from the first push. This is
        per-trajectory state -- build one witness per lineage.
    odometer_enabled:
        The caller declares whether the mechanism is on. When it is not,
        every push returns UNAVAILABLE and `summary()` refuses to report a
        count.
    """

    def __init__(self, min_blank_frames: int, frames_per_step: int,
                 death_debounce: int = 3, seen=None,
                 odometer_enabled: bool = True):
        if min_blank_frames < 1:
            raise ValueError("min_blank_frames must be >= 1; it is a per-game "
                             "calibration, not a switch")
        if frames_per_step < 1:
            raise ValueError("frames_per_step must be >= 1")
        self.min_blank_frames = int(min_blank_frames)
        self.frames_per_step = int(frames_per_step)
        self.death_debounce = int(death_debounce)
        self.odometer_enabled = bool(odometer_enabled)
        self.seen = set() if seen is None else {tuple(k) for k in seen}
        self._seeded = seen is not None
        self._prev_blank = None
        self._open = None          # dict while a blank run is open
        self._dead_streak = 0
        self._signal_observed = False
        self.observations = 0
        self.events = []           # one record per closed blank run
        self.splices = 0

    # -- the stream ------------------------------------------------------
    def push(self, blank, dead: bool, area_key) -> str:
        """One observation. Returns this observation's verdict.

        `blank`  -- `Pool.get_odometer_blank_per_worker()[w]`, or None if the
                    caller has no reading. None is NOT zero.
        `dead`   -- the profile's own `is_dead` for this observation. This
                    class does not know what a lives byte means.
        `area_key` -- any hashable identity tuple that is stable inside a
                    room. Rygar's is `RYGAR_AREA_KEY_ADDRS`.
        """
        if not self.odometer_enabled or blank is None:
            return UNAVAILABLE
        self.observations += 1
        key = tuple(area_key)
        if not self._seeded:
            self.seen.add(key)
            self._seeded = True
        self._dead_streak = self._dead_streak + 1 if dead else 0
        blank = int(blank)
        prev, self._prev_blank = self._prev_blank, blank
        if prev is None:
            return NONE
        delta = blank - prev
        # SPLICE. `odo_blank` rides inside OdoState, so a savestate restore
        # carries the *saved* count back in: the counter is per-trajectory,
        # never a monotone global. A decrease, or a jump larger than one
        # observation's worth of frames, means it stopped describing one
        # timeline. Drop any open run rather than closing a run whose two
        # ends came from different trajectories.
        if delta < 0 or delta > self.frames_per_step:
            self.splices += 1
            self._open = None
            return SPLICE
        if delta > 0:
            self._signal_observed = True
            if self._open is None:
                self._open = {"start": self.observations, "frames": 0,
                              "max_dead": self._dead_streak}
            self._open["frames"] += delta
            self._open["max_dead"] = max(self._open["max_dead"],
                                         self._dead_streak)
            return IN_BLANK
        if self._open is None:
            return NONE
        return self._close(key)

    def _close(self, key) -> str:
        run, self._open = self._open, None
        frames = run["frames"]
        # Two independent rejections, evaluated as a conjunction so neither
        # can shadow the other's evidence in the record.
        is_death = run["max_dead"] >= self.death_debounce
        long_enough = frames >= self.min_blank_frames
        if is_death or not long_enough:
            verdict = DEATH if is_death else SHORT
        else:
            verdict = REVISIT if key in self.seen else NOVEL
            self.seen.add(key)
        self.events.append({"verdict": verdict, "start": run["start"],
                            "end": self.observations, "frames": frames,
                            "max_dead": run["max_dead"], "area": list(key)})
        return verdict

    # -- the report ------------------------------------------------------
    def summary(self) -> dict:
        """What the whole stream said -- including when it said nothing.

        `verdict` is the field that keeps this honest:
          UNAVAILABLE     the caller declared the mechanism off.
          UNINSTRUMENTED  the counter never moved once. Zero transitions
                          here is not a measurement of zero transitions.
          OK              the counter moved; the counts below mean what
                          they say.
        """
        counts = collections.Counter(e["verdict"] for e in self.events)
        if not self.odometer_enabled:
            verdict = "UNAVAILABLE"
        elif not self._signal_observed:
            verdict = "UNINSTRUMENTED"
        else:
            verdict = "OK"
        return {
            "verdict": verdict,
            "signal_observed": self._signal_observed,
            "observations": self.observations,
            "transitions": counts[REVISIT] + counts[NOVEL],
            "novel": counts[NOVEL],
            "revisit": counts[REVISIT],
            "deaths": counts[DEATH],
            "short_runs": counts[SHORT],
            "splices": self.splices,
            # A trajectory that ends mid-blackout has evidence the witness
            # never got to classify. Say so rather than rounding it away.
            "open_run": self._open is not None,
            "areas": sorted(tuple(a) for a in self.seen),
            "min_blank_frames": self.min_blank_frames,
            "death_debounce": self.death_debounce,
            "frames_per_step": self.frames_per_step,
        }


def rygar_witness(**kw) -> TransitionWitness:
    """Rygar's calibrated witness. Overridable so the guard-removal tests
    can build a deliberately broken one and show what it costs."""
    kw.setdefault("min_blank_frames", RYGAR_MIN_BLANK_FRAMES)
    kw.setdefault("death_debounce", RYGAR_DEATH_DEBOUNCE)
    kw.setdefault("frames_per_step", 4)
    return TransitionWitness(**kw)


# ------------------------------------------------------------------------
# Stream banking / replay
# ------------------------------------------------------------------------
def rle(rows):
    out = []
    for r in rows:
        r = list(r)
        if out and out[-1][1:] == r:
            out[-1][0] += 1
        else:
            out.append([1] + r)
    return out


def unrle(rows):
    for r in rows:
        for _ in range(r[0]):
            yield r[1:]


def run_stream(witness: TransitionWitness, rows):
    """Push an `(blank, dead, k0, k1, ...)` stream through a witness."""
    verdicts = []
    for row in rows:
        verdicts.append(witness.push(row[0], bool(row[1]), tuple(row[2:])))
    return verdicts


# ------------------------------------------------------------------------
# CLI: blind discovery of the area key, and stream banking
# ------------------------------------------------------------------------
def _load_profile(repo, path):
    import yaml
    return yaml.safe_load((repo / path).read_text())


def _open_pool(repo, prof, n=1):
    import nes_core
    rom = repo / prof["solve"]["rom"]
    pool = nes_core.Pool(rom_path=str(rom), num_workers=n,
                         frame_skip=prof["frame_skip"])
    pool.set_headless(True)
    pool.set_odometer_enabled(True)
    pool.reset_all()
    st = (repo / prof["start_state_path"]).read_bytes()
    for w in range(n):
        pool.load_worker_state(w, st)
    return pool


def _replay_tape(repo, prof, actions, frames=False):
    """Replay a banked tape, returning per-observation RAM/blank/x."""
    import numpy as np
    from src.training.profile_utils import action_space_to_bitmasks
    bm = action_space_to_bitmasks(prof["action_space"])
    pool = _open_pool(repo, prof)
    try:
        ram, blank, ox, fr = [], [], [], []

        def rd(out):
            ram.append(np.frombuffer(out[0][2], np.uint8).copy())
            blank.append(int(pool.get_odometer_blank_per_worker()[0]))
            ox.append(int(pool.get_odometer_per_worker()[0][0]))
            if frames:
                fr.append(np.asarray(out[0][1], np.uint8).copy())

        rd(pool.step_all(np.zeros(1, dtype=np.uint8)))
        for a in actions:
            rd(pool.step_all(np.array([bm[int(a)]], dtype=np.uint8)))
        scene = int(pool.get_odometer_scene_per_worker()[0])
    finally:
        pool.shutdown()
    return (np.stack(ram), np.array(blank), np.array(ox), scene,
            np.stack(fr) if frames else None)


def _blank_runs(blank, frames_per_step):
    """Edge-detect a blank counter into runs. Never read the raw value as a
    count -- 4,329 on Rygar's tape is blank FRAMES, not 4,329 transitions."""
    runs, i = [], 0
    d = [int(blank[k + 1]) - int(blank[k]) for k in range(len(blank) - 1)]
    while i < len(d):
        if 0 < d[i] <= frames_per_step:
            j = i
            total = 0
            while j < len(d) and 0 < d[j] <= frames_per_step:
                total += d[j]
                j += 1
            runs.append((i + 1, j, total))
            i = j
        else:
            i += 1
    return runs


def cmd_discover(args):
    """Blind search for the area key: which RAM bytes are constant inside
    every inter-blackout segment and vary between them?

    This is a statistical search over the profile's own rollout, the same
    class as `discover_observables.py` and `find_wrap_pair.py`. Purity
    (Tier 3): no disassembly, no RAM map, no walkthrough, no recall of the
    title. A party who had never seen this game could run exactly this.
    """
    repo = pathlib.Path(args.repo).resolve()
    prof = _load_profile(repo, args.profile)
    tape = json.loads((repo / args.tape).read_text())
    ram, blank, ox, scene, fr = _replay_tape(repo, prof, tape["actions"],
                                             frames=True)
    fs = prof["frame_skip"]
    runs = _blank_runs(blank, fs)
    print(f"blank runs: {len(runs)}  lengths="
          f"{dict(collections.Counter(r[2] for r in runs))}")
    print(f"odometer_scene over the whole tape: {scene}  "
          f"(the counter this instrument exists to replace)")
    segs, prev_end = [], runs[0][1]
    for a, b, _ in runs[1:]:
        if a - 1 > prev_end + 2:
            segs.append((prev_end + 1, a - 1))
        prev_end = b
    segs.append((prev_end + 1, len(ram) - 1))
    print(f"segments between blackouts: {len(segs)}")
    cands = []
    for ad in range(ram.shape[1]):
        vals = []
        for a, b in segs:
            lo = a + max(2, (b - a) // 4)
            col = ram[lo:b + 1, ad]
            if len(col) == 0 or col.min() != col.max():
                vals = None
                break
            vals.append(int(col[0]))
        if vals and len(set(vals)) > 1:
            cands.append((len(set(vals)), ad, vals))
    cands.sort(reverse=True)
    print(f"\naddresses constant in ALL {len(segs)} segments and varying "
          f"across them: {len(cands)}")
    for dv, ad, vals in cands:
        mono = all(y >= x for x, y in zip(vals, vals[1:]))
        print(f"  ${ad:04X}  distinct={dv}  monotone={mono}  {vals[:14]}...")
    for k in (1, 2, 3):
        sel = [ad for _, ad, _ in cands[:k]]
        keys = [tuple(int(ram[b, a]) for a in sel) for _, b in segs]
        print(f"  key {[hex(a) for a in sel]} -> {len(set(keys))} areas "
              f"{sorted(set(keys))}")

    # CROSS-CHECK AGAINST PIXELS. A RAM key that partitions the segments is
    # only an AREA key if the screen agrees. Cluster each segment's midpoint
    # frame by the key and compare within-cluster spread to between-cluster
    # distance. The comparison has to survive a long SCROLLING area, whose
    # frames legitimately differ from each other end to end -- so the bar is
    # that the largest intra stays below the smallest inter, not that intra
    # is near zero.
    sel = [a for a in args.area_key_addrs] if args.area_key_addrs else \
        [ad for _, ad, _ in cands[:2]]
    import numpy as np
    groups = collections.defaultdict(list)
    for a, b in segs:
        key = tuple(int(ram[b, ad]) for ad in sel)
        groups[key].append(fr[(a + b) // 2].astype("int16"))
    print(f"\n  frame cross-check on key {[hex(a) for a in sel]}:")
    means, intra = {}, {}
    for key, imgs in sorted(groups.items()):
        arr = np.stack(imgs)
        means[key] = arr.mean(0)
        intra[key] = float(np.mean([abs(x - means[key]).mean() for x in arr]))
        print(f"    area {key}: {len(imgs):3d} segments  intra="
              f"{intra[key]:6.2f}")
    ks = sorted(means)
    inter = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            dist = float(abs(means[ks[i]] - means[ks[j]]).mean())
            inter.append(dist)
            print(f"    inter {ks[i]} vs {ks[j]}: {dist:6.2f}")
    if inter:
        print(f"    max intra {max(intra.values()):.2f} < min inter "
              f"{min(inter):.2f}: {max(intra.values()) < min(inter)}")
    return 0


def cmd_bank(args):
    """Bank the observation streams the tests replay: the R1 tape, and death
    rollouts. Tracked, so the tests do not need the ROM."""
    import numpy as np
    from src.training.profile_utils import action_space_to_bitmasks
    repo = pathlib.Path(args.repo).resolve()
    prof = _load_profile(repo, args.profile)
    tape = json.loads((repo / args.tape).read_text())
    addrs = [int(a, 0) for a in args.area_key]
    fs = prof["frame_skip"]
    lives_addr = int(prof["solve"]["lives"])

    ram, blank, ox, scene, _ = _replay_tape(repo, prof, tape["actions"])
    lives0 = int(ram[0, lives_addr])
    tape_rows = [[int(blank[i]), int(1 <= (lives0 - int(ram[i, lives_addr]))
                                     % 256 <= 8)]
                 + [int(ram[i, a]) for a in addrs] for i in range(len(ram))]

    # Death rollouts: scripted right/noop/left plus seeded uniform-random.
    n = args.death_workers
    bm = action_space_to_bitmasks(prof["action_space"])
    pool = _open_pool(repo, prof, n)
    rng = np.random.default_rng(args.seed)
    cols = [[] for _ in range(n)]
    try:
        for _ in range(args.death_steps):
            a = np.zeros(n, np.uint8)
            a[0], a[1], a[2] = bm[1], bm[0], bm[2]
            for w in range(3, n):
                a[w] = bm[int(rng.integers(0, len(bm)))]
            out = pool.step_all(a)
            bl = pool.get_odometer_blank_per_worker()
            for w in range(n):
                r = np.frombuffer(out[w][2], np.uint8)
                cols[w].append([int(bl[w]),
                                int(1 <= (lives0 - int(r[lives_addr])) % 256
                                    <= 8)] + [int(r[a2]) for a2 in addrs])
    finally:
        pool.shutdown()

    doc = {
        "what_this_is":
            "Per-observation (odo_blank, is_dead, area_key...) streams for "
            "Rygar, run-length encoded. Banked so tests/test_transition_"
            "witness.py exercises the real mechanism on real measurements "
            "without the ROM, which is gitignored and not distributable.",
        "ledger": "EXHIBITION",
        "provenance": {
            "rom_sha256": hashlib.sha256(
                (repo / prof["solve"]["rom"]).read_bytes()).hexdigest(),
            "start_state_sha256": hashlib.sha256(
                (repo / prof["start_state_path"]).read_bytes()).hexdigest(),
            "tape": args.tape,
            "frame_skip": fs,
            "lives_addr": lives_addr,
            "lives_at_start": lives0,
            "area_key_addrs": [f"0x{a:04X}" for a in addrs],
            "death_seed": args.seed,
            "death_steps": args.death_steps,
        },
        "tape": {"terminal_odometer_x": int(ox[-1]),
                 "terminal_odo_blank": int(blank[-1]),
                 "odometer_scene": scene,
                 "rows": rle(tape_rows)},
        "deaths": [{"policy": p, "rows": rle(c)}
                   for p, c in zip(["right", "noop", "left"]
                                   + [f"random{w}" for w in range(3, n)],
                                   cols)],
    }
    out = repo / args.out
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    w = rygar_witness()
    run_stream(w, list(unrle(doc["tape"]["rows"])))
    print("tape summary:", json.dumps(w.summary()))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", default=str(pathlib.Path(__file__).resolve()
                                         .parent.parent))
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover", help="blind search for the area key")
    d.add_argument("--profile", default="configs/rygar.yaml")
    d.add_argument("--tape", default="docs/receipts/rygar/r1_tape_gx6242.json")
    d.add_argument("--area-key-addrs", nargs="*", type=lambda s: int(s, 0),
                   default=list(RYGAR_AREA_KEY_ADDRS),
                   help="addresses to cross-check against pixels; empty "
                        "uses the top two the search itself nominated")
    d.set_defaults(fn=cmd_discover)
    b = sub.add_parser("bank", help="bank the observation streams")
    b.add_argument("--profile", default="configs/rygar.yaml")
    b.add_argument("--tape", default="docs/receipts/rygar/r1_tape_gx6242.json")
    b.add_argument("--area-key", nargs="+", default=["0x14", "0x1c"])
    b.add_argument("--death-workers", type=int, default=18)
    b.add_argument("--death-steps", type=int, default=1400)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--out",
                   default="docs/receipts/rygar/transition_streams.json")
    b.set_defaults(fn=cmd_bank)
    args = p.parse_args(argv)
    sys.path.insert(0, args.repo)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
