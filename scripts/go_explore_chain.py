"""Hands-off Go-Explore CHAIN: solve consecutive SMB levels from clean
entrances, extracting each next entrance from the prior level's clear.

For each level: subprocess go_explore_solve.py from the current entrance until
it produces a solution (a warp-guarded forward clear); replay that solution to
the level transition, settle a few frames, and snapshot the NEXT level's clean
entrance; repeat. Stops on a stall (a level not solved within the per-level
budget) or when max_levels is reached. Every entrance + solution is banked, so
the whole progression is receipted and resumable.

This demonstrates the search agent clearing level after level from power-on-style
clean entrances (no mid-level seeds), machine-busy and hands-off.

Chain-scope sequence invariant: the driver keeps its own history of every
SETTLED level key it has banked (visited_keys.json, next to chain.jsonl) and
refuses a stage that settles on a key this campaign already visited — a
re-entered room is not a new level, and banking one fabricates progress the
campaign never made. Stage k+1 is only ever attempted from a banked stage k.
--allow-revisit turns the refusal off for campaigns that legitimately backtrack.

The history is POSITIONED at --start-state's own settled key: a re-run keeps
the campaign's walk up to the entrance it resumes from and nothing after it,
so a crash-restart or a deliberate re-do of an earlier stage re-walks rather
than refusing the rooms it was launched to re-solve.

Usage:
  python scripts/go_explore_chain.py --start-state <entrance.state> \
      --start-label 2-2 --profile configs/mario_1_2_solo.yaml \
      --out runs/ge_chain --max-levels 12 --minutes-per-level 20 --workers 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from scripts.go_explore_solve import (  # noqa: E402
    apply_hw_flags, hw_provenance, make_game, resolve_hw_flags, sidecar_path,
    write_state_sidecar,
)
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

SOLVE = str(REPO / "scripts" / "go_explore_solve.py")

# Campaign-scope visited-key history, banked next to chain.jsonl.
VISITED_FILE = "visited_keys.json"


def _norm_key(key) -> tuple:
    """Level keys arrive as tuples from the game adapters and as lists
    from a JSON round-trip; membership has to see one shape."""
    return tuple(int(x) for x in key)


def _history_from_chain_log(chain_log: Path) -> list:
    """The campaign's walk, oldest first, rebuilt from an existing
    chain.jsonl (each banked stage records its `next_wd`).

    A campaign dir written before the history file existed has no
    visited_keys.json, so the chain log is the fallback source of truth.
    This returns the WHOLE recorded walk; `_position_history` decides how
    much of it the current run is actually standing behind."""
    recs: list = []
    if not chain_log.is_file():
        return recs
    for line in chain_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue          # a torn last line from a killed run
        if r.get("next_wd"):
            recs.append({"key": list(_norm_key(r["next_wd"])),
                         "label": r.get("next_label"),
                         "after": r.get("level")})
    return recs


def _clean_history(recs) -> list:
    """Normalise loaded records and drop any that carry no usable key.
    The driver indexes every record's `key`, so one hand-edited or
    half-written entry must not crash an unattended campaign."""
    clean = []
    for r in recs if isinstance(recs, list) else []:
        if not isinstance(r, dict):
            continue
        try:
            key = list(_norm_key(r["key"]))
        except (KeyError, TypeError, ValueError):
            continue
        clean.append({"key": key, "label": r.get("label"),
                      "after": r.get("after")})
    return clean


def _position_history(recs: list, root_key) -> list:
    """The prefix of a recorded walk that the run STARTING FROM
    `root_key` is actually standing behind.

    A recorded walk is an ordered sequence, and a re-run does not
    necessarily resume at its end. Handing back the whole thing
    regardless is what turns a crash-restart into a false campaign end:
    re-run the identical command from the campaign's original root, the
    stage-0 solve succeeds, the extraction settles on the key the log
    already records, the invariant refuses it as a revisit, and the
    driver writes `next: null` — a whole per-level solve budget burned,
    unattended, for a run that should simply have re-walked.

    So the history is truncated at the start state's own settled key:
    everything up to and including it is genuinely behind this run, and
    everything after it is a stage this run has NOT made yet and must be
    free to re-bank. When the key is unknown (a hand-captured root with
    no sidecar) or absent from the receipts (a restart from the
    campaign's own root, a root captured elsewhere), nothing is behind
    this run and the history is empty — the invariant then guards only
    what this run banks itself, which is the pre-history behavior and
    never fabricates a campaign end.

    The FIRST occurrence wins: with the invariant on a key cannot be
    banked twice, and if a `--allow-revisit` campaign did bank one twice
    the shorter prefix is the conservative read."""
    if root_key is None:
        return []
    rk = _norm_key(root_key)
    for i, r in enumerate(recs):
        if _norm_key(r["key"]) == rk:
            return recs[:i + 1]
    return []


def _history_sources(out: Path):
    """The campaign dir's recorded walks, best receipt first: the banked
    history file, then chain.jsonl. The log is consulted when the file is
    missing or unreadable, and also when the file simply does not reach
    the entrance being resumed from (a dir re-walked from an earlier
    stage has a file shorter than its own log)."""
    p = out / VISITED_FILE
    if p.is_file():
        try:
            yield _clean_history(json.loads(p.read_text()).get("visited"))
        except (OSError, ValueError, AttributeError) as e:
            print(f"[chain] WARNING: unreadable {p} ({e}); rebuilding the "
                  f"visited-key history from chain.jsonl", flush=True)
    yield _history_from_chain_log(out / "chain.jsonl")


def load_key_history(out: Path, root_key=None) -> list:
    """The SETTLED level keys this campaign banked BEFORE arriving at
    `root_key`, oldest first, ending with `root_key` itself. Records are
    {key, label, after}; `after` is the level whose clear produced the
    key (None = the campaign's own root).

    `root_key` is the settled key of the state this run starts from (see
    `root_key_record`). It positions the history — see
    `_position_history` for why an unpositioned rebuild dead-ends any
    re-run that does not resume at the last banked entrance. Unknown
    (the default) means no recorded stage can be proven to be behind
    this run, so the history is empty."""
    if root_key is None:
        return []      # nothing to position against; don't even read
    for recs in _history_sources(out):
        positioned = _position_history(recs, root_key)
        if positioned:
            return positioned
    return []


def save_key_history(out: Path, recs: list) -> Path:
    p = out / VISITED_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"visited": list(recs)}, indent=2) + "\n")
    return p


def root_key_record(start_state, label: str):
    """The campaign's root room is visited by definition — the chain is
    standing in it before the first solve. A chain-produced entrance
    carries its settled key in the sidecar written next to the blob, so
    a resumed or handed-off campaign can seed the history with it and
    refuse a later transit BACK to the root. That key is also what
    POSITIONS a resumed campaign's history (`_position_history`). A
    hand-captured blob has no sidecar key; the root then simply is not
    in the history and only judge_transition's start_key check guards
    it. Returns None when there is nothing trustworthy to seed from."""
    try:
        rec = json.loads(sidecar_path(start_state).read_text())
    except (OSError, ValueError):
        return None
    key = rec.get("settled_key")
    if not key:
        return None
    try:
        return {"key": list(_norm_key(key)), "label": label, "after": None}
    except (TypeError, ValueError):
        return None


def extract_next_entrance(profile, root_bytes: bytes, actions, out_path: Path,
                          settle: int = 8, stable_for: int = 45,
                          settle_cap: int = 600, hw_flags=(),
                          transit_timeout_frames: int = 0,
                          visited=(), allow_revisit: bool = False):
    """Replay `actions` from the root; at the first level-key transition
    that lands on a genuinely NEW, ALIVE, stable state, snapshot it as the
    next level's entrance. Returns (path, next_key) or (None, None) if no
    such transition is ever reached.

    The old fixed 8-step settle could snapshot MID-TRANSITION: Bubble
    Bobble's umbrella warp animates ~400 frames with the round counter
    ticking through values, and a mid-warp "entrance" seeded the next
    solve inside an animation whose counter blips stalled it outright
    (round 52, 2026-08-06). Stability-settling lands the snapshot in the
    destination level's actual play. `settle` no-ops still run first
    (rendering-stability, the original purpose).

    Two more failure shapes surfaced the same day settling on Bubble
    Bobble round 9's arcade GAME OVER / PASSWORD screen (a live-show
    stream stall of several minutes): (1) a transient level-key blip
    during a death animation can drift back to the SAME value it started
    at, which settles "stable" but isn't a forward transition at all —
    reject and keep scanning if the settled key == start_key; (2) the
    settled state can be dead (GAME OVER: the round HUD digit freezes at
    whatever it last showed, satisfying both the not-equal-to-start check
    on the transient blip and the later stability check, while lives has
    already hit 0) — reject via the game adapter's own is_dead() using
    lives captured at the true root, before any transition. A doomed
    landing is treated as if no transition were seen on this action
    range; the search resumes scanning the rest of `actions`.

    REJECTION MUST REWIND. Settling is destructive: it steps up to
    `settle + settle_cap` NOOPs into the pool. When the settle is then
    REJECTED, "keep scanning `actions`" is only honest if the pool is
    put back where the blip fired — otherwise the remaining actions
    replay against a machine that idled for hundreds of steps and the
    real clear never reproduces. Measured on Bubble Bobble round 67
    (2026-08-09): the round counter $0401 blips DOWN one step to the
    previous round's value at replay step 3, the settle burned 53 steps,
    settled back on the start key, and the 200 remaining actions then
    ran desynchronised — the genuine 67->68 transition on the very last
    action was invisible and the chain reported "no forward transition
    captured after 67". Round 66 survived the same blip only by luck:
    its blip landed at action 438 of 440, so the destructive settle
    coasted into the next round instead of wrecking a replay tail. The
    rewind is a save/restore round-trip around the settle.

    `transit_timeout_frames` extends the scan PAST the end of `actions`
    with NOOPs (converted to steps by `frame_skip`) for games whose
    level/round byte only turns over after an uncontrollable interlude
    that outlasts the solution trace. 0 (the default) stops at the last
    action, exactly as before.

    `hw_flags` must be the SAME set the solve ran under: this replay
    machine is what stamps the next level's entrance blob, so a
    mismatch bakes a foreign-machine state into the chain and every
    downstream level inherits it. Empty (the default) reproduces the
    pre-existing behavior exactly.

    `visited` is the CAMPAIGN-SCOPE sequence invariant: the settled keys
    the chain has already banked. "Not the key we started this level on
    and not dead" is too weak a definition of forward progress — a room
    the campaign already passed through satisfies both, so a re-entered
    or revisited room banks as a new stage and the chain reports
    progress it never made (the Kirby fabrication class). A settled key
    already in `visited` is REFUSED and logged; the scan rewinds and
    keeps looking for a genuinely new room. Empty (the default) means
    no key can be a revisit, so a first level — and every existing
    caller that passes nothing — behaves exactly as before.
    `allow_revisit=True` turns the check off for campaigns that really
    do re-enter rooms (hub worlds, deliberate backtracking).

    A refusal NEVER blacklists the key's VALUE for the rest of the scan.
    Every action whose live reading leaves `start_key` is settled and
    judged on its own, however many times the same reading recurs. The
    reason is the round-67 receipt again from the other side: the live
    reading at the instant a genuine transit fires is a TRANSIENT, and
    it can equal a room the campaign really has visited (BB's counter
    dips to the previous round's number). Screening that raw reading
    against previously-refused settled keys skips the settle entirely,
    so the one action carrying the real transition is never judged and
    the stage reports "no forward transition" — the same lost-transition
    class this function was fixed for, re-created by an optimization.
    The cost of judging instead is bounded and small: one settle
    (`settle` + at most `settle_cap` NOOP steps, ~53 at the defaults
    once the key stabilises) per action that left the start key, against
    a per-level solve budget measured in minutes."""
    game = make_game(profile)
    bm = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))
    pool = Pool(rom_path=game.rom, num_workers=1, frame_skip=fs)
    apply_hw_flags(pool, hw_flags)   # before reset_all(); order is load-bearing
    pool.set_headless(True)
    pool.reset_all()
    pool.load_worker_state(0, root_bytes)

    def step(a):
        x = np.zeros(1, dtype=np.uint8)
        x[0] = bm[a]
        return pool.step_all(x)[0][2]

    r = step(0)
    start_key = game.level_key(r)
    start_lives = game.lives(r)

    # Campaign history + the keys this replay has already refused as
    # revisits. `refused` is a LOG/REPORTING set only — it must never
    # gate whether an action gets judged (see the docstring). Both are
    # empty unless a caller opts in, so the default scan is byte-identical.
    seen = set() if allow_revisit else {_norm_key(k) for k in visited}
    refused: set = set()

    def judge_transition():
        """NOOP-settle the candidate transition and rule on it. Returns
        the settled key on acceptance; None on rejection, having REWOUND
        the pool to the pre-settle point so the caller's replay of the
        remaining actions stays in sync."""
        mark = pool.save_worker_state(0)
        rr = None
        for _ in range(settle):
            rr = step(0)
        # Settle until the key stops moving (transition fully over).
        key = game.level_key(rr)
        stable = 0
        for _ in range(settle_cap):
            if stable >= stable_for:
                break
            rr = step(0)
            k = game.level_key(rr)
            if k == key:
                stable += 1
            else:
                key, stable = k, 0
        settled_key = game.level_key(rr)
        if settled_key == start_key or game.is_dead(rr, start_lives):
            pool.load_worker_state(0, mark)
            return None    # false/doomed transition — scan resumes in sync
        # The sequence invariant is judged on the SETTLED key, never on
        # the transient reading that triggered the settle: Bubble Bobble
        # round 67's counter dips to 66 for a step, and a guard reading
        # that raw blip would refuse the campaign's own history back at
        # itself. Only settled keys are compared, and only settled keys
        # are ever recorded by the driver.
        nk = _norm_key(settled_key)
        if nk in seen:
            if nk not in refused:
                print(f"[chain] *** REVISIT REFUSED *** settled level key "
                      f"{nk} is ALREADY in this campaign's visited history "
                      f"— a re-entered room is not forward progress. "
                      f"Scanning on (pass --allow-revisit to bank it).",
                      flush=True)
            refused.add(nk)
            pool.load_worker_state(0, mark)
            return None
        return settled_key

    # NOOP tail past the end of the trace, for level bytes that only turn
    # over after an interlude the solution does not cover. 0 = stop at the
    # last action (the pre-existing behavior).
    tail = max(0, int(transit_timeout_frames)) // max(1, fs)
    result = (None, None)
    for a in list(actions) + [0] * tail:
        r = step(int(a))
        k_now = game.level_key(r)
        if k_now == start_key:
            continue
        # No screening of `k_now` against anything beyond start_key. A
        # raw reading is a transient; a previously-refused SETTLED key
        # can recur as the trigger reading of a genuine transit, and
        # skipping it would silently lose that transit (docstring).
        settled_key = judge_transition()
        if settled_key is None:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(pool.save_worker_state(0))
        # Sidecar, not blob: record the machine this entrance was
        # produced on so the next solve can be run on the same one.
        write_state_sidecar(out_path, hw_provenance(hw_flags, fs),
                            {"settled_key": list(settled_key)})
        result = (str(out_path), settled_key)
        break
    if result[0] is None and refused:
        print(f"[chain] sequence invariant refused {len(refused)} revisited "
              f"key(s) this replay: {sorted(refused)}", flush=True)
    pool.shutdown()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--start-label", required=True, help="e.g. 2-2")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-levels", type=int, default=12)
    ap.add_argument("--minutes-per-level", type=float, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # Coverage-recipe passthrough to each per-level solve. Defaults
    # reproduce the plain solver exactly (sel-mode legacy, gx-bucket 16,
    # y-band 32); pass --sel-mode count (+ finer buckets) to grind the
    # hard-exploration levels that walled the plain chain (e.g. Lost
    # Levels 1-2, which the coverage ladder cracked where defaults froze).
    ap.add_argument("--sel-mode", choices=("legacy", "count"), default="legacy")
    ap.add_argument("--gx-bucket", type=int, default=16)
    ap.add_argument("--y-band", type=int, default=32)
    # Burst length (steps) per exploration rollout. Default 64 = the solver
    # default. Raise for games with long uncontrollable transition
    # animations: Bubble Bobble's umbrella item warps forward across a
    # ~400-frame animation, and a 64-step (256-frame at frame_skip 4)
    # burst ends INSIDE it — every burst re-archives the same mid-warp
    # cell and the chain stalls at 1 cell forever (round 52, 2026-08-06).
    ap.add_argument("--burst", type=int, default=64)
    # On a per-level stall, retry the SAME level with a fresh seed
    # (seed+1, seed+2, ...) up to this many times before declaring a real
    # stall. A coverage-reachable wall the solver freezes at is often a
    # low-probability stochastic passage that different exploration
    # randomness finds — e.g. Lost Levels 2-1 froze at gx~2985 on seed 0
    # but cleared on seed 1 with identical params. 0 = no retries (old
    # behavior).
    ap.add_argument("--seed-retries", type=int, default=0)
    # Room-transition domination cap passthrough (see go_explore_solve.py's
    # --sect-cap). Default 16 = byte-identical to every existing SMB1
    # caller. This chain driver is the primary unattended-multi-level
    # caller the knob needs to reach but didn't (found by adversarial
    # review, 2026-08-06) — Lost Levels needed --sect-cap 64 by hand
    # outside the chain to avoid silently freezing half the cell key.
    ap.add_argument("--sect-cap", type=int, default=16)
    # nes_core hw timing flags for the WHOLE chain — the per-level solves
    # and this driver's own entrance-extraction replay must run the same
    # machine, or the entrance blob this banks came from a machine the
    # next solve never reproduces (the CV cv_chain_hw2 failure mode).
    # Default: whatever the profile pins, else NONE (byte-identical to
    # every existing chain run). See go_explore_solve.py --hw-flags.
    ap.add_argument("--hw-flags", type=str, default=None, metavar="a,b,c")
    # Entrance extraction keeps scanning for the forward level-key
    # transition this many FRAMES past the end of the solution trace,
    # stepping NOOPs. For games that end a level on an uncontrollable
    # interlude (bonus/round-clear screens) the byte can turn over after
    # the trace stops. 0 = stop at the last action (pre-existing
    # behavior). Frames, not steps: divided by the profile's frame_skip.
    ap.add_argument("--transit-timeout-frames", type=int, default=0)
    # Sequence invariant (chain scope). The driver keeps its own history
    # of every SETTLED key it has banked (visited_keys.json, next to
    # chain.jsonl) and refuses a stage whose settled key is already in
    # it: a re-entered room is not a new level, and banking one fabricates
    # progress the campaign never made. On by default — the history is
    # positioned at --start-state's own settled key, so a NEW campaign
    # (and any re-run that is not a true continuation) starts with only
    # its own root and the first level is unaffected. Pass this to bank
    # revisits anyway (hub worlds, deliberate backtracking).
    # GATE-OPENER passthrough. The receipted dead-flag lineage this
    # closes: --kill-key and --ortho are BOTH unreachable through this
    # driver today (it hardcodes twelve flags), exactly as eval's
    # --start-state and --sect-cap were before they were found dead. A
    # per-level arm nobody can reach from the campaign driver is an arm
    # that never runs. Defaults leave the solver's own defaults alone:
    # --gate-opener off, no sidecar, no attestation.
    ap.add_argument("--gate-opener", choices=("off", "enumerate"),
                    default="off")
    ap.add_argument("--gate-sweep-frac", type=float, default=0.10)
    ap.add_argument("--gate-sweep-roots", type=int, default=16)
    ap.add_argument("--gate-sweep-repeats", type=int, default=2)
    ap.add_argument("--gate-pin-secs", type=float, default=None)
    ap.add_argument("--gate-target-typed", action="store_true")
    ap.add_argument("--gate-band", type=int, default=24)
    ap.add_argument("--gate-sham-roots", type=int, default=4)
    ap.add_argument("--gate-axes", type=str, default=None)
    ap.add_argument("--contact-bits", type=int, default=3)
    ap.add_argument("--allow-revisit", action="store_true",
                    help="bank a stage even when its settled level key is "
                         "already in this campaign's visited history")
    args = ap.parse_args()
    if args.gate_opener != "off" and args.gate_pin_secs is None:
        ap.error("--gate-pin-secs is REQUIRED when --gate-opener is not off "
                 "(mined defaults: 600 CV / 120 BB; negative disables).")

    profile = yaml.safe_load(Path(args.profile).read_text())
    hw_flags = resolve_hw_flags(profile, args.hw_flags)
    if hw_flags:
        print(f"[chain] hw flags: {hw_flags}", flush=True)
    game = make_game(profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chain_log = out / "chain.jsonl"

    # Campaign-scope visited-key history, POSITIONED at the entrance this
    # run actually starts from: the campaign dir's receipts up to and
    # including the start state's own settled key, and nothing after it.
    # A true continuation therefore keeps everything it already banked,
    # while a restart (same root, after a crash) or a deliberate re-do of
    # an earlier stage re-walks with a clean history instead of refusing
    # the very rooms it was launched to re-solve.
    root_rec = root_key_record(args.start_state, args.start_label)
    history = load_key_history(out, root_rec["key"] if root_rec else None)
    if root_rec and not history:
        # The start state's key is nowhere in this dir's receipts (a new
        # campaign, or a root captured elsewhere). The chain is standing
        # in that room before the first solve, so it is visited by
        # definition and a transit BACK to it is refusable from stage 0.
        history = [dict(root_rec)]
    visited = [_norm_key(h["key"]) for h in history]
    if args.allow_revisit:
        print("[chain] --allow-revisit: the sequence invariant is OFF",
              flush=True)
    elif visited:
        print(f"[chain] sequence invariant ON: {len(visited)} key(s) already "
              f"visited this campaign; a stage settling on one is refused",
              flush=True)
    if history:
        save_key_history(out, history)

    cur_state = str(args.start_state)
    cur_label = args.start_label
    solved = []
    for i in range(args.max_levels):
        base_out = out / f"lvl_{i:02d}_{cur_label}"
        print(f"\n===== CHAIN level {i}: solving {cur_label} from "
              f"{cur_state} =====", flush=True)
        # Try seed, then seed+1..seed+seed_retries on a stall. Each attempt
        # gets its own out dir so a fresh seed searches from scratch (no
        # archive carry-over from the frozen attempt).
        sols = []
        lvl_out = base_out
        for attempt in range(args.seed_retries + 1):
            seed = args.seed + attempt
            lvl_out = base_out if attempt == 0 else Path(f"{base_out}_s{seed}")
            if attempt > 0:
                print(f"[chain] {cur_label} stalled; retry {attempt}/"
                      f"{args.seed_retries} with seed {seed}", flush=True)
            cmd = [
                sys.executable, SOLVE, "--out", str(lvl_out),
                "--root-state", cur_state, "--profile", args.profile,
                "--workers", str(args.workers),
                "--minutes", str(args.minutes_per_level),
                "--want-solutions", "1", "--seed", str(seed),
                "--sel-mode", args.sel_mode,
                "--gx-bucket", str(args.gx_bucket),
                "--y-band", str(args.y_band),
                "--burst", str(args.burst),
                "--sect-cap", str(args.sect_cap),
                # Pass the RESOLVED set explicitly rather than the raw
                # CLI string: the subprocess then cannot re-resolve to
                # something different from the machine this driver uses
                # for entrance extraction. Empty resolves to 'none',
                # which sets nothing (the pre-existing behavior).
                "--hw-flags", ",".join(hw_flags) or "none",
                # Every --gate-* knob, echoed so the per-level receipt
                # records the arm the campaign actually ran.
                "--gate-opener", str(args.gate_opener),
                "--gate-sweep-frac", str(args.gate_sweep_frac),
                "--gate-sweep-roots", str(args.gate_sweep_roots),
                "--gate-sweep-repeats", str(args.gate_sweep_repeats),
                "--gate-band", str(args.gate_band),
                "--gate-sham-roots", str(args.gate_sham_roots),
                "--contact-bits", str(args.contact_bits),
            ]
            # Value-less / optional flags stay OFF the command line at
            # their defaults, so a chain that never asked for the arm
            # invokes the solver with exactly the argv it always did
            # (plus the inert --gate-* defaults above).
            if args.gate_pin_secs is not None:
                cmd += ["--gate-pin-secs", str(args.gate_pin_secs)]
            if args.gate_target_typed:
                cmd += ["--gate-target-typed"]
            if args.gate_axes:
                cmd += ["--gate-axes", str(args.gate_axes)]
            subprocess.run(cmd, check=False)
            sols = sorted(lvl_out.glob("solutions/sol_*.actions.npy"))
            if sols:
                break
        if not sols:
            print(f"[chain] STALL: {cur_label} not solved in "
                  f"{args.minutes_per_level} min x {args.seed_retries + 1} seed(s) "
                  f"— chain stops at {len(solved)} levels.", flush=True)
            with open(chain_log, "a") as f:
                f.write(json.dumps({"level": cur_label, "status": "stall"}) + "\n")
            break
        actions = np.load(sols[0]).astype(np.int64)
        nxt_path = out / "entrances" / f"entrance_after_{cur_label}.state"
        npath, nwd = extract_next_entrance(
            profile, Path(cur_state).read_bytes(), actions, nxt_path,
            hw_flags=hw_flags,
            transit_timeout_frames=args.transit_timeout_frames,
            visited=visited, allow_revisit=args.allow_revisit)
        rec = {"level": cur_label, "status": "solved",
               "actions": int(len(actions)), "solution": str(sols[0]),
               "hw_flags": list(hw_flags)}
        solved.append(cur_label)
        print(f"[chain] SOLVED {cur_label} ({len(actions)} actions)", flush=True)
        if npath is None:
            rec["next"] = None
            with open(chain_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[chain] no forward transition captured after {cur_label}; "
                  f"chain stops ({len(solved)} solved).", flush=True)
            if visited and not args.allow_revisit:
                print("[chain] (the sequence invariant refuses keys this "
                      "campaign already visited — --allow-revisit banks "
                      "them)", flush=True)
            break
        rec["next_wd"] = list(nwd)
        rec["next_label"] = game.label(nwd)
        rec["next_entrance"] = npath
        with open(chain_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # Only a SETTLED, accepted key enters the history — a transient
        # level-byte blip never reaches here (judge_transition rejects it
        # before the bank), so the campaign's invariant cannot be poisoned
        # by a down-blip like Bubble Bobble's 67->66->67.
        visited.append(_norm_key(nwd))
        history.append({"key": list(_norm_key(nwd)),
                        "label": game.label(nwd), "after": cur_label})
        save_key_history(out, history)
        cur_state = npath
        cur_label = game.label(nwd)

    print(f"\n[chain] DONE: solved {len(solved)} consecutive levels: "
          f"{solved}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
