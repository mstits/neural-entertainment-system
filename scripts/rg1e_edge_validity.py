"""RG-1e edge-validity re-check (corrected methodology).

Two prior hand-rolled attempts at this measurement are not trusted (one had
a wrong room_fp mask config key, caught this session; the second reported a
suspicious 55% match rate using nt_fingerprint + hold-last semantics
modeled on the internals of `_replay_room_ord`, rather than by calling the
real `Solver` machinery end to end). This script instead:

  * constructs a REAL, short-lived `Solver` instance (--workers 1) against
    the exact profile (`configs/zelda_roomfp.yaml`) the RG-1 runs used, so
    the room_fp mask/settle/min_lines/pan_odo/etc all come from the real
    argparse + profile-parsing path (`solve.room_fp.mask`, not a
    hand-typed key name);
  * loads the FROZEN `RoomIndex` the live run actually banked
    (`room_index.json`) via `RoomIndex.load`, exactly like a resumed run
    would, instead of rebuilding one;
  * derives the post-replay room ordinal by calling the Solver's own
    `_replay_room_ord` (via `_xram_local`) on a FRESH single-worker pool,
    in the exact call sequence `replay_verify` (the one existing TESTED
    "restore + replay + observe outcome" precedent in this codebase, see
    tests/test_room_fp.py:528-606) already uses: fresh Pool -> hw flags ->
    headless/skip_preprocess -> reset_all -> load_worker_state -> one
    rooting NOOP -> per action: step_all then `_xram_local`. Zero
    invented mechanism beyond what a tested code path already does.

What this measures, precisely (and does NOT measure):

  This checks "does the room ordinal recorded as the destination of a
  banked edge reproduce, per hold-last identity semantics, when the
  archived exemplar_cell is restored and exemplar_actions replayed
  (optionally under a sticky perturbation)". `_replay_room_ord` is a
  single-frame frozen-table lookup with no settle counter and NO concept
  of transition `kind` (pan/fade/warp) at all -- it cannot, and does not
  try to, reproduce the full pre-registered RG-1e criterion of matching
  (src, dst, kind) via the live settle+classify+RoomIndex.record_edge
  path, because that path has never been exercised outside the hot loop
  against a foreign/restored pool (no test coverage exists for it). The
  number this script reports is a necessary-but-not-sufficient proxy for
  RG-1e's full gate: destination-ordinal reproduction. `kind` (from the
  ALREADY-recorded room_index.json edge, not from anything this replay
  computes) is used only as a POST-HOC grouping key over the mismatches,
  never as part of the pass/fail criterion here.

A second, independently damning methodology hazard was FOUND (not
assumed) empirically while validating this script and is measured and
reported explicitly, not silently absorbed into a "sticky robustness"
number: `RoomIndex.record_edge`'s `exemplar_cell` is an ARCHIVE CELL KEY,
not a frozen copy of the state blob active when the edge was recorded.
`GoExploreArchive` is domination-based and mutates `.state` under a key
whenever a LATER, unrelated visit reaches the same (sect, tb, kk, psig,
loops, route_sig, area, phase, y_band, vx_sign, gx_bucket) tuple with a
higher score (or same score, fewer steps) -- and `archive.pkl` is only
ever flushed as the CURRENT (for a still-running search) or FINAL (for a
completed one) snapshot, never a per-edge-record-time copy. Near-root
keys in a 90-minute, 12-worker run can rack up hundreds of visits, so by
the time `archive.pkl` was written, a non-trivial fraction of recorded
edges' `exemplar_cell` keys may already point at a *different* underlying
NES state than the one that actually produced that edge -- independent of
this script, independent of sticky, independent of settle-vs-hold-last.
This script measures it directly per edge (a zero-action "does the
restored state already read as `src`" probe) and reports the sticky/
zero-sticky match rates BOTH overall AND restricted to the subset where
the restore was clean, so a real drift-driven mismatch is never folded
into a claimed sticky-robustness number.

A THIRD confound was FOUND the same way (empirically, not assumed) even
among the "clean start" edges: `exemplar_actions` is a hard
`deque(maxlen=32)` ring, but a Zelda `fade` transition is a largely
input-independent scripted screen-wipe/reload whose total duration
(unstable churn + an 8-step measured plateau + `settle=14` confirmation
samples, per docs/receipts/room_fp/zelda.md) can approach or exceed that
32-step cap before the wipe's own time is even counted -- so the ring can
end BEFORE a transition already irreversibly in flight finishes. This
script's `probe_extra_idle_steps_to_dst` measures, for every clean-start
zero-sticky mismatch, whether pure idling for a few more steps (capped,
to avoid trading this confound for a second unrelated auto-drift into a
THIRD room, also observed directly) reaches `dst` anyway; those cases are
reported and counted separately from genuine non-reproductions, never
silently treated as either a pass or a sticky-robustness failure.

Net effect: THREE independent, code-grounded reasons an edge can fail to
"reproduce" here that have nothing to do with sticky-perturbation
robustness (wrong `kind` semantics is out of scope by design; archive
drift; ring-too-short-for-a-fade). Whatever match rate this script prints
should be read against that background, not as a bare verdict on the
room-graph engine.

Usage:
    .venv/bin/python scripts/rg1e_edge_validity.py \
        --run runs/room_graph/rg1_zelda_seed0_bias025 \
        --run runs/room_graph/rg1_zelda_seed1_bias025 \
        --profile configs/zelda_roomfp.yaml \
        --root-state roms/zelda_start_ctrl.state.bin \
        --n-edges 20 --seed 0 --sticky-p 0.25
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "go_explore_solve", str(REPO_ROOT / "scripts" / "go_explore_solve.py"))
ges = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ges)


class _StopInit(Exception):
    """Sentinel used to pull a real argparse.Namespace out of
    go_explore_solve.main() without letting it construct a Solver (which
    would build a pool) or run the search loop."""


def build_real_args(argv: list[str]) -> argparse.Namespace:
    """Get a REAL args Namespace with every flag's REAL default, by
    running the actual `main()` up to (not including) `Solver(args)`.

    This deliberately avoids hand-listing argparse defaults (the exact
    "hand-rolled reconstruction" failure mode this whole audit exists to
    distrust) -- `ap.add_argument(...)` lives inline inside `main()`, so
    the only zero-drift way to get its defaults is to let that exact code
    run. `Solver.__init__` is monkeypatched to capture `args` and abort
    immediately (raising before touching nes_core), so this costs nothing
    beyond argparse + yaml parsing."""
    captured: dict = {}
    orig_init = ges.Solver.__init__

    def _capture_init(self, args):
        captured["args"] = args
        raise _StopInit()

    ges.Solver.__init__ = _capture_init
    old_argv = sys.argv
    try:
        sys.argv = ["go_explore_solve.py"] + argv
        try:
            ges.main()
        except _StopInit:
            pass
    finally:
        sys.argv = old_argv
        ges.Solver.__init__ = orig_init
    return captured["args"]


def fresh_pool(solver) -> "ges.Pool":
    """A brand-new single-worker pool wired exactly like `replay_verify`'s
    (scripts/go_explore_solve.py:4232) -- "FRESH pool, never the search
    pool" -- so every edge replay starts from a clean nes_core instance,
    never from state a previous edge's replay left behind."""
    pool = ges.Pool(rom_path=solver.game.rom, num_workers=1,
                     frame_skip=solver.frame_skip)
    ges.apply_hw_flags(pool, solver.hw_flags)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    return pool


def replay_ordinal(solver, state_bytes: bytes, actions, *, sticky_p: float,
                    sticky_rng, tail_margin: int = 0) -> int:
    """Restore `state_bytes`, then replay `actions` and return the final
    room ordinal per `_replay_room_ord`'s hold-last semantics (called via
    the real `Solver._xram_local`, never reimplemented).

    Call sequence is byte-for-byte `replay_verify`'s tested pattern: fresh
    pool -> load_worker_state -> one rooting NOOP ("exactly as seed()
    does") -> per action, step_all then `_xram_local` with a `hold` dict
    that starts EMPTY (ROOM_UNKNOWN), exactly as `replay_verify`'s `ctx =
    {}` does -- no pre-seeding of the ordinal is invented here, because
    none exists in the tested precedent.

    `sticky_p`: with this per-step probability the ACTUAL action applied
    is the previous step's action instead of the next scripted one from
    `actions` (mirrors the live sampler's own `a = c["prev"] if
    rng.random() < sticky else <newly chosen action>`, substituting the
    scripted next action for "newly chosen") -- the number of pool steps
    taken always equals len(actions) regardless of how many times sticky
    fires, so a sticky-perturbed and a zero-sticky replay of the same
    edge run for the same number of frames. `sticky_p=0` is the
    zero-added-randomness control.

    `tail_margin`: extra NOOP steps appended after the scripted actions,
    off by default to stay byte-identical to `replay_verify`'s own
    pattern; pass >0 only if a validation run shows hold-last needs a
    frame or two to catch up (documented at the call site if so)."""
    pool = fresh_pool(solver)
    try:
        pool.load_worker_state(0, state_bytes)
        acts = np.zeros(1, dtype=np.uint8)
        pool.step_all(acts)  # rooting NOOP, exactly as seed() does
        hold: dict = {}
        prev_a = None
        for scripted in list(actions) + [0] * int(tail_margin):
            if (sticky_p > 0 and prev_a is not None
                    and sticky_rng.random() < sticky_p):
                a = prev_a
            else:
                a = int(scripted)
            prev_a = a
            acts[0] = solver.bitmasks[a]
            ram = pool.step_all(acts)[0][2]
            ram = solver._xram_local(ram, pool, hold)
        return int(hold.get("_room_ord", ges.ROOM_UNKNOWN))
    finally:
        pool.shutdown()


def probe_extra_idle_steps_to_dst(solver, state_bytes: bytes, actions,
                                  dst: int, cap: int = 48) -> int | None:
    """DIAGNOSTIC ONLY, for clean-start mismatches: after replaying the
    raw `actions` (zero sticky) and NOT reaching `dst`, keep idling
    (NOOP, no further scripted input) for up to `cap` more steps and
    report how many extra idle steps it took to reach `dst`, or None if
    it never does within the cap.

    This exists because `exemplar_actions` is a hard `deque(maxlen=32)`
    ring (scripts/go_explore_solve.py `_room_seed`), while a Zelda
    `fade` transition is a scripted, largely input-independent screen-
    wipe/reload whose total duration is unrelated to that cap and was
    independently measured (docs/receipts/room_fp/zelda.md) to need an
    8-step mid-pan plateau PLUS `settle=14` confirmation samples on top
    of unstable churn -- a span that "can plausibly approach or exceed
    32 pool-steps" for the ring alone, before the fade's own screen-wipe
    time is even counted. A small `extra_steps_to_dst` here means "the
    ring was simply a few steps short of a transition already fully in
    flight"; None (never, within a generous cap) means idling alone does
    not explain the mismatch. Capped (not unbounded) because idling
    indefinitely risks a SECOND, unrelated automatic room change chaining
    on top of the first (observed directly while validating this script:
    one edge's raw replay landed correctly on its recorded destination,
    then drifted AGAIN into a third, unrelated room after ~26 further
    idle steps) -- an unbounded margin would silently trade one confound
    for another, so this is reported as its own labeled diagnostic, never
    folded into the primary match-rate numbers."""
    pool = fresh_pool(solver)
    try:
        pool.load_worker_state(0, state_bytes)
        acts = np.zeros(1, dtype=np.uint8)
        pool.step_all(acts)  # rooting NOOP
        hold: dict = {}
        for scripted in actions:
            acts[0] = solver.bitmasks[int(scripted)]
            ram = pool.step_all(acts)[0][2]
            solver._xram_local(ram, pool, hold)
        acts[0] = 0
        for extra in range(1, cap + 1):
            ram = pool.step_all(acts)[0][2]
            solver._xram_local(ram, pool, hold)
            if int(hold.get("_room_ord", ges.ROOM_UNKNOWN)) == dst:
                return extra
        return None
    finally:
        pool.shutdown()


def probe_initial_ordinal(solver, state_bytes: bytes, warmup_steps: int) -> int:
    """DIAGNOSTIC ONLY (not part of the pass/fail criterion): what room
    ordinal does the restored `exemplar_cell` state already read as,
    before a single one of its recorded `exemplar_actions` is applied?
    `exemplar_cell` is supposed to be "the last cell observed BEFORE the
    transition began" (scripts/go_explore_solve.py `_room_step`'s onset
    comment) -- i.e. this should read back as the edge's recorded `src`.
    A live run's archive is domination-based and mutates a cell key's
    `.state` whenever a later, unrelated visit scores higher (see the
    docstring at the top of this file); when that happened to THIS key
    after this edge was recorded, this probe reads something other than
    `src` even though no replay logic has run yet -- proof the mismatch
    that follows is archive drift, not a property of the replay or of
    engine robustness. `warmup_steps` NOOPs (no buttons) are stepped
    first so a restored frame that starts blank/transitional has time to
    render before being judged, mirroring the live engine's own settle
    span rather than an arbitrarily chosen number."""
    pool = fresh_pool(solver)
    try:
        pool.load_worker_state(0, state_bytes)
        acts = np.zeros(1, dtype=np.uint8)
        pool.step_all(acts)  # rooting NOOP
        hold: dict = {}
        o = ges.ROOM_UNKNOWN
        for _ in range(warmup_steps):
            ram = pool.step_all(acts)[0][2]
            ram = solver._xram_local(ram, pool, hold)
            o = int(hold.get("_room_ord", ges.ROOM_UNKNOWN))
        return o
    finally:
        pool.shutdown()


def load_run(run_dir: Path):
    idx = ges.RoomIndex.load(run_dir / "room_index.json")
    with open(run_dir / "archive.pkl", "rb") as f:
        cells = pickle.load(f)
    edges = []
    for src, dsts in idx.adj.items():
        for dst, e in dsts.items():
            if e.get("exemplar_cell") is not None and e.get("exemplar_actions"):
                edges.append((int(src), int(dst), e))
    edges.sort(key=lambda t: (t[0], t[1]))
    return idx, cells, edges


def sample_edges(edges, n_edges: int, seed: int):
    rng = random.Random(seed)
    return rng.sample(edges, min(n_edges, len(edges)))


def resolve_exemplar_state(e: dict, cells: dict) -> "bytes | None":
    """The bytes to restore before replaying `e["exemplar_actions"]`.

    Prefers `exemplar_state`, the frozen bytes copy `RoomIndex.record_edge`
    captures at edge-commit time precisely so a reader never has to trust
    `archive.pkl`'s mutable, domination-overwritten `.state` at key
    `exemplar_cell` (see the module docstring's archive-drift section, and
    `RoomIndex.load`'s "go back to the OLD mutable-key behavior" comment).
    Falls back to the archive cell lookup only when `exemplar_state` is
    absent -- an edge recorded before the frozen-copy fix existed."""
    frozen = e.get("exemplar_state")
    if frozen is not None:
        return frozen
    cell = cells.get(e["exemplar_cell"])
    if cell is None or cell.state is None:
        return None
    return cell.state


def run_one(run_dir: Path, solver, n_edges: int, sample_seed: int,
            sticky_p: float, sticky_seed: int, warmup_steps: int,
            tail_margin: int) -> dict:
    idx, cells, edges = load_run(run_dir)
    if idx.config_sha != solver.game.room_fp_sha:
        raise SystemExit(
            f"[rg1e] {run_dir}: room_index.json config_sha {idx.config_sha!r} "
            f"!= this profile's {solver.game.room_fp_sha!r} -- refusing to "
            f"score a run against a mismatched room_fp config (same refusal "
            f"discipline as _resume_room_index).")
    # CRITICAL: Solver.__init__ built its OWN fresh, empty RoomIndex (armed
    # for a live search that interns as it goes). Every replay in this
    # script is a FROZEN lookup against the graph the live RG-1 run
    # actually banked, so the solver must be pointed at the index we just
    # loaded from this run's room_index.json -- an empty table means every
    # lookup silently returns None forever and every replay holds at
    # ROOM_UNKNOWN regardless of what the restored state actually shows
    # (caught empirically while validating this script: omitting this line
    # produced a 0% match rate on every sampled edge across both runs).
    solver.room_index = idx
    sample = sample_edges(edges, n_edges, sample_seed)
    sticky_rng = np.random.default_rng(sticky_seed)

    rows = []
    for src, dst, e in sample:
        state = resolve_exemplar_state(e, cells)
        if state is None:
            rows.append({"run": run_dir.name, "src": src, "dst": dst,
                         "kind": e["kind"], "dir": e.get("dir"),
                         "n_actions": len(e["exemplar_actions"]),
                         "error": "exemplar_cell missing from archive.pkl"})
            continue
        cell = cells.get(e["exemplar_cell"])
        actions = e["exemplar_actions"]

        init_ord = probe_initial_ordinal(solver, state, warmup_steps)
        if init_ord == src:
            start_bucket = "clean"
        elif init_ord == dst:
            start_bucket = "vacuous"  # already past the transition pre-replay
        else:
            start_bucket = "drifted"

        final_zero = replay_ordinal(solver, state, actions, sticky_p=0.0,
                                    sticky_rng=sticky_rng,
                                    tail_margin=tail_margin)
        final_sticky = replay_ordinal(solver, state, actions,
                                      sticky_p=sticky_p, sticky_rng=sticky_rng,
                                      tail_margin=tail_margin)
        extra_steps_to_dst = None
        if start_bucket == "clean" and final_zero != dst and tail_margin == 0:
            extra_steps_to_dst = probe_extra_idle_steps_to_dst(
                solver, state, actions, dst)
        rows.append({
            "run": run_dir.name, "src": src, "dst": dst, "kind": e["kind"],
            "dir": e.get("dir"), "n_actions": len(actions),
            "cell_visits": (int(cell.visits) if cell is not None else None),
            "init_ordinal": init_ord, "start_bucket": start_bucket,
            "final_zero": final_zero, "match_zero": final_zero == dst,
            "final_sticky": final_sticky, "match_sticky": final_sticky == dst,
            "extra_idle_steps_to_dst": extra_steps_to_dst,
        })
    return {"run": str(run_dir), "config_sha": idx.config_sha,
            "n_total_edges": len(edges), "rows": rows}


def summarize(all_rows: list[dict], sticky_p: float) -> str:
    def rate(rows, key):
        rows = [r for r in rows if "error" not in r]
        if not rows:
            return "n=0"
        n = len(rows)
        k = sum(1 for r in rows if r[key])
        return f"{k}/{n} = {100.0 * k / n:.1f}%"

    out = []
    out.append(f"n = {len(all_rows)} sampled edges "
               f"({len([r for r in all_rows if 'error' in r])} errored)")
    out.append("")
    out.append("=== OVERALL (all sampled edges, includes archive-drift confound) ===")
    out.append(f"zero-sticky (raw replay, no perturbation): {rate(all_rows, 'match_zero')}")
    out.append(f"sticky p={sticky_p} replay:                 {rate(all_rows, 'match_sticky')}")
    out.append("")
    for bucket in ("clean", "vacuous", "drifted"):
        sub = [r for r in all_rows if r.get("start_bucket") == bucket]
        out.append(f"--- start_bucket = {bucket} (n={len(sub)}) ---")
        if sub:
            out.append(f"  zero-sticky: {rate(sub, 'match_zero')}   "
                       f"sticky: {rate(sub, 'match_sticky')}")
    out.append("")
    out.append("=== CLEAN-START-ONLY (the scientifically meaningful subset: "
               "exemplar_cell restored to the recorded src BEFORE any "
               "exemplar_actions were applied) ===")
    clean = [r for r in all_rows if r.get("start_bucket") == "clean"]
    out.append(f"n = {len(clean)}")
    out.append(f"zero-sticky match rate: {rate(clean, 'match_zero')}")
    out.append(f"sticky p={sticky_p} match rate:  {rate(clean, 'match_sticky')}")
    out.append("")
    out.append("=== breakdown by recorded kind (clean-start only) ===")
    for kind in ("pan", "fade"):
        sub = [r for r in clean if r.get("kind") == kind]
        if sub:
            out.append(f"  kind={kind} (n={len(sub)}): zero={rate(sub,'match_zero')} "
                       f"sticky={rate(sub,'match_sticky')}")
    out.append("")
    out.append("=== breakdown by exemplar_actions length (clean-start only) ===")
    short = [r for r in clean if r.get("n_actions", 0) < 24]
    long_ = [r for r in clean if r.get("n_actions", 0) >= 24]
    out.append(f"  short (<24 actions, n={len(short)}): zero={rate(short,'match_zero')} "
               f"sticky={rate(short,'match_sticky')}")
    out.append(f"  long (>=24 actions, n={len(long_)}): zero={rate(long_,'match_zero')} "
               f"sticky={rate(long_,'match_sticky')}")
    out.append("")
    zero_mismatches = [r for r in clean if not r.get("match_zero")]
    ring_explained = [r for r in zero_mismatches
                      if r.get("extra_idle_steps_to_dst") is not None]
    out.append(f"=== clean-start zero-sticky mismatches: {len(zero_mismatches)} total, "
               f"{len(ring_explained)} resolve to dst within 48 extra idle "
               f"steps (ring-length/fade-duration confound, not a real "
               f"failure), {len(zero_mismatches) - len(ring_explained)} do "
               f"NOT resolve even given that idle margin (genuine "
               f"non-reproduction) ===")
    out.append("")
    out.append("=== every mismatch, clean-start only (per-edge detail) ===")
    out.append("(extra_idle_steps_to_dst: after the raw zero-sticky replay's "
               "32-action ring runs out without reaching dst, how many more "
               "NOOP steps of pure idling until it gets there anyway -- 'N' "
               "= ring-length/fade-duration confound, not a real failure; "
               "'never (<=48)' = does not resolve even given generous extra "
               "idle time, i.e. a genuine non-reproduction; blank = matched "
               "already or tail_margin>0 made this probe moot)")
    for r in clean:
        if not r.get("match_sticky") or not r.get("match_zero"):
            extra = r.get("extra_idle_steps_to_dst")
            extra_s = ("never (<=48)" if (not r["match_zero"] and extra is None)
                      else (f"+{extra}" if extra is not None else "-"))
            out.append(
                f"  run={r['run']} {r['src']}->{r['dst']} kind={r['kind']} "
                f"dir={r.get('dir')} n_actions={r['n_actions']} "
                f"visits={r.get('cell_visits')} "
                f"final_zero={r['final_zero']}(match={r['match_zero']}) "
                f"final_sticky={r['final_sticky']}(match={r['match_sticky']}) "
                f"extra_idle_steps_to_dst={extra_s}")
    out.append("")
    out.append("=== every drifted/vacuous-start edge, matched or not "
               "(methodology confound -- archive-state drift -- NOT scored "
               "against the 80% bar either way) ===")
    for r in all_rows:
        if r.get("start_bucket") in ("drifted", "vacuous"):
            out.append(
                f"  run={r['run']} {r['src']}->{r['dst']} kind={r['kind']} "
                f"start_bucket={r['start_bucket']} init_ordinal={r['init_ordinal']} "
                f"n_actions={r['n_actions']} visits={r.get('cell_visits')} "
                f"final_zero={r['final_zero']} final_sticky={r['final_sticky']}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True,
                    help="Run directory (repeatable); expects room_index.json "
                         "+ archive.pkl inside it.")
    ap.add_argument("--profile", default="configs/zelda_roomfp.yaml")
    ap.add_argument("--root-state", default="roms/zelda_start_ctrl.state.bin")
    ap.add_argument("--n-edges", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds BOTH the edge sample and the sticky RNG, "
                         "independently per run directory.")
    ap.add_argument("--sticky-p", type=float, default=0.25)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="NOOP steps for the initial-ordinal drift probe; "
                         "defaults to the profile's own room_fp.settle + 2.")
    ap.add_argument("--tail-margin", type=int, default=0)
    ap.add_argument("--out-json", default=None,
                    help="Optional path to dump full per-edge rows as JSON.")
    args = ap.parse_args()

    tmp_out = tempfile.mkdtemp(prefix="rg1e_edge_validity_")
    real_args = build_real_args([
        "--out", tmp_out, "--root-state", args.root_state,
        "--profile", args.profile, "--workers", "1", "--seed", "0",
    ])
    solver = ges.Solver(real_args)
    if solver.room_fp is None:
        raise SystemExit(f"[rg1e] {args.profile} has no solve.room_fp block "
                         f"-- wrong profile for this check.")
    warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                    else int(solver.room_fp["settle"]) + 2)
    print(f"[rg1e] Solver built. profile room_fp_sha={solver.game.room_fp_sha} "
         f"settle={solver.room_fp['settle']} mask={solver.room_fp['mask']} "
         f"warmup_steps={warmup_steps}", flush=True)

    all_rows = []
    per_run_results = []
    try:
        for run_str in args.run:
            run_dir = Path(run_str)
            print(f"[rg1e] scoring {run_dir} ...", flush=True)
            result = run_one(run_dir, solver, args.n_edges, args.seed,
                             args.sticky_p, args.seed, warmup_steps,
                             args.tail_margin)
            per_run_results.append(result)
            all_rows.extend(result["rows"])
    finally:
        solver.pool.shutdown()

    report = summarize(all_rows, args.sticky_p)
    print()
    print(report)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            {"per_run": per_run_results, "sticky_p": args.sticky_p,
             "seed": args.seed, "warmup_steps": warmup_steps}, indent=2,
            default=str) + "\n")
        print(f"\n[rg1e] full rows -> {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
