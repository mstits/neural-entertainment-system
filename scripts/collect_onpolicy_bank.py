#!/usr/bin/env python3
"""On-policy transition-bank collector for the V_adv B5 re-score.

Registration: `docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md` §4. Reuses
the driving pattern `scripts/gen_iq_transitions.py` and `scripts/eval_game.py`
already proved: `nes_core.Pool`, `load_worker_state`, `step_all`,
`TileFeatureStacker`, `resolve_encoder`, sticky gated on `step > 0`.

WHY THIS EXISTS. The first V_adv computation (`docs/research/VADV_B5_2026-08-27.md`)
scored a critic against go-explore expert-window states the policy itself
never visited — off-distribution for the critic being scored. This collector
rolls each banked checkpoint out under ITS OWN policy (sampled from its own
softmax, its own training-time sticky noise) from the tape's own restart
windows, so the wall states a downstream reader scores are on-distribution
for that checkpoint's critic.

THREE START-STATE POPULATIONS, drawn from the run's own tape
(`checkpoints/backward_states/1-2`) and its own entrance state — a HARD
PARTITION, never merged:

  WALL_SRC  tau 893, uniform over tape entries [853, 893]  -> WALL / INTERIOR
  PC_SRC    tau in {1093, 1053, 1013}, uniform rung then window -> PC_B5
  ENTR_SRC  the true entrance state                         -> EARLY (secondary)

CROSS-POPULATION IS DROPPED, NOT MERGED: a PC_SRC row landing in the WALL
band, or a WALL_SRC row landing in the PC_B5 band, is discarded before the
bank is written — letting a clearing trajectory's rows inflate WALL's own
action-discrimination would be an artifact unrelated to the critic at rung
893, and it would push R toward CAPABILITY (see the registration §4).

LEVEL-IDENTITY PURITY GUARD (hard, revert-verified): a row is admitted only
if (world, level) == (0, 1) at its start state s, and — for every row that
does NOT end the episode — at its successor s' too. A terminal row's s' is
explicitly exempt because a level clear or a death IS a change in that
identity by definition; the "no row is ever recorded after the transition"
rule (below) is what actually prevents the aliasing bug this guard exists to
catch (the unguarded pilot recorded 1-3 states leaking into 1-2's own gx
bands after a clear, because nothing stopped recording at the clear frame).
Any violation raises RuntimeError, which VOIDs the whole job per the
registration's abort table.

Honest: no game internals beyond the config's own declared `ram_mapping`
(world/level/player_state); noise is the training config's own sticky model;
actions are the checkpoint's own policy, sampled from its own softmax.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.score_banked_iterates import (  # noqa: E402
    build_tile_net,
    decode_gx,
    iterate_number,
    load_iterate,
    sort_iterates,
)
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks,
    resolve_encoder,
)

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")

# The config's own declared ram_mapping (configs/mario_1_2_backward.yaml) —
# no new address is introduced anywhere in this file.
R_PSTATE = 0x000E
DEATH = (6, 11)
WORLD_ADDR = 0x075F
LEVEL_ADDR = 0x075C
TARGET_WORLD_LEVEL = (0, 1)  # SMB 1-2 in this project's 0-indexed encoding

WALL_BAND = (2674, 2872)
PC_B5_BAND = (2872, 3267)


# ===========================================================================
# PURE FUNCTIONS — no torch, no emulator, no I/O. Unit-tested directly.
# ===========================================================================

def window_for_rung(entries: Sequence[dict], rung: int, *, span: int = 40) -> list:
    """The `span+1`-entry restart window ending at tape entry `rung`.

    `window_frames 160 / frames_per_step 4` (the config's own numbers) is a
    40-tape-entry window; the registered draw is uniform over
    `[rung - span, rung]` inclusive.
    """
    lo = int(rung) - int(span)
    return [e for e in entries if lo <= int(e["step"]) <= int(rung)]


def episode_step_cap(entry_step: int, *, n_entries: int = 1094,
                     base: int = 600, per_entry: float = 2.0,
                     global_cap: int = 1536) -> int:
    """The run's own per-rung budget: `min(global_cap, base + per_entry*(n_entries - r))`.

    `configs/mario_1_2_backward.yaml` `backward_curriculum.rung_step_budget`.
    """
    return int(min(global_cap, base + per_entry * (int(n_entries) - int(entry_step))))


def sticky_should_apply(step: int, roll: float, sticky_prob: float) -> bool:
    """Sticky roll gated on `step > 0` — `sticky_episode_boundary_reset: true`,
    the config's own setting and the honest harness's own rule
    (`scripts/eval_game.py`). Never sticks the first action of an episode."""
    return bool(step > 0 and float(roll) < float(sticky_prob))


def level_identity(ram_bytes: Any) -> tuple:
    """`(world, level)` read off the config's own declared `ram_mapping`."""
    arr = np.frombuffer(bytes(ram_bytes), dtype=np.uint8)
    return int(arr[WORLD_ADDR]), int(arr[LEVEL_ADDR])


def is_dead(ram_bytes: Any, pool_done: bool = False) -> bool:
    """Death per `gen_iq_transitions.py`'s own convention: `RAM[$000E]` in
    `{6, 11}`, or the pool's own done flag."""
    arr = np.frombuffer(bytes(ram_bytes), dtype=np.uint8)
    return bool(int(arr[R_PSTATE]) in DEATH) or bool(pool_done)


def check_purity(world_s: int, level_s: int, world_ns: int, level_ns: int,
                 terminal: bool, target: tuple = TARGET_WORLD_LEVEL) -> None:
    """The hard, revert-verified guard. Raises RuntimeError on violation.

    `s` must always carry the target identity. `s'` must too, UNLESS this row
    is the one that ends the episode (a level clear or a death legitimately
    changes — or, for death, may transiently read — a different identity);
    the "no row after the transition" rule is what stops a silent multi-row
    leak, so this check only needs to catch a MISSED transition, not police
    the terminal row's own successor.
    """
    if (int(world_s), int(level_s)) != tuple(target):
        raise RuntimeError(
            f"purity violation at s: (world,level)=({world_s},{level_s}) "
            f"!= {target}"
        )
    if not terminal and (int(world_ns), int(level_ns)) != tuple(target):
        raise RuntimeError(
            f"purity violation at s' of a NON-terminal row: "
            f"(world,level)=({world_ns},{level_ns}) != {target}"
        )


def append_transition_row(rows: list, s: np.ndarray, a: int, ns: np.ndarray,
                          terminal: bool) -> None:
    """Append one `[s, a, ns, done, truncated]` row unconditionally.

    `truncated` is always written 0 here; `finalize_truncation` below flips
    the LAST row's truncated bit to 1 if the episode never terminated (the
    `scripts/smodice_data.py` convention: `done` is the absorbing/terminal
    bit, `truncated` marks partial-episode bootstrapping).
    """
    rows.append([np.asarray(s, dtype=np.int8).copy(), int(a),
                 np.asarray(ns, dtype=np.int8).copy(), int(bool(terminal)), 0])


def finalize_truncation(rows: list, terminal_hit: bool) -> list:
    """Mark a cap-truncated episode's last row `truncated=1` (`done` stays
    whatever it already is — 0, since a terminal row already stopped
    recording)."""
    if not terminal_hit and rows:
        rows[-1][4] = 1
    return rows


def assert_bank_wellformed(state: np.ndarray, next_state: np.ndarray,
                           episode_id: np.ndarray,
                           row_step: Optional[np.ndarray] = None) -> None:
    """Raise unless the written arrays are actually a transition bank.

    Two THRESHOLD-FREE invariants, checked on the artifact that goes to disk
    rather than on the values the loop held in hand. Both were absent on
    2026-08-27, when a reused stacker buffer wrote `(s', a, s')` on 100 % of
    rows across 26 banks and every one of gates A1-A8 passed anyway:

    1. **Chain.** Within one episode the successor recorded at step `i` IS the
       antecedent recorded at step `i+1`. Exact, no tolerance, and true of any
       correct bank whether the scene moves or is frozen. Under the aliasing
       defect it is false at every step, because row `i` held `s'_i` in both
       slots while row `i+1` holds `s'_{i+1}`.
    2. **Non-degeneracy.** Not EVERY row may satisfy `s == s'`. Individual
       frozen rows are legal (five identical frames inside a 4-frame stack),
       so no fraction below 1.0 is asserted -- picking one would be a
       threshold nobody checked against its acting range. 1.0 is the
       defect's own signature and is impossible for a bank containing motion.

    `row_step` (per-row within-episode step index, written into the bank)
    makes the registered cross-population DROP visible to invariant 1: a
    dropped interior row leaves surviving neighbours whose steps are not
    consecutive, and the chain is asserted ONLY across pairs whose steps
    are. Without it the guard's first live run (2026-08-28) false-positived
    on iter 30: a PC_SRC rung-1013 episode traversed the WALL band on its
    way up, the registered drop removed those rows, and the legitimate gap
    read as CHAIN BROKEN. The guard loses nothing against the defect class
    it exists for -- aliasing corrupts EVERY adjacent pair, and drops are
    a handful of rows in ~60k. `row_step` must be strictly increasing
    within an episode; `None` keeps the gap-free contract for banks that
    carry no step column.
    """
    if state.shape != next_state.shape:
        raise RuntimeError(
            f"[bank] malformed: state {state.shape} != next_state "
            f"{next_state.shape}")
    if state.shape[0] == 0:
        return
    identical = np.asarray((state == next_state).all(axis=1))
    if bool(identical.all()):
        raise RuntimeError(
            "[bank] DEGENERATE: state == next_state on 100% of "
            f"{state.shape[0]} rows. The advantage r + gamma*V(s') - V(s) "
            "collapses to (gamma-1)*V(s') -- a function of ONE state carrying "
            "zero action information by construction. This bank cannot be "
            "scored; it is VOID, not a reading.")
    steps = None if row_step is None else np.asarray(row_step, dtype=np.int64)
    if steps is not None and steps.shape[0] != state.shape[0]:
        raise RuntimeError(
            f"[bank] malformed: row_step {steps.shape} != state rows "
            f"{state.shape[0]}")
    for ep in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == ep)
        if idx.size < 2:
            continue
        if steps is not None:
            ep_steps = steps[idx]
            if not bool((np.diff(ep_steps) > 0).all()):
                raise RuntimeError(
                    f"[bank] ROW ORDER BROKEN in episode {int(ep)}: row_step "
                    "is not strictly increasing; the rows are not an ordered "
                    "trajectory and the bank is VOID.")
            adjacent = np.diff(ep_steps) == 1
        else:
            adjacent = np.ones(idx.size - 1, dtype=bool)
        pred = idx[:-1][adjacent]
        succ = idx[1:][adjacent]
        if pred.size and not np.array_equal(next_state[pred], state[succ]):
            raise RuntimeError(
                f"[bank] CHAIN BROKEN in episode {int(ep)}: the successor "
                "recorded at one step is not the antecedent recorded at the "
                "next. The rows are not consecutive transitions of one "
                "trajectory; the bank is VOID.")


def cross_population_mask(gx: np.ndarray, src: np.ndarray, *,
                          wall_band: tuple = WALL_BAND,
                          pc_band: tuple = PC_B5_BAND) -> np.ndarray:
    """Boolean KEEP mask implementing the registered hard partition.

    A `WALL_SRC` row landing in `PC_B5` is dropped; a `PC_SRC` row landing in
    `WALL` is dropped. Every other row (including `ENTR_SRC` rows, and rows
    from either source landing outside both bands) is kept — band membership
    for SCORING is decided later, by the scorer's own band masks; this
    function only removes the two named cross-population artifacts.
    """
    gx = np.asarray(gx)
    src = np.asarray(src)
    in_wall = (gx >= wall_band[0]) & (gx < wall_band[1])
    in_pc = (gx >= pc_band[0]) & (gx < pc_band[1])
    drop = ((src == "WALL_SRC") & in_pc) | ((src == "PC_SRC") & in_wall)
    return ~drop


def penetration_receipt(gx_by_episode: Sequence[np.ndarray], *,
                        threshold: int = 2676) -> dict:
    """§5.2 penetration receipts for one iterate's WALL_SRC episodes.

    `gx_by_episode` is one gx array per WALL_SRC episode (its own
    `next_state` gx trace). `pen_rate` counts an episode as having
    penetrated if ANY of its rows exceeds `threshold`.
    """
    n = len(gx_by_episode)
    if n == 0:
        return {"pen_rate": None, "gx_max": None, "n_episodes": 0}
    penetrated = sum(1 for g in gx_by_episode
                     if g.size and int(np.max(g)) > int(threshold))
    gx_max = max((int(np.max(g)) for g in gx_by_episode if g.size), default=None)
    return {"pen_rate": penetrated / float(n), "gx_max": gx_max, "n_episodes": n}


# ===========================================================================
# THIN SHELLS — torch / emulator.
# ===========================================================================

def select_action_sampled(logits: Any, temperature: float, generator: Any) -> int:
    """Draw once from `softmax(logits / temperature)`. Registered mode: SAMPLED,
    never greedy (§4 of the registration — greedy could never qualify a cell
    under the registered `>= 2 distinct actions` cell rule)."""
    import torch

    probs = torch.softmax(logits[0].float() / float(temperature), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


class _Lane:
    __slots__ = ("worker", "src", "src_rung", "episode_id", "cap", "step",
                "obs", "prev_ram", "prev_action", "done", "terminal_hit",
                "rows", "gx_trace", "cleared", "died")

    def __init__(self, worker: int, src: str, src_rung: int, episode_id: int,
                cap: int) -> None:
        self.worker = worker
        self.src = src
        self.src_rung = src_rung
        self.episode_id = episode_id
        self.cap = cap
        self.step = 0
        self.obs = None
        self.prev_ram = None
        self.prev_action = 0
        self.done = False
        self.terminal_hit = False
        self.rows: list = []
        self.gx_trace: list = []
        self.cleared = False
        self.died = False


def run_lanes(pool, net, bitmasks, extractor, stacker_factory, specs: list,
             *, workers: int, sticky_prob: float, temperature: float,
             rng: "np.random.Generator", torch_gen: Any) -> list:
    """Drive `specs` (one dict per episode) in waves of `workers` pool lanes.

    Wave scheduling matches `scripts/eval_game.py`'s `_run_episodes_parallel`:
    one `reset_all()` per wave, each lane's start state loaded fresh, nothing
    carried across waves. Returns the list of finished `_Lane` objects, in
    the same order as `specs`.
    """
    import torch

    finished: list[Optional[_Lane]] = [None] * len(specs)
    n_workers = int(pool.num_workers)

    for wave_start in range(0, len(specs), workers):
        batch = specs[wave_start:wave_start + workers]
        pool.reset_all()
        lanes: list[_Lane] = []
        for worker, spec in enumerate(batch):
            pool.load_worker_state(worker, spec["state"])
            lanes.append(_Lane(worker, spec["src"], spec["src_rung"],
                               spec["episode_id"], spec["cap"]))
        for worker in range(len(batch), n_workers):
            pool.set_worker_done(worker, True)
        # One mandatory no-op to flush the post-load frame (the trainer's own
        # warm-start convention; no start-jitter — see module docstring).
        r0 = pool.step_all(np.zeros(n_workers, dtype=np.uint8))
        stackers = [stacker_factory() for _ in lanes]
        for lane, stk in zip(lanes, stackers):
            ram = r0[lane.worker][2]
            lane.prev_ram = ram
            # `.copy()` is LOAD-BEARING, not defensive: TileFeatureStacker
            # returns its own reused `_out` buffer, so without the copy
            # `lane.obs` aliases the buffer the next push() mutates in
            # place and every recorded (s, a, s') collapses to (s', a, s').
            lane.obs = stk.reset(extractor.extract(ram)).copy()

        active = list(lanes)
        while any(not lane.done for lane in active):
            actions = np.zeros(n_workers, dtype=np.uint8)
            for lane in active:
                if lane.done:
                    continue
                with torch.no_grad():
                    x = torch.from_numpy(lane.obs[None, :]).float()
                    logits, _ = net.forward_ac(x)
                action_idx = select_action_sampled(logits, temperature, torch_gen)
                if sticky_should_apply(lane.step, float(rng.random()), sticky_prob):
                    action_idx = lane.prev_action
                lane.prev_action = action_idx
                actions[lane.worker] = bitmasks[action_idx]
            r = pool.step_all(actions)
            for lane, stk in zip(lanes, stackers):
                if lane.done:
                    continue
                out = r[lane.worker]
                ram_new = out[2]
                pool_done = bool(out[3])
                dead = is_dead(ram_new, pool_done)
                w_ns, l_ns = level_identity(ram_new)
                cleared = (w_ns, l_ns) != TARGET_WORLD_LEVEL
                terminal = dead or cleared
                w_s, l_s = level_identity(lane.prev_ram)
                check_purity(w_s, l_s, w_ns, l_ns, terminal)
                # `.copy()` LOAD-BEARING — see the reset above. `lane.obs`
                # must remain the antecedent while `s_new` is the successor.
                s_new = stk.push(extractor.extract(ram_new)).copy()
                # `lane.prev_action` still holds THIS timestep's executed
                # action — it is only overwritten in the next timestep's
                # action-selection pass above, which has not run yet.
                append_transition_row(
                    lane.rows, lane.obs, lane.prev_action, s_new, terminal,
                )
                lane.gx_trace.append(int(decode_gx(s_new[None, :])[0]))
                lane.obs = s_new
                lane.prev_ram = ram_new
                lane.step += 1
                if terminal:
                    lane.terminal_hit = True
                    lane.cleared = cleared
                    lane.died = dead
                    lane.done = True
                    pool.set_worker_done(lane.worker, True)
                elif lane.step >= lane.cap:
                    lane.done = True
                    pool.set_worker_done(lane.worker, True)
        for i, lane in enumerate(lanes):
            finalize_truncation(lane.rows, lane.terminal_hit)
            finished[wave_start + i] = lane

    missing = [i for i, f in enumerate(finished) if f is None]
    if missing:  # pragma: no cover - defensive; the wave loop covers every index
        raise RuntimeError(f"collector lost episodes {missing}")
    return [f for f in finished if f is not None]


def load_tape_entries(states_dir: Path) -> list:
    idx = json.loads((states_dir / "index.json").read_text())
    return idx["entries"]


def load_state_bytes(states_dir: Path, entry: dict) -> bytes:
    return (states_dir / entry["file"]).read_bytes()


def build_specs(entries: list, states_dir: Path, entrance_bytes: bytes, *,
                wall_rung: int, pc_rungs: Sequence[int], probe_rung: Optional[int],
                wall_episodes: int, pc_episodes: int, entr_episodes: int,
                probe_episodes: int, rng: "np.random.Generator") -> tuple:
    """Build the episode spec list for one iterate (WALL/PC/ENTR) plus,
    separately, the diagnostic probe specs (never merged into the main list).
    """
    wall_window = window_for_rung(entries, wall_rung)
    pc_windows = {r: window_for_rung(entries, r) for r in pc_rungs}
    specs: list = []
    eid = 0
    for _ in range(wall_episodes):
        e = wall_window[int(rng.integers(len(wall_window)))]
        specs.append({"src": "WALL_SRC", "src_rung": wall_rung,
                     "state": load_state_bytes(states_dir, e),
                     "cap": episode_step_cap(e["step"]), "episode_id": eid})
        eid += 1
    for _ in range(pc_episodes):
        r = int(pc_rungs[int(rng.integers(len(pc_rungs)))])
        w = pc_windows[r]
        e = w[int(rng.integers(len(w)))]
        specs.append({"src": "PC_SRC", "src_rung": r,
                     "state": load_state_bytes(states_dir, e),
                     "cap": episode_step_cap(e["step"]), "episode_id": eid})
        eid += 1
    for _ in range(entr_episodes):
        specs.append({"src": "ENTR_SRC", "src_rung": -1,
                     "state": entrance_bytes, "cap": 1536, "episode_id": eid})
        eid += 1
    probe_specs: list = []
    if probe_rung is not None:
        probe_window = window_for_rung(entries, probe_rung)
        for pid in range(probe_episodes):
            e = probe_window[int(rng.integers(len(probe_window)))]
            probe_specs.append({"src": "PROBE_SRC", "src_rung": probe_rung,
                                "state": load_state_bytes(states_dir, e),
                                "cap": episode_step_cap(e["step"]),
                                "episode_id": pid})
    return specs, probe_specs


def lanes_to_bank(lanes: list) -> dict:
    """Assemble one iterate's npz-ready arrays from finished lanes, applying
    the cross-population drop before anything is written to disk."""
    S, A, NS, D, T, SRC, EID, STEP = [], [], [], [], [], [], [], []
    for lane in lanes:
        for step_i, (s, a, ns, d, t) in enumerate(lane.rows):
            S.append(s); A.append(a); NS.append(ns); D.append(d); T.append(t)
            SRC.append(lane.src_rung); EID.append(lane.episode_id)
            # Within-episode step index, RECORDED BEFORE the drop below, so
            # a dropped row leaves a visible gap in the written artifact and
            # `assert_bank_wellformed`'s chain invariant knows exactly which
            # surviving pairs were adjacent when recorded.
            STEP.append(step_i)
    if not S:
        return {"state": np.zeros((0, 0), dtype=np.int8), "action": np.zeros(0, dtype=np.int64),
               "next_state": np.zeros((0, 0), dtype=np.int8), "done": np.zeros(0, dtype=np.int64),
               "truncated": np.zeros(0, dtype=np.int64), "src_rung": np.zeros(0, dtype=np.int64),
               "episode_id": np.zeros(0, dtype=np.int64),
               "row_step": np.zeros(0, dtype=np.int64)}
    state = np.stack(S).astype(np.int8)
    next_state = np.stack(NS).astype(np.int8)
    gx = decode_gx(state)
    # Per-row source NAME (not rung — rungs are not unique to one source),
    # taken from the owning lane so cross-population filtering is exact.
    src_names = np.array(
        [lane.src for lane in lanes for _ in lane.rows], dtype=object)
    keep = cross_population_mask(gx, src_names)
    return {
        "state": state[keep], "action": np.asarray(A, dtype=np.int64)[keep],
        "next_state": next_state[keep], "done": np.asarray(D, dtype=np.int64)[keep],
        "truncated": np.asarray(T, dtype=np.int64)[keep],
        "src_rung": np.asarray(SRC, dtype=np.int64)[keep],
        "episode_id": np.asarray(EID, dtype=np.int64)[keep],
        "row_step": np.asarray(STEP, dtype=np.int64)[keep],
        "src": src_names[keep],
        "n_dropped_cross_population": int((~keep).sum()),
    }


def episode_sidecar(lanes: list) -> list:
    out = []
    for lane in lanes:
        out.append({
            "episode_id": lane.episode_id, "src": lane.src,
            "src_rung": lane.src_rung, "n_steps": lane.step,
            "cleared": bool(lane.cleared), "died": bool(lane.died),
            "truncated": bool(not lane.terminal_hit),
            "max_gx": (max(lane.gx_trace) if lane.gx_trace else None),
        })
    return out


def collect_one_iterate(ckpt_path: Path, *, pool, entries: list, states_dir: Path,
                        entrance_bytes: bytes, bitmasks, extractor,
                        stack_size: int, feature_dim: int, args, out_dir: Path) -> dict:
    import torch

    iter_num = iterate_number(ckpt_path)
    seed = int(args.seed) + int(iter_num or 0)
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed)

    sd = load_iterate(ckpt_path)
    width = int(np.asarray(sd["fc1.weight"]).shape[1])
    bank_width = feature_dim * stack_size
    if width != bank_width:
        return {"iter": iter_num, "path": str(ckpt_path), "void": True,
               "void_reason": f"A8: checkpoint input width {width} != bank width {bank_width}"}
    net = build_tile_net(sd)
    net.eval()

    specs, probe_specs = build_specs(
        entries, states_dir, entrance_bytes,
        wall_rung=args.wall_rung, pc_rungs=args.pc_rungs,
        probe_rung=(args.probe_rung if iter_num in args.probe_iters else None),
        wall_episodes=args.wall_episodes, pc_episodes=args.pc_episodes,
        entr_episodes=args.entr_episodes, probe_episodes=args.probe_episodes,
        rng=rng,
    )

    def stacker_factory():
        from src.emulation.frame_utils import TileFeatureStacker
        return TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)

    lanes = run_lanes(pool, net, bitmasks, extractor, stacker_factory, specs,
                      workers=args.workers, sticky_prob=args.sticky_prob,
                      temperature=1.0, rng=rng, torch_gen=torch_gen)
    bank = lanes_to_bank(lanes)
    sidecar = episode_sidecar(lanes)

    wall_gx_by_ep = [np.array(lane.gx_trace) for lane in lanes if lane.src == "WALL_SRC"]
    pen = penetration_receipt(wall_gx_by_ep)

    probe_result = None
    if probe_specs:
        probe_lanes = run_lanes(pool, net, bitmasks, extractor, stacker_factory,
                                probe_specs, workers=args.workers,
                                sticky_prob=args.sticky_prob, temperature=1.0,
                                rng=rng, torch_gen=torch_gen)
        probe_result = {
            "iter": iter_num, "src_rung": args.probe_rung,
            "n_episodes": len(probe_lanes),
            "clear_rate": float(np.mean([l.cleared for l in probe_lanes])),
            "max_gx": max((max(l.gx_trace) for l in probe_lanes if l.gx_trace),
                          default=None),
            "deposits_in_wall": int(sum(
                1 for l in probe_lanes
                if any(WALL_BAND[0] <= g < WALL_BAND[1] for g in l.gx_trace))),
        }

    # Guard the ARTIFACT, not the loop's local variables -- the 2026-08-27
    # purity guard ran on the true antecedents and passed while the rows
    # written to disk held the successor twice.
    assert_bank_wellformed(bank["state"], bank["next_state"],
                           bank["episode_id"], bank["row_step"])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"iter_{iter_num:05d}.npz"
    np.savez(
        npz_path, state=bank["state"], action=bank["action"],
        next_state=bank["next_state"], done=bank["done"],
        truncated=bank["truncated"], src_rung=bank["src_rung"],
        episode_id=bank["episode_id"], row_step=bank["row_step"],
    )
    sidecar_path = out_dir / f"iter_{iter_num:05d}_episodes.json"
    sidecar_path.write_text(json.dumps({
        "iter": iter_num, "checkpoint": str(ckpt_path), "seed": seed,
        "episodes": sidecar, "penetration": pen,
        "n_dropped_cross_population": bank["n_dropped_cross_population"],
        "n_rows": int(bank["state"].shape[0]),
    }, indent=2, sort_keys=True))

    return {
        "iter": iter_num, "path": str(ckpt_path), "void": False,
        "n_rows": int(bank["state"].shape[0]), "n_episodes": len(lanes),
        "penetration": pen, "npz": str(npz_path), "sidecar": str(sidecar_path),
        "probe": probe_result,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iterates", action="append", default=[], metavar="GLOB")
    ap.add_argument("--profile", default="configs/mario_1_2_backward.yaml")
    ap.add_argument("--rom", default=ROM)
    ap.add_argument("--tape-dir", default="checkpoints/backward_states/1-2")
    ap.add_argument("--entrance-state",
                    default="runs/live_show/smb_4_4_micro/entrance_after_1-1.state")
    ap.add_argument("--out-dir", default="runs/vadv_onpolicy")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wall-rung", type=int, default=893)
    ap.add_argument("--pc-rungs", type=int, nargs="+", default=[1093, 1053, 1013])
    ap.add_argument("--probe-rung", type=int, default=933)
    ap.add_argument("--probe-iters", type=int, nargs="+", default=[10, 70, 130, 190, 250])
    ap.add_argument("--wall-episodes", type=int, default=40)
    ap.add_argument("--pc-episodes", type=int, default=24)
    ap.add_argument("--entr-episodes", type=int, default=60)
    ap.add_argument("--probe-episodes", type=int, default=24)
    ap.add_argument("--sticky-prob", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--max-collect-hours", type=float, default=3.0)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args(argv)

    import yaml
    from nes_core import Pool

    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    states_dir = REPO / args.tape_dir
    entries = load_tape_entries(states_dir)
    entrance_bytes = (REPO / args.entrance_state).read_bytes()

    paths = []
    import glob
    for pat in args.iterates:
        paths.extend(Path(p) for p in glob.glob(pat))
    paths = sort_iterates(paths)
    if not paths:
        print("no iterates matched", file=sys.stderr)
        return 2

    pool = Pool(rom_path=args.rom, num_workers=int(args.workers), frame_skip=fs)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)

    out_dir = Path(args.out_dir)
    results = []
    probes = []
    t0 = time.monotonic()
    aborted = False
    for p in paths:
        elapsed_h = (time.monotonic() - t0) / 3600.0
        if elapsed_h > float(args.max_collect_hours):
            print(f"[collect] {elapsed_h:.2f}h > max-collect-hours "
                 f"{args.max_collect_hours}; stopping (registered abort)",
                 file=sys.stderr)
            aborted = True
            break
        t1 = time.monotonic()
        try:
            res = collect_one_iterate(
                p, pool=pool, entries=entries, states_dir=states_dir,
                entrance_bytes=entrance_bytes, bitmasks=bitmasks,
                extractor=extractor, stack_size=stack_size,
                feature_dim=feature_dim, args=args, out_dir=out_dir,
            )
        except RuntimeError as exc:
            pool.shutdown()
            kind = "BANK GUARD" if "[bank]" in str(exc) else "PURITY GUARD"
            print(f"{kind} RAISED on {p}: {exc}", file=sys.stderr)
            print("registered abort: the whole job is VOID, no partial "
                 "bank is scored", file=sys.stderr)
            return 3
        dt = time.monotonic() - t1
        res["wallclock_s"] = dt
        results.append(res)
        if res.get("probe"):
            probes.append(res["probe"])
        print(f"[collect] iter {res.get('iter')}: rows={res.get('n_rows')} "
             f"pen_rate={res.get('penetration', {}).get('pen_rate')} "
             f"gx_max={res.get('penetration', {}).get('gx_max')} "
             f"({dt:.1f}s)", flush=True)

    pool.shutdown()

    (out_dir / "probe_933.json").write_text(
        json.dumps(probes, indent=2, sort_keys=True))
    summary = {
        "n_iterates_requested": len(paths), "n_iterates_collected": len(results),
        "aborted_on_wallclock": aborted,
        "wallclock_h_total": (time.monotonic() - t0) / 3600.0,
        "results": results,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    out_path = Path(args.summary_out) if args.summary_out else out_dir / "collect_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
