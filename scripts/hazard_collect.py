"""Causal data engine — micro-forking hazard-label collector.

Research synthesis Phase 1 (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md,
"Build order", v18): the highest-value next artifact is a discrete-time
hazard model, and this script is its data source.

Mechanism (micro-forking): from a buffer of saved states, restore a worker to
one of them, COMMIT one action for up to `--horizon` further ticks (an
INTERVENTION, not a passively-observed rollout), and record whether the
worker died and, if so, on which tick. A worker still alive when the horizon
runs out is CENSORED — right-censored survival data, not a survivor-labelled
zero — because a hazard model trained on "0 = safe" pseudo-labels for
censored samples would systematically underestimate risk near the horizon
boundary (the whole point of using IPCW / a survival loss downstream in
Phase 2). This is what makes the dataset support causal survival analysis
rather than a correlational classifier: the (state, action) pair is chosen
by the collector, not by whatever the acting policy happened to prefer.

Restore points come from either a directory of `*.state` save-state blobs,
or a single Go-Explore solution tape (`--states some.npy` + `--root-state`)
replayed once to snapshot a state after every action — mirrors
scripts/smodice_data.py / scripts/gen_iq_transitions.py exactly (same
`load_worker_state` / `save_worker_state` / `step_all` idiom; see those
files' `replay_states`).

Observation history — genuine where the source makes it possible, disclosed
where it doesn't: the recorded `obs` is a `stack_size`-frame stacked tile
observation (the same encoding Phase 2's hazard MLP and the trainer/policy
both consume). When restore points come from a `.npy` solution tape
(`--states tape.npy --root-state ...`), `replay_tape_to_states` already steps
through every real, distinct tick of that trajectory, so the stack is built
from `stack_size` genuinely consecutive real frames ending at the restore
point (`stack_history_window` + `make_stacked_obs`) — real derivative signal
(enemy motion, jump arcs), not a duplicate. When restore points come from a
directory of independent `*.state` blobs, there is no recoverable trajectory
between one file and the next — a save state is only ever the current RAM,
never its history — so `obs` falls back to `TileFeatureStacker.reset()`'s
single-frame duplicate (every stack slot equal to the one instant that was
saved). That fallback looks, to any downstream consumer, like a true
episode-start observation even though most `*.state` restore points are not
episode starts; every dataset's `meta_json` provenance records which regime
produced it (`obs_history_mode`: `"genuine"` for tape sources,
`"reset-duplicate"` for directory sources) so this is never silent.

Death detection reuses the established SMB convention from
scripts/go_explore_solve.py (`DEATH_STATES`): the profile's
`ram_mapping.player_state` byte reading one of the game's dying/death-pit
state codes. This is copied from that already-audited code path, not
authored here from any prior knowledge of the ROM — see `resolve_death_addr`
and `DEATH_STATES` below. It is SMB-shaped (a single enum byte); a profile
for a game whose death signal is a lives-decrement instead (several other
profiles in configs/ use that pattern) is out of scope for this Phase 1 pass
and would need its own death predicate before this script applies to it.

The only emulator-touching entry point is `main()`, which imports
`nes_core.Pool` lazily inside itself — exactly the pattern smodice_data.py
documents ("Emulator imports live here so the commit-logic units above stay
importable in test environments without the built core") — so every
bookkeeping function above it (`run_fork_batch`, `collect_labels`,
`benchmark_stats`, ...) is unit-testable against a stubbed pool with no ROM,
no ctypes, no nes_core import at all.

`load_worker_state` already clears the worker's `frame_cycle_target` inside
the Rust core (nes_core/src/pool.rs, the CRITICAL comment on the load path;
CHANGELOG.md 2026-xx "Fix: `w.frame_cycle_target = None` in
`load_worker_state`") — the known post-restore stall trap is closed at the
binding, so no Python-side workaround is needed here; this file does not
touch that field.

THE GATE (measured by this script itself, not asserted): `--benchmark` runs
a short micro-fork sample (no 100k-label collection) and reports
labels-per-second, steps-per-second, and the projected wall-clock time to
100,000 labels. Two DIFFERENT thresholds from the research round are both
checked, and `benchmark_stats()` keeps them as two separate fields rather
than collapsing them into one boolean:

  * THE GATE itself, `gate_pass` — the Phase 1 numeric target: 100,000
    cleanly labeled transitions in under 1 hour on the M4
    (docs/proposals/RESEARCH_SYNTHESIS_2026-08-17.md line 105). This is
    judged on `projected_seconds_to_target` (extrapolated from the
    benchmark's measured `labels_per_second`) against `--gate-seconds`
    (default 3600). Labels/second is NOT the same number as steps/second —
    each fork spends `horizon + 1` ticks (the settle step included) to
    produce exactly one label, so `labels_per_second == steps_per_second /
    (horizon + 1)` in the common case of one job per worker per batch —
    which is why a run sitting right at the kill threshold below can still
    miss this gate by a wide margin.
  * The KILL criterion, `kill_triggered` — the literal one from the
    research round: below 1,000 steps/second (worker-ticks/second, the
    same "env_steps_per_s" convention scripts/bench_throughput.py uses),
    the causal micro-forking approach is abandoned in favour of harvesting
    observational deaths from ordinary rollout logs instead. This is a
    raw-throughput floor, not a stand-in for the numeric Gate.

`--benchmark`'s exit code and printed "GATE: PASS/FAIL" verdict reflect
`gate_pass` (the numeric target), never `kill_triggered` alone; the KILL
line is printed alongside as its own, separately-labeled verdict.

Usage:
  # Gate check only — no dataset written, no full collection.
  python scripts/hazard_collect.py --profile configs/mario_1_2_solo.yaml \\
      --states checkpoints/.../smb_curriculum/ --benchmark \\
      --forks-per-state 4 --horizon 30 --workers 8

  # Full collection.
  python scripts/hazard_collect.py --profile configs/mario_1_2_solo.yaml \\
      --states checkpoints/.../smb_curriculum/ \\
      --forks-per-state 64 --horizon 60 --workers 16 --seed 0 \\
      --out runs/hazard/mario_1_2_v1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Pure-Python, no emulator import (src/emulation/frame_utils.py's own
# docstring: "No emulator imports — safe to load without any backend
# installed") — safe at module top, unlike the nes_core / profile_utils
# imports below, which stay lazy inside main() per the module docstring.
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402

DEFAULT_ROM = str(REPO / "roms/Super Mario Bros. (World).nes")

# Established SMB death convention, copied verbatim from
# scripts/go_explore_solve.py (`DEATH_STATES`, `R_PSTATE`) — dying (6) and
# death-pit (11) player-state codes at RAM $000E. Not derived from any
# out-of-band knowledge of the ROM; this is the same address/values every
# other honest-eval and solver script in this repo already keys on.
R_PSTATE_DEFAULT = 0x000E
DEATH_STATES = (6, 11)

STATE_FILE_GLOB = "*.state"

NPZ_ARRAY_KEYS = (
    "obs", "action", "died", "steps_to_event", "censored", "source_state_idx",
)

# `obs_history_mode` provenance values — see module docstring "Observation
# history": tape-replayed restore points get a genuine consecutive-frame
# stack; independent *.state files fall back to a single-frame duplicate
# because no real antecedent frames are recoverable from them.
OBS_HISTORY_GENUINE = "genuine"
OBS_HISTORY_RESET_DUPLICATE = "reset-duplicate"


# ---------------------------------------------------------------------
# Death predicate — profile-driven address, established state codes.
# ---------------------------------------------------------------------

def resolve_death_addr(profile: dict) -> int:
    """RAM address of the player-state byte this profile exposes.

    Reads `ram_mapping.player_state` when the profile declares one (every
    Mario profile does); falls back to the go_explore_solve.py hardcoded
    default only when the key is absent, so a profile that customizes the
    address is honored rather than silently overridden.
    """
    ram_mapping = profile.get("ram_mapping") or {}
    addr = ram_mapping.get("player_state")
    if addr is None:
        return R_PSTATE_DEFAULT
    return int(addr, 0) if isinstance(addr, str) else int(addr)


def is_dead(ram: Sequence[int], addr: int,
            death_states: tuple = DEATH_STATES) -> bool:
    return int(ram[addr]) in death_states


# ---------------------------------------------------------------------
# Fork bookkeeping — pure, no emulator.
# ---------------------------------------------------------------------

def label_fork(died: bool, death_step: int | None, horizon: int) -> dict:
    """Turn one fork's raw outcome into its survival label.

    `death_step` is the 1-indexed tick (within `[1, horizon]`) death was
    first observed on, or `None`/ignored when `died` is False.

    A worker still alive when the horizon runs out is CENSORED:
    `censored=1`, `steps_to_event=horizon` — the horizon length itself,
    standing in for "survived at least this long", never a bare `0` that a
    naive reader could mistake for "hazard-free". This is the semantic a
    survival loss with IPCW (Phase 2's C-index gate) depends on.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if died:
        if death_step is None or not (1 <= death_step <= horizon):
            raise ValueError(
                f"died=True requires death_step in [1, {horizon}], got "
                f"{death_step!r}")
        return {"died": 1, "steps_to_event": int(death_step), "censored": 0}
    return {"died": 0, "steps_to_event": int(horizon), "censored": 1}


def stack_history_window(features: Sequence[np.ndarray], idx: int,
                         stack_size: int) -> list[np.ndarray]:
    """Oldest -> newest window of up to `stack_size` REAL per-frame feature
    vectors ending at `features[idx]`, drawn from a trajectory that was
    actually stepped tick-by-tick (`replay_tape_to_states`'s `features`
    output) — i.e. genuine temporal history, not a duplicated instant.

    Near the trajectory's own start (`idx < stack_size - 1`) there simply
    are no earlier real frames yet — a true episode start — so the
    earliest available frame is repeated to fill the remaining slots
    (matches `TileFeatureStacker.reset`'s own "no zero-padding leaks at
    episode start" contract) rather than inventing frames that never
    happened.
    """
    if stack_size <= 0:
        raise ValueError("stack_size must be >= 1")
    if idx < 0 or idx >= len(features):
        raise ValueError(f"idx {idx} out of range for {len(features)} features")
    start = idx - stack_size + 1
    return [features[max(0, start + k)] for k in range(stack_size)]


def make_stacked_obs(window: Sequence[np.ndarray]) -> np.ndarray:
    """Push a real oldest->newest window of per-frame feature vectors
    through a fresh `TileFeatureStacker` via `.push()` (after a single
    `.reset()` seeded on the oldest frame) so the result is `len(window)`
    genuinely distinct frames — the same construction the trainer uses
    incrementally over a live rollout, not `reset()`'s single-frame
    duplicate applied to one instant.
    """
    if not window:
        raise ValueError("window must have at least one frame")
    feature_dim = len(window[0])
    stk = TileFeatureStacker(stack_size=len(window), feature_dim=feature_dim)
    out = stk.reset(np.asarray(window[0]))
    for f in window[1:]:
        out = stk.push(np.asarray(f))
    return np.asarray(out, dtype=np.int8)


def build_fork_jobs(num_states: int, forks_per_state: int, num_actions: int,
                     rng: np.random.Generator) -> list[tuple[int, int]]:
    """One (state_idx, action_id) job per fork, action sampled uniformly.

    Uniform random action assignment — not "whatever a policy would have
    picked" — is what makes each fork a valid intervention rather than a
    passively-observed, policy-confounded transition (the same reasoning
    the module docstring gives for treating this as causal data).
    """
    if num_states <= 0:
        raise ValueError("num_states must be >= 1")
    if forks_per_state <= 0:
        raise ValueError("forks_per_state must be >= 1")
    if num_actions <= 0:
        raise ValueError("num_actions must be >= 1")
    jobs = []
    for s in range(num_states):
        actions = rng.integers(0, num_actions, size=forks_per_state)
        jobs.extend((s, int(a)) for a in actions)
    return jobs


def batch_jobs(jobs: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    for i in range(0, len(jobs), batch_size):
        yield jobs[i:i + batch_size]


def run_fork_batch(pool, jobs: list[tuple[int, int]],
                    source_states: list[bytes], bitmasks: Sequence[int],
                    horizon: int,
                    make_obs: Callable[[np.ndarray, int], np.ndarray],
                    death_fn: Callable[[np.ndarray], bool]) -> list[dict]:
    """Run one batch of forks (<= pool.num_workers) to completion.

    Per job: `pool.load_worker_state(i, source_states[state_idx])`, one
    NOOP settle step to read the pre-intervention RAM (the same
    restore-then-noop idiom `smodice_data.gen_for_solution` and
    `gen_iq_transitions.replay_states` use to get a consistent post-restore
    observation), then the forked action is committed for up to `horizon`
    further ticks. A worker that dies stops receiving new input (holds
    NOOP) for the rest of the batch so every worker in the batch advances
    the same number of ticks and stays index-aligned with its neighbours —
    the same "alive" bookkeeping `smodice_data.gen_for_solution` uses.

    `make_obs(ram, state_idx)` takes the job's `source_state_idx` as well
    as the settle-step ram — callers use `state_idx` to look up genuine
    per-restore-point history (see `stack_history_window`) instead of
    stacking a single instant, when that history is available.

    Returns one dict per job (job order preserved): `obs`, `action`,
    `died`, `steps_to_event`, `censored`, `source_state_idx`.
    """
    b = len(jobs)
    w = getattr(pool, "num_workers")
    if b == 0:
        return []
    if b > w:
        raise ValueError(f"batch of {b} jobs exceeds pool size {w}")

    for i, (state_idx, _action_id) in enumerate(jobs):
        pool.load_worker_state(i, source_states[state_idx])

    noop = np.zeros(w, dtype=np.uint8)
    settled = pool.step_all(noop)
    obs0 = [make_obs(settled[i][2], jobs[i][0]) for i in range(b)]

    died = [False] * b
    death_step: list[int | None] = [None] * b
    for t in range(1, horizon + 1):
        acts = np.zeros(w, dtype=np.uint8)
        for i, (_state_idx, action_id) in enumerate(jobs):
            if not died[i]:
                acts[i] = bitmasks[action_id]
        stepped = pool.step_all(acts)
        for i in range(b):
            if died[i]:
                continue
            if death_fn(stepped[i][2]):
                died[i] = True
                death_step[i] = t

    results = []
    for i, (state_idx, action_id) in enumerate(jobs):
        label = label_fork(died[i], death_step[i], horizon)
        results.append({
            "obs": np.asarray(obs0[i], dtype=np.int8).copy(),
            "action": int(action_id),
            "source_state_idx": int(state_idx),
            **label,
        })
    return results


def collect_labels(pool, source_states: list[bytes], bitmasks: Sequence[int],
                    forks_per_state: int, horizon: int,
                    make_obs: Callable[[np.ndarray, int], np.ndarray],
                    death_fn: Callable[[np.ndarray], bool],
                    rng: np.random.Generator,
                    progress_cb: Callable[[int, int], None] | None = None,
                    ) -> list[dict]:
    """Drive `run_fork_batch` over every job, chunked to `pool.num_workers`."""
    jobs = build_fork_jobs(len(source_states), forks_per_state,
                           len(bitmasks), rng)
    results: list[dict] = []
    for chunk in batch_jobs(jobs, pool.num_workers):
        results.extend(run_fork_batch(pool, chunk, source_states, bitmasks,
                                      horizon, make_obs, death_fn))
        if progress_cb is not None:
            progress_cb(len(results), len(jobs))
    return results


# ---------------------------------------------------------------------
# Restore-point sources — dir of save-states, or a solution tape.
# ---------------------------------------------------------------------

def discover_state_files(states_dir: Path) -> list[Path]:
    files = sorted(Path(states_dir).glob(STATE_FILE_GLOB))
    if not files:
        raise ValueError(
            f"no state files ({STATE_FILE_GLOB}) found under {states_dir}")
    return files


def load_state_bytes(paths: list[Path]) -> list[bytes]:
    return [Path(p).read_bytes() for p in paths]


def sample_restore_indices(available: int, requested: int,
                            rng: np.random.Generator) -> list[int]:
    """Pick up to `requested` distinct indices out of `[0, available)`.

    `requested <= 0` means "use every available restore point". A request
    at or above `available` also uses all of them (sorted, not sampled) —
    sampling can only ever thin the buffer, never pad it.
    """
    if available <= 0:
        raise ValueError("no restore points available to sample from")
    if requested <= 0 or requested >= available:
        return list(range(available))
    idx = rng.choice(available, size=requested, replace=False)
    return sorted(int(i) for i in idx)


def replay_tape_to_states(pool, bitmasks: Sequence[int], root_bytes: bytes,
                           actions: Sequence[int],
                           extract_fn: Callable[[np.ndarray], np.ndarray],
                           worker_id: int = 0
                           ) -> tuple[list[bytes], list[np.ndarray]]:
    """Replay `actions` on one worker from `root_bytes`, snapshot state
    (and its extracted tile features) after every action. Returns
    `(states, features)`, each `len(actions) + 1` long (index 0 is the
    settled root). Mirrors `smodice_data.replay_states` /
    `gen_iq_transitions.replay_states` exactly (same restore -> NOOP-settle
    -> step-and-snapshot loop), generalized off worker 0 only for API
    symmetry with the rest of this module.

    Because this function genuinely steps the emulator tick-by-tick,
    `features[i]` is a REAL frame at position `i` of this actual
    trajectory — not a reconstruction. Callers use `stack_history_window`
    over this `features` list to seed a fork's observation stack with true
    antecedent frames instead of `TileFeatureStacker.reset()`'s single-frame
    duplicate (see the module docstring's "Observation history" section).
    """
    pool.load_worker_state(worker_id, root_bytes)
    w = pool.num_workers

    def step(a: int):
        x = np.zeros(w, dtype=np.uint8)
        x[worker_id] = bitmasks[a]
        return pool.step_all(x)[worker_id][2]

    ram = step(0)
    states = [pool.save_worker_state(worker_id)]
    features = [extract_fn(ram)]
    for a in actions:
        ram = step(int(a))
        states.append(pool.save_worker_state(worker_id))
        features.append(extract_fn(ram))
    return states, features


# ---------------------------------------------------------------------
# npz schema + provenance.
# ---------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def pack_results(results: list[dict]) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("no fork results to pack — collection produced 0 labels")
    obs_dim = {len(r["obs"]) for r in results}
    if len(obs_dim) != 1:
        raise ValueError(f"inconsistent obs width across results: {obs_dim}")
    arrays = {
        "obs": np.stack([r["obs"] for r in results]).astype(np.int8),
        "action": np.array([r["action"] for r in results], dtype=np.int64),
        "died": np.array([r["died"] for r in results], dtype=np.int8),
        "steps_to_event": np.array([r["steps_to_event"] for r in results],
                                   dtype=np.int32),
        "censored": np.array([r["censored"] for r in results], dtype=np.int8),
        "source_state_idx": np.array([r["source_state_idx"] for r in results],
                                     dtype=np.int64),
    }
    assert set(arrays) == set(NPZ_ARRAY_KEYS), (
        f"pack_results schema drifted from NPZ_ARRAY_KEYS: "
        f"{set(arrays)} != {set(NPZ_ARRAY_KEYS)}")
    return arrays


def build_provenance(profile_path, rom_path, source_names: list[str],
                     seed: int, forks_per_state: int, horizon: int,
                     num_labels: int,
                     obs_history_mode: str = OBS_HISTORY_RESET_DUPLICATE
                     ) -> dict:
    """`obs_history_mode` discloses how `obs` was built for this dataset:
    `OBS_HISTORY_GENUINE` (tape-replay source — real consecutive frames)
    or `OBS_HISTORY_RESET_DUPLICATE` (directory-of-*.state source — no
    recoverable antecedent frames, every stack slot is one duplicated
    instant). See module docstring "Observation history"."""
    if obs_history_mode not in (OBS_HISTORY_GENUINE, OBS_HISTORY_RESET_DUPLICATE):
        raise ValueError(f"unknown obs_history_mode: {obs_history_mode!r}")
    rom_p = Path(rom_path)
    return {
        "profile": str(profile_path),
        "rom_path": str(rom_path),
        "rom_sha256": sha256_file(rom_p) if rom_p.exists() else None,
        "source_states": list(source_names),
        "num_source_states": len(source_names),
        "seed": int(seed),
        "forks_per_state": int(forks_per_state),
        "horizon": int(horizon),
        "num_labels": int(num_labels),
        "death_states": list(DEATH_STATES),
        "obs_history_mode": obs_history_mode,
    }


def save_dataset(out_path, results: list[dict], provenance: dict
                  ) -> dict[str, np.ndarray]:
    arrays = pack_results(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta_json=np.array(json.dumps(provenance, sort_keys=True)),
             **arrays)
    return arrays


# ---------------------------------------------------------------------
# THE GATE — benchmark math (pure; the emulator wiring calls this).
# ---------------------------------------------------------------------

def benchmark_stats(num_workers: int, ticks_per_worker: int, elapsed_s: float,
                    num_labels: int, target_labels: int = 100_000,
                    kill_threshold: float = 1_000.0,
                    gate_seconds: float = 3_600.0) -> dict:
    """Compute the throughput numbers THE GATE and the KILL criterion are
    each judged on — kept as two separate booleans, `gate_pass` and
    `kill_triggered`, because they are two different thresholds from the
    research round, not one:

    * `gate_pass` is the actual Phase 1 numeric target — "100,000 cleanly
      labeled transitions in under 1 hour" — judged on
      `projected_seconds_to_target` (extrapolated from this benchmark's
      measured `labels_per_second`) against `gate_seconds`.
    * `kill_triggered` is the literal research-round kill criterion:
      `steps_per_second` (worker-ticks/second, `num_workers *
      ticks_per_worker`, the same "env_steps_per_s" convention
      scripts/bench_throughput.py reports) falling below `kill_threshold`.

    `ticks_per_worker` is the number of `step_all` calls issued during the
    timed window (settle-step included). Note `labels_per_second` is
    NOT `steps_per_second` — each fork spends `horizon + 1` ticks to
    produce one label — so a run can clear the kill threshold on raw
    steps/s while still missing the numeric gate on projected time; that
    gap is exactly why the two verdicts are computed and reported
    independently instead of one standing in for the other.
    """
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be > 0")
    if num_workers <= 0 or ticks_per_worker <= 0:
        raise ValueError("num_workers and ticks_per_worker must be >= 1")
    if num_labels < 0:
        raise ValueError("num_labels must be >= 0")
    if gate_seconds <= 0:
        raise ValueError("gate_seconds must be > 0")
    total_steps = num_workers * ticks_per_worker
    steps_per_second = total_steps / elapsed_s
    labels_per_second = num_labels / elapsed_s
    projected_seconds = (target_labels / labels_per_second
                        if labels_per_second > 0 else math.inf)
    kill_triggered = steps_per_second < kill_threshold
    gate_pass = projected_seconds <= gate_seconds
    return {
        "elapsed_s": elapsed_s,
        "num_workers": num_workers,
        "ticks_per_worker": ticks_per_worker,
        "total_steps": total_steps,
        "steps_per_second": steps_per_second,
        "num_labels": num_labels,
        "labels_per_second": labels_per_second,
        "target_labels": target_labels,
        "projected_seconds_to_target": projected_seconds,
        "projected_minutes_to_target": projected_seconds / 60.0,
        "kill_threshold_steps_per_second": kill_threshold,
        "kill_triggered": kill_triggered,
        "gate_seconds": gate_seconds,
        "gate_pass": gate_pass,
    }


def format_benchmark_report(stats: dict) -> str:
    verdict = "PASS" if stats["gate_pass"] else "FAIL"
    kill_line = (
        f"[hazard_collect] KILL: TRIGGERED — {stats['steps_per_second']:.1f} "
        f"steps/s < {stats['kill_threshold_steps_per_second']:.0f} steps/s, "
        "abandon micro-forking in favour of observational deaths from "
        "ordinary rollout logs"
        if stats["kill_triggered"] else
        f"[hazard_collect] KILL: clear — {stats['steps_per_second']:.1f} "
        f"steps/s >= {stats['kill_threshold_steps_per_second']:.0f} steps/s"
    )
    gate_minutes = stats["gate_seconds"] / 60.0
    lines = [
        f"[hazard_collect] benchmark: {stats['elapsed_s']:.2f}s, "
        f"{stats['num_workers']} workers x {stats['ticks_per_worker']} ticks "
        f"= {stats['total_steps']} steps, {stats['num_labels']} labels",
        f"[hazard_collect] steps/s = {stats['steps_per_second']:.1f} "
        f"(kill threshold {stats['kill_threshold_steps_per_second']:.0f})",
        f"[hazard_collect] labels/s = {stats['labels_per_second']:.2f}",
        f"[hazard_collect] projected time to {stats['target_labels']:,} "
        f"labels: {stats['projected_minutes_to_target']:.1f} min "
        f"(gate: under {gate_minutes:.0f} min)",
        kill_line,
        f"[hazard_collect] GATE: {verdict} — "
        + (f"{stats['projected_minutes_to_target']:.1f} min <= "
           f"{gate_minutes:.0f} min, 100,000 labels reachable within the "
           "Phase 1 time budget"
           if stats["gate_pass"] else
           f"{stats['projected_minutes_to_target']:.1f} min > "
           f"{gate_minutes:.0f} min, 100,000 labels NOT reachable within "
           "the Phase 1 time budget at this throughput"),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rom", default=None,
                    help="Overrides the profile's rom_path.")
    ap.add_argument("--states", required=True,
                    help="Directory of *.state save-states, or a single "
                         ".npy Go-Explore solution tape.")
    ap.add_argument("--root-state", default=None,
                    help="Required when --states is a .npy solution tape.")
    ap.add_argument("--forks-per-state", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=60,
                    help="Frames (ticks) to observe per fork.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="Output npz path. Required unless --benchmark.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states", type=int, default=0,
                    help="Cap on restore points sampled (0 = use all "
                         "discovered/replayed states).")
    ap.add_argument("--benchmark", action="store_true",
                    help="Measure labels/s and steps/s on a small sample "
                         "and report THE GATE verdict; write no dataset.")
    ap.add_argument("--target-labels", type=int, default=100_000)
    ap.add_argument("--kill-threshold", type=float, default=1_000.0)
    ap.add_argument("--gate-seconds", type=float, default=3_600.0,
                    help="THE GATE's time budget in seconds (default 3600 "
                         "= 1 hour) — 'projected time to --target-labels "
                         "labels' must be <= this to PASS.")
    return ap


def validate_args(args: argparse.Namespace) -> list[str]:
    """Pure CLI validation — no filesystem-existence assumptions beyond
    what argparse itself can't express, no emulator. Returns a list of
    human-readable problems (empty == valid)."""
    errs: list[str] = []

    profile_path = Path(args.profile)
    if not profile_path.exists():
        errs.append(f"--profile: {profile_path} does not exist")

    states_path = Path(args.states)
    if not states_path.exists():
        errs.append(f"--states: {states_path} does not exist")
    elif states_path.is_file():
        if states_path.suffix != ".npy":
            errs.append(
                f"--states: {states_path} is a file but not a .npy "
                "solution tape (and not a directory of *.state files)")
        elif not args.root_state:
            errs.append(
                "--root-state is required when --states is a .npy "
                "solution tape")
        elif not Path(args.root_state).exists():
            errs.append(f"--root-state: {args.root_state} does not exist")

    if args.forks_per_state <= 0:
        errs.append("--forks-per-state must be >= 1")
    if args.horizon <= 0:
        errs.append("--horizon must be >= 1")
    if args.workers <= 0:
        errs.append("--workers must be >= 1")
    if args.num_states < 0:
        errs.append("--num-states must be >= 0 (0 = use all)")
    if args.target_labels <= 0:
        errs.append("--target-labels must be >= 1")
    if args.kill_threshold <= 0:
        errs.append("--kill-threshold must be > 0")
    if args.gate_seconds <= 0:
        errs.append("--gate-seconds must be > 0")
    if not args.benchmark and not args.out:
        errs.append("--out is required unless --benchmark is set")

    return errs


def _resolve_states_and_rng(args, pool, bitmasks, extract_fn, stack_size):
    """Emulator-touching restore-point resolution shared by --benchmark
    and full collection. Returns
    `(state_bytes_list, source_names, rng, history, obs_history_mode)`.

    `history[i]` is either a genuine `stack_size`-frame oldest->newest
    window (`stack_history_window` over the actual replayed trajectory,
    tape source) or `None` (directory-of-*.state source — no recoverable
    antecedent frames; `make_obs` falls back to a duplicated single frame).
    `obs_history_mode` records which of the two applies, for
    `build_provenance`.
    """
    rng = np.random.default_rng(args.seed)
    states_path = Path(args.states)
    if states_path.is_dir():
        paths = discover_state_files(states_path)
        idx = sample_restore_indices(len(paths), args.num_states, rng)
        chosen = [paths[i] for i in idx]
        history = [None] * len(chosen)
        return (load_state_bytes(chosen), [str(p) for p in chosen], rng,
                history, OBS_HISTORY_RESET_DUPLICATE)

    actions = np.load(states_path).astype(np.int64)
    root_bytes = Path(args.root_state).read_bytes()
    all_states, all_features = replay_tape_to_states(
        pool, bitmasks, root_bytes, actions, extract_fn)
    idx = sample_restore_indices(len(all_states), args.num_states, rng)
    chosen = [all_states[i] for i in idx]
    names = [f"{states_path.name}#{i}" for i in idx]
    history = [stack_history_window(all_features, i, stack_size) for i in idx]
    return chosen, names, rng, history, OBS_HISTORY_GENUINE


def main(argv: list[str] | None = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    errs = validate_args(args)
    if errs:
        for e in errs:
            print(f"[hazard_collect] error: {e}", file=sys.stderr)
        return 2

    # Emulator + profile-encoder imports live here (not at module top) so
    # every function above stays importable/testable with no built
    # nes_core wheel — matches scripts/smodice_data.py's `main()` import
    # block exactly. (TileFeatureStacker itself is pure-Python and is
    # imported at module top — see the comment there.)
    from nes_core import Pool
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder,
    )

    profile = yaml.safe_load(Path(args.profile).read_text())
    rom_path = args.rom or profile.get("rom_path") or DEFAULT_ROM
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    frame_skip = int(profile.get("frame_skip", 4))
    death_addr = resolve_death_addr(profile)
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    def extract_fn(ram: np.ndarray) -> np.ndarray:
        return extractor.extract(ram)

    def make_obs(ram: np.ndarray, state_idx: int) -> np.ndarray:
        # `history[state_idx]` is a genuine multi-frame window when the
        # restore points came from a solution tape; `None` when they came
        # from independent *.state files with no recoverable antecedent
        # frames (see module docstring "Observation history").
        window = history[state_idx]
        if window is not None:
            return make_stacked_obs(window)
        return TileFeatureStacker(
            stack_size=stack_size, feature_dim=feature_dim
        ).reset(extract_fn(ram))

    def death_fn(ram: np.ndarray) -> bool:
        return is_dead(ram, death_addr)

    pool = Pool(rom_path=str(rom_path), num_workers=args.workers,
               frame_skip=frame_skip)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)

    source_states, source_names, rng, history, obs_history_mode = (
        _resolve_states_and_rng(args, pool, bitmasks, extract_fn, stack_size))
    print(f"[hazard_collect] {len(source_states)} restore points "
          f"({rom_path}, {profile.get('name', args.profile)})", flush=True)

    if args.benchmark:
        t0 = time.perf_counter()
        results = collect_labels(pool, source_states, bitmasks,
                                 args.forks_per_state, args.horizon,
                                 make_obs, death_fn, rng)
        elapsed = time.perf_counter() - t0
        n_jobs = len(source_states) * args.forks_per_state
        n_batches = math.ceil(n_jobs / args.workers)
        ticks_per_worker = n_batches * (args.horizon + 1)  # +1 settle step
        stats = benchmark_stats(args.workers, ticks_per_worker, elapsed,
                                len(results), args.target_labels,
                                args.kill_threshold, args.gate_seconds)
        print(format_benchmark_report(stats), flush=True)
        pool.shutdown()
        return 0 if stats["gate_pass"] else 1

    def progress(done, total):
        if done % max(1, args.workers * 10) == 0 or done == total:
            print(f"[hazard_collect] {done}/{total} labels", flush=True)

    t0 = time.perf_counter()
    results = collect_labels(pool, source_states, bitmasks,
                             args.forks_per_state, args.horizon,
                             make_obs, death_fn, rng, progress_cb=progress)
    elapsed = time.perf_counter() - t0
    pool.shutdown()

    provenance = build_provenance(args.profile, rom_path, source_names,
                                  args.seed, args.forks_per_state,
                                  args.horizon, len(results),
                                  obs_history_mode)
    arrays = save_dataset(args.out, results, provenance)
    n_batches = math.ceil(len(results) / args.workers) if results else 0
    ticks_per_worker = n_batches * (args.horizon + 1)
    stats = benchmark_stats(args.workers, max(1, ticks_per_worker), elapsed,
                            len(results), args.target_labels,
                            args.kill_threshold, args.gate_seconds)
    print(format_benchmark_report(stats), flush=True)
    print(f"[hazard_collect] DONE: {len(arrays['action'])} labels "
          f"(died={int(arrays['died'].sum())}, "
          f"censored={int(arrays['censored'].sum())}) -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
