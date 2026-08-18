"""hazard_collect.py — the causal data engine (research synthesis Phase 1).

Everything here runs against a stubbed pool: no ROM, no nes_core import, no
emulator step. `StubPool` reproduces exactly the slice of the `nes_core.Pool`
surface this module drives (`num_workers`, `load_worker_state`, `step_all`,
`save_worker_state`) and `step_all` returns the same 4-tuple-per-worker shape
(`frame, preprocessed, ram, done`) the real binding does (nes_core/src/
pool.rs `build_result_list`), so `run_fork_batch` / `collect_labels` exercise
the real batching logic while `main()`'s `Pool(...)` construction and ROM
loading never execute.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.hazard_collect import (
    DEATH_STATES,
    NPZ_ARRAY_KEYS,
    OBS_HISTORY_GENUINE,
    OBS_HISTORY_RESET_DUPLICATE,
    build_arg_parser,
    build_fork_jobs,
    build_provenance,
    batch_jobs,
    benchmark_stats,
    collect_labels,
    discover_state_files,
    format_benchmark_report,
    is_dead,
    label_fork,
    load_state_bytes,
    make_stacked_obs,
    pack_results,
    replay_tape_to_states,
    resolve_death_addr,
    run_fork_batch,
    sample_restore_indices,
    save_dataset,
    stack_history_window,
    validate_args,
)

R_PSTATE = 0x000E
RAM_SIZE = 0x0800
FDIM = 6


# ---------------------------------------------------------------------
# StubPool — deterministic fake emulator.
#
# Each worker's "state" is just an integer death-tick script assigned at
# `load_worker_state` time (encoded into the fake state bytes as
# `b"S<idx>"`), so tests control exactly when each worker dies without
# touching a real ROM. `step_all` advances every worker's tick counter by
# one and reports death once `tick >= die_at`.
# ---------------------------------------------------------------------

class StubPool:
    def __init__(self, num_workers, die_at_by_state):
        self.num_workers = num_workers
        self.die_at_by_state = die_at_by_state  # {state_idx: die_at or None}
        self._loaded_state_idx = [None] * num_workers
        self._tick = [0] * num_workers
        # `die_at` counts ticks of the FORKED horizon loop, not the settle
        # step — mirrors run_fork_batch, which never death-checks the
        # settle step's ram (it only reads it for `obs0`). Each worker's
        # first step_all after a load is that free settle step.
        self._settled = [False] * num_workers
        self.load_calls = []
        self.step_calls = 0

    def load_worker_state(self, wid, blob):
        state_idx = int(blob.decode().split("#")[1])
        self._loaded_state_idx[wid] = state_idx
        self._tick[wid] = 0
        self._settled[wid] = False
        self.load_calls.append((wid, state_idx))

    def step_all(self, actions):
        self.step_calls += 1
        out = []
        for i in range(self.num_workers):
            sidx = self._loaded_state_idx[i]
            ram = np.zeros(RAM_SIZE, dtype=np.uint8)
            if sidx is None:
                out.append((None, None, ram, False))
                continue
            if not self._settled[i]:
                self._settled[i] = True
                ram[:FDIM] = (sidx * 10) % 256
            else:
                self._tick[i] += 1
                ram[:FDIM] = (sidx * 10 + self._tick[i]) % 256
                die_at = self.die_at_by_state.get(sidx)
                if die_at is not None and self._tick[i] >= die_at:
                    ram[R_PSTATE] = DEATH_STATES[0]
            out.append((None, None, ram, False))
        return out

    def save_worker_state(self, wid):
        return f"state#{self._loaded_state_idx[wid]}#tick{self._tick[wid]}".encode()


def fake_state_bytes(idx: int) -> bytes:
    return f"state#{idx}".encode()


def make_obs(ram: np.ndarray, state_idx: int) -> np.ndarray:
    # `state_idx` is accepted (matches the real `make_obs(ram, state_idx)`
    # contract run_fork_batch calls) but unused here — these batching
    # tests don't exercise per-restore-point history.
    del state_idx
    return np.asarray(ram[:FDIM], dtype=np.int8)


def death_fn(ram: np.ndarray) -> bool:
    return is_dead(ram, R_PSTATE)


BITMASKS = (0, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------------
# Death predicate.
# ---------------------------------------------------------------------

def test_resolve_death_addr_from_profile():
    assert resolve_death_addr({"ram_mapping": {"player_state": 14}}) == 14
    assert resolve_death_addr({"ram_mapping": {"player_state": "0x14"}}) == 20


def test_resolve_death_addr_falls_back_to_default():
    assert resolve_death_addr({}) == 0x000E
    assert resolve_death_addr({"ram_mapping": {}}) == 0x000E


def test_is_dead_checks_death_states():
    ram = np.zeros(RAM_SIZE, dtype=np.uint8)
    assert not is_dead(ram, R_PSTATE)
    ram[R_PSTATE] = 6
    assert is_dead(ram, R_PSTATE)
    ram[R_PSTATE] = 11
    assert is_dead(ram, R_PSTATE)
    ram[R_PSTATE] = 3
    assert not is_dead(ram, R_PSTATE)


# ---------------------------------------------------------------------
# Censoring semantics — the core correctness property of this dataset.
# ---------------------------------------------------------------------

def test_label_fork_death_within_horizon():
    lbl = label_fork(died=True, death_step=7, horizon=20)
    assert lbl == {"died": 1, "steps_to_event": 7, "censored": 0}


def test_label_fork_death_on_last_tick_is_not_censored():
    lbl = label_fork(died=True, death_step=20, horizon=20)
    assert lbl == {"died": 1, "steps_to_event": 20, "censored": 0}


def test_label_fork_alive_at_horizon_is_censored_not_a_survivor_zero():
    lbl = label_fork(died=False, death_step=None, horizon=20)
    # Right-censored: steps_to_event carries the horizon length, NOT 0 —
    # a bare 0 would misread as "died immediately" to any downstream
    # consumer that doesn't also check `censored`.
    assert lbl == {"died": 0, "steps_to_event": 20, "censored": 1}
    assert lbl["steps_to_event"] != 0


def test_label_fork_rejects_inconsistent_death_step():
    with pytest.raises(ValueError):
        label_fork(died=True, death_step=None, horizon=10)
    with pytest.raises(ValueError):
        label_fork(died=True, death_step=0, horizon=10)
    with pytest.raises(ValueError):
        label_fork(died=True, death_step=11, horizon=10)


def test_label_fork_rejects_bad_horizon():
    with pytest.raises(ValueError):
        label_fork(died=False, death_step=None, horizon=0)


# ---------------------------------------------------------------------
# Fork job construction + batching.
# ---------------------------------------------------------------------

def test_build_fork_jobs_count_and_ranges():
    rng = np.random.default_rng(0)
    jobs = build_fork_jobs(num_states=3, forks_per_state=4, num_actions=6,
                           rng=rng)
    assert len(jobs) == 12
    by_state = {}
    for s, a in jobs:
        by_state.setdefault(s, []).append(a)
        assert 0 <= a < 6
    assert sorted(by_state) == [0, 1, 2]
    assert all(len(v) == 4 for v in by_state.values())


def test_build_fork_jobs_deterministic_under_seed():
    jobs_a = build_fork_jobs(2, 5, 6, np.random.default_rng(42))
    jobs_b = build_fork_jobs(2, 5, 6, np.random.default_rng(42))
    assert jobs_a == jobs_b


@pytest.mark.parametrize("bad_kwargs", [
    dict(num_states=0, forks_per_state=4, num_actions=6),
    dict(num_states=3, forks_per_state=0, num_actions=6),
    dict(num_states=3, forks_per_state=4, num_actions=0),
])
def test_build_fork_jobs_validates(bad_kwargs):
    with pytest.raises(ValueError):
        build_fork_jobs(rng=np.random.default_rng(0), **bad_kwargs)


def test_batch_jobs_chunks_and_keeps_remainder():
    jobs = list(range(10))
    chunks = list(batch_jobs(jobs, 4))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_batch_jobs_rejects_bad_size():
    with pytest.raises(ValueError):
        list(batch_jobs([1, 2], 0))


# ---------------------------------------------------------------------
# run_fork_batch — the batched intervention itself, against StubPool.
# ---------------------------------------------------------------------

def test_run_fork_batch_records_death_tick():
    # State 0 dies on tick 3; state 1 never dies.
    pool = StubPool(num_workers=2, die_at_by_state={0: 3, 1: None})
    jobs = [(0, 1), (1, 2)]
    states = [fake_state_bytes(0), fake_state_bytes(1)]
    results = run_fork_batch(pool, jobs, states, BITMASKS, horizon=10,
                             make_obs=make_obs, death_fn=death_fn)
    assert len(results) == 2
    r0, r1 = results
    assert r0["died"] == 1 and r0["steps_to_event"] == 3 and r0["censored"] == 0
    assert r0["action"] == 1 and r0["source_state_idx"] == 0
    assert r1["died"] == 0 and r1["steps_to_event"] == 10 and r1["censored"] == 1
    assert r1["action"] == 2 and r1["source_state_idx"] == 1


def test_run_fork_batch_obs_is_pre_intervention_settle_frame():
    # obs must come from the settle NOOP step (tick 0), not from stepping
    # under the forked action — i.e. it does not depend on which action
    # was forked.
    pool_a = StubPool(num_workers=1, die_at_by_state={0: None})
    pool_b = StubPool(num_workers=1, die_at_by_state={0: None})
    states = [fake_state_bytes(0)]
    ra = run_fork_batch(pool_a, [(0, 1)], states, BITMASKS, horizon=5,
                        make_obs=make_obs, death_fn=death_fn)
    rb = run_fork_batch(pool_b, [(0, 4)], states, BITMASKS, horizon=5,
                        make_obs=make_obs, death_fn=death_fn)
    np.testing.assert_array_equal(ra[0]["obs"], rb[0]["obs"])


def test_run_fork_batch_make_obs_receives_source_state_idx_for_history_lookup():
    # Mirrors how main() wires make_obs: state_idx selects a per-restore-
    # point history window instead of every job getting the same
    # duplicate-of-one-frame treatment regardless of its source.
    pool = StubPool(num_workers=2, die_at_by_state={0: None, 1: None})
    history = {0: [_feat(10), _feat(11)], 1: None}

    def history_aware_make_obs(ram, state_idx):
        window = history[state_idx]
        if window is not None:
            return make_stacked_obs(window)
        return np.asarray(ram[:FDIM], dtype=np.int8)

    results = run_fork_batch(pool, [(0, 0), (1, 0)],
                             [fake_state_bytes(0), fake_state_bytes(1)],
                             BITMASKS, horizon=3,
                             make_obs=history_aware_make_obs,
                             death_fn=death_fn)
    np.testing.assert_array_equal(
        results[0]["obs"], np.concatenate([_feat(10), _feat(11)]))
    # State 1 has no genuine history -> falls back to its OWN settle-frame
    # ram (StubPool encodes state 1's settle ram as all-10s), not state 0's
    # 12-wide stacked window.
    assert results[1]["obs"].shape == (FDIM,)
    np.testing.assert_array_equal(results[1]["obs"], _feat(10))


def test_run_fork_batch_dead_worker_holds_noop_rest_of_batch():
    pool = StubPool(num_workers=1, die_at_by_state={0: 2})
    run_fork_batch(pool, [(0, 5)], [fake_state_bytes(0)], BITMASKS,
                   horizon=6, make_obs=make_obs, death_fn=death_fn)
    # settle step + 6 horizon ticks = 7 step_all calls regardless of the
    # tick-2 death — every worker in a batch advances the same tick count.
    assert pool.step_calls == 7


def test_run_fork_batch_rejects_oversized_batch():
    pool = StubPool(num_workers=1, die_at_by_state={0: None})
    with pytest.raises(ValueError):
        run_fork_batch(pool, [(0, 0), (0, 1)], [fake_state_bytes(0)],
                       BITMASKS, horizon=5, make_obs=make_obs,
                       death_fn=death_fn)


def test_run_fork_batch_empty_jobs_returns_empty():
    pool = StubPool(num_workers=2, die_at_by_state={})
    assert run_fork_batch(pool, [], [], BITMASKS, horizon=5,
                          make_obs=make_obs, death_fn=death_fn) == []


# ---------------------------------------------------------------------
# collect_labels — multi-batch orchestration.
# ---------------------------------------------------------------------

def test_collect_labels_spans_multiple_batches():
    # 2 workers, 3 states x 2 forks = 6 jobs -> 3 batches of 2.
    pool = StubPool(num_workers=2, die_at_by_state={0: None, 1: 1, 2: None})
    states = [fake_state_bytes(i) for i in range(3)]
    rng = np.random.default_rng(1)
    results = collect_labels(pool, states, BITMASKS, forks_per_state=2,
                             horizon=4, make_obs=make_obs, death_fn=death_fn,
                             rng=rng)
    assert len(results) == 6
    by_state = {}
    for r in results:
        by_state.setdefault(r["source_state_idx"], []).append(r)
    assert len(by_state[1]) == 2
    assert all(r["died"] == 1 and r["steps_to_event"] == 1
              for r in by_state[1])
    assert all(r["censored"] == 1 for r in by_state[0])
    assert all(r["censored"] == 1 for r in by_state[2])


def test_collect_labels_progress_callback_reaches_total():
    pool = StubPool(num_workers=3, die_at_by_state={0: None})
    seen = []
    collect_labels(pool, [fake_state_bytes(0)], BITMASKS, forks_per_state=7,
                   horizon=2, make_obs=make_obs, death_fn=death_fn,
                   rng=np.random.default_rng(0),
                   progress_cb=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (7, 7)
    assert seen[-1][0] <= seen[-1][1]


# ---------------------------------------------------------------------
# Restore-point discovery.
# ---------------------------------------------------------------------

def test_discover_state_files_sorted(tmp_path):
    for name in ("stage_02.state", "stage_10.state", "stage_01.state"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"ignore me")
    files = discover_state_files(tmp_path)
    assert [f.name for f in files] == [
        "stage_01.state", "stage_02.state", "stage_10.state"]


def test_discover_state_files_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        discover_state_files(tmp_path)


def test_load_state_bytes_roundtrip(tmp_path):
    p = tmp_path / "a.state"
    p.write_bytes(b"\x01\x02\x03")
    assert load_state_bytes([p]) == [b"\x01\x02\x03"]


def test_sample_restore_indices_uses_all_when_requested_le_zero():
    rng = np.random.default_rng(0)
    assert sample_restore_indices(5, 0, rng) == [0, 1, 2, 3, 4]


def test_sample_restore_indices_uses_all_when_requested_ge_available():
    rng = np.random.default_rng(0)
    assert sample_restore_indices(5, 99, rng) == [0, 1, 2, 3, 4]


def test_sample_restore_indices_thins_and_is_sorted_no_dupes():
    rng = np.random.default_rng(3)
    idx = sample_restore_indices(20, 5, rng)
    assert len(idx) == 5
    assert len(set(idx)) == 5
    assert idx == sorted(idx)
    assert all(0 <= i < 20 for i in idx)


def test_sample_restore_indices_rejects_empty():
    with pytest.raises(ValueError):
        sample_restore_indices(0, 3, np.random.default_rng(0))


# ---------------------------------------------------------------------
# Observation history — genuine tape-replay stacking vs the disclosed
# reset-duplicate fallback (defect: obs used to be `stack_size` copies of
# one instant for every source, indistinguishable from a real episode
# start, with no way for a downstream consumer to tell).
# ---------------------------------------------------------------------

def _feat(v, dim=FDIM):
    return np.full(dim, v, dtype=np.int8)


def test_stack_history_window_uses_real_preceding_frames():
    features = [_feat(0), _feat(1), _feat(2), _feat(3), _feat(4)]
    window = stack_history_window(features, idx=4, stack_size=3)
    # Oldest -> newest, three genuinely distinct real frames, not one
    # frame duplicated three times.
    np.testing.assert_array_equal(window[0], _feat(2))
    np.testing.assert_array_equal(window[1], _feat(3))
    np.testing.assert_array_equal(window[2], _feat(4))


def test_stack_history_window_pads_only_at_true_trajectory_start():
    features = [_feat(0), _feat(1), _feat(2)]
    # idx=1 with stack_size=4: only 2 real frames exist yet (idx 0, 1) —
    # the true start of the trajectory — so the earliest is repeated.
    window = stack_history_window(features, idx=1, stack_size=4)
    np.testing.assert_array_equal(window[0], _feat(0))
    np.testing.assert_array_equal(window[1], _feat(0))
    np.testing.assert_array_equal(window[2], _feat(0))
    np.testing.assert_array_equal(window[3], _feat(1))


def test_stack_history_window_validates():
    features = [_feat(0), _feat(1)]
    with pytest.raises(ValueError):
        stack_history_window(features, idx=5, stack_size=2)
    with pytest.raises(ValueError):
        stack_history_window(features, idx=0, stack_size=0)


def test_make_stacked_obs_differs_from_reset_duplicate_for_varying_frames():
    from src.emulation.frame_utils import TileFeatureStacker

    window = [_feat(0), _feat(1), _feat(2)]
    genuine = make_stacked_obs(window)
    duplicate = TileFeatureStacker(
        stack_size=3, feature_dim=FDIM).reset(_feat(2))
    # Genuine history carries the real per-frame derivative; the
    # reset-duplicate fallback is flat (same instant repeated) — they must
    # not be equal, otherwise the "fix" changed nothing observable.
    assert not np.array_equal(genuine, duplicate)
    np.testing.assert_array_equal(genuine[:FDIM], _feat(0))
    np.testing.assert_array_equal(genuine[FDIM:2 * FDIM], _feat(1))
    np.testing.assert_array_equal(genuine[2 * FDIM:], _feat(2))


def test_make_stacked_obs_matches_incremental_push():
    from src.emulation.frame_utils import TileFeatureStacker

    window = [_feat(5), _feat(6), _feat(7), _feat(8)]
    got = make_stacked_obs(window)
    stk = TileFeatureStacker(stack_size=4, feature_dim=FDIM)
    want = stk.reset(window[0])
    for f in window[1:]:
        want = stk.push(f)
    np.testing.assert_array_equal(got, want)


def test_make_stacked_obs_rejects_empty_window():
    with pytest.raises(ValueError):
        make_stacked_obs([])


def test_replay_tape_to_states_features_are_real_distinct_frames():
    # StubPool's ram[:FDIM] encodes (state_idx*10 + tick) — a genuinely
    # different value every tick, so features pulled straight from the
    # replay must differ step to step, unlike a duplicated single frame.
    pool = StubPool(num_workers=1, die_at_by_state={0: None})

    def extract_fn(ram):
        return np.asarray(ram[:FDIM], dtype=np.int8)

    states, features = replay_tape_to_states(
        pool, BITMASKS, fake_state_bytes(0), actions=[0, 1, 2],
        extract_fn=extract_fn)
    assert len(states) == len(features) == 4
    values = [int(f[0]) for f in features]
    # Monotonically increasing tick signature -> every frame distinct.
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_replay_tape_to_states_history_window_feeds_make_stacked_obs():
    pool = StubPool(num_workers=1, die_at_by_state={0: None})

    def extract_fn(ram):
        return np.asarray(ram[:FDIM], dtype=np.int8)

    _states, features = replay_tape_to_states(
        pool, BITMASKS, fake_state_bytes(0), actions=[0, 1, 2, 3],
        extract_fn=extract_fn)
    window = stack_history_window(features, idx=4, stack_size=3)
    obs = make_stacked_obs(window)
    # Reconstructing via a fresh reset()-duplicate on just the last frame
    # would be flat; the genuine window is not.
    duplicate_equiv = np.concatenate([features[4]] * 3)
    assert not np.array_equal(obs, duplicate_equiv)


# ---------------------------------------------------------------------
# npz schema + provenance.
# ---------------------------------------------------------------------

def _fake_results(n=4, obs_dim=FDIM):
    out = []
    for i in range(n):
        died = i % 2 == 0
        lbl = label_fork(died=died, death_step=(i + 1) if died else None,
                         horizon=10)
        out.append({
            "obs": np.full(obs_dim, i, dtype=np.int8),
            "action": i % 3,
            "source_state_idx": i % 2,
            **lbl,
        })
    return out


def test_pack_results_schema_and_dtypes():
    arrays = pack_results(_fake_results(6))
    assert set(arrays) == set(NPZ_ARRAY_KEYS)
    assert arrays["obs"].dtype == np.int8
    assert arrays["obs"].shape == (6, FDIM)
    assert arrays["action"].dtype == np.int64
    assert arrays["died"].dtype == np.int8
    assert arrays["steps_to_event"].dtype == np.int32
    assert arrays["censored"].dtype == np.int8
    assert arrays["source_state_idx"].dtype == np.int64
    # died and censored are complementary (this dataset never leaves a
    # third state) for every row.
    np.testing.assert_array_equal(
        arrays["died"] + arrays["censored"], np.ones(6, dtype=np.int8))


def test_pack_results_rejects_empty():
    with pytest.raises(ValueError):
        pack_results([])


def test_pack_results_rejects_ragged_obs_width():
    results = _fake_results(2)
    results[1]["obs"] = np.zeros(FDIM + 1, dtype=np.int8)
    with pytest.raises(ValueError):
        pack_results(results)


def test_build_provenance_fields(tmp_path):
    rom = tmp_path / "rom.nes"
    rom.write_bytes(b"NES\x1a" + b"\x00" * 20)
    prov = build_provenance("configs/mario_1_2_solo.yaml", rom,
                            ["a.state", "b.state"], seed=7,
                            forks_per_state=8, horizon=30, num_labels=16)
    assert prov["seed"] == 7
    assert prov["forks_per_state"] == 8
    assert prov["horizon"] == 30
    assert prov["num_labels"] == 16
    assert prov["num_source_states"] == 2
    assert prov["source_states"] == ["a.state", "b.state"]
    assert prov["death_states"] == list(DEATH_STATES)
    assert prov["rom_sha256"] is not None
    assert len(prov["rom_sha256"]) == 64
    # Default (unspecified) is the honest conservative assumption: no
    # genuine history unless the caller explicitly proves otherwise.
    assert prov["obs_history_mode"] == OBS_HISTORY_RESET_DUPLICATE


def test_build_provenance_missing_rom_hash_is_none(tmp_path):
    prov = build_provenance("p.yaml", tmp_path / "does_not_exist.nes", [],
                            seed=0, forks_per_state=1, horizon=1,
                            num_labels=0)
    assert prov["rom_sha256"] is None


def test_build_provenance_records_genuine_obs_history_mode(tmp_path):
    prov = build_provenance("p.yaml", tmp_path / "missing.nes", ["tape#0"],
                            seed=0, forks_per_state=1, horizon=1,
                            num_labels=1, obs_history_mode=OBS_HISTORY_GENUINE)
    assert prov["obs_history_mode"] == OBS_HISTORY_GENUINE


def test_build_provenance_rejects_unknown_obs_history_mode(tmp_path):
    with pytest.raises(ValueError):
        build_provenance("p.yaml", tmp_path / "missing.nes", [],
                         seed=0, forks_per_state=1, horizon=1, num_labels=0,
                         obs_history_mode="bogus")


def test_save_dataset_roundtrip(tmp_path):
    results = _fake_results(5)
    prov = build_provenance("p.yaml", tmp_path / "missing.nes", ["s0"],
                            seed=1, forks_per_state=5, horizon=10,
                            num_labels=5)
    out = tmp_path / "sub" / "hazard.npz"
    arrays = save_dataset(out, results, prov)
    assert out.exists()

    d = np.load(out)
    for key in ("obs", "action", "died", "steps_to_event", "censored",
               "source_state_idx", "meta_json"):
        assert key in d.files
    np.testing.assert_array_equal(d["obs"], arrays["obs"])
    meta = json.loads(str(d["meta_json"]))
    assert meta["seed"] == 1
    assert meta["num_labels"] == 5


# ---------------------------------------------------------------------
# THE GATE — benchmark math.
#
# `gate_pass` is the actual Phase 1 numeric target (100,000 labels in
# under 1 hour, judged on projected_seconds_to_target vs gate_seconds);
# `kill_triggered` is the separate literal steps/s floor from the research
# round. They are DIFFERENT thresholds and must not collapse into one
# boolean — that collapse was the defect (see
# test_benchmark_stats_at_kill_threshold_can_still_miss_the_gate below).
# ---------------------------------------------------------------------

def test_benchmark_stats_pass_above_threshold():
    # 16 workers x 100 ticks / 1s = 1600 steps/s >= 1000 -> kill clear.
    # labels/s = 200 -> projected 500s <= 3600s gate -> PASS.
    stats = benchmark_stats(num_workers=16, ticks_per_worker=100,
                            elapsed_s=1.0, num_labels=200)
    assert stats["total_steps"] == 1600
    assert stats["steps_per_second"] == pytest.approx(1600.0)
    assert stats["labels_per_second"] == pytest.approx(200.0)
    assert stats["projected_seconds_to_target"] == pytest.approx(500.0)
    assert stats["kill_triggered"] is False
    assert stats["gate_pass"] is True


def test_benchmark_stats_fail_below_threshold():
    # 4 workers x 100 ticks / 1s = 400 steps/s < 1000 -> kill triggered,
    # and projected (100000/10=10000s) also blows the 3600s gate.
    stats = benchmark_stats(num_workers=4, ticks_per_worker=100,
                            elapsed_s=1.0, num_labels=10)
    assert stats["steps_per_second"] == pytest.approx(400.0)
    assert stats["kill_triggered"] is True
    assert stats["gate_pass"] is False


def test_benchmark_stats_custom_threshold_and_target():
    stats = benchmark_stats(num_workers=1, ticks_per_worker=100,
                            elapsed_s=1.0, num_labels=50,
                            target_labels=1000, kill_threshold=50.0)
    assert stats["kill_triggered"] is False  # 100 steps/s >= 50
    assert stats["projected_seconds_to_target"] == pytest.approx(20.0)
    assert stats["gate_pass"] is True  # 20s <= default 3600s gate


def test_benchmark_stats_zero_labels_projects_infinite():
    stats = benchmark_stats(num_workers=8, ticks_per_worker=10,
                            elapsed_s=1.0, num_labels=0)
    assert stats["labels_per_second"] == 0.0
    assert stats["projected_seconds_to_target"] == float("inf")
    assert stats["gate_pass"] is False


def test_benchmark_stats_at_kill_threshold_can_still_miss_the_gate():
    """The defect this reproduces: a run sitting exactly at the 1,000
    steps/s kill threshold, at the default horizon=60 (61 ticks/worker
    per fork including the settle step), produces only 1000/61 ~= 16.39
    labels/s — a projected ~101.7 minutes to 100,000 labels, well past
    the 1-hour Gate — even though it clears the kill floor. The old
    `gate_pass = steps_per_second >= kill_threshold` printed "GATE: PASS"
    here; the fix must print FAIL because the numeric Gate is missed.
    """
    stats = benchmark_stats(num_workers=1000, ticks_per_worker=61,
                            elapsed_s=61.0, num_labels=1000)
    assert stats["steps_per_second"] == pytest.approx(1000.0)
    assert stats["kill_triggered"] is False  # exactly at the floor, not below
    assert stats["labels_per_second"] == pytest.approx(1000.0 / 61.0)
    assert stats["projected_minutes_to_target"] == pytest.approx(
        101.6666, rel=1e-3)
    assert stats["gate_pass"] is False
    assert "GATE: FAIL" in format_benchmark_report(stats)
    assert "KILL: clear" in format_benchmark_report(stats)


@pytest.mark.parametrize("bad_kwargs", [
    dict(num_workers=0, ticks_per_worker=10, elapsed_s=1.0, num_labels=1),
    dict(num_workers=8, ticks_per_worker=0, elapsed_s=1.0, num_labels=1),
    dict(num_workers=8, ticks_per_worker=10, elapsed_s=0.0, num_labels=1),
    dict(num_workers=8, ticks_per_worker=10, elapsed_s=1.0, num_labels=-1),
    dict(num_workers=8, ticks_per_worker=10, elapsed_s=1.0, num_labels=1,
        gate_seconds=0.0),
])
def test_benchmark_stats_validates(bad_kwargs):
    with pytest.raises(ValueError):
        benchmark_stats(**bad_kwargs)


def test_format_benchmark_report_mentions_verdict():
    passing = benchmark_stats(16, 100, 1.0, 200)
    failing = benchmark_stats(4, 100, 1.0, 10)
    assert "GATE: PASS" in format_benchmark_report(passing)
    assert "GATE: FAIL" in format_benchmark_report(failing)
    assert "KILL: TRIGGERED" in format_benchmark_report(failing)
    assert "abandon micro-forking" in format_benchmark_report(failing)


def test_format_benchmark_report_kill_and_gate_are_independent_lines():
    # kill clear but gate still fails (see the kill-threshold defect test
    # above) — both verdicts must appear, and correctly, in the same
    # report.
    stats = benchmark_stats(num_workers=1000, ticks_per_worker=61,
                            elapsed_s=61.0, num_labels=1000)
    report = format_benchmark_report(stats)
    assert "KILL: clear" in report
    assert "GATE: FAIL" in report


# ---------------------------------------------------------------------
# CLI validation.
# ---------------------------------------------------------------------

def _profile_and_states(tmp_path):
    profile = tmp_path / "p.yaml"
    profile.write_text("name: test\n")
    states_dir = tmp_path / "states"
    states_dir.mkdir()
    (states_dir / "stage_00.state").write_bytes(b"x")
    return profile, states_dir


def test_validate_args_happy_path(tmp_path):
    profile, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(states_dir),
        "--out", str(tmp_path / "out.npz"),
    ])
    assert validate_args(args) == []


def test_validate_args_benchmark_does_not_require_out(tmp_path):
    profile, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(states_dir),
        "--benchmark",
    ])
    assert validate_args(args) == []


def test_validate_args_missing_out_without_benchmark(tmp_path):
    profile, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(states_dir),
    ])
    errs = validate_args(args)
    assert any("--out" in e for e in errs)


def test_validate_args_missing_profile(tmp_path):
    _, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(tmp_path / "nope.yaml"), "--states", str(states_dir),
        "--out", str(tmp_path / "out.npz"),
    ])
    errs = validate_args(args)
    assert any("--profile" in e for e in errs)


def test_validate_args_tape_requires_root_state(tmp_path):
    profile, _ = _profile_and_states(tmp_path)
    tape = tmp_path / "sol.npy"
    np.save(tape, np.zeros(5, dtype=np.int64))
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(tape),
        "--out", str(tmp_path / "out.npz"),
    ])
    errs = validate_args(args)
    assert any("--root-state" in e for e in errs)


def test_validate_args_tape_with_root_state_ok(tmp_path):
    profile, _ = _profile_and_states(tmp_path)
    tape = tmp_path / "sol.npy"
    np.save(tape, np.zeros(5, dtype=np.int64))
    root = tmp_path / "root.state"
    root.write_bytes(b"x")
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(tape),
        "--root-state", str(root), "--out", str(tmp_path / "out.npz"),
    ])
    assert validate_args(args) == []


def test_validate_args_non_npy_file_rejected(tmp_path):
    profile, _ = _profile_and_states(tmp_path)
    bogus = tmp_path / "solution.txt"
    bogus.write_text("nope")
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(bogus),
        "--out", str(tmp_path / "out.npz"),
    ])
    errs = validate_args(args)
    assert any("--states" in e for e in errs)


@pytest.mark.parametrize("flag,value", [
    ("--forks-per-state", "0"),
    ("--horizon", "0"),
    ("--workers", "0"),
    ("--num-states", "-1"),
    ("--target-labels", "0"),
    ("--kill-threshold", "0"),
    ("--gate-seconds", "0"),
])
def test_validate_args_rejects_nonpositive_numeric_flags(tmp_path, flag, value):
    profile, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(states_dir),
        "--out", str(tmp_path / "out.npz"), flag, value,
    ])
    assert validate_args(args) != []


def test_validate_args_defaults_are_sane(tmp_path):
    profile, states_dir = _profile_and_states(tmp_path)
    args = build_arg_parser().parse_args([
        "--profile", str(profile), "--states", str(states_dir),
        "--out", str(tmp_path / "out.npz"),
    ])
    assert args.forks_per_state == 8
    assert args.horizon == 60
    assert args.workers == 8
    assert args.seed == 0
    assert args.num_states == 0
    assert args.benchmark is False
    assert args.target_labels == 100_000
    assert args.kill_threshold == 1_000.0
    assert args.gate_seconds == 3_600.0
