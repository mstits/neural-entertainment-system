"""Unit tests for the backward start-state curriculum
(src/training/backward_curriculum.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.training.backward_curriculum import (
    DEFAULT_GX_RESET_MAX,
    DEFAULT_GX_TOLERANCE,
    ENTRANCE,
    INDEX_NAME,
    STATE_VERSION,
    StateEntry,
    TauScheduler,
    draw_restart,
    gx_report,
    load_blobs,
    load_index,
    snapshot_steps,
    state_filename,
    stride_steps,
    write_index,
)
from src.training.go_explore import start_window

REPO = Path(__file__).resolve().parent.parent


def _entries(spec) -> list:
    """Build entries from (step, gx, area) triples; frame = step * 4."""
    return [StateEntry(step=s, frame=s * 4, gx=gx, area=area,
                       file=state_filename(s))
            for s, gx, area in spec]


# ---- mint planning --------------------------------------------------


def test_snapshot_steps_one_state_per_step_at_native_stride() -> None:
    # every_frames == frames_per_step is the default cadence: every step.
    assert snapshot_steps(5, frames_per_step=4, every_frames=4) == [0, 1, 2, 3, 4]


def test_snapshot_steps_coarser_stride_skips_steps() -> None:
    assert snapshot_steps(10, frames_per_step=4, every_frames=16) == [0, 4, 8]
    assert stride_steps(4, 16) == 4


def test_snapshot_steps_rounds_sub_step_stride_up_to_one_step() -> None:
    # The emulator has no state between two frames of one policy action.
    assert snapshot_steps(3, frames_per_step=4, every_frames=1) == [0, 1, 2]
    assert stride_steps(4, 1) == 1


def test_snapshot_steps_omits_the_post_clear_state() -> None:
    # A restart after the last action has already won.
    assert max(snapshot_steps(7, frames_per_step=4, every_frames=4)) == 6
    assert snapshot_steps(0, frames_per_step=4, every_frames=4) == []


@pytest.mark.parametrize("kwargs", [
    {"n_actions": -1, "frames_per_step": 4, "every_frames": 4},
    {"n_actions": 4, "frames_per_step": 0, "every_frames": 4},
    {"n_actions": 4, "frames_per_step": 4, "every_frames": 0},
])
def test_snapshot_steps_rejects_nonsense(kwargs) -> None:
    with pytest.raises(ValueError):
        snapshot_steps(kwargs.pop("n_actions"), **kwargs)


def test_state_filename_is_sortable() -> None:
    names = [state_filename(s) for s in (0, 9, 10, 100)]
    assert names == sorted(names)


# ---- gx monotonicity gate -------------------------------------------


def test_gx_report_tolerates_sub_tile_jitter() -> None:
    # Landing recoil / deceleration dips x a few pixels — measured on the
    # banked 1-1 tape (47 dips, max 4 px) and not a regression.
    rep = gx_report(_entries([(0, 100, 0), (1, 98, 0), (2, 140, 0)]))
    assert rep["decreases"] == 1
    assert rep["max_drop"] == 2
    assert rep["monotone"] is True
    assert rep["strict_monotone"] is False


def test_gx_report_flags_a_real_regression() -> None:
    rep = gx_report(_entries([(0, 2000, 0), (1, 1200, 0)]))
    assert rep["decreases_over_tolerance"] == 1
    assert rep["monotone"] is False


def test_gx_report_segments_on_area_change() -> None:
    # 1-2 step 115: area 1 -> 2 re-bases x at the new region's origin.
    rep = gx_report(_entries([(0, 160, 1), (1, 0, 2), (2, 60, 2)]))
    assert rep["areas"] == [1, 2]
    assert rep["segments"] == 2
    assert rep["monotone"] is True


def test_gx_report_segments_on_same_area_odometer_reset() -> None:
    # 1-2 step 971: the pipe out of the underground re-bases x while the
    # area byte does NOT move. Without this rule the gate fails the tape.
    rep = gx_report(_entries([(0, 2656, 2), (1, 0, 2), (2, 40, 2)]))
    assert rep["areas"] == [2]
    assert rep["segments"] == 2
    assert rep["monotone"] is True
    assert rep["max_drop"] == 0


def test_gx_report_reset_rule_does_not_excuse_a_mid_level_drop() -> None:
    # Landing above reset_max is a regression, not a re-base.
    rep = gx_report(_entries([(0, 2656, 2), (1, DEFAULT_GX_RESET_MAX + 1, 2)]))
    assert rep["monotone"] is False
    assert rep["segments"] == 1


def test_gx_report_reset_rule_does_not_swallow_jitter_near_the_origin() -> None:
    # Early in a level gx is already below reset_max, so the re-base rule
    # needs the drop-size half too or every dip there reads as a segment.
    rep = gx_report(_entries([(0, 100, 0), (1, 98, 0)]))
    assert rep["segments"] == 1
    assert rep["decreases"] == 1


def test_gx_report_empty_is_vacuously_monotone() -> None:
    rep = gx_report([])
    assert rep == dict(rep, n=0, monotone=True, segments=0,
                       tolerance=DEFAULT_GX_TOLERANCE)


# ---- index IO -------------------------------------------------------


def test_index_round_trips(tmp_path) -> None:
    entries = _entries([(0, 40, 0), (1, 56, 0), (2, 72, 0)])
    write_index(tmp_path, entries, {"level": "1-1", "every_frames": 4})
    got, meta = load_index(tmp_path)
    assert got == entries
    assert meta["level"] == "1-1"
    assert meta["every_frames"] == 4
    assert meta["dir"] == str(tmp_path)
    assert "entries" not in meta


def test_load_index_sorts_by_step(tmp_path) -> None:
    entries = _entries([(4, 90, 0), (0, 40, 0), (2, 72, 0)])
    write_index(tmp_path, entries, {})
    got, _ = load_index(tmp_path)
    assert [e.step for e in got] == [0, 2, 4]


def test_load_index_rejects_an_empty_index(tmp_path) -> None:
    (tmp_path / INDEX_NAME).write_text(json.dumps({"entries": []}))
    with pytest.raises(ValueError):
        load_index(tmp_path)


def test_load_blobs_reads_in_entry_order(tmp_path) -> None:
    entries = _entries([(0, 40, 0), (1, 56, 0)])
    for i, e in enumerate(entries):
        (tmp_path / e.file).write_bytes(bytes([i]) * 4)
    assert load_blobs(tmp_path, entries) == [b"\x00" * 4, b"\x01" * 4]


# ---- the backward cursor --------------------------------------------


def _sched(**kw) -> TauScheduler:
    kw.setdefault("advance_entries", 10)
    kw.setdefault("advance_threshold", 0.2)
    kw.setdefault("min_attempts", 30)
    return TauScheduler(kw.pop("n_entries", 100), **kw)


def test_tau_init_minus_one_is_the_last_entry() -> None:
    assert _sched().tau == 99


def test_tau_init_zero_is_the_entrance() -> None:
    s = _sched(tau_init=0)
    assert s.tau == 0 and s.at_entrance


def test_tau_init_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        _sched(tau_init=100)
    with pytest.raises(IndexError):
        _sched(tau_init=-101)


@pytest.mark.parametrize("kw", [
    {"n_entries": 0}, {"advance_entries": 0}, {"advance_threshold": 1.5},
    {"min_attempts": 0},
])
def test_scheduler_rejects_nonsense(kw) -> None:
    with pytest.raises(ValueError):
        _sched(**kw)


def test_advance_needs_the_attempt_floor_even_at_a_perfect_rate() -> None:
    # The project's 3-episode gates were mirages; the floor is hard.
    s = _sched()
    for _ in range(29):
        s.record(True)
    assert s.rate == 1.0
    assert s.maybe_advance() is False
    assert s.tau == 99
    s.record(True)
    assert s.maybe_advance() is True
    assert s.tau == 89


def test_advance_needs_the_threshold_even_with_many_attempts() -> None:
    s = _sched()
    for i in range(60):
        s.record(i % 10 == 0)      # 0.10 < 0.20
    assert s.maybe_advance() is False
    assert s.tau == 99


def test_advance_clears_the_trailing_window() -> None:
    s = _sched()
    for _ in range(30):
        s.record(True)
    s.maybe_advance()
    assert s.attempts == 0 and s.successes == 0 and s.rate == 0.0


def test_advance_clamps_at_the_entrance_and_then_stops() -> None:
    s = _sched(n_entries=15, tau_init=-1, advance_entries=10)
    for _ in range(30):
        s.record(True)
    assert s.maybe_advance() is True and s.tau == 4
    for _ in range(30):
        s.record(True)
    assert s.maybe_advance() is True and s.tau == 0 and s.at_entrance
    for _ in range(30):
        s.record(True)
    # At the entrance the curriculum is over — the cursor cannot move on.
    assert s.maybe_advance() is False and s.tau == 0


def test_stale_tau_attempts_are_dropped() -> None:
    s = _sched()
    for _ in range(30):
        s.record(True)
    s.maybe_advance()
    assert s.record(False, tau=99) is False   # started on the retired rung
    assert s.attempts == 0
    assert s.record(False, tau=89) is True
    assert s.attempts == 1


def test_trailing_window_forgets_old_attempts() -> None:
    s = _sched(min_attempts=10)
    for _ in range(10):
        s.record(False)
    for _ in range(10):
        s.record(True)
    assert s.attempts == 10 and s.rate == 1.0


def test_entrance_attempts_are_scored_separately() -> None:
    s = _sched()
    for _ in range(40):
        s.record_entrance(False)
    s.record_entrance(True)
    assert s.attempts == 0            # never pollutes the rung
    assert s.maybe_advance() is False
    snap = s.snapshot()
    assert snap["entrance_attempts"] == 41
    assert snap["entrance_successes"] == 1
    assert snap["entrance_rate"] == pytest.approx(1 / 41)


def test_snapshot_reports_the_cursor_state() -> None:
    s = _sched()
    for _ in range(30):
        s.record(True)
    s.maybe_advance()
    snap = s.snapshot()
    assert snap["tau"] == 89
    assert snap["n_entries"] == 100
    assert snap["advances"] == 1
    assert snap["at_entrance"] is False


# ---- cursor persistence (resume) ------------------------------------
#
# A curriculum whose cursor is not checkpointed restarts at the ladder top
# on every relaunch (B4 v3: resumed at iter 80 with tau back at 757/757,
# so the at-entrance winner gate never fired again).


def _mid_ladder(**kw) -> TauScheduler:
    """A cursor that has advanced twice and is part-way up a third rung."""
    s = _sched(**kw)
    for _ in range(60):
        s.record(True)
        s.maybe_advance()
    for _ in range(7):
        s.record(True)
    for _ in range(3):
        s.record(False)
    for _ in range(5):
        s.record_entrance(False)
    s.record_entrance(True)
    return s


def test_state_dict_round_trips_the_whole_cursor() -> None:
    s = _mid_ladder()
    fresh = _sched()                      # would start at the ladder top
    assert fresh.tau != s.tau
    fresh.load_state_dict(s.state_dict())
    assert fresh.snapshot() == s.snapshot()


def test_state_dict_survives_a_json_round_trip() -> None:
    # The payload rides inside a torch.save'd checkpoint; keeping it plain
    # data (no deque, no numpy) is what makes it inspectable and portable.
    s = _mid_ladder()
    revived = json.loads(json.dumps(s.state_dict()))
    fresh = _sched()
    fresh.load_state_dict(revived)
    assert fresh.snapshot() == s.snapshot()


def test_restore_keeps_the_partial_window_not_just_the_rate() -> None:
    """The attempt floor must not be silently re-armed by a resume.

    A rung 20 attempts into its 30-attempt floor resumes 20 attempts in —
    10 more decide it. Restoring only tau (or an empty window) would make
    every relaunch re-pay the full floor at the current rung.
    """
    s = _sched(min_attempts=30)
    for _ in range(20):
        s.record(True)
    resumed = _sched(min_attempts=30)
    resumed.load_state_dict(s.state_dict())
    assert resumed.attempts == 20 and resumed.rate == 1.0
    for _ in range(9):
        resumed.record(True)
    assert resumed.maybe_advance() is False       # 29 < 30
    resumed.record(True)
    assert resumed.maybe_advance() is True
    assert resumed.tau == 89


def test_restore_advances_from_the_restored_rung() -> None:
    s = _mid_ladder()
    resumed = _sched()
    resumed.load_state_dict(s.state_dict())
    for _ in range(30):
        s.record(True)
        resumed.record(True)
    assert s.maybe_advance() is True
    assert resumed.maybe_advance() is True
    assert resumed.tau == s.tau
    assert resumed.snapshot()["advances"] == s.snapshot()["advances"]


def test_state_dict_is_a_detached_copy() -> None:
    s = _mid_ladder()
    state = s.state_dict()
    before = s.snapshot()
    state["window"].append(True)
    state["tau"] = 0
    state["advances"] = 999
    assert s.snapshot() == before


def test_loaded_state_is_not_aliased_by_the_caller() -> None:
    s = _mid_ladder()
    state = s.state_dict()
    resumed = _sched()
    resumed.load_state_dict(state)
    after = resumed.snapshot()
    state["window"].clear()
    state["window"].extend([False] * 5)
    assert resumed.snapshot() == after


def test_restored_window_is_rebound_to_the_current_trailing_length() -> None:
    # Retuning min_attempts between runs must take effect on resume: the
    # window is a trailing view, so it keeps the most recent attempts.
    s = _sched(min_attempts=30)
    for i in range(30):
        s.record(i >= 25)                 # last 5 are the successes
    resumed = _sched(min_attempts=5)
    resumed.load_state_dict(s.state_dict())
    assert resumed.attempts == 5 and resumed.successes == 5


def test_config_is_not_frozen_into_the_state() -> None:
    # Gates come from the config file, so a retuned run picks them up.
    s = _sched(advance_threshold=0.2, min_attempts=30, advance_entries=10)
    state = s.state_dict()
    assert not ({"advance_threshold", "min_attempts", "advance_entries",
                 "trailing"} & set(state))
    resumed = _sched(advance_threshold=0.9, min_attempts=5, advance_entries=3)
    resumed.load_state_dict(state)
    assert resumed.advance_threshold == 0.9
    assert resumed.min_attempts == 5
    assert resumed.advance_entries == 3


def test_load_rejects_a_state_from_a_different_tape() -> None:
    # Entry 400 of a 757-rung tape is not entry 400 of a re-minted one.
    s = _sched(n_entries=100)
    for _ in range(30):
        s.record(True)
    s.maybe_advance()
    other = _sched(n_entries=150)
    with pytest.raises(ValueError):
        other.load_state_dict(s.state_dict())


@pytest.mark.parametrize("bad", [
    {"tau": 100},                                     # off the end
    {"tau": -1},                                      # not a sentinel here
    {"advances": -1},
    {"entrance_attempts": 2, "entrance_successes": 5},
])
def test_load_rejects_nonsense(bad) -> None:
    s = _sched()
    with pytest.raises(ValueError):
        s.load_state_dict({**s.state_dict(), **bad})


def test_load_requires_a_tau() -> None:
    with pytest.raises(KeyError):
        _sched().load_state_dict({"n_entries": 100, "window": []})
    with pytest.raises(KeyError):
        _sched().load_state_dict(None)


def test_a_rejected_load_leaves_the_cursor_untouched() -> None:
    s = _mid_ladder()
    before = s.snapshot()
    with pytest.raises(ValueError):
        s.load_state_dict({**s.state_dict(), "tau": 5, "n_entries": 7})
    with pytest.raises(ValueError):
        s.load_state_dict({**s.state_dict(), "tau": 12, "advances": -4})
    assert s.snapshot() == before


def test_state_dict_carries_a_schema_tag() -> None:
    assert _sched().state_dict()["version"] == STATE_VERSION


# ---- the restart draw -----------------------------------------------


def test_draw_restart_splits_window_and_entrance_length_balanced() -> None:
    # 4 window entries + 1 entrance-worth of mass: the entrance owns the
    # top fifth of the unit interval, the window is uniform below it.
    assert draw_restart(4, 1.0, 0.0) == 0
    assert draw_restart(4, 1.0, 0.24) == 1
    assert draw_restart(4, 1.0, 0.55) == 2
    assert draw_restart(4, 1.0, 0.79) == 3
    assert draw_restart(4, 1.0, 0.81) == ENTRANCE
    assert draw_restart(4, 1.0, 0.999) == ENTRANCE


def test_draw_restart_entrance_share_is_fixed_by_weight() -> None:
    n = 40
    hits = sum(draw_restart(n, 1.0, i / 10000) == ENTRANCE
               for i in range(10000))
    assert hits / 10000 == pytest.approx(1 / (n + 1), abs=1e-3)


def test_draw_restart_with_zero_weight_never_picks_the_entrance() -> None:
    assert all(draw_restart(4, 0.0, u / 100) != ENTRANCE for u in range(100))


def test_draw_restart_with_an_empty_window_is_the_entrance() -> None:
    assert draw_restart(0, 1.0, 0.0) == ENTRANCE
    assert draw_restart(0, 0.0, 0.5) == ENTRANCE


def test_draw_restart_never_runs_off_the_window() -> None:
    # rand is documented as [0, 1); guard the boundary anyway.
    assert draw_restart(3, 0.0, 1.0 - 1e-12) == 2


@pytest.mark.parametrize("kw", [(-1, 1.0), (3, -0.5)])
def test_draw_restart_rejects_nonsense(kw) -> None:
    with pytest.raises(ValueError):
        draw_restart(kw[0], kw[1], 0.5)


# ---- composition with the window the trainer actually uses ----------


def test_window_behind_tau_is_the_trailer_the_trainer_draws_from() -> None:
    tape = list(range(200))
    win = start_window(tape, 150, window_frames=160, frames_per_step=4)
    assert win == list(range(110, 151))     # 40 entries behind + the anchor
    # Every draw indexes a real tape entry.
    picks = {draw_restart(len(win), 1.0, u / 1000) for u in range(1000)}
    assert picks <= set(range(len(win))) | {ENTRANCE}


def test_window_at_tau_zero_collapses_to_the_entrance_region() -> None:
    tape = list(range(200))
    assert start_window(tape, 0, window_frames=160, frames_per_step=4) == [0]


def test_restart_site_composition_walks_the_cursor_home() -> None:
    """The exact sequence the trainer's restart branch performs.

    scheduler -> start_window over tape INDICES -> length-balanced draw ->
    blob lookup. Run with an oracle that succeeds just often enough to
    clear the threshold, and the cursor must reach the entrance without
    ever producing an out-of-range index.
    """
    n = 903                                   # the banked 1-1 tape length
    tape = list(range(n))
    sched = TauScheduler(n, tau_init=-1, advance_entries=40,
                         advance_threshold=0.2, min_attempts=30)
    seen_entrance = 0
    seen_window = 0
    for k in range(40_000):
        win = start_window(tape, sched.tau, window_frames=160,
                           frames_per_step=4)
        pick = draw_restart(len(win), 1.0, (k * 0.37) % 1.0)
        if pick == ENTRANCE:
            seen_entrance += 1
            sched.record_entrance(False)
        else:
            assert 0 <= win[pick] <= sched.tau
            seen_window += 1
            sched.record(k % 4 == 0, tau=sched.tau)   # 0.25 > 0.20
        sched.maybe_advance()
        if sched.at_entrance:
            break
    assert sched.at_entrance
    assert sched.snapshot()["advances"] == 23         # ceil(902 / 40)
    assert seen_entrance > 0 and seen_window > 0


# ---- the minted tapes (skipped when they have not been minted) ------


@pytest.mark.parametrize("level", ["1-1", "1-2"])
def test_backward_config_wires_the_curriculum_and_nothing_else(level) -> None:
    cfg = yaml.safe_load(
        (REPO / "configs" / f"mario_{level.replace('-', '_')}_backward.yaml")
        .read_text()
    )
    rl = cfg["reinforce"]
    blk = rl["backward_curriculum"]
    assert blk["states_dir"] == f"checkpoints/backward_states/{level}"
    assert blk["tau_init"] == -1
    assert blk["window_frames"] == 160
    assert blk["min_attempts"] >= 30       # 3-episode gates are mirages
    assert blk["pin_entrance"] is False    # the guard knob is not a run knob
    assert rl["sticky_action_prob"] == 0.25
    # Every rung restart is an episode boundary. Off, the sticky roll on
    # the fresh attempt's first step replays the action the previous life
    # died holding — straight into the rate the advance gate consumes.
    assert rl["sticky_episode_boundary_reset"] is True
    assert rl["smb_curriculum"] is False   # the stage-0 restart path
    assert rl["async_pipeline"] is False   # breaks frame-perfect platformers
    assert rl["encoder"].startswith("smb_tiles")
    # The other claimants on the same restart site stay off.
    assert "cgsa" not in rl
    assert rl.get("go_explore", {}).get("enabled", False) is False
    assert rl.get("wavefront_reward", {}).get("enabled", False) is False
    assert (REPO / cfg["start_state_path"]).exists()


@pytest.mark.parametrize("level", ["1-1", "1-2"])
def test_backward_config_start_state_is_the_tape_root(level) -> None:
    # The entrance rides in the per-episode draw, so a start state from a
    # different lineage than the tape trains on a two-lineage mixture.
    states_dir = REPO / "checkpoints" / "backward_states" / level
    if not (states_dir / INDEX_NAME).exists():
        pytest.skip(f"{level} not minted (scripts/mint_backward_states.py)")
    _, meta = load_index(states_dir)
    cfg = yaml.safe_load(
        (REPO / "configs" / f"mario_{level.replace('-', '_')}_backward.yaml")
        .read_text()
    )
    assert (REPO / cfg["start_state_path"]).resolve() == \
        Path(meta["root_state"]).resolve()


@pytest.mark.parametrize("level", ["1-1", "1-2"])
def test_minted_tape_index_passes_its_own_gate(level) -> None:
    states_dir = REPO / "checkpoints" / "backward_states" / level
    if not (states_dir / INDEX_NAME).exists():
        pytest.skip(f"{level} not minted (scripts/mint_backward_states.py)")
    entries, meta = load_index(states_dir)
    assert len(entries) == len(set(e.step for e in entries))
    assert meta["reached_clear"] is True
    assert meta["gx"]["monotone"] is True
    assert gx_report(entries)["monotone"] is True
    # Frames are the tape's own clock: step i starts at i * frame_skip.
    fs = int(meta["frame_skip"])
    assert all(e.frame == e.step * fs for e in entries)


def test_no_backward_config_trains_sticky_without_the_boundary_guard() -> None:
    """Repo-wide invariant, not a two-config spot check.

    `StickyBoundary.mark_restart` is the only site that zeroes the carried
    action, and it early-returns unless `sticky_episode_boundary_reset` is
    set. So a backward config with sticky > 0 and the flag unset makes
    `_sticky_restart` a no-op at every rung restart: the fresh attempt's
    first step re-executes the input the previous life died holding, and
    the contamination lands in the success rate the advance gate reads.
    Any future backward config inherits this check.
    """
    offenders = []
    for path in sorted((REPO / "configs").glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        rl = (cfg or {}).get("reinforce") or {}
        if not isinstance(rl, dict) or not rl.get("backward_curriculum"):
            continue
        if float(rl.get("sticky_action_prob", 0.0)) <= 0.0:
            continue
        if not rl.get("sticky_episode_boundary_reset", False):
            offenders.append(path.name)
    assert not offenders, (
        "backward configs train sticky with the episode-boundary guard "
        f"off (dead life's held action leaks into each rung): {offenders}"
    )
