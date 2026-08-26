"""Confluence clear-detector v2 — the three fabricated-clear failure modes.

Every case here is a regression test for a MEASURED false positive, replayed
from the 2026-08-06 finding (memory: project_confluence_detector_unsafe):

  * Gradius       fired at 205 actions on a real DEATH (lives 3->2, progress
                  reset to 0): a shmup respawn wipes entity slots and resets
                  position exactly like a stage load.
  * Double Dragon fired at 27 actions on a transient combat RAM blip
                  (progress 72 -> 846 -> 88 within 5 steps, no advance, no
                  death).
  * Kirby         fired 3x in a 24 s smoke run, every time on an ORDINARY
                  ROOM transition (area 58 -> 62), never on a stage clear.

The signatures are synthesized as duck-typed RAM streams with the same SHAPE
the receipts describe and pushed through the REAL detector and the REAL
GenericGame.is_clear — nothing here stubs the thing under test. Each stream is
first asserted to reproduce the false fire with the v2 knobs OFF (otherwise the
regression test would be vacuous), then asserted to be suppressed with them ON.

Also covered: Solver.observe()'s dead-before-clear ordering, and the
replay-verification gate on the banking path (happy + mismatch).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import yaml

from scripts.clear_detect import StreamingConfluenceDetector, trailing_median
from scripts.clear_reachability import clear_reachability
from scripts.go_explore_solve import GenericGame, Solver

#: Top-level profile directory, for the shipped-profile assertions below.
CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# --------------------------------------------------------------------------
# A synthetic 2 KiB RAM layout. Addresses are arbitrary test fixtures — the
# point is the SHAPE of the stream, not any real game's map.
# --------------------------------------------------------------------------
A_GX_LO, A_GX_HI = 0x10, 0x11
A_LIVES = 0x20
A_Y = 0x30
A_ROOM = 0x40
A_TIMER, A_SCORE = 0x50, 0x51
ENTITY_LO, ENTITY_HI = 0x60, 0x70          # 16 contiguous "entity slots"


def _ram(gx: int, lives: int = 3, room: int = 1, timer: int = 200,
         score: int = 0, entities: bool = True, y: int = 100) -> bytes:
    buf = bytearray(2048)
    buf[A_GX_LO] = gx & 0xFF
    buf[A_GX_HI] = (gx >> 8) & 0xFF
    buf[A_LIVES] = lives
    buf[A_Y] = y
    buf[A_ROOM] = room
    buf[A_TIMER] = timer
    buf[A_SCORE] = score
    for a in range(ENTITY_LO, ENTITY_HI):
        buf[a] = 9 if entities else 0
    return bytes(buf)


def _tally(i: int) -> dict:
    """A timer->score conversion cadence running throughout every stream.

    Holding the `tally` signal high in ALL streams isolates each test on the
    `coord` signal, which is the one every recorded false positive rode in on
    (min_signals defaults to 2 = both must agree, so coord is the swing vote).
    Cadence 1/8 steps keeps both bytes inside 0..255 for the longest stream
    here (1200 samples), so neither wraps and creates a spurious delta.
    """
    return {"timer": 250 - i // 8, "score": i // 8}


def _game(**clear) -> GenericGame:
    # GenericGame.__init__ only reads the profile dict (no ROM/Pool I/O).
    return GenericGame({
        "solve": {
            "rom": "roms/does-not-need-to-exist.nes",
            "progress": {"lo": A_GX_LO, "hi": A_GX_HI},
            "y": A_Y,
            "area": A_ROOM,
            "level_key": [],
            "lives": A_LIVES,
            "clear": dict({"mode": "confluence"}, **clear),
        }
    })


class _NullArchive:
    """Records nothing; observe() only needs `.cells` and `.record`."""
    cells: dict = {}

    def record(self, ram, blob, score, steps, key=None) -> bool:
        return False


class _NullPool:
    def save_worker_state(self, wid):
        return None


def _observe_driver(game, start_lives: int = 3) -> SimpleNamespace:
    """Duck-typed Solver carrying only what observe() reads."""
    fake = SimpleNamespace(
        game=game, start_wd=game.level_key(bytes(2048)), start_lives=start_lives,
        archive=_NullArchive(), pool=_NullPool(), traces={},
        max_area=0, max_gx_in_area={}, max_sect=0, _pin_time=0.0,
        ortho_mode="off", door_weight=0.0, time_bins=False, kill_key=False,
        # gate-opener arm at its shipped default: observe() pays one
        # string compare and nothing else.
        gate_mode="off",
        _recorded_new=False, banked=[],
    )
    fake._dump_solution = lambda *a, **k: (fake.banked.append(a) or True)
    return fake


def _run(game: GenericGame, stream) -> int | None:
    """Drive a RAM stream through Solver.observe() exactly the way explore()
    does — including a FRESH ctx whenever the lineage ends, which is what
    _assign() hands the next burst. Returns the index of the first banked
    clear, or None.

    Going through observe() rather than calling is_clear() directly is the
    point: the dead-before-clear ordering and the per-burst ctx lifetime are
    half of what makes the vetoes work, and a test that bypassed them would
    pass while the solver still fabricated wins."""
    fake = _observe_driver(game)
    ctx: dict = {}
    for i, ram in enumerate(stream):
        status = Solver.observe(fake, 0, ram, [0], i, "entrance", ctx=ctx)
        if status == "clear":
            return i
        if status == "dead":
            ctx = {}          # lineage abandoned; the next burst starts clean
    return None


# --------------------------------------------------------------------------
# The three recorded streams
# --------------------------------------------------------------------------

def _gradius_death_stream(n: int = 400, at: int = 200) -> list[bytes]:
    """Scrolling shmup: position climbs, then a death — lives 3->2, position
    reset to 0, entity slots wiped — then a respawned climb."""
    out = []
    for i in range(n):
        t = _tally(i)
        if i < at:
            out.append(_ram(gx=400 + 3 * i, lives=3, **t))
        else:
            out.append(_ram(gx=4 * (i - at), lives=2, entities=False, **t))
    return out


def _double_dragon_blip_stream(n: int = 400, at: int = 30) -> list[bytes]:
    """Beat-'em-up: a 2-sample combat RAM spike (72 -> 846) that collapses
    straight back to the pre-spike neighborhood (88) and stays there. No
    advance, no death, no room change. Spike placed on a coord sub-window
    boundary — the alignment that makes it fire eight checks in a row."""
    out = []
    for i in range(n):
        t = _tally(i)
        if i < at:
            gx, ent = 72, True
        elif i < at + 2:
            gx, ent = 846, True
        else:
            gx, ent = 88 + (i - at) // 8, False
        out.append(_ram(gx=gx, entities=ent, **t))
    return out


def _kirby_room_stream(n: int = 400, at: int = 200) -> list[bytes]:
    """Room-based platformer: an ORDINARY room transition. The room byte steps
    58 -> 62, world-x resets from ~900 to ~20 and climbs again from there,
    entity slots wipe. Lives never move. Every ingredient of a stage clear
    except the stage actually being over."""
    out = []
    for i in range(n):
        t = _tally(i)
        if i < at:
            out.append(_ram(gx=600 + 2 * i, room=58, **t))
        else:
            out.append(_ram(gx=20 + 2 * (i - at), room=62,
                            entities=False, **t))
    return out


# --------------------------------------------------------------------------
# Failure mode 1 — Gradius: a death read as a win
# --------------------------------------------------------------------------

def test_gradius_death_stream_still_trips_the_raw_detector() -> None:
    # Not vacuous: the raw two-signal confluence really does fire on this
    # shape. That is why the veto has to exist at all.
    det = StreamingConfluenceDetector(
        lambda r: int(r[A_GX_LO]) | int(r[A_GX_HI]) << 8)
    assert any(det.push(r) for r in _gradius_death_stream())


def test_gradius_death_is_never_banked_as_a_clear() -> None:
    # Two mechanisms have to hold for this: observe() resolves is_dead first
    # (so the death step ends the lineage before the hook is consulted) and
    # the fresh ctx that follows denies the next burst the pre-death samples
    # the coord signature was built from.
    assert _run(_game(), _gradius_death_stream()) is None


def test_the_lives_drop_veto_holds_even_without_the_ordering_fix() -> None:
    # Belt-and-braces for any caller that consults the hook on its own: on the
    # exact step lives drop, is_clear refuses regardless of the detector.
    game = _game()
    ctx = {"_clear_det": SimpleNamespace(push=lambda ram: True)}
    game.is_clear((), _ram(gx=10, lives=3), ctx)
    assert game.is_clear((), _ram(gx=10, lives=2), ctx) is False


def test_a_clear_that_costs_no_life_is_not_suppressed_by_the_veto() -> None:
    # The veto must not turn the hook into a permanent no-op: the same
    # position-reset + entity-wipe shape WITHOUT a life lost still fires.
    stream = [_ram(gx=(400 + 3 * i if i < 200 else 4 * (i - 200)),
                   lives=3, entities=i < 200, **_tally(i))
              for i in range(400)]
    assert _run(_game(), stream) is not None


# --------------------------------------------------------------------------
# Failure mode 2 — Double Dragon: a transient combat blip read as a win
# --------------------------------------------------------------------------

def test_double_dragon_blip_fires_with_the_v2_knobs_off() -> None:
    # Pins the pre-fix behavior (and the default-identical claim): with no
    # knobs set, the shipped detector still fabricates this clear.
    assert _run(_game(), _double_dragon_blip_stream()) is not None


def test_persistence_alone_does_not_defeat_an_aligned_blip() -> None:
    # The honest negative result that motivates progress_median: a spike
    # parked on a sub-window boundary reproduces the coord signature at eight
    # consecutive checks, so N=2..8 buys nothing. Recorded as a test so nobody
    # "simplifies" the median filter away believing persistence covers it.
    assert _run(_game(persist_checks=4), _double_dragon_blip_stream()) is not None


def _streak(stream, **kw) -> tuple[int, int, int]:
    """Longest consecutive-vote run the detector sees over a stream, with an
    unreachable persist bar so nothing latches early."""
    det = StreamingConfluenceDetector(
        lambda r: int(r[A_GX_LO]) | int(r[A_GX_HI]) << 8,
        persist_checks=10 ** 9, **kw)
    best = 0
    for ram in stream:
        det.push(ram)
        best = max(best, det._streak)
    return best, det.n_checks, det.n_votes


def test_persist_checks_counts_consecutive_confluence_checks() -> None:
    # Pins the streak counter itself, on the exact arithmetic worked out in
    # the detector docstring: an aligned 2-sample spike votes at 8 consecutive
    # checks, so N=8 still latches and N=9 does not.
    blip = _double_dragon_blip_stream(at=30)
    assert _streak(blip)[0] == 8
    assert _run(_game(persist_checks=8), blip) is not None
    assert _run(_game(persist_checks=9), blip) is None


def test_the_persistence_margin_over_a_real_load_is_only_two_checks() -> None:
    # WHY the median filter is the primary mechanism and persistence only the
    # secondary one: a genuine sustained load votes 10 consecutive checks
    # against the aligned blip's 8. The N that kills the blip (9) leaves one
    # single check of headroom on a real clear — not a margin anyone should
    # bet a banked receipt on.
    assert _streak(_kirby_room_stream())[0] == 10
    assert _streak(_double_dragon_blip_stream(at=30))[0] == 8


def test_persist_checks_delays_a_real_load_rather_than_blocking_it() -> None:
    load = _kirby_room_stream()
    early = _run(_game(), load)
    late = _run(_game(persist_checks=4), load)
    assert early is not None and late is not None and late > early


def _two_loads_stream(n: int = 1100, a: int = 200, b: int = 700,
                      quiet: int = 60) -> list[bytes]:
    """Two separate room loads with an ordinary stretch of play between them:
    two runs of votes separated by checks that do not vote."""
    out = []
    for i in range(n):
        t = _tally(i)
        if i < a:
            out.append(_ram(gx=600 + 2 * i, room=1, **t))
        elif i < a + quiet:
            out.append(_ram(gx=20, room=2, entities=False, **t))
        elif i < b:
            out.append(_ram(gx=20 + 2 * (i - a - quiet), room=2, **t))
        elif i < b + quiet:
            out.append(_ram(gx=20, room=3, entities=False, **t))
        else:
            out.append(_ram(gx=20 + 2 * (i - b - quiet), room=3, **t))
    return out


def test_the_streak_is_CONSECUTIVE_and_not_a_running_total() -> None:
    # Two 11-vote runs with quiet checks between them. persist_checks=12 is
    # above either run but below their sum: a counter that never reset would
    # latch on the total and re-open the same fabricated-clear hole from the
    # other side.
    assert _streak(_two_loads_stream())[0] == 11
    assert _streak(_two_loads_stream())[2] == 22          # 22 votes overall
    assert _run(_game(persist_checks=12), _two_loads_stream()) is None
    assert _run(_game(persist_checks=11), _two_loads_stream()) is not None


def test_an_unaligned_blip_never_votes_at_all() -> None:
    # Alignment is the whole story: shift the same spike by one sample and the
    # shipped detector is silent. That is why "it didn't fire in a smoke test"
    # was never evidence of safety.
    assert _streak(_double_dragon_blip_stream(at=31))[2] == 0


def test_the_median_filter_kills_the_two_sample_blip() -> None:
    assert _run(_game(progress_median=5), _double_dragon_blip_stream()) is None


@pytest.mark.parametrize("at", [30, 31, 32, 37, 45, 53])
def test_the_blip_is_suppressed_at_every_window_alignment(at: int) -> None:
    # Alignment-independence is the whole reason the filter beats persistence.
    stream = _double_dragon_blip_stream(at=at)
    assert _run(_game(progress_median=5), stream) is None


def test_the_median_filter_leaves_a_real_sustained_load_detectable() -> None:
    # The filter must cost recall on impulses only. A genuine load is a STEP,
    # which a median filter preserves (delayed by k//2 samples).
    stream = [_ram(gx=(600 + 2 * i if i < 200 else 20 + 2 * (i - 200)),
                   entities=i < 200, **_tally(i))
              for i in range(400)]
    assert _run(_game(progress_median=5), stream) is not None


def test_trailing_median_preserves_a_step_and_erases_an_impulse() -> None:
    x = np.array([10] * 10 + [900, 900] + [10] * 10, dtype=np.int64)
    assert trailing_median(x, 5).max() == 10          # impulse gone
    step = np.array([900] * 10 + [10] * 10, dtype=np.int64)
    out = trailing_median(step, 5)
    assert out[0] == 900 and out[-1] == 10            # step survives
    # k <= 1 is the identity, so the default path is byte-identical.
    assert np.array_equal(trailing_median(x, 1), x)


# --------------------------------------------------------------------------
# The median filter's LEADING EDGE — fixed 2026-08-26
#
# The impulse tests above all park the spike in the MIDDLE of the series,
# where every output position has a full k samples of real history behind
# it. The leading k-1 positions do not, and what stands in for that history
# decides whether the filter suppresses a head-of-window impulse or replays
# it. The shipped fill was x[0], i.e. the oldest sample repeated k-1 times —
# so whenever the oldest sample WAS the impulse, the filter made it the
# majority of its own window.
#
# This matters because StreamingConfluenceDetector filters a ROLLING window:
# every sample becomes x[0] eventually, and a check lands on a spike sitting
# at relative index 0 whenever the spike's absolute index is a multiple of
# the check stride.
# --------------------------------------------------------------------------

def _growing_trailing_median(x: np.ndarray, k: int) -> np.ndarray:
    """The "just delete the pad" alternative: for the leading positions, take
    the median of however many real samples exist so far. Written out here so
    the tests below can show it is NOT the fix — it is measured, not assumed.
    """
    return np.array([np.median(x[max(0, i - k + 1):i + 1]) for i in range(x.size)])


def test_a_head_of_series_impulse_is_not_replicated_by_the_pad() -> None:
    # 2-sample impulse at the HEAD. With an x[0] back-fill the pad is
    # [846]*(k-1), so 846 is the majority of every one of the first ~k/2
    # windows: at k=15 the first NINE outputs read 846. Raising k — turning
    # the suppression knob UP — made the surviving run LONGER.
    x = np.array([846, 846] + [88] * 40, dtype=np.int64)
    for k in (5, 9, 15):
        assert trailing_median(x, k).max() == 88, f"impulse survives at k={k}"


def test_the_leading_edge_is_decided_by_k_real_samples_not_by_one() -> None:
    # The distinguishing case between the two candidate fixes. Dropping the
    # pad (a growing trailing window) still leaves out[0] == x[0]: one real
    # sample decides that position however you dress it. Only seeding from a
    # full k-sample window removes it.
    x = np.array([846, 846] + [88] * 40, dtype=np.int64)
    assert _growing_trailing_median(x, 15)[0] == 846      # not a fix
    assert trailing_median(x, 15)[0] == 88                # is a fix


def test_a_head_of_window_spike_no_longer_fabricates_a_clear() -> None:
    # The detector-level consequence, and the reason the pure-function tests
    # above are not academic. window=240 / stride=20: a spike whose absolute
    # index is a multiple of the stride sits at relative index 0 of the full
    # rolling window at some check. Before the fix this fired at step 539
    # while the SAME spike one sample over (301) was silent — so the
    # "alignment-independent" claim held at every alignment except the one
    # the filter's own edge handling created.
    for at in (300, 320, 340):
        assert _run(_game(progress_median=5),
                    _double_dragon_blip_stream(n=700, at=at)) is None


def test_the_leading_edge_fix_costs_no_recall_on_a_real_load() -> None:
    # What the fix must NOT do: a genuine sustained load is still detected,
    # at the same step it was before (219). Suppression bought with recall
    # would be the wrong trade — the census already has enough silence in it.
    stream = [_ram(gx=(600 + 2 * i if i < 200 else 20 + 2 * (i - 200)),
                   entities=i < 200, **_tally(i))
              for i in range(400)]
    assert _run(_game(progress_median=5), stream) == 219


# --------------------------------------------------------------------------
# Failure mode 3 — Kirby: an ordinary room transition read as a win
# --------------------------------------------------------------------------

def test_kirby_room_transition_fires_with_the_veto_disarmed() -> None:
    assert _run(_game(), _kirby_room_stream()) is not None


def test_kirby_room_transition_is_vetoed_when_progress_did_not_advance() -> None:
    assert _run(_game(room_veto={"steps": 120}), _kirby_room_stream()) is None


def test_the_room_veto_reads_explicitly_named_bytes_when_given() -> None:
    # Same suppression via `addrs` instead of the adapter's room identity.
    assert _run(_game(room_veto={"steps": 120, "addrs": [A_ROOM]}),
                _kirby_room_stream()) is None


def _advancing_room_stream(n: int = 400, at: int = 200) -> list[bytes]:
    """A room change whose progress observable DOES go on to beat where it
    stood before the transition, inside the veto window."""
    out = []
    for i in range(n):
        t = _tally(i)
        if i < at:
            out.append(_ram(gx=600 + 2 * i, room=58, **t))
        elif i < at + 60:
            out.append(_ram(gx=20, room=62, entities=False, **t))
        else:
            out.append(_ram(gx=20 + 60 * (i - at - 60), room=62,
                            entities=False, **t))
    return out


def test_a_room_change_that_ADVANCES_progress_is_not_vetoed() -> None:
    # The veto's escape clause, and the reason HOLD keeps the evidence rather
    # than discarding it on sight: progress crosses its pre-transition high
    # inside the window, so the transition is exonerated and the fire stands.
    assert _run(_game(room_veto={"steps": 200}),
                _advancing_room_stream()) is not None


def test_the_same_advancing_stream_is_vetoed_once_the_window_is_too_short() -> None:
    # Same stream, window shorter than the advance takes: the evidence is
    # discarded before exoneration arrives. The knob really is a recall dial.
    assert _run(_game(room_veto={"steps": 20}),
                _advancing_room_stream()) is None


def test_a_fire_with_no_room_change_at_all_survives_the_veto() -> None:
    stream = [_ram(gx=(400 + 3 * i if i < 200 else 4 * (i - 200)),
                   room=7, entities=i < 200, **_tally(i))
              for i in range(400)]
    assert _run(_game(room_veto={"steps": 120}), stream) is not None


def test_a_vetoed_fire_discards_its_evidence_instead_of_re_firing() -> None:
    # THE subtle failure: suppressing the return value alone is a no-op,
    # because the samples that produced the fire stay in the rolling window
    # and land again at the next check once the veto window expires. Run the
    # Kirby stream three times past the veto's own horizon and require
    # silence throughout.
    assert _run(_game(room_veto={"steps": 30}),
                _kirby_room_stream(n=1200, at=200)) is None


def test_the_room_veto_state_machine_walks_hold_then_discard() -> None:
    game = _game(room_veto={"steps": 3})
    ctx: dict = {}
    assert game.room_veto_step(_ram(gx=900, room=1), ctx) == game.RV_NONE
    # The change step itself is observation 0 of the window, so steps=3 buys
    # four HOLDs before the window expires.
    assert game.room_veto_step(_ram(gx=20, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=22, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=24, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=26, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=28, room=2), ctx) == game.RV_DISCARD
    # Exactly once per change: the step after must be quiet again.
    assert game.room_veto_step(_ram(gx=30, room=2), ctx) == game.RV_NONE


def test_the_room_veto_releases_the_moment_progress_beats_the_old_room() -> None:
    game = _game(room_veto={"steps": 50})
    ctx: dict = {}
    game.room_veto_step(_ram(gx=900, room=1), ctx)
    assert game.room_veto_step(_ram(gx=20, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=901, room=2), ctx) == game.RV_NONE
    # Released for good — it must not re-arm on its own next step.
    assert game.room_veto_step(_ram(gx=30, room=2), ctx) == game.RV_NONE


# --------------------------------------------------------------------------
# The room veto vs. transition-frame GARBAGE progress reads
#
# A room load is exactly when the page byte reads huge. The escape clause runs
# on the arming frame too, so an unguarded read let ONE garbage sample satisfy
# "progress advanced" in the same call that armed the veto -- permanently
# disarming it and re-banking the Kirby false positive. Replay verification
# cannot catch this: the emulator is deterministic, so the replay reproduces
# the same garbage read and the same fabricated verdict.
# --------------------------------------------------------------------------

GARBAGE_GX = 0xF000        # 61440, far above GenericGame's 30000 default cap


@pytest.mark.parametrize("idx", [199, 200, 201, 205, 240, 319])
def test_one_garbage_progress_read_cannot_disarm_the_room_veto(idx) -> None:
    stream = _kirby_room_stream()
    stream[idx] = _ram(gx=GARBAGE_GX, room=62 if idx >= 200 else 58,
                       entities=idx >= 200, **_tally(idx))
    assert _run(_game(room_veto={"steps": 120}), stream) is None


def test_the_unguarded_read_really_did_bank_the_false_positive() -> None:
    # Not vacuous: without the cap guard this exact stream banks a clear. The
    # guard is one comparison, so the regression is re-derived here rather
    # than mocked -- a hand-rolled step over the same frames with the cap
    # raised out of the way reproduces the disarm.
    game = _game(room_veto={"steps": 120})
    game.progress_cap = 10 ** 9              # simulate the missing guard
    ctx: dict = {}
    stream = _kirby_room_stream()
    stream[200] = _ram(gx=GARBAGE_GX, room=62, entities=False, **_tally(200))
    verdicts = [game.room_veto_step(r, ctx) for r in stream[:260]]
    assert game.RV_HOLD not in verdicts[201:], (
        "with the guard removed the veto must be disarmed, or the guarded "
        "test above proves nothing")


def test_a_burst_of_garbage_across_the_transition_is_still_vetoed() -> None:
    stream = _kirby_room_stream()
    for idx in range(199, 206):
        stream[idx] = _ram(gx=GARBAGE_GX, room=62 if idx >= 200 else 58,
                           entities=idx < 200, **_tally(idx))
    assert _run(_game(room_veto={"steps": 120}), stream) is None


def test_garbage_never_becomes_the_baseline_a_later_escape_is_measured_from() -> None:
    # The stored prev-progress must be the last GOOD sample. If garbage were
    # kept, the baseline would be astronomically high and no real advance
    # could ever escape the veto -- the mirror-image bug (recall, not
    # precision), and just as wrong.
    game = _game(room_veto={"steps": 50})
    ctx: dict = {}
    game.room_veto_step(_ram(gx=900, room=1), ctx)
    game.room_veto_step(_ram(gx=GARBAGE_GX, room=1), ctx)   # garbage, no change
    assert ctx["_rv_prog_prev"] == 900
    assert game.room_veto_step(_ram(gx=20, room=2), ctx) == game.RV_HOLD
    assert ctx["_rv_prog_before"] == 900       # not 61440, not 20
    assert game.room_veto_step(_ram(gx=901, room=2), ctx) == game.RV_NONE


def test_a_garbage_read_does_not_satisfy_the_escape_on_the_arming_frame() -> None:
    # The precise reported shape: the transition frame itself reads garbage,
    # and the escape clause runs in that same call.
    game = _game(room_veto={"steps": 50})
    ctx: dict = {}
    game.room_veto_step(_ram(gx=900, room=1), ctx)
    assert game.room_veto_step(_ram(gx=GARBAGE_GX, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=20, room=2), ctx) == game.RV_HOLD


def test_a_veto_armed_with_no_usable_baseline_fails_safe() -> None:
    # Both the pre-transition and the transition sample are garbage, so there
    # is no baseline to measure an advance from. Precision-over-recall: the
    # veto holds and then discards rather than releasing on an unknown.
    game = _game(room_veto={"steps": 2})
    ctx: dict = {}
    game.room_veto_step(_ram(gx=GARBAGE_GX, room=1), ctx)
    # The change step is observation 0 of the window, so steps=2 buys three
    # HOLDs before it expires.
    assert game.room_veto_step(_ram(gx=GARBAGE_GX, room=2), ctx) == game.RV_HOLD
    assert ctx["_rv_prog_before"] is None
    assert game.room_veto_step(_ram(gx=GARBAGE_GX, room=2), ctx) == game.RV_HOLD
    # A GOOD sample now arrives, but there is still nothing to compare it
    # against. Treating the missing baseline as 0 would read any positive
    # position as "progress advanced" and release the veto on the spot.
    assert game.room_veto_step(_ram(gx=20, room=2), ctx) == game.RV_HOLD
    assert game.room_veto_step(_ram(gx=GARBAGE_GX, room=2), ctx) == game.RV_DISCARD
    # Self-clearing, not a permanent mute: the window closed, so the next
    # room change gets a fresh (and by now well-defined) baseline.
    assert game.room_veto_step(_ram(gx=25, room=2), ctx) == game.RV_NONE
    assert game.room_veto_step(_ram(gx=5, room=3), ctx) == game.RV_HOLD
    assert ctx["_rv_prog_before"] == 25


def test_a_real_advance_still_escapes_after_a_garbage_read() -> None:
    # The guard must not cost recall: an ordinary in-cap advance following a
    # garbage frame still exonerates the transition.
    stream = _advancing_room_stream()
    stream[200] = _ram(gx=GARBAGE_GX, room=62, entities=False, **_tally(200))
    assert _run(_game(room_veto={"steps": 200}), stream) is not None


def test_reset_clears_both_the_latch_and_the_window() -> None:
    det = StreamingConfluenceDetector(
        lambda r: int(r[A_GX_LO]) | int(r[A_GX_HI]) << 8)
    stream = _kirby_room_stream()
    fired = next(i for i, r in enumerate(stream) if det.push(r))
    assert det.push(stream[fired]) is True            # latched
    det.reset()
    assert det.push(stream[fired]) is False           # un-latched
    assert len(det._ram) == 1                         # window emptied too


# --------------------------------------------------------------------------
# Default-identity: every v2 knob is inert unless the profile sets it
# --------------------------------------------------------------------------

def test_the_v2_defaults_reproduce_the_shipped_detector_exactly() -> None:
    d = StreamingConfluenceDetector(lambda r: 0)
    assert (d.persist_checks, d.progress_median) == (1, 1)


def test_an_unconfigured_profile_arms_none_of_the_v2_knobs() -> None:
    g = _game()
    assert (g._conf_persist, g._conf_median, g._rv_steps) == (None, None, 0)


def test_the_shipped_confluence_profiles_stay_on_the_default_path() -> None:
    # Whichever profiles ship `clear: {mode: confluence}` must not change
    # behavior under a v2 binary — the knobs are opt-in per profile.
    #
    # DISCOVERED, NOT HARDCODED (2026-08-26). This test named contra.yaml
    # and gradius.yaml literally and asserted each still declared the hook.
    # That is the census pattern in miniature: it checked that a hook was
    # DECLARED and never that it could FIRE, so it stayed green for the
    # eighteen days Gradius's hook was inert after its progress was swapped
    # to the odometer (commit 09299fa), and it would have gone red for the
    # correct withdrawal of that dead hook rather than for the defect.
    # Globbing keeps it honest as profiles come and go; the reachability
    # assertion is the part the old form was missing.
    found = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        prof = yaml.safe_load(path.read_text())
        if not isinstance(prof, dict):
            continue
        cl = (prof.get("solve") or {}).get("clear") or {}
        if cl.get("mode") == "confluence":
            found[path.name] = (prof, cl)

    assert found, ("no profile ships clear: {mode: confluence} — if the hook "
                   "was retired, retire this test with it rather than "
                   "leaving it to pass vacuously")
    for name, (prof, cl) in found.items():
        for knob in ("persist_checks", "progress_median", "room_veto"):
            assert knob not in cl, f"{name} silently opted into {knob}"
        r = clear_reachability(prof)
        assert r.ok, f"{name} declares a hook that cannot fire: {r.reason}"


# --------------------------------------------------------------------------
# observe(): death is resolved before any clear
# --------------------------------------------------------------------------

class _AlwaysClear:
    """Duck-typed game whose clear hook fires unconditionally — exactly the
    fabricating detector the reorder exists to contain."""

    progress_cap = 10 ** 9

    def __init__(self, dead: bool = False, finale: bool = False) -> None:
        self._dead, self._finale = dead, finale
        self.clear_calls = 0

    def is_dead(self, ram, start_lives) -> bool:
        return self._dead

    def is_finale(self, start_key, ram) -> bool:
        return self._finale

    def is_clear(self, start_key, ram, ctx=None) -> bool:
        self.clear_calls += 1
        return True

    def level_key(self, ram) -> tuple:
        return (0,)

    def progress(self, ram) -> int:
        return 0

    def area(self, ram) -> int:
        return 0

    def y(self, ram) -> int:
        return 0

    def cell_fn(self, ram) -> tuple:
        return (0, 0, 0, 0, 0)

    def score_bonus(self, ram) -> int:
        return 0

    def room_id(self, ram) -> tuple:
        return (0,)


def _observe_solver(game) -> SimpleNamespace:
    banked: list = []
    fake = SimpleNamespace(
        game=game, start_wd=(0,), start_lives=3,
        _dumped=banked,
        ortho_mode="off", door_weight=0.0, time_bins=False, kill_key=False,
        # gate-opener arm at its shipped default: observe() pays one
        # string compare and nothing else.
        gate_mode="off",
        max_area=0, max_gx_in_area={}, max_sect=0, archive=None,
    )
    fake._dump_solution = lambda *a, **k: (banked.append(a) or True)
    return fake


def test_a_step_that_is_both_dead_and_clear_resolves_dead() -> None:
    # THE structural bug (2026-08-06): is_clear ran first, so every detector
    # false positive won the race unconditionally and a real death was banked
    # as a win. is_clear must not even be consulted.
    game = _AlwaysClear(dead=True)
    fake = _observe_solver(game)
    assert Solver.observe(fake, 0, bytearray(2048), [1], 1, "entrance") == "dead"
    assert fake._dumped == []
    assert game.clear_calls == 0


def test_a_step_that_is_both_dead_and_FINALE_resolves_dead() -> None:
    # The finale hook used to precede is_dead too; same reasoning applies.
    game = _AlwaysClear(dead=True, finale=True)
    fake = _observe_solver(game)
    assert Solver.observe(fake, 0, bytearray(2048), [1], 1, "entrance") == "dead"
    assert fake._dumped == []


def test_a_live_clear_is_still_banked_and_reported() -> None:
    fake = _observe_solver(_AlwaysClear(dead=False))
    assert Solver.observe(fake, 0, bytearray(2048), [1], 1, "entrance") == "clear"
    assert len(fake._dumped) == 1


def test_a_candidate_that_fails_verification_abandons_the_lineage() -> None:
    # _dump_solution returning False means "not a clear at all". observe must
    # report dead, not clear: reporting clear would end the burst AND leave the
    # caller believing a solution exists.
    fake = _observe_solver(_AlwaysClear(dead=False))
    fake._dump_solution = lambda *a, **k: False
    assert Solver.observe(fake, 0, bytearray(2048), [1], 1, "entrance") == "dead"


# --------------------------------------------------------------------------
# Replay verification on the banking path
# --------------------------------------------------------------------------

class _ScriptedPool:
    """Duck-typed nes_core.Pool that replays a pre-scripted RAM sequence.

    step_all returns the next scripted frame; the first call is the rooting
    NOOP, matching seed()'s convention.
    """

    instances: list = []

    def __init__(self, rom_path=None, num_workers=1, frame_skip=4) -> None:
        self.rom_path, self.num_workers, self.frame_skip = (
            rom_path, num_workers, frame_skip)
        self.loaded: list = []
        self.reset_calls = 0
        self.shutdown_calls = 0
        self.steps = 0
        self.headless = None
        _ScriptedPool.instances.append(self)

    # -- config surface the verifier drives
    def set_headless(self, v) -> None:
        self.headless = v

    def set_skip_preprocess(self, v) -> None:
        self.skip_pre = v

    def reset_all(self) -> None:
        self.reset_calls += 1

    def load_worker_state(self, wid, blob) -> None:
        self.loaded.append((wid, blob))

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def step_all(self, acts):
        i = self.steps
        self.steps += 1
        frames = _ScriptedPool.script
        ram = frames[min(i, len(frames) - 1)]
        return [(None, None, ram)]


class _ReplayGame:
    """Clear/dead verdicts keyed off a single scripted RAM byte.

    STATELESS on purpose (a pure function of the current frame), which is the
    shape of every hook that needs no verify margin. It is explicitly NOT the
    shape that broke: see _StridedGame below."""

    def is_dead(self, ram, start_lives) -> bool:
        return ram[0] == 2

    def is_finale(self, start_key, ram) -> bool:
        return ram[0] == 3

    def is_clear(self, start_key, ram, ctx=None) -> bool:
        return ram[0] == 1

    def level_key(self, ram) -> tuple:
        return (0,)

    def clear_verify_margin(self) -> int:
        return 0

    rom = "roms/does-not-need-to-exist.nes"


def _verify_solver(tmp_path, root_path) -> SimpleNamespace:
    return SimpleNamespace(
        game=_ReplayGame(), bitmasks=[0, 1, 2, 3, 4, 5],
        frame_skip=4, hw_flags=[], start_wd=(0,), start_lives=3,
        roots={"entrance": {"path": str(root_path)}},
    )


@pytest.fixture()
def scripted(monkeypatch, tmp_path):
    import scripts.go_explore_solve as ges

    _ScriptedPool.instances = []
    monkeypatch.setattr(ges, "Pool", _ScriptedPool)
    root = tmp_path / "entrance.state"
    root.write_bytes(b"rootblob")
    return root


def _frames(codes) -> list[bytes]:
    return [bytes([c]) + bytes(2047) for c in codes]


def test_replay_verification_accepts_a_clear_that_reproduces(scripted) -> None:
    # rooting NOOP, then 3 actions, the last of which reads clear.
    _ScriptedPool.script = _frames([0, 0, 0, 1])
    fake = _verify_solver(None, scripted)
    out = Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert out["ok"] is True
    assert (out["verdict"], out["at"], out["n_actions"]) == ("clear", 3, 3)
    pool = _ScriptedPool.instances[-1]
    assert pool.num_workers == 1 and pool.frame_skip == 4
    assert pool.loaded == [(0, b"rootblob")]      # rooted at the ROOT state
    assert pool.headless is True
    assert pool.shutdown_calls == 1               # never leaks a pool


def test_replay_verification_accepts_a_reproducing_finale(scripted) -> None:
    _ScriptedPool.script = _frames([0, 0, 3])
    fake = _verify_solver(None, scripted)
    assert Solver.replay_verify(fake, "entrance", [1, 1])["ok"] is True


def test_replay_verification_rejects_a_candidate_that_dies_instead(scripted) -> None:
    # The Gradius signature: what the live hook called a clear replays as a
    # death. This is the case that must never reach solutions/.
    # Persistent death (>= 3 frames — the post-debounce contract; a
    # 1-frame flicker is a transition blip and survives by design).
    _ScriptedPool.script = _frames([0, 2, 2, 2])
    fake = _verify_solver(None, scripted)
    # trace long enough that all three dead reads land IN-TRACE (the
    # margin region only adjudicates clears, never deaths).
    out = Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert out["ok"] is False and out["verdict"] == "dead"


def test_replay_verification_rejects_a_candidate_that_never_clears(scripted) -> None:
    _ScriptedPool.script = _frames([0, 0, 0, 0])
    fake = _verify_solver(None, scripted)
    out = Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert out["ok"] is False and out["verdict"] == "no_clear"


def test_replay_verification_fails_closed_on_an_infrastructure_error(
        scripted) -> None:
    # A missing root, a dead pool, a bad blob: never raise into the search
    # loop, and never report ok. Banking an unverified clear is the outcome
    # this whole path exists to prevent.
    _ScriptedPool.script = _frames([0, 1])
    fake = _verify_solver(None, scripted)
    fake.roots = {}
    out = Solver.replay_verify(fake, "entrance", [1])
    assert out["ok"] is False and out["verdict"] == "error"
    assert "KeyError" in out["error"]


def test_replay_verification_shuts_the_pool_down_even_when_it_throws(
        scripted) -> None:
    class _Exploding(_ScriptedPool):
        def step_all(self, acts):
            raise RuntimeError("core panic")

    import scripts.go_explore_solve as ges
    ges.Pool = _Exploding
    _ScriptedPool.script = _frames([0, 1])
    out = Solver.replay_verify(_verify_solver(None, scripted), "entrance", [1])
    assert out["ok"] is False and "RuntimeError" in out["error"]
    assert _ScriptedPool.instances[-1].shutdown_calls == 1


# --------------------------------------------------------------------------
# Replay verification vs. a WINDOWED, STRIDE-PHASED hook
#
# Every test above this point drives _ReplayGame, whose is_clear is a pure
# function of the current frame. That fake cannot express the actual defect:
# the streaming detector only EVALUATES when its own observation counter hits
# a multiple of `stride`, and that counter starts when the ctx is created --
# per BURST live, at the ROOT in a replay. When the phases differ the replay's
# last check falls before the evidence and a GENUINE clear is rejected.
# --------------------------------------------------------------------------

class _StridedGame(_ReplayGame):
    """A clear hook with the one property that broke verification: it only
    LOOKS at the RAM every `stride` observations, counting from ctx creation.

    Deliberately minimal -- no rolling window, no confluence -- so the test
    isolates the phase bug rather than re-testing the detector."""

    stride = 20

    def __init__(self, margin: int) -> None:
        self._margin = margin

    def is_clear(self, start_key, ram, ctx=None) -> bool:
        if ctx is None:
            return False
        n = ctx["_n"] = ctx.get("_n", 0) + 1
        return n % self.stride == 0 and ram[0] == 1

    def clear_verify_margin(self) -> int:
        return self._margin


def _strided_live_fire(game, stream, burst_start: int):
    """Where the LIVE hook fires, with the ctx _assign() builds per burst."""
    ctx: dict = {}
    for i in range(burst_start, len(stream)):
        if game.is_clear((0,), stream[i], ctx):
            return i
    return None


# script[0] is the rooting NOOP, so script index j is trace index j-1: the
# clear signature therefore begins at trace index 200, and the ROOT-phase
# grid's first check at or after it is trace index 219. Any burst whose own
# grid fires before 219 produces a trace too short for the root to reach its
# check -- which is the whole bug.
_STRIDED_STREAM_CODES = [0] * 201 + [1] * 200


def _strided_case(scripted, burst_start: int, margin: int):
    stream = _frames(_STRIDED_STREAM_CODES)
    game = _StridedGame(margin=margin)
    live_at = _strided_live_fire(game, stream[1:], burst_start)
    assert live_at is not None, "the live hook must fire, or this is vacuous"
    _ScriptedPool.script = stream
    fake = _verify_solver(None, scripted)
    fake.game = game
    # The banked trace ends ON the live fire frame -- no steps left for the
    # replay's stride phase to re-align in.
    return Solver.replay_verify(fake, "entrance", [1] * (live_at + 1))


@pytest.mark.parametrize("burst_start", [0, 1, 7, 13, 19, 20, 31])
def test_a_genuine_clear_reproduces_from_the_root_at_every_burst_alignment(
        scripted, burst_start) -> None:
    out = _strided_case(scripted, burst_start, _StridedGame.stride)
    assert out["ok"] is True, f"genuine clear rejected: {out}"
    assert out["verdict"] == "clear"
    assert out["margin"] == _StridedGame.stride


def test_without_the_margin_most_burst_alignments_reject_a_genuine_clear(
        scripted) -> None:
    # The regression the margin exists for, pinned so it cannot silently come
    # back -- and quantified, so "the margin fixes it" is not vacuous. Only
    # the alignments that happen to share the root's phase survive.
    rejected = [b for b in range(20)
                if not _strided_case(scripted, b, 0)["ok"]]
    assert len(rejected) >= 15, rejected
    # ...and every one of them passes once the margin is restored.
    assert all(_strided_case(scripted, b, _StridedGame.stride)["ok"]
               for b in rejected)


def test_the_margin_enforces_dead_before_clear(scripted) -> None:
    # CONTRACT CHANGE (2026-08-24 hardening wave, audit P1): the margin
    # used to run ONLY the clear check, which was a hole — a
    # Gradius-class death arriving in the tail could not block a
    # phase-lagged clear hook firing after it, so a dying lineage
    # verified as a win. The margin now runs the SAME dead-before-clear
    # ordering as the trace (with the 3-frame blip debounce), so a
    # persistent tail death rejects.
    class _ClearThenDie(_ReplayGame):
        def clear_verify_margin(self) -> int:
            return 5

    _ScriptedPool.script = _frames([0, 0, 0, 2, 2, 2, 2, 2])
    fake = _verify_solver(None, scripted)
    fake.game = _ClearThenDie()
    # Trace of 2 actions, no clear; frames 3+ (margin) read persistently
    # DEAD -> the debounce confirms at the 3rd consecutive read and the
    # verdict is 'dead', never a fabricated clear.
    out = Solver.replay_verify(fake, "entrance", [1, 1])
    assert out["verdict"] == "dead" and out["ok"] is False
    # And a REAL phase-lagged clear that fires before any persistent
    # death still verifies: frames read alive, clear fires in the tail.
    _ScriptedPool.script = _frames([0, 0, 0, 1, 1, 1, 1, 1])
    fake2 = _verify_solver(None, scripted)
    fake2.game = _ClearThenDie()   # margin 5 — the tail must be reachable
    out2 = Solver.replay_verify(fake2, "entrance", [1, 1])
    assert out2["verdict"] == "clear" and out2["ok"] is True


def test_a_death_INSIDE_the_trace_still_rejects(scripted) -> None:
    # The bound must not become a hole: dying during the trace itself is still
    # a rejection, margin or no margin.
    class _M(_ReplayGame):
        def clear_verify_margin(self) -> int:
            return 5

    # Death contract (2026-08-24, the Rygar door lesson): a PERSISTENT
    # dead read (>= 3 consecutive frames) rejects; a single-frame
    # flicker is a transition blip and survives with nothing recorded.
    _ScriptedPool.script = _frames([0, 2, 2, 2, 1, 1, 1])
    fake = _verify_solver(None, scripted)
    fake.game = _M()
    out = Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert out["verdict"] == "dead" and out["at"] == 3 and out["ok"] is False
    # ...and the blip form no longer rejects (it would have poisoned
    # every door crossing):
    _ScriptedPool.script = _frames([0, 0, 2, 1, 1, 1, 1])
    out2 = Solver.replay_verify(_verify_solver(None, scripted), "entrance",
                                [1, 1, 1])
    assert out2["verdict"] != "dead"


def test_a_clear_inside_the_margin_reports_an_at_past_the_trace(
        scripted) -> None:
    # `at` > n_actions is the honest signal that the clear reproduced in the
    # margin -- a phase difference, not a different trajectory.
    class _M(_ReplayGame):
        def clear_verify_margin(self) -> int:
            return 10

    _ScriptedPool.script = _frames([0, 0, 0, 0, 0, 1])
    fake = _verify_solver(None, scripted)
    fake.game = _M()
    out = Solver.replay_verify(fake, "entrance", [1, 1])
    assert out["ok"] is True and out["at"] == 5 > out["n_actions"] == 2


def test_the_margin_is_a_bounded_noop_tail_not_an_open_ended_search(
        scripted) -> None:
    # A clear that needs MORE than the margin is still a rejection: the tail
    # exists to re-align the check phase, not to keep playing until it wins.
    class _M(_ReplayGame):
        def clear_verify_margin(self) -> int:
            return 3

    _ScriptedPool.script = _frames([0, 0, 0, 0, 0, 0, 1])
    fake = _verify_solver(None, scripted)
    fake.game = _M()
    out = Solver.replay_verify(fake, "entrance", [1, 1])
    assert out["ok"] is False and out["verdict"] == "no_clear"
    assert _ScriptedPool.instances[-1].steps == 1 + 2 + 3   # NOOP + trace + margin


def test_the_margin_tail_is_fed_as_noops(scripted) -> None:
    # The tail must not extend the TRAJECTORY -- it replays action 0.
    class _M(_ReplayGame):
        def clear_verify_margin(self) -> int:
            return 3

    class _Recording(_ScriptedPool):
        def step_all(self, acts):
            self.acts = getattr(self, "acts", [])
            self.acts.append(int(acts[0]))
            return super().step_all(acts)

    import scripts.go_explore_solve as ges
    ges.Pool = _Recording
    _ScriptedPool.script = _frames([0])
    fake = _verify_solver(None, scripted)
    fake.game = _M()
    fake.bitmasks = [0x00, 0x81, 0x82]
    Solver.replay_verify(fake, "entrance", [1, 2])
    # rooting NOOP, the two trace actions, then 3 NOOPs.
    assert _ScriptedPool.instances[-1].acts == [0x00, 0x81, 0x82, 0, 0, 0]


# --------------------------------------------------------------------------
# The margin the REAL adapter asks for
# --------------------------------------------------------------------------

def test_every_game_adapter_defines_the_verify_margin() -> None:
    """replay_verify calls clear_verify_margin() unconditionally, and its
    broad except turns any AttributeError into verdict 'error' -- which is
    ok=False, i.e. EVERY real clear rejected for that game. SmbGame does not
    inherit from GenericGame, so this is not free."""
    import scripts.go_explore_solve as ges

    adapters = [v for v in vars(ges).values()
                if isinstance(v, type) and v.__name__.endswith("Game")
                and v.__module__ == ges.__name__]
    assert len(adapters) >= 2, adapters
    for cls in adapters:
        assert callable(getattr(cls, "clear_verify_margin", None)), cls.__name__


def test_smb_asks_for_no_margin_so_its_replays_are_unchanged() -> None:
    import scripts.go_explore_solve as ges
    assert ges.SmbGame.clear_verify_margin(object()) == 0


def test_stateless_clear_hooks_ask_for_no_margin_at_all() -> None:
    # SMB/CV level_key, the finale, byte_change and score_jump all fire on the
    # exact frame, so their replays must stay exactly as cheap as before.
    base = {"rom": "x.nes", "y": A_Y, "lives": A_LIVES, "area": A_ROOM,
            "progress": {"lo": A_GX_LO, "hi": A_GX_HI}, "level_key": []}
    assert GenericGame({"solve": base}).clear_verify_margin() == 0
    for mode in ({"mode": "score_jump", "threshold": 500},
                 {"mode": "byte_change", "addr": 0x75}):
        g = GenericGame({"solve": dict(base, clear=mode)})
        assert g.clear_verify_margin() == 0


def test_the_confluence_margin_covers_a_full_check_period() -> None:
    # >= stride-1 is what re-aligns the phase (measured: 136/200 genuine
    # clears rejected at margin 0, 0/200 at stride-1), and persist_checks
    # multiplies the number of checks the latch needs.
    assert _game().clear_verify_margin() == 20
    assert _game(stride=15).clear_verify_margin() == 15
    assert _game(stride=15, persist_checks=3).clear_verify_margin() == 45
    assert _game().clear_verify_margin() >= _game().__dict__["_conf_stride"] - 1


def test_a_genuine_confluence_clear_reproduces_through_the_REAL_detector(
        scripted) -> None:
    """End-to-end on the real GenericGame + real StreamingConfluenceDetector:
    the shape test_a_clear_that_costs_no_life_is_not_suppressed_by_the_veto
    says must be bankable has to survive replay verification too."""
    stream = [_ram(gx=(400 + 3 * i if i < 200 else 4 * (i - 200)),
                   lives=3, entities=i < 200, **_tally(i))
              for i in range(400)]
    game = _game()
    start_wd = game.level_key(bytes(2048))
    rejected = fired = 0
    for burst_start in range(0, 160, 7):
        ctx: dict = {}
        live_at = None
        for i in range(burst_start, len(stream)):
            if game.is_clear(start_wd, stream[i], ctx):
                live_at = i
                break
        if live_at is None:
            continue
        fired += 1
        _ScriptedPool.script = [bytes(2048)] + stream   # [0] = rooting NOOP
        fake = _verify_solver(None, scripted)
        fake.game, fake.start_wd = game, start_wd
        if not Solver.replay_verify(fake, "entrance", [1] * (live_at + 1))["ok"]:
            rejected += 1
    assert fired >= 20, "not enough live fires to be a real sweep"
    assert rejected == 0, f"{rejected}/{fired} genuine clears rejected"


# --------------------------------------------------------------------------
# _dump_solution: verification gates the write
# --------------------------------------------------------------------------

def _bank_solver(tmp_path, verdict_ok: bool, verify_bank: bool = True):
    (tmp_path / "solutions").mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        best_sol_len=10 ** 9, sol_counter=0, n_solutions=0, out=tmp_path,
        verify_bank=verify_bank, verify_checks=0, verify_rejections=0,
        args=SimpleNamespace(profile="configs/kirby.yaml", workers=8),
        roots={"entrance": {"path": "does/not/matter.state"}},
        provenance={"hw_flags": [], "frame_skip": 4},
        start_wd=(0,), game=SimpleNamespace(level_key=lambda ram: (1,)),
    )
    fake.replay_verify = lambda root_id, trace: {
        "ok": verdict_ok, "verdict": "clear" if verdict_ok else "dead",
        "at": 3, "n_actions": len(trace), "elapsed_s": 0.1}
    return fake


def test_a_verified_clear_is_banked_and_marked_verified(tmp_path) -> None:
    fake = _bank_solver(tmp_path, verdict_ok=True)
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is True
    rec = json.loads((tmp_path / "solutions" / "sol_000.json").read_text())
    assert rec["replay_verified"] is True
    assert rec["provenance"] == "search"          # honest-origin marker intact
    assert (fake.n_solutions, fake.verify_rejections) == (1, 0)


def test_a_rejected_clear_writes_nothing_at_all(tmp_path) -> None:
    fake = _bank_solver(tmp_path, verdict_ok=False)
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is False
    assert list((tmp_path / "solutions").iterdir()) == []
    # None of the banking counters may move — a rejected candidate must not
    # consume a sol_NNN slot or lower the best-length bar for later clears.
    assert (fake.n_solutions, fake.sol_counter) == (0, 0)
    assert fake.best_sol_len == 10 ** 9
    assert fake.verify_rejections == 1


def test_a_rejection_is_logged_loudly_on_stderr(tmp_path, capsys) -> None:
    Solver._dump_solution(_bank_solver(tmp_path, verdict_ok=False),
                          "entrance", [1, 2, 3], bytearray(2048), 3)
    err = capsys.readouterr().err
    assert "CLEAR REJECTED" in err and "NOTHING banked" in err


def test_no_verify_bank_skips_the_replay_and_says_so_in_the_receipt(
        tmp_path) -> None:
    fake = _bank_solver(tmp_path, verdict_ok=False, verify_bank=False)
    called = []
    fake.replay_verify = lambda *a, **k: called.append(a)
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is True
    rec = json.loads((tmp_path / "solutions" / "sol_000.json").read_text())
    assert rec["replay_verified"] is False
    assert called == []


def test_a_non_shorter_re_clear_is_real_but_pays_no_replay(tmp_path) -> None:
    fake = _bank_solver(tmp_path, verdict_ok=True)
    fake.best_sol_len = 5
    called = []
    fake.replay_verify = lambda *a, **k: called.append(a)
    # Real clear, just not worth another receipt: True, nothing written, and
    # no seconds burned re-simulating it.
    assert Solver._dump_solution(fake, "entrance", [1] * 10,
                                 bytearray(2048), 10) is True
    assert list((tmp_path / "solutions").iterdir()) == []
    assert called == []
