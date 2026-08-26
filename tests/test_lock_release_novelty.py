"""LockReleaseNoveltyTrack -- the one-way-door discriminator.

Every case here drives the signal with a synthesized (locked, room_fp)
stream -- no RAM layout, no game, no address. `room_fp` is any hashable
identity token; these tests use small ints purely for readability. That is
the whole point of the class under test: the discriminating power is in the
RISE / FALL / RE-ENTRY pattern of the token, not in what produced it.

Ordering discipline: the FIRST test below (test_no_lock_ever_never_fires)
is the one that proves the signal CAN return false at all -- it feeds a
trace in which the mechanism (a lock window) never occurs and asserts the
vote never leaves 0. A suite that only ever exercises the fire path could
pass unchanged if push() were replaced by a stub that always returns 1;
this test is what would catch that.
"""

from __future__ import annotations

import pytest

from scripts.clear_detect import LockReleaseNoveltyTrack


def _drive(track: LockReleaseNoveltyTrack, events) -> int:
    """Feed (locked, room_fp) pairs in order; return the final vote."""
    v = track.vote()
    for locked, room_fp in events:
        v = track.push(locked, room_fp)
    return v


def _hold(locked: bool, room_fp, n: int):
    return [(locked, room_fp)] * n


# --------------------------------------------------------------------------
# 0. The mechanism-absent control -- must come first.
# --------------------------------------------------------------------------

def test_no_lock_ever_never_fires():
    """The mechanism (a lock window) never occurs: `locked` is False for
    every observation, even though room_fp keeps producing brand-new,
    never-before-seen identities the whole time (exactly the shape a
    scrolling level or a room crawl produces). With no rise/fall of
    `locked` there is no candidate to ever arm, so the vote must stay 0 no
    matter how long the stream runs or how much "novelty" flows past."""
    track = LockReleaseNoveltyTrack(lock_max=100, m=50)
    events = [(False, room) for room in range(500)]
    assert _drive(track, events) == 0
    assert track.stats()["n_candidates"] == 0
    assert track.stats()["n_fires"] == 0


def test_no_room_fp_signal_wired_never_fires():
    """A caller that never wires a room-identity signal at all (every
    room_fp is None, forever) has handed this signal no evidence of
    novelty. A lock rising and falling with no fingerprint information on
    either side must not fire -- there is nothing to call "never seen
    before" without a fingerprint."""
    track = LockReleaseNoveltyTrack(lock_max=100, m=20)
    events = (_hold(False, None, 10) + _hold(True, None, 20)
              + _hold(False, None, 200))
    assert _drive(track, events) == 0


# --------------------------------------------------------------------------
# The two shapes named directly in the design.
# --------------------------------------------------------------------------

def test_respawn_shape_is_not_a_clear():
    """Synthesize lock -> release where the post-lock fingerprint EQUALS a
    pre-lock one (the textbook respawn: you die, control locks for the
    death animation, and you land back exactly where you were). Vote must
    stay 0 -- forever, not just immediately -- even with a long run of
    ordinary play afterward."""
    track = LockReleaseNoveltyTrack(lock_max=100, m=20)
    events = (_hold(False, "room_A", 10)      # ordinary play, room A settles
              + _hold(True, None, 15)          # death lock, mid-churn
              + _hold(False, "room_A", 300))   # released back into room A
    assert _drive(track, events) == 0
    stats = track.stats()
    assert stats["n_candidates"] == 1
    assert stats["n_respawn_shaped"] == 1
    assert stats["n_fires"] == 0


def test_one_way_shape_is_a_clear():
    """Post-lock fingerprint novel, pre-lock fingerprint not re-entered
    within `m` observations after release -- the genuine level-exit shape.
    Vote must become 1, and only after the `m`-observation confirmation
    window has actually elapsed (the delay is the point: see the class
    docstring's WHY THE m-OBSERVATION DELAY IS CORRECT section)."""
    m = 20
    track = LockReleaseNoveltyTrack(lock_max=100, m=m)
    events = (_hold(False, "room_A", 10)
              + _hold(True, None, 15)
              + _hold(False, "room_B", 1))     # release into a NEW room
    v = _drive(track, events)
    assert v == 0, "must not fire before the re-entry horizon has elapsed"

    # Feed m-1 more observations of staying away from room_A: still no fire.
    v = _drive(track, _hold(False, "room_B", m - 1))
    assert v == 0

    # One more observation crosses the horizon: fire.
    v = _drive(track, [(False, "room_B")])
    assert v == 1
    assert track.stats()["n_fires"] == 1


def test_kirby_room_shape_still_votes_on_lock_release_novelty():
    """THE FP CONTROL, as a positive assertion. An ORDINARY room transition
    -- a door leading to a novel room, with no notion of "level progress"
    anywhere in this signal's input -- MUST vote 1 here. That is the
    executable statement that this class is a transition-class discriminator
    and not a clear detector on its own: it cannot see progress, so it
    cannot be expected to refuse this shape. `progress_advance` is what
    must be required alongside it on any room-based profile; this test is
    what stops a future "fix" from quietly making that unnecessary by
    weakening this signal until it can no longer see genuine clears either."""
    m = 10
    track = LockReleaseNoveltyTrack(lock_max=100, m=m)
    events = (_hold(False, "hub", 5)
              + _hold(True, None, 8)            # door animation locks input
              + _hold(False, "kirby_room_62", m + 1))  # novel room, no re-entry
    assert _drive(track, events) == 1


# --------------------------------------------------------------------------
# The documented residual false positives -- pinned as positive assertions,
# not silently avoided.
# --------------------------------------------------------------------------

def test_death_that_respawns_into_a_never_visited_room_defeats_the_novelty_check():
    """Documented residual FP, proved on the record: a death whose respawn
    point has never been visited this episode (a game whose checkpoint is
    not fixed) satisfies the novelty test exactly like a genuine clear, and
    this signal cannot tell the two apart. Only a `lives_drop` veto one
    layer up covers this -- this test exists so a future change that
    quietly "fixes" it here is required to explain what replaced this
    guarantee, rather than have the gap vanish unnoticed."""
    m = 10
    track = LockReleaseNoveltyTrack(lock_max=100, m=m)
    events = (_hold(False, "room_A", 5)
              + _hold(True, None, 6)                     # death lock
              + _hold(False, "never_seen_checkpoint", m + 1))
    assert _drive(track, events) == 1


def test_combat_blip_lock_in_place_is_not_a_clear():
    """A brief hit-stun freeze (input-locked for a handful of frames) that
    releases back into the SAME room it started in is respawn-shaped
    (post-lock fp == pre-lock fp) and must be rejected by the plain novelty
    check, with no special-casing for "this was combat" anywhere -- the
    same mechanism that suppresses a death suppresses this."""
    track = LockReleaseNoveltyTrack(lock_max=30, m=15)
    events = (_hold(False, "arena", 20)
              + _hold(True, None, 4)            # hit-stun
              + _hold(False, "arena", 200))     # same room, fight continues
    assert _drive(track, events) == 0


def test_a_lock_window_longer_than_lock_max_is_never_a_candidate():
    """An attract-loop-shaped lock -- one that DOES eventually release, but
    holds far longer than any real clear's lock window -- must never even
    become a candidate, regardless of how novel or non-re-entrant the
    post-release fingerprint looks. This is `lock_max` doing its job."""
    lock_max = 50
    track = LockReleaseNoveltyTrack(lock_max=lock_max, m=10)
    events = (_hold(False, "title", 5)
              + _hold(True, None, lock_max + 1)   # exceeds the bound
              + _hold(False, "demo_stage_1", 100))  # novel, never re-entered
    assert _drive(track, events) == 0
    assert track.stats()["n_candidates"] == 0


def test_an_unbounded_lock_that_never_releases_never_fires():
    """The pure attract-loop / GAME OVER shape: `locked` goes high and
    simply never comes back down for the rest of the stream. No falling
    edge means no candidate is ever armed -- caught by construction, not
    by any threshold."""
    track = LockReleaseNoveltyTrack(lock_max=1_000_000, m=5)
    events = (_hold(False, "title", 5)
              + _hold(True, "irrelevant_while_locked", 5000))
    assert _drive(track, events) == 0
    assert track.stats()["n_candidates"] == 0


# --------------------------------------------------------------------------
# Re-entry, latching, None-handling, and the per-archive scope variant.
# --------------------------------------------------------------------------

def test_re_entry_during_the_countdown_cancels_the_pending_candidate():
    """Re-entry does not have to land on the very first post-release
    observation to defeat the candidate -- walking back into the pre-lock
    room ANY time before the `m`-observation horizon elapses must cancel
    it, and no further observations may resurrect it."""
    m = 30
    track = LockReleaseNoveltyTrack(lock_max=100, m=m)
    events = (_hold(False, "room_A", 5)
              + _hold(True, None, 5)
              + _hold(False, "room_B", 10)     # novel, pending armed
              + _hold(False, "room_A", 1)      # walked back mid-countdown
              + _hold(False, "room_B", m + 5))  # long tail afterward
    assert _drive(track, events) == 0
    assert track.stats()["n_fires"] == 0


def test_a_none_room_fp_cannot_satisfy_re_entry_but_time_still_passes():
    """Mid-churn observations (room_fp=None -- no settled identity yet)
    can never THEMSELVES satisfy the re-entry check (None is never equal
    to a real pre-lock fingerprint), but they are still real elapsed
    observations and legitimately count toward the `m`-observation
    confirmation horizon -- an M measured in observations should not
    silently stretch just because the room-identity signal churned for a
    while without ever actually reading the pre-lock room again."""
    m = 10
    track = LockReleaseNoveltyTrack(lock_max=100, m=m)
    events = (_hold(False, "room_A", 5)
              + _hold(True, None, 5)
              + _hold(False, "room_B", 1))       # release: novel, pending armed
    assert _drive(track, events) == 0
    # m-1 churn observations: not fired yet, and none of them resolved as
    # a (nonexistent) re-entry of room_A.
    v = _drive(track, _hold(False, None, m - 1))
    assert v == 0
    # One more churn observation crosses the horizon.
    v = _drive(track, [(False, None)])
    assert v == 1


def test_a_none_room_fp_is_never_recorded_as_a_visit():
    """None must never be recorded in the visited set -- a run of pure
    churn must leave `seen` empty, so a real fingerprint's first genuine
    appearance is still correctly novel no matter how much None traffic
    preceded it."""
    track = LockReleaseNoveltyTrack(lock_max=100, m=5)
    _drive(track, _hold(False, None, 200))
    assert track.stats()["n_seen"] == 0


def test_the_fire_latches_and_reset_clears_it():
    """Once fired, the vote is a LATCH -- it must stay 1 through further
    unrelated observations, mirroring StreamingConfluenceDetector's "stays
    True thereafter" contract. reset() drops the latch back to 0."""
    m = 5
    track = LockReleaseNoveltyTrack(lock_max=50, m=m)
    events = (_hold(False, "A", 3) + _hold(True, None, 3)
              + _hold(False, "B", m + 1))
    assert _drive(track, events) == 1
    # Further, totally ordinary observations do not un-fire it.
    assert _drive(track, _hold(False, "B", 1000)) == 1
    track.reset()
    assert track.vote() == 0
    assert track.stats()["n_fires"] == 0


def test_seen_can_be_preseeded_for_the_per_archive_variant():
    """The per-archive scope variant: construct with `per_episode=False`
    and a `seen` set pre-populated with a lineage's ancestor visits. A
    room that is novel to THIS episode but was already visited by an
    ancestor lineage must NOT be treated as novel."""
    ancestor_seen = {"room_A", "room_B"}
    track = LockReleaseNoveltyTrack(lock_max=100, m=10, per_episode=False,
                                     seen=ancestor_seen)
    events = (_hold(False, "room_C", 5)   # last settled pre-lock fp
              + _hold(True, None, 5)
              + _hold(False, "room_B", 200))  # ancestor already visited this
    assert _drive(track, events) == 0


def test_reset_preserves_seen_when_per_episode_is_false_but_clears_it_when_true():
    seeded = {"x"}
    persistent = LockReleaseNoveltyTrack(lock_max=10, m=5, per_episode=False,
                                          seen=set(seeded))
    persistent.push(False, "y")
    persistent.reset()
    assert "y" in persistent._seen and "x" in persistent._seen

    episodic = LockReleaseNoveltyTrack(lock_max=10, m=5, per_episode=True)
    episodic.push(False, "y")
    episodic.reset()
    assert "y" not in episodic._seen
