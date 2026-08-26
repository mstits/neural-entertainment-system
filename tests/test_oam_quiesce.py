"""OAM sprite-population quiescence signal (clear_detect.OamQuiesceSignal).

The OAM-hardware sibling of ApuActivitySignal: a sustained collapse of the
visible-sprite population, scored against a null this run self-measures from
its own history. Renderability is read straight off our own PPU's hide rule
(ppu.rs:990 is_sprite_at_y_on_scanline: y >= 240 can never be evaluated onto
any of the 240 rendered scanlines) -- a hardware fact, not a game convention
-- so a game whose entities live in an unusual RAM region still shows the
collapse here even though entity_wipe_windows (the RAM heuristic) cannot see
it.

The FIRST test below is the CAN-FAIL proof: ordinary, unchanging play must
never fire this signal. Two vacuous gates shipped this week because nobody
wrote that test first for them.

Every documented false-positive class this signal cannot discriminate is
asserted here as a POSITIVE control (it must fire): death, room transition.
A brief combat blip is asserted as a NEGATIVE control (it must not fire) --
exactly the blip-immunity bound this signal is built to prove, not assume.
"""

from __future__ import annotations

import pytest

from scripts.clear_detect import (
    OAM_COLLAPSE,
    OAM_HIDE_Y,
    OAM_MIN_BASELINE,
    OAM_SHORT_WINDOW,
    OAM_SUSTAIN,
    OamQuiesceSignal,
    oam_census,
)

# A populated, steady baseline: 40 visible sprites drawn from 10 distinct
# (tile, attr) pairs -- arbitrary test fixture numbers, not any game's real
# sprite budget.
POPULATED = (40, 10)
SPARSE = (5, 2)
EMPTY = (0, 0)

# Observations needed before the signal has a null at all (mirrors
# ApuActivitySignal's WARMUP in tests/test_detector_v3.py).
WARMUP = OAM_SHORT_WINDOW + OAM_MIN_BASELINE

# The exact blip-immunity boundary the class docstring derives:
# L > short_window * (1 - collapse) is required to fire, so this many
# observations of total collapse must NOT fire at any alignment, and one
# more must.
BOUND = round(OAM_SHORT_WINDOW * (1 - OAM_COLLAPSE))


def _feed(sig: OamQuiesceSignal, censuses) -> OamQuiesceSignal:
    for c in censuses:
        sig.push(c)
    return sig


def _steady(n: int, census=POPULATED) -> list:
    return [census] * n


def _transient(n: int, at: int, length: int, census=EMPTY,
               base=POPULATED) -> list:
    """`length` observations of a different population, then straight back
    to the baseline census."""
    out = _steady(n, base)
    for i in range(at, min(at + length, n)):
        out[i] = census
    return out


# ---------------------------------------------------------------------------
# THE CAN-FAIL PROOF -- written first. If this does not hold, nothing below
# it means anything: a signal that fires on ordinary, unchanging play is not
# measuring a collapse, it is measuring the act of being fed data.
# ---------------------------------------------------------------------------

def test_the_signal_does_not_fire_on_ordinary_unchanging_play() -> None:
    sig = _feed(OamQuiesceSignal(), _steady(600))
    assert sig.trigger_n is None
    assert sig.vote() == 0
    assert sig.n_evals > 0          # it DID evaluate -- this is a real "no",
                                     # not a silent warm-up shortfall


def test_stubbing_the_mechanism_out_reports_no_fire_not_a_crash() -> None:
    # The mechanism "absent" means: nothing ever collapses. A detector that
    # can only ever say yes would pass every other test in this file
    # vacuously. This is that check made explicit and separate from the one
    # above, over a different, sparser steady population.
    sig = _feed(OamQuiesceSignal(), _steady(500, SPARSE))
    assert sig.trigger_n is None and sig.vote() == 0


# ---------------------------------------------------------------------------
# oam_census() -- the hardware-fact extraction, tested with no signal at all
# ---------------------------------------------------------------------------

def _oam(entries: dict) -> bytes:
    """256-byte OAM buffer, every sprite hidden (y=255) by default, with
    `entries` (slot index -> (y, tile, attr)) overriding specific slots."""
    buf = bytearray([255] * 256)
    for slot, (y, tile, attr) in entries.items():
        buf[4 * slot] = y
        buf[4 * slot + 1] = tile
        buf[4 * slot + 2] = attr
    return bytes(buf)


def test_a_sprite_at_y_239_is_visible_and_240_is_hidden() -> None:
    # ppu.rs:990's own boundary: y <= scanline < y+height with scanline in
    # 0..239 -- y=239 can still match scanline 239, y=240 can never match
    # anything. OAM_HIDE_Y pins the constant to that hardware fact.
    assert OAM_HIDE_Y == 240
    nv, nd = oam_census(_oam({0: (239, 1, 0)}))
    assert (nv, nd) == (1, 1)
    nv, nd = oam_census(_oam({0: (240, 1, 0)}))
    assert (nv, nd) == (0, 0)


def test_a_fully_hidden_table_is_zero_and_zero() -> None:
    assert oam_census(_oam({})) == (0, 0)


def test_distinct_pairs_are_counted_only_among_visible_sprites() -> None:
    # A hidden slot's stale leftover tile/attr bytes must not inflate the
    # distinct count -- 63 hidden slots each carrying a different stale tile
    # byte would otherwise read as 63 fictitious objects.
    entries = {0: (100, 1, 0)}
    for slot in range(1, 64):
        entries[slot] = (255, slot, slot)   # hidden, but all-different bytes
    nv, nd = oam_census(_oam(entries))
    assert (nv, nd) == (1, 1)


def test_two_visible_sprites_sharing_tile_and_attr_count_as_one_pair() -> None:
    nv, nd = oam_census(_oam({0: (100, 5, 2), 1: (120, 5, 2)}))
    assert (nv, nd) == (2, 1)


def test_two_visible_sprites_with_different_attr_count_as_two_pairs() -> None:
    nv, nd = oam_census(_oam({0: (100, 5, 2), 1: (120, 5, 3)}))
    assert (nv, nd) == (2, 2)


def test_a_short_oam_buffer_reads_as_empty_rather_than_crashing() -> None:
    assert oam_census(b"") == (0, 0)
    assert oam_census(bytes(10)) == (0, 0)


# ---------------------------------------------------------------------------
# The null has to be earned before there is a vote
# ---------------------------------------------------------------------------

def test_the_vote_does_not_exist_until_the_null_is_measured() -> None:
    sig = _feed(OamQuiesceSignal(), _steady(WARMUP - 1))
    assert sig.n_evals == 0 and sig.vote() == 0 and sig.trigger_n is None
    sig.push(POPULATED)
    assert sig.n_evals == 1


def test_a_missing_census_is_ignored_rather_than_read_as_a_collapse() -> None:
    # A caller that has not plumbed the OAM modality yet passes None.
    # Counting that as "every sprite vanished" would fabricate the loudest
    # possible collapse signal out of an absent measurement -- the OAM
    # analogue of ApuActivitySignal's missing-mask test.
    sig = _feed(OamQuiesceSignal(), _steady(WARMUP + 60))
    n_before, evals_before = sig.n, sig.n_evals
    for _ in range(200):
        sig.push(None)
    assert (sig.n, sig.n_evals) == (n_before, evals_before)
    assert sig.trigger_n is None


# ---------------------------------------------------------------------------
# The blip-immunity bound: L > short_window * (1 - collapse), alignment-
# independent, derived (not assumed) in the class docstring.
# ---------------------------------------------------------------------------

def test_the_bound_matches_the_documented_derivation() -> None:
    assert BOUND == 24            # short_window=30, collapse=0.2 defaults


@pytest.mark.parametrize("at", [95, 100, 111, 137, 200, 349])
def test_a_collapse_at_the_bound_never_fires_at_any_alignment(at: int) -> None:
    sig = _feed(OamQuiesceSignal(), _transient(400, at, BOUND))
    assert sig.trigger_n is None, (at, sig.stats())


@pytest.mark.parametrize("at", [95, 100, 111, 137, 200, 349])
def test_one_observation_longer_fires_at_every_alignment(at: int) -> None:
    sig = _feed(OamQuiesceSignal(), _transient(400, at, BOUND + 1))
    assert sig.trigger_n is not None, (at, sig.stats())


def test_the_bound_is_derived_from_the_live_knobs_not_hardcoded() -> None:
    # Raising `collapse` (a smaller required drop) shortens the bound;
    # a caller that changes the knob gets a correspondingly different
    # boundary, not a constant frozen at construction time.
    loose = OamQuiesceSignal(collapse=0.5)   # bound = 30*0.5 = 15
    at = 137
    short = _feed(loose, _transient(400, at, 15))
    assert short.trigger_n is None
    longer = _feed(OamQuiesceSignal(collapse=0.5), _transient(400, at, 16))
    assert longer.trigger_n is not None


def test_the_bound_is_independent_of_the_games_own_population_scale() -> None:
    # `scale` cancels out of the fire condition algebraically -- a densely
    # populated game and a sparsely populated one must cross at the SAME L.
    populous_short = _feed(OamQuiesceSignal(),
                            _transient(400, 137, BOUND, base=(50, 20)))
    sparse_short = _feed(OamQuiesceSignal(),
                          _transient(400, 137, BOUND, base=(6, 3)))
    assert populous_short.trigger_n is None
    assert sparse_short.trigger_n is None
    populous_fire = _feed(OamQuiesceSignal(),
                           _transient(400, 137, BOUND + 1, base=(50, 20)))
    sparse_fire = _feed(OamQuiesceSignal(),
                         _transient(400, 137, BOUND + 1, base=(6, 3)))
    assert populous_fire.trigger_n is not None
    assert sparse_fire.trigger_n is not None


def test_a_baseline_that_was_already_empty_cannot_collapse_further() -> None:
    # scale == 0: there is no population to collapse FROM. The gate becomes
    # unreachable, which is the safe direction (never fires on nothing).
    sig = _feed(OamQuiesceSignal(), _steady(600, EMPTY))
    assert sig.trigger_n is None


# ---------------------------------------------------------------------------
# Direction: only a DROP counts. A population RISE is the opposite of a
# clear and must never contribute a particle of vote (unlike
# ApuActivitySignal, which is deliberately direction-agnostic).
# ---------------------------------------------------------------------------

def test_a_population_rise_never_fires() -> None:
    sig = _feed(OamQuiesceSignal(),
                _steady(200, SPARSE) + _steady(300, (60, 20)))
    assert sig.trigger_n is None


def test_a_sharp_rise_then_a_real_collapse_still_fires_on_the_collapse() -> None:
    sig = _feed(OamQuiesceSignal(),
                _steady(200, SPARSE) + _steady(200, (60, 20)))
    assert sig.trigger_n is None
    _feed(sig, _steady(60, EMPTY))
    assert sig.trigger_n is not None


# ---------------------------------------------------------------------------
# ANTI-VACUITY CONTROLS, stated as positive assertions: this signal is a
# population-collapse detector, not a clear detector, and these are the
# measured/documented shapes it cannot tell apart from a real clear. If a
# later "fix" makes any of these go quiet, it broke the premise these
# controls encode, not improved discrimination.
# ---------------------------------------------------------------------------

def test_oam_quiesce_fires_on_a_synthesized_death() -> None:
    # The player and the enemies that killed them explode off the sprite
    # table together and stay gone for a death animation -- a SUSTAINED
    # wipe, not a blip.
    sig = _feed(OamQuiesceSignal(), _steady(200) + _steady(120, EMPTY))
    assert sig.trigger_n is not None


def test_oam_quiesce_fires_on_a_synthesized_room_transition() -> None:
    # The new room's entities have not spawned in yet -- only the player
    # sprite (or nothing) is visible for a stretch well past the bound.
    sig = _feed(OamQuiesceSignal(),
                _steady(200) + _steady(80, (1, 1)))
    assert sig.trigger_n is not None


def test_oam_quiesce_fires_on_a_synthesized_screen_fade() -> None:
    sig = _feed(OamQuiesceSignal(), _steady(200) + _steady(90, EMPTY))
    assert sig.trigger_n is not None


def test_an_ordinary_oscillating_wave_rhythm_can_repeatedly_false_fire() -> None:
    # AN UNEXAMINED RESIDUAL, found while writing this file's tests, pinned
    # here rather than left for someone else to rediscover: this signal's
    # gate is a FIXED fraction of the baseline's MAGNITUDE only, with no
    # variance term the way ApuActivitySignal's `gate_k * null_sigma +
    # gate_floor` has. A game whose sprite population legitimately swings
    # between a busy wave and a near-empty lull on its own ordinary rhythm
    # -- no clear ever happening, just normal play -- can cross the
    # collapse gate on every lull if the lull outlasts the blip-immunity
    # bound (a little over 1s of real gameplay at 30 Hz observations).
    # This is a POSITIVE assertion of a real false-positive mode, not a
    # "should never happen": it documents why oam_quiesce must never stand
    # in a vote without a transition-confirming corroborator, sharper than
    # the transient (one-off) false positives above because this one
    # repeats for as long as the game runs.
    busy, lull, period = (30, 8), (2, 1), 40
    rhythm = ([busy] * period + [lull] * period) * 20
    sig = _feed(OamQuiesceSignal(), rhythm)
    assert sig.n_triggers >= 2, (
        "if this goes quiet, the gate grew a variance term and this "
        "docstring's warning is stale -- update both together")


def test_oam_quiesce_does_not_fire_on_a_synthesized_combat_blip() -> None:
    # A wave of enemies despawning together and a fresh wave spawning right
    # back in -- brief, well under the blip-immunity bound. This is the
    # negative mirror of the three positive controls above: the same
    # mechanism that reads a death as a clear correctly refuses to read an
    # ordinary firefight lull as one, because the lull does not SUSTAIN.
    for at in (90, 150, 210):
        sig = _feed(OamQuiesceSignal(), _transient(400, at, 8))
        assert sig.trigger_n is None, (at, sig.stats())


# ---------------------------------------------------------------------------
# Hold / re-arm / reset -- identical shape to ApuActivitySignal, reused
# rather than re-derived.
# ---------------------------------------------------------------------------

def test_the_vote_is_held_for_a_bounded_run_of_observations() -> None:
    sig = OamQuiesceSignal(hold=10)
    _feed(sig, _steady(200))
    for _ in range(60):
        sig.push(EMPTY)
        if sig.trigger_n is not None:
            break
    assert sig.trigger_n is not None and sig.vote() == 1
    _feed(sig, _steady(9, EMPTY))
    assert sig.vote() == 1
    sig.push(EMPTY)
    assert sig.vote() == 0          # hold expired; the fire does not persist


def test_the_signal_re_arms_after_the_population_recovers() -> None:
    sig = OamQuiesceSignal(hold=20)
    _feed(sig, _steady(200))
    _feed(sig, _steady(60, EMPTY))
    assert sig.n_triggers == 1
    # Recovery: population returns to (above) baseline for a full baseline
    # window, so the null re-centers on "populated" and the vote drops.
    _feed(sig, _steady(sig.baseline_window + OAM_SHORT_WINDOW, POPULATED))
    assert sig.vote() == 0 and sig.n_triggers == 1
    for _ in range(80):
        sig.push(EMPTY)
        if sig.n_triggers == 2:
            break
    assert sig.n_triggers == 2 and sig.vote() == 1


def test_one_long_collapse_is_not_re_declared_every_hold_window() -> None:
    sig = OamQuiesceSignal(hold=5)
    _feed(sig, _steady(200))
    _feed(sig, _steady(150, EMPTY))
    assert sig.n_triggers == 1


def test_reset_drops_both_the_latch_and_the_calibration() -> None:
    sig = _feed(OamQuiesceSignal(), _steady(200) + _steady(60, EMPTY))
    assert sig.vote() == 1
    sig.reset()
    assert sig.trigger_n is None and sig.vote() == 0 and sig.n_evals == 0
    assert sig.n == 0


# ---------------------------------------------------------------------------
# warmup_observations()
# ---------------------------------------------------------------------------

def test_the_warmup_bound_is_exact_not_a_guess() -> None:
    # Fastest physically possible fire: a full null of a populated baseline
    # for exactly `min_baseline` observations, then a total collapse from
    # the very next observation -- by the time the null is first eligible
    # to evaluate (at short_window + min_baseline pushes), the ENTIRE short
    # window already postdates the collapse, so `sustain` consecutive fires
    # follow with no further delay. Nothing can trigger sooner, and this
    # really does trigger at exactly the advertised observation, so the
    # bound is neither optimistic nor padded.
    bound = OamQuiesceSignal().warmup_observations()
    assert bound == WARMUP + OAM_SUSTAIN - 1
    loud_then_gone = _steady(OAM_MIN_BASELINE, POPULATED) + _steady(80, EMPTY)
    sig = _feed(OamQuiesceSignal(), loud_then_gone)
    assert sig.trigger_n == bound

    early = _feed(OamQuiesceSignal(), loud_then_gone[: bound - 1])
    assert early.trigger_n is None, "fired before its own bound"
