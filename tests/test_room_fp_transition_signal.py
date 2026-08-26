"""RoomFpTransitionSignal — the room-fingerprint clear-vote signal
(CLEAR_DETECTION_CAMPAIGN_2026-08-26, "room_fp_transition").

Everything here drives the WRAPPER class in scripts/clear_detect.py
directly with synthetic nametable-shaped byte arrays; it does not touch a
ROM, a Pool or a Solver. The pure mechanism underneath (room_fp_mask,
nt_fingerprint, fp_settle, classify_transition, RoomIndex) already has its
own exhaustive coverage in tests/test_room_fp.py -- these tests are about
whether the WRAPPER turns "a new room identity settled" into a vote
correctly, not about re-proving the hashing/classifier math.

Test order is deliberate: the very first test proves the signal CAN return
false when its mechanism never fires, before any test proves it can return
true. A signal that always votes 0 (or always votes 1) would pass every
test below it undetected otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.clear_detect import (
    ROOM_FP_DEFAULT_KIND,
    ROOM_FP_KINDS,
    RoomFpTransitionSignal,
)

# ---------------------------------------------------------------------
# Synthetic nametable fixtures. `_content_nt` is NOT uniform -- bytes
# 0:16 are a separate "HUD" region from bytes 16:2048's "room content"
# region, so a mask over [0, 16) is actually distinguishing something,
# not just zeroing an already-uniform array.
# ---------------------------------------------------------------------
HUD_RANGE = (0, 16)


def _content_nt(tile: int) -> np.ndarray:
    a = np.zeros(2048, dtype=np.uint8)
    a[16:2048] = tile & 0xFF
    return a


def _hud(nt: np.ndarray, digit: int) -> np.ndarray:
    """A COPY of `nt` with only its HUD region rewritten -- score/timer
    digits ticking, or a pause/menu overlay redrawing that region."""
    out = nt.copy()
    out[HUD_RANGE[0]:HUD_RANGE[1]] = digit & 0xFF
    return out


def _settle_seq(sig: RoomFpTransitionSignal, nt: np.ndarray, odos,
                 scene: int = 0, lines=None) -> None:
    """Push the SAME `nt` (a fixed hash) once per entry of `odos`, in
    order -- scripting the camera's motion across the churn window
    `fp_settle` measures onset-to-settle over. `len(odos)` must be >=
    `sig.settle` for a genuinely new hash to settle."""
    for odo in odos:
        sig.push(nt, odo_xy=odo, scene=scene, rendered_lines=lines)


# ---------------------------------------------------------------------
# THE CAN-FAIL TEST — must come first.
# ---------------------------------------------------------------------


def test_the_signal_can_return_false_when_the_mechanism_never_fires():
    """Prove vote() can be 0 by construction: a stream that settles once
    (the entrance room, baseline) and then NEVER produces a different
    settled fingerprint again -- the mechanism this signal exists to
    detect (a new room settling) never fires -- must vote 0 forever."""
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    room = _content_nt(1)
    _settle_seq(sig, room, [(0, 0)] * 3)          # baseline adoption
    assert sig.vote() == 0 and sig.n_triggers == 0
    for step in range(200):                        # ordinary play, same room
        sig.push(room, odo_xy=(step, 0), scene=0)
        assert sig.vote() == 0
    assert sig.n_triggers == 0 and sig.n_rooms() == 1


# ---------------------------------------------------------------------
# The mask does real work (design's own specified failing test).
# ---------------------------------------------------------------------


def test_room_fp_mask_is_doing_work():
    room = _content_nt(1)
    hud_only_flip = _hud(room, 0xFF)

    # WITH the mask over the HUD range: a HUD-only rewrite (score/timer
    # digits, a pause overlay) must never settle into "a new room".
    masked = RoomFpTransitionSignal(mask_ranges=[HUD_RANGE], settle=3)
    _settle_seq(masked, room, [(0, 0)] * 3)
    for _ in range(10):
        masked.push(hud_only_flip, odo_xy=(0, 0), scene=0)
    assert masked.n_triggers == 0 and masked.n_rooms() == 1

    # ANTI-VACUITY CONTROL: the IDENTICAL mutation, with no mask at all,
    # DOES change the fingerprint and DOES settle as a new room. Without
    # this half, a mask that accidentally zeroed the whole nametable
    # would have passed the first half for the wrong reason.
    unmasked = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    _settle_seq(unmasked, room, [(0, 0)] * 3)
    _settle_seq(unmasked, hud_only_flip, [(0, 0)] * 3)
    assert unmasked.n_triggers == 1
    assert unmasked.last_kind in ROOM_FP_DEFAULT_KIND


# ---------------------------------------------------------------------
# First settle = baseline, never a transition.
# ---------------------------------------------------------------------


def test_first_settle_is_baseline_not_a_transition():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    _settle_seq(sig, _content_nt(9), [(0, 0)] * 3)
    assert sig.n_rooms() == 1            # identity WAS interned (seeded)
    assert sig.last_kind is None         # ... but never classified
    assert sig.last_novel is None
    assert sig.trigger_step is None and sig.n_triggers == 0 and sig.vote() == 0


# ---------------------------------------------------------------------
# A genuine novel transition fires.
# ---------------------------------------------------------------------


def test_a_genuine_novel_transition_fires():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    _settle_seq(sig, _content_nt(1), [(0, 0)] * 3)                     # A
    _settle_seq(sig, _content_nt(2), [(0, 0), (128, 0), (256, 0)])     # A->B, pan E
    assert sig.vote() == 1
    assert sig.last_kind == "pan" and sig.last_novel is True
    assert sig.n_rooms() == 2 and sig.n_triggers == 1


# ---------------------------------------------------------------------
# ROOM TRANSITION is the target, not a false positive -- but a REVISIT
# to an already-discovered room must not spend a second vote.
# ---------------------------------------------------------------------


def test_revisiting_an_already_seen_room_does_not_refire():
    """The measured Kirby shape (3 fires in 24s, every one an ordinary
    room change) is exactly what novel_only exists to tame: walking back
    to a room this instance has already discovered must not vote again,
    or a two-room game with a corridor between them would 'clear' on
    every lap."""
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    room_a, room_b = _content_nt(1), _content_nt(2)
    _settle_seq(sig, room_a, [(0, 0)] * 3)                       # baseline A
    _settle_seq(sig, room_b, [(0, 0), (128, 0), (256, 0)])       # A -> B: novel
    assert sig.n_triggers == 1
    _settle_seq(sig, room_a, [(256, 0), (128, 0), (0, 0)])       # B -> A: revisit
    assert sig.n_triggers == 1                                    # unchanged
    assert sig.last_novel is False


def test_novel_only_false_lets_a_revisit_fire():
    """Documents the escape hatch: with novel_only off, kind is the only
    filter, so the identical revisit above DOES fire."""
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3, novel_only=False)
    room_a, room_b = _content_nt(1), _content_nt(2)
    _settle_seq(sig, room_a, [(0, 0)] * 3)
    _settle_seq(sig, room_b, [(0, 0), (128, 0), (256, 0)])
    assert sig.n_triggers == 1
    _settle_seq(sig, room_a, [(256, 0), (128, 0), (0, 0)])
    assert sig.n_triggers == 2


# ---------------------------------------------------------------------
# DEATH: the Zelda death-flash signature classifies `warp`, excluded by
# default kind -- the clear vote inherits RG-2's adjacency refusal.
# ---------------------------------------------------------------------


def test_a_death_fade_shaped_transition_is_excluded_by_default_kind():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3, warp_scene_min=2)
    _settle_seq(sig, _content_nt(1), [(0, 0)] * 3)     # baseline, scene 0
    # Odometer flat across the churn, scene bumps to 2 -- the measured
    # Zelda death signature (modal 16 -> 272 -> 16, scene +2).
    for scene in (0, 1, 2):
        sig.push(_content_nt(2), odo_xy=(0, 0), scene=scene)
    assert sig.last_kind == "warp" and sig.last_novel is True
    assert sig.vote() == 0 and sig.n_triggers == 0

    # ANTI-VACUITY: a profile that explicitly opts warp back into `kind`
    # DOES fire on the byte-identical stream -- proving the default
    # exclusion above is a real filter, not dead code that never runs.
    opted_in = RoomFpTransitionSignal(mask_ranges=[], settle=3,
                                      warp_scene_min=2, kind=ROOM_FP_KINDS)
    _settle_seq(opted_in, _content_nt(1), [(0, 0)] * 3)
    for scene in (0, 1, 2):
        opted_in.push(_content_nt(2), odo_xy=(0, 0), scene=scene)
    assert opted_in.last_kind == "warp" and opted_in.vote() == 1


# ---------------------------------------------------------------------
# COMBAT BLIP: a transient one-or-two-frame flicker must not settle.
# ---------------------------------------------------------------------


def test_a_transient_flicker_does_not_settle_the_way_a_combat_blip_would():
    """A wave of enemies despawning/respawning together can flicker the
    nametable for a frame or two with no real room change. fp_settle's
    own cancellation rule (a sample matching the settled hash cancels
    the churn) already covers this at the pure-function level
    (tests/test_room_fp.py test_matching_the_settled_hash_cancels_the_
    churn); this pins the guarantee through the wrapper."""
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    room, blip = _content_nt(1), _content_nt(99)
    _settle_seq(sig, room, [(0, 0)] * 3)
    sig.push(blip, odo_xy=(0, 0), scene=0)     # 1-frame flicker
    sig.push(room, odo_xy=(0, 0), scene=0)     # ... and it's gone
    assert sig.n_triggers == 0 and sig.n_rooms() == 1


def test_mid_churn_hashes_never_settle_produces_zero_votes():
    """Clear-vote-wrapper extension of tests/test_room_fp.py's mid-churn
    guarantee: a stream churning through 30 DISTINCT partial-draw hashes,
    none repeating `settle` times, must produce zero clear votes -- not
    just zero settles."""
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    _settle_seq(sig, _content_nt(1), [(0, 0)] * 3)
    for i in range(30):
        sig.push(_content_nt(100 + i), odo_xy=(i * 4, 0), scene=0)
    assert sig.n_triggers == 0 and sig.vote() == 0
    assert sig.n_rooms() == 1        # nothing ever settled long enough to intern


# ---------------------------------------------------------------------
# PAUSE / MENU overlay confined to the mask.
# ---------------------------------------------------------------------


def test_pause_menu_overlay_confined_to_the_mask_creates_no_new_room():
    sig = RoomFpTransitionSignal(mask_ranges=[HUD_RANGE], settle=3)
    room = _content_nt(1)
    _settle_seq(sig, room, [(0, 0)] * 3)
    for digit in range(40):        # a long pause screen, HUD churning
        sig.push(_hud(room, digit), odo_xy=(0, 0), scene=0)
    assert sig.n_triggers == 0 and sig.n_rooms() == 1


# ---------------------------------------------------------------------
# Held-pulse vote shape.
# ---------------------------------------------------------------------


def test_vote_is_a_held_pulse_that_drops_after_hold_observations():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3, hold=5)
    _settle_seq(sig, _content_nt(1), [(0, 0)] * 3)
    _settle_seq(sig, _content_nt(2), [(0, 0), (128, 0), (256, 0)])
    assert sig.vote() == 1
    for _ in range(4):
        sig.push(_content_nt(2), odo_xy=(256, 0), scene=0)  # same settled room
        assert sig.vote() == 1
    sig.push(_content_nt(2), odo_xy=(256, 0), scene=0)
    assert sig.vote() == 0


# ---------------------------------------------------------------------
# Construction-time validation.
# ---------------------------------------------------------------------


def test_invalid_kind_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not_a_kind"):
        RoomFpTransitionSignal(kind=("pan", "not_a_kind"))


# ---------------------------------------------------------------------
# reset() rebuilds the discovery table -- per-episode novelty.
# ---------------------------------------------------------------------


def test_reset_clears_the_discovery_table_for_a_fresh_episode():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3)
    room_a, room_b = _content_nt(1), _content_nt(2)
    _settle_seq(sig, room_a, [(0, 0)] * 3)
    _settle_seq(sig, room_b, [(0, 0), (128, 0), (256, 0)])
    assert sig.n_rooms() == 2 and sig.n_triggers == 1

    sig.reset()
    assert sig.n_rooms() == 0 and sig.trigger_step is None and sig.vote() == 0

    # The SAME room_b hash is baseline (then novel) all over again.
    _settle_seq(sig, room_b, [(0, 0)] * 3)
    assert sig.last_kind is None            # baseline, not a transition
    _settle_seq(sig, room_a, [(0, 0), (128, 0), (256, 0)])
    assert sig.last_novel is True and sig.n_triggers == 1


# ---------------------------------------------------------------------
# A capped discovery table must never fabricate novelty (hold-last).
# ---------------------------------------------------------------------


def test_a_capped_room_index_never_fabricates_novelty():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3, max_rooms=1)
    _settle_seq(sig, _content_nt(1), [(0, 0)] * 3)      # fills the 1-room cap
    _settle_seq(sig, _content_nt(2), [(0, 0), (128, 0), (256, 0)])
    assert sig.last_ordinal is None and sig.last_novel is False
    assert sig.vote() == 0 and sig.n_triggers == 0


# ---------------------------------------------------------------------
# Blank frames (stage wipe / fade-to-black) cancel pending churn.
# ---------------------------------------------------------------------


def test_a_blank_frame_cancels_pending_churn_instead_of_settling_black():
    sig = RoomFpTransitionSignal(mask_ranges=[], settle=3, min_lines=100)
    room = _content_nt(1)
    blank_content = _content_nt(2)
    _settle_seq(sig, room, [(0, 0)] * 3)

    sig.push(blank_content, odo_xy=(0, 0), scene=0)                     # 1
    sig.push(blank_content, odo_xy=(0, 0), scene=0)                     # 2
    sig.push(blank_content, odo_xy=(0, 0), scene=0, rendered_lines=50)  # BLANK: cancels
    assert sig.n_triggers == 0

    sig.push(blank_content, odo_xy=(0, 0), scene=0)      # fresh count: 1
    sig.push(blank_content, odo_xy=(0, 0), scene=0)      # fresh count: 2
    assert sig.n_triggers == 0                            # not yet 3 in a row
    sig.push(blank_content, odo_xy=(0, 0), scene=0)      # fresh count: 3
    assert sig.n_triggers == 1                            # NOW it settles


# ---------------------------------------------------------------------
# Palette co-key.
# ---------------------------------------------------------------------


def test_palette_cokey_distinguishes_identical_nametables_only_when_armed():
    room = _content_nt(1)

    off = RoomFpTransitionSignal(mask_ranges=[], settle=3, palette_cokey=False)
    _settle_seq(off, room, [(0, 0)] * 3)
    for _ in range(3):
        off.push(room, odo_xy=(0, 0), scene=0, palette=bytes([9] * 32))
    assert off.n_rooms() == 1     # palette ignored: same identity throughout

    on = RoomFpTransitionSignal(mask_ranges=[], settle=3, palette_cokey=True)
    for _ in range(3):
        on.push(room, odo_xy=(0, 0), scene=0, palette=bytes(32))
    for _ in range(3):
        on.push(room, odo_xy=(0, 0), scene=0, palette=bytes([9] * 32))
    assert on.n_rooms() == 2      # same NT, different palette: co-keyed identity
