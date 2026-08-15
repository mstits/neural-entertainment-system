"""`progress: {smooth: hampel}` — a robust median3 generalisation that
still rejects the gx-767 tear but, unlike median3, admits a genuine
sustained step-change (a warp / room load) within one sample of delay
instead of aliasing it as noise forever.

See scripts/go_explore_solve.py, progress_glitch's `hampel` branch
(~lines 697-800) for the mechanism, and
tests/test_go_explore_solve_repairs.py for the sibling `borrow`/
`median3` coverage this file mirrors.
"""

from __future__ import annotations

import pytest

from scripts.go_explore_solve import (
    HAMPEL_KEEP,
    PROGRESS_JUMP,
    PROGRESS_SMOOTH_MODES,
    GenericGame,
    progress_glitch,
    push_progress_sample,
)

#: The §12 signature, frame by frame, as (value, hi, lo): the player is
#: walking LEFT past x=512 and $0040 wraps one frame before $0041
#: borrows. Frame 2 is the phantom; 767 is a position never occupied.
PHANTOM = [(513, 2, 1), (512, 2, 0), (767, 2, 255), (511, 1, 255),
           (510, 1, 254)]


def _run_filter(mode, stream, jump=PROGRESS_JUMP, keep=HAMPEL_KEEP):
    """Feed a raw sample stream through the online filter; return the
    indices it rejected."""
    hist, rejected = [], []
    for i, sample in enumerate(stream):
        if progress_glitch(mode, hist, sample, jump):
            rejected.append(i)
        push_progress_sample(hist, sample, keep=keep)
    return rejected


def test_hampel_is_listed_as_a_progress_smooth_mode():
    assert "hampel" in PROGRESS_SMOOTH_MODES


def test_the_reconstructed_gx_767_tear_is_rejected():
    assert _run_filter("hampel", PHANTOM) == [2]


def test_a_sustained_warp_is_admitted_with_at_most_one_sample_delay():
    # A room load / warp: the observable jumps far away and STAYS there —
    # unlike the phantom, which reverts on the very next frame.
    stream = [700, 704, 708, 712, 5000, 5010, 5020, 5030, 5040]
    warp_start = stream.index(5000)
    rejected = _run_filter("hampel", [(v, None, None) for v in stream])
    # median3 would keep rejecting every one of these as "not the local
    # median" forever; hampel must not reject anything from one sample
    # after the warp begins onward.
    assert all(i <= warp_start + 1 for i in rejected)
    assert (warp_start + 1) not in rejected


def test_ordinary_motion_is_never_called_torn():
    stream = [(v, v >> 8, v & 0xFF) for v in range(700, 780, 4)]
    assert _run_filter("hampel", stream) == []


def test_ordinary_motion_with_a_single_byte_progress_is_never_called_torn():
    stream = [(v, None, None) for v in range(10, 90, 3)]
    assert _run_filter("hampel", stream) == []


def test_hampel_withholds_judgement_until_it_has_two_prior_samples():
    assert _run_filter("hampel", [(767, 2, 255)]) == []
    assert _run_filter("hampel", [(512, 2, 0), (767, 2, 255)]) == []


def test_hampel_is_judged_against_the_raw_stream_not_the_kept_one():
    hist = []
    for s in PHANTOM[:3]:
        push_progress_sample(hist, s, keep=HAMPEL_KEEP)
    assert hist[-1] == (767, 2, 255)
    assert not progress_glitch("hampel", hist, PHANTOM[3])


def test_a_bigger_jump_is_never_less_permissive_for_hampel():
    rejects = [len(_run_filter("hampel", PHANTOM, jump=j))
               for j in (1, 32, 128, 200, 254, 1000)]
    assert rejects == sorted(rejects, reverse=True)
    assert rejects[-1] == 0


def test_a_two_sample_reversion_is_not_mistaken_for_a_sustained_warp():
    # A tear that happens to repeat for exactly one extra frame before
    # snapping back must not be laundered by the persistence check: the
    # THIRD sample reverts to the old baseline instead of continuing the
    # new one, so it is not "close to the previous sample" and the
    # window rule applies to it on its own terms.
    stream = [700, 704, 708, 5000, 5005, 712, 716]
    rejected = _run_filter("hampel", [(v, None, None) for v in stream])
    assert 3 in rejected     # 5000: first sample of the excursion
    assert 5 not in rejected  # 712: reverts smoothly, no false rejection


def test_hampel_default_off_and_configurable_via_profile():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"lo": 0x0040, "hi": 0x0041,
                                   "smooth": "hampel"},
                      "y": 0x3F, "level_key": [0x28], "lives": 0x2A}}
    g = GenericGame(prof)
    assert g.progress_smooth == "hampel"


def test_an_unknown_smooth_mode_is_still_refused_at_profile_load():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"lo": 0x0040, "smooth": "kalman"},
                      "y": 0x3F, "level_key": [0x28], "lives": 0x2A}}
    with pytest.raises(SystemExit, match="unknown progress smooth mode"):
        GenericGame(prof)
