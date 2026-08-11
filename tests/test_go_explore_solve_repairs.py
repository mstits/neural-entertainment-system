"""The three owed solver repairs from GATE_OPENER_CAMPAIGN_2026-08-11.

Kept out of tests/test_go_explore_solve.py so these land as one file
rather than as a hunk in the middle of that file's gate sections.

1. --resume-archive LINEAGE CHECK (§12). Resuming an archive built on a
   different machine, or under a different cell-key schema, was silent.
   runs/cv_chain_hw/lvl_03_overnight was cited as the deepest banked
   attack on the Castlevania hall — 560,410 cells — and was built under
   ONE hw flag against the arms' four, over a disjoint tb/kk key
   subspace; 560,410 collapsed to 88,212 on a like-for-like recount.

2. TORN MULTI-BYTE PROGRESS READS (§12, "THE gx-767 PIN IS A PHANTOM").
   All 35 band-95 cells across five archives held a one-frame
   non-atomic 16-bit borrow: $0040 wraps 0x00 -> 0xFF one frame before
   $0041 borrows 2 -> 1, so gx reads 0x02FF = 767 while the player is
   moving LEFT past 512. Because `score = sect*10000 + gx`, that single
   frame dominated its cell and topped every archive on the wall. True
   frontier: 751.

3. c_local TELEMETRY (docs/receipts/dispatch/size_decoupled_statistic_
   2026-08-11.md §11). The hall's real signature is a derivative —
   distinct_spatial 932 -> 1,102 (+18%) while cells went 28,929 ->
   560,410 (19x) — and a banked archive holds exactly one reading of
   it. 22 cross-sectional surrogates were scored and none separated.
"""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest

from scripts.go_explore_solve import (
    GX_BUCKET,
    LINEAGE_KEY_AXES,
    PROGRESS_JUMP,
    GenericGame,
    Solver,
    coverage_elasticity,
    format_lineage_report,
    hw_provenance,
    key_config_axes,
    local_coverage,
    local_coverage_band,
    progress_glitch,
    push_progress_sample,
    resume_lineage_diff,
    stamp_stats_provenance,
    state_sig_sha,
)

# ---------------------------------------------------------------------
# 1. --resume-archive lineage check
# ---------------------------------------------------------------------

#: The four flags the T1 arms ran under (campaign §12 / the root manifest
#: runs/cv_t1_roots_2026-08-11.json resume_archive.hw_provenance).
ARM_FLAGS = ["reset_alignment", "mmio_read_timing", "dmc_stall_timing",
             "nmi_poll_timing"]


def _prov(flags=ARM_FLAGS, frame_skip=4, sha="e09e8191b8d40490") -> dict:
    return {"hw_flags": list(flags), "frame_skip": frame_skip,
            "nes_core": {"dist_version": "0.1.0",
                         "module": "nes_core.abi3.so", "sha256_16": sha}}


def _keycfg(**over) -> dict:
    # sig_sha is the CV profile's real digest, so a fixture and a live
    # adapter agree (test_the_key_config_a_run_banks_... round-trips it).
    base = {"tb": 0, "kk": 1, "sig_arity": 1,
            "sig_sha": state_sig_sha([(0x0004, frozenset({8}), 0)]),
            "gx_bucket": 8, "y_band": 16, "cell_fn": "GenericGame"}
    base.update(over)
    return base


def _stats(prov=None, key=None, **over) -> dict:
    s = {"cells": 149153, "records": 9052551, "best_score": 767}
    if prov is not None:
        s["hw_provenance"] = prov
    if key is not None:
        s["key_config"] = key
    s.update(over)
    return s


def test_a_matching_lineage_reports_nothing_at_all():
    diff = resume_lineage_diff(_stats(_prov(), _keycfg()), _prov(), _keycfg())
    assert diff == {"mismatch": [], "unverifiable": [], "caveats": []}


def test_hw_flag_order_is_not_lineage_but_the_set_is():
    # The resolver preserves declaration order, so two runs that named the
    # same four flags differently are the SAME machine and must not warn.
    shuffled = list(reversed(ARM_FLAGS))
    assert resume_lineage_diff(_stats(_prov(shuffled), _keycfg()),
                               _prov(), _keycfg())["mismatch"] == []
    one = resume_lineage_diff(_stats(_prov(["nmi_poll_timing"]), _keycfg()),
                              _prov(), _keycfg())["mismatch"]
    assert len(one) == 1 and "hw_flags" in one[0]


def test_the_lvl_03_overnight_shape_is_exactly_what_the_check_refuses():
    # The §12 verdict, as data: a 1-flag lineage resumed into the 4-flag
    # arms over a disjoint tb/kk key subspace. Both halves must fire, and
    # each must name which axis moved — "incompatible" with no axis is a
    # verdict nobody can act on.
    overnight = _stats(_prov(["nmi_poll_timing"]),
                       _keycfg(tb=1, kk=16, sig_arity=3, sig_sha="0badc0de"))
    diff = resume_lineage_diff(overnight, _prov(), _keycfg())
    assert diff["unverifiable"] == []
    blob = " | ".join(diff["mismatch"])
    assert "hw_flags" in blob
    assert "time-bin" in blob and "kill-count" in blob and "state_sig" in blob
    # hw_flags + tb + kk + the sig, counted BY CONTENT as well as by bit
    # count (a 3-axis sig cannot share a 1-axis sig's digest).
    assert len(diff["mismatch"]) == 5


@pytest.mark.parametrize("axis,value", [("tb", 1), ("kk", 16),
                                        ("sig_arity", 2),
                                        ("sig_sha", "0badc0de"),
                                        ("gx_bucket", 16),
                                        ("y_band", 32),
                                        ("cell_fn", "SmbGame")])
def test_every_declared_key_axis_is_actually_compared(axis, value):
    diff = resume_lineage_diff(_stats(_prov(), _keycfg(**{axis: value})),
                               _prov(), _keycfg())
    assert len(diff["mismatch"]) == 1, axis
    assert repr(value) in diff["mismatch"][0]


def test_frame_skip_and_the_core_binary_are_machine_lineage_too():
    assert resume_lineage_diff(_stats(_prov(frame_skip=1), _keycfg()),
                               _prov(), _keycfg())["mismatch"]
    assert resume_lineage_diff(_stats(_prov(sha="deadbeefdeadbeef"),
                                      _keycfg()),
                               _prov(), _keycfg())["mismatch"]


def test_a_legacy_sidecar_is_unverifiable_and_never_reported_as_agreement():
    # Every archive on disk predates `key_config`. "Not recorded" is a
    # weaker finding than "matches" and has to read as such, or the check
    # launders silence into a clean bill of health.
    diff = resume_lineage_diff(_stats(_prov()), _prov(), _keycfg())
    assert diff["mismatch"] == []
    assert len(diff["unverifiable"]) == 1
    assert "key_config" in diff["unverifiable"][0]
    both = resume_lineage_diff({}, _prov(), _keycfg())
    assert both["mismatch"] == [] and len(both["unverifiable"]) == 2


def test_a_partially_recorded_key_config_reports_only_the_gaps_it_has():
    partial = _keycfg()
    del partial["sig_arity"]
    partial["tb"] = 1
    diff = resume_lineage_diff(_stats(_prov(), partial), _prov(), _keycfg())
    assert len(diff["mismatch"]) == 1 and "time-bin" in diff["mismatch"][0]
    assert len(diff["unverifiable"]) == 1
    assert "sig_arity" in diff["unverifiable"][0]


def test_the_report_offers_the_override_only_when_there_is_one_to_offer():
    clean = {"mismatch": [], "unverifiable": ["key_config: absent"]}
    assert "--allow-lineage-mismatch" not in format_lineage_report(
        "d", clean, False)
    bad = {"mismatch": ["hw_flags: archive [] vs run [x]"], "unverifiable": []}
    assert "--allow-lineage-mismatch" in format_lineage_report("d", bad, False)
    assert "proceeding" in format_lineage_report("d", bad, True)


def test_the_key_config_a_run_banks_is_the_one_a_resume_compares(tmp_path):
    # Round trip through the real sidecar writer: whatever key_config_axes
    # produces must survive JSON and compare equal to itself, or the check
    # fires on every resume of our own archives.
    args = SimpleNamespace(time_bins=True, kill_key=False, gx_bucket=8,
                           y_band=16)
    game = _cv_game()
    cfg = key_config_axes(args, game)
    prov = hw_provenance(ARM_FLAGS, 4)
    (tmp_path / "archive.stats.json").write_text(json.dumps({"cells": 3}))
    stamp_stats_provenance(tmp_path / "archive.pkl", prov, cfg)
    banked = json.loads((tmp_path / "archive.stats.json").read_text())
    assert banked["cells"] == 3          # the archive's own stats survive
    assert banked["key_config"] == cfg
    assert resume_lineage_diff(banked, prov, cfg) == {
        "mismatch": [], "unverifiable": [], "caveats": []}


def test_key_config_records_cardinalities_so_the_flags_can_be_renamed():
    off = key_config_axes(SimpleNamespace(time_bins=False, kill_key=False,
                                          gx_bucket=16, y_band=32), _cv_game())
    on = key_config_axes(SimpleNamespace(time_bins=True, kill_key=True,
                                         gx_bucket=16, y_band=32), _cv_game())
    assert (off["tb"], off["kk"]) == (0, 1)
    assert (on["tb"], on["kk"]) == (1, 16)
    # The merged --gate-axes sidecar lands in state_sig before make_game,
    # so the arity travels with the archive rather than with the flag.
    assert off["sig_arity"] == 1


def _sig_profile(sig):
    p = _cv_profile()
    p["solve"]["state_sig"] = sig
    return GenericGame(p)


def _sig_cfg(sig):
    return key_config_axes(SimpleNamespace(time_bins=False, kill_key=False,
                                           gx_bucket=8, y_band=16),
                           _sig_profile(sig))


def test_two_one_bit_sigs_over_different_bytes_are_not_the_same_lineage():
    # The arity-only hole: --gate-axes MERGES its axes into state_sig
    # before the adapter reads it, so "one bit over $0004" and "one bit
    # over $0090" are the ordinary pair of arms, identical in arity and
    # keying different spaces. Counting bits would call them one lineage.
    a = _sig_cfg([{"addr": 0x0004, "match": [8]}])
    b = _sig_cfg([{"addr": 0x0090, "match": [8]}])
    assert a["sig_arity"] == b["sig_arity"] == 1
    assert a["sig_sha"] != b["sig_sha"]
    diff = resume_lineage_diff(_stats(_prov(), a), _prov(), b)
    assert len(diff["mismatch"]) == 1
    assert "state_sig contents" in diff["mismatch"][0]


def test_the_sig_digest_is_content_ordered_not_cosmetic():
    # Axis i owns bit i, so ORDER is lineage; a re-listed `match:` set is
    # the same predicate and must not be.
    base = [{"addr": 0x0004, "match": [8, 9]},
            {"addr": 0x0090, "match": [1]}]
    assert _sig_cfg(base)["sig_sha"] == _sig_cfg(
        [{"addr": 0x0004, "match": [9, 8]}, {"addr": 0x0090, "match": [1]}]
    )["sig_sha"]
    assert _sig_cfg(base)["sig_sha"] != _sig_cfg(list(reversed(base)))["sig_sha"]
    # A `mod:` phase test over the same address is a different predicate.
    assert (_sig_cfg([{"addr": 0x001A, "match": [7], "mod": 16}])["sig_sha"]
            != _sig_cfg([{"addr": 0x001A, "match": [7]}])["sig_sha"])
    # No sig at all is the empty digest, not a hash of nothing.
    assert _sig_cfg([])["sig_sha"] == ""


def test_state_sig_sha_is_a_pure_function_of_the_spec():
    spec = [(0x0004, frozenset({8, 9}), 0)]
    assert state_sig_sha(spec) == state_sig_sha(
        [(4, frozenset({9, 8}), 0)])
    assert len(state_sig_sha(spec)) == 8
    assert state_sig_sha([]) == "" and state_sig_sha(None) == ""


def test_resuming_an_unfiltered_archive_into_a_filtered_run_is_said_out_loud():
    # The filter only screens NEW samples. Every hall archive already has
    # the phantom banked (§12), and the resume that arms the filter is
    # exactly the one the campaign wants to run — so this warns and never
    # refuses, or the repair would refuse itself.
    args = SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=8,
                           y_band=16)
    unfiltered = key_config_axes(args, _cv_game())
    filtered = key_config_axes(args, _cv_game(smooth="borrow"))
    assert unfiltered["smooth"] == "off" and filtered["smooth"] == "borrow"
    diff = resume_lineage_diff(_stats(_prov(), unfiltered), _prov(), filtered)
    assert diff["mismatch"] == [] and diff["unverifiable"] == []
    assert len(diff["caveats"]) == 1
    assert "progress smoothing" in diff["caveats"][0]
    # And it is not an axis anyone can be refused on. CAVEATS is its own
    # tier for exactly that reason: UNVERIFIABLE is a refusal now (an
    # axis nothing recorded is not an axis that matches), and a torn-read
    # rule that changed is neither a key meaning nor a machine.
    assert "smooth" not in dict(LINEAGE_KEY_AXES)


def test_the_filter_caveat_is_silent_on_every_unfiltered_resume():
    # Default runs and every archive banked before the field existed:
    # absent reads as off on both sides, so no line appears.
    args = SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=8,
                           y_band=16)
    cfg = key_config_axes(args, _cv_game())
    assert resume_lineage_diff(_stats(_prov(), cfg), _prov(), cfg) == {
        "mismatch": [], "unverifiable": [], "caveats": []}
    legacy = {k: v for k, v in cfg.items() if k != "smooth"}
    assert resume_lineage_diff(_stats(_prov(), legacy), _prov(),
                               cfg)["caveats"] == []


def test_cosmetic_run_differences_are_not_lineage():
    # seed/workers/minutes/out-dir change nothing about what a cell means
    # or what a blob restores into; a check that fires on them is a check
    # operators learn to pass --allow past reflexively.
    args = SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=8,
                           y_band=16, seed=7, workers=8, minutes=90,
                           out="/tmp/other")
    cfg = key_config_axes(args, _cv_game())
    assert resume_lineage_diff(_stats(_prov(), cfg), _prov(),
                               cfg)["mismatch"] == []


def _resume_solver(tmp_path, *, allow=False, flags=ARM_FLAGS,
                   allow_unverified=False):
    f = SimpleNamespace(
        args=SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=8,
                             y_band=16, allow_lineage_mismatch=allow,
                             allow_unverified_lineage=allow_unverified),
        game=_cv_game(), provenance=hw_provenance(flags, 4))
    for name in ("key_config", "check_resume_lineage", "audit_resumed_keys"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


def test_a_mismatched_resume_is_refused_before_the_archive_is_opened(tmp_path):
    # SystemExit, not a warning: the campaign's hole was that this ran.
    # And it must fire off the sidecar alone — there is deliberately no
    # archive.pkl here, so a check that unpickled first would crash
    # differently and prove nothing.
    (tmp_path / "archive.stats.json").write_text(json.dumps(
        _stats(_prov(["nmi_poll_timing"]), _keycfg(tb=1))))
    f = _resume_solver(tmp_path)
    with pytest.raises(SystemExit) as e:
        f.check_resume_lineage(tmp_path)
    assert "--allow-lineage-mismatch" in str(e.value)


def test_allow_lineage_mismatch_is_the_only_way_past_it(tmp_path, capsys):
    (tmp_path / "archive.stats.json").write_text(json.dumps(
        _stats(_prov(["nmi_poll_timing"]), _keycfg(tb=1))))
    diff = _resume_solver(tmp_path, allow=True).check_resume_lineage(tmp_path)
    assert len(diff["mismatch"]) == 2
    err = capsys.readouterr().err
    assert "RESUME LINEAGE MISMATCH" in err and "proceeding" in err


def test_a_clean_resume_says_nothing_and_a_legacy_one_is_refused(
        tmp_path, capsys):
    cfg = _resume_solver(tmp_path).key_config()
    (tmp_path / "archive.stats.json").write_text(
        json.dumps(_stats(_prov(), cfg)))
    assert _resume_solver(tmp_path).check_resume_lineage(tmp_path) == {
        "mismatch": [], "unverifiable": [], "caveats": []}
    out = capsys.readouterr()
    assert out.err == "" and "lineage" not in out.out

    # A recorded machine and an UNRECORDED key schema: half the check
    # could not run, which is not half a pass. The first build warned and
    # resumed here, which is how the §12 archive got in.
    (tmp_path / "archive.stats.json").write_text(json.dumps(_stats(_prov())))
    with pytest.raises(SystemExit) as e:
        _resume_solver(tmp_path).check_resume_lineage(tmp_path)
    assert "--allow-unverified-lineage" in str(e.value)
    assert "UNVERIFIABLE" in capsys.readouterr().err
    diff = _resume_solver(tmp_path, allow_unverified=True
                          ).check_resume_lineage(tmp_path)
    assert diff["mismatch"] == [] and len(diff["unverifiable"]) == 1


def test_an_unreadable_sidecar_degrades_loudly_instead_of_crashing(tmp_path):
    (tmp_path / "archive.stats.json").write_text("{not json")
    with pytest.raises(SystemExit):
        _resume_solver(tmp_path).check_resume_lineage(tmp_path)
    diff = _resume_solver(tmp_path, allow_unverified=True
                          ).check_resume_lineage(tmp_path)
    assert diff["mismatch"] == [] and len(diff["unverifiable"]) == 2


# ---------------------------------------------------------------------
# 2. torn multi-byte progress reads (the gx-767 phantom)
# ---------------------------------------------------------------------

#: The §12 signature, frame by frame, as (value, hi, lo): the player is
#: walking LEFT past x=512 and $0040 wraps one frame before $0041
#: borrows. Frame 2 is the phantom; 767 is a position never occupied.
PHANTOM = [(513, 2, 1), (512, 2, 0), (767, 2, 255), (511, 1, 255),
           (510, 1, 254)]


def _run_filter(mode, stream, jump=PROGRESS_JUMP):
    """Feed a raw sample stream through the online filter; return the
    indices it rejected."""
    hist, rejected = [], []
    for i, sample in enumerate(stream):
        if progress_glitch(mode, hist, sample, jump):
            rejected.append(i)
        push_progress_sample(hist, sample)
    return rejected


@pytest.mark.parametrize("mode", ["borrow", "median3"])
def test_the_phantom_frame_is_the_only_one_either_filter_rejects(mode):
    assert _run_filter(mode, PHANTOM) == [2]


def test_the_filter_is_inert_when_it_is_off():
    assert _run_filter("off", PHANTOM) == []


@pytest.mark.parametrize("stream", [
    # walking RIGHT across the same page boundary, atomically
    [(510, 1, 254), (511, 1, 255), (512, 2, 0), (513, 2, 1)],
    # walking LEFT across it, atomically — the borrow the phantom apes
    [(513, 2, 1), (512, 2, 0), (511, 1, 255), (510, 1, 254)],
])
def test_an_atomic_page_crossing_is_not_a_torn_read(stream):
    assert _run_filter("borrow", stream) == []


def test_ordinary_motion_is_never_called_torn():
    stream = [(v, v >> 8, v & 0xFF) for v in range(700, 780, 4)]
    assert _run_filter("borrow", stream) == []
    assert _run_filter("median3", stream) == []


def test_borrow_needs_both_bytes_and_says_so_by_staying_silent():
    # A single-byte or HUD-decoded observable reports (value, None, None);
    # there is no carry to tear, so the rule must be inert rather than
    # guess. (The profile parser refuses the combination up front too.)
    stream = [(v, None, None) for v in (10, 11, 250, 12)]
    assert _run_filter("borrow", stream) == []
    assert _run_filter("median3", stream) == [2]


def test_median3_withholds_judgement_until_it_has_three_samples():
    assert _run_filter("median3", [(767, 2, 255)]) == []
    assert _run_filter("median3", [(512, 2, 0), (767, 2, 255)]) == []


def test_a_torn_read_is_judged_against_the_raw_stream_not_the_kept_one():
    # The sample AFTER the phantom is compared against the phantom, which
    # is what the game's bytes actually did. Dropping rejects from the
    # history would compare 511 against 512 and, on the borrow rule, read
    # the lo wrap as a second tear.
    hist = []
    for s in PHANTOM[:3]:
        push_progress_sample(hist, s)
    assert hist[-1] == (767, 2, 255)
    assert not progress_glitch("borrow", hist, PHANTOM[3])


def test_the_history_never_grows_past_what_the_rules_read():
    hist = []
    for i in range(50):
        push_progress_sample(hist, (i, 0, i))
    assert len(hist) == 2 and hist[-1] == (49, 0, 49)


def test_the_jump_threshold_is_the_knob_and_it_is_honoured():
    small = [(20, 0, 20), (0, 0, 0), (200, 0, 200)]
    assert _run_filter("median3", small, jump=PROGRESS_JUMP) == [2]
    assert _run_filter("median3", small, jump=250) == []


@pytest.mark.parametrize("mode", ["borrow", "median3"])
def test_a_bigger_jump_is_always_more_permissive_never_less(mode):
    # The knob reads "how far the observable may move before we may call
    # a sample torn", so raising it must only ever REMOVE rejections. On
    # the borrow rule it is a low-byte distance, so an inverted threshold
    # would turn "be permissive" into "reject everything".
    rejects = [len(_run_filter(mode, PHANTOM, jump=j))
               for j in (1, 32, 128, 200, 254, 1000)]
    assert rejects == sorted(rejects, reverse=True)
    assert rejects[0] >= 1
    # median3 relaxes all the way out; borrow saturates at its byte clamp,
    # which is the point of the clamp and is pinned by its own test.
    assert rejects[-1] == (0 if mode == "median3" else 1)


def test_a_borrow_threshold_is_clamped_into_the_byte_it_is_about():
    # jump 255 would demand a low-byte delta of 256, which cannot happen,
    # so the rule would silently never fire. Clamped to 254 it still
    # catches the full 0x00 -> 0xFF wrap and nothing else.
    assert _run_filter("borrow", PHANTOM, jump=255) == [2]
    assert _run_filter("borrow", [(v, 0, v) for v in (0, 253, 0)],
                       jump=255) == []


def test_a_zero_width_tolerance_is_refused_at_profile_load():
    with pytest.raises(SystemExit, match="progress.jump must be >= 1"):
        _cv_game(smooth="median3", jump=0)


# --- the profile surface ---------------------------------------------

def _cv_profile(**progress):
    p = {"lo": 0x0040, "hi": 0x0041}
    p.update(progress)
    return {"solve": {"rom": "roms/x.nes", "progress": p, "y": 0x003F,
                      "level_key": [0x0028], "lives": 0x002A,
                      "state_sig": [{"addr": 0x0004, "match": [8]}]}}


def _cv_game(**progress) -> GenericGame:
    return GenericGame(_cv_profile(**progress))


def test_the_filter_is_off_unless_a_profile_asks_for_it():
    g = _cv_game()
    assert g.progress_smooth == "off" and g.progress_jump == PROGRESS_JUMP
    assert _cv_game(smooth="borrow", jump=64).progress_jump == 64


def test_an_unknown_smooth_mode_is_refused_at_profile_load():
    with pytest.raises(SystemExit, match="unknown progress smooth mode"):
        _cv_game(smooth="kalman")


def test_borrow_on_a_one_byte_progress_is_refused_instead_of_silently_inert():
    prof = {"solve": {"rom": "roms/x.nes",
                      "progress": {"lo": 0x0040, "smooth": "borrow"},
                      "y": 0x3F, "level_key": [0x28], "lives": 0x2A}}
    with pytest.raises(SystemExit, match="two-byte progress read"):
        GenericGame(prof)


def test_progress_pair_reports_the_bytes_the_value_was_composed_from():
    g = _cv_game()
    ram = bytearray(2048)
    ram[0x0041], ram[0x0040] = 2, 0xFF
    assert g.progress(ram) == 767
    assert g.progress_pair(ram) == (767, 2, 255)
    hud = GenericGame({"solve": {"rom": "r", "y": 1, "level_key": [2],
                                 "lives": 3,
                                 "progress": {"tiles": [0x10, 0x11]}}})
    assert hud.progress_pair(bytearray(2048)) == (0, None, None)


# --- end to end through observe() ------------------------------------

class _KeyArchive:
    """Archive stand-in that keeps keys, so a test can ask whether the
    phantom's bucket ever entered the archive at all."""

    def __init__(self) -> None:
        self.cells: dict = {}

    def __len__(self) -> int:
        return len(self.cells)

    def record(self, ram, state, score, steps, key=None) -> bool:
        new = key not in self.cells
        self.cells[key] = SimpleNamespace(key=key, best_score=score,
                                          best_steps=steps, state=state)
        return new


def _observe_solver(smooth: str):
    f = SimpleNamespace(
        game=_cv_game(smooth=smooth) if smooth != "off" else _cv_game(),
        archive=_KeyArchive(), traces={}, roots={},
        pool=SimpleNamespace(save_worker_state=lambda wid: b"blob"),
        start_wd=(0,), start_lives=5,
        max_area=0, max_gx_in_area={}, max_sect=0, _pin_time=0.0,
        ortho_mode="off", gate_mode="off", door_weight=0.0,
        time_bins=False, kill_key=False, _recorded_new=False,
        progress_smooth=smooth, progress_jump=PROGRESS_JUMP,
        _prog_glitches=0, _c_local=set())
    f.observe = MethodType(Solver.observe, f)
    return f


def _walk(f, values):
    """Drive observe() over a left-walk whose raw bytes are `values`,
    one burst, one ctx — the way explore() does."""
    ctx: dict = {}
    for step, v in enumerate(values):
        ram = bytearray(2048)
        ram[0x002A] = 5                       # lives, so is_dead is False
        ram[0x0041], ram[0x0040] = v >> 8, v & 0xFF
        f.observe(0, ram, [], step, "entrance", ctx=ctx)
    return ctx


LEFT_WALK = [513, 512, 767, 511, 510]        # PHANTOM, as raw values


def test_the_default_path_still_archives_the_phantom_and_crowns_it():
    # The control half: this is the defect, reproduced. Without the
    # filter the torn frame enters the archive as its own cell AND, at
    # score = gx, becomes the best-scoring cell on the wall — which is
    # how one frame came to define a 10.7-hour frontier.
    f = _observe_solver("off")
    _walk(f, LEFT_WALK)
    buckets = {k[-1] for k in f.archive.cells}
    assert 767 // GX_BUCKET in buckets
    assert f.max_gx_in_area[0] == 767
    best = max(f.archive.cells.values(), key=lambda c: c.best_score)
    assert best.key[-1] == 767 // GX_BUCKET


@pytest.mark.parametrize("mode", ["borrow", "median3"])
def test_the_filter_keeps_the_phantom_out_of_the_archive_entirely(mode):
    f = _observe_solver(mode)
    _walk(f, LEFT_WALK)
    assert 767 // GX_BUCKET not in {k[-1] for k in f.archive.cells}
    assert f.max_gx_in_area[0] == 513         # the true frontier, not 767
    assert f._prog_glitches == 1
    # and every honest frame still got archived
    assert {k[-1] for k in f.archive.cells} == {
        v // GX_BUCKET for v in (513, 512, 511, 510)}


def test_a_rejected_frame_does_not_move_the_pin_clock():
    # _pin_time is the arming clock for the inversion/ortho arms; a
    # phantom frontier advance would reset it and keep them disarmed.
    f = _observe_solver("borrow")
    f._pin_time = 0.0
    _walk(f, [513, 512, 767])
    assert f._pin_time > 0.0                  # 513 legitimately moved it
    pinned = f._pin_time
    _walk(f, [512, 767])                      # fresh ctx, no real advance
    assert f._pin_time == pinned


def test_a_rejected_frame_is_live_so_the_lineage_keeps_stepping():
    # "dead" would abandon the burst at the tear; the frame is merely
    # unrecordable, exactly like a progress_cap garbage read.
    f = _observe_solver("borrow")
    ctx: dict = {}
    ram = bytearray(2048)
    ram[0x002A] = 5
    for v in (513, 512):
        ram[0x0041], ram[0x0040] = v >> 8, v & 0xFF
        f.observe(0, ram, [], 0, "entrance", ctx=ctx)
    ram[0x0041], ram[0x0040] = 2, 255
    assert f.observe(0, ram, [], 2, "entrance", ctx=ctx) == "live"


def test_the_default_path_never_even_reads_the_second_byte():
    # Default-inert, measured rather than asserted: with the filter off,
    # observe() must not so much as call progress_pair. This is the
    # byte-identity guarantee every banked run's receipts depend on.
    f = _observe_solver("off")
    calls = []
    real = f.game.progress_pair
    f.game.progress_pair = lambda ram: (calls.append(1), real(ram))[1]
    _walk(f, LEFT_WALK)
    assert calls == []
    assert f._prog_glitches == 0


def test_the_solver_resolves_the_mode_off_its_adapter_and_short_circuits():
    # The real constructor needs a ROM and a Pool, so pin the two lines
    # that make the knob reachable at all: __init__ takes the mode from
    # the adapter (not from a flag — it is a profile fact), and observe
    # tests the mode BEFORE it touches the game or the ctx.
    import inspect

    init = inspect.getsource(Solver.__init__)
    assert 'getattr(self.game, "progress_smooth", "off")' in init
    assert 'getattr(self.game, "progress_jump"' in init
    body = inspect.getsource(Solver.observe)
    head = body.split('smooth = getattr(self, "progress_smooth", "off")', 1)[1]
    assert head.lstrip().startswith('if smooth != "off" and ctx is not None:')


def test_each_burst_filters_its_own_stream():
    # The history rides on the burst's ctx, so a re-root — where a
    # positional observable legitimately teleports — never looks torn,
    # and two workers never contaminate each other's pairing.
    f = _observe_solver("borrow")
    _walk(f, [513, 512])
    _walk(f, [3000, 3001])                    # different worker/root
    assert f._prog_glitches == 0


# ---------------------------------------------------------------------
# 3. c_local telemetry
# ---------------------------------------------------------------------

def _key(area, y, gx, *, sect=0, tb=0, kk=0, loops=0):
    """A full solver cell key: (sect, tb, kk, psig, loops, route_sig)
    + (area, hp, sig, y_band, gx_bucket)."""
    return (sect, tb, kk, (), loops, ()) + (area, 0, 0, y, gx)


def test_c_local_counts_local_positions_and_not_key_space():
    # The §12 collapse, in miniature: 12 cells that differ only in key
    # PREFIX axes are one place. This is why `cells` cannot stand in for
    # the footprint, and why 560,410 became 88,212.
    keys = [_key(0, 3, 40, tb=t, kk=k, loops=lp)
            for t in range(2) for k in range(3) for lp in range(2)]
    assert len(keys) == 12
    assert local_coverage(keys) == {(0, 3, 40)}


def test_c_local_separates_area_band_and_bucket():
    keys = [_key(0, 3, 40), _key(1, 3, 40), _key(0, 4, 40), _key(0, 3, 41)]
    assert len(local_coverage(keys)) == 4
    assert local_coverage([]) == set()


def test_the_band_series_is_the_frontier_slice_of_the_footprint():
    triples = {(0, 1, gx) for gx in range(100)}
    assert local_coverage_band(triples, band=24) == 25   # buckets 75..99
    assert local_coverage_band(triples, band=0) == 1
    assert local_coverage_band(set()) == 0


def test_the_band_slice_follows_the_frontier_rather_than_a_constant():
    near = {(0, 1, 10), (0, 1, 11)}
    far = near | {(0, 1, 900)}
    assert local_coverage_band(near, band=24) == 2
    assert local_coverage_band(far, band=24) == 1        # the two fell out


def test_the_elasticity_reproduces_the_halls_own_signature():
    # distinct_spatial 932 -> 1,102 while cells 28,929 -> 560,410: the
    # footprint is nearly frozen while the key space explodes 19x. That
    # ratio is the statistic the receipt says nobody recorded.
    e = coverage_elasticity(932, 28_929, 1_102, 560_410)
    assert 0.0 < e < 0.1
    assert round(e, 4) == 0.0565


def test_a_search_that_is_actually_covering_ground_reads_near_one():
    assert coverage_elasticity(100, 1_000, 200, 2_000) == pytest.approx(1.0)


@pytest.mark.parametrize("args", [
    (None, 100, 10, 200),          # no previous window
    (10, 100, 10, 100),            # no new cells: log ratio undefined
    (0, 100, 10, 200),             # a footprint that had not started
    (10, 0, 20, 200),
])
def test_an_undefined_elasticity_is_none_and_never_a_flat_zero(args):
    # "not measurable yet" and "measured flat" are the two readings the
    # series exists to tell apart; a silent 0.0 fakes the plateau.
    assert coverage_elasticity(*args) is None


def _line_solver(tmp_path, n_cells, triples):
    return SimpleNamespace(
        archive=list(range(n_cells)),
        _stall={"last_cells": 0, "last_t": 0.0, "flat_windows": 0},
        max_area=0, max_gx_in_area={}, max_sect=0,
        n_solutions=0, best_sol_len=0, steps_done=100,
        door_weight=0, transition_macros=[], ortho_mode="off",
        _c_local=set(triples), _c_local_prev=None, out=tmp_path)


def _lines(tmp_path):
    return [json.loads(x) for x in
            (tmp_path / "progress.jsonl").read_text().splitlines()]


def test_the_progress_line_carries_the_series_on_the_default_path(tmp_path):
    # Unconditional, unlike every arm's telemetry: the field cannot be
    # back-filled from a banked archive, so a run that omits it is a run
    # that can never be scored. The name and definition are the ones
    # wall_taxonomy.ProgressRecord already reserves.
    f = _line_solver(tmp_path, 5, {(0, 1, 2), (0, 1, 3)})
    Solver.progress_line(f, 10.0)
    line = _lines(tmp_path)[-1]
    assert line["c_local"] == 2
    assert line["c_local_band"] == 2
    assert line["c_local_delta"] is None          # first window
    assert line["c_local_band_delta"] is None
    assert line["c_local_elasticity"] is None


def test_the_second_window_reports_real_derivatives(tmp_path):
    f = _line_solver(tmp_path, 100, {(0, 1, i) for i in range(10)})
    Solver.progress_line(f, 60.0)
    f._c_local |= {(0, 1, 10), (0, 1, 11)}
    f.archive = list(range(400))
    Solver.progress_line(f, 120.0)
    line = _lines(tmp_path)[-1]
    assert line["c_local"] == 12 and line["c_local_delta"] == 2
    assert line["c_local_band_delta"] == 2
    assert line["c_local_elasticity"] == pytest.approx(
        coverage_elasticity(10, 100, 12, 400), abs=1e-4)


def test_a_pinned_footprint_under_exploding_cells_reads_as_a_flat_series(
        tmp_path):
    # The hall's shape, through the actual emitter: the derivative goes
    # to ~0 while `cells` keeps climbing. Reported, never classified —
    # nothing here decides anything about the wall.
    f = _line_solver(tmp_path, 28_929, {(0, 1, i) for i in range(932)})
    Solver.progress_line(f, 60.0)
    f._c_local |= {(0, 1, i) for i in range(932, 1102)}
    f.archive = list(range(560_410))
    Solver.progress_line(f, 120.0)
    line = _lines(tmp_path)[-1]
    assert line["c_local"] == 1102
    assert 0.0 < line["c_local_elasticity"] < 0.1


def test_a_solver_without_the_bookkeeping_omits_the_series_and_survives(
        tmp_path):
    # Every duck-typed stand-in and in-process construction predates the
    # field; the emitter must degrade rather than AttributeError on a
    # 60-second cadence nobody is watching.
    f = _line_solver(tmp_path, 5, set())
    del f._c_local
    Solver.progress_line(f, 10.0)
    assert [k for k in _lines(tmp_path)[-1] if k.startswith("c_local")] == []


def test_the_torn_read_counter_is_silent_until_the_filter_is_armed(tmp_path):
    f = _line_solver(tmp_path, 5, set())
    Solver.progress_line(f, 10.0)
    assert "progress_glitches" not in _lines(tmp_path)[-1]
    f.progress_smooth, f._prog_glitches = "borrow", 0
    Solver.progress_line(f, 20.0)
    line = _lines(tmp_path)[-1]
    # zero is reported once armed: "armed and never fired" and "never
    # armed" are different readings, and the phantom cost 10.7 h of not
    # being able to tell them apart.
    assert line["progress_smooth"] == "borrow"
    assert line["progress_glitches"] == 0


def test_observe_grows_the_footprint_only_on_a_new_local_position():
    f = _observe_solver("off")
    _walk(f, [513, 512, 511])
    assert f._c_local == {(0, 0, v // GX_BUCKET) for v in (513, 512, 511)}
    before = set(f._c_local)
    _walk(f, [513, 512, 511])                 # same places again
    assert f._c_local == before


def test_the_emitted_field_is_the_one_the_taxonomy_already_parses(tmp_path):
    # wall_taxonomy reserved `c_local` with exactly this definition
    # (MISSING_TELEMETRY / ProgressRecord). Emitting under a different
    # name or shape would bank another unreadable corpus.
    from src.training.wall_taxonomy import MISSING_TELEMETRY, ProgressRecord

    assert "c_local" in MISSING_TELEMETRY
    assert "(area, " in MISSING_TELEMETRY["c_local"]
    f = _line_solver(tmp_path, 5, {(0, 1, 2)})
    Solver.progress_line(f, 10.0)
    line = _lines(tmp_path)[-1]
    rec = ProgressRecord(elapsed_s=line["elapsed_s"], cells=line["cells"],
                         steps=line["steps"], c_local=line["c_local"])
    assert rec.c_local == 1 and isinstance(line["c_local"], int)


def _hall_shaped_log(n=90, *, c_local=False):
    """A pinned-frontier log the shape of runs/cv_hall_ortho_ctrl: cells
    climb steadily, the map never moves, the footprint crawls."""
    objs = []
    for i in range(n):
        o = {"elapsed_s": 60 * (i + 1), "cells": 1_000 * (i + 1),
             "steps": 500_000 * (i + 1), "solutions": 0, "max_area": 0,
             "max_sect": 0, "max_gx_in_max_area": 751,
             "stall_flat_windows": 0}
        if c_local:
            o["c_local"] = 932 + round(170 * i / (n - 1))
        objs.append(o)
    return objs


def test_emitting_c_local_is_evidence_and_therefore_moves_verdicts():
    # The honest version of "reporting only". The solver does not
    # classify anything, but `c_local` is MISSING_TELEMETRY entry #1 and
    # the discriminator branches on it, so a log that carries it stops
    # abstaining for want of it. Measured on the real 90-line
    # runs/cv_hall_ortho_ctrl/progress.jsonl (2026-08-11): INDETERMINATE
    # / missing=('c_local', ...) without, COVERAGE_LIMITED with. Pinned
    # here so nobody re-asserts inertness the code does not have.
    from src.training.wall_taxonomy import (WallTelemetry,
                                            gated_wall_verdict,
                                            record_from_json)

    def verdict(objs):
        return gated_wall_verdict(WallTelemetry(
            label="hall", records=tuple(record_from_json(o) for o in objs)))

    without = verdict(_hall_shaped_log())
    with_it = verdict(_hall_shaped_log(c_local=True))
    assert "c_local" in without.missing
    assert "c_local" not in with_it.missing
    assert with_it.evidence["c_local"] == 1_102
    assert with_it.wall_class != without.wall_class
    # And what it may NOT do: name a wall opened. The series can only
    # refute a plateau, name a stagnant map, or name a blind key.
    assert with_it.wall_class.name in ("COVERAGE_LIMITED", "BARREN",
                                       "KEY_BLIND")
