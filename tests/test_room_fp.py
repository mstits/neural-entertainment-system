"""Room-graph engine T1: fingerprint core + classifier + schema +
lineage (ROOMGRAPH_ENGINE_2026-08-24 §2/§3 rows 1-3, 10-12).

Everything here runs without a ROM, a Pool or a Solver constructor —
the identity layer is pure by design, and the Solver methods under test
(_xram, _xram_local, _replay_room_ord, _resume_room_index) are
exercised through duck-typed stand-ins, the same pattern the rest of
the solver suite uses. The hot-loop wiring (settle block in explore(),
edge commits at the transit point, _assign seeding) is T2's and is
covered by its Zelda smoke + the RG-0 fixture falsifier (T4).
"""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from scripts.go_explore_solve import (
    ODO_ALT,
    ODO_LO,
    ROOM_HI,
    ROOM_LO,
    ROOM_UNKNOWN,
    GenericGame,
    RoomIndex,
    Solver,
    classify_transition,
    fp_settle,
    key_config_axes,
    nt_fingerprint,
    resume_lineage_diff,
    room_fp_config_sha,
    room_fp_mask,
)

# ---------------------------------------------------------------------
# mask + fingerprint
# ---------------------------------------------------------------------


def test_mask_zeroes_declared_ranges_and_keeps_the_rest():
    m = room_fp_mask([[0, 4], [2044, 2048]])
    assert m.shape == (2048,) and m.dtype == np.uint8
    assert m[:4].sum() == 0 and m[2044:].sum() == 0
    assert m[4:2044].sum() == 2040


def test_fingerprint_is_deterministic_and_64_bit():
    nt = bytes(range(256)) * 8
    m = room_fp_mask([])
    a, b = nt_fingerprint(nt, m), nt_fingerprint(nt, m)
    assert a == b
    assert 0 <= a < 2 ** 64
    # bytes and ndarray input agree — the hot loop hands ndarrays, the
    # replay path hands PyBytes.
    assert nt_fingerprint(np.frombuffer(nt, dtype=np.uint8), m) == a


def test_fingerprint_ignores_masked_bytes():
    m = room_fp_mask([[0, 64]])
    base = bytearray(2048)
    noisy = bytearray(2048)
    noisy[0:64] = bytes(range(64))          # animated tiles churn here
    assert nt_fingerprint(bytes(base), m) == nt_fingerprint(bytes(noisy), m)


def test_fingerprint_separates_unmasked_content():
    m = room_fp_mask([[0, 64]])
    a = bytearray(2048)
    b = bytearray(2048)
    b[100] = 1
    assert nt_fingerprint(bytes(a), m) != nt_fingerprint(bytes(b), m)


def test_palette_cokey_changes_the_hash_only_when_folded():
    nt = bytes(2048)
    m = room_fp_mask([])
    plain = nt_fingerprint(nt, m)
    assert nt_fingerprint(nt, m, palette=bytes(32)) != plain
    assert (nt_fingerprint(nt, m, palette=bytes([1] * 32))
            != nt_fingerprint(nt, m, palette=bytes(32)))


# ---------------------------------------------------------------------
# transition-kind classifier (constants probe-receipted 2026-08-24)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("d_odo,want", [((256, 0), ("pan", "E")),
                                        ((-256, 0), ("pan", "W")),
                                        ((0, 256), ("pan", "S")),
                                        ((0, -256), ("pan", "N"))])
def test_pan_directions_come_from_the_single_moving_axis(d_odo, want):
    assert classify_transition(d_odo, 0) == want
    assert classify_transition(d_odo, 1) == want
    assert classify_transition(d_odo, 2) == want     # scene noise: still pan
    assert classify_transition(d_odo, 5) == want


def test_a_pan_co_occurring_with_scene_noise_is_still_a_pan():
    # 2026-08-25 hardening finding: the core's scene-cut heuristic
    # fires on ordinary camera clamp/seam noise near screen edges —
    # exactly where real pans happen (RG-0's own Metroid fixture:
    # scene reaches 8 during two ordinary walking laps in ONE room,
    # zero real transitions). The RG-0-receipted real door (+254px,
    # Δscene+1) must classify identically whether or not a spurious
    # extra scene bump lands in the same churn window — a `pan` gate
    # that also required Δscene <= 1 silently demoted real doors to
    # unrouted fades whenever noise and a pan coincided.
    assert classify_transition((254, 0), 1) == ("pan", "E")
    assert classify_transition((254, 0), 2) == ("pan", "E")
    assert classify_transition((254, 0), 4) == ("pan", "E")


def test_zelda_death_signature_classifies_warp():
    # Measured: odometer modal 16 -> 272 -> 16 (flat at settle), scene
    # +2. A warp adopts the new identity but must NEVER become an edge.
    assert classify_transition((0, 0), 2) == ("warp", None)
    assert classify_transition((16, -8), 3) == ("warp", None)


def test_pan_and_warp_windows_stay_mutually_exclusive():
    # The pan fix (dropping the Δscene gate) is safe only because a
    # pan-sized single-axis delta can never ALSO satisfy the warp
    # branch's both-axes-near-zero requirement. Pin that invariant
    # directly so a future window-constant change can't silently
    # reopen the smuggled-warp risk the docstring reasons about.
    assert classify_transition((256, 0), 9) == ("pan", "E")   # not warp
    assert classify_transition((0, 0), 9) == ("warp", None)   # not pan


def test_fade_is_the_otherwise_class():
    # Zelda caves/dungeons, Rygar doors: hash flip, odo ~flat, scene
    # 0..1 — the class the scene core is blind to by design.
    assert classify_transition((0, 0), 0) == ("fade", None)
    assert classify_transition((4, 4), 1) == ("fade", None)


def test_both_axes_in_the_pan_window_is_not_a_pan():
    assert classify_transition((256, 256), 0) == ("fade", None)


def test_a_pan_sized_move_with_a_scene_jump_is_still_a_pan():
    # SUPERSEDES the pre-2026-08-25 contract (a pan-sized move with
    # Δscene >= 2 used to read as "fade" — the exact bug the hardening
    # audit caught: RG-0's own fixture shows Δscene noise firing on
    # ORDINARY movement, so gating pan on Δscene <= 1 silently demoted
    # real doors whenever noise coincided). Not a warp either (the
    # odometer moved) — pan is correct: real spatial displacement always
    # wins over a noisy scene counter.
    assert classify_transition((256, 0), 2) == ("pan", "E")


def test_warp_needs_a_flat_odometer_on_both_axes():
    assert classify_transition((32, 0), 2) == ("fade", None)
    assert classify_transition((0, 40), 2) == ("fade", None)


def test_pan_window_bounds_are_inclusive_and_custom():
    assert classify_transition((128, 0), 0) == ("pan", "E")
    assert classify_transition((384, 0), 0) == ("pan", "E")
    assert classify_transition((127, 0), 0) == ("fade", None)
    assert classify_transition((385, 0), 0) == ("fade", None)
    assert classify_transition((100, 0), 0,
                               pan_odo=(64, 120)) == ("pan", "E")


# ---------------------------------------------------------------------
# fp_settle — the pure settle-state transition
# ---------------------------------------------------------------------


def test_settle_fires_after_n_consecutive_identical_hashes():
    pend, fired = fp_settle(None, 0xB, 0xA, (0, 0), 0, 10, settle=3)
    assert fired is None and pend[0] == 0xB and pend[1] == 1
    pend, fired = fp_settle(pend, 0xB, 0xA, (0, 0), 0, 11, settle=3)
    assert fired is None and pend[1] == 2
    pend, fired = fp_settle(pend, 0xB, 0xA, (0, 0), 0, 12, settle=3)
    assert pend is None
    h, d_odo, d_scene, frames = fired
    assert h == 0xB and d_odo == (0, 0) and d_scene == 0 and frames == 2


def test_matching_the_settled_hash_cancels_the_churn():
    pend, _ = fp_settle(None, 0xB, 0xA, (0, 0), 0, 10, settle=3)
    pend, fired = fp_settle(pend, 0xA, 0xA, (0, 0), 0, 11, settle=3)
    assert pend is None and fired is None    # false alarm: back on A


def test_intra_churn_hash_changes_preserve_the_onset():
    # A Zelda pan: the nametable churns while the camera scrolls, so
    # the FINAL hash first appears near pan end. The classifier needs
    # Δodo integrated over the whole churn window — onset must survive
    # the intermediate hash flips or every pan reads as a fade.
    pend, _ = fp_settle(None, 0xC1, 0xA, (10, 0), 0, 100, settle=3)
    pend, _ = fp_settle(pend, 0xC2, 0xA, (120, 0), 0, 101, settle=3)
    pend, _ = fp_settle(pend, 0xB, 0xA, (250, 0), 0, 102, settle=3)
    pend, _ = fp_settle(pend, 0xB, 0xA, (266, 0), 0, 103, settle=3)
    pend, fired = fp_settle(pend, 0xB, 0xA, (266, 0), 0, 104, settle=3)
    assert pend is None and fired is not None
    h, d_odo, d_scene, frames = fired
    assert h == 0xB
    assert d_odo == (256, 0)                 # 266 - onset 10: full pan
    assert frames == 4
    assert classify_transition(d_odo, d_scene) == ("pan", "E")


def test_mid_pan_settle_cannot_fire_while_hashes_churn():
    pend = None
    for step, h in enumerate([0x1, 0x2, 0x3, 0x4, 0x5, 0x6]):
        pend, fired = fp_settle(pend, h, 0xA, (step * 40, 0), 0, step,
                                settle=3)
        assert fired is None                 # never repeats: never fires
    assert pend[1] == 1


def test_first_adoption_from_unknown_settles_too():
    # A worker seeded ROOM_UNKNOWN has no settled hash (None): its
    # first stable fingerprint must settle normally (the adoption-from-
    # unknown guard downstream makes it transit-free and edge-free).
    pend = None
    fired = None
    for step in range(3):
        pend, fired = fp_settle(pend, 0xD, None, (5, 5), 1, step, settle=3)
    assert fired is not None and fired[0] == 0xD


def test_warp_and_fade_streams_classify_end_to_end():
    # Warp — the measured Zelda death: churn begins pre-bump, the
    # odometer spikes 16 -> 272 -> 16 mid-churn (flat again at settle),
    # scene lands +2 from the churn ONSET. Δ is onset-to-settle, so the
    # transient spike never enters the classifier.
    pend, fired = fp_settle(None, 0xD1, 0xA, (16, 0), 0, 0, settle=3)
    pend, fired = fp_settle(pend, 0xDEAD, 0xA, (272, 0), 2, 1, settle=3)
    pend, fired = fp_settle(pend, 0xDEAD, 0xA, (16, 0), 2, 2, settle=3)
    pend, fired = fp_settle(pend, 0xDEAD, 0xA, (16, 0), 2, 3, settle=3)
    assert fired is not None
    assert classify_transition(fired[1], fired[2]) == ("warp", None)
    # Fade: cave entry — hash flips, odo flat, scene bumps at most once
    # after the churn begins.
    pend, fired = fp_settle(None, 0xC1, 0xA, (8, 4), 0, 0, settle=3)
    for step in range(1, 4):
        pend, fired = fp_settle(pend, 0xCAFE, 0xA, (8, 4), 1, step,
                                settle=3)
    assert fired is not None
    assert classify_transition(fired[1], fired[2]) == ("fade", None)


# ---------------------------------------------------------------------
# RoomIndex
# ---------------------------------------------------------------------


def test_intern_assigns_discovery_order_ordinals():
    idx = RoomIndex(cap=8)
    assert idx.intern(0xAAA) == 0
    assert idx.intern(0xBBB) == 1
    assert idx.intern(0xAAA) == 0            # idempotent
    assert idx.ordinals == [0xAAA, 0xBBB]
    assert idx.n_rooms() == 2


def test_intern_counts_visits_and_grows_the_bbox():
    idx = RoomIndex(cap=8)
    idx.intern(0xAAA, odo_xy=(10, 20))
    idx.intern(0xAAA, odo_xy=(300, 5))
    m = idx.meta[0]
    assert m["visits"] == 2
    assert m["bbox"] == [10, 5, 300, 20]


def test_cap_holds_last_and_counts_instead_of_crashing():
    idx = RoomIndex(cap=2)
    assert idx.intern(0x1) == 0 and idx.intern(0x2) == 1
    assert idx.intern(0x3) is None           # hold-last, never crash
    assert idx.cap_hits == 1
    assert idx.intern(0x1) == 0              # known hashes still resolve


def test_lookup_is_frozen_and_never_interns():
    idx = RoomIndex(cap=8)
    idx.intern(0xAAA)
    assert idx.lookup(0xAAA) == 0
    assert idx.lookup(0xBBB) is None
    assert idx.n_rooms() == 1


def test_record_edge_accumulates_and_keeps_the_first_exemplar():
    idx = RoomIndex(cap=8)
    key = (0, 0, 0, (1, 2), 0, (), 3, 1, 0, 4, 12)   # a real nested key
    idx.record_edge(0, 1, "pan", "E", 10, exemplar_cell=key,
                    exemplar_actions=list(range(40)))
    idx.record_edge(0, 1, "pan", "E", 20, exemplar_cell=(9,),
                    exemplar_actions=[1, 2])
    e = idx.adj[0][1]
    assert e["count"] == 2 and e["frames_mean"] == 15.0
    assert e["exemplar_cell"] == key         # FIRST exemplar is stable
    assert len(e["exemplar_actions"]) == 32  # last-32 action ring


def test_record_edge_self_heals_a_fade_into_a_pan():
    # 2026-08-25 hardening finding: a fade minted by an unlucky
    # scene-noise co-occurrence used to be PERMANENT — a repeat
    # traversal only ever touched count/frames_mean, never kind/dir,
    # so a real door misclassified once stayed unrouted forever even
    # after a clean re-traversal proved it was a pan.
    idx = RoomIndex(cap=8)
    idx.record_edge(0, 1, "fade", None, 12)
    e = idx.adj[0][1]
    assert e["kind"] == "fade" and e["dir"] is None
    idx.record_edge(0, 1, "pan", "E", 8)              # clean traversal
    e = idx.adj[0][1]
    assert e["kind"] == "pan" and e["dir"] == "E"      # healed
    assert e["count"] == 2                             # still accumulates


def test_record_edge_never_downgrades_a_pan_to_a_fade():
    # The reverse must not happen: noise on a LATER visit must not
    # erase a clean direction read from an earlier one.
    idx = RoomIndex(cap=8)
    idx.record_edge(0, 1, "pan", "E", 8)
    idx.record_edge(0, 1, "fade", None, 12)            # noisy repeat
    e = idx.adj[0][1]
    assert e["kind"] == "pan" and e["dir"] == "E"       # unchanged
    assert e["count"] == 2


def test_record_edge_refuses_the_warp_kind():
    idx = RoomIndex(cap=8)
    with pytest.raises(ValueError, match="warp"):
        idx.record_edge(0, 1, "warp", None, 5)


def test_record_warp_is_telemetry_never_adjacency():
    idx = RoomIndex(cap=8)
    idx.record_warp(0, 1, 2, (0, 0))
    idx.record_warp(None, 2, 3, (4, -4))     # adoption-side warp: no src
    assert idx.warp_count == 2
    assert idx.adj == {}                     # NEVER an edge


def test_save_load_roundtrip_preserves_identity_and_exemplar_keys(tmp_path):
    idx = RoomIndex(cap=8, config_sha="cafe1234")
    idx.intern(0xAAA, odo_xy=(1, 2))
    idx.intern(0xBBB, odo_xy=(600, 2))
    key = (2, 0, 0, (7, 9), 1, ((3, 1),), 4, 1, 0, 4, 12)
    idx.record_edge(0, 1, "fade", None, 30, exemplar_cell=key,
                    exemplar_actions=[3, 3, 5])
    idx.record_warp(1, 0, 2, (0, 0))
    idx.save(tmp_path / "room_index.json")
    back = RoomIndex.load(tmp_path / "room_index.json")
    assert back.config_sha == "cafe1234"
    assert back.hashes == idx.hashes and back.ordinals == idx.ordinals
    assert back.meta == idx.meta
    assert back.warp_count == 1
    e = back.adj[0][1]
    # The exemplar cell key must survive JSON as the SAME nested tuple,
    # or the edge-replay audit can never find its archive cell.
    assert e["exemplar_cell"] == key
    assert e["exemplar_actions"] == [3, 3, 5]
    assert e["kind"] == "fade" and e["count"] == 1


def test_load_refuses_a_version_it_does_not_speak(tmp_path):
    p = tmp_path / "room_index.json"
    p.write_text(json.dumps({"version": 99, "config_sha": "x",
                             "hashes": []}))
    with pytest.raises(ValueError, match="version"):
        RoomIndex.load(p)


# ---------------------------------------------------------------------
# config sha — the lineage identity of the room_fp block
# ---------------------------------------------------------------------


def _cfg(**over):
    cfg = {"mask": [[0, 256], [1024, 1280]], "settle": 3,
           "min_lines": 200, "pan_odo": [128, 384], "warp_scene_min": 2,
           "palette_cokey": False, "max_rooms": 1024, "sample_every": 1}
    cfg.update(over)
    return cfg


def test_config_sha_canonicalises_mask_order():
    a = room_fp_config_sha(_cfg(mask=[[0, 256], [1024, 1280]]))
    b = room_fp_config_sha(_cfg(mask=[[1024, 1280], [0, 256]]))
    assert a == b


@pytest.mark.parametrize("knob,value", [("mask", [[0, 128]]),
                                        ("settle", 5),
                                        ("min_lines", 100),
                                        ("pan_odo", [64, 384]),
                                        ("warp_scene_min", 3),
                                        ("palette_cokey", True),
                                        ("max_rooms", 512)])
def test_config_sha_moves_with_every_identity_knob(knob, value):
    assert room_fp_config_sha(_cfg(**{knob: value})) != \
        room_fp_config_sha(_cfg())


def test_config_sha_ignores_the_sample_every_perf_fallback():
    # RG-1d's re-measure path (`sample_every: 2`) changes when the
    # detector looks, not what an ordinal means: not lineage.
    assert room_fp_config_sha(_cfg(sample_every=2)) == \
        room_fp_config_sha(_cfg())


# ---------------------------------------------------------------------
# profile parse (GenericGame)
# ---------------------------------------------------------------------


def _profile(room_fp=None):
    s = {"rom": "roms/x.nes", "progress": {"lo": 0x0070}, "y": 0x0084,
         "level_key": [], "lives": 0x002A}
    if room_fp is not None:
        s["room_fp"] = room_fp
    return {"solve": s}


def test_profile_without_room_fp_is_inert():
    g = GenericGame(_profile())
    assert g.room_fp is None and g.room_fp_sha == ""


def test_room_fp_parse_applies_defaults_and_hashes_them():
    g = GenericGame(_profile({"mask": [[0, 256]]}))
    assert g.room_fp == {"mask": [[0, 256]], "settle": 3,
                         "min_lines": 200, "pan_odo": [128, 384],
                         "warp_scene_min": 2, "palette_cokey": False,
                         "max_rooms": 1024, "sample_every": 1}
    assert g.room_fp_sha == room_fp_config_sha(g.room_fp)
    assert len(g.room_fp_sha) == 8


def test_an_empty_room_fp_block_still_arms_the_feature():
    g = GenericGame(_profile({}))
    assert g.room_fp is not None and g.room_fp["mask"] == []


@pytest.mark.parametrize("bad", [{"mask": [[0, 3000]]},
                                 {"mask": [[100, 100]]},
                                 {"mask": [[-1, 5]]},
                                 {"settle": 0},
                                 {"max_rooms": 5000},
                                 {"max_rooms": 0},
                                 {"pan_odo": [0, 384]},
                                 {"pan_odo": [384, 128]},
                                 {"pan_odo": [128]},
                                 {"warp_scene_min": 0},
                                 {"min_lines": 300},
                                 {"sample_every": 0}])
def test_room_fp_validation_is_loud_and_at_construction(bad):
    with pytest.raises(SystemExit):
        GenericGame(_profile(bad))


# ---------------------------------------------------------------------
# _xram — the 4-vs-7-byte extension switch
# ---------------------------------------------------------------------


def _xram_self(room_fp=None, ord_=5, axis="x", odo=(0x123456, 0x0ABCDE),
               scene=7, odo_on=True):
    return SimpleNamespace(
        _odo=odo_on, _odo_now=[odo], _odo_scene=[scene],
        game=SimpleNamespace(odometer_axis=axis),
        room_fp=room_fp,
        _room_ord=np.array([ord_], dtype=np.uint16))


def test_xram_off_is_a_passthrough():
    ram = bytes(0x800)
    assert Solver._xram(_xram_self(odo_on=False), ram, 0) is ram


def test_xram_without_room_fp_is_the_exact_4_byte_extension():
    ext = Solver._xram(_xram_self(), bytes(0x800), 0)
    assert len(ext) == 0x800 + 4
    assert (ext[ODO_LO], ext[ODO_LO + 1], ext[ODO_LO + 2]) == \
        (0x56, 0x34, 0x12)                   # 0x123456 little-endian
    assert ext[ODO_LO + 3] == 7              # scene mod 256


def test_xram_with_room_fp_extends_7_and_writes_ordinal_and_alt_axis():
    ext = Solver._xram(_xram_self(room_fp=_cfg(), ord_=0x0102), bytes(0x800),
                       0)
    assert len(ext) == 0x800 + 7
    assert (ext[ROOM_LO], ext[ROOM_HI]) == (0x02, 0x01)   # uint16 LE
    # ODO_ALT carries the OTHER axis, (v >> 4) & 0xFF: y = 0x0ABCDE.
    assert ext[ODO_ALT] == (0x0ABCDE >> 4) & 0xFF


def test_xram_prefix_is_byte_identical_between_the_two_modes():
    ram = bytes(range(256)) * 8
    off = Solver._xram(_xram_self(), ram, 0)
    on = Solver._xram(_xram_self(room_fp=_cfg()), ram, 0)
    assert bytes(off) == bytes(on[:0x804])


def test_xram_room_unknown_rides_the_slot_as_ffff():
    ext = Solver._xram(_xram_self(room_fp=_cfg(), ord_=ROOM_UNKNOWN),
                       bytes(0x800), 0)
    assert ext[ROOM_LO] == 0xFF and ext[ROOM_HI] == 0xFF


def test_xram_axis_y_swaps_primary_and_alt():
    ext = Solver._xram(_xram_self(room_fp=_cfg(), axis="y"), bytes(0x800), 0)
    assert (ext[ODO_LO], ext[ODO_LO + 1], ext[ODO_LO + 2]) == \
        (0xDE, 0xBC, 0x0A)                   # y = 0x0ABCDE at 0x800
    assert ext[ODO_ALT] == (0x123456 >> 4) & 0xFF


def test_xram_axis_none_defaults_primary_to_x():
    # room_fp forced the odometer on for a RAM-progress profile (Zelda):
    # odometer_axis is None, primary slot carries x, ODO_ALT carries y.
    ext = Solver._xram(_xram_self(room_fp=_cfg(), axis=None), bytes(0x800), 0)
    assert (ext[ODO_LO], ext[ODO_LO + 1], ext[ODO_LO + 2]) == \
        (0x56, 0x34, 0x12)


# ---------------------------------------------------------------------
# _replay_room_ord — frozen-index re-derivation with hold-last
# ---------------------------------------------------------------------


class _ReplayPool:
    def __init__(self, nt, lines=240):
        self.nt, self.lines = nt, lines

    def odo_debug(self, wid):
        return (0, 0, self.lines)

    def peek_nametables(self, wid):
        return bytes(self.nt)

    def get_odometer_per_worker(self):
        return [(0, 0)]

    def get_odometer_scene_per_worker(self):
        return [0]


def _replay_self(index, cfg=None):
    cfg = cfg or _cfg(mask=[])
    return SimpleNamespace(room_fp=cfg,
                           _room_mask=room_fp_mask(cfg["mask"]),
                           room_index=index,
                           game=SimpleNamespace(odometer_axis=None),
                           _odo=True)


def test_replay_room_ord_reads_a_known_hash():
    nt = bytes([3]) * 2048
    idx = RoomIndex(cap=8)
    idx.intern(nt_fingerprint(nt, room_fp_mask([])))
    ctx = {}
    fake = _replay_self(idx)
    assert Solver._replay_room_ord(fake, _ReplayPool(nt), ctx) == 0
    assert ctx["_room_ord"] == 0


def test_replay_room_ord_holds_last_on_unknown_hash_and_blank_frames():
    known = bytes([3]) * 2048
    idx = RoomIndex(cap=8)
    idx.intern(nt_fingerprint(known, room_fp_mask([])))
    fake = _replay_self(idx)
    ctx = {}
    # Unknown before anything settled: ROOM_UNKNOWN, not a crash.
    assert Solver._replay_room_ord(fake, _ReplayPool(bytes(2048)),
                                   ctx) == ROOM_UNKNOWN
    assert Solver._replay_room_ord(fake, _ReplayPool(known), ctx) == 0
    # Mid-transition churn hash: unknown -> hold the last ordinal.
    assert Solver._replay_room_ord(fake, _ReplayPool(bytes(2048)), ctx) == 0
    # Blank frame (fade): below min_lines, never sampled -> hold.
    assert Solver._replay_room_ord(fake, _ReplayPool(bytes(2048), lines=0),
                                   ctx) == 0


def test_replay_room_ord_never_interns_into_the_frozen_index():
    idx = RoomIndex(cap=8)
    fake = _replay_self(idx)
    Solver._replay_room_ord(fake, _ReplayPool(bytes([9]) * 2048), {})
    assert idx.n_rooms() == 0


def test_xram_local_writes_the_derived_ordinal():
    nt = bytes([3]) * 2048
    idx = RoomIndex(cap=8)
    idx.intern(nt_fingerprint(nt, room_fp_mask([])))
    fake = _replay_self(idx)
    fake._replay_room_ord = MethodType(Solver._replay_room_ord, fake)
    ext = Solver._xram_local(fake, bytes(0x800), _ReplayPool(nt), {})
    assert len(ext) == 0x800 + 7
    assert (ext[ROOM_LO], ext[ROOM_HI]) == (0, 0)


def test_xram_local_without_room_fp_keeps_the_4_byte_extension():
    fake = _replay_self(None, cfg=None)
    fake.room_fp = None
    ext = Solver._xram_local(fake, bytes(0x800), _ReplayPool(bytes(2048)))
    assert len(ext) == 0x800 + 4


# ---------------------------------------------------------------------
# lineage axis + cross-schema resume refusal
# ---------------------------------------------------------------------

_ARGS = SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=16,
                        y_band=32)


def test_key_config_records_the_room_fp_sha():
    on = key_config_axes(_ARGS, GenericGame(_profile({"mask": [[0, 8]]})))
    off = key_config_axes(_ARGS, GenericGame(_profile()))
    assert off["room_fp"] == ""
    assert on["room_fp"] == room_fp_config_sha(
        GenericGame(_profile({"mask": [[0, 8]]})).room_fp)


def _diff(prev_cfg, run_cfg):
    prov = {"hw_flags": [], "frame_skip": 4, "nes_core": {}}
    return resume_lineage_diff({"hw_provenance": prov,
                                "key_config": prev_cfg}, prov, run_cfg)


def test_legacy_archive_without_the_axis_matches_a_room_fp_off_run():
    # Every archive banked before the feature existed was built with it
    # off: absent reads as "", so flags-off resume is untouched.
    prev = key_config_axes(_ARGS, GenericGame(_profile()))
    del prev["room_fp"]
    run = key_config_axes(_ARGS, GenericGame(_profile()))
    d = _diff(prev, run)
    assert d["mismatch"] == [] and d["unverifiable"] == []


def test_legacy_archive_is_refused_when_the_run_fingerprints():
    prev = key_config_axes(_ARGS, GenericGame(_profile()))
    del prev["room_fp"]
    run = key_config_axes(_ARGS, GenericGame(_profile({"mask": [[0, 8]]})))
    d = _diff(prev, run)
    assert len(d["mismatch"]) == 1
    assert "room" in d["mismatch"][0]


def test_recorded_off_vs_run_on_is_a_mismatch_both_ways():
    off = key_config_axes(_ARGS, GenericGame(_profile()))
    on = key_config_axes(_ARGS, GenericGame(_profile({"mask": [[0, 8]]})))
    assert len(_diff(off, on)["mismatch"]) == 1
    assert len(_diff(on, off)["mismatch"]) == 1


def test_matching_room_fp_shas_are_clean():
    on = key_config_axes(_ARGS, GenericGame(_profile({"mask": [[0, 8]]})))
    d = _diff(dict(on), dict(on))
    assert d["mismatch"] == [] and d["unverifiable"] == []


def test_two_different_room_fp_schemas_are_a_mismatch():
    a = key_config_axes(_ARGS, GenericGame(_profile({"mask": [[0, 8]]})))
    b = key_config_axes(_ARGS, GenericGame(_profile({"settle": 5})))
    assert len(_diff(a, b)["mismatch"]) == 1


# ---------------------------------------------------------------------
# _resume_room_index — the room_index.json half (hard refusals)
# ---------------------------------------------------------------------


def _banked(tmp_path, sha, with_index=True, index_sha=None):
    (tmp_path / "archive.stats.json").write_text(json.dumps(
        {"key_config": {"room_fp": sha}}))
    if with_index:
        idx = RoomIndex(cap=8, config_sha=index_sha or sha)
        idx.intern(0xAAA)
        idx.save(tmp_path / "room_index.json")
    return tmp_path


def _resume_self(run_sha):
    return SimpleNamespace(game=SimpleNamespace(room_fp_sha=run_sha),
                           room_index=None)


def test_resume_with_both_sides_off_is_a_no_op(tmp_path):
    (tmp_path / "archive.stats.json").write_text(json.dumps({}))
    fake = _resume_self("")
    Solver._resume_room_index(fake, tmp_path)
    assert fake.room_index is None


def test_resume_refused_when_the_index_file_is_missing(tmp_path):
    _banked(tmp_path, "cafe1234", with_index=False)
    with pytest.raises(SystemExit, match="room_index.json is missing"):
        Solver._resume_room_index(_resume_self("cafe1234"), tmp_path)


def test_resume_refused_when_the_index_sha_disagrees_with_lineage(tmp_path):
    _banked(tmp_path, "cafe1234", index_sha="deadbeef")
    with pytest.raises(SystemExit, match="not the one the"):
        Solver._resume_room_index(_resume_self("cafe1234"), tmp_path)


def test_resume_refused_across_room_fp_schemas(tmp_path):
    _banked(tmp_path, "cafe1234")
    with pytest.raises(SystemExit, match="different mask/settle"):
        Solver._resume_room_index(_resume_self("0badc0de"), tmp_path)


def test_resume_refused_on_an_unreadable_index(tmp_path):
    _banked(tmp_path, "cafe1234", with_index=False)
    (tmp_path / "room_index.json").write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable"):
        Solver._resume_room_index(_resume_self("cafe1234"), tmp_path)


def test_resume_adopts_the_matching_index_and_its_ordinals(tmp_path):
    _banked(tmp_path, "cafe1234")
    fake = _resume_self("cafe1234")
    Solver._resume_room_index(fake, tmp_path)
    assert fake.room_index is not None
    assert fake.room_index.n_rooms() == 1
    assert fake.room_index.lookup(0xAAA) == 0   # SAME intern table
    # New rooms continue discovery order, never restart it.
    assert fake.room_index.intern(0xBBB) == 1


def test_resume_run_off_archive_on_loads_nothing_here(tmp_path):
    # The on/off axis is the generic lineage diff's refusal; this
    # method only owns the index file. It must not adopt an index into
    # a run that is not fingerprinting.
    _banked(tmp_path, "cafe1234")
    fake = _resume_self("")
    Solver._resume_room_index(fake, tmp_path)
    assert fake.room_index is None


def test_resume_run_on_archive_off_defers_to_the_lineage_diff(tmp_path):
    (tmp_path / "archive.stats.json").write_text(json.dumps(
        {"key_config": {"room_fp": ""}}))
    fake = _resume_self("cafe1234")
    Solver._resume_room_index(fake, tmp_path)   # no raise HERE
    assert fake.room_index is None


# ---------------------------------------------------------------------
# T2 — hot-loop wiring: _room_seed / _room_step / _room_transit
# (settle+classify before the ram fetch, kind-tagged edge commit with
# the warp veto at the transit point, exemplar ring, _assign seeding +
# restore-lockstep invariants — ROOMGRAPH_ENGINE_2026-08-24 §2, §3
# rows 4-5/9)
# ---------------------------------------------------------------------


def _nt(v: int) -> bytes:
    return bytes([v]) * 2048


class _HotPool:
    """Mutable pool stand-in the settle loop reads each step."""

    def __init__(self):
        self.nt = _nt(0)
        self.lines = 240
        self.pal = bytes(32)
        self.peeks = 0

    def odo_debug(self, wid):
        return (0, 0, self.lines)

    def peek_nametables(self, wid):
        self.peeks += 1
        return self.nt

    def palette_ram(self, wid):
        return self.pal


def _hot_self(cfg=None, cap=None):
    cfg = cfg or _cfg(mask=[], settle=2)
    s = SimpleNamespace(
        room_fp=cfg,
        _room_mask=room_fp_mask(cfg["mask"]),
        room_index=RoomIndex(cap=(cap if cap is not None
                                  else cfg["max_rooms"]),
                             config_sha="cafe1234"),
        _room_ord=np.full(1, ROOM_UNKNOWN, dtype=np.uint16),
        # room_sig = [ROOM_LO, ROOM_HI] exactly as §4 wires it.
        _room_psig_off=(-2, -1, 2),
        _room_settle_rejects=0, _room_edges_committed=0,
        _room_edges_dropped=0, _room_restore_transits=0,
        _room_adoptions=0,
        _odo_now=[(0, 0)], _odo_scene=[0],
        pool=_HotPool(), frame_skip=4)
    s._room_seed = MethodType(Solver._room_seed, s)
    s._room_step = MethodType(Solver._room_step, s)
    s._room_transit = MethodType(Solver._room_transit, s)
    return s


def _ctx(s, psig=()):
    c = {"burst_step": 0, "p0750": None, "pending": 0}
    s._room_seed(0, c, psig)
    return c


def _step(s, c, nt, odo=(0, 0), scene=0, action=0, lines=240):
    """One explore()-ordered iteration slice: settle block first (it
    runs before the ram fetch), then the burst_step increment."""
    s.pool.nt = nt
    s.pool.lines = lines
    s._odo_now[0] = odo
    s._odo_scene[0] = scene
    c["pending"] = action
    s._room_step(0, c)
    c["burst_step"] += 1
    return int(s._room_ord[0])


def test_seed_derives_the_ordinal_from_the_psig_tail():
    s = _hot_self()
    h = nt_fingerprint(_nt(3), s._room_mask)
    s.room_index.intern(h)
    c = _ctx(s, psig=(9, 9, 0, 0))     # ... + ordinal 0 as (lo, hi)
    assert int(s._room_ord[0]) == 0
    assert c["fp_h"] == h
    # Derived, never accumulated: live-confirmation starts DOWN so a
    # stale seed cannot bridge a restore into an edge.
    assert c["fp_live"] is False
    assert c["fp_pend"] is None and c["fp_edge"] is None


def test_seed_refuses_a_tail_naming_no_interned_ordinal():
    s = _hot_self()
    assert _ctx(s, psig=(9, 9, 5, 0))["fp_h"] is None
    assert int(s._room_ord[0]) == ROOM_UNKNOWN
    # The ROOM_UNKNOWN bytes a pre-adoption transit threads are refused
    # the same way (0xFFFF is never interned).
    _ctx(s, psig=(9, 9, 0xFF, 0xFF))
    assert int(s._room_ord[0]) == ROOM_UNKNOWN


def test_seed_without_psig_offsets_is_room_unknown():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))
    s._room_psig_off = None            # room_sig without 0x804/0x805
    _ctx(s, psig=(9, 9, 0, 0))
    assert int(s._room_ord[0]) == ROOM_UNKNOWN


def test_root_first_settle_adopts_transit_free_and_edge_free():
    s = _hot_self()
    c = _ctx(s)                        # root path: ROOM_UNKNOWN
    c["p0750"] = ("sentinel",)         # as if steps already stamped it
    assert _step(s, c, _nt(3)) == ROOM_UNKNOWN
    assert _step(s, c, _nt(3)) == 0    # settle=2 fires: adoption
    assert c["p0750"] is None          # adoption != transit (§2)
    assert c["fp_edge"] is None and s.room_index.adj == {}
    assert s._room_adoptions == 1 and c["fp_live"] is True


def test_steady_state_confirms_the_seeded_identity_live():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3))                # h == settled hash: confirmed
    assert c["fp_live"] is True and c["fp_pend"] is None


def test_a_stale_seed_resettles_without_minting_an_edge():
    # The restore false-edge guard: seeded room A but the restored
    # frame is really room B (frozen psig past sect_cap, drift) — the
    # re-settle self-heals the identity and stages NOTHING.
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))   # A=0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(4))
    assert _step(s, c, _nt(4)) == 1    # adopted B without confirmation
    assert c["fp_edge"] is None and s.room_index.adj == {}
    assert c["fp_live"] is True        # ... and B itself is now live


def test_a_live_pan_stages_the_edge_and_the_transit_commits_it():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))   # A=0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3), odo=(0, 0), action=1)          # confirm live
    c["cur_key"] = ("cellA",)          # last cell observed pre-churn
    _step(s, c, _nt(4), odo=(0, 0), action=2)          # churn onset
    _step(s, c, _nt(4), odo=(256, 0), action=3)        # +256 px: pan E
    assert int(s._room_ord[0]) == 1
    src, dst, kind, direction, frames, exemplar, acts, cap_sig = c["fp_edge"]
    assert (src, dst, kind, direction) == (0, 1, "pan", "E")
    assert exemplar == ("cellA",)
    assert acts[-3:] == [1, 2, 3]      # the last-32 action ring
    # Churn window = 1 solver step from the first diverging sample,
    # frame-skip scaled.
    assert frames == 1 * s.frame_skip
    # item-sig graft unarmed on this duck-typed stand-in (no
    # _item_sig_armed attribute at all) => cap_sig defaults to 0.
    assert cap_sig == 0
    # ... and the transit block (this same iteration) commits it:
    s._room_transit(c)
    e = s.room_index.adj[0][1]
    assert (e["kind"], e["dir"], e["count"]) == ("pan", "E", 1)
    assert e["exemplar_cell"] == ("cellA",)
    assert e["exemplar_actions"][-3:] == [1, 2, 3]
    assert e["cap_hist"] == {"0": 1}
    assert c["fp_edge"] is None and s._room_edges_committed == 1


def test_a_warp_settle_adopts_identity_but_mints_no_edge():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))   # A=0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3), scene=0)                       # confirm live
    # Zelda death: the scene bumps DURING the churn window (odometer
    # flat, scene +2 by settle) — onset captures scene 0.
    _step(s, c, _nt(4), scene=0)
    assert _step(s, c, _nt(4), scene=2) == 1           # adopted
    assert c["fp_edge"] is None and s.room_index.adj == {}
    assert s.room_index.warp_count == 1
    assert s.room_index.warps[0]["src"] == 0
    assert s.room_index.warps[0]["d_scene"] == 2
    # Identity adopted => the room is keyable/routable-to, just never
    # via a warp edge; and the worker is live there.
    assert c["fp_live"] is True


def test_an_uncommitted_stage_is_dropped_next_step_never_late():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3))
    _step(s, c, _nt(4), odo=(0, 0))
    _step(s, c, _nt(4), odo=(256, 0))
    assert c["fp_edge"] is not None
    _step(s, c, _nt(4), odo=(256, 0))  # no transit consumed it
    assert c["fp_edge"] is None
    assert s._room_edges_dropped == 1 and s.room_index.adj == {}


def test_blank_frames_reset_the_pend_and_count_rejects():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3))
    _step(s, c, _nt(4))                # churn begins
    _step(s, c, _nt(4), lines=0)       # fade blanks the screen
    assert c["fp_pend"] is None and s._room_settle_rejects == 1
    _step(s, c, _nt(4))
    _step(s, c, _nt(4))                # fresh churn settles cleanly
    assert int(s._room_ord[0]) == 1


def test_cap_hold_breaks_the_chain_and_never_bridges_an_edge():
    s = _hot_self(cap=2)
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))   # A=0
    s.room_index.intern(nt_fingerprint(_nt(4), s._room_mask))   # B=1
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3))                                # live in A
    _step(s, c, _nt(9))
    _step(s, c, _nt(9))                                # C: cap hit
    assert int(s._room_ord[0]) == 0                    # hold-last
    assert c["fp_live"] is False and s.room_index.cap_hits == 1
    # A→C→B traversal: the B settle must NOT mint A→B.
    _step(s, c, _nt(4))
    _step(s, c, _nt(4))
    assert int(s._room_ord[0]) == 1                    # adopted B
    assert c["fp_edge"] is None and s.room_index.adj == {}
    assert c["fp_live"] is True                        # B itself live


def test_sample_every_skips_the_hash_but_keeps_the_ring():
    s = _hot_self(cfg=_cfg(mask=[], settle=2, sample_every=2))
    c = _ctx(s)
    _step(s, c, _nt(3), action=7)      # burst_step 0: sampled
    _step(s, c, _nt(3), action=8)      # burst_step 1: skipped
    assert s.pool.peeks == 1
    assert list(c["fp_ring"]) == [7, 8]


def test_the_ring_carries_at_most_the_last_32_actions():
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))
    c = _ctx(s, psig=(9, 9, 0, 0))
    for a in range(40):
        _step(s, c, _nt(3), action=a)
    _step(s, c, _nt(4), action=40)
    _step(s, c, _nt(4), action=41)
    acts = c["fp_edge"][6]
    assert len(acts) == 32 and acts[-1] == 41 and acts[0] == 10


def test_the_restore_transit_tripwire_counts_only_first_step_transits():
    s = _hot_self()
    s._room_transit({"burst_step": 1})
    assert s._room_restore_transits == 1
    s._room_transit({"burst_step": 2})
    assert s._room_restore_transits == 1


def test_onset_baseline_is_the_pre_churn_sample_so_a_straddled_warp_classifies():
    """The RG-0-falsified convention (docs/receipts/room_fp/zelda.md
    deviation 2): the measured Zelda death flash bumps the scene
    ordinal in the very sampled step of its first NT rewrite. The
    churn onset must therefore integrate from the PREVIOUS rendered
    sample — stamped with the current one, Δscene reads 0 and the
    death mis-classifies fade, minting the exact edge the warp veto
    exists to refuse."""
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))    # A = 0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3), scene=0)       # steady in A: baseline scene 0
    assert c["fp_base"] == ((0, 0), 0)
    # First flip arrives WITH the scene already bumped +2 (the
    # straddle), odometer flat; settle=2 fires on the repeat.
    _step(s, c, _nt(7), scene=2)
    _step(s, c, _nt(7), scene=2)
    assert int(s._room_ord[0]) == 1
    assert s.room_index.warp_count == 1        # warp, not fade
    assert c["fp_edge"] is None and s.room_index.adj == {}


def test_a_blank_breaks_the_onset_baseline():
    """Across a lines<min_lines gap the pre-blank sample is no
    baseline: the next rendered sample re-anchors on itself (matching
    replay_room_stream), so a scene delta swallowed by the blank can
    never be attributed to the following settle."""
    s = _hot_self()
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))    # A = 0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3), scene=0)                 # baseline armed
    _step(s, c, _nt(3), scene=0, lines=0)        # blank: baseline broken
    assert c["fp_base"] is None
    _step(s, c, _nt(7), scene=2)                 # onset = this sample
    _step(s, c, _nt(7), scene=2)
    assert int(s._room_ord[0]) == 1
    # Δscene measured 0 from the self-anchored onset: fade, not warp.
    assert s.room_index.warp_count == 0
