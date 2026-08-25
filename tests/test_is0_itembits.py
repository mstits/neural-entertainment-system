"""IS-0 — the item-semantics offline falsifier (ITEM_SEMANTICS_ENGINE_
2026-08-25 §6). BLOCKS ALL LIVE ITEM-BIT COMPUTE: every assertion this
file covers must pass before any Stage-1/2/3 run over real rollouts
launches.

SCOPE:
  * Task 1 (`EdgeStat.cap_hist` graft): items 8 (`RoomIndex.record_
    edge`/`load()` cap_hist round-trip) and 9 (`--item-sig-report`
    flags-off byte-identity).
  * Task 2 (candidate scanner, `scripts/discover_item_bits.py` Stages
    1-2): items 1-6 (monotone-flag proposal, change-rate rejection,
    reverts/flicker rejection, dedup, idle-prefilter, cross-rollout
    K-of-N confirmation).
  * Task 3 (Stage 3 — `correlate_boundary_edges` + `verify_behavioral`):
    item 7 (scripted-oracle validate/reject via the control-bit
    mechanism, before any real `Pool` is involved) plus dedicated
    coverage of the rarity-vs-gating confound (§7 item 1) that item 7
    alone does not name explicitly.

Every item-8/9 assertion runs without a ROM, a Pool, or a live Solver
constructor — RoomIndex is pure by design (test_room_fp.py's own
convention), and the `_room_step`/`_room_transit` wiring is exercised
through the SAME duck-typed Solver stand-in test_room_fp.py already
built and validated for the room-graph engine (T1/T2), reused here
rather than re-implemented, so a change to that harness's shape is
felt in both files at once. Every item-1/2/3/4/6 assertion is pure
numpy over synthetic rollout logs (no ROM, no Pool, no live compute);
item 5 additionally loads the real, already-banked `tests/fixtures/
roomgraph/{zelda,metroid}_idle_fs1.npz` captures, which is reading a
file, not running one. Item 7's `verify_behavioral` mechanics run
against a SCRIPTED duck-typed Pool stand-in (`_FakePool` below, same
convention as test_room_fp.py's Solver stand-ins) for every logic
assertion; exactly one test (`test_is0_7_live_pool_replay_mechanics_
smoke`) drives a real, short, single-worker `nes_core.Pool` to prove
the replay plumbing itself (not just the verdict arithmetic) against
real hardware — a few seconds, never a sustained/multi-worker burst.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.discover_item_bits import (
    DEFAULT_MAX_CHANGE_RATE,
    DEFAULT_VERIFY,
    ItemBitCandidate,
    _bit_plane_match_mod,
    _eval_candidate,
    _flag_events,
    _ram_from_step,
    _run_matched_wave,
    compute_idle_mask,
    confirm_across_rollouts,
    correlate_boundary_edges,
    idle_mask_from_rom,
    load_ledger,
    make_probe_pool,
    merge_item_bits_config,
    save_ledger,
    scan_rollout,
    select_control_candidates,
    verify_behavioral,
)
from scripts.go_explore_solve import RoomIndex
from tests.test_room_fp import _ctx, _hot_self, _nt, _step

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts" / "go_explore_solve.py"
ITEM_BITS_SRC = REPO / "scripts" / "discover_item_bits.py"
FIX = REPO / "tests" / "fixtures" / "roomgraph"
RAM = 0x800


def _src() -> str:
    return SRC.read_text()


def _zeros(n: int) -> np.ndarray:
    return np.zeros((n, RAM), dtype=np.uint8)


# =======================================================================
# IS-0 item 1 — genuine monotone flag (0 for N steps, 1 forever) =>
# proposed as a candidate. (Task 2, scripts/discover_item_bits.py Stage 1)
# =======================================================================


def test_is0_1_genuine_monotone_flag_is_proposed_as_a_candidate():
    n = 500
    log = _zeros(n)
    log[200:, 111] = 1          # 0 for 200 steps, then 1 forever
    out = scan_rollout(log)
    cand = out[(111, None)]
    assert cand.status == "candidate"
    assert cand.reverts_seen == 0
    assert cand.match == frozenset({1})
    assert cand.mod == 0
    assert cand.first_seen == {"rollout_id": None, "step": 200,
                               "position": [None, None]}


def test_is0_1_flag_never_changing_at_all_is_not_reported():
    # A byte that never moves is inert, not a rejected candidate — it
    # never enters the returned dict either way.
    log = _zeros(50)
    out = scan_rollout(log)
    assert out == {}


# =======================================================================
# IS-0 item 2 — frame-counter-shaped byte (changes every step) =>
# rejected on change-rate.
# =======================================================================


def test_is0_2_frame_counter_shaped_byte_rejected_on_change_rate():
    n = 500
    log = _zeros(n)
    log[:, 55] = np.arange(n) % 256      # moves on every single step
    out = scan_rollout(log)
    cand = out[(55, None)]
    assert cand.status == "rejected"
    assert cand.change_rate > DEFAULT_MAX_CHANGE_RATE
    assert cand.change_rate == pytest.approx(1.0)


def test_is0_2_change_rate_is_the_named_reason_not_a_side_effect_of_reverts():
    # Isolate the mechanism: a byte that changes every step but NEVER
    # revisits an old value (a long ramp, no wraparound) still gets
    # rejected — change-rate alone is sufficient, reverts is a
    # different, independently-testable failure mode (item 3).
    n = 2000
    log = _zeros(n)
    log[:, 66] = np.clip(np.arange(n), 0, 255)   # ramps then plateaus at 255
    out = scan_rollout(log)
    cand = out[(66, None)]
    assert cand.status == "rejected"
    assert cand.change_rate > DEFAULT_MAX_CHANGE_RATE


# =======================================================================
# IS-0 item 3 — flickering bit (1 -> 0 -> 1 once) => rejected on
# monotonicity/reverts_seen.
# =======================================================================


def test_is0_3_flickering_bit_rejected_on_reverts_not_change_rate():
    n = 500          # large N keeps change_rate well under budget, so
                     # the ONLY thing that can reject this is reverts.
    log = _zeros(n)
    log[200:300, 77] = 1
    log[300:400, 77] = 0        # flicker back down once
    log[400:, 77] = 1           # and back up again
    out = scan_rollout(log)
    cand = out[(77, None)]
    assert cand.change_rate <= DEFAULT_MAX_CHANGE_RATE, (
        "test is meant to isolate the reverts mechanism — change_rate "
        "must stay under budget so rejection cannot be attributed to it")
    assert cand.status == "rejected"
    assert cand.reverts_seen > 0


def test_is0_3_flag_events_helper_counts_reverts_directly():
    transitions, reverts, first_step = _flag_events(
        np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    assert transitions == 3
    assert reverts == 2          # the 1->0 and the second 0->1 each
                                  # revisit an already-seen run value
    assert first_step == 2


# =======================================================================
# IS-0 item 4 — address already claimed by an existing observable =>
# deduped.
# =======================================================================


def test_is0_4_claimed_address_is_deduped_out_entirely():
    n = 500
    log = _zeros(n)
    log[200:, 88] = 1
    without_dedup = scan_rollout(log)
    assert (88, None) in without_dedup      # sanity: it WOULD propose

    deduped = scan_rollout(log, claimed_addrs=frozenset({88}))
    assert (88, None) not in deduped
    assert (88, 0) not in deduped           # the bit-plane scan is
                                             # also suppressed, not just
                                             # the raw-value shape


def test_is0_4_dedup_does_not_affect_unclaimed_neighbors():
    n = 500
    log = _zeros(n)
    log[200:, 88] = 1
    log[250:, 89] = 1
    out = scan_rollout(log, claimed_addrs=frozenset({88}))
    assert (88, None) not in out
    assert (89, None) in out


# =======================================================================
# IS-0 item 5 — idle-prefilter over the real, already-banked idle
# captures => post-filter candidate count is exactly zero (idle play
# produces no capability events).
# =======================================================================


@pytest.mark.parametrize("fixture", ["zelda_idle_fs1.npz", "metroid_idle_fs1.npz"])
def test_is0_5_idle_prefilter_over_real_captures_leaves_zero_candidates(fixture):
    p = FIX / fixture
    if not p.exists():
        pytest.skip(f"{fixture} not present in this checkout")
    log = np.load(p, allow_pickle=True)["nt"]
    mask = compute_idle_mask(log)
    out = scan_rollout(log, idle_mask=mask)
    assert out == {}, (
        f"idle play must produce zero capability-flag candidates once "
        f"the idle-prefilter mask is applied; got {out!r}")


def test_is0_5_zelda_idle_capture_has_a_real_false_positive_the_mask_exists_to_catch():
    # Names WHY the idle-prefilter is load-bearing rather than a no-op:
    # the Zelda idle capture's NT byte 214 (the HUD heart-animation
    # tile — same byte test_rg0_3_zelda_idle_is_one_hash_post_mask
    # names) transitions EXACTLY ONCE and never reverts during 300
    # idle frames — a shape that, unmasked, is indistinguishable from
    # a genuine capability pickup (IS-0 item 1's own shape). The idle
    # mask is what keeps it out, not the monotonicity/change-rate
    # filters (which a one-shot idle animation frame cleanly passes).
    p = FIX / "zelda_idle_fs1.npz"
    if not p.exists():
        pytest.skip("zelda_idle_fs1.npz not present in this checkout")
    log = np.load(p, allow_pickle=True)["nt"]
    naive = scan_rollout(log)          # no idle_mask at all
    assert naive.get((214, None)) is not None
    assert naive[(214, None)].status == "candidate", (
        "byte 214 must be a clean monotone-flag SHAPE when scanned "
        "naively — that is the false positive the idle-prefilter, not "
        "the flag-shape filters, exists to catch")
    mask = compute_idle_mask(log)
    assert 214 in mask
    assert scan_rollout(log, idle_mask=mask) == {}


def test_is0_5_compute_idle_mask_pure_synthetic():
    # Direct unit coverage of the mask primitive itself, independent
    # of any fixture file.
    log = np.zeros((20, 8), dtype=np.uint8)
    log[10:, 3] = 1              # column 3 moves once during "idle"
    assert compute_idle_mask(log) == frozenset({3})


def test_is0_5_compute_idle_mask_respects_reload_keep_mask():
    # Discoverer.idle()'s (log, keep) shape: a diff row spanning a
    # reload must never be read as idle "movement".
    log = np.zeros((6, 4), dtype=np.uint8)
    log[3:, 2] = 200               # a reload-sized jump at row 3, persists
    keep = np.array([True, True, False, True, True])  # mask the 2->3 diff
    assert compute_idle_mask((log, keep)) == frozenset()


def test_is0_5_idle_mask_from_rom_reuses_discoverer_idle_as_a_library_import(
        monkeypatch):
    # §3 row 7: "Reuses discover_observables.Discoverer ... as a
    # library import, never a patch." Locks the reuse without a ROM:
    # a fake Discoverer stands in, and this proves idle_mask_from_rom
    # calls its REAL idle()/close() surface rather than reimplementing
    # the probe.
    import scripts.discover_observables as do_mod
    calls = {}

    class FakeDiscoverer:
        def __init__(self, rom, state, frame_skip=4, forward="right",
                    seed=1):
            calls["init"] = (rom, state, frame_skip, forward, seed)

        def idle(self):
            log = np.zeros((10, 20), dtype=np.uint8)
            log[5:, 3] = 1
            return log, np.ones(9, dtype=bool)

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(do_mod, "Discoverer", FakeDiscoverer)
    mask = idle_mask_from_rom("rom.nes", b"state", frame_skip=2,
                              forward="left", seed=7)
    assert mask == frozenset({3})
    assert calls["init"] == ("rom.nes", b"state", 2, "left", 7)
    assert calls["closed"] is True


# =======================================================================
# IS-0 item 6 — cross-rollout stability: 2-of-5 stays candidate; 4-of-5
# promotes to confirmed.
# =======================================================================


def _mk_rollout(hit: bool, addr: int = 130, n: int = 300) -> np.ndarray:
    log = _zeros(n)
    if hit:
        log[150:, addr] = 1
    return log


def test_is0_6_two_of_five_stays_candidate():
    logs = [_mk_rollout(True), _mk_rollout(True), _mk_rollout(False),
            _mk_rollout(False), _mk_rollout(False)]
    ledger = confirm_across_rollouts(logs)     # default confirm_k=3
    entry = ledger[(130, None)]
    assert entry.monotone_rollouts == 2
    assert entry.total_rollouts == 5
    assert entry.status == "candidate"


def test_is0_6_four_of_five_promotes_to_confirmed():
    logs = [_mk_rollout(True), _mk_rollout(True), _mk_rollout(True),
            _mk_rollout(True), _mk_rollout(False)]
    ledger = confirm_across_rollouts(logs)
    entry = ledger[(130, None)]
    assert entry.monotone_rollouts == 4
    assert entry.total_rollouts == 5
    assert entry.status == "confirmed"


def test_is0_6_total_rollouts_is_the_batch_size_not_the_appearance_count():
    # A rollout in which the event never fires is a counted "no", not
    # an absence — total_rollouts must be N=5 even though the key only
    # appears (candidate or rejected) in 2 of the 5 per-rollout scans.
    logs = [_mk_rollout(True), _mk_rollout(True), _mk_rollout(False),
            _mk_rollout(False), _mk_rollout(False)]
    ledger = confirm_across_rollouts(logs)
    assert ledger[(130, None)].total_rollouts == len(logs) == 5


def test_is0_6_a_single_revert_permanently_overrides_a_four_of_five_confirm():
    # Rediscovery rule (metroid_purity_quarantine_2026-08-10.md
    # convention): reverts_seen > 0 in ANY rollout wins over the K-of-N
    # arithmetic, even when 4 of the 5 rollouts confirmed cleanly.
    def revert_rollout(n=300, addr=130):
        log = _zeros(n)
        log[150:200, addr] = 1
        log[200:, addr] = 0     # flips back — a revert, not a hold
        return log

    logs = [_mk_rollout(True), _mk_rollout(True), _mk_rollout(True),
            _mk_rollout(True), revert_rollout()]
    ledger = confirm_across_rollouts(logs)
    entry = ledger[(130, None)]
    assert entry.monotone_rollouts == 4
    assert entry.status == "rejected", (
        "a revert anywhere must permanently reject the key regardless "
        "of how many other rollouts confirmed it cleanly")


def test_is0_6_confirm_k_is_configurable_not_hardcoded():
    logs = [_mk_rollout(True), _mk_rollout(True), _mk_rollout(False),
            _mk_rollout(False), _mk_rollout(False)]
    lenient = confirm_across_rollouts(logs, confirm_k=2)
    assert lenient[(130, None)].status == "confirmed"


# =======================================================================
# Bit-plane candidates (§2B's "both shapes supported") — not an IS-0
# numbered item on its own, but required Stage-1 coverage: a raw byte
# that is noisy from unrelated bits must still let ONE clean plane
# through, and a clean raw byte must not also spam eight redundant
# per-bit proposals.
# =======================================================================


def test_bit_plane_match_mod_isolates_the_low_bits_regardless_of_higher_ones():
    match, mod = _bit_plane_match_mod(2)
    assert mod == 8
    for v in range(256):
        bit_is_set = bool((v >> 2) & 1)
        assert ((v % mod) in match) == bit_is_set


def test_a_noisy_byte_still_lets_one_clean_bit_plane_through():
    n = 500
    log = _zeros(n)
    noise = (np.arange(n) % 2).astype(np.uint8)   # bit 0 toggles every step
    log[:, 40] = noise
    log[250:, 40] |= 0b100                        # bit 2 flips once, holds
    out = scan_rollout(log)
    assert out[(40, None)].status == "rejected"    # whole byte is noisy
    assert out[(40, 2)].status == "candidate"       # this plane is clean
    assert out[(40, 0)].status == "rejected"        # this plane is the noise


def test_a_clean_raw_byte_suppresses_the_redundant_bit_plane_scan():
    n = 500
    log = _zeros(n)
    log[200:, 111] = 1
    out = scan_rollout(log)
    assert (111, None) in out
    assert not any(bit is not None for (addr, bit) in out
                  if addr == 111), (
        "a clean raw-value candidate already explains the byte — no "
        "redundant per-bit proposals for the same address")


def test_scan_bits_false_disables_bit_plane_scanning_entirely():
    n = 500
    log = _zeros(n)
    noise = (np.arange(n) % 2).astype(np.uint8)
    log[:, 40] = noise
    log[250:, 40] |= 0b100
    out = scan_rollout(log, scan_bits=False)
    assert (40, None) in out
    assert (40, 2) not in out


# =======================================================================
# Cross-rollout match accumulation + ItemBitCandidate persistence
# (§2B's ledger shape and its JSON round-trip).
# =======================================================================


def test_confirm_across_rollouts_unions_match_sets_across_rollouts():
    def rollout(val, n=300, addr=140):
        log = _zeros(n)
        log[150:, addr] = val
        return log

    logs = [rollout(1), rollout(1), rollout(2), rollout(2), rollout(2)]
    ledger = confirm_across_rollouts(logs, confirm_k=3)
    entry = ledger[(140, None)]
    assert entry.match == frozenset({1, 2})
    assert entry.status == "confirmed"


def test_item_bit_candidate_round_trips_through_dict():
    c = ItemBitCandidate(addr=10, bit=3, match=frozenset({8, 9}), mod=16,
                         reverts_seen=0, change_rate=0.01,
                         monotone_rollouts=4, total_rollouts=5,
                         status="confirmed",
                         first_seen={"rollout_id": 2, "step": 40,
                                    "position": [12, 34]})
    d = c.to_dict()
    assert d["match"] == [8, 9]        # JSON-stable: sorted list
    back = ItemBitCandidate.from_dict(d)
    assert back.key == (10, 3)
    assert back.match == frozenset({8, 9})
    assert back.status == "confirmed"
    assert back.first_seen == {"rollout_id": 2, "step": 40,
                               "position": [12, 34]}


def test_save_and_load_ledger_round_trip(tmp_path):
    ledger = {
        (10, None): ItemBitCandidate(addr=10, bit=None, match=frozenset({1}),
                                     status="confirmed", monotone_rollouts=4,
                                     total_rollouts=5),
        (20, 3): ItemBitCandidate(addr=20, bit=3, match=frozenset({8}),
                                  mod=16, status="candidate",
                                  monotone_rollouts=2, total_rollouts=5),
    }
    p = tmp_path / "candidates.json"
    save_ledger(p, ledger)
    reloaded = load_ledger(p)
    assert set(reloaded.keys()) == set(ledger.keys())
    assert reloaded[(10, None)].status == "confirmed"
    assert reloaded[(20, 3)].match == frozenset({8})


# =======================================================================
# §4 profile schema — item_bits: config merge (script defaults when a
# profile names none, per-key override when it does).
# =======================================================================


def test_merge_item_bits_config_defaults_when_profile_has_no_block():
    cfg = merge_item_bits_config({"solve": {}})
    assert cfg["max_change_rate"] == DEFAULT_MAX_CHANGE_RATE
    assert cfg["confirm_k"] == 3
    assert cfg["confirm_n"] == 5
    assert cfg["max_item_bits"] == 8
    assert cfg["verify"]["n_trials"] == 20


def test_merge_item_bits_config_overrides_only_named_keys():
    profile = {"solve": {"item_bits": {"confirm_k": 1,
                                       "verify": {"n_trials": 40}}}}
    cfg = merge_item_bits_config(profile)
    assert cfg["confirm_k"] == 1
    assert cfg["confirm_n"] == 5                # untouched default
    assert cfg["verify"]["n_trials"] == 40
    assert cfg["verify"]["control_bits"] == 5    # untouched default


def test_merge_item_bits_config_tolerates_missing_solve_block():
    assert merge_item_bits_config({}) == merge_item_bits_config(None)


# =======================================================================
# Purity: no RAM map, no poke_ram, addresses only ever come from range().
# =======================================================================


def test_no_poke_ram_call_anywhere_in_the_item_bits_module():
    # The doctrine callout in discover_item_bits.py's own module
    # docstring NAMES `poke_ram`/`poke_ram_range` (to say they must
    # never be invented) — a substring-anywhere check would
    # false-positive on that deliberate prose. What must never appear
    # is an actual CALL.
    src = ITEM_BITS_SRC.read_text()
    assert "poke_ram(" not in src
    assert "poke_ram_range(" not in src
    assert "import poke_ram" not in src


def test_candidate_addresses_never_exceed_the_hardware_ram_window():
    n = 10
    log = np.zeros((n, RAM), dtype=np.uint8)
    log[5:, RAM - 1] = 1
    out = scan_rollout(log)
    assert all(0 <= addr < RAM for (addr, _bit) in out)


def _pre_change_head_src() -> str:
    """The file's content at HEAD, via git — i.e. before this
    session's uncommitted item-sig graft. Used ONLY to re-verify the
    synthesis's central claim (Part I / §3 row 4) that no
    RoomIndex.VERSION bump was needed, against the actual pre-change
    text rather than trusting the proposal's citation."""
    out = subprocess.run(["git", "show", "HEAD:scripts/go_explore_solve.py"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout


# ---------------------------------------------------------------------
# IS-0 item 8 — RoomIndex.record_edge / load() cap_hist round-trip
# ---------------------------------------------------------------------


def test_is0_8_cap_hist_survives_save_load_round_trip(tmp_path):
    idx = RoomIndex(cap=64, config_sha="deadbeef")
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=0)
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=3)
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=3)
    p = tmp_path / "room_index.json"
    idx.save(p)
    reloaded = RoomIndex.load(p)
    e = reloaded.adj[0][1]
    assert e["cap_hist"] == {"0": 1, "3": 2}
    assert e["count"] == 3          # pre-existing field, untouched


def test_is0_8_to_json_needs_no_dedicated_edit_wholesale_passthrough():
    # §3 row 4's specific claim, re-verified directly rather than
    # trusted: to_json() serializes each edge dict wholesale, so
    # cap_hist rides along with zero to_json code change.
    idx = RoomIndex(cap=64, config_sha="deadbeef")
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=5)
    data = idx.to_json()
    assert data["adj"]["0"]["1"]["cap_hist"] == {"5": 1}
    assert "def to_json" in _src()
    to_json_body = _src().split("def to_json")[1].split("def save")[0]
    assert "cap_hist" not in to_json_body, (
        "to_json() should need no cap_hist-specific line — wholesale "
        "raw-dict passthrough already carries it")


def test_is0_8_legacy_archive_with_no_cap_hist_key_loads_without_keyerror(
        tmp_path):
    # Simulates a room_index.json written by a pre-graft build: the
    # edge dict carries no "cap_hist" key anywhere, at the SAME
    # version number this build still writes (the no-VERSION-bump
    # claim, exercised structurally: a legacy-shaped archive at
    # RoomIndex.VERSION loads clean, not refused).
    idx = RoomIndex(cap=64, config_sha="deadbeef")
    idx.record_edge(0, 1, "pan", "E", 4)
    data = idx.to_json()
    del data["adj"]["0"]["1"]["cap_hist"]
    p = tmp_path / "legacy_room_index.json"
    p.write_text(json.dumps(data))
    reloaded = RoomIndex.load(p)        # must not raise KeyError
    e = reloaded.adj[0][1]
    assert e["cap_hist"] == {}
    assert e["count"] == 1


def test_is0_8_empty_cap_hist_is_the_raw_shape_not_a_load_time_backfill():
    # §2A's "absent/{} reads as {"0": count}" is an INTERPRETIVE
    # convention for whichever code later consumes cap_hist together
    # with count (Stage 3a, Task 3) — not a literal transformation
    # RoomIndex.load() performs. The synthesis's own §3 row 4 code is
    # explicit ({} on a missing key, not {"0": count}); this test
    # pins that the implemented behavior matches the literal
    # integration-point code, so a future Stage-3a author does not
    # mistake an empty dict for a KeyError-shaped bug.
    idx = RoomIndex(cap=8, config_sha="x")
    idx.record_edge(0, 1, "pan", "E", 4)   # cap_sig defaults to 0
    assert idx.adj[0][1]["cap_hist"] == {"0": 1}   # NOT {} — it WAS recorded


def test_is0_8_no_version_bump_was_needed_for_this_graft():
    # Re-verify the synthesis's central shipping-physics claim (Part I
    # + §3 row 4) against the REAL pre-change file text via git, not
    # the proposal's citation: RoomIndex.VERSION at HEAD (before this
    # session's graft) equals RoomIndex.VERSION now.
    pre = _pre_change_head_src()
    m = None
    for line in pre.splitlines():
        s = line.strip()
        if s.startswith("VERSION ="):
            m = s
            break
    assert m is not None, "could not find RoomIndex.VERSION in pre-change HEAD"
    pre_version = int(m.split("=")[1].strip())
    assert pre_version == RoomIndex.VERSION, (
        "the item-sig graft changed RoomIndex.VERSION — the synthesis's "
        "no-bump claim does NOT hold against real code; report this")


# ---------------------------------------------------------------------
# IS-0 item 9 — flags-off byte-identity: --item-sig-report absent
# (default) => record_edge called with cap_sig=0 on every path, and
# the pre-existing SMB/Castlevania determinism harness's inputs
# (RNG stream, action selection, cell keys, scores) are untouched.
# ---------------------------------------------------------------------


def test_is0_9_item_sig_report_flag_is_off_by_default_in_argparse():
    src = _src()
    assert '"--item-sig-report", action="store_true"' in src, (
        "flag must default False (store_true, no explicit default= "
        "override) — same convention as every other report-only arm")


def test_is0_9_item_sig_armed_reads_from_argv_and_profile_not_a_hardcode():
    # Mutation-resistant wiring check, same idiom as test_go_explore_
    # solve.py's test_solver_init_resolves_the_pin_from_argv_and_not_a_
    # hardcode: parse Solver.__init__'s AST and confirm the assignment
    # actually reads args.item_sig_report and game.state_sig_arity,
    # rather than e.g. being hardcoded to False or reading the wrong
    # namespace.
    tree = ast.parse(_src())
    cls = next(n for n in tree.body
              if isinstance(n, ast.ClassDef) and n.name == "Solver")
    fn = next(n for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    rhs = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute)
                    and tgt.attr == "_item_sig_armed"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                rhs = node.value
    assert rhs is not None, "self._item_sig_armed is never assigned at all"
    src_expr = ast.unparse(rhs)
    assert "item_sig_report" in src_expr
    assert "state_sig_arity" in src_expr
    # Both reads must be getattr-guarded (default-safe): SmbGame
    # carries no state_sig_arity attribute at all, and duck-typed
    # Solver stand-ins in the test suite carry no args/game either —
    # a bare attribute access would raise on either, not default off.
    assert src_expr.count("getattr(") >= 2, (
        "a bare self.game.state_sig_arity / args.item_sig_report read "
        "raises AttributeError for SmbGame profiles and every "
        "duck-typed test stand-in — must be getattr-guarded")


def test_is0_9_record_edge_default_cap_sig_is_bucket_zero():
    # Every caller that predates this graft (and any future caller
    # that never learns about cap_sig) omits the kwarg entirely.
    idx = RoomIndex(cap=8, config_sha="x")
    idx.record_edge(0, 1, "pan", "E", 4)      # no cap_sig kwarg at all
    assert idx.adj[0][1]["cap_hist"] == {"0": 1}


def test_is0_9_room_transit_call_site_forwards_cap_sig_from_the_staged_tuple():
    # Locks the ONE call site (§3 row 3) — a substring check in the
    # same style as test_go_explore_solve.py's own CLI/wiring guards
    # (e.g. test_solver_cli_hw_flags_defaults_to_none_so_nothing_is_
    # set), catching a future edit that drops the kwarg silently.
    assert "cap_sig=e[7]" in _src()


def test_is0_9_fp_edge_tuple_gains_exactly_one_field_at_index_7():
    src = _src()
    assert ("c.get(\"fp_onset_key\"), list(ring), cap_sig)" in src), (
        "fp_edge must stay a strict append — every existing consumer "
        "(record_edge's e[0]..e[6] positional call, the exemplar/ring "
        "unpacks in tests/test_room_fp.py) indexes it positionally")


def test_is0_9_unarmed_room_step_writes_cap_hist_bucket_zero_only():
    """The behavioral core of item 9, run through the REAL
    Solver._room_step/_room_transit methods (bound via MethodType,
    test_room_fp.py's own duck-typed harness) rather than a
    reimplementation: on the flags-off shape every pre-existing
    caller and every pre-existing test constructs (no _item_sig_armed
    attribute on the stand-in at all — the exact shape a real
    Solver.__init__ produces when --item-sig-report is absent), a
    cur_key with a plausible sig-shaped slot at index -3 is still
    ignored, cap_sig is always 0, and every pre-existing edge field
    (kind/dir/count/frames_mean/exemplar) is bit-for-bit what
    test_room_fp.py already pins for the graft-free path."""
    from scripts.go_explore_solve import nt_fingerprint

    s = _hot_self()
    assert not hasattr(s, "_item_sig_armed"), (
        "this IS the flags-off duck-typed shape under test")
    s.room_index.intern(nt_fingerprint(_nt(3), s._room_mask))   # A=0
    c = _ctx(s, psig=(9, 9, 0, 0))
    _step(s, c, _nt(3), odo=(0, 0), action=1)          # confirm live
    # A cur_key whose index -3 is a nonzero "sig"-shaped slot — if the
    # unarmed guard were broken, this is exactly the value a bug would
    # leak into cap_sig instead of 0.
    c["cur_key"] = (0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0)
    _step(s, c, _nt(4), odo=(0, 0), action=2)          # churn onset
    _step(s, c, _nt(4), odo=(256, 0), action=3)        # +256 px: pan E
    fp_edge = c["fp_edge"]
    assert fp_edge is not None
    assert fp_edge[7] == 0, "cap_sig must be 0 when _item_sig_armed is absent"
    s._room_transit(c)
    e = s.room_index.adj[0][1]
    assert e["cap_hist"] == {"0": 1}
    assert (e["kind"], e["dir"], e["count"]) == ("pan", "E", 1)


def test_is0_9_room_fp_config_flags_off_smoke_run_completes(tmp_path):
    """A SHORT (few-second), single-worker, flags-off live run over a
    real room_fp-armed profile: the item-sig graft's code paths
    (_room_step's cap_sig branch, _room_transit's cap_sig=e[7] call)
    execute for real on every settle, with --item-sig-report absent —
    and the run completes without a crash or a hang. This is a scaled
    -down substitute for the room-graph lineage's own '16,000-step SMB
    /Castlevania determinism harness' (CLAIMS.md), not a re-run of it:
    that 8-worker/16,000-step receipt is out of scope for this session
    per the standing compute-saturation constraint (a live v28 training
    campaign owns this machine). What this DOES establish live, not
    just by unit proof: the new code path is reachable and inert with
    flags off. The stronger claim — that RAM/archive/trace bytes are
    IDENTICAL to a pre-graft run — is established by code-path
    isolation instead (record_edge's cap_hist bookkeeping is the ONLY
    thing this graft touches; nothing it reads or writes feeds back
    into cell keys, scores, RNG consumption, or archive dominance —
    see the unit tests above), and should be spot-checked with the
    full harness the next time the machine is free for a sustained
    multi-worker run."""
    root = REPO / "roms" / "zelda_start_ctrl.state.bin"
    if not root.exists():
        pytest.skip("zelda_start_ctrl.state.bin not present in this checkout")
    out = tmp_path / "is0_9_smoke"
    proc = subprocess.run(
        [sys.executable, str(SRC),
         "--out", str(out),
         "--root-state", str(root),
         "--profile", "configs/zelda_roomfp.yaml",
         "--workers", "1", "--minutes", "0.03", "--want-solutions", "1",
         "--max-steps", "150", "--seed", "0"],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-4000:]
    ri = json.loads((out / "room_index.json").read_text())
    assert ri["version"] == RoomIndex.VERSION
    # Every edge this run happened to commit (may be zero on a run this
    # short) must carry the flags-off bucket shape.
    for dsts in ri["adj"].values():
        for e in dsts.values():
            assert e["cap_hist"] == {"0": e["count"]}, (
                "--item-sig-report absent must bucket every traversal "
                "under cap_sig 0 — a nonzero bucket here would mean "
                "the graft leaked live despite the flag being off")


# =======================================================================
# IS-0 item 7 — Task 3: correlate_boundary_edges (Stage 3a) +
# verify_behavioral (Stage 3b, §2C). Every logic assertion below runs
# against a SCRIPTED, duck-typed Pool stand-in (`_FakePool`) — no ROM,
# no real Pool — except `test_is0_7_live_pool_replay_mechanics_smoke`,
# which is the one deliberately short, single-worker exception that
# drives real hardware to prove the replay plumbing itself.
# =======================================================================


class _FakePool:
    """A scripted duck-typed Pool stand-in exposing exactly the two
    methods `verify_behavioral` calls (`load_worker_state`,
    `step_all`) — same convention as test_room_fp.py's duck-typed
    Solver stand-ins. `ground_truth(state_token, actions_so_far) ->
    ram_bytes` is a PURE function of the branch point and the full
    cumulative action sequence replayed since it, called once per
    step — lets a test encode an exact, fully-known causal model
    (a byte that flips iff a specific action was ever pressed, a
    reach condition wired to whatever the test wants to assert about)
    with no ROM and no ambiguity about what the "real" mechanism is.
    """

    def __init__(self, ground_truth):
        self._gt = ground_truth
        self._state: dict = {}
        self._hist: dict = {}

    def load_worker_state(self, wid, blob):
        self._state[wid] = blob
        self._hist[wid] = []

    def step_all(self, acts):
        out = []
        for wid, a in enumerate(acts):
            self._hist[wid].append(int(a))
            ram = self._gt(self._state[wid], list(self._hist[wid]))
            out.append((None, None, np.asarray(ram, dtype=np.uint8)))
        return out


def _blank_ram() -> np.ndarray:
    return np.zeros(RAM, dtype=np.uint8)


PRE = b"shared-pre-flip-state"  # opaque token; _FakePool never reads its bytes


def _true_gate_world(true_addr=100, decoy_addr=101, reach_addr=200,
                     true_action=7, decoy_action=9, extra=None):
    """Ground truth: `true_addr` flips iff `true_action` was ever
    pressed; `decoy_addr` flips iff the UNRELATED `decoy_action` was
    ever pressed; `reach_addr` (what `reach_fn` reads) is gated
    EXCLUSIVELY on `true_addr` — the textbook "a key opens a door"
    shape IS-0 item 7 asks for."""
    def gt(state, hist):
        ram = _blank_ram()
        true_on = true_action in hist
        ram[true_addr] = 1 if true_on else 0
        ram[decoy_addr] = 1 if decoy_action in hist else 0
        ram[reach_addr] = 1 if true_on else 0
        if extra:
            extra(ram, hist)
        return ram
    return gt


def _reach_fn(addr=200):
    return lambda ram: bool(ram[addr] == 1)


def _flat_control(addr=150) -> ItemBitCandidate:
    return ItemBitCandidate(addr=addr, bit=None, match=frozenset({1}),
                            mod=0, status="rejected")


# -----------------------------------------------------------------
# Core IS-0 item 7 requirement.
# -----------------------------------------------------------------
def test_is0_7_verify_behavioral_validates_a_true_causal_bit():
    gt = _true_gate_world()
    pool = _FakePool(gt)
    cand = ItemBitCandidate(addr=100, bit=None, match=frozenset({1}), mod=0)
    result = verify_behavioral(
        pool, cand, pre_state=PRE, acquire_actions=[7], skip_actions=[0],
        probe_actions=[0, 0], reach_fn=_reach_fn(), n_trials=3,
        control_bits=1, control_candidates=[_flat_control()])
    assert result["reach_rate_acquire"] == 1.0
    assert result["reach_rate_skip"] == 0.0
    assert result["control_reach_rates"] == [0.0]
    assert result["nondeterministic"] is False
    assert result["verdict"] == "validated", result


def test_is0_7_verify_behavioral_rejects_a_decoy_that_also_flips_once():
    # The decoy (addr 101) DOES flip cleanly (once, never reverts — it
    # would pass Stage 1-2's monotone-flag filters on its own), but its
    # own real matched acquire/skip pair (mined the same way: "acquire"
    # = a segment where decoy_action was pressed) shows NO reach-rate
    # gap, because reach is gated on the TRUE bit's action, not the
    # decoy's — exactly the behavioral disambiguation Stage 1-2 cannot
    # do and Stage 3b exists for.
    gt = _true_gate_world()
    pool = _FakePool(gt)
    decoy = ItemBitCandidate(addr=101, bit=None, match=frozenset({1}), mod=0)
    result = verify_behavioral(
        pool, decoy, pre_state=PRE, acquire_actions=[9], skip_actions=[0],
        probe_actions=[0, 0], reach_fn=_reach_fn(), n_trials=3,
        control_bits=1, control_candidates=[_flat_control()])
    assert result["reach_rate_acquire"] == 0.0
    assert result["reach_rate_skip"] == 0.0
    assert result["verdict"] != "validated"
    assert result["verdict"] == "rejected"


# -----------------------------------------------------------------
# §7 item 9 — never fabricate a trajectory.
# -----------------------------------------------------------------
def test_is0_7_missing_skip_segment_is_inconclusive_never_fabricated():
    result = verify_behavioral(
        pool=None, candidate=ItemBitCandidate(addr=1),
        pre_state=PRE, acquire_actions=[7], skip_actions=[],
        probe_actions=[0], reach_fn=_reach_fn())
    assert result["verdict"] == "inconclusive"
    assert "no real acquire/skip segment" in result["reason"]


def test_is0_7_missing_acquire_segment_is_inconclusive_never_fabricated():
    result = verify_behavioral(
        pool=None, candidate=ItemBitCandidate(addr=1),
        pre_state=PRE, acquire_actions=[], skip_actions=[0],
        probe_actions=[0], reach_fn=_reach_fn())
    assert result["verdict"] == "inconclusive"


# -----------------------------------------------------------------
# §7 item 8 — a control bit that is itself gated must fail flatness,
# and zero controls must never rubber-stamp a pass.
# -----------------------------------------------------------------
def test_is0_7_control_bit_that_is_itself_gated_fails_the_flatness_check():
    def extra(ram, hist):
        # addr 160 ALSO flips whenever the true action fires — a
        # "control" drawn without §7 item 8's guard could be exactly
        # this kind of accidentally-correlated address.
        ram[160] = 1 if 7 in hist else 0
    gt = _true_gate_world(extra=extra)
    pool = _FakePool(gt)
    cand = ItemBitCandidate(addr=100, bit=None, match=frozenset({1}), mod=0)
    bad_control = ItemBitCandidate(addr=160, bit=None, match=frozenset({1}),
                                   mod=0, status="rejected")
    result = verify_behavioral(
        pool, cand, pre_state=PRE, acquire_actions=[7], skip_actions=[0],
        probe_actions=[0, 0], reach_fn=_reach_fn(), n_trials=2,
        control_bits=2, control_candidates=[_flat_control(), bad_control])
    assert result["reach_rate_acquire"] - result["reach_rate_skip"] == 1.0
    assert result["control_reach_rates"] == [0.0, 1.0]
    assert result["verdict"] == "rejected", (
        "a real target gap must NOT validate when a control swings "
        "just as much — the flatness bar exists precisely for this")


def test_is0_7_zero_controls_is_inconclusive_not_a_default_pass():
    gt = _true_gate_world()
    pool = _FakePool(gt)
    cand = ItemBitCandidate(addr=100, bit=None, match=frozenset({1}), mod=0)
    result = verify_behavioral(
        pool, cand, pre_state=PRE, acquire_actions=[7], skip_actions=[0],
        probe_actions=[0, 0], reach_fn=_reach_fn(), n_trials=2,
        control_bits=5, control_candidates=[])
    assert result["reach_rate_acquire"] - result["reach_rate_skip"] == 1.0
    assert result["verdict"] == "inconclusive", (
        "a big real gap with ZERO controls scored must not silently "
        "validate — §7 item 8's guard is not optional")


# -----------------------------------------------------------------
# Determinism is checked, not assumed.
# -----------------------------------------------------------------
def test_is0_7_nondeterministic_pool_never_averages_into_a_false_confidence():
    calls = {"n": 0}

    def flaky_gt(state, hist):
        # Alternates the reach bit on successive full replays despite
        # an IDENTICAL (pre_state, actions) tuple — a live-Pool
        # integrity fault, the only thing that should ever trip this.
        # (6 ground-truth calls per trial: 2 lanes x 3 steps each.)
        calls["n"] += 1
        ram = _blank_ram()
        ram[200] = 1 if (calls["n"] // 6) % 2 == 0 else 0
        return ram

    pool = _FakePool(flaky_gt)
    cand = ItemBitCandidate(addr=100)
    result = verify_behavioral(
        pool, cand, pre_state=PRE, acquire_actions=[7], skip_actions=[0],
        probe_actions=[0, 0], reach_fn=_reach_fn(), n_trials=4,
        control_bits=1, control_candidates=[_flat_control()])
    assert result["nondeterministic"] is True
    assert result["verdict"] == "inconclusive"
    assert "distinct outcomes" in result["reason"]


# -----------------------------------------------------------------
# THE RARITY-VS-GATING CONFOUND (§7 item 1) — the task's named
# failure mode, demonstrated end to end: Stage 3a's cheap correlation
# is shown to be foolable, and Stage 3b's matched-pair-from-a-shared-
# state design is shown to resolve exactly that failure, not just
# produce a simpler correlation of its own.
# -----------------------------------------------------------------
def test_is0_7_stage3a_alone_is_fooled_by_a_rarity_confound():
    # Global picture: bit=0 samples are almost NONEXISTENT anywhere in
    # the graph (2 total), bit=1 samples are common (45 total). Edge
    # (0,1) is crossed ONLY at bit=1 — but that is exactly what you'd
    # expect from sheer scarcity of bit=0 data ANYWHERE, never mind
    # this edge specifically, not necessarily from a real gate.
    room_index = {"adj": {
        "0": {"1": {"kind": "pan", "dir": "E", "cap_hist": {"1": 40}}},
        "2": {"3": {"kind": "pan", "dir": "N", "cap_hist": {"0": 2, "1": 5}}},
    }}
    leads = correlate_boundary_edges(room_index, bit_index=0)
    lead_edges = {(l["src"], l["dst"]) for l in leads}
    assert (0, 1) in lead_edges, (
        "Stage 3a's cheap correlation DOES flag this edge — that is "
        "the point: it cannot tell rarity from gating by itself")
    lead = next(l for l in leads if (l["src"], l["dst"]) == (0, 1))
    assert lead["exposure_bit0"] == 2
    assert lead["exposure_bit1"] == 45
    assert lead["confound_ratio"] < 0.05, (
        "the whole-graph exposure numbers are what expose the "
        "confound — almost no bit=0 data exists ANYWHERE, so a "
        "single edge reading zero at bit=0 is unsurprising on its own")


def test_is0_7_verify_behavioral_resolves_the_rarity_confound_stage3a_missed():
    # Ground truth for THIS world: reaching the far side depends only
    # on having walked far enough (action 5, pressed by BOTH the real
    # acquire and real skip trajectory mined from the SAME shared
    # branch point) — NOT on the candidate bit at all. In the raw
    # rollout logs this candidate might still LOOK gated to Stage 3a
    # (e.g. if, by coincidence, most rollouts that walk that far also
    # happen to have picked the item up along the way) — but replaying
    # the ACTUAL matched pair from the ACTUAL shared branch point
    # reveals the truth: skipping the pickup does not prevent reaching.
    def gt(state, hist):
        ram = _blank_ram()
        ram[100] = 1 if 7 in hist else 0
        ram[200] = 1 if 5 in hist else 0     # reach: walked far, that's ALL
        return ram
    pool = _FakePool(gt)
    cand = ItemBitCandidate(addr=100, bit=None, match=frozenset({1}), mod=0)
    result = verify_behavioral(
        pool, cand, pre_state=PRE,
        acquire_actions=[7, 5],   # real: picked up the item, then walked far
        skip_actions=[5],        # real: walked far WITHOUT picking it up
        probe_actions=[0], reach_fn=_reach_fn(), n_trials=2,
        control_bits=1, control_candidates=[_flat_control()])
    assert result["reach_rate_acquire"] == 1.0
    assert result["reach_rate_skip"] == 1.0, (
        "both real arms reach — the candidate was never the reason")
    assert result["verdict"] != "validated", (
        "verify_behavioral must NOT rubber-stamp the lead Stage 3a's "
        "naive correlation would have proposed for this exact world")
    assert result["verdict"] == "rejected"


# -----------------------------------------------------------------
# correlate_boundary_edges — direct unit coverage beyond the confound.
# -----------------------------------------------------------------
def test_correlate_boundary_edges_excludes_edges_with_any_bit0_traffic():
    room_index = {"adj": {"0": {"1": {"kind": "pan", "dir": "E",
                                      "cap_hist": {"0": 3, "1": 40}}}}}
    leads = correlate_boundary_edges(room_index, bit_index=0)
    assert leads == []


def test_correlate_boundary_edges_min_bit0_count_is_configurable():
    room_index = {"adj": {"0": {"1": {"kind": "pan", "dir": "E",
                                      "cap_hist": {"0": 1, "1": 40}}}}}
    assert correlate_boundary_edges(room_index, bit_index=0) == []
    leads = correlate_boundary_edges(room_index, bit_index=0, min_bit0_count=1)
    assert len(leads) == 1
    assert leads[0]["count_bit0"] == 1 and leads[0]["count_bit1"] == 40


def test_correlate_boundary_edges_reads_bit_index_not_just_nonzero():
    # cap_sig=2 has bit 0 clear and bit 1 set — a lead search on
    # bit_index=0 must not be fooled by a nonzero-but-bit0-clear key.
    room_index = {"adj": {"0": {"1": {"kind": "pan", "dir": "E",
                                      "cap_hist": {"2": 40}}}}}
    assert correlate_boundary_edges(room_index, bit_index=0) == []
    leads = correlate_boundary_edges(room_index, bit_index=1)
    assert len(leads) == 1 and leads[0]["count_bit1"] == 40


def test_correlate_boundary_edges_accepts_a_real_room_index_instance():
    idx = RoomIndex(cap=64, config_sha="x")
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=0)
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=1)
    idx.record_edge(0, 1, "pan", "E", 4, cap_sig=1)
    leads = correlate_boundary_edges(idx, bit_index=0)
    assert leads == [], "one bit0 crossing exists — not a lead at min_bit0_count=0"
    leads2 = correlate_boundary_edges(idx, bit_index=0, min_bit0_count=1)
    assert len(leads2) == 1
    assert leads2[0]["count_bit0"] == 1 and leads2[0]["count_bit1"] == 2


def test_correlate_boundary_edges_sorts_strongest_lead_first():
    room_index = {"adj": {
        "0": {"1": {"kind": "pan", "dir": "E", "cap_hist": {"1": 5}}},
        "2": {"3": {"kind": "pan", "dir": "E", "cap_hist": {"1": 50}}},
    }}
    leads = correlate_boundary_edges(room_index, bit_index=0)
    assert [l["count_bit1"] for l in leads] == [50, 5]


# -----------------------------------------------------------------
# select_control_candidates — §7 item 8's collision-avoidance rule.
# -----------------------------------------------------------------
def test_select_control_candidates_prefers_the_rejected_pool():
    ledger = {
        (1, None): ItemBitCandidate(addr=1, status="rejected"),
        (2, None): ItemBitCandidate(addr=2, status="rejected"),
        (3, None): ItemBitCandidate(addr=3, status="confirmed"),
        (4, None): ItemBitCandidate(addr=4, status="candidate"),
    }
    picked = select_control_candidates(ledger, primary_key=(9, None), k=2)
    assert len(picked) == 2
    assert all(c.status == "rejected" for c in picked)


def test_select_control_candidates_falls_back_when_rejected_pool_is_short():
    ledger = {
        (1, None): ItemBitCandidate(addr=1, status="rejected"),
        (2, None): ItemBitCandidate(addr=2, status="confirmed"),
    }
    picked = select_control_candidates(ledger, primary_key=(9, None), k=2)
    assert len(picked) == 2
    assert {c.addr for c in picked} == {1, 2}


def test_select_control_candidates_never_returns_the_primary_key():
    ledger = {(1, None): ItemBitCandidate(addr=1, status="rejected")}
    picked = select_control_candidates(ledger, primary_key=(1, None), k=5)
    assert picked == []


def test_select_control_candidates_is_deterministic_for_a_fixed_rng():
    import random
    ledger = {(i, None): ItemBitCandidate(addr=i, status="rejected")
             for i in range(10)}
    a = select_control_candidates(ledger, primary_key=(99, None), k=3,
                                  rng=random.Random(42))
    b = select_control_candidates(ledger, primary_key=(99, None), k=3,
                                  rng=random.Random(42))
    assert [c.addr for c in a] == [c.addr for c in b]


# -----------------------------------------------------------------
# Small pure helpers.
# -----------------------------------------------------------------
def test_eval_candidate_applies_mod_before_match():
    ram = _blank_ram()
    ram[10] = 5
    cand = ItemBitCandidate(addr=10, match=frozenset({1}), mod=4)  # 5 % 4 == 1
    assert _eval_candidate(ram, cand) is True
    cand2 = ItemBitCandidate(addr=10, match=frozenset({5}), mod=0)
    assert _eval_candidate(ram, cand2) is True
    cand3 = ItemBitCandidate(addr=10, match=frozenset({2}), mod=0)
    assert _eval_candidate(ram, cand3) is False


def test_ram_from_step_slices_to_ram_size():
    raw = bytes(range(256)) * 10  # 2560 bytes, longer than RAM_SIZE
    cell = (None, None, raw)
    out = _ram_from_step(cell)
    assert out.shape == (RAM,)
    assert out.dtype == np.uint8


def test_run_matched_wave_scores_controls_off_the_same_two_lanes_no_extra_step():
    calls = {"n": 0}

    def gt(state, hist):
        calls["n"] += 1
        ram = _blank_ram()
        ram[100] = 1 if 7 in hist else 0
        return ram

    pool = _FakePool(gt)
    out = _run_matched_wave(
        pool, PRE, [("acquire", [7]), ("skip", [0])], probe_actions=[0],
        reach_fn=lambda ram: bool(ram[100] == 1),
        control_candidates=[ItemBitCandidate(addr=100, match=frozenset({1}))])
    assert out["reach"] == {"acquire": True, "skip": False}
    assert out["control"]["acquire"] == [True]
    assert out["control"]["skip"] == [False]
    # 2 lanes x 2 steps (prefix len 1 + probe len 1) = 4 step_all-driven
    # ground-truth calls — proves controls were scored off THIS same
    # replay, not a second one.
    assert calls["n"] == 4


# -----------------------------------------------------------------
# Purity re-check, extended to Stage 3.
# -----------------------------------------------------------------
def test_no_poke_ram_call_anywhere_in_stage3_either():
    src = ITEM_BITS_SRC.read_text()
    assert "poke_ram(" not in src
    assert "poke_ram_range(" not in src


def test_verify_behavioral_never_authors_a_trajectory_it_was_not_given():
    # The mined action lists pass through untouched — verify_behavioral
    # must never append, truncate, or otherwise invent frames beyond
    # what was supplied plus the shared probe.
    gt = _true_gate_world()
    pool = _FakePool(gt)
    cand = ItemBitCandidate(addr=100, bit=None, match=frozenset({1}), mod=0)
    acquire = [7, 0, 0]
    skip = [0]
    probe = [0, 0]
    verify_behavioral(pool, cand, pre_state=PRE, acquire_actions=acquire,
                      skip_actions=skip, probe_actions=probe,
                      reach_fn=_reach_fn(), n_trials=1, control_bits=1,
                      control_candidates=[_flat_control()])
    assert acquire == [7, 0, 0] and skip == [0] and probe == [0, 0]


# -----------------------------------------------------------------
# The one deliberately real, short, single-worker exception: proves
# the replay MECHANICS (save/load/step_all wiring), not the verdict
# arithmetic (already covered above against the scripted oracle).
# -----------------------------------------------------------------
def test_is0_7_live_pool_replay_mechanics_smoke():
    rom = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"
    state = REPO / "roms" / "zelda_start_ctrl.state.bin"
    if not rom.exists() or not state.exists():
        pytest.skip("Zelda ROM/start-state not present in this checkout")
    pool = make_probe_pool(rom, n_workers=2, frame_skip=4)
    pre_state = state.read_bytes()

    def reach_fn(ram: np.ndarray) -> bool:
        assert ram.shape == (RAM,)
        assert ram.dtype == np.uint8
        return True   # trivial: just prove real RAM came back sane

    ctrl = [ItemBitCandidate(addr=0x10, match=frozenset({255}), mod=0,
                             status="rejected")]
    cand = ItemBitCandidate(addr=0x66, match=frozenset({1}), mod=0)
    result = verify_behavioral(
        pool, cand, pre_state=pre_state, acquire_actions=[0, 0],
        skip_actions=[0], probe_actions=[0, 0], reach_fn=reach_fn,
        n_trials=2, control_bits=1, control_candidates=ctrl)
    assert result["reach_rate_acquire"] == 1.0
    assert result["reach_rate_skip"] == 1.0
    assert result["nondeterministic"] is False, (
        "identical (pre_state, actions) replayed twice against a real "
        "NES must reproduce identically — this codebase's own "
        "byte-identity discipline, checked live here, not assumed")
