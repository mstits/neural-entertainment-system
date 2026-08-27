"""Guard for the banked Rygar R1 tape (docs/receipts/rygar/r1_tape_gx6242.json).

WHY THIS FILE EXISTS, AND WHY HALF OF IT CANNOT SKIP
----------------------------------------------------
Earlier this week a control receipt lived only under the gitignored `runs/`
tree. On a fresh checkout the file was absent, its one test skipped silently,
and nothing guarded the claim. This file is written so that cannot happen
again.

The tape's *provenance and measured invariants* live in a tracked JSON, so
every assertion in `TestTapeRecord` runs on any checkout, with no ROM, no
emulator, and no savestate. Those tests fail loudly if the receipt is deleted,
truncated, renumbered, or quietly re-labelled from EXHIBITION to LEARNED.

Only `TestTapeReplays` needs the ROM, because ROMs are not distributable and
`roms/` is gitignored. It skips when the ROM is absent -- but it is NOT the
only guard, so an absent ROM can no longer reduce this receipt to zero
coverage.

WHAT THE TAPE IS
----------------
EXHIBITION -- Go-Explore search output. Not a learned policy, not an
honest-protocol result. Under CLAIMS.md it may never be described with "the AI
learned/plays/beat".

The R1 verdict recorded here is FAIL, and these tests pin that too: a receipt
that could be edited into a PASS without a test noticing would be worse than no
receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RECEIPT = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"
PROFILE = REPO / "configs/rygar.yaml"


@pytest.fixture(scope="module")
def rec() -> dict:
    assert RECEIPT.exists(), (
        f"{RECEIPT} is missing. This receipt is deliberately tracked rather "
        "than left under gitignored runs/; if it moved, update this test "
        "rather than deleting it.")
    return json.loads(RECEIPT.read_text())


class TestTapeRecord:
    """Runs on every checkout. No ROM, no emulator, no savestate."""

    def test_the_tape_is_filed_as_exhibition_not_learned(self, rec):
        # The single most important assertion in this file. Search output
        # relabelled as a learned result is the failure CLAIMS.md exists to
        # prevent.
        assert rec["ledger"] == "EXHIBITION"
        blurb = rec["what_this_is"].lower()
        assert "not a learned policy" in blurb
        # Scan the DESCRIPTIVE text only. The prohibition itself necessarily
        # quotes these phrases, so it lives in its own field -- otherwise this
        # check fires on the very sentence that forbids them.
        for forbidden in ("the ai learned", "the ai plays", "the ai beat"):
            assert forbidden not in blurb
        policy = rec["claims_policy"].lower()
        assert "exhibition" in policy
        for forbidden in ("the ai learned", "the ai plays", "the ai beat"):
            assert forbidden in policy, (
                "the receipt must name the phrasings it forbids, so a future "
                "reader cannot relabel this as a learned result by accident")

    def test_the_r1_verdict_is_fail_and_says_why(self, rec):
        assert rec["r1_verdict"] == "FAIL"
        cond = rec["r1_conditions"]
        # Two fail, two pass -- and all four are required, so the overall
        # verdict cannot be anything but FAIL.
        assert cond["1_DEPTH"]["verdict"] == "FAIL"
        assert cond["2_CLEAR_PREDICATE"]["verdict"] == "FAIL"
        assert cond["3_REPRODUCIBILITY"]["verdict"] == "PASS"
        assert cond["4_LIVENESS"]["verdict"] == "PASS"
        assert cond["1_DEPTH"]["raw_instrument"] < 9000

    def test_depth_carries_its_ratchet_caveat(self, rec):
        # The headline 6242 is farmable; 4608 is the honest number. If the
        # caveat is ever dropped, the number becomes a lie by omission.
        cond = rec["r1_conditions"]["1_DEPTH"]
        assert cond["artifact_free"] == 4608
        assert cond["artifact_free"] < cond["raw_instrument"]
        assert "ratchet" in rec["depth_caveat"].lower()

    def test_the_clear_predicate_is_recorded_as_absent(self, rec):
        # `solutions: 0` in a Rygar archive is a compile-time constant, so the
        # receipt must never present it as a search result.
        measured = rec["r1_conditions"]["2_CLEAR_PREDICATE"]["measured"]
        assert "none wired" in measured

    def test_actions_match_the_recorded_count_and_action_space(self, rec):
        acts = rec["actions"]
        inv = rec["invariants_for_the_guard"]
        assert len(acts) == inv["n_actions"] == 6018
        lo, hi = inv["action_index_range"]
        assert min(acts) == lo and max(acts) == hi
        # Every index must address the profile's own generic 14-action set.
        n_actions = len(rec["provenance"]["action_space"])
        assert n_actions == 14
        assert 0 <= lo and hi < n_actions

    def test_liveness_invariant_is_recorded_and_self_consistent(self, rec):
        inv = rec["invariants_for_the_guard"]
        hist = inv["dead_run_histogram"]
        # Real Rygar deaths pin lives at 0 for thousands of observations;
        # transition blips are exactly 2. The >=3 debounce sits between them.
        assert set(hist) == {"2"}
        assert inv["longest_dead_run"] == 2 < 3
        assert inv["terminal_alive"] is True
        # One 2-step lives blip per 20-step door blackout.
        assert int(hist["2"]) == int(inv["blackout_run_lengths"]["20"]) == 55

    def test_the_ratchet_arithmetic_is_internally_consistent(self, rec):
        inv = rec["invariants_for_the_guard"]
        # Perfect alternation: one zero-gain far-room visit per gaining
        # near-room visit. 27 door cycles.
        assert inv["post_door_segments_dx_zero"] == 27
        assert inv["post_door_segments_dx_positive"] == 27
        assert (inv["post_door_segments_dx_zero"]
                + inv["post_door_segments_dx_positive"]
                == inv["post_door_segments_total"] == 54)
        # The banked px is the gap between the honest and headline depths.
        gap = inv["terminal_odometer_x"] - inv["artifact_free_depth"]
        assert abs(inv["post_door_px_banked"] - gap) < 40, (
            "post-door banked px should account for the whole headline-minus-"
            "honest gap; a large mismatch means the ratchet accounting drifted")

    def test_the_loop_detector_had_a_working_positive_control(self, rec):
        # Without this, "screens repeat after the door" is not a finding --
        # the detector has to be shown capable of saying "different".
        inv = rec["invariants_for_the_guard"]
        assert (inv["positive_control_pre_door_distinct_screens"]
                == inv["positive_control_pre_door_pieces"] == 51), (
            "the pre-door control must be all-distinct, else the post-door "
            "repeat count proves nothing")
        # And it does say "same" after the door: 9 screens across 54 segments.
        assert inv["post_door_distinct_end_screens"] < inv[
            "post_door_segments_total"] / 4

    def test_scene_counter_is_recorded_as_blind(self, rec):
        # 0 cuts across a tape that provably crosses 55 blackout transitions:
        # the scene counter cannot fire on this profile, so no room claim may
        # ever rest on it.
        assert rec["invariants_for_the_guard"]["scene_cuts_total"] == 0

    def test_provenance_is_complete_enough_to_reproduce(self, rec):
        p = rec["provenance"]
        for key in ("rom_sha256", "start_state_sha256", "nes_core_sha256_16",
                    "frame_skip", "action_space"):
            assert p.get(key) not in (None, "", []), f"missing provenance: {key}"
        assert len(p["rom_sha256"]) == 64
        assert len(p["start_state_sha256"]) == 64
        assert p["frame_skip"] == 4
        assert rec["search_argv"], "argv is needed to re-run the search"

    def test_the_profile_still_has_no_clear_predicate(self):
        # If someone wires a clear:/finale:/level_key for Rygar, this receipt's
        # condition-2 FAIL is stale and must be re-adjudicated, not left to rot.
        import yaml
        solve = yaml.safe_load(PROFILE.read_text())["solve"]
        assert solve.get("level_key") == [], (
            "configs/rygar.yaml grew a level_key -- re-run the R1 clear-"
            "predicate condition and update docs/receipts/rygar/")
        assert "clear" not in solve and "finale" not in solve


class TestTapeReplays:
    """Full emulator replay. Needs the ROM, which is gitignored and not
    distributable -- so this half may skip. TestTapeRecord above may not."""

    def test_tape_replays_to_its_recorded_terminal_and_stays_alive(self, rec):
        import numpy as np
        import yaml

        prof = yaml.safe_load(PROFILE.read_text())
        rom = REPO / prof["solve"]["rom"]
        start = REPO / prof["start_state_path"]
        if not rom.exists() or not start.exists():
            pytest.skip(f"ROM or start state absent ({rom.name}); the "
                        "TestTapeRecord assertions above still ran")

        import nes_core
        from src.training.profile_utils import action_space_to_bitmasks

        inv = rec["invariants_for_the_guard"]
        bm = action_space_to_bitmasks(prof["action_space"])
        pool = nes_core.Pool(rom_path=str(rom), num_workers=1,
                             frame_skip=prof["frame_skip"])
        try:
            pool.set_headless(True)
            pool.set_skip_preprocess(True)
            pool.set_odometer_enabled(True)
            pool.reset_all()
            pool.load_worker_state(0, start.read_bytes())
            ram0 = pool.step_all(np.zeros(1, dtype=np.uint8))[0][2]
            lives0 = int(ram0[0x0303])
            assert lives0 == inv["lives_at_start"]

            xs, dead = [], []
            for a in rec["actions"]:
                ram = pool.step_all(
                    np.array([bm[int(a)]], dtype=np.uint8))[0][2]
                xs.append(int(pool.get_odometer_per_worker()[0][0]))
                dead.append(1 <= (lives0 - int(ram[0x0303])) % 256 <= 8)
            scene = int(pool.get_odometer_scene_per_worker()[0])
        finally:
            pool.shutdown()

        assert xs[-1] == inv["terminal_odometer_x"], (
            f"tape replayed to {xs[-1]}, receipt records "
            f"{inv['terminal_odometer_x']}")
        assert not dead[-1], "terminal frame must be alive"
        assert scene == inv["scene_cuts_total"]

        # Liveness: longest consecutive dead run, against the debounce's own
        # >=3 threshold.
        longest = cur = 0
        for d in dead:
            cur = cur + 1 if d else 0
            longest = max(longest, cur)
        assert longest == inv["longest_dead_run"] < 3

        # No odometer may be banked while the death predicate is true -- the
        # Contra post-death-scroll failure mode.
        gain_while_dead = sum(
            max(0, xs[i] - xs[i - 1]) for i in range(1, len(xs)) if dead[i])
        assert gain_while_dead < 0.01 * xs[-1], (
            f"{gain_while_dead} px banked while dead -- post-death scroll "
            "contamination")

        # The honest depth is reached where the receipt says it is.
        first = next(i for i, x in enumerate(xs)
                     if x >= inv["artifact_free_depth"])
        assert first == inv["first_step_reaching_artifact_free_depth"]
