"""The eleven review findings against the first K0 repair pass.

The gate-opener campaign disarmed itself on its own registration
(GATE_OPENER_CAMPAIGN_2026-08-11 §13). The repair pass that followed was
reviewed against its own receipts and eleven defects came back — four
blockers among them, two of which would have made a re-forged instrument
miss the same blind keys for the same reasons. Every test here is a
MUTATION CHECK on one of them: it fails if the repair is reverted.

  R1  rank_sort_key led with effect size and never read `significant`,
      so on a 4-family / 8-root probe the ONLY BH-significant row — a
      one-family mode bit moving by 8, the shape of blind key $0004 —
      sorted LAST, behind three tied nuisances that moved further.
  R2  the prior holds only pre-press frames, so a latched candidate's
      post-press value has seen == 0 BY CONSTRUCTION: 40 latched rows,
      one novelty value, one score. The plateau was manufactured
      upstream of the tie-break the campaign blamed.
  R3  the effect size was Euclidean on a wrapping byte: a counter
      ticking 250 -> 254 -> 3 read 247/255 = 0.97 AND escaped the
      monotone COUNTER test, taking rank 1 as "latched".
  R4  addr_families sorted DESC in the rank key while the leave-family-
      out background rate — hence the p-value — RISES with it: one
      quantity, opposite signs, one pipeline.
  R5  FARM_MIN_STEPS was defeated by pseudo-replication: the settle
      prefixes of all 120x2 programs at a root are byte-identical
      replays of ONE all-NOOP trajectory, so ~78k "observed steps" came
      from ~2k distinct frames.
  R6  the K1 sham null reported 0.0 (= clean) when no sham root existed.
  R7  resume_lineage_diff on the REAL lvl_03_overnight stats returns
      mismatch=[] — that file records neither hw_provenance nor
      key_config — so the exact §12 case warned and resumed.
  R8  no test read the real archive; the proof was unread on disk
      (entrance_after_2.state.json records hw_flags=[mmio_read_timing]).
  R9  the key-schema half was UNVERIFIABLE for every archive on disk,
      though tb/kk/sig are recoverable from the keys seed() already
      walks.
  R10 banked gx-767 phantoms survived resume: _sel_topgx and
      max_gx_in_area both read ortho_ctrl's 14 bucket-95 cells, whose
      blobs restore at gx 507-511. §12's bucket-93 refreeze existed
      nowhere in code.
  R11 progress smooth=borrow assumed hi is lo's page byte, guarded only
      by hi existing — 6 of 11 two-byte profiles are (screen, x)
      composites where a lo wrap with hi flat is ordinary motion.

The campaign stays DISARMED. A re-forged instrument faces a NEW K0
registration under the same blind law.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import src.training.interaction_basis as ib
from scripts.go_explore_solve import Solver
from tests.test_gate_instrument_repair import _obs, _solver, _sweepable

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# R1 / R4 — what the total order reads
# ---------------------------------------------------------------------

def _legacy_sort_key(row):
    """The key as it stood before this repair: score, effect size,
    family diversity, coin. `significant` does not appear."""
    return (-float(row.get("score", 0.0)),
            -float(row.get("effect_size", 0.0)),
            -int(row.get("addr_families", 0)),
            int(row.get("tie_key", 0)))


def _four_family_probe():
    """The receipted probe: 4 families x 8 roots. One address (0x0004)
    moves in ONE family by 8 — no other family touches it, so its
    leave-family-out background rate is floor-level and BH keeps it.
    Three nuisances move in ALL four families by 40, so their background
    rate saturates and BH refuses them. Every row is latched, seen from
    every root of its family, so all four tie at score 1.0."""
    fams = ("hold", "tap", "combo", "contact")
    roots = [f"r{i}" for i in range(8)]
    rows = []
    for fam in fams:
        for root in roots:
            rows.append(_obs(root, 0, "control", {}))
            moves = {a: (40, 40) for a in (0x0300, 0x0301, 0x0302)}
            if fam == "hold":
                moves[0x0004] = (8, 8)
            rows.append(_obs(root, 1, fam, moves))
    return rows


def test_the_only_significant_row_is_not_ranked_last_behind_the_nuisances():
    ranked = ib.rank_candidates(_four_family_probe(), tie_seed=0)
    sig = [r for r in ranked if r["significant"]]
    assert [r["addr"] for r in sig] == [0x0004], "fixture is not the probe"
    scores = {round(r["score"], 12) for r in ranked}
    assert len(scores) == 1, "fixture is not a score plateau"
    # THE DEFECT, reproduced against the key it was found in.
    legacy = sorted(ranked, key=_legacy_sort_key)
    assert legacy[-1]["addr"] == 0x0004
    # THE REPAIR: evidence leads the plateau.
    assert ranked[0]["addr"] == 0x0004
    assert ib.admissible_rows(ranked)[0]["addr"] == 0x0004
    assert [r["addr"] for r in ib.admitted_candidates(ranked)] == [0x0004]


def test_a_refused_row_cannot_outrank_an_admissible_one_at_the_same_score():
    # The sibling of R1: K4's refusal is the other verdict that decides
    # whether a row may be admitted at all, and it was not in the key.
    rows = [{"score": 1.0, "effect_size": 0.9, "significant": True,
             "refused": "farmable", "tie_key": 1, "addr": 1},
            {"score": 1.0, "effect_size": 0.1, "significant": True,
             "refused": None, "tie_key": 2, "addr": 2}]
    assert [r["addr"] for r in sorted(rows, key=ib.rank_sort_key)] == [2, 1]


def test_family_diversity_is_not_counted_twice_in_one_pipeline():
    # R4. addr_families raises the leave-family-out background rate,
    # hence the p-value, hence the evidence AGAINST the row. It must not
    # simultaneously raise its rank.
    a = {"score": 1.0, "effect_size": 0.5, "addr_families": 4, "tie_key": 9,
         "addr": 1, "family": "hold", "significant": True}
    b = dict(a)
    b["addr_families"] = 1
    assert ib.rank_sort_key(a) == ib.rank_sort_key(b)
    # ...and it is still receipted on the row, where a reader can see it.
    ranked = ib.rank_candidates(_four_family_probe(), tie_seed=0)
    assert {r["addr_families"] for r in ranked} == {1, 4}


def test_more_families_means_a_worse_p_value_which_is_the_only_sign_it_has():
    ranked = ib.rank_candidates(_four_family_probe(), tie_seed=0)
    by_addr = {r["addr"]: r for r in ranked if r["family"] == "hold"}
    assert by_addr[0x0004]["addr_families"] == 1
    assert by_addr[0x0300]["addr_families"] == 4
    assert by_addr[0x0004]["p"] < by_addr[0x0300]["p"]


# ---------------------------------------------------------------------
# R3 — a byte is a ring
# ---------------------------------------------------------------------

@pytest.mark.parametrize("a,b,want", [
    (250, 3, 9), (3, 250, -9), (0, 128, 128), (128, 0, 128),
    (0, 129, -127), (10, 20, 10), (255, 0, 1), (7, 7, 0),
])
def test_the_ring_delta_is_the_shortest_signed_path(a, b, want):
    assert ib.ring_delta(a, b) == want
    assert ib.ring_distance(a, b) == abs(want)


def test_a_wrapped_counter_tick_is_a_counter_and_not_a_latch():
    # THE DEFECT, in one line: 250 -> 254 -> 3 is +4 then +5.
    assert ib.classify_persistence(250, 254, 3) == "counter"
    assert ib.classify_persistence(3, 254, 250) == "counter"
    # ...and the unwrapped classes are exactly as they were.
    assert ib.classify_persistence(0, 10, 20) == "counter"
    assert ib.classify_persistence(0, 8, 8) == "latched"
    assert ib.classify_persistence(0, 8, 0) == "timed"
    # A LARGE step across the ring stays LATCHED: it is equally
    # consistent with a latch plus a partial revert, and the bound is
    # what keeps the wrap extension from re-classing real gates.
    assert ib.classify_persistence(0, 200, 150) == "latched"


def test_a_wrapped_counter_no_longer_wins_the_effect_size_tie_break():
    base = np.zeros(ib.RAM_SIZE, dtype=np.uint8)
    base[0x20] = 250
    rows = []
    for root in ("r0", "r1", "r2", "r3"):
        rows.append(_obs(root, 0, "control", {}, base=base))
        rows.append(_obs(root, 1, "hold", {0x20: (254, 3), 0x30: (8, 8)},
                         base=base))
    ranked = {(r["addr"]): r for r in ib.rank_candidates(rows, tie_seed=0)}
    counter, gate = ranked[0x20], ranked[0x30]
    assert counter["class"] == "counter" and gate["class"] == "latched"
    # 9/128, not 247/255.
    assert counter["effect_size"] == pytest.approx(9 / 128.0)
    assert gate["effect_size"] == pytest.approx(8 / 128.0)
    # The counter's persistence weight is 0, so it cannot rank at all.
    assert counter["score"] == 0.0 and gate["score"] > 0.0
    assert ib.rank_candidates(rows, tie_seed=0)[0]["addr"] == 0x30


def test_the_effect_size_is_normalised_by_the_largest_move_a_byte_has():
    rows = []
    for root in ("r0", "r1", "r2", "r3"):
        rows.append(_obs(root, 0, "control", {}))
        rows.append(_obs(root, 1, "hold", {0x40: (128, 128)}))
    r = ib.rank_candidates(rows, tie_seed=0)[0]
    assert r["effect_size"] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# R2 — the prior is a distribution, not a lookup
# ---------------------------------------------------------------------

def test_an_unseen_value_is_not_one_constant_for_every_address():
    # THE DEFECT: with only (seen, total) an unseen value scores
    # log2(total+1) whatever address it is at — one number for every
    # latched row on the run.
    assert ib.novelty_score(0, 5000) == ib.novelty_score(0, 5000)
    pinned = ib.novelty_score(0, 5000, support=1)
    busy = ib.novelty_score(0, 5000, support=200)
    assert pinned > busy > 0.0
    # A value the byte shows constantly is still ~0 bits of surprise.
    assert ib.novelty_score(5000, 5000, support=1) < 0.01
    # No support recorded at all falls back rather than inventing one.
    assert ib.novelty_score(0, 5000, support=0) == ib.novelty_score(0, 5000)


def test_the_prior_separates_a_pinned_byte_from_a_free_runner(tmp_path):
    f = _solver(tmp_path)
    f._gate_hist_alloc()
    vals = np.zeros(ib.RAM_SIZE, dtype=np.uint8)
    for t in range(300):
        vals[0x10] = 7                     # pinned across the whole prior
        vals[0x20] = t % 200               # free-running
        f._gate_baseline_sample(vals.copy(), None)
    # Neither address has ever shown 0x40 — the post-press value of a
    # latched candidate is unseen at BOTH by construction.
    pinned = f._gate_novelty(0x10, 0x40)
    busy = f._gate_novelty(0x20, 0x40)
    assert pinned > busy > 0.0, "the latched plateau is not relieved"


def test_forty_latched_rows_do_not_collapse_onto_one_score(tmp_path):
    f = _solver(tmp_path)
    f._gate_hist_alloc()
    vals = np.zeros(ib.RAM_SIZE, dtype=np.uint8)
    for t in range(400):
        for i in range(40):
            # each candidate address has its own undirected volatility
            vals[0x100 + i] = (t % (i + 1)) if i else 0
        f._gate_baseline_sample(vals.copy(), None)
    rows = []
    for root in ("r0", "r1", "r2", "r3"):
        rows.append(_obs(root, 0, "control", {}))
        rows.append(_obs(root, 1, "hold",
                         {0x100 + i: (0xC0, 0xC0) for i in range(40)}))
    ranked = ib.rank_candidates(rows, novelty=f._gate_novelty, tie_seed=0)
    assert len(ranked) == 40
    assert len({round(r["novelty"], 9) for r in ranked}) > 1
    assert len({round(r["score"], 9) for r in ranked}) > 1


# ---------------------------------------------------------------------
# R5 — pseudo-replication under the K4 refusal floor
# ---------------------------------------------------------------------

CTX = [{"key": ("a",)}, {"key": ("b",)}]


def test_the_settle_prefixes_are_one_trajectory_not_one_per_program(tmp_path):
    # THE DEFECT: every program at a root replays the SAME all-NOOP
    # frames from the SAME blob. The two-action basis is 3 all-NOOP
    # controls (undirected for all 153 steps) + 3 patterns (undirected
    # for their 8-step settle) = 483 frames stepped, of which 153 are
    # distinct. At the real basis size that ratio is ~97%, which is how
    # ~2k frames were reported as ~78k observed steps.
    f, cells = _sweepable(tmp_path)
    f._gate_sweep(CTX, [(cells[6], False)])
    stepped = 3 * ib.PROGRAM_LEN_PASS_A + 3 * 8
    assert f._gate_baseline_frames == ib.PROGRAM_LEN_PASS_A
    assert f._gate_change_sweep_n == ib.PROGRAM_LEN_PASS_A - 1
    assert f._gate_counters["baseline_dupes"] == stepped - \
        ib.PROGRAM_LEN_PASS_A
    assert f._boundary_hist_total == ib.PROGRAM_LEN_PASS_A


def test_re_sweeping_a_root_does_not_re_bank_its_baseline(tmp_path):
    # Roots are re-swept by design (--gate-sweep-roots picks the LEAST
    # swept), so the replication compounds across sweeps too.
    f, cells = _sweepable(tmp_path)
    f._gate_sweep(CTX, [(cells[6], False)])
    once = f._gate_baseline_frames
    f._gate_disarmed = False
    f._gate_sweep(CTX, [(cells[6], False)])
    assert f._gate_baseline_frames == once


def test_two_distinct_roots_are_two_trajectories(tmp_path):
    # The de-replicator must not eat real evidence: a different root is
    # a different restore point and its frames are counted in full.
    f, cells = _sweepable(tmp_path)
    a = _sweepable(tmp_path)[0]
    a._gate_sweep(CTX, [(cells[6], False)])
    one = a._gate_baseline_frames
    for i, c in enumerate(cells[6:8]):
        c.state = bytes([i]) + bytes(c.state)[1:]     # distinct blobs
    f._gate_sweep(CTX, [(c, False) for c in cells[6:8]])
    assert f._gate_baseline_frames == 2 * one


def test_the_pooled_farm_rate_is_charged_distinct_steps(tmp_path):
    # The floor exists so a 0.0 reading means "measured and clean". Under
    # replication the denominator was fiction, so a slow mover read
    # 0.0/1k as MEASURED off a horizon that could not resolve it.
    f, cells = _sweepable(tmp_path)
    f._gate_sweep(CTX, [(cells[6], False)])
    src = dict((n, (cnt, stride))
               for n, _c, cnt, stride in f._gate_farm_sources())
    assert src["sweep_undirected"] == (ib.PROGRAM_LEN_PASS_A - 1, 1)
    # 152 distinct steps is BELOW the refusal floor, so the axis is
    # refused rather than reported clean off replayed frames.
    assert ib.PROGRAM_LEN_PASS_A - 1 < ib.FARM_MIN_STEPS
    assert f._gate_farmability(0x40) is None


def test_the_replication_the_run_refused_is_on_the_receipt(tmp_path):
    f, cells = _sweepable(tmp_path)
    f._gate_sweep(CTX, [(cells[6], False)])
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["boundary_frames_sweep"] == ib.PROGRAM_LEN_PASS_A
    assert rec["boundary_frames_replayed"] > rec["boundary_frames_sweep"]
    assert rec["farm_steps_total"] == ib.PROGRAM_LEN_PASS_A - 1


# ---------------------------------------------------------------------
# R6 — K1's sham null
# ---------------------------------------------------------------------

def test_a_null_that_never_ran_is_null_and_not_a_clean_zero(tmp_path):
    # THE DEFECT: sham_yield 0.0 with no sham root reads to K1 ("sham
    # yield >= 50% of wall yield -> the ranking is noise, kill") as a
    # clean null. The sham arm draws from BELOW the band floor and that
    # region can be empty by construction — Bubble Bobble r99_1 has
    # spatial_span 1, so there is no freely-advancing region at all.
    f, cells = _sweepable(tmp_path)
    assert f._gate_counters["sham_yield"] is None      # and starts there
    f._gate_sweep(CTX, [(c, False) for c in cells[4:]])
    assert f._gate_counters["sham_roots"] == 0
    assert f._gate_counters["sham_yield"] is None, \
        "a null nobody ran is being reported as a clean null"
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["sham_roots"] == 0 and rec["sham_yield"] is None


def test_a_null_that_did_run_reports_the_ratio_and_both_counts(tmp_path):
    f, cells = _sweepable(tmp_path)
    roots = [(c, False) for c in cells[4:]] + [(cells[0], True)]
    f._gate_sweep(CTX, roots)
    c = f._gate_counters
    assert c["sham_roots"] == 1
    assert isinstance(c["sham_yield"], float)
    assert c["wall_hits"] >= 1
    # The kill reads a quotient; the receipt has to carry its inputs.
    assert c["sham_yield"] == c["sham_hits"] / max(1, c["wall_hits"])
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["sham_hits"] == c["sham_hits"]
    assert rec["wall_hits"] == c["wall_hits"]


def test_the_progress_line_carries_the_unmeasured_null_as_null(tmp_path):
    # The kill reads progress.jsonl. `gate_sham_yield: 0.0` there was the
    # form the campaign banked, with no companion count to catch it.
    f, cells = _sweepable(tmp_path)
    f._gate_sweep(CTX, [(c, False) for c in cells[4:]])
    line = {}
    for k in ("sham_yield", "sham_roots", "sham_hits", "wall_hits"):
        line[f"gate_{k}"] = f._gate_counters[k]
    assert json.loads(json.dumps(line))["gate_sham_yield"] is None
    assert line["gate_sham_roots"] == 0


# ---------------------------------------------------------------------
# R7 / R8 — the §12 archive, read off the disk it is actually on
# ---------------------------------------------------------------------

#: The four flags the T1 arms ran under (campaign §12 / the root manifest
#: runs/cv_t1_roots_2026-08-11.json).
ARM_FLAGS = ["reset_alignment", "mmio_read_timing", "dmc_stall_timing",
             "nmi_poll_timing"]
OVERNIGHT = REPO / "runs/cv_chain_hw/lvl_03_overnight"
ORTHO_CTRL = REPO / "runs/cv_hall_ortho_ctrl"

real_archive = pytest.mark.skipif(
    not (OVERNIGHT / "archive.stats.json").exists(),
    reason="the §12 archive is not on this machine")


def _cv_sig_game():
    return SimpleNamespace(state_sig_arity=1, progress_smooth="off",
                           state_sig_sha="deadbeef")


def _resume_solver(*, allow=False, allow_unverified=False,
                   flags=ARM_FLAGS, **argover):
    from scripts.go_explore_solve import hw_provenance
    args = SimpleNamespace(time_bins=False, kill_key=False, gx_bucket=8,
                           y_band=16, allow_lineage_mismatch=allow,
                           allow_unverified_lineage=allow_unverified)
    for k, v in argover.items():
        setattr(args, k, v)
    f = SimpleNamespace(args=args, game=_cv_sig_game(),
                        provenance=hw_provenance(flags, 4))
    for name in ("key_config", "check_resume_lineage", "audit_resumed_keys"):
        setattr(f, name, MethodType(getattr(Solver, name), f))
    return f


@real_archive
def test_the_real_overnight_stats_records_neither_field():
    # The premise, verified against the file rather than asserted: this
    # is why the check that was written FOR this archive returned
    # mismatch=[] on it.
    stats = json.loads((OVERNIGHT / "archive.stats.json").read_text())
    assert "hw_provenance" not in stats and "key_config" not in stats
    assert stats["cells"] == 560410 and stats["best_score"] == 767


@real_archive
def test_the_real_overnight_lineage_is_recovered_and_refused():
    # THE §12 CASE, END TO END, ON THE REAL DIRECTORY. The proof was one
    # file away the whole time: roots.json names the entrance blob and
    # entrance_after_2.state.json records hw_flags=["mmio_read_timing"]
    # — one flag against the arms' four.
    from scripts.go_explore_solve import recover_lineage
    stats = json.loads((OVERNIGHT / "archive.stats.json").read_text())
    prov, notes = recover_lineage(OVERNIGHT, stats)
    assert prov["hw_flags"] == ["mmio_read_timing"]
    assert notes and "entrance_after_2.state.json" in notes[0]
    with pytest.raises(SystemExit) as e:
        _resume_solver().check_resume_lineage(OVERNIGHT)
    assert "--allow-lineage-mismatch" in str(e.value)


@real_archive
def test_the_real_overnight_refusal_names_the_flags_on_both_sides(capsys):
    diff = _resume_solver(allow=True).check_resume_lineage(OVERNIGHT)
    assert any("hw_flags" in m for m in diff["mismatch"])
    err = capsys.readouterr().err
    assert "mmio_read_timing" in err and "reset_alignment" in err
    assert "RECOVERED" in err                     # and it says where from


@real_archive
def test_the_matching_lineage_archive_is_not_refused_for_its_machine():
    # The control: runs/cv_hall_ortho_ctrl DOES record hw_provenance, and
    # it is the four-flag one — so the machine half passes and only the
    # key half (which no archive on disk recorded) is left unchecked.
    from scripts.go_explore_solve import (recover_lineage,
                                          resume_lineage_diff, hw_provenance)
    if not (ORTHO_CTRL / "archive.stats.json").exists():
        pytest.skip("ortho_ctrl is not on this machine")
    stats = json.loads((ORTHO_CTRL / "archive.stats.json").read_text())
    prov, notes = recover_lineage(ORTHO_CTRL, stats)
    assert prov["hw_flags"] == ARM_FLAGS and notes == []
    # This is the machine-half control: the subject is the four flags and the
    # key half being left unchecked, not the emulator binary. The archive was
    # banked on whatever nes_core build was live then, so hold the run on the
    # SAME binary (its own recorded one) — otherwise a since-rebuilt core reads
    # as a real nes_core mismatch and masks what this test is about.
    run_prov = hw_provenance(ARM_FLAGS, 4)
    run_prov["nes_core"] = dict(prov["nes_core"])
    diff = resume_lineage_diff(stats, run_prov, _resume_solver().key_config())
    assert diff["mismatch"] == []
    assert any("key_config" in u for u in diff["unverifiable"])


def test_recovery_prefers_roots_json_then_the_blob_sidecar(tmp_path):
    from scripts.go_explore_solve import recover_lineage, write_state_sidecar
    blob = tmp_path / "root.state"
    blob.write_bytes(b"x")
    write_state_sidecar(blob, {"hw_flags": ["mmio_read_timing"],
                               "frame_skip": 4, "nes_core": {}})
    (tmp_path / "roots.json").write_text(json.dumps(
        {"entrance": {"path": str(blob)}}))
    prov, notes = recover_lineage(tmp_path, {})
    assert prov["hw_flags"] == ["mmio_read_timing"] and len(notes) == 1
    # roots.json's own record wins when it has one.
    (tmp_path / "roots.json").write_text(json.dumps(
        {"entrance": {"path": str(blob),
                      "hw_provenance": {"hw_flags": ARM_FLAGS,
                                        "frame_skip": 4, "nes_core": {}}}}))
    prov, notes = recover_lineage(tmp_path, {})
    assert prov["hw_flags"] == ARM_FLAGS and "roots.json" in notes[0]
    # ...and a recorded stats sidecar beats both, with nothing to note.
    assert recover_lineage(tmp_path, {"hw_provenance": {"hw_flags": []}}) == (
        {"hw_flags": []}, [])


def test_nothing_recoverable_is_a_refusal_and_not_a_pass(tmp_path):
    (tmp_path / "archive.stats.json").write_text(json.dumps({"cells": 1}))
    with pytest.raises(SystemExit) as e:
        _resume_solver().check_resume_lineage(tmp_path)
    assert "could not be checked" in str(e.value)
    diff = _resume_solver(allow_unverified=True).check_resume_lineage(tmp_path)
    assert len(diff["unverifiable"]) == 2 and diff["mismatch"] == []


# ---------------------------------------------------------------------
# R9 — the key schema, recovered from the keys themselves
# ---------------------------------------------------------------------

def _key(*, sect=0, tb=0, kk=0, area=0, sig=0, yb=1, gx=10):
    return (sect, tb, kk, (), 0, (), area, 3, sig, yb, gx)


def test_the_keys_answer_what_no_archive_on_disk_recorded():
    from scripts.go_explore_solve import key_schema_from_keys
    got = key_schema_from_keys([_key(tb=4, kk=7, sig=3), _key(tb=2)])
    assert got == {"n": 2, "arity": 11, "tb_max": 4, "kk_max": 7,
                   "sig_bits": 2}
    assert key_schema_from_keys([])["arity"] is None
    assert key_schema_from_keys([_key(), (1, 2, 3)])["arity"] == -1


def test_a_disjoint_tb_kk_subspace_is_caught_after_the_load():
    # §12's own words for lvl_03_overnight: "a disjoint tb/kk key
    # subspace". tb is a bit-length slot that is >= 1 whenever --time-
    # bins is on, so an archive carrying one and a run building none can
    # never be the same partition.
    from scripts.go_explore_solve import (key_schema_conflicts,
                                          key_schema_from_keys)
    run = {"tb": 0, "kk": 1, "sig_arity": 1}
    banked = key_schema_from_keys([_key(tb=3, kk=5)])
    bad = key_schema_conflicts(banked, run)
    assert any("time-bin" in m for m in bad["mismatch"])
    assert any("kill-count" in m for m in bad["mismatch"])
    clean = key_schema_conflicts(key_schema_from_keys([_key(sig=1)]), run)
    assert clean == {"mismatch": [], "unverifiable": []}


def test_an_ambiguous_axis_is_unverifiable_and_a_wider_sig_is_a_mismatch():
    from scripts.go_explore_solve import (key_schema_conflicts,
                                          key_schema_from_keys)
    obs = key_schema_from_keys([_key(sig=7)])
    assert any("state_sig" in m for m in
               key_schema_conflicts(obs, {"tb": 0, "kk": 1,
                                          "sig_arity": 2})["mismatch"])
    # kill_key ON with nothing killed looks exactly like kill_key OFF.
    amb = key_schema_conflicts(key_schema_from_keys([_key()]),
                               {"tb": 0, "kk": 16, "sig_arity": 0})
    assert amb["mismatch"] == [] and len(amb["unverifiable"]) == 1


def test_the_arity_contract_the_frontier_readers_depend_on_is_checked():
    from scripts.go_explore_solve import (key_schema_conflicts,
                                          key_schema_from_keys)
    short = key_schema_from_keys([(0, 0, 0, (), 0, (), 0, 1, 10)])
    bad = key_schema_conflicts(short, {"tb": 0, "kk": 1, "sig_arity": 0})
    assert any("arity" in m for m in bad["mismatch"])


def test_the_post_load_audit_refuses_and_can_be_overridden():
    f = _resume_solver()
    with pytest.raises(SystemExit) as e:
        f.audit_resumed_keys("dir", [_key(tb=3)])
    assert "--allow-lineage-mismatch" in str(e.value)
    out = _resume_solver(allow=True).audit_resumed_keys("dir", [_key(tb=3)])
    assert out["mismatch"]
    # An empty archive says nothing and refuses nothing.
    assert _resume_solver().audit_resumed_keys("dir", []) == {
        "mismatch": [], "unverifiable": []}


# ---------------------------------------------------------------------
# R10 — the banked phantom survives resume
# ---------------------------------------------------------------------

def _hall_keys():
    """The §12 shape: a 2,805-cell body topping out at bucket 93, bucket
    94 EMPTY in every archive, and a 14-cell island at 95 whose blobs
    restore at gx 507-511. Keys are distinct, because an archive is a
    dict and a fixture whose keys collide is not the archive."""
    keys = [_key(gx=90 + (i % 4), yb=i // 4) for i in range(2805)]
    keys += [_key(gx=95, yb=i) for i in range(14)]
    return keys


def test_the_banked_island_is_named_by_its_shape_and_not_its_address():
    from scripts.go_explore_solve import phantom_top_buckets
    assert phantom_top_buckets(_hall_keys()) == {(0, 95)}
    # A frontier that was actually REACHED arrives with a population and
    # without a gap, so nothing is refrozen.
    honest = [_key(gx=b, yb=i) for b in range(90, 96) for i in range(20)]
    assert phantom_top_buckets(honest) == set()
    # ...and an island in a different area is refused there and only
    # there, since the frontier is per-area.
    two = _hall_keys() + [_key(area=1, gx=40, yb=i) for i in range(50)]
    assert phantom_top_buckets(two) == {(0, 95)}


def test_the_frontier_readers_stop_crowning_the_phantom(tmp_path):
    # THE DEFECT: max_gx_in_area and _sel_topgx are both maxima, so 14
    # cells out of 149,153 defined every resumed run's frontier.
    from scripts.go_explore_solve import GX_BUCKET, phantom_top_buckets
    cells = [SimpleNamespace(key=k, state=b"s", best_score=1.0,
                             best_steps=1, times_chosen=0, explored=False,
                             barren=0) for k in _hall_keys()]
    f = SimpleNamespace(
        archive=SimpleNamespace(cells={c.key: c for c in cells}),
        max_area=0, max_sect=0, ortho_mode="off", _gx_phantoms=set(),
        _sel_cells=None, _sel_n=0, _sel_area=None)
    f._refresh_sel_cache = MethodType(Solver._refresh_sel_cache, f)
    f._refresh_sel_cache()
    assert f._sel_topgx == 95                      # the defect, reproduced
    f._gx_phantoms = phantom_top_buckets(f.archive.cells)
    f._refresh_sel_cache()
    assert f._sel_topgx == 93                      # §12's true frontier
    # The cells are NOT deleted — only the frontier readers ignore them.
    assert len(f.archive.cells) == 2819
    assert sum(1 for c in f._sel_deep if c.key[-1] == 95) == 0
    assert GX_BUCKET


def test_the_refreeze_is_inert_without_a_resume(tmp_path):
    # Default runs never build the set, so nothing on the hot path
    # changes for a search that started from a root.
    from scripts.go_explore_solve import phantom_top_buckets
    assert phantom_top_buckets([]) == set()
    assert phantom_top_buckets([_key(gx=1)]) == set()


def test_the_post_load_audit_does_not_re_refuse_what_the_stats_check_owns():
    # Double jeopardy: an unrecorded schema is already a refusal in
    # check_resume_lineage, and a sig bit the archive never set is not
    # evidence of anything. The key audit fires on CONTRADICTIONS only.
    f = _resume_solver()
    out = f.audit_resumed_keys("dir", [_key(sig=0)])
    assert out["mismatch"] == [] and len(out["unverifiable"]) == 1


# ---------------------------------------------------------------------
# R11 — the borrow rule assumed a pair it never checked
# ---------------------------------------------------------------------

def test_the_frame_after_a_rejection_says_what_the_rejection_was():
    from scripts.go_explore_solve import borrow_followup
    # The §12 tear: $0040 wraps one frame before $0041 borrows, so the
    # carry lands on the NEXT frame and hi moves 2 -> 1.
    assert borrow_followup((767, 2, 255), (511, 1, 255)) == "tear"
    # A (screen, x) composite: x wrapped, the screen byte never moves.
    assert borrow_followup((770, 3, 2), (774, 3, 6)) == "not_a_pair"
    # A single-byte or HUD observable has nothing to rule on.
    assert borrow_followup((5, None, None), (6, None, None)) == "unknown"
    assert borrow_followup(None, (6, 1, 2)) == "unknown"


def _borrow_solver():
    f = SimpleNamespace(progress_smooth="borrow", _borrow_tears=0,
                        _borrow_flat=0, _borrow_off=False)
    f._borrow_audit = MethodType(Solver._borrow_audit, f)
    return f


def test_a_pair_that_is_not_a_pair_disarms_the_rule_and_says_so(capsys):
    from scripts.go_explore_solve import BORROW_REFUTE_LIMIT
    f = _borrow_solver()
    for _ in range(BORROW_REFUTE_LIMIT - 1):
        f._borrow_audit("not_a_pair")
    assert f.progress_smooth == "borrow"           # not yet
    f._borrow_audit("not_a_pair")
    assert f.progress_smooth == "off" and f._borrow_off is True
    out = capsys.readouterr().out
    assert "borrow DISARMED" in out and "median3" in out


def test_a_filter_that_is_working_is_never_disarmed_by_one_coincidence():
    from scripts.go_explore_solve import BORROW_REFUTE_LIMIT
    f = _borrow_solver()
    for _ in range(20):
        f._borrow_audit("tear")
    for _ in range(BORROW_REFUTE_LIMIT + 1):
        f._borrow_audit("not_a_pair")
    assert f.progress_smooth == "borrow" and f._borrow_off is False
    assert (f._borrow_tears, f._borrow_flat) == (20, BORROW_REFUTE_LIMIT + 1)


def test_an_unknown_verdict_counts_as_nothing():
    f = _borrow_solver()
    for _ in range(50):
        f._borrow_audit("unknown")
    assert (f._borrow_tears, f._borrow_flat) == (0, 0)
    assert f.progress_smooth == "borrow"


def _screen_x_walk(n=200, screen=3, step=4, start=250):
    """A (screen, x) composite in ordinary motion: x advances by `step`
    and wraps inside a screen the high byte never leaves. Composed as
    screen*256 + x, which is exactly how `progress` composes a two-byte
    read, so the borrow rule sees a 252-count lo jump with hi flat."""
    return [screen * 256 + ((start + step * i) % 256) for i in range(n)]


def test_a_screen_x_composite_disarms_the_rule_instead_of_losing_frames():
    # THE DEFECT, end to end through observe(). Six of the eleven
    # two-byte profiles pair a screen with an in-screen x, and on those
    # every wrap is ordinary motion that the rule deletes.
    from tests.test_go_explore_solve_repairs import _observe_solver, _walk
    f = _observe_solver("borrow")
    f._borrow_audit = MethodType(Solver._borrow_audit, f)
    f._borrow_tears = f._borrow_flat = 0
    f._borrow_off = False
    _walk(f, _screen_x_walk())
    assert f._borrow_flat >= 3 and f._borrow_tears == 0
    assert f._borrow_off is True and f.progress_smooth == "off"
    # It disarmed EARLY: the losses stop at the refutation limit instead
    # of continuing for the whole run.
    assert f._prog_glitches <= 4


def test_the_castlevania_tear_keeps_the_rule_armed_and_keeps_working():
    # The control. $0040/$0041 IS a page/offset pair: the carry lands one
    # frame later, the audit confirms every rejection, and the filter
    # that §12 asked for is still there at the end of the run.
    from tests.test_go_explore_solve_repairs import (LEFT_WALK,
                                                     _observe_solver, _walk)
    f = _observe_solver("borrow")
    f._borrow_audit = MethodType(Solver._borrow_audit, f)
    f._borrow_tears = f._borrow_flat = 0
    f._borrow_off = False
    for _ in range(5):
        _walk(f, LEFT_WALK)                        # fresh ctx per burst
    assert f._borrow_tears == 5 and f._borrow_flat == 0
    assert f._borrow_off is False and f.progress_smooth == "borrow"
    assert f._prog_glitches == 5                   # the phantom, every time
    assert 767 // 8 not in {k[-1] for k in f.archive.cells}


def test_the_audit_is_silent_on_median3_and_on_the_default_path():
    from tests.test_go_explore_solve_repairs import (LEFT_WALK,
                                                     _observe_solver, _walk)
    for mode in ("off", "median3"):
        f = _observe_solver(mode)
        f._borrow_audit = MethodType(Solver._borrow_audit, f)
        f._borrow_tears = f._borrow_flat = 0
        f._borrow_off = False
        _walk(f, LEFT_WALK)
        assert (f._borrow_tears, f._borrow_flat, f._borrow_off) == (0, 0,
                                                                    False)


def test_the_resume_path_runs_both_halves_and_the_refreeze_in_order():
    # Structural, because the pieces are only correct in ONE arrangement:
    # the machine check must precede the multi-GB unpickle (a lineage
    # verdict is not worth an 11.9 GB read), the key audit can only run
    # after it, and the refreeze has to land before the frontier walk it
    # is supposed to correct.
    src = (REPO / "scripts" / "go_explore_solve.py").read_text()
    block = src.split('prev = getattr(self.args, "resume_archive", None)')[1]
    block = block.split("def key_config")[0]
    for call in ("self.check_resume_lineage(prev)",
                 "self.audit_resumed_keys(prev, keys)",
                 "phantom_top_buckets(keys)",
                 "not in self._gx_phantoms"):
        assert call in block, call
    assert (block.index("check_resume_lineage")
            < block.index("archive.load")
            < block.index("audit_resumed_keys")
            < block.index("phantom_top_buckets")
            < block.index("self.max_gx_in_area[area] = gx"))
