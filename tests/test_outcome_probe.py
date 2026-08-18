"""Tests for scripts/outcome_probe.py — the comparison-outcome detector.

Everything here runs against SYNTHETIC traces with known ground truth
(built by the synth_* functions in the module itself, reused by
`--selftest`) or hand-built numpy arrays. Nothing imports or requires
nes_core / a ROM / a save state: this lane is token-bound, the 2-1
campaign owns the emulator, and `record_trace_live` (the one function
that *can* touch it) is exercised only for its refuse-without-`allow`
guard.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.outcome_probe import (
    Field,
    Outcome,
    Trace,
    TEResult,
    bh_correct,
    classify_trace,
    compute_field_stats,
    enumerate_fields,
    find_symmetric_pairs,
    load_fields,
    load_trace,
    monotone_timer_score,
    record_trace_live,
    save_trace_npz,
    symmetry_score,
    synth_ambiguous_pair_trace,
    synth_driven_field_trace,
    synth_mirrored_pair_trace,
    synth_noise_trace,
    synth_timer_trace,
    te_significance,
    transfer_entropy,
    verdict_to_dict,
)

N_SURROGATES = 150  # smaller than the CLI default (200) for test speed


# ---------------------------------------------------------------------
# Field: extraction, hashability, validation.
# ---------------------------------------------------------------------


def test_field_extract_single_byte():
    ram = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    f = Field("b1", (1,))
    np.testing.assert_array_equal(f.extract(ram), [2, 5])


def test_field_extract_le_and_be():
    ram = np.array([[0x34, 0x12]], dtype=np.uint8)  # offset 0 = lo, offset 1 = hi
    le = Field("le", (0, 1), endianness="le")
    be = Field("be", (0, 1), endianness="be")
    assert int(le.extract(ram)[0]) == 0x1234
    assert int(be.extract(ram)[0]) == 0x3412


def test_field_rejects_bad_offsets_and_endianness():
    with pytest.raises(ValueError):
        Field("bad", (0, 1, 2))
    with pytest.raises(ValueError):
        Field("bad", (0, 1), endianness="middle")


def test_field_is_hashable_and_comparable():
    a = Field("x", (2,))
    b = Field("x", (2,))
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_enumerate_fields_count():
    n = 6
    fields = enumerate_fields(n)
    singles = [f for f in fields if f.width_bytes == 1]
    pairs = [f for f in fields if f.width_bytes == 2]
    assert len(singles) == n
    assert len(pairs) == 2 * (n - 1)  # le + be per adjacent offset

    only_singles = enumerate_fields(n, include_pairs=False)
    assert len(only_singles) == n


# ---------------------------------------------------------------------
# Trace container + validation + I/O round trips.
# ---------------------------------------------------------------------


def test_trace_requires_matching_lengths():
    ram = np.zeros((10, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        Trace(ram=ram, inputs=np.zeros(10, dtype=np.uint8))  # off by one
    Trace(ram=ram, inputs=np.zeros(9, dtype=np.uint8))  # this one is fine


def test_trace_requires_2d_ram():
    with pytest.raises(ValueError):
        Trace(ram=np.zeros(10, dtype=np.uint8), inputs=np.zeros(9, dtype=np.uint8))


def test_npz_round_trip(tmp_path):
    ram = np.random.default_rng(0).integers(0, 256, size=(20, 5)).astype(np.uint8)
    inputs = np.random.default_rng(1).integers(0, 256, size=19).astype(np.uint8)
    path = tmp_path / "trace.npz"
    save_trace_npz(path, ram, inputs, meta={"rom": "test.nes", "n": 3})
    t = load_trace(path)
    np.testing.assert_array_equal(t.ram, ram)
    np.testing.assert_array_equal(t.inputs, inputs)
    assert t.meta == {"rom": "test.nes", "n": 3}


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {"ram": [1, 2, 3], "input": 0x01},
        {"ram": [1, 3, 3], "input": 0x00},
        {"ram": [1, 3, 4]},  # last row: no further transition, "input" optional
    ]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    t = load_trace(path)
    assert t.ram.shape == (3, 3)
    np.testing.assert_array_equal(t.inputs, [0x01, 0x00])


def test_load_trace_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("nope")
    with pytest.raises(ValueError):
        load_trace(path)


def test_load_fields(tmp_path):
    path = tmp_path / "fields.json"
    path.write_text(json.dumps([
        {"name": "score", "offsets": [2]},
        {"name": "pos", "offsets": [4, 5], "endianness": "be"},
    ]))
    fields = load_fields(path)
    assert fields[0] == Field("score", (2,))
    assert fields[1] == Field("pos", (4, 5), endianness="be")


# ---------------------------------------------------------------------
# Transfer entropy: known-dependent vs known-independent processes.
# ---------------------------------------------------------------------


def test_te_high_for_deterministic_dependence():
    """values[n+1] = controller[n] deterministically -> the controller
    fully determines the next value given nothing; TE should be large
    (close to the 1-bit ceiling of a binary target)."""
    rng = np.random.default_rng(0)
    n = 2000
    controller = rng.integers(0, 2, size=n).astype(np.uint8)
    values = np.empty(n + 1, dtype=np.int64)
    values[0] = 0
    values[1:] = controller
    te = transfer_entropy(controller, values)
    assert te > 0.5


def test_te_near_zero_for_independent_process():
    rng = np.random.default_rng(1)
    n = 2000
    controller = rng.integers(0, 256, size=n).astype(np.uint8)
    values = rng.integers(0, 4, size=n + 1).astype(np.int64)
    te = transfer_entropy(controller, values)
    assert te < 0.1


def test_te_requires_matching_lengths():
    with pytest.raises(ValueError):
        transfer_entropy(np.zeros(5), np.zeros(5))  # needs len(values) == len(controller)+1


def test_te_significance_detects_dependence_and_independence():
    rng = np.random.default_rng(0)
    n = 600
    controller = rng.integers(0, 2, size=n).astype(np.uint8)
    dependent = np.empty(n + 1, dtype=np.int64)
    dependent[0] = 0
    dependent[1:] = controller
    r_dep = te_significance(controller, dependent, n_surrogates=N_SURROGATES, seed=0)
    assert isinstance(r_dep, TEResult)
    assert r_dep.p_value < 0.01

    independent = rng.integers(0, 4, size=n + 1).astype(np.int64)
    r_indep = te_significance(controller, independent, n_surrogates=N_SURROGATES, seed=1)
    assert r_indep.p_value > 0.05


# ---------------------------------------------------------------------
# BH correction.
# ---------------------------------------------------------------------


def test_bh_correct_rejects_only_the_small_pvalues():
    pvals = [0.001, 0.002, 0.5, 0.6]
    reject = bh_correct(pvals, alpha=0.05)
    assert list(reject) == [True, True, False, False]


def test_bh_correct_empty_and_none_significant():
    assert bh_correct([], alpha=0.05).size == 0
    reject = bh_correct([0.9, 0.8, 0.95], alpha=0.05)
    assert not reject.any()


# ---------------------------------------------------------------------
# Monotone-timer test.
# ---------------------------------------------------------------------


def test_monotone_timer_score_on_clean_decrement():
    values = np.arange(200, 0, -1)
    stats = monotone_timer_score(values)
    assert stats["is_monotone"] is True
    assert stats["direction"] == -1
    assert stats["step_consistency"] == pytest.approx(1.0)
    assert stats["frac_nonzero"] == pytest.approx(1.0)


def test_monotone_timer_score_rejects_noise():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 256, size=200)
    stats = monotone_timer_score(values)
    assert stats["is_monotone"] is False


def test_monotone_timer_score_on_constant_field():
    values = np.full(50, 7)
    stats = monotone_timer_score(values)
    assert stats["is_monotone"] is False
    assert stats["frac_nonzero"] == 0.0


# ---------------------------------------------------------------------
# Symmetry detector.
# ---------------------------------------------------------------------


def _stats_from_values(name, offset, values):
    ram = values.reshape(-1, 1).astype(np.uint8)
    f = Field(name, (0,))
    t = Trace(ram=ram, inputs=np.zeros(len(values) - 1, dtype=np.uint8))
    return compute_field_stats(t, f)


def test_symmetry_score_identical_fields_is_high():
    v = np.cumsum(np.random.default_rng(0).integers(0, 2, size=100))
    a = _stats_from_values("a", 0, v)
    b = _stats_from_values("b", 0, v.copy())
    assert symmetry_score(a, b) > 0.9


def test_symmetry_score_different_width_is_zero():
    values = np.arange(20)
    a = _stats_from_values("a", 0, values)
    b_field = Field("b", (0, 1))
    ram = np.zeros((20, 2), dtype=np.uint8)
    ram[:, 0] = values % 256
    b = compute_field_stats(Trace(ram=ram, inputs=np.zeros(19, dtype=np.uint8)), b_field)
    assert symmetry_score(a, b) == 0.0


def test_find_symmetric_pairs_skips_overlapping_offsets():
    ram = np.zeros((20, 3), dtype=np.uint8)
    ram[:, 0] = np.arange(20) % 256
    ram[:, 1] = np.arange(20) % 256
    trace = Trace(ram=ram, inputs=np.zeros(19, dtype=np.uint8))
    f_byte = Field("byte0", (0,))
    f_pair = Field("pair01", (0, 1))  # overlaps byte0 at offset 0
    stats = [compute_field_stats(trace, f_byte), compute_field_stats(trace, f_pair)]
    pairs = find_symmetric_pairs(stats, min_score=0.0)
    assert pairs == []


# ---------------------------------------------------------------------
# classify_trace on the five synthetic ground-truth cases.
# ---------------------------------------------------------------------


def test_classify_driven_field_is_threshold_reached():
    trace = synth_driven_field_trace()
    v = classify_trace(trace, [Field("score", (2,))], n_surrogates=N_SURROGATES, seed=0)[0]
    assert v.outcome == Outcome.THRESHOLD_REACHED
    assert v.evidence["te"]["significant"] is True


def test_classify_mirrored_pair_splits_player_and_opponent():
    trace = synth_mirrored_pair_trace()
    fields = [Field("player_score", (2,)), Field("opp_score", (3,))]
    player, opp = classify_trace(trace, fields, n_surrogates=N_SURROGATES, seed=1)
    assert player.outcome == Outcome.SCORE_WIN_VS_OPPONENT
    assert player.paired_with == Field("opp_score", (3,))
    assert opp.outcome == Outcome.UNSCORABLE


def test_classify_timer_is_time_target_beaten():
    trace = synth_timer_trace()
    v = classify_trace(trace, [Field("timer", (4,))], n_surrogates=N_SURROGATES, seed=2)[0]
    assert v.outcome == Outcome.TIME_TARGET_BEATEN
    assert v.evidence["timer"]["is_monotone"] is True


def test_classify_noise_field_abstains():
    trace = synth_noise_trace()
    v = classify_trace(trace, [Field("noise", (6,))], n_surrogates=N_SURROGATES, seed=3)[0]
    assert v.outcome == Outcome.UNSCORABLE


def test_classify_lag2_driven_field_not_misclassified_as_timer():
    """A genuinely controller-driven, strictly non-decreasing score field
    whose response lands TWO steps after the trace's own order-1
    convention (increments when the controller bit was held one step
    EARLIER than order-1 assumes) must never be confidently mislabeled
    TIME_TARGET_BEATEN: it is either correctly detected via the
    multi-lag search, or -- when the signal is too weak for even that
    to clear the FDR bar -- it must ABSTAIN (UNSCORABLE), not commit to
    the wrong pre-registered category. Regression test for a confirmed
    defect: the order-1-only TE test had no power to see this lag and
    fell straight into TIME_TARGET_BEATEN with no abstain branch."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = 600
        inputs = np.where(rng.random(t - 1) < 0.35, 0x01, 0x00).astype(np.uint8)
        ram = np.zeros((t, 8), dtype=np.uint8)
        score = 0
        for i in range(t - 1):
            if i >= 1 and (inputs[i - 1] & 0x01):
                score += 1
            ram[i + 1, 2] = score & 0xFF
        trace = Trace(ram=ram, inputs=inputs)
        v = classify_trace(trace, [Field("lag2", (2,))], n_surrogates=N_SURROGATES, seed=0)[0]
        assert v.outcome != Outcome.TIME_TARGET_BEATEN, (
            f"seed {seed}: lag-2-driven score field misclassified as TIME_TARGET_BEATEN"
        )


def test_classify_noisy_partial_coupling_not_misclassified_as_timer():
    """A field driven by the controller only part of the time (real
    coupling, diluted by spontaneous same-step increments) must not be
    confidently mislabeled TIME_TARGET_BEATEN either -- same defect
    class as the lag-2 case, different mechanism (weak/noisy order-1
    coupling the raw TE prefilter alone has no power to flag)."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = 600
        inputs = np.where(rng.random(t - 1) < 0.5, 0x01, 0x00).astype(np.uint8)
        ram = np.zeros((t, 8), dtype=np.uint8)
        score = 0
        for i in range(t - 1):
            if (inputs[i] & 0x01) or (rng.random() < 0.30):
                score += 1
            ram[i + 1, 2] = score & 0xFF
            ram[i + 1, 3] = (score >> 8) & 0xFF
        trace = Trace(ram=ram, inputs=inputs)
        v = classify_trace(trace, [Field("noisy16", (2, 3))], n_surrogates=N_SURROGATES, seed=0)[0]
        assert v.outcome != Outcome.TIME_TARGET_BEATEN, (
            f"seed {seed}: noisy partially-coupled score field misclassified as TIME_TARGET_BEATEN"
        )


def test_classify_independent_timer_false_abstain_rate_not_inflated():
    """Guard against re-introducing a selection-bias bug: picking
    whichever of several lags scores highest on the REAL data and then
    testing only that lag against a null built without the same
    "best of several" advantage systematically inflates z-scores even
    for genuinely independent fields. With that bug present, a
    genuinely-independent synthetic timer false-abstained (UNSCORABLE
    instead of TIME_TARGET_BEATEN) in roughly half of trials; correctly
    maximizing the null over the same lag set keeps it near the nominal
    one-sided rate for TIMER_NULL_Z_MAX=1.0 (~16%). Assert well below
    the buggy regime rather than pin an exact rate, to avoid flakiness."""
    false_abstains = 0
    trials = 20
    for seed in range(trials):
        trace = synth_timer_trace(seed=seed)
        v = classify_trace(trace, [Field("timer", (4,))], n_surrogates=N_SURROGATES, seed=seed)[0]
        if v.outcome != Outcome.TIME_TARGET_BEATEN:
            false_abstains += 1
    assert false_abstains <= trials * 0.35, (
        f"{false_abstains}/{trials} false-abstains on a genuinely independent timer "
        "-- looks like the max-lag-selection-bias bug is back"
    )


def test_classify_ambiguous_pair_abstains_on_both_sides():
    """Deliberately ambiguous ground truth: two structurally identical
    fields both driven by the same controller bit. Nothing in a single
    controller stream can tell them apart, so precision-over-recall
    means the classifier must ABSTAIN on both rather than crown either
    one the "player"."""
    trace = synth_ambiguous_pair_trace()
    fields = [Field("a", (2,)), Field("b", (3,))]
    va, vb = classify_trace(trace, fields, n_surrogates=N_SURROGATES, seed=4)
    assert va.outcome == Outcome.UNSCORABLE
    assert vb.outcome == Outcome.UNSCORABLE
    assert "ambiguous" in va.evidence["reason"]


def test_classify_short_trace_abstains_everywhere():
    ram = np.zeros((5, 4), dtype=np.uint8)
    inputs = np.zeros(4, dtype=np.uint8)
    trace = Trace(ram=ram, inputs=inputs)
    verdicts = classify_trace(trace, n_surrogates=N_SURROGATES)
    assert all(v.outcome == Outcome.UNSCORABLE for v in verdicts)
    assert "too short" in verdicts[0].evidence["reason"]


def test_verdict_to_dict_is_json_serializable():
    trace = synth_timer_trace()
    v = classify_trace(trace, [Field("timer", (4,))], n_surrogates=N_SURROGATES, seed=2)[0]
    row = verdict_to_dict(v)
    json.dumps(row)  # raises on any leftover numpy scalar
    assert row["outcome"] == Outcome.TIME_TARGET_BEATEN.value
    assert row["field"] == "timer"


# ---------------------------------------------------------------------
# The full auto-scan (enumerate_fields) on a small synthetic RAM image,
# end to end — not just hand-picked single fields.
# ---------------------------------------------------------------------


def test_classify_trace_auto_scan_finds_the_driven_field():
    trace = synth_driven_field_trace(n_bytes=6, offset=2)
    verdicts = classify_trace(trace, n_surrogates=N_SURROGATES, seed=0)
    by_name = {v.field.name: v for v in verdicts}
    assert by_name["b2"].outcome in (Outcome.THRESHOLD_REACHED, Outcome.SCORE_WIN_VS_OPPONENT,
                                      Outcome.POSITION_TARGET_MET)
    # the constant junk byte at offset 5 must never be classified as a win condition
    assert by_name["b5"].outcome == Outcome.UNSCORABLE


# ---------------------------------------------------------------------
# CLI end to end (classify subcommand) against an on-disk trace.
# ---------------------------------------------------------------------


def test_cli_classify_writes_json_report(tmp_path):
    from scripts.outcome_probe import main

    trace = synth_timer_trace()
    trace_path = tmp_path / "timer.npz"
    save_trace_npz(trace_path, trace.ram, trace.inputs, meta=trace.meta)

    fields_path = tmp_path / "fields.json"
    fields_path.write_text(json.dumps([{"name": "timer", "offsets": [4]}]))

    out_path = tmp_path / "report.json"
    rc = main([
        "classify", "--trace", str(trace_path), "--fields", str(fields_path),
        "--surrogates", str(N_SURROGATES), "--seed", "2", "--out", str(out_path),
    ])
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report[0]["outcome"] == Outcome.TIME_TARGET_BEATEN.value


def test_cli_selftest_passes():
    from scripts.outcome_probe import main
    assert main(["selftest"]) == 0


def test_cli_no_command_prints_help_and_returns_2(capsys):
    from scripts.outcome_probe import main
    rc = main([])
    assert rc == 2
    out = capsys.readouterr().out
    assert "usage" in out.lower()


# ---------------------------------------------------------------------
# record_trace_live: the ONLY function allowed to touch the emulator,
# and only ever with an explicit allow=True this test never sets.
# ---------------------------------------------------------------------


def test_record_trace_live_refuses_without_allow(tmp_path):
    with pytest.raises(RuntimeError, match="allow=True"):
        record_trace_live("roms/fake.nes", b"\x00" * 8, tmp_path / "out.npz")


def test_module_does_not_import_nes_core_at_top_level():
    """Static guard: `nes_core` must only ever be imported from inside
    record_trace_live's own body (never at module scope), so importing
    this module or running its classifier/tests never touches the
    emulator regardless of whether nes_core is even installed."""
    import scripts.outcome_probe as mod

    src = mod.__file__
    with open(src) as fh:
        lines = fh.readlines()
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("import nes_core") or stripped.startswith("from nes_core"):
            pytest.fail(f"top-level nes_core import found: {stripped!r}")
