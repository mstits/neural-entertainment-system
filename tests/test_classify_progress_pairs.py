"""scripts/classify_progress_pairs.py — the co-occurrence rule applied
to every two-byte `progress: {lo, hi}` game profile, sourced entirely
from already-banked docs/receipts/ telemetry (no emulator driven).
"""

from __future__ import annotations

import pytest

from scripts.classify_progress_pairs import (
    DIRTY_CONFIGS,
    THRESHOLD,
    Verdict,
    _add_smooth_to_text,
    _apply_smooth,
    _castlevania,
    _contra,
    _double_dragon,
    _excitebike,
    _ghosts_n_goblins,
    _gradius,
    _kid_icarus,
    _kirby,
    _megaman,
    _metroid,
    _rule,
)

# --- the rule itself ----------------------------------------------------

def test_the_rule_calls_carry_coupled_above_the_threshold():
    assert _rule(91, 100) == "carry-coupled"


def test_the_rule_calls_composite_when_flat_share_clears_the_threshold():
    assert _rule(9, 100) == "composite"          # 9% coupled, 91% flat


def test_the_rule_is_unknown_in_the_middle():
    assert _rule(50, 100) == "UNKNOWN"


def test_the_rule_boundary_is_strictly_greater_than_not_greater_equal():
    exact = int(THRESHOLD * 100)
    assert _rule(exact, 100) == "UNKNOWN"        # exactly 90% is not > 90%


# --- per-game extraction is grounded in the actual receipted numbers ----

def test_contra_is_carry_coupled_seven_of_seven():
    v = _contra()
    assert (v.coupled, v.wraps) == (7, 7)
    assert v.verdict == "carry-coupled"


def test_contra_note_cites_the_correct_reentry_wrap_count_not_a_shifted_cell():
    # The re-verification table's candidate cell embeds an escaped pipe
    # (`` `$0065\|$0064<<8` ``); a naive split on every "|" shifts every
    # later cell over by one and would quote "net 0 (flat)" as the wrap
    # count instead of "3".
    v = _contra()
    assert "3/3 wrap-coupled" in v.note
    assert "flat" not in v.note


def test_ghosts_n_goblins_is_carry_coupled_twenty_eight_of_twenty_eight():
    v = _ghosts_n_goblins()
    assert (v.coupled, v.wraps) == (28, 28)
    assert v.verdict == "carry-coupled"


def test_castlevania_probe_alone_misses_the_bar_but_the_combined_verdict_is_carry_coupled():
    v = _castlevania()
    assert (v.coupled, v.wraps) == (8, 9)
    assert _rule(v.coupled, v.wraps) == "UNKNOWN"   # 88.9% < 90%, on the raw count
    assert v.verdict == "carry-coupled"             # combined with the gx-767 record
    assert v.profile in DIRTY_CONFIGS


def test_kirby_lands_in_unknown_just_under_the_bar_despite_a_large_n():
    v = _kirby()
    assert v.wraps == 151 + 46
    assert v.coupled == 128 + 46
    assert v.verdict == "UNKNOWN"


def test_double_dragon_is_unknown_on_too_small_a_sample():
    v = _double_dragon()
    assert (v.coupled, v.wraps) == (2, 3)
    assert v.verdict == "UNKNOWN"


def test_kid_icarus_is_composite_by_receipted_structural_evidence():
    v = _kid_icarus()
    assert v.verdict == "composite"


def test_excitebike_is_composite_by_receipted_exhaustive_search():
    v = _excitebike()
    assert v.verdict == "composite"


@pytest.mark.parametrize("fn", [_gradius, _metroid, _megaman])
def test_games_with_no_receipted_wrap_count_are_unknown_not_guessed(fn):
    v = fn()
    assert v.wraps is None and v.coupled is None
    assert v.verdict == "UNKNOWN"


def test_every_extractor_returns_a_verdict_object():
    for fn in (_castlevania, _contra, _ghosts_n_goblins, _kirby,
               _double_dragon, _kid_icarus, _excitebike, _gradius,
               _metroid, _megaman):
        v = fn()
        assert isinstance(v, Verdict)
        assert v.verdict in ("carry-coupled", "composite", "UNKNOWN")
        assert v.source.startswith("docs/receipts/")


# --- config-editing is pure, guarded, and idempotent ---------------------

def test_add_smooth_splices_into_a_bare_lo_hi_line():
    text = "solve:\n  progress: {lo: 0x0065, hi: 0x0064}\n  y: 0x031A\n"
    out = _add_smooth_to_text(text, "hampel")
    assert "progress: {lo: 0x0065, hi: 0x0064, smooth: hampel}" in out
    assert out.count("progress:") == 1


def test_add_smooth_is_a_noop_when_smooth_is_already_set():
    text = "progress: {lo: 0x0040, hi: 0x0041, smooth: median3}\n"
    assert _add_smooth_to_text(text, "hampel") is None


def test_add_smooth_is_a_noop_with_no_bare_lo_hi_line():
    assert _add_smooth_to_text("progress: {tiles: [1, 2]}\n", "hampel") is None


def test_apply_smooth_refuses_a_dirty_config_even_if_carry_coupled(tmp_path, monkeypatch):
    import scripts.classify_progress_pairs as m
    cfg = tmp_path / "configs" / "castlevania.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("solve:\n  progress: {lo: 0x0040, hi: 0x0041}\n")
    monkeypatch.setattr(m, "REPO", tmp_path)
    v = Verdict("castlevania", "configs/castlevania.yaml", 0x40, 0x41,
               8, 9, "carry-coupled", "docs/receipts/x.json", "")
    result = _apply_smooth(v, "hampel")
    assert result == "not applied (guard)"
    assert "smooth" not in cfg.read_text()


def test_apply_smooth_refuses_a_non_carry_coupled_verdict(tmp_path, monkeypatch):
    import scripts.classify_progress_pairs as m
    cfg = tmp_path / "configs" / "kirby.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("solve:\n  progress: {lo: 0x0083, hi: 0x0095}\n")
    monkeypatch.setattr(m, "REPO", tmp_path)
    v = Verdict("kirby", "configs/kirby.yaml", 0x83, 0x95, 174, 197,
               "UNKNOWN", "docs/receipts/x.json", "")
    result = _apply_smooth(v, "hampel")
    assert result == "not applied (guard)"
    assert "smooth" not in cfg.read_text()


def test_apply_smooth_writes_a_clean_carry_coupled_config(tmp_path, monkeypatch):
    import scripts.classify_progress_pairs as m
    cfg = tmp_path / "configs" / "contra.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("solve:\n  progress: {lo: 0x0065, hi: 0x0064}\n  y: 1\n")
    monkeypatch.setattr(m, "REPO", tmp_path)
    v = Verdict("contra", "configs/contra.yaml", 0x65, 0x64, 7, 7,
               "carry-coupled", "docs/receipts/x.json", "")
    result = _apply_smooth(v, "hampel")
    assert "progress.smooth: hampel added" in result
    assert "smooth: hampel" in cfg.read_text()
