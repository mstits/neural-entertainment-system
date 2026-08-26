"""Unsafe-inventory gate (scripts/check_unsafe_inventory.py).

`nes_core/SECURITY.md`'s per-file `unsafe` breakdown drifted for ~4
months while the crate grew from 3 call sites to 155 across 11 files —
undetected, because nothing mechanical ever re-counted it. These tests
pin the checker's parsing and comparison logic against synthetic
fixtures (never the live repo tree, which changes out from under a
fixed assertion) plus a mutation check: feed the checker a deliberately
drifted inventory and assert it fails, so the lint cannot rot into
matching nothing the way the document it audits already did once.
"""
from __future__ import annotations

from scripts.check_unsafe_inventory import (
    actual_unsafe_counts,
    compare,
    is_stale_placeholder,
    parse_security_md_inventory,
    parse_summary_totals,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---------------------------------------------------------------------------
# actual_unsafe_counts — the grep -c-style tree scan
# ---------------------------------------------------------------------------

def test_actual_counts_matches_grep_c_line_semantics(tmp_path):
    src = tmp_path / "src"
    _write(src / "pool.rs", "fn a() {\n  unsafe { x() }\n}\n// unsafe: ok\n")
    _write(src / "clean.rs", "fn b() {}\n")

    counts = actual_unsafe_counts(src)

    assert counts == {"src/pool.rs": 2}


def test_actual_counts_counts_a_line_once_even_with_two_matches(tmp_path):
    src = tmp_path / "src"
    # unsafe_op_in_unsafe_fn contains the substring "unsafe" twice on one
    # line; grep -c counts matching LINES, not occurrences, and
    # SECURITY.md's own methodology says it does the same.
    _write(src / "lib.rs", "#![allow(unsafe_op_in_unsafe_fn)]\nfn a() {}\n")

    counts = actual_unsafe_counts(src)

    assert counts == {"src/lib.rs": 1}


def test_actual_counts_recurses_into_subdirectories(tmp_path):
    src = tmp_path / "src"
    _write(src / "mapper" / "mapper1.rs", "unsafe { x() }\n")

    counts = actual_unsafe_counts(src)

    assert counts == {"src/mapper/mapper1.rs": 1}


def test_actual_counts_omits_files_with_zero_matches(tmp_path):
    src = tmp_path / "src"
    _write(src / "clean.rs", "fn a() {}\n")

    assert actual_unsafe_counts(src) == {}


def test_actual_counts_on_missing_src_root_is_empty(tmp_path):
    assert actual_unsafe_counts(tmp_path / "does_not_exist") == {}


# ---------------------------------------------------------------------------
# parse_security_md_inventory — tolerant of formatting variation
# ---------------------------------------------------------------------------

def test_parses_numbered_em_dash_heading():
    text = "### 1. `src/pool.rs` — worker-pool concurrency (75)\n"
    assert parse_security_md_inventory(text) == {"src/pool.rs": (75, 1)}


def test_parses_unnumbered_hyphen_heading():
    text = "### `src/lib.rs` - crate-wide lint allow (2)\n"
    assert parse_security_md_inventory(text) == {"src/lib.rs": (2, 1)}


def test_parses_heading_with_trailing_qualifier_text_in_parens():
    text = ("### 6. `src/preprocess.rs` — NEON SIMD intrinsics "
            "(5, 2 of them real)\n")
    assert parse_security_md_inventory(text) == {"src/preprocess.rs": (5, 1)}


def test_parses_heading_at_different_heading_depth():
    text = "#### 2. `src/cpu_asm.rs` -- ASM core (28)\n"
    assert parse_security_md_inventory(text) == {"src/cpu_asm.rs": (28, 1)}


def test_parses_multiple_files_and_tracks_doc_line_numbers():
    text = (
        "intro line\n"
        "### 1. `src/pool.rs` — pool (75)\n"
        "some prose\n"
        "more prose\n"
        "### 2. `src/nes.rs` — glue (5)\n"
    )
    inv = parse_security_md_inventory(text)
    assert inv == {"src/pool.rs": (75, 2), "src/nes.rs": (5, 5)}


def test_normalizes_nes_core_prefixed_paths():
    text = "### 1. `nes_core/src/pool.rs` — pool (75)\n"
    assert parse_security_md_inventory(text) == {"src/pool.rs": (75, 1)}


def test_ignores_non_heading_lines_that_merely_contain_a_count():
    text = "This paragraph mentions (75) in passing, not as a heading.\n"
    assert parse_security_md_inventory(text) == {}


def test_parse_summary_totals_reads_bolded_crate_line():
    text = "**155 lines match `unsafe` across 11 files** (re-counted)\n"
    assert parse_summary_totals(text) == (155, 11)


def test_parse_summary_totals_none_when_absent():
    assert parse_summary_totals("nothing here\n") is None


# ---------------------------------------------------------------------------
# is_stale_placeholder — the ordering-problem detector
# ---------------------------------------------------------------------------

def test_detects_stale_placeholder_with_no_inventory_headings():
    text = (
        "## `unsafe` surface\n\n"
        "There are three call sites in the entire crate:\n"
        "1. foo\n2. bar\n3. baz\n"
    )
    assert is_stale_placeholder(text) is True


def test_refreshed_doc_with_headings_is_not_stale_even_if_marker_lingers():
    # Defensive: if a future refresh keeps the old phrase in a "here's
    # what used to be wrong" history note alongside a real per-file
    # inventory, that is not the ordering-problem case.
    text = (
        "This section previously said three call sites in the entire "
        "crate.\n\n### 1. `src/pool.rs` — pool (75)\n"
    )
    assert is_stale_placeholder(text) is False


def test_normal_refreshed_doc_is_not_stale():
    text = "### 1. `src/pool.rs` — pool (75)\n"
    assert is_stale_placeholder(text) is False


# ---------------------------------------------------------------------------
# compare — the actual gate logic
# ---------------------------------------------------------------------------

def test_compare_clean_when_everything_matches():
    actual = {"src/pool.rs": 75, "src/nes.rs": 5}
    claimed = {"src/pool.rs": (75, 10), "src/nes.rs": (5, 20)}
    assert compare(actual, claimed) == []


def test_compare_tolerates_drift_within_tolerance():
    actual = {"src/pool.rs": 76}
    claimed = {"src/pool.rs": (75, 10)}
    assert compare(actual, claimed, tolerance=2) == []


def test_compare_fails_on_file_missing_from_doc():
    actual = {"src/pool.rs": 75, "src/new_file.rs": 4}
    claimed = {"src/pool.rs": (75, 10)}

    errors = compare(actual, claimed)

    assert len(errors) == 1
    assert "src/new_file.rs" in errors[0]
    assert "no entry at all" in errors[0]


def test_compare_fails_on_stale_doc_entry_for_deleted_file():
    actual = {"src/pool.rs": 75}
    claimed = {"src/pool.rs": (75, 10), "src/deleted.rs": (9, 30)}

    errors = compare(actual, claimed)

    assert len(errors) == 1
    assert "src/deleted.rs" in errors[0]
    assert "no longer exists" in errors[0]


def test_compare_fails_when_count_drifts_beyond_tolerance():
    actual = {"src/pool.rs": 3}
    claimed = {"src/pool.rs": (150, 10)}

    errors = compare(actual, claimed, tolerance=2)

    assert len(errors) == 1
    assert "src/pool.rs" in errors[0]
    assert "150" in errors[0] and "3" in errors[0]


def test_compare_names_every_violation_not_just_the_first():
    actual = {"src/a.rs": 10, "src/b.rs": 1}
    claimed = {"src/a.rs": (2, 5), "src/c.rs": (7, 9)}

    errors = compare(actual, claimed, tolerance=0)

    joined = "\n".join(errors)
    assert "src/a.rs" in joined  # drifted
    assert "src/b.rs" in joined  # missing from doc
    assert "src/c.rs" in joined  # stale doc entry
    assert len(errors) == 3


# ---------------------------------------------------------------------------
# mutation check — the lint must actually have teeth
# ---------------------------------------------------------------------------

def test_lint_has_teeth_against_a_deliberately_drifted_inventory():
    """Take an inventory that is honest against its tree (must pass),
    then apply the exact drift that bit SECURITY.md for real — a file's
    unsafe surface grows by two orders of magnitude while the doc still
    claims the old, small number — and assert the checker now fails.
    Guards against a comparison that accidentally matches everything.
    """
    honest_tree = {"src/pool.rs": 75, "src/nes.rs": 5, "src/lib.rs": 2}
    honest_doc = {
        "src/pool.rs": (75, 10),
        "src/nes.rs": (5, 20),
        "src/lib.rs": (2, 30),
    }
    assert compare(honest_tree, honest_doc) == []

    # Mutate: pool.rs grows from 75 to 150 in the tree (new unsafe impls
    # land) but the doc is not updated — the real "three call sites"
    # failure mode, reproduced at file scale instead of crate scale.
    drifted_tree = dict(honest_tree, **{"src/pool.rs": 150})
    errors = compare(drifted_tree, honest_doc)
    assert errors, (
        "the checker passed a tree where a file's unsafe count doubled "
        "against an unchanged doc — it is not actually comparing counts")
    assert any("src/pool.rs" in e for e in errors)

    # Mutate differently: a brand-new unsafe file lands and the doc's
    # per-file inventory is never touched.
    tree_with_new_file = dict(honest_tree, **{"src/ppu_neon.rs": 3})
    errors2 = compare(tree_with_new_file, honest_doc)
    assert errors2, (
        "the checker passed a tree with a wholly new unsafe file absent "
        "from the doc — it is not actually checking for missing entries")
    assert any("src/ppu_neon.rs" in e for e in errors2)


def test_lint_has_teeth_against_a_reworded_but_hollow_heading():
    """A heading that keeps the file/count shape but reads nothing like
    the ones this parser was built against must still parse — and a
    heading that drops the count entirely must NOT silently parse as
    zero drift (i.e. as if the file were simply absent from the doc,
    which IS the correct, loud failure — not a silent pass)."""
    hollow = "### `src/pool.rs` — some new description with no count at all\n"
    inv = parse_security_md_inventory(hollow)
    assert inv == {}, (
        "a heading with no parenthesized count parsed as SOME claimed "
        "count — it must parse as nothing, so compare() reports the "
        "file as missing rather than silently trusting a bogus number")

    errors = compare({"src/pool.rs": 75}, inv)
    assert errors and "src/pool.rs" in errors[0]
