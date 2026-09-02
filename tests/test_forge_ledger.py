"""src/forge/ledger.py -- the FORGE ledger writer.

FORGE_SPEC_2026-09-01.md §2e. Each test is revert-verified against a
named corruption in its own docstring. No test here writes real
CLAIMS.md; every fixture lands under pytest's ``tmp_path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forge.ledger import (  # noqa: E402
    VocabularyViolation,
    append_addendum,
    check_vocabulary,
    iter_docstrings,
    render_entry,
)

LEDGER_PATH = ROOT / "src" / "forge" / "ledger.py"
THIS_TEST_PATH = ROOT / "tests" / "test_forge_ledger.py"

#: The order fixed by FORGE_SPEC_2026-09-01.md §2e, hardcoded here
#: rather than imported from ledger.SECTION_ORDER -- importing it would
#: make the test below tautological (a corrupted module order would
#: reorder both the render AND this expectation together, and the test
#: could never fail). Label text matches ledger.SECTION_LABELS exactly.
EXPECTED_SECTION_LABELS = (
    "Detection", "Mechanism", "Review", "Gate", "Tier-3 sentence",
    "Status at forging", "Citable as", "Addenda",
)


def _valid_entry(**overrides) -> dict:
    """A FORGE entry with clean vocabulary throughout every field --
    the baseline every test below perturbs one field of."""
    entry = {
        "status": "FORGE-PENDING-VALIDATION",
        "arm": "orthogonal-frontier arm",
        "flag": "--ortho",
        "commit": "abc1234",
        "date": "2026-09-02",
        "detection": (
            "The run's own selection telemetry showed the archive "
            "frontier pinned above y-band 9 across every member."),
        "mechanism": (
            "An opt-in selection arm that redirects pressure to "
            "per-column orthogonal extremes once the frontier pins."),
        "review": (
            "The adversarial review round found and fixed an "
            "area-restriction blocker."),
        "gate": (
            "Default-off byte-identity, covered by the inertness "
            "mirror at the shipped CLI defaults."),
        "telemetry": ["runs/cv_hall_ortho_a/progress.jsonl",
                       "runs/cv_hall_ortho_a/archive.stats.json"],
        "gate_met": True,
        "gate_measure": "the activity counter FIRED with n_obs greater than zero",
        "citable_as": "agent-forged; no clear may be attributed to it.",
        "addenda": [],
    }
    entry.update(overrides)
    return entry


# --------------------------------------------------------------- render


def test_render_has_all_eight_sections_in_order():
    """The eight body sections (spec §2e "fields in this order") appear
    in the rendered Markdown in SECTION_ORDER, every time.

    Revert-verify: swap "review" and "gate" in ledger.SECTION_ORDER (or
    in this test's expected list) and the strictly-increasing index
    assertion below fails -- Gate's label would appear before Review's.
    """
    text = render_entry(_valid_entry())

    positions = [text.index(f"**{label}.**") for label in EXPECTED_SECTION_LABELS]

    assert positions == sorted(positions), (
        "section labels are not in the spec's fixed order inside the "
        f"rendered entry: {list(zip(EXPECTED_SECTION_LABELS, positions))}")
    # Sanity: eight distinct sections were actually found, not one
    # substring match standing in for several.
    assert len(set(positions)) == 8


# ---------------------------------------------------------- vocabulary


def test_banned_verb_refused():
    """A mechanism sentence built around the banned-verb family's past
    tense (``bad`` below) is refused, both by the bare check and by
    render_entry() end to end.

    Revert-verify: drop the past-tense form from
    ledger._BANNED_VERB_RE's alternation and both assertions below
    fail -- check_vocabulary returns [] and render_entry returns text
    instead of raising.
    """
    bad = "the arm learned to redirect pressure toward the frontier"
    violations = check_vocabulary(bad)
    assert violations, "banned verb 'learned' was not flagged"
    assert any("learned" in v for v in violations)

    with pytest.raises(VocabularyViolation) as exc_info:
        render_entry(_valid_entry(mechanism=bad))
    assert any("learned" in v for v in exc_info.value.violations)


def test_clear_rate_refused_in_forge_entry():
    """A count-of-attempts phrase paired with a rate word (``bad``
    below) is refused inside a FORGE entry -- a FORGE entry states
    whether a gate was met, never an EXHIBITION clear rate.

    Revert-verify: change ledger._RATE_PROXIMITY_WINDOW to zero (or
    drop the rate words from _RATE_WORDS) and both assertions below
    fail -- the fixture no longer trips check_vocabulary.
    """
    bad = "the pilot banked 3/10 clears against the control"
    violations = check_vocabulary(bad)
    assert violations, "digit-bearing clear-rate phrase was not flagged"
    assert any("clear" in v.lower() for v in violations)

    with pytest.raises(VocabularyViolation):
        render_entry(_valid_entry(review=bad))


def test_metadiscourse_phrase_refused():
    """A field opening on one of the three fixed interpretive-
    metadiscourse phrases (``bad`` below) is refused, both by the bare
    check and by render_entry() end to end -- the 2026-09-01
    correction this vocabulary check exists to enforce (spec §2e
    header note).

    Revert-verify: drop the phrase from ledger._METADISCOURSE_PHRASES
    and both assertions below fail -- check_vocabulary returns [] and
    render_entry returns text instead of raising.
    """
    bad = "To be clear, the frontier pinned above y-band 9."
    violations = check_vocabulary(bad)
    assert violations, "interpretive-metadiscourse phrase was not flagged"
    assert any("metadiscourse" in v for v in violations)

    with pytest.raises(VocabularyViolation) as exc_info:
        render_entry(_valid_entry(detection=bad))
    assert any("metadiscourse" in v for v in exc_info.value.violations)


def test_whole_line_bold_refused():
    """A line that is nothing but a double-asterisk bold span is
    refused as the aphorism-kicker shape that leans on bold instead of
    italics.

    Revert-verify: change ledger._WHOLE_LINE_BOLD_RE to require three
    leading asterisks instead of two and both assertions below fail --
    the fixture's two-asterisk line no longer matches.
    """
    bad = "**the frontier pins and nothing else needs saying**"
    violations = check_vocabulary(bad)
    assert violations, "whole-line bold was not flagged"
    assert any("whole-line bold" in v for v in violations)

    with pytest.raises(VocabularyViolation):
        render_entry(_valid_entry(review=bad))


def test_caps_label_line_refused():
    """A standalone shouted-label line is refused; the same text
    lower-cased, in running prose, is not.

    Revert-verify: raise ledger._CAPS_LABEL_RE's uppercase-letter floor
    from two to three and the first assertion below fails -- "OK" (two
    uppercase letters) no longer matches.
    """
    bad = "the frontier pinned\nOK\nand stayed pinned"
    violations = check_vocabulary(bad)
    assert violations, "caps-label line was not flagged"
    assert any("caps-label" in v for v in violations)
    assert check_vocabulary("the frontier pinned ok and stayed pinned") == [], (
        "the same word in running prose, not on its own line, must not "
        "be flagged")

    with pytest.raises(VocabularyViolation):
        render_entry(_valid_entry(gate=bad))


def test_italic_kicker_line_refused():
    """A standalone single-asterisk italic line with no digit and no
    path separator is refused as an aphorism kicker; the same shape
    naming a receipt path (a digit or a "/" in it) is exempted.

    Revert-verify: drop the digit/path exemption from
    check_vocabulary's italic-kicker branch and the second assertion
    below fails -- the receipt-citation fixture starts tripping
    check_vocabulary too.
    """
    bad = "the frontier pinned\n*and that settles the matter*\nfor every member"
    violations = check_vocabulary(bad)
    assert violations, "italic kicker line was not flagged"
    assert any("aphorism-kicker" in v for v in violations)

    receipt = ("the frontier pinned\n"
               "*runs/cv_hall_ortho_a/A8_result.json*\n"
               "for every member")
    assert check_vocabulary(receipt) == [], (
        "an italic line naming a receipt path must be exempt from the "
        "aphorism-kicker check")


# -------------------------------------------------------------- tier-3


def test_tier3_sentence_present_and_names_telemetry():
    """The rendered Tier-3 sentence names every telemetry source passed
    in, verbatim, and an empty telemetry list is refused outright
    (spec §1's Tier-3 sentence requires knob settings "derived only
    from this run's own telemetry (named in the entry)").

    Revert-verify: change render_entry to default an empty telemetry
    list to `["none"]` instead of raising, and the pytest.raises block
    below fails to raise.
    """
    telemetry = ["runs/contra_wall/A1/boundary_probe.json",
                 "runs/contra_wall/A8/A8_result.json"]
    text = render_entry(_valid_entry(telemetry=telemetry))

    for source in telemetry:
        assert source in text, f"telemetry source {source!r} not named in the entry"
    assert "game-agnostic machinery" in text

    with pytest.raises(ValueError):
        render_entry(_valid_entry(telemetry=[]))


# ------------------------------------------------------------ addenda


def test_addendum_appends_never_rewrites(tmp_path):
    """append_addendum() only ever appends: every byte written by the
    original render_entry() call is still a byte-for-byte prefix of the
    file after an addendum lands.

    Revert-verify: change append_addendum's `open(entry_path, "ab")` to
    `"wb"` and this fails -- the original bytes are no longer a prefix
    of the post-addendum file (they are gone).
    """
    entry_path = tmp_path / "CLAIMS_ENTRY.md"
    original_text = render_entry(_valid_entry())
    entry_path.write_bytes(original_text.encode("utf-8"))
    original_bytes = entry_path.read_bytes()

    append_addendum(entry_path, {
        "date": "2026-09-03",
        "status": "STANDS",
        "text": "A later pilot block confirmed the same reading.",
    })

    new_bytes = entry_path.read_bytes()
    assert new_bytes.startswith(original_bytes), (
        "original entry bytes were not preserved as a prefix -- the "
        "file was rewritten, not appended to")
    assert new_bytes != original_bytes
    assert b"STANDS" in new_bytes[len(original_bytes):]

    # A malformed addendum (unknown status) is refused before any byte
    # is written -- the file is untouched.
    with pytest.raises(ValueError):
        append_addendum(entry_path, {
            "date": "2026-09-04", "status": "MAYBE", "text": "x"})
    assert entry_path.read_bytes() == new_bytes


# --------------------------------------------------------- self-check


def test_vocabulary_self_check_clean_over_own_docstrings():
    """FORGE_SPEC_2026-09-01.md §5's pre-commit gate: check_vocabulary
    at zero hits over every new .md file and docstring. Applied here,
    literally, to this piece's own two source files -- every
    module/class/function docstring in each, via iter_docstrings(),
    not the regex/tuple literals or test fixture strings those same
    files must contain to define and prove the bans (iter_docstrings
    draws that line structurally, with ast, so this test cannot be
    satisfied by weakening it to skip the offending prose).

    Revert-verify: reintroduce the struck "Status at forging, stated
    plainly: ..." phrase into ledger.py's ``_status_plain`` docstring
    and this fails -- check_vocabulary no longer returns [] for that
    docstring, and the assertion below names ledger.py and quotes the
    violation.
    """
    for path in (LEDGER_PATH, THIS_TEST_PATH):
        source = path.read_text()
        docstrings = iter_docstrings(source)
        assert docstrings, f"{path}: iter_docstrings found none -- check the parse"
        for doc in docstrings:
            violations = check_vocabulary(doc)
            assert violations == [], (
                f"{path}: a docstring fails check_vocabulary: {violations}\n"
                f"docstring: {doc!r}")


# ------------------------------------------------- header and dashes


def test_rendered_entry_parses_as_a_forge_entry_and_carries_no_em_dash():
    """The rendered entry opens with the bold ``**FORGE-...`` lead
    ``scripts/provenance_check.py`` actually parses, and no rendered
    surface emits U+2014.

    Both halves are load-bearing and were both wrong. The retired
    ``## FORGE-...`` heading parsed as nothing: ``parse_forge_entries``
    ends its section at the first ``## ``, so an entry rendered that
    way was invisible to the very checker meant to gate it. And the
    entry carried two U+2014, which the wave's prose check refuses in
    an added line. The whitespace after the status word is part of the
    parse: ``**FORGE-VOID:`` and ``**FORGE-VOID,`` both fail that
    regex, so a punctuation mark cannot stand in for the retired dash.

    Revert-verify (live, this session): render the header as
    ``## {status} ...`` again and the first assertion fails (the regex
    does not match) and the section-membership assertion fails (the
    real ``parse_forge_entries`` finds no entry). Second corruption:
    put U+2014 back in the header and in ``TIER3_TEMPLATE`` and the
    dash assertion fails naming both. Both restored, re-passed.
    """
    import re

    # The regex from scripts/provenance_check.py:179, copied rather
    # than imported: importing it would make this tautological if the
    # checker's own parse were ever loosened to admit what we emit.
    forge_entry_re = re.compile(r'^\*\*FORGE(?:-([A-Z][A-Z-]*))?\s')

    for status in ("FORGE-PENDING-VALIDATION", "FORGE-VOID",
                   "FORGE-VALIDATED-MECHANISM"):
        text = render_entry(_valid_entry(status=status))
        header = text.splitlines()[0]
        match = forge_entry_re.match(header)
        assert match, (
            f"provenance_check would not parse this entry's header: "
            f"{header!r}")
        assert status.split("-", 1)[1] == match.group(1)
        assert "\u2014" not in text, (
            f"rendered {status} entry emits U+2014: "
            f"{[l for l in text.splitlines() if chr(0x2014) in l]}")
        # The header is not a whole-line bold span, which the entry's
        # own check_vocabulary refuses -- the first body section rides
        # on the header line, as it does in every entry in CLAIMS.md.
        assert not header.endswith("**")
        assert "**Detection.**" in header
        assert check_vocabulary(text) == []

    # An addendum, rendered inline and appended, is a FORGE surface too.
    with_addendum = render_entry(_valid_entry(addenda=[
        {"date": "2026-09-03", "status": "STANDS",
         "text": "A later pilot block read the same."}]))
    assert "\u2014" not in with_addendum
    assert "Addendum, 2026-09-03, STANDS." in with_addendum


def test_appended_addendum_carries_no_em_dash(tmp_path):
    """``append_addendum`` writes to the same file the landing commit
    lands, so its block is held to the same no-U+2014 rule as the
    entry it follows.

    Revert-verify (live, this session): put U+2014 back between the
    date and the status in ``append_addendum``'s block and the dash
    assertion fails. Restored, re-passed.
    """
    entry_path = tmp_path / "CLAIMS_ENTRY.md"
    entry_path.write_text(render_entry(_valid_entry()))

    append_addendum(entry_path, {
        "date": "2026-09-03", "status": "WITHDRAWN",
        "text": "The pilot block was rerun under a corrected budget."})

    text = entry_path.read_text()
    assert "\u2014" not in text
    assert "*Addendum, 2026-09-03, WITHDRAWN.*" in text
    assert check_vocabulary(text) == []
