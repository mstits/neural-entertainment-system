"""README's Bubble Bobble round numbers must match the banked chain receipts.

The round count is a public claim printed at five README sites, and its raw
evidence is a gitignored run directory. So this file checks two separate
things: that the numerals README prints agree with what
`runs/bubble_bobble/*/chain.jsonl` actually banked, and that the in-repo
receipt a fresh clone can read is present, cited, and says the same numbers.
The first three tests are keyed on a number rather than on a line, so putting
a stale figure back at any one of the five sites turns one of them red.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
RECEIPT = REPO / "docs" / "receipts" / "bubble_bobble_chain_census_2026-09-02.md"
CHAIN_DIR = REPO / "runs" / "bubble_bobble"

# Not preceded by a word character or hyphen: README also carries
# checkpoint names like "consolidation-round-2", which are not Bubble
# Bobble round numerals and must not be read as one.
ROUND_NUMERAL = re.compile(r"(?<![\w-])round[- ](\d+)\b")
SUB_LEVEL = re.compile(r"^(\d+)-(\d+)$")


def _chain_files():
    files = sorted(CHAIN_DIR.glob("*/chain.jsonl")) if CHAIN_DIR.is_dir() else []
    if not files:
        pytest.skip(
            "runs/ is gitignored: no runs/bubble_bobble/*/chain.jsonl on this "
            "checkout, so the banked chain cannot be re-derived here"
        )
    return files


def _chain_rows():
    rows = []
    for path in _chain_files():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _banked():
    """(highest contiguous whole round, split round, solved half, open half)."""
    rows = _chain_rows()
    solved = {r["level"] for r in rows if r.get("status") == "solved"}
    unsolved = {r["level"] for r in rows if r.get("status") != "solved"} - solved

    whole = sorted(int(lvl) for lvl in solved if lvl.isdigit())
    assert whole, "no whole-numbered round is marked solved in any chain.jsonl"
    highest = 0
    for n in range(1, whole[-1] + 1):
        if n not in whole:
            break
        highest = n

    solved_halves = sorted(lvl for lvl in solved if SUB_LEVEL.match(lvl))
    open_halves = sorted(lvl for lvl in unsolved if SUB_LEVEL.match(lvl))
    assert solved_halves, "no split round is marked solved in any chain.jsonl"
    assert open_halves, "no split round is left open in any chain.jsonl"
    solved_half = solved_halves[-1]
    split = int(SUB_LEVEL.match(solved_half).group(1))
    open_half = sorted(
        lvl for lvl in open_halves if int(SUB_LEVEL.match(lvl).group(1)) == split
    )[0]
    return highest, split, solved_half, open_half


def test_readme_round_numerals_match_the_banked_chain():
    highest, split, _, _ = _banked()
    printed = {int(n) for n in ROUND_NUMERAL.findall(README.read_text())}
    assert printed == {highest, split}, (
        f"README prints round numerals {sorted(printed)}; the banked chain "
        f"supports only {sorted({highest, split})} "
        f"(round {highest} solved contiguously, round {split} split)"
    )


def test_readme_names_both_halves_of_the_split_round():
    _, split, solved_half, open_half = _banked()
    text = README.read_text()
    assert solved_half in text, (
        f"README does not name {solved_half}, the solved half of round {split}"
    )
    assert open_half in text, (
        f"README does not name {open_half}, the half of round {split} still open"
    )


def test_receipt_matches_the_banked_chain():
    highest, split, solved_half, open_half = _banked()
    assert RECEIPT.is_file(), f"missing in-repo receipt {RECEIPT.relative_to(REPO)}"
    text = RECEIPT.read_text()
    for token in (f"round {highest}", solved_half, open_half):
        assert token in text, (
            f"receipt {RECEIPT.name} does not state {token!r}, which the banked "
            "chain does"
        )


def test_readme_cites_the_in_repo_receipt():
    """Runs on a clean clone: no gitignored path is read."""
    assert RECEIPT.is_file(), f"missing in-repo receipt {RECEIPT.relative_to(REPO)}"
    assert RECEIPT.name in README.read_text(), (
        f"README's Bubble Bobble round claim does not cite {RECEIPT.name}, so a "
        "clone has no evidence for the number"
    )
