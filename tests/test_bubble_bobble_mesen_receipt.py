"""The Bubble Bobble Mesen cross-check: its receipts, and the docs that quote it.

Three gates, each guarding a different way this claim can rot.

1. **The receipt set is whole and unaltered.** `docs/receipts/parity/
   bubble_bobble_mesen_2026-09-01/` carries its own manifest,
   `SHA256SUMS.ruling`, listing all 30 measurement files by bare filename.
   Thirteen of those files end in `.log`, which the repo's `*.log` glob
   ignores; they are tracked only because `.gitignore` re-includes
   `docs/receipts/**/*.log`. Drop that negation and the thirteen never
   reach a clone, so this test fails on a fresh checkout rather than the
   claim quietly losing its evidence. The manifest lives beside the files
   it names, not one directory above, so `shasum -c` works from inside the
   directory with no path rewriting.

2. **README and CLAIMS.md move together.** The README fidelity paragraph
   names two banked tapes because CLAIMS.md banks the second one. Either
   both say so or neither does; a README that claims two tapes with no
   ledger entry behind it is the failure this pins.

3. **Every public number is read off a receipt.** The frame counts, the
   lives agreement, and the RAM percentiles quoted in README.md, CLAIMS.md
   and the parity coverage map are matched against the literal text of the
   log they came from. Edit a number in a doc without re-measuring and this
   fails naming the number.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "docs" / "receipts" / "parity" / "bubble_bobble_mesen_2026-09-01"
MANIFEST = RECEIPTS / "SHA256SUMS.ruling"
CLAIMS = REPO / "CLAIMS.md"
README = REPO / "README.md"
COVERAGE_MAP = REPO / "docs" / "proposals" / "parity_coverage_map.md"

CLAIMS_HEADING = "## BUBBLE BOBBLE MESEN CROSS-CHECK 2026-09-01: HELD"

# Files in the receipt directory that the manifest deliberately does not
# cover: the manifest cannot hash itself, and the ruling prose is not a
# measurement.
UNMANIFESTED = {"SHA256SUMS.ruling", "RULING.md"}


def _manifest_rows() -> list[tuple[str, str]]:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        rows.append((digest.strip(), name.strip()))
    return rows


def test_receipt_manifest_is_complete_and_every_file_hashes():
    """All 30 measurement files present, correct, and none unaccounted for."""
    assert MANIFEST.is_file(), f"manifest missing: {MANIFEST}"
    rows = _manifest_rows()
    assert len(rows) == 30, f"manifest lists {len(rows)} files, expected 30"

    missing, wrong = [], []
    for digest, name in rows:
        path = RECEIPTS / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            wrong.append(f"{name}: {actual[:16]} != manifest {digest[:16]}")
    assert not missing, (
        "receipt files absent from the tree. The 13 '.log' files here are "
        "tracked only via the '!docs/receipts/**/*.log' negation in "
        f".gitignore; check it is still there. Missing: {missing}")
    assert not wrong, f"receipt contents changed: {wrong}"

    on_disk = {p.name for p in RECEIPTS.iterdir() if p.is_file()}
    unlisted = on_disk - {n for _, n in rows} - UNMANIFESTED
    assert not unlisted, (
        f"files in the receipt directory that no manifest line covers: "
        f"{sorted(unlisted)}")


def test_readme_names_two_tapes_exactly_when_claims_banks_the_second():
    """The README fidelity sentence and the CLAIMS entry are one claim."""
    claims = CLAIMS.read_text()
    readme = README.read_text()

    banked = CLAIMS_HEADING in claims
    names_smb = "12,000 frames" in readme
    names_bb = "6,964 frames" in readme
    names_two_tapes = "two banked tapes" in readme

    assert banked == (names_smb and names_bb and names_two_tapes), (
        "README and CLAIMS.md disagree about how many tapes are banked: "
        f"CLAIMS entry present={banked}, README names SMB={names_smb}, "
        f"Bubble Bobble={names_bb}, 'two banked tapes'={names_two_tapes}")


@pytest.mark.parametrize("number,receipt,literal", [
    # lives agreement, from the corrected +2 mapping
    ("6,964/6,964", "corrected_pair_stats.log", "lives agree 6964/6964"),
    # RAM percentiles, from the one-frame-tolerant pass
    ("median 24", "corrected_pair_stats_part2.log", "median=24"),
    ("p90 34", "corrected_pair_stats_part2.log", "p90=34"),
    ("max 135", "corrected_pair_stats_part2.log", "max=135"),
    # nes_core round-clear frames
    ("3112", "nescore_fs1_corrected.log", "raw_frame= 3112 round 1->2"),
    ("5186", "nescore_fs1_corrected.log", "raw_frame= 5186 round 2->3"),
    ("6961", "nescore_fs1_corrected.log", "raw_frame= 6961 round 3->4"),
    # the withdrawal of the frame_skip finding
    ("1,287 rows", "nescore_fs1_corrected.log",
     "BYTE-EXACT on all 1287 rows"),
])
def test_public_number_appears_in_its_receipt(number, receipt, literal):
    """Each number the docs publish is literally in the log it came from."""
    text = (RECEIPTS / receipt).read_text(errors="replace")
    assert literal in text, (
        f"the public number {number!r} cites {receipt}, but that receipt "
        f"does not contain {literal!r}")


def test_the_docs_that_quote_those_numbers_still_quote_them():
    """The three published surfaces carry the numbers the receipts back."""
    claims = CLAIMS.read_text()
    coverage = COVERAGE_MAP.read_text()
    readme = README.read_text()

    for fragment in ("3112 / 5186 / 6961", "6,964/6,964",
                     "median 24 bytes per frame", "1,287 rows"):
        assert fragment in claims, f"CLAIMS.md no longer states {fragment!r}"

    assert "median 24 / p90 34 / max 135" in coverage, (
        "the parity coverage map's Bubble Bobble row lost its percentiles")
    assert "Bubble Bobble (USA)" in coverage

    # Line wrapping is not part of the claim, so compare on one line.
    flat = re.sub(r"\s+", " ", readme)
    assert "median of 17 (SMB) and 24 (Bubble Bobble) per frame" in flat, (
        "the README fidelity sentence lost its per-tape medians")


def test_the_withdrawn_report_is_named_as_withdrawn():
    """The superseded NOT HELD result is retracted where the claim lives."""
    claims = CLAIMS.read_text()
    entry = claims.split(CLAIMS_HEADING, 1)
    assert len(entry) == 2, f"CLAIMS.md is missing {CLAIMS_HEADING!r}"
    body = entry[1]
    assert "mesen-parity-second-game.md" in body, (
        "the CLAIMS entry does not name the report it withdraws")
    assert "withdrawn" in body, (
        "the CLAIMS entry does not say the earlier NOT HELD is withdrawn")
    assert "4,032" in body, (
        "the CLAIMS entry no longer states what the withdrawn run reported")
