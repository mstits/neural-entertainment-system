"""README library-compatibility claims must agree with each other and a receipt.

README once carried two library-compatibility figures at the same time: a
"~99.9%" that traced to `reports/full_library.md` (a 2026-04-27 scan over 794
ROMs whose `ok` only meant "no panic in 300 frames") and the live-screen census
figure of 793 of 796. Same numerator by coincidence, different denominator,
different definition, four months apart. These tests pin the numbers together so
they cannot drift again:

* every `N of 796` / `N/796` claim in README uses the same N;
* every library-compatibility percentage in README is that N over 796, rounded;
* the newest census receipt's bolded RESULT states that same N, and README
  cites that receipt by path;
* the superseded 794-ROM scan keeps its banner, and README does not point
  readers at it.

The receipt check reads a receipt's **result**, not any `N of 796` anywhere in
its prose. A census receipt narrates the count it superseded as well as the one
it establishes, so a plain substring search over the file would let a
coordinated drift to a historical count pass. The convention these receipts
follow, and this test enforces, is that the count a receipt establishes is
written in bold: `**793 of 796**`.

The asm-core hit rate (`99.97%`) is a different measurement (the AArch64
assembly 6502 core against the pure-Rust reference, not the ROM library) and is
allowlisted here by name.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
CENSUS_RECEIPT_DIR = REPO / "docs" / "receipts" / "rom_census"
SUPERSEDED_SCAN = REPO / "reports" / "full_library.md"

LIBRARY_TOTAL = 796

# Percentages in README that are not library-compatibility figures. Each entry
# is (literal, why it is not a compatibility claim).
NON_COMPAT_PERCENTAGES = {
    "99.97%": "asm 6502 core hit rate against the pure-Rust reference",
}

# `793 of 796`, `793/796`, `793 of 796 ROMs`, `(793/796 ROMs ...)`.
NUMERATOR_RE = re.compile(r"(\d{3})\s*(?:of|/)\s*%d\b" % LIBRARY_TOTAL)
# Any percentage with one or more decimals, e.g. `99.6%`, `99.623%`.
PERCENT_RE = re.compile(r"\b(\d{2}\.\d+)%")
# A receipt's RESULT line: the count it establishes, written in bold.
RECEIPT_RESULT_RE = re.compile(r"\*\*(\d{3}) of %d\*\*" % LIBRARY_TOTAL)
# The census date a receipt's filename carries, e.g. `..._2026-09-02.md`.
RECEIPT_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _claimed_numerators(text: str) -> list[int]:
    return [int(m) for m in NUMERATOR_RE.findall(text)]


def _current_census() -> tuple[Path, int]:
    """The newest census receipt that states a result, and the count it states.

    Newest by the date in the filename; a tie is only a tie if the tied
    receipts agree, which is asserted rather than assumed.
    """
    dated: list[tuple[str, Path, int]] = []
    for receipt in sorted(CENSUS_RECEIPT_DIR.glob("*.md")):
        date = RECEIPT_DATE_RE.search(receipt.name)
        result = RECEIPT_RESULT_RE.search(receipt.read_text(encoding="utf-8"))
        if date and result:
            dated.append((date.group(1), receipt, int(result.group(1))))

    assert dated, (
        f"no receipt under {CENSUS_RECEIPT_DIR} states a bolded "
        f"`**N of {LIBRARY_TOTAL}**` result; the README's public "
        "compatibility number has nothing behind it"
    )
    newest = max(d for d, _, _ in dated)
    latest = [(r, n) for d, r, n in dated if d == newest]
    counts = {n for _, n in latest}
    assert len(counts) == 1, (
        f"census receipts dated {newest} disagree on the library boot count: "
        + ", ".join(f"{r.name} says {n}" for r, n in sorted(latest))
    )
    return latest[0][0], latest[0][1]


def test_readme_states_one_library_boot_numerator() -> None:
    found = _claimed_numerators(_readme())
    assert found, "README states no `N of 796` library-compatibility claim"
    assert len(set(found)) == 1, (
        f"README states conflicting library boot counts over {LIBRARY_TOTAL}: "
        f"{sorted(set(found))}"
    )


def test_readme_compat_percentages_match_the_boot_count() -> None:
    text = _readme()
    numerators = _claimed_numerators(text)
    assert numerators, "README states no `N of 796` library-compatibility claim"
    passing = numerators[0]

    exact = passing / LIBRARY_TOTAL * 100.0
    allowed = {
        f"{round(exact, places):.{places}f}" for places in range(1, 4)
    } | {p.rstrip("%") for p in NON_COMPAT_PERCENTAGES}

    # Only percentages high enough to be a plausible compatibility claim; the
    # README's many low percentages (win rates, overheads) are not in scope.
    suspect = {p for p in PERCENT_RE.findall(text) if float(p) >= 90.0}
    stale = sorted(p for p in suspect if p not in allowed)
    assert not stale, (
        f"README carries percentage(s) {stale} that are neither "
        f"{passing}/{LIBRARY_TOTAL} rounded nor a known non-compatibility "
        f"figure ({sorted(NON_COMPAT_PERCENTAGES)}). Either the boot count "
        "moved without every site moving with it, or a new measurement needs "
        "an entry in NON_COMPAT_PERCENTAGES."
    )


def test_readme_boot_count_is_backed_by_a_census_receipt() -> None:
    numerators = _claimed_numerators(_readme())
    assert numerators, "README states no `N of 796` library-compatibility claim"
    receipt, counted = _current_census()
    assert numerators[0] == counted, (
        f"README claims {numerators[0]} of {LIBRARY_TOTAL} but the current "
        f"census receipt {receipt.name} states {counted} of {LIBRARY_TOTAL}. "
        "A public compatibility number moves when a receipt moves it, not "
        "before."
    )


def test_readme_cites_the_receipt_that_backs_the_count() -> None:
    text = _readme()
    receipt, _ = _current_census()
    assert str(receipt.relative_to(REPO)) in text, (
        f"README does not cite {receipt.relative_to(REPO)}, the current "
        "census receipt; a public compatibility number needs its receipt "
        "named where a reader can find it"
    )


def test_the_superseded_scan_keeps_its_banner() -> None:
    """The 794-ROM scan may stay in the tree, but not without its warning.

    Its `ok` meant "no panic in 300 frames" over a 794-ROM library, which is
    neither the current metric nor the current denominator. Deleting the banner
    is what makes it quotable again, so the banner is what this pins.
    """
    if not SUPERSEDED_SCAN.exists():
        pytest.skip(f"{SUPERSEDED_SCAN.relative_to(REPO)} no longer in the tree")
    scan = SUPERSEDED_SCAN.read_text(encoding="utf-8")
    if "794" not in scan:
        return  # regenerated against the current library; nothing to warn about
    head = "\n".join(scan.splitlines()[:15]).upper()
    assert "SUPERSEDED" in head, (
        f"{SUPERSEDED_SCAN.relative_to(REPO)} still reports a 794-ROM library "
        "but no longer opens with a SUPERSEDED banner, so its numbers read as "
        "current compatibility again"
    )


def test_readme_does_not_point_readers_at_a_superseded_scan() -> None:
    if not SUPERSEDED_SCAN.exists():
        pytest.skip(f"{SUPERSEDED_SCAN.relative_to(REPO)} no longer in the tree")
    scan = SUPERSEDED_SCAN.read_text(encoding="utf-8")
    if "SUPERSEDED" not in scan.upper():
        return  # a regenerated, current scan may be cited freely
    assert str(SUPERSEDED_SCAN.relative_to(REPO)) not in _readme(), (
        f"{SUPERSEDED_SCAN.relative_to(REPO)} is marked superseded but README "
        "still sends readers to it for compatibility numbers"
    )


def test_docs_enumerate_every_supported_mapper() -> None:
    """README and ARCHITECTURE must list the mappers the core actually ships.

    A count label over a list that does not match it is the same defect as a
    stale number: the label said 37 while the tree drew 36.
    """
    nes_core = pytest.importorskip("nes_core")
    supported = set(nes_core.supported_mappers())

    readme = _readme()
    section = readme[readme.index("\n## Compatibility") :]
    section = section[: section.index("\n## ", 1)]
    listed = section[section.index("- Discrete logic:") : section.index("- Unsupported")]
    readme_ids = {
        int(n)
        for group in re.findall(r"\((\d[\d,\s]*)\)", listed)
        for n in group.replace(",", " ").split()
    }
    assert readme_ids == supported, (
        "README's Compatibility enumeration disagrees with "
        f"nes_core.supported_mappers(): missing {sorted(supported - readme_ids)}, "
        f"extra {sorted(readme_ids - supported)}"
    )

    arch = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    tree = arch[arch.index("## Mapper tree") :]
    tree = tree[: tree.index("```", tree.index("graph TD"))]
    arch_ids: set[int] = set()
    for line in tree.splitlines():
        if "-->" not in line:
            continue
        label = re.search(r"\[([^\]]*)\]\s*$", line.strip())
        if not label:
            continue
        trailing = re.search(
            r"((?:\d+\s*/\s*)*\d+)\s*(?:\+\s*audio)?$", label.group(1).strip()
        )
        if trailing:
            arch_ids.update(int(n) for n in re.findall(r"\d+", trailing.group(1)))
    assert arch_ids == supported, (
        "docs/ARCHITECTURE.md's mapper tree disagrees with "
        f"nes_core.supported_mappers(): missing {sorted(supported - arch_ids)}, "
        f"extra {sorted(arch_ids - supported)}"
    )
