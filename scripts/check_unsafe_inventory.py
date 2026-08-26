"""Unsafe-inventory gate for `nes_core/SECURITY.md`.

`SECURITY.md`'s per-file `unsafe` breakdown went stale for ~4 months
while the crate grew from 3 call sites to 155 across 11 files — the
document that exists specifically to track this drifted out from under
itself, silently, because nothing mechanical ever re-counted it. This
script is that mechanical re-count.

It does three things, failing loud on any violation:

  1. Counts `unsafe`-matching *lines* per `.rs` file under
     `nes_core/src` — the exact `grep -c unsafe <file>` semantics
     `SECURITY.md` documents itself as using (a line with the substring
     "unsafe" twice, e.g. `unsafe_op_in_unsafe_fn`, still counts once).
  2. Parses `SECURITY.md`'s per-file inventory — the `### N. `src/x.rs`
     — ... (COUNT)` headings under "## `unsafe` surface" — tolerantly:
     numbering, dash style, and trailing qualifier text inside the
     parens (e.g. "(5, 2 of them real)") are all allowed to vary,
     because the heading *layout* is not the thing being audited, the
     *numbers* are.
  3. Diffs the two, failing on:
       - a file present in the tree with `unsafe` but absent from the
         doc's inventory entirely;
       - a file listed in the doc whose file no longer exists in the
         tree (a stale entry left behind by a rename/delete);
       - a file present in both whose count has drifted by more than
         `TOLERANCE` lines in either direction.

If `SECURITY.md` still carries the old "three call sites in the entire
crate" placeholder prose instead of a per-file inventory, this script
fails loudly with a distinct, specific message rather than silently
treating every tracked file as "missing from the doc" — the two
failure modes are different problems and should not be confused at a
glance.

Run directly:

    python scripts/check_unsafe_inventory.py

or via `make unsafe-inventory-check`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NES_CORE = REPO / "nes_core"
SECURITY_MD = NES_CORE / "SECURITY.md"

# How many lines a per-file count may drift from the doc before this
# gate fails. Small on purpose: the whole point is catching drift while
# it is still small, not waiting for it to become "155 vs 3" again. A
# couple of lines of slack absorbs the doc rounding a qualifier
# ("5, 2 of them real") differently than a future re-count would, or a
# same-day edit landing a line or two off the last recount, without
# opening the door to the multi-month, multi-hundred-line silent drift
# this checker exists to prevent.
TOLERANCE = 2

STALE_PLACEHOLDER_MARKER = "three call sites in the entire crate"

# One inventory heading line, tolerant of formatting variation:
#   ### 1. `src/pool.rs` — worker-pool concurrency + NEON unpack (75)
#   ### `src/lib.rs` -- crate-wide lint allow (2)
#   #### 6. `src/preprocess.rs` — NEON SIMD intrinsics (5, 2 of them real)
# Required: a `##`+ heading, a backtick-quoted path ending in `.rs`, and
# a parenthesized count as the LAST thing on the line (optionally
# followed by ", <qualifier text>" before the closing paren). Anything
# else about the heading — numbering, the description prose, the dash
# glyph — is free to vary.
HEADING_RE = re.compile(
    r'^#{2,6}\s*(?:\d+[.)]\s*)?`([^`]+\.rs)`.*?\((\d+)(?:\s*,[^)]*)?\)\s*$'
)

# The bolded crate-wide summary line, e.g.:
#   **155 lines match `unsafe` across 11 files** (`grep -rn unsafe ...`
# Used only as a bonus cross-check when present; its absence is not
# itself a failure; some future rewording of the doc may not restate a
# totals line at all.
SUMMARY_RE = re.compile(
    r'\*\*(\d+)\s+lines?\s+match\s+`unsafe`\s+across\s+(\d+)\s+files?\*\*'
)


def actual_unsafe_counts(src_root: Path) -> dict[str, int]:
    """Per-file `unsafe`-matching line counts under `src_root`.

    Keys are paths relative to `src_root`'s parent (`nes_core`), e.g.
    `"src/pool.rs"`, `"src/mapper/mapper1.rs"` — matching the form
    `SECURITY.md` writes them in. Only files with at least one match
    are included, mirroring `grep -rln unsafe src` (a file with zero
    hits has nothing to list).
    """
    counts: dict[str, int] = {}
    if not src_root.exists():
        return counts
    for f in sorted(src_root.rglob("*.rs")):
        try:
            lines = f.read_text(errors="replace").split("\n")
        except OSError:
            continue
        n = sum(1 for ln in lines if "unsafe" in ln)
        if n > 0:
            rel = str(f.relative_to(src_root.parent))
            counts[rel] = n
    return counts


def is_stale_placeholder(text: str) -> bool:
    """True if `SECURITY.md` still shows the old pre-refresh prose
    ("three call sites in the entire crate") with no per-file inventory
    headings — i.e. the sibling content refresh has not landed yet.
    """
    if STALE_PLACEHOLDER_MARKER not in text:
        return False
    return not HEADING_RE.search(text) and not any(
        HEADING_RE.match(ln) for ln in text.split("\n")
    )


def parse_security_md_inventory(text: str) -> dict[str, tuple[int, int]]:
    """Per-file (claimed_count, doc_line_number) from the `### ... (N)`
    headings in `SECURITY.md`. `doc_line_number` is 1-indexed, for
    error messages that point somewhere useful.
    """
    inventory: dict[str, tuple[int, int]] = {}
    for i, line in enumerate(text.split("\n"), start=1):
        m = HEADING_RE.match(line.strip())
        if not m:
            continue
        path, count = m.group(1), int(m.group(2))
        # Normalize: headings write paths as `src/x.rs` (relative to
        # nes_core/); tolerate an accidental `nes_core/src/x.rs` too.
        path = path.removeprefix("nes_core/")
        inventory[path] = (count, i)
    return inventory


def parse_summary_totals(text: str) -> tuple[int, int] | None:
    m = SUMMARY_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def compare(
    actual: dict[str, int],
    claimed: dict[str, tuple[int, int]],
    tolerance: int = TOLERANCE,
) -> list[str]:
    errors: list[str] = []

    for path, count in sorted(actual.items()):
        if path not in claimed:
            errors.append(
                f"{path}: {count} unsafe-matching line(s) in the tree, "
                f"but this file has no entry at all in SECURITY.md's "
                f"inventory")
            continue
        claimed_count, doc_line = claimed[path]
        drift = abs(count - claimed_count)
        if drift > tolerance:
            errors.append(
                f"{path}: SECURITY.md:{doc_line} claims {claimed_count}, "
                f"tree actually has {count} (drift {drift} > "
                f"tolerance {tolerance})")

    for path, (claimed_count, doc_line) in sorted(claimed.items()):
        if path not in actual:
            errors.append(
                f"{path}: SECURITY.md:{doc_line} lists {claimed_count} "
                f"unsafe-matching line(s), but this file no longer "
                f"exists (or no longer contains \"unsafe\") in the tree "
                f"— stale inventory entry")

    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tolerance", type=int, default=TOLERANCE,
        help=f"max allowed per-file count drift (default: {TOLERANCE})")
    args = ap.parse_args(argv)

    if not SECURITY_MD.exists():
        print(f"UNSAFE INVENTORY CHECK FAILED: {SECURITY_MD} does not exist")
        return 1

    text = SECURITY_MD.read_text()

    if is_stale_placeholder(text):
        print(
            "UNSAFE INVENTORY CHECK FAILED: nes_core/SECURITY.md still "
            "shows the stale pre-refresh placeholder text "
            f"({STALE_PLACEHOLDER_MARKER!r}) with no per-file `### ... "
            "(N)` inventory headings under \"## `unsafe` surface\". This "
            "is not a count mismatch — the refreshed inventory has not "
            "landed in this file yet. Once it does, re-run this check.")
        return 1

    actual = actual_unsafe_counts(NES_CORE / "src")
    claimed = parse_security_md_inventory(text)

    if not claimed:
        print(
            "UNSAFE INVENTORY CHECK FAILED: found no `### ... (N)` "
            "per-file inventory headings in nes_core/SECURITY.md at "
            "all (looked under \"## `unsafe` surface\"). Either the "
            "section was renamed/restructured in a way this parser no "
            "longer recognizes, or the inventory is missing.")
        return 1

    errors = compare(actual, claimed, args.tolerance)

    totals = parse_summary_totals(text)
    if totals is not None:
        summary_lines, summary_files = totals
        real_lines, real_files = sum(actual.values()), len(actual)
        if summary_lines != real_lines or summary_files != real_files:
            errors.append(
                f"crate-wide summary line claims {summary_lines} lines "
                f"across {summary_files} files, but the tree actually "
                f"has {real_lines} lines across {real_files} files")

    if errors:
        print("UNSAFE INVENTORY CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(
            f"\n{len(errors)} problem(s) across {len(actual)} file(s) "
            f"with unsafe in the tree, {len(claimed)} entries in "
            f"SECURITY.md's inventory. Re-run the count yourself: "
            f"grep -rn unsafe nes_core/src")
        return 1

    print(
        f"unsafe inventory check OK: {len(actual)} file(s), "
        f"{sum(actual.values())} unsafe-matching line(s), all present "
        f"and within tolerance ({args.tolerance}) of SECURITY.md's "
        f"inventory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
