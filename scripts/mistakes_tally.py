"""Derive the graduation counts from MISTAKES.md itself.

The graduation watch table was hand-maintained beside the entries rather than
derived from them, and drifted: it claimed 9 [unverified-claim] against 6 tagged,
5 [stale-artifact] against 2. That is the exact defect the engine purity sweep
named on 2026-08-27 -- "enforcement must be DERIVED from the declaration, never
listed beside it, or the list is the same defect one level up" -- committed
inside the log that records it.

Run: .venv/bin/python scripts/mistakes_tally.py [--check]
"""
import re, sys, pathlib, collections

SRC = pathlib.Path(__file__).resolve().parent.parent / "MISTAKES.md"
HDR = re.compile(r'^## (\d{4}-\d{2}-\d{2}|\(recurring[^)]*\))\s+—\s+(?:\[([a-z-]+)\]\s+)?(.+)$')

REPO = SRC.parent
ROW = re.compile(r'^\|\s+`\[([a-z-]+)\]`\s+\|[^|]*\|\s+(.+?)\s+\|\s*$')
TOKEN = re.compile(r'`([^`]+)`')
ENFORCED = re.compile(r'\bSHIPPED\b|\bPROMOTED\b')
ROOTS = ("scripts", "src", "tests", "nes_core/src")

def _sources():
    for root in ROOTS:
        base = REPO / root
        for f in (base.rglob("*") if base.is_dir() else ()):
            if f.is_file() and f.suffix in (".py", ".rs"):
                yield f

def _defines(sym):
    """(int, path) if sym is a module-level int constant; (None, path) if it
    merely appears; (None, None) if nowhere in the tree."""
    assign = re.compile(rf'^{re.escape(sym)}\s*=\s*(\d+)\s*$', re.M)
    seen = None
    for f in _sources():
        text = f.read_text(errors="ignore")
        if (m := assign.search(text)):
            return int(m.group(1)), str(f.relative_to(REPO))
        seen = seen or (str(f.relative_to(REPO)) if sym in text else None)
    return None, seen

def _quoted_value(tok, cell):
    """The single number the cell attributes to `tok`: the first integer after
    the symbol's own mention.

    Rule B used to be set membership over every integer anywhere in the cell,
    so a cell that also recorded a superseded value ("at iter 40 (raised from
    25 ...)") accepted both, and a regression of the constant back to 25
    passed --check. Anchoring on the first integer after the token makes it
    one number against one number, in the direction the cell reads. Returns an
    empty set when no integer follows the token; the caller then falls back to
    the whole-cell set, which is the older, looser check rather than none.
    """
    parts = cell.split(f"`{tok}`", 1)
    if len(parts) != 2:
        return set()
    m = re.search(r'\b(\d+)\b', parts[1])
    return {int(m.group(1))} if m else set()

def _resolve(tok):
    if tok.startswith("make "):
        t = tok[5:].strip()
        mk = (REPO / "Makefile").read_text()
        return bool(re.search(rf'^{re.escape(t)}:', mk, re.M)), f"Makefile target `{t}:`"
    if tok.endswith((".py", ".rs", ".sh", ".md")):
        return (REPO / tok).exists(), f"path {tok}"
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', tok):
        return _defines(tok)[1] is not None, f"symbol {tok}"
    return None, tok            # prose; not checkable, not checked

def check_enforcement(text):
    """A SHIPPED/PROMOTED cell names only things that exist (rule A), and the
    number it attributes to an integer constant it names is that constant's
    current value (rule B).

    Rule B reads one number, not a set: the first integer after the symbol's
    own mention (see _quoted_value). A cell is free to record superseded
    values as history alongside it without blinding the check to a regression
    back to one of them.

    The project-instruction-file rule (a cell citing "(project instruction
    file)") runs only when CLAUDE.md exists in this checkout; on a checkout
    without it (the file is gitignored), the rule is skipped with a printed
    note rather than failing, so `make test` stays green on a clean clone.
    """
    problems = []
    for n, line in enumerate(text.splitlines(), 1):
        m = ROW.match(line)
        if not m or not ENFORCED.search(m.group(2)):
            continue
        cat, cell = m.group(1), m.group(2)
        if "(project instruction file)" in cell:
            rules = REPO / "CLAUDE.md"
            if not rules.exists():
                print(f"note: CLAUDE.md absent, skipping project-instruction-file "
                      f"check for MISTAKES.md:{n}")
                continue
            if "Enforced invariants" not in rules.read_text():
                problems.append((n, cat, "cites the project instruction file, "
                                        "which has no 'Enforced invariants' section"))
        nums = {int(x) for x in re.findall(r'\b(\d+)\b', cell)}
        for tok in TOKEN.findall(cell):
            ok, what = _resolve(tok)
            if ok is None:
                continue
            if not ok:
                problems.append((n, cat, f"names {what}, which does not exist"))
                continue
            val, where = _defines(tok)
            quoted = _quoted_value(tok, cell) or nums
            if val is not None and quoted and val not in quoted:
                problems.append((n, cat, f"`{tok}` = {val} in {where}, "
                                         f"but the cell quotes {sorted(quoted)}"))
    return problems

def tally(text):
    counts, untagged = collections.Counter(), []
    for line in text.splitlines():
        m = HDR.match(line)
        if not m:
            continue
        cat, title = m.group(2), m.group(3)
        if cat:
            counts[cat] += 1
        else:
            untagged.append(title)
    return counts, untagged

def main():
    counts, untagged = tally(SRC.read_text())
    total = sum(counts.values()) + len(untagged)
    print(f"entries: {total}  tagged: {sum(counts.values())}  untagged: {len(untagged)}")
    for cat, n in counts.most_common():
        flag = "  <-- AT/PAST THRESHOLD (4)" if n >= 4 else ""
        print(f"  {cat:22s} {n}{flag}")
    if untagged:
        print(f"\n{len(untagged)} untagged entries (pre-date the tag convention):")
        for t in untagged[:20]:
            print(f"  - {t[:72]}")
    if "--check" in sys.argv:
        # Match bold (**N**, at/past threshold) AND plain (N) counts — the
        # bold-only version left every sub-threshold row invisible to the
        # drift check, so a stale "3" could sit beside 5 real entries
        # forever without --check ever failing (found by external audit,
        # 2026-08-28).
        stated = re.findall(
            r'\|\s+`\[([a-z-]+)\]`[^|]*\|\s+(?:\*\*(\d+)\*\*|(\d+))\s+\|',
            SRC.read_text())
        stated = [(c, b or p) for c, b, p in stated]
        bad = [(c, int(n), counts[c]) for c, n in stated if counts[c] != int(n)]
        if bad:
            print("\nDRIFT — watch table disagrees with the entries:")
            for c, said, real in bad:
                print(f"  [{c}] table says {said}, entries show {real}")
            return 1
        problems = check_enforcement(SRC.read_text())
        if problems:
            print("\nDRIFT: an enforcement cell names something that is not true:")
            for n, cat, msg in problems:
                print(f"  MISTAKES.md:{n} [{cat}] {msg}")
            return 1
        print("\nwatch table matches the entries.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
