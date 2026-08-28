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
        stated = re.findall(r'\|\s+`\[([a-z-]+)\]`[^|]*\|\s+\*\*(\d+)\*\*', SRC.read_text())
        bad = [(c, int(n), counts[c]) for c, n in stated if counts[c] != int(n)]
        if bad:
            print("\nDRIFT — watch table disagrees with the entries:")
            for c, said, real in bad:
                print(f"  [{c}] table says {said}, entries show {real}")
            return 1
        print("\nwatch table matches the entries.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
