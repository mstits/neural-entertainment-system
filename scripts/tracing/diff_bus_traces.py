"""Diff two bus-access traces from `capture_bus_traces.py`.

Finds the FIRST line where the (r/w, addr, value) tuple differs between
nes_core (ours) and nes-py (theirs), printing context around that line.
Ignores differences in cpu_cycle / scanline / slc (timing-sensitive,
expected to differ by a few cycles).

Usage:
  python scripts/tracing/diff_bus_traces.py OURS_TRACE THEIRS_TRACE \\
        [--frame-offset 1] [--mmio-only] [--exclude-prg-reads]

--frame-offset lets you align frame numbers. Phase-comp means nes-py's
frame counter is 1 ahead of ours, so pass `--frame-offset 1` to subtract
1 from theirs' frames before comparing.

--mmio-only filters to addresses $2000-$401F (PPU/APU/controller MMIO).
Useful when ours' `cpu.try_bulk_step` skips some opcode-byte bus reads,
mis-aligning PRG-ROM-read positions between the two traces.

--exclude-prg-reads drops $8000-$FFFF reads (opcode/operand fetches from
PRG). Keeps RAM accesses + MMIO. Same bulk_step alignment concern.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

# Format emitted by both instrumentations:
#   Ours:   [BUS f=<frame> sl=<sl> slc=<slc> R|W $<addr> =0x<val>]
#   Theirs: [BUS f=<frame> c=<cyc> sl=<sl> slc=<slc> R|W $<addr> =0x<val>]
OURS_RE = re.compile(
    r"\[BUS f=(?P<f>\d+) sl=(?P<sl>\d+) slc=(?P<slc>\d+) "
    r"(?P<rw>[RW]) \$(?P<addr>[0-9A-Fa-f]+) =0x(?P<val>[0-9A-Fa-f]+)\]"
)
THEIRS_RE = re.compile(
    r"\[BUS f=(?P<f>\d+) c=(?P<c>\d+) sl=(?P<sl>\d+) slc=(?P<slc>\d+) "
    r"(?P<rw>[RW]) \$(?P<addr>[0-9A-Fa-f]+) =0x(?P<val>[0-9A-Fa-f]+)\]"
)


def parse(path: Path, regex: re.Pattern, frame_adj: int = 0) -> list[dict]:
    out = []
    for ln in path.read_text().splitlines():
        m = regex.search(ln)
        if not m:
            continue
        d = m.groupdict()
        d["f"] = int(d["f"]) + frame_adj
        d["sl"] = int(d["sl"])
        d["slc"] = int(d["slc"])
        d["addr"] = int(d["addr"], 16)
        d["val"] = int(d["val"], 16)
        out.append(d)
    return out


def fmt(e: dict) -> str:
    return (
        f"f={e['f']} sl={e['sl']} slc={e['slc']} "
        f"{e['rw']} ${e['addr']:04X} =0x{e['val']:02X}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ours", type=Path)
    p.add_argument("theirs", type=Path)
    p.add_argument("--frame-offset", type=int, default=-1,
                   help="Adjust theirs' frame numbers by this amount "
                        "before comparing (default -1 to account for "
                        "nes-py's phase-comp extra step)")
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--mmio-only", action="store_true",
                   help="Only compare MMIO accesses ($2000-$401F). "
                        "Filters out PRG/RAM reads that may mis-align "
                        "due to bulk_step skipping opcode-byte reads.")
    p.add_argument("--exclude-prg-reads", action="store_true",
                   help="Drop reads from $8000-$FFFF (PRG ROM). "
                        "Similar alignment concern as --mmio-only but "
                        "keeps RAM accesses.")
    args = p.parse_args()

    ours = parse(args.ours, OURS_RE)
    theirs = parse(args.theirs, THEIRS_RE, args.frame_offset)

    def filter_ops(ops: list[dict]) -> list[dict]:
        out = ops
        if args.mmio_only:
            out = [e for e in out if 0x2000 <= e["addr"] <= 0x401F]
        elif args.exclude_prg_reads:
            out = [e for e in out
                   if not (e["rw"] == "R" and e["addr"] >= 0x8000)]
        return out

    ours = filter_ops(ours)
    theirs = filter_ops(theirs)
    print(f"ours:   {len(ours)} bus ops")
    print(f"theirs: {len(theirs)} bus ops")

    # Find first frame present in both after frame alignment.
    ours_frames = sorted({e["f"] for e in ours})
    theirs_frames = sorted({e["f"] for e in theirs})
    common_frames = sorted(set(ours_frames) & set(theirs_frames))
    if not common_frames:
        print("No common frames between traces; nothing to diff.")
        return 1
    print(f"common frames: {common_frames[:4]}...{common_frames[-2:]}")

    # Only compare bus ops within common frames.
    fmin = common_frames[0]
    fmax = common_frames[-1]
    ours_in = [e for e in ours if fmin <= e["f"] <= fmax]
    theirs_in = [e for e in theirs if fmin <= e["f"] <= fmax]
    print(f"in range: ours={len(ours_in)}, theirs={len(theirs_in)}")

    # Walk both sequences and find first semantic divergence.
    n = min(len(ours_in), len(theirs_in))
    first_diff = None
    for i in range(n):
        a = ours_in[i]
        b = theirs_in[i]
        if (a["rw"] != b["rw"] or a["addr"] != b["addr"]
                or a["val"] != b["val"]):
            first_diff = i
            break

    if first_diff is None:
        if len(ours_in) == len(theirs_in):
            print("No semantic divergence in common-frame bus ops.")
        else:
            print(f"Sequences match for first {n} ops, but lengths differ "
                  f"(ours={len(ours_in)}, theirs={len(theirs_in)}).")
        return 0

    i = first_diff
    print(f"\nFIRST SEMANTIC DIVERGENCE at bus-op index {i}:")
    lo = max(0, i - args.context)
    hi = min(n, i + args.context + 1)
    for j in range(lo, hi):
        mark = " <-- DIVERGE" if j == i else ""
        print(f"  [{j:>6}] ours:   {fmt(ours_in[j])}")
        print(f"           theirs: {fmt(theirs_in[j])}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
