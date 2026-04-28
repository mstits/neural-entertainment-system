"""Diff two CPU instruction traces (nes_core vs Mesen).

Trace format produced by both `nes_core_smb_coldboot.py` and
`mesen_smb_coldboot.lua`:
    PC OP A:AA X:XX Y:YY P:PP SP:SP PPU:sl,cyc CYC:total

Aligns the two traces by subtracting their initial CPU-cycle and
PPU-cycle anchors, masks the unused bit 5 of the P register
(Mesen displays it as 0, real 6502 / nes_core keep it as 1), then
walks instruction-by-instruction looking for the first PC / A / X /
Y / SP / P / CYC / PPU divergence.

Usage:
    python diff_cpu_traces.py OURS.txt MESEN.txt [--context 5] [--cyc] [--ppu]

Flags:
    --cyc   Treat CYC mismatch as a divergence (default off; off makes
            the tool find the first REGISTER divergence ignoring
            timing drift).
    --ppu   Treat PPU mismatch as a divergence (default off).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

LINE_RE = re.compile(
    r"(?P<pc>[0-9A-Fa-f]{4})\s+(?P<op>[0-9A-Fa-f]{2})\s+"
    r"A:(?P<a>[0-9A-Fa-f]{2})\s+X:(?P<x>[0-9A-Fa-f]{2})\s+"
    r"Y:(?P<y>[0-9A-Fa-f]{2})\s+P:(?P<p>[0-9A-Fa-f]{2})\s+"
    r"SP:(?P<sp>[0-9A-Fa-f]{2})\s+"
    r"PPU:\s*(?P<sl>-?\d+),\s*(?P<slc>-?\d+)\s+"
    r"CYC:(?P<cyc>\d+)"
)


def parse(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text().splitlines():
        m = LINE_RE.search(ln)
        if not m:
            continue
        g = m.groupdict()
        out.append({
            "pc": int(g["pc"], 16),
            "a": int(g["a"], 16),
            "x": int(g["x"], 16),
            "y": int(g["y"], 16),
            "sp": int(g["sp"], 16),
            "p": int(g["p"], 16) & ~0x20,  # mask unused bit-5 (display convention)
            "sl": int(g["sl"]),
            "slc": int(g["slc"]),
            "cyc": int(g["cyc"]),
        })
    return out


def fmt(e: dict) -> str:
    return (f"PC={e['pc']:04X} A={e['a']:02X} X={e['x']:02X} "
            f"Y={e['y']:02X} P={e['p']:02X} SP={e['sp']:02X} "
            f"PPU={e['sl']:>3},{e['slc']:>3} CYC={e['cyc']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ours", type=Path)
    p.add_argument("mesen", type=Path)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--cyc", action="store_true",
                   help="treat CPU-cycle mismatch as divergence")
    p.add_argument("--ppu", action="store_true",
                   help="treat PPU-state mismatch as divergence")
    args = p.parse_args()

    ours = parse(args.ours)
    mesen = parse(args.mesen)
    print(f"ours:  {len(ours)} instructions")
    print(f"mesen: {len(mesen)} instructions")

    if not ours or not mesen:
        print("empty trace")
        return 1

    # Anchor offsets — Mesen counts reset cycles (CPU cyc starts at 7,
    # PPU at ~25), nes_core / nes-py start the cycle counter at 0.
    cyc_offset = mesen[0]["cyc"] - ours[0]["cyc"]
    ppu_offset = mesen[0]["slc"] - ours[0]["slc"]
    sl_offset = mesen[0]["sl"] - ours[0]["sl"]
    print(f"alignment offsets: CYC={cyc_offset:+} PPU_dot={ppu_offset:+} "
          f"PPU_sl={sl_offset:+}")

    n = min(len(ours), len(mesen))
    first_diff = None
    diff_kind = None
    for i in range(n):
        o = ours[i]
        m = mesen[i]
        m_cyc = m["cyc"] - cyc_offset
        m_slc = m["slc"] - ppu_offset
        m_sl = m["sl"] - sl_offset

        if (o["pc"] != m["pc"] or o["a"] != m["a"] or
                o["x"] != m["x"] or o["y"] != m["y"] or
                o["sp"] != m["sp"] or o["p"] != m["p"]):
            first_diff = i
            diff_kind = "REGISTER"
            break
        if args.cyc and o["cyc"] != m_cyc:
            first_diff = i
            diff_kind = f"CYC (ours={o['cyc']} mesen-adj={m_cyc})"
            break
        if args.ppu and (o["sl"] != m_sl or o["slc"] != m_slc):
            first_diff = i
            diff_kind = (f"PPU (ours={o['sl']},{o['slc']} "
                         f"mesen-adj={m_sl},{m_slc})")
            break

    if first_diff is None:
        print(f"\nNo divergence in {n} common instructions.")
        return 0

    i = first_diff
    print(f"\nFIRST {diff_kind} DIVERGENCE at instruction index {i}:")
    lo = max(0, i - args.context)
    hi = min(n, i + args.context + 1)
    for j in range(lo, hi):
        mark = "  <-- DIVERGE" if j == i else ""
        m_adj = dict(mesen[j])
        m_adj["cyc"] -= cyc_offset
        m_adj["slc"] -= ppu_offset
        m_adj["sl"] -= sl_offset
        print(f"[{j:>7}] ours:  {fmt(ours[j])}")
        print(f"          mesen: {fmt(m_adj)}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
