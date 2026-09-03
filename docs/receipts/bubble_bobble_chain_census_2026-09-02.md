# Bubble Bobble chain census (2026-09-02)

Receipt for the Bubble Bobble round count printed in `README.md`. The chains
themselves live under `runs/bubble_bobble/`, which is gitignored, so this file
is the copy a clone can read.

## What is banked

- **Rounds 1 through 98: solved, contiguous, no gaps.** 98 whole rounds, every
  one of them carrying a `status: "solved"` row.
- **Round 99 is split.** `99-0` is solved. `99-1`, the boss room, is open: it
  is recorded as `stall` in two separate attempts (`chain_day2g`,
  `chain_day2h_item`).
- The highest whole round solved is therefore **round 98**, and the campaign
  front is the round-99 boss.

The figure "round 60" that this receipt replaces was never supported by the
chain it cited: `chain_day2c` ends at round 67.

## Per-chain census

| chain | rows | solved | first | last | not solved | sha256 of `chain.jsonl` |
|---|---:|---:|---|---|---|---|
| `chain_overnight` | 21 | 21 | 1 | 21 | none | `5f04e96feaec502c409d8476a16ce1bfc9b68d9e7e7ea602b2c40f502d71c603` |
| `chain_day2` | 30 | 30 | 22 | 51 | none | `3e40ac38ac7cbe3b4b4dea7ebf940e7b47bb02d4bd0c88d725b6c1a0f5bcb92f` |
| `chain_day2c` | 16 | 16 | 52 | 67 | none | `9e79ec2e6853d096fddf77fef955cf57fedf65f15cfa3239139e883a07bdb739` |
| `chain_day2d` | 2 | 1 | 67 | 67 | 68 stall | `338d26827d5721123c9c9d05289f6991dbb1a78d4980024160d69bda4659040b` |
| `chain_day2e` | 2 | 1 | 67 | 67 | 68 stall | `88d116392a0acba3ab37b9e33b2b98e933839eeb03ac6bfbc1da266eeec0c9ef` |
| `chain_day2f` | 33 | 31 | 68 | 98 | 68 stall, 99 stall | `ca1c9ba546604026f79a69bfa9007177975e634a7eb53dd5533fecf3ddf14a4b` |
| `chain_day2g` | 2 | 1 | 99-0 | 99-0 | 99-1 stall | `dd29b405d3bf64506008503f7c7cc4d44b5056c22cba24cc938630bd4e0c2805` |
| `chain_day2h_item` | 1 | 0 | none | none | 99-1 stall | `0b7d6945c20967dffe2ddc001a96536238a1a46e403b9beec313286db4ecfb00` |

The two chains that carry the front are `chain_day2f` (rounds 68 to 98, dated
2026-08-09) and `chain_day2g` (round 99-0, dated 2026-08-09). Solution
artifacts for the two frontier levels:
`runs/bubble_bobble/chain_day2f/lvl_29_98/solutions/sol_000.actions.npy` and
`runs/bubble_bobble/chain_day2g/lvl_00_99-0/solutions/sol_000.actions.npy`.

## How to re-derive it

```sh
python3 - <<'PY'
import glob, hashlib, json
for path in sorted(glob.glob("runs/bubble_bobble/*/chain.jsonl")):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    solved = [r["level"] for r in rows if r.get("status") == "solved"]
    other = [(r["level"], r.get("status")) for r in rows if r.get("status") != "solved"]
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(path, len(rows), len(solved), solved[:1], solved[-1:], other, digest)
PY
```

`tests/test_readme_bubble_round.py` runs the same derivation and compares it
against both this receipt and the numerals in `README.md`. It skips, rather
than passes, on a checkout where `runs/bubble_bobble/` is absent.

## Scope

This is a census of what the solver banked. It is not a fidelity claim. The
Mesen cross-check of the Bubble Bobble rounds 1 to 3 tape is a separate
receipt.
