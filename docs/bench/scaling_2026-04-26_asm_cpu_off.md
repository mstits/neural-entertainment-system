# nes_core vs nes-py — scaling bench (2026-04-26, asm_cpu disabled)

Captured via `scripts/bench_vs_nes_py_sweep.py` on M4 Max (16 cores).
ROM: Super Mario Bros. (World).nes. Both backends warmed 2 steps
before timing. nes-py runs as `multiprocessing.spawn` workers; nes_core
runs as a single-process `Pool` (rayon thread pool).

## fs=1 (raw per-frame)
| N | nes-py fps | core fps | ratio  | core/realtime |
|---|------------|----------|--------|---------------|
|  1|     1 406  |     643  | 0.46×  |       10.7×   |
|  2|     2 911  |   1 300  | 0.45×  |       21.7×   |
|  3|     4 681  |   1 920  | 0.41×  |       32.0×   |
|  4|     5 865  |   2 547  | 0.43×  |       42.4×   |
|  5|     7 208  |   3 137  | 0.44×  |       52.3×   |
|  6|     8 318  |   3 713  | 0.45×  |       61.9×   |
|  7|     9 923  |   4 325  | 0.44×  |       72.1×   |
|  8|    11 790  |   4 908  | 0.42×  |       81.8×   |
|  9|    12 550  |   5 531  | 0.44×  |       92.2×   |
| 10|    12 581  |   6 078  | 0.48×  |      101.3×   |
| 11|    14 034  |   6 518  | 0.46×  |      108.6×   |
| 12|    15 354  |   6 538  | 0.43×  |      109.0×   |

## fs=4 (typical RL cadence)
| N | nes-py fps | core fps | ratio  | core/realtime |
|---|------------|----------|--------|---------------|
|  1|     1 477  |     842  | 0.57×  |       14.0×   |
|  2|     2 887  |   1 746  | 0.60×  |       29.1×   |
|  3|     4 438  |   2 556  | 0.58×  |       42.6×   |
|  4|     6 070  |   3 391  | 0.56×  |       56.5×   |
|  5|     7 287  |   4 166  | 0.57×  |       69.4×   |
|  6|     8 599  |   5 010  | 0.58×  |       83.5×   |
|  7|     9 912  |   5 807  | 0.59×  |       96.8×   |
|  8|    11 399  |   6 630  | 0.58×  |      110.5×   |
|  9|    12 682  |   7 397  | 0.58×  |      123.3×   |
| 10|    14 372  |   8 220  | 0.57×  |      137.0×   |
| 11|    14 833  |   9 003  | 0.61×  |      150.1×   |
| 12|    15 809  |   9 457  | 0.60×  |      157.6×   |

## fs=16 (aggressive RL)
| N | nes-py fps | core fps | ratio  | core/realtime |
|---|------------|----------|--------|---------------|
|  1|     1 497  |     928  | 0.62×  |       15.5×   |
|  2|     2 891  |   1 879  | 0.65×  |       31.3×   |
|  3|     4 457  |   2 784  | 0.62×  |       46.4×   |
|  4|     5 982  |   3 711  | 0.62×  |       61.8×   |
|  5|     7 380  |   4 583  | 0.62×  |       76.4×   |
|  6|     8 739  |   5 494  | 0.63×  |       91.6×   |
|  7|    10 080  |   6 403  | 0.64×  |      106.7×   |
|  8|    11 333  |   7 282  | 0.64×  |      121.4×   |
|  9|    12 791  |   8 167  | 0.64×  |      136.1×   |
| 10|    14 314  |   9 094  | 0.64×  |      151.6×   |
| 11|    15 520  |   9 907  | 0.64×  |      165.1×   |
| 12|    15 957  |  10 192  | 0.64×  |      169.9×   |

## Comparison to prior asm_cpu-ENABLED baseline (2026-04-20)

| config            | was (asm_cpu ON) | now (asm_cpu OFF) | cost of disable     |
|-------------------|------------------|-------------------|---------------------|
| fs=1   single     | 0.46×            | 0.46×             | none — render-bound |
| fs=4   12-parallel| 2.15×            | 0.60×             | ~3.6× slowdown      |
| fs=16  12-parallel| 1.88×            | 0.64×             | ~2.9× slowdown      |

Per-worker scaling stays roughly flat 1→10, drops slightly 11→12 as the
M4 Max's 12 P-cores saturate. No rayon contention pathology.

asm_cpu fix (#40) buys back the ~3× parallel training throughput.
