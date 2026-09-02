# ROM library compatibility scan (SUPERSEDED, 2026-04-27)

> **Superseded. Do not quote the numbers below as current compatibility.**
> This scan ran on 2026-04-27 (commit `55e5333`) over a 794-ROM library, and
> its `ok` status means only "loaded and ran 300 frames without a panic or a
> timeout". It never distinguished a live screen from a frozen one, so its
> "793 (99.9%)" is not the live-screen boot figure the README quotes. The
> current library is 796 unique ROMs (806 files, 10 md5 duplicates) and the
> current figure is 793 of 796 (99.6%) booting into a live screen. The
> receipts are in `docs/receipts/rom_census/`; the scanner that produces them
> is `scripts/rom_library_scan.py --static-check`. Kept here for history.

Total ROMs: **794**  
Pass: **793** (99.9%)  

## Status breakdown
| status | count | % |
|---|---:|---:|
| `ok` | 793 | 99.9% |
| `header_parse_err` | 1 | 0.1% |

## Mapper coverage (known-header ROMs)
| mapper | total | ok | ok% | fail | supported? |
|---:|---:|---:|---:|---:|:---:|
| 0 | 58 | 58 | 100.0% | 0 | ✓ |
| 1 | 227 | 227 | 100.0% | 0 | ✓ |
| 2 | 98 | 98 | 100.0% | 0 | ✓ |
| 3 | 60 | 60 | 100.0% | 0 | ✓ |
| 4 | 223 | 223 | 100.0% | 0 | ✓ |
| 5 | 8 | 8 | 100.0% | 0 | ✓ |
| 7 | 32 | 32 | 100.0% | 0 | ✓ |
| 9 | 2 | 2 | 100.0% | 0 | ✓ |
| 11 | 30 | 30 | 100.0% | 0 | ✓ |
| 13 | 1 | 1 | 100.0% | 0 | ✓ |
| 34 | 2 | 2 | 100.0% | 0 | ✓ |
| 37 | 1 | 1 | 100.0% | 0 | ✓ |
| 41 | 1 | 1 | 100.0% | 0 | ✓ |
| 47 | 1 | 1 | 100.0% | 0 | ✓ |
| 64 | 5 | 5 | 100.0% | 0 | ✓ |
| 66 | 4 | 4 | 100.0% | 0 | ✓ |
| 68 | 1 | 1 | 100.0% | 0 | ✓ |
| 69 | 2 | 2 | 100.0% | 0 | ✓ |
| 71 | 10 | 10 | 100.0% | 0 | ✓ |
| 79 | 13 | 13 | 100.0% | 0 | ✓ |
| 105 | 1 | 1 | 100.0% | 0 | ✓ |
| 113 | 1 | 1 | 100.0% | 0 | ✓ |
| 118 | 4 | 4 | 100.0% | 0 | ✓ |
| 119 | 2 | 2 | 100.0% | 0 | ✓ |
| 228 | 2 | 2 | 100.0% | 0 | ✓ |
| 232 | 3 | 3 | 100.0% | 0 | ✓ |
| 234 | 1 | 1 | 100.0% | 0 | ✓ |

## Top error buckets
| status | error prefix | count |
|---|---|---:|
| `header_parse_err` | `failed to parse iNES header for <truncated dump>` | 1 |
