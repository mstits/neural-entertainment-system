# RULING: Bubble Bobble Mesen divergence

_2026-09-01 18:30-18:45 PDT. Repo `/Users/stits/Documents/macos-emulation-and-training` read-only
(no edits, no git writes). Six Mesen runs (~20 s each) and four nes_core replays (~10 s each), all
under `nice -n 15`, ~3 min machine time. Receipts, hashes and scripts in `ruling-receipts/`,
manifest `SHA256SUMS.ruling`. ROM sha256 `fc521e89...09d29`, the same file both sides load._

## Ruling: HARNESS ARTIFACT. Falsifier 4 did not fire.

The tape fed to Mesen was built wrong in two independent ways, both of them conventions the repo
already documents and one of them the exact bug the SMB report fixed the same morning. Feed Mesen
the correctly built tape at the correct phase and it clears rounds 1, 2 and 3 on the same frames
nes_core does, lives 3 throughout, at the SMB report's parity grade.

| Artifact | Where the convention is written down | Effect on the original run |
|---|---|---|
| **A1. Input phase 0 instead of +2.** The tape's byte 0 was applied at Mesen frame 0. The lineage is `env.reset()`, which advances one frame before the first input (`nes_core/src/python.rs:340`), plus the one-frame `inputPolled` phase between `env.step` and `mesen_cv_tape_dump.lua`. | `nes_core_cv_ram_dump.py:6-12` (`--mesen-align`); `B-fidelity-and-compat.md` §2.4, where SMB at phase 0 and phase 1 also died and only phase 2 held | Mesen's title-screen Start press lands one 16-frame pulse late (lives→3 at 273 vs 257, round→1 at 680 vs 664). Every gameplay input is then 16 frames early relative to the game. |
| **A2. Missing materialize block.** `TapePlayer.play()` runs one NOOP pool-step (4 frames) between the root state and the first action (`src/training/tape_replay.py:249`, "the no-op materializes RAM"), and so does the solver's lineage. The assembled tape goes straight from the prefix into action 0 (`bb_continuous_tape.bin[1816:1820] = [3,3,3,3]`). | `tape_replay.py:249`; the report's own `bb_build_reference.py` runs `env.step(0x00)  # materialize` after `load_state`, then labels the next row 1816 as if those 4 frames had not happened | Every gameplay input lands 4 frames early. The report's timeline frames after the prefix are off by 4 (round-1 clear is at 3112, not 3108). |

**Corrected run's result** (Mesen 2, same ROM, tape = 2 leading NOOPs + prefix + 4 NOOPs + actions;
`bb_tape_materialized_shift2_for_mesen.bin`, 6,966 bytes, sha256 `6a5c55b1...`; dump sha256
`a5217697...`):

| | nes_core (fs=1, `env.reset()`, no flags, no `load_state`) | Mesen 2 (+2 phase) |
|---|---:|---:|
| lives→3 / round→1 | 257 / 664 | 258 / 665 |
| round 1 clears (round byte → 2) | 3112 | 3112 |
| round 2 clears (→ 3) | 5186 | 5186 |
| round 3 clears (→ 4) | 6961 | 6961 |
| lives lost | none | none |
| end of tape | lives 3, round 4 | lives 3, round 4 |

Per-frame agreement over all 6,964 frames, same-tape-byte mapping (`corrected_pair_stats.log`):
lives **6,964/6,964**, round 6,958/6,964 (the 6 are the three round transitions' 2-frame
flicker landing one frame apart), player x 6,960/6,964, player y 6,899/6,964, enemies-left
6,959/6,964. 414 frames byte-exact. RAM diff per frame with the SMB report's ±1-frame tolerance:
**median 24, p90 34, max 135**, final frame 2; prefix median 13. SMB's corresponding numbers were
17 / 27 / 37. The one frame my strict gameplay test flagged (2028) is a score-digit write caught
mid-update on nes_core (`0000390` vs `39393939391`), identical one frame later. The max-diff frame
(6959, 385 bytes, 374 of them in the $0300-$07FF map area) is the round-3→4 scene reload one frame
apart, with lives, round, x and y all agreeing.

There is no gameplay fork in 6,964 frames. The Bubble Bobble rounds 1-3 tape is true of Mesen.

## The receipts, in the order they settled it

1. **Phase sweep, tape as built** (`run_shifts.sh`, `analyze_shifts.log`). Mesen at +2 loses the
   16-frame skew exactly: lives→3 at 259 and round→1 at 666, nes_core's 257/664 plus the 2-frame
   shift; prefix RAM diff median drops from ~100 to 13. Phases +0/+1/+3/-1 all keep the skew
   (272-274 / 679-681). So A1 is real and +2 is the unique fix, not a cherry-pick. But +2 alone is
   not enough: Mesen clears round 1 at 3455 instead of 3112 and loses two lives in round 2.
2. **nes_core against itself, tape as built** (`nescore_fs1_fulltape.log`). nes_core at
   `frame_skip=1`, one step per tape byte, `env.reset()`, no `load_state`, no flags, on the same
   uncorrected tape: round 1 clears at 3454, lives lost at 3963 and 5070, ends lives 1 round 2.
   That is Mesen(+2)'s trajectory (3455 / 3965 / 5072), matching it on lives 6,960/6,960 frames
   and round 6,958/6,960, median 29 bytes. It is NOT the solver lineage's trajectory. So the tape
   itself did not encode the lineage. The only difference between the two nes_core paths at the
   prefix/action boundary is the 4-frame materialize step.
3. **Corrected tape, nes_core fs=1** (`nescore_fs1_corrected.log`). Insert 4 NOOP bytes at offset
   1816: nes_core fs=1 is **byte-exact with the fs=4 solver lineage on all 1,287 rows** (row =
   label+4). This also retracts the report's §3 "real nes_core finding": the skip-render
   `frame_skip=4` path IS RAM-neutral for Bubble Bobble. The 159-byte discrepancy the report
   measured came from its fs=1 variants stepping 1 or 0 NOOP frames at the boundary
   (`bb_debug_pool_vs_env.py`), never 4.
4. **Corrected tape, Mesen +2** (`mesen_shift2_materialized.log`, `corrected_pair_stats.log`).
   The table above. Parity holds.

What the two skeptics did with the phase hypothesis: both re-sliced the same two recorded dumps
at different row offsets (±2 and ±40) and reported "no shift rescues parity". A row offset
between two recordings of two different playthroughs can never rescue anything; the SMB fix was
three separate Mesen runs with three shifted tapes (`mesen_smb_banked_segment{,_shifted,_shift2}`).
Both skeptic verdicts are correct about what they measured (start state, ROM hash, RAM init,
determinism) and wrong about what those measurements rule out. Neither touched A2 because neither
diffed the tape bytes at the boundary against the lineage's step sequence.

## Mechanism of what remains

The only divergence left is the cold-boot residue class SMB already has: 17 bytes at frame 1
under the same-frame-count mapping ($0000-$0002, $0084-$0085, $00F9-$00FA, $00FE-$00FF,
$0100-$0101, $01FA-$01FF; nes_core non-zero, Mesen zero), stack slots below the pointer and
zero-page temps, plus a one-frame skew at scene transitions where the two cores commit the reload
a frame apart. Same shape as SMB's 7-byte / 1-frame residue, somewhat larger, never reaching a
gameplay byte in 6,964 frames. Nothing here implicates MMC1, PPU timing, open bus or RAM init.
The `hw_flags` re-solve the report proposed is unnecessary for this tape.

## What CLAIMS.md and README must say now

Ruling in force: **HARNESS ARTIFACT**. The Bubble Bobble EXHIBITION claim (round 60 banked,
`README.md:68-70`, `:802`, `:859`, `:1227`) stands unchanged and gains a fidelity receipt.

- **CLAIMS.md, new entry** (append, dated 2026-09-01):
  > **BUBBLE BOBBLE MESEN CROSS-CHECK 2026-09-01: HELD.** The banked `chain_overnight` rounds 1-3
  > tape (1,268 actions, 5,072 gameplay frames, plus the 1,816-frame cold-boot prefix that
  > reproduces `Bubble Bobble (USA)_start.state.bin` byte-exact) replayed on Mesen 2 from power-on
  > clears rounds 1, 2 and 3 on the same frames as nes_core (3112 / 5186 / 6961), lives 3
  > throughout; lives agree on 6,964/6,964 frames, RAM differs by median 24 bytes per frame under a
  > one-frame tolerance. A same-day report (`mesen-parity-second-game.md`) recorded NOT HELD with
  > Mesen at GAME OVER by frame 4,032; that run fed the tape at input phase 0 and omitted the
  > 4-frame materialize step of `tape_replay.py:249`, and is withdrawn. Its §3 claim that
  > `frame_skip=4` native stepping is not equivalent to 4× `frame_skip=1` is also withdrawn:
  > they are byte-exact on all 1,287 rows once the materialize step is included. Receipts:
  > `personal_os/reports/.../bubble-bobble-divergence/RULING.md`.
- **README.md, FIDELITY paragraph (`:180-183`)**, replace the sentence beginning "Cross-checked
  against Mesen 2 on 12,000 frames" with:
  > Cross-checked against Mesen 2 under real input on two banked tapes (2026-09-01): 12,000 frames
  > of the Super Mario Bros run and 6,964 frames of the Bubble Bobble rounds 1-3 chain. On both,
  > nes_core and Mesen agree on lives and every level or round transition frame; scratch bytes
  > differ by a median of 17 (SMB) and 24 (Bubble Bobble) per frame under a one-frame tolerance.
  > The core is never claimed byte-identical to Mesen.
- **`docs/proposals/parity_coverage_map.md`**: do NOT append the report's §6 row as written. The
  Bubble Bobble row should read `HELD (rounds 1-3 clear on the same frames; lives 6,964/6,964;
  median 24 / p90 34 / max 135 ±1 frame)`, citing this ruling.
- **MISTAKES.md** (the OS's, not the repo's): one entry. Two documented harness conventions (the
  +2 input phase, the materialize step) were skipped by the second-game report, and two skeptic
  passes then tested the phase hypothesis by re-slicing recordings instead of re-running the
  reference. Rule that would have caught it: a skeptic pass on "harness artifact?" must re-run the
  reference under the hypothesized correction, not re-read the existing dumps; and any tape built
  for an external emulator must be replayed on nes_core `frame_skip=1` from `env.reset()` and
  shown byte-exact against the solver lineage BEFORE it is ever fed to Mesen (receipt 3 above is
  that check; it takes 10 s and would have ended this in one step).

Annotation text under the other two rulings, kept here so the choice is visible, not silent:

- *Had it been REAL DIVERGENCE:* CLAIMS.md would need "Bubble Bobble rounds 1-3: the banked tape
  clears on nes_core and fails on Mesen 2 at the matched phase (first gameplay divergence at frame
  N, byte $X); the round-60 EXHIBITION entry is a nes_core-only result until re-solved under an
  anchored lineage", README `:859` would change "round-clear observed live" to "round-clear
  observed live on nes_core; not reproduced on Mesen", and falsifier 4 would have fired.
- *Had it been UNDECIDED:* CLAIMS.md would carry "Mesen cross-check of the Bubble Bobble tape is
  open: run X pending", and no README wording would change.

## Falsifier 4 and tonight's priorities

`STRATEGY_2026-08-08.md:55`: "Bisect finds real Mesen divergence altering solver reachability →
fidelity jumps above basis conversion; integrity outranks throughput." No real divergence was
found; the banked tape reaches round 4 on the reference. **Falsifier 4 did not fire.** Fidelity
does not jump above basis conversion. The red-team's "a tape true of nes_core and false of the
NES" gap narrows rather than opens: Bubble Bobble is now the second input-driven game with a
Mesen receipt, and it is a mapper-1 (MMC1) title, which SMB (mapper 0) was not.

Tonight's priorities are unchanged. One hygiene item, ~30 min total, so this artifact cannot
recur and the wrong verdict cannot get quoted:

1. Mark `mesen-parity-second-game.md` withdrawn at the top with a two-line pointer to this ruling
   (~5 min). Do not apply its §6 items as written.
2. When the generalized `nes_core_ram_dump.py` from that report is applied, add the materialize
   step to the tape builder and default the Mesen feed to the +2 phase (or make
   `--mesen-align` count 2), with the 10-second fs=1-vs-lineage byte-exact check as its test
   (~20 min).
3. CLAIMS.md and README annotations above (~5 min).

## Receipts

`ruling-receipts/` (manifest `SHA256SUMS.ruling`, 30 files, 5.4 MB): `run_shifts.sh` +
`run_shifts.log` and `mesen_shift{-1,1,2,3}.log` (the phase sweep, tape sha256s inline),
`mesen_bb_shift{0,1,2,3,-1}.bin.gz` (per-frame Mesen RAM at each phase; shift0 is the report's
own dump), `analyze_shifts.{py,log}` (timelines and reference-match stats per phase),
`prefix_perframe.{py,log}` (first divergence per phase), `fork_shift2.{py,log}` (where +2 alone
forks), `nescore_fs1_fulltape.{py,log}` + `ours_fs1_fulltape_uncorrected.npz` (receipt 2),
`bb_tape_materialized.bin` (6,964 bytes, sha256 `4a491183...`), `nescore_fs1_corrected.{py,log}` +
`ours_fs1_corrected.npz` (receipt 3, byte-exact), `bb_tape_materialized_shift2_for_mesen.bin`,
`mesen_shift2_materialized.log`, `mesen_bb_shift2_materialized.bin.gz` (receipt 4),
`corrected_pair_stats.{py,log}` and `corrected_pair_stats_part2.log` (the parity table).
`/tmp/cv_tape.bin` was restored to the report's original tape (sha256 `56c2d545...`) after each
run.
