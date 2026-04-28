# Implementation Plan: nes_core Reference-Emulator Diff Harness

Spec: `docs/proposals/nes_parity_harness.md`
Status: DRAFT — awaiting review
Date: 2026-04-22

## Overview

Four phases, 14 tasks, vertically sliced so Mario baseline lands
end-to-end before Zelda/Contra tapes enter the picture. Phase 4 is the
"does this harness actually catch bugs" proof that closes the spec's
success criteria #3 and #4.

## Architecture Decisions

- **Hybrid diff: `cross_emulator` + `golden_hash` modes.** nes_core is
  non-cycle-accurate; idle tapes match nes-py byte-exact, input-driven
  tapes drift by thousands of pixels. Cross-emulator mode works only for
  the former. Golden-hash mode (nes_core self-reference against
  committed per-frame hashes) catches regressions on input-driven tapes.
- **Drivers are concrete classes, not ABC/Protocol.** Two drivers total;
  formalizing the interface is premature. Revisit if Mesen driver lands.
- **Tape authoring is CLI-scripted, not GUI-recorded** in Phase 1. Good
  enough for three hand-crafted tapes; revisit if we need a dozen.
- **Regression proof lives on a scratch branch**, not `master`. Tasks 13
  and 14 are ephemeral experiments that validate the suite, then revert.
- **Early failure over thorough failure.** First frame past tolerance
  short-circuits the run. PNG dump captures the divergence; the full
  log of every diverged frame is not worth the runtime cost.

## Dependency Graph

```
T1 scaffold
 │
 ├── T2 palette-LUT extraction ──┐
 │                                │
 ├── T3 drivers ──────────────────┼── T6 run.py CLI
 │                                │        │
 └── T4 tape loader ──┬───────────┘        │
                      │                    │
                      └── T5 diff + PNG ───┴── T7 Mario tape
                                                   │
                                                   └── T8 pytest + make
                                                         │
                                           ┌─────────────┤
                                           │             │
                              T9 author.py │             │
                                    │      │             │
                                    ├── T10 Zelda tape   │
                                    │      │             │
                                    └── T11 Contra tape  │
                                           │             │
                                           └── T12 inspect.py
                                                   │
                                           ┌───────┴───────┐
                                           │               │
                                  T13 palette-revert   T14 scroll-offset
                                  regression proof    regression proof
```

## Task List

### Phase 1: Foundation

#### Task 1: Directory scaffold + pytest marker

**Description:** Create `tests/parity/` tree with `__init__.py`,
`conftest.py` registering `@pytest.mark.parity`, and an empty tapes
directory. No logic yet — just make the import surface exist so later
tasks don't get blocked by missing-path errors.

**Acceptance:**
- [ ] `tests/parity/__init__.py`, `conftest.py`, `tapes/` exist
- [ ] `pytest -m parity` runs (zero tests, exit 0)
- [ ] `pytest --markers | grep parity` prints the marker description

**Verify:** `.venv/bin/pytest -m parity -q` exits 0.

**Dependencies:** None.

**Files touched:**
- `tests/parity/__init__.py`
- `tests/parity/conftest.py`
- `tests/parity/tapes/.gitkeep`

**Size:** XS.

---

#### Task 2: Palette parity guard

**Description:** Empirical check established during planning that nes_core
and nes-py share the Laines LUT and produce byte-identical RGB. This
task makes that invariant a pre-flight test so any future LUT drift in
either emulator fails loud with a clear message, not as a cryptic cascade
across every tape.

One test: boot Mario Bros., advance 30 frames with zero input, assert
`nes_core.get_frame() == nes_py.screen` byte-for-byte. This check is
cheap (<1s) and runs as part of `make parity`.

Note on scope: the original T2 had us extracting palette LUTs and
building an RGB→index normalizer. That's unnecessary — both emulators
match at the RGB level already, and future reference emulators (Mesen
Phase 2) can slot a normalizer in if/when needed. Don't build it on
speculation.

**Acceptance:**
- [ ] `test_palette_parity.py` passes: 0 pixel diff at Mario frame 30
- [ ] Deliberately perturbing one emulator's output (artificial +1 to
      red channel, verified manually) causes the test to fail with
      a message naming the expected vs actual RGB and pixel count

**Verify:** `.venv/bin/pytest tests/parity/test_palette_parity.py -q` passes.

**Dependencies:** T1.

**Files touched:**
- `tests/parity/test_palette_parity.py`

**Size:** XS.

---

#### Task 3: Drivers (NESCoreDriver, NESPyDriver)

**Description:** Thin wrappers with duck-typed `step(buttons: int)` and
`frame() -> np.ndarray` methods returning `(240, 256, 3)` uint8 RGB.
Drivers own construction, reset, and button dispatch. No diff logic here.

**Acceptance:**
- [ ] `NESCoreDriver(rom_path)` constructs and returns a valid frame
      after one step
- [ ] `NESPyDriver(rom_path)` same
- [ ] Both drivers accept the NES 8-bit button bitmask (A=1, B=2,
      Select=4, Start=8, Up=16, Down=32, Left=64, Right=128) and advance
      exactly one frame per `step`
- [ ] Button-bit alignment test: holding Start (bit 3) on both drivers
      produces identical PPU $2002 behavior for the first 10 frames of
      Mario (sanity — both should see title-screen animation advance)
- [ ] Two consecutive runs of the same tape produce bit-identical
      RGB frames on both drivers (determinism check)

**Verify:** `.venv/bin/pytest tests/parity/test_drivers.py -q` passes.

**Dependencies:** T1, T2.

**Files touched:**
- `tests/parity/drivers.py`
- `tests/parity/test_drivers.py`

**Size:** S.

---

### Checkpoint: Foundation (after T1–T3)

- [ ] `pytest tests/parity/ -q` passes (normalize + drivers)
- [ ] Both drivers produce aligned frames from the same ROM + input
- [ ] Palette LUTs committed and reproducible via extract script
- [ ] **Review with Matthew before proceeding**

---

### Phase 2: Mario baseline end-to-end

#### Task 4: Tape format + loader with md5 verify

**Description:** JSON tape loader matching the spec's schema (`name`,
`rom`, `rom_md5`, `frames`, `frame_skip`, `mode`, `tolerance_*`,
`inputs` list of ranges, optional `golden_hashes` path). Build a
`(frames,)` uint8 button-bitmask array from disjoint/overlapping ranges
(last-write-wins). Verify ROM md5 on load. For `golden_hash` mode,
load the sibling hash blob as a `(frames,)` uint64 array.

**Acceptance:**
- [ ] `Tape.load(path)` returns dataclass with fields populated
- [ ] ROM md5 mismatch raises with both expected and actual hash
- [ ] `Tape.button_sequence()` returns `(frames,)` uint8 array
- [ ] Overlapping ranges: last-write-wins (unit-tested with two
      overlapping ranges)
- [ ] Malformed tape (missing field, bad JSON) raises with field name
- [ ] `mode="golden_hash"` loads the golden blob; length mismatch
      between `frames` and blob-length raises
- [ ] `mode="cross_emulator"` ignores `golden_hashes` if present

**Verify:** `.venv/bin/pytest tests/parity/test_tape.py -q` passes.

**Dependencies:** T1.

**Files touched:**
- `tests/parity/tape.py`
- `tests/parity/test_tape.py`

**Size:** S.

---

#### Task 5: Diff module (cross-emulator + golden-hash)

**Description:** Two diff paths. `scanline_divergence(a_rgb, b_rgb)`
returns `(240,) int32` pixel-count diff per scanline — used by
cross-emulator mode. `fnv1a_frame(rgb) -> int` hashes a frame's raw
bytes — used to compare against golden blob in golden-hash mode. Plus
`dump_failure(...)` writes PNGs on failure:
- cross_emulator: `ours.png`, `theirs.png`, `diff.png`
- golden_hash: `ours.png` + `expected_hash.txt` (no theirs frame —
  the golden blob only stores hashes, not frames)

**Acceptance:**
- [ ] Scanline divergence matches hand-computed Hamming on a synthetic
      `(240, 256, 3)` uint8 pair with known-different rows
- [ ] `fnv1a_frame` returns stable, deterministic output (two calls on
      the same buffer produce the same hash; one-byte change produces
      different hash)
- [ ] Cross-emulator PNG dump writes three files; diff.png uses red =
      diverged pixel on desaturated background (golden PNG fixture)
- [ ] Golden-hash PNG dump writes `ours.png` + text file with
      `expected_fnv=<hex> actual_fnv=<hex>`
- [ ] No dependency on matplotlib; pure PIL/numpy

**Verify:** `.venv/bin/pytest tests/parity/test_diff.py -q` passes.

**Dependencies:** T1.

**Files touched:**
- `tests/parity/diff.py`
- `tests/parity/test_diff.py`
- `tests/parity/golden/diff_reference.png` (test fixture)

**Size:** S.

---

#### Task 6: run.py CLI (single-tape runner)

**Description:** `python -m tests.parity.run --tape <path> [--verbose]`
loads tape, constructs required driver(s) based on mode, advances
frame by frame, runs the appropriate diff path (cross-emulator scanline
Hamming vs nes-py, or golden-hash FNV-1a vs committed blob),
short-circuits on first over-tolerance frame, writes failure artifacts,
prints single-line result. Exit 0 pass, 1 fail.

**Acceptance:**
- [ ] `--tape <valid>` exits 0 on a passing tape
- [ ] `--tape <valid>` exits 1 on a failing tape with message
      `<tape>: frame <N>: <K> scanlines diverged → <png_path>`
- [ ] `--verbose` prints per-frame `[<N>/<total>] OK` progress
- [ ] Early exit on first over-tolerance frame (verified via a tape
      rigged to fail on frame 10 of 600 — run completes in <500ms)

**Verify:** Manual — `python -m tests.parity.run --tape tests/parity/tapes/mario_walk_right.json` (once T7 lands).

**Dependencies:** T3, T4, T5.

**Files touched:**
- `tests/parity/run.py`

**Size:** S.

---

#### Task 7: Author + commit `mario_walk_right` tape

**Description:** Craft the Mario baseline tape by hand (simple: hold
Right for 600 frames, starting at frame ~120 after BIOS/title). Run
`run.py` against it; tape passes byte-exact. If it doesn't, we've found
a bug — stop and file it, don't force the tape.

**Acceptance:**
- [ ] `tests/parity/tapes/mario_walk_right.json` exists, md5 verified
- [ ] `python -m tests.parity.run --tape <mario>` exits 0
- [ ] Tape file has `notes` field explaining what it exercises

**Verify:** `time python -m tests.parity.run --tape tests/parity/tapes/mario_walk_right.json` — exit 0, <10s.

**Dependencies:** T6.

**Files touched:**
- `tests/parity/tapes/mario_walk_right.json`

**Size:** XS (authoring, no new code).

---

#### Task 8: pytest integration + Makefile `parity` target

**Description:** `test_tapes.py` parametrizes over every `.json` in
`tapes/` and asserts `run.py` exits 0. Also includes the spec's
determinism check: each tape runs twice, second run's final-frame
palette-index hash must match the first. Add `parity` target to the
top-level `Makefile` invoking `pytest -q -m parity`.

**Acceptance:**
- [ ] `pytest -m parity` runs all tapes, currently just Mario, passes
- [ ] Determinism check runs each tape twice, fails if hashes differ
- [ ] `make parity` exits 0 with Mario tape present
- [ ] Total runtime <15s (only one tape at this point; well within budget)

**Verify:** `time make parity` — exit 0, <15s.

**Dependencies:** T7.

**Files touched:**
- `tests/parity/test_tapes.py`
- `Makefile`

**Size:** S.

---

### Checkpoint: Mario baseline (after T4–T8)

- [ ] `make parity` runs end-to-end in <15s
- [ ] Mario tape passes byte-exact
- [ ] Failure path (deliberately break the tape) produces a PNG artifact
- [ ] **Review with Matthew before proceeding to Zelda/Contra**

---

### Phase 3: Bug-catching tapes

#### Task 9: author.py (tape recorder)

**Description:** CLI that runs nes_core headless, reads button state
per frame (from stdin interactively, or `--from-file <script>`),
writes both the tape JSON and the golden hash blob (for
`golden_hash`-mode tapes). `--mode {cross_emulator,golden_hash}`
selects output shape.

**Acceptance:**
- [ ] `python -m tests.parity.author --rom <rom> --out <tape> --mode golden_hash --from-file <script>`
      produces a loadable tape + sibling `.golden.bin` blob
- [ ] `--mode cross_emulator` produces tape JSON with no golden blob
- [ ] Generated tape's `rom_md5` is correct
- [ ] Round-trip: author a tape, load it via T4 loader, re-run through
      `run.py` (T6) — passes green
- [ ] Interactive stdin mode works (smoke — not tested in CI)

**Verify:** Use it to author a throwaway tape of each mode, round-trip
through `run.py`.

**Dependencies:** T3, T4, T5.

**Files touched:**
- `tests/parity/author.py`

**Size:** S.

---

#### Task 10: Author + commit `zelda_title_sprites` tape (golden_hash)

**Description:** Originally spec'd as `zelda_sword_swing`. Scripted-input
cold-boot to the overworld turned out to be brittle — Zelda 1 requires
entering at least one name character before REGISTER is selectable,
and name-entry cursor navigation under pure scripted input without
per-frame visual feedback is unreliable. Pivoted to
`zelda_title_sprites`: exercises the same sprite-rendering code paths
(cursor sprites, letter sprites, title-screen animation) through the
title + file-select + name-entry flow. Any OAM or sprite-0-hit
regression in nes_core will shift the golden hash.

**Acceptance:**
- [ ] `tests/parity/tapes/zelda_title_sprites.json` + `.golden.bin` exist,
      md5 verified, mode=`golden_hash`
- [ ] Tape passes against its own golden hashes on HEAD `b7f26d6`
- [ ] `notes` field explains "OAM/sprite-0 regression target"
- [ ] Tape is ≤900 frames (15s at 60fps; keeps suite budget intact)

**Verify:** `python -m tests.parity.run --tape tests/parity/tapes/zelda_title_sprites.json` exits 0, <15s.

**Dependencies:** T9.

**Files touched:**
- `tests/parity/tapes/zelda_title_sprites.json`
- `tests/parity/tapes/zelda_title_sprites.script.txt` (authoring script, committed for reproducibility)

**Size:** S (likely iterative — may take a few authoring attempts).

---

#### Task 11: Author + commit `contra_title` tape (golden_hash)

**Description:** Cold-boot Contra, let the title screen render for 600
frames with a Start press in frames 120-130 (triggers the scroll-split
timing window that reveals the 1px horizontal drift). Much simpler
than Zelda — no navigation, just title-screen behavior. Authored in
`golden_hash` mode; cross-emulator mode isn't usable (11,860 px drift
vs nes-py empirically).

**Acceptance:**
- [ ] `tests/parity/tapes/contra_title.json` + `.golden.bin` exist,
      md5 verified, mode=`golden_hash`
- [ ] Tape passes against its own golden hashes on HEAD `b7f26d6`
- [ ] `notes` field explains "horizontal scroll-split regression target"

**Verify:** `python -m tests.parity.run --tape tests/parity/tapes/contra_title.json` exits 0, <10s.

**Dependencies:** T9.

**Files touched:**
- `tests/parity/tapes/contra_title.json`
- `tests/parity/tapes/contra_title.script.txt`

**Size:** XS.

---

#### Task 12: inspect.py (frame-dump utility)

**Description:** Debug aid. `python -m tests.parity.inspect --tape <t>
--frame <n>` advances both emulators to frame N, writes
`ours.png`, `theirs.png`, and `both.png` (side-by-side). No diff logic;
this is for when the diff PNGs aren't enough and you need raw frame
context at arbitrary points.

**Acceptance:**
- [ ] Writes three PNGs at user-specified output dir
- [ ] Works on any tape, not just failing ones
- [ ] Frame N defaults to the last frame if `--frame` omitted

**Verify:** Manual — run against mario tape at frame 300.

**Dependencies:** T3, T4.

**Files touched:**
- `tests/parity/inspect.py`

**Size:** XS.

---

### Checkpoint: Full suite (after T9–T12)

- [ ] `make parity` runs all three tapes in <30s (spec success #1)
- [ ] All three tapes pass byte-exact on `b7f26d6` (spec success #2)
- [ ] `inspect.py` works on any tape
- [ ] **Review with Matthew before Phase 4 (regression proofs)**

---

### Phase 4: Regression proofs (ephemeral branch work)

#### Task 13: Palette-revert regression proof

**Description:** On a scratch branch `parity-proof-palette`, revert the
Laines palette in `nes_core/src/sink/video_sink.rs` (replace with an
arbitrary perturbation or the prior Blargg LUT). Rebuild nes_core
Python binding. Run `make parity` — `mario_walk_right` must fail first
(cross-emulator mode catches palette drift against nes-py); the golden-
hash tapes will also fail because their frame hashes change. Record
the failure PNGs, log, then **delete the branch**.

**Acceptance:**
- [ ] Branch `parity-proof-palette` exists with the palette revert
- [ ] `make parity` fails on all three tapes; `mario_walk_right`'s
      failure PNG visibly shows wrong palette
- [ ] `docs/proposals/parity_harness_proof_log.md` records the PNGs +
      timing (one-paragraph log entry, not a full report)
- [ ] Branch deleted after proof (not merged)

**Verify:** Manual — exit code 1 from `make parity` on the scratch branch.

**Dependencies:** Phase 3 complete.

**Files touched (scratch branch only):**
- `nes_core/src/video_sink.rs` (reverted)
- `docs/proposals/parity_harness_proof_log.md` (on master, merged after)

**Size:** S.

---

#### Task 14: Scroll-offset regression proof

**Description:** Same pattern as T13. Scratch branch
`parity-proof-scroll`, introduce a synthetic `+1` to the fine-X scroll
in `ppu_neon::render_scanline`, rebuild, run harness. Must fail
`contra_title` in golden-hash mode (1px shift → different pixels →
different FNV-1a). `mario_walk_right` may or may not fail depending
on whether walk-right touches the affected scroll path — either
outcome is fine; we only need Contra to fire. Log, then delete branch.

**Acceptance:**
- [ ] Branch `parity-proof-scroll` exists with the synthetic offset
- [ ] `contra_title` fails on the first frame whose hash diverges; PNG
      dump visibly shows 1px horizontal shift
- [ ] `parity_harness_proof_log.md` updated with PNG + frame number
- [ ] Branch deleted after proof

**Verify:** Manual — Contra tape fails, PNG shows the drift.

**Dependencies:** Phase 3 complete.

**Files touched (scratch branch only):**
- `nes_core/src/ppu_neon.rs`
- `docs/proposals/parity_harness_proof_log.md` (on master)

**Size:** S.

---

### Final Checkpoint (after T13–T14)

- [ ] All spec success criteria met (1–6)
- [ ] `parity_harness_proof_log.md` committed to master
- [ ] Scratch branches deleted
- [ ] `make parity` is green on `master`
- [ ] Ready for `code-review-and-quality` pass

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Zelda tape authoring takes many attempts (sprite state depends on subtle input timing) | Med | T9 landing before T10; iterate the input script, don't fight the harness |
| nes-py or nes_core palette LUT silently changes (e.g., after a dep upgrade or refactor) | Med | T2 palette-parity guard runs as part of `make parity`; failure names the LUT-drift condition before any tape is consulted |
| nes-py step semantics differ subtly from nes_core (button held mid-frame vs whole frame) | High | T3 button-bit alignment test catches it; if divergence is real, tighten the Python binding contract before Phase 2 |
| Mario tape fails on HEAD (active bug we didn't know about) | Low | Good find — file it, defer Phase 2 until fixed. Don't ship a lowered tolerance to mask it |
| Tape files explode in size (committed golden PNGs) | Low | PNG artifacts go to `failures/` which is gitignored; only `tapes/` JSONs are committed |
| Determinism check flakes (non-deterministic drift in nes_core) | Med | Stop and fix nes_core. A flaky parity suite is worse than no parity suite |

## Open Questions (carried from spec)

1. nes-py mid-frame button toggle semantics — resolve in T3.
2. nes_core vs nes-py initial-frame alignment — resolve in T3, document in
   `tapes/README.md` if offset > 0.
3. Palette-LUT-drift guard — handled by the rescoped T2 palette-parity
   test. Resolved.

## Parallelization

- T10 (Zelda) and T11 (Contra) are independent and could be parallelized
  across sessions. In practice, one session writes both since they
  share the author tool from T9.
- T13 and T14 are independent scratch-branch experiments; if reviewing
  async with another engineer, safe to split.
- Nothing in Phases 1-2 parallelizes safely — strict dependency chain.

## Estimated timing

Sizing says S/XS across the board. Rough walltime on M4 Max with me
driving:

| Phase | Wall time |
|---|---|
| Phase 1 (T1-T3) | 2-3h |
| Phase 2 (T4-T8) | 3-4h |
| Phase 3 (T9-T12) | 2-3h (Zelda authoring is the wildcard) |
| Phase 4 (T13-T14) | 1h |
| **Total** | **8-11h** |

Single focused session can reach the Phase 2 checkpoint (Mario green).
Phase 3-4 is a second session.
