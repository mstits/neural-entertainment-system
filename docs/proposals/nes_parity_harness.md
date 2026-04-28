# Spec: nes_core Reference-Emulator Diff Harness

Status: DRAFT — awaiting review
Author: Matthew Stits
Date: 2026-04-22

## Objective

Build a headless, reproducible regression harness that catches PPU/scroll/
sprite/mapper regressions in nes_core before they reach the trainer or GUI.

Each tape runs in one of two modes:

- **`cross_emulator`** — runs nes_core and nes-py, diffs XRGB8888
  framebuffers scanline by scanline. Only valid for tapes that stay in
  timing-stable states (idle baselines, constant inputs). Catches broad
  breakage: palette LUTs, mapper loading, core rendering.
- **`golden_hash`** — runs nes_core alone, hashes each frame, compares
  against per-frame hashes committed alongside the tape. Works for
  *any* tape including state-transitioning ones, but only catches
  regressions relative to the frozen known-good nes_core build.

The hybrid is forced by empirical reality: nes_core is explicitly
non-cycle-accurate (per its README). Zero-input tapes converge
byte-exact with nes-py; input-driven tapes diverge by hundreds to
thousands of pixels (Contra title + Start = 11,860 px diff at frame 600).
A 1px scroll regression is invisible against that noise, so input-driven
tapes must use `golden_hash` mode to have any signal.

### Palette note

Both nes_core (post-commit `dd3ff1a`) and nes-py use the same Laines
palette LUT. Verified empirically: Mario Bros. frame 30 produces
byte-identical RGB frames in both emulators (0 pixels differ out of
61,440). This is what makes `cross_emulator` mode viable on idle tapes.

If a Phase 2 reference emulator uses a different LUT (e.g., Mesen with
its own palette), a normalization layer slots in between `drivers` and
`diff` without touching either. Not building that layer now.

The harness exists to catch PPU/scroll/sprite regressions before they reach
the trainer or the GUI. It replaces the ad-hoc "eyeball Contra on the screen"
loop we fell into during the 2026-04-22 palette chase.

### Who/why

- **User:** me (this repo's sole maintainer). Run locally before every merge
  to `master` and from `make parity` in pre-push.
- **Downstream consumer:** eventual GitHub Actions CI (out of scope for
  Phase 1 — hook up when the harness is proven locally).

### Must catch (acceptance)

Phase 1 ships three tapes that each trigger a specific known-bad behavior if
regressed:

| Tape | ROM | Mode | What it proves |
|---|---|---|---|
| `mario_walk_right` | Mario Bros. (World) | cross_emulator | baseline — byte-exact RGB parity with nes-py after 600 frames of walk-right (empirically verified zero drift for this specific input) |
| `zelda_title_sprites` | Legend of Zelda (USA) (Rev A) | golden_hash | sprite OAM correctness exercised via title/file-select/name-entry sprite animation (renamed from `zelda_sword_swing` — see T10 addendum) |
| `contra_title` | Contra (USA) | golden_hash | horizontal scroll-split on title screen (1px drift manifests here) |

All three must pass byte-exact (after palette-index normalization) on HEAD
as of commit `b7f26d6`. If they don't, the harness authoring also uncovers
an active bug — that's a find, not a blocker, and gets filed separately.

## Tech Stack

- **Rust:** nes_core (this repo), called via its existing PyO3 bindings
  (`feature = "python"`).
- **Python 3.11+** in `.venv/`. Drives the harness.
- **nes-py 8.x** (already installed in `.venv`). Reference emulator.
- **pytest** for the integration tests (fits `tests/` layout).
- **numpy** for framebuffer normalization and diff math.

No new third-party deps. `numpy` and `pytest` are already in
`requirements.txt`.

## Commands

```
# Run the full 3-tape Phase 1 suite (target <30s)
make parity

# Individual tape, with divergence report + PNG dump on failure
.venv/bin/python -m tests.parity.run --tape tapes/contra_title.json --verbose

# Author a new tape (records input over an interactive session)
.venv/bin/python -m tests.parity.author --rom roms/Zelda.nes --out tapes/zelda_sword_swing.json

# Re-run the normalize-and-dump path for a single frame (debugging)
.venv/bin/python -m tests.parity.inspect --tape <tape> --frame <n>
```

`make parity` invokes `pytest -q tests/parity/ -m parity`.

## Project Structure

```
tests/parity/
    __init__.py
    conftest.py                  pytest marker registration
    test_tapes.py                one test per tape under tapes/
    test_palette_parity.py       asserts nes_core == nes-py RGB byte-exact
    run.py                       CLI entry for single-tape runs
    author.py                    interactive tape recorder
    inspect.py                   frame-dump utility
    diff.py                      scanline Hamming + PNG dump
    drivers.py                   thin wrappers: NESPyDriver, NESCoreDriver
tests/parity/tapes/
    mario_walk_right.json        Phase 1 tape
    zelda_sword_swing.json       Phase 1 tape
    contra_title.json            Phase 1 tape
tests/parity/golden/             diff PNG reference fixture for T5
docs/proposals/
    nes_parity_harness.md        this spec
```

No changes to `nes_core/` source — the harness uses existing public API
(`NESEnvironment` with `frame_skip=1`, per-frame button input, `screen`
accessor returning XRGB8888 numpy array).

## Tape Format

JSON, deterministic, human-authorable. One schema, two modes:

```json
{
  "name": "contra_title",
  "rom": "roms/Contra (USA).nes",
  "rom_md5": "88e67b5a9c8e8b9a...",
  "frames": 600,
  "frame_skip": 1,
  "mode": "golden_hash",
  "warmup_frames": 10,
  "tolerance_scanlines_diverged": 0,
  "tolerance_pixels_per_scanline": 0,
  "inputs": [
    {"start_frame": 0,   "end_frame": 600, "buttons": 0},
    {"start_frame": 120, "end_frame": 130, "buttons": 8}
  ],
  "golden_hashes": "contra_title.golden.bin",
  "notes": "Title screen, press Start @120-130 to trigger scroll-split."
}
```

- `mode`: `"cross_emulator"` (diff vs nes-py, tolerances apply) or
  `"golden_hash"` (diff vs committed hashes, zero tolerance — any hash
  mismatch fails). Default is `"cross_emulator"`.
- `buttons` is the NES button bitmask: A=1, B=2, Select=4, Start=8,
  Up=16, Down=32, Left=64, Right=128 (matches `nes_core.BUTTON_*`).
- `rom_md5` is verified at load time. Mismatch is a hard fail.
- Ranges are half-open `[start, end)`. Disjoint ranges overwrite
  earlier ones; last-write-wins for overlapping ranges.
- `tolerance_*` only apply in `cross_emulator` mode. Default 0/0
  (byte-exact). Raising either requires a `notes` justification.
- `warmup_frames` skips comparison on the opening frames while both
  emulators stabilize. Empirically, Mario Bros. shows transient
  full-frame divergence on frames 1-6 post-reset before settling into
  lockstep at frame 7. Default 10 for cross_emulator tapes; 0 for
  golden_hash (self-reference doesn't drift).
- `golden_hashes`: for `golden_hash` mode, path (relative to tape file)
  to a binary file of `frames` × 8-byte little-endian FNV-1a hashes of
  each frame's RGB buffer. Generated by `author.py`; committed alongside
  the tape JSON.

## Code Style

Short, boring, numpy-native. One real example:

```python
# tests/parity/diff.py
import numpy as np

def scanline_divergence(
    a_idx: np.ndarray,  # (240, 256) uint8, NES palette indices 0..63
    b_idx: np.ndarray,  # (240, 256) uint8
) -> np.ndarray:
    """Returns (240,) int32 — pixel count that differs on each scanline."""
    assert a_idx.shape == b_idx.shape == (240, 256), (a_idx.shape, b_idx.shape)
    return (a_idx != b_idx).sum(axis=1).astype(np.int32)
```

Conventions:
- Drivers return `(rgb_frame, palette_idx_frame)` tuples. Normalization is
  a pure function, no emulator state.
- No abstract base class for drivers in Phase 1 — two concrete classes
  with a duck-typed `step(buttons: int) -> None` and `frame() -> np.ndarray`
  is enough. Add a Protocol if a third driver lands.
- Error messages name the tape, frame number, and divergence count:
  `contra_title @ frame 137: 14 scanlines diverged (240px total)`.

## Testing Strategy

- **Framework:** pytest, marker `@pytest.mark.parity` registered in
  `conftest.py`.
- **Location:** `tests/parity/test_tapes.py`, one parametrized test
  function iterating tapes in `tests/parity/tapes/`.
- **Determinism check:** each tape runs twice in the test; frame hashes
  must match between runs (catches hidden RNG leaks in nes_core).
- **Budget:** full suite <30s on M4 Max. Mario baseline ~600 frames should
  land under 5s per emulator; Contra/Zelda similar. If we blow the budget,
  truncate tapes rather than remove them.
- **Failure output:** on divergence, harness writes
  `tests/parity/failures/<tape>_<frame>_ours.png`,
  `_theirs.png`, and `_diff.png` (red pixels where indices differ). This
  is the artifact that tells me *which* scanline is wrong without re-running.
- **No coverage target** — this is a black-box regression suite, not a
  unit-tested module.

### Cross-emulator diff

`cross_emulator`-mode tapes: run both emulators, compare
`(240, 256, 3)` uint8 buffers with `(a != b).any(axis=2)` collapsed to
scanline sums. No normalization (both on Laines). Budget: <0.3ms/frame.

### Golden-hash diff

`golden_hash`-mode tapes: run nes_core only. Hash each frame with
FNV-1a 64-bit over the raw RGB bytes. Compare the live hash stream
against the committed `golden_hashes` blob frame-by-frame. First
mismatch triggers a PNG dump of the diverged frame against the golden
hash's expected hash (expected frame isn't stored — only the hash —
so `theirs.png` is omitted in golden mode; `ours.png` + a text file
naming expected-vs-actual hash is enough).

Storage cost: 600 frames × 8 bytes = 4.8 KB per tape. Committed.

### Palette parity guard

A single `test_palette_parity.py` asserts that nes_core and nes-py
produce byte-identical RGB on frame 30 of Mario Bros. with no inputs.
If either emulator's LUT ever shifts, this test fails first, loud, and
names the problem — avoiding a cryptic "all tapes failed" cascade.

## Boundaries

**Always:**
- Verify `rom_md5` before running any tape.
- Fail loudly on the first frame that exceeds tolerance; don't
  summarize-and-continue. Early exit keeps the suite fast.
- Check tape files into git. They are test fixtures.
- Dump PNG artifacts on failure. Nobody debugs divergence from a log line.

**Ask first:**
- Adding a fourth Phase 1 tape (keep scope tight).
- Raising a tape's tolerance above 0/0 — that's a policy decision.
- Modifying nes_core's public API (e.g., exposing the palette LUT) — it
  ripples into the trainer. Separate ADR if it's non-trivial.
- Switching reference emulator away from nes-py (Mesen, FCEUX trace logs).

**Never:**
- Couple tapes to save-state files. Mapper-specific state breaks fast.
- Mark a failing parity test `skip` to "unblock" anything. Parity failure
  means there's a real bug; fix it or file it, don't mute it.
- Embed nes-py binary dumps or ROMs in the repo. ROMs stay in `roms/`
  (already gitignored).

## Success Criteria

1. `make parity` runs in <30s on M4 Max, cold cache.
2. All three Phase 1 tapes pass on commit `b7f26d6`:
   - `mario_walk_right` passes byte-exact in `cross_emulator` mode
     against nes-py (empirically 0 drift over 600 frames).
   - `zelda_title_sprites` and `contra_title` pass in `golden_hash` mode
     against committed per-frame hashes recorded on `b7f26d6`.
3. Reverting the palette fix (Laines → Blargg in `video_sink.rs`) causes
   `mario_walk_right` to fail within the first 60 frames. Proves the
   cross-emulator mode catches palette regressions.
4. Re-introducing a synthetic 1px horizontal scroll offset in
   `ppu_neon::render_scanline` causes `contra_title` to fail on the
   first frame whose hash diverges, with the PNG dump showing the 1px
   shift against the golden reference frame.
5. A new tape can be authored end-to-end (record → commit → tests green)
   in <5 minutes. If tape authoring is painful, the harness dies of
   neglect.
6. Harness output on failure names (tape, frame, mode, divergence
   summary, PNG artifact path) in a single grep-friendly line.

## Open Questions

1. **nes-py input timing.** nes-py's `step(action)` advances one frame with
   the action held for the whole frame. nes_core's Python binding matches
   (verified via `scripted_input_bench.rs`). Confirm that mid-frame button
   toggles — which our existing `nes_core` API does not expose — are not
   needed for any Phase 1 tape. If they are, we widen the API.
2. **Frame counter alignment.** nes-py starts counting from post-BIOS; does
   nes_core? Need a 1-frame alignment probe in the first test run,
   documented in `tapes/README.md`.
3. **Palette LUT drift.** If `video_sink.rs` changes LUT again, every tape
   breaks until the extraction is re-run. Worth a
   `cargo test --features python` guard that re-exports the LUT and diffs
   against the committed `.npy`? Probably yes — add as a follow-up task.
4. **Phase 2 scope.** Mesen trace-log diff is much richer (per-instruction
   CPU + per-cycle PPU state) but needs Mesen binary + trace-log format
   parser. Defer unless Phase 1 misses a bug class it should've caught.

## Non-goals

- Audio parity. APU accuracy matters for RL reward signals, not for this
  visual regression harness. Separate spec if/when we need it.
- Full-library parity sweep (all 793 ROMs). Out of scope — per-title
  boot/palette checks already live in `tests/test_properties.py`.
- Performance benchmarking. `scripts/bench_vs_nes_py.py` owns that.
- GUI/Metal backend validation. This is a pure-CPU framebuffer comparison.
  Metal backend has its own verify path (`ppu_verify_parity.rs`).

## Phase 2 (deferred)

- Mesen trace-log driver (CPU register + PPU state stream compared cycle
  by cycle).
- Tape authoring via GUI record/replay (rather than a CLI that reads
  stdin).
- GitHub Actions workflow: `.github/workflows/parity.yml`, runs on PR.
- Frame-skip-N tapes (currently tapes pin `frame_skip=1`; verify that
  trainer-cadence `frame_skip=4` stays byte-exact).
- Golden-frame committed-PNG regression (diff against committed PNG, not
  just against live nes-py), for when we eventually retire nes-py.

## References

- `nes_core/examples/ppu_verify_parity.rs` — existing intra-nes_core
  verify path between `BatchedRenderMode::Off` and `::Replace`. Reuse the
  FNV-1a frame hashing pattern.
- `nes_core/examples/scripted_input_bench.rs` — canonical example of
  driving nes_core with a scripted per-frame input sequence.
- `scripts/bench_vs_nes_py.py` — shows both emulators imported in the
  same process; drivers in this harness will look similar.
- Memory: `project_palette_FIXED_2026-04-22.md`,
  `feedback_lockstep_nespy_for_fidelity.md` — both explicitly call out
  nes-py lockstep as the first move for any "graphics wrong" report.
  This harness institutionalizes that reflex.
