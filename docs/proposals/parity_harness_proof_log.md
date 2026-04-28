# Parity Harness Proof Log

Records of the Phase 4 regression proofs for `tests/parity/` — each
entry confirms that a known class of nes_core regression produces a
clear, actionable failure signal from `make parity`.

---

## T13 — Palette perturbation (2026-04-22)

**Mechanism.** Scratch branch `parity-proof-palette`. Modified
`nes_core/src/sink/video_sink.rs` to swap R and B channels across all
64 entries of `XRGB8888_PALETTE`. Rebuilt (`make build`, copied
`target/release/libnes_core.dylib` into venv site-packages — see
caveat below), then ran `make parity`.

**Result.** 7 failures / 33 passes, 6.7s:

| Failure | First signal |
|---|---|
| `test_laines_palette_matches_between_emulators` (T2 guard) | Names the expected vs actual RGB at sample pixel (100,128), which is where the perturbed palette diverges from nes-py's Laines LUT. Fires before any tape runs — exactly the "fail first, fail loud" role it was designed for. |
| `mario_walk_right` (cross_emulator) | frame 10: 98 scanlines diverged, max 186 px/scanline. Diff PNG shows the title screen with swapped color channels — Mario-red still red (because R=0xFC in perturbed entry 1 was originally B=0xFC), but the "MARIO BROS." logo's blue outline is gone. Visually unmistakable. |
| `contra_title` (golden_hash) | Hash mismatch within the warmup window — expected vs actual 64-bit BLAKE2b diverge after the perturbed palette shifts every rendered byte. |
| `zelda_title_sprites` (golden_hash) | frame 36: `expected 56bc7afe4eb9b3f0 got fc0d60e2f6c9fad8`, `ours.png` dumped alongside expected/actual hash in `*_hash.txt`. |
| Three driver/author tests | Duplicates of the above invariants — all trip on the R/B-swap. |

**Verdict.** All three Phase 1 tapes catch the palette perturbation.
The `test_palette_parity` guard fires first, as designed. Cross-emulator
mode gives the most information (PNG overlay); golden-hash mode still
names the exact frame + expected/actual hash. Harness works.

**Caveat (logged for future reference).** `maturin develop --release`
on this machine did NOT actually replace the installed `.so` in the
venv — `nes_core/target/release/libnes_core.dylib` had to be `cp`'d
manually to `.venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so`
for Python to pick up the rebuild. Without this step the perturbation
is invisible and parity appears to pass. Worth tracking down
separately; filed as a known gotcha, not a harness bug.

**Cleanup.** Master's `video_sink.rs` restored via `git checkout --`,
rebuilt cleanly, `make parity` → 40/40 green. Scratch branch
`parity-proof-palette` never received a commit (perturbation lived only
in the working tree) and was implicitly discarded on `git checkout master`.
Remaining branch pointer deleted with `git branch -D`.

---

## T14 — Scroll-offset perturbation (2026-04-22)

**Mechanism.** Scratch branch `parity-proof-scroll`. First attempt
perturbed `fine_x` in `nes_core/src/ppu_neon.rs` — tapes passed,
because `ppu_neon` is opt-in (`BatchedRenderMode::Replace`) and the
default `Off` routes through `nes_core/src/ppu.rs::background_pixel`.
Lesson: the default PPU path is canonical, not `ppu_neon.rs`. Moved
the perturbation to `ppu.rs`:

```rust
// In background_pixel():
let fine_x = (self.regs.x + 1).min(7);
```

Rebuilt, manually copied the dylib into venv (same maturin caveat as
T13), ran `make parity`.

**Result.** 7 failures / 33 passes, 6.6s. Same three tapes fail as T13.
Specific to this proof:

| Signal | Detail |
|---|---|
| `contra_title` first failure | frame 40, hash mismatch, `expected=37cfae0000dcf692 actual=4e2136abcbdc74a3`. |
| Visual confirmation via `inspect.py --frame 200` | Contra title "CONTRA" logo and letter outlines visibly offset 1px to the right on the perturbed build (ours/left) vs nes-py (theirs/right). |
| `mario_walk_right` (cross_emulator) | Also fails — a 1px BG shift against nes-py is immediate divergence. Expected bonus catch. |
| `zelda_title_sprites` (golden_hash) | Also fails at frame 38 — name-entry screen BG tiles shift under fine_x perturbation. |

**Verdict.** Contra tape catches the exact regression class the spec
calls out (1px horizontal scroll drift), and `inspect.py` gives a
side-by-side visual proof. Both diff modes work as designed.

**Cleanup.** `ppu.rs` restored via `git checkout --`, rebuilt,
`make parity` → 40/40 green. Scratch branch deleted (no commits
landed on it).

---

## Lessons

1. **When perturbing nes_core for proof, target `ppu.rs` not
   `ppu_neon.rs`.** The batched NEON kernel is opt-in via
   `pool.set_batched_render_mode("replace")`; the default runs through
   `ppu.rs`. Proofs against `ppu_neon.rs` alone do nothing visible.

2. **`maturin develop --release` does not reliably update the
   installed `.so`.** Whether this is a cache hash or uv editable-install
   quirk is untracked. Workaround:
   `cp nes_core/target/release/libnes_core.dylib
   .venv/lib/python3.11/site-packages/nes_core/nes_core.abi3.so`
   after every `make build`. Worth fixing upstream but out of scope
   for this harness.

3. **The T2 palette-parity guard fires before any tape runs when the
   LUT drifts.** Confirmed in T13. That was the designed behavior and
   it delivered — saves ~5 seconds of tape-runtime confusion when the
   real problem is a palette shift.

