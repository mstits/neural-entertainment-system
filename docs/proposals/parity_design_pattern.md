# Parity Design Pattern

## Purpose

A repeatable workflow for taking nes_core from "mostly works" to "byte-exact with a reference emulator on every ROM in the library." Every correctness guarantee is a test; tests fail loudly when the guarantee breaks; the set of guarantees only grows.

## The Five Layers

Tests are stratified by what they assert. Pick the right layer for what you're trying to lock in.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 5: Library-wide bucket walls                                     │
│ test_library_buckets.py — every nes-py-comparable ROM (439 today)      │
│ stays in or improves its sweep-recorded bucket. Catches any            │
│ correctness regression anywhere in the library.                        │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Byte-exact fleet                                              │
│ test_byte_exact_fleet.py — every ROM that's currently 0-byte           │
│ divergence with nes-py MUST stay 0. Hardest correctness wall.          │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Lockstep ratchet                                              │
│ test_lockstep_baseline.py — 5 priority games (SMB, Zelda, Contra,      │
│ Mega Man, Metroid) with hand-tuned ceilings, fail-on-IMPROVEMENT       │
│ to force tightening when fixes land.                                   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Golden-hash tapes                                             │
│ test_tapes.py + tests/parity/tapes/*.json — 17 representative games    │
│ with frozen blake2b frame hashes. Catches any pixel-level render       │
│ regression. Fast (10s).                                                │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Gameplay-critical replays                                     │
│ test_zelda_input_replay.py (template) — scripted button sequence       │
│ + assertion on a small set of game-state RAM bytes (sword acquired,    │
│ Mario reached 1-2, hearts gained, etc.). xfail(strict=True) while      │
│ broken so the fix is announced by the test xpassing.                   │
└────────────────────────────────────────────────────────────────────────┘

Foundation: tests/parity/lockstep.py is the shared diff harness.
Foundation: parity_sweep.json is the ground-truth ledger.
```

## The Workflow

### When a parity gap is discovered (game misbehaves vs nes-py)

1. **Reproduce in the harness.**
   ```python
   from tests.parity.lockstep import run_lockstep, idle_tape
   div = run_lockstep("roms/<game>.nes", idle_tape(N))
   print(summarize(div))
   ```
   Get a specific frame + addresses where the divergence first appears.

2. **Add a Layer-1 test** that asserts the SPECIFIC game-state property
   that's broken. Use `test_zelda_input_replay.py` as the template:
   - Scripted input sequence (legacy `.state.bin` action-replay format works).
   - Assert on a handful of named RAM addresses (sword inventory, Link X, etc.).
   - `xfail(strict=True)` while broken — fix announces itself.

3. **Localize the cause.** The lockstep harness's `Divergence.diff_addrs`
   tells you what bytes differ first. Trace those addresses' write-paths in
   `nes_core/src/cpu.rs` (or PPU, mapper) using temporary `eprintln!` probes.

4. **Apply the smallest fix that closes the gap.** Re-run all 5 layers of
   tests; nothing should regress, and Layer 1 should xpass (now flips to
   pass — remove the xfail marker and commit).

5. **Re-run `scripts/parity_sweep.py`.** ROMs that improved bucket should
   tighten the Layer-5 ceiling. Commit the new `parity_sweep.json`.

### When a fix lands and you want to verify no collateral damage

```bash
# Fast (~30s): the 17 hand-curated representative games + 5-game baselines
make parity

# Full (~3min): every nes-py-comparable ROM in the library
pytest tests/parity/ -q --timeout=300
```

### When a regression slips in

The first failing test tells you exactly what broke:

| Failing layer | Means |
|---------------|-------|
| Layer 5 | Some ROM moved into a worse bucket. Check `parity_sweep.json` to see what was expected vs what's now happening. |
| Layer 4 | A previously-byte-exact ROM is no longer byte-exact. Critical regression — bisect immediately. |
| Layer 3 | Lockstep baseline changed. Test message says was X now Y; investigate. |
| Layer 2 | Pixel render output changed for a representative game. Check `tests/parity/failures/` PNG dumps. |
| Layer 1 | Gameplay-critical RAM byte no longer matches expected. Game logic regressed. |

## Why this pattern works

- **No silent loss of correctness.** Every committed guarantee fires when broken.
- **No premature pessimism.** Bucket ceilings have headroom to absorb noise without weakening the regression signal.
- **Improvement is auto-rewarded.** Fail-on-improvement on Layer 3 forces ceiling tightening when work lands.
- **Scales.** Adding more correctness signals = more rows in `parity_sweep.json` = more autogenerated tests.
- **No dependency on perfect parity.** Even if total cycle-accuracy is multi-month work, the pattern keeps every ROM at its CURRENT correctness level today.

## Concrete next applications

Templates for adding new Layer-1 tests as the structural CPU work lands:

- `test_mario_reaches_1_2.py` — scripted "press Start, run right, jump on goombas" → assert player progressed to world 1-2 ($075F or whatever the address is).
- `test_megaman_kills_cutman.py` — Mega Man is byte-exact, so a scripted input that beats Cutman's level should produce identical RAM to nes-py at the level-clear screen.
- `test_metroid_first_morph_ball.py` — drive through Brinstar to morph ball pickup.

Each new Layer-1 test is ~50 lines: load ROM, script input, assert. Pattern doesn't change — just the named addresses.

## See also

- `docs/proposals/parity_coverage_map.md` — current bucket distribution.
- `docs/proposals/archive/cycle_accuracy_plan.md` — ranked list of structural fixes.
- `docs/proposals/archive/parity_and_perf_plan.md` — phase plan for the remaining work.
- `parity_sweep.json` — the ground-truth ledger (one entry per ROM).
