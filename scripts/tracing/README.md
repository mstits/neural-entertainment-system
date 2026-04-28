# Side-by-side emulator tracing

Tools for diffing nes_core behavior against an instrumented nes-py,
for bugs where RAM-level lockstep tests can't distinguish the failing
path (e.g., PPU/CPU timing divergences that don't manifest as byte
diffs until many frames later).

## Instrumented nes-py

Source: `/tmp/nespy_instr/nes_py-8.2.1` (copied from the `pip download`
of nes_py-8.2.1 for research purposes).

Instrumented files:

- `nes_py/nes/src/emulator.cpp`: per-frame FRAME_START / FRAME_END
  banners including `$06`, `$11`, `$84`, `$70`, `$657` snapshots
- `nes_py/nes/src/cpu.cpp`: `NMI_ENTRY` log including PC and SP
- `nes_py/nes/src/ppu.cpp`: `PPU_STATUS` read log including returned
  byte and vblank/sprite-0 state

Enable via `NESPY_TRACE=1` env var; output goes to stderr.

Build + install (run from the repo root):

```bash
REPO=$(git rev-parse --show-toplevel)
cd /tmp/nespy_instr
"$REPO"/.venv/bin/python setup.py build_ext --inplace
cp nes_py/lib_nes_env.cpython-311-darwin.so \
   "$REPO"/.venv/lib/python3.11/site-packages/nes_py/lib_nes_env.cpython-311-darwin.so
codesign --force --sign - \
   "$REPO"/.venv/lib/python3.11/site-packages/nes_py/lib_nes_env.cpython-311-darwin.so
```

To restore the vanilla nes-py, backup was saved at
`/tmp/lib_nes_env.original.so.bak` (during the cave-stuck session,
2026-04-24) — copy it back to the same site-packages path.

## Scripts

- `nespy_tape_trace.py` — replay the 3354-frame Zelda cave tape
  through instrumented nes-py, trace to stderr (redirect externally).
- `nes_core_nmi_trace.py` — same tape through nes_core with
  per-instruction stepping around the divergence window, detects
  NMI service via PC jump to $E45B.

## Use case — cave-stuck investigation (2026-04-24)

Confirmed nes-py's NMI fires at vblank-edge of its frame 1013; ours
fires at vblank-edge of ppu_frame=1013 (one emulated frame earlier
relative to the game code). This happens because our CPU reaches
the `$2000` NMI-enable write *before* vblank of that frame while
nes-py's reaches it *after*. Root cause is the structural CPU↔PPU
step-boundary difference: LaiNES/nes-py runs fixed 29781 CPU cycles
per frame, we run instructions until `frame_written` (±5 cycles per
frame).

The mid-vblank-PPUCTRL-quirk fix attempted in write_ppu_ctrl was
NOT the issue — at the moment of the write, `nmi_occurred` is
already false (game has read $2002 first). The problem is timing
of the write relative to vblank, not register state at write time.
