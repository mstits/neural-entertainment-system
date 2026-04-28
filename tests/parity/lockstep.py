"""Lockstep nes_core vs nes-py RAM-diff harness.

Both emulators run from cold boot with identical per-frame button masks.
After each frame we compare 2 KB of CPU RAM ($0000-$07FF). First frame
with any divergence is reported along with the specific addresses that
differ — points directly at a CPU/PPU/mapper timing disagreement.

Use when:
  - A game shows different behavior in nes_core vs nes-py (stuck cutscenes,
    missing sprites, wrong palette, RNG desync).
  - A parity-harness golden_hash test regresses but cross_emulator mode
    doesn't apply because the tape includes state transitions.
  - Before investigating any "graphics wrong" report, per
    `feedback_lockstep_nespy_for_fidelity.md`.

Pixel diffs can be masked by a small timing skew (frame N-1 vs frame N);
RAM diffs are absolute. A one-cycle CPU disagreement propagates to RAM
within ~300 instructions, so a 60-fps diff is essentially a 60×-per-second
bug detector.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass
class Divergence:
    frame: int
    ours_ram: bytes            # 2048 bytes of nes_core RAM
    theirs_ram: bytes          # 2048 bytes of nes-py RAM
    diff_addrs: list[int]      # addresses in $0000-$07FF that differ
    context: dict              # extra fields (action at frame, cycle, etc.)


def _load_ours(rom: str):
    import nes_core
    env = nes_core.NESEnvironment(str(rom), frame_skip=1)
    env.reset()
    return env


def _load_theirs(rom: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import nes_py
    env = nes_py.NESEnv(str(rom))
    env.reset()
    # nes_core's NESEnvironment.reset() advances 1 frame (so the GUI's
    # Play window has a valid first image); nes-py's NESEnv.reset()
    # does not. Compensate here so the two emulators are phase-aligned
    # for RAM / frame diffs. Removing the reset-advance to match nes-py
    # instead was tried and broke the GUI (user-visible black frame);
    # keeping it in the live code and compensating in tests is the
    # least-invasive path.
    env.step(0)
    return env


def _ram_ours(env) -> bytes:
    # nes_core's get_ram_range returns a numpy uint8 array for [start, end).
    return bytes(env.get_ram_range(0, 0x0800))


def _ram_theirs(env) -> bytes:
    return bytes(env.ram)


def run_lockstep(
    rom_path: str | Path,
    actions: Sequence[int],
    *,
    max_frames: int | None = None,
    compare_from: int = 0,
    mask_addrs: Iterable[int] = (),
) -> Divergence | None:
    """Step both emulators with `actions[i]` on frame i and compare RAM
    at end of each frame. Returns the first `Divergence` found, or None
    if the run completed with no divergence.

    `compare_from` — skip RAM comparison for frames [0, compare_from).
      Use when cold-boot RNG is expected to diverge before first input.
    `mask_addrs` — iterable of $0000-$07FF addresses whose bytes are
      ignored (e.g. known non-deterministic counters). Empty by default.
    """
    rom = str(rom_path)
    ours = _load_ours(rom)
    theirs = _load_theirs(rom)
    mask = set(mask_addrs)
    total = len(actions) if max_frames is None else min(len(actions), max_frames)

    for i in range(total):
        a = int(actions[i]) & 0xFF
        ours.step(a)
        theirs.step(a)
        if i < compare_from:
            continue
        a_ram = _ram_ours(ours)
        b_ram = _ram_theirs(theirs)
        if a_ram == b_ram:
            continue
        # Collect diff addresses, skipping masked bytes.
        diffs = [
            addr for addr in range(0x0800)
            if addr not in mask and a_ram[addr] != b_ram[addr]
        ]
        if not diffs:
            continue
        return Divergence(
            frame=i,
            ours_ram=a_ram,
            theirs_ram=b_ram,
            diff_addrs=diffs,
            context={"action": a},
        )
    return None


def summarize(d: Divergence, label: str = "") -> str:
    """Human-readable one-shot summary — prints the first divergent
    frame, first few diff addrs, and the specific byte values for each.
    """
    lines = [
        f"{label}FIRST DIVERGENCE at frame {d.frame} (action={d.context.get('action',0):#04x})",
        f"  {len(d.diff_addrs)} address(es) differ in $0000-$07FF",
    ]
    for addr in d.diff_addrs[:20]:
        lines.append(
            f"    ${addr:04X}: ours=0x{d.ours_ram[addr]:02X}  theirs=0x{d.theirs_ram[addr]:02X}"
        )
    if len(d.diff_addrs) > 20:
        lines.append(f"    ... +{len(d.diff_addrs)-20} more")
    return "\n".join(lines)


def idle_tape(n: int) -> list[int]:
    """All-zero button mask, n frames. For 'does the emulator free-run
    identically to nes-py' sanity checks."""
    return [0] * n


def load_legacy_replay(path: str | Path) -> list[int]:
    """Read a legacy action-replay .state.bin — raw u8-per-frame button
    sequence produced by the old recorder. Used as a deterministic
    input source when no state API is shared between emulators."""
    return list(Path(path).read_bytes())
