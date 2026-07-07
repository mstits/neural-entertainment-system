"""Mesen-as-oracle parity tests.

Compares per-frame internal RAM ($0000-$07FF) of nes_core against a
pre-captured Mesen tape over 120 cold-boot idle frames. Mesen is the
gold-standard NES emulator reference — much more accurate than nes-py
(which wraps LaiNES, with documented timing shortcuts).

Why this exists alongside the nes-py-based parity tests
(`test_byte_exact_fleet.py`, `test_library_buckets.py`):

    For some ROMs (e.g. Tom Sawyer, Flight of the Intruder, California
    Games), nes_core is 4-7× CLOSER to Mesen than nes-py is. The
    nes-py-based tests would mark us as "regressed" while we're
    actually MORE correct vs the reference. This test files the
    correctness ground truth.

Capture a new tape with:
    scripts/parity/capture_mesen_tape.sh "<rom_basename>" 120

Tape format: raw concatenated 2KB RAM dumps, one per frame
(`tests/parity/mesen_tapes/<rom_basename>.bin`, 120 * 2048 bytes).

Ceilings are derived from the 2026-04-26 baseline measurement (max
divergence across all 120 frames) plus a small headroom for natural
variance. As nes_core's PPU/CPU accuracy improves, ceilings tighten.
A ROM going significantly OVER its ceiling is a regression to
investigate. A ROM going significantly UNDER (e.g., to 0) is a
correctness improvement worth tightening for.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TAPES_DIR = REPO / "tests" / "parity" / "mesen_tapes"
FRAMES = 120

# (rom_basename, max_byte_ceiling_over_120_frames)
# Baseline: 2026-04-26, asm_cpu disabled, after open-bus warmup fix
# (commits ed440e9 + 6a4304d).
#
# Ceiling = baseline max + 5-byte headroom. Most divergence is from
# the 8-cycle PPU vblank-anchor offset vs Mesen's reset-cycle modeling
# (task #38) — fixing that would drop most ceilings significantly.
#
# Zelda is the outlier (max 263) because its title-screen sprite list
# is rebuilt every frame and the cycle-anchor drift causes a
# snowballing position offset until a stable equilibrium settles.
#
# 2026-04-26 update: tightened after the 8-cycle CPU-reset modeling
# (Nes::reset advances PPU by 8 CPU cycles via tick_cpu_cycles_discard)
# AND switching the Mesen capture from `endFrame` (mid-vblank, before
# NMI) to `startFrame` (vblank end, NMI complete) to match our
# env.step() return point. Most ceilings dropped 2-4×; Mega Man went
# fully byte-exact.
CASES = [
    # 2026-04-26 STA $4014 deferred-write fix (commit pending) tightened
    # EVERY ceiling by ~5 bytes — the consistent OAM-DMA dummy_read
    # parity drift between cycle_zero_early_commit and real-hardware-
    # style late-commit was eliminated. Slow path now matches real HW
    # (and ASM CPU) on DMA timing.
    ("Super Mario Bros. (World).nes",                   12),
    ("Contra (USA).nes",                                 6),
    ("Legend of Zelda, The (USA) (Rev A).nes",         263),  # title sprite spike
    ("Mega Man (USA).nes",                               0),  # BYTE EXACT vs Mesen
    ("Metroid (USA).nes",                                9),
    ("California Games (USA).nes",                      17),
    ("Adventures of Tom Sawyer (USA).nes",              17),
    ("Barbie (USA) (Rev A).nes",                        21),
    ("Flight of the Intruder (USA).nes",                16),
    ("Sqoon (USA).nes",                                  9),
    ("Super Mario Bros. 2 (USA) (Rev A).nes",            7),
    ("Super Mario Bros. 3 (USA) (Rev A).nes",          321),  # MMC3 IRQ timing
    ("Castlevania (USA) (Rev A).nes",                   15),  # +1 with asm_cpu
    ("Castlevania II - Simon's Quest (USA).nes",         7),  # +3 with asm_cpu
    ("Castlevania III - Dracula's Curse (USA).nes",    630),  # MMC5 timing + audio
    ("Mega Man 2 (USA).nes",                            12),
    ("Mega Man 3 (USA).nes",                           444),  # MMC3 IRQ timing
    ("Battletoads (USA).nes",                           12),
    ("Ninja Gaiden (USA).nes",                           8),
    ("Punch-Out!! (USA).nes",                            8),  # MMC2
    ("DuckTales (USA).nes",                             18),
    ("Final Fantasy (USA).nes",                          2),
    # MMC1 SUROM 512 KB (PRG-A18 from CHR bank bit 4), Mesen-validated:
    # measured max divergence vs Mesen at 120 idle frames on the shipping
    # (asm-on) build — DW3=370, DW4=5. DW3's residual is a pre-existing
    # non-banking CPU/PPU accuracy gap (boot seeds from uninit RAM $06F0),
    # not reachable by mapper work; ceilings carry headroom.
    ("Dragon Warrior III (USA).nes",                   380),  # SUROM 512KB
    ("Dragon Warrior IV (USA).nes",                     12),  # SUROM 512KB
    ("Tetris (USA).nes",                                12),
    ("Excitebike (Japan, USA).nes",                     18),
    ("Ice Climber.nes",                                 20),
    ("Donkey Kong.nes",                                 18),
    ("Gradius (USA).nes",                                5),
    ("Kirby's Adventure (USA) (Rev A).nes",             19),  # +2 with asm_cpu
    ("Crystalis (USA).nes",                            265),  # MMC3 IRQ timing
    ("Rygar (USA).nes",                                  6),
    ("Ghosts'n Goblins (USA).nes",                      22),
]


@pytest.fixture(autouse=True)
def _silence_gym():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _max_divergence(rom_basename: str, mesen_tape_path: Path) -> tuple[int, int]:
    """Return (max_diff_over_frames, last_frame_diff) ours vs Mesen."""
    import nes_core
    rom = REPO / "roms" / rom_basename
    mesen = mesen_tape_path.read_bytes()
    n_frames = len(mesen) // 2048
    env = nes_core.NESEnvironment(rom_path=str(rom), frame_skip=1)
    # reset_no_advance leaves CPU at cycle 7 / PPU at cycle 25,
    # matching Mesen's first endFrame state. Standard reset() would
    # auto-advance 1 frame and shift our snapshot off by one.
    env.reset_no_advance()
    max_diff = 0
    last_diff = 0
    for f in range(n_frames):
        env.step(0)
        a = bytes(env.get_ram(i) for i in range(0x800))
        b = mesen[f * 2048:(f + 1) * 2048]
        d = sum(1 for i in range(0x800) if a[i] != b[i])
        max_diff = max(max_diff, d)
        last_diff = d
    return max_diff, last_diff


@pytest.mark.parity
@pytest.mark.parametrize("rom, ceiling", CASES, ids=[c[0] for c in CASES])
def test_mesen_ram_divergence_within_ceiling(rom: str, ceiling: int):
    """nes_core stays within `ceiling` bytes of Mesen at any of the
    first 120 cold-boot idle frames."""
    rom_path = REPO / "roms" / rom
    if not rom_path.exists():
        pytest.skip(f"ROM missing: {rom}")
    tape_path = TAPES_DIR / f"{rom}.bin"
    if not tape_path.exists():
        pytest.skip(
            f"Mesen tape missing: {tape_path}. Capture with "
            f"`scripts/parity/capture_mesen_tape.sh \"{rom}\" {FRAMES}`"
        )

    max_diff, last_diff = _max_divergence(rom, tape_path)

    # Tighten signal: regression OR improvement should both surface.
    assert max_diff <= ceiling, (
        f"{rom}: peaked at {max_diff} bytes diverged from Mesen "
        f"(last frame {last_diff}), exceeds ceiling {ceiling}. "
        f"Investigate recent CPU/PPU/mapper changes. To re-baseline "
        f"a true improvement, tighten the ceiling and commit."
    )
