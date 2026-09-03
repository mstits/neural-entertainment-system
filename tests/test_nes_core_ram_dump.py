"""Gates on `scripts/tracing/nes_core_ram_dump.py`, the generalized
Mesen-parity RAM dump.

Two of these encode the tape-construction rules whose absence produced a
withdrawn parity verdict on 2026-09-01: the input phase
(`nes_core/src/python.rs:340` plus the Lua `inputPolled` phase) and the
no-op materialize block (`src/training/tape_replay.py:249`). Two more
are the byte-identity check against the Castlevania-only script this
generalizes, and the ten-second lineage check the ruling asks for before
any tape is fed to a reference emulator. The rest pin the refusals, so
no phase or framing offset can be guessed silently.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.tracing.nes_core_ram_dump import (  # noqa: E402
    align_feed,
    build_parser,
    emit_tape,
    materialize_tape,
    validate_args,
)

SCRIPT = REPO / "scripts" / "tracing" / "nes_core_ram_dump.py"
CV_SCRIPT = REPO / "scripts" / "tracing" / "nes_core_cv_ram_dump.py"
CV_ROM = REPO / "roms" / "Castlevania (USA).nes"

#: The five flags `nes_core_cv_ram_dump.py` sets before reset, plus the
#: `mmio_read_timing` it sets after. Measured 2026-09-02: applying all
#: six before reset reproduces that script byte for byte on this tape.
CV_FLAGS = ("reset_alignment,dmc_stall_timing,frame_anchor,"
            "nmi_poll_timing,mmio_write_timing,mmio_read_timing")

ROW = 2048


def _tape(n: int, seed: int = 17) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.choice([0, 1, 2, 8, 0x80, 0x40, 0x81]) for _ in range(n))


def _rows(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    assert len(raw) % ROW == 0, f"{path} is not a whole number of rows"
    return [raw[i:i + ROW] for i in range(0, len(raw), ROW)]


def _args(*extra: str):
    return build_parser().parse_args(
        ["--rom", "x.nes", "--tape", "t.bin", *extra])


def test_materialize_inserts_the_noop_block_at_the_boundary() -> None:
    """Rule 2: `TapePlayer.play()` runs one no-op step between the root
    state and the first action (`src/training/tape_replay.py:249`), four
    real frames on the default profile. A tape spliced from a cold-boot
    prefix straight into action 0 is missing them, and every later input
    lands four frames early."""
    tape = bytes([9, 9, 9, 7, 7])
    out = materialize_tape(tape, 4, 3)
    assert out == bytes([9, 9, 9, 0, 0, 0, 0, 7, 7])
    assert len(out) == len(tape) + 4, "the block extends the tape"
    assert materialize_tape(tape, 0, 3) == tape, "count 0 is a no-op"
    with pytest.raises(ValueError):
        materialize_tape(tape, 4, len(tape) + 1)


def test_materialize_at_one_is_the_first_offset_after_the_root_frame() -> None:
    """`main()` dumps tape byte 0 as row 0 and steps bytes 1..N, so the
    stepped feed is `materialize_tape(tape, n, at)[1:]`. At `at=1` all
    `n` no-ops survive into the feed and the root byte stays the root.
    At `at=0` the block goes in front of the root byte: one no-op is
    eaten as row 0 and the root byte becomes a stepped input, which is
    exactly the one-frame shift this script exists to prevent."""
    tape = bytes([5, 6, 7, 8])
    assert materialize_tape(tape, 4, 1)[1:] == bytes([0, 0, 0, 0, 6, 7, 8])
    assert materialize_tape(tape, 4, 0)[1:] == bytes([0, 0, 0, 5, 6, 7, 8])
    assert build_parser().parse_args(
        ["--rom", "x.nes", "--tape", "t.bin"]).materialize_at == 1


def test_emit_tape_puts_the_input_phase_ahead_of_the_materialized_tape() -> None:
    """Rule 1: an `env.reset()` lineage advances one frame before the
    first input (`nes_core/src/python.rs:340`) and the Mesen Lua applies
    a byte the frame after `inputPolled`, so the reference is fed the
    tape two frames late. The phase EXTENDS the tape (the reference runs
    from its own power-on); it does not overwrite its first bytes."""
    tape = bytes([9, 9, 7, 7])
    assert emit_tape(tape, 2) == bytes([0, 0, 9, 9, 7, 7])
    assert emit_tape(tape, 0) == tape
    both = emit_tape(materialize_tape(tape, 4, 2), 2)
    assert both == bytes([0, 0, 9, 9, 0, 0, 0, 0, 7, 7])
    assert len(both) == len(tape) + 4 + 2


def test_mesen_align_delays_every_byte_and_keeps_the_row_count() -> None:
    """The other half of rule 1, applied to our own dump instead of the
    reference's tape: `nes_core_cv_ram_dump.py`'s one-frame
    `--mesen-align`, generalized past one frame. Row count is preserved
    so the two dumps stay comparable row for row."""
    feed = bytes([1, 2, 3, 4])
    assert align_feed(feed, 1) == bytes([0, 1, 2, 3])
    assert align_feed(feed, 2) == bytes([0, 0, 1, 2])
    assert align_feed(feed, 0) == feed
    assert len(align_feed(feed, 3)) == len(feed)
    assert align_feed(feed, 9) == bytes([0, 0, 0, 0])


def test_frame_skip_above_one_requires_one_byte_per_action() -> None:
    """A lineage tape at `frame_skip=N` holds one byte per action; a
    Mesen tape holds one per real frame. Guessing between them drops or
    duplicates frames, so the script refuses rather than picking."""
    with pytest.raises(SystemExit) as err:
        validate_args(_args("--frame-skip", "4"))
    assert "--one-byte-per-action" in str(err.value)
    ok = _args("--frame-skip", "4", "--one-byte-per-action")
    assert ok.frame_skip == 4 and ok.one_byte_per_action
    validate_args(ok)


def test_mesen_align_and_emit_phase_are_refused_together() -> None:
    """Both shift the same phase, one on our dump and one on the
    reference's tape. Applying both double-counts it."""
    with pytest.raises(SystemExit) as err:
        validate_args(_args("--mesen-align", "2", "--emit-phase", "2"))
    assert "Pick one" in str(err.value)


def test_emit_tape_refuses_to_guess_the_input_phase() -> None:
    """`--emit-tape` writes the bytes a reference emulator is fed, so a
    defaulted phase is a silently wrong reference run. The 2026-09-01
    SMB receipt shipped a reproduce command that leaned on the phase
    default and missed the pinned alignment by two frames; the phase is
    now stated or the run does not start."""
    with pytest.raises(SystemExit) as err:
        validate_args(_args("--emit-tape", "m.bin"))
    assert "--emit-phase" in str(err.value)
    validate_args(_args("--emit-tape", "m.bin", "--emit-phase", "0"))
    validate_args(_args("--emit-tape", "m.bin", "--emit-phase", "2"))


def test_materialize_at_zero_is_refused() -> None:
    """The offset that reads as "the beginning" is the one that eats the
    root frame. Refuse it by name rather than shifting every later input
    by one frame, which is the artifact the 2026-09-01 ruling withdrew a
    verdict over."""
    with pytest.raises(SystemExit) as err:
        validate_args(_args("--materialize", "4", "--materialize-at", "0"))
    assert "root frame" in str(err.value)
    validate_args(_args("--materialize", "4"))
    validate_args(_args("--materialize", "0", "--materialize-at", "0"))


@pytest.mark.slow
def test_dump_matches_the_castlevania_script_byte_for_byte(tmp_path) -> None:
    """This script generalizes `nes_core_cv_ram_dump.py`. On that
    script's ROM and flag set the two dumps must be identical, or the
    generalization changed behaviour instead of widening it."""
    if not CV_ROM.exists():
        pytest.skip(f"ROM missing: {CV_ROM}")
    tape = tmp_path / "tape.bin"
    tape.write_bytes(_tape(300))
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    subprocess.run([sys.executable, str(CV_SCRIPT), "--tape", str(tape),
                    "--out", str(old)], check=True, cwd=REPO)
    subprocess.run([sys.executable, str(SCRIPT), "--rom", str(CV_ROM),
                    "--tape", str(tape), "--out", str(new),
                    "--hw-flags", CV_FLAGS], check=True, cwd=REPO)
    assert old.read_bytes() == new.read_bytes()
    assert len(_rows(new)) == 300


@pytest.mark.slow
def test_materialize_through_main_matches_a_hand_materialized_tape(
        tmp_path) -> None:
    """The end-to-end half of rule 2. `--materialize 4` at the default
    offset must be exactly the tape with four no-ops spliced in after
    the root byte: same rows, same count, and the emitted reference tape
    carries an unbroken run of four no-ops at offset 1."""
    if not CV_ROM.exists():
        pytest.skip(f"ROM missing: {CV_ROM}")
    tape = _tape(60, seed=5)
    plain, hand = tmp_path / "plain.bin", tmp_path / "hand.bin"
    plain.write_bytes(tape)
    hand.write_bytes(materialize_tape(tape, 4, 1))
    auto_out, hand_out = tmp_path / "auto.bin", tmp_path / "hand_out.bin"
    emitted = tmp_path / "emitted.bin"
    subprocess.run([sys.executable, str(SCRIPT), "--rom", str(CV_ROM),
                    "--tape", str(plain), "--out", str(auto_out),
                    "--materialize", "4",
                    "--emit-tape", str(emitted), "--emit-phase", "0"],
                   check=True, cwd=REPO)
    subprocess.run([sys.executable, str(SCRIPT), "--rom", str(CV_ROM),
                    "--tape", str(hand), "--out", str(hand_out)],
                   check=True, cwd=REPO)
    assert auto_out.read_bytes() == hand_out.read_bytes()
    assert len(_rows(auto_out)) == len(tape) + 4
    feed = emitted.read_bytes()
    assert feed[1:5] == bytes(4), "four no-ops, unbroken, after the root"
    assert feed[5] == tape[1], "the first real input follows the block"


@pytest.mark.slow
def test_a_lineage_tape_replays_byte_exact_at_frame_skip_one(tmp_path) -> None:
    """The ten-second check the 2026-09-01 ruling asks for before any
    tape reaches a reference emulator: the same actions stepped natively
    at `frame_skip=4` and expanded to one byte per real frame at
    `frame_skip=1` must agree on every lineage row. A tape that cannot
    reproduce its own lineage says nothing about a second emulator.

    It also holds the retraction in place: `frame_skip=4` native
    stepping IS 4x `frame_skip=1` with the same mask, contra the
    withdrawn `mesen-parity-second-game.md` section 3.
    """
    if not CV_ROM.exists():
        pytest.skip(f"ROM missing: {CV_ROM}")
    actions = _tape(75)
    lineage = tmp_path / "lineage.bin"
    lineage.write_bytes(bytes([0]) + actions)
    expanded = tmp_path / "expanded.bin"
    expanded.write_bytes(bytes([0]) + b"".join(bytes([a]) * 4
                                               for a in actions))
    fs4, fs1 = tmp_path / "fs4.bin", tmp_path / "fs1.bin"
    subprocess.run([sys.executable, str(SCRIPT), "--rom", str(CV_ROM),
                    "--tape", str(lineage), "--out", str(fs4),
                    "--frame-skip", "4", "--one-byte-per-action"],
                   check=True, cwd=REPO)
    subprocess.run([sys.executable, str(SCRIPT), "--rom", str(CV_ROM),
                    "--tape", str(expanded), "--out", str(fs1)],
                   check=True, cwd=REPO)
    native, every_frame = _rows(fs4), _rows(fs1)
    assert len(native) == 76 and len(every_frame) == 301
    assert native == every_frame[::4]


def test_the_mesen_cross_check_row_cites_a_receipt_that_exists() -> None:
    """The coverage map's Mesen section carries public parity numbers.
    Each row names a receipt file, and that file has to be in the tree:
    a public number whose receipt went missing is the failure this
    repo's ledger keeps recording."""
    doc = (REPO / "docs" / "proposals" / "parity_coverage_map.md").read_text()
    section = doc.split("## Mesen 2 cross-checks under real input")[-1]
    cited = [c for c in re.findall(r"`([^`]+)`", section)
             if c.startswith("docs/receipts/")]
    assert cited, "the Mesen section names no receipt file"
    for path in cited:
        assert (REPO / path).exists(), f"receipt missing from the tree: {path}"
