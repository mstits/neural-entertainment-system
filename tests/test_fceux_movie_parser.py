"""Pin behavior of the FCEUX movie parser for both .fm2 and .fm3.

FM3 differs from FM2 only by adding optional auxiliary sections
(embedded base64 savestates, subtitles, comment lines). The
input-frame line format is identical: `|<cmd>|<pad1>|<pad2>|<fds>|`
with pad1 in 8-char RLDUTSBA order. Our parser filters lines by
`startswith("|")` so the auxiliary sections naturally drop out.

These tests pin that contract: a synthetic FM3 with savestate +
subtitle interleaved between input frames must parse to the SAME
controller-1 byte sequence as the equivalent FM2.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    """Load convert_fm2.py as a module without polluting sys.path."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "convert_fm2.py"
    spec = importlib.util.spec_from_file_location("convert_fm2_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parser():
    return _load_module()


_FM2_HEADER = """\
version 3
emuVersion 22020
rerecordCount 0
palFlag 0
romFilename Super Mario Bros. (World)
romChecksum base64:DSqdM5DPLOzqUiZH8j99cw==
guid 00000000-0000-0000-0000-000000000000
fourscore 0
microphone 0
port0 1
port1 1
port2 0
"""

# Three input frames: NOOP, RIGHT, RIGHT+A.
_FRAMES = (
    "|0|........|........||\n"
    "|0|R.......|........||\n"
    "|0|R......A|........||\n"
)


def test_fm2_basic_parse(parser, tmp_path):
    """Sanity: header + 3 input frames → 3 bytes."""
    f = tmp_path / "demo.fm2"
    f.write_text(_FM2_HEADER + _FRAMES)
    out = parser.fceux_movie_to_bytes(f)
    assert out == bytes([0x00, 0x80, 0x80 | 0x01])


def test_fm3_with_savestate_block_parses_same_as_fm2(parser, tmp_path):
    """An FM3 with an embedded base64 savestate between input frames
    must parse to the SAME byte sequence as the equivalent FM2 — the
    savestate lines don't start with `|` so they're skipped."""
    fm3_with_savestate = (
        _FM2_HEADER
        + "comment author: synthetic test fixture\n"
        + "subtitle 0 hello\n"
        + "subtitle 30 world\n"
        + "|0|........|........||\n"
        + "savestate begin\n"
        + "gAAAU2F2ZXN0YXRlQmluYXJ5RGF0YUJhc2U2NEVuY29kZWQK\n"
        + "AnotherBase64LineWithoutAnyPipeChars==\n"
        + "savestate end\n"
        + "|0|R.......|........||\n"
        + "|0|R......A|........||\n"
    )
    f = tmp_path / "demo.fm3"
    f.write_text(fm3_with_savestate)
    out = parser.fceux_movie_to_bytes(f)
    # Same expected output as the plain FM2 test above — auxiliary
    # sections were correctly skipped.
    assert out == bytes([0x00, 0x80, 0x80 | 0x01])


def test_backward_compat_alias(parser, tmp_path):
    """The old `fm2_to_bytes` name must still work — existing callers
    (GUI _convert_fm2 historic path, scripts) import it by that name."""
    f = tmp_path / "demo.fm2"
    f.write_text(_FM2_HEADER + _FRAMES)
    via_alias = parser.fm2_to_bytes(f)
    via_canonical = parser.fceux_movie_to_bytes(f)
    assert via_alias == via_canonical


def test_pad1_bit_order_RLDUTSBA(parser, tmp_path):
    """A frame with all 8 buttons pressed must produce 0xFF — verifies
    the pad-byte bit positions match the project's button bitmask."""
    f = tmp_path / "demo.fm2"
    f.write_text(_FM2_HEADER + "|0|RLDUTSBA|........||\n")
    out = parser.fceux_movie_to_bytes(f)
    assert out == bytes([0xFF])


def test_malformed_short_pad_does_not_crash(parser, tmp_path):
    """A truncated frame line should pad with `.` (not pressed) rather
    than throwing — graceful degradation on partial recordings."""
    f = tmp_path / "demo.fm2"
    # pad1 only 3 chars
    f.write_text(_FM2_HEADER + "|0|R..|........||\n")
    out = parser.fceux_movie_to_bytes(f)
    # First three bits should be set per RLD; rest are padded ".".
    # R=0x80 only because L=. and D=. → just R pressed.
    assert out == bytes([0x80])


def test_empty_movie(parser, tmp_path):
    """An FM2/FM3 with header only and no input frames yields zero bytes."""
    f = tmp_path / "demo.fm2"
    f.write_text(_FM2_HEADER)
    assert parser.fceux_movie_to_bytes(f) == b""


def test_fm3_subtitles_with_pipe_in_text_are_filtered(parser, tmp_path):
    """Edge case: a subtitle line that happens to contain `|` chars
    should NOT be parsed as a frame (subtitles do not start with `|`).
    Confirms the `startswith("|")` filter is correct."""
    f = tmp_path / "demo.fm3"
    f.write_text(
        _FM2_HEADER
        + "subtitle 100 say |well-done| Mario\n"  # contains pipes mid-text
        + "|0|R.......|........||\n"
    )
    out = parser.fceux_movie_to_bytes(f)
    # Just the one input frame — RIGHT only.
    assert out == bytes([0x80])


# ============================================================
# BK2 (BizHawk movie) tests
# ============================================================


def _write_bk2(tmp_path: Path, input_log: str, name: str = "demo.bk2") -> Path:
    """Helper: build a minimal BK2 ZIP with just `Input Log.txt` set
    to the provided string. Real BK2s carry many more files
    (Header.txt, SyncSettings.json, etc) but our parser only reads
    the input log."""
    import zipfile
    bk2 = tmp_path / name
    with zipfile.ZipFile(bk2, "w") as zf:
        zf.writestr("Input Log.txt", input_log)
    return bk2


_BK2_HEADER = "[Input]\nLogKey:#Reset|Power|FDS Eject|FDS Insert 0|FDS Insert 1|VS Coin 1|VS Coin 2|VS Insert Coin P1|VS Insert Coin P2|VS Service Switch|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Start|P1 Select|P1 B|P1 A|#P2 Up|P2 Down|P2 Left|P2 Right|P2 Start|P2 Select|P2 B|P2 A\n"


def test_bk2_basic_parse(parser, tmp_path):
    """Three frames: NOOP, RIGHT, RIGHT+A. UDLRSsBA pad order →
    Right is at column 3, A is at column 7."""
    log = (
        _BK2_HEADER
        + "|..........|........|........|\n"
        + "|..........|...R....|........|\n"
        + "|..........|...R...A|........|\n"
        + "[/Input]\n"
    )
    bk2 = _write_bk2(tmp_path, log)
    out = parser.bk2_to_bytes(bk2)
    # Same expected output as the corresponding FM2 test even though
    # the column layout differs — the parser knows which column is
    # which button per format.
    assert out == bytes([0x00, 0x80, 0x80 | 0x01])


def test_bk2_pad_bit_order_UDLRSsBA(parser, tmp_path):
    """All 8 P1 buttons pressed → 0xFF, regardless of source layout.
    Verifies the BK2 column-to-bit mapping covers every button."""
    log = _BK2_HEADER + "|..........|UDLRSsBA|........|\n[/Input]\n"
    bk2 = _write_bk2(tmp_path, log)
    out = parser.bk2_to_bytes(bk2)
    assert out == bytes([0xFF])


def test_bk2_individual_buttons(parser, tmp_path):
    """Each button alone produces only its bit. Pins the per-column
    mapping precisely."""
    cases = [
        ("U.......", 0x10),  # Up
        (".D......", 0x20),  # Down
        ("..L.....", 0x40),  # Left
        ("...R....", 0x80),  # Right
        ("....S...", 0x08),  # Start
        (".....s..", 0x04),  # Select
        ("......B.", 0x02),  # B
        (".......A", 0x01),  # A
    ]
    for pad, expected in cases:
        log = _BK2_HEADER + f"|..........|{pad}|........|\n[/Input]\n"
        bk2 = _write_bk2(tmp_path, log, name=f"button_{expected:02x}.bk2")
        out = parser.bk2_to_bytes(bk2)
        assert out == bytes([expected]), (
            f"BK2 pad '{pad}' produced 0x{out.hex()}, expected 0x{expected:02x}"
        )


def test_bk2_ignores_system_controls(parser, tmp_path):
    """The 10-char system controls column (Reset/Power/FDS/VS) is NOT
    a NES button input — even with system bits set, the resulting
    NES-bitmask must be 0 if no P1 button is pressed."""
    log = _BK2_HEADER + "|RPEEE.....|........|........|\n[/Input]\n"
    bk2 = _write_bk2(tmp_path, log)
    out = parser.bk2_to_bytes(bk2)
    assert out == bytes([0x00])


def test_bk2_missing_input_log_raises(parser, tmp_path):
    """A BK2 archive without the input log file should raise a
    descriptive ValueError, not silently produce empty output."""
    import zipfile
    bk2 = tmp_path / "broken.bk2"
    with zipfile.ZipFile(bk2, "w") as zf:
        zf.writestr("Header.txt", "version 1")
    with pytest.raises(ValueError, match="Input Log.txt"):
        parser.bk2_to_bytes(bk2)


def test_movie_to_bytes_dispatcher(parser, tmp_path):
    """The format-agnostic dispatcher picks the right parser by
    extension. Sanity-check both branches."""
    # FM2 path
    fm2 = tmp_path / "demo.fm2"
    fm2.write_text(_FM2_HEADER + "|0|R.......|........||\n")
    assert parser.movie_to_bytes(fm2) == bytes([0x80])
    # BK2 path
    log = _BK2_HEADER + "|..........|...R....|........|\n[/Input]\n"
    bk2 = _write_bk2(tmp_path, log)
    assert parser.movie_to_bytes(bk2) == bytes([0x80])


def test_bk2_old_format_without_input_brackets(parser, tmp_path):
    """Some older BizHawk versions emit Input Log.txt with just the
    LogKey + frame lines, no [Input] / [/Input] brackets. Parser
    should still produce frames in that case."""
    # No [Input] header at all — just LogKey then frames.
    log = (
        "LogKey:#Reset|Power|...\n"
        + "|..........|...R....|........|\n"
        + "|..........|...R...A|........|\n"
    )
    bk2 = _write_bk2(tmp_path, log)
    out = parser.bk2_to_bytes(bk2)
    assert out == bytes([0x80, 0x81])
