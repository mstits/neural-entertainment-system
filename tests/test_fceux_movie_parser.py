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
