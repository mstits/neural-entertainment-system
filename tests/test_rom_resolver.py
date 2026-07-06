"""Pin behavior of the fuzzy ROM resolver + integrity helpers.

`make train GAME=mario` must succeed from a fresh clone even when the
user's dump isn't named exactly `Super Mario Bros. (World).nes`: the
resolver accepts a short `roms/mario.nes`, a case-differing name, or a
regional retag, and — when the match is ambiguous or empty — fails with
a clean message that names the candidates and the exact expected file.
It also validates the resolved dump's whole-file MD5 against the
profile's declared `rom_hashes`, which the run manifest records for
reproducibility.

The resolver is loaded straight from `scripts/train_game.py` (which is
kept import-light — no torch/nes_core at module scope — precisely so
these unit tests stay fast and hermetic).
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_train_game():
    script = Path(__file__).resolve().parent.parent / "scripts" / "train_game.py"
    spec = importlib.util.spec_from_file_location("train_game_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tg():
    return _load_train_game()


CANONICAL = "roms/Super Mario Bros. (World).nes"


def _touch(path: Path, data: bytes = b"NES\x1a") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------
# match_roms — the pure ranking core
# --------------------------------------------------------------------------

def test_match_exact_basename(tg, tmp_path):
    f = _touch(tmp_path / "Super Mario Bros. (World).nes")
    assert tg.match_roms("mario", CANONICAL, [f]) == [f]


def test_match_case_insensitive_basename(tg, tmp_path):
    f = _touch(tmp_path / "super mario bros. (world).nes")
    assert tg.match_roms("mario", CANONICAL, [f]) == [f]


def test_match_short_name_by_key(tg, tmp_path):
    f = _touch(tmp_path / "mario.nes")
    assert tg.match_roms("mario", CANONICAL, [f]) == [f]


def test_match_token_from_canonical(tg, tmp_path):
    # Key "smb" isn't in the filename, but the canonical token "mario" is.
    f = _touch(tmp_path / "Super Mario Brothers.nes")
    assert tg.match_roms("smb", CANONICAL, [f]) == [f]


def test_match_none_when_unrelated(tg, tmp_path):
    f = _touch(tmp_path / "Contra (USA).nes")
    assert tg.match_roms("mario", CANONICAL, [f]) == []


def test_match_exact_ranks_before_fuzzy(tg, tmp_path):
    exact = _touch(tmp_path / "Super Mario Bros. (World).nes")
    fuzzy = _touch(tmp_path / "mario.nes")
    ranked = tg.match_roms("mario", CANONICAL, [fuzzy, exact])
    assert ranked[0] == exact
    assert set(ranked) == {exact, fuzzy}


def test_match_region_tag_not_a_false_positive(tg, tmp_path):
    # "usa"/"world" are noise words; a bare region dump must not match Mario.
    f = _touch(tmp_path / "Some Game (USA).nes")
    assert tg.match_roms("mario", CANONICAL, [f]) == []


# --------------------------------------------------------------------------
# resolve_rom — orchestration + clean SystemExit errors
# --------------------------------------------------------------------------

def test_resolve_explicit_rom_wins(tg, tmp_path):
    rom = _touch(tmp_path / "whatever.nes")
    assert tg.resolve_rom("mario", str(rom), {}) == str(rom)


def test_resolve_explicit_rom_missing_exits(tg, tmp_path):
    with pytest.raises(SystemExit) as ei:
        tg.resolve_rom("mario", str(tmp_path / "nope.nes"), {})
    assert "--rom not found" in str(ei.value)


def test_resolve_profile_rom_path(tg, tmp_path):
    rom = _touch(tmp_path / "declared.nes")
    profile = {"rom_path": str(rom)}
    assert tg.resolve_rom("mario", None, profile) == str(rom)


def test_resolve_canonical_exact(tg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "roms" / "Super Mario Bros. (World).nes")
    assert tg.resolve_rom("mario", None, {}) == CANONICAL


def test_resolve_fuzzy_single_short_name(tg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = _touch(tmp_path / "roms" / "mario.nes")
    resolved = tg.resolve_rom("mario", None, {})
    assert Path(resolved).resolve() == f.resolve()


def test_resolve_none_lists_expected(tg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "roms").mkdir()
    _touch(tmp_path / "roms" / "Contra (USA).nes")
    with pytest.raises(SystemExit) as ei:
        tg.resolve_rom("mario", None, {})
    msg = str(ei.value)
    assert "No ROM found" in msg
    assert "Super Mario Bros. (World).nes" in msg  # exact expected name shown
    assert "Contra (USA).nes" in msg               # present files listed


def test_resolve_ambiguous_lists_candidates(tg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "roms" / "mario.nes")
    _touch(tmp_path / "roms" / "super mario bros usa.nes")
    with pytest.raises(SystemExit) as ei:
        tg.resolve_rom("mario", None, {})
    msg = str(ei.value)
    assert "Ambiguous ROM" in msg
    assert "mario.nes" in msg
    assert "super mario bros usa.nes" in msg


def test_resolve_no_canonical_and_no_profile_exits(tg):
    with pytest.raises(SystemExit) as ei:
        tg.resolve_rom("nonexistent_game", None, {})
    assert "No ROM resolvable" in str(ei.value)


# --------------------------------------------------------------------------
# MD5 helpers
# --------------------------------------------------------------------------

def test_rom_md5_matches_hashlib(tg, tmp_path):
    data = b"NES\x1a" + b"\x01\x02\x03" * 999
    rom = _touch(tmp_path / "x.nes", data)
    assert tg.rom_md5(rom) == hashlib.md5(data).hexdigest()


def test_profile_rom_hashes_filters_and_lowercases(tg):
    profile = {"rom_hashes": ["", "  ", "ABC123", "def456"]}
    assert tg.profile_rom_hashes(profile) == ["abc123", "def456"]


def test_profile_rom_hashes_includes_expected_md5_scalar(tg):
    profile = {"expected_md5": "ABCDEF", "rom_hashes": ["", "0123"]}
    assert set(tg.profile_rom_hashes(profile)) == {"abcdef", "0123"}


def test_profile_rom_hashes_empty_when_only_placeholders(tg):
    assert tg.profile_rom_hashes({"rom_hashes": [""]}) == []


def test_validate_rom_md5_match_no_warning(tg, tmp_path, caplog):
    data = b"NES\x1a123"
    rom = _touch(tmp_path / "ok.nes", data)
    md5 = hashlib.md5(data).hexdigest()
    with caplog.at_level("WARNING"):
        got = tg.validate_rom_md5(str(rom), {"rom_hashes": [md5]})
    assert got == md5
    assert "MISMATCH" not in caplog.text


def test_validate_rom_md5_mismatch_warns_but_returns(tg, tmp_path, caplog):
    rom = _touch(tmp_path / "bad.nes", b"NES\x1a999")
    with caplog.at_level("WARNING"):
        got = tg.validate_rom_md5(str(rom), {"rom_hashes": ["deadbeef"]})
    assert got == hashlib.md5(b"NES\x1a999").hexdigest()  # never raises
    assert "MISMATCH" in caplog.text


# --------------------------------------------------------------------------
# start-state resolution
# --------------------------------------------------------------------------

def test_resolve_start_state_prefers_profile(tg, tmp_path):
    ss = _touch(tmp_path / "explicit.state.bin", b"state")
    rom = _touch(tmp_path / "g.nes")
    profile = {"start_state_path": str(ss)}
    assert tg.resolve_start_state(profile, str(rom)) == str(ss)


def test_resolve_start_state_sidecar_fallback(tg, tmp_path):
    rom = _touch(tmp_path / "g.nes")
    sidecar = _touch(tmp_path / "g_start.state.bin", b"state")
    assert tg.resolve_start_state({}, str(rom)) == str(sidecar)


def test_resolve_start_state_none_when_missing(tg, tmp_path):
    rom = _touch(tmp_path / "g.nes")
    assert tg.resolve_start_state({}, str(rom)) is None


def test_resolve_start_state_ignores_missing_profile_path(tg, tmp_path):
    rom = _touch(tmp_path / "g.nes")
    profile = {"start_state_path": str(tmp_path / "gone.state.bin")}
    assert tg.resolve_start_state(profile, str(rom)) is None
