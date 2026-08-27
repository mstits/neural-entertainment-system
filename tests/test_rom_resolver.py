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
import os
import sys
import time
import types
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


def test_resolve_start_state_missing_profile_path_is_fatal(tg, tmp_path):
    """A DECLARED start_state_path that doesn't exist must raise (naming
    the path), not silently resolve to None/cold boot — that trained
    the title-screen demo for a whole run before anyone noticed."""
    rom = _touch(tmp_path / "g.nes")
    profile = {"start_state_path": str(tmp_path / "gone.state.bin")}
    with pytest.raises(SystemExit, match="gone.state.bin"):
        tg.resolve_start_state(profile, str(rom))


def test_resolve_start_state_sidecar_does_not_mask_missing_declared(tg, tmp_path):
    """Even with a valid sidecar on disk, a missing DECLARED path stays
    fatal — the sidecar may hold a different level than the profile
    configured, so falling back would silently train the wrong one."""
    rom = _touch(tmp_path / "g.nes")
    _touch(tmp_path / "g_start.state.bin", b"state")
    profile = {"start_state_path": str(tmp_path / "gone.state.bin")}
    with pytest.raises(SystemExit, match="gone.state.bin"):
        tg.resolve_start_state(profile, str(rom))


# --------------------------------------------------------------------------
# main() — GA-mode --resume must resolve the latest checkpoint
# --------------------------------------------------------------------------

def test_main_resume_resolves_latest_ga_checkpoint(tg, tmp_path, monkeypatch):
    """`--resume` (the default) must resolve the latest gen_*.pt in the
    trainer's checkpoint dir and pass it to Trainer.run() as
    `resume_from`. Before the fix this block was a bare `pass`, so
    every GA-mode launch — including every supervisor-triggered
    restart after a crash — silently cold-started from a fresh random
    population instead of resuming."""
    rom = _touch(tmp_path / "g.nes")
    state = _touch(tmp_path / "start.state.bin", b"state")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        f"name: Test Game\nstart_state_path: {state}\n"
    )

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    older = _touch(ckpt_dir / "gen_00001.pt", b"old")
    newer = _touch(ckpt_dir / "gen_00002.pt", b"new")
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now - 10, now - 10))

    calls: dict = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.checkpoint_dir = ckpt_dir

        def run(self, num_generations, resume_from=None, fresh_start=False):
            calls["resume_from"] = resume_from
            calls["fresh_start"] = fresh_start

    fake_mod = types.ModuleType("src.training.trainer")
    fake_mod.Trainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "src.training.trainer", fake_mod)

    monkeypatch.setattr(sys, "argv", [
        "train_game.py",
        "--profile", str(profile_path),
        "--rom", str(rom),
        "--iters", "1",
        "--no-supervise",
    ])

    rc = tg.main()

    assert rc == 0
    assert calls.get("resume_from") == str(newer), (
        "GA-mode --resume must resolve to the newest gen_*.pt checkpoint, "
        f"got {calls.get('resume_from')!r}"
    )


# --------------------------------------------------------------------------
# match_roms — a same-franchise sequel must never fuzzy-match a different
# installment's canonical name. Distinct games in the same franchise
# (Double Dragon vs Double Dragon II, Zelda vs Zelda II, Mega Man 2 vs
# Mega Man 3) have distinct RAM maps; a token-substring match alone
# (e.g. "dragon", "zelda", "mega") cannot tell them apart, and binding
# the wrong dump silently trains a reward function against the wrong
# game's memory layout.
# --------------------------------------------------------------------------

DOUBLE_DRAGON = "roms/Double Dragon (USA).nes"
ZELDA1 = "roms/Legend of Zelda, The (USA) (Rev A).nes"


def test_match_rejects_sequel_when_canonical_is_the_original(tg, tmp_path):
    f = _touch(tmp_path / "Double Dragon II - The Revenge (USA) (Rev A).nes")
    assert tg.match_roms("double_dragon", DOUBLE_DRAGON, [f]) == []


def test_match_rejects_second_sequel_too(tg, tmp_path):
    f = _touch(tmp_path / "Double Dragon III - The Sacred Stones (USA).nes")
    assert tg.match_roms("double_dragon", DOUBLE_DRAGON, [f]) == []


def test_match_rejects_zelda_ii_for_zelda_1(tg, tmp_path):
    f = _touch(tmp_path / "Zelda II - The Adventure of Link (USA).nes")
    assert tg.match_roms("zelda", ZELDA1, [f]) == []


def test_resolve_single_sequel_candidate_is_no_rom_found_not_a_silent_bind(
    tg, tmp_path, monkeypatch,
):
    """The single-wrong-candidate case: only a sequel dump is present, no
    exact canonical file. Before the fix this silently resolved (with a
    warning) to the sequel's ROM, binding the original's reward function
    to a different game's RAM map. It must instead behave exactly like
    'no candidate at all' — a clean SystemExit naming what's missing."""
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "roms" / "Double Dragon II - The Revenge (USA) (Rev A).nes")
    with pytest.raises(SystemExit) as ei:
        tg.resolve_rom("double_dragon", None, {})
    assert "No ROM found" in str(ei.value)


def test_match_still_allows_a_true_retag_of_the_same_sequel(tg, tmp_path):
    """The fix must not overcorrect: a retagged/renamed dump of the SAME
    installment (canonical already names it, e.g. Mega Man 2) still
    fuzzy-matches."""
    canonical = "roms/Mega Man 2 (USA).nes"
    f = _touch(tmp_path / "megaman2.nes")
    assert tg.match_roms("megaman", canonical, [f]) == [f]
