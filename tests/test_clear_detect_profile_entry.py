"""The clear-detector's replay harness must be PROFILE-DRIVEN, not SMB-wired.

HISTORY. `run_ground_truth_test` opened with `game = SmbGame()` and replayed
every trace through this module's own SMB action space at SMB's frame_skip.
Three hardcodes, one consequence: no non-SMB profile could reach the detector
at all. The 2026-08-26 clear-detection census
(docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md) surveyed 29 profiles
and exercised the detector on exactly ONE of them for this reason, then
reported the other 28 silences as nulls about those games. They were nulls
about this function.

EVERY TEST HERE IS WRITTEN TO FAIL AGAINST THE HARDCODED VERSION, and the
last two prove that rather than asserting it: one drives the same replay with
the harness the hardcode would have produced and shows it errors out, the
other reads the function's own source. The discipline being applied is the
one that caught two vacuous gates this week -- ask what the test would report
if the mechanism were absent, and then actually run that case.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.clear_detect import (  # noqa: E402
    ACTION_SPACE, FS, SmbGame, build_harness, run_ground_truth_test,
)
import scripts.clear_detect as clear_detect  # noqa: E402


# --------------------------------------------------------------------------
# A fake NESEnvironment. Deliberately not a real emulator: the subject here is
# WHICH ADAPTER the harness builds, and reaching for a real ROM would pin the
# test back to SMB -- the exact coupling under test.
# --------------------------------------------------------------------------

FAKE_LEVEL_KEY_ADDR = 0x0500
FAKE_LEVEL_KEY_VALUE = 7
FAKE_PROGRESS_ADDR = 0x0300
FAKE_Y_ADDR = 0x0301
FAKE_LIVES_ADDR = 0x0302


def _fake_ram() -> np.ndarray:
    ram = np.zeros(2048, dtype=np.uint8)
    ram[FAKE_LEVEL_KEY_ADDR] = FAKE_LEVEL_KEY_VALUE
    ram[FAKE_PROGRESS_ADDR] = 42
    ram[FAKE_Y_ADDR] = 100
    ram[FAKE_LIVES_ADDR] = 3
    return ram


class _FakeEnv:
    """Only the NESEnvironment surface run_ground_truth_test/run_episode use.

    RAM is constant, so no signal ever fires and no clear is ever declared --
    which is fine, because what is being measured is the harness, not the
    detector. `opened` records the ROM path each construction was handed: that
    is the SMB hardcode's fingerprint, since SmbGame().rom is the SMB ROM no
    matter which profile was asked for."""

    opened: list[str] = []

    def __init__(self, rom, frame_skip: int = 1) -> None:
        self.rom = str(rom)
        self.sample_rate = 44100
        self.n_steps = 0
        self._t = 0
        _FakeEnv.opened.append(self.rom)

    def reset(self) -> None:
        self._t = 0

    def set_audio_output_enabled(self, enabled: bool) -> None:
        pass

    def save_state(self) -> bytes:
        return bytes([self._t & 0xFF])

    def load_state(self, blob) -> None:
        self._t = blob[0] if blob else 0

    def step(self, mask: int) -> None:
        self._t = (self._t + 1) & 0xFF
        self.n_steps += 1

    def get_audio(self):
        return np.zeros(8, dtype=np.int16)

    def get_ram_range(self, lo: int, hi: int):
        return _fake_ram()[lo:hi]

    def close(self) -> None:
        pass


class _FakeNesCore:
    NESEnvironment = _FakeEnv


FAKE_FRAME_SKIP = 3          # deliberately != FS (4)
FAKE_ACTION_SPACE = [[], ["right"], ["left"], ["A"]]   # len 4 != len(ACTION_SPACE)


def _write_fake_profile(tmp_path: Path) -> Path:
    rom = tmp_path / "not-a-real-rom.nes"
    rom.write_bytes(b"")
    profile = tmp_path / "fake_game.yaml"
    profile.write_text(yaml.safe_dump({
        "name": "FakeGame",
        "frame_skip": FAKE_FRAME_SKIP,
        "action_space": FAKE_ACTION_SPACE,
        "solve": {
            # GenericGame does `REPO / rom`, and pathlib lets an absolute
            # right-hand side win, so an absolute path here lands verbatim.
            "rom": str(rom),
            "progress": {"lo": FAKE_PROGRESS_ADDR},
            "y": FAKE_Y_ADDR,
            "level_key": [FAKE_LEVEL_KEY_ADDR],
            "lives": FAKE_LIVES_ADDR,
        },
    }))
    return profile


def _write_fake_trace(tmp_path: Path, actions: list[int]) -> str:
    """A solution trace rooted at the fake env's start, keyed by the FAKE
    game's level_key -- so the SMB adapter cannot even validate the root."""
    root = tmp_path / "root.state"
    root.write_bytes(b"\x00")
    base = tmp_path / "sol_000"
    base.with_suffix(".json").write_text(json.dumps({
        "root_state": str(root),
        "start_wd": [FAKE_LEVEL_KEY_VALUE],
        "clear_wd": None,
    }))
    np.save(str(base) + ".actions", np.array(actions, dtype=np.int64))
    return str(base)


@pytest.fixture
def fake_core(monkeypatch):
    _FakeEnv.opened.clear()
    monkeypatch.setattr(clear_detect, "nes_core", _FakeNesCore)
    return _FakeEnv


# --------------------------------------------------------------------------
# build_harness
# --------------------------------------------------------------------------

def test_no_profile_is_still_the_historical_smb_harness() -> None:
    # The default path has to stay bit-for-bit what every banked receipt was
    # produced under, or "profile-driven" would be a rewrite rather than a
    # generalization.
    h = build_harness()
    assert isinstance(h.game, SmbGame)
    assert h.action_space == ACTION_SPACE
    assert h.frame_skip == FS
    assert h.profile is None
    assert h.action_space[h.dir_index] == ["right"]


def test_a_profile_builds_that_profiles_adapter(tmp_path) -> None:
    profile = _write_fake_profile(tmp_path)
    h = build_harness(str(profile))

    # 1. the adapter comes from the profile, not from SmbGame()
    assert not isinstance(h.game, SmbGame)
    assert type(h.game).__name__ == "GenericGame"
    assert h.game.rom.endswith("not-a-real-rom.nes")
    # 2. the action space the recorded indices index into comes from it too
    assert h.action_space == FAKE_ACTION_SPACE
    # 3. and so does the frame_skip the trace was recorded at
    assert h.frame_skip == FAKE_FRAME_SKIP

    # Anti-vacuity: every one of those three would have to CHANGE for this to
    # be a real assertion, so state what the hardcoded harness returns.
    smb = build_harness()
    assert type(h.game) is not type(smb.game)
    assert h.game.rom != smb.game.rom
    assert h.action_space != smb.action_space
    assert h.frame_skip != smb.frame_skip


def test_an_smb_engine_profile_still_routes_to_the_smb_adapter() -> None:
    # `make_game` sends a profile with no `solve:` block to SmbGame. That is
    # the correct answer for an SMB-engine game and must survive: making the
    # entry point generic must not make it refuse the game it was built for.
    ll = REPO / "configs" / "lost_levels.yaml"
    if not ll.exists():
        pytest.skip("lost_levels profile not present")
    prof = yaml.safe_load(ll.read_text())
    if "solve" in prof:
        pytest.skip("profile has a solve: block; not the SMB-engine case")
    h = build_harness(str(ll))
    assert isinstance(h.game, SmbGame)
    assert h.action_space == [list(a) for a in prof["action_space"]]


def test_a_real_shipped_profile_loads() -> None:
    # configs/castlevania.yaml is one of the three positive controls the
    # census was supposed to gate on and never ran. Loading it here is cheap
    # and pins the reader to a profile that actually exists on disk.
    cv = REPO / "configs" / "castlevania.yaml"
    if not cv.exists():
        pytest.skip("castlevania profile not present")
    h = build_harness(str(cv))
    assert type(h.game).__name__ == "GenericGame"
    assert h.game.rom.endswith("Castlevania (USA).nes")
    assert h.frame_skip == 4
    assert h.action_space[h.dir_index] == ["right"]


def test_a_missing_profile_is_refused_by_name(tmp_path) -> None:
    with pytest.raises(SystemExit, match="profile not found"):
        build_harness(str(tmp_path / "nope.yaml"))


def test_a_profile_without_an_action_space_is_refused(tmp_path) -> None:
    p = tmp_path / "no_space.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "solve": {"rom": "r.nes"}}))
    with pytest.raises(SystemExit, match="action_space"):
        build_harness(str(p))


def test_a_space_with_no_direction_is_refused_rather_than_probing_noop(tmp_path) -> None:
    # The input-LOCK signal asks whether holding a DIRECTION moves RAM
    # relative to holding nothing. Falling back to NOOP would compare NOOP
    # against NOOP, read "locked" on every frame, and hand the detector a
    # free permanent vote. A signal that is structurally always-on is worse
    # than an absent one: it looks like evidence.
    p = tmp_path / "no_dpad.yaml"
    p.write_text(yaml.safe_dump({
        "name": "x", "action_space": [[], ["A"], ["B"], ["A", "B"]],
        "solve": {"rom": "r.nes", "progress": {"lo": 1}, "y": 2,
                  "level_key": [3], "lives": 4},
    }))
    with pytest.raises(SystemExit, match="no directional action"):
        build_harness(str(p))


# --------------------------------------------------------------------------
# run_ground_truth_test actually DRIVES the profile's adapter
# --------------------------------------------------------------------------

def test_the_replay_is_driven_by_the_profiles_adapter(tmp_path, fake_core) -> None:
    profile = _write_fake_profile(tmp_path)
    actions = [1, 2, 0, 1]
    base = _write_fake_trace(tmp_path, actions)

    summary = run_ground_truth_test([base], verbose=False, profile=str(profile))
    result = summary["per_run"][0]

    # The root validated: the FAKE level_key byte was read, so the adapter
    # that read it was the profile's. SmbGame would have read $075F/$075C,
    # got (0, 0) against the recorded (7,), and errored out here.
    assert "error" not in result, result.get("error")
    assert result["start_wd"] == [FAKE_LEVEL_KEY_VALUE]

    # The ROM the emulator was actually opened on is the profile's.
    assert fake_core.opened == [str(tmp_path / "not-a-real-rom.nes")]
    assert fake_core.opened[0] != SmbGame().rom

    # The replay ran at the PROFILE's frame_skip, not FS. run_episode pads
    # with margin_actions=90 NOOPs, and each action step is frame_skip raw
    # frames -- at SMB's FS this count would be 4/3 larger.
    assert result["n_frames_replayed"] == (len(actions) + 90) * FAKE_FRAME_SKIP


def test_the_receipt_names_the_harness_that_replayed(tmp_path, fake_core) -> None:
    # A receipt that does not say what replayed cannot be audited later --
    # which is how 29 nulls got filed as measurements of 29 games.
    profile = _write_fake_profile(tmp_path)
    base = _write_fake_trace(tmp_path, [1, 1])
    summary = run_ground_truth_test([base], verbose=False, profile=str(profile))
    assert summary["harness"] == {
        "profile": str(profile),
        "game_adapter": "GenericGame",
        "rom": str(tmp_path / "not-a-real-rom.nes"),
        "frame_skip": FAKE_FRAME_SKIP,
        "n_actions_in_space": len(FAKE_ACTION_SPACE),
        "lock_probe_action": ["right"],
    }


def test_the_default_receipt_still_names_the_smb_harness() -> None:
    h = build_harness()
    assert h.provenance() == {
        "profile": None,
        "game_adapter": "SmbGame",
        "rom": SmbGame().rom,
        "frame_skip": FS,
        "n_actions_in_space": len(ACTION_SPACE),
        "lock_probe_action": ["right"],
    }


# --------------------------------------------------------------------------
# The trip-wires: these fail if the hardcode comes back
# --------------------------------------------------------------------------

def test_the_hardcoded_harness_really_would_fail_this_replay(tmp_path, fake_core) -> None:
    # Not an assertion ABOUT the regression -- the regression, run. Same
    # trace, same fake emulator, harness built the old way (no profile =
    # SmbGame + SMB's space + SMB's frame_skip). If this ever stops erroring,
    # the tests above have gone quiet for some reason other than the fix.
    _write_fake_profile(tmp_path)
    base = _write_fake_trace(tmp_path, [1, 2, 0, 1])

    summary = run_ground_truth_test([base], verbose=False)   # profile=None

    result = summary["per_run"][0]
    assert "root replay mismatch" in result.get("error", "")
    assert summary["harness"]["game_adapter"] == "SmbGame"
    assert fake_core.opened == [SmbGame().rom]


def test_run_ground_truth_test_constructs_no_game_of_its_own() -> None:
    # The static trip-wire, kept because the behavioural ones above depend on
    # a fake emulator and a synthetic profile that a future refactor could
    # quietly stop exercising. The ONLY legitimate `SmbGame()` construction
    # in the replay path is build_harness's no-profile branch.
    src = inspect.getsource(run_ground_truth_test)
    assert "SmbGame(" not in src, (
        "run_ground_truth_test constructs a game adapter directly again; the "
        "adapter must come from build_harness(profile) or no non-SMB profile "
        "can reach the detector.")
    assert "build_harness(" in src
    assert "SmbGame(" in inspect.getsource(build_harness)
