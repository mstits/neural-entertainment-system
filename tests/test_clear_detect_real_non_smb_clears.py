"""Real, witnessed, non-SMB clears -- the fixture shape 186 green detector
tests never had.

WHY THIS FILE EXISTS. Every case in test_clear_detect_ground_truth.py is SMB.
Every case in test_confluence_v2.py and test_detector_v3.py is a synthetic RAM
stream built by hand to the detector's own shape (a fixed byte offset for
"lives", a fixed offset for "room", entity slots that zero out exactly the way
`coord_entity_windows` expects). test_clear_reachability.py and
test_clear_quorum_reachability.py are excellent on the arithmetic, but even
there the Bubble Bobble / Tetris-B spans arrive as hand-typed numbers copied
from a receipt (`progress={"lo": 0x0401}`), never as bytes the emulator
actually produced. A suite built entirely that way cannot discover that the
instrument is shaped wrong, because it never runs the instrument against
anything that ISN'T shaped the way the instrument expects.

This file replays two REAL, independently witnessed clears end to end through
the real emulator, the real profile, and the real detector -- nothing here is
a stub or a hand-built RAM dict:

  * Bubble Bobble round 69 -> 70. Trace banked at
    docs/receipts/games/bubble_bobble_round69_win_trace.npy (299 actions,
    copied verbatim from runs/bubble_bobble/chain_day2f/lvl_00_69/solutions/
    sol_000.actions.npy, itself `replay_verified: true`). The root save-state
    is NOT committed here -- like every other .state.bin in this repo, it is
    a derived capture of copyrighted cartridge data, and roms/ has zero
    tracked files for the same reason (see test_clear_detect_ground_truth.py's
    ROM.exists() guard). Both fixtures are gated the same way: present on
    every machine that ran the 2026-08-26 clear-detection campaign, and
    skipped (never failed) on a fresh checkout that lacks them.
  * Tetris B-TYPE's 4,329-action win. Trace already tracked at
    docs/receipts/games/tetris_b_cf_gate_2026-08-10/win_trace_4329.npy; root
    state is roms/Tetris (USA)_btype_start.state.bin, same not-tracked/skip
    convention.

Both games have their OWN real, CONFIRMED clear predicate (Bubble Bobble:
level_key on the round counter; Tetris-B: byte_change on the quota byte) --
that predicate is not what is under test here and is asserted first, as the
fixture's own sanity check. What IS under test is the CONFLUENCE mechanism
(StreamingConfluenceDetector / coord_entity_windows / score_tally_windows) --
the generic, game-agnostic fallback that would be the ONLY available
instrument on any of the 26 gap-roster profiles that have no clean byte
predicate of their own. Measured directly here, not inferred: it never fires
on either trace, for an arithmetic reason (`coord_entity_windows` requires a
position drop of >= 300 units; Bubble Bobble's own progress byte spans 1 unit
per clear, Tetris-B's spans 32) that is identical in kind to the
`cv_odometer_swap` finding already guarded permanently in
tests/test_clear_quorum_reachability.py, and to the arithmetic-reachability
tests in tests/test_clear_reachability.py -- both of which this file
corroborates against real bytes instead of a hand-typed span.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

nes_core = pytest.importorskip(
    "nes_core", reason="needs the compiled nes_core extension")
clear_detect = pytest.importorskip(
    "clear_detect", reason="needs the compiled nes_core extension")
StreamingConfluenceDetector = clear_detect.StreamingConfluenceDetector
UnfireableHook = clear_detect.UnfireableHook

from go_explore_solve import make_game  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402
import clear_reachability  # noqa: E402


# ==========================================================================
# Fixtures
# ==========================================================================

class RealClear:
    """One real, witnessed clear: a profile, an action trace, and a root
    state to replay it from. `missing()` names exactly which file is absent
    rather than a bare "skip" -- the same discipline clear_reachability's
    refusals use, so a vanished fixture reads as a named gap, not silence."""

    def __init__(self, name: str, profile: Path, actions: Path, root_state: Path,
                 rom: Path):
        self.name = name
        self.profile = profile
        self.actions = actions
        self.root_state = root_state
        self.rom = rom

    def missing(self) -> str | None:
        for p in (self.profile, self.actions, self.root_state, self.rom):
            if not p.exists():
                return f"{self.name}: missing {p.relative_to(REPO)}"
        return None


BUBBLE_BOBBLE = RealClear(
    "bubble_bobble_round_69",
    profile=REPO / "configs/bubble_bobble.yaml",
    actions=REPO / "docs/receipts/games/bubble_bobble_round69_win_trace.npy",
    root_state=REPO / "runs/bubble_bobble/chain_day2f/entrances/entrance_after_68.state",
    rom=REPO / "roms/Bubble Bobble (USA).nes",
)

TETRIS_B = RealClear(
    "tetris_b_4329_action_win",
    profile=REPO / "configs/tetris_b.yaml",
    actions=REPO / "docs/receipts/games/tetris_b_cf_gate_2026-08-10/win_trace_4329.npy",
    root_state=REPO / "roms/Tetris (USA)_btype_start.state.bin",
    rom=REPO / "roms/Tetris (USA).nes",
)

FIXTURES = [BUBBLE_BOBBLE, TETRIS_B]


def _skip_if_missing(fx: RealClear) -> None:
    reason = fx.missing()
    if reason:
        pytest.skip(reason)


def _replay(fx: RealClear):
    """Replay the real action trace through the real emulator and return
    (ram_hist[T,2048] uint8, truth_action, game, prof). `truth_action` comes
    from the game's OWN is_clear -- level_key for Bubble Bobble, byte_change
    for Tetris-B -- neither of which is the confluence mechanism under test.

    `_replay_modalities` below is the same replay with the other hardware
    surfaces recorded alongside, for the armed-profile test in section 4."""
    prof = yaml.safe_load(fx.profile.read_text())
    game = make_game(prof)
    space = [list(a) for a in prof["action_space"]]
    masks = action_space_to_bitmasks(space)
    fs = int(prof.get("frame_skip", 4))
    actions = np.load(fx.actions).tolist()

    env = nes_core.NESEnvironment(game.rom, frame_skip=1)
    env.reset()
    env.load_state(fx.root_state.read_bytes())
    for _ in range(fs):
        env.step(0)
    ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
    if hasattr(game, "note_start"):
        game.note_start(ram0)
    start_wd = tuple(game.level_key(ram0))

    ctx: dict = {}
    truth_action = None
    hist = []
    for i, a in enumerate(actions):
        m = int(masks[a])
        for _ in range(fs):
            env.step(m)
        ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        hist.append(ram)
        if truth_action is None and game.is_clear(start_wd, ram, ctx):
            truth_action = i
    env.close()
    return np.stack(hist), truth_action, game, prof


def _replay_modalities(fx: RealClear, margin: int = 90):
    """The same replay, recording the odometer triple per observation, plus
    `margin` NOOP actions after the trace ends.

    THE MARGIN IS NOT AN EXTENSION OF THE TRAJECTORY. Both fixtures were
    recorded to end EXACTLY on the clearing action (section 1 asserts it),
    so without a tail there is not one observation after the transition for
    a windowed detector to see it in -- its silence would be a structural
    no-op, not a verdict. clear_detect.run_episode pads every episode with
    `margin_actions` of NOOP for this exact reason, and the live solver's
    replay_verify does the same with `clear_verify_margin()`. The hook is
    given the observations it needs to reach the verdict, and nothing
    else: the actions are NOOPs, so no new progress is driven.

    One observation = one ACTION (fs raw frames), which is the cadence the
    live solver's is_clear hook sees -- not the per-frame cadence the
    offline harness uses. A signal that only works at one of those two
    cadences would be a wiring artifact, so the live claim is made at the
    live cadence."""
    prof = yaml.safe_load(fx.profile.read_text())
    game = make_game(prof)
    space = [list(a) for a in prof["action_space"]]
    masks = action_space_to_bitmasks(space)
    fs = int(prof.get("frame_skip", 4))
    actions = np.load(fx.actions).tolist()

    env = nes_core.NESEnvironment(game.rom, frame_skip=1)
    env.reset()
    env.set_odometer_enabled(True)
    env.load_state(fx.root_state.read_bytes())
    for _ in range(fs):
        env.step(0)
    ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
    if hasattr(game, "note_start"):
        game.note_start(ram0)
    start_wd = tuple(game.level_key(ram0))

    ctx: dict = {}
    truth_action = None
    obs = []
    for i, a in enumerate(list(actions) + [0] * margin):
        m = int(masks[a])
        for _ in range(fs):
            env.step(m)
        ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        obs.append({"ram": ram, "odo": env.get_odometer(),
                    "scene": env.get_odometer_scene(),
                    "blank": env.get_odometer_blank(),
                    "oam": env.peek_oam()})
        if truth_action is None and game.is_clear(start_wd, ram, ctx):
            truth_action = i
    env.close()
    return obs, truth_action, game, prof


# ==========================================================================
# 1. The fixtures really do straddle a clear -- if this goes red the trace
#    is stale, not the detector. Guards every test below it.
# ==========================================================================

@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.name)
def test_the_fixture_really_is_a_witnessed_clear(fx: RealClear) -> None:
    _skip_if_missing(fx)
    hist, truth_action, _game, _prof = _replay(fx)
    assert truth_action is not None, (
        f"{fx.name}: the game's OWN predicate never fired on this trace -- "
        "the fixture is stale, regenerate it before trusting anything below")
    # Both traces were recorded to end exactly on the clearing action.
    assert truth_action == hist.shape[0] - 1


# ==========================================================================
# 2. The generic confluence mechanism, as every one of the 186 pre-existing
#    detector tests constructs it: StreamingConfluenceDetector(progress_fn),
#    no eligibility table, the exact call shape live_control.py used to
#    produce runs/clear_control_2026-08-26/{bb,tetris}_live.json.
#
#    xfail, not a bare assertion: the cause is already fully diagnosed in
#    this repo (clear_detect.py's COORD_RESET_DROP_MIN commentary, the
#    entity_wipe_windows section built for exactly this gap and marked
#    NOT_WIRED in docs/receipts/clear_control/cv_odometer_swap_v4_2026-08-26
#    .json) rather than a fresh discovery this file is making. strict=True
#    so a future fix that wires entity_wipe_windows (or otherwise widens the
#    live vote) turns this into a loud XPASS instead of a silent pass --
#    forcing the marker to be removed rather than forgotten.
# ==========================================================================

@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.name)
@pytest.mark.xfail(
    strict=True,
    reason="coord_entity_windows requires a position drop >= "
           "COORD_RESET_DROP_MIN (300 units); this game's own progress byte "
           "spans far fewer units than that across the ENTIRE trace (Bubble "
           "Bobble: 1 unit per round; Tetris-B: 0..32), so coord is "
           "arithmetically DEAD and tally alone can never reach "
           "min_signals=2. STILL XFAIL AFTER THE 2026-08-26 WIRE-UP, and "
           "the reason moved: the six shelf signals now reach this vote, "
           "but WIRED IS NOT ARMED and this construction "
           "(StreamingConfluenceDetector(progress_fn), no profile) arms "
           "none of them. The armed-profile counterpart is section 4 below, "
           "which fires. Note also that entity_wipe -- the corroborator the "
           "original version of this reason expected to close the gap -- "
           "could not have: it answers 'something emptied', which is what a "
           "DEATH looks like, so Rule 5 forbids it carrying a clear alone. "
           "What closes it is transition evidence (scene_cut).")
def test_the_generic_confluence_signal_misses_the_real_clear(fx: RealClear) -> None:
    _skip_if_missing(fx)
    hist, truth_action, _game, prof = _replay(fx)
    assert truth_action is not None

    det = StreamingConfluenceDetector(_game_progress(prof))
    fire_action = None
    for i, ram in enumerate(hist):
        if det.push(ram):
            fire_action = i
            break

    assert fire_action is not None, (
        f"{fx.name}: the generic confluence detector never fired on "
        f"{hist.shape[0]} real, driven observations that DID end in a "
        f"witnessed clear at action {truth_action}")
    assert abs(fire_action - truth_action) <= 30


def _game_progress(prof: dict):
    return make_game(prof).progress


# ==========================================================================
# 3. The fix that DOES generalize: forcing `clear.mode: confluence` onto
#    these same real profiles, clear_reachability's arithmetic gate refuses
#    construction outright (UnfireableHook) instead of silently building a
#    detector that can only ever return False. This is the real-RAM
#    counterpart to test_clear_quorum_reachability.py's
#    test_a_profile_whose_signals_cannot_reach_quorum_reports_unreachable_
#    not_a_miss, which makes the same claim from a hand-typed progress span
#    rather than a byte an emulator actually produced.
# ==========================================================================

@pytest.mark.parametrize("fx", FIXTURES, ids=lambda fx: fx.name)
def test_forcing_confluence_mode_is_refused_not_silently_built(fx: RealClear) -> None:
    _skip_if_missing(fx)
    prof = yaml.safe_load(fx.profile.read_text())
    forced = copy.deepcopy(prof)
    forced["solve"]["clear"] = {"mode": "confluence"}

    q = clear_reachability.clear_quorum(forced)
    assert q.verdict == clear_reachability.UNREACHABLE
    assert q.signal_state["coord"].state == clear_reachability.DEAD

    with pytest.raises(UnfireableHook):
        StreamingConfluenceDetector.from_profile(forced, lambda r: 0)


# ==========================================================================
# 4. THE ARMED PROFILE, at the LIVE cadence -- the counterpart to section 2.
#
# Section 2 pins what the GENERIC construction still misses. This pins what
# the profile's OWN armed signals do on the identical trace, because "the
# detector was improved" is not a claim anybody should accept without both
# halves: a wire-up that only ever moves the synthetic tests has not fixed
# anything, and a wire-up that quietly makes every profile fire has broken
# something else.
#
# Bubble Bobble arms scene_cut at a gate MEASURED from its own pre-clear
# play (docs/receipts/clear_control/bubble_bobble_scene_cut_null_2026-08-26
# .json: 226 checks, max d_scene 0, max d_blank 0). Tetris-B arms the same
# signal at the same gate and is NOT expected to land inside 30 actions --
# its screen turns over ~2.5 s after the quota byte reaches 0, which is a
# property of the game, not of the instrument.
# ==========================================================================

def _live_detector(prof: dict):
    return StreamingConfluenceDetector.from_profile(
        prof, make_game(prof).progress)


def _drive(det, obs) -> int | None:
    for i, o in enumerate(obs):
        if det.push(o["ram"], oam=o["oam"], odo=o["odo"], scene=o["scene"],
                    blank=o["blank"]):
            return i
    return None


def test_the_armed_bubble_bobble_profile_fires_on_the_real_clear() -> None:
    """GATE (b) on the LIVE vote, not just the offline harness. Same
    witnessed round-69 clear the generic detector misses in section 2,
    same trace, same emulator -- the profile's own armed signal is the
    only difference."""
    fx = BUBBLE_BOBBLE
    _skip_if_missing(fx)
    obs, truth_action, _game, prof = _replay_modalities(fx)
    assert truth_action is not None

    q = clear_reachability.clear_quorum(prof)
    assert q.signal_state["coord"].state == clear_reachability.DEAD
    assert q.signal_state["scene_cut"].state == clear_reachability.ALIVE
    assert q.verdict == clear_reachability.FIREABLE, (
        "coord is dead, and the profile is reachable ANYWAY -- that is the "
        "whole point of arming a second transition signal")

    det = _live_detector(prof)
    fired = _drive(det, obs)
    assert fired is not None, (
        f"the armed detector missed a witnessed clear at action "
        f"{truth_action} over {len(obs)} driven observations")
    assert 0 <= fired - truth_action <= 30, (
        f"fired at {fired}, truth at {truth_action}")
    assert det.shelf_stats()["scene_cut"]["n_triggers"] >= 1


def test_the_armed_detector_is_silent_for_the_whole_run_up_to_the_clear() -> None:
    """THE FALSE-POSITIVE HALF. A detector that fires early and often would
    pass the test above and be worthless. Drive only the PRE-clear portion
    -- 500+ real driven observations of ordinary play -- and require
    silence."""
    fx = BUBBLE_BOBBLE
    _skip_if_missing(fx)
    obs, truth_action, _game, prof = _replay_modalities(fx)
    assert truth_action is not None and truth_action > 100
    det = _live_detector(prof)
    fired = _drive(det, obs[:truth_action])
    assert fired is None, f"fired at action {fired}, before the clear"
    assert det.n_checks >= 14, "and it really was evaluated, repeatedly"


def test_arming_does_not_make_every_profile_fire() -> None:
    """THE OVER-CORRECTION GUARD. Tetris-B arms the same signal at the same
    measured gate and does NOT land within 30 actions of its clear: the
    quota byte hits 0 while the board is still on screen, and the SUCCESS
    curtain follows ~2.5 s later (blank folds measured at true_clear + 151
    and + 152 raw frames). The instrument is now capable of a positive on
    this profile and reports a LATE one; that is a different fact from the
    silence it used to return, and pretending otherwise by widening a
    tolerance is the move this campaign exists to refuse."""
    fx = TETRIS_B
    _skip_if_missing(fx)
    obs, truth_action, _game, prof = _replay_modalities(fx)
    assert truth_action is not None
    det = _live_detector(prof)
    fired = _drive(det, obs)
    assert fired is None or fired - truth_action > 30, (
        f"fired at {fired} against truth {truth_action}: if this now lands "
        "inside 30 actions the measurement above has changed and the "
        "profile comment in configs/tetris_b.yaml must be re-derived")
