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
    (ram_hist[T,2048] uint8, truth_action, game). `truth_action` comes from
    the game's OWN is_clear -- level_key for Bubble Bobble, byte_change for
    Tetris-B -- neither of which is the confluence mechanism under test."""
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
           "min_signals=2. Will pass once a position-free corroborator "
           "(clear_detect.entity_wipe_windows, built+tested 2026-08-26 but "
           "NOT_WIRED per docs/receipts/clear_control/cv_odometer_swap_v4_"
           "2026-08-26.json) is wired into the live vote for profiles where "
           "clear_reachability.clear_quorum marks coord DEAD on arithmetic.")
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
