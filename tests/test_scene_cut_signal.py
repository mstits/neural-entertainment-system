"""Tests for clear_detect.SceneCutSignal (the scene-cut / blank-fold
transition classifier -- CLEAR_DETECTION_CAMPAIGN_2026-08-26).

The signal turns the PPU scroll odometer's own re-anchor events into
transition evidence: a RENDERED scroll discontinuity bumps the scene
ordinal (nes_core odometer_scene), a BLACKOUT (< 120 rendered lines --
stage wipe, level-load blank, death fade) bumps the new odo_blank
counter. Neither branch integrates a position delta across itself; the
re-anchor EVENT is the evidence, never the (untrustworthy) delta.

Test order is deliberate. The FIRST test proves the signal CAN return
False -- the check two vacuous gates shipped this week without. The
next several are pure/synthetic and need no ROM. The last three replay
a real emulator session (skipped where the fixture is unavailable,
exactly like tests/test_clear_detect_ground_truth.py's own convention)
and are where the false-positive classes (death, room transition) and
the true-positive case get their honest, measured proof -- including
one deliberate, documented departure from the fixture this signal's
design doc originally named (see
test_fires_near_the_real_level_key_advance_in_a_banked_smb_clear).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ROM = REPO / "roms/Super Mario Bros. (World).nes"
# A banked Go-Explore SMB solve (power-on 1-1 -> 1-2), already one of
# clear_detect.py's own DEFAULT_RUNS ground-truth fixtures. `runs/` is
# gitignored (see .gitignore), so this is a local-environment fixture,
# not a committed one -- skip, don't fail, when it is absent.
SOL_BASE = REPO / "runs/regress_pre/solutions/sol_000"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.clear_detect import ACTION_SPACE, SceneCutSignal  # noqa: E402
from scripts.go_explore_solve import SmbGame  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402


# ===========================================================================
# FIRST: the signal must be able to return False.
# ===========================================================================

def test_scene_cut_does_not_fire_on_ordinary_steady_motion() -> None:
    """Proves the gate CAN fail closed, which is exactly the check two
    vacuous gates shipped this week without (see CLAIMS.md / the campaign
    doc's "ask what your test would report IF THE MECHANISM WERE ABSENT").

    A steady walk (odometer +1 px/observation, scene and blank both
    flat) accumulates a per-window delta of ~240 px at the default
    window -- squarely inside classify_transition's own pan_odo range
    (128-384), so classify_transition itself happily calls this window
    a "pan". The magnitude gate (scene_min/blank_min), NOT the kind
    filter, is what has to keep this silent: nothing re-anchored, so
    nothing should vote, no matter what label the classifier attaches
    to an odometer delta that is really just ordinary forward progress."""
    sig = SceneCutSignal(scene_min=1, blank_min=1)
    x = 0
    for _ in range(600):
        x += 1
        sig.push((x, 0), scene=0, blank=0)
        assert sig.vote() == 0
    assert sig.n_triggers == 0


# ===========================================================================
# Construction / bookkeeping (pure, no ROM).
# ===========================================================================

def test_construction_rejects_a_kind_outside_the_classifier_alphabet() -> None:
    with pytest.raises(ValueError):
        SceneCutSignal(scene_min=1, blank_min=1, kind=("pan", "not_a_real_kind"))


def test_scene_min_and_blank_min_have_no_default_and_must_be_supplied() -> None:
    """No defensible global default exists for either threshold -- Metroid
    was measured noisy at camera clamp/seam (real seam noise, not a real
    transition) and Zelda fades are invisible to the scene ordinal
    entirely, which is exactly why a profile must measure its OWN null
    and supply both numbers rather than this class guessing one. That
    discipline is only real if construction actually enforces it."""
    with pytest.raises(TypeError):
        SceneCutSignal()
    with pytest.raises(TypeError):
        SceneCutSignal(scene_min=1)
    with pytest.raises(TypeError):
        SceneCutSignal(blank_min=1)


def test_reset_drops_the_latch_and_the_rolling_window() -> None:
    sig = SceneCutSignal(scene_min=1, blank_min=1, window=8, stride=2, hold=10)
    for i in range(8):
        sig.push((0, 0), scene=i, blank=0)  # scene climbs -> a real warp shape
    assert sig.vote() == 1
    assert sig.n_triggers > 0

    sig.reset()

    assert sig.vote() == 0
    assert sig.trigger_step is None
    assert sig.n_triggers == 0
    assert sig.n_checks == 0
    assert len(sig._buf) == 0


def test_vote_is_a_held_pulse_not_a_latch() -> None:
    """ApuActivitySignal/RoomFpTransitionSignal's shape, not a one-shot
    latch: a single re-anchor event must eventually release the vote,
    or it would look identical on the fiftieth cut as on the first."""
    sig = SceneCutSignal(scene_min=1, blank_min=1, window=8, stride=1, hold=5)
    for _ in range(8):
        sig.push((0, 0), scene=0, blank=0)
    assert sig.vote() == 0, "no transition yet -- must not have fired"

    sig.push((0, 0), scene=1, blank=0)  # the one-off bump
    assert sig.vote() == 1, "must fire on the bump"

    # Scene held flat forever afterward: the window eventually forgets
    # the pre-bump baseline (window pushes) and the held pulse itself
    # expires (hold pushes past the LAST qualifying check) -- 50 more
    # pushes clears both with room to spare at these tiny window/hold.
    for _ in range(50):
        sig.push((0, 0), scene=1, blank=0)
    assert sig.vote() == 0, "a one-off event must not latch the vote on forever"


# ===========================================================================
# The documented false-positive classes, as positive assertions (pure).
# ===========================================================================

def test_fires_on_the_canonical_zelda_death_warp_shape() -> None:
    """DEATH and ROOM-TRANSITION-WITHOUT-PROGRESS are, to this signal, THE
    SAME EVENT: classify_transition's own docstring in go_explore_solve.py
    quotes the measured Zelda death signature verbatim -- "odometer modal
    16->272->16 (flat at settle), scene +2" -- which is exactly the `warp`
    branch (flat odometer, scene bump >= warp_scene_min). This test
    reproduces that shape (odometer flat, scene +2) with no real ROM,
    using only the pre-registered constants already receipted elsewhere
    in this repo, and asserts a fire.

    This is a POSITIVE assertion of a documented false positive, not a
    bug report: unlike RoomFpTransitionSignal (which excludes `warp` from
    its own default vote set to protect its adjacency/novelty bookkeeping)
    this signal has no such state to protect and does not exclude it by
    default. A caller arming `kind` with "warp" is choosing to accept the
    death/room-transition ambiguity -- the same shape apu_change/
    oam_quiesce/input_lock's own "fires on a synthesized death" tests
    pin for their signals. If this test ever goes green by the signal
    going silent instead of firing, it is no longer measuring this
    ambiguity at all."""
    sig = SceneCutSignal(scene_min=1, blank_min=1, window=8, stride=1, hold=5)
    for _ in range(4):
        sig.push((100, 0), scene=0, blank=0)   # flat odometer, pre-death
    sig.push((100, 0), scene=1, blank=0)        # the death flash
    sig.push((100, 0), scene=2, blank=0)        # scene +2, odometer still flat
    for _ in range(4):
        sig.push((100, 0), scene=2, blank=0)    # settled, flat

    assert sig.n_triggers > 0
    assert sig.last_kind == "warp"


def test_is_structurally_blind_to_a_combat_blip_shaped_change() -> None:
    """A combat blip (a wave of enemies despawning together) is an OAM/RAM
    entity-array phenomenon with no PPU-scroll or rendered-line signature
    at all. This signal never receives RAM or OAM -- only the odometer,
    scene ordinal and blank-fold count -- so there is no way to encode
    "a combat blip happened" into its inputs at all: holding all three
    perfectly flat is the closest any RAM-level churn could ever get to
    reaching this signal, and that must stay silent.

    Kept as its own test rather than folded into the ordinary-motion test
    above: "immune by construction" (no input surface exists for this FP
    class) and "happens not to fire on this particular input" are
    different claims, and this repo's own audit found gates that
    conflated the two."""
    sig = SceneCutSignal(scene_min=1, blank_min=1)
    for _ in range(300):
        sig.push((500, 500), scene=0, blank=0)
    assert sig.n_triggers == 0


# ===========================================================================
# Real-replay tests. Skipped (never failed) when the ROM or the local
# solve receipt is unavailable -- tests/test_clear_detect_ground_truth.py's
# own convention (`runs/` is gitignored; the ROM is a user-provided asset).
#
# All three share ONE real root: the banked power-on-1-1 gameplay state
# runs/regress_pre/solutions/sol_000.json already names as its
# `root_state` (this file's own DEFAULT_RUNS entry) -- never a bare
# post-reset save, which would be the title-screen demo
# (project_start_state_demo_bug_2026-05-28: "no start_state_path ->
# trains on title-screen demo").
# ===========================================================================

FRAME_SKIP = 4
_ROOT_ACTION_INDEX = ACTION_SPACE.index(["right", "A"])
_SOL_JSON = SOL_BASE.with_suffix(".json")


def _load_sol_root() -> tuple[dict, Path] | None:
    """`(meta, root_path)`, or None with the caller expected to skip."""
    if not _SOL_JSON.exists():
        return None
    meta = json.loads(_SOL_JSON.read_text())
    root_path = REPO / meta["root_state"]
    if not root_path.exists():
        return None
    return meta, root_path


def _bitmasks():
    return action_space_to_bitmasks(ACTION_SPACE)


def _pool_at_root(rom: str, root_bytes: bytes):
    import nes_core
    pool = nes_core.Pool(rom_path=rom, num_workers=1, frame_skip=FRAME_SKIP)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    pool.set_odometer_enabled(True)
    pool.load_worker_state(0, root_bytes)
    return pool


def _step(pool, bitmasks, action_idx: int) -> np.ndarray:
    mask = int(bitmasks[action_idx])
    result = pool.step_all(np.array([mask], dtype=np.uint8))
    ram = np.frombuffer(bytes(result[0][2]), dtype=np.uint8)
    return ram


def _odo_triple(pool):
    return (pool.get_odometer_per_worker()[0],
            pool.get_odometer_scene_per_worker()[0],
            pool.get_odometer_blank_per_worker()[0])


@pytest.mark.skipif(not ROM.exists(), reason="SMB ROM not present")
@pytest.mark.skipif(
    not _SOL_JSON.exists(),
    reason="runs/regress_pre solve receipt not present locally (runs/ is gitignored)")
def test_ignores_this_games_own_seam_noise_over_a_real_noop_and_forward_hold_drive() -> None:
    """The "does this game's own ordinary play look like a transition"
    check -- a real NOOP drive (standing still, verified clean: Mario
    survives it) followed by a SHORT forward-hold drive (running right
    while holding jump) from the same banked start-state this file's own
    ground-truth harness (DEFAULT_RUNS) already replays. Measured null
    over both drives: scene and blank both stay at exactly 0 for the
    whole 480 steps, so scene_min=blank_min=1 -- the very next integer
    above the measured null -- is a real calibrated choice, not a guess.

    (The forward-hold drive is deliberately short: continued long enough
    it runs Mario into a pit and dies, which is a REAL re-anchor event,
    not seam noise -- see test_fires_on_a_measured_death_mid_forward_hold
    below, which uses exactly that.)"""
    loaded = _load_sol_root()
    if loaded is None:
        pytest.skip("root state missing")
    _meta, root_path = loaded
    game = SmbGame()
    bitmasks = _bitmasks()

    pool = _pool_at_root(game.rom, root_path.read_bytes())
    sig = SceneCutSignal(scene_min=1, blank_min=1)
    try:
        for _ in range(400):
            _step(pool, bitmasks, 0)  # NOOP
            odo, scene, blank = _odo_triple(pool)
            sig.push(odo, scene, blank)
            assert sig.vote() == 0
        for _ in range(80):
            _step(pool, bitmasks, _ROOT_ACTION_INDEX)  # forward-hold, verified pit-free
            odo, scene, blank = _odo_triple(pool)
            sig.push(odo, scene, blank)
            assert sig.vote() == 0
    finally:
        pool.shutdown()

    assert sig.n_triggers == 0


@pytest.mark.skipif(not ROM.exists(), reason="SMB ROM not present")
@pytest.mark.skipif(
    not (SOL_BASE.with_suffix(".json").exists()
         and SOL_BASE.with_suffix(".actions.npy").exists()),
    reason="runs/regress_pre solve receipt not present locally (runs/ is gitignored)")
def test_fires_near_the_real_level_key_advance_in_a_banked_smb_clear() -> None:
    """The "fires on a real, labelled clear" oracle proof.

    DEPARTURE FROM THE DESIGN DOC, measured and documented rather than
    silently substituted: the doc named runs/cv_smoke/solutions/sol_000
    (Castlevania block 0 -> 1) as this fixture. Replaying it (root
    roms/Castlevania (USA)_start.state.bin, the recorded 368 actions,
    odometer armed) measured ZERO scene bumps and ZERO dropped folds for
    the entire replay -- the stage_number byte that defines this
    profile's level_key advances mid-stride, with the camera still
    scrolling continuously (matches the profile's own receipted note
    that block 0's max_gx "is the block's extent at the clear
    transition, not a frozen frontier" -- i.e. nothing visually cuts).
    That transition is real but INVISIBLE to this specific mechanism; it
    is not a usable "fires" oracle for scene_cut, and asserting a fire
    against it would be exactly the fabricated-positive result this
    campaign exists to purge.

    This test uses runs/regress_pre/solutions/sol_000 instead -- already
    one of THIS FILE's own DEFAULT_RUNS ground-truth fixtures (a banked
    power-on 1-1 -> 1-2 Go-Explore SMB clear) -- because replaying it
    (root + the recorded actions + a 200-action NOOP margin, since the
    flag-slide/fanfare/black "WORLD 1-2" card all happen AFTER the
    recorded trace ends) measures real dropped-fold clusters starting
    the observation right after the recorded level_key advance."""
    loaded = _load_sol_root()
    if loaded is None:
        pytest.skip("root state missing")
    meta, root_path = loaded
    game = SmbGame()
    bitmasks = _bitmasks()
    actions = np.load(str(SOL_BASE) + ".actions.npy").tolist()
    start_wd = tuple(meta["start_wd"])

    pool = _pool_at_root(game.rom, root_path.read_bytes())
    try:
        _step(pool, bitmasks, 0)  # rooting NOOP, matches run_episode's convention

        sig = SceneCutSignal(scene_min=1, blank_min=1)
        true_step = None
        votes = []
        for i, a in enumerate(actions + [0] * 200):
            ram = _step(pool, bitmasks, int(a))
            if true_step is None:
                lk = tuple(int(v) for v in game.level_key(ram))
                if lk != start_wd:
                    true_step = i
            odo, scene, blank = _odo_triple(pool)
            sig.push(odo, scene, blank)
            votes.append(sig.vote())
    finally:
        pool.shutdown()

    assert true_step is not None, "the banked solution never actually cleared on replay"
    rising = [i for i in range(1, len(votes)) if votes[i] and not votes[i - 1]]
    assert rising, "scene_cut never fired anywhere in the replay + margin"
    closest = min(abs(f - true_step) for f in rising)
    # Measured: fires 13 observations after the true advance at these
    # defaults. A generous multiple of that as the tolerance -- tight
    # enough to mean something, loose enough not to chase phase noise.
    assert closest <= 60, (
        f"nearest fire was {closest} observations from the true level_key "
        f"advance at step {true_step} (fires: {rising})")


@pytest.mark.skipif(not ROM.exists(), reason="SMB ROM not present")
@pytest.mark.skipif(
    not SOL_BASE.with_suffix(".json").exists(),
    reason="runs/regress_pre solve receipt not present locally (runs/ is gitignored)")
def test_fires_on_a_measured_death_mid_forward_hold() -> None:
    """Anti-vacuity control, as a positive assertion (the same discipline
    apu_change/oam_quiesce/input_lock's own death tests use): this
    signal CANNOT discriminate a death from a clear, and this test
    proves it fires on a REAL, measured death rather than merely
    claiming so.

    A blind forward-hold (right+A, no adaptation) from the same banked
    power-on-1-1 root runs Mario into a pit within ~100 observations
    (measured: repeated dropped-fold clusters around observations
    100-140, 240-280, 380 -- the games's own "MARIO x N" life-card
    blackout between a death and the respawn). At the SAME calibration
    this file's null test uses (scene_min=blank_min=1), that death
    fires -- proof that a caller putting this signal in `require`
    without a `lives_drop` veto is accepting exactly this ambiguity,
    not something this signal can resolve on its own."""
    loaded = _load_sol_root()
    if loaded is None:
        pytest.skip("root state missing")
    _meta, root_path = loaded
    game = SmbGame()
    bitmasks = _bitmasks()

    pool = _pool_at_root(game.rom, root_path.read_bytes())
    sig = SceneCutSignal(scene_min=1, blank_min=1)
    try:
        for _ in range(200):
            _step(pool, bitmasks, _ROOT_ACTION_INDEX)
            odo, scene, blank = _odo_triple(pool)
            sig.push(odo, scene, blank)
    finally:
        pool.shutdown()

    assert sig.n_triggers > 0, (
        "expected the blind forward-hold to die and re-anchor at least "
        "once in 200 observations -- if it no longer does, this test's "
        "premise (a measured death, not a synthetic one) needs a new "
        "action budget, not a change to the assertion")
