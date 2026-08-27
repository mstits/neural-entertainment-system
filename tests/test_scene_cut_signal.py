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
next several are pure/synthetic and need no ROM, including the death
discriminant (`lives` in push()) and the `n_events` re-anchor-ratchet
discriminant, each proved with its own mutation case. The real-emulator
tests that follow (skipped where their fixture is unavailable, exactly
like tests/test_clear_detect_ground_truth.py's own convention) are
where the false-positive classes (death, room transition) and the
true-positive case get their honest, measured proof against real
tapes -- an SMB power-on-1-1 clear, a real SMB death, and the Rygar R1
tape's 27-door re-anchor ratchet -- including one deliberate, documented
departure from the fixture this signal's design doc originally named
(see test_fires_near_the_real_level_key_advance_in_a_banked_smb_clear).
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


# ===========================================================================
# The death discriminant (`lives` in push()), synthetic and debounced.
#
# THE RYGAR SHAPE, verified on the real R1 tape replay
# (docs/receipts/rygar/r1_tape_gx6242.json): the declared lives byte
# ($0303) blips through 0 for exactly 2 observations on every one of 55
# real door crossings, recovering to 1 before the next check. A
# single-sample veto would discard all 55. go_explore_solve.py's
# `_dead_mm` debounce (commit 547434e) already established, on this same
# tape, that 2 is a blip and 3 is a death -- `death_debounce` reuses that
# threshold rather than inventing one.
# ===========================================================================

def test_death_veto_does_not_discard_a_two_observation_lives_blip() -> None:
    """The blip a real transition itself produces (2 observations, per
    the Rygar tape) must NOT veto the window it sits inside -- an
    undebounced veto would read every one of Rygar's 55 real transitions
    as a death and leave the signal permanently silent on the profile it
    was built for."""
    sig = SceneCutSignal(scene_min=1, blank_min=1, window=8, stride=1,
                         hold=5, death_debounce=3)
    for _ in range(3):
        sig.push((100, 0), scene=0, blank=0, lives=1)
    sig.push((100, 0), scene=0, blank=1, lives=0)     # blip obs 1
    sig.push((100, 0), scene=0, blank=2, lives=0)     # blip obs 2 -- still < 3
    sig.push((100, 0), scene=0, blank=3, lives=1)     # recovered
    for _ in range(4):
        sig.push((100, 0), scene=0, blank=3, lives=1)

    assert sig.n_triggers > 0, (
        "a 2-observation lives blip (the shape every real Rygar door "
        "crossing produces) must not be read as a death")
    assert sig.n_death_vetoes == 0


def test_death_veto_discards_a_sustained_three_observation_lives_drop() -> None:
    """A REAL death (>= death_debounce consecutive dropped-lives
    observations, the same threshold go_explore_solve.py's `_dead_mm`
    already pins) must be discarded: `n_triggers` stays 0 and
    `n_death_vetoes` counts it instead. Run WITHOUT `lives` on an
    otherwise-identical push sequence is the mutation proof: the same
    blank/scene shape fires when the veto has nothing to compare
    against, so the veto -- not some other gate -- is what suppresses
    it here."""
    def run(with_lives: bool) -> SceneCutSignal:
        # stride=3 == death_debounce: the check cadence does not outrun
        # the debounce, so the very first stride check after the drop
        # already sees a confirmed (>= 3 observation) death -- a stride
        # faster than the debounce would let a couple of checks land
        # inside the still-ambiguous grace window, which is a real and
        # separately-named property of any online debounce, not this
        # test's concern.
        sig = SceneCutSignal(scene_min=1, blank_min=1, window=8, stride=3,
                             hold=5, death_debounce=3)
        for _ in range(3):
            sig.push((100, 0), scene=0, blank=0,
                     lives=1 if with_lives else None)
        for k in range(1, 7):
            sig.push((100, 0), scene=0, blank=k,
                     lives=(0 if with_lives else None))
        return sig

    vetoed = run(with_lives=True)
    assert vetoed.n_triggers == 0, (
        "a sustained (>= death_debounce) lives drop must be vetoed, not "
        "counted as a transition")
    assert vetoed.n_death_vetoes > 0
    assert vetoed.last_death_vetoed is True

    unvetoed = run(with_lives=False)
    assert unvetoed.n_triggers > 0, (
        "control: the identical blank/scene shape must fire when `lives` "
        "is never supplied -- proving the veto above did the suppressing, "
        "not some other gate")


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
# n_events vs n_triggers -- the re-anchor-ratchet discriminant.
#
# THE RYGAR SHAPE, verified on the real R1 tape replay: 27 door-pairs
# banking dx=0 alternating with dx=+53..64, back-to-back, through what
# the room_fp census shows is mostly the same recycled background.
# `n_triggers` counts every qualifying STRIDE check (192, on that tape,
# for that stretch alone) and is NOT a room count. `n_events` -- a
# rising edge of the held-pulse `vote()` -- reads the entire tightly
# packed stretch as ONE continuous transition regime once `hold` is
# calibrated at or above the gap between consecutive doors, which is
# exactly what the real tape does (any hold >= SCENE_CUT_STRIDE=20
# collapses the whole 4608-6178 band to a single event).
# ===========================================================================

def _tightly_packed_doors(sig, n_doors: int = 12, gap: int = 3) -> int:
    """`n_doors` back-to-back re-anchor pairs (2 bumps then `gap` flat
    pushes each -- the Rygar door-pair shape), followed by ONE clearly
    separate event far away. Returns the final `blank` value."""
    blank = 0
    for _ in range(n_doors):
        blank += 1
        sig.push((0, 0), scene=0, blank=blank)
        blank += 1
        sig.push((0, 0), scene=0, blank=blank)
        for _ in range(gap):
            sig.push((0, 0), scene=0, blank=blank)
    for _ in range(50):
        sig.push((0, 0), scene=0, blank=blank)
    blank += 1
    sig.push((0, 0), scene=0, blank=blank)
    blank += 1
    sig.push((0, 0), scene=0, blank=blank)
    for _ in range(10):
        sig.push((0, 0), scene=0, blank=blank)
    return blank


def test_n_events_reads_a_tight_door_ratchet_as_one_regime_not_n_rooms() -> None:
    """A hold calibrated above the intra-ratchet gap (here: gap=3,
    window=4, hold=8) must read 12 tightly packed doors plus one
    genuinely separate later event as 2 events, never 13 -- the "naive
    reading is N rooms" failure this signal exists to not repeat."""
    sig = SceneCutSignal(scene_min=999, blank_min=1, window=4, stride=1, hold=8)
    _tightly_packed_doors(sig)
    assert sig.n_triggers > 40, "sanity: the raw stride-check tally is large"
    assert sig.n_events == 2, (
        f"expected the 12-door ratchet to collapse into one held regime "
        f"plus the one separate event (2 total), got {sig.n_events} -- "
        f"n_triggers ({sig.n_triggers}) is never the room count")


def test_n_events_inflates_when_hold_is_not_calibrated_above_the_gap() -> None:
    """Mutation proof: the SAME door sequence with `hold` too small to
    bridge the ratchet's own gap (hold=1, below the 3-push gap between
    doors) reads each door as its own event again -- proving the
    collapse above comes from `hold` being calibrated correctly, not
    from some other property of the sequence."""
    sig = SceneCutSignal(scene_min=999, blank_min=1, window=4, stride=1, hold=1)
    _tightly_packed_doors(sig)
    assert sig.n_events >= 10, (
        f"expected an uncalibrated hold to re-inflate toward one event "
        f"per door (12 doors + 1 separate), got only {sig.n_events} -- "
        f"this test no longer demonstrates the failure mode it exists to "
        f"guard against")


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


# ===========================================================================
# The real Rygar R1 tape -- the exact case both defenses above exist for.
# Skipped (never failed) when the ROM/start-state are absent, exactly like
# every other real-replay test in this file; the tape itself is tracked
# (docs/receipts/rygar/r1_tape_gx6242.json) so this guard is the only one
# that can skip.
# ===========================================================================

RYGAR_PROFILE = REPO / "configs/rygar.yaml"
RYGAR_TAPE = REPO / "docs/receipts/rygar/r1_tape_gx6242.json"


def _rygar_fixture():
    """`(profile_dict, tape_dict)`, or None with the caller expected to
    skip. Purity note: `profile["solve"]["lives"]` is the address this
    profile's own docstring says was discovered by scripts/
    discover_observables.py's 3-probe protocol, not recalled or guessed --
    the same address the live confluence detector already threads into
    RoomFpTransitionSignal for this and every other profile."""
    import yaml
    if not RYGAR_TAPE.exists() or not RYGAR_PROFILE.exists():
        return None
    profile = yaml.safe_load(RYGAR_PROFILE.read_text())
    rom = REPO / profile["solve"]["rom"]
    start = REPO / profile["start_state_path"]
    if not rom.exists() or not start.exists():
        return None
    tape = json.loads(RYGAR_TAPE.read_text())
    return profile, tape


@pytest.mark.skipif(_rygar_fixture() is None,
                    reason="Rygar ROM/start-state not present locally "
                           "(roms/ is gitignored); docs/receipts/rygar/"
                           "r1_tape_gx6242.json's own TestTapeRecord "
                           "coverage in test_rygar_r1_tape.py still runs")
def test_the_real_rygar_ratchet_collapses_to_a_handful_of_events_not_54() -> None:
    """The task this whole module exists for, replayed for real: the
    banked 6,018-action R1 tape (odometer_x terminal 6242, matching the
    receipt exactly) crosses 27 door-pairs (54 segments) back-to-back
    with none of them a real death (lives_at_start=1, dead_run_histogram
    {"2": 55} -- every dip is a 2-observation blip). At the shipped
    SCENE_CUT_WINDOW/STRIDE/HOLD (240/20/60, hold well above stride),
    `n_events` must stay small (a handful of held regimes across the
    WHOLE tape: the boot blackout, the one real door at x=1536, and the
    ratchet band read as one continuous regime) while `n_triggers` -- the
    raw stride-check tally -- is two orders of magnitude larger. Neither
    number is vetoed to zero: none of the 55 blips reaches
    death_debounce=3, so the death discriminant stays silent throughout,
    exactly as it must on a tape with no real death in it."""
    import numpy as np
    fixture = _rygar_fixture()
    assert fixture is not None
    profile, tape = fixture
    rom = REPO / profile["solve"]["rom"]
    start = REPO / profile["start_state_path"]
    lives_addr = profile["solve"]["lives"]
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile["frame_skip"])

    import nes_core
    env = nes_core.NESEnvironment(str(rom), frame_skip=fs)
    env.reset()
    env.set_odometer_enabled(True)
    env.load_state(start.read_bytes())

    sig = SceneCutSignal(scene_min=999, blank_min=5)
    try:
        for a in [0] + list(tape["actions"]):
            env.step(int(bitmasks[a]))
            odo = env.get_odometer()
            scene = env.get_odometer_scene()
            blank = env.get_odometer_blank()
            ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
            sig.push(odo, scene, blank, lives=int(ram[lives_addr]))
    finally:
        env.close()

    inv = tape["invariants_for_the_guard"]
    assert odo[0] == inv["terminal_odometer_x"], (
        "the tape must replay to its own recorded terminal odometer_x, "
        "or nothing below is measuring what it claims to")

    assert sig.n_death_vetoes == 0, (
        "none of this tape's 55 lives blips is a real death (all length "
        "2, debounce is 3) -- a nonzero veto count here means the "
        "debounce threshold stopped matching the measured Rygar shape")
    assert sig.n_triggers > 100, (
        "sanity: the raw stride-check tally over 27 door-pairs must be "
        "large -- this is the number a naive caller would mistake for a "
        "room count")
    assert sig.n_events <= 5, (
        f"expected the whole tape to collapse to a handful of held "
        f"regimes (boot + the one real door + the ratchet band), got "
        f"{sig.n_events} -- n_events inflating back toward n_triggers "
        f"means the hold-based collapse stopped working on the exact "
        f"tape it was calibrated against")
