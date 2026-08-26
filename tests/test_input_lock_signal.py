"""InputLockSignal -- the K-branch, self-calibrated generalization of the
differential input-lock probe (clear_detect.differential_input_lock_probe),
armed for in-loop use per docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md.

Every fixture here is a duck-typed fake NESEnvironment (save_state /
load_state / step / get_ram_range / peek_oam / get_audio, the exact surface
_run_lock_branches and measure_input_lock_null call) whose RAM+OAM evolution
rule is stated in the fixture itself, not borrowed from any real game --
purity by construction: nothing here has ever seen a ROM.

THE FIRST TEST BELOW (test_probe_reports_unlocked_when_ram_tracks_input) is
the CAN-FAIL check: it feeds the probe a stream where the locking mechanism
is structurally absent (RAM genuinely depends on which branch's input was
held) and asserts the signal does NOT fire. A signal that cannot fail this
is measuring nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.clear_detect import (
    InputLockSignal,
    LOCK_FRAC,
    differential_input_lock_probe_k,
    input_lock_preflight,
    lock_null_threshold,
    measure_input_lock_null,
)

RAM_SIZE = 2048
OAM_SIZE = 256
# A tiny, fully synthetic action space: index 0 is NOOP (bitmask 0x00), the
# rest are arbitrary nonzero bitmasks -- exactly the shape
# action_space_to_bitmasks produces from a profile's YAML.
BITMASKS = (0x00, 0x01, 0x02, 0x04, 0x08)


class FakeEnv:
    """A minimal duck-typed NESEnvironment.

    `rule(state, mask) -> new_state` is the ONLY game-specific thing any
    fixture below supplies; everything else (save/load/step/audio/RAM/OAM
    extraction) is the same plumbing every fixture shares, matching the real
    NESEnvironment's save_state/load_state/step/get_ram_range/peek_oam/
    get_audio surface exactly (nes_core/src/python.rs)."""

    def __init__(self, rule, state=None):
        self.rule = rule
        self.state = state if state is not None else {"t": 0, "mask_sum": 0}
        self.n_steps = 0

    def save_state(self):
        return dict(self.state)  # a plain dict copy is our "state blob"

    def load_state(self, blob):
        self.state = dict(blob)

    def step(self, mask):
        self.n_steps += 1
        self.state = self.rule(self.state, int(mask))

    def get_audio(self):
        return np.zeros(0, dtype=np.int16)

    def get_ram_range(self, start, length):
        return self._render()[0]

    def peek_oam(self):
        ram, oam = self._render()
        return bytes(oam)

    def _render(self):
        raise NotImplementedError


class ResponsiveEnv(FakeEnv):
    """ORDINARY, UNLOCKED play: RAM byte 0 tracks a position that advances by
    `mask` every frame, RAM byte 1 is a free-running frame counter (advances
    identically regardless of mask -- every real game has bytes like this,
    e.g. a music-driver tick), and OAM byte 0 mirrors the position mod 256
    (a sprite that visibly follows the tracked object). Different masks
    produce genuinely different trajectories: this is the CAN-FAIL fixture.
    """

    def _render(self):
        pos = self.state.get("pos", 0) & 0xFF
        t = self.state.get("t", 0) & 0xFF
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = pos
        ram[1] = t
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = pos
        return ram, oam


def _responsive_rule(state, mask):
    return {"pos": state.get("pos", 0) + mask + 1, "t": state.get("t", 0) + 1}


def make_responsive_env():
    return ResponsiveEnv(_responsive_rule, {"pos": 0, "t": 0})


class FrozenEnv(FakeEnv):
    """INPUT-INDEPENDENT RAM/OAM evolution: every branch advances the SAME
    internal clock `c` regardless of `mask`, and RAM/OAM are pure functions
    of `c` alone. This one fixture class stands in for every false-positive
    class the signal cannot discriminate (pause, cutscene, death animation,
    attract loop) -- the mechanism (input has zero causal effect on the
    observable surface) is identical in all four; only the label differs."""

    def _render(self):
        c = self.state.get("c", 0) & 0xFF
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = c
        ram[5] = (c * 3) % 256
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = 255 if c < 30 else 0  # e.g. a sprite fading out over time
        return ram, oam


def _frozen_rule(state, mask):
    return {"c": state.get("c", 0) + 1}  # mask is never consulted


def make_frozen_env():
    return FrozenEnv(_frozen_rule, {"c": 0})


# --------------------------------------------------------------------------
# THE CAN-FAIL TEST -- must come first, must genuinely fail if the mechanism
# (branch divergence under different inputs) is stubbed out.
# --------------------------------------------------------------------------

def test_probe_reports_unlocked_when_ram_tracks_input():
    """Ordinary, responsive play: RAM/OAM genuinely differ branch-to-branch
    because each branch held a different action. The probe MUST NOT fire
    LOCKED. This is the CAN-FAIL check -- a threshold or fraction rule that
    always reports LOCKED would pass every other test in this file yet be
    worthless, and this is the one test that catches that."""
    env = make_responsive_env()
    rng = np.random.default_rng(0)
    null = measure_input_lock_null(env, BITMASKS, probe_frames=10, branches=4,
                                    n_samples=20, drive_frames=5, rng=rng)
    assert null.size > 0
    threshold = lock_null_threshold(null, quantile=0.01)

    locked, frac, diffs = differential_input_lock_probe_k(
        env, BITMASKS, threshold, probe_frames=10, branches=4,
        lock_frac=LOCK_FRAC, rng=rng)

    assert diffs, "probe produced no branch pairs at all"
    assert not locked, (
        f"responsive/unlocked play read as LOCKED (frac={frac}, "
        f"threshold={threshold}, diffs={diffs}) -- the signal cannot "
        f"return False and is measuring nothing")


def test_probe_can_return_false_across_repeated_ordinary_probes():
    """Not a one-shot fluke: repeat the ordinary-play probe several times
    (advancing real play between each, exactly like the in-loop hot path
    would) and demand UNLOCKED every single time."""
    env = make_responsive_env()
    rng = np.random.default_rng(1)
    null = measure_input_lock_null(env, BITMASKS, probe_frames=8, branches=4,
                                    n_samples=15, drive_frames=5, rng=rng)
    threshold = lock_null_threshold(null, quantile=0.01)
    for _ in range(5):
        for _ in range(6):
            env.step(int(rng.choice(BITMASKS)))
        locked, frac, _ = differential_input_lock_probe_k(
            env, BITMASKS, threshold, probe_frames=8, branches=4,
            lock_frac=LOCK_FRAC, rng=rng)
        assert not locked, f"ordinary play mis-read as LOCKED (frac={frac})"


# --------------------------------------------------------------------------
# The three false-positive classes named in the signal spec, asserted as
# POSITIVE claims: the signal DOES fire on each, on the record, so a
# validator can require lives_drop / attract_loop vetoes rather than trust
# this signal alone.
# --------------------------------------------------------------------------

def _calibrated_signal(env, rng):
    sig = InputLockSignal(BITMASKS, probe_frames=10, branches=4,
                           quantile=0.01, lock_frac=LOCK_FRAC, rng=rng)
    sig.calibrate(env, n_samples=20, drive_frames=5)
    return sig


def test_input_lock_reports_unlocked_on_ordinary_play():
    """PREFLIGHT shape: from a start state, after settle_steps of ordinary
    play, the signal must report UNLOCKED. A game that reads LOCKED here is
    either stuck in an attract loop or wired to a broken probe."""
    env = make_responsive_env()
    rng = np.random.default_rng(2)
    sig = InputLockSignal(BITMASKS, probe_frames=10, branches=4,
                           quantile=0.01, lock_frac=LOCK_FRAC, rng=rng)
    result = input_lock_preflight(sig, env, settle_steps=30)
    assert result["ok"], result
    assert not result["locked"]


def test_input_lock_reports_locked_on_a_paused_state():
    """Pause: input is explicitly disabled. Every branch (NOOP and every
    sampled action alike) lands on the identical frozen RAM/OAM -- the
    probe must report LOCKED."""
    env = make_frozen_env()
    rng = np.random.default_rng(3)
    sig = _calibrated_signal(env, rng)
    locked = sig.probe(env)
    assert locked, sig.stats()
    assert sig.last_frac == pytest.approx(1.0)


def test_input_lock_reports_locked_on_a_death_state():
    """FP CONTROL, stated as a positive assertion (not a bug report): a
    death animation runs identically regardless of input, so this signal
    CANNOT discriminate a death from a genuine lock, and must be shown
    firing on one so a caller is never tempted to put this in `require`
    without a `lives_drop` veto alongside it."""
    env = make_frozen_env()  # same input-independent mechanism as pause/death
    rng = np.random.default_rng(4)
    sig = _calibrated_signal(env, rng)
    locked = sig.probe(env)
    assert locked, (
        "this test is REQUIRED to pass -- if it goes green by the signal "
        "being silent instead, the FP-control assertion below it is "
        "meaningless")


def test_input_lock_reports_locked_on_a_cutscene_or_attract_loop():
    """SCRIPTED INTRO / CUTSCENE / ATTRACT LOOP -- named separately in the
    spec from pause/death, but mechanically identical (input-independent
    RAM/OAM), so the fixture is the same class again. The distinct name
    matters at the wiring layer (an attract loop's lock window never
    releases, which is what a caller's `probe_every` budget / a separate
    attract_loop veto has to catch), not at this signal's own decision
    rule."""
    env = make_frozen_env()
    rng = np.random.default_rng(5)
    sig = _calibrated_signal(env, rng)
    for _ in range(3):  # an attract loop's lock never releases
        assert sig.probe(env)


# --------------------------------------------------------------------------
# Mechanics: threshold direction, OAM inclusion, fraction-of-pairs voting.
# --------------------------------------------------------------------------

def test_lock_null_threshold_sits_at_the_low_tail_not_the_middle():
    """CORRECTION 1's whole point: the threshold is a LOW quantile of a
    distribution that runs into the hundreds, not some mid-range 'typical'
    value -- otherwise ordinary play would trip it constantly."""
    rng = np.random.default_rng(6)
    null = rng.integers(150, 400, size=500).astype(np.int64)  # "hundreds of bytes"
    threshold = lock_null_threshold(null, quantile=0.01)
    assert threshold < np.median(null)
    assert threshold <= np.quantile(null, 0.02) + 1  # near the bottom sliver


def test_lock_null_threshold_is_not_the_retired_global_constant():
    """The shipped LOCK_DIFF_TOL=2 must not silently reappear as this
    function's answer on a game whose real null lives nowhere near 2."""
    rng = np.random.default_rng(7)
    null = rng.integers(200, 600, size=300).astype(np.int64)
    threshold = lock_null_threshold(null, quantile=0.01)
    assert threshold > 2


def test_probe_uses_oam_not_only_ram():
    """CORRECTION 2's '(and OAM)': a branch that leaves CPU RAM identical
    but moves sprites must not be read as more locked than it is. Build an
    env whose RAM is CONSTANT across every branch but whose OAM position
    tracks the held mask -- if OAM were ignored, every pair would show diff
    0 (fully locked); with OAM included, branches holding different masks
    diverge in OAM and the pair reads unlocked."""

    class OamOnlyEnv(FakeEnv):
        def _render(self):
            pos = self.state.get("pos", 0) & 0xFF
            ram = np.zeros(RAM_SIZE, dtype=np.uint8)  # RAM never varies
            oam = np.zeros(OAM_SIZE, dtype=np.uint8)
            oam[0] = pos
            return ram, oam

    def rule(state, mask):
        return {"pos": state.get("pos", 0) + mask + 1}

    env = OamOnlyEnv(rule, {"pos": 0})
    rng = np.random.default_rng(8)
    # A branch-diff of 0 is a certainty if OAM is included and masks differ
    # (NOOP holds pos flat, any nonzero-masked branch does not); force
    # branches to sample only nonzero masks so every non-control pair
    # necessarily diverges from the frozen control.
    locked, frac, diffs = differential_input_lock_probe_k(
        env, (0x01, 0x02, 0x04, 0x08), threshold=0, probe_frames=5,
        branches=4, lock_frac=LOCK_FRAC, rng=rng)
    assert any(d > 0 for d in diffs), (
        "every pair read diff=0 -- OAM is not being consulted by the probe")


def test_locked_fraction_is_computed_over_all_branch_pairs():
    """A fixture where each branch's held mask is written as a one-hot bit
    across 8 RAM bytes: two branches holding the SAME mask read identical
    (diff 0), control (mask 0) vs. any single-bit branch differs in exactly
    1 byte, and two DIFFERENT single-bit branches differ in exactly 2 bytes.
    At threshold=1 the 3 pairs touching control read "under threshold" and
    the 3 pairs among the diverging branches do not. With branches=4 there
    are 6 pairs; 3/6 = 0.5 must NOT count as locked, because lock_frac=0.5
    (the default) requires the fraction to EXCEED 0.5, not equal it."""

    class MixedEnv(FakeEnv):
        def _render(self):
            last_mask = self.state.get("last_mask", 0) & 0xFF
            ram = np.zeros(RAM_SIZE, dtype=np.uint8)
            for i in range(8):
                ram[i] = (last_mask >> i) & 1
            oam = np.zeros(OAM_SIZE, dtype=np.uint8)
            return ram, oam

    def rule(state, mask):
        return {"last_mask": mask}

    env = MixedEnv(rule, {"last_mask": 0})
    # Force the 3 non-control branches to draw 3 DISTINCT single-bit masks
    # by feeding a tiny rng-like stub that cycles through them
    # deterministically (a real rng could collide branches and would test a
    # different, also-real scenario -- see the sampling-collision test below).
    calls = {"i": 0}
    distinct = [0x01, 0x02, 0x04]

    class _SeqRng:
        def choice(self, arr):
            v = distinct[calls["i"] % len(distinct)]
            calls["i"] += 1
            return v

    locked, frac, diffs = differential_input_lock_probe_k(
        env, BITMASKS, threshold=1, probe_frames=5, branches=4,
        lock_frac=0.5, rng=_SeqRng())
    assert len(diffs) == 6
    n_under = sum(1 for d in diffs if d <= 1)
    assert n_under == 3, diffs
    assert frac == pytest.approx(0.5)
    assert not locked, "3/6 == lock_frac must not count as EXCEEDING it"


def test_probe_restores_env_to_its_pre_probe_state():
    """None of this ever happened on the real timeline: after probe(), env
    must read exactly as it did before the probe ran."""
    env = make_responsive_env()
    env.state = {"pos": 17, "t": 4}
    before = env.save_state()
    rng = np.random.default_rng(9)
    differential_input_lock_probe_k(env, BITMASKS, threshold=5,
                                     probe_frames=6, branches=3, rng=rng)
    assert env.save_state() == before


def test_probe_raises_without_calibration_rather_than_using_a_default():
    """CORRECTION 1's other half: no silent fallback to any built-in
    threshold. A profile that never calibrated must fail loudly."""
    env = make_responsive_env()
    sig = InputLockSignal(BITMASKS)
    with pytest.raises(RuntimeError):
        sig.probe(env)


# --------------------------------------------------------------------------
# A residual false-positive found while building this: sampling collisions.
# Documented as a positive assertion, same discipline as the death/pause
# tests above, so nobody "fixes" it later without noticing what breaks.
# --------------------------------------------------------------------------

def test_duplicate_branch_actions_can_read_as_a_locked_pair_by_chance():
    """`bitmasks` is sampled WITH REPLACEMENT per branch (see
    _run_lock_branches's docstring). On a small action space, two
    non-control branches can legitimately draw the IDENTICAL action by
    chance during otherwise fully ordinary, responsive play -- and that one
    pair reads as perfectly locked (diff 0) even though nothing about the
    game is locked. With only 2 nonzero actions and 3 non-control branches,
    a repeat is forced (pigeonhole), so this is not a rare edge case on a
    game with a tiny action space; it is guaranteed every single probe."""

    class _FixedPairRng:
        """Deterministically returns the SAME nonzero action for every
        non-control draw, to make the collision unconditional rather than
        probabilistic (a real RNG would only do this sometimes)."""

        def choice(self, arr):
            return 0x01

    env = make_responsive_env()
    locked, frac, diffs = differential_input_lock_probe_k(
        env, (0x00, 0x01, 0x02), threshold=0, probe_frames=10, branches=3,
        lock_frac=LOCK_FRAC, rng=_FixedPairRng())
    # branches 1 and 2 both held 0x01 for the whole probe: their pair must
    # read as an exact, coincidental lock even though play is ordinary.
    assert 0 in diffs, (
        "expected the identical-action pair to read diff=0; if this ever "
        "fails, note it in the report -- the collision risk may have "
        "changed shape, not disappeared")
