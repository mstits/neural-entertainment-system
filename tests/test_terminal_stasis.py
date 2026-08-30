"""TerminalStasisSignal -- the D1 repair: a terminal state the lives byte
never reports.

D1, reproduced 2026-08-26 on configs/ninja_gaiden_ii.yaml from its own start
state (test_ng2_game_over_is_invisible_to_the_lives_predicate below is that
reproduction, run against the real ROM): the profile's discovered lives byte
$004C ticks 1 -> 0 -> 1 on every individual death, so `(start - cur) % 256 in
1..8` fires correctly on each of them -- and then GAME OVER arrives, $004C
sits at 1 forever, and the predicate is False on every subsequent frame. The
wave-4 smoke run banked 246,836 steps and 2,216 cells grinding that screen.

THE ORDER OF THIS FILE IS THE ARGUMENT, and it is deliberately
negative-controls-first:

  1. FIVE FALSE-POSITIVE CONTROLS. Each is a trace in which the terminal
     mechanism is ABSENT and the check must NOT fire, and each is written so
     that DELETING THE PART OF THE MECHANISM IT GUARDS makes it fail --
     verified by mutation, not asserted:
       * a cutscene / attract loop  -> input-dead but ANIMATING.
         Fails if the frozen ARMING track is dropped.
       * an idle player             -> frozen but INPUT-LIVE.
         Fails if the absorbing PROBE is dropped.
       * a scripted trigger         -> quiet, then input-independent motion.
         Perfect branch-to-branch agreement, so an input-lock reading fires
         here. Fails if the probe is scored on branch AGREEMENT instead of
         DRIFT FROM THE START FRAME -- this is the control that separates
         this signal from InputLockSignal.
       * pinned RAM, moving sprites -> fails if the probe's surface is cut
         back to CPU RAM and stops reading OAM.
       * bursty play (healthy median, quiet tail) -> fails if churn_tol is
         capped against the median alone and not against the quiet tail.
  2. the calibration refusals -- a profile this instrument cannot score must
     say so and stay disarmed, not accept a tolerance with nothing under it.
  3. only then the positive control, and the latency bound that is the whole
     point of the repair.

Every synthetic fixture is a duck-typed fake NESEnvironment whose evolution
rule is stated in the fixture itself (the same construction, and the same
purity-by-construction argument, as tests/test_input_lock_signal.py): nothing
in the synthetic half has ever seen a ROM. The last two tests replay the real
Ninja Gaiden II ROM -- one is the D1 reproduction, the other is the repair
firing on it -- and are skipped (never failed) when the ROM is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.clear_detect import (  # noqa: E402
    _lock_pair_diffs,
    _run_lock_branches,
    STASIS_TOL_FLOOR,
    STASIS_WINDOW,
    TerminalStasisSignal,
    measure_stasis_null,
    ram_surface,
)

RAM_SIZE = 2048
OAM_SIZE = 256
# A tiny synthetic action space: index 0 is NOOP, the rest arbitrary nonzero
# bitmasks -- the shape action_space_to_bitmasks produces from a profile.
BITMASKS = (0x00, 0x01, 0x02, 0x04, 0x08)

# Short knobs so a test runs in milliseconds. The RATIOS that matter are
# the shipped ones: a tolerance many times smaller than ordinary churn, and
# a probe branch several times LONGER than the arming window (45:150
# shipped, 24:80 here) -- the probe is where the evidence comes from.
WINDOW = 24
CALIB = 200
PROBE_STEPS = 80


class FakeEnv:
    """save_state / load_state / step / get_ram_range / peek_oam / get_audio
    -- the exact surface measure_stasis_null and _run_lock_branches call."""

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.n_steps = 0

    def save_state(self):
        return dict(self.state)

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
        return bytes(self._render()[1])

    def rule(self, state, mask):
        raise NotImplementedError

    def _render(self):
        raise NotImplementedError


def _ram_from(vals: dict) -> np.ndarray:
    ram = np.zeros(RAM_SIZE, dtype=np.uint8)
    for a, v in vals.items():
        ram[a] = v & 0xFF
    return ram


class LivePlayEnv(FakeEnv):
    """ORDINARY, RESPONSIVE PLAY. A position advances by (mask + 1) every
    step and 400 RAM bytes are stirred by a free-running clock, so both
    conjuncts are absent: the surface moves a lot AND it moves differently
    under different inputs."""

    def __init__(self):
        super().__init__({"pos": 0, "t": 0})

    def rule(self, state, mask):
        return {"pos": state["pos"] + mask + 1, "t": state["t"] + 1}

    def _render(self):
        t, pos = self.state["t"], self.state["pos"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = pos & 0xFF
        ram[1] = t & 0xFF
        # A busy game: several hundred bytes churn every step.
        ram[0x400:0x400 + 400] = ((np.arange(400) * 7 + t * 13 + pos) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = pos & 0xFF
        return ram, oam


class CutsceneEnv(FakeEnv):
    """FALSE-POSITIVE CONTROL 1 -- INPUT-DEAD BUT ANIMATING.

    A scripted sequence: every branch advances the same internal clock
    whatever the mask, so a differential input-lock probe reads LOCKED with
    total confidence (this is InputLockSignal's own documented
    false-positive class -- cutscene, scripted intro, pause, death
    animation, attract loop; the mechanism is identical in all of them and
    only the label differs). But the screen is BUSY: hundreds of bytes move
    every step, because something is being animated.

    The terminal check must NOT fire here, and the conjunct that stops it is
    FROZEN. Delete that conjunct and this test fails."""

    def __init__(self):
        super().__init__({"c": 0})

    def rule(self, state, mask):
        return {"c": state["c"] + 1}          # mask is never consulted

    def _render(self):
        c = self.state["c"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0x300:0x300 + 300] = ((np.arange(300) * 11 + c * 5) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = (c * 3) & 0xFF
        return ram, oam


class IdlePlayerEnv(FakeEnv):
    """FALSE-POSITIVE CONTROL 2 -- FROZEN BUT INPUT-LIVE.

    A fully responsive game in which the agent is holding NOOP. Nothing but
    a 1-byte frame counter moves while mask == 0, so the surface reads
    frozen to within the tolerance; the moment ANY nonzero mask is held the
    position advances and 300 bytes follow it. The solver reaches states
    like this constantly (--sticky 0.5 defaults to holding the previous
    action half the time, hold macros hold one action for tens of steps,
    and every macro is preceded by a NOOP settle).

    The terminal check must NOT fire here, and the conjunct that stops it is
    INPUT-DEAD. Delete that conjunct and this test fails."""

    def __init__(self):
        super().__init__({"pos": 0, "t": 0})

    def rule(self, state, mask):
        return {"pos": state["pos"] + (1 if mask else 0), "t": state["t"] + 1}

    def _render(self):
        pos = self.state["pos"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = pos & 0xFF
        # The frame counter is the ONLY input-independent motion, and it is
        # 1 byte -- under the tolerance floor, exactly like the 4 bytes the
        # real NG2 GAME OVER screen still moves.
        ram[1] = self.state["t"] & 0xFF
        ram[0x300:0x300 + 300] = ((np.arange(300) * 11 + pos * 9) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = pos & 0xFF
        return ram, oam


class ScriptedTriggerEnv(FakeEnv):
    """FALSE-POSITIVE CONTROL 3 -- QUIET, THEN A SCRIPTED SEQUENCE.

    The screen holds still (so the frozen track ARMS), and then, from the
    armed frame onward, an input-independent script runs: hundreds of bytes
    move and every branch moves them IDENTICALLY, whatever it holds. A boss
    walking on, a door opening, a room load, a cutscene that begins while
    the player is standing still -- the solver reaches these by idling at a
    trigger, which is exactly what `--sticky 0.5` and the NOOP settle before
    every hold macro produce.

    Every branch pair is byte-identical here, so the LOCKED half of the
    probe passes with total confidence. The state is nevertheless the
    OPPOSITE of absorbing: the game is going somewhere. The clause that
    stops it is NO BRANCH ESCAPES -- every branch has drifted far from the
    frame the probe started at. Delete that clause and this test fails."""

    def __init__(self, quiet: int):
        super().__init__({"t": 0})
        self.quiet = int(quiet)

    def rule(self, state, mask):
        return {"t": state["t"] + 1}          # mask is never consulted

    def _render(self):
        t = self.state["t"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        if t > self.quiet:                    # the script fires
            c = t - self.quiet
            ram[0x300:0x300 + 400] = ((np.arange(400) * 13 + c * 7) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        return ram, oam


class OamOnlyResponsiveEnv(FakeEnv):
    """FALSE-POSITIVE CONTROL 4 -- CPU RAM PINNED, SPRITES STILL RESPONDING.

    Nothing in the 2 KB of CPU RAM this signal's cheap track watches ever
    moves, so the frozen track arms and a drift score computed on the RAM
    prefix alone is exactly zero for every branch. But a 16-sprite
    metasprite -- a 32x32 character, or the scripted object
    `Solver.counterfactual_probe` reasons about -- follows the stick: its
    y and x bytes track the held mask, 32 OAM bytes in all.

    _lock_snapshot's own docstring names this class ("a branch that leaves
    CPU RAM alone but moves sprites") and it is why that snapshot reads
    RAM+OAM rather than the original RAM-only 2,048 bytes.

    THE HONEST LIMIT, since the fixture makes it concrete: the probe's
    tolerance is a byte COUNT, so a metasprite small enough to move fewer
    than `churn_tol` OAM bytes while CPU RAM is byte-pinned would still read
    absorbing. That is a narrow case -- every real game keeps the player's
    position in CPU RAM, so a pinned CPU RAM means a pinned player -- but it
    is a case, and it is not papered over."""

    N_SPRITES = 16

    def __init__(self):
        super().__init__({"held": 0})

    def rule(self, state, mask):
        return {"held": state["held"] + mask}

    def _render(self):
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)     # never moves
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        h = self.state["held"]
        for i in range(self.N_SPRITES):
            oam[4 * i] = (h + i) & 0xFF              # y
            oam[4 * i + 3] = (h * 3 + i) & 0xFF      # x
        return ram, oam


class BurstyPlayEnv(FakeEnv):
    """FALSE-POSITIVE CONTROL 5 -- A HEALTHY MEDIAN OVER A QUIET TAIL.

    Ordinary, fully responsive play that comes in phases: 40 busy steps
    (400 bytes churning) then 30 nearly-still ones, forever. Its MEDIAN
    window churn is perfectly healthy, so a tolerance set as a fraction of
    the median alone would sit far above what the quiet phases move -- and
    the arming track would then fire on ordinary play, every 70 steps, on a
    game that is not stuck at all.

    Real games look like this (a screen wipe, a between-wave lull, a boss
    walking on), which is why churn_tol is CAPPED at a fraction of the QUIET
    TAIL and not only of the median. Drop that cap and this test fails."""

    BUSY, QUIET = 40, 30

    def __init__(self):
        super().__init__({"t": 0, "pos": 0})

    def rule(self, state, mask):
        return {"t": state["t"] + 1, "pos": state["pos"] + mask + 1}

    def _render(self):
        t, pos = self.state["t"], self.state["pos"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = pos & 0xFF                       # responsive in both phases
        if t % (self.BUSY + self.QUIET) < self.BUSY:
            ram[0x400:0x400 + 400] = ((np.arange(400) * 7 + t * 13) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = pos & 0xFF
        return ram, oam


class GameOverEnv(FakeEnv):
    """POSITIVE CONTROL -- the D1 shape, synthetically.

    Responsive play for `alive` steps, then an absorbing screen: input stops
    mattering AND the surface stops moving except for one free-running
    counter byte. No lives byte changes -- there is no lives byte at all,
    which is the point: the terminal state is recognised from the two
    observables, not from any address anyone had to know about."""

    def __init__(self, alive: int = CALIB + WINDOW):
        super().__init__({"pos": 0, "t": 0})
        self.alive = int(alive)

    def rule(self, state, mask):
        if state["t"] >= self.alive:
            return {"pos": state["pos"], "t": state["t"] + 1}
        return {"pos": state["pos"] + mask + 1, "t": state["t"] + 1}

    def _render(self):
        pos, t = self.state["pos"], self.state["t"]
        ram = np.zeros(RAM_SIZE, dtype=np.uint8)
        ram[0] = pos & 0xFF
        ram[1] = t & 0xFF            # the counter that never stops
        ram[0x300:0x300 + 300] = ((np.arange(300) * 11 + pos * 9) % 256).astype(np.uint8)
        oam = np.zeros(OAM_SIZE, dtype=np.uint8)
        oam[0] = pos & 0xFF
        return ram, oam


def _signal(seed: int = 7, **kw) -> TerminalStasisSignal:
    kw.setdefault("window", WINDOW)
    kw.setdefault("probe_steps", PROBE_STEPS)
    return TerminalStasisSignal(BITMASKS, rng=np.random.default_rng(seed), **kw)


def _calibrated(env, seed: int = 7, **kw) -> TerminalStasisSignal:
    sig = _signal(seed, **kw)
    sig.calibrate(env, steps=CALIB)
    return sig


def _drive(sig, env, steps: int, mask: int | None, seed: int = 3):
    """Play `steps` steps (a fixed mask, or uniformly random when None),
    feeding every REAL frame to push() and running confirm() whenever it
    arms. Returns (n_armed, n_fired)."""
    rng = np.random.default_rng(seed)
    state: dict = {}
    armed = fired = 0
    for _ in range(steps):
        env.step(int(rng.choice(BITMASKS)) if mask is None else int(mask))
        if sig.push(state, env.get_ram_range(0, 2048)):
            armed += 1
            if sig.confirm(state, env):
                fired += 1
    return armed, fired


# ==========================================================================
# 1. FALSE-POSITIVE CONTROLS -- the mechanism is absent, nothing may fire
# ==========================================================================

def test_cutscene_is_input_dead_but_must_not_read_as_terminal():
    """CONTROL 1 (guards the FROZEN arming track).

    Asserts BOTH halves, so the test cannot pass vacuously:
      (a) the shipped branch-agreement reading really is LOCKED here -- every
          branch pair is byte-identical, so a signal scored the way
          InputLockSignal is scored would fire with total confidence, and
          this trace really is exercising that false-positive class;
      (b) the terminal check nevertheless never fires, over a run many
          windows long, because the screen is going somewhere."""
    sig = _calibrated(LivePlayEnv())          # thresholds from ordinary play
    assert sig.ready(), sig.reason

    scene = CutsceneEnv()
    for _ in range(WINDOW):                   # let it get going
        scene.step(0)
    # (a) the false-positive class is real: input has zero causal effect,
    # so every branch pair agrees to the byte.
    base = scene.save_state()
    snaps = _run_lock_branches(scene, BITMASKS, base, PROBE_STEPS, 4,
                               np.random.default_rng(0))
    scene.load_state(base)
    assert max(_lock_pair_diffs(snaps)) == 0

    # (b) ... and the conjunction still refuses.
    armed, fired = _drive(sig, scene, WINDOW * 8, mask=None)
    assert armed == 0, "an animating cutscene must never arm the frozen track"
    assert fired == 0


def test_idle_player_is_frozen_but_must_not_read_as_terminal():
    """CONTROL 2 (guards the ABSORBING probe).

    Again both halves:
      (a) holding NOOP really does freeze this surface, so the frozen track
          arms -- the trace exercises the class it claims to;
      (b) the confirming probe rejects every single arm, so nothing fires."""
    env = IdlePlayerEnv()
    sig = _calibrated(env)
    assert sig.ready(), sig.reason

    armed, fired = _drive(sig, env, WINDOW * 8, mask=0)
    assert armed > 0, "the idle surface must actually arm, or (b) proves nothing"
    assert fired == 0, "an idle player is not a terminal state"
    assert sig.n_rejected == armed


def test_a_scripted_sequence_that_starts_from_a_quiet_screen_must_not_fire():
    """CONTROL 3 -- THE VERDICT CLAUSE'S OWN CONTROL, and the one that
    separates this signal from an input-lock signal.

    A quiet screen ARMS, and then a scripted sequence runs from the armed
    frame: every branch moves identically whatever it holds. Branch-to-branch
    agreement is therefore PERFECT -- an input-lock reading fires here with
    total confidence -- and the state is the exact opposite of absorbing.

    Both halves are asserted:
      (a) the branches really do agree to the byte, so this trace is
          exercising the class it claims to;
      (b) the probe refuses anyway, because every branch has LEFT the frame
          it started from.
    Delete the drift clause and (b) fails."""
    sig = _calibrated(LivePlayEnv())
    assert sig.ready(), sig.reason

    env = ScriptedTriggerEnv(quiet=WINDOW + 2)
    state: dict = {}
    for _ in range(WINDOW + 1):
        env.step(0)
        armed = sig.push(state, env.get_ram_range(0, 2048))
    assert armed, "the quiet screen must arm, or (b) proves nothing"

    base = env.save_state()
    snaps = _run_lock_branches(env, BITMASKS, base, PROBE_STEPS, 4,
                               np.random.default_rng(0))
    env.load_state(base)
    assert max(_lock_pair_diffs(snaps)) == 0            # (a)

    fired = sig.confirm(state, env)
    assert min(sig.last_drift) > sig.churn_tol, sig.last_drift   # (b)
    assert fired is False, "a scripted sequence is not an absorbing state"


def test_pinned_ram_with_sprites_still_tracking_input_must_not_fire():
    """CONTROL 4 -- the probe must read OAM, not just CPU RAM.

    Nothing in the 2 KB the cheap arming track watches ever moves here, so
    the lineage arms and a drift score computed on the RAM prefix alone is
    exactly zero for every branch. A sprite is nevertheless still following
    the stick. Slice OAM off the probe's surface -- score drift on
    `s[:2048]` instead of the whole `_lock_snapshot` vector -- and this test
    fails."""
    sig = _calibrated(LivePlayEnv())
    env = OamOnlyResponsiveEnv()
    state: dict = {}
    for _ in range(WINDOW + 1):
        env.step(1)
        armed = sig.push(state, env.get_ram_range(0, 2048))
    assert armed, "CPU RAM never moves here, so the lineage must arm"

    fired = sig.confirm(state, env)
    assert fired is False, "a sprite still following the stick is not absorbing"
    assert max(sig.last_drift) > sig.churn_tol, sig.last_drift


def test_a_game_whose_ordinary_play_goes_quiet_must_not_arm_on_the_quiet():
    """CONTROL 5 (guards the QUIET-TAIL CAP on churn_tol).

    This game's median window churn is healthy, so `churn_frac * median`
    alone would set a tolerance far above what its regular quiet phases
    move. The tolerance is capped at a fraction of the QUIET TAIL as well,
    so the instrument either refuses this profile outright or arms too
    tightly to be fooled by it -- and either way ordinary play produces no
    arms at all.

    Drop the quiet-tail cap and the arming track fires every 70 steps on a
    game that is not stuck."""
    env = BurstyPlayEnv()
    sig = _calibrated(env)

    # It says WHY, whichever way it went.
    if not sig.ready():
        assert "cannot be told apart" in sig.reason
    else:
        assert sig.churn_tol * sig.quiet_mult <= np.quantile(sig.churn_null, 0.01)

    # The property that matters, and the one the cap delivers.
    armed, fired = _drive(sig, BurstyPlayEnv(), (BurstyPlayEnv.BUSY
                                                 + BurstyPlayEnv.QUIET) * 6,
                          mask=None)
    assert (armed, fired) == (0, 0), f"armed {armed} times on ordinary play"


def test_ordinary_random_play_never_arms():
    """The everyday case: a busy, responsive game under the solver's own
    random action stream never even reaches the confirming probe."""
    env = LivePlayEnv()
    sig = _calibrated(env)
    assert sig.ready(), sig.reason
    armed, fired = _drive(sig, env, WINDOW * 20, mask=None)
    assert (armed, fired) == (0, 0)


def test_a_rejected_probe_escalates_the_next_hold_this_lineage_must_earn():
    """COST CONTROL, and it is not cosmetic: each probe costs
    branches * probe_steps emulated steps (600 at the shipped knobs) against
    a 45-step window. A lineage the probe has already declined to kill --
    a long idle at a wall, the commonest thing a solver does -- must not
    re-ask every window forever, or the repair becomes a 13x tax on exactly
    the lineages it decided were alive.

    Delete the escalation (make required_hold return `window`) and the arm
    count below goes from 2 to 7."""
    env = IdlePlayerEnv()
    sig = _calibrated(env)
    state: dict = {}
    assert sig.required_hold(state) == WINDOW
    armed = 0
    for _ in range(WINDOW * 8):
        env.step(0)
        if sig.push(state, env.get_ram_range(0, 2048)):
            armed += 1
            assert sig.confirm(state, env) is False
    # 1st probe at WINDOW, 2nd at +WINDOW*4, 3rd would be at +WINDOW*16.
    assert armed == 2, f"re-armed {armed} times in {WINDOW * 8} steps"
    assert state["_stasis_rejects"] == 2
    assert sig.required_hold(state) == WINDOW * 16


def test_an_arm_that_survives_confirm_must_re_earn_its_hold():
    """A false arm re-anchors. Without this, one idle stretch would re-arm
    (and re-probe) on EVERY subsequent step off the same stale anchor,
    turning a tier-2 signal into a per-step cost."""
    env = IdlePlayerEnv()
    sig = _calibrated(env)
    state: dict = {}
    for _ in range(WINDOW + 1):
        env.step(0)
        armed = sig.push(state, env.get_ram_range(0, 2048))
    assert armed and state["_stasis_held"] == 0, "the arm must re-anchor"


# ==========================================================================
# 2. CALIBRATION REFUSALS -- an unscoreable profile must stay disarmed
# ==========================================================================

def test_a_game_with_no_resolution_left_refuses_to_arm():
    """A game whose ordinary play barely moves RAM cannot be told apart from
    a frozen screen by this instrument. It must report that and stay
    disarmed rather than arm on a tolerance with nothing under it."""

    class QuietEnv(FakeEnv):
        def __init__(self):
            super().__init__({"t": 0})

        def rule(self, state, mask):
            return {"t": state["t"] + (1 if mask else 0)}

        def _render(self):
            return _ram_from({0: self.state["t"]}), np.zeros(OAM_SIZE, dtype=np.uint8)

    sig = _calibrated(QuietEnv())
    assert not sig.ready()
    assert "cannot be told apart" in sig.reason
    # And a disarmed signal is inert, not merely quiet about it.
    assert sig.push({}, np.zeros(RAM_SIZE, dtype=np.uint8)) is False
    assert sig.confirm({}, QuietEnv()) is False


def test_probe_before_calibration_never_fires():
    sig = _signal()
    assert not sig.ready()
    assert sig.reason == "not calibrated"
    assert sig.push({}, np.zeros(RAM_SIZE, dtype=np.uint8)) is False


def test_calibration_shorter_than_the_window_is_reported_not_guessed():
    sig = _signal(window=500)
    sig.calibrate(LivePlayEnv(), steps=50)
    assert not sig.ready()
    assert "no lagged pair" in sig.reason


# ==========================================================================
# 3. POSITIVE CONTROL -- the mechanism is present and must be caught
# ==========================================================================

def test_absorbing_input_dead_screen_fires():
    env = GameOverEnv(alive=CALIB + 4)
    sig = _calibrated(env)                    # calibration ends just before it dies
    assert sig.ready(), sig.reason
    armed, fired = _drive(sig, env, WINDOW * 4, mask=None)
    assert armed >= 1
    assert fired >= 1


def test_a_state_already_proven_absorbing_is_not_reproven():
    """COST CONTROL with teeth. A game whose run ends on a CONTINUE screen
    sends every lineage to the same absorbing frame; re-proving it costs
    branches * probe_steps emulated steps each time (measured on 1942:
    123 probes in five minutes, and a 5x throughput loss). The second
    arrival at the same frame must be convicted from the ring, not
    re-simulated.

    Delete the ring (convicted_max=0 is the same code path) and the probe
    count below goes from 1 to 3."""
    sig = _calibrated(LivePlayEnv())

    def one_kill(seed):
        env = GameOverEnv(alive=0)          # absorbing from the first frame
        state: dict = {}
        rng = np.random.default_rng(seed)
        for _ in range(WINDOW * 3):
            env.step(int(rng.choice(BITMASKS)))
            if sig.push(state, env.get_ram_range(0, 2048)):
                assert sig.confirm(state, env) is True
                return True
        return False

    assert one_kill(1)
    assert sig.n_probes == 1 and sig.n_cached == 0
    assert one_kill(2) and one_kill(3)
    assert sig.n_probes == 1, "the same absorbing frame was re-simulated"
    assert sig.n_cached == 2
    assert sig.n_confirmed == 3


def test_the_convicted_ring_still_demands_the_full_arming_hold():
    """The ring shortcuts the PROBE, never the arming track. A live frame
    that happens to sit within churn_tol of a convicted one is not convicted
    until it has also held still for the whole window -- and a live game
    never does."""
    sig = _calibrated(LivePlayEnv())
    env = GameOverEnv(alive=0)
    state: dict = {}
    rng = np.random.default_rng(1)
    for _ in range(WINDOW * 3):
        env.step(int(rng.choice(BITMASKS)))
        if sig.push(state, env.get_ram_range(0, 2048)):
            sig.confirm(state, env)
            break
    assert sig._convicted, "nothing was convicted, so this proves nothing"
    # A busy game whose frames are nowhere near the convicted one: the ring
    # is consulted only on an ARM, and ordinary play never arms.
    armed, fired = _drive(sig, LivePlayEnv(), WINDOW * 10, mask=None)
    assert (armed, fired) == (0, 0)


def test_it_fires_within_about_one_window_of_the_state_becoming_absorbing():
    """The whole value of the repair is the cut-off latency: D1 cost 246,836
    steps. Assert it costs about one window, not an open-ended grind."""
    env = GameOverEnv(alive=CALIB + 4)
    sig = _calibrated(env)
    rng = np.random.default_rng(11)
    state: dict = {}
    for i in range(WINDOW * 4):
        env.step(int(rng.choice(BITMASKS)))
        if sig.push(state, env.get_ram_range(0, 2048)) and sig.confirm(state, env):
            assert i <= 2 * WINDOW + 4, f"fired only after {i} steps"
            return
    pytest.fail("never fired on an absorbing screen")


# ==========================================================================
# Per-lineage state is the CALLER's -- one calibrated signal, many workers
# ==========================================================================

def test_two_lineages_do_not_share_an_anchor():
    """A shared-index race is what dropped RoomFpTransitionSignal (a66cc74).
    One signal instance serves every worker in the pool; interleaving two
    lineages through it must not let either see the other's evidence."""
    env = IdlePlayerEnv()
    sig = _calibrated(env)
    a, b = {}, {}
    ram_idle = np.zeros(RAM_SIZE, dtype=np.uint8)
    busy = np.arange(RAM_SIZE, dtype=np.uint8)
    for i in range(WINDOW * 2):
        # lineage a holds still; lineage b churns every step
        assert sig.push(b, (busy + i).astype(np.uint8)) is False
        armed_a = sig.push(a, ram_idle)
        if i + 1 >= WINDOW:
            assert armed_a is True or a["_stasis_held"] < WINDOW
    assert sig.n_armed >= 1


def test_reset_drops_the_anchor_so_a_rewind_cannot_bank_a_hold():
    """Frames either side of a load_worker_state are not consecutive real
    play. Holding the anchor across one would credit a hold that never
    happened."""
    env = IdlePlayerEnv()
    sig = _calibrated(env)
    state: dict = {}
    ram = np.zeros(RAM_SIZE, dtype=np.uint8)
    for _ in range(WINDOW - 1):
        sig.push(state, ram)
    assert state["_stasis_held"] == WINDOW - 2
    sig.reset(state)
    assert "_stasis_anchor" not in state
    assert sig.push(state, ram) is False      # the run restarts from zero


def test_ram_surface_accepts_the_bytes_the_hot_loop_hands_out():
    """pool.step_all returns raw `bytes`; np.asarray(bytes, uint8) mangles
    it into a scalar (the ConfluenceDetector.push lesson, same file)."""
    raw = bytes(range(256)) * 8
    arr = ram_surface(raw)
    assert arr.dtype == np.uint8 and arr.shape == (2048,)
    assert int(arr[5]) == 5


def test_measure_stasis_null_advances_the_real_trajectory():
    """Calibration play is real play -- documented, and load-bearing: a
    caller that believed this restored the env would replay 400 steps it
    had already spent."""
    env = LivePlayEnv()
    null = measure_stasis_null(env, BITMASKS, window=WINDOW, steps=CALIB,
                               rng=np.random.default_rng(1))
    assert env.n_steps == CALIB
    assert null.size == CALIB - WINDOW
    assert null.min() > STASIS_TOL_FLOOR


# ==========================================================================
# 4. THE REAL D1 TRACE -- Ninja Gaiden II, against the actual ROM
# ==========================================================================

NG2 = REPO / "configs/ninja_gaiden_ii.yaml"


def _ng2_profile():
    import yaml
    prof = yaml.safe_load(NG2.read_text())
    return prof


def _ng2_missing() -> bool:
    if not NG2.exists():
        return True
    try:
        prof = _ng2_profile()
    except Exception:
        return True
    return not ((REPO / prof["rom_path"]).exists()
                and (REPO / prof["start_state_path"]).exists())


@pytest.mark.skipif(_ng2_missing(), reason="Ninja Gaiden II ROM/start state not present")
def test_ng2_game_over_is_invisible_to_the_lives_predicate():
    """THE D1 REPRODUCTION, and the test that fails if anyone 'fixes' D1 by
    widening the lives-byte arithmetic instead.

    Hold RIGHT from the profile's own start state. The lives predicate fires
    on the individual deaths (it is not broken -- it is BLIND), and then the
    game reaches GAME OVER and it is False for hundreds of consecutive steps
    while the screen is provably frozen."""
    import nes_core
    from go_explore_solve import make_game
    from src.training.profile_utils import action_space_to_bitmasks

    prof = _ng2_profile()
    game = make_game(prof)
    bm = action_space_to_bitmasks(prof["action_space"])
    pool = nes_core.Pool(rom_path=str(REPO / prof["rom_path"]), num_workers=1,
                         frame_skip=int(prof.get("frame_skip", 4)))
    try:
        pool.set_headless(True)
        pool.set_skip_preprocess(True)
        pool.reset_all()
        pool.load_worker_state(0, (REPO / prof["start_state_path"]).read_bytes())
        acts = np.zeros(1, dtype=np.uint8)
        pool.step_all(acts)
        start_lives = game.lives(pool.step_all(acts)[0][2])
        right = bm[prof["action_space"].index(["right"])]

        dead_flags, surfs = [], []
        for _ in range(1400):
            acts[0] = right
            ram = pool.step_all(acts)[0][2]
            dead_flags.append(bool(game.is_dead(ram, start_lives)))
            surfs.append(ram_surface(ram).copy())
    finally:
        pool.shutdown()

    # The predicate DOES work on individual deaths ...
    assert any(dead_flags[:700]), "the lives predicate should catch single deaths"
    # ... and is blind for the whole GAME OVER tail.
    tail = dead_flags[1000:]
    assert not any(tail), "expected the lives byte to be flat through GAME OVER"
    # The tail is frozen, by a margin of two orders of magnitude.
    churn = lambda i, w: int(np.count_nonzero(surfs[i] != surfs[i - w]))  # noqa: E731
    tail_churn = [churn(i, STASIS_WINDOW) for i in range(1200, 1400)]
    live_churn = [churn(i, STASIS_WINDOW) for i in range(STASIS_WINDOW, 500)]
    assert max(tail_churn) <= 32, f"GAME OVER tail churn {max(tail_churn)}"
    assert float(np.median(live_churn)) > 20 * max(tail_churn)


@pytest.mark.skipif(_ng2_missing(), reason="Ninja Gaiden II ROM/start state not present")
def test_ng2_the_repair_fires_on_the_screen_the_lives_predicate_missed():
    """THE END-TO-END CHECK, and the one the whole repair exists to pass.

    Calibrated from the profile's OWN start state -- never from the dead
    screen, which would measure a frozen surface and call it ordinary -- the
    signal must:

      (a) fire on the real GAME OVER screen, inside about one arming window,
          against the 246,836 steps D1 spent there; and
      (b) produce ZERO arms over 600 steps of live play out of the same
          start state, so (a) is not simply a check that always fires.

    Both halves matter. (a) alone would pass for a signal hardwired to True;
    (b) alone would pass for one hardwired to False."""
    import nes_core

    from src.training.profile_utils import action_space_to_bitmasks

    prof = _ng2_profile()
    bm = action_space_to_bitmasks(prof["action_space"])
    fs = int(prof.get("frame_skip", 4))
    rom = str(REPO / prof["rom_path"])
    blob = (REPO / prof["start_state_path"]).read_bytes()

    def fresh():
        e = nes_core.NESEnvironment(rom, frame_skip=fs)
        e.reset()
        e.load_state(blob)
        e.step(0)
        return e

    sig = TerminalStasisSignal(bm, rng=np.random.default_rng(1))
    sig.calibrate(fresh())
    assert sig.ready(), sig.reason
    # The calibration is a measurement, so say what it measured: NG2's
    # ordinary play is two orders of magnitude away from a frozen screen.
    assert sig.churn_tol >= STASIS_TOL_FLOOR
    assert sig.stats()["churn_null_p01"] >= 3 * sig.churn_tol

    # (a) drive to the GAME OVER screen the D1 reproduction above lands on,
    # then hand the signal the solver's own random action stream.
    env = fresh()
    right = bm[prof["action_space"].index(["right"])]
    for _ in range(850):
        env.step(right)
    rng = np.random.default_rng(5)
    state: dict = {}
    fired_at = None
    for t in range(4 * sig.window):
        env.step(int(rng.choice(bm)))
        if sig.push(state, env.get_ram_range(0, 2048)) and sig.confirm(state, env):
            fired_at = t + 1
            break
    assert fired_at is not None, "the repair missed the screen it was built for"
    assert fired_at <= 2 * sig.window, f"fired only after {fired_at} steps"
    assert max(sig.last_drift) <= sig.churn_tol, sig.last_drift

    # (b) the same calibrated signal, on live play from the same root.
    live = fresh()
    lstate: dict = {}
    armed = 0
    rng2 = np.random.default_rng(6)
    for _ in range(600):
        live.step(int(rng2.choice(bm)))
        if sig.push(lstate, live.get_ram_range(0, 2048)):
            armed += 1
            assert sig.confirm(lstate, live) is False
    assert armed == 0, f"live play armed {armed} times"


# ==========================================================================
# 5. THE SOLVER SIDE -- an absorbing cell must stop being SELECTED
# ==========================================================================
#
# Killing the burst alone is not the repair. D1's cost was not one burst: it
# was 15,579 revisits to one cell, and every one of those revisits would
# still happen if the cell stayed in the selection pool.

from types import MethodType, SimpleNamespace  # noqa: E402

from scripts.go_explore_solve import STASIS_RETIRE_AFTER, Solver  # noqa: E402


def _cell(gx: int):
    key = (0, 0, 0, (), 0, (), 0, 3, 1, 0, gx)
    return SimpleNamespace(key=key, state=b"s", best_score=1.0, best_steps=1,
                           visits=1, times_chosen=0, explored=False, barren=0)


def _sel_solver(cells, retired=None):
    """The duck-typed selection stand-in tests/test_go_explore_solve.py
    already uses for these arms -- only what _refresh_sel_cache reads."""
    f = SimpleNamespace(
        archive=SimpleNamespace(cells={c.key: c for c in cells}),
        max_area=0, max_sect=0, ortho_mode="off", sel_mode="legacy",
        door_weight=0.0, _doors=frozenset(), _key_ids={}, _gx_phantoms=set(),
        _ortho_pool=[], _ortho_ids=set(), _ortho_ext={},
        _ortho_deep_yband=None, _ortho_cols_improved=0, _ortho_selections=0,
        gate_mode="off", gate_weight=1.0, frontier_throttle=0,
        _sel_cells=None, _sel_n=0, _sel_area=None)
    if retired is not None:
        f._stasis_retired = set(retired)
    f._refresh_sel_cache = MethodType(Solver._refresh_sel_cache, f)
    return f


def test_a_retired_cell_leaves_the_selection_pool_but_stays_in_the_archive():
    cells = [_cell(0), _cell(16), _cell(32)]
    f = _sel_solver(cells, retired={cells[1].key})
    f._refresh_sel_cache()
    assert [c.key for c in f._sel_cells] == [cells[0].key, cells[2].key]
    # ... and the archive still holds it. The receipt for WHY a cell was
    # retired is the cell.
    assert len(f.archive.cells) == 3


def test_selection_is_untouched_when_nothing_is_retired_or_the_attr_is_absent():
    """Two paths that must both be the shipped one: an empty retired set,
    and a caller (every duck-typed stand-in in the suite) that has no such
    attribute at all."""
    cells = [_cell(0), _cell(16)]
    for retired in (set(), None):
        f = _sel_solver(cells, retired=retired)
        f._refresh_sel_cache()
        assert [c.key for c in f._sel_cells] == [c.key for c in cells]


class _FakeSig:
    """Always arms, always convicts -- this test is about what the SOLVER
    does with a verdict, not about how the verdict is reached."""

    window = STASIS_WINDOW

    def push(self, state, ram):
        return True

    def confirm(self, state, env):
        return True


def _kill_solver(**over):
    f = SimpleNamespace(
        _stasis=_FakeSig(), _stasis_env=SimpleNamespace(load_state=lambda b: None),
        pool=SimpleNamespace(save_worker_state=lambda wid: b"blob"),
        _stasis_arms=0, _stasis_kills=0, _stasis_errors=0,
        _stasis_retired=set(), _stasis_kill_counts={}, _sel_cells=["stale"])
    for k, v in over.items():
        setattr(f, k, v)
    f._stasis_terminal = MethodType(Solver._stasis_terminal, f)
    return f


def test_one_absorbing_burst_does_not_retire_the_cell_it_started_from():
    """A single unlucky continuation must not delete a live frontier cell:
    the kill is evidence about the action stream as much as about the cell,
    and a retired cell is one nobody visits again for the rest of the run."""
    f = _kill_solver()
    key = _cell(0).key
    assert f._stasis_terminal(0, {"key": key, "burst_step": 200}, b"\0" * 8)
    assert f._stasis_retired == set()
    assert f._sel_cells == ["stale"], "the selection cache must not be dropped"
    assert f._stasis_kill_counts == {key: 1}


def test_repeated_absorbing_bursts_from_one_cell_retire_it():
    """Two INDEPENDENT continuations both ending in a state no branch can
    leave is evidence about the CELL. This is D1's actual cost -- 15,579
    revisits to one doomed cell -- and nothing else in the solver removes it
    on a default run (`barren` is opt-in behind --frontier-throttle)."""
    f = _kill_solver()
    key = _cell(0).key
    for _ in range(STASIS_RETIRE_AFTER):
        assert f._stasis_terminal(0, {"key": key, "burst_step": 200}, b"\0" * 8)
    assert f._stasis_retired == {key}
    assert f._sel_cells is None, "the selection cache must be invalidated"
    # A different cell keeps its own count: the evidence is per cell.
    other = _cell(16).key
    f._stasis_terminal(0, {"key": other, "burst_step": 200}, b"\0" * 8)
    assert f._stasis_retired == {key}


def test_an_entrance_rooted_burst_has_no_cell_to_retire():
    """_assign's fallback path roots at the entrance with key None. Counting
    that against a cell key of None would retire the fallback itself."""
    f = _kill_solver()
    for _ in range(STASIS_RETIRE_AFTER + 2):
        assert f._stasis_terminal(0, {"key": None, "burst_step": 200}, b"\0" * 8)
    assert f._stasis_retired == set() and f._stasis_kill_counts == {}


def test_a_probe_that_cannot_run_keeps_the_lineage():
    """FAILS OPEN. A savestate that will not round-trip must never
    terminate a live lineage: a missed kill costs compute, a wrong kill
    costs a level nobody can reach any more."""
    def boom(wid):
        raise RuntimeError("savestate refused")

    f = _kill_solver(pool=SimpleNamespace(save_worker_state=boom))
    assert f._stasis_terminal(0, {"key": None, "burst_step": 1}, b"\0" * 8) is False
    assert (f._stasis_errors, f._stasis_kills) == (1, 0)


def test_an_unarmed_solver_never_touches_the_pool():
    """--stasis off, or a profile that declined to calibrate: the hot loop
    must cost one attribute test and nothing else."""
    def boom(wid):
        raise AssertionError("the pool must not be touched when disarmed")

    f = _kill_solver(_stasis=None, pool=SimpleNamespace(save_worker_state=boom))
    assert f._stasis_terminal(0, {"key": None, "burst_step": 1}, b"\0" * 8) is False
    assert f._stasis_arms == 0
