"""Clear-detector v3 — the two D3-adopted mechanisms, both opt-in.

  1. The APU CHANNEL-ACTIVITY vote (clear_detect.ApuActivitySignal): a
     sustained, coordinated change in this game's per-channel activity vector
     measured against a null this run self-measured from its own history. The
     verdict that adopted it explicitly killed the content-prior version ("a
     fanfare sounds like X"), so what is tested here is exclusively
     self-referential behavior: the same absolute change means different
     things to a game whose channels churn and a game whose channels do not,
     and a transient shorter than the window's own resolution cannot fire at
     any alignment.

  2. The COUNTERFACTUAL PERTURBATION PROBE as a candidate gate
     (Solver.counterfactual_probe): K short branches re-simulated from a
     pre-clear state with perturbed inputs, on the BANKING path only, at its
     true multi-second cost. Never in-loop — the same verdict that adopted it
     killed the in-loop form on measured cost.

Both default OFF, and the default-identity of every v1/v2 path is pinned here
as well as in test_confluence_v2.py (which must stay green untouched).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.clear_detect import (
    APU_GATE_FLOOR,
    APU_MIN_BASELINE,
    APU_SHORT_WINDOW,
    ApuActivitySignal,
    StreamingConfluenceDetector,
)
from scripts.go_explore_solve import (
    GenericGame,
    SmbGame,
    Solver,
    resolve_apu_sampling,
    resolve_counterfactual_gate,
)

# ==========================================================================
# Part 1 -- the APU channel-activity signal
#
# Masks are 5-bit $4015 length-counter vectors: bit 0 pulse1, 1 pulse2,
# 2 triangle, 3 noise, 4 DMC. The particular bit patterns below are arbitrary
# test fixtures, NOT any game's music -- the signal never looks at which bits
# moved, only at how much of this run's own activity mass moved.
# ==========================================================================

MUSIC = 0b00111        # three channels busy
OTHER = 0b11000        # the other two busy -- all five bits differ from MUSIC
SILENT = 0b00000

# Observations needed before the signal has a null at all.
WARMUP = APU_SHORT_WINDOW + APU_MIN_BASELINE


def _feed(sig: ApuActivitySignal, masks) -> ApuActivitySignal:
    for m in masks:
        sig.push(m)
    return sig


def _steady(n: int, mask: int = MUSIC) -> list[int]:
    return [mask] * n


def _transient(n: int, at: int, length: int, mask: int = OTHER) -> list[int]:
    """`length` observations of a completely different activity vector, then
    straight back to the baseline one."""
    out = _steady(n)
    for i in range(at, min(at + length, n)):
        out[i] = mask
    return out


# -- the null has to be earned before there is a vote ----------------------

def test_the_vote_does_not_exist_until_the_null_is_measured() -> None:
    sig = _feed(ApuActivitySignal(), _steady(WARMUP - 1))
    assert sig.n_evals == 0 and sig.vote() == 0 and sig.trigger_n is None
    sig.push(MUSIC)
    assert sig.n_evals == 1          # first observation with a full null


def test_a_missing_mask_is_ignored_rather_than_read_as_silence() -> None:
    # A caller that has not plumbed the modality yet passes None. Counting
    # that as "every channel went quiet" would be the loudest possible change
    # signal fabricated out of an absent measurement.
    sig = _feed(ApuActivitySignal(), _steady(WARMUP + 60))
    n_before, evals_before = sig.n, sig.n_evals
    for _ in range(200):
        sig.push(None)
    assert (sig.n, sig.n_evals) == (n_before, evals_before)
    assert sig.trigger_n is None


def test_only_the_low_five_bits_of_the_mask_are_read() -> None:
    # $4015 bits 5-7 are open bus / frame-IRQ status, not channel activity.
    a = _feed(ApuActivitySignal(), _steady(WARMUP + 40, MUSIC))
    b = _feed(ApuActivitySignal(), _steady(WARMUP + 40, MUSIC | 0b11100000))
    assert (a.last_dev, a.last_gate) == (b.last_dev, b.last_gate)


# -- sustained change fires; transients structurally cannot ----------------

def test_a_sustained_coordinated_change_fires() -> None:
    sig = _feed(ApuActivitySignal(), _steady(200))
    assert sig.trigger_n is None, "steady play must not fire"
    _feed(sig, _steady(40, OTHER))
    assert sig.trigger_n is not None


def test_the_signal_is_direction_agnostic() -> None:
    # Channels going quiet and channels starting up are the same event here.
    # Anything else would be a content prior about what a win sounds like.
    start = _feed(ApuActivitySignal(), _steady(200, SILENT) + _steady(40, MUSIC))
    stop = _feed(ApuActivitySignal(), _steady(200, MUSIC) + _steady(40, SILENT))
    assert start.trigger_n is not None and stop.trigger_n is not None


@pytest.mark.parametrize("at", [95, 100, 111, 137, 200, 349])
def test_a_one_observation_blip_never_fires_at_any_alignment(at: int) -> None:
    # THE requirement: a single-observation flip of EVERY channel -- a drum
    # hit, a jump SFX, one frame of a length counter reloading -- cannot fire
    # this signal, wherever in the window it lands. The coord signal's
    # alignment lottery (test_confluence_v2: an aligned 2-sample spike votes 8
    # checks in a row) is exactly what this must not reproduce.
    sig = _feed(ApuActivitySignal(), _transient(400, at, 1))
    assert sig.trigger_n is None


@pytest.mark.parametrize("length", [1, 2, 3])
def test_no_transient_shorter_than_the_windows_resolution_can_fire(
        length: int) -> None:
    # dev <= length/short_window for ANY transient, so firing needs
    # length > gate_floor * short_window = 3 at the defaults. Pinned as a
    # bound, not as a lucky alignment.
    assert length <= APU_GATE_FLOOR * APU_SHORT_WINDOW
    for at in (95, 137, 211):
        sig = _feed(ApuActivitySignal(), _transient(400, at, length))
        assert sig.trigger_n is None, (length, at)


def test_the_bound_is_tight_so_the_signal_is_not_simply_deaf() -> None:
    # One observation past the bound the same shape DOES fire: the floor is a
    # resolution limit, not a mute.
    sig = _feed(ApuActivitySignal(), _transient(400, 137, 4))
    assert sig.trigger_n is not None


def test_a_lone_channel_toggling_needs_most_of_the_window_to_count() -> None:
    # Coordination falls out of the L1 sum: one channel changing contributes
    # at most 1/N_APU_BITS = 0.2, so it must hold for more than half the short
    # window to clear the default floor, while five channels moving together
    # need a fifth of that. No bit is privileged -- this is arithmetic, not a
    # rule about melodies.
    one_bit = _feed(ApuActivitySignal(),
                    _steady(200) + _steady(10, MUSIC ^ 0b00001))
    all_bits = _feed(ApuActivitySignal(), _steady(200) + _steady(10, OTHER))
    assert one_bit.trigger_n is None
    assert all_bits.trigger_n is not None


# -- self-calibration: the null is this run's own history ------------------

def test_the_gate_is_the_floor_for_a_game_whose_channels_are_steady() -> None:
    sig = _feed(ApuActivitySignal(), _steady(WARMUP + 40))
    assert sig.last_gate == pytest.approx(APU_GATE_FLOOR)


def test_the_gate_rises_with_the_games_own_measured_churn() -> None:
    # THE self-calibration property: identical code, no per-game constant, and
    # a game whose channels flicker constantly is held to a strictly higher
    # bar than one whose channels do not -- because the bar is computed from
    # the sampling noise of the game's OWN measured per-bit rates.
    steady = _feed(ApuActivitySignal(), _steady(400))
    rng = np.random.default_rng(0)
    churn = _feed(ApuActivitySignal(),
                  [int(rng.integers(32)) for _ in range(400)])
    assert churn.last_gate > steady.last_gate * 2


def test_a_run_of_pure_churn_never_fires_on_its_own_null() -> None:
    # The null must not detect itself. 600 observations of maximal
    # channel churn is the worst case for a fixed threshold.
    rng = np.random.default_rng(7)
    sig = _feed(ApuActivitySignal(),
                [int(rng.integers(32)) for _ in range(600)])
    assert sig.trigger_n is None


def test_a_churning_game_still_detects_its_channels_going_quiet() -> None:
    # The mirror of the test above: raising the bar for a noisy game must not
    # cost it the real event.
    rng = np.random.default_rng(3)
    sig = _feed(ApuActivitySignal(),
                [int(rng.integers(32)) for _ in range(400)])
    assert sig.trigger_n is None
    _feed(sig, _steady(60, SILENT))
    assert sig.trigger_n is not None


def test_the_null_is_ROLLING_so_a_new_regime_becomes_the_new_normal() -> None:
    # The baseline is the last `baseline_window` observations, not the run's
    # opening. Once a changed regime has persisted, the deviation collapses
    # back toward zero -- the signal reports CHANGE, never a preferred state.
    sig = _feed(ApuActivitySignal(), _steady(400))
    _feed(sig, _steady(40, OTHER))
    peak = sig.last_dev
    _feed(sig, _steady(sig.baseline_window + APU_SHORT_WINDOW, OTHER))
    assert peak > 0.5 and sig.last_dev < 0.01


# -- latch / hold / reset --------------------------------------------------

def test_the_vote_is_held_for_a_bounded_run_of_observations() -> None:
    sig = ApuActivitySignal(hold=10)
    _feed(sig, _steady(200))
    for _ in range(60):
        sig.push(OTHER)
        if sig.trigger_n is not None:
            break
    assert sig.trigger_n is not None and sig.vote() == 1
    _feed(sig, _steady(9, OTHER))
    assert sig.vote() == 1
    sig.push(OTHER)
    assert sig.vote() == 0          # hold expired; the fire does not persist


def test_the_signal_re_arms_after_it_has_returned_to_its_own_null() -> None:
    # MEASURED on real play (Contra from power-on): the first sustained
    # coordinated change after the null is measured is the title music
    # starting, at observation 93. A one-shot latch would spend the signal's
    # only vote there and be blind for the rest of the burst.
    sig = ApuActivitySignal(hold=20)
    _feed(sig, _steady(200))
    _feed(sig, _steady(40, OTHER))
    assert sig.n_triggers == 1
    # Let the new regime become the normal one: the deviation decays back
    # under this game's own gate and the vote drops.
    _feed(sig, _steady(sig.baseline_window + APU_SHORT_WINDOW, OTHER))
    assert sig.vote() == 0 and sig.n_triggers == 1
    for _ in range(40):
        sig.push(MUSIC)
        if sig.n_triggers == 2:
            break
    assert sig.n_triggers == 2 and sig.vote() == 1


def test_one_long_change_is_not_re_declared_every_hold_window() -> None:
    # Re-arming must require the return to null, not merely the hold
    # expiring, or a single sustained change would announce itself over and
    # over for as long as it lasted.
    sig = ApuActivitySignal(hold=5)
    _feed(sig, _steady(200))
    _feed(sig, _steady(150, OTHER))
    assert sig.n_triggers == 1


def test_reset_drops_both_the_latch_and_the_calibration() -> None:
    # Same rule StreamingConfluenceDetector.reset() follows: a vetoed fire
    # must not be able to land again off the very samples that produced it.
    sig = _feed(ApuActivitySignal(), _steady(200) + _steady(40, OTHER))
    assert sig.vote() == 1
    sig.reset()
    assert sig.vote() == 0 and sig.trigger_n is None and sig.n == 0
    _feed(sig, _steady(WARMUP - 1, OTHER))
    assert sig.n_evals == 0         # the null has to be re-earned in full


# ==========================================================================
# Part 2 -- the vote inside StreamingConfluenceDetector
#
# The RAM fixtures mirror test_confluence_v2's synthetic layout: arbitrary
# addresses, real detector.
# ==========================================================================

A_GX_LO, A_GX_HI = 0x10, 0x11
A_LIVES, A_Y, A_ROOM = 0x20, 0x30, 0x40
A_TIMER, A_SCORE = 0x50, 0x51
ENTITY_LO, ENTITY_HI = 0x60, 0x70


def _ram(gx: int, lives: int = 3, room: int = 1, timer: int = 200,
         score: int = 0, entities: bool = True, y: int = 100) -> bytes:
    buf = bytearray(2048)
    buf[A_GX_LO] = gx & 0xFF
    buf[A_GX_HI] = (gx >> 8) & 0xFF
    buf[A_LIVES], buf[A_Y], buf[A_ROOM] = lives, y, room
    buf[A_TIMER], buf[A_SCORE] = timer, score
    for a in range(ENTITY_LO, ENTITY_HI):
        buf[a] = 9 if entities else 0
    return bytes(buf)


def _tally(i: int) -> dict:
    return {"timer": 250 - i // 8, "score": i // 8}


def _progress(ram) -> int:
    return int(ram[A_GX_LO]) | int(ram[A_GX_HI]) << 8


def _load_stream(n: int = 400, at: int = 200) -> list[bytes]:
    """A position reset toward a level start plus an entity wipe: the RAM
    shape both remaining v2 signals vote on."""
    return [_ram(gx=(600 + 2 * i if i < at else 20 + 2 * (i - at)),
                 entities=i < at, **_tally(i))
            for i in range(n)]


def _fires(det: StreamingConfluenceDetector, stream, masks=None) -> int | None:
    for i, ram in enumerate(stream):
        hit = det.push(ram) if masks is None else det.push(ram, masks[i])
        if hit:
            return i
    return None


def test_the_v3_default_builds_no_apu_signal_at_all() -> None:
    d = StreamingConfluenceDetector(_progress)
    assert d.apu_weight == 0.0 and d._apu is None


def test_the_default_path_is_byte_identical_with_or_without_the_knob() -> None:
    stream = _load_stream()
    assert (_fires(StreamingConfluenceDetector(_progress), stream)
            == _fires(StreamingConfluenceDetector(_progress, apu_weight=0),
                      stream))


def test_a_mask_passed_to_a_disarmed_detector_is_ignored() -> None:
    # A caller may plumb the modality unconditionally; a disarmed detector
    # must neither read it nor change behavior because of it.
    stream = _load_stream()
    plain = _fires(StreamingConfluenceDetector(_progress), stream)
    withm = _fires(StreamingConfluenceDetector(_progress), stream,
                   [OTHER] * len(stream))
    assert plain == withm is not None


def _masks_flat(n: int) -> list[int]:
    return [MUSIC] * n


def _masks_change(n: int, at: int = 200) -> list[int]:
    return [MUSIC if i < at else SILENT for i in range(n)]


def test_raising_the_bar_makes_the_audio_vote_REQUIRED() -> None:
    # min_signals 3 with apu_weight 1.0 = all three signals must agree. The
    # RAM signature alone (tally + coord, the two votes that carried every
    # recorded false positive) then no longer suffices.
    stream = _load_stream()
    strict = StreamingConfluenceDetector(_progress, min_signals=3,
                                         apu_weight=1.0)
    assert _fires(strict, stream, _masks_flat(len(stream))) is None
    assert strict.n_checks > 0, "the detector must actually have run"

    heard = StreamingConfluenceDetector(_progress, min_signals=3,
                                        apu_weight=1.0)
    assert _fires(heard, stream, _masks_change(len(stream))) is not None
    assert heard.n_apu_votes > 0


def test_the_audio_vote_alone_cannot_carry_a_clear() -> None:
    # No coord signature anywhere: ordinary forward play with the music
    # changing under it. tally(1) + apu(1) = 2 < 3.
    stream = [_ram(gx=600 + 2 * i, **_tally(i)) for i in range(400)]
    det = StreamingConfluenceDetector(_progress, min_signals=3,
                                      apu_weight=1.0)
    assert _fires(det, stream, _masks_change(len(stream))) is None


def test_a_fractional_weight_is_honored_as_a_partial_vote() -> None:
    stream = _load_stream()
    # 1 + 1 + 0.5 = 2.5 clears a 2.5 bar; the same stream with a flat
    # activity vector scores 2.0 and does not.
    for weight, bar, want in ((0.5, 2.5, True), (0.25, 2.5, False)):
        det = StreamingConfluenceDetector(_progress, min_signals=2.5,
                                          apu_weight=weight)
        got = _fires(det, stream, _masks_change(len(stream))) is not None
        assert got is want, (weight, bar)


def test_the_detectors_reset_also_resets_the_audio_signal() -> None:
    stream = _load_stream()
    det = StreamingConfluenceDetector(_progress, min_signals=3,
                                      apu_weight=1.0)
    fired = _fires(det, stream, _masks_change(len(stream)))
    assert fired is not None
    det.reset()
    assert det._apu.n == 0 and det._apu.trigger_n is None
    # Re-earning the null takes the full warm-up, so the same stale evidence
    # cannot immediately re-fire.
    assert det.push(stream[fired], SILENT) is False


def test_apu_params_reach_the_signal() -> None:
    det = StreamingConfluenceDetector(_progress, apu_weight=1.0,
                                      apu_params={"short_window": 7,
                                                  "gate_floor": 0.3,
                                                  "hold": 5})
    assert (det._apu.short_window, det._apu.gate_floor, det._apu.hold) == (
        7, 0.3, 5)


# ==========================================================================
# Part 3 -- profile wiring (GenericGame) and the sampling resolver
# ==========================================================================

def _game(**clear) -> GenericGame:
    return GenericGame({
        "solve": {
            "rom": "roms/does-not-need-to-exist.nes",
            "progress": {"lo": A_GX_LO, "hi": A_GX_HI},
            "y": A_Y, "area": A_ROOM, "level_key": [], "lives": A_LIVES,
            "clear": dict({"mode": "confluence"}, **clear),
        }
    })


def test_an_unconfigured_profile_asks_for_no_audio_modality() -> None:
    g = _game()
    assert g._conf_apu_weight == 0.0 and g._conf_apu == {}
    assert g.needs_apu() is False


def test_a_profile_that_opts_in_asks_for_it() -> None:
    g = _game(apu_weight=1.0, min_signals=3, apu={"sustain": 9})
    assert g.needs_apu() is True and g._conf_apu == {"sustain": 9}


def test_a_non_confluence_profile_never_asks_for_the_modality() -> None:
    # apu_weight is meaningless outside the confluence hook; a profile that
    # sets it anyway must not make the solver pay for a per-step pool call.
    g = _game(mode="byte_change", addr=0x75, apu_weight=1.0)
    assert g.needs_apu() is False


def test_smb_defines_needs_apu_and_says_no() -> None:
    # Same reasoning as clear_verify_margin: the callers ask every adapter,
    # and SmbGame does not inherit from GenericGame.
    assert SmbGame.needs_apu(object()) is False


def test_every_game_adapter_defines_needs_apu() -> None:
    import scripts.go_explore_solve as ges
    adapters = [v for v in vars(ges).values()
                if isinstance(v, type) and v.__name__.endswith("Game")
                and v.__module__ == ges.__name__]
    assert len(adapters) >= 2, adapters
    for cls in adapters:
        assert callable(getattr(cls, "needs_apu", None)), cls.__name__


def test_the_shipped_confluence_profiles_stay_on_the_default_path() -> None:
    import pathlib

    import yaml
    repo = pathlib.Path(__file__).resolve().parent.parent
    for name in ("contra.yaml", "gradius.yaml"):
        cl = ((yaml.safe_load((repo / "configs" / name).read_text())
               .get("solve") or {}).get("clear") or {})
        assert cl.get("mode") == "confluence", name
        for knob in ("apu_weight", "apu"):
            assert knob not in cl, f"{name} silently opted into {knob}"


class _PoolWithApu:
    def apu_activity_all(self) -> bytes:
        return bytes([MUSIC])


class _PoolWithoutApu:
    pass


def test_sampling_is_off_unless_the_adapter_asks() -> None:
    assert resolve_apu_sampling(_game(), _PoolWithApu()) is False


def test_sampling_is_on_when_adapter_and_core_agree() -> None:
    assert resolve_apu_sampling(_game(apu_weight=1.0), _PoolWithApu()) is True


def test_an_older_core_degrades_loudly_instead_of_raising(capsys) -> None:
    assert resolve_apu_sampling(_game(apu_weight=1.0),
                                _PoolWithoutApu()) is False
    assert "apu_activity_all" in capsys.readouterr().out


def test_an_adapter_without_the_hook_at_all_is_simply_off() -> None:
    assert resolve_apu_sampling(object(), _PoolWithApu()) is False


# ==========================================================================
# Part 4 -- the counterfactual perturbation probe (stubbed pool)
# ==========================================================================

class _CfPool:
    """Duck-typed nes_core.Pool whose RAM is a pure function of the actions
    applied SINCE the last load_worker_state.

    That is exactly the shape the probe needs to be tested against: every
    branch reloads the same snapshot, so `script(actions_since_load)` decides
    whether that branch's inputs still reach the clear.
    """

    instances: list = []
    script = staticmethod(lambda acts: 0)

    def __init__(self, rom_path=None, num_workers=1, frame_skip=4) -> None:
        self.rom_path, self.num_workers, self.frame_skip = (
            rom_path, num_workers, frame_skip)
        self.loaded: list = []
        self.branches: list = []      # action list of every completed segment
        self.since_load: list = []
        self.saves = 0
        self.total_steps = 0
        self.shutdown_calls = 0
        self.headless = None
        _CfPool.instances.append(self)

    def set_headless(self, v) -> None:
        self.headless = v

    def set_skip_preprocess(self, v) -> None:
        self.skip_pre = v

    def reset_all(self) -> None:
        pass

    def _close_segment(self) -> None:
        if self.since_load:
            self.branches.append(list(self.since_load))
        self.since_load = []

    def load_worker_state(self, wid, blob) -> None:
        self._close_segment()
        self.loaded.append(blob)

    def save_worker_state(self, wid) -> bytes:
        self.saves += 1
        return b"pivot"

    def step_all(self, acts):
        self.since_load.append(int(acts[0]))
        self.total_steps += 1
        code = _CfPool.script(self.since_load)
        return [(None, None, bytes([code]) + bytes(2047))]

    def shutdown(self) -> None:
        self._close_segment()
        self.shutdown_calls += 1


class _CfGame:
    """Clear/dead keyed off the scripted RAM byte; 1 = clear, 2 = dead."""

    rom = "roms/does-not-need-to-exist.nes"

    def __init__(self, margin: int = 0, budget: int = 0) -> None:
        self._margin = margin
        self._budget = budget

    def clear_verify_margin(self) -> int:
        return self._margin

    def clear_observation_budget(self) -> int:
        return self._budget

    def needs_apu(self) -> bool:
        return False

    def is_dead(self, ram, start_lives) -> bool:
        return ram[0] == 2

    def is_finale(self, start_key, ram) -> bool:
        return ram[0] == 3

    def is_clear(self, start_key, ram, ctx=None) -> bool:
        return ram[0] == 1

    def level_key(self, ram) -> tuple:
        return (0,)


@pytest.fixture()
def cf_root(monkeypatch, tmp_path):
    import scripts.go_explore_solve as ges

    _CfPool.instances = []
    monkeypatch.setattr(ges, "Pool", _CfPool)
    root = tmp_path / "entrance.state"
    root.write_bytes(b"rootblob")
    return root


def _cf_solver(root, **kw) -> SimpleNamespace:
    fake = SimpleNamespace(
        game=_CfGame(), bitmasks=list(range(6)),   # action index == bitmask
        frame_skip=4, hw_flags=[], start_wd=(0,), start_lives=3,
        roots={"entrance": {"path": str(root)}},
        cf_branches=8, cf_pre_steps=4, cf_perturb_p=0.5, cf_agree=0.5,
        cf_seed=0, _needs_apu=False)
    for k, v in kw.items():
        setattr(fake, k, v)
    return fake


TRACE = [1] * 12          # 8 head actions + a 4-action perturbed tail


def _commits(acts) -> int:
    """The game committed: three steps past the snapshot it clears, whatever
    the pad does. This is the level-load / flag-slide / tally shape."""
    return 1 if len(acts) >= 3 else 0


def _contingent(acts) -> int:
    """The 'clear' hangs off the exact inputs -- change one and it evaporates.
    The combat-blip shape."""
    return 1 if acts == [1, 1, 1, 1] else 0


def _never(acts) -> int:
    return 0


def test_a_committed_transition_reproduces_across_the_branches(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["ok"] is True and out["verdict"] == "commits"
    assert out["agreement"] == 1.0 and out["agree"] == out["k"] == 8
    assert out["control"]["verdict"] == "clear"


def test_a_contingent_clear_is_refused(cf_root) -> None:
    _CfPool.script = staticmethod(_contingent)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["ok"] is False and out["verdict"] == "contingent"
    assert out["control"]["verdict"] == "clear", (
        "the control must still clear, or this measures the harness")
    assert out["agreement"] < out["threshold"]


def test_the_branches_really_do_perturb_the_inputs(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert sum(b["perturbed"] for b in out["branches"]) > 0
    # ...and the control is the trace verbatim.
    assert out["control"]["perturbed"] == 0


def test_perturb_p_zero_makes_every_branch_the_control(cf_root) -> None:
    # Pins what the knob MEANS: at p=0 even the contingent shape reproduces
    # everywhere, because nothing was perturbed.
    _CfPool.script = staticmethod(_contingent)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root, cf_perturb_p=0.0), "entrance", TRACE)
    assert out["ok"] is True and out["agreement"] == 1.0
    assert all(b["perturbed"] == 0 for b in out["branches"])


def test_the_probe_is_reproducible_from_its_recorded_seed(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    a = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    branches_a = list(_CfPool.instances[-1].branches)
    b = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert branches_a == _CfPool.instances[-1].branches
    assert a["branches"] == b["branches"] and a["seed"] == b["seed"] == 0


def test_a_different_seed_probes_differently(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    first = list(_CfPool.instances[-1].branches)
    Solver.counterfactual_probe(_cf_solver(cf_root, cf_seed=99), "entrance",
                                TRACE)
    assert first != _CfPool.instances[-1].branches


def test_every_branch_starts_from_the_same_pre_clear_snapshot(
        cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    pool = _CfPool.instances[-1]
    assert pool.saves == 1
    # root blob once, then the pivot for the control and each of K branches.
    assert pool.loaded == [b"rootblob"] + [b"pivot"] * 9
    # The head is replayed exactly once: rooting NOOP + 8 head actions.
    assert pool.branches[0] == [0] + [1] * 8
    assert all(len(seg) <= 4 for seg in pool.branches[1:])
    assert pool.headless is True and pool.shutdown_calls == 1


def test_the_probe_is_bounded_by_pre_steps_not_by_the_trace_length(
        cf_root) -> None:
    # Cost control: a 1,700-action candidate must not cost 9 full replays.
    _CfPool.script = staticmethod(_commits)
    long_trace = [1] * 500
    Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", long_trace)
    steps = _CfPool.instances[-1].total_steps
    assert steps == 1 + (500 - 4) + 9 * 3   # NOOP + head + 9 branches x 3


def test_pre_steps_larger_than_the_trace_clamps(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root, cf_pre_steps=999), "entrance", [1, 1, 1, 1, 1])
    assert out["pre_steps"] == 5
    assert _CfPool.instances[-1].branches[0] == [0]     # head is empty


def test_a_branch_that_dies_counts_as_disagreement_not_an_error(
        cf_root) -> None:
    def script(acts):
        if acts == [1, 1, 1, 1]:
            return 1
        return 2 if acts != [1] * len(acts) else 0   # deviated => died

    _CfPool.script = staticmethod(script)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["verdict"] == "contingent" and "error" not in out
    assert any(b["verdict"] == "dead" for b in out["branches"])


def test_a_finale_counts_as_a_reproduced_clear(cf_root) -> None:
    _CfPool.script = staticmethod(lambda acts: 3 if len(acts) >= 3 else 0)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["ok"] is True and out["agreement"] == 1.0


def test_the_branches_get_the_same_noop_margin_replay_verification_uses(
        cf_root) -> None:
    # A windowed hook's verdict lands a stride late; without the margin the
    # branches would all miss it and a real clear would be refused.
    _CfPool.script = staticmethod(lambda acts: 1 if len(acts) >= 7 else 0)
    fake = _cf_solver(cf_root)
    fake.game = _CfGame(margin=5)
    out = Solver.counterfactual_probe(fake, "entrance", TRACE)
    assert out["ok"] is True
    assert all(b["at"] > out["pre_steps"] for b in out["branches"])


# -- the control branch is a guard, not a vote -----------------------------

def _starved_probe(cf_root):
    """A probe whose hook is starved of its warm-up: still inconclusive, still
    refuses nothing (the reason that is a hole in OUR harness)."""
    _CfPool.script = staticmethod(_never)
    return Solver.counterfactual_probe(
        _cf_solver(cf_root, game=_CfGame(margin=0, budget=100)),
        "entrance", [1] * 30)


def test_a_probe_that_could_not_look_reports_inconclusive(cf_root) -> None:
    # The probe could not judge -- refusing here would be reporting a verdict
    # about our own instrument as if it were a verdict about the candidate.
    out = _starved_probe(cf_root)
    assert out["verdict"] == "inconclusive" and out["ok"] is True
    assert out["reason"] == "warmup_short" and out["agree"] == 0


def test_inconclusive_does_not_block_banking(cf_root, tmp_path) -> None:
    fake = _bank_solver(tmp_path, verdict_ok=True, cf_gate=True)
    fake.counterfactual_probe = lambda root_id, trace: _starved_probe(cf_root)
    assert Solver._dump_solution(fake, "entrance", [1] * 30,
                                 bytearray(2048), 30) is True
    rec = json.loads((tmp_path / "solutions" / "sol_000.json").read_text())
    assert rec["counterfactual_gate"]["verdict"] == "inconclusive"


# -- the deep retry: a blind control is not a free pass --------------------
#
# Receipt for the escape this closes:
# runs/detector_gate_20260810/double_dragon_cf.log -- an 88-action Double
# Dragon candidate whose control saw nothing, whose eight branches saw
# nothing, and which was banked anyway on verdict inconclusive/control_blind.

def _root_state_artifact(acts) -> int:
    """The Double Dragon shape: the 'clear' is not a game event at all, it is
    the clear hook's own rolling state reaching a coincidence after it has
    been fed this candidate's whole observation history FROM THE ROOT.

    Only a replay that starts at the root ever sees it -- which is exactly
    what replay verification does, so that check passes it by construction.
    No snapshot inside the trajectory can reproduce it.
    """
    pool = _CfPool.instances[-1]
    return 1 if pool.loaded[-1] == b"rootblob" and len(acts) >= 12 else 0


def _commit_before_the_snapshot(acts) -> int:
    """The Kirby shape: a real transition that latched EARLIER than the first
    snapshot. It needs ten observations of the transition running itself out,
    so a four-action branch cannot see it however faithful the inputs -- but
    it is committed, so any branch long enough to replay through the commit
    point reaches it whatever the pad does."""
    return 1 if len(acts) >= 10 else 0


def test_replay_verification_passes_the_artifact_this_probe_refuses(
        cf_root) -> None:
    # The two checks side by side on the SAME candidate. If they agreed there
    # would be nothing here to add.
    _CfPool.script = staticmethod(_root_state_artifact)
    fake = _cf_solver(cf_root)
    assert Solver.replay_verify(fake, "entrance", TRACE)["ok"] is True
    out = Solver.counterfactual_probe(fake, "entrance", TRACE)
    assert out["ok"] is False and out["verdict"] == "state_artifact"


def test_a_state_artifact_is_refused_rather_than_called_inconclusive(
        cf_root) -> None:
    _CfPool.script = staticmethod(_root_state_artifact)
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["ok"] is False and out["verdict"] == "state_artifact"
    assert out["control"]["verdict"] == "no_clear" and out["agree"] == 0
    # ...and the receipt says which snapshot was tried, and why the deeper one
    # could not be believed.
    assert out["retry"] == {"pre_steps_requested": 16, "pre_steps": 12,
                            "ran": False, "reason": "root_pivot"}
    assert out["reason"] == "root_pivot"


def test_the_deep_retry_rescues_a_commit_that_predates_the_snapshot(
        cf_root) -> None:
    # THE NUANCE. A real clear can be invisible from a shallow snapshot simply
    # because it latched before it. Re-snapshot four times deeper and the
    # control replays through the commit point -- Kirby-class candidates must
    # survive this gate, or it is a blanket mute with extra steps.
    _CfPool.script = staticmethod(_commit_before_the_snapshot)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root), "entrance", [1] * 100)
    assert out["ok"] is True and out["verdict"] == "commits"
    assert out["retry"] == {"pre_steps_requested": 16, "pre_steps": 16,
                            "ran": True}
    assert out["pre_steps"] == 16 and out["control"]["verdict"] == "clear"
    # The superseded shallow pass is kept, blindness and all.
    assert out["first_pass"]["pre_steps"] == 4
    assert out["first_pass"]["control"]["verdict"] == "no_clear"
    assert "reason" not in out


def test_the_deep_retry_re_snapshots_rather_than_walking_forward(
        cf_root) -> None:
    # It has to go BACKWARD from the first snapshot, which no amount of
    # stepping can do: the retry must re-root and replay a shorter head.
    _CfPool.script = staticmethod(_commit_before_the_snapshot)
    Solver.counterfactual_probe(_cf_solver(cf_root, cf_branches=2),
                                "entrance", [1] * 100)
    pool = _CfPool.instances[-1]
    assert pool.loaded == ([b"rootblob"] + [b"pivot"] * 3     # first pass
                           + [b"rootblob"] + [b"pivot"] * 3)  # deep retry
    assert pool.saves == 2
    assert _segments(pool) == [1 + 96, 4, 4, 4,      # head, control, 2 branches
                               1 + 84, 10, 10, 10]   # deeper head, then 10s


def test_a_still_blind_deeper_control_is_a_state_artifact(cf_root) -> None:
    # The retry ran, looked from four times further back, and still saw
    # nothing: that is the refusal, and the receipt says which of the three
    # ways the retry failed to exonerate the candidate this was.
    _CfPool.script = staticmethod(_never)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root), "entrance", [1] * 100)
    assert out["ok"] is False and out["verdict"] == "state_artifact"
    assert out["reason"] == "control_blind_at_depth"
    assert out["retry"]["ran"] is True and out["pre_steps"] == 16


def test_the_retry_is_capped_at_the_trace_length(cf_root) -> None:
    # 4 x 32 pre-steps against an 88-action candidate (the Double Dragon
    # arithmetic): the deeper snapshot would BE the root, where the control is
    # just replay verification -- which already passed by construction on
    # exactly this class. Capped, not run, not believed.
    _CfPool.script = staticmethod(_root_state_artifact)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root, cf_pre_steps=32), "entrance", [1] * 88)
    assert out["retry"]["pre_steps_requested"] == 128
    assert out["retry"]["pre_steps"] == 88 == out["n_actions"]
    assert out["retry"]["ran"] is False
    assert out["verdict"] == "state_artifact" and out["ok"] is False
    # One pass' worth of emulator, not two: the retry that cannot help is not
    # paid for either.
    assert _CfPool.instances[-1].saves == 1


def test_the_retry_is_skipped_when_the_floor_already_pinned_the_snapshot(
        cf_root) -> None:
    # The warm-up floor can already sit deeper than 4x the request; re-running
    # the identical pass would re-measure the same blind control.
    _CfPool.script = staticmethod(_never)
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root, game=_CfGame(margin=0, budget=40)),
        "entrance", [1] * 100)
    assert out["pre_steps"] == 40
    assert out["retry"] == {"pre_steps_requested": 16, "pre_steps": 40,
                            "ran": False, "reason": "no_deeper"}
    assert out["verdict"] == "state_artifact"


def test_the_double_dragon_receipt_shape(cf_root, tmp_path) -> None:
    # End to end on the banking path, in the shape the re-run receipt has to
    # have: replay verification passes, the gate refuses, nothing is banked,
    # and the counters name the failure class.
    _CfPool.script = staticmethod(_root_state_artifact)
    fake = _bank_solver(tmp_path, verdict_ok=True, cf_gate=True)
    fake.counterfactual_probe = lambda root_id, trace: (
        Solver.counterfactual_probe(_cf_solver(cf_root), root_id, trace))
    assert Solver._dump_solution(fake, "entrance", TRACE,
                                 bytearray(2048), 12) is False
    assert not list((tmp_path / "solutions").glob("sol_*"))
    assert fake.cf_checks == 1 and fake.cf_rejections == 1
    assert fake.cf_state_artifacts == 1
    assert getattr(fake, "cf_retries", 0) == 0   # the retry could not help


def test_a_state_artifact_rejection_is_logged_as_its_own_failure(
        cf_root, tmp_path, capsys) -> None:
    _CfPool.script = staticmethod(_root_state_artifact)
    fake = _bank_solver(tmp_path, verdict_ok=True, cf_gate=True)
    fake.counterfactual_probe = lambda root_id, trace: (
        Solver.counterfactual_probe(_cf_solver(cf_root), root_id, trace))
    Solver._dump_solution(fake, "entrance", TRACE, bytearray(2048), 12)
    err = capsys.readouterr().err
    # Not "the clear did not survive perturbation" -- nothing about the game
    # happened here at all, and a branch count would misdescribe it.
    assert "STATE ARTIFACT" in err and "root_pivot" in err
    assert "BY CONSTRUCTION" in err


def test_the_deep_retry_is_counted_separately_from_the_rejection(
        cf_root, tmp_path) -> None:
    _CfPool.script = staticmethod(_never)
    fake = _bank_solver(tmp_path, verdict_ok=True, cf_gate=True)
    fake.counterfactual_probe = lambda root_id, trace: (
        Solver.counterfactual_probe(_cf_solver(cf_root), root_id, trace))
    Solver._dump_solution(fake, "entrance", [1] * 100, bytearray(2048), 100)
    assert (fake.cf_retries, fake.cf_state_artifacts) == (1, 1)


def test_the_progress_line_reports_the_new_counters(tmp_path) -> None:
    fake = SimpleNamespace(
        archive=[0] * 7,
        _stall={"last_cells": 0, "last_t": 0.0, "flat_windows": 0},
        max_area=0, max_gx_in_area={}, max_sect=0, n_solutions=0,
        best_sol_len=10 ** 9, steps_done=100, ortho_mode="off",
        door_weight=0.0, transition_macros=[], verify_rejections=0,
        cf_checks=3, cf_rejections=2, cf_state_artifacts=2, cf_retries=1,
        out=tmp_path)
    Solver.progress_line(fake, 10.0)
    line = json.loads(
        (tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert line["cf_checks"] == 3 and line["cf_rejections"] == 2
    assert line["cf_state_artifacts"] == 2 and line["cf_retries"] == 1


# -- failure handling ------------------------------------------------------

def test_the_probe_fails_closed_on_an_infrastructure_error(cf_root) -> None:
    _CfPool.script = staticmethod(_commits)
    fake = _cf_solver(cf_root)
    fake.roots = {}
    out = Solver.counterfactual_probe(fake, "entrance", TRACE)
    assert out["ok"] is False and out["verdict"] == "error"
    assert "KeyError" in out["error"]


def test_the_probe_never_leaks_a_pool_when_the_core_throws(cf_root) -> None:
    class _Exploding(_CfPool):
        def step_all(self, acts):
            raise RuntimeError("core panic")

    import scripts.go_explore_solve as ges
    ges.Pool = _Exploding
    out = Solver.counterfactual_probe(_cf_solver(cf_root), "entrance", TRACE)
    assert out["ok"] is False and "RuntimeError" in out["error"]
    assert _CfPool.instances[-1].shutdown_calls == 1


# ==========================================================================
# Part 5 -- the gate on the banking path
# ==========================================================================

def _bank_solver(tmp_path, verdict_ok: bool = True, verify_bank: bool = True,
                 **kw) -> SimpleNamespace:
    (tmp_path / "solutions").mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        best_sol_len=10 ** 9, sol_counter=0, n_solutions=0, out=tmp_path,
        verify_bank=verify_bank, verify_checks=0, verify_rejections=0,
        args=SimpleNamespace(profile="configs/kirby.yaml", workers=8),
        roots={"entrance": {"path": "does/not/matter.state"}},
        provenance={"hw_flags": [], "frame_skip": 4},
        start_wd=(0,), game=SimpleNamespace(level_key=lambda ram: (1,)),
        cf_checks=0, cf_rejections=0,
    )
    fake.replay_verify = lambda root_id, trace: {
        "ok": verdict_ok, "verdict": "clear" if verdict_ok else "dead",
        "at": 3, "n_actions": len(trace), "elapsed_s": 0.1}
    for k, v in kw.items():
        setattr(fake, k, v)
    return fake


def _stub_probe(ok: bool, verdict: str = "commits"):
    calls: list = []

    def probe(root_id, trace):
        calls.append((root_id, list(trace)))
        return {"ok": ok, "verdict": verdict, "agree": 8 if ok else 1,
                "k": 8, "agreement": 1.0 if ok else 0.125, "threshold": 0.5,
                "pre_steps": 32, "perturb_p": 0.25, "n_actions": len(trace),
                "seed": 0, "control": {"verdict": "clear"}, "branches": [],
                "elapsed_s": 4.2}
    return probe, calls


def test_the_gate_is_off_by_default_and_the_receipt_is_unchanged(
        tmp_path) -> None:
    # Default identity: an existing invocation must produce the receipt it
    # always produced, with no new key and no seconds burned.
    fake = _bank_solver(tmp_path)
    probe, calls = _stub_probe(ok=False)
    fake.counterfactual_probe = probe
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is True
    rec = json.loads((tmp_path / "solutions" / "sol_000.json").read_text())
    assert "counterfactual_gate" not in rec
    assert calls == [] and fake.cf_checks == 0


def test_a_solver_predating_the_flag_does_not_run_the_gate(tmp_path) -> None:
    # getattr-guarded: an in-process construction (the live show) that has
    # never heard of cf_gate must not crash and must not be gated.
    fake = _bank_solver(tmp_path)
    assert not hasattr(fake, "cf_gate")
    fake.counterfactual_probe = lambda *a, **k: pytest.fail("gate ran")
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is True


def test_a_committed_candidate_is_banked_with_its_probe_receipt(
        tmp_path) -> None:
    fake = _bank_solver(tmp_path, cf_gate=True)
    probe, calls = _stub_probe(ok=True)
    fake.counterfactual_probe = probe
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is True
    rec = json.loads((tmp_path / "solutions" / "sol_000.json").read_text())
    assert rec["counterfactual_gate"]["verdict"] == "commits"
    assert rec["replay_verified"] is True
    assert rec["provenance"] == "search"          # honest-origin marker intact
    assert (fake.cf_checks, fake.cf_rejections) == (1, 0)
    assert calls == [("entrance", [1, 2, 3])]


def test_a_contingent_candidate_writes_nothing_at_all(tmp_path) -> None:
    fake = _bank_solver(tmp_path, cf_gate=True)
    fake.counterfactual_probe = _stub_probe(ok=False, verdict="contingent")[0]
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is False
    assert list((tmp_path / "solutions").iterdir()) == []
    # No banking counter may move: a refused candidate must not consume a
    # sol_NNN slot or lower the bar for later, real clears.
    assert (fake.n_solutions, fake.sol_counter) == (0, 0)
    assert fake.best_sol_len == 10 ** 9
    assert (fake.cf_checks, fake.cf_rejections) == (1, 1)


def test_a_counterfactual_rejection_is_logged_loudly(tmp_path, capsys) -> None:
    fake = _bank_solver(tmp_path, cf_gate=True)
    fake.counterfactual_probe = _stub_probe(ok=False, verdict="contingent")[0]
    Solver._dump_solution(fake, "entrance", [1, 2, 3], bytearray(2048), 3)
    err = capsys.readouterr().err
    assert "CLEAR REJECTED (counterfactual gate)" in err
    assert "NOTHING banked" in err


def test_the_gate_never_runs_on_a_candidate_replay_already_rejected(
        tmp_path) -> None:
    # Ordering: the cheap deterministic check first, and no seconds spent
    # perturbing something that did not even reproduce.
    fake = _bank_solver(tmp_path, verdict_ok=False, cf_gate=True)
    probe, calls = _stub_probe(ok=True)
    fake.counterfactual_probe = probe
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is False
    assert calls == [] and fake.cf_checks == 0


def test_a_non_shorter_re_clear_pays_for_neither_check(tmp_path) -> None:
    fake = _bank_solver(tmp_path, cf_gate=True)
    fake.best_sol_len = 5
    probe, calls = _stub_probe(ok=True)
    fake.counterfactual_probe = probe
    assert Solver._dump_solution(fake, "entrance", [1] * 10,
                                 bytearray(2048), 10) is True
    assert calls == [] and list((tmp_path / "solutions").iterdir()) == []


def test_the_gate_still_runs_when_replay_verification_is_disabled(
        tmp_path) -> None:
    # The two checks are independent instruments; --no-verify-bank must not
    # silently disable the one the operator explicitly asked for.
    fake = _bank_solver(tmp_path, verify_bank=False, cf_gate=True)
    fake.counterfactual_probe = _stub_probe(ok=False, verdict="contingent")[0]
    assert Solver._dump_solution(fake, "entrance", [1, 2, 3],
                                 bytearray(2048), 3) is False
    assert fake.cf_rejections == 1


def test_the_flag_resolves_off_when_absent() -> None:
    assert resolve_counterfactual_gate(SimpleNamespace()) is False
    assert resolve_counterfactual_gate(
        SimpleNamespace(counterfactual_gate=True)) is True


def test_the_cli_exposes_the_gate_and_leaves_it_disarmed() -> None:
    # The flag has to exist and default off; parsed straight out of main()'s
    # own parser so the help text and the default cannot drift apart.
    import inspect
    import re

    import scripts.go_explore_solve as ges
    src = inspect.getsource(ges.main)
    assert '"--counterfactual-gate"' in src
    assert '"--no-counterfactual-gate"' in src
    for flag, default in (("--cf-branches", "8"), ("--cf-pre-steps", "32"),
                          ("--cf-agree", "0.5"), ("--cf-seed", "None")):
        m = re.search(re.escape(f'"{flag}"') + r".*?default=([^,)]+)", src,
                      re.S)
        assert m and m.group(1).strip() == default, flag
    m = re.search(r'"--counterfactual-gate".*?default=([^,)]+)', src, re.S)
    assert m.group(1).strip() == "False"


# ==========================================================================
# Part 6 -- the APU mask reaches the clear hook on the verification path
# ==========================================================================

class _ApuPool(_CfPool):
    masks = [MUSIC]

    def apu_activity_all(self) -> bytes:
        i = min(self.total_steps - 1, len(_ApuPool.masks) - 1)
        return bytes([_ApuPool.masks[max(i, 0)]])


class _MaskRecordingGame(_CfGame):
    def __init__(self) -> None:
        super().__init__(margin=0)
        self.seen: list = []

    def needs_apu(self) -> bool:
        return True

    def is_clear(self, start_key, ram, ctx=None) -> bool:
        self.seen.append(None if ctx is None else ctx.get("_apu_mask"))
        return ram[0] == 1


def test_replay_verification_feeds_the_same_modality_the_live_hook_saw(
        cf_root, monkeypatch) -> None:
    # A live fire that used the audio vote must be judged by a replay that
    # also has the audio vote, or verification measures a different detector
    # than the one that fired.
    import scripts.go_explore_solve as ges
    monkeypatch.setattr(ges, "Pool", _ApuPool)
    _ApuPool.masks = [MUSIC, MUSIC, SILENT, SILENT]
    _CfPool.script = staticmethod(lambda acts: 1 if len(acts) >= 4 else 0)
    game = _MaskRecordingGame()
    fake = _cf_solver(cf_root)
    fake.game, fake._needs_apu = game, True
    out = Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert out["ok"] is True
    assert game.seen and all(m is not None for m in game.seen)
    assert SILENT in game.seen


def test_the_mask_is_not_sampled_when_the_adapter_did_not_ask(
        cf_root, monkeypatch) -> None:
    import scripts.go_explore_solve as ges
    monkeypatch.setattr(ges, "Pool", _ApuPool)
    _CfPool.script = staticmethod(lambda acts: 1 if len(acts) >= 4 else 0)
    game = _MaskRecordingGame()
    fake = _cf_solver(cf_root)
    fake.game = game               # _needs_apu stays False
    Solver.replay_verify(fake, "entrance", [1, 1, 1])
    assert game.seen and all(m is None for m in game.seen)


def test_the_counterfactual_branches_feed_the_modality_too(
        cf_root, monkeypatch) -> None:
    import scripts.go_explore_solve as ges
    monkeypatch.setattr(ges, "Pool", _ApuPool)
    _ApuPool.masks = [SILENT]
    _CfPool.script = staticmethod(_commits)
    game = _MaskRecordingGame()
    fake = _cf_solver(cf_root)
    fake.game, fake._needs_apu = game, True
    out = Solver.counterfactual_probe(fake, "entrance", TRACE)
    assert out["ok"] is True
    assert game.seen and all(m == SILENT for m in game.seen)


# ==========================================================================
# Part 7 -- the WARM-UP BUDGET (repair, 2026-08-10)
#
# The defect this part exists for: a bounded replay into a FRESH ctx was
# feeding a windowed clear hook fewer observations than the hook needs before
# it can fire AT ALL, and reading the resulting silence as a verdict. Measured
# on the shipped counterfactual gate: cf_pre_steps 32 + clear_verify_margin 20
# = 52 observations against the 100 the `min_signals: 3, apu_weight: 1.0`
# configuration the detector's own docs recommend needs -- so the control
# branch could never see any clear, every probe returned 'inconclusive' with
# ok=True, and the gate refused nothing while still costing 4-40 s a
# candidate. A gate that cannot fail is worse than no gate: it shows up in the
# receipt as a check that ran.
#
# The repair is the same shape clear_verify_margin already had -- ASK the
# adapter, derive nothing twice -- plus a loud line for the residue the floor
# cannot fix (a trace shorter than the warm-up itself).
# ==========================================================================

def test_the_signals_warmup_bound_is_exact_not_a_guess() -> None:
    # Fastest physically possible fire: a full null of every channel busy,
    # then every channel silent from the first observation the short window
    # can hold. Nothing can trigger this signal sooner, and this does trigger
    # at exactly the advertised observation -- so the bound is neither
    # optimistic (a caller meeting it really can see a vote) nor padded (it
    # does not make a caller buy observations it will not use).
    bound = ApuActivitySignal().warmup_observations()
    loud_then_silent = _steady(60, 0b11111) + _steady(60, SILENT)
    sig = _feed(ApuActivitySignal(), loud_then_silent)
    assert sig.trigger_n == bound == WARMUP + ApuActivitySignal().sustain - 1

    early = _feed(ApuActivitySignal(), loud_then_silent[: bound - 1])
    assert early.trigger_n is None, "fired before its own bound"


def test_the_detectors_warmup_is_the_first_check_it_could_pass() -> None:
    # A check needs both `_n % stride == 0` and 16 buffered samples, so a
    # short stride does NOT buy an earlier verdict.
    assert StreamingConfluenceDetector(_progress).warmup_observations() == 20
    assert StreamingConfluenceDetector(
        _progress, stride=5).warmup_observations() == 20
    assert StreamingConfluenceDetector(
        _progress, stride=30).warmup_observations() == 30


def test_persistence_extends_the_warmup_by_the_checks_it_adds() -> None:
    for persist, want in ((1, 20), (2, 40), (3, 60)):
        assert StreamingConfluenceDetector(
            _progress, persist_checks=persist).warmup_observations() == want


def test_an_additive_audio_vote_does_not_hold_the_warmup_up() -> None:
    # At the default bar of 2 the two RAM signals can carry a clear alone, so
    # the audio signal's own 93-observation warm-up is not on the critical
    # path and a caller must not be charged for it.
    assert StreamingConfluenceDetector(
        _progress, apu_weight=1.0).warmup_observations() == 20


def test_a_REQUIRED_audio_vote_does_hold_it_up() -> None:
    # Above a bar of 2 the RAM signals cannot reach the bar by themselves, so
    # nothing can pass until the audio vote exists: 93 rounded up to the next
    # check point. THIS is the number the counterfactual gate was 48 short of.
    det = StreamingConfluenceDetector(_progress, min_signals=3,
                                      apu_weight=1.0)
    assert det.warmup_observations() == 100
    assert 32 + 20 < det.warmup_observations()          # the shipped defect


def test_the_budget_tracks_the_signals_own_knobs() -> None:
    det = StreamingConfluenceDetector(
        _progress, min_signals=3, apu_weight=1.0,
        apu_params={"short_window": 10, "min_baseline": 10, "sustain": 1})
    assert det._apu.warmup_observations() == 20
    assert det.warmup_observations() == 20


def test_the_budget_is_sufficient_and_not_merely_larger() -> None:
    # The point of a budget is that meeting it makes the hook USABLE. Fed a
    # stream that changes as early as it possibly can, the recommended
    # configuration fires on exactly the observation the budget names.
    det = StreamingConfluenceDetector(_progress, min_signals=3,
                                      apu_weight=1.0)
    fired = _fires(det, _load_stream(n=200, at=60), _masks_change(200, at=60))
    assert fired is not None and fired + 1 == det.warmup_observations() == 100


def test_a_window_too_small_to_check_is_reported_unsatisfiable() -> None:
    # push() also demands 16 buffered samples, so a window under that can
    # never evaluate. Better an obviously unmeetable number (which makes
    # every caller's warm-up check fail loudly) than a small one that reads
    # like a hook which merely needs a moment.
    det = StreamingConfluenceDetector(_progress, window=8)
    assert det.warmup_observations() > 10 ** 6
    assert _fires(det, _load_stream()) is None


# ---- the adapter side ----------------------------------------------------

def test_every_game_adapter_answers_the_budget_question() -> None:
    # Same contract as clear_verify_margin and needs_apu: the callers ask
    # every adapter unconditionally.
    import scripts.go_explore_solve as ges
    adapters = [v for v in vars(ges).values()
                if isinstance(v, type) and v.__name__.endswith("Game")
                and v.__module__ == ges.__name__]
    assert len(adapters) >= 2, adapters
    for cls in adapters:
        assert callable(getattr(cls, "clear_observation_budget", None)), cls


def test_a_stateless_hook_needs_no_warm_up_at_all() -> None:
    assert SmbGame.clear_observation_budget(object()) == 0
    assert _game(mode="byte_change", addr=0x75).clear_observation_budget() == 0


def test_the_confluence_budget_comes_from_the_hooks_own_detector() -> None:
    # Not a second copy of the arithmetic: the adapter builds the detector it
    # would run and asks it, so a knob can never move in one place only.
    g = _game(stride=25, persist_checks=2, min_signals=3, apu_weight=1.0)
    assert (g.clear_observation_budget()
            == g._new_clear_detector().warmup_observations() == 125)


def test_the_recommended_configuration_reports_its_real_cost() -> None:
    strict = _game(min_signals=3, apu_weight=1.0)
    assert strict.clear_observation_budget() == 100
    assert _game(min_signals=3, apu_weight=1.0,
                 apu={"min_baseline": 10,
                      "short_window": 10}).clear_observation_budget() == 40


def test_the_shipped_profiles_warm_up_inside_a_single_burst() -> None:
    # contra/gradius run the default confluence: 20 observations of warm-up
    # against a 64-action burst, so nothing about them changes and nothing
    # warns.
    import pathlib

    import yaml
    repo = pathlib.Path(__file__).resolve().parent.parent
    for name in ("contra.yaml", "gradius.yaml"):
        prof = yaml.safe_load((repo / "configs" / name).read_text())
        g = GenericGame(prof)
        assert g.clear_observation_budget() == g.clear_verify_margin() == 20
        assert g.clear_observation_budget() < 64        # the --burst default


# ---- the live-burst diagnostic -------------------------------------------

def test_a_burst_shorter_than_the_hooks_warm_up_is_called_out(capsys) -> None:
    # The live ctx is per burst, so this configuration's hook could never
    # fire during a search however long the run lasted -- the same structural
    # no-op as the gate's, one level up.
    from scripts.go_explore_solve import warn_if_burst_starves_clear_hook
    assert warn_if_burst_starves_clear_hook(
        _game(min_signals=3, apu_weight=1.0), 64) == 100
    out = capsys.readouterr().out
    assert "never fire" in out and "100" in out and "64" in out


def test_a_burst_that_covers_the_warm_up_is_not_second_guessed(capsys) -> None:
    from scripts.go_explore_solve import warn_if_burst_starves_clear_hook
    assert warn_if_burst_starves_clear_hook(_game(), 64) == 20
    assert capsys.readouterr().out == ""


def test_the_diagnostic_tolerates_an_adapter_that_predates_it(capsys) -> None:
    # Unlike the gate (whose verdict depends on the number and so fails
    # closed), a construction-time diagnostic must never take a run down.
    from scripts.go_explore_solve import warn_if_burst_starves_clear_hook
    assert warn_if_burst_starves_clear_hook(object(), 64) == 0
    assert capsys.readouterr().out == ""


def test_the_solver_runs_the_diagnostic_at_construction() -> None:
    import inspect

    import scripts.go_explore_solve as ges
    assert "warn_if_burst_starves_clear_hook" in inspect.getsource(
        ges.Solver.__init__)


# ---- the probe's floor ---------------------------------------------------

def _segments(pool: _CfPool) -> list[int]:
    """Length of every replay segment the probe drove, in order:
    [head, control, branch 0 .. branch k-1]."""
    return [len(b) for b in pool.branches]


def test_the_probe_buys_the_hook_the_observations_it_needs(cf_root) -> None:
    # THE REGRESSION for the shipped no-op. A hook that needs 100
    # observations and is handed a 20-action margin must get 80 pre-steps,
    # whatever the flag asked for -- so every branch, control included, feeds
    # it exactly its warm-up.
    _CfPool.script = staticmethod(_never)
    fake = _cf_solver(cf_root, game=_CfGame(margin=20, budget=100),
                      cf_pre_steps=32, cf_branches=3)
    out = Solver.counterfactual_probe(fake, "entrance", [1] * 200)
    assert out["pre_steps_requested"] == 32
    assert out["first_pass"]["pre_steps"] == 80
    assert out["warmup_budget"] == 100 and out["margin"] == 20
    assert out.get("warmup_short") is None
    # head, then the control and the three branches of the FIRST pass -- the
    # blind control then triggers the deep retry, whose segments are longer
    # still (its own test); what is pinned here is that no branch of the pass
    # the floor sized was ever handed less than the hook's 100 observations.
    assert _segments(_CfPool.instances[-1])[1:5] == [100] * 4


def test_the_floor_only_ever_raises_the_budget(cf_root) -> None:
    _CfPool.script = staticmethod(_never)
    fake = _cf_solver(cf_root, game=_CfGame(margin=20, budget=100),
                      cf_pre_steps=150, cf_branches=1)
    out = Solver.counterfactual_probe(fake, "entrance", [1] * 200)
    assert out["pre_steps"] == 150


def test_a_stateless_hook_pays_nothing_for_the_floor(cf_root) -> None:
    # Byte-identical to the pre-fix probe on every profile shipping today:
    # budget 0 means the requested pre-steps are used verbatim.
    _CfPool.script = staticmethod(_commits)
    fake = _cf_solver(cf_root, cf_pre_steps=4)
    out = Solver.counterfactual_probe(fake, "entrance", TRACE)
    assert out["pre_steps"] == 4 and out["warmup_budget"] == 0
    assert out["verdict"] == "commits"


def test_a_trace_too_short_for_the_warm_up_says_so_out_loud(
        cf_root, capsys) -> None:
    # The floor cannot conjure actions that do not exist. What it must not do
    # is report the resulting blindness as an ordinary inconclusive.
    _CfPool.script = staticmethod(_commits)
    fake = _cf_solver(cf_root, game=_CfGame(margin=0, budget=100),
                      cf_branches=2)
    out = Solver.counterfactual_probe(fake, "entrance", [1] * 30)
    assert out["warmup_short"] is True and out["pre_steps"] == 30
    err = capsys.readouterr().err
    assert "CANNOT JUDGE" in err and "MISSING check" in err


def test_a_starved_probe_is_labelled_differently_from_a_blind_control(
        cf_root) -> None:
    # One is a hole in the harness -- the hook was never fed enough to look --
    # and the other is a property of the candidate: it looked and saw nothing.
    # A receipt that cannot tell them apart is how the no-op stayed invisible,
    # and since 2026-08-10 they do not even share a verdict.
    _CfPool.script = staticmethod(_never)
    starved = Solver.counterfactual_probe(
        _cf_solver(cf_root, game=_CfGame(margin=0, budget=100),
                   cf_branches=1), "entrance", [1] * 30)
    blind = Solver.counterfactual_probe(
        _cf_solver(cf_root, cf_branches=1), "entrance", TRACE)
    assert starved["verdict"] == "inconclusive" and starved["ok"] is True
    assert starved["reason"] == "warmup_short"
    assert "retry" not in starved, "a starved probe has nothing to re-snapshot"
    assert blind["verdict"] == "state_artifact" and blind["ok"] is False
    assert "warmup_short" not in blind


def test_an_adapter_that_cannot_state_its_budget_fails_closed(
        cf_root) -> None:
    # A silent 0 here would restore the exact defect: a number nobody
    # measured, standing in for the one that decides whether the branches
    # could see anything.
    class _LegacyGame(_CfGame):
        def __getattribute__(self, name):
            if name == "clear_observation_budget":
                raise AttributeError(name)
            return super().__getattribute__(name)

    _CfPool.script = staticmethod(_commits)
    _CfPool.instances = []
    out = Solver.counterfactual_probe(
        _cf_solver(cf_root, game=_LegacyGame()), "entrance", TRACE)
    assert out["ok"] is False and out["verdict"] == "error"
    assert "clear_observation_budget" in out["error"]
    assert _CfPool.instances == [], "spent an emulator on an unjudgeable probe"


def test_the_real_confluence_adapter_gets_a_usable_branch(cf_root) -> None:
    # End to end on the REAL GenericGame, configured exactly as the detector
    # docs recommend for RAM-shaped false positives -- the combination the
    # gate provably could not judge before. Every branch now feeds the live
    # hook at least its own warm-up.
    _CfPool.script = staticmethod(_never)
    game = _game(min_signals=3, apu_weight=1.0)
    fake = _cf_solver(cf_root, game=game, start_wd=(), start_lives=0,
                      cf_branches=2, cf_pre_steps=32)
    out = Solver.counterfactual_probe(fake, "entrance", [1] * 300)
    assert out["warmup_budget"] == 100 and out["margin"] == 20
    assert out["pre_steps"] + out["margin"] >= out["warmup_budget"]
    assert out.get("warmup_short") is None
    assert all(seg >= 100 for seg in _segments(_CfPool.instances[-1])[1:])
