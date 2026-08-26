"""Multi-modal stage-clear confluence detector.

Declares a level-clear event by combining four independent, cheap-to-compute
signals into a weighted vote instead of trusting any single one of them.
Every signal is derived purely from observables already produced by our own
rollouts (audio samples, RAM snapshots, and differential re-simulation from a
saved state) -- no external RAM maps or disassembly are consulted anywhere
in this file.

Signals (equal weight 0.25 each; declare "clear" the first frame the summed
vote reaches >= 0.75, i.e. any 3 of the 4 agree):

  1. audio    - a sustained structural change in the per-second 6-band FFT
                fingerprint of the emitted audio (a victory jingle starting,
                or the level music cutting to silence). A CUSUM statistic
                over (per-second fingerprint delta - 0.25) must stay
                positive for >= 2 consecutive seconds before the signal
                fires, which is what makes this "sustained" rather than a
                one-frame audio blip (a jump/stomp sound effect).
  2. tally    - a rapid, periodic, anti-correlated pair of RAM bytes (one
                decrementing, one incrementing) within a sliding window --
                the generic fingerprint of a timer-to-score conversion
                tally. The candidate byte pair is discovered fresh from the
                rollout's own RAM deltas every time; no fixed addresses are
                assumed.
  3. lock     - a differential input-lock probe: from the current state,
                run N frames holding a directional input and, separately,
                N frames of NOOP, then diff the resulting RAM. If the two
                branches land on (near-)identical RAM, player input had no
                effect -- control is locked (cutscene / tally / transition).
  4. coord    - a sharp reset of the primary position readout back toward a
                level-start value, concurrent with a contiguous block of
                RAM (entity slots) collapsing to zero -- the fingerprint of
                a fresh level/room loading in.

A fifth signal exists but is OPT-IN and contributes nothing unless a profile
asks for it (so every existing receipt is unchanged):

  5. apu      - a sustained, coordinated change in the game's own per-channel
                APU activity vector (the 5-bit $4015 length-counter mask),
                measured against a null this run self-measured from its own
                history. NO game-content priors: nothing here knows what a
                fanfare is, only that this game's channels stopped behaving
                the way this game's channels have been behaving.

The ground-truth self-test (--test) replays real Go-Explore solution traces
from their recorded root state, finds the frame the game's own clear
predicate truly fires (the same check the solver itself uses), runs the
detector over the replay, and reports firing accuracy.

THE SELF-TEST HARNESS IS PROFILE-DRIVEN (`--profile configs/<game>.yaml`).
It used to open with `game = SmbGame()` and replay every trace through SMB's
action space at SMB's frame_skip, which meant no non-SMB profile could reach
the detector AT ALL. That is why the 2026-08-26 clear-detection census
(docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md) surveyed 29 profiles
and exercised the detector on exactly one of them: the 28 silences it
collected were a property of this entry point, not of those games. With no
--profile the harness is still the historical SMB one, byte-identically;
with one, the game adapter (go_explore_solve.make_game), the action space the
recorded action indices index into, and the frame_skip the trace was recorded
at all come from the profile. Every receipt now carries a `harness:` block
naming what actually replayed. Regression guard:
tests/test_clear_detect_profile_entry.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nes_core  # noqa: E402

from go_explore_solve import SmbGame  # noqa: E402  (repo's own verified SMB adapter)
# Room-graph identity layer (ROOMGRAPH_ENGINE_2026-08-24 Sec2): pure, already
# receipted in tests/test_room_fp.py. Reused verbatim by RoomFpTransitionSignal
# below -- no new hashing/classification logic is introduced in this file.
from go_explore_solve import (  # noqa: E402
    RoomIndex, classify_transition, fp_settle, nt_fingerprint, room_fp_mask,
)
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402
# Per-signal admissibility. Pure: no ROM, no emulator, no nes_core -- it
# reads this file's own constants by AST rather than importing it, so the
# dependency runs one way only and a bare checkout can still lint a profile.
import clear_reachability  # noqa: E402

FRAME_HZ = 60                     # NTSC
FS = 4                             # solver convention: frame_skip used to record traces
# The 11-action space used by the later (post-maze) solver campaigns
# (configs/smb_4_4_micro.yaml) is the canonical 6-action space (shared
# verbatim by every mario_*_solo/canonical/vanilla config) plus 5 more
# entries -- the first 6 button combos are byte-identical across every SMB
# profile in configs/, so this one space replays traces recorded under
# either family correctly (verified against both: replaying reproduces the
# solver's own recorded clear_wd at the recorded step count for every run
# in DEFAULT_RUNS below).
ACTION_SPACE = [[], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
                ["A"], ["left"], ["left", "A"], ["down"], ["down", "right"],
                ["down", "left"]]

# ---------------------------------------------------------------------------
# Weights / threshold
# ---------------------------------------------------------------------------
WEIGHTS = {"audio": 0.25, "tally": 0.25, "lock": 0.25, "coord": 0.25}
THRESHOLD = 0.75


# ===========================================================================
# Signal 1 -- audio cadence
# ===========================================================================

N_BANDS = 6
BAND_EDGES_HZ = (0, 250, 500, 1000, 2000, 4000, None)  # None -> Nyquist
AUDIO_DELTA_GATE = 0.25     # per the validated recipe: deltas > 0.25 matter
AUDIO_SUSTAIN_SECS = 2      # consecutive seconds the CUSUM must stay positive
AUDIO_HOLD_SECS = 3         # how long the vote stays raised once it fires


def band_fingerprint(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalized 6-band energy fingerprint (sums to 1) of ~1s of audio."""
    if samples is None or samples.size < 8:
        return np.zeros(N_BANDS, dtype=np.float64)
    x = samples.astype(np.float64)
    x = x - x.mean()
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    nyq = sample_rate / 2.0 + 1.0
    edges = [e if e is not None else nyq for e in BAND_EDGES_HZ]
    fp = np.zeros(N_BANDS, dtype=np.float64)
    for b in range(N_BANDS):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        fp[b] = spec[m].sum() if m.any() else 0.0
    total = fp.sum()
    if total > 1e-9:
        fp /= total
    return fp


class AudioCadenceSignal:
    """Accumulates raw audio, emits one fingerprint per elapsed second, and
    runs a CUSUM change-point detector over the fingerprint deltas.

    NOTE (empirically found, see the ground-truth test report): NES chiptune
    music churns second-to-second on its own (melody notes, drum hits) hard
    enough that per-second band-fingerprint deltas > 0.25 happen constantly
    during ordinary play, not just at a real structural change. A CUSUM
    over the raw deltas therefore usually declares its first change point
    within the first few seconds of a level on perfectly ordinary music,
    not at the clear. This signal is kept exactly per the validated recipe
    (deltas > 0.25, CUSUM change-point) because it is still a real,
    honestly-computed vote -- it just rarely ends up being one of the three
    that carry a given clear detection for a single level replay in
    practice (see per_run contributions_at_detect in the receipt). It is
    not deleted: on games/segments with calmer audio (or a hard cutoff to
    silence) it is expected to contribute, and a false-early trigger here
    is harmless by construction -- it is only 0.25 of the vote, so it can
    never cross the 0.75 threshold by itself."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self._buf = np.zeros(0, dtype=np.int16)
        self.fingerprints: list[np.ndarray] = []
        self.deltas: list[float] = []
        self.fp_frame: list[int] = []   # raw frame index the fingerprint window closed on
        self._cusum = 0.0
        self._consec = 0
        self.trigger_sec: int | None = None
        self._frame = -1
        self.n_frames_seen = 0

    def push_frame(self, audio_samples) -> None:
        self._frame += 1
        self.n_frames_seen += 1
        if audio_samples is not None and len(audio_samples):
            self._buf = np.concatenate([self._buf,
                                         np.asarray(audio_samples, dtype=np.int16)])
        if self._buf.size >= self.sample_rate:
            window = self._buf[: self.sample_rate]
            self._buf = self._buf[self.sample_rate:]
            fp = band_fingerprint(window, self.sample_rate)
            self.fp_frame.append(self._frame)
            if self.fingerprints:
                d = float(np.abs(fp - self.fingerprints[-1]).sum())
                self.deltas.append(d)
                self._cusum = max(0.0, self._cusum + (d - AUDIO_DELTA_GATE))
                self._consec = self._consec + 1 if self._cusum > 0 else 0
                if self._consec >= AUDIO_SUSTAIN_SECS and self.trigger_sec is None:
                    self.trigger_sec = len(self.fingerprints) - AUDIO_SUSTAIN_SECS + 1
            self.fingerprints.append(fp)

    def trigger_frame(self) -> int | None:
        if self.trigger_sec is None:
            return None
        return self.fp_frame[self.trigger_sec]

    def vote_at(self, frame: int) -> int:
        tf = self.trigger_frame()
        if tf is None:
            return 0
        return int(tf <= frame < tf + AUDIO_HOLD_SECS * FRAME_HZ)


# ===========================================================================
# Signal 2 -- score-tally cadence (anti-correlated periodic RAM byte pair)
# ===========================================================================

def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def score_tally_windows(ram_hist: np.ndarray, window: int = 180, stride: int = 30,
                         min_events: int = 4, period_tol: float = 0.75,
                         coincide_tol: int = 2, max_duty: float = 0.9
                         ) -> list[tuple[int, int]]:
    """Scan RAM deltas for a byte that decrements on a roughly-regular
    cadence while a *different* byte concurrently increments -- the generic
    signature of a timer/score tally. Purely data-driven: no addresses are
    assumed, every candidate is discovered fresh from this rollout's own
    deltas. Returns the list of (start, end) frame windows where such a
    pair was found."""
    ram_i = ram_hist.astype(np.int16)
    t_total = ram_i.shape[0]
    hits: list[tuple[int, int]] = []
    for start in range(0, max(t_total - window, 0) + 1, stride):
        end = min(start + window, t_total)
        seg = ram_i[start:end]
        if seg.shape[0] < 8:
            continue
        deltas = np.diff(seg, axis=0)
        neg_mask = deltas < 0
        pos_mask = deltas > 0
        neg_counts = neg_mask.sum(axis=0)
        pos_counts = pos_mask.sum(axis=0)
        n = deltas.shape[0]
        dec_candidates = np.where(
            (neg_counts >= min_events) &
            (neg_counts >= 2 * (pos_counts + 1)) &
            (neg_counts <= max_duty * n)
        )[0]
        inc_candidates = np.where(
            (pos_counts >= min_events) &
            (pos_counts >= 2 * (neg_counts + 1)) &
            (pos_counts <= max_duty * n)
        )[0]
        if dec_candidates.size == 0 or inc_candidates.size == 0:
            continue
        found = False
        for d in dec_candidates:
            d_events = np.where(neg_mask[:, d])[0]
            if d_events.size < min_events:
                continue
            gaps = np.diff(d_events)
            if gaps.size and gaps.mean() > 0 and gaps.std() > period_tol * gaps.mean():
                continue  # not periodic enough -- reject noisy/one-off decrements
            for i in inc_candidates:
                i_events = np.where(pos_mask[:, i])[0]
                if i_events.size == 0:
                    continue
                matches = sum(1 for ev in d_events
                              if np.any(np.abs(i_events - ev) <= coincide_tol))
                if matches / len(d_events) >= 0.5:
                    found = True
                    break
            if found:
                break
        if found:
            hits.append((start, end))
    return hits


# ===========================================================================
# Signal 3 -- input-lock differential probe
# ===========================================================================

LOCK_PROBE_FRAMES = 60
LOCK_DIFF_TOL = 2      # bytes allowed to differ and still call it "locked"


def differential_input_lock_probe(env, dir_bitmask: int,
                                   probe_frames: int = LOCK_PROBE_FRAMES
                                   ) -> tuple[bool, int]:
    """From env's CURRENT state: run probe_frames holding dir_bitmask, then
    (from the same saved point) probe_frames of NOOP, diff the resulting
    RAM, and restore env to the pre-probe state before returning. A
    near-zero diff means player input had no effect: control is locked."""
    state = env.save_state()
    for _ in range(probe_frames):
        env.step(int(dir_bitmask))
    env.get_audio()  # drain -- these samples never happened on the real timeline
    ram_a = np.array(env.get_ram_range(0, 2048), dtype=np.int16)
    env.load_state(state)
    for _ in range(probe_frames):
        env.step(0)
    env.get_audio()
    ram_b = np.array(env.get_ram_range(0, 2048), dtype=np.int16)
    env.load_state(state)   # resume the real trajectory exactly where it left off
    n_diff = int(np.count_nonzero(np.abs(ram_a - ram_b)))
    return n_diff <= LOCK_DIFF_TOL, n_diff


class InputLockTrack:
    """Records discrete probe results and fills a per-frame vote by holding
    each probe's verdict for the stride that follows it."""

    def __init__(self, stride: int):
        self.stride = stride
        self.probes: list[tuple[int, bool, int]] = []  # (frame, locked, n_diff)

    def record(self, frame: int, locked: bool, n_diff: int) -> None:
        self.probes.append((frame, locked, n_diff))

    def vote_at(self, frame: int) -> int:
        # last probe at or before `frame` governs the block it opened.
        best = None
        for f, locked, _ in self.probes:
            if f <= frame:
                best = locked
            else:
                break
        return int(bool(best))


# ===========================================================================
# Signal 3b -- input-lock differential probe, K-branch, self-calibrated
# (in-loop generalization of differential_input_lock_probe above)
# ===========================================================================

LOCK_BRANCHES_K = 4          # 1 NOOP control + (K-1) branches from the
                              # profile's own action space
LOCK_NULL_QUANTILE = 0.01    # low tail of the ORDINARY-play diff null
LOCK_FRAC = 0.5              # fraction of branch PAIRS that must read
                              # "suspiciously similar" to call it LOCKED
LOCK_CALIB_SAMPLES = 30      # ordinary-play points sampled for the null
LOCK_CALIB_DRIVE_FRAMES = 30 # frames of real play advanced between samples
LOCK_PROBE_EVERY = 0         # 0 = no fixed duty cycle (see class docstring)


def _lock_snapshot(env) -> np.ndarray:
    """CPU RAM ($0000-$07FF, 2048 bytes) concatenated with primary OAM (256
    bytes) as one int16 vector. This is CORRECTION 2's "(and OAM)": a branch
    that leaves CPU RAM alone but moves sprites -- the exit-pipe /
    scripted-object case Solver.counterfactual_probe's own docstring names
    -- must not read as locked just because the original 2048-byte RAM-only
    diff (clear_detect.py:266-288) never looked at OAM at all.

    Purity: both surfaces are raw hardware bytes read through the emulator's
    own accessors (get_ram_range, peek_oam); nothing here is addressed by
    meaning."""
    ram = np.asarray(env.get_ram_range(0, 2048), dtype=np.int16)
    oam_bytes = env.peek_oam()
    oam = np.frombuffer(oam_bytes, dtype=np.uint8).astype(np.int16)
    return np.concatenate([ram, oam])


def _run_lock_branches(env, bitmasks, base_state, probe_frames: int,
                        branches: int, rng: np.random.Generator
                        ) -> list[np.ndarray]:
    """From `base_state`, replay `branches` independent futures of
    `probe_frames` frames each: branch 0 is the NOOP control, branches
    1..branches-1 each sample `bitmasks` (the profile's OWN action space)
    uniformly at random, ONE draw per branch held for the whole probe (a
    constant hold per branch, not a fresh draw per frame -- the same
    "committed" shape counterfactual_probe's sparse-perturbation note
    reasons about, just simpler here since this is a short in-loop probe
    rather than a multi-second banking-gate replay).

    env is restored to `base_state` between every branch and again before
    returning, so this never advances the real trajectory."""
    snaps: list[np.ndarray] = []
    for b in range(max(2, int(branches))):
        env.load_state(base_state)
        mask = 0 if b == 0 else int(rng.choice(bitmasks))
        for _ in range(max(1, int(probe_frames))):
            env.step(mask)
        env.get_audio()  # drain -- these samples never happened on the real timeline
        snaps.append(_lock_snapshot(env))
    env.load_state(base_state)
    return snaps


def _lock_pair_diffs(snaps: list[np.ndarray]) -> list[int]:
    """Every C(len(snaps), 2) pairwise byte-diff count, RAM+OAM combined."""
    out = []
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            out.append(int(np.count_nonzero(np.abs(snaps[i] - snaps[j]))))
    return out


def measure_input_lock_null(env, bitmasks, probe_frames: int = LOCK_PROBE_FRAMES,
                             branches: int = LOCK_BRANCHES_K,
                             n_samples: int = LOCK_CALIB_SAMPLES,
                             drive_frames: int = LOCK_CALIB_DRIVE_FRAMES,
                             rng: np.random.Generator | None = None
                             ) -> np.ndarray:
    """CORRECTION 1: measure the branch-pair diff distribution this SPECIFIC
    game produces under ORDINARY, unlocked play, so the LOCKED threshold can
    be a per-profile quantile of that instead of the shipped global
    LOCK_DIFF_TOL=2 -- a constant with no per-game meaning that the campaign
    doc's own language notes "runs to hundreds of bytes" once real,
    responsive play is diffed branch-to-branch.

    From env's CURRENT state, repeats `n_samples` times: drive the REAL
    trajectory forward `drive_frames` frames with a uniformly random action
    (so every sample lands at a different, ordinary point rather than
    probing the same instant `n_samples` times), then from that point run
    the identical K-branch probe the live signal runs and record every
    pairwise diff. Each sample resumes the real trajectory exactly where
    the drive left it -- the same "never happened on the real timeline"
    contract every probe in this file keeps -- so the only lasting effect
    on env is the `n_samples * drive_frames` frames of ordinary play the
    calibration itself drove.

    Returns the flat array of all measured pairwise diffs. A profile whose
    game logic never varies the RAM/OAM surface under different real inputs
    (vanishingly unlikely, but not provably impossible from this file alone)
    would return a null with no separable spread; lock_null_threshold does
    not check for that, callers must (see InputLockSignal.calibrate)."""
    rng = rng or np.random.default_rng()
    diffs: list[int] = []
    for _ in range(max(1, int(n_samples))):
        for _ in range(max(0, int(drive_frames))):
            env.step(int(rng.choice(bitmasks)))
        env.get_audio()
        sample_state = env.save_state()
        snaps = _run_lock_branches(env, bitmasks, sample_state, probe_frames,
                                    branches, rng)
        diffs.extend(_lock_pair_diffs(snaps))
        env.load_state(sample_state)  # ordinary drive continues from here
    return np.array(diffs, dtype=np.int64)


def lock_null_threshold(null_diffs: np.ndarray,
                         quantile: float = LOCK_NULL_QUANTILE) -> int:
    """The per-profile LOCKED threshold CORRECTION 1 calls for: the
    `quantile`-th percentile of a MEASURED ordinary-play branch-diff null
    (measure_input_lock_null's output).

    Deliberately a LOW quantile by default (1%). Ordinary, unlocked play
    diverges branch-to-branch by "hundreds of bytes" (the campaign doc's own
    measurement) because position, animation phase and timers all depend on
    which input each branch held; the threshold sits at the bottom sliver of
    THAT distribution, so a diff this small or smaller is evidence the
    branches did NOT diverge the way this game's ordinary play does -- not
    "still within a normal range" read off some other game's constant.

    A null with no separable lower tail (this game's branches diverge by
    only a handful of bytes even under real, unlocked play) makes any
    threshold drawn from it indistinguishable from noise; a profile in that
    shape should decline to arm this signal rather than accept a number,
    which is why this function does not attempt to detect or refuse that
    case itself -- it has no opinion on what "separable" means for a game it
    has never seen, only on the arithmetic of one quantile."""
    if null_diffs is None or null_diffs.size == 0:
        return 0
    q = max(0.0, min(1.0, float(quantile)))
    return int(np.quantile(null_diffs, q))


def differential_input_lock_probe_k(env, bitmasks, threshold: int,
                                     probe_frames: int = LOCK_PROBE_FRAMES,
                                     branches: int = LOCK_BRANCHES_K,
                                     lock_frac: float = LOCK_FRAC,
                                     rng: np.random.Generator | None = None
                                     ) -> tuple[bool, float, list[int]]:
    """CORRECTION 2, in-loop form: from env's CURRENT state, replay one NOOP
    control branch plus `branches - 1` branches each sampling `bitmasks`
    (the profile's OWN action space) for `probe_frames` frames, diff every
    resulting pair of RAM+OAM snapshots, and report LOCKED when the FRACTION
    of pairs whose diff falls AT OR UNDER `threshold` exceeds `lock_frac`.

    `threshold` has no default here -- unlike the 2-byte LOCK_DIFF_TOL this
    generalizes, "how many bytes is suspiciously few" is a per-game question
    this function refuses to answer for the caller. It must come from
    lock_null_threshold(measure_input_lock_null(...)) (or an
    InputLockSignal that has run calibrate()).

    Two branches (a single held direction vs. NOOP) is not enough: a real
    clear whose last second contains a REQUIRED input -- the exit-pipe case
    Solver.counterfactual_probe's own docstring measured, where a branch
    holding NOOP or `right` simply never enters the pipe -- can score as
    "diverged" on that one branch while every other branch agrees, and a
    2-branch probe has no way to average that outlier away. Voting on the
    FRACTION of C(branches, 2) pairs is what lets this generalize past 2
    without hand-picking which single branch is "the" comparison.

    env is restored to its pre-probe state before returning: none of this
    ever happened on the real timeline, exactly like differential_input_lock_probe."""
    rng = rng or np.random.default_rng()
    base_state = env.save_state()
    snaps = _run_lock_branches(env, bitmasks, base_state, probe_frames,
                                branches, rng)
    env.load_state(base_state)
    diffs = _lock_pair_diffs(snaps)
    n_pairs = len(diffs)
    n_under = sum(1 for d in diffs if d <= threshold)
    frac = (n_under / n_pairs) if n_pairs else 0.0
    locked = frac > float(lock_frac)
    return locked, frac, diffs


class InputLockSignal:
    """TIER-2 signal: the differential input-lock probe promoted from
    offline-only (differential_input_lock_probe above, still unchanged and
    still used by run_episode's four-signal replay) to an armed, in-loop
    form, with the two corrections the campaign doc's strategy phase named.

    CORRECTION 1 -- NO GLOBAL CONSTANT. The shipped LOCK_DIFF_TOL=2 bytes
    out of 2048 has no per-game meaning; this class instead holds a
    `threshold` earned by calibrate() from THIS profile's own measured
    branch-diff null (measure_input_lock_null / lock_null_threshold).
    probe() raises rather than fall back to any default if calibrate() was
    never run -- a profile that cannot be calibrated must not silently
    inherit a number that was never about it.

    CORRECTION 2 -- K BRANCHES, NOT 2. One NOOP control plus K-1 branches
    sampling the profile's OWN action space, voted by the FRACTION of
    branch pairs that read as suspiciously similar (>= lock_frac), so one
    branch landing on a required input (the exit-pipe case) cannot by
    itself flip a real clear to "unlocked".

    FALSE POSITIVES -- every one of these is INPUT-INDEPENDENT RAM/OAM
    evolution, which is exactly what this signal cannot tell apart from a
    committed transition, because it never looks at WHAT changed, only
    whether changing the input changed it:
      * DEATH -- a death animation runs the same whether the player holds
        A or nothing; this signal reads LOCKED on every death.
      * PAUSE -- input is explicitly disabled; LOCKED by definition.
      * SCRIPTED INTRO / CUTSCENE -- LOCKED for its whole duration.
      * A LAG or DMA-STALL frame can read LOCKED at short probe_frames
        purely because too few frames elapsed for any branch to diverge yet.
      * ATTRACT LOOP -- the measured Galaga case ("lives, progress and PPU
        vertical scroll are all byte-identical under hold_right vs hold_A")
        is a non-interactive presentation loop that reads LOCKED 100% of
        the time, forever, with no probe_frames budget large enough to
        outlast it. This is NOT a nuisance case; it is why a caller wiring
        this signal into a vote MUST pair it with a separate attract_loop
        veto rather than trust `require` alone.
      * A SAMPLING COLLISION is a real, if narrower, residual: `bitmasks`
        is sampled WITH REPLACEMENT per branch (see _run_lock_branches), so
        two non-control branches can legitimately draw the identical action
        by chance, and on a small action space (or a small `branches`) that
        pair reads as perfectly LOCKED even in fully ordinary, responsive
        play. This is a property of "sample uniformly per branch" as
        specified, not a bug introduced here; measure_input_lock_null's own
        null is exposed to the exact same collisions (it runs the same
        _run_lock_branches), so a game where this matters shows it as
        thicker low-end mass in the null BEFORE it ever reaches probe() --
        which is one more reason the threshold must come from that
        game's own measured null and never from a hand-picked constant.
      Because every listed cause but the last is a strict superset of "the
      game stopped responding to input", this signal can assert LOCKED with
      total confidence and STILL be wrong about whether a clear happened --
      it must never be the sole term in a clear vote, only a corroborator
      alongside a signal that answers "did anything actually change"
      (entity_wipe, oam_quiesce, room_fp_transition, scene_cut) and,
      wherever lives can drop, a `lives_drop` veto.

    CALIBRATION COST, why this is TIER-2. Each probe costs
      2 * branches * probe_frames  frames of emulation
      + 2 * branches               save/load_state round-trips,
    which is the same order of magnitude as the measured banking-path
    counterfactual probe (4-40s per candidate at its much larger
    probe_frames). It must therefore be armed by a caller ONLY once a
    tier-0 signal (a cheap, always-on RAM/OAM/scene signal) has already
    raised its own vote for the current window -- never run on a fixed
    stride in the hot loop. `probe_every` exists here ONLY as a
    fallback fixed-duty mode (probe_every=0 disables it, the default);
    should_probe() returning True is never itself sufficient justification
    to call probe(), it is the SIMPLEST possible arming rule and a caller
    with a real tier-0 signal should use that instead and ignore
    should_probe() entirely.

    PREFLIGHT. input_lock_preflight() below runs this signal from a
    profile's own start state after ordinary settle play and demands
    UNLOCKED -- a profile that reads LOCKED there is either stuck in an
    attract loop or wired to a broken action space/probe, and either way
    every number this signal produces downstream is void until that is
    fixed. This must run once per profile at construction time, the same
    discipline ApuActivitySignal.warmup_observations() and this file's
    other unsatisfiable-configuration checks already apply."""

    def __init__(self, bitmasks, probe_frames: int = LOCK_PROBE_FRAMES,
                 branches: int = LOCK_BRANCHES_K,
                 quantile: float = LOCK_NULL_QUANTILE,
                 lock_frac: float = LOCK_FRAC,
                 probe_every: int = LOCK_PROBE_EVERY,
                 rng: np.random.Generator | None = None):
        self.bitmasks = [int(b) for b in bitmasks]
        if len(self.bitmasks) < 1:
            raise ValueError("InputLockSignal needs a non-empty action space")
        self.probe_frames = max(1, int(probe_frames))
        self.branches = max(2, int(branches))
        self.quantile = float(quantile)
        self.lock_frac = float(lock_frac)
        self.probe_every = max(0, int(probe_every))
        self._rng = rng if rng is not None else np.random.default_rng()
        self.threshold: int | None = None
        self.null_diffs: np.ndarray | None = None
        self.n_probes = 0
        self.last_locked: bool | None = None
        self.last_frac = 0.0
        self.last_diffs: list[int] = []
        self._since_probe = 0

    def calibrate(self, env, n_samples: int = LOCK_CALIB_SAMPLES,
                  drive_frames: int = LOCK_CALIB_DRIVE_FRAMES) -> int:
        """Earn `threshold` from THIS env's own ordinary-play null. Costs
        `n_samples * drive_frames` frames of real advancement (env is left
        that far forward, not restored -- calibration play is real play)
        plus the K-branch probe cost at each of the `n_samples` points."""
        self.null_diffs = measure_input_lock_null(
            env, self.bitmasks, self.probe_frames, self.branches,
            n_samples, drive_frames, self._rng)
        self.threshold = lock_null_threshold(self.null_diffs, self.quantile)
        return self.threshold

    def ready(self) -> bool:
        return self.threshold is not None

    def should_probe(self) -> bool:
        """Fixed-duty FALLBACK only -- see the class docstring's TIER-2
        note. Returns False forever at the default probe_every=0."""
        if self.probe_every <= 0:
            return False
        self._since_probe += 1
        if self._since_probe >= self.probe_every:
            self._since_probe = 0
            return True
        return False

    def probe(self, env) -> bool:
        if self.threshold is None:
            raise RuntimeError(
                "InputLockSignal.probe() called before calibrate() -- there "
                "is no default threshold on purpose (see lock_null_threshold "
                "and CORRECTION 1); a profile that cannot be calibrated must "
                "not silently fall back to the retired global LOCK_DIFF_TOL=2.")
        locked, frac, diffs = differential_input_lock_probe_k(
            env, self.bitmasks, self.threshold, self.probe_frames,
            self.branches, self.lock_frac, self._rng)
        self.n_probes += 1
        self.last_locked = locked
        self.last_frac = frac
        self.last_diffs = diffs
        return locked

    def stats(self) -> dict:
        return {"n_probes": self.n_probes, "threshold": self.threshold,
                "last_locked": self.last_locked,
                "last_frac": round(self.last_frac, 4),
                "branches": self.branches, "probe_frames": self.probe_frames,
                "lock_frac": self.lock_frac,
                "null_n": int(self.null_diffs.size)
                          if self.null_diffs is not None else 0}


def input_lock_preflight(signal: InputLockSignal, env,
                          settle_steps: int = 60) -> dict:
    """PREFLIGHT this signal's failing_test names explicitly: from the
    profile's OWN start state, after `settle_steps` of ordinary forward
    play, the probe MUST report UNLOCKED. Calibrates first if the signal
    has not been calibrated yet. Returns ok=False (never raises) so a
    caller can decide whether an unmet preflight refuses construction or
    only disarms this one signal -- the same "reason in the message, not a
    silent no-op" shape this file's other validators use."""
    move = signal.bitmasks[1] if len(signal.bitmasks) > 1 else signal.bitmasks[0]
    for _ in range(max(0, int(settle_steps))):
        env.step(move)
    if not signal.ready():
        signal.calibrate(env)
    locked = signal.probe(env)
    return {"ok": not locked, "locked": locked, "last_frac": signal.last_frac,
            "threshold": signal.threshold, "null_n": signal.stats()["null_n"]}


# ===========================================================================
# Signal 3c -- TERMINAL STASIS: the absorbing, input-dead search dead end
# (the D1 repair -- a death the lives byte never reports)
# ===========================================================================
#
# WHAT THIS IS FOR. go_explore_solve's death predicate is a LIVES-BYTE
# predicate: `(start_lives - lives) % 256 in 1..8`, plus an optional
# player-state whitelist. It therefore cannot see a terminal state that
# leaves the lives byte UNCHANGED. Measured, reproduced 2026-08-26 on
# configs/ninja_gaiden_ii.yaml from its own start state:
#
#     hold RIGHT from the profile's start state
#       step  54..652  the profile's discovered lives byte $004C ticks
#                      1 -> 0 -> 1 on every individual death; is_dead()
#                      fires correctly on each one (delta = 1)
#       step ~800..    GAME OVER.  $004C reads 1 and NEVER MOVES AGAIN.
#                      is_dead() is False on every one of those frames.
#
# The wave-4 smoke run banked 246,836 steps and 2,216 cells with no death
# pruning past that point, and its deepest frontier cell logged 15,579
# revisits: the search ground a frozen GAME OVER screen. Nothing in the
# RAM the lives predicate reads distinguishes that screen from play.
#
# WHAT IS OBSERVABLE WITHOUT GAME KNOWLEDGE. Two things, and this signal
# is their CONJUNCTION -- neither alone is safe:
#
#   FROZEN (tier-0, always on, ~free).  The CPU-RAM surface stops moving:
#     it stays within `churn_tol` byte positions of a single anchor for
#     `window` consecutive steps of REAL play, while the agent is emitting
#     its own varied actions.  Measured separation on NG2, 2026-08-26:
#     ordinary play moves a MEDIAN of 420 of 2,048 RAM bytes per 45-step
#     window and 332 at its 1st percentile; the GAME OVER screen holds 8-12
#     bytes away from any anchor, indefinitely, drawn from a fixed set of 14
#     addresses (frame and RNG counters, their mirrors, stack churn).
#     `churn_tol` is NOT a shipped constant -- it is derived from the null
#     THIS profile measured from its own start state (the same discipline as
#     lock_null_threshold / CORRECTION 1); on NG2 it lands at 21.
#     This track only ARMS; it never convicts, because 45 steps of one
#     agent's own action stream holding still is exactly what an idle
#     player produces.
#
#   ABSORBING (tier-2, rare, one probe -- THE VERDICT).  From the armed
#     frame, replay `probe_steps` (150, ~10 s of game time) on each of K
#     branches -- the NOOP control plus branches each HOLDING one action
#     sampled from the profile's own space -- and require that NO BRANCH
#     ESCAPES: every one ends within `churn_tol` byte positions of the frame
#     it started from, over RAM AND OAM.  This runs the SHIPPED K-branch
#     machinery (_run_lock_branches, the same re-simulation
#     differential_input_lock_probe_k drives, with the same held-action
#     construction and the same restore contract), not a parallel mechanism.
#
# WHY BOTH, AND WHY THE PROBE CARRIES THE VERDICT.  The two false-positive
# classes are each other's opposites, and each half kills one:
#
#   FROZEN alone false-positives on an IDLE PLAYER.  A character standing
#     still on flat ground with no animation is frozen and perfectly alive;
#     the solver reaches that state routinely (`--sticky 0.5`, hold macros,
#     the NOOP settle before every macro).  The absorbing probe rejects it,
#     and rejects it decisively rather than marginally, because a branch
#     that HOLDS a direction for 150 steps is precisely what moves a live
#     player off the spot.  This is why the arming track never convicts.
#
#   THE PROBE alone would false-positive on every CUTSCENE, SCRIPTED INTRO,
#     PAUSE, DEATH ANIMATION and ATTRACT LOOP if it were scored the way
#     InputLockSignal is (branch-to-branch agreement) -- that signal's own
#     docstring lists exactly those classes, and they are why it is
#     documented as never sufficient alone.  Scoring DRIFT FROM THE START
#     FRAME instead of branch agreement is what changes the question from
#     "did input matter?" (yes for all of them) to "is the game going
#     anywhere?" (yes for all of them, so: rejected).  The frozen arming
#     track then makes sure the probe is not even paid for on a busy screen.
#
# ONE VERDICT CLAUSE, ON PURPOSE.  An earlier draft scored the probe twice
# -- drift from the start frame AND branch-to-branch agreement -- and called
# them corroborating conjuncts.  Branches that all sit within `churn_tol` of
# one frame are within 2*churn_tol of each other by construction, so the
# second test could only restate the first; mutation-testing confirmed it by
# deleting the drift clause and failing no test.  Two views of one
# measurement presented as corroboration is the shape of a vacuous gate, and
# it is now one clause with a test that fails when it is removed.
#
# WHY THE ARMING WINDOW IS SHORT AND THE PROBE IS LONG.  A lineage's ctx --
# and with it the held-still counter -- is rebuilt at every re-root, and
# scripts/onboard_game.py runs the solver at the default --burst 64.  An
# arming window over ~60 steps therefore cannot fire in the configuration
# D1 actually happened in.  The evidence had to move somewhere that a burst
# boundary cannot truncate, and a probe that runs off a savestate is exactly
# such a place.
#
# WHAT IT ACTUALLY CLAIMS.  Deliberately NOT "the player is dead" -- this
# signal has no access to that and never will without game knowledge.  It
# claims something weaker, fully observable, and sufficient for the caller:
# THIS LINEAGE IS IN AN ABSORBING, INPUT-DEAD STATE AND WILL NOT LEAVE IT.
# A GAME OVER screen, a soft-lock and a pause the action space cannot exit
# are the same fact to a search: no continuation of this trajectory can
# reach anything new.  Terminating the lineage is the correct action for
# all three, whatever the game calls them.
#
# WHAT IT COSTS TO BE WRONG.  A false positive terminates one burst, and
# after a second one from the same cell the caller may stop selecting that
# cell; a false NEGATIVE is what D1 was.  That asymmetry is real but bounded,
# and it is why the frozen track is a hard PRECONDITION rather than one vote
# in a tally: a mandatory cutscene wrongly read as terminal would make the
# level behind it unreachable forever.
#
# MEASURED FALSE-POSITIVE RATE, 2026-08-26.  29 solve profiles, calibrated
# from their own start states; 25 armed and 4 declined with a measured
# reason.  150,000 steps of ordinary play on the armed 25 (3,000 random-
# action + 3,000 NOOP-held per profile).  142 arms, 136 fires -- every fire
# on 1942 and Bad Dudes, and every screenshotted fire reads GAME OVER or
# CONTINUE.  The other 6 arms (Arkanoid x2, Blaster Master, Punch-Out x3)
# were live states and the probe rejected all 6.  ZERO false positives.
# Two of those rejections were close -- Blaster Master drifted 9/9/11/16
# against a tolerance of 14, Punch-Out 4/14/18/16 against 14 -- so the
# margin on a live-but-quiet state is one branch wide, not comfortable.

STASIS_WINDOW = 45         # consecutive REAL steps the surface must hold still
                           # before the confirming probe is even considered.
                           # BOUNDED BY THE BURST, not chosen for elegance:
                           # scripts/onboard_game.py runs the solver without
                           # --burst, i.e. at the default 64, and a lineage's
                           # ctx (and with it this counter) is rebuilt at every
                           # re-root. A window over ~60 can never arm in the
                           # configuration D1 actually happened in -- which is
                           # why the DISCRIMINATING work was moved into the
                           # probe below rather than into a long arming window.
STASIS_CHURN_FRAC = 0.05   # `churn_tol` as a fraction of THIS profile's own
                           # median window churn under ordinary play. Not a
                           # byte count -- a byte count has no cross-game
                           # meaning, which is the retired LOCK_DIFF_TOL=2
                           # lesson.
STASIS_QUIET_MULT = 3.0    # SEPARABILITY, stated against the quantity that
                           # actually causes false arms: the QUIET TAIL of
                           # ordinary play, not its median. churn_tol is
                           # CAPPED at (1st-percentile ordinary window churn)
                           # / this, so even this game's quietest ordinary
                           # moments are a multiple away from looking frozen
                           # BY CONSTRUCTION rather than by a check that could
                           # be dropped. A median-only rule cannot say that: a
                           # game can have a fat, ordinary quiet tail under a
                           # perfectly healthy median.
STASIS_TOL_FLOOR = 8       # REFUSAL floor -- the point below which the
                           # instrument has no resolution left, not a value
                           # churn_tol is raised TO. A frozen screen is not a
                           # still image in RAM: the NG2 GAME OVER screen
                           # keeps 14 distinct addresses alive (frame and RNG
                           # counters, their mirrors, and stack churn at
                           # $01F9/$01FA) and reads 8-12 of them differing
                           # from any given anchor at any instant. A tolerance
                           # under 8 cannot hold still across that on any game
                           # measured here, so a profile whose two caps drive
                           # it below 8 is told it cannot be scored rather
                           # than armed blind. A profile that arms with a
                           # SMALL tolerance is armed but insensitive -- it
                           # will only ever see a nearly perfectly still
                           # screen -- which is why the number is printed at
                           # launch and carried in every receipt.
STASIS_CALIB_STEPS = 400   # steps of ordinary random play used to measure the
                           # churn null.
STASIS_PROBE_STEPS = 150   # branch length for the confirming absorbing probe,
                           # in STEPS (the caller's env.step is one frame_skip
                           # block), not raw frames. ~10 s of game time per
                           # branch at frame_skip 4: long enough that a held
                           # direction visibly moves a live player, and long
                           # enough to outlast a transition. This is where the
                           # evidence comes from, so it is deliberately much
                           # longer than the 45-step arming window.
STASIS_CONVICTED_MAX = 16  # frames kept in the convicted ring (see
                           # TerminalStasisSignal.confirm). Small on purpose:
                           # it is a cache, and a lineage that arms is
                           # compared against every entry, so the cost of the
                           # ring is paid per ARM (rare) and never per step.
STASIS_BACKOFF = 4         # multiplier on the required hold after each
                           # REJECTED probe on the same lineage. A state that
                           # sits just inside churn_tol without being
                           # absorbing would otherwise re-arm every `window`
                           # steps forever, at branches * probe_steps = 600
                           # emulated steps a time against a 45-step window.


def ram_surface(ram) -> np.ndarray:
    """One CPU-RAM snapshot as uint8, accepting either the raw `bytes` the
    solver's hot loop hands out or an array. RAM ONLY, deliberately: the
    always-on arming path already holds this buffer from the step it just
    took, so watching it costs one comparison and no extra emulator call,
    whereas OAM would cost a `peek_oam()` per worker per step (the batching
    note in the OamQuiesceSignal section). The confirming probe below still
    diffs RAM+OAM via _lock_snapshot, so sprite-only motion is covered where
    it is affordable."""
    return (np.frombuffer(ram, dtype=np.uint8)
            if isinstance(ram, (bytes, bytearray)) else
            np.asarray(ram, dtype=np.uint8))


def measure_stasis_null(env, bitmasks, window: int = STASIS_WINDOW,
                        steps: int = STASIS_CALIB_STEPS,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """From env's CURRENT state, play `steps` steps of ordinary uniformly
    random action and return every `window`-lagged RAM churn count observed
    (`|surf_t - surf_{t-window}|_0` for t >= window).

    This ADVANCES the real trajectory by `steps` and does not restore it --
    calibration play is real play, exactly as InputLockSignal.calibrate
    documents for its own drive.

    A start state whose ordinary play reaches a terminal screen inside
    `steps` contaminates the LOW tail of this null, which drags the median
    DOWN, which makes churn_tol SMALLER, which makes the live test STRICTER.
    Contamination therefore costs detections, never false positives -- and
    the extreme case (a null with no separable spread at all) is refused
    outright by TerminalStasisSignal.calibrate."""
    rng = rng or np.random.default_rng()
    bm = [int(b) for b in bitmasks]
    ring: list[np.ndarray] = []
    out: list[int] = []
    for _ in range(max(1, int(steps))):
        env.step(int(rng.choice(bm)))
        env.get_audio()          # drain, same contract as every probe here
        cur = ram_surface(env.get_ram_range(0, 2048))
        ring.append(cur)
        if len(ring) > window:
            out.append(int(np.count_nonzero(cur != ring.pop(0))))
    return np.array(out, dtype=np.int64)


class TerminalStasisSignal:
    """FROZEN (always-on arming, per lineage) then ABSORBING (one probe) =
    a search dead end. See the section header above for what each half is
    for and what the claim is and is not.

    ONE PIECE OF EVIDENCE, MEASURED ONCE. An earlier draft of this class
    scored the probe twice -- "no branch drifted from the start frame" AND
    "the branches agree with each other" -- and presented them as
    independent conjuncts. They are not: branches that all sit within
    `churn_tol` of the same frame are within 2*churn_tol of each other by
    construction, so the second test could only ever restate the first.
    Mutation-testing said so out loud (deleting the drift clause failed no
    test at the time), and two views of one measurement dressed as
    corroboration is the shape of a vacuous gate. There is now exactly one
    verdict clause, and tests/test_terminal_stasis.py fails when it is
    removed.

    USE:
        sig = TerminalStasisSignal(bitmasks)
        sig.calibrate(env)                  # once per profile, from its start state
        if not sig.ready(): ...             # sig.reason says why; do not arm
        ...
        for each step of a lineage:
            if sig.push(state, ram) and sig.confirm(state, env):
                # this lineage is absorbing: terminate it
        sig.reset(state)                    # when the lineage is re-rooted

    The per-lineage state is a plain dict OWNED BY THE CALLER (the solver
    already threads one `ctx` per worker), not an attribute of this object:
    a single calibrated signal is shared by every worker in a pool, and
    hiding one worker's anchor inside it would let workers overwrite each
    other's evidence -- the shared-index race that dropped
    RoomFpTransitionSignal (commit 698f142)."""

    def __init__(self, bitmasks, window: int = STASIS_WINDOW,
                 churn_frac: float = STASIS_CHURN_FRAC,
                 quiet_mult: float = STASIS_QUIET_MULT,
                 tol_floor: int = STASIS_TOL_FLOOR,
                 probe_steps: int = STASIS_PROBE_STEPS,
                 branches: int = LOCK_BRANCHES_K,
                 backoff: int = STASIS_BACKOFF,
                 convicted_max: int = STASIS_CONVICTED_MAX,
                 rng: np.random.Generator | None = None):
        self.bitmasks = [int(b) for b in bitmasks]
        if not self.bitmasks:
            raise ValueError("TerminalStasisSignal needs a non-empty action space")
        self.window = max(1, int(window))
        self.churn_frac = float(churn_frac)
        self.quiet_mult = max(1e-9, float(quiet_mult))
        self.tol_floor = max(1, int(tol_floor))
        self.probe_steps = max(1, int(probe_steps))
        self.branches = max(2, int(branches))
        self.backoff = max(1, int(backoff))
        self.convicted_max = max(0, int(convicted_max))
        self._convicted: list[np.ndarray] = []
        self._rng = rng if rng is not None else np.random.default_rng()
        self.churn_tol: int | None = None
        self.churn_null: np.ndarray | None = None
        self.reason = "not calibrated"
        # Telemetry only; never consulted by a verdict.
        self.n_armed = 0
        self.n_confirmed = 0
        self.n_rejected = 0
        self.n_cached = 0
        self.n_probes = 0
        self.last_drift: list[int] = []

    # ---- calibration -------------------------------------------------

    def calibrate(self, env, steps: int = STASIS_CALIB_STEPS) -> dict:
        """Earn `churn_tol` from this env's own ordinary play. Advances the
        real trajectory by `steps` (calibration play is real play) and
        returns the receipt block.

        Never raises on an unusable null: it sets `reason` and leaves
        ready() False, because a profile that cannot be calibrated must run
        with this signal DISARMED rather than inherit a number that was
        never about it."""
        self.churn_null = measure_stasis_null(env, self.bitmasks, self.window,
                                              steps, self._rng)
        if self.churn_null.size == 0:
            self.churn_tol = None
            self.reason = (f"calibration drove {steps} steps but the window is "
                           f"{self.window}, so no lagged pair was ever formed")
            return self.stats()
        # TWO CAPS, and the tighter one wins. The first says the tolerance
        # must be a small fraction of what this game's ordinary play moves;
        # the second says it must be several times SMALLER than what its
        # quietest ordinary play moves, which is the one that actually
        # bounds false arms. Taking the minimum makes the separation a
        # property of the arithmetic rather than a separate check some later
        # edit could delete while leaving the threshold behind.
        med = float(np.median(self.churn_null))
        quiet = float(np.quantile(self.churn_null, 0.01))
        tol = min(int(round(self.churn_frac * med)),
                  int(quiet / self.quiet_mult))
        if tol < self.tol_floor:
            self.churn_tol = None
            self.reason = (
                f"no resolution left: {self.churn_frac:g}x the median ordinary "
                f"{self.window}-step churn ({med:.0f}) and 1/{self.quiet_mult:g} "
                f"of its quietest 1 percent ({quiet:.0f}) cap the tolerance at "
                f"{tol} bytes, under the {self.tol_floor}-byte floor a real "
                f"frozen screen's own frame/RNG bookkeeping needs. Ordinary "
                f"quiet and frozen cannot be told apart here")
            return self.stats()
        self.churn_tol = tol
        self.reason = "ok"
        return self.stats()

    def ready(self) -> bool:
        return self.churn_tol is not None

    # ---- tier-0: the always-on frozen track --------------------------

    def reset(self, state: dict) -> None:
        """Drop this lineage's anchor. Call whenever the lineage is
        re-rooted (a load_worker_state): the frames before and after a
        rewind are not consecutive real play, and holding an anchor across
        one would score a hold that never happened.

        The solver gets this for free -- `_assign` builds a fresh ctx dict
        per burst -- so this exists for callers that reuse one."""
        state.pop("_stasis_anchor", None)
        state["_stasis_held"] = 0
        state["_stasis_rejects"] = 0

    def required_hold(self, state: dict) -> int:
        """Held-still steps this lineage must accumulate before the next
        probe. ESCALATES with each rejected probe on the same lineage.

        Without this, a state that sits just inside `churn_tol` but is not
        absorbing re-arms every `window` steps forever, and each rearm costs
        branches * probe_steps emulated steps -- 600 at the shipped knobs,
        against a 45-step window. That is a 13x tax on exactly the lineages
        the probe already declined to kill."""
        return self.window * (self.backoff ** int(state.get("_stasis_rejects", 0)))

    def push(self, state: dict, ram) -> bool:
        """Feed one REAL-timeline RAM snapshot for one lineage. Returns True
        when the surface has held within `churn_tol` of a single anchor for
        `required_hold(state)` consecutive steps -- i.e. this lineage is a
        confirm() candidate.

        Returns False (and never raises) when the signal is not calibrated,
        so a disarmed profile costs one attribute test per step."""
        if self.churn_tol is None:
            return False
        cur = ram_surface(ram)
        anchor = state.get("_stasis_anchor")
        if anchor is None or int(np.count_nonzero(cur != anchor)) > self.churn_tol:
            # Re-anchor on the CURRENT frame: the run of held-still steps
            # restarts here, it does not merely pause.
            state["_stasis_anchor"] = cur
            state["_stasis_held"] = 0
            return False
        held = state["_stasis_held"] = int(state.get("_stasis_held", 0)) + 1
        if held < self.required_hold(state):
            return False
        # Armed. Re-anchor so a lineage that survives confirm() must earn a
        # fresh (and longer) hold before asking again, instead of re-arming
        # on every subsequent step off the same stale anchor.
        state["_stasis_anchor"] = cur
        state["_stasis_held"] = 0
        self.n_armed += 1
        return True

    # ---- the verdict: one long-horizon differential probe --------------

    def confirm(self, state: dict, env) -> bool:
        """THE LONG-HORIZON ABSORBING PROBE, and the only thing here that
        convicts.

        From `env`'s current state -- which must be this lineage's current
        frame -- run the shipped K-branch machinery (`_run_lock_branches`,
        the same re-simulation differential_input_lock_probe_k drives, with
        the same one-held-action-per-branch construction and the same
        restore contract): a NOOP control plus `branches - 1` branches each
        HOLDING one action sampled from the profile's own space for
        `probe_steps`. TERMINAL iff NO BRANCH ESCAPES -- every one of them
        ends within `churn_tol` byte positions of the frame the probe
        started from, over RAM AND OAM.

        WHY THIS IS THE CLAUSE. 45 steps of one agent's own action stream
        holding still is weak evidence: an idle player produces it, and the
        solver idles constantly (`--sticky 0.5`, hold macros, the NOOP
        settle before every macro). 150 steps of a HELD DIRECTION failing to
        move anything is not weak. A held direction is precisely what moves
        a live player off a spot, and running K of them from a savestate is
        precisely what the differential probe was built to do.

        WHY IT READS OAM. `_lock_snapshot` concatenates primary OAM onto CPU
        RAM for the reason its own docstring gives: a branch can leave CPU
        RAM alone and still move sprites. Scoring drift on the RAM prefix
        alone would call such a state absorbing.

        WHAT IT STILL CANNOT SEE, stated rather than hidden: an ABSORBING
        SCREEN THAT ANIMATES. A GAME OVER that blinks, runs a countdown, or
        keeps a sprite alive moves more than `churn_tol` and this returns
        False. That is a deliberate trade: the clause is the only thing
        standing between this repair and killing every lineage that enters a
        mandatory cutscene, which would make the level behind it unreachable
        forever. A missed kill costs compute; a wrong kill costs a game."""
        if not self.ready():
            return False
        here = _lock_snapshot(env)
        # ALREADY PROVEN. A game whose run ends on a CONTINUE screen sends
        # every lineage to the SAME absorbing frame, and re-proving it costs
        # branches * probe_steps emulated steps EVERY time. Measured on 1942,
        # 2026-08-26, 5 minutes, 4 workers: without this ring 123 probes and
        # 217 sps against 1,156 with the signal off; with it 567 arms cost 46
        # probes and 521 ring hits, and throughput came back to 808 sps. A
        # frame that is within `churn_tol` of one already
        # proven absorbing, and that has ALSO just held still for the full
        # arming window, is that state again. The tolerance is the same one
        # the arming track uses and is capped at a third of this profile's
        # quietest ordinary churn, so "within churn_tol of a proven-dead
        # frame" is not a loose match.
        for known in self._convicted:
            if int(np.count_nonzero(here[:known.size] != known)) <= self.churn_tol:
                self.n_cached += 1
                self.n_confirmed += 1
                self.last_drift = []
                return True
        base = env.save_state()
        snaps = _run_lock_branches(env, self.bitmasks, base,
                                   self.probe_steps, self.branches, self._rng)
        env.load_state(base)
        self.n_probes += 1
        drift = [int(np.count_nonzero(s != here)) for s in snaps]
        self.last_drift = drift
        terminal = bool(drift) and max(drift) <= self.churn_tol
        if terminal:
            self.n_confirmed += 1
            if self.convicted_max:
                self._convicted.append(here.copy())
                del self._convicted[:-self.convicted_max]
        else:
            self.n_rejected += 1
            state["_stasis_rejects"] = int(state.get("_stasis_rejects", 0)) + 1
        return terminal

    def stats(self) -> dict:
        null = self.churn_null
        return {"ready": self.ready(), "reason": self.reason,
                "window": self.window, "churn_tol": self.churn_tol,
                "churn_null_n": int(null.size) if null is not None else 0,
                "churn_null_median": (float(np.median(null))
                                      if null is not None and null.size else None),
                "churn_null_p01": (float(np.quantile(null, 0.01))
                                   if null is not None and null.size else None),
                "churn_null_min": (int(null.min())
                                   if null is not None and null.size else None),
                "probe_steps": self.probe_steps, "branches": self.branches,
                "backoff": self.backoff, "last_drift": list(self.last_drift),
                "n_armed": self.n_armed, "n_confirmed": self.n_confirmed,
                "n_rejected": self.n_rejected, "n_probes": self.n_probes,
                "n_cached": self.n_cached, "convicted": len(self._convicted)}


# ===========================================================================
# Signal 4 -- coordinate reset + entity wipe
# ===========================================================================

COORD_RESET_ABS_MAX = 200     # "toward init": landed near a level-start x
COORD_RESET_DROP_MIN = 300    # must have dropped by at least this much
ENTITY_WIPE_MIN_BYTES = 8     # size of the contiguous zeroed RAM block
ENTITY_WIPE_TOL = 4           # "nonzero" before / "zero" after tolerance


def coord_entity_windows(ram_hist: np.ndarray, gx_series: np.ndarray,
                          window: int = 60, stride: int = 15
                          ) -> list[tuple[int, int]]:
    """A short window where the position readout drops sharply toward a
    level-start value AND a contiguous RAM block collapses to (near) zero
    -- the fingerprint of a fresh level/room load. gx_series comes from the
    same already-verified position readout the solver itself uses
    (go_explore_solve.SmbGame.progress); no new addresses are introduced
    here. The "entity wipe" block is discovered generically by scanning for
    the longest contiguous nonzero->zero run, not assumed at a fixed
    address."""
    t_total = ram_hist.shape[0]
    hits: list[tuple[int, int]] = []
    for start in range(0, max(t_total - window, 0) + 1, stride):
        end = min(start + window, t_total)
        if end - start < 4:
            continue
        gx_before = gx_series[start]
        gx_after = gx_series[end - 1]
        reset = (gx_after <= COORD_RESET_ABS_MAX and
                 (gx_before - gx_after) >= COORD_RESET_DROP_MIN)
        if not reset:
            continue
        before = ram_hist[start].astype(np.int16)
        after = ram_hist[end - 1].astype(np.int16)
        wiped = (before > ENTITY_WIPE_TOL) & (after <= ENTITY_WIPE_TOL)
        if _longest_true_run(wiped) >= ENTITY_WIPE_MIN_BYTES:
            hits.append((start, end))
    return hits


# ===========================================================================
# Signal 4b -- entity wipe, divorced from the position-reset precondition
# ===========================================================================
#
# coord_entity_windows above requires BOTH halves: a position reset toward a
# level-start value AND the RAM wipe. That joint requirement is SMB-shaped
# in a way the wipe half is not. Verified at the top level: on an odometer
# profile (progress: {source: odometer}) the position readout deliberately
# re-anchors WITHOUT integrating across a scene cut, so a real stage wipe
# never reads as gx dropping -- the reset half is structurally unreachable,
# not merely untuned. Worse, feeding coord_entity_windows a BACKWARD-WALK
# gx series (a player retreating toward the origin, e.g. 1200 -> 20) still
# satisfies the reset half, because "dropped by >= 300 and landed <= 200"
# describes retreat just as well as it describes a level-start teleport --
# see test_the_old_coord_did_fire_on_backward_walk. And on a single ram8
# byte progress readout the reset half can never fire at all: the required
# 300-unit drop does not fit in one byte (max range 0-255). A game-agnostic
# detector cannot carry a term whose sole surviving trigger on some profiles
# is the opposite of progress and whose trigger on others is arithmetically
# impossible, so the position half is deleted here rather than repaired.
#
# entity_wipe_windows is the wipe half of coord_entity_windows (historical
# lines 346-350 above), extracted verbatim -- same window/stride shape, same
# before/after single-frame comparison, same _longest_true_run threshold --
# and run with NO position readout of any kind: no gx_series parameter, no
# game-specific observable, nothing but the raw RAM history.
#
# CORROBORATION ONLY, NEVER A STANDALONE OR MAJORITY-CARRIER VOTE. Every
# documented false-positive class below is at least as strong a trigger as a
# real clear: a death wipes the entity table exactly like a stage load (the
# 2026-08-06 Gradius finding), a room transition wipes it (the Kirby
# 3-fires-in-24s finding), a wave of enemies despawning together wipes it
# (a combat blip), and an attract-loop restart wipes it. See
# tests/test_entity_wipe_signal.py, which asserts this signal DOES fire on
# every one of those shapes as a positive anti-vacuity control, not just on
# a genuine clear.

ENTITY_WIPE_EXCLUDE_DEFAULT: list[tuple[int, int]] = [(0x0100, 0x0200)]
# The CPU stack page. Its call/return depth oscillates constantly and
# manufactures nonzero->zero runs with no object-array semantics at all --
# an unexamined false-positive source in coord_entity_windows above, which
# scans the undifferentiated full 2 KiB. Excluded from the scan by default;
# a profile whose measured null shows no structure here may still widen
# `region` to cover it.


def _entity_wipe_mask(n_bytes: int, region: list[tuple[int, int]] | None,
                       exclude: list[tuple[int, int]] | None) -> np.ndarray:
    """Boolean per-byte-position allow-list for the wipe scan. `region`
    (default: the whole array) is the positive list of ranges to consider;
    `exclude` (default: the stack page) is then subtracted from it. A byte
    outside the mask reads as `False` in `wiped`, which both keeps it out of
    any run AND stops a run from crossing an excluded gap and welding two
    unrelated collapses into one contiguous one."""
    if region:
        mask = np.zeros(n_bytes, dtype=bool)
        for lo, hi in region:
            mask[lo:hi] = True
    else:
        mask = np.ones(n_bytes, dtype=bool)
    for lo, hi in (exclude or ()):
        mask[lo:hi] = False
    return mask


def entity_wipe_windows(ram_hist: np.ndarray, window: int = 60, stride: int = 15,
                         region: list[tuple[int, int]] | None = None,
                         exclude: list[tuple[int, int]] | None = None,
                         min_bytes: int = ENTITY_WIPE_MIN_BYTES,
                         tol: int = ENTITY_WIPE_TOL) -> list[tuple[int, int]]:
    """A short window where a contiguous RAM block collapses from occupied
    to (near) empty -- the object-array half of coord_entity_windows, with
    NO position precondition of any kind. The run is discovered generically
    by scanning for the longest contiguous occupied->empty span inside the
    allowed region; no address is assumed to mean anything.

    `min_bytes` and `tol` MUST come from a measured per-profile null (see
    scripts/clear_calibrate.py): the null is driven from the profile's own
    start state under ordinary play, and min_bytes is set above the 99.9th
    percentile of the observed longest-wipe-run distribution. A profile
    whose null has no separable upper tail should declare this signal
    `enabled: false` rather than pick a number."""
    if exclude is None:
        exclude = ENTITY_WIPE_EXCLUDE_DEFAULT
    t_total, n_bytes = ram_hist.shape
    include = _entity_wipe_mask(n_bytes, region, exclude)
    hits: list[tuple[int, int]] = []
    for start in range(0, max(t_total - window, 0) + 1, stride):
        end = min(start + window, t_total)
        if end - start < 4:
            continue
        before = ram_hist[start].astype(np.int16)
        after = ram_hist[end - 1].astype(np.int16)
        wiped = (before > tol) & (after <= tol) & include
        if _longest_true_run(wiped) >= min_bytes:
            hits.append((start, end))
    return hits


def trailing_median(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing k-sample median of a 1-D series (k <= 1 returns it unchanged).

    Edge-preserving by construction: a step change survives it (delayed by
    k//2 samples), an impulse shorter than k/2 samples does not. That is
    exactly the discrimination the coord signal needs and cannot make on its
    own, since it compares the FIRST and LAST sample of a sub-window and a
    2-sample spike parked on a sub-window boundary reads identically to a
    sustained level.

    THE LEADING EDGE (fixed 2026-08-26; measured, see below). The first k-1
    positions have no k-sample history, so something has to stand in for it.
    Back-filling with x[0] -- the shipped behavior -- replicates the OLDEST
    SAMPLE k-1 times, which makes it the majority of every one of the first
    ~k/2 windows. When that oldest sample is itself the impulse, the filter
    does not merely fail to remove it, it AMPLIFIES it into a synthetic run,
    and the run gets LONGER as k goes up: on [846, 846, 88, 88, ...] at k=15
    the first nine outputs are all 846. Turning the suppression knob up made
    the suppression worse.

    That is not theoretical. StreamingConfluenceDetector median-filters a
    ROLLING window, so every sample eventually becomes x[0]; a check lands on
    a spike sitting at relative index 0 whenever the spike's absolute index is
    a multiple of the check stride. Measured on the Double Dragon blip stream
    (tests/test_confluence_v2.py) with progress_median=5: a spike at index 300
    fabricates a clear at step 539, while the same spike at 301 or 305 is
    silent. The "alignment-independent" claim in the docstrings below was
    false at exactly one alignment -- the window head.

    NOTE FOR ANYONE "FIXING" THIS BY DELETING THE PAD: that does nothing.
    A growing trailing window (out[i] = median(x[:i+1]) for i < k-1) still
    yields out[0] == x[0] -- one real sample decides the position no matter
    how you dress it -- and reproduces the same fabricated clear at index 539.
    Both were run against the detector; they are indistinguishable there. The
    seed has to come from a FULL k-sample window, so no output position is
    ever decided by fewer than k real samples. Cost of the fix, measured on
    the same harness: a genuine sustained load is still detected at the same
    step (219), so this is suppression bought with no recall.

    The look-ahead this introduces (out[0] depends on x[0:k]) is safe here
    because this is an offline filter over a buffer the caller already holds
    in full, not a causal online filter -- push() re-filters the whole rolling
    window at every check."""
    if k <= 1 or x.size == 0:
        return x
    k = min(int(k), int(x.size))
    seed = float(np.median(x[:k]))
    pad = np.concatenate([np.full(k - 1, seed, dtype=np.float64),
                          np.asarray(x, dtype=np.float64)])
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    return np.median(win, axis=1)


# ===========================================================================
# Signal 5 -- APU channel-activity change (opt-in)
# ===========================================================================

N_APU_BITS = 5                  # $4015 low 5 bits: pulse1 pulse2 tri noise DMC
APU_SHORT_WINDOW = 30           # observations in the "now" rate estimate
APU_BASELINE_WINDOW = 300       # observations in the self-measured null
APU_MIN_BASELINE = 60           # null samples required before the vote exists
APU_SUSTAIN = 4                 # consecutive positive CUSUM evaluations to fire
APU_GATE_K = 3.0                # multiples of the null's own sampling noise
APU_GATE_FLOOR = 0.10           # floor on the gate, in activity-mass units
# Observations the vote stays raised once fired. Sized as 2*short_window,
# which spans three checks at the streaming detector's default stride of 20 --
# long enough that a fire and a RAM signature a check or two apart still
# coincide, short enough that the vote is not simply up all the time. NOTE the
# unit is OBSERVATIONS, not frames: an observation is a raw frame offline and
# one action (frame_skip frames) in the solver's hot loop, so the same number
# is ~1 s offline and ~4 s live at frame_skip 4.
APU_HOLD = 60

# mask -> bit vector, precomputed: the hot loop pushes one of 32 values.
_APU_BIT_TABLE = np.array(
    [[(m >> b) & 1 for b in range(N_APU_BITS)] for m in range(1 << N_APU_BITS)],
    dtype=np.int32)


class ApuActivitySignal:
    """Sustained, coordinated change in the per-channel APU activity vector,
    scored against a null this run measured from its own history.

    INPUT is the 5-bit $4015 length-counter mask (nes_core:
    `env.apu_channel_activity()` / `pool.apu_activity_all()`), one value per
    observation. Bit b is set while channel b's length counter (DMC: bytes
    remaining) is non-zero -- i.e. while that channel is *doing something*.
    Nothing here decodes pitch, volume or timbre, and nothing here knows what
    any particular game's music sounds like.

    CONTENT-FREE BY CONSTRUCTION. The statistic is:

        short_rate[b] = fraction of the last `short_window` observations in
                        which channel b was active
        base_rate[b]  = the same fraction over the `baseline_window`
                        observations that PRECEDE the short window (the null:
                        how this game's channels have been behaving lately)
        dev           = sum_b |short_rate[b] - base_rate[b]| / N_APU_BITS

    `dev` is the fraction of this game's own channel-activity mass that moved,
    in [0, 1]. There is no reference fingerprint, no expected jingle, no
    channel singled out as "the melody" -- swap the ROM and the null swaps
    with it.

    THE GATE IS ALSO SELF-MEASURED. Under the null, short_rate[b] is a mean of
    `short_window` Bernoulli(base_rate[b]) draws, so pure sampling noise
    already produces
        null_sigma = sum_b sqrt(base_rate[b](1-base_rate[b]) / short_window)
                     / N_APU_BITS
    of deviation. The gate is `gate_k * null_sigma + gate_floor`: a game whose
    channels flicker constantly gets a proportionally higher bar, and a game
    with dead-steady channels is held to the floor instead of firing on
    nothing. A CUSUM over (dev - gate) must then stay positive for `sustain`
    consecutive evaluations -- the same "sustained, not a blip" shape the
    audio signal uses, on the 5-bit vector instead of the FFT fingerprint.

    WHY A SHORT BLIP CANNOT FIRE, AT ANY ALIGNMENT (the property the tests
    pin). A transient spanning L observations can flip at most all
    N_APU_BITS bits for those L observations, so it can move short_rate by at
    most L/short_window per bit, hence

        dev <= L / short_window                     (alignment-independent)

    Firing needs dev > gate >= gate_floor, so it needs

        L > gate_floor * short_window               (= 3 observations at the
                                                     defaults)

    A one-observation blip -- every channel slamming on or off for a single
    frame, a drum hit, a jump SFX -- is structurally incapable of firing this
    signal no matter where in the window it lands. That is the same kind of
    guarantee `trailing_median` gives the coord signal, and it is why this is
    an honest vote rather than another alignment lottery.

    COORDINATION falls out of the sum: L observations of a c-channel change
    give dev = L*c / (short_window * N_APU_BITS), so channels moving TOGETHER
    reach the gate in proportionally fewer observations than any one channel
    toggling alone (one channel alone must change its duty over more than half
    the short window to clear the default floor).

    Deliberately NOT modelled: which direction the change went (music starting
    and music stopping are the same event to this signal), and which channel
    moved. Both would be content priors.

    The vote is a HELD PULSE, not a latch: 1 for `hold` observations after a
    fire, then 0, and the signal re-arms once it has returned to its own null
    (see _may_trigger for the measurement that forced this)."""

    def __init__(self, short_window: int = APU_SHORT_WINDOW,
                 baseline_window: int = APU_BASELINE_WINDOW,
                 min_baseline: int = APU_MIN_BASELINE,
                 sustain: int = APU_SUSTAIN,
                 gate_k: float = APU_GATE_K,
                 gate_floor: float = APU_GATE_FLOOR,
                 hold: int = APU_HOLD):
        self.short_window = max(1, int(short_window))
        self.baseline_window = max(1, int(baseline_window))
        self.min_baseline = max(1, int(min_baseline))
        self.sustain = max(1, int(sustain))
        self.gate_k = float(gate_k)
        self.gate_floor = float(gate_floor)
        self.hold = max(1, int(hold))
        self.reset()

    def reset(self) -> None:
        """Drop the latch AND the calibration.

        Same reasoning as StreamingConfluenceDetector.reset(): the samples
        that produced a fire are still inside both windows, so keeping them
        would let the identical fire land again the moment a veto expires.
        Re-earning the null costs `min_baseline` fresh observations, which is
        the precision-over-recall direction the rest of this detector takes."""
        self._short: deque = deque()
        self._long: deque = deque()
        self._short_sum = np.zeros(N_APU_BITS, dtype=np.int64)
        self._long_sum = np.zeros(N_APU_BITS, dtype=np.int64)
        self._cusum = 0.0
        self._consec = 0
        self._rearm_ready = True
        self.n = 0
        self.trigger_n: int | None = None
        self.n_triggers = 0
        self.last_dev = 0.0
        self.last_gate = 0.0
        self.n_evals = 0

    def push(self, mask) -> None:
        """Feed one 5-bit activity mask (int, or anything int()-able).

        O(1): the short window's evictions become the baseline's arrivals, so
        neither window is ever rescanned."""
        if mask is None:
            return
        v = _APU_BIT_TABLE[int(mask) & ((1 << N_APU_BITS) - 1)]
        self.n += 1
        self._short.append(v)
        self._short_sum += v
        if len(self._short) > self.short_window:
            old = self._short.popleft()
            self._short_sum -= old
            self._long.append(old)
            self._long_sum += old
            if len(self._long) > self.baseline_window:
                self._long_sum -= self._long.popleft()
        self._evaluate()

    def _evaluate(self) -> None:
        if (len(self._short) < self.short_window
                or len(self._long) < self.min_baseline):
            return
        self.n_evals += 1
        short_rate = self._short_sum / float(self.short_window)
        base_rate = self._long_sum / float(len(self._long))
        dev = float(np.abs(short_rate - base_rate).sum()) / N_APU_BITS
        null_sigma = float(np.sqrt(base_rate * (1.0 - base_rate)
                                    / self.short_window).sum()) / N_APU_BITS
        gate = self.gate_k * null_sigma + self.gate_floor
        self.last_dev, self.last_gate = dev, gate
        # CAPPED CUSUM. Uncapped, a change that persists for hundreds of
        # observations accumulates hundreds of gate-widths of credit and then
        # takes just as many observations of perfectly null behavior to bleed
        # back down -- the statistic saturates and the signal is stuck
        # "changed" long after the game has settled into its new normal
        # (found by running this over real Contra play). Capping at `sustain`
        # gate-widths keeps exactly as much history as the sustain rule needs
        # and no more.
        cap = self.sustain * max(gate, 1e-6)
        self._cusum = min(max(0.0, self._cusum + (dev - gate)), cap)
        self._consec = self._consec + 1 if self._cusum > 0.0 else 0
        if self._cusum <= 0.0:
            self._rearm_ready = True
        if self._consec >= self.sustain and self._may_trigger():
            self.trigger_n = self.n
            self.n_triggers += 1
            self._rearm_ready = False
            # Restart the accumulator: the change has been declared, so the
            # NEXT declaration must be earned from scratch.
            self._cusum = 0.0
            self._consec = 0

    def _may_trigger(self) -> bool:
        """A one-shot latch is useless to a live burst.

        Measured on real play (Contra, 1,500 observations from power-on): the
        FIRST sustained coordinated change after the null is measured is the
        title music starting -- an entirely genuine change, fired at
        observation 93. With a one-shot trigger the signal would then have
        spent its only vote on the intro and been blind for the rest of the
        burst, which is exactly the window a clear happens in.

        So the signal re-arms, but only after it has (a) let its hold expire
        and (b) come all the way back to its own null (CUSUM decayed to 0).
        Requiring the return to null is what stops one long change from being
        re-declared every `hold` observations: the deviation has to fall back
        under this game's own gate before another change can be announced."""
        return (self.trigger_n is None
                or (self.n - self.trigger_n >= self.hold
                    and self._rearm_ready))

    def vote(self) -> int:
        """1 while the fire is inside its hold window, else 0."""
        if self.trigger_n is None:
            return 0
        return int(self.n - self.trigger_n < self.hold)

    def warmup_observations(self) -> int:
        """Observations a FRESH signal must be fed before vote() can be 1 --
        the point below which this signal has no opinion at all, as opposed
        to an opinion of 'no'.

        Derived from the same three numbers _evaluate enforces, so the two
        cannot drift: an evaluation happens only once the short window is
        full AND the null holds `min_baseline` samples, and because the null
        is fed exclusively by the short window's evictions that is
        `short_window + min_baseline` pushes. `sustain` consecutive positive
        evaluations must then follow, the earliest of which lands
        `sustain - 1` observations after the first.

        A LOWER BOUND, not a prediction: it says when the vote becomes
        possible, never that it will happen. Callers use it to tell a replay
        that was too short to look apart from one that looked and disagreed
        -- a distinction that is invisible in the output otherwise, because
        both read as vote() == 0 (the honest, strict direction for a live
        detector, and a silent no-op for a fixed-length replay harness; see
        Solver.counterfactual_probe in go_explore_solve.py, which measured
        exactly that)."""
        return self.short_window + self.min_baseline + self.sustain - 1

    def stats(self) -> dict:
        return {"n": self.n, "n_evals": self.n_evals,
                "trigger_n": self.trigger_n, "n_triggers": self.n_triggers,
                "dev": round(self.last_dev, 5),
                "gate": round(self.last_gate, 5),
                "baseline_n": len(self._long)}


# ===========================================================================
# Signal -- OAM sprite-population quiescence (opt-in)
#
# entity_wipe_windows (Signal 4b, above) finds its "occupied -> empty" run by
# scanning undifferentiated CPU RAM for a byte-array convention that is a
# per-game accident -- which addresses hold entity slots, and whether they
# zero on death, varies game to game (entity_wipe's own docstring names the
# stack page $0100-$01FF as one unexamined false-positive source of scanning
# RAM blind). OAM has no such ambiguity: which 256 bytes are "the sprite
# table" and which sprites are hidden are NES PPU HARDWARE FACTS our own
# emulator implements (ppu.rs:990 is_sprite_at_y_on_scanline), not a
# convention any particular game chose. A game whose entities live in an
# unusual RAM region still shows the collapse here, because every game's
# entities that render at all render through OAM.
# ===========================================================================

OAM_SPRITES = 64          # primary OAM: 64 sprites x 4 bytes
OAM_ENTRY_BYTES = 4       # (y, tile, attr, x) per sprite -- pool.rs/python.rs peek_oam layout
OAM_HIDE_Y = 240          # ppu.rs:990 is_sprite_at_y_on_scanline: y <= scanline < y+height,
                          # scanline in 0..239 (rendered lines) => y >= 240 can NEVER match any
                          # scanline, i.e. the sprite can never be evaluated onto the picture.
                          # This is our own PPU's hide rule, not a game convention.

OAM_SHORT_WINDOW = APU_SHORT_WINDOW        # reused verbatim: same null shape, same proof carries over
OAM_BASELINE_WINDOW = APU_BASELINE_WINDOW
OAM_MIN_BASELINE = APU_MIN_BASELINE
OAM_SUSTAIN = APU_SUSTAIN
OAM_HOLD = APU_HOLD
OAM_COLLAPSE = 0.2        # fire once the short-window population rate falls under this FRACTION
                          # of the game's own measured baseline rate (i.e. an 80% drop)


def oam_census(oam) -> tuple[int, int]:
    """(n_visible, n_distinct_tile_attr) from one raw 256-byte primary OAM
    snapshot (`env.peek_oam()` / `pool.peek_oam(wid)`).

    n_visible counts sprite slots i in 0..63 with oam[4*i] < OAM_HIDE_Y --
    exactly the renderable predicate our own PPU evaluates, nothing a game
    author chose. n_distinct is the number of distinct (tile, attribute)
    byte pairs among only the VISIBLE sprites -- a hidden slot's leftover
    tile/attr bytes (commonly a stale, never-cleared value) must not
    contribute, or a game that parks 60 dead sprites at y=0xFF with 60
    different stale tile bytes would read as a huge, entirely fictitious
    population of distinct objects.

    Purity: reads only the OAM byte layout and the y>=240 hide rule, both
    NES/our-PPU hardware facts already implemented in ppu.rs -- no game is
    consulted, no address is assumed to mean anything beyond "a sprite
    slot"."""
    buf = (np.frombuffer(oam, dtype=np.uint8)
           if isinstance(oam, (bytes, bytearray)) else
           np.asarray(oam, dtype=np.uint8))
    if buf.size < OAM_SPRITES * OAM_ENTRY_BYTES:
        return 0, 0
    entries = buf[: OAM_SPRITES * OAM_ENTRY_BYTES].reshape(OAM_SPRITES, OAM_ENTRY_BYTES)
    visible = entries[:, 0] < OAM_HIDE_Y
    n_visible = int(visible.sum())
    if n_visible == 0:
        return 0, 0
    pairs = set(zip(entries[visible, 1].tolist(), entries[visible, 2].tolist()))
    return n_visible, len(pairs)


class OamQuiesceSignal:
    """Sustained collapse of the visible-sprite population, scored against a
    null this run measured from its own history -- the OAM-hardware sibling
    of ApuActivitySignal, whose self-calibration shape (a short window
    against a preceding rolling baseline, a floor-style gate, a
    hold-then-rearm latch) this class reuses verbatim rather than
    re-deriving.

    INPUT is one `(n_visible, n_distinct)` pair per observation (from
    `oam_census()` above, or a batched Rust `oam_census_all()` accessor a
    live caller should add rather than calling `peek_oam()` per worker per
    step -- see the design note in
    docs/research/CLEAR_DETECTION_CAMPAIGN_2026-08-26.md). Each of the two
    counts is normalized to a rate in [0, 1] by dividing by OAM_SPRITES,
    exactly as ApuActivitySignal normalizes a bit to an active/inactive
    rate; the two rates are then tracked as a 2-wide vector in the same
    short-window / baseline-window construction ApuActivitySignal runs over
    its 5-wide bit vector.

    CONTENT-FREE BY CONSTRUCTION, ONE-SIDED BY DESIGN. Unlike
    ApuActivitySignal (deliberately direction-agnostic -- a fanfare starting
    and the level music cutting are the same event to it), this signal is
    deliberately direction-SPECIFIC: only a DROP counts. A sprite population
    growing (more enemies on screen) is the opposite of a clear and must
    never contribute a particle of vote. So instead of Apu's symmetric
    `dev = sum |short - base|`, this drops the absolute value:

        drop[i] = max(0, base_rate[i] - short_rate[i])   i in {visible, distinct}
        dev     = (drop[visible] + drop[distinct]) / 2
        scale   = (base_rate[visible] + base_rate[distinct]) / 2   ("its own
                  baseline rate" -- the population this game's OWN recent
                  history says is normal)

    and fires once `dev > (1 - collapse) * scale`, held for `sustain`
    consecutive evaluations before it latches (no CUSUM accumulator here --
    "stays there" is a plain consecutive-count, simpler than the audio/APU
    change-point machinery because this is a level test, not a slope test).
    For a CLEAN collapse (both dims falling together, so neither `drop[i]`
    clamps to 0) this condition is algebraically identical to
    `short_rate_combined < collapse * scale`: the combined population rate
    has fallen under `collapse` x its own baseline rate -- the plain-English
    statement the `collapse` knob is calibrated against.

    WHY A SHORT COLLAPSE CANNOT FIRE, AT ANY ALIGNMENT (the property the
    tests pin, derived fresh rather than assumed). Every pushed rate lies in
    [0, 1], so for a transient of length L observations landing anywhere
    inside the short window, at most L of the short_window samples differ
    from the value the window would otherwise hold, hence for EACH dimension

        drop[i] <= (L / short_window) * base_rate[i]     (alignment-independent)

    (the transient can pull that dimension's short-window average down by at
    most base_rate[i] on each of its L observations, no more -- the same
    counting argument ApuActivitySignal's docstring uses for its bits, run
    over a continuous rate instead of a 0/1 one). Summing and halving gives
    `dev <= (L / short_window) * scale`, so firing (dev > (1-collapse)*scale,
    scale > 0) needs

        L > short_window * (1 - collapse)

    -- a bound that does not depend on the game's own baseline magnitude
    (`scale` cancels out of both sides), exactly the way ApuActivitySignal's
    `L > gate_floor * short_window` does not depend on which channels a
    particular game happens to use. At the defaults (short_window=30,
    collapse=0.2) that is L > 24: a 24-observation total sprite wipe cannot
    fire this signal at any alignment, wherever in the window it lands; one
    observation more, it does. A `scale` of 0 (a baseline that was ALREADY
    fully collapsed -- the game showed no sprites for the whole baseline
    window) makes the gate `(1-collapse)*0 = 0` unreachable by a drop from
    nothing, which is the correct, safe direction: there is no population
    left to collapse FROM.

    FALSE POSITIVES this signal cannot tell apart from a clear (documented,
    not discovered later): DEATH fires it hardest -- the player and the
    enemies that killed them explode off the sprite table together, and
    death is precisely a SUSTAINED wipe (a death/explosion animation holds
    the table empty for well over `sustain` evaluations), so this signal
    reads every death as a population collapse with total confidence.
    SCREEN BLANK / FADE hides every sprite (any fade, including a death fade
    and a pause menu's fade-to-black). ROOM TRANSITION fires it (the new
    room's entities have not spawned in yet). BOSS INTRO fires it (the arena
    clears before the boss appears). ATTRACT LOOP fires it at every demo
    restart. Because this signal's false-positive set is a strict superset
    of entity_wipe_windows' (both fire on every one of the above; OAM adds
    nothing entity_wipe's RAM heuristic didn't already cover -- it only
    makes the same collapse visible on a game whose entity RAM layout
    entity_wipe cannot see), a caller wiring both into a vote MUST treat
    {entity_wipe, oam_quiesce} as ONE corroborating slot, never as two
    independent votes -- counting them separately double-counts a single
    piece of evidence.

    AN UNEXAMINED RESIDUAL, measured while writing this class's tests, worth
    naming rather than leaving for someone else to rediscover: unlike
    ApuActivitySignal's gate (`gate_k * null_sigma + gate_floor`, which
    WIDENS proportionally to this game's own measured churn), this signal's
    gate is a FIXED fraction of the baseline's MAGNITUDE only -- it carries
    no term for how much that baseline naturally VARIES. A game whose sprite
    population legitimately oscillates on its own ordinary rhythm (a
    shmup's busy-wave / near-empty-lull cycle, measured here at a ~15:1
    amplitude with a lull lasting a little over the blip-immunity bound)
    crosses the collapse gate on EVERY lull, repeatedly, for as long as the
    game runs, with no clear ever happening
    (test_an_ordinary_oscillating_wave_rhythm_can_repeatedly_false_fire pins
    the measured shape). This is a SHARPER case than the transient
    death/fade/room-transition list above: those are one-off state changes;
    this is normal, REPEATING gameplay. It is a direct consequence of
    leaving the variance term out, not a bug in the arithmetic -- flagged
    here rather than shipped silently, so a caller with a game in this shape
    knows to raise `collapse` (a smaller required drop) or, better, to never
    let this signal stand without a transition-confirming corroborator
    (room_fp_transition, lock_release_novelty) in the same vote."""

    def __init__(self, collapse: float = OAM_COLLAPSE,
                 sustain: int = OAM_SUSTAIN,
                 short_window: int = OAM_SHORT_WINDOW,
                 baseline_window: int = OAM_BASELINE_WINDOW,
                 min_baseline: int = OAM_MIN_BASELINE,
                 hold: int = OAM_HOLD):
        self.collapse = float(collapse)
        self.sustain = max(1, int(sustain))
        self.short_window = max(1, int(short_window))
        self.baseline_window = max(1, int(baseline_window))
        self.min_baseline = max(1, int(min_baseline))
        self.hold = max(1, int(hold))
        self.reset()

    def reset(self) -> None:
        """Drop the latch AND the calibration -- same reasoning as
        ApuActivitySignal.reset(): the samples that produced a fire are
        still inside both windows, so keeping them would let the identical
        fire land again the moment a veto expires."""
        self._short: deque = deque()
        self._long: deque = deque()
        self._short_sum = np.zeros(2, dtype=np.float64)
        self._long_sum = np.zeros(2, dtype=np.float64)
        self._consec = 0
        self._rearm_ready = True
        self.n = 0
        self.trigger_n: int | None = None
        self.n_triggers = 0
        self.last_dev = 0.0
        self.last_scale = 0.0
        self.n_evals = 0

    def push(self, census) -> None:
        """Feed one `(n_visible, n_distinct)` pair. `census=None` -- a
        caller that has not plumbed the OAM modality yet -- is IGNORED
        rather than read as `(0, 0)`: treating an absent measurement as
        "every sprite vanished" would fabricate the loudest possible
        collapse out of nothing, exactly the failure
        test_a_missing_census_is_ignored_rather_than_read_as_a_collapse
        pins (the same concern ApuActivitySignal.push documents for a
        missing mask)."""
        if census is None:
            return
        n_visible, n_distinct = census
        v = np.array([n_visible, n_distinct], dtype=np.float64) / OAM_SPRITES
        self.n += 1
        self._short.append(v)
        self._short_sum += v
        if len(self._short) > self.short_window:
            old = self._short.popleft()
            self._short_sum -= old
            self._long.append(old)
            self._long_sum += old
            if len(self._long) > self.baseline_window:
                self._long_sum -= self._long.popleft()
        self._evaluate()

    def _evaluate(self) -> None:
        if (len(self._short) < self.short_window
                or len(self._long) < self.min_baseline):
            return
        self.n_evals += 1
        short_rate = self._short_sum / float(self.short_window)
        base_rate = self._long_sum / float(len(self._long))
        drop = np.maximum(0.0, base_rate - short_rate)
        dev = float(drop.sum()) / 2.0
        scale = float(base_rate.sum()) / 2.0
        self.last_dev, self.last_scale = dev, scale
        below = dev > (1.0 - self.collapse) * scale + 1e-12
        self._consec = self._consec + 1 if below else 0
        if not below:
            self._rearm_ready = True
        if self._consec >= self.sustain and self._may_trigger():
            self.trigger_n = self.n
            self.n_triggers += 1
            self._rearm_ready = False
            self._consec = 0

    def _may_trigger(self) -> bool:
        """Re-arm only after the hold has expired AND the population has
        actually recovered above the gate -- identical reasoning to
        ApuActivitySignal._may_trigger (a sustained collapse must not
        re-declare itself every `hold` observations for as long as it
        lasts)."""
        return (self.trigger_n is None
                or (self.n - self.trigger_n >= self.hold
                    and self._rearm_ready))

    def vote(self) -> int:
        """1 while the fire is inside its hold window, else 0."""
        if self.trigger_n is None:
            return 0
        return int(self.n - self.trigger_n < self.hold)

    def warmup_observations(self) -> int:
        """Observations a FRESH signal must be fed before vote() can be 1 --
        identical derivation to ApuActivitySignal.warmup_observations()."""
        return self.short_window + self.min_baseline + self.sustain - 1

    def stats(self) -> dict:
        return {"n": self.n, "n_evals": self.n_evals,
                "trigger_n": self.trigger_n, "n_triggers": self.n_triggers,
                "dev": round(self.last_dev, 5),
                "scale": round(self.last_scale, 5),
                "baseline_n": len(self._long)}


# ===========================================================================
# Signal 6 -- lock-release novelty (the one-way-door discriminator)
# ===========================================================================

class LockReleaseNoveltyTrack:
    """The composite that actually carries the clear verdict, built from
    ORDER and CONSEQUENCE rather than from any new byte -- the discriminator
    the census named ("a stage clear is a lock window that ends with the
    world irreversibly different; a death is a lock window that ends with
    the world restored") but never got to run, because the census never
    reached a working detector on 28 of its 29 profiles.

    Every OTHER signal in this file answers "did something change" (a music
    cue, a tally, a position reset, an entity wipe, a channel-activity
    shift). None of them can tell a stage clear from a death, because a
    death produces the exact same co-occurring changes -- which is the
    measured story behind every false positive on record here (Gradius'
    respawn wipe, Kirby's room transitions, the Galaga attract loop reading
    LOCKED forever). A clear and a death share their SIGNATURE but not
    their SHAPE:

      * a DEATH is a lock window that releases back into a fingerprint the
        episode has already produced -- the respawn/checkpoint room -- and
        the room you were standing in before the lock is one you can walk
        straight back into.
      * a CLEAR is a lock window that releases into a fingerprint the
        episode has never produced before, and the room you were standing
        in before the lock is gone: it does not reappear no matter how long
        you keep playing.

    INPUT is two already-computed per-observation values; this class is
    composition ONLY and consults no address and no new hardware surface of
    its own:

      locked   -- bool, this observation's input-lock verdict. In the live
                  loop this is InputLockSignal.probe()'s return value (or
                  InputLockTrack.vote_at's, offline); a test may synthesize
                  it directly. Purity holds transitively -- this class does
                  not care HOW `locked` was computed, only its rise/fall
                  pattern.
      room_fp  -- a hashable identity token for "which room/scene is
                  showing right now", SETTLED (not mid-transition), or None
                  when no settled identity exists yet for this observation
                  (mid-churn, or a caller that has not wired a room-identity
                  signal at all). The token's provenance does not matter --
                  a nametable-VRAM hash (room_fp_transition), a scene
                  ordinal (scene_cut), or a test's own synthetic int drive
                  IDENTICAL logic here, because the discriminating power is
                  in the token's RISE/FALL/RE-ENTRY pattern, not in what
                  produced it.

    MECHANISM. Track the current lock window's start (t0) and the last
    settled room_fp seen before it opened (the "pre-lock" fingerprint). On
    the falling edge of `locked` (t1), if (t1 - t0) <= lock_max:
      1. NOVELTY -- is the settled room_fp AT t1 one this scope has not
         produced before now? If not (a RESPAWN -- most commonly landing
         back in the exact pre-lock room, but any previously-visited room
         reads the same way), discard: no candidate armed, vote unchanged.
      2. If novel, arm a PENDING candidate holding the pre-lock fingerprint
         and start counting.
      3. RE-ENTRY CHECK -- if the pre-lock fingerprint is read again
         (settled) at any point during the count, the candidate was a
         one-way door WITHIN a level that the player walked back around
         from some other route, not a level exit: discard.
      4. If the count reaches `m` observations without the pre-lock
         fingerprint reappearing, the candidate survives and FIRES. The
         vote is a LATCH (mirrors StreamingConfluenceDetector.push's "stays
         True thereafter" contract) -- a clear is a discrete banked event,
         not a transient blip that un-fires.

    WHY THE `m`-OBSERVATION DELAY IS CORRECT, NOT A BUG. Firing needs `m`
    observations of CONFIRMED non-re-entry before it can be declared,
    because that confirmation is the one piece of evidence this surface can
    gather for free that distinguishes a level exit from an ordinary
    walk-around loop. Collapsing it to "fire the instant room_fp is novel"
    would not create a new false positive (novelty ALONE already votes 1 on
    an ordinary one-way door -- see the Kirby control below) but it WOULD
    throw away the one corroborating check available, for no latency
    benefit worth naming.

    THIS SIGNAL CANNOT, BY ITSELF, TELL A CLEAR FROM AN OTHERWISE-ORDINARY
    ONE-WAY DOOR STILL FIRMLY INSIDE THE LEVEL (Kirby's doors, Metroid's
    rooms). That is not a defect to patch here -- it is the reason
    `progress_advance` remains a REQUIRED corroborator wherever this signal
    is declared on a room-based game (the generalized room_veto escape
    clause), and it is why the false-positive control below is written as a
    POSITIVE assertion instead of being quietly avoided.

    FALSE POSITIVES:
      * A ONE-WAY DOOR INSIDE A LEVEL is genuinely novel and genuinely
        non-re-entrant at this surface, and this signal WILL vote 1 on it --
        proved as a positive assertion by
        test_kirby_room_shape_still_votes_on_lock_release_novelty, not
        hidden as an accidental gap. `progress_advance` is what tells the
        two apart, one layer up.
      * A DEATH THAT RESPAWNS INTO A NEVER-VISITED ROOM defeats the
        first-visit clause (rare, but real on a game whose respawn point is
        not fixed) -- pinned as a positive assertion too; only the
        `lives_drop` veto one layer up covers it.
      * A COMBAT BLIP that locks input briefly (hit-stun) and releases back
        into the SAME room is respawn-shaped (post-lock fp == pre-lock fp)
        and is rejected by the novelty check exactly like a death, with no
        special-casing needed.
      * ATTRACT LOOP -- an unbounded lock never releases, so no falling
        edge is ever observed and no candidate is ever armed: caught by
        construction. A loop that DOES cycle but holds each lock far longer
        than any real clear's window is caught by `lock_max` rejecting the
        candidate outright.
      * GAME OVER -- same as the unbounded attract case: a lock that never
        releases arms nothing.

    CALIBRATION:
      lock_max     -- upper bound on a clear's lock window, in observations.
                      REQUIRED, no default: a game-agnostic default here
                      would repeat the exact mistake this campaign's own
                      structural finding names (COORD_RESET_DROP_MIN,
                      LOCK_DIFF_TOL -- SMB-shaped constants with no
                      per-game meaning). Measure it per profile from that
                      profile's own oracle clear trace(s) -- the lock
                      window's measured duration across a real recorded
                      clear (SMB's flagpole-to-next-level, Castlevania's
                      block transition) -- and pass it in.
      m            -- the re-entry horizon, in observations, after release.
                      Also REQUIRED and also per-profile: too short and a
                      slow walk back into the pre-lock room reads as a
                      clear before the walk completes; too long and a
                      genuine clear's vote is delayed past the point a
                      caller needed it.
      per_episode  -- True (default): `seen` starts empty at construction
                      and accumulates only what THIS episode has settled
                      on -- the honest default per the design doc (a
                      lineage restored from an archived cell has not
                      "visited" anything yet in the sense this signal
                      means). For the per-archive variant, construct with
                      `per_episode=False` and pass `seen` pre-populated
                      with a lineage's ancestor visits; `reset()` then
                      preserves it across episode boundaries instead of
                      wiping it.
      seen         -- optional pre-seeded set of already-visited
                      fingerprints (the per-archive variant); leave None
                      for a fresh per-episode set.

    NOT MODELLED, deliberately: WHICH room the player ends up in (any novel
    room is treated identically), and how long the pre-lock room stays
    unvisited AFTER the `m`-observation window closes (a level that loops
    back to an early room after `m` more observations is not re-litigated;
    `m` is a confirmation horizon, not a permanent ban)."""

    def __init__(self, lock_max: int, m: int, per_episode: bool = True,
                 seen: set | None = None):
        self.lock_max = max(1, int(lock_max))
        self.m = max(1, int(m))
        self.per_episode = bool(per_episode)
        self._seen: set = set() if seen is None else seen
        self.reset()

    def reset(self) -> None:
        """Un-latch AND drop the in-flight lock/pending state (same
        reasoning as StreamingConfluenceDetector.reset(): the evidence a
        fire used is still in the pending/seen state, so a caller that
        vetoes a fire and does not reset would re-earn the identical fire
        off stale evidence the instant the veto lifted).

        Deliberately does NOT clear `_seen` when `per_episode` is False --
        the whole point of the per-archive variant is that visited-room
        history survives a reset. A fresh per-episode instance is expected
        to be reconstructed, not reset, at episode boundaries; reset() here
        exists for the latch/pending state regardless of scope."""
        self._locked = False
        self._lock_start: int | None = None
        self._pre_fp = None
        self._pending: dict | None = None
        self._last_fp = None
        self._vote = 0
        self.n = 0
        self.n_candidates = 0
        self.n_respawn_shaped = 0
        self.n_fires = 0
        if self.per_episode:
            self._seen = set()

    def push(self, locked: bool, room_fp=None) -> int:
        """Feed one observation's (locked, room_fp) pair. Returns vote().

        Order matters and is deliberate: a pending candidate's re-entry
        check runs BEFORE this observation's room_fp is folded into `seen`
        or compared for THIS frame's own edge, so nothing can satisfy its
        own novelty test and no re-entry check can be short-circuited by
        the very frame it is checking."""
        locked = bool(locked)
        self.n += 1

        if self._pending is not None:
            if room_fp is not None and room_fp == self._pending["pre_fp"]:
                self._pending = None            # walked straight back in
            else:
                self._pending["age"] += 1
                if self._pending["age"] >= self.m:
                    self.n_fires += 1
                    self._vote = 1
                    self._pending = None

        if locked and not self._locked:
            self._lock_start = self.n
            self._pre_fp = self._last_fp
        elif self._locked and not locked and self._lock_start is not None:
            dur = self.n - self._lock_start
            if dur <= self.lock_max:
                self.n_candidates += 1
                post_fp = room_fp if room_fp is not None else self._last_fp
                if post_fp is not None and post_fp not in self._seen:
                    self._pending = {"pre_fp": self._pre_fp, "age": 0}
                else:
                    self.n_respawn_shaped += 1
            self._lock_start = None

        if room_fp is not None:
            self._last_fp = room_fp
            self._seen.add(room_fp)

        self._locked = locked
        return self.vote()

    def vote(self) -> int:
        return self._vote

    def stats(self) -> dict:
        return {"n": self.n, "n_candidates": self.n_candidates,
                "n_respawn_shaped": self.n_respawn_shaped,
                "n_fires": self.n_fires, "n_seen": len(self._seen)}


# ===========================================================================
# Signal -- room-fingerprint transition (opt-in; novel-room discovery)
# ===========================================================================
#
# The ROOMGRAPH_ENGINE_2026-08-24 identity layer (masked NT-hash -> settle ->
# classify -> intern) already exists and is already receipted -- it drives
# cells and door macros in go_explore_solve.py today, and the clear hook
# simply cannot see it. Nothing below is new hashing or classification
# logic; every pure step is imported verbatim (room_fp_mask, nt_fingerprint,
# fp_settle, classify_transition, RoomIndex). This is the thin, stateful
# wrapper that feeds those functions one observation at a time and turns
# "a new room identity settled" into a vote.

#: `classify_transition`'s three possible kinds. `warp` is excluded from the
#: default vote set: RG-2 already refuses to mint a warp-classified settle as
#: adjacency (it is a scripted, largely input-independent identity change --
#: the measured Zelda death-flash signature, odometer flat / scene +2), and
#: the clear vote must inherit that refusal rather than re-derive it. A
#: profile with a game-specific reason to trust warp settles may opt back in
#: explicitly via `kind`.
ROOM_FP_KINDS = ("pan", "fade", "warp")
ROOM_FP_DEFAULT_KIND = ("pan", "fade")
#: Observations the vote stays raised after a qualifying settle -- the same
#: held-pulse shape ApuActivitySignal uses (see its `hold`), so a
#: corroborating signal arriving a few observations later still overlaps
#: this vote instead of needing to land on the exact settle step.
ROOM_FP_HOLD = 60


class RoomFpTransitionSignal:
    """A SETTLED change of room identity: the masked blake2b-64 hash of
    nametable VRAM held a new value for `settle` consecutive samples, the
    churn window's (d_odo, d_scene) classified, and the resulting identity
    interned to a discovery ordinal. Votes when the settled identity is one
    NOT previously seen by this instance (`novel_only`) and its classified
    kind is one this instance was told to trust (`kind`).

    PURITY. The surface is 2 KB physical nametable VRAM (`Pool.
    peek_nametables`), optionally co-keyed with the 32-byte palette RAM
    (`Pool.palette_ram`) -- both hardware surfaces of the same class as
    pixels and OAM. No address is named by meaning anywhere in this class:
    the mask that reaches `nt_fingerprint` is an opaque KEEP/DROP array a
    caller measured from ITS OWN idle/walk frames (scripts/
    room_fp_calibrate.py), not an authored map of what a byte means.

    THE FIRST SETTLE IS BASELINE, NOT A TRANSITION. Exactly like the live
    solver's own worker-seed adoption (`_room_transit`'s "adoption !=
    transit" invariant; tests/test_room_fp.py
    test_root_first_settle_adopts_transit_free_and_edge_free), the very
    first fingerprint this instance ever settles on has no prior room to
    have transitioned FROM. It is interned quietly to seed identity #0 and
    is NEVER classified or voted on -- a fresh detector dropped into the
    middle of an ordinary room must not fire on the act of noticing that
    room for the first time.

    NOVELTY is scoped to THIS INSTANCE's own discovery table, and `reset()`
    rebuilds that table from empty -- i.e. per-episode by construction (the
    honest default: a lineage restored from an archived cell has visited
    nothing as far as a fresh instance is concerned). A caller that wants
    per-run (not per-episode) novelty simply never calls reset() mid-run.

    KIND IS A VOTE FILTER, NOT A DISCOVERY FILTER: identity is interned (and
    can seed later novelty checks) for EVERY settle regardless of its
    classified kind, but the VOTE only considers settles whose kind is in
    `kind` (default {pan, fade} -- see ROOM_FP_KINDS above for why warp is
    excluded by default).

    FALSE POSITIVES THIS SIGNAL CANNOT DISCRIMINATE ON ITS OWN. An ordinary
    ROOM TRANSITION is not a false positive of this signal, it is its
    TARGET -- the measured Kirby result (3 fires in 24 s, every one an
    ordinary area 58->62 room change) is exactly what `novel_only` exists to
    tame, and even then only within ONE episode: a room-based game's first
    visit to every one of its rooms is, correctly, novel. This signal's role
    in a larger vote is therefore "something loaded", never "the level
    ended" -- it must never be a standalone or majority-carrier vote, the
    same discipline entity_wipe/oam_quiesce are held to, and a room-based
    profile needs a progress corroborator (progress_advance) to turn
    "novel room" into "level ended". A DEATH FADE settles a new fingerprint
    too (the measured Zelda death flash) but classifies `warp`, excluded by
    default. PAUSE/MENU overlays rewrite the nametable and are mitigated by
    the mask, not eliminated by this class.
    """

    def __init__(self, mask_ranges=(), *, settle: int = 3,
                 min_lines: int = 200, pan_odo=(128, 384),
                 warp_scene_min: int = 2, palette_cokey: bool = False,
                 max_rooms: int = 1024, kind=ROOM_FP_DEFAULT_KIND,
                 novel_only: bool = True, hold: int = ROOM_FP_HOLD):
        bad_kind = set(kind) - set(ROOM_FP_KINDS)
        if bad_kind:
            raise ValueError(
                f"room_fp_transition kind must be a subset of "
                f"{ROOM_FP_KINDS}, got {sorted(bad_kind)}")
        self.mask = room_fp_mask(mask_ranges)
        self.settle = max(1, int(settle))
        self.min_lines = int(min_lines)
        self.pan_odo = (int(pan_odo[0]), int(pan_odo[1]))
        self.warp_scene_min = int(warp_scene_min)
        self.palette_cokey = bool(palette_cokey)
        self.max_rooms = max(1, int(max_rooms))
        self.kind = frozenset(kind)
        self.novel_only = bool(novel_only)
        self.hold = max(1, int(hold))
        self.reset()

    def reset(self) -> None:
        """Drop the settle machinery AND the discovery table.

        A fresh episode has visited nothing -- the same "re-earn it"
        direction every other signal's reset() takes here (see
        ApuActivitySignal.reset()): keeping the table would let a room
        visited just before a veto/episode boundary read as "not novel"
        the instant the new episode starts."""
        self._pend = None
        self._settled_h: int | None = None
        self._rooms = RoomIndex(cap=self.max_rooms)
        self._step = 0
        self.trigger_step: int | None = None
        self.n_triggers = 0
        self.n_settles = 0
        self.last_kind: str | None = None
        self.last_direction = None
        self.last_novel: bool | None = None
        self.last_ordinal: int | None = None

    def push(self, nt, odo_xy=(0, 0), scene: int = 0,
             rendered_lines: int | None = None, palette=None) -> None:
        """Feed one observation: the nametable snapshot, the odometer's
        integrated (x, y) and the scene ordinal (both already produced by
        the certified odometer for every profile that arms room_fp), and
        optionally the rendered-scanline count and the 32-byte palette RAM.

        `rendered_lines`, when supplied, gates a BLANK frame (stage wipe /
        fade-to-black / load screen) exactly the way the live hot loop does:
        a blank cancels any pending churn instead of letting `fp_settle`
        adopt "the screen is black" as a room of its own (`fp_settle`'s own
        docstring: "Blank frames never reach here"). A caller that never
        observes a blank simply never passes it, and this gate is then
        permanently open (matches every non-blank push)."""
        self._step += 1
        if rendered_lines is not None and rendered_lines < self.min_lines:
            self._pend = None
            return
        h = nt_fingerprint(nt, self.mask,
                            palette if self.palette_cokey else None)
        was_baseline = self._settled_h is None
        self._pend, fired = fp_settle(self._pend, h, self._settled_h,
                                      odo_xy, scene, self._step, self.settle)
        if fired is None:
            return
        h_settled, d_odo, d_scene, _frames = fired
        self._settled_h = h_settled
        self.n_settles += 1
        ordinal = self._rooms.intern(h_settled, odo_xy)
        novel = ordinal is not None and self._rooms.meta[ordinal]["visits"] == 1
        if was_baseline:
            # Adoption, not a transition: seed identity and stop -- the same
            # transit-free/edge-free rule the live worker-seed path enforces.
            # No kind, no novelty check, no vote.
            self.last_kind = None
            self.last_direction = None
            self.last_novel = None
            self.last_ordinal = ordinal
            return
        kind, direction = classify_transition(d_odo, d_scene, self.pan_odo,
                                              self.warp_scene_min)
        self.last_kind = kind
        self.last_direction = direction
        self.last_novel = novel
        self.last_ordinal = ordinal
        if kind in self.kind and (novel or not self.novel_only):
            self.trigger_step = self._step
            self.n_triggers += 1

    def vote(self) -> int:
        """1 while the last qualifying fire is inside its hold window, else
        0 -- ApuActivitySignal's held-pulse shape, not a latch: a vote that
        never dropped back to 0 would look the same on the second room a
        game ever loads as it does on the fiftieth."""
        if self.trigger_step is None:
            return 0
        return int(self._step - self.trigger_step < self.hold)

    def n_rooms(self) -> int:
        return self._rooms.n_rooms()

    def stats(self) -> dict:
        return {"step": self._step, "n_rooms": self.n_rooms(),
                "n_settles": self.n_settles, "n_triggers": self.n_triggers,
                "trigger_step": self.trigger_step,
                "last_kind": self.last_kind, "last_novel": self.last_novel,
                "cap_hits": self._rooms.cap_hits}


# ===========================================================================
# Signal -- scene-cut / blank-fold transition classifier (opt-in)
# ===========================================================================
#
# Turns the PPU scroll odometer's OWN re-anchor events -- deliberately
# blind to position on both of its branches -- into transition evidence
# instead of reading them as no signal at all. odo_fold_frame (nes_core/
# src/ppu.rs) has TWO re-anchor branches and, until now, only one of them
# was ever surfaced to a caller:
#
#   BRANCH B -- a RENDERED scroll discontinuity (|dx| or |dy| > 64 px in
#     one frame): bumps `odometer_scene` and re-anchors WITHOUT
#     integrating -- a stage wipe must not read as a multi-hundred-pixel
#     walk backwards. Already surfaced as `Pool.get_odometer_scene_per_
#     worker` / pseudo-address 0x803 -- every odometer profile's is_clear
#     hook already receives this byte and ignores it.
#   BRANCH A -- a BLACKOUT (< 120 rendered lines: stage wipe, level-load
#     blank, death fade): drops the anchor silently. Nothing read this at
#     all before `Pool.get_odometer_blank_per_worker` / `ppu::odo_blank`
#     (this signal's reason for existing).
#
# The RE-ANCHOR EVENT is the evidence; the position delta across it never
# was -- that is the whole reason odo_fold_frame drops the anchor instead
# of integrating across the discontinuity. This class folds a rolling
# window of (odometer, scene, blank) observations into exactly that
# evidence, classified by the SAME `classify_transition` every room_fp
# consumer already trusts (imported at the top of this file) -- no new
# hashing or classification logic is introduced here either.

#: `classify_transition`'s three possible kinds -- see its own docstring
#: for the pan/warp/fade definitions. Unlike RoomFpTransitionSignal this
#: signal has no adjacency/novelty semantics to protect, so nothing is
#: excluded from the default: the MAGNITUDE gate (`scene_min`/`blank_min`,
#: which a profile MUST measure and supply -- see the class docstring) is
#: what keeps ordinary play from voting, not the kind filter.
SCENE_CUT_KINDS = ("pan", "warp", "fade")
#: Reused verbatim from StreamingConfluenceDetector's own defaults so a
#: profile that arms both mechanisms sees the same check cadence.
SCENE_CUT_WINDOW = 240
SCENE_CUT_STRIDE = 20
#: Observations the vote stays raised after a qualifying window -- the
#: same held-pulse shape ApuActivitySignal/RoomFpTransitionSignal use.
SCENE_CUT_HOLD = 60


class SceneCutSignal:
    """Rolling-window transition classifier over the PPU scroll odometer's
    own scene ordinal and dropped-fold counter (see the module banner
    above for the two re-anchor branches this reads).

    Over the current window (oldest buffered observation vs the newest),
    computes:

      d_odo   = integrated odometer delta (dx, dy) across the window
      d_scene = change in the scene ordinal across the window
      d_blank = number of dropped (blackout) folds across the window --
                a plain delta of a monotonically-increasing counter, so
                this IS the count of blank folds inside the window

    classifies (d_odo, d_scene) with the existing `classify_transition`,
    and votes when the classified kind is in `kind` AND EITHER magnitude
    threshold is met: `d_scene >= scene_min or d_blank >= blank_min`. The
    magnitude gate is load-bearing on its own: ordinary forward motion
    produces d_scene == 0 and d_blank == 0 regardless of what
    classify_transition happens to label the odometer delta (a plain
    walk covering the pan-sized 128-384px window can and does classify as
    "pan" -- see false positives below), so nothing votes unless a
    re-anchor EVENT actually happened inside the window.

    PURITY. The surface is PPU scroll-register state via the certified
    odometer (`Pool.get_odometer_per_worker`, `get_odometer_scene_per_
    worker`, `get_odometer_blank_per_worker`) -- hardware facts our own
    core derives from $2005/$2006 and the rendered-line count, consulting
    no game content. `classify_transition`'s pre-registered constants
    (pan_odo, warp_scene_min) are reused, not re-derived.

    NO DEFENSIBLE GLOBAL DEFAULT for `scene_min`/`blank_min` -- this is
    why they are required keyword arguments with NO default value here.
    Metroid was measured noisy at camera clamp/seam (ordinary play throws
    spurious scene bumps of its own), and Zelda fades are invisible to
    the scene ordinal entirely (kind: [fade] leaning on blank_min is the
    only way that game votes at all). A profile must measure its own
    null d_scene/d_blank distribution (scripts/clear_calibrate.py: a
    NOOP + forward-hold drive over the profile's own start state) and set
    both minimums above the observed null, or declare the signal
    `enabled: false` with a reason -- never guess a number.

    FALSE POSITIVES THIS SIGNAL CANNOT DISCRIMINATE ON ITS OWN. DEATH is
    a measured `warp`: the ROOMGRAPH probe receipts record Zelda's death
    flash as scene +2 with the odometer flat -- exactly classify_
    transition's warp signature, so `kind: [warp]` with a low scene_min
    is a death detector by itself. CAMERA CLAMP / SEAM NOISE is the
    documented per-frame false positive at the classifier level
    (classify_transition's own docstring: "the core's scene-cut
    heuristic fires on ordinary camera clamp/seam noise near screen
    edges -- exactly where real pans happen"), which is exactly why
    scene_min cannot be 1 and cannot be a global constant. ATTRACT LOOP
    cuts constantly. PAUSE screens that disable rendering trip the blank
    half (< 120 rendered lines) just as a real blackout does. GAME OVER
    trips the blank half and never releases -- a caller holding this
    signal's vote open-ended needs its own persistence veto; this class
    only offers the held pulse below, not that veto.
    """

    def __init__(self, *, scene_min, blank_min, kind=SCENE_CUT_KINDS,
                 window: int = SCENE_CUT_WINDOW, stride: int = SCENE_CUT_STRIDE,
                 pan_odo=(128, 384), warp_scene_min: int = 2,
                 hold: int = SCENE_CUT_HOLD):
        bad_kind = set(kind) - set(SCENE_CUT_KINDS)
        if bad_kind:
            raise ValueError(
                f"scene_cut kind must be a subset of {SCENE_CUT_KINDS}, "
                f"got {sorted(bad_kind)}")
        self.scene_min = float(scene_min)
        self.blank_min = float(blank_min)
        self.kind = frozenset(kind)
        self.window = max(2, int(window))
        self.stride = max(1, int(stride))
        self.pan_odo = (int(pan_odo[0]), int(pan_odo[1]))
        self.warp_scene_min = int(warp_scene_min)
        self.hold = max(1, int(hold))
        self.reset()

    def reset(self) -> None:
        """Drop the latch AND the rolling window -- the samples that
        produced a fire are still inside the window otherwise, and the
        very next check would re-fire on the same stale evidence
        (identical reasoning to every other reset() in this file)."""
        self._buf: deque = deque(maxlen=self.window)
        self._n = 0
        self.trigger_step: int | None = None
        self.n_triggers = 0
        self.n_checks = 0
        self.last_kind: str | None = None
        self.last_direction = None
        self.last_d_scene = 0
        self.last_d_blank = 0
        self.last_d_odo = (0, 0)

    def push(self, odo_xy, scene: int, blank: int) -> None:
        """Feed one observation: the odometer's integrated (x, y), the
        scene ordinal, and the dropped-fold counter -- all three already
        produced by the certified odometer for every profile that arms
        it (`Pool.get_odometer_per_worker`, `get_odometer_scene_per_
        worker`, `get_odometer_blank_per_worker`)."""
        self._n += 1
        self._buf.append((int(odo_xy[0]), int(odo_xy[1]), int(scene), int(blank)))
        if self._n % self.stride != 0 or len(self._buf) < 2:
            return
        self.n_checks += 1
        x0, y0, s0, b0 = self._buf[0]
        x1, y1, s1, b1 = self._buf[-1]
        d_odo = (x1 - x0, y1 - y0)
        d_scene = s1 - s0
        d_blank = b1 - b0
        kind, direction = classify_transition(d_odo, d_scene, self.pan_odo,
                                              self.warp_scene_min)
        self.last_kind, self.last_direction = kind, direction
        self.last_d_scene, self.last_d_blank, self.last_d_odo = d_scene, d_blank, d_odo
        if (kind in self.kind
                and (d_scene >= self.scene_min or d_blank >= self.blank_min)):
            self.trigger_step = self._n
            self.n_triggers += 1

    def vote(self) -> int:
        """1 while the last qualifying window is inside its hold window,
        else 0 -- a held pulse, not a latch (ApuActivitySignal/
        RoomFpTransitionSignal's shape): a vote that never dropped back
        to 0 would look the same on the fiftieth cut as the first."""
        if self.trigger_step is None:
            return 0
        return int(self._n - self.trigger_step < self.hold)

    def stats(self) -> dict:
        return {"n": self._n, "n_checks": self.n_checks,
                "n_triggers": self.n_triggers, "trigger_step": self.trigger_step,
                "last_kind": self.last_kind, "last_direction": self.last_direction,
                "last_d_scene": self.last_d_scene, "last_d_blank": self.last_d_blank,
                "last_d_odo": self.last_d_odo}


# ===========================================================================
# Streaming confluence detector (live solver hot-loop form)
# ===========================================================================

class UnfireableHook(RuntimeError):
    """A clear detector was asked to run on a profile it cannot fire for.

    Raised by StreamingConfluenceDetector.from_profile rather than
    returned, and carrying clear_reachability's per-signal table in the
    message, because the failure this ends is a silent one: a harness
    constructed the detector on a profile whose `coord` half was
    arithmetically dead, ran its checks, and wrote the resulting silence
    down as a measurement. An instrument that could not have said yes has
    to refuse to start, not produce a zero."""


class StreamingConfluenceDetector:
    """Live, RAM-only streaming form of the confluence clear detector, meant
    for the Go-Explore solver's per-step is_clear hook (GenericGame with
    `solve: clear: {mode: confluence}`).

    It reuses the SAME ground-truthed signal functions as the offline --test
    detector -- score_tally_windows and coord_entity_windows -- evaluated over
    a BOUNDED rolling RAM window, so the per-step cost stays O(window)
    amortized instead of the O(episode^2) a full re-scan every step would be.

    Availability note (kept honest): the offline detector fuses four signals
    -- audio, tally, lock, coord. Inside the solver's is_clear the only thing
    handed to us per step is a RAM snapshot: there is no audio stream, and no
    env handle for the differential input-LOCK probe (that one needs
    save/restore of the emulator). So this streaming form votes on the two
    purely-RAM-derivable signals -- `tally` (a timer->score conversion cadence)
    and `coord` (a position reset toward a level-start value co-occurring with a
    contiguous entity-slot wipe = a fresh room/level loading in) -- and declares
    a clear when at least `min_signals` of them fire inside the same rolling
    window (default 2 = BOTH must agree, a genuine two-signal confluence and
    the RAM fingerprint of a level load). It never fires on either signal alone,
    so an ordinary tally (an in-level 1-up) or an ordinary scroll cannot fake a
    clear. The full weighted-0.75 four-signal detector remains the authority for
    offline verification (clear_detect.py --test).

    v2 (2026-08-08) -- THE COMBAT-BLIP FIX, two knobs, both default-inert:

      persist_checks: N -- the confluence must hold for N CONSECUTIVE checks
        before the latch closes. Default 1 == the shipped behavior,
        byte-identical. Cheap insurance against a one-off evaluation.

      progress_median: K -- median-filter the progress series over K trailing
        samples before the coord test. Default 0/1 = off.

    Two knobs and not one because persistence ALONE provably does not deliver
    "a 1-2 sample RAM spike must not clear" (the Double Dragon failure:
    progress 72 -> 846 -> 88 inside 5 steps at 27 actions, no advance, no
    death). coord scans sub-windows of 60 at stride 15 and compares each
    sub-window's FIRST and LAST sample, so when a spike happens to land on a
    sub-window boundary it reproduces the high->low endpoint signature at
    EIGHT consecutive checks (worked out over the exact window arithmetic:
    window 240 / stride 20 / sub 60 / sub-stride 15, spike at relative index
    30 fires checks 100..240). Defeating that with persistence alone needs
    N >= 9, which is more checks than a genuine level load's evidence even
    survives in the rolling window (~window/stride = 12) and delays every real
    detection by 9*stride steps. The median filter kills the impulse outright
    and is alignment-independent, while leaving a real load's step change
    intact; persistence then rides on top as a second, cheaper filter.

    Neither knob addresses a ROOM transition, whose position reset is just as
    sustained and just as step-shaped as a stage clear's -- that one needs the
    progress-aware room veto in the caller (GenericGame.is_clear).

    v3 (2026-08-09) -- THE APU CHANNEL-ACTIVITY VOTE, opt-in, default absent:

      apu_weight: W > 0 arms a third signal (ApuActivitySignal) fed the 5-bit
        $4015 activity mask the caller passes to push(). The vote becomes
        `tally + coord + W*apu >= min_signals` instead of `tally + coord >=
        min_signals`. W defaults to 0, and at 0 no ApuActivitySignal is even
        constructed, so the arithmetic reduces to the shipped integer path
        byte-for-byte and no existing receipt moves.

      The vote is ADDITIVE, not a veto: at the default min_signals=2, arming
      it with W>=1 makes firing EASIER (coord+apu can carry a clear without
      tally). To make it a REQUIREMENT instead, raise the bar with it --
      `min_signals: 3, apu_weight: 1.0` means all three must agree. That is
      the configuration to reach for on a game whose false positives are
      RAM-shaped (combat blips, room loads), because those move no channels.

      Masks are optional per push: with none supplied the signal simply never
      accumulates and votes 0, which is the safe direction (a caller that
      forgot to plumb the audio modality gets a stricter detector, not a
      looser one).

    v4 (2026-08-26) -- ELIGIBILITY AND THE REQUIRED CLASS.

      A profile votes solely on signals that CAN fire for it. `eligibility`
      is the per-signal table from clear_reachability.clear_quorum: a
      signal it marks DEAD (coord on an odometer/fight_gate/single-byte
      progress) or DEGENERATE (a MEASURED null fire-rate at or above
      MAX_NULL_RATE -- `tally` fired on 22/22, 28/28, 43/43 Castlevania
      checks and 30/30 Bubble Bobble checks) contributes 0 to the vote
      instead of being counted as a corroborator that happened to be
      silent. Default None = every wired signal eligible, which is the
      shipped path byte-for-byte: no profile carries a measured null today
      and a DEAD signal already votes 0 by never firing.

      RULE 5, THE REQUIRED CLASS: a fire additionally requires TRANSITION
      EVIDENCE -- an eligible signal answering "a scene committed" or "the
      world did not come back". With the six shelf signals unwired that is
      `coord` alone. At the shipped min_signals=2 over {tally, coord} this
      changes nothing (the sum already forces coord), and every existing
      receipt is unmoved. What it forbids is the arithmetic that WAS
      measured firing: runs/clear_control_2026-08-26/bb_offline_r99.json
      crossed at frame 320 on {audio: 1, tally: 1, lock: 1, coord: 0},
      1736 frames before the true clear -- three corroborators agreeing
      with each other about a scene change none of them observed. Arming
      another corroborator can no longer make firing easier, which is the
      trap the v3 apu_weight note above warns about in writing.

      `from_profile` is the guarded constructor: it REFUSES to build a
      detector whose quorum is UNREACHABLE. That is the point the offline
      control harness needed and did not have -- it constructed this class
      directly on bubble_bobble and tetris_b, whose progress readouts span
      1 and 32 units against coord's required 300, ran 30 and 220 checks,
      and wrote `streaming_hit: false`."""

    #: Signals this class can actually derive. Anything else in the
    #: eligibility table is either the offline harness's (audio, lock) or
    #: on the shelf, and is ignored here rather than silently counted.
    VOTING_SIGNALS = ("tally", "coord", "apu")

    @classmethod
    def from_profile(cls, profile: dict, progress_fn, **overrides):
        """Build the detector this profile describes, or refuse.

        Raises UnfireableHook when clear_reachability.clear_quorum reports
        UNREACHABLE, in the same "reason in the message, never a silent
        no-op" shape InputLockSignal.probe() and SceneCutSignal.__init__
        already use. A harness that cannot construct cannot go on to write
        a hit rate over rows that were incapable of a hit."""
        q = clear_reachability.clear_quorum(profile)
        if not q.ok:
            raise UnfireableHook(q.reason + "\n" + q.table())
        solve = profile.get("solve") or {}
        cl = solve.get("clear") or {}
        kw = dict(window=int(cl.get("window", 240)),
                  stride=int(cl.get("stride", 20)),
                  min_signals=cl.get("min_signals"),
                  persist_checks=cl.get("persist_checks"),
                  progress_median=cl.get("progress_median"),
                  apu_weight=float(cl.get("apu_weight", 0.0) or 0.0),
                  apu_params=dict(cl.get("apu") or {}),
                  eligibility=q)
        kw.update(overrides)
        return cls(progress_fn, **kw)

    def __init__(self, progress_fn, window: int = 240, stride: int = 20,
                 min_signals: int | None = None,
                 persist_checks: int | None = None,
                 progress_median: int | None = None,
                 apu_weight: float | None = None,
                 apu_params: dict | None = None,
                 eligibility=None):
        self._progress = progress_fn
        self.window = int(window)
        self.stride = int(stride)
        # int for an integral bar (so the disarmed path's comparison is the
        # shipped integer one to the byte), float only when a profile really
        # asks for a fractional bar -- which only makes sense together with a
        # fractional apu_weight.
        self.min_signals = (
            2 if min_signals is None else
            int(min_signals) if float(min_signals).is_integer()
            else float(min_signals))
        self.persist_checks = (1 if persist_checks is None
                               else max(1, int(persist_checks)))
        self.progress_median = (1 if progress_median is None
                                else max(1, int(progress_median)))
        self.apu_weight = 0.0 if apu_weight is None else float(apu_weight)
        self._apu = (ApuActivitySignal(**(apu_params or {}))
                     if self.apu_weight > 0 else None)
        # Rule 1 + Rule 2. None = every signal this class derives is
        # eligible, which is the shipped path exactly; a quorum table
        # narrows it to the signals that can fire for this profile.
        self.eligibility = eligibility
        table = getattr(eligibility, "signal_state", None) or {}
        self._eligible = frozenset(
            n for n in self.VOTING_SIGNALS
            if n not in table or table[n].eligible)
        # Rule 5. The signals that answer "a scene committed" / "the world
        # did not come back", as opposed to corroborators that answer
        # "something changed". Read off the table rather than hardcoded so
        # wiring a new transition signal does not need a second edit here.
        self._transition = frozenset(
            n for n in self.VOTING_SIGNALS
            if (table[n].transition_evidence if n in table else n == "coord"))
        # A signal must be BOTH eligible and transition evidence to satisfy
        # Rule 5. An ineligible one contributing 0 to the sum while still
        # unlocking the required class would be the same double-standard
        # the eligibility mask exists to remove.
        self._required = self._transition & self._eligible
        self._ram: list[np.ndarray] = []
        self._gx: list[int] = []
        self._n = 0
        self._fired = False
        self._streak = 0
        # Telemetry (read by tests / receipts; never by the vote).
        self.n_checks = 0
        self.n_votes = 0
        self.n_apu_votes = 0
        self.n_required_class_vetoes = 0

    def reset(self) -> None:
        """Un-latch AND discard the rolling evidence window.

        A veto that only suppresses the return value is a no-op in practice:
        the samples that produced the fire are still inside the window, so the
        very next check re-fires on the same stale evidence the moment the veto
        window expires. A vetoed fire therefore throws its evidence away with
        it -- the detector has to earn a fresh confluence out of samples
        observed AFTER the veto."""
        self._fired = False
        self._streak = 0
        self._ram.clear()
        self._gx.clear()
        self._n = 0
        if self._apu is not None:
            self._apu.reset()

    def push(self, ram, apu_mask=None) -> bool:
        """Feed one RAM snapshot (and optionally this observation's 5-bit APU
        activity mask); returns True once the confluence has fired (and stays
        True thereafter -- the clear is a latching event, until reset()
        explicitly drops the latch).

        `apu_mask` is ignored entirely unless apu_weight > 0, so every
        existing single-argument call site is unchanged."""
        if self._apu is not None:
            self._apu.push(apu_mask)
        if self._fired:
            return True
        # The solver hands raw `bytes` in the live hot loop (pool step
        # results), not an ndarray -- np.asarray(bytes_obj, dtype=uint8)
        # treats the whole blob as a single string-like scalar and tries
        # to int()-parse it (fails on any non-ASCII-digit byte); frombuffer
        # correctly unpacks it into individual byte values instead.
        arr = (np.frombuffer(ram, dtype=np.uint8)
               if isinstance(ram, (bytes, bytearray)) else
               np.asarray(ram, dtype=np.uint8))
        self._ram.append(arr)
        self._gx.append(int(self._progress(ram)))
        if len(self._ram) > self.window:
            self._ram.pop(0)
            self._gx.pop(0)
        self._n += 1
        if self._n % self.stride != 0 or len(self._ram) < 16:
            return False
        hist = np.stack(self._ram)
        gx = np.array(self._gx, dtype=np.int64)
        if self.progress_median > 1:
            gx = trailing_median(gx, self.progress_median)
        fired = {
            "tally": 1 if score_tally_windows(hist) else 0,
            "coord": 1 if coord_entity_windows(hist, gx) else 0,
        }
        self.n_checks += 1
        # A signal this profile cannot fire contributes 0 rather than
        # being counted as a corroborator that happened to be quiet.
        tally = fired["tally"] if "tally" in self._eligible else 0
        coord = fired["coord"] if "coord" in self._eligible else 0
        if self._apu is None:
            # Byte-identical to the shipped integer path.
            passed = (tally + coord) >= self.min_signals
        else:
            apu = self._apu.vote()
            fired["apu"] = apu
            self.n_apu_votes += apu
            if "apu" not in self._eligible:
                apu = 0
            passed = ((tally + coord + self.apu_weight * apu)
                      >= self.min_signals - 1e-9)
        # RULE 5. Corroborators alone cannot carry a clear, however many of
        # them agree: at least one signal answering "a scene committed"
        # must have fired. At the shipped min_signals=2 over {tally, coord}
        # the sum already forces this, so no existing receipt moves; what
        # it forbids is the frame-320 shape (three corroborators, coord=0).
        if passed and not any(fired.get(n) for n in self._required):
            passed = False
            self.n_required_class_vetoes += 1
        if passed:
            self.n_votes += 1
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.persist_checks:
            self._fired = True
        return self._fired

    def warmup_observations(self) -> int:
        """Observations a FRESH detector must be fed before push() can return
        True -- the phase/warm-up counterpart to the caller's NOOP margin, and
        the number any fixed-length replay harness has to budget for.

        Read straight off this instance's own knobs, so no caller has to
        re-derive (or drift from) the arithmetic push() actually enforces:

          * a check happens only at `_n % stride == 0` with at least 16
            samples buffered, so the FIRST one lands at the smallest multiple
            of `stride` that is >= 16;
          * `persist_checks` consecutive checks must pass, which adds
            `stride * (persist_checks - 1)`;
          * when the audio vote is REQUIRED -- apu_weight armed AND a bar
            above 2, the most the two RAM signals can ever sum to -- the
            first check that can pass is also the first one at or after
            ApuActivitySignal.warmup_observations(). Below that bar the audio
            vote is additive (tally+coord can carry a clear alone), so it
            does not hold the detector's warm-up up.

        WHY THIS IS PUBLIC. Feeding a windowed detector fewer observations
        than this and reading its silence as a verdict is a structural no-op,
        not a strict result: the measured case is the counterfactual gate's
        52-observation branches against the 100 this returns for the
        `min_signals: 3, apu_weight: 1.0` configuration these very docs
        recommend for RAM-shaped false positives -- every branch returned
        no_clear because none of them ever reached an evaluation.

        A window under 16 can never satisfy push()'s own sample guard, so the
        hook is unfirable by construction; that is reported as an
        unsatisfiable budget rather than as a number a caller could meet."""
        if self.window < 16:
            return 1 << 30
        if not self._required:
            # Rule 5 can never be satisfied: no eligible signal answers
            # "a scene committed". Reported as an unsatisfiable budget
            # rather than as a number a caller could meet and then read
            # the resulting silence as a verdict.
            return 1 << 30
        stride = max(1, self.stride)
        need = 16
        if self._apu is not None and self.min_signals > 2 + 1e-9:
            need = max(need, self._apu.warmup_observations())
        first = ((need + stride - 1) // stride) * stride
        return first + stride * (self.persist_checks - 1)


# ===========================================================================
# Episode driver -- ties the four signals together over one replay
# ===========================================================================

def run_episode(env, game, bitmasks, dir_bitmask, actions: list[int],
                 start_wd: tuple, margin_actions: int = 90,
                 probe_stride: int = 20, probe_frames: int = LOCK_PROBE_FRAMES,
                 fs: int = FS) -> dict:
    """Replays `actions` (already positioned at the root + rooting NOOP),
    then `margin_actions` more of NOOP (each action step is `fs` raw frames
    -- the PROFILE's frame_skip, the one the trace was recorded at, not a
    fixed 4 -- so this is ~margin_actions*fs/60 seconds), collecting
    everything the detector needs. Returns a report with the true clear
    frame (the game adapter's own clear check, at raw single-frame
    resolution) and the detector's verdict."""
    sample_rate = env.sample_rate
    audio_sig = AudioCadenceSignal(sample_rate)
    lock_track = InputLockTrack(probe_stride)
    ram_hist: list[np.ndarray] = []
    gx_hist: list[int] = []

    true_clear_frame = None
    frame_idx = -1
    # Per-replay ctx, exactly as Solver.observe threads one per worker. SMB
    # ignores it (byte-identical); a GenericGame `clear: {mode: ...}` WIN-
    # CONDITION is short-circuited to False without it.
    clear_ctx: dict = {}

    def observe_frame():
        nonlocal true_clear_frame, frame_idx
        frame_idx += 1
        ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        audio = env.get_audio()
        audio_sig.push_frame(audio)
        ram_hist.append(ram)
        gx_hist.append(game.progress(ram))
        if true_clear_frame is None and (game.is_clear(start_wd, ram, clear_ctx)
                                          or game.is_finale(start_wd, ram)):
            true_clear_frame = frame_idx
        if frame_idx % probe_stride == 0:
            locked, n_diff = differential_input_lock_probe(env, dir_bitmask, probe_frames)
            lock_track.record(frame_idx, locked, n_diff)

    all_actions = list(actions) + [0] * margin_actions
    for a in all_actions:
        mask = int(bitmasks[a])
        for _ in range(fs):
            env.step(mask)
            observe_frame()

    ram_hist_arr = np.stack(ram_hist)
    gx_arr = np.array(gx_hist, dtype=np.int64)
    tally_hits = score_tally_windows(ram_hist_arr)
    coord_hits = coord_entity_windows(ram_hist_arr, gx_arr)

    def in_hits(hits, f):
        return any(s <= f < e for s, e in hits)

    n = ram_hist_arr.shape[0]
    votes = {"audio": np.zeros(n, dtype=np.int8), "tally": np.zeros(n, dtype=np.int8),
             "lock": np.zeros(n, dtype=np.int8), "coord": np.zeros(n, dtype=np.int8)}
    for f in range(n):
        votes["audio"][f] = audio_sig.vote_at(f)
        votes["tally"][f] = int(in_hits(tally_hits, f))
        votes["lock"][f] = lock_track.vote_at(f)
        votes["coord"][f] = int(in_hits(coord_hits, f))

    weighted = (WEIGHTS["audio"] * votes["audio"] + WEIGHTS["tally"] * votes["tally"]
                + WEIGHTS["lock"] * votes["lock"] + WEIGHTS["coord"] * votes["coord"])
    crossed = weighted >= THRESHOLD - 1e-9
    rising = [f for f in range(n) if crossed[f] and (f == 0 or not crossed[f - 1])]

    detected_frame = rising[0] if rising else None
    contributions_at_detect = None
    if detected_frame is not None:
        contributions_at_detect = {k: int(v[detected_frame]) for k, v in votes.items()}

    return {
        "n_frames": n,
        "true_clear_frame": true_clear_frame,
        "detected_frame": detected_frame,
        "all_crossings": rising,
        "contributions_at_detect": contributions_at_detect,
        "signal_debug": {
            "audio_trigger_frame": audio_sig.trigger_frame(),
            "audio_deltas": audio_sig.deltas,
            "tally_windows": tally_hits,
            "coord_windows": coord_hits,
            "lock_probes": lock_track.probes,
        },
    }


# ===========================================================================
# Ground-truth self-test
# ===========================================================================

TOLERANCE_FRAMES = 120

# Hand-picked, diverse SMB Go-Explore solution runs (excludes anything under
# cv_chain_hw2, which is a live solver run this script must not touch, and
# anything not clearly a Mario/SMB solve).
DEFAULT_RUNS = [
    "runs/regress_pre/solutions/sol_000",     # power-on 1-1 -> 1-2
    "runs/ge_1_2_solve/solutions/sol_000",    # 1-2 -> 1-3 (pipe/underground transit)
    "runs/ge_1_3_solve/solutions/sol_000",    # 1-3 -> 1-4
    "runs/ge_1_4_solve/solutions/sol_000",    # 1-4 -> 2-1 (CASTLE clear, no flagpole)
    "runs/ge_2_1_solve/solutions/sol_000",    # 2-1 -> 2-2
]


def discover_smb_solutions() -> list[str]:
    """glob-based discovery, restricted to mario/smb-labelled run dirs, with
    the live solver run explicitly excluded."""
    found = []
    for p in sorted(glob.glob(str(REPO / "runs/**/solutions/sol_*.actions.npy"),
                               recursive=True)):
        rel = str(Path(p).relative_to(REPO))
        if "cv_chain_hw2" in rel:
            continue
        base = rel[: -len(".actions.npy")]
        json_path = REPO / (base + ".json")
        if not json_path.exists():
            continue
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            continue
        root_state = str(meta.get("root_state", ""))
        # Restrict to states that are clearly Mario/SMB (path convention
        # used across the whole solver pipeline for this game).
        low = root_state.lower()
        if "mario" in low or "smb" in low or "1-1" in low or "8-1" in low:
            found.append(base)
    return found


# ===========================================================================
# The harness -- the three things that used to be hardcoded to SMB
# ===========================================================================

DIRECTIONS = ("right", "left", "up", "down")


def _lock_probe_index(action_space: list) -> int:
    """Index of the action `differential_input_lock_probe` holds.

    The probe asks one question: does holding a direction move RAM relative
    to holding nothing? So it needs *a* directional action, and which one
    barely matters. Preference order is right (SMB's historical choice, which
    keeps the SMB harness bit-for-bit what every banked receipt used), then
    any other bare direction, then any action that merely CONTAINS one.

    It raises rather than falling back to NOOP on a space with no direction
    at all. That fallback would probe NOOP against NOOP, land a near-zero
    diff on every frame, and hand the `lock` signal a free permanent vote --
    a signal that is structurally always-on is worse than an absent one,
    because it looks like evidence."""
    for want in DIRECTIONS:
        if [want] in action_space:
            return action_space.index([want])
    for i, combo in enumerate(action_space):
        if any(b in DIRECTIONS for b in combo):
            return i
    raise SystemExit(
        "[clear_detect] this profile's action_space contains no directional "
        "action, so the differential input-LOCK probe has nothing to hold "
        "against NOOP. Refusing rather than probing NOOP vs NOOP, which "
        "would vote 'locked' on every frame.")


class Harness(NamedTuple):
    """Game adapter + action space + frame_skip for one replay.

    ALL THREE USED TO BE HARDCODED TO SMB. `run_ground_truth_test` opened
    with `game = SmbGame()` and replayed every trace through SMB's 11-action
    space at SMB's frame_skip, so no other profile could reach the detector
    at all -- which is why the 2026-08-26 clear-detection census exercised
    the detector on exactly one of the 29 profiles it surveyed, and why the
    28 silences it collected were not measurements of those games. Regression
    guard: tests/test_clear_detect_profile_entry.py."""
    game: object
    action_space: list
    frame_skip: int
    bitmasks: object
    dir_index: int
    profile: str | None
    #: The parsed profile, not just its path. summarize_runs resolves
    #: per-signal eligibility from it, and re-reading the YAML there would
    #: let the receipt's quorum drift from the adapter that actually ran.
    profile_dict: dict | None = None

    def provenance(self) -> dict:
        """Receipt block naming what actually replayed, so a reader can tell
        an SMB run from a Castlevania run without trusting the filename."""
        return {
            "profile": self.profile,
            "game_adapter": type(self.game).__name__,
            "rom": getattr(self.game, "rom", None),
            "frame_skip": self.frame_skip,
            "n_actions_in_space": len(self.action_space),
            "lock_probe_action": list(self.action_space[self.dir_index]),
        }


def build_harness(profile_path: str | None = None) -> Harness:
    """Build the replay harness from a profile YAML, or the SMB one.

    With no profile this is the historical SMB triple, byte-identical to what
    every banked receipt was produced under (SmbGame, this module's
    ACTION_SPACE, FS). With one, all three come from the profile: the game
    adapter via `go_explore_solve.make_game` (so an SMB-engine profile still
    gets SmbGame and a `solve:` profile gets GenericGame), the action space
    the recorded action indices index into, and the frame_skip the trace was
    recorded at."""
    if not profile_path:
        game = SmbGame()
        return Harness(game, ACTION_SPACE, FS,
                       action_space_to_bitmasks(ACTION_SPACE),
                       _lock_probe_index(ACTION_SPACE), None)

    from go_explore_solve import make_game

    path = Path(profile_path)
    if not path.is_absolute() and not path.exists():
        path = REPO / profile_path
    if not path.exists():
        raise SystemExit(f"[clear_detect] profile not found: {profile_path}")
    prof = yaml.safe_load(path.read_text())
    if not isinstance(prof, dict) or "action_space" not in prof:
        raise SystemExit(
            f"[clear_detect] {path} has no `action_space:`. The recorded "
            f"action indices in a solution trace are indices INTO that list; "
            f"without it there is nothing to replay them against.")
    space = [list(a) for a in prof["action_space"]]
    return Harness(make_game(prof), space,
                   int(prof.get("frame_skip", FS)),
                   action_space_to_bitmasks(space),
                   _lock_probe_index(space), str(profile_path), prof)


def run_ground_truth_test(run_bases: list[str], verbose: bool = True,
                          profile: str | None = None) -> dict:
    harness = build_harness(profile)
    game, action_space, fs = harness.game, harness.action_space, harness.frame_skip
    bitmasks = harness.bitmasks
    dir_bitmask = bitmasks[harness.dir_index]

    per_run = []
    for base in run_bases:
        json_path = REPO / (base + ".json")
        npy_path = REPO / (base + ".actions.npy")
        if not json_path.exists() or not npy_path.exists():
            per_run.append({"run": base, "error": "missing solution files"})
            continue
        meta = json.loads(json_path.read_text())
        actions = np.load(npy_path).tolist()
        root_path = REPO / meta["root_state"]
        if not root_path.exists():
            per_run.append({"run": base, "error": f"root state missing: {root_path}"})
            continue
        root_bytes = root_path.read_bytes()

        env = None
        try:
            env = nes_core.NESEnvironment(game.rom, frame_skip=1)
            env.reset()
            env.set_audio_output_enabled(True)
            env.load_state(root_bytes)
            for _ in range(fs):   # rooting convention: one NOOP = fs raw frames
                env.step(0)

            ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
            # byte_change WIN-CONDITIONs compare against the entrance value the
            # solver itself latches in Solver.seed; without this the hook is
            # inert (baseline None) and cannot fire on any replay.
            if hasattr(game, "note_start"):
                game.note_start(ram0)
            start_wd = tuple(game.level_key(ram0))
            expected_start = tuple(meta["start_wd"])
            # A profile whose `level_key` gained a byte after a trace was
            # recorded replays to a LONGER key with the recorded one as its
            # prefix. That is profile drift, not a bad replay: the root is the
            # same root. Accept it, record it, and keep using the REPLAYED key
            # as the baseline -- comparing against the shorter recorded tuple
            # would make `is_clear` true on frame 0 by tuple-length alone.
            key_drift = (start_wd != expected_start
                         and (start_wd[:len(expected_start)] == expected_start
                              or expected_start[:len(start_wd)] == start_wd))
            if start_wd != expected_start and not key_drift:
                per_run.append({"run": base, "error":
                                 f"root replay mismatch: got {start_wd}, expected {expected_start}"})
                continue

            t0 = time.time()
            report = run_episode(env, game, bitmasks, dir_bitmask, actions,
                                 start_wd, fs=fs)
            elapsed = time.time() - t0
        except Exception as e:
            per_run.append({"run": base, "error": f"replay failed: {e}"})
            continue
        finally:
            if env is not None:
                env.close()

        true_f = report["true_clear_frame"]
        det_f = report["detected_frame"]
        hit = (det_f is not None and true_f is not None
               and abs(det_f - true_f) <= TOLERANCE_FRAMES)
        # False positives: any OTHER rising-edge crossing, beyond the one
        # (if any) that actually lands within tolerance of the truth.
        fp = 0
        matched_one = False
        for f in report["all_crossings"]:
            if not matched_one and true_f is not None and abs(f - true_f) <= TOLERANCE_FRAMES:
                matched_one = True
                continue
            fp += 1

        result = {
            "run": base,
            "start_wd": list(start_wd),
            "level_key_arity_drift": bool(key_drift),
            "clear_wd_expected": meta.get("clear_wd"),
            "n_actions": len(actions),
            "n_frames_replayed": report["n_frames"],
            "true_clear_frame": true_f,
            "detected_frame": det_f,
            "delta_frames": None if (det_f is None or true_f is None) else det_f - true_f,
            "within_tolerance": hit,
            "false_positive_crossings": fp,
            "contributions_at_detect": report["contributions_at_detect"],
            "n_all_crossings": len(report["all_crossings"]),
            "wall_s": round(elapsed, 2),
        }
        per_run.append(result)
        if verbose:
            print(f"[clear_detect] {base}: true={true_f} detected={det_f} "
                  f"hit={hit} fp={fp} contrib={report['contributions_at_detect']} "
                  f"({elapsed:.1f}s)", flush=True)

    return summarize_runs(per_run, profile=harness.profile_dict,
                          harness=harness.provenance())


HIT_RATE_GATE = 0.80
MAX_FALSE_POSITIVES_PER_LEVEL = 1


def summarize_runs(per_run: list[dict], profile: dict | None = None,
                   harness: dict | None = None,
                   roster: str = clear_reachability.OFFLINE) -> dict:
    """Fold replay rows into a receipt, with the denominator fixed.

    THE DEFECT THIS REPLACES. The old fold was

        n_valid  = sum(1 for r in per_run if "error" not in r)
        hit_rate = (n_hit / n_valid) if n_valid else 0.0

    which puts a row the instrument could not have scored into the
    denominator and then reports the resulting quotient as a failure.
    runs/clear_control_2026-08-26/bb_offline_r99.json therefore reads
    `n_valid: 2, hit_rate: 0.0, hit_rate_pass: false` on two Bubble Bobble
    rows whose progress observable spans ONE unit against the >= 300 drop
    `coord` requires -- a FAIL-shaped number manufactured from a
    VOID-shaped measurement, inside the instrument built to end exactly
    that confusion. That is the 41-VOID/0-FAIL adjudication in miniature.

    THE RULE, applied here and everywhere else a clear result is recorded:
    an instrument that could not have said yes contributes no denominator.
    Unreachable rows get their own column, leave `n_valid`, and when
    nothing measurable remains `hit_rate` is None -- not 0.0 -- with
    `hit_rate_pass` None rather than False. `all([])` is True, so the
    false-positive gate is given the same treatment: a vacuous PASS over
    zero rows is the same defect pointing the other way.

    Each row is classified with the four-valued verdict. A row may carry
    its own `profile` dict (overriding the argument), and a bounded replay
    may carry `n_observations` against `warmup_observations` -- feeding a
    windowed detector fewer observations than it needs and reading its
    silence as a verdict is a structural no-op, which is what UNDER_WARMUP
    exists to say out loud."""
    quorums: dict[int, object] = {}
    rows = []
    for raw in per_run:
        row = dict(raw)
        prof = row.pop("profile", None)
        if not isinstance(prof, dict):
            prof = profile
        q = None
        if isinstance(prof, dict):
            q = quorums.get(id(prof))
            if q is None:
                q = clear_reachability.clear_quorum(prof, roster=roster)
                quorums[id(prof)] = q
        need = row.get("warmup_observations")
        seen = row.get("n_observations")
        if "error" in row:
            row["verdict"] = clear_reachability.ERROR
        elif q is not None and not q.ok:
            row["verdict"] = clear_reachability.UNREACHABLE
            row["unreachable_reason"] = q.reason
        elif row.get("within_tolerance"):
            row["verdict"] = clear_reachability.CLEAR
        elif (need is not None and seen is not None and seen < need
              and not row.get("within_tolerance")):
            row["verdict"] = clear_reachability.UNDER_WARMUP
        else:
            row["verdict"] = clear_reachability.NO_CLEAR
        rows.append(row)

    measured = [r for r in rows
                if r["verdict"] in clear_reachability.MEASURED_VERDICTS]
    n_valid = len(measured)
    n_hit = sum(1 for r in measured if r.get("within_tolerance"))
    n_unreachable = sum(1 for r in rows
                        if r["verdict"] == clear_reachability.UNREACHABLE)
    n_under_warmup = sum(1 for r in rows
                         if r["verdict"] == clear_reachability.UNDER_WARMUP)
    n_error = sum(1 for r in rows
                  if r["verdict"] == clear_reachability.ERROR)
    total_fp = sum(r.get("false_positive_crossings", 0) for r in measured)
    hit_rate = (n_hit / n_valid) if n_valid else None
    fp_pass = (all(r.get("false_positive_crossings", 0)
                   <= MAX_FALSE_POSITIVES_PER_LEVEL for r in measured)
               if n_valid else None)

    # per-signal contribution tally, over the runs that produced a detection
    signal_fire_counts = {"audio": 0, "tally": 0, "lock": 0, "coord": 0}
    for r in measured:
        c = r.get("contributions_at_detect")
        if c:
            for k, v in c.items():
                signal_fire_counts[k] = signal_fire_counts.get(k, 0) + int(v)

    if n_valid:
        verdict = (clear_reachability.CLEAR if n_hit
                   else clear_reachability.NO_CLEAR)
    elif n_unreachable:
        verdict = clear_reachability.UNREACHABLE
    elif n_under_warmup:
        verdict = clear_reachability.UNDER_WARMUP
    else:
        verdict = clear_reachability.ERROR

    summary = {
        "harness": harness,
        "verdict": verdict,
        "tolerance_frames": TOLERANCE_FRAMES,
        "n_runs": len(rows),
        "n_valid": n_valid,
        "n_unreachable": n_unreachable,
        "n_under_warmup": n_under_warmup,
        "n_error": n_error,
        "n_hit_within_tolerance": n_hit,
        "hit_rate": hit_rate,
        "hit_rate_gate": HIT_RATE_GATE,
        "hit_rate_pass": (None if hit_rate is None
                          else hit_rate >= HIT_RATE_GATE),
        "total_false_positive_crossings": total_fp,
        "max_false_positives_per_level_gate": MAX_FALSE_POSITIVES_PER_LEVEL,
        "false_positive_gate_pass": fp_pass,
        "signal_fire_counts_at_detect": signal_fire_counts,
        "weights": WEIGHTS,
        "threshold": THRESHOLD,
        "per_run": rows,
    }
    if quorums:
        summary["clear_quorum"] = next(iter(quorums.values())).as_dict()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true",
                     help="run the ground-truth self-test against solution traces")
    ap.add_argument("--runs", nargs="*", default=None,
                     help="explicit solution basenames (without .json/.actions.npy); "
                          "default = a curated 5-run SMB set. REQUIRED with "
                          "--profile.")
    ap.add_argument("--discover", action="store_true",
                     help="glob-discover all mario/smb solution traces instead of the "
                          "curated default set (SMB-only)")
    ap.add_argument("--out", default=str(REPO / "runs" / "clear_detect_receipt.json"))
    ap.add_argument("--profile", default=None,
                     help="game profile YAML (configs/*.yaml). Without it the "
                          "harness is the historical SMB one; with it the game "
                          "adapter, action space and frame_skip all come from "
                          "the profile, so a non-SMB trace can be replayed.")
    args = ap.parse_args()

    if not args.test:
        print(__doc__)
        return 0

    if args.profile and not args.runs:
        # Both trace sources below are SMB-only: DEFAULT_RUNS is a curated
        # five-run SMB list and --discover filters on mario/smb path names.
        # Replaying an SMB trace through another game's adapter would still
        # print a hit rate, and that number would mean nothing -- exactly the
        # kind of confidently-empty result this entry point exists to stop.
        print("[clear_detect] --profile requires --runs: the default trace "
              "set and --discover are both SMB-only, and replaying an SMB "
              "trace against another game's adapter measures nothing.",
              file=sys.stderr)
        return 2

    if args.runs:
        run_bases = args.runs
    elif args.discover:
        run_bases = discover_smb_solutions()
        print(f"[clear_detect] discovered {len(run_bases)} candidate solution traces")
    else:
        run_bases = DEFAULT_RUNS

    summary = run_ground_truth_test(run_bases, profile=args.profile)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    def _gate(v):
        # None is not FAIL. A gate with nothing measurable under it has no
        # verdict, and printing one is how a VOID becomes a FAIL in a doc.
        return "PASS" if v else ("VOID" if v is None else "FAIL")

    rate = summary["hit_rate"]
    print(f"\n[clear_detect] hit rate "
          f"{'n/a' if rate is None else format(rate, '.2f')} "
          f"({summary['n_hit_within_tolerance']}/{summary['n_valid']}) "
          f"gate>={HIT_RATE_GATE:.2f}: {_gate(summary['hit_rate_pass'])}")
    if summary["n_unreachable"]:
        print(f"[clear_detect] {summary['n_unreachable']} run(s) UNREACHABLE "
              f"— excluded from the denominator, not counted as misses")
        q = summary.get("clear_quorum") or {}
        for name, sig in (q.get("signals") or {}).items():
            print(f"[clear_detect]   {name:<20} {sig['state']:<10} "
                  f"{sig['reason']}")
    print(f"[clear_detect] false positives total={summary['total_false_positive_crossings']} "
          f"gate<={MAX_FALSE_POSITIVES_PER_LEVEL}/level: "
          f"{_gate(summary['false_positive_gate_pass'])}")
    print(f"[clear_detect] receipt written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
