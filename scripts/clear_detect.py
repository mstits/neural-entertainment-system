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

The ground-truth self-test (--test) replays real SMB Go-Explore solution
traces from their recorded root state, finds the frame the world/level
bytes truly advance (the same forward-clear check the solver itself uses),
runs the detector over the replay, and reports firing accuracy.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nes_core  # noqa: E402

from go_explore_solve import SmbGame  # noqa: E402  (repo's own verified SMB adapter)
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

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
# Streaming confluence detector (live solver hot-loop form)
# ===========================================================================

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
    offline verification (clear_detect.py --test)."""

    def __init__(self, progress_fn, window: int = 240, stride: int = 20,
                 min_signals: int | None = None):
        self._progress = progress_fn
        self.window = int(window)
        self.stride = int(stride)
        self.min_signals = 2 if min_signals is None else int(min_signals)
        self._ram: list[np.ndarray] = []
        self._gx: list[int] = []
        self._n = 0
        self._fired = False

    def push(self, ram) -> bool:
        """Feed one RAM snapshot; returns True once the confluence has fired
        (and stays True thereafter -- the clear is a latching event)."""
        if self._fired:
            return True
        self._ram.append(np.asarray(ram, dtype=np.uint8))
        self._gx.append(int(self._progress(ram)))
        if len(self._ram) > self.window:
            self._ram.pop(0)
            self._gx.pop(0)
        self._n += 1
        if self._n % self.stride != 0 or len(self._ram) < 16:
            return False
        hist = np.stack(self._ram)
        gx = np.array(self._gx, dtype=np.int64)
        tally = 1 if score_tally_windows(hist) else 0
        coord = 1 if coord_entity_windows(hist, gx) else 0
        if tally + coord >= self.min_signals:
            self._fired = True
        return self._fired


# ===========================================================================
# Episode driver -- ties the four signals together over one replay
# ===========================================================================

def run_episode(env, game, bitmasks, dir_bitmask, actions: list[int],
                 start_wd: tuple, margin_actions: int = 90,
                 probe_stride: int = 20, probe_frames: int = LOCK_PROBE_FRAMES
                 ) -> dict:
    """Replays `actions` (already positioned at the root + rooting NOOP),
    then `margin_actions` more of NOOP (each action step is FS=4 raw
    frames, so this is ~margin_actions*FS/60 seconds), collecting
    everything the detector needs. Returns a report with the true clear
    frame (the solver's own forward-clear check, at raw single-frame
    resolution) and the detector's verdict."""
    sample_rate = env.sample_rate
    audio_sig = AudioCadenceSignal(sample_rate)
    lock_track = InputLockTrack(probe_stride)
    ram_hist: list[np.ndarray] = []
    gx_hist: list[int] = []

    true_clear_frame = None
    frame_idx = -1

    def observe_frame():
        nonlocal true_clear_frame, frame_idx
        frame_idx += 1
        ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        audio = env.get_audio()
        audio_sig.push_frame(audio)
        ram_hist.append(ram)
        gx_hist.append(game.progress(ram))
        if true_clear_frame is None and (game.is_clear(start_wd, ram)
                                          or game.is_finale(start_wd, ram)):
            true_clear_frame = frame_idx
        if frame_idx % probe_stride == 0:
            locked, n_diff = differential_input_lock_probe(env, dir_bitmask, probe_frames)
            lock_track.record(frame_idx, locked, n_diff)

    all_actions = list(actions) + [0] * margin_actions
    for a in all_actions:
        mask = int(bitmasks[a])
        for _ in range(FS):
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


def run_ground_truth_test(run_bases: list[str], verbose: bool = True) -> dict:
    bitmasks = action_space_to_bitmasks(ACTION_SPACE)
    dir_bitmask = bitmasks[ACTION_SPACE.index(["right"])]
    game = SmbGame()

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

        env = nes_core.NESEnvironment(game.rom, frame_skip=1)
        env.reset()
        env.set_audio_output_enabled(True)
        env.load_state(root_bytes)
        for _ in range(FS):   # rooting convention: one NOOP = FS raw frames
            env.step(0)

        ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        start_wd = tuple(game.level_key(ram0))
        expected_start = tuple(meta["start_wd"])
        if start_wd != expected_start:
            per_run.append({"run": base, "error":
                             f"root replay mismatch: got {start_wd}, expected {expected_start}"})
            env.close()
            continue

        t0 = time.time()
        report = run_episode(env, game, bitmasks, dir_bitmask, actions, start_wd)
        elapsed = time.time() - t0
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

    n_valid = sum(1 for r in per_run if "error" not in r)
    n_hit = sum(1 for r in per_run if r.get("within_tolerance"))
    total_fp = sum(r.get("false_positive_crossings", 0) for r in per_run if "error" not in r)
    hit_rate = (n_hit / n_valid) if n_valid else 0.0

    # per-signal contribution tally, over the runs that produced a detection
    signal_fire_counts = {"audio": 0, "tally": 0, "lock": 0, "coord": 0}
    for r in per_run:
        c = r.get("contributions_at_detect")
        if c:
            for k, v in c.items():
                signal_fire_counts[k] += int(v)

    summary = {
        "tolerance_frames": TOLERANCE_FRAMES,
        "n_runs": len(per_run),
        "n_valid": n_valid,
        "n_hit_within_tolerance": n_hit,
        "hit_rate": hit_rate,
        "hit_rate_gate": 0.80,
        "hit_rate_pass": hit_rate >= 0.80,
        "total_false_positive_crossings": total_fp,
        "max_false_positives_per_level_gate": 1,
        "false_positive_gate_pass": all(r.get("false_positive_crossings", 0) <= 1
                                         for r in per_run if "error" not in r),
        "signal_fire_counts_at_detect": signal_fire_counts,
        "weights": WEIGHTS,
        "threshold": THRESHOLD,
        "per_run": per_run,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true",
                     help="run the ground-truth self-test against SMB solution traces")
    ap.add_argument("--runs", nargs="*", default=None,
                     help="explicit solution basenames (without .json/.actions.npy); "
                          "default = a curated 5-run SMB set")
    ap.add_argument("--discover", action="store_true",
                     help="glob-discover all mario/smb solution traces instead of the "
                          "curated default set")
    ap.add_argument("--out", default=str(REPO / "runs" / "clear_detect_receipt.json"))
    args = ap.parse_args()

    if not args.test:
        print(__doc__)
        return 0

    if args.runs:
        run_bases = args.runs
    elif args.discover:
        run_bases = discover_smb_solutions()
        print(f"[clear_detect] discovered {len(run_bases)} candidate solution traces")
    else:
        run_bases = DEFAULT_RUNS

    summary = run_ground_truth_test(run_bases)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n[clear_detect] hit rate {summary['hit_rate']:.2f} "
          f"({summary['n_hit_within_tolerance']}/{summary['n_valid']}) "
          f"gate>=0.80: {'PASS' if summary['hit_rate_pass'] else 'FAIL'}")
    print(f"[clear_detect] false positives total={summary['total_false_positive_crossings']} "
          f"gate<=1/level: {'PASS' if summary['false_positive_gate_pass'] else 'FAIL'}")
    print(f"[clear_detect] receipt written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
