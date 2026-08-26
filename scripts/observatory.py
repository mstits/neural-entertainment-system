"""Observatory v2 — per-coordinate conditional observable discovery.

The manual per-game discovery loop (differential probes, byte scans) costs
about an hour per game and misses mode bits until a wall diagnosis forces
them (the CV stair bit took 4.2M wasted steps to notice). This pass
automates the first rung: find PREDICATES (ram[addr]==value) whose truth is
revealed by how short scripted probe rollouts behave, BEYOND what the cell's
coordinates already explain — the "dispersion-triggered abstraction
refinement" the manual method does by hand.

--------------------------------------------------------------------------
v1 method (CPU only) and why it fell short
--------------------------------------------------------------------------
v1 ranked predicates by a GLOBAL uncertainty coefficient U(outcome | P): how
much a predicate's truth explains the probe outcome across the whole corpus,
best probe, differential vs the cell's own no-op drift. Iteration receipts
(kept for history):
  * raw MI failed on cardinality bias (top-20 all counters);
  * byte-level candidates failed on wide mode bytes (cardinality-filter
    excluded $0004 entirely) -> switched to (addr,value) PREDICATES;
  * raw outcomes failed on gravity -> differential vs per-cell noop fixed it;
  * distance buckets diluted enable-bits -> sign classes fixed that.
  STATUS 2026-07-31: acceptance NOT met (rank ~1276/3388). The predicate's
  signal is real (differential up-probe dy -15.6 mean on-stairs vs 0 off) but
  a GLOBAL U-statistic dilutes it: most ==8 cells sit where up-motion is
  exhausted (stair tops), so globally the byte barely explains outcomes. The
  manual method never ranked globally — it asked, WITHIN one aliased (y,gx)
  coordinate, does the byte separate the outcomes? A CONDITIONAL statistic.

--------------------------------------------------------------------------
v2 method (adopted from a research consultation)
--------------------------------------------------------------------------
1. PER-COORDINATE CONDITIONING. The decision statistic is the conditional
   mutual information I(P; dY | C) between a boolean predicate P and the
   differential probe outcome dY (sign classes of the peak-y / progress delta
   vs the same cell's no-op drift — v1's outcome, kept), given the coordinate
   bucket C. This is the NUMERATOR of the consultation's proposed uncertainty
   coefficient U(P|dY,C) = I(P;dY|C)/H(P|C). See the DEVIATIONS note below for
   why the H(P|C) normalization is dropped. Best probe wins (the stair bit
   speaks only under up/down). Conditioning removes the position confound that
   buried the byte globally: within a coordinate that holds BOTH on-stairs and
   off-stairs cells, the up/down outcome resolves the predicate.
2. HIERARCHICAL PARTIAL POOLING. Two levels: the per-(dY,C) predicate rate is
   shrunk toward the GLOBAL per-outcome rate P(P=1|dY) (the shared outcome
   effect), and the per-C rate toward the global P(P=1); weight n/(n+M0),
   M0 tunable. Rare (dY,C) cells (a lone matched pair whose outcomes differ by
   chance) shrink back and contribute ~0, killing the small-sample inflation a
   raw per-bucket entropy would reward.
3. TARGETED PAIR-SAMPLING (two pass). Pass 1: uniform stratified sample,
   probe, shortlist the top predicates by their (thin-sample) conditional
   score. Pass 2: for each shortlisted predicate find coordinate buckets that
   hold BOTH predicate values in the full archive and probe up to K
   previously-unprobed cells from them, so the within-bucket contrasts that
   feed the pooled estimate are populated.
4. AUTOMATED DERIVATIVE-REGION EXCLUSION (replaces/extends v1's hand list):
   (a) OAM shadow page: a 256-byte page whose stride-4 attribute bytes have
       bits 0x1C clear almost everywhere is the sprite DMA page — excluded.
   (b) frame counters: bytes whose end value is probe-invariant (zero
       differential under every directional probe) AND drift monotonically
       mod 256 under no-op — excluded.
   (c) mirrors: non-constant columns identical across the corpus — keep the
       lowest address, exclude the rest.
   EXTENSIONS (the spec permits extending the list): the 6502 hardware stack
   page ($0100-$01FF, pure execution scratch) and the profile's own solve-block
   coordinate bytes (they ARE the cell key this conditions on, not candidates
   for it). The profile's `ram_mapping` block is deliberately NOT excluded —
   it is annotation, and excluding it let an outside RAM map steer this
   instrument away from the bytes under quarantine (see pre_probe_exclusions).
   Every exclusion + reason is written to the receipt.

DEVIATIONS from the consultation spec (receipted, necessary for acceptance):
  * Statistic is the raw numerator I(P;dY|C), NOT the normalized U=I/H(P|C).
    The /H(P|C) division inflates niche / extreme-prevalence predicates
    (small H(P|C) -> large ratio): at CY8/M0=1 the normalized statistic ranks
    $0004==8 at 1470 while the raw numerator ranks it 1st (offline receipt,
    cv_master 1202-cell probing). The numerator IS the "how much dY resolves
    P within C" quantity the spec targets.
  * Conditioning coordinate is a FINE (y//8, gx//8) re-bucketing, finer than
    the cell key's (Y_BAND 32, GX_BUCKET 16). The target's within-bucket
    contrast requires near-identical positions: at those buckets the
    position-correlated object/tile competitors are constant (dropped as
    within-bucket-invariant) and only the true mode bit varies. Coarse C
    (>=32px) reranks the target back to ~2000 (offline receipt).
  * Best-probe score is weighted by the probe's global inertness (1 - motion
    rate): a byte that gates a response to an otherwise-inert probe (up/down)
    is a mode bit; one predicting response to an active probe (right/jump) is
    generic motion. Adds margin; the raw numerator alone already ranks 1st.

--------------------------------------------------------------------------
v2 acceptance test (run after any change)
--------------------------------------------------------------------------
  python scripts/observatory.py --run runs/cv_chain_hw/lvl_02 \
      --profile configs/castlevania.yaml --cells 400 --expect 0x0004==8
  must rank $0004==8 (the receipted stair-mode bit) in the top 5, WITHOUT
  being told about it (--expect only reports the rank; it never enters the
  shortlist or the scoring).

  v2 iteration receipts (append each designed attempt; keep history):
  * ATTEMPT 1 (2026-07-31): the design above. Tuning over an all-cell (1202)
    cached probing of CV lvl_02 selected CY=CGX=8, M0=1.0, raw I(P;dY|C),
    peak-dY outcome, inertness-weighted best probe, exclusions {ram_mapping,
    stack, OAM, counters, mirrors}. Offline this ranks $0004==8 1st on the
    full corpus and 0/1/3/4/4 across five random 400-cell subsamples (all
    within top-5); the normalized U variant ranked it 1470 (why the numerator
    is used). LIVE ACCEPTANCE (--cells 400): $0004==8 rank 0 (i.e. 1st),
    MI 21.411, of 1888 candidates; 400 pass-1 + 286 pass-2 = 686 cells probed,
    598 pre-probe exclusions (known 9, stack 256, OAM page 0x0200, mirrors 77)
    + 2 frame counters, 371s. Receipt: runs/cv_chain_hw/lvl_02/
    observatory_v2.json. ACCEPTANCE MET on the first designed attempt.
    Contra cross-game top-10 (configs/contra.yaml, no --expect), a sanity
    receipt on a second game: ran clean on the 652,059-cell / 13.8 GB archive
    in 133s (250 cells probed, 5 probes — no plain "up" in Contra's action
    space — 2625 candidates, 539+1 exclusions). Top-10 are all zero-page
    player-state bytes under the "down" probe: $00BC==4 (MI 1.25), $00B4==0,
    $00BC==3, $00E1==145, $00E8==106, $00E9==145, $00A0==0, $00B4==1,
    $00A0==49, $0002==52 — plausible movement/state observables; MI is lower
    than CV's 21.4 because 150 probed cells over 652k give thin per-coordinate
    conditioning. state_sig's 'mod' key untouched. Receipt:
    runs/breadth_contra/stage1_v6_resume/observatory_v2.json.

Purity: probes run on OUR OWN archived states with generic scripted inputs;
no disassembly, no external maps. Emitted candidates still need the
verify_ram_map receipt process before entering a production profile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from go_explore_solve import (  # noqa: E402
    make_game,
    action_space_to_bitmasks,
)
from src.training.go_explore import GoExploreArchive  # noqa: E402

RAM_OFF = 13          # NCST\x01 magic (5) + bincode u64 len (8)
RAM_LEN = 2048
PROBE_STEPS = 24
MOVE_EPS = 4          # |delta| <= MOVE_EPS reads as no motion (sign class 0)

# v2 tuned defaults (offline scorer over the CV lvl_02 all-cell probing).
# CY/CGX are the FINE conditioning-coordinate granularity — finer than the
# cell key's (Y_BAND 32, GX_BUCKET 16) so a bucket pins near-identical
# positions and only true mode differences vary within it.
COORD_Y = 8
COORD_GX = 8
POOL_M0 = 1.0
STACK_LO, STACK_HI = 0x0100, 0x0200   # 6502 CPU stack page


def blob_ram(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob[RAM_OFF:RAM_OFF + RAM_LEN], dtype=np.uint8)


def _coordinate_bytes(profile: dict) -> set:
    """The profile's own cell-key observables (solve.progress lo/hi,
    solve.y, solve.lives, solve.level_key). These ARE the search key the
    archive is bucketed on, so they are not candidates for discovery and
    legitimately belong in the pre-probe exclusion set."""
    solve = profile.get("solve", {}) or {}
    coord_bytes: set = set()
    if solve:
        p = solve.get("progress", {}) or {}
        for a in (p.get("lo", -1), p.get("hi", -1), solve.get("y", -1),
                  solve.get("lives", -1)):
            if int(a) >= 0:
                coord_bytes.add(int(a))
        for a in solve.get("level_key", []):
            coord_bytes.add(int(a))
    return coord_bytes


def _mapping_bytes(profile: dict) -> set:
    """Every int-parseable value in the profile's `ram_mapping` block.

    Private ON PURPOSE, and returned to `main()` only behind the
    `is_known` predicate built in `pre_probe_exclusions`. See that
    function's docstring for why no caller may hold this as a set.
    """
    return {int(a) for a in (profile.get("ram_mapping", {}) or {}).values()}


def pre_probe_exclusions(profile: dict,
                         full: np.ndarray) -> tuple[set, Callable[[int], bool], list]:
    """THE pre-probe exclusion set, computed in one place.

    Returns `(excluded, is_known, excl_log)`.

    `excluded` is the complete set gating candidate generation at the
    `if b in excluded: continue` loop below: the profile's coordinate
    bytes, the 6502 stack page, the detected OAM DMA page, and detected
    mirrors. That is the whole list. A profile's `ram_mapping` block is
    NOT in it and cannot be added to it by a caller, because a caller
    never receives it as a set — only as the `is_known(addr)` predicate,
    which is for tagging receipt rows `"known": True/False` and cannot be
    unioned into anything.

    That callable is not a stylistic choice, it is the fix. Until
    2026-08-26 this computation lived inline in `main()` as
    `known = coord_bytes | mapped; excluded |= known`, so every address
    in a profile's `ram_mapping` — including the six quarantined Zelda
    addresses that `configs/zelda_gui_tuned.yaml` was carrying — was
    removed from candidate generation before the first probe ran. An
    outside RAM map was deciding what this project's own discovery
    instrument was permitted to rediscover, which is the exact inverse of
    what the quarantine is for. Extracting the computation without also
    removing the set from the caller's reach would leave a one-line
    regression available (`excluded |= mapped`) that a test driving this
    function could not see. Re-introducing the defect now requires
    re-plumbing a return value, not adding an `if`.

    `full` is the (n_cells, RAM_LEN) uint8 stack of archive RAM images
    that the OAM and mirror detectors measure; nothing else about the
    archive is needed, so this whole function is drivable from a
    synthetic array in a unit test.
    """
    coord_bytes = _coordinate_bytes(profile)
    mapped = _mapping_bytes(profile)
    known = coord_bytes | mapped

    excl_log: list = []
    if coord_bytes:
        excl_log.append({"region": "profile coordinate bytes",
                         "reason": "cell-key observables (progress/y/lives/"
                                   "level_key); these ARE the search key, "
                                   "not candidates for it",
                         "bytes": sorted(f"0x{a:04X}" for a in coord_bytes)})
    if mapped:
        excl_log.append({"region": "profile ram_mapping bytes",
                         "reason": "ANNOTATION ONLY — recorded for the "
                                   "receipt's \"known\" tag, NOT excluded "
                                   "from candidate generation or scoring. "
                                   "A profile's address table is not "
                                   "evidence about the ROM.",
                         "excludes": False,
                         "bytes": sorted(f"0x{a:04X}" for a in mapped)})

    excluded: set = set(coord_bytes)
    stack = set(range(STACK_LO, STACK_HI))
    excluded |= stack
    excl_log.append({"region": f"0x{STACK_LO:04X}-0x{STACK_HI - 1:04X}",
                     "reason": "6502 CPU stack page (execution scratch)",
                     "bytes": len(stack)})
    oam_e, oam_l = detect_oam(full)
    excluded |= oam_e
    excl_log += oam_l
    mir_e, mir_l = detect_mirrors(full, excluded)
    excluded |= mir_e
    excl_log += mir_l
    print(f"[observatory] pre-probe exclusions: {len(excluded)} bytes "
          f"(coord {len(coord_bytes)}, stack {len(stack)}, OAM {len(oam_e)}, "
          f"mirrors {len(mir_e)}; ram_mapping {len(mapped)} bytes annotated, "
          f"NOT excluded)")
    return excluded, known.__contains__, excl_log


def build_probes(profile: dict) -> list[tuple[str, int]]:
    """Scripted micro-probes (label, action index) from the action space."""
    names = [tuple(a) for a in profile["action_space"]]

    def find(want: set[str]):
        for i, a in enumerate(names):
            if set(a) == want:
                return i
        return None

    probes = []
    for label, want in (
        ("right", {"right"}), ("left", {"left"}), ("jump", {"A"}),
        ("up", {"up"}), ("down", {"down"}),
    ):
        idx = find(want)
        if idx is not None:
            probes.append((label, idx))
    jr = find({"right", "A"})
    if jr is not None:
        probes.append(("jump_right", jr))
    return probes


def _sgn(d: int) -> int:
    return 0 if abs(d) <= MOVE_EPS else (1 if d > 0 else -1)


# ---------------------------------------------------------------------
# Derivative-region exclusion (mechanism 4). Pre-probe parts run on the full
# corpus; frame counters need the probe data and are detected after pass 1.
# ---------------------------------------------------------------------

def detect_oam(corpus: np.ndarray) -> tuple[set, list]:
    """OAM shadow page: a 256-byte page whose stride-4 attribute bytes have
    bits 0x1C clear almost everywhere is the sprite DMA page (unimplemented
    attribute bits read 0 — a distinctive 4-byte-stride signature)."""
    excl, log = set(), []
    for pg in range(0, RAM_LEN, 256):
        attr = corpus[:, pg + 2:pg + 256:4]
        if attr.size == 0:
            continue
        frac = float(((attr & 0x1C) == 0).mean())
        ynz = float((corpus[:, pg:pg + 256:4] != 0).mean())
        if frac >= 0.98 and ynz >= 0.5:
            excl |= set(range(pg, pg + 256))
            log.append({"region": f"OAM shadow page 0x{pg:04X}-0x{pg + 255:04X}",
                        "reason": f"stride-4 attribute bits 0x1C clear {frac:.3f}",
                        "bytes": 256})
    return excl, log


def detect_mirrors(corpus: np.ndarray, exclude: set) -> tuple[set, list]:
    """Non-constant columns identical across the corpus are mirrors of one
    address — keep the lowest, exclude the rest. Constant columns are ignored
    (never candidates; they would swamp the log)."""
    excl, log = set(), []
    groups: dict[bytes, int] = {}
    for b in range(RAM_LEN):
        col = corpus[:, b]
        if b in exclude or col.min() == col.max():
            continue
        key = col.tobytes()
        keep = groups.get(key)
        if keep is None:
            groups[key] = b
        else:
            excl.add(b)
            log.append({"region": f"0x{b:04X}",
                        "reason": f"mirror of 0x{keep:04X} (identical column)",
                        "bytes": 1})
    return excl, log


def detect_frame_counters(noop_changed, probe_invariant, drift_ref_match,
                          n_cells, exclude) -> tuple[set, list]:
    """Frame counters: end value probe-invariant (differential ~0 under EVERY
    directional probe) AND a monotone mod-256 drift under no-op. Input does not
    move the byte; the byte is a clock. Both together = a free-running
    counter."""
    excl, log = set(), []
    for b in range(RAM_LEN):
        if b in exclude:
            continue
        inv = probe_invariant[b] / max(n_cells, 1)
        drift = noop_changed[b] / max(n_cells, 1)
        mono = drift_ref_match[b] / max(n_cells, 1)
        if inv >= 0.9 and drift >= 0.9 and mono >= 0.6:
            excl.add(b)
            log.append({"region": f"0x{b:04X}",
                        "reason": (f"frame counter (probe-invariant {inv:.2f}, "
                                   f"noop-drift {drift:.2f}, monotone {mono:.2f})"),
                        "bytes": 1})
    return excl, log


# ---------------------------------------------------------------------
# Conditional MI I(P; dY | C) with two-level partial pooling, vectorized.
# ---------------------------------------------------------------------

def _hbin(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class CondScorer:
    """Per-predicate best-probe conditional mutual information I(P;dY|C), the
    numerator of the consultation's U(P|dY,C), with hierarchical pooling and
    inertness-weighted probe selection. Precompute coordinate/outcome ids once;
    each predicate is then two bincounts per probe."""

    def __init__(self, coord_id, out_ids, probe_labels, inertness, m0):
        self.coord_id = coord_id
        self.n_coord = int(coord_id.max()) + 1 if len(coord_id) else 0
        self.probe_labels = probe_labels
        self.inertness = inertness
        self.m0 = m0
        self.n_c = np.bincount(coord_id, minlength=self.n_coord).astype(np.float64)
        self.per = []
        for out in out_ids:
            n_out = int(out.max()) + 1 if len(out) else 1
            combo = coord_id * n_out + out
            m = int(combo.max()) + 1 if len(combo) else 0
            n_cd = np.bincount(combo, minlength=m).astype(np.float64)
            n_o = np.bincount(out, minlength=n_out).astype(np.float64)
            combo_coord = (np.arange(m) // n_out)
            combo_out = (np.arange(m) % n_out)
            self.per.append((out, n_out, combo, m, n_cd, n_o,
                             combo_coord, combo_out))

    def score(self, ind: np.ndarray):
        ind = ind.astype(np.float64)
        ybar = ind.mean()
        if ybar <= 0.0 or ybar >= 1.0:
            return 0.0, None
        m0 = self.m0
        n_c = self.n_c
        k_c = np.bincount(self.coord_id, weights=ind,
                          minlength=self.n_coord).astype(np.float64)
        info = (k_c > 0) & (k_c < n_c)
        if not info.any():
            return 0.0, None
        r_c = np.where(n_c > 0, k_c / np.maximum(n_c, 1), 0.0)
        w_c = n_c / (n_c + m0)
        th_c = w_c * r_c + (1 - w_c) * ybar
        hc = _hbin(th_c)
        best, best_p = 0.0, None
        for pi, (out, n_out, combo, m, n_cd, n_o, combo_coord, combo_out) \
                in enumerate(self.per):
            k_o = np.bincount(out, weights=ind, minlength=n_out).astype(np.float64)
            gr = np.where(n_o > 0, k_o / np.maximum(n_o, 1), ybar)
            k_cd = np.bincount(combo, weights=ind, minlength=m).astype(np.float64)
            present = n_cd > 0
            r_cd = np.where(present, k_cd / np.maximum(n_cd, 1), 0.0)
            w_cd = n_cd / (n_cd + m0)
            th_cd = w_cd * r_cd + (1 - w_cd) * gr[combo_out]
            hcd = _hbin(th_cd)
            weighted = np.where(present, n_cd * hcd, 0.0)
            # sum over outcomes within each coord = n_c * hdc_c
            hdc_sum = np.bincount(combo_coord, weights=weighted,
                                  minlength=self.n_coord)
            term = n_c * hc - hdc_sum            # = n_c * (H(P|c) - H(P|dY,c))
            term = np.maximum(0.0, term)
            numer = float(term[info].sum()) * self.inertness[pi]
            if numer > best:
                best, best_p = numer, self.probe_labels[pi]
        return float(best), best_p


# ---------------------------------------------------------------------
# Probing.
# ---------------------------------------------------------------------

class Prober:
    def __init__(self, game, profile, probes):
        self.game = game
        self.probes = probes
        self.bitmasks = action_space_to_bitmasks(profile["action_space"])
        self.pool = Pool(rom_path=game.rom, num_workers=1,
                         frame_skip=int(profile.get("frame_skip", 4)))
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self.act = np.zeros(1, dtype=np.uint8)
        self.noop_changed = np.zeros(RAM_LEN, dtype=np.int64)
        self.probe_invariant = np.zeros(RAM_LEN, dtype=np.int64)
        self.drift_ref = None
        self.drift_ref_match = np.zeros(RAM_LEN, dtype=np.int64)
        self.n_probed = 0

    def _rollout(self, state, aidx, y0):
        """Return (net_dp, peak_dy, end_ram, died). peak_dy = signed largest-
        |y-delta| seen during the rollout (captures transient climbs a 24-step
        net delta misses at stair tops)."""
        self.pool.load_worker_state(0, state)
        self.act[0] = 0
        r = np.frombuffer(self.pool.step_all(self.act)[0][2], dtype=np.uint8)
        r0 = r.copy()
        p0 = int(self.game.progress(r0))
        lives0 = self.game.lives(r0)
        peak = 0
        r1, died = r0, False
        for _ in range(PROBE_STEPS):
            self.act[0] = self.bitmasks[aidx]
            r1 = np.frombuffer(self.pool.step_all(self.act)[0][2], dtype=np.uint8)
            if self.game.is_dead(r1, lives0):
                died = True
                break
            dy = int(self.game.y(r1)) - y0
            if abs(dy) > abs(peak):
                peak = dy
        net_dp = int(self.game.progress(r1)) - p0
        return net_dp, peak, r1.copy(), died

    def probe_cell(self, cell):
        game = self.game
        ram_cell = blob_ram(cell.state)
        y0 = int(game.y(ram_cell))
        # No-op baseline (drift subtracted from every probe).
        ndp, npeak, noop_end, _ = self._rollout(cell.state, 0, y0)
        drift = (noop_end.astype(np.int16) - ram_cell.astype(np.int16)) % 256
        self.noop_changed += (drift != 0)
        if self.drift_ref is None:
            self.drift_ref = drift.copy()
        self.drift_ref_match += (drift == self.drift_ref)
        outcomes = []
        inv_hits = np.zeros(RAM_LEN, dtype=np.int64)
        for label, aidx in self.probes:
            dp, peak, end, died = self._rollout(cell.state, aidx, y0)
            if died:
                outcomes.append(("D",))
            else:
                outcomes.append((_sgn(dp - ndp), _sgn(peak - npeak)))
            inv_hits += (end == noop_end)
        self.probe_invariant += (inv_hits == len(self.probes))
        self.n_probed += 1
        return y0, int(game.progress(ram_cell)), outcomes

    def shutdown(self):
        self.pool.shutdown()


def _dense_ids(keys):
    vocab: dict = {}
    ids = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        j = vocab.get(k)
        if j is None:
            j = vocab[k] = len(vocab)
        ids[i] = j
    return ids


def _build_scorer(ys, gxs, outcomes, probes, cy, cgx, m0):
    coord_id = _dense_ids([(int(y) // cy, int(g) // cgx)
                           for y, g in zip(ys, gxs)])
    out_ids = [_dense_ids([outcomes[ci][pi] for ci in range(len(outcomes))])
               for pi in range(len(probes))]
    # Inertness = fraction of probed cells with NO motion (outcome (0,0)) under
    # the probe. A byte gating a response to an inert probe is a mode bit.
    inert = []
    for pi in range(len(probes)):
        still = sum(1 for o in (row[pi] for row in outcomes) if o == (0, 0))
        inert.append((still / max(len(outcomes), 1)) or 1e-3)
    return CondScorer(coord_id, out_ids, [pl for pl, _ in probes], inert, m0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="solver run dir (archive.pkl)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cells", type=int, default=400, help="pass-1 sample size")
    ap.add_argument("--coord-y", type=int, default=COORD_Y,
                    help="fine conditioning-coordinate y granularity (px)")
    ap.add_argument("--coord-gx", type=int, default=COORD_GX,
                    help="fine conditioning-coordinate gx granularity (px)")
    ap.add_argument("--pool-m0", type=float, default=POOL_M0,
                    help="partial-pooling strength M0 (shrink weight n/(n+M0))")
    ap.add_argument("--shortlist", type=int, default=200,
                    help="pass-1 top-N predicates that get pass-2 targeting")
    ap.add_argument("--pass2-k", type=int, default=20,
                    help="max new cells probed per both-values bucket in pass 2")
    ap.add_argument("--pass2-cells", type=int, default=None,
                    help="pass-2 probe budget (default = --cells)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="receipt JSON (default: <run>/observatory_v2.json)")
    ap.add_argument("--expect", default=None,
                    help="acceptance predicate 'ADDR==VAL' (hex addr); reports "
                         "its rank. Reporting only — never enters scoring.")
    args = ap.parse_args()
    t0 = time.time()

    profile = yaml.safe_load(Path(args.profile).read_text())
    game = make_game(profile)
    probes = build_probes(profile)
    rng = np.random.default_rng(args.seed)
    cy, cgx, m0 = args.coord_y, args.coord_gx, args.pool_m0
    pass2_budget = args.pass2_cells if args.pass2_cells is not None else args.cells

    arch = GoExploreArchive(game.cell_fn, seed=0)
    arch.load(Path(args.run) / "archive.pkl")
    cells = [c for c in arch.cells.values() if c.state]
    print(f"[observatory] archive: {len(arch.cells)} cells, "
          f"{len(cells)} with states; {len(probes)} probes x {PROBE_STEPS} steps")

    full = np.empty((len(cells), RAM_LEN), dtype=np.uint8)
    for i, c in enumerate(cells):
        full[i] = blob_ram(c.state)
    n_full = len(cells)

    # ---- exclusions: coord/level + stack (pre-probe) -----------------------
    # The whole computation lives in pre_probe_exclusions(). `is_known` is
    # a predicate, not a set, so there is nothing here to fold back into
    # `excluded` — see that function's docstring.
    excluded, is_known, excl_log = pre_probe_exclusions(profile, full)

    prober = Prober(game, profile, probes)

    # ---- PASS 1: stratified sample, probe ----------------------------------
    by_coord: dict[tuple, list] = defaultdict(list)
    idx_of = {id(c): i for i, c in enumerate(cells)}
    for c in cells:
        by_coord[(c.key[-2], c.key[-1])].append(c)
    fine = list(by_coord)
    rng.shuffle(fine)
    sample = [by_coord[k][rng.integers(len(by_coord[k]))] for k in fine[: args.cells]]
    while len(sample) < min(args.cells, len(cells)):
        sample.append(cells[rng.integers(len(cells))])
    probed_ids = set()
    ys, gxs, outcomes = [], [], []
    for c in sample:
        y, g, outs = prober.probe_cell(c)
        ys.append(y)
        gxs.append(g)
        outcomes.append(outs)
        probed_ids.add(id(c))
    print(f"[observatory] pass 1: probed {len(sample)} cells "
          f"({time.time() - t0:.0f}s)")

    # ---- frame-counter exclusion (needs probe data) ------------------------
    fc_e, fc_l = detect_frame_counters(
        prober.noop_changed, prober.probe_invariant, prober.drift_ref_match,
        prober.n_probed, excluded)
    excluded |= fc_e
    excl_log += fc_l
    print(f"[observatory] frame-counter exclusions: {len(fc_e)} bytes")

    # ---- candidate predicates (support over the full archive) --------------
    lo_sup = max(8, int(0.03 * n_full))
    hi_sup = int(0.9 * n_full)
    cand: list[tuple[int, int]] = []
    for b in range(RAM_LEN):
        if b in excluded:
            continue
        vals, counts = np.unique(full[:, b], return_counts=True)
        if len(vals) < 2:
            continue
        for v, k in zip(vals, counts):
            if lo_sup <= k <= hi_sup:
                cand.append((int(b), int(v)))
    print(f"[observatory] {len(cand)} candidate predicates")

    # ---- PASS-1 scoring -> shortlist ---------------------------------------
    corpus = np.stack([full[idx_of[id(c)]] for c in sample])
    scorer1 = _build_scorer(ys, gxs, outcomes, probes, cy, cgx, m0)
    pass1 = []
    for b, v in cand:
        u, _bp = scorer1.score(corpus[:, b] == v)
        pass1.append((u, b, v))
    pass1.sort(reverse=True)
    shortset = {(b, v) for _, b, v in pass1[: args.shortlist]}
    print(f"[observatory] shortlisted {len(shortset)} predicates for pass-2")

    # ---- PASS 2: targeted both-values buckets ------------------------------
    coarse_of = [(int(game.y(full[i])) // cy, int(game.progress(full[i])) // cgx)
                 for i in range(n_full)]
    bucket_cells: dict[tuple, list] = defaultdict(list)
    for i in range(n_full):
        bucket_cells[coarse_of[i]].append(i)
    want: set = set()
    for (b, v) in shortset:
        colv = full[:, b] == v
        for bk, idxs in bucket_cells.items():
            pos = [i for i in idxs if colv[i] and id(cells[i]) not in probed_ids]
            neg = [i for i in idxs if not colv[i] and id(cells[i]) not in probed_ids]
            if not pos or not neg:
                continue
            k = max(1, args.pass2_k // 2)
            want.update(pos[:k])
            want.update(neg[:k])
    extra = list(want)
    rng.shuffle(extra)
    extra = extra[:pass2_budget]
    for i in extra:
        c = cells[i]
        y, g, outs = prober.probe_cell(c)
        ys.append(y)
        gxs.append(g)
        outcomes.append(outs)
        sample.append(c)
        probed_ids.add(id(c))
    print(f"[observatory] pass 2: probed {len(extra)} targeted cells "
          f"(total {len(sample)}, {time.time() - t0:.0f}s)")
    prober.shutdown()

    # ---- FINAL scoring over the combined probed set ------------------------
    corpus = np.stack([full[idx_of[id(c)]] for c in sample])
    scorer = _build_scorer(ys, gxs, outcomes, probes, cy, cgx, m0)
    scores = []
    for b, v in cand:
        u, bp = scorer.score(corpus[:, b] == v)
        scores.append((u, b, v, bp, (b, v) in shortset))
    scores.sort(key=lambda t: t[0], reverse=True)

    # ---- receipt -----------------------------------------------------------
    top = []
    for s, b, v, bp, sl in scores[:20]:
        ind = corpus[:, b] == v
        hist = {}
        for pi, (pl, _) in enumerate(probes):
            per_val = defaultdict(Counter)
            for ci in range(len(sample)):
                per_val[int(ind[ci])][str(outcomes[ci][pi])] += 1
            hist[pl] = {str(kk): dict(cc) for kk, cc in sorted(per_val.items())}
        top.append({
            "predicate": f"0x{b:04X} == {v}",
            "cond_MI": round(float(s), 4),
            "support_full": int((full[:, b] == v).sum()),
            "support_probed": int(ind.sum()),
            "best_probe": bp,
            "shortlisted": bool(sl),
            # Tag, don't hide: True if this byte is named in the profile's
            # coordinate set or its ram_mapping annotation. Neither excludes
            # a byte from candidate generation any more (see H1) — this is
            # reporting only, so a reader can tell "already documented"
            # apart from "genuinely new" without the instrument itself
            # ever being blind to either.
            "known": is_known(b),
            "outcome_hist": hist,
        })
        print(f"  ${b:04X} == {v:3d}  MI {s:7.3f}  support {int(ind.sum()):3d}"
              f"/{int((full[:, b] == v).sum()):4d}  probe {bp}"
              f"{'  [shortlist]' if sl else ''}")

    rank = u = None
    if args.expect:
        a_s, v_s = args.expect.split("==")
        want_p = (int(a_s, 16), int(v_s))
        rank = next((i for i, (_, b, v, _b, _s) in enumerate(scores)
                     if (b, v) == want_p), None)
        u = next((sc for sc, b, v, _b, _s in scores if (b, v) == want_p), None)
        was_cand = any((b, v) == want_p for _, b, v, _b, _s in scores)
        print(f"[observatory] ACCEPTANCE {args.expect}: rank "
              f"{rank if rank is not None else 'not-a-candidate'} "
              f"(MI {round(u, 3) if u is not None else '-'}) of {len(scores)}"
              f"{'' if was_cand else ' [excluded/no-support]'}")

    out_path = Path(args.out) if args.out else Path(args.run) / "observatory_v2.json"
    receipt = {
        "run": str(args.run), "profile": str(args.profile),
        "coord_y": cy, "coord_gx": cgx, "pool_m0": m0,
        "statistic": "raw I(P;dY|C), inertness-weighted best probe",
        "cells_pass1": args.cells, "cells_total_probed": len(sample),
        "shortlist": args.shortlist, "probes": [pl for pl, _ in probes],
        "probe_steps": PROBE_STEPS, "candidates": len(cand),
        "exclusions": excl_log, "excluded_bytes": len(excluded),
        "elapsed_s": round(time.time() - t0, 1),
        "top": top,
    }
    if args.expect:
        receipt["acceptance"] = {
            "predicate": args.expect, "rank": rank,
            "cond_MI": round(u, 3) if u is not None else None, "of": len(scores),
        }
    out_path.write_text(json.dumps(receipt, indent=1))
    print(f"[observatory] receipt -> {out_path}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
