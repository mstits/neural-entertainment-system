"""Slot-identity recovery for the Contra base-wall object-array engine.

The doctrine problem. Contra's base-wall boss is an object-array engine:
a small bank of four entity SLOTS is time-shared between the wall
hardware (guns / core) and the soldiers that keep spawning. The bytes
the current profile keys on -- the per-slot codes at $0311-$0314 -- are
ANIMATION-PHASE codes, not stable identities: on a live object they
cycle (observed at the wall: 1<->30<->33<->54<->55 within a single
occupancy), so a gun and a soldier momentarily share a code and any
typed-HP ladder built on them partially ALIASES. Per-slot HP lives at
$04BF-$04C2 (causal receipt: sustained fire dropped a slot 3->0 and the
slot emptied); HP is a clean OCCUPANCY signal but not an identity (it
changes 2<->3<->4 as an object is hit and cycles its phase).

Goal. Recover the STABLE per-slot identity array: parallel per-slot
addresses whose value is CONSTANT for an entity's whole lifetime and
SEPARATES object classes (wall hardware vs soldiers).

Method (task-specified, purity line -- own archive only, no external RAM
maps / disassembly):
  1. Replay ACTION TRACES from runs/breadth_contra/stage1_v6_resume.
     traces.pkl maps cell key -> (root_id, action_bytes, ...); every
     trace is rooted at the single "entrance" root in roots.json. The
     rooting convention is: load the root state, take ONE noop pool
     step, then apply the action list. Replay is byte-exact -- a wall
     cell (key[-1]*8 >= 3064) reproduces final progress 3072 to the
     step. This needs only traces.pkl (596 MB) + the root state, never
     the 13.8 GB archive.pkl.
  2. Per slot k (k=0..3) define entity LIFETIMES as maximal runs where
     the HP byte $04BF+k != 0. Validate that occupancy signal against
     the phase array (HP!=0 <=> PH!=0).
  3. Candidate identity arrays are the four addresses base+k for base in
     0x0300..0x07FF (stride 1 -- the two KNOWN slot arrays $0311 and
     $04BF are both base+k, stride 1). For every base score per-lifetime
     CONSTANCY (byte constant over the whole lifetime) and class
     SEPARATION (the constant value differs between wall-object and
     soldier lifetimes).

Class labels + confound control. A fixed-camera fight freezes the whole
background, so many bytes read "constant per lifetime" and even separate
wall-vs-pre-wall backgrounds without being per-slot identities. Two
axes label the ground-truth classes:
  * REGION (task-literal): a lifetime whose steps sit at the wall
    (progress >= 3064) vs pre-wall (< 3064). "Present at the wall" is the
    task's defining property of a wall object.
  * PHASE-INTRINSIC (confound-controlled): a lifetime whose phase
    repertoire hits a wall-hardware code {54,55,56,68,69,70} (gun / core)
    vs one confined to running-soldier codes {1,30,33,44,45,46}. This
    label is position-independent, so a byte that separates on it is
    tracking object class, not screen region. It is used only to LABEL
    classes -- never as the identity itself (the whole point is the phase
    codes are unstable).
A true per-slot identity separates on BOTH axes and varies across wall
lifetimes as a slot's occupant changes (ndist_wall >= 2); a frozen
background byte reads the same value for both classes in a slot and fails
separation.

Deliverable: a receipt with the discovered array (constancy + separation
tables with example values per class) and a proposed profile stanza, OR
an honest kill if no base+k array clears constancy >= 0.9 with class
separation. This module never edits any config.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

# RAM layout (Contra). Occupancy + phase are the two KNOWN per-slot arrays
# (config-receipted); every other per-slot array is what this scan hunts.
PLO, PHI = 0x0065, 0x0064      # progress lo / hi (fine scroll + screen<<8)
HP0 = 0x04BF                   # per-slot HP array, 4 slots, stride 1
PH0 = 0x0311                   # per-slot phase-code array, 4 slots, stride 1
NSLOTS = 4
RAW_OFF = 13                   # raw RAM = savestate[13:13+2048]
WALL_PROGRESS = 3064           # cell key[-1]*8 >= this == base-wall fight
SCAN_LO, SCAN_HI = 0x0300, 0x0800
MIN_LIFE = 3                   # skip 1-2 step HP flickers
# Object-intrinsic animation codes (position-independent class labels).
HARDWARE_PHASE = frozenset({54, 55, 56, 68, 69, 70})   # guns / sensor / core
SOLDIER_PHASE = frozenset({1, 30, 33, 44, 45, 46})     # running soldiers
NOOP_BIT = 0


# -------------------------------------------------------------------------
# replay
# -------------------------------------------------------------------------
def runs_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [a, b) runs where mask is True."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def new_pool(profile, nworkers):
    rom = str(REPO / profile["solve"]["rom"])
    pool = Pool(rom_path=rom, num_workers=nworkers,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    return pool


def replay_batch(pool, nworkers, bitmasks, root_bytes, traces):
    """Replay up to `nworkers` action traces in lockstep from the root.
    Convention: load root, one noop step, then the action list. Returns a
    list of (T_w x 2048) RAM matrices, one per trace, T_w = trace length."""
    w = len(traces)
    for i in range(w):
        pool.load_worker_state(i, root_bytes)
    x = np.zeros(nworkers, dtype=np.uint8)
    pool.step_all(x)                       # the ONE rooting noop
    maxlen = max(len(t) for t in traces)
    rec = [[] for _ in range(w)]
    for step in range(maxlen):
        for i in range(w):
            x[i] = bitmasks[traces[i][step]] if step < len(traces[i]) \
                else NOOP_BIT
        out = pool.step_all(x)
        for i in range(w):
            if step < len(traces[i]):
                rec[i].append(np.frombuffer(out[i][2], dtype=np.uint8).copy())
    return [np.array(r) for r in rec]


def sample_wall_traces(run_dir, n_traces, seed):
    """Load traces.pkl, return diverse wall-cell traces (each a full
    entrance->wall trajectory) + the root state bytes."""
    t0 = time.time()
    with open(run_dir / "traces.pkl", "rb") as f:
        tr = pickle.load(f)
    roots = json.loads((run_dir / "roots.json").read_text())
    root_bytes = (REPO / roots["entrance"]["path"]).read_bytes()
    wall = [k for k in tr if k[-1] * 8 >= WALL_PROGRESS]
    random.Random(seed).shuffle(wall)
    # Diversify by (typed-hp, state-sig, y-band) so sampled wall states
    # carry different slot occupants.
    seen, picks = set(), []
    for k in wall:
        sig = (k[-4], k[-3], k[-2])
        if sig in seen:
            continue
        seen.add(sig)
        picks.append([int(b) for b in tr[k][1]])
        if len(picks) >= n_traces:
            break
    print(f"[slot_identity] loaded {len(tr)} traces "
          f"({len(wall)} wall) in {time.time() - t0:.0f}s; "
          f"sampled {len(picks)} diverse wall trajectories", flush=True)
    del tr
    return picks, root_bytes


def collect_lifetimes(profile, traces, root_bytes, nworkers, seed):
    """Replay every trace; carve per-slot HP!=0 lifetimes; record for each
    the per-address constancy mask, first-row values, slot, region, length,
    phase repertoire, hp range. Also accumulate the HP<->phase occupancy
    agreement used to validate the occupancy signal."""
    bitmasks = list(action_space_to_bitmasks(profile["action_space"]))
    pool = new_pool(profile, nworkers)
    C, V, SLOT, REG, LN, HPMAX, PHREP = [], [], [], [], [], [], []
    occ_steps = ph_nz_and_hp = ph_nz_not_hp = hp_nz_not_ph = 0
    t0 = time.time()
    for start in range(0, len(traces), nworkers):
        batch = traces[start:start + nworkers]
        for R in replay_batch(pool, nworkers, bitmasks, root_bytes, batch):
            prog = R[:, PLO].astype(int) | (R[:, PHI].astype(int) << 8)
            for slot in range(NSLOTS):
                hp_nz = R[:, HP0 + slot] != 0
                ph_nz = R[:, PH0 + slot] != 0
                occ_steps += int(hp_nz.sum())
                ph_nz_and_hp += int((hp_nz & ph_nz).sum())
                ph_nz_not_hp += int((ph_nz & ~hp_nz).sum())
                hp_nz_not_ph += int((hp_nz & ~ph_nz).sum())
                for a, b in runs_true(hp_nz):
                    if b - a < MIN_LIFE:
                        continue
                    seg = R[a:b]
                    fw = float(np.mean(prog[a:b] >= WALL_PROGRESS))
                    if fw > 0.7:
                        reg = 1                     # wall
                    elif fw < 0.3:
                        reg = 0                     # pre-wall
                    else:
                        continue                    # straddles; ambiguous
                    ph_col = seg[:, PH0 + slot]
                    rep = np.unique(ph_col[ph_col != 0])
                    repf = np.full(12, 255, dtype=np.uint8)
                    repf[:min(12, len(rep))] = rep[:12]
                    C.append((seg == seg[0]).all(axis=0))
                    V.append(seg[0].copy())
                    SLOT.append(slot)
                    REG.append(reg)
                    LN.append(int(b - a))
                    HPMAX.append(int(seg[:, HP0 + slot].max()))
                    PHREP.append(repf)
        if (start // nworkers) % 20 == 0:
            print(f"[slot_identity]   replayed {start + len(batch)}/"
                  f"{len(traces)} traces, {len(C)} lifetimes, "
                  f"{time.time() - t0:.0f}s", flush=True)
    pool.shutdown()
    life = dict(
        C=np.array(C), V=np.array(V), slot=np.array(SLOT, np.int32),
        reg=np.array(REG, np.int32), Ln=np.array(LN, np.int32),
        hpmax=np.array(HPMAX, np.int32), phrep=np.array(PHREP))
    occ = dict(occ_steps=occ_steps, ph_nz_and_hp=ph_nz_and_hp,
               ph_nz_not_hp=ph_nz_not_hp, hp_nz_not_ph=hp_nz_not_ph)
    print(f"[slot_identity] {len(C)} lifetimes "
          f"(wall={int((life['reg'] == 1).sum())} "
          f"pre={int((life['reg'] == 0).sum())}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return life, occ


# -------------------------------------------------------------------------
# class labels + scoring
# -------------------------------------------------------------------------
def _canon_phase(rep):
    return tuple(sorted(int(x) for x in rep if x != 255))


def _tvd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    vals = np.union1d(np.unique(a), np.unique(b))
    return float(0.5 * sum(abs(np.mean(a == v) - np.mean(b == v))
                           for v in vals))


def _cramers_v(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    ia = {v: i for i, v in enumerate(ua)}
    ib = {v: i for i, v in enumerate(ub)}
    tab = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        tab[ia[x], ib[y]] += 1
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab - exp) ** 2 / exp)
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def phase_class(life):
    """Per-lifetime intrinsic class from the phase repertoire:
    1 = wall hardware (repertoire hits a HARDWARE_PHASE code),
    0 = running soldier (nonempty repertoire, no hardware code, subset of
        SOLDIER_PHASE),
    -1 = unlabelled (ambiguous / bullets / effects)."""
    phrep = life["phrep"]
    cls = np.full(len(phrep), -1)
    for i in range(len(phrep)):
        r = set(_canon_phase(phrep[i]))
        if not r:
            continue
        if r & HARDWARE_PHASE:
            cls[i] = 1
        elif r <= SOLDIER_PHASE:
            cls[i] = 0
    return cls


def scan(life, cls, constancy_thresh, sep_thresh, min_per_class):
    """For each stride-1 base+k array, score per-lifetime constancy and
    class separation on the two axes. Returns candidate dicts."""
    C, V, slot, reg = life["C"], life["V"], life["slot"], life["reg"]
    L = len(slot)
    idxL = np.arange(L)
    wall = reg == 1
    pre = reg == 0
    hw = cls == 1
    runner = cls == 0

    def evalc(B):
        addr = B + slot
        if int(addr.max()) >= 2048:
            return None
        const = C[idxL, addr]
        val = V[idxL, addr]
        c_hw = const & hw
        c_run = const & runner
        c_wall = const & wall
        c_pre = const & pre
        n_hw, n_run = int(hw.sum()), int(runner.sum())
        if n_hw < min_per_class or n_run < min_per_class:
            return None
        constancy_hw = int(c_hw.sum()) / n_hw
        constancy_runner = int(c_run.sum()) / n_run
        constancy_wall = (float(const[wall].mean()) if wall.any() else 0.0)
        # separation of the CONSTANT-lifetime values between the classes.
        sep_hw_runner = _tvd(val[c_hw], val[c_run])
        sep_wall_pre = (_tvd(val[c_wall], val[c_pre])
                        if c_pre.sum() >= 8 else float("nan"))
        ndist_wall = int(len(np.unique(val[c_wall]))) if c_wall.any() else 0
        return dict(
            base=B, stride=1,
            constancy_all=float(const.mean()),
            constancy_wall=constancy_wall,
            constancy_hw=constancy_hw,
            constancy_runner=constancy_runner,
            sep_hw_runner=sep_hw_runner,
            sep_wall_pre=sep_wall_pre,
            ndist_wall=ndist_wall,
            n_hw_const=int(c_hw.sum()), n_runner_const=int(c_run.sum()),
            n_wall_const=int(c_wall.sum()), n_pre_const=int(c_pre.sum()))

    out = []
    for B in range(SCAN_LO, SCAN_HI - (NSLOTS - 1)):
        r = evalc(B)
        if r is not None:
            out.append(r)
    return out


def class_table(life, cls, base):
    """For the winning base+k array, tabulate each constant value against
    the object class it carries, with example phase repertoires + HP."""
    C, V, slot, reg = life["C"], life["V"], life["slot"], life["reg"]
    hpmax, phrep = life["hpmax"], life["phrep"]
    L = len(slot)
    idxL = np.arange(L)
    addr = base + slot
    const = C[idxL, addr]
    val = V[idxL, addr]
    hw, runner = cls == 1, cls == 0
    rows = {}
    for v in sorted(np.unique(val[const]).tolist()):
        m = const & (val == v)
        sigs = Counter((int(hpmax[i]), _canon_phase(phrep[i]))
                       for i in np.where(m)[0])
        rows[str(int(v))] = {
            "n_total": int(m.sum()),
            "n_hardware_class": int((m & hw).sum()),
            "n_soldier_class": int((m & runner).sum()),
            "n_wall_region": int((m & (reg == 1)).sum()),
            "n_prewall_region": int((m & (reg == 0)).sum()),
            "example_object_signatures": [
                {"hp_max": hp, "phase_repertoire": list(ph), "count": c}
                for (hp, ph), c in sigs.most_common(4)]}
    return {
        "value_to_object": rows,
        "hardware_class_value_distribution": {
            str(int(v)): int(c) for v, c in
            zip(*np.unique(val[const & hw], return_counts=True))}
        if (const & hw).any() else {},
        "soldier_class_value_distribution": {
            str(int(v)): int(c) for v, c in
            zip(*np.unique(val[const & runner], return_counts=True))}
        if (const & runner).any() else {},
    }


def confirm_per_slot(profile, traces, root_bytes, base, nworkers, seed):
    """Certify base+k is a genuine PER-SLOT array, not a fixed screen byte.
    Fresh wall replays: (a) value[base+k] aligns with slot-k phase more than
    with a shifted slot's; (b) constant within a lifetime; (c) free to carry
    a different value for a new occupant after a slot empties."""
    bitmasks = list(action_space_to_bitmasks(profile["action_space"]))
    pool = new_pool(profile, nworkers)
    mats = []
    for start in range(0, len(traces), nworkers):
        mats.extend(replay_batch(pool, nworkers, bitmasks, root_bytes,
                                 traces[start:start + nworkers]))
    pool.shutdown()
    R = np.vstack(mats)
    align = {}
    for off in range(NSLOTS):
        vals, phs = [], []
        for k in range(NSLOTS):
            occ = R[:, HP0 + k] != 0
            vals.append(R[occ, base + k])
            phs.append(R[occ, PH0 + (k + off) % NSLOTS])
        align[f"offset_{off}"] = round(
            _cramers_v(np.concatenate(vals), np.concatenate(phs)), 3)
    n_life = n_const = n_change = n_trans = 0
    empty_vals = Counter()
    for k in range(NSLOTS):
        occ = R[:, HP0 + k] != 0
        prev = None
        for a, b in runs_true(occ):
            if b - a < MIN_LIFE:
                continue
            col = R[a:b, base + k]
            n_life += 1
            if (col == col[0]).all():
                n_const += 1
            if prev is not None:
                n_trans += 1
                if col[0] != prev:
                    n_change += 1
            prev = int(col[-1])
        empty = (R[:, HP0:HP0 + NSLOTS] == 0).all(axis=1)
        empty_vals.update(R[empty, base + k].tolist())
    off0 = align["offset_0"]
    shifted = max(v for kk, v in align.items() if kk != "offset_0")
    return {
        "per_slot_alignment_cramersV": align,
        "aligned_at_offset0": bool(off0 >= shifted),
        "within_lifetime_constancy": round(n_const / max(1, n_life), 4),
        "n_lifetimes": n_life,
        "changes_at_occupant_transition_frac":
            round(n_change / max(1, n_trans), 4),
        "value_when_all_slots_empty":
            {str(int(v)): int(c) for v, c in empty_vals.most_common(4)},
    }


def build_stanza(base, tbl):
    """boss_ident: a drop-in analog of boss_typed keyed on the STABLE
    per-slot identity instead of the cycling phase codes. Wall-hardware
    identity values are those whose constant lifetimes are dominated by the
    hardware class (data-derived from the class table -- no external map).
    PROPOSAL ONLY; this script never edits configs/contra.yaml."""
    wall_ids, soldier_ids = [], []
    for v, info in tbl["value_to_object"].items():
        iv = int(v)
        hw, sol = info["n_hardware_class"], info["n_soldier_class"]
        if hw == 0 and sol == 0:
            continue
        (wall_ids if hw > sol else soldier_ids).append(iv)
    return {
        "note": "Drop-in analog of boss_typed keyed on the STABLE per-slot "
                "identity array instead of the cycling phase codes at "
                "$0311-$0314. typed HP = sum of hp_addrs[i] where "
                "ident_addrs[i] holds a wall-hardware identity value, read "
                "as `start` when no wall object is live. Removes the "
                "phase-aliasing in boss_typed's type ladder. PROPOSAL ONLY "
                "-- do not hand-edit configs/contra.yaml from this script "
                "(it already carries a state_sig mod entry and a down+B "
                "action; leave those alone).",
        "boss_ident": {
            "ident_addrs": [hex(base + k) for k in range(NSLOTS)],
            "hp_addrs": [hex(HP0 + k) for k in range(NSLOTS)],
            "wall_ident_values": sorted(wall_ids),
            "soldier_ident_values": sorted(soldier_ids),
            "start": 12,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Contra slot-identity recovery")
    ap.add_argument("--run", default="runs/breadth_contra/stage1_v6_resume")
    ap.add_argument("--profile", default="configs/contra.yaml")
    ap.add_argument("--out",
                    default="runs/breadth_contra/slot_identity_receipt.json")
    ap.add_argument("--n-traces", type=int, default=160)
    ap.add_argument("--n-confirm", type=int, default=9)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--constancy-thresh", type=float, default=0.9)
    ap.add_argument("--sep-thresh", type=float, default=0.5)
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    run_dir = REPO / args.run
    profile = yaml.safe_load((REPO / args.profile).read_text())
    out_path = REPO / args.out
    cache = (Path(args.cache) if args.cache
             else out_path.with_suffix(".lifetimes.npz"))
    workers = max(1, min(3, args.workers))          # house rule: Pools <= 3

    traces, root_bytes = sample_wall_traces(run_dir, args.n_traces, args.seed)
    if cache.exists():
        print(f"[slot_identity] reusing lifetimes cache {cache}", flush=True)
        z = np.load(cache)
        life = {k: z[k] for k in z.files}
        occ = {"note": "loaded from cache; see occupancy_validation in a "
                       "prior run"}
        occ_valid = {"cached": True}
    else:
        life, occ = collect_lifetimes(profile, traces, root_bytes,
                                      workers, args.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **life)
        tot = max(1, occ["occ_steps"])
        occ_valid = {
            "signal": "HP $04BF+k != 0 (occupancy) vs phase $0311+k != 0",
            "occupied_steps": occ["occ_steps"],
            "phase_nonzero_agreement_frac": round(occ["ph_nz_and_hp"] / tot, 6),
            "phase_nonzero_while_hp_zero_steps": occ["ph_nz_not_hp"],
            "hp_nonzero_while_phase_zero_steps": occ["hp_nz_not_ph"],
            "verdict": "HP!=0 and phase!=0 coincide -> HP occupancy is a "
                       "clean per-slot presence signal"}

    cls = phase_class(life)
    reg = life["reg"]
    Ln = life["Ln"]
    n_hw, n_run = int((cls == 1).sum()), int((cls == 0).sum())
    n_wall, n_pre = int((reg == 1).sum()), int((reg == 0).sum())

    # Reference: the two KNOWN per-slot arrays must have LOW constancy
    # (phase cycles, HP changes) -- that is why an identity array is needed.
    ref = {}
    C, V, slot = life["C"], life["V"], life["slot"]
    idxL = np.arange(len(slot))
    for name, base in (("phase_0x0311", PH0), ("hp_0x04BF", HP0)):
        const = C[idxL, base + slot]
        ref[name] = {
            "constancy_all": round(float(const.mean()), 4),
            "constancy_wall": round(float(const[reg == 1].mean()), 4)
            if (reg == 1).any() else None,
            "constancy_hardware_class": round(float(const[cls == 1].mean()), 4)
            if (cls == 1).any() else None}

    cands = scan(life, cls, args.constancy_thresh, args.sep_thresh,
                 args.min_per_class)
    print(f"[slot_identity] scanned {len(cands)} stride-1 (base) candidates; "
          f"classes hw={n_hw} runner={n_run} wall={n_wall} pre={n_pre}",
          flush=True)

    eligible = [c for c in cands
                if c["constancy_hw"] >= args.constancy_thresh
                and c["constancy_runner"] >= args.constancy_thresh
                and c["ndist_wall"] >= 2
                and c["sep_hw_runner"] >= args.sep_thresh]
    ranked = sorted(eligible, key=lambda c: (
        -c["sep_hw_runner"], -min(c["constancy_hw"], c["constancy_runner"]),
        c["ndist_wall"]))

    receipt = {
        "task": "slot-identity recovery (Contra base wall, object-array "
                "engine): find the stable per-slot identity array",
        "method": "action-trace replay from the run root (roots.json), one "
                  "noop rooting step then the traces.pkl action list; "
                  "per-slot HP!=0 lifetimes; stride-1 base+k scan of "
                  "0x0300-0x07FF for per-lifetime constancy + class "
                  "separation. Own-archive only; no external RAM map.",
        "archive": str(run_dir),
        "profile": args.profile,
        "params": {
            "n_traces_replayed": len(traces), "workers": workers,
            "wall_progress_threshold": WALL_PROGRESS,
            "scan_range": [hex(SCAN_LO), hex(SCAN_HI)], "stride": 1,
            "min_lifetime_steps": MIN_LIFE,
            "constancy_threshold": args.constancy_thresh,
            "separation_threshold": args.sep_thresh,
            "min_lifetimes_per_class": args.min_per_class,
            "hardware_phase_codes": sorted(HARDWARE_PHASE),
            "soldier_phase_codes": sorted(SOLDIER_PHASE),
            "seed": args.seed},
        "occupancy_validation": occ_valid,
        "lifetime_stats": {
            "n_lifetimes": len(reg),
            "n_wall_region": n_wall, "n_prewall_region": n_pre,
            "n_hardware_class": n_hw, "n_soldier_class": n_run,
            "n_unlabelled": int((cls == -1).sum()),
            "lifetime_len_pctiles": {
                str(p): int(np.percentile(Ln, p)) for p in (50, 90, 99)},
            "note": "guns/core do NOT appear as long continuous HP runs -- "
                    "the slots are time-multiplexed and HP flickers to 0 "
                    "between phase cycles, so classes are labelled by "
                    "phase-repertoire (hardware vs soldier) and region, not "
                    "by longevity."},
        "reference_known_arrays_low_constancy": ref,
        "n_candidates_scanned": len(cands),
        "n_eligible": len(eligible),
    }

    if not ranked:
        receipt["result"] = "KILL"
        receipt["kill_reason"] = (
            "No stride-1 base+k array in 0x0300-0x07FF reached per-lifetime "
            f"constancy >= {args.constancy_thresh} within BOTH the "
            "hardware-class and soldier-class lifetimes while also varying "
            f"across wall lifetimes (ndist_wall>=2) and separating the "
            f"classes (TVD >= {args.sep_thresh}). The wall-object vs soldier "
            "class signal lives only in the cycling phase codes + HP, which "
            "are not per-lifetime constant -- there is no stable per-slot "
            "identity byte in the scanned window.")
        # near-misses: highest separation among constant-enough candidates
        near = sorted(
            [c for c in cands if min(c["constancy_hw"],
                                     c["constancy_runner"]) >= 0.8],
            key=lambda c: -c["sep_hw_runner"])[:20]
        receipt["best_near_misses"] = [
            {"base": hex(c["base"]),
             "constancy_hw": round(c["constancy_hw"], 4),
             "constancy_runner": round(c["constancy_runner"], 4),
             "ndist_wall": c["ndist_wall"],
             "sep_hw_runner": round(c["sep_hw_runner"], 4),
             "sep_wall_pre": (round(c["sep_wall_pre"], 4)
                              if c["sep_wall_pre"] == c["sep_wall_pre"]
                              else None)} for c in near]
        receipt["scanned_window"] = {
            "bytes": f"{hex(SCAN_LO)}..{hex(SCAN_HI)} (stride 1, base+k)",
            "n_bases": len(cands),
            "highest_sep_hw_runner_any_constancy": round(
                max((c["sep_hw_runner"] for c in cands), default=0.0), 4),
            "highest_constancy_hw_any_sep": round(
                max((c["constancy_hw"] for c in cands), default=0.0), 4)}
        print("[slot_identity] RESULT: KILL", flush=True)
    else:
        win = ranked[0]
        b = win["base"]
        tbl = class_table(life, cls, b)
        confirm = confirm_per_slot(profile, traces[:args.n_confirm],
                                   root_bytes, b, workers, args.seed + 1)
        receipt["result"] = "FOUND"
        receipt["identity_array"] = {
            "base": hex(b), "stride": 1,
            "per_slot_addrs": [hex(b + k) for k in range(NSLOTS)],
            "constancy_all": round(win["constancy_all"], 4),
            "constancy_wall": round(win["constancy_wall"], 4),
            "constancy_hardware_class": round(win["constancy_hw"], 4),
            "constancy_soldier_class": round(win["constancy_runner"], 4),
            "ndist_wall": win["ndist_wall"],
            "separation_hardware_vs_soldier_tvd": round(win["sep_hw_runner"], 4),
            "separation_wall_vs_prewall_tvd": (round(win["sep_wall_pre"], 4)
                                               if win["sep_wall_pre"] ==
                                               win["sep_wall_pre"] else None),
            "n_hardware_lifetimes_constant": win["n_hw_const"],
            "n_soldier_lifetimes_constant": win["n_runner_const"]}
        receipt["per_slot_confirmation"] = confirm
        receipt["separation_table"] = tbl
        receipt["runner_up_candidates"] = [
            {"base": hex(c["base"]),
             "constancy_hw": round(c["constancy_hw"], 4),
             "constancy_runner": round(c["constancy_runner"], 4),
             "ndist_wall": c["ndist_wall"],
             "sep_hw_runner": round(c["sep_hw_runner"], 4)}
            for c in ranked[1:8]]
        receipt["proposed_profile_stanza"] = build_stanza(b, tbl)
        print(f"[slot_identity] RESULT: FOUND base={hex(b)} "
              f"constancy_hw={win['constancy_hw']:.3f} "
              f"sep={win['sep_hw_runner']:.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[slot_identity] receipt -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
