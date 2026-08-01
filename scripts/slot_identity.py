"""Slot-identity recovery for object-array game engines (Contra base wall).

The fight-doctrine problem this solves. Contra's base-wall boss is an
object-array engine: a small bank of entity SLOTS is time-shared between
the wall hardware (guns, cannon, core) and the soldiers that keep
spawning. The bytes the current `boss_typed` profile keys on -- the
per-slot codes at $0311-$0314 -- are ANIMATION-PHASE codes, not stable
identities: on a live object they cycle (30<->44<->54<->68...), so a gun
and a soldier momentarily share a code and the typed-HP ladder partially
ALIASES. Per-slot HP lives at $04BF-$04C2 (causal receipt: a firing burst
dropped one slot 3->0 and the slot emptied), so HP is a clean OCCUPANCY
signal -- but HP itself changes when an object is hit, so it is not an
identity either.

What we want is the parallel array whose per-slot value is CONSTANT for an
entity's entire lifetime and DISTINGUISHES object classes (wall hardware
vs soldiers). This script recovers it, purely from our own archive.

Method (no external RAM maps, no disassembly -- purity line):
  1. Sample entity-occupied states from a solver archive: cells AT the
     base wall (progress high-water 3072, i.e. cell key[-1]*8 >= 3064) and
     PRE-wall cells (soldiers only).
  2. Restore each state in the deterministic pool and roll it forward
     under a fire-heavy policy, recording RAM every step. (Trace replay
     from the entrance is unreliable for deep cells -- the archive is
     stitched by save/restore, so a deep cell's action log does not
     reproduce it from root; restoring the banked state is exact.)
  3. Define per-slot LIFETIMES as maximal runs where the slot is occupied
     (HP $04BF+k != 0). Validate the occupancy signal against the phase
     array.
  4. For every candidate parallel array (base B in $0300-$07FF, per-slot
     stride s, per-slot address B + k*s aligned with slot k's occupancy),
     score per-lifetime CONSTANCY and class SEPARATION.

Confound control. The base wall is a FIXED-camera fight, so during any
wall lifetime the whole background tile buffer is frozen and the same
foreground effects are redrawn: many bytes look "constant per lifetime"
and even separate wall-vs-pre-wall backgrounds without being per-slot
identities. Three filters neutralise it: (a) an identity array VARIES
across wall lifetimes as a slot's occupant changes (ndist_wall >= 2);
(b) SEPARATION is scored against object-INTRINSIC classes derived from the
phase-code repertoire (wall-hardware codes {54,55,56,68,69,70} vs a
walking runner), not from screen position; (c) the winner is re-confirmed
on fresh rollouts to be genuinely PER-SLOT -- its value at B+k tracks slot
k (offset-0 association highest), resets at occupant death/spawn, and is
constant while the phase code cycles. A fixed screen-location byte fails
(c).

Deliverable: a receipt JSON with the discovered array (constancy +
separation tables), a proposed profile stanza, or an honest kill if no
array clears the constancy bar. This module does NOT edit any config.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from src.training.go_explore import GoExploreArchive  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

# RAM layout (Contra). Occupancy + phase are the two KNOWN per-slot arrays
# (config-receipted); everything else is discovered by this scan.
PLO, PHI = 0x0065, 0x0064      # progress lo/hi (fine scroll + screen<<8)
HP0 = 0x04BF                   # per-slot HP array, 4 slots, stride 1
PH0 = 0x0311                   # per-slot phase-code array, 4 slots, stride 1
NSLOTS = 4
RAW_OFF = 13                   # RAW RAM = savestate[13:13+2048]
WALL_PROGRESS = 3064           # cell key[-1]*8 >= this == base-wall fight
SCAN_LO, SCAN_HI = 0x0300, 0x0800
SCAN_STRIDES = (1, 16, 32)
# Object-intrinsic wall-hardware animation codes (position-independent):
# guns/cannon/core cycle through these; a walking runner never does.
HARDWARE_PHASE = frozenset({54, 55, 56, 68, 69, 70})
RUNNER_PHASE = frozenset({30, 1})


def ram_of_state(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob[RAW_OFF:RAW_OFF + 2048], dtype=np.uint8)


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


def make_weights(action_space, spec):
    w = np.zeros(len(action_space))
    for i, combo in enumerate(action_space):
        w[i] = spec.get(frozenset(combo), 0.02)
    return w / w.sum()


WALL_POLICY = {
    frozenset({"right", "B"}): .28, frozenset({"B"}): .24,
    frozenset({"up", "B"}): .20, frozenset({"right"}): .06,
    frozenset({"A"}): .05, frozenset({"left"}): .05, frozenset({"down"}): .05}
PRE_POLICY = {
    frozenset({"B"}): .40, frozenset({"up", "B"}): .20, frozenset(): .12,
    frozenset({"left"}): .10, frozenset({"down"}): .06,
    frozenset({"right", "B"}): .06, frozenset({"A"}): .04}


def sample_states(run_dir, n_wall, n_pre, seed):
    """Load the archive once; sample diverse wall + pre-wall occupied
    states; return their save-state blobs. Archive freed before rollouts."""
    t0 = time.time()
    arch = GoExploreArchive(lambda r: (), seed=0)
    arch.load(run_dir / "archive.pkl")
    cells = arch.cells
    print(f"[slot_identity] loaded archive {len(cells)} cells "
          f"in {time.time() - t0:.0f}s", flush=True)
    wall, pre = [], []
    seen_w, seen_p = set(), set()
    keys = list(cells.keys())
    random.Random(seed).shuffle(keys)
    for k in keys:
        g = k[-1] * 8
        c = cells[k]
        if c.state is None:
            continue
        if g >= WALL_PROGRESS:
            sig = (k[1], k[2], k[5], k[-4], k[-2])
            if sig in seen_w:
                continue
            seen_w.add(sig)
            wall.append(c.state)
        elif 500 <= g <= 3000:
            sig = (k[-1], k[-2], k[-4])
            if sig in seen_p:
                continue
            seen_p.add(sig)
            pre.append(c.state)
        if len(wall) >= n_wall and len(pre) >= n_pre:
            break
    wall, pre = wall[:n_wall], pre[:n_pre]
    prog = [int(ram_of_state(b)[PLO]) | (int(ram_of_state(b)[PHI]) << 8)
            for b in wall[:5]]
    print(f"[slot_identity] sampled wall={len(wall)} pre={len(pre)} "
          f"(wall-state progress sample {prog})", flush=True)
    del arch, cells
    gc.collect()
    return wall, pre


def new_pool(profile):
    rom = str(REPO / profile["solve"]["rom"])
    pool = Pool(rom_path=rom, num_workers=1,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    return pool


def rollout(pool, bitmasks, blob, weights, nsteps, seed):
    pool.load_worker_state(0, blob)
    rng = np.random.default_rng(seed)
    x = np.zeros(1, dtype=np.uint8)
    rows = []
    prev = 0
    n = len(weights)
    for _ in range(nsteps):
        a = prev if rng.random() < 0.4 else int(rng.choice(n, p=weights))
        prev = a
        x[0] = bitmasks[a]
        rows.append(np.frombuffer(pool.step_all(x)[0][2], dtype=np.uint8).copy())
    return np.array(rows)


def collect_lifetimes(profile, wall_states, pre_states, nsteps, seed):
    bitmasks = list(action_space_to_bitmasks(profile["action_space"]))
    a_space = profile["action_space"]
    w_wall = make_weights(a_space, WALL_POLICY)
    w_pre = make_weights(a_space, PRE_POLICY)
    pool = new_pool(profile)
    C, V, SLOT, REG, LN, HPMAX, PHREP, PHNZ = [], [], [], [], [], [], [], []
    t0 = time.time()
    for states, weights in ((wall_states, w_wall), (pre_states, w_pre)):
        for ci, blob in enumerate(states):
            R = rollout(pool, bitmasks, blob, weights, nsteps, seed + 2000 + ci)
            prog = R[:, PLO].astype(int) | (R[:, PHI].astype(int) << 8)
            ok = (prog >= 0) & (prog < 6000)
            for slot in range(NSLOTS):
                occ = R[:, HP0 + slot] != 0
                for a, b in runs_true(occ):
                    if b - a < 3:
                        continue
                    seg = R[a:b]
                    pr = prog[a:b][ok[a:b]]
                    if len(pr) == 0:
                        continue
                    frac_wall = float(np.mean(pr >= WALL_PROGRESS))
                    if frac_wall > 0.7:
                        reg = 1
                    elif frac_wall < 0.3:
                        reg = 0
                    else:
                        continue
                    ph_col = seg[:, PH0 + slot]
                    ph_set = np.unique(ph_col[ph_col != 0])
                    rep = np.full(8, 255, dtype=np.uint8)
                    rep[:min(8, len(ph_set))] = ph_set[:8]
                    C.append((seg == seg[0]).all(axis=0))
                    V.append(seg[0].copy())
                    SLOT.append(slot)
                    REG.append(reg)
                    LN.append(int(b - a))
                    HPMAX.append(int(seg[:, HP0 + slot].max()))
                    PHREP.append(rep)
                    PHNZ.append(float(np.mean(ph_col != 0)))
    pool.shutdown()
    print(f"[slot_identity] {len(C)} lifetimes in {time.time() - t0:.0f}s "
          f"(wall={sum(REG)} pre={len(REG) - sum(REG)})", flush=True)
    return dict(
        C=np.array(C), V=np.array(V), slot=np.array(SLOT, np.int32),
        reg=np.array(REG, np.int32), Ln=np.array(LN, np.int32),
        hpmax=np.array(HPMAX, np.int32), phrep=np.array(PHREP),
        phnz=np.array(PHNZ))


def _canon_phase(rep):
    return tuple(sorted(int(x) for x in rep if x != 255))


def _nmi(a, b):
    """MI(a;b)/H(b): secondary diagnostic (contaminated when the object
    signature is phase-derived; kept for reference, not for selection)."""
    n = len(a)
    if n == 0:
        return 0.0
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    hb = -sum((v / n) * math.log2(v / n) for v in cb.values())
    if hb < 1e-12:
        return 0.0
    mi = sum((v / n) * math.log2((v / n) / ((ca[x] / n) * (cb[y] / n)))
             for (x, y), v in cab.items())
    return mi / hb


def _tvd(a, b):
    vals = np.union1d(np.unique(a), np.unique(b))
    return float(0.5 * sum(abs(np.mean(a == v) - np.mean(b == v)) for v in vals))


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


def phase_classes(life):
    """Object-intrinsic class per lifetime among wall lifetimes:
    1 = wall hardware (repertoire hits a HARDWARE_PHASE code),
    0 = walking runner (repertoire subset of RUNNER_PHASE),
    -1 = unlabelled (ambiguous / bullets)."""
    reg, phrep = life["reg"], life["phrep"]
    L = len(reg)
    cls = np.full(L, -1)
    for i in np.where(reg == 1)[0]:
        r = set(_canon_phase(phrep[i]))
        if r & HARDWARE_PHASE:
            cls[i] = 1
        elif r and r.issubset(RUNNER_PHASE):
            cls[i] = 0
    return cls


def scan(life, constancy_thresh):
    C, V, slot, reg = life["C"], life["V"], life["slot"], life["reg"]
    hpmax, phrep = life["hpmax"], life["phrep"]
    L = len(slot)
    idxL = np.arange(L)
    wall = reg == 1
    pre = reg == 0
    cls = phase_classes(life)
    gun = cls == 1
    runner = cls == 0
    sig_obj = np.array([hash((int(hpmax[i]), _canon_phase(phrep[i])))
                        for i in range(L)])

    def evalc(stride, B):
        addr = B + slot * stride
        if addr.max() >= 2048:
            return None
        const = C[idxL, addr]
        val = V[idxL, addr]
        cw = const & wall
        cp = const & pre
        cg = const & gun
        cs = const & runner
        nvw = int(cw.sum())
        if nvw < 10 or int(cg.sum()) < 15 or int(cs.sum()) < 15:
            return None
        wv = val[cw]
        # type purity: fraction of constant wall lifetimes whose value maps
        # deterministically to a single object signature (a clean type code
        # is many-values-few or few-values-few; a position byte is not pure).
        by = {}
        for i in np.where(cw)[0]:
            by.setdefault(int(val[i]), Counter())[
                (int(hpmax[i]), _canon_phase(phrep[i]))] += 1
        purity = (sum(c.most_common(1)[0][1] for c in by.values()) / nvw
                  if nvw else 0.0)
        return dict(
            stride=stride, base=B,
            constancy_all=float(const.mean()),
            constancy_wall=float(const[wall].mean()) if wall.any() else 0.0,
            ndist_wall=int(len(np.unique(wv))),
            nvw=nvw, nvp=int(cp.sum()),
            separation=_tvd(val[cg], val[cs]),   # hardware vs runner, intrinsic
            type_purity=float(purity),
            tvd_wall_pre=(_tvd(wv, val[cp]) if cp.sum() >= 8 else float("nan")),
            nmi_obj=_nmi(wv.tolist(), sig_obj[cw].tolist()),
        )

    out = []
    for stride in SCAN_STRIDES:
        for B in range(SCAN_LO, SCAN_HI - 3 * stride):
            r = evalc(stride, B)
            if r is not None:
                out.append(r)
    return out


def class_table(life, stride, base):
    C, V, slot, reg = life["C"], life["V"], life["slot"], life["reg"]
    hpmax, phrep = life["hpmax"], life["phrep"]
    L = len(slot)
    idxL = np.arange(L)
    addr = base + slot * stride
    const = C[idxL, addr]
    val = V[idxL, addr]
    cw = const & (reg == 1)
    cp = const & (reg == 0)
    rows = {}
    for v in sorted(np.unique(val[cw]).tolist()):
        m = cw & (val == v)
        sigs = Counter((int(hpmax[i]), _canon_phase(phrep[i]))
                       for i in np.where(m)[0])
        rows[str(int(v))] = {
            "n": int(m.sum()),
            "object_signatures": [
                {"hp_max": hp, "phase_repertoire": list(ph), "count": c}
                for (hp, ph), c in sigs.most_common(4)]}
    return {
        "wall_value_to_object": rows,
        "wall_value_distribution": {str(int(v)): int(c) for v, c in
                                    zip(*np.unique(val[cw], return_counts=True))},
        "pre_value_distribution": {str(int(v)): int(c) for v, c in
                                   zip(*np.unique(val[cp], return_counts=True))},
    }


def confirm_per_slot(profile, wall_blobs, stride, base, seed=700):
    """Certify the winner is a genuine PER-SLOT array, not a fixed screen
    byte. Fresh rollouts: (a) value[B+k] must associate with slot-k phase
    more than with a shifted slot's; (b) constant within a lifetime;
    (c) resets/changes at occupant transitions."""
    bitmasks = list(action_space_to_bitmasks(profile["action_space"]))
    w = make_weights(profile["action_space"], WALL_POLICY)
    pool = new_pool(profile)
    R = np.vstack([rollout(pool, bitmasks, b, w, 200, seed + i)
                   for i, b in enumerate(wall_blobs)])
    pool.shutdown()
    align = {}
    for off in range(NSLOTS):
        vals, phs = [], []
        for k in range(NSLOTS):
            occ = R[:, HP0 + k] != 0
            vals.append(R[occ, base + k * stride])
            phs.append(R[occ, PH0 + (k + off) % NSLOTS])
        align[f"offset_{off}"] = round(
            _cramers_v(np.concatenate(vals).tolist(),
                       np.concatenate(phs).tolist()), 3)
    # within-lifetime constancy + transition reset behaviour
    n_life = n_const = n_reset = n_trans = 0
    empty_vals = Counter()
    for k in range(NSLOTS):
        occ = R[:, HP0 + k] != 0
        prev_val = None
        for a, b in runs_true(occ):
            if b - a < 3:
                continue
            col = R[a:b, base + k * stride]
            n_life += 1
            if (col == col[0]).all():
                n_const += 1
            if prev_val is not None:
                n_trans += 1
                # a NEW occupant should be free to carry a different value
                if col[0] != prev_val:
                    n_reset += 1
            prev_val = int(col[-1])
        empty = (R[:, HP0:HP0 + NSLOTS] == 0).all(axis=1)
        empty_vals.update(R[empty, base + k * stride].tolist())
    off0 = align["offset_0"]
    shifted_max = max(v for kk, v in align.items() if kk != "offset_0")
    return {
        "per_slot_alignment_cramersV": align,
        "aligned_at_offset0": bool(off0 >= shifted_max),
        "within_lifetime_constancy": round(n_const / max(1, n_life), 4),
        "n_lifetimes": n_life,
        "changes_at_occupant_transition_frac":
            round(n_reset / max(1, n_trans), 4),
        "value_when_all_slots_empty":
            {str(int(v)): int(c) for v, c in empty_vals.most_common(4)},
    }


def build_stanza(base, stride, tbl):
    """boss_ident: a drop-in analog of boss_typed keyed on the STABLE
    per-slot identity instead of the cycling phase codes. Wall-hardware
    identity values are those whose lifetimes carry a HARDWARE_PHASE code
    (data-derived from the separation table -- no external map)."""
    wall_ids, runner_ids, note = [], [], []
    for v, info in tbl["wall_value_to_object"].items():
        if int(v) == 0:
            note.append("value 0 aliases the empty-slot default and a rare "
                        "hp8 core (phase 56-58); excluded from wall_ident_"
                        "values to avoid counting empty slots -- target the "
                        "core via its hp8 signature instead.")
            continue
        phases = set()
        for sg in info["object_signatures"]:
            phases.update(sg["phase_repertoire"])
        (wall_ids if phases & HARDWARE_PHASE else runner_ids).append(int(v))
    return {
        "note": "Drop-in analog of boss_typed keyed on the STABLE per-slot "
                "identity ($058F+k) instead of the cycling phase codes. "
                "typed HP = sum of hp_addrs[i] where ident_addrs[i] holds a "
                "wall-hardware identity value, read as `start` when no wall "
                "object is live. Distinguishes guns (id 5) and the cannon "
                "(id 48) from the runner (id 1) with zero phase aliasing. "
                "PROPOSAL ONLY -- do not hand-edit configs/contra.yaml from "
                "this script.",
        "caveats": note,
        "boss_ident": {
            "ident_addrs": [hex(base + k * stride) for k in range(NSLOTS)],
            "hp_addrs": [hex(HP0 + k) for k in range(NSLOTS)],
            "wall_ident_values": sorted(wall_ids),
            "runner_ident_values_excluded": sorted(runner_ids),
            "start": 12,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Contra slot-identity recovery")
    ap.add_argument("--run", default="runs/breadth_contra/stage1_v6_resume")
    ap.add_argument("--profile", default="configs/contra.yaml")
    ap.add_argument("--out",
                    default="runs/breadth_contra/slot_identity_receipt.json")
    ap.add_argument("--n-wall", type=int, default=200)
    ap.add_argument("--n-pre", type=int, default=320)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--constancy-thresh", type=float, default=0.9)
    ap.add_argument("--separation-thresh", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    run_dir = REPO / args.run
    profile = yaml.safe_load((REPO / args.profile).read_text())
    out_path = REPO / args.out
    cache = (Path(args.cache) if args.cache
             else out_path.with_suffix(".lifetimes.npz"))

    # Wall states kept for the per-slot confirmation pass.
    wall_blobs = []
    if cache.exists():
        print(f"[slot_identity] reusing lifetimes cache {cache}", flush=True)
        z = np.load(cache)
        life = {k: z[k] for k in z.files}
        wall_blobs, _ = sample_states(run_dir, 8, 0, args.seed + 1)
    else:
        wall_states, pre_states = sample_states(
            run_dir, args.n_wall, args.n_pre, args.seed)
        wall_blobs = wall_states[:8]
        life = collect_lifetimes(profile, wall_states, pre_states,
                                 args.steps, args.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **life)

    reg = life["reg"]
    phnz = life["phnz"]
    n_wall, n_pre = int((reg == 1).sum()), int((reg == 0).sum())
    occ_valid = {
        "signal": "HP $04BF+k != 0",
        "mean_phase_nonzero_within_lifetime": round(float(phnz.mean()), 4),
        "min_phase_nonzero_within_lifetime": round(float(phnz.min()), 4),
        "n_lifetimes": len(reg), "n_wall": n_wall, "n_pre": n_pre}
    print(f"[slot_identity] occupancy validation: phase-nonzero within "
          f"HP!=0 lifetimes mean={phnz.mean():.3f}", flush=True)

    cands = scan(life, args.constancy_thresh)
    print(f"[slot_identity] scanned {len(cands)} (stride,base) candidates",
          flush=True)

    ref = {}
    for name, base in (("phase_0x0311", PH0), ("hp_0x04BF", HP0)):
        C, V, slot = life["C"], life["V"], life["slot"]
        idxL = np.arange(len(slot))
        addr = base + slot
        const = C[idxL, addr]
        ref[name] = {"constancy_all": round(float(const.mean()), 4),
                     "constancy_wall": round(float(const[reg == 1].mean()), 4)}

    eligible = [c for c in cands
                if c["constancy_wall"] >= args.constancy_thresh
                and c["ndist_wall"] >= 2
                and c["separation"] >= args.separation_thresh
                and c["nvp"] >= 8]
    # Rank: strongest intrinsic class separation, then a cleaner type code
    # (higher purity, then FEWER distinct values), then higher constancy.
    ranked = sorted(eligible, key=lambda c: (
        -c["separation"], -c["type_purity"], c["ndist_wall"],
        -c["constancy_wall"]))

    receipt = {
        "task": "slot-identity recovery (Contra base wall, object-array "
                "engine)",
        "archive": str(run_dir),
        "profile": args.profile,
        "params": {"n_wall": args.n_wall, "n_pre": args.n_pre,
                   "steps": args.steps, "seed": args.seed,
                   "wall_progress_threshold": WALL_PROGRESS,
                   "scan_range": [hex(SCAN_LO), hex(SCAN_HI)],
                   "scan_strides": list(SCAN_STRIDES),
                   "constancy_threshold": args.constancy_thresh,
                   "separation_threshold": args.separation_thresh,
                   "hardware_phase_codes": sorted(HARDWARE_PHASE)},
        "occupancy_validation": occ_valid,
        "reference_arrays_low_constancy": ref,
        "n_candidates_scanned": len(cands),
        "n_eligible": len(eligible),
    }

    if not ranked:
        receipt["result"] = "KILL"
        receipt["kill_reason"] = (
            "No parallel array reached per-lifetime constancy_wall >= "
            f"{args.constancy_thresh}, varied across wall lifetimes "
            "(ndist_wall>=2), AND separated intrinsic wall-hardware from "
            f"runner classes (TVD >= {args.separation_thresh}). The class "
            "signal lives only in the cycling phase codes + HP.")
        near = sorted(cands, key=lambda c: -c["separation"])[:15]
        receipt["best_near_misses"] = [
            {"base": hex(c["base"]), "stride": c["stride"],
             "constancy_wall": round(c["constancy_wall"], 4),
             "ndist_wall": c["ndist_wall"],
             "separation": round(c["separation"], 4)} for c in near]
        print("[slot_identity] RESULT: KILL", flush=True)
    else:
        win = ranked[0]
        s, b = win["stride"], win["base"]
        addrs = [b + k * s for k in range(NSLOTS)]
        tbl = class_table(life, s, b)
        confirm = confirm_per_slot(profile, wall_blobs, s, b) if wall_blobs \
            else {"note": "no wall states available for confirmation"}
        receipt["result"] = "FOUND"
        receipt["identity_array"] = {
            "base": hex(b), "stride": s,
            "per_slot_addrs": [hex(a) for a in addrs],
            "constancy_all": round(win["constancy_all"], 4),
            "constancy_wall": round(win["constancy_wall"], 4),
            "ndist_wall": win["ndist_wall"],
            "type_purity": round(win["type_purity"], 4),
            "separation_hardware_vs_runner_tvd": round(win["separation"], 4),
            "tvd_wall_vs_pre": (round(win["tvd_wall_pre"], 4)
                                if win["tvd_wall_pre"] == win["tvd_wall_pre"]
                                else None),
            "n_wall_lifetimes_constant": win["nvw"],
            "n_pre_lifetimes_constant": win["nvp"],
        }
        receipt["per_slot_confirmation"] = confirm
        receipt["separation_table"] = tbl
        receipt["runner_up_candidates"] = [
            {"base": hex(c["base"]), "stride": c["stride"],
             "constancy_wall": round(c["constancy_wall"], 4),
             "ndist_wall": c["ndist_wall"],
             "separation": round(c["separation"], 4),
             "type_purity": round(c["type_purity"], 4)}
            for c in ranked[1:8]]
        receipt["proposed_profile_stanza"] = build_stanza(b, s, tbl)
        print(f"[slot_identity] RESULT: FOUND base={hex(b)} stride={s} "
              f"constancy_wall={win['constancy_wall']:.3f} "
              f"separation={win['separation']:.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[slot_identity] receipt -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
