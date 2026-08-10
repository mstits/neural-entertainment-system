"""Lexicographic objective learning over a RAM trace (a learnfun port).

Port of Tom7's `learnfun` (tasbot, SIGBOVIK 2013, "The First Level of
Super Mario Bros. is Easy with Lexicographic Orderings and Time
Travel"), against OUR banked solver tapes:

    objective.{h,cc}           -> Objective / EnumerateFull /
                                  EnumeratePartial / CheckOrdering
    weighted-objectives.{h,cc} -> WeightByExamples (value fraction,
                                  VF(M_last) - VF(M_first)), Observe /
                                  GetNormalizedValue
    learnfun.cc                -> the slicing schedule (whole movie x50,
                                  tenths x3, strided subsets)

WHAT THIS IS FOR, AND WHAT IT MUST NEVER BE USED FOR
----------------------------------------------------
This is a byte-SHORTLISTING instrument. Given a chain of our own solved
tapes it ranks the 2048 RAM locations by how much lexicographic
"progress" weight they carry, so a human (or `emit_solve_yaml`) has a
short list of addresses to adjudicate instead of 2048. That is all it
is licensed to do.

The learned objective MUST NOT be wired as a reward term, an archive
score, a cell key, or any other training/search signal. Measured on our
own tapes (`NEVER_WIRE_AS_REWARD` below): objectives fit on Bubble
Bobble rounds 69..79 score in-sample at Spearman rho +0.999 vs elapsed
time and score the held-out rounds 80..84 at rho -0.771. Off the fit
range it is anti-correlated with progress; optimizing it would actively
train the agent backwards. The SMB world-1 -> world-2 held-out pair is
milder but the same shape (+0.992 in-sample, +0.851 held out with
delta(end-begin) collapsing +0.875 -> +0.038).

Purity: the only input is a 2048-byte RAM trace produced by replaying
our own banked tapes on our own emulator (src.training.tape_replay). No
disassembly, no external RAM maps, no hand-authored addresses enter the
discovery path.

Validation receipts these functions were lifted with, RE-MEASURED
against the productionized replay (2026-08-10; see the note on seams
below for why re-measuring was necessary):

  * SMB 32-tape chain (1-1..8-4, 31590 memories). 15 of the 17 addresses
    in the hand-authored SMB table appear in the ranking; $075F (world)
    is #1 by mass over all 2048 bytes (241.0) at lead-rank 5, ahead of
    $0760 area (146.3) and $07DF score-d1 (143.0). $075A lives takes
    lead-rank 9, $07DE score-d0 lead-rank 7. Concentration: 31 leaders,
    3.85 bits.
  * Bubble Bobble 30-round chain (rounds 69..98, 16077 memories):
    recovers $0401 (the round counter) at lead-rank 28, mass 70.7, while
    the known-bad $0021 scratch byte does not enter the ranking at all.
    Concentration: 61 leaders, 5.47 bits.
  * SINGLE tapes fail. The same BB analysis on one round (67->68, 214
    memories) does not surface $0401 at all. Chains only.
  * Free-running timers outrank true progress ON SHORT TRACES, and this
    module cannot see it. On the SMB 1-1 tape alone the frame-cadence
    byte $07C7 takes lead-rank 6 with the largest mass in the top ten
    (43.7), above every hand-authored byte but $006D. It is
    cadence-invariant — it scores the same on a shuffled tape — so no
    amount of trajectory reasoning rejects it; that is Gate 1's job
    (`scripts/discover_observables._flat_under_noop`). The trap does
    weaken with trace length: over the full 32-tape chain $07C7 falls to
    lead 0.0 at rank 1402 (mass 57.2, still substantial). Never rely on
    that — a fresh game's first chain is short.
  * Total objective weight is NOT a validity gate; the negative control
    inverts it. SMB 1-1 correct: total weight 82.1. Same tape from the
    WRONG root: 205.2 (2.5x MORE). Same tape SHUFFLED: 218.7 (2.7x).
    Use `concentration` / `concentration_verdict` instead, which ranked
    all 7 test trajectories correctly at matched length (correct 1-1: 77
    leaders / 5.01 bits; wrong-root: 99 / 6.07; shuffled: 148 / 6.92).

A note on those numbers, because two of them moved: the proof-of-concept
these functions came from reused one controller buffer across chain
segments, so every seam's "settle" frame replayed the previous segment's
last button (fixed in `tape_replay.replay_segments`). The SMB chain is
bit-for-bit unaffected — it reproduces the PoC's 31 leaders / 3.85 bits
and $075F's 241.03 mass exactly — but the Bubble Bobble chain is not:
$0401 read lead-rank 10 / mass 92.5 before the fix and 28 / 70.7 after.
The conclusion (chains recover the round counter, single tapes do not)
survives; the specific rank did not, which is why the figures above are
the post-fix ones.
"""
from __future__ import annotations

import bisect
import time
from typing import Optional, Sequence

import numpy as np

U64 = (1 << 64) - 1
I32 = 0xFFFFFFFF

#: The one-line rule that travels with any artifact this module produces.
NEVER_WIRE_AS_REWARD = (
    "learnfun objectives are a byte SHORTLIST, never a reward/archive "
    "score: held-out Spearman rho vs time is -0.771 off the fit range "
    "(BB fit 69..79, held out 80..84)."
)

# learnfun.cc MakeObjectives' published counts.
WHOLE_MOVIE_REPS = 50


# --------------------------------------------------------------------------
# Objective enumeration (port of objective.cc)
# --------------------------------------------------------------------------

def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Deterministic 64-bit mix, standing in for Tom7's CompareByHash.

    DEVIATION: CrapHash's rotate-by-(a&3)+1 loop is UB in C++ for the i=0
    rotate and its exact bit pattern carries no meaning — all that matters
    is that candidate order is a deterministic, seed-dependent permutation.
    """
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(U64)
        z = x
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(U64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(U64)
        return z ^ (z >> np.uint64(31))


def _shuffle(v: np.ndarray, seed: int) -> np.ndarray:
    """objective.cc Shuffle(): sort the candidate list by a seeded hash."""
    if v.size <= 1:
        return v
    h = _splitmix64(v.astype(np.uint64) ^ np.uint64(_splitmix64(
        np.array([seed & I32], dtype=np.uint64))[0]))
    return v[np.argsort(h, kind="stable")]


class Objective:
    """Enumerate maximal tight valid lexicographic orderings over a trace.

    An ordering L1..Ln is VALID over a set of observation indices `look`
    when for every consecutive pair (i, j) in look, MEM_i <=_lex MEM_j. It
    is TIGHT (Tom7's "maximal") when no further location can be appended:
    every remaining location either decreases somewhere within a span
    where the prefix is constant, or never moves there at all.
    """

    #: cap on the (rows x n_loc) comparison temporary
    CHUNK = 8192

    def __init__(self, memories: np.ndarray, decrease_tol: float = 0.0):
        self.M = np.ascontiguousarray(memories, dtype=np.uint8)
        self.n_loc = self.M.shape[1]
        self.nodes = 0
        # EXTENSION (not in Tom7): allow a location to decrease on up to
        # this FRACTION of the equal-prefix transitions and still be
        # admitted. tol=0 is the published algorithm exactly. Our own
        # tapes contain single-frame blips in real progress counters
        # (Bubble Bobble's round byte reads 67->66->68 across its clear),
        # which tol=0 rejects outright; the repo's own clear detector
        # debounces the same blip.
        self.decrease_tol = float(decrease_tol)
        self.A = self.B = None

    # -- objective.cc EnumeratePartial ------------------------------------
    def _partial(self, rows: np.ndarray, prefix: list, left: np.ndarray):
        """`rows` indexes the consecutive-pair arrays, already restricted
        to pairs EQUAL ON THE PREFIX. Returns (candidates, remain).

        Rows are carried as INDICES, never as sub-array copies: a 32-level
        SMB chain is 31.6k x 2048 and the recursion runs ~80 deep, so
        copying at every level would be tens of GB. The comparison
        temporaries are chunked so peak scratch stays a few MB regardless
        of trace length.
        """
        n = rows.size
        if n == 0:
            e = np.empty(0, dtype=np.int32)
            return e, e
        n_gt = np.zeros(self.n_loc, dtype=np.int64)
        ever_lt = np.zeros(self.n_loc, dtype=bool)
        for s in range(0, n, self.CHUNK):
            r = rows[s:s + self.CHUNK]
            a, b = self.A[r], self.B[r]
            n_gt += (a > b).sum(axis=0)
            ever_lt |= (a < b).any(axis=0)
        budget = int(self.decrease_tol * n)
        ever_gt = n_gt > budget

        gt, lt = ever_gt[left], ever_lt[left]
        if prefix:
            in_pref = np.isin(left, np.asarray(prefix, dtype=left.dtype))
            gt, lt = gt | in_pref, lt & ~in_pref   # never re-offer the prefix
            keep = ~in_pref
        else:
            keep = True
        return left[(~gt) & lt], left[(gt | lt) & keep]

    # -- objective.cc EnumeratePartialRec ---------------------------------
    def _rec(self, rows, prefix, left, out, limit, seed):
        self.nodes += 1
        cands, remain = self._partial(rows, prefix, left)

        if seed != 0:
            seed = (seed + limit[0] + len(prefix)) & I32
            seed = (seed ^ rows.size) & I32       # look.size() proxy
            cands = _shuffle(cands, seed)

        if cands.size == 0:
            out.append(list(prefix))              # maximal: emit
            if limit[0] > 0:
                limit[0] -= 1
            return

        prefix.append(-1)
        for c in cands:
            prefix[-1] = int(c)
            sub = rows[self.A[rows, c] == self.B[rows, c]]   # equal-span
            self._rec(sub, prefix, remain, out, limit, seed)
            if limit[0] == 0:
                prefix.pop()
                return
        prefix.pop()

    def enumerate_full(self, look: np.ndarray, limit: int, seed: int) -> list:
        """objective.cc EnumerateFull: up to `limit` orderings over `look`."""
        look = np.asarray(look, dtype=np.int32)
        if look.size < 2:
            return []
        lo, hi = int(look[0]), int(look[-1])
        if look.size == hi - lo + 1:          # contiguous -> views, no copy
            self.A, self.B = self.M[lo:hi], self.M[lo + 1:hi + 1]
        else:
            self.A, self.B = self.M[look[:-1]], self.M[look[1:]]
        out: list = []
        self._rec(np.arange(look.size - 1, dtype=np.int64), [],
                  np.arange(self.n_loc, dtype=np.int32), out, [limit], seed)
        self.A = self.B = None
        return [o for o in out if o]

    def enumerate_full_all(self, limit: int, seed: int) -> list:
        """objective.cc EnumerateFullAll: every frame, duplicates dropped."""
        return self.enumerate_full(changing_frames(self.M), limit, seed)


def changing_frames(M: np.ndarray) -> np.ndarray:
    """Row indices of `M` with duplicate consecutive memories dropped."""
    keep = np.ones(M.shape[0], dtype=bool)
    keep[1:] = (M[1:] != M[:-1]).any(axis=1)
    return np.nonzero(keep)[0].astype(np.int32)


def check_ordering(M: np.ndarray, look: np.ndarray, order: Sequence[int]) -> tuple:
    """objective.cc CheckOrdering. Returns (n_violating_pairs, n_pairs).

    A pair violates when the first position where the two memories differ
    goes DOWN. With decrease_tol == 0 a correct port must report zero;
    this is the property test every learned ordering is held to.
    """
    look = np.asarray(look, dtype=np.int64)
    order = list(order)
    if len(order) == 0 or look.size < 2:
        return 0, 0
    A = M[look[:-1]][:, order].astype(np.int16)
    B = M[look[1:]][:, order].astype(np.int16)
    d = B - A
    nz = d != 0
    any_nz = nz.any(axis=1)
    first = np.argmax(nz, axis=1)
    lead = d[np.arange(d.shape[0]), first]
    return int(np.sum(any_nz & (lead < 0))), int(d.shape[0])


# --------------------------------------------------------------------------
# Value-fraction weighting (port of weighted-objectives.cc)
# --------------------------------------------------------------------------

def value_fraction_weight(M: np.ndarray, obj: Sequence[int]) -> tuple:
    """WeightByExamples: VF(M_last) - VF(M_first).

    VF(m) = (index of m's value among the sorted UNIQUE values this
    objective takes across the trace) / (number of unique values).
    Non-positive scores are zeroed — "bad objective lost more than
    gained".

    This is a WEIGHT, used to order objectives against each other on one
    trace. It is not a validity statistic: see `concentration`.
    """
    obj = list(obj)
    k = len(obj)
    sub = np.ascontiguousarray(M[:, obj])
    raw = sub.tobytes()
    rows = [raw[i * k:(i + 1) * k] for i in range(sub.shape[0])]
    uniq = sorted(set(rows))
    n = len(uniq)
    vf_end = bisect.bisect_left(uniq, rows[-1]) / n
    vf_beg = bisect.bisect_left(uniq, rows[0]) / n
    score = vf_end - vf_beg
    return (score if score > 0.0 else 0.0), n, vf_beg, vf_end


# --------------------------------------------------------------------------
# learnfun.cc's slicing schedule
# --------------------------------------------------------------------------

def skip_until_first_input(memories: np.ndarray, masks: np.ndarray) -> tuple:
    """learnfun.cc: "The very beginning of most games start with RAM
    initialization, which we really should ignore." Drop the prefix of the
    trace before the first non-zero controller bitmask.

    Our tapes start from a mid-game entrance, so this is usually a no-op
    or a handful of frames — but the rule is part of the algorithm.
    """
    nz = np.nonzero(masks)[0]
    start = int(nz[0]) if nz.size else 0
    return memories[start:], start


def learn_objectives(memories: np.ndarray, *, whole: int = WHOLE_MOVIE_REPS,
                     verify: bool = True, decrease_tol: float = 0.0) -> dict:
    """learnfun.cc MakeObjectives, with the published counts kept as-is.

    `memories` is an (N, n_loc) uint8 trace — for us, N frames of the
    2048-byte NES RAM produced by `src.training.tape_replay`.

    Returns the weighted, de-duplicated objective set plus the
    self-check: with `decrease_tol == 0` a correct port reports zero
    violating pairs, because every emitted ordering is valid over its OWN
    look set (a tenth-slice ordering is not claimed to hold globally).
    """
    M = np.ascontiguousarray(memories, dtype=np.uint8)
    obj = Objective(M, decrease_tol=decrease_tol)
    T = M.shape[0]
    raw: list = []          # (ordering, look, stage)
    stage_counts: dict = {}

    def emit(stage, look, orderings):
        for o in orderings:
            raw.append((o, look, stage))
        stage_counts[stage] = stage_counts.get(stage, 0) + len(orderings)

    t0 = time.time()
    look_all = changing_frames(M)
    for i in range(whole):
        emit("whole", look_all, obj.enumerate_full(look_all, 1, i))

    onenth = T // 10                                   # GenerateNthSlices(10, 3)
    if onenth >= 2:
        for s in range(10):
            look = np.arange(s * onenth, s * onenth + onenth, dtype=np.int32)
            for i in range(3):
                emit("tenths", look, obj.enumerate_full(look, 1, s * 0xBEAD + i))

    for stride, offsets, num in ((100, 10, 10), (250, 10, 10), (1000, 10, 1)):
        tag = f"stride{stride}"                        # GenerateOccasional
        stage_counts.setdefault(tag, 0)
        for off in range(offsets):
            look = np.arange(off, T, stride, dtype=np.int32)
            if look.size < 2:
                continue
            for i in range(num):
                emit(tag, look, obj.enumerate_full(look, 1, off * 0xF00D + i))
    t_enum = time.time() - t0

    bad = None
    if verify:
        v = [check_ordering(M, lk, o) for o, lk, _ in raw]
        n_bad = sum(1 for nv, npair in v if nv > decrease_tol * npair)
        bad = {"orderings_over_tol": n_bad,
               "violating_pairs": sum(nv for nv, _ in v),
               "total_pairs": sum(np_ for _, np_ in v)}

    # dedup (WeightedObjectives' map<vector<int>, Info*>) then weight
    t0 = time.time()
    uniq: dict = {}
    for o, _, stage in raw:
        rec = uniq.setdefault(tuple(o), {"stages": {}})
        rec["stages"][stage] = rec["stages"].get(stage, 0) + 1
    weighted = []
    for key, rec in uniq.items():
        w, nvals, vb, ve = value_fraction_weight(M, list(key))
        weighted.append({"obj": list(key), "weight": w, "n_values": nvals,
                         "vf_begin": vb, "vf_end": ve,
                         "stages": rec["stages"]})
    weighted.sort(key=lambda r: -r["weight"])
    t_weight = time.time() - t0

    return {
        "n_memories": int(T),
        "n_locations": int(M.shape[1]),
        "n_raw": len(raw), "n_unique": len(uniq),
        "n_positive": sum(1 for r in weighted if r["weight"] > 0),
        "stage_counts": stage_counts,
        "nodes": obj.nodes,
        "invalid_orderings": bad,
        "decrease_tol": float(decrease_tol),
        "t_enumerate_s": t_enum, "t_weight_s": t_weight,
        "objectives": weighted,
    }


# --------------------------------------------------------------------------
# Per-location aggregation
# --------------------------------------------------------------------------

def aggregate_locations(result: dict) -> tuple[dict, dict]:
    """Roll objective weight back onto individual RAM locations.

    Tom7 weights whole ORDERINGS; to compare against a per-byte
    observable table we aggregate three ways:

      mass  -- total weight of objectives containing the location
      lead  -- total weight of objectives where it is the MOST
               SIGNIFICANT byte (position 0)
      disc  -- weight x 2^-position (lexicographic significance discount)

    Returns `(by_location, lead_weight_per_stage)`.
    """
    agg: dict = {}
    per_stage: dict = {}
    for r in result["objectives"]:
        w = r["weight"]
        if w <= 0:
            continue
        for pos, loc in enumerate(r["obj"]):
            a = agg.setdefault(int(loc), {"mass": 0.0, "lead": 0.0,
                                          "disc": 0.0, "n": 0, "best_pos": 99})
            a["mass"] += w
            a["disc"] += w * (0.5 ** pos)
            a["n"] += 1
            a["best_pos"] = min(a["best_pos"], pos)
            if pos == 0:
                a["lead"] += w
        for stage in r["stages"]:
            d = per_stage.setdefault(stage, {})
            head = int(r["obj"][0])
            d[head] = d.get(head, 0.0) + w
    return agg, per_stage


def rank_locations(result: dict, *, by: str = "lead",
                   top: Optional[int] = None,
                   memories: Optional[np.ndarray] = None) -> list[dict]:
    """Ranked per-location records: the SHORTLIST this module exists for.

    `by` is "lead" (default; the most-significant-byte weight, which is
    how $075F and $0401 were recovered), "mass", or "disc". Pass
    `memories` to attach the per-byte singleton VF diagnostic and the
    up/down step counts.

    Ranks are 1-based and always assigned by `by`, so "lead-rank 28"
    (Bubble Bobble's $0401) and "1st by mass" (SMB's $075F) are both
    quotable, and mean different orderings of the same table.
    """
    if by not in ("lead", "mass", "disc"):
        raise ValueError(f"rank by lead|mass|disc, got {by!r}")
    agg, _ = aggregate_locations(result)
    order = sorted(agg.items(), key=lambda kv: (-kv[1][by], kv[0]))
    out = []
    for rank, (loc, a) in enumerate(order, 1):
        rec = {"addr": int(loc), "rank": rank, "lead": a["lead"],
               "mass": a["mass"], "disc": a["disc"], "n_obj": a["n"],
               "best_pos": a["best_pos"]}
        if memories is not None:
            col = memories[:, loc].astype(np.int16)
            d = np.diff(col)
            rec["singleton_vf"] = value_fraction_weight(memories, [loc])[0]
            rec["n_up"] = int((d > 0).sum())
            rec["n_down"] = int((d < 0).sum())
            rec["span"] = int(col.max() - col.min())
        out.append(rec)
    return out[:top] if top else out


def singleton_vf(M: np.ndarray, locs) -> dict:
    """VF weight of each one-byte objective [L]. A cheap per-byte
    monotonicity score; NOT tightness-filtered, diagnostic only."""
    return {int(L): value_fraction_weight(M, [int(L)])[0] for L in locs}


# --------------------------------------------------------------------------
# Concentration — the trajectory-quality statistic (NOT total weight)
# --------------------------------------------------------------------------

def concentration(result_or_agg) -> dict:
    """How much of the most-significant-byte mass sits on ONE location.

    A trajectory that really is progress produces a lead distribution
    dominated by a handful of bytes; a desynced or scrambled trajectory
    scatters it across many. This — NOT total objective weight — is the
    trajectory-validity statistic:

        SMB 1-1 correct     77 leaders, 5.01 bits, total weight  82.1
        same tape, wrong root  99 leaders, 6.07 bits, total weight 205.2
        same tape, shuffled   148 leaders, 6.92 bits, total weight 218.7

    Total weight moves the WRONG way (the broken runs score 2.5x more);
    leader count and lead entropy both move the right way. Compare only
    at matched trace length — see `concentration_verdict`.
    """
    # Accept either a `learn_objectives` result or an already-aggregated
    # {location: {...}} table.
    agg = (aggregate_locations(result_or_agg)[0]
           if "objectives" in result_or_agg else result_or_agg)
    leads = sorted((a["lead"] for a in agg.values()), reverse=True)
    tot = float(sum(leads))
    if tot <= 0:
        return {"lead_total": 0.0, "top1_share": 0.0, "top3_share": 0.0,
                "lead_entropy_bits": 0.0, "n_leaders": 0}
    p = np.array([x / tot for x in leads if x > 0])
    return {
        "lead_total": tot,
        "top1_share": float(p[0]),
        "top3_share": float(p[:3].sum()),
        "lead_entropy_bits": float(-(p * np.log2(p)).sum()),
        "n_leaders": int(p.size),
    }


def concentration_verdict(fit: dict, reference: Optional[dict], *,
                          n_fit: int, n_reference: int = 0,
                          length_tol: float = 0.10) -> dict:
    """Read a fit trajectory's concentration against a reference one.

    The reference is normally the SAME tapes with their actions permuted
    (`tape_replay.resolve_chain(..., shuffle_seed=N)`), which is matched
    in length by construction. Concentration is strongly length-dependent
    — the 31.6k-memory SMB chain concentrates onto 31 leaders where the
    903-memory single level spreads over 77 — so a comparison across
    materially different lengths is refused rather than reported.

    `verdict` is "concentrated" (the fit trajectory is more ordered than
    the reference on BOTH leader count and lead entropy), "diffuse" (the
    reference is), "mixed", "no_reference", or
    "inconclusive_length_mismatch".
    """
    out = {"fit": fit, "reference": reference,
           "n_fit": int(n_fit), "n_reference": int(n_reference),
           "length_matched": None, "d_leaders": None,
           "d_entropy_bits": None, "verdict": "no_reference",
           "statistic": "lead-mass concentration (distinct leaders + lead "
                        "entropy at matched length)",
           "not_used": "total objective weight / VF — the negative control "
                       "inverts it (broken trajectories score 2.5x MORE)"}
    if reference is None:
        return out
    lo, hi = min(n_fit, n_reference), max(n_fit, n_reference)
    out["length_matched"] = bool(hi and (hi - lo) <= length_tol * hi)
    out["d_leaders"] = int(fit["n_leaders"] - reference["n_leaders"])
    out["d_entropy_bits"] = float(fit["lead_entropy_bits"]
                                  - reference["lead_entropy_bits"])
    if not out["length_matched"]:
        out["verdict"] = "inconclusive_length_mismatch"
        return out
    tighter = out["d_leaders"] < 0 and out["d_entropy_bits"] < 0
    looser = out["d_leaders"] > 0 and out["d_entropy_bits"] > 0
    out["verdict"] = ("concentrated" if tighter else
                      "diffuse" if looser else "mixed")
    return out


# --------------------------------------------------------------------------
# Held-out scoring (weighted-objectives.h Observe / GetNormalizedValue)
# --------------------------------------------------------------------------

def build_scorer(M_fit: np.ndarray, objectives) -> list:
    """Freeze each positive-weight objective's observed value ladder.

    This is WeightedObjectives::Observe() over the fit trace: the sorted
    unique values an objective takes, which is what VF() bisects into.

    `objectives` is `learn_objectives(...)["objectives"]` (or the result
    dict itself). The frozen ladder is for MEASUREMENT ONLY — see
    `NEVER_WIRE_AS_REWARD`.
    """
    if isinstance(objectives, dict):
        objectives = objectives["objectives"]
    out = []
    for r in objectives:
        if r["weight"] <= 0:
            continue
        obj, k = list(r["obj"]), len(r["obj"])
        sub = np.ascontiguousarray(M_fit[:, obj])
        raw = sub.tobytes()
        uniq = sorted({raw[i * k:(i + 1) * k] for i in range(sub.shape[0])})
        out.append((obj, k, uniq, r["weight"]))
    return out


def score_trace(entries: list, M: np.ndarray) -> tuple:
    """GetNormalizedValue() for every memory: (unweighted, weighted) curves."""
    n = M.shape[0]
    s = np.zeros(n)
    sw = np.zeros(n)
    W = 0.0
    for obj, k, uniq, w in entries:
        raw = np.ascontiguousarray(M[:, obj]).tobytes()
        nu = len(uniq)
        vf = np.fromiter(
            (bisect.bisect_left(uniq, raw[i * k:(i + 1) * k]) / nu
             for i in range(n)), dtype=float, count=n)
        s += vf
        sw += w * vf
        W += w
    return s / max(len(entries), 1), sw / max(W, 1e-12)


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(a.size, dtype=float)
    return r


def curve_stats(y: np.ndarray) -> dict:
    """How well does a learned score track elapsed time on this trace?

    `spearman_vs_time` in-sample is meaningless as validation (it is what
    the fit maximizes); it is the HELD-OUT value that matters, and that
    is the number that says this must never be a reward (-0.771).
    """
    y = np.asarray(y, dtype=float)
    t = np.arange(y.size, dtype=float)
    rt, ry = _rank(t), _rank(y)
    rt -= rt.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rt * rt).sum() * (ry * ry).sum()))
    rho = float((rt * ry).sum() / den) if den > 0 else 0.0
    d = np.diff(y)
    nz = d != 0
    return {
        "spearman_vs_time": rho,
        "frac_steps_up": float((d > 0).sum() / max(d.size, 1)),
        "up_over_moves": float((d > 0).sum() / max(int(nz.sum()), 1)),
        "delta_end_begin": float(y[-1] - y[0]) if y.size else 0.0,
        "max_drawdown": float(np.max(np.maximum.accumulate(y) - y))
        if y.size else 0.0,
    }
