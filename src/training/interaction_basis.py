"""Deterministic controller-interaction basis + differential ranking.

THE INSTRUMENT, not a strategy. A saturated archive shows that every
POSITION near a boundary was visited many times; it shows nothing at all
about which INTERACTIONS were tried there. This module builds the
enumeration that answers the second question and the pure statistics that
read the answer back out of RAM.

Everything here is a function of the profile's own DECLARED action list
plus the RAM the sweep measured. There is no address allowlist, no game
knowledge, no target-derived constant: the ladders below (holds, tap
duties, settle phases, tails, the FDR level, the rank cutoff) are FIXED
DESIGN CONSTANTS declared once, here, and were not derived from any of
the graded targets. That separation is the whole point — a ranker tuned
until a known answer byte floats to the top on the target it will later
be graded on is not an instrument, it is a lookup table with extra steps.

GENERATION TABLE (fixture data; the per-target counts are asserted by
tests/test_go_explore_solve.py, not recomputed here):

    family order  HOLD -> TAP -> COMBO
    HOLD          every declared mask, held for each of HOLD_STEPS
                  -> 4 * |A| patterns
    TAP           every declared mask, at each of TAP_DUTIES
                  -> 2 * |A| patterns
    COMBO         pairwise OR of declared masks, deduped against
                  everything already emitted (so an OR that merely
                  reproduces a declared mask is dropped), capped at
                  COMBO_CAP. Enumerated RUNG-OUTER / PAIR-INNER: every
                  novel pair at HOLD_STEPS[0], then every novel pair at
                  HOLD_STEPS[1], and so on until the cap binds
                  -> COMBO_CAP patterns whenever the pairs supply that
                     many novel (mask, duration) programs

      Castlevania |A|=16 -> 64 + 32 + 24 = 120
      Bubble Bobble |A|=8 -> 32 + 16 + 24 =  72
      Contra      |A|=11 -> 44 + 22 + 24 =  90

    The rung-outer order is what makes those 24 combos WIDE. The action
    spaces supply 37 / 6 / 20 distinct novel pairwise masks (CV / BB /
    Contra), so the cap buys 24 masks x 1 rung on Castlevania, 20 + 4 on
    Contra, and 6 x 4 on Bubble Bobble — which has only six novel
    pairwise masks in existence, so depth is the only thing left for its
    cap to buy. Pair-outer spends the whole cap on the first six pairs
    at four durations for EVERY target, which is a duration sweep the
    HOLD family already ran and leaves 31 of Castlevania's combinations
    unmeasured.

SLOT 0 IS THE PAIRED CONTROL. Every fleet profile declares the empty
action first, so `hold(NOOP, 4)` is naturally the first pattern the
ladder emits and slot 0 is already the all-NOOP program. The hoist below
makes that a GUARANTEE rather than a coincidence of the YAML ordering,
because NOOP subtraction (`rank_candidates`) is meaningless without it.

...BUT ONE CONTROL IS NOT ENOUGH, AND SLOT 0 IS NOT THE ONLY ONE. A
paired control only subtracts the nuisances that move inside the window
it MEASURED. Slot 0 is a 4-step hold, so at phase 8 it measures steps
8..36 — while `hold_a0_h108` at the same phase measures 8..140. Subtract
the first from the second and every input-independent byte with a period
longer than ~28 steps survives, lands 3-root-invariant (it is a clock:
it reproduces everywhere), classes LATCHED, and is receipted as a
discovery. That is the exact failure the whole instrument exists to
prevent, so the ladder's OWN all-NOOP programs are all controls: the
NOOP action rides the same hold and tap rungs as every other action, so
the basis already contains an all-NOOP program at every length it
generates ({4,12,36,108} from HOLD/COMBO, {32,48} from TAP). Six for
every target. `rank_candidates` pairs each observation with the control
that shares its (phase, pattern length) — i.e. its exact t0/t1/t2 — and
no all-NOOP program is ever ranked as a candidate.

CONTACT IS NOT IN THE BASIS. A contact pattern is state-dependent and
therefore cannot be a function of `action_space` alone; `contact_admitted`
is a sweep-time admission evaluated against the profile's own gx/y
telemetry over settle, closed by the pattern's first frame, with its
parameters (eps, K and the window rule) declared here and echoed in the
run header.

THE PRIOR AND THE SCREEN ARE MEASURED ON THE POPULATION THE INSTRUMENT
IS ROOTED IN (repair D-PRIOR, receipted in runs/gate_opener_k0_2026-08-11
/k0_verdict.json). The first build sampled the novelty histogram and the
farmability counter ONLY from the live search's in-band observations —
and the sweep roots in exactly the band tail a pinned search essentially
never re-enters, so on the graded Castlevania run the histogram was never
allocated at all: `farm_samples` ended at 0 after 24 minutes, novelty
returned its 1.0 default for every row, and every BH-significant row was
refused `farm_unmeasured`. The admissible list was empty BY CONSTRUCTION.
The repair is that THE SWEEP FEEDS THE SAME STATISTICS: every frame of a
sweep program in which nothing has been pressed yet (the settle prefix of
every program, and the whole body of the ladder's all-NOOP controls) is
an undirected observation at a sweep root, and it is counted into the
same boundary histogram and the same change counter the live path feeds
(`baseline_frames`, `farm_rate_pooled`). ONLY undirected frames: folding
a pattern's own post-onset RAM into the prior would let the instrument
cancel its own discoveries — the value a pattern just created would score
as "already seen here". And a run that still has no reading reports None,
never 0.0: a fabricated measurement in a column called `farmability` is
the failure the refusal exists to prevent.

DEATH-AWARE CONTROL PAIRING (repair for the slot-0 asymmetry receipted in
runs/cv_t1_roots_2026-08-11.json finding 3: none of the 16 verified
Castlevania band roots survives the 153-step all-NOOP program, deaths at
step 31-115, while 3 of 4 sham roots do). Duration matching is already in
place — a control is paired on (phase, pattern length), so it only ever
speaks for the window it measured, and the pattern-truncation half of the
repair is that machinery. What duration matching CANNOT fix is liveness:
a control that dies inside its window measures a death-and-respawn RAM
cascade rather than a quiescent baseline, so subtracting it deletes every
address the death touched from the candidate pool; and the mirror case (a
pattern that dies while its control does not) promotes the same cascade
to a discovery — which is the "zero-page state going to 0 under 108-step
holds" block that plateaued the Contra tuning grade. So the repair is
SYMMETRIC and applied to the window, not to the program: an observation
whose liveness changed inside its own differential window is excluded
from the differential entirely, as a control and as a candidate alike,
and the exclusions are counted into the caller's `stats` rather than
absorbed. A root whose controls ALL died is dropped whole (its candidates
have no baseline left), while a root that ran no control at all keeps the
pre-existing behaviour — "no control was run" and "the control died" are
different facts and are not merged.

A BYTE IS A RING, NOT AN INTERVAL (repair for the Euclidean effect size).
The first build measured the differential with int16 subtraction and
called it "signed-safe". It is not: on a wrapping byte the Euclidean
distance is the WRONG metric in both directions at once. A counter
ticking 250 -> 254 -> 3 reads |3 - 250| = 247 (0.97 of full scale, so it
wins the effect-size tie-break outright) and escapes the monotone COUNTER
test (3 is not greater than 254), so it is classed LATCHED and takes rank
1. Both halves are fixed by measuring on the ring the byte actually
lives on: `ring_delta` is the shortest signed path (that tick is +4 then
+5, magnitude 9), `classify_persistence` reads monotonicity in signed
ring steps bounded by COUNTER_WRAP_STEP, and the effect size is
normalised by BYTE_HALF, the largest distance a byte can express.

THE PRIOR IS A DISTRIBUTION, NOT A LOOKUP (repair D-TIE's other half).
The novelty term was log2((total + 1) / (seen + 1)) over the count of
undirected frames at this exact (addr, value). By construction the prior
holds only PRE-PRESS frames, so a latched candidate's post-press value
has `seen = 0` always, and every latched row on a run comes back with
one identical novelty — 40 rows, one value, one score, and the plateau
the tie-break was blamed for is manufactured upstream of the tie-break.
The repair keeps the prior undirected and changes the ESTIMATOR: with an
address's observed SUPPORT (how many distinct values it took in the
undirected prior) the escape probability of an unseen value is
Witten-Bell's D/(N+D), spread over the values never seen, so a byte
pinned at one value for 5,000 frames scores an unseen value ~20 bits
while a free-runner that has already visited 200 values scores the same
event ~10. That is a measured property of the same undirected
population, it discriminates exactly where the plateau was, and it never
folds a post-press value into the prior.

D-IDENT — LIFETIME-CONSTANCY BYTES ARE NOT VALID ANSWER TARGETS FOR THIS
INSTRUMENT. `rank_candidates` is a differential onset/persist ranker: it
can only see an address that MOVES inside a 153-step program rooted at a
frozen archive blob. An answer key defined by CONSTANCY — Contra's slot
identity array 0x058F-92, whose own receipt defines it as constant for
the lifetime of a slot — is therefore close to unreachable by
construction: it moves only on slot turnover, which a paired NOOP control
at the same root usually reproduces and subtracts. Grading the instrument
against such a key measures the key's unsuitability, not the ranker. A
future K0 registration tuning on Contra should use the hp array
(0x04BF-C2), which changes on contact and is exactly the onset shape the
differ is built to read.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from src.training.profile_utils import action_space_to_bitmasks

# --- families ---------------------------------------------------------
HOLD = "hold"
TAP = "tap"
COMBO = "combo"
CONTROL = "control"
#: NOT a generated family — a SWEEP-TIME relabel. A longest hold whose
#: settle window shows the position observable frozen was pressed against
#: something, and "held right into a wall" is a different interaction
#: from "held right across open ground". The distinction can only be made
#: against live state, so it is applied where the state is (the sweep),
#: never in the pure basis.
CONTACT = "contact"
FAMILY_ORDER = (HOLD, TAP, COMBO)

# --- the fixed design ladders (target-independent) --------------------
#: Hold durations, a geometric-ish ladder 4 -> 108.
HOLD_STEPS = (4, 12, 36, 108)
#: (on, off, repeats) duty cycles. Both land inside the 108-step budget.
TAP_DUTIES = ((2, 2, 8), (2, 10, 4))
#: Pairwise-OR combinations kept, after dedupe.
COMBO_CAP = 24
#: Whole-basis ceiling. Truncation (if it ever binds) drops from the
#: COMBO tail, never the control.
DEFAULT_CAP = 128

#: Settle steps before the pattern starts — the phase replicates.
SETTLE_PHASES_PASS_A = (8, 21)
SETTLE_PHASES_PASS_B = (8, 13, 21, 34)
#: Post-pattern observation window.
TAIL_PASS_A = 24
TAIL_PASS_B = 512
#: One uniform program length per pass, so a whole wave of workers can be
#: stepped in lockstep by a single `step_all` loop. max settle + max
#: pattern + tail; shorter programs are NOOP-padded at the END (the tail
#: keeps its declared length, and the capture indices are per-program).
PROGRAM_LEN_PASS_A = max(SETTLE_PHASES_PASS_A) + max(HOLD_STEPS) + TAIL_PASS_A
PROGRAM_LEN_PASS_B = max(SETTLE_PHASES_PASS_B) + max(HOLD_STEPS) + TAIL_PASS_B

# --- the fixed statistical ladders ------------------------------------
FDR_Q = 0.01
RANK_CUTOFF = 5
#: A candidate must reproduce from at least this many DISTINCT roots.
MIN_INVARIANT_ROOTS = 3
#: Persistence weights: a latched change is what a gate looks like, a
#: timed one is a transient, a monotone counter is nuisance.
PERSIST_WEIGHT = {"latched": 1.0, "timed": 0.6, "counter": 0.0}
#: K4 refusals.
MAX_AXIS_VALUES = 8
MAX_FARM_EVENTS_PER_1K = 1.0
#: Shadow-ledger value bucketing: MAX_AXIS_VALUES buckets over a byte.
VALUE_BUCKET = 256 // MAX_AXIS_VALUES
#: One in every STRIDE in-band observations is sampled into the boundary
#: histogram and the farmability counter. Declared here because both the
#: sampler and the estimator that divides by it have to agree.
BAND_SAMPLE_STRIDE = 32
#: OBSERVED STEPS needed before a farm rate is reported at all. This is
#: the real floor and the only one that survives pooling two samplers at
#: different strides: below 1,000 observed steps the smallest resolvable
#: nonzero rate, 1000/steps, sits above MAX_FARM_EVENTS_PER_1K, so the
#: estimate could not decide the refusal it exists to decide.
FARM_MIN_STEPS = 1000
#: The same floor in stride-32 sample pairs (32 x 32 = 1,024 steps), kept
#: as a named constant because the live band sampler counts in pairs.
FARM_MIN_SAMPLES = -(-FARM_MIN_STEPS // BAND_SAMPLE_STRIDE)
#: The sweep's own undirected frames are consecutive, so they are counted
#: at stride 1 — an exact per-step change count, not a gap estimate.
SWEEP_SAMPLE_STRIDE = 1

# --- contact admission (sweep-time, self-measured) --------------------
#: |dgx| <= CONTACT_EPS and |dy| <= CONTACT_EPS for CONTACT_K steps.
CONTACT_EPS = 0
CONTACT_K = 8

RAM_SIZE = 2048

# --- the byte ring ----------------------------------------------------
#: A RAM byte wraps, so the distance between two readings is a distance
#: on Z/256 and the largest one expressible is 128, not 255.
BYTE_MOD = 256
BYTE_HALF = BYTE_MOD // 2
#: The largest per-capture step still readable as a WRAPPED counter tick.
#: The plain in-range monotone test needs no bound (it cannot be faked by
#: a wrap); the ring extension does, because "moved -56 twice" is also
#: what a latch followed by a partial revert looks like, and only a small
#: consistent tick is unambiguously a free-runner crossing 0xFF/0x00.
COUNTER_WRAP_STEP = 32


def ring_delta(a: int, b: int) -> int:
    """b - a as the SHORTEST signed path on the byte ring, in
    [-127, 128]. 0xFA -> 0x03 is +9, not -247.

    The antipode is genuinely ambiguous — 0 and 128 are 128 apart in
    both directions — and is reported as +128 from either side, so the
    function stays total and the magnitude stays correct."""
    d = (int(b) - int(a)) % BYTE_MOD
    return d - BYTE_MOD if d > BYTE_HALF else d


def ring_distance(a: int, b: int) -> int:
    """|ring_delta| — 0 .. 128."""
    return abs(ring_delta(a, b))


def _ring_delta_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised `ring_delta` over two uint8 arrays, as int16."""
    d = (b.astype(np.int16) - a.astype(np.int16)) % BYTE_MOD
    return np.where(d > BYTE_HALF, d - BYTE_MOD, d).astype(np.int16)


class Pattern(NamedTuple):
    """One interaction. `masks` is the PATTERN SEGMENT (unpadded) — the
    program it becomes depends on the settle phase and tail, which are
    sweep parameters rather than properties of the interaction."""

    name: str
    family: str
    masks: tuple

    def program(self, settle: int, tail: int = TAIL_PASS_A,
                length: int = PROGRAM_LEN_PASS_A) -> tuple:
        return build_program(self, settle, tail, length)


def interaction_basis(action_space: Sequence, cap: int = DEFAULT_CAP) -> list:
    """The deterministic interaction basis for a declared action space.

    Pure and order-stable: same `action_space`, same list, every time.
    Never invents a button — every mask is one declared mask or the OR of
    two of them.
    """
    masks = tuple(action_space_to_bitmasks(list(action_space)))
    out: list = []
    seen: set = set()

    def emit(name: str, family: str, seg: tuple) -> bool:
        if seg in seen:
            return False
        seen.add(seg)
        out.append(Pattern(name, family, seg))
        return True

    for ai, m in enumerate(masks):
        for h in HOLD_STEPS:
            emit(f"hold_a{ai}_h{h}", HOLD, (m,) * h)
    for ai, m in enumerate(masks):
        for on, off, reps in TAP_DUTIES:
            emit(f"tap_a{ai}_{on}on{off}off_x{reps}", TAP,
                 ((m,) * on + (0,) * off) * reps)
    kept = 0
    # RUNGS OUTER, PAIRS INNER. The nesting is the whole content of the
    # cap: with pairs outer, COMBO_CAP=24 is spent by the first six
    # novel pairs at four durations apiece, so the family whose entire
    # question is "which BUTTON COMBINATIONS were never tried here"
    # ships six distinct masks on a 16-action space that offers 37 —
    # and the other 18 slots re-ask a duration question the HOLD family
    # already answered. Breadth first: buy distinct masks until the
    # pairs run out, and only then buy a second duration for them.
    for h in HOLD_STEPS:
        if kept >= COMBO_CAP:
            break
        for i, j in itertools.combinations(range(len(masks)), 2):
            if kept >= COMBO_CAP:
                break
            combined = masks[i] | masks[j]
            if emit(f"combo_a{i}+a{j}_h{h}", COMBO, (combined,) * h):
                kept += 1

    # Slot 0 must be the all-NOOP control. For every fleet profile the
    # ladder already put it there (the declared action list starts with
    # []); this makes it structural instead of incidental.
    noop_at = next((k for k, p in enumerate(out) if not any(p.masks)), None)
    if noop_at is None:
        out.insert(0, Pattern("control_noop", CONTROL, (0,) * HOLD_STEPS[0]))
    elif noop_at:
        out.insert(0, out.pop(noop_at))
    return out[:cap] if cap and len(out) > cap else out


def control_slots(basis: Sequence) -> dict:
    """{pattern length -> basis slot} over the all-NOOP programs.

    Every length the ladder generates has one, because the NOOP action
    rides the same hold and tap rungs as every other action. NOOP
    subtraction is only valid between two programs that measured the SAME
    window, so any pass that re-runs a survivor has to re-run its length
    twin alongside it or the survivor comes back uncontrolled.
    """
    out: dict = {}
    for slot, pat in enumerate(basis):
        if not any(pat.masks):
            out.setdefault(len(pat.masks), slot)
    return out


def macro_injection(pattern: Pattern, action_masks: Sequence):
    """(declared action index, hold steps) if `pattern` is a constant
    hold of ONE declared action; None if the search cannot express it.

    The search's shared macro slot is (macro_a, macro_hold) — an action
    INDEX plus a hold count — and the trace alphabet the archive, the
    replay verifier and the suppression ablation all speak is action
    indices too. So a TAP duty cycle (not a constant hold) and a COMBO
    mask (by construction an OR that is NOT any declared action) have no
    in-run injection channel at all. On Castlevania that is 60 of the 120
    patterns: 32 taps + 24 combos + the 4 all-NOOP holds (a control
    presses nothing, so there is nothing to inject). 60 remain — the
    sweep MEASURES twice what the search can be HANDED. That is a
    property of the channel §1 names, and the caller is expected to count
    every refusal rather than let the gap pass as a null (R2).
    """
    seg = tuple(pattern.masks)
    if not seg or len(set(seg)) != 1 or seg[0] == 0:
        return None
    try:
        return (list(action_masks).index(seg[0]), len(seg))
    except ValueError:
        return None


def farm_rate(changes: int, samples: int, stride: int = BAND_SAMPLE_STRIDE,
              min_samples: int = FARM_MIN_SAMPLES):
    """K4's farmability, as a MEASUREMENT: value-change events per 1,000
    in-band search steps. None — never 0.0 — when the run has not sampled
    enough to resolve the threshold.

    `changes` is the number of consecutive band SAMPLES at which the
    address differed and `samples` the number of sample pairs behind it,
    so each observed change stands for at least one change somewhere in a
    `stride`-step gap:

        rate per 1k steps = 1000 * changes / (samples * stride)

    In the rare regime K4's 1-event/1k threshold actually lives in,
    P(change in a gap) ~= stride * per-step rate, so the estimator is
    unbiased exactly where the decision is made. In the fast regime it
    saturates at 1000/stride (31.25 at stride 32) — an address that
    differs at almost every sample is refused either way, so the bias
    there cannot change a verdict.
    """
    return farm_rate_pooled(((changes, samples, stride),),
                            min_steps=int(min_samples) * int(stride))


def farm_rate_pooled(sources: Iterable, min_steps: int = FARM_MIN_STEPS):
    """K4's farmability pooled over several samplers: value-change events
    per 1,000 observed steps. None — never 0.0 — below `min_steps`.

    `sources` is an iterable of (changes, samples, stride). Pooling is in
    STEP units, which is what makes two samplers at different strides
    combinable at all: each contributes `samples * stride` steps observed
    and `changes` changes seen inside them, and the ratio is the quantity
    K4's threshold is written in. That matters because the two real
    sources are asymmetric by construction — the live search samples the
    band at stride 32 and, at a pinned wall whose band tail it never
    re-enters, may sample it a handful of times or never (D-PRIOR), while
    the sweep's own undirected frames are consecutive and counted exactly
    at stride 1.

    ZERO IS NOT A DEFAULT. A source with no samples contributes nothing
    to either total instead of contributing a zero numerator, and a pool
    that never reaches `min_steps` reports None — so an unmeasured axis is
    REFUSED by the ranker rather than admitted on a number no sampler
    produced.
    """
    changes = 0
    steps = 0
    for c, n, stride in sources:
        n, stride = int(n), int(stride)
        if n <= 0 or stride <= 0:
            continue
        changes += int(c)
        steps += n * stride
    if steps < int(min_steps):
        return None
    return 1000.0 * float(changes) / float(steps)


def noop_prefix_len(program: Sequence) -> int:
    """How many leading frames of `program` press nothing.

    THE UNDIRECTED PREFIX. After this many steps the program has begun
    pressing, so every later frame is a consequence of the interaction
    and cannot be counted into the prior that same interaction is scored
    against (module docstring, D-PRIOR). For an all-NOOP control it is
    the whole program; for a pattern at settle phase p it is exactly p.
    """
    n = 0
    for a in program:
        if a:
            break
        n += 1
    return n


def tie_break_key(seed: int, addr: int, family: str) -> int:
    """A stable pseudo-random total order over rows, keyed by the RUN
    SEED — the last resort once score, effect size and family diversity
    have all tied.

    THE DEFECT THIS REPLACES (D-TIE, receipted in
    runs/gate_opener_k0_2026-08-11/k0_revision_log.json): the first build
    broke the score plateau with the row's ADDRESS, ascending. The score
    is a product of four coarse factors, so its plateaus are large — the
    Contra tuning grade wrote 32 rows at one identical score, still
    unbroken at the truncation boundary — and an ascending-address
    tie-break makes `rank <= 5` a function of where a byte happens to sit
    in the memory map. Both blind-graded answers were low addresses, so
    even the PASS branch would have been partly a sort-key artifact.

    Keyed by the seed rather than fixed, so the arbitrary part of the
    order is a declared, logged property of the run and two seeds are a
    real replication of the ranking instead of the same coin twice.
    Content-addressed and therefore independent of the order the
    observations arrived in.
    """
    h = hashlib.blake2b(f"{int(seed)}|{int(addr)}|{family}".encode(),
                        digest_size=8)
    return int.from_bytes(h.digest(), "big")


def rank_sort_key(row: Mapping[str, Any]) -> tuple:
    """The instrument's total order over ranked rows.

    score desc -> SIGNIFICANT first -> UNREFUSED first -> effect size
    desc -> the seeded pseudo-random key. Address and family appear only
    as the determinism backstop behind an 8-byte hash, so a low address
    buys no rank. Both `rank_candidates` and the pass-B re-sort read this
    one definition, because two sort keys is how a repaired tie-break
    comes back through the confirmation pass.

    THE EVIDENCE OUTRANKS THE PLATEAU. The score is a product of four
    coarse factors and its plateaus are enormous, while `significant` —
    the BH-FDR verdict the whole pipeline exists to produce — was not in
    the key at all. On a four-family, eight-root probe the ONLY
    BH-significant row (a one-family mode bit moving by 8, the shape of
    the blind key) sorted LAST, behind three non-significant nuisances
    tied at its score, because they moved further. Significance is now
    the first tie-break and the K4 refusal the second, so the rows an
    admissible list is allowed to contain lead the rows it may not.

    FAMILY DIVERSITY IS NOT IN THE KEY, DELIBERATELY. `addr_families`
    used to sort DESC here while the leave-family-out background rate
    that sets the row's p-value RISES with it: the same quantity was
    evidence against significance in step 4 and evidence for rank in
    step 6, in one pipeline. It stays on the row as a receipt field,
    where a reader can see it without it being counted twice.
    """
    return (-float(row.get("score", 0.0)),
            0 if row.get("significant") else 1,
            1 if row.get("refused") else 0,
            -float(row.get("effect_size", 0.0)),
            int(row.get("tie_key", 0)),
            int(row.get("addr", 0)), str(row.get("family", "")))


def build_program(pattern: Pattern, settle: int, tail: int = TAIL_PASS_A,
                  length: int = PROGRAM_LEN_PASS_A) -> tuple:
    """settle NOOPs + the pattern + `tail` NOOPs, NOOP-padded to `length`.

    The pad rides at the END, past the tail: the tail is the persistence
    window and must keep its declared length, while the uniform length
    exists only so one `step_all` loop can drive a whole wave.
    """
    body = (0,) * int(settle) + tuple(pattern.masks) + (0,) * int(tail)
    if len(body) > length:
        raise ValueError(
            f"program {pattern.name} at settle={settle}/tail={tail} is "
            f"{len(body)} steps, over the uniform length {length}")
    return body + (0,) * (length - len(body))


def capture_points(pattern: Pattern, settle: int,
                   tail: int = TAIL_PASS_A) -> tuple:
    """(t0, t1, t2) step indices: end of settle, end of pattern, end of
    tail. Indices are 1-based in "steps taken", i.e. capture AFTER the
    t-th step of the program."""
    t0 = int(settle)
    t1 = t0 + len(pattern.masks)
    return (t0, t1, t1 + int(tail))


def _frozen(window: Sequence, eps: int) -> bool:
    """Every consecutive pair in `window` within `eps` on BOTH axes."""
    return all(abs(int(b[0]) - int(a[0])) <= eps
               and abs(int(b[1]) - int(a[1])) <= eps
               for a, b in zip(window, window[1:]))


def contact_admitted(positions: Sequence, eps: int = CONTACT_EPS,
                     k: int = CONTACT_K,
                     n_settle: int | None = None) -> bool:
    """Self-measured contact test over the profile's OWN (gx, y) telemetry.

    True when the observable moved by at most `eps` in both axes across
    `k` consecutive SETTLE steps and — when the caller sampled it — across
    the step into the pattern's first frame as well. The solver's own
    reading of "this hold is pressed against something", with no collision
    map and no game internals.

    `n_settle` is how many of `positions` are settle samples; anything
    past it is the pattern already running (the default, None, says the
    caller sampled settle and nothing else).

    THE FREEZE WINDOW IS ANCHORED TO SETTLE AND NEVER SLIDES. A settle of
    t0 steps yields only t0 samples — the root frame is not readable,
    because the pool exposes RAM only as the result of a step — so at the
    shortest registered phase, 8, a k=8 freeze is one sample short and the
    window has to reach one frame into the pattern to exist at all. That
    extension is a LAST RESORT, not the general rule. Taking the last k+1
    of ALL samples instead shifts the window forward at every longer
    phase: it drops the earliest settle transition to make room for a
    frame the test did not need, and the dropped transition is real
    evidence. A root still decelerating when settle begins then comes out
    REFUSED at phase 8 (which still sees the motion) and ADMITTED at 13,
    21 and 34 (which no longer do) — the same interaction, from the same
    root, classed differently per phase. Phase replication exists to
    detect that kind of instability, not to manufacture it.

    THE PATTERN'S FIRST FRAME IS A SEPARATE CONJUNCT, not the ninth
    sample of a slid window. It is the only sample that can tell "pressed
    against something" from "standing still on open ground": both are
    frozen while the pad is NOOP, and only one of them is still frozen
    one frame after the pattern starts pressing. Kept as its own term it
    is checked at EVERY phase and can only ever REFUSE an admission —
    never fill in for a settle sample the window dropped.
    """
    if k <= 0:
        return True
    pos = list(positions)
    n = len(pos)
    n_settle = n if n_settle is None else max(0, min(int(n_settle), n))
    if n_settle >= k + 1:
        window = pos[n_settle - (k + 1):n_settle]      # settle only
    elif n >= k + 1:
        window = pos[:k + 1]                           # + the first frame
    else:
        return False
    if not _frozen(window, eps):
        return False
    if n > n_settle >= 1:
        return _frozen(pos[n_settle - 1:n_settle + 1], eps)
    return True


def classify_persistence(v0: int, v1: int, v2: int,
                         wrap_step: int = COUNTER_WRAP_STEP) -> str:
    """TIMED (moved and came back), LATCHED (moved and stayed), COUNTER
    (moved monotonically at both samples — a free-running nuisance).

    MONOTONE ON THE RING (repair). The plain test — v0 < v1 < v2 or
    v0 > v1 > v2 — cannot see a counter that crossed the 0xFF/0x00
    boundary between two captures, and a counter is exactly the thing
    that crosses it: 250 -> 254 -> 3 is a byte ticking +4 then +5, and
    the plain test calls that LATCHED (moved and stayed), which is the
    class the whole rank is built to reward. So a wrapped tick is read as
    monotone too — but only when BOTH signed ring steps are small
    (<= `wrap_step`), because a large step in one ring direction is
    equally consistent with a latch followed by a partial revert and the
    ring gives no way to tell those apart. Below the bound the reading is
    unambiguous; above it the pre-existing (conservative) class stands.
    """
    v0, v1, v2 = int(v0), int(v1), int(v2)
    onset = v1 != v0
    persist = v2 != v0
    if onset and persist:
        if (v0 < v1 < v2) or (v0 > v1 > v2):
            return "counter"
        d1, d2 = ring_delta(v0, v1), ring_delta(v1, v2)
        if d1 and d2 and (d1 > 0) == (d2 > 0) \
                and max(abs(d1), abs(d2)) <= int(wrap_step):
            return "counter"
    if persist:
        return "latched"
    return "timed"


def binom_sf(k: int, n: int, p: float) -> float:
    """P[X >= k] for X ~ Binomial(n, p). Exact, no SciPy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for i in range(int(k), int(n) + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


def comparison_grid(n_families: int, ram_size: int = RAM_SIZE) -> int:
    """The FULL comparison count one sweep performs: the addr x family
    grid (§1), every cell of which is a hypothesis the differ tested.

    This is the denominator BH is registered against. The tempting
    alternative — the handful of rows left after the >= MIN_INVARIANT
    _ROOTS filter — is the multiple-comparisons error the correction
    exists to prevent: a filter chosen because it keeps the interesting
    rows cannot then define the family of tests those rows were selected
    from. On Castlevania the grid is 2,048 x 4 families, so q=0.01
    buys a first-rank bar near 1e-6 rather than 1e-2, and R1's
    comparison flood is actually paid for.
    """
    return int(ram_size) * max(1, int(n_families))


def bh_significant(pvalues: Sequence, q: float = FDR_Q,
                   m: int | None = None) -> list:
    """Benjamini-Hochberg step-up. Returns a bool per input position.

    `m` is the number of hypotheses TESTED, which is not in general the
    number of p-values handed in: a caller that pre-filters its rows
    still performed every comparison behind the filter, and BH over the
    survivors alone reports the FDR of a family that was chosen after
    seeing the data. Positions absent from `pvalues` are treated as
    p=1 (they sort last and can never be kept), which makes the
    correction conservative in the only safe direction — it can cost a
    real candidate, never invent one. Defaults to len(pvalues).
    """
    n = len(pvalues)
    if n == 0:
        return []
    m = n if m is None else max(int(m), n)
    order = sorted(range(n), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * q:
            cutoff = rank
    keep = [False] * n
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff:
            keep[idx] = True
    return keep


def novelty_score(seen: int, total: int, support: int | None = None,
                  alphabet: int = BYTE_MOD) -> float:
    """Surprise, in bits, of seeing `value` at this address under the
    run's own undirected prior.

    `seen` is how many undirected frames showed that value, `total` how
    many undirected frames the prior holds, and `support` how many
    DISTINCT values the address took across them.

    WITHOUT `support` this is the original log2((total+1)/(seen+1)) —
    kept because it is what a caller with only a (seen, total) pair can
    honestly compute, and because the pass-through keeps every existing
    reading reproducible.

    WITH `support` it is the Witten-Bell estimate, and that is the repair
    (module docstring). The prior holds only pre-press frames, so every
    LATCHED candidate has seen == 0 by construction and the first form
    returns one constant for all of them: 40 latched rows, one novelty,
    one score, an unbreakable plateau. Witten-Bell splits that constant
    by how volatile the ADDRESS is in the same undirected population —
    an event's probability mass is (support)/(total+support) for values
    never seen, shared over the values never seen, and seen/(total+
    support) otherwise:

        unseen: log2((total + support) * (alphabet - support) / support)
        seen:   log2((total + support) / seen)

    A byte pinned at one value for 5,000 undirected frames scores an
    unseen value at ~20 bits; a free-runner that already visited 200
    values scores the same event at ~10; the value a byte shows
    constantly still scores ~0. Nothing post-press enters the prior.
    """
    total = float(total)
    seen = float(seen)
    if support is None:
        return math.log2((total + 1.0) / (seen + 1.0))
    d = float(max(0, int(support)))
    if d <= 0.0:
        # The address was never observed at all: there is no support to
        # estimate an escape mass from, so fall back rather than invent.
        return math.log2((total + 1.0) / (seen + 1.0))
    if seen > 0.0:
        return math.log2((total + d) / seen)
    unseen = float(max(1, int(alphabet) - int(d)))
    return math.log2((total + d) * unseen / d)


def as_ram(x) -> np.ndarray:
    """A RAM capture as uint8, whatever the caller had. The pool hands
    back arrays; savestate/test paths hand back bytes, and np.asarray
    would try to parse those as a scalar."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(x), dtype=np.uint8)
    return np.asarray(x, dtype=np.uint8)


_as_ram = as_ram


def _is_control(ob: Mapping[str, Any]) -> bool:
    """Is this observation a paired NOOP control? The sweep says so
    explicitly (it knows the pattern); a hand-built fixture that predates
    the flag falls back to the slot-0 convention."""
    flag = ob.get("control")
    if flag is None:
        return int(ob.get("slot", 0)) == 0
    return bool(flag)


def _is_alive(ob: Mapping[str, Any]) -> bool:
    """Did this observation's own differential window run without the
    liveness signal changing? Absent the key the answer is YES, so a
    fixture (or a profile with no readable liveness telemetry) behaves
    exactly as it did before the death-aware repair existed."""
    return bool(ob.get("alive", True))


def _window(ob: Mapping[str, Any]) -> tuple:
    """The differential WINDOW an observation measured over, as
    (settle phase, pattern length) — which fixes t0/t1/t2 exactly. Two
    observations share a nuisance background only if they share this
    key, so it is the key the paired control is looked up under."""
    return (ob.get("phase"), ob.get("plen"))


def rank_candidates(
    observations: Iterable[Mapping[str, Any]],
    *,
    novelty: Callable[[int, int], float] | None = None,
    farmability: Callable[[int], float] | None = None,
    min_roots: int = MIN_INVARIANT_ROOTS,
    q: float = FDR_Q,
    m: int | None = None,
    tie_seed: int = 0,
    stats: dict | None = None,
) -> list:
    """Rank differential RAM candidates. PURE, and order-independent.

    `observations` are dicts {root, slot, family, ram0, ram1, ram2} plus
    the optional {control, phase, plen} the sweep fills in. The pipeline,
    in order:

      1. NOOP SUBTRACTION — any address that moves in the control from
         the SAME root and the SAME WINDOW is dropped for that root.
         Frame counters, RNG, timers and music free-run whatever the pad
         does; subtracting a measured control removes them without an
         allowlist. The window match is the load-bearing half: a control
         that ran 28 steps cannot speak for a pattern that ran 132, and
         a mismatched pair silently promotes every slow clock to a
         latched discovery. Controls are matched on (phase, plen); with
         no exact twin the union of that root's controls is used, which
         over-subtracts — the direction that can cost a real candidate
         but can never invent one. No control is ever ranked itself.
      2. CROSS-ROOT INVARIANCE — an (addr, family) must reproduce from at
         least `min_roots` DISTINCT roots.
      3. PERSISTENCE CLASS — timed / latched / counter, by majority.
      4. BH-FDR at `q` over the addr x family grid, against each
         address's own leave-family-out background rate. The grid — all
         RAM_SIZE x |families| comparisons the differ ran — is the
         denominator, NOT the post-invariance survivor list: the filter
         selects on the data, so it cannot also define the family of
         tests it selected from. `m` overrides that denominator with one
         the CALLER computed: a sweep that ranks its wall roots and its
         sham-null roots in two calls performed one grid of comparisons,
         not two, and each call can only see the families ITS OWN half
         happened to carry. Left None, the grid is derived from the
         families present here.
      5. RANK = novelty x persistence weight x controllability x
         (1 - farmability), with the K4 refusals recorded (never silently
         dropped) as `refused`. `farmability` is a callable or nothing:
         with no callable the row reports farmability None and is not
         screened; with one, a None reading means UNMEASURED and refuses
         the axis. A row never carries a farmability number that no
         probe produced.
      6. ORDER — `rank_sort_key`: score, then the BH verdict, then the
         K4 refusal, then EFFECT SIZE (the mean ring distance of the
         differential, in bytes), then a pseudo-random key drawn from
         `tie_seed`. Neither the address (D-TIE) nor the family count
         (which already sets the p-value) is a tie-break.

    LIVENESS. Step 1 uses only controls that survived their own window,
    and step 2 only counts patterns that survived theirs; a root whose
    controls all died is dropped whole. `stats`, if supplied, receives
    the counts (roots, roots_uncontrolled, controls, dropped_dead_control,
    dropped_dead_pattern, observations) so a receipt can report what the
    differential declined to read instead of silently shrinking.
    """
    obs = list(observations)
    st = stats if stats is not None else {}
    for k in ("roots", "roots_uncontrolled", "controls",
              "dropped_dead_control", "dropped_dead_pattern",
              "observations"):
        st.setdefault(k, 0)
    by_root: dict = {}
    for ob in obs:
        by_root.setdefault(ob["root"], []).append(ob)

    events: dict = {}
    family_roots: dict = {}
    for root in sorted(by_root, key=str):
        rows = by_root[root]
        ctls = [c for c in rows if _is_control(c)]
        live_ctls = [c for c in ctls if _is_alive(c)]
        st["controls"] += len(ctls)
        st["dropped_dead_control"] += len(ctls) - len(live_ctls)
        if ctls and not live_ctls:
            # EVERY control this root ran died inside its own window, so
            # there is no quiescent baseline left to subtract and the
            # only honest reading of its patterns is none. Dropping the
            # root is the conservative direction: the alternative is to
            # rank uncontrolled differentials, which is how a death
            # cascade becomes a receipted discovery. A root that ran NO
            # control keeps the pre-existing behaviour — "no control was
            # run" and "the control died" are different facts.
            st["roots_uncontrolled"] += 1
            continue
        st["roots"] += 1
        # ONE FREE-RUNNER MASK PER WINDOW, not one per root: an
        # all-NOOP program only certifies the span it actually stepped.
        free_by_window: dict = {}
        free_any = np.zeros(RAM_SIZE, dtype=bool)
        for ctl in live_ctls:
            r0, r1, r2 = (_as_ram(ctl["ram0"]), _as_ram(ctl["ram1"]),
                          _as_ram(ctl["ram2"]))
            moved = (r1 != r0) | (r2 != r0)
            win = _window(ctl)
            prev = free_by_window.get(win)
            free_by_window[win] = moved if prev is None else (prev | moved)
            free_any |= moved
        for ob in rows:
            if _is_control(ob):
                continue
            if not _is_alive(ob):
                # The mirror of the dead control: a pattern that died
                # inside its window measured the death cascade, and its
                # live control cannot subtract what it never saw. That is
                # how "zero-page state going to 0 under a 108-step hold"
                # became the tied block at the top of the Contra grade.
                st["dropped_dead_pattern"] += 1
                continue
            st["observations"] += 1
            fam = ob["family"]
            family_roots.setdefault(fam, set()).add(root)
            r0, r1, r2 = (_as_ram(ob["ram0"]), _as_ram(ob["ram1"]),
                          _as_ram(ob["ram2"]))
            free = free_by_window.get(_window(ob))
            if free is None:
                free = free_any
            moved = ((r1 != r0) | (r2 != r0)) & ~free
            if not moved.any():
                continue
            # EFFECT SIZE, in the units the differential is measured in:
            # how far the byte actually moved ON THE RING it lives on.
            # int16 subtraction is Euclidean, and a byte is not an
            # interval: it made a counter's 250 -> 3 tick read 247/255 =
            # 0.97 and win the tie-break outright. Computed only for the
            # addresses that survived subtraction, so the cost rides on
            # the survivors.
            dmax = np.maximum(np.abs(_ring_delta_np(r0, r1)),
                              np.abs(_ring_delta_np(r0, r2)))
            for addr in np.flatnonzero(moved).tolist():
                rec = events.setdefault((addr, fam), {
                    "roots": set(), "classes": {}, "values": set(),
                    "slots": set(), "mag": 0, "n": 0})
                rec["roots"].add(root)
                rec["slots"].add(int(ob.get("slot", -1)))
                cls = classify_persistence(int(r0[addr]), int(r1[addr]),
                                           int(r2[addr]))
                rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
                rec["values"].add(int(r2[addr]))
                rec["mag"] += int(dmax[addr])
                rec["n"] += 1

    # Leave-family-out background rate per address, in ROOT units so it
    # is comparable with the invariance count.
    addr_roots: dict = {}
    for (addr, fam), rec in events.items():
        addr_roots.setdefault(addr, {})[fam] = len(rec["roots"])
    total_roots = {f: len(rs) for f, rs in family_roots.items()}
    grand = sum(total_roots.values())

    survivors = []
    for (addr, fam), rec in events.items():
        k = len(rec["roots"])
        if k < min_roots:
            continue
        n = total_roots.get(fam, 0)
        other_hits = sum(v for f, v in addr_roots.get(addr, {}).items()
                         if f != fam)
        other_trials = grand - n
        # No other family ran (a single-family sweep, or a fixture): the
        # leave-one-out background is undefined, so fall back to the
        # maximally conservative coin flip rather than to a rate this
        # family's own hits would inflate.
        rate = (other_hits / other_trials) if other_trials > 0 else 0.5
        rate = min(max(rate, 1.0 / (grand + 1.0)), 1.0 - 1e-9)
        survivors.append({
            "addr": int(addr),
            "family": fam,
            "roots": k,
            "family_roots": n,
            "class": max(rec["classes"].items(), key=lambda kv: (kv[1], kv[0]))[0],
            "values": sorted(rec["values"]),
            # Which basis slots produced this row. The confirmation pass
            # needs them to re-run the right programs, and the in-run
            # injection needs them to know WHICH interaction to inject.
            "slots": sorted(rec["slots"]),
            # The magnitude behind the p-value. A p-value collapses into
            # plateaus under discreteness (k = n at every row is one
            # value of p); the mean ring distance does not, and it is a
            # property of the measurement rather than of the address's
            # position in the map. Normalised by BYTE_HALF, the largest
            # distance a byte can express.
            "effect_size": (rec["mag"] / max(1, rec["n"])) / float(BYTE_HALF),
            "addr_families": len(addr_roots.get(addr, {})),
            "tie_key": tie_break_key(tie_seed, addr, fam),
            "tie_seed": int(tie_seed),
            "p": binom_sf(k, n, rate),
        })

    # BH over the FULL addr x family grid, not over the rows that
    # happened to clear the invariance filter. The filter runs first
    # because it is cheap, but it selects ON the data, so letting it set
    # m would shrink the denominator from thousands to a handful and
    # hand every survivor a bar ~m times too generous.
    grid_m = (comparison_grid(len(family_roots)) if m is None
              else max(1, int(m)))
    keep = bh_significant([s["p"] for s in survivors], q, m=grid_m)
    screening = farmability is not None
    for s, sig in zip(survivors, keep):
        s["significant"] = bool(sig)
        # The denominator the significance was charged against, on the
        # row, so a receipt reader can reproduce the verdict from the
        # receipt alone instead of re-deriving how many families ran.
        s["fdr_m"] = grid_m
        farm = farmability(s["addr"]) if screening else None
        farm = None if farm is None else float(farm)
        nov = 1.0 if novelty is None else float(novelty(s["addr"],
                                                        s["values"][-1]))
        control = s["roots"] / max(1, s["family_roots"])
        s["novelty"] = nov
        s["farmability"] = farm
        s["score"] = (nov * PERSIST_WEIGHT.get(s["class"], 0.0) * control
                      * (1.0 if farm is None else max(0.0, 1.0 - farm)))
        refused = None
        if len(s["values"]) > MAX_AXIS_VALUES:
            refused = "cardinality"
        elif farm is None:
            # K4's second refusal is a MEASUREMENT. A caller that asked
            # for farmability screening and got no reading has an axis
            # that did not pass the test, not one that scored zero on it
            # — writing 0.0 here would put a measured-looking constant no
            # probe produced into every receipt (§8 PURITY).
            refused = "farm_unmeasured" if screening else None
        elif farm > MAX_FARM_EVENTS_PER_1K:
            refused = "farmable"
        s["refused"] = refused
    # Deterministic total order that gives no address a privileged
    # position — see rank_sort_key and tie_break_key. Content-addressed
    # throughout, so a permutation of the input cannot permute the output.
    survivors.sort(key=rank_sort_key)
    return survivors


def admissible_rows(ranked: Sequence) -> list:
    """Every BH-significant, unrefused, non-nuisance row, in rank order.

    The list a K0-style grade measures rank against, in FULL: the cut to
    `RANK_CUTOFF` is a separate decision, and a receipt that only records
    the cut cannot answer "did the answer rank at all" (D-TRUNC).
    """
    return [s for s in ranked
            if s.get("significant") and not s.get("refused")
            and s.get("score", 0.0) > 0.0]


def admitted_candidates(ranked: Sequence, cutoff: int = RANK_CUTOFF) -> list:
    """The top `cutoff` admissible ADDRESSES, one row each.

    BY ADDRESS, NOT BY ROW (D-TIE's sibling, D-DUP). A row is an
    (addr, family) pair, so slicing rows spends admission slots on the
    same byte seen through several families: the graded Bubble Bobble
    sweep admitted ['0x0033', '0x0303', '0x0303', '0x0304', '0x0304'] —
    five slots, three addresses. Since the sidecar promotes ADDRESSES to
    state_sig bits, and `rank <= 5` is defined on addresses, the cut has
    to be taken there too. The row kept for an address is its
    best-ranked one, so the family that carried it is still receipted.
    """
    out: list = []
    seen: set = set()
    for s in admissible_rows(ranked):
        addr = int(s["addr"])
        if addr in seen:
            continue
        seen.add(addr)
        out.append(s)
        if len(out) >= int(cutoff):
            break
    return out


def sweep_step_cost(n_patterns: int, n_roots: int, repeats: int,
                    n_phases: int, program_len: int = PROGRAM_LEN_PASS_A
                    ) -> int:
    """Worker-steps one full sweep costs. The displacement figure §6
    budgets against — measured, never inferred from sps."""
    return int(n_patterns) * int(n_roots) * int(repeats) * int(n_phases) \
        * int(program_len)


def sweep_interval_steps(sweep_cost: int, frac: float) -> int:
    """Search steps between sweeps so the sweep consumes `frac` of the
    step budget. frac <= 0 disables (returned as a huge interval)."""
    frac = float(frac)
    if frac <= 0.0:
        return 1 << 62
    if frac >= 1.0:
        return 1
    return max(1, int(round(sweep_cost * (1.0 - frac) / frac)))
