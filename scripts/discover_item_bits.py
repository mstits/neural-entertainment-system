"""Item/capability-bit discovery — Stages 1-3 (ITEM_SEMANTICS_ENGINE_
2026-08-25 §9 Tasks 2-3).

Finds candidate capability FLAGS (a byte or a single bit-plane that
goes from one value to another and never comes back — the boolean
"do you now have the key" shape, as opposed to a counted resource like
rupees) from our own rollout RAM logs, with no address ever supplied
from outside the emulator's own measured behavior — the same
provenance discipline as `discover_observables.py`
(CLAIMS.md/`verify_ram_map.py`). No RAM map, no walkthrough, no
`poke_ram`/`poke_ram_range` (that accessor does not exist in this
codebase and this file must never invent it).

STAGE 1 — per-rollout candidate proposal (`scan_rollout`).
    For every RAM address not already claimed by an existing observable
    and not flagged volatile by the frozen per-lineage idle mask, walk
    its run-length-collapsed value sequence:
      * `_flag_events` counts TRANSITIONS (value changes) and REVERTS
        (a later run repeating a value already seen earlier in this
        same rollout) — a genuine flag transitions once (or a few
        times through a monotone chain) and never reverts;
      * a high per-step CHANGE RATE marks a frame-counter/timer-shaped
        byte (moves on a fixed cadence, not on an event) — rejected
        regardless of monotonicity;
      * every address is ALSO scanned bit-plane-by-bit-plane (0..7),
        for the case where an item byte is a bitfield packing several
        independent flags and only one plane is clean while its
        siblings toggle for unrelated reasons. The bit-plane's
        promotion target still comes out as a `{addr, match, mod}`
        triple (`_bit_plane_match_mod`): `mod = 2**(bit+1)` isolates
        the low `bit+1` bits by construction (`v % 2**(bit+1)` depends
        only on those bits), so `match` is simply "which residues in
        that range have the bit set" — correct regardless of what the
        untested higher bits do, with no need to enumerate the full
        256-value table.
    A raw-value (bit=None) candidate found clean at an address
    supersedes that address's bit-plane scan — the whole byte already
    explains the event, and proposing all eight planes on top of it is
    noise, not evidence.
    DEATH TRUNCATION (`lives_addr`, IS-1a mitigation 1,
    `runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`): when given
    the column index of the SAME lives byte death detection already
    uses elsewhere (`go_explore_solve.py`'s `SmbGame`/`GenericGame`
    `.lives`), the rollout is cut at the first frame whose lives value
    has moved away from the rollout's own starting value by the same
    wrap-aware amount `GenericGame.is_dead` already treats as death
    (`(start - cur) % 256 in 1..8`) BEFORE any address is scanned — so
    a death's post-death menu/HUD RAM rewrite is never read as a
    candidate item-bit flag. No new address; the same one death
    detection already claims.

STAGE 2 — cross-rollout confirmation (`confirm_across_rollouts`).
    Folds N independent rollouts' Stage-1 proposals into one ledger per
    `(addr, bit)` key. `monotone_rollouts` counts how many of the N
    rollouts proposed the key cleanly ("candidate" shape);
    `total_rollouts` is N itself (the batch size), not merely how many
    rollouts happened to mention the key — a rollout in which the
    player never triggers the event is a real, counted "no" vote, not
    an absence. K-of-N stability bar (`confirm_k`, default 3 of N):
    below it the key stays "candidate", at or above it the key
    promotes to "confirmed". REDISCOVERY RULE (same convention as
    `metroid_purity_quarantine_2026-08-10.md`): `reverts_seen > 0` in
    ANY single rollout permanently marks that key "rejected" regardless
    of how many other rollouts confirmed it cleanly — a key does not
    self-heal back to candidate/confirmed; it must be re-derived from
    scratch (dropped from the ledger and resubmitted).
    ROOT-CLOCK ARTIFACT REJECTION (`root_clock_min_lineages`/`root_
    clock_tolerance`, IS-1a mitigation 2,
    `runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`): a real
    gameplay-contingent flag's first-flip step should vary with what
    the player actually did across independently-diverging rollouts —
    a key whose first-flip step lands within `root_clock_tolerance`
    steps across `root_clock_min_lineages`-or-more of the rollouts that
    proposed it cleanly is instead a fixed-timing engine-init artifact
    (the IS-1a residual-13 finding: 8-of-8 surviving rollouts proposing
    the same address at the identical elapsed step despite each
    replaying a genuinely different real trajectory) and is rejected
    with `root_clock_artifact = True`, never promoted on K-of-N count
    alone.

IDLE-PREFILTER (`compute_idle_mask`, INDEPENDENT §2). A mask, frozen
once per lineage, of every address that moves AT ALL during genuine
idle play — computed the same way `room_fp_calibrate.py`'s volatility
mask is (variance over our own captures, never a hand-picked address).
Feed it into `scan_rollout`/`confirm_across_rollouts` as `idle_mask=`
to exclude those addresses BEFORE the expensive monotonicity scan
runs. `idle_mask_from_rom` builds it live by reusing
`discover_observables.Discoverer.idle()` — a LIBRARY IMPORT, never a
patch (§3 row 7) — and is imported lazily so pulling in this module's
pure Stage 1-2 functions never requires `nes_core` to be built.

STAGE 3a — cap_hist / boundary-visit correlational read
(`correlate_boundary_edges`). Reads an already-armed `RoomIndex`'s
`cap_hist` (the `EdgeStat.cap_hist` graft, Task 1) for edges whose
crossings are lopsided toward one value of a candidate's bit position
in `cap_sig` — a LEAD, never a claim. THE CONFOUND NAMED IN §7 ITEM 1
("rarity vs. gating"): an edge lopsided toward bit=1 may simply be far
from where bit=0 workers ever wander, for reasons that have nothing to
do with any gate (e.g. the candidate flips almost immediately in
nearly every rollout, so almost no rollout-steps exist ANYWHERE in the
graph at bit=0). This function cannot tell the two apart by itself —
it reports whole-graph `exposure_bit0`/`exposure_bit1` alongside every
lead precisely so a caller sees the confound rather than trusting the
correlation blind (see `test_is0_7_stage3a_alone_is_fooled_by_a_
rarity_confound`, which demonstrates this function proposing a lead
from a scenario with no real gate at all).

STAGE 3b — `verify_behavioral` (§2C), the ONLY thing in this engine
allowed to turn a Stage-3a lead into a claim. A matched REAL
acquire/skip trajectory PAIR — both mined from the ledger's own
rollout history, never authored — is replayed from a SHARED pre-flip
`save_worker_state` blob, each followed by the same `probe_actions`.
Because both arms start from the IDENTICAL point in space and time,
"how close sig=0 workers get to the boundary in ordinary undirected
search" is controlled away BY CONSTRUCTION, not by a threshold — this
is what makes Track A a structural answer to the rarity-vs-gating
confound rather than a better-tuned version of the same correlation
(see `test_is0_7_verify_behavioral_resolves_the_rarity_confound_stage3a_
missed`). `control_bits` REJECTED-pool candidates (§7 item 8,
`select_control_candidates`) are scored for free off the SAME two
replayed lanes — no extra `Pool.step_all` call — as the false-positive
control: if an address KNOWN not to be a monotone capability flag
(reverted at least once, permanently rejected by Stage 2) ALSO swings
between the acquire and skip arms, the acquire/skip split itself is
confounded (these are just two generically-different real
trajectories) and no bit can be validated from it, target included.
`n_trials` replicate waves of the IDENTICAL (pre_state, actions) tuple
are run and must agree — since NES replay is bit-exact deterministic,
a real disagreement across replicates is a live-Pool integrity fault,
surfaced as `nondeterministic: True` and an `inconclusive` verdict,
never averaged into a false confidence number. Only primitives both
source designs confirm are already shipped:
`pool.save_worker_state`/`load_worker_state` (`pool.rs:1515`/`1542`)
and `pool.step_all` (`python.rs:380`) — no new Rust, no
`poke_ram`/`poke_ram_range` (still unbuilt, deferred to v2 per §8).

NOT IN THIS FILE (Tasks 4-5): any live per-game rollout collection, IS
-1a/IS-1b runs, or receipt-filing.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: System RAM window this whole engine is scoped to (`get_ram_range`,
#: `python.rs:709`, hard `PyValueError` above this — §5). A candidate
#: address is never generated outside `range(RAM_SIZE)`.
RAM_SIZE = 0x800

#: Profile `solve.item_bits:` defaults (§4) — a profile only overrides
#: what it names; everything else keeps the script default.
DEFAULT_MAX_CHANGE_RATE = 0.01
DEFAULT_CONFIRM_K = 3
DEFAULT_CONFIRM_N = 5
DEFAULT_MAX_ITEM_BITS = 8
DEFAULT_VERIFY = {"n_trials": 20, "control_bits": 5,
                  "verdict_gap": 0.5, "control_flatness": 0.2}

#: Cross-lineage step-offset consistency gate (IS1A_VERDICT_2026-08-25.md
#: mitigation 2) — see `confirm_across_rollouts`. The default is 8, not
#: `DEFAULT_CONFIRM_K`'s 3: the IS-1a residual-13 finding's own evidence
#: bar was "8 of the 8 (or more)" surviving rollouts sharing an
#: identical first-flip step, and a bar this low must stay well above
#: the small (3-5 rollout) batch sizes this file's own K-of-N tests use
#: for their synthetic fixtures — those intentionally reuse one fixed
#: offset across every "hit" rollout to isolate the counting logic
#: under test, which is indistinguishable, by step-offset alone, from a
#: real root-clock artifact at a low lineage count.
DEFAULT_ROOT_CLOCK_MIN_LINEAGES = 8
DEFAULT_ROOT_CLOCK_TOLERANCE = 0

STATUSES = ("candidate", "confirmed", "rejected")


# ---------------------------------------------------------------------
# §2B — the offline discovery ledger entry.
# ---------------------------------------------------------------------
@dataclass
class ItemBitCandidate:
    """One `(addr, bit)` candidate's accumulated evidence.

    `bit=None` is a raw-value (state_sig-style) candidate — the whole
    byte is the flag. `bit=<int 0..7>` is a bit-plane candidate — one
    plane of the byte is the flag while the rest of the byte is not
    trusted. Either way the promotion target a profile's `state_sig:`
    consumes is the same `{addr, match, mod}` triple.

    `root_clock_artifact` names WHY a `"rejected"` status was assigned
    when the reason is the cross-lineage step-offset check
    (`confirm_across_rollouts`'s `root_clock_min_lineages`/`root_clock_
    tolerance`, IS1A_VERDICT_2026-08-25.md mitigation 2) rather than
    `reverts_seen`/a low `monotone_rollouts` count — the same
    named-reason convention `idle_excluded`/`reverts_seen`/`change_rate`
    already use.
    """
    addr: int
    bit: Optional[int] = None
    match: frozenset = field(default_factory=frozenset)
    mod: int = 0
    idle_excluded: bool = False
    reverts_seen: int = 0
    change_rate: float = 0.0
    monotone_rollouts: int = 0
    total_rollouts: int = 0
    status: str = "candidate"
    first_seen: Optional[dict] = None
    root_clock_artifact: bool = False

    @property
    def key(self) -> tuple:
        return (int(self.addr), self.bit)

    def to_dict(self) -> dict:
        return {
            "addr": int(self.addr),
            "bit": (int(self.bit) if self.bit is not None else None),
            "match": sorted(int(v) for v in self.match),
            "mod": int(self.mod),
            "idle_excluded": bool(self.idle_excluded),
            "reverts_seen": int(self.reverts_seen),
            "change_rate": float(self.change_rate),
            "monotone_rollouts": int(self.monotone_rollouts),
            "total_rollouts": int(self.total_rollouts),
            "status": str(self.status),
            "first_seen": self.first_seen,
            "root_clock_artifact": bool(self.root_clock_artifact),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "ItemBitCandidate":
        return cls(
            addr=int(d["addr"]),
            bit=(int(d["bit"]) if d.get("bit") is not None else None),
            match=frozenset(int(v) for v in d.get("match", ())),
            mod=int(d.get("mod", 0)),
            idle_excluded=bool(d.get("idle_excluded", False)),
            reverts_seen=int(d.get("reverts_seen", 0)),
            change_rate=float(d.get("change_rate", 0.0)),
            monotone_rollouts=int(d.get("monotone_rollouts", 0)),
            total_rollouts=int(d.get("total_rollouts", 0)),
            status=str(d.get("status", "candidate")),
            first_seen=d.get("first_seen"),
            root_clock_artifact=bool(d.get("root_clock_artifact", False)),
        )


# ---------------------------------------------------------------------
# Pure run-length / flag-shape math (no ROM, no Pool — testable directly).
# ---------------------------------------------------------------------
def _run_collapse(values: np.ndarray) -> list:
    """Consecutive repeats collapsed to one entry per run."""
    runs = [int(values[0])]
    for v in values[1:]:
        iv = int(v)
        if iv != runs[-1]:
            runs.append(iv)
    return runs


def _flag_events(values: np.ndarray) -> tuple:
    """`(transitions, reverts, first_transition_step)` over ONE channel
    (a raw byte column or a single bit-plane's 0/1 column) for ONE
    rollout.

    `transitions` = number of run boundaries (value changes).
    `reverts` = number of runs whose value repeats a value ALREADY SEEN
    in an earlier run — the flicker signature (`1 -> 0 -> 1`) as well
    as a wrapping/free-running counter that eventually revisits an old
    value. `first_transition_step` is the index (into the ORIGINAL,
    uncollapsed array) of the first row whose value differs from
    `values[0]`, or None if the channel never changes.
    """
    n = len(values)
    if n == 0:
        return 0, 0, None
    runs = _run_collapse(values)
    seen = {runs[0]}
    reverts = 0
    for v in runs[1:]:
        if v in seen:
            reverts += 1
        seen.add(v)
    transitions = len(runs) - 1
    if transitions == 0:
        return 0, 0, None
    v0 = int(values[0])
    first_step = next(i for i in range(1, n) if int(values[i]) != v0)
    return transitions, reverts, first_step


def _bit_plane_match_mod(bit: int) -> tuple:
    """`(match, mod)` for "bit `bit` of the raw byte is set", expressed
    as the `{addr, match, mod}` triple `state_sig` already runs
    (`v % mod in match`). `mod = 2**(bit+1)` isolates exactly the low
    `bit+1` bits of the raw value (modulo a power of two keeps the low
    bits unchanged and zeroes the rest), so this is correct for EVERY
    raw byte value regardless of what its higher, untested bits do —
    no need to observe all 256 combinations to build a complete table.
    """
    width = 1 << (bit + 1)
    match = frozenset(v for v in range(width) if (v >> bit) & 1)
    return match, width


# ---------------------------------------------------------------------
# Idle-prefilter (INDEPENDENT §2) — a frozen per-lineage volatility mask.
# ---------------------------------------------------------------------
def compute_idle_mask(idle) -> frozenset:
    """Every RAM (or, for the offline falsifier, NT-shaped) address
    that changes value AT LEAST ONCE across a genuinely idle capture.

    `idle` is either a plain `(steps, ncols)` array or a
    `(log, keep)` pair exactly like `discover_observables.Discoverer.
    idle()` returns — `keep` masks diff rows that span a reload, so a
    reload's RAM rewrite is never read as idle "movement". Mirrors
    `room_fp_calibrate.py`'s auto-volatility mask: a byte earns
    exclusion by MEASURED variance over a real capture, never by a
    hand-picked address.
    """
    if isinstance(idle, tuple):
        log, keep = idle
        keep = np.asarray(keep, dtype=bool)
    else:
        log, keep = np.asarray(idle), None
    if log.shape[0] < 2:
        return frozenset()
    diffs = np.diff(log.astype(np.int16), axis=0) != 0
    if keep is not None:
        diffs = diffs[keep]
    if diffs.shape[0] == 0:
        return frozenset()
    return frozenset(int(i) for i in np.where(diffs.any(axis=0))[0])


def idle_mask_from_rom(rom, state, *, frame_skip: int = 4,
                       forward: str = "right", seed: int = 1) -> frozenset:
    """Live convenience: build the frozen idle mask for one lineage by
    reusing `discover_observables.Discoverer.idle()` — a LIBRARY
    IMPORT, never a patch (§3 row 7). Imported lazily so importing this
    module's pure Stage 1-2 functions never requires `nes_core` to be
    built; one short single-worker idle probe, not a hot-loop presence.
    """
    from scripts.discover_observables import Discoverer
    disc = Discoverer(rom, state, frame_skip=frame_skip, forward=forward,
                      seed=seed)
    try:
        return compute_idle_mask(disc.idle())
    finally:
        disc.close()


# ---------------------------------------------------------------------
# STAGE 1 — per-rollout candidate proposal.
# ---------------------------------------------------------------------
def scan_rollout(log: np.ndarray, *, rollout_id=None, xy: Optional[np.ndarray] = None,
                 claimed_addrs: frozenset = frozenset(),
                 idle_mask: frozenset = frozenset(),
                 max_change_rate: float = DEFAULT_MAX_CHANGE_RATE,
                 scan_bits: bool = True,
                 addrs: Optional[Iterable[int]] = None,
                 lives_addr: Optional[int] = None) -> dict:
    """Propose `(addr, bit)` candidates from ONE rollout's RAM log
    (`log[step, addr]`, uint8-valued). Pure — no ROM, no Pool.

    Returns `{(addr, bit): ItemBitCandidate}` for every address that
    changed at least once and was not excluded by `claimed_addrs`
    (already-owned observables — dedup, §6 item 4) or `idle_mask`
    (§6 item 5). A candidate's `status` is `"candidate"` when it is
    flag-shaped (no revert, change rate within budget) and `"rejected"`
    otherwise (`reverts_seen` and/or `change_rate` name why) — an
    address that never changes at all is not reported either way, it
    is simply inert over this rollout.

    `lives_addr`, when given, is the column index of the SAME lives
    byte this project's own death detection already keys off of
    (`go_explore_solve.py`'s `SmbGame.lives`/`GenericGame.lives` — a
    profile's already-claimed `solve.lives:` address, never a new one
    discovered here). The rollout is truncated to the steps strictly
    BEFORE the first frame whose lives value has moved away from
    `log[0, lives_addr]` by the same wrap-aware amount
    `GenericGame.is_dead` already treats as death (`(start - cur) % 256
    in 1..8` — the 2026-08-23 Ninja Gaiden underflow fix, robust to
    both a plain decrement and a 0 -> 255 wrap). Without this, a
    rollout that ends in death keeps scanning through the post-death
    menu's RAM rewrite and reports its unrelated HUD/menu-state churn
    as candidate item-bit flags — the IS-1a root cause
    (`runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`).
    """
    log = np.asarray(log)
    if (lives_addr is not None and log.shape[0] > 0
            and 0 <= lives_addr < log.shape[1]):
        lives_col = log[:, lives_addr].astype(np.int64)
        delta = (int(lives_col[0]) - lives_col) % 256
        died = np.flatnonzero((delta >= 1) & (delta <= 8))
        if died.size:
            log = log[:int(died[0])]
    n = log.shape[0]
    ncols = min(log.shape[1], RAM_SIZE)
    scan_addrs = range(ncols) if addrs is None else addrs
    out: dict = {}

    def _first_seen(step: Optional[int]) -> Optional[dict]:
        if step is None:
            return None
        pos = None
        if xy is not None and step < len(xy):
            pos = [int(xy[step][0]), int(xy[step][1])]
        return {"rollout_id": rollout_id, "step": int(step),
               "position": pos if pos is not None else [None, None]}

    for addr in scan_addrs:
        if addr in claimed_addrs or addr in idle_mask:
            continue
        raw = log[:, addr]
        transitions, reverts, first_step = _flag_events(raw)
        if transitions == 0:
            continue
        change_rate = transitions / max(n - 1, 1)
        rejected = bool(reverts > 0 or change_rate > max_change_rate)
        runs = _run_collapse(raw)
        match = frozenset(runs[1:])
        cand = ItemBitCandidate(
            addr=int(addr), bit=None, match=match, mod=0,
            idle_excluded=False, reverts_seen=int(reverts),
            change_rate=float(change_rate),
            monotone_rollouts=(0 if rejected else 1), total_rollouts=1,
            status=("rejected" if rejected else "candidate"),
            first_seen=_first_seen(first_step))
        out[cand.key] = cand
        if not scan_bits or not rejected:
            # scan_bits=False: never plane-scan. A clean (accepted) raw
            # read already explains the byte, so the whole byte covers
            # every plane and a per-bit scan would only add redundant
            # proposals — skip it. Only a REJECTED raw read (revert or
            # timer-shaped) falls through to give an individual plane a
            # chance to redeem an otherwise-noisy byte.
            continue
        for bit in range(8):
            bitvals = (raw.astype(np.int64) >> bit) & 1
            bt, br, bstep = _flag_events(bitvals)
            if bt == 0:
                continue
            bchange = bt / max(n - 1, 1)
            brejected = bool(br > 0 or bchange > max_change_rate)
            bmatch, bmod = _bit_plane_match_mod(bit)
            bcand = ItemBitCandidate(
                addr=int(addr), bit=int(bit), match=bmatch, mod=bmod,
                idle_excluded=False, reverts_seen=int(br),
                change_rate=float(bchange),
                monotone_rollouts=(0 if brejected else 1), total_rollouts=1,
                status=("rejected" if brejected else "candidate"),
                first_seen=_first_seen(bstep))
            out[bcand.key] = bcand
    return out


# ---------------------------------------------------------------------
# STAGE 2 — cross-rollout confirmation.
# ---------------------------------------------------------------------
def confirm_across_rollouts(rollout_logs: Sequence[np.ndarray], *,
                            confirm_k: int = DEFAULT_CONFIRM_K,
                            claimed_addrs: frozenset = frozenset(),
                            idle_mask: frozenset = frozenset(),
                            max_change_rate: float = DEFAULT_MAX_CHANGE_RATE,
                            scan_bits: bool = True,
                            rollout_ids: Optional[Sequence] = None,
                            xy_logs: Optional[Sequence] = None,
                            lives_addr: Optional[int] = None,
                            root_clock_min_lineages: int =
                                DEFAULT_ROOT_CLOCK_MIN_LINEAGES,
                            root_clock_tolerance: int =
                                DEFAULT_ROOT_CLOCK_TOLERANCE) -> dict:
    """Fold `len(rollout_logs)` independent Stage-1 scans into one
    ledger keyed by `(addr, bit)`.

    `total_rollouts` is ALWAYS the batch size `N` — a rollout that
    never triggers the event is a counted "no", not a rollout the key
    was never exposed to. `monotone_rollouts` is how many of those `N`
    proposed the key cleanly. K-of-N bar: `monotone_rollouts >=
    confirm_k` promotes to `"confirmed"`; `1 <= monotone_rollouts <
    confirm_k` stays `"candidate"`; a key that never proposed cleanly
    anywhere (only ever seen rejected) reads `"rejected"`.

    REDISCOVERY RULE: `reverts_seen > 0` in ANY single rollout's scan
    permanently overrides the key to `"rejected"` regardless of how
    many other rollouts confirmed it — this is checked LAST, after the
    K-of-N arithmetic, so a revert can never be out-voted.

    `lives_addr` is forwarded verbatim to `scan_rollout` on every
    rollout — see its docstring. Each rollout is truncated at its OWN
    death boundary before scanning, so a mixed batch (some rollouts die
    in-window, some don't) is handled per-rollout, not batch-wide.

    ROOT-CLOCK ARTIFACT REJECTION (IS-1a mitigation 2,
    `runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`): a
    gameplay-contingent flag fires at a step that depends on what the
    player actually did, so its first-flip step should VARY across
    independently-diverging rollouts. A key whose per-rollout
    `first_seen["step"]` lands within `root_clock_tolerance` steps of
    every other rollout's, across at least `root_clock_min_lineages` of
    the rollouts that proposed it cleanly, is instead the signature of
    a deterministic engine-init/sound-engine/animation-parity timer
    that fires at the same elapsed step regardless of the 8+
    independently-mined real trajectories that triggered it (the IS-1a
    residual-13 root cause) — rejected here with `root_clock_artifact
    = True` rather than promoted, exactly the same way a revert is
    rejected regardless of an otherwise-passing K-of-N count. Additive:
    a key seen candidate-shaped in fewer than `root_clock_min_lineages`
    rollouts, or whose first-flip steps actually spread out (real
    gameplay-contingent timing), is completely unaffected.
    """
    n = len(rollout_logs)
    ledger: dict = {}
    ever_reverted: set = set()
    first_steps: dict = {}

    for i, log in enumerate(rollout_logs):
        rid = rollout_ids[i] if rollout_ids is not None else i
        xy = xy_logs[i] if xy_logs is not None else None
        per_roll = scan_rollout(
            log, rollout_id=rid, xy=xy, claimed_addrs=claimed_addrs,
            idle_mask=idle_mask, max_change_rate=max_change_rate,
            scan_bits=scan_bits, lives_addr=lives_addr)
        for key, cand in per_roll.items():
            if cand.reverts_seen > 0:
                ever_reverted.add(key)
            entry = ledger.get(key)
            if entry is None:
                entry = ItemBitCandidate(addr=cand.addr, bit=cand.bit,
                                         match=frozenset(), mod=cand.mod)
                ledger[key] = entry
            entry.match = entry.match | cand.match
            entry.reverts_seen = max(entry.reverts_seen, cand.reverts_seen)
            entry.change_rate = max(entry.change_rate, cand.change_rate)
            if cand.status == "candidate":
                entry.monotone_rollouts += 1
                if entry.first_seen is None:
                    entry.first_seen = cand.first_seen
                if cand.first_seen is not None:
                    first_steps.setdefault(key, []).append(
                        cand.first_seen["step"])

    root_clock_flagged: set = set()
    # A floor of 2: a single lineage's first-flip step has zero spread
    # by definition, so "identical across lineages" is meaningless
    # (and would falsely flag every single-rollout candidate) below 2.
    min_lineages = max(2, int(root_clock_min_lineages))
    tolerance = max(0, int(root_clock_tolerance))
    for key, steps in first_steps.items():
        if len(steps) >= min_lineages and (max(steps) - min(steps)) <= tolerance:
            root_clock_flagged.add(key)

    for key, entry in ledger.items():
        entry.total_rollouts = n
        if key in ever_reverted:
            entry.status = "rejected"
        elif key in root_clock_flagged:
            entry.status = "rejected"
            entry.root_clock_artifact = True
        elif entry.monotone_rollouts >= confirm_k:
            entry.status = "confirmed"
        elif entry.monotone_rollouts >= 1:
            entry.status = "candidate"
        else:
            entry.status = "rejected"
    return ledger


# ---------------------------------------------------------------------
# STAGE 3a — cap_hist / boundary-visit correlational read.
# ---------------------------------------------------------------------
def correlate_boundary_edges(room_index, bit_index: int, *,
                             min_bit0_count: int = 0) -> list:
    """A CHEAP, PASSIVE first filter over an already-armed `RoomIndex`'s
    `cap_hist` (`EdgeStat.cap_hist`, Task 1's graft) — no live compute
    of its own, just a read of edges a `--item-sig-report` run already
    committed. `room_index` may be a live `RoomIndex` instance or its
    `to_json()`-shaped dict (str-keyed `adj`) — both read the same way.

    `bit_index` is the candidate's position in the profile's
    `state_sig:` list AT THE TIME that run collected data (§3 row 8: a
    confirmed candidate is appended to that list before the
    measurement burst, exactly like Castlevania's existing bits, and
    `GenericGame.cell_fn` — `go_explore_solve.py:2278` — folds each
    entry into `cap_sig` as `sig |= 1 << i` in list order) — so
    `cap_sig`'s bit `bit_index` is 1 exactly when this candidate's
    `{addr, match, mod}` condition held at the moment an edge staged.

    Returns every edge whose crossings are lopsided toward bit=1 —
    `count_bit1 > 0` and `count_bit0 <= min_bit0_count` (default: LITERALLY
    zero crossings ever recorded at bit=0) — sorted strongest lead
    first. Every returned lead additionally carries `exposure_bit0`/
    `exposure_bit1` (the WHOLE-GRAPH totals at each bit value, across
    every edge, not just this one) and `confound_ratio` (`exposure_bit0
    / total`) — a lead whose own counts look damning but whose
    `exposure_bit0` is a tiny sliver of total traffic is exactly the
    shape the rarity confound (§7 item 1) produces, and this function
    cannot tell that apart from a real gate by itself. `verify_
    behavioral` (Stage 3b, below) is the only thing in this engine
    allowed to turn a lead into a claim — see this module's docstring
    and `test_is0_7_stage3a_alone_is_fooled_by_a_rarity_confound`.
    """
    adj = room_index.adj if hasattr(room_index, "adj") else (room_index or {}).get("adj", {})
    exposure_bit0 = 0
    exposure_bit1 = 0
    rows = []
    for src, dsts in adj.items():
        for dst, e in dsts.items():
            cap_hist = e.get("cap_hist") or {}
            c0 = c1 = 0
            for k_str, cnt in cap_hist.items():
                cnt = int(cnt)
                if (int(k_str) >> bit_index) & 1:
                    c1 += cnt
                else:
                    c0 += cnt
            exposure_bit0 += c0
            exposure_bit1 += c1
            rows.append({"src": int(src), "dst": int(dst),
                        "kind": e.get("kind"), "dir": e.get("dir"),
                        "count_bit0": c0, "count_bit1": c1})
    total = max(exposure_bit0 + exposure_bit1, 1)
    leads = []
    for r in rows:
        if r["count_bit1"] > 0 and r["count_bit0"] <= min_bit0_count:
            r = dict(r, exposure_bit0=exposure_bit0, exposure_bit1=exposure_bit1,
                     confound_ratio=exposure_bit0 / total)
            leads.append(r)
    leads.sort(key=lambda r: (-r["count_bit1"], r["count_bit0"]))
    return leads


# ---------------------------------------------------------------------
# STAGE 3b — verify_behavioral (§2C): Track A behavioral verification.
# ---------------------------------------------------------------------
def select_control_candidates(ledger: Mapping, primary_key, k: int, *,
                              rng: Optional[random.Random] = None) -> list:
    """§7 item 8 (INDEPENDENT §7.9's collision-avoidance rule): draw up
    to `k` controls preferentially from the ledger's REJECTED pool.
    A permanently-rejected key (`reverts_seen > 0` somewhere, Stage 2's
    rediscovery rule) is guaranteed non-causal BY CONSTRUCTION — a flag
    that reverts is not a monotone capability flag at all — so a
    control drawn from there can never turn out to secretly be a real,
    different capability flag skewing the flatness check for the wrong
    reason. Falls back to non-rejected entries only when the rejected
    pool has fewer than `k`. Never returns `primary_key` itself. Pure
    and deterministic for a given `rng` (defaults to a fixed seed so a
    repeated call over the same ledger reproduces the same draw).
    """
    rng = rng or random.Random(0)
    others = [c for key, c in ledger.items() if key != primary_key]
    rejected = [c for c in others if c.status == "rejected"]
    rest = [c for c in others if c.status != "rejected"]
    rng.shuffle(rejected)
    rng.shuffle(rest)
    return (rejected + rest)[:max(0, int(k))]


def _ram_from_step(cell) -> np.ndarray:
    """Normalize one `pool.step_all()` worker result to a `uint8[:RAM_
    SIZE]` array — the same `results[wid][2]` convention `Solver.
    _gate_wave`/`_step0` and `discover_observables.Discoverer._step`
    already use; introduces no new access pattern."""
    return np.frombuffer(bytes(cell[2]), dtype=np.uint8)[:RAM_SIZE]


def _eval_candidate(ram: np.ndarray, cand: ItemBitCandidate) -> bool:
    """Does `cand`'s own `{addr, match, mod}` promotion condition hold
    in this RAM snapshot? The same `v % mod in match` shape `state_sig`
    and `_bit_plane_match_mod` already use."""
    v = int(ram[int(cand.addr)])
    if cand.mod:
        v = v % int(cand.mod)
    return v in cand.match


def make_probe_pool(rom, *, n_workers: int, frame_skip: int = 4):
    """A small, DEDICATED `nes_core.Pool` sized to exactly the lanes one
    `verify_behavioral` battery needs (always 2 — see `_run_matched_
    wave`) — NEVER the live Solver's shared pool, so a battery can
    never perturb a live search's worker state. Lazy import, same
    convention as `idle_mask_from_rom` (§3 row 7): importing this
    module's pure Stage 1-3a logic never requires `nes_core` to be
    built."""
    from nes_core import Pool
    pool = Pool(rom_path=str(rom), num_workers=int(n_workers),
               frame_skip=int(frame_skip))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    return pool


def _run_matched_wave(pool, pre_state: bytes, arms: Sequence[tuple],
                      probe_actions: Sequence[int],
                      reach_fn: Callable[[np.ndarray], bool],
                      control_candidates: Sequence[ItemBitCandidate]) -> dict:
    """One lockstep wave: `arms` is `[(tag, prefix_actions), ...]` — for
    `verify_behavioral` always exactly `[("acquire", ...), ("skip",
    ...)]`, the whole battery. Every lane loads the IDENTICAL
    `pre_state` (Ecoffet et al.'s return-then-explore point) — the
    structural device that answers the rarity-vs-gating confound
    (§7 item 1): a naive `cap_hist` read (Stage 3a) compares sig=0 vs
    sig!=0 exposure ACROSS THE WHOLE RUN, so a boundary that sig=0
    workers simply never wander near — for reasons that have nothing
    to do with any gate — reads identically to a genuinely gated one.
    Here, both arms start from the SAME point in space and time; only
    what happens AFTER differs (whether the real trajectory picked up
    the candidate or walked past it) — proximity to the boundary is
    controlled away BY CONSTRUCTION, not by a threshold.

    Each control candidate's OWN `{addr, match, mod}` condition is
    scored for free off these SAME two lanes' RAM trace — no extra
    `pool.step_all` call — as the false-positive control (§7 item 8).

    `reach_fn`/each control condition is evaluated at every PROBE frame
    (i.e. from the end of that lane's own mined prefix onward, since
    the two arms' prefixes may differ in length) and OR'd across the
    whole probe window — a transient crossing counts as a reach, not
    only a final-frame read. Returns `{"reach": {tag: bool}, "control":
    {tag: [bool, ...]}}`.
    """
    scripts = [list(prefix) + list(probe_actions) for _tag, prefix in arms]
    probe_from = [len(prefix) for _tag, prefix in arms]
    n = len(arms)
    for wid in range(n):
        pool.load_worker_state(wid, pre_state)
    reach_hit = [False] * n
    ctrl_hit = [[False] * len(control_candidates) for _ in range(n)]
    max_len = max((len(s) for s in scripts), default=0)
    acts = np.zeros(n, dtype=np.uint8)
    for t in range(max_len):
        acts[:] = 0
        for wid, s in enumerate(scripts):
            if t < len(s):
                acts[wid] = int(s[t]) & 0xFF
        results = pool.step_all(acts)
        for wid in range(n):
            if t < probe_from[wid]:
                continue
            ram = _ram_from_step(results[wid])
            if reach_fn(ram):
                reach_hit[wid] = True
            for ci, cand in enumerate(control_candidates):
                if _eval_candidate(ram, cand):
                    ctrl_hit[wid][ci] = True
    tags = [tag for tag, _ in arms]
    return {"reach": dict(zip(tags, reach_hit)),
            "control": dict(zip(tags, ctrl_hit))}


def verify_behavioral(pool, candidate: ItemBitCandidate, *, pre_state: bytes,
                      acquire_actions: Sequence[int],
                      skip_actions: Sequence[int],
                      probe_actions: Sequence[int],
                      reach_fn: Callable[[np.ndarray], bool],
                      n_trials: int = 20, control_bits: int = 5,
                      control_candidates: Sequence[ItemBitCandidate] = (),
                      verdict_gap: float = DEFAULT_VERIFY["verdict_gap"],
                      control_flatness: float = DEFAULT_VERIFY["control_flatness"],
                      ) -> dict:
    """Stage 3b (§2C). From a SHARED pre-flip save-state
    (`pool.save_worker_state`), replay ONE matched real {ACQUIRE, SKIP}
    action-sequence PAIR — both mined from the ledger's own rollout
    history, never authored — each followed by the same `probe_
    actions`, `n_trials` times, plus up to `control_bits` REJECTED-pool
    candidates (`select_control_candidates`) scored for free off the
    SAME two lanes, as the false-positive control (§7 item 8).
    `reach_fn(ram) -> bool` names what "reached" means (e.g. crossed a
    specific room edge, entered a specific screen) — necessarily
    supplied by the caller, since this is game-specific, but it is a
    plain predicate over already-read RAM (`get_ram_range`'s own
    window), never a new observable and never game-internals knowledge.

    NEVER FABRICATES A TRAJECTORY (§7 item 9): if `acquire_actions` or
    `skip_actions` is empty — no real skip segment (or no real acquire
    segment) exists in the ledger's rollout history for this candidate
    — returns `verdict: 'inconclusive'` immediately, before touching
    `pool` at all.

    DETERMINISM IS A FIRST-CLASS CHECK, NOT AN ASSUMPTION: replaying
    the IDENTICAL `(pre_state, actions)` tuple `n_trials` times against
    a real NES must reproduce the identical outcome every time (this
    codebase's own byte-identity discipline) — a real disagreement
    across replicates is a live-`Pool` integrity fault, reported as
    `nondeterministic: True` with an `inconclusive` verdict, never
    averaged into a false confidence number.

    Verdict: `'validated'` iff `reach_rate_acquire - reach_rate_skip >=
    verdict_gap` AND `max(control_reach_rates) - min(control_reach_
    rates) < control_flatness` AND at least one control was actually
    scored (zero controls ⇒ `'inconclusive'`, never a default pass —
    §7 item 8's guard is not optional). `control_reach_rates[i]` is
    control candidate `i`'s OWN `abs(acquire_rate - skip_rate)` — a
    REJECTED (definitely non-causal) bit should show close to zero
    swing between the two arms; a wide spread among controls means the
    acquire/skip split itself is confounded by something ELSE that
    differs between the two real trajectories (both generically, and
    thus for the primary candidate too), not by this bit's gating —
    the same failure mode `test_is0_7_verify_behavioral_resolves_the_
    rarity_confound_stage3a_missed` demonstrates directly.
    """
    if not acquire_actions or not skip_actions:
        return {"reach_rate_acquire": None, "reach_rate_skip": None,
                "control_reach_rates": [], "nondeterministic": False,
                "verdict": "inconclusive",
                "reason": ("no real acquire/skip segment exists in the "
                          "ledger's rollout history for this candidate "
                          "— returning inconclusive rather than "
                          "fabricating one (§7 item 9)")}
    controls = list(control_candidates)[:max(0, int(control_bits))]
    arms = [("acquire", list(acquire_actions)), ("skip", list(skip_actions))]
    trials = max(1, int(n_trials))
    reach_counts = {"acquire": 0, "skip": 0}
    ctrl_counts = {"acquire": [0] * len(controls), "skip": [0] * len(controls)}
    outcomes = set()
    for _ in range(trials):
        out = _run_matched_wave(pool, pre_state, arms, probe_actions,
                                reach_fn, controls)
        outcomes.add((out["reach"]["acquire"], out["reach"]["skip"],
                     tuple(out["control"]["acquire"]),
                     tuple(out["control"]["skip"])))
        reach_counts["acquire"] += int(out["reach"]["acquire"])
        reach_counts["skip"] += int(out["reach"]["skip"])
        for ci in range(len(controls)):
            ctrl_counts["acquire"][ci] += int(out["control"]["acquire"][ci])
            ctrl_counts["skip"][ci] += int(out["control"]["skip"][ci])
    nondeterministic = len(outcomes) > 1
    reach_rate_acquire = reach_counts["acquire"] / trials
    reach_rate_skip = reach_counts["skip"] / trials
    control_reach_rates = [
        abs(ctrl_counts["acquire"][ci] / trials - ctrl_counts["skip"][ci] / trials)
        for ci in range(len(controls))]
    gap = reach_rate_acquire - reach_rate_skip
    result = {"reach_rate_acquire": reach_rate_acquire,
             "reach_rate_skip": reach_rate_skip,
             "control_reach_rates": control_reach_rates,
             "nondeterministic": nondeterministic}
    if nondeterministic:
        result["verdict"] = "inconclusive"
        result["reason"] = (f"{len(outcomes)} distinct outcomes across "
                            f"{trials} identical replicate replays of "
                            f"the same (pre_state, actions) — a live-Pool "
                            f"determinism fault, never averaged into a "
                            f"confidence number")
    elif not controls:
        result["verdict"] = "inconclusive"
        result["reason"] = ("no control candidates supplied — cannot "
                            "bound the false-positive rate (§7 item 8)")
    elif (gap >= verdict_gap
          and (max(control_reach_rates) - min(control_reach_rates))
             < control_flatness):
        result["verdict"] = "validated"
        result["reason"] = None
    else:
        result["verdict"] = "rejected"
        result["reason"] = (
            f"acquire/skip reach gap {gap:.2f} "
            f"{'below' if gap < verdict_gap else 'present'} the "
            f"{verdict_gap} bar, or the {len(controls)} control(s) "
            f"were not flat (spread "
            f"{(max(control_reach_rates) - min(control_reach_rates)):.2f} "
            f"vs the {control_flatness} bar)")
    return result


# ---------------------------------------------------------------------
# Profile schema (§4) — script defaults, overridden per-key by whatever
# a profile's `solve.item_bits:` block names. Read ONLY by this offline
# script; never touches the live solver process.
# ---------------------------------------------------------------------
def merge_item_bits_config(profile: Mapping) -> dict:
    cfg = dict((profile or {}).get("solve", {}).get("item_bits", {}) or {})
    verify = dict(DEFAULT_VERIFY)
    verify.update(cfg.get("verify", {}) or {})
    return {
        "max_change_rate": float(cfg.get("max_change_rate",
                                         DEFAULT_MAX_CHANGE_RATE)),
        "confirm_k": int(cfg.get("confirm_k", DEFAULT_CONFIRM_K)),
        "confirm_n": int(cfg.get("confirm_n", DEFAULT_CONFIRM_N)),
        "max_item_bits": int(cfg.get("max_item_bits",
                                     DEFAULT_MAX_ITEM_BITS)),
        "verify": verify,
    }


# ---------------------------------------------------------------------
# Ledger persistence (§2B) — mirrors room_fp_calibrate.py's convention:
# a raw JSON ledger a later stage/run can reload and keep folding into.
# ---------------------------------------------------------------------
def save_ledger(path, ledger: Mapping) -> None:
    items = sorted(ledger.values(),
                   key=lambda c: (c.addr, -1 if c.bit is None else c.bit))
    Path(path).write_text(json.dumps(
        {"candidates": [c.to_dict() for c in items]}, indent=2, sort_keys=True))


def load_ledger(path) -> dict:
    data = json.loads(Path(path).read_text())
    out = {}
    for d in data.get("candidates", ()):
        c = ItemBitCandidate.from_dict(d)
        out[c.key] = c
    return out
