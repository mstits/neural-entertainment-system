"""The eight receipted defects that disarmed the gate-opener K0, each
pinned so it cannot come back.

Every test here is a MUTATION CHECK: it fails if the repair is reverted,
not merely if the code stops importing. The defects and their receipts:

  D-PRIOR  runs/gate_opener_k0_2026-08-11/k0_verdict.json — the novelty
           histogram and the K4 screen were fed only by the live search's
           in-band sampling, and the sweep roots in the band tail the
           search does not re-enter. farm_samples ended at 0, novelty
           defaulted to 1.0 on every row, every significant row was
           refused `farm_unmeasured`, and the admissible list was empty
           BY CONSTRUCTION.
  D-TIE    k0_revision_log.json — 32 rows at one identical score, ordered
           by ASCENDING ADDRESS at the truncation boundary.
  D-TRUNC  ranked[:32] cannot distinguish "ranked 137th of 2,135" from
           "never ranked".
  D-DUP    admission sliced five ROWS: ['0x0033','0x0303','0x0303',
           '0x0304','0x0304'] is three addresses in five slots.
  D-ARM    four band-growth checkpoints at a hardcoded 60 s cadence = a
           240 s arming floor on runs granted fifteen minutes.
  NOOP-CONTROL ASYMMETRY  runs/cv_t1_roots_2026-08-11.json finding 3 —
           none of the 16 verified Castlevania band roots survives the
           153-step all-NOOP control (deaths at step 31-115) while 3 of 4
           sham roots do.
  D-IDENT  a lifetime-CONSTANCY answer key cannot be found by an
           onset/persist differ; recorded in the module docstring.
  D-ROOTSEL runs/gate_opener_k0v2_2026-08-11/k0v2_revision_log.json —
           `(times_swept, key[-1])` is ASCENDING GX on a fresh sweep, so
           the selector drew the most-rearward 32 cells of a 15,852-cell
           band: 0.0% of them held an occupied answer slot against the
           band's own 61.5%, and 19 of 32 died identically under the
           all-NOOP control. Found by K0-v2, which halted NOT-READY with
           its blind keys unspent.

The campaign is DISARMED (campaign doc §13) and these repairs do not
re-arm it: a re-forged instrument faces a NEW K0 registration.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.training.interaction_basis as ib                    # noqa: E402
from scripts.go_explore_solve import (                         # noqa: E402
    Solver, band_growth_stalled, gate_arm_floor_secs, gate_counters,
    gate_run_header,
)

SOLVER_SRC = Path(
    Path(__file__).resolve().parents[1] / "scripts" / "go_explore_solve.py"
).read_text()


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

def _obs(root, slot, family, moves, *, control=None, alive=None,
         plen=None, phase=8, base=None):
    """One capture triple, optionally stating the window it measured and
    whether its liveness survived that window."""
    r0 = (np.zeros(ib.RAM_SIZE, dtype=np.uint8) if base is None
          else np.array(base, np.uint8))
    r1, r2 = r0.copy(), r0.copy()
    for addr, (v1, v2) in moves.items():
        r1[addr], r2[addr] = v1, v2
    ob = {"root": root, "slot": slot, "family": family,
          "ram0": r0, "ram1": r1, "ram2": r2, "phase": phase}
    if control is not None:
        ob["control"] = control
    if alive is not None:
        ob["alive"] = alive
    if plen is not None:
        ob["plen"] = plen
    return ob


class _Cell:
    def __init__(self, key, blob):
        self.key = key
        self.state = blob


class _Pool:
    """A pool whose RAM is fully declarative: a per-worker script of
    (address -> value) edits keyed by step, so a test can state exactly
    which frames move and which die."""

    def __init__(self, workers, script=None, lives_addr=0x07F0):
        self.rams = [bytearray(ib.RAM_SIZE) for _ in range(workers)]
        self.t = [0] * workers
        self.held = [0] * workers
        self.script = script or {}
        self.lives_addr = lives_addr
        for r in self.rams:
            r[lives_addr] = 3

    def save_worker_state(self, wid):
        return bytes(self.rams[wid])

    def load_worker_state(self, wid, blob):
        self.rams[wid] = bytearray(blob)
        self.t[wid] = 0
        self.held[wid] = 0

    def step_all(self, acts):
        out = []
        for i, r in enumerate(self.rams):
            a = int(acts[i])
            self.t[i] += 1
            self.held[i] = self.held[i] + 1 if a else 0
            for addr, val in self.script.get(("step", self.t[i]), {}).items():
                r[addr] = val
            if self.held[i] >= 8:
                r[0x40] = 8
            out.append((None, None, np.frombuffer(bytes(r), dtype=np.uint8)))
        return out


_GATE_METHODS = ("_gate_armed", "_gate_observe", "_gate_novelty",
                 "_gate_farmability", "_gate_farm_sources", "_gate_roots",
                 "_gate_maybe_sweep", "_gate_hist_alloc", "_gate_checkpoint",
                 "_gate_baseline_sample", "_gate_baseline_fold",
                 "_gate_root_signature", "_gate_liveness", "_gate_sweep",
                 "_gate_wave", "_gate_pass_b", "_gate_admit",
                 "_gate_queue_injections", "shadow_yield")


def _solver(tmp_path, *, workers=2, lives=True, script=None, **over):
    """A duck-typed Solver carrying only the gate-opener surface."""
    f = SimpleNamespace()
    d = dict(
        gate_mode="enumerate", gate_pin_secs=0.0, gate_target_typed=True,
        gate_band=5, gate_sweep_frac=0.10, gate_sweep_roots=6,
        gate_sweep_repeats=1, gate_sham_roots=0, gate_arm_cadence=60.0,
        contact_bits=3, gate_weight=1.0, gate_axes=[], gate_axes_sha=None,
        bitmasks=[0, 1], steps_done=0, _pin_time=0.0, _sel_topgx=11,
        _gate_rng=np.random.default_rng(0), _gate_basis=[],
        _gate_phases=(8,), _gate_swept={}, _gate_admitted=[],
        _gate_shadow={}, _gate_positions=set(), _gate_band_hist=[],
        _gate_marks_kind="pattern", _gate_obs_n=0, _gate_fdr_m=None,
        _boundary_hist=None, _boundary_hist_total=0, _boundary_rows=None,
        _gate_boundary_hit=False, _gate_change=None, _gate_change_n=0,
        _gate_prev_vals=None, _gate_change_sweep=None,
        _gate_change_sweep_n=0, _gate_baseline_frames=0,
        _gate_baseline_len={}, _gate_baseline_prev={}, _gate_root_sig={},
        _gate_rank_stats={}, _gate_last_ckpt=0.0, _gate_band_last=0,
        _gate_inject=[], _gate_inject_i=0, _gate_next_sweep=0,
        _gate_last_pin=0.0, _gate_disarmed=False, _gate_armed_secs=0.0,
        _gate_armed_since=None, _gate_axes_live=None,
        _gate_counters=gate_counters(),
    )
    d.update(over)
    for k, v in d.items():
        setattr(f, k, v)
    f.args = SimpleNamespace(workers=workers, seed=0)
    f.out = Path(tmp_path)
    f.pool = _Pool(workers, script=script)
    game = {"progress": lambda ram: 100, "y": lambda ram: 40}
    if lives:
        game["lives"] = lambda ram: int(ram[f.pool.lives_addr])
    f.game = SimpleNamespace(**game)
    for name in _GATE_METHODS:
        setattr(f, name, MethodType(getattr(Solver, name), f))
    f._gate_rank = Solver._gate_rank
    return f


def _two_action_basis():
    """[NOOP hold 4, NOOP hold 108, NOOP tap, act hold 4, act hold 108,
    act tap] — three controls paired by length with three candidates."""
    full = ib.interaction_basis([[], ["right"]])
    return [full[0], full[3], full[8], full[4], full[7], full[10]]


def _sweepable(tmp_path, **over):
    f = _solver(tmp_path, **over)
    f._gate_basis = _two_action_basis()
    f.bitmasks = [0, 1]
    cells = []
    for gx in range(12):
        # DISTINCT BLOBS. A root's identity for the undirected prior is
        # its state blob, not its key (two keys holding one blob restore
        # one machine and replay one trajectory), so a fixture whose
        # cells share a blob is ONE root however many keys it has. The
        # marker sits high, away from every address the fixture reads.
        blob = bytearray(_Pool(1).rams[0])
        blob[0x7FF] = gx
        cells.append(_Cell((0, 0, 0, (), 0, (), 0, 0, 0, 1, gx),
                           bytes(blob)))
    f.archive = SimpleNamespace(cells={c.key: c for c in cells})
    return f, cells


# ---------------------------------------------------------------------
# D-PRIOR — the killer
# ---------------------------------------------------------------------

def test_the_sweep_feeds_the_prior_the_live_search_cannot_reach(tmp_path):
    # THE DEFECT, at its own scale of causation. `_gate_observe` samples
    # the top --gate-band, and the sweep is ROOTED in that band — a tail
    # a pinned search essentially never re-enters. On the graded run the
    # histogram was therefore never allocated at all. Everything
    # downstream followed mechanically: novelty 1.0 on every row (the
    # `_boundary_hist is None` default), farmability None, every
    # significant row refused `farm_unmeasured`, admissible list empty.
    f, cells = _sweepable(tmp_path)
    assert f._boundary_hist is None and f._gate_change_n == 0
    assert f._gate_novelty(0x40, 8) == 1.0        # the default, pre-sweep
    assert f._gate_farmability(0x40) is None

    # EIGHT roots, because the screen's refusal floor is charged in
    # DISTINCT steps now: every program at a root replays one all-NOOP
    # trajectory, so a root is worth 152 pairs and six of them cannot
    # resolve one event per thousand (repair R5).
    f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}],
                  [(c, False) for c in cells[4:]])

    # The sweep observed into the same statistics, so both are now live.
    assert f._boundary_hist is not None
    assert f._gate_baseline_frames > 0
    assert f._gate_change_sweep_n > 0
    assert f._gate_change_n == 0, \
        "the LIVE sampler is still silent — the sweep is what fixed this"
    # ...and the prior now discriminates instead of returning a constant.
    seen = f._gate_novelty(0x00, 0)                # a value the band holds
    unseen = f._gate_novelty(0x00, 0xFE)           # one it never showed
    assert unseen > seen, "the prior is not reading its own histogram"
    assert f._gate_farmability(0x40) is not None


def test_the_admissible_list_is_not_empty_by_construction(tmp_path):
    # The K0 signature, as a regression: a sweep that finds a real,
    # reproducible, unfarmable latch must be able to ADMIT it. Under the
    # defect this list was empty whatever the sweep found, so no
    # parameter setting could have produced a passing grade.
    f, cells = _sweepable(tmp_path)
    ranked = f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}],
                           [(c, False) for c in cells[4:]])
    assert ranked, "the sweep found nothing at all"
    assert not any(r["refused"] == "farm_unmeasured" for r in ranked), \
        "every row refused farm_unmeasured is the D-PRIOR signature"
    assert [r["addr"] for r in ib.admissible_rows(ranked)] == [0x40]
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["admitted"] == [0x40]
    assert rec["boundary_frames_sweep"] > 0


def test_only_undirected_frames_enter_the_prior(tmp_path):
    # THE SELF-REFERENCE GUARD. If a pattern's own post-onset RAM went
    # into the histogram, the value it just created would score as
    # "already seen at this boundary" and novelty — the term the whole
    # rank is multiplied by — would shrink toward zero exactly where it
    # is meant to be largest. 0x40 reads 8 only after a mask has been
    # held eight frames, i.e. never inside an undirected window.
    f, cells = _sweepable(tmp_path)
    f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}],
                  [(c, False) for c in cells[6:]])
    assert int(f._boundary_hist[0x40, 0]) > 0          # the quiet value
    assert int(f._boundary_hist[0x40, 8]) == 0, \
        "the instrument counted its own discovery into its own prior"


def test_pass_b_never_feeds_the_prior_or_the_screen(tmp_path):
    # Pass B re-runs the slots that SURVIVED pass A, from three roots
    # chosen after the fact. Letting a data-selected sub-population set
    # the horizon the prior is measured over is the same error the BH
    # denominator is charged the full grid to avoid: two sweeps of one
    # wall would then screen against different priors.
    f, cells = _sweepable(tmp_path)
    roots = [(c, False) for c in cells[6:]]
    seen = {}

    real = f._gate_wave

    def spy(jobs, tail=ib.TAIL_PASS_A, length=ib.PROGRAM_LEN_PASS_A,
            baseline=True):
        seen.setdefault(length, set()).add(baseline)
        return real(jobs, tail=tail, length=length, baseline=baseline)

    f._gate_wave = spy
    f._gate_sweep([{"key": ("a",)}, {"key": ("b",)}], roots)
    assert seen[ib.PROGRAM_LEN_PASS_A] == {True}
    assert seen[ib.PROGRAM_LEN_PASS_B] == {False}, \
        "pass B fed the prior it was selected by"


def test_a_pooled_farm_rate_is_measured_in_steps_and_never_fake_zero():
    # The estimator the two samplers meet in. Pooling has to be in STEP
    # units, because the live sampler counts stride-32 gaps and the
    # sweep counts consecutive frames — and the refusal floor is a
    # resolution floor, so it is a floor on STEPS.
    assert ib.FARM_MIN_SAMPLES * ib.BAND_SAMPLE_STRIDE >= ib.FARM_MIN_STEPS
    assert (ib.FARM_MIN_SAMPLES - 1) * ib.BAND_SAMPLE_STRIDE \
        < ib.FARM_MIN_STEPS
    # Below the floor there is no number, at any stride, from any mix.
    assert ib.farm_rate_pooled([(0, 10, 32)]) is None
    assert ib.farm_rate_pooled([(0, 999, 1)]) is None
    assert ib.farm_rate_pooled([]) is None
    # A source with no samples contributes nothing rather than a zero.
    assert ib.farm_rate_pooled(
        [(0, 0, 32), (2, 2000, 1)]) == pytest.approx(1.0)
    # Two strides, one rate: 1 change in 1,024 live steps + 1 in 1,024
    # swept steps is 2 per 2,048 steps.
    assert ib.farm_rate_pooled([(1, 32, 32), (1, 1024, 1)]) == pytest.approx(
        1000.0 * 2 / 2048)
    # The single-source contract the live path already had is unchanged.
    assert ib.farm_rate(0, ib.FARM_MIN_SAMPLES - 1) is None
    assert ib.farm_rate(0, ib.FARM_MIN_SAMPLES) == 0.0
    assert ib.farm_rate(4, 125) == pytest.approx(ib.MAX_FARM_EVENTS_PER_1K)


def test_the_undirected_prefix_is_the_whole_control_and_only_the_settle():
    # What "undirected" means, exactly: an all-NOOP control is undirected
    # for its whole length (that is why it can carry the screen at all),
    # and every other program only until its first press.
    basis = _two_action_basis()
    ctl = ib.build_program(basis[1], 8)            # all-NOOP, 108 long
    pat = ib.build_program(basis[4], 8)            # pressed, 108 long
    assert ib.noop_prefix_len(ctl) == len(ctl) == ib.PROGRAM_LEN_PASS_A
    assert ib.noop_prefix_len(pat) == 8
    assert ib.noop_prefix_len(ib.build_program(basis[4], 21)) == 21
    assert ib.noop_prefix_len(()) == 0


# ---------------------------------------------------------------------
# D-TIE — the score plateau
# ---------------------------------------------------------------------

def _plateau(n=12, families=("hold",)):
    """Rows that tie on score AND on every principled tie-break, so the
    ordering that comes out is the seeded one and nothing else. Addresses
    are deliberately spread across the map."""
    rows = []
    for fam in families:
        for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
            rows.append(_obs(root, 0, "control", {}))
            rows.append(_obs(root, 1, fam,
                             {0x10 * (i + 1): (8, 8) for i in range(n)}))
    return rows


def test_a_synthetic_score_plateau_does_not_order_by_address():
    # THE DEFECT: rank_candidates' final key was (-score, addr, family),
    # so at a plateau the operative order was ASCENDING ADDRESS and an
    # answer above the tied block was unreachable at rank<=5 at ANY
    # parameter setting. Every row below ties on score, on effect size
    # and on family diversity by construction.
    ranked = ib.rank_candidates(_plateau(), tie_seed=7)
    scores = {round(r["score"], 12) for r in ranked}
    effects = {round(r["effect_size"], 12) for r in ranked}
    assert len(scores) == len(effects) == 1, "the fixture is not a plateau"
    addrs = [r["addr"] for r in ranked]
    assert sorted(addrs) == addrs[:0] + sorted(addrs)      # same multiset
    assert addrs != sorted(addrs), \
        "the plateau is still ordered by ascending address"
    assert addrs != sorted(addrs, reverse=True)


def test_the_plateau_order_is_seeded_reproducible_and_permutation_proof():
    # Arbitrary, but a DECLARED and logged arbitrary: same seed same
    # order (so a receipt reproduces), different seed different order (so
    # two seeds are a real replication of the ranking rather than the
    # same coin twice), and independent of input order (so the ranking
    # cannot be steered by the order observations arrive in).
    rows = _plateau()
    a = [r["addr"] for r in ib.rank_candidates(rows, tie_seed=1)]
    b = [r["addr"] for r in ib.rank_candidates(rows, tie_seed=1)]
    c = [r["addr"] for r in ib.rank_candidates(rows, tie_seed=2)]
    assert a == b and a != c
    shuffled = list(reversed(rows))
    assert [r["addr"] for r in ib.rank_candidates(shuffled, tie_seed=1)] == a
    # ...and the seed is on the row, so no rank can be cited without it.
    assert {r["tie_seed"] for r in ib.rank_candidates(rows, tie_seed=1)} == {1}


def test_effect_size_breaks_the_plateau_before_the_coin_does():
    # The principled tie-break comes FIRST: a p-value collapses into
    # plateaus under discreteness (k == n at every row is one value of
    # p), but how far the byte actually moved does not. A high address
    # that moved 200 outranks a low one that moved 1, which is the exact
    # inversion the address-ordered key could never produce.
    rows = []
    for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
        rows.append(_obs(root, 0, "control", {}))
        rows.append(_obs(root, 1, "hold", {0x08: (1, 1), 0x7F0: (200, 200)}))
    ranked = ib.rank_candidates(rows, tie_seed=0)
    assert [r["addr"] for r in ranked] == [0x7F0, 0x08]
    assert ranked[0]["effect_size"] > ranked[1]["effect_size"]
    # Both orderings of the same two rows are reachable by effect size,
    # so the key is reading the measurement and not the address.
    flip = []
    for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
        flip.append(_obs(root, 0, "control", {}))
        flip.append(_obs(root, 1, "hold", {0x08: (200, 200), 0x7F0: (1, 1)}))
    assert [r["addr"] for r in ib.rank_candidates(flip, tie_seed=0)] == \
        [0x08, 0x7F0]


def test_no_ranking_path_sorts_by_address_any_more():
    # Structural, because the defect's second home was the pass-B
    # re-sort: repairing rank_candidates alone leaves the confirmation
    # pass free to put the address back at the top of the key.
    assert '(-r["score"], r["addr"], r["family"])' not in SOLVER_SRC
    assert '(-s["score"], s["addr"], s["family"])' not in SOLVER_SRC
    assert SOLVER_SRC.count("ib.rank_sort_key") >= 1
    src = Path(ib.__file__).read_text()
    assert src.count("sort(key=rank_sort_key)") == 1
    # The address survives ONLY behind the 8-byte hash, as the
    # determinism backstop. Key shape: score, BH verdict, K4 refusal,
    # effect size, seeded coin — see rank_sort_key.
    key = ib.rank_sort_key({"score": 1.0, "effect_size": 0.5,
                            "addr_families": 2, "tie_key": 9, "addr": 4,
                            "family": "hold", "significant": True})
    assert key == (-1.0, 0, 0, -0.5, 9, 4, "hold")


# ---------------------------------------------------------------------
# D-TRUNC / D-DUP — the receipt
# ---------------------------------------------------------------------

def _ranked_rows(n, admissible_from=0):
    out = []
    for i in range(n):
        out.append({"addr": 0x100 + i, "family": "hold", "roots": 6,
                    "family_roots": 6, "class": "latched", "values": [1],
                    "slots": [1], "p": 1e-9, "significant": True,
                    "fdr_m": 8192, "novelty": 1.0, "farmability": 0.0,
                    "effect_size": 0.5, "addr_families": 1, "tie_key": i,
                    "tie_seed": 0, "score": 1.0,
                    "refused": None if i >= admissible_from else "farmable"})
    return out


def test_the_receipt_distinguishes_missing_from_ranked_low(tmp_path):
    # D-TRUNC. ranked[:32] cannot answer the only question a K0 grade
    # asks. The graded sweep produced 2,135 rows and its receipt could
    # say just "not in the top 32" — which is compatible with "ranked
    # 33rd" and with "never ranked", opposite verdicts about the
    # instrument.
    f = _solver(tmp_path)
    f._gate_counters["sweeps"] = 1
    ranked = _ranked_rows(60)
    f._gate_admit(ranked, [])
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())

    assert len(rec["ranked"]) == 32 and rec["ranked_head"] == 32
    assert rec["ranked_total"] == 60
    # The rank of EVERY address that ranked, including the 33rd.
    assert rec["ranked_addr_rank"]["0x0120"] == 33
    assert len(rec["ranked_addr_rank"]) == 60
    assert "0x0004" not in rec["ranked_addr_rank"]      # genuinely absent
    # ...and the whole table, compressed, with its sha.
    blob = (tmp_path / "gate" / rec["ranked_full"]).read_bytes()
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == rec["ranked_full_sha256"]
    assert len(json.loads(gzip.decompress(blob))) == 60
    # The sha is a function of the RANKING, not of the wall clock, or it
    # cannot be used to check that two runs ranked the same.
    g = _solver(tmp_path / "again")
    g._gate_counters["sweeps"] = 1
    g._gate_admit(_ranked_rows(60), [])
    again = json.loads(
        (tmp_path / "again" / "gate" / "candidates_001.json").read_text())
    assert again["ranked_full_sha256"] == rec["ranked_full_sha256"]


def test_the_receipt_ranks_the_admissible_list_the_grade_is_defined_on(
        tmp_path):
    # `rank <= 5` is a position in the ADMISSIBLE list, so that list has
    # to be receipted with its own ranks and its own total — the ranked
    # table alone cannot answer it, because refusals renumber it.
    f = _solver(tmp_path)
    f._gate_counters["sweeps"] = 1
    f._gate_admit(_ranked_rows(20, admissible_from=8), [])
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["admissible_total"] == 12
    assert rec["admissible_addr_rank"]["0x0108"] == 1     # 9th ranked row
    assert rec["ranked_addr_rank"]["0x0108"] == 9
    assert "0x0100" not in rec["admissible_addr_rank"]    # refused farmable


def test_admission_is_five_addresses_not_five_rows():
    # D-DUP, in its receipted shape. An (addr, family) pair is a row, so
    # slicing rows spends admission slots on one byte seen through
    # several families: the graded Bubble Bobble sweep admitted
    # ['0x0033','0x0303','0x0303','0x0304','0x0304'] — five slots, three
    # addresses. The sidecar promotes ADDRESSES and the grade is defined
    # on addresses, so the cut belongs there.
    rows = []
    for i, (addr, fam) in enumerate([(0x33, "hold"), (0x303, "hold"),
                                     (0x303, "tap"), (0x304, "hold"),
                                     (0x304, "tap"), (0x305, "hold"),
                                     (0x306, "hold")]):
        rows.append({"addr": addr, "family": fam, "significant": True,
                     "refused": None, "score": 1.0 - i * 1e-9})
    got = [r["addr"] for r in ib.admitted_candidates(rows)]
    assert got == [0x33, 0x303, 0x304, 0x305, 0x306]
    assert len(set(got)) == 5
    # The row kept for an address is its best-ranked one, so the family
    # that carried it is still receipted.
    assert ib.admitted_candidates(rows)[1]["family"] == "hold"
    # admissible_rows keeps every row (ranks are defined over rows);
    # only the CUT is by address.
    assert len(ib.admissible_rows(rows)) == 7


# ---------------------------------------------------------------------
# D-ARM — the arming floor
# ---------------------------------------------------------------------

def test_the_arming_floor_is_derived_from_the_cadence_and_stated():
    # The floor was invisible while the cadence was a hardcoded 60 s
    # inside the progress line: four checkpoints are structurally
    # required, so nothing could arm inside four minutes on runs granted
    # fifteen. Measured arming times in the K0 receipts: 240-300 s.
    assert gate_arm_floor_secs(60.0) == 240.0
    assert gate_arm_floor_secs(5.0) == 20.0
    assert gate_arm_floor_secs(-1.0) == 0.0
    # ...and it is genuinely four, not three: the conjunct needs need+1.
    assert band_growth_stalled([5, 5, 5]) is False
    assert band_growth_stalled([5, 5, 5, 5]) is True


def test_the_cadence_is_a_flag_whose_default_does_not_move_the_arm():
    assert '"--gate-arm-cadence-secs", type=float, default=60.0' in SOLVER_SRC
    assert 'getattr(args, "gate_arm_cadence_secs", 60.0)' in SOLVER_SRC
    hdr = gate_run_header(
        SimpleNamespace(seed=3, workers=8, burst=64, gate_opener="enumerate",
                        gate_pin_secs=600.0, gate_target_typed=True,
                        gate_band=24, gate_arm_cadence_secs=15.0),
        commit="abc", hw_flags=[], root_sha="s", sidecar_sha=None, axes=[],
        active_predicate="p")
    assert hdr["gate_arm_cadence_secs"] == 15.0
    assert hdr["gate_arm_floor_secs"] == 60.0
    assert hdr["rank_tie_seed"] == 3
    # A default header records the receipted floor unchanged.
    plain = gate_run_header(
        SimpleNamespace(seed=0, workers=8, burst=64, gate_opener="enumerate",
                        gate_pin_secs=600.0, gate_target_typed=True,
                        gate_band=24),
        commit="abc", hw_flags=[], root_sha="s", sidecar_sha=None, axes=[],
        active_predicate="p")
    assert plain["gate_arm_floor_secs"] == 240.0


def test_a_short_cadence_arms_a_bounded_run_inside_its_budget(tmp_path):
    # Wired end to end on the checkpoint the conjunct actually reads: at
    # a 5 s cadence four checkpoints land at t=20 s, so a K0-class run
    # can arm inside a budget the 60 s floor would consume whole.
    f, cells = _sweepable(tmp_path, gate_arm_cadence=5.0, _gate_band_hist=[])
    for i in range(1, 5):
        assert f._gate_checkpoint(i * 5.0) == 6      # buckets 6..11
    assert len(f._gate_band_hist) == 4
    assert f._gate_armed(20.0) is True
    assert f._gate_last_ckpt == 20.0
    # Three checkpoints is still not enough, whatever the cadence.
    g, _ = _sweepable(tmp_path, gate_arm_cadence=5.0, _gate_band_hist=[])
    for i in range(1, 4):
        g._gate_checkpoint(i * 5.0)
    assert g._gate_armed(15.0) is False


def test_the_checkpoint_series_is_minted_once_not_by_the_progress_line():
    # A progress line that also appended would double-count the series
    # the growth conjunct measures over, making the arming floor a
    # function of how often the run happens to log.
    assert SOLVER_SRC.count("self._gate_band_hist.append(band)") == 1
    assert "_gate_checkpoint" in SOLVER_SRC.split("def progress_line")[0]


# ---------------------------------------------------------------------
# the NOOP-control asymmetry (root-staging finding 3)
# ---------------------------------------------------------------------

def test_a_control_that_died_inside_its_window_is_not_subtracted():
    # THE ASYMMETRY. None of the 16 verified Castlevania band roots
    # survives the 153-step all-NOOP control; 3 of 4 sham roots do. A
    # control that dies measures a death-and-respawn cascade, and
    # subtracting it deletes every address the death touched from the
    # candidate pool — so the wall arm loses candidates the sham arm
    # keeps, on exactly the axis K1 compares them on.
    # The real shape at that wall: the SHORT control survives its window
    # (deaths land at step 31-115, and a plen-4 program closes at 36)
    # while the 108-step one does not. So the root keeps a live baseline
    # and only the long window loses its exact twin.
    def rows(long_control_alive):
        out = []
        for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
            out.append(_obs(root, 0, "control", {0x11: (1, 1)},
                            control=True, alive=True, plen=4))
            out.append(_obs(root, 1, "control",
                            {0x40: (9, 9), 0x11: (1, 1)},
                            control=True, alive=long_control_alive,
                            plen=108))
            out.append(_obs(root, 2, "hold", {0x40: (8, 8), 0x11: (1, 1)},
                            control=False, alive=True, plen=108))
            out.append(_obs(root, 3, "tap", {0x42: (5, 5)},
                            control=False, alive=True, plen=108))
        return out

    # A LIVE 108-step control that moves 0x40 is a genuine free-runner
    # over that window: subtract it.
    live = {r["addr"] for r in ib.rank_candidates(rows(True))}
    assert 0x40 not in live and 0x42 in live
    # A DEAD one measured a death cascade and may not veto anything. The
    # long window falls back to the union of the root's LIVE controls,
    # which still removes the fast free-runner 0x11 — over-subtraction is
    # the direction that can cost a candidate, never invent one.
    stats: dict = {}
    dead = {r["addr"] for r in ib.rank_candidates(rows(False), stats=stats)}
    assert 0x40 in dead, "a dead control deleted a real candidate"
    assert 0x11 not in dead and 0x42 in dead
    assert stats["dropped_dead_control"] == 6
    assert stats["roots"] == 6 and stats["roots_uncontrolled"] == 0


def test_a_root_whose_controls_all_died_is_dropped_whole():
    # The end of the same spectrum, and the conservative direction. With
    # no live control anywhere at a root there is no quiescent baseline
    # to subtract, and ranking uncontrolled differentials is precisely
    # how a death cascade becomes a receipted discovery. So the root is
    # dropped, and the drop is COUNTED rather than showing up as a
    # quieter wall.
    rows = []
    for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
        rows.append(_obs(root, 0, "control", {0x40: (9, 9)},
                         control=True, alive=False, plen=108))
        rows.append(_obs(root, 1, "hold", {0x40: (8, 8)},
                         control=False, alive=True, plen=108))
    stats: dict = {}
    assert ib.rank_candidates(rows, stats=stats) == []
    assert stats["roots_uncontrolled"] == 6
    assert stats["roots"] == 0
    assert stats["dropped_dead_control"] == 6


def test_a_pattern_that_died_inside_its_window_is_never_ranked():
    # The mirror, and the one the Contra tuning grade actually hit: the
    # tied block at the top of that ranking was zero-page state going to
    # 0 under 108-step holds — the signature of the run ending — which
    # the paired NOOP control did not subtract because the NOOP-108
    # control did not die.
    rows = []
    for root in ("r0", "r1", "r2", "r3", "r4", "r5"):
        rows.append(_obs(root, 0, "control", {}, control=True, alive=True,
                         plen=108))
        rows.append(_obs(root, 1, "hold", {0x2C: (0, 0), 0x40: (8, 8)},
                         control=False, alive=False, plen=108))
        rows.append(_obs(root, 2, "tap", {0x42: (5, 5)},
                         control=False, alive=True, plen=108))
    stats: dict = {}
    got = {r["addr"] for r in ib.rank_candidates(rows, stats=stats)}
    assert 0x2C not in got and 0x40 not in got, \
        "a death cascade was receipted as a discovery"
    assert 0x42 in got
    assert stats["dropped_dead_pattern"] == 6
    assert stats["observations"] == 6


def test_a_root_that_ran_no_control_at_all_is_left_exactly_as_it_was():
    # DEFAULT-INERT. "No control was run" (a fixture, a pass with no
    # length twin) and "the control died" are different facts and are
    # not merged: only the second drops the root. Absent the key
    # entirely, every observation is live and the pipeline is the one
    # that predates the repair.
    rows = []
    for root in ("r0", "r1", "r2"):
        rows.append(_obs(root, 1, "hold", {0x40: (8, 8)}))
        rows.append(_obs(root, 2, "tap", {0x42: (5, 5)}))
    stats: dict = {}
    got = {r["addr"] for r in ib.rank_candidates(rows, stats=stats)}
    assert got == {0x40, 0x42}
    assert stats["roots"] == 3 and stats["roots_uncontrolled"] == 0
    assert stats["dropped_dead_control"] == 0


def test_duration_matching_is_the_other_half_and_is_still_in_place():
    # The repair is death-awareness BECAUSE per-pattern truncation is
    # already the pairing rule: a control speaks only for the (phase,
    # plen) window it measured, so it need only survive as long as the
    # pattern it is compared against. Both halves are named in the
    # header, or an A/B pair cannot be checked after the fact.
    hdr = gate_run_header(
        SimpleNamespace(seed=0, workers=8, burst=64, gate_opener="enumerate",
                        gate_pin_secs=600.0, gate_target_typed=True,
                        gate_band=24),
        commit="abc", hw_flags=[], root_sha="s", sidecar_sha=None, axes=[],
        active_predicate="p")
    assert hdr["control_pairing"]["duration"] == \
        "matched on (settle phase, pattern length)"
    assert "excluded as control and as candidate" in \
        hdr["control_pairing"]["liveness"]
    # ...and the shortest control cannot veto a window it never entered.
    rows = []
    for root in ("r0", "r1", "r2", "r3"):
        rows.append(_obs(root, 0, "control", {}, control=True, plen=4))
        rows.append(_obs(root, 1, "control", {0x11: (1, 1)}, control=True,
                         plen=108))
        rows.append(_obs(root, 2, "hold", {0x11: (1, 1)}, control=False,
                         plen=4))
    assert 0x11 in {r["addr"] for r in ib.rank_candidates(rows)}


def test_the_wave_reads_liveness_from_the_profiles_own_telemetry(tmp_path):
    # No new address: the death-aware conjunct reads the lives byte the
    # search already depends on, through the same adapter. A death
    # anywhere inside a program's differential window marks it, and the
    # ranker does the rest.
    f, cells = _sweepable(tmp_path, script={("step", 30): {0x07F0: 2}})
    basis = f._gate_basis
    jobs = [(cells[6], False, 1, basis[1], 8, 0),      # NOOP 108: dies
            (cells[7], False, 4, basis[4], 8, 0)]      # held 108: dies too
    out = f._gate_wave(jobs)
    assert [o["alive"] for o in out] == [False, False]
    assert [o["lives"] for o in out] == [[3, 2, 2], [3, 2, 2]]
    # ...and the baseline window closed at the death rather than
    # measuring the cascade as an undirected change rate.
    assert f._gate_counters["baseline_truncated"] >= 1
    assert f._gate_baseline_frames <= 30 + 8

    # A profile that declares no liveness reading gets the pre-repair
    # behaviour: every observation live, nothing dropped.
    g, gcells = _sweepable(tmp_path, lives=False,
                           script={("step", 30): {0x07F0: 2}})
    out = g._gate_wave([(gcells[6], False, 1, g._gate_basis[1], 8, 0)])
    assert out[0]["alive"] is True and out[0]["lives"] == [None, None, None]
    assert g._gate_counters["baseline_truncated"] == 0


def test_the_sweep_receipts_what_the_differential_declined_to_read(tmp_path):
    # A shrinking candidate pool has to be a REPORTED fact. Without the
    # counters, "the wall produced nothing" and "the wall's controls all
    # died so nothing could be read" are the same empty receipt — and
    # they are opposite verdicts about the instrument.
    f = _solver(tmp_path)
    f._gate_counters["sweeps"] = 1
    f._gate_rank_stats = {"roots": 4, "roots_uncontrolled": 12,
                          "dropped_dead_control": 30,
                          "dropped_dead_pattern": 7, "observations": 90,
                          "controls": 36}
    f._gate_admit([], [])
    rec = json.loads((tmp_path / "gate" / "candidates_001.json").read_text())
    assert rec["differential"]["roots_uncontrolled"] == 12
    assert rec["differential"]["dropped_dead_pattern"] == 7
    assert rec["ranked_total"] == 0 and rec["admissible_total"] == 0


# ---------------------------------------------------------------------
# D-IDENT — recorded, not fixed in code
# ---------------------------------------------------------------------

def test_the_module_records_that_constancy_bytes_are_bad_answer_targets():
    # Half of the Contra tuning key was an array its own receipt defines
    # by per-slot-lifetime CONSTANCY. A differential onset/persist ranker
    # keys on bytes that MOVE, so that array is close to unreachable by
    # construction — grading against it measures the key, not the
    # ranker. The note belongs where the next registration will read it.
    doc = ib.__doc__ or ""
    assert "D-IDENT" in doc
    assert "CONSTANCY" in doc
    assert "0x058F-92" in doc and "0x04BF-C2" in doc
    assert "onset" in doc


# ---------------------------------------------------------------------
# D-ROOTSEL — the selector's own corner
# ---------------------------------------------------------------------

def _band_solver(tmp_path, *, columns=100, rows=20, band=24, roots=32,
                 **over):
    """A wide FRESH band: `columns` gx buckets x `rows` y bands, every
    cell unswept and carrying a blob. Shaped after the receipted Contra
    band (15,852 cells across 25 gx buckets) so the tie-break is the only
    thing deciding which cells get measured."""
    f = _solver(tmp_path, gate_band=band, gate_sweep_roots=roots,
                gate_sham_roots=0, **over)
    cells = [_Cell((0, 0, 0, (), 0, (), 0, 0, 0, yb, gx),
                   bytes([gx & 0xFF, yb & 0xFF]))
             for gx in range(columns) for yb in range(rows)]
    f.archive = SimpleNamespace(cells={c.key: c for c in cells})
    return f


def _root_gx(f):
    return [int(c.key[-1]) for c, sham in f._gate_roots() if not sham]


def test_a_fresh_band_is_sampled_across_its_whole_width_not_its_rear_corner():
    # THE DEFECT: the key was `(times_swept, key[-1])` and on a fresh
    # sweep every band cell has times_swept 0 — so the operative term was
    # ASCENDING GX and the 32 roots were the 32 most-rearward cells of a
    # 15,852-cell band, a correlated near-duplicate cluster. Receipt:
    # runs/gate_opener_k0v2_2026-08-11/k0v2_revision_log.json D-ROOTSEL
    # (200 random band cells held an occupied Contra hp slot 123 times;
    # the 32 cells the selector drew held one zero times, at bands 24, 96
    # and 192 alike).
    #
    # Held over EIGHT seeds, because a spread that only appears at one
    # seed is the old corner with a different address on it.
    for seed in range(8):
        f = _band_solver(Path("."), _gate_rng=np.random.default_rng(seed))
        gxs = _root_gx(f)
        assert len(gxs) == 32
        lo, hi = 75, 99                       # top 99, band 24 -> floor 75
        assert min(gxs) >= lo and max(gxs) <= hi, "roots left the band"
        # Spread: the 25 band columns are sampled broadly, and the draw
        # reaches both ends. Under the defect all 32 sat in column 75.
        assert len(set(gxs)) >= 12, \
            f"seed {seed}: only {len(set(gxs))} of 25 band columns drawn"
        assert max(gxs) - min(gxs) >= 15
        # ...and the rearmost quarter of the band gets its share, not all
        # of it. Expectation is ~8 of 32; the defect put all 32 there.
        rear = sum(1 for g in gxs if g < lo + 6)
        assert rear <= 16, f"seed {seed}: {rear}/32 roots in the rear corner"
        assert sum(1 for g in gxs if g >= hi - 5) >= 1, \
            f"seed {seed}: the forward end of the band was never drawn"


def test_the_root_draw_is_seeded_reproducible_and_privileges_no_address():
    # Arbitrary, but a DECLARED arbitrary, exactly as D-TIE's coin is:
    # same seed same roots (so a receipt reproduces from its header),
    # different seed different roots (so two seeds are a real replication
    # of the measurement rather than the same corner twice).
    def draw(seed):
        f = _band_solver(Path("."), _gate_rng=np.random.default_rng(seed))
        return [c.key for c, _s in f._gate_roots()]

    a, b, c = draw(3), draw(3), draw(4)
    assert a == b, "the root draw is not reproducible from its seed"
    assert a != c, "the root draw ignores its seed"
    # No key term is privileged: the draw is not the gx order, nor the
    # gx order reversed, nor the archive's own insertion order.
    f = _band_solver(Path("."), _gate_rng=np.random.default_rng(3))
    keys = a
    assert keys != sorted(keys) and keys != sorted(keys, reverse=True)
    band = [k for k in f.archive.cells if int(k[-1]) >= 75]
    assert keys != band[:32] and keys != band[-32:][::-1]
    # The tie is broken by the SWEEP's stream, so the search stream is
    # untouched however wide the band is.
    g = _band_solver(Path("."), _gate_rng=np.random.default_rng(3),
                     rng=np.random.default_rng(99))
    before = g.rng.bit_generator.state
    g._gate_roots()
    assert g.rng.bit_generator.state == before


def test_the_least_swept_first_contract_survives_the_random_tie_break():
    # The coin breaks TIES; it does not replace the ordering. A cell the
    # sweep has already visited must not be re-drawn while an unvisited
    # one is available, or the sweep re-measures one root all run.
    f = _band_solver(Path("."), roots=40, _gate_rng=np.random.default_rng(0))
    band = [k for k in f.archive.cells if int(k[-1]) >= 75]
    assert len(band) == 500
    # All but 30 band cells already swept once: the draw must take those
    # 30 first, then fill from the swept remainder.
    fresh = set(band[::17][:30])
    f._gate_swept = {k: 1 for k in band if k not in fresh}
    keys = [c.key for c, _s in f._gate_roots()]
    assert set(keys[:30]) == fresh, "a swept cell outranked an unswept one"
    assert len(set(keys)) == 40
    # Twice-swept cells come last for the same reason.
    once = band[1]                            # band[0] is in `fresh`
    f._gate_swept = {k: (0 if k in fresh else 2) for k in band}
    f._gate_swept[once] = 1
    f._gate_rng = np.random.default_rng(0)
    keys = [c.key for c, _s in f._gate_roots()]
    assert set(keys[:30]) == fresh and keys[30] == once


def test_no_gx_ordered_root_key_survives_in_the_selector():
    # Structural, because the defect is one line and reverting it is one
    # line. The gx term must not reappear ahead of the coin.
    src = SOLVER_SRC.split("def _gate_roots", 1)[1].split("\n    def ", 1)[0]
    assert "self._gate_swept.get(c.key, 0), c.key[-1]" not in src
    assert "self._gate_rng.random(len(band))" in src
    assert "D-ROOTSEL" in src


# ---------------------------------------------------------------------
# nothing above re-arms anything
# ---------------------------------------------------------------------

def test_the_arm_is_still_default_off_and_the_repairs_are_inert_without_it():
    # The campaign is DISARMED (§13) and these repairs do not re-arm it.
    # Every knob still defaults to the receipted value, and a run that
    # names none of them samples exactly as before.
    assert '"--gate-opener", choices=("off", "enumerate"),\n' \
           '                    default="off"' in SOLVER_SRC
    assert '"--gate-pin-secs", type=float, default=None' in SOLVER_SRC
    assert '"--gate-sweep-roots", type=int, default=16' in SOLVER_SRC
    assert '"--gate-band", type=int, default=24' in SOLVER_SRC
    # The two new pure helpers are inert by default too: tie_seed 0 and
    # no stats dict reproduce a plain call.
    rows = _plateau(n=3)
    assert ib.rank_candidates(rows) == ib.rank_candidates(rows, tie_seed=0)
