"""Tests for the cross-fit split-sample reducer
(V31_REDO_SURGICAL_2026-08-27.md §6, §12 item 5).

"Both Theta and Delta must come out of ONE committed cross-fit reducer
with tests, not arithmetic in a document, or they are not the same
estimator." These tests exercise that reducer directly, plus the
receipt-file loader with the naming convention already in use in
runs/v27_readjudication_2026-08-27/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.score_cross_fit import (  # noqa: E402
    RomMismatchError,
    compute_delta,
    compute_theta,
    cross_fit_seed_score,
    load_receipts,
)

CANDIDATES = list(range(10, 241, 10))


def _write_receipt(d: Path, *, seed: int, it: int, es: int, clear_rate: float):
    d.mkdir(parents=True, exist_ok=True)
    name = f"mario_1_1_v31_redo_seed{seed}_it{it:03d}_es{es}.json"
    (d / name).write_text(json.dumps({
        "game": "mario", "clear_rate": clear_rate, "n_episodes": 50,
        "eval_seed": es, "checkpoint": f"iter_{it:05d}.pt",
    }))


def test_argmax_ties_break_to_later_iter():
    """Registered tie-break rule (§6): 'Ties -> later iter, both
    directions reported.'
    """
    by_iter = {
        10: {0: 0.5, 1: 0.4},
        20: {0: 0.5, 1: 0.6},  # ties with iter 10 on es0
    }
    s = cross_fit_seed_score(by_iter, seed=0, candidates=[10, 20])
    assert s.selected_iter_a == 20, "tie on es0 selection must pick the later iter"


def test_split_sample_never_scores_a_checkpoint_on_its_own_selection_data():
    """score_A is read from eval_seed 1 at the checkpoint selected using
    ONLY eval_seed 0, and vice versa — no episode is used for both
    selecting and scoring the same checkpoint.
    """
    by_iter = {
        10: {0: 0.9, 1: 0.1},   # best on es0
        20: {0: 0.2, 1: 0.9},   # best on es1
    }
    s = cross_fit_seed_score(by_iter, seed=0, candidates=[10, 20])
    assert s.selected_iter_a == 10
    assert s.score_a == 0.1     # scored on es1 AT the es0-selected checkpoint
    assert s.selected_iter_b == 20
    assert s.score_b == 0.2     # scored on es0 AT the es1-selected checkpoint
    assert s.seed_score == (0.1 + 0.2) / 2


def test_incomplete_ladder_yields_no_seed_score():
    """A seed missing an entire eval seed's ladder cannot be cross-fit
    scored — None, not a silently-biased number.
    """
    by_iter = {10: {0: 0.5}}  # no eval_seed 1 anywhere
    s = cross_fit_seed_score(by_iter, seed=0, candidates=[10, 20])
    assert s.selected_iter_b is None
    assert s.score_b is None
    assert s.seed_score is None


def test_theta_is_best_of_n_and_requires_all_seeds_armed(tmp_path):
    """N is fixed at 4 (§6): fewer than 4 complete seed scores ->
    VOID-UNDERPOWERED, no Theta issued — never a best-of-fewer against
    the best-of-4 bar.
    """
    receipts_dir = tmp_path / "receipts"
    for seed, (a, b) in enumerate([(0.9, 0.9), (0.5, 0.5), (0.6, 0.6)]):
        _write_receipt(receipts_dir, seed=seed, it=10, es=0, clear_rate=a)
        _write_receipt(receipts_dir, seed=seed, it=10, es=1, clear_rate=b)
    receipts = load_receipts(receipts_dir, seeds=[0, 1, 2, 3])
    result = compute_theta(receipts, candidates=[10], required_seeds=4)
    assert result.theta is None
    assert result.n_armed_seeds == 3
    assert "VOID-UNDERPOWERED" in result.verdict

    # Now complete the 4th seed — Theta becomes best-of-4 = max seed_score.
    _write_receipt(receipts_dir, seed=3, it=10, es=0, clear_rate=0.95)
    _write_receipt(receipts_dir, seed=3, it=10, es=1, clear_rate=0.95)
    receipts2 = load_receipts(receipts_dir, seeds=[0, 1, 2, 3])
    result2 = compute_theta(receipts2, candidates=[10], required_seeds=4)
    assert result2.theta == 0.95
    assert result2.winning_seed == 3
    assert result2.verdict.startswith("PASS")


def test_theta_bar_thresholds_verbatim():
    """PASS >= 0.80, FAIL <= 0.767, MARGINAL between — inherited
    verbatim (§6), not reopened.
    """
    def _theta_for(rate: float) -> float | None:
        receipts = {
            s: {10: {0: rate, 1: rate}} for s in range(4)
        }
        return compute_theta(receipts, candidates=[10]).theta

    assert compute_theta(
        {s: {10: {0: 0.767, 1: 0.767}} for s in range(4)}, candidates=[10],
    ).verdict.startswith("FAIL")
    assert compute_theta(
        {s: {10: {0: 0.80, 1: 0.80}} for s in range(4)}, candidates=[10],
    ).verdict.startswith("PASS")
    r = compute_theta(
        {s: {10: {0: 0.78, 1: 0.78}} for s in range(4)}, candidates=[10],
    )
    assert r.verdict.startswith("MARGINAL")


def test_theta_adj_applies_the_registered_winners_curse_budget():
    receipts = {s: {10: {0: 0.85, 1: 0.85}} for s in range(4)}
    result = compute_theta(receipts, candidates=[10], winners_curse=0.05)
    assert result.theta == 0.85
    assert result.theta_adj == pytest.approx(0.80)


def test_load_receipts_ignores_files_outside_the_naming_convention(tmp_path):
    d = tmp_path / "receipts"
    d.mkdir()
    (d / "not_a_receipt.json").write_text("{}")
    (d / "readjudication.json").write_text("{}")
    _write_receipt(d, seed=0, it=10, es=0, clear_rate=0.5)
    receipts = load_receipts(d, seeds=[0])
    assert receipts[0] == {10: {0: 0.5}}


def test_delta_thresholds():
    assert compute_delta(0.65, 0.50)["verdict"].startswith(
        "ReDo at a surgical dose is a real lever"
    )
    assert compute_delta(0.53, 0.50)["verdict"] == (
        "ReDo at a surgical dose is not a lever"
    )
    assert "indeterminate" in compute_delta(0.60, 0.50)["verdict"]
    assert compute_delta(None, 0.50)["verdict"] == "NOT COMPUTED"


def test_cli_end_to_end(tmp_path):
    import subprocess

    receipts_dir = tmp_path / "receipts"
    control_dir = tmp_path / "control"
    for seed in range(4):
        _write_receipt(receipts_dir, seed=seed, it=10, es=0, clear_rate=0.85)
        _write_receipt(receipts_dir, seed=seed, it=10, es=1, clear_rate=0.85)
        _write_receipt(control_dir, seed=seed, it=10, es=0, clear_rate=0.50)
        _write_receipt(control_dir, seed=seed, it=10, es=1, clear_rate=0.50)
    out = tmp_path / "theta.json"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "score_cross_fit.py"),
         "--receipts-dir", str(receipts_dir),
         "--control-receipts-dir", str(control_dir),
         "--iters", "10", "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["theta"] == 0.85
    assert payload["delta"]["delta"] == 0.35


def _write_receipt_with_rom(d, *, seed, it, es, clear_rate, rom_sha256=None):
    d.mkdir(parents=True, exist_ok=True)
    name = f"mario_1_1_v31_redo_seed{seed}_it{it:03d}_es{es}.json"
    payload = {
        "game": "mario", "clear_rate": clear_rate, "n_episodes": 50,
        "eval_seed": es, "checkpoint": f"iter_{it:05d}.pt",
    }
    if rom_sha256 is not None:
        payload["rom_sha256"] = rom_sha256
    (d / name).write_text(json.dumps(payload))


def test_load_receipts_accepts_agreeing_rom_sha256(tmp_path):
    """Every receipt naming the same rom_sha256 -- the ordinary case --
    loads exactly as before; the hash-pin is invisible when nothing is
    wrong.
    """
    d = tmp_path / "receipts"
    same = "a" * 64
    for seed in range(2):
        _write_receipt_with_rom(d, seed=seed, it=10, es=0, clear_rate=0.5,
                                 rom_sha256=same)
        _write_receipt_with_rom(d, seed=seed, it=10, es=1, clear_rate=0.6,
                                 rom_sha256=same)
    receipts = load_receipts(d, seeds=[0, 1])
    assert receipts[0] == {10: {0: 0.5, 1: 0.6}}
    assert receipts[1] == {10: {0: 0.5, 1: 0.6}}


def test_load_receipts_tolerates_receipts_with_no_rom_sha256(tmp_path):
    """Older eval_game.py receipts that never wrote rom_sha256 must keep
    loading -- the check only fires when there is something to compare.
    """
    d = tmp_path / "receipts"
    _write_receipt(d, seed=0, it=10, es=0, clear_rate=0.5)  # no rom_sha256
    receipts = load_receipts(d, seeds=[0])
    assert receipts[0] == {10: {0: 0.5}}


def test_load_receipts_voids_on_rom_sha256_mismatch(tmp_path):
    """The fixture this ships with: seeds 0-2 plus seed 3's es0 receipt
    were all scored against one ROM; seed 3's es1 receipt carries a
    DIFFERENT rom_sha256 -- same naming convention, different bytes.
    load_receipts must refuse to hand back a receipts dict at all.
    """
    d = tmp_path / "receipts"
    real, tampered = "a" * 64, "b" * 64
    for seed in range(4):
        _write_receipt_with_rom(d, seed=seed, it=10, es=0, clear_rate=0.85,
                                 rom_sha256=real)
        _write_receipt_with_rom(d, seed=seed, it=10, es=1, clear_rate=0.85,
                                 rom_sha256=real if seed != 3 else tampered)
    with pytest.raises(RomMismatchError, match="rom_sha256"):
        load_receipts(d, seeds=[0, 1, 2, 3])


def test_cli_reports_void_rom_mismatch_and_computes_no_theta(tmp_path):
    """End-to-end: the CLI must not let a ROM-inconsistent receipts dir
    reach compute_theta -- verdict says VOID-ROM-MISMATCH, theta is
    null, exit code is still 0 (this is a reported verdict, not a
    crash).
    """
    import subprocess

    d = tmp_path / "receipts"
    real, tampered = "a" * 64, "b" * 64
    for seed in range(4):
        _write_receipt_with_rom(d, seed=seed, it=10, es=0, clear_rate=0.85,
                                 rom_sha256=real)
        _write_receipt_with_rom(d, seed=seed, it=10, es=1, clear_rate=0.85,
                                 rom_sha256=real if seed != 3 else tampered)
    out = tmp_path / "theta.json"
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "score_cross_fit.py"),
         "--receipts-dir", str(d), "--iters", "10", "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["theta"] is None
    assert "VOID-ROM-MISMATCH" in payload["verdict"]
