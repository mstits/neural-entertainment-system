"""Regression test for scripts/gru_ab_eval.py's verdict persistence.

main()'s --seeds loop used to accumulate every seed's honest-eval
results into an in-memory dict and write verdict.json only once, after
the whole loop finished. A crash partway through a later seed (an
eval_game.py subprocess hanging past its timeout, or any other
unexpected exception) propagated straight out of the unguarded loop,
so an earlier seed's already-completed, multi-hour results were
discarded instead of reaching disk.
"""
from __future__ import annotations

import json

import pytest

from scripts import gru_ab_eval


def test_earlier_seed_results_survive_a_later_seed_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(gru_ab_eval, "REPO", tmp_path)
    monkeypatch.setattr(
        gru_ab_eval, "pick_checkpoint",
        lambda seed: (tmp_path / "ckpt.pt",
                      {"selected": str(tmp_path / "ckpt.pt"),
                       "trailing_entrance": 0.5, "table": []}))

    results = iter([{"pooled_clear_rate": 0.42}, RuntimeError("worker deadlock")])

    def fake_honest_eval(ckpt, episodes):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gru_ab_eval, "honest_eval", fake_honest_eval)
    monkeypatch.setattr(
        "sys.argv", ["gru_ab_eval.py", "--seeds", "0", "1", "--episodes", "50"])

    with pytest.raises(RuntimeError, match="worker deadlock"):
        gru_ab_eval.main()

    out = tmp_path / "runs/gru_ab/verdict.json"
    assert out.exists(), (
        "seed 0's completed honest-eval results must reach disk even "
        "though seed 1 crashed before the --seeds loop finished")
    verdict = json.loads(out.read_text())
    assert verdict["arms"]["seed0"]["seed_score"] == 0.42
    assert "seed1" not in verdict["arms"]
