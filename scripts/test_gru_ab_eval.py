#!/usr/bin/env python3
"""Regression test for gru_ab_eval.honest_eval's partial-run handling.

If one of the two `--eval-seed` subprocess runs fails to produce a
`clear_rate` (crash, bad-args JSON, OOM), honest_eval must not silently
pool the single surviving run and report it as the full "50 eps x 2 eval
seeds = 100 episodes" number from the docstring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import gru_ab_eval  # noqa: E402


def _fake_run(good_seed: int):
    """subprocess.run stand-in: `good_seed` returns a clear_rate blob, the
    other --eval-seed returns eval_game.py's real bad_args JSON shape
    (returncode=1, no clear_rate key) — mirrors the reproduced failure."""

    def run(cmd, **_kw):
        es = int(cmd[cmd.index("--eval-seed") + 1])
        if es == good_seed:
            stdout = json.dumps({
                "status": "ok", "episodes": 50, "clear_rate": 0.9,
            })
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")
        stdout = json.dumps({"status": "bad_args", "detail": "no rom"})
        return subprocess.CompletedProcess(cmd, returncode=1, stdout=stdout, stderr="")

    return run


def test_honest_eval_does_not_pool_a_single_surviving_run():
    with mock.patch("gru_ab_eval.subprocess.run", side_effect=_fake_run(good_seed=0)):
        out = gru_ab_eval.honest_eval(Path("fake.pt"), episodes=50)
    assert "pooled_clear_rate" not in out, (
        "must not report a pooled number when only one of two eval-seed "
        "runs succeeded")
    assert "pooled_clear_rate_error" in out
    assert out["eval_seed_0"]["clear_rate"] == 0.9
    assert "clear_rate" not in (out["eval_seed_1"] or {})


def test_honest_eval_pools_when_both_runs_succeed():
    def run(cmd, **_kw):
        stdout = json.dumps({"status": "ok", "episodes": 50, "clear_rate": 0.8})
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")

    with mock.patch("gru_ab_eval.subprocess.run", side_effect=run):
        out = gru_ab_eval.honest_eval(Path("fake.pt"), episodes=50)
    assert out["pooled_clear_rate"] == 0.8
    assert "pooled_clear_rate_error" not in out


if __name__ == "__main__":
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", "-v", __file__]).returncode)
