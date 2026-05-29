"""Tests for the multi-game scoreboard's artifact join."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scoreboard import summarize  # noqa: E402


def test_summarize_untrained_game(tmp_path: Path) -> None:
    out = summarize("contra", "Contra", base=tmp_path)
    assert out["status"] == "untrained"
    assert out["game"] == "contra"


def test_summarize_joins_manifest_metrics_eval(tmp_path: Path) -> None:
    slug = "super_mario_bros"
    d = tmp_path / slug
    d.mkdir(parents=True)
    # a couple of checkpoints
    (d / "vanilla_ppo_iter_00010.pt").write_bytes(b"x")
    (d / "vanilla_ppo_iter_00020.pt").write_bytes(b"x")
    # metrics.jsonl (current run)
    (d / "metrics.jsonl").write_text(
        json.dumps({"generation": 10, "best_fitness": 700.0,
                    "vanilla_ppo_clears": 2, "ppo_entropy": 1.8,
                    "vanilla_ppo_realtime_x": 150.0}) + "\n"
        + json.dumps({"generation": 20, "best_fitness": 1200.0,
                      "vanilla_ppo_clears": 5, "ppo_entropy": 1.4,
                      "vanilla_ppo_realtime_x": 160.0}) + "\n"
    )
    # run manifest (provenance)
    (d / "run_manifest.json").write_text(json.dumps({
        "trainer_mode": "vanilla_ppo", "device": "cpu",
        "seed": 42, "git_commit": "abc1234",
    }))
    # eval.jsonl
    (d / "eval.jsonl").write_text(json.dumps({"clear_rate": 0.3, "mean_max_byte": 2.0}) + "\n")
    # curriculum stages
    (d / "smb_curriculum").mkdir()
    (d / "smb_curriculum" / "stage_01.state").write_bytes(b"x")
    (d / "smb_curriculum" / "stage_02.state").write_bytes(b"x")

    out = summarize("mario", "Super Mario Bros.", base=tmp_path)
    assert out["status"] == "trained"
    assert out["latest_iter"] == 20
    assert out["n_checkpoints"] == 2
    assert out["best_fitness"] == 1200.0          # max over rows
    assert out["total_clears"] == 7               # 2 + 5
    assert out["last_entropy"] == 1.4             # last row
    assert out["realtime_x"] == 160.0
    assert out["curriculum_stage"] == 2
    assert out["mode"] == "vanilla_ppo"
    assert out["seed"] == 42
    assert out["commit"] == "abc1234"
    assert out["eval_clear_rate"] == 0.3


def test_summarize_tolerates_corrupt_jsonl(tmp_path: Path) -> None:
    d = tmp_path / "contra"
    d.mkdir(parents=True)
    (d / "vanilla_ppo_iter_00010.pt").write_bytes(b"x")
    # a truncated/garbage line mixed with a valid one
    (d / "metrics.jsonl").write_text(
        '{"generation": 1, "best_fitness": 5.0}\n{ this is no'
    )
    out = summarize("contra", "Contra", base=tmp_path)
    assert out["status"] == "trained"
    assert out["best_fitness"] == 5.0  # the valid row parsed; garbage skipped
