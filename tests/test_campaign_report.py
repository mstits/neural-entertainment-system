"""Unit + CLI tests for scripts/campaign_report.py, entirely against
synthetic JSONL/log fixtures written under tmp_path — no real campaign
data is touched."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import campaign_report as cr  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


CAMPAIGN_START = {
    "type": "campaign_start",
    "config": {
        "base_profile": "configs/fake.yaml",
        "bottleneck_x": 2674,
        "kill_kl_threshold": 0.15,
        "gate_sticky_clear": 0.8,
    },
    "phases": [
        {"idx": 0, "name": "critic_warmup", "gate": "critic_warmup", "budget_env_steps": 100},
        {"idx": 1, "name": "local_clear", "gate": "det_local_clears", "budget_env_steps": 200},
    ],
    "timestamp": 1000.0,
}


def _phase_start(idx, name, ts, cum=0.0, iters=10, entropy=0.01):
    return {
        "type": "phase_start",
        "phase": idx,
        "name": name,
        "cum_env_steps": cum,
        "iters": iters,
        "entropy_coef": entropy,
        "timestamp": ts,
    }


def _probe(phase, ts, env_steps, median_max_x, surv, clear_rate, status="ok", n_episodes=30):
    return {
        "type": "probe",
        "phase": phase,
        "env_steps": env_steps,
        "n_episodes": n_episodes,
        "median_max_x": median_max_x,
        "bottleneck_survival": surv,
        "clear_rate": clear_rate,
        "status": status,
        "timestamp": ts,
    }


def _gate_pass(phase, gate, ts, env_steps):
    return {"type": "gate_pass", "phase": phase, "gate": gate, "env_steps": env_steps, "timestamp": ts}


def _phase_complete(phase, name, ts, env_steps):
    return {"type": "phase_complete", "phase": phase, "name": name, "env_steps": env_steps, "timestamp": ts}


def _abort(phase, reason, ts, env_steps):
    return {"type": "abort", "phase": phase, "reason": reason, "env_steps": env_steps, "timestamp": ts}


def _metric_row(ts, gen, kl=None, vloss=None, sil_total=None, sil_trajs=None):
    row = {"generation": gen, "timestamp": ts, "ppo_entropy": 1.0}
    if kl is not None:
        row[cr.KL_FIELD] = kl
    if vloss is not None:
        row[cr.VLOSS_FIELD] = vloss
    if sil_total is not None:
        row["sil_clears_total"] = sil_total
    if sil_trajs is not None:
        row["sil_buffer_trajs"] = sil_trajs
    return row


BACKWARD_LOG_LINES = """\
20:00:00 [INFO] src.training.trainer: [vanilla_ppo] iter 1 throughput: 1000 env-steps/s
20:00:01 [INFO] src.training.trainer: [backward] iter 1: tau=1/5 (step 100 frame 400 gx 500) trailing 0/30=0.00 (advance at >=0.30 over 30) advances=0 | entrance 0/10=0.000 | truncated 0 (5 scored) | budget 1536 steps
20:00:34 [INFO] src.training.trainer: [backward] iter 2: tau=2/5 (step 150 frame 600 gx 750) trailing 5/30=0.17 (advance at >=0.30 over 30) advances=1 | entrance 1/20=0.050 | truncated 2 (15 scored) | budget 1200 steps
this line has no backward marker and should be ignored
20:01:07 [INFO] src.training.trainer: [backward] iter 3: tau=3/5 (step 200 frame 800 gx 1000) trailing 10/30=0.33 (advance at >=0.30 over 30) advances=2 | entrance 3/30=0.100 | truncated 4 (30 scored) | budget 1000 steps
"""


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_missing_file_returns_empty(tmp_path):
    assert cr.load_jsonl(tmp_path / "nope.jsonl") == []


def test_load_jsonl_skips_malformed_lines(tmp_path, capsys):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{not valid json\n{"a": 2}\n\n', encoding="utf-8")
    rows = cr.load_jsonl(path)
    assert rows == [{"a": 1}, {"a": 2}]
    err = capsys.readouterr().err
    assert "x.jsonl:2" in err


def test_load_jsonl_ignores_non_object_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('[1, 2, 3]\n{"a": 1}\n', encoding="utf-8")
    rows = cr.load_jsonl(path)
    assert rows == [{"a": 1}]


# ---------------------------------------------------------------------------
# load_campaign / load_metrics / load_backward_lines
# ---------------------------------------------------------------------------


def test_load_campaign_reads_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "campaign.jsonl", [CAMPAIGN_START, _phase_start(0, "critic_warmup", 1000.5)])
    events = cr.load_campaign(run_dir)
    assert len(events) == 2
    assert events[0]["type"] == "campaign_start"


def test_load_metrics_merges_current_and_rotated_sorted_by_timestamp(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    # Rotated (older) phase metrics, gens 0..2
    _write_jsonl(
        ckpt_dir / "runs" / "20260101_000000" / "metrics.jsonl",
        [_metric_row(100.0, 0), _metric_row(101.0, 1), _metric_row(102.0, 2)],
    )
    # Current (newer) phase metrics, gens overlap on purpose (resume replay)
    _write_jsonl(
        ckpt_dir / "metrics.jsonl",
        [_metric_row(200.0, 1), _metric_row(201.0, 2)],
    )
    rows = cr.load_metrics(ckpt_dir)
    assert [r["timestamp"] for r in rows] == [100.0, 101.0, 102.0, 200.0, 201.0]


def test_load_metrics_missing_dirs_returns_empty(tmp_path):
    assert cr.load_metrics(tmp_path / "nope") == []


def test_load_backward_lines_parses_fields_and_skips_other_lines(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "phase_2.log").write_text(BACKWARD_LOG_LINES, encoding="utf-8")
    by_phase = cr.load_backward_lines(run_dir)
    assert set(by_phase.keys()) == {2}
    recs = by_phase[2]
    assert len(recs) == 3
    assert recs[0] == {
        "iter": 1,
        "tau": 1,
        "tau_max": 5,
        "gx": 500,
        "trail_rate": 0.0,
        "advances": 0,
        "ent_n": 0,
        "ent_d": 10,
        "ent_rate": 0.0,
        "truncated": 0,
        "scored": 5,
        "budget": 1536,
    }
    assert recs[-1]["tau"] == 3
    assert recs[-1]["advances"] == 2


def test_load_backward_lines_no_phase_logs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert cr.load_backward_lines(run_dir) == {}


# ---------------------------------------------------------------------------
# phase_config_index / phase_starts / assign_phase / grouping
# ---------------------------------------------------------------------------


def test_phase_config_index():
    idx = cr.phase_config_index([CAMPAIGN_START])
    assert idx[0]["name"] == "critic_warmup"
    assert idx[1]["gate"] == "det_local_clears"


def test_phase_config_index_no_campaign_start():
    assert cr.phase_config_index([_phase_start(0, "x", 1.0)]) == {}


def test_phase_starts_sorted():
    events = [_phase_start(1, "b", 500.0), _phase_start(0, "a", 100.0)]
    starts = cr.phase_starts(events)
    assert starts == [(100.0, 0, "a"), (500.0, 1, "b")]


def test_assign_phase_buckets_by_last_start_before_ts():
    starts = [(100.0, 0, "a"), (500.0, 1, "b")]
    assert cr.assign_phase(50.0, starts) is None
    assert cr.assign_phase(100.0, starts) == 0
    assert cr.assign_phase(300.0, starts) == 0
    assert cr.assign_phase(500.0, starts) == 1
    assert cr.assign_phase(9999.0, starts) == 1


def test_group_metrics_by_phase():
    starts = [(100.0, 0, "a"), (500.0, 1, "b")]
    rows = [_metric_row(50.0, 0), _metric_row(150.0, 1), _metric_row(600.0, 2)]
    grouped = cr.group_metrics_by_phase(rows, starts)
    assert grouped[None] == [rows[0]]
    assert grouped[0] == [rows[1]]
    assert grouped[1] == [rows[2]]


# ---------------------------------------------------------------------------
# summarize_series
# ---------------------------------------------------------------------------


def test_summarize_series_empty():
    s = cr.summarize_series([])
    assert s["n"] == 0
    assert s["min"] is None
    assert s["flat"] is None
    assert s["direction"] is None


def test_summarize_series_flat_sequence_is_plateau():
    values = [0.30, 0.31, 0.30, 0.29, 0.30] * 6  # 30 pts, tight band
    s = cr.summarize_series(values)
    assert s["flat"] is True
    assert s["direction"] == "flat"


def test_summarize_series_clearly_increasing():
    values = [float(i) for i in range(1, 41)]  # 1..40, strictly rising
    s = cr.summarize_series(values)
    assert s["min"] == 1.0
    assert s["max"] == 40.0
    assert s["flat"] is False
    assert s["direction"] == "increasing"


def test_summarize_series_clearly_decreasing():
    values = [float(i) for i in range(40, 0, -1)]
    s = cr.summarize_series(values)
    assert s["direction"] == "decreasing"


def test_summarize_series_ignores_none_entries():
    values = [None, 1.0, None, 1.0, None, 1.0]
    s = cr.summarize_series(values)
    assert s["n"] == 3


def test_summarize_series_short_series_direction_none():
    s = cr.summarize_series([1.0])
    assert s["n"] == 1
    assert s["direction"] is None


# ---------------------------------------------------------------------------
# summarize_sil / summarize_backward
# ---------------------------------------------------------------------------


def test_summarize_sil_present():
    rows = [
        _metric_row(1.0, 0, sil_total=5, sil_trajs=2),
        _metric_row(2.0, 1, sil_total=9, sil_trajs=4),
    ]
    s = cr.summarize_sil(rows)
    assert s == {"start": 5, "end": 9, "delta": 4, "buffer_trajs_end": 4}


def test_summarize_sil_absent():
    rows = [_metric_row(1.0, 0)]
    assert cr.summarize_sil(rows) is None


def test_summarize_backward_present():
    run_dir_recs = [
        {"tau": 1, "tau_max": 5, "advances": 0, "ent_n": 0, "ent_d": 10, "ent_rate": 0.0, "budget": 1536},
        {"tau": 3, "tau_max": 5, "advances": 2, "ent_n": 3, "ent_d": 30, "ent_rate": 0.1, "budget": 1000},
    ]
    s = cr.summarize_backward(run_dir_recs)
    assert s["tau_start"] == "1/5"
    assert s["tau_end"] == "3/5"
    assert s["advances"] == 2
    assert s["entrance_start"] == "0/10=0.000"
    assert s["entrance_end"] == "3/30=0.100"
    assert s["budget_end"] == 1000
    assert s["n_lines"] == 2


def test_summarize_backward_empty():
    assert cr.summarize_backward([]) is None


# ---------------------------------------------------------------------------
# build_timeline / build_probe_table / determine_status
# ---------------------------------------------------------------------------


def test_build_timeline_formats_all_event_types():
    events = [
        CAMPAIGN_START,
        _phase_start(0, "critic_warmup", 1000.5),
        _gate_pass(0, "critic_warmup", 1050.0, 100.0),
        _probe(0, 1060.0, 100.0, 187.0, 0.0, 0.0),
        _phase_complete(0, "critic_warmup", 1070.0, 100.0),
        _abort(1, "KILL something", 1200.0, 300.0),
    ]
    lines = cr.build_timeline(events)
    assert len(lines) == 6
    assert "campaign_start" in lines[0]
    assert "phase 0 'critic_warmup' START" in lines[1]
    assert "GATE PASS 'critic_warmup'" in lines[2]
    assert "PROBE" in lines[3] and "median_max_x=187.0" in lines[3]
    assert "COMPLETE" in lines[4]
    assert "ABORT" in lines[5] and "KILL something" in lines[5]


def test_build_probe_table_filters_type():
    events = [CAMPAIGN_START, _probe(0, 1.0, 1.0, 5.0, 0.0, 0.0), _gate_pass(0, "g", 2.0, 2.0)]
    probes = cr.build_probe_table(events)
    assert len(probes) == 1
    assert probes[0]["type"] == "probe"


def test_determine_status_empty():
    s = cr.determine_status([], {})
    assert s["state"] == "UNKNOWN"


def test_determine_status_aborted():
    phase_cfg = {0: {"name": "critic_warmup"}, 1: {"name": "local_clear"}}
    events = [_phase_start(0, "critic_warmup", 1.0), _abort(0, "KILL x", 2.0, 50.0)]
    s = cr.determine_status(events, phase_cfg)
    assert s["state"] == "ABORTED"
    assert "critic_warmup" in s["detail"]
    assert "KILL x" in s["detail"]


def test_determine_status_completed_final_phase():
    phase_cfg = {0: {"name": "a"}, 1: {"name": "b"}}
    events = [_phase_start(1, "b", 1.0), _phase_complete(1, "b", 2.0, 500.0)]
    s = cr.determine_status(events, phase_cfg)
    assert s["state"] == "COMPLETED"


def test_determine_status_completed_non_final_phase_is_running():
    phase_cfg = {0: {"name": "a"}, 1: {"name": "b"}}
    events = [_phase_start(0, "a", 1.0), _phase_complete(0, "a", 2.0, 100.0)]
    s = cr.determine_status(events, phase_cfg)
    assert s["state"] == "RUNNING"


def test_determine_status_in_progress():
    phase_cfg = {0: {"name": "a"}}
    events = [_phase_start(0, "a", 1.0), _probe(0, 2.0, 10.0, 1.0, 0.0, 0.0)]
    s = cr.determine_status(events, phase_cfg)
    assert s["state"] == "RUNNING"
    assert "phase 0" in s["detail"]


# ---------------------------------------------------------------------------
# build_report / build_verdict / render_report (integration, synthetic dirs)
# ---------------------------------------------------------------------------


def _build_synthetic_attempt(tmp_path: Path, aborted: bool = True) -> tuple[Path, Path]:
    run_dir = tmp_path / "runs" / "fake_attempt"
    ckpt_dir = tmp_path / "checkpoints" / "fake_ckpt"

    events = [
        CAMPAIGN_START,
        _phase_start(0, "critic_warmup", 1000.0),
        _probe(0, 1050.0, 50.0, 187.0, 0.0, 0.0),
        _gate_pass(0, "critic_warmup", 1051.0, 50.0),
        _phase_complete(0, "critic_warmup", 1052.0, 50.0),
        _phase_start(1, "local_clear", 1053.0, cum=50.0),
    ]
    if aborted:
        events.append(_abort(1, "KILL kl_anchor_div>0.15 sustained", 1200.0, 150.0))
    _write_jsonl(run_dir / "campaign.jsonl", events)

    # phase_0 metrics: flat KL, flat value loss
    phase0_rows = [
        _metric_row(1000.0 + i, i, kl=0.05, vloss=10.0, sil_total=i, sil_trajs=i)
        for i in range(5)
    ]
    _write_jsonl(ckpt_dir / "runs" / "20260101_000000" / "metrics.jsonl", phase0_rows)

    # phase_1 metrics: rising KL (heading toward the kill threshold).
    # Needs more rows than PLATEAU_WINDOW so the leading/trailing windows
    # used for direction detection don't fully overlap.
    phase1_rows = [
        _metric_row(1053.0 + i, 50 + i, kl=0.05 + 0.02 * i, vloss=10.0 + i, sil_total=5 + i, sil_trajs=5 + i)
        for i in range(25)
    ]
    _write_jsonl(ckpt_dir / "metrics.jsonl", phase1_rows)

    # backward-curriculum log line for phase 1
    (run_dir / "phase_1.log").write_text(
        "20:00:00 [INFO] src.training.trainer: [backward] iter 1: tau=1/5 (step 100 frame 400 gx 500) "
        "trailing 0/30=0.00 (advance at >=0.30 over 30) advances=0 | entrance 0/10=0.000 | "
        "truncated 0 (5 scored) | budget 1536 steps\n"
        "20:00:34 [INFO] src.training.trainer: [backward] iter 2: tau=2/5 (step 150 frame 600 gx 750) "
        "trailing 5/30=0.17 (advance at >=0.30 over 30) advances=1 | entrance 1/20=0.050 | "
        "truncated 2 (15 scored) | budget 1200 steps\n",
        encoding="utf-8",
    )
    return run_dir, ckpt_dir


def test_build_report_aborted_attempt(tmp_path):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    report = cr.build_report(run_dir, ckpt_dir)

    assert report["status"]["state"] == "ABORTED"
    assert "KILL kl_anchor_div" in report["status"]["detail"]
    assert len(report["probes"]) == 1
    assert report["probes"][0]["median_max_x"] == 187.0

    phases_by_idx = {p["idx"]: p for p in report["phases"]}
    assert 0 in phases_by_idx and 1 in phases_by_idx

    p0 = phases_by_idx[0]
    assert p0["kl"]["n"] == 5
    assert p0["kl"]["flat"] is True

    p1 = phases_by_idx[1]
    assert p1["kl"]["n"] == 25
    assert p1["kl"]["direction"] == "increasing"
    assert p1["backward"] is not None
    assert p1["backward"]["tau_start"] == "1/5"
    assert p1["backward"]["tau_end"] == "2/5"
    assert p1["sil"]["delta"] == 24


def test_build_report_running_attempt(tmp_path):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=False)
    report = cr.build_report(run_dir, ckpt_dir)
    assert report["status"]["state"] == "RUNNING"


def test_build_report_missing_everything(tmp_path):
    report = cr.build_report(tmp_path / "no_run", tmp_path / "no_ckpt")
    assert report["status"]["state"] == "UNKNOWN"
    assert report["phases"] == []
    assert report["probes"] == []


def test_build_verdict_mentions_abort_reason_and_probe_stats(tmp_path):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    report = cr.build_report(run_dir, ckpt_dir)
    verdict = cr.build_verdict(report)
    assert "ABORTED" in verdict
    assert "KILL kl_anchor_div" in verdict
    assert "187" in verdict
    assert "tau" not in verdict.lower() or "1/5" in verdict  # backward clause present when data exists


def test_render_report_contains_expected_sections(tmp_path):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    report = cr.build_report(run_dir, ckpt_dir)
    text = cr.render_report(report)
    assert "# Campaign report:" in text
    assert "## Timeline" in text
    assert "## Probe table" in text
    assert "## Per-phase summary" in text
    assert "## Status" in text
    assert "## Verdict" in text
    assert "Phase 0" in text
    assert "Phase 1" in text


def test_render_report_empty_dirs_does_not_crash(tmp_path):
    report = cr.build_report(tmp_path / "empty_run", tmp_path / "empty_ckpt")
    text = cr.render_report(report)
    assert "no campaign.jsonl events found" in text
    assert "no probes found" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_cli_stdout(tmp_path, capsys):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    rc = cr.main(["--run-dir", str(run_dir), "--ckpt-dir", str(ckpt_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Campaign report" in out
    assert "ABORTED" in out


def test_main_cli_writes_out_file(tmp_path):
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    out_path = tmp_path / "report.md"
    rc = cr.main(
        ["--run-dir", str(run_dir), "--ckpt-dir", str(ckpt_dir), "--out", str(out_path)]
    )
    assert rc == 0
    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert "# Campaign report:" in content


def test_cli_subprocess_smoke(tmp_path):
    """Run the script as an actual subprocess (not just calling main() in
    process) to catch import-time / argparse-level breakage."""
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    script = Path(__file__).resolve().parents[1] / "scripts" / "campaign_report.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run-dir", str(run_dir), "--ckpt-dir", str(ckpt_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Campaign report" in result.stdout


def test_main_cli_never_writes_inside_run_or_ckpt_dir(tmp_path):
    """Guard against regressions that would make this script unsafe to
    point at a live campaign directory: after running (with --out
    elsewhere), run_dir and ckpt_dir must contain exactly the files we
    put there."""
    run_dir, ckpt_dir = _build_synthetic_attempt(tmp_path, aborted=True)
    before_run = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    before_ckpt = sorted(p.relative_to(ckpt_dir) for p in ckpt_dir.rglob("*"))

    cr.main(
        [
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--out",
            str(tmp_path / "elsewhere" / "report.md"),
        ]
    )

    after_run = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    after_ckpt = sorted(p.relative_to(ckpt_dir) for p in ckpt_dir.rglob("*"))
    assert before_run == after_run
    assert before_ckpt == after_ckpt
