"""scripts/eval_shared_substrate.py -- the shared-substrate (trunk +
per-level heads) vs. isolated-baseline evaluation harness.

Everything here runs on SYNTHETIC eval_game.py JSON and injected
`run_eval` stubs -- no emulator, no rollout, no subprocess that steps
the Rust pool. The one live-ish test is the --dry-run subprocess, which
is explicitly allowed to parse configs, torch.load the four baseline
checkpoints, and assemble eval_game.py commands (none of which is a
rollout).

Pre-registration under test (mirrors the in-file CONFIG):
* baseline = 43 (1-1) + 38 (1-2) + 21 (1-3) + 51 (1-4) = 153/400, all
  definitive strict-predicate numbers this repo already owns (CLAIMS.md,
  docs/proposals/RESUME_2026-08-17.md) -- but NOT all measured under one
  uniform protocol shape. CONFIG["baseline_provenance"] records, per
  level, exactly what protocol produced each count, and the
  test_baseline_provenance_* tests below open the real receipt file(s)
  for every level and check that recorded provenance against them: 1-3
  and 1-4 match this harness's own re-eval protocol exactly; 1-1 matches
  in protocol shape but comes from a single 100-episode seed rather than
  the two-seed 50/50 split; 1-2 does NOT match (eval_workers 1 not 5,
  eval_rng shared-stream not per-episode, no --sequential/--level-clear)
  and re-runs 5 points higher (43/100) under this harness's own protocol
  than its banked 38/100 -- the noisiest leg of the comparison;
* one eval_game.py leg per level per seed, same argv shape as
  run_online_campaign.build_probe_command, against that level's own
  profile and its own start_state_path (the cold entrance every banked
  baseline number above was measured from) -- this is THIS harness's
  own re-eval shape for scoring the shared-substrate export, not a
  retroactive claim about how every baseline number was produced;
* verdict: SUPERSEDES (beats baseline, nothing collapses),
  MIXED (beats baseline, something collapses more than 10 points below
  its own baseline -- the disguised-interference case), FAILED
  (does not beat baseline), UNUSABLE (any leg broken) -- precedence
  UNUSABLE first, always;
* an exact one-sided binomial test against the pooled baseline rate is
  reported as advisory statistical context alongside the raw-count
  verdict, never a substitute for it.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Guard against the pollution this module SHIPPED for 12 days: a test
# passing the production CONFIG through run() unredirected appended 4
# mock legs + 1 mock SUPERSEDES verdict to runs/shared_substrate/'s
# REAL receipt log on every full-suite run — 204 fabricated verdicts
# (checkpoints under /fake/dir/, every rate exactly 0.5) that read as a
# completed successful experiment to anyone listing the dir. Every test
# here must leave the production receipt paths byte-identical.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _production_receipts_untouched():
    prod = [Path(__file__).resolve().parent.parent / "runs" /
            "shared_substrate" / n
            for n in ("eval_shared_substrate.jsonl", "manifest.json")]
    def snap():
        return [(p.exists(), p.stat().st_mtime_ns if p.exists() else 0,
                 p.stat().st_size if p.exists() else 0) for p in prod]
    before = snap()
    yield
    assert snap() == before, (
        "a test wrote to the PRODUCTION shared_substrate receipt paths "
        "— redirect receipt_log/manifest_out to tmp_path")


from scripts.eval_shared_substrate import (  # noqa: E402
    CONFIG, VERDICTS, aggregate_result, aggregate_significance,
    build_level_command, build_manifest, classify_verdict,
    format_verdict_table, head_checkpoint_path, level_delta,
    placeholder_table, run, run_level_leg, strict_clears_from_eval_json,
    write_manifest,
)


# ---- fixtures --------------------------------------------------------


def _eval_json(*, status="ok", n=50, clear_rate=0.5, seq_clear_rate=None):
    row = {"status": status, "n_episodes": n, "clear_rate": clear_rate}
    if seq_clear_rate is not None:
        row["seq_clear_rate"] = seq_clear_rate
    return row


def _leg(shared, *, status="ok", errors=None):
    return {"strict_clears": shared, "status": status,
            "errors": list(errors or [])}


def _stub_run_eval(rates_by_seed):
    """A run_eval callable returning canned eval_game JSON keyed by the
    --eval-seed value pulled out of the assembled argv."""
    def _run(cmd):
        seed = int(cmd[cmd.index("--eval-seed") + 1])
        rate = rates_by_seed[seed]
        return _eval_json(clear_rate=rate)
    return _run


def _load_real_json(relpath):
    """Reads one real receipt JSON off disk (relative to ROOT). Pure
    filesystem read -- no subprocess, no emulator."""
    return json.loads((ROOT / relpath).read_text())


def _load_real_jsonl_record(relpath, **match):
    """Finds the first record in a real .jsonl receipt log whose fields
    match every kwarg given. Pure filesystem read."""
    for line in (ROOT / relpath).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    raise AssertionError(f"no record matching {match} in {relpath}")


# ---- baseline / CONFIG sanity -----------------------------------------


def test_baseline_sum_matches_the_preregistered_total():
    total = sum(int(s["strict_clears"]) for s in CONFIG["baseline"].values())
    assert total == CONFIG["baseline_sum"] == 153


def test_baseline_numbers_match_the_banked_receipts():
    assert CONFIG["baseline"]["1-1"]["strict_clears"] == 43
    assert CONFIG["baseline"]["1-2"]["strict_clears"] == 38
    assert CONFIG["baseline"]["1-3"]["strict_clears"] == 21
    assert CONFIG["baseline"]["1-4"]["strict_clears"] == 51


def test_baseline_checkpoints_match_the_banked_paths():
    assert CONFIG["baseline"]["1-1"]["checkpoint"] == (
        "checkpoints/_preserved/backward_1_1_best_honest_greedy_047.pt")
    assert CONFIG["baseline"]["1-2"]["checkpoint"] == (
        "checkpoints/_preserved/consol2_40pct_strict_iter01120.pt")
    assert CONFIG["baseline"]["1-3"]["checkpoint"] == (
        "checkpoints/_preserved/one_three_FINAL_consol2_iter00690.pt")
    assert CONFIG["baseline"]["1-4"]["checkpoint"] == (
        "checkpoints/_preserved/one_four_MEASURED60_iter00960.pt")


def test_harness_own_eval_protocol_config():
    """This is the harness's OWN fixed re-eval protocol (what
    build_level_command assembles) -- it says nothing on its own about
    what protocol produced the banked baseline numbers on the other
    side of the comparison. That cross-check lives in the
    test_baseline_provenance_* tests below, which open the real
    receipt files."""
    assert CONFIG["episodes_per_seed"] == 50
    assert tuple(CONFIG["eval_seeds"]) == (7, 101)
    assert CONFIG["sticky_prob"] == 0.25
    assert CONFIG["start_jitter"] == 16
    assert CONFIG["eval_workers"] == 5
    assert CONFIG["eval_rng"] == "per-episode"
    assert CONFIG["baseline_n_episodes"] == 400


def test_baseline_provenance_1_1_matches_the_real_receipt():
    """1-1's banked 43/100 traces to ONE 100-episode eval-seed run
    (20260816), not the two-seed [7, 101] x 50 split CONFIG's own
    eval-protocol fields describe -- protocol SHAPE matches (sequential
    + level-clear, 5 workers, per-episode RNG), only the seed/episode
    split differs. Reads the real receipt off disk; would fail if
    CONFIG["baseline_provenance"]["1-1"] silently drifted from it."""
    prov = CONFIG["baseline_provenance"]["1-1"]
    rec = _load_real_jsonl_record(prov["receipts"][0], **prov["receipt_selector"])
    cmd = rec["cmd"]
    assert "--sequential" in cmd and "--level-clear" in cmd
    assert re.search(r"--eval-workers 5\b", cmd)
    assert re.search(r"--eval-rng per-episode\b", cmd)
    seed_match = re.search(r"--eval-seed (\d+)\b", cmd)
    assert (int(seed_match.group(1)),) == prov["eval_seeds"] == (20260816,)
    assert rec["n_episodes"] == prov["n_episodes_total"] == 100
    clears = round(rec["clear_rate"] * rec["n_episodes"])
    assert clears == CONFIG["baseline"]["1-1"]["strict_clears"] == 43
    assert prov["matches_harness_protocol"] is True
    assert prov["matches_two_seed_fifty_split"] is False


def test_baseline_provenance_1_2_does_not_match_the_harness_protocol():
    """1-2's banked 38/100 was measured under a DIFFERENT protocol than
    build_level_command assembles: eval_workers 1 (not 5), eval_rng
    shared-stream (not per-episode), no --sequential/--level-clear.
    Reads both real per-seed receipts off disk -- this is the exact
    check the removed no-op version of this test never performed, and
    it must NOT assert the two protocols are equal."""
    prov = CONFIG["baseline_provenance"]["1-2"]
    total_clears = 0
    for path, seed in zip(prov["receipts"], prov["eval_seeds"]):
        rec = _load_real_json(path)
        assert rec["eval_seed"] == seed
        assert rec["eval_workers"] == 1
        assert rec["eval_rng"] == "shared-stream"
        assert "sequential" not in rec
        assert rec["checkpoint"] == CONFIG["baseline"]["1-2"]["checkpoint"]
        total_clears += round(rec["clear_rate"] * rec["n_episodes"])
    assert total_clears == CONFIG["baseline"]["1-2"]["strict_clears"] == 38
    assert prov["eval_workers"] == 1
    assert prov["eval_rng"] == "shared-stream"
    assert prov["sequential"] is False
    assert prov["matches_harness_protocol"] is False


def test_baseline_provenance_1_2_reruns_higher_under_the_harness_protocol():
    """Independent confirmation that the 1-2 protocol mismatch is
    material, not academic: re-running the SAME 1-2 checkpoint under
    THIS harness's own protocol (sequential/level-clear, eval_workers
    5, eval_rng per-episode) reads a materially different strict-clear
    count than the banked 38/100 -- within level_collapse_margin, which
    is exactly why 1-2 is the noisiest leg of the comparison."""
    rec = _load_real_jsonl_record(
        "runs/interference/interference.jsonl",
        type="probe", level="1-2", role="control")
    cmd = rec["cmd"]
    assert "--sequential" in cmd and "--level-clear" in cmd
    assert re.search(r"--eval-workers 5\b", cmd)
    assert re.search(r"--eval-rng per-episode\b", cmd)
    rerun_clears = round(rec["clear_rate"] * rec["n_episodes"])
    banked_clears = CONFIG["baseline"]["1-2"]["strict_clears"]
    assert rerun_clears == 43
    assert rerun_clears != banked_clears
    assert 0 < abs(rerun_clears - banked_clears) < CONFIG["level_collapse_margin"]


@pytest.mark.parametrize("level,receipt_dir", [
    ("1-3", "runs/consol2_1_3_round2"), ("1-4", "runs/online_1_4"),
])
def test_baseline_provenance_matches_the_harness_protocol_exactly(
        level, receipt_dir):
    """1-3 and 1-4's banked numbers were measured under exactly this
    harness's own protocol shape and seed/episode split -- reads both
    real per-seed receipts off disk and checks every protocol field,
    not just the count."""
    prov = CONFIG["baseline_provenance"][level]
    total_clears = 0
    for path, seed in zip(prov["receipts"], prov["eval_seeds"]):
        rec = _load_real_json(path)
        assert path.startswith(receipt_dir)
        assert rec["eval_seed"] == seed
        assert rec["sequential"] is True
        assert rec["level_clear"] is True
        assert rec["eval_workers"] == CONFIG["eval_workers"] == 5
        assert rec["eval_rng"] == CONFIG["eval_rng"] == "per-episode"
        assert rec["sticky_prob"] == CONFIG["sticky_prob"]
        assert rec["start_jitter"] == CONFIG["start_jitter"]
        assert rec["checkpoint"] == CONFIG["baseline"][level]["checkpoint"]
        total_clears += round(rec["clear_rate"] * rec["n_episodes"])
    assert total_clears == CONFIG["baseline"][level]["strict_clears"]
    assert prov["matches_harness_protocol"] is True
    assert prov["matches_two_seed_fifty_split"] is True


# ---- head_checkpoint_path ----------------------------------------------


@pytest.mark.parametrize("level,expected", [
    ("1-1", "1-1.pt"), ("1-2", "1-2.pt"), ("1-3", "1-3.pt"), ("1-4", "1-4.pt"),
])
def test_head_checkpoint_path_naming_convention(level, expected):
    """Must match train_shared_substrate.py's own export_checkpoints()
    naming (_slug(level) + ".pt") so the eval harness looks in the same
    place the trainer writes to."""
    p = head_checkpoint_path(level, "checkpoints/shared_substrate_v1")
    assert p == Path("checkpoints/shared_substrate_v1") / expected


# ---- argv assembly -------------------------------------------------------


@pytest.mark.parametrize("level,profile,game,start_state_suffix", [
    ("1-1", "configs/mario_1_1_backward.yaml", "mario_1_1_backward",
     "runs/live_show/smb_4_4_micro/entrance_start.state"),
    ("1-2", "configs/mario_1_2_consol2.yaml", "mario_1_2_consol2",
     "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/"
     "stage_03.state"),
    ("1-3", "configs/mario_1_3_online_v1.yaml", "mario_1_3_online_v1",
     "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/"
     "stage_05.state"),
    ("1-4", "configs/mario_1_4_online_v1.yaml", "mario_1_4_online_v1",
     "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/"
     "stage_10.state"),
])
def test_build_level_command_uses_the_levels_own_profile_and_entrance(
        level, profile, game, start_state_suffix):
    cmd = build_level_command(
        level, checkpoint="/fake/head.pt", seed=7, cfg=CONFIG)
    assert cmd[0] == sys.executable
    assert cmd[1] == str(ROOT / "scripts" / "eval_game.py")
    assert "--game" in cmd and cmd[cmd.index("--game") + 1] == game
    assert cmd[cmd.index("--profile") + 1] == str(ROOT / profile)
    assert cmd[cmd.index("--checkpoint") + 1] == "/fake/head.pt"
    assert cmd[cmd.index("--start-state") + 1] == str(ROOT / start_state_suffix)


def test_build_level_command_is_the_honest_protocol_shape():
    cmd = build_level_command(
        "1-1", checkpoint="/fake/head.pt", seed=101, cfg=CONFIG)
    assert "--sequential" in cmd and "--level-clear" in cmd
    assert cmd[cmd.index("--episodes") + 1] == "50"
    assert cmd[cmd.index("--max-steps") + 1] == "3000"
    assert cmd[cmd.index("--eval-seed") + 1] == "101"
    assert cmd[cmd.index("--sticky-prob") + 1] == "0.25"
    assert cmd[cmd.index("--start-jitter") + 1] == "16"
    assert cmd[cmd.index("--eval-workers") + 1] == "5"
    assert cmd[cmd.index("--eval-rng") + 1] == "per-episode"
    # Greedy is eval_game's default action-select; this harness never
    # overrides it, so --action-select must not appear.
    assert "--action-select" not in cmd


def test_build_level_command_argv_assembles_against_real_paths():
    """Every path-shaped arg (besides the fake --checkpoint) must exist
    on disk -- the same 'assembles against real paths' contract the
    other campaign controllers' dry-runs check."""
    for level in CONFIG["levels"]:
        cmd = build_level_command(
            level, checkpoint="/fake/head.pt", seed=7, cfg=CONFIG)
        for arg in cmd:
            if "/" in arg and not arg.startswith("-") and arg != "/fake/head.pt":
                assert Path(arg).exists(), f"{level}: missing {arg}"


# ---- strict_clears_from_eval_json ---------------------------------------


def test_strict_clears_uses_clear_rate_not_seq_clear_rate():
    """clear_rate is the strict (episode_success) predicate;
    seq_clear_rate (chain-advance) must never substitute -- the
    2026-08-15 forensics found the two disagree 13:1 on real data."""
    row = _eval_json(n=100, clear_rate=0.21, seq_clear_rate=0.26)
    assert strict_clears_from_eval_json(row) == 21


def test_strict_clears_none_on_non_ok_status():
    assert strict_clears_from_eval_json(_eval_json(status="eval_timeout")) is None
    assert strict_clears_from_eval_json(_eval_json(status="eval_failed")) is None


def test_strict_clears_none_on_zero_episodes():
    assert strict_clears_from_eval_json(_eval_json(n=0)) is None


def test_strict_clears_none_when_clear_rate_missing():
    row = {"status": "ok", "n_episodes": 50}
    assert strict_clears_from_eval_json(row) is None


def test_strict_clears_none_on_non_dict():
    assert strict_clears_from_eval_json(None) is None
    assert strict_clears_from_eval_json("not json") is None


def test_strict_clears_rounds_to_nearest_episode():
    # 24/50 = 0.48 exactly -> 24, not a float artifact.
    assert strict_clears_from_eval_json(_eval_json(n=50, clear_rate=0.48)) == 24


# ---- run_level_leg (subprocess injected, no emulator) --------------------


def test_run_level_leg_sums_both_seeds():
    leg = run_level_leg(
        "1-2", checkpoint="/fake/head.pt", cfg=CONFIG,
        run_eval=_stub_run_eval({7: 0.50, 101: 0.30}))
    assert leg["status"] == "ok"
    # 50*0.50 + 50*0.30 = 25 + 15 = 40
    assert leg["strict_clears"] == 40
    assert leg["n_episodes"] == 100
    assert len(leg["per_seed"]) == 2


def test_run_level_leg_records_errors_and_returns_no_total_on_failure():
    def flaky(cmd):
        seed = int(cmd[cmd.index("--eval-seed") + 1])
        if seed == 101:
            return {"status": "eval_timeout"}
        return _eval_json(n=50, clear_rate=0.5)
    leg = run_level_leg(
        "1-1", checkpoint="/fake/head.pt", cfg=CONFIG, run_eval=flaky)
    assert leg["status"] == "error"
    assert leg["strict_clears"] is None
    assert leg["errors"] and "101" in leg["errors"][0]


def test_run_level_leg_passes_the_given_checkpoint_through():
    seen = []

    def capture(cmd):
        seen.append(cmd[cmd.index("--checkpoint") + 1])
        return _eval_json(n=50, clear_rate=0.4)

    run_level_leg("1-3", checkpoint="/fake/head_1_3.pt", cfg=CONFIG,
                  run_eval=capture)
    assert seen == ["/fake/head_1_3.pt", "/fake/head_1_3.pt"]


# ---- _run_eval_subprocess (subprocess.run monkeypatched, no emulator) ----


def test_run_eval_subprocess_recovers_status_json_on_nonzero_exit():
    """eval_game.py's own guards (e.g. no_rom) print a status JSON to
    stdout and then exit non-zero with nothing on stderr -- that JSON
    must still come back, not get swapped for an empty stderr detail
    (regression: returncode used to be checked before stdout was ever
    read)."""
    import scripts.eval_shared_substrate as mod

    no_rom_json = json.dumps(
        {"status": "no_rom", "rom_path": "/missing/game.nes"})
    real_subprocess_run = mod.subprocess.run

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout=no_rom_json, stderr="")

    mod.subprocess.run = fake_run
    try:
        result = mod._run_eval_subprocess(["fake", "cmd"], timeout_s=1.0)
    finally:
        mod.subprocess.run = real_subprocess_run
    assert result == {"status": "no_rom", "rom_path": "/missing/game.nes"}


# ---- level_delta ----------------------------------------------------------


def test_level_delta_computes_baseline_shared_delta():
    d = level_delta("1-1", _leg(50), CONFIG)
    assert d == {"level": "1-1", "baseline": 43, "shared": 50, "delta": 7,
                "collapsed": False, "status": "ok", "errors": []}


def test_level_delta_collapsed_at_eleven_points_below():
    # 1-2 baseline is 38; 38 - 11 = 27 is MORE than 10 points below.
    d = level_delta("1-2", _leg(27), CONFIG)
    assert d["delta"] == -11
    assert d["collapsed"] is True


def test_level_delta_not_collapsed_at_exactly_ten_points_below():
    # "falls MORE THAN 10 points below" -- exactly 10 does not collapse.
    d = level_delta("1-2", _leg(28), CONFIG)
    assert d["delta"] == -10
    assert d["collapsed"] is False


def test_level_delta_shared_and_delta_none_on_errored_leg():
    d = level_delta("1-4", _leg(None, status="error", errors=["boom"]), CONFIG)
    assert d["shared"] is None
    assert d["delta"] is None
    assert d["collapsed"] is False
    assert d["status"] == "error"
    assert d["errors"] == ["boom"]


# ---- aggregate_result -----------------------------------------------------


def test_aggregate_result_sums_all_four_levels():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 50), ("1-2", 45), ("1-3", 25), ("1-4", 55))]
    agg = aggregate_result(deltas, CONFIG)
    assert agg["shared_sum"] == 175
    assert agg["baseline_sum"] == 153
    assert agg["delta"] == 22
    assert agg["beats_baseline"] is True
    assert agg["collapsed_levels"] == []
    assert agg["errored_levels"] == []


def test_aggregate_result_none_when_any_level_errored():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 50), ("1-2", 45), ("1-3", 25))]
    deltas.append(level_delta("1-4", _leg(None, status="error")))
    agg = aggregate_result(deltas, CONFIG)
    assert agg["shared_sum"] is None
    assert agg["delta"] is None
    assert agg["beats_baseline"] is None
    assert agg["errored_levels"] == ["1-4"]


def test_aggregate_result_reports_collapsed_levels():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 60), ("1-2", 20), ("1-3", 25), ("1-4", 55))]
    agg = aggregate_result(deltas, CONFIG)
    assert agg["collapsed_levels"] == ["1-2"]


def test_aggregate_result_tie_does_not_beat_baseline():
    # 43 + 38 + 21 + 51 == 153: an exact tie is not "beats".
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 43), ("1-2", 38), ("1-3", 21), ("1-4", 51))]
    agg = aggregate_result(deltas, CONFIG)
    assert agg["shared_sum"] == 153
    assert agg["beats_baseline"] is False


# ---- aggregate_significance (exact binomial) ------------------------------


def test_aggregate_significance_pooled_null_matches_baseline_rate():
    sig = aggregate_significance(153, CONFIG)
    assert sig["n"] == 400
    assert sig["k"] == 153
    assert sig["p0"] == pytest.approx(153 / 400)
    # At the null point itself, neither tail should reject at alpha=0.05.
    assert sig["decision"] == "indistinguishable_from_baseline"


def test_aggregate_significance_detects_a_large_beat():
    # A generous win (200/400 vs null 153/400) must read significant.
    sig = aggregate_significance(200, CONFIG)
    assert sig["decision"] == "significantly_above_baseline"
    assert sig["p_above"] <= CONFIG["aggregate_alpha"]


def test_aggregate_significance_detects_a_large_loss():
    sig = aggregate_significance(80, CONFIG)
    assert sig["decision"] == "significantly_below_baseline"
    assert sig["p_below"] <= CONFIG["aggregate_alpha"]


def test_aggregate_significance_a_small_beat_can_be_indeterminate():
    # One extra clear out of 400 cannot be statistically distinguished
    # from the null rate -- this is exactly the "noise as signal" trap
    # the pre-registration exists to avoid.
    sig = aggregate_significance(154, CONFIG)
    assert sig["decision"] == "indistinguishable_from_baseline"


def test_aggregate_significance_matches_hand_computed_binomial_tail():
    from math import comb
    n, p0, k = 400, 153 / 400, 200
    p_above = sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
                 for i in range(k, n + 1))
    sig = aggregate_significance(200, CONFIG)
    assert sig["p_above"] == pytest.approx(p_above, rel=1e-9)


# ---- classify_verdict: all four classes, incl. disguised interference ----


def test_verdict_supersedes_when_aggregate_wins_and_nothing_collapses():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 50), ("1-2", 45), ("1-3", 25), ("1-4", 55))]
    v = classify_verdict(deltas, CONFIG)
    assert v["verdict"] == "SUPERSEDES"
    assert v["reasons"] == []


def test_verdict_mixed_is_the_disguised_interference_case():
    """The exact scenario named in the spec: aggregate total climbs
    past 153 only because 1-1 is massively overshooting, while 1-2
    collapses more than the margin below its own baseline -- the
    interference failure wearing a disguise, not a genuine win."""
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 90), ("1-2", 10), ("1-3", 21), ("1-4", 51))]
    agg_check = 90 + 10 + 21 + 51
    assert agg_check > CONFIG["baseline_sum"]  # aggregate DOES beat baseline
    v = classify_verdict(deltas, CONFIG)
    assert v["verdict"] == "MIXED"
    assert any("1-2" in r and "disguise" in r for r in v["reasons"])
    assert v["aggregate"]["collapsed_levels"] == ["1-2"]


def test_verdict_failed_when_aggregate_does_not_beat_baseline():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 40), ("1-2", 35), ("1-3", 20), ("1-4", 50))]
    v = classify_verdict(deltas, CONFIG)
    assert v["verdict"] == "FAILED"
    assert v["reasons"]


def test_verdict_unusable_when_any_leg_errored():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 90), ("1-2", 90), ("1-3", 90))]
    deltas.append(level_delta("1-4", _leg(None, status="error",
                                           errors=["timeout"])))
    v = classify_verdict(deltas, CONFIG)
    assert v["verdict"] == "UNUSABLE"
    assert v["aggregate"]["shared_sum"] is None


def test_verdict_unusable_takes_precedence_over_a_would_be_supersede():
    """Even when every OTHER leg looks like a clean win, one broken leg
    must still void the whole verdict -- a broken harness is never a
    verdict, regardless of how good the rest looks."""
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 90), ("1-2", 90), ("1-3", 90), ("1-4", 90))]
    deltas[1] = level_delta("1-2", _leg(None, status="error"))
    v = classify_verdict(deltas, CONFIG)
    assert v["verdict"] == "UNUSABLE"


def test_verdict_all_four_classes_are_reachable():
    assert set(VERDICTS) == {"SUPERSEDES", "MIXED", "FAILED", "UNUSABLE"}


# ---- format_verdict_table / placeholder_table -----------------------------


def test_placeholder_table_shows_real_baseline_and_pending_everywhere_else():
    table = placeholder_table(CONFIG)
    assert "43" in table and "38" in table and "21" in table and "51" in table
    assert "153" in table
    assert table.count("PENDING") >= 4  # at least one per level
    assert "VERDICT: PENDING" in table


def test_format_verdict_table_marks_collapsed_levels():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 90), ("1-2", 10), ("1-3", 21), ("1-4", 51))]
    v = classify_verdict(deltas, CONFIG)
    table = format_verdict_table(deltas, v, CONFIG)
    assert "*collapsed*" in table
    assert "MIXED" in table


# ---- manifest mirroring ----------------------------------------------------


def test_manifest_mirrors_config_verbatim():
    man = build_manifest(CONFIG)
    assert man["config"] == CONFIG
    assert man["status"] == "pending"
    assert "level_deltas" not in man
    assert "verdict" not in man


def test_manifest_carries_verdict_rules_for_all_four_classes():
    man = build_manifest(CONFIG)
    for name in VERDICTS:
        assert name in man["verdict_rules"]
    assert "aggregate_test" in man["verdict_rules"]


def test_manifest_carries_the_verdict_and_significance_when_evaluated():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 50), ("1-2", 45), ("1-3", 25), ("1-4", 55))]
    man = build_manifest(CONFIG, level_deltas=deltas)
    assert man["status"] == "evaluated"
    assert man["verdict"]["verdict"] == "SUPERSEDES"
    assert man["aggregate_significance"]["k"] == 175


def test_manifest_omits_significance_when_a_leg_errored():
    deltas = [level_delta(lvl, _leg(v), CONFIG) for lvl, v in
             (("1-1", 50), ("1-2", 45), ("1-3", 25))]
    deltas.append(level_delta("1-4", _leg(None, status="error")))
    man = build_manifest(CONFIG, level_deltas=deltas)
    assert man["verdict"]["verdict"] == "UNUSABLE"
    assert "aggregate_significance" not in man


def test_write_manifest_roundtrips(tmp_path):
    man = build_manifest(CONFIG)
    out = tmp_path / "manifest.json"
    write_manifest(out, man)
    assert json.loads(out.read_text())["experiment"] == man["experiment"]


# ---- run() end-to-end, with an injected run_eval (no emulator) -----------


def test_run_end_to_end_writes_manifest_and_returns_zero_on_supersedes(
        tmp_path, monkeypatch):
    cfg = {**CONFIG,
          "receipt_log": str(tmp_path / "eval.jsonl"),
          "manifest_out": str(tmp_path / "manifest.json")}
    # Rates chosen so every level clears well above its own baseline.
    rates = {
        "1-1": {7: 0.60, 101: 0.60},
        "1-2": {7: 0.55, 101: 0.55},
        "1-3": {7: 0.35, 101: 0.35},
        "1-4": {7: 0.65, 101: 0.65},
    }

    def run_eval(cmd):
        game = cmd[cmd.index("--game") + 1]
        level = {"mario_1_1_backward": "1-1", "mario_1_2_consol2": "1-2",
                 "mario_1_3_online_v1": "1-3",
                 "mario_1_4_online_v1": "1-4"}[game]
        seed = int(cmd[cmd.index("--eval-seed") + 1])
        return _eval_json(n=50, clear_rate=rates[level][seed])

    monkeypatch.chdir(ROOT)
    code = run(cfg, checkpoint_dir=str(tmp_path / "heads"), run_eval=run_eval)
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["verdict"]["verdict"] == "SUPERSEDES"
    assert (tmp_path / "eval.jsonl").exists()


def test_run_returns_nonzero_on_unusable(tmp_path, monkeypatch):
    cfg = {**CONFIG,
          "receipt_log": str(tmp_path / "eval.jsonl"),
          "manifest_out": str(tmp_path / "manifest.json")}

    def run_eval(cmd):
        game = cmd[cmd.index("--game") + 1]
        if game == "mario_1_2_consol2":
            return {"status": "eval_timeout"}
        return _eval_json(n=50, clear_rate=0.5)

    monkeypatch.chdir(ROOT)
    code = run(cfg, checkpoint_dir=str(tmp_path / "heads"), run_eval=run_eval)
    assert code == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["verdict"]["verdict"] == "UNUSABLE"


def test_run_never_invokes_a_real_subprocess(tmp_path):
    """Structural guarantee that run() with an injected run_eval touches
    subprocess.run zero times -- the emulator-safety property this
    entire test module depends on."""
    import scripts.eval_shared_substrate as mod

    calls = []
    real_subprocess_run = mod.subprocess.run

    def poison(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("subprocess.run must not be called when "
                             "run_eval is injected")

    cfg = {**CONFIG,
          "receipt_log": str(tmp_path / "eval.jsonl"),
          "manifest_out": str(tmp_path / "manifest.json")}
    mod.subprocess.run = poison
    try:
        run(cfg, checkpoint_dir="/fake/dir",
            run_eval=lambda cmd: _eval_json(n=50, clear_rate=0.5))
    finally:
        mod.subprocess.run = real_subprocess_run
    assert calls == []


# ---- --dry-run subprocess (live; config parse + checkpoint load only) ----


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_dry_run_passes_live():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_shared_substrate.py"),
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=150)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
    # "FAILED" legitimately appears as one of the four verdict-class
    # names in the registered-synthetic-cases check output; only a
    # "[dry-run] FAIL " report line is an actual failed check.
    assert "[dry-run] FAIL " not in proc.stdout
    # All four baseline checkpoints must be reported, with their
    # inferred widths.
    assert "h64/trunk32" in proc.stdout
    assert "h256/trunk64" in proc.stdout
    assert "share one observation width" in proc.stdout
    assert "disguised-interference" in proc.stdout
    assert "PENDING" in proc.stdout
    assert "eval_shared_substrate.py --checkpoint-dir" in proc.stdout


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_dry_run_never_calls_eval_game_subprocess(monkeypatch):
    """--dry-run must not itself shell out to eval_game.py (which would
    step the emulator) -- only assemble argv and check paths."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "import scripts.eval_shared_substrate as m; "
         "m.subprocess.run = lambda *a, **k: (_ for _ in ()).throw("
         "AssertionError('subprocess.run called during --dry-run')); "
         "sys.exit(m.main())",
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=150)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AssertionError" not in proc.stderr


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_dry_run_writes_the_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, json; sys.path.insert(0, '.'); "
         "import scripts.eval_shared_substrate as m; "
         "cfg = dict(m.CONFIG); "
         f"cfg['manifest_out'] = {str(manifest_path)!r}; "
         "sys.exit(m.dry_run(cfg))"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=150)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    man = json.loads(manifest_path.read_text())
    assert man["experiment"] == "shared_substrate_trunk_plus_heads"
    assert man["status"] == "pending"
