"""Consolidation round 2 (SMB 1-2) — config schema, probe assembly, and
gate/kill logic of scripts/run_consol2.py.

Everything here runs on STUBBED metric/probe streams — synthetic probe
summaries and synthetic eval_game JSON. No emulator, no rollout, no
training subprocess. The one live-ish test is the --dry-run subprocess,
which is explicitly allowed to parse config, load the source checkpoint,
and assemble probe commands (none of which is a rollout).

Pre-registration under test (mirrors of the in-file CONFIG):
* single phase, 30M env-step budget, NO advance gates (rate-building);
* dual-protocol honest probes every 5M env-steps on the SAME checkpoint:
  (a) probe-protocol   — parallel, per-episode RNG (5 workers), the
      protocol consolidation-1 scored 14.4% pooled on;
  (b) sequential shared-stream — serial, shared-stream RNG, the protocol
      the same checkpoints scored 0.00-0.04 on (final_eval*.json);
* kills: competence floor (median max-x < 150 on EITHER protocol,
  immediate), and probe-rate collapse (probe-protocol clear rate < 5%
  for 2 consecutive OK probes after having exceeded 10% at least once).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_consol2 import (  # noqa: E402
    CONFIG, ProbeMonitor, build_manifest, build_probe_command,
    probe_boundaries, protocol_gap,
)
from scripts.run_online_campaign import probe_summary  # noqa: E402

SPI = 92_160  # 1536 rollout_steps x 60 envs — unchanged from the campaign


def _profile() -> dict:
    return yaml.safe_load((ROOT / CONFIG["base_profile"]).read_text())


def _summ(*, status="ok", n=30, median=1500.0, clear=0.15,
          survival=0.3) -> dict:
    return {"status": status, "n_episodes": n, "median_max_x": median,
            "bottleneck_survival": survival, "clear_rate": clear}


# ---- config: strict schema -------------------------------------------


def test_config_passes_strict_schema():
    from src.training.config_schema import check_profile
    prof = _profile()
    assert check_profile(prof, strict=False) == []
    check_profile(prof, strict=True)  # must not raise


def test_config_name_isolates_checkpoint_dir_from_online_v2():
    from src.training.profile_utils import profile_slug
    prof = _profile()
    assert profile_slug(prof["name"]) == "mario_1_2_consol2"
    assert profile_slug(prof["name"]) != "mario_1_2_online_v2"


# ---- config: the pre-registered consol2 deltas ------------------------


def test_self_anchor_points_at_the_continued_checkpoint():
    """Drift insurance: the frozen KL prior IS the checkpoint the run
    continues from (consolidation-1's product), so kl_anchor_div reads
    drift from the 14.4% policy itself — without freezing learning
    (loss coef 0.1, not 1.0)."""
    prof = _profile()
    rl = prof["reinforce"]
    assert rl["kl_anchor_checkpoint"] == CONFIG["source_checkpoint"]
    assert rl["kl_anchor_loss_coef"] == 0.1


def test_reward_level_kl_channel_is_zeroed():
    """Only the loss-level tether is armed; the reward-level beta channel
    (attempt-2's failed containment) stays at exactly 0."""
    rl = _profile()["reinforce"]
    assert rl["kl_beta_start"] == 0.0
    assert rl["kl_beta_end"] == 0.0


def test_actor_unfrozen_from_step_zero():
    # Continuing a trained actor+critic pair: no critic-warmup freeze.
    assert _profile()["reinforce"]["actor_freeze_steps"] == 0


def test_house_sticky_and_entropy_match_consolidation1_regime():
    rl = _profile()["reinforce"]
    assert rl["sticky_action_prob"] == 0.25
    assert rl["entropy_coef"] == 0.003
    # The trainer's entropy_floor knob is a policy-entropy (nats)
    # controller, not a coef floor — it stays off; the coef floor is
    # realized by the runner never rewriting entropy_coef (single phase).
    assert rl["entropy_floor"] == 0.0


def test_sil_enabled_with_campaign_bc_coef():
    sil = _profile()["reinforce"]["sil"]
    assert sil["enabled"] is True
    assert sil["bc_coef"] == 0.5
    assert sil["buffer_size"] == 64


def test_entrance_pinned_restarts():
    bwd = _profile()["reinforce"]["backward_curriculum"]
    assert bwd["pin_entrance"] is True
    assert bwd["count_truncations"] is True


def test_gamma_gae_as_campaign_and_pbrs_invariant():
    rl = _profile()["reinforce"]
    assert rl["gamma"] == 0.99
    assert rl["gae_lambda"] == 0.95
    assert rl["wavefront_reward"]["gamma"] == rl["gamma"]


def test_batch_shape_unchanged_from_campaign():
    rl = _profile()["reinforce"]
    assert rl["rollout_steps"] == 1536
    assert rl["num_envs"] == 60
    assert rl["rollout_steps"] * rl["num_envs"] == SPI


def test_adversary_stays_inert():
    # No `mode` key => the adversary block is inert (campaign semantics).
    assert "mode" not in _profile()["reinforce"]["adversary"]


# ---- CONFIG: pre-registered thresholds --------------------------------


def test_preregistered_budget_and_probe_cadence():
    assert CONFIG["budget_env_steps"] == 30_000_000
    assert CONFIG["probe_every_env_steps"] == 5_000_000
    assert CONFIG["probe_episodes"] == 30
    assert CONFIG["probe_sticky"] == 0.25
    assert CONFIG["probe_jitter"] == 16


def test_preregistered_kills_and_no_gates():
    assert CONFIG["kill_median_floor"] == 150.0
    assert CONFIG["kill_collapse_rate"] == 0.05
    assert CONFIG["kill_collapse_arm_rate"] == 0.10
    assert CONFIG["kill_collapse_consecutive"] == 2
    # Rate-building run: no advance gates exist to pin.
    assert not any(k.startswith("gate_") for k in CONFIG)


def test_source_checkpoint_is_pinned_by_sha256_and_iter():
    assert CONFIG["source_checkpoint"] == (
        "checkpoints/_preserved/online_v2_FINAL_consolidated.pt")
    sha = CONFIG["source_checkpoint_sha256"]
    assert len(sha) == 64 and int(sha, 16) >= 0  # well-formed hex digest
    assert CONFIG["source_iter"] == 910


# ---- probe scheduling --------------------------------------------------


def test_probe_boundaries_six_probes_over_30m():
    start = CONFIG["source_iter"] * SPI  # 83,865,600
    bounds = probe_boundaries(start, CONFIG["budget_env_steps"],
                              CONFIG["probe_every_env_steps"])
    assert len(bounds) == 6
    assert bounds[0] == start + 5_000_000
    assert bounds[-1] == start + 30_000_000
    assert all(b2 - b1 == 5_000_000 for b1, b2 in zip(bounds, bounds[1:]))


def test_probe_boundaries_partial_budget():
    assert probe_boundaries(0, 12_000_000, 5_000_000) == [
        5_000_000, 10_000_000]


# ---- probe command assembly (both protocols) ---------------------------


def test_probe_protocol_command_is_parallel_per_episode():
    cmd = build_probe_command(
        checkpoint=Path("/x/ckpt.pt"), protocol="probe", eval_seed=1234)
    assert "--eval-rng" in cmd and cmd[cmd.index("--eval-rng") + 1] == \
        "per-episode"
    assert cmd[cmd.index("--eval-workers") + 1] == str(
        CONFIG["probe_eval_workers"])
    assert cmd[cmd.index("--sticky-prob") + 1] == "0.25"
    assert cmd[cmd.index("--start-jitter") + 1] == "16"
    assert cmd[cmd.index("--episodes") + 1] == "30"
    assert cmd[cmd.index("--eval-seed") + 1] == "1234"
    assert "--level-clear" in cmd and "--sequential" in cmd
    assert "/x/ckpt.pt" in cmd


def test_sequential_protocol_command_is_serial_shared_stream():
    cmd = build_probe_command(
        checkpoint=Path("/x/ckpt.pt"), protocol="sequential", eval_seed=7)
    assert cmd[cmd.index("--eval-rng") + 1] == "shared-stream"
    assert cmd[cmd.index("--eval-workers") + 1] == "1"
    # Same honest noise profile — the ONLY difference is RNG plumbing.
    assert cmd[cmd.index("--sticky-prob") + 1] == "0.25"
    assert cmd[cmd.index("--start-jitter") + 1] == "16"
    assert cmd[cmd.index("--episodes") + 1] == "30"


def test_both_protocols_probe_the_same_checkpoint_and_start_state():
    a = build_probe_command(checkpoint=Path("/x/c.pt"), protocol="probe",
                            eval_seed=1)
    b = build_probe_command(checkpoint=Path("/x/c.pt"),
                            protocol="sequential", eval_seed=1)
    ck = a[a.index("--checkpoint") + 1]
    assert ck == b[b.index("--checkpoint") + 1]
    ss = a[a.index("--start-state") + 1]
    assert ss == b[b.index("--start-state") + 1]
    assert ss.endswith("stage_03.state")  # the 1-2 entrance, cold


def test_unknown_protocol_rejected():
    with pytest.raises(ValueError):
        build_probe_command(checkpoint=Path("/x/c.pt"), protocol="warm",
                            eval_seed=1)


# ---- protocol gap -------------------------------------------------------


def test_protocol_gap_on_consolidation1_shaped_numbers():
    # Consolidation-1's measured shape: 0.20 probe-protocol vs 0.04
    # shared-stream on the same checkpoint.
    gap = protocol_gap(_summ(clear=0.20), _summ(clear=0.04))
    assert gap["clear_rate_gap"] == pytest.approx(0.16)
    assert gap["probe_clear_rate"] == pytest.approx(0.20)
    assert gap["sequential_clear_rate"] == pytest.approx(0.04)
    assert gap["status"] == "ok"


def test_protocol_gap_degrades_on_failed_leg():
    gap = protocol_gap(_summ(clear=0.2), _summ(status="probe_failed"))
    assert gap["clear_rate_gap"] is None
    assert gap["status"] == "probe_failed"


# ---- ProbeMonitor: competence floor ------------------------------------


def _monitor() -> ProbeMonitor:
    return ProbeMonitor(CONFIG)


def test_healthy_stream_never_kills():
    m = _monitor()
    for clear in (0.12, 0.15, 0.08, 0.14, 0.2, 0.18):
        assert m.observe(_summ(clear=clear), _summ(clear=clear / 4)) is None


def test_competence_floor_kills_on_probe_protocol():
    m = _monitor()
    reason = m.observe(_summ(median=140.0), _summ())
    assert reason is not None and "floor" in reason.lower()
    assert "probe" in reason


def test_competence_floor_kills_on_sequential_protocol():
    m = _monitor()
    reason = m.observe(_summ(), _summ(median=90.0))
    assert reason is not None and "floor" in reason.lower()
    assert "sequential" in reason


def test_floor_ignores_failed_or_empty_probes():
    m = _monitor()
    assert m.observe(_summ(status="probe_timeout", median=0.0, n=0),
                     _summ()) is None
    assert m.observe(_summ(n=0, median=0.0), _summ()) is None


# ---- ProbeMonitor: probe-rate collapse ---------------------------------


def test_collapse_kills_after_two_consecutive_lows_once_armed():
    m = _monitor()
    assert m.observe(_summ(clear=0.144), _summ(clear=0.0)) is None  # arms
    assert m.observe(_summ(clear=0.03), _summ(clear=0.0)) is None   # low 1
    reason = m.observe(_summ(clear=0.02), _summ(clear=0.0))         # low 2
    assert reason is not None and "collapse" in reason.lower()


def test_collapse_never_fires_before_arming():
    m = _monitor()
    for _ in range(5):
        assert m.observe(_summ(clear=0.0), _summ(clear=0.0)) is None


def test_recovery_between_lows_resets_the_streak():
    m = _monitor()
    assert m.observe(_summ(clear=0.15), _summ(clear=0.0)) is None  # arms
    assert m.observe(_summ(clear=0.03), _summ(clear=0.0)) is None  # low 1
    assert m.observe(_summ(clear=0.07), _summ(clear=0.0)) is None  # resets
    assert m.observe(_summ(clear=0.04), _summ(clear=0.0)) is None  # low 1
    reason = m.observe(_summ(clear=0.04), _summ(clear=0.0))        # low 2
    assert reason is not None and "collapse" in reason.lower()


def test_failed_probe_leaves_collapse_state_unchanged():
    m = _monitor()
    assert m.observe(_summ(clear=0.15), _summ(clear=0.0)) is None  # arms
    assert m.observe(_summ(clear=0.03), _summ(clear=0.0)) is None  # low 1
    assert m.observe(_summ(status="probe_failed", n=0, median=0.0,
                           clear=0.0), _summ(clear=0.0)) is None   # no-op
    reason = m.observe(_summ(clear=0.02), _summ(clear=0.0))        # low 2
    assert reason is not None and "collapse" in reason.lower()


def test_sequential_rate_does_not_arm_or_trip_collapse():
    """The collapse kill is probe-protocol only (pre-registered): the
    sequential leg sat at 0.00-0.04 on consolidation-1's own healthy
    checkpoints, so it must not be able to arm or trip this kill."""
    m = _monitor()
    assert m.observe(_summ(clear=0.06), _summ(clear=0.5)) is None  # seq high
    assert m.observe(_summ(clear=0.03), _summ(clear=0.0)) is None
    assert m.observe(_summ(clear=0.03), _summ(clear=0.0)) is None  # unarmed


# ---- probe_summary interop (reused verbatim from the campaign) ----------


def test_probe_summary_reuse_matches_campaign_semantics():
    s = probe_summary(
        {"status": "ok", "n_episodes": 4,
         "max_gx_per_episode": [100, 2650, 2700, 900], "clear_rate": 0.25},
        survival_x=CONFIG["bottleneck_survival_x"])
    assert s["median_max_x"] == pytest.approx((900 + 2650) / 2)
    assert s["bottleneck_survival"] == pytest.approx(0.5)
    assert s["clear_rate"] == pytest.approx(0.25)


# ---- manifest mirror -----------------------------------------------------


def test_manifest_mirrors_config_and_cites_evidence():
    man = build_manifest({"name": "Mario 1-2 consol2"})
    assert man["config"] == CONFIG
    assert man["campaign"] == "smb_1_2_consol2"
    ev = man["preregistration_evidence"]
    # Consolidation-1 receipts must ride in the manifest verbatim.
    assert ev["consolidation1_pooled_probe_clears"] == "13/90 (14.4%)"
    assert "runs/online_1_2/campaign.jsonl" in ev["probe_receipts"]
    assert any("final_eval" in p for p in ev["sequential_receipts"])


# ---- dry-run (live; config parse + checkpoint load + probe assembly) ----


@pytest.mark.slow
def test_dry_run_passes_live():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_consol2.py"),
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
    assert "FAIL" not in proc.stdout
