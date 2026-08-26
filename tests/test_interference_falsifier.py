"""Generalist-interference falsifier (SMB 1-1 x 1-2) —
scripts/interference_falsifier.py.

Everything here runs on SYNTHETIC arrays and stubbed probe summaries —
no emulator, no rollout, no training subprocess. The one live-ish test
is the --dry-run subprocess, which is explicitly allowed to parse
configs, torch.load both specialist checkpoints, read receipt files and
assemble eval commands (none of which is a rollout).

Pre-registration under test (mirrors of the in-file CONFIG):
* step 1 — success-trajectory collection: BOTH specialists rolled out
  SAMPLED (T=1.0) under the house noise (sticky 0.25, jitter 16,
  per-episode RNG), each on ITS OWN level, 300 episodes each, keeping
  STRICT successes only (episode_success — max_gx never substitutes),
  pooled 50/50 by state-action pairs;
* step 2 — joint BC: a FRESH ~200k-param TilePolicyNetwork (h256 /
  trunk64 — the same class the eval stack loads), 50 epochs
  cross-entropy, lr 3e-4 with step decay, saved to
  runs/interference/joint.pt in the standard net_state_dict payload;
* step 3 — the FOUR-leg gate: each level is evaluated twice under one
  identical protocol — once on its own SPECIALIST (the positive
  control) and once on the joint net — 100 episodes each, cold argmax,
  sticky 0.25, jitter 16, per-episode RNG, on a gate seed DISJOINT from
  the collection seed. The hold threshold is 50% of the MEASURED
  control, never a constant transcribed from prose; each leg is decided
  by an exact one-sided binomial test at alpha, so a leg that n cannot
  discriminate returns `indeterminate` and the run is `inconclusive`
  rather than a coin-flip verdict. n=100 is not a round number: it is
  the smallest count clearing the pre-registered discrimination floor
  on BOTH legs, which the dry-run recomputes and enforces.

The verdict CALIBRATES the generalist plan (Level-ID token vs capacity)
and never gates the 1-3 campaign — asserted on the manifest.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.interference_falsifier import (  # noqa: E402
    CONFIG, _run_eval, assert_seed_independence, balance_fifty_fifty,
    binom_p_at_least, binom_p_at_most, build_collection_reference_command,
    build_gate_command, build_manifest, collection_plan,
    discrimination_power, interference_verdict, joint_bc, leg_decision,
    reference_prior, steps_from, step3_gate, success_pairs_from_episodes,
    verify_reference_receipts, write_joint_checkpoint, write_pooled_pairs,
    write_success_pairs,
)


# ---- fixtures -----------------------------------------------------------


def _episode(length, *, success, max_gx=1000, feature_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "episode_index": seed,
        "obs": rng.integers(-4, 4, size=(length, feature_dim)).astype(np.int8),
        "actions": rng.integers(0, 4, size=(length,)).astype(np.int64),
        "max_gx": int(max_gx),
        "episode_success": bool(success),
        "length": int(length),
    }


def _pairs(n, feature_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.integers(-4, 4, size=(n, feature_dim)).astype(np.int8)
    act = rng.integers(0, 4, size=(n,)).astype(np.int64)
    return obs, act


def _summ(*, status="ok", n=None, strict=0.5):
    n = CONFIG["gate_episodes"] if n is None else n
    return {"status": status, "n_episodes": n, "median_max_x": 2000.0,
            "clear_rate": strict, "clear_rate_strict": strict,
            "clear_rate_chain": strict}


def _verdict(joint_11, joint_12, *, control_11=0.48, control_12=0.40, **kw):
    """A verdict over the four legs. Controls default well clear of the
    collapse floor so the joint rates are what move the classification."""
    return interference_verdict(
        {"1-1": _summ(strict=joint_11, **kw.get("kw_11", {})),
         "1-2": _summ(strict=joint_12, **kw.get("kw_12", {}))},
        {"1-1": _summ(strict=control_11, **kw.get("kw_c11", {})),
         "1-2": _summ(strict=control_12, **kw.get("kw_c12", {}))})


# ---- step 1: strict-success filter + pooled balance ----------------------


def test_success_pairs_keep_strict_only():
    """STRICT successes only: a high-max_gx episode without
    episode_success is DROPPED (unlike night2's OR-keep) — the falsifier
    distills what the specialists can actually finish, not what they can
    reach."""
    eps = [
        _episode(9, success=True, max_gx=4000, seed=1),
        _episode(7, success=False, max_gx=3300, seed=2),  # reached, no clear
        _episode(5, success=True, max_gx=3500, seed=3),
        _episode(4, success=False, max_gx=500, seed=4),
    ]
    obs, act, traj_len, kept = success_pairs_from_episodes(eps)
    assert [e["episode_index"] for e in kept] == [1, 3]
    assert traj_len.tolist() == [9, 5]
    assert obs.shape == (14, 5)
    assert act.shape == (14,) and act.dtype == np.int64
    np.testing.assert_array_equal(obs[:9], eps[0]["obs"])
    np.testing.assert_array_equal(act[9:], eps[2]["actions"])


def test_success_pairs_empty_when_no_strict_clears():
    obs, act, traj_len, kept = success_pairs_from_episodes(
        [_episode(6, success=False, max_gx=9000, seed=1)])
    assert kept == [] and obs.shape[0] == 0 and act.shape[0] == 0


def test_balance_trims_larger_side_to_smaller_pair_count():
    o1, a1 = _pairs(30, seed=1)
    o2, a2 = _pairs(11, seed=2)
    pooled = balance_fifty_fifty({"1-1": (o1, a1), "1-2": (o2, a2)})
    # 50/50 BY PAIRS: both sides trimmed to min(30, 11) = 11, in
    # collection order (leading pairs kept, trailing dropped).
    assert pooled["counts"] == {"1-1": 11, "1-2": 11}
    assert pooled["obs"].shape == (22, 5)
    np.testing.assert_array_equal(pooled["obs"][:11], o1[:11])
    np.testing.assert_array_equal(pooled["obs"][11:], o2)
    np.testing.assert_array_equal(pooled["act"][:11], a1[:11])
    # level_ids index into CONFIG["levels"] order.
    assert pooled["level_ids"].tolist() == [0] * 11 + [1] * 11
    assert pooled["levels"] == list(CONFIG["levels"])


def test_balance_refuses_an_empty_side():
    o1, a1 = _pairs(8, seed=1)
    with pytest.raises(RuntimeError):
        balance_fifty_fifty({"1-1": (o1, a1),
                             "1-2": (np.zeros((0, 5), dtype=np.int8),
                                     np.zeros((0,), dtype=np.int64))})


def test_success_pairs_npz_roundtrip(tmp_path):
    o, a = _pairs(11, seed=3)
    out = tmp_path / "success_1_1.npz"
    write_success_pairs(
        out, obs=o, act=a, traj_len=np.array([7, 4]),
        label_max_gx=np.array([3400, 800]),
        label_episode_success=np.array([True, True]),
        provenance={"level": "1-1", "mode": "collection"})
    d = np.load(out)
    assert int(d["n"]) == 1  # house demo convention: distill-consumable
    np.testing.assert_array_equal(d["obs_0"], o)
    np.testing.assert_array_equal(d["act_0"], a)
    assert d["traj_len"].tolist() == [7, 4]
    assert d["label_episode_success"].tolist() == [True, True]
    prov = json.loads(str(d["provenance"].item()))
    assert prov["level"] == "1-1"


def test_pooled_pairs_npz_roundtrip(tmp_path):
    o1, a1 = _pairs(6, seed=1)
    o2, a2 = _pairs(9, seed=2)
    pooled = balance_fifty_fifty({"1-1": (o1, a1), "1-2": (o2, a2)})
    out = tmp_path / "pooled_pairs.npz"
    write_pooled_pairs(out, pooled, provenance={"step": 1})
    d = np.load(out)
    assert d["obs"].shape == (12, 5)
    assert d["level_ids"].tolist() == [0] * 6 + [1] * 6
    assert json.loads(str(d["levels"].item())) == list(CONFIG["levels"])
    assert json.loads(str(d["counts"].item())) == {"1-1": 6, "1-2": 6}


# ---- step 2: joint BC on a FRESH net -------------------------------------


def _bc(obs, act, **overrides):
    kw = dict(num_actions=4, feature_dim=obs.shape[1], hidden_dim=8,
              trunk_dim=6, epochs=3, lr=1e-2, batch_size=32,
              lr_step_epochs=2, lr_gamma=0.5, seed=0)
    kw.update(overrides)
    return joint_bc(obs, act, **kw)


def test_joint_bc_trains_a_fresh_net_of_the_registered_shape():
    obs, act = _pairs(200, feature_dim=12, seed=5)
    sd, stats = _bc(obs, act)
    # Fresh TilePolicyNetwork at the requested widths — the same class +
    # key set the eval stack shape-infers from.
    assert tuple(sd["fc1.weight"].shape) == (8, 12)
    assert tuple(sd["fc2.weight"].shape) == (6, 8)
    assert tuple(sd["actor.weight"].shape) == (4, 6)
    assert "critic.weight" in sd
    assert stats["n_pairs"] == 200
    assert stats["lr_schedule"] == "step_decay"
    assert stats["param_count"] == sum(int(v.numel()) for v in sd.values())


def test_joint_bc_is_deterministic():
    obs, act = _pairs(150, seed=6)
    a, _ = _bc(obs, act, seed=11)
    b, _ = _bc(obs, act, seed=11)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_joint_bc_reduces_loss_on_learnable_labels():
    obs, _ = _pairs(200, seed=7)
    act = np.zeros(200, dtype=np.int64)  # constant label: learnable
    _, stats = _bc(obs, act, epochs=6, lr=5e-2)
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["final_acc"] > 0.9


def test_joint_bc_refuses_zero_pairs():
    with pytest.raises(ValueError):
        _bc(np.zeros((0, 5), dtype=np.int8), np.zeros((0,), dtype=np.int64))


def test_registered_joint_net_is_about_200k_params():
    from src.models.tile_policy import TilePolicyNetwork
    net = TilePolicyNetwork(num_actions=6, feature_dim=712,
                            hidden_dim=CONFIG["bc_hidden_dim"],
                            trunk_dim=CONFIG["bc_trunk_dim"])
    assert 190_000 <= net.num_params <= 210_000


def test_joint_checkpoint_standard_payload_and_shape_infer(tmp_path):
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    obs, act = _pairs(100, feature_dim=12, seed=8)
    sd, stats = _bc(obs, act)
    out = tmp_path / "joint.pt"
    write_joint_checkpoint(out, sd, stats=stats, provenance={"step": 2})
    payload = torch.load(str(out), map_location="cpu", weights_only=False)
    # Standard payload: net_state_dict + iter, cpu tensors — exactly what
    # eval_game's checkpoint loader + shape inference consume.
    assert payload["iter"] == 0
    assert set(payload["net_state_dict"]) == set(sd)
    for v in payload["net_state_dict"].values():
        assert v.device.type == "cpu"
    assert payload["provenance"] == "interference_joint_bc"
    net, is_recurrent = build_tile_policy_from_checkpoint(
        payload, num_actions=4, feature_dim=12)
    assert is_recurrent is False
    missing, unexpected = net.load_state_dict(payload["net_state_dict"],
                                              strict=True)
    assert not missing and not unexpected
    assert net.hidden_dim == 8 and net.trunk_dim == 6


# ---- DEFECT 1: the specialist reference must be receipt-backed -----------


def test_no_unsourced_specialist_rate_constant_in_config():
    """REGRESSION (defect 1). The old CONFIG carried
    specialist_strict_rate=0.76 for 1-1 — a number that appears nowhere
    except as prose, and never attached to a checkpoint filename. No
    hardcoded specialist rate may drive the gate again: the threshold
    comes from the control leg measured in the same session, and the
    only rates in CONFIG are receipt rows re-verifiable on disk."""
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        assert "specialist_strict_rate" not in spec, level
    assert "hold_fraction_of_specialist" not in CONFIG
    assert CONFIG["hold_fraction_of_control"] == 0.50
    # 0.76 may survive ONLY inside the rejected-candidate note that
    # explains where it really came from — never as a live value.
    scrubbed = json.loads(json.dumps(CONFIG))
    for level in CONFIG["levels"]:
        scrubbed["specialists"][level].pop("rejected_candidates")
    assert "0.76" not in json.dumps(scrubbed)


@pytest.mark.parametrize("level", ["1-1", "1-2"])
def test_reference_receipts_reverify_on_disk(level):
    """Every reference row CONFIG cites is re-read from its own file and
    must match on checkpoint, episode count and clear rate — and the
    receipted checkpoint must be byte-identical (sha256) to the
    specialist this falsifier actually rolls out. This is the check the
    0.76 could never have passed."""
    got = verify_reference_receipts(level)
    prior = reference_prior(level)
    assert got["clears"] == prior["clears"]
    assert got["episodes"] == prior["episodes"]
    assert got["episodes"] >= 100, "a prior under 100 episodes is not a prior"
    assert 0.0 < prior["rate"] < 1.0
    assert prior["rate"] == pytest.approx(got["clears"] / got["episodes"])
    # Each row names the file it came from.
    for row in got["rows"]:
        assert Path(ROOT / row["receipt"]).exists()
        assert row["same_weights_as_specialist"] is True


def test_one_one_reference_is_the_honest_greedy_lineage():
    """The 1-1 specialist is the checkpoint whose honest receipts exist
    on the exact entrance this falsifier evals from — pooled 56/120 =
    0.467 over three seeds — not the unreceipted seed-3 file."""
    spec = CONFIG["specialists"]["1-1"]
    assert spec["checkpoint"] == \
        "checkpoints/_preserved/backward_1_1_best_honest_greedy_047.pt"
    prior = reference_prior("1-1")
    assert (prior["clears"], prior["episodes"]) == (56, 120)
    assert prior["rate"] == pytest.approx(0.4667, abs=5e-4)


def test_one_two_reference_is_the_consol2_peak_pair():
    prior = reference_prior("1-2")
    assert (prior["clears"], prior["episodes"]) == (38, 100)
    assert prior["rate"] == pytest.approx(0.38)


def test_rejected_1_1_candidate_records_why():
    """The previously-credited checkpoint stays in the record with the
    reason it was rejected, so the misattribution cannot be re-made."""
    rej = CONFIG["specialists"]["1-1"]["rejected_candidates"]
    entry = next(r for r in rej
                 if r["checkpoint"].endswith("backward_1_1_seed3_iter140.pt"))
    reason = entry["reason"].lower()
    assert "no eval receipt" in reason
    assert "tile_gate_v2" in reason        # where the 0.76 actually lives
    assert "0.330" in entry["reason"]      # its own embedded telemetry


def test_reference_protocol_deltas_are_declared_not_denied():
    """DEFECT 3. The prior receipts were NOT measured on the gate
    protocol (shared-stream RNG, 1 worker, no --sequential). CONFIG must
    say so out loud — the honest comparison is the measured control, and
    the prior is only a drift band on it."""
    for level in CONFIG["levels"]:
        deltas = CONFIG["specialists"][level]["reference_protocol_deltas"]
        assert deltas, level
        joined = " ".join(deltas).lower()
        assert "shared-stream" in joined and "worker" in joined
        assert CONFIG["specialists"][level]["reference_role"] == \
            "prior_band_only"


# ---- DEFECT 5: the gate seed must not reuse the collection seed ----------


def test_gate_seed_is_disjoint_from_collection_seed():
    assert CONFIG["gate_eval_seed"] != CONFIG["collect_seed"]
    assert_seed_independence()  # does not raise


def test_seed_collision_is_refused_at_assembly(monkeypatch):
    """A gate that replays the collection episodes grades the joint net
    on exactly the noise realizations its training data came from."""
    monkeypatch.setitem(CONFIG, "gate_eval_seed", CONFIG["collect_seed"])
    with pytest.raises(RuntimeError, match="collection seed"):
        assert_seed_independence()
    with pytest.raises(RuntimeError, match="collection seed"):
        build_gate_command("1-1", checkpoint=ROOT / CONFIG["joint_out"],
                           require_paths=False)


# ---- exact binomial machinery (DEFECT 2) ---------------------------------


def test_binomial_tails_match_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for n, p in ((50, 0.19), (50, 0.2333), (30, 0.5), (12, 0.03)):
        for k in (0, 1, 5, 9, n // 2, n):
            assert binom_p_at_most(k, n, p) == pytest.approx(
                float(scipy_stats.binom.cdf(k, n, p)), abs=1e-12)
            assert binom_p_at_least(k, n, p) == pytest.approx(
                float(scipy_stats.binom.sf(k - 1, n, p)), abs=1e-12)


def test_leg_decision_is_three_way_not_a_point_threshold():
    """A rate a hair over the threshold is NOT evidence of holding at
    n=50 — it is indeterminate. This is the defect-2 fix: the old rule
    called a coin flip either way."""
    p0, n, alpha = 0.2333, 50, 0.05
    assert leg_decision(24, n, p0, alpha)["decision"] == "holds"
    assert leg_decision(2, n, p0, alpha)["decision"] == "fails"
    just_over = leg_decision(13, n, p0, alpha)   # 0.26 vs threshold 0.233
    assert just_over["decision"] == "indeterminate"
    assert just_over["p_above"] > alpha and just_over["p_below"] > alpha


def test_leg_decision_reports_both_tails_and_inputs():
    d = leg_decision(19, 50, 0.19, 0.05)
    assert d["clears"] == 19 and d["n"] == 50
    assert d["p0"] == 0.19 and d["alpha"] == 0.05
    assert 0.0 <= d["p_above"] <= 1.0 and 0.0 <= d["p_below"] <= 1.0
    assert d["p_above"] + d["p_below"] >= 1.0  # the tails overlap at k


@pytest.mark.parametrize("level", ["1-1", "1-2"])
def test_registered_episode_count_can_actually_discriminate(level):
    """THE defect-2 acceptance test. At the registered n, a perfect clone
    of the specialist must be classified `holds` with at least the
    pre-registered power, and a net at half the threshold must be
    classified `fails` with at least it too. The OLD design failed this
    outright: with a 0.38 threshold against a ~0.36 truth, a clone held
    only ~44% of the time."""
    prior = reference_prior(level)
    pw = discrimination_power(
        CONFIG["gate_episodes"], prior["rate"],
        frac=CONFIG["hold_fraction_of_control"],
        alpha=CONFIG["control_alpha"])
    floor = CONFIG["min_clone_hold_power"]
    assert pw["clone_holds"] >= floor, (level, pw)
    assert pw["interfered_fails"] >= floor, (level, pw)
    # And the registered n is what made that true: at the old n=50 the
    # 1-2 leg could convict an interfered net less than half the time.
    at_50 = discrimination_power(
        50, prior["rate"], frac=CONFIG["hold_fraction_of_control"],
        alpha=CONFIG["control_alpha"])
    assert at_50["interfered_fails"] < floor, level


def test_old_point_threshold_would_have_been_a_coin_flip():
    """Documents the defect being repaired: 19+/50 required against a
    lineage that truly runs ~0.36 is barely better than chance, which is
    why a hardcoded reference could manufacture `partial_interference`
    out of pure noise."""
    assert binom_p_at_least(19, 50, 0.36) == pytest.approx(0.436, abs=0.01)


# ---- step 3: the four-leg verdict ----------------------------------------


def test_verdict_compatible_when_both_legs_hold_against_measured_control():
    v = _verdict(0.48, 0.40, control_11=0.48, control_12=0.40)
    assert v["verdict"] == "compatible"
    assert v["holds"] == {"1-1": True, "1-2": True}
    # The thresholds come from the MEASURED control, not a constant.
    assert v["hold_thresholds"] == {"1-1": 0.24, "1-2": 0.20}


def test_verdict_threshold_follows_the_control_not_a_constant():
    """DEFECT 1+4. Identical joint rates, different measured control ->
    different answer. Under the old design the answer could not move,
    because the denominator was typed in by hand."""
    strong_control = _verdict(0.20, 0.20, control_11=0.60, control_12=0.60)
    weak_control = _verdict(0.20, 0.20, control_11=0.24, control_12=0.24)
    assert strong_control["hold_thresholds"]["1-1"] == 0.30
    assert weak_control["hold_thresholds"]["1-1"] == 0.12
    assert strong_control["verdict"] != weak_control["verdict"]
    assert weak_control["verdict"] == "compatible"
    assert strong_control["verdict"] == "degraded"


def test_verdict_severe_when_both_collapse():
    v = _verdict(0.02, 0.02)
    assert v["verdict"] == "severe_interference"


def test_verdict_partial_when_exactly_one_holds():
    v = _verdict(0.48, 0.02, control_11=0.48, control_12=0.40)
    assert v["verdict"] == "partial_interference"
    assert v["holds"] == {"1-1": True, "1-2": False}


def test_verdict_degraded_when_both_fail_but_not_collapsed():
    # Both legs significantly below a 0.30 threshold, but 1-1 is not
    # under the 0.10 collapse ceiling, so this is degraded, not severe.
    v = _verdict(0.12, 0.06, control_11=0.60, control_12=0.60)
    assert v["verdict"] == "degraded"


def test_verdict_inconclusive_when_a_leg_cannot_discriminate():
    """The honest outcome when the registered n cannot separate the
    joint net from its threshold — reported, never rounded into a
    verdict."""
    v = _verdict(0.28, 0.40, control_11=0.48, control_12=0.40)
    assert v["leg_decisions"]["1-1"]["decision"] == "indeterminate"
    assert v["verdict"] == "inconclusive"
    assert any("indeterminate" in r for r in v["reasons"])


def test_verdict_unusable_when_the_positive_control_collapses():
    """DEFECT 4. If the specialist itself cannot clear under the gate
    protocol, a low joint rate is a harness result, not interference."""
    v = _verdict(0.02, 0.30, control_11=0.04, control_12=0.40)
    assert v["verdict"] == "unusable"
    assert any("positive control" in r for r in v["reasons"])


def test_verdict_failed_leg_is_unusable_never_a_verdict():
    v = _verdict(0.9, 0.9, kw_11={"status": "probe_failed"})
    assert v["verdict"] == "unusable"
    v2 = _verdict(0.9, 0.9, kw_c12={"status": "probe_timeout"})
    assert v2["verdict"] == "unusable"


def test_verdict_empty_leg_is_unusable():
    assert _verdict(0.9, 0.9, kw_12={"n": 0})["verdict"] == "unusable"
    assert _verdict(0.9, 0.9, kw_c11={"n": 0})["verdict"] == "unusable"


def test_verdict_missing_strict_predicate_is_unusable():
    s = _summ(strict=0.5)
    s["clear_rate_strict"] = None
    v = interference_verdict({"1-1": _summ(strict=0.5), "1-2": s},
                             {"1-1": _summ(strict=0.5),
                              "1-2": _summ(strict=0.5)})
    assert v["verdict"] == "unusable"


def test_verdict_flags_control_drift_from_the_receipted_prior():
    """The prior is advisory (different protocol) — a big gap is a
    reported warning about the harness, not a silent pass."""
    v = _verdict(0.10, 0.10, control_11=0.10, control_12=0.90)
    drift = v["prior_check"]
    assert drift["1-2"]["within_band"] is False
    assert drift["1-2"]["prior_rate"] == pytest.approx(0.38)
    assert any("drift" in w.lower() for w in v["advisories"])


def test_verdict_carries_measured_controls_and_priors():
    v = _verdict(0.40, 0.20)
    assert v["control_strict_rates"] == {"1-1": 0.48, "1-2": 0.40}
    assert v["joint_strict_rates"] == {"1-1": 0.40, "1-2": 0.20}
    assert v["prior_check"]["1-1"]["prior_rate"] == pytest.approx(0.4667,
                                                                  abs=5e-4)


# ---- step 3 wiring: the control legs must actually run -------------------


def test_step3_runs_a_control_leg_and_a_joint_leg_per_level(tmp_path,
                                                            monkeypatch):
    """DEFECT 4 wiring. Four legs, one protocol: for each level the
    specialist and the joint net differ ONLY in --checkpoint."""
    import scripts.interference_falsifier as mod
    joint = tmp_path / "joint.pt"
    joint.write_bytes(b"stub")
    monkeypatch.setitem(CONFIG, "joint_out", str(joint))
    monkeypatch.setattr(mod, "_receipt", lambda row: None)

    seen: list[list[str]] = []

    def fake_eval(cmd):
        seen.append(cmd)
        rate = 0.48 if "--checkpoint" in cmd and str(joint) not in cmd else 0.30
        return {"status": "ok", "n_episodes": 50, "clear_rate": rate,
                "max_gx_per_episode": [2600] * 50}

    verdict = step3_gate(run_eval=fake_eval)
    assert len(seen) == 4
    ckpts = [cmd[cmd.index("--checkpoint") + 1] for cmd in seen]
    assert sum(1 for c in ckpts if c == str(joint)) == 2
    assert sum(1 for c in ckpts if c != str(joint)) == 2
    # Identical protocol on every leg: the argv differ only by checkpoint.
    for level in CONFIG["levels"]:
        legs = [c for c in seen if f"--game {CONFIG['specialists'][level]['game']}"
                in " ".join(c)]
        assert len(legs) == 2
        a, b = ([x for x in legs[0] if x not in ckpts],
                [x for x in legs[1] if x not in ckpts])
        assert a == b
    assert verdict["control_strict_rates"] == {"1-1": 0.48, "1-2": 0.48}


# ---- eval command assembly ------------------------------------------------


def test_gate_command_1_1_is_strict_honest_from_the_canonical_entrance():
    cmd = build_gate_command("1-1", checkpoint=ROOT / CONFIG["joint_out"],
                             require_paths=False)
    joined = " ".join(cmd)
    assert "--game mario_1_1_backward" in joined
    assert "--episodes 100" in joined
    assert "--sticky-prob 0.25" in joined and "--start-jitter 16" in joined
    assert "sampled" not in joined  # cold ARGMAX (eval_game's default)
    assert "--eval-workers 5" in joined and "per-episode" in joined
    assert "--sequential" in cmd and "--level-clear" in cmd
    # The canonical 1-1 entrance: the backward tape's own root.
    assert "entrance_start.state" in joined
    assert "mario_1_1_backward.yaml" in joined
    # The gate seed, not the collection seed.
    assert str(CONFIG["gate_eval_seed"]) in joined
    assert str(CONFIG["collect_seed"]) not in joined


def test_gate_command_1_2_is_strict_honest_from_stage_03():
    cmd = build_gate_command("1-2", checkpoint=ROOT / CONFIG["joint_out"],
                             require_paths=False)
    joined = " ".join(cmd)
    assert "--game mario_1_2_consol2" in joined
    assert "--episodes 100" in joined
    assert "--sticky-prob 0.25" in joined and "--start-jitter 16" in joined
    assert "sampled" not in joined
    assert "--eval-workers 5" in joined and "per-episode" in joined
    assert "stage_03.state" in joined


def test_gate_command_paths_exist_for_both_levels():
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        cmd = build_gate_command(level,
                                 checkpoint=ROOT / spec["checkpoint"])
        for a in cmd:
            if "/" in a and not a.startswith("-"):
                assert Path(a).exists(), a


def test_collection_reference_command_is_sampled_house_noise():
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        cmd = build_collection_reference_command(
            level, checkpoint=ROOT / spec["checkpoint"])
        joined = " ".join(cmd)
        assert "--action-select sampled" in joined
        assert "--temperature 1.0" in joined
        assert "--episodes 300" in joined
        assert "--sticky-prob 0.25" in joined
        assert "--start-jitter 16" in joined
        assert "per-episode" in joined
        for a in cmd:
            if "/" in a and not a.startswith("-"):
                assert Path(a).exists(), a


def test_collection_plan_carries_the_strict_keep_predicate():
    for level in CONFIG["levels"]:
        plan = collection_plan(level)
        assert plan["level"] == level
        assert plan["action_select"] == "sampled"
        assert plan["temperature"] == 1.0
        assert plan["episodes"] == 300
        assert plan["sticky_prob"] == 0.25 and plan["start_jitter"] == 16
        assert plan["keep"] == "strict_episode_success_only"
        assert plan["checkpoint"].endswith(
            CONFIG["specialists"][level]["checkpoint"])


# ---- _run_eval (subprocess.run monkeypatched, no emulator) ---------------


def test_run_eval_recovers_status_json_on_nonzero_exit(monkeypatch):
    """eval_game.py's own guards (e.g. no_rom) print a status JSON to
    stdout and then exit non-zero with nothing on stderr -- that JSON
    must still come back, not get swapped for an empty stderr detail
    (regression: returncode used to be checked before stdout was ever
    read)."""
    no_rom_json = json.dumps(
        {"status": "no_rom", "rom_path": "/missing/game.nes"})

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout=no_rom_json, stderr="")

    monkeypatch.setattr(
        "scripts.interference_falsifier.subprocess.run", fake_run)
    result = _run_eval(["fake", "cmd"])
    assert result == {"status": "no_rom", "rom_path": "/missing/game.nes"}


# ---- config / manifest / resumability -------------------------------------


def test_preregistered_constants():
    assert CONFIG["levels"] == ("1-1", "1-2")
    s11 = CONFIG["specialists"]["1-1"]
    s12 = CONFIG["specialists"]["1-2"]
    assert s11["checkpoint"] == \
        "checkpoints/_preserved/backward_1_1_best_honest_greedy_047.pt"
    assert s12["checkpoint"] == \
        "checkpoints/_preserved/consol2_40pct_strict_iter01120.pt"
    assert CONFIG["collect_episodes"] == 300
    assert CONFIG["collect_sticky"] == 0.25
    assert CONFIG["collect_jitter"] == 16
    assert CONFIG["collect_temperature"] == 1.0
    assert CONFIG["bc_hidden_dim"] == 256
    assert CONFIG["bc_trunk_dim"] == 64
    assert CONFIG["bc_epochs"] == 50
    assert CONFIG["bc_lr"] == 3e-4
    assert CONFIG["gate_episodes"] == 100
    assert CONFIG["severe_strict_ceiling"] == 0.10
    assert CONFIG["hold_fraction_of_control"] == 0.50
    assert CONFIG["control_alpha"] == 0.05
    assert CONFIG["min_control_strict_rate"] == 0.15
    assert CONFIG["min_clone_hold_power"] == 0.80
    assert CONFIG["joint_out"] == "runs/interference/joint.pt"


def test_start_states_match_each_profile():
    """Coordination invariant: the per-level start state IS the level
    profile's own start_state_path — the blob every specialist number was
    measured from. A drifted pair would silently eval a different
    entrance."""
    import yaml
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        profile = yaml.safe_load((ROOT / spec["profile"]).read_text())
        assert spec["start_state"] == profile["start_state_path"], level


def test_manifest_mirrors_config_and_scopes_the_verdict():
    man = build_manifest()
    assert man["config"] == CONFIG
    assert man["campaign"] == "smb_generalist_interference_falsifier"
    rules = man["verdict_rules"]
    assert rules["severe_interference"].startswith("BOTH")
    assert "measured" in rules["reference"].lower()
    # The only 0.76 that may survive is the rejected-candidate note.
    scrubbed = json.loads(json.dumps(man))
    for level in CONFIG["levels"]:
        scrubbed["config"]["specialists"][level].pop("rejected_candidates")
    assert "0.76" not in json.dumps(scrubbed)
    # The verdict CALIBRATES the generalist plan; it does NOT gate the
    # 1-3 campaign — the manifest must say so in as many words.
    assert "calibrates" in man["scope_note"].lower()
    assert "does not gate the 1-3" in man["scope_note"].lower()
    assert len(man["steps"]) == 3
    # Four legs, and the power pre-registration travels with it.
    assert man["gate_legs"] == 4
    for level in CONFIG["levels"]:
        pw = man["power_preregistration"][level]
        assert pw["clone_holds"] >= CONFIG["min_clone_hold_power"]
        assert man["reference_priors"][level]["episodes"] >= 100


def test_steps_from_skip_to():
    assert steps_from(1) == [1, 2, 3]
    assert steps_from(3) == [3]
    with pytest.raises(ValueError):
        steps_from(0)
    with pytest.raises(ValueError):
        steps_from(4)


# ---- dry-run (live; config parse + checkpoint load, no rollouts) ----------


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_dry_run_passes_live():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "interference_falsifier.py"),
         "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=540)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout
    assert "FAIL" not in proc.stdout
    # Both specialist loads must be REPORTED with their inferred widths.
    assert "h64" in proc.stdout and "h256" in proc.stdout
    # The repairs are visible in the dry-run surface.
    assert "reference receipts re-verify" in proc.stdout
    assert "gate seed is disjoint" in proc.stdout
    assert "discriminate" in proc.stdout
