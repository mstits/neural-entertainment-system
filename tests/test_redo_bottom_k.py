"""Rank-based bottom-k ReDo selection (src/training/redo.py).

Registered: docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md §2, §3, §12.1,
§13. The licence for this rule is v31's banked stopping statement: on a
`Linear -> LayerNorm -> SiLU` 32-unit trunk there is NO fixed dormancy
threshold that is simultaneously firing, surgical and sustained (tau 0.10
equilibrated at 12/32 and tripped the dose ceiling; tau 0.075 recycled the
same two indices for the whole run). A rank rule caps the dose BY
CONSTRUCTION.

What these tests pin, and every one of them fails if `bottom_k_indices`
is reverted to a threshold rule (§13, executed not asserted):

  1. The rank rule fires when NO unit is below any plausible tau — the
     property that distinguishes it from every prior ReDo arm, and the
     one that makes it immune to the measured downward tail drift.
  2. The dose is exactly k for ANY score distribution, including one
     where 20 of 32 units sit under the threshold that voided v31 —
     the cap-by-construction claim, tested at the drift extreme.
  3. Scope is fc2 ONLY (fc1 is 0/64 and 0/96 dormant across 86 measured
     iterations; a rank rule there would be damage with no mechanism).
  4. Ties break toward the lower index and selection is a deterministic
     function of the logged score vector, so the B2 artifact-match gate
     can recompute the selection offline from the bytes on disk.
  5. `RedoStats.fc2_scores` IS that vector, and recomputing bottom-k
     from it reproduces the recorded indices exactly.
  6. Default-off: `mode` defaults to "threshold" and the threshold path
     is bit-identical to the pre-v32 behaviour.
  7. A declared-but-unreachable `k < 1` raises instead of running an arm
     that looks armed and recycles nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.tile_policy import TilePolicyNetwork
from src.training.redo import (
    SELECT_BOTTOM_K,
    SELECT_THRESHOLD,
    bottom_k_indices,
    check_and_recycle,
    dormancy_scores,
    hidden_activations,
    maybe_check_and_recycle,
)

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_PATH = ROOT / "src" / "training" / "trainer.py"

_NUM_ACTIONS = 6
_FEATURE_DIM = 12
_HIDDEN = 8
_TRUNK = 6


def _net(seed: int = 0) -> TilePolicyNetwork:
    torch.manual_seed(seed)
    return TilePolicyNetwork(
        num_actions=_NUM_ACTIONS, feature_dim=_FEATURE_DIM,
        hidden_dim=_HIDDEN, trunk_dim=_TRUNK,
    )


def _obs(rows: int = 32, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, _FEATURE_DIM, generator=g)


# ---------------------------------------------------------------------------
# 1. The rank rule itself
# ---------------------------------------------------------------------------

def test_bottom_k_picks_the_k_smallest_sorted_ascending():
    s = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7])
    assert bottom_k_indices(s, 2).tolist() == [1, 3]
    assert bottom_k_indices(s, 3).tolist() == [1, 2, 3]


def test_bottom_k_ties_break_toward_the_lower_index():
    """Determinism is load-bearing: the B2 artifact-match gate recomputes
    the selection offline from the logged score vector, so a tie that
    resolved differently in the trainer than in the gate would VOID a
    valid seed."""
    s = torch.tensor([0.5, 0.2, 0.2, 0.2, 0.9])
    assert bottom_k_indices(s, 1).tolist() == [1]
    assert bottom_k_indices(s, 2).tolist() == [1, 2]


def test_bottom_k_clamps_at_both_ends():
    s = torch.tensor([0.4, 0.1, 0.7])
    assert bottom_k_indices(s, 0).tolist() == []
    assert bottom_k_indices(s, -3).tolist() == []
    assert bottom_k_indices(s, 99).tolist() == [0, 1, 2]


def test_bottom_k_rejects_a_non_vector():
    with pytest.raises(ValueError):
        bottom_k_indices(torch.zeros(2, 3), 1)


# ---------------------------------------------------------------------------
# 2. The two properties the fixed-tau rule could not deliver
# ---------------------------------------------------------------------------

def test_rank_rule_fires_when_no_unit_is_below_any_plausible_tau():
    """v31's tau 0.075 did not fire until iter 29 and three campaigns have
    died of never firing. A rank rule has no abstention: with the whole
    layer scored well above the threshold, the threshold rule selects
    nothing and bottom-k still selects exactly k."""
    net, obs = _net(), _obs()
    _, h2, _ = hidden_activations(net, obs)
    scores = dormancy_scores(h2)
    assert float(scores.min()) > 0.25, "fixture must have no dormant unit"

    thr = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=0.25,
        mode=SELECT_THRESHOLD,
    )
    assert thr.recycled == 0

    rank = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=0.25,
        mode=SELECT_BOTTOM_K, bottom_k=2,
    )
    assert rank.recycled == 2
    assert rank.dormant_fc2 == 2


def test_rank_rule_dose_is_exactly_k_at_the_drift_extreme():
    """The cap-by-construction claim, tested where v31 died. A tau that
    swallows most of the trunk recycles most of the trunk; the rank rule
    recycles k on the SAME activations."""
    net, obs = _net(), _obs()
    _, h2, _ = hidden_activations(net, obs)
    scores = dormancy_scores(h2)
    fat_tau = float(scores.max()) + 1.0  # every unit "dormant"

    over = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=fat_tau,
        mode=SELECT_THRESHOLD,
    )
    assert over.dormant_fc2 == _TRUNK, "fixture must saturate the threshold"

    for k in (1, 2, 3):
        rank = check_and_recycle(
            net=_net(), optimizer=None, sample_obs=obs, tau=fat_tau,
            mode=SELECT_BOTTOM_K, bottom_k=k,
        )
        assert rank.dormant_fc2 == k
        assert rank.recycled == k


def test_rank_rule_never_touches_fc1():
    """Registered scope restriction: fc1 is 0/64 and 0/96 dormant across
    all 86 measured iterations, min never below 0.3086."""
    obs = _obs()
    net = _net()
    fc1_before = net.fc1.weight.detach().clone()
    norm1_before = net.norm1.weight.detach().clone()

    stats = check_and_recycle(
        net=net, optimizer=None, sample_obs=obs, tau=99.0,
        mode=SELECT_BOTTOM_K, bottom_k=2,
    )
    assert stats.fc1_indices == []
    assert stats.dormant_fc1 == 0
    assert torch.equal(net.fc1.weight, fc1_before)
    assert torch.equal(net.norm1.weight, norm1_before)


# ---------------------------------------------------------------------------
# 3. The artifact-match input (V32 §3 B2)
# ---------------------------------------------------------------------------

def test_stats_carry_the_full_score_vector_and_it_reproduces_the_selection():
    """The gate recomputes bottom-k offline from these numbers. If the
    logged vector and the selection can disagree, the campaign inherits
    Lane A's defect class: gates that certify the loop, never the file."""
    obs = _obs()
    stats = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=0.05,
        mode=SELECT_BOTTOM_K, bottom_k=2,
    )
    assert len(stats.fc2_scores) == _TRUNK
    recomputed = bottom_k_indices(torch.tensor(stats.fc2_scores), 2).tolist()
    assert recomputed == stats.fc2_indices
    assert stats.mode == SELECT_BOTTOM_K


def test_score_vector_is_recorded_before_the_recycle():
    """A vector measured AFTER the reset would describe the fresh units,
    not the units that were selected — and the offline recomputation
    would then disagree with the logged indices for a correct run."""
    obs = _obs()
    net = _net()
    _, h2, _ = hidden_activations(net, obs)
    expected = dormancy_scores(h2).tolist()

    stats = check_and_recycle(
        net=net, optimizer=None, sample_obs=obs, tau=0.05,
        mode=SELECT_BOTTOM_K, bottom_k=2,
    )
    assert stats.fc2_scores == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Default-off, and the loud failure for a mis-declared knob
# ---------------------------------------------------------------------------

def test_default_mode_is_threshold_and_is_unchanged():
    obs = _obs()
    default = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=99.0,
    )
    explicit = check_and_recycle(
        net=_net(), optimizer=None, sample_obs=obs, tau=99.0,
        mode=SELECT_THRESHOLD,
    )
    assert default.mode == SELECT_THRESHOLD
    assert default.fc1_indices == explicit.fc1_indices
    assert default.fc2_indices == explicit.fc2_indices
    # The threshold rule keeps fc1 in scope; the rank rule does not.
    assert default.dormant_fc1 == _HIDDEN


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown selection mode"):
        check_and_recycle(
            net=_net(), optimizer=None, sample_obs=_obs(), tau=0.1,
            mode="bottom-k",
        )


def test_bottom_k_below_one_raises_instead_of_running_inert():
    valid = np.arange(16)
    obs = torch.randn(16, _FEATURE_DIM)
    with pytest.raises(ValueError, match="requires bottom_k >= 1"):
        maybe_check_and_recycle(
            enabled=True, net=_net(), optimizer=None, obs_all=obs,
            valid_indices=valid, tau=0.1, sample_batch=8,
            check_every_iters=1, global_it=0,
            mode=SELECT_BOTTOM_K, bottom_k=0,
        )


def test_disabled_gate_is_still_inert_under_bottom_k():
    """OFF must consume no RNG and raise nothing, whatever the mode says."""
    before_np = np.random.get_state()[2]
    before_torch = torch.random.get_rng_state().clone()
    out = maybe_check_and_recycle(
        enabled=False, net=_net(), optimizer=None, obs_all=None,
        valid_indices=np.arange(4), tau=0.1, sample_batch=8,
        check_every_iters=1, global_it=0,
        mode=SELECT_BOTTOM_K, bottom_k=0,
    )
    assert out is None
    assert np.random.get_state()[2] == before_np
    assert torch.equal(torch.random.get_rng_state(), before_torch)


# ---------------------------------------------------------------------------
# 5. Cadence, and the production path (V32 §3 B1)
# ---------------------------------------------------------------------------

def test_cadence_gates_the_rank_rule():
    """The registered cadence is C=5 and it rides the EXISTING
    `redo_check_every_iters` key — no new cadence knob is declared."""
    valid = np.arange(16)
    obs = torch.randn(16, _FEATURE_DIM)
    fired = [
        it for it in range(12)
        if maybe_check_and_recycle(
            enabled=True, net=_net(), optimizer=None, obs_all=obs,
            valid_indices=valid, tau=0.0, sample_batch=8,
            check_every_iters=5, global_it=it,
            mode=SELECT_BOTTOM_K, bottom_k=2,
        ) is not None
    ]
    assert fired == [0, 5, 10]


def test_trainer_reads_the_registered_knobs_and_reaches_the_rank_rule():
    """§13: the mechanism firing in a unit test is not the mechanism
    firing in training. This pins the wiring; the live 5-iterate smoke in
    runs/v32_redo_bottom_k_2026-08-28/smoke/ is the receipt that the hot
    path executes it."""
    src = _TRAINER_PATH.read_text()
    assert 'rl_cfg.get("redo_mode", _REDO_SELECT_THRESHOLD)' in src
    assert 'rl_cfg.get("redo_bottom_k", 0)' in src
    # Passed into the end-of-iteration hook, not merely parsed.
    assert "mode=self.redo_mode," in src
    assert "bottom_k=self.redo_bottom_k," in src
    # B1 grep target and the B2 artifact-match line.
    assert "mode=%s k=%d recycle_scope=%s" in src
    assert '"[redo] fc2 scores: %s"' in src


def test_bottom_k_keys_are_registered_in_the_config_schema():
    """An unregistered key is rejected under --strict-config, which every
    v32 phase runs with."""
    from src.training.config_schema import KNOWN_REINFORCE_KEYS

    assert "redo_mode" in KNOWN_REINFORCE_KEYS
    assert "redo_bottom_k" in KNOWN_REINFORCE_KEYS
