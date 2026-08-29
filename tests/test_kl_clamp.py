"""vanilla_ppo_approx_kl must never reach metrics.jsonl as Infinity.

The k3 estimator `(ratio - 1) - log(ratio)` hits +inf the moment one
importance ratio underflows to 0 (log(0) = -inf), and the raw mean then
lands in metrics.jsonl as literal JSON `Infinity` — measured in the v32
campaign at up to 84/250 generations on one seed, invisible to jq
(silently DBL_MAX) and dropped by analyze.py's finite filter. The fix
mirrors `_safe_sample_from_logits` on the other network path: clamp to
a finite ceiling and COUNT the clamped rows (`kl_clamped_rows` →
`vanilla_ppo_kl_clamped_this_gen`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.ppo import _KL_DIAG_MAX, ppo_losses  # noqa: E402


def _call(logits, actions, log_probs_old):
    B = logits.shape[0]
    diag: dict = {}
    ppo_losses(
        logits,
        torch.zeros(B),
        actions,
        log_probs_old,
        torch.ones(B),          # advantages
        torch.zeros(B),         # value targets
        clip_eps=0.2,
        value_coef=0.25,
        entropy_coef=0.01,
        value_loss_kind="huber",
        diagnostics=diag,
    )
    return diag


def test_underflowed_ratio_yields_finite_kl_and_counts_the_row():
    # Row 0: taking an action the current policy gives ~exp(-2000)
    # probability while the behavior policy gave it probability 1 →
    # ratio underflows to exactly 0.0 in float32 → raw k3 = +inf.
    # Row 1: healthy (identical old/new → ratio 1, k3 0).
    logits = torch.tensor([[0.0, -2000.0],
                           [0.0, 0.0]])
    actions = torch.tensor([1, 0])
    log_probs_old = torch.tensor([0.0, math.log(0.5)])
    diag = _call(logits, actions, log_probs_old)
    kl = float(diag["approx_kl"])
    assert math.isfinite(kl), "approx_kl must never be Infinity/NaN"
    assert kl <= _KL_DIAG_MAX
    assert int(diag["kl_clamped_rows"]) == 1
    # The blowup stays LOUD: one row at the ceiling over two rows.
    assert kl == pytest.approx(_KL_DIAG_MAX / 2, rel=1e-3)


def test_healthy_minibatch_counts_zero_and_matches_raw_k3():
    torch.manual_seed(0)
    B, A = 32, 6
    logits = torch.randn(B, A)
    actions = torch.randint(0, A, (B,))
    lp_all = torch.log_softmax(logits, dim=-1)
    lp_new = lp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    # Behavior policy slightly off the current one: finite ratios.
    log_probs_old = lp_new + 0.1 * torch.randn(B)
    diag = _call(logits, actions, log_probs_old)
    assert int(diag["kl_clamped_rows"]) == 0
    ratio = torch.exp(lp_new - log_probs_old)
    raw_k3 = ((ratio - 1.0) - torch.log(ratio)).mean()
    assert float(diag["approx_kl"]) == pytest.approx(float(raw_k3), rel=1e-5)


def test_counts_are_tensors_not_synced_scalars():
    """The count must come back as a detached tensor so the updater can
    accumulate across minibatches without a per-minibatch MPS sync."""
    logits = torch.zeros(4, 3)
    diag = _call(logits, torch.zeros(4, dtype=torch.long), torch.zeros(4))
    assert torch.is_tensor(diag["kl_clamped_rows"])
    assert not diag["kl_clamped_rows"].requires_grad
    assert torch.is_tensor(diag["approx_kl"])
    assert not diag["approx_kl"].requires_grad
