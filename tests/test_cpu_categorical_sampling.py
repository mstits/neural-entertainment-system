"""Guards for CPU categorical action sampling in the collection paths.

The pixel collection loops (vanilla_ppo at trainer.py and ga_ppo via
`_safe_sample_from_logits`) sample the policy's categorical action on
CPU instead of on MPS. On MPS `torch.multinomial` decomposes into ~10
serial primitive kernels (~0.83 ms/step, dispatch-bound — more than the
CNN forward), while the CPU draw on the same tiny logits is ~0.008 ms;
the collection loop already syncs actions to host every step, so moving
the sample off-device replaces a transfer rather than adding one.

These tests pin the two invariants that make the swap safe:

  * DISTRIBUTION EQUIVALENCE — sampling on CPU draws from the SAME
    categorical as sampling on MPS. Proven with a seeded chi-square on
    100k draws (CPU empirical vs MPS empirical, and each vs the exact
    softmax probabilities).
  * LOG-PROB PAIRING — the recorded log-prob is bit-identical to the
    full log-prob tensor indexed at the sampled action, for BOTH the
    vanilla replica and `_safe_sample_from_logits`. PPO's importance
    ratio is only correct if `log_probs_taken == log_probs_all[i, a]`.

Plus a guard on the fused-Adam optimizer selection (CPU-only, with a
graceful fallback when a torch build rejects `fused=True`).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.training.trainer import Trainer, _safe_sample_from_logits

_MPS = torch.backends.mps.is_available()


# --------------------------------------------------------------------------
# Distribution equivalence: CPU sampling == MPS sampling (seeded chi-square)
# --------------------------------------------------------------------------

def _sample_counts(probs: torch.Tensor, n: int, seed: int) -> np.ndarray:
    """Draw `n` categorical samples with replacement from a single
    probability row on `probs.device`, seeding that device's RNG, and
    return per-action counts."""
    if probs.device.type == "mps":
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    draws = torch.multinomial(probs, n, replacement=True)
    counts = torch.bincount(draws.cpu(), minlength=probs.numel())
    return counts.numpy().astype(np.float64)


@pytest.mark.skipif(not _MPS, reason="needs MPS to compare CPU vs MPS draws")
def test_cpu_vs_mps_sampling_same_distribution():
    """100k draws from an identical categorical on CPU and on MPS are
    statistically indistinguishable (chi-square), and both track the
    exact softmax probabilities."""
    from scipy import stats

    # Realistic policy logits over a Contra-sized action set.
    logits = torch.tensor(
        [2.0, -1.0, 0.5, 3.0, -0.5, 1.5, 0.0, -2.0, 0.8, 1.2],
        dtype=torch.float32,
    )
    n_actions = logits.numel()
    probs_cpu = F.log_softmax(logits, dim=-1).exp()
    probs_mps = F.log_softmax(logits.to("mps"), dim=-1).exp()

    N = 100_000
    counts_cpu = _sample_counts(probs_cpu, N, seed=1234)
    counts_mps = _sample_counts(probs_mps, N, seed=1234)

    # (1) CPU empirical vs MPS empirical must not be separable.
    contingency = np.vstack([counts_cpu, counts_mps])
    _, p_cpu_vs_mps, _, _ = stats.chi2_contingency(contingency)
    assert p_cpu_vs_mps > 0.01, (
        f"CPU and MPS draws differ (chi2 p={p_cpu_vs_mps:.4g}); the CPU "
        "path must sample the same categorical as MPS"
    )

    # (2) Each empirical distribution must fit the exact softmax pmf.
    # Rescale the expected counts so their sum matches the observed N
    # exactly (float rounding of probs*N leaves a ~1e-7 gap that
    # scipy.chisquare rejects on its sum-consistency check).
    expected = probs_cpu.numpy().astype(np.float64)
    for label, obs in (("cpu", counts_cpu), ("mps", counts_mps)):
        exp_scaled = expected * (obs.sum() / expected.sum())
        _, p_fit = stats.chisquare(f_obs=obs, f_exp=exp_scaled)
        assert p_fit > 0.01, (
            f"{label} draws diverge from the softmax pmf (chi2 p={p_fit:.4g})"
        )

    # (3) Hard, RNG-robust backstop: max per-action frequency gap is tiny.
    freq_cpu = counts_cpu / N
    freq_mps = counts_mps / N
    assert np.max(np.abs(freq_cpu - freq_mps)) < 0.01
    assert n_actions == 10


# --------------------------------------------------------------------------
# Log-prob pairing: chosen lp == full-lp indexed at sampled action (bitwise)
# --------------------------------------------------------------------------

def _vanilla_sample_replica(logits: torch.Tensor):
    """Byte-for-byte the trainer's vanilla_ppo sampling block: move the
    tiny logits to CPU, then log_softmax / exp / multinomial / gather all
    on the SAME CPU distribution."""
    logits_cpu = logits.float().cpu()
    log_probs_all = F.log_softmax(logits_cpu, dim=-1)
    probs = log_probs_all.exp()
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    log_probs_taken = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions, log_probs_taken, log_probs_all


def _assert_pairing(actions, log_probs_taken, log_probs_all):
    """`log_probs_taken[i]` must be the EXACT value sitting at
    `log_probs_all[i, actions[i]]` — no re-derivation, no cross-device
    numeric drift."""
    rows = torch.arange(actions.shape[0])
    indexed = log_probs_all[rows, actions]
    assert torch.equal(log_probs_taken, indexed), (
        "recorded log-prob is not bit-identical to the sampled action's "
        "entry in the distribution it was drawn from"
    )


def test_vanilla_replica_logprob_pairing_cpu():
    torch.manual_seed(0)
    logits = torch.randn(16, 10)
    actions, lp_taken, lp_all = _vanilla_sample_replica(logits)
    assert lp_all.device.type == "cpu"
    assert actions.device.type == "cpu"
    _assert_pairing(actions, lp_taken, lp_all)


@pytest.mark.skipif(not _MPS, reason="needs MPS for the device-logits path")
def test_vanilla_replica_logprob_pairing_from_mps_logits():
    """When the net emits logits on MPS, the replica still pairs bit-for-
    bit and lands everything on CPU (one small DtoH, no extra sync)."""
    torch.manual_seed(1)
    logits = torch.randn(16, 10, device="mps")
    actions, lp_taken, lp_all = _vanilla_sample_replica(logits)
    assert lp_all.device.type == "cpu"
    assert lp_taken.device.type == "cpu"
    _assert_pairing(actions, lp_taken, lp_all)


def test_safe_sample_logprob_pairing_cpu():
    torch.manual_seed(2)
    logits = torch.randn(12, 8)
    sampled, chosen_lp, log_probs_all, _ = _safe_sample_from_logits(logits)
    assert log_probs_all.device.type == "cpu"
    assert chosen_lp.device.type == "cpu"
    assert sampled.device.type == "cpu"
    _assert_pairing(sampled, chosen_lp, log_probs_all)


@pytest.mark.skipif(not _MPS, reason="needs MPS for the device-logits path")
def test_safe_sample_returns_cpu_from_mps_logits():
    """ga_ppo hands `_safe_sample_from_logits` MPS logits; the sanitise /
    sample / gather run on CPU and the returned tensors are on CPU so the
    caller's three `.cpu()` transfers collapse to a no-op."""
    torch.manual_seed(3)
    logits = torch.randn(12, 8, device="mps")
    sampled, chosen_lp, log_probs_all, _ = _safe_sample_from_logits(logits)
    assert sampled.device.type == "cpu"
    assert chosen_lp.device.type == "cpu"
    assert log_probs_all.device.type == "cpu"
    _assert_pairing(sampled, chosen_lp, log_probs_all)


# --------------------------------------------------------------------------
# Fused Adam selection (CPU-only, with graceful fallback)
# --------------------------------------------------------------------------

def _bare_trainer(device: str, rnd=None, lr: float = 2.5e-4) -> Trainer:
    """A Trainer shell with only the attributes `_build_ppo_optimizer`
    touches — avoids the heavyweight real __init__ (pool, ROM, threads)."""
    t = Trainer.__new__(Trainer)
    t.device = torch.device(device)
    t._rnd = rnd
    t.reinforce_lr = lr
    return t


def test_fused_adam_selected_on_cpu():
    net = torch.nn.Linear(8, 4)
    opt = _bare_trainer("cpu")._build_ppo_optimizer(net)
    assert opt.param_groups[0].get("fused") is True
    # And it actually steps (construction success is not enough).
    loss = net(torch.randn(3, 8)).sum()
    loss.backward()
    opt.step()


@pytest.mark.skipif(not _MPS, reason="needs MPS to check the non-CPU branch")
def test_non_cpu_device_keeps_default_adam():
    net = torch.nn.Linear(8, 4).to("mps")
    opt = _bare_trainer("mps")._build_ppo_optimizer(net)
    # MPS has no fused Adam — the builder must not request it.
    assert not opt.param_groups[0].get("fused")


def test_fused_adam_falls_back_when_torch_rejects(monkeypatch):
    """If a torch build rejects `fused=True`, the builder must retry with
    the default impl instead of taking the run down."""
    real_adam = torch.optim.Adam

    def picky_adam(params, **kwargs):
        if kwargs.get("fused"):
            raise RuntimeError("fused not supported on this build")
        return real_adam(params, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", picky_adam)
    net = torch.nn.Linear(8, 4)
    opt = _bare_trainer("cpu")._build_ppo_optimizer(net)
    assert not opt.param_groups[0].get("fused")
    # Fallback optimizer is a working Adam.
    loss = net(torch.randn(3, 8)).sum()
    loss.backward()
    opt.step()
