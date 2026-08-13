"""Guard tests for the per-iter RND target-feature cache in the
vanilla_ppo update loop (src/training/trainer.py).

The cache precomputes the frozen RND target's embedding
`target(normalize_obs(obs))` once per iteration and reuses it across the
K-epoch minibatch loop instead of re-running the target CNN K times per
observation. It is a behaviour-IDENTITY transformation: the target is
frozen and `obs_rms` is held fixed for the whole update, so the cached
embedding equals the one the old per-minibatch `forward` recomputed.

Two things must hold and are guarded here:

1. Behaviour identity — cache-on and cache-off produce the same RND
   losses and the same post-update predictor/policy parameters, driven
   through the REAL RND / PolicyNetwork / ppo_losses op sequence.
2. Ordering — the cache MUST be built AFTER `update_normalization`
   (post-update `obs_rms`), because the minibatch RND forwards use the
   post-update stats. Building it earlier (stale rms) silently changes
   behaviour. Guarded both functionally and by source order.

Run on CPU with float32 for determinism (MPS kernel selection can add
last-ULP noise between batch sizes; the mechanism is identical either
way and the CPU path pins the exact-identity claim).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.policy_network import PolicyNetwork
from src.models.rnd import RND
from src.training.ppo import ppo_losses


DEV = torch.device("cpu")
NUM_ACTIONS = 6
FRAME_STACK = 4
FRAME_SIZE = 84
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
VALUE_LOSS_KIND = "huber"
GRAD_CLIP = 0.5
LR = 3.0e-4


def _make_scenario(total_n: int = 192, seed: int = 0):
    """Small synthetic rollout: obs + the four scalar vectors the update
    loop indexes, plus a frozen net/rnd pair with a NON-trivial obs_rms
    (as if the intrinsic pass' update_normalization already ran)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs = rng.random(
        (total_n, FRAME_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.float32
    )
    actions = rng.integers(0, NUM_ACTIONS, size=total_n).astype(np.int64)
    logp_old = rng.standard_normal(total_n).astype(np.float32)
    adv = rng.standard_normal(total_n).astype(np.float32)
    target = rng.standard_normal(total_n).astype(np.float32)

    net = PolicyNetwork(
        num_actions=NUM_ACTIONS, frame_stack=FRAME_STACK,
        frame_size=FRAME_SIZE, encoder="nature_dqn", use_layernorm=True,
    ).to(DEV)
    rnd = RND(in_channels=FRAME_STACK).to(DEV)

    # Simulate the intrinsic-reward block: update_normalization runs ONCE
    # per iter BEFORE the update loop, so obs_rms is post-update here.
    obs_t = torch.from_numpy(obs).to(DEV)
    with torch.no_grad():
        raw_err = rnd(obs_t)
    rnd.update_normalization(obs_t, raw_err)

    return obs, actions, logp_old, adv, target, net, rnd


def _snapshot(module: torch.nn.Module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _build_target_cache(rnd: RND, obs: np.ndarray, total_n: int,
                        chunk: int) -> torch.Tensor:
    """Mirror trainer.py's chunked no-grad cache build: full
    (total_n, feat_dim) tensor, target features scattered in by row."""
    cache = torch.zeros(total_n, rnd.feat_dim, device=DEV)
    for c0 in range(0, total_n, chunk):
        rows = np.arange(c0, min(c0 + chunk, total_n))
        rows_t = torch.from_numpy(rows).to(DEV)
        obs_chunk = torch.from_numpy(
            np.ascontiguousarray(obs[rows])
        ).to(DEV).float()
        cache.index_copy_(0, rows_t, rnd.target_features(obs_chunk))
    return cache


def _run_update_loop(net, rnd, obs, actions, logp_old, adv, target,
                     perms, mb_size, use_cache, cache_chunk=1024,
                     predictor_update_fraction=1.0, rnd_gate_enabled=True,
                     rnd_step_log=None, policy_step_log=None):
    """Faithful reconstruction of the vanilla_ppo K-epoch minibatch loop
    (trainer.py ~5161-5225): the ONLY things that differ between calls are
    the RND-loss computation (cached vs per-minibatch forward) and the
    predictor-update subsampling schedule. Everything else — ppo_losses,
    backward, clip, optimizer.step — is identical, so any post-update
    parameter drift can only come from those two levers.

    Predictor-update subsampling mirrors the trainer exactly: with
    `rnd_gate_enabled`, processed minibatch i (counted across all epochs)
    carries RND grads iff `i % round(1/predictor_update_fraction) == 0`;
    skipped minibatches build no RND graph, so the predictor's grads stay
    None (Adam's set_to_none zero_grad) and its params are frozen those
    steps. `rnd_gate_enabled=False` is the un-gated HEAD reference: the
    predictor trains every minibatch (equivalent to the fraction knob being
    absent entirely).

    Optional `rnd_step_log` / `policy_step_log` are appended one bool per
    processed minibatch: did the predictor / policy params change on that
    step. Returns (per-updated-mb rnd losses, post-update net + predictor
    params).
    """
    total_n = obs.shape[0]
    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(rnd.predictor.parameters()), lr=LR
    )
    actions_all = torch.from_numpy(actions).to(DEV)
    logp_old_all = torch.from_numpy(logp_old).to(DEV).float()
    adv_all = torch.from_numpy(adv).to(DEV).float()
    target_all = torch.from_numpy(target).to(DEV).float()

    tgt_cache = (
        _build_target_cache(rnd, obs, total_n, cache_chunk)
        if use_cache else None
    )

    stride = max(1, int(round(1.0 / predictor_update_fraction)))
    mb_index = 0

    net.train()
    rnd_losses: list[float] = []
    for perm in perms:
        n_valid = perm.shape[0]
        for mb_start in range(0, n_valid, mb_size):
            mb_np = perm[mb_start:mb_start + mb_size]
            if mb_np.size < 2:
                continue
            do_rnd = (mb_index % stride == 0) if rnd_gate_enabled else True
            mb_index += 1
            mb_idx = torch.from_numpy(mb_np).to(DEV)
            states_t = torch.from_numpy(
                np.ascontiguousarray(obs[mb_np])
            ).to(DEV).float()
            actions_t = actions_all[mb_idx]
            logp_old_t = logp_old_all[mb_idx]
            adv_t = adv_all[mb_idx]
            target_t = target_all[mb_idx]

            logits, values_pred = net.forward_ac(states_t)
            loss, _pl, _vl, _ent = ppo_losses(
                logits, values_pred, actions_t, logp_old_t, adv_t, target_t,
                clip_eps=CLIP_EPS, value_coef=VALUE_COEF,
                entropy_coef=ENTROPY_COEF, value_loss_kind=VALUE_LOSS_KIND,
            )
            if do_rnd:
                if tgt_cache is not None:
                    rnd_loss = rnd.predictor_loss(
                        states_t, tgt_cache[mb_idx]
                    ).mean()
                else:
                    rnd_loss = rnd(states_t).mean()
                loss = loss + rnd_loss
                rnd_losses.append(float(rnd_loss.detach().item()))

            pred_before = (
                [p.detach().clone() for p in rnd.predictor.parameters()]
                if rnd_step_log is not None else None
            )
            net_before = (
                [p.detach().clone() for p in net.parameters()]
                if policy_step_log is not None else None
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()

            if rnd_step_log is not None:
                rnd_step_log.append(any(
                    not torch.equal(a, b) for a, b in
                    zip(pred_before, rnd.predictor.parameters())
                ))
            if policy_step_log is not None:
                policy_step_log.append(any(
                    not torch.equal(a, b) for a, b in
                    zip(net_before, net.parameters())
                ))

    net_params = [p.detach().clone() for p in net.parameters()]
    pred_params = [p.detach().clone() for p in rnd.predictor.parameters()]
    return rnd_losses, net_params, pred_params


def test_predictor_loss_matches_forward_bit_exact() -> None:
    """The RND helper identity: for a fixed obs_rms, the frozen target is
    a constant, so `predictor_loss(obs, target_features(obs))` reproduces
    `forward(obs)` — the value the old per-minibatch path computed — to
    the bit (same batch, same normalize, same predictor forward)."""
    _obs, _a, _lp, _adv, _tg, _net, rnd = _make_scenario(total_n=128, seed=1)
    obs_t = torch.from_numpy(_obs).to(DEV)

    baseline = rnd(obs_t)
    cached = rnd.predictor_loss(obs_t, rnd.target_features(obs_t))

    assert torch.equal(baseline, cached), (
        "predictor_loss(obs, target_features(obs)) must bit-equal "
        f"forward(obs); max|Δ|={float((baseline - cached).abs().max()):.3e}"
    )


def test_update_loop_cache_on_vs_off_identical() -> None:
    """End-to-end behaviour identity through the real update op sequence:
    cache-on vs cache-off must yield the same RND losses AND the same
    post-update predictor + policy parameters. Uses cache_chunk=1024 vs a
    per-minibatch baseline (the production geometry) — on CPU float32 the
    per-row target features are batch-composition-independent, so this is
    exact."""
    total_n, mb_size, k_epochs = 192, 64, 3
    obs, actions, logp_old, adv, target, net, rnd = _make_scenario(
        total_n=total_n, seed=0
    )
    net_snap = _snapshot(net)
    rnd_snap = _snapshot(rnd)

    # Same minibatch permutations for both runs — isolate the cache.
    perm_rng = np.random.default_rng(123)
    valid = np.arange(total_n)
    perms = [perm_rng.permutation(valid) for _ in range(k_epochs)]

    losses_off, net_off, pred_off = _run_update_loop(
        net, rnd, obs, actions, logp_old, adv, target,
        perms, mb_size, use_cache=False,
    )

    # Reset to the identical starting point for the cached run.
    net.load_state_dict(net_snap)
    rnd.load_state_dict(rnd_snap)
    losses_on, net_on, pred_on = _run_update_loop(
        net, rnd, obs, actions, logp_old, adv, target,
        perms, mb_size, use_cache=True, cache_chunk=1024,
    )

    loss_off_t = torch.tensor(losses_off)
    loss_on_t = torch.tensor(losses_on)
    max_loss_diff = float((loss_off_t - loss_on_t).abs().max())
    max_net_diff = max(
        float((a - b).abs().max()) for a, b in zip(net_off, net_on)
    )
    max_pred_diff = max(
        float((a - b).abs().max()) for a, b in zip(pred_off, pred_on)
    )

    # Tight tolerance (should be exact on CPU). Report the verdict.
    bit_exact = (
        max_loss_diff == 0.0 and max_net_diff == 0.0 and max_pred_diff == 0.0
    )
    print(
        f"\n[rnd-cache identity] bit_exact={bit_exact} "
        f"max|Δrnd_loss|={max_loss_diff:.3e} "
        f"max|Δnet_param|={max_net_diff:.3e} "
        f"max|Δpred_param|={max_pred_diff:.3e}"
    )

    assert torch.allclose(loss_off_t, loss_on_t, rtol=0.0, atol=1e-6), (
        f"RND losses diverged: max|Δ|={max_loss_diff:.3e}"
    )
    for a, b in zip(net_off, net_on):
        assert torch.allclose(a, b, rtol=0.0, atol=1e-6)
    for a, b in zip(pred_off, pred_on):
        assert torch.allclose(a, b, rtol=0.0, atol=1e-6)


def test_predictor_fraction_one_is_bit_identical_to_head() -> None:
    """reinforce.rnd_predictor_update_fraction=1.0 must reproduce today's
    (HEAD) behaviour to the bit. HEAD trains the predictor on EVERY
    minibatch (no schedule gate). f=1.0 -> stride round(1/1.0)=1 ->
    `i % 1 == 0` is always true, so the gate is a proven no-op. This drives
    both the gated f=1.0 path and the un-gated HEAD reference through the
    identical op sequence and asserts the RND losses, policy params and
    predictor params all match exactly — a regression guard that the
    default knob value changes nothing (off-by-one in the schedule, a
    wrong stride for f=1.0, or a mis-placed gate would break it)."""
    total_n, mb_size, k_epochs = 192, 64, 3
    obs, actions, logp_old, adv, target, net, rnd = _make_scenario(
        total_n=total_n, seed=0
    )
    net_snap = _snapshot(net)
    rnd_snap = _snapshot(rnd)

    perm_rng = np.random.default_rng(123)
    valid = np.arange(total_n)
    perms = [perm_rng.permutation(valid) for _ in range(k_epochs)]

    # HEAD reference: predictor trained unconditionally, no schedule gate.
    losses_ref, net_ref, pred_ref = _run_update_loop(
        net, rnd, obs, actions, logp_old, adv, target,
        perms, mb_size, use_cache=True, rnd_gate_enabled=False,
    )

    # f=1.0 with the schedule gate active.
    net.load_state_dict(net_snap)
    rnd.load_state_dict(rnd_snap)
    losses_f1, net_f1, pred_f1 = _run_update_loop(
        net, rnd, obs, actions, logp_old, adv, target,
        perms, mb_size, use_cache=True,
        predictor_update_fraction=1.0, rnd_gate_enabled=True,
    )

    assert len(losses_f1) == len(losses_ref), (
        "f=1.0 must update the predictor on every minibatch, matching HEAD"
    )
    assert torch.equal(torch.tensor(losses_ref), torch.tensor(losses_f1)), (
        "f=1.0 RND losses diverged from the un-gated HEAD reference"
    )
    for a, b in zip(net_ref, net_f1):
        assert torch.equal(a, b), "f=1.0 policy params diverged from HEAD"
    for a, b in zip(pred_ref, pred_f1):
        assert torch.equal(a, b), "f=1.0 predictor params diverged from HEAD"


def test_predictor_fraction_quarter_schedule() -> None:
    """reinforce.rnd_predictor_update_fraction=0.25 must distill the
    predictor on EXACTLY the deterministic schedule (processed minibatch i
    carries RND grads iff i % round(1/0.25)=4 == 0) and nowhere else. The
    skipped minibatches build no RND graph, so — via Adam's set_to_none
    zero_grad — the predictor's params stay frozen there while policy/value
    still update every step. Asserts (a) predictor params change on exactly
    the scheduled steps, (b) policy params change on every step, and (c)
    exactly one RND loss is recorded per scheduled step."""
    total_n, mb_size, k_epochs = 192, 64, 4
    obs, actions, logp_old, adv, target, net, rnd = _make_scenario(
        total_n=total_n, seed=0
    )

    perm_rng = np.random.default_rng(123)
    valid = np.arange(total_n)
    perms = [perm_rng.permutation(valid) for _ in range(k_epochs)]

    rnd_step_log: list[bool] = []
    policy_step_log: list[bool] = []
    losses, _net_p, _pred_p = _run_update_loop(
        net, rnd, obs, actions, logp_old, adv, target,
        perms, mb_size, use_cache=True,
        predictor_update_fraction=0.25,
        rnd_step_log=rnd_step_log, policy_step_log=policy_step_log,
    )

    stride = 4  # round(1 / 0.25)
    n_mb = len(rnd_step_log)
    # 192/64 = 3 full minibatches/epoch, 4 epochs -> 12 processed steps.
    assert n_mb == 12, f"expected 12 processed minibatches, got {n_mb}"

    scheduled = [i for i in range(n_mb) if i % stride == 0]  # 0, 4, 8
    predictor_changed = [i for i, ch in enumerate(rnd_step_log) if ch]
    assert predictor_changed == scheduled, (
        f"predictor changed on {predictor_changed}; schedule is {scheduled} "
        "(steps off-schedule must leave the predictor frozen)"
    )

    # Policy/value learn on EVERY minibatch — the schedule only gates RND.
    assert all(policy_step_log), (
        "policy params must update on every minibatch, including the "
        f"RND-skipped ones; got change-flags {policy_step_log}"
    )

    # Exactly one recorded RND loss per scheduled step.
    assert len(losses) == len(scheduled), (
        f"expected {len(scheduled)} RND-loss records, got {len(losses)}"
    )


def test_trainer_gates_rnd_on_predictor_update_fraction() -> None:
    """Source guard: the vanilla update loop must read
    reinforce.rnd_predictor_update_fraction, derive the deterministic
    stride, and gate the RND predictor loss on it — so the reconstruction
    tests above stay tied to the real trainer. Fails loudly if the knob is
    dropped or the gate is removed from _run_vanilla_ppo."""
    # The knob is still parsed in the trainer's __init__ (config parse stays
    # in the conductor). The update stretch that derives the stride + gates the
    # RND predictor loss on it moved verbatim into PPOUpdater.update
    # (trainer-decomposition plan, Task 2), so those two anchors now read
    # src/training/ppo_updater.py. Assertions unchanged (the gate's `self.`->`t.`
    # trainer-ref form moved with the code).
    src = Path("src/training/trainer.py").read_text()
    assert 'rl_cfg.get("rnd_predictor_update_fraction"' in src, (
        "trainer must read reinforce.rnd_predictor_update_fraction from cfg"
    )
    upd = Path("src/training/ppo_updater.py").read_text()
    i_update = upd.find("def update(")
    assert i_update != -1, "PPOUpdater.update not found"
    i_stride = upd.find("_rnd_pred_stride", i_update)
    assert i_stride != -1, (
        "predictor-update stride not derived inside PPOUpdater.update"
    )
    # The RND predictor block must be guarded by the per-minibatch flag.
    i_gate = upd.find("t._rnd is not None and _rnd_update_this_mb", i_update)
    assert i_gate != -1, (
        "RND predictor loss must be gated by the per-minibatch schedule "
        "flag (_rnd_update_this_mb) in PPOUpdater.update"
    )


def test_cache_uses_post_update_rms() -> None:
    """Ordering guard (functional): the cache MUST be built with the
    post-`update_normalization` obs_rms. Build a cache before the update
    and after it; they must differ (proving order is load-bearing), and
    the after-update cache must equal target_features under the live
    (post-update) obs_rms. If a refactor moved the build ahead of
    update_normalization, the trainer would populate the cache with the
    pre-update stats — this asserts the two are NOT interchangeable."""
    total_n = 128
    rng = np.random.default_rng(7)
    obs = rng.random(
        (total_n, FRAME_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.float32
    )
    torch.manual_seed(7)
    rnd = RND(in_channels=FRAME_STACK).to(DEV)
    obs_t = torch.from_numpy(obs).to(DEV)

    # Cache as it WOULD be built with the pre-update (init) obs_rms.
    cache_pre = _build_target_cache(rnd, obs, total_n, chunk=1024)

    # Run the intrinsic-pass normalization update, then build again.
    with torch.no_grad():
        raw_err = rnd(obs_t)
    rnd.update_normalization(obs_t, raw_err)
    cache_post = _build_target_cache(rnd, obs, total_n, chunk=1024)

    # The two must differ — otherwise the ordering constraint would be
    # vacuous and a mis-ordered build would go unnoticed.
    assert not torch.allclose(cache_pre, cache_post, rtol=0.0, atol=1e-5), (
        "obs_rms update did not change the target features; the ordering "
        "guard cannot distinguish pre- vs post-update caches."
    )
    # And the post-update cache must equal the live target features.
    live = rnd.target_features(obs_t)
    assert torch.equal(cache_post, live)


def test_trainer_builds_cache_after_update_normalization() -> None:
    """Ordering guard (source): in the vanilla_ppo update, the RND target
    cache (`target_features`) must be built AFTER `update_normalization`.
    Fails loudly if someone moves the cache build ahead of the intrinsic
    block's obs_rms update, which would bake in stale stats.

    The update stretch moved verbatim from `Trainer._run_vanilla_ppo` into
    `PPOUpdater.update` (trainer-decomposition plan, Task 2), so this anchor now
    reads src/training/ppo_updater.py. Assertions unchanged (the `self.`->`t.`
    trainer-ref form moved with the code)."""
    src = Path("src/training/ppo_updater.py").read_text()

    # Anchor inside PPOUpdater.update: the trainer's GA path has its own
    # update_normalization call, but that stayed in trainer.py; here the
    # updater body contains exactly one intrinsic-block update_normalization.
    i_update = src.find("def update(")
    assert i_update != -1, "PPOUpdater.update not found"
    i_cache_build = src.find("t._rnd.target_features(", i_update)
    assert i_cache_build != -1, (
        "target_features cache build not found inside PPOUpdater.update"
    )
    i_update_norm = src.rfind(
        "t._rnd.update_normalization(", i_update, i_cache_build
    )
    assert i_update_norm != -1, (
        "RND target cache (target_features) must be built AFTER the "
        "vanilla intrinsic block's update_normalization — the minibatch "
        "loop needs post-update obs_rms. No update_normalization call "
        "precedes the cache build inside PPOUpdater.update."
    )

    # The cached-loss consumer must also be present (predictor_loss),
    # guarding against a half-applied revert that leaves a dead cache.
    assert "t._rnd.predictor_loss(" in src
