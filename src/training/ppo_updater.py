"""Core PPO-update owner for the vanilla-PPO conductor.

`PPOUpdater` owns the net-covered core of `Trainer._run_vanilla_ppo`'s update
half: the RND intrinsic fold (with the single obs-rms `update_normalization`
call), the count-bonus fold, the batched GAE-lambda sweep, advantage
normalization over the valid mask, and the K-epoch minibatch PPO update loop
(RND target-feature cache, demo anchor, SHAPO/SAM three-pass variant, and the
non-finite-loss backstop). The logic lives here; the conductor keeps only the
un-net-covered blocks (PR-MDP adversary update, CGSA, backward curriculum) that
the tile-vanilla golden profile disables. This is the second strangler step of
`docs/proposals/trainer_decomposition_plan.md` (Task 2).

The updater holds a reference back to the owning `Trainer` so the update reads
the live hyperparameters, `_rnd`/`_demo_bank` modules, `device`, and `_gen_timer`
exactly as the inline code did -- every read, side effect, and log line is
preserved verbatim, just relocated. Loop-local buffers, the net/optimizer, and
per-iter scalars arrive as keyword arguments; the folded `reward_buf`, the valid
mask, the shared obs tensors, `mb_size`, and the reported scalars are handed back
so the conductor's left-behind PR-MDP block reuses the identical intermediates
(byte-identical behavior on that disabled path). The SHAPO actor-parameter set is
cached on the updater across iters, matching the old once-per-run local.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.training.ppo import (
    batched_gae, demo_anchor_loss, fold_intrinsic_into_rewards, ppo_losses,
)

log = logging.getLogger(__name__)


class PPOUpdater:
    """K-epoch PPO update owner for one vanilla-PPO run.

    `trainer` is the owning `Trainer`; the updater reads its live training
    knobs and modules through the `self.trainer` ref (aliased `t` inside
    `update`) so the update sees exactly the state the inline loop expected.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        # SHAPO/SAM actor set, resolved lazily once per run (matches the old
        # `_sam_actor_params = None` local hoisted above the iter loop).
        self._sam_actor_params = None

    def update(
        self, *, net, optimizer, obs_buf, action_buf, reward_buf, value_buf,
        log_prob_buf, done_buf, valid_buf, bonus_buf, final_values_np,
        rollout_steps, num_envs, obs_shape, global_it, sam_rho,
        trunc_buf=None, entropy_weight_buf=None,
    ) -> dict:
        """Run the core PPO update on one filled rollout, mutating `net`,
        `optimizer`, and the RND predictor in place.

        Returns a dict of the folded `reward_buf`, the flat valid mask +
        indices, the intrinsic/count means, the final-minibatch losses, the
        demo-anchor coefficient/accumulator, and the shared obs tensors +
        `mb_size` the conductor's PR-MDP block reuses. The recurrent branch
        delegates to the trainer's `_recurrent_ppo_update`.
        """
        # Module-load-time thread default lives on the trainer module; import
        # lazily to avoid an import cycle (trainer imports this module).
        from src.training.trainer import _TORCH_DEFAULT_NUM_THREADS
        t = self.trainer
        # ============== RND INTRINSIC REWARD ==============
        # Fold the per-state novelty bonus into the reward stream
        # BEFORE GAE so it bootstraps like any reward (single-stream
        # RND). Computed once on the full rollout obs with the
        # current predictor (no grad); the predictor is then trained
        # in the update below, driving the bonus down on familiar
        # states. Zeroed on done/padded steps by the fold helper.
        rnd_intrinsic_mean = 0.0
        # Attribute the RND full-rollout intrinsic pass — previously
        # part of the unbucketed ~17% iter-wall gap. Diagnostic only.
        _rnd_intrinsic_t0 = time.perf_counter_ns()
        if t._rnd is not None:
            # This full-rollout pass (rollout_steps × num_envs rows in
            # one op) is the only torch call in the loop big enough to
            # profit from the intra-op pool (62 ms at 1T vs 24 ms at
            # default threads). Lift the CPU 1-thread cap from
            # __init__ just for this block, then restore it for the
            # small-tensor minibatch loop below.
            _rnd_threads_raised = (
                t.device.type == "cpu"
                and torch.get_num_threads() < _TORCH_DEFAULT_NUM_THREADS
            )
            if _rnd_threads_raised:
                torch.set_num_threads(_TORCH_DEFAULT_NUM_THREADS)
            try:
                with torch.no_grad():
                    rnd_obs_t = (
                        torch.from_numpy(
                            obs_buf.reshape((rollout_steps * num_envs,) + obs_shape)
                        ).to(t.device).float()
                    )
                    if not t._is_tile_mode and not t.preprocess_f16:
                        rnd_obs_t = rnd_obs_t.div_(255.0)
                    intrinsic_raw = t._rnd(rnd_obs_t)  # raw per-sample MSE
                    bonus_t = t._rnd.normalize_bonus(intrinsic_raw)
                    # Update running stats with the RAW error, not the
                    # normalized bonus (avoids the reward_rms self-loop).
                    t._rnd.update_normalization(rnd_obs_t, intrinsic_raw)
                    intrinsic_np = (
                        bonus_t.cpu().numpy().astype(np.float32)
                        * t.rnd_intrinsic_coef
                    ).reshape(rollout_steps, num_envs)
            finally:
                if _rnd_threads_raised:
                    torch.set_num_threads(1)
            rnd_intrinsic_mean = float(intrinsic_np.mean())
            reward_buf = fold_intrinsic_into_rewards(
                reward_buf, intrinsic_np, done_buf
            )
        t._gen_timer.add(
            "rnd_intrinsic", time.perf_counter_ns() - _rnd_intrinsic_t0
        )
        # Fold the count-based frontier bonus exactly like RND's
        # intrinsic stream (same done-masked helper).
        count_bonus_mean = 0.0
        if t._gx_count_beta > 0.0:
            count_bonus_mean = float(bonus_buf.mean())
            reward_buf = fold_intrinsic_into_rewards(
                reward_buf, bonus_buf, done_buf
            )

        # ============== GAE-λ BACKWARD SWEEP ==============
        # Batched, per-step-done-masked GAE (src/training/ppo.py).
        # A done at step t zeroes both the bootstrapped next-value
        # and the running accumulator, breaking the episode
        # boundary so advantage never leaks across death/clear —
        # required now that auto-reset gives multiple dones per env
        # per rollout.
        _gae_t0 = time.perf_counter_ns()
        advantages, value_targets = batched_gae(
            reward_buf, value_buf, done_buf, final_values_np,
            t.reinforce_gamma, t.gae_lambda, trunc_buf=trunc_buf,
        )
        t._gen_timer.add("gae", time.perf_counter_ns() - _gae_t0)

        # Global advantage normalization across the entire (env × step)
        # batch. This is the canonical PPO advantage normalization
        # — preserves magnitude information between trajectories,
        # unlike per-trajectory z-scoring which would erase the
        # "this env had a high-return episode" signal.
        # Normalize over VALID steps only. Post-done frozen padding
        # carries advantage = -V(stale); including it would skew the
        # batch mean/std (and it is excluded from the minibatch below).
        valid_flat = valid_buf.reshape(-1)
        _valid_adv = advantages.reshape(-1)[valid_flat]
        if _valid_adv.size > 1:
            adv_mean = float(_valid_adv.mean())
            adv_std = float(_valid_adv.std()) + 1e-8
        else:
            adv_mean, adv_std = 0.0, 1.0
        advantages_norm = (advantages - adv_mean) / adv_std

        # Critic explained variance: 1 - Var(returns - V) / Var(returns),
        # on the SAME valid rows as the advantage stats above, using the
        # rollout-time critic predictions (`value_buf`, pre-update) against
        # the GAE targets (`value_targets`) that just came out of
        # `batched_gae` — both already in scope, so this is a read of
        # existing arrays, not a new pass. This is the live-training
        # sibling of the offline `scripts/critic_explained_variance.py`
        # (V29_STABILITY_2026-08-25.md F1): EV == 1 means the critic's
        # pre-update predictions perfectly explain this iter's returns;
        # EV <= 0 means it does no better (or worse) than predicting the
        # mean return.
        _valid_targets = value_targets.reshape(-1)[valid_flat]
        _valid_values = value_buf.reshape(-1)[valid_flat]
        _target_var = float(np.var(_valid_targets)) if _valid_targets.size else 0.0
        explained_variance = (
            1.0 - float(np.var(_valid_targets - _valid_values)) / _target_var
            if _target_var > 1e-8 else 0.0
        )

        # ============== K-EPOCH PPO UPDATE ==============
        # Flatten (rollout_steps, num_envs, ...) → (rollout_steps * num_envs, ...)
        total_n = rollout_steps * num_envs
        obs_flat = obs_buf.reshape((total_n,) + obs_shape)
        action_flat = action_buf.reshape(-1)
        log_prob_old_flat = log_prob_buf.reshape(-1)
        adv_flat = advantages_norm.reshape(-1)
        target_flat = value_targets.reshape(-1)
        # Only real (valid) steps are trained on — drop frozen padding.
        valid_indices = np.where(valid_flat)[0]

        net.train()
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_loss = 0.0
        last_rnd_loss = 0.0
        # Hold the final minibatch's loss tensors and convert to Python
        # floats ONCE after the update loop, instead of .item()-syncing
        # the MPS stream on every minibatch (only the last is logged).
        # Each .item() drains the GPU queue, serializing CPU-side
        # minibatch prep against GPU compute — costliest for the tiny
        # tile MLP. None-guarded so the recurrent path (which skips this
        # loop and sets last_* itself) is unaffected.
        _last_policy_t = _last_value_t = _last_entropy_t = None
        _last_loss_t = _last_rnd_t = None
        # Final-minibatch trust-region diagnostics (V29_STABILITY_2026-
        # 08-25.md F0), captured the same way as the loss scalars above:
        # tensors held across the loop, floated once at the end so no
        # extra MPS sync happens per minibatch. `_diag` is the out-dict
        # `ppo_losses` fills in place with `clip_fraction`/`approx_kl`
        # from the ratio it already computes — passed only on the
        # primary (non-SHAPO-pass-B/C) call below, since that call's
        # policy/value/entropy/loss are the ones reported everywhere
        # else in this function.
        _diag: dict = {}
        _last_grad_norm_t = None
        _last_clip_frac_t = None
        _last_approx_kl_t = None
        mb_size = max(1, t.ppo_minibatch_size)
        # Build the full-rollout tensors ONCE per iter and index them
        # with a torch permutation inside the minibatch loop, instead
        # of rebuilding 5 tensors from numpy (cast + host-copy) for
        # every one of ~K*total_n/mb minibatches. The four scalar
        # vectors are tiny; the obs tensor is materialized up front
        # only for tile mode (~170 MB float32) — for the pixel path
        # that would be multi-GB, so pixel obs stays per-minibatch.
        actions_all = torch.from_numpy(
            action_flat.astype(np.int64)
        ).to(t.device)
        log_probs_old_all = torch.from_numpy(log_prob_old_flat).to(t.device).float()
        adv_all = torch.from_numpy(adv_flat).to(t.device).float()
        # Commitment options: per-row entropy weights (the intended
        # duration k at decision rows). None on every other path — and
        # None reproduces legacy entropy exactly (tested in ppo.py).
        # FULL flat like actions_all/adv_all: mb_idx carries full-flat
        # indices (a permutation of valid_indices), so pre-filtering here
        # would misalign every minibatch row.
        entw_all = None
        if entropy_weight_buf is not None:
            entw_all = torch.from_numpy(
                entropy_weight_buf.reshape(-1)
            ).to(t.device).float()
        target_all = torch.from_numpy(target_flat).to(t.device).float()
        obs_all = (
            torch.from_numpy(obs_flat).to(t.device).float()
            if (t._is_tile_mode and not t._recurrent) else None
        )
        _upd_t0 = time.perf_counter_ns()
        if t._recurrent:
            # Recurrent sequence-BPTT update. The feedforward epoch
            # loop below is skipped (its range is zeroed when
            # recurrent) so the proven path stays byte-for-byte intact.
            (last_policy_loss, last_value_loss, last_entropy,
             last_loss, last_rnd_loss) = t._recurrent_ppo_update(
                net, optimizer, obs_buf, action_buf, log_prob_buf,
                advantages_norm, value_targets, done_buf,
                num_envs, rollout_steps,
            )
        # RND target-feature cache (behaviour-identical). The frozen
        # target's embedding target(normalize_obs(obs)) is a constant
        # for the entire K-epoch update: the target is frozen AND
        # obs_rms was already updated once this iter, in the intrinsic
        # block ABOVE — so the minibatch RND loss must (and does) see
        # the post-update stats. Precompute it ONCE here, in a no-grad
        # chunked pass keyed by flat rollout index, then index it per
        # minibatch instead of re-running the target CNN K times per
        # observation. Only valid on the feedforward vanilla path:
        # the GA `_reinforce_update` mutates obs_rms inside its loop
        # (cache would go stale) and the recurrent path has its own
        # update — both are excluded here. Only the frozen target is
        # cached; the predictor trains every step and is never cached.
        rnd_tgt_feat_cache = None
        if (t._rnd is not None and not t._recurrent
                and valid_indices.size):
            rnd_tgt_feat_cache = torch.zeros(
                total_n, t._rnd.feat_dim, device=t.device
            )
            # Chunk the build so peak memory stays bounded (the full
            # cache is (total_n, 512) f32; each chunk adds one obs
            # tensor + one feature tensor on top).
            _rnd_cache_chunk = 1024
            for _c0 in range(0, valid_indices.shape[0], _rnd_cache_chunk):
                _rows = valid_indices[_c0:_c0 + _rnd_cache_chunk]
                _rows_t = torch.from_numpy(_rows).to(t.device)
                if obs_all is not None:
                    _obs_chunk = obs_all[_rows_t]
                else:
                    _obs_chunk = torch.from_numpy(
                        np.ascontiguousarray(obs_flat[_rows])
                    ).to(t.device).float()
                    if not t.preprocess_f16:
                        _obs_chunk = _obs_chunk.div_(255.0)
                rnd_tgt_feat_cache.index_copy_(
                    0, _rows_t, t._rnd.target_features(_obs_chunk)
                )
        # RND predictor-update subsampling schedule. The predictor is
        # distilled on a DETERMINISTIC cadence: processed minibatch i
        # (counted across all K epochs) carries RND grads iff
        # i % _rnd_pred_stride == 0. f=1.0 -> stride 1 -> every
        # minibatch, byte-identical to the un-subsampled path. Skipped
        # minibatches never build the RND graph, so (with Adam's
        # set_to_none zero_grad) the predictor params stay frozen on
        # those steps while policy/value still update. The cache above
        # is built unconditionally — it still serves the minibatches
        # that DO update, and its cost pays for itself at f>=0.25.
        _rnd_pred_stride = max(
            1, int(round(1.0 / t.rnd_predictor_update_fraction))
        )
        _rnd_mb_index = 0
        # Demo-anchor coefficient: linear decay from coef0 to final
        # over decay_iters (resume-safe via the absolute iteration),
        # so early training rides the demo spine and late training
        # lets the reward gradient own — and exceed — the demos.
        if t._demo_bank is not None:
            _da_frac = min(
                1.0,
                max(0, global_it - t.demo_anchor_decay_start)
                / max(1, t.demo_anchor_decay_iters),
            )
            _demo_coef = t.demo_anchor_coef0 + _da_frac * (
                t.demo_anchor_final - t.demo_anchor_coef0
            )
        else:
            _demo_coef = 0.0
        _demo_loss_accum = 0.0
        _demo_loss_n = 0
        # Self-imitation BC (reinforce.sil): each minibatch also draws from
        # the stored level clears and adds cross-entropy at bc_coef —
        # exactly the demo-anchor plug shape, but the bank is the policy's
        # OWN clears. Inert when unset/empty. Not applied under SHAPO
        # (`loss` never backprops there); the trainer warns on that combo.
        _sil_buf = getattr(t, "_sil_buffer", None)
        _sil_coef = float(getattr(t, "_sil_bc_coef", 0.0) or 0.0)
        _sil_active = (
            _sil_buf is not None and len(_sil_buf) > 0 and _sil_coef > 0.0
            and sam_rho <= 0.0
        )
        _sil_loss_accum = 0.0
        _sil_loss_n = 0
        # KL-anchored warm start (reinforce.kl_anchor_checkpoint): during
        # the critic-warmup phase the actor set's grads are dropped before
        # every optimizer step, so only the fresh critic moves.
        _kl_anchor = getattr(t, "_kl_anchor", None)
        _actor_frozen = _kl_anchor is not None and _kl_anchor.frozen
        # Loss-level anchor tether (reinforce.kl_anchor_loss_coef): each
        # minibatch adds coef * mean KL(prior(.|s) || pi_theta(.|s)) on its
        # own states directly to the loss — the prior logits come no-grad
        # from the same frozen prior the reward-level penalty holds, so the
        # two paths compose. 0.0 (the default) never enters the branch:
        # no extra tensor work, bit-identical update. Not applied under
        # SHAPO (`loss` never backprops there), matching the SIL plug.
        _kl_loss_coef = float(getattr(t, "_kl_anchor_loss_coef", 0.0) or 0.0)
        _kl_loss_active = (
            _kl_anchor is not None and _kl_loss_coef > 0.0 and sam_rho <= 0.0
        )
        _kl_loss_accum = 0.0
        _kl_loss_n = 0
        for epoch in range(0 if t._recurrent else t.reinforce_steps):
            perm = np.random.permutation(valid_indices)
            n_valid = perm.shape[0]
            for mb_start in range(0, n_valid, mb_size):
                mb_end = min(mb_start + mb_size, n_valid)
                mb_np = perm[mb_start:mb_end]
                if mb_np.size < 2:
                    continue
                mb_idx = torch.from_numpy(mb_np).to(t.device)
                # Advance the schedule counter per PROCESSED minibatch
                # (post skip-guard) and decide whether this step trains
                # the predictor. Cheap host-side ints; no tensor work.
                _rnd_update_this_mb = (_rnd_mb_index % _rnd_pred_stride == 0)
                _rnd_mb_index += 1

                if obs_all is not None:
                    states_t = obs_all[mb_idx]
                else:
                    states_t = torch.from_numpy(
                        np.ascontiguousarray(obs_flat[mb_np])
                    ).to(t.device).float()
                    # Pixel path: /255 only when the pool delivered
                    # uint8; preprocess_f16 obs are already [0,1].
                    if not t.preprocess_f16:
                        states_t = states_t.div_(255.0)
                actions_t = actions_all[mb_idx]
                log_probs_old_t = log_probs_old_all[mb_idx]
                adv_t = adv_all[mb_idx]
                target_t = target_all[mb_idx]

                logits, values_pred = net.forward_ac(states_t)
                # PPO clipped-surrogate + value + entropy loss
                # (src/training/ppo.py). Forward pass and optimizer
                # step stay here; the pure loss math is shared so
                # the Phase 1 RND intrinsic term has one plug point.
                loss, policy_loss, value_loss, entropy = ppo_losses(
                    logits, values_pred, actions_t, log_probs_old_t,
                    adv_t, target_t,
                    clip_eps=t.ppo_clip_eps,
                    value_coef=t.value_coef,
                    entropy_coef=t.entropy_coef,
                    value_loss_kind=t.value_loss_kind,
                    entropy_weights=(entw_all[mb_idx]
                                     if entw_all is not None else None),
                    diagnostics=_diag,
                )
                # Pulled out right after the call (not read from `_diag`
                # at the end of the function): a minibatch that goes on
                # to hit the non-finite-loss backstop below never
                # updates `_last_policy_t`/etc either, so these two must
                # stay in lockstep with that "last SUCCESSFUL minibatch"
                # convention rather than reporting a skipped step's
                # diagnostics.
                _mb_clip_frac_t = _diag.get("clip_fraction")
                _mb_approx_kl_t = _diag.get("approx_kl")
                # Demo anchor (DQfD-style): every PPO minibatch also
                # draws a demo minibatch from the fixed bank and adds
                # CE(+large-margin) on the demo actions, decayed by
                # _demo_coef (computed per iter). Same backward pass —
                # demonstrations and reward shape one gradient.
                if t._demo_bank is not None and _demo_coef > 0:
                    d_obs, d_act = t._demo_bank.sample(
                        t.demo_anchor_mb
                    )
                    d_logits, _ = net.forward_ac(d_obs)
                    _da_loss = demo_anchor_loss(
                        d_logits, d_act, margin=t.demo_anchor_margin
                    )
                    loss = loss + _demo_coef * _da_loss
                    # Tensor-accumulate; a single sync at metrics
                    # emission instead of one .item() per minibatch.
                    _demo_loss_accum = _demo_loss_accum + _da_loss.detach()
                    _demo_loss_n += 1
                if _sil_active:
                    s_obs, s_act = _sil_buf.sample(mb_size)
                    s_obs_t = torch.from_numpy(s_obs).to(t.device).float()
                    if not t._is_tile_mode and not t.preprocess_f16:
                        s_obs_t = s_obs_t.div_(255.0)
                    s_logits, _ = net.forward_ac(s_obs_t)
                    _sil_loss = F.cross_entropy(
                        s_logits.float(),
                        torch.from_numpy(s_act).to(t.device),
                    )
                    loss = loss + _sil_coef * _sil_loss
                    _sil_loss_accum = _sil_loss_accum + _sil_loss.detach()
                    _sil_loss_n += 1
                # Loss-level anchor tether: coef * mean KL(prior || pi) on
                # THIS minibatch's states, straight into the policy loss —
                # the per-update pull the reward-level beta penalty cannot
                # provide once the surrogate ratio clips.
                if _kl_loss_active:
                    with torch.no_grad():
                        _kl_prior_logits, _ = _kl_anchor.prior.forward_ac(
                            states_t
                        )
                        _kl_prior_logp = F.log_softmax(
                            _kl_prior_logits.float(), dim=-1
                        )
                    _kl_pi_logp = F.log_softmax(logits.float(), dim=-1)
                    _kl_loss = (
                        _kl_prior_logp.exp() * (_kl_prior_logp - _kl_pi_logp)
                    ).sum(dim=-1).mean()
                    loss = loss + _kl_loss_coef * _kl_loss
                    _kl_loss_accum = _kl_loss_accum + _kl_loss.detach()
                    _kl_loss_n += 1
                # RND predictor loss: train the predictor to mimic
                # the frozen target on visited states (its params are
                # in this optimizer). Forward with grad here, unlike
                # the no-grad intrinsic-reward pass above.
                if t._rnd is not None and _rnd_update_this_mb:
                    # Cached frozen-target features (built once per
                    # iter above): the target CNN runs once per obs
                    # this iter instead of K times. The predictor half
                    # still forwards with grad, so the gradient path —
                    # and the loss value — are unchanged. Falls back to
                    # the full forward when the cache is absent
                    # (recurrent path / no valid steps).
                    if rnd_tgt_feat_cache is not None:
                        rnd_loss = t._rnd.predictor_loss(
                            states_t, rnd_tgt_feat_cache[mb_idx]
                        ).mean()
                    else:
                        rnd_loss = t._rnd(states_t).mean()
                    loss = loss + t.rnd_loss_coef * rnd_loss
                    _last_rnd_t = rnd_loss.detach()

                optimizer.zero_grad()
                # NaN backstop: clip_grad_norm_ is NOT one — a
                # non-finite loss yields clip_coef=NaN, which
                # multiplies EVERY gradient (net + RND, same
                # optimizer) to NaN and optimizer.step() then writes
                # NaN into all weights. Skip the whole step instead;
                # the rollout continues and the next minibatch
                # usually recovers. This subsumes the per-source
                # log-prob clamp as the general guard.
                if not torch.isfinite(loss):
                    log.error(
                        "[vanilla_ppo] non-finite loss (%s) — skipping "
                        "optimizer step this minibatch", loss.item(),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if sam_rho > 0.0:
                    # SHAPO (v7 report, three-pass shared-trunk
                    # variant). Pass A: policy(+entropy,+demo)
                    # gradient at θ over the actor set (trunk +
                    # actor head; the policy objective never
                    # touches the critic head). ε = ρ·g/‖g‖.
                    # Param set is static — resolve once per run
                    # (audit: the per-minibatch named_parameters()
                    # rescan cost a string-match traversal × 2,400
                    # minibatches/iter).
                    if self._sam_actor_params is None:
                        self._sam_actor_params = [
                            p for n, p in net.named_parameters()
                            if not n.startswith("critic")
                        ]
                    _actor_params = self._sam_actor_params
                    _pol_obj = (policy_loss
                                - t.entropy_coef * entropy)
                    if t._demo_bank is not None and _demo_coef > 0:
                        _pol_obj = _pol_obj + _demo_coef * _da_loss
                    _pol_obj.backward(retain_graph=False)
                    with torch.no_grad():
                        _gsq = None
                        for p in _actor_params:
                            if p.grad is not None:
                                _s = (p.grad.detach() ** 2).sum()
                                _gsq = _s if _gsq is None else _gsq + _s
                        _gn = torch.sqrt(_gsq) + 1e-12 \
                            if _gsq is not None else None
                        _eps_list = []
                        for p in _actor_params:
                            if p.grad is None or _gn is None:
                                _eps_list.append(None)
                                continue
                            _e = p.grad.detach() * (sam_rho / _gn)
                            _eps_list.append(_e)
                            p.add_(_e)
                    # Pass B: pessimistic policy gradient at θ+ε
                    # (value_coef=0 keeps the critic head out).
                    optimizer.zero_grad(set_to_none=True)
                    _lg2, _vp2 = net.forward_ac(states_t)
                    _l2, _pl2, _vl2, _en2 = ppo_losses(
                        _lg2, _vp2, actions_t, log_probs_old_t,
                        adv_t, target_t,
                        clip_eps=t.ppo_clip_eps,
                        value_coef=0.0,
                        entropy_coef=t.entropy_coef,
                        value_loss_kind=t.value_loss_kind,
                    )
                    if t._demo_bank is not None and _demo_coef > 0:
                        _dl2, _ = net.forward_ac(d_obs)
                        _l2 = _l2 + _demo_coef * demo_anchor_loss(
                            _dl2, d_act, margin=t.demo_anchor_margin
                        )
                    if not torch.isfinite(_l2):
                        with torch.no_grad():
                            for p, _e in zip(_actor_params, _eps_list):
                                if _e is not None:
                                    p.sub_(_e)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    _l2.backward()
                    with torch.no_grad():
                        for p, _e in zip(_actor_params, _eps_list):
                            if _e is not None:
                                p.sub_(_e)
                    # Pass C: critic (+RND) standard at restored θ;
                    # grads ACCUMULATE onto pass B's actor grads —
                    # trunk gets pessimistic-policy + standard-value
                    # contributions, matching baseline semantics.
                    _lg3, _vp3 = net.forward_ac(states_t)
                    _vp3 = _vp3.float()
                    if t.value_loss_kind == "mse":
                        _vloss3 = F.mse_loss(_vp3, target_t)
                    else:
                        _vloss3 = F.smooth_l1_loss(_vp3, target_t)
                    _c_loss = t.value_coef * _vloss3
                    if t._rnd is not None and _rnd_update_this_mb:
                        if rnd_tgt_feat_cache is not None:
                            _rl3 = t._rnd.predictor_loss(
                                states_t, rnd_tgt_feat_cache[mb_idx]
                            ).mean()
                        else:
                            _rl3 = t._rnd(states_t).mean()
                        _c_loss = _c_loss + t.rnd_loss_coef * _rl3
                        _last_rnd_t = _rl3.detach()
                    if not torch.isfinite(_c_loss):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    _c_loss.backward()
                    # clip_grad_norm_ returns the PRE-clip total norm —
                    # already computed for the clip itself, so reporting
                    # it is free (V29_STABILITY_2026-08-25.md F0).
                    _last_grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), t.reinforce_grad_clip
                    )
                    if _actor_frozen:
                        _kl_anchor.zero_actor_grads(net)
                    optimizer.step()
                else:
                    loss.backward()
                    _last_grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), t.reinforce_grad_clip
                    )
                    if _actor_frozen:
                        _kl_anchor.zero_actor_grads(net)
                    optimizer.step()

                _last_policy_t = policy_loss.detach()
                _last_value_t = value_loss.detach()
                _last_entropy_t = entropy.detach()
                _last_loss_t = loss.detach()
                _last_clip_frac_t = _mb_clip_frac_t
                _last_approx_kl_t = _mb_approx_kl_t
        # Single MPS sync for the final minibatch's scalars.
        last_clip_fraction = 0.0
        last_approx_kl = 0.0
        last_grad_norm = 0.0
        if _last_policy_t is not None:
            last_policy_loss = float(_last_policy_t.item())
            last_value_loss = float(_last_value_t.item())
            last_entropy = float(_last_entropy_t.item())
            last_loss = float(_last_loss_t.item())
        if _last_rnd_t is not None:
            last_rnd_loss = float(_last_rnd_t.item())
        if _last_clip_frac_t is not None:
            last_clip_fraction = float(_last_clip_frac_t.item())
            last_approx_kl = float(_last_approx_kl_t.item())
        if _last_grad_norm_t is not None:
            last_grad_norm = float(_last_grad_norm_t.item())
        t._gen_timer.add("update", time.perf_counter_ns() - _upd_t0)
        return {
            "reward_buf": reward_buf,
            "valid_flat": valid_flat,
            "valid_indices": valid_indices,
            "rnd_intrinsic_mean": rnd_intrinsic_mean,
            "count_bonus_mean": count_bonus_mean,
            "last_policy_loss": last_policy_loss,
            "last_value_loss": last_value_loss,
            "last_entropy": last_entropy,
            "last_loss": last_loss,
            "last_rnd_loss": last_rnd_loss,
            # V29_STABILITY_2026-08-25.md F0: the five missing scalars —
            # final-minibatch clip fraction / approx KL / grad norm
            # (trust-region pressure), and the once-per-iter advantage
            # mean/std + critic explained variance computed above the
            # K-epoch loop over the full valid batch.
            "last_clip_fraction": last_clip_fraction,
            "last_approx_kl": last_approx_kl,
            "last_grad_norm": last_grad_norm,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "explained_variance": explained_variance,
            "demo_coef": _demo_coef,
            "demo_loss_accum": _demo_loss_accum,
            "demo_loss_n": _demo_loss_n,
            "sil_loss_accum": (
                float(_sil_loss_accum) if _sil_loss_n else 0.0
            ),
            "sil_loss_n": _sil_loss_n,
            "kl_loss_coef": _kl_loss_coef,
            "kl_loss_accum": (
                float(_kl_loss_accum) if _kl_loss_n else 0.0
            ),
            "kl_loss_n": _kl_loss_n,
            "obs_all": obs_all,
            "obs_flat": obs_flat,
            "mb_size": mb_size,
        }
