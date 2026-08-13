"""C2 golden — the PPO-update pipeline anchor (pre-extraction for Task 2).

This is Task 0 / test C2 of `docs/proposals/trainer_decomposition_plan.md`:
the value-level safety net for the *update* half of `Trainer._run_vanilla_ppo`
(the intrinsic/count fold -> batched GAE -> advantage-norm-over-valid-mask ->
K-epoch minibatch update -> NaN backstop stretch, currently trainer.py
~7263-7689). When Task 2 lifts that stretch into a `PPOUpdater`, this golden
must stay green — a pure structural move reproduces every pinned number
bit-for-bit; a move that changes behavior does not.

How the fixture is produced (never hand-authored):

  * `tests/fixtures/char_ppo_update.pt` is recorded FROM THE REAL CODE by a
    single seeded tile+CPU `_run_vanilla_ppo` iteration under
    `CHAR_GOLDEN_REGEN=1`. Two spies capture the update's inputs and the real
    fold/GAE reference outputs at the collector->updater seam: `batched_gae`
    (introspecting its caller frame for the raw rollout buffers) and
    `fold_intrinsic_into_rewards` (its first call carries the raw extrinsic
    reward + the RND intrinsic vector). The captured hyperparameters are read
    off the live trainer at update time.
  * The recorded buffers are then fed through a self-contained updater harness
    (`_run_update_harness`) that mirrors the trainer's update glue line-for-line
    while calling the ALREADY-EXTRACTED pure functions from `src.training.ppo`
    (`fold_intrinsic_into_rewards`, `batched_gae`, `ppo_losses`) and the real
    `TileRND`. Its deterministic outputs are recorded alongside the inputs.

What is pinned (each assertion BITES if the pinned behavior drifts):

  1. ppo.py on REAL data: `fold_intrinsic_into_rewards` and `batched_gae`
     reproduce the trainer's actual folded reward buffer and GAE advantages /
     value-targets from the recorded real inputs, incl. the done-mask
     semantics (intrinsic zeroed on done steps, GAE boundary break).
  2. The harness update pipeline: the folded reward buffer, the raw + valid-
     mask-normalized advantages, the final-minibatch (policy_loss, value_loss,
     entropy, total loss, rnd_loss), and the post-update net checksum.
  3. THE §2.5 CONTRACT: `update_normalization` (RND obs-rms) is called EXACTLY
     ONCE — in the intrinsic fold, before the per-minibatch RND target-feature
     cache and predictor loss. This is the RND self-referential-norm /
     f16 double-norm bug surface; the runtime call-count and a source anchor
     both pin it.

Determinism basis (per the plan's §3 / the C0 golden): the tile path runs on
CPU and is fully seedable, so the harness trajectory is bit-reproducible. The
harness fixes its own seed and pins to 1 CPU thread (matching the trainer's
single-thread minibatch regime) so regen and check agree bit-for-bit. Floats
compare at rel/abs=1e-6 (harmless-reassociation guard); the net checksum is the
strict whole-update anchor.

Regenerate deliberately (only when adopting an intended behavior change) with:

    CHAR_GOLDEN_REGEN=1 .venv/bin/pytest tests/test_char_ppo_update.py

and review the fixture diff before committing. (Regen needs the SMB ROM; the
assertion path only needs the committed fixture and runs ROM-free.)

Explicitly NOT covered (documented honestly so nobody over-trusts this net):

  * The recurrent (GRU) update path (`_recurrent_ppo_update`) — this profile is
    feedforward; the harness asserts `not _recurrent` and mirrors only the
    feedforward branch. The recurrent branch is a separate anchor.
  * The demo-anchor (DQfD) term and the SHAPO/SAM three-pass variant — both are
    disabled in `mario_tiles_vanilla` (`_demo_bank is None`, `sam_rho == 0`), so
    the harness mirrors the plain `loss.backward()` path only.
  * The count-based frontier fold — `gx_count_bonus_coef` is unset in this
    profile (`_gx_count_beta == 0`), so that second fold is a no-op here; the
    fold helper's count semantics are covered by `test_ppo`.
  * The PR-MDP adversary update (`_adv_net is None` here).
  * Cross-machine checksum stability — the checksum is strict on the tile+CPU
    target; a non-target host may differ in the last ULPs (the float
    comparisons carry the 1e-6 tolerance for that case).
"""

from __future__ import annotations

import hashlib
import os
import queue as _queue
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"
_FIXTURE = ROOT / "tests" / "fixtures" / "char_ppo_update.pt"

# Capture-run geometry (tiny + seeded; matches the C0 master golden so the
# recorded rollout is the same shape/scale the whole-loop anchor sees).
_SEED = 1234
_ROLLOUT_STEPS = 32
_NUM_ENVS = 2
_MINIBATCH = 16
_MAX_EP_STEPS = 200

# The harness re-seeds to a fixed value of its own so its net/RND init and
# minibatch permutations are reproducible independent of the capture run.
_HARNESS_SEED = 20260813

# Hyperparameters read off the live trainer at update time and replayed by the
# harness. Names mirror the trainer attributes the update stretch reads.
_PARAM_KEYS = (
    "_is_tile_mode", "_recurrent", "preprocess_f16", "num_actions",
    "_tile_feature_dim", "_tile_hidden_dim", "_tile_trunk_dim",
    "reinforce_lr", "reinforce_gamma", "gae_lambda", "reinforce_steps",
    "reinforce_grad_clip", "ppo_clip_eps", "value_coef", "value_loss_kind",
    "entropy_coef", "rnd_intrinsic_coef", "rnd_loss_coef",
    "rnd_predictor_update_fraction", "_gx_count_beta", "ppo_minibatch_size",
)


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _net_checksum(net) -> str:
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Bare updater trainer + the self-contained update-pipeline harness.
# ---------------------------------------------------------------------------
def _bare_updater_trainer(params: dict):
    """A `Trainer` shell carrying only the attributes the vanilla_ppo update
    stretch reads — no pool / ROM / emulator booted."""
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.device = torch.device("cpu")
    t._demo_bank = None
    t._rnd = None
    for k in _PARAM_KEYS:
        setattr(t, k, params[k])
    return t


def _run_update_harness(params: dict, inputs: dict) -> dict:
    """Mirror of trainer.py's vanilla_ppo update stretch (~7263-7689), calling
    the extracted `src.training.ppo` functions + the real `TileRND`. Fully
    self-contained and seeded: builds a fresh net/RND/optimizer, folds
    intrinsic, runs GAE + valid-mask advantage norm, then the K-epoch
    minibatch update with the RND target cache + NaN backstop.

    Returns the pinned outputs plus the observed obs-rms update-call count.
    Only the feedforward, no-demo, no-SAM path (this profile) is mirrored.
    """
    from src.models.tile_rnd import TileRND
    from src.training.ppo import (
        batched_gae, fold_intrinsic_into_rewards, ppo_losses,
    )

    assert not params["_recurrent"], "harness mirrors the feedforward path only"

    _threads0 = torch.get_num_threads()
    torch.set_num_threads(1)  # match the trainer's single-thread minibatch regime
    try:
        _seed_all(_HARNESS_SEED)
        t = _bare_updater_trainer(params)
        net = t._make_network()
        net.to("cpu")
        # Build order matches the trainer: net, then RND, then the optimizer
        # (which folds in the RND predictor's params).
        t._rnd = TileRND(feature_dim=t._tile_feature_dim).to("cpu")
        optimizer = t._build_ppo_optimizer(net)

        # Instrument the RND obs-rms update so the §2.5 "exactly once" contract
        # is checked at runtime (a double-norm regression trips it).
        _obs_rms_calls = {"n": 0}
        _orig_update_norm = t._rnd.update_normalization

        def _counting_update_norm(obs, err):
            _obs_rms_calls["n"] += 1
            return _orig_update_norm(obs, err)

        t._rnd.update_normalization = _counting_update_norm

        obs_buf = inputs["obs_buf"]
        action_buf = inputs["action_buf"]
        reward_buf = inputs["reward_buf"].copy()  # fold reassigns; keep input pure
        value_buf = inputs["value_buf"]
        log_prob_buf = inputs["log_prob_buf"]
        done_buf = inputs["done_buf"]
        valid_buf = inputs["valid_buf"]
        bonus_buf = inputs["bonus_buf"]
        final_values_np = inputs["final_values"]

        rollout_steps, num_envs = reward_buf.shape
        obs_shape = tuple(obs_buf.shape[2:])

        # ---- RND intrinsic fold (obs-rms updated ONCE here) ----
        if t._rnd is not None:
            with torch.no_grad():
                rnd_obs_t = (
                    torch.from_numpy(
                        obs_buf.reshape((rollout_steps * num_envs,) + obs_shape)
                    ).to(t.device).float()
                )
                if not t._is_tile_mode and not t.preprocess_f16:
                    rnd_obs_t = rnd_obs_t.div_(255.0)
                intrinsic_raw = t._rnd(rnd_obs_t)
                bonus_t = t._rnd.normalize_bonus(intrinsic_raw)
                t._rnd.update_normalization(rnd_obs_t, intrinsic_raw)
                intrinsic_np = (
                    bonus_t.cpu().numpy().astype(np.float32) * t.rnd_intrinsic_coef
                ).reshape(rollout_steps, num_envs)
            reward_buf = fold_intrinsic_into_rewards(reward_buf, intrinsic_np, done_buf)

        # ---- count-bonus fold (no-op here: _gx_count_beta == 0) ----
        if t._gx_count_beta > 0.0:
            reward_buf = fold_intrinsic_into_rewards(reward_buf, bonus_buf, done_buf)

        folded_reward = reward_buf.copy()

        # ---- GAE-lambda backward sweep ----
        advantages, value_targets = batched_gae(
            reward_buf, value_buf, done_buf, final_values_np,
            t.reinforce_gamma, t.gae_lambda,
        )

        # ---- advantage normalization over the valid mask ----
        valid_flat = valid_buf.reshape(-1)
        _valid_adv = advantages.reshape(-1)[valid_flat]
        if _valid_adv.size > 1:
            adv_mean = float(_valid_adv.mean())
            adv_std = float(_valid_adv.std()) + 1e-8
        else:
            adv_mean, adv_std = 0.0, 1.0
        advantages_norm = (advantages - adv_mean) / adv_std

        # ---- K-epoch minibatch update ----
        total_n = rollout_steps * num_envs
        obs_flat = obs_buf.reshape((total_n,) + obs_shape)
        action_flat = action_buf.reshape(-1)
        log_prob_old_flat = log_prob_buf.reshape(-1)
        adv_flat = advantages_norm.reshape(-1)
        target_flat = value_targets.reshape(-1)
        valid_indices = np.where(valid_flat)[0]

        net.train()
        last_policy_loss = last_value_loss = last_entropy = last_loss = 0.0
        last_rnd_loss = 0.0
        _last_policy_t = _last_value_t = _last_entropy_t = None
        _last_loss_t = _last_rnd_t = None
        mb_size = max(1, t.ppo_minibatch_size)

        actions_all = torch.from_numpy(action_flat.astype(np.int64)).to(t.device)
        log_probs_old_all = torch.from_numpy(log_prob_old_flat).to(t.device).float()
        adv_all = torch.from_numpy(adv_flat).to(t.device).float()
        target_all = torch.from_numpy(target_flat).to(t.device).float()
        obs_all = (
            torch.from_numpy(obs_flat).to(t.device).float()
            if (t._is_tile_mode and not t._recurrent) else None
        )

        # RND target-feature cache (built once per iter, post-obs-rms-update).
        rnd_tgt_feat_cache = None
        if t._rnd is not None and not t._recurrent and valid_indices.size:
            rnd_tgt_feat_cache = torch.zeros(
                total_n, t._rnd.feat_dim, device=t.device
            )
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

        _rnd_pred_stride = max(
            1, int(round(1.0 / t.rnd_predictor_update_fraction))
        )
        _rnd_mb_index = 0

        for _epoch in range(0 if t._recurrent else t.reinforce_steps):
            perm = np.random.permutation(valid_indices)
            n_valid = perm.shape[0]
            for mb_start in range(0, n_valid, mb_size):
                mb_end = min(mb_start + mb_size, n_valid)
                mb_np = perm[mb_start:mb_end]
                if mb_np.size < 2:
                    continue
                mb_idx = torch.from_numpy(mb_np).to(t.device)
                _rnd_update_this_mb = (_rnd_mb_index % _rnd_pred_stride == 0)
                _rnd_mb_index += 1

                if obs_all is not None:
                    states_t = obs_all[mb_idx]
                else:
                    states_t = torch.from_numpy(
                        np.ascontiguousarray(obs_flat[mb_np])
                    ).to(t.device).float()
                    if not t.preprocess_f16:
                        states_t = states_t.div_(255.0)
                actions_t = actions_all[mb_idx]
                log_probs_old_t = log_probs_old_all[mb_idx]
                adv_t = adv_all[mb_idx]
                target_t = target_all[mb_idx]

                logits, values_pred = net.forward_ac(states_t)
                loss, policy_loss, value_loss, entropy = ppo_losses(
                    logits, values_pred, actions_t, log_probs_old_t,
                    adv_t, target_t,
                    clip_eps=t.ppo_clip_eps,
                    value_coef=t.value_coef,
                    entropy_coef=t.entropy_coef,
                    value_loss_kind=t.value_loss_kind,
                )
                if t._rnd is not None and _rnd_update_this_mb:
                    if rnd_tgt_feat_cache is not None:
                        rnd_loss = t._rnd.predictor_loss(
                            states_t, rnd_tgt_feat_cache[mb_idx]
                        ).mean()
                    else:
                        rnd_loss = t._rnd(states_t).mean()
                    loss = loss + t.rnd_loss_coef * rnd_loss
                    _last_rnd_t = rnd_loss.detach()

                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), t.reinforce_grad_clip)
                optimizer.step()

                _last_policy_t = policy_loss.detach()
                _last_value_t = value_loss.detach()
                _last_entropy_t = entropy.detach()
                _last_loss_t = loss.detach()

        if _last_policy_t is not None:
            last_policy_loss = float(_last_policy_t.item())
            last_value_loss = float(_last_value_t.item())
            last_entropy = float(_last_entropy_t.item())
            last_loss = float(_last_loss_t.item())
        if _last_rnd_t is not None:
            last_rnd_loss = float(_last_rnd_t.item())

        return {
            "folded_reward": folded_reward,
            "advantages": advantages,
            "advantages_norm": advantages_norm.astype(np.float32),
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "loss": last_loss,
            "rnd_loss": last_rnd_loss,
            "net_checksum": _net_checksum(net),
            "obs_rms_update_count": _obs_rms_calls["n"],
        }
    finally:
        torch.set_num_threads(_threads0)


# ---------------------------------------------------------------------------
# Fixture regeneration: one REAL seeded tile+CPU iteration, spies at the
# collector->updater seam capture the raw buffers + the real fold/GAE outputs.
# ---------------------------------------------------------------------------
def _capture_real_iteration() -> dict:
    """Run one real vanilla_ppo iteration and capture the update inputs + the
    real fold/GAE reference outputs. Requires the SMB ROM/profile."""
    import src.training.ppo_updater as updater_mod
    from src.training.trainer import Trainer

    _seed_all(_SEED)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = _ROLLOUT_STEPS
    profile["reinforce"]["ppo_minibatch_size"] = _MINIBATCH
    profile["reinforce"]["device"] = "cpu"

    cap: dict = {}
    # The fold + protagonist GAE moved from `_run_vanilla_ppo` into
    # `PPOUpdater.update` (trainer-decomposition plan, Task 2), so the spies
    # patch the names in the ppo_updater module (where they are now called).
    orig_fold = updater_mod.fold_intrinsic_into_rewards
    orig_gae = updater_mod.batched_gae

    def _fold_spy(reward_buf, intrinsic, done_buf, *a, **k):
        out = orig_fold(reward_buf, intrinsic, done_buf, *a, **k)
        if "raw_reward" not in cap:  # first call == the RND intrinsic fold
            cap["raw_reward"] = np.array(reward_buf, copy=True)
            cap["intrinsic_np"] = np.array(intrinsic, copy=True)
            cap["real_folded_reward"] = np.array(out, copy=True)
        return out

    def _gae_spy(reward_buf, value_buf, done_buf, final_values, gamma, lam, *a, **k):
        adv, vt = orig_gae(
            reward_buf, value_buf, done_buf, final_values, gamma, lam, *a, **k
        )
        if "gae_adv" not in cap:  # first (protagonist) GAE call
            cap["gae_adv"] = np.array(adv, copy=True)
            cap["gae_vt"] = np.array(vt, copy=True)
            # Snapshot the live rollout buffers + hyperparameters from the
            # caller frame (PPOUpdater.update) — everything the update reads.
            # The buffers are `update`'s keyword params; the hyperparameters
            # live on the owning Trainer, reached via the updater's `.trainer`.
            fr = sys._getframe(1)
            lv = fr.f_locals
            t = lv["self"].trainer
            cap["inputs"] = {
                "obs_buf": np.array(lv["obs_buf"], copy=True),
                "action_buf": np.array(lv["action_buf"], copy=True),
                "reward_buf": cap["raw_reward"],  # pre-fold extrinsic
                "value_buf": np.array(lv["value_buf"], copy=True),
                "log_prob_buf": np.array(lv["log_prob_buf"], copy=True),
                "done_buf": np.array(lv["done_buf"], copy=True),
                "valid_buf": np.array(lv["valid_buf"], copy=True),
                "bonus_buf": np.array(lv["bonus_buf"], copy=True),
                "final_values": np.array(lv["final_values_np"], copy=True),
            }
            cap["params"] = {key: getattr(t, key) for key in _PARAM_KEYS}
        return adv, vt

    updater_mod.fold_intrinsic_into_rewards = _fold_spy
    updater_mod.batched_gae = _gae_spy
    try:
        metrics_q: _queue.Queue = _queue.Queue()
        with tempfile.TemporaryDirectory(prefix="c2_capture_") as tmp:
            trainer = Trainer(
                rom_path=str(_SMB_ROM),
                game_profile=profile,
                num_instances=_NUM_ENVS,
                population_size=_NUM_ENVS,
                checkpoint_dir=tmp,
                start_state_path=profile.get("start_state_path"),
                env_spec="nes_core",
                max_episode_steps=_MAX_EP_STEPS,
                metrics_queue=metrics_q,
                device_override="cpu",
                seed=_SEED,
            )
            trainer.run(num_generations=1, resume_from=None, fresh_start=True)
    finally:
        updater_mod.fold_intrinsic_into_rewards = orig_fold
        updater_mod.batched_gae = orig_gae

    assert "inputs" in cap, "the update never ran — no GAE call captured"
    return cap


def _regen_fixture() -> None:
    cap = _capture_real_iteration()
    harness = _run_update_harness(cap["params"], cap["inputs"])
    payload = {
        "params": cap["params"],
        "inputs": cap["inputs"],
        "real_fold_gae": {
            "raw_reward": cap["raw_reward"],
            "intrinsic_np": cap["intrinsic_np"],
            "folded_reward": cap["real_folded_reward"],
            "gae_adv": cap["gae_adv"],
            "gae_vt": cap["gae_vt"],
            "gamma": float(cap["params"]["reinforce_gamma"]),
            "gae_lambda": float(cap["params"]["gae_lambda"]),
        },
        "harness_out": harness,
    }
    _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(_FIXTURE))


def _load_fixture() -> dict:
    return torch.load(str(_FIXTURE), weights_only=False)


@pytest.fixture(scope="module")
def golden() -> dict:
    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        if not (_SMB_ROM.exists() and _PROFILE.exists()):
            pytest.skip("CHAR_GOLDEN_REGEN=1 but SMB ROM / profile not present")
        _regen_fixture()
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")
    if not _FIXTURE.exists():
        pytest.skip(
            f"golden fixture missing: {_FIXTURE}. Generate it with "
            f"CHAR_GOLDEN_REGEN=1 pytest {Path(__file__).name} (needs the SMB ROM)."
        )
    return _load_fixture()


# ---------------------------------------------------------------------------
# 1. ppo.py pure functions reproduce the trainer's REAL fold + GAE on the
#    recorded real inputs. (No ROM: reads the committed fixture only.)
# ---------------------------------------------------------------------------
def test_fold_and_gae_reproduce_real_trainer_outputs(golden) -> None:
    from src.training.ppo import batched_gae, fold_intrinsic_into_rewards

    rfg = golden["real_fold_gae"]
    raw = rfg["raw_reward"]
    intrinsic = rfg["intrinsic_np"]
    done = golden["inputs"]["done_buf"]

    # fold_intrinsic_into_rewards on the real captured raw reward + RND
    # intrinsic must reproduce the trainer's folded reward buffer, byte-tight.
    got_folded = fold_intrinsic_into_rewards(raw, intrinsic, done)
    np.testing.assert_allclose(
        got_folded, rfg["folded_reward"], rtol=1e-6, atol=1e-6,
        err_msg="fold_intrinsic_into_rewards drifted from the trainer's folded reward",
    )
    # Done-mask semantics: intrinsic is zeroed on done steps; added on live ones.
    if done.any():
        np.testing.assert_allclose(
            got_folded[done], raw[done], rtol=1e-6, atol=1e-6,
            err_msg="intrinsic leaked onto done/padded steps",
        )
    live = ~done
    if live.any():
        np.testing.assert_allclose(
            got_folded[live], (raw + intrinsic)[live], rtol=1e-6, atol=1e-6,
            err_msg="intrinsic not added on live steps",
        )

    # batched_gae on the real folded reward reproduces the trainer's GAE
    # advantages AND value-targets, with value_targets == advantages + value.
    adv, vt = batched_gae(
        rfg["folded_reward"], golden["inputs"]["value_buf"], done,
        golden["inputs"]["final_values"], rfg["gamma"], rfg["gae_lambda"],
    )
    np.testing.assert_allclose(
        adv, rfg["gae_adv"], rtol=1e-6, atol=1e-6,
        err_msg="batched_gae advantages drifted from the trainer's GAE sweep",
    )
    np.testing.assert_allclose(
        vt, rfg["gae_vt"], rtol=1e-6, atol=1e-6,
        err_msg="batched_gae value_targets drifted from the trainer's GAE sweep",
    )
    np.testing.assert_allclose(
        vt, adv + golden["inputs"]["value_buf"], rtol=1e-6, atol=1e-6,
        err_msg="value_targets != advantages + value_buf",
    )


# ---------------------------------------------------------------------------
# 2. The update-pipeline harness: folded reward + advantages + losses +
#    post-update net checksum are pinned to the recorded reference.
# ---------------------------------------------------------------------------
def test_update_pipeline_matches_golden(golden) -> None:
    out = _run_update_harness(golden["params"], golden["inputs"])
    want = golden["harness_out"]

    np.testing.assert_allclose(
        out["folded_reward"], want["folded_reward"], rtol=1e-6, atol=1e-6,
        err_msg="harness folded reward buffer changed",
    )
    np.testing.assert_allclose(
        out["advantages"], want["advantages"], rtol=1e-6, atol=1e-6,
        err_msg="harness GAE advantages changed",
    )
    np.testing.assert_allclose(
        out["advantages_norm"], want["advantages_norm"], rtol=1e-6, atol=1e-6,
        err_msg="harness valid-mask-normalized advantages changed",
    )
    for key in ("policy_loss", "value_loss", "entropy", "loss", "rnd_loss"):
        assert out[key] == pytest.approx(want[key], rel=1e-6, abs=1e-6), (
            f"harness {key} changed: {out[key]!r} vs golden {want[key]!r} — the "
            f"update pipeline's behavior drifted. If intended, regenerate the "
            f"fixture (CHAR_GOLDEN_REGEN=1) and review the diff."
        )
    # Strict whole-update anchor: every post-update weight bit-identical.
    assert out["net_checksum"] == want["net_checksum"], (
        f"post-update net checksum changed: {out['net_checksum']} vs golden "
        f"{want['net_checksum']}. The K-epoch update diverged — a structural "
        f"PPOUpdater extraction must be behavior-preserving."
    )


# ---------------------------------------------------------------------------
# 3. §2.5 contract — RND obs-rms updated EXACTLY ONCE, before the per-minibatch
#    RND cache/predictor loss (the double-norm bug surface).
# ---------------------------------------------------------------------------
def test_obs_rms_updated_exactly_once(golden) -> None:
    """The intrinsic fold calls `update_normalization` once; the per-minibatch
    RND target-feature cache + predictor loss must NOT touch obs-rms again, or
    the frozen-target cache goes stale and the intrinsic bonus self-references
    (the RND double-norm regression). Pinned at runtime AND recorded."""
    out = _run_update_harness(golden["params"], golden["inputs"])
    assert out["obs_rms_update_count"] == 1, (
        f"RND obs-rms was updated {out['obs_rms_update_count']}x this update — "
        f"the §2.5 fold-once-before-cache contract broke (double-norm surface)."
    )
    # The regen-time count was recorded too; hold them equal so the fixture and
    # the live harness can never silently disagree on this invariant.
    assert out["obs_rms_update_count"] == golden["harness_out"]["obs_rms_update_count"]


def test_obs_rms_once_and_pipeline_order_anchored_in_source() -> None:
    """Anchor the §2.5 contract + the fold->GAE->K-epoch order in the real
    update code, so the harness above cannot silently drift from the code it
    characterizes (no ROM needed).

    The update stretch was lifted verbatim from `Trainer._run_vanilla_ppo` into
    `PPOUpdater.update` (trainer-decomposition plan, Task 2), so this anchor now
    reads `src/training/ppo_updater.py`. Every assertion is unchanged in strength
    -- only the anchored file (and the `self.`->`t.` trainer-ref form the move
    introduced) moved with the code."""
    src = (ROOT / "src" / "training" / "ppo_updater.py").read_text()
    start = src.find("def update(")
    assert start >= 0, "PPOUpdater.update not found"
    body = src[start:]

    # obs-rms is updated exactly once in the whole update body.
    assert body.count("update_normalization") == 1, (
        "the updater now touches the RND obs-rms a different number of times — "
        "the fold-once contract (plan §2.5) changed; update this test deliberately."
    )
    # The update stretch exists and is ordered fold -> GAE -> K-epoch, with the
    # single obs-rms update ahead of the K-epoch loop.
    i_norm = body.find("update_normalization")
    i_fold = body.find("fold_intrinsic_into_rewards")
    i_gae = body.find("batched_gae")
    i_kepoch = body.find(
        "for epoch in range(0 if t._recurrent else t.reinforce_steps)"
    )
    assert -1 < i_norm < i_fold < i_gae < i_kepoch, (
        "the fold -> GAE -> K-epoch update order (or the obs-rms-before-cache "
        "ordering) changed in the updater."
    )
    # The NaN backstop that skips the optimizer step still guards the loop.
    assert "if not torch.isfinite(loss):" in body
    assert "optimizer.zero_grad(set_to_none=True)" in body
