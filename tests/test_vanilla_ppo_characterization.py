"""Characterization tests for the vanilla PPO trainer.

These pin the *current* behavior of `Trainer._run_vanilla_ppo` and its
guards so a later structural extraction (see
`docs/proposals/trainer_decomposition_plan.md`) can be proven
behavior-preserving. They are the safe FIRST step of that plan — the
safety net that must stay green before and after every extraction.

Design constraints followed (per the plan's §3 determinism basis and
this task's brief):

  * tile mode (`configs/mario_tiles_vanilla.yaml`) on `device=cpu` —
    the tile path is seedable and CPU-bound, unlike the pixel path
    (MPS `multinomial` is not bit-reproducible on the target M4).
  * tiny config: num_envs <= 4, one iteration, shrunk rollout.
  * runtime budget << 60s (the full-iteration run is ~0.2s locally).

What these tests assert are STRUCTURAL / BEHAVIORAL invariants and the
just-added guards, NOT exact float values that legitimately vary
run-to-run:

  1. metrics-dict: the canonical dashboard keys are present and each is
     a finite number of the right Python/NumPy type; entropy is bounded.
  2. reward buffer: shape `(rollout_steps, num_envs)` and dtype float32
     handed to GAE; the done mask is bool.
  3. NaN backstop: a non-finite loss skips the optimizer step (weights
     preserved); a finite loss does take the step.
  4. `_safe_sample_from_logits`: NaN/Inf rows fall back to a uniform
     distribution with finite log-probs, on CPU.
  5. sticky log-prob clamp floor: a stuck near-zero-probability action's
     recorded log-prob is floored at -13.0.
  6. checkpoint payload round-trips: save -> resume load -> identical
     net + optimizer-offset + RND state.

Explicitly NOT covered (documented honestly so nobody over-trusts this
net): exact loss/return values, GAE numerics, the curriculum advance /
capture gate, Go-Explore, the consolidation/ladder modes, the recurrent
(GRU) path, the anti-collapse guard, and warm-start partitioning. Those
are the C0/C3/C4/C5 goldens the plan schedules separately.
"""

from __future__ import annotations

import math
import queue as _queue
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"

# Real smb_tiles feature dim (175) x the profile's 4-frame stack.
_TILE_FEATURE_DIM = 175 * 4
_NUM_ACTIONS = 6  # mario_tiles_vanilla.yaml action_space length


def _seed_everything(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(
        x, bool
    )


# ---------------------------------------------------------------------------
# Group 4 — `_safe_sample_from_logits` (the sampling-layer NaN backstop).
# Pure function, no ROM/pool. Always runs.
# ---------------------------------------------------------------------------
def test_safe_sample_from_nonfinite_logits_never_crashes_and_stays_finite() -> None:
    """Pin the ACTUAL sanitize behavior (nan_to_num -> clamp(-50, 50) ->
    log_softmax), which is subtler than "everything becomes uniform":

      * a NaN logit is zeroed (nan_to_num nan=0.0);
      * a lone +Inf is clamped to +50 and DOMINATES its row — it does NOT
        collapse to uniform, because after sanitize the row is a finite,
        valid distribution and never reaches the `bad_rows` fallback;
      * an all-(-Inf) row sanitizes to all-equal -> uniform;
      * every returned log-prob is finite (the crash guard's real job),
        three CPU tensors of the right shape, and finite rows are intact.

    (See honest_note: the explicit uniform `bad_rows` substitution is
    effectively unreachable through logits alone — nan_to_num+clamp always
    yields a finite distribution first. The guarantee that survives is
    finite log-probs, never a `multinomial` crash.)"""
    from src.training.trainer import _safe_sample_from_logits

    _seed_everything()
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 0.0, 0.0],           # 0: normal row
            [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0],  # 1: NaN -> zeroed
            [float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0],  # 2: +Inf -> clamped +50
            [-float("inf")] * 6,                       # 3: all -Inf -> uniform
        ]
    )

    out = _safe_sample_from_logits(logits)
    assert len(out) == 3, "expected (sampled, chosen_lp, log_probs_all)"
    sampled, chosen_lp, log_probs_all = out

    # Shapes + CPU (the helper pulls device->host once).
    assert sampled.shape == (4,)
    assert chosen_lp.shape == (4,)
    assert log_probs_all.shape == (4, _NUM_ACTIONS)
    assert sampled.device.type == "cpu"
    assert log_probs_all.device.type == "cpu"

    # The core guarantee: no non-finite log-prob escapes (this is what
    # keeps torch.multinomial from raising and killing the run).
    assert torch.isfinite(chosen_lp).all()
    assert torch.isfinite(log_probs_all).all()
    assert int(sampled.min()) >= 0 and int(sampled.max()) < _NUM_ACTIONS

    uniform_lp = math.log(1.0 / _NUM_ACTIONS)

    # Row 0 (clean): argmax stays on the largest logit (action 3).
    assert int(log_probs_all[0].argmax()) == 3

    # Row 1 (NaN -> 0): all logits become 0 -> uniform.
    assert torch.allclose(
        log_probs_all[1], torch.full((_NUM_ACTIONS,), uniform_lp), atol=1e-5
    )

    # Row 2 (+Inf clamped to +50): DOMINATES action 0, NOT uniform.
    assert int(log_probs_all[2].argmax()) == 0
    assert float(log_probs_all[2, 0]) > uniform_lp

    # Row 3 (all -Inf -> all -50): uniform.
    assert torch.allclose(
        log_probs_all[3], torch.full((_NUM_ACTIONS,), uniform_lp), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Group 3 — the PPO-update NaN backstop (loss-level skip).
# The guard is inline in `_run_vanilla_ppo`; its documented behavior is
# replicated here in isolation and pinned to the source text so a change
# to the guard trips this test. No ROM/pool.
# ---------------------------------------------------------------------------
def _guarded_optimizer_step(net, optimizer, loss, grad_clip: float) -> bool:
    """Mirror of the vanilla_ppo minibatch guard (trainer.py ~6219-6239):

        optimizer.zero_grad()
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue            # -> return False (step skipped)
        loss.backward()
        clip_grad_norm_(...)
        optimizer.step()        # -> return True (step taken)

    Returns True iff the optimizer step was taken.
    """
    optimizer.zero_grad()
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    return True


def test_nan_backstop_skips_optimizer_step_on_nonfinite_loss() -> None:
    """Pin: a non-finite loss skips the whole optimizer step so weights are
    preserved (a NaN otherwise multiplies through clip_grad_norm_ into
    every parameter). A finite loss DOES step."""
    _seed_everything()
    net = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    before = {k: v.detach().clone() for k, v in net.state_dict().items()}

    # NaN loss -> step skipped -> weights byte-identical.
    x = torch.randn(8, 4)
    nan_loss = net(x).sum() * float("nan")
    took_step = _guarded_optimizer_step(net, optimizer, nan_loss, grad_clip=0.5)
    assert took_step is False
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"NaN loss must not mutate {k}"

    # Inf loss -> also skipped.
    inf_loss = net(x).sum() * float("inf")
    assert _guarded_optimizer_step(net, optimizer, inf_loss, grad_clip=0.5) is False
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"Inf loss must not mutate {k}"

    # Finite loss -> step taken -> at least one weight changes.
    finite_loss = net(x).pow(2).mean()
    assert _guarded_optimizer_step(net, optimizer, finite_loss, grad_clip=0.5) is True
    changed = any(
        not torch.equal(v, before[k]) for k, v in net.state_dict().items()
    )
    assert changed, "a finite loss must actually update the network"


def test_nan_backstop_guard_present_in_trainer_source() -> None:
    """Anchor the guard's existence in the update code so the isolated
    replication above cannot silently drift from the real code.

    The minibatch update (with this NaN backstop) was lifted verbatim from
    `Trainer._run_vanilla_ppo` into `PPOUpdater.update` (trainer-decomposition
    plan, Task 2), so this anchor now reads `src/training/ppo_updater.py`. The
    assertions are unchanged."""
    src = (ROOT / "src" / "training" / "ppo_updater.py").read_text()
    start = src.find("def update(")
    assert start >= 0, "PPOUpdater.update not found"
    body = src[start:]
    assert "if not torch.isfinite(loss):" in body, (
        "the vanilla_ppo minibatch NaN backstop is gone or was reworded — "
        "update this characterization test deliberately."
    )
    assert "optimizer.zero_grad(set_to_none=True)" in body


# ---------------------------------------------------------------------------
# Group 5 — sticky-action log-prob clamp floor (-13.0).
# The clamp is inline in the rollout (trainer.py ~5431); its arithmetic is
# pinned here and anchored to the source constant. No ROM/pool.
# ---------------------------------------------------------------------------
def test_sticky_logprob_clamp_floor_at_minus_13() -> None:
    """Pin: when a sticky override reuses the previous action, its recorded
    log-prob is `clamp(log_prob, min=-13.0)`. A near-deterministic policy
    gives a stuck action ~0 probability; the unclamped log-prob (-30..-46)
    would explode PPO's importance ratio into NaN. The floor caps it."""
    _seed_everything()
    # Peaked distribution: action 0 dominates, action 5 has ~0 probability
    # (log-prob well below -13).
    logits = torch.tensor([[40.0, 0.0, 0.0, 0.0, 0.0, -40.0]])
    log_probs_all = torch.log_softmax(logits, dim=-1)

    stuck_action = 5  # the vanishingly-improbable one
    raw_lp = log_probs_all[0, stuck_action]
    assert raw_lp < -13.0, "test setup: stuck action must be below the floor"

    # Exact clamp the trainer applies.
    floored = torch.clamp(log_probs_all[[0], torch.tensor([stuck_action])], min=-13.0)
    assert float(floored) == pytest.approx(-13.0), (
        "a sub-floor sticky log-prob must be clamped up to exactly -13.0"
    )

    # An action already above the floor is left untouched. Action 0 is the
    # dominant one here (log-prob ~= 0), comfortably above -13.
    above_action = 0
    above_lp = log_probs_all[0, above_action]
    assert above_lp > -13.0, "test setup: dominant action must be above the floor"
    unchanged = torch.clamp(
        log_probs_all[[0], torch.tensor([above_action])], min=-13.0
    )
    assert float(unchanged) == pytest.approx(float(above_lp), abs=1e-5), (
        "an above-floor log-prob must pass through the clamp unchanged"
    )


def test_sticky_clamp_floor_constant_present_in_trainer_source() -> None:
    """Anchor the -13.0 floor constant in the vanilla rollout so the value
    can't be edited without deliberately updating this test."""
    src = (ROOT / "src" / "training" / "trainer.py").read_text()
    assert "min=-13.0" in src, (
        "the sticky log-prob clamp floor (-13.0) changed or vanished."
    )


# ---------------------------------------------------------------------------
# Group 6 — checkpoint payload round-trip.
# Builds a real tile net + optimizer, writes the exact payload the save
# block emits, then loads it back through the REAL resume path
# (`_maybe_resume_vanilla_ppo`) and asserts identity. No ROM/pool needed.
# ---------------------------------------------------------------------------
def _bare_tile_trainer(tmp: Path):
    """A `Trainer` shell with just the attributes the net build + resume
    path touch — no pool / ROM / emulator booted."""
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t._is_tile_mode = True
    t._recurrent = False
    t.num_actions = _NUM_ACTIONS
    t._tile_feature_dim = _TILE_FEATURE_DIM
    t._tile_hidden_dim = 64
    t._tile_trunk_dim = 32
    t._rnd = None
    t.device = torch.device("cpu")
    t.reinforce_lr = 3.0e-4
    t.checkpoint_dir = Path(tmp)
    t._pending_rnd_state = None
    t._pending_anticollapse = None
    t._gx_counts = {}
    t._vppo_resumed_from_iter = None
    return t


def test_checkpoint_payload_roundtrips_net_and_offset() -> None:
    """Pin: the save->load round-trip restores byte-identical net weights,
    the absolute iteration offset, and (when present) the RND state — the
    resume-safety contract the C1 golden protects."""
    _seed_everything()
    with tempfile.TemporaryDirectory(prefix="vppo_char_ckpt_") as tmp:
        t = _bare_tile_trainer(Path(tmp))

        # Source net perturbed away from init so equality is meaningful.
        src_net = t._make_network()
        src_net.to("cpu")
        with torch.no_grad():
            for p in src_net.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        src_opt = t._build_ppo_optimizer(src_net)
        # One step so the optimizer accrues real moment state.
        loss = sum(p.pow(2).sum() for p in src_net.parameters())
        src_opt.zero_grad()
        loss.backward()
        src_opt.step()

        # Payload constructed exactly as the trainer's save block does
        # (trainer.py ~7482): cpu-detached net state + optimizer state +
        # absolute iter. (The save block is inline, not a callable, so we
        # mirror its dict; the LOAD path below is the real trainer code.)
        payload = {
            "iter": 10,
            "net_state_dict": {
                k: v.detach().cpu() for k, v in src_net.state_dict().items()
            },
            "optimizer_state_dict": src_opt.state_dict(),
        }
        ckpt = Path(tmp) / "vanilla_ppo_iter_00010.pt"
        torch.save(payload, str(ckpt))

        # Fresh net (different random weights) + fresh optimizer.
        _seed_everything(4321)
        fresh_net = t._make_network()
        fresh_net.to("cpu")
        # Sanity: fresh weights differ from source before the load.
        assert any(
            not torch.equal(fresh_net.state_dict()[k], src_net.state_dict()[k])
            for k in src_net.state_dict()
        ), "test setup: fresh net must start different from the source net"
        fresh_opt = t._build_ppo_optimizer(fresh_net)

        # REAL resume path.
        iter_offset = t._maybe_resume_vanilla_ppo(
            fresh_net, fresh_opt, fresh_start=False
        )

        assert iter_offset == 10, "resume must recover the absolute iter offset"
        for k in src_net.state_dict():
            assert torch.equal(fresh_net.state_dict()[k], src_net.state_dict()[k]), (
                f"net param {k} did not round-trip identically"
            )
        assert t._vppo_resumed_from_iter == 10


def test_checkpoint_roundtrips_rnd_state_when_present() -> None:
    """Pin: an RND-bearing checkpoint stashes `rnd_state_dict` into
    `_pending_rnd_state` on resume (the novelty memory must survive a
    restart, or every explored state is re-rewarded)."""
    _seed_everything()
    with tempfile.TemporaryDirectory(prefix="vppo_char_rnd_") as tmp:
        t = _bare_tile_trainer(Path(tmp))
        net = t._make_network()
        net.to("cpu")
        opt = t._build_ppo_optimizer(net)

        from src.models.tile_rnd import TileRND

        rnd = TileRND(feature_dim=_TILE_FEATURE_DIM).to("cpu")
        rnd_state = {k: v.detach().cpu() for k, v in rnd.state_dict().items()}

        payload = {
            "iter": 20,
            "net_state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "optimizer_state_dict": opt.state_dict(),
            "rnd_state_dict": rnd_state,
        }
        torch.save(payload, str(Path(tmp) / "vanilla_ppo_iter_00020.pt"))

        fresh_net = t._make_network()
        fresh_net.to("cpu")
        fresh_opt = t._build_ppo_optimizer(fresh_net)
        t._maybe_resume_vanilla_ppo(fresh_net, fresh_opt, fresh_start=False)

        assert t._pending_rnd_state is not None, "RND state was not staged on resume"
        assert set(t._pending_rnd_state.keys()) == set(rnd_state.keys())
        for k in rnd_state:
            assert torch.equal(t._pending_rnd_state[k], rnd_state[k]), (
                f"RND param {k} did not round-trip identically"
            )


# ---------------------------------------------------------------------------
# Groups 1 + 2 — one real vanilla_ppo iteration (metrics dict + reward
# buffer). Tile mode, CPU, tiny config. ROM/profile-gated so it skips
# cleanly on a checkout without the SMB ROM.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_one_iteration_metrics_dict_and_reward_buffer(monkeypatch) -> None:
    """Run ONE real vanilla_ppo iteration on the tile+CPU path and pin:

      * the reward buffer handed to GAE has shape (rollout_steps, num_envs)
        and dtype float32; the done mask is bool; values are float32;
      * the emitted metrics dict carries every canonical dashboard key,
        each a finite number of the right type, with entropy in
        [0, ln(num_actions)].

    Deliberately not asserting the loss/return VALUES — those legitimately
    vary; only the shape/type/finiteness contract is pinned here.
    """
    import src.training.ppo_updater as ppo_updater_mod
    from src.training.trainer import Trainer

    _seed_everything()

    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    # Shrink so one iteration is trivial.
    rollout_steps = 32
    num_envs = 2
    profile["reinforce"]["rollout_steps"] = rollout_steps
    profile["reinforce"]["steps"] = 1
    profile["reinforce"]["ppo_minibatch_size"] = 16
    profile["reinforce"]["device"] = "cpu"

    # Spy on the GAE entry point (the collector->updater seam) to capture
    # the buffers the loop actually fills, without reaching into locals. The
    # protagonist GAE call moved into PPOUpdater (trainer-decomposition plan,
    # Task 2), so patch the name in the ppo_updater module (where it is now
    # called) rather than in the trainer module.
    captured: dict[str, np.ndarray] = {}
    orig_batched_gae = ppo_updater_mod.batched_gae

    def _spy_batched_gae(reward_buf, value_buf, done_buf, *args, **kwargs):
        captured.setdefault("reward_buf", reward_buf)
        captured.setdefault("value_buf", value_buf)
        captured.setdefault("done_buf", done_buf)
        return orig_batched_gae(reward_buf, value_buf, done_buf, *args, **kwargs)

    monkeypatch.setattr(ppo_updater_mod, "batched_gae", _spy_batched_gae)

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="vppo_char_run_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=num_envs,
            population_size=num_envs,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
            device_override="cpu",
        )
        trainer.run(num_generations=1, resume_from=None, fresh_start=True)

    # --- Group 2: reward buffer shape / dtype ---
    assert "reward_buf" in captured, "GAE was never called — the update didn't run"
    reward_buf = captured["reward_buf"]
    value_buf = captured["value_buf"]
    done_buf = captured["done_buf"]
    assert reward_buf.shape == (rollout_steps, num_envs), (
        f"reward_buf shape {reward_buf.shape} != {(rollout_steps, num_envs)}"
    )
    assert reward_buf.dtype == np.float32
    assert value_buf.shape == (rollout_steps, num_envs)
    assert value_buf.dtype == np.float32
    assert done_buf.shape == (rollout_steps, num_envs)
    assert done_buf.dtype == np.bool_

    # --- Group 1: metrics dict keys present + typed ---
    from src.training.metrics_sink import DASHBOARD_REQUIRED_KEYS

    emitted = []
    while not metrics_q.empty():
        emitted.append(metrics_q.get_nowait())
    assert emitted, "no metrics emitted — the loop never finished an iteration"
    ppo_rows = [m for m in emitted if "ppo_loss" in m]
    assert ppo_rows, "no ppo_loss row — the PPO update didn't emit"
    row = ppo_rows[-1]

    for key in DASHBOARD_REQUIRED_KEYS:
        assert key in row, f"required dashboard key {key!r} missing from metrics"

    # `generation` is the absolute iter (int); the rest are finite scalars.
    assert isinstance(row["generation"], (int, np.integer))
    for key in (
        "best_fitness",
        "avg_fitness",
        "ppo_loss",
        "ppo_policy_loss",
        "ppo_value_loss",
        "ppo_entropy",
    ):
        assert _is_number(row[key]), f"{key} is {type(row[key])}, expected a number"
        assert math.isfinite(float(row[key])), f"{key} is non-finite ({row[key]})"

    # Entropy is a real distribution's entropy: within [0, ln(num_actions)].
    assert 0.0 <= float(row["ppo_entropy"]) <= math.log(_NUM_ACTIONS) + 1e-3

    # RND was enabled in this profile (rnd_intrinsic_coef 0.1) so the loop
    # must have built + exercised the module on the rollout path.
    assert trainer._rnd is not None, "rnd_intrinsic_coef > 0 but RND never built"
