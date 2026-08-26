"""Regression test: RND rebuild after a mid-run coefficient ramp.

Pins the fix for the reported defect where `rnd_intrinsic_coef` starts at
0.0 (RND "off") and a later consolidate/consolidate_level schedule ramps it
up — `ExplorationController.build_rnd` must (a) actually build the module
once the coefficient turns positive, even though an earlier call saw a zero
coefficient and no-opped, and (b) register the newly-built predictor's
parameters into an already-existing PPO optimizer so they actually train
instead of being silently dropped (the same hazard `Trainer.
_build_ppo_optimizer` documents for the anti-collapse rollback rebuild,
reached here from the opposite direction: RND arriving late instead of the
optimizer being rebuilt without it).
"""
from __future__ import annotations

import torch

from src.training.exploration_controller import ExplorationController


class _StubTrainer:
    """Minimal stand-in exposing only the attributes `build_rnd` reads."""

    def __init__(self) -> None:
        self.rnd_intrinsic_coef = 0.0
        self.rnd_loss_coef = 0.1
        self._rnd = None
        self._is_tile_mode = True
        self._tile_feature_dim = 16
        self.device = torch.device("cpu")
        self._pending_rnd_state = None
        # Mirrors the real vanilla_ppo ordering: the PPO optimizer is
        # already built (over the policy net only) by the time a mid-run
        # schedule first raises rnd_intrinsic_coef off a zero baseline.
        self._ppo_optimizer = torch.optim.Adam(
            [torch.nn.Parameter(torch.zeros(4))], lr=1e-3
        )


_LOG_MSG = "test RND (%s): predictor=%d params, coef=%.3f, loss_coef=%.3f"


def test_build_rnd_rebuilds_after_zero_baseline_ramp() -> None:
    t = _StubTrainer()
    ctrl = ExplorationController(t)

    # First call, at the zero baseline: must stay a no-op (the normal
    # "RND off" case every vanilla_ppo run hits before any consolidate
    # schedule fires).
    ctrl.build_rnd(log_msg=_LOG_MSG)
    assert t._rnd is None
    assert len(t._ppo_optimizer.param_groups) == 1

    # A consolidate/consolidate_level schedule ramps the coefficient up
    # off the zero baseline mid-run (trainer.py's lerp_coef step).
    t.rnd_intrinsic_coef = 0.3

    # The re-fire trainer.py now performs right after applying that
    # schedule.
    ctrl.build_rnd(log_msg=_LOG_MSG)

    assert t._rnd is not None, (
        "build_rnd must build the module once rnd_intrinsic_coef rises "
        "off a zero baseline, not just on an earlier call"
    )
    assert len(t._ppo_optimizer.param_groups) == 2, (
        "a build that lands after the PPO optimizer already exists must "
        "register the predictor's params into it, or the predictor "
        "trains via backward() but is never stepped"
    )
    _rnd_group_params = {
        id(p) for p in t._ppo_optimizer.param_groups[-1]["params"]
    }
    assert _rnd_group_params == {
        id(p) for p in t._rnd.predictor.parameters()
    }

    # Idempotent: a third call (e.g. the next iteration's re-fire) must
    # not double-register the predictor into the optimizer.
    ctrl.build_rnd(log_msg=_LOG_MSG)
    assert len(t._ppo_optimizer.param_groups) == 2
