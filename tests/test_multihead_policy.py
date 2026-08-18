"""Tests for the shared-trunk, per-level multi-head tile policy."""

from __future__ import annotations

import torch

from src.models.tile_policy import TilePolicyNetwork, build_tile_policy_from_checkpoint
from src.training.multihead_policy import (
    HEAD_KEYS,
    TRUNK_KEYS,
    MultiHeadTilePolicy,
)

FEATURE_DIM = 175
NUM_ACTIONS = 8
LEVELS = ["1-1", "1-2", "1-3", "1-4"]


def _make_net(num_levels: int = len(LEVELS), **kwargs) -> MultiHeadTilePolicy:
    kwargs.setdefault("levels", LEVELS[:num_levels])
    return MultiHeadTilePolicy(
        num_actions=NUM_ACTIONS,
        num_levels=num_levels,
        feature_dim=FEATURE_DIM,
        **kwargs,
    )


# ----- construction / shape -------------------------------------------


def test_forward_ac_shapes() -> None:
    net = _make_net()
    x = torch.randn(6, FEATURE_DIM)
    level_ids = torch.tensor([0, 1, 2, 3, 0, 1])
    logits, value = net.forward_ac(x, level_ids)
    assert logits.shape == (6, NUM_ACTIONS)
    assert value.shape == (6,)


def test_forward_ac_rejects_wrong_feature_dim() -> None:
    net = _make_net()
    x = torch.randn(2, FEATURE_DIM + 1)
    try:
        net.forward_ac(x, torch.tensor([0, 0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched feature_dim")


def test_forward_ac_rejects_out_of_range_level_ids() -> None:
    net = _make_net()
    x = torch.randn(2, FEATURE_DIM)
    try:
        net.forward_ac(x, torch.tensor([0, 99]))
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for out-of-range level id")


def test_scalar_int_level_ids_broadcasts_to_whole_batch() -> None:
    net = _make_net()
    net.eval()
    x = torch.randn(5, FEATURE_DIM)
    with torch.no_grad():
        logits_scalar, value_scalar = net.forward_ac(x, 2)
        logits_tensor, value_tensor = net.forward_ac(x, torch.full((5,), 2, dtype=torch.long))
    assert torch.equal(logits_scalar, logits_tensor)
    assert torch.equal(value_scalar, value_tensor)


# ----- per-sample head routing ------------------------------------------


def test_mixed_batch_routing_matches_per_level_forward() -> None:
    """A mixed-level batch must produce exactly the outputs each row
    would get if run alone at its own level -- the normal PPO-minibatch
    case, not an edge case."""
    net = _make_net(num_levels=4)
    net.eval()
    torch.manual_seed(0)
    x = torch.randn(12, FEATURE_DIM)
    level_ids = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3])

    with torch.no_grad():
        mixed_logits, mixed_value = net.forward_ac(x, level_ids)

        for lvl in range(4):
            mask = level_ids == lvl
            rows = x[mask]
            solo_logits, solo_value = net.forward_ac(rows, lvl)
            # allclose, not equal: matmul over a differently-shaped batch
            # can take a different BLAS tiling and land a ULP or two off,
            # even though the math is per-row independent. The routing
            # claim is "same head, same computation", not "identical
            # floating-point reduction order".
            assert torch.allclose(mixed_logits[mask], solo_logits, atol=1e-6)
            assert torch.allclose(mixed_value[mask], solo_value, atol=1e-6)


def test_routing_by_string_label_matches_index() -> None:
    net = _make_net()
    net.eval()
    x = torch.randn(3, FEATURE_DIM)
    with torch.no_grad():
        by_index = net.act(x[:1], level=1, deterministic=True)
        by_label = net.act(x[:1], level="1-2", deterministic=True)
    assert by_index == by_label


# ----- export round-trip -------------------------------------------------


def test_export_head_loads_through_eval_shape_infer_helper() -> None:
    """export_head's state_dict must load through the SAME shape-infer
    helper scripts/eval_game.py uses (build_tile_policy_from_checkpoint),
    not a second reimplementation of it."""
    net = _make_net(hidden_dim=256, trunk_dim=64)
    net.eval()
    x = torch.randn(4, FEATURE_DIM)
    level = 2
    with torch.no_grad():
        expected_logits, expected_value = net.forward_ac(x, level)

    sd = net.export_head(level)
    assert set(sd.keys()) == set(TRUNK_KEYS) | set(HEAD_KEYS)

    built, is_recurrent = build_tile_policy_from_checkpoint(
        sd, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
    )
    assert is_recurrent is False
    assert isinstance(built, TilePolicyNetwork)
    assert built.hidden_dim == 256
    assert built.trunk_dim == 64

    load_result = built.load_state_dict(sd, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    built.eval()
    with torch.no_grad():
        got_logits, got_value = built.forward_ac(x)
    assert torch.allclose(got_logits, expected_logits, atol=1e-6)
    assert torch.allclose(got_value, expected_value, atol=1e-6)


def test_export_head_is_independent_per_level() -> None:
    net = _make_net()
    sd_a = net.export_head(0)
    sd_b = net.export_head(1)
    # Trunk identical across exports (shared); actor/critic differ
    # (independently initialized heads).
    for k in TRUNK_KEYS:
        assert torch.equal(sd_a[k], sd_b[k])
    assert not torch.equal(sd_a["actor.weight"], sd_b["actor.weight"])
    assert not torch.equal(sd_a["critic.weight"], sd_b["critic.weight"])


def test_export_checkpoint_wraps_net_state_dict_and_loads() -> None:
    """The on-disk convention the trainer/bc_distill checkpoints use is
    {"net_state_dict": ...}; eval_game.py reads that key directly."""
    net = _make_net()
    net.eval()
    x = torch.randn(2, FEATURE_DIM)
    level = "1-3"
    with torch.no_grad():
        expected_logits, _ = net.forward_ac(x, net._level_index(level))

    payload = net.export_checkpoint(level, iter=42, provenance="test")
    assert payload["net_state_dict"] is not None
    assert payload["level"] == "1-3"
    assert payload["iter"] == 42

    built, _ = build_tile_policy_from_checkpoint(
        payload, num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM,
    )
    built.load_state_dict(payload["net_state_dict"], strict=True)
    built.eval()
    with torch.no_grad():
        got_logits, _ = built.forward_ac(x)
    assert torch.allclose(got_logits, expected_logits, atol=1e-6)


def test_export_head_rejects_unknown_level() -> None:
    net = _make_net()
    try:
        net.export_head("9-9")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown level label")
    try:
        net.export_head(99)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for out-of-range level index")


# ----- parameter counts ---------------------------------------------------


def test_trunk_is_shared_once_heads_scale_with_num_levels() -> None:
    single = TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=256, trunk_dim=64,
    )
    single_trunk_params = sum(
        p.numel()
        for mod in (single.fc1, single.norm1, single.fc2, single.norm2)
        for p in mod.parameters()
    )

    for num_levels in (1, 3, 4, 6):
        net = _make_net(num_levels=num_levels, levels=None, hidden_dim=256, trunk_dim=64)
        assert net.trunk_param_count == single_trunk_params
        assert net.num_params == net.trunk_param_count + num_levels * net.head_param_count


def test_num_levels_must_be_positive() -> None:
    try:
        MultiHeadTilePolicy(num_actions=8, num_levels=0, feature_dim=FEATURE_DIM)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for num_levels=0")


def test_levels_length_must_match_num_levels() -> None:
    try:
        MultiHeadTilePolicy(
            num_actions=8, num_levels=2, feature_dim=FEATURE_DIM, levels=["1-1"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for levels/num_levels mismatch")


# ----- gradient isolation -------------------------------------------------


def test_loss_on_one_head_leaves_other_heads_bit_identical() -> None:
    """A loss confined to level A's rows must move level A's head and
    the shared trunk, and leave level B's head bit-identical."""
    net = _make_net(num_levels=4)
    net.train()

    before = {
        name: p.detach().clone() for name, p in net.named_parameters()
    }

    torch.manual_seed(1)
    x = torch.randn(16, FEATURE_DIM)
    level_a, level_b = 0, 2
    level_ids = torch.full((16,), level_a, dtype=torch.long)

    logits, value = net.forward_ac(x, level_ids)
    loss = logits.pow(2).mean() + value.pow(2).mean()
    loss.backward()

    # Level A's head must have received a nonzero gradient.
    for p in net.actor_heads[level_a].parameters():
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0
    for p in net.critic_heads[level_a].parameters():
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0

    # Level B's head must receive an EXACT zero gradient (or no
    # gradient at all), not merely a small one.
    for p in net.actor_heads[level_b].parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0
    for p in net.critic_heads[level_b].parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0

    # The shared trunk must move.
    trunk_moved = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for mod_name in ("fc1", "norm1", "fc2", "norm2")
        for p in getattr(net, mod_name).parameters()
    )
    assert trunk_moved

    # Apply a plain SGD step and confirm level B's head is bit-identical
    # to its pre-update values while trunk + level A moved.
    with torch.no_grad():
        for p in net.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad

    for name, p in net.named_parameters():
        if name.startswith(f"actor_heads.{level_b}.") or name.startswith(
            f"critic_heads.{level_b}."
        ):
            assert torch.equal(p, before[name]), f"{name} changed but should not have"
        elif name.startswith(f"actor_heads.{level_a}.") or name.startswith(
            f"critic_heads.{level_a}."
        ) or name.startswith(("fc1.", "fc2.", "norm1.", "norm2.")):
            assert not torch.equal(p, before[name]), f"{name} should have moved"


def test_gradient_isolation_holds_for_every_untouched_head() -> None:
    """Same isolation property, checked across ALL untouched heads at
    once (not just one), on a batch that touches a single level."""
    net = _make_net(num_levels=4)
    net.train()
    torch.manual_seed(2)
    x = torch.randn(8, FEATURE_DIM)
    touched = 3
    level_ids = torch.full((8,), touched, dtype=torch.long)

    logits, value = net.forward_ac(x, level_ids)
    (logits.sum() + value.sum()).backward()

    for lvl in range(4):
        actor_grad_sum = sum(
            p.grad.abs().sum().item() for p in net.actor_heads[lvl].parameters() if p.grad is not None
        )
        critic_grad_sum = sum(
            p.grad.abs().sum().item() for p in net.critic_heads[lvl].parameters() if p.grad is not None
        )
        if lvl == touched:
            assert actor_grad_sum > 0
            assert critic_grad_sum > 0
        else:
            assert actor_grad_sum == 0.0
            assert critic_grad_sum == 0.0


# ----- warm start -----------------------------------------------------


def test_init_trunk_from_copies_trunk_and_nothing_else() -> None:
    specialist = TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=256, trunk_dim=64,
    )
    net = _make_net(num_levels=3, hidden_dim=256, trunk_dim=64, levels=None)

    heads_before = [
        net.actor_heads[i].weight.detach().clone() for i in range(3)
    ]

    net.init_trunk_from({"net_state_dict": specialist.state_dict()})

    for k in TRUNK_KEYS:
        assert torch.equal(net._resolve_param(k), specialist.state_dict()[k])

    # Heads must be untouched by a trunk-only warm start.
    for i in range(3):
        assert torch.equal(net.actor_heads[i].weight, heads_before[i])


def test_init_trunk_from_rejects_shape_mismatch() -> None:
    specialist = TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=32, trunk_dim=16,
    )
    net = _make_net(hidden_dim=256, trunk_dim=64)
    try:
        net.init_trunk_from({"net_state_dict": specialist.state_dict()})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for trunk width mismatch")


def test_init_head_from_copies_only_that_heads_weights() -> None:
    specialist = TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=256, trunk_dim=64,
    )
    net = _make_net(num_levels=3, hidden_dim=256, trunk_dim=64, levels=None)

    other_head_before = net.actor_heads[2].weight.detach().clone()
    trunk_before = net.fc1.weight.detach().clone()

    net.init_head_from(1, {"net_state_dict": specialist.state_dict()})

    assert torch.equal(net.actor_heads[1].weight, specialist.actor.weight)
    assert torch.equal(net.actor_heads[1].bias, specialist.actor.bias)
    assert torch.equal(net.critic_heads[1].weight, specialist.critic.weight)
    assert torch.equal(net.critic_heads[1].bias, specialist.critic.bias)

    # Other heads and the trunk are untouched.
    assert torch.equal(net.actor_heads[2].weight, other_head_before)
    assert torch.equal(net.fc1.weight, trunk_before)


def test_init_head_from_rejects_shape_mismatch() -> None:
    specialist = TilePolicyNetwork(
        num_actions=NUM_ACTIONS + 1, feature_dim=FEATURE_DIM, hidden_dim=256, trunk_dim=64,
    )
    net = _make_net(hidden_dim=256, trunk_dim=64)
    try:
        net.init_head_from(0, {"net_state_dict": specialist.state_dict()})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action-count mismatch")


def test_warm_and_cold_variants_diverge() -> None:
    """Sanity: a warm-started trunk differs from a cold one, so 'cold
    variant' vs 'warm variant' in an experiment config actually means
    something different at init time."""
    specialist = TilePolicyNetwork(
        num_actions=NUM_ACTIONS, feature_dim=FEATURE_DIM, hidden_dim=256, trunk_dim=64,
    )
    cold = _make_net(hidden_dim=256, trunk_dim=64)
    warm = _make_net(hidden_dim=256, trunk_dim=64)
    warm.init_trunk_from({"net_state_dict": specialist.state_dict()})

    assert not torch.equal(cold.fc1.weight, warm.fc1.weight)
    assert torch.equal(warm.fc1.weight, specialist.fc1.weight)
