"""Tests for the latent-cell archive abstraction (src/training/latent_cells.py).

Everything here runs against synthetic tensors — no emulator, no ROM, no
checkpoint on disk. That matches the module's own scope: it is the VQ-VAE
abstraction only, not the solver integration.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from src.training.latent_cells import (
    LATENT_CODEBOOK_HARD_CAP,
    CellDiscoveryEvent,
    KillCheckResult,
    LatentCellCodebook,
    LatentCellConfig,
    build_discovery_log,
    check_discovery_kill_criterion,
    dead_code_count,
    discovery_rate_per_hour,
    occupancy,
)


def _tiny_config(**overrides) -> LatentCellConfig:
    """A small config for fast tests: tiny window/RAM/hidden/latent dims
    so a training loop of a few hundred steps runs in well under a
    second on CPU."""
    base = dict(
        window_frames=2,
        ram_bytes=4,
        latent_dim=4,
        hidden_dim=16,
        codebook_size=8,
        seed=0,
        dead_code_reinit_interval=0,  # off by default; enabled per-test
        lr=5e-3,
    )
    base.update(overrides)
    return LatentCellConfig(**base)


def _cluster_window(cluster_id: int, config: LatentCellConfig, rng: np.random.Generator) -> np.ndarray:
    """A synthetic RAM window for one of several well-separated clusters.
    Baselines are spaced far apart in [0, 255] and each draw adds a
    little noise, mimicking "distinct game situations with jitter"
    without encoding any actual game semantics."""
    base = np.full((config.window_frames, config.ram_bytes), fill_value=cluster_id * 50 + 10, dtype=np.float32)
    noise = rng.integers(-3, 4, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ---- config validation -----------------------------------------------


def test_codebook_hard_cap_is_512() -> None:
    assert LATENT_CODEBOOK_HARD_CAP == 512


def test_codebook_size_over_hard_cap_rejected() -> None:
    with pytest.raises(ValueError):
        LatentCellConfig(codebook_size=LATENT_CODEBOOK_HARD_CAP + 1)


def test_codebook_size_at_hard_cap_accepted() -> None:
    config = LatentCellConfig(codebook_size=LATENT_CODEBOOK_HARD_CAP)
    assert config.codebook_size == LATENT_CODEBOOK_HARD_CAP


def test_defaults_match_research_spec() -> None:
    config = LatentCellConfig()
    assert config.window_frames == 16
    assert config.ram_bytes == 2048
    assert config.latent_dim == 64
    assert config.codebook_size == 512


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_frames", 0),
        ("ram_bytes", 0),
        ("apu_channels", -1),
        ("latent_dim", 0),
        ("hidden_dim", 0),
        ("codebook_size", 0),
        ("dead_code_reinit_interval", -1),
        ("dead_code_usage_threshold", -1),
    ],
)
def test_invalid_config_fields_rejected(field, value) -> None:
    with pytest.raises(ValueError):
        LatentCellConfig(**{field: value})


# ---- encode_to_cell: shape + determinism ------------------------------


def test_encode_to_cell_returns_int_within_codebook_range() -> None:
    config = _tiny_config()
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(1)
    window = _cluster_window(0, config, rng)
    cell_id = codebook.encode_to_cell(window)
    assert isinstance(cell_id, int)
    assert 0 <= cell_id < config.codebook_size


def test_encode_to_cell_is_deterministic() -> None:
    config = _tiny_config()
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(2)
    window = _cluster_window(1, config, rng)
    first = codebook.encode_to_cell(window)
    for _ in range(5):
        assert codebook.encode_to_cell(window) == first


def test_encode_to_cell_rejects_wrong_ram_window_shape() -> None:
    config = _tiny_config()
    codebook = LatentCellCodebook(config)
    bad_window = np.zeros((config.window_frames, config.ram_bytes + 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        codebook.encode_to_cell(bad_window)


def test_encode_to_cell_requires_apu_window_when_configured() -> None:
    config = _tiny_config(apu_channels=2)
    codebook = LatentCellCodebook(config)
    window = np.zeros((config.window_frames, config.ram_bytes), dtype=np.uint8)
    with pytest.raises(ValueError):
        codebook.encode_to_cell(window)  # missing apu_window


def test_encode_to_cell_accepts_ram_plus_apu_window() -> None:
    config = _tiny_config(apu_channels=2)
    codebook = LatentCellCodebook(config)
    window = np.zeros((config.window_frames, config.ram_bytes), dtype=np.uint8)
    apu = np.zeros((config.window_frames, config.apu_channels), dtype=np.float32)
    cell_id = codebook.encode_to_cell(window, apu_window=apu)
    assert 0 <= cell_id < config.codebook_size


def test_encode_to_cell_never_exceeds_codebook_cap_regardless_of_distinct_inputs() -> None:
    # More distinct clusters than codebook slots -> collisions are
    # expected and fine, but every returned id must still be in range.
    config = _tiny_config(codebook_size=4)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(3)
    seen_ids = set()
    for cluster_id in range(20):
        window = _cluster_window(cluster_id, config, rng)
        cell_id = codebook.encode_to_cell(window)
        assert 0 <= cell_id < config.codebook_size
        seen_ids.add(cell_id)
    assert max(seen_ids) < config.codebook_size
    assert len(seen_ids) <= config.codebook_size


# ---- training separates distinct clusters into distinct codes --------


def test_training_maps_distinct_clusters_to_distinct_codes() -> None:
    config = _tiny_config(codebook_size=16, hidden_dim=32, latent_dim=8, lr=1e-2)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(42)
    n_clusters = 4
    batch_size = 16

    for _ in range(300):
        windows = [
            _cluster_window(step % n_clusters, config, rng) for step in range(batch_size)
        ]
        codebook.train_step(windows)

    # For held-out samples from each cluster, the majority code should
    # differ across clusters -- the whole point of training the VQ-VAE.
    majority_codes = []
    for cluster_id in range(n_clusters):
        codes = Counter(
            codebook.encode_to_cell(_cluster_window(cluster_id, config, rng))
            for _ in range(20)
        )
        majority_codes.append(codes.most_common(1)[0][0])

    assert len(set(majority_codes)) == n_clusters, (
        f"expected {n_clusters} distinct majority codes, got {majority_codes}"
    )


# ---- dead-code reinitialization ---------------------------------------


def test_reinit_dead_codes_resets_usage_and_reports_count() -> None:
    config = _tiny_config(codebook_size=8, dead_code_usage_threshold=1)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(5)

    # Feed only two distinct clusters through an untrained, randomly
    # initialized model -- with 8 codebook slots this reliably leaves
    # several codewords never selected.
    windows = [_cluster_window(cid % 2, config, rng) for cid in range(30)]
    for w in windows:
        codebook.encode_to_cell(w)

    dead_before = codebook.dead_code_count()
    assert dead_before > 0, "test setup expects some dead codewords before reinit"

    reinit_count = codebook.reinit_dead_codes(windows)
    assert reinit_count == dead_before

    # Usage-since-reinit is reset by the pass itself.
    assert codebook.dead_code_count() == config.codebook_size
    assert codebook.reinit_pass_count == 1
    assert codebook.total_codes_reinitialized == reinit_count


def test_reinit_dead_codes_increases_occupancy() -> None:
    config = _tiny_config(codebook_size=8, dead_code_usage_threshold=1, seed=7)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(9)

    windows = [_cluster_window(cid % 3, config, rng) for cid in range(60)]
    for w in windows:
        codebook.encode_to_cell(w)
    occupancy_before = codebook.occupancy()

    reinit_count = codebook.reinit_dead_codes(windows)
    assert reinit_count > 0

    # Re-run a wider variety of windows now that dead codewords have
    # been repositioned into populated regions of latent space.
    more_windows = [_cluster_window(cid % 6, config, rng) for cid in range(60)]
    for w in more_windows:
        codebook.encode_to_cell(w)
    occupancy_after = codebook.occupancy()

    assert occupancy_after > occupancy_before


def test_reinit_dead_codes_noop_when_nothing_dead() -> None:
    config = _tiny_config(codebook_size=2, dead_code_usage_threshold=1)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(11)
    windows = [_cluster_window(cid % 2, config, rng) for cid in range(40)]
    for w in windows:
        codebook.encode_to_cell(w)

    dead_before = codebook.dead_code_count()
    assert codebook.reinit_dead_codes(windows) == dead_before


def test_train_step_triggers_automatic_reinit_on_interval() -> None:
    config = _tiny_config(codebook_size=8, dead_code_reinit_interval=5, dead_code_usage_threshold=1)
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(13)

    fired = False
    for step in range(1, 11):
        windows = [_cluster_window(step % 2, config, rng) for _ in range(4)]
        metrics = codebook.train_step(windows)
        if step % config.dead_code_reinit_interval == 0:
            # A reinit pass runs exactly on multiples of the interval;
            # it may reinit zero codewords but the bookkeeping must fire.
            assert codebook.reinit_pass_count == step // config.dead_code_reinit_interval
            if metrics["reinit_count"] > 0:
                fired = True
        else:
            assert codebook.reinit_pass_count == step // config.dead_code_reinit_interval

    assert fired, "expected at least one interval pass to reinit a dead codeword"


# ---- diagnostics on a stubbed stream -----------------------------------


def test_build_discovery_log_flags_first_sightings() -> None:
    events = [(0.0, 1), (1.0, 2), (2.0, 1), (3.0, 3), (4.0, 2)]
    log = build_discovery_log(events)
    assert [e.is_new for e in log] == [True, True, False, True, False]
    assert all(isinstance(e, CellDiscoveryEvent) for e in log)


def test_occupancy_counts_distinct_cells_over_codebook_size() -> None:
    log = build_discovery_log([(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 2)])
    assert occupancy(log, codebook_size=8) == pytest.approx(3 / 8)


def test_occupancy_clamped_to_one_when_codebook_size_undercounts() -> None:
    log = build_discovery_log([(0.0, i) for i in range(10)])
    assert occupancy(log, codebook_size=4) == 1.0


def test_dead_code_count_is_codebook_size_minus_distinct() -> None:
    log = build_discovery_log([(0.0, 0), (1.0, 1)])
    assert dead_code_count(log, codebook_size=8) == 6


def test_discovery_rate_per_hour_counts_only_new_cells_in_window() -> None:
    # 3 new cells in the first hour, then only repeats in the second hour.
    events = [
        (0.0, 1), (600.0, 2), (1800.0, 3),
        (3700.0, 1), (5000.0, 2),
    ]
    log = build_discovery_log(events)
    rate_hour_2 = discovery_rate_per_hour(log, window_hours=1.0, now_s=7200.0)
    assert rate_hour_2 == 0.0
    rate_hour_1 = discovery_rate_per_hour(log, window_hours=1.0, now_s=1800.0)
    assert rate_hour_1 == 3.0


def test_discovery_kill_criterion_triggers_on_sustained_low_rate() -> None:
    # One new cell discovered per hour for 4 hours straight -- below the
    # default 2/hr threshold for 3 consecutive hours.
    events = [(h * 3600.0, h) for h in range(4)]
    log = build_discovery_log(events)
    result = check_discovery_kill_criterion(log, rate_threshold_per_hour=2.0, consecutive_hours=3)
    assert isinstance(result, KillCheckResult)
    assert result.triggered is True


def test_discovery_kill_criterion_does_not_trigger_on_healthy_discovery() -> None:
    # 5 new cells every hour for 4 hours -- comfortably above threshold.
    events = []
    cell_id = 0
    for hour in range(4):
        for _ in range(5):
            events.append((hour * 3600.0 + 100.0, cell_id))
            cell_id += 1
    log = build_discovery_log(events)
    result = check_discovery_kill_criterion(log, rate_threshold_per_hour=2.0, consecutive_hours=3)
    assert result.triggered is False


def test_discovery_kill_criterion_empty_log_does_not_trigger() -> None:
    result = check_discovery_kill_criterion([], rate_threshold_per_hour=2.0, consecutive_hours=3)
    assert result.triggered is False
    assert result.reason == "no events"


def test_live_discovery_log_populated_only_when_timestamp_given() -> None:
    config = _tiny_config()
    codebook = LatentCellCodebook(config)
    rng = np.random.default_rng(21)
    window = _cluster_window(0, config, rng)

    codebook.encode_to_cell(window)  # no timestamp -> no log growth
    assert codebook.discovery_log() == []

    codebook.encode_to_cell(window, timestamp_s=0.0)
    codebook.encode_to_cell(_cluster_window(1, config, rng), timestamp_s=1.0)
    log = codebook.discovery_log()
    assert len(log) == 2
    assert log[0].timestamp_s == 0.0


# ---- construction is side-effect free on the global RNG ----------------


def test_construction_does_not_perturb_global_torch_rng() -> None:
    torch.manual_seed(1234)
    before = torch.rand(4)

    torch.manual_seed(1234)
    LatentCellCodebook(_tiny_config(seed=999))
    after = torch.rand(4)

    assert torch.equal(before, after)
