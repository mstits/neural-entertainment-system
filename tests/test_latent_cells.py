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


def test_reinit_dead_codes_noop_still_counts_as_pass() -> None:
    # A fully-utilized codebook: every slot cleared the usage threshold,
    # so the early-return (n_dead == 0) branch fires. It still resets
    # usage-since-reinit -- that's a real, meaningful "a pass ran" side
    # effect -- so the pass counter must move too, or callers checking
    # reinit_pass_count can't tell a scheduled pass ran from it never
    # having fired at all.
    config = _tiny_config(codebook_size=4, dead_code_usage_threshold=1)
    codebook = LatentCellCodebook(config)
    codebook._usage_since_reinit = torch.tensor([1, 2, 1, 3])

    z_e = torch.randn(6, config.latent_dim)
    reinit_count = codebook._reinit_dead_codes(z_e)

    assert reinit_count == 0
    assert codebook.reinit_pass_count == 1
    assert torch.equal(codebook._usage_since_reinit, torch.zeros(4, dtype=torch.long))


def test_cell_id_is_a_codebook_slot_not_a_permanent_identity() -> None:
    # KNOWN LIMITATION, pinned deliberately. A reinit pass overwrites a
    # dead codeword's embedding in place, so the raw index it lived at
    # gets handed to whatever region of latent space moves in -- the
    # same integer can mean two different things across a pass. Ditto,
    # more pervasively, for ordinary training: `train_step`'s codebook
    # loss moves every live codeword a little on every step, so
    # `encode_to_cell` only promises the stability its docstring claims
    # ("the same window always returns the same id until the model is
    # trained or a reinit pass moves a codeword").
    #
    # The tempting fix -- folding a per-slot reinit generation counter
    # into the returned id (raw_id + generation * codebook_size) -- was
    # tried and rejected: it pushes ids outside [0, codebook_size) and
    # silently inverts `occupancy` and `dead_code_count`, which both use
    # codebook_size as their denominator. See
    # test_cell_ids_stay_inside_the_codebook_across_many_reinit_passes.
    # A real fix has to give the diagnostics a generation-aware
    # denominator (or map (slot, generation) through a monotonic id
    # registry and teach occupancy/dead_code_count about it), not just
    # rescale the id.
    config = _tiny_config(
        window_frames=1,
        ram_bytes=16,
        codebook_size=8,
        hidden_dim=32,
        latent_dim=8,
        dead_code_reinit_interval=0,
        dead_code_usage_threshold=1,
        seed=0,
    )
    codebook = LatentCellCodebook(config)
    window_a = np.full((config.window_frames, config.ram_bytes), 20, dtype=np.uint8)
    window_c = np.full((config.window_frames, config.ram_bytes), 200, dtype=np.uint8)

    id_a = codebook.encode_to_cell(window_a, timestamp_s=0.0)

    # Every slot but window_a's was revisited, so window_a's is the one
    # reinit targets.
    codebook._usage_since_reinit[:] = 1
    codebook._usage_since_reinit[id_a] = 0
    assert codebook.reinit_dead_codes([window_c]) == 1

    id_c = codebook.encode_to_cell(window_c, timestamp_s=1.0)

    # The contract that IS load-bearing: an id is always a real slot.
    assert 0 <= id_a < config.codebook_size
    assert 0 <= id_c < config.codebook_size
    # And the limitation itself: a recycled slot re-uses its id.
    assert id_c == id_a


def test_cell_ids_stay_inside_the_codebook_across_many_reinit_passes() -> None:
    # The whole diagnostics layer treats `codebook_size` as the
    # denominator for a cell id: `occupancy` divides distinct ids by it,
    # `dead_code_count` subtracts distinct ids from it, and
    # `_VQVAE.quantize` documents its indices as "always in
    # [0, config.codebook_size)". So whatever `encode_to_cell` returns
    # has to live in that range too, over a whole run -- not just over
    # the first reinit pass.
    #
    # This is the regime that matters: production K=512 with the
    # production reinit machinery running many passes, which is what a
    # real run does (dead_code_reinit_interval=200 over a
    # hundred-thousand-step run is hundreds of passes). If ids can grow
    # past K, both log-level diagnostics silently invert -- occupancy
    # pins at 1.0 and dead_code_count pins at 0 no matter how much of
    # the codebook is actually dead -- and because these ids are the
    # Go-Explore cell identity, that reads out as "the archive is
    # discovering everything" precisely when it isn't.
    k = LatentCellConfig().codebook_size  # production size: 512
    config = _tiny_config(
        ram_bytes=64,
        latent_dim=8,
        hidden_dim=32,
        codebook_size=k,
        dead_code_reinit_interval=5,
        dead_code_usage_threshold=1,
        lr=1e-2,
        seed=0,
    )
    codebook = LatentCellCodebook(config)
    windows = [
        np.full((config.window_frames, config.ram_bytes), int(v), dtype=np.uint8)
        for v in np.linspace(5, 250, 40)
    ]

    timestamp = 0.0
    for step in range(1, 101):
        codebook.train_step(windows)
        if step % config.dead_code_reinit_interval == 0:
            # Re-encode the same states after every reinit pass, the way
            # a Go-Explore archive re-cells states it keeps returning to.
            for window in windows:
                codebook.encode_to_cell(window, timestamp_s=timestamp)
                timestamp += 1.0

    # Guards on the regime, so this can never pass vacuously: the run
    # must actually have churned the codebook, and must have logged more
    # encodes than there are slots (otherwise distinct-id saturation
    # could not be observed either way).
    assert codebook.reinit_pass_count >= 20
    assert codebook.total_codes_reinitialized > k
    log = codebook.discovery_log()
    assert len(log) > k

    ids = {e.cell_id for e in log}
    assert min(ids) >= 0
    assert max(ids) < k, (
        f"encode_to_cell returned cell id {max(ids)} for a codebook of size {k}; "
        "ids outside [0, codebook_size) break occupancy()/dead_code_count(), "
        "which use codebook_size as their denominator"
    )
    assert len(ids) <= k, (
        f"{len(ids)} distinct cell ids from a {k}-slot codebook -- distinct cells "
        "cannot outnumber the slots that produce them"
    )

    # The log sees a subset of usage (encode_to_cell calls only; the
    # live counters also see train_step), so the log-level diagnostics
    # must bound the live ones from the pessimistic side. Reporting MORE
    # of the codebook occupied, or FEWER codes dead, than the live model
    # itself knows about is the inversion this test exists to catch.
    live_used = int((codebook._lifetime_usage > 0).sum().item())
    assert occupancy(log, k) <= codebook.occupancy() + 1e-9, (
        f"log occupancy {occupancy(log, k)} exceeds live occupancy "
        f"{codebook.occupancy()} -- the log cannot have seen usage the live "
        "counters did not"
    )
    assert dead_code_count(log, k) >= k - live_used, (
        f"log reports {dead_code_count(log, k)} dead codes but the live codebook "
        f"has {k - live_used} entries never used at all"
    )

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


def test_repeated_reinit_of_one_slot_keeps_the_kill_criterion_readable() -> None:
    # The many-pass version of the single-pass aliasing scenario above,
    # at the production codebook size. Same premise, iterated: one state
    # region, one codebook slot, and a reinit pass each interval in
    # which that slot was the one under the usage threshold. A real run
    # does hundreds of passes (dead_code_reinit_interval=200 over a
    # 100k-step run), so "survives one pass" is not the bar.
    #
    # The trajectory here is the encoder's worst case and the exact
    # thing the v19 round pre-registered a kill for: the agent is stuck,
    # revisiting one region forever, discovering nothing. Every
    # log-level diagnostic in this module has to keep saying so --
    # `occupancy` and `dead_code_count` divide by / subtract from
    # `codebook_size`, and `check_discovery_kill_criterion` counts
    # first-sightings, so all three are only meaningful while a cell id
    # names a codebook slot. A per-slot counter folded into the id makes
    # each revisit of an unchanged region read as a brand-new cell, and
    # that inverts all three at once: occupancy pins at 1.0,
    # dead_code_count pins at 0, and the pre-registered kill can never
    # fire on the very run it was written to stop.
    k = LatentCellConfig().codebook_size  # production size: 512
    config = _tiny_config(
        window_frames=1,
        ram_bytes=16,
        latent_dim=8,
        hidden_dim=32,
        codebook_size=k,
        dead_code_reinit_interval=0,  # passes forced explicitly below
        dead_code_usage_threshold=1,
        seed=0,
    )
    codebook = LatentCellCodebook(config)
    window = np.full((config.window_frames, config.ram_bytes), 20, dtype=np.uint8)

    # 600 revisits, one per minute, with a reinit pass after each -- long
    # enough to outnumber the 512 slots, so distinct-id saturation is
    # observable rather than merely asserted about.
    n_visits = 600
    for visit in range(n_visits):
        cell_id = codebook.encode_to_cell(window, timestamp_s=visit * 60.0)
        # The slot this region occupies is the one that fell under the
        # threshold this interval; everything else saw traffic. Same
        # setup the single-pass test above uses, held for every pass.
        slot = cell_id % k
        codebook._usage_since_reinit[:] = 1
        codebook._usage_since_reinit[slot] = 0
        assert codebook.reinit_dead_codes([window]) == 1

    # Regime guards, so this cannot pass vacuously.
    assert codebook.reinit_pass_count == n_visits
    assert int((codebook._lifetime_usage > 0).sum().item()) == 1, (
        "test setup expects the region to keep landing on one slot"
    )
    log = codebook.discovery_log()
    assert len(log) == n_visits > k

    # One slot of 512 was ever selected. Both readouts of "how much of
    # the codebook has ever been used" have to agree on that; the log
    # saw every encode_to_cell call and no train_step ran, so there is
    # no usage for them to legitimately disagree about.
    assert codebook.occupancy() == pytest.approx(1 / k)
    assert occupancy(log, k) == pytest.approx(codebook.occupancy()), (
        f"log occupancy {occupancy(log, k)} vs live {codebook.occupancy()} -- "
        "distinct cell ids outran the slots that produced them, so occupancy "
        "no longer measures a fraction of the codebook"
    )
    assert dead_code_count(log, k) == k - 1, (
        f"log reports {dead_code_count(log, k)} dead codes; {k - 1} slots were "
        "never selected even once"
    )

    # And the kill the module exists to compute: 10 hours of revisiting
    # one region is zero discovery after the first sighting, which is
    # below 2 new cells/hr for far more than 3 consecutive hours.
    result = check_discovery_kill_criterion(log)
    assert result.triggered is True, (
        f"discovery rates {result.rates_per_hour[:6]} -- a run that visited one "
        "state region 600 times and nothing else must trip the pre-registered kill"
    )
