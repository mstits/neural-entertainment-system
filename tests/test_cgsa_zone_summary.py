"""Regression tests for the CGSA telemetry correction (L5).

Three things are pinned here:

1. `cgsa_zone_summary` reports UNCENSORED zone averages beside the
   frontier-only ones. The frontier average drops a zone the moment the
   SPRT welds it, so a succeeding curriculum and a dead one produce the
   same trace — which is how the 1-2 protocol seeds came to be recorded
   as "gauntlet stuck at p≈0.01–0.03" when the END-OF-RUN population
   figure was 0.165 / 0.148.
2. Re-scoring the two ARCHIVED `cgsa_stats.json` files reproduces those
   corrected numbers exactly, so `RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md`
   §Correction stays checkable instead of being a remembered claim.
   NOTE: those archives are final dumps (seed 1 iter 5975, seed 2 iter
   13695) — the file is overwritten every telemetry sample. They say
   nothing about the Signpost-2 (it800) verdicts, which stand as logged.
3. No top-level profile YAML carries a duplicate mapping key. PyYAML is
   silently last-wins, and a stale duplicate `maintenance_weight: 0.01`
   trailing the documented v6 value of 0.003 reverted the fix in all
   three CGSA configs without a word in any log.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest
import yaml

from src.training.trainer import cgsa_zone_summary


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CGSA_CONFIGS = (
    "mario_1_2_cgsa.yaml",
    "mario_1_2_cgsa_s1.yaml",
    "mario_1_2_cgsa_s2.yaml",
)

# The archived (END-OF-RUN) curriculum state of the two completed 1-2
# protocol seeds, and what re-scoring them must produce.
# `gauntlet_frontier` is what the run logs printed at that same final
# iteration (s1 5975, s2 13695); `gauntlet_all` is the same population
# without the welded zones censored out.
ARCHIVED_SEEDS = {
    "s1": {
        "path": REPO_ROOT / "checkpoints/mario_1_2_cgsa_s1/cgsa_stats.json",
        "cells": 2586,
        "welded": 1648,
        "welded_frac": 0.637,
        "gauntlet_frontier": 0.022,
        "gauntlet_all": 0.165,
        "avg_p_all": 0.176,
    },
    "s2": {
        "path": REPO_ROOT / "checkpoints/mario_1_2_cgsa_s2/cgsa_stats.json",
        "cells": 2686,
        "welded": 1967,
        "welded_frac": 0.732,
        "gauntlet_frontier": 0.009,
        "gauntlet_all": 0.148,
        "avg_p_all": 0.180,
    },
}


def _zone(gx: int, p: float, welded: bool, rate: float = 1.0,
          windows: int = 5) -> dict:
    return {
        "gx": gx, "p": p, "welded": welded, "rate": rate,
        "windows": windows, "att": 15, "succ": 15, "lam": 0.0,
    }


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key."""


def _no_duplicate_mapping(loader, node, deep: bool = False) -> dict:
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = sorted(
        str(k) for k, n in collections.Counter(keys).items() if n > 1
    )
    if dupes:
        raise ValueError(
            f"duplicate key(s) {dupes} at line {node.start_mark.line + 1}"
        )
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_mapping
)


def test_empty_curriculum_summarises_to_zeros() -> None:
    """No tracked zones must not divide by zero."""
    summary = cgsa_zone_summary({})
    assert summary["cells"] == 0
    assert summary["welded_frac"] == 0.0
    assert summary["avg_p_all"] == 0.0
    assert summary["gauntlet_avg_p_all"] == 0.0


def test_welding_censors_the_frontier_average_but_not_avg_p_all() -> None:
    """The whole point of the correction, in miniature.

    Four gauntlet zones annealed to the 0.25 target and welded, one
    still on the frontier at 0.0: the frontier figure reads 0.0 (looks
    like total failure) while the uncensored figure reads 0.20.
    """
    stats = {
        0: _zone(1700, 0.25, welded=True),
        1: _zone(1800, 0.25, welded=True),
        2: _zone(1900, 0.25, welded=True),
        3: _zone(2000, 0.25, welded=True),
        4: _zone(2100, 0.00, welded=False),
    }
    summary = cgsa_zone_summary(stats)

    assert summary["gauntlet_n"] == 1
    assert summary["gauntlet_avg_p"] == pytest.approx(0.0)
    assert summary["gauntlet_n_all"] == 5
    assert summary["gauntlet_avg_p_all"] == pytest.approx(0.20)
    assert summary["avg_p_all"] == pytest.approx(0.20)
    assert summary["welded_frac"] == pytest.approx(0.8)


def test_gauntlet_window_excludes_zones_outside_gx_1600_2200() -> None:
    """Only gx in [1600, 2200] counts toward the gauntlet figures."""
    stats = {
        0: _zone(1000, 0.25, welded=False),   # before the window
        1: _zone(1600, 0.05, welded=False),   # inclusive lower bound
        2: _zone(2200, 0.15, welded=False),   # inclusive upper bound
        3: _zone(2400, 0.25, welded=False),   # past the window
    }
    summary = cgsa_zone_summary(stats)

    assert summary["gauntlet_n_all"] == 2
    assert summary["gauntlet_avg_p_all"] == pytest.approx(0.10)
    assert summary["avg_p_all"] == pytest.approx(0.175)


def test_untracked_cells_without_gx_are_ignored() -> None:
    """Archive entries that never got a global-x are not zones."""
    stats = {
        0: _zone(1700, 0.25, welded=True),
        1: {"p": 0.0, "welded": False, "rate": 0.0, "windows": 0},
    }
    summary = cgsa_zone_summary(stats)
    assert summary["cells"] == 1
    assert summary["avg_p_all"] == pytest.approx(0.25)


@pytest.mark.parametrize("seed", sorted(ARCHIVED_SEEDS))
def test_archived_seed_rescores_to_the_corrected_numbers(seed: str) -> None:
    """Offline recompute of the archived curriculum state reproduces the
    figures published in RESULTS_1_2_HONEST_PROTOCOL_2026-07-24 §Correction:
    gauntlet uncensored 0.165 (seed 1) / 0.148 (seed 2), against the
    0.022 / 0.009 the run logs reported."""
    expect = ARCHIVED_SEEDS[seed]
    path: Path = expect["path"]
    if not path.exists():
        pytest.skip(f"archived curriculum state missing: {path}")

    with path.open() as fh:
        summary = cgsa_zone_summary(json.load(fh))

    assert summary["cells"] == expect["cells"]
    assert summary["welded"] == expect["welded"]
    assert round(summary["welded_frac"], 3) == expect["welded_frac"]
    assert round(summary["gauntlet_avg_p"], 3) == expect["gauntlet_frontier"]
    assert round(summary["gauntlet_avg_p_all"], 3) == expect["gauntlet_all"]
    assert round(summary["avg_p_all"], 3) == expect["avg_p_all"]
    # The censored figure is an order of magnitude below the honest one
    # — that gap is the reason the record needed correcting at all.
    assert summary["gauntlet_avg_p_all"] > 5 * summary["gauntlet_avg_p"]


@pytest.mark.parametrize("cfg_name", CGSA_CONFIGS)
def test_cgsa_config_maintenance_weight_is_the_v6_value(cfg_name: str) -> None:
    """The v6 fix (0.01 leaked ~1/3 of attempts to welded/waiting zones)
    must survive a `yaml.safe_load`, not be reverted by a trailing dupe."""
    cfg_path = CONFIG_DIR / cfg_name
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh)
    assert data["reinforce"]["cgsa"]["maintenance_weight"] == 0.003


@pytest.mark.parametrize(
    "cfg_path",
    sorted(CONFIG_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_profile_yaml_has_no_duplicate_keys(cfg_path: Path) -> None:
    """PyYAML resolves duplicates last-wins in silence, so a stale
    leftover key overrides the documented value with no error and no log
    line. Fail loudly at collection time instead."""
    try:
        yaml.load(cfg_path.read_text(), Loader=_DuplicateKeyLoader)
    except ValueError as exc:
        pytest.fail(f"{cfg_path.name}: {exc}")
