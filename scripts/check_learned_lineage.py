#!/usr/bin/env python3
"""Lineage guard for the four banked LEARNED rates.

Answers one question, mechanically, so it stops being re-asked: was any
banked LEARNED number trained with the dense x-ladder bonus on?

For each banked lineage this reads the run manifest CLAIMS.md names,
follows the manifest's OWN recorded profile pointer (never a config
picked by name or by start-state filename shape), and asserts:

  1. the manifest exists, parses, and carries the profile pointer at its
     recorded key path;
  2. the profile it names exists and sets
     `reward_weights.checkpoint_scale: 0.0` explicitly;
  3. every `<run_dir>/phase_configs/*.yaml` recorded by that run sets
     `0.0` too, if it sets the key at all, and there are at least as
     many phase configs as the lineage pins (a phase-config directory
     that empties out must not pass this vacuously);
  4. the run's `metrics.jsonl` has at least the pinned number of rows,
     every row parses, no row anywhere contains a `reward_checkpoint`
     key, and at least one row carries `reward_forward` (a metrics file
     that logs no reward categories at all would otherwise satisfy the
     no-`reward_checkpoint` test by logging nothing).

Checks 3's floor and 4's row floor and positive control exist because
the interesting failure here is not a flipped value, it is an input
that quietly stops being there. A guard that passes on a missing file
settles nothing.

The attribution rule this enforces, in one line: a claim that a banked
number was trained under a given setting cites the run manifest and the
recorded phase config, never a config chosen by name.

Read-only. Exit 0 when every lineage is clean, 1 on any violation, 2 on
a usage error. `--json` for machine output.

Run: .venv/bin/python scripts/check_learned_lineage.py
     make lineage-check

Not a `make test` prerequisite: `runs/` and `checkpoints/` are
gitignored, so a clean checkout has none of these inputs and would fail
it for the wrong reason. The pytest file that covers this script
(`tests/test_check_learned_lineage.py`) runs entirely on tmp fixtures
and is safe in the suite.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# Every field below is pinned from the artifacts as read on 2026-09-01
# (judged sweep, reports/.../2026-09-01-outstanding/reward-sweep-JUDGED.md
# section 2). profile_key is the path INTO the manifest, not into the
# profile: runs/interference records its 1-1 specialist under
# config.specialists["1-1"].profile, the four campaign runs record
# config.base_profile.
LINEAGES = (
    # 1-1 carries a seam the other four rows do not, named here rather
    # than papered over. runs/interference records this specialist's
    # profile and its checkpoint
    # (checkpoints/_preserved/backward_1_1_best_honest_greedy_047.pt),
    # so the profile half of this row is manifest-recorded like the
    # rest. The metrics file below is that same manifest's
    # reference_role: prior_band_only receipt run, NOT the 43/100
    # policy's own training log: no run_manifest.json under any
    # checkpoints/mario_1_1_backward* records a profile or config path
    # at all, so no metrics file in this repo is manifest-tied to that
    # checkpoint. The row's name says so, and ADDENDUM RL-1 in
    # CLAIMS.md claims no more than this row can carry.
    {
        "name": "1-1 43/100 (metrics: manifest reference-receipt run)",
        "manifest": "runs/interference/manifest.json",
        "profile_key": ("config", "specialists", "1-1", "profile"),
        "expect_profile": "configs/mario_1_1_backward.yaml",
        "phase_configs_min": 0,
        "metrics": "checkpoints/mario_1_1_consolidate_exp/metrics.jsonl",
        "metrics_rows_min": 120,
    },
    {
        "name": "1-2 38/100 shared-stream, 31/100 canonical",
        "manifest": "runs/consol2/manifest.json",
        "profile_key": ("config", "base_profile"),
        "expect_profile": "configs/mario_1_2_consol2.yaml",
        "phase_configs_min": 0,
        "metrics": "checkpoints/mario_1_2_consol2/metrics.jsonl",
        "metrics_rows_min": 326,
    },
    {
        "name": "1-3 21/100",
        "manifest": "runs/consol2_1_3/manifest.json",
        "profile_key": ("config", "base_profile"),
        "expect_profile": "configs/mario_1_3_online_v1.yaml",
        "phase_configs_min": 1,
        "metrics": "checkpoints/mario_1_3_online_v1/metrics.jsonl",
        "metrics_rows_min": 30,
    },
    {
        "name": "1-3 21/100 (round 2)",
        "manifest": "runs/consol2_1_3_round2/manifest.json",
        "profile_key": ("config", "base_profile"),
        "expect_profile": "configs/mario_1_3_online_v1.yaml",
        "phase_configs_min": 1,
        "metrics": "checkpoints/mario_1_3_online_v1/metrics.jsonl",
        "metrics_rows_min": 30,
    },
    {
        "name": "1-4 51/100",
        "manifest": "runs/online_1_4/manifest.json",
        "profile_key": ("config", "base_profile"),
        "expect_profile": "configs/mario_1_4_online_v1.yaml",
        "phase_configs_min": 1,
        "metrics": "checkpoints/mario_1_4_online_v1/metrics.jsonl",
        "metrics_rows_min": 109,
    },
)

BANNED_REWARD_KEY = "reward_checkpoint"
REQUIRED_REWARD_KEY = "reward_forward"
LADDER_KEY = "checkpoint_scale"


def _dig(obj, key_path):
    """Follow key_path through nested dicts. Returns (value, None) or
    (None, the prefix that was missing)."""
    cur = obj
    for i, k in enumerate(key_path):
        if not isinstance(cur, dict) or k not in cur:
            return None, ".".join(str(p) for p in key_path[: i + 1])
        cur = cur[k]
    return cur, None


def _ladder_value(doc):
    """The reward_weights.checkpoint_scale of a loaded profile, or None
    when the profile does not set it."""
    if not isinstance(doc, dict):
        return None
    weights = doc.get("reward_weights")
    if not isinstance(weights, dict) or LADDER_KEY not in weights:
        return None
    return weights[LADDER_KEY]


def _load_yaml(path, problems, label):
    try:
        return yaml.safe_load(path.read_text())
    except Exception as exc:                      # noqa: BLE001 - reported
        problems.append(f"{label}: {path} does not parse as YAML ({exc})")
        return None


def check_lineage(spec, repo=REPO):
    """Return a list of human-readable violations for one lineage.

    Empty list means: this lineage's manifest, profile, recorded phase
    configs and metrics all say the dense x-ladder never paid.
    """
    problems = []
    name = spec["name"]
    repo = Path(repo)

    manifest_path = repo / spec["manifest"]
    if not manifest_path.is_file():
        problems.append(f"{name}: manifest {spec['manifest']} is missing")
        return problems
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:                      # noqa: BLE001 - reported
        problems.append(f"{name}: manifest {spec['manifest']} does not parse ({exc})")
        return problems

    key_path = tuple(spec["profile_key"])
    pretty_key = ".".join(str(k) for k in key_path)
    profile_rel, missing = _dig(manifest, key_path)
    if missing is not None:
        problems.append(
            f"{name}: manifest {spec['manifest']} has no profile pointer at "
            f"{pretty_key} (missing at {missing})")
        return problems
    if not isinstance(profile_rel, str):
        problems.append(
            f"{name}: manifest {spec['manifest']} key {pretty_key} is "
            f"{profile_rel!r}, not a profile path")
        return problems

    if profile_rel != spec["expect_profile"]:
        problems.append(
            f"{name}: manifest {spec['manifest']} key {pretty_key} records "
            f"{profile_rel}, but this lineage is banked against "
            f"{spec['expect_profile']}")

    profile_path = repo / profile_rel
    if not profile_path.is_file():
        problems.append(f"{name}: recorded profile {profile_rel} is missing")
    else:
        doc = _load_yaml(profile_path, problems, name)
        if doc is not None:
            scale = _ladder_value(doc)
            if scale is None:
                problems.append(
                    f"{name}: recorded profile {profile_rel} does not set "
                    f"reward_weights.{LADDER_KEY}; the lineage claim needs it "
                    f"written down, not inherited")
            elif float(scale) != 0.0:
                problems.append(
                    f"{name}: recorded profile {profile_rel} sets "
                    f"reward_weights.{LADDER_KEY}: {scale}, not 0.0")

    run_dir = manifest_path.parent
    phase_dir = run_dir / "phase_configs"
    phase_files = sorted(phase_dir.glob("*.yaml")) if phase_dir.is_dir() else []
    if len(phase_files) < spec["phase_configs_min"]:
        problems.append(
            f"{name}: {phase_dir.relative_to(repo)} holds {len(phase_files)} "
            f"phase configs, fewer than the {spec['phase_configs_min']} this "
            f"lineage recorded; the check would pass vacuously")
    for pf in phase_files:
        doc = _load_yaml(pf, problems, name)
        if doc is None:
            continue
        scale = _ladder_value(doc)
        if scale is not None and float(scale) != 0.0:
            problems.append(
                f"{name}: phase config {pf.relative_to(repo)} sets "
                f"reward_weights.{LADDER_KEY}: {scale}, not 0.0")

    problems.extend(check_metrics(spec, repo))
    return problems


def check_metrics(spec, repo=REPO):
    """Violations found in one lineage's metrics.jsonl."""
    problems = []
    name = spec["name"]
    metrics_path = Path(repo) / spec["metrics"]
    if not metrics_path.is_file():
        problems.append(f"{name}: metrics {spec['metrics']} is missing")
        return problems

    rows = 0
    banned_rows = []
    saw_required = False
    for n, line in enumerate(metrics_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except Exception as exc:                  # noqa: BLE001 - reported
            problems.append(
                f"{name}: {spec['metrics']}:{n} does not parse as JSON ({exc})")
            continue
        keys = _all_keys(row)
        if BANNED_REWARD_KEY in keys:
            banned_rows.append(n)
        if REQUIRED_REWARD_KEY in keys:
            saw_required = True

    if banned_rows:
        shown = ", ".join(str(n) for n in banned_rows[:5])
        more = "" if len(banned_rows) <= 5 else f" (+{len(banned_rows) - 5} more)"
        problems.append(
            f"{name}: {spec['metrics']} logs {BANNED_REWARD_KEY} on "
            f"{len(banned_rows)} row(s): {shown}{more}; the dense x-ladder "
            f"paid in this lineage")
    if rows < spec["metrics_rows_min"]:
        problems.append(
            f"{name}: {spec['metrics']} has {rows} rows, fewer than the "
            f"{spec['metrics_rows_min']} banked; a truncated log cannot clear "
            f"this lineage")
    elif not saw_required:
        problems.append(
            f"{name}: {spec['metrics']} logs no {REQUIRED_REWARD_KEY} in any "
            f"row, so its silence about {BANNED_REWARD_KEY} proves nothing")
    return problems


def _all_keys(obj):
    """Every dict key anywhere in a decoded JSON row."""
    out = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            out.update(k for k in cur if isinstance(k, str))
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def check_all(repo=REPO, lineages=LINEAGES):
    return {spec["name"]: check_lineage(spec, repo) for spec in lineages}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(REPO),
                    help="tree to check (default: this checkout)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    results = check_all(Path(args.repo))
    failed = sum(1 for v in results.values() if v)

    if args.json:
        print(json.dumps({
            "repo": args.repo,
            "lineages": len(results),
            "failed": failed,
            "violations": results,
        }, indent=2, sort_keys=True))
    else:
        for spec in LINEAGES:
            hits = results[spec["name"]]
            mark = "FAIL" if hits else "ok  "
            print(f"{mark} {spec['name']:38s} {spec['manifest']}")
            for h in hits:
                print(f"       {h}")
        print()
        if failed:
            print(f"LINEAGE GUARD FAILED: {failed} of {len(results)} banked "
                  f"lineages did not clear the dense x-ladder check")
        else:
            print(f"lineage guard clean: {len(results)} banked lineages, every "
                  f"recorded profile and phase config at {LADDER_KEY}: 0.0, no "
                  f"{BANNED_REWARD_KEY} in any metrics row")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
