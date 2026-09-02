"""Shared-substrate (trunk + per-level heads) evaluation harness.

THE QUESTION. An external research consultation (docs/proposals/
RESEARCH_SYNTHESIS_2026-08-17.md, round v20) ranked "shared substrate
with per-level heads" #1 of five interventions, projecting large
sample-efficiency gains across the 28 remaining SMB levels. A NAIVE
version was already falsified in this repo
(scripts/interference_falsifier.py): pooling 1-1 and 1-2 success
trajectories into one undifferentiated 200k-param net CAPTURED 1-1
(52/100 strict, beating its own 43/100 specialist) and DESTROYED 1-2
(4/100 vs 42/100) -- partial_interference, banked in
docs/receipts/one_three_coordination_2026-08-16.md. Trunk-plus-heads is
the untested form: ONE shared representation, ISOLATED per-level output
heads. The hypothesis is that this keeps the transfer the naive net
showed on 1-1 while removing the interference that ate 1-2.

This module does not train that net. It SCORES one, honestly, against
the isolated-specialist baseline this repo already owns -- the same
eval_game.py protocol the specialists were measured on, run per
exported head, per level.

BASELINE (all definitive, two-seed [7, 101], 100-episode, strict
predicate, cold entrance + greedy + sticky-0.25 + jitter-16 --
CLAIMS.md / docs/proposals/RESUME_2026-08-17.md):
    1-1  43/100  backward_1_1_best_honest_greedy_047.pt
    1-2  38/100  consol2_40pct_strict_iter01120.pt
    1-3  21/100  one_three_FINAL_consol2_iter00690.pt
    1-4  51/100  one_four_MEASURED60_iter00960.pt
    sum 153/400

For each level this harness assembles the SAME eval_game.py argv shape
the campaign controller uses for its honest probes
(scripts/run_online_campaign.py build_probe_command) against that
level's own profile, own cold-entrance start state (the profile's own
start_state_path -- the same blob every banked number above was
measured from), greedy, sticky 0.25, jitter 16, per-episode RNG, 5
parallel workers, --sequential --level-clear, 50 episodes on each of
two fresh seeds. That is THIS harness's own fixed protocol for scoring
the shared-substrate export -- it is NOT a claim that every banked
baseline number on the other side of the comparison was itself
measured under that same shape. It was not, uniformly:
CONFIG["baseline_provenance"] records, per level and sourced straight
from the receipt file(s) that actually produced each strict_clears
count, which legs match this protocol (1-3, 1-4 match exactly; 1-1
matches in shape but comes from one 100-episode seed rather than a
two-seed 50/50 split -- immaterial to its count) and which do not (1-2:
measured at eval_workers=1, eval_rng=shared-stream, no
--sequential/--level-clear -- runs/consol2/peak_eval_seed{7,101}.json).
Re-running 1-2's own baseline checkpoint under THIS harness's protocol
reads 43/100 (runs/interference/interference.jsonl, probe/1-2/control)
against the banked 38/100 -- a 5-point/13% swing from protocol choice
alone, about half the width of `level_collapse_margin` (10). Treat
1-2's leg of this comparison as the noisiest one, for exactly that
reason. The strict predicate (episode_success -- flagpole/castle) is
eval_game.py's `clear_rate` field; chain-advance (`seq_clear_rate`)
never substitutes (the 2026-08-15 forensics: the two disagree 13:1 on
real data -- docs/receipts/eval_rng_regimes_2026-08-15.md).

VERDICT (pre-registered in CONFIG, mirrored verbatim into the run's
manifest.json):
    SUPERSEDES -- aggregate strict clears beat 153/400 AND no
                  individual level falls more than
                  `level_collapse_margin` (10) points below its own
                  baseline. A substrate that wins on aggregate by
                  destroying one level is the interference failure
                  wearing a disguise, not a win.
    MIXED      -- aggregate beats baseline but at least one level
                  collapses past the margin.
    FAILED     -- aggregate does not beat baseline_sum.
    UNUSABLE   -- any eval leg errors, times out, or returns no strict
                  predicate. A broken harness is never a verdict.
An exact one-sided binomial test (aggregate_significance, reusing the
exact-binomial machinery from scripts/interference_falsifier.py) is
reported alongside the raw-count comparison as ADVISORY statistical
context -- it never substitutes for the raw-count rule above, and its
pooled-null assumption is documented at its definition.

`--dry-run` assembles every eval_game.py command against real paths
(profile, ROM, start state, baseline checkpoint), shape-infers all four
baseline checkpoints (also confirming they share one observation
width -- a precondition for a literal shared trunk), classifies the
verdict rule against registered synthetic cases, prints the verdict
table with PENDING placeholders, writes the pre-registered manifest,
and RUNS NOTHING against the emulator. `--checkpoint-dir` names where
the (not yet trained) exported per-level heads will live; their absence
during --dry-run is reported but never fails the dry-run -- that
export is the deferred, not-yet-run half of this experiment. Actually
scoring a real export is `run()` / plain invocation with no --dry-run,
which subprocesses eval_game.py and therefore steps the emulator: do
not invoke it while another campaign holds the pool.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reused, not reinvented: the exact-binomial decision machinery and the
# jsonl/yaml plumbing already live in the house's other pre-registered
# gate runners.
from scripts.interference_falsifier import (  # noqa: E402
    binom_p_at_least, binom_p_at_most,
)
from scripts.run_online_campaign import _append_jsonl, _load_yaml  # noqa: E402
from src.utils.run_lock import (  # noqa: E402
    acquire_resource, release_resource,
)

POOL_RESOURCE = "emulator-pool"

# ---------------------------------------------------------------------------
# CONFIG -- every threshold this harness runs on. Mirrored verbatim into
# runs/shared_substrate/manifest.json so pre-registration and run cannot
# drift apart.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "run_dir": "runs/shared_substrate",
    "receipt_log": "runs/shared_substrate/eval_shared_substrate.jsonl",
    "manifest_out": "runs/shared_substrate/manifest.json",

    # Level order is load-bearing: it fixes the aggregate and the
    # printed table order.
    "levels": ("1-1", "1-2", "1-3", "1-4"),

    # The isolated-specialist baseline this experiment must beat in
    # aggregate. Every number is DEFINITIVE (CLAIMS.md /
    # docs/proposals/RESUME_2026-08-17.md), never re-derived here.
    "baseline": {
        "1-1": {
            "game": "mario_1_1_backward",
            "profile": "configs/mario_1_1_backward.yaml",
            "checkpoint":
                "checkpoints/_preserved/"
                "backward_1_1_best_honest_greedy_047.pt",
            "strict_clears": 43,
        },
        "1-2": {
            "game": "mario_1_2_consol2",
            "profile": "configs/mario_1_2_consol2.yaml",
            "checkpoint":
                "checkpoints/_preserved/consol2_40pct_strict_iter01120.pt",
            "strict_clears": 38,
        },
        "1-3": {
            "game": "mario_1_3_online_v1",
            "profile": "configs/mario_1_3_online_v1.yaml",
            "checkpoint":
                "checkpoints/_preserved/"
                "one_three_FINAL_consol2_iter00690.pt",
            "strict_clears": 21,
        },
        "1-4": {
            "game": "mario_1_4_online_v1",
            "profile": "configs/mario_1_4_online_v1.yaml",
            "checkpoint":
                "checkpoints/_preserved/one_four_MEASURED60_iter00960.pt",
            "strict_clears": 51,
        },
    },
    "baseline_sum": 153,
    "baseline_n_episodes": 400,

    # WHERE and UNDER WHAT PROTOCOL each "baseline" strict_clears count
    # above actually came from -- verified by hand against the receipt
    # file(s) on 2026-08-17 and re-verified against those same files by
    # tests/test_eval_shared_substrate.py's
    # test_baseline_provenance_* tests, so this table cannot silently
    # drift from the receipts it describes. This is NOT the same thing
    # as "episodes_per_seed" / "eval_workers" / "eval_rng" below, which
    # is this harness's OWN fixed protocol for scoring the
    # shared-substrate export -- see "matches_harness_protocol" on each
    # entry for whether the two agree.
    "baseline_provenance": {
        "1-1": {
            "receipts": ("runs/interference/interference.jsonl",),
            "receipt_selector": {
                "type": "probe", "level": "1-1", "role": "control"},
            "eval_seeds": (20260816,),
            "n_episodes_total": 100,
            "sequential": True,
            "eval_workers": 5,
            "eval_rng": "per-episode",
            "matches_harness_protocol": True,
            "matches_two_seed_fifty_split": False,
            "note": (
                "one 100-episode eval-seed (20260816), not the two-seed "
                "[7, 101] x 50 split this harness's own re-eval "
                "assembles -- protocol SHAPE (sequential/level-clear, "
                "eval_workers=5, per-episode RNG) matches exactly, only "
                "the seed/episode split differs. Immaterial to the "
                "43/100 count."),
        },
        "1-2": {
            "receipts": ("runs/consol2/peak_eval_seed7.json",
                          "runs/consol2/peak_eval_seed101.json"),
            "eval_seeds": (7, 101),
            "n_episodes_total": 100,
            "sequential": False,
            "eval_workers": 1,
            "eval_rng": "shared-stream",
            "matches_harness_protocol": False,
            "matches_two_seed_fifty_split": True,
            "note": (
                "measured under a DIFFERENT protocol than this harness's "
                "own re-eval shape: eval_rng shared-stream (not "
                "per-episode), eval_workers 1 (not 5), no "
                "--sequential/--level-clear (the harness always passes "
                "both) -- see runs/interference/interference.jsonl's own "
                "reference_protocol_deltas note for 1-2. Re-running this "
                "exact checkpoint under the harness's own protocol "
                "(runs/interference/interference.jsonl, probe/1-2/"
                "control, eval-seed 20260816) reads clear_rate=0.43 "
                "(43/100) against the banked 38/100 here -- a 5-point/"
                "13% swing from protocol alone, about half the width of "
                "level_collapse_margin (10). The noisiest leg of this "
                "comparison, and the one the whole experiment's "
                "narrative is about."),
        },
        "1-3": {
            "receipts": ("runs/consol2_1_3_round2/final_eval_seed7.json",
                          "runs/consol2_1_3_round2/final_eval_seed101.json"),
            "eval_seeds": (7, 101),
            "n_episodes_total": 100,
            "sequential": True,
            "eval_workers": 5,
            "eval_rng": "per-episode",
            "matches_harness_protocol": True,
            "matches_two_seed_fifty_split": True,
            "note": "protocol shape and seed/episode split both match "
                    "this harness's own re-eval exactly.",
        },
        "1-4": {
            "receipts": ("runs/online_1_4/final_eval_seed7.json",
                          "runs/online_1_4/final_eval_seed101.json"),
            "eval_seeds": (7, 101),
            "n_episodes_total": 100,
            "sequential": True,
            "eval_workers": 5,
            "eval_rng": "per-episode",
            "matches_harness_protocol": True,
            "matches_two_seed_fifty_split": True,
            "note": "protocol shape and seed/episode split both match "
                    "this harness's own re-eval exactly.",
        },
    },

    # Where the (not yet trained) shared-substrate export will live.
    # One checkpoint per level, named "<level>.pt" (e.g. "1-1.pt") inside
    # this directory, matching train_shared_substrate.py's own
    # export_checkpoints() naming -- see head_checkpoint_path().
    "default_checkpoint_dir": "checkpoints/shared_substrate_v1",

    # This harness's OWN fixed re-eval protocol, used to score the
    # shared-substrate export: cold entrance (the level's own profile
    # start_state_path), greedy, sticky 0.25, jitter 16, per-episode
    # RNG, 5 parallel workers, --sequential --level-clear, 50 episodes
    # on each of two fresh seeds. Whether each BASELINE number was
    # itself measured under this same shape is answered per level by
    # "baseline_provenance" above, not assumed here -- it was not, for
    # 1-2.
    "episodes_per_seed": 50,
    "eval_seeds": (7, 101),
    "max_steps": 3000,
    "sticky_prob": 0.25,
    "start_jitter": 16,
    "eval_workers": 5,
    "eval_rng": "per-episode",
    "probe_timeout_s": 7200.0,

    # ---- Verdict thresholds, pre-registered (strict predicate ONLY) --
    # A drop strictly greater than this many points below a level's own
    # baseline counts as "collapsed" -- exactly 10 points down does not.
    "level_collapse_margin": 10,
    # One-sided exact-binomial alpha for the advisory aggregate
    # significance test (see aggregate_significance()).
    "aggregate_alpha": 0.05,
}


# ---------------------------------------------------------------------------
# Pure pieces -- unit-tested on synthetic inputs
# (tests/test_eval_shared_substrate.py). No filesystem, no subprocess,
# no emulator.
# ---------------------------------------------------------------------------


def head_checkpoint_path(level: str, checkpoint_dir, cfg: Optional[dict] = None) -> Path:
    """Where the exported per-level head for `level` lives inside
    `checkpoint_dir` -- "1-1" -> "1-1.pt". Matches
    scripts/train_shared_substrate.py's own export_checkpoints()
    naming (`_slug(level) + ".pt"`, which only touches "/" -- SMB level
    ids have none, so the dash passes through unchanged) so the two
    halves of this experiment agree on where the export lands without
    either one needing to know about the other's internals."""
    del cfg  # naming convention does not depend on CONFIG
    return Path(checkpoint_dir) / f"{level}.pt"


def build_level_command(
    level: str, *, checkpoint, seed: int, cfg: Optional[dict] = None,
) -> list[str]:
    """The honest-eval eval_game.py argv for one level, one seed.

    Same shape as run_online_campaign.build_probe_command /
    interference_falsifier._eval_game_command: explicit --profile and
    --rom (so --game never falls through DEFAULT_PROFILES/DEFAULT_ROMS
    lookup), explicit --start-state from the PROFILE's own
    start_state_path -- the level's own cold entrance, the exact blob
    every banked baseline number was measured from (verified against
    docs/proposals/RESUME_2026-08-17.md and
    docs/receipts/one_three_coordination_2026-08-16.md for all four
    levels). Greedy is eval_game's default action-select; this harness
    never overrides it.
    """
    cfg = cfg or CONFIG
    spec = cfg["baseline"][level]
    profile_path = REPO / spec["profile"]
    base = _load_yaml(profile_path)
    return [
        sys.executable, str(REPO / "scripts" / "eval_game.py"),
        "--game", str(spec["game"]),
        "--profile", str(profile_path),
        "--rom", str(REPO / base["rom_path"]),
        "--checkpoint", str(checkpoint),
        "--episodes", str(int(cfg["episodes_per_seed"])),
        "--max-steps", str(int(cfg["max_steps"])),
        "--sequential", "--level-clear",
        "--start-state", str(REPO / base["start_state_path"]),
        "--eval-seed", str(int(seed)),
        "--sticky-prob", str(float(cfg["sticky_prob"])),
        "--start-jitter", str(int(cfg["start_jitter"])),
        "--eval-workers", str(int(cfg["eval_workers"])),
        "--eval-rng", str(cfg["eval_rng"]),
    ]


def strict_clears_from_eval_json(eval_json: Any) -> Optional[int]:
    """One eval_game.py JSON result -> its strict clear COUNT (not
    rate), or None with the caller left to record why.

    `clear_rate` is eval_game's episode_success (flagpole/castle)
    predicate at the top level -- the strict one. `seq_clear_rate`
    (chain-advance) never substitutes here (2026-08-15 forensics: the
    two disagreed 13:1 on real data). None on any non-"ok" status, a
    non-positive episode count, or a missing clear_rate -- a broken or
    empty leg is never silently scored as zero clears (zero clears and
    "the probe never ran" must never look the same downstream)."""
    if not isinstance(eval_json, dict):
        return None
    if eval_json.get("status") != "ok":
        return None
    n = int(eval_json.get("n_episodes", 0) or 0)
    if n <= 0:
        return None
    rate = eval_json.get("clear_rate")
    if rate is None:
        return None
    return int(round(float(rate) * n))


def level_delta(level: str, leg_result: dict, cfg: Optional[dict] = None) -> dict:
    """One level's baseline/shared/delta row, plus the collapse flag."""
    cfg = cfg or CONFIG
    baseline = int(cfg["baseline"][level]["strict_clears"])
    shared = leg_result.get("strict_clears")
    delta = None if shared is None else int(shared) - baseline
    collapsed = delta is not None and delta < -int(cfg["level_collapse_margin"])
    return {
        "level": level,
        "baseline": baseline,
        "shared": shared,
        "delta": delta,
        "collapsed": bool(collapsed),
        "status": leg_result.get("status"),
        "errors": list(leg_result.get("errors") or []),
    }


def aggregate_result(level_deltas: list[dict], cfg: Optional[dict] = None) -> dict:
    """The total-strict-clears comparison against baseline_sum.

    `shared_sum`/`delta`/`beats_baseline` are None whenever any level
    errored -- an aggregate cannot be computed from a leg that never
    produced a number, and must not silently drop that leg's
    contribution instead."""
    cfg = cfg or CONFIG
    baseline_sum = int(cfg["baseline_sum"])
    errored = [d["level"] for d in level_deltas if d.get("status") != "ok"]
    collapsed = [d["level"] for d in level_deltas if d.get("collapsed")]
    if errored:
        return {
            "baseline_sum": baseline_sum, "shared_sum": None, "delta": None,
            "beats_baseline": None, "collapsed_levels": collapsed,
            "errored_levels": errored,
        }
    shared_sum = sum(int(d["shared"]) for d in level_deltas)
    return {
        "baseline_sum": baseline_sum, "shared_sum": shared_sum,
        "delta": shared_sum - baseline_sum,
        "beats_baseline": shared_sum > baseline_sum,
        "collapsed_levels": collapsed, "errored_levels": [],
    }


def aggregate_significance(shared_sum: int, cfg: Optional[dict] = None) -> dict:
    """One-sided exact binomial test of the pooled aggregate against
    the pooled baseline rate.

    ASSUMPTIONS -- read alongside the p-value, never instead of it:
    treats all `baseline_n_episodes` (400: 4 levels x 2 seeds x 50
    episodes) evaluation episodes as INDEPENDENT Bernoulli trials at
    one pooled null rate p0 = baseline_sum / baseline_n_episodes
    (153/400 = 0.3825) -- the same pooled-null exact-binomial
    construction scripts/interference_falsifier.py's leg_decision uses,
    reused here (binom_p_at_least/binom_p_at_most) rather than
    reinvented. This is an APPROXIMATION: the four levels have
    different true per-level rates (43%, 38%, 21%, 51%), so the pooled
    binomial is not the exact sampling distribution of a sum of four
    heterogeneous binomials. It is reported as ADVISORY statistical
    context alongside the raw-count verdict rule (SUPERSEDES / MIXED /
    FAILED / UNUSABLE) -- it never substitutes for that rule."""
    cfg = cfg or CONFIG
    n = int(cfg["baseline_n_episodes"])
    p0 = float(cfg["baseline_sum"]) / n
    alpha = float(cfg["aggregate_alpha"])
    k = int(shared_sum)
    p_above = binom_p_at_least(k, n, p0)
    p_below = binom_p_at_most(k, n, p0)
    if p_above <= alpha:
        decision = "significantly_above_baseline"
    elif p_below <= alpha:
        decision = "significantly_below_baseline"
    else:
        decision = "indistinguishable_from_baseline"
    return {
        "n": n, "p0": p0, "alpha": alpha, "k": k,
        "p_above": p_above, "p_below": p_below, "decision": decision,
    }


VERDICTS = ("SUPERSEDES", "MIXED", "FAILED", "UNUSABLE")


def classify_verdict(level_deltas: list[dict], cfg: Optional[dict] = None) -> dict:
    """THE pre-registered verdict over the four per-level legs.

    Precedence: UNUSABLE (any leg broken) is checked first and pre-empts
    everything else -- a broken harness is never a verdict. Otherwise:
    SUPERSEDES (beats baseline, nothing collapsed), MIXED (beats
    baseline, something collapsed -- the interference failure wearing a
    disguise), else FAILED (does not beat baseline)."""
    cfg = cfg or CONFIG
    errored = [d["level"] for d in level_deltas if d.get("status") != "ok"]
    reasons: list[str] = []
    if errored:
        reasons.append(
            f"eval leg(s) errored or returned no strict predicate: "
            f"{errored} -- a broken probe is never a verdict")
        return {"verdict": "UNUSABLE", "reasons": reasons,
                "aggregate": aggregate_result(level_deltas, cfg)}

    agg = aggregate_result(level_deltas, cfg)
    collapsed = agg["collapsed_levels"]
    margin = int(cfg["level_collapse_margin"])
    if agg["beats_baseline"] and not collapsed:
        verdict = "SUPERSEDES"
    elif agg["beats_baseline"] and collapsed:
        verdict = "MIXED"
        reasons.append(
            f"aggregate beats baseline ({agg['shared_sum']} > "
            f"{agg['baseline_sum']}) but {collapsed} fell more than "
            f"{margin} points below its own baseline -- the "
            f"interference failure wearing a disguise")
    else:
        verdict = "FAILED"
        reasons.append(
            f"aggregate does not beat baseline "
            f"({agg['shared_sum']} <= {agg['baseline_sum']})")
    return {"verdict": verdict, "reasons": reasons, "aggregate": agg}


def format_verdict_table(
    level_deltas: list[dict], verdict: dict, cfg: Optional[dict] = None,
) -> str:
    """The human-readable table: per-level baseline/shared/delta, the
    aggregate row, and the verdict + its reasons."""
    cfg = cfg or CONFIG
    lines = [
        f"{'level':<6}{'baseline':>10}{'shared':>10}{'delta':>10}",
        "-" * 36,
    ]
    for d in level_deltas:
        shared = "PENDING" if d["shared"] is None else str(d["shared"])
        delta = "PENDING" if d["delta"] is None else f"{d['delta']:+d}"
        flag = "  *collapsed*" if d.get("collapsed") else ""
        lines.append(
            f"{d['level']:<6}{d['baseline']:>10}{shared:>10}{delta:>10}"
            f"{flag}")
    agg = verdict["aggregate"]
    shared_sum = ("PENDING" if agg.get("shared_sum") is None
                  else str(agg["shared_sum"]))
    delta = ("PENDING" if agg.get("delta") is None
             else f"{agg['delta']:+d}")
    lines.append("-" * 36)
    lines.append(
        f"{'TOTAL':<6}{agg['baseline_sum']:>10}{shared_sum:>10}{delta:>10}")
    lines.append(f"VERDICT: {verdict['verdict']}")
    for r in verdict["reasons"]:
        lines.append(f"  - {r}")
    return "\n".join(lines)


def placeholder_table(cfg: Optional[dict] = None) -> str:
    """The --dry-run table: real baseline numbers, PENDING everywhere
    an actual eval leg would report, verdict PENDING. No leg has run."""
    cfg = cfg or CONFIG
    level_deltas = [
        {"level": lvl, "baseline": int(cfg["baseline"][lvl]["strict_clears"]),
         "shared": None, "delta": None, "collapsed": False, "status": "ok"}
        for lvl in cfg["levels"]
    ]
    verdict = {
        "verdict": "PENDING",
        "reasons": ["evaluation deferred -- the emulator is busy with "
                   "another campaign; see the command printed below"],
        "aggregate": aggregate_result(
            [{**d, "status": "pending"} for d in level_deltas], cfg),
    }
    return format_verdict_table(level_deltas, verdict, cfg)


def build_manifest(
    cfg: Optional[dict] = None, *, level_deltas: Optional[list[dict]] = None,
) -> dict:
    """CONFIG + the verdict pre-registration, verbatim, for
    runs/shared_substrate/manifest.json. With `level_deltas` (a real
    run's per-level rows) it also carries the computed verdict and the
    advisory aggregate significance test; without them (--dry-run) it
    is the pre-registration alone, status "pending"."""
    cfg = cfg or CONFIG
    margin = int(cfg["level_collapse_margin"])
    manifest: dict[str, Any] = {
        "experiment": "shared_substrate_trunk_plus_heads",
        "hypothesis": (
            "a single shared-trunk network with per-level output heads "
            "keeps the cross-level transfer a naive pooled net showed "
            "on 1-1 while removing the interference that destroyed 1-2 "
            "(interference_falsifier's partial_interference verdict, "
            "docs/receipts/one_three_coordination_2026-08-16.md)"),
        "config": copy.deepcopy(cfg),
        "verdict_rules": {
            "predicate": "clear_rate (episode_success, strict) ONLY -- "
                        "chain-advance (seq_clear_rate) never "
                        "substitutes",
            "SUPERSEDES": (
                f"aggregate strict clears > baseline_sum AND no "
                f"individual level falls more than {margin} points "
                f"below its own baseline"),
            "MIXED": (
                "aggregate beats baseline but at least one level "
                "collapses past the margin -- the interference failure "
                "wearing a disguise"),
            "FAILED": "aggregate does not beat baseline_sum",
            "UNUSABLE": "any eval leg errors, times out, or returns no "
                       "strict predicate",
            "precedence": "UNUSABLE > MIXED/SUPERSEDES/FAILED "
                          "(computed only once no leg is broken)",
            "aggregate_test": (
                "one-sided exact binomial test, pooled null "
                "p0 = baseline_sum / baseline_n_episodes -- ADVISORY "
                "statistical context alongside the raw-count "
                "comparison above, never a substitute for it; "
                "assumptions documented at aggregate_significance()"),
        },
        "status": "pending" if level_deltas is None else "evaluated",
    }
    if level_deltas is not None:
        manifest["level_deltas"] = level_deltas
        verdict = classify_verdict(level_deltas, cfg)
        manifest["verdict"] = verdict
        shared_sum = verdict["aggregate"].get("shared_sum")
        if shared_sum is not None:
            manifest["aggregate_significance"] = aggregate_significance(
                shared_sum, cfg)
    return manifest


# ---------------------------------------------------------------------------
# Runtime pieces (subprocess / filesystem / torch load; exercised by
# --dry-run with real paths and by the tests via injected stubs -- never
# by the pure-function unit tests above).
# ---------------------------------------------------------------------------


def write_manifest(path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _run_eval_subprocess(cmd: list[str], *, timeout_s: float) -> dict:
    """Runs one eval_game.py subprocess; returns its JSON (or an error
    dict). Same one-JSON-blob-on-stdout contract every other gate
    runner in this repo parses (run_online_campaign.run_probe,
    interference_falsifier.run_probe) -- reused shape, not reinvented."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "eval_timeout"}
    # Parse stdout before looking at the return code: eval_game.py's own
    # guards (e.g. no_rom) print a status JSON to stdout and then exit
    # non-zero, and that JSON is more informative than an empty stderr.
    text = proc.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            if proc.returncode == 0:
                return {"status": "eval_failed", "detail": f"bad JSON: {e}"}
    if proc.returncode != 0:
        return {"status": "eval_failed", "detail": proc.stderr.strip()[-500:]}
    return {"status": "eval_failed", "detail": "no JSON on stdout"}


def run_level_leg(
    level: str, *, checkpoint, cfg: Optional[dict] = None,
    run_eval: Optional[Callable[[list[str]], dict]] = None,
) -> dict:
    """The two-seed, 100-episode leg for one level's exported head.

    `run_eval` is the injection point: production leaves it None (a
    real eval_game.py subprocess per seed); tests pass a stub that
    returns synthetic eval_game JSON without touching the emulator."""
    cfg = cfg or CONFIG
    run_eval = run_eval or (
        lambda cmd: _run_eval_subprocess(cmd, timeout_s=cfg["probe_timeout_s"]))
    per_seed = []
    errors: list[str] = []
    total = 0
    for seed in cfg["eval_seeds"]:
        cmd = build_level_command(level, checkpoint=checkpoint, seed=seed, cfg=cfg)
        result = run_eval(cmd)
        clears = strict_clears_from_eval_json(result)
        per_seed.append({"seed": int(seed), "command": cmd, "result": result,
                         "strict_clears": clears})
        if clears is None:
            errors.append(
                f"seed {seed}: status={result.get('status')!r} "
                f"{result.get('detail', '')}".strip())
        else:
            total += clears
    status = "ok" if not errors else "error"
    return {
        "level": level, "checkpoint": str(checkpoint),
        "n_episodes": int(cfg["episodes_per_seed"]) * len(cfg["eval_seeds"]),
        "per_seed": per_seed,
        "strict_clears": total if status == "ok" else None,
        "status": status, "errors": errors,
    }


def dry_run(cfg: Optional[dict] = None, *, checkpoint_dir=None) -> int:
    """Validate the whole assembly without touching the emulator.
    Returns 0 if every REQUIRED check passes. The shared-substrate head
    checkpoints not existing yet is reported but never fails the
    dry-run -- their export is the deferred half of this experiment."""
    cfg = cfg or CONFIG
    checkpoint_dir = Path(checkpoint_dir or cfg["default_checkpoint_dir"])
    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" -- {detail}" if detail else ""), flush=True)

    def note(label: str, detail: str = "") -> None:
        print(f"[dry-run] INFO {label}" + (f" -- {detail}" if detail else ""),
              flush=True)

    # 1. Baseline sum matches the pre-registered total.
    computed_sum = sum(int(s["strict_clears"]) for s in cfg["baseline"].values())
    report("baseline sum matches the pre-registered total",
           computed_sum == int(cfg["baseline_sum"]),
           f"{computed_sum} vs {cfg['baseline_sum']}")

    # 2. Per level: profile/rom/start-state/baseline checkpoint are all
    #    real paths; the eval_game.py command assembles against them;
    #    the baseline checkpoint shape-infers into its own profile.
    obs_dims: dict[str, int] = {}
    for level in cfg["levels"]:
        spec = cfg["baseline"][level]
        profile_path = REPO / spec["profile"]
        try:
            base = _load_yaml(profile_path)
            rom_path = REPO / base["rom_path"]
            start_state = REPO / base["start_state_path"]
            baseline_ckpt = REPO / spec["checkpoint"]
            missing = [str(p) for p in
                       (profile_path, rom_path, start_state, baseline_ckpt)
                       if not p.exists()]
            report(f"{level} real paths (profile/rom/start-state/"
                   f"baseline checkpoint)", not missing,
                   "all present" if not missing else f"missing: {missing}")
        except Exception as e:
            report(f"{level} real paths", False, repr(e))
            continue

        try:
            cmds = [build_level_command(level, checkpoint=baseline_ckpt,
                                        seed=seed, cfg=cfg)
                   for seed in cfg["eval_seeds"]]
            missing_args = [a for cmd in cmds for a in cmd
                            if "/" in a and not a.startswith("-")
                            and not Path(a).exists()]
            report(f"{level} eval_game.py command assembles against "
                   f"real paths", not missing_args,
                   " && ".join(" ".join(c) for c in cmds)
                   if not missing_args else f"missing: {missing_args}")
        except Exception as e:
            report(f"{level} eval_game.py command assembles against "
                   f"real paths", False, repr(e))

        try:
            import torch
            from src.training.profile_utils import resolve_encoder
            from src.models.tile_policy import build_tile_policy_from_checkpoint
            _, _, stacked = resolve_encoder(base)
            obs_dims[level] = stacked
            payload = torch.load(str(baseline_ckpt), map_location="cpu",
                                 weights_only=False)
            net, is_recurrent = build_tile_policy_from_checkpoint(
                payload, num_actions=len(base["action_space"]),
                feature_dim=stacked)
            missing_keys, _ = net.load_state_dict(
                payload["net_state_dict"], strict=False)
            report(f"{level} baseline checkpoint loads with shape "
                   f"inference", not is_recurrent and not missing_keys,
                   f"h{net.hidden_dim}/trunk{net.trunk_dim} obs={stacked} "
                   f"recurrent={is_recurrent} missing={missing_keys}")
        except Exception as e:
            report(f"{level} baseline checkpoint loads with shape "
                   f"inference", False, repr(e))

        head_ckpt = head_checkpoint_path(level, checkpoint_dir, cfg)
        note(f"{level} shared-substrate head checkpoint",
             f"{head_ckpt} "
             + ("present" if head_ckpt.exists()
                else "PENDING -- not yet exported/trained; this level "
                     "cannot be scored until it exists"))

    # 3. A literal shared TRUNK requires one observation width across
    #    every level -- report whether that precondition holds.
    if obs_dims:
        widths = set(obs_dims.values())
        report("all four levels share one observation width "
               "(precondition for a literal shared trunk)",
               len(widths) == 1, f"{obs_dims}")

    # 4. The verdict rule classifies the registered synthetic cases,
    #    INCLUDING the disguised-interference (MIXED) case.
    try:
        def leg(shared, status="ok"):
            return {"strict_clears": shared, "status": status, "errors": []}

        cases = [
            # (per-level shared clears, expected verdict)
            ({"1-1": 50, "1-2": 45, "1-3": 25, "1-4": 55}, "SUPERSEDES"),
            ({"1-1": 60, "1-2": 20, "1-3": 25, "1-4": 55}, "MIXED"),
            ({"1-1": 40, "1-2": 35, "1-3": 20, "1-4": 50}, "FAILED"),
        ]
        got = []
        for shared_by_level, _want in cases:
            deltas = [level_delta(lvl, leg(shared_by_level[lvl]), cfg)
                     for lvl in cfg["levels"]]
            got.append(classify_verdict(deltas, cfg)["verdict"])
        # UNUSABLE case: one errored leg.
        errored_deltas = [
            level_delta(lvl, leg(None, status="error") if lvl == "1-2"
                       else leg(40), cfg)
            for lvl in cfg["levels"]]
        got.append(classify_verdict(errored_deltas, cfg)["verdict"])
        want = [w for _, w in cases] + ["UNUSABLE"]
        report("verdict rule classifies the registered synthetic cases "
               "(incl. disguised-interference/MIXED)", got == want,
               f"{got}")
    except Exception as e:
        report("verdict rule classifies the registered synthetic cases",
               False, repr(e))

    # 5. Manifest mirrors CONFIG + the verdict pre-registration; write
    #    it to disk (the pre-registration surface).
    try:
        man = build_manifest(cfg)
        good = (man["config"] == cfg and man["status"] == "pending"
               and set(man["verdict_rules"]) >= set(VERDICTS))
        write_manifest(REPO / cfg["manifest_out"], man)
        report("manifest mirrors CONFIG + verdict pre-registration", good,
               str(REPO / cfg["manifest_out"]))
    except Exception as e:
        report("manifest mirrors CONFIG + verdict pre-registration", False,
               repr(e))

    print(placeholder_table(cfg), flush=True)
    print(f"\n[dry-run] to run for real once the emulator is free "
          f"(this steps the pool -- do not run while another campaign "
          f"holds it):\n"
          f"    .venv/bin/python scripts/eval_shared_substrate.py "
          f"--checkpoint-dir {checkpoint_dir}", flush=True)

    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


def run(
    cfg: Optional[dict] = None, *, checkpoint_dir=None,
    run_eval: Optional[Callable[[list[str]], dict]] = None,
) -> int:
    """The real evaluation: one eval_game.py leg per level per seed
    against the exported shared-substrate head, scored against the
    banked baseline. STEPS THE EMULATOR (via eval_game.py subprocesses)
    -- never call this while another campaign holds the pool; enforced
    by the shared "emulator-pool" resource lock
    (src/utils/run_lock.acquire_resource) held for the whole call.
    `run_eval` is the test injection point; production leaves it None."""
    cfg = cfg or CONFIG
    holder = acquire_resource(POOL_RESOURCE, extra="eval_shared_substrate")
    if holder is not None:
        print(f"[eval_shared_substrate] {POOL_RESOURCE} lock held by live "
              f"PID {holder.pid} — refusing to step the pool while "
              "another writer holds it.", flush=True)
        return 75
    try:
        checkpoint_dir = Path(checkpoint_dir or cfg["default_checkpoint_dir"])
        level_deltas = []
        for level in cfg["levels"]:
            head_ckpt = head_checkpoint_path(level, checkpoint_dir, cfg)
            leg = run_level_leg(level, checkpoint=head_ckpt, cfg=cfg,
                                run_eval=run_eval)
            _append_jsonl(REPO / cfg["receipt_log"], {"type": "leg", **leg})
            level_deltas.append(level_delta(level, leg, cfg))

        manifest = build_manifest(cfg, level_deltas=level_deltas)
        write_manifest(REPO / cfg["manifest_out"], manifest)
        _append_jsonl(REPO / cfg["receipt_log"],
                     {"type": "verdict", **manifest["verdict"]})

        print(format_verdict_table(level_deltas, manifest["verdict"], cfg),
              flush=True)
        return 0 if manifest["verdict"]["verdict"] != "UNUSABLE" else 1
    finally:
        release_resource(POOL_RESOURCE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a shared-substrate (trunk + per-level heads) "
                    "SMB run against the isolated per-level baseline, "
                    "same honest eval_game.py protocol per level.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Assemble every command against real paths, print the "
             "verdict table with PENDING placeholders, write the "
             "pre-registered manifest, run nothing. Safe with the "
             "emulator busy.")
    parser.add_argument(
        "--out", type=str, default=None, metavar="DIR",
        help="Redirect ALL writes (receipt log, manifest) into DIR "
             "instead of the production run_dir. The test suite must "
             "always pass this: 12 days of mock verdicts accumulated "
             "in the production receipt log because a test invoked "
             "this CLI without redirection (MISTAKES.md 2026-08-29).")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, metavar="DIR",
        help="Directory holding the exported per-level shared-substrate "
             "head checkpoints (default: "
             f"{CONFIG['default_checkpoint_dir']}). One file per level, "
             "named <level>.pt (e.g. 1-1.pt) -- see "
             "head_checkpoint_path().")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or CONFIG["default_checkpoint_dir"]
    cfg = CONFIG
    if args.out:
        out = Path(args.out)
        cfg = {**CONFIG,
               "run_dir": str(out),
               "receipt_log": str(out / "eval_shared_substrate.jsonl"),
               "manifest_out": str(out / "manifest.json")}
    if args.dry_run:
        return dry_run(cfg, checkpoint_dir=checkpoint_dir)
    return run(cfg, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    sys.exit(main())
