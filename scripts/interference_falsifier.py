"""Generalist-interference falsifier (SMB 1-1 x 1-2) — the 2-hour
attended experiment from the accepted research spec, as a one-command
gated runner in the night2_runner mold.

THE QUESTION. Two specialists exist with honest strict receipts on
their own levels: the 1-1 backward policy preserved as
backward_1_1_best_honest_greedy_047.pt (byte-identical to
checkpoints/mario_1_1_consolidate_exp/best_1-1.pt — 56/120 clears
pooled over eval seeds 0/1/2 on the very entrance this runner uses) and
the consol2 1-2 peak at iter 1120 (38/100 over seeds 7/101 from
stage_03). Does ONE fresh net of the same capacity, trained jointly on
success trajectories from BOTH, keep both competences — or do the
levels interfere? The answer CALIBRATES the
generalist plan (whether a Level-ID token or more capacity is needed
before multi-level distillation); it does NOT gate the 1-3 campaign,
which the parallel lane runs independently off its own artifacts
(configs/mario_1_3_backward.yaml, checkpoints/backward_states/1-3/ —
read-only coordination references for this runner, never inputs).

Three gated steps, receipts to runs/interference/interference.jsonl:

STEP 1 — success-trajectory collection (the emulator step: assembled
  now, run in the attended window). Each specialist is rolled out
  SAMPLED (temperature 1.0) under the house noise — sticky 0.25, no-op
  start-jitter 16, per-episode RNG — on ITS OWN level: the 1-1 policy
  from the canonical 1-1 entrance (the backward tape's own root,
  runs/live_show/smb_4_4_micro/entrance_start.state — the profile
  declares "Eval with the SAME blob", and its receipted rate was
  measured from it) and the 1-2 peak from stage_03.state (the 1-2
  entrance every 1-2 campaign number is measured from). 300 episodes
  each, on collect_seed. Keep
  STRICT successes ONLY (episode_success — the flagpole/axe predicate;
  max_gx never substitutes here, unlike night2's OR-keep: the
  falsifier distills what the specialists finish, not what they
  reach). The collection loop is IMPORTED from night2_runner
  (collect_trajectories — eval_game's episode_rng_seeds/select_action
  mechanics, lane-count invariant); this module only assembles the
  per-level pool around it. Per-level npz in the house demo
  convention, then the pooled set is balanced 50/50 BY STATE-ACTION
  PAIRS (both sides trimmed to the smaller side's pair count, episode
  order preserved) — pair-level, not episode-level, because the BC
  loss is per-pair and 1-2 trajectories are systematically longer;
  episode-balance would let 1-2 dominate the gradient.

STEP 2 — joint BC: a FRESH TilePolicyNetwork at h256/trunk64 (~200k
  params — the SAME class + widths the eval stack shape-infers, and
  the same capacity as the 1-2 specialist, so a capacity excuse is off
  the table), 50 epochs cross-entropy on the pooled pairs, lr 3e-4
  with step decay (x0.5 every 20 epochs), batch 256, deterministic
  from CONFIG's seed. Saved to runs/interference/joint.pt in the
  standard payload (net_state_dict + iter) that eval_game loads.

STEP 3 — the FOUR-leg gate, via eval_game subprocesses. Each level is
  measured TWICE under one identical command — once on its own
  specialist (the POSITIVE CONTROL) and once on the joint net: 50
  episodes, cold ARGMAX, sticky 0.25, jitter 16, per-episode RNG, on
  gate_eval_seed, which is deliberately DISJOINT from collect_seed
  (eval_game's episode_rng_seeds is a pure function of
  (seed, episode_index), so a shared seed would grade the joint net on
  the same noise realizations its training trajectories came from).

  Two things follow from the control legs. First, the hold threshold is
  50% of the MEASURED control, so a harness or protocol shift moves the
  control, the joint leg and the threshold together instead of
  masquerading as interference. Second, nothing is compared against a
  rate transcribed from prose: the receipted priors in CONFIG
  (1-1 56/120 = 0.467, 1-2 38/100 = 0.38) are re-verified against their
  own files by the dry-run but are ADVISORY ONLY — they were measured
  with shared-stream RNG on 1 worker without --sequential, and are used
  solely as a drift band on the control.

  Each leg is decided by an exact one-sided binomial test at
  control_alpha, because at n=50 a point comparison reads sampling
  noise as signal. Verdict rules pre-registered in CONFIG (strict
  predicate only):
    unusable             — any leg failed/empty/strict-less, or a
                           positive control below 0.15;
    inconclusive         — some leg is statistically indeterminate at
                           this n;
    severe_interference  — both legs fail AND both joint rates < 0.10;
    compatible           — both legs hold;
    partial_interference — exactly one leg holds;
    degraded             — both fail, not both under the ceiling.
  A broken harness is never a verdict, and neither is a coin flip.

`--dry-run` validates the whole assembly without rollouts: both
profiles parse + strict schema + agree on action space and encoder
width, both start states exist and match their profiles, both
specialist checkpoints sha256-match their pins and load with shape
inference (1-1 at h64/trunk32, 1-2 at h256/trunk64 — different
capacities, same obs/action interface), the fresh joint net constructs
at ~200k params, all four eval commands assemble against real paths,
every reference rate re-verifies against its own receipt file on
weights byte-identical to the specialist rolled out, the gate seed is
disjoint from the collection seed, the registered episode count clears
its pre-registered discrimination power (a clone must be classified
`holds`, and a genuinely-interfered net `fails`, at least
min_clone_hold_power of the time — learned HERE, not after the attended
window), the verdict table classifies the registered synthetic cases,
and the manifest mirrors CONFIG. `--skip-to N` resumes at step N off
the prior steps' on-disk artifacts.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from math import comb
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reused verbatim from the campaign controllers (same jsonl plumbing,
# same probe JSON summary, same sha256) so receipts stay byte-comparable,
# and the collection loop is night2's — imported, not duplicated.
from scripts.night2_runner import collect_trajectories  # noqa: E402
from scripts.run_consol2 import _sha256  # noqa: E402
from scripts.run_online_campaign import (  # noqa: E402
    _append_jsonl, _load_yaml, probe_summary,
)

# ---------------------------------------------------------------------------
# CONFIG — every threshold the falsifier runs on. Mirrored verbatim into
# runs/interference/manifest.json so pre-registration and run cannot drift.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "run_dir": "runs/interference",
    "receipt_log": "runs/interference/interference.jsonl",

    # Level order is load-bearing: it fixes level_ids in the pooled set.
    "levels": ("1-1", "1-2"),
    "specialists": {
        "1-1": {
            # The 1-1 policy whose honest rate is RECEIPTED on the exact
            # entrance this falsifier evals from. Byte-identical
            # (sha256 f709c0d2...) to
            # checkpoints/mario_1_1_consolidate_exp/best_1-1.pt, the file
            # those receipt rows name.
            "profile": "configs/mario_1_1_backward.yaml",
            "game": "mario_1_1_backward",
            "checkpoint":
                "checkpoints/_preserved/"
                "backward_1_1_best_honest_greedy_047.pt",
            "checkpoint_sha256":
                "f709c0d2ac6ee23e73ae05967613bba176c97e510cba882b581"
                "bdb0142b76d54",
            # The canonical 1-1 entrance = the backward tape's own root
            # (the profile's start_state_path; "Eval with the SAME blob").
            "start_state":
                "runs/live_show/smb_4_4_micro/entrance_start.state",
            "pairs_out": "runs/interference/success_1_1.npz",
            # PRIOR ONLY — never the gate threshold. Every row is
            # re-read from its own file by verify_reference_receipts().
            "reference_role": "prior_band_only",
            "reference_receipts": [
                {"receipt": "checkpoints/mario_1_1_consolidate_exp/eval.jsonl",
                 "checkpoint":
                     "checkpoints/mario_1_1_consolidate_exp/best_1-1.pt",
                 "eval_seed": 0, "n_episodes": 40, "clears": 19,
                 "clear_rate": 0.475},
                {"receipt": "checkpoints/mario_1_1_consolidate_exp/eval.jsonl",
                 "checkpoint":
                     "checkpoints/mario_1_1_consolidate_exp/best_1-1.pt",
                 "eval_seed": 1, "n_episodes": 40, "clears": 19,
                 "clear_rate": 0.475},
                {"receipt": "checkpoints/mario_1_1_consolidate_exp/eval.jsonl",
                 "checkpoint":
                     "checkpoints/mario_1_1_consolidate_exp/best_1-1.pt",
                 "eval_seed": 2, "n_episodes": 40, "clears": 18,
                 "clear_rate": 0.45},
            ],
            "reference_protocol_deltas": [
                "eval_rng shared-stream (the gate runs per-episode)",
                "eval_workers 1 (the gate runs 5 workers)",
                "no --sequential/--level-clear (the gate passes both)",
            ],
            "rejected_candidates": [
                {"checkpoint":
                     "checkpoints/_preserved/backward_1_1_seed3_iter140.pt",
                 "reason":
                     "no eval receipt anywhere ties a clear rate to this "
                     "file; the 0.76 it had been credited with belongs to "
                     "the mario_1_1_tile_gate_v2_consolidate lineage "
                     "(vanilla_ppo_iter_00170.pt, n=30, start_state / "
                     "sticky_prob / start_jitter all unrecorded), not to "
                     "the backward lineage or this entrance. Its own "
                     "embedded backward_curriculum telemetry is "
                     "491/1487 = 0.330 entrance success."},
            ],
        },
        "1-2": {
            # The consol2 strict peak (40%-strict probe at iter 1120).
            "profile": "configs/mario_1_2_consol2.yaml",
            "game": "mario_1_2_consol2",
            "checkpoint":
                "checkpoints/_preserved/consol2_40pct_strict_iter01120.pt",
            "checkpoint_sha256":
                "413548b98a85d4d1ab4086556b957c2a17a28a579f86603d8b2"
                "d5af5375443ce",
            # The 1-2 entrance every campaign number is measured from.
            "start_state": "checkpoints/super_mario_bros_one_shot_tiles/"
                           "smb_curriculum/stage_03.state",
            "pairs_out": "runs/interference/success_1_2.npz",
            "reference_role": "prior_band_only",
            "reference_receipts": [
                {"receipt": "runs/consol2/peak_eval_seed7.json",
                 "checkpoint":
                     "checkpoints/_preserved/"
                     "consol2_40pct_strict_iter01120.pt",
                 "eval_seed": 7, "n_episodes": 50, "clears": 24,
                 "clear_rate": 0.48},
                {"receipt": "runs/consol2/peak_eval_seed101.json",
                 "checkpoint":
                     "checkpoints/_preserved/"
                     "consol2_40pct_strict_iter01120.pt",
                 "eval_seed": 101, "n_episodes": 50, "clears": 14,
                 "clear_rate": 0.28},
            ],
            "reference_protocol_deltas": [
                "eval_rng shared-stream (the gate runs per-episode)",
                "eval_workers 1 (the gate runs 5 workers)",
                "no --sequential/--level-clear (the gate passes both)",
            ],
            "rejected_candidates": [],
        },
    },

    # Step 1 — success-trajectory collection (per level).
    "collect_episodes": 300,
    "collect_max_steps": 3000,
    "collect_sticky": 0.25,
    "collect_jitter": 16,
    "collect_temperature": 1.0,
    "collect_seed": 20260815,
    "collect_workers": 6,
    "pooled_out": "runs/interference/pooled_pairs.npz",

    # Step 2 — joint BC into a FRESH h256/trunk64 net (~200k params).
    "bc_hidden_dim": 256,
    "bc_trunk_dim": 64,
    "bc_epochs": 50,
    "bc_lr": 3e-4,
    "bc_batch_size": 256,
    "bc_lr_step_epochs": 20,   # lr x bc_lr_gamma every 20 epochs
    "bc_lr_gamma": 0.5,
    "bc_seed": 20260815,
    "joint_out": "runs/interference/joint.pt",

    # Step 3 — the FOUR-leg strict honest gate (per level: the
    # specialist = positive control, then the joint net; one protocol).
    # 100, not 50: at n=50 the leg rule cannot prove a net holding only
    # a quarter of its control's competence has failed (exact power 0.64
    # on 1-1, 0.48 on 1-2 — see power_preregistration in the manifest).
    # 100 clears the registered floor on both legs. The dry-run recomputes
    # this and REFUSES a blind n.
    "gate_episodes": 100,
    "gate_sticky": 0.25,
    "gate_jitter": 16,
    "gate_workers": 5,
    "gate_max_steps": 3000,
    # MUST differ from collect_seed: eval_game's episode_rng_seeds is a
    # pure function of (eval_seed, episode_index), so a shared seed would
    # grade the joint net on exactly the noise realizations its own
    # training trajectories were collected under. Enforced by
    # assert_seed_independence().
    "gate_eval_seed": 20260816,
    "probe_timeout_s": 7200.0,

    # Verdict thresholds (pre-registered; strict predicate ONLY). The
    # hold threshold is a fraction of the MEASURED control leg — no
    # specialist rate is ever typed in.
    "severe_strict_ceiling": 0.10,      # both strictly below -> severe
    "hold_fraction_of_control": 0.50,   # threshold = frac x measured
    "control_alpha": 0.05,              # one-sided exact-binomial alpha
    "min_control_strict_rate": 0.15,    # below -> control collapsed
    "min_clone_hold_power": 0.80,       # dry-run refuses a blind n
    # The pre-registered alternative the gate must be able to CONVICT:
    # a joint net retaining only a quarter of its control's competence
    # (i.e. half the hold threshold). Detecting subtler interference than
    # this is out of reach at any n the attended window affords, and the
    # manifest says so rather than implying otherwise.
    "interfered_fraction_of_control": 0.25,
    "prior_drift_band": 0.20,           # |measured - prior| advisory
}


# ---------------------------------------------------------------------------
# Seed hygiene
# ---------------------------------------------------------------------------


def assert_seed_independence(cfg: Optional[dict] = None) -> None:
    """The gate must not replay the collection episodes.

    eval_game.episode_rng_seeds(eval_seed, episode_index) is a pure
    function of the pair, so a gate sharing collect_seed would score the
    joint net on byte-identical start-jitter offsets and sticky-repeat
    streams to the first `gate_episodes` collection episodes — the exact
    noise realizations its training trajectories came from. The house
    convention is a definitive eval on fresh seeds; a distinct seed costs
    nothing."""
    cfg = cfg or CONFIG
    if int(cfg["gate_eval_seed"]) == int(cfg["collect_seed"]):
        raise RuntimeError(
            f"gate_eval_seed {cfg['gate_eval_seed']} equals the collection "
            f"seed — the gate would replay the very episodes the training "
            f"trajectories were collected under; pick a fresh gate seed")


# ---------------------------------------------------------------------------
# Reference receipts — the specialist prior, re-derived FROM DISK
# ---------------------------------------------------------------------------


def _read_receipt_rows(path: Path) -> list[dict]:
    """Rows out of an eval.jsonl (one JSON object per line) or a single
    eval JSON file."""
    text = Path(path).read_text()
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def verify_reference_receipts(level: str,
                              cfg: Optional[dict] = None) -> dict:
    """Re-read every reference row this level cites and prove it exists.

    A reference rate is only a reference if a file on disk carries it
    NEXT TO the checkpoint's own name. Each registered row must be found
    in its receipt with matching checkpoint / eval_seed / n_episodes /
    clear_rate, and the checkpoint that row names must be byte-identical
    (sha256) to the specialist this falsifier actually rolls out —
    otherwise the number belongs to some other lineage and cannot speak
    for this policy. Returns the pooled clears/episodes plus the verified
    rows. Raises RuntimeError on any mismatch."""
    cfg = cfg or CONFIG
    spec = cfg["specialists"][level]
    pin = spec["checkpoint_sha256"]
    clears = episodes = 0
    rows: list[dict] = []
    for ref in spec["reference_receipts"]:
        path = REPO / ref["receipt"]
        if not path.exists():
            raise RuntimeError(
                f"{level} reference receipt missing: {ref['receipt']}")
        named = REPO / ref["checkpoint"]
        if not named.exists():
            raise RuntimeError(
                f"{level} reference names a checkpoint that is not on "
                f"disk: {ref['checkpoint']}")
        same_weights = _sha256(named) == pin
        if not same_weights:
            raise RuntimeError(
                f"{level} reference row cites {ref['checkpoint']}, whose "
                f"sha256 differs from the specialist this falsifier rolls "
                f"out ({spec['checkpoint']}) — a rate measured on other "
                f"weights is not this policy's reference")
        hits = [
            r for r in _read_receipt_rows(path)
            if str(r.get("checkpoint")) == ref["checkpoint"]
            and int(r.get("n_episodes") or 0) == int(ref["n_episodes"])
            and r.get("eval_seed") == ref["eval_seed"]
            and abs(float(r.get("clear_rate")) - float(ref["clear_rate"]))
            < 1e-9
        ]
        if not hits:
            raise RuntimeError(
                f"{level} reference row not found in {ref['receipt']}: "
                f"checkpoint={ref['checkpoint']} seed={ref['eval_seed']} "
                f"n={ref['n_episodes']} rate={ref['clear_rate']}")
        row = hits[0]
        k = int(round(float(row["clear_rate"]) * int(row["n_episodes"])))
        if k != int(ref["clears"]):
            raise RuntimeError(
                f"{level} reference row clears mismatch: receipt implies "
                f"{k}, CONFIG registers {ref['clears']}")
        clears += k
        episodes += int(row["n_episodes"])
        rows.append({**ref, "same_weights_as_specialist": same_weights,
                     "start_state": row.get("start_state"),
                     "sticky_prob": row.get("sticky_prob"),
                     "eval_rng": row.get("eval_rng"),
                     "eval_workers": row.get("eval_workers")})
    return {"level": level, "clears": clears, "episodes": episodes,
            "rate": clears / episodes if episodes else 0.0, "rows": rows}


def reference_prior(level: str, cfg: Optional[dict] = None) -> dict:
    """The pooled receipt-backed prior for a level.

    ADVISORY ONLY. It was measured under a different eval protocol
    (see reference_protocol_deltas), so it never becomes a threshold —
    it is the band the MEASURED control leg is sanity-checked against,
    and the number the dry-run pre-registers statistical power on."""
    cfg = cfg or CONFIG
    spec = cfg["specialists"][level]
    clears = sum(int(r["clears"]) for r in spec["reference_receipts"])
    episodes = sum(int(r["n_episodes"]) for r in spec["reference_receipts"])
    return {"level": level, "clears": clears, "episodes": episodes,
            "rate": clears / episodes if episodes else 0.0,
            "role": spec["reference_role"],
            "sources": sorted({r["receipt"]
                               for r in spec["reference_receipts"]}),
            "protocol_deltas": list(spec["reference_protocol_deltas"])}


# ---------------------------------------------------------------------------
# Exact binomial decision machinery — the answer to "is this leg's
# deficit real, or is 50 episodes just noisy?"
# ---------------------------------------------------------------------------


def _binom_pmf(k: int, n: int, p: float) -> float:
    return comb(int(n), int(k)) * (p ** int(k)) * ((1.0 - p) ** (n - int(k)))


def binom_p_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return sum(_binom_pmf(i, n, p) for i in range(max(0, int(k)), int(n) + 1))


def binom_p_at_most(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), exact."""
    return sum(_binom_pmf(i, n, p) for i in range(0, min(int(k), int(n)) + 1))


def leg_decision(clears: int, n: int, p0: float, alpha: float) -> dict:
    """One leg's three-way outcome against threshold p0.

    A point comparison ("rate >= p0") reads sampling noise as signal:
    at n=50 the two are routinely indistinguishable. So each leg gets an
    exact one-sided binomial test and THREE outcomes:

      holds         — P(X >= clears | n, p0) <= alpha: significantly
                      above the threshold;
      fails         — P(X <= clears | n, p0) <= alpha: significantly
                      below it;
      indeterminate — neither; n cannot separate this leg from its
                      threshold, and the run says so instead of guessing.
    """
    k, n = int(clears), int(n)
    p_above = binom_p_at_least(k, n, p0)
    p_below = binom_p_at_most(k, n, p0)
    if p_above <= alpha:
        decision = "holds"
    elif p_below <= alpha:
        decision = "fails"
    else:
        decision = "indeterminate"
    return {"decision": decision, "clears": k, "n": n, "p0": float(p0),
            "alpha": float(alpha), "p_above": p_above, "p_below": p_below}


def discrimination_power(n: int, control_rate: float, *, frac: float,
                         alpha: float,
                         interfered_rate: Optional[float] = None) -> dict:
    """Exact operating characteristics of the leg rule at this n.

    `clone_holds` is the probability a joint net that is a PERFECT clone
    of the specialist (true rate = control_rate) is classified `holds`;
    `interfered_fails` is the probability a genuinely-interfered net
    (true rate = interfered_fraction_of_control x control, unless
    overridden) is classified `fails`. Pre-registered in the manifest
    and enforced by the dry-run:
    an n that cannot clear both floors cannot produce a verdict worth
    reading, and finding that out must not cost the attended window."""
    n = int(n)
    p0 = float(frac) * float(control_rate)
    interfered = (float(interfered_rate) if interfered_rate is not None
                  else float(CONFIG["interfered_fraction_of_control"])
                  * float(control_rate))
    decisions = [leg_decision(k, n, p0, alpha)["decision"]
                 for k in range(n + 1)]
    out = {"n": n, "control_rate": float(control_rate), "p0": p0,
           "alpha": float(alpha), "interfered_rate": interfered}
    for name, rate in (("clone", control_rate), ("interfered", interfered)):
        for outcome in ("holds", "fails", "indeterminate"):
            out[f"{name}_{outcome}"] = sum(
                _binom_pmf(k, n, rate) for k in range(n + 1)
                if decisions[k] == outcome)
    return out


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested on synthetic arrays/summaries —
# tests/test_interference_falsifier.py)
# ---------------------------------------------------------------------------


def success_pairs_from_episodes(
    episodes: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """STRICT-success filter + concat: (obs, act, traj_len, kept).

    Keeps episodes with episode_success ONLY — max_gx never substitutes
    (the falsifier distills finished levels, not deep reaches). Empty
    (0, 0)-shaped arrays when nothing cleared, so the caller can fail
    loud with the counts in hand."""
    kept = [e for e in episodes if bool(e["episode_success"])]
    if not kept:
        return (np.zeros((0, 0), dtype=np.int8),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64), [])
    obs = np.concatenate([e["obs"] for e in kept], axis=0)
    act = np.concatenate([e["actions"] for e in kept], axis=0)
    traj_len = np.array([e["length"] for e in kept], dtype=np.int64)
    return obs, act.astype(np.int64), traj_len, kept


def balance_fifty_fifty(
    by_level: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """The pooled set, balanced 50/50 BY STATE-ACTION PAIRS.

    Both sides are trimmed to the smaller side's pair count, keeping
    the LEADING pairs in collection order (deterministic; the trailing
    trim may cut mid-episode, which is harmless to a per-pair BC loss).
    Pair-level rather than episode-level balance because 1-2
    trajectories are systematically longer — episode balance would let
    1-2 dominate the gradient. level_ids index into CONFIG["levels"]
    order. Raises RuntimeError on an empty side: a one-level pool
    cannot falsify interference."""
    levels = list(CONFIG["levels"])
    counts = {lvl: int(by_level[lvl][1].shape[0]) for lvl in levels}
    empty = [lvl for lvl, n in counts.items() if n == 0]
    if empty:
        raise RuntimeError(
            f"no strict-success pairs for level(s) {empty} — a one-level "
            f"pool cannot falsify interference (counts: {counts})")
    target = min(counts.values())
    obs_parts, act_parts, id_parts = [], [], []
    for idx, lvl in enumerate(levels):
        o, a = by_level[lvl]
        obs_parts.append(np.asarray(o)[:target])
        act_parts.append(np.asarray(a, dtype=np.int64)[:target])
        id_parts.append(np.full(target, idx, dtype=np.int64))
    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "act": np.concatenate(act_parts, axis=0),
        "level_ids": np.concatenate(id_parts, axis=0),
        "levels": levels,
        "counts": {lvl: target for lvl in levels},
        "counts_before_balance": counts,
    }


def write_success_pairs(path, *, obs: np.ndarray, act: np.ndarray,
                        traj_len: np.ndarray, label_max_gx: np.ndarray,
                        label_episode_success: np.ndarray,
                        provenance: dict) -> None:
    """Per-level npz in the house demo convention (n=1 + obs_0/act_0,
    consumable by the distill scripts unchanged) plus per-trajectory
    labels and a provenance JSON blob."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        n=1,
        obs_0=np.asarray(obs),
        act_0=np.asarray(act, dtype=np.int64),
        traj_len=np.asarray(traj_len, dtype=np.int64),
        label_max_gx=np.asarray(label_max_gx, dtype=np.int64),
        label_episode_success=np.asarray(label_episode_success, dtype=bool),
        provenance=json.dumps(provenance),
    )


def write_pooled_pairs(path, pooled: dict, *, provenance: dict) -> None:
    """The balanced pooled set. level_ids ride along (indexed into the
    `levels` order) so the Level-ID-token follow-up — if the verdict
    calls for it — can consume this exact file without re-collection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=np.asarray(pooled["obs"]),
        act=np.asarray(pooled["act"], dtype=np.int64),
        level_ids=np.asarray(pooled["level_ids"], dtype=np.int64),
        levels=json.dumps(list(pooled["levels"])),
        counts=json.dumps(pooled["counts"]),
        provenance=json.dumps(provenance),
    )


def joint_bc(
    obs: np.ndarray,
    act: np.ndarray,
    *,
    num_actions: int,
    feature_dim: int,
    hidden_dim: int,
    trunk_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    lr_step_epochs: int,
    lr_gamma: float,
    seed: int,
) -> tuple[dict, dict]:
    """Joint BC into a FRESH TilePolicyNetwork — no warm start from
    either specialist, so the verdict measures what the pooled DATA
    supports, not what a lucky initialization retains. Cross-entropy,
    step-decayed lr (x lr_gamma every lr_step_epochs), deterministic
    from `seed` (net init AND shuffle). Returns (cpu state_dict,
    stats)."""
    from src.models.tile_policy import TilePolicyNetwork

    n = int(np.asarray(act).shape[0])
    if n == 0:
        raise ValueError("joint_bc on zero pairs")
    torch.manual_seed(int(seed))
    net = TilePolicyNetwork(num_actions=int(num_actions),
                            feature_dim=int(feature_dim),
                            hidden_dim=int(hidden_dim),
                            trunk_dim=int(trunk_dim))

    X = torch.from_numpy(np.asarray(obs)).float()
    Y = torch.from_numpy(np.asarray(act)).long()
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(lr_step_epochs), gamma=float(lr_gamma))
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    net.eval()
    with torch.no_grad():
        logits, _ = net.forward_ac(X)
        initial_loss = float(F.cross_entropy(logits, Y))

    net.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            logits, _ = net.forward_ac(X[idx])
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()  # per-EPOCH step decay

    net.eval()
    with torch.no_grad():
        logits, _ = net.forward_ac(X)
        final_loss = float(F.cross_entropy(logits, Y))
        final_acc = float((logits.argmax(-1) == Y).float().mean())

    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    stats = {
        "n_pairs": n,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "lr_schedule": "step_decay",
        "lr_step_epochs": int(lr_step_epochs),
        "lr_gamma": float(lr_gamma),
        "seed": int(seed),
        "hidden_dim": int(hidden_dim),
        "trunk_dim": int(trunk_dim),
        "param_count": int(net.num_params),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_acc": final_acc,
    }
    return sd, stats


def write_joint_checkpoint(path, sd: dict, *, stats: dict,
                           provenance: dict) -> None:
    """runs/interference/joint.pt in the standard payload format
    (net_state_dict + iter) — eval_game's loader + shape inference
    consume it unchanged. iter 0: this net is nobody's continuation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iter": 0,
        "net_state_dict": {k: torch.as_tensor(v).detach().cpu()
                           for k, v in sd.items()},
        "provenance": "interference_joint_bc",
        "interference": {"stats": dict(stats), **dict(provenance)},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def _leg_rate(summ: dict, label: str,
              reasons: list[str]) -> Optional[float]:
    """One probe summary -> its strict rate, or None with a recorded
    reason. Classification runs on clear_rate_strict ONLY (2026-08-15
    forensics: the chain predicate disagreed 13:1 with strict on real
    data; a verdict on the wrong predicate would calibrate the
    generalist plan off a mirage)."""
    if (summ.get("status") != "ok"
            or int(summ.get("n_episodes", 0) or 0) <= 0):
        reasons.append(
            f"{label} leg unusable (status={summ.get('status')!r}, "
            f"n_episodes={summ.get('n_episodes')}) — a broken probe is "
            f"never a verdict")
        return None
    if summ.get("clear_rate_strict") is None:
        reasons.append(
            f"{label} leg carries no strict predicate — the verdict is "
            f"registered on clear_rate_strict only")
        return None
    return float(summ["clear_rate_strict"])


def interference_verdict(joint_summs: dict[str, dict],
                         control_summs: dict[str, dict],
                         cfg: Optional[dict] = None) -> dict:
    """THE pre-registered verdict over the four strict honest legs.

    Every level contributes TWO measurements taken under ONE protocol in
    ONE session: the specialist (positive control) and the joint net.
    The hold threshold is `hold_fraction_of_control` x the MEASURED
    control — never a rate transcribed from prose — so a protocol shift
    moves both legs together and cannot masquerade as interference.
    Each leg is then decided by an exact one-sided binomial test at
    `control_alpha`, so a deficit smaller than the sampling noise at
    `gate_episodes` returns `indeterminate` rather than a coin flip.

      unusable             — any leg failed/empty/strict-less, OR a
                             positive control below
                             min_control_strict_rate (the specialist
                             cannot clear under this harness, so a low
                             joint rate is not evidence about the joint
                             net);
      inconclusive         — every leg usable, but at least one is
                             statistically indeterminate at this n;
      compatible           — both levels hold;
      severe_interference  — both legs fail AND both joint rates are
                             strictly below severe_strict_ceiling;
      partial_interference — exactly one level holds, the other fails;
      degraded             — both fail, but not both under the ceiling.
    Precedence: unusable > inconclusive > severe > compatible >
    partial > degraded.

    The receipted prior (reference_prior) never enters the decision; it
    is compared against the measured control and any gap beyond
    prior_drift_band is reported as an advisory, because the prior was
    measured under a different eval protocol."""
    cfg = cfg or CONFIG
    levels = list(cfg["levels"])
    reasons: list[str] = []
    advisories: list[str] = []

    joint_rates = {lvl: _leg_rate(joint_summs[lvl], f"{lvl} joint", reasons)
                   for lvl in levels}
    control_rates = {
        lvl: _leg_rate(control_summs[lvl], f"{lvl} control", reasons)
        for lvl in levels}

    frac = float(cfg["hold_fraction_of_control"])
    alpha = float(cfg["control_alpha"])
    ceiling = float(cfg["severe_strict_ceiling"])
    min_control = float(cfg["min_control_strict_rate"])
    band = float(cfg["prior_drift_band"])

    # The positive control has to actually clear, or nothing downstream
    # of it means anything.
    for lvl in levels:
        rate = control_rates[lvl]
        if rate is not None and rate < min_control:
            reasons.append(
                f"{lvl} positive control collapsed ({rate:.3f} < "
                f"{min_control}) — the specialist itself does not clear "
                f"under this gate protocol, so a joint deficit cannot be "
                f"separated from a harness shift")

    # Prior drift: advisory only (different protocol by construction).
    prior_check: dict[str, dict] = {}
    for lvl in levels:
        prior = reference_prior(lvl, cfg)
        measured = control_rates[lvl]
        within = (measured is not None
                  and abs(measured - prior["rate"]) <= band)
        prior_check[lvl] = {
            "prior_rate": prior["rate"], "prior_episodes": prior["episodes"],
            "measured_control": measured, "band": band,
            "within_band": bool(within), "sources": prior["sources"],
            "protocol_deltas": prior["protocol_deltas"]}
        if measured is not None and not within:
            advisories.append(
                f"{lvl} control drift: measured {measured:.3f} vs receipted "
                f"prior {prior['rate']:.3f} (band {band}) — expected to "
                f"differ somewhat (prior protocol: "
                f"{'; '.join(prior['protocol_deltas'])}), but a gap this "
                f"size is worth reading before the verdict")

    thresholds: dict[str, Optional[float]] = {}
    decisions: dict[str, Optional[dict]] = {}
    holds: dict[str, Optional[bool]] = {}
    for lvl in levels:
        jr, cr = joint_rates[lvl], control_rates[lvl]
        if jr is None or cr is None:
            thresholds[lvl] = decisions[lvl] = holds[lvl] = None
            continue
        n = int(joint_summs[lvl]["n_episodes"])
        p0 = frac * cr
        d = leg_decision(int(round(jr * n)), n, p0, alpha)
        thresholds[lvl] = p0
        decisions[lvl] = d
        holds[lvl] = (d["decision"] == "holds")

    if reasons:
        verdict = "unusable"
        holds = {lvl: None for lvl in levels}
    else:
        outcomes = [decisions[lvl]["decision"] for lvl in levels]
        if "indeterminate" in outcomes:
            verdict = "inconclusive"
            for lvl in levels:
                if decisions[lvl]["decision"] == "indeterminate":
                    reasons.append(
                        f"{lvl} leg indeterminate at n="
                        f"{decisions[lvl]['n']}: "
                        f"{decisions[lvl]['clears']} clears against a "
                        f"threshold of {decisions[lvl]['p0']:.3f} separates "
                        f"from neither side at alpha={alpha} "
                        f"(p_above={decisions[lvl]['p_above']:.3f}, "
                        f"p_below={decisions[lvl]['p_below']:.3f}) — more "
                        f"episodes, not a guess")
        elif all(o == "fails" for o in outcomes) and all(
                joint_rates[lvl] < ceiling for lvl in levels):
            verdict = "severe_interference"
        elif all(o == "holds" for o in outcomes):
            verdict = "compatible"
        elif sum(1 for o in outcomes if o == "holds") == 1:
            verdict = "partial_interference"
        else:
            verdict = "degraded"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "advisories": advisories,
        "joint_strict_rates": joint_rates,
        "control_strict_rates": control_rates,
        "hold_thresholds": thresholds,
        "leg_decisions": decisions,
        "holds": holds,
        "prior_check": prior_check,
        "hold_fraction_of_control": frac,
        "control_alpha": alpha,
        "severe_strict_ceiling": ceiling,
        "min_control_strict_rate": min_control,
    }


def steps_from(skip_to: int) -> list[int]:
    """--skip-to N resumability: the ordered steps still to run."""
    if not 1 <= int(skip_to) <= 3:
        raise ValueError(f"--skip-to must be 1..3, got {skip_to!r}")
    return list(range(int(skip_to), 4))


# ---- eval command assembly ------------------------------------------------


def _eval_game_command(level: str, *, checkpoint, episodes: int,
                       sticky: float, jitter: int, workers: int,
                       eval_seed: int, action_select: str = "greedy",
                       temperature: Optional[float] = None) -> list[str]:
    """One eval_game argv under the LEVEL'S OWN profile (the profile the
    specialist reference was measured on). Greedy is eval_game's
    default; only a sampled run opts in explicitly."""
    spec = CONFIG["specialists"][level]
    base = _load_yaml(REPO / spec["profile"])
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_game.py"),
        "--game", spec["game"],
        "--profile", str(REPO / spec["profile"]),
        "--rom", str(REPO / base["rom_path"]),
        "--checkpoint", str(checkpoint),
        "--episodes", str(int(episodes)),
        "--max-steps", str(int(CONFIG["gate_max_steps"])),
        "--sequential", "--level-clear",
        "--start-state", str(REPO / spec["start_state"]),
        "--eval-seed", str(int(eval_seed)),
        "--sticky-prob", str(float(sticky)),
        "--start-jitter", str(int(jitter)),
        "--eval-workers", str(int(workers)),
        "--eval-rng", "per-episode",
    ]
    if action_select != "greedy":
        cmd += ["--action-select", str(action_select),
                "--temperature", str(float(temperature))]
    return cmd


def build_gate_command(level: str, *, checkpoint,
                       require_paths: bool = True) -> list[str]:
    """One gate leg: a gate_episodes COLD ARGMAX strict honest eval — sticky
    0.25, jitter 16, per-episode RNG, from the level's own entrance, on
    a seed disjoint from collection.

    The SAME builder serves both roles: the positive-control leg
    (checkpoint = the level's specialist) and the joint leg (checkpoint
    = joint.pt) differ in nothing but --checkpoint, which is what makes
    the control a control. `require_paths=False` lets tests assemble
    against the not-yet-built joint.pt."""
    assert_seed_independence()
    cmd = _eval_game_command(
        level, checkpoint=checkpoint,
        episodes=CONFIG["gate_episodes"],
        sticky=CONFIG["gate_sticky"], jitter=CONFIG["gate_jitter"],
        workers=CONFIG["gate_workers"],
        eval_seed=CONFIG["gate_eval_seed"])
    if require_paths:
        missing = [a for a in cmd if "/" in a and not a.startswith("-")
                   and not Path(a).exists()]
        if missing:
            raise RuntimeError(f"gate command paths missing: {missing}")
    return cmd


def build_collection_reference_command(level: str, *,
                                       checkpoint) -> list[str]:
    """The eval_game argv step-1 collection is protocol-equivalent to
    (sampled T=1.0, house noise, per-episode RNG). eval_game has no
    trace-saving flag (checked for night2), so this command is the
    RECEIPT of the protocol — the traces come from the imported
    collection loop."""
    return _eval_game_command(
        level, checkpoint=checkpoint,
        episodes=CONFIG["collect_episodes"],
        sticky=CONFIG["collect_sticky"], jitter=CONFIG["collect_jitter"],
        workers=CONFIG["collect_workers"],
        eval_seed=CONFIG["collect_seed"],
        action_select="sampled",
        temperature=CONFIG["collect_temperature"])


def collection_plan(level: str) -> dict:
    """The step-1 plan for one level, in one receipt-able dict."""
    spec = CONFIG["specialists"][level]
    return {
        "level": level,
        "mode": "collection",
        "collector": ("night2_runner.collect_trajectories (imported) — "
                      "eval_game's episode_rng_seeds/select_action "
                      "mechanics, lane-count invariant"),
        "profile": str(REPO / spec["profile"]),
        "checkpoint": str(REPO / spec["checkpoint"]),
        "start_state": str(REPO / spec["start_state"]),
        "episodes": CONFIG["collect_episodes"],
        "max_steps": CONFIG["collect_max_steps"],
        "sticky_prob": CONFIG["collect_sticky"],
        "start_jitter": CONFIG["collect_jitter"],
        "action_select": "sampled",
        "temperature": CONFIG["collect_temperature"],
        "eval_rng": "per-episode",
        "eval_seed": CONFIG["collect_seed"],
        "lanes": CONFIG["collect_workers"],
        "keep": "strict_episode_success_only",
        "out": str(REPO / spec["pairs_out"]),
        "reference_command": build_collection_reference_command(
            level, checkpoint=REPO / spec["checkpoint"]),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest() -> dict:
    """CONFIG + the verdict pre-registration, verbatim, for
    runs/interference/manifest.json."""
    frac = CONFIG["hold_fraction_of_control"]
    alpha = CONFIG["control_alpha"]
    priors = {lvl: reference_prior(lvl) for lvl in CONFIG["levels"]}
    return {
        "campaign": "smb_generalist_interference_falsifier",
        "config": copy.deepcopy(CONFIG),
        "gate_legs": 2 * len(CONFIG["levels"]),
        "reference_priors": priors,
        "power_preregistration": {
            lvl: discrimination_power(
                CONFIG["gate_episodes"], priors[lvl]["rate"],
                frac=frac, alpha=alpha)
            for lvl in CONFIG["levels"]},
        "verdict_rules": {
            "predicate": "clear_rate_strict ONLY (chain never substitutes)",
            "reference": (
                "the hold threshold is "
                f"{frac} x the MEASURED positive-control leg — each "
                "level's own specialist run through the identical gate "
                "command in the same session. No specialist rate is "
                "hardcoded. The receipted priors below are advisory only "
                "(they were measured under a different eval protocol) and "
                "enter solely as a drift band on the control."),
            "leg_rule": (
                "exact one-sided binomial at alpha="
                f"{alpha}: holds if P(X >= clears | n, threshold) <= "
                "alpha, fails if P(X <= clears | n, threshold) <= alpha, "
                "otherwise indeterminate"),
            "severe_interference":
                "BOTH legs fail AND both joint strict rates < "
                f"{CONFIG['severe_strict_ceiling']}",
            "compatible": "both levels hold",
            "partial_interference": "exactly one level holds",
            "degraded": "both fail, but not both under the ceiling",
            "inconclusive": (
                "at least one leg is statistically indeterminate at "
                f"n={CONFIG['gate_episodes']} — reported, never rounded "
                "into a verdict"),
            "unusable": (
                "any failed/empty/strict-less probe leg, or a positive "
                f"control below {CONFIG['min_control_strict_rate']} — a "
                "broken harness is never a verdict"),
        },
        "scope_note": (
            "The verdict CALIBRATES the generalist plan — compatible "
            "means plain pooled distillation scales; partial/severe "
            "means the Level-ID token and/or a capacity bump is "
            "required before multi-level distillation. It does not "
            "gate the 1-3 campaign, which runs independently off "
            "configs/mario_1_3_backward.yaml and "
            "checkpoints/backward_states/1-3/."),
        "steps": [
            {"step": 1, "name": "success_trajectory_collection"},
            {"step": 2, "name": "joint_bc"},
            {"step": 3, "name": "bilateral_gate"},
        ],
        "created_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Runtime pieces (subprocess / filesystem / emulator; exercised by --dry-run
# and the real run, not by the unit tests)
# ---------------------------------------------------------------------------


def _load_payload(path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _verify_checkpoint(level: str) -> Path:
    spec = CONFIG["specialists"][level]
    src = REPO / spec["checkpoint"]
    got = _sha256(src)
    if got != spec["checkpoint_sha256"]:
        raise RuntimeError(
            f"{level} specialist sha256 mismatch: {src} is {got}, pinned "
            f"{spec['checkpoint_sha256']} — refusing to distill an "
            f"unverified policy")
    return src


def _receipt(row: dict) -> None:
    _append_jsonl(REPO / CONFIG["receipt_log"], row)


def _run_eval(cmd: list[str]) -> dict:
    """Run one eval_game subprocess; return its JSON or an error dict
    (same contract as run_consol2.run_probe)."""
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=CONFIG["probe_timeout_s"])
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout"}
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
                return {"status": "probe_failed", "detail": f"bad JSON: {e}"}
    if proc.returncode != 0:
        return {"status": "probe_failed",
                "detail": proc.stderr.strip()[-500:]}
    return {"status": "probe_failed", "detail": "no JSON on stdout"}


def _collect_with_real_pool(plan: dict) -> list[dict]:
    """Assemble the level's pool + policy + reward around the imported
    collection loop. THE ONLY rollout surface in this module — never
    reached by --dry-run or the unit tests."""
    from nes_core import Pool
    from scripts.eval_game import _TilePolicy
    from src.emulation.frame_utils import TileFeatureStacker
    from src.models.tile_policy import build_tile_policy_from_checkpoint
    from src.training.profile_utils import (
        action_space_to_bitmasks, resolve_encoder,
    )
    from src.utils.reward_functions import build_reward_function

    profile = _load_yaml(plan["profile"])
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    payload = _load_payload(plan["checkpoint"])
    net, is_recurrent = build_tile_policy_from_checkpoint(
        payload, num_actions=len(bitmasks), feature_dim=stacked_dim)
    missing, _ = net.load_state_dict(payload["net_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing policy weights: {missing}")
    net.eval()

    lanes = max(1, min(int(plan["lanes"]), int(plan["episodes"]),
                       int(os.cpu_count() or 1)))
    pool = Pool(rom_path=str(REPO / profile["rom_path"]),
                num_workers=lanes,
                frame_skip=int(profile.get("frame_skip", 4)))
    pool.set_headless(True)
    pool.set_skip_preprocess(True)  # tile obs read RAM, never pixels
    start_blob = Path(plan["start_state"]).read_bytes()
    try:
        return collect_trajectories(
            pool,
            n_episodes=int(plan["episodes"]),
            lanes=lanes,
            max_steps=int(plan["max_steps"]),
            bitmasks=bitmasks,
            make_policy=lambda: _TilePolicy(
                net,
                TileFeatureStacker(stack_size=stack_size,
                                   feature_dim=feature_dim),
                extractor, is_recurrent),
            make_reward_fn=lambda: build_reward_function(profile),
            start_blob=start_blob,
            sticky_prob=float(plan["sticky_prob"]),
            start_jitter=int(plan["start_jitter"]),
            eval_seed=int(plan["eval_seed"]),
            action_select=str(plan["action_select"]),
            temperature=float(plan["temperature"]),
        )
    finally:
        pool.shutdown()


# ---------------------------------------------------------------------------
# The three steps
# ---------------------------------------------------------------------------


def step1_collect() -> Path:
    """Success-trajectory collection from BOTH specialists, each on its
    own level; strict-success filter; per-level npz; 50/50 pooled set."""
    by_level: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for level in CONFIG["levels"]:
        _verify_checkpoint(level)
        plan = collection_plan(level)
        print(f"[interference] step 1: collecting {plan['episodes']} "
              f"sampled episodes on {level} "
              f"({Path(plan['checkpoint']).name})", flush=True)
        episodes = _collect_with_real_pool(plan)
        obs, act, traj_len, kept = success_pairs_from_episodes(episodes)
        spec = CONFIG["specialists"][level]
        _receipt({"type": "step", "step": 1, "level": level,
                  "name": "success_trajectory_collection",
                  "episodes_run": len(episodes), "kept_strict": len(kept),
                  "pairs": int(act.shape[0]),
                  "out": str(REPO / spec["pairs_out"])})
        if not kept:
            raise RuntimeError(
                f"collection kept 0 strict successes of {len(episodes)} "
                f"episodes on {level} — the {level} specialist produced "
                f"no finished trajectories under the sampled protocol; "
                f"the falsifier cannot run (receipt written)")
        write_success_pairs(
            REPO / spec["pairs_out"], obs=obs, act=act, traj_len=traj_len,
            label_max_gx=np.array([e["max_gx"] for e in kept],
                                  dtype=np.int64),
            label_episode_success=np.ones(len(kept), dtype=bool),
            provenance={k: v for k, v in plan.items()
                        if k != "reference_command"})
        by_level[level] = (obs, act)
        print(f"[interference] step 1: {level} kept {len(kept)}/"
              f"{len(episodes)} strict clears ({int(act.shape[0])} pairs)",
              flush=True)

    pooled = balance_fifty_fifty(by_level)
    out = REPO / CONFIG["pooled_out"]
    write_pooled_pairs(out, pooled, provenance={
        "step": 1, "balance": "50/50 by state-action pairs",
        "counts_before_balance": pooled["counts_before_balance"]})
    _receipt({"type": "step", "step": 1, "name": "pooled_balance",
              "counts_before_balance": pooled["counts_before_balance"],
              "counts": pooled["counts"], "out": str(out)})
    print(f"[interference] step 1: pooled 50/50 "
          f"{pooled['counts']} (from {pooled['counts_before_balance']}) "
          f"-> {out}", flush=True)
    return out


def step2_joint_bc() -> Path:
    pooled_path = REPO / CONFIG["pooled_out"]
    if not pooled_path.exists():
        raise RuntimeError(
            f"{pooled_path} missing — run step 1 first (or drop --skip-to)")
    d = np.load(pooled_path)
    obs, act = d["obs"], d["act"]

    # num_actions from the profiles' (identical, dry-run-asserted)
    # action space; feature_dim from the pooled obs themselves.
    base = _load_yaml(REPO / CONFIG["specialists"]["1-2"]["profile"])
    sd, stats = joint_bc(
        obs, act,
        num_actions=len(base["action_space"]),
        feature_dim=int(obs.shape[1]),
        hidden_dim=CONFIG["bc_hidden_dim"],
        trunk_dim=CONFIG["bc_trunk_dim"],
        epochs=CONFIG["bc_epochs"], lr=CONFIG["bc_lr"],
        batch_size=CONFIG["bc_batch_size"],
        lr_step_epochs=CONFIG["bc_lr_step_epochs"],
        lr_gamma=CONFIG["bc_lr_gamma"], seed=CONFIG["bc_seed"])
    out = REPO / CONFIG["joint_out"]
    write_joint_checkpoint(
        out, sd, stats=stats,
        provenance={"pooled": str(pooled_path),
                    "levels": list(CONFIG["levels"]),
                    "specialists": {
                        lvl: CONFIG["specialists"][lvl]["checkpoint"]
                        for lvl in CONFIG["levels"]}})
    _receipt({"type": "step", "step": 2, "name": "joint_bc",
              "stats": stats, "out": str(out)})
    print(f"[interference] step 2: joint BC "
          f"(loss {stats['initial_loss']:.4f} -> {stats['final_loss']:.4f}, "
          f"acc {stats['final_acc']:.3f}, {stats['param_count']} params) "
          f"-> {out}", flush=True)
    return out


def step3_gate(run_eval=None) -> dict:
    """The FOUR-leg gate: per level, the specialist (positive control)
    and then the joint net, under one identical protocol.

    Running the specialists through the very same command is what
    separates interference from a harness/protocol shift: if the
    specialist's own rate moves, both legs move with it and the
    threshold moves too. Two of the four legs buy that; without them a
    low joint rate is uninterpretable. `run_eval` is injectable so the
    wiring is testable without an emulator."""
    run_eval = run_eval or _run_eval
    joint = REPO / CONFIG["joint_out"]
    if not joint.exists():
        raise RuntimeError(
            f"{joint} missing — run step 2 first (or drop --skip-to)")
    assert_seed_independence()

    joint_summs: dict[str, dict] = {}
    control_summs: dict[str, dict] = {}
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        for role, ckpt, sink in (
                ("control", REPO / spec["checkpoint"], control_summs),
                ("joint", joint, joint_summs)):
            cmd = build_gate_command(level, checkpoint=ckpt)
            res = run_eval(cmd)
            # survival_x is 1-2's bottleneck mark; carried on both rows
            # for comparability, meaningful only on the 1-2 leg.
            summ = probe_summary(res, 2600)
            summ["status"] = res.get("status")
            sink[level] = summ
            _receipt({"type": "probe", "level": level, "role": role,
                      "checkpoint": str(ckpt), "cmd": " ".join(cmd),
                      **{k: v for k, v in summ.items() if k != "status"},
                      "status": summ["status"]})
            print(f"[interference] step 3: {level} {role} leg "
                  f"status={summ['status']} "
                  f"strict={summ.get('clear_rate_strict')} "
                  f"median_max_x={summ.get('median_max_x')}", flush=True)

    verdict = interference_verdict(joint_summs, control_summs)
    _receipt({"type": "verdict", **verdict,
              "note": ("calibrates the generalist plan (Level-ID token / "
                       "capacity); does not gate the 1-3 campaign")})
    print(f"[interference] step 3: VERDICT = {verdict['verdict']} "
          f"(joint {verdict['joint_strict_rates']}, "
          f"control {verdict['control_strict_rates']}, "
          f"thresholds {verdict['hold_thresholds']}, "
          f"holds {verdict['holds']})", flush=True)
    for line in verdict["reasons"] + verdict["advisories"]:
        print(f"[interference]   - {line}", flush=True)
    return verdict


# ---------------------------------------------------------------------------
# Dry run — config parse + checkpoint loads + command assembly + verdict
# table. NOT rollouts.
# ---------------------------------------------------------------------------


def dry_run() -> int:
    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" — {detail}" if detail else ""), flush=True)

    from src.training.config_schema import check_profile
    from src.training.profile_utils import resolve_encoder

    # 1. Both profiles parse + strict schema.
    profiles: dict[str, dict] = {}
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        try:
            base = _load_yaml(REPO / spec["profile"])
            warns = check_profile(base, strict=False)
            profiles[level] = base
            report(f"{level} profile schema (strict)", not warns,
                   "; ".join(warns) if warns else spec["profile"])
        except Exception as e:
            report(f"{level} profile schema (strict)", False, repr(e))
            return 1

    # 2. The profiles agree on the joint net's interface: identical
    #    action space, same ROM, same encoder width — pooled BC pairs
    #    from both levels feed ONE net, so any disagreement poisons it.
    dims: dict[str, int] = {}
    try:
        acts_equal = (profiles["1-1"]["action_space"]
                      == profiles["1-2"]["action_space"])
        roms_equal = (profiles["1-1"]["rom_path"]
                      == profiles["1-2"]["rom_path"])
        for level, base in profiles.items():
            _, _, stacked_dim = resolve_encoder(base)
            dims[level] = stacked_dim
        report("profiles agree (action space, rom, encoder width)",
               acts_equal and roms_equal and dims["1-1"] == dims["1-2"],
               f"actions_equal={acts_equal} roms_equal={roms_equal} "
               f"obs={dims}")
    except Exception as e:
        report("profiles agree (action space, rom, encoder width)", False,
               repr(e))
        return 1

    # 3. Both start states exist and match their profile's own
    #    start_state_path (the blob the specialist reference was
    #    measured from).
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        p = REPO / spec["start_state"]
        matches = (spec["start_state"]
                   == profiles[level].get("start_state_path"))
        report(f"{level} start state exists + matches its profile",
               p.exists() and matches,
               f"{spec['start_state']} (profile match={matches})")

    # 4 + 5. Both specialist checkpoints sha256-match their pins and
    #    load with shape inference at the expected capacities.
    expected_dims = {"1-1": (64, 32), "1-2": (256, 64)}
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        try:
            src = _verify_checkpoint(level)
            report(f"{level} specialist sha256 matches pin", True,
                   spec["checkpoint_sha256"][:16] + "...")
        except Exception as e:
            report(f"{level} specialist sha256 matches pin", False, repr(e))
            continue
        try:
            from src.models.tile_policy import (
                build_tile_policy_from_checkpoint,
            )
            payload = _load_payload(src)
            net, is_recurrent = build_tile_policy_from_checkpoint(
                payload, num_actions=len(profiles[level]["action_space"]),
                feature_dim=dims[level])
            missing, _ = net.load_state_dict(payload["net_state_dict"],
                                             strict=False)
            h, t = expected_dims[level]
            good = (not is_recurrent and not missing
                    and net.hidden_dim == h and net.trunk_dim == t
                    and net.feature_dim == dims[level])
            report(f"{level} specialist loads with shape inference", good,
                   f"h{net.hidden_dim}/trunk{net.trunk_dim} "
                   f"obs={net.feature_dim} recurrent={is_recurrent} "
                   f"missing={missing}")
        except Exception as e:
            report(f"{level} specialist loads with shape inference", False,
                   repr(e))

    # 6. The fresh joint net constructs at the registered ~200k params.
    try:
        from src.models.tile_policy import TilePolicyNetwork
        net = TilePolicyNetwork(
            num_actions=len(profiles["1-2"]["action_space"]),
            feature_dim=dims["1-2"],
            hidden_dim=CONFIG["bc_hidden_dim"],
            trunk_dim=CONFIG["bc_trunk_dim"])
        report("fresh joint net constructs (~200k params)",
               190_000 <= net.num_params <= 210_000,
               f"h{CONFIG['bc_hidden_dim']}/trunk{CONFIG['bc_trunk_dim']} "
               f"= {net.num_params} params")
    except Exception as e:
        report("fresh joint net constructs (~200k params)", False, repr(e))

    # 7 + 8. All four eval commands assemble against real paths (the
    #    specialist stands in for joint.pt, which exists after step 2).
    for level in CONFIG["levels"]:
        spec = CONFIG["specialists"][level]
        for label, builder in (
                (f"{level} collection reference command assembles",
                 lambda lvl=level, s=spec: build_collection_reference_command(
                     lvl, checkpoint=REPO / s["checkpoint"])),
                (f"{level} gate command assembles",
                 lambda lvl=level, s=spec: build_gate_command(
                     lvl, checkpoint=REPO / s["checkpoint"]))):
            try:
                cmd = builder()
                missing_paths = [
                    a for a in cmd if "/" in a and not a.startswith("-")
                    and not Path(a).exists()]
                report(label, not missing_paths,
                       " ".join(cmd) if not missing_paths
                       else f"missing paths: {missing_paths}")
            except Exception as e:
                report(label, False, repr(e))

    # 9. Every reference rate re-verifies against its own receipt file,
    #    on weights byte-identical to the specialist we roll out.
    for level in CONFIG["levels"]:
        try:
            got = verify_reference_receipts(level)
            prior = reference_prior(level)
            report(f"{level} reference receipts re-verify on disk",
                   got["clears"] == prior["clears"]
                   and got["episodes"] == prior["episodes"]
                   and got["episodes"] >= 100,
                   f"{got['clears']}/{got['episodes']} = "
                   f"{prior['rate']:.4f} (prior band only) from "
                   f"{', '.join(prior['sources'])}")
        except Exception as e:
            report(f"{level} reference receipts re-verify on disk", False,
                   repr(e))

    # 10. The gate seed cannot replay the collection episodes.
    try:
        assert_seed_independence()
        report("gate seed is disjoint from the collection seed", True,
               f"collect={CONFIG['collect_seed']} "
               f"gate={CONFIG['gate_eval_seed']}")
    except Exception as e:
        report("gate seed is disjoint from the collection seed", False,
               repr(e))

    # 11. The registered episode count can actually discriminate. If it
    #     cannot, the operator learns it HERE, not after the window.
    for level in CONFIG["levels"]:
        try:
            prior = reference_prior(level)
            pw = discrimination_power(
                CONFIG["gate_episodes"], prior["rate"],
                frac=CONFIG["hold_fraction_of_control"],
                alpha=CONFIG["control_alpha"])
            floor = CONFIG["min_clone_hold_power"]
            report(f"{level} n={CONFIG['gate_episodes']} can discriminate",
                   pw["clone_holds"] >= floor
                   and pw["interfered_fails"] >= floor,
                   f"clone holds {pw['clone_holds']:.3f}, interfered "
                   f"(p={pw['interfered_rate']:.3f}) fails "
                   f"{pw['interfered_fails']:.3f}, floor {floor} "
                   f"[threshold {pw['p0']:.3f} vs prior {prior['rate']:.3f}]")
        except Exception as e:
            report(f"{level} n={CONFIG['gate_episodes']} can discriminate",
                   False, repr(e))

    # 12. The verdict table classifies the registered synthetic cases —
    #     now against MEASURED controls, four legs at a time.
    try:
        def s(rate):
            return {"status": "ok", "n_episodes": CONFIG["gate_episodes"],
                    "clear_rate_strict": rate}
        # (joint 1-1, joint 1-2, control 1-1, control 1-2, expected)
        cases = [
            (0.48, 0.40, 0.48, 0.40, "compatible"),
            (0.02, 0.02, 0.48, 0.40, "severe_interference"),
            (0.48, 0.02, 0.48, 0.40, "partial_interference"),
            (0.12, 0.06, 0.60, 0.60, "degraded"),
            (0.28, 0.40, 0.48, 0.40, "inconclusive"),
            (0.02, 0.30, 0.04, 0.40, "unusable"),
        ]
        got = [interference_verdict({"1-1": s(a), "1-2": s(b)},
                                    {"1-1": s(c), "1-2": s(d)})["verdict"]
               for a, b, c, d, _ in cases]
        want = [w for *_, w in cases]
        report("verdict table classifies the registered cases",
               got == want, f"{got}")
    except Exception as e:
        report("verdict table classifies the registered cases", False,
               repr(e))

    # 13. Manifest mirrors CONFIG + scope note; write it
    #     (pre-registration surface).
    try:
        man = build_manifest()
        good = (man["config"] == CONFIG and len(man["steps"]) == 3
                and "does not gate the 1-3" in man["scope_note"])
        run_dir = REPO / CONFIG["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(man, indent=2) + "\n")
        report("manifest mirrors CONFIG + scope note", good,
               str(run_dir / "manifest.json"))
    except Exception as e:
        report("manifest mirrors CONFIG + scope note", False, repr(e))

    _receipt({"type": "dry_run", "ok": ok})
    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The gated sequence
# ---------------------------------------------------------------------------


def run(skip_to: int = 1) -> int:
    steps = steps_from(skip_to)
    # Fail here, not two hours from here, if the gate would replay the
    # collection episodes.
    assert_seed_independence()
    run_dir = REPO / CONFIG["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(build_manifest(), indent=2) + "\n")
    _receipt({"type": "falsifier_start", "skip_to": int(skip_to),
              "steps": steps, "config": CONFIG})

    if 1 in steps:
        step1_collect()
    if 2 in steps:
        step2_joint_bc()
    verdict = None
    if 3 in steps:
        verdict = step3_gate()

    _receipt({"type": "falsifier_complete", "steps": steps,
              "verdict": verdict["verdict"] if verdict else None})
    print("[interference] complete.", flush=True)
    # unusable = the harness broke, not an answer — exit 2 so the
    # operator re-runs the gate instead of reading a verdict.
    return 2 if (verdict and verdict["verdict"] == "unusable") else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate every step's assembly (profiles, "
                         "checkpoint loads with shape inference, start "
                         "states, eval commands, verdict table) without "
                         "rollouts or training.")
    ap.add_argument("--skip-to", type=int, default=1, metavar="N",
                    help="Resume at step N (1=collection, 2=joint BC, "
                         "3=bilateral gate) off the prior steps' on-disk "
                         "artifacts.")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()
    return run(skip_to=args.skip_to)


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
