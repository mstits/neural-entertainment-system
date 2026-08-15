"""SMB 1-2 consolidation round 2 — single-phase rate-building runner.

Continues checkpoints/_preserved/online_v2_FINAL_consolidated.pt (the
consolidation-1 product) for 30M env-steps in the consolidation-1 regime
(configs/mario_1_2_consol2.yaml: entrance-pinned restarts, sticky 0.25,
entropy 0.003, SIL on, SELF-anchored with loss-level KL tether 0.1).

PRE-REGISTERED EVIDENCE (consolidation-1, runs/online_1_2):
  * Probe-protocol honest clears (30 eps, greedy, sticky 0.25, jitter 16,
    cold 1-2 entrance, parallel per-episode RNG) rose from ~3% pooled
    before consolidation (5/150, attempt-5 probes — see the phase-2
    re-registration note in scripts/run_online_campaign.py) to 14.4%
    pooled across consolidation-1's three probes: 0.20 (iter 860) +
    0.1667 (iter 910) + 0.0667 (final) = 13/90
    (runs/online_1_2/campaign.jsonl).
  * The SAME checkpoints scored 0.00-0.04 on 50-episode SEQUENTIAL
    shared-stream evals (runs/online_1_2/final_eval*.json — iter 910:
    0.00 seed 7 / 0.04 seed 101; iter 860: 0.04 seed 7 / 0.00 seed 101).
    Identical noise profile; only the eval-RNG plumbing differs.

Therefore every 5M-step probe here runs BOTH protocols on the SAME
checkpoint — one parallel per-episode-RNG probe (the "probe protocol")
and one serial shared-stream probe — so the protocol gap is measured
continuously instead of discovered post-hoc.

PRE-REGISTERED GATES: none. This is a rate-building run; nothing
advances. The run completes when the 30M-step budget is spent.

PRE-REGISTERED KILLS (ProbeMonitor; unit-tested on stubbed streams in
tests/test_consol2.py):
  * competence floor — an OK probe on EITHER protocol whose median
    max-x falls below 150 kills immediately (carried from the online
    campaign: the A7 prior scores ~187 median, so <150 means the policy
    lost even frozen-prior competence);
  * probe-rate collapse — probe-protocol clear rate < 5% on 2
    consecutive OK probes AFTER the run has exceeded 10% on some earlier
    probe. Consolidation-1's own trajectory (0.20 -> 0.167 -> 0.067)
    shows single-probe dips are normal; two-in-a-row below 5% after
    demonstrated >10% capability is consolidation actively destroying
    the rate it exists to build.

`--dry-run` validates the whole assembly without training: the profile
passes the strict schema, the self-anchor invariant holds, the source
checkpoint loads + sha256-matches its pin + shape-infers into the
configured net, the seeding plan for checkpoints/mario_1_2_consol2/ is
coherent, and both probe commands assemble against real paths. Config
parse + checkpoint load + probe assembly only — no rollouts.

Long runs are launched by the operator, not by edits to this file: every
threshold lives in CONFIG below and is mirrored into
runs/consol2/manifest.json at start.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reused verbatim from the campaign controller (same probe JSON shape,
# same jsonl plumbing, same trainer stop semantics).
from scripts.run_online_campaign import (  # noqa: E402
    _append_jsonl, _load_yaml, _stop_trainer, _tail_jsonl,
    latest_checkpoint, load_checkpoint_payload, probe_summary,
    steps_per_iter,
)

# ---------------------------------------------------------------------------
# CONFIG — every threshold round 2 runs on. Mirrored verbatim into
# runs/consol2/manifest.json so pre-registration and run cannot drift.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "base_profile": "configs/mario_1_2_consol2.yaml",
    "run_dir": "runs/consol2",
    "campaign_log": "runs/consol2/campaign.jsonl",

    # Continuation source, pinned by content. sha256 verified at seed
    # time AND at dry-run; byte-identical to
    # checkpoints/mario_1_2_online_v2/vanilla_ppo_iter_00910.pt.
    "source_checkpoint":
        "checkpoints/_preserved/online_v2_FINAL_consolidated.pt",
    "source_checkpoint_sha256":
        "daa34bbedd252e317d2a313224033574210231b22f15c15ce22a68fad383b091",
    "source_iter": 910,

    # Budget: single phase, rate-building, no gates.
    "budget_env_steps": 30_000_000,

    # Honest probe protocol (both legs share this noise profile; only
    # the eval-RNG plumbing differs — that difference IS the measurement).
    "probe_every_env_steps": 5_000_000,
    "probe_episodes": 30,
    "probe_sticky": 0.25,
    "probe_jitter": 16,
    "probe_max_steps": 3000,
    "probe_eval_workers": 5,       # probe protocol only; sequential = 1
    "probe_seed": 20260815,
    "probe_timeout_s": 7200.0,     # serial shared-stream leg is ~5x slower

    # The 1-2 wall: x~2674 off-manifold-drift barrier.
    "bottleneck_x": 2674,
    "bottleneck_survival_x": 2600,

    # Kills (pre-registered; rationale in the module docstring).
    "kill_median_floor": 150.0,        # either protocol, immediate
    "kill_collapse_rate": 0.05,        # probe protocol only
    "kill_collapse_arm_rate": 0.10,    # must exceed this before arming
    "kill_collapse_consecutive": 2,

    # Controller cadence.
    "poll_secs": 15.0,
    "trainer_stop_grace_s": 180.0,
}

# The two probe legs. "probe" is the protocol consolidation-1's 14.4%
# was measured on; "sequential" is the protocol the same checkpoints
# scored 0.00-0.04 on. Names are load-bearing: they key campaign.jsonl
# rows and the ProbeMonitor's per-leg semantics.
PROTOCOLS: tuple[str, ...] = ("probe", "sequential")


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested with stubbed streams — tests/test_consol2.py)
# ---------------------------------------------------------------------------


def probe_boundaries(start_env_steps: float, budget_env_steps: int,
                     every_env_steps: int) -> list[float]:
    """Absolute env-step marks for every scheduled probe pair.

    start + every, start + 2*every, ... up to start + budget inclusive.
    30M / 5M = 6 probe pairs for the registered budget.
    """
    n = int(budget_env_steps) // int(every_env_steps)
    return [float(start_env_steps) + (k + 1) * int(every_env_steps)
            for k in range(n)]


def build_probe_command(*, checkpoint: Path, protocol: str,
                        eval_seed: int) -> list[str]:
    """One honest-probe eval_game argv for the given protocol leg.

    Both legs: 30 episodes, greedy, sticky 0.25, jitter 16, cold from
    the 1-2 entrance, --level-clear. The ONLY difference:
      probe       — --eval-workers 5 --eval-rng per-episode (parallel;
                    eval_game refuses shared-stream parallel stochastic);
      sequential  — --eval-workers 1 --eval-rng shared-stream (serial;
                    the final_eval*.json receipt protocol).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, "
                         f"got {protocol!r}")
    base = _load_yaml(REPO / CONFIG["base_profile"])
    cmd = [
        sys.executable, str(REPO / "scripts" / "eval_game.py"),
        "--game", "mario_1_2_consol2",
        "--profile", str(REPO / CONFIG["base_profile"]),
        "--rom", str(REPO / base["rom_path"]),
        "--checkpoint", str(checkpoint),
        "--episodes", str(int(CONFIG["probe_episodes"])),
        "--max-steps", str(int(CONFIG["probe_max_steps"])),
        "--sequential", "--level-clear",
        "--start-state", str(REPO / base["start_state_path"]),
        "--eval-seed", str(int(eval_seed)),
        "--sticky-prob", str(float(CONFIG["probe_sticky"])),
        "--start-jitter", str(int(CONFIG["probe_jitter"])),
    ]
    if protocol == "probe":
        cmd += ["--eval-workers", str(int(CONFIG["probe_eval_workers"])),
                "--eval-rng", "per-episode"]
    else:
        cmd += ["--eval-workers", "1", "--eval-rng", "shared-stream"]
    return cmd


def protocol_gap(probe_summ: dict, seq_summ: dict) -> dict:
    """The continuously-measured protocol gap for one checkpoint.

    Gap = probe-protocol clear rate minus sequential shared-stream clear
    rate on the SAME checkpoint. None (with a degraded status) when
    either leg failed — a failed leg must never masquerade as a 0.0.
    """
    ok = (probe_summ.get("status") == "ok"
          and seq_summ.get("status") == "ok")
    status = "ok" if ok else (
        probe_summ.get("status") if probe_summ.get("status") != "ok"
        else seq_summ.get("status"))
    return {
        "probe_clear_rate": probe_summ.get("clear_rate"),
        "sequential_clear_rate": seq_summ.get("clear_rate"),
        # Both predicates, named, on every row (2026-08-15 forensics:
        # chain-advance vs strict episode_success disagreed 13:1; a rate
        # without its predicate is uninterpretable).
        "probe_clear_rate_strict": probe_summ.get("clear_rate_strict"),
        "sequential_clear_rate_strict": seq_summ.get("clear_rate_strict"),
        "probe_clear_rate_chain": probe_summ.get("clear_rate_chain"),
        "sequential_clear_rate_chain": seq_summ.get("clear_rate_chain"),
        "clear_rate_gap": (
            float(probe_summ["clear_rate"]) - float(seq_summ["clear_rate"])
            if ok else None),
        "probe_median_max_x": probe_summ.get("median_max_x"),
        "sequential_median_max_x": seq_summ.get("median_max_x"),
        "status": status,
    }


class ProbeMonitor:
    """The two pre-registered kill criteria over the probe-pair stream.

    Fed one (probe_summ, seq_summ) pair per scheduled probe via
    `observe`; returns None while healthy, or a human-readable kill
    reason. Pure over the summaries — no filesystem, no subprocess — so
    the whole surface is testable with synthetic streams.

    Semantics (pre-registered):
      * competence floor: an OK leg (n_episodes > 0) on EITHER protocol
        with median_max_x < kill_median_floor kills immediately;
      * collapse: probe-protocol clear_rate > kill_collapse_arm_rate on
        any OK probe ARMS the criterion; once armed, clear_rate <
        kill_collapse_rate on kill_collapse_consecutive consecutive OK
        probes kills. An OK probe at or above kill_collapse_rate resets
        the streak (arming is permanent). Failed/empty probes change
        NOTHING — a broken harness is not evidence of recovery or decay.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = dict(cfg)
        self._armed = False
        self._low_streak = 0

    @staticmethod
    def _leg_ok(summ: dict) -> bool:
        return (summ.get("status") == "ok"
                and int(summ.get("n_episodes", 0) or 0) > 0)

    def observe(self, probe_summ: dict, seq_summ: dict) -> Optional[str]:
        # --- competence floor, either protocol, immediate ---
        for name, summ in (("probe", probe_summ), ("sequential", seq_summ)):
            if self._leg_ok(summ) and (
                    float(summ["median_max_x"])
                    < self.cfg["kill_median_floor"]):
                return (
                    f"KILL competence floor: {name} protocol "
                    f"median_max_x {float(summ['median_max_x']):.0f} < "
                    f"{self.cfg['kill_median_floor']:.0f}")

        # --- probe-rate collapse (probe protocol only) ---
        if self._leg_ok(probe_summ):
            rate = float(probe_summ.get("clear_rate", 0.0) or 0.0)
            if rate > self.cfg["kill_collapse_arm_rate"]:
                self._armed = True
                self._low_streak = 0
            elif self._armed and rate < self.cfg["kill_collapse_rate"]:
                self._low_streak += 1
                if self._low_streak >= self.cfg["kill_collapse_consecutive"]:
                    return (
                        f"KILL probe-rate collapse: probe-protocol clear "
                        f"rate < {self.cfg['kill_collapse_rate']:.2f} on "
                        f"{self._low_streak} consecutive probes after "
                        f"having exceeded "
                        f"{self.cfg['kill_collapse_arm_rate']:.2f}")
            else:
                self._low_streak = 0
        return None


def build_manifest(base_profile: dict) -> dict:
    """The run manifest: CONFIG + the consolidation-1 receipts, verbatim."""
    return {
        "campaign": "smb_1_2_consol2",
        "profile_name": base_profile.get("name"),
        "config": copy.deepcopy(CONFIG),
        "protocols": list(PROTOCOLS),
        "gates": [],  # rate-building: advance nothing (pre-registered)
        "preregistration_evidence": {
            "consolidation1_pooled_probe_clears": "13/90 (14.4%)",
            "consolidation1_probe_rates": [0.2, 0.16666666666666666,
                                           0.06666666666666667],
            "pre_consolidation_pooled": "5/150 (~3%, attempt-5 probes)",
            "probe_receipts": ["runs/online_1_2/campaign.jsonl"],
            "sequential_receipts": [
                "runs/online_1_2/final_eval_seed7.json",
                "runs/online_1_2/final_eval_seed101.json",
                "runs/online_1_2/final_eval_i860_seed7.json",
                "runs/online_1_2/final_eval_i860_seed101.json",
            ],
            "sequential_rates_same_checkpoints": [0.0, 0.04, 0.04, 0.0],
            "source_twin":
                "checkpoints/mario_1_2_online_v2/vanilla_ppo_iter_00910.pt",
        },
        "created_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Runtime pieces (subprocess / filesystem; exercised by --dry-run and the
# real run, not by the unit tests)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def consol2_checkpoint_dir(base: dict) -> Path:
    from src.training.profile_utils import derive_checkpoint_dir
    return REPO / derive_checkpoint_dir("./checkpoints", base.get("name"))


def seed_checkpoint_dir(base: dict, *, plan_only: bool = False) -> str:
    """Seed checkpoints/mario_1_2_consol2/ with the pinned source.

    The trainer's tear-tolerant auto-resume is the continuation
    mechanism: the source checkpoint is copied in at its TRUE iter
    (vanilla_ppo_iter_00910.pt) so generation numbering, entropy
    bookkeeping and probe scheduling all stay on the campaign's axis.
    sha256 is verified on the source before copying and on any
    already-seeded file (idempotent relaunch). Returns a description of
    what was (or would be) done; raises on any integrity mismatch.
    """
    src = REPO / CONFIG["source_checkpoint"]
    got = _sha256(src)
    if got != CONFIG["source_checkpoint_sha256"]:
        raise RuntimeError(
            f"source checkpoint sha256 mismatch: {src} is {got}, pinned "
            f"{CONFIG['source_checkpoint_sha256']} — refusing to continue "
            f"from an unverified policy")
    ckpt_dir = consol2_checkpoint_dir(base)
    dst = ckpt_dir / f"vanilla_ppo_iter_{CONFIG['source_iter']:05d}.pt"
    existing = sorted(ckpt_dir.glob("vanilla_ppo_iter_*.pt")) \
        if ckpt_dir.exists() else []
    if existing:
        if dst.exists() and _sha256(dst) != CONFIG["source_checkpoint_sha256"]:
            raise RuntimeError(
                f"{dst} exists but does not sha256-match the pinned "
                f"source — the dir holds a different lineage; refusing")
        return (f"already seeded ({len(existing)} checkpoint(s), newest "
                f"{existing[-1].name}); resume continues that lineage")
    if plan_only:
        return f"will seed {dst} from {src} (sha256 verified)"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".pt.tmp")
    shutil.copyfile(src, tmp)
    if _sha256(tmp) != CONFIG["source_checkpoint_sha256"]:
        tmp.unlink()
        raise RuntimeError(f"copy of {src} -> {tmp} failed verification")
    os.replace(tmp, dst)
    return f"seeded {dst} from {src} (sha256 verified)"


def run_probe(*, checkpoint: Path, protocol: str, eval_seed: int) -> dict:
    """Run one probe leg subprocess; returns eval_game's JSON or an error."""
    cmd = build_probe_command(checkpoint=checkpoint, protocol=protocol,
                              eval_seed=eval_seed)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=CONFIG["probe_timeout_s"])
    except subprocess.TimeoutExpired:
        return {"status": "probe_timeout"}
    if proc.returncode != 0:
        return {"status": "probe_failed",
                "detail": proc.stderr.strip()[-500:]}
    text = proc.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"status": "probe_failed", "detail": "no JSON on stdout"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return {"status": "probe_failed", "detail": f"bad JSON: {e}"}


def run_probe_pair(*, checkpoint: Path, probe_idx: int, phase_label: str,
                   env_steps: float, log_path: Path,
                   monitor: ProbeMonitor) -> Optional[str]:
    """Both protocol legs on the same checkpoint + the gap row.

    Logs one `probe` row per leg and one `probe_pair` row, then feeds
    the monitor. Returns a kill reason or None. Seeds: probe_seed +
    probe_idx for BOTH legs (pre-registered; the legs draw from
    different RNG plumbing by construction, so sharing the integer
    keeps the pair addressable by one index).
    """
    summs: dict[str, dict] = {}
    for protocol in PROTOCOLS:
        res = run_probe(checkpoint=checkpoint, protocol=protocol,
                        eval_seed=CONFIG["probe_seed"] + probe_idx)
        summ = probe_summary(res, CONFIG["bottleneck_survival_x"])
        summ["status"] = res.get("status")
        summs[protocol] = summ
        _append_jsonl(log_path, {
            "type": "probe", "phase": phase_label, "protocol_leg": protocol,
            "checkpoint": str(checkpoint), "env_steps": env_steps,
            "probe_idx": probe_idx,
            "protocol": {
                "sticky": CONFIG["probe_sticky"],
                "jitter": CONFIG["probe_jitter"],
                "episodes": CONFIG["probe_episodes"],
                "greedy": True,
                "eval_rng": ("per-episode" if protocol == "probe"
                             else "shared-stream"),
                "eval_workers": (CONFIG["probe_eval_workers"]
                                 if protocol == "probe" else 1),
            },
            **{k: v for k, v in summ.items() if k != "status"},
            "status": summ["status"]})
    _append_jsonl(log_path, {
        "type": "probe_pair", "checkpoint": str(checkpoint),
        "env_steps": env_steps, "probe_idx": probe_idx,
        **protocol_gap(summs["probe"], summs["sequential"])})
    return monitor.observe(summs["probe"], summs["sequential"])


# ---------------------------------------------------------------------------
# Dry run — config parse + checkpoint load + probe assembly. NOT rollouts.
# ---------------------------------------------------------------------------


def dry_run() -> int:
    """Validate the whole assembly without training. Returns exit code."""
    from src.training.config_schema import check_profile
    from src.training.profile_utils import resolve_encoder

    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" — {detail}" if detail else ""), flush=True)

    # 1. Profile parses + strict schema.
    base_path = REPO / CONFIG["base_profile"]
    try:
        base = _load_yaml(base_path)
        warns = check_profile(base, strict=False)
        report("profile schema (strict)", not warns,
               "; ".join(warns) if warns else str(base_path))
    except Exception as e:
        report("profile schema (strict)", False, repr(e))
        return 1

    # 2. Self-anchor invariant: the frozen prior IS the continued policy.
    rl = base.get("reinforce", {})
    report(
        "self-anchor invariant (prior == continuation source, coef 0.1)",
        rl.get("kl_anchor_checkpoint") == CONFIG["source_checkpoint"]
        and rl.get("kl_anchor_loss_coef") == 0.1
        and rl.get("kl_beta_start") == 0.0 and rl.get("kl_beta_end") == 0.0
        and rl.get("actor_freeze_steps") == 0,
        f"anchor={rl.get('kl_anchor_checkpoint')} "
        f"coef={rl.get('kl_anchor_loss_coef')}")

    # 3. Source checkpoint sha256 matches the pin.
    try:
        src = REPO / CONFIG["source_checkpoint"]
        sha = _sha256(src)
        report("source checkpoint sha256 matches pin",
               sha == CONFIG["source_checkpoint_sha256"], sha[:16] + "...")
    except Exception as e:
        report("source checkpoint sha256 matches pin", False, repr(e))

    # 4. Source checkpoint loads at the pinned iter and shape-infers into
    #    the configured net (this is a torch.load, not a rollout).
    try:
        import torch
        from src.models.tile_policy import build_tile_policy_from_checkpoint
        _, _, obs_width = resolve_encoder(base)
        payload = torch.load(str(REPO / CONFIG["source_checkpoint"]),
                             map_location="cpu", weights_only=False)
        sd = payload.get("net_state_dict", payload)
        prior, is_recurrent = build_tile_policy_from_checkpoint(
            payload, num_actions=len(base["action_space"]),
            feature_dim=obs_width)
        missing, _ = prior.load_state_dict(sd, strict=False)
        fc1 = tuple(sd["fc1.weight"].shape)
        fc2 = tuple(sd["fc2.weight"].shape)
        it = int(payload.get("iter", -1))
        good = (
            not is_recurrent and not missing
            and it == CONFIG["source_iter"]
            and fc1 == (int(rl["tile_hidden_dim"]), obs_width)
            and fc2 == (int(rl["tile_trunk_dim"]), int(rl["tile_hidden_dim"]))
        )
        report("source checkpoint loads at pinned iter + net dims", good,
               f"iter={it} fc1={fc1} fc2={fc2} obs={obs_width} "
               f"recurrent={is_recurrent} missing={missing}")
    except Exception as e:
        report("source checkpoint loads at pinned iter + net dims", False,
               repr(e))

    # 5. Seeding plan for the consol2 checkpoint dir is coherent.
    try:
        plan = seed_checkpoint_dir(base, plan_only=True)
        report("checkpoint-dir seeding plan", True, plan)
    except Exception as e:
        report("checkpoint-dir seeding plan", False, repr(e))

    # 6. Both probe commands assemble against real paths.
    for protocol in PROTOCOLS:
        try:
            cmd = build_probe_command(
                checkpoint=REPO / CONFIG["source_checkpoint"],
                protocol=protocol, eval_seed=CONFIG["probe_seed"])
            missing_paths = [
                a for a in cmd
                if ("/" in a and not a.startswith("-")
                    and not Path(a).exists())]
            report(f"{protocol}-protocol probe command assembles",
                   not missing_paths,
                   " ".join(cmd) if not missing_paths
                   else f"missing paths: {missing_paths}")
        except Exception as e:
            report(f"{protocol}-protocol probe command assembles", False,
                   repr(e))

    # 7. Probe schedule: 6 pairs over the 30M budget.
    bounds = probe_boundaries(
        CONFIG["source_iter"] * steps_per_iter(base),
        CONFIG["budget_env_steps"], CONFIG["probe_every_env_steps"])
    report("probe schedule (6 dual-protocol probes over 30M)",
           len(bounds) == 6, f"boundaries={[int(b) for b in bounds]}")

    # 8. Manifest mirrors CONFIG + evidence.
    man = build_manifest(base)
    report("manifest mirrors CONFIG + consolidation-1 evidence",
           man["config"] == CONFIG and man["gates"] == []
           and man["preregistration_evidence"][
               "consolidation1_pooled_probe_clears"] == "13/90 (14.4%)")

    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The single-phase run
# ---------------------------------------------------------------------------


def run() -> int:
    base = _load_yaml(REPO / CONFIG["base_profile"])
    run_dir = REPO / CONFIG["run_dir"]
    log_path = REPO / CONFIG["campaign_log"]
    ckpt_dir = consol2_checkpoint_dir(base)
    metrics_path = ckpt_dir / "metrics.jsonl"
    spi = steps_per_iter(base)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_note = seed_checkpoint_dir(base)

    manifest = build_manifest(base)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    _append_jsonl(log_path, {"type": "campaign_start", "config": CONFIG,
                             "steps_per_iter": spi, "seed_note": seed_note})

    # Resume-aware baseline: budget is consumed RELATIVE to the newest
    # seeded/produced checkpoint, so a relaunch never re-buys spent steps.
    payload = load_checkpoint_payload(ckpt_dir)
    start_iter = int(payload.get("iter", CONFIG["source_iter"])) \
        if isinstance(payload, dict) else CONFIG["source_iter"]
    start_steps = float(start_iter) * spi
    end_steps = float(CONFIG["source_iter"]) * spi + CONFIG["budget_env_steps"]
    remaining = max(0.0, end_steps - start_steps)
    iters = int(remaining // spi) + (1 if remaining % spi else 0)
    if iters <= 0:
        _append_jsonl(log_path, {"type": "campaign_complete",
                                 "env_steps": start_steps,
                                 "note": "budget already spent"})
        print("[consol2] budget already spent; nothing to run.", flush=True)
        return 0

    boundaries = probe_boundaries(
        CONFIG["source_iter"] * spi, CONFIG["budget_env_steps"],
        CONFIG["probe_every_env_steps"])
    boundaries = [b for b in boundaries if b > start_steps]

    _append_jsonl(log_path, {
        "type": "phase_start", "phase": "consolidation2",
        "profile": str(REPO / CONFIG["base_profile"]),
        "iters": iters, "start_env_steps": start_steps,
        "end_env_steps": end_steps,
        "probe_boundaries": boundaries})

    cmd = [sys.executable, str(REPO / "scripts" / "train_game.py"),
           "--profile", str(REPO / CONFIG["base_profile"]),
           "--iters", str(iters), "--no-supervise", "--strict-config"]
    phase_log = run_dir / "consolidation2.log"
    with open(phase_log, "ab") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf,
                                stderr=subprocess.STDOUT)

    monitor = ProbeMonitor(CONFIG)
    offset = metrics_path.stat().st_size if metrics_path.exists() else 0
    cum_env_steps = start_steps
    probe_idx = 0

    def _abort(reason: str) -> int:
        _stop_trainer(proc)
        _append_jsonl(log_path, {"type": "abort", "reason": reason,
                                 "env_steps": cum_env_steps})
        print(f"[consol2] ABORT: {reason}", flush=True)
        return 2

    while True:
        rows, offset = _tail_jsonl(metrics_path, offset)
        for row in rows:
            cum_env_steps = max(
                cum_env_steps, (float(row.get("generation", 0)) + 1.0) * spi)

        while boundaries and cum_env_steps >= boundaries[0]:
            boundary = boundaries.pop(0)
            ckpt = latest_checkpoint(ckpt_dir)
            if ckpt is None:
                continue
            probe_idx += 1
            reason = run_probe_pair(
                checkpoint=ckpt, probe_idx=probe_idx,
                phase_label="consolidation2", env_steps=cum_env_steps,
                log_path=log_path, monitor=monitor)
            if reason:
                return _abort(reason)
            _append_jsonl(log_path, {"type": "probe_boundary_done",
                                     "boundary": boundary,
                                     "probe_idx": probe_idx})

        if proc.poll() is not None:
            rows, offset = _tail_jsonl(metrics_path, offset)
            for row in rows:
                cum_env_steps = max(
                    cum_env_steps,
                    (float(row.get("generation", 0)) + 1.0) * spi)
            if proc.returncode != 0:
                return _abort(
                    f"trainer subprocess exited {proc.returncode}")
            break

        time.sleep(CONFIG["poll_secs"])

    # Final dual-protocol probe for the record (also covers the 30M
    # boundary if the trainer finished between polls).
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is not None:
        probe_idx += 1
        reason = run_probe_pair(
            checkpoint=ckpt, probe_idx=probe_idx, phase_label="final",
            env_steps=cum_env_steps, log_path=log_path, monitor=monitor)
        if reason:
            _append_jsonl(log_path, {"type": "abort", "reason": reason,
                                     "env_steps": cum_env_steps,
                                     "note": "kill condition met on the "
                                             "final probe; run already "
                                             "complete — recorded, not "
                                             "stopped"})
    _append_jsonl(log_path, {"type": "campaign_complete",
                             "env_steps": cum_env_steps})
    print("[consol2] complete.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate profile, source checkpoint, seeding "
                         "plan and both probe commands without training.")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()
    return run()


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
