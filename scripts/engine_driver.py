"""The engine. Decides what to run next, runs it, judges it, repeats.

DESIGN BAR: the machine is left alone for three weeks and the software is
meaningfully better when its owner returns. That rules out three things
this project has previously shipped and called automation:

  * a status reporter (`engine_status.py`) — it describes, it does not act;
  * a fixed action list — four actions is one day, then silence;
  * anything that dies with the launching session.

So the next action is COMPUTED FROM STATE, never consumed from a list.
`plan()` looks at what exists on disk and returns the highest-priority
thing not yet done. That cannot run dry: finishing a level makes the next
level's onboarding the new top priority, and the SMB pipeline alone is
28 unbanked levels deep.

WHAT IT MAY DO. Only the action kinds enumerated in `plan()`, each with a
pre-registered gate in configs/engine_queue.yaml. It never invents a
command, never edits CLAIMS.md, a gate, or any banked artifact, and never
deletes anything. When it decides a run failed, it quarantines by
renaming and records why.

SAFETY, because three weeks unattended is a long time to be wrong:

  * ONE emulator job at a time (the physics budget), enforced by probing
    the process table, not by trusting our own state file;
  * a circuit breaker — CONSECUTIVE_FAILURE_LIMIT failures in a row and
    the engine halts and says so, rather than burning three weeks
    relaunching a broken step;
  * per-action attempt cap, so one poisoned action cannot monopolise the
    queue;
  * a disk floor, because a full disk corrupts checkpoints rather than
    failing cleanly;
  * a wall-clock cap per action, so a hung run is killed instead of
    holding the machine forever;
  * every decision appended to runs/engine/journal.jsonl BEFORE it is
    acted on, so a three-week-old trail explains itself.

Its own verdicts are advisory: banking a claim into CLAIMS.md stays a
human act. The engine writes runs/engine/proposed_claims.jsonl instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO / "runs" / "engine"
JOURNAL = ENGINE_DIR / "journal.jsonl"
STATE_PATH = ENGINE_DIR / "state.json"
PROPOSED = ENGINE_DIR / "proposed_claims.jsonl"

CONSECUTIVE_FAILURE_LIMIT = 3
MAX_ATTEMPTS_PER_ACTION = 2
DISK_FLOOR_GB = 40.0
DEFAULT_ACTION_TIMEOUT_H = 14.0

# The SMB ladder, in the order the pipeline has been onboarding them.
# Banked: 1-1 1-2 1-3 1-4. 2-1 in flight.
SMB_LEVELS = ["1-1", "1-2", "1-3", "1-4",
              "2-1", "2-2", "2-3", "2-4",
              "3-1", "3-2", "3-3", "3-4",
              "4-1", "4-2", "4-3", "4-4"]


@dataclass
class Action:
    id: str
    kind: str
    cmd: list[str]
    needs_emulator: bool
    gate: str
    timeout_h: float = DEFAULT_ACTION_TIMEOUT_H
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- state

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            # A truncated state file must not wedge the engine for three
            # weeks; start clean and say so in the journal.
            journal({"type": "state_corrupt_reset"})
    return {"attempts": {}, "consecutive_failures": 0, "halted": None,
            "running": None, "completed": {}}


def save_state(state: dict) -> None:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_PATH)          # atomic; a torn state file is worse
                                     # than a stale one


def journal(row: dict) -> None:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("t", time.time())
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------- machine checks

def emulator_busy() -> Optional[int]:
    """PID of a live emulator-heavy job, or None. Probes ps, not our state.

    Trusting the state file here would let one crash leave the engine
    convinced the machine is busy forever.
    """
    pats = ("run_online_campaign", "train_game", "go_explore_solve",
            "hazard_collect", "replay_sweep", "eval_game", "soak_harness")
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, timeout=20)
    except (subprocess.SubprocessError, OSError):
        # Unknown is treated as BUSY, never as free. Catching OSError
        # matters as much as SubprocessError: if `ps` cannot be executed
        # at all the tick would otherwise raise, and an engine that dies
        # on an unreadable process table is an engine that does nothing
        # for the rest of the three weeks.
        return -1
    me = os.getpid()
    for line in out.stdout.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        if not pid_s.isdigit() or int(pid_s) == me:
            continue
        if "engine_driver" in cmd:
            continue
        if any(p in cmd for p in pats):
            return int(pid_s)
    return None


def disk_free_gb(path: Path = REPO) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


# --------------------------------------------------------- observations

def campaign_terminal(run_dir: Path) -> Optional[str]:
    """Terminal state of a campaign from its own log, post-last-resume."""
    log = run_dir / "campaign.jsonl"
    if not log.exists():
        return None
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None
    last_start = max((i for i, r in enumerate(rows)
                      if r.get("type") == "campaign_start"), default=-1)
    for r in reversed(rows[last_start + 1:]):
        if r.get("type") in ("campaign_complete", "abort", "kill"):
            return str(r.get("type"))
    return "running"


def honest_eval_done(level: str, repo: Path = REPO) -> bool:
    """Has this level been scored under the honest protocol, on 2 seeds?

    Detected by CONTENT, not by path. Eval receipts do not live in a
    predictable directory — 1-2's and 1-4's sit under `runs/online_<tag>/`
    while 1-3's ended up in `runs/consol2_1_3_round2/` — so a path glob
    reports a banked level as unscored and the engine re-runs six hours of
    eval it already has. Each receipt names its own `game`, which is the
    fact worth matching on.

    Two distinct seeds are required because one seed is not the honest
    protocol, and a level scored once is not scored.
    """
    tag = level.replace("-", "_")
    want = f"mario_{tag}_"
    seeds: set[Any] = set()
    for p in (repo / "runs").glob("**/*eval*seed*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        if str(rec.get("game", "")).startswith(want) and \
                int(rec.get("n_episodes") or 0) >= 50 and \
                float(rec.get("sticky_prob") or 0) > 0:
            seeds.add(rec.get("eval_seed"))
    return len(seeds) >= 2


def level_has_campaign(level: str, repo: Path = REPO) -> bool:
    return (repo / "runs" / f"online_{level.replace('-', '_')}").exists()


def level_has_config(level: str, repo: Path = REPO) -> bool:
    return (repo / "configs" / f"campaign_{level.replace('-', '_')}.yaml").exists()


def level_has_tape(level: str, repo: Path = REPO) -> bool:
    tag = level.replace("-", "_")
    return any((repo / "runs").glob(f"ge_{tag}_solve/solutions/sol_*.json"))


def level_has_ladder(level: str, repo: Path = REPO) -> bool:
    tag = level.replace("-", "_")
    return (repo / "checkpoints" / f"online_{tag}"
            / "restart_states" / "index.json").exists()


# --------------------------------------------------- action validation

def script_flags(script: Path) -> set[str]:
    """Every --flag the script's argparse declares, read from its source.

    Source-scraping rather than `--help` because importing or running an
    unknown script to discover its interface is exactly the kind of side
    effect an unattended engine must not have.
    """
    try:
        text = script.read_text()
    except OSError:
        return set()
    import re
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


def validate_action(action: Action, repo: Path = REPO) -> tuple[bool, str]:
    """Could this action actually run? Checked BEFORE it is ever launched.

    This is the guard that separates an engine from a liability. A driver
    that emits a command whose script or flag does not exist fails, retries,
    trips the circuit breaker and then sits dead — which over a three-week
    absence is indistinguishable from having built nothing. Three of this
    planner's first seven action kinds named interfaces that did not exist
    (`eval_game --honest-suite`, `go_explore_solve --level`, and
    `make_campaign_config.py`, which was not a file at all).

    So an action is emittable only if its script exists, every flag it
    passes is declared by that script, and every path-shaped argument is
    present on disk. Anything failing is reported as needing a human and
    SKIPPED — the engine then does the work it can do, instead of halting.
    """
    if not action.cmd:
        return False, "empty command"
    script = repo / action.cmd[0]
    if not script.exists():
        return False, f"no such script: {action.cmd[0]}"
    declared = script_flags(script)
    for tok in action.cmd[1:]:
        if tok.startswith("--") and tok not in declared:
            return False, f"{script.name} does not accept {tok}"
    for tok in action.cmd[1:]:
        if tok.startswith("--") or tok.startswith("-"):
            continue
        looks_like_path = ("/" in tok and not tok.startswith("runs/engine"))
        if looks_like_path and not (repo / tok).exists():
            # outputs are allowed not to exist; inputs are not
            if not any(o in action.cmd for o in ("--out", "-o")) or \
                    action.cmd[action.cmd.index(tok) - 1] not in ("--out", "-o"):
                return False, f"input path missing: {tok}"
    return True, "ok"


def rom_for(profile_path: Path, repo: Path = REPO) -> Optional[str]:
    try:
        import yaml
        prof = yaml.safe_load((repo / profile_path).read_text()) or {}
    except Exception:
        return None
    rom = prof.get("rom_path")
    return rom if rom and (repo / rom).exists() else None


def honest_eval_action(level: str, seed: int,
                       repo: Path = REPO) -> Optional[Action]:
    """The honest protocol, argv built from what is on disk.

    Protocol constants are the banked ones and are not parameters: cold
    entrance start, greedy argmax, sticky 0.25, jitter 16, strict clear
    predicate, 50 episodes per seed over two seeds. Changing any of them
    would make the result incomparable with every banked level.
    """
    tag = level.replace("-", "_")
    profile = Path(f"configs/mario_{tag}_online_v1.yaml")
    if not (repo / profile).exists():
        return None
    try:
        import yaml
        prof = yaml.safe_load((repo / profile).read_text()) or {}
    except Exception:
        return None
    rom = rom_for(profile, repo)
    start = prof.get("start_state_path")
    ckpts = sorted((repo / "checkpoints" / f"mario_{tag}_online_v1").glob("*.pt"))
    if not (rom and start and ckpts):
        return None
    return Action(
        id=f"honest_eval_{tag}_seed{seed}", kind="honest_eval",
        needs_emulator=True, timeout_h=6.0,
        cmd=["scripts/eval_game.py",
             "--game", f"mario_{tag}_online_v1",
             "--profile", str(profile),
             "--rom", rom,
             "--checkpoint", str(ckpts[-1].relative_to(repo)),
             "--episodes", "50", "--max-steps", "3000",
             "--sequential", "--level-clear",
             "--start-state", str(start),
             "--eval-seed", str(seed),
             "--sticky-prob", "0.25", "--start-jitter", "16",
             "--eval-workers", "5", "--eval-rng", "per-episode"],
        gate="Honest protocol, strict predicate, pooled over two seeds "
             "(>=100 episodes). Any non-zero pooled rate is a candidate "
             "claim; the human banks it, not the engine.",
        meta={"level": level, "seed": seed})


# ---------------------------------------------------------------- plan

def plan(state: dict, repo: Path = REPO) -> Optional[Action]:
    """The highest-priority undone thing. Pure w.r.t. the filesystem.

    Order is deliberate: finish and judge what is already started before
    starting anything new, then the cheap gated research step, then extend
    the level pipeline. Extending last is what keeps three weeks of work
    available without ever letting half-finished work rot.
    """
    done = state.get("completed", {})
    skipped: list[str] = []

    def offer(a: Optional[Action]) -> Optional[Action]:
        """Emit an action only if it is genuinely runnable."""
        if a is None or a.id in done:
            return None
        ok, why = validate_action(a, repo)
        if ok:
            return a
        skipped.append(f"{a.id}: {why}")
        return None

    candidates: list[Optional[Action]] = []

    # 1. Score finished campaigns before starting anything new.
    for level in SMB_LEVELS:
        tag = level.replace("-", "_")
        rd = repo / "runs" / f"online_{tag}"
        if rd.exists() and campaign_terminal(rd) in ("campaign_complete",
                                                     "abort", "kill"):
            if not honest_eval_done(level, repo):
                for seed in (7, 101):
                    candidates.append(honest_eval_action(level, seed, repo))

    # 2. The synthesis's Phase-1 gate: cheap, and everything the three
    #    converged DR rounds recommend sits downstream of it.
    candidates.append(Action(
        id="hazard_phase1", kind="benchmark", needs_emulator=True,
        timeout_h=2.0,
        cmd=["scripts/hazard_collect.py", "--benchmark",
             "--profile", "configs/mario_tiles.yaml",
             "--rom", "roms/Super Mario Bros. (World).nes"],
        gate=">=1000 worker-ticks/s and <1h projected for 100k labels; "
             "below that Phase 1 is KILLED per the synthesis and the "
             "fallback is observational deaths from rollout logs."))

    # 3. Verify the banked EXHIBITION corpus once.
    candidates.append(Action(
        id="replay_sweep_full", kind="sweep", needs_emulator=True,
        timeout_h=8.0,
        cmd=["scripts/replay_sweep.py", "--glob",
             "runs/**/solutions/*.json", "--out",
             "runs/engine/replay_sweep_full.json"],
        gate="Zero genuine replay FAILUREs. ERRORs are unverifiable, "
             "reported separately, never counted as passes."))

    # 4. Extend the pipeline: run the campaign for any level that already
    #    has a validated config but no run. Onboarding a level that lacks
    #    a config needs a human (see the preflight report) — the engine
    #    will not fabricate one.
    for level in SMB_LEVELS:
        tag = level.replace("-", "_")
        if honest_eval_done(level, repo) or level_has_campaign(level, repo):
            continue
        if level_has_config(level, repo):
            candidates.append(Action(
                id=f"campaign_{tag}", kind="campaign", needs_emulator=True,
                timeout_h=14.0,
                cmd=["scripts/run_online_campaign.py", "--campaign-config",
                     f"configs/campaign_{tag}.yaml"],
                gate="Phase gates as pre-registered in the campaign config; "
                     "the honest eval that follows is the real claim.",
                meta={"level": level}))

    for cand in candidates:
        got = offer(cand)
        if got is not None:
            return got
    if skipped:
        journal({"type": "plan_skipped", "skipped": skipped[:20]})
    return None


# --------------------------------------------------------------- launch

def launch(action: Action, repo: Path = REPO) -> int:
    """Start an action detached, in its own session. Returns pid."""
    sys.path.insert(0, str(repo))
    from scripts.detach import launch as detach_launch
    log = ENGINE_DIR / "logs" / f"{action.id}.log"
    cmd = [str(repo / ".venv/bin/python")] + [
        str(repo / c) if c.startswith("scripts/") else c for c in action.cmd]
    return detach_launch(cmd, log, cwd=repo)


def guard_reasons(state: dict, repo: Path = REPO) -> list[str]:
    """Everything that forbids launching anything right now."""
    out: list[str] = []
    if state.get("halted"):
        out.append(f"halted: {state['halted']}")
    if state.get("consecutive_failures", 0) >= CONSECUTIVE_FAILURE_LIMIT:
        out.append(f"circuit breaker: {state['consecutive_failures']} "
                   f"consecutive failures")
    free = disk_free_gb(repo)
    if free < DISK_FLOOR_GB:
        out.append(f"disk floor: {free:.1f} GB free < {DISK_FLOOR_GB}")
    return out


def tick(state: dict, repo: Path = REPO, dry: bool = False) -> dict:
    """One decision. Returns a record of what was decided and why."""
    busy = emulator_busy()
    if busy is not None:
        rec = {"type": "tick", "decision": "wait",
               "reason": f"emulator job live (pid {busy})"}
        journal(rec)
        return rec

    blocked = guard_reasons(state, repo)
    if blocked:
        rec = {"type": "tick", "decision": "blocked", "reason": "; ".join(blocked)}
        journal(rec)
        return rec

    action = plan(state, repo)
    if action is None:
        rec = {"type": "tick", "decision": "idle",
               "reason": "nothing left in the computed plan"}
        journal(rec)
        return rec

    attempts = state.setdefault("attempts", {})
    n = attempts.get(action.id, 0)
    if n >= MAX_ATTEMPTS_PER_ACTION:
        state.setdefault("completed", {})[action.id] = {
            "status": "abandoned", "attempts": n}
        rec = {"type": "tick", "decision": "abandon", "action": action.id,
               "reason": f"{n} attempts reached the cap"}
        journal(rec)
        return rec

    # Journal the intent BEFORE acting, so a crash mid-launch is legible.
    journal({"type": "launch_intent", "action": asdict(action),
             "attempt": n + 1})
    if dry:
        return {"type": "tick", "decision": "dry-run", "action": action.id}

    pid = launch(action, repo)
    attempts[action.id] = n + 1
    state["running"] = {"id": action.id, "pid": pid,
                        "started": time.time(),
                        "timeout_h": action.timeout_h}
    rec = {"type": "tick", "decision": "launched", "action": action.id,
           "pid": pid}
    journal(rec)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=float, default=600.0)
    ap.add_argument("--preflight", action="store_true",
                    help="validate every action the planner can emit")
    ap.add_argument("--plan", action="store_true",
                    help="print the computed next action and exit")
    args = ap.parse_args(argv)

    state = load_state()
    if args.preflight:
        empty = {"completed": {}, "attempts": {}}
        rows, seen = [], set()
        for lvl in SMB_LEVELS:
            for seed in (7, 101):
                a = honest_eval_action(lvl, seed)
                if a and a.id not in seen:
                    seen.add(a.id)
                    rows.append((a, validate_action(a)))
        probe = plan(empty)
        bad = 0
        for a, (ok, why) in rows:
            if not ok:
                bad += 1
            print(f"  [{'ok  ' if ok else 'FAIL'}] {a.id}: {why}")
        print(f"\n{len(rows) - bad}/{len(rows)} honest-eval actions runnable")
        print(f"next planned action: {probe.id if probe else 'none'}")
        return 1 if probe is None else 0
    if args.plan:
        a = plan(state)
        print(json.dumps(asdict(a), indent=2) if a else "nothing planned")
        return 0
    if args.once or args.dry_run:
        rec = tick(state, dry=args.dry_run)
        save_state(state)
        print(json.dumps(rec))
        return 0
    while True:
        state = load_state()
        rec = tick(state)
        save_state(state)
        print(json.dumps(rec), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
