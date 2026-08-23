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
# A full suite is ~6 minutes; running one every tick is pure
# thrash. Recurring maintenance earns a slot a few times a day,
# not a hundred times.
RECURRING_COOLDOWN_H = 6.0
DISK_FLOOR_GB = 40.0
# A benchmark measures the machine as much as the method, so it may only
# run on a quiet one. The 1-minute load average must sit below this, and
# the machine must have been quiet for QUIET_SETTLE_S, before any action
# whose verdict is a throughput number.
QUIET_LOAD_MAX = 3.0
QUIET_SETTLE_S = 900.0
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
    # True for actions whose result is a throughput measurement. Such an
    # action is deferred until the machine is demonstrably quiet.
    needs_quiet: bool = False
    meta: dict = field(default_factory=dict)
    # A path that exists if and only if this action succeeded. Without one
    # the reaper can only say "the process ended", which is not the same
    # as "the work is done".
    done_marker: Optional[str] = None
    # Recurring actions are meant to run again — they are never recorded
    # as completed and are exempt from the attempt cap. The suite check is
    # the archetype: capping it at two runs would silence exactly the
    # watchdog that caught a test red for three days.
    recurring: bool = False


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


def campaign_interrupted(run_dir: Path, repo: Path = REPO) -> Optional[int]:
    """Phase index of a campaign that stopped without recording an end.

    Returns None if the campaign is live, finished, or absent.

    This is the one place liveness may be inferred from the process
    table, and the distinction matters because inferring the OPPOSITE is
    what once cost a wrong ledger entry and a 30M-step re-run. That error
    read a missing process as "the run died" — an OUTCOME, which only the
    log can supply. Here nothing is concluded about the outcome: the log
    is still authoritative that no terminal row exists, and `ps` is
    authoritative that nothing is executing. Together those mean
    interrupted, which is a resumable state, not a verdict.

    Without this, any reboot, crash or stray kill strands a campaign
    forever: the log says running, no process is running, and the
    pipeline skips the level because a run directory exists.
    """
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
    live = rows[last_start + 1:]
    if any(r.get("type") in ("campaign_complete", "abort", "kill")
           for r in live):
        return None                      # it recorded how it ended
    if emulator_busy() is not None:
        return None                      # something is executing; leave it
    phases = [r.get("phase") for r in live if r.get("type") == "phase_start"]
    return int(phases[-1]) if phases and phases[-1] is not None else 0


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

    def consider(rec: Any) -> None:
        if not isinstance(rec, dict):
            return
        if str(rec.get("game", "")).startswith(want) and \
                int(rec.get("n_episodes") or 0) >= 50 and \
                float(rec.get("sticky_prob") or 0) > 0:
            seeds.add(rec.get("eval_seed"))

    for p in (repo / "runs").glob("**/*eval*seed*.json"):
        try:
            consider(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    # eval_game appends EVERY result to <ckpt_dir>/eval.jsonl, which is the
    # receipt an engine-launched eval actually leaves — the runbook's
    # `> final_eval_seedN.json` is a shell redirect no detached launch
    # performs. Reading only the redirected files made a completed eval
    # invisible, so the engine would have re-run it until the attempt cap.
    for p in (repo / "checkpoints").glob("**/eval.jsonl"):
        try:
            for line in p.read_text().splitlines():
                if line.strip():
                    consider(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
    return len(seeds) >= 2


def honest_eval_current(level: str, repo: Path = REPO) -> bool:
    """Scored, AND scored against the policy as it stands now.

    `honest_eval_done` asks whether a level was ever scored. That is the
    wrong question after a re-consolidation: 1-4 was banked at 51%, its
    campaign was resumed and finished with a final probe at 0.633, and
    because an eval existed the engine refused to measure the improved
    policy at all. An eval is stale the moment a checkpoint newer than it
    appears.
    """
    if not honest_eval_done(level, repo):
        return False
    tag = level.replace("-", "_")
    ckpt_dir = repo / "checkpoints" / f"mario_{tag}_online_v1"
    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        return True
    newest_ckpt = max(c.stat().st_mtime for c in ckpts)

    # The file's mtime is useless here: eval_game appends EVERY probe to
    # the same eval.jsonl, so a campaign's own 30-episode probes keep
    # refreshing it and a stale honest eval looks current. Compare against
    # the timestamp carried by the honest records themselves.
    newest_honest = 0.0
    log = ckpt_dir / "eval.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(rec.get("n_episodes") or 0) >= 50 and \
                    float(rec.get("sticky_prob") or 0) > 0:
                newest_honest = max(newest_honest,
                                    float(rec.get("timestamp") or 0.0))
    return newest_honest >= newest_ckpt


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

def script_flags(script: Path) -> tuple[set[str], set[str]]:
    """(declared, required) --flags, parsed from the script's own source.

    Parsed with `ast` rather than executed, because importing or running
    an unknown script to discover its interface is exactly the side
    effect an unattended engine must not have. Regex was the first
    version and could only see declarations.

    REQUIRED matters as much as declared, and missing it cost a real
    result: the hazard Phase-1 action passed only flags the script
    accepts, so validation said 'ok', and hazard_collect.py then exited
    immediately with "the following arguments are required: --states".
    Because that action declared no done_marker, the reaper recorded it
    as 'finished' — the exact "an exit is not an outcome" case the reaper
    documents — and the gate the whole research synthesis sits on was
    silently skipped.
    """
    import ast
    try:
        tree = ast.parse(script.read_text())
    except (OSError, SyntaxError):
        return set(), set()
    declared: set[str] = set()
    required: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
        if not names:
            continue
        declared.update(names)
        for kw in node.keywords:
            if kw.arg == "required" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value is True:
                required.update(names)
    return declared, required


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
    if action.cmd[0] == "-m":
        mod = action.cmd[1].replace(".", "/") + ".py"
        if not (repo / mod).exists():
            return False, f"no such module: {action.cmd[1]}"
        return True, "ok (module)"
    script = repo / action.cmd[0]
    if not script.exists():
        return False, f"no such script: {action.cmd[0]}"
    declared, required = script_flags(script)
    for tok in action.cmd[1:]:
        if tok.startswith("--") and tok not in declared:
            return False, f"{script.name} does not accept {tok}"
    missing = sorted(required - set(action.cmd))
    if missing:
        return False, (f"{script.name} requires {', '.join(missing)}, "
                       f"not supplied")
    for tok in action.cmd[1:]:
        if tok.startswith("--") or tok.startswith("-"):
            continue
        # A glob is a pattern, not a path: it never exists as a file, so
        # checking it as one made replay_sweep_full permanently
        # un-runnable — validation rejected it every tick, silently, and
        # the corpus went unverified.
        if any(ch in tok for ch in "*?["):
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


def onboarding_actions(level: str, repo: Path = REPO) -> list[Action]:
    """The runbook's W-chain for one level, as validated actions.

    Encodes docs/receipts/two_one_runbook_2026-08-17.md, which is the
    sequence 1-3, 1-4 and 2-1 were all onboarded by. Each step is offered
    only when its own output is missing, so the chain resumes wherever it
    stopped rather than redoing work.

    Two provenance assertions are passed explicitly rather than left to
    convention: `--expect-level` and `--expect-root` on the rung
    selection. Those flags already existed and were not used when 2-1 and
    1-4 were onboarded by hand, which is how both ended up consuming
    another level's ladder.

    Not included: capturing an entrance state. A brand-new level's
    entrance comes either from `capture_start_state.py` against a real
    boot or from the previous level's chain handoff, and choosing between
    those is a judgement about what the level's start legitimately IS —
    not something to guess unattended.
    """
    t = tag_of(level)
    prof = f"configs/mario_{t}_online_v1.yaml"
    entrance = f"checkpoints/ge_entrances/smb_{t}_entrance.state"
    rom = "roms/Super Mario Bros. (World).nes"
    ladder_raw = f"checkpoints/backward_states/{level}_online"
    rungs = f"checkpoints/online_{t}/restart_states"
    out: list[Action] = []

    if not (repo / prof).exists():
        out.append(Action(
            id=f"config_{t}", kind="config", needs_emulator=False,
            timeout_h=0.2,
            cmd=["scripts/make_campaign_config.py", "--level", level],
            gate="Both configs written by derivation with zero residual "
                 "references to the template level.",
            meta={"level": level}))
        return out          # everything downstream needs the profile

    if not (repo / ladder_raw).exists():
        out.append(Action(
            id=f"mint_{t}", kind="mint", needs_emulator=True, timeout_h=1.0,
            cmd=["scripts/mint_backward_states.py", "--level", level,
                 "--run", f"runs/ge_{t}_solve", "--profile", prof,
                 "--out", ladder_raw],
            gate="Aborts unless the tape replays to its banked clear — "
                 "the replay IS the verification.",
            meta={"level": level}))
    elif not (repo / rungs / "index.json").exists():
        out.append(Action(
            id=f"select_{t}", kind="select", needs_emulator=False,
            timeout_h=0.2,
            cmd=["scripts/select_restart_states.py", "--ladder", ladder_raw,
                 "--out", rungs, "--auto-targets", "6",
                 "--expect-level", level, "--expect-root", entrance],
            gate="6 rungs whose index names THIS level and THIS root; the "
                 "expect-* flags make a foreign ladder a hard failure.",
            meta={"level": level}))

    if not (repo / f"checkpoints/wavefront/mario_{t}_dmap.pkl").exists():
        tapes = sorted((repo / "runs" / f"ge_{t}_solve" / "solutions")
                       .glob("sol_*.actions.npy"))
        if tapes:
            out.append(Action(
                id=f"dmap_{t}", kind="dmap", needs_emulator=True,
                timeout_h=1.0,
                cmd=["-m", "src.utils.wavefront_reward", "--solutions"]
                    + [str(x.relative_to(repo)) for x in tapes]
                    + ["--root-state", entrance, "--profile", prof,
                       "--rom", rom, "--out",
                       f"checkpoints/wavefront/mario_{t}_dmap.pkl"],
                gate="D_start below the tape length (1-3 read 455 against "
                     "540; 1-4 read 456 against 490).",
                meta={"level": level}))

    anchor_ck = f"checkpoints/bc_{t}/anchor_h256/vanilla_ppo_iter_00000.pt"
    if not (repo / anchor_ck).exists():
        demo_dir = repo / f"checkpoints/bc_{t}/demos"
        if not any(demo_dir.glob("*.npz")):
            for tp in sorted((repo / "runs" / f"ge_{t}_solve" / "solutions")
                             .glob("sol_*.actions.npy")):
                idx = tp.name.split("_")[1].split(".")[0]
                out.append(Action(
                    id=f"demo_{t}_{idx}", kind="demo", needs_emulator=True,
                    timeout_h=0.5,
                    cmd=["scripts/replay_to_demos.py",
                         "--start-state", entrance,
                         "--actions", str(tp.relative_to(repo)),
                         "--profile", prof, "--root-id", "entrance",
                         "--out",
                         f"checkpoints/bc_{t}/demos/demos_{t}_sol_{idx}.npz"],
                    gate="Expect some tapes to fail; quarantine those and "
                         "train on the rest.",
                    meta={"level": level}))
        else:
            out.append(Action(
                id=f"anchor_{t}", kind="anchor", needs_emulator=False,
                timeout_h=1.0,
                cmd=["scripts/bc_distill.py", "--demos",
                     f"checkpoints/bc_{t}/demos/demos_{t}_sol_*.npz",
                     "--profile", prof, "--out",
                     f"checkpoints/bc_{t}/anchor_h256",
                     "--hidden-dim", "256", "--trunk-dim", "64",
                     "--epochs", "120", "--lr", "1e-3", "--seed", "0"],
                gate="An anchor the profile's net shape-infers, per the "
                     "campaign dry-run's KL-anchor check.",
                meta={"level": level}))
    return out


def tag_of(level: str) -> str:
    return level.replace("-", "_")


# ---------------------------------------------------------------- plan

def plan(state: dict, repo: Path = REPO,
         emulator_only: Optional[bool] = None) -> Optional[Action]:
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
        if a is None or (a.id in done and not a.recurring):
            return None
        if a.needs_quiet:
            ok_q, why_q = machine_quiet(state)
            if not ok_q:
                skipped.append(f"{a.id}: deferred, machine not quiet ({why_q})")
                return None
        if a.recurring:
            last = float(state.get("last_run", {}).get(a.id, 0.0))
            if last and (time.time() - last) / 3600.0 < RECURRING_COOLDOWN_H:
                return None
        if emulator_only is False and a.needs_emulator:
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
            if not honest_eval_current(level, repo):
                for seed in (7, 101):
                    candidates.append(honest_eval_action(level, seed, repo))

    # 2. The synthesis's Phase-1 gate: cheap, and everything the three
    #    converged DR rounds recommend sits downstream of it.
    candidates.append(Action(
        id="hazard_phase1", kind="benchmark", needs_emulator=True,
        timeout_h=2.0,
        cmd=["scripts/hazard_collect.py", "--benchmark",
             "--profile", "configs/mario_1_2_online_v2.yaml",
             "--rom", "roms/Super Mario Bros. (World).nes",
             # Restore points: the minted 1-2 rungs are real saved states
             # spread along a solved level, which is what micro-forking
             # wants — varied, reachable, and already provenance-checked.
             "--states", "checkpoints/online_1_2/restart_states"],
        done_marker=None,
        needs_quiet=True,
        gate=">=1000 worker-ticks/s and <1h projected for 100k labels; "
             "below that Phase 1 is KILLED per the synthesis and the "
             "fallback is observational deaths from rollout logs."))

    # 2b. Phase 1 proper, then Phase 2. Each is offered only once the
    #     step before it has left its artifact on disk, so the synthesis's
    #     gate order is enforced by construction rather than by intent.
    haz_npz = "runs/engine/hazard_labels.npz"
    bench = repo / "runs/engine/logs/hazard_phase1.log"
    if bench.exists() and "GATE: PASS" in bench.read_text()[-4000:] \
            and not (repo / haz_npz).exists():
        candidates.append(Action(
            id="hazard_collect_full", kind="collect", needs_emulator=True,
            timeout_h=4.0,
            # A TAPE, not the rung directory. The 1-2 ladder holds six
            # restore points, so 64 forks each caps at 384 labels — three
            # orders of magnitude short of the gate, and far too few to
            # fit a survival model without memorising it. Replaying the
            # 871-step tape yields a restore point per step, which is
            # what "100,000 cleanly labelled transitions" assumes.
            cmd=["scripts/hazard_collect.py",
                 "--profile", "configs/mario_1_2_online_v2.yaml",
                 "--rom", "roms/Super Mario Bros. (World).nes",
                 "--states", "runs/ge_1_2_div_s1/solutions/sol_000.actions.npy",
                 "--root-state",
                 "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state",
                 "--forks-per-state", "120", "--out", haz_npz],
            done_marker=haz_npz,
            gate="100,000 cleanly labelled transitions; a worker alive at "
                 "horizon end is CENSORED, never a survivor-labelled zero."))
    if (repo / haz_npz).exists() and not (
            repo / "runs/engine/hazard/hazard_report.json").exists():
        candidates.append(Action(
            id="hazard_phase2_train", kind="train", needs_emulator=False,
            timeout_h=3.0,
            # --out is a DIRECTORY: train_hazard writes hazard_model.pt
            # and hazard_report.json inside it. Naming it "...pt" made the
            # marker path wrong, so a run that PASSED at C-index 0.917 was
            # recorded as failed and counted toward the circuit breaker.
            cmd=["scripts/train_hazard.py", "--data", haz_npz,
                 "--out", "runs/engine/hazard", "--gate", "0.85"],
            done_marker="runs/engine/hazard/hazard_report.json",
            gate="Uno IPCW C-index >= 0.85 on a held-out trajectory set, "
                 "split by source state. Below that the synthesis says the "
                 "13x13 tile observation lacks the resolution to see "
                 "threats: do not integrate."))

    # 2c. SHELF DISPOSITIONS (PROCESS_AUDIT_2026-08-23): signals we
    #     nearly left on the table, each owed its >=100-episode answer.
    #     The joint-policy follow-up is the big one — the interference
    #     falsifier's joint checkpoint scored 0.52 on 1-1 vs the
    #     specialist's 0.43 and the positive signal was filed under the
    #     experiment's negative headline. Argv mirrors the falsifier's
    #     own _eval_game_command settings so the numbers are comparable
    #     to the originals.
    _shelf = [
        ("shelf_joint_1_1", "runs/interference/joint.pt",
         "mario_1_1_backward", "configs/mario_1_1_backward.yaml",
         "runs/live_show/smb_4_4_micro/entrance_start.state"),
        ("shelf_joint_1_2", "runs/interference/joint.pt",
         "mario_1_2_consol2", "configs/mario_1_2_consol2.yaml",
         "checkpoints/super_bros_placeholder"),
        ("shelf_1_4_endpoint", None,
         "mario_1_4_online_v1", "configs/mario_1_4_online_v1.yaml",
         None),
    ]
    for sid, ck, game, prof_rel, start in _shelf:
        prof_p = repo / prof_rel
        if not prof_p.exists():
            continue
        try:
            import yaml as _yaml
            _prof = _yaml.safe_load(prof_p.read_text()) or {}
        except Exception:
            continue
        rom = _prof.get("rom_path")
        if start is None or "placeholder" in str(start):
            start = _prof.get("start_state_path")
        if ck is None:
            cks = sorted((repo / "checkpoints" / game).glob(
                "vanilla_ppo_iter_*.pt"))
            ck = str(cks[-1].relative_to(repo)) if cks else None
        if not (rom and start and ck):
            continue
        for seed in (7, 101):
            candidates.append(Action(
                id=f"{sid}_seed{seed}", kind="shelf_eval",
                needs_emulator=True, timeout_h=4.0,
                cmd=["scripts/eval_game.py", "--game", game,
                     "--profile", prof_rel, "--rom", str(rom),
                     "--checkpoint", str(ck),
                     "--episodes", "50", "--max-steps", "3000",
                     "--sequential", "--level-clear",
                     "--start-state", str(start),
                     "--eval-seed", str(seed),
                     "--sticky-prob", "0.25", "--start-jitter", "16",
                     "--eval-workers", "5", "--eval-rng", "per-episode"],
                gate="Honest protocol; the record lands in the checkpoint "
                     "dir's eval.jsonl. Joint gate: pooled 1-1 rate "
                     "meaningfully above the specialist's 0.43 revives "
                     "the shared-substrate line; at-or-below closes the "
                     "signal with receipts. 1-4 endpoint gate: pooled "
                     "rate vs the banked 0.51 decides whether "
                     "re-consolidation genuinely raised the level.",
                meta={"shelf": sid, "seed": seed}))

    # 3. Verify the banked EXHIBITION corpus once.
    candidates.append(Action(
        id="replay_sweep_full", kind="sweep", needs_emulator=True,
        timeout_h=8.0,
        cmd=["scripts/replay_sweep.py", "--glob",
             "runs/**/solutions/*.json", "--out",
             "runs/engine/replay_sweep_full.json"],
        done_marker="runs/engine/replay_sweep_full.json",
        gate="Zero genuine replay FAILUREs. ERRORs are unverifiable, "
             "reported separately, never counted as passes."))

    # 3b. Resume campaigns interrupted without recording an end, so a
    #     reboot or stray kill self-heals rather than stranding a run.
    #
    #     Split by whether the level has already been scored. "Resume
    #     interrupted work before starting new work" is right in general
    #     and wrong here: an ALREADY-SCORED level's interrupted campaign
    #     is worth less than an unscored level's first one. 1-4, banked at
    #     51%, held the machine for four hours on exactly this rule while
    #     the hazard Phase-1 gate and 2-2's pipeline waited. Consolidation
    #     on a banked level is a real experiment — it is what produced
    #     1-4's rate — but it can also collapse one, as 2-1 just did
    #     (median 2596 -> 1171, ending 0/100), so it is speculative work
    #     and ranks accordingly.
    deferred_resumes: list[Action] = []
    for level in SMB_LEVELS:
        t = tag_of(level)
        rd = repo / "runs" / f"online_{t}"
        phase = campaign_interrupted(rd, repo)
        if phase is None or not (repo / f"configs/campaign_{t}.yaml").exists():
            continue
        act = Action(
            id=f"resume_{t}_phase{phase}", kind="campaign",
            needs_emulator=True, timeout_h=14.0,
            cmd=["scripts/run_online_campaign.py", "--campaign-config",
                 f"configs/campaign_{t}.yaml", "--start-phase", str(phase)],
            gate="Resumes at the phase it was interrupted in; earned gates "
                 "are not re-litigated, which is why the phase index is "
                 "carried rather than restarting at 0.",
            meta={"level": level, "resumed_from_phase": phase})
        if honest_eval_done(level, repo):
            deferred_resumes.append(act)     # already scored: speculative
        else:
            candidates.append(act)           # unscored: this is the result

    # 4. Extend the pipeline: run the campaign for any level that already
    #    has a validated config but no run. Onboarding a level that lacks
    #    a config needs a human (see the preflight report) — the engine
    #    will not fabricate one.
    for level in SMB_LEVELS:
        tag = level.replace("-", "_")
        if honest_eval_done(level, repo) or level_has_campaign(level, repo):
            continue
        candidates.extend(onboarding_actions(level, repo))
        # A config alone is not readiness. Generating one is cheap and
        # happens early in the W-chain, so a level can have configs while
        # still lacking its ladder, dmap and BC anchor. Launching a
        # campaign there aborts in preflight and burns an attempt for
        # nothing, so the campaign is offered only once the artifacts it
        # actually consumes exist.
        t = tag_of(level)
        ready = all((repo / x).exists() for x in (
            f"checkpoints/online_{t}/restart_states/index.json",
            f"checkpoints/wavefront/mario_{t}_dmap.pkl",
            f"checkpoints/bc_{t}/anchor_h256/vanilla_ppo_iter_00000.pt",
        ))
        if level_has_config(level, repo) and ready:
            candidates.append(Action(
                id=f"campaign_{tag}", kind="campaign", needs_emulator=True,
                timeout_h=14.0,
                cmd=["scripts/run_online_campaign.py", "--campaign-config",
                     f"configs/campaign_{tag}.yaml"],
                gate="Phase gates as pre-registered in the campaign config; "
                     "the honest eval that follows is the real claim.",
                meta={"level": level}))

    # Speculative re-consolidation of already-scored levels: after every
    # unscored result and every pipeline step, before maintenance.
    candidates.extend(deferred_resumes)

    # Maintenance goes LAST. It was first, which meant a recurring suite
    # check outranked 2-1's honest eval the moment its campaign completed
    # — the engine choosing housekeeping over the result it exists to
    # produce.
    # Token-bound and always available: keep the suite honest. A red test
    # sat unnoticed in `make test` for three days because nothing ran it
    # to completion — the same shape as HEAD not compiling for 40 hours.
    # Cheap, needs no emulator, and runs beside a campaign.
    candidates.append(Action(
        id="suite_check", kind="suite", needs_emulator=False, timeout_h=1.0,
        cmd=["scripts/run_suite_check.py", "--out",
             "runs/engine/suite_check.json", "--timeout", "5400"],
        recurring=True,
        done_marker="runs/engine/suite_check.json",
        gate="Records pass/fail counts and the failing node ids. It never "
             "edits a test: a red suite is a finding for the human, and "
             "weakening a test to green it is mission failure."))


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


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lane_of(action_or_record: Any) -> str:
    """Which slot this occupies: 'emulator' or 'token'.

    A record written before lanes existed carries no `needs_emulator`
    key, and the safe reading of a missing value is EMULATOR. The live
    example proved why: the record for a running 1-4 campaign lacked the
    key, so a permissive default filed a campaign in the token lane and
    left the engine believing the machine was free. `emulator_busy` would
    still have refused the second launch, but bookkeeping that disagrees
    with reality is how the next bug hides.
    """
    if isinstance(action_or_record, Action):
        return "emulator" if action_or_record.needs_emulator else "token"
    needs = action_or_record.get("needs_emulator")
    return "token" if needs is False else "emulator"


def running_slots(state: dict) -> dict:
    """state['running'] as {lane: record}, migrating the old single form.

    Originally one global slot, which fixed a real bug — fifteen suite
    checks launched with four running at once — and introduced another:
    the engine then did nothing at all for five and a half hours while a
    campaign held the machine, because a single slot cannot express "one
    emulator job AND one token-bound job". Two lanes keep the physics
    budget (one emulator job, always) without idling the rest.
    """
    running = state.get("running")
    if not running:
        return {}
    records = ([running] if "id" in running          # legacy single form
               else [v for v in running.values() if v])
    # The lane is RE-DERIVED from each record, never trusted from the
    # stored key. A record misfiled once would otherwise stay misfiled
    # forever: the first tick after lanes shipped wrote a running campaign
    # under "token", and reading the key back kept it there across every
    # subsequent tick.
    return {lane_of(r): r for r in records}


def reap(state: dict, repo: Path = REPO) -> Optional[dict]:
    """Settle the previously launched action. Returns a record, or None.

    Nothing did this before, which had two consequences worth naming.
    Actions were never marked completed, so any action without a
    filesystem-derived done-check re-ran until the attempt cap and was
    then abandoned. And `consecutive_failures` was initialised and read
    but never incremented, so the circuit breaker could not fire — the
    engine had a brake with no linkage to the pedal.

    Success is judged by the action's `done_marker`, not by the process
    having ended: an exit is not an outcome. Where an action declares no
    marker, an ended process is recorded as `finished` and treated as
    neither success nor failure, because claiming either would be an
    inference the engine cannot support.
    """
    slots = running_slots(state)
    if not slots:
        return None
    records = []
    for lane in list(slots):
        rec = _reap_one(state, slots, lane, repo)
        if rec:
            records.append(rec)
    state["running"] = {k: v for k, v in slots.items() if v}
    return records[0] if records else None


def _reap_one(state: dict, slots: dict, lane: str,
              repo: Path) -> Optional[dict]:
    running = slots.get(lane)
    if not running:
        return None
    pid = int(running.get("pid", -1))
    started = float(running.get("started", 0))
    marker = running.get("done_marker")
    age_h = (time.time() - started) / 3600.0

    if pid_alive(pid):
        if age_h <= float(running.get("timeout_h", DEFAULT_ACTION_TIMEOUT_H)):
            return None
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        slots[lane] = None
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        state.setdefault("completed", {})[running["id"]] = {
            "status": "timeout", "hours": round(age_h, 2)}
        rec = {"type": "reap", "action": running["id"], "outcome": "timeout",
               "hours": round(age_h, 2)}
        journal(rec)
        return rec

    if marker:
        ok = (repo / marker).exists()
        outcome = "succeeded" if ok else "failed"
    else:
        ok, outcome = None, "finished"

    slots[lane] = None
    if running.get("needs_emulator"):
        state["last_heavy_finish"] = time.time()
    if ok is True:
        state["consecutive_failures"] = 0
    elif ok is False:
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    if not running.get("recurring"):
        state.setdefault("completed", {})[running["id"]] = {
            "status": outcome, "marker": marker}
    rec = {"type": "reap", "action": running["id"], "outcome": outcome,
           "marker": marker}
    journal(rec)
    return rec


def load_average() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("inf")          # unknown is not quiet


def machine_quiet(state: dict) -> tuple[bool, str]:
    """Is the machine quiet enough to trust a throughput measurement?

    The hazard Phase-1 gate is why this exists. It ran twelve minutes
    after a thirteen-hour campaign ended and measured 100.1 steps/s
    against a 1,000 steps/s kill threshold — a KILL, on the research
    direction three independent Deep Research rounds had converged on.
    Re-run on a settled machine the same command measured 2,318.6
    steps/s and PASSED with 43.8 minutes projected against a 60-minute
    budget: a 23x difference attributable entirely to when it ran.

    The brief already required that benches run only on a quiet machine.
    Nothing enforced it, so the engine cheerfully benchmarked a hot one
    and recorded a false negative as a pre-registered result.
    """
    la = load_average()
    if la > QUIET_LOAD_MAX:
        return False, f"load {la:.2f} > {QUIET_LOAD_MAX}"
    last_heavy = float(state.get("last_heavy_finish", 0.0))
    if last_heavy:
        waited = time.time() - last_heavy
        if waited < QUIET_SETTLE_S:
            return False, (f"only {waited/60:.1f} min since the last heavy "
                           f"job; need {QUIET_SETTLE_S/60:.0f}")
    return True, f"load {la:.2f}, settled"


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
    reap(state, repo)

    # An action reap() left in place is still executing. Launching anyway
    # is how fifteen suite checks were started in one session, four of
    # them running at once: reap() correctly returns None for a live pid,
    # tick() then planned and launched regardless. One action at a time.
    slots = running_slots(state)
    busy = emulator_busy()

    blocked = guard_reasons(state, repo)
    if blocked:
        rec = {"type": "tick", "decision": "blocked", "reason": "; ".join(blocked)}
        journal(rec)
        return rec

    # An action is launchable only if its lane is free AND, when it needs
    # the emulator, nothing else already holds the machine. Both
    # conditions are checked on every path: an earlier version applied the
    # busy-fallback without re-checking the lane, which re-planned
    # token-bound work straight into an occupied token slot.
    def launchable(a: Optional[Action]) -> bool:
        if a is None:
            return False
        if lane_of(a) in slots:
            return False
        return not (a.needs_emulator and busy is not None)

    action = plan(state, repo)
    if not launchable(action):
        action = plan(state, repo, emulator_only=False)
        if not launchable(action):
            action = None

    # "nothing to do" and "something else holds the machine" are
    # different states and are not merged: only the first means the plan
    # is genuinely exhausted.
    if action is None and (slots or busy is not None):
        reasons = [f"{r['id']} running (pid {r['pid']})"
                   for r in slots.values() if r]
        if busy is not None:
            reasons.append(f"emulator held externally (pid {busy}), "
                           f"no token-bound action available")
        rec = {"type": "tick", "decision": "wait", "reason": "; ".join(reasons)}
        journal(rec)
        return rec

    if action is None:
        rec = {"type": "tick", "decision": "idle",
               "reason": "nothing left in the computed plan"}
        journal(rec)
        return rec

    attempts = state.setdefault("attempts", {})
    n = attempts.get(action.id, 0)
    if n >= MAX_ATTEMPTS_PER_ACTION and not action.recurring:
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
    state.setdefault("last_run", {})[action.id] = time.time()
    slots[lane_of(action)] = {"id": action.id, "pid": pid,
                              "started": time.time(),
                              "timeout_h": action.timeout_h,
                              "done_marker": action.done_marker,
                              "recurring": action.recurring,
                              "needs_emulator": action.needs_emulator}
    state["running"] = slots
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
