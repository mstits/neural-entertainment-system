"""The engine board, re-derived from receipts on every run.

WHY THIS EXISTS. The mission brief and the empirical protocol used to
live in a prompt that had to be re-pasted to stay in effect, and the
board's state used to be reconstructed from whoever's memory was
handy. Both failed in measurable ways in the week of 2026-08-11..18:

  * A campaign was declared dead from a missing OS process while its own
    `campaign.jsonl` said `campaign_complete`. That cost a wrong ledger
    entry, a retraction of a correct conclusion and a 30M-step re-run.
  * `configs/mario_2_1_online_v1.yaml` trained on 1-3's restart ladder
    for 20.1M steps because the profile was cloned and one path never
    repointed. Every instrument read plausibly.
  * The same defect was then found in `configs/mario_1_4_online_v1.yaml`
    — the level with the HIGHEST banked honest rate — by pointing the
    new preflight at every config instead of the one being worked on.

So this tool has one rule: READ, NEVER INFER. Every line names where its
value came from, and anything not backed by a file on disk is printed as
unknown rather than guessed. `--check` turns the rigor section into an
exit code so `make` can gate on it.

The rigor checks are not generic hygiene. Each one is a falsifier for a
specific failure this project has actually suffered, and each names it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MISSION = Path.home() / "nes-mission"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    failure_class: str  # the historical failure this guards against


@dataclass
class Board:
    running: list[str] = field(default_factory=list)
    campaigns: list[str] = field(default_factory=list)
    banked: list[str] = field(default_factory=list)
    oracle: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)


# ---- running processes, with detachment status ------------------------

def _pgid(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip()) if out.stdout.strip() else None
    except (ValueError, subprocess.SubprocessError):
        return None


def find_long_jobs() -> list[tuple[int, str]]:
    """PIDs of our emulator-heavy processes, by command match.

    Deliberately narrow: matching broadly picks up unrelated Python.
    """
    pats = ("run_online_campaign", "train_game", "soak_harness",
            "go_explore_solve", "eval_game")
    jobs: list[tuple[int, str]] = []
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, timeout=15)
    except subprocess.SubprocessError:
        return jobs
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        if not pid_s.isdigit():
            continue
        if any(p in cmd for p in pats):
            jobs.append((int(pid_s), cmd))
    return jobs


def report_running(board: Board) -> None:
    jobs = find_long_jobs()
    if not jobs:
        board.running.append("no emulator-heavy process running (ps)")
        return
    for pid, cmd in jobs:
        pg = _pgid(pid)
        detached = (pg == pid)
        short = cmd.split("/")[-1][:90]
        board.running.append(
            f"pid {pid} pgid {pg} "
            f"{'DETACHED' if detached else 'NOT DETACHED (a stray Ctrl-C kills it)'}"
            f"  {short}")


# ---- campaigns, read from their own logs ------------------------------

def campaign_logs() -> list[Path]:
    """Live campaign logs only.

    Quarantined runs keep their logs on purpose (the record is never
    deleted), but listing them beside live ones invites exactly the
    confusion the quarantine exists to prevent, so they are filtered here
    and counted by check_quarantine instead.
    """
    return sorted(p for p in REPO.glob("runs/**/campaign.jsonl")
                  if "INVALID" not in str(p))


def summarize_campaign(path: Path) -> str:
    """Terminal state straight out of the log — never from process state.

    The 2026-08-17 error was inferring death from a missing process. The
    log is authoritative: it records `campaign_complete`, `abort` and
    `kill` explicitly.
    """
    rows = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        return f"{path.parent.name}: unreadable ({e})"
    if not rows:
        return f"{path.parent.name}: empty log"

    # A resume appends a fresh `campaign_start`, so any terminal row BEFORE
    # it belongs to a previous attempt and is superseded. Without this the
    # board reports a live, progressing run as aborted — which is exactly
    # the "state inferred from a stale artifact" failure it exists to
    # prevent, committed by the tool itself.
    last_start = max((i for i, r in enumerate(rows)
                      if r.get("type") == "campaign_start"), default=-1)
    live = rows[last_start + 1:] if last_start >= 0 else rows

    terminal = "in progress (no terminal row)"
    for r in reversed(live):
        if r.get("type") in ("campaign_complete", "abort", "kill"):
            terminal = f"{r['type']}"
            if r.get("reason"):
                terminal += f" — {str(r['reason'])[:60]}"
            break
    if last_start > 0 and terminal.startswith("in progress"):
        terminal += " (resumed)"

    phase = next((r.get("name") for r in reversed(live)
                  if r.get("type") == "phase_start"), "?")
    gate = next((r for r in reversed(rows)
                 if r.get("type") == "gate_probe"), None)
    probe = next((r for r in reversed(rows) if r.get("type") == "probe"), None)

    bits = [f"phase={phase}", f"state={terminal}"]
    if gate:
        bits.append(f"gate(rung,det) med={gate.get('median_max_x')} "
                    f"strict={gate.get('clear_rate_strict')}")
    if probe:
        bits.append(f"honest(entrance) @{probe.get('env_steps',0)/1e6:.1f}M "
                    f"med={probe.get('median_max_x')} "
                    f"strict={probe.get('clear_rate_strict')}")
    return f"{path.parent.name}: " + "  ".join(str(b) for b in bits)


def report_campaigns(board: Board) -> None:
    logs = campaign_logs()
    if not logs:
        board.campaigns.append("no campaign logs found")
    for p in logs:
        board.campaigns.append(summarize_campaign(p))


# ---- banked claims, parsed from CLAIMS.md -----------------------------

# CLAIMS.md maintains a canonical roll-up line ("now holds four levels:
# 1-1 43%, 1-2 38%, ..."). Parsing that is far more robust than scanning
# backwards from each **n/100** for a nearby level label, which mislabels
# whenever the prose mentions another level in between — the first version
# of this function reported 1-2's rate as 1-3's for exactly that reason.
ROLLUP_RE = re.compile(r"holds\s+\w+\s+levels?:\s*((?:\d-\d\s+\d+%[,\s]*)+)")
PAIR_RE = re.compile(r"(\d-\d)\s+(\d+)%")
RATE_RE = re.compile(r"\*\*(\d+)/(\d+)")


def report_banked(board: Board) -> None:
    claims = REPO / "CLAIMS.md"
    if not claims.exists():
        board.banked.append("CLAIMS.md missing")
        return
    text = claims.read_text()
    rollup = ROLLUP_RE.search(text)
    if rollup:
        pairs = PAIR_RE.findall(rollup.group(1))
        board.banked.append(
            "roll-up (CLAIMS.md): "
            + ", ".join(f"{lvl} {pct}%" for lvl, pct in sorted(pairs)))
    # Cross-check the roll-up against the individual **n/100** claims.
    # A disagreement means the summary line drifted from the entries, so
    # report BOTH rather than trusting either.
    rates = sorted({f"{m.group(1)}/{m.group(2)}"
                    for m in RATE_RE.finditer(text)
                    if int(m.group(2)) >= 100})
    board.banked.append(f"individual >=100-episode entries: {', '.join(rates)}")
    if rollup:
        roll_pcts = {int(pct) for _, pct in PAIR_RE.findall(rollup.group(1))}
        entry_pcts = {round(100 * int(r.split("/")[0]) / int(r.split("/")[1]))
                      for r in rates}
        missing = roll_pcts - entry_pcts
        if missing:
            board.banked.append(
                f"MISMATCH: roll-up cites {sorted(missing)}% with no matching "
                f"entry — the summary line and the entries disagree")
    else:
        board.banked.append("no roll-up line found; entries above are the only source")


# ---- oracle, read from the mission scoreboard -------------------------

def report_oracle(board: Board) -> None:
    sb = MISSION / "SCOREBOARD.md"
    if not sb.exists():
        board.oracle.append(f"SCOREBOARD.md not found at {sb}")
        return
    text = sb.read_text()
    m = re.search(r"- State: \*\*(.+?)\*\*", text, re.S)
    board.oracle.append("state: " + (m.group(1).replace("\n", " ")
                                     if m else "unparsed"))
    for pat, label in ((r"(\d+) segments", "segments"),
                       (r"interventions (\d+)", "interventions"),
                       (r"\*\*([\d,]+\.?\d*) s = ([\d.]+) h\*\*", "duration")):
        mm = re.search(pat, text)
        if mm:
            board.oracle.append(f"{label}: {' '.join(mm.groups())}")
    # Vacuous-pass honesty: surface it, never let it be forgotten.
    if "VACUOUSLY" in text or "vacuously" in text:
        board.oracle.append(
            "CAVEAT (from SCOREBOARD): a clause passed VACUOUSLY — "
            "read the scoreboard before quoting the oracle")


# ---- rigor checks: one per historical failure -------------------------

def check_ladders() -> Check:
    """Every campaign config's trainer ladder == its gates' ladder.

    Failure class: the 2-1 (20.1M steps voided) and 1-4 (highest banked
    rate) config-clone defects.
    """
    sys.path.insert(0, str(REPO))
    try:
        import scripts.run_online_campaign as m
    except Exception as e:
        return Check("ladder provenance", False, f"import failed: {e!r}",
                     "config-clone / wrong-level ladder")
    bad: list[str] = []
    n = 0
    for cfg in sorted((REPO / "configs").glob("campaign_*.yaml")):
        n += 1
        try:
            m.apply_campaign_config(cfg)
            base = m._load_yaml(REPO / m.CONFIG["base_profile"])
            ok, notes = m.preflight_restart_ladders(base, m.CONFIG)
            if not ok:
                bad.append(f"{cfg.name}: {'; '.join(notes)[:80]}")
        except Exception as e:
            bad.append(f"{cfg.name}: {type(e).__name__}")
    return Check("ladder provenance", not bad,
                 f"{n - len(bad)}/{n} configs agree"
                 + (f" — FAIL: {bad}" if bad else ""),
                 "config-clone / wrong-level ladder")


def check_quarantine() -> Check:
    """Contaminated artifacts stay visibly quarantined, never deleted.

    Failure class: silent reuse of a voided run's checkpoints.
    """
    q = sorted(list(REPO.glob("checkpoints/*INVALID*"))
               + list(REPO.glob("runs/*INVALID*")))
    return Check("quarantine inventory", True,
                 f"{len(q)} quarantined artifact(s): "
                 + (", ".join(p.name for p in q) if q else "none"),
                 "silent reuse of voided artifacts")


def check_so_freshness() -> Check:
    """The loaded nes_core .so is newer than the Rust it was built from.

    Failure class: a committed core fix that the running binary predates
    (the standing DMC/ASM migration gate).
    """
    so = next((p for p in (REPO / ".venv/lib").rglob("nes_core*.so")), None)
    if so is None:
        return Check("nes_core .so freshness", False,
                     "no nes_core .so found in .venv",
                     "stale binary masking a committed fix")
    srcs = list((REPO / "nes_core/src").rglob("*.rs"))
    if not srcs:
        return Check("nes_core .so freshness", False, "no Rust sources found",
                     "stale binary masking a committed fix")
    newest = max(srcs, key=lambda p: p.stat().st_mtime)
    fresh = so.stat().st_mtime >= newest.stat().st_mtime
    return Check("nes_core .so freshness", fresh,
                 f"{so.name} {'newer' if fresh else 'OLDER'} than "
                 f"{newest.relative_to(REPO)}"
                 + ("" if fresh else " — rebuild before trusting core results"),
                 "stale binary masking a committed fix")


def check_running_detached() -> Check:
    """Any long job must lead its own process group.

    Failure class: the 2-1 phase-2 trainer killed at 12.07M by a
    group-wide signal, because nohup does not change process group.
    """
    jobs = find_long_jobs()
    attached = [pid for pid, _ in jobs if _pgid(pid) != pid]
    # A trainer spawned BY a detached controller legitimately shares the
    # controller's group; only a top-level job sharing OUR group is bad.
    ours = os.getpgrp()
    exposed = [pid for pid in attached if _pgid(pid) == ours]
    return Check("long jobs detached", not exposed,
                 f"{len(jobs)} long job(s); {len(exposed)} in this session's "
                 f"group" + (f" {exposed}" if exposed else ""),
                 "group-wide signal killing a detached run")


def check_ledger_receipts() -> Check:
    """Every level in CLAIMS.md's roll-up has a receipted entry of its own.

    Failure class: a number that lives only in a summary sentence. Found
    on this check's first run — 1-1's 43% appeared in two prose roll-ups
    with no **43/100** entry, no checkpoint sha256 and no receipt path,
    while 1-2/1-3/1-4 each had all three. The measurement itself was
    real (runs/interference/interference.jsonl, 100 episodes,
    clear_rate_strict 0.43); the ledger entry was missing.

    This check never edits CLAIMS.md — the ledger is append-only by the
    human. It only refuses to let the gap go unnoticed.
    """
    claims = REPO / "CLAIMS.md"
    if not claims.exists():
        return Check("ledger roll-up receipted", False, "CLAIMS.md missing",
                     "a number that lives only in a summary sentence")
    text = claims.read_text()
    rollup = ROLLUP_RE.search(text)
    if not rollup:
        return Check("ledger roll-up receipted", True,
                     "no roll-up line to cross-check",
                     "a number that lives only in a summary sentence")
    roll = {int(pct) for _, pct in PAIR_RE.findall(rollup.group(1))}
    entries = {round(100 * int(m.group(1)) / int(m.group(2)))
               for m in RATE_RE.finditer(text) if int(m.group(2)) >= 100}
    missing = sorted(roll - entries)
    return Check(
        "ledger roll-up receipted", not missing,
        f"{len(roll) - len(missing)}/{len(roll)} roll-up levels have entries"
        + (f" — {missing}% cited with no **n/100** entry" if missing else ""),
        "a number that lives only in a summary sentence")


RIGOR = (check_ladders, check_quarantine, check_so_freshness,
         check_running_detached, check_ledger_receipts)


def build_board() -> Board:
    board = Board()
    report_running(board)
    report_campaigns(board)
    report_banked(board)
    report_oracle(board)
    for fn in RIGOR:
        try:
            board.checks.append(fn())
        except Exception as e:  # a broken check is a failing check
            board.checks.append(Check(fn.__name__, False, repr(e), "unknown"))
    return board


def render(board: Board) -> str:
    L: list[str] = ["ENGINE BOARD — derived from receipts, not memory", ""]
    for title, rows in (("RUNNING", board.running),
                        ("CAMPAIGNS (state read from campaign.jsonl)",
                         board.campaigns),
                        ("BANKED — LEARNED ledger (CLAIMS.md)", board.banked),
                        ("ORACLE (~/nes-mission/SCOREBOARD.md)", board.oracle)):
        L.append(f"## {title}")
        L.extend(f"  {r}" for r in rows)
        L.append("")
    L.append("## RIGOR — each check guards a failure this project suffered")
    for c in board.checks:
        L.append(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        L.append(f"         guards: {c.failure_class}")
    L.append("")
    failed = [c for c in board.checks if not c.ok]
    L.append(f"VERDICT: {len(board.checks) - len(failed)}/{len(board.checks)} "
             f"rigor checks pass" + (" — ALL GREEN" if not failed else ""))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any rigor check fails (for make)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    board = build_board()
    if args.json:
        print(json.dumps({
            "running": board.running, "campaigns": board.campaigns,
            "banked": board.banked, "oracle": board.oracle,
            "checks": [vars(c) for c in board.checks]}, indent=2))
    else:
        print(render(board))
    if args.check and any(not c.ok for c in board.checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
