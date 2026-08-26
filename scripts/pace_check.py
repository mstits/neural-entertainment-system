"""Positive control for the author's own pace: prove the human didn't
run hot without anyone noticing.

experiment_preflight.py refuses to spend a training budget on a run
whose actor never moved — it asserts LEARNING IS ALIVE from evidence,
not from config. There is no equivalent instrument on the other side
of the keyboard: nothing reads back the one log that already records
the author's working hours, commit by commit, with a timestamp on
every entry. git log IS the depth_memo.jsonl of the human loop; this
tool just reads it.

A day whose commits span more than MAX_SPAN_HOURS, or whose commit
count exceeds MAX_COMMITS_PER_DAY, gets flagged in the printed report.
That is the entire contract. This tool does not moralize, does not
block a commit, does not gate a workflow, and does not know what
"too much" means beyond the two constants below — it reports hours
and counts, the way a fitness tracker reports steps, not a doctor's
note. Adjust the constants freely; they are not calibrated to
anything but the shape of one Growth-and-Underinvestment week.

    .venv/bin/python scripts/pace_check.py
    .venv/bin/python scripts/pace_check.py --days 21 --repo /path/to/repo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent

# Thresholds — trivially adjustable, and load-bearing nowhere else.
# Nothing downstream of these two numbers gates or blocks anything;
# they only decide which rows in the printed report carry a flag.
MAX_SPAN_HOURS = 12.0
MAX_COMMITS_PER_DAY = 25
TRAILING_DAYS = 14

# git log invocation this tool parses. Author date (not commit date,
# which rebases/amends can silently rewrite) in a fixed-offset ISO
# form so `datetime.fromisoformat` round-trips it without a timezone
# database.
GIT_LOG_PRETTY = "%H|%ad"
GIT_LOG_DATE_FORMAT = "iso-strict"


@dataclass(frozen=True)
class Commit:
    sha: str
    when: datetime  # tz-aware; author's local time as git recorded it


@dataclass(frozen=True)
class DayStats:
    day: date
    count: int
    first: Optional[datetime]
    last: Optional[datetime]
    span_hours: float
    span_flag: bool
    count_flag: bool

    @property
    def flagged(self) -> bool:
        return self.span_flag or self.count_flag


def parse_git_log(lines: Iterable[str]) -> list[Commit]:
    """Parse `git log --pretty=format:%H|%ad --date=iso-strict` output
    into Commit records. PURE — no filesystem, no subprocess. This is
    the function the test suite exercises directly against a
    synthesized fixture, so the real repo's ever-changing history
    never has to be depended on for a pass/fail."""
    commits = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        sha, sep, when_s = line.partition("|")
        if not sep or not when_s:
            continue  # tolerate a stray/partial line rather than crash
        try:
            when = datetime.fromisoformat(when_s)
        except ValueError:
            continue
        commits.append(Commit(sha=sha, when=when))
    return commits


def fetch_git_log(repo: Path, since_days: int) -> list[str]:
    """Impure: shells out to `git log` in `repo`. Kept separate from
    parse_git_log so nothing in the test suite touches a real repo or
    a real clock."""
    cmd = [
        "git", "-C", str(repo), "log",
        f"--since={since_days + 1} days ago",
        f"--pretty=format:{GIT_LOG_PRETTY}",
        f"--date={GIT_LOG_DATE_FORMAT}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def group_by_day(commits: Iterable[Commit]) -> dict[date, list[Commit]]:
    """Group commits by the calendar date of their author-local
    timestamp. PURE."""
    grouped: dict[date, list[Commit]] = {}
    for c in commits:
        grouped.setdefault(c.when.date(), []).append(c)
    return grouped


def compute_day_stats(day: date, commits_on_day: list[Commit]) -> DayStats:
    """Reduce one day's commits to first/last/span/count, and apply
    the two thresholds. PURE — the same "assert from evidence" shape
    as experiment_preflight.assess_learning."""
    if not commits_on_day:
        return DayStats(day=day, count=0, first=None, last=None,
                        span_hours=0.0, span_flag=False, count_flag=False)
    times = sorted(c.when for c in commits_on_day)
    first, last = times[0], times[-1]
    span_hours = (last - first).total_seconds() / 3600.0
    return DayStats(
        day=day,
        count=len(commits_on_day),
        first=first,
        last=last,
        span_hours=span_hours,
        span_flag=span_hours > MAX_SPAN_HOURS,
        count_flag=len(commits_on_day) > MAX_COMMITS_PER_DAY,
    )


def trailing_day_stats(
    commits: Iterable[Commit], as_of: date, trailing_days: int = TRAILING_DAYS,
) -> list[DayStats]:
    """One DayStats per calendar day in the closed range
    [as_of - trailing_days + 1, as_of], oldest first. Days with no
    commits still appear, with count=0 — a quiet day is data too.
    PURE (as_of is a parameter, never `date.today()` internally, so
    this is deterministic for any fixture)."""
    grouped = group_by_day(commits)
    days = [as_of - timedelta(days=i) for i in range(trailing_days - 1, -1, -1)]
    return [compute_day_stats(d, grouped.get(d, [])) for d in days]


def format_report(stats: list[DayStats]) -> str:
    """Render the trailing-window table. Purely informational: hours
    and counts, no verdict, no "should", no red/green judgment beyond
    the literal word FLAG on the rows that cross a threshold."""
    header = f"{'date':<12}{'commits':>8}  {'first':<9}{'last':<9}{'span_h':>7}  flag"
    lines = [header, "-" * len(header)]
    for s in stats:
        first_s = s.first.strftime("%H:%M") if s.first else "-"
        last_s = s.last.strftime("%H:%M") if s.last else "-"
        flag_bits = []
        if s.span_flag:
            flag_bits.append(f"span>{MAX_SPAN_HOURS:g}h")
        if s.count_flag:
            flag_bits.append(f"count>{MAX_COMMITS_PER_DAY}")
        flag_s = ", ".join(flag_bits)
        lines.append(
            f"{s.day.isoformat():<12}{s.count:>8}  {first_s:<9}{last_s:<9}"
            f"{s.span_hours:>7.1f}  {flag_s}"
        )

    total_commits = sum(s.count for s in stats)
    active_days = sum(1 for s in stats if s.count > 0)
    flagged_days = [s for s in stats if s.flagged]
    avg_per_active_day = total_commits / active_days if active_days else 0.0

    lines.append("")
    lines.append(
        f"{total_commits} commits over {len(stats)} days "
        f"({active_days} active, {avg_per_active_day:.1f}/active day). "
        f"{len(flagged_days)} day(s) flagged "
        f"(span > {MAX_SPAN_HOURS:g}h or count > {MAX_COMMITS_PER_DAY})."
    )
    if flagged_days:
        lines.append("Flagged: " + ", ".join(s.day.isoformat() for s in flagged_days))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Report commit-timestamp span and count per day, "
                    "same instinct as experiment_preflight.py pointed "
                    "at the author instead of the model. Informational "
                    "only — never exits non-zero over a flagged day.",
    )
    ap.add_argument("--repo", default=str(REPO), help="path to the git repo")
    ap.add_argument("--days", type=int, default=TRAILING_DAYS,
                    help="trailing window size in calendar days")
    args = ap.parse_args(argv)

    raw_lines = fetch_git_log(Path(args.repo), since_days=args.days)
    commits = parse_git_log(raw_lines)
    stats = trailing_day_stats(commits, as_of=date.today(), trailing_days=args.days)
    print(format_report(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
