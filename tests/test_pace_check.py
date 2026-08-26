"""pace_check — the author-pace positive control, pinned against a
synthesized git-log-shaped fixture.

Deliberately does NOT touch this repo's real history: that changes
with every commit (including the ones this task itself makes), which
would make the test non-deterministic. All fixtures below are
hand-built strings shaped exactly like
`git log --pretty=format:%H|%ad --date=iso-strict` output.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.pace_check import (  # noqa: E402
    MAX_COMMITS_PER_DAY, MAX_SPAN_HOURS, Commit, compute_day_stats,
    format_report, group_by_day, parse_git_log, trailing_day_stats)


def _line(sha: str, iso: str) -> str:
    return f"{sha}|{iso}"


def test_parse_git_log_reads_sha_and_author_date():
    lines = [
        _line("abc123", "2026-08-25T02:02:11-07:00"),
        _line("def456", "2026-08-25T20:42:59-07:00"),
    ]
    commits = parse_git_log(lines)
    assert [c.sha for c in commits] == ["abc123", "def456"]
    assert commits[0].when.hour == 2
    assert commits[1].when.hour == 20


def test_parse_git_log_tolerates_blank_and_malformed_lines():
    lines = [
        "",
        "   ",
        "no-pipe-here",
        _line("abc123", "2026-08-25T02:02:11-07:00"),
        "def456|not-a-date",
    ]
    commits = parse_git_log(lines)
    assert len(commits) == 1
    assert commits[0].sha == "abc123"


def test_group_by_day_buckets_on_author_local_calendar_date():
    commits = parse_git_log([
        _line("a", "2026-08-25T23:50:00-07:00"),
        _line("b", "2026-08-26T00:05:00-07:00"),
    ])
    grouped = group_by_day(commits)
    assert len(grouped[date(2026, 8, 25)]) == 1
    assert len(grouped[date(2026, 8, 26)]) == 1


def test_quiet_day_is_not_flagged():
    stats = compute_day_stats(date(2026, 8, 20), [])
    assert stats.count == 0
    assert not stats.span_flag
    assert not stats.count_flag
    assert not stats.flagged


def test_normal_day_under_both_thresholds_is_not_flagged():
    # 5 commits inside a normal 3-hour afternoon session.
    commits = parse_git_log([
        _line(f"c{i}", f"2026-08-20T{14 + i:02d}:00:00-07:00")
        for i in range(5)
    ])
    stats = compute_day_stats(date(2026, 8, 20), commits)
    assert stats.count == 5
    assert stats.span_hours == 4.0
    assert not stats.span_flag
    assert not stats.count_flag


def test_long_span_day_is_flagged_on_span_not_count():
    # Two commits, but 18 hours apart — the 08-25 "Wednesday Push"
    # shape from the audit: 02:02 to 20:42.
    commits = parse_git_log([
        _line("a", "2026-08-25T02:02:00-07:00"),
        _line("b", "2026-08-25T20:42:00-07:00"),
    ])
    stats = compute_day_stats(date(2026, 8, 25), commits)
    assert stats.count == 2
    assert stats.span_hours > MAX_SPAN_HOURS
    assert stats.span_flag
    assert not stats.count_flag
    assert stats.flagged


def test_high_count_day_is_flagged_on_count_even_within_a_short_span():
    # MAX_COMMITS_PER_DAY + 5 commits packed into one hour.
    n = MAX_COMMITS_PER_DAY + 5
    commits = parse_git_log([
        _line(f"c{i}", f"2026-08-24T10:{i % 60:02d}:00-07:00")
        for i in range(n)
    ])
    stats = compute_day_stats(date(2026, 8, 24), commits)
    assert stats.count == n
    assert stats.count_flag
    assert stats.span_hours <= MAX_SPAN_HOURS
    assert not stats.span_flag
    assert stats.flagged


def test_trailing_day_stats_fills_gaps_and_orders_oldest_first():
    commits = parse_git_log([
        _line("a", "2026-08-24T09:00:00-07:00"),
        _line("b", "2026-08-25T09:00:00-07:00"),
    ])
    stats = trailing_day_stats(commits, as_of=date(2026, 8, 25), trailing_days=3)
    assert [s.day for s in stats] == [
        date(2026, 8, 23), date(2026, 8, 24), date(2026, 8, 25),
    ]
    assert stats[0].count == 0
    assert stats[1].count == 1
    assert stats[2].count == 1


def test_synthesized_four_day_acceleration_matches_audit_shape():
    """Reproduce the exact pattern the audit measured: 21 -> 18 -> 23
    -> 30 commits/day, with the last day spanning 02:02 to 20:42. This
    is a synthesized fixture, not this repo's real log — the point is
    to pin the tool's behavior against a known, fixed shape."""
    day_counts = {
        "2026-08-22": 21,
        "2026-08-23": 18,
        "2026-08-24": 23,
        "2026-08-25": 30,
    }
    lines = []
    i = 0
    for day, n in day_counts.items():
        for k in range(n):
            hour = 2 + (k * 18) // max(n - 1, 1)  # spread 02:xx -> 20:xx
            minute = (k * 7) % 60
            lines.append(_line(f"sha{i}", f"{day}T{hour:02d}:{minute:02d}:00-07:00"))
            i += 1
    commits = parse_git_log(lines)
    stats = trailing_day_stats(commits, as_of=date(2026, 8, 25), trailing_days=4)

    by_day = {s.day.isoformat(): s for s in stats}
    assert by_day["2026-08-22"].count == 21
    assert by_day["2026-08-23"].count == 18
    assert by_day["2026-08-24"].count == 23
    assert by_day["2026-08-25"].count == 30

    # 30 > MAX_COMMITS_PER_DAY (25) — flagged on count.
    assert by_day["2026-08-25"].count_flag
    # Every one of these four days spans most of the day -> span-flagged.
    assert all(s.span_flag for s in stats)
    assert all(s.flagged for s in stats)


def test_format_report_lists_only_flagged_days_in_the_summary_line():
    quiet = compute_day_stats(date(2026, 8, 18), [])
    busy = compute_day_stats(date(2026, 8, 25), parse_git_log([
        _line("a", "2026-08-25T02:02:00-07:00"),
        _line("b", "2026-08-25T20:42:00-07:00"),
    ]))
    report = format_report([quiet, busy])
    assert "2026-08-18" in report
    assert "2026-08-25" in report
    assert "Flagged: 2026-08-25" in report
    assert "2026-08-18" not in report.split("Flagged:")[1]


def test_format_report_is_purely_descriptive_no_verdict_language():
    stats = trailing_day_stats([], as_of=date(2026, 8, 25), trailing_days=3)
    report = format_report(stats)
    lowered = report.lower()
    for banned in ("should", "must", "stop", "too many", "warning:", "error:"):
        assert banned not in lowered


def test_dataclasses_importable_as_public_surface():
    # Guards the module's public surface used by any future caller
    # (e.g. a fold-in to daily_recap.py) — Commit stays a plain,
    # tz-aware (sha, when) pair.
    c = Commit(sha="deadbeef", when=datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc))
    assert c.sha == "deadbeef"
    assert c.when.tzinfo is not None
