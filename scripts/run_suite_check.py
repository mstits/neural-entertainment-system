"""Run the test suite and record the result. Never edits a test.

Exists because a failing test went unnoticed in `make test` for three
days: `tests/test_night2_runner.py::test_dry_run_passes_live` began
failing the moment the consol2 workload it dry-runs actually ran, since
the runner then correctly refuses to seed over 32 newer checkpoints. The
guard is right and the assertion is stale, but nothing was watching.

This is deliberately a RECORDER, not a fixer. Weakening a test to make it
green is mission failure, so the only output is a report.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_summary(text: str) -> dict:
    """Counts and failing node ids from pytest's output."""
    out: dict = {"passed": 0, "failed": 0, "errors": 0, "failing": []}
    for key in ("passed", "failed", "error", "skipped", "xfailed"):
        m = re.search(rf"(\d+) {key}", text)
        if m:
            out[key if key != "error" else "errors"] = int(m.group(1))
    out["failing"] = sorted(set(re.findall(r"^FAILED (\S+)", text,
                                           re.MULTILINE)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/engine/suite_check.json")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(argv)

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--timeout=180"],
        cwd=str(REPO), capture_output=True, text=True, timeout=args.timeout)
    summary = parse_summary(proc.stdout + proc.stderr)
    summary["returncode"] = proc.returncode
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
