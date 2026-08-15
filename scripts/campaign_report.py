"""Turn a campaign attempt's run-dir + checkpoint-dir into a compact
post-mortem (finished/aborted attempt) or live-status (still running)
report.

Usage:
    python scripts/campaign_report.py \\
        --run-dir runs/online_1_2_attempt3 \\
        --ckpt-dir checkpoints/mario_1_2_online_v2_attempt3

    python scripts/campaign_report.py \\
        --run-dir runs/online_1_2_attempt3 \\
        --ckpt-dir checkpoints/mario_1_2_online_v2_attempt3 \\
        --out /tmp/attempt3_report.md

This script only *reads* files under --run-dir / --ckpt-dir. It never
writes into either directory, which makes it safe to point at a
campaign that is still running (e.g. the live `runs/online_1_2` +
`checkpoints/<ckpt>` pair) — pass --out to a path outside those
directories if you want a saved copy.

Inputs it understands, all optional (missing/partial files degrade
gracefully rather than crashing):
  - <run-dir>/campaign.jsonl        one JSON object per campaign event
                                     (campaign_start, phase_start,
                                     gate_pass, probe, phase_complete,
                                     abort)
  - <run-dir>/phase_*.log           per-phase trainer stdout; only the
                                     "[backward] iter N: tau=..." lines
                                     are parsed (backward-curriculum
                                     rung/tau progression)
  - <ckpt-dir>/metrics.jsonl        current-phase training metrics
                                     (one JSON object per iteration)
  - <ckpt-dir>/runs/*/metrics.jsonl rotated metrics.jsonl from earlier
                                     phases of the same attempt (the
                                     trainer archives-then-resets this
                                     file on every relaunch)

Metrics rows are joined to a campaign phase by wall-clock timestamp
(each row's "timestamp" field is bucketed under the last phase_start
event at or before it) rather than by "generation", because generation
counters are per-launch and can overlap across phases (a resumed phase
replays generations already seen by the phase it resumed from).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PLATEAU_WINDOW = 20
PLATEAU_REL_BAND = 0.15  # "flat within 15%" per spec

BACKWARD_LINE_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): tau=(?P<tau>\d+)/(?P<tau_max>\d+) "
    r"\(step (?P<step>\d+) frame (?P<frame>\d+) gx (?P<gx>\d+)\) "
    r"trailing (?P<trail_n>\d+)/(?P<trail_d>\d+)=(?P<trail_rate>[\d.]+) "
    r"\(advance at >=[\d.]+ over \d+\) advances=(?P<advances>\d+) "
    r"\| entrance (?P<ent_n>\d+)/(?P<ent_d>\d+)=(?P<ent_rate>[\d.]+) "
    r"\| truncated (?P<trunc>\d+) \((?P<scored>\d+) scored\) "
    r"\| budget (?P<budget>\d+) steps"
)

PHASE_LOG_RE = re.compile(r"phase_(\d+)\.log$")

KL_FIELD = "kl_anchor_div"
VLOSS_FIELD = "ppo_value_loss"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Missing file -> []. Lines
    that fail to parse are skipped (a warning is printed to stderr) so a
    truncated/partial file (e.g. a campaign that was killed mid-write)
    still yields everything readable."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[campaign_report] skip {path}:{lineno}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_campaign(run_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(run_dir / "campaign.jsonl")


def load_metrics(ckpt_dir: Path) -> list[dict[str, Any]]:
    """Merge the current metrics.jsonl with every rotated
    runs/*/metrics.jsonl, sorted by each row's own timestamp (not by
    file/generation, since generation ranges can overlap across a
    resume boundary)."""
    rows: list[dict[str, Any]] = []
    rows.extend(load_jsonl(ckpt_dir / "metrics.jsonl"))
    runs_dir = ckpt_dir / "runs"
    if runs_dir.is_dir():
        for sub in sorted(runs_dir.iterdir()):
            candidate = sub / "metrics.jsonl"
            if candidate.is_file():
                rows.extend(load_jsonl(candidate))
    rows.sort(key=lambda r: r.get("timestamp", 0.0))
    return rows


def load_backward_lines(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Parse every phase_*.log in run_dir for `[backward] iter ...` lines.
    Returns {phase_idx: [record, ...]} in file order (== iteration order)."""
    out: dict[int, list[dict[str, Any]]] = {}
    if not run_dir.is_dir():
        return out
    for path in sorted(run_dir.glob("phase_*.log")):
        m = PHASE_LOG_RE.search(path.name)
        if not m:
            continue
        phase_idx = int(m.group(1))
        records: list[dict[str, Any]] = []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[campaign_report] skip {path}: {exc}", file=sys.stderr)
            continue
        for line in text.splitlines():
            mm = BACKWARD_LINE_RE.search(line)
            if not mm:
                continue
            d = mm.groupdict()
            records.append(
                {
                    "iter": int(d["iter"]),
                    "tau": int(d["tau"]),
                    "tau_max": int(d["tau_max"]),
                    "gx": int(d["gx"]),
                    "trail_rate": float(d["trail_rate"]),
                    "advances": int(d["advances"]),
                    "ent_n": int(d["ent_n"]),
                    "ent_d": int(d["ent_d"]),
                    "ent_rate": float(d["ent_rate"]),
                    "truncated": int(d["trunc"]),
                    "scored": int(d["scored"]),
                    "budget": int(d["budget"]),
                }
            )
        if records:
            out[phase_idx] = records
    return out


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def phase_config_index(campaign_events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """{phase_idx: {"name": ..., "gate": ..., "budget_env_steps": ...}} from
    the campaign_start event's "phases" list, if present."""
    for ev in campaign_events:
        if ev.get("type") == "campaign_start":
            phases = ev.get("phases") or []
            return {p["idx"]: p for p in phases if "idx" in p}
    return {}


def phase_starts(campaign_events: list[dict[str, Any]]) -> list[tuple[float, int, str]]:
    """[(timestamp, phase_idx, name), ...] sorted by timestamp, from
    phase_start events."""
    out = []
    for ev in campaign_events:
        if ev.get("type") == "phase_start":
            out.append((float(ev.get("timestamp", 0.0)), int(ev["phase"]), ev.get("name", "")))
    out.sort(key=lambda t: t[0])
    return out


def assign_phase(timestamp: float, starts: list[tuple[float, int, str]]) -> int | None:
    """Bucket `timestamp` under the last phase_start at/before it."""
    best: int | None = None
    for ts, idx, _name in starts:
        if ts <= timestamp:
            best = idx
        else:
            break
    return best


def group_metrics_by_phase(
    metrics_rows: list[dict[str, Any]], starts: list[tuple[float, int, str]]
) -> dict[int | None, list[dict[str, Any]]]:
    grouped: dict[int | None, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        idx = assign_phase(float(row.get("timestamp", 0.0)), starts)
        grouped.setdefault(idx, []).append(row)
    return grouped


def summarize_series(
    values: list[Any], window: int = PLATEAU_WINDOW, band: float = PLATEAU_REL_BAND
) -> dict[str, Any]:
    """min/max over the whole series plus trailing-window plateau/trend
    detection.

    "flat" (plateau): the trailing window of up to `window` samples has
    (max-min) <= band * mean(|window|).

    "direction": trailing-window mean vs leading-window mean of the same
    size, +/- the same band -> "increasing" / "decreasing" / "flat".
    None fields mean "no usable data" (empty series or too few points).
    """
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "flat": None,
            "direction": None,
            "window_mean": None,
            "lead_mean": None,
        }
    w = clean[-window:]
    lead = clean[:window]
    w_mean = statistics.fmean(w)
    lead_mean = statistics.fmean(lead)

    flat: bool | None
    if len(w) >= 2:
        w_range = max(w) - min(w)
        denom = max(abs(w_mean), 1e-9)
        flat = w_range <= band * denom
    else:
        flat = None

    if len(lead) < 2 or len(w) < 2:
        direction = None
    else:
        eps = 1e-9
        if abs(lead_mean) < eps:
            if w_mean > band:
                direction = "increasing"
            elif w_mean < -band:
                direction = "decreasing"
            else:
                direction = "flat"
        elif w_mean > lead_mean * (1.0 + band):
            direction = "increasing"
        elif w_mean < lead_mean * (1.0 - band):
            direction = "decreasing"
        else:
            direction = "flat"

    return {
        "n": n,
        "min": min(clean),
        "max": max(clean),
        "flat": flat,
        "direction": direction,
        "window_mean": w_mean,
        "lead_mean": lead_mean,
    }


def summarize_sil(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    vals = [r.get("sil_clears_total") for r in rows if r.get("sil_clears_total") is not None]
    trajs = [r.get("sil_buffer_trajs") for r in rows if r.get("sil_buffer_trajs") is not None]
    if not vals:
        return None
    return {
        "start": vals[0],
        "end": vals[-1],
        "delta": vals[-1] - vals[0],
        "buffer_trajs_end": trajs[-1] if trajs else None,
    }


def summarize_backward(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    first, last = records[0], records[-1]
    max_advances = max(r["advances"] for r in records)
    return {
        "tau_start": f"{first['tau']}/{first['tau_max']}",
        "tau_end": f"{last['tau']}/{last['tau_max']}",
        "advances": max_advances,
        "entrance_start": f"{first['ent_n']}/{first['ent_d']}={first['ent_rate']:.3f}",
        "entrance_end": f"{last['ent_n']}/{last['ent_d']}={last['ent_rate']:.3f}",
        "budget_end": last["budget"],
        "n_lines": len(records),
    }


def build_timeline(campaign_events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for ev in campaign_events:
        ts = ev.get("timestamp")
        when = _fmt_ts(ts)
        typ = ev.get("type")
        if typ == "campaign_start":
            cfg = ev.get("config", {})
            lines.append(
                f"{when} campaign_start  base_profile={cfg.get('base_profile')}"
                f"  bottleneck_x={cfg.get('bottleneck_x')}"
                f"  kill_kl_threshold={cfg.get('kill_kl_threshold')}"
                f"  gate_sticky_clear={cfg.get('gate_sticky_clear')}"
            )
        elif typ == "phase_start":
            lines.append(
                f"{when} phase {ev.get('phase')} '{ev.get('name')}' START"
                f"  cum_env_steps={ev.get('cum_env_steps')}"
                f"  iters={ev.get('iters')}"
                f"  entropy_coef={ev.get('entropy_coef')}"
            )
        elif typ == "gate_pass":
            lines.append(
                f"{when} phase {ev.get('phase')} GATE PASS '{ev.get('gate')}'"
                f"  env_steps={ev.get('env_steps')}"
            )
        elif typ == "probe":
            lines.append(
                f"{when} phase {ev.get('phase')} PROBE  env_steps={ev.get('env_steps')}"
                f"  median_max_x={ev.get('median_max_x')}"
                f"  bottleneck_survival={ev.get('bottleneck_survival')}"
                f"  clear_rate={ev.get('clear_rate')}"
                f"  status={ev.get('status')}"
            )
        elif typ == "phase_complete":
            lines.append(
                f"{when} phase {ev.get('phase')} '{ev.get('name')}' COMPLETE"
                f"  env_steps={ev.get('env_steps')}"
            )
        elif typ == "abort":
            lines.append(
                f"{when} phase {ev.get('phase')} ABORT  env_steps={ev.get('env_steps')}"
                f"  reason={ev.get('reason')!r}"
            )
        else:
            lines.append(f"{when} {typ} {ev}")
    return lines


def build_probe_table(campaign_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [ev for ev in campaign_events if ev.get("type") == "probe"]


def _fmt_ts(ts: Any) -> str:
    if ts is None:
        return "?" .ljust(19)
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def determine_status(
    campaign_events: list[dict[str, Any]], phase_cfg: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Classify the attempt as ABORTED / COMPLETED / RUNNING / UNKNOWN from
    the tail of the event stream."""
    if not campaign_events:
        return {"state": "UNKNOWN", "detail": "no campaign.jsonl events found"}

    last = campaign_events[-1]
    typ = last.get("type")
    n_phases = len(phase_cfg)

    if typ == "abort":
        return {
            "state": "ABORTED",
            "detail": (
                f"phase {last.get('phase')} "
                f"({phase_cfg.get(last.get('phase'), {}).get('name', '?')}): "
                f"{last.get('reason')}"
            ),
            "env_steps": last.get("env_steps"),
        }
    if typ == "phase_complete":
        idx = last.get("phase")
        if n_phases and idx == max(phase_cfg):
            return {
                "state": "COMPLETED",
                "detail": f"all {n_phases} configured phases completed",
                "env_steps": last.get("env_steps"),
            }
        return {
            "state": "RUNNING",
            "detail": f"phase {idx} complete, next phase not yet started in the log",
            "env_steps": last.get("env_steps"),
        }
    if typ in ("phase_start", "gate_pass", "probe"):
        # Find the most recent phase_start to report which phase is in flight.
        idx = None
        name = ""
        for ev in reversed(campaign_events):
            if ev.get("type") == "phase_start":
                idx = ev.get("phase")
                name = ev.get("name", "")
                break
        return {
            "state": "RUNNING",
            "detail": f"phase {idx} ('{name}') in progress",
            "env_steps": last.get("env_steps"),
        }
    return {"state": "UNKNOWN", "detail": f"unrecognized trailing event type {typ!r}"}


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(run_dir: Path, ckpt_dir: Path) -> dict[str, Any]:
    campaign_events = load_campaign(run_dir)
    metrics_rows = load_metrics(ckpt_dir)
    backward_by_phase = load_backward_lines(run_dir)

    phase_cfg = phase_config_index(campaign_events)
    starts = phase_starts(campaign_events)
    metrics_by_phase = group_metrics_by_phase(metrics_rows, starts)

    phase_indices = sorted(
        {idx for idx in metrics_by_phase if idx is not None}
        | set(phase_cfg)
        | set(backward_by_phase)
    )

    phases: list[dict[str, Any]] = []
    for idx in phase_indices:
        rows = metrics_by_phase.get(idx, [])
        kl = summarize_series([r.get(KL_FIELD) for r in rows])
        vloss = summarize_series([r.get(VLOSS_FIELD) for r in rows])
        sil = summarize_sil(rows)
        backward = summarize_backward(backward_by_phase.get(idx, []))
        phases.append(
            {
                "idx": idx,
                "name": phase_cfg.get(idx, {}).get("name", "?"),
                "gate": phase_cfg.get(idx, {}).get("gate", "?"),
                "n_metrics_rows": len(rows),
                "kl": kl,
                "value_loss": vloss,
                "sil": sil,
                "backward": backward,
            }
        )

    return {
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "timeline": build_timeline(campaign_events),
        "probes": build_probe_table(campaign_events),
        "phases": phases,
        "status": determine_status(campaign_events, phase_cfg),
        "n_campaign_events": len(campaign_events),
        "n_metrics_rows": len(metrics_rows),
    }


def build_verdict(report: dict[str, Any]) -> str:
    status = report["status"]
    run_dir = report["run_dir"]
    probes = report["probes"]

    if probes:
        best_x = max(p.get("median_max_x") or 0 for p in probes)
        best_surv = max(p.get("bottleneck_survival") or 0.0 for p in probes)
        best_clear = max(p.get("clear_rate") or 0.0 for p in probes)
        probe_clause = (
            f"Across {len(probes)} probe(s), the best median max-x reached was "
            f"{best_x:g}, best bottleneck survival {best_surv:.3f}, "
            f"best clear rate {best_clear:.3f}."
        )
    else:
        probe_clause = "No probes ran (or none are in campaign.jsonl yet)."

    last_phase_with_kl = next(
        (p for p in reversed(report["phases"]) if p["kl"]["n"]), None
    )
    if last_phase_with_kl is not None:
        kl = last_phase_with_kl["kl"]
        kl_clause = (
            f"In the last phase with KL data (phase {last_phase_with_kl['idx']} "
            f"'{last_phase_with_kl['name']}'), kl_anchor_div ranged "
            f"[{kl['min']:.4f}, {kl['max']:.4f}] and its trailing window is "
            f"{'flat (plateaued)' if kl['flat'] else 'not flat' if kl['flat'] is not None else 'undetermined'}."
        )
    else:
        kl_clause = "No kl_anchor_div data was found in any phase."

    backward_phases = [p for p in report["phases"] if p["backward"]]
    if backward_phases:
        bp = backward_phases[-1]
        b = bp["backward"]
        backward_clause = (
            f"Backward curriculum (phase {bp['idx']} '{bp['name']}') moved tau "
            f"{b['tau_start']} -> {b['tau_end']} with {b['advances']} rung "
            f"advance(s); entrance-restart coverage went "
            f"{b['entrance_start']} -> {b['entrance_end']}."
        )
    else:
        backward_clause = "No backward-curriculum log lines were found."

    return (
        f"Attempt at {run_dir}: {status['state']} ({status['detail']}). "
        f"{probe_clause} {kl_clause} {backward_clause} "
        f"Next step: inspect the timeline and per-phase KL/value-loss trend "
        f"above before deciding whether to relaunch with adjusted gate/kill "
        f"thresholds."
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any, spec: str = "") -> str:
    if v is None:
        return "n/a"
    if spec:
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Campaign report: {report['run_dir']}")
    lines.append("")
    lines.append(f"ckpt-dir: {report['ckpt_dir']}")
    lines.append(
        f"{report['n_campaign_events']} campaign event(s), "
        f"{report['n_metrics_rows']} metrics row(s)"
    )
    lines.append("")

    lines.append("## Timeline")
    if report["timeline"]:
        lines.extend(f"- {row}" for row in report["timeline"])
    else:
        lines.append("- (no campaign.jsonl events found)")
    lines.append("")

    lines.append("## Probe table")
    probes = report["probes"]
    if probes:
        lines.append(
            "| phase | env_steps | n_episodes | median_max_x | bottleneck_survival | clear_rate | status |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for p in probes:
            lines.append(
                f"| {_fmt(p.get('phase'))} | {_fmt(p.get('env_steps'))} | "
                f"{_fmt(p.get('n_episodes'))} | {_fmt(p.get('median_max_x'))} | "
                f"{_fmt(p.get('bottleneck_survival'), '.3f') if p.get('bottleneck_survival') is not None else 'n/a'} | "
                f"{_fmt(p.get('clear_rate'), '.3f') if p.get('clear_rate') is not None else 'n/a'} | "
                f"{_fmt(p.get('status'))} |"
            )
    else:
        lines.append("(no probes found)")
    lines.append("")

    lines.append("## Per-phase summary")
    for p in report["phases"]:
        lines.append(f"### Phase {p['idx']} — {p['name']} (gate: {p['gate']})")
        lines.append(f"- metrics rows joined: {p['n_metrics_rows']}")
        kl = p["kl"]
        if kl["n"]:
            lines.append(
                f"- KL trace ({KL_FIELD}): n={kl['n']} min={_fmt(kl['min'], '.4f')} "
                f"max={_fmt(kl['max'], '.4f')} "
                f"plateau={'yes' if kl['flat'] else 'no' if kl['flat'] is not None else 'n/a'} "
                f"direction={kl['direction'] or 'n/a'}"
            )
        else:
            lines.append(f"- KL trace ({KL_FIELD}): no data")
        vl = p["value_loss"]
        if vl["n"]:
            lines.append(
                f"- Value loss ({VLOSS_FIELD}): n={vl['n']} min={_fmt(vl['min'], '.4f')} "
                f"max={_fmt(vl['max'], '.4f')} direction={vl['direction'] or 'n/a'}"
            )
        else:
            lines.append(f"- Value loss ({VLOSS_FIELD}): no data")
        if p["sil"]:
            s = p["sil"]
            lines.append(
                f"- SIL clears: {s['start']} -> {s['end']} (delta {s['delta']}), "
                f"buffer_trajs_end={s['buffer_trajs_end']}"
            )
        else:
            lines.append("- SIL clears: no data")
        if p["backward"]:
            b = p["backward"]
            lines.append(
                f"- Backward curriculum: tau {b['tau_start']} -> {b['tau_end']} "
                f"({b['n_lines']} log lines), advances={b['advances']}, "
                f"entrance {b['entrance_start']} -> {b['entrance_end']}, "
                f"final budget={b['budget_end']} steps"
            )
        else:
            lines.append("- Backward curriculum: no data")
        lines.append("")

    lines.append("## Status")
    lines.append(f"{report['status']['state']}: {report['status']['detail']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append(build_verdict(report))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="campaign run dir (has campaign.jsonl, phase_*.log)")
    parser.add_argument("--ckpt-dir", required=True, type=Path, help="checkpoint dir (has metrics.jsonl, runs/*/metrics.jsonl)")
    parser.add_argument("--out", type=Path, default=None, help="optional path to also write the markdown report to")
    args = parser.parse_args(argv)

    report = build_report(args.run_dir, args.ckpt_dir)
    text = render_report(report)
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[campaign_report] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
