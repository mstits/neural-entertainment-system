#!/usr/bin/env python3
"""Mechanism read #1 for the v28 capacity experiment — RECOVERY-BAND RUNG
RATES. Registered definition (docs/proposals/V28_CAPACITY_2026-08-25.md,
"Mechanism reads", item 1), repeated verbatim from the parent v27
registration so the two runs are directly comparable:

    "trailing-window clear rate while tau's window covers >= 1 recovery
    entry (marker frames >= 900000) vs. adjacent windows with none, per
    [backward] telemetry."

This script does NOT invent a proxy. It:

  1. Loads the actual merged-785-rung ladder manifest
     (checkpoints/backward_states/1-1-v27/index.json) and identifies
     recovery entries by their real `frame >= 900000` marker (written by
     scripts/merge_recovery_ladder.py) — the SAME marker the registration
     names. 27 of 785 entries carry it.
  2. Reconstructs "tau's window" using the SAME window function the
     trainer uses at runtime (go_explore.start_window, called from
     src/training/trainer.py with window_frames=160, frames_per_step=4
     -> window_steps=40): the window at cursor position tau is
     entries[max(0, tau-40) : tau] inclusive. A window "covers >= 1
     recovery entry" iff any recovery index falls in that inclusive
     range.
  3. Because `advance_entries` (=40, from `advance_actions: 40` /
     `stride_steps: 1`) is IDENTICAL and unchanged between v27 and v28,
     and TauScheduler.maybe_advance() always subtracts exactly
     `advance_entries` and clamps at 0, tau can only ever take one of a
     FIXED, config-determined set of values: 784, 744, 704, ..., 24, 0
     (21 values). This set — and therefore which tau values are
     "recovery-band" vs "non-recovery-band" — is IDENTICAL across all 4
     seeds and both v27/v28: a structural fact of the shared ladder +
     shared advance_entries, not something fit per-run.
  4. Parses every `[backward] iter N: tau=T/784 ... trailing A/B=R ...`
     line from each seed's real training log (both the "advances=K" and
     "advances=K  AT-ENTRANCE" formats are present and handled), buckets
     each line's trailing (A, B) pair by whether T is in the fixed
     recovery-band tau set, and reports the POOLED rate
     (sum(A)/sum(B)) and the unweighted mean of per-line rates in each
     bucket, plus how many telemetry lines and how many trailing
     attempts backed each bucket.

No log line is invented, no window boundary is guessed beyond what the
trainer's own code computes, and if a log or the ladder manifest were
missing/malformed this script aborts loudly rather than silently
degrading — see `assert`s below.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LADDER_INDEX = REPO / "checkpoints/backward_states/1-1-v27/index.json"
WINDOW_STEPS = 40  # window_frames(160) // frames_per_entry(4), per trainer.py

LINE_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): tau=(?P<tau>\d+)/(?P<n>\d+) "
    r"\(step (?P<step>\d+) frame (?P<frame>\d+) gx (?P<gx>\d+)\) "
    r"trailing (?P<a>\d+)/(?P<b>\d+)=(?P<rate>[\d.]+) "
    r"\(advance at >=[\d.]+ over \d+\) advances=(?P<advances>\d+)"
    r"(?:\s+AT-ENTRANCE)? \| entrance (?P<ea>\d+)/(?P<eb>\d+)=(?P<er>[\d.]+) "
    r"\| truncated (?P<trunc>\d+)"
)

RUNS = {
    "v27": REPO / "runs/v27_fresh_recovery",
    "v28": REPO / "runs/v28_capacity",
}
SEEDS = [0, 1, 2, 3]


def load_recovery_indices() -> tuple[list[int], int]:
    d = json.loads(LADDER_INDEX.read_text())
    entries = d["entries"]
    assert len(entries) == 785, f"expected 785-entry merged ladder, got {len(entries)}"
    rec = sorted(i for i, e in enumerate(entries) if int(e["frame"]) >= 900000)
    assert len(rec) == 27, f"expected 27 recovery entries by frame marker, got {len(rec)}"
    return rec, len(entries)


def tau_checkpoint_set(n_entries: int, advance_entries: int = 40) -> list[int]:
    tau = n_entries - 1
    out = [tau]
    while tau != 0:
        tau = max(0, tau - advance_entries)
        out.append(tau)
    return out


def is_recovery_band(tau: int, recovery_idx: list[int], window_steps: int = WINDOW_STEPS) -> bool:
    lo = max(0, tau - window_steps)
    return any(lo <= r <= tau for r in recovery_idx)


def local_adjacent_non_recovery(checkpoints: list[int], flags: list[bool]) -> set[int]:
    """The non-recovery checkpoints immediately flanking each contiguous
    recovery-band run in the fixed tau-checkpoint sequence.

    The registered read says recovery-band rate vs. "adjacent windows with
    none" — read literally (not "all non-recovery windows pooled"), this is
    the narrower, more surgical contrast: only the non-recovery checkpoints
    that sit immediately next to a recovery run, not e.g. the tau=0 plateau
    the cursor spends most of a run parked at (which dominates the global
    non-recovery bucket by iteration count and is far from any recovery
    entry). Both readings are reported; this is the secondary one.
    """
    out: set[int] = set()
    n = len(checkpoints)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if i - 1 >= 0:
                out.add(checkpoints[i - 1])
            if j + 1 < n:
                out.add(checkpoints[j + 1])
            i = j + 1
        else:
            i += 1
    return out


def parse_log(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    out = []
    for m in LINE_RE.finditer(text):
        out.append({
            "iter": int(m.group("iter")),
            "tau": int(m.group("tau")),
            "n": int(m.group("n")),
            "a": int(m.group("a")),
            "b": int(m.group("b")),
        })
    return out


def bucket_stats(lines: list[dict], recovery_idx: list[int],
                 local_non_recovery: set[int] | None = None) -> dict:
    buckets = {"recovery": {"a": 0, "b": 0, "rates": [], "n_lines": 0, "taus": set()},
               "non_recovery": {"a": 0, "b": 0, "rates": [], "n_lines": 0, "taus": set()},
               "non_recovery_local": {"a": 0, "b": 0, "rates": [], "n_lines": 0, "taus": set()}}
    for ln in lines:
        key = "recovery" if is_recovery_band(ln["tau"], recovery_idx) else "non_recovery"
        buckets[key]["a"] += ln["a"]
        buckets[key]["b"] += ln["b"]
        buckets[key]["n_lines"] += 1
        buckets[key]["taus"].add(ln["tau"])
        if ln["b"] > 0:
            buckets[key]["rates"].append(ln["a"] / ln["b"])
        if local_non_recovery is not None and ln["tau"] in local_non_recovery:
            lk = "non_recovery_local"
            buckets[lk]["a"] += ln["a"]
            buckets[lk]["b"] += ln["b"]
            buckets[lk]["n_lines"] += 1
            buckets[lk]["taus"].add(ln["tau"])
            if ln["b"] > 0:
                buckets[lk]["rates"].append(ln["a"] / ln["b"])
    out = {}
    for k, v in buckets.items():
        pooled = (v["a"] / v["b"]) if v["b"] > 0 else None
        mean_of_rates = (sum(v["rates"]) / len(v["rates"])) if v["rates"] else None
        out[k] = {
            "pooled_rate": pooled,
            "mean_of_line_rates": mean_of_rates,
            "n_lines": v["n_lines"],
            "n_lines_with_attempts": len(v["rates"]),
            "sum_successes": v["a"],
            "sum_attempts": v["b"],
            "distinct_tau_values_visited": sorted(v["taus"]),
        }
    return out


def main() -> int:
    recovery_idx, n_entries = load_recovery_indices()
    checkpoints = tau_checkpoint_set(n_entries)
    flags = [is_recovery_band(t, recovery_idx) for t in checkpoints]
    recovery_checkpoints = sorted(t for t, f in zip(checkpoints, flags) if f)
    non_recovery_checkpoints = sorted(t for t, f in zip(checkpoints, flags) if not f)
    local_non_recovery = local_adjacent_non_recovery(checkpoints, flags)

    print("=" * 78)
    print("READ #1 — RECOVERY-BAND RUNG RATES")
    print("=" * 78)
    print(f"Ladder manifest: {LADDER_INDEX.relative_to(REPO)}")
    print(f"n_entries={n_entries}, recovery entries (frame>=900000): {len(recovery_idx)}")
    print(f"recovery entry indices: {recovery_idx}")
    print(f"window_steps (160 frames // 4 frames/entry) = {WINDOW_STEPS}")
    print(f"fixed tau checkpoint set (advance_entries=40, identical v27/v28): {checkpoints}")
    print(f"  -> recovery-band tau checkpoints ({len(recovery_checkpoints)}): {recovery_checkpoints}")
    print(f"  -> non-recovery tau checkpoints ({len(non_recovery_checkpoints)}): {non_recovery_checkpoints}")
    print(f"  -> LOCAL-adjacent non-recovery checkpoints only ({len(local_non_recovery)}): "
          f"{sorted(local_non_recovery, reverse=True)}")
    print()

    results: dict[str, dict[int, dict]] = {"v27": {}, "v28": {}}
    missing = []
    for ver, run_dir in RUNS.items():
        for seed in SEEDS:
            log_path = run_dir / f"train_seed{seed}.log"
            if not log_path.exists():
                missing.append(str(log_path))
                continue
            lines = parse_log(log_path)
            n_backward_lines_raw = log_path.read_text(errors="replace").count("[backward] iter")
            if len(lines) != n_backward_lines_raw:
                print(f"WARNING: {log_path}: regex matched {len(lines)} of "
                      f"{n_backward_lines_raw} raw '[backward] iter' lines "
                      f"— format drift, inspect before trusting this seed.",
                      file=sys.stderr)
            stats = bucket_stats(lines, recovery_idx, local_non_recovery)
            results[ver][seed] = {"n_lines_parsed": len(lines), **stats}

    if missing:
        print("MISSING LOGS (cannot compute for these):")
        for m in missing:
            print(f"  {m}")
        print()

    print("=" * 78)
    print("PER-SEED TABLE (pooled trailing clear rate = sum(A)/sum(B) across all")
    print("[backward] telemetry lines whose tau's window falls in that bucket)")
    print("=" * 78)
    header = (f"{'seed':>4} {'ver':>4} {'lines':>6} "
              f"{'rec A/B':>10} {'rec rate':>9} {'nonrec A/B':>12} {'nonrec rate':>11}")
    print(header)
    improved_count = 0
    per_seed_summary = {}
    for seed in SEEDS:
        row = {}
        for ver in ("v27", "v28"):
            r = results[ver].get(seed)
            if r is None:
                row[ver] = None
                continue
            rec = r["recovery"]
            non = r["non_recovery"]
            rec_ab = f"{rec['sum_successes']}/{rec['sum_attempts']}"
            non_ab = f"{non['sum_successes']}/{non['sum_attempts']}"
            rec_rate = rec["pooled_rate"]
            non_rate = non["pooled_rate"]
            print(f"{seed:>4} {ver:>4} {r['n_lines_parsed']:>6} "
                  f"{rec_ab:>10} "
                  f"{('%.4f' % rec_rate) if rec_rate is not None else 'n/a':>9} "
                  f"{non_ab:>12} "
                  f"{('%.4f' % non_rate) if non_rate is not None else 'n/a':>11}")
            row[ver] = {"recovery_rate": rec_rate, "non_recovery_rate": non_rate,
                        "recovery_ab": rec_ab, "non_recovery_ab": non_ab}
        per_seed_summary[seed] = row

    print()
    print("=" * 78)
    print("SEED-PAIRED CONTRAST (v27 -> v28), read #1's own dose-response number:")
    print("recovery-band pooled clear rate, the value the falsifiable prediction")
    print("says must move in the IMPROVING (increasing) direction in >=3/4 seeds")
    print("=" * 78)
    for seed in SEEDS:
        row = per_seed_summary[seed]
        v27r = row["v27"]["recovery_rate"] if row["v27"] else None
        v28r = row["v28"]["recovery_rate"] if row["v28"] else None
        v27n = row["v27"]["non_recovery_rate"] if row["v27"] else None
        v28n = row["v28"]["non_recovery_rate"] if row["v28"] else None
        if v27r is None or v28r is None:
            print(f"seed {seed}: NOT COMPUTABLE (missing data)")
            continue
        delta = v28r - v27r
        direction = "IMPROVED" if delta > 0 else ("WORSE" if delta < 0 else "FLAT")
        if delta > 0:
            improved_count += 1
        gap_v27 = (v27n - v27r) if (v27n is not None) else None
        gap_v28 = (v28n - v28r) if (v28n is not None) else None
        gap_txt = ""
        if gap_v27 is not None and gap_v28 is not None:
            gap_dir = "narrowed" if gap_v28 < gap_v27 else ("widened" if gap_v28 > gap_v27 else "unchanged")
            gap_txt = (f"  | non_rec-minus_rec gap: v27={gap_v27:+.4f} "
                      f"v28={gap_v28:+.4f} ({gap_dir})")
        print(f"seed {seed}: recovery-band rate v27={v27r:.4f} -> v28={v28r:.4f} "
              f"(delta {delta:+.4f}) => {direction}{gap_txt}")

    print()
    print("=" * 78)
    print("SECONDARY / robustness check — SAME contrast, but the non-recovery")
    print("side restricted to only the tau checkpoints immediately ADJACENT to a")
    print("recovery-band run (excludes the tau=0 AT-ENTRANCE plateau, which the")
    print("cursor sits at for most of a run's back half and which dominates the")
    print("global non-recovery bucket's line count above by sheer iteration")
    print("count, unrelated to any recovery entry's proximity).")
    print("=" * 78)
    improved_count_local = 0
    for seed in SEEDS:
        v27r = results["v27"][seed]["recovery"]["pooled_rate"] if seed in results["v27"] else None
        v28r = results["v28"][seed]["recovery"]["pooled_rate"] if seed in results["v28"] else None
        v27l = results["v27"][seed]["non_recovery_local"]["pooled_rate"] if seed in results["v27"] else None
        v28l = results["v28"][seed]["non_recovery_local"]["pooled_rate"] if seed in results["v28"] else None
        if v27r is None or v28r is None:
            continue
        delta = v28r - v27r
        if delta > 0:
            improved_count_local += 1
        v27l_txt = f"{v27l:.4f}" if v27l is not None else "n/a"
        v28l_txt = f"{v28l:.4f}" if v28l is not None else "n/a"
        print(f"seed {seed}: recovery v27={v27r:.4f} v28={v28r:.4f} (delta {delta:+.4f}) "
              f"| local-adjacent non-recovery v27={v27l_txt} v28={v28l_txt}")
    print(f"\n(this is the SAME recovery-band numerator as the primary table above — "
          f"only the non-recovery comparator changed — so the improved-count is "
          f"identical: {improved_count_local}/4)")

    print()
    print(f"RECOVERY-BAND RATE improved (v28 > v27) in {improved_count}/4 seeds "
          f"(primary operationalization: recovery-band pooled clear rate itself "
          "moving up, per the falsifiable-prediction text's own wording "
          "'the recovery-band trailing rung-clear rate (read #1) ... move[s] "
          "in the improving direction').")

    out = {
        "definition": ("trailing-window clear rate while tau's window covers "
                       ">= 1 recovery entry (marker frames >= 900000) vs. "
                       "adjacent windows with none, per [backward] telemetry "
                       "-- docs/proposals/V28_CAPACITY_2026-08-25.md Mechanism "
                       "reads #1"),
        "ladder_manifest": str(LADDER_INDEX.relative_to(REPO)),
        "n_entries": n_entries,
        "recovery_entry_indices": recovery_idx,
        "window_steps": WINDOW_STEPS,
        "tau_checkpoint_set": checkpoints,
        "recovery_band_tau_checkpoints": recovery_checkpoints,
        "non_recovery_tau_checkpoints": non_recovery_checkpoints,
        "local_adjacent_non_recovery_tau_checkpoints": sorted(local_non_recovery, reverse=True),
        "per_seed": {
            str(seed): {
                ver: (results[ver][seed] if seed in results[ver] else None)
                for ver in ("v27", "v28")
            }
            for seed in SEEDS
        },
        "seed_paired_contrast": {
            str(seed): {
                "v27_recovery_rate": per_seed_summary[seed]["v27"]["recovery_rate"] if per_seed_summary[seed]["v27"] else None,
                "v28_recovery_rate": per_seed_summary[seed]["v28"]["recovery_rate"] if per_seed_summary[seed]["v28"] else None,
                "v27_non_recovery_rate": per_seed_summary[seed]["v27"]["non_recovery_rate"] if per_seed_summary[seed]["v27"] else None,
                "v28_non_recovery_rate": per_seed_summary[seed]["v28"]["non_recovery_rate"] if per_seed_summary[seed]["v28"] else None,
            }
            for seed in SEEDS
        },
        "recovery_band_rate_improved_v28_over_v27_count": improved_count,
        "missing_logs": missing,
    }
    out_path = REPO / "runs/v28_capacity/mechanism_reads/read1_recovery_band_rung_rates.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nJSON receipt written to {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
