#!/usr/bin/env python3
"""Quantify WHEN ppo_value_loss shifts regime relative to WHEN success_rate
(the in-run, on-policy proxy for capability) collapses, for all 8 runs.

Method (deliberately simple / auditable, not a fitted model):
  1. Smooth ppo_value_loss and success_rate with a centered rolling mean
     (window=5) to kill single-iteration noise without hiding real shifts.
  2. trough_iter = iter of the global minimum of smoothed value loss.
  3. vloss_regime_change_iter = first iter AFTER trough_iter where smoothed
     value loss exceeds 1.5x the trough value AND stays above that line for
     >=5 consecutive iters after (kills one-off blips).
  4. sr_peak_val = max of smoothed success_rate over the whole run.
     collapse_iter = first iter AFTER the run's honest peak_iter where
     smoothed success_rate drops below 0.5x sr_peak_val AND never rises
     back above that line for more than 2 consecutive iters for the rest
     of the run (kills one-off dips, requires a PERMANENT collapse).
  5. lead = collapse_iter - vloss_regime_change_iter.
     lead > 0  => value loss regime-changed BEFORE success rate collapsed
                  (causal-shaped: consistent with the critic destabilizing
                  first).
     lead <= 0 => value loss regime-changed AT/AFTER the collapse
                  (symptom-shaped: consistent with value loss just tracking
                  return-variance produced by an already-destabilizing
                  policy).

This is descriptive timing evidence, not a causal proof -- see the writeup
for the return-variance confound this method cannot rule out on its own.

Usage:
    .venv/bin/python runs/peak_instability/value_critic_health/changepoint_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = json.loads(
    (ROOT / "runs" / "peak_instability" / "value_critic_health" / "value_trajectories.json").read_text()
)

PEAK_ITER = {
    "v27_seed0": 60, "v27_seed1": 50, "v27_seed2": 90, "v27_seed3": 60,
    "v28_seed0": 70, "v28_seed1": 60, "v28_seed2": 120, "v28_seed3": 90,
}


def rolling_mean(vals: list[float | None], window: int) -> list[float | None]:
    n = len(vals)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = [v for v in vals[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def find_trough(gens, smoothed, min_gen=0):
    """Global min of smoothed value loss, restricted to gens >= min_gen.

    min_gen matters: all 8 runs share an early cold-start transient (iters
    ~0-15) where value loss starts artificially low (near-zero init output
    vs near-zero early returns) before spiking through the real "learning
    to estimate returns" hump around iter 20-30. Without excluding that
    window, the "trough" trivially locks onto iter 0-1 for any run whose
    later, meaningful trough never dips that low -- which fires the
    regime-change detector almost immediately and inflates lead spuriously.
    Default min_gen=0 preserves the naive/raw behavior for comparison.
    """
    best_i = None
    for i, v in enumerate(smoothed):
        if v is None or gens[i] < min_gen:
            continue
        if best_i is None or v < smoothed[best_i]:
            best_i = i
    return gens[best_i], smoothed[best_i], best_i


def find_vloss_regime_change(gens, smoothed, trough_i, trough_val, sustain=5, factor=1.5):
    thresh = trough_val * factor
    n = len(smoothed)
    for i in range(trough_i + 1, n):
        if smoothed[i] is None or smoothed[i] <= thresh:
            continue
        # check sustained for `sustain` consecutive points from i
        window = smoothed[i:i + sustain]
        if len(window) == sustain and all(v is not None and v > thresh for v in window):
            return gens[i]
    return None


def find_collapse(gens, smoothed, peak_iter, sr_peak_val, tolerate=2, factor=0.5):
    thresh = sr_peak_val * factor
    n = len(smoothed)
    peak_i = gens.index(peak_iter) if peak_iter in gens else 0
    for i in range(peak_i, n):
        if smoothed[i] is None or smoothed[i] >= thresh:
            continue
        # candidate collapse start at i -- verify it's permanent: no run
        # of > `tolerate` consecutive points for the rest of the series
        # ever climbs back above thresh
        ok = True
        j = i
        while j < n:
            if smoothed[j] is not None and smoothed[j] >= thresh:
                # count how long it stays above
                k = j
                run_len = 0
                while k < n and smoothed[k] is not None and smoothed[k] >= thresh:
                    run_len += 1
                    k += 1
                if run_len > tolerate:
                    ok = False
                    break
                j = k
            else:
                j += 1
        if ok:
            return gens[i]
    return None


def main():
    import sys
    min_gen = 20 if "--exclude-warmup" in sys.argv else 0
    label_suffix = " (trough search excludes gens<20 cold-start transient)" if min_gen else " (RAW, no warmup exclusion)"
    print(f"### {label_suffix}")
    print(f"{'run':10s} {'peak_it':>7s} {'vl_trough':>9s}@{'it':<4s} "
          f"{'vl_regime_chg':>13s} {'sr_peak':>7s} {'collapse_it':>11s} {'lead(iters)':>11s}")
    leads = []
    rows_out = []
    for label, series in DATA.items():
        gens = series["generation"]
        vl = series["ppo_value_loss"]
        sr = series["success_rate"]
        vl_s = rolling_mean(vl, 5)
        sr_s = rolling_mean(sr, 5)

        trough_iter, trough_val, trough_i = find_trough(gens, vl_s, min_gen=min_gen)
        regime_chg = find_vloss_regime_change(gens, vl_s, trough_i, trough_val)

        sr_peak_val = max(v for v in sr_s if v is not None)
        collapse_iter = find_collapse(gens, sr_s, PEAK_ITER[label], sr_peak_val)

        lead = (collapse_iter - regime_chg) if (regime_chg is not None and collapse_iter is not None) else None
        if lead is not None:
            leads.append(lead)

        print(f"{label:10s} {PEAK_ITER[label]:>7d} {trough_val:>9.3f}@{trough_iter:<4d} "
              f"{str(regime_chg):>13s} {sr_peak_val:>7.3f} {str(collapse_iter):>11s} {str(lead):>11s}")
        rows_out.append(dict(run=label, peak_iter=PEAK_ITER[label], vl_trough_val=trough_val,
                              vl_trough_iter=trough_iter, vl_regime_change_iter=regime_chg,
                              sr_peak_smoothed=sr_peak_val, collapse_iter=collapse_iter, lead_iters=lead))

    print()
    if leads:
        n_positive = sum(1 for l in leads if l > 0)
        print(f"lead computed for {len(leads)}/8 runs")
        print(f"  positive lead (vloss regime-changes BEFORE collapse) in {n_positive}/{len(leads)}")
        print(f"  leads: {sorted(leads)}")
        print(f"  median lead: {sorted(leads)[len(leads)//2]}")
    else:
        print("lead not computable for any run under this method")

    out_path = ROOT / "runs" / "peak_instability" / "value_critic_health" / (
        "changepoint_summary_excl_warmup.json" if min_gen else "changepoint_summary_raw.json"
    )
    out_path.write_text(json.dumps(rows_out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
