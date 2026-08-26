#!/usr/bin/env python3
"""Sensitivity check on changepoint_analysis.py's two free parameters
(the "how big a jump counts as a regime change" factor, and the "how many
iters must it hold" sustain window).

Why this exists: a lead/lag finding computed at one arbitrarily-chosen
threshold is not trustworthy on its own -- this project's discipline calls
for naming and checking confounds, and "I picked the threshold that gave
the story I wanted" is exactly the kind of confound a threshold-sensitivity
sweep is supposed to catch. Run this BEFORE trusting changepoint_analysis.py's
single-threshold output.

Usage:
    .venv/bin/python runs/peak_instability/value_critic_health/threshold_sensitivity.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("changepoint_analysis", HERE / "changepoint_analysis.py")
cp = importlib.util.module_from_spec(spec)
sys.modules["changepoint_analysis"] = cp
spec.loader.exec_module(cp)

DATA = cp.DATA
PEAK_ITER = cp.PEAK_ITER


def sweep():
    print(f"{'factor':>6s} {'sustain':>7s} {'n_computable':>12s} {'n_positive':>10s}  leads")
    for factor in (1.3, 1.5, 2.0):
        for sustain in (3, 5, 8):
            leads = []
            for label, series in DATA.items():
                gens = series["generation"]
                vl = series["ppo_value_loss"]
                sr = series["success_rate"]
                vl_s = cp.rolling_mean(vl, 5)
                sr_s = cp.rolling_mean(sr, 5)
                trough_iter, trough_val, trough_i = cp.find_trough(gens, vl_s, min_gen=20)
                regime_chg = cp.find_vloss_regime_change(
                    gens, vl_s, trough_i, trough_val, sustain=sustain, factor=factor
                )
                sr_peak_val = max(v for v in sr_s if v is not None)
                collapse_iter = cp.find_collapse(gens, sr_s, PEAK_ITER[label], sr_peak_val)
                if regime_chg is not None and collapse_iter is not None:
                    leads.append(collapse_iter - regime_chg)
            pos = sum(1 for l in leads if l > 0)
            print(f"{factor:>6.1f} {sustain:>7d} {len(leads):>12d} {pos:>10d}  {sorted(leads)}")

    print()
    print("Reading: at a LOOSE threshold (1.3x-1.5x the post-warmup trough), value")
    print("loss appears to lead the success-rate collapse in 7/7 computable runs by")
    print("a wide, threshold-dependent margin. At a STRICT threshold (2.0x, i.e. the")
    print("value loss must actually double before counting as 'destabilized'), the")
    print("lead collapses to near-zero and goes NEGATIVE in most runs -- meaning a")
    print("large value-loss blowup is contemporaneous with or slightly LAGS the")
    print("collapse, not a clean leading indicator. Report both; do not quote only")
    print("the threshold that supports a leading-indicator story.")


if __name__ == "__main__":
    sweep()
