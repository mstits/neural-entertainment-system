"""Cross-run summary over the 8 probe_checkpoints.py outputs.

Answers, per run and pooled:
  (a) does argmax-action usage collapse to 1-2 actions (classic failure)?
  (b) is fc1/actor weight growth still moving (not frozen) at iter 240,
      and does it saturate near the peak or keep growing past it?
  (c) fraction of the 6 actions ever selected, over the run
  (d) logit magnitude trajectory / saturation point relative to peak
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = [
    ("v27_seed0", 0),
    ("v27_seed1", 0),
    ("v27_seed2", 0),
    ("v27_seed3", 0),
    ("v28_seed0", 0),
    ("v28_seed1", 0),
    ("v28_seed2", 0),
    ("v28_seed3", 0),
]

# honest@peak / honest@final from the phenomenon table, for correlation only
HONEST = {
    "v27_seed0": (0.040, 0.020, 60),
    "v27_seed1": (0.290, 0.020, 50),
    "v27_seed2": (0.530, 0.000, 90),
    "v27_seed3": (0.170, 0.010, 60),
    "v28_seed0": (0.450, 0.000, 70),
    "v28_seed1": (0.230, 0.050, 60),
    "v28_seed2": (0.370, 0.000, 120),
    "v28_seed3": (0.670, 0.000, 90),
}

DIR = Path(__file__).resolve().parent


def load(name):
    return json.loads((DIR / f"{name}.json").read_text())


def main():
    print(f"{'run':10s} {'peak':>5s} {'nact@peak':>10s} {'nact@240':>9s} "
          f"{'minfrac@peak':>12s} {'maxfrac@240':>12s} "
          f"{'actorW@peak':>11s} {'actorW@240':>11s} {'actorW growth':>14s} "
          f"{'fc1W@peak':>10s} {'fc1W@240':>9s} "
          f"{'critW@peak':>11s} {'critW@240':>10s} "
          f"{'logit@peak':>11s} {'logit@240':>10s}")
    rows_all = []
    for name, _ in RUNS:
        d = load(name)
        rows = {r["iter"]: r for r in d["rows"]}
        peak = d["peak_iter"]
        r_peak = rows[peak]
        r_240 = rows[240]
        actor_growth_post_peak = (
            r_240["weight_norms"]["actor.weight"]
            / r_peak["weight_norms"]["actor.weight"]
        )
        print(f"{name:10s} {peak:5d} "
              f"{r_peak['n_actions_used']:10d} {r_240['n_actions_used']:9d} "
              f"{min(r_peak['action_fracs']):12.3f} {max(r_240['action_fracs']):12.3f} "
              f"{r_peak['weight_norms']['actor.weight']:11.3f} "
              f"{r_240['weight_norms']['actor.weight']:11.3f} "
              f"{actor_growth_post_peak:14.2f}x "
              f"{r_peak['weight_norms']['fc1.weight']:10.3f} "
              f"{r_240['weight_norms']['fc1.weight']:9.3f} "
              f"{r_peak['weight_norms']['critic.weight']:11.3f} "
              f"{r_240['weight_norms']['critic.weight']:10.3f} "
              f"{r_peak['logit_abs_max']:11.3f} {r_240['logit_abs_max']:10.3f}")
        rows_all.append((name, d, rows, peak))

    print()
    print("=== (a)/(c) action-usage collapse check: n_actions_used trajectory ===")
    for name, d, rows, peak in rows_all:
        traj = [rows[it]["n_actions_used"] for it in sorted(rows)]
        min_traj = min(traj)
        print(f"{name:10s} peak={peak:3d} min_n_actions_used_over_run={min_traj} "
              f"trajectory={traj}")

    print()
    print("=== weight-delta-from-prev (is the net still moving at iter 240?) ===")
    for name, d, rows, peak in rows_all:
        r_240 = rows[240]
        r_230 = rows[230]
        print(f"{name:10s} relative_delta(230->240)={r_240['weight_delta_relative_from_prev']:.4f} "
              f"cos(230,240)={r_240['weight_delta_cos_from_prev']:.5f}")

    print()
    print("=== entropy vs weight-norm saturation timing (fc1/actor never plateau; "
          "norm2/critic plateau near peak) ===")
    for name, d, rows, peak in rows_all:
        iters = sorted(rows)
        # crude saturation index: iter at which fc2/norm2/critic weight norm
        # reaches 95% of its iter-240 value
        def sat_iter(key):
            final = rows[240]["weight_norms"][key]
            for it in iters:
                if rows[it]["weight_norms"][key] >= 0.95 * final:
                    return it
            return None
        print(f"{name:10s} peak={peak:3d} "
              f"critic_sat95={sat_iter('critic.weight')} "
              f"norm2_sat95={sat_iter('norm2.weight')} "
              f"fc1_sat95={sat_iter('fc1.weight')} "
              f"actor_sat95={sat_iter('actor.weight')}")

    print()
    print("=== honest score correlation sanity (from task-supplied table) ===")
    for name, d, rows, peak in RUNS and rows_all:
        h_peak, h_final, ph = HONEST[name]
        r_peak = rows[peak]
        print(f"{name:10s} honest@peak={h_peak:.3f} honest@final={h_final:.3f} "
              f"entropy@peak={r_peak['entropy_mean']:.4f} "
              f"maxfrac@peak={r_peak['max_action_frac']:.3f} "
              f"nact@peak={r_peak['n_actions_used']}")


if __name__ == "__main__":
    main()
