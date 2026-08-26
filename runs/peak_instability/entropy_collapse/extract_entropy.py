"""Entropy-collapse dimension: extract ppo_entropy trajectories for all 8
from-scratch runs (v27 48k x4 seeds, v28 72k x4 seeds), find where entropy
crosses 0.5 / 0.3 / 0.1, and relate those crossings to the AUTHORITATIVE
peak iter recorded in each run's winners/best.json.

Run: .venv/bin/python runs/peak_instability/entropy_collapse/extract_entropy.py

Writes runs/peak_instability/entropy_collapse/entropy_trajectories.json
(raw per-run per-generation entropy + a few training-time proxy fields)
and prints the crossing/lag table plus the ruling-out checks to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

RUNS = [
    ("v27_seed0", "checkpoints/mario_1_1_v27_recovery_seed0", 48000),
    ("v27_seed1", "checkpoints/mario_1_1_v27_recovery_seed1", 48000),
    ("v27_seed2", "checkpoints/mario_1_1_v27_recovery_seed2", 48000),
    ("v27_seed3", "checkpoints/mario_1_1_v27_recovery_seed3", 48000),
    ("v28_seed0", "checkpoints/mario_1_1_v28_capacity_seed0", 72000),
    ("v28_seed1", "checkpoints/mario_1_1_v28_capacity_seed1", 72000),
    ("v28_seed2", "checkpoints/mario_1_1_v28_capacity_seed2", 72000),
    ("v28_seed3", "checkpoints/mario_1_1_v28_capacity_seed3", 72000),
]

# honest-protocol scores handed down in the task prompt (cold entrance,
# greedy, sticky 0.25, jitter +-16, 100 eps pooled) -- not recomputed here,
# just carried alongside so the report can line entropy up against them.
HONEST = {
    "v27_seed0": dict(peak_iter=60, honest_peak=0.040, honest_final=0.020),
    "v27_seed1": dict(peak_iter=50, honest_peak=0.290, honest_final=0.020),
    "v27_seed2": dict(peak_iter=90, honest_peak=0.530, honest_final=0.000),
    "v27_seed3": dict(peak_iter=60, honest_peak=0.170, honest_final=0.010),
    "v28_seed0": dict(peak_iter=70, honest_peak=0.450, honest_final=0.000),
    "v28_seed1": dict(peak_iter=60, honest_peak=0.230, honest_final=0.050),
    "v28_seed2": dict(peak_iter=120, honest_peak=0.370, honest_final=0.000),
    "v28_seed3": dict(peak_iter=90, honest_peak=0.670, honest_final=0.000),
}

THRESHOLDS = [0.5, 0.3, 0.1]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def first_crossing(gens: list[int], entropy: list[float], threshold: float) -> tuple[int | None, bool]:
    """First generation at/below `threshold`. Also flags whether entropy
    ever goes back ABOVE threshold after that point (non-monotonic
    crossing) so a single-iter noise dip doesn't get reported as a clean
    collapse point without a flag.
    """
    cross_gen = None
    for g, e in zip(gens, entropy):
        if e <= threshold:
            cross_gen = g
            break
    if cross_gen is None:
        return None, False
    # noisy if entropy pops back above threshold at any LATER generation
    noisy = any(e > threshold for g, e in zip(gens, entropy) if g > cross_gen)
    return cross_gen, noisy


def main() -> None:
    all_data = {}
    print(f"{'run':10} {'width':6} {'peak_it':7} {'ent@peak':9} "
          f"{'x0.5':6} {'lag.5':6} {'x0.3':6} {'lag.3':6} {'x0.1':6} {'lag.1':6} {'ent@final':10}")
    rows_for_summary = []
    for name, rel, width in RUNS:
        d = REPO / rel
        rows = read_jsonl(d / "metrics.jsonl")
        rows.sort(key=lambda r: r["generation"])
        gens = [r["generation"] for r in rows]
        entropy = [r.get("ppo_entropy") for r in rows]
        succ = [r.get("success_rate") for r in rows]
        clears = [r.get("vanilla_ppo_clears") for r in rows]
        maxx = [r.get("vanilla_ppo_max_x") for r in rows]

        peak_it = HONEST[name]["peak_iter"]
        # entropy AT the peak generation (exact row match; peak_it is a
        # logged generation number, all 8 runs log every generation 0..249)
        ent_at_peak = None
        for g, e in zip(gens, entropy):
            if g == peak_it:
                ent_at_peak = e
                break
        ent_at_final = entropy[-1]
        gen_at_final = gens[-1]

        crossings = {}
        for th in THRESHOLDS:
            cg, noisy = first_crossing(gens, entropy, th)
            crossings[th] = (cg, noisy)

        def fmt_cross(th):
            cg, noisy = crossings[th]
            if cg is None:
                return "never", None
            lag = cg - peak_it
            return f"{cg}{'*' if noisy else ''}", lag

        c5, l5 = fmt_cross(0.5)
        c3, l3 = fmt_cross(0.3)
        c1, l1 = fmt_cross(0.1)

        print(f"{name:10} {width:<6} {peak_it:<7} {ent_at_peak:<9.4f} "
              f"{c5:6} {str(l5):6} {c3:6} {str(l3):6} {c1:6} {str(l1):6} {ent_at_final:<10.4f}")

        all_data[name] = dict(
            width=width,
            peak_iter=peak_it,
            honest_peak=HONEST[name]["honest_peak"],
            honest_final=HONEST[name]["honest_final"],
            entropy_at_peak=ent_at_peak,
            entropy_at_final=ent_at_final,
            gen_at_final=gen_at_final,
            crossings={str(th): dict(gen=crossings[th][0], noisy=crossings[th][1]) for th in THRESHOLDS},
            lags={str(th): (crossings[th][0] - peak_it if crossings[th][0] is not None else None) for th in THRESHOLDS},
            generations=gens,
            ppo_entropy=entropy,
            success_rate=succ,
            vanilla_ppo_clears=clears,
            vanilla_ppo_max_x=maxx,
        )
        rows_for_summary.append((name, width, peak_it, ent_at_peak, l5, l3, l1, ent_at_final))

    out_path = REPO / "runs/peak_instability/entropy_collapse/entropy_trajectories.json"
    out_path.write_text(json.dumps(all_data, indent=2))
    print(f"\nwrote {out_path}")

    # ---- summary stats on the lag columns ----
    import statistics as stats
    print("\n=== lag summary (crossing_iter - peak_iter); positive = entropy still")
    print("=== above threshold AT the peak, crosses down AFTER; negative = already")
    print("=== below threshold BEFORE the peak was reached ===")
    for label, idx in [("0.5", 4), ("0.3", 5), ("0.1", 6)]:
        vals = [r[idx] for r in rows_for_summary if r[idx] is not None]
        if not vals:
            print(f"th={label}: no run crossed")
            continue
        print(f"th={label}: n={len(vals)}/8  vals={vals}  "
              f"median={stats.median(vals):.0f}  "
              f"min={min(vals)}  max={max(vals)}  "
              f"mean={stats.mean(vals):.1f}")


if __name__ == "__main__":
    main()
