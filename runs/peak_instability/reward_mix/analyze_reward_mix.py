import csv
from collections import defaultdict

rows = defaultdict(list)
with open("runs/peak_instability/reward_mix/reward_components_all_runs.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        key = (row["run"], row["seed"])
        rows[key].append(row)

def f(x):
    return float(x) if x not in (None, "", "None") else None

print(f"{'run/seed':12} {'peak':>5} {'sh_tp@0':>8} {'sh_tp@peak':>10} {'sh_tp@240':>9} {'sh_fwd@0':>8} {'sh_fwd@peak':>11} {'sh_fwd@240':>10} {'net@0':>9} {'net@peak':>9} {'net@240':>9}")
summary = []
for (run, seed), rs in rows.items():
    rs_sorted = sorted(rs, key=lambda x: int(x["generation"]))
    peak_iter = int(rs_sorted[0]["peak_iter"])
    by_gen = {int(x["generation"]): x for x in rs_sorted}
    g0 = by_gen.get(0)
    gpeak = by_gen.get(peak_iter)
    g240 = by_gen.get(240) or rs_sorted[-1]
    def get(g, k):
        v = f(g[k]) if g else None
        return v
    row_out = dict(
        run=run, seed=seed, peak=peak_iter,
        sh_tp_0=get(g0,"share_time_penalty"), sh_tp_peak=get(gpeak,"share_time_penalty"), sh_tp_240=get(g240,"share_time_penalty"),
        sh_fwd_0=get(g0,"share_forward"), sh_fwd_peak=get(gpeak,"share_forward"), sh_fwd_240=get(g240,"share_forward"),
        net_0=get(g0,"net_logged_reward"), net_peak=get(gpeak,"net_logged_reward"), net_240=get(g240,"net_logged_reward"),
    )
    summary.append(row_out)
    print(f"{run+'/'+seed:12} {peak_iter:>5} {row_out['sh_tp_0']:>8.3f} {row_out['sh_tp_peak']:>10.3f} {row_out['sh_tp_240']:>9.3f} "
          f"{row_out['sh_fwd_0']:>8.3f} {row_out['sh_fwd_peak']:>11.3f} {row_out['sh_fwd_240']:>10.3f} "
          f"{row_out['net_0']:>9.2f} {row_out['net_peak']:>9.2f} {row_out['net_240']:>9.2f}")

print()
print("=== monotonicity check: does share_time_penalty trend upward over the FULL run? ===")
for (run, seed), rs in rows.items():
    rs_sorted = sorted(rs, key=lambda x: int(x["generation"]))
    gens = [int(x["generation"]) for x in rs_sorted]
    shares = [f(x["share_time_penalty"]) for x in rs_sorted]
    valid = [(g,s) for g,s in zip(gens,shares) if s is not None]
    # simple correlation of gen vs share
    n = len(valid)
    gs = [v[0] for v in valid]; ss = [v[1] for v in valid]
    mg = sum(gs)/n; ms = sum(ss)/n
    cov = sum((g-mg)*(s-ms) for g,s in valid)
    vg = sum((g-mg)**2 for g in gs)
    vs = sum((s-ms)**2 for s in ss)
    corr = cov / (vg*vs)**0.5 if vg>0 and vs>0 else float('nan')
    print(f"{run}/{seed}: pearson r(gen, share_time_penalty) = {corr:+.3f}  (n={n})")
