import csv
from collections import defaultdict

rows = defaultdict(list)
with open("runs/peak_instability/reward_mix/reward_components_all_runs.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows[(row["run"], row["seed"])].append(row)

def f(x):
    return float(x) if x not in (None, "", "None") else None

print(f"{'run/seed':12} {'peak':>5} {'max_sh_tp':>10} {'@iter':>6} {'min_sh_tp':>10} {'@iter':>6} {'is_min@peak':>12} {'sh_tp@peak+10':>14} {'sh_tp@peak+40':>14}")
for (run, seed), rs in rows.items():
    rs_sorted = sorted(rs, key=lambda x: int(x["generation"]))
    by_gen = {int(x["generation"]): x for x in rs_sorted}
    peak_iter = int(rs_sorted[0]["peak_iter"])
    vals = [(int(x["generation"]), f(x["share_time_penalty"])) for x in rs_sorted if f(x["share_time_penalty"]) is not None]
    max_g, max_v = max(vals, key=lambda t: t[1])
    min_g, min_v = min(vals, key=lambda t: t[1])
    sh_peak = f(by_gen[peak_iter]["share_time_penalty"]) if peak_iter in by_gen else None
    is_min_at_peak = abs(min_g - peak_iter) <= 10
    def near(it):
        # nearest logged generation to it (metrics logged every gen 0..249 presumably every iter given 250 rows/250 iters -> every gen present)
        return by_gen.get(it)
    g10 = near(peak_iter+10); g40 = near(peak_iter+40)
    sh10 = f(g10["share_time_penalty"]) if g10 else None
    sh40 = f(g40["share_time_penalty"]) if g40 else None
    print(f"{run+'/'+seed:12} {peak_iter:>5} {max_v:>10.4f} {max_g:>6} {min_v:>10.4f} {min_g:>6} {str(is_min_at_peak):>12} "
          f"{(sh10 if sh10 is not None else float('nan')):>14.4f} {(sh40 if sh40 is not None else float('nan')):>14.4f}")
