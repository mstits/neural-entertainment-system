import csv
from collections import defaultdict

rows = defaultdict(list)
with open("runs/peak_instability/reward_mix/reward_components_all_runs.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows[(row["run"], row["seed"])].append(row)

def f(x):
    return float(x) if x not in (None, "", "None") else None

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")

print(f"{'run/seed':12} {'peak':>5} {'net_pre':>9} {'net_post':>9} {'fwd_pre':>9} {'fwd_post':>9} {'sh_tp_pre':>10} {'sh_tp_post':>11}")
for (run, seed), rs in rows.items():
    peak = int(rs[0]["peak_iter"])
    pre = [row for row in rs if int(row["generation"]) < peak]
    post = [row for row in rs if int(row["generation"]) >= peak]
    net_pre = mean([f(x["net_logged_reward"]) for x in pre])
    net_post = mean([f(x["net_logged_reward"]) for x in post])
    fwd_pre = mean([f(x["reward_forward"]) for x in pre])
    fwd_post = mean([f(x["reward_forward"]) for x in post])
    sh_pre = mean([f(x["share_time_penalty"]) for x in pre])
    sh_post = mean([f(x["share_time_penalty"]) for x in post])
    print(f"{run+'/'+seed:12} {peak:>5} {net_pre:>9.1f} {net_post:>9.1f} {fwd_pre:>9.1f} {fwd_post:>9.1f} {sh_pre:>10.4f} {sh_post:>11.4f}")
