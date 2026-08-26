import csv
from collections import defaultdict

rows = defaultdict(list)
with open("runs/peak_instability/reward_mix/reward_components_all_runs.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows[(row["run"], row["seed"])].append(row)

print(f"{'run/seed':12} {'peak':>5} {'pre_missing/total':>20} {'post_missing/total':>20}")
tot_pre_missing = tot_pre = tot_post_missing = tot_post = 0
for (run, seed), rs in rows.items():
    peak = int(rs[0]["peak_iter"])
    pre = [row for row in rs if int(row["generation"]) < peak]
    post = [row for row in rs if int(row["generation"]) >= peak]
    pre_missing = sum(1 for row in pre if row["reward_completion"] == "")
    post_missing = sum(1 for row in post if row["reward_completion"] == "")
    tot_pre_missing += pre_missing; tot_pre += len(pre)
    tot_post_missing += post_missing; tot_post += len(post)
    print(f"{run+'/'+seed:12} {peak:>5} {f'{pre_missing}/{len(pre)}':>20} {f'{post_missing}/{len(post)}':>20}")

print()
print(f"TOTAL pre-peak:  {tot_pre_missing}/{tot_pre} = {100*tot_pre_missing/tot_pre:.1f}%")
print(f"TOTAL post-peak: {tot_post_missing}/{tot_post} = {100*tot_post_missing/tot_post:.1f}%")
