import csv

neg_fwd = neg_comp = pos_tp = missing = n = 0
with open("runs/peak_instability/reward_mix/reward_components_all_runs.csv") as fh:
    r = csv.DictReader(fh)
    for row in r:
        n += 1
        if row["reward_forward"] == "" or row["reward_completion"] == "" or row["reward_time_penalty"] == "":
            missing += 1
            continue
        f = float(row["reward_forward"]); c = float(row["reward_completion"]); t = float(row["reward_time_penalty"])
        if f < 0: neg_fwd += 1
        if c < 0: neg_comp += 1
        if t > 0: pos_tp += 1

print("rows checked:", n, "rows with a missing component:", missing)
print("reward_forward < 0 occurrences:      ", neg_fwd)
print("reward_completion < 0 occurrences:   ", neg_comp)
print("reward_time_penalty > 0 occurrences: ", pos_tp)
