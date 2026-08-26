import json, glob, os

OUT = "/Users/stits/Documents/macos-emulation-and-training/runs/peak_instability/decay_curve"

def load(seed):
    rows = []
    for path in sorted(glob.glob(f"{OUT}/seed{seed}_iter*.json")):
        d = json.load(open(path))
        it = int(os.path.basename(path).split("iter")[1].split(".")[0])
        rows.append((it, d))
    rows.sort()
    return rows

for seed, peak in [(3, 90), (0, 70)]:
    print(f"\n=== seed{seed} (peak iter {peak}) ===")
    rows = load(seed)
    for it, d in rows:
        status = d.get("status")
        cr = d.get("clear_rate")
        ml = d.get("mean_length")
        mr = d.get("mean_return")
        n = d.get("n_episodes")
        tag = ""
        if it == peak:
            tag = "  <-- PEAK"
        elif it == peak - 10:
            tag = "  <-- peak-10"
        elif it == peak + 10:
            tag = "  <-- peak+10"
        print(f"iter={it:4d}  status={status:6s}  n={n}  clear_rate={cr}  mean_return={mr}  mean_len={ml}{tag}")
