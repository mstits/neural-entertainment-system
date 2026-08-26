import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

RUNS = {
    "v27_seed0": dict(width="48k", ckpt="mario_1_1_v27_recovery_seed0", log="runs/v27_fresh_recovery/train_seed0.log"),
    "v27_seed1": dict(width="48k", ckpt="mario_1_1_v27_recovery_seed1", log="runs/v27_fresh_recovery/train_seed1.log"),
    "v27_seed2": dict(width="48k", ckpt="mario_1_1_v27_recovery_seed2", log="runs/v27_fresh_recovery/train_seed2.log"),
    "v27_seed3": dict(width="48k", ckpt="mario_1_1_v27_recovery_seed3", log="runs/v27_fresh_recovery/train_seed3.log"),
    "v28_seed0": dict(width="72k", ckpt="mario_1_1_v28_capacity_seed0", log="runs/v28_capacity/train_seed0.log"),
    "v28_seed1": dict(width="72k", ckpt="mario_1_1_v28_capacity_seed1", log="runs/v28_capacity/train_seed1.log"),
    "v28_seed2": dict(width="72k", ckpt="mario_1_1_v28_capacity_seed2", log="runs/v28_capacity/train_seed2.log"),
    "v28_seed3": dict(width="72k", ckpt="mario_1_1_v28_capacity_seed3", log="runs/v28_capacity/train_seed3.log"),
}

LINE_RE = re.compile(
    r"\[backward\] iter (\d+): tau=(\d+)/(\d+).*?trailing (\d+)/(\d+)=([\d.]+).*?"
    r"(AT-ENTRANCE)?\s*\|\s*entrance (\d+)/(\d+)=([\d.]+) \| truncated (\d+)"
)

def parse_log(path):
    rows = {}
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            it = int(m.group(1))
            tau = int(m.group(2))
            tau_max = int(m.group(3))
            trail_n = int(m.group(4)); trail_d = int(m.group(5)); trail_rate = float(m.group(6))
            at_entrance = m.group(7) is not None
            ent_n = int(m.group(8)); ent_d = int(m.group(9)); ent_cum_rate = float(m.group(10))
            trunc = int(m.group(11))
            rows[it] = dict(tau=tau, tau_max=tau_max, trail_n=trail_n, trail_d=trail_d,
                             trail_rate=trail_rate, at_entrance=at_entrance,
                             ent_cum_n=ent_n, ent_cum_d=ent_d, ent_cum_rate=ent_cum_rate,
                             truncated=trunc)
    return rows

def parse_metrics(path):
    rows = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows[int(d["generation"])] = d
    return rows

def load_best(path):
    with open(path) as f:
        return json.load(f)

out = {}
for name, cfg in RUNS.items():
    ckpt_dir = REPO / "checkpoints" / cfg["ckpt"]
    log_path = REPO / cfg["log"]
    metrics = parse_metrics(ckpt_dir / "metrics.jsonl")
    logrows = parse_log(log_path)
    best = load_best(ckpt_dir / "winners" / "best.json")
    out[name] = dict(width=cfg["width"], metrics=metrics, log=logrows, best=best)

json.dump(
    {k: {"width": v["width"], "best": v["best"],
         "metrics": {str(i): m for i, m in v["metrics"].items()},
         "log": {str(i): l for i, l in v["log"].items()}}
     for k, v in out.items()},
    open(REPO / "runs/peak_instability/entropy_collapse_refutation/raw_extract.json", "w"),
)
print("wrote raw_extract.json;", {k: (len(v["metrics"]), len(v["log"])) for k, v in out.items()})
