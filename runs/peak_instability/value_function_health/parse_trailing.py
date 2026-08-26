import re, json
from pathlib import Path

RUNS = {
    "v27_seed0": "checkpoints/mario_1_1_v27_recovery_seed0",
    "v27_seed1": "checkpoints/mario_1_1_v27_recovery_seed1",
    "v27_seed2": "checkpoints/mario_1_1_v27_recovery_seed2",
    "v27_seed3": "checkpoints/mario_1_1_v27_recovery_seed3",
    "v28_seed0": "checkpoints/mario_1_1_v28_capacity_seed0",
    "v28_seed1": "checkpoints/mario_1_1_v28_capacity_seed1",
    "v28_seed2": "checkpoints/mario_1_1_v28_capacity_seed2",
    "v28_seed3": "checkpoints/mario_1_1_v28_capacity_seed3",
}

pat = re.compile(
    r"\[backward\] iter (\d+): .*trailing (\d+)/(\d+)=([\d.]+) \(advance"
)

out = {}
for name, path in RUNS.items():
    log_path = Path(path) / "run.log"
    series = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            it = int(m.group(1))
            trailing_rate = float(m.group(4))
            series[it] = trailing_rate  # last occurrence per iter wins (post force-complete not in log anyway)
    out[name] = series

Path("runs/peak_instability/value_function_health/trailing_series.json").write_text(json.dumps(out, indent=1))
for name, s in out.items():
    its = sorted(s.keys())
    print(name, "n=", len(its), "range", its[0], its[-1])
