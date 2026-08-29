import json, pathlib, subprocess, time
OUT = pathlib.Path("runs/v27_corrected_ladder")
STATE = "runs/live_show/smb_4_4_micro/entrance_start.state"
RUNS = [(f"mario_1_1_v27_recovery_seed{s}", f"configs/mario_1_1_v27_seed{s}.yaml") for s in range(4)]
ITERS = list(range(10, 250, 10))   # 24 checkpoints on the 10-iter grid
ok = bad = skip = 0; t0 = time.time()
for run, prof in RUNS:
    for it in ITERS:
        ck = pathlib.Path(f"checkpoints/{run}/vanilla_ppo_iter_{it:05d}.pt")
        if not ck.exists():
            continue
        for es in (0, 1):
            op = OUT / f"{run}_it{it:03d}_es{es}.json"
            if op.exists():
                try:
                    if json.load(open(op)).get("status") == "ok":
                        skip += 1; continue
                except Exception:
                    pass
            cmd = [".venv/bin/python", "scripts/eval_game.py", "--game", "mario",
                   "--profile", prof, "--checkpoint", str(ck), "--start-state", STATE,
                   "--episodes", "50", "--max-steps", "1500", "--sticky-prob", "0.25",
                   "--start-jitter", "16", "--action-select", "greedy",
                   "--eval-seed", str(es), "--eval-workers", "8", "--eval-rng", "per-episode"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            op.write_text(r.stdout)
            try:
                d = json.loads(r.stdout)
                if d.get("status") == "ok":
                    ok += 1
                    print(f"ok {run} it{it} es{es} clear={d['clear_rate']}", flush=True)
                else:
                    bad += 1
                    print(f"BAD {run} it{it} es{es}: {d.get('status')}", flush=True)
            except Exception:
                bad += 1
                print(f"BAD {run} it{it} es{es}: UNPARSEABLE {r.stdout[:100]}", flush=True)
print(f"\nF0 DONE ok={ok} bad={bad} skipped={skip} elapsed={time.time()-t0:.0f}s", flush=True)
