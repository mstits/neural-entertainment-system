import json, pathlib, subprocess, sys, time
REPO = pathlib.Path(".")
OUT = REPO/"runs/peak_instability/p1_falsifier"
STATE = "runs/live_show/smb_4_4_micro/entrance_start.state"
RUNS = [
 ("mario_1_1_v27_recovery_seed0","configs/mario_1_1_v27_seed0.yaml",60),
 ("mario_1_1_v27_recovery_seed1","configs/mario_1_1_v27_seed1.yaml",50),
 ("mario_1_1_v27_recovery_seed2","configs/mario_1_1_v27_seed2.yaml",90),
 ("mario_1_1_v27_recovery_seed3","configs/mario_1_1_v27_seed3.yaml",60),
 ("mario_1_1_v28_capacity_seed0","configs/mario_1_1_v28_seed0.yaml",70),
 ("mario_1_1_v28_capacity_seed1","configs/mario_1_1_v28_seed1.yaml",60),
 ("mario_1_1_v28_capacity_seed2","configs/mario_1_1_v28_seed2.yaml",120),
 ("mario_1_1_v28_capacity_seed3","configs/mario_1_1_v28_seed3.yaml",90),
]
done=0; bad=0; t0=time.time()
for run, prof, pk in RUNS:
    for it in (pk, pk+20, pk+40, pk+60):
        ck = f"checkpoints/{run}/vanilla_ppo_iter_{it:05d}.pt"
        for sel in ("greedy","sampled"):
            tag = f"{run}_it{it}_{sel}"
            op = OUT/f"{tag}.json"
            if op.exists():
                try:
                    if json.load(open(op)).get("status")=="ok": done+=1; continue
                except Exception: pass
            cmd = [".venv/bin/python","scripts/eval_game.py","--game","mario",
                   "--profile",prof,"--checkpoint",ck,"--start-state",STATE,
                   "--episodes","50","--max-steps","1500","--sticky-prob","0.25",
                   "--start-jitter","16","--eval-seed","0","--action-select",sel,
                   "--eval-workers","6","--eval-rng","per-episode"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            try: op.write_text(r.stdout)
            except Exception: pass
            try:
                st = json.loads(r.stdout).get("status")
            except Exception:
                st = "UNPARSEABLE"
            if st != "ok":
                bad+=1; print(f"BAD {tag}: {st} :: {r.stdout[:160]} {r.stderr[-160:]}", flush=True)
            else:
                done+=1
                print(f"ok {tag}: clear={json.loads(r.stdout).get('clear_rate')}", flush=True)
print(f"\nP1 DONE ok={done} bad={bad} elapsed={time.time()-t0:.0f}s", flush=True)
