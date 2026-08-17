# World 2-1 — in-window runbook (staged 2026-08-17)

Assets verified before staging: 4 banked solver tapes in
`runs/ge_2_1_solve/solutions/` (763/748/737/721 steps), all rooted at
`checkpoints/ge_entrances/smb_2_1_entrance.state` (21,164 bytes, present),
all `start_wd [1,0] -> clear_wd [1,1]` — a level-advance clear, same
predicate family as 1-3. Configs staged: `configs/mario_2_1_online_v1.yaml`
and `configs/campaign_2_1.yaml` (zero residual 1-4 references).

Run these in order once the emulator is free. Times are from the 1-3 and
1-4 windows, which followed the identical sequence.

    # W1 — mint the backward ladder (~2 min; aborts unless the tape
    #      replays to its banked clear, which IS the verification)
    .venv/bin/python scripts/mint_backward_states.py --level 2-1 \
        --run runs/ge_2_1_solve \
        --profile configs/mario_2_1_online_v1.yaml \
        --out checkpoints/backward_states/2-1_online

    # W2 — select the 6 restart rungs (~10 s)
    .venv/bin/python scripts/select_restart_states.py \
        --ladder checkpoints/backward_states/2-1_online \
        --out checkpoints/online_2_1/restart_states --auto-targets 6

    # W3 — wavefront dmap (~3 min). Acceptance: D_start below the tape
    #      length (min steps-to-goal per cell sits under it); 1-3 read
    #      455 against a 540 tape, 1-4 read 456 against 490.
    .venv/bin/python -m src.utils.wavefront_reward \
        --solutions runs/ge_2_1_solve/solutions/sol_00{0,1,2,3}.actions.npy \
        --root-state checkpoints/ge_entrances/smb_2_1_entrance.state \
        --profile configs/mario_2_1_online_v1.yaml \
        --rom "roms/Super Mario Bros. (World).nes" \
        --out checkpoints/wavefront/mario_2_1_dmap.pkl

    # W4 — BC demos, one per tape (~2 min). EXPECT SOME TO FAIL: 2 of 8
    #      tapes across 1-3 and 1-4 did not replay from their own root
    #      (final gx 0). Quarantine any that do; train on the rest.
    for i in 000 001 002 003; do
      .venv/bin/python scripts/replay_to_demos.py \
        --start-state checkpoints/ge_entrances/smb_2_1_entrance.state \
        --actions runs/ge_2_1_solve/solutions/sol_$i.actions.npy \
        --profile configs/mario_2_1_online_v1.yaml \
        --out checkpoints/bc_2_1/demos/demos_2_1_sol_$i.npz --root-id entrance
    done

    # W5 — BC anchor (~3 min)
    .venv/bin/python scripts/bc_distill.py \
        --demos "checkpoints/bc_2_1/demos/demos_2_1_sol_*.npz" \
        --profile configs/mario_2_1_online_v1.yaml \
        --out checkpoints/bc_2_1/anchor_h256 \
        --hidden-dim 256 --trunk-dim 64 --epochs 120 --lr 1e-3 --seed 0

    # W6 — MUST NOT SKIP: measure the anchor's honest median and set the
    #      competence floor to 0.8x it in configs/campaign_2_1.yaml
    #      (1-2: 150=0.8x187 | 1-3: 346=0.8x433 | 1-4: 477=0.8x596).
    #      The config ships with 0.0 (disarmed) precisely so an
    #      uncalibrated floor cannot ride into a launch unnoticed.
    .venv/bin/python scripts/eval_game.py --game mario_2_1_online_v1 \
        --profile configs/mario_2_1_online_v1.yaml \
        --rom "roms/Super Mario Bros. (World).nes" \
        --checkpoint checkpoints/bc_2_1/anchor_h256/vanilla_ppo_iter_00000.pt \
        --episodes 30 --max-steps 3000 --sequential --level-clear \
        --start-state checkpoints/ge_entrances/smb_2_1_entrance.state \
        --eval-seed 20260817 --sticky-prob 0.25 --start-jitter 16 \
        --eval-workers 5 --eval-rng per-episode

    # W7 — dry run (all checks must pass) then W8 launch
    .venv/bin/python scripts/run_online_campaign.py \
        --campaign-config configs/campaign_2_1.yaml --dry-run
    .venv/bin/python scripts/run_online_campaign.py \
        --campaign-config configs/campaign_2_1.yaml

Expected shape from the three prior campaigns: phase-0 critic gate at
the 5M floor, phase-1 rung gate 10/10 (1-3 first probe, 1-4 second),
phase 2 budget-complete, reverse walk reaching the entrance, then the
rate arriving in consolidation. Preserve on every peak probe — peaks
have been transient in all three campaigns, and preserve-on-peak is what
saved 1-2's 38% and 1-3's 21%.
