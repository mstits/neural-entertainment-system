#!/usr/bin/env python3
"""Two supporting checks for the splitting-question writeup:

1. Episode-length triangulation: training's approx mean episode length
   (TOTAL_ENV_STEPS / episodes-completed-this-iter, a training-side-only
   derived quantity) vs the honest gate's directly-measured mean_length,
   at both peak and final checkpoints, for all 8 runs. Two independently
   computed length statistics from two disjoint measurement pipelines.

2. Confound audit for the four honest-vs-train protocol differences
   named in the assignment (sticky actions, jitter, cold start, greedy
   selection), read directly out of the registered training configs
   rather than assumed:
     - sticky_action_prob: 0.25 IS applied during training (configs/
       mario_1_1_v2{7,8}_seed*.yaml, "train under the protocol the gate
       measures") -- NOT a train/honest difference for these 8 runs.
     - jitter: the honest gate applies +-16 frames; training's backward
       curriculum instead draws restarts from a 160-frame window around
       tau -- a *training*-side position-randomization that is wider
       than honest's, not absent.
     - cold start: already established (at_entrance_iters.txt) that the
       curriculum reaches the true entrance (tau=0) by iter 22-29 in
       every run, hundreds of iterations before the peak-to-final
       comparison window -- not a live confound after iter ~30.
     - greedy vs sampled: training samples from the policy's categorical
       distribution; honest takes argmax. This IS a genuine, uncontrolled
       difference in these configs. Its predicted impact shrinks as
       training entropy collapses (reported by a sibling dimension at
       ~0.05-0.08 nats by iter 200+) because a near-one-hot categorical's
       argmax and its typical sample coincide with high probability --
       recorded here as a plausibility note, not re-measured.

Run: .venv/bin/python runs/peak_instability/training_signal_specialization/length_triangulation_and_confounds.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TOTAL_ENV_STEPS = 1024 * 60

comp = json.loads((ROOT / "runs/peak_instability/honest_vs_telemetry/comparison.json").read_text())

rows = []
for e in comp:
    tp_eps = e["telemetry_peak"]["episodes"]
    tf_eps = e["telemetry_final"]["episodes"]
    hp_len = e["honest_peak"]["mean_length"]
    hf_len = e["honest_final"]["mean_length"]
    train_len_peak = TOTAL_ENV_STEPS / tp_eps if tp_eps else None
    train_len_final = TOTAL_ENV_STEPS / tf_eps if tf_eps else None
    rows.append({
        "run": e["run"],
        "train_approx_len_peak": round(train_len_peak, 1) if train_len_peak else None,
        "honest_mean_len_peak": hp_len,
        "train_approx_len_final": round(train_len_final, 1) if train_len_final else None,
        "honest_mean_len_final": hf_len,
        "train_len_pct_change": round((train_len_final - train_len_peak) / train_len_peak * 100, 1) if train_len_peak else None,
        "honest_len_pct_change": round((hf_len - hp_len) / hp_len * 100, 1) if hp_len else None,
    })

confound_audit = {
    "sticky_action_prob": {
        "value": 0.25,
        "applied_in_training": True,
        "applied_in_honest_gate": True,
        "source": "configs/mario_1_1_v27_seed0.yaml:138, mario_1_1_v28_seed0.yaml:138 -- comment: 'train under the protocol the gate measures'",
        "verdict": "MATCHED -- not a train/honest divergence for these 8 runs",
    },
    "jitter": {
        "honest_gate_value": "+/-16 frames",
        "training_analog": "backward_curriculum.window_frames: 160 (restart drawn uniformly from a 160-frame window ending at tau)",
        "source": "configs/mario_1_1_v27_seed0.yaml:184 (window_frames: 160)",
        "verdict": "training's own position-randomization window is 10x WIDER than honest's jitter, not absent -- does not support a 'training is easier because unjittered' story",
    },
    "cold_start": {
        "at_entrance_iters": "all 8 runs reach tau=0 (true entrance) by iter 22-29",
        "peak_iters_range": "50-120",
        "verdict": "NOT a live confound for the peak-to-final comparison window (iter 30-240) -- curriculum is already maxed out for the entire window under study; see honest_vs_telemetry/at_entrance_iters.txt",
    },
    "greedy_vs_sampled": {
        "training": "categorical sample from policy logits",
        "honest_gate": "argmax",
        "verdict": "genuine, uncontrolled difference in these configs -- but a sibling dimension reports training entropy collapsing to ~0.05-0.08 nats by iter 200+, at which point a near-one-hot categorical's argmax and its typical sample coincide with high probability, shrinking (not eliminating) any greedy-specific effect late in training. Not independently re-measured here.",
    },
}

out = {"length_triangulation": rows, "confound_audit": confound_audit}
(HERE / "length_triangulation_and_confounds.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
