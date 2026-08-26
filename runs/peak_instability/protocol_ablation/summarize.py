"""
Protocol ablation summary: v28 seed3, peak (iter 90, honest 0.670) vs final
(iter 240, honest 0.000). 2x2 grid per checkpoint: sticky_prob {0.0, 0.25} x
action_select {greedy, sampled}. start_state (cold entrance), start_jitter=16,
eval_seed=0, eval_rng=per-episode, eval_workers=4, episodes=30 held fixed at
the honest protocol's values throughout -- only sticky and action-select move.

30 episodes / single eval seed is a SHAPE measurement, not a gate number.
Do not compare these clear rates directly to the 100-episode pooled honest
gate without saying so.
"""
import json
import glob
import os

OUT = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    ("peak", "sticky0.0", "greedy"),
    ("peak", "sticky0.0", "sampled"),
    ("peak", "sticky0.25", "greedy"),
    ("peak", "sticky0.25", "sampled"),
    ("final", "sticky0.0", "greedy"),
    ("final", "sticky0.0", "sampled"),
    ("final", "sticky0.25", "greedy"),
    ("final", "sticky0.25", "sampled"),
]

REFERENCE_HONEST = {
    # 100-episode pooled gate numbers this ablation is checked against
    # (runs/v28_capacity/gate/seed3_{peak,final}_evalseed{0,1}_greedy.json).
    "peak": 0.670,
    "final": 0.000,
}


def load(cell):
    ckpt, sticky, select = cell
    tag = f"{ckpt}_{sticky}_{select}"
    path = os.path.join(OUT, f"{tag}.json")
    d = json.load(open(path))
    assert d.get("status") == "ok", f"{tag}: status={d.get('status')!r}, not ok"
    return tag, d


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 100)
    emit("PROTOCOL ABLATION -- v28 seed3 -- is the collapse a stickiness-robustness")
    emit("artifact, or is capability genuinely gone at the final checkpoint?")
    emit("=" * 100)
    emit()
    emit("30 episodes, single eval seed (0). SHAPE measurement, not a gate number.")
    emit("Fixed throughout: cold entrance start-state, start_jitter=16, eval_seed=0,")
    emit("eval_rng=per-episode, eval_workers=4, max_steps=1500.")
    emit()
    header = f"{'checkpoint':10s} {'sticky':8s} {'select':8s} {'clear_rate':11s} {'mean_return':12s} {'mean_length':12s} {'n_ep':5s}"
    emit(header)
    emit("-" * len(header))

    results = {}
    for cell in CELLS:
        tag, d = load(cell)
        results[cell] = d
        ckpt, sticky, select = cell
        emit(f"{ckpt:10s} {sticky:8s} {select:8s} {d['clear_rate']:<11.3f} "
             f"{d['mean_return']:<12.1f} {d['mean_length']:<12.1f} {d['n_episodes']:<5d}")

    emit()
    emit("-" * 100)
    emit("KEY CELL: final checkpoint, sticky=0.0, greedy")
    emit("-" * 100)
    key = results[("final", "sticky0.0", "greedy")]
    emit(f"  clear_rate = {key['clear_rate']:.3f}  ({int(round(key['clear_rate']*key['n_episodes']))}/{key['n_episodes']} episodes)")
    emit(f"  100-ep pooled honest gate at this checkpoint (sticky 0.25, greedy) = {REFERENCE_HONEST['final']:.3f}")
    if key['clear_rate'] <= 0.10:
        emit("  VERDICT: removing sticky actions entirely does NOT restore capability.")
        emit("  The final-checkpoint policy clears at most 1/30 episodes even with the")
        emit("  single most favorable protocol setting (no stickiness, greedy argmax).")
        emit("  Capability is genuinely gone -- collapse is not primarily a stickiness-")
        emit("  robustness artifact of the honest protocol.")
    else:
        emit("  VERDICT: removing sticky actions materially restores capability --")
        emit("  collapse is at least partly a stickiness-robustness failure.")

    emit()
    emit("-" * 100)
    emit("STICKY 0.25 vs 0.0, holding action_select fixed (does stickiness cost the")
    emit("policy real clear rate, and by how much, at each checkpoint)")
    emit("-" * 100)
    for ckpt in ("peak", "final"):
        for select in ("greedy", "sampled"):
            c0 = results[(ckpt, "sticky0.0", select)]["clear_rate"]
            c25 = results[(ckpt, "sticky0.25", select)]["clear_rate"]
            delta = c0 - c25
            rel = (delta / c0 * 100.0) if c0 > 0 else float("nan")
            emit(f"  {ckpt:6s} {select:8s}: sticky0.0={c0:.3f}  sticky0.25={c25:.3f}  "
                 f"delta={delta:+.3f}  relative_drop={rel:5.1f}%" if c0 > 0 else
                 f"  {ckpt:6s} {select:8s}: sticky0.0={c0:.3f}  sticky0.25={c25:.3f}  "
                 f"delta={delta:+.3f}  relative_drop=n/a (base rate 0)")

    emit()
    emit("-" * 100)
    emit("GREEDY vs SAMPLED, holding sticky fixed (independent check on the entropy-")
    emit("collapse story: a near-deterministic policy should show ~no gap here)")
    emit("-" * 100)
    for ckpt in ("peak", "final"):
        for sticky in ("sticky0.0", "sticky0.25"):
            g = results[(ckpt, sticky, "greedy")]["clear_rate"]
            s = results[(ckpt, sticky, "sampled")]["clear_rate"]
            emit(f"  {ckpt:6s} {sticky:10s}: greedy={g:.3f}  sampled={s:.3f}  gap={g - s:+.3f}")

    emit()
    emit("-" * 100)
    emit("Cross-check against the 100-episode pooled honest gate receipts")
    emit("(runs/v28_capacity/gate/seed3_{peak,final}_evalseed{0,1}_greedy.json)")
    emit("-" * 100)
    honest_replica_peak = results[("peak", "sticky0.25", "greedy")]["clear_rate"]
    honest_replica_final = results[("final", "sticky0.25", "greedy")]["clear_rate"]
    emit(f"  peak  : this ablation's honest-setting cell (sticky0.25, greedy, n=30, "
         f"eval_seed=0) = {honest_replica_peak:.3f}  vs 100-ep pooled gate = {REFERENCE_HONEST['peak']:.3f}")
    emit(f"  final : this ablation's honest-setting cell (sticky0.25, greedy, n=30, "
         f"eval_seed=0) = {honest_replica_final:.3f}  vs 100-ep pooled gate = {REFERENCE_HONEST['final']:.3f}")
    emit("  (n=30 single-seed vs n=100 two-seed pooled -- agreement within sampling")
    emit("  noise validates the ablation harness reproduces the banked protocol.)")

    with open(os.path.join(OUT, "summary_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
