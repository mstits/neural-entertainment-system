#!/usr/bin/env python3
"""
Mechanism read #3 (v28 capacity registration): ladder telemetry vs v27,
seed-paired.

Definition, taken verbatim from docs/proposals/V28_CAPACITY_2026-08-25.md
"Mechanism reads" item 3:

    "Ladder telemetry vs. v27: iters-to-entrance, rungs/100 iters,
    entrance-rate trajectory, compared seed-for-seed against v27's own
    recorded numbers for the same seed."

Parses the per-iteration "[backward] iter N: tau=X/784 ... trailing
A/30=R ... advances=K [ AT-ENTRANCE] | entrance E/T=F | truncated U"
lines emitted by src/training/trainer.py (see the log-line format at
trainer.py's backward-curriculum telemetry block, and the field
semantics in src/training/backward_curriculum.py's BackwardScheduler
snapshot()/record()/record_entrance()).

Field semantics that matter for this read (verified by reading
backward_curriculum.py, not assumed):

  - tau=X/N            : ladder cursor. X counts DOWN from a per-seed
                          starting rung to 0. tau==0 ("AT-ENTRANCE") is
                          the true level entrance.
  - trailing A/B=R      : self._window, a deque(maxlen=trailing) scored
                          against whichever rung tau CURRENTLY sits on.
                          It resets every time tau advances (maybe_advance
                          clears the window). Once tau==0, this window is
                          scored ENTIRELY against entrance attempts, so
                          post-AT-ENTRANCE this field IS the windowed
                          ("trailing") entrance success rate -- the same
                          quantity src/training/trainer.py's
                          _select_winner_metric() reads (under the name
                          "entrance_trailing_rate") to pick winners/best.pt.
                          This is the right series for "entrance-rate
                          trajectory", not the next field.
  - entrance E/T=F      : self._ent_succ / self._ent_att, a CUMULATIVE,
                          unwindowed counter that starts accumulating the
                          iter tau first hits 0 and never resets or
                          decays. It is monotonically diluted by episode
                          count and does not describe "current" policy
                          performance at the entrance -- reported here
                          only as supporting context, not as the primary
                          entrance-rate series.

Per seed, per version (v27, v28), computes:
  (a) iters_to_entrance -- the iter of the FIRST log line carrying
      tau==0 (equivalently the first AT-ENTRANCE marker). Undefined
      (None) if the run never reaches tau==0.
  (b) rungs_per_100_iters -- initial_tau / iters_to_entrance * 100,
      i.e. the average ladder-descent rate needed to explain reaching
      the entrance in that many iters. Also a binned (25-iter windows)
      trajectory of the same rate, to show whether descent is roughly
      constant, front-loaded, or back-loaded.
  (c) entrance_rate_trajectory -- the trailing-window rate series
      (from field "trailing", NOT "entrance") over all iters from
      iters_to_entrance to the end of the run. Reports the peak value
      (ties -> later iter, matching the project's own winner-selection
      tie-break convention) and the mean over the last 30 and last 50
      logged entrance-era iters, plus the full series for shape
      inspection.

Then prints a seed-paired v27-vs-v28 comparison and a per-seed
IMPROVING / WORSE / FLAT-AMBIGUOUS call, plus the count of seeds that
improved out of 4.
"""
import json
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"\[backward\] iter (?P<iter>\d+): "
    r"tau=(?P<tau>\d+)/(?P<n>\d+) "
    r"\(step (?P<step>\d+) frame (?P<frame>\d+) gx (?P<gx>-?\d+)\) "
    r"trailing (?P<t_succ>\d+)/(?P<t_att>\d+)=(?P<t_rate>[\d.]+) "
    r"\(advance at >=[\d.]+ over \d+\) "
    r"advances=(?P<advances>\d+)"
    r"(?P<at_entrance>  AT-ENTRANCE)? \| "
    r"entrance (?P<e_succ>\d+)/(?P<e_att>\d+)=(?P<e_rate>[\d.]+) \| "
    r"truncated (?P<trunc>\d+)"
)


def parse_log(path: Path):
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "iter": int(d["iter"]),
                "tau": int(d["tau"]),
                "n": int(d["n"]),
                "trailing_succ": int(d["t_succ"]),
                "trailing_att": int(d["t_att"]),
                "trailing_rate": float(d["t_rate"]),
                "advances": int(d["advances"]),
                "at_entrance": d["at_entrance"] is not None,
                "entrance_succ": int(d["e_succ"]),
                "entrance_att": int(d["e_att"]),
                "entrance_rate_cum": float(d["e_rate"]),
                "truncated": int(d["trunc"]),
            })
    return rows


def rungs_binned_trajectory(rows, bin_size=25):
    """Rungs descended per 100 iters, in bin_size-iter windows, until
    tau first reaches 0 (inclusive of the bin it lands in)."""
    if not rows:
        return []
    out = []
    max_iter = rows[-1]["iter"]
    by_iter = {r["iter"]: r["tau"] for r in rows}
    iters_sorted = sorted(by_iter)
    lo = iters_sorted[0]
    prev_tau = by_iter[lo]
    bin_start = lo
    reached_zero = False
    for hi in range(lo + bin_size, max_iter + bin_size, bin_size):
        # tau at the last logged iter <= hi-1 (or max_iter)
        candidates = [i for i in iters_sorted if bin_start <= i < hi]
        if not candidates:
            break
        end_iter = candidates[-1]
        end_tau = by_iter[end_iter]
        delta = prev_tau - end_tau
        span = end_iter - bin_start if end_iter != bin_start else bin_size
        span = max(span, 1)
        rate = delta / span * 100.0
        out.append({
            "bin_start_iter": bin_start,
            "bin_end_iter": end_iter,
            "tau_start": prev_tau,
            "tau_end": end_tau,
            "rungs_descended": delta,
            "rungs_per_100_iters": round(rate, 2),
        })
        prev_tau = end_tau
        bin_start = end_iter + 1
        if end_tau == 0:
            reached_zero = True
            break
    return out


def analyze_seed(rows, seed, version):
    if not rows:
        return {"seed": seed, "version": version, "error": "no backward lines parsed"}
    rows = sorted(rows, key=lambda r: r["iter"])
    initial_tau = rows[0]["tau"]
    n_entries = rows[0]["n"]
    at_entrance_rows = [r for r in rows if r["at_entrance"]]
    if not at_entrance_rows:
        return {
            "seed": seed, "version": version,
            "initial_tau": initial_tau, "n_entries": n_entries,
            "iters_to_entrance": None,
            "note": "run never reached tau=0 (AT-ENTRANCE) in this log",
        }
    iters_to_entrance = at_entrance_rows[0]["iter"]
    rungs_per_100_avg = (
        (initial_tau / iters_to_entrance) * 100.0 if iters_to_entrance > 0
        else None
    )
    binned = rungs_binned_trajectory(rows, bin_size=25)

    # entrance-rate trajectory: the "trailing" (windowed) field from the
    # first AT-ENTRANCE iter through the end of the run.
    ent_series = [
        {"iter": r["iter"], "trailing_succ": r["trailing_succ"],
         "trailing_att": r["trailing_att"], "trailing_rate": r["trailing_rate"]}
        for r in rows if r["iter"] >= iters_to_entrance
    ]
    # peak, ties -> later iter. NOTE: this is a LOWER BOUND on the true
    # in-training peak the winner-selection logic sees -- see
    # LOG-VS-CHECKPOINT DISCREPANCY note in main(): a force-completion
    # scoring pass runs AFTER the per-iter "[backward]" line is printed
    # (trainer.py records into the SAME window again before the retention
    # block reads it), so the live winners/best.json sidecar the trainer
    # itself wrote can show a higher value at the same iter than the log
    # line captured. Both v27 and v28 are subject to the identical code
    # path, so this does not bias the SEED-PAIRED direction call, but it
    # does mean this number is not the exact winner-selection figure.
    peak_rate = max(e["trailing_rate"] for e in ent_series)
    peak_iter = max(e["iter"] for e in ent_series if e["trailing_rate"] == peak_rate)
    # sustained-quality: fraction of entrance-era logged iters at/above
    # 0.5 -- robust to a single saturated point-peak (both v27 and v28
    # peak near/at the 0.87-1.00 ceiling per the authoritative
    # winners/best.json values; peak alone barely discriminates them).
    frac_ge_050 = sum(1 for e in ent_series if e["trailing_rate"] >= 0.50) / len(ent_series)
    frac_ge_030 = sum(1 for e in ent_series if e["trailing_rate"] >= 0.30) / len(ent_series)

    def mean_last_n(series, n):
        tail = series[-n:] if len(series) >= 1 else []
        tail = series[-n:]
        if not tail:
            return None
        return sum(e["trailing_rate"] for e in tail) / len(tail)

    def mean_first_n(series, n):
        head = series[:n]
        if not head:
            return None
        return sum(e["trailing_rate"] for e in head) / len(head)

    last_iter = rows[-1]["iter"]
    final_row = rows[-1]
    return {
        "seed": seed,
        "version": version,
        "initial_tau": initial_tau,
        "n_entries": n_entries,
        "total_logged_iters": len(rows),
        "last_iter": last_iter,
        # (a)
        "iters_to_entrance": iters_to_entrance,
        # (b)
        "rungs_per_100_iters_avg": round(rungs_per_100_avg, 2) if rungs_per_100_avg else None,
        "rungs_binned_trajectory": binned,
        # (c)
        "entrance_era_n_logged_iters": len(ent_series),
        "entrance_trailing_rate_series": ent_series,
        "entrance_trailing_rate_peak": peak_rate,
        "entrance_trailing_rate_peak_iter": peak_iter,
        "entrance_trailing_rate_frac_ge_050": round(frac_ge_050, 4),
        "entrance_trailing_rate_frac_ge_030": round(frac_ge_030, 4),
        "entrance_trailing_rate_mean_first30": mean_first_n(ent_series, 30),
        "entrance_trailing_rate_mean_last30": mean_last_n(ent_series, 30),
        "entrance_trailing_rate_mean_last50": mean_last_n(ent_series, 50),
        "entrance_trailing_rate_mean_all": (
            sum(e["trailing_rate"] for e in ent_series) / len(ent_series)
            if ent_series else None
        ),
        # cumulative entrance rate, context only (not the primary series)
        "cumulative_entrance_rate_final": final_row["entrance_rate_cum"],
        "cumulative_entrance_final_frac": f"{final_row['entrance_succ']}/{final_row['entrance_att']}",
        "advances_at_entrance": at_entrance_rows[0]["advances"],
        "truncated_final": final_row["truncated"],
    }


def direction(v27_val, v28_val, higher_is_better, tol_frac=0.0, tol_abs=0.0):
    """Return 'improve' / 'worse' / 'flat' given a tolerance band.

    tol_frac is relative to |v27_val|; tol_abs is an absolute floor on
    top of it (needed for rate metrics estimated off small trailing
    windows, where sub-tolerance deltas are indistinguishable from
    sampling noise -- see NOISE FLOOR note in main()).
    """
    if v27_val is None or v28_val is None:
        return "unknown"
    if v27_val == 0 and v28_val == 0:
        return "flat"
    diff = v28_val - v27_val
    band = max(tol_frac * max(abs(v27_val), 1e-9), tol_abs)
    if higher_is_better:
        if diff > band:
            return "improve"
        elif diff < -band:
            return "worse"
        return "flat"
    else:
        if diff < -band:
            return "improve"
        elif diff > band:
            return "worse"
        return "flat"


def load_winner_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    root = Path(__file__).resolve().parents[3]  # repo root from runs/v28_capacity/mechanism_reads
    v28_dir = root / "runs" / "v28_capacity"
    v27_dir = root / "runs" / "v27_fresh_recovery"
    v27_ckpt_dir = root / "checkpoints"
    v28_ckpt_dir = root / "checkpoints"

    per_seed = {}
    for seed in range(4):
        v28_rows = parse_log(v28_dir / f"train_seed{seed}.log")
        v27_rows = parse_log(v27_dir / f"train_seed{seed}.log")
        a28 = analyze_seed(v28_rows, seed, "v28")
        a27 = analyze_seed(v27_rows, seed, "v27")
        # Authoritative peak entrance_trailing_rate, straight from the
        # trainer's own winners/best.json sidecar (ground truth -- this
        # is the EXACT value/iter the retention block selected on, not a
        # log reconstruction). See LOG-VS-CHECKPOINT DISCREPANCY:
        # cross-checking one example (v27 seed0) found the sidecar
        # (0.8667 @ iter 60) disagrees with the SAME iteration's printed
        # "[backward] iter 60" trailing field (0.53) -- trainer.py runs a
        # second force-completion record() pass after the telemetry line
        # prints and before the retention block reads the window, so the
        # sidecar sees additional scored episodes the log line doesn't.
        # Both versions share this code path, so it's not a v27-vs-v28
        # bias, but the sidecar is the more authoritative single number
        # for "peak" while the log series remains the only source for
        # trajectory shape / mean / iters-to-entrance.
        w27 = load_winner_json(v27_ckpt_dir / f"mario_1_1_v27_recovery_seed{seed}" / "winners" / "best.json")
        w28 = load_winner_json(v28_ckpt_dir / f"mario_1_1_v28_capacity_seed{seed}" / "winners" / "best.json")
        a27["authoritative_peak"] = w27["metric_value"] if w27 else None
        a27["authoritative_peak_iter"] = w27["source_iter"] if w27 else None
        a28["authoritative_peak"] = w28["metric_value"] if w28 else None
        a28["authoritative_peak_iter"] = w28["source_iter"] if w28 else None
        per_seed[seed] = {"v27": a27, "v28": a28}

    out_path = Path(__file__).resolve().parent / "read3_raw.json"
    with open(out_path, "w") as fh:
        json.dump(per_seed, fh, indent=2)

    print(f"Wrote raw per-seed data to {out_path}")
    print()

    verdicts = {}
    print("=" * 100)
    print("READ #3 -- LADDER TELEMETRY, v28 vs v27, SEED-PAIRED")
    print("=" * 100)
    for seed in range(4):
        a27 = per_seed[seed]["v27"]
        a28 = per_seed[seed]["v28"]
        print(f"\n--- seed {seed} ---")
        print(f"  initial_tau: v27={a27.get('initial_tau')}  v28={a28.get('initial_tau')}  "
              f"(n_entries: v27={a27.get('n_entries')} v28={a28.get('n_entries')})")

        # (a) iters-to-entrance
        it27 = a27.get("iters_to_entrance")
        it28 = a28.get("iters_to_entrance")
        dir_a = direction(it27, it28, higher_is_better=False, tol_frac=0.0)
        print(f"  (a) iters_to_entrance:      v27={it27:>4}   v28={it28:>4}   "
              f"delta={it28 - it27:+d}   -> {dir_a}")

        # (b) rungs/100 iters (avg)
        r27 = a27.get("rungs_per_100_iters_avg")
        r28 = a28.get("rungs_per_100_iters_avg")
        dir_b = direction(r27, r28, higher_is_better=True, tol_frac=0.0)
        print(f"  (b) rungs_per_100_iters:    v27={r27:>7.2f}  v28={r28:>7.2f}  "
              f"delta={r28 - r27:+.2f}   -> {dir_b}")

        # (c) entrance-rate trajectory. Three sub-signals, in order of
        # how much weight each deserves:
        #   1. AUTHORITATIVE PEAK (winners/best.json ground truth) --
        #      the exact metric_value the trainer's own retention logic
        #      selected on. See LOG-VS-CHECKPOINT DISCREPANCY above main():
        #      this can exceed the log-derived peak below because a
        #      force-completion scoring pass runs after the log line
        #      prints. Ceiling effect: ALL 8 runs land in 0.867-1.000, so
        #      this alone barely discriminates v27 from v28 -- reported,
        #      but downweighted for exactly that reason (also consistent
        #      with v27's own documented finding that in-training
        #      telemetry saturates well above the honest rate).
        #   2. mean(all entrance-era) -- log-derived, ~200+ points, the
        #      lowest-variance signal, captures overall area-under-the-
        #      bell-curve quality rather than a single saturated point.
        #   3. frac_iters>=0.50 -- log-derived "sustained competence":
        #      fraction of entrance-era logged iters at/above a 0.50
        #      trailing rate. Also robust to the peak's ceiling effect.
        # mean(last30) is reported as CONTEXT ONLY, excluded from the
        # composite: both v27 and v28 decay to a near-zero trailing rate
        # by the run's final ~50-70 iters in every one of the 8 logs (see
        # the bell-shaped rise/decay in entrance_trailing_rate_series in
        # read3_raw.json) -- a near-zero-vs-near-zero tail comparison is
        # noise, not signal.
        # NOISE FLOOR: "trailing" is a successes/attempts ratio over a
        # deque(maxlen=30) -- one episode is a 1/30=.033 step, binomial SE
        # at p~0.5,n=30 is ~0.09. tol_abs=0.08 on point-estimate peaks,
        # tol_abs=0.03 on the full-era mean, tol_abs=0.08 on frac>=0.50
        # (also effectively a proportion over ~200 correlated iters, not
        # 200 independent draws, so treated with the same wide band as a
        # single window).
        ap27 = a27.get("authoritative_peak")
        ap28 = a28.get("authoritative_peak")
        p27 = a27.get("entrance_trailing_rate_peak")
        p28 = a28.get("entrance_trailing_rate_peak")
        m27_30 = a27.get("entrance_trailing_rate_mean_last30")
        m28_30 = a28.get("entrance_trailing_rate_mean_last30")
        m27_all = a27.get("entrance_trailing_rate_mean_all")
        m28_all = a28.get("entrance_trailing_rate_mean_all")
        f27_50 = a27.get("entrance_trailing_rate_frac_ge_050")
        f28_50 = a28.get("entrance_trailing_rate_frac_ge_050")
        dir_c_authpeak = direction(ap27, ap28, higher_is_better=True, tol_abs=0.08)
        dir_c_peak = direction(p27, p28, higher_is_better=True, tol_abs=0.08)
        dir_c_last30 = direction(m27_30, m28_30, higher_is_better=True, tol_abs=0.03)
        dir_c_all = direction(m27_all, m28_all, higher_is_better=True, tol_abs=0.03)
        dir_c_frac50 = direction(f27_50, f28_50, higher_is_better=True, tol_abs=0.08)
        print(f"  (c) AUTHORITATIVE peak (winners/best.json): v27={ap27:.4f} (iter {a27.get('authoritative_peak_iter')})"
              f"   v28={ap28:.4f} (iter {a28.get('authoritative_peak_iter')})   -> {dir_c_authpeak}"
              f"  [near-ceiling in both -- low discriminating power]")
        print(f"      log-derived peak (lower bound, see caveat): v27={p27:.2f} (iter {a27.get('entrance_trailing_rate_peak_iter')})"
              f"   v28={p28:.2f} (iter {a28.get('entrance_trailing_rate_peak_iter')})   -> {dir_c_peak}")
        print(f"      mean(all entrance-era):     v27={m27_all:.3f}  v28={m28_all:.3f}   -> {dir_c_all}")
        print(f"      frac_iters>=0.50 (sustain): v27={f27_50:.3f}  v28={f28_50:.3f}   -> {dir_c_frac50}")
        print(f"      [context only, excluded from verdict -- both versions decay near 0 by run end] "
              f"mean(last30): v27={m27_30:.3f}  v28={m28_30:.3f}   (nominal dir {dir_c_last30})")
        print(f"      [context, cumulative not windowed] entrance F/T: "
              f"v27={a27.get('cumulative_entrance_final_frac')}={a27.get('cumulative_entrance_rate_final'):.3f}  "
              f"v28={a28.get('cumulative_entrance_final_frac')}={a28.get('cumulative_entrance_rate_final'):.3f}")

        # rungs binned trajectory shape (compact)
        def shape_str(binned):
            return " -> ".join(f"{b['rungs_per_100_iters']:.0f}" for b in binned)
        print(f"      rungs/100iters by 25-iter bin: v27=[{shape_str(a27['rungs_binned_trajectory'])}]  "
              f"v28=[{shape_str(a28['rungs_binned_trajectory'])}]")

        # combine (a)+(b) into one ladder-speed signal (they are
        # mathematically coupled given matched initial_tau: faster
        # iters-to-entrance == higher rungs/100iters). c is independent.
        ladder_speed_dir = dir_a if dir_a == dir_b else "AMBIGUOUS(a!=b)"

        # entrance-quality: two composite rules, reported side by side
        # rather than silently picking one. Uses the three PRIMARY (c)
        # signals -- authoritative peak, mean(all-era), frac>=0.50 -- and
        # excludes the log-derived peak (a known lower-bound proxy, see
        # its caveat above) and mean(last30) (noise-floor tail, see its
        # caveat above) from the vote.
        #   STRICT  -- all three primary signals must read the same
        #              non-flat direction, else "mixed/flat".
        #   LENIENT -- a "flat" reading (inside the noise band) does not
        #              veto a non-flat reading from the OTHER sub-metrics;
        #              only an actual improve-vs-worse conflict is
        #              "mixed". Justified independently of any seed's
        #              outcome by the noise-floor note above: the
        #              authoritative peak is a single point estimate off
        #              a 30-episode window near a 0.87-1.00 ceiling in
        #              EVERY run (SE ~0.09, plus near-zero headroom to
        #              move), while mean(all-era) and frac>=0.50 average
        #              ~200+ points each (far lower variance, real
        #              headroom) -- when the ceiling-saturated peak
        #              disagrees with either lower-variance signal, the
        #              lower-variance estimator should carry the call.
        c_dirs = {dir_c_authpeak, dir_c_all, dir_c_frac50}
        if c_dirs == {"improve"}:
            entrance_quality_strict = "improve"
        elif c_dirs == {"worse"}:
            entrance_quality_strict = "worse"
        elif "flat" in c_dirs and len(c_dirs) == 1:
            entrance_quality_strict = "flat"
        else:
            entrance_quality_strict = "mixed"

        if "improve" in c_dirs and "worse" in c_dirs:
            entrance_quality_lenient = "mixed"
        elif "improve" in c_dirs:
            entrance_quality_lenient = "improve"
        elif "worse" in c_dirs:
            entrance_quality_lenient = "worse"
        else:
            entrance_quality_lenient = "flat"

        def combine(ladder_dir, ent_dir):
            if ladder_dir == "improve" and ent_dir in ("improve", "flat"):
                return "IMPROVING"
            if ladder_dir == "worse" and ent_dir in ("worse", "flat"):
                return "WORSE"
            if ladder_dir == "improve" and ent_dir == "worse":
                return "MIXED (faster ladder walk, worse entrance quality)"
            if ladder_dir == "worse" and ent_dir == "improve":
                return "MIXED (slower ladder walk, better entrance quality)"
            if ladder_dir == "flat" and ent_dir == "improve":
                return "IMPROVING (entrance quality only; ladder speed flat)"
            if ladder_dir == "flat" and ent_dir == "worse":
                return "WORSE (entrance quality only; ladder speed flat)"
            return "FLAT/AMBIGUOUS"

        seed_verdict_strict = combine(ladder_speed_dir, entrance_quality_strict)
        seed_verdict_lenient = combine(ladder_speed_dir, entrance_quality_lenient)
        agree_flag = "" if seed_verdict_strict == seed_verdict_lenient else "  ** STRICT/LENIENT DISAGREE **"

        print(f"  ladder_speed_dir={ladder_speed_dir}")
        print(f"  entrance_quality: strict={entrance_quality_strict}  lenient={entrance_quality_lenient}")
        print(f"  => SEED VERDICT (strict rule):  {seed_verdict_strict}")
        print(f"  => SEED VERDICT (lenient rule, PRIMARY): {seed_verdict_lenient}{agree_flag}")
        verdicts[seed] = {
            "ladder_speed_dir": ladder_speed_dir,
            "entrance_quality_strict": entrance_quality_strict,
            "entrance_quality_lenient": entrance_quality_lenient,
            "seed_verdict_strict": seed_verdict_strict,
            "seed_verdict_lenient": seed_verdict_lenient,
            "seed_verdict": seed_verdict_lenient,  # primary, per noise-floor rationale above
        }

    def tally(key):
        n_imp = sum(1 for v in verdicts.values() if v[key].startswith("IMPROVING"))
        n_wor = sum(1 for v in verdicts.values() if v[key].startswith("WORSE"))
        n_mix = 4 - n_imp - n_wor
        return n_imp, n_wor, n_mix

    n_improving, n_worse, n_mixed_or_flat = tally("seed_verdict_lenient")
    n_improving_s, n_worse_s, n_mixed_s = tally("seed_verdict_strict")
    print()
    print("=" * 100)
    print(f"READ #3 SUMMARY (LENIENT rule, PRIMARY): {n_improving}/4 seeds IMPROVING, "
          f"{n_worse}/4 WORSE, {n_mixed_or_flat}/4 MIXED/FLAT/AMBIGUOUS")
    print(f"READ #3 SUMMARY (STRICT rule, cross-check): {n_improving_s}/4 seeds IMPROVING, "
          f"{n_worse_s}/4 WORSE, {n_mixed_s}/4 MIXED/FLAT/AMBIGUOUS")
    print("=" * 100)

    verdict_path = Path(__file__).resolve().parent / "read3_verdicts.json"
    with open(verdict_path, "w") as fh:
        json.dump({
            "per_seed": verdicts,
            "lenient": {"n_improving": n_improving, "n_worse": n_worse, "n_mixed_or_flat": n_mixed_or_flat},
            "strict": {"n_improving": n_improving_s, "n_worse": n_worse_s, "n_mixed_or_flat": n_mixed_s},
        }, fh, indent=2)
    print(f"\nWrote verdict summary to {verdict_path}")


if __name__ == "__main__":
    main()
