"""Pixel-phase mining for armor-plated fight objects.

Fight-doctrine mechanism, rank 2: some bosses expose a vulnerable core only
during a periodic "open" window while an armor plate hides it the rest of the
cycle. The window is invisible to the progress/HP dimensions the solver keys
on, so the archive grinds forever against a plate. This tool recovers the
window as an OBSERVABLE the solver can key cells on (and time fire to):

  1. Restore a spread of banked wall states in a rendering environment and
     advance each ~600 frames under neutral input, capturing every frame AND
     the full 2 KB RAM per frame.
  2. Localize the core bbox programmatically from a per-pixel temporal signal
     over the wall region -- specifically the periodic power at the flicker
     rate, which isolates the cycling core from aperiodic bullets/enemies and
     from the fast-blinking decorative light. The RED channel is used because
     the core cycles hue (near iso-luminant), which a grayscale difference is
     blind to.
  3. Frame-difference signal dV_t = mean |I_t - I_{t-1}| over the bbox.
  4. Cross-correlate: mutual information between the discretized dV_t phase
     signal and every RAM byte's time series (pooled across states to defeat
     the global-clock confound), plus MI against the open/closed appearance and
     a deterministic exact-match predicate search. GATE: the top byte's MI(dV)
     must clear a pre-registered kill threshold, else the clean negative is the
     receipted verdict.
  5. Verify the winner: render open-phase and closed-phase PNG receipts
     annotated with the byte value, and report the cycle period + open window.

First real case: Contra stage-1 base-wall core (the X-plated circle low on the
right wall). Observables are derived purely from our own banked rollouts -- no
external RAM maps, no disassembly.

Usage:
  python scripts/pixel_phase.py \
      --archive runs/breadth_contra/stage1_v6_resume \
      --profile configs/contra.yaml \
      --out runs/breadth_contra --states 12 --frames 600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nes_core  # noqa: E402
from go_explore_solve import make_game  # noqa: E402
from src.training.go_explore import GoExploreArchive  # noqa: E402

NOOP = 0
MI_GATE = 0.30                 # pre-registered kill threshold on MI(top byte; dV)
CROP = (130, 96, 214, 204)     # wall search window (x0, y0, x1, y1) in the frame
LOWER_PRIOR = (145, 150, 210, 196)  # x0,y0,x1,y1: "low on the wall" localize prior
FLICKER_PERIODS = (8, 16, 32)  # palette-change / open-close / full-palette cycle
PRED_MODULI = (8, 16, 32)      # a usable open-window is a simple recurring schedule


def _py(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not serializable: {type(o)}")


# --------------------------------------------------------------------------
# wall-state selection
# --------------------------------------------------------------------------
def select_wall_states(archive_dir: Path, cell_fn, wall_progress: int, n: int,
                       min_bucket=200):
    """Load the archive and return a spread of banked wall states.

    Wall cells are key[-1]*8 >= wall_progress (last key component is the
    progress bucket). We spread the selection across the boss-HP key component
    (fight stage) so at least some states have the core clearly on screen, and
    within each HP bucket take the deepest trajectory (camera settled).
    Degenerate HP buckets (few cells: phase-aliasing fallbacks) are skipped."""
    arch = GoExploreArchive(cell_fn, seed=0)
    t0 = time.time()
    arch.load(str(archive_dir / "archive.pkl"))
    wall = [(k, c) for k, c in arch.cells.items()
            if k[-1] * 8 >= wall_progress and c.state is not None]
    print(f"loaded {len(arch.cells)} cells in {time.time()-t0:.1f}s; "
          f"{len(wall)} wall cells (key[-1]*8 >= {wall_progress})", flush=True)
    if not wall:
        return []
    HP, YB = 7, 9  # key = transit[0:6] + (area=6, hp=7, sig=8, y_band=9, gx=10)
    by_hp: dict[int, list] = {}
    for k, c in wall:
        by_hp.setdefault(k[HP], []).append((k, c))
    buckets = {hp: v for hp, v in by_hp.items() if len(v) >= min_bucket}
    if not buckets:                       # tiny archive: fall back to all
        buckets = by_hp
    for hp in buckets:
        buckets[hp].sort(key=lambda kc: kc[1].best_steps, reverse=True)
    hp_order = sorted(buckets, key=lambda h: -len(buckets[h]))
    picked, cursor = [], {}
    while len(picked) < n:
        progressed = False
        for hp in hp_order:
            lst, idx = buckets[hp], cursor.get(hp, 0)
            seen_yb = {k[YB] for k, _ in picked if k[HP] == hp}
            while idx < len(lst) and lst[idx][0][YB] in seen_yb:
                idx += 1
            if idx < len(lst):
                picked.append(lst[idx])
                cursor[hp] = idx + 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    out = [{"key": list(k), "hp": int(k[HP]), "gx": int(k[-1]), "y_band": int(k[YB]),
            "best_score": float(c.best_score), "best_steps": int(c.best_steps),
            "state": bytes(c.state)} for k, c in picked]
    del arch  # release ~14 GB
    return out


# --------------------------------------------------------------------------
# rollout capture
# --------------------------------------------------------------------------
def capture(rom: str, states, frames: int):
    x0, y0, x1, y1 = CROP
    Hc, Wc = y1 - y0, x1 - x0
    ns = len(states)
    red = np.empty((ns, frames, Hc, Wc), np.float32)   # redness R-(G+B)/2
    ram = np.empty((ns, frames, 2048), np.uint8)
    env = nes_core.NESEnvironment(rom, frame_skip=1)
    env.reset()
    for i, s in enumerate(states):
        env.load_state(s["state"])
        for t in range(frames):
            f, _ = env.step(NOOP)
            f = np.asarray(f)[y0:y1, x0:x1].astype(np.float32)
            red[i, t] = f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])
            ram[i, t] = np.asarray(env.get_ram_range(0, 2048))
        print(f"  captured state {i} (hp={s['hp']})", flush=True)
    return red, ram


# --------------------------------------------------------------------------
# bbox derivation (periodic-power map -> connected component around the peak)
# --------------------------------------------------------------------------
def derive_bbox(red):
    """Locate the flickering core low on the wall. Per-pixel periodic power at
    the flicker rate, averaged across states, isolates the cycling core: the
    aperiodic bullets/enemies wash out across states, the decorative light
    blinks at a different (period-2) rate, and the static wall has zero power.
    We take the smoothed peak in the lower-wall prior and the compact component
    around it (the X-plated core is the lowest cycling element)."""
    x0, y0, x1, y1 = CROP
    ns, nf, Hc, Wc = red.shape
    sig = red - red.mean(1, keepdims=True)
    t = np.arange(nf)
    pw = np.zeros((Hc, Wc))
    for P in FLICKER_PERIODS:
        c = np.cos(2 * np.pi * t / P)[None, :, None, None]
        s = np.sin(2 * np.pi * t / P)[None, :, None, None]
        a = (sig * c).mean(1)
        b = (sig * s).mean(1)
        pw += np.sqrt(a * a + b * b).mean(0)
    pw = ndimage.gaussian_filter(pw, 1.5)
    px0, py0, px1, py1 = LOWER_PRIOR
    prior = np.zeros((Hc, Wc), bool)
    prior[max(0, py0 - y0):py1 - y0, max(0, px0 - x0):px1 - x0] = True
    sp = np.where(prior, pw, 0.0)
    pk = np.unravel_index(sp.argmax(), sp.shape)
    lbl, _ = ndimage.label(sp >= 0.25 * sp[pk])
    ys, xs = np.where(lbl == lbl[pk])
    by0, by1 = max(0, ys.min() - 2), min(Hc, ys.max() + 3)
    bx0, bx1 = max(0, xs.min() - 2), min(Wc, xs.max() + 3)
    bbox = (bx0 + x0, bx1 + x0, by0 + y0, by1 + y0)     # full-frame x0,x1,y0,y1
    return bbox, (slice(by0, by1), slice(bx0, bx1))


# --------------------------------------------------------------------------
# signals + MI scan
# --------------------------------------------------------------------------
def mi_bits(a, b, na, nb):
    ct = np.zeros((na, nb))
    np.add.at(ct, (a, b), 1)
    n = ct.sum()
    if n == 0:
        return 0.0
    pa = ct.sum(1, keepdims=True) / n
    pb = ct.sum(0, keepdims=True) / n
    p = ct / n
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa * pb)[nz])).sum())


def _otsu(x, nbins=64):
    """Parameter-free bimodal split (steady vs transition) for the frame-diff
    signal, robust to how sharply bimodal dV is (a tight bbox makes it nearly
    a two-spike distribution that fixed percentiles mishandle)."""
    h, edges = np.histogram(x, bins=nbins)
    h = h.astype(float)
    p = h / h.sum()
    cx = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(p)
    mu = np.cumsum(p * cx)
    muT = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (muT * w - mu) ** 2 / (w * (1 - w))
    sb[~np.isfinite(sb)] = 0
    return float(cx[int(np.argmax(sb))])


def analyze(red, ram, sl):
    ns, nf, _, _ = red.shape
    sy, sx = sl
    core = red[:, :, sy, sx].reshape(ns, nf, -1)
    A = core.mean(axis=2)                                  # appearance (redness)
    dV = np.abs(np.diff(core, axis=1)).mean(axis=2)        # (ns, nf-1)

    # open/closed appearance phase: threshold between the high/low palette
    # clusters (exposed high-red core = "open").
    p25, p75 = np.percentile(A, 25), np.percentile(A, 75)
    appthr = (p25 + p75) / 2
    openf = (A >= appthr).astype(np.int8)

    dv_thr = _otsu(dV.reshape(-1))
    dV_al = (dV >= dv_thr).astype(np.int8).reshape(-1)
    ram_al = ram[:, 1:, :].reshape(ns * (nf - 1), 2048)    # RAM at diff frame t+1
    open_al = openf[:, 1:].reshape(-1)
    N = ram_al.shape[0]

    rows = []
    for b in range(2048):
        col = ram_al[:, b]
        card = int((np.bincount(col, minlength=256) > 0).sum())
        if card <= 1:
            continue
        mi_dv = mi_bits(col, dV_al, 256, 2)
        mi_op = mi_bits(col, open_al, 256, 2)
        null = (card - 1) / (2 * N * np.log(2))            # E[MI] under indep.
        rows.append({"byte": b, "mi_dv": mi_dv, "mi_open": mi_op,
                     "card": card, "mi_null": null})
    rows.sort(key=lambda r: r["mi_dv"], reverse=True)
    return rows, A, dV, openf, float(appthr), float(dv_thr)


# --------------------------------------------------------------------------
# predicate + timing for a candidate byte
# --------------------------------------------------------------------------
def predicate(byte, ram, openf, moduli=PRED_MODULI):
    """Best deterministic open predicate on `byte` over simple recurring
    schedules (byte mod m). Returns the modulus, the value set that fires the
    open phase, and the pooled + per-state accuracy."""
    ns, nf = openf.shape
    vals = ram[:, 1:, byte].reshape(-1).astype(np.int32)
    op = openf[:, 1:].reshape(-1)
    best = None
    for m in moduli:
        key = vals % m
        pos = np.zeros(m)
        cnt = np.zeros(m)
        np.add.at(pos, key, op)
        np.add.at(cnt, key, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pofk = np.where(cnt > 0, pos / cnt, 0.0)
        openset = [int(k) for k in range(m) if cnt[k] > 0 and pofk[k] > 0.5]
        acc = float((np.isin(key, openset).astype(np.int8) == op).mean())
        cand = {"modulus": m, "open_values": openset, "accuracy": acc,
                "n_open_values": len(openset)}
        if best is None or acc > best["accuracy"] + 1e-6 or (
                abs(acc - best["accuracy"]) <= 1e-6
                and m < best["modulus"]):
            best = cand
    m = best["modulus"]
    per_state = []
    for i in range(ns):
        k = ram[i, 1:, byte].astype(np.int32) % m
        per_state.append(round(float(
            (np.isin(k, best["open_values"]).astype(np.int8) == openf[i, 1:]).mean()), 3))
    best["per_state_accuracy"] = per_state
    return best


def timing(openf):
    ns = openf.shape[0]

    def runlengths(want):
        out = []
        for i in range(ns):
            c = 0
            for v in openf[i]:
                if v == want:
                    c += 1
                elif c:
                    out.append(c); c = 0
            if c:
                out.append(c)
        return np.array(out) if out else np.array([0])
    orl, crl = runlengths(1), runlengths(0)
    return {"open_window_frames": int(np.median(orl)),
            "closed_window_frames": int(np.median(crl)),
            "open_closed_cycle_frames": int(np.median(orl) + np.median(crl))}


def choose_winner(rows, ram, openf, topk=6):
    """Among the top gate-passing bytes by MI(dV), pick the one that admits the
    best simple recurring open schedule (highest predicate accuracy; ties to
    the smallest modulus). This favors a directly actionable observable over a
    high-MI byte that only predicts the phase via a full lookup table."""
    cands = [r for r in rows if r["mi_dv"] >= MI_GATE][:topk]
    best = None
    for r in cands:
        pred = predicate(r["byte"], ram, openf)
        score = (round(pred["accuracy"], 2), -pred["modulus"], -pred["n_open_values"])
        if best is None or score > best[0]:
            best = (score, r, pred)
    return best[1], best[2]


# --------------------------------------------------------------------------
# receipts (annotated PNGs)
# --------------------------------------------------------------------------
def _annotate(frame, bbox, lines, scale=3):
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = [v * scale for v in bbox]
    d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
    yl = 4
    for ln in lines:
        d.text((5, yl + 1), ln, fill=(0, 0, 0))       # shadow
        d.text((6, yl), ln, fill=(255, 255, 0))
        yl += 12
    return img


def render_receipts(rom, states, red, ram, sl, bbox, byte, pred, out):
    sy, sx = sl
    strength = red[:, :, sy, sx].reshape(red.shape[0], red.shape[1], -1).std(1).mean(1)
    rep = int(strength.argmax())
    m, openset = pred["modulus"], set(pred["open_values"])

    def find(is_open):
        for t in range(60, red.shape[1] - 4):
            if ((int(ram[rep, t, byte]) % m) in openset) == is_open:
                return t
        return 80 if is_open else 88

    env = nes_core.NESEnvironment(rom, frame_skip=1)
    env.reset()
    env.load_state(states[rep]["state"])
    to, tc = find(True), find(False)
    grabbed = {}
    for t in range(max(to, tc) + 1):
        f, _ = env.step(NOOP)
        if t in (to, tc):
            grabbed[t] = np.asarray(f).copy()

    for phase, t, path in (("OPEN", to, out / "pixel_phase_open.png"),
                           ("CLOSED", tc, out / "pixel_phase_closed.png")):
        v = int(ram[rep, t, byte])
        lines = [
            "Contra stage-1 base-wall core  (pixel-phase mining)",
            f"phase: {phase}    frame t={t}    state hp={states[rep]['hp']}",
            "$%04X = %d   ($%04X %% %d = %d)" % (byte, v, byte, m, v % m),
            f"rule: OPEN when ($%04X %% %d) in %s" % (byte, m, sorted(openset)),
            "core bbox  x[%d:%d] y[%d:%d]" % (bbox[0], bbox[1], bbox[2], bbox[3]),
        ]
        _annotate(grabbed[t], bbox, lines).save(path)
        print("  wrote %s  (%s, $%04X=%d, mod=%d)" % (path.name, phase, byte, v, v % m))
    return rep, to, tc


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="runs/breadth_contra/stage1_v6_resume")
    ap.add_argument("--profile", default="configs/contra.yaml")
    ap.add_argument("--out", default="runs/breadth_contra")
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--wall-progress", type=int, default=3064,
                    help="wall cell filter: key[-1]*8 >= this")
    args = ap.parse_args()

    profile = yaml.safe_load(open(REPO / args.profile))
    game = make_game(profile)
    rom = str(REPO / profile["solve"]["rom"])
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] selecting wall states ...", flush=True)
    states = select_wall_states(REPO / args.archive, game.cell_fn,
                                args.wall_progress, args.states)
    if not states:
        print("no wall states found; aborting", flush=True)
        return
    print(f"[2/5] rendering {len(states)} rollouts x {args.frames} frames "
          f"(neutral input) ...", flush=True)
    red, ram = capture(rom, states, args.frames)

    print("[3/5] deriving core bbox (periodic-power localizer) ...", flush=True)
    bbox, sl = derive_bbox(red)
    print(f"      derived bbox (x0,x1,y0,y1) = {bbox}", flush=True)

    print("[4/5] MI scan over 2048 RAM bytes (pooled across states) ...", flush=True)
    rows, A, dV, openf, appthr, dv_thr = analyze(red, ram, sl)
    top = rows[0]
    passed = top["mi_dv"] >= MI_GATE
    print("      gate top byte $%04X  MI(dV)=%.3f  MI(open)=%.3f  card=%d"
          % (top["byte"], top["mi_dv"], top["mi_open"], top["card"]), flush=True)
    print(f"      gate MI(dV) >= {MI_GATE}: {'PASS' if passed else 'FAIL'}", flush=True)

    receipt = {
        "case": "contra_stage1_base_wall_core",
        "target": "X-plated circle low on the right wall (armored core)",
        "method": "pixel-phase mining: frame-diff dV_t over the core bbox vs "
                  "every RAM byte (pooled MI + open/closed MI + exact-match)",
        "archive": str(Path(args.archive)),
        "profile": args.profile,
        "input": "neutral (NOOP) every frame",
        "n_states": len(states),
        "frames_per_state": args.frames,
        "wall_progress_filter": f"key[-1]*8 >= {args.wall_progress}",
        "states": [{"hp": s["hp"], "gx": s["gx"], "y_band": s["y_band"],
                    "best_score": s["best_score"], "best_steps": s["best_steps"]}
                   for s in states],
        "core_bbox_xyxy": {"x0": bbox[0], "x1": bbox[1], "y0": bbox[2], "y1": bbox[3]},
        "core_bbox_note": "derived: peak of per-pixel periodic power (periods "
                          f"{list(FLICKER_PERIODS)}) in the lower-wall prior "
                          f"{list(LOWER_PRIOR)}, then the compact component "
                          "around it. Isolates the cycling core from aperiodic "
                          "bullets and the period-2 decorative light.",
        "appearance_open_threshold_redness": round(appthr, 3),
        "dv_transition_threshold_otsu": round(dv_thr, 4),
        "mi_gate_bits": MI_GATE,
        "gate_metric": "MI(top_byte; discretized dV_t), bits",
        "gate_passed": bool(passed),
        "gate_top_byte": {"addr": "$%04X" % top["byte"], "mi_dv_bits": round(top["mi_dv"], 4),
                          "mi_open_bits": round(top["mi_open"], 4), "cardinality": top["card"]},
        "top_bytes": [
            {"addr": "$%04X" % r["byte"], "addr_dec": r["byte"],
             "mi_dv_bits": round(r["mi_dv"], 4), "mi_open_bits": round(r["mi_open"], 4),
             "cardinality": r["card"], "mi_null_bits": round(r["mi_null"], 4)}
            for r in rows[:8]],
    }

    if passed:
        print("[5/5] selecting actionable observable + rendering receipts ...", flush=True)
        wrow, pred = choose_winner(rows, ram, openf)
        byte = wrow["byte"]
        tw = timing(openf)
        rep, to, tc = render_receipts(rom, states, red, ram, sl, bbox, byte, pred, out)
        receipt["winner"] = {
            "addr": "$%04X" % byte, "addr_dec": byte,
            "selected_by": "highest simple-schedule (byte mod m) open-predicate "
                           "accuracy among gate-passing top-MI bytes; ties to "
                           "smallest modulus",
            "mi_dv_bits": round(wrow["mi_dv"], 4),
            "mi_open_bits": round(wrow["mi_open"], 4),
            "cardinality": wrow["card"],
            "semantics": ("frame counter (increments +1/frame, wraps 256)"
                          if wrow["card"] >= 250 else
                          "frame-locked counter / phase byte"),
            "open_predicate": pred,
            "palette_change_period_frames": 8,
            **tw,
            "full_palette_supercycle_frames_note": "the two 'closed' palettes "
                "alternate, so the full palette identity cycle is 32 frames; the "
                "open/closed vulnerability window is the 16-frame sub-cycle.",
            "receipt_frames": {"representative_state_hp": states[rep]["hp"],
                               "open_frame_t": to, "closed_frame_t": tc},
            "open_semantics": ("open = exposed high-red palette phase (armor "
                               "retracted appearance), the presumed-vulnerable "
                               "window. Damage-gating is NOT independently "
                               "confirmed here (prior fire tests left the core "
                               "intact) -- this observable is what feeds a "
                               "window-timed fire experiment / an open-phase "
                               "cell dimension."),
        }
        if byte != top["byte"]:
            receipt["winner"]["note"] = (
                "the max-MI byte is %s (MI_dV=%.3f); it is the same frame clock "
                "in a derived form (needs a full value table). $%04X is reported "
                "as the observable because its open window is a minimal "
                "(byte %% %d) schedule." % ("$%04X" % top["byte"], top["mi_dv"],
                                            byte, pred["modulus"]))
        print("      observable $%04X: OPEN when (byte %% %d) in %s  acc=%.3f  "
              "cycle=%df (open %df / closed %df)"
              % (byte, pred["modulus"], pred["open_values"], pred["accuracy"],
                 tw["open_closed_cycle_frames"], tw["open_window_frames"],
                 tw["closed_window_frames"]), flush=True)
    else:
        receipt["verdict"] = ("NEGATIVE: no RAM byte's time series carries "
                              f">= {MI_GATE} bits about the core's frame-diff "
                              "phase. The open window is not a free-running "
                              "timer observable in CPU RAM under neutral input.")

    with open(out / "pixel_phase_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, default=_py)
    print(f"wrote {out/'pixel_phase_receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
