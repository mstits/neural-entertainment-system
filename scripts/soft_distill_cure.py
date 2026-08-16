"""Third-generation greedy-cure (SMB 1-2) — soft-target self-distillation
on the night-2 success states. Token-bound: loads checkpoints/npz and
trains on CPU tensors only; NEVER rolls out (the consolidation run owns
the emulator). The gate is a printed handoff, not a probe.

WHY (receipts in runs/night2/night2.jsonl): both BC cures on the
collected success trajectories FAILED the pre-registered gate —
actor-only (390 params) broke the x2979 det lock-in but taxed the honest
median (2059 -> 1731) and never fixed the flag-grab; actor+trunk
regressed everything (honest 1635, det chain-clear lost). Root cause:
the pairs' action labels are EXECUTED actions under sticky 0.25 — ~25%
are forced repeats, i.e. structured label noise that plain cross-entropy
faithfully imitates. Meanwhile the policy's own SAMPLED distribution
clears strictly 31.7% (190/600) where greedy clears ~2%: the knowledge
is already in the net's probabilities; only its argmax is broken.

MECHANISM — distill the net into itself on the success-state manifold,
ignoring the executed-action labels entirely:

* Teacher = the SOURCE net (sha256-pinned, the same pin night2_runner
  verifies), bit-frozen; its T=1 logits over all 117,575 success states
  are computed once under no_grad and become the only training signal.
* Student = a trainable copy of the same net. No action labels exist
  anywhere in the objective, so the sticky-repeat noise that sank both
  BC cures is structurally unreachable (`soft_distill_cure(payload,
  obs, ...)` — there is no argument to pass labels through).

THE OBJECTIVE, AND WHY THIS EXACT FORM. Classic knowledge distillation
tempers the TEACHER above 1 (T ~ 1.2-4) to SOFTEN targets — rejected
here: softening pushes the student toward broader distributions, the
opposite of a greedy cure (--teacher-temp stays a flag, default 1.0).
Hard CE against the teacher's argmax is circular — the teacher's argmax
IS what greedy already plays. The non-circular signal is that these
states were visited by SAMPLED play that actually cleared: distill the
teacher's full T=1 distribution into a student whose logits are tempered
at tau < 1 inside the loss —

    fkl (default):  KL( softmax(z_t / T_t)  ||  softmax(z_s / tau) )
    rkl (flag):     KL( softmax(z_s / tau)  ||  softmax(z_t / T_t) )

with T_t = 1.0 and tau = 0.7. Per state, the fkl fixed point is
z_s = tau * log p_t (+ const): same argmax, compressed logit gaps —
which is precisely the lever. Dividing the student's logits by tau
shrinks the target gap of every razor-thin split (log 0.45 - log 0.40
compresses by 0.7x), so under the shared trunk fitting 117k states, a
per-state razor-thin argmax is CHEAP to override by the neighborhood-
consistent preference — the action the sampled mode actually executes
around that state — while confidently-preferred actions (large gaps)
stay put. The cure is the function-approximation smoothing, not the
per-state target (greedy play is temperature-invariant); expect a SMALL
but targeted argmax-changed fraction, concentrated where the teacher is
nearly tied (the known argmax-tie-at-the-pole defect).

fkl vs rkl: forward KL is mass-covering per state — its failure mode is
"too gentle" (measurable; retune tau downward). Reverse KL is
mode-seeking — the student collapses onto ONE teacher mode per region
(the collapse-toward-majority-mode choice); its failure mode is
committing to the WRONG mode where the teacher is genuinely bimodal,
i.e. a behavioral regression of exactly the kind gen-2 produced. fkl is
therefore the registered default; rkl is the pre-registered escalation
if fkl moves too few argmaxes to matter.

SCOPE: default actor+trunk (unlike night2's BC default of actor-only) —
the targets are the teacher's own outputs, so width cannot inject label
noise; the trunk is where the smoothing capacity lives. The critic is
ALWAYS frozen (this loss has no value target), and an L2-to-teacher
anchor on the trainable trunk params (--l2-coef, default 1e-4) bounds
representation drift as insurance for the consol2 resume.

OUTPUT: runs/night2/cured_v3.pt in the exact night2 cured.pt payload
shape — iter = CURED_SEED_ITER (911) + net_state_dict + provenance +
night2 metadata; deliberately NO optimizer_state_dict (the cure
invalidates the source's Adam moments) and NO anticollapse (its snapshot
is the pre-cure net; carrying it would arm a rollback that undoes the
cure). GATE: the emulator is busy, so this script PRINTS the exact two
eval_game commands from night2_runner's step 3 (checkpoint swapped to
cured_v3.pt) for the operator's next window; `--install` additionally
copies cured_v3.pt over runs/night2/cured.pt so `night2_runner.py
--skip-to 3` gates it unchanged (default OFF; refuses an existing
cured.pt without --force, and --force backs the old one up first).
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Same plumbing as the night-2 orchestrator (same jsonl receipts, same
# sha pin via _verify_source, same gate-command builders) so the v3
# lineage stays byte-comparable with the BC-cure receipts.
from scripts.night2_runner import (  # noqa: E402
    CONFIG as NIGHT2_CONFIG, CURED_SEED_ITER, _TRUNK_PREFIXES,
    _load_payload, _verify_source, build_det_probe_command,
    build_honest_probe_command, trainable_param_names,
)
from scripts.run_consol2 import _sha256  # noqa: E402
from scripts.run_online_campaign import _append_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG — every knob the v3 cure runs on (all overridable by flag).
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "run_dir": "runs/night2",
    "receipt_log": NIGHT2_CONFIG["receipt_log"],

    # Same pinned source + success states as the BC cures.
    "source_checkpoint": NIGHT2_CONFIG["source_checkpoint"],
    "source_checkpoint_sha256": NIGHT2_CONFIG["source_checkpoint_sha256"],
    "pairs_in": NIGHT2_CONFIG["sil_pairs_out"],

    # The objective (see module docstring for the fkl-vs-rkl decision).
    "objective": "fkl",
    "tau_student": 0.7,
    "teacher_temp": 1.0,

    # Scope + drift insurance. actor+trunk is SAFE here (teacher-output
    # targets carry no label noise); critic always frozen regardless.
    "scope": "actor+trunk",
    "l2_coef": 1e-4,

    # Optimization block mirrors night2's step 2 for comparability.
    "epochs": 15,
    "lr": 1e-4,
    "batch_size": 256,
    "seed": 20260815,
    "teacher_batch": 8192,   # no_grad chunk for the one-time teacher pass

    # Outputs.
    "cured_v3_out": "runs/night2/cured_v3.pt",
    "install_target": NIGHT2_CONFIG["cured_out"],   # runs/night2/cured.pt
}

_OBJECTIVES: tuple[str, ...] = ("fkl", "rkl")


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested on synthetic tensors —
# tests/test_soft_distill_cure.py)
# ---------------------------------------------------------------------------


def soft_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    objective: str,
    tau_student: float,
    teacher_temp: float,
) -> torch.Tensor:
    """The v3 objective over one batch of logits.

    fkl: full KL(p_t || q_tau) — includes the -H(p_t) constant so the
    loss is exactly 0 at the fixed point z_s = tau*log(p_t) + c.
    rkl: KL(q_tau || p_t), the mode-seeking reverse form.
    Gradients flow through the STUDENT logits only; teacher logits are
    targets (precomputed under no_grad by the caller)."""
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    if not float(tau_student) > 0.0:
        raise ValueError(f"tau_student must be > 0, got {tau_student!r}")
    if not float(teacher_temp) > 0.0:
        raise ValueError(f"teacher_temp must be > 0, got {teacher_temp!r}")
    log_p = F.log_softmax(teacher_logits.detach() / float(teacher_temp),
                          dim=-1)
    log_q = F.log_softmax(student_logits / float(tau_student), dim=-1)
    if objective == "fkl":
        p = log_p.exp()
        return (p * (log_p - log_q)).sum(dim=-1).mean()
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(dim=-1).mean()


def argmax_report(
    teacher_logits: torch.Tensor,
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
) -> dict:
    """The cure's whole point, quantified: argmax agreement with the
    teacher's T=1 argmax before/after, the fraction of states whose
    argmax CHANGED (expect small but targeted), and the mean action-gap
    (top1 - top2 of the T=1 logits) before/after."""
    n_actions = int(teacher_logits.shape[-1])
    t = teacher_logits.argmax(dim=-1)
    a = pre_logits.argmax(dim=-1)
    b = post_logits.argmax(dim=-1)

    def mean_gap(z: torch.Tensor) -> float:
        top2 = z.topk(2, dim=-1).values
        return float((top2[:, 0] - top2[:, 1]).mean())

    gap_pre, gap_post = mean_gap(pre_logits), mean_gap(post_logits)
    return {
        "teacher_agreement_pre": float((a == t).float().mean()),
        "teacher_agreement_post": float((b == t).float().mean()),
        "argmax_changed_frac": float((b != a).float().mean()),
        "n_changed": int((b != a).sum()),
        "mean_gap_pre": gap_pre,
        "mean_gap_post": gap_post,
        "mean_gap_delta": gap_post - gap_pre,
        "argmax_counts_pre": torch.bincount(
            a, minlength=n_actions).tolist(),
        "argmax_counts_post": torch.bincount(
            b, minlength=n_actions).tolist(),
    }


def load_success_obs(path) -> tuple[np.ndarray, dict]:
    """The success STATES from runs/night2/sil_pairs.npz — and only the
    states. The executed-action labels (act_0) are ~25% sticky-forced
    repeats, the exact structured noise that sank both BC cures; this
    loader never returns them, so no caller can train on them."""
    d = np.load(path)
    obs = np.asarray(d["obs_0"])
    meta = {
        "n_states": int(obs.shape[0]),
        "feature_dim": int(obs.shape[1]) if obs.ndim == 2 else None,
        "n_trajs": (int(d["traj_len"].shape[0])
                    if "traj_len" in d.files else None),
        "n_strict_clears": (int(np.asarray(
            d["label_episode_success"]).sum())
            if "label_episode_success" in d.files else None),
        "has_action_labels": "act_0" in d.files,
        "action_labels_used": False,
    }
    return obs, meta


def _batched_logits(net, X: torch.Tensor, batch: int) -> torch.Tensor:
    """One no_grad pass over all states, chunked."""
    outs = []
    with torch.no_grad():
        for i in range(0, int(X.shape[0]), int(batch)):
            logits, _ = net.forward_ac(X[i:i + int(batch)])
            outs.append(logits)
    return torch.cat(outs, dim=0)


def soft_distill_cure(
    payload: dict,
    obs: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    scope: str,
    objective: str,
    tau_student: float,
    teacher_temp: float,
    l2_coef: float,
    teacher_batch: int = 8192,
) -> tuple[dict, dict]:
    """The v3 cure: self-distill the source policy on the success
    states. Teacher = bit-frozen copy of the source net (verified
    bit-identical after training — stats['teacher_bit_frozen']);
    student = trainable copy under the night2 freeze partition (critic
    ALWAYS frozen; everything out of scope stays BIT-IDENTICAL). The L2
    anchor pulls trainable TRUNK params back toward the teacher's.
    Deterministic from `seed` (own torch.Generator, CPU ops). Returns
    (cured state_dict on cpu, stats). Takes STATES ONLY — action labels
    have no way in."""
    from src.models.tile_policy import build_tile_policy_from_checkpoint

    sd = payload.get("net_state_dict", payload)
    num_actions = int(sd["actor.bias"].shape[0])
    feature_dim = int(sd["fc1.weight"].shape[1])

    def build():
        net, is_recurrent = build_tile_policy_from_checkpoint(
            payload, num_actions=num_actions, feature_dim=feature_dim)
        if is_recurrent:
            raise ValueError("the soft-distill cure targets the stateless "
                             "tile policy; got a recurrent checkpoint")
        net.load_state_dict(sd, strict=True)
        return net

    X = torch.from_numpy(np.asarray(obs)).float()
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("soft_distill_cure on zero states")
    # Fail fast on bad knobs before any compute.
    soft_distill_loss(torch.zeros(1, num_actions),
                      torch.zeros(1, num_actions), objective=objective,
                      tau_student=tau_student, teacher_temp=teacher_temp)

    teacher = build()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_logits = _batched_logits(teacher, X, teacher_batch)

    student = build()
    trainable = trainable_param_names(
        [name for name, _ in student.named_parameters()], scope)
    params = []
    for name, p in student.named_parameters():
        p.requires_grad = name in trainable
        if p.requires_grad:
            params.append(p)
    # L2-to-teacher anchors: trainable TRUNK params only (drift
    # insurance; the actor head is the thing being cured, so it is not
    # anchored — a no-op under scope='actor').
    anchored = {
        name: sd[name].detach().clone()
        for name in trainable if name.startswith(_TRUNK_PREFIXES)
    }
    student_params = dict(student.named_parameters())

    student.eval()
    pre_logits = _batched_logits(student, X, teacher_batch)
    with torch.no_grad():
        initial_loss = float(soft_distill_loss(
            pre_logits, teacher_logits, objective=objective,
            tau_student=tau_student, teacher_temp=teacher_temp))

    steps_per_epoch = (n + int(batch_size) - 1) // int(batch_size)
    total_steps = max(1, int(epochs) * steps_per_epoch)
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1.0 - float(s) / float(total_steps)))
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    student.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            logits, _ = student.forward_ac(X[idx])
            loss = soft_distill_loss(
                logits, teacher_logits[idx], objective=objective,
                tau_student=tau_student, teacher_temp=teacher_temp)
            if float(l2_coef) > 0.0 and anchored:
                reg = sum((student_params[k] - anchored[k]).pow(2).sum()
                          for k in anchored)
                loss = loss + float(l2_coef) * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    student.eval()
    post_logits = _batched_logits(student, X, teacher_batch)
    with torch.no_grad():
        final_loss = float(soft_distill_loss(
            post_logits, teacher_logits, objective=objective,
            tau_student=tau_student, teacher_temp=teacher_temp))

    teacher_ok = all(torch.equal(v, sd[k])
                     for k, v in teacher.state_dict().items())
    cured = {k: v.detach().cpu().clone()
             for k, v in student.state_dict().items()}
    stats = {
        "n_states": n,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "lr_schedule": "linear_decay_to_zero",
        "seed": int(seed),
        "scope": scope,
        "objective": objective,
        "tau_student": float(tau_student),
        "teacher_temp": float(teacher_temp),
        "l2_coef": float(l2_coef),
        "action_labels_used": False,
        "trainable_params": int(sum(p.numel() for p in params)),
        "frozen_params": int(sum(
            p.numel() for name, p in student.named_parameters()
            if name not in trainable)),
        "opt_steps": total_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "teacher_bit_frozen": teacher_ok,
        "pre_bitexact_teacher": bool(torch.equal(pre_logits,
                                                 teacher_logits)),
        **argmax_report(teacher_logits, pre_logits, post_logits),
    }
    return cured, stats


def write_cured_v3(path, cured_sd: dict, *, stats: dict,
                   provenance: dict) -> None:
    """runs/night2/cured_v3.pt in the exact night2 cured.pt payload
    shape (iter/net_state_dict/provenance/night2), iter = CURED_SEED_ITER
    so `night2_runner --skip-to 3` + the consol2 seeding accept it
    unchanged. Same deliberate omissions as night2's writer: NO
    optimizer_state_dict, NO anticollapse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iter": CURED_SEED_ITER,
        "net_state_dict": {
            k: torch.as_tensor(v).detach().cpu()
            for k, v in cured_sd.items()
        },
        "provenance": "night2_soft_distill_cure",
        "night2": {"stats": dict(stats), **dict(provenance)},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def install_cured(v3_path, target, *, force: bool) -> str:
    """Copy cured_v3.pt over runs/night2/cured.pt so night2_runner
    --skip-to 3 gates the v3 cure with zero changes. Refuses an existing
    target without --force (that file is the failed-BC-cure lineage the
    receipts reference); --force backs it up to <name>.pre_v3.<unixts>
    first, then replaces atomically with sha verification."""
    v3_path = Path(v3_path)
    target = Path(target)
    if not v3_path.exists():
        raise RuntimeError(f"cured_v3 checkpoint missing: {v3_path}")
    backup_note = ""
    if target.exists():
        if not force:
            raise RuntimeError(
                f"{target} already exists (the prior cure lineage) — "
                f"refusing to overwrite without --force")
        backup = target.with_name(
            f"{target.name}.pre_v3.{int(time.time())}")
        shutil.copyfile(target, backup)
        backup_note = f"; prior cure backed up to {backup.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".pt.tmp")
    shutil.copyfile(v3_path, tmp)
    if _sha256(tmp) != _sha256(v3_path):
        tmp.unlink()
        raise RuntimeError(f"copy of {v3_path} -> {tmp} failed verification")
    os.replace(tmp, target)
    return f"installed {v3_path.name} -> {target}{backup_note}"


def build_gate_commands(checkpoint) -> dict[str, list[str]]:
    """The pre-registered gate, verbatim from night2_runner's step 3
    (same builders, same thresholds), checkpoint swapped to the v3 cure.
    Printed for the operator — NEVER run here (emulator is busy)."""
    return {
        "honest": build_honest_probe_command(checkpoint=checkpoint),
        "det": build_det_probe_command(checkpoint=checkpoint),
    }


# ---------------------------------------------------------------------------
# Runtime pieces
# ---------------------------------------------------------------------------


def _receipt(row: dict) -> None:
    _append_jsonl(REPO / CONFIG["receipt_log"], row)


def _print_gate_handoff(checkpoint) -> None:
    cmds = build_gate_commands(checkpoint)
    print("[cure_v3] gate handoff — run these in the next emulator "
          "window (night2 step-3 protocol, checkpoint swapped):",
          flush=True)
    print("[cure_v3]   honest leg (median must EXCEED "
          f"{NIGHT2_CONFIG['gate_honest_median_baseline']:.0f}):",
          flush=True)
    print("    " + shlex.join(str(a) for a in cmds["honest"]), flush=True)
    print("[cure_v3]   det leg (median must EXCEED "
          f"{NIGHT2_CONFIG['gate_det_median_baseline']:.0f}):", flush=True)
    print("    " + shlex.join(str(a) for a in cmds["det"]), flush=True)
    print("[cure_v3]   or: --install (with --force if cured.pt exists) "
          "then `python scripts/night2_runner.py --skip-to 3`", flush=True)


def _resolved_flags(args) -> dict:
    return {
        "objective": args.objective,
        "tau_student": args.tau_student,
        "teacher_temp": args.teacher_temp,
        "scope": args.scope,
        "l2_coef": args.l2_coef,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }


def dry_run(args) -> int:
    """Load everything, verify the assembly, report — no training, no
    rollouts. Mirrors night2_runner's --dry-run report style."""
    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"[dry-run] {'PASS' if good else 'FAIL'} {label}"
              + (f" — {detail}" if detail else ""), flush=True)

    # 1. Source checkpoint sha256 matches the night2/consol2 pin.
    try:
        src = _verify_source()
        report("source checkpoint sha256 matches pin", True,
               CONFIG["source_checkpoint_sha256"][:16] + "...")
    except Exception as e:
        report("source checkpoint sha256 matches pin", False, repr(e))
        return 1

    # 2. Payload loads; net dims reported off the state dict.
    try:
        payload = _load_payload(src)
        sd = payload["net_state_dict"]
        n_act = int(sd["actor.bias"].shape[0])
        feat = int(sd["fc1.weight"].shape[1])
        report("source payload loads + net dims", True,
               f"iter={payload.get('iter')} fc1={tuple(sd['fc1.weight'].shape)} "
               f"actions={n_act} feature_dim={feat}")
    except Exception as e:
        report("source payload loads + net dims", False, repr(e))
        return 1

    # 3. Success states load; action labels present but IGNORED.
    try:
        obs, meta = load_success_obs(REPO / CONFIG["pairs_in"])
        good = (meta["n_states"] > 0 and meta["feature_dim"] == feat
                and meta["action_labels_used"] is False)
        report("success states load (action labels IGNORED)", good,
               f"{meta['n_states']} states x {meta['feature_dim']} feats, "
               f"{meta['n_trajs']} trajs ({meta['n_strict_clears']} strict "
               f"clears); act_0 present={meta['has_action_labels']} and "
               f"IGNORED by construction")
    except Exception as e:
        report("success states load (action labels IGNORED)", False,
               repr(e))
        return 1

    # 4. Freeze partition under the requested scope, on the real net.
    try:
        trainable = trainable_param_names(list(sd.keys()), args.scope)
        n_train = sum(int(sd[k].numel()) for k in trainable)
        n_frozen = sum(int(v.numel()) for k, v in sd.items()
                       if k not in trainable)
        good = bool(trainable) and not any(
            k.startswith("critic.") for k in trainable)
        report(f"freeze partition (scope={args.scope}, critic frozen)",
               good, f"trainable {n_train} params "
                     f"({sorted(trainable)}), frozen {n_frozen}")
    except Exception as e:
        report(f"freeze partition (scope={args.scope}, critic frozen)",
               False, repr(e))

    # 5. Teacher forward on a sample; the student copy must agree 1.0
    #    (bit-exact copy check) and the initial loss must be finite.
    try:
        from src.models.tile_policy import build_tile_policy_from_checkpoint
        sample = torch.from_numpy(
            np.asarray(obs[:min(4096, len(obs))])).float()
        teacher, _ = build_tile_policy_from_checkpoint(
            payload, num_actions=n_act, feature_dim=feat)
        teacher.load_state_dict(sd, strict=True)
        teacher.eval()
        student, _ = build_tile_policy_from_checkpoint(
            payload, num_actions=n_act, feature_dim=feat)
        student.load_state_dict(sd, strict=True)
        student.eval()
        with torch.no_grad():
            t_logits, _ = teacher.forward_ac(sample)
            s_logits, _ = student.forward_ac(sample)
            agree = float((t_logits.argmax(-1)
                           == s_logits.argmax(-1)).float().mean())
            loss0 = float(soft_distill_loss(
                s_logits, t_logits, objective=args.objective,
                tau_student=args.tau_student,
                teacher_temp=args.teacher_temp))
        good = (agree == 1.0 and np.isfinite(loss0)
                and bool(torch.isfinite(t_logits).all()))
        report("teacher forward + pre-training student agreement", good,
               f"agreement={agree:.4f} on {sample.shape[0]} states, "
               f"initial {args.objective} loss={loss0:.6f} "
               f"(tau={args.tau_student}, T_t={args.teacher_temp})")
    except Exception as e:
        report("teacher forward + pre-training student agreement", False,
               repr(e))

    # 6. Gate handoff commands assemble against real paths (the source
    #    stands in for cured_v3.pt, which appears after training).
    try:
        stand_in = (REPO / CONFIG["cured_v3_out"]
                    if (REPO / CONFIG["cured_v3_out"]).exists() else src)
        cmds = build_gate_commands(stand_in)
        missing = [a for c in cmds.values() for a in c
                   if "/" in a and not a.startswith("-")
                   and not Path(a).exists()]
        report("gate handoff commands assemble", not missing,
               "night2 step-3 protocol, checkpoint swapped to "
               + str(stand_in.name) if not missing
               else f"missing paths: {missing}")
    except Exception as e:
        report("gate handoff commands assemble", False, repr(e))

    # 7. Install plan (night2_runner --skip-to 3 integration).
    target = REPO / CONFIG["install_target"]
    if target.exists():
        detail = (f"{target} exists (prior cure lineage) — --install "
                  f"will require --force and back it up first")
    else:
        detail = f"{target} absent — --install would write it directly"
    report("install plan coherent", True, detail)

    _print_gate_handoff(REPO / CONFIG["cured_v3_out"])
    _receipt({"type": "cure_v3_dry_run", "ok": ok,
              "flags": _resolved_flags(args)})
    print(f"[dry-run] {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}",
          flush=True)
    return 0 if ok else 1


def run(args) -> int:
    src = _verify_source()
    payload = _load_payload(src)
    obs, meta = load_success_obs(REPO / CONFIG["pairs_in"])
    print(f"[cure_v3] {meta['n_states']} success states loaded "
          f"({meta['n_trajs']} trajs, {meta['n_strict_clears']} strict "
          f"clears); executed-action labels present="
          f"{meta['has_action_labels']} and IGNORED", flush=True)

    t0 = time.time()
    cured, stats = soft_distill_cure(
        payload, obs,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, scope=args.scope, objective=args.objective,
        tau_student=args.tau_student, teacher_temp=args.teacher_temp,
        l2_coef=args.l2_coef, teacher_batch=CONFIG["teacher_batch"])
    train_s = time.time() - t0
    if not stats["teacher_bit_frozen"]:
        raise RuntimeError("teacher moved during distillation — invalid "
                           "run, refusing to write a cured checkpoint")

    out = REPO / (args.out or CONFIG["cured_v3_out"])
    write_cured_v3(
        out, cured, stats=stats,
        provenance={"scope": args.scope, "objective": args.objective,
                    "pairs": str(REPO / CONFIG["pairs_in"]),
                    "source": str(src),
                    "source_sha256": CONFIG["source_checkpoint_sha256"]})

    print(f"[cure_v3] trained in {train_s:.1f}s — loss "
          f"{stats['initial_loss']:.6f} -> {stats['final_loss']:.6f} "
          f"({stats['opt_steps']} steps, scope={args.scope}, "
          f"objective={args.objective}, tau={args.tau_student})",
          flush=True)
    print(f"[cure_v3] teacher@T=1 argmax agreement: "
          f"pre {stats['teacher_agreement_pre']:.4f} -> post "
          f"{stats['teacher_agreement_post']:.4f}", flush=True)
    print(f"[cure_v3] argmax CHANGED on {stats['n_changed']} / "
          f"{stats['n_states']} states "
          f"({stats['argmax_changed_frac']:.4f}) — the cure's whole "
          f"point; expect small but targeted", flush=True)
    print(f"[cure_v3] mean action-gap (top1-top2 @ T=1): "
          f"{stats['mean_gap_pre']:.4f} -> {stats['mean_gap_post']:.4f} "
          f"(delta {stats['mean_gap_delta']:+.4f})", flush=True)
    print(f"[cure_v3] argmax counts pre : {stats['argmax_counts_pre']}",
          flush=True)
    print(f"[cure_v3] argmax counts post: {stats['argmax_counts_post']}",
          flush=True)
    print(f"[cure_v3] wrote {out}", flush=True)
    gate_cmds = build_gate_commands(out)
    _receipt({"type": "cure_v3_train", "stats": stats,
              "flags": _resolved_flags(args), "out": str(out),
              "train_seconds": train_s,
              "gate_cmd_honest": shlex.join(
                  str(a) for a in gate_cmds["honest"]),
              "gate_cmd_det": shlex.join(
                  str(a) for a in gate_cmds["det"])})

    install_note = None
    if args.install:
        install_note = install_cured(out, REPO / CONFIG["install_target"],
                                     force=args.force)
        print(f"[cure_v3] {install_note} — night2_runner --skip-to 3 now "
              f"gates the v3 cure", flush=True)
        _receipt({"type": "cure_v3_install", "note": install_note})

    _print_gate_handoff(out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Load the pinned checkpoint + success states, "
                         "verify the assembly (shapes, freeze partition, "
                         "pre-training teacher-student agreement, gate "
                         "commands), train nothing.")
    ap.add_argument("--objective", choices=list(_OBJECTIVES),
                    default=CONFIG["objective"],
                    help="fkl (default, covering; see module docstring) "
                         "or rkl (mode-seeking escalation).")
    ap.add_argument("--tau-student", type=float,
                    default=CONFIG["tau_student"],
                    help="Student temperature inside the loss; < 1 "
                         "compresses target gaps (the cure's lever).")
    ap.add_argument("--teacher-temp", type=float,
                    default=CONFIG["teacher_temp"],
                    help="Teacher temperature; 1.0 (default) targets the "
                         "distribution sampled play actually drew from. "
                         "> 1 softens (classic distillation) — rejected "
                         "for a greedy cure, kept for ablation.")
    ap.add_argument("--scope", choices=["actor", "actor+trunk"],
                    default=CONFIG["scope"],
                    help="Trainable partition; critic always frozen. "
                         "Default actor+trunk — teacher-output targets "
                         "cannot inject label noise, so width is safe.")
    ap.add_argument("--l2-coef", type=float, default=CONFIG["l2_coef"],
                    help="L2-to-teacher anchor on trainable trunk params "
                         "(drift insurance).")
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"])
    ap.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--out", type=str, default=None,
                    help=f"Output path (default {CONFIG['cured_v3_out']}).")
    ap.add_argument("--install", action="store_true",
                    help="Also copy the result over "
                         f"{CONFIG['install_target']} so night2_runner "
                         "--skip-to 3 gates it. Refuses an existing "
                         "cured.pt without --force.")
    ap.add_argument("--force", action="store_true",
                    help="With --install: back up and replace an "
                         "existing cured.pt.")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run(args)
    return run(args)


if __name__ == "__main__":
    os.chdir(REPO)
    sys.exit(main())
