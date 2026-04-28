"""
Headless training launcher — same pipeline the GUI uses, but no Qt.
Useful when you just want to kick off training and collect data
(especially with NES_EVOLVE_DEBUG=1) without clicking Start.

Usage:
    python scripts/headless_train.py
    NES_EVOLVE_DEBUG=1 python scripts/headless_train.py --max-generations 5

All flags have sensible defaults mirroring the GUI's "Start" with Zelda
selected: 16 instances, 16 genomes, Zelda profile, default start-state.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default=str(ROOT / "roms" / "zelda.nes"))
    p.add_argument("--profile", default=str(ROOT / "configs" / "zelda.yaml"))
    p.add_argument("--start-state", default=str(ROOT / "roms" / "zelda.ines1_start.state.bin"))
    p.add_argument("--num-instances", type=int, default=16)
    p.add_argument("--population-size", type=int, default=16)
    p.add_argument("--max-generations", type=int, default=10_000)
    p.add_argument(
        "--bc-demo",
        default=None,
        help="Demo recording(s) for behavior-cloning pretrain. Accepts: "
             "a single .state.bin, a colon-separated list of paths, or a "
             "directory (every *.state.bin inside is used). Multiple "
             "demos give the policy ~N× the state coverage of a single "
             "recording — directly addresses the 'BC didn't teach it "
             "anything' result on long games like Zelda.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Where checkpoints, metrics, BC cache live. Use a separate "
             "dir to keep this run independent of any GUI-driven run.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging (BC pretrain stats, startup phases, "
             "etc.). Without this, the run is nearly silent — only the "
             "[headless] startup line + warnings/errors reach stdout.",
    )
    args = p.parse_args()

    # Surface trainer INFO logs (BC dataset size, pretrain loss curve,
    # startup phase timings) so a headless run is diagnoseable. The
    # trainer's own main() does this too via src.utils.logging_config;
    # we mirror it here so callers of this launcher don't need to know.
    from src.utils.logging_config import configure as _configure_logging
    _configure_logging(verbose=args.verbose, log_dir=args.checkpoint_dir)

    from src.training.trainer import Trainer, load_game_profile, find_latest_checkpoint

    profile = load_game_profile(args.profile)

    trainer = Trainer(
        rom_path=args.rom,
        game_profile=profile,
        num_instances=args.num_instances,
        population_size=args.population_size,
        checkpoint_dir=args.checkpoint_dir,
        start_state_path=args.start_state if Path(args.start_state).exists() else None,
        bc_demo_path=args.bc_demo,
    )

    # Clean shutdown on Ctrl-C. Without this, the worker pool gets left
    # in weird states and we'd rely on the parent-death watchdog. Much
    # nicer to signal `trainer.stop()` and let the loop exit.
    def _sigint(_sig, _frame):
        print("\n[headless] SIGINT → trainer.stop()", flush=True)
        trainer.stop()
    signal.signal(signal.SIGINT, _sigint)

    # Auto-resume from the latest checkpoint in the dir if one exists,
    # so a Ctrl-C'd run can pick up where it left off.
    latest = find_latest_checkpoint(args.checkpoint_dir)
    resume_from = str(latest) if latest else None

    print(
        f"[headless] starting: rom={Path(args.rom).name} "
        f"instances={args.num_instances} pop={args.population_size} "
        f"frame_skip={trainer.frame_skip}"
        + (f" resume={Path(resume_from).name}" if resume_from else ""),
        flush=True,
    )
    trainer.run(num_generations=args.max_generations, resume_from=resume_from)
    print("[headless] training loop exited.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
