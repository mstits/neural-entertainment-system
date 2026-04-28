#!/usr/bin/env python3
"""Record a trained policy's gameplay as an H.264 MP4 (AAC audio optional).

Loads a PolicyNetwork (or GA-checkpoint genome) from a .pt file, runs it
against a single NES environment, pipes raw video to ffmpeg for H.264,
and (with --with-audio) muxes the APU's PCM dump as an AAC track.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.emulation.frame_utils import (  # noqa: E402
    BUTTON_A, BUTTON_B, BUTTON_DOWN, BUTTON_LEFT, BUTTON_NOOP,
    BUTTON_RIGHT, BUTTON_SELECT, BUTTON_START, BUTTON_UP,
    FrameStacker,
)
from src.models.policy_network import PolicyNetwork, get_best_device  # noqa: E402

_NAME_TO_BIT = {
    "NOOP": BUTTON_NOOP, "A": BUTTON_A, "B": BUTTON_B,
    "up": BUTTON_UP, "down": BUTTON_DOWN,
    "left": BUTTON_LEFT, "right": BUTTON_RIGHT,
    "start": BUTTON_START, "select": BUTTON_SELECT,
}
_DEFAULT_ACTION_SPACE = (
    [], ["up"], ["down"], ["left"], ["right"],
    ["A"], ["B"], ["up", "A"], ["down", "A"], ["left", "A"], ["right", "A"],
)


def build_bitmasks(action_space):
    masks = []
    for buttons in action_space:
        m = 0
        for b in buttons:
            m |= _NAME_TO_BIT[b]
        masks.append(m)
    return masks


def load_policy(path: Path, device: torch.device) -> PolicyNetwork:
    """Load PolicyNetwork.save() format, a bare state_dict, or a GA population checkpoint."""
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and "num_actions" in raw:
        net = PolicyNetwork(
            num_actions=raw["num_actions"], frame_stack=raw["frame_stack"],
            frame_size=raw["frame_size"])
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and "population" in raw:
        pop = raw["population"]
        if not pop:
            raise RuntimeError(f"GA checkpoint {path} has empty population")
        best = max(pop, key=lambda g: g.get("fitness", float("-inf")))
        sd = best["state_dict"]
        print(f"[info] loaded best genome fitness={best.get('fitness')}", file=sys.stderr)
        net = PolicyNetwork(
            num_actions=int(sd["fc2.weight"].shape[0]),
            frame_stack=int(sd["conv1.weight"].shape[1]))
    elif isinstance(raw, dict) and "fc2.weight" in raw:
        sd = raw
        net = PolicyNetwork(
            num_actions=int(sd["fc2.weight"].shape[0]),
            frame_stack=int(sd["conv1.weight"].shape[1]))
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {path}")
    net.load_state_dict(sd, strict=False)
    return net.to(device).eval()


def spawn_video_ffmpeg(output: Path):
    """Video-only pipe. Stdin takes raw rgb24 256x240 @ 60fps."""
    cmd = ["ffmpeg", "-y", "-loglevel", "warning",
           "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", "256x240", "-framerate", "60", "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", str(output)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mux_audio(video_path: Path, audio_pcm: Path, audio_rate: int, output: Path) -> int:
    """Mux an on-disk H.264 MP4 with a raw s16le mono PCM dump into final MP4."""
    cmd = ["ffmpeg", "-y", "-loglevel", "warning",
           "-i", str(video_path),
           "-f", "s16le", "-ar", str(audio_rate), "-ac", "1", "-i", str(audio_pcm),
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(output)]
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a trained policy's gameplay as MP4.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--rom", required=True, type=Path)
    ap.add_argument("--profile", type=Path, default=None, help="YAML with an action_space block.")
    ap.add_argument("--start-state", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("showcase.mp4"))
    ap.add_argument("--frames", type=int, default=3600)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--with-audio", action="store_true")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH (brew install ffmpeg)", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.rom.exists():
        print(f"error: rom not found: {args.rom}", file=sys.stderr)
        return 2

    action_space = list(_DEFAULT_ACTION_SPACE)
    if args.profile is not None and args.profile.exists():
        with args.profile.open() as f:
            prof = yaml.safe_load(f) or {}
        if prof.get("action_space"):
            action_space = prof["action_space"]
    bitmasks = build_bitmasks(action_space)

    device = get_best_device()
    print(f"[info] device={device}", file=sys.stderr)
    policy = load_policy(args.checkpoint, device)

    import nes_core
    env = nes_core.NESEnvironment(
        str(args.rom),
        frame_skip=max(1, int(args.frame_skip)),
        start_state_path=str(args.start_state) if args.start_state else None,
    )
    audio_rate = int(env.sample_rate)
    stacker = FrameStacker(stack_size=policy.conv1.in_channels, target_size=policy.frame_size)

    tmpdir = Path(tempfile.mkdtemp(prefix="showcase_"))
    video_tmp = tmpdir / "video.mp4" if args.with_audio else args.output
    audio_tmp: Path | None = tmpdir / "audio.pcm" if args.with_audio else None
    audio_out = open(audio_tmp, "wb") if audio_tmp is not None else None

    proc = spawn_video_ffmpeg(video_tmp)
    assert proc.stdin is not None

    def _emit(frame_rgb):
        proc.stdin.write(frame_rgb.tobytes())
        if audio_out is not None:
            audio_out.write(env.get_audio().tobytes())

    try:
        frame = env.reset()
        stacked = stacker.reset(frame)
        _emit(frame)
        for step in range(args.frames):
            obs = torch.from_numpy(stacked).float().div_(255.0).to(device)
            action_idx = policy.act(obs, deterministic=True)
            frame, done = env.step(bitmasks[action_idx % len(bitmasks)])
            stacked = stacker.push(frame)
            _emit(frame)
            if done:
                print(f"[info] env reported done at step {step + 1}", file=sys.stderr)
                break
    finally:
        try: proc.stdin.close()
        except Exception: pass
        if audio_out is not None:
            try: audio_out.close()
            except Exception: pass
        rc = proc.wait()

    if rc != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"error: ffmpeg video encode exited with code {rc}", file=sys.stderr)
        return rc

    if args.with_audio and audio_tmp is not None:
        rc = mux_audio(video_tmp, audio_tmp, audio_rate, args.output)
        if rc != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"error: ffmpeg mux exited with code {rc}", file=sys.stderr)
            return rc
    shutil.rmtree(tmpdir, ignore_errors=True)
    size = args.output.stat().st_size if args.output.exists() else 0
    print(f"[ok] wrote {args.output} ({size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
