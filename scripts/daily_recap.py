#!/usr/bin/env python3
"""
Daily stream recap — mint a 60s summary MP4 from the run's artefacts.

Walks two sources:

1. `<checkpoint_dir>/depth_memo.jsonl` — every new-depth-record event
   the training loop emitted, with timestamp + caption + genome name.
2. `<highlights_dir>/*.mp4` — clips the HighlightRecorder saved when
   banner events fired (first dungeon entry, triforce pickup, etc.).

The output is a single concatenated MP4 with intertitle frames between
clips announcing what happened and who did it. The intertitles are
synthesised as black PNGs with gradient text (no external graphic
asset needed), and ffmpeg stitches everything into a single ~60s
`highlights/daily_recap_<YYYYMMDD>.mp4`.

Usage:

    python scripts/daily_recap.py --checkpoint-dir checkpoints \
        --highlights-dir highlights \
        --max-duration-s 60

Requirements:

* `ffmpeg` on PATH (every host with highlights already has it).
* `PIL` (Pillow) for synthesising intertitle PNGs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# Allow `python scripts/daily_recap.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read_depth_memo(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue  # tolerate partial writes
    return records


def _pair_clips_with_records(
    highlights_dir: Path, depth_records: list[dict],
) -> list[tuple[Path, dict | None]]:
    """Return a timeline ordering — each highlight MP4 paired with
    the nearest-in-time depth record so we can caption it. Clips are
    sorted by filesystem mtime (the recorder writes them as events
    fire). Records without a clip, and clips without a record,
    still land in the output list; the recap should show both
    separately.
    """
    clips = sorted(
        highlights_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
    ) if highlights_dir.exists() else []
    paired: list[tuple[Path, dict | None]] = []
    for clip in clips:
        mt = clip.stat().st_mtime
        best: dict | None = None
        best_delta = float("inf")
        for rec in depth_records:
            d = abs(float(rec.get("timestamp", 0)) - mt)
            if d < best_delta:
                best_delta = d
                best = rec
        # If the nearest record is more than 30s away, it's probably
        # an unrelated event — leave the clip uncaptioned.
        paired.append((clip, best if best_delta <= 30.0 else None))
    return paired


def _render_title_card(
    out_path: Path,
    title: str,
    subtitle: str,
    footer: str = "",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Synthesise an intertitle PNG — black bg, large centered title,
    smaller subtitle, optional footer. No external graphic asset
    required; Pillow ships with reasonable default fonts."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(8, 8, 12))
    draw = ImageDraw.Draw(img)

    def _load_font(size: int):
        # Try common monospace paths; fall back to default bitmap font.
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ]
        for p in candidates:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = _load_font(72)
    sub_font = _load_font(36)
    foot_font = _load_font(22)

    # Center each piece of text.
    def _draw_centered(text: str, y: int, font, fill=(235, 235, 235)):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((width - w) // 2, y), text, font=font, fill=fill)

    _draw_centered(title, height // 2 - 100, title_font)
    if subtitle:
        _draw_centered(subtitle, height // 2 + 10, sub_font, fill=(180, 210, 255))
    if footer:
        _draw_centered(footer, height - 60, foot_font, fill=(120, 120, 130))

    img.save(out_path, format="PNG")


def _title_card_clip(
    png_path: Path, dur_s: float, out_path: Path, fps: int = 30,
) -> bool:
    """Turn a PNG into a static MP4 clip of the given duration.
    Called for intertitles. Returns True on success."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-loop", "1", "-t", f"{dur_s:.2f}", "-i", str(png_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-preset", "veryfast", str(out_path),
    ]
    rc = subprocess.call(cmd)
    return rc == 0 and out_path.exists()


def _rescale_clip(
    src: Path, dst: Path, width: int, height: int, fps: int = 30,
) -> bool:
    """Re-encode clip at the recap's output resolution + framerate so
    ffmpeg concat demuxer can stitch without codec mismatch."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-preset", "veryfast",
        str(dst),
    ]
    rc = subprocess.call(cmd)
    return rc == 0 and dst.exists()


def _concat_mp4s(clips: Iterable[Path], out_path: Path) -> bool:
    """Use ffmpeg's concat demuxer — fast, no re-encode as long as
    each input shares codec/parameters (we just rescaled them)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False,
    ) as listf:
        for clip in clips:
            listf.write(f"file '{clip.resolve()}'\n")
        list_path = Path(listf.name)
    try:
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ]
        rc = subprocess.call(cmd)
        return rc == 0 and out_path.exists()
    finally:
        list_path.unlink(missing_ok=True)


def build_recap(
    checkpoint_dir: Path,
    highlights_dir: Path,
    out_dir: Path,
    max_duration_s: float = 60.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Optional[Path]:
    """Assemble a recap MP4 from the run's artefacts. Returns the
    output path, or None when there's nothing worth recapping
    (e.g. no highlight clips and no depth records)."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not on PATH — cannot render recap", file=sys.stderr)
        return None

    records = _read_depth_memo(checkpoint_dir / "depth_memo.jsonl")
    pairings = _pair_clips_with_records(highlights_dir, records)

    if not pairings and not records:
        print("nothing to recap: no highlight clips, no depth records")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"daily_recap_{stamp}.mp4"

    # Budget: max_duration_s total. Divide equally across clips;
    # each clip gets its own 1-second intertitle preamble.
    n_clips = len(pairings) or 1
    per_clip_dur = max(3.0, (max_duration_s - n_clips * 1.0) / n_clips)

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        segments: list[Path] = []

        # Opener title card — overall run summary.
        deepest = records[-1] if records else None
        opener_png = work_dir / "opener.png"
        _render_title_card(
            opener_png,
            title="Training Daily Recap",
            subtitle=(
                f"Deepest reached: {deepest['caption']}"
                if deepest else "Run progress summary"
            ),
            footer=datetime.now().strftime("%A %B %d — %H:%M"),
            width=width, height=height,
        )
        opener_mp4 = work_dir / "opener.mp4"
        if _title_card_clip(opener_png, dur_s=3.0, out_path=opener_mp4, fps=fps):
            segments.append(opener_mp4)

        # Per-clip: intertitle → rescaled clip.
        for i, (clip, rec) in enumerate(pairings):
            title_png = work_dir / f"title_{i:03d}.png"
            subtitle = (
                rec["caption"] if rec else "highlight"
            )
            name = rec["genome_name"] if rec else "unknown"
            _render_title_card(
                title_png,
                title=name,
                subtitle=subtitle,
                footer=f"highlight {i + 1} / {len(pairings)}",
                width=width, height=height,
            )
            title_mp4 = work_dir / f"title_{i:03d}.mp4"
            if not _title_card_clip(title_png, dur_s=1.0, out_path=title_mp4, fps=fps):
                continue
            segments.append(title_mp4)

            rescaled = work_dir / f"clip_{i:03d}.mp4"
            if _rescale_clip(clip, rescaled, width=width, height=height, fps=fps):
                segments.append(rescaled)
            # Trim if we exceeded budget. Rough: count segments * ~4s.
            if len(segments) * 4 > max_duration_s:
                break

        # Closer card.
        closer_png = work_dir / "closer.png"
        total_events = len(records)
        _render_title_card(
            closer_png,
            title="Tune in tomorrow",
            subtitle=f"{total_events} new depth records this run",
            footer="nes-evolve",
            width=width, height=height,
        )
        closer_mp4 = work_dir / "closer.mp4"
        if _title_card_clip(closer_png, dur_s=2.5, out_path=closer_mp4, fps=fps):
            segments.append(closer_mp4)

        if not segments:
            print("no segments rendered — recap skipped", file=sys.stderr)
            return None
        if not _concat_mp4s(segments, out_path):
            print("ffmpeg concat failed — recap skipped", file=sys.stderr)
            return None

    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Mint a 60s daily-recap MP4")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--highlights-dir", default="highlights")
    p.add_argument("--out-dir", default="highlights",
                   help="Where the recap MP4 lands; defaults to the highlights directory")
    p.add_argument("--max-duration-s", type=float, default=60.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    build_recap(
        checkpoint_dir=Path(args.checkpoint_dir),
        highlights_dir=Path(args.highlights_dir),
        out_dir=Path(args.out_dir),
        max_duration_s=args.max_duration_s,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
