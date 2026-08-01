"""Hardware-encoded highlight clips wired to show trigger events.

A ClipRecorder keeps a rolling ring buffer of the show's last N seconds of
frames (and, optionally, audio). When something worth keeping happens —
a solution found, a new frontier record, a boss breach, a manual hotkey —
the caller fires `trigger()`. The buffered clip PLUS the next ~10 s of
live frames get written to an .mp4 using hardware H.264 encode. The
caller's thread is only touched long enough to snapshot the ring buffer
and hand it to a background worker: `trigger()` costs microseconds, never
the encode.

Encoder selection (checked once, cached):
  1. ffmpeg -c:v h264_videotoolbox  (this Mac has ffmpeg 8.1 built with
     --enable-videotoolbox; this is the supported, tested path).
  2. AVFoundation (pyobjc) AVAssetWriter, which also rides VideoToolbox
     hardware encode under the hood. Kept as a fallback for a host
     without ffmpeg. NOT exercised in this environment — pyobjc isn't
     installed here — so treat it as reference code, not a verified
     path (see ClipRecorder.hardware for what actually ran).
  3. Neither present: ClipRecorder raises at construction time with an
     explicit message instead of silently no-op'ing.

Usage (integration contract — see the bottom of the module docstring in
the completion report for the exact 5 calls)::

    rec = ClipRecorder(out_dir=SHOW_ROOT / "clips", buffer_seconds=20.0)
    ...
    rec.push_frame(frame_rgb, audio_i16)          # every emitted frame
    ...
    job = rec.trigger("solution", extra_seconds=10.0)   # <1 ms hand-off
    ...
    rec.close()                                    # on shutdown

Standalone demo (also the receipt-producing CLI)::

    scripts/show_clips.py --tape-steps 500
"""
from __future__ import annotations

import argparse
import ctypes
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Hardware-encoder detection
# ---------------------------------------------------------------------------

_CODEC_CACHE: dict | None = None


def detect_codec(prefer: str = "auto") -> dict:
    """Probe this machine once for a hardware H.264 encode path.

    Returns {"backend": "ffmpeg_videotoolbox" | "avfoundation" | "none",
             "detail": <human string, goes straight into the receipt>}.
    Cached at module scope (prefer="auto") so repeated ClipRecorder
    construction doesn't re-shell out to ffmpeg every time.
    """
    global _CODEC_CACHE
    if prefer == "auto" and _CODEC_CACHE is not None:
        return _CODEC_CACHE
    result = {"backend": "none",
              "detail": "no hardware H.264 encode path found: ffmpeg is "
                        "not on PATH and pyobjc/AVFoundation is not "
                        "installed. Install one (brew install ffmpeg is "
                        "the fast path) — ClipRecorder refuses to "
                        "silently fall back to software encode."}
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path and prefer in ("auto", "ffmpeg"):
        try:
            probe = subprocess.run([ffmpeg_path, "-hide_banner", "-encoders"],
                                    capture_output=True, text=True, timeout=10)
            if "h264_videotoolbox" in probe.stdout:
                result = {"backend": "ffmpeg_videotoolbox",
                          "detail": f"ffmpeg ({ffmpeg_path}) -c:v h264_videotoolbox"}
        except Exception as e:  # pragma: no cover - defensive
            result["detail"] = f"ffmpeg probe failed: {e!r}"
    if result["backend"] == "none" and prefer in ("auto", "avfoundation"):
        try:
            import AVFoundation  # noqa: F401
            result = {"backend": "avfoundation",
                      "detail": "pyobjc AVFoundation (AVAssetWriter -> "
                                "VideoToolbox hw encode). UNVERIFIED in "
                                "this environment — reference path only."}
        except Exception:
            pass
    if prefer == "auto":
        _CODEC_CACHE = result
    return result


# ---------------------------------------------------------------------------
# ffmpeg encoder (primary, tested path)
# ---------------------------------------------------------------------------

def _encode_ffmpeg(frames: list[np.ndarray], audio_chunks: list[np.ndarray],
                    audio_rate: int, out_path: Path, fps: float,
                    ffmpeg_path: str, bitrate: str = "3M") -> str:
    """Pipe raw RGB frames into ffmpeg's stdin; hardware-encode to mp4.

    stderr goes to a log file, never a pipe the caller doesn't drain —
    an ffmpeg progress line flood into a PIPE with no reader deadlocks
    the whole write loop (verified while building this: a naive
    stdout=PIPE/stderr=PIPE Popen hung indefinitely on real footage).
    """
    h, w = frames[0].shape[0], frames[0].shape[1]
    audio_raw_path = None
    if audio_chunks:
        audio = np.concatenate([np.asarray(a, dtype=np.int16).reshape(-1)
                                 for a in audio_chunks])
        audio_raw_path = out_path.with_suffix(".audio.raw")
        audio.tofile(audio_raw_path)

    cmd = [ffmpeg_path, "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-"]
    if audio_raw_path is not None:
        cmd += ["-f", "s16le", "-ar", str(int(audio_rate)), "-ac", "1",
                "-i", str(audio_raw_path)]
    cmd += ["-vf", "format=yuv420p", "-c:v", "h264_videotoolbox",
            "-b:v", bitrate]
    if audio_raw_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-movflags", "+faststart", str(out_path)]

    log_path = out_path.with_suffix(".ffmpeg.log")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=log_f)
        try:
            for f in frames:
                proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
            proc.stdin.close()
            rc = proc.wait(timeout=120)
        except BrokenPipeError:
            proc.wait(timeout=5)
            rc = proc.returncode
        finally:
            if audio_raw_path is not None:
                try:
                    audio_raw_path.unlink()
                except OSError:
                    pass
    if rc != 0:
        tail = log_path.read_text()[-2000:] if log_path.exists() else ""
        raise RuntimeError(f"ffmpeg exited {rc} encoding {out_path}: ...{tail}")
    log_path.unlink(missing_ok=True)
    return "h264_videotoolbox (ffmpeg)"


# ---------------------------------------------------------------------------
# AVFoundation encoder (fallback, UNVERIFIED — no pyobjc in this env)
# ---------------------------------------------------------------------------

def _encode_avfoundation(frames: list[np.ndarray], out_path: Path,
                          fps: float) -> str:
    """AVAssetWriter fallback for hosts without ffmpeg. Video only (the
    ffmpeg path is what carries audio); H.264 output still rides
    VideoToolbox hardware encode under the hood on Apple Silicon.

    Honesty note: this function has never been exercised — pyobjc is not
    installed in this environment (checked: `import AVFoundation` fails)
    and this project runs no CI, so nothing else has run it either. It
    is written against the documented AVAssetWriter/pixel-buffer-adaptor
    pattern but ships unverified. If ffmpeg is genuinely unavailable on
    a target Mac, `brew install ffmpeg` is the faster and proven fix;
    only fall through to this path if that is not an option.
    """
    from Foundation import NSURL
    import AVFoundation
    import CoreMedia
    import CoreVideo

    h, w = frames[0].shape[0], frames[0].shape[1]
    out_path.unlink(missing_ok=True)
    url = NSURL.fileURLWithPath_(str(out_path))
    writer, err = AVFoundation.AVAssetWriter.alloc().initWithURL_fileType_error_(
        url, AVFoundation.AVFileTypeMPEG4, None)
    if writer is None:
        raise RuntimeError(f"AVAssetWriter init failed: {err}")

    settings = {
        AVFoundation.AVVideoCodecKey: AVFoundation.AVVideoCodecTypeH264,
        AVFoundation.AVVideoWidthKey: w,
        AVFoundation.AVVideoHeightKey: h,
    }
    inp = AVFoundation.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
        AVFoundation.AVMediaTypeVideo, settings)
    inp.setExpectsMediaDataInRealTime_(False)
    px_attrs = {
        CoreVideo.kCVPixelBufferPixelFormatTypeKey: CoreVideo.kCVPixelFormatType_24RGB,
        CoreVideo.kCVPixelBufferWidthKey: w,
        CoreVideo.kCVPixelBufferHeightKey: h,
    }
    adaptor = (AVFoundation.AVAssetWriterInputPixelBufferAdaptor
               .alloc().initWithAssetWriterInput_sourcePixelBufferAttributes_(
                   inp, px_attrs))
    writer.addInput_(inp)
    writer.startWriting()
    writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, int(round(fps))))

    for i, frame in enumerate(frames):
        while not inp.isReadyForMoreMediaData():
            time.sleep(0.001)
        status, pixel_buffer = CoreVideo.CVPixelBufferPoolCreatePixelBuffer(
            None, adaptor.pixelBufferPool(), None)
        if status != 0 or pixel_buffer is None:
            raise RuntimeError(f"CVPixelBufferPoolCreatePixelBuffer failed: {status}")
        CoreVideo.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
        try:
            base = CoreVideo.CVPixelBufferGetBaseAddress(pixel_buffer)
            stride = CoreVideo.CVPixelBufferGetBytesPerRow(pixel_buffer)
            row_bytes = w * 3
            src = np.ascontiguousarray(frame, dtype=np.uint8)
            for row in range(h):
                dst_addr = base + row * stride
                ctypes.memmove(dst_addr, src[row].tobytes(), row_bytes)
        finally:
            CoreVideo.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
        pts = CoreMedia.CMTimeMake(i, int(round(fps)))
        adaptor.appendPixelBuffer_withPresentationTime_(pixel_buffer, pts)

    inp.markAsFinished()
    done = threading.Event()
    writer.finishWritingWithCompletionHandler_(lambda: done.set())
    done.wait(timeout=60)
    if writer.status() != AVFoundation.AVAssetWriterStatusCompleted:
        raise RuntimeError(f"AVAssetWriter finished with status "
                           f"{writer.status()}: {writer.error()}")
    return "h264 (AVFoundation/VideoToolbox, unverified fallback)"


# ---------------------------------------------------------------------------
# ClipRecorder
# ---------------------------------------------------------------------------

@dataclass
class ClipJob:
    name: str
    out_path: Path
    trigger_wall_t: float
    extra_seconds: float
    extra_frames: int
    pre: list = field(default_factory=list)
    post: list = field(default_factory=list, repr=False)
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False
    error: str | None = None
    codec_used: str | None = None
    handoff_latency_s: float | None = None
    encode_wall_s: float | None = None
    file_size_bytes: int | None = None
    n_frames: int | None = None

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)

    def as_receipt(self) -> dict:
        return {
            "name": self.name, "out_path": str(self.out_path), "ok": self.ok,
            "error": self.error, "codec_used": self.codec_used,
            "handoff_latency_ms": (round(self.handoff_latency_s * 1000, 4)
                                   if self.handoff_latency_s is not None else None),
            "encode_wall_s": (round(self.encode_wall_s, 4)
                              if self.encode_wall_s is not None else None),
            "file_size_bytes": self.file_size_bytes,
            "n_frames": self.n_frames,
        }


class ClipRecorder:
    """Rolling ring buffer + background hardware-encode-on-trigger.

    `push_frame` is the only per-frame call and must stay cheap; it is
    called from the show's render loop. `trigger` is the only call the
    event source makes and is a hand-off: it snapshots the ring buffer
    (a `list()` of a bounded deque — a pointer copy, not a pixel copy)
    and appends a job descriptor to a small pending list, then returns.
    All encoding work happens on one dedicated background thread.
    """

    def __init__(self, out_dir: Path | str, frame_shape=(240, 256, 3),
                 fps: float = 60.0, buffer_seconds: float = 20.0,
                 audio_rate: int = 43653, codec_prefer: str = "auto",
                 bitrate: str = "3M", copy_frames: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frame_shape = frame_shape
        self.fps = float(fps)
        self.buffer_seconds = float(buffer_seconds)
        self.audio_rate = int(audio_rate)
        self.bitrate = bitrate
        self.copy_frames = copy_frames
        self.capacity = max(1, int(round(self.fps * self.buffer_seconds)))

        codec = detect_codec(codec_prefer)
        if codec["backend"] == "none":
            raise RuntimeError(codec["detail"])
        self.hardware = codec  # {"backend", "detail"} — goes in the receipt
        self._ffmpeg_path = shutil.which("ffmpeg") if codec["backend"] == "ffmpeg_videotoolbox" else None

        self._buf: deque = deque(maxlen=self.capacity)
        self._buf_lock = threading.Lock()
        self._pending: list[ClipJob] = []
        self._pending_lock = threading.Lock()
        self._queue: "queue.Queue[ClipJob | None]" = queue.Queue()
        self.frames_pushed = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="clip-encoder")
        self._worker.start()

    # -- producer side (render loop) --------------------------------------
    def push_frame(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        """Append one frame (+ optional audio chunk) to the ring buffer."""
        f = np.array(frame, dtype=np.uint8, copy=self.copy_frames)
        a = None
        if audio is not None and len(audio):
            a = np.array(audio, dtype=np.int16, copy=self.copy_frames)
        entry = (time.monotonic(), f, a)
        with self._buf_lock:
            self._buf.append(entry)
        self.frames_pushed += 1
        if self._pending:
            self._feed_pending(entry)

    def _feed_pending(self, entry) -> None:
        # Gated on FRAME COUNT, not wall-clock elapsed: push_frame() can be
        # called at real showtime pace or, in a fast/offline replay (the
        # demo CLI below), far faster than real time. Wall-clock gating
        # made a fast replay accumulate minutes of "post-trigger" frames
        # for a nominal 10s window (caught while building this — the
        # demo took 3+ minutes and produced a ~4 min clip on a 30s ask).
        # Frame-count gating makes clip length exactly buffer_seconds +
        # extra_seconds of video regardless of production speed.
        finished = []
        with self._pending_lock:
            for job in self._pending:
                job.post.append(entry)
                if len(job.post) >= job.extra_frames:
                    finished.append(job)
            for job in finished:
                self._pending.remove(job)
        for job in finished:
            self._queue.put(job)

    # -- event side (trigger callers) --------------------------------------
    def trigger(self, name: str = "manual", extra_seconds: float = 10.0,
                out_path: Path | None = None) -> ClipJob:
        """Fire a clip event. Returns immediately (<1 ms): snapshots the
        ring buffer and hands off to the background encoder. The
        returned ClipJob's `.done` event / `.wait()` reports completion;
        polling it is optional — encoding proceeds regardless."""
        t0 = time.perf_counter()
        with self._buf_lock:
            pre = list(self._buf)
        if out_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.out_dir / f"{name}_{stamp}_{int(t0 * 1000) % 100000}.mp4"
        extra_frames = max(0, int(round(self.fps * extra_seconds)))
        job = ClipJob(name=name, out_path=Path(out_path),
                      trigger_wall_t=time.monotonic(),
                      extra_seconds=extra_seconds, extra_frames=extra_frames,
                      pre=pre)
        job.handoff_latency_s = time.perf_counter() - t0
        if extra_frames <= 0:
            self._queue.put(job)
        else:
            with self._pending_lock:
                self._pending.append(job)
        return job

    # -- background worker ---------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            self._encode(job)

    def _encode(self, job: ClipJob) -> None:
        entries = job.pre + job.post
        frames = [e[1] for e in entries]
        audio_chunks = [e[2] for e in entries if e[2] is not None]
        job.n_frames = len(frames)
        t0 = time.perf_counter()
        try:
            if not frames:
                raise RuntimeError("clip job has zero frames (buffer was empty at trigger time)")
            if self.hardware["backend"] == "ffmpeg_videotoolbox":
                job.codec_used = _encode_ffmpeg(
                    frames, audio_chunks, self.audio_rate, job.out_path,
                    self.fps, self._ffmpeg_path, self.bitrate)
            elif self.hardware["backend"] == "avfoundation":
                job.codec_used = _encode_avfoundation(frames, job.out_path, self.fps)
            else:  # pragma: no cover - guarded at __init__
                raise RuntimeError(self.hardware["detail"])
            job.encode_wall_s = time.perf_counter() - t0
            job.file_size_bytes = job.out_path.stat().st_size
            job.ok = True
        except Exception as e:
            job.error = repr(e)
            job.encode_wall_s = time.perf_counter() - t0
        finally:
            job.done.set()

    # -- lifecycle -------------------------------------------------------
    def close(self, timeout: float = 30.0) -> None:
        """Flush any in-flight pending jobs (whatever post-frames they've
        collected so far) and stop the encoder thread."""
        with self._pending_lock:
            leftover = list(self._pending)
            self._pending.clear()
        for job in leftover:
            self._queue.put(job)
        self._queue.put(None)
        self._worker.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Standalone demo / receipt CLI
# ---------------------------------------------------------------------------

def _replay_source(rom_path: str, tape_path: str, frame_skip: int, n_tape_steps: int):
    """Yield (frame, audio) pairs by replaying an SMB solution tape (raw
    per-step controller masks, cold-boot-relative) through a fresh
    nes_core.NESEnvironment, one single NES frame at a time."""
    import nes_core
    env = nes_core.NESEnvironment(rom_path)
    env.reset()
    tape = np.load(tape_path)[:n_tape_steps]
    for mask in tape:
        for _ in range(frame_skip):
            frame, _done = env.step(int(mask))
            audio = env.get_audio()
            yield np.asarray(frame), np.asarray(audio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", default=str(REPO / "roms/Super Mario Bros. (World).nes"))
    ap.add_argument("--tape", default=str(REPO / "docs/receipts/full_run/full_tape.npy"),
                    help="uint8 controller-mask array (cold-boot relative)")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--tape-steps", type=int, default=700,
                    help="how many tape entries to replay (each = frame-skip frames)")
    ap.add_argument("--buffer-seconds", type=float, default=20.0)
    ap.add_argument("--extra-seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=60.0988)
    ap.add_argument("--out-dir", default=str(REPO / "docs/receipts/show_lane"))
    ap.add_argument("--receipt", default=str(REPO / "docs/receipts/show_lane/clips_receipt.json"))
    ap.add_argument("--bitrate", default="1500k")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = detect_codec()
    print(f"[show_clips] hardware encode backend: {codec['backend']} — {codec['detail']}")

    recorder = ClipRecorder(out_dir=out_dir, fps=args.fps,
                            buffer_seconds=args.buffer_seconds,
                            bitrate=args.bitrate)
    print(f"[show_clips] ring buffer capacity: {recorder.capacity} frames "
          f"(~{recorder.capacity / args.fps:.1f}s @ {args.fps:.2f} fps)")

    job = None
    triggered_at_frame = None
    target_total_frames = None
    t_wall0 = time.perf_counter()
    for frame, audio in _replay_source(args.rom, args.tape, args.frame_skip, args.tape_steps):
        recorder.push_frame(frame, None if args.no_audio else audio)
        n = recorder.frames_pushed
        if job is None and n >= recorder.capacity:
            # Ring buffer is fully warmed up — fire the fake 'solution'
            # trigger mid-replay, exactly as a real show event would.
            job = recorder.trigger("solution", extra_seconds=args.extra_seconds)
            triggered_at_frame = n
            target_total_frames = n + job.extra_frames
            print(f"[show_clips] TRIGGER 'solution' fired at frame {n} "
                  f"(t={n / args.fps:.1f}s) — handoff latency "
                  f"{job.handoff_latency_s * 1000:.4f} ms")
        # Stop generating the INSTANT the clip's frame budget is met —
        # not when the encode finishes. Generation and encode run on
        # separate threads under one GIL; letting this loop keep
        # simulating NES frames while the encoder thread is trying to
        # write to ffmpeg's stdin measurably starves the encoder of the
        # GIL (caught while building this: encode_wall_s read ~60s for
        # a clip that hardware-encodes in a few seconds once the
        # generator loop is stopped promptly).
        if target_total_frames is not None and n >= target_total_frames:
            break

    if job is None:
        recorder.close()
        raise SystemExit(
            f"tape too short to fill the {recorder.capacity}-frame ring "
            f"buffer ({recorder.frames_pushed} frames pushed) — raise "
            f"--tape-steps")

    # If the tape ran out before the post-trigger window filled, keep
    # stepping the environment on no-ops so the demo still produces a
    # real clip of the requested length.
    if recorder.frames_pushed < target_total_frames:
        import nes_core
        env = nes_core.NESEnvironment(args.rom)
        env.reset()
        while recorder.frames_pushed < target_total_frames:
            frame, _ = env.step(0)
            audio = env.get_audio()
            recorder.push_frame(np.asarray(frame), None if args.no_audio else np.asarray(audio))

    generation_wall_s = time.perf_counter() - t_wall0
    job.wait(timeout=90)
    replay_wall_s = time.perf_counter() - t_wall0
    recorder.close()

    if not job.ok:
        print(f"[show_clips] ENCODE FAILED: {job.error}")

    receipt = {
        "rom": Path(args.rom).name,
        "tape": str(Path(args.tape).relative_to(REPO)) if Path(args.tape).is_relative_to(REPO) else str(args.tape),
        "frame_skip": args.frame_skip,
        "fps": args.fps,
        "buffer_seconds": args.buffer_seconds,
        "extra_seconds": args.extra_seconds,
        "hardware": codec,
        "trigger": {
            "name": "solution",
            "frame_index": triggered_at_frame,
            "sim_time_s": round(triggered_at_frame / args.fps, 3),
            "handoff_latency_ms": round(job.handoff_latency_s * 1000, 4),
        },
        "clip": job.as_receipt(),
        "demo_frame_generation_wall_s": round(generation_wall_s, 3),
        "demo_total_wall_s": round(replay_wall_s, 3),
        "notes": [
            "handoff_latency_ms is measured inside ClipRecorder.trigger() "
            "itself (buffer snapshot + job hand-off only) — it excludes "
            "encoding, which runs on a background thread.",
            "encode_wall_s is the ffmpeg subprocess's wall time for the "
            "full clip (buffer_seconds + extra_seconds of frames), timed "
            "on the background encoder thread only.",
            "demo_frame_generation_wall_s is the un-paced replay's own "
            "cost to simulate the frames (this CLI runs nes_core as fast "
            "as it can, not at 60 fps) — it is not a ClipRecorder cost.",
            "AVFoundation/pyobjc fallback exists in this module but is "
            "UNVERIFIED in this environment (pyobjc not installed); "
            "this run used the ffmpeg h264_videotoolbox path.",
        ],
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"[show_clips] receipt written to {args.receipt}")
    return 0 if job.ok else 1


if __name__ == "__main__":
    sys.exit(main())
