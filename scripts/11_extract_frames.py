#!/usr/bin/env python3
"""Extract frames from videos for DINOv3 image-domain continued pretraining.

Reads a manifest (train_videos.txt / val_videos.txt), samples frames at a fixed
fps, resizes so the short side = --short px (keeps aspect), and writes JPGs to
  <out>/<split>/<stage>/<video_stem>_f%04d.jpg
which doubles as an ImageFolder for the linear-probe / kNN evaluation.

Uses ffmpeg (system) + a process pool. Idempotent: videos whose output folder
already has frames are skipped, so it resumes cleanly.
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from _common import DT_ROOT, parse_stage, read_manifest


def probe_duration(video: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video],
            check=True, capture_output=True, text=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def extract_one(video: str, out_root: str, split: str, fps: float, short: int,
                quality: int, max_frames: int, grayscale: bool = False):
    vp = Path(video)
    stage = parse_stage(vp.name)
    out_dir = Path(out_root) / split / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{vp.stem}_f%04d.jpg")
    # resume: skip if this video already produced frames
    if any(out_dir.glob(f"{vp.stem}_f*.jpg")):
        return (video, "skip", 0)
    # Cap frames per video and spread evenly: lower the sampling fps for long
    # clips so no single session dominates the SSL set with near-duplicates.
    eff_fps = fps
    if max_frames > 0:
        dur = probe_duration(video)
        if dur > 0:
            eff_fps = min(fps, max_frames / dur)
    vf = (f"fps={eff_fps},scale='if(gt(iw,ih),-2,{short})':'if(gt(iw,ih),{short},-2)'")
    if grayscale:
        vf += ",format=gray"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(vp),
           "-vf", vf, "-q:v", str(quality), pattern]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return (video, f"ERROR: {e.stderr.strip()[:200]}", 0)
    n = len(list(out_dir.glob(f"{vp.stem}_f*.jpg")))
    return (video, "ok", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="train_videos.txt or val_videos.txt")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", default=f"{DT_ROOT}/frames")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second")
    ap.add_argument("--short", type=int, default=256, help="short-side resize (px)")
    ap.add_argument("--quality", type=int, default=3, help="ffmpeg -q:v (2=best..31)")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="cap frames per video (evenly spread); 0 = no cap")
    ap.add_argument("--grayscale", action="store_true",
                    help="输出灰度 JPG(省磁盘;显微图本就灰度,细路 I-JEPA 用)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (smoke test)")
    args = ap.parse_args()

    videos = read_manifest(args.manifest, args.limit)
    print(f"[{args.split}] {len(videos)} videos -> {args.out} "
          f"(fps={args.fps}, short={args.short}, workers={args.workers})")

    total_frames = ok = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, v, args.out, args.split, args.fps,
                          args.short, args.quality, args.max_frames, args.grayscale)
                for v in videos]
        for i, fut in enumerate(as_completed(futs), 1):
            video, status, n = fut.result()
            if status == "ok":
                ok += 1; total_frames += n
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print(f"  [{i}/{len(videos)}] {Path(video).name}: {status}")
            if i % 100 == 0 or i == len(videos):
                print(f"  progress {i}/{len(videos)}  ok={ok} skip={skipped} "
                      f"err={errors} frames={total_frames}")

    print(f"DONE [{args.split}]: ok={ok} skip={skipped} err={errors} "
          f"total_frames={total_frames}")


if __name__ == "__main__":
    main()
