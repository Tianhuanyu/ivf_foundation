#!/usr/bin/env python3
"""Generate motion-energy sidecar maps for MotionCropSampler (dinov3 CropSampler
refactor, see repos/dinov3/dinov3/data/crop_sampler.py).

For each frame already kept by 11_extract_frames.py (at frames_hires, sampled
at eff_fps << native fps), this re-decodes the SOURCE VIDEO at its native
frame rate (usually 30fps) and computes a motion-energy map from the native
frames immediately surrounding that kept frame's timestamp -- NOT from the
neighboring *kept* frames, which are ~200ms (at 5fps) apart and would wash out
the fast, local motion (instrument tip, sperm) this signal is meant to find.

Output: <kept_frame_path>.me.png, grayscale, short side --out-short (default
96, matching MotionCropSampler's expected sidecar), one per kept frame.
Idempotent/resumable: a video is skipped once every one of its kept frames
already has a sidecar.

Algorithm per video (single whole-clip decode, then vectorized):
  1. List this video's already-extracted kept frames, sorted by index ->
     n_kept. Target timestamp for kept frame k (0-based): t_k = k / eff_fps,
     eff_fps := n_kept / duration (recovers whatever fps 11_extract_frames.py
     effectively used, including its per-video --max-frames throttling,
     without needing to know its exact args).
  2. Decode the whole clip at native fps, downscaled to --calc-short
     (grayscale). Contrast-stretch the whole clip by a global percentile
     affine map -- this footage is natively very low-contrast (~46-64/255 in
     a measured sample), so a raw abs-diff is dominated by encoder
     quantization noise, not real motion. Consecutive-frame abs-diff then
     gives a per-native-frame motion map, timestamped at the midpoint of the
     pair.
  3. Per-pixel baseline normalization: divide each pixel's diff by that
     pixel's own typical window-max magnitude over the whole clip (see
     process_one's docstring comment for why this -- not a single frame-wide
     ceiling -- is needed to keep a dominant static high-contrast boundary
     from drowning out weaker-but-real motion elsewhere).
  4. Sweep: for each kept-frame target t_k, take the max over all
     (normalized) native-frame diff maps within [t_k - window/2, t_k +
     window/2], Gaussian-blur it spatially, clip to --ceiling-mult and scale
     to uint8, downsample to --out-short, write the PNG.
"""
import argparse
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d

PER_VIDEO_TIMEOUT_SEC = 180

cv2.setNumThreads(0)  # avoid cv2-internal-threads + process-pool fork issues


def to_native_path(p: str) -> str:
    """Manifests/configs are written as WSL paths (/mnt/d/...). This script
    also runs directly on native Windows (see README/session notes -- WSL's
    own command-execution service turned out to be too unstable for this
    long-running a job), where that path doesn't resolve. Translate
    /mnt/<drive>/... -> <DRIVE>:/... only on Windows; a no-op everywhere else."""
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)", p)
        if m:
            return f"{m.group(1).upper()}:/{m.group(2)}"
    return p


def parse_stage(name: str) -> str:
    stem = Path(name).stem
    stage = stem.split("_MI_", 1)[1] if "_MI_" in stem else stem
    return re.sub(r"[_\d]+$", "", stage) or "UNKNOWN"


def resize_gray(frame_bgr, short: int):
    h, w = frame_bgr.shape[:2]
    if w <= h:
        new_w, new_h = short, max(1, round(h * short / w))
    else:
        new_h, new_w = short, max(1, round(w * short / h))
    small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def finalize_map(window_diffs, sigma: float, out_short: int, ceiling_mult: float, gamma: float):
    # window_diffs are ALREADY per-pixel-baseline-normalized (see process_one):
    # each pixel's diff has been divided by that pixel's own typical jitter
    # level, so a value of 1.0 here means "normal for this location", not some
    # frame-wide absolute brightness. That's what lets a genuinely-flat window
    # stay dark while still giving the static high-contrast boundary (which
    # has a HIGH baseline of its own, factored out) a fair comparison against
    # weaker-but-real motion elsewhere (which has a LOW baseline, so the same
    # absolute jitter shows up as a much bigger multiple of ITS baseline).
    #
    # ceiling_mult: output saturates to 255 at this many multiples of a
    # pixel's own baseline -- i.e. "how anomalous is this, for this location",
    # not "how anomalous is this, compared to the single hottest pixel in the
    # frame" (which is what a plain max-normalize would measure, and why that
    # approach let the boundary crush every other edge toward zero: at the
    # scale of the whole frame's own max, everything else IS small).
    if not window_diffs:
        return None
    motion = np.max(np.stack(window_diffs, axis=0), axis=0)
    if sigma > 0:
        motion = cv2.GaussianBlur(motion, (0, 0), sigma)
    motion = np.clip(motion, 0, ceiling_mult) / max(ceiling_mult, 1e-6)
    if gamma != 1.0:
        motion = motion ** gamma
    motion = np.clip(motion * 255.0, 0, 255).astype(np.uint8)
    if motion.shape[0] != out_short and motion.shape[1] != out_short:
        h, w = motion.shape
        if w <= h:
            new_w, new_h = out_short, max(1, round(h * out_short / w))
        else:
            new_h, new_w = out_short, max(1, round(w * out_short / h))
        motion = cv2.resize(motion, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return motion


def list_kept_frames(video: str, frames_root: str, split: str, stage_cache: dict = None):
    """Cheap (no cv2/video decode) lookup of a video's already-extracted kept
    frames -- used both by process_one and by the parent process's pre-check
    (main() skips spawning a whole subprocess, with its cv2/numpy/scipy
    import cost, for videos that are already fully done).

    stage_cache: many videos share one stage directory (e.g. ICSI_injection
    has 225k+ files across hundreds of videos), and that directory lives on
    the actual Windows NTFS drive here (no more WSL/drvfs bridge, but still
    slow to re-glob per call). main()'s pre-check over all ~2777 videos was
    taking many minutes doing a fresh glob per video; pass a dict here (kept
    alive across calls in the same process) to list each stage directory
    ONCE and look members up from that in-memory grouping instead."""
    video = to_native_path(video)
    vp = Path(video)
    stage = parse_stage(vp.name)
    frame_dir = Path(frames_root) / split / stage
    if stage_cache is None:
        kept = sorted(frame_dir.glob(f"{vp.stem}_f*.jpg"),
                      key=lambda p: int(re.search(r"_f(\d+)$", p.stem).group(1)))
        return video, kept

    key = str(frame_dir)
    if key not in stage_cache:
        by_stem = {}
        if frame_dir.is_dir():
            for f in frame_dir.iterdir():
                m = re.match(r"^(.*)_f(\d+)$", f.stem)
                if f.suffix == ".jpg" and m:
                    by_stem.setdefault(m.group(1), []).append(f)
        stage_cache[key] = by_stem
    kept = sorted(stage_cache[key].get(vp.stem, []),
                  key=lambda p: int(re.search(r"_f(\d+)$", p.stem).group(1)))
    return video, kept


def has_all_sidecars(kept, frame_dir: Path, sidecar_cache: dict) -> bool:
    """Same one-listing-per-directory trick as list_kept_frames's stage_cache,
    for the ".me.png already exists" check -- doing one os.stat() per kept
    frame (hundreds per video) across ~2777 videos was the other half of why
    main()'s pre-check was taking many minutes before any real work started."""
    key = str(frame_dir)
    if key not in sidecar_cache:
        existing = set()
        if frame_dir.is_dir():
            for f in frame_dir.iterdir():
                if f.name.endswith(".me.png"):
                    existing.add(f.name[: -len(".me.png")])
        sidecar_cache[key] = existing
    existing = sidecar_cache[key]
    return all(p.name in existing for p in kept)


def process_one(video: str, frames_root: str, split: str, out_short: int, calc_short: int,
                window_sec: float, sigma: float, stretch_pct: tuple, baseline_pct: float,
                ceiling_mult: float, gamma: float):
    video, kept = list_kept_frames(video, frames_root, split)
    if not kept:
        return (video, "no_kept_frames", 0)
    if all(Path(str(p) + ".me.png").exists() for p in kept):
        return (video, "skip", len(kept))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return (video, "ERROR: cv2 could not open video", 0)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Duration from cv2's own metadata (frame_count/fps) instead of shelling
    # out to ffprobe -- one less external-tool dependency, and works
    # identically whether this runs under WSL or native Windows Python.
    # CAP_PROP_FRAME_COUNT is 0 for some encodings in this dataset (confirmed:
    # cv2 opens and reads them fine, just can't report a count upfront) -- in
    # that case we can't compute a duration-relative frame cap in advance, so
    # decode is capped at ABSOLUTE_MAX_FRAMES alone and duration is derived
    # from however many frames actually got decoded (see below).
    #
    # ABSOLUTE_MAX_FRAMES: guards against (a) corrupted clips where cv2's
    # ffmpeg backend keeps returning frames well past the video's real
    # duration, and (b) a video whose own duration metadata is itself
    # anomalous (found one reporting 741s vs. a first sample of the dataset
    # being 5-140s) -- a duration-relative cap alone is useless there since it
    # just scales up right along with the bad value. Originally set to 5000
    # (166s) from that first sample, but the dataset spans many months and a
    # later batch has legitimate 200s+ clips (confirmed one at 225s/6766
    # frames, not corrupted) that 5000 was wrongly rejecting -- raised to
    # 12000 (400s/6.7min) to clear real cases like that while still bounding
    # the 741s (~22k frame) one.
    ABSOLUTE_MAX_FRAMES = 12000
    reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if reported_frames > 0 and native_fps > 0:
        duration = reported_frames / native_fps
        max_frames = max(min(int(duration * native_fps * 2), ABSOLUTE_MAX_FRAMES), 300)
    else:
        duration = None  # derived from actual decode count below
        max_frames = ABSOLUTE_MAX_FRAMES

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_gray(frame, calc_short))
        if len(frames) > max_frames:
            break
    cap.release()
    if len(frames) < 2:
        return (video, "ERROR: too few native frames decoded", 0)
    if len(frames) > max_frames:
        return (video, f"ERROR: decoded frame count exceeded {max_frames} "
                        f"(corrupted video?), aborted", 0)
    if duration is None:
        duration = len(frames) / native_fps
    if duration <= 0:
        return (video, "ERROR: bad duration", 0)

    n_kept = len(kept)
    eff_fps = n_kept / duration
    targets = [k / eff_fps for k in range(n_kept)]
    half = window_sec / 2.0

    frames = np.stack(frames)
    lo, hi = np.percentile(frames, stretch_pct)
    frames = np.clip((frames - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255)
    diffs = np.abs(frames[1:] - frames[:-1])
    del frames  # not needed past this point; a long clip's copy is ~350MB+, free it now
    diff_t = [(i + 0.5) / native_fps for i in range(len(diffs))]

    # Per-pixel baseline normalization: a global (whole-frame) amplitude limit
    # doesn't work here because the dominant static high-contrast boundary
    # (egg/dish edge) isn't a rare outlier -- it's the largest diff in nearly
    # EVERY window, so any single global ceiling that lets it saturate also
    # saturates almost everything else (empirically: median output pixel was
    # 216/255 with a global ceiling). The boundary's jitter is real and
    # location-specific, not frame-wide, so factor it out per PIXEL instead:
    # divide each pixel's diff by that pixel's own typical jitter magnitude
    # over the whole clip. A pixel that's always noisy (the boundary) needs a
    # large multiple of ITS baseline to register as anomalous; a pixel that's
    # normally dead quiet (most of the frame) only needs a small absolute diff
    # to register the same way -- exactly the "don't let the boundary drown
    # out other, weaker edges" behavior asked for, without suppressing the
    # boundary where it legitimately is the strongest signal.
    #
    # The baseline must be calibrated against the same statistic we actually
    # threshold (window-max over ~window_sec of native frames), not against
    # single-frame diffs: max-of-~6 is an order statistic already pulled well
    # above any single-frame percentile, so a single-frame-calibrated baseline
    # made nearly every window's max exceed it (empirically: median output
    # saturated at 255 with baseline_pct=75 calibrated on single-frame diffs).
    win_n = max(1, round(window_sec * native_fps))
    windowed = maximum_filter1d(diffs, size=win_n, axis=0, mode="nearest")
    pixel_baseline = np.percentile(windowed, baseline_pct, axis=0)
    floor = max(float(np.percentile(windowed, 50)), 0.5)
    pixel_baseline = np.maximum(pixel_baseline, floor)
    del windowed  # same size as diffs (~350MB+ for a long clip), free before dividing
    diffs /= pixel_baseline[None, :, :]  # in-place: skip allocating a second full-size array

    written = 0
    lo_ptr = 0
    for kept_idx in range(n_kept):
        t_lo, t_hi = targets[kept_idx] - half, targets[kept_idx] + half
        while lo_ptr < len(diff_t) and diff_t[lo_ptr] < t_lo:
            lo_ptr += 1
        hi_ptr = lo_ptr
        while hi_ptr < len(diff_t) and diff_t[hi_ptr] <= t_hi:
            hi_ptr += 1
        window_diffs = diffs[lo_ptr:hi_ptr]
        motion = finalize_map(list(window_diffs), sigma, out_short, ceiling_mult, gamma)
        if motion is not None:
            cv2.imwrite(str(kept[kept_idx]) + ".me.png", motion)
            written += 1

    status = "ok" if written == n_kept else f"partial ({written}/{n_kept})"
    return (video, status, written)


def _worker_entry(video, frames_root, split, out_short, calc_short, window_sec, sigma,
                  stretch_pct, baseline_pct, ceiling_mult, gamma, q):
    # Runs process_one in its own subprocess (see main()'s dispatch loop) so a
    # single video that hangs -- observed in practice: a bare cv2.read() call
    # that never returns on some corrupted file, which ABSOLUTE_MAX_FRAMES
    # can't catch since that check only runs AFTER a read call returns -- can
    # be killed by the parent (TerminateProcess/SIGKILL always works, unlike
    # trying to interrupt a stuck C-level call from a signal handler or
    # another thread) without taking the rest of the run down with it.
    try:
        result = process_one(video, frames_root, split, out_short, calc_short, window_sec,
                              sigma, stretch_pct, baseline_pct, ceiling_mult, gamma)
    except Exception as e:
        result = (video, f"ERROR: exception: {e}", 0)
    q.put(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--frames-root", default="/mnt/d/Video/domain_transfer/frames_hires")
    ap.add_argument("--out-short", type=int, default=96,
                    help="sidecar short-side px, must match MotionCropSampler's expectation")
    ap.add_argument("--calc-short", type=int, default=192,
                    help="working resolution for decode+diff (higher than --out-short to "
                         "preserve thin-instrument signal; downsampled to --out-short at save)")
    ap.add_argument("--stretch-pct", type=float, nargs=2, default=(1.0, 99.0),
                    help="whole-clip percentile range stretched to [0,255] before differencing "
                         "-- this footage is natively very low-contrast (a sampled clip measured "
                         "~46-64/255), so a raw abs-diff is dominated by encoder quantization "
                         "noise rather than real motion")
    ap.add_argument("--window-sec", type=float, default=0.2,
                    help="total temporal window (sec) of native frames around each kept "
                         "frame's timestamp to compute motion energy from")
    ap.add_argument("--sigma", type=float, default=1.5, help="spatial Gaussian blur sigma")
    ap.add_argument("--baseline-pct", type=float, default=75.0,
                    help="per-pixel amplitude limiting: each pixel's diff is divided by ITS OWN "
                         "baseline_pct-th percentile jitter magnitude over the whole clip before "
                         "aggregating. The dominant static high-contrast boundary (egg/dish edge) "
                         "legitimately produces the largest diff almost every frame at ITS "
                         "location; a single frame-wide ceiling/max there crushes every other, "
                         "weaker-but-real edge (needle, sperm) toward zero because they're compared "
                         "against the boundary's scale instead of their own. Per-pixel normalization "
                         "keeps the boundary's own signal intact there while restoring sensitivity "
                         "elsewhere.")
    ap.add_argument("--ceiling-mult", type=float, default=4.0,
                    help="output saturates to 255 at this many multiples of a pixel's own "
                         "baseline (see --baseline-pct)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="post-clip gamma (<1 lifts mid/low values); usually unneeded once "
                         "per-pixel baseline normalization is applied, kept as a knob")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (smoke test)")
    args = ap.parse_args()
    args.frames_root = to_native_path(args.frames_root)

    videos = [l.strip() for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        videos = videos[: args.limit]

    # Cheap pre-check in the parent (no cv2/numpy import, just directory
    # listings): skip spawning a whole subprocess for videos that are already
    # fully done. With most re-runs mostly hitting "skip", this avoids paying
    # per-video process-spawn overhead (Windows uses "spawn", which re-imports
    # cv2/numpy/scipy in every child) for work that's already finished.
    # stage_cache/sidecar_cache: list each stage directory ONCE (many videos
    # share one, e.g. ICSI_injection has 225k+ files across hundreds of
    # videos) instead of re-globbing/re-stat'ing it per video -- without this
    # the pre-check alone took several minutes before any real work started.
    stage_cache, sidecar_cache = {}, {}
    pending = []
    ok = skip = errors = total_written = 0
    for v in videos:
        video, kept = list_kept_frames(v, args.frames_root, args.split, stage_cache)
        if not kept:
            errors += 1
            print(f"  {Path(video).name}: no_kept_frames")
        elif has_all_sidecars(kept, kept[0].parent, sidecar_cache):
            skip += 1
        else:
            pending.append(video)

    print(f"[{args.split}] {len(videos)} videos ({len(pending)} pending, {skip} already done) "
          f"-> sidecars under {args.frames_root} "
          f"(out_short={args.out_short}, window_sec={args.window_sec}, workers={args.workers}, "
          f"per_video_timeout={PER_VIDEO_TIMEOUT_SEC}s)")

    def launch(video):
        q = mp.Queue()
        p = mp.Process(target=_worker_entry, args=(
            video, args.frames_root, args.split, args.out_short, args.calc_short,
            args.window_sec, args.sigma, tuple(args.stretch_pct), args.baseline_pct,
            args.ceiling_mult, args.gamma, q))
        p.start()
        return p, q

    remaining = list(pending)
    inflight = {}  # Process -> (video, start_time, Queue)
    for _ in range(min(args.workers, len(remaining))):
        v = remaining.pop(0)
        p, q = launch(v)
        inflight[p] = (v, time.time(), q)

    i = 0
    while inflight:
        time.sleep(0.3)
        timed_out = [p for p, (_, start, _) in inflight.items()
                     if time.time() - start > PER_VIDEO_TIMEOUT_SEC]
        finished = [p for p in inflight if not p.is_alive() and p not in timed_out]

        for p in timed_out + finished:
            video, start_time, q = inflight.pop(p)
            i += 1
            if p in timed_out:
                p.terminate()
                p.join(5)
                if p.is_alive():
                    p.kill()
                status, n = f"ERROR: timeout after {PER_VIDEO_TIMEOUT_SEC}s", 0
            else:
                # Drain the queue BEFORE join(): the child's feeder thread may
                # still be flushing its (tiny) result tuple into the pipe for
                # a moment after the process itself looks exited, and joining
                # first risks missing it.
                try:
                    _, status, n = q.get(timeout=5)
                except Exception:
                    status, n = "ERROR: worker exited with no result (crashed?)", 0
                p.join()
            q.close()

            if status == "skip":
                skip += 1
            elif status.startswith("ERROR") or status == "no_kept_frames":
                errors += 1
                print(f"  [{i}/{len(pending)}] {Path(video).name}: {status}")
            else:
                ok += 1
                total_written += n
                if status != "ok":
                    print(f"  [{i}/{len(pending)}] {Path(video).name}: {status}")
            if i % 100 == 0 or i == len(pending):
                print(f"  progress {i}/{len(pending)} pending processed "
                      f"(+{skip} pre-skipped)  ok={ok} skip={skip} "
                      f"err={errors} sidecars_written={total_written}")

            if remaining:
                v = remaining.pop(0)
                p2, q2 = launch(v)
                inflight[p2] = (v, time.time(), q2)

    print(f"DONE [{args.split}]: ok={ok} skip={skip} err={errors} "
          f"total_sidecars_written={total_written}")


if __name__ == "__main__":
    main()
