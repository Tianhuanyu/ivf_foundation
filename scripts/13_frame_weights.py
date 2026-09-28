#!/usr/bin/env python3
"""Per-frame temporal importance weights for DAPT SSL pretraining, derived from
the same .me.png motion-energy sidecars scripts/12_motion_energy.py already
generated (spatial crop guidance). This is the TEMPORAL counterpart: crop_sampler
(dinov3/data/crop_sampler.py) decides WHERE in a chosen frame to crop; this
decides how likely each frame is to be CHOSEN at all.

Why: frames are currently sampled uniformly at random across the whole flat
pool (dinov3/data/samplers.py's InfiniteSampler etc, over Frames.samples).
For an event-driven procedure like ICSI injection, the actual needle-tissue
interaction is a small fraction of a video's kept frames (measured: the
visible injection event in one clip spanned ~9/54 kept frames, ~17%, and
that's a comparatively SHORT clip -- longer ones are worse). Uniform sampling
massively under-represents exactly the frames most informative for the
target classes (needle_tip) this whole line of work cares about.

Approach (per MGSampler, arXiv:2104.09952 -- motion-uniform sampling from a
cumulative motion distribution; adapted here to a static importance-weighted
resampling since we don't re-extract frames, just re-weight the ones we
already have): per-frame score = a percentile of its .me.png map (a peak-
intensity read, not a mean -- an eventful frame's signal is spatially
localized, e.g. concentrated on the egg/needle, not spread across the whole
frame, so a percentile isolates "is there a hot region" better than an
average that gets diluted by background). Weight = score ** gamma, gamma<1
so oversampling is tempered rather than making training collapse onto a
handful of frames repeated every epoch -- SSL/self-distillation specifically
needs broad view diversity, not just "the right" frames (same tension the
crop_sampler temperature already manages spatially).

Usage:
  python scripts/13_frame_weights.py --frames-root D:/Video/domain_transfer/frames_hires --split train
  python scripts/13_frame_weights.py --frames-root D:/Video/domain_transfer/frames_hires --split val

Output: <frames-root>/<split>/frame_weights.npy (float32, aligned index-for-
index with the EXACT sample ordering dinov3/data/datasets/frames.py's
Frames.__init__ produces: os.walk + extension filter excluding *.me.png,
then samples.sort()) plus frame_weights_meta.json for provenance.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from _common import SIDECAR_SUFFIX, sidecar_path

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_samples_like_frames_dataset(root: str):
    """Mirrors Frames.__init__ exactly (dinov3/data/datasets/frames.py) so the
    weight array this script produces stays index-aligned with that class's
    self.samples, without needing the dinov3 package importable here."""
    samples = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(SIDECAR_SUFFIX):
                continue
            if fn.lower().endswith(_EXTS):
                samples.append(os.path.join(dirpath, fn))
    samples.sort()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="percentile of each frame's .me.png map used as its "
                         "raw importance score")
    ap.add_argument("--gamma", type=float, default=0.4,
                    help="score compression exponent (<1): tempers oversampling "
                         "so SSL training doesn't collapse onto a handful of "
                         "high-motion frames repeated every epoch")
    args = ap.parse_args()

    root = str(Path(args.frames_root) / args.split)
    t0 = time.time()
    samples = list_samples_like_frames_dataset(root)
    n = len(samples)
    print(f"[{args.split}] {n} frames (listing took {time.time()-t0:.1f}s)")

    scores = np.full(n, np.nan, dtype=np.float32)
    n_missing = 0
    for i, jpg_path in enumerate(samples):
        me_path = sidecar_path(jpg_path)
        me = cv2.imread(me_path, cv2.IMREAD_GRAYSCALE)
        if me is None:
            n_missing += 1
            continue
        scores[i] = np.percentile(me, args.percentile)
        if i % 100000 == 0 and i > 0:
            print(f"  {i}/{n} ({time.time()-t0:.0f}s elapsed, {n_missing} missing so far)")

    valid = ~np.isnan(scores)
    if not valid.any():
        raise RuntimeError("no valid .me.png sidecars found under this root")
    median_score = float(np.median(scores[valid]))
    scores[~valid] = median_score  # neutral default for the ~2.5% videos with no sidecar,
                                    # same "degrade gracefully, don't zero out" philosophy as
                                    # crop_sampler.py's own uniform fallback

    weights = np.power(np.clip(scores, 1e-6, None), args.gamma).astype(np.float32)
    weights /= weights.sum()

    out_path = Path(root) / "frame_weights.npy"
    np.save(out_path, weights)
    meta = {
        "n_frames": n,
        "n_missing_sidecar": n_missing,
        "percentile": args.percentile,
        "gamma": args.gamma,
        "median_score_used_for_missing": median_score,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_ratio_max_over_median": float(weights.max() / np.median(weights)),
    }
    with open(Path(root) / "frame_weights_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"DONE [{args.split}]: wrote {out_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
