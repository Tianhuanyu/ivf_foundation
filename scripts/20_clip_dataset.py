#!/usr/bin/env python3
"""Video clip dataset for V-JEPA2 continued pretraining.

Reads a manifest of video paths and yields normalized clips of shape
[C, T, H, W] (channels-first, time second) — the layout V-JEPA2 expects.

Frame reading tries decord first, then torchvision.io.read_video (PyAV backend),
so it works whether or not decord installed cleanly on Python 3.12.

Run directly for a smoke test:
  python 20_clip_dataset.py --manifest .../val_videos.txt --limit 4
"""
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

# ---- pluggable frame reader -------------------------------------------------
_READER = None


def _pick_reader():
    global _READER
    if _READER is not None:
        return _READER
    try:
        import decord  # noqa
        _READER = "decord"
    except Exception:
        _READER = "torchvision"
    return _READER


def read_clip_frames(path, num_frames, stride, train):
    """Return uint8 tensor [T, H, W, C] sampled from the video."""
    reader = _pick_reader()
    if reader == "decord":
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, num_threads=1)
        total = len(vr)
        idx = _clip_indices(total, num_frames, stride, train)
        return vr.get_batch(idx)  # [T,H,W,C] uint8
    else:
        from torchvision.io import read_video
        # read_video is not frame-indexed cheaply; read all then sample.
        v, _, _ = read_video(path, output_format="THWC", pts_unit="sec")
        total = v.shape[0]
        idx = _clip_indices(total, num_frames, stride, train)
        return v[idx]


def _clip_indices(total, num_frames, stride, train):
    span = min(num_frames * stride, max(total, 1))
    if total <= num_frames:
        base = list(range(total)) + [total - 1] * (num_frames - total)
        return torch.tensor(base[:num_frames])
    max_start = max(0, total - span)
    start = random.randint(0, max_start) if train else max_start // 2
    idx = [min(start + i * stride, total - 1) for i in range(num_frames)]
    return torch.tensor(idx)


# ---- dataset ----------------------------------------------------------------
class VideoClipDataset(Dataset):
    def __init__(self, manifest, num_frames=16, stride=4, size=256, train=True):
        self.videos = [l.strip() for l in Path(manifest).read_text().splitlines() if l.strip()]
        self.num_frames = num_frames
        self.stride = stride
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.videos)

    def _spatial(self, clip):  # clip: float [T,C,H,W] in [0,1]
        import torch.nn.functional as F
        T, C, H, W = clip.shape
        scale = self.size / min(H, W)
        nh, nw = int(round(H * scale)), int(round(W * scale))
        clip = F.interpolate(clip, size=(nh, nw), mode="bilinear", align_corners=False)
        if self.train:
            top = random.randint(0, nh - self.size)
            left = random.randint(0, nw - self.size)
        else:
            top, left = (nh - self.size) // 2, (nw - self.size) // 2
        return clip[:, :, top:top + self.size, left:left + self.size]

    def __getitem__(self, i):
        path = self.videos[i]
        frames = read_clip_frames(path, self.num_frames, self.stride, self.train)  # [T,H,W,C] uint8
        clip = frames.permute(0, 3, 1, 2).float() / 255.0            # [T,C,H,W]
        clip = self._spatial(clip)                                    # [T,C,size,size]
        clip = clip.permute(1, 0, 2, 3)                               # [C,T,H,W]
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=4)
    args = ap.parse_args()

    ds = VideoClipDataset(args.manifest, args.num_frames, args.stride, args.size, train=True)
    ds.videos = ds.videos[: args.limit]
    print(f"reader={_pick_reader()}  clips={len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch, num_workers=0)
    for b, clips in enumerate(dl):
        print(f"batch {b}: {tuple(clips.shape)}  dtype={clips.dtype} "
              f"min={clips.min():.2f} max={clips.max():.2f}")
    print("OK: clip pipeline produces [B, C, T, H, W] tensors")


if __name__ == "__main__":
    main()
