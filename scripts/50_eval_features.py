#!/usr/bin/env python3
"""Feature-quality evaluation for the domain-transfer backbones.

Freeze a backbone, extract features on labeled frames (the 12 operation stages),
split BY VIDEO (no frame leakage) and stratified so all stages appear in both
probe-train and probe-test, then report kNN + linear-probe accuracy.

Run once on the ORIGINAL weights for a baseline, again on an adapted checkpoint
to quantify the domain transfer (before vs after).

  # DINOv3 baseline (original weights)
  python 50_eval_features.py --backbone dinov3 --weights weights/dinov3_vits16
  # DINOv3 after DAPT
  python 50_eval_features.py --backbone dinov3 --weights weights/dinov3_vits16_dapt
"""
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FRAMES_ROOT = os.environ.get("DT_ROOT", "/mnt/d/Video/domain_transfer") + "/frames"


def video_stem(fp: Path) -> str:
    import re
    return re.sub(r"_f\d+$", "", fp.stem)


def gather_frames(cap_per_stage, seed):
    """Return list of (path, stage, video_stem), capped per stage, spread over videos."""
    rng = random.Random(seed)
    by_stage_video = defaultdict(lambda: defaultdict(list))
    for split in ("train", "val"):
        root = Path(FRAMES_ROOT) / split
        if not root.exists():
            continue
        for stage_dir in sorted(root.iterdir()):
            if not stage_dir.is_dir():
                continue
            for jpg in stage_dir.glob("*.jpg"):
                by_stage_video[stage_dir.name][video_stem(jpg)].append(jpg)
    items = []
    for stage, vids in by_stage_video.items():
        pool = []
        for vstem, frames in vids.items():
            for f in frames:
                pool.append((f, vstem))
        rng.shuffle(pool)
        for f, vstem in pool[:cap_per_stage]:
            items.append((f, stage, vstem))
    return items


def split_by_video(items, test_frac, seed):
    """Assign whole videos (per stage) to train/test so no frame leaks & all stages present."""
    rng = random.Random(seed)
    by_stage_vids = defaultdict(set)
    for _, stage, v in items:
        by_stage_vids[stage].add(v)
    test_vids = set()
    for stage, vids in by_stage_vids.items():
        vids = sorted(vids)
        rng.shuffle(vids)
        k = max(1, int(round(len(vids) * test_frac)))
        test_vids.update(vids[:k])
    train = [(f, s) for (f, s, v) in items if v not in test_vids]
    test = [(f, s) for (f, s, v) in items if v in test_vids]
    return train, test


@torch.no_grad()
def extract_dinov3(weights, samples, batch, device):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(weights).eval().to(device)
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    feats, labels = [], []
    buf, lab = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x).pooler_output
        feats.append(out.float().cpu().numpy())
        labels.extend(lab)
        buf.clear(); lab.clear()

    for i, (fp, stage) in enumerate(samples):
        buf.append(tf(Image.open(fp).convert("RGB")))
        lab.append(stage)
        if len(buf) >= batch:
            flush()
    flush()
    return np.concatenate(feats), np.array(labels)


def run_probes(Xtr, ytr, Xte, yte):
    # L2-normalize for kNN (cosine-like)
    def norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    knn = KNeighborsClassifier(n_neighbors=20, metric="cosine")
    knn.fit(norm(Xtr), ytr)
    knn_pred = knn.predict(norm(Xte))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    lin_pred = clf.predict(Xte)
    return {
        "knn_acc": accuracy_score(yte, knn_pred),
        "knn_bal_acc": balanced_accuracy_score(yte, knn_pred),
        "linear_acc": accuracy_score(yte, lin_pred),
        "linear_bal_acc": balanced_accuracy_score(yte, lin_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov3"], default="dinov3")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cap-per-stage", type=int, default=250)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")

    items = gather_frames(args.cap_per_stage, args.seed)
    train, test = split_by_video(items, args.test_frac, args.seed)
    stages = sorted({s for _, s, _ in items})
    print(f"backbone={args.backbone} weights={args.weights}")
    print(f"stages({len(stages)}): {stages}")
    print(f"frames: {len(items)}  probe-train={len(train)}  probe-test={len(test)}")

    Xtr, ytr = extract_dinov3(args.weights, train, args.batch, device)
    Xte, yte = extract_dinov3(args.weights, test, args.batch, device)
    print(f"features: train {Xtr.shape}  test {Xte.shape}")

    res = run_probes(Xtr, ytr, Xte, yte)
    print("=== results ===")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
