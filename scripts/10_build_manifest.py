#!/usr/bin/env python3
"""Scan all videos, parse stage labels from filenames, split train/val by date.

Outputs (under domain_transfer/manifests/):
  all_videos.csv        path,date,session,stage
  train_videos.txt      video paths for SSL training (DINOv3 frames + V-JEPA2 clips)
  val_videos.txt        held-out video paths (for feature-quality eval)
  label_stats.txt       stage distribution + split summary

Split rule: hold out the LAST `--val-dates` distinct dates as validation, so no
session/clip leaks across the split. Pure stdlib; no torch needed.
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from _common import DT_ROOT, parse_stage

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def parse_date(path: Path) -> str:
    m = re.match(r"(\d{8})_", path.name)
    if m:
        return m.group(1)
    for part in path.parts:  # fall back to an 8-digit dir in the path
        if re.fullmatch(r"\d{8}", part):
            return part
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/Video")
    ap.add_argument("--out", default=f"{DT_ROOT}/manifests")
    ap.add_argument("--val-dates", type=int, default=2,
                    help="number of most-recent dates held out for validation")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            date = parse_date(p)
            # session = the immediate grandparent-ish grouping (date/<session>/videos)
            session = date
            for i, part in enumerate(p.parts):
                if re.fullmatch(r"\d{8}", part) and i + 1 < len(p.parts):
                    session = f"{part}/{p.parts[i+1]}"
                    break
            rows.append({
                "path": str(p),
                "date": date,
                "session": session,
                "stage": parse_stage(p.name),
            })

    rows.sort(key=lambda r: r["path"])
    dates = sorted({r["date"] for r in rows if r["date"] != "unknown"})
    val_dates = set(dates[-args.val_dates:]) if args.val_dates > 0 else set()

    with (out / "all_videos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "date", "session", "stage"])
        w.writeheader()
        w.writerows(rows)

    train = [r for r in rows if r["date"] not in val_dates]
    val = [r for r in rows if r["date"] in val_dates]
    (out / "train_videos.txt").write_text("\n".join(r["path"] for r in train) + "\n")
    (out / "val_videos.txt").write_text("\n".join(r["path"] for r in val) + "\n")

    stage_counts = Counter(r["stage"] for r in rows)
    per_date = defaultdict(int)
    for r in rows:
        per_date[r["date"]] += 1

    lines = []
    lines.append(f"total_videos: {len(rows)}")
    lines.append(f"num_dates: {len(dates)}")
    lines.append(f"train_videos: {len(train)}  val_videos: {len(val)}")
    lines.append(f"val_dates (held out): {sorted(val_dates)}")
    lines.append(f"num_stages: {len(stage_counts)}")
    lines.append("")
    lines.append("=== stage distribution ===")
    for s, c in stage_counts.most_common():
        lines.append(f"{c:6d}  {s}")
    lines.append("")
    lines.append("=== videos per date ===")
    for d in dates:
        tag = "  [VAL]" if d in val_dates else ""
        lines.append(f"{per_date[d]:6d}  {d}{tag}")
    report = "\n".join(lines)
    (out / "label_stats.txt").write_text(report + "\n")
    print(report)
    print("\nWrote manifests to:", out)


if __name__ == "__main__":
    main()
