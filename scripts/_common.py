"""Shared helpers for the data-prep scripts (10_build_manifest / 11_extract_frames /
12_motion_energy / 13_frame_weights). Pure stdlib so every script can import it.

Scripts are run as `python scripts/NN_*.py`, which puts scripts/ on sys.path, so a plain
`from _common import ...` works (also in multiprocessing "spawn" children on Windows).
"""
import os
import re
from pathlib import Path

# Project root; override with DT_ROOT (e.g. on a GCP VM). Default is the WSL path.
DT_ROOT = os.environ.get("DT_ROOT", "/mnt/d/Video/domain_transfer")

# Suffix of the motion-energy sidecar written next to each kept frame by 12_motion_energy.py
# and read by 13_frame_weights.py and repos/dinov3 (augmentations.py / crop_sampler.py).
SIDECAR_SUFFIX = ".me.png"


def parse_stage(name: str) -> str:
    """'20251120_163400_MI_Sperm_selImmo.avi' -> 'Sperm_selImmo' (the 12 operation-stage labels)."""
    stem = Path(name).stem
    stage = stem.split("_MI_", 1)[1] if "_MI_" in stem else stem
    return re.sub(r"[_\d]+$", "", stage) or "UNKNOWN"


def to_native_path(p: str) -> str:
    """Manifests store WSL paths (/mnt/d/...). On native Windows translate them to D:/...;
    a no-op everywhere else."""
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)", p)
        if m:
            return f"{m.group(1).upper()}:/{m.group(2)}"
    return p


def read_manifest(path: str, limit: int = 0) -> list:
    """One video path per line; blank lines ignored. limit>0 keeps only the first N."""
    videos = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return videos[:limit] if limit else videos


def sidecar_path(frame_path) -> str:
    return str(frame_path) + SIDECAR_SUFFIX
