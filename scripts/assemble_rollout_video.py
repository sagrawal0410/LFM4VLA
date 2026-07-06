#!/usr/bin/env python3
"""Assemble MP4 from incremental rollout PNG frames (e.g. after a segfault)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

from eval.calvin.video_io import save_rollout_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", help="e.g. runs/calvin_eval/close_drawer_lfm450m/seq000-0-close_drawer_frames")
    ap.add_argument("--output", default=None, help="Output path stem (default: frames_dir parent / rollout)")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise SystemExit(f"No frame_*.png in {frames_dir}")

    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    stem = args.output or (frames_dir.parent / frames_dir.name.replace("_frames", "-FAIL"))
    out = save_rollout_video(frames, Path(stem), fps=args.fps)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
