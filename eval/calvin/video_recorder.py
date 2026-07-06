"""Incremental rollout frame recorder — survives segfaults via per-step PNG writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from eval.calvin.video_io import save_rollout_video


class FrameRecorder:
    """Write each sim frame to disk immediately; assemble MP4 when rollout ends."""

    def __init__(self, out_dir: Path, stem: str, fps: int = 10):
        self.frame_dir = out_dir / f"{stem}_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_stem = out_dir / stem
        self.fps = fps
        self.count = 0

    def add(self, frame: np.ndarray) -> None:
        path = self.frame_dir / f"frame_{self.count:05d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        self.count += 1

    def finalize(self, tag: str) -> Path | None:
        if self.count == 0:
            print(f"  [warn] no frames in {self.frame_dir}")
            return None

        frames = [
            np.array(Image.open(path).convert("RGB"))
            for path in sorted(self.frame_dir.glob("frame_*.png"))
        ]
        return save_rollout_video(frames, self.video_stem.parent / f"{self.video_stem.name}-{tag}", self.fps)
