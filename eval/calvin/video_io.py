"""Write rollout frame sequences to MP4/GIF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_rollout_video(frames, out_path: Path, fps: int = 10) -> Path | None:
    if not frames:
        print("  [warn] no frames captured for video")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected frames [T,H,W,3], got {arr.shape}")

    mp4_path = out_path.with_suffix(".mp4")
    errors = []
    for writer_name, writer_fn in (
        ("imageio", _write_mp4_imageio),  # H.264, most portable (needs imageio-ffmpeg)
        ("opencv", _write_mp4_cv2),       # mp4v fallback, no extra deps
    ):
        try:
            writer_fn(arr, mp4_path, fps)
            print(f"  saved rollout video: {mp4_path} ({len(arr)} frames @ {fps} fps via {writer_name})")
            return mp4_path
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{writer_name}: {exc}")

    # Both MP4 writers failed — now it's worth complaining loudly.
    for err in errors:
        print(f"  [warn] mp4 writer unavailable ({err})")

    gif_path = out_path.with_suffix(".gif")
    pil_frames = [Image.fromarray(f) for f in arr]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(1000 / max(fps, 1))),
        loop=0,
        disposal=2,
    )
    print(f"  saved rollout gif (fallback): {gif_path} ({len(arr)} frames)")
    return gif_path


def _write_mp4_imageio(arr: np.ndarray, path: Path, fps: int) -> None:
    """Write H.264 MP4 via imageio + imageio-ffmpeg.

    imageio v3 no longer registers ``plugin="ffmpeg"`` even when imageio-ffmpeg
    is installed; the v2 ``mimwrite`` API still picks up the bundled ffmpeg exe.
    """
    import imageio

    # v2 API — works with ``pip install imageio-ffmpeg``.
    # macro_block_size=1 keeps native 200x200 (no auto-resize-to-208 warning).
    try:
        imageio.mimwrite(path, arr, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        return
    except TypeError:
        # Older imageio builds may not accept these kwargs.
        imageio.mimwrite(path, arr, fps=fps, codec="libx264")
        return
    except Exception:
        pass

    # v3 pyav backend (optional: ``pip install av``).
    import imageio.v3 as iio

    with iio.imopen(path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        for frame in arr:
            writer.write_frame(frame)


def _write_mp4_cv2(arr: np.ndarray, path: Path, fps: int) -> None:
    import cv2

    h, w = arr.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    for frame in arr:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
