"""Humanoid Everyday dataset (USC-GVL, LeRobot v2.0 layout) for LFM4VLA fine-tuning.

Map-style dataset over the 4 core manipulation skills (pick/place, push/pull,
stack/align, pour) of https://humanoideveryday.github.io — Unitree G1 episodes only
by default (consistent 28-D action space: 14 arm joints ``sol_q`` + 14 Dex3 hand
joints).

Expected on-disk layout (created by ``scripts/download_humanoid_everyday.py``):

    <data_root_dir>/
      meta/info.json
      meta/tasks.jsonl               # task_index -> "Category/task_name"
      meta/episodes.jsonl            # episode_index -> tasks, length, robot_type, instruction
      data/chunk-XXX/episode_XXXXXX.parquet    # slim parquet (action + indices)
      videos/chunk-XXX/egocentric/episode_XXXXXX.mp4

Actions are absolute joint targets; every dim is normalized to [norm_min, norm_max]
with q01/q99 bounds computed over the selected episodes (same idea as the RLDS
BOUNDS_Q99 normalization used for LIBERO). Stats are cached under ``meta/`` so eval /
deployment can denormalize. Use with ``act_head.type = "FCContinuousDecoder"`` and
``act_head.action_dim = 28`` (no binary gripper).

Videos are decoded lazily per frame with PyAV (``pip install av``); parquet is read
with pyarrow. Map-style => normal multi-worker DataLoader works (no TF involved).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Task-name keywords defining each skill bucket (matched against the lowercase
# "Category/task_name" string from meta/tasks.jsonl).
SKILL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "pick_place": ("pick", "place"),
    "push_pull": ("push", "pull"),
    "stack_align": ("stack", "align"),
    "pour": ("pour",),
}

DEFAULT_SKILLS = ("pick_place", "push_pull", "stack_align", "pour")


def task_matches_skills(task_name: str, skills: Sequence[str]) -> bool:
    t = task_name.lower()
    return any(kw in t for s in skills for kw in SKILL_KEYWORDS[s])


def load_meta(data_root_dir: str) -> Tuple[Dict[int, dict], List[dict]]:
    meta = Path(data_root_dir) / "meta"
    tasks = {}
    with open(meta / "tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_index"]] = t
    episodes = []
    with open(meta / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return tasks, episodes


class HumanoidEverydayDataset(Dataset):
    """LFM4VLA-compatible dataset over Humanoid Everyday episodes."""

    def __init__(
        self,
        image_fn: Callable[[List[Image.Image]], torch.Tensor],
        tokenizer: Any = None,  # unused; trainer tokenizes raw strings
        data_root_dir: str = "",
        skills: Sequence[str] = DEFAULT_SKILLS,
        robot_type: str = "g1",
        window_size: int = 1,
        fwd_pred_next_n: int = 10,
        train: bool = True,
        val_every_n_episodes: int = 20,
        norm_action: bool = False,  # accepted for API parity; q01/q99 norm always applied
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        data_source: str = "humanoid_everyday_action",
        max_open_videos: int = 8,
        **kwargs: Any,
    ) -> None:
        del tokenizer, norm_action, kwargs
        assert window_size == 1, "HumanoidEverydayDataset currently supports window_size=1"
        unknown = set(skills) - set(SKILL_KEYWORDS)
        if unknown:
            raise ValueError(f"Unknown skills {unknown}; available: {list(SKILL_KEYWORDS)}")

        self.image_fn = image_fn
        self.root = Path(data_root_dir)
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.chunk_len = window_size + fwd_pred_next_n - 1
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.data_source = data_source
        self.train = train

        tasks, episodes = load_meta(str(self.root))

        # Collect every skill-matching episode that is present on disk, then split
        # train/val. Action q01/q99 must be computed over *all* of them (not just
        # the current split) so train and val share the same denorm bounds.
        eligible: List[dict] = []
        missing = 0
        for ep in episodes:
            task = tasks[ep["tasks"][0]]
            if robot_type and ep.get("robot_type") != robot_type:
                continue
            if not task_matches_skills(task["task"], skills):
                continue
            if not (self._parquet_path(ep["episode_index"]).is_file()
                    and self._video_path(ep["episode_index"]).is_file()):
                missing += 1
                continue
            ep = dict(ep)
            ep["task_name"] = task["task"]
            ep["instruction"] = (
                ep.get("instruction")
                or task.get("description")
                or task["task"].split("/")[-1].replace("_", " ")
            )
            eligible.append(ep)

        if missing:
            print(f"[HumanoidEveryday] skipped {missing} episodes with missing local files "
                  f"(run scripts/download_humanoid_everyday.py to fetch them)")
        if not eligible:
            raise FileNotFoundError(
                f"No local episodes found under {self.root} for skills={list(skills)} "
                f"robot={robot_type}. Run scripts/download_humanoid_everyday.py."
            )

        self.q01, self.q99, self.stats_path = self._load_or_compute_stats(
            sorted(ep["episode_index"] for ep in eligible), robot_type, tuple(skills)
        )

        selected = [
            ep for ep in eligible
            if ((ep["episode_index"] % val_every_n_episodes) == 0) == (not train)
        ]
        if not selected:
            raise FileNotFoundError(
                f"No {'val' if not train else 'train'} episodes under {self.root} "
                f"for skills={list(skills)} robot={robot_type} "
                f"(val_every_n_episodes={val_every_n_episodes})."
            )

        self.episodes = {ep["episode_index"]: ep for ep in selected}
        # Flat sample index: one sample per (episode, timestep).
        self.samples: List[Tuple[int, int]] = [
            (ep["episode_index"], t) for ep in selected for t in range(ep["length"])
        ]

        self._action_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._video_cache: "OrderedDict[int, Any]" = OrderedDict()
        self._max_open_videos = max_open_videos

        n_frames = len(self.samples)
        print(f"[HumanoidEveryday] {'train' if train else 'val'}: "
              f"{len(selected)} episodes / {n_frames} frames | skills={list(skills)} "
              f"robot={robot_type} action_dim={len(self.q01)} | stats={self.stats_path}")

    # ------------------------------------------------------------------
    # Paths (LeRobot v2.0 layout; chunks of 1000 episodes)
    # ------------------------------------------------------------------
    def _parquet_path(self, ep_idx: int) -> Path:
        return self.root / f"data/chunk-{ep_idx // 1000:03d}/episode_{ep_idx:06d}.parquet"

    def _video_path(self, ep_idx: int) -> Path:
        return self.root / f"videos/chunk-{ep_idx // 1000:03d}/egocentric/episode_{ep_idx:06d}.mp4"

    # ------------------------------------------------------------------
    # Action stats (q01/q99 per dim, cached under meta/)
    # ------------------------------------------------------------------
    def _load_or_compute_stats(self, ep_indices, robot_type, skills):
        key = hashlib.md5(
            json.dumps({"eps": ep_indices, "robot": robot_type, "skills": skills}).encode()
        ).hexdigest()[:10]
        stats_path = self.root / "meta" / f"action_stats_{robot_type}_{key}.json"
        if stats_path.is_file():
            with open(stats_path) as f:
                stats = json.load(f)
            return (np.asarray(stats["q01"], dtype=np.float32),
                    np.asarray(stats["q99"], dtype=np.float32), str(stats_path))

        print(f"[HumanoidEveryday] computing q01/q99 action stats over {len(ep_indices)} episodes...")
        all_actions = np.concatenate(
            [self._read_actions(ep) for ep in ep_indices], axis=0
        )
        q01 = np.quantile(all_actions, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(all_actions, 0.99, axis=0).astype(np.float32)
        stats = {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "num_frames": int(all_actions.shape[0]),
            "num_episodes": len(ep_indices),
            "robot_type": robot_type,
            "skills": list(skills),
        }
        tmp = stats_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, stats_path)
        return q01, q99, str(stats_path)

    def _read_actions(self, ep_idx: int) -> np.ndarray:
        import pyarrow.parquet as pq

        table = pq.read_table(self._parquet_path(ep_idx), columns=["action"])
        return np.stack(table.column("action").to_pylist()).astype(np.float32)

    def _episode_actions(self, ep_idx: int) -> np.ndarray:
        if ep_idx in self._action_cache:
            self._action_cache.move_to_end(ep_idx)
            return self._action_cache[ep_idx]
        actions = self._read_actions(ep_idx)
        self._action_cache[ep_idx] = actions
        if len(self._action_cache) > 64:
            self._action_cache.popitem(last=False)
        return actions

    def _normalize(self, action: np.ndarray) -> np.ndarray:
        span = np.maximum(self.q99 - self.q01, 1e-8)
        unit = (action - self.q01) / span  # -> [0, 1] within bounds
        out = unit * (self.norm_max - self.norm_min) + self.norm_min
        return np.clip(out, self.norm_min, self.norm_max)

    # ------------------------------------------------------------------
    # Video frame access (PyAV, per-worker container cache)
    # ------------------------------------------------------------------
    def _get_container(self, ep_idx: int):
        import av

        if ep_idx in self._video_cache:
            self._video_cache.move_to_end(ep_idx)
            return self._video_cache[ep_idx]
        container = av.open(str(self._video_path(ep_idx)))
        self._video_cache[ep_idx] = container
        if len(self._video_cache) > self._max_open_videos:
            _, old = self._video_cache.popitem(last=False)
            old.close()
        return container

    def _read_frame(self, ep_idx: int, t: int) -> np.ndarray:
        container = self._get_container(ep_idx)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        target_sec = t / fps
        # Seek to the keyframe at/before the target, then decode forward.
        container.seek(int(target_sec / stream.time_base), stream=stream, backward=True)
        last = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            last = frame
            if float(frame.pts * stream.time_base) >= target_sec - 0.5 / fps:
                break
        if last is None:
            raise RuntimeError(f"Failed to decode frame {t} of episode {ep_idx}")
        return last.to_ndarray(format="rgb24")

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ep_idx, t = self.samples[index]
        ep = self.episodes[ep_idx]

        actions = self._episode_actions(ep_idx)
        end = min(t + self.chunk_len, actions.shape[0])
        chunk = actions[t:end]
        n_valid = chunk.shape[0]
        if n_valid < self.chunk_len:
            pad = np.repeat(chunk[-1:], self.chunk_len - n_valid, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        mask = np.zeros(self.chunk_len, dtype=np.float32)
        mask[:n_valid] = 1.0

        action = self._normalize(chunk.astype(np.float32))

        frame = self._read_frame(ep_idx, t)
        rgb = self.image_fn([Image.fromarray(frame)])  # [window_size=1, C, H, W]

        return {
            "rgb": rgb,
            "action": action,          # [chunk_len, action_dim], normalized
            "action_mask": mask,       # [chunk_len]
            "lang": ep["instruction"],
        }

    def collater(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_tensors = torch.stack([s["rgb"] for s in sample])[:, : self.window_size]
        action_tensors = torch.from_numpy(np.stack([s["action"] for s in sample])).float()
        action_mask = torch.from_numpy(np.stack([s["action_mask"] for s in sample])).float()
        stacked_language = [s["lang"] for s in sample]

        # [B, window_size, fwd_pred_next_n, action_dim] / [B, window_size, fwd_pred_next_n]
        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        chunck_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)

        return {
            "rgb": image_tensors,
            "hand_rgb": None,
            "action": action_tensors,
            "text": stacked_language,
            "text_mask": None,
            "action_chunck": action_chunck,
            "chunck_mask": chunck_mask,
            "raw_text": stacked_language,
            "data_source": self.data_source,
        }
