"""RobotNav Deliverable-1 mixture dataset for LFM4VLA (85/15 co-training).

Streams the frozen Qwen-RobotNav replication release (trajectory shards +
general-VL conversations) and yields **batch-homogeneous** draws: each batch
is entirely trajectory (waypoint regression) or entirely VL (next-token
prediction), sampled 85/15 — the paper's batch-level registry sampling
(per-dataset weights configurable, mirroring configs/dataset_registry.yaml).

Trajectory batches reproduce the exact LIBERO/CALVIN collater contract so the
existing trainer/model consume them without modification:
    rgb [B, ws, C, H, W] · text list[str] · action_chunck [B, ws, K, 3]
    chunck_mask [B, ws, K] (real chunk only at the LAST window slot; padded
    waypoints masked via terminal_mask) · data_source "robotnav_traj"

VL batches carry raw PIL images + user/answer strings under
data_source "robotnav_vl"; train.robotnav_trainer.RobotNavTrainer routes them
through the backbone LM with labels (loss_vl_cotrain hook, ratio = config
``vl_cotrain_ratio`` — set 1.0 for the paper's λ).

Observation config follows the mentor-simplified fixed recipe: front camera
only, up to ws-1 uniformly-spaced history frames + current frame, instruction
text as-is (paraphrase-variant rows resolve text via instruction_variants).

Config keys (train_dataset block):
    type: "RobotNavMixtureDataset"
    release_dir:   .../robotnav_release/deliverable1
    generated_root:.../datasets/robotnav_generated
    mixture_trajectory: 0.85
    family_weights: {vln_rxr: 4.14, vln_r2r: 1.491, objectnav_hm3d: 1.448,
                     objectnav_mp3d: 0.552, pointnav_hm3d: 0.492,
                     pointnav_mp3d: 0.492}
    batch_size: (must equal the DataLoader batch size)
    max_samples: optional cap (val slices)
    seed: 20260617
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

K_WAYPOINTS = 8
ACTION_DIM = 3  # (x, y, yaw)


def _load_scale_factors(release_dir: Path) -> Dict[str, Dict[str, float]]:
    txt = (release_dir / "scale_factors.yaml").read_text()
    # file is a comment line + JSON body (written by the assembly script)
    body = txt[txt.index("{"):]
    return json.loads(body)


class _ShardStream:
    """Endless seeded shuffle-stream over one family's JSONL shards."""

    def __init__(self, shard_paths: List[str], rng: random.Random):
        assert shard_paths, "no shards found"
        self.paths = list(shard_paths)
        self.rng = rng
        self._rows: List[str] = []

    def _refill(self) -> None:
        path = self.rng.choice(self.paths)
        with open(path) as f:
            self._rows = f.readlines()
        self.rng.shuffle(self._rows)

    def next(self) -> Dict[str, Any]:
        while not self._rows:
            self._refill()
        return json.loads(self._rows.pop())


class RobotNavMixtureDataset(IterableDataset):
    def __init__(
        self,
        release_dir: str,
        generated_root: str,
        image_fn: Callable,
        tokenizer=None,                    # unused; kept for build_dataset common kwargs
        window_size: int = 9,
        fwd_pred_next_n: int = K_WAYPOINTS,
        mixture_trajectory: float = 0.85,
        mixture_mode: str = "batch",
        obs_randomization: Optional[Dict[str, Any]] = None,
        family_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 16,
        max_samples: int = 0,
        seed: int = 20260617,
        norm_action: bool = True,          # accepted from common kwargs
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        **_ignored,
    ):
        super().__init__()
        assert fwd_pred_next_n == K_WAYPOINTS, "release targets are 8-waypoint chunks"
        self.release = Path(release_dir)
        self.gen_root = Path(generated_root)
        self.image_fn = image_fn
        self.ws = int(window_size)
        self.mix_traj = float(mixture_trajectory)
        # "batch": paper-style batch-level sampling (each batch homogeneous —
        # one trajectory family or VL). "sample": per-sample Bernoulli mixing,
        # so a single batch blends ~85% trajectory and ~15% VL samples.
        self.mixture_mode = str(mixture_mode)
        assert self.mixture_mode in ("batch", "sample")
        # Qwen-RobotNav per-sample observation randomization, single-view
        # variant (camera-view + per-camera-weight axes intentionally absent).
        # Axes: total visual token budget B ~ U[budget_min, budget_max];
        # temporal decay gamma ~ U[gamma_min, gamma_max] (older frames get
        # geometrically fewer tokens); history-selection mode 50/50
        # random-global vs latest-window. Budgets are realized by resolution
        # scaling (tokens ~ pixels for the LFM processor; frames are never
        # upscaled past native, so the current frame stays sharpest).
        self.obs_rand = None
        if obs_randomization and obs_randomization.get("enabled", True):
            self.obs_rand = {
                "budget_min": float(obs_randomization.get("budget_min", 2048)),
                "budget_max": float(obs_randomization.get("budget_max", 4096)),
                "gamma_min": float(obs_randomization.get("gamma_min", 1.0)),
                "gamma_max": float(obs_randomization.get("gamma_max", 3.0)),
                "frame_tokens_min": float(obs_randomization.get("frame_tokens_min", 64)),
                "frame_tokens_max": float(obs_randomization.get("frame_tokens_max", 256)),
                "native_tokens": float(obs_randomization.get("native_tokens", 256)),
                "p_latest": float(obs_randomization.get("p_latest", 0.5)),
            }
        self.batch_size = int(batch_size)
        self.max_samples = int(max_samples)
        # Stream position is not checkpointed: without a salt, every resumed
        # attempt would replay the identical sample sequence from the seed.
        # Mixing the Slurm job id + restart count gives each attempt a fresh
        # draw order (set data_seed_salt: "none" for strict reproducibility
        # of uninterrupted runs).
        salt = 0
        rank_salt = 0
        if str(_ignored.get("data_seed_salt", "auto")) == "auto":
            import os
            # Distributed: each rank MUST draw a distinct stream, else N-rank
            # data parallelism silently trains N copies of the same batches.
            rank = int(os.environ.get("SLURM_PROCID",
                                      os.environ.get("RANK", "0")) or 0)
            rank_salt = rank * 7919
            salt = (int(os.environ.get("SLURM_JOB_ID", "0") or 0) * 1009
                    + int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
                    + rank_salt)
        self.rng = random.Random(seed + salt)
        # Batch-mode multi-rank safety: the traj-vs-VL TYPE decision must be
        # identical on every rank at every step (FSDP/DDP collectives hang if
        # ranks run different module branches), so it draws from a stream
        # salted with job+restart only. Content draws stay rank-salted.
        self.type_rng = random.Random(seed + salt - rank_salt)
        self.data_source = "robotnav_traj"

        weights = family_weights or {
            "vln_rxr": 4.14, "vln_r2r": 1.491, "objectnav_hm3d": 1.448,
            "objectnav_mp3d": 0.552, "pointnav_hm3d": 0.492, "pointnav_mp3d": 0.492}
        self.families: Dict[str, _ShardStream] = {}
        self.family_w: List[float] = []
        self.family_names: List[str] = []
        for fam, w in weights.items():
            shards = sorted(glob.glob(str(self.release / fam / "shard_*.jsonl")))
            if shards:
                self.families[fam] = _ShardStream(shards, random.Random(seed + hash(fam) % 9973))
                self.family_names.append(fam)
                self.family_w.append(float(w))
        assert self.families, f"no trajectory shards under {self.release}"

        vl_shards = sorted(glob.glob(str(self.release / "general_vl" / "*.jsonl")))
        self.vl_stream = _ShardStream(vl_shards, random.Random(seed + 77)) if vl_shards else None

        self.scale = _load_scale_factors(self.release)
        self._variants: Optional[Dict[str, List[str]]] = None
        self._instr_cache: Dict[str, str] = {}

    # ------------------------------------------------------------- helpers --
    def _variants_map(self) -> Dict[str, List[str]]:
        if self._variants is None:
            self._variants = {}
            vp = self.release / "instruction_variants.jsonl"
            if vp.exists():
                for line in open(vp):
                    r = json.loads(line)
                    for eid in r.get("episode_ids", []):
                        self._variants[f"{r['source']}::{eid}"] = r.get("variants", [])
        return self._variants

    def _episode_dir(self, family: str, episode_id: str) -> Path:
        # episode_id: <source>/train/<scene>/<ep_id>
        _, _, scene, ep = episode_id.split("/")
        return self.gen_root / family / "episodes" / scene / ep

    def _instruction(self, family: str, row: Dict[str, Any]) -> str:
        eid = row["episode_id"]
        variant_id = int(row.get("instruction_variant_id", 0))
        if variant_id > 0:
            src_ep = str(row.get("provenance", {}).get("src_ep", ""))
            src = eid.split("/")[0]
            vs = self._variants_map().get(f"{src}::{src_ep}", [])
            if vs:
                return vs[(variant_id - 1) % len(vs)]
        if eid not in self._instr_cache:
            ep = json.loads((self._episode_dir(family, eid) / "episode.json").read_text())
            instr = ep.get("instruction")
            if not instr:                       # pointnav coordinate goals
                g = (ep.get("goal") or {}).get("value") or {}
                if isinstance(g, dict) and "ego_xy" in g:
                    x, y = g["ego_xy"]
                    instr = f"Navigate to the point ({x:.2f}, {y:.2f}) in your frame."
                else:
                    instr = str(g) if g else "Navigate to the goal."
            if len(self._instr_cache) > 200_000:
                self._instr_cache.clear()
            self._instr_cache[eid] = instr
        return self._instr_cache[eid]

    def _randomized_frames(self, ep_dir, hist: List[int], t: int):
        """Single-view per-sample observation randomization (see __init__)."""
        r = self.obs_rand
        B = self.rng.uniform(r["budget_min"], r["budget_max"])
        gamma = self.rng.uniform(r["gamma_min"], r["gamma_max"])
        past = [int(i) for i in hist[:-1]] if len(hist) > 1 else []
        n_hist = min(self.ws - 1, len(past))
        if n_hist > 0:
            if self.rng.random() < r["p_latest"]:            # recency window
                sel = past[-n_hist:]
            else:                                            # random / global
                sel = sorted(self.rng.sample(past, n_hist))
        else:
            sel = []
        frame_ids = sel + [t]
        while len(frame_ids) < self.ws:
            frame_ids.insert(0, frame_ids[0])
        # per-frame token budgets: current frame age 0, decaying into the past
        ages = list(range(len(frame_ids) - 1, -1, -1))
        wts = [gamma ** (-a) for a in ages]
        s = sum(wts)
        budgets = [min(r["frame_tokens_max"],
                       max(r["frame_tokens_min"], B * w / s)) for w in wts]
        frames = []
        for fi, b in zip(frame_ids, budgets):
            im = Image.open(ep_dir / f"{fi:03d}_front.jpg").convert("RGB")
            scale = min(1.0, (b / r["native_tokens"]) ** 0.5)  # tokens ~ pixels
            if scale < 0.995:
                im = im.resize((max(64, int(im.width * scale)) // 2 * 2,
                                max(64, int(im.height * scale)) // 2 * 2),
                               Image.BILINEAR)
            frames.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous())
        return frames                                        # list of [C, H, W]

    def _traj_sample(self, family: str, row: Dict[str, Any]) -> Dict[str, Any]:
        eid = row["episode_id"]
        ep_dir = self._episode_dir(family, eid)
        t = int(row["t"])
        hist = row.get("available_history") or list(range(t + 1))
        if self.obs_rand is not None:
            rgb = self._randomized_frames(ep_dir, hist, t)   # list of [C,H,W]
        else:
            n_hist = min(self.ws - 1, len(hist) - 1) if len(hist) > 1 else 0
            if n_hist > 0:
                idxs = sorted({int(i) for i in np.linspace(hist[0], hist[-2], n_hist)})
            else:
                idxs = []
            frame_ids = idxs + [t]
            while len(frame_ids) < self.ws:      # left-pad by repeating earliest
                frame_ids.insert(0, frame_ids[0])
            pils = [Image.open(ep_dir / f"{fi:03d}_front.jpg").convert("RGB")
                    for fi in frame_ids]
            rgb = self.image_fn(pils)
            if not torch.is_tensor(rgb):
                rgb = torch.stack(rgb)
        sf = self.scale[family]
        w = np.asarray(row["future_waypoints_robot"], dtype=np.float32)
        w[:, 0] = np.clip(w[:, 0] / max(sf["x"], 1e-6), -1, 1)
        w[:, 1] = np.clip(w[:, 1] / max(sf["y"], 1e-6), -1, 1)
        w[:, 2] = np.clip(w[:, 2] / max(sf["yaw"], 1e-6), -1, 1)
        return {
            "sample_type": "traj",
            "rgb": rgb,                                   # [ws, C, H, W]
            "lang": self._instruction(family, row),
            "chunk": torch.from_numpy(w),                 # [K, 3] normalized
            "chunk_mask": torch.tensor(row["terminal_mask"], dtype=torch.float32),
            "family": family,
        }

    def _vl_sample(self) -> Dict[str, Any]:
        row = self.vl_stream.next()
        conv = row["conversation"]
        user_text = " ".join(c["text"] for c in conv[0]["content"] if c.get("type") == "text")
        answer = " ".join(c["text"] for c in conv[-1]["content"] if c.get("type") == "text")
        images = [Image.open(p).convert("RGB") for p in row.get("images", [])[:2]]
        return {"sample_type": "vl", "images": images,
                "user_text": user_text, "answer_text": answer,
                "category": row.get("category", "vl")}

    # ------------------------------------------------------------- iterate --
    def _one_traj_sample(self) -> Dict[str, Any]:
        fam = self.rng.choices(self.family_names, weights=self.family_w)[0]
        stream = self.families[fam]
        for _attempt in range(5):
            row = stream.next()
            try:
                return self._traj_sample(fam, row)
            except FileNotFoundError:
                continue
        return self._traj_sample(fam, stream.next())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yielded = 0
        if self.mixture_mode == "sample":
            # Per-sample Bernoulli mixing: each batch blends trajectory + VL.
            # Emitted in batch_size blocks with >=1 trajectory sample forced
            # per block: multi-rank strategies (FSDP reduce-scatter, sync_dist
            # logging) hang if ranks disagree on which modules ran, and an
            # all-VL block would skip the action head entirely. Distortion is
            # negligible (traj is the 85% majority). The all-traj case is
            # handled trainer-side with a zero-weighted dummy VL forward.
            while True:
                block = []
                for _ in range(self.batch_size):
                    if self.vl_stream is not None and self.rng.random() >= self.mix_traj:
                        block.append(self._vl_sample())
                    else:
                        block.append(self._one_traj_sample())
                if all(s["sample_type"] == "vl" for s in block):
                    block[self.rng.randrange(len(block))] = self._one_traj_sample()
                for s in block:
                    yield s
                    yielded += 1
                    if self.max_samples and yielded >= self.max_samples:
                        return
        while True:
            if self.vl_stream is not None and self.type_rng.random() >= self.mix_traj:
                for _ in range(self.batch_size):          # homogeneous VL batch
                    yield self._vl_sample()
                    yielded += 1
            else:
                fam = self.rng.choices(self.family_names, weights=self.family_w)[0]
                stream = self.families[fam]
                for _ in range(self.batch_size):          # homogeneous traj batch
                    for _attempt in range(5):
                        row = stream.next()
                        try:
                            s = self._traj_sample(fam, row)
                            break
                        except FileNotFoundError:
                            continue                       # tolerate missing frame
                    yield s
                    yielded += 1
            if self.max_samples and yielded >= self.max_samples:
                return

    # -------------------------------------------------------------- collate --
    def collater(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.mixture_mode == "sample":
            traj = [s for s in samples if s["sample_type"] != "vl"]
            vl = [s for s in samples if s["sample_type"] == "vl"]
            return {
                "data_source": "robotnav_mixed",
                "traj": self._collate_traj(traj) if traj else None,
                "vl": self._collate_vl(vl) if vl else None,
            }
        if samples[0]["sample_type"] == "vl":
            return self._collate_vl(samples)
        return self._collate_traj(samples)

    def _collate_vl(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "data_source": "robotnav_vl",
            "vl_images": [s["images"] for s in samples],
            "vl_user": [s["user_text"] for s in samples],
            "vl_answer": [s["answer_text"] for s in samples],
            "raw_text": [s["user_text"] for s in samples],
        }

    def _collate_traj(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Frames keep their native per-episode resolutions (640 x U[320,480]),
        # so rgb is a LIST of [ws, C, H, W] tensors — never stacked. The
        # trainer converts frames to PIL individually and the LFM processor
        # handles arbitrary sizes; vision_x is only used for (bs, ws) shape.
        rgb = [s["rgb"] for s in samples]
        b = len(rgb)
        chunks = torch.stack([s["chunk"] for s in samples])       # [B, K, 3]
        masks = torch.stack([s["chunk_mask"] for s in samples])   # [B, K]
        action_chunck = torch.zeros(b, self.ws, K_WAYPOINTS, ACTION_DIM)
        chunck_mask = torch.zeros(b, self.ws, K_WAYPOINTS)
        action_chunck[:, -1] = chunks
        chunck_mask[:, -1] = masks
        texts = [s["lang"] for s in samples]
        # Per-sample denormalization vectors (frozen 99th-pct scale factors) so
        # the trainer can compute task-space metrics (ADE/FDE meters, yaw deg).
        wp_scale = torch.tensor(
            [[self.scale[s["family"]]["x"], self.scale[s["family"]]["y"],
              self.scale[s["family"]]["yaw"]] for s in samples],
            dtype=torch.float32)
        return {
            "rgb": rgb,
            "family": [s["family"] for s in samples],
            "wp_scale": wp_scale,
            "hand_rgb": None,
            "action": torch.zeros(b, self.ws, ACTION_DIM),
            "text": texts,
            "text_mask": None,
            "action_chunck": action_chunck,
            "chunck_mask": chunck_mask,
            "raw_text": texts,
            "data_source": "robotnav_traj",
        }
