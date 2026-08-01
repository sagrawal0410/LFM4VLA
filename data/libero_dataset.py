"""Streaming LIBERO dataset for LFM2.5-VL training.

Wraps the OpenVLA/OXE RLDS (TensorFlow) pipeline and emits batches that match the
exact contract produced by :class:`data.calvin_dataset.DiskCalvinDataset` so the
same trainer/model consume both sources without changes.

Unlike CALVIN (map-style, disk ``.npz``), LIBERO is streamed frame-by-frame from
TFDS RLDS. Each demo is expanded into sliding-window samples: one current image +
instruction -> a chunk of ``fwd_pred_next_n`` future actions.

Depth training uses a **rotating cached shuffle window** to avoid TF's host-RAM
leak on infinite ``repeat → shuffle`` streams:

  fill cache with N frames → train for K steps → drop / rebuild with skip → …

Data: ``modified_libero_rlds`` (HuggingFace, ~10 GB), no-op actions removed.
Requires ``tensorflow==2.15``, ``tensorflow_datasets==4.9.3`` and ``dlimp`` in the env.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from data.data_utils import normalize_action

# Minimal vendored slice of OpenVLA's `prismatic` package (OXE/RLDS pipeline only),
# made importable without pip-installing the full OpenVLA repo. See data/rlds/.
_RLDS_ROOT = Path(__file__).resolve().parent / "rlds"
if _RLDS_ROOT.is_dir() and str(_RLDS_ROOT) not in sys.path:
    sys.path.append(str(_RLDS_ROOT))


class LiberoRLDSDataset(IterableDataset):
    """Iterable LIBERO dataset backed by the OpenVLA RLDS pipeline.

    The RLDS stage already normalizes arm actions to ``[-1, 1]`` via Q01/Q99 bounds
    and flips the gripper to ``{0=close, 1=open}`` (``libero_dataset_transform``), so
    ``norm_action`` should normally stay ``False`` for LIBERO.
    """

    def __init__(
        self,
        image_fn: Callable[[List[Image.Image]], torch.Tensor],
        tokenizer: Any,
        data_root_dir: str,
        data_mix: str = "libero_10_no_noops",
        window_size: int = 1,
        fwd_pred_next_n: int = 10,
        image_size: int = 224,
        shuffle_buffer_size: int = 51200,
        train: bool = True,
        image_aug: bool = False,
        norm_action: bool = False,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        data_source: str = "libero_action",
        load_depth: bool = False,
        batch_size: int = 1,
        cache_refresh_every_n_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.image_fn = image_fn
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.data_source = data_source
        self.load_depth = bool(load_depth)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.batch_size = max(int(batch_size), 1)

        # Persist build knobs so we can rebuild the TF pipeline on cache refresh.
        self._data_root_dir = str(data_root_dir)
        self._data_mix = data_mix
        self._image_aug = bool(image_aug)
        self._shuffle_buffer_size = int(shuffle_buffer_size)
        if self.load_depth:
            # Cap buffer: uncached TF shuffle leaks host RAM; cached windows must fit.
            self._shuffle_buffer_size = min(
                self._shuffle_buffer_size, 2048 if self.train else 256
            )

        # Rotating cache: train on a pinned window for K optimizer steps, then
        # skip ahead and materialize a new window. Default-on for depth train.
        if cache_refresh_every_n_steps is None and self.load_depth and self.train:
            cache_refresh_every_n_steps = 1000
        self.cache_refresh_every_n_steps = (
            int(cache_refresh_every_n_steps)
            if cache_refresh_every_n_steps is not None and int(cache_refresh_every_n_steps) > 0
            else None
        )
        self._cache_refresh_every_n_samples = (
            self.cache_refresh_every_n_steps * self.batch_size
            if self.cache_refresh_every_n_steps is not None
            else None
        )
        self._cache_window_idx = 0

        self.dataset, self.dataset_length, self.dataset_statistics = self._build_rlds(
            cache_offset=0
        )
        if self.load_depth and self.train and self._cache_refresh_every_n_samples:
            print(
                f"[libero-depth] rotating cache: N={self._shuffle_buffer_size} frames, "
                f"refresh every {self.cache_refresh_every_n_steps} steps "
                f"({self._cache_refresh_every_n_samples} samples, bs={self.batch_size})",
                flush=True,
            )

    def _build_rlds(self, cache_offset: int = 0):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        from prismatic.vla.datasets.rlds.oxe import (
            OXE_NAMED_MIXTURES,
            get_oxe_dataset_kwargs_and_weights,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        mixture_spec = OXE_NAMED_MIXTURES.get(self._data_mix, [(self._data_mix, 1.0)])
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self._data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),  # agentview only; wrist not fed to the model
            load_depth=self.load_depth,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        # Depth float32 maps make TF frame transforms + shuffle much heavier than RGB.
        parallel = 2 if self.load_depth else 16
        frame_transform_kwargs: Dict[str, Any] = dict(
            resize_size=(self.image_size, self.image_size),
            num_parallel_calls=parallel,
        )
        if self.load_depth:
            frame_transform_kwargs["depth_resize_size"] = (
                self.image_size,
                self.image_size,
            )
        if self._image_aug:
            frame_transform_kwargs["image_augment_kwargs"] = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=self.window_size,
                chunk_action=True,
                frame_num=-1,
                future_action_window_size=self.fwd_pred_next_n,
                left_pad=False,
                window_sample="sliding",
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=self._shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=self.train,
            # Pin shuffle window in RAM for depth to stop the TF shuffle leak.
            cache_shuffle_buffer=bool(self.load_depth),
            cache_offset=int(cache_offset),
        )
        return make_interleaved_dataset(**rlds_config)

    def _refresh_cache_window(self) -> None:
        """Drop the current TF pipeline and materialize the next N-frame window."""
        self._cache_window_idx += 1
        offset = self._cache_window_idx * self._shuffle_buffer_size
        print(
            f"[libero-depth] cache refresh #{self._cache_window_idx}: "
            f"skip={offset} take={self._shuffle_buffer_size}",
            flush=True,
        )
        old = getattr(self, "dataset", None)
        self.dataset = None
        del old
        gc.collect()
        self.dataset, self.dataset_length, self.dataset_statistics = self._build_rlds(
            cache_offset=offset
        )

    def __len__(self) -> int:
        return self.dataset_length

    @staticmethod
    def _normalize_depth_window(depth: np.ndarray) -> torch.Tensor:
        """Convert RLDS depth ``[W, H, W, 1|C]`` → ``[W, 1, H, W]`` float in ``[0, 1]``."""
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., None]
        if d.shape[-1] > 1:
            d = d[..., :1]
        # Per-frame min-max (matches eval preprocessing).
        flat = d.reshape(d.shape[0], -1)
        dmin = flat.min(axis=1).reshape(-1, 1, 1, 1)
        dmax = flat.max(axis=1).reshape(-1, 1, 1, 1)
        d = (d - dmin) / (dmax - dmin + 1e-6)
        # [W, H, W, 1] → [W, 1, H, W]
        return torch.from_numpy(np.transpose(d, (0, 3, 1, 2)).copy())

    def _frame_to_sample(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        images = np.asarray(frame["observation"]["image_primary"])  # [W, H, W, 3] uint8
        if images.ndim == 3:
            images = images[None]

        # Action window: [window_size + fwd_pred_next_n, 7]; drop the last
        # overlapping step so chunking yields exactly `window_size` chunks.
        # `np.array(..., copy=True)`: the TF-backed buffer is read-only, but we
        # binarize the gripper in place below, so we need a writable array.
        action = np.array(frame["action"], dtype=np.float32, copy=True)[:-1]
        action_mask = np.asarray(frame["chunk_mask"], dtype=np.float32)[:-1]

        if self.norm_action:
            action = normalize_action(
                action, self.norm_min, self.norm_max, maintain_last=True
            )
        # Gripper already {0=close, 1=open}; binarize to clean BCE labels.
        action[..., -1] = (action[..., -1] == 1).astype(np.float32)

        rgb = self.image_fn([Image.fromarray(img) for img in images])  # [W, C, H, W]
        lang = frame["task"]["language_instruction"].decode()

        sample = {
            "rgb": rgb,
            "action": action,
            "action_mask": action_mask,
            "lang": lang,
        }
        if self.load_depth:
            if "depth_primary" not in frame["observation"]:
                raise KeyError(
                    "load_depth=True but observation has no 'depth_primary'. "
                    "Rebuild LIBERO RLDS with agentview depth (see "
                    "scripts/regenerate_libero_hdf5_with_depth.py)."
                )
            depth = np.asarray(frame["observation"]["depth_primary"])
            if depth.ndim == 2:
                depth = depth[None]
            sample["depth"] = self._normalize_depth_window(depth)
        return sample

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # One-shot cached (or non-depth) stream: iterate forever on current pipeline.
        if self._cache_refresh_every_n_samples is None:
            for frame in self.dataset.as_numpy_iterator():
                yield self._frame_to_sample(frame)
            return

        # Rotating windows: train on cached N for K*bs samples, then rebuild.
        while True:
            n_yielded = 0
            for frame in self.dataset.as_numpy_iterator():
                yield self._frame_to_sample(frame)
                n_yielded += 1
                if n_yielded >= self._cache_refresh_every_n_samples:
                    break
            self._refresh_cache_window()

    def collater(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_tensors = torch.stack([s["rgb"] for s in sample])[:, : self.window_size]
        action_tensors = torch.from_numpy(np.stack([s["action"] for s in sample])).float()
        action_mask = torch.from_numpy(np.stack([s["action_mask"] for s in sample])).float()
        stacked_language = [s["lang"] for s in sample]

        # [B, window_size, fwd_pred_next_n, 7] and [B, window_size, fwd_pred_next_n]
        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
        chunck_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)

        out = {
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
        if self.load_depth:
            out["depth"] = torch.stack([s["depth"] for s in sample])[:, : self.window_size]
        return out
