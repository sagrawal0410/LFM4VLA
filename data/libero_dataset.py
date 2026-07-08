"""Streaming LIBERO dataset for LFM2.5-VL training.

Wraps the OpenVLA/OXE RLDS (TensorFlow) pipeline and emits batches that match the
exact contract produced by :class:`data.calvin_dataset.DiskCalvinDataset` so the
same trainer/model consume both sources without changes.

Unlike CALVIN (map-style, disk ``.npz``), LIBERO is streamed frame-by-frame from
TFDS RLDS. Each demo is expanded into sliding-window samples: one current image +
instruction -> a chunk of ``fwd_pred_next_n`` future actions.

Data: ``modified_libero_rlds`` (HuggingFace, ~10 GB), no-op actions removed.
Requires ``tensorflow==2.15``, ``tensorflow_datasets==4.9.3`` and ``dlimp`` in the env.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

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

        self.dataset, self.dataset_length, self.dataset_statistics = self._build_rlds(
            data_root_dir=str(data_root_dir),
            data_mix=data_mix,
            image_size=image_size,
            window_size=window_size,
            fwd_pred_next_n=fwd_pred_next_n,
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
        )

    @staticmethod
    def _build_rlds(
        data_root_dir: str,
        data_mix: str,
        image_size: int,
        window_size: int,
        fwd_pred_next_n: int,
        shuffle_buffer_size: int,
        train: bool,
        image_aug: bool,
    ):
        from prismatic.vla.datasets.rlds import make_interleaved_dataset
        from prismatic.vla.datasets.rlds.oxe import (
            OXE_NAMED_MIXTURES,
            get_oxe_dataset_kwargs_and_weights,
        )
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

        mixture_spec = OXE_NAMED_MIXTURES.get(data_mix, [(data_mix, 1.0)])
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),  # agentview only; wrist not fed to the model
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        frame_transform_kwargs: Dict[str, Any] = dict(
            resize_size=(image_size, image_size),
            num_parallel_calls=16,
        )
        if image_aug:
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
                window_size=window_size,
                chunk_action=True,
                frame_num=-1,
                future_action_window_size=fwd_pred_next_n,
                left_pad=False,
                window_sample="sliding",
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )
        return make_interleaved_dataset(**rlds_config)

    def __len__(self) -> int:
        return self.dataset_length

    def __iter__(self):
        for frame in self.dataset.as_numpy_iterator():
            images = np.asarray(frame["observation"]["image_primary"])  # [W, H, W, 3] uint8
            if images.ndim == 3:
                images = images[None]

            # Action window: [window_size + fwd_pred_next_n, 7]; drop the last
            # overlapping step so chunking yields exactly `window_size` chunks.
            action = np.asarray(frame["action"], dtype=np.float32)[:-1]
            action_mask = np.asarray(frame["chunk_mask"], dtype=np.float32)[:-1]

            if self.norm_action:
                action = normalize_action(action, self.norm_min, self.norm_max, maintain_last=True)
            # Gripper already {0=close, 1=open}; binarize to clean BCE labels.
            action[..., -1] = (action[..., -1] == 1).astype(np.float32)

            rgb = self.image_fn([Image.fromarray(img) for img in images])  # [W, C, H, W]
            lang = frame["task"]["language_instruction"].decode()

            yield {"rgb": rgb, "action": action, "action_mask": action_mask, "lang": lang}

    def collater(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_tensors = torch.stack([s["rgb"] for s in sample])[:, : self.window_size]
        action_tensors = torch.from_numpy(np.stack([s["action"] for s in sample])).float()
        action_mask = torch.from_numpy(np.stack([s["action_mask"] for s in sample])).float()
        stacked_language = [s["lang"] for s in sample]

        # [B, window_size, fwd_pred_next_n, 7] and [B, window_size, fwd_pred_next_n]
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
