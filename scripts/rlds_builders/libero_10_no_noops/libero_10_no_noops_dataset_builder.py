"""TFDS builder: LIBERO-10 HDF5 (with agentview_depth) → RLDS ``libero_10_no_noops``.

Expects HDF5 from ``scripts/regenerate_libero_hdf5_with_depth.py``:
  data/demo_i/obs/{agentview_rgb, eye_in_hand_rgb, agentview_depth, ee_states, gripper_states, joint_states}
  data/demo_i/actions

Set env ``LIBERO_DEPTH_HDF5_GLOB`` to the glob of input ``*.hdf5`` files before ``tfds build``.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds


def _language_from_filename(episode_path: str) -> str:
    raw = os.path.basename(episode_path)
    words = raw[:-10].split("_")  # strip "_demo.hdf5"
    command = ""
    for w in words:
        if "SCENE" in w:
            command = ""
            continue
        command = command + w + " "
    return command[:-1]


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _parse_example(episode_path: str, demo_id: int):
        with h5py.File(episode_path, "r") as F:
            key = f"demo_{demo_id}"
            if key not in F["data"].keys():
                return None
            demo = F["data"][key]
            actions = demo["actions"][()]
            states = demo["obs"]["ee_states"][()]
            gripper_states = demo["obs"]["gripper_states"][()]
            joint_states = demo["obs"]["joint_states"][()]
            images = demo["obs"]["agentview_rgb"][()]
            wrist_images = demo["obs"]["eye_in_hand_rgb"][()]
            depths = demo["obs"]["agentview_depth"][()]

        command = _language_from_filename(episode_path)
        episode = []
        for i in range(actions.shape[0]):
            depth_i = np.asarray(depths[i], dtype=np.float32)
            if depth_i.ndim == 2:
                depth_i = depth_i[..., None]
            # Match OpenVLA RGB convention: rot180.
            depth_i = depth_i[::-1, ::-1]
            episode.append(
                {
                    "observation": {
                        "image": images[i][::-1, ::-1],
                        "wrist_image": wrist_images[i][::-1, ::-1],
                        "depth": depth_i.astype(np.float32),
                        "state": np.asarray(
                            np.concatenate((states[i], gripper_states[i]), axis=-1),
                            np.float32,
                        ),
                        "joint_state": np.asarray(joint_states[i], dtype=np.float32),
                    },
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "discount": 1.0,
                    "reward": float(i == (actions.shape[0] - 1)),
                    "is_first": i == 0,
                    "is_last": i == (actions.shape[0] - 1),
                    "is_terminal": i == (actions.shape[0] - 1),
                    "language_instruction": command,
                }
            )
        return episode_path + f"_{demo_id}", {
            "steps": episode,
            "episode_metadata": {"file_path": episode_path},
        }

    for sample in paths:
        with h5py.File(sample, "r") as F:
            n_demos = len(F["data"])
        idx = cnt = 0
        while cnt < n_demos:
            ret = _parse_example(sample, idx)
            if ret is not None:
                cnt += 1
                yield ret
            idx += 1


class Libero10NoNoops(tfds.core.GeneratorBasedBuilder):
    # CamelCase ``Libero10NoNoops`` would become ``libero10_no_noops``; force the
    # OpenVLA / OXE name used by configs (``libero_10_no_noops``).
    name = "libero_10_no_noops"
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "LIBERO-10 with agentview GT depth."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="LIBERO-10 no-noop demos with agentview GT depth.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "depth": tfds.features.Tensor(
                                        shape=(256, 256, 1),
                                        dtype=np.float32,
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,), dtype=np.float32
                                    ),
                                    "joint_state": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {"file_path": tfds.features.Text()}
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        pattern = os.environ.get(
            "LIBERO_DEPTH_HDF5_GLOB",
            "/home/teams/research/robotics/libero_hdf5_depth/libero_10_no_noops/*.hdf5",
        )
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No HDF5 files for LIBERO_DEPTH_HDF5_GLOB={pattern!r}"
            )
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths):
        return _generate_examples(paths)
