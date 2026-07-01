"""Minimal data helpers for CALVIN + LFM2.5-VL training."""

from __future__ import annotations

import functools
from typing import List

import numpy as np
import torch
from PIL import Image


def normalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    action = np.clip(action, a_min=float(action_min), a_max=float(action_max))
    res = 2 * (action - action_min) / (action_max - action_min) - 1
    if maintain_last:
        res[..., -1] = last_val
    return res


def regularize_action(x, x_mean, x_std, eps=1e-6, maintain_last=True):
    last_val = x[-1]
    res = (x - x_mean) / (x_std + eps)
    if maintain_last:
        res[-1] = last_val
    return res


def mu_law_companding(x, mu=255, maintain_last=True):
    last_val = x[-1]
    res = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if maintain_last:
        res[-1] = last_val
    return res


def world_to_tcp_frame(action, robot_obs):
    from data.pose_transforms import euler_angles_to_matrix, matrix_to_euler_angles

    flag = False
    if len(action.shape) == 4:
        flag = True
        b, s, f, _ = action.shape
        action = action.reshape(b, s * f, -1)
        robot_obs = robot_obs.reshape(b, s * f, -1)
    b, s, _ = action.shape
    world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ").float().reshape(-1, 3, 3)
    tcp_T_world = torch.inverse(world_T_tcp)
    pos_w_rel = action[..., :3].reshape(-1, 3, 1)
    pos_tcp_rel = tcp_T_world @ pos_w_rel
    orn_w_rel = action[..., 3:6] * 0.01
    world_T_tcp_new = euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ").float().reshape(-1, 3, 3)
    tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
    orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ").float()
    orn_tcp_rel = torch.where(orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel)
    orn_tcp_rel = torch.where(orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel)
    orn_tcp_rel *= 100
    action_tcp = torch.cat(
        [pos_tcp_rel.reshape(b, s, -1), orn_tcp_rel.reshape(b, s, -1), action[..., -1:]],
        dim=-1,
    )
    if flag:
        action_tcp = action_tcp.reshape(b, s, -1, action_tcp.shape[-1])
    return action_tcp


def preprocess_image(sample: List[Image.Image], image_processor, model_type: str) -> torch.Tensor:
    """Convert PIL frames to CHW tensors. LFM keeps native resolution (no resize/normalize)."""
    _ = model_type
    tensors = [image_processor(img).unsqueeze(0) for img in sample]
    return torch.cat(tensors, dim=0)


def get_text_function(tokenizer, tokenizer_type, max_length=256):
    if tokenizer_type in ("lfm2.5", "lfm2.5vl", "qwen25vl", "qwen3vl", "qwen3vlmoe"):

        def preprocess_text_vlm(sample, tokenizer=None):
            # LFM/Qwen-VL: tokenize in trainer via build_processor_inputs; pass raw strings.
            return sample, None

        return functools.partial(preprocess_text_vlm, tokenizer=tokenizer)

    def preprocess_text_default(sample, tokenizer):
        tokenizer.padding_side = "right"
        texts = [f"<|endoftext|>{s.strip()}" for s in sample]
        encoded = tokenizer(
            texts,
            truncation="only_first",
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    return functools.partial(preprocess_text_default, tokenizer=tokenizer)
