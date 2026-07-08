"""Build CALVIN / LIBERO datasets wired to the LFM policy preprocessing."""

from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any, Dict

import data
from data.data_utils import preprocess_image


def build_dataset(dataset_cfg: Dict[str, Any], variant: Dict[str, Any], policy_model):
    cfg = copy.deepcopy(dataset_cfg)
    dataset_type = cfg.pop("type")

    model_type = variant.get("model", "lfm1.6")
    tokenizer_type = variant["tokenizer"].get("tokenizer_type", "lfm2.5")

    image_fn = functools.partial(
        preprocess_image,
        image_processor=policy_model.image_processor,
        model_type=model_type,
    )

    common = dict(
        image_fn=image_fn,
        tokenizer=policy_model.tokenizer,
        window_size=variant["window_size"],
        fwd_pred_next_n=variant["fwd_pred_next_n"],
        norm_action=variant.get("norm_action", False),
        norm_min=variant.get("norm_min", -1),
        norm_max=variant.get("norm_max", 1),
    )

    if dataset_type == "DiskCalvinDataset":
        data_dir = _resolve_path(cfg.pop("data_dir"), variant)
        return data.DiskCalvinDataset(
            data_dir=Path(data_dir),
            model_name=cfg.pop("model_name", tokenizer_type),
            **common,
            **cfg,
        )

    if dataset_type == "LiberoRLDSDataset":
        data_root_dir = _resolve_path(cfg.pop("data_root_dir"), variant)
        return data.LiberoRLDSDataset(
            data_root_dir=data_root_dir,
            **common,
            **cfg,
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _resolve_path(path: str, variant: Dict[str, Any]) -> str:
    if not os.path.isabs(path):
        data_root = variant.get("data_root", "")
        if data_root:
            path = os.path.join(data_root, path)
    return path
