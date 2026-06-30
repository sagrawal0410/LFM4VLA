"""Build CALVIN datasets wired to the LFM policy preprocessing."""

from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any, Dict

import data
from data.data_utils import get_text_function, preprocess_image


def build_dataset(dataset_cfg: Dict[str, Any], variant: Dict[str, Any], policy_model) -> data.DiskCalvinDataset:
    cfg = copy.deepcopy(dataset_cfg)
    dataset_type = cfg.pop("type")
    if dataset_type != "DiskCalvinDataset":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    data_dir = cfg.pop("data_dir")
    if not os.path.isabs(data_dir):
        data_root = variant.get("data_root", "")
        if data_root:
            data_dir = os.path.join(data_root, data_dir)

    model_type = variant.get("model", "lfm1.6")
    tokenizer_type = variant["tokenizer"].get("tokenizer_type", "lfm2.5")

    image_fn = functools.partial(
        preprocess_image,
        image_processor=policy_model.image_processor,
        model_type=model_type,
    )

    dataset_cls = getattr(data, dataset_type)
    return dataset_cls(
        image_fn=image_fn,
        tokenizer=policy_model.tokenizer,
        data_dir=Path(data_dir),
        window_size=variant["window_size"],
        fwd_pred_next_n=variant["fwd_pred_next_n"],
        norm_action=variant.get("norm_action", False),
        norm_min=variant.get("norm_min", -1),
        norm_max=variant.get("norm_max", 1),
        model_name=cfg.pop("model_name", tokenizer_type),
        **cfg,
    )
