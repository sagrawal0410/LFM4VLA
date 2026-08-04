from __future__ import annotations

import copy
import os

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def build_vlm(vlm_config, tokenizer_config, precision="bfloat16"):
    """Build LFM VLM + processor.

    When ``vlm_config['_init_without_pretrained_weights']`` is True (eval /
    ``from_checkpoint``), construct the model from config only and load the
    processor/tokenizer from the resolved id. Weights then come from the
    Lightning ckpt — no need to download a full base checkpoint just to overwrite it.
    """
    vlm_config = copy.deepcopy(vlm_config)
    model_id = vlm_config.get("model_id") or vlm_config.get(
        "pretrained_model_name_or_path", "LiquidAI/LFM2.5-VL-1.6B"
    )
    # Prefer tokenizer path when it exists (may already be remapped to HF).
    processor_id = (
        (tokenizer_config or {}).get("pretrained_model_name_or_path") or model_id
    )
    dtype = precision if precision else "bfloat16"
    if isinstance(dtype, str):
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
    else:
        torch_dtype = dtype

    init_without_weights = bool(
        vlm_config.pop("_init_without_pretrained_weights", False)
    )

    # Local missing absolute paths should already be remapped by resolve_vlm_paths;
    # keep a last-resort hub default if something still looks like a dead path.
    if isinstance(model_id, str) and model_id.startswith("/") and not os.path.isdir(model_id):
        raise FileNotFoundError(
            f"VLM path does not exist: {model_id}. "
            "Copy the base checkpoint, or set vlm.model_id / model_url to a "
            "HuggingFace id (e.g. LiquidAI/LFM2.5-VL-450M)."
        )

    processor = AutoProcessor.from_pretrained(processor_id)

    if init_without_weights:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_config(config)
        model = model.to(dtype=torch_dtype)
        print(
            f"[vlm] initialized from config only ({model_id}); "
            "weights will be loaded from the Lightning checkpoint",
            flush=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )

    return model, processor
