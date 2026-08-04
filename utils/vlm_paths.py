"""Resolve VLM / tokenizer pretrained paths for cluster → local eval.

Training configs often bake in a cluster absolute path such as
``/home/teams/research/robotics/checkpoints/lfm25_vl_...``. That path is not
present on workstations. When the local directory is missing, fall back to the
public HuggingFace id derived from ``model_url`` / ``model``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse


_MODEL_TO_HF = {
    "lfm450m": "LiquidAI/LFM2.5-VL-450M",
    "lfm1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    "lfm1.6": "LiquidAI/LFM2.5-VL-1.6B",
}


def hf_id_from_configs(configs: Dict[str, Any]) -> Optional[str]:
    model_url = str(configs.get("model_url") or "")
    if "huggingface.co/" in model_url:
        # https://huggingface.co/LiquidAI/LFM2.5-VL-450M -> LiquidAI/LFM2.5-VL-450M
        path = urlparse(model_url).path.strip("/")
        if path.count("/") >= 1:
            return path
    return _MODEL_TO_HF.get(str(configs.get("model", "")).lower())


def resolve_pretrained_ref(path_or_id: Optional[str], configs: Dict[str, Any]) -> Optional[str]:
    """Return an existing local dir, or a HuggingFace hub id if the local path is gone."""
    if not path_or_id:
        return path_or_id
    if os.path.isdir(path_or_id):
        return path_or_id
    # Already a hub id (org/name), not an absolute filesystem path.
    if "/" in path_or_id and not path_or_id.startswith("/") and not path_or_id.startswith("."):
        return path_or_id

    fallback = hf_id_from_configs(configs)
    if fallback:
        print(
            f"[vlm] local path missing: {path_or_id!r}\n"
            f"      falling back to HuggingFace id: {fallback}",
            flush=True,
        )
        return fallback
    return path_or_id


def resolve_vlm_paths_in_configs(configs: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``configs`` so vlm/tokenizer pretrained refs are loadable on this machine."""
    vlm = configs.setdefault("vlm", {})
    tok = configs.setdefault("tokenizer", {})
    for key in ("model_id", "pretrained_model_name_or_path"):
        if key in vlm and vlm[key]:
            vlm[key] = resolve_pretrained_ref(vlm[key], configs)
    if tok.get("pretrained_model_name_or_path"):
        tok["pretrained_model_name_or_path"] = resolve_pretrained_ref(
            tok["pretrained_model_name_or_path"], configs
        )
    return configs
