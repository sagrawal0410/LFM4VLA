"""LoRA adapters for the last N VLM blocks.

Required by the plan for A2 and for the B/C world posterior: the posterior
needs a low-dimensional foothold inside the representation-producing network,
because a posterior confined to the action head or the world adapter measures
"how unsure is that head?" rather than "where is the VLA representation
underdetermined?" (plan document, section 3.7 and the Plan B posterior note).

Spec: last_n_vlm_blocks=4, rank=32, alpha=64, dropout=0.0, bias=none,
attention_targets=[q_proj,k_proj,v_proj,o_proj],
mlp_targets=[gate_proj,up_proj,down_proj].

The base VLM weights stay trainable -- these adapters add an explicit
coordinate set for the posterior, they do not freeze anything.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

DEFAULT_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_MLP = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(x)). Identity at init because B starts at 0."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)          # exact identity at step 0
        self.lora_A.weight.data = self.lora_A.weight.data.to(base.weight.dtype)
        self.lora_B.weight.data = self.lora_B.weight.data.to(base.weight.dtype)

    def forward(self, x):
        out = self.base(x)
        h = self.lora_A(self.lora_drop(x).to(self.lora_A.weight.dtype))
        return out + self.scaling * self.lora_B(h).to(out.dtype)


LANG_HINTS = ("language_model", "text_model", "llm", "language_tower", "decoder")
VISION_HINTS = ("vision", "visual", "image_tower", "siglip", "vit")


def _find_blocks(model, prefer_language: bool = True) -> List[nn.Module]:
    """Locate the LANGUAGE tower's transformer block list.

    Targeting must be explicit, not "longest stack with attention names": a VLM
    also has a vision tower whose blocks carry q/k/v/out_proj, and adapters
    placed there would express uncertainty about image features rather than
    about the representation the action expert actually consumes -- the same
    error as confining the posterior to an adapter, just relocated.
    """
    cands = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 4:
            leaves = {n.split(".")[-1] for n, sub in mod[0].named_modules()
                      if isinstance(sub, nn.Linear)}
            if not leaves:
                continue
            low = name.lower()
            is_vision = any(h in low for h in VISION_HINTS)
            is_lang = any(h in low for h in LANG_HINTS)
            cands.append((name, mod, is_lang, is_vision, len(mod)))
    if not cands:
        return []
    if prefer_language:
        lang = [c for c in cands if c[2] and not c[3]]
        if lang:
            lang.sort(key=lambda c: c[4])
            return list(lang[-1][1])
        nonvis = [c for c in cands if not c[3]]
        if nonvis:
            nonvis.sort(key=lambda c: c[4])
            return list(nonvis[-1][1])
    cands.sort(key=lambda c: c[4])
    return list(cands[-1][1])


def inject_lora(model, last_n: int = 4, rank: int = 32, alpha: int = 64,
                dropout: float = 0.0,
                attention_targets: Sequence[str] = DEFAULT_ATTN,
                mlp_targets: Sequence[str] = DEFAULT_MLP) -> int:
    """Wrap the target Linears in the last `last_n` blocks. Returns sites wrapped."""
    blocks = _find_blocks(model)
    if not blocks:
        raise RuntimeError("inject_lora: could not locate the VLM block stack")
    leaves = sorted({n.split(".")[-1] for n, sub in blocks[-1].named_modules()
                     if isinstance(sub, nn.Linear)})
    print(f"[lepig] LoRA stack: {len(blocks)} blocks, leaf linears={leaves}",
          flush=True)
    targets = tuple(attention_targets) + tuple(mlp_targets)
    wrapped = 0
    for blk in blocks[-int(last_n):]:
        for name, mod in list(blk.named_modules()):
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear):
                parent = blk
                for part in name.split(".")[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, leaf,
                        LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))
                wrapped += 1
    return wrapped


def lora_parameters(model) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
