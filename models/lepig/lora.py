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


def _find_blocks(model) -> List[nn.Module]:
    """Locate the VLM transformer block list, whatever it is called here."""
    cands = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 4:
            child = mod[0]
            names = {n for n, _ in child.named_modules()}
            if any(t in n for n in names for t in DEFAULT_ATTN):
                cands.append((name, mod))
    if not cands:
        return []
    # Deepest/longest stack is the language tower, not a small adapter stack.
    cands.sort(key=lambda kv: (len(kv[1]), -kv[0].count(".")))
    return list(cands[-1][1])


def inject_lora(model, last_n: int = 4, rank: int = 32, alpha: int = 64,
                dropout: float = 0.0,
                attention_targets: Sequence[str] = DEFAULT_ATTN,
                mlp_targets: Sequence[str] = DEFAULT_MLP) -> int:
    """Wrap the target Linears in the last `last_n` blocks. Returns sites wrapped."""
    blocks = _find_blocks(model)
    if not blocks:
        raise RuntimeError("inject_lora: could not locate the VLM block stack")
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
