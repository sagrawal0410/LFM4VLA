"""RobotNav trainer: BaseTrainer + VL co-training batches (85/15 mixture).

Trajectory batches ("robotnav_traj") follow the standard contract and flow
through BaseTrainer untouched. VL batches ("robotnav_vl") are routed through
the backbone LM with next-token labels; the resulting LM loss is returned as
``loss_vl_cotrain``, which BaseTrainer._get_loss already scales by
``vl_cotrain_ratio`` (set 1.0 in the config for the paper's λ = 1.0).

Minimal-change design: only ``_forward_batch`` is overridden; everything else
(loss aggregation, logging, optimizers, checkpointing) is inherited.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from train.base_trainer import BaseTrainer


class RobotNavTrainer(BaseTrainer):

    def _forward_batch(self, batch: Dict[str, Any], mode: str = "train"):
        if isinstance(batch, dict) and batch.get("data_source") == "robotnav_vl":
            return self._forward_vl_batch(batch)
        return super()._forward_batch(batch, mode=mode)

    # ------------------------------------------------------------------ VL --
    def _build_conversations(self, batch: Dict[str, Any]) -> List[List[dict]]:
        convs = []
        for imgs, user, answer in zip(batch["vl_images"], batch["vl_user"],
                                      batch["vl_answer"]):
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": user})
            convs.append([
                {"role": "user", "content": content},
                {"role": "assistant",
                 "content": [{"type": "text", "text": answer}]},
            ])
        return convs

    def _forward_vl_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        model = self.model                      # RoboLFM25VL
        processor = model.processor
        convs = self._build_conversations(batch)

        full = processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", padding=True)
        prompts = processor.apply_chat_template(
            [c[:1] for c in convs], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True)

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        # mask everything up to (and including) the generation prompt: only the
        # assistant answer supervises. prompt length = per-row non-pad count.
        prompt_lens = prompts["attention_mask"].sum(dim=1)
        pad_left = bool(getattr(processor.tokenizer, "padding_side", "right") == "left")
        for i in range(input_ids.shape[0]):
            n = int(prompt_lens[i])
            if pad_left:
                pad = int((attention_mask[i] == 0).sum())
                labels[i, : pad + n] = -100
            else:
                labels[i, :n] = -100

        device = next(model.backbone.parameters()).device
        fwd = {k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in full.items()}
        fwd["labels"] = labels.to(device)
        out = model.backbone(**fwd)

        lm_loss = out.loss
        if lm_loss is None or not torch.isfinite(lm_loss):
            lm_loss = input_ids.new_zeros((), dtype=torch.float32)
        return {"loss": None, "loss_vl_cotrain": lm_loss}
