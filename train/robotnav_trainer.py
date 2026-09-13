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

import math
from typing import Any, Dict, List

import torch

from train.base_trainer import BaseTrainer


class RobotNavTrainer(BaseTrainer):

    # ------------------------------------------------- task-space metrics --
    # Config-gated (val_waypoint_metrics: true). Loss scales differ across
    # heads (Huber/MSE/L1/flow-matching), so cross-implementation comparison
    # uses geometry computed from each head's INFERENCE output on the shared
    # val set: ADE / FDE (meters), yaw error (deg), FDE-success@0.25m, and
    # minADE-of-K for stochastic (flow-matching) heads (val_minade_k > 1).

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = super().validation_step(batch, batch_idx, dataloader_idx)
        if not self.configs.get("val_waypoint_metrics", False):
            return out
        traj = None
        if isinstance(batch, dict):
            if batch.get("data_source") == "robotnav_mixed":
                traj = batch.get("traj")
            elif batch.get("data_source") == "robotnav_traj":
                traj = batch
        if traj is not None and traj.get("wp_scale") is not None:
            try:
                self._log_waypoint_metrics(traj)
            except Exception as e:  # noqa: BLE001 — metrics must never kill val
                print(f"[wpm] waypoint metrics failed: {e}", flush=True)
        return out

    def _predict_waypoints(self, traj: Dict[str, Any]) -> torch.Tensor:
        """One inference pass -> denormalizable waypoints [B, K, 3] (last slot).

        Rebuilds processor inputs each call: the model's forward consumes
        (pops) the processor dict, and stochastic heads need fresh noise.
        """
        inputs = self._process_batch(traj)
        pred = self.model.forward(
            inputs["rgb"],
            inputs["language"],
            attention_mask=inputs["text_mask"],
            action_labels=None,
            action_mask=None,
            vision_gripper=inputs["hand_rgb"],
            raw_text=inputs["raw_text"],
            rel_state=inputs["rel_state"],
            depth=inputs["depth"],
            frame_offsets=inputs.get("frame_offsets"),
            mode="inference",
        )
        if isinstance(pred, dict):
            pred = pred.get("action", pred.get("actions"))
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return pred[:, -1].float()                       # [B, K, 3]

    @torch.no_grad()
    def _log_waypoint_metrics(self, traj: Dict[str, Any]) -> None:
        gt = traj["action_chunck"][:, -1].to(self.device).float()   # [B, K, 3]
        mask = traj["chunck_mask"][:, -1].to(self.device).bool()    # [B, K]
        valid = mask.any(-1)
        if not valid.any():
            return
        scale = traj["wp_scale"].to(self.device).unsqueeze(1)       # [B, 1, 3]
        n_draws = max(1, int(self.configs.get("val_minade_k", 1)))
        preds = torch.stack(
            [self._predict_waypoints(traj).to(self.device) * scale
             for _ in range(n_draws)])                              # [S, B, K, 3]
        gt_m = gt * scale

        derr = torch.linalg.norm(preds[..., :2] - gt_m[None, ..., :2], dim=-1)
        nvalid = mask.sum(-1).clamp(min=1)
        ade_s = (derr * mask[None]).sum(-1) / nvalid                # [S, B]
        last = mask.float().cumsum(-1).argmax(-1)                   # [B]
        fde_s = derr[:, torch.arange(derr.shape[1], device=derr.device), last]
        yd = preds[..., 2] - gt_m[None, ..., 2]
        yd = torch.remainder(yd + math.pi, 2 * math.pi) - math.pi
        yerr_s = (yd.abs() * mask[None]).sum(-1) / nvalid           # [S, B]

        bsz = int(valid.sum())
        log = dict(sync_dist=True, on_epoch=True, on_step=False, batch_size=bsz)
        self.log("val_ade_m", ade_s[0][valid].mean(), **log)
        self.log("val_fde_m", fde_s[0][valid].mean(), **log)
        self.log("val_yaw_deg", yerr_s[0][valid].mean() * 180.0 / math.pi, **log)
        self.log("val_fde_success_25cm",
                 (fde_s[0][valid] < 0.25).float().mean(), **log)
        if n_draws > 1:
            self.log(f"val_minade{n_draws}_m",
                     ade_s.min(0).values[valid].mean(), **log)
            self.log(f"val_avgade{n_draws}_m",
                     ade_s.mean(0)[valid].mean(), **log)
            self.log(f"val_avgfde{n_draws}_m",
                     fde_s.mean(0)[valid].mean(), **log)

    # ------------------------------------------------------------ LEPIG --
    def _world_loss(self, traj_batch, weights=None):
        """Plans B/C: predict the future V-JEPA2 latent, score against the real one.

        Targets are frozen: the encoder is eval-mode, requires_grad=False, and
        the target is stop-gradient, so this trains the bridge+predictor only
        and can never collapse by moving the target.
        """
        wb = getattr(self, "world_branch", None)
        if wb is None or traj_batch.get("future_rgb") is None:
            return None
        fut = traj_batch["future_rgb"].to(self.device)          # [B, H, C, h, w]
        with torch.no_grad():
            tgt = self.vjepa(fut)                                # [B, H, N, D] frozen
        ctx = self._world_context(traj_batch)                    # backbone features
        pred = self.world_branch(ctx)                            # [B, H, N, D]
        loss = self.world_branch.loss(pred, tgt, weights=weights)
        return loss

    def _world_context(self, traj_batch):
        """Backbone hidden states the world branch reads from."""
        hs = getattr(self, "_last_action_hs", None)
        if hs is None:
            raise RuntimeError(
                "world branch needs backbone features; _last_action_hs unset")
        return hs


    def _lepig_step(self, traj_batch):
        """Per-step LEPIG lifecycle: snapshots, refresh, scoring, weights.

        Returns detached per-example weights, or None when LEPIG is disabled or
        still in warmup (in which case training is the uniform baseline, which
        is exactly the control the plan document requires).
        """
        lep = getattr(self, "lepig", None)
        if lep is None or not lep.enabled:
            return None
        from models.lepig.hooks import selected_params, jacobian_rows
        step = int(self.global_step)
        max_steps = int(self.configs["trainer"].get("max_steps", 80000))
        params = selected_params(self.model, lep.plan)
        lep.on_step(step, params)
        lep.warm(step, max_steps)
        if lep.should_refresh(step) and lep.refresh():
            # curvature + anchors from THIS batch's geometry; both are rebuilt
            # every refresh and never reused across snapshot versions.
            G = jacobian_rows(self, traj_batch, params, lep.subspace,
                              lep.rank)
            if G is not None:
                for i in range(G.shape[0]):
                    lep.add_calibration(G[i])
                    lep.add_anchor(G[i])
        if not lep.ready:
            return None
        G = jacobian_rows(self, traj_batch, params, lep.subspace, lep.rank)
        if G is None:
            return None
        scores = lep.score_batch([G[i] for i in range(G.shape[0])])
        w = lep.weights(scores, G.shape[0], device=self.device)
        self._lepig_last_w = w
        return w

    def _forward_batch(self, batch: Dict[str, Any], mode: str = "train"):
        multi = getattr(self.trainer, "world_size", 1) > 1
        if isinstance(batch, dict) and batch.get("data_source") == "robotnav_vl":
            out = dict(self._forward_vl_batch(batch))
            if multi and mode == "train":
                # Batch-mode multi-rank parity: FSDP's grad bookkeeping
                # asserts (`_saved_grad_shard`) when module participation
                # differs across accumulated micro-batches, so a homogeneous
                # VL batch also runs the action branch at ZERO weight.
                dummy = super()._forward_batch(self._dummy_traj_batch(),
                                               mode=mode)
                zero = sum(v.float().sum() for v in dummy.values()
                           if torch.is_tensor(v) and v.requires_grad)
                out["loss"] = zero * 0.0
            return out
        if isinstance(batch, dict) and batch.get("data_source") == "robotnav_mixed":
            # Sample-level 85/15 mixing: one batch carries a trajectory
            # sub-batch and a VL sub-batch; both losses join in ONE update.
            out: Dict[str, Any] = {"loss": None}
            if batch.get("traj") is not None:
                w = self._lepig_step(batch["traj"]) if mode == "train" else None
                if w is not None:
                    batch["traj"]["lepig_w"] = w
                out = dict(super()._forward_batch(batch["traj"], mode=mode))
            if batch.get("vl") is not None:
                out["loss_vl_cotrain"] = self._forward_vl_batch(
                    batch["vl"])["loss_vl_cotrain"]
            else:
                # Multi-rank safety: every rank must execute the LM branch on
                # every step (FSDP/sync_dist collectives hang on divergence).
                # A batch with no VL samples runs a tiny dummy conversation at
                # ZERO weight — module participation without statistics.
                out["loss_vl_cotrain"] = self._forward_vl_batch(
                    self._dummy_vl_batch())["loss_vl_cotrain"] * 0.0
            return out
        out = super()._forward_batch(batch, mode=mode)
        if (multi and mode == "train" and isinstance(batch, dict)
                and batch.get("data_source") == "robotnav_traj"):
            # Mirror of the VL-batch parity above: homogeneous trajectory
            # batches run the LM branch at ZERO weight.
            out = dict(out)
            out["loss_vl_cotrain"] = self._forward_vl_batch(
                self._dummy_vl_batch())["loss_vl_cotrain"] * 0.0
        return out

    def _dummy_traj_batch(self) -> Dict[str, Any]:
        """Tiny action-branch batch for zero-weighted participation parity."""
        if not hasattr(self, "_dummy_traj"):
            ws = int(self.configs["window_size"])
            K = int(self.configs["fwd_pred_next_n"])
            mask = torch.zeros(1, ws, K)
            mask[:, -1, :] = 1.0                 # avoid all-masked loss NaNs
            dev = self.device
            self._dummy_traj = {
                "rgb": [[torch.zeros(3, 96, 96, dtype=torch.uint8)
                         for _ in range(ws)]],
                "hand_rgb": None,
                "action": torch.zeros(1, ws, 3, device=dev),
                "text": ["stop."],
                "text_mask": None,
                "action_chunck": torch.zeros(1, ws, K, 3, device=dev),
                "chunck_mask": mask.to(dev),
                "raw_text": ["stop."],
                "data_source": "robotnav_traj",
                "family": ["vln_r2r"],
                "wp_scale": torch.ones(1, 3, device=dev),
                "frame_offsets": torch.stack([
                    torch.arange(ws - 1, -1, -1, dtype=torch.float32),
                    torch.ones(ws)], -1)[None].to(dev),
            }
        return self._dummy_traj

    def _dummy_vl_batch(self) -> Dict[str, Any]:
        if not hasattr(self, "_dummy_vl"):
            from PIL import Image
            self._dummy_vl = {
                "data_source": "robotnav_vl",
                "vl_images": [[Image.new("RGB", (96, 96))]],
                "vl_user": ["Describe the image."],
                "vl_answer": ["Empty."],
                "raw_text": ["Describe the image."],
            }
        return self._dummy_vl

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
