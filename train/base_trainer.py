import os
from functools import partial

import lightning.pytorch as pl
import torch
import torch.distributed as dist

import models as RoboVLM_Backbone
from train.train_utils import adjust_learning_rate, convert_old_state_dict
from utils.dist_train import get_rank


class BaseTrainer(pl.LightningModule):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model_fn = getattr(RoboVLM_Backbone, configs["robovlm_name"])
        self._initialize()
        self._init_lepig()
        self.save_hyperparameters()

        val_dataset = configs["val_dataset"]
        if isinstance(val_dataset, list):
            self.val_set_names = [self._parse_dataset_name(cfg) for cfg in val_dataset]
        elif isinstance(val_dataset, dict):
            self.val_set_names = None
        else:
            raise NotImplementedError

    def _init_policy(self):
        model = self.model_fn(
            configs=self.configs,
            train_setup_configs=self.configs["train_setup"],
            fwd_head_configs=self.configs["fwd_head"],
            window_size=self.configs["window_size"],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs["act_head"],
            fwd_pred_next_n=self.configs["fwd_pred_next_n"],
            use_vision_resampler=self.configs.get("use_vision_resampler", False),
            vision_resampler_configs=self.configs.get("vision_resampler", None),
            use_clip_norm=self.configs.get("use_clip_norm", False),
            use_state=self.configs.get("use_state", False),
            use_depth=self.configs.get("use_depth", False),
            depth_configs=self.configs.get("depth"),
        )
        model.train()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Trainable Model Parameters: {total_params / 1e6:.2f}M")
        return model

    def _parse_dataset_name(self, dataset_config):
        dataset_path = dataset_config["data_dir"]
        for name in ("calvin", "bridge", "libero", "humanoid"):
            if name in dataset_path.lower():
                return name
        return "UNKNOWN_DATA"

    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):
        self.use_hand_rgb = self.configs["use_hand_rgb"]
        self.model = self._init_policy()

        self.cap_loss_ratio = self.configs["cap_loss_ratio"]
        self.arm_gripper_loss_ratio = self.configs["arm_gripper_loss_ratio"]
        self.fwd_loss_ratio = self.configs["fwd_loss_ratio"]
        self.kl_div_ratio = self.configs.get("kl_div_ratio", 0.05)
        self.clip_norm_ratio = self.configs.get("clip_norm_ratio", 0.05)
        self.vl_cotrain_ratio = self.configs.get("vl_cotrain_ratio", 0.05)

        self.act_pred = self.configs["train_setup"]["predict_action"]
        self.fwd_pred = self.configs["train_setup"]["predict_forward"]
        self.fwd_pred_hand = self.configs["train_setup"]["predict_forward_hand"]
        self.cap_pred = self.configs["train_setup"]["predict_caption"]

    @classmethod
    def _init_lepig(self):
        """Build the LEPIG controller and (for plans B/C) the world branch."""
        from models.lepig.controller import LepigController
        cfg = self.configs.get("lepig") or {}
        self.lepig = LepigController(cfg)
        self.world_branch = None
        self.vjepa = None
        if not self.lepig.enabled:
            return
        if self.lepig.weights_the_world_loss or cfg.get("world_loss", False):
            from models.lepig.world import WorldBranch
            wcfg = cfg.get("world_branch", {}) or {}
            in_features = int(cfg.get("backbone_hidden", 2048))
            self.world_branch = WorldBranch(
                in_features=in_features,
                action_dim=int(self.configs.get("act_head", {}).get("action_dim", 3)),
                action_conditioned=bool(cfg.get("action_conditioned", False)),
                n_horizons=len(wcfg.get("horizons_seconds", [0.5, 1.0, 2.0])),
                target_dim=int(cfg.get("target_dim", 1024)),
                **{k: v for k, v in wcfg.items() if k != "horizons_seconds"})
            self.lambda_world = float(cfg.get("lambda_world", 1.0))

    def from_checkpoint(cls, ckpt_path=None, ckpt_source="torch", configs=None):
        if ckpt_path is None:
            return cls(configs)

        # Eval / resume: build architecture from HF config+tokenizer only, then
        # load finetuned weights from the Lightning ckpt. Avoids requiring the
        # cluster base-VLM directory and downloading full pretrained weights.
        if configs is not None:
            from utils.vlm_paths import resolve_vlm_paths_in_configs

            resolve_vlm_paths_in_configs(configs)
            configs.setdefault("vlm", {})["_init_without_pretrained_weights"] = True

        model = cls(configs)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = convert_old_state_dict(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        cls._main_rank_print(msg)
        return model

    def configure_optimizers(self):
        eff_lr = self.configs["learning_rate"]
        optimizer = torch.optim.AdamW(self.get_grouped_params(self.model), lr=eff_lr)

        num_training_batches = self.trainer.estimated_stepping_batches
        max_epochs = self.configs["trainer"]["max_epochs"]
        if not num_training_batches or num_training_batches <= 0:
            num_training_batches = max_epochs * 1000

        iter_per_epoch = max(num_training_batches / max_epochs, 1.0)
        warmup_epochs = self.configs.get("warmup_epochs", 0)
        warmup_steps = self.configs.get("warmup_steps", 0)
        max_iters = self.configs["trainer"].get("max_steps", -1)
        if max_iters == -1:
            max_iters = self.configs["trainer"]["max_epochs"] * iter_per_epoch

        lr_scheduler_configs = {
            "warmup_iters": warmup_epochs * iter_per_epoch + warmup_steps,
            "iters": max_iters,
            "min_lr_scale": self.configs["min_lr_scale"],
        }

        scheduler_type = self.configs.get("scheduler", "constant")
        from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, int(lr_scheduler_configs["warmup_iters"]))
        elif scheduler_type == "half-cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=partial(adjust_learning_rate, configs=lr_scheduler_configs),
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                int(lr_scheduler_configs["warmup_iters"]),
                num_training_steps=int(lr_scheduler_configs["iters"]),
            )
        else:
            raise NotImplementedError(scheduler_type)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _get_loss(self, prediction):
        loss_arm_act = prediction.get("loss_arm_act")
        loss_gripper_act = prediction.get("loss_gripper_act")
        loss_depth_act = prediction.get("loss_depth_act")
        loss_obs = prediction.get("loss_obs_fwd")
        loss_hand_obs = prediction.get("loss_hand_obs_fwd")
        acc_gripper_act = prediction.get("acc_gripper_act")
        loss_cap = prediction.get("loss_cap")
        loss_kl = prediction.get("loss_kl")
        loss_vl_cotrain = prediction.get("loss_vl_cotrain")
        # head returns loss_stop; _update_loss suffixes it with the modality
        loss_stop = prediction.get("loss_stop_act") or prediction.get("loss_stop")
        clip_l1 = prediction.get("text_l1_clip")

        loss = prediction.get("loss")
        if loss is None:
            loss = torch.tensor(0.0, device=self.device)
        elif not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)

        if self.act_pred:
            loss_act = (loss_arm_act or 0) + (
                (loss_gripper_act or 0) * self.arm_gripper_loss_ratio)
            if not isinstance(loss_act, torch.Tensor):
                loss_act = torch.tensor(float(loss_act), device=self.device)
            if prediction.get("loss") is None:
                loss = loss + loss_act
            if loss_kl is not None:
                loss = loss + self.kl_div_ratio * loss_kl
            if clip_l1 is not None:
                loss = loss + self.clip_norm_ratio * clip_l1
        else:
            loss_act = None

        if self.fwd_pred:
            if loss_obs is not None:
                loss = loss + self.fwd_loss_ratio * loss_obs
            if self.fwd_pred_hand and loss_hand_obs is not None:
                loss = loss + self.fwd_loss_ratio * loss_hand_obs
        if loss_cap is not None:
            loss = loss + self.cap_loss_ratio * loss_cap
        if loss_vl_cotrain is not None:
            loss = loss + self.vl_cotrain_ratio * loss_vl_cotrain
        # Explicit stop channel; already scaled by stop_loss_weight in the head.
        if loss_stop is not None:
            loss = loss + loss_stop

        return {
            "loss": loss,
            "loss_stop_act": loss_stop,
            "loss_act": loss_act,
            "loss_arm_act": loss_arm_act,
            "loss_gripper_act": loss_gripper_act,
            "loss_depth_act": loss_depth_act,
            "acc_gripper_act": acc_gripper_act,
            "loss_obs": loss_obs,
            "loss_hand_obs": loss_hand_obs,
            "loss_kl": loss_kl,
            "clip_l1": clip_l1,
            "loss_vl_cotrain": loss_vl_cotrain,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Periodic allocator reclaim + RSS telemetry.

        Variable-size native-resolution image batches leave glibc holding
        freed-but-unreturned heap (observed as a steady RSS climb to the
        cgroup limit). Every 200 steps: drop cycles, malloc_trim(0), and
        print RSS so the slope is visible in the job log.
        """
        del outputs, batch
        if batch_idx % 200 != 0 or batch_idx == 0:
            return
        try:
            import ctypes
            import gc

            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        print(f"[mem] step={self.global_step} {line.strip()}",
                              flush=True)
                        break
        except Exception:
            pass

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        # Lightning's FSDPPrecision rejects gradient_clip_algorithm='norm';
        # under FSDP the wrapped root module clips shard-aware instead.
        from lightning.pytorch.strategies import FSDPStrategy
        if isinstance(self.trainer.strategy, FSDPStrategy):
            if gradient_clip_val:
                self.trainer.strategy.model.clip_grad_norm_(gradient_clip_val)
            return
        self.clip_gradients(optimizer, gradient_clip_val,
                            gradient_clip_algorithm)

    def on_load_checkpoint(self, checkpoint):
        """Defensive resume: if the saved optimizer's param-group sizes don't
        match this process's trainable params, drop optimizer/scheduler state
        and warm-start them — weights, global step, loops, and the wandb run
        all still resume. Prevents a hard crash in restore_optimizers."""
        fresh = [len(g["params"]) for g in self.get_grouped_params(self.model)]
        saved_states = checkpoint.get("optimizer_states") or []
        saved = [[len(g["params"]) for g in s.get("param_groups", [])]
                 for s in saved_states]
        if saved and saved != [fresh]:
            print(f"[resume] optimizer group mismatch (saved {saved} vs fresh "
                  f"{[fresh]}); dropping optimizer moments (warm-start) but "
                  "KEEPING scheduler state so the LR stays at the resumed "
                  "step's value (weights + step + loops + wandb continue).",
                  flush=True)
            checkpoint["optimizer_states"] = []
            # Scheduler state is param-group-shape-free (counters + base_lrs);
            # keep it unless its group count disagrees with the live optimizer.
            for sched in checkpoint.get("lr_schedulers", []):
                if len(sched.get("base_lrs", [0])) != len(fresh):
                    checkpoint["lr_schedulers"] = []
                    print("[resume] scheduler group count unexpected; "
                          "dropped scheduler state too.", flush=True)
                    break

    def _infer_batch_size(self, batch):
        """Explicit batch size for self.log — Lightning cannot infer it from
        batches without uniformly-shaped tensors (RobotNav VL batches are PIL
        lists; traj batches carry list-of-native-res rgb)."""
        if isinstance(batch, dict):
            if batch.get("data_source") == "robotnav_mixed":
                n = 0
                for part in (batch.get("traj"), batch.get("vl")):
                    if part:
                        n += self._infer_batch_size(part) or 0
                return n or None
            for key in ("rgb", "vl_user", "text", "action_chunck"):
                v = batch.get(key)
                if v is not None and hasattr(v, "__len__"):
                    return len(v)
        return None

    def _log_output(self, output, phase, prog_bar_set=None, dataset=None, **kwargs):
        prog_bar_set = prog_bar_set or set()
        for key, value in output.items():
            if value is None:
                continue
            log_name = f"{phase}_{key}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"
            log_value = self._scalar_for_log(value)
            self.log(log_name, log_value, prog_bar=(key in prog_bar_set), **kwargs)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value

    def _scalar_for_log(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean()
        return float(value)

    def _frame_to_pil(self, frame: torch.Tensor):
        from torchvision.transforms.functional import to_pil_image

        frame = frame.cpu()
        if frame.dtype.is_floating_point:
            frame = frame.clamp(0, 255).to(torch.uint8)
        else:
            frame = frame.to(torch.uint8)
        return to_pil_image(frame)

    def _frame_tags(self, batch, seq_len):
        """Per-(sample, slot) natural-language temporal tags, or None.

        Only used with act_head.history_type == "pre", where all window slots
        are packed into ONE sequence and the model would otherwise have no way
        to tell which frame is which. The tag states the frame's distance from
        the present in steps, which also makes an irregular history stride
        explicit (frames are not evenly spaced).
        """
        ah = self.configs.get("act_head") or {}
        if ah.get("history_type", "post") != "pre":
            return None
        if not ah.get("language_temporal_tags", True):
            return None
        offs = batch.get("frame_offsets")
        n = len(batch["text"])
        out = []
        for i in range(n):
            row = []
            for j in range(seq_len):
                d = (int(offs[i][j][0].item()) if offs is not None
                     else seq_len - 1 - j)
                row.append("Current view. " if d <= 0
                           else f"View from {d} step{'' if d == 1 else 's'} ago. ")
            out.append(row)
        return out

    def _build_language_inputs(self, batch, rgb):
        seq_len = self.configs["window_size"]

        if isinstance(batch["text"], list) and isinstance(batch["text"][0], str):
            hand_rgb = batch.get("hand_rgb")
            images_per_sample = 1
            if self.use_hand_rgb:
                if hand_rgb is None:
                    raise ValueError(
                        "use_hand_rgb=True but batch has no hand_rgb. "
                        "Set train_dataset.load_wrist=true."
                    )
                images_per_sample = 2

            image_inputs = []
            texts = []
            # len() == shape[0] for tensors; also supports list-of-tensor rgb
            # (RobotNav keeps native per-episode resolutions, so no stacking).
            # history_type="pre": temporal order is communicated with
            # natural-language tags on each frame's text (paper's approach --
            # no architectural change, no positional embeddings).
            tags = self._frame_tags(batch, seq_len)
            for i in range(len(rgb)):
                for j in range(seq_len):
                    image_inputs.append(self._frame_to_pil(rgb[i][j]))
                    if images_per_sample == 2:
                        image_inputs.append(self._frame_to_pil(hand_rgb[i][j]))
                    pre = tags[i][j] if tags is not None else ""
                    texts.append(pre + batch["text"][i])

            image_inputs = self.model.process_vision_info(image_inputs)
            if hasattr(self.model, "build_processor_inputs"):
                inputs = self.model.build_processor_inputs(
                    texts, image_inputs, images_per_sample=images_per_sample
                )
            else:
                if images_per_sample != 1:
                    raise NotImplementedError(
                        "Multi-image batches require model.build_processor_inputs."
                    )
                if hasattr(self.model, "tokenizer"):
                    self.model.tokenizer.padding_side = "right"
                inputs = self.model.processor(
                    text=texts,
                    images=image_inputs,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = dict(inputs)
            inputs = self._to_device(inputs)
            return inputs, inputs["attention_mask"], seq_len

        if isinstance(batch["text"], torch.Tensor):
            language = batch["text"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            return language, text_mask, seq_len

        if isinstance(batch["text"], dict) and "attention_mask" in batch["text"]:
            inputs = self._to_device(batch["text"])
            return inputs, inputs["attention_mask"], seq_len

        raise TypeError(f"Unsupported batch['text'] type: {type(batch['text'])}")

    def _process_batch(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        rgb = batch["rgb"]
        if isinstance(rgb, list):
            # List-form rgb (native/per-frame resolutions) is only ever
            # converted to CPU PIL for the processor — no device move needed,
            # and elements may themselves be lists of per-frame tensors.
            pass
        else:
            rgb = rgb.to(self.device)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            assert rgb.ndim == 5

        language, text_mask, seq_len = self._build_language_inputs(batch, rgb)

        action = batch.get("action")
        if action is not None:
            action = action.to(self.device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        hand_rgb = batch.get("hand_rgb")
        if self.use_hand_rgb and hand_rgb is not None:
            hand_rgb = hand_rgb.to(self.device)
        else:
            hand_rgb = None

        depth = batch.get("depth")
        # use_depth: input conditioning; predict_depth: aux depth-map head (needs GT too).
        need_depth = bool(
            self.configs.get("use_depth", False) or self.configs.get("predict_depth", False)
        )
        if need_depth and depth is not None:
            depth = depth.to(self.device)
        else:
            depth = None

        arm_action_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get("action_chunck")
        if action_chunck is not None:
            action_chunck = action_chunck.to(self.device)
            if action_chunck.shape[-1] == 7:
                arm_action_chunck = action_chunck[..., :6]
                # Collater binarizes gripper to {0, 1} (closed/open); do not remap again.
                gripper_action_chunck = action_chunck[..., -1]
            else:
                arm_action_chunck = action_chunck

        if isinstance(rgb, torch.Tensor):
            rgb = rgb[:, :seq_len]
            if hand_rgb is not None:
                hand_rgb = hand_rgb[:, :seq_len]
            if depth is not None and depth.ndim >= 5:
                depth = depth[:, :seq_len]

        chunck_mask = batch.get("chunck_mask")
        if chunck_mask is not None:
            chunck_mask = chunck_mask.to(self.device)
        stop_label = batch.get("stop_label")
        if stop_label is not None:
            stop_label = stop_label.to(self.device)

        rel_state = batch.get("rel_state")
        if rel_state is not None:
            rel_state = rel_state.to(self.device)

        frame_offsets = batch.get("frame_offsets")
        if frame_offsets is not None:
            frame_offsets = frame_offsets.to(self.device)

        return {
            "frame_offsets": frame_offsets,
            "rgb": rgb,
            "hand_rgb": hand_rgb,
            "depth": depth,
            "attention_mask": attention_mask,
            "language": language,
            "text_mask": text_mask,
            "arm_action_chunck": arm_action_chunck,
            "gripper_action_chunck": gripper_action_chunck,
            "chunck_mask": chunck_mask,
            "stop_label": stop_label,
            "raw_text": batch.get("raw_text"),
            "rel_state": rel_state,
            "data_source": batch.get("data_source", "calvin_action"),
        }

    def _forward_batch(self, batch, mode="train"):
        inputs = self._process_batch(batch)
        return self.model.forward(
            inputs["rgb"],
            inputs["language"],
            attention_mask=inputs["text_mask"],
            action_labels=(inputs["arm_action_chunck"], inputs["gripper_action_chunck"]),
            action_mask=inputs["chunck_mask"],
            stop_label=inputs.get("stop_label"),
            vision_gripper=inputs["hand_rgb"],
            raw_text=inputs["raw_text"],
            rel_state=inputs["rel_state"],
            depth=inputs["depth"],
            data_source=inputs["data_source"],
            frame_offsets=inputs.get("frame_offsets"),
            mode=mode,
        )

    def _collect_metrics(self, prediction, output):
        """Merge raw prediction keys into the logging dict (VLM4VLA-style)."""
        for key, value in prediction.items():
            if key not in output and value is not None:
                output[key] = value
        return output

    def training_step(self, batch, batch_idx):
        del batch_idx
        if isinstance(batch, tuple):
            batch = batch[0]
        prediction = self._forward_batch(batch, mode="train")
        output = self._get_loss(prediction)
        output = self._collect_metrics(prediction, output)
        prog_bar_set = {"loss", "loss_arm_act", "loss_gripper_act", "acc_gripper_act"}
        self._log_output(
            output,
            phase="train",
            prog_bar_set=prog_bar_set,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self._infer_batch_size(batch),
        )
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx
        if isinstance(batch, tuple):
            batch = batch[0]
        with torch.no_grad():
            prediction = self._forward_batch(batch, mode="val")
            output = self._get_loss(prediction)
            output = self._collect_metrics(prediction, output)

        dataset = None
        if self.val_set_names is not None:
            dataset = self.val_set_names[dataloader_idx]

        prog_bar_set = {"loss", "loss_arm_act", "loss_gripper_act", "acc_gripper_act"}
        self._log_output(
            output,
            phase="val",
            prog_bar_set=prog_bar_set,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            dataset=dataset,
            batch_size=self._infer_batch_size(batch),
        )
        return output["loss"]

    def inference_step(self, batch):
        with torch.no_grad():
            inputs = self._process_batch(batch)
            return self.model.inference(
                inputs["rgb"],
                inputs["language"],
                attention_mask=inputs["text_mask"],
                action_labels=(inputs["arm_action_chunck"], inputs["gripper_action_chunck"]),
                action_mask=inputs["chunck_mask"],
                stop_label=inputs.get("stop_label"),
                vision_gripper=inputs["hand_rgb"],
                raw_text=inputs["raw_text"],
                rel_state=inputs["rel_state"],
                depth=inputs.get("depth"),
            )

    def get_grouped_params(self, model):
        """One param group by default; with ``head_learning_rate`` set, the
        action pathway (act_head + ActionQuery token) gets its own group and
        peak LR (paper: backbone 2e-5, action head ~1e-4). The LR schedule
        scales each group's own peak multiplicatively."""
        wd = self.configs["weight_decay"]
        head_lr = float(self.configs.get("head_learning_rate", 0) or 0)
        if not head_lr:
            return [{
                "params": [p for _, p in model.named_parameters() if p.requires_grad],
                "weight_decay": wd,
            }]
        head, backbone = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (head if ("act_head" in name or "action_token" in name)
             else backbone).append(p)
        return [
            {"params": backbone, "weight_decay": wd},
            {"params": head, "weight_decay": wd, "lr": head_lr},
        ]
