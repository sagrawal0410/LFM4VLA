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
        )
        model.train()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Trainable Model Parameters: {total_params / 1e6:.2f}M")
        return model

    def _parse_dataset_name(self, dataset_config):
        dataset_path = dataset_config["data_dir"]
        for name in ("calvin", "bridge", "libero"):
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
    def from_checkpoint(cls, ckpt_path=None, ckpt_source="torch", configs=None):
        if ckpt_path is None:
            return cls(configs)

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
        iter_per_epoch = num_training_batches / self.trainer.max_epochs
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
        loss_obs = prediction.get("loss_obs_fwd")
        loss_hand_obs = prediction.get("loss_hand_obs_fwd")
        acc_gripper_act = prediction.get("acc_gripper_act")
        loss_cap = prediction.get("loss_cap")
        loss_kl = prediction.get("loss_kl")
        loss_vl_cotrain = prediction.get("loss_vl_cotrain")
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

        return {
            "loss": loss,
            "loss_act": loss_act,
            "loss_arm_act": loss_arm_act,
            "loss_gripper_act": loss_gripper_act,
            "acc_gripper_act": acc_gripper_act,
            "loss_obs": loss_obs,
            "loss_hand_obs": loss_hand_obs,
            "loss_kl": loss_kl,
            "clip_l1": clip_l1,
            "loss_vl_cotrain": loss_vl_cotrain,
        }

    def _log_output(self, output, phase, prog_bar_set=None, dataset=None, **kwargs):
        prog_bar_set = prog_bar_set or set()
        for key, value in output.items():
            if value is None:
                continue
            log_name = f"{phase}_{key}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"
            self.log(log_name, value, prog_bar=(key in prog_bar_set), **kwargs)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value

    def _build_language_inputs(self, batch, rgb):
        seq_len = self.configs["window_size"]

        if isinstance(batch["text"], list) and isinstance(batch["text"][0], str):
            assert not self.use_hand_rgb
            from torchvision.transforms.functional import to_pil_image

            image_inputs = []
            texts = []
            for i in range(rgb.shape[0]):
                for j in range(seq_len):
                    frame = rgb[i][j].cpu()
                    if frame.dtype.is_floating_point:
                        frame = frame.clamp(0, 255).to(torch.uint8)
                    else:
                        frame = frame.to(torch.uint8)
                    image_inputs.append(to_pil_image(frame))
                    texts.append(batch["text"][i])

            image_inputs = self.model.process_vision_info(image_inputs)
            if hasattr(self.model, "build_processor_inputs"):
                inputs = self.model.build_processor_inputs(texts, image_inputs)
            else:
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
            language = batch["text"].cuda()
            text_mask = batch["text_mask"].cuda()
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
            rgb = [x.cuda() for x in rgb]
        else:
            rgb = rgb.cuda()
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            assert rgb.ndim == 5

        language, text_mask, seq_len = self._build_language_inputs(batch, rgb)

        action = batch.get("action")
        if action is not None:
            action = action.cuda()

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()

        hand_rgb = batch.get("hand_rgb")
        if self.use_hand_rgb and hand_rgb is not None:
            hand_rgb = hand_rgb.cuda()
        else:
            hand_rgb = None

        arm_action_chunck = None
        gripper_action_chunck = None
        action_chunck = batch.get("action_chunck")
        if action_chunck is not None:
            action_chunck = action_chunck.cuda()
            if action_chunck.shape[-1] == 7:
                arm_action_chunck = action_chunck[..., :6]
                gripper_action_chunck = action_chunck[..., -1]
            else:
                arm_action_chunck = action_chunck

        if gripper_action_chunck is not None:
                gripper_action_chunck = (gripper_action_chunck + 1.0) / 2

        if isinstance(rgb, torch.Tensor):
            rgb = rgb[:, :seq_len]
            if hand_rgb is not None:
                hand_rgb = hand_rgb[:, :seq_len]

        chunck_mask = batch.get("chunck_mask")
        if chunck_mask is not None:
            chunck_mask = chunck_mask.cuda()

        return {
            "rgb": rgb,
            "hand_rgb": hand_rgb,
            "attention_mask": attention_mask,
            "language": language,
            "text_mask": text_mask,
            "arm_action_chunck": arm_action_chunck,
            "gripper_action_chunck": gripper_action_chunck,
            "chunck_mask": chunck_mask,
            "raw_text": batch.get("raw_text"),
            "rel_state": batch.get("rel_state"),
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
            vision_gripper=inputs["hand_rgb"],
            raw_text=inputs["raw_text"],
            rel_state=inputs["rel_state"],
            data_source=inputs["data_source"],
            mode=mode,
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        if isinstance(batch, tuple):
            batch = batch[0]
        prediction = self._forward_batch(batch, mode="train")
        output = self._get_loss(prediction)
        prog_bar_set = {"loss", "loss_arm_act", "loss_gripper_act", "acc_gripper_act"}
        self._log_output(output, phase="train", prog_bar_set=prog_bar_set, on_step=True, on_epoch=False)
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx
        if isinstance(batch, tuple):
            batch = batch[0]
        with torch.no_grad():
            prediction = self._forward_batch(batch, mode="train")
            output = self._get_loss(prediction)

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
        )

    def inference_step(self, batch):
        with torch.no_grad():
            inputs = self._process_batch(batch)
            return self.model.inference(
                inputs["rgb"],
                inputs["language"],
                attention_mask=inputs["text_mask"],
                action_labels=(inputs["arm_action_chunck"], inputs["gripper_action_chunck"]),
                action_mask=inputs["chunck_mask"],
                vision_gripper=inputs["hand_rgb"],
                raw_text=inputs["raw_text"],
                rel_state=inputs["rel_state"],
            )

    def get_grouped_params(self, model):
        return [{
            "params": [p for _, p in model.named_parameters() if p.requires_grad],
            "weight_decay": self.configs["weight_decay"],
        }]
