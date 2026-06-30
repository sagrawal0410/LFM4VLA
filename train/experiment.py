import os

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from data.build_dataset import build_dataset
from train.base_trainer import BaseTrainer
from utils.dist_train import get_rank


def _build_strategy(strategy_name):
    if strategy_name in (None, "auto", "ddp"):
        return "auto"
    if strategy_name == "deepspeed_stage_2":
        from lightning.pytorch.strategies import DeepSpeedStrategy

        return DeepSpeedStrategy(stage=2)
    return strategy_name


def experiment(variant):
    if get_rank() == 0:
        print("--------------- experiment configs ---------------")
        print(variant)

    trainer_module = BaseTrainer(variant)

    train_dataset = build_dataset(variant["train_dataset"], variant, trainer_module.model)
    val_dataset = build_dataset(variant["val_dataset"], variant, trainer_module.model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=variant["batch_size"],
        shuffle=True,
        num_workers=variant.get("num_workers", 4),
        collate_fn=train_dataset.collater,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=variant["batch_size"],
        shuffle=False,
        num_workers=variant.get("num_workers", 4),
        collate_fn=val_dataset.collater,
        pin_memory=torch.cuda.is_available(),
    )

    trainer_cfg = variant["trainer"]
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(variant["output_root"], variant["task_name"]),
            filename="{epoch}-{step}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", "auto"),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        strategy=_build_strategy(trainer_cfg.get("strategy", "auto")),
        precision=trainer_cfg.get("precision", "bf16"),
        max_epochs=trainer_cfg["max_epochs"],
        max_steps=trainer_cfg.get("max_steps", -1),
        logger=trainer_cfg.get("logger", True),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        use_distributed_sampler=trainer_cfg.get("use_distributed_sampler", True),
        callbacks=callbacks,
    )

    ckpt_path = variant.get("resume")
    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=ckpt_path)

    if dist.is_initialized():
        dist.destroy_process_group()
