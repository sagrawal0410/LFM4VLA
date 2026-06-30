import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from data.build_dataset import build_dataset
from train.base_trainer import BaseTrainer
from train.experiment_utils import build_loggers, prepare_experiment
from utils.dist_train import get_rank
from utils.setup_callback import SetupCallback


def _build_strategy(strategy_name):
    if strategy_name in (None, "auto", "ddp"):
        return "auto"
    if strategy_name == "deepspeed_stage_2":
        from lightning.pytorch.strategies import DeepSpeedStrategy

        return DeepSpeedStrategy(stage=2)
    return strategy_name


def experiment(variant):
    variant, exp_name, log_dir, ckpt_dir = prepare_experiment(variant)

    if get_rank() == 0:
        print("--------------- experiment configs ---------------")
        print(f"run: {exp_name}")
        print(f"log_dir: {log_dir}")
        print(f"ckpt_dir: {ckpt_dir}")

    ckpt_path = variant.get("resume") or variant.get("model_load_path")
    if ckpt_path:
        trainer_module = BaseTrainer.from_checkpoint(ckpt_path, variant.get("model_load_source", "torch"), variant)
    else:
        trainer_module = BaseTrainer(variant)

    train_dataset = build_dataset(variant["train_dataset"], variant, trainer_module.model)
    val_dataset = build_dataset(variant["val_dataset"], variant, trainer_module.model)

    loader_kwargs = dict(
        batch_size=variant["batch_size"],
        num_workers=variant.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collater,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=val_dataset.collater,
        **loader_kwargs,
    )

    trainer_cfg = variant["trainer"]
    callbacks = [
        SetupCallback(log_dir, ckpt_dir, variant, exp_name),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:02d}-{step}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", "auto"),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        strategy=_build_strategy(trainer_cfg.get("strategy", "auto")),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        max_epochs=trainer_cfg["max_epochs"],
        max_steps=trainer_cfg.get("max_steps", -1),
        logger=build_loggers(variant, exp_name, log_dir),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        use_distributed_sampler=trainer_cfg.get("use_distributed_sampler", True),
        callbacks=callbacks,
    )

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=variant.get("resume"))
