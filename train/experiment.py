import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset

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


def _build_model_checkpoints(ckpt_dir: str, trainer_cfg: dict) -> list[ModelCheckpoint]:
    """Step-based and (optional) best-val checkpoints as separate callbacks.

    Lightning couples ``every_n_train_steps`` and ``monitor`` awkwardly: a single
    callback with both forces ``every_n_epochs=0`` (disabling the val-end save) and
    skips/defers step saves whenever the monitored metric is absent. Splitting them
    keeps step checkpoints unconditional and best-val checkpoints independent.
    """
    callbacks: list[ModelCheckpoint] = []

    every_n = trainer_cfg.get("checkpoint_every_n_train_steps")
    if every_n:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="step-{epoch:02d}-{step}",
                every_n_train_steps=every_n,
                save_top_k=trainer_cfg.get("checkpoint_step_save_top_k", -1),
                save_last=trainer_cfg.get("checkpoint_save_last", True),
            )
        )

    monitor = trainer_cfg.get("checkpoint_monitor", "val_loss")
    if monitor:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-{epoch:02d}-{step}",
                monitor=monitor,
                mode=trainer_cfg.get("checkpoint_monitor_mode", "min"),
                save_top_k=trainer_cfg.get("checkpoint_save_top_k", 1),
                save_last=False,
            )
        )

    if not callbacks:
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, save_last=True))

    return callbacks


def _build_loader(dataset, variant, train: bool) -> DataLoader:
    """Build a DataLoader, adapting to map-style (CALVIN) vs iterable (LIBERO RLDS).

    Iterable RLDS streams handle their own shuffling (TF shuffle buffer) and cannot
    use ``shuffle`` or a ``sampler``. They must run with ``num_workers=0`` (in the
    main process): TensorFlow does not survive ``os.fork()``, so streaming the RLDS
    pipeline inside a forked DataLoader worker deadlocks (no batch is ever yielded).
    """
    if isinstance(dataset, IterableDataset):
        return DataLoader(
            dataset,
            batch_size=variant["batch_size"],
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=dataset.collater,
            drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=variant["batch_size"],
        num_workers=variant.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        shuffle=train,
        collate_fn=dataset.collater,
        drop_last=train,
    )


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

    train_loader = _build_loader(train_dataset, variant, train=True)
    val_loader = _build_loader(val_dataset, variant, train=False)

    trainer_cfg = variant["trainer"]
    callbacks = [
        SetupCallback(log_dir, ckpt_dir, variant, exp_name),
        LearningRateMonitor(logging_interval="step"),
        *_build_model_checkpoints(ckpt_dir, trainer_cfg),
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
        use_distributed_sampler=(
            trainer_cfg.get("use_distributed_sampler", True)
            and not isinstance(train_dataset, IterableDataset)
        ),
        callbacks=callbacks,
    )

    trainer.fit(trainer_module, train_loader, val_loader, ckpt_path=variant.get("resume"))
