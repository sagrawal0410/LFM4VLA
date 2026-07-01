# LFM4VLA

Fine-tune [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) for robot action prediction on CALVIN.

## Setup

```bash
pip install -r requirements.txt
# CALVIN loading also requires calvin_agent
export WANDB_API_KEY=...   # if using wandb logger
```

Set `train_dataset.data_dir` and `val_dataset.data_dir` in the config.

## Train

```bash
python main.py --config configs/lfm2.5-vl-1.6b-finetune.json
```

Resume a Lightning checkpoint:

```bash
python main.py --config configs/lfm2.5-vl-1.6b-finetune.json --resume runs/checkpoints/.../last.ckpt
```

## Pipeline

1. **Data** — `DiskCalvinDataset` loads CALVIN at native resolution (~200×200)
2. **Trainer** — builds LFM chat-template processor inputs per batch
3. **Model** — `RoboLFM25VL` fuses vision tokens, runs LFM backbone, predicts actions via `FCDecoder`
4. **Logging** — Lightning logs to wandb (configurable): `train_loss`, `train_loss_arm_act`, `train_loss_gripper_act`, `train_acc_gripper_act`, and `val_*` counterparts
5. **Checkpoints** — best `val_loss` + `last.ckpt` under `runs/checkpoints/<date>/<exp_name>/`
