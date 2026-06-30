import json
import os

from lightning.pytorch.callbacks import Callback


class SetupCallback(Callback):
    """Save run config to log_dir on train start (rank 0 only)."""

    def __init__(self, logdir, ckptdir, config, run_name):
        super().__init__()
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.config = config
        self.run_name = run_name

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        config_path = os.path.join(self.logdir, f"{self.run_name}-config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved run config to {config_path}")
