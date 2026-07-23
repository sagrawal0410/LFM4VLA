"""Periodic host-memory logging to diagnose RSS growth / cgroup OOM kills.

Logs both the Slurm job cgroup usage (covers DataLoader worker processes too)
and this process's own RSS, so leaks can be attributed to the main process vs.
workers and correlated with training steps in wandb/CSV logs.
"""

from pathlib import Path

from lightning.pytorch.callbacks import Callback


def _cgroup_memory_gib():
    """Current memory usage of our cgroup (what the Slurm OOM killer enforces)."""
    try:
        rel = Path("/proc/self/cgroup").read_text().splitlines()[0].split(":", 2)[2].strip()
        for name in ("memory.current", "memory/memory.usage_in_bytes"):
            p = Path("/sys/fs/cgroup") / rel.lstrip("/") / name
            if p.exists():
                return int(p.read_text()) / 2**30
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_rss_gib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20  # kB -> GiB
    except (OSError, ValueError):
        pass
    return None


class MemoryMonitor(Callback):
    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps

    def _report(self, trainer, pl_module, where: str):
        rss = _process_rss_gib()
        cg = _cgroup_memory_gib()
        parts = []
        metrics = {}
        if rss is not None:
            parts.append(f"rss={rss:.1f}GiB")
            metrics["mem/rss_gib"] = rss
        if cg is not None:
            parts.append(f"cgroup={cg:.1f}GiB")
            metrics["mem/cgroup_gib"] = cg
        if not parts:
            return
        print(f"[mem] step={trainer.global_step} {where} " + " ".join(parts), flush=True)
        if trainer.logger is not None:
            pl_module.log_dict(metrics, on_step=True, on_epoch=False, rank_zero_only=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0:
            self._report(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._report(trainer, pl_module, "val_end")

    def on_train_start(self, trainer, pl_module):
        self._report(trainer, pl_module, "train_start")
