"""Load HE MuJoCo proxy tasks from ``tasks.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from eval.humanoid.g1_env import TaskSpec

_TASKS_YAML = Path(__file__).resolve().parent / "tasks.yaml"


def load_tasks(path: Optional[Path] = None) -> Dict[str, TaskSpec]:
    path = Path(path) if path else _TASKS_YAML
    with open(path) as f:
        raw = yaml.safe_load(f)
    out: Dict[str, TaskSpec] = {}
    for item in raw["tasks"]:
        spec = TaskSpec(
            name=item["name"],
            suite=item["suite"],
            instruction=item["instruction"],
            max_steps=int(item.get("max_steps", 400)),
            objects=dict(item.get("objects") or {}),
            success_type=item["success_type"],
            success_params=dict(item.get("success_params") or {}),
        )
        out[spec.name] = spec
    return out


def filter_suite(tasks: Dict[str, TaskSpec], suite: str) -> List[TaskSpec]:
    """suite: ``in`` | ``ood`` | ``all``."""
    if suite == "all":
        return list(tasks.values())
    if suite not in ("in", "ood"):
        raise ValueError(f"suite must be in|ood|all, got {suite}")
    return [t for t in tasks.values() if t.suite == suite]
