"""G1 + Dex3 joint layout for Humanoid Everyday 28-D actions.

HE parquet ``action`` is documented as 14 arm (``sol_q``) + 14 Dex3 hand joints.
This module defines the canonical MuJoCo joint-name order used by the eval env.

Verify against a downloaded episode with::

    python scripts/verify_he_g1_action_map.py --data_root_dir <he_root>
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 7-DoF arm (Unitree / Menagerie order), left then right.
LEFT_ARM_JOINTS: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS: Tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Dex3-1: thumb (3) + index (2) + middle (2) — Unitree SDK order.
LEFT_HAND_JOINTS: Tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
RIGHT_HAND_JOINTS: Tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

# HE 28-D: [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
POLICY_JOINTS: Tuple[str, ...] = (
    LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
)
assert len(POLICY_JOINTS) == 28

# Held by PD at stand keyframe (not driven by the VLA).
LEG_WAIST_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

ACTION_DIM = 28


def default_robot_xml() -> Path:
    """Path to Menagerie ``g1_with_hands.xml`` (fetched by fetch script)."""
    candidates = [
        _REPO_ROOT / "eval/humanoid/assets/robot/unitree_g1/g1_with_hands.xml",
        _REPO_ROOT / "third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "G1+hands MJCF not found. Run: bash scripts/fetch_unitree_g1_mujoco.sh\n"
        f"Looked in: {candidates}"
    )


def menagerie_unitree_g1_dir() -> Path:
    return default_robot_xml().parent


def policy_joint_names() -> List[str]:
    return list(POLICY_JOINTS)


def held_joint_names() -> List[str]:
    return list(LEG_WAIST_JOINTS)


def split_action(action28: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return (arm14, hand14) from a 28-D vector."""
    a = list(action28)
    if len(a) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}-D action, got {len(a)}")
    return a[:14], a[14:]
