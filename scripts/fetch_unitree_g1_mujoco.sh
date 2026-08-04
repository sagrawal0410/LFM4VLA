#!/usr/bin/env bash
# Fetch official Unitree G1 MuJoCo assets for HE closed-loop eval.
#
# Primary robot model (29 DoF body + Dex3 hands):
#   google-deepmind/mujoco_menagerie  →  unitree_g1/g1_with_hands.xml
#   (Unitree-authored kinematics packaged in Menagerie; includes finger DoFs.)
#
# Also pulls unitree_mujoco body-only models for reference / joint-index docs.
#
# Usage:
#   bash scripts/fetch_unitree_g1_mujoco.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
MENAGERIE_DIR="${TP}/mujoco_menagerie"
UNITREE_MJ_DIR="${TP}/unitree_mujoco"
PIN_MENAGERIE_REF="${MENAGERIE_REF:-main}"
PIN_UNITREE_REF="${UNITREE_MUJOCO_REF:-main}"

mkdir -p "$TP"

echo "=== Fetching mujoco_menagerie/unitree_g1 (G1 + Dex3 hands) ==="
if [[ -d "${MENAGERIE_DIR}/.git" ]]; then
  git -C "$MENAGERIE_DIR" fetch --depth 1 origin "$PIN_MENAGERIE_REF"
  git -C "$MENAGERIE_DIR" checkout -q FETCH_HEAD
else
  rm -rf "$MENAGERIE_DIR"
  git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
  git -C "$MENAGERIE_DIR" sparse-checkout set unitree_g1
  git -C "$MENAGERIE_DIR" checkout "$PIN_MENAGERIE_REF"
fi

HANDS_XML="${MENAGERIE_DIR}/unitree_g1/g1_with_hands.xml"
ASSETS_DIR="${MENAGERIE_DIR}/unitree_g1/assets"
if [[ ! -f "$HANDS_XML" ]]; then
  echo "[ERROR] missing ${HANDS_XML}" >&2
  exit 1
fi
if [[ ! -d "$ASSETS_DIR" ]]; then
  echo "[ERROR] missing mesh dir ${ASSETS_DIR}" >&2
  exit 1
fi
echo "[ok] ${HANDS_XML}"
echo "[ok] meshes: $(find "$ASSETS_DIR" -type f | wc -l | tr -d ' ') files"

echo "=== Fetching unitreerobotics/unitree_mujoco (body models + DDS joint docs) ==="
if [[ -d "${UNITREE_MJ_DIR}/.git" ]]; then
  git -C "$UNITREE_MJ_DIR" fetch --depth 1 origin "$PIN_UNITREE_REF"
  git -C "$UNITREE_MJ_DIR" checkout -q FETCH_HEAD
else
  rm -rf "$UNITREE_MJ_DIR"
  git clone --filter=blob:none --sparse \
    https://github.com/unitreerobotics/unitree_mujoco.git "$UNITREE_MJ_DIR"
  git -C "$UNITREE_MJ_DIR" sparse-checkout set unitree_robots/g1
  git -C "$UNITREE_MJ_DIR" checkout "$PIN_UNITREE_REF"
fi

# Symlink used by eval/humanoid scene includes.
LINK_DIR="${ROOT}/eval/humanoid/assets/robot"
mkdir -p "$LINK_DIR"
ln -sfn "${MENAGERIE_DIR}/unitree_g1" "${LINK_DIR}/unitree_g1"
echo "[ok] symlink ${LINK_DIR}/unitree_g1 -> menagerie unitree_g1"

cat <<EOF

Done. Robot MJCF for HE eval:
  ${HANDS_XML}

Resolve at runtime via:
  eval.humanoid.g1_joint_map.default_robot_xml()
EOF
