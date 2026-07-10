#!/usr/bin/env bash
# Export conda env specs into envs/ for GitHub (small, portable).
# Run on the cluster login node:
#   bash scripts/export_conda_envs.sh
#
# Recreate elsewhere:
#   conda env create -f envs/lfm4vla-libero.yml
#   bash scripts/install_libero_rlds_deps.sh   # for LIBERO TF/protobuf pins

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/envs"
mkdir -p "$OUT"

export_env() {
  local name="$1"
  if ! conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "[skip] env not found: $name"
    return
  fi
  echo "[export] $name"
  conda run -n "$name" conda env export --from-history > "$OUT/${name}.yml"
  conda run -n "$name" pip freeze > "$OUT/${name}-pip-freeze.txt"
}

for env in lfm4vla lfm4vla-libero calvin_eval env_isaaclab; do
  export_env "$env"
done

echo "Wrote env specs to $OUT/ (commit these to GitHub; do not commit runs/ or *.tar.gz)"
