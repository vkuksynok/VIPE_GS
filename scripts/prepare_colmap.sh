#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
require_dir "${VIPE_OUTPUT_DIR}"
require_dir "${VIPE_SOURCE_DIR}"
ensure_conda

mkdir -p "${COLMAP_RAW_DIR}"
LOG_FILE="${LOG_DIR}/vipe-to-colmap-${VIPE_SEQUENCE}.log"

log "Converting VIPE artifacts to COLMAP text"
(
  cd "${VIPE_SOURCE_DIR}"
  UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
    conda run --prefix "${VIPE_CONDA_PREFIX}" \
    uv run python scripts/vipe_to_colmap.py \
      "${VIPE_OUTPUT_DIR}" \
      --sequence "${VIPE_SEQUENCE}" \
      --use_slam_map \
      --output "${COLMAP_RAW_DIR}"
) 2>&1 | tee "${LOG_FILE}"

RAW_SEQUENCE_DIR="${COLMAP_RAW_DIR}/${VIPE_SEQUENCE}"
require_dir "${RAW_SEQUENCE_DIR}"

log "Normalizing COLMAP paths for Nerfstudio"
python3 "${SCRIPT_DIR}/prepare_colmap.py" \
  --source "${RAW_SEQUENCE_DIR}" \
  --output "${SPLAT_DATASET_DIR}"

log "COLMAP dataset complete: ${SPLAT_DATASET_DIR}"
