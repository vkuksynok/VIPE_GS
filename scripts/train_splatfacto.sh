#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
require_dir "${SPLAT_DATASET_DIR}/images"
require_dir "${SPLAT_DATASET_DIR}/sparse/0"
require_file "${SPLAT_DATASET_DIR}/sparse/0/cameras.txt"
require_file "${SPLAT_DATASET_DIR}/sparse/0/images.txt"
require_file "${SPLAT_DATASET_DIR}/sparse/0/points3D.txt"

use_nerfstudio_env
NS_TRAIN="${NERFSTUDIO_CONDA_PREFIX}/bin/ns-train"
[[ -x "${NS_TRAIN}" ]] || die "ns-train not found: run scripts/setup_splatfacto.sh first"

MAX_ITERATIONS="${SPLATFACTO_MAX_ITERATIONS:-30000}"
DOWNSCALE="${SPLATFACTO_DOWNSCALE_FACTOR:-1}"
# Hold every Nth frame out of Gaussian optimization so renders can be compared
# against images the model never fitted.
EVAL_INTERVAL="${SPLATFACTO_EVAL_INTERVAL:-8}"
LOG_FILE="${LOG_DIR}/splatfacto-${VIPE_SEQUENCE}.log"

log "Training Splatfacto for ${MAX_ITERATIONS} iterations"
"${NS_TRAIN}" splatfacto \
  --output-dir "${SPLAT_OUTPUT_DIR}" \
  --experiment-name "${VIPE_SEQUENCE}" \
  --max-num-iterations "${MAX_ITERATIONS}" \
  --machine.seed "${PIPELINE_SEED}" \
  --viewer.quit-on-train-completion True \
  colmap \
  --data "${SPLAT_DATASET_DIR}" \
  --images-path images \
  --colmap-path sparse/0 \
  --downscale-factor "${DOWNSCALE}" \
  --eval-mode interval \
  --eval-interval "${EVAL_INTERVAL}" \
  2>&1 | tee "${LOG_FILE}"

CONFIG="$(find_latest_splat_config)"
require_file "${CONFIG}"
log "Splatfacto training complete: ${CONFIG}"
