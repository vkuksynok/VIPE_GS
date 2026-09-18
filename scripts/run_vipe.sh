#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
require_file "${INPUT_VIDEO}"
require_dir "${VIPE_SOURCE_DIR}"
ensure_conda

mkdir -p "${VIPE_OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/vipe-${VIPE_SEQUENCE}.log"
CONFIG_FILE="${ENV_REPORT_DIR}/vipe-config-${VIPE_SEQUENCE}.yaml"

VIPE_ARGS=(
  pipeline=default
  streams=raw_mp4_stream
  "streams.base_path=${INPUT_VIDEO}"
  streams.frame_skip=1
  "pipeline.output.path=${VIPE_OUTPUT_DIR}"
  pipeline.output.save_artifacts=true
  pipeline.output.save_slam_map=true
)

# --resolve is not usable here: VIPE registers its own `neq` interpolation
# resolver at runtime, so Hydra cannot resolve the config from the CLI.
log "Recording resolved VIPE config to ${CONFIG_FILE}"
(
  cd "${VIPE_SOURCE_DIR}"
  UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
    conda run --prefix "${VIPE_CONDA_PREFIX}" \
    uv run python run.py "${VIPE_ARGS[@]}" --cfg job
) > "${CONFIG_FILE}"

log "Running VIPE on ${INPUT_VIDEO}"
(
  cd "${VIPE_SOURCE_DIR}"
  UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
    conda run --prefix "${VIPE_CONDA_PREFIX}" \
    uv run python run.py "${VIPE_ARGS[@]}"
) 2>&1 | tee "${LOG_FILE}"

[[ -s "${LOG_FILE}" ]] || die "VIPE log was not created"
find "${VIPE_OUTPUT_DIR}" -type f -print -quit | grep -q . || die "VIPE produced no output files"
log "VIPE run complete: ${VIPE_OUTPUT_DIR}"
