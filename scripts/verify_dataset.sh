#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
require_command python3
require_file "${RAW_DATASET}"

RAW_FRAME_DIR="${RAW_FRAME_DIR:-${DATA_ROOT}/raw/$(basename -- "${RAW_DATASET}" .zip)}"
REPORT="${ENV_REPORT_DIR}/dataset-$(date -u +'%Y%m%dT%H%M%SZ').json"
LOG_FILE="${LOG_DIR}/verify-dataset.log"

ARCHIVE_ARGS=(--archive "${RAW_DATASET}")
if [[ -n "${RAW_DATASET_FULL:-}" && -f "${RAW_DATASET_FULL}" ]]; then
  ARCHIVE_ARGS+=(--archive "${RAW_DATASET_FULL}")
fi

if [[ -d "${RAW_FRAME_DIR}" ]]; then
  log "Frame directory already present, verifying archives only: ${RAW_FRAME_DIR}"
  EXTRACT_ARGS=()
else
  EXTRACT_ARGS=(--extract "${RAW_DATASET}" --extract-to "${RAW_FRAME_DIR}")
fi

log "Verifying dataset archives on ${WORKSPACE_ROOT}"
python3 "${SCRIPT_DIR}/verify_dataset.py" \
  "${ARCHIVE_ARGS[@]}" \
  "${EXTRACT_ARGS[@]}" \
  --expected-count "${DATASET_EXPECTED_FRAMES:-126}" \
  --report "${REPORT}" 2>&1 | tee "${LOG_FILE}"

log "Storage after verification"
{
  printf 'workspace_usage_bytes=%s\n' "$(du -sb "${WORKSPACE_ROOT}" | cut -f1)"
  printf 'datasets_usage_bytes=%s\n' "$(du -sb "$(dirname -- "${RAW_DATASET}")" | cut -f1)"
  df -h /workspace
} | tee -a "${LOG_FILE}"

log "Dataset verification complete: ${REPORT}"
