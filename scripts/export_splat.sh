#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
use_nerfstudio_env
ensure_open3d_libraries
NS_EXPORT="${NERFSTUDIO_CONDA_PREFIX}/bin/ns-export"
[[ -x "${NS_EXPORT}" ]] || die "ns-export not found: run scripts/setup_splatfacto.sh first"

if [[ $# -gt 1 ]]; then
  die "Usage: $0 [config.yml]"
fi
if [[ $# -eq 1 ]]; then
  SPLAT_CONFIG="$1"
fi
CONFIG="$(find_latest_splat_config)"
require_file "${CONFIG}"

mkdir -p "${EXPORT_DIR}"
LOG_FILE="${LOG_DIR}/export-splat-${VIPE_SEQUENCE}.log"
log "Exporting Gaussian splat from ${CONFIG}"
"${NS_EXPORT}" gaussian-splat \
  --load-config "${CONFIG}" \
  --output-dir "${EXPORT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

PLY="$(find "${EXPORT_DIR}" -maxdepth 2 -type f -name '*.ply' -print -quit)"
[[ -n "${PLY}" && -s "${PLY}" ]] || die "No non-empty Gaussian PLY was exported"
log "Gaussian export complete: ${PLY}"
