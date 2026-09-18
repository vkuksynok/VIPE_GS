#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
if [[ $# -lt 1 || $# -gt 2 ]]; then
  die "Usage: $0 CAMERA_PATH.json [config.yml]"
fi

CAMERA_PATH="$1"
require_file "${CAMERA_PATH}"
if [[ $# -eq 2 ]]; then
  SPLAT_CONFIG="$2"
fi
CONFIG="$(find_latest_splat_config)"
require_file "${CONFIG}"

use_nerfstudio_env
NS_RENDER="${NERFSTUDIO_CONDA_PREFIX}/bin/ns-render"
[[ -x "${NS_RENDER}" ]] || die "ns-render not found: run scripts/setup_splatfacto.sh first"
require_command ffprobe

mkdir -p "${RENDER_DIR}"
OUTPUT="${RENDER_DIR}/${VIPE_SEQUENCE}-camera-path.mp4"
LOG_FILE="${LOG_DIR}/render-${VIPE_SEQUENCE}.log"

log "Rendering camera path to ${OUTPUT}"
"${NS_RENDER}" camera-path \
  --load-config "${CONFIG}" \
  --camera-path-filename "${CAMERA_PATH}" \
  --output-path "${OUTPUT}" \
  2>&1 | tee "${LOG_FILE}"

FRAMES="$(ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames \
  -of default=nokey=1:noprint_wrappers=1 "${OUTPUT}")"
[[ "${FRAMES}" =~ ^[0-9]+$ && "${FRAMES}" -gt 0 ]] || die "Rendered video has no frames"
log "Camera-path render complete: ${OUTPUT} (${FRAMES} frames)"
