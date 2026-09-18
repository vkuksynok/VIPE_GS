#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

MODE="${1:-host}"
[[ "${MODE}" == "host" || "${MODE}" == "full" ]] || die "Usage: $0 [host|full]"

ensure_runtime_dirs
require_command nvidia-smi
require_command python3

REPORT="${ENV_REPORT_DIR}/environment-$(date -u +'%Y%m%dT%H%M%SZ').txt"
{
  printf 'checked_at_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'kernel=%s\n' "$(uname -srvmo)"
  printf 'runpod_template=%s\n' "${RUNPOD_TEMPLATE_ID}"
  printf 'runpod_image_expected=%s\n' "${RUNPOD_IMAGE}"
  printf 'network_volume_expected=%s\n' "${RUNPOD_NETWORK_VOLUME_ID}"
  printf 'workspace_root=%s\n' "${WORKSPACE_ROOT}"
  printf '\n[nvidia-smi]\n'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  printf '\n[filesystem]\n'
  df -h /workspace
  printf '\n[python]\n'
  python3 --version
} | tee "${REPORT}"

[[ -d /workspace && -w /workspace ]] || die "/workspace is not a writable mount"

if [[ "${MODE}" == "full" ]]; then
  ensure_conda
  require_file "${VIPE_CONDA_PREFIX}/bin/uv"
  require_file "${NERFSTUDIO_CONDA_PREFIX}/bin/python"
  [[ "$(git -C "${VIPE_SOURCE_DIR}" rev-parse HEAD)" == "${VIPE_COMMIT}" ]] || die "VIPE SHA mismatch"
  [[ "$(git -C "${NERFSTUDIO_SOURCE_DIR}" rev-parse HEAD)" == "${NERFSTUDIO_COMMIT}" ]] || die "Nerfstudio SHA mismatch"

  UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
    conda run --prefix "${VIPE_CONDA_PREFIX}" \
    uv run --project "${VIPE_SOURCE_DIR}" python -c \
    'import torch; assert torch.cuda.is_available(); print("vipe torch", torch.__version__)' \
    | tee -a "${REPORT}"
  "${NERFSTUDIO_CONDA_PREFIX}/bin/python" -c \
    'import gsplat, nerfstudio, torch; assert torch.cuda.is_available(); print("nerfstudio torch", torch.__version__); print("gsplat", gsplat.__version__)' \
    | tee -a "${REPORT}"
fi

log "Environment verification complete: ${REPORT}"
