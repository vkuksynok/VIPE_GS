#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
ensure_conda
require_command git

log "Checking out VIPE ${VIPE_COMMIT}"
clone_pinned_repo "${VIPE_REPOSITORY}" "${VIPE_COMMIT}" "${VIPE_SOURCE_DIR}"

if [[ ! -x "${VIPE_CONDA_PREFIX}/bin/uv" ]]; then
  log "Creating VIPE native/CUDA environment at ${VIPE_CONDA_PREFIX}"
  conda env create --yes --prefix "${VIPE_CONDA_PREFIX}" --file "${VIPE_SOURCE_DIR}/envs/cu128.yml"
else
  log "VIPE Conda environment already exists"
fi

# The image's system interpreter keeps pyconfig.h under /usr/include/<triplet>/,
# which conda's compiler sysroot does not search, so building the CUDA extension
# against it fails. A uv-managed interpreter carries self-contained headers.
export UV_PYTHON_PREFERENCE="only-managed"

log "Installing VIPE from its frozen uv lock on a uv-managed Python ${VIPE_PYTHON_VERSION}"
UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
  conda run --prefix "${VIPE_CONDA_PREFIX}" \
  uv sync --frozen --project "${VIPE_SOURCE_DIR}" --python "${VIPE_PYTHON_VERSION}"

log "Verifying VIPE installation"
UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
  conda run --prefix "${VIPE_CONDA_PREFIX}" \
  uv run --project "${VIPE_SOURCE_DIR}" python -c \
  'import sys, torch, vipe; print("python", sys.version.split()[0]); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("vipe", vipe.__file__)'

UV_PROJECT_ENVIRONMENT="${VIPE_UV_ENVIRONMENT}" \
  conda run --prefix "${VIPE_CONDA_PREFIX}" \
  uv export --frozen --project "${VIPE_SOURCE_DIR}" \
  > "${ENV_REPORT_DIR}/vipe-requirements.lock.txt"

git -C "${VIPE_SOURCE_DIR}" rev-parse HEAD > "${ENV_REPORT_DIR}/vipe-git-sha.txt"
log "VIPE setup complete"
