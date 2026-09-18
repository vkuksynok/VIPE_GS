#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
ensure_conda
require_command git

log "Checking out Nerfstudio ${NERFSTUDIO_COMMIT}"
clone_pinned_repo "${NERFSTUDIO_REPOSITORY}" "${NERFSTUDIO_COMMIT}" "${NERFSTUDIO_SOURCE_DIR}"

if [[ ! -x "${NERFSTUDIO_CONDA_PREFIX}/bin/python" ]]; then
  log "Creating isolated Nerfstudio environment at ${NERFSTUDIO_CONDA_PREFIX}"
  # Channels must precede the package specs, otherwise conda's parser rejects
  # everything that follows -c.
  conda create --yes --prefix "${NERFSTUDIO_CONDA_PREFIX}" \
    -c "nvidia/label/cuda-${NERFSTUDIO_CUDA_VERSION}.0" \
    -c conda-forge \
    "python=${NERFSTUDIO_PYTHON_VERSION}" pip \
    "cuda-toolkit=${NERFSTUDIO_CUDA_VERSION}"
else
  log "Nerfstudio Conda environment already exists"
fi

NS_PYTHON="${NERFSTUDIO_CONDA_PREFIX}/bin/python"
CONSTRAINTS="${PROJECT_ROOT}/config/nerfstudio-constraints.txt"

log "Installing the documented PyTorch/CUDA stack"
TORCH_INDEX="https://download.pytorch.org/whl/cu${NERFSTUDIO_CUDA_VERSION//./}"
"${NS_PYTHON}" -m pip install \
  --extra-index-url "${TORCH_INDEX}" \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

log "Installing pinned Nerfstudio from source"
"${NS_PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
  "${NERFSTUDIO_SOURCE_DIR}"

log "Verifying Splatfacto environment"
"${NS_PYTHON}" -c \
  'import gsplat, nerfstudio, numpy, torch; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("capability", torch.cuda.get_device_capability(0)); print("arch_list", torch.cuda.get_arch_list()); print("numpy", numpy.__version__); print("gsplat", gsplat.__version__); print("nerfstudio", nerfstudio.__file__)'

"${NS_PYTHON}" -m pip freeze --all > "${ENV_REPORT_DIR}/nerfstudio-freeze.txt"
git -C "${NERFSTUDIO_SOURCE_DIR}" rev-parse HEAD > "${ENV_REPORT_DIR}/nerfstudio-git-sha.txt"
log "Nerfstudio setup complete"
