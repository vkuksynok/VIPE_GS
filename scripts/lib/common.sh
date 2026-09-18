#!/usr/bin/env bash

set -Eeuo pipefail

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${COMMON_DIR}/../.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/config/versions.env"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/config/paths.env.example"
if [[ -f "${PROJECT_ROOT}/config/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/config/paths.env"
fi

export PROJECT_ROOT WORKSPACE_ROOT DATA_ROOT THIRD_PARTY_ROOT ENV_ROOT CACHE_ROOT CONDA_ROOT
export HF_HOME TORCH_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR TORCH_EXTENSIONS_DIR
export RUNS_ROOT ARTIFACT_ROOT ENV_REPORT_DIR EXPORT_DIR RENDER_DIR LOG_DIR

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Required directory not found: $1"
}

ensure_runtime_dirs() {
  mkdir -p \
    "${DATA_ROOT}" "${THIRD_PARTY_ROOT}" "${ENV_ROOT}" "${CACHE_ROOT}" \
    "${HF_HOME}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" \
    "${PIP_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
    "${RUNS_ROOT}" "${ARTIFACT_ROOT}" "${ENV_REPORT_DIR}" \
    "${EXPORT_DIR}" "${RENDER_DIR}" "${LOG_DIR}"
}

ensure_conda() {
  # The RunPod PyTorch image has no conda. Install Miniforge onto the network
  # volume so the CUDA toolchain survives pod replacement.
  if [[ ! -x "${CONDA_ROOT}/bin/conda" ]] && ! command -v conda >/dev/null 2>&1; then
    require_command curl
    local installer="${CACHE_ROOT}/Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh"
    local url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-$(uname -m).sh"

    log "Installing Miniforge ${MINIFORGE_VERSION} into ${CONDA_ROOT}"
    mkdir -p "${CACHE_ROOT}"
    [[ -f "${installer}" ]] || curl -fsSL "${url}" -o "${installer}"
    bash "${installer}" -b -p "${CONDA_ROOT}"
  fi

  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    export PATH="${CONDA_ROOT}/bin:${PATH}"
  fi

  require_command conda
  log "Using $(conda --version) at $(command -v conda)"
}

use_nerfstudio_env() {
  # gsplat compiles its CUDA kernels on first use and finds nvcc through
  # CUDA_HOME and PATH, which are only set when the environment is activated.
  # Calling its entry points directly leaves gsplat silently disabled.
  require_file "${NERFSTUDIO_CONDA_PREFIX}/bin/python"
  export CUDA_HOME="${NERFSTUDIO_CONDA_PREFIX}"
  export PATH="${NERFSTUDIO_CONDA_PREFIX}/bin:${PATH}"
  require_command nvcc

  # conda's CUDA packages keep the headers under targets/<arch>-linux/include
  # rather than in $CUDA_HOME/include, where torch's extension builder looks,
  # so the build fails on cuda_runtime_api.h without these search paths.
  local cuda_target="${NERFSTUDIO_CONDA_PREFIX}/targets/$(uname -m)-linux"
  require_file "${cuda_target}/include/cuda_runtime_api.h"
  export CPATH="${cuda_target}/include${CPATH:+:${CPATH}}"
  export LIBRARY_PATH="${cuda_target}/lib:${NERFSTUDIO_CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"

  # Nerfstudio 1.1.5 predates the torch 2.6 switch to weights_only=True and calls
  # torch.load without it, so it cannot reload its own checkpoints on torch 2.9.
  # Only checkpoints produced by this pipeline are ever loaded here.
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
}

ensure_open3d_libraries() {
  # ns-export imports open3d, which links against libEGL and libGL. The RunPod
  # PyTorch image ships neither, so the export dies on an ImportError.
  if ! ldconfig -p 2>/dev/null | grep -q "libEGL.so.1"; then
    log "Installing the OpenGL runtime libraries open3d needs"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 >/dev/null
  fi
  ldconfig -p 2>/dev/null | grep -q "libEGL.so.1" || die "libEGL.so.1 is still missing"
}

clone_pinned_repo() {
  local repository="$1"
  local commit="$2"
  local destination="$3"

  require_command git
  mkdir -p "$(dirname -- "${destination}")"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${repository}" "${destination}"
  fi

  local configured_remote
  configured_remote="$(git -C "${destination}" remote get-url origin)"
  [[ "${configured_remote%.git}" == "${repository%.git}" ]] || \
    die "Unexpected origin for ${destination}: ${configured_remote}"

  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
  [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]] || \
    die "Failed to check out ${commit} in ${destination}"
}

find_latest_splat_config() {
  if [[ -n "${SPLAT_CONFIG:-}" ]]; then
    require_file "${SPLAT_CONFIG}"
    printf '%s\n' "${SPLAT_CONFIG}"
    return
  fi

  require_dir "${SPLAT_OUTPUT_DIR}"
  local config
  config="$(find "${SPLAT_OUTPUT_DIR}" -type f -name config.yml -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "${config}" ]] || die "No Splatfacto config.yml found under ${SPLAT_OUTPUT_DIR}"
  printf '%s\n' "${config}"
}
