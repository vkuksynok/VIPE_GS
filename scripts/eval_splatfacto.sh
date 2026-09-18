#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

ensure_runtime_dirs
use_nerfstudio_env

NS_EVAL="${NERFSTUDIO_CONDA_PREFIX}/bin/ns-eval"
[[ -x "${NS_EVAL}" ]] || die "ns-eval not found: run scripts/setup_splatfacto.sh first"

CONFIG="${1:-$(find_latest_splat_config)}"
require_file "${CONFIG}"

METRICS="${ENV_REPORT_DIR}/splatfacto-metrics-${VIPE_SEQUENCE}.json"
RENDER_OUTPUT="${RENDER_DIR}/eval-${VIPE_SEQUENCE}"
LOG_FILE="${LOG_DIR}/splatfacto-eval-${VIPE_SEQUENCE}.log"

# The eval split is the frames held out of Gaussian optimization. VIPE itself
# saw the whole sequence, so this measures rendering, not independent geometry.
log "Evaluating ${CONFIG} on the held-out frames"
"${NS_EVAL}" \
  --load-config "${CONFIG}" \
  --output-path "${METRICS}" \
  --render-output-path "${RENDER_OUTPUT}" \
  2>&1 | tee "${LOG_FILE}"

require_file "${METRICS}"
python3 -c "
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    document = json.load(handle)
results = document.get('results', {})
for key in ('psnr', 'ssim', 'lpips', 'num_rays_per_sec', 'fps'):
    if key in results:
        print(f'{key}: {results[key]}')
" "${METRICS}"

log "Evaluation complete: ${METRICS}"
