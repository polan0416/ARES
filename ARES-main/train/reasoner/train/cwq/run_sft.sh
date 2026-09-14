#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REASONER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARES_ROOT="$(cd "${REASONER_ROOT}/../../.." && pwd)"
LLAMA_FACTORY_DIR="${ARES_ROOT}/LLaMA-Factory"

DEFAULT_CONFIG="${REASONER_ROOT}/configs/llamafactory/cwq/llama31_8b_answer_sft.yaml"
if [ "$#" -ge 1 ]; then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="$DEFAULT_CONFIG"
fi

CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
RESOLVED_CONFIG_DIR="${REASONER_ROOT}/.cache/resolved_configs"
mkdir -p "${RESOLVED_CONFIG_DIR}"
RESOLVED_CONFIG="${RESOLVED_CONFIG_DIR}/llama31_8b_answer_sft.cwq.resolved.yaml"
python3 "${REASONER_ROOT}/scripts/resolve_llamafactory_config.py" "${CONFIG_PATH}" "${RESOLVED_CONFIG}"

LOG_DIR="${ARES_ROOT}/checkpoint/cwq/kgqa_answer/llama31_8b_sft/logs"
mkdir -p "${LOG_DIR}"
SFT_LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

export LLAMA_FACTORY_DIR
export RESOLVED_CONFIG
export SFT_LOG_FILE
export WANDB_PROJECT="${WANDB_PROJECT:-kgqa_answer_cwq}"
export WANDB_NAME="${WANDB_NAME:-llama31-8b-answer-sft}"
export WANDB_DIR="${WANDB_DIR:-${ARES_ROOT}/checkpoint/cwq/kgqa_answer/llama31_8b_sft/wandb}"

echo "Using resolved config: ${RESOLVED_CONFIG}"
conda run -n sft --no-capture-output bash -lc '
  set -o pipefail
  cd "${LLAMA_FACTORY_DIR}"
  echo "Logging to: ${SFT_LOG_FILE}"
  llamafactory-cli train "${RESOLVED_CONFIG}" "$@" 2>&1 | tee -a "${SFT_LOG_FILE}"
' _ "$@"
