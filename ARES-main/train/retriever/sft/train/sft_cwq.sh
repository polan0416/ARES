#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARES_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
LLAMA_FACTORY_DIR="${ARES_ROOT}/LLaMA-Factory"

DEFAULT_CONFIG="${LLAMA_FACTORY_DIR}/examples/train_lora/qwen3_4b_agent_cwq.yaml"
if [ "$#" -ge 1 ]; then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="$DEFAULT_CONFIG"
fi

LOG_DIR="${ARES_ROOT}/checkpoint/cwq/retriever_sft/qwen3-4b-sft_cwq/logs"
mkdir -p "${LOG_DIR}"
SFT_LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
export SFT_LOG_FILE

export WANDB_PROJECT="${WANDB_PROJECT:-retriever_sft}"
export WANDB_NAME="${WANDB_NAME:-qwen3-4b-sft_cwq}"
export WANDB_DIR="${WANDB_DIR:-${ARES_ROOT}/checkpoint/cwq/retriever_sft/qwen3-4b-sft_cwq/wandb}"

conda run -n sft bash -lc '
  set -o pipefail
  cd "'"${LLAMA_FACTORY_DIR}"'"
  echo "Logging to: ${SFT_LOG_FILE}"
  llamafactory-cli train "$1" "${@:2}" 2>&1 | tee -a "${SFT_LOG_FILE}"
' _ "${CONFIG_PATH}" "$@"
