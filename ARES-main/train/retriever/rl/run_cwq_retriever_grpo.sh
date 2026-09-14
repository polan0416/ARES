#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARES_MAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARES_ROOT="$(cd "${ARES_MAIN_ROOT}/.." && pwd)"
VERL_DIR="${ARES_ROOT}/verl"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ARES_ROOT}/checkpoint}"

DEFAULT_CONFIG_NAME="cwq_retriever_grpo"
DEFAULT_CONFIG="${SCRIPT_DIR}/${DEFAULT_CONFIG_NAME}.yaml"

if [ "$#" -ge 1 ] && [[ "$1" != *"="* ]]; then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="${DEFAULT_CONFIG}"
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_PATH}"
fi
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

RESOLVED_CONFIG_DIR="${SCRIPT_DIR}/.cache/resolved_configs"
mkdir -p "${RESOLVED_CONFIG_DIR}"
RESOLVED_CONFIG="${RESOLVED_CONFIG_DIR}/${DEFAULT_CONFIG_NAME}.resolved.yaml"
python3 "${SCRIPT_DIR}/resolve_verl_grpo_config.py" "${CONFIG_PATH}" "${RESOLVED_CONFIG}"

TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${CHECKPOINT_ROOT}/cwq/retriever_grpo/qwen3-4b-grpo}"
export TRAINER_DEFAULT_LOCAL_DIR
LOG_DIR="${LOG_DIR:-${TRAINER_DEFAULT_LOCAL_DIR}/logs}"
mkdir -p "${LOG_DIR}"
GRPO_LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
export GRPO_LOG_FILE

export WANDB_PROJECT="${WANDB_PROJECT:-cwq_retriever_grpo}"
export WANDB_NAME="${WANDB_NAME:-qwen3_4b_grpo_cwq}"
export WANDB_DIR="${WANDB_DIR:-${TRAINER_DEFAULT_LOCAL_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${WANDB_NAME}}"

export PYTHONPATH="${RETRIEVER_ROOT}:${VERL_DIR}:${PYTHONPATH:-}"
export CWQ_REMOTE_ANSWER_URL="${CWQ_REMOTE_ANSWER_URL:-http://10.87.135.152:8002/answer}"

export RESOLVED_CONFIG
export VERL_DIR
export CONFIG_NAME="${DEFAULT_CONFIG_NAME}"

echo "Using resolved config: ${RESOLVED_CONFIG}"
conda run -n rl --no-capture-output bash -lc '
  set -o pipefail
  cd "${VERL_DIR}"
  echo "Logging to: ${GRPO_LOG_FILE}"
  python -m verl.trainer.main_ppo \
    --config-path "$(dirname "${RESOLVED_CONFIG}")" \
    --config-name "$(basename "${RESOLVED_CONFIG}" .yaml)" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}" \
    "$@" 2>&1 | tee -a "${GRPO_LOG_FILE}"
' _ "$@"
