#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARES_MAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARES_ROOT="$(cd "${ARES_MAIN_ROOT}/.." && pwd)"
VERL_DIR="${ARES_ROOT}/verl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="${ARES_MAIN_ROOT}:${VERL_DIR}:${PYTHONPATH:-}"

MODEL_PATH="${CWQ_REMOTE_ANSWER_MODEL_NAME:-${ARES_ROOT}/checkpoint/cwq/kgqa_answer/llama3.1-8b-sft-merged}"
TOKENIZER_PATH="${CWQ_REMOTE_ANSWER_BASE_MODEL:-${HOME}/.cache/huggingface/model/Meta-Llama-3.1-8B-Instruct}"

ANSWER_HOST="${CWQ_REMOTE_ANSWER_HOST:-0.0.0.0}"
ANSWER_PORT="${CWQ_REMOTE_ANSWER_PORT:-8002}"
VLLM_HOST="${CWQ_VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${CWQ_VLLM_PORT:-8003}"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NEW_TOKENS="${CWQ_REMOTE_ANSWER_MAX_NEW_TOKENS:-256}"

VLLM_LOG="${CWQ_VLLM_LOG:-/tmp/cwq_vllm_serve.log}"
VLLM_PID_FILE="${CWQ_VLLM_PID_FILE:-/tmp/cwq_vllm_serve.pid}"

cleanup() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${VLLM_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping CWQ vLLM serve (pid=${pid})..."
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${VLLM_PID_FILE}"
  fi
}
trap cleanup EXIT INT TERM

echo "Starting CWQ vLLM serve on ${VLLM_HOST}:${VLLM_PORT} (model=${MODEL_PATH})"
echo "  gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  max_num_seqs=${VLLM_MAX_NUM_SEQS} max_model_len=${VLLM_MAX_MODEL_LEN}"

conda run -n rl vllm serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --enable-prefix-caching \
  --disable-log-stats \
  >"${VLLM_LOG}" 2>&1 &

echo $! >"${VLLM_PID_FILE}"

echo "Waiting for CWQ vLLM API at ${VLLM_API_BASE} ..."
for _ in $(seq 1 120); do
  if curl -sf "${VLLM_API_BASE}/models" >/dev/null 2>&1; then
    echo "CWQ vLLM is ready."
    break
  fi
  if ! kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
    echo "CWQ process exited early. Last log lines:"
    tail -n 40 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 2
done

if ! curl -sf "${VLLM_API_BASE}/models" >/dev/null 2>&1; then
  echo "Timed out waiting for CWQ vLLM. See ${VLLM_LOG}"
  exit 1
fi

echo "Starting CWQ /answer FastAPI on ${ANSWER_HOST}:${ANSWER_PORT} (backend=vllm-api -> ${VLLM_API_BASE})"

exec conda run -n rl python "${SCRIPT_DIR}/remote_answer_server.py" \
  --backend vllm \
  --host "${ANSWER_HOST}" \
  --port "${ANSWER_PORT}" \
  --base_model_path "${TOKENIZER_PATH}" \
  --model_name "${MODEL_PATH}" \
  --vllm-api-base "${VLLM_API_BASE}" \
  --vllm-served-model-name "${MODEL_PATH}" \
  --prompt_mode "${CWQ_REMOTE_ANSWER_PROMPT_MODE:-scored_200}" \
  --llm_mode "${CWQ_REMOTE_ANSWER_LLM_MODE:-sys_icl_dc}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"
