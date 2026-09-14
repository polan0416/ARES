#!/usr/bin/env bash
# KGQA agent inference for train + val splits.
# Outputs retrieval_result_{split}.pth under kgqa_data/{dataset}/ (JSON intermediates are removed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARES_ROOT="$(cd "${SFT_DIR}/../../../../" && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ARES_ROOT}/checkpoint}"
EVAL_PY="${SFT_DIR}/eval_kgqa.py"
DATASET="${DATASET:-webqsp}"
SPLITS="${SPLITS:-train val}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ "${DATASET}" == "cwq" ]]; then
    MODEL_PATH="${CHECKPOINT_ROOT}/cwq/retriever_sft/qwen3-4b-sft_cwq-merged"
  else
    MODEL_PATH="${CHECKPOINT_ROOT}/webqsp/retriever_sft/qwen3-4b-sft_webqsp-merged"
  fi
fi
MAX_ACTIVE_BATCH_SIZE="${MAX_ACTIVE_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-64}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
GPU_UTIL="${GPU_UTIL:-0.95}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
MAX_STEPS="${MAX_STEPS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
N_ROLLOUTS="${N_ROLLOUTS:-4}"
TOP_P="${TOP_P:-0.95}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-10}"
EXPORT_TOP_K="${EXPORT_TOP_K:-100}"
RESUME="${RESUME:-1}"
CONDA_ENV="${CONDA_ENV:-rea3}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "error: MODEL_PATH is required" >&2
  exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "error: inference script not found: ${EVAL_PY}" >&2
  exit 1
fi

cd "${SFT_DIR}"

MS_ARG=()
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  MS_ARG=(--max_samples "${MAX_EVAL_SAMPLES}")
fi

RESUME_ARG=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARG=(--resume)
fi

PREFIX_CACHE_ARG=()
if [[ "${ENABLE_PREFIX_CACHING}" == "0" || "${ENABLE_PREFIX_CACHING}" == "false" ]]; then
  PREFIX_CACHE_ARG=(--disable_prefix_caching)
fi

TEMP_ARG=()
if [[ -n "${ROLLOUT_TEMPERATURE}" ]]; then
  TEMP_ARG=(--rollout_temperature "${ROLLOUT_TEMPERATURE}")
fi

if [[ -n "${GPU_IDS}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
else
  GPU_LIST=()
  for ((g=0; g<NUM_GPUS; g++)); do
    GPU_LIST+=("${g}")
  done
fi

if (( NUM_GPUS > ${#GPU_LIST[@]} )); then
  echo "error: NUM_GPUS=${NUM_GPUS} exceeds available GPUs (${#GPU_LIST[@]} in GPU_IDS='${GPU_IDS}')" >&2
  exit 1
fi

sub100_json_for_split() {
  local split="$1"
  if [[ "${split}" == "test" ]]; then
    echo "${SFT_DIR}/eval_data/${DATASET}/sub100.json"
  else
    echo "${SFT_DIR}/eval_data/${DATASET}/sub100${split}.json"
  fi
}

out_json_for_split() {
  local split="$1"
  if [[ "${split}" == "test" ]]; then
    echo "${SFT_DIR}/kgqa_data/${DATASET}/inference_result.json"
  else
    echo "${SFT_DIR}/kgqa_data/${DATASET}/inference_result_${split}.json"
  fi
}

out_pth_for_split() {
  local split="$1"
  if [[ "${split}" == "test" ]]; then
    echo "${SFT_DIR}/kgqa_data/${DATASET}/retrieval_result.pth"
  else
    echo "${SFT_DIR}/kgqa_data/${DATASET}/retrieval_result_${split}.pth"
  fi
}

run_split() {
  local split="$1"
  local sub100_json out_json out_pth base
  sub100_json="$(sub100_json_for_split "${split}")"
  out_json="$(out_json_for_split "${split}")"
  out_pth="$(out_pth_for_split "${split}")"

  if [[ ! -f "${sub100_json}" ]]; then
    echo "error: missing input for split=${split}: ${sub100_json}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${out_json}")"
  base="${out_json%.json}"
  local shards=()
  local pids=()

  echo "========== [${DATASET}/${split}] inference =========="
  echo "  input : ${sub100_json}"
  echo "  pth   : ${out_pth}"

  for ((i=0; i<NUM_GPUS; i++)); do
    local shard_out="${base}_shard${i}.json"
    shards+=("${shard_out}")
    local gpu_id="${GPU_LIST[$i]}"
    echo "Starting shard ${i}/${NUM_GPUS} on GPU ${gpu_id} -> ${shard_out}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" conda run -n "${CONDA_ENV}" --no-capture-output python "${EVAL_PY}" \
      --dataset "${DATASET}" \
      --subgraphs_json "${sub100_json}" \
      --model_path "${MODEL_PATH}" \
      --out_json "${shard_out}" \
      --n_rollouts "${N_ROLLOUTS}" \
      --max_active_batch_size "${MAX_ACTIVE_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      "${PREFIX_CACHE_ARG[@]}" \
      --max_steps "${MAX_STEPS}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --top_p "${TOP_P}" \
      --gpu_memory_utilization "${GPU_UTIL}" \
      --tensor_parallel_size 1 \
      --num_shards "${NUM_GPUS}" \
      --shard_index "${i}" \
      "${TEMP_ARG[@]}" \
      "${MS_ARG[@]}" \
      "${RESUME_ARG[@]}" &
    pids+=($!)
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "error: one or more shard jobs failed for split=${split}" >&2
    return 1
  fi

  echo "Merging [${split}] -> ${out_json}"
  conda run -n "${CONDA_ENV}" --no-capture-output python "${EVAL_PY}" \
    --merge_shards "${shards[@]}" \
    --out_json "${out_json}"

  echo "Exporting [${split}] -> ${out_pth}"
  conda run -n "${CONDA_ENV}" --no-capture-output python "${EVAL_PY}" \
    --dataset "${DATASET}" \
    --subgraphs_json "${sub100_json}" \
    --export_retrieval_pth_from "${out_json}" \
    --export_top_k "${EXPORT_TOP_K}" \
    --out_retrieval_pth "${out_pth}"

  echo "Removing intermediate JSON for [${split}]"
  rm -f "${out_json}" "${shards[@]}"
}

# shellcheck disable=SC2206
SPLIT_ARR=(${SPLITS})
for split in "${SPLIT_ARR[@]}"; do
  run_split "${split}"
done

echo "All splits done: ${DATASET} [${SPLITS}]"
