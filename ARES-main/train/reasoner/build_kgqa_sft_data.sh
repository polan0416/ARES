#!/usr/bin/env bash
# Build reasoner KGQA answer SFT data (no Question Explain) from kgqa_data pth.
# Output: ARES-main/train/reasoner/data/{dataset}/sft/{train.json,val.json,dataset_info.json}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REASONER_DATA_ROOT="${REASONER_DATA_ROOT:-${SCRIPT_DIR}/data}"
DATASETS="${DATASETS:-webqsp cwq}"
SPLITS="${SPLITS:-train val}"
TOP_K="${TOP_K:-100}"
KGQA_DATA_ROOT="${KGQA_DATA_ROOT:-${SCRIPT_DIR}/../retriever/sft/kgqa_data}"
CONDA_ENV="${CONDA_ENV:-rea3}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
SAMPLE_LIMITS_JSON="${SAMPLE_LIMITS_JSON:-}"
INJECT_GOLD_PATHS="${INJECT_GOLD_PATHS:-1}"

cd "${SCRIPT_DIR}"

KGQA_ARG=()
if [[ -n "${KGQA_DATA_ROOT}" ]]; then
  KGQA_ARG=(--kgqa_data_root "${KGQA_DATA_ROOT}")
fi

LIMIT_ARG=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  LIMIT_ARG=(--sample_limit "${SAMPLE_LIMIT}")
fi

LIMITS_JSON_ARG=()
if [[ -n "${SAMPLE_LIMITS_JSON}" ]]; then
  LIMITS_JSON_ARG=(--sample_limits_json "${SAMPLE_LIMITS_JSON}")
fi

GOLD_ARG=()
if [[ "${INJECT_GOLD_PATHS}" == "0" || "${INJECT_GOLD_PATHS}" == "false" ]]; then
  GOLD_ARG=(--no-inject_gold_paths)
fi

# shellcheck disable=SC2206
DATASET_ARR=(${DATASETS})
# shellcheck disable=SC2206
SPLIT_ARR=(${SPLITS})

for DATASET in "${DATASET_ARR[@]}"; do
  DATASET_OUT="${REASONER_DATA_ROOT}/${DATASET}/sft"
  echo "Building reasoner KGQA answer data for ${DATASET} -> ${DATASET_OUT}"
  conda run -n "${CONDA_ENV}" --no-capture-output python build_kgqa_sft_data.py \
    --datasets "${DATASET}" \
    --splits "${SPLIT_ARR[@]}" \
    --top_k "${TOP_K}" \
    --output_root "${DATASET_OUT}" \
    "${KGQA_ARG[@]}" \
    "${LIMIT_ARG[@]}" \
    "${LIMITS_JSON_ARG[@]}" \
    "${GOLD_ARG[@]}"
done

echo "Done."
