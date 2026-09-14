#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARES_MAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARES_ROOT="$(cd "${ARES_MAIN_ROOT}/.." && pwd)"
cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_vllm.py \
  -d cwq \
  --split test \
  --prompt_mode scored_200 \
  --tensor_parallel_size 4 \
  --num_workers 64 \
  --temperature 0.7 \
  -m "${ARES_ROOT}/checkpoint/cwq/kgqa_answer/llama3.1-8b-sft-merged" \
  --output_dir "${ARES_ROOT}/checkpoint/cwq/result/grponoex4" \
  -p "${ARES_MAIN_ROOT}/retrieve/stage2_runs/inference/cwqeval/retrieval_grpo4noex_prefix4_only.pth"
