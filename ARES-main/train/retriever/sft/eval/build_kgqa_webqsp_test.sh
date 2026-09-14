#!/usr/bin/env bash
# KGQA agent inference for WebQSP test split.
# Wrapper around build_kgqa_webqsp.sh with test defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-webqsp}"
export SPLITS="${SPLITS:-test}"
export N_ROLLOUTS="${N_ROLLOUTS:-1}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

exec bash "${SCRIPT_DIR}/build_kgqa_webqsp.sh" "$@"
