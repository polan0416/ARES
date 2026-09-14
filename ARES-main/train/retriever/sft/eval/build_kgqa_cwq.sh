#!/usr/bin/env bash
# KGQA agent inference for CWQ train + val splits.
# Wrapper around build_kgqa_webqsp.sh with CWQ defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="${DATASET:-cwq}"
export SPLITS="${SPLITS:-train val}"

exec bash "${SCRIPT_DIR}/build_kgqa_webqsp.sh" "$@"
