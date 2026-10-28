#!/usr/bin/env bash
# Clean and normalise the paper texts.
#
# Usage: scripts/04_preprocess.sh [--force] [--retry-errors]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

LIMIT_ARGS=()
if [ "$LIMIT" -ne 0 ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

python -m src.preprocessing.preprocess_papers \
    --input "$DATA_DIR/$RAW_PAPERS_DIR" \
    --output "$DATA_DIR/$PREPROCESSED_DIR" \
    --error-output "$DATA_DIR/$PREPROCESSING_ERROR_DIR" \
    --stats-output "$DATA_DIR/$PREPROCESSING_STATS_DIR" \
    --workers "$WORKERS" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
    "$@"

echo "Preprocessed papers: $DATA_DIR/$PREPROCESSED_DIR"
