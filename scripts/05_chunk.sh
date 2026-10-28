#!/usr/bin/env bash
# Split the preprocessed papers into semantic chunks.
#
# Usage: scripts/05_chunk.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

echo "Input directory: $DATA_DIR/$PREPROCESSED_DIR"
echo "Output directory: $DATA_DIR/$CHUNKS_DIR"
echo "Chunk size: $CHUNK_SIZE tokens"
echo "Workers: $WORKERS"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

OPTIONAL_ARGS=()
if [ "$LIMIT" -ne 0 ]; then
    OPTIONAL_ARGS+=(--limit "$LIMIT")
fi

python -m src.chunking.chunk_papers \
    --input "$DATA_DIR/$PREPROCESSED_DIR" \
    --output "$DATA_DIR/$CHUNKS_DIR" \
    --error-output "$DATA_DIR/$CHUNK_ERROR_DIR" \
    --stats-output "$DATA_DIR/$CHUNK_STATS_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --workers "$WORKERS" \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}

echo "Chunks: $DATA_DIR/$CHUNKS_DIR"
echo "Chunk statistics: $DATA_DIR/$CHUNK_STATS_DIR/chunk_stats.csv"
