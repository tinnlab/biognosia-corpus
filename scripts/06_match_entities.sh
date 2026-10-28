#!/usr/bin/env bash
# Match n-grams of each chunk against the entities_* collections in Milvus.
# The Genes and Gene_Synonyms collections are searched together.
#
# Usage: scripts/06_match_entities.sh [--paper-ids ID1,ID2] [--clear-cache]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

echo "Input directory: $DATA_DIR/$CHUNKS_DIR"
echo "Output directory: $DATA_DIR/$ENTITIES_NGRAM_DIR"
echo "Redis cache: $NGRAM_REDIS_HOST:$NGRAM_REDIS_PORT"
echo "Milvus: $MILVUS_HOST:$MILVUS_PORT"

OPTIONAL_ARGS=()
if [ "$LIMIT" -ne 0 ]; then
    OPTIONAL_ARGS+=(--limit "$LIMIT")
fi
if [ "$OFFSET" -ne 0 ]; then
    OPTIONAL_ARGS+=(--offset "$OFFSET")
fi

python -m src.entity_extraction.match_ngram_entities_parallel \
    --input "$DATA_DIR/$CHUNKS_DIR" \
    --output "$DATA_DIR/$ENTITIES_NGRAM_DIR" \
    --error-output "$DATA_DIR/$ENTITIES_NGRAM_ERROR_DIR" \
    --stats-output "$DATA_DIR/$ENTITIES_NGRAM_STATS_DIR" \
    --chunk-load-workers "$NGRAM_CHUNK_LOAD_WORKERS" \
    --embedding-workers "$NGRAM_EMBEDDING_WORKERS" \
    --query-workers-per-collection "$NGRAM_QUERY_WORKERS" \
    --paper-batch "$NGRAM_PAPER_BATCH" \
    --embedding-batch-size "$NGRAM_EMBEDDING_BATCH" \
    --k-values "$NGRAM_K_VALUES" \
    --min-similarity "$NGRAM_MIN_SIMILARITY" \
    --milvus-host "$MILVUS_HOST" \
    --milvus-port "$MILVUS_PORT" \
    --workspace "$WORKSPACE" \
    --redis-host "$NGRAM_REDIS_HOST" \
    --redis-port "$NGRAM_REDIS_PORT" \
    --redis-batch-size "$NGRAM_REDIS_BATCH_SIZE" \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"} \
    "$@"

echo "Matched entities: $DATA_DIR/$ENTITIES_NGRAM_DIR"
echo "Matching statistics: $DATA_DIR/$ENTITIES_NGRAM_STATS_DIR/matching_stats.csv"
