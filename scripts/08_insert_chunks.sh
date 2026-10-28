#!/usr/bin/env bash
# Insert the chunks into Milvus, Redis and Elasticsearch.
#
# Usage: scripts/08_insert_chunks.sh [--drop-existing] [--clear-redis] [--no-elasticsearch] [--force-reinsert]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DROP_EXISTING=false
CLEAR_REDIS=false
ENABLE_ELASTICSEARCH="$ELASTICSEARCH_ENABLED"
ES_ONLY_MODE=false
SKIP_EXISTING=true

while [ $# -gt 0 ]; do
    case "$1" in
        --drop-existing) DROP_EXISTING=true; SKIP_EXISTING=false ;;
        --clear-redis) CLEAR_REDIS=true ;;
        --no-elasticsearch) ENABLE_ELASTICSEARCH=false ;;
        --es-only-mode) ES_ONLY_MODE=true ;;
        --force-reinsert) SKIP_EXISTING=false ;;
        --workspace) WORKSPACE="$2"; shift ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "Chunks directory: $DATA_DIR/$CHUNKS_DIR"
echo "Milvus: $MILVUS_HOST:$MILVUS_PORT"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "Workspace: $WORKSPACE"

ARGS=(
    --input "$DATA_DIR/$CHUNKS_DIR"
    --chunk-stats "$DATA_DIR/$CHUNK_STATS_DIR/chunk_stats.csv"
    --milvus-host "$MILVUS_HOST"
    --milvus-port "$MILVUS_PORT"
    --redis-host "$REDIS_HOST"
    --redis-port "$REDIS_PORT"
    --workspace "$WORKSPACE"
    --batch-size "$INSERT_BATCH_SIZE"
)
if [ "$ENABLE_ELASTICSEARCH" = true ]; then
    ARGS+=(--elasticsearch-host "$ELASTICSEARCH_HOST" --elasticsearch-port "$ELASTICSEARCH_PORT")
fi
if [ "$ES_ONLY_MODE" = true ]; then ARGS+=(--es-only-mode); fi
if [ "$SKIP_EXISTING" = true ]; then ARGS+=(--skip-existing); fi
if [ "$LIMIT" -ne 0 ]; then ARGS+=(--limit "$LIMIT"); fi
if [ "$OFFSET" -ne 0 ]; then ARGS+=(--offset "$OFFSET"); fi
if [ "$DROP_EXISTING" = true ]; then
    ARGS+=(--drop-existing)
    rm -f "$DATA_DIR/chunks-inserted.csv"
fi
if [ "$CLEAR_REDIS" = true ]; then ARGS+=(--clear-redis); fi

python -m src.insert.insert_chunks "${ARGS[@]}"
