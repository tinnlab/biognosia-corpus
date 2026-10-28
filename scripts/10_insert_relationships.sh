#!/usr/bin/env bash
# Insert the relationships between matched entities into Milvus and Neo4j.
#
# Usage: scripts/10_insert_relationships.sh [--drop-existing] [--neo4j-only]
# Environment: NEO4J_PASSWORD (see .env.example)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

if [ -z "${NEO4J_PASSWORD:-}" ]; then
    echo "ERROR: NEO4J_PASSWORD is not set. Copy .env.example to .env and set it." >&2
    exit 1
fi
export NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD

DROP_EXISTING=false
NEO4J_ONLY=false

while [ $# -gt 0 ]; do
    case "$1" in
        --drop-existing) DROP_EXISTING=true ;;
        --neo4j-only) NEO4J_ONLY=true ;;
        --workspace) WORKSPACE="$2"; shift ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "Milvus: $MILVUS_HOST:$MILVUS_PORT"
echo "Neo4j: $NEO4J_URI"
echo "Workspace: $WORKSPACE"

ARGS=(
    --milvus-host "$MILVUS_HOST"
    --milvus-port "$MILVUS_PORT"
    --workspace "$WORKSPACE"
    --paper-batch-size "$RELATIONSHIP_PAPER_BATCH_SIZE"
    --relationship-batch-size "$RELATIONSHIP_UPSERT_BATCH"
    --workers "$WORKERS"
    --query-workers "$RELATIONSHIP_NUM_QUERY_WORKERS"
    --data-dir "$DATA_DIR"
    --entities-ngram-dir "$DATA_DIR/$ENTITIES_NGRAM_DIR"
    --similarity-threshold "$ENTITY_SIMILARITY_THRESHOLD"
    --limit "$LIMIT"
)
if [ -n "$ENTITY_STATS_FILE" ] && [ -f "$ENTITY_STATS_FILE" ]; then
    ARGS+=(--entity-stats "$ENTITY_STATS_FILE")
fi
if [ -n "$MAX_PCT_PAPER" ]; then ARGS+=(--max-pct-paper "$MAX_PCT_PAPER"); fi
if [ -n "$MAX_PCT_CHUNK" ]; then ARGS+=(--max-pct-chunk "$MAX_PCT_CHUNK"); fi
if [ "$NEO4J_ONLY" = true ]; then ARGS+=(--neo4j-only); fi
if [ "$DROP_EXISTING" = true ]; then
    ARGS+=(--drop-existing)
    rm -f "$DATA_DIR/relationships-inserted.csv"
fi

python -m src.insert.insert_relationships "${ARGS[@]}"
