#!/usr/bin/env bash
# Link the matched entities to reference entities and insert them into Milvus and Redis.
#
# Usage: scripts/09_insert_entities.sh [--drop-existing] [--paper-ids ID1,ID2] [--insert-genes] ...

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

DROP_EXISTING=false
PAPER_IDS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --drop-existing) DROP_EXISTING=true ;;
        --workspace) WORKSPACE="$2"; shift ;;
        --paper-ids) PAPER_IDS="$2"; shift ;;
        --insert-genes) INSERT_GENES=true ;;
        --insert-disease) INSERT_DISEASE=true ;;
        --insert-chemical) INSERT_CHEMICAL=true ;;
        --insert-anatomy) INSERT_ANATOMY=true ;;
        --insert-cell) INSERT_CELL=true ;;
        --insert-taxonomy) INSERT_TAXONOMY=true ;;
        --insert-go) INSERT_GO=true ;;
        --insert-reactome) INSERT_REACTOME=true ;;
        --insert-kegg) INSERT_KEGG=true ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "Input: $DATA_DIR/$ENTITIES_NGRAM_DIR"
echo "Milvus: $MILVUS_HOST:$MILVUS_PORT"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo "Workspace: $WORKSPACE"

ARGS=(
    --input "$DATA_DIR/$ENTITIES_NGRAM_DIR"
    --milvus-host "$MILVUS_HOST"
    --milvus-port "$MILVUS_PORT"
    --redis-host "$REDIS_HOST"
    --redis-port "$REDIS_PORT"
    --workspace "$WORKSPACE"
    --similarity-threshold "$ENTITY_SIMILARITY_THRESHOLD"
    --batch-size "$INSERT_BATCH_SIZE"
    --papers-per-batch "$PAPERS_PER_BATCH"
    --redis-batch-size "$REDIS_BATCH_SIZE"
    --workers "$WORKERS"
    --insert-genes "$INSERT_GENES"
    --insert-disease "$INSERT_DISEASE"
    --insert-chemical "$INSERT_CHEMICAL"
    --insert-anatomy "$INSERT_ANATOMY"
    --insert-cell "$INSERT_CELL"
    --insert-taxonomy "$INSERT_TAXONOMY"
    --insert-go "$INSERT_GO"
    --insert-reactome "$INSERT_REACTOME"
    --insert-kegg "$INSERT_KEGG"
    --reference-batch-size "$REFERENCE_INSERT_BATCH_SIZE"
)
if [ -n "$ENTITY_STATS_FILE" ] && [ -f "$ENTITY_STATS_FILE" ]; then
    ARGS+=(--entity-stats "$ENTITY_STATS_FILE")
fi
if [ -n "$MAX_PCT_PAPER" ]; then ARGS+=(--max-pct-paper "$MAX_PCT_PAPER"); fi
if [ -n "$MAX_PCT_CHUNK" ]; then ARGS+=(--max-pct-chunk "$MAX_PCT_CHUNK"); fi
if [ "$LIMIT" -ne 0 ]; then ARGS+=(--limit "$LIMIT"); fi
if [ -n "$PAPER_IDS" ]; then ARGS+=(--paper-ids "$PAPER_IDS"); fi
if [ "$DROP_EXISTING" = true ]; then
    ARGS+=(--drop-existing)
    rm -f "$DATA_DIR/entities-inserted.csv"
fi

python -m src.insert.insert_entities "${ARGS[@]}"
