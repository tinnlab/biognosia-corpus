#!/usr/bin/env bash
# Compute per-entity occurrence statistics from the matching output.
#
# Usage: scripts/07_entity_statistics.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

python -m src.entity_extraction.analyze_entity_statistics \
    --input "$DATA_DIR/$ENTITIES_NGRAM_DIR" \
    --output "$DATA_DIR/$ENTITIES_NGRAM_STATS_DIR/entity_statistics.tsv" \
    --min-similarity "$NGRAM_MIN_SIMILARITY" \
    --case-insensitive

echo "Statistics: $DATA_DIR/$ENTITIES_NGRAM_STATS_DIR/entity_statistics_GLOBAL.tsv"
