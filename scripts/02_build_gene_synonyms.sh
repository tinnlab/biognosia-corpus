#!/usr/bin/env bash
# Build the entities_Gene_Synonyms collection in Milvus.
#
# Usage: scripts/02_build_gene_synonyms.sh [--drop-existing] [--device cuda:0]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config db_init.conf

python src/db_init/build_gene_synonyms.py \
    --gene-info-file "$GENE_INFO_FILE" \
    --milvus-host "$MILVUS_HOST" \
    --milvus-port "$MILVUS_PORT" \
    "$@"
