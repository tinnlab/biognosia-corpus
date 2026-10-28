#!/usr/bin/env bash
# Load the reference databases into Milvus (entities_* and relationships_* collections).
#
# Usage: scripts/01_init_reference_db.sh [loader ...]
# Loaders: genes gene2go ctd-diseases ctd-chemicals kegg reactome msigdb cell-ontology uberon taxonomy
# Without arguments all loaders run in this order.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config db_init.conf

ALL_LOADERS=(genes gene2go ctd-diseases ctd-chemicals kegg reactome msigdb cell-ontology uberon taxonomy)
LOADERS=("$@")
if [ ${#LOADERS[@]} -eq 0 ]; then
    LOADERS=("${ALL_LOADERS[@]}")
fi

flag_if_true() {
    # flag_if_true <value> <flag>: print the flag when the value is "true"
    if [ "$1" = "true" ]; then
        echo "$2"
    fi
}

run_loader() {
    local name="$1"
    echo "=================================================="
    echo "Loader: $name"
    echo "=================================================="
    case "$name" in
        genes)
            python src/db_init/insert_gene_entities.py
            ;;
        gene2go)
            python src/db_init/insert_gene2go.py --drop-existing
            ;;
        ctd-diseases)
            python src/db_init/insert_ctd_diseases.py \
                $(flag_if_true "$CTD_DISEASE_SKIP_GENE_DISEASE" --skip-gene-disease) \
                $(flag_if_true "$CTD_DISEASE_SKIP_KEGG_PATHWAY" --skip-kegg-pathway) \
                $(flag_if_true "$CTD_DISEASE_SKIP_REACTOME_PATHWAY" --skip-reactome-pathway)
            ;;
        ctd-chemicals)
            python src/db_init/insert_ctd_chemicals.py
            ;;
        kegg)
            python src/db_init/insert_kegg_pathways.py \
                --config "$PROJECT_ROOT/config/db_init.conf" \
                --organisms "$KEGG_ORGANISMS" \
                --batch-size "$KEGG_BATCH_SIZE" \
                --workers "$WORKERS" \
                --drop-existing \
                $(flag_if_true "$KEGG_SKIP_PATHWAYS" --skip-entities) \
                $(flag_if_true "$KEGG_SKIP_RELATIONSHIPS" --skip-relationships)
            ;;
        reactome)
            python src/db_init/insert_reactome.py \
                $(flag_if_true "$REACTOME_SKIP_PATHWAYS" --skip-pathways) \
                $(flag_if_true "$REACTOME_SKIP_RELATIONSHIPS" --skip-relationships)
            ;;
        msigdb)
            if [ "$MSIGDB_SPECIES" = "HS" ]; then
                MSIGDB_DB="$MSIGDB_DB_HUMAN"
            else
                MSIGDB_DB="$MSIGDB_DB_MOUSE"
            fi
            if [ ! -f "$MSIGDB_DB" ]; then
                echo "ERROR: MSigDB database not found: $MSIGDB_DB" >&2
                echo "Run scripts/00_download_resources.sh for download instructions." >&2
                exit 1
            fi
            python src/db_init/insert_msigdb.py \
                --msigdb-db "$MSIGDB_DB" \
                --species "$MSIGDB_SPECIES" \
                --batch-size "$MSIGDB_BATCH_SIZE" \
                ${MSIGDB_COLLECTIONS:+--collections "$MSIGDB_COLLECTIONS"} \
                $([ "$MSIGDB_LIMIT" != "0" ] && echo "--limit $MSIGDB_LIMIT") \
                $(flag_if_true "$MSIGDB_SKIP_GENE_SETS" --skip-gene-sets) \
                $(flag_if_true "$MSIGDB_SKIP_RELATIONSHIPS" --skip-relationships)
            ;;
        cell-ontology)
            python src/db_init/insert_cell_ontology.py --drop-existing
            ;;
        uberon)
            python src/db_init/insert_uberon.py --drop-existing
            ;;
        taxonomy)
            python src/db_init/insert_taxonomy.py
            ;;
        *)
            echo "ERROR: unknown loader: $name" >&2
            exit 1
            ;;
    esac
}

for loader in "${LOADERS[@]}"; do
    run_loader "$loader"
done

echo "Reference databases loaded."
