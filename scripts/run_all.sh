#!/usr/bin/env bash
# Run the numbered pipeline steps in order.
#
# Usage: scripts/run_all.sh [--from N] [--to N]
#   --from N   first step to run (default 0)
#   --to N     last step to run (default 10)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FROM=0
TO=10
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift ;;
        --to) TO="$2"; shift ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

STEPS=(
    "00_download_resources.sh"
    "01_init_reference_db.sh"
    "02_build_gene_synonyms.sh"
    "03_download_papers.sh"
    "04_preprocess.sh"
    "05_chunk.sh"
    "06_match_entities.sh"
    "07_entity_statistics.sh"
    "08_insert_chunks.sh"
    "09_insert_entities.sh"
    "10_insert_relationships.sh"
)

for step in "${STEPS[@]}"; do
    number=$((10#${step%%_*}))
    if [ "$number" -lt "$FROM" ] || [ "$number" -gt "$TO" ]; then
        continue
    fi
    echo "=================================================="
    echo "Step $step"
    echo "=================================================="
    bash "$SCRIPT_DIR/$step"
done

echo "Pipeline finished."
