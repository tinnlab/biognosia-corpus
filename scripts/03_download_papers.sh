#!/usr/bin/env bash
# Download full texts for the papers listed in data/papers.tsv.
#
# Usage: scripts/03_download_papers.sh [--limit N] [--retry-failed]
# Environment: SEMANTIC_SCHOLAR_API_KEY (see .env.example), CONTACT_EMAIL (optional)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config pipeline.conf

python -m src.papers.download_papers \
    --papers-tsv "$PAPER_LIST" \
    --output-dir "$DATA_DIR/$RAW_PAPERS_DIR" \
    --pdf-dir "$DATA_DIR/$PAPER_PDF_DIR" \
    --status-file "$DATA_DIR/$PAPER_DOWNLOAD_STATUS_FILE" \
    --workers "$PAPER_DOWNLOAD_WORKERS" \
    --s2-interval "$SEMANTIC_SCHOLAR_REQUEST_INTERVAL" \
    --epmc-interval "$EUROPE_PMC_REQUEST_INTERVAL" \
    "$@"

echo "Texts: $DATA_DIR/$RAW_PAPERS_DIR"
echo "Status: $DATA_DIR/$PAPER_DOWNLOAD_STATUS_FILE"
