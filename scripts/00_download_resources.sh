#!/usr/bin/env bash
# Download the curated resources and the models used by the pipeline.
#
# Usage: scripts/00_download_resources.sh [--force] [--skip-models]
#
# Files are written to $DATA_DIR/resources. The source URL, download date, size and
# SHA-256 of every file are recorded in $DATA_DIR/resources/VERSIONS.tsv.
# KEGG is not downloaded here. The KEGG loader fetches it from the KEGG REST API.
# MSigDB requires a registration and must be downloaded manually (see below).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_config db_init.conf

FORCE=false
SKIP_MODELS=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --skip-models) SKIP_MODELS=true ;;
        *) echo "ERROR: unknown option: $arg" >&2; exit 1 ;;
    esac
done

RESOURCES="$RESOURCE_DIR"
VERSIONS="$RESOURCES/VERSIONS.tsv"
mkdir -p "$RESOURCES"/{genes,ctd,reactome,msigdb,ontologies,taxonomy}
if [ ! -f "$VERSIONS" ]; then
    printf 'resource\tfile\tsource_url\tdownload_date\tbytes\tsha256\n' > "$VERSIONS"
fi

record_version() {
    # record_version <resource> <file> <url>
    local resource="$1" file="$2" url="$3"
    local relative="${file#"$RESOURCES"/}"
    local tmp
    tmp="$(mktemp)"
    awk -F'\t' -v f="$relative" '$2 != f' "$VERSIONS" > "$tmp"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$resource" "$relative" "$url" "$(date -u +%Y-%m-%d)" \
        "$(stat -c %s "$file")" "$(sha256sum "$file" | cut -d' ' -f1)" >> "$tmp"
    mv "$tmp" "$VERSIONS"
}

fetch() {
    # fetch <resource> <url> <destination>
    local resource="$1" url="$2" dest="$3"
    if [ -f "$dest" ] && [ "$FORCE" = false ]; then
        echo "exists: ${dest#"$RESOURCES"/}"
        return
    fi
    echo "downloading: $url"
    curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
    record_version "$resource" "$dest" "$url"
}

# NCBI Gene
fetch "NCBI Gene" "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz" "$RESOURCES/genes/gene_info.gz"
fetch "NCBI Gene" "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz" "$RESOURCES/genes/gene2go.gz"

# Comparative Toxicogenomics Database
for name in CTD_diseases CTD_chemicals CTD_curated_genes_diseases CTD_diseases_pathways \
            CTD_chem_gene_ixns CTD_curated_chemicals_diseases CTD_chem_pathways_enriched CTD_chem_go_enriched; do
    fetch "CTD" "https://ctdbase.org/reports/$name.csv.gz" "$RESOURCES/ctd/$name.csv.gz"
done

# Reactome
fetch "Reactome" "https://reactome.org/download/current/ReactomePathways.txt" "$RESOURCES/reactome/ReactomePathways.txt"
if [ ! -f "$RESOURCES/reactome/ReactomePathways.gmt" ] || [ "$FORCE" = true ]; then
    fetch "Reactome" "https://reactome.org/download/current/ReactomePathways.gmt.zip" "$RESOURCES/reactome/ReactomePathways.gmt.zip"
    unzip -o -q "$RESOURCES/reactome/ReactomePathways.gmt.zip" -d "$RESOURCES/reactome"
    record_version "Reactome" "$RESOURCES/reactome/ReactomePathways.gmt" "https://reactome.org/download/current/ReactomePathways.gmt.zip"
else
    echo "exists: reactome/ReactomePathways.gmt"
fi

# Cell Ontology and Uberon
fetch "Cell Ontology" "http://purl.obolibrary.org/obo/cl.owl" "$RESOURCES/ontologies/cl.owl"
fetch "Uberon" "http://purl.obolibrary.org/obo/uberon.owl" "$RESOURCES/ontologies/uberon.owl"

# NCBI Taxonomy
if [ ! -f "$RESOURCES/taxonomy/nodes.dmp" ] || [ "$FORCE" = true ]; then
    fetch "NCBI Taxonomy" "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz" "$RESOURCES/taxonomy/taxdump.tar.gz"
    tar -xzf "$RESOURCES/taxonomy/taxdump.tar.gz" -C "$RESOURCES/taxonomy"
    for dump in nodes.dmp names.dmp merged.dmp; do
        record_version "NCBI Taxonomy" "$RESOURCES/taxonomy/$dump" "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
    done
else
    echo "exists: taxonomy/nodes.dmp"
fi

# MSigDB (registration required)
MSIGDB_FILE="$MSIGDB_DB_HUMAN"
if [ -f "$MSIGDB_FILE" ]; then
    echo "exists: ${MSIGDB_FILE#"$RESOURCES"/}"
else
    cat <<MSG

MSigDB requires a free registration and cannot be downloaded automatically.
1. Register at https://www.gsea-msigdb.org/gsea/register.jsp
2. Download the SQLite release for human from
   https://www.gsea-msigdb.org/gsea/downloads.jsp (file msigdb_v2025.1.Hs.db.zip, SQLite release)
3. Unzip it and place msigdb_v2025.1.Hs.db at
   $MSIGDB_FILE
4. Run this script again to record its checksum in VERSIONS.tsv.

MSG
fi
if [ -f "$MSIGDB_FILE" ] && ! grep -q "msigdb/$(basename "$MSIGDB_FILE")" "$VERSIONS"; then
    record_version "MSigDB" "$MSIGDB_FILE" "https://www.gsea-msigdb.org/gsea/downloads.jsp (manual download)"
fi

# NLTK data used for n-gram generation
python - <<PY
import nltk
target = "$NLTK_DATA"
for package in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords"):
    nltk.download(package, download_dir=target, quiet=True)
print("NLTK data:", target)
PY

# Models
if [ "$SKIP_MODELS" = false ]; then
    python - <<PY
from huggingface_hub import snapshot_download
for repo in ("BAAI/bge-m3", "ncbi/MedCPT-Query-Encoder", "minishlab/potion-base-32M"):
    path = snapshot_download(repo)
    print("model:", repo, path)
PY
fi

echo "Resources written to $RESOURCES"
echo "Versions recorded in $VERSIONS"
